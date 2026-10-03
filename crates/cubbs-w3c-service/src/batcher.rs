use std::{
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use serde::Serialize;
use tokio::{
    sync::{mpsc, oneshot, Semaphore},
    time::Instant,
};

type BatchFuture<R> = Pin<Box<dyn Future<Output = Vec<Result<R, String>>> + Send>>;
type BatchProcessor<T, R> = dyn Fn(Vec<T>) -> BatchFuture<R> + Send + Sync;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BatcherConfig {
    pub interval_ms: u64,
    pub max_batch_size: usize,
    pub queue_capacity: usize,
    pub pipelines: usize,
}

impl BatcherConfig {
    pub fn from_env() -> Self {
        Self {
            interval_ms: env_u64("CUBBS_W3C_BATCH_INTERVAL_MS", 5),
            max_batch_size: env_usize("CUBBS_W3C_MAX_BATCH_SIZE", 128).max(1),
            queue_capacity: env_usize("CUBBS_W3C_QUEUE_CAPACITY", 32_768).max(1),
            pipelines: env_usize("CUBBS_W3C_PIPELINES", 20).max(1),
        }
    }
    pub fn from_env_for(
        operation: &str,
        default_max_batch_size: usize,
        default_pipelines: usize,
    ) -> Self {
        let operation = operation.to_ascii_uppercase().replace('-', "_");
        let interval_name = format!("CUBBS_W3C_{operation}_BATCH_INTERVAL_MS");
        let max_batch_name = format!("CUBBS_W3C_{operation}_MAX_BATCH_SIZE");
        let pipelines_name = format!("CUBBS_W3C_{operation}_PIPELINES");
        Self {
            interval_ms: env_u64_optional(&interval_name)
                .or_else(|| env_u64_optional("CUBBS_W3C_BATCH_INTERVAL_MS"))
                .unwrap_or(5),
            max_batch_size: env_usize_optional(&max_batch_name)
                .or_else(|| env_usize_optional("CUBBS_W3C_MAX_BATCH_SIZE"))
                .unwrap_or(default_max_batch_size)
                .max(1),
            queue_capacity: env_usize("CUBBS_W3C_QUEUE_CAPACITY", 32_768).max(1),
            pipelines: env_usize_optional(&pipelines_name)
                .or_else(|| env_usize_optional("CUBBS_W3C_PIPELINES"))
                .unwrap_or(default_pipelines)
                .max(1),
        }
    }
}

#[derive(Debug)]
struct BatcherMetrics {
    queued: AtomicUsize,
    dispatches: AtomicU64,
    dispatched_items: AtomicU64,
    max_observed_batch: AtomicUsize,
    last_batch_size: AtomicUsize,
    total_queue_wait_micros: AtomicU64,
    batch_size_counts: Vec<AtomicU64>,
}

impl BatcherMetrics {
    fn new(max_batch_size: usize) -> Self {
        Self {
            queued: AtomicUsize::new(0),
            dispatches: AtomicU64::new(0),
            dispatched_items: AtomicU64::new(0),
            max_observed_batch: AtomicUsize::new(0),
            last_batch_size: AtomicUsize::new(0),
            total_queue_wait_micros: AtomicU64::new(0),
            batch_size_counts: (0..=max_batch_size).map(|_| AtomicU64::new(0)).collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BatcherSnapshot {
    pub queued: usize,
    pub dispatches: u64,
    pub dispatched_items: u64,
    pub max_observed_batch: usize,
    pub last_batch_size: usize,
    pub total_queue_wait_ms: f64,
    pub average_queue_wait_ms: f64,
    pub batch_size_counts: Vec<u64>,
}

struct QueueItem<T, R> {
    input: T,
    enqueued_at: Instant,
    response: oneshot::Sender<Result<R, String>>,
}

pub struct BatchReceiver<R>(oneshot::Receiver<Result<R, String>>);

impl<R> BatchReceiver<R> {
    pub async fn receive(self) -> Result<R, String> {
        self.0
            .await
            .map_err(|_| "W3C batch worker stopped before replying".to_owned())?
    }
}

#[derive(Clone)]
pub struct BatchQueue<T, R> {
    sender: mpsc::Sender<QueueItem<T, R>>,
    metrics: Arc<BatcherMetrics>,
    config: BatcherConfig,
}

impl<T, R> BatchQueue<T, R>
where
    T: Send + 'static,
    R: Send + 'static,
{
    pub fn spawn<F, Fut>(config: BatcherConfig, processor: F) -> Self
    where
        F: Fn(Vec<T>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Vec<Result<R, String>>> + Send + 'static,
    {
        Self::spawn_with_slots(
            config,
            Arc::new(Semaphore::new(config.pipelines)),
            processor,
        )
    }

    pub fn spawn_with_slots<F, Fut>(
        config: BatcherConfig,
        pipeline_slots: Arc<Semaphore>,
        processor: F,
    ) -> Self
    where
        F: Fn(Vec<T>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Vec<Result<R, String>>> + Send + 'static,
    {
        let (sender, receiver) = mpsc::channel(config.queue_capacity);
        let metrics = Arc::new(BatcherMetrics::new(config.max_batch_size));
        let processor: Arc<BatchProcessor<T, R>> =
            Arc::new(move |items| Box::pin(processor(items)));
        let queue_slots = Arc::new(Semaphore::new(config.pipelines));
        tokio::spawn(run_worker(
            receiver,
            config,
            metrics.clone(),
            processor,
            pipeline_slots,
            queue_slots,
        ));
        Self {
            sender,
            metrics,
            config,
        }
    }

    pub async fn enqueue(&self, input: T) -> Result<BatchReceiver<R>, String> {
        let (response, receiver) = oneshot::channel();
        self.metrics.queued.fetch_add(1, Ordering::Relaxed);
        if self
            .sender
            .send(QueueItem {
                input,
                enqueued_at: Instant::now(),
                response,
            })
            .await
            .is_err()
        {
            self.metrics.queued.fetch_sub(1, Ordering::Relaxed);
            return Err("W3C batch queue is closed".to_owned());
        }
        Ok(BatchReceiver(receiver))
    }

    pub async fn submit(&self, input: T) -> Result<R, String> {
        self.enqueue(input).await?.receive().await
    }

    pub fn config(&self) -> BatcherConfig {
        self.config
    }

    pub fn snapshot(&self) -> BatcherSnapshot {
        let dispatched_items = self.metrics.dispatched_items.load(Ordering::Relaxed);
        let wait_micros = self.metrics.total_queue_wait_micros.load(Ordering::Relaxed);
        BatcherSnapshot {
            queued: self.metrics.queued.load(Ordering::Relaxed),
            dispatches: self.metrics.dispatches.load(Ordering::Relaxed),
            dispatched_items,
            max_observed_batch: self.metrics.max_observed_batch.load(Ordering::Relaxed),
            last_batch_size: self.metrics.last_batch_size.load(Ordering::Relaxed),
            total_queue_wait_ms: wait_micros as f64 / 1000.0,
            average_queue_wait_ms: if dispatched_items == 0 {
                0.0
            } else {
                wait_micros as f64 / dispatched_items as f64 / 1000.0
            },
            batch_size_counts: self
                .metrics
                .batch_size_counts
                .iter()
                .map(|count| count.load(Ordering::Relaxed))
                .collect(),
        }
    }
}

async fn run_worker<T, R>(
    mut receiver: mpsc::Receiver<QueueItem<T, R>>,
    config: BatcherConfig,
    metrics: Arc<BatcherMetrics>,
    processor: Arc<BatchProcessor<T, R>>,
    pipeline_slots: Arc<Semaphore>,
    queue_slots: Arc<Semaphore>,
) where
    T: Send + 'static,
    R: Send + 'static,
{
    let interval = Duration::from_millis(config.interval_ms);

    while let Some(first) = receiver.recv().await {
        let queue_slot = match queue_slots.clone().acquire_owned().await {
            Ok(slot) => slot,
            Err(_) => break,
        };
        let pipeline_slot = match pipeline_slots.clone().acquire_owned().await {
            Ok(slot) => slot,
            Err(_) => break,
        };
        let mut batch = Vec::with_capacity(config.max_batch_size);
        batch.push(first);
        drain_ready(&mut receiver, &mut batch, config.max_batch_size);

        // The interval is a coalescing window for a partial batch, not a
        // throughput throttle. Under sustained load, a complete batch is
        // dispatched as soon as a pipeline slot is available.
        if batch.len() < config.max_batch_size {
            let deadline = Instant::now() + interval;
            while batch.len() < config.max_batch_size && deadline > Instant::now() {
                match tokio::time::timeout_at(deadline, receiver.recv()).await {
                    Ok(Some(item)) => {
                        batch.push(item);
                        drain_ready(&mut receiver, &mut batch, config.max_batch_size);
                    }
                    Ok(None) | Err(_) => break,
                }
            }
        }

        let dispatch_time = Instant::now();
        let batch_size = batch.len();
        metrics.queued.fetch_sub(batch_size, Ordering::Relaxed);
        metrics.dispatches.fetch_add(1, Ordering::Relaxed);
        metrics
            .dispatched_items
            .fetch_add(batch_size as u64, Ordering::Relaxed);
        metrics.last_batch_size.store(batch_size, Ordering::Relaxed);
        metrics.batch_size_counts[batch_size].fetch_add(1, Ordering::Relaxed);
        metrics
            .max_observed_batch
            .fetch_max(batch_size, Ordering::Relaxed);
        let queue_wait_micros = batch
            .iter()
            .map(|item| {
                dispatch_time
                    .saturating_duration_since(item.enqueued_at)
                    .as_micros()
                    .min(u64::MAX as u128) as u64
            })
            .sum::<u64>();
        metrics
            .total_queue_wait_micros
            .fetch_add(queue_wait_micros, Ordering::Relaxed);

        let mut responses = Vec::with_capacity(batch_size);
        let mut inputs = Vec::with_capacity(batch_size);
        for item in batch {
            inputs.push(item.input);
            responses.push(item.response);
        }
        let processor = Arc::clone(&processor);
        tokio::spawn(async move {
            let mut outputs = processor(inputs).await.into_iter();
            for response in responses {
                let result = outputs.next().unwrap_or_else(|| {
                    Err("W3C batch processor returned fewer results than inputs".to_owned())
                });
                let _ = response.send(result);
            }
            drop(pipeline_slot);
            drop(queue_slot);
        });
    }

    fn drain_ready<T, R>(
        receiver: &mut mpsc::Receiver<QueueItem<T, R>>,
        batch: &mut Vec<QueueItem<T, R>>,
        max_batch_size: usize,
    ) {
        while batch.len() < max_batch_size {
            match receiver.try_recv() {
                Ok(item) => batch.push(item),
                Err(_) => break,
            }
        }
    }
}

fn env_u64_optional(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.parse().ok()
}

fn env_usize_optional(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn coalesces_requests_while_respecting_dispatch_interval() {
        let queue = BatchQueue::spawn(
            BatcherConfig {
                interval_ms: 20,
                max_batch_size: 16,
                queue_capacity: 32,
                pipelines: 1,
            },
            |items: Vec<usize>| async move { items.into_iter().map(Ok).collect() },
        );

        assert_eq!(queue.submit(1).await.unwrap(), 1);
        let receivers = futures_for(&queue, 2..=6).await;
        for (expected, receiver) in (2..=6).zip(receivers) {
            assert_eq!(receiver.receive().await.unwrap(), expected);
        }
        let snapshot = queue.snapshot();
        assert_eq!(snapshot.dispatches, 2);
        assert_eq!(snapshot.last_batch_size, 5);
        assert_eq!(snapshot.max_observed_batch, 5);
    }

    #[tokio::test]
    async fn keeps_backlog_for_later_dispatches_at_max_batch_size() {
        let queue = BatchQueue::spawn(
            BatcherConfig {
                interval_ms: 1,
                max_batch_size: 4,
                queue_capacity: 32,
                pipelines: 1,
            },
            |items: Vec<usize>| async move { items.into_iter().map(Ok).collect() },
        );

        assert_eq!(queue.submit(0).await.unwrap(), 0);
        let receivers = futures_for(&queue, 1..10).await;
        for (expected, receiver) in (1..10).zip(receivers) {
            assert_eq!(receiver.receive().await.unwrap(), expected);
        }
        let snapshot = queue.snapshot();
        assert_eq!(snapshot.dispatched_items, 10);
        assert_eq!(snapshot.dispatches, 4);
        assert_eq!(snapshot.max_observed_batch, 4);
    }

    #[tokio::test]
    async fn dispatches_complete_backlog_batches_without_interval_throttling() {
        let queue = BatchQueue::spawn(
            BatcherConfig {
                interval_ms: 100,
                max_batch_size: 4,
                queue_capacity: 32,
                pipelines: 1,
            },
            |items: Vec<usize>| async move { items.into_iter().map(Ok).collect() },
        );

        assert_eq!(queue.submit(0).await.unwrap(), 0);
        let started = Instant::now();
        let receivers = futures_for(&queue, 1..=8).await;
        for (expected, receiver) in (1..=8).zip(receivers) {
            assert_eq!(receiver.receive().await.unwrap(), expected);
        }
        assert!(
            started.elapsed() < Duration::from_millis(80),
            "two complete batches must bypass the 100 ms coalescing interval"
        );
    }

    #[tokio::test]
    async fn dispatches_as_soon_as_a_staggered_batch_becomes_full() {
        let queue = BatchQueue::spawn(
            BatcherConfig {
                interval_ms: 100,
                max_batch_size: 4,
                queue_capacity: 32,
                pipelines: 1,
            },
            |items: Vec<usize>| async move { items.into_iter().map(Ok).collect() },
        );

        assert_eq!(queue.submit(0).await.unwrap(), 0);
        let started = Instant::now();
        let first = queue.enqueue(1).await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        let rest = futures_for(&queue, 2..=4).await;
        assert_eq!(first.receive().await.unwrap(), 1);
        for (expected, receiver) in (2..=4).zip(rest) {
            assert_eq!(receiver.receive().await.unwrap(), expected);
        }
        assert!(
            started.elapsed() < Duration::from_millis(80),
            "a complete batch must not wait for the rest of the 100 ms window"
        );
        assert_eq!(queue.snapshot().last_batch_size, 4);
    }

    #[tokio::test]
    async fn partial_batch_gets_a_fresh_window_after_a_slow_dispatch() {
        let queue = BatchQueue::spawn(
            BatcherConfig {
                interval_ms: 20,
                max_batch_size: 4,
                queue_capacity: 32,
                pipelines: 1,
            },
            |items: Vec<usize>| async move {
                if items == [0] {
                    tokio::time::sleep(Duration::from_millis(30)).await;
                }
                items.into_iter().map(Ok).collect()
            },
        );

        assert_eq!(queue.submit(0).await.unwrap(), 0);
        let first = queue.enqueue(1).await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        let rest = futures_for(&queue, 2..=4).await;
        assert_eq!(first.receive().await.unwrap(), 1);
        for (expected, receiver) in (2..=4).zip(rest) {
            assert_eq!(receiver.receive().await.unwrap(), expected);
        }
        assert_eq!(queue.snapshot().dispatches, 2);
        assert_eq!(queue.snapshot().last_batch_size, 4);
    }

    #[tokio::test]
    async fn overlaps_batches_when_multiple_pipelines_are_enabled() {
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let queue = BatchQueue::spawn(
            BatcherConfig {
                interval_ms: 0,
                max_batch_size: 2,
                queue_capacity: 32,
                pipelines: 2,
            },
            {
                let active = Arc::clone(&active);
                let max_active = Arc::clone(&max_active);
                move |items: Vec<usize>| {
                    let active = Arc::clone(&active);
                    let max_active = Arc::clone(&max_active);
                    async move {
                        let now_active = active.fetch_add(1, Ordering::SeqCst) + 1;
                        max_active.fetch_max(now_active, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(20)).await;
                        active.fetch_sub(1, Ordering::SeqCst);
                        items.into_iter().map(Ok).collect()
                    }
                }
            },
        );

        let receivers = futures_for(&queue, 0..8).await;
        for (expected, receiver) in (0..8).zip(receivers) {
            assert_eq!(receiver.receive().await.unwrap(), expected);
        }
        assert_eq!(max_active.load(Ordering::SeqCst), 2);
    }

    async fn futures_for(
        queue: &BatchQueue<usize, usize>,
        values: impl IntoIterator<Item = usize>,
    ) -> Vec<BatchReceiver<usize>> {
        let mut receivers = Vec::new();
        for value in values {
            receivers.push(queue.enqueue(value).await.unwrap());
        }
        receivers
    }
}
