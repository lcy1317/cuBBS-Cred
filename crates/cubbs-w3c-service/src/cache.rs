//! Bounded HTTP prepared-template and verification-result caches.

use std::any::Any;
use std::cell::RefCell;
use std::sync::atomic::AtomicUsize;

use super::*;

#[repr(align(64))]
#[derive(Debug, Default)]
struct PaddedAtomicU64(AtomicU64);

const COUNTER_SLOTS: usize = 32;
static NEXT_RAW_CACHE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_THREAD_SLOT: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static THREAD_COUNTER_SLOT: usize =
        NEXT_THREAD_SLOT.fetch_add(1, Ordering::Relaxed) % COUNTER_SLOTS;
    static L1_RAW_TEMPLATE_SLOTS: RefCell<[(u64, u64, Bytes, Arc<dyn Any + Send + Sync>); 4]> =
        RefCell::new(std::array::from_fn(|_| {
            (0, 0, Bytes::new(), Arc::new(()) as Arc<dyn Any + Send + Sync>)
        }));
}

#[inline]
fn thread_counter_slot() -> usize {
    THREAD_COUNTER_SLOT.with(|slot| *slot)
}

#[derive(Debug)]
pub(super) struct RawTemplateCacheState<V> {
    pub(super) entries: HashMap<Bytes, Arc<V>>,
    pub(super) insertion_order: VecDeque<Bytes>,
    pub(super) hot: Option<(Bytes, Arc<V>)>,
}

impl<V> Default for RawTemplateCacheState<V> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            insertion_order: VecDeque::new(),
            hot: None,
        }
    }
}

#[derive(Debug)]
pub(super) struct RawTemplateCache<V> {
    id: u64,
    generation: AtomicU64,
    pub(super) capacity: usize,
    pub(super) state: RwLock<RawTemplateCacheState<V>>,
    hits: Box<[PaddedAtomicU64; COUNTER_SLOTS]>,
    misses: Box<[PaddedAtomicU64; COUNTER_SLOTS]>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct RawTemplateCacheSnapshot {
    pub(super) capacity: usize,
    pub(super) entries: usize,
    pub(super) hits: u64,
    pub(super) misses: u64,
}

impl<V: Send + Sync + 'static> RawTemplateCache<V> {
    pub(super) fn from_env() -> Self {
        let capacity = env::var("CUBBS_W3C_PREPARED_CACHE_CAPACITY")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(4_096);
        Self::with_capacity(capacity)
    }

    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            id: NEXT_RAW_CACHE_ID.fetch_add(1, Ordering::Relaxed),
            generation: AtomicU64::new(1),
            capacity,
            state: RwLock::new(RawTemplateCacheState::default()),
            hits: Box::new(std::array::from_fn(|_| PaddedAtomicU64::default())),
            misses: Box::new(std::array::from_fn(|_| PaddedAtomicU64::default())),
        }
    }

    pub(super) fn get(&self, key: &Bytes) -> Option<Arc<V>> {
        let slot = thread_counter_slot();
        if self.capacity == 0 {
            self.misses[slot].0.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let gen = self.generation.load(Ordering::Relaxed);
        let l1_idx = (self.id as usize) & 3;
        if let Some(hit) = L1_RAW_TEMPLATE_SLOTS.with(|slots| {
            let slots = slots.borrow();
            let entry = &slots[l1_idx];
            if entry.0 == self.id && entry.1 == gen && entry.2 == *key {
                Arc::clone(&entry.3).downcast::<V>().ok()
            } else {
                None
            }
        }) {
            self.hits[slot].0.fetch_add(1, Ordering::Relaxed);
            return Some(hit);
        }
        let state = self.state.read().expect("raw template cache lock poisoned");
        let value = state
            .hot
            .as_ref()
            .filter(|(cached_key, _)| cached_key == key)
            .map(|(_, value)| Arc::clone(value))
            .or_else(|| state.entries.get(key).cloned());
        if let Some(found) = value {
            L1_RAW_TEMPLATE_SLOTS.with(|slots| {
                let mut slots = slots.borrow_mut();
                slots[l1_idx] = (
                    self.id,
                    gen,
                    key.clone(),
                    Arc::clone(&found) as Arc<dyn Any + Send + Sync>,
                );
            });
            self.hits[slot].0.fetch_add(1, Ordering::Relaxed);
            Some(found)
        } else {
            self.misses[slot].0.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    pub(super) fn insert(&self, key: Bytes, value: V) -> Arc<V> {
        if self.capacity == 0 {
            return Arc::new(value);
        }
        let mut state = self
            .state
            .write()
            .expect("raw template cache lock poisoned");
        if let Some(existing) = state.entries.get(&key).cloned() {
            state.hot = Some((key, Arc::clone(&existing)));
            return existing;
        }
        let mut evicted = false;
        while state.entries.len() >= self.capacity {
            let Some(oldest) = state.insertion_order.pop_front() else {
                break;
            };
            state.entries.remove(&oldest);
            evicted = true;
        }
        if evicted {
            self.generation.fetch_add(1, Ordering::Relaxed);
        }
        let value = Arc::new(value);
        state.insertion_order.push_back(key.clone());
        state.entries.insert(key.clone(), Arc::clone(&value));
        state.hot = Some((key, Arc::clone(&value)));
        value
    }

    pub(super) fn snapshot(&self) -> RawTemplateCacheSnapshot {
        let hits = self
            .hits
            .iter()
            .map(|c| c.0.load(Ordering::Relaxed))
            .sum();
        let misses = self
            .misses
            .iter()
            .map(|c| c.0.load(Ordering::Relaxed))
            .sum();
        RawTemplateCacheSnapshot {
            capacity: self.capacity,
            entries: self
                .state
                .read()
                .expect("raw template cache lock poisoned")
                .entries
                .len(),
            hits,
            misses,
        }
    }
}

#[derive(Debug)]
pub(super) struct VerificationResultCache {
    pub(super) capacity: usize,
    pub(super) state: Mutex<VerificationResultCacheState>,
    pub(super) hits: AtomicU64,
    pub(super) misses: AtomicU64,
    pub(super) inserts: AtomicU64,
}

#[derive(Debug, Default)]
pub(super) struct VerificationResultCacheState {
    pub(super) entries: HashSet<[u8; 32]>,
    pub(super) insertion_order: VecDeque<[u8; 32]>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct VerificationResultCacheSnapshot {
    pub(super) capacity: usize,
    pub(super) entries: usize,
    pub(super) hits: u64,
    pub(super) misses: u64,
    pub(super) inserts: u64,
}

impl VerificationResultCache {
    pub(super) fn from_env() -> Self {
        Self::with_capacity(
            env::var("CUBBS_W3C_VERIFY_CACHE_CAPACITY")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(65_536),
        )
    }

    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            state: Mutex::new(VerificationResultCacheState::default()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            inserts: AtomicU64::new(0),
        }
    }

    pub(super) fn enabled(&self) -> bool {
        self.capacity != 0
    }

    pub(super) fn contains(&self, document: &Value) -> bool {
        if self.capacity == 0 {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let Ok(key) = json_content_digest(document) else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return false;
        };
        let contains = self
            .state
            .lock()
            .expect("verification cache mutex poisoned")
            .entries
            .contains(&key);
        if contains {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        contains
    }

    pub(super) fn insert(&self, document: &Value) {
        if self.capacity == 0 {
            return;
        }
        let Ok(key) = json_content_digest(document) else {
            return;
        };
        let mut state = self
            .state
            .lock()
            .expect("verification cache mutex poisoned");
        if state.entries.contains(&key) {
            return;
        }
        while state.entries.len() >= self.capacity {
            let Some(oldest) = state.insertion_order.pop_front() else {
                break;
            };
            state.entries.remove(&oldest);
        }
        state.insertion_order.push_back(key);
        state.entries.insert(key);
        self.inserts.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn snapshot(&self) -> VerificationResultCacheSnapshot {
        VerificationResultCacheSnapshot {
            capacity: self.capacity,
            entries: self
                .state
                .lock()
                .expect("verification cache mutex poisoned")
                .entries
                .len(),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            inserts: self.inserts.load(Ordering::Relaxed),
        }
    }
}
