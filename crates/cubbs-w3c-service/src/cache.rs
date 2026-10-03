//! Bounded HTTP prepared-template and verification-result caches.

use super::*;

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
    pub(super) capacity: usize,
    pub(super) state: RwLock<RawTemplateCacheState<V>>,
    pub(super) hits: AtomicU64,
    pub(super) misses: AtomicU64,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct RawTemplateCacheSnapshot {
    pub(super) capacity: usize,
    pub(super) entries: usize,
    pub(super) hits: u64,
    pub(super) misses: u64,
}

impl<V> RawTemplateCache<V> {
    pub(super) fn from_env() -> Self {
        let capacity = env::var("CUBBS_W3C_PREPARED_CACHE_CAPACITY")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(4_096);
        Self {
            capacity,
            state: RwLock::new(RawTemplateCacheState::default()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub(super) fn get(&self, key: &Bytes) -> Option<Arc<V>> {
        if self.capacity == 0 {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let state = self.state.read().expect("raw template cache lock poisoned");
        let value = state
            .hot
            .as_ref()
            .filter(|(cached_key, _)| cached_key == key)
            .map(|(_, value)| Arc::clone(value))
            .or_else(|| state.entries.get(key).cloned());
        if value.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        value
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
        while state.entries.len() >= self.capacity {
            let Some(oldest) = state.insertion_order.pop_front() else {
                break;
            };
            state.entries.remove(&oldest);
        }
        let value = Arc::new(value);
        state.insertion_order.push_back(key.clone());
        state.entries.insert(key.clone(), Arc::clone(&value));
        state.hot = Some((key, Arc::clone(&value)));
        value
    }

    pub(super) fn snapshot(&self) -> RawTemplateCacheSnapshot {
        RawTemplateCacheSnapshot {
            capacity: self.capacity,
            entries: self
                .state
                .read()
                .expect("raw template cache lock poisoned")
                .entries
                .len(),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
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
