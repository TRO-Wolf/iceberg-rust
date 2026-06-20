// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Commit lock coordination — Rust parity for Java `org.apache.iceberg.LockManager`
//! (`iceberg-api`) and `org.apache.iceberg.util.LockManagers` (`iceberg-core`).
//!
//! A [`LockManager`] guards table commits behind a named entity lock: a holder takes the lock
//! with [`LockManager::acquire`] and frees it with [`LockManager::release`]. Iceberg catalogs use
//! this to serialize concurrent commits to the same table.
//!
//! # Java mapping
//!
//! - `LockManager` (interface) → the [`LockManager`] trait. Java's `acquire`/`release` return
//!   `boolean`; we mirror that exactly (the boolean *is* the success signal, not an error channel).
//!   Java's `void initialize(Map)` is folded into construction (Rust constructs-then-uses); the
//!   property parsing it performed lives in [`LockManagers::from`] / [`InMemoryLockManager::new`].
//!   `AutoCloseable.close()` → [`Drop`] (cancels heartbeats, clears the lock table).
//! - `LockManagers.{defaultLockManager,from}` → [`LockManagers::default_lock_manager`] /
//!   [`LockManagers::from`].
//! - `LockManagers.InMemoryLockManager` → [`InMemoryLockManager`].
//! - The `lock.*` keys + defaults from `CatalogProperties` → the [`LOCK_IMPL`] … constants.
//!
//! # Behavioral contract (decoded from `iceberg-core` 1.10.0 bytecode)
//!
//! - `acquire(entity, owner)` retries a single-shot attempt at a fixed `acquire-interval-ms`
//!   cadence until it succeeds (`true`) or `acquire-timeout-ms` elapses (`false`). A single attempt
//!   fails (and is retried) only while the entity is held by a **live, unexpired** lock — note Java
//!   does *not* special-case re-acquisition by the same owner.
//! - A taken lock carries an expiry `heartbeat-timeout-ms` in the future; a background heartbeat
//!   pushes that expiry forward every `heartbeat-interval-ms` so an actively-held lock never lapses.
//!   A lock whose expiry has passed (a crashed holder) is reclaimable by the next `acquire`.
//! - `release(entity, owner)` returns `false` (logging at error level) when the entity is unlocked
//!   or held by a different owner; otherwise it cancels the heartbeat, drops the lock, returns `true`.
//!
//! # Deliberate divergence from Java
//!
//! Java's `InMemoryLockManager` keeps its lock table in `static` fields, so *every* instance in a
//! JVM shares one process-global table. We scope the table **per instance** (for test isolation and
//! Rust idiom) and instead expose the process-global coordination Java relies on through the
//! [`LockManagers::default_lock_manager`] singleton — behaviorally identical for that singleton
//! entry point, which is the sanctioned default. Two independently-constructed
//! [`InMemoryLockManager`]s do not coordinate (Java's would); construct one and share it via `Arc`.
//!
//! Java drives heartbeats from one shared `ScheduledExecutorService` sized by `lock.heartbeat-threads`;
//! we use one parked thread per held lock. The thread count is retained for surface parity
//! ([`InMemoryLockManager::heartbeat_threads`]) but does not gate functionality.
//!
//! # Lock ordering
//!
//! When both inner mutexes are needed, `locks` is always acquired before `heartbeats`, and the
//! `locks` guard is dropped before scheduling a heartbeat — the two guards are never held at once
//! in the reverse order, so the pair cannot deadlock.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::{Error, ErrorKind, Result};

/// Catalog property selecting the `LockManager` implementation (`CatalogProperties.LOCK_IMPL`).
pub const LOCK_IMPL: &str = "lock-impl";

/// Catalog property: heartbeat cadence in ms (`CatalogProperties.LOCK_HEARTBEAT_INTERVAL_MS`).
pub const LOCK_HEARTBEAT_INTERVAL_MS: &str = "lock.heartbeat-interval-ms";
/// Default for [`LOCK_HEARTBEAT_INTERVAL_MS`] — `TimeUnit.SECONDS.toMillis(3)`.
pub const LOCK_HEARTBEAT_INTERVAL_MS_DEFAULT: i64 = 3_000;

/// Catalog property: lock expiry in ms (`CatalogProperties.LOCK_HEARTBEAT_TIMEOUT_MS`).
pub const LOCK_HEARTBEAT_TIMEOUT_MS: &str = "lock.heartbeat-timeout-ms";
/// Default for [`LOCK_HEARTBEAT_TIMEOUT_MS`] — `TimeUnit.SECONDS.toMillis(15)`.
pub const LOCK_HEARTBEAT_TIMEOUT_MS_DEFAULT: i64 = 15_000;

/// Catalog property: heartbeat scheduler thread count (`CatalogProperties.LOCK_HEARTBEAT_THREADS`).
pub const LOCK_HEARTBEAT_THREADS: &str = "lock.heartbeat-threads";
/// Default for [`LOCK_HEARTBEAT_THREADS`].
pub const LOCK_HEARTBEAT_THREADS_DEFAULT: i32 = 4;

/// Catalog property: retry cadence in ms (`CatalogProperties.LOCK_ACQUIRE_INTERVAL_MS`).
pub const LOCK_ACQUIRE_INTERVAL_MS: &str = "lock.acquire-interval-ms";
/// Default for [`LOCK_ACQUIRE_INTERVAL_MS`] — `TimeUnit.SECONDS.toMillis(5)`.
pub const LOCK_ACQUIRE_INTERVAL_MS_DEFAULT: i64 = 5_000;

/// Catalog property: total acquire timeout in ms (`CatalogProperties.LOCK_ACQUIRE_TIMEOUT_MS`).
pub const LOCK_ACQUIRE_TIMEOUT_MS: &str = "lock.acquire-timeout-ms";
/// Default for [`LOCK_ACQUIRE_TIMEOUT_MS`] — `TimeUnit.MINUTES.toMillis(3)`.
pub const LOCK_ACQUIRE_TIMEOUT_MS_DEFAULT: i64 = 180_000;

/// Catalog property naming the lock table (`CatalogProperties.LOCK_TABLE`; used by DynamoDB locking).
pub const LOCK_TABLE: &str = "lock.table";

/// The Java class name of the in-memory implementation, accepted verbatim by [`LockManagers::from`]
/// so a config written for the JVM (`lock-impl=…InMemoryLockManager`) resolves here.
const IN_MEMORY_LOCK_MANAGER_IMPL: &str =
    "org.apache.iceberg.util.LockManagers$InMemoryLockManager";

/// Coordinates exclusive, owner-scoped locks over named entities (typically a table identifier),
/// serializing concurrent commits. Parity for Java `org.apache.iceberg.LockManager`.
pub trait LockManager: std::fmt::Debug + Send + Sync {
    /// Acquire the lock on `entity_id` for `owner_id`, retrying until success or the configured
    /// acquire-timeout elapses. Returns `true` if the lock is now held by `owner_id`.
    fn acquire(&self, entity_id: &str, owner_id: &str) -> bool;

    /// Release `owner_id`'s lock on `entity_id`. Returns `false` if the entity is not locked or is
    /// held by a different owner; `true` once released.
    fn release(&self, entity_id: &str, owner_id: &str) -> bool;
}

/// Millisecond clock seam, injected so lock-expiry behavior is deterministically testable.
trait Clock: std::fmt::Debug + Send + Sync {
    /// Milliseconds since the Unix epoch.
    fn now_millis(&self) -> i64;
}

/// Wall-clock [`Clock`] backed by [`SystemTime`].
#[derive(Debug)]
struct SystemClock;

impl Clock for SystemClock {
    fn now_millis(&self) -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .and_then(|d| i64::try_from(d.as_millis()).ok())
            .unwrap_or(i64::MAX)
    }
}

/// The five timing knobs shared by every `LockManager` (Java `LockManagers.BaseLockManager`),
/// read from catalog properties with the `CatalogProperties` defaults.
#[derive(Debug, Clone, Copy)]
pub struct LockConfig {
    acquire_timeout_ms: i64,
    acquire_interval_ms: i64,
    heartbeat_interval_ms: i64,
    heartbeat_timeout_ms: i64,
    heartbeat_threads: i32,
}

impl Default for LockConfig {
    fn default() -> Self {
        Self {
            acquire_timeout_ms: LOCK_ACQUIRE_TIMEOUT_MS_DEFAULT,
            acquire_interval_ms: LOCK_ACQUIRE_INTERVAL_MS_DEFAULT,
            heartbeat_interval_ms: LOCK_HEARTBEAT_INTERVAL_MS_DEFAULT,
            heartbeat_timeout_ms: LOCK_HEARTBEAT_TIMEOUT_MS_DEFAULT,
            heartbeat_threads: LOCK_HEARTBEAT_THREADS_DEFAULT,
        }
    }
}

impl LockConfig {
    /// Parse the `lock.*` timing properties, falling back to the `CatalogProperties` defaults
    /// (mirrors `BaseLockManager.initialize`). Mirrors Java `PropertyUtil.propertyAsLong/AsInt`:
    /// a present-but-unparsable value is a hard error rather than a silent fallback.
    pub fn from_properties(properties: &HashMap<String, String>) -> Result<Self> {
        Ok(Self {
            acquire_timeout_ms: property_as_i64(
                properties,
                LOCK_ACQUIRE_TIMEOUT_MS,
                LOCK_ACQUIRE_TIMEOUT_MS_DEFAULT,
            )?,
            acquire_interval_ms: property_as_i64(
                properties,
                LOCK_ACQUIRE_INTERVAL_MS,
                LOCK_ACQUIRE_INTERVAL_MS_DEFAULT,
            )?,
            heartbeat_interval_ms: property_as_i64(
                properties,
                LOCK_HEARTBEAT_INTERVAL_MS,
                LOCK_HEARTBEAT_INTERVAL_MS_DEFAULT,
            )?,
            heartbeat_timeout_ms: property_as_i64(
                properties,
                LOCK_HEARTBEAT_TIMEOUT_MS,
                LOCK_HEARTBEAT_TIMEOUT_MS_DEFAULT,
            )?,
            heartbeat_threads: property_as_i32(
                properties,
                LOCK_HEARTBEAT_THREADS,
                LOCK_HEARTBEAT_THREADS_DEFAULT,
            )?,
        })
    }

    /// Total time `acquire` will retry before giving up, in ms.
    pub fn acquire_timeout_ms(&self) -> i64 {
        self.acquire_timeout_ms
    }

    /// Delay between `acquire` retries, in ms.
    pub fn acquire_interval_ms(&self) -> i64 {
        self.acquire_interval_ms
    }

    /// Delay between heartbeat refreshes of a held lock, in ms.
    pub fn heartbeat_interval_ms(&self) -> i64 {
        self.heartbeat_interval_ms
    }

    /// How far in the future a heartbeat pushes a held lock's expiry, in ms.
    pub fn heartbeat_timeout_ms(&self) -> i64 {
        self.heartbeat_timeout_ms
    }

    /// Configured heartbeat scheduler thread count (surface parity; see the module docs).
    pub fn heartbeat_threads(&self) -> i32 {
        self.heartbeat_threads
    }
}

/// Parse a property as an `i64`, or return `default` when the key is absent (Java `propertyAsLong`).
fn property_as_i64(properties: &HashMap<String, String>, key: &str, default: i64) -> Result<i64> {
    match properties.get(key) {
        None => Ok(default),
        Some(value) => value.trim().parse::<i64>().map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot parse lock property '{key}' as an integer: '{value}'"),
            )
            .with_source(e)
        }),
    }
}

/// Parse a property as an `i32`, or return `default` when the key is absent (Java `propertyAsInt`).
fn property_as_i32(properties: &HashMap<String, String>, key: &str, default: i32) -> Result<i32> {
    match properties.get(key) {
        None => Ok(default),
        Some(value) => value.trim().parse::<i32>().map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot parse lock property '{key}' as an integer: '{value}'"),
            )
            .with_source(e)
        }),
    }
}

/// A held lock: its owner and the epoch-ms after which it is considered abandoned.
#[derive(Debug, Clone)]
struct LockContent {
    owner_id: String,
    expire_ms: i64,
}

/// Live heartbeat for one held lock: a parked thread that refreshes the lock's expiry. Dropping or
/// [`Heartbeat::cancel`]-ing it signals the thread to stop and wakes it immediately.
#[derive(Debug)]
struct Heartbeat {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    thread: Thread,
}

impl Heartbeat {
    /// Signal the heartbeat thread to stop and unpark it so it exits without waiting out its
    /// current interval.
    fn cancel(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        self.thread.unpark();
    }
}

impl Drop for Heartbeat {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        self.thread.unpark();
        // Join so the worker (which holds clones of the lock-table Arcs) is gone before we return;
        // it wakes promptly via the unpark above, so this does not block for a full interval.
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// In-process [`LockManager`] holding locks in memory. Parity for Java
/// `LockManagers.InMemoryLockManager`. Cheap to clone-share via [`Arc`]; see the module docs for the
/// per-instance vs. Java-static lock-table divergence.
#[derive(Debug)]
pub struct InMemoryLockManager {
    config: LockConfig,
    locks: Arc<Mutex<HashMap<String, LockContent>>>,
    heartbeats: Arc<Mutex<HashMap<String, Heartbeat>>>,
    clock: Arc<dyn Clock>,
}

impl InMemoryLockManager {
    /// Build an [`InMemoryLockManager`] from catalog properties (the `lock.*` timing knobs).
    pub fn new(properties: &HashMap<String, String>) -> Result<Self> {
        Ok(Self::with_clock(
            LockConfig::from_properties(properties)?,
            Arc::new(SystemClock),
        ))
    }

    /// Build from an explicit [`LockConfig`] and clock — the seam used by tests to drive expiry
    /// deterministically.
    fn with_clock(config: LockConfig, clock: Arc<dyn Clock>) -> Self {
        Self {
            config,
            locks: Arc::new(Mutex::new(HashMap::new())),
            heartbeats: Arc::new(Mutex::new(HashMap::new())),
            clock,
        }
    }

    /// Configured heartbeat scheduler thread count (surface parity; see the module docs).
    pub fn heartbeat_threads(&self) -> i32 {
        self.config.heartbeat_threads
    }

    /// A single acquire attempt. `Ok(())` means the lock is now held by `owner_id`; `Err(())` means
    /// it is held by a live, unexpired holder and the caller should retry. Mirrors
    /// `InMemoryLockManager.acquireOnce`: a present-and-unexpired lock blocks regardless of owner.
    fn acquire_once(&self, entity_id: &str, owner_id: &str) -> std::result::Result<(), ()> {
        {
            let mut locks = lock_guard(&self.locks);
            let now = self.clock.now_millis();
            if let Some(existing) = locks.get(entity_id)
                && existing.expire_ms > now
            {
                return Err(());
            }
            let expire_ms = now.saturating_add(self.config.heartbeat_timeout_ms);
            locks.insert(entity_id.to_string(), LockContent {
                owner_id: owner_id.to_string(),
                expire_ms,
            });
        } // release the `locks` guard before touching `heartbeats` (lock-ordering rule)

        self.schedule_heartbeat(entity_id, owner_id);
        Ok(())
    }

    /// Cancel any existing heartbeat for `entity_id`, then start a fresh one that pushes the lock's
    /// expiry forward every `heartbeat-interval-ms`.
    fn schedule_heartbeat(&self, entity_id: &str, owner_id: &str) {
        let mut heartbeats = lock_guard(&self.heartbeats);
        if let Some(mut previous) = heartbeats.remove(entity_id) {
            previous.cancel();
        }

        // A zero/negative interval would busy-spin; clamp to a sane minimum (the lock still works,
        // expiry is just refreshed at this floor). Java leans on ScheduledExecutorService semantics.
        let interval = Duration::from_millis(
            u64::try_from(self.config.heartbeat_interval_ms.max(1)).unwrap_or(u64::MAX),
        );
        let timeout_ms = self.config.heartbeat_timeout_ms;
        let stop = Arc::new(AtomicBool::new(false));

        let worker_stop = Arc::clone(&stop);
        let worker_locks = Arc::clone(&self.locks);
        let worker_clock = Arc::clone(&self.clock);
        let worker_entity = entity_id.to_string();
        let worker_owner = owner_id.to_string();

        let handle = thread::spawn(move || {
            while !worker_stop.load(Ordering::SeqCst) {
                thread::park_timeout(interval);
                if worker_stop.load(Ordering::SeqCst) {
                    break;
                }
                let now = worker_clock.now_millis();
                let mut locks = lock_guard(&worker_locks);
                // Refresh only if we still own the entry — never resurrect a released/stolen lock.
                if let Some(content) = locks.get_mut(&worker_entity)
                    && content.owner_id == worker_owner
                {
                    content.expire_ms = now.saturating_add(timeout_ms);
                }
            }
        });

        let thread = handle.thread().clone();
        heartbeats.insert(entity_id.to_string(), Heartbeat {
            stop,
            handle: Some(handle),
            thread,
        });
    }

    /// Stop and forget the heartbeat for `entity_id`, if any.
    fn cancel_heartbeat(&self, entity_id: &str) {
        let mut heartbeats = lock_guard(&self.heartbeats);
        if let Some(mut heartbeat) = heartbeats.remove(entity_id) {
            heartbeat.cancel();
        }
    }
}

impl LockManager for InMemoryLockManager {
    fn acquire(&self, entity_id: &str, owner_id: &str) -> bool {
        // Fixed-cadence retry bounded by total timeout — Java drives this with
        // `Tasks.foreach(...).exponentialBackoff(interval, interval, timeout, 1.0)` (scale 1.0 ⇒
        // constant interval) and treats timeout exhaustion as `false`.
        let deadline = self
            .clock
            .now_millis()
            .saturating_add(self.config.acquire_timeout_ms);
        let interval = Duration::from_millis(
            u64::try_from(self.config.acquire_interval_ms.max(0)).unwrap_or(u64::MAX),
        );
        loop {
            if self.acquire_once(entity_id, owner_id).is_ok() {
                return true;
            }
            if self.clock.now_millis() >= deadline {
                return false;
            }
            thread::sleep(interval);
        }
    }

    fn release(&self, entity_id: &str, owner_id: &str) -> bool {
        let mut locks = lock_guard(&self.locks);
        match locks.get(entity_id) {
            None => {
                tracing::error!(entity_id, "Cannot find lock for entity");
                false
            }
            Some(content) if content.owner_id != owner_id => {
                tracing::error!(
                    entity_id,
                    owner_id,
                    current_owner = content.owner_id,
                    "Cannot unlock entity by owner; held by current owner"
                );
                false
            }
            Some(_) => {
                locks.remove(entity_id);
                drop(locks); // release `locks` before taking `heartbeats` (lock-ordering rule)
                self.cancel_heartbeat(entity_id);
                true
            }
        }
    }
}

impl Drop for InMemoryLockManager {
    fn drop(&mut self) {
        // Mirror Java `close()`: cancel every heartbeat and clear the tables. Dropping each
        // `Heartbeat` stops + joins its worker.
        if let Ok(mut heartbeats) = self.heartbeats.lock() {
            heartbeats.clear();
        }
        if let Ok(mut locks) = self.locks.lock() {
            locks.clear();
        }
    }
}

/// Factory for [`LockManager`] instances. Parity for Java `org.apache.iceberg.util.LockManagers`.
#[derive(Debug)]
pub struct LockManagers;

/// Process-wide default [`InMemoryLockManager`] (mirrors Java's static `LOCK_MANAGER_DEFAULT`); this
/// is the single shared instance through which the default path gets process-global coordination.
static DEFAULT_LOCK_MANAGER: LazyLock<Arc<dyn LockManager>> = LazyLock::new(|| {
    Arc::new(InMemoryLockManager::with_clock(
        LockConfig::default(),
        Arc::new(SystemClock),
    ))
});

impl LockManagers {
    /// The shared default in-memory lock manager (Java `LockManagers.defaultLockManager`).
    pub fn default_lock_manager() -> Arc<dyn LockManager> {
        Arc::clone(&DEFAULT_LOCK_MANAGER)
    }

    /// Build a [`LockManager`] from catalog properties, dispatching on [`LOCK_IMPL`]
    /// (Java `LockManagers.from`). An absent `lock-impl`, or the in-memory Java class name, yields an
    /// [`InMemoryLockManager`]; any other value is rejected (JVM dynamic class loading has no Rust
    /// analogue — register a custom [`LockManager`] directly instead).
    pub fn from(properties: &HashMap<String, String>) -> Result<Arc<dyn LockManager>> {
        match properties.get(LOCK_IMPL).map(String::as_str) {
            None | Some(IN_MEMORY_LOCK_MANAGER_IMPL) => {
                Ok(Arc::new(InMemoryLockManager::new(properties)?))
            }
            Some(other) => Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "Unsupported lock-impl '{other}': dynamic lock-manager class loading is \
                     JVM-specific. Use the in-memory manager or construct a custom LockManager."
                ),
            )),
        }
    }
}

/// Recover a poisoned `Mutex` guard rather than propagating the panic — a poisoned lock table only
/// means some unrelated thread panicked while holding it; the map itself is still consistent.
fn lock_guard<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicI64;

    use super::*;

    /// A manually-advanceable [`Clock`] for deterministic expiry tests.
    #[derive(Debug)]
    struct FakeClock {
        now: AtomicI64,
    }

    impl FakeClock {
        fn new(start: i64) -> Arc<Self> {
            Arc::new(Self {
                now: AtomicI64::new(start),
            })
        }

        fn advance(&self, ms: i64) {
            self.now.fetch_add(ms, Ordering::SeqCst);
        }
    }

    impl Clock for FakeClock {
        fn now_millis(&self) -> i64 {
            self.now.load(Ordering::SeqCst)
        }
    }

    fn manager_with(clock: Arc<dyn Clock>, config: LockConfig) -> InMemoryLockManager {
        InMemoryLockManager::with_clock(config, clock)
    }

    #[test]
    fn defaults_match_java_catalog_properties() {
        let config = LockConfig::default();
        assert_eq!(config.acquire_timeout_ms(), 180_000);
        assert_eq!(config.acquire_interval_ms(), 5_000);
        assert_eq!(config.heartbeat_interval_ms(), 3_000);
        assert_eq!(config.heartbeat_timeout_ms(), 15_000);
        assert_eq!(config.heartbeat_threads(), 4);
    }

    #[test]
    fn from_properties_overrides_and_falls_back() {
        let mut props = HashMap::new();
        props.insert(LOCK_ACQUIRE_TIMEOUT_MS.to_string(), "1234".to_string());
        props.insert(LOCK_HEARTBEAT_THREADS.to_string(), "9".to_string());
        let config = LockConfig::from_properties(&props).expect("parse");
        assert_eq!(
            config.acquire_timeout_ms(),
            1234,
            "overridden value is read"
        );
        assert_eq!(config.heartbeat_threads(), 9);
        // Untouched keys keep the Java defaults.
        assert_eq!(config.acquire_interval_ms(), 5_000);
        assert_eq!(config.heartbeat_timeout_ms(), 15_000);
    }

    #[test]
    fn from_properties_rejects_unparsable_value() {
        let mut props = HashMap::new();
        props.insert(
            LOCK_ACQUIRE_TIMEOUT_MS.to_string(),
            "not-a-number".to_string(),
        );
        let err = LockConfig::from_properties(&props).expect_err("should reject");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
    }

    #[test]
    fn acquire_then_release_round_trips() {
        let clock = FakeClock::new(1_000);
        let mgr = manager_with(clock, LockConfig::default());

        assert!(
            mgr.acquire("table_a", "owner_1"),
            "free entity is acquirable"
        );
        assert!(
            mgr.release("table_a", "owner_1"),
            "owner can release its own lock"
        );
        // Released ⇒ re-acquirable, even by a different owner.
        assert!(mgr.acquire("table_a", "owner_2"));
        assert!(mgr.release("table_a", "owner_2"));
    }

    #[test]
    fn second_owner_blocks_while_held_then_times_out() {
        let clock = FakeClock::new(1_000);
        // Zero acquire-timeout ⇒ exactly one attempt, no sleeping, deterministic with a fake clock.
        let config = LockConfig {
            acquire_timeout_ms: 0,
            ..LockConfig::default()
        };
        let mgr = manager_with(clock, config);

        assert!(mgr.acquire("t", "owner_1"));
        // Held + unexpired ⇒ another owner cannot take it (Java throws ISE → retried → false).
        assert!(
            !mgr.acquire("t", "owner_2"),
            "held lock blocks a second owner"
        );
        // Even the same owner cannot re-acquire an unexpired lock (mirrors Java acquireOnce).
        assert!(!mgr.acquire("t", "owner_1"));
    }

    #[test]
    fn expired_lock_is_reclaimable() {
        let clock = FakeClock::new(1_000);
        let config = LockConfig {
            acquire_timeout_ms: 0,
            heartbeat_timeout_ms: 15_000,
            // Huge heartbeat interval ⇒ the background heartbeat never fires during the test, so the
            // lock genuinely lapses once we advance the clock past its expiry.
            heartbeat_interval_ms: i64::MAX,
            ..LockConfig::default()
        };
        let mgr = manager_with(Arc::clone(&clock) as Arc<dyn Clock>, config);

        assert!(mgr.acquire("t", "owner_1"));
        assert!(!mgr.acquire("t", "owner_2"), "still held before expiry");

        clock.advance(15_001); // step just past expire_ms
        assert!(
            mgr.acquire("t", "owner_2"),
            "a lapsed lock is reclaimable by a new owner"
        );
    }

    #[test]
    fn release_rejects_wrong_owner_and_missing_entity() {
        let clock = FakeClock::new(1_000);
        let mgr = manager_with(clock, LockConfig::default());

        assert!(
            !mgr.release("never_locked", "owner_1"),
            "missing entity ⇒ false"
        );

        assert!(mgr.acquire("t", "owner_1"));
        assert!(
            !mgr.release("t", "owner_2"),
            "a non-owner cannot release the lock"
        );
        // The real owner still holds it and can release.
        assert!(mgr.release("t", "owner_1"));
    }

    #[test]
    fn heartbeat_refreshes_expiry_so_a_held_lock_does_not_lapse() {
        // Real wall-clock + tiny intervals: prove the background heartbeat actually pushes expiry.
        let config = LockConfig {
            heartbeat_interval_ms: 5,
            heartbeat_timeout_ms: 40,
            ..LockConfig::default()
        };
        let mgr = manager_with(Arc::new(SystemClock), config);

        assert!(mgr.acquire("t", "owner_1"));
        let initial_expiry = lock_guard(&mgr.locks).get("t").map(|c| c.expire_ms);

        // Sleep well past one heartbeat interval but the heartbeat keeps the lock alive.
        thread::sleep(Duration::from_millis(40));
        let refreshed_expiry = lock_guard(&mgr.locks).get("t").map(|c| c.expire_ms);

        assert!(initial_expiry.is_some() && refreshed_expiry.is_some());
        assert!(
            refreshed_expiry > initial_expiry,
            "heartbeat should advance expiry: {initial_expiry:?} -> {refreshed_expiry:?}"
        );
        assert!(mgr.release("t", "owner_1"));
    }

    #[test]
    fn acquire_blocks_then_succeeds_after_release_across_threads() {
        // Real clock; owner_2 retries on a short interval until owner_1 releases.
        let config = LockConfig {
            acquire_timeout_ms: 5_000,
            acquire_interval_ms: 2,
            ..LockConfig::default()
        };
        let mgr = Arc::new(manager_with(Arc::new(SystemClock), config));
        assert!(mgr.acquire("t", "owner_1"));

        let mgr_clone = Arc::clone(&mgr);
        let waiter = thread::spawn(move || mgr_clone.acquire("t", "owner_2"));

        thread::sleep(Duration::from_millis(20));
        assert!(mgr.release("t", "owner_1"), "owner_1 releases");

        assert!(
            waiter.join().expect("waiter thread"),
            "owner_2 acquires once the lock frees"
        );
        assert!(mgr.release("t", "owner_2"));
    }

    #[test]
    fn from_dispatches_in_memory_and_rejects_unknown_impl() {
        // Absent lock-impl ⇒ in-memory.
        let default = LockManagers::from(&HashMap::new()).expect("default impl");
        assert!(default.acquire("t", "o"));
        assert!(default.release("t", "o"));

        // Explicit Java in-memory class name ⇒ in-memory.
        let mut props = HashMap::new();
        props.insert(
            LOCK_IMPL.to_string(),
            IN_MEMORY_LOCK_MANAGER_IMPL.to_string(),
        );
        assert!(LockManagers::from(&props).is_ok());

        // Any other impl ⇒ FeatureUnsupported (no JVM class loading).
        props.insert(
            LOCK_IMPL.to_string(),
            "com.example.CustomLockManager".to_string(),
        );
        let err = LockManagers::from(&props).expect_err("unknown impl rejected");
        assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
    }

    #[test]
    fn default_lock_manager_is_a_shared_singleton() {
        let a = LockManagers::default_lock_manager();
        let b = LockManagers::default_lock_manager();
        assert!(Arc::ptr_eq(&a, &b), "default manager is process-global");
        // Use unique entity ids to avoid interfering with other tests touching the singleton.
        assert!(a.acquire("singleton_entity_xyz", "o1"));
        assert!(b.release("singleton_entity_xyz", "o1"));
    }
}
