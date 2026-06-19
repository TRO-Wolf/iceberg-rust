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

//! Events and listeners — the engine-agnostic port of Java's `org.apache.iceberg.events`.
//!
//! This module ports Java's process-global event bus: a [`Listener`] trait, a process-wide
//! static [registry](register) ([`register`] / [`notify_all`]), and the three immutable event
//! value types Iceberg fires from its scan and commit paths ([`ScanEvent`],
//! [`IncrementalScanEvent`], [`CreateSnapshotEvent`]).
//!
//! # Java parity
//!
//! - [`Listener`] mirrors Java's `interface Listener<E> { void notify(E event); }`. The Rust
//!   trait takes the event by shared reference (`fn notify(&self, event: &E)`): Java passes the
//!   immutable event by reference-semantics anyway, and Rust's borrow keeps the dispatcher's
//!   single event instance shared across every listener without a clone per listener.
//! - The [registry](register) mirrors Java `org.apache.iceberg.events.Listeners`: a
//!   process-global `Map<Class<?>, Queue<Listener<?>>>` keyed by the event's EXACT runtime type
//!   (no supertype walk), with exactly two operations — [`register`] (Java
//!   `register(Listener<E>, Class<E>)`) appends a listener to the per-type queue in insertion
//!   order, and [`notify_all`] (Java `notifyAll(E event)`) dispatches an event to every listener
//!   registered for that exact type, in registration order. Rust keys on [`TypeId`] (the
//!   compile-time analogue of `event.getClass()`); a listener registered for a DIFFERENT event
//!   type is never called.
//! - [`ScanEvent`], [`IncrementalScanEvent`], and [`CreateSnapshotEvent`] mirror the Java event
//!   classes of the same names field-for-field (private-final value objects in Java; immutable
//!   structs with read accessors here), in Java-exact constructor-argument order.
//!
//! # Divergences from Java (documented)
//!
//! - **Panic propagation in [`notify_all`].** Java's `Listeners.notifyAll` has no try/catch, so a
//!   throwing listener propagates to the firing site. [`notify_all`] preserves that: a panicking
//!   listener unwinds to the caller. The *commit* emit site (`Transaction::do_commit`) isolates a
//!   panicking listener so a notification failure never fails a committed transaction (Java
//!   `SnapshotProducer.notifyListeners` wraps the call in a best-effort try/catch); the *scan*
//!   emit site does not isolate, matching Java `SnapshotScan.planFiles`, which fires inside the
//!   planning call with no guard.
//! - **Commit-site panic is logged, not swallowed.** Java's commit-site catch logs a `LOG.warn`.
//!   The commit emit site (`Transaction::do_commit`) mirrors that: when a listener panics, the
//!   `catch_unwind` contains it (the committed transaction still returns `Ok`) and a
//!   `tracing::warn!` records the failure with the snapshot id / sequence number / operation /
//!   table name. The panic payload itself is never logged (it could carry caller data), so the
//!   message is generic and only the non-sensitive identifiers are structured fields.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};

use crate::expr::Predicate;
use crate::spec::SchemaRef;

/// A listener for events of type `E`.
///
/// Mirrors Java's `interface Listener<E> { void notify(E event); }`. Implementations are held
/// in the process-global [registry](register) behind an [`Arc`], so they must be `Send + Sync`.
/// A listener receives the firing site's single event instance by shared reference.
///
/// =====================================================================================
pub trait Listener<E>: Send + Sync {
    /// Handles an event. Mirrors Java `Listener.notify(E)`.
    ///
    /// Like Java, this may panic (throw): the dispatcher [`notify_all`] does not guard the call,
    /// so a panic propagates to the firing site unless that site isolates it (the commit site
    /// does; the scan site does not — see the module docs).
    fn notify(&self, event: &E);
}

/// The process-global listener registry.
///
/// Mirrors Java `Listeners`' static `Map<Class<?>, Queue<Listener<?>>>`: a map from the event's
/// exact [`TypeId`] to the insertion-ordered list of listeners registered for that type. The
/// listeners are stored type-erased (`Box<dyn Any + Send + Sync>` wrapping an
/// `Arc<dyn Listener<E>>`); [`notify_all`] recovers the concrete type by [downcast](Any) keyed on
/// the same `TypeId`, so a listener can only ever be invoked for the event type it was registered
/// for.
///
/// The lock is a coarse [`RwLock`] (registration is rare, dispatch is read-mostly), matching the
/// concurrency profile of Java's `ConcurrentMap` of `ConcurrentLinkedQueue`s.
///
/// =====================================================================================
static LISTENERS: LazyLock<RwLock<ListenerMap>> = LazyLock::new(|| RwLock::new(HashMap::new()));

/// The registry map: event [`TypeId`] → its insertion-ordered, type-erased listeners. Each entry
/// is a `Box<dyn Any + Send + Sync>` wrapping an `Arc<dyn Listener<E>>`, recovered by downcast in
/// [`notify_all`].
type ListenerMap = HashMap<TypeId, Vec<Box<dyn Any + Send + Sync>>>;

/// Registers `listener` for events of type `E`.
///
/// Mirrors Java `Listeners.register(Listener<E>, Class<E>)`:
/// `computeIfAbsent(cls, new Queue()).add(listener)`. Appends to the per-type queue in
/// registration order; [`notify_all`] then dispatches in that order. Registering the same
/// listener twice registers it twice (Java's `Queue.add` does not deduplicate), so it is
/// notified twice.
pub fn register<E: 'static>(listener: Arc<dyn Listener<E>>) {
    let mut guard = LISTENERS
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard
        .entry(TypeId::of::<E>())
        .or_default()
        // Box the `Arc<dyn Listener<E>>` itself (not the listener) so the stored `Any` downcasts
        // back to exactly `Arc<dyn Listener<E>>` in `notify_all`.
        .push(Box::new(listener));
}

/// Dispatches `event` to every listener registered for type `E`, in registration order.
///
/// Mirrors Java `Listeners.notifyAll(E event)`: look up the queue for `event.getClass()` (here,
/// `TypeId::of::<E>()`), then call `listener.notify(event)` on each. Keyed on the EXACT type —
/// a listener registered for a different event type is not called.
///
/// Like Java, this does NOT guard the per-listener call: a panicking listener unwinds to the
/// caller. To avoid holding the registry lock across the (arbitrary, possibly re-entrant)
/// listener callbacks — which would deadlock a listener that itself calls [`register`] or
/// [`notify_all`], and would violate the no-lock-across-callback rule — the matching listeners
/// are cloned into a local `Vec` and the read guard is DROPPED before any `notify` runs.
pub fn notify_all<E: 'static>(event: &E) {
    // Test isolation: the registry is a process-global static shared by EVERY test in the crate,
    // but the suite runs many real scans/commits in parallel. Under `cfg(test)`, dispatch only on
    // a thread that has ARMED itself (held the event-test lock) — so a concurrent NON-event test's
    // scan/commit, on a different thread, never fires another test's registered listener. The
    // `#[tokio::test]` default `current_thread` runtime polls a test's whole body (and these
    // synchronous emit sites) on that one test thread, so the per-thread arm is exact. In
    // production (`not(test)`) this gate is compiled out and dispatch is unconditional, faithful
    // to Java's always-on `Listeners.notifyAll`.
    #[cfg(test)]
    if !test_support::is_armed() {
        return;
    }

    // Clone the matching `Arc`s under the read lock, then drop the guard before notifying.
    let listeners: Vec<Arc<dyn Listener<E>>> = {
        let guard = LISTENERS
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match guard.get(&TypeId::of::<E>()) {
            None => return,
            Some(entries) => entries
                .iter()
                .filter_map(|entry| entry.downcast_ref::<Arc<dyn Listener<E>>>())
                .map(Arc::clone)
                .collect(),
        }
    };

    for listener in listeners {
        listener.notify(event);
    }
}

/// Removes every registered listener.
///
/// Test-only: the registry is process-global, so listeners registered by one test leak into the
/// next. Tests that assert on dispatch take a shared test mutex and call this to start from a
/// clean registry. Not part of the public API (Java's `Listeners` has no clear either).
#[cfg(test)]
pub(crate) fn clear() {
    LISTENERS
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
}

/// Crate-wide test support for the process-global event registry.
///
/// The registry is a single process-static shared by EVERY test in the crate, while the suite
/// runs many real scans/commits in parallel. Two layers give isolation:
/// - a process-wide [lock](self::test_support::lock) serializes the event tests against EACH
///   OTHER and clears the registry on acquisition (no leaked listeners between event tests); and
/// - a per-thread ARM flag (set while the guard is held) gates [`notify_all`] under `cfg(test)`,
///   so a concurrent non-event test's scan/commit — on a different thread — fires nothing.
///
/// Hold the [`EventTestGuard`] for the whole test body. Dropping it disarms the thread and
/// releases the lock.
#[cfg(test)]
pub(crate) mod test_support {
    use std::cell::Cell;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    use super::clear;

    /// The process-wide serialization lock for event-registry tests.
    static EVENT_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    thread_local! {
        /// Whether the current thread has armed event dispatch (holds the test lock). `notify_all`
        /// consults this under `cfg(test)`; a non-event thread leaves it `false` and fires nothing.
        static ARMED: Cell<bool> = const { Cell::new(false) };
    }

    /// Returns whether the current thread has armed event dispatch.
    pub(super) fn is_armed() -> bool {
        ARMED.with(Cell::get)
    }

    /// An RAII guard that holds the event-test lock and keeps the current thread armed.
    ///
    /// Dropping it disarms the thread (so a later test on the same reused thread starts un-armed)
    /// and releases the lock.
    pub(crate) struct EventTestGuard {
        _lock: MutexGuard<'static, ()>,
    }

    impl Drop for EventTestGuard {
        fn drop(&mut self) {
            ARMED.with(|armed| armed.set(false));
        }
    }

    /// Acquires the process-wide event-test lock, clears the registry, and arms the current
    /// thread for event dispatch.
    ///
    /// Hold the returned guard for the whole test body: it serializes all event tests across the
    /// crate, starts each from an empty registry, and arms dispatch on this thread so the test's
    /// real scans/commits actually fire — while concurrent non-event tests stay silent. Recovers
    /// a poisoned lock (a panicking best-effort test must not wedge the rest of the suite).
    pub(crate) fn lock() -> EventTestGuard {
        let lock = EVENT_TEST_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        clear();
        ARMED.with(|armed| armed.set(true));
        EventTestGuard { _lock: lock }
    }
}

/// Fired when a [table scan](crate::scan::TableScan) plans its files.
///
/// Mirrors Java `org.apache.iceberg.events.ScanEvent`. Constructed by
/// [`TableScan::plan_files`](crate::scan::TableScan::plan_files) for a non-empty scan, just
/// before the file plan runs — exactly Java `SnapshotScan.planFiles`, which fires the event when
/// the scanned snapshot is non-null, before `doPlanFiles()`.
///
/// =====================================================================================
#[derive(Clone, Debug, PartialEq)]
pub struct ScanEvent {
    table_name: String,
    snapshot_id: i64,
    filter: Predicate,
    projection: SchemaRef,
}

impl ScanEvent {
    /// Builds a scan event. Java-exact argument order:
    /// `ScanEvent(String tableName, long snapshotId, Expression filter, Schema projection)`.
    pub fn new(
        table_name: String,
        snapshot_id: i64,
        filter: Predicate,
        projection: SchemaRef,
    ) -> Self {
        Self {
            table_name,
            snapshot_id,
            filter,
            projection,
        }
    }

    /// The fully-qualified table name. Java `tableName()`.
    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    /// The id of the snapshot the scan reads. Java `snapshotId()`.
    pub fn snapshot_id(&self) -> i64 {
        self.snapshot_id
    }

    /// The UNBOUND row filter applied to the scan. Java `filter()`.
    pub fn filter(&self) -> &Predicate {
        &self.filter
    }

    /// The schema the scan projects against. Java `projection()`.
    pub fn projection(&self) -> &SchemaRef {
        &self.projection
    }
}

/// Fired when an [incremental scan](crate::scan::IncrementalAppendScan) plans its files.
///
/// Mirrors Java `org.apache.iceberg.events.IncrementalScanEvent`. Fired from the shared
/// `BaseIncrementalScan.planFiles()` (so both the append scan and the changelog scan fire it),
/// before `doPlanFiles()`. The `from` bound is resolved to a concrete snapshot id and an
/// `inclusive` flag: an explicit exclusive `from` yields `(from, inclusive = false)`; an absent
/// `from` yields `(oldestAncestorOf(to), inclusive = true)`.
///
/// =====================================================================================
#[derive(Clone, Debug, PartialEq)]
pub struct IncrementalScanEvent {
    table_name: String,
    from_snapshot_id: i64,
    to_snapshot_id: i64,
    filter: Predicate,
    projection: SchemaRef,
    from_snapshot_inclusive: bool,
}

impl IncrementalScanEvent {
    /// Builds an incremental scan event. Java-exact argument order:
    /// `IncrementalScanEvent(String tableName, long fromSnapshotId, long toSnapshotId,
    /// Expression filter, Schema projection, boolean fromSnapshotInclusive)`.
    pub fn new(
        table_name: String,
        from_snapshot_id: i64,
        to_snapshot_id: i64,
        filter: Predicate,
        projection: SchemaRef,
        from_snapshot_inclusive: bool,
    ) -> Self {
        Self {
            table_name,
            from_snapshot_id,
            to_snapshot_id,
            filter,
            projection,
            from_snapshot_inclusive,
        }
    }

    /// The fully-qualified table name. Java `tableName()`.
    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    /// The resolved `from` snapshot id. Java `fromSnapshotId()`.
    pub fn from_snapshot_id(&self) -> i64 {
        self.from_snapshot_id
    }

    /// The inclusive `to` snapshot id. Java `toSnapshotId()`.
    pub fn to_snapshot_id(&self) -> i64 {
        self.to_snapshot_id
    }

    /// The UNBOUND row filter applied. Java `filter()`.
    pub fn filter(&self) -> &Predicate {
        &self.filter
    }

    /// The schema the scan projects against. Java `projection()`.
    pub fn projection(&self) -> &SchemaRef {
        &self.projection
    }

    /// Whether the `from` snapshot is itself included in the range. Java
    /// `isFromSnapshotInclusive()`.
    pub fn is_from_snapshot_inclusive(&self) -> bool {
        self.from_snapshot_inclusive
    }
}

/// Fired after a snapshot is successfully committed to a table.
///
/// Mirrors Java `org.apache.iceberg.events.CreateSnapshotEvent`. Fired from
/// [`Transaction::do_commit`](crate::transaction::Transaction) once per added snapshot, AFTER the
/// catalog commit succeeds — exactly Java `SnapshotProducer.notifyListeners`, called from
/// `commit()` after `ops.commit` returns, in a best-effort guard so a listener failure never
/// fails the commit.
///
/// =====================================================================================
#[derive(Clone, Debug, PartialEq)]
pub struct CreateSnapshotEvent {
    table_name: String,
    operation: String,
    snapshot_id: i64,
    sequence_number: i64,
    summary: HashMap<String, String>,
}

impl CreateSnapshotEvent {
    /// Builds a create-snapshot event. Java-exact argument order:
    /// `CreateSnapshotEvent(String tableName, String operation, long snapshotId,
    /// long sequenceNumber, Map<String,String> summary)`.
    pub fn new(
        table_name: String,
        operation: String,
        snapshot_id: i64,
        sequence_number: i64,
        summary: HashMap<String, String>,
    ) -> Self {
        Self {
            table_name,
            operation,
            snapshot_id,
            sequence_number,
            summary,
        }
    }

    /// The fully-qualified table name. Java `tableName()`.
    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    /// The snapshot operation (`append` / `replace` / `overwrite` / `delete`). Java
    /// `operation()`.
    pub fn operation(&self) -> &str {
        &self.operation
    }

    /// The id of the committed snapshot. Java `snapshotId()`.
    pub fn snapshot_id(&self) -> i64 {
        self.snapshot_id
    }

    /// The sequence number of the committed snapshot. Java `sequenceNumber()`.
    pub fn sequence_number(&self) -> i64 {
        self.sequence_number
    }

    /// The snapshot summary map. Java `summary()`.
    pub fn summary(&self) -> &HashMap<String, String> {
        &self.summary
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::test_support::lock as registry_lock;
    use super::*;
    use crate::spec::{NestedField, PrimitiveType, Schema, Type};

    fn test_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_fields(vec![Arc::new(NestedField::required(
                    1,
                    "x",
                    Type::Primitive(PrimitiveType::Long),
                ))])
                .build()
                .expect("schema"),
        )
    }

    /// A listener that records every event it receives into a shared sink.
    struct Recorder<E: Clone> {
        sink: Arc<Mutex<Vec<E>>>,
    }

    impl<E: Clone + Send + Sync> Listener<E> for Recorder<E> {
        fn notify(&self, event: &E) {
            self.sink
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(event.clone());
        }
    }

    /// Risk: dispatch goes to the wrong type (or to no one), so the registry silently drops
    /// events. Pins that `notify_all` reaches every listener registered for the EXACT type, in
    /// registration order, and reaches NO listener of a different type.
    #[test]
    fn test_notify_all_dispatches_by_exact_type() {
        let _guard = registry_lock();

        let scan_sink = Arc::new(Mutex::new(Vec::<ScanEvent>::new()));
        let other_sink = Arc::new(Mutex::new(Vec::<CreateSnapshotEvent>::new()));

        register::<ScanEvent>(Arc::new(Recorder {
            sink: scan_sink.clone(),
        }));
        // A second scan listener: both must fire, in registration order.
        let scan_sink2 = Arc::new(Mutex::new(Vec::<ScanEvent>::new()));
        register::<ScanEvent>(Arc::new(Recorder {
            sink: scan_sink2.clone(),
        }));
        // A listener for a DIFFERENT event type must NOT be called by a ScanEvent.
        register::<CreateSnapshotEvent>(Arc::new(Recorder {
            sink: other_sink.clone(),
        }));

        let event = ScanEvent::new(
            "db.tbl".to_string(),
            7,
            Predicate::AlwaysTrue,
            test_schema(),
        );
        notify_all(&event);

        assert_eq!(
            scan_sink.lock().unwrap().as_slice(),
            std::slice::from_ref(&event)
        );
        assert_eq!(
            scan_sink2.lock().unwrap().as_slice(),
            std::slice::from_ref(&event)
        );
        assert!(
            other_sink.lock().unwrap().is_empty(),
            "a CreateSnapshotEvent listener must not fire on a ScanEvent"
        );
    }

    /// Risk: dispatching for a type with no listeners panics or errors instead of being a no-op.
    #[test]
    fn test_notify_all_with_no_listeners_is_a_noop() {
        let _guard = registry_lock();
        // No registration: must not panic.
        notify_all(&ScanEvent::new(
            "db.tbl".to_string(),
            1,
            Predicate::AlwaysTrue,
            test_schema(),
        ));
    }

    /// Risk: a listener that re-enters the registry (registers/dispatches from inside `notify`)
    /// deadlocks because the dispatcher held the lock across the callback. Pins that the lock is
    /// dropped before the callback runs.
    #[test]
    fn test_reentrant_listener_does_not_deadlock() {
        let _guard = registry_lock();

        struct Reentrant;
        impl Listener<ScanEvent> for Reentrant {
            fn notify(&self, _event: &ScanEvent) {
                // Re-enter: a fresh dispatch of a different type from inside a callback. If the
                // dispatcher held the read lock across this call it would deadlock on the inner
                // read (and a write would deadlock outright).
                notify_all(&CreateSnapshotEvent::new(
                    "db.tbl".to_string(),
                    "append".to_string(),
                    1,
                    1,
                    HashMap::new(),
                ));
                register::<CreateSnapshotEvent>(Arc::new(Reentrant2));
            }
        }
        struct Reentrant2;
        impl Listener<CreateSnapshotEvent> for Reentrant2 {
            fn notify(&self, _event: &CreateSnapshotEvent) {}
        }

        register::<ScanEvent>(Arc::new(Reentrant));
        notify_all(&ScanEvent::new(
            "db.tbl".to_string(),
            1,
            Predicate::AlwaysTrue,
            test_schema(),
        ));
        // Reaching here without hanging is the assertion.
    }

    /// Risk: the event accessors silently reorder or drop a field, so a listener reads a
    /// different value than was constructed (Java-exact arg order is load-bearing). Pins every
    /// field of all three event types round-trips through `new` → accessor.
    #[test]
    fn test_event_accessors_round_trip() {
        let schema = test_schema();

        let scan = ScanEvent::new(
            "db.t".to_string(),
            9,
            Predicate::AlwaysFalse,
            schema.clone(),
        );
        assert_eq!(scan.table_name(), "db.t");
        assert_eq!(scan.snapshot_id(), 9);
        assert_eq!(scan.filter(), &Predicate::AlwaysFalse);
        assert_eq!(scan.projection(), &schema);

        let inc = IncrementalScanEvent::new(
            "db.t".to_string(),
            3,
            8,
            Predicate::AlwaysTrue,
            schema.clone(),
            true,
        );
        assert_eq!(inc.table_name(), "db.t");
        assert_eq!(inc.from_snapshot_id(), 3);
        assert_eq!(inc.to_snapshot_id(), 8);
        assert_eq!(inc.filter(), &Predicate::AlwaysTrue);
        assert_eq!(inc.projection(), &schema);
        assert!(inc.is_from_snapshot_inclusive());

        let mut summary = HashMap::new();
        summary.insert("added-data-files".to_string(), "1".to_string());
        let create = CreateSnapshotEvent::new(
            "db.t".to_string(),
            "append".to_string(),
            42,
            5,
            summary.clone(),
        );
        assert_eq!(create.table_name(), "db.t");
        assert_eq!(create.operation(), "append");
        assert_eq!(create.snapshot_id(), 42);
        assert_eq!(create.sequence_number(), 5);
        assert_eq!(create.summary(), &summary);
    }
}
