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

//! Java behavioral-conformance interop for `LockManager` / `InMemoryLockManager` (GAP_MATRIX row R129).
//!
//! `InMemoryLockManager` is an IN-PROCESS primitive with no on-disk artifact, so "interop" here is
//! OUTCOME-CONFORMANCE rather than a byte round-trip: both impls run the IDENTICAL deterministic
//! acquire/release sequence and their observable boolean outcomes must agree. The Java oracle
//! (`generate-interop-lock`, [`InteropOracle$LockOracle`]) drives the REAL Java
//! `org.apache.iceberg.util.LockManagers` default `InMemoryLockManager` (reconfigured to small
//! acquire-timeouts) through the same 7 steps and emits `java_lock_outcomes.json`.
//!
//! THREE INDEPENDENT ANCHORS (anti-circular):
//!   1. Java self-checks its outcomes against a hand-declared expected (`generate-interop-lock: 0
//!      failures`, fail-closed in the harness).
//!   2. This test asserts the Rust outcomes equal the SAME hand-declared expected, declared here
//!      independently — neither side echoes the other.
//!   3. This test asserts the Rust outcomes equal the Java-emitted outcomes step-for-step.
//!
//! NO BEHAVIORAL DIVERGENCE (acquire-while-held): Java `InMemoryLockManager.acquire` internally catches
//! the `IllegalStateException` that `Tasks.throwFailureWhenFinished()` raises on acquire-timeout
//! exhaustion and RETURNS `false` (1.10.0 bytecode: exception table `[0,52)->53` => `iconst_0`/`ireturn`)
//! — it does NOT throw to the caller. Rust `acquire` likewise returns `false` on timeout, so both sides
//! return the identical observable boolean and the conformance is exact.
//!
//! Env-gated: without `ICEBERG_INTEROP_LOCK_DIR` (set by `dev/java-interop/run-interop-lock.sh`) the
//! test returns early, so the offline `cargo test` gate is unaffected.

use std::collections::HashMap;
use std::path::PathBuf;

use iceberg::{InMemoryLockManager, LockManager};

/// The fixed property knobs — MUST match the Java oracle's `LockOracle.props()` (same keys ⇒ same
/// config). Keys are the `catalog/lock.rs` `LOCK_*` constants, verified identical to Java
/// `CatalogProperties`.
fn props() -> HashMap<String, String> {
    let mut p = HashMap::new();
    // small ⇒ "acquire while held" fails fast (vs the 180_000 ms default)
    p.insert("lock.acquire-timeout-ms".to_string(), "100".to_string());
    p.insert("lock.acquire-interval-ms".to_string(), "10".to_string());
    // large ⇒ a held lock will NOT expire mid-test, heartbeat will NOT fire mid-test
    p.insert("lock.heartbeat-timeout-ms".to_string(), "60000".to_string());
    p.insert("lock.heartbeat-interval-ms".to_string(), "5000".to_string());
    p.insert("lock.heartbeat-threads".to_string(), "1".to_string());
    p
}

#[test]
fn test_lock_conformance_interop() {
    let Ok(dir) = std::env::var("ICEBERG_INTEROP_LOCK_DIR") else {
        // Offline / no Java oracle present — clean skip (keeps the offline gate green).
        return;
    };
    let dir = PathBuf::from(dir);

    let lm = InMemoryLockManager::new(&props()).expect("build the Rust InMemoryLockManager");
    let entity = "tableX";
    let ghost = "ghostEntity";
    let owner1 = "owner-alpha";
    let owner2 = "owner-beta";

    // The deterministic 7-step sequence — byte-for-byte the same ops/entities/owners as the Java oracle.
    let outcomes = [
        lm.acquire(entity, owner1), // 0: free            -> true
        lm.acquire(entity, owner2), // 1: held by owner1  -> false (both return false on acquire-timeout)
        lm.release(entity, owner2), // 2: wrong owner     -> false
        lm.release(ghost, owner1),  // 3: never locked    -> false
        lm.release(entity, owner1), // 4: owner           -> true
        lm.acquire(entity, owner2), // 5: now free        -> true
        lm.release(entity, owner2), // 6: owner           -> true
    ];

    // Anchor #2: hand-declared expected, independent of Java.
    let expected = [true, false, false, false, true, true, true];
    assert_eq!(
        outcomes, expected,
        "Rust InMemoryLockManager outcomes diverged from the hand-declared contract"
    );

    // Anchor #3: cross-impl equality vs the REAL Java InMemoryLockManager outcomes.
    let raw = std::fs::read_to_string(dir.join("java_lock_outcomes.json"))
        .unwrap_or_else(|e| panic!("read java_lock_outcomes.json: {e}"));
    let java: serde_json::Value =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse java_lock_outcomes.json: {e}"));
    let steps = java["steps"]
        .as_array()
        .expect("java_lock_outcomes.json must have a `steps` array");
    assert_eq!(
        steps.len(),
        outcomes.len(),
        "Java emitted {} steps, Rust ran {}",
        steps.len(),
        outcomes.len()
    );
    for (i, (rust, step)) in outcomes.iter().zip(steps).enumerate() {
        let java_result = step["result"]
            .as_bool()
            .unwrap_or_else(|| panic!("step {i}: java `result` is not a bool"));
        assert_eq!(
            *rust, java_result,
            "step {i}: Rust outcome {rust} != Java outcome {java_result}"
        );
    }
}
