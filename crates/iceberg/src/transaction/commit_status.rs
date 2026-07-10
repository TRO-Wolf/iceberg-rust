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

//! Reconciliation-by-refresh for ambiguous commit outcomes (GAP_MATRIX row R157).
//!
//! Port of Java `BaseMetastoreOperations.checkCommitStatus` / `checkCommitStatusStrict`
//! (`core/src/main/java/org/apache/iceberg/BaseMetastoreOperations.java`, iceberg 1.10.0) and its
//! `CommitStatus` enum (`SUCCESS` / `FAILURE` / `UNKNOWN`). When a commit fails with
//! [`ErrorKind::CommitStateUnknown`](crate::ErrorKind::CommitStateUnknown) — the request may have
//! durably landed but the response was lost — the library re-reads the catalog with bounded
//! retries and decides whether the attempted commit is present.
//!
//! ## The Java contract, decoded from source + 1.10.0 bytecode
//!
//! - **What is compared.** Java checks whether the attempted `newMetadataLocation` is the
//!   refreshed table's CURRENT metadata location **or appears in `previousFiles()`** — the
//!   metadata-log history — "on the chance that a second committer was able to successfully
//!   commit on top of our commit" (`BaseMetastoreTableOperations.checkCurrentMetadataLocation`,
//!   L334-341). This fork's seam is catalog-agnostic ([`Transaction::commit`]
//!   (crate::transaction::Transaction::commit)), where the metadata location is assigned by the
//!   catalog and never known to the client on the failed attempt — so the equivalent evidence is
//!   the attempted **snapshot ids** (client-generated before `update_table`): the commit landed
//!   iff any attempted snapshot id is present in the reloaded metadata's snapshot set, which —
//!   like Java's `previousFiles()` walk — still finds a commit BURIED under later third-party
//!   commits. Corollary: a commit that adds NO snapshot (property/schema-only) carries no such
//!   evidence and is not reconciled (a named divergence — Java's location check covers those).
//! - **The bounded retry schedule** (`checkCommitStatusStrict` bytecode, iceberg-core-1.10.0.jar):
//!   `Tasks.foreach(..).retry(maxAttempts)` where `maxAttempts` is the
//!   `commit.status-check.num-retries` property (default 3, offsets 0-7) — and
//!   `Tasks.Builder.retry(n)` sets `maxAttempts = n + 1` (`core/util/Tasks.java` L163), so the
//!   default budget is **4 refresh attempts**; `.exponentialBackoff(minWaitMs, maxWaitMs,
//!   totalRetryMs, 2.0)` (offsets 73-82: `ldc2_w 2.0d`) with the delay formula
//!   `min(minSleepTimeMs * 2.0^(attempt-1), maxSleepTimeMs)` and the stop rule
//!   `attempt >= maxAttempts || (durationMs > maxDurationMs && attempt > 1)`
//!   (`core/util/Tasks.java` L418, L452-455); `.suppressFailureWhenFinished()` (offset 70) so an
//!   exhausted budget yields `UNKNOWN`, never a thrown refresh error. Java's 0..10% sleep jitter
//!   (`Tasks.java` L456) is deliberately omitted here (deterministic waits; the jitter only
//!   de-synchronizes herds and carries no contract).
//! - **A refresh that SUCCEEDS decides immediately.** `Tasks` retries only *thrown* failures; a
//!   supplier that returns `false` completes the task, leaving `FAILURE` after ONE successful
//!   read (`lambda$checkCommitStatusStrict$1`). The retries exist for the *refresh itself*
//!   failing, not for polling an absent commit into existence.
//! - **Strict vs non-strict.** `checkCommitStatusStrict` returns `FAILURE` when the location is
//!   not found. The NON-strict `checkCommitStatus` — the ONLY variant with production callers in
//!   1.10.0 (`GlueTableOperations.doCommit` L174, `DynamoDbTableOperations` L136; `grep -rn
//!   checkCommitStatusStrict` finds zero non-test callers) — **converts `FAILURE` to `UNKNOWN`**
//!   (`BaseMetastoreOperations.java` L71-78; bytecode offsets 11-34: `if_acmpne` on `FAILURE`,
//!   `areturn UNKNOWN`), "because possible pending retries might still commit the change": a
//!   timed-out request may still be executing server-side and land AFTER the check, so declaring
//!   failure and letting the caller re-apply is the double-commit corruption class. This module
//!   exposes the STRICT classifier; the caller in `Transaction::commit` applies the non-strict
//!   conversion (surfacing the original `CommitStateUnknown`) exactly as Java production does.
//! - **What each outcome does** (`GlueTableOperations.doCommit` L179-191): `SUCCESS` ⇒ the
//!   persist failure is swallowed and the commit stands; `UNKNOWN` ⇒
//!   `CommitStateUnknownException(persistFailure)` — the ORIGINAL failure is the cause. A
//!   definite `CommitFailedException` is rethrown at L162-163 WITHOUT ever invoking
//!   reconciliation.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::error::{Error, ErrorKind, Result};
use crate::table::Table;
use crate::{Catalog, TableIdent};

/// Property key for the number of status-check retries after an ambiguous commit (Java
/// `TableProperties.COMMIT_NUM_STATUS_CHECKS`, `core/TableProperties.java` L98). The refresh is
/// attempted `n + 1` times (Java `Tasks.Builder.retry(n)` semantics, `core/util/Tasks.java` L163).
pub(crate) const COMMIT_NUM_STATUS_CHECKS: &str = "commit.status-check.num-retries";
/// Default status-check retries: 3 (Java `COMMIT_NUM_STATUS_CHECKS_DEFAULT`, bytecode-verified
/// vs iceberg-core-1.10.0.jar: `iconst_3` at `checkCommitStatusStrict` offset 3).
pub(crate) const COMMIT_NUM_STATUS_CHECKS_DEFAULT: u32 = 3;

/// Property key for the minimum wait (ms) between status-check attempts (Java
/// `TableProperties.COMMIT_STATUS_CHECKS_MIN_WAIT_MS`, `core/TableProperties.java` L101).
pub(crate) const COMMIT_STATUS_CHECKS_MIN_WAIT_MS: &str = "commit.status-check.min-wait-ms";
/// Default minimum status-check wait: 1 s (Java `COMMIT_STATUS_CHECKS_MIN_WAIT_MS_DEFAULT`,
/// bytecode-verified: `ldc2_w 1000l` at offset 12).
pub(crate) const COMMIT_STATUS_CHECKS_MIN_WAIT_MS_DEFAULT: u64 = 1000;

/// Property key for the maximum wait (ms) between status-check attempts (Java
/// `TableProperties.COMMIT_STATUS_CHECKS_MAX_WAIT_MS`, `core/TableProperties.java` L104).
pub(crate) const COMMIT_STATUS_CHECKS_MAX_WAIT_MS: &str = "commit.status-check.max-wait-ms";
/// Default maximum status-check wait: 1 min (Java `COMMIT_STATUS_CHECKS_MAX_WAIT_MS_DEFAULT`,
/// bytecode-verified: `ldc2_w 60000l` at offset 23).
pub(crate) const COMMIT_STATUS_CHECKS_MAX_WAIT_MS_DEFAULT: u64 = 60 * 1000;

/// Property key for the total status-check timeout (ms) (Java
/// `TableProperties.COMMIT_STATUS_CHECKS_TOTAL_WAIT_MS`, `core/TableProperties.java` L107-108).
pub(crate) const COMMIT_STATUS_CHECKS_TOTAL_WAIT_MS: &str = "commit.status-check.total-timeout-ms";
/// Default total status-check timeout: 30 min (Java `COMMIT_STATUS_CHECKS_TOTAL_WAIT_MS_DEFAULT`,
/// bytecode-verified: `ldc2_w 1800000l` at offset 34).
pub(crate) const COMMIT_STATUS_CHECKS_TOTAL_WAIT_MS_DEFAULT: u64 = 30 * 60 * 1000;

/// The `commit.status-check.*` knobs controlling reconciliation-by-refresh, parsed from table
/// properties with Java's names and defaults (`BaseMetastoreOperations.checkCommitStatusStrict`
/// reads them via `PropertyUtil` from the committed metadata's properties).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct StatusCheckConfig {
    /// `commit.status-check.num-retries`: the refresh runs `num_retries + 1` times at most.
    pub(crate) num_retries: u32,
    /// `commit.status-check.min-wait-ms`: first-retry delay and backoff base.
    pub(crate) min_wait_ms: u64,
    /// `commit.status-check.max-wait-ms`: per-delay clamp.
    pub(crate) max_wait_ms: u64,
    /// `commit.status-check.total-timeout-ms`: elapsed-time stop rule (checked after attempt 1,
    /// Java `Tasks.java` L418).
    pub(crate) total_timeout_ms: u64,
}

impl StatusCheckConfig {
    /// Parse the four `commit.status-check.*` properties, falling back to Java's defaults for
    /// absent keys. An unparsable value is an error (Java's `PropertyUtil.propertyAsInt/Long`
    /// throws `NumberFormatException`); the caller decides how a parse failure interacts with the
    /// in-flight unknown outcome.
    pub(crate) fn from_properties(props: &HashMap<String, String>) -> Result<Self> {
        Ok(StatusCheckConfig {
            num_retries: parse_status_check_property(
                props,
                COMMIT_NUM_STATUS_CHECKS,
                COMMIT_NUM_STATUS_CHECKS_DEFAULT,
            )?,
            min_wait_ms: parse_status_check_property(
                props,
                COMMIT_STATUS_CHECKS_MIN_WAIT_MS,
                COMMIT_STATUS_CHECKS_MIN_WAIT_MS_DEFAULT,
            )?,
            max_wait_ms: parse_status_check_property(
                props,
                COMMIT_STATUS_CHECKS_MAX_WAIT_MS,
                COMMIT_STATUS_CHECKS_MAX_WAIT_MS_DEFAULT,
            )?,
            total_timeout_ms: parse_status_check_property(
                props,
                COMMIT_STATUS_CHECKS_TOTAL_WAIT_MS,
                COMMIT_STATUS_CHECKS_TOTAL_WAIT_MS_DEFAULT,
            )?,
        })
    }
}

fn parse_status_check_property<T: std::str::FromStr>(
    props: &HashMap<String, String>,
    key: &str,
    default: T,
) -> Result<T>
where
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    props.get(key).map_or(Ok(default), |value| {
        value.parse::<T>().map_err(|parse_error| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Invalid value for {key}: {parse_error}"),
            )
        })
    })
}

/// Outcome of a reconciliation-by-refresh check — the Rust `BaseMetastoreOperations.CommitStatus`
/// (`SUCCESS` / `FAILURE` / `UNKNOWN`). `Success` carries the reloaded table (the state that
/// PROVES the commit landed) so the caller can hand it back without another read.
#[derive(Debug)]
pub(crate) enum CommitStatus {
    /// The attempted commit is present in the reloaded metadata — it landed.
    Success(Box<Table>),
    /// A refresh SUCCEEDED and the attempted commit is absent. STRICT classification only: Java's
    /// non-strict production path converts this to `Unknown` before acting on it
    /// (`BaseMetastoreOperations.checkCommitStatus` L71-78 / bytecode offsets 11-34), because a
    /// still-in-flight request may land AFTER the check.
    Failure,
    /// Every refresh attempt failed within the bounded budget — the state remains unknown.
    Unknown,
}

/// Java `Tasks.runTaskWithRetry`'s backoff delay (`core/util/Tasks.java` L452-455):
/// `min(minSleepTimeMs * 2.0^(attempt-1), maxSleepTimeMs)`, with the scale factor 2.0 pinned by
/// `checkCommitStatusStrict` bytecode (offset 79: `ldc2_w 2.0d`). Jitter omitted (module docs).
fn status_check_delay_ms(min_wait_ms: u64, max_wait_ms: u64, attempt: u32) -> u64 {
    // Clamp the exponent so the shift cannot overflow; saturating_mul absorbs the rest. For any
    // real schedule the min(., max_wait) clamp dominates long before either saturates.
    let exponent = attempt.saturating_sub(1).min(63);
    min_wait_ms
        .saturating_mul(1u64 << exponent)
        .min(max_wait_ms)
}

/// Port of Java `BaseMetastoreOperations.checkCommitStatusStrict` (1.10.0): re-read the catalog
/// with bounded retries and classify the attempted commit as landed / absent / still-unknown.
///
/// - A refresh that **succeeds decides immediately**: `Success` if any attempted snapshot id is
///   in the reloaded metadata's snapshot set (current-pointer moves by OTHER writers do not hide
///   it — the set is the history walk, mirroring Java's `previousFiles()` search), else `Failure`.
/// - A refresh that **fails** is retried up to `num_retries + 1` total attempts with
///   exponential backoff (factor 2.0, clamped at `max_wait_ms`), stopping early once
///   `total_timeout_ms` has elapsed (checked after the first attempt, Java `Tasks.java` L418);
///   exhaustion yields `Unknown` (Java `suppressFailureWhenFinished`).
///
/// `attempted_snapshot_ids` must be non-empty — the caller skips reconciliation entirely for
/// commits without snapshot evidence (module docs).
pub(crate) async fn check_commit_status_strict(
    catalog: &dyn Catalog,
    table_ident: &TableIdent,
    attempted_snapshot_ids: &[i64],
    config: &StatusCheckConfig,
) -> CommitStatus {
    // Java `Tasks.Builder.retry(n)`: maxAttempts = n + 1 (`core/util/Tasks.java` L163).
    let max_attempts = config.num_retries.saturating_add(1);
    let total_timeout = Duration::from_millis(config.total_timeout_ms);
    let started = Instant::now();
    let mut attempt: u32 = 0;

    loop {
        attempt += 1;
        match catalog.load_table(table_ident).await {
            Ok(reloaded) => {
                // One SUCCESSFUL refresh decides: Java's Tasks loop retries only thrown
                // failures — a supplier returning false completes with FAILURE after one read.
                let landed = attempted_snapshot_ids
                    .iter()
                    .any(|id| reloaded.metadata().snapshot_by_id(*id).is_some());
                return if landed {
                    tracing::info!(
                        table = %table_ident,
                        "commit status check: the ambiguous commit is present in the reloaded \
                         metadata — it succeeded"
                    );
                    CommitStatus::Success(Box::new(reloaded))
                } else {
                    CommitStatus::Failure
                };
            }
            Err(check_error) => {
                // Java `onFailure`: LOG.error("Cannot check if commit to {} exists.", ...).
                tracing::error!(
                    table = %table_ident,
                    ?check_error,
                    "cannot check if the ambiguous commit exists"
                );
                if attempt >= max_attempts || (started.elapsed() > total_timeout && attempt > 1) {
                    // Java: "Cannot determine commit state to {}. Failed during checking {}
                    // times. Treating commit state as unknown." (suppressFailureWhenFinished).
                    tracing::error!(
                        table = %table_ident,
                        attempts = attempt,
                        "cannot determine the commit state after the bounded status checks; \
                         treating the commit state as unknown"
                    );
                    return CommitStatus::Unknown;
                }
                tokio::time::sleep(Duration::from_millis(status_check_delay_ms(
                    config.min_wait_ms,
                    config.max_wait_ms,
                    attempt,
                )))
                .await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;
    use crate::catalog::MockCatalog;
    use crate::transaction::tests::make_v2_table;

    /// Risk: a renamed knob or a drifted default silently detaches the reconciliation schedule
    /// from Java's (`commit.status-check.*`, TableProperties.java L98-110 + 1.10.0 bytecode
    /// constants) — tables tuned for Java behave differently here.
    #[test]
    fn test_status_check_property_names_and_java_defaults() {
        assert_eq!(COMMIT_NUM_STATUS_CHECKS, "commit.status-check.num-retries");
        assert_eq!(
            COMMIT_STATUS_CHECKS_MIN_WAIT_MS,
            "commit.status-check.min-wait-ms"
        );
        assert_eq!(
            COMMIT_STATUS_CHECKS_MAX_WAIT_MS,
            "commit.status-check.max-wait-ms"
        );
        assert_eq!(
            COMMIT_STATUS_CHECKS_TOTAL_WAIT_MS,
            "commit.status-check.total-timeout-ms"
        );

        let defaults = StatusCheckConfig::from_properties(&HashMap::new())
            .expect("defaults must always parse");
        assert_eq!(defaults, StatusCheckConfig {
            num_retries: 3,
            min_wait_ms: 1000,
            max_wait_ms: 60 * 1000,
            total_timeout_ms: 30 * 60 * 1000,
        });
    }

    /// Risk: the parser reads the wrong key (or ignores the properties entirely) so operator
    /// tuning of the status-check schedule is silently discarded.
    #[test]
    fn test_status_check_properties_parsed_from_table_properties() {
        let props = HashMap::from([
            (
                "commit.status-check.num-retries".to_string(),
                "7".to_string(),
            ),
            (
                "commit.status-check.min-wait-ms".to_string(),
                "25".to_string(),
            ),
            (
                "commit.status-check.max-wait-ms".to_string(),
                "50".to_string(),
            ),
            (
                "commit.status-check.total-timeout-ms".to_string(),
                "5000".to_string(),
            ),
        ]);
        let config = StatusCheckConfig::from_properties(&props).expect("valid values must parse");
        assert_eq!(config, StatusCheckConfig {
            num_retries: 7,
            min_wait_ms: 25,
            max_wait_ms: 50,
            total_timeout_ms: 5000,
        });
    }

    /// Risk: an unparsable knob is silently swallowed into a default, hiding an operator typo
    /// (Java's `PropertyUtil` throws `NumberFormatException` — the error must be visible).
    #[test]
    fn test_invalid_status_check_property_is_an_error() {
        let props = HashMap::from([(
            "commit.status-check.num-retries".to_string(),
            "abc".to_string(),
        )]);
        let error = StatusCheckConfig::from_properties(&props)
            .expect_err("an unparsable knob must be an error, not a silent default");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error
                .message()
                .contains("Invalid value for commit.status-check.num-retries"),
            "the offending key must be named: {}",
            error.message()
        );
    }

    /// Risk: the backoff drops Java's 2.0 factor or the max-wait clamp (Tasks.java L452-455 —
    /// `min(min * 2^(attempt-1), max)`), hammering the catalog or stalling reconciliation. The
    /// third attempt sits exactly ON the clamp boundary (100 * 2^2 = 400 = max), distinguishing
    /// `min(...)` from an off-by-one clamp.
    #[test]
    fn test_status_check_backoff_doubles_and_clamps_at_max_wait() {
        assert_eq!(status_check_delay_ms(100, 400, 1), 100);
        assert_eq!(status_check_delay_ms(100, 400, 2), 200);
        assert_eq!(status_check_delay_ms(100, 400, 3), 400); // exactly the boundary
        assert_eq!(status_check_delay_ms(100, 400, 4), 400); // clamped
        // Degenerate large attempt must not overflow.
        assert_eq!(status_check_delay_ms(100, 400, u32::MAX), 400);
    }

    fn fast_config(num_retries: u32) -> StatusCheckConfig {
        StatusCheckConfig {
            num_retries,
            min_wait_ms: 1,
            max_wait_ms: 2,
            total_timeout_ms: 10_000,
        }
    }

    /// Risk (STRICT layer): an absent commit is classified as landed, silently converting a lost
    /// write into a reported success. The attempted id is absent from the fixture's snapshot set,
    /// so the strict classification must be `Failure` — never `Success`.
    #[tokio::test]
    async fn test_strict_check_absent_classifies_failure_never_success() {
        let mut mock_catalog = MockCatalog::new();
        mock_catalog
            .expect_load_table()
            .times(1) // a successful refresh DECIDES — a second read is a schedule bug
            .returning_st(|_| Box::pin(async move { Ok(make_v2_table()) }));

        let ident = TableIdent::from_strs(["ns1", "test1"]).expect("ident");
        let status =
            check_commit_status_strict(&mock_catalog, &ident, &[4242424242], &fast_config(3)).await;
        assert!(
            matches!(status, CommitStatus::Failure),
            "an absent commit must classify strict-FAILURE, got {status:?}"
        );
    }

    /// Risk: the landed check compares only the CURRENT pointer, so a commit buried under a later
    /// third-party commit reads as absent (Java searches `previousFiles()` history for exactly
    /// this reason). The v2 fixture's current snapshot is 3055729675574597004; the attempted id
    /// 3051729675574597004 is its PARENT — in the snapshot set but not current — and must still
    /// resolve `Success`.
    #[tokio::test]
    async fn test_strict_check_success_via_history_not_current_pointer() {
        let parent_snapshot_id = 3051729675574597004_i64;
        let current = make_v2_table()
            .metadata()
            .current_snapshot_id()
            .expect("fixture has a current snapshot");
        assert_ne!(
            current, parent_snapshot_id,
            "fixture precondition: the attempted id must NOT be the current pointer"
        );

        let mut mock_catalog = MockCatalog::new();
        mock_catalog
            .expect_load_table()
            .returning_st(|_| Box::pin(async move { Ok(make_v2_table()) }));

        let ident = TableIdent::from_strs(["ns1", "test1"]).expect("ident");
        let status = check_commit_status_strict(
            &mock_catalog,
            &ident,
            &[parent_snapshot_id],
            &fast_config(3),
        )
        .await;
        match status {
            CommitStatus::Success(reloaded) => {
                assert!(
                    reloaded
                        .metadata()
                        .snapshot_by_id(parent_snapshot_id)
                        .is_some(),
                    "the returned table must be the state that proves the commit landed"
                );
            }
            other => panic!("a commit present in history must resolve Success, got {other:?}"),
        }
    }

    /// Risk: the refresh loop is unbounded (spins forever on a dead catalog) or ignores the
    /// configured budget. With `num_retries = 2` the refresh must run exactly 2 + 1 = 3 times
    /// (Java `Tasks.Builder.retry(n)` ⇒ n+1 attempts) and then classify `Unknown`
    /// (`suppressFailureWhenFinished`).
    #[tokio::test]
    async fn test_strict_check_bounded_attempts_then_unknown() {
        let load_calls = std::sync::Arc::new(AtomicU32::new(0));
        let load_calls_in_mock = std::sync::Arc::clone(&load_calls);
        let mut mock_catalog = MockCatalog::new();
        mock_catalog.expect_load_table().returning_st(move |_| {
            load_calls_in_mock.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                Err(Error::new(
                    ErrorKind::Unexpected,
                    "refresh keeps failing during the status check",
                ))
            })
        });

        let ident = TableIdent::from_strs(["ns1", "test1"]).expect("ident");
        let status =
            check_commit_status_strict(&mock_catalog, &ident, &[4242424242], &fast_config(2)).await;
        assert!(
            matches!(status, CommitStatus::Unknown),
            "exhausted refreshes must classify Unknown, got {status:?}"
        );
        assert_eq!(
            load_calls.load(Ordering::SeqCst),
            3,
            "num_retries = 2 must budget exactly 3 refresh attempts (Java retry(n) = n+1)"
        );
    }
}
