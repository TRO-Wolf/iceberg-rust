<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# F-GLUE-FAULT-SEAM-1 ledger — Glue commit-response fault seam outside `cfg(test)`, feature-gated

Brief: devin-worker lane F-GLUE-FAULT-SEAM-1, first step of RePark card C-1 /
ICE-AWS-UNKNOWN-DRILL-1. Branch `feat/f-glue-fault-seam-1` off fork `main`.
Consumer: RePark's AWS acceptance suite runs a normal (non-test) build and must drive a Glue
commit whose `UpdateTable` the service applies while the response is dropped, then observe the
typed unknown outcome carrying the RePark operation id.
No AWS credentials, network, `aws` CLI, or credentialed test path was used at any point;
`credentialed_requested()` stays false — every test here is offline over the memory `FileIO`
plus the scripted inner transport.

## 1. Reading (step 1)

- **Transport install.** `GlueCatalog::new` (`crates/catalog/glue/src/catalog.rs`) builds one
  `LiveGlueCommitTransport` (Glue client + `config.catalog_id`, attempt-counted) inside
  `Arc<dyn GlueCommitTransport>` and stores it in `commit_transport`. `update_table` builds the
  `GlueUpdateTableCall`, awaits `self.commit_transport.send_update_table(...)`, and maps the
  outcome through `map_glue_commit_send`. `catalog/replace_publish.rs::publish` shares the same
  send + map path. The existing `DiscardingGlueCommitTransport` — a wrapper that turns the
  inner transport's `Success` into `AcceptedResponseLost` — lived under `#[cfg(test)]`.
- **Reconcile decision.** `Transaction::commit` (`crates/iceberg/src/transaction/mod.rs`)
  retries only `retryable() && kind != CommitStateUnknown`; an unknown outcome is never
  resent. On `CommitStateUnknown` it calls `reconcile_unknown_commit_outcome`: when
  `latest_attempt_snapshot_ids` is empty — true for metadata-only commits, which add no
  `TableUpdate::AddSnapshot` — it returns the original error unchanged. Otherwise
  `check_commit_status_strict` reloads the catalog pointer and searches for the attempted
  snapshot ids: found → `Ok(reloaded)`; absent-after-refresh or still-unknown → the original
  error. That is how an append whose response was dropped reconciles to success with exactly
  one catalog commit attempt while a metadata-only commit stays unknown.
- **Operation-id gap (confirmed).** Before this change the unknown error named neither the
  operation id nor the attempted snapshot ids: `map_glue_commit_send` /
  `map_glue_commit_sdk_error` / `map_update_table_service_error` build
  `Error::new(CommitStateUnknown, msg).with_source(...)` — no `with_context`. The RePark
  `OPERATION_ID_PROP` stamp (`engine.operation-id`, written by DataFusion via
  `set_snapshot_properties`) lands only in `snapshot.summary().additional_properties` or table
  properties and never reached the error.

## 2. Design (step 2)

- Cargo feature **`commit-fault-injection`** on `iceberg-catalog-glue`, OFF by default.
- Catalog property **`glue.fault.drop-update-table-response=<n>`**
  (`GLUE_CATALOG_PROP_DROP_UPDATE_TABLE_RESPONSE`, re-exported at the crate root): drop the
  response of the next `n` successful `UpdateTable` calls after the service applied them.
- One shared property path: `fault_drop_count(props)` parses the value (under the feature) or
  refuses with typed `ErrorKind::FeatureUnsupported` naming the property and the feature
  (without it); a non-integer value is `ErrorKind::DataInvalid` naming the property. Absent →
  `Ok(None)`. `build_commit_transport` is the production entry `GlueCatalog::new` calls;
  `build_commit_transport_parts` is the `cfg(any(test, feature))` entry that also hands back
  the installed wrapper for assertion.
- `DiscardingGlueCommitTransport` moved to `#[cfg(any(test, feature =
  "commit-fault-injection"))]` and gained a `drop_remaining: AtomicU64` budget claimed via
  `fetch_update(..., checked_sub)` — only `Success` outcomes are rewritten to
  `AcceptedResponseLost`; every other outcome passes through. `new(inner, drop_count)`;
  existing test callers pass `u64::MAX`.
- Operation-id context: `commit_send_operation_ids(base, staged)` diffs the staged metadata
  against the commit base — new snapshots' `summary().additional_properties` plus a changed
  `engine.operation-id` table property — and `map_glue_commit_send_identified` folds them into
  `with_context("engine.operation-id", id)` when (and only when) the outcome kind is
  `CommitStateUnknown`. `update_table` uses it for every send; `replace_publish::publish` uses
  `published_metadata_operation_ids` (current snapshot + properties of the published metadata),
  so every path that can end unknown carries the stamp. The `&'static str` context key is the
  smallest mechanism the fork's `Error` allows.
- Test seam: `GlueCatalog::for_commit_fault_tests_at_version` (`#[cfg(test)]`) runs
  `build_commit_transport_parts` — the same property path `new` uses — over a caller-supplied
  inner transport and returns `(catalog, Option<Arc<DiscardingGlueCommitTransport>>)`, so the
  scripted transport sits underneath the property-installed wrapper.
- `resolve_file_io_props` moved verbatim to `utils.rs` to keep `catalog.rs` under its legacy
  size ceiling; the ratchet row in `scripts/check_rust_file_size.py` is lowered 1048 → 1024.

## 3. Red cells (step 3, commit `f4e5e1e6`)

With the property path present but `fault_drop_count` stubbed to `Ok(None)`:

- `cargo test -p iceberg-catalog-glue --lib` → `test result: FAILED. 50 passed; 1 failed`
  — `commit_fault_tests::disabled::fault_property_refuses_at_construction_without_the_feature`
  panics on `expect_err` (catalog built instead of refusing).
- `cargo test -p iceberg-catalog-glue --lib --features commit-fault-injection` →
  `test result: FAILED. 51 passed; 4 failed` —
  `dropped_append_response_reconciles_with_one_catalog_commit`,
  `dropped_metadata_only_commit_stays_unknown_and_names_the_operation_id`,
  `drop_count_two_discards_two_responses_then_the_third_passes` (all red at
  `fault.expect("the property installs the response-dropping wrapper")`), and
  `fault_property_rejects_a_non_integer_count` (red at `expect_err`).
  `unset_fault_property_leaves_commit_behavior_unchanged` was already green — absence of the
  property is genuinely unchanged behavior.

## 4. Green (step 3, commit `186e7f2b`)

- `cargo test -p iceberg-catalog-glue --lib` → `test result: ok. 51 passed; 0 failed`
- `cargo test -p iceberg-catalog-glue --lib --features commit-fault-injection` →
  `test result: ok. 55 passed; 0 failed`
- `cargo clippy -p iceberg-catalog-glue --all-targets -- -D warnings` → clean
- `cargo clippy -p iceberg-catalog-glue --all-targets --features commit-fault-injection -- -D
  warnings` → clean
- `python3 scripts/check_rust_file_size.py` → `rust-file-size: 513 files clean (96 legacy
  ceilings)`
- `cargo fmt --all` → clean

## 5. Mutation proof (step 4; reverts not committed)

| Mutation | Command | Result |
|---|---|---|
| M1 — `build_commit_transport_parts` never installs the wrapper (`let fault = None`) | `cargo test -p iceberg-catalog-glue --lib --features commit-fault-injection` | `FAILED. 52 passed; 3 failed`: `dropped_append_response_reconciles_with_one_catalog_commit`, `dropped_metadata_only_commit_stays_unknown_and_names_the_operation_id`, `drop_count_two_discards_two_responses_then_the_third_passes`. `fault_property_rejects_a_non_integer_count` and `unset_fault_property_leaves_commit_behavior_unchanged` correctly stayed green (parse path untouched). |
| M2 — `fault_drop_count` `not(feature)` arm returns `Ok(None)` (refusal removed) | `cargo test -p iceberg-catalog-glue --lib` | `FAILED. 50 passed; 1 failed`: `fault_property_refuses_at_construction_without_the_feature`. |
| Restore | both commands above | `ok. 51 passed` / `ok. 55 passed` — no reverts committed (`git status` clean before the docs commit). |

## 6. Names the next step consumes

- Feature: `commit-fault-injection` (off by default) on `iceberg-catalog-glue`.
- Property: `glue.fault.drop-update-table-response=<n>` — drops the next `n` successful
  `UpdateTable` responses after service-side apply; `n=0` installs the wrapper with an empty
  budget (no drops), a non-integer refuses `DataInvalid`, and the property without the feature
  refuses `FeatureUnsupported` naming both.
- Operation-id key in the unknown error context: `engine.operation-id`
  (`GLUE_COMMIT_OPERATION_ID_PROP`), attached once per stamped operation id found in the
  commit's new snapshots / changed properties.
- RePark acceptance flow this enables: build the catalog with the feature + property → real
  `UpdateTable` lands while the response is dropped → metadata-only commits surface typed
  `CommitStateUnknown` (non-retryable) carrying the operation id; snapshot commits reconcile to
  success after reload with exactly one catalog commit attempt.
