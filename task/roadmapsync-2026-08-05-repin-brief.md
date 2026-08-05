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

# RoadMapSync — fork repin brief (2026-08-05)

**Audience:** RePark (and any other consumer pinning the fork).
**Current consumer pin:** `b009ac15` — the tip of the now-deleted pre-recut `chore/df54-family-bump`
branch. Preserved as tag **`archive/df54-family-bump-b009ac15`** so the pin stays resolvable; the tag
is retired once you repin.
**Repin target:** `403625eb` (#188), current `main`.

## Read this first

`b009ac15` predates **BUG-001** (#184). On that commit, a fork-written v2 position delete under an
evolved partition spec commits successfully, attaches to nothing, and **deleted rows reappear on the
next scan**. Your repin is what delivers the fix — this is not routine hygiene.

Separately: **this brief is late.** Unit 3's coordination duty was to warn you *before* your next
repin, and the surfaces below have been accumulating across four merges since #182 without an
announcement. Nothing here is a surprise we discovered late; it is a message we owed earlier.

## Breaking — source changes required

| # | Surface | Was → Now |
|---|---|---|
| #182 | `PartitionKey::new` | infallible → **`Result`**. Validates arity, spec/schema binding (absent source column id), and per-value type compatibility. Java parity: an invalid tuple is unrepresentable (`StructTransform.java:49/63`, throws on missing accessor `:57-58`). 58 call sites moved in-tree; expect the same at your call sites, including any inside `rust,no_run` doc fences — those still compile, and missing one fails with an unrelated-looking error. |
| #183 | `FileScanTask` fields | `data_file_path: String → Arc<str>`; `project_field_ids: Vec<i32> → Arc<[i32]>`; `predicate: Option<BoundPredicate> → Option<Arc<BoundPredicate>>`; `deletes: Vec<..> → Arc<[FileScanTaskDeleteFile]>`. **Serde wire shape is unchanged and byte-identical** — this is a construction/field-access break only. Your Group T/Y harness fixtures construct these directly. |
| #183 | `DeleteFilter::deleted_row_positions` | `Arc<Mutex<DeleteVector>>` → **`Option<Arc<DeleteVector>>`** (vectors are frozen after load; the lock is gone). |
| #187 | **MSRV** | **1.92 → 1.94.** Toolchain pin moves to `nightly-2026-03-05` (lint gate only; downstream needs 1.94). |
| #187 | Dependency family | datafusion 52.2 → **54.1.0**, arrow 57.3 → **58.4**, parquet 57.3 → **58.4**, `orc-rust` 0.7 → **0.8**, `sqllogictest` 0.28.3 → 0.29. If you pin any of these yourself they must move together — arrow/parquet 57 and 58 `RecordBatch` types do not interoperate. |

## Format-visible — written output changes

| # | Surface | Detail |
|---|---|---|
| #182 | **Partition-path rendering for binary/fixed values** | Segments move from UPPERCASE hex to **standard Base64** (`TransformUtil.base64encode` = `java.util.Base64.getEncoder()`, **not** URL-safe), then URL-escaped — e.g. bytes `61 2F 62` → `YS9i`; `FB FF` → `+/8=` → `%2B%2F8%3D`. D8-approved. **Existing tables stay readable** — readers resolve every data file from the manifest's recorded `file_path`, never by re-deriving the partition directory. `Display for Datum` still renders UPPERCASE hex (orthogonal surface, unchanged). |
| #184 | Position-delete `file_path` bounds | Now **full** instead of 64-byte-truncated (parquet-rs default truncation was dropping them as non-exact, so the read side's equal-bounds routing never recognised fork-written deletes). Larger footer stats, bounded by the few distinct paths per delete file. |
| #184 | Partition stamps | Data and position files now stamp the **true `default_spec_id`** rather than a fabricated `0`. Post-`DROP PARTITION FIELD` that id is non-zero. |

**Correctness direction is one-way for #184:** deletes that previously attached still attach; deletes
that previously missed now attach.

## Behavior — no signature change

- **#182 `CurrentFileStatus`** — the three post-`close()` accessors returned by panicking on a closed
  writer; they now return `""` / `0` / `0`. Closed non-breakingly, so the API break predicted for
  this item did not occur.

## Additive — opt-in, no action needed

- **#183** `TableMetadataCache` at `MemoryCatalog::load_table` (default **OFF**, fail-closed on
  location equality); FileIO property `client.list-stat-concurrency` (default 16).
- **#184** `position_delete_writer_properties()`.

## Evidence at the repin target

`make interop` **52/52** (floor 52), run independently twice on the #187 tree; iceberg lib **3137**;
iceberg-datafusion 177/73/12/4/2; sqllogictest 9/9; clippy `-D warnings`; `--no-default-features`
green; GAP_MATRIX anchors 75 rows. **No on-disk encoding change from the dependency bump** — every
encoding-relevant parquet writer default is byte-identical between 57.3.1 and 58.4.0 (the only
changed default governs bloom filters, which this repo never enables).

## After you repin

Tell us, and we retire `archive/df54-family-bump-b009ac15`. Until then it stays so your current pin
does not become unreachable.
