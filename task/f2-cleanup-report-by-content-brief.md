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

# Scope brief — F-2: split `CleanupReport`'s content-file funnel by content type

**Ledger id:** `F2-CLEANUP-BY-CONTENT-2026-08-23`
**Branch:** `parity/f2-cleanup-report-by-content` (cut off `main` = `fdc8fa27`)
**Handoff item:** F-2 (P1), `~/.claude/plans/2026-08-23-opus-iceberg-rust-fork-handoff.md` §3
**Matrix row in scope:** R133 (`ExpireSnapshots`) — status cell only.

## 1. The gap

`crates/iceberg/src/transaction/expire_cleanup.rs:228` — `CleanupReport.deleted_content_files:
Vec<String>` funnels DATA, POSITION-DELETE, EQUALITY-DELETE and deletion-vector puffin files into
ONE vector. The engine consumes this to fill Spark's `expire_snapshots` result and can therefore
report only four of Spark's six columns honestly.

**Where the type is lost:** `:382` opens `content_files_to_delete: BTreeSet<String>` and `:389`
inserts `entry.file_path()` alone. The manifest entry — which carries the content type — is in
hand at that moment and is discarded. The subsequent set algebra (live-entry subtraction at `:416`,
the fail-closed `clear()` at `:436`) operates on paths only.

## 2. Oracle — Java 1.10.0, bytecode-verified 2026-08-23

Jars (Maven Central; record sha256 of each in the ledger):
`iceberg-api-1.10.0.jar`, `iceberg-spark-3.5_2.12-1.10.0.jar`.

- **`org.apache.iceberg.FileContent`** (api jar) has exactly three members: `DATA`,
  `POSITION_DELETES`, `EQUALITY_DELETES`. The fork's `DataContentType`
  (`spec/manifest/data_file.rs:359`) mirrors it 1:1 with the same ordinals (0/1/2).
- **`BaseSparkAction$ReadManifest.toFileInfo(ContentFile<?>)`** tags every content file with
  `file.content().toString()` — **`content()` ALONE. File format is never consulted.**
- **`BaseSparkAction$DeleteSummary.deletedFile(String, String)`** dispatches
  `equalsIgnoreCase` against `FileContent.DATA.name()` / `POSITION_DELETES.name()` /
  `EQUALITY_DELETES.name()`, then the literals `"Manifest"`, `"Manifest List"`,
  `"Statistics Files"`, `"Others"`; an unmatched type throws `"Illegal file type: %s"`.
- **`ExpireSnapshotsProcedure` `OUTPUT_TYPE`** = six `LongType` columns in this order:
  `deleted_data_files_count`, `deleted_position_delete_files_count`,
  `deleted_equality_delete_files_count`, `deleted_manifest_files_count`,
  `deleted_manifest_lists_count`, `deleted_statistics_files_count`.
  **Verify the nullability flag** on each `StructField.<init>(String, DataType, Z, Metadata)` and
  record it — the engine has an open nullable-vs-non-nullable divergence riding on this.

### 2a. THE DELETION-VECTOR QUESTION — answered, pin it

The handoff asked which bucket DVs land in. **Bytecode answer: `POSITION_DELETES`.** A DV is a
`DeleteFile` whose `content()` is `POSITION_DELETES`; its Puffin format is irrelevant because the
tagging reads `content()` only. **DVs are therefore counted as position-delete files and are NOT
separable from Parquet position deletes in Spark's counts.** Do NOT invent a fourth bucket.
Pin this with a test whose name says so, and cite the `toFileInfo` + `DeleteSummary` evidence.

**Note `DeleteSummary` has SEVEN counters** — the six above plus `otherFilesCount` ("Others") —
while the procedure emits SIX columns. Record what "Others" covers and whether the fork's walk can
ever produce such a file. If it cannot, say so; do not add a seventh vector speculatively.

## 3. The ask — ADDITIVE ONLY

The engine consumes `deleted_content_files` at the pinned rev. **It must still compile and behave
identically through the repin.** Keep the field; keep it populated with the union, in the same
deterministic order.

Add typed access alongside it. Choose ONE shape and justify it in the ledger:
(a) three new `Vec<String>` fields, or (b) one accessor returning paths filtered by
`DataContentType`. Prefer whichever keeps `CleanupReport`'s construction honest — note the struct
derives `Default` and is built field-by-field, so a shape that can desync the union from its parts
is the wrong shape. **State explicitly in the ledger whether the union is stored or derived.**

Classification must come from the manifest entry's content type carried through the walk — thread
it from `:389` (e.g. path → `DataContentType`) so the `:416` subtraction and the `:436` fail-closed
`clear()` keep working unchanged. **The fail-closed posture is load-bearing: when the live set
cannot be proven, NO content file may die. Do not weaken it.**

## 4. Test duty

- One test per bucket: a data file, a Parquet position delete, and an equality delete deleted by one
  expiry, asserting each lands in the right typed vector AND in the union.
- **A DV test asserting the puffin lands in POSITION deletes**, named for the claim (§2a).
- **The union must equal the concatenation of the parts** — assert it, and mutation-prove it (make
  one bucket drop a file; the union assertion must redden).
- **Preserve the existing fail-closed tests** (`test_unreadable_retained_manifest_spares_all_content_files`
  and the `:436` clear path) and add: when the retained-manifest read fails, EVERY typed vector is
  empty, not just the union.
- Determinism: paths sorted within each funnel, funnels in Java's deletion order — assert it.
- Mutation-verify each new assertion INDIVIDUALLY and report "N of M" with M named.
- Confirm the gate RUNS the new tests.

## 5. Out of scope — name, do not fix

- F-11 / R133's other remainder: `IncrementalFileCleanup`, `cleanExpiredMetadata`, `max_ref_age_ms`.
- The engine's nullable-vs-non-nullable divergence — fork side only records the Java nullability.
- Everything else in the handoff queue.

## 6. Consumed-surface duty

`CleanupReport` is named in handoff §2 as an engine-consumed surface. The ledger MUST state
explicitly whether any existing field changed name, type, or population semantics — and if the
answer is "none", say that in those words, because the engine's repin unit reads it.

## 7. Repo law

`AGENTS.md` + `CLAUDE.md` govern. No `Cargo.toml`/`Cargo.lock`. No `git add -A`. Gate chained to
the commit in ONE `&&`. `.expect(ctx)` in tests; every test asserts. Update `map.md` in lockstep.
Run `make check-matrix-anchors` after the R133 edit.
