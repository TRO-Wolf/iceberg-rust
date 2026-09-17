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

# F-EVO-SCAN-1 — unpinned scans and the DataFusion UPDATE/DELETE execs after ADD / RENAME COLUMN with no write since

**Date:** 2026-09-17. **Branch:** `fix/evo-scan-1` (stacked on `fix/ice-promote-read-1` `e8db2ac0`).
**Model:** muse-spark-1.3-contributor.
**Consumer:** RePark ICE-EVO-DML-1 residual red (33 plain-UPDATE cells).

## The defect

After `ALTER TABLE … ADD COLUMN extra` or `RENAME COLUMN w TO v` (or a name swap) with no
data write since, the current schema is newer than the current snapshot's schema.
`TableScanBuilder::build` (`crates/iceberg/src/scan/mod.rs:447`) binds the selected CURRENT
column names against the snapshot's schema, so an added or renamed-to name fails loud
(`DataInvalid => Column extra not found in table`), and a swapped name binds the other
field's id and reads the other column's values silent.

Java's contract: unpinned `table.newScan()` binds `table.schema()` (current); only
`useSnapshot(id)` binds `SnapshotUtil.schemaFor(table, id)`. Spark's DML scan does
`useSnapshot(id)…project(expectedSchema)` — the snapshot's files under the current
projection by field id. The fork's own compaction already uses that recipe (F-RDF-EVO-1,
`maintenance/rewrite_data_files_write.rs:104-116`: re-point `FileScanTask.schema` and
`project_field_ids` at the current schema; the Arrow reader NULL-fills added columns and
reads renamed columns by field id). RePark fixed its own DML scans RePark-side
(`current_schema_scan.rs`); the fork's DataFusion UPDATE/DELETE execs
(`mor_scan_stream`, `cow_scan_stream`) still pin the snapshot and select current names.

## Decisions

- **D-1** Unpinned scan (no `snapshot_id`, no `snapshot_ref`) binds the current schema.
  Pinned scan keeps the snapshot schema (time travel unchanged).
- **D-2** Pinned scans gain an opt-in projection of the current schema over the pinned
  snapshot's files (Spark `project(expectedSchema)`), used by the two DML scan seams.
  The SELECT path (`IcebergTableScan`) keeps translating names snapshot-side and is untouched.
- **D-3** `scan/mod.rs` sits exactly at its size ceiling, so the fix removes the
  name-validation loop that the field-id loop below already performs, and the ceiling
  follows the file down.

## Proposition ledger

| Clause | Checkable proposition | Proof obligation | Status |
|---|---|---|---|
| C-001 | Unpinned `table.scan().select(current names)` after ADD COLUMN with no write since reads NULL for the added column. | `scan::evo_scan_tests` add-column pin; red on base. | PROVEN |
| C-002 | Same after RENAME COLUMN: the renamed column's values read by field id. | rename pin; red on base. | PROVEN |
| C-003 | Same after a name swap: each swapped name reads its own field. | swap pin; red on base. | PROVEN |
| C-004 | Pinned scan keeps binding the snapshot schema (time travel). | new pinned pin + full lib suite green. | PROVEN |
| C-005 | SQL `UPDATE … WHERE id = 1` and `DELETE … WHERE id = 1` after each of the three evolutions, two data files, CoW and MoR, assert post-statement rows. | `tests/evo_schema_dml.rs` (12 pins); red on base. | PROVEN |
| C-006 | Gates per brief step 5. | Command → result below. | PROVEN |
| C-007 | `use_ref("main")` is a no-op binding the current schema; a non-main ref still pins the snapshot schema. | `main_ref_*` pins (red on pre-fix tree) + tag guard; mirror of `BatchScan::use_ref`. | PROVEN |
| C-008 | C-004 pins are discriminating: forcing the current schema on pinned scans reds them. | three `snapshot_pinned_*` pins + hardened tag pin; `if false` mutation. | PROVEN |

## Red evidence

`CARGO_BUILD_JOBS=10 cargo test -p iceberg --lib scan::evo_scan_tests`:

```
test result: FAILED. 1 passed; 3 failed; 0 ignored; 0 measured; 3705 filtered out
```

- `unpinned_scan_after_add_column_null_fills_the_added_column` → `scan: DataInvalid
  => Column extra not found in table. Schema: table { 1: id: required long, 2: v:
  optional string }`
- `unpinned_scan_after_rename_reads_the_renamed_column_by_field_id` → `Column v not
  found in table` against the `w` schema
- `unpinned_scan_after_swapping_two_names_reads_each_field_by_id` → plans, then
  `left: [Some("a"), Some("b")] right: [Some("e1"), Some("e2")]` (the other field's
  values — the silent arm)
- `snapshot_pinned_scan_still_binds_the_snapshot_schema` green on base and stays green.

`CARGO_BUILD_JOBS=10 cargo test -p iceberg-datafusion --test evo_schema_dml`:

```
test result: FAILED. 2 passed; 10 failed; 0 ignored; 0 measured; 0 filtered out
```

- 8 loud: every add-column and rename UPDATE/DELETE refuses `Column extra` /
  `Column v not found in table` (the fork UPDATE/DELETE execs select the full current
  projection against the pinned snapshot schema).
- 2 silent: both swap UPDATE statements commit `["1", "x", "e1"]` where the field holds `"a"`.
- 2 green on base: both swap DELETE statements — a position delete writes no values and the
  final SELECT is evolution-aware, so they stand as regression guards per the brief's
  required 12.

Fixture note: three renames that swap two names need three sequential schema commits;
one action cannot see the `tmp` name it just created
(`Cannot rename missing column: tmp`).

## Round 2 red evidence (critic-289 L-001 / L-002, 2026-09-17)

`CARGO_BUILD_JOBS=10 cargo test -p iceberg --lib scan::evo_scan_tests`:

```
test result: FAILED. 7 passed; 3 failed; 0 ignored; 0 measured; 3723 filtered out
```

- L-001: `main_ref_scan_after_add_column_null_fills_the_added_column`,
  `main_ref_select_all_after_add_column_includes_the_added_column`,
  `main_ref_scan_after_swapping_two_names_reads_each_field_by_id` — all
  `DataInvalid => Column extra not found` (or the snapshot-name bind on swap):
  `use_ref("main")` counts as a pin and binds the snapshot schema.
- L-002: the hollow C-004 row-count pin is replaced by three discriminating pins
  (`snapshot_pinned_select_of_an_added_column_fails`,
  `snapshot_pinned_select_of_a_pre_rename_name_reads_the_field`,
  `snapshot_pinned_scan_after_a_name_swap_reads_snapshot_names`) — green on this
  tree, proven by the round-2 mutation below.
- Guard `tag_ref_on_the_pre_ddl_snapshot_binds_the_snapshot_schema` green throughout
  (non-main refs already bind the snapshot schema).

Fixture note: DDL writes no snapshot, so the pre-DDL snapshot is the current one at
tag time — the first tag-fixture draft searched for a non-current snapshot and found
none.

## Round 2 fix (critic-289 L-001)

`TableScanBuilder::use_ref` returns `self` unchanged for `"main"`
(`crate::spec::MAIN_BRANCH`), mirroring `BatchScan::use_ref` — `"main"` never reaches
`snapshot_ref`, so `build` treats the scan as unpinned and binds the current schema,
and `use_ref("main")` alongside `snapshot_id` no longer conflicts (Java no-op parity
with the batch adapter). Non-main refs store and pin as before. The existing `use_ref`
doc gains the `"main"` sentence (edited, not added). `scan/mod.rs` stays at its 6878
ceiling: the new body uses one declarative assignment instead of a branch block.

## Round 2 mutation (critic-289 L-002)

`if pinned && !self.project_current_schema` → `if false` (current schema for every
scan): `scan::evo_scan_tests` 6 passed / 4 failed — exactly the three
`snapshot_pinned_*` pins plus the hardened tag pin; all unpinned and `main`-ref pins
stay green. Restore `cmp` clean, re-green 10 / 10. The tag pin first stayed green
under this mutant (its `select(["id","v"])` names exist in both schemas) and was
hardened with a `select(["extra"])` refusal assertion before the re-run.

Out of scope, observed: `scan/incremental.rs:242` still reads "The `to` snapshot
supplies the schema, as `TableScanBuilder::build` does" — true of the incremental
behavior (unchanged) but now an incomplete description of `build` (snapshot schema
only when pinned). Left untouched per scope; named for a follow-up.

## Implemented fix

`crates/iceberg/src/scan/mod.rs` (`TableScanBuilder::build`): the bind schema is the
snapshot's schema only for an explicit pin (`snapshot_id` / `snapshot_ref`, Java
`useSnapshot`); otherwise it is the table's current schema (Java `newScan` binds
`table.schema()`). File planning still reads the resolved snapshot's manifests, and the
Arrow reader already projects by field id with NULL-fill, so older files read under the
current schema. New public `project_current_schema()` (one-line doc) opts a pinned scan
into current-schema binding — Spark `project(expectedSchema)` — over the pinned
snapshot's files. `crates/integrations/datafusion`: `mor_scan_stream` (`mor_scan.rs`)
and `cow_scan_stream` (`row_lineage.rs`) set it; both keep pinning the snapshot for
conflict detection. The SELECT path (`scan.rs`) keeps translating names snapshot-side
and is untouched, as is `resolve_affected_data_files` (manifest walk, no name binding).

`scan/mod.rs` sat exactly at its size ceiling: the pre-existing name-validation loop
was removed (the field-id loop below reports the identical missing-column error on the
identical condition — `field_by_name` and `field_id_by_name` read the same map), and
the ceiling follows the file 6879 → 6878.

Test determinism note: the first evo-scan pins asserted per-column encounter order and
failed once under full-suite load (batch order varies); all three now collect
id-keyed row tuples and sort.

## Execution evidence

`CARGO_BUILD_JOBS=10 cargo test -p iceberg --lib scan::evo_scan_tests` → 4 passed.
`CARGO_BUILD_JOBS=10 cargo test -p iceberg-datafusion --test evo_schema_dml` → 12 passed.
`cargo clippy -p iceberg --all-targets -- -D warnings` → exit 0.
`cargo clippy -p iceberg-datafusion --all-targets -- -D warnings` → exit 0.
`cargo fmt --all -- --check` → clean.

## Mutation evidence

Backups via plain `cp` (never `-p`), restore verified with `cmp`, `touch` after
restore, re-green before the next leg.

- M1 `scan/mod.rs`: `if pinned && !self.project_current_schema` → `if true` —
  `scan::evo_scan_tests` 1 passed / 3 failed (exactly the three unpinned pins; the
  pinned pin stays green). Restore `cmp` clean, re-green 4 / 4.
- M2a `mor_scan.rs`: `.project_current_schema()` removed — `evo_schema_dml` 7 passed /
  5 failed (exactly the five MoR pins that red on base; all six CoW green). Restore
  `cmp` clean.
- M2b `row_lineage.rs` (`cow_scan_stream`): same removal — 7 passed / 5 failed
  (exactly the five CoW pins; all six MoR green). Restore `cmp` clean, re-green 12 / 12.

## Gates

Round 2 (critic-289, rebased onto fork main `96fc9f1f`):

| Command | Result |
|---|---|
| `CARGO_BUILD_JOBS=10 RUST_TEST_THREADS=8 cargo test -p iceberg --lib` | ok. 3725 passed; 0 failed; 8 ignored |
| `CARGO_BUILD_JOBS=10 RUST_TEST_THREADS=8 cargo test -p iceberg-datafusion --lib` | ok. 228 passed; 0 failed; 1 ignored |
| `cargo clippy -p iceberg --all-targets -- -D warnings` | exit 0 |
| `cargo clippy -p iceberg-datafusion --all-targets -- -D warnings` | exit 0 |
| `python3 scripts/check_rust_file_size.py` | 485 files clean (99 legacy ceilings) |
| `typos .` | exit 0 |
| `./scripts/check_comment_blocks.sh` | OK |
| `./scripts/check_agent_artifacts.sh` | OK |
| `./scripts/check_matrix_anchors.sh` | OK (85 rows anchored) |

Round 1 (pre-rebase):

| Command | Result |
|---|---|
| `CARGO_BUILD_JOBS=10 RUST_TEST_THREADS=8 cargo test -p iceberg --lib` | ok. 3701 passed; 0 failed; 8 ignored |
| `CARGO_BUILD_JOBS=10 cargo clippy -p iceberg --all-targets -- -D warnings` | exit 0 |
| `CARGO_BUILD_JOBS=10 RUST_TEST_THREADS=8 cargo test -p iceberg-datafusion --lib` | ok. 228 passed; 0 failed; 1 ignored |
| `CARGO_BUILD_JOBS=10 RUST_TEST_THREADS=8 cargo test -p iceberg-datafusion --tests` | round 1: ok. 476 passed; 0 failed; 7 ignored over 31 targets (lib 228; every integration suite incl. `evo_schema_dml` 12; the 7 ignores are the pre-existing measure/probe pins; no suite needs Docker). Round 2: ok. 504 passed; 0 failed; 7 ignored (same ignores; rebase added suites) |
| `cargo clippy -p iceberg-datafusion --all-targets -- -D warnings` | exit 0 |
| `cargo fmt --all -- --check` | clean |
| `python3 scripts/check_rust_file_size.py` | 478 files clean (99 legacy ceilings) |
| `typos .` | exit 0 |
| `./scripts/check_comment_blocks.sh` | OK |
| `./scripts/check_agent_artifacts.sh` | OK |
| `./scripts/check_matrix_anchors.sh` | OK (84 rows anchored) |

## Open questions

None.
