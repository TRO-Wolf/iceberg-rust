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

# F-OUTPUT-SPEC-ID-1 — a write can target any existing partition spec (Java `output-spec-id`)

## Finding (RePark parity inventory IPI-06, run 24c)

Spark's writer option `output-spec-id` writes new files under an existing, possibly older
partition spec: after `ADD PARTITION FIELD cat` (default becomes spec 1), `output-spec-id=0`
writes unpartitioned files stamped `spec_id=0`; after `DROP PARTITION FIELD cat` it still
partitions by `cat` under spec 0. A missing id fails
`IllegalArgumentException: Output spec id 9 is not a valid spec id for table`; a non-integer
fails at the caller's option parse. `overwritePartitions` honors the same option.
Oracle cells: `/tmp/oc-worker/qb-units/spark_output_spec_cells.json` (`OS-*`, v2 and v3).

## Fork-main measurement (commit 457fd2c0)

Fixture: `(id long, data string, cat string)`; spec 0 unpartitioned → `add_field(cat)` ⇒ spec 1
(`identity(cat)`, new default). One commit carries a spec-0 file (empty tuple) and a spec-1 file
(`cat=w`). Measured via `cargo test -p iceberg --lib output_spec_id_tests` (6 passed, 1 failed —
the merge-append pin fails on main by design):

| door | measured on main |
|---|---|
| `fast_append` | 2 new manifests, one per spec; `partition_spec_id` matches entries; tuples preserved (spec 0 → `()`, spec 1 → `(w)`); `files`/`entries` answer `[spec_id, partition, record_count]` = `[(0, (), 2), (1, (w), 1)]`. GREEN. |
| `merge_append` | PANICS at `merge_append.rs` `split_and_reorder` `debug_assert!(new_added_data.len() <= 1)`: "merge_append expects at most one new added data manifest (got 2)". The producer already emits one manifest per spec; the merge pass assumes a single new added manifest. **GAP.** |
| `overwrite_files` (delete + adds) | Per-spec manifests for the adds; spec-0 file keeps empty tuple, spec-1 keeps `(w)`; seed file deleted. GREEN. |
| `replace_partitions` (adds) | Per-spec manifests; replaces `(spec_id, partition)` tuples keyed on each added file's own spec. GREEN. |
| own-spec partition validation | spec-0 file with a spec-1 arity-1 tuple → `DataInvalid` at commit (validated against the file's own spec, not the default). GREEN. |
| older partitioned spec | PART-TO-UNPART fixture (spec 0 `identity(cat)`, spec 1 empty default): spec-0 file `cat=w` commits and keeps its tuple. GREEN. |
| unknown spec id | file stamped `partition_spec_id=9` → `DataInvalid` "Cannot find partition spec 9 for data file: <path>" at commit. GREEN. |

Validation errors surface at `Transaction::commit` (inside `SnapshotProducer::validate_added_data_files`),
not at `apply` — `apply` only registers the action.

### Writer-side inventory (hard-wired defaults found)

- `IcebergTableProvider::insert_into` → `project_with_partition` (`project.rs`) uses
  `metadata.default_partition_spec()`; `repartition` (`repartition.rs`) uses
  `table_metadata.default_partition_spec()`; `IcebergWriteExec::input_requirements`/`execute`
  (`write.rs`) use `default_partition_spec()` / `default_partition_type()`;
  `IcebergCommitExec::execute` (`commit.rs`) uses `default_partition_spec_id()` /
  `default_partition_type()` for data-file deserialization. **GAP** — no way to target an older spec.
- `iceberg::writer::TaskWriter` (DataFusion `task_writer.rs`), `RecordBatchPartitionSplitter`,
  `DataFileWriterBuilder::with_partition_spec`, `PartitionKey` all take an explicit spec already.
- `RewriteDataFiles::output_spec_id(i32)` exists (private `resolve_output_spec_in`) with the
  Java rewrite-action message, a different surface from the write option.
- **Missing shared helper**: no public `resolve_output_spec(table, Option<i32>)` raising the
  write-path Java error `Output spec id <n> is not a valid spec id for table` as `DataInvalid`.
  **GAP.**

## Gaps → work

1. `merge_append` `split_and_reorder` must carry every new added data manifest (one per spec)
   ahead of existing manifests, not assert ≤1.
2. Public `iceberg::writer::resolve_output_spec(&Table, Option<i32>) -> Result<PartitionSpecRef>`:
   `None` → default spec; `Some(id)` → that spec or `DataInvalid`
   `Output spec id <n> is not a valid spec id for table` (verbatim from the finding; Java source
   not consultable offline — the pin asserts this exact shape).
3. `IcebergTableProvider::with_output_spec_id(i32)`; `insert_into` resolves once and threads the
   spec through `project_with_partition`, `repartition`, `IcebergWriteExec`, `IcebergCommitExec`.

## RED

- `measure_merge_append_mixed_specs_groups_manifests_per_spec` fails on main with the
  `debug_assert` panic above — the merge-append gap.
- The writer-door gaps are missing API: a red test cannot compile before the API exists, so the
  `resolve_output_spec` / `with_output_spec_id` pins land with the implementation (tests with the
  code, same change).

## Implementation decisions

### merge_append fix

`split_and_reorder` no longer asserts ≤1 new added data manifest: every manifest whose
`added_snapshot_id == this snapshot` is this commit's new work and is carried ahead of the
existing manifests, preserving input order within each group. Java's
`Iterables.concat(prepareNewDataManifests(), filtered)` puts ALL new data manifests (one per
spec, HashMap spec order ≈ ascending for realistic spec ids) first; the new-added prefix is
sorted by `partition_spec_id` ascending so `first` (the stream head the bin-packer's
min-count rule protects, Java `ManifestFile first = manifestIter.next()`) is the lowest-spec
new manifest, matching Java's HashMap-ascending `prepareNewDataManifests` order.
Post-fix: `measure_merge_append_mixed_specs_groups_manifests_per_spec` GREEN; all 24
merge_append + 7 output_spec_id tests pass.

## API changes

- `iceberg::writer::resolve_output_spec(&Table, Option<i32>) -> Result<PartitionSpecRef>`
  (pub, `#[allow(missing_docs)]` per the comment ban). `None` → the table's default spec;
  `Some(id)` → that existing spec; unknown id → `ErrorKind::DataInvalid` with Java's write-path
  message `Output spec id <n> is not a valid spec id for table`.
- `IcebergTableProvider::with_output_spec_id(i32) -> Self` (pub, `output_spec_id: Option<i32>`
  field, preserved by `try_new`'s refresh path and `loaded.rs::from_planning_load` initializes
  `None`). `insert_into` resolves once via `resolve_output_spec` and threads the selected spec
  through `project_with_partition` (new `partition_spec` param), `repartition` (now takes
  `&PartitionSpec` instead of `TableMetadataRef`), `IcebergWriteExec::new` (new
  `partition_spec` param — input requirements, task writer, partition type), and
  `IcebergCommitExec::new` (new `output_spec` param — commit-side data-file deserialization
  uses the selected spec's id + partition type). `sort_for_write` needs no spec: it keys off
  the projected `_partition` column plus the table sort order.
- `repartition` is now `pub(crate)` (was `pub`); its doc block was stale after the signature
  change and was deleted, not reworded (comment ban: moved code sheds its comments).
- Parsing of the option string stays the caller's job (RePark): `with_output_spec_id` takes a
  typed `i32`, so a non-integer is unrepresentable at this layer (`OS-NOT-INT` is a
  caller-layer error by construction).

## Pins (v2 + v3)

Commit door — `crates/iceberg/src/transaction/tests/output_spec_id_tests.rs`
(`transaction::tests::output_spec_id_tests`; moved under `tests/` because `transaction/mod.rs`
sits at its 1937-line legacy ceiling — the `#[cfg(test)] mod` wiring lives inside the existing
`mod tests` block):

| cell (Spark `OS-*`) | pin | assertion |
|---|---|---|
| `OS-ADD-FIELD-NEW-1-{V2,V3}` | `pin_fast_append_mixed_specs_v{2,3}` | `files` = `[(0,(),2),(1,(w),1)]`, one manifest per spec, manifest `partition_spec_id` == entries' `partition_spec_id` |
| (merge-append door, same shape) | `pin_merge_append_mixed_specs_v{2,3}` | same files answer through `merge_append` |
| `OS-PART-TO-UNPART-0-{V2,V3}` | `pin_older_partitioned_spec_retains_tuple_v{2,3}` | spec-0 file keeps `cat=w` tuple after `DROP PARTITION FIELD` leaves an empty default |
| `OS-OVERWRITE-PARTS-0-{V2,V3}` | `pin_overwrite_parts_0_v{2,3}` | `replace_partitions` under spec 0 deletes the `(0,())`-keyed seed file → live files `[(0,(),2)]` |
| `OS-TWO-FIELDS-1-{V2,V3}` | `pin_two_fields_1_v{2,3}` | cell shape (round-3 V-03 reshape): two spec-1 files `cat ∈ {w,x}` with arity-2 tuples + one spec-0 `()` file; `specs` = `{0→1, 1→2}` |
| `OS-BAD-ID-{V2,V3}` (commit door) | `measure_added_file_unknown_spec_id_fails` (v2), `pin_added_file_unknown_spec_id_fails_v3` | `DataInvalid` "Cannot find partition spec 9 for data file" |

DataFusion `INSERT INTO` door — `crates/integrations/datafusion/src/table/output_spec_id_tests.rs`
(`table::output_spec_id_tests`; real parquet writes + `SELECT` reads back through the provider):

| cell | pin | assertion |
|---|---|---|
| `OS-ADD-FIELD-OLD-0` / `OS-INSERTINTO-0` / `OS-V1-SAVEASTABLE` (same oracle shape: two spec-0 files, rows 1,2,7,8) | `pin_insert_into_targets_older_spec_v{2,3}` | `with_output_spec_id(0)`: `files` = `[(0,(),2),(0,(),2)]`; rows = `{1a x, 2b y, 7g x, 8h w}` |
| `OS-ADD-FIELD-NEW-1-{V2,V3}` | `pin_insert_into_targets_new_spec_v{2,3}` | `files` = `[(0,(),2),(1,(w),1),(1,(x),1)]`; rows = all four |
| `OS-PART-TO-UNPART-0-{V2,V3}` | `pin_insert_into_partitioned_older_spec_v{2,3}` | `files` = four spec-0 files with cat tuples `{w,x,x,y}`; rows = all four |
| `OS-TWO-FIELDS-1-{V2,V3}` | `pin_insert_into_two_field_spec_v{2,3}` | spec-1 tuples arity 2, `cat ∈ {w,x}`, `id_bucket_2 ∈ [0,2)`; spec-0 tuple `()` |
| `OS-BAD-ID-{V2,V3}` (writer door) | `pin_insert_into_bad_spec_id_v{2,3}` | `with_output_spec_id(9)` insert fails with "Output spec id 9 is not a valid spec id for table" |
| CoW DELETE under `output-spec-id=0` (round-3 V-01) | `pin_delete_copy_on_write_targets_older_spec_v{2,3}` | `DELETE WHERE id=1` → survivor file `spec_id=0`, empty tuple, count 1; rows = `{2b y}` |
| CoW UPDATE under `output-spec-id=0` (round-3 V-01) | `pin_update_copy_on_write_targets_older_spec_v{2,3}` | `UPDATE SET data='z' WHERE id=1` → rewritten file `spec_id=0`, empty tuple, count 2; rows = `{1z x, 2b y}` |
| resolver unit pins | `resolve_output_spec_{none_resolves_table_default,returns_older_spec,unknown_id_is_data_invalid}` | default/older/error contract |

Spark `files`-table rendering note: the cells render each file's partition tuple through spec
field names (e.g. `[["cat", null]]` for a spec-0 file under the ADD-FIELD fixture). The fork pins
assert the stored tuple itself (`Struct::empty()` / `cat=w`), which is the on-disk ground truth
the inspect answer is derived from.

## Mutations

Run against committed state `e0c7c988`, tree restored between mutations (`git checkout`).

| mutation | site | pin(s) run | result |
|---|---|---|---|
| M1: default spec forced in the manifest writer | `snapshot.rs::write_added_manifests` — `new_cluster_manifest_writer(partition_spec_id, …)` → `new_cluster_manifest_writer(default_partition_spec_id(), …)` | `pin_fast_append_mixed_specs_v{2,3}` | RED: `mixed-spec commit: DataInvalid => Partition value has 0 fields but partition type has 1` |
| | | `pin_insert_into_targets_older_spec_v{2,3}` (`ADD-FIELD-OLD-0` shape) | RED: `External(DataInvalid => Partition value has 0 fields but partition type has 1)` |
| M2a: output-spec validation removed (writer door) | `writer/mod.rs::resolve_output_spec` — `Some(id)` → `unwrap_or_else(default)` | `resolve_output_spec_unknown_id_is_data_invalid` | RED: returned `PartitionSpec { spec_id: 1, … }` instead of erroring |
| | | `pin_insert_into_bad_spec_id_v{2,3}` | RED: `an unknown output spec id must fail: ()` — the insert committed under the default spec |
| M2b: unknown-spec validation removed (commit door) | `snapshot.rs::partition_type_for_added_file` — `ok_or_else(DataInvalid)` → `unwrap_or_else(default)` | `measure_added_file_unknown_spec_id_fails` + `pin_added_file_unknown_spec_id_fails_v3` | RED — but on a changed message: `Cannot rewrite manifests: unknown partition spec id 9` (the backstop in `cluster_partition_spec` still rejects the unknown id at manifest-writer construction). The pins assert the exact Java message, so the mutation is caught; note the commit door has TWO unknown-spec layers — removing both is needed to reach disk |
| M3: default spec forced in the CoW rewrite writer (round 3, run at `d4457038`) | `row_lineage.rs::StreamingDataFileWriter::try_new` — the `partition_spec` argument shadowed by `table.metadata().default_partition_spec().clone()` | `pin_delete_copy_on_write_targets_older_spec_v{2,3}` | RED: `files` = `[(1,(y),1)]` — the survivor was restamped under the default spec — expected `[(0,(),1)]` |
| | | `pin_update_copy_on_write_targets_older_spec_v{2,3}` | RED: `files` = `[(1,(x),1),(1,(y),1)]`, expected `[(0,(),2)]` |
| merge_append reverted alone (V-02 verification, round 3) | `git checkout 9c755bd8~1 -- merge_append.rs` | `cargo test -p iceberg --lib output_spec` (25 tests) | RED: `measure_merge_append_mixed_specs_groups_manifests_per_spec`, `pin_merge_append_mixed_specs_v{2,3}` — `merge_append expects at most one new added data manifest (got 2)`. GREEN (unchanged): `pin_fast_append_mixed_specs`, `pin_older_partitioned_spec_retains_tuple`, `pin_overwrite_parts_0`, `pin_two_fields_1` on v2+v3 — they measure pre-existing per-spec manifest grouping, not this unit's change |

## Round 3 — verification findings (PR #328)

| finding | severity | disposition | evidence | mutation |
|---|---|---|---|---|
| V-01: CoW DELETE/UPDATE restamps survivors under the DEFAULT spec, ignoring `with_output_spec_id(0)` | S2 behavioural | FIXED: `delete_from`/`update` resolve `resolve_output_spec(&table, self.output_spec_id)` exactly like `insert_into`; the resolved `PartitionSpecRef` rides `IcebergDeleteExec`/`IcebergUpdateExec` → `copy_on_write_delete`/`copy_on_write_update`/`merge_on_read_update` → `StreamingDataFileWriter::try_new(table, spec)` (was `default_partition_spec()` hard-wired). MoR DELETE writes no data files — no spec needed. `StreamingDataFileWriter` keeps one field used by `ensure_writer`, so CoW, MoR-UPDATE new rows and the INSERT-free DML doors all stamp the resolved spec | RED first: `pin_{delete,update}_copy_on_write_targets_older_spec_v{2,3}` failed as `[(1,(y),1)]`/`[(1,(x),1),(1,(y),1)]` vs expected `[(0,(),1)]`/`[(0,(),2)]` — byte-for-byte the critic's observation; after the fix all four pass | M3 (above): default spec forced back in `try_new` → all four pins red on exactly those tuples |
| V-02: four commit-door pins stay green when merge_append alone is reverted — they measure pre-existing behaviour | S3 coverage | ACCEPTED + documented: verified by reverting `merge_append.rs` to `9c755bd8~1` — only the merge-append pins redden (`expects at most one new added data manifest (got 2)`); `pin_fast_append_mixed_specs`, `pin_older_partitioned_spec_retains_tuple`, `pin_overwrite_parts_0`, `pin_two_fields_1` stay green on v2+v3. They are kept as the fork-main measurement record. Per-door red coverage of THIS unit's change: merge_append → `pin_merge_append_mixed_specs` (revert); every commit door's per-spec manifest write → M1 (`write_added_manifests` forces default spec → fast_append + DF INSERT pins red; the same shared path serves overwrite/replace_partitions); writer door → M2a; commit validation → M2b; CoW door → M3 | revert run above | each named mutation in the Mutations table |
| V-03: `pin_two_fields_1` asserted a hand-built one-file spec-1 tuple, not the `OS-TWO-FIELDS-1` cell | S3 oracle fidelity | FIXED: pin reshaped to the cell — two spec-1 files (`cat ∈ {w,x}`, one row each, arity-2 tuples) + one spec-0 `()` two-row file; per-spec counts `{0→1, 1→2}` match the cell's `specs` answer. Spark renders `id_bucket_2` as `null` in the `files` inspect answer; the pin asserts the stored tuple (bucket ints in `[0,2)`), consistent with `pin_insert_into_two_field_spec` on the DataFusion door | `pin_two_fields_1_v{2,3}` green after reshape | covered by M1-class revert like every per-spec commit-door pin |

## Gates

Round 3 (final tree, at the head commit):

- `cargo fmt --all -- --check` → clean
- `cargo clippy -p iceberg -p iceberg-datafusion --all-targets -- -D warnings` → clean
- `cargo test -p iceberg --lib output_spec` → 25 passed, 0 failed
- `cargo test -p iceberg --lib merge_append` → 27 passed, 0 failed
- `cargo test -p iceberg-datafusion --lib output_spec` → 14 passed, 0 failed
- `cargo test -p iceberg-datafusion --lib table::` → 95 passed, 0 failed
- `cargo test -p iceberg-datafusion --lib delete` → 51 passed, 0 failed, 1 ignored (touched-path regression filter)
- `bash scripts/check_rust_file_size.sh` → 602 files clean (90 legacy ceilings); `delete.rs` split
  (`delete_decode.rs` — the `_file`/`_pos` decoders) lowered its ceiling 1145 → 1047
- `python3 /tmp/oc-worker/_lib/comment_ban.py /tmp/qb-fork origin/main` → `comment-ban hits=0`

Round-2 gate runs (same commands at `254398a2`, pre-rebase) all passed; the list above is the
authoritative re-run after the round-3 rebase (`254398a2` → fork main at #325, replayed as
`5dc19f6c…b95785be`) and the V-01/V-03 changes.

## Residuals / ambiguity

- `OS-NOT-INT`: the fork's writer API takes a typed `i32`, so the non-integer case is
  unrepresentable here — RePark's option parsing owns `NumberFormatException`-equivalent
  behavior. No fork-level pin is possible; the typed API is the pin.
- `OS-V1-SAVEASTABLE` / `OS-INSERTINTO-0` share the `ADD-FIELD-OLD-0` oracle shape
  (`files = [[0,(),2],[0,(),2]]`, rows `{1,2,7,8}`); `pin_insert_into_targets_older_spec` covers
  all three through the provider's `INSERT INTO` door — the fork has no distinct
  saveAsTable door.
- The `IcebergCommitExec` node now deserializes commit output under the SELECTED spec. On a
  mixed-spec write (spec-0 files + spec-1 files in one commit — not producible through the
  provider's single `output_spec_id`, only through hand-built plans) the node would need
  per-file spec resolution; out of scope for the single-target option.
- `merge_append` ordering: new-added manifests are sorted ascending by `partition_spec_id` so
  the bin-packer stream head is the lowest-spec manifest, matching Java's HashMap-ascending
  `prepareNewDataManifests` order; the manifest list compares as a set downstream.
- The `#[allow(missing_docs)]` on `resolve_output_spec` is the sanctioned substitute for a doc
  comment under the comment ban (doc comments the compiler demands are allowed only as the
  minimum that compiles, and the ban script counts any added `///` line).
