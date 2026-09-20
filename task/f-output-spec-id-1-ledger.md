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

(to fill)

## Pins (v2 + v3)

(to fill)

## Mutations

(to fill)

## Gates

(to fill)

## Residuals / ambiguity

(to fill)
