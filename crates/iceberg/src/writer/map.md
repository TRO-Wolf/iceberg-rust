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

# map.md — crates/iceberg/src/writer/

## Purpose

The file-writing layer (Java `data/` writers): Arrow batches in → data / delete files out, as
`DataFile`s ready for a transaction action to commit. Layered: **partitioning** writers route rows →
**base** writers add Iceberg semantics → **file** writers do the physical format IO.

## Contents

| Path | What it does |
|---|---|
| `mod.rs` | the `IcebergWriter` / `IcebergWriterBuilder` traits + layering docs |
| `base_writer/data_file_writer.rs` | plain data files. `CurrentFileStatus` is empty/zero before the first write and after `close` |
| `base_writer/equality_delete_writer.rs` | equality-delete files (equality ids → projected schema) |
| `base_writer/position_delete_writer.rs` | position-delete files: `file_path` (id 2147483546) + `pos` (id 2147483545), `content(PositionDeletes)`, **write-as-given** (no sorting/merging — Java-faithful) |
| `base_writer/deletion_vector_writer.rs` | deletion vectors (V3 Puffin DVs, Java `BaseDVFileWriter`): accumulate `delete(path, pos, partition)`, `close()` → ONE Puffin file, one `deletion-vector-v1` blob per referenced data file (sorted-path order), `DeleteFile` per `createDV` L145-159; serialization in `../delete_vector.rs` (`serialize_deletion_vector_v1`, byte-identical to Java incl. run containers). **Previous-deletes MERGE hook (Arc-E Inc 2):** `with_previous_deletes(HashMap<path, PreviousDeletes>)` mirrors Java `loadPreviousDeletes`; `close_with_result()` → `DVWriteResult { delete_files, rewritten_delete_files }` (Java `DeleteWriteResult`): previous positions unioned into the new DV, file-scoped (`is_file_scoped` = Java `ContentFileUtil.isFileScoped`) source files returned for `RowDelta.remove_deletes`. Read a committed DV back with `delete_vector::load_delete_vector(&FileIO, &DataFile)` (F-13 U3a) — the public mirror of Java `BaseDeleteLoader.readDV`, and the thing that feeds `with_previous_deletes`. `DVWriteResult` also exposes Java `DeleteWriteResult`'s third member as DERIVED accessors — `referenced_data_files()` / `references_data_files()` (F-13 U3b) — feeding `RowDelta`'s `validateDataFilesExist` arming. Shared-Puffin DML/maintenance closure lives in `../delete_vector_container.rs` (`close_touched_dv_containers` on current snapshot, `close_touched_dv_containers_at` for a scanned/branch snapshot, `rewrite_siblings_for_dropped_references`). Status: GAP_MATRIX row R114 |
| `file_writer/parquet_writer.rs` | Parquet IO + per-column metrics collection (the bounds the evaluators later prune on); normalizes TOP-LEVEL UTC-alias `timestamptz` batches (`"UTC"`↔`"+00:00"`) to the writer schema (`UTC_TIME_ZONE` = `"UTC"`) metadata-only at the `write` funnel (F-A2-3, values bit-identical). **Refuses a variant-bearing schema at `build()`**, before any bytes reach storage — file-level variant I/O is unimplemented (row R88), and the Iceberg→Arrow conversion of `variant` now SUCCEEDS, so without this the failure would land at `close()` and leave an orphan file. Shares `arrow::schema::variant_path_within` with the reader's guard. **Refuses an `unknown`-bearing schema at `build()`** (row R91) with `FeatureUnsupported` naming the column — Arrow maps `unknown` to `DataType::Null`, and writing that column produced a parquet file no Iceberg scan can visit |
| `file_writer/avro_writer.rs` | Avro OCF data-file IO (`AvroWriterBuilder`/`AvroWriter`): Arrow batch → Iceberg `Literal` (`arrow_struct_to_literal`) → `RawLiteral` → resolved `apache_avro` value, write schema from `avro/schema.rs::schema_to_avro_schema`. Metrics are **rowCount + file size only** (Java `AvroMetrics.fromWriter` — no column metrics); variant/unknown rejected (reader-symmetric). Slots into the same `FileWriter` seam as parquet (engine landed; interop + GAP_MATRIX flip pending) |
| `write_defaults.rs` | fill missing top-level primitive columns from `write-default` (row R92); called from `DataFileWriter::write` only. Extra batch columns are dropped; a write schema that includes the reserved row-lineage pair (`schema_with_row_lineage`) keeps them, which is how v3 rewrite/COW persist stored `_row_id` / `_last_updated_sequence_number` (row R166) |
| `file_writer/rolling_writer.rs` | size-based file rolling |
| `file_writer/location_generator.rs` | file naming/placement |
| `partitioning/fanout_writer.rs` | concurrent multi-partition fanout |
| `partitioning/clustered_writer.rs` | sorted-input single-partition-at-a-time |
| `partitioning/unpartitioned_writer.rs` | passthrough |

Parity gaps live in the GAP_MATRIX (ORC data files, sort-order-aware writing; the Avro data-file
**writer engine** has landed in `file_writer/avro_writer.rs` — the GAP_MATRIX flip waits on the
interop round-trip). Deletion-vector capability status and shared-Puffin DML closure limits live on
GAP_MATRIX row R114.

## I want to...

| I want to... | go to |
|---|---|
| Write data for a partitioned table | `partitioning/` (fanout for unsorted, clustered for sorted input) |
| Write an unpartitioned file without a `PartitionKey` | `unpartitioned()` on the base writer, then `build(None)` |
| Persist stored `_row_id` / `_last_updated_sequence_number` on a rewrite | join them into the write schema via `metadata_columns::schema_with_row_lineage`, then `DataFileWriter` |
| Produce position deletes for `RowDelta` | `base_writer/position_delete_writer.rs` → commit via [../transaction/map.md](../transaction/map.md) `row_delta` |
| Touch metrics written into files | `file_writer/parquet_writer.rs` — these bounds feed the metrics evaluators; exact-byte sensitive |
| Add a new physical format | `file_writer/` behind the `FileWriter` trait |

## Pointers

- **Up:** [crates/iceberg/src/](..) · **Related:** `../spec/` (`DataFile`/manifest types),
  [../transaction/map.md](../transaction/map.md) (commits what this produces),
  `../arrow/` (schema conversion the writers rely on)

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| Java can't read a Rust-written delete file | Field-id mismatch — position-delete columns carry the reserved ids (2147483546/2147483545); equality deletes must carry the equality ids of the *projected* schema |
| Wrong field ids tolerated by Rust but not Java (or vice versa) | The Rust pos-delete READER matches by column POSITION (col 0 = file_path, col 1 = pos — it never reads field ids); JAVA matches by field id. Both contracts must hold: build the schema from `delete_file_path_field()`/`delete_file_pos_field()` in canonical order |
| Pruning broken on Rust-written files | Metrics/bounds written by `parquet_writer.rs` diverge from Java `Conversions.toByteBuffer` encoding — exact-byte fixture territory |
| Rows land in the wrong partition file | Partition-value computation in the partitioning writer vs the spec's transforms — check transform application, not the writer plumbing |
| A committed POSITION delete deletes nothing (rows "resurrect"), nothing errors | Its `partition_spec_id` differs from the data files'. `build(None)` with no spec now errors; `unpartitioned()` stamps spec 0 empty and still never applies on partitioned data. Pass `with_partition_spec` or a `PartitionKey`. Contract: ENGINE_CONTRACT §7a; pins in `position_delete_writer.rs::spec_stamp_e2e_test`. |
| A committed DELETION VECTOR is grouped into the wrong manifest, or (unverified) is pruned away and its rows resurrect | Its `partition_spec_id` is not the spec of the data files it references. A DV is NOT paired on `(spec_id, partition)` — the read side routes it by referenced-file PATH (`delete_file_index.rs`) — so the position-delete row above does not describe this. The verified cost is wrong per-spec manifest grouping; whether the DV's delete manifest can then be partition-pruned during planning is UNVERIFIED. Pass `DVFileWriter::with_partition_spec`, or a `PartitionKey` on `delete`, which wins. A partitioned spec with no key is rejected at `close` before any byte is written. Contract: ENGINE_CONTRACT §7a |
| An EQUALITY delete removed matching rows from partitions it was never meant to touch | Opposite direction, same root cause: a writer built with no `PartitionKey` emits an EMPTY partition tuple, and per the Iceberg spec an equality delete stored with an unpartitioned spec is a GLOBAL delete — applied to every data file, ignoring the spec id it claims (`PopulatedDeleteFileIndex::new` → `global_equality_deletes`; Java `DeleteFileIndex.Builder.add`). Configuring the spec does NOT scope it; pass the `PartitionKey` |
| Every write fails `Partition value is not compatible with partition type` on a table nothing else complains about | The current spec is unpartitioned with a non-zero id, and the file claimed spec 0. Call `with_partition_spec` with the current spec. |
| `Arrow: Incompatible type. Field '…' has type Timestamp(_, "UTC"), array has type Timestamp(_, "+00:00")` at write | A historical `"+00:00"` `timestamptz` batch vs the writer's canonical `"UTC"` schema (`UTC_TIME_ZONE`). `ParquetWriter::write` normalizes TOP-LEVEL UTC-alias timestamps metadata-only (`normalize_utc_alias_timestamps`); Spark `"UTC"` batches now match the writer schema with no relabel. A NESTED alias mismatch (inside a struct/list) is deliberately left to fail loud here (top-level-only seam — widening it is a fork follow-up). A genuinely different timezone (`"+05:00"`) is a real mismatch, not this — and stays loud. |
| `FeatureUnsupported` naming `Writing the unknown column` on the first `DataFileWriter::write` | The Iceberg schema (or a nested field) carries `unknown`. Parquet maps it to Arrow `Null` and used to commit an unreadable file. Refusal is at `ParquetWriterBuilder::build`, before bytes land. |

### First checks

- Round-trip the file through the Rust reader first (write → scan → compare); if that passes but
  Java fails, it's an encoding/field-id parity bug → go to the oracle.

### Escalate to

- Commit-side issues → [../transaction/map.md#debug](../transaction/map.md#debug).
- Cross-engine readability → [dev/java-interop/map.md#debug](../../../../dev/java-interop/map.md#debug).
