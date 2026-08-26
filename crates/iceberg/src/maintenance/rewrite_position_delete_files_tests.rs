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

//! Tests for [`RewritePositionDeleteFiles`]. Each is a corruption-class READ-IDENTITY proof: the
//! merge-on-read live row set is asserted IDENTICAL before and after the rewrite, plus the four
//! `Result` counts. The file count moves in BOTH directions across this file and identity holds in
//! each: most fixtures FUSE many parquet pos-deletes into fewer compacted ones, while the split
//! battery (`test_output_splits_into_multiple_files_at_a_small_explicit_config` and its siblings)
//! rewrites a bin into MORE files than it consumed. The crown jewel + the seq-stamp test pin the
//! silent-corruption staller (the compacted file must carry the group MAX rewritten data seq); the
//! grouping + partition-isolation tests pin the `(spec, partition)` planning; the C-008 format
//! battery pins the V2-parquet-only scope over all four `DataFileFormat` variants (a Puffin
//! deletion vector, a V2 ORC pos-delete and a V2 Avro pos-delete are all SKIPPED, not compacted).

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use tempfile::TempDir;

use super::*;
use crate::delete_file_index::referenced_data_file_location;
use crate::expr::Reference;
use crate::io::LocalFsStorageFactory;
use crate::memory::MemoryCatalogBuilder;
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, Datum, FormatVersion, Literal, ManifestContentType,
    NestedField, Operation, PartitionKey, PartitionSpec, PrimitiveType, Schema as IcebergSchema,
    SnapshotRef, Struct, Transform, Type,
};
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
};
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::file_writer::{FileWriter, FileWriterBuilder};
use crate::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};

// =================================================================================================
// Helpers (table build / data + position-delete writers / scan) — same shape as the convert tests.
// =================================================================================================

async fn local_fs_catalog() -> (impl Catalog, TempDir) {
    let temp_dir = TempDir::new().expect("temp dir");
    let warehouse = temp_dir
        .path()
        .to_str()
        .expect("utf8 temp path")
        .to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            std::collections::HashMap::from([("warehouse".to_string(), warehouse)]),
        )
        .await
        .expect("load local-fs memory catalog");
    (catalog, temp_dir)
}

fn three_long_schema() -> IcebergSchema {
    IcebergSchema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "x",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                2,
                "y",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                3,
                "z",
                Type::Primitive(PrimitiveType::Long),
            )),
        ])
        .build()
        .expect("build schema")
}

/// A partitioned table at a DELIBERATELY SHORT location. `parquet-rs` truncates byte-array
/// statistics at 64 bytes and the metrics aggregator drops a non-exact bound, so a data-file path
/// longer than that leaves a position delete with NO `file_path` bounds — the reader then treats it
/// as partition-scoped. A short path is what lets a fixture reach the real file-scoped routing.
async fn create_short_path_partitioned_table(
    catalog: &impl Catalog,
    warehouse: &Path,
    format_version: FormatVersion,
) -> Table {
    let schema = three_long_schema();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("x", "x", Transform::Identity)
        .expect("add partition field")
        .build()
        .expect("build spec");
    let namespace = NamespaceIdent::new("n".to_string());
    catalog
        .create_namespace(&namespace, std::collections::HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("t".to_string())
        .location(format!("{}/w", warehouse.to_str().expect("utf8 temp path")))
        .schema(schema)
        .partition_spec(spec)
        .format_version(format_version)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

async fn create_partitioned_table(catalog: &impl Catalog, format_version: FormatVersion) -> Table {
    let schema = three_long_schema();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("x", "x", Transform::Identity)
        .expect("add partition field")
        .build()
        .expect("build spec");
    create_table_with_spec(catalog, schema, spec, format_version).await
}

async fn create_unpartitioned_table(
    catalog: &impl Catalog,
    format_version: FormatVersion,
) -> Table {
    let schema = three_long_schema();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .build()
        .expect("build spec");
    create_table_with_spec(catalog, schema, spec, format_version).await
}

async fn create_table_with_spec(
    catalog: &impl Catalog,
    schema: IcebergSchema,
    spec: PartitionSpec,
    format_version: FormatVersion,
) -> Table {
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, std::collections::HashMap::new())
        .await
        .expect("create namespace");
    let table_ident = TableIdent::new(namespace.clone(), "t".to_string());
    let creation = TableCreation::builder()
        .name(table_ident.name().to_string())
        .schema(schema)
        .partition_spec(spec)
        .format_version(format_version)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

/// Write a DATA file in partition `part_value` holding `rows` (the file path is returned for use as the
/// position-delete `file_path` target).
async fn write_data_file(
    table: &Table,
    file_name: &str,
    part_value: i64,
    rows: &[(i64, i64, i64)],
) -> DataFile {
    use crate::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).unwrap());

    let xs: Vec<i64> = rows.iter().map(|(x, _, _)| *x).collect();
    let ys: Vec<i64> = rows.iter().map(|(_, y, _)| *y).collect();
    let zs: Vec<i64> = rows.iter().map(|(_, _, z)| *z).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(xs)) as ArrayRef,
        Arc::new(Int64Array::from(ys)) as ArrayRef,
        Arc::new(Int64Array::from(zs)) as ArrayRef,
    ])
    .unwrap();

    let file_path = format!("{}/data/{}", table.metadata().location(), file_name);
    let output = table.file_io().new_output(file_path).unwrap();
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let mut writer = parquet_builder.build(output).await.unwrap();
    writer.write(&batch).await.unwrap();
    let data_file_builders = writer.close().await.unwrap();

    let mut builder = data_file_builders.into_iter().next().unwrap();
    let partition = if table.metadata().default_partition_spec().is_unpartitioned() {
        Struct::empty()
    } else {
        Struct::from_iter([Some(Literal::long(part_value))])
    };
    builder
        .content(DataContentType::Data)
        .partition_spec_id(0)
        .partition(partition)
        .build()
        .unwrap()
}

/// Write a real PARQUET position-delete file masking the given `(target_path, pos)` pairs in partition
/// `part_value`. Returns the resulting position-delete [`DataFile`] (so it can be committed via
/// `add_deletes`). This is the multi-pos-delete fixture the action compacts.
async fn write_position_delete_file(
    table: &Table,
    part_value: Option<i64>,
    pairs: &[(&str, i64)],
) -> DataFile {
    let schema = table.metadata().current_schema().clone();
    let config = PositionDeleteWriterConfig::new().unwrap();

    let location_gen = DefaultLocationGenerator::new(table.metadata().clone()).unwrap();
    let file_name_gen = DefaultFileNameGenerator::new(
        "pos-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        config.schema().clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );

    let partition_key = part_value.map(|pv| {
        PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            schema.clone(),
            Struct::from_iter([Some(Literal::long(pv))]),
        )
        .expect("PartitionKey::new: valid partition tuple")
    });
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(partition_key)
        .await
        .unwrap();

    let paths: Vec<&str> = pairs.iter().map(|(p, _)| *p).collect();
    let positions: Vec<i64> = pairs.iter().map(|(_, p)| *p).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions)) as ArrayRef,
    ])
    .unwrap();
    writer.write(batch).await.unwrap();
    writer.close().await.unwrap().into_iter().next().unwrap()
}

/// Write a FILE-SCOPED parquet position delete: one that carries FULL, untruncated `file_path`
/// bounds, so the reader routes it by PATH with no partition condition.
///
/// The plain [`write_position_delete_file`] helper leaves the default `truncate(16)` metrics config
/// in place, which shortens the bounds and makes every delete it writes partition-scoped. Java's
/// `MetricsConfig.forPositionDelete` is what a real position-delete writer uses, and the fork's own
/// production path sets it too.
async fn write_file_scoped_position_delete_file(
    table: &Table,
    part_value: i64,
    target_path: &str,
    positions: &[i64],
) -> DataFile {
    let schema = table.metadata().current_schema().clone();
    let config = PositionDeleteWriterConfig::new().unwrap();
    let location_gen = DefaultLocationGenerator::new(table.metadata().clone()).unwrap();
    let file_name_gen = DefaultFileNameGenerator::new(
        "fs-pos-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        config.schema().clone(),
    )
    .with_metrics_config(MetricsConfig::for_position_delete());
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let partition_key = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        schema,
        Struct::from_iter([Some(Literal::long(part_value))]),
    )
    .expect("PartitionKey::new: valid partition tuple");
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(Some(partition_key))
        .await
        .unwrap();
    let paths: Vec<&str> = positions.iter().map(|_| target_path).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions.to_vec())) as ArrayRef,
    ])
    .unwrap();
    writer.write(batch).await.unwrap();
    writer.close().await.unwrap().into_iter().next().unwrap()
}

async fn append_files(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let action = tx.fast_append().add_data_files(files);
    let tx = action.apply(tx).unwrap();
    tx.commit(catalog).await.unwrap()
}

async fn add_deletes(catalog: &impl Catalog, table: &Table, deletes: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let action = tx.row_delta().add_deletes(deletes);
    let tx = action.apply(tx).unwrap();
    tx.commit(catalog).await.unwrap()
}

/// Scan the table and collect the `y` column values (merge-on-read deletes applied) — the read signal.
async fn scan_y_values(table: &Table) -> HashSet<i64> {
    let stream = table
        .scan()
        .select(["y"])
        .build()
        .unwrap()
        .to_arrow()
        .await
        .unwrap();
    let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
    let mut values = HashSet::new();
    for batch in batches {
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for index in 0..col.len() {
            values.insert(col.value(index));
        }
    }
    values
}

/// Every live DELETE file in the current snapshot.
async fn live_delete_files(table: &Table) -> Vec<DataFile> {
    let snapshot = table.metadata().current_snapshot().unwrap();
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .unwrap();
    let mut files = Vec::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
        for entry in manifest.entries() {
            if entry.is_alive() {
                files.push(entry.data_file().clone());
            }
        }
    }
    files
}

/// The `(data_file, sequence_number)` of every live DELETE entry — for the seq-stamp staller.
async fn live_delete_entries_with_seq(table: &Table) -> Vec<(DataFile, Option<i64>)> {
    let snapshot = table.metadata().current_snapshot().unwrap();
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .unwrap();
    let mut out = Vec::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
        for entry in manifest.entries() {
            if entry.is_alive() {
                out.push((entry.data_file().clone(), entry.sequence_number()));
            }
        }
    }
    out
}

fn count_pos(files: &[DataFile]) -> usize {
    files
        .iter()
        .filter(|f| f.content_type() == DataContentType::PositionDeletes)
        .count()
}

// =================================================================================================
// CROWN JEWEL — read-identity over a data file masked by 2+ parquet position-delete files.
// =================================================================================================

/// THE CROWN JEWEL (read-identity). A single data file is masked by TWO separate parquet position-delete
/// files (one masking pos 1 = y=20, one masking pos 3 = y=40). Compact them; the post-compaction MoR
/// scan must return the SAME live rows ({10,30,50}), the two old pos-delete files must be GONE, exactly
/// ONE compacted pos-delete added, and the Result counts must be (2 rewritten, 1 added).
///
/// MUTATION COVERAGE: grouping — if compaction collected positions from only one of the two files (e.g.
/// a `break` after the first), one masked row would resurrect and the after-set would differ from
/// before, failing the read-identity assertion.
/// MIGRATED (size gate): this fixture is a TWO-file group, deliberately BELOW Java's
/// `MIN_INPUT_FILES_DEFAULT` of 5, because its subject is READ IDENTITY across a compaction — not
/// admission. It therefore sets `.min_input_files(2)` explicitly, which is also acceptance item
/// 2: the floor is configuration, not a hard-coded constant. Removing that knob must RED this
/// test — it asserts post-`execute` SHAPE, never read identity alone.
#[tokio::test]
async fn test_crown_jewel_read_identity_data_file_masked_by_two_pos_deletes() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    // Data file: y = 10,20,30,40,50 at positions 0..5.
    let x = write_data_file(&table, "x.parquet", 0, &[
        (0, 10, 100),
        (0, 20, 200),
        (0, 30, 300),
        (0, 40, 400),
        (0, 50, 500),
    ])
    .await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await; // X: data seq 1

    // Two SEPARATE parquet pos-delete files, each masking a distinct position of X.
    let pd1 = write_position_delete_file(&table, Some(0), &[(&x_path, 1)]).await; // y=20
    let table = add_deletes(&catalog, &table, vec![pd1]).await; // seq 2
    let pd2 = write_position_delete_file(&table, Some(0), &[(&x_path, 3)]).await; // y=40
    let table = add_deletes(&catalog, &table, vec![pd2]).await; // seq 3

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 30, 50]),
        "before: two pos-deletes mask y=20 (pos 1) and y=40 (pos 3)"
    );
    assert_eq!(
        count_pos(&live_delete_files(&table).await),
        2,
        "before: two live position-delete files"
    );

    let result = RewritePositionDeleteFiles::new(table.clone())
        .min_input_files(2)
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(
        result.rewritten_delete_files_count, 2,
        "two files rewritten"
    );
    assert_eq!(
        result.added_delete_files_count, 1,
        "one compacted file added"
    );
    assert!(
        result.rewritten_bytes_count > 0,
        "rewritten bytes must be non-zero"
    );
    assert!(result.added_bytes_count > 0, "added bytes must be non-zero");

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();

    // READ IDENTITY: the live row set is unchanged.
    let after = scan_y_values(&reloaded).await;
    assert_eq!(
        after, before,
        "read identity: live rows IDENTICAL before vs after compaction"
    );

    // Exactly one compacted pos-delete is live (fewer files).
    assert_eq!(
        count_pos(&live_delete_files(&reloaded).await),
        1,
        "the two pos-deletes are compacted into exactly one"
    );
}

// =================================================================================================
// STALLER (seq stamping) — the compacted file must carry the group MAX rewritten data seq.
// =================================================================================================

/// SEQ STAMPING (the silent-corruption staller). Data X is at seq 1; two pos-deletes mask it at seqs 2
/// and 3. The compacted file MUST carry the BIN MAX rewritten data seq (3) — NOT the inherited
/// (higher) rewrite-snapshot seq, NOT the min (2). If it carried the inherited seq it would still apply
/// here (4 > 1) so the read would look fine — this test therefore asserts the EXACT stamped seq is 3,
/// pinning the precise stamp.
///
/// C-010 element 1 — the ONE-OUTPUT bin, which fixes the stamp's BASE VALUE. The two other elements
/// fix the other two dimensions: `test_every_split_output_carries_bin_max_rewritten_seq` the FAN-OUT
/// (every output of a split bin, not just the first) and
/// `test_each_bin_output_carries_its_own_bin_max_not_the_partition_max` the RANGING (this bin's
/// entries, not the partition's).
///
/// MUTATION COVERAGE: change `add_delete_file_with_sequence_number(.., max_seq)` to
/// `add_delete_file(..)` (inherit) and the live compacted pos-delete seq becomes the rewrite snapshot's
/// seq (4), not 3; the seq assertion fails. Change `.max()` to `.min()` and the stamp becomes 2; fails.
/// MIGRATED (size gate): this fixture is a TWO-file group, deliberately BELOW Java's
/// `MIN_INPUT_FILES_DEFAULT` of 5, because its subject is the sequence-number STAMP the compacted
/// file carries — not admission. It therefore sets `.min_input_files(2)` explicitly, which is
/// also acceptance item 2: the floor is configuration, not a hard-coded constant. Removing that
/// knob must RED this test — it asserts post-`execute` SHAPE, never read identity alone.
#[tokio::test]
async fn test_compacted_file_carries_bin_max_rewritten_seq() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 0, &[
        (0, 10, 1),
        (0, 20, 2),
        (0, 30, 3),
    ])
    .await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await; // X: data seq 1

    let pd1 = write_position_delete_file(&table, Some(0), &[(&x_path, 1)]).await; // y=20
    let table = add_deletes(&catalog, &table, vec![pd1]).await; // seq 2
    let pd2 = write_position_delete_file(&table, Some(0), &[(&x_path, 0)]).await; // y=10
    let table = add_deletes(&catalog, &table, vec![pd2]).await; // seq 3

    // Confirm the fixture seqs: the two pos-deletes are at seq 2 and 3 (max = 3).
    let seqs: Vec<i64> = live_delete_entries_with_seq(&table)
        .await
        .into_iter()
        .filter(|(f, _)| f.content_type() == DataContentType::PositionDeletes)
        .filter_map(|(_, seq)| seq)
        .collect();
    assert_eq!(
        seqs.iter().copied().max(),
        Some(3),
        "fixture: this bin's MAX rewritten pos-delete seq is 3"
    );

    let before = scan_y_values(&table).await;
    assert_eq!(before, HashSet::from([30]), "before: y=10 and y=20 masked");

    RewritePositionDeleteFiles::new(table.clone())
        .min_input_files(2)
        .execute(&catalog)
        .await
        .unwrap();

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    let pos_entries: Vec<(DataFile, Option<i64>)> = live_delete_entries_with_seq(&reloaded)
        .await
        .into_iter()
        .filter(|(f, _)| f.content_type() == DataContentType::PositionDeletes)
        .collect();
    assert_eq!(
        pos_entries.len(),
        1,
        "exactly one compacted pos-delete after compaction"
    );
    assert_eq!(
        pos_entries[0].1,
        Some(3),
        "the compacted pos-delete MUST carry the BIN MAX rewritten data seq (3), \
         not the inherited rewrite seq and not the min"
    );

    // And read identity still holds (the stamped delete still masks y=10 and y=20).
    assert_eq!(scan_y_values(&reloaded).await, HashSet::from([30]));
}

/// SEQ STAMPING — the resurrection guard. Data X at seq 1 is masked by two pos-deletes; a SECOND data
/// file W at seq 4 (committed AFTER the deletes) also lives. The compacted file must be stamped seq 3
/// (the group max of the rewritten deletes), which is `< 4` so it never touches W, and `> 1` so it still
/// masks X. If the stamp were inherited (seq 5 from the rewrite), it would `> 1` so X stays masked
/// (looks fine) — but the resurrection failure mode is the INVERSE: an OVER-low stamp. We pin the read
/// identity across BOTH data files so any wrong stamp that changes the masked set fails.
/// MIGRATED (size gate): this fixture is a TWO-file group, deliberately BELOW Java's
/// `MIN_INPUT_FILES_DEFAULT` of 5, because its subject is the sequence-number STAMP (resurrection
/// / over-apply) — not admission. It therefore sets `.min_input_files(2)` explicitly, which is
/// also acceptance item 2: the floor is configuration, not a hard-coded constant. Removing that
/// knob must RED this test — it asserts post-`execute` SHAPE, never read identity alone.
#[tokio::test]
async fn test_seq_stamp_does_not_resurrect_or_over_apply() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 1), (0, 20, 2)]).await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await; // X: seq 1

    let pd1 = write_position_delete_file(&table, Some(0), &[(&x_path, 0)]).await; // X.y=10
    let table = add_deletes(&catalog, &table, vec![pd1]).await; // seq 2
    let pd2 = write_position_delete_file(&table, Some(0), &[(&x_path, 1)]).await; // X.y=20
    let table = add_deletes(&catalog, &table, vec![pd2]).await; // seq 3

    // A NEW data file W (seq 4) with the same y values lives — the deletes must NOT touch it.
    let w = write_data_file(&table, "w.parquet", 0, &[(0, 10, 9), (0, 20, 10)]).await;
    let table = append_files(&catalog, &table, vec![w]).await; // W: seq 4

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 20]),
        "before: X fully masked; W (seq 4 > delete seqs) survives with y=10,20"
    );

    let result = RewritePositionDeleteFiles::new(table.clone())
        .min_input_files(2)
        .execute(&catalog)
        .await
        .unwrap();

    // SHAPE FIRST (acceptance criterion 4 / design call 2). Read identity ALONE is satisfied by a
    // DECLINED bin doing nothing, so this test asserts the bin was actually ADMITTED before it
    // asserts anything about what the admitted run produced. Remove the `.min_input_files(2)` knob
    // above and these two assertions red.
    assert_eq!(
        result.rewritten_delete_files_count, 2,
        "the two pos-deletes must actually be rewritten — a declined bin rewrites nothing"
    );
    assert_eq!(result.added_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    let pos_entries: Vec<(DataFile, Option<i64>)> = live_delete_entries_with_seq(&reloaded)
        .await
        .into_iter()
        .filter(|(f, _)| f.content_type() == DataContentType::PositionDeletes)
        .collect();
    assert_eq!(
        pos_entries.len(),
        1,
        "the two pos-deletes compacted into ONE — a declined bin would leave two"
    );
    // POWER ON ITS OWN SUBJECT: pin the EXACT stamp, not just the row set. Read identity cannot
    // separate `max` from `inherit` here (both are > X's seq 1, so X stays masked either way), and
    // it cannot separate `max` from `min` either (2 and 3 both mask X and both miss W). The stamp
    // assertion can, and does.
    assert_eq!(
        pos_entries[0].1,
        Some(3),
        "the compacted pos-delete MUST carry the group MAX rewritten data seq (3): `< 4` so it \
         never touches W, `> 1` so it still masks X"
    );
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity: the compacted delete still masks X (seq 3 > 1) and never touches W (seq 4)"
    );
}

// =================================================================================================
// GROUPING + PARTITION ISOLATION.
// =================================================================================================

/// MULTI-FILE GROUPING across DATA files in one partition. Two data files in partition 0, each masked by
/// its own pos-delete file. Both pos-deletes share `(spec 0, partition 0)`, so they compact into ONE
/// file carrying both data files' positions. Read identity must hold.
/// MIGRATED (size gate): this fixture is a TWO-file group, deliberately BELOW Java's
/// `MIN_INPUT_FILES_DEFAULT` of 5, because its subject is GROUPING across data files in one
/// partition — not admission. It therefore sets `.min_input_files(2)` explicitly, which is also
/// acceptance item 2: the floor is configuration, not a hard-coded constant. Removing that knob
/// must RED this test — it asserts post-`execute` SHAPE, never read identity alone.
#[tokio::test]
async fn test_multi_file_grouping_one_partition() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let a = write_data_file(&table, "a.parquet", 0, &[(0, 10, 1), (0, 20, 2)]).await;
    let b = write_data_file(&table, "b.parquet", 0, &[(0, 30, 3), (0, 40, 4)]).await;
    let a_path = a.file_path().to_string();
    let b_path = b.file_path().to_string();
    let table = append_files(&catalog, &table, vec![a, b]).await;

    let pda = write_position_delete_file(&table, Some(0), &[(&a_path, 1)]).await; // a.y=20
    let table = add_deletes(&catalog, &table, vec![pda]).await;
    let pdb = write_position_delete_file(&table, Some(0), &[(&b_path, 0)]).await; // b.y=30
    let table = add_deletes(&catalog, &table, vec![pdb]).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 40]),
        "before: y=20 and y=30 masked"
    );

    let result = RewritePositionDeleteFiles::new(table.clone())
        .min_input_files(2)
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(result.rewritten_delete_files_count, 2);
    assert_eq!(result.added_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity: the compacted file carries BOTH data files' positions"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 1);
}

/// PARTITION ISOLATION + MULTI-GROUP TABLE ADVANCE. Two partitions, each with two pos-delete
/// files. The action compacts EACH partition's group SEPARATELY (one compacted file per
/// partition, never merging across partitions). Read identity per-partition must hold.
///
/// Each group commits its own `Replace` snapshot; the action advances the base table after each
/// group commit (mirrors `RewriteDataFiles`) so the next `Transaction` is built on the prior
/// group's tip. Without advance, `do_commit` still refreshes + re-applies (correctness holds;
/// cost is extra re-apply work — not forced CAS conflict retries).
///
/// MUTATION COVERAGE: collapse the `(spec, partition)` group key to spec-only and both partitions' files
/// would merge into one group; the compacted file's partition would be wrong and the per-partition read
/// identity would break (or the writer/commit would error on a partition mismatch).
/// MIGRATED (size gate): this fixture is a TWO-file group, deliberately BELOW Java's
/// `MIN_INPUT_FILES_DEFAULT` of 5, because its subject is PARTITION ISOLATION and the per-group
/// commit — not admission. It therefore sets `.min_input_files(2)` explicitly, which is also
/// acceptance item 2: the floor is configuration, not a hard-coded constant. Removing that knob
/// must RED this test — it asserts post-`execute` SHAPE, never read identity alone.
#[tokio::test]
async fn test_partition_isolation_compacts_each_group_separately() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let p0 = write_data_file(&table, "p0.parquet", 0, &[(0, 10, 1), (0, 20, 2)]).await;
    let p1 = write_data_file(&table, "p1.parquet", 1, &[(1, 30, 3), (1, 40, 4)]).await;
    let p0_path = p0.file_path().to_string();
    let p1_path = p1.file_path().to_string();
    let table = append_files(&catalog, &table, vec![p0, p1]).await;

    // Partition 0: two pos-deletes (mask y=20 in two parts). Partition 1: two pos-deletes (mask y=40).
    let p0d1 = write_position_delete_file(&table, Some(0), &[(&p0_path, 1)]).await; // p0.y=20
    let p1d1 = write_position_delete_file(&table, Some(1), &[(&p1_path, 1)]).await; // p1.y=40
    let table = add_deletes(&catalog, &table, vec![p0d1, p1d1]).await;
    let p0d2 = write_position_delete_file(&table, Some(0), &[(&p0_path, 1)]).await; // dup p0.y=20
    let p1d2 = write_position_delete_file(&table, Some(1), &[(&p1_path, 1)]).await; // dup p1.y=40
    let table = add_deletes(&catalog, &table, vec![p0d2, p1d2]).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 30]),
        "before: y=20 (part 0) and y=40 (part 1) masked"
    );
    let history_before = table.metadata().history().len();

    let result = RewritePositionDeleteFiles::new(table.clone())
        .min_input_files(2)
        .execute(&catalog)
        .await
        .unwrap();
    // Two groups (one per partition), each compacting two files into one.
    assert_eq!(result.rewritten_delete_files_count, 4, "4 files rewritten");
    assert_eq!(
        result.added_delete_files_count, 2,
        "one compacted file per partition group"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity per-partition after independent group compaction"
    );
    assert_eq!(
        count_pos(&live_delete_files(&reloaded).await),
        2,
        "exactly two compacted files (one per partition)"
    );
    // Two sequential group commits each append a snapshot log entry. (Advance is a re-apply
    // avoidance / RewriteDataFiles parity seam; history +2 pins multi-group commits, not that
    // CAS would fail without advance — `do_commit` refreshes a stale base on first attempt.)
    assert_eq!(
        reloaded.metadata().history().len(),
        history_before + 2,
        "two group commits must each produce a Replace snapshot"
    );
}

/// FILTER restriction. Two partitions, each with two pos-deletes. `filter(x == 0)` compacts ONLY the
/// partition-0 group; partition 1's pos-deletes are left untouched. Read identity holds throughout.
/// MIGRATED (size gate): this fixture is a TWO-file group, deliberately BELOW Java's
/// `MIN_INPUT_FILES_DEFAULT` of 5, because its subject is the user FILTER stage (C-005 S2) — not
/// admission. It therefore sets `.min_input_files(2)` explicitly, which is also acceptance item
/// 2: the floor is configuration, not a hard-coded constant. Removing that knob must RED this
/// test — it asserts post-`execute` SHAPE, never read identity alone.
#[tokio::test]
async fn test_filter_restricts_compacted_partitions() {
    use crate::expr::Reference;
    use crate::spec::Datum;

    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let p0 = write_data_file(&table, "p0.parquet", 0, &[(0, 10, 1), (0, 20, 2)]).await;
    let p1 = write_data_file(&table, "p1.parquet", 1, &[(1, 30, 3), (1, 40, 4)]).await;
    let p0_path = p0.file_path().to_string();
    let p1_path = p1.file_path().to_string();
    let table = append_files(&catalog, &table, vec![p0, p1]).await;

    let p0d1 = write_position_delete_file(&table, Some(0), &[(&p0_path, 1)]).await;
    let p1d1 = write_position_delete_file(&table, Some(1), &[(&p1_path, 1)]).await;
    let table = add_deletes(&catalog, &table, vec![p0d1, p1d1]).await;
    let p0d2 = write_position_delete_file(&table, Some(0), &[(&p0_path, 1)]).await;
    let p1d2 = write_position_delete_file(&table, Some(1), &[(&p1_path, 1)]).await;
    let table = add_deletes(&catalog, &table, vec![p0d2, p1d2]).await;

    let before = scan_y_values(&table).await;
    assert_eq!(before, HashSet::from([10, 30]));

    let result = RewritePositionDeleteFiles::new(table.clone())
        .min_input_files(2)
        .filter(Reference::new("x").equal_to(Datum::long(0)))
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(
        result.rewritten_delete_files_count, 2,
        "only the partition-0 group is compacted"
    );
    assert_eq!(result.added_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity preserved under filter"
    );
    // Partition 0 compacted (2 -> 1); partition 1 untouched (still 2). Total 3 live pos-deletes.
    assert_eq!(
        count_pos(&live_delete_files(&reloaded).await),
        3,
        "partition 0 compacted to 1; partition 1's two files remain"
    );
}

// =================================================================================================
// C-008 — THE FORMAT SKIP. `file_format() != Parquet` drops EVERY non-Parquet position delete, a
// FORK DIVERGENCE from Java's format-blind `BinPackRewritePositionDeletePlanner` (zero
// case-insensitive `FileFormat` / `PUFFIN` matches in its class file; Java's DV avoidance lives one
// level up, in the Spark action). `DataFileFormat` is a CLOSED four-variant enum
// (`spec/manifest/data_file.rs:387-396`) and the predicate reads exactly that field, so the four
// variants exhaust the domain:
//
//   Parquet -> KEPT.    Pinned by every admission test below.
//   Puffin  -> DROPPED, and now UNREACHABLE end to end (see the Puffin note below the table).
//   Orc     -> DROPPED. `test_non_parquet_position_deletes_skipped_at_collection_orc`.
//   Avro    -> DROPPED. `test_non_parquet_position_deletes_skipped_at_collection_avro`.
//
// Every one of the three DROPPED pins carries the SAME applied mutation — delete the
// `file_format() != Parquet` skip — and each is armed independently: five same-partition entries at
// the DEFAULT floor of five, so the mutated build forms one admissible bin.
//
// NON-REDUNDANT, MEASURED not argued. Four mutations were APPLIED against the whole lib suite
// (population 3378) and their failure sets recorded:
//
//   delete the skip entirely                       -> 3 RED: puffin + orc + avro, nothing else
//   exempt Orc    (`Parquet | Orc` in the skip)    -> 1 RED: orc only
//   exempt Avro   (`Parquet | Avro`)               -> 1 RED: avro only
//   exempt Puffin (`Parquet | Puffin`)             -> 1 RED: puffin only
//
// THE PUFFIN ELEMENT IS NOW UNKILLABLE, recorded rather than quietly counted. Since the V3 arm
// landed, `execute` dispatches a V3 table away before this skip runs, and V1/V2 cannot commit a
// Puffin position delete at all (`validate_delete_file_for_version` rejects a DV below V3). So no
// table reaches the skip's Puffin leg. It stays as defensive code against externally written
// metadata. `test_v3_deletion_vectors_are_not_compacted` now pins the V3 arm's honest zeros
// instead. The Orc and Avro elements are unaffected and still kill their own mutants.
//
// So each variant of the closed enum has a mutant that ONLY its own pin kills — none of the three
// is a duplicate of another, and no other test in the suite covers any of them.
//
// DISCLOSURE — C-036 NEEDS A THIRTEENTH RECIPE, and by its own rule ("a pin needing a tenth recipe
// is a finding") that is a finding, filed here rather than silently invented. All three pins below
// stand on a fixture the enumeration does not describe, and so do two PRE-EXISTING tests:
//
//   * RECIPE 13 — the DEFAULT-FLOOR ADMISSIBLE BIN. FIVE SUB-MIN position-delete entries in ONE
//     partition at the SHIPPED defaults, NO knobs; asserted by every measured size < the resolved
//     `min_file_size_bytes` and `min_input_files == 5`, so the five are all candidates, pack into
//     one bin and clear `enough_input_files` on their own.
//
// It is NOT recipe 8 (five IN-RANGE files, zero candidates, the candidate filter's existence pin)
// and not recipe 7 (four files behind four explicit knobs). It is the fixture a pin needs whenever
// the thing under test must be reached THROUGH admission at the default config rather than around
// it — which is why `test_admission_min_input_files_default_five_declines_four_admits_five` (G2)
// and `test_admitted_bin_with_zero_pairs_is_skipped` (G4) already built it before this group did.
// This extends G3's recipe 10/11/12 finding; the ledger edit is outside this group's fence and is
// carried as a hand-off.
//
// ADJUDICATION, recorded because two clauses name this fixture. C-008's enumeration names the
// Puffin pin `test_non_parquet_position_deletes_skipped_at_collection_puffin`, while C-014 element 7
// rules the PRE-EXISTING `test_v3_deletion_vectors_are_not_compacted` "reworked to FIVE Puffin DVs,
// NO KNOB". Both describe the SAME fixture (five Puffin DVs, one partition, default config) and the
// SAME pin form (the applied skip mutation), so they are ONE test, not two: a duplicate would share
// the single mutant and discriminate nothing. The pre-existing NAME is kept, because C-014's
// disposition table cites it and the V3-DV scope is the older, wider claim; C-008's Puffin element
// is discharged here and cross-referenced above, so neither scope loses its pin.
// =================================================================================================

/// CASE 1 — HONEST ZEROS. On a V3 table, FIVE data files in partition 0 are each masked by a Puffin
/// DELETION VECTOR. A DV is file-scoped, so there is nothing to bin-pack and nothing to convert: the
/// arm LOOKS at all five and returns zero counts with no commit. On the V3 arm those zeros are a
/// total statement — an input the arm cannot express is an `Err` — so a caller can tell "looked,
/// found nothing to do" from "did not look". The five DVs stay live and the read set is unchanged.
///
/// NO KNOB, deliberately (C-014 element 7). The previous form was a TWO-DV fixture carrying
/// `.min_input_files(2)`, and that knob was load-bearing only for the MUTATED build — under the
/// size gate a two-file group is below the floor, so the mutant that deletes the skip would have
/// been declined by admission rather than caught by this test. FIVE DVs at the shipped defaults
/// remove the knob and the dependency on it: the mutated build clears `enough_input_files` on its
/// own (5 > 1 and 5 >= 5), so this pin now measures the SKIP and nothing else.
///
/// MUTATION COVERAGE — APPLIED: classify a Puffin DV as a convertible legacy delete in
/// `admit_position_delete` and the arm tries to convert all five. RED. The former claim on the
/// `file_format() != Parquet` skip no longer holds — see the Puffin note in the C-008 banner above,
/// and deleting the version dispatch leaves this test GREEN because that skip still drops all five.
///
/// FIXTURE PRECONDITIONS asserted before `execute` (so a drifted fixture reds instead of passing
/// vacuously): five DVs in ONE partition, every one sub-min, and the resolved floor is FIVE.
#[tokio::test]
async fn test_v3_deletion_vectors_are_not_compacted() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V3).await;

    // FIVE data files in partition 0, each holding two rows; the DV below masks the SECOND.
    let mut data_paths = Vec::new();
    let mut files = Vec::new();
    for (index, y) in [10i64, 20, 30, 40, 50].into_iter().enumerate() {
        let file = write_data_file(&table, &format!("dv-target-{index}.parquet"), 0, &[
            (0, y, 1),
            (0, y + 1, 2),
        ])
        .await;
        data_paths.push(file.file_path().to_string());
        files.push(file);
    }
    let table = append_files(&catalog, &table, files).await;

    // FIVE Puffin DVs in the SAME partition 0 — one per data file, each masking its position 1.
    let mut table = table;
    for path in &data_paths {
        let dv = write_deletion_vector(&table, path, &[1]).await;
        table = add_deletes(&catalog, &table, vec![dv]).await;
    }

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 20, 30, 40, 50]),
        "before: each DV masks its data file's second row (y = 11, 21, 31, 41, 51)"
    );

    // FIXTURE PRECONDITIONS at the DEFAULT config — the mutated build must be ADMITTED, or the
    // skip is not what this test measures.
    let action = || RewritePositionDeleteFiles::new(table.clone());
    let config = action().resolve_config().expect("the defaults are legal");
    let dvs = live_delete_files(&table).await;
    assert_eq!(dvs.len(), 5, "fixture: FIVE live deletion vectors");
    assert!(
        dvs.iter()
            .all(|f| f.file_format() == DataFileFormat::Puffin),
        "fixture: every live delete is a Puffin DV"
    );
    assert!(
        dvs.iter()
            .all(|f| f.content_type() == DataContentType::PositionDeletes),
        "fixture NON-VACUITY: a DV carries content PositionDeletes, so it clears the CONTENT \
         filter and reaches the FORMAT skip — this test would prove nothing if it were dropped one \
         line earlier"
    );
    assert!(
        dvs.iter()
            .all(|f| f.file_size_in_bytes < config.min_file_size_bytes),
        "fixture: every DV is SUB-MIN, so the mutated build makes all five candidates"
    );
    assert_eq!(
        config.min_input_files, 5,
        "fixture: FIVE files clear the DEFAULT floor with no knob — the literal, so a moved \
         constant reds here rather than silently re-shaping the fixture"
    );

    let snapshot_before = table.metadata().current_snapshot_id();
    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "DVs are NOT compacted by this action — zero counts, no commit, even at five same-partition DVs"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "post-execute SHAPE: a skipped group must NOT commit a new snapshot"
    );
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity: the DVs are untouched, the live set unchanged"
    );
    // All five Puffin DVs are still live, and none became a parquet pos-delete.
    let deletes = live_delete_files(&reloaded).await;
    assert_eq!(deletes.len(), 5, "all five DVs remain live");
    assert!(
        deletes
            .iter()
            .all(|f| f.file_format() == DataFileFormat::Puffin),
        "every surviving delete is a Puffin DV (none was compacted into a parquet pos-delete)"
    );
}

/// Write a single-data-file Puffin DELETION VECTOR masking the given absolute positions of `target_path`,
/// in partition x=0. Uses the [`DVFileWriter`] (the same writer the DV write path uses), so the produced
/// `DeleteFile` is a faithful Puffin DV the scan applies.
async fn write_deletion_vector(table: &Table, target_path: &str, positions: &[u64]) -> DataFile {
    use crate::writer::base_writer::deletion_vector_writer::DVFileWriter;

    let dv_path = format!(
        "{}/data/dv-{}.puffin",
        table.metadata().location(),
        uuid::Uuid::now_v7()
    );
    let output = table.file_io().new_output(&dv_path).unwrap();
    let partition_key = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        Struct::from_iter([Some(Literal::long(0))]),
    )
    .expect("PartitionKey::new: valid partition tuple");
    let mut writer = DVFileWriter::new(output);
    for &pos in positions {
        writer
            .delete(target_path, pos, Some(&partition_key))
            .expect("record DV position");
    }
    writer
        .close()
        .await
        .expect("close DV writer")
        .into_iter()
        .next()
        .expect("one DV delete file")
}

/// Commit FIVE live position-delete manifest entries in partition 0 whose `file_format` is
/// `format` — the fixture for C-008's V2 ORC and Avro elements — and return their measured sizes.
///
/// WHY FABRICATED, AND WHY THAT IS SOUND. The format skip fires during the manifest walk, BEFORE
/// any file is opened (`collect_position_delete_groups` reads `file_format()` off the entry and
/// `continue`s; nothing between the walk and the skip touches the bytes). So the pin needs a
/// manifest ENTRY that says ORC/Avro, not an ORC/Avro encoder the fork does not have. Each entry is
/// produced by a real [`write_position_delete_file`] write — i.e. through the writer's own
/// `DataFileBuilder`, so every required field is populated exactly as a genuine entry's would be —
/// and only the `file_format` field is re-stamped. The bytes on disk stay PARQUET on purpose: under
/// the applied mutation (skip deleted) the entries are then read successfully and COMPACTED, so the
/// mutant reds on this test's own count/shape assertions rather than on an IO error. That is the
/// stronger red — it proves the SKIP is what drops these files, not a failed read.
///
/// The five entries mask the same `(target_path, pos)` pair, which keeps them the same size class
/// and makes the group's composition the only thing that varies between the Orc and Avro pins.
async fn add_fabricated_non_parquet_pos_deletes(
    catalog: &impl Catalog,
    table: &Table,
    target_path: &str,
    format: DataFileFormat,
) -> (Table, Vec<u64>) {
    assert_ne!(
        format,
        DataFileFormat::Parquet,
        "helper misuse: this fixture exists to build NON-Parquet entries"
    );
    let mut files = Vec::new();
    for _ in 0..5 {
        let mut file = write_position_delete_file(table, Some(0), &[(target_path, 1)]).await;
        file.file_format = format;
        files.push(file);
    }
    let sizes: Vec<u64> = files.iter().map(|f| f.file_size_in_bytes).collect();
    let table = add_deletes(catalog, table, files).await;
    (table, sizes)
}

/// The shared body of C-008's ORC and Avro elements: FIVE live position-delete entries in one
/// partition whose `file_format` is `format` are dropped at collection, so the action is a no-op.
///
/// FIXTURE PRECONDITIONS asserted before `execute`: all five entries live and carrying `format`,
/// content `PositionDeletes` (so they clear the CONTENT filter and actually REACH the format skip),
/// every one sub-min, and the resolved floor is FIVE — i.e. the mutated build forms exactly one
/// ADMISSIBLE bin. Without those the pin could pass because the group was declined on count.
///
/// A post-`execute` SCAN is deliberately NOT asserted, and the omission is not a gap: the table's
/// metadata now claims five ORC/Avro delete files the scan has no reader for, so a read would fail
/// for a reason that has nothing to do with this action. The post-`execute` signal is SHAPE —
/// zero counts, no new snapshot, all five entries still live and still carrying `format`.
///
/// MUTATION COVERAGE — APPLIED, not predicted: delete the `file_format() != Parquet` skip and the
/// five entries become one admissible five-file bin, are read (the bytes really are parquet),
/// compacted into one file and committed. RED on the counts, the snapshot id and the live set.
async fn assert_non_parquet_pos_deletes_are_skipped(format: DataFileFormat) {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 1), (0, 20, 2)]).await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await;

    let (table, sizes) =
        add_fabricated_non_parquet_pos_deletes(&catalog, &table, &x_path, format).await;

    let action = || RewritePositionDeleteFiles::new(table.clone());
    let config = action().resolve_config().expect("the defaults are legal");
    let live = live_delete_files(&table).await;
    assert_eq!(live.len(), 5, "fixture: FIVE live delete entries");
    assert!(
        live.iter().all(|f| f.file_format() == format),
        "fixture: every live delete entry carries {format:?}"
    );
    assert!(
        live.iter()
            .all(|f| f.content_type() == DataContentType::PositionDeletes),
        "fixture NON-VACUITY: the entries are POSITION DELETES, so they pass the CONTENT filter \
         and reach the FORMAT skip — dropped one line earlier they would prove nothing"
    );
    assert!(
        sizes.iter().all(|s| *s < config.min_file_size_bytes),
        "fixture: every entry is SUB-MIN, so the mutated build makes all five candidates"
    );
    assert_eq!(
        config.min_input_files, 5,
        "fixture: FIVE entries clear the DEFAULT floor with no knob — the mutated build forms ONE \
         ADMISSIBLE bin, which is what arms this pin"
    );

    let snapshot_before = table.metadata().current_snapshot_id();
    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "{format:?} position deletes are dropped at collection — zero counts, no commit"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "post-execute SHAPE: a skipped group must NOT commit a new snapshot"
    );
    let after = live_delete_files(&reloaded).await;
    assert_eq!(after.len(), 5, "all five entries remain live");
    assert!(
        after.iter().all(|f| f.file_format() == format),
        "no {format:?} entry was rewritten into a parquet position delete"
    );
}

/// C-008, `Orc` element. A V2 table carrying FIVE live ORC position-delete entries in one partition
/// is left completely alone: the `file_format() != Parquet` skip drops them at collection.
///
/// THIS IS THE DIVERGENCE, not a parity claim. Java's `BinPackRewritePositionDeletePlanner` is
/// format-blind and WOULD compact these five; the fork cannot, because `compact_group` reads
/// through the parquet reader and writes through `ParquetWriterBuilder`. Recorded in
/// `docs/parity/GAP_MATRIX.md` row R136 and at the skip itself.
#[tokio::test]
async fn test_non_parquet_position_deletes_skipped_at_collection_orc() {
    assert_non_parquet_pos_deletes_are_skipped(DataFileFormat::Orc).await;
}

/// C-008, `Avro` element — the same fixture and the same applied mutation as the ORC pin, over the
/// last of `DataFileFormat`'s four variants. Both are listed because the skip is a single
/// inequality against `Parquet`: a future implementation that special-cased ORC (say, by adding an
/// ORC reader) would keep one pin green and red the other.
#[tokio::test]
async fn test_non_parquet_position_deletes_skipped_at_collection_avro() {
    assert_non_parquet_pos_deletes_are_skipped(DataFileFormat::Avro).await;
}

// =================================================================================================
// NO-OP edges. The SINGLE-FILE no-op moved into C-021's size-class trio (see
// `test_admission_sub_min_single_file_is_declined`): a lone position-delete file is a no-op only in
// size classes 1 and 2 — class 3, above `max_file_size`, is ADMITTED and SPLIT.
// =================================================================================================

/// No current snapshot → no-op, zero counts, no commit.
#[tokio::test]
async fn test_no_current_snapshot_is_a_no_op() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let result = RewritePositionDeleteFiles::new(table)
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(result, RewritePositionDeleteFilesResult::default());
}

/// Unpartitioned table: two pos-delete files in the single unpartitioned group compact into one. Read
/// identity holds.
/// MIGRATED (size gate): this fixture is a TWO-file group, deliberately BELOW Java's
/// `MIN_INPUT_FILES_DEFAULT` of 5, because its subject is the UNPARTITIONED group key — not
/// admission. It therefore sets `.min_input_files(2)` explicitly, which is also acceptance item
/// 2: the floor is configuration, not a hard-coded constant. Removing that knob must RED this
/// test — it asserts post-`execute` SHAPE, never read identity alone.
#[tokio::test]
async fn test_unpartitioned_group_compacts() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_unpartitioned_table(&catalog, FormatVersion::V2).await;

    let a = write_data_file(&table, "a.parquet", 0, &[(1, 10, 1), (2, 20, 2)]).await;
    let b = write_data_file(&table, "b.parquet", 0, &[(3, 30, 3), (4, 40, 4)]).await;
    let a_path = a.file_path().to_string();
    let b_path = b.file_path().to_string();
    let table = append_files(&catalog, &table, vec![a, b]).await;

    let pda = write_position_delete_file(&table, None, &[(&a_path, 1)]).await; // a.y=20
    let table = add_deletes(&catalog, &table, vec![pda]).await;
    let pdb = write_position_delete_file(&table, None, &[(&b_path, 0)]).await; // b.y=30
    let table = add_deletes(&catalog, &table, vec![pdb]).await;

    let before = scan_y_values(&table).await;
    assert_eq!(before, HashSet::from([10, 40]));

    let result = RewritePositionDeleteFiles::new(table.clone())
        .min_input_files(2)
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(result.rewritten_delete_files_count, 2);
    assert_eq!(result.added_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity (unpartitioned)"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 1);
}

// =================================================================================================
// CONFIG SEAM — the size/count thresholds, their Java defaults, the delete-specific property, the
// `Long.parseLong`-exact parse, and Java's `sizeThresholds` preconditions.
//
// Every default here is asserted WHITE-BOX through `resolve_config()` (private, but reachable — this
// is a child module), never end to end: observing 50331648 / 120795955 end to end would need ~48 MiB
// and ~115 MiB fixtures. These pins are therefore named `test_config_*` / `test_resolve_config_*` /
// `test_parse_delete_target_*` and NEVER `test_admission_*`, so no later change lowers the target to
// make them end-to-end and thereby unpins the ratio constants.
// =================================================================================================

/// A minimal unpartitioned table carrying `properties` — the fixture for the white-box config pins.
/// The config seam reads only `metadata().properties()`, so the table needs no snapshot and no files.
async fn config_table(properties: &[(&str, &str)]) -> (TempDir, Table) {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let schema = three_long_schema();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .build()
        .expect("build spec");
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, std::collections::HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("t".to_string())
        .schema(schema)
        .partition_spec(spec)
        .format_version(FormatVersion::V2)
        .properties(
            properties
                .iter()
                .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
                .collect::<Vec<_>>(),
        )
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table");
    (temp_dir, table)
}

/// `write.delete.target-file-size-bytes` as a `HashMap`, for the pins that exercise the parse function
/// alone (no table needed).
fn delete_target_property(value: &str) -> std::collections::HashMap<String, String> {
    std::collections::HashMap::from([(
        TableProperties::PROPERTY_WRITE_DELETE_TARGET_FILE_SIZE_BYTES.to_string(),
        value.to_string(),
    )])
}

// ------------------------------------------------------------------------------------------------
// C-035 — the parse: one function whose accept/reject domain is `Long.parseLong`'s, EXACTLY.
// ------------------------------------------------------------------------------------------------

/// C-035 element 1 (ABSENT). No property ⇒ the 64 MiB delete default.
#[test]
fn test_parse_delete_target_absent_is_64_mib() {
    let properties = std::collections::HashMap::new();
    assert_eq!(
        parse_delete_target_file_size(&properties).expect("an absent property takes the default"),
        67108864
    );
}

/// C-035 element 2 (a well-formed decimal in `[2, i64::MAX - 1]`) — the only class that can also
/// SURVIVE the preconditions. A leading `+` is accepted, matching `Long.parseLong`.
#[test]
fn test_parse_delete_target_mid_band_is_accepted() {
    assert_eq!(
        parse_delete_target_file_size(&delete_target_property("12345678")).expect("mid band"),
        12345678
    );
    assert_eq!(
        parse_delete_target_file_size(&delete_target_property("+12345678"))
            .expect("Long.parseLong accepts a leading '+', and so must this"),
        12345678
    );
}

/// C-035 element 3, upper endpoint. `i64::MAX` PARSES (this is a parse-function assertion only —
/// whether it then survives the preconditions is `resolve_config`'s business, pinned separately by
/// `test_resolve_config_rejects_target_at_i64_max_on_the_max_precondition`).
#[test]
fn test_parse_delete_target_at_i64_max_is_accepted_by_the_parse() {
    assert_eq!(
        parse_delete_target_file_size(&delete_target_property("9223372036854775807"))
            .expect("Long.parseLong accepts Long.MAX_VALUE"),
        i64::MAX
    );
}

/// C-035 element 3, lower endpoint. `"1"` PARSES (parse-function assertion only).
#[test]
fn test_parse_delete_target_at_one_is_accepted_by_the_parse() {
    assert_eq!(
        parse_delete_target_file_size(&delete_target_property("1")).expect("'1' parses"),
        1
    );
}

/// C-035 element 6 (unparsable / empty). Every one of these throws `NumberFormatException` in Java,
/// so every one must be a `DataInvalid` here — including the forms Rust's parser could plausibly have
/// accepted (`1_000` is a Rust literal, not a `Long.parseLong` input).
#[test]
fn test_parse_delete_target_unparsable_is_data_invalid() {
    for value in ["", "abc", "1_000", " 12", "12 ", "0x10", "1.0", "12,3", "+"] {
        let error = parse_delete_target_file_size(&delete_target_property(value))
            .expect_err("an unparsable delete target must be rejected loudly");
        assert_eq!(error.kind(), ErrorKind::DataInvalid, "value {value:?}");
        assert_eq!(
            error.message(),
            format!(
                "Invalid value '{value}' for table property 'write.delete.target-file-size-bytes'"
            ),
            "value {value:?}"
        );
    }
}

/// C-035 element 6 (magnitude outside the `long` range) — THE anti-`u64` pin. `Long.parseLong` throws
/// on anything above `Long.MAX_VALUE`, so these must be rejected AT THE PARSE. A `u64` parse would
/// accept every one of them and hand the action a threshold Java cannot express.
#[test]
fn test_parse_delete_target_above_i64_max_is_rejected() {
    for value in [
        "9223372036854775808",  // i64::MAX + 1
        "18446744073709551615", // u64::MAX — accepted by a u64 parse, rejected by Long.parseLong
        "99999999999999999999999",
    ] {
        let error = parse_delete_target_file_size(&delete_target_property(value))
            .expect_err("a value outside Java's long range must be rejected");
        assert_eq!(error.kind(), ErrorKind::DataInvalid, "value {value:?}");
    }
    // ... and the boundary directly below it still parses, so the rejection is two-sided.
    assert_eq!(
        parse_delete_target_file_size(&delete_target_property("9223372036854775807"))
            .expect("i64::MAX itself parses"),
        i64::MAX
    );
}

/// C-035 element 4 (`"0"`). It PARSES — and is then rejected downstream by precondition (1) carrying
/// Java's verbatim `'%s' is set to %s but must be > 0`. A `u64` parse would also parse it, so the
/// discriminating half is the message.
#[tokio::test]
async fn test_parse_delete_target_zero_is_rejected_by_the_target_precondition() {
    assert_eq!(
        parse_delete_target_file_size(&delete_target_property("0")).expect("'0' parses"),
        0
    );

    let (_temp, table) = config_table(&[("write.delete.target-file-size-bytes", "0")]).await;
    let error = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect_err("a zero target must be rejected");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "'target-file-size-bytes' is set to 0 but must be > 0"
    );
}

/// C-035 element 5 (a NEGATIVE decimal). `"-1"` PARSES as `i64` exactly as `Long.parseLong` parses it
/// — a `u64` parse would reject it at the parse with a fork-only message — and is then rejected by
/// precondition (1).
///
/// This is also the ORDER pin for C-006: at `target = -1` this port's `d2l` saturates BOTH ratio
/// products to `0`, so `min = 0` and precondition (3) `target > min` (`-1 > 0`) is INDEPENDENTLY
/// false. Hoisting (3) above (1) therefore reports the min-threshold message instead, and this
/// assertion reds.
#[tokio::test]
async fn test_parse_delete_target_negative_is_rejected_by_the_target_precondition() {
    assert_eq!(
        parse_delete_target_file_size(&delete_target_property("-1"))
            .expect("'-1' parses as i64, exactly as Long.parseLong parses it"),
        -1
    );

    let (_temp, table) = config_table(&[("write.delete.target-file-size-bytes", "-1")]).await;
    let error = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect_err("a negative target must be rejected");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "'target-file-size-bytes' is set to -1 but must be > 0"
    );
}

// ------------------------------------------------------------------------------------------------
// C-002 / C-032 — the target resolves from the DELETE property, never from the data-file one.
// ------------------------------------------------------------------------------------------------

/// C-002. No property ⇒ 64 MiB (Java `defaultTargetFileSize`'s `ldc2_w 67108864L`).
#[tokio::test]
async fn test_config_target_file_size_default_is_64_mib() {
    let (_temp, table) = config_table(&[]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect("the defaults satisfy every precondition");
    assert_eq!(config.target_file_size_bytes, 67108864);
}

/// C-002. The DELETE-specific property moves the target.
#[tokio::test]
async fn test_config_target_file_size_resolves_delete_property() {
    let (_temp, table) = config_table(&[("write.delete.target-file-size-bytes", "12345678")]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect("12345678 satisfies every precondition");
    assert_eq!(config.target_file_size_bytes, 12345678);
    // The ratios follow the resolved target, not the default.
    assert_eq!(config.min_file_size_bytes, 9259258);
    assert_eq!(config.max_file_size_bytes, 22222220);
}

/// C-002, the NEGATIVE CONTROL. `write.target-file-size-bytes` is the DATA-file target (512 MiB) and
/// must not move the delete target by so much as a byte — Java's
/// `BinPackRewritePositionDeletePlanner.defaultTargetFileSize` never reads it. Reds if the port
/// resolves from `TableProperties::PROPERTY_WRITE_TARGET_FILE_SIZE_BYTES`.
#[tokio::test]
async fn test_config_write_target_file_size_does_not_move_delete_target() {
    let (_temp, table) = config_table(&[("write.target-file-size-bytes", "536870912")]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect("the delete defaults still hold");
    assert_eq!(
        config.target_file_size_bytes, 67108864,
        "the data-file target must not leak into the delete target"
    );
    assert_ne!(config.target_file_size_bytes, 536870912);
}

/// C-001. The builder wins over the property.
#[tokio::test]
async fn test_config_target_file_size_builder_overrides_property() {
    let (_temp, table) = config_table(&[("write.delete.target-file-size-bytes", "12345678")]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .target_file_size_bytes(99_999_999)
        .resolve_config()
        .expect("99999999 satisfies every precondition");
    assert_eq!(config.target_file_size_bytes, 99_999_999);
}

// ------------------------------------------------------------------------------------------------
// C-007 — the ratio defaults through the shared `d2l` helper.
// ------------------------------------------------------------------------------------------------

/// C-007. `min = d2l(target * 0.75)`; at the 64 MiB delete default that is EXACTLY 50331648 (dyadic,
/// so no rounding is involved). Asserted as a literal so the ratio constant cannot drift.
#[tokio::test]
async fn test_config_min_file_size_default_is_three_quarters_target() {
    let (_temp, table) = config_table(&[]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect("the defaults satisfy every precondition");
    assert_eq!(config.min_file_size_bytes, 50331648);
}

/// C-007. `max = d2l(target * 1.8)`; at the 64 MiB delete default `1.8 * 2^26 = 120795955.2`, which
/// Java's `d2l` TRUNCATES to 120795955. Asserted as a literal, so both the ratio and the truncation
/// (vs rounding) are pinned.
#[tokio::test]
async fn test_config_max_file_size_default_is_one_point_eight_target() {
    let (_temp, table) = config_table(&[]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect("the defaults satisfy every precondition");
    assert_eq!(config.max_file_size_bytes, 120795955);
}

/// C-007, THE CLAMP. Java's `d2l` saturates at `Long.MAX_VALUE`; Rust's `as u64` saturates at
/// `u64::MAX`, so `.min(i64::MAX as u64)` is the parity act.
///
/// The window in which it is OBSERVABLE is `target ∈ (2^63 / 1.8, i64::MAX - 1]`: there the clamped
/// max IS `i64::MAX` while the unclamped one is larger, and BOTH are above the target, so
/// `resolve_config` returns `Ok` either way and the equality is the discriminator rather than the
/// Ok/Err split. `6e18` sits inside it.
#[tokio::test]
async fn test_config_max_file_size_clamps_to_java_long_max() {
    const TARGET: u64 = 6_000_000_000_000_000_000;
    // Fixture preconditions: inside the observable window, and the unclamped product really does
    // exceed Java's ceiling (otherwise this test would pass with the clamp deleted).
    assert!(
        (TARGET as f64) > (i64::MAX as f64) / 1.8,
        "fixture: the target must sit above 2^63 / 1.8"
    );
    let unclamped = (TARGET as f64 * 1.8) as u64;
    assert!(
        unclamped > i64::MAX as u64,
        "fixture: the UNCLAMPED product must exceed Long.MAX_VALUE, else the clamp is unobservable"
    );

    let (_temp, table) = config_table(&[]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .target_file_size_bytes(TARGET)
        .resolve_config()
        .expect("target < clamped max, so this resolves");
    assert_eq!(
        config.max_file_size_bytes,
        i64::MAX as u64,
        "d2l must saturate at Long.MAX_VALUE, not at u64::MAX"
    );
    assert_ne!(config.max_file_size_bytes, unclamped);
}

// ------------------------------------------------------------------------------------------------
// C-001 — the remaining two ported options and their defaults.
// ------------------------------------------------------------------------------------------------

/// C-001. `min_input_files` defaults to Java's `MIN_INPUT_FILES_DEFAULT = 5`. Asserted with
/// `assert_eq!` so it reds on a raise AND on a lower.
///
/// TODO(tests): the two-sided ADMISSION pin — four files declined, five admitted, at the DEFAULT
/// config — belongs to the gate, not to this config seam, and is owed by the test increment.
#[tokio::test]
async fn test_config_min_input_files_default_is_five() {
    let (_temp, table) = config_table(&[]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect("the defaults satisfy every precondition");
    assert_eq!(config.min_input_files, 5);
}

/// C-001. `max_file_group_size_bytes` defaults to Java's
/// `MAX_FILE_GROUP_SIZE_BYTES_DEFAULT = 107374182400` (100 GiB).
#[tokio::test]
async fn test_config_max_file_group_size_default_is_100_gib() {
    let (_temp, table) = config_table(&[]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect("the defaults satisfy every precondition");
    assert_eq!(config.max_file_group_size_bytes, 107374182400);
    assert_eq!(config.max_file_group_size_bytes, 100 * 1024 * 1024 * 1024);
}

/// C-001. All five ported builders land on the resolved config — one assertion per option, so a
/// builder wired to the wrong field reds.
#[tokio::test]
async fn test_config_explicit_overrides_are_returned_unchanged() {
    let (_temp, table) = config_table(&[]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .target_file_size_bytes(200_000)
        .min_file_size_bytes(100_000)
        .max_file_size_bytes(400_000)
        .min_input_files(2)
        .max_file_group_size_bytes(1_000_000)
        .resolve_config()
        .expect("a legal knob combination");
    assert_eq!(config.target_file_size_bytes, 200_000);
    assert_eq!(config.min_file_size_bytes, 100_000);
    assert_eq!(config.max_file_size_bytes, 400_000);
    assert_eq!(config.min_input_files, 2);
    assert_eq!(config.max_file_group_size_bytes, 1_000_000);
}

// ------------------------------------------------------------------------------------------------
// C-006 — Java's `sizeThresholds` preconditions, each with Java's verbatim message.
// ------------------------------------------------------------------------------------------------

/// C-006 precondition (1), via the builder path.
#[tokio::test]
async fn test_resolve_config_rejects_target_zero() {
    let (_temp, table) = config_table(&[]).await;
    let error = RewritePositionDeleteFiles::new(table)
        .target_file_size_bytes(0)
        .resolve_config()
        .expect_err("a zero target must be rejected");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "'target-file-size-bytes' is set to 0 but must be > 0"
    );
}

/// C-006 precondition (1) at a NEGATIVE target — the ORDER-discriminating pin. See
/// `test_parse_delete_target_negative_is_rejected_by_the_target_precondition` for why `min` is `0`
/// here and why that makes precondition (3) independently false.
#[tokio::test]
async fn test_resolve_config_rejects_target_negative_with_the_must_be_positive_message() {
    let (_temp, table) = config_table(&[("write.delete.target-file-size-bytes", "-7")]).await;
    let error = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect_err("a negative target must be rejected");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "'target-file-size-bytes' is set to -7 but must be > 0",
        "precondition (1) must report first: at a negative target the saturating d2l makes min = 0, \
         so (3) `target > min` is independently false and a hoisted (3) would report instead"
    );
}

/// C-006 precondition (3), STRICT: `target == min` is rejected, `target == min + 1` is not.
#[tokio::test]
async fn test_resolve_config_rejects_target_le_min() {
    let (_temp, table) = config_table(&[]).await;
    let error = RewritePositionDeleteFiles::new(table.clone())
        .target_file_size_bytes(200_000)
        .min_file_size_bytes(200_000)
        .max_file_size_bytes(400_000)
        .resolve_config()
        .expect_err("target == min must be rejected (STRICT >)");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "'target-file-size-bytes' (200000) must be > 'min-file-size-bytes' (200000), all new files \
         will be smaller than the min threshold"
    );

    // Two-sided: one byte above the min is legal, so the pin is not vacuous.
    RewritePositionDeleteFiles::new(table)
        .target_file_size_bytes(200_001)
        .min_file_size_bytes(200_000)
        .max_file_size_bytes(400_000)
        .resolve_config()
        .expect("target == min + 1 is legal");
}

/// C-006 precondition (4), STRICT: `target == max` is rejected, `target == max - 1` is not.
#[tokio::test]
async fn test_resolve_config_rejects_target_ge_max() {
    let (_temp, table) = config_table(&[]).await;
    let error = RewritePositionDeleteFiles::new(table.clone())
        .target_file_size_bytes(400_000)
        .min_file_size_bytes(100_000)
        .max_file_size_bytes(400_000)
        .resolve_config()
        .expect_err("target == max must be rejected (STRICT <)");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "'target-file-size-bytes' (400000) must be < 'max-file-size-bytes' (400000), all new files \
         will be larger than the max threshold"
    );

    // Two-sided: one byte below the max is legal.
    RewritePositionDeleteFiles::new(table)
        .target_file_size_bytes(399_999)
        .min_file_size_bytes(100_000)
        .max_file_size_bytes(400_000)
        .resolve_config()
        .expect("target == max - 1 is legal");
}

/// C-006 precondition (5).
#[tokio::test]
async fn test_resolve_config_rejects_min_input_files_zero() {
    let (_temp, table) = config_table(&[]).await;
    let error = RewritePositionDeleteFiles::new(table.clone())
        .min_input_files(0)
        .resolve_config()
        .expect_err("min_input_files = 0 must be rejected");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "'min-input-files' is set to 0 but must be > 0"
    );

    RewritePositionDeleteFiles::new(table)
        .min_input_files(1)
        .resolve_config()
        .expect("min_input_files = 1 is legal");
}

/// C-006 precondition (6).
#[tokio::test]
async fn test_resolve_config_rejects_max_file_group_size_zero() {
    let (_temp, table) = config_table(&[]).await;
    let error = RewritePositionDeleteFiles::new(table.clone())
        .max_file_group_size_bytes(0)
        .resolve_config()
        .expect_err("max_file_group_size_bytes = 0 must be rejected");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "'max-file-group-size-bytes' is set to 0 but must be > 0"
    );

    RewritePositionDeleteFiles::new(table)
        .max_file_group_size_bytes(1)
        .resolve_config()
        .expect("max_file_group_size_bytes = 1 is legal");
}

/// C-006 precondition (7) — ONE LEG PER KNOB. Java's thresholds are `long`s fed by `Long.parseLong`,
/// so an override above `i64::MAX` is a config Java cannot express; admitting it would open a state
/// in which `too_much_content` (`input_size > max`) is unreachable for every possible input.
///
/// Each leg sets exactly ONE over-range knob and asserts that knob's own message, which is what
/// forces (7) to run at override-READ time: checked after (3)/(4) instead, the `min` leg would be
/// caught by (3) and the `target` leg by (4), both with the wrong message.
#[tokio::test]
async fn test_resolve_config_rejects_size_override_above_i64_max() {
    const ABOVE: u64 = i64::MAX as u64 + 1;
    let (_temp, table) = config_table(&[]).await;

    for (option, action) in [
        (
            "target-file-size-bytes",
            RewritePositionDeleteFiles::new(table.clone()).target_file_size_bytes(ABOVE),
        ),
        (
            "min-file-size-bytes",
            RewritePositionDeleteFiles::new(table.clone()).min_file_size_bytes(ABOVE),
        ),
        (
            "max-file-size-bytes",
            RewritePositionDeleteFiles::new(table.clone()).max_file_size_bytes(ABOVE),
        ),
    ] {
        let error = action
            .resolve_config()
            .expect_err("an override above i64::MAX must be rejected");
        assert_eq!(error.kind(), ErrorKind::DataInvalid, "option {option}");
        assert_eq!(
            error.message(),
            format!(
                "Invalid value '{ABOVE}' for '{option}': it must be <= 9223372036854775807 — \
                 Java's option is a `long`, so a larger threshold has no Java analogue"
            ),
            "option {option}"
        );
    }

    // Two-sided: `i64::MAX` itself is accepted BY (7) on every knob (the `max` leg then resolves
    // cleanly; the other two are rejected later, by (3)/(4), which is a different message).
    RewritePositionDeleteFiles::new(table)
        .max_file_size_bytes(i64::MAX as u64)
        .resolve_config()
        .expect("i64::MAX is inside Java's long domain");
}

/// C-035 element 3 downstream. `i64::MAX` PARSES, then falls to precondition (4): its defaulted max
/// CLAMPS to `i64::MAX`, so `target < max` is false. Asserts `Err` + the message, never a resolved
/// value.
#[tokio::test]
async fn test_resolve_config_rejects_target_at_i64_max_on_the_max_precondition() {
    let (_temp, table) =
        config_table(&[("write.delete.target-file-size-bytes", "9223372036854775807")]).await;
    let error = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect_err("target == the clamped max must be rejected");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "'target-file-size-bytes' (9223372036854775807) must be < 'max-file-size-bytes' \
         (9223372036854775807), all new files will be larger than the max threshold"
    );
}

// =================================================================================================
// THE ADMISSION GATE — the candidate filter (C-004), the six-stage pipeline order (C-005), the
// three-clause group filter (C-003), the shared bin packer (C-027) and the saturating input sum
// (C-041).
//
// FIXTURE DISCIPLINE (C-036), applied without exception below: a size-class fixture is NEVER sized
// to fixed knobs. The position-delete file is written first, its `file_size_in_bytes` is MEASURED
// off the returned `DataFile`, and the knobs are then set AROUND that measurement. Every size-class
// test asserts its measured size against the RESOLVED thresholds BEFORE `execute`, so a fixture that
// drifts out of its size class fails loudly instead of passing vacuously. Where a knob derivation
// makes a window assert an algebraic identity in the fixture size, that is RECORDED as true by
// construction rather than dressed up as a check.
// =================================================================================================

/// Write a position-delete file masking `count` consecutive positions of `target_path` starting at
/// `first_pos`, in partition 0. This is the SIZE DIAL for the size-class fixtures: more masked
/// positions ⇒ a strictly larger file (the `file_path` column dictionary-encodes to one value, so the
/// growth is the `pos` column). The size is never predicted — it is measured off the return value.
async fn write_sized_pos_delete(
    table: &Table,
    target_path: &str,
    first_pos: i64,
    count: i64,
) -> DataFile {
    let pairs: Vec<(&str, i64)> = (first_pos..first_pos + count)
        .map(|pos| (target_path, pos))
        .collect();
    write_position_delete_file(table, Some(0), &pairs).await
}

/// A partitioned table with ONE five-row data file (y = 10,20,30,40,50 at positions 0..5) already
/// committed — the base every gate fixture below builds its position-delete files on. Returns the
/// catalog, the temp dir (kept alive), the table and the data file's path.
async fn gate_table() -> (impl Catalog, TempDir, Table, String) {
    let (catalog, temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let x = write_data_file(&table, "x.parquet", 0, &[
        (0, 10, 100),
        (0, 20, 200),
        (0, 30, 300),
        (0, 40, 400),
        (0, 50, 500),
    ])
    .await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await;
    (catalog, temp, table, x_path)
}

/// The live position-delete files in MANIFEST ORDER — the same order
/// `collect_position_delete_groups` walks, so a fixture whose packing depends on order can assert
/// the order it actually gets instead of assuming it.
async fn live_pos_delete_paths(table: &Table) -> Vec<String> {
    live_delete_files(table)
        .await
        .iter()
        .filter(|f| f.content_type() == DataContentType::PositionDeletes)
        .map(|f| f.file_path().to_string())
        .collect()
}

// ------------------------------------------------------------------------------------------------
// C-003 element 1 — `enough_input_files`'s `size > 1` conjunct.
// ------------------------------------------------------------------------------------------------

/// C-003 element 1. `.min_input_files(1)` with a SINGLE sub-min position-delete file. The file IS a
/// candidate (sub-min) and forms a bin of one, so the gate is reached; `enough_input_files` is
/// `size > 1 && size >= min_input_files`, and only the `size > 1` conjunct declines it — `1 >= 1`
/// holds. `enough_content` is false for the same reason, and the file is far below `max`, so
/// `too_much_content` is false too.
///
/// MUTATION COVERAGE: this is the ONLY killer of `enough_input_files`'s `size > 1` conjunct. Drop it
/// and the bin is admitted, producing a Replace snapshot and non-zero counts.
#[tokio::test]
async fn test_admission_min_input_files_one_still_declines_lone_sub_min_file() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    let pd = write_sized_pos_delete(&table, &x_path, 1, 1).await;
    let size = pd.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd]).await;

    let action = || RewritePositionDeleteFiles::new(table.clone()).min_input_files(1);
    let config = action()
        .resolve_config()
        .expect("the defaults plus min_input_files(1) are legal");

    // PRECONDITIONS (measured against the RESOLVED thresholds, before execute).
    assert!(
        size < config.min_file_size_bytes,
        "fixture: the lone file must be SUB-MIN so it is a candidate and reaches the gate \
         (measured {size}, resolved min {})",
        config.min_file_size_bytes
    );
    assert!(
        size <= config.max_file_size_bytes,
        "fixture: too_much_content must NOT fire, or the pin would pass for the wrong reason"
    );
    assert_eq!(
        config.min_input_files, 1,
        "fixture: the floor is lowered to 1"
    );

    let before = scan_y_values(&table).await;
    let snapshot_before = table.metadata().current_snapshot_id();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "a bin of ONE is declined by `size > 1` even at min_input_files(1)"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "a declined bin must NOT commit"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 1);
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

// ------------------------------------------------------------------------------------------------
// C-003 element 2 (+ C-001, and the execute-altitude proof that the resolved config is CONSULTED).
// ------------------------------------------------------------------------------------------------

/// C-003 element 2, TWO-SIDED at the DEFAULT config — the acceptance-criterion-1 pin and THE parity
/// number this whole change is about. Four small position-delete files in one partition are DECLINED
/// under Java's `MIN_INPUT_FILES_DEFAULT = 5`; a fifth flips the same partition to ADMITTED.
///
/// NON-VACUITY: the fixture asserts that the four/five-file input size is at or below BOTH the
/// resolved target and the resolved max, so neither `enough_content` nor `too_much_content` can be
/// the admitter — the fifth file admits through `enough_input_files` and nothing else.
///
/// MUTATION COVERAGE, two-sided and both applied: `MIN_INPUT_FILES_DEFAULT` 5 → 4 admits the
/// four-file case (first half reds); 5 → 6 declines the five-file case (second half reds). It is
/// also the EXECUTE-ALTITUDE pin that the resolved config is actually consumed by the planner: the
/// action sets no builder overrides at all, so a planner that ignored `config.min_input_files` (say,
/// by keeping a hard-coded floor of 2) admits the four-file case and reds here.
#[tokio::test]
async fn test_admission_min_input_files_default_five_declines_four_admits_five() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    // FOUR position-delete files, each masking one position of X.
    let mut sizes: Vec<u64> = Vec::new();
    let mut table = table;
    for pos in 0..4i64 {
        let pd = write_sized_pos_delete(&table, &x_path, pos, 1).await;
        sizes.push(pd.file_size_in_bytes);
        table = add_deletes(&catalog, &table, vec![pd]).await;
    }

    let config = RewritePositionDeleteFiles::new(table.clone())
        .resolve_config()
        .expect("the pure defaults are legal");
    assert_eq!(
        config.min_input_files, 5,
        "fixture: this pin runs at Java's DEFAULT floor, with no builder override"
    );
    let four_input: u64 = sizes.iter().sum();
    for size in &sizes {
        assert!(
            *size < config.min_file_size_bytes,
            "fixture: every file must be SUB-MIN so all four are candidates \
             (measured {size}, resolved min {})",
            config.min_file_size_bytes
        );
    }
    assert!(
        four_input <= config.target_file_size_bytes && four_input <= config.max_file_size_bytes,
        "fixture NON-VACUITY: the four-file input ({four_input}) must clear NEITHER the target nor \
         the max, so only the COUNT clause can ever admit this partition"
    );

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([50]),
        "before: positions 0..4 are masked"
    );
    let snapshot_before = table.metadata().current_snapshot_id();

    // (a) FOUR files at the default floor of five ⇒ DECLINED.
    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "FOUR small position-delete files are BELOW Java's floor of five — not rewritten"
    );
    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "a declined partition must NOT commit"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 4);

    // (b) A FIFTH file in the same partition ⇒ ADMITTED.
    let pd5 = write_sized_pos_delete(&reloaded, &x_path, 4, 1).await;
    let five_input = four_input + pd5.file_size_in_bytes;
    assert!(
        pd5.file_size_in_bytes < config.min_file_size_bytes,
        "fixture: the fifth file is SUB-MIN too"
    );
    assert!(
        five_input <= config.target_file_size_bytes && five_input <= config.max_file_size_bytes,
        "fixture NON-VACUITY: the five-file input ({five_input}) still clears neither size clause"
    );
    let table = add_deletes(&catalog, &reloaded, vec![pd5]).await;
    let before_five = scan_y_values(&table).await;

    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(
        result.rewritten_delete_files_count, 5,
        "FIVE files meet Java's floor — the whole bin is rewritten"
    );
    assert_eq!(result.added_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        count_pos(&live_delete_files(&reloaded).await),
        1,
        "the five position-delete files compact into one"
    );
    assert_eq!(
        scan_y_values(&reloaded).await,
        before_five,
        "read identity across the admitted compaction"
    );
}

// ------------------------------------------------------------------------------------------------
// C-003 element 4 — `enough_content` and its STRICTNESS at the target.
// ------------------------------------------------------------------------------------------------

/// C-003 element 4. TWO sub-min position-delete files whose MEASURED sizes sum to just over the
/// target: below the count floor of five, so only `enough_content` can admit them.
///
/// KNOBS FROM THE MEASUREMENTS: `target := S_A + S_B - 1`, with min/max defaulted (0.75x / 1.8x), so
/// both files are sub-min candidates and the bin's input is one byte over the target.
///
/// RECORDED AS TRUE BY CONSTRUCTION, not asserted as if it were falsifiable: `input_size > target`
/// is an identity under `target := S_A + S_B - 1`. The falsifiable content is on the MEASURED
/// OUTPUT — one Replace snapshot, two files rewritten, one added — plus the preconditions that make
/// the OTHER two clauses provably false.
///
/// MUTATION COVERAGE: delete the `enough_content` disjunct and the bin is declined (2 < 5, and the
/// input is far below max) — zero counts, no snapshot.
#[tokio::test]
async fn test_admission_enough_content_admits_two_files_over_target() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    let pd_a = write_sized_pos_delete(&table, &x_path, 1, 1).await;
    let size_a = pd_a.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd_a]).await;
    let pd_b = write_sized_pos_delete(&table, &x_path, 3, 1).await;
    let size_b = pd_b.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd_b]).await;

    let target = size_a + size_b - 1;
    let action = || RewritePositionDeleteFiles::new(table.clone()).target_file_size_bytes(target);
    let config = action().resolve_config().expect("legal knobs");

    // PRECONDITIONS.
    assert!(
        size_a < config.min_file_size_bytes && size_b < config.min_file_size_bytes,
        "fixture: both files must be SUB-MIN candidates (measured {size_a}/{size_b}, resolved min {})",
        config.min_file_size_bytes
    );
    assert!(
        2 < config.min_input_files,
        "fixture NON-VACUITY: two files are BELOW the count floor, so enough_input_files is false"
    );
    assert!(
        size_a + size_b <= config.max_file_size_bytes,
        "fixture NON-VACUITY: the input must NOT exceed max, so too_much_content is false too"
    );

    let before = scan_y_values(&table).await;
    let history_before = table.metadata().history().len();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result.rewritten_delete_files_count, 2,
        "enough_content admits a two-file bin whose bytes exceed the target"
    );
    assert_eq!(result.added_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().history().len(),
        history_before + 1,
        "exactly one Replace snapshot"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 1);
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

/// C-003 element 4, THE BOUNDARY. The same two sub-min files, but `target := S_A + S_B` exactly.
/// `enough_content` is `input_size > target`, STRICT, so an input EQUAL to the target is declined.
///
/// RECORDED AS TRUE BY CONSTRUCTION: `input_size == target` is an identity under the knob choice.
/// The falsifiable content is the MEASURED OUTCOME — zero counts and no new snapshot — plus the
/// preconditions that rule out the other two clauses.
///
/// MUTATION COVERAGE: relax `input_size > target` to `>=` and this bin is admitted.
#[tokio::test]
async fn test_admission_input_size_exactly_target_is_declined() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    let pd_a = write_sized_pos_delete(&table, &x_path, 1, 1).await;
    let size_a = pd_a.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd_a]).await;
    let pd_b = write_sized_pos_delete(&table, &x_path, 3, 1).await;
    let size_b = pd_b.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd_b]).await;

    let target = size_a + size_b;
    let action = || RewritePositionDeleteFiles::new(table.clone()).target_file_size_bytes(target);
    let config = action().resolve_config().expect("legal knobs");

    assert!(
        size_a < config.min_file_size_bytes && size_b < config.min_file_size_bytes,
        "fixture: both files must be SUB-MIN candidates, so the bin REACHES the gate"
    );
    assert!(2 < config.min_input_files, "fixture: below the count floor");
    assert!(
        size_a + size_b <= config.max_file_size_bytes,
        "fixture: too_much_content must be false"
    );

    let before = scan_y_values(&table).await;
    let snapshot_before = table.metadata().current_snapshot_id();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "an input EXACTLY at the target is declined — the comparison is STRICT"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "a declined bin must NOT commit"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 2);
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

// ------------------------------------------------------------------------------------------------
// C-003 element 5 — `too_much_content` EXISTS and carries NO `size > 1` guard.
// ------------------------------------------------------------------------------------------------

/// C-003 element 5, and acceptance criterion 3's ADMISSION leg. A LONE position-delete file above
/// `max_file_size_bytes` IS admitted: `too_much_content` is `input_size > max_file_size_bytes` with
/// no `size > 1` guard, unlike the other two clauses.
///
/// KNOBS FROM THE MEASUREMENT: `max := S - 1`, `target := S - 2`, `min := S - 3`, so the single file
/// is oversized (hence a candidate) and forms a bin of one.
///
/// This test pins ADMISSION only. What the writer then does with an oversized input — the roll bound
/// and the fixed point — is the writer increment's subject, not this clause's.
///
/// MUTATION COVERAGE: delete the `too_much_content` disjunct and this bin is declined by both
/// `size > 1` guards — zero counts, no snapshot.
#[tokio::test]
async fn test_admission_too_much_content_admits_lone_oversized_file() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    let pd = write_sized_pos_delete(&table, &x_path, 1, 1).await;
    let size = pd.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd]).await;

    let action = || {
        RewritePositionDeleteFiles::new(table.clone())
            .min_file_size_bytes(size - 3)
            .target_file_size_bytes(size - 2)
            .max_file_size_bytes(size - 1)
    };
    let config = action().resolve_config().expect("legal knobs");

    assert!(
        size > config.max_file_size_bytes,
        "fixture: the lone file must be OVERSIZED (measured {size}, resolved max {})",
        config.max_file_size_bytes
    );
    assert!(
        1 < config.min_input_files,
        "fixture NON-VACUITY: one file is below the count floor, so enough_input_files is false"
    );

    let before = scan_y_values(&table).await;
    let history_before = table.metadata().history().len();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result.rewritten_delete_files_count, 1,
        "too_much_content admits a LONE oversized file — it has no `size > 1` guard"
    );
    assert!(result.added_delete_files_count >= 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().history().len(),
        history_before + 1,
        "exactly one Replace snapshot"
    );
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

// ------------------------------------------------------------------------------------------------
// C-021 — SINGLE-FILE GROUPS, all three size classes. `is_candidate` and `group_qualifies` read
// exactly ONE scalar per file (its length) against exactly TWO thresholds, and both comparisons are
// STRICT, so the real line is TRICHOTOMOUS and both boundary values fall in the middle class:
//
//   1. `size < min_file_size`            -> candidate, bin of 1, all three clauses FALSE -> DECLINED
//   2. `min <= size <= max`              -> NOT a candidate, never reaches packing       -> DECLINED
//   3. `size > max_file_size`            -> candidate, bin of 1, too_much_content TRUE   -> ADMITTED
//
// Element 3 is `test_admission_too_much_content_admits_lone_oversized_file`, directly above (it is
// also C-003 element 5's pin and acceptance criterion 3's admission leg); elements 1 and 2 follow.
//
// WHY EACH CARRIES A WHITE-BOX `is_candidate` LEG. Classes 1 and 2 are INDISTINGUISHABLE BY OUTCOME
// — both are DECLINED with zero counts and no snapshot — so the end-to-end assertions alone cannot
// tell which mechanism did the work, and a fixture that drifted from one class into the other would
// still pass. The measured-size precondition (asserted FIRST, per C-021 and C-036) fixes the class;
// the `is_candidate` leg then asserts the MECHANISM that class implies. It is the same white-box
// seam `test_gate_*_white_box` already opens on `group_qualifies`, used here for the same reason:
// the fact is real, and no black-box fixture can express it.
//
// The two BOUNDARY values (`size == min`, `size == max`) are deliberately NOT pinned here: a LONE
// boundary file cannot discriminate the strictness mutants (both leave the observable DECLINED with
// zero counts), which is why C-004's pins pair the boundary file with a sub-min COMPANION.
// ------------------------------------------------------------------------------------------------

/// C-021 element 1 (C-036 recipe 1). A LONE SUB-MIN position-delete file is DECLINED. It IS a
/// candidate — `outsideDesiredFileSizeRange` is true below `min` — so it forms a bin of one and
/// REACHES the gate, where all three clauses are false: `enough_input_files` and `enough_content`
/// are both killed by their `size > 1` conjunct, and `too_much_content` is false because the file is
/// nowhere near `max`.
///
/// KNOBS FROM THE MEASUREMENT (recipe 1): `min := S + 1`, `target := S + 2`, `max := S + 3`. The
/// gaps are the smallest values that satisfy C-006's STRICT `min < target < max`.
///
/// TRUE BY CONSTRUCTION, RECORDED not dressed up (C-036): with `min := S + 1` the precondition
/// `S < min` is an algebraic identity in `S`. The knobs are derived FROM the measured size, so the
/// fixture CANNOT drift out of its class and the assert cannot red on drift. It is asserted anyway
/// because it is not tautological against the SEAM — it still reds if `resolve_config` ever stopped
/// honouring the explicit overrides — but it is NOT this test's falsifiable content. That content is
/// the `is_candidate` leg plus the declined outcome, each with an APPLIED killer recorded below.
///
/// PROVENANCE — this test REPLACES `test_single_file_group_is_a_no_op`, whose name and doc asserted
/// a universal the size gate FALSIFIES: "Java's planner drops single-file groups" is false, because
/// `too_much_content` has no `size > 1` guard, so a lone file above `max` is admitted (element 3
/// above). The old fixture also lost its mutant — under the deleted `entries.len() < 2` guard it
/// pinned that guard, and the guard is gone. C-014 element 9 routes it instead to C-003 element 1's
/// pin; that destination is ALREADY OCCUPIED by
/// `test_admission_min_input_files_one_still_declines_lone_sub_min_file`, which carries the
/// `.min_input_files(1)` knob that mutant needs, so following C-014 element 9 literally would have
/// produced a duplicate sharing a single mutant. C-021's own proposition names THIS destination.
/// Recorded as a deviation from C-014 element 9, in favour of C-021.
///
/// MUTATION COVERAGE, each APPLIED against the whole lib suite: `is_candidate`'s
/// `length < min_file_size` disjunct deleted (the file stops being a candidate — the white-box leg
/// reds); and `too_much_content`'s `input_size > max` flipped to `input_size < max` (the lone bin is
/// admitted — the end-to-end legs red). What this test does NOT kill is recorded next to element 2.
#[tokio::test]
async fn test_admission_sub_min_single_file_is_declined() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    let pd = write_sized_pos_delete(&table, &x_path, 1, 1).await;
    let size = pd.file_size_in_bytes;
    let entry = LiveDeleteEntry {
        data_file: pd.clone(),
        sequence_number: 1,
    };
    let table = add_deletes(&catalog, &table, vec![pd]).await;

    let action = || {
        RewritePositionDeleteFiles::new(table.clone())
            .min_file_size_bytes(size + 1)
            .target_file_size_bytes(size + 2)
            .max_file_size_bytes(size + 3)
    };
    let config = action().resolve_config().expect("recipe 1 knobs are legal");

    // PRECONDITION FIRST — the measured size against the RESOLVED thresholds. See the doc: an
    // identity in `S` given the knob derivation, kept because it still checks the resolve_config
    // seam, but NOT this test's falsifiable content.
    assert!(
        size < config.min_file_size_bytes,
        "fixture: the lone file must be SUB-MIN (measured {size}, resolved min {})",
        config.min_file_size_bytes
    );
    // MECHANISM — what separates this class from element 2: sub-min means CANDIDATE, so the file
    // does reach the gate and is declined THERE, not by the candidate filter.
    assert!(
        is_candidate(&entry, &config),
        "class 1 mechanism: a sub-min file IS a candidate and reaches the gate as a bin of one"
    );
    assert_eq!(
        config.min_input_files, 5,
        "fixture: the DEFAULT floor, no count knob — the decline is the `size > 1` conjunct's"
    );

    let before = scan_y_values(&table).await;
    let snapshot_before = table.metadata().current_snapshot_id();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "a LONE SUB-MIN file is declined: every clause of the three-way gate is false"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "post-execute SHAPE: a declined bin must NOT commit a new snapshot"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 1);
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

/// C-021 element 2 (C-036 recipe 2). A LONE IN-RANGE position-delete file is DECLINED for a
/// DIFFERENT REASON than element 1: `outsideDesiredFileSizeRange` is false, so it is NOT a candidate
/// and never reaches packing at all. The gate is not consulted.
///
/// KNOBS FROM THE MEASUREMENT (recipe 2): `min := S - 1`, `target := S`, `max := S + 1`, which puts
/// the measured size strictly inside the band with the smallest legal gaps.
///
/// TRUE BY CONSTRUCTION, RECORDED not dressed up (C-036): with knobs derived as `S-1` / `S+1`, the
/// window `min <= S <= max` is an algebraic identity in `S`. It is asserted anyway because it is NOT
/// tautological against the SEAM — it still reds if `resolve_config` ever stopped honouring the
/// explicit overrides in a direction that moved the band off `S` — but it is NOT this test's
/// falsifiable content. That content is the `is_candidate` leg plus the declined outcome.
///
/// MUTATION COVERAGE, APPLIED: `is_candidate` forced to `true` (the candidate filter deleted) reds
/// the white-box leg. HONESTLY RECORDED, with the arithmetic: the END-TO-END legs of this test
/// survive that mutant, because a lone in-range file promoted to candidacy forms a bin of ONE and is
/// then declined by the `size > 1` conjuncts anyway — outcome unchanged. That is exactly why
/// C-004's existence pin (`test_candidate_filter_drops_in_range_files_before_packing`) uses FIVE
/// in-range files, and why this element gets a white-box leg instead of pretending the end-to-end
/// assertions cover the filter.
#[tokio::test]
async fn test_admission_in_range_single_file_is_declined() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    let pd = write_sized_pos_delete(&table, &x_path, 1, 1).await;
    let size = pd.file_size_in_bytes;
    let entry = LiveDeleteEntry {
        data_file: pd.clone(),
        sequence_number: 1,
    };
    let table = add_deletes(&catalog, &table, vec![pd]).await;

    let action = || {
        RewritePositionDeleteFiles::new(table.clone())
            .min_file_size_bytes(size - 1)
            .target_file_size_bytes(size)
            .max_file_size_bytes(size + 1)
    };
    let config = action().resolve_config().expect("recipe 2 knobs are legal");

    // PRECONDITION FIRST. See the doc: an identity in `S` given the knob derivation, kept because it
    // still checks the resolve_config seam, but NOT this test's falsifiable content.
    assert!(
        config.min_file_size_bytes <= size && size <= config.max_file_size_bytes,
        "fixture: the lone file must be IN RANGE (measured {size}, resolved band [{}, {}])",
        config.min_file_size_bytes,
        config.max_file_size_bytes
    );
    // MECHANISM — what separates this class from element 1: in-range means NOT a candidate, so the
    // file is dropped BEFORE packing and the gate never sees it.
    assert!(
        !is_candidate(&entry, &config),
        "class 2 mechanism: an in-range file is NOT a candidate and never reaches packing"
    );
    assert_eq!(
        config.min_input_files, 5,
        "fixture: the DEFAULT floor, no count knob"
    );

    let before = scan_y_values(&table).await;
    let snapshot_before = table.metadata().current_snapshot_id();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "a LONE IN-RANGE file is declined by the candidate filter, before packing"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "post-execute SHAPE: a declined group must NOT commit a new snapshot"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 1);
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

// ------------------------------------------------------------------------------------------------
// C-004 — the size-only candidate filter, and its two STRICT boundaries.
// ------------------------------------------------------------------------------------------------

/// C-004 element 1 (C-036 recipe 8), and C-042's OUTCOME pin. FIVE position-delete files in one
/// partition, every one of them strictly IN RANGE, so `outsideDesiredFileSizeRange` rejects all
/// five: zero candidates, zero bins, zero commits — even though five files WOULD satisfy
/// `enough_input_files` had they reached the gate.
///
/// KNOBS FROM THE MEASUREMENTS: `min := min_i(S_i)`, `max := max_i(S_i) + 1`, `target := min + 1`
/// (the five files are written with different masked-position counts so their sizes differ).
///
/// MUTATION COVERAGE: delete the candidate filter and all five files reach the packer as one bin,
/// which `enough_input_files` (5 >= 5) then admits — non-zero counts and a Replace snapshot.
///
/// EXPLICITLY NOT A MUTANT-KILLER for C-042's `if candidates.is_empty() { continue; }`
/// short-circuit: deleting that branch is observationally a no-op, because `pack_bins` on an empty
/// input returns an empty `Vec` and the per-bin loop then never runs. That line is recorded as
/// UNKILLABLE, not as covered.
#[tokio::test]
async fn test_candidate_filter_drops_in_range_files_before_packing() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    let mut sizes: Vec<u64> = Vec::new();
    let mut table = table;
    for count in 1..=5i64 {
        let pd = write_sized_pos_delete(&table, &x_path, 100, count).await;
        sizes.push(pd.file_size_in_bytes);
        table = add_deletes(&catalog, &table, vec![pd]).await;
    }
    let smallest = *sizes.iter().min().expect("five files");
    let largest = *sizes.iter().max().expect("five files");
    assert!(
        largest >= smallest + 2,
        "fixture: the five measured sizes must span at least 2 bytes so `target = min + 1` is \
         strictly below `max` (measured {smallest}..{largest})"
    );

    let action = || {
        RewritePositionDeleteFiles::new(table.clone())
            .min_file_size_bytes(smallest)
            .target_file_size_bytes(smallest + 1)
            .max_file_size_bytes(largest + 1)
    };
    let config = action().resolve_config().expect("legal knobs");

    // PRECONDITIONS. The two window legs are true by construction (min := min_i, max := max_i + 1);
    // they are asserted anyway because they also check that the builders reach the resolved config.
    for size in &sizes {
        assert!(
            *size >= config.min_file_size_bytes && *size <= config.max_file_size_bytes,
            "fixture: every file must be IN RANGE (measured {size}, resolved [{}, {}])",
            config.min_file_size_bytes,
            config.max_file_size_bytes
        );
    }
    assert!(
        5 >= config.min_input_files,
        "fixture NON-VACUITY: five files WOULD satisfy enough_input_files if the candidate filter \
         did not drop them first"
    );

    let before = scan_y_values(&table).await;
    let snapshot_before = table.metadata().current_snapshot_id();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "a partition of well-sized files yields no candidate, no bin and no commit"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "no candidate ⇒ no snapshot"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 5);
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

/// C-004 element 2 (C-036 recipe 4). `length < min_file_size_bytes` is STRICT: a file whose MEASURED
/// size is EXACTLY the resolved min is NOT a candidate.
///
/// A lone boundary file cannot discriminate this (both the mutated and the unmutated planner leave a
/// declined, zero-count table), so the fixture adds a strictly smaller COMPANION that makes
/// candidacy outcome-bearing: unmutated, only the companion is a candidate and its bin of one is
/// declined; mutated to `<=`, the boundary file joins it and the two-file bin clears the target.
///
/// MUTATION COVERAGE: `length < min` → `length <= min` admits the bin and reds this test.
#[tokio::test]
async fn test_candidate_filter_keeps_file_at_exactly_min_file_size() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    // A is the BOUNDARY file (more masked positions ⇒ strictly larger); B is the sub-min companion.
    let pd_a = write_sized_pos_delete(&table, &x_path, 100, 5).await;
    let size_a = pd_a.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd_a]).await;
    let pd_b = write_sized_pos_delete(&table, &x_path, 1, 1).await;
    let size_b = pd_b.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd_b]).await;

    let action = || {
        RewritePositionDeleteFiles::new(table.clone())
            .min_file_size_bytes(size_a)
            .target_file_size_bytes(size_a + 1)
            .max_file_size_bytes(size_a + 2)
    };
    let config = action().resolve_config().expect("legal knobs");

    assert_eq!(
        size_a, config.min_file_size_bytes,
        "fixture: A sits EXACTLY on the resolved min"
    );
    assert!(
        size_b < config.min_file_size_bytes,
        "fixture: B is STRICTLY sub-min (measured {size_b}, resolved min {})",
        config.min_file_size_bytes
    );
    assert!(
        size_a + size_b > config.target_file_size_bytes,
        "fixture NON-VACUITY: were A wrongly a candidate, the two-file bin WOULD clear the target"
    );
    assert!(2 < config.min_input_files, "fixture: below the count floor");

    let before = scan_y_values(&table).await;
    let snapshot_before = table.metadata().current_snapshot_id();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "a file EXACTLY at min is not a candidate, so only B is packed — and a bin of one is declined"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "declined ⇒ no snapshot"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 2);
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

/// C-004 element 3 (C-036 recipe 4, mirrored). `length > max_file_size_bytes` is STRICT: a file whose
/// MEASURED size is EXACTLY the resolved max is NOT a candidate. Same companion construction as the
/// min boundary, for the same reason.
///
/// MUTATION COVERAGE: `length > max` → `length >= max` admits the bin and reds this test.
#[tokio::test]
async fn test_candidate_filter_keeps_file_at_exactly_max_file_size() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    let pd_a = write_sized_pos_delete(&table, &x_path, 100, 5).await;
    let size_a = pd_a.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd_a]).await;
    let pd_b = write_sized_pos_delete(&table, &x_path, 1, 1).await;
    let size_b = pd_b.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd_b]).await;

    let action = || {
        RewritePositionDeleteFiles::new(table.clone())
            .min_file_size_bytes(size_a - 2)
            .target_file_size_bytes(size_a - 1)
            .max_file_size_bytes(size_a)
    };
    let config = action().resolve_config().expect("legal knobs");

    assert_eq!(
        size_a, config.max_file_size_bytes,
        "fixture: A sits EXACTLY on the resolved max"
    );
    assert!(
        size_b < config.min_file_size_bytes,
        "fixture: B is STRICTLY sub-min (measured {size_b}, resolved min {})",
        config.min_file_size_bytes
    );
    assert!(
        size_a + size_b > config.target_file_size_bytes,
        "fixture NON-VACUITY: were A wrongly a candidate, the two-file bin WOULD clear the target"
    );
    assert!(
        size_b <= config.max_file_size_bytes,
        "fixture: B's own bin must not clear too_much_content"
    );
    assert!(2 < config.min_input_files, "fixture: below the count floor");

    let before = scan_y_values(&table).await;
    let snapshot_before = table.metadata().current_snapshot_id();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "a file EXACTLY at max is not a candidate, so only B is packed — and a bin of one is declined"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "declined ⇒ no snapshot"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 2);
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

// ------------------------------------------------------------------------------------------------
// C-005 — the pipeline ORDER (S4 strictly before S5) and the filter STAGE.
// ------------------------------------------------------------------------------------------------

/// C-005's ORDER pin (C-036 recipe 5), discriminating BY BIN COUNT. One partition, THREE files in
/// manifest order A, X, B: A and B sub-min candidates, X strictly in range. The knobs come from the
/// measurements — `min := S_X - 1`, `target := S_X`, `max := S_X + 1`, and a group size
/// `W := S_A + S_B` that A and B exactly fill.
///
/// FILTER-THEN-PACK (Java's order, and this port's): X is dropped at S4, so A and B pack into ONE
/// bin whose input clears the target ⇒ one Replace snapshot, two files rewritten.
///
/// PACK-THEN-FILTER (the mutant): X is still present at packing, and since `S_A + S_X > W` and
/// `S_X + S_B > W`, the greedy lookback-1 packer emits the three SINGLETONS [A], [X], [B]. Dropping X
/// afterwards leaves two bins of one, both declined by the `size > 1` guards ⇒ ZERO snapshots.
///
/// RECORDED AS TRUE BY CONSTRUCTION, not asserted as falsifiable: `min < S_X < max` and
/// `S_A + S_B <= W` are identities under the knob choice. The falsifiable preconditions are the
/// ones that make the mutant emit singletons (`S_X > S_A`, `S_X > S_B`), the ones that make A and B
/// candidates while X is not, the MANIFEST ORDER, and `S_A + S_B > target`.
#[tokio::test]
async fn test_candidate_filter_runs_before_packing() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    // ONE commit, so the three entries land in ONE manifest in this order: A, X, B.
    let pd_a = write_sized_pos_delete(&table, &x_path, 1, 1).await; // masks y=20
    let pd_x = write_sized_pos_delete(&table, &x_path, 100, 40).await; // masks nothing live
    let pd_b = write_sized_pos_delete(&table, &x_path, 3, 1).await; // masks y=40
    let (size_a, size_x, size_b) = (
        pd_a.file_size_in_bytes,
        pd_x.file_size_in_bytes,
        pd_b.file_size_in_bytes,
    );
    let (path_a, path_x, path_b) = (
        pd_a.file_path().to_string(),
        pd_x.file_path().to_string(),
        pd_b.file_path().to_string(),
    );
    let table = add_deletes(&catalog, &table, vec![pd_a, pd_x, pd_b]).await;

    assert_eq!(
        live_pos_delete_paths(&table).await,
        vec![path_a, path_x, path_b],
        "fixture: the manifest order must be A, X, B — the packing the mutant produces depends on it"
    );

    let group_size = size_a + size_b;
    let action = || {
        RewritePositionDeleteFiles::new(table.clone())
            .min_file_size_bytes(size_x - 1)
            .target_file_size_bytes(size_x)
            .max_file_size_bytes(size_x + 1)
            .max_file_group_size_bytes(group_size)
    };
    let config = action().resolve_config().expect("legal knobs");

    // PRECONDITIONS.
    assert!(
        size_a < config.min_file_size_bytes && size_b < config.min_file_size_bytes,
        "fixture: A and B are SUB-MIN candidates (measured {size_a}/{size_b}, resolved min {})",
        config.min_file_size_bytes
    );
    assert!(
        size_x >= config.min_file_size_bytes && size_x <= config.max_file_size_bytes,
        "fixture: X is strictly IN RANGE, so S4 must drop it"
    );
    assert!(
        size_x > size_a && size_x > size_b,
        "fixture: X must be larger than each of A and B, or the pack-then-filter mutant would NOT \
         emit singletons and the pin would not discriminate (measured A={size_a}, X={size_x}, B={size_b})"
    );
    assert!(
        size_a + size_b > config.target_file_size_bytes,
        "fixture: the A+B bin must clear the target so the correct order ADMITS it"
    );
    assert!(
        2 < config.min_input_files,
        "fixture NON-VACUITY: two files are below the count floor, so enough_content is the only \
         admitting clause"
    );

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 30, 50]),
        "before: A masks y=20, B masks y=40"
    );
    let history_before = table.metadata().history().len();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result.rewritten_delete_files_count, 2,
        "filter-then-pack: A and B form ONE admitted bin; the in-range X never reaches the packer"
    );
    assert_eq!(result.added_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().history().len(),
        history_before + 1,
        "exactly ONE Replace snapshot — the pack-then-filter mutant commits none"
    );
    assert_eq!(
        count_pos(&live_delete_files(&reloaded).await),
        2,
        "the compacted A+B file plus the untouched X"
    );
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

/// C-005's FILTER-STAGE pin for the hoist's NAMED CONSEQUENCE (behaviour flip 2). The user `filter`
/// is now bound ONCE, right after the no-snapshot early return and BEFORE the manifest walk, exactly
/// where Java binds it (at the `PositionDeletes` scan). So an UNBINDABLE filter fails on ANY table
/// with a current snapshot — here one that has no position-delete files at all, and therefore no
/// group that could ever be admitted.
///
/// Under the pre-hoist, per-group binding this same call returned `Ok` with zero counts, because the
/// bind only ran inside a group that already had two files. The flip is deliberate: a filter the
/// engine cannot bind is a caller error, and failing loudly beats silently compacting nothing.
#[tokio::test]
async fn test_unbindable_filter_errors_even_when_no_group_is_admissible() {
    use crate::expr::Reference;
    use crate::spec::Datum;

    let (catalog, _temp, table, _x_path) = gate_table().await;
    assert!(
        table.metadata().current_snapshot().is_some(),
        "fixture: the table HAS a current snapshot (the data-file append)"
    );
    assert!(
        live_delete_files(&table).await.is_empty(),
        "fixture: and NO delete files, so no group is admissible under any gate"
    );

    let error = RewritePositionDeleteFiles::new(table.clone())
        .filter(Reference::new("no_such_column").equal_to(Datum::long(0)))
        .execute(&catalog)
        .await
        .expect_err("an unbindable filter must fail at the bind, not be silently ignored");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error
            .message()
            .contains("filter could not be bound to the table schema"),
        "unexpected message: {}",
        error.message()
    );
}

// ------------------------------------------------------------------------------------------------
// C-027 — the SHARED bin packer, reused through a weight closure.
// ------------------------------------------------------------------------------------------------

/// C-027 (C-036 recipe 7's fixture shape). `max_file_group_size_bytes` splits ONE partition into TWO
/// bins, and each admitted bin commits its own Replace snapshot — which is only possible if the
/// position-delete planner really goes through the shared `pack_bins` with the delete file's size as
/// its weight.
///
/// KNOBS FROM THE MEASUREMENTS: four files A, B, C, D in manifest order, all sub-min (so all four are
/// candidates), `.min_input_files(2)`, and `W := max(S_A + S_B, S_C + S_D)` with `W < S_A + S_B + S_C`
/// asserted — under the greedy forward first-fit that yields exactly {A, B} and {C, D}.
///
/// MUTATION COVERAGE: pass a constant weight (or `u64::MAX` as the group size) and the four files
/// pack into ONE bin — one snapshot, one added file, and both assertions red. Reverse the packer's
/// emission order and the manifest-order precondition reds.
#[tokio::test]
async fn test_admission_max_file_group_size_splits_partition_into_bins() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    // Distinct sizes, so the packing genuinely depends on the weights and on the manifest order.
    let pd_a = write_sized_pos_delete(&table, &x_path, 100, 1).await;
    let pd_b = write_sized_pos_delete(&table, &x_path, 200, 2).await;
    let pd_c = write_sized_pos_delete(&table, &x_path, 300, 3).await;
    let pd_d = write_sized_pos_delete(&table, &x_path, 400, 4).await;
    let sizes = [
        pd_a.file_size_in_bytes,
        pd_b.file_size_in_bytes,
        pd_c.file_size_in_bytes,
        pd_d.file_size_in_bytes,
    ];
    let paths: Vec<String> = [&pd_a, &pd_b, &pd_c, &pd_d]
        .iter()
        .map(|f| f.file_path().to_string())
        .collect();
    let table = add_deletes(&catalog, &table, vec![pd_a, pd_b, pd_c, pd_d]).await;

    assert_eq!(
        live_pos_delete_paths(&table).await,
        paths,
        "fixture: the manifest order must be A, B, C, D — the bin boundaries depend on it"
    );

    let group_size = (sizes[0] + sizes[1]).max(sizes[2] + sizes[3]);
    let action = || {
        RewritePositionDeleteFiles::new(table.clone())
            .min_input_files(2)
            .min_file_size_bytes(100_000)
            .target_file_size_bytes(200_000)
            .max_file_size_bytes(400_000)
            .max_file_group_size_bytes(group_size)
    };
    let config = action().resolve_config().expect("legal knobs");

    // PRECONDITIONS, all over MEASURED sizes.
    for size in &sizes {
        assert!(
            *size < config.min_file_size_bytes,
            "fixture: every file must be SUB-MIN so all four are candidates \
             (measured {size}, resolved min {})",
            config.min_file_size_bytes
        );
    }
    assert!(
        2 >= config.min_input_files,
        "fixture: a bin of two must clear the count floor"
    );
    assert!(
        sizes[0] + sizes[1] <= config.max_file_group_size_bytes,
        "fixture: A and B must fit ONE bin"
    );
    assert!(
        config.max_file_group_size_bytes < sizes[0] + sizes[1] + sizes[2],
        "fixture: C must NOT fit alongside A and B — this is what forces the split"
    );
    assert!(
        sizes[2] + sizes[3] <= config.max_file_group_size_bytes,
        "fixture: C and D must fit ONE bin"
    );

    let before = scan_y_values(&table).await;
    let history_before = table.metadata().history().len();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result.rewritten_delete_files_count, 4,
        "all four candidates are rewritten, but across TWO bins"
    );
    assert_eq!(
        result.added_delete_files_count, 2,
        "one compacted file per BIN — a single bin would add one"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().history().len(),
        history_before + 2,
        "two admitted bins ⇒ two Replace snapshots"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 2);
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

// ------------------------------------------------------------------------------------------------
// C-041 — the group input-size sum SATURATES where Java's `long` sum wraps.
// ------------------------------------------------------------------------------------------------

/// C-041. The bin input-size sum is `saturating_add`, so an overflowing bin resolves to `u64::MAX`
/// and stays ADMITTED. Java's `inputSize` is `stream().mapToLong(..).sum()`, which WRAPS to a small
/// (or negative) value and would DECLINE — a named, recorded divergence in the safe direction.
///
/// WHITE-BOX on `group_qualifies` with two fabricated entries, because an end-to-end fixture would
/// then try to READ 16 EiB of position deletes. The thresholds are chosen so the two arithmetics
/// disagree on the OUTCOME: saturating gives `u64::MAX > target`, while wrapping gives
/// `u64::MAX + 10 = 9`, which clears neither the target nor the max, and the count floor is above
/// the bin size either way.
///
/// MUTATION COVERAGE: change `saturating_add` to `wrapping_add` in `group_qualifies` and the bin is
/// declined.
#[tokio::test]
async fn test_group_input_size_saturates_not_wraps() {
    let (_catalog, _temp, table, x_path) = gate_table().await;

    let mut huge = write_sized_pos_delete(&table, &x_path, 1, 1).await;
    huge.file_size_in_bytes = u64::MAX;
    let mut small = write_sized_pos_delete(&table, &x_path, 3, 1).await;
    small.file_size_in_bytes = 10;

    let bin = vec![
        LiveDeleteEntry {
            data_file: huge,
            sequence_number: 1,
        },
        LiveDeleteEntry {
            data_file: small,
            sequence_number: 1,
        },
    ];
    let config = ResolvedConfig {
        target_file_size_bytes: 100,
        min_file_size_bytes: 50,
        max_file_size_bytes: 1_000,
        // Above the bin size, so `enough_input_files` cannot be the admitter under EITHER arithmetic.
        min_input_files: 10,
        max_file_group_size_bytes: 1_000_000,
        write_max_file_size: 550,
        chunk_budget: 225,
    };

    assert!(
        group_qualifies(&bin, &config),
        "u64::MAX + 10 must SATURATE to u64::MAX (admitted via enough_content), not WRAP to 9 \
         (which clears neither the target 100 nor the max 1000, and would be declined)"
    );
}

// ------------------------------------------------------------------------------------------------
// C-003 elements 3 and 5b — the two gate leaves that are unreachable END TO END, pinned WHITE BOX.
//
// C-003 recorded both as "UNKILLABLE, never claimed pinned". That scoping is now SUPERSEDED, in the
// fork's favour: both proofs rest on candidate-filter REACHABILITY (a bin in those states cannot be
// produced by `plan_bins`, because its member would not have been a candidate), not on the config
// space, and this module already opens a white-box seam on `group_qualifies`. Through that seam the
// states ARE constructible and both mutants DO die. The end-to-end unreachability argument stays
// true and stays recorded on `group_qualifies`; what changes is that neither leaf is now unpinned.
// ------------------------------------------------------------------------------------------------

/// A [`LiveDeleteEntry`] whose `file_size_in_bytes` is SET to `size` — the same white-box seam
/// `test_group_input_size_saturates_not_wraps` uses. The underlying file is real but is never read:
/// `group_qualifies` reads nothing off the entry except its size.
async fn entry_of_size(table: &Table, target_path: &str, size: u64) -> LiveDeleteEntry {
    let mut data_file = write_sized_pos_delete(table, target_path, 1, 1).await;
    data_file.file_size_in_bytes = size;
    LiveDeleteEntry {
        data_file,
        sequence_number: 1,
    }
}

/// The thresholds both white-box gate pins run against: `min < target < max` (C-006 preconditions
/// (3) and (4)), with `min_input_files` well above the bin size so `enough_input_files` cannot be
/// the admitter under any of the mutants below.
fn white_box_gate_config() -> ResolvedConfig {
    ResolvedConfig {
        target_file_size_bytes: 100,
        min_file_size_bytes: 50,
        max_file_size_bytes: 1_000,
        min_input_files: 10,
        max_file_group_size_bytes: 1_000_000,
        // Java `writeMaxFileSize()` = 100 + (1000 - 100) * 0.5. Neither gate leaf reads it.
        write_max_file_size: 550,
        // min(16384, (1000 - 550) / 2). Neither gate leaf reads it either.
        chunk_budget: 225,
    }
}

/// C-003 element 5b — `too_much_content`'s BOUNDARY STRICTNESS. A bin whose input size is EXACTLY
/// `max_file_size_bytes` is DECLINED, because the comparison is `input_size > max`, strict.
///
/// END-TO-END UNREACHABLE, and that is why C-003 recorded it as unkillable: such a bin is either of
/// size 1 — and then its file's length equals max, so `outsideDesiredFileSizeRange` never made it a
/// candidate and it never reaches the gate — or of size >= 2, and then `enough_content` is already
/// true because `max > target`. WHITE BOX the state is trivially constructible, so the mutant dies.
///
/// MUTATION COVERAGE: `input_size > max` → `>=` admits this bin.
#[tokio::test]
async fn test_gate_input_size_exactly_max_is_declined_white_box() {
    let (_catalog, _temp, table, x_path) = gate_table().await;
    let config = white_box_gate_config();

    let bin = vec![entry_of_size(&table, &x_path, config.max_file_size_bytes).await];

    assert!(
        !group_qualifies(&bin, &config),
        "an input size EXACTLY at max is declined — too_much_content is `input_size > max`, STRICT"
    );
}

/// C-003 element 3 — `enough_content`'s `size > 1` conjunct. A LONE file whose length is above the
/// target but at or below the max is DECLINED: `enough_content` is `size > 1 && input_size > target`,
/// and the `size > 1` conjunct is the only thing that declines it.
///
/// END-TO-END UNREACHABLE, and that is why C-003 recorded it as unkillable: a bin of one that
/// reaches the gate must be a candidate, so its length is either below min — and `min < target`, so
/// `enough_content` is false either way — or above max, in which case `too_much_content` is already
/// true and carries the admission. WHITE BOX the in-band lone file is constructible, so the mutant
/// dies.
///
/// MUTATION COVERAGE: drop the `size > 1` conjunct from `enough_content` and this bin is admitted.
#[tokio::test]
async fn test_gate_enough_content_size_guard_declines_lone_over_target_file_white_box() {
    let (_catalog, _temp, table, x_path) = gate_table().await;
    let config = white_box_gate_config();

    let size = 500;
    assert!(
        size > config.target_file_size_bytes && size < config.max_file_size_bytes,
        "fixture: the lone file is strictly BETWEEN target and max, so enough_content's size guard \
         is the ONLY clause declining it"
    );
    let bin = vec![entry_of_size(&table, &x_path, size).await];

    assert!(
        !group_qualifies(&bin, &config),
        "a LONE file above the target is declined by enough_content's `size > 1` conjunct"
    );
}

// =================================================================================================
// G3 — THE WRITER. C-009 (the roll bound), C-025 (the bounded chunk feed), C-026 (the fixed point),
// C-044 (global sort before split), C-046 (the fail-closed guard at the new arity), C-010 (the
// `Vec<DataFile>` fan-out and the BIN-max stamp).
//
// FIXTURE DISCIPLINE (C-036, N-C2, N-C3). Recipes 3 and 9 derive their target `T` from the fixture's
// MEASURED size, which makes every `S`- or `B`-relative window an ALGEBRAIC IDENTITY that can never
// red. Those windows are RECORDED in prose as true by construction and are NOT asserted; the
// falsifiable content of both recipes lives on the MEASURED OUTPUTS instead.
//
// DISCLOSURE — C-036's NINE-RECIPE ENUMERATION IS INCOMPLETE, and by its own rule ("a pin needing a
// tenth recipe is a finding") that is a finding, filed here rather than left silent. This group
// needed THREE size-class fixtures the nine do not describe, named below at their definition sites:
//
//   * RECIPE 10 — the SMALL-EXPLICIT-BAND SPLIT (`split_fixture_run`). Two sub-min files in one bin
//     at `min = 0.55C / target = 0.60C / max = 0.75C`, `min_input_files(2)`. An ENGINEERING CHOICE,
//     not a clause requirement: it is the cheap counterpart to recipe 3, carrying four split pins at
//     ~0.24 MB instead of ~4.7 MB. Recipe 3 keeps the two assertions that genuinely need the wide
//     DEFAULT band (outputs inside `[min, max]`, and the fixed point).
//   * RECIPE 11 — FIVE BINS OF ONE, each splitting to a sub-min tail (C-026 counterexample 2).
//   * RECIPE 12 — TWO BINS OF ONE, each splitting to a sub-min tail (C-026 counterexample 3).
//
// Recipes 11 and 12 are MANDATED BY C-026's own pin form — it names both counterexamples and
// requires them pinned — while C-036 describes no fixture that can build either. That half is a
// ledger-completeness defect this build surfaced, not a scope escape. The ledger edit is outside
// this unit's two-file fence and is carried as a hand-off.
// =================================================================================================

/// Every live PARQUET position-delete file in the current snapshot.
async fn live_pos_delete_files(table: &Table) -> Vec<DataFile> {
    live_delete_files(table)
        .await
        .into_iter()
        .filter(|f| f.content_type() == DataContentType::PositionDeletes)
        .collect()
}

/// Read one position-delete file's `(file_path, pos)` pairs back off disk, by RESERVED FIELD ID —
/// the same way the action reads them. This is what turns "two files exist" into "these exact pairs
/// are in these exact files", which is what C-044's ordering pin needs.
async fn read_pos_delete_pairs(table: &Table, delete_file: &DataFile) -> Vec<(String, i64)> {
    let loader = BasicDeleteFileLoader::new(table.file_io().clone());
    let mut stream = loader
        .parquet_to_batch_stream(delete_file.file_path(), delete_file.file_size_in_bytes)
        .await
        .expect("open the compacted position-delete file");
    let mut pairs = Vec::new();
    while let Some(batch) = stream.next().await {
        let batch = batch.expect("read a compacted position-delete batch");
        let (path_col, pos_col) =
            locate_reserved_columns(&batch, delete_file.file_path()).expect("reserved columns");
        for row in 0..batch.num_rows() {
            pairs.push((path_col.value(row).to_string(), pos_col.value(row)));
        }
    }
    pairs
}

/// The live position-delete files with their pairs, in WRITE order — sorted by each file's FIRST
/// pair, not by manifest order, which does not track the order the rolling writer produced them in.
/// Every "first output" / "output k vs k+1" assertion below is made through this, so none of them
/// silently depends on a manifest ordering nobody pinned.
async fn outputs_in_write_order(table: &Table) -> Vec<(DataFile, Vec<(String, i64)>)> {
    let mut outputs = Vec::new();
    for file in live_pos_delete_files(table).await {
        let pairs = read_pos_delete_pairs(table, &file).await;
        outputs.push((file, pairs));
    }
    outputs.sort_by(|a, b| a.1.first().cmp(&b.1.first()));
    outputs
}

/// Write ONE position-delete file holding `path_count` pairs, each naming a DISTINCT data-file path.
///
/// This is C-036's dictionary-defeating size dial: `write_sized_pos_delete` grows the `pos` column
/// against ONE repeated path, which dictionary-encodes to a single value, so it cannot reach the
/// multi-megabyte size class recipe 3 needs. Growing the PATH COUNT is what recipe 3 mandates —
/// never growing the knobs, which is what would make the window tautological.
async fn write_wide_path_pos_delete(table: &Table, path_count: i64) -> DataFile {
    let base = format!("{}/data", table.metadata().location());
    let paths: Vec<String> = (0..path_count)
        .map(|i| format!("{base}/wide-{i:012}-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.parquet"))
        .collect();
    let pairs: Vec<(&str, i64)> = paths
        .iter()
        .enumerate()
        .map(|(i, path)| (path.as_str(), i as i64))
        .collect();
    write_position_delete_file(table, Some(0), &pairs).await
}

/// C-036 RECIPE 3 — the LONE OVERSIZED file, WIDE BAND.
///
/// ONE position-delete file whose MEASURED size `S` clears `240 * CHUNK_MAX_SERIALIZED_BYTES`
/// (3_932_160), built by growing the DISTINCT PATH COUNT. Only `.target_file_size_bytes(T)` is set,
/// with `T = S * 10 / 24`; min / max are DEFAULTED, giving `min = 0.75T`, `max = 1.8T`,
/// `write_max = 1.4T` and — since `(max - write_max) / 2 = 0.2T` is far above the cap —
/// `chunk_budget = 16384`.
///
/// RECORDED as TRUE BY CONSTRUCTION and deliberately NOT asserted, because `T := S * 10 / 24` makes
/// each an identity in `S` that no drift could ever red (N-C3):
/// - `S > resolved max` (i.e. `S > 1.8T = 0.75S`), which is why `too_much_content` admits the lone
///   file at all; and
/// - `2.15T <= S <= 2.8T`, the two-output window expressed on the INPUT.
///
/// Returns the catalog, the temp dir (kept alive), the committed table, `S` and `T`.
async fn recipe_3_lone_oversized_fixture() -> (impl Catalog, TempDir, Table, u64, u64) {
    let (catalog, temp, table, _x_path) = gate_table().await;
    // 34_000 distinct ~105-byte paths measure ~4.74 MB here — ~20% above the 3_932_160 floor, which
    // the test asserts rather than assumes. The path count is the only dial; the knobs never move.
    let pd = write_wide_path_pos_delete(&table, 34_000).await;
    let s = pd.file_size_in_bytes;
    let t = s * 10 / 24;
    let table = add_deletes(&catalog, &table, vec![pd]).await;
    (catalog, temp, table, s, t)
}

/// Recipe 3's shared PRE-ASSERTIONS, run before `execute` in both of its tests: the fixture really is
/// in the size class the recipe claims, and the chunk budget really is the ruled constant. A size
/// drift therefore fails HERE, loudly, instead of reddening the fixed point for an unrelated reason.
fn assert_recipe_3_preconditions(s: u64, config: &ResolvedConfig) {
    assert!(
        s >= 240 * CHUNK_MAX_SERIALIZED_BYTES,
        "recipe 3 precondition: the measured fixture size S = {s} must clear \
         240 * CHUNK_MAX_SERIALIZED_BYTES = {} — grow the DISTINCT PATH COUNT, never the knobs",
        240 * CHUNK_MAX_SERIALIZED_BYTES
    );
    assert_eq!(
        config.chunk_budget, CHUNK_MAX_SERIALIZED_BYTES,
        "recipe 3 precondition: the band is wide, so the 16 KiB CAP binds, not the headroom half"
    );
}

// ------------------------------------------------------------------------------------------------
// C-009 — the roll bound is Java's `writeMaxFileSize()`, NOT the resolved target.
// ------------------------------------------------------------------------------------------------

/// C-009 pin 1 (C-036 recipe 6 — no fixture). The VALUE, white-box through the `resolve_config`
/// seam: at the delete defaults `write_max_file_size` is EXACTLY Java's 93952409, and in particular
/// is NOT the resolved target 67108864.
///
/// `writeMaxFileSize()` = `target + (max - target) * 0.5` with the subtraction a LONG `lsub` BEFORE
/// the `l2d` — `67108864 + (120795955 - 67108864) * 0.5 = 93952409.5`, `d2l` → 93952409.
///
/// MUTATION COVERAGE: any drift in the ratio, in the subtraction order, or a revert to "roll at the
/// target" moves this literal. The `assert_ne!` is the specific guard against the reversion R-1
/// exists to prevent, which an `assert!(write_max > 0)` style check would not catch.
#[tokio::test]
async fn test_config_write_max_file_size_default_is_java_write_max() {
    let (_temp, table) = config_table(&[]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect("the delete defaults are a legal config");

    assert_eq!(
        config.write_max_file_size, 93_952_409,
        "Java writeMaxFileSize() at the delete defaults: 67108864 + (120795955 - 67108864) * 0.5"
    );
    assert_ne!(
        config.write_max_file_size, config.target_file_size_bytes,
        "the roll bound is write-max, NOT the resolved target — that reversion is what R-1 forbids"
    );

    // THE SUBTRACTION ORDER, which nothing else in this suite discriminates. Java's bytecode is
    // `getfield maxFileSize; getfield targetFileSize; lsub; l2d` — a LONG subtraction BEFORE the
    // widening — so the Rust mirror must subtract in `u64` and only then convert. At the delete
    // defaults both orders agree (the values sit far below 2^53), so the property needs a config
    // where the two roundings diverge. This is one: the `u64` subtraction yields 48764580662105708
    // exactly, whose `l2d` differs from `l2d(max) - l2d(target)` by enough to move the result by 32.
    //
    // MUTATION COVERAGE: rewrite the derivation as
    // `d2l(target as f64 + ((max_file_size_bytes as f64) - (target as f64)) * RATIO)` and this
    // reads 198299551875299584.
    let (_temp, table) = config_table(&[]).await;
    let ordered = RewritePositionDeleteFiles::new(table)
        .target_file_size_bytes(173_917_261_544_246_756)
        .max_file_size_bytes(222_681_842_206_352_464)
        .resolve_config()
        .expect("min defaults to 0.75 * target < target < max <= i64::MAX, so this resolves");
    assert_eq!(
        ordered.write_max_file_size, 198_299_551_875_299_616,
        "the (max - target) subtraction happens in the INTEGER domain, BEFORE the widening — \
         subtracting after it gives 198299551875299584"
    );
}

/// C-009 pin 2 (A-001 — `write_max_file_size` gets its OWN clamp pair, never a borrowed one).
/// Java's `d2l` saturates at `Long.MAX_VALUE`; Rust's `as u64` saturates at `u64::MAX`, so the
/// `.min(i64::MAX as u64)` inside [`d2l`] IS the parity act on this call site too.
///
/// The fixture is the only shape in which the two disagree: `target = 2^63 - 512` (whose `l2d`
/// rounds UP to `2^63`) with an EXPLICIT `max = i64::MAX`. Then
/// `write_max = d2l(2^63 + (i64::MAX - target) * 0.5) = d2l(2^63 + 255.5)`, whose `f64` value is
/// `2^63` exactly (the spacing there is 2048) — 9223372036854775808 unclamped, i64::MAX clamped.
/// Both branches resolve Ok, so the equality assertion is the discriminator, not the Ok/Err.
///
/// MUTATION COVERAGE: drop the `.min(i64::MAX as u64)` from `d2l` and this reads
/// 9223372036854775808.
#[tokio::test]
async fn test_config_write_max_file_size_clamps_to_java_long_max() {
    let (_temp, table) = config_table(&[]).await;
    let target: u64 = 9_223_372_036_854_775_296; // 2^63 - 512
    let config = RewritePositionDeleteFiles::new(table)
        .target_file_size_bytes(target)
        .max_file_size_bytes(i64::MAX as u64)
        .resolve_config()
        .expect("target < max and target > default min, so this config resolves");

    assert!(
        (target as f64) > (i64::MAX as f64) - 2048.0,
        "fixture: the target's `l2d` must round UP to 2^63, or the two branches agree and the \
         assertion below is vacuous"
    );
    assert_eq!(
        config.write_max_file_size,
        i64::MAX as u64,
        "Java's d2l SATURATES at Long.MAX_VALUE; unclamped this would be 9223372036854775808"
    );
}

/// C-009 pin 3 (C-036 RECIPE 9) — the CALL-SITE DISCRIMINATOR, and the only pin that can tell the
/// two candidate bounds apart.
///
/// The bound reaches [`RollingFileWriterBuilder::new`] as an opaque `usize`, so no white-box config
/// assertion can distinguish "passed `write_max`" from "passed the resolved target"; and rolling at
/// the smaller target only ever produces MORE outputs, so a bare "more than one file" assertion
/// cannot discriminate either. Only an OUTPUT COUNT inside the window where the two bounds DISAGREE
/// does — and per N-C3 that window is asserted on the MEASURED OUTPUT, never implied by an identity
/// in the fixture size.
///
/// RECIPE 9: FIVE sub-min position-delete files in one partition whose MEASURED total `B` clears
/// `30 * CHUNK_MAX_SERIALIZED_BYTES` (491_520), with `T = B * 10 / 12` and min / max DEFAULTED —
/// `min = 0.625B`, `max = 1.5B`, `write_max = 1.1667B`.
///
/// RECORDED as TRUE BY CONSTRUCTION and deliberately NOT asserted (STRUCK per N-C3): `target < B <=
/// write_max`, an identity in `B` once `T := B * 10 / 12` is substituted.
///
/// NAMED MUTANT, APPLIED: `RollingFileWriterBuilder::new(builder,
/// usize::try_from(config.write_max_file_size)...)` → `...config.target_file_size_bytes...`. The
/// asserted `target + 2 * chunk_budget < O` IS that mutant's roll condition, so the pin proves its
/// own non-vacuity at runtime: under the mutant the single output would have rolled into two.
#[tokio::test]
async fn test_roll_bound_is_write_max_not_target() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    // Five files, each ~118 KB, over DISJOINT position ranges of the same data file.
    let mut deletes = Vec::new();
    for k in 0..5i64 {
        deletes.push(write_sized_pos_delete(&table, &x_path, 1 + k * 20_000, 12_000).await);
    }
    let sizes: Vec<u64> = deletes.iter().map(|f| f.file_size_in_bytes).collect();
    let b: u64 = sizes.iter().sum();
    let t = b * 10 / 12;
    let table = add_deletes(&catalog, &table, deletes).await;

    let action = || RewritePositionDeleteFiles::new(table.clone()).target_file_size_bytes(t);
    let config = action().resolve_config().expect("legal knobs");

    // PRECONDITIONS, all over MEASURED quantities.
    assert!(
        b >= 30 * CHUNK_MAX_SERIALIZED_BYTES,
        "recipe 9 precondition: the measured total B = {b} must clear \
         30 * CHUNK_MAX_SERIALIZED_BYTES = {}, or the named mutant can survive a re-encode drift",
        30 * CHUNK_MAX_SERIALIZED_BYTES
    );
    for size in &sizes {
        assert!(
            *size < config.min_file_size_bytes,
            "recipe 9 precondition: every input must be SUB-MIN so all five are candidates \
             (measured {size}, resolved min {})",
            config.min_file_size_bytes
        );
    }
    assert!(
        5 >= config.min_input_files,
        "recipe 9 precondition: five files must clear the DEFAULT count floor of \
         {} — this is what admits the one bin",
        config.min_input_files
    );
    assert_eq!(
        config.chunk_budget, CHUNK_MAX_SERIALIZED_BYTES,
        "recipe 9 precondition: the band is wide, so the 16 KiB cap binds"
    );

    let before = scan_y_values(&table).await;
    let result = action().execute(&catalog).await.expect("run 1");

    assert_eq!(
        result.rewritten_delete_files_count, 5,
        "all five are one bin and all five are rewritten"
    );
    assert_eq!(
        result.added_delete_files_count, 1,
        "ONE output: the bin is below write_max, so the rolling writer never rolls. Rolling at the \
         resolved target instead would have produced TWO — that is the named mutant"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    let outputs = live_pos_delete_files(&reloaded).await;
    assert_eq!(outputs.len(), 1);
    let o = outputs[0].file_size_in_bytes;

    // NON-VACUITY, ON THE MEASURED OUTPUT. The left inequality is the mutant's roll condition; the
    // right is the reason the real bound does NOT roll.
    assert!(
        config.target_file_size_bytes + 2 * config.chunk_budget < o,
        "the single output ({o}) must exceed target + 2 * chunk_budget ({}) — this is exactly the \
         condition under which rolling at the TARGET would have split it, so without it the \
         `added == 1` assertion above would be vacuous",
        config.target_file_size_bytes + 2 * config.chunk_budget
    );
    assert!(
        o <= config.write_max_file_size,
        "the single output ({o}) must not exceed write_max ({}) — otherwise the real bound rolled \
         too and the pin is measuring something else",
        config.write_max_file_size
    );
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

/// Write a position-delete file masking `count` positions of `target_path` STRIDED by `stride` from
/// `first_pos`. Two strided files offset by one INTERLEAVE, which is what makes the global sort
/// observable: with consecutive, disjoint ranges the concatenated input is ALREADY sorted and
/// C-044's ordering pin is vacuous — a mutation proof caught exactly that.
async fn write_strided_pos_delete(
    table: &Table,
    target_path: &str,
    first_pos: i64,
    count: i64,
    stride: i64,
) -> DataFile {
    let pairs: Vec<(&str, i64)> = (0..count)
        .map(|index| (target_path, first_pos + index * stride))
        .collect();
    write_position_delete_file(table, Some(0), &pairs).await
}

/// Everything the SMALL-EXPLICIT-BAND split fixture's tests read off one run.
struct SplitFixture {
    table: Table,
    input_sizes: Vec<u64>,
    /// The two inputs' pairs READ BACK OFF DISK and concatenated in MANIFEST ORDER — the exact
    /// sequence `compact_group` builds before it sorts. This is measured, never reconstructed from
    /// the fixture's own construction, because the ordering pin's non-vacuity depends on this
    /// sequence being UNSORTED and a reconstructed value could not witness that.
    input_concat: Vec<(String, i64)>,
    /// `input_concat` sorted — the multiset the split outputs must reproduce exactly.
    input_pairs: Vec<(String, i64)>,
    before: HashSet<i64>,
    config: ResolvedConfig,
    result: RewritePositionDeleteFilesResult,
}

/// **C-036 RECIPE 10 (NEW — not one of the ledger's nine; see the section banner's disclosure).**
///
/// The SMALL-EXPLICIT-BAND SPLIT fixture, built and run once: TWO ~118 KB position-delete files over
/// DISJOINT position ranges of one data file, committed in TWO separate snapshots so their data
/// sequence numbers are 2 and 3 (bin max = 3). The knobs are set as fractions of the MEASURED
/// combined size `C` — `min = 0.55C`, `target = 0.60C`, `max = 0.75C`, so `write_max = 0.675C` —
/// which puts the ONE bin comfortably above the roll bound and makes it split.
///
/// This is the cheap counterpart to recipe 3: it exercises the SPLIT (fan-out, ordering, counts,
/// stamping) without paying for a multi-megabyte fixture. Recipe 3 stays the home of the two
/// assertions that genuinely need the wide default band — the run-1 outputs landing inside
/// `[min, max]`, and the fixed point.
async fn split_fixture_run() -> (impl Catalog, TempDir, SplitFixture) {
    let (catalog, temp, table, x_path) = gate_table().await;

    // ODD positions and EVEN positions over the SAME range: whichever order the manifest walk
    // reads them in, the concatenated pair list is NOT sorted, so only the GLOBAL sort in
    // `compact_group` can produce ascending outputs (C-044).
    let pd1 = write_strided_pos_delete(&table, &x_path, 1, 12_000, 2).await;
    let table = add_deletes(&catalog, &table, vec![pd1]).await; // seq 2
    let pd2 = write_strided_pos_delete(&table, &x_path, 2, 12_000, 2).await;
    let table = add_deletes(&catalog, &table, vec![pd2]).await; // seq 3

    let input_sizes: Vec<u64> = live_pos_delete_files(&table)
        .await
        .iter()
        .map(|f| f.file_size_in_bytes)
        .collect();
    let c: u64 = input_sizes.iter().sum();
    // READ the inputs back in MANIFEST ORDER — the same order `collect_position_delete_groups`
    // walks them — so the fixture can witness what the writer is actually fed.
    let mut input_concat: Vec<(String, i64)> = Vec::new();
    for file in live_pos_delete_files(&table).await {
        input_concat.extend(read_pos_delete_pairs(&table, &file).await);
    }
    let mut input_pairs = input_concat.clone();
    input_pairs.sort();

    let action = || {
        RewritePositionDeleteFiles::new(table.clone())
            .min_input_files(2)
            .min_file_size_bytes(c * 55 / 100)
            .target_file_size_bytes(c * 60 / 100)
            .max_file_size_bytes(c * 75 / 100)
    };
    let config = action().resolve_config().expect("legal knobs");

    // PRECONDITIONS, over MEASURED sizes.
    assert_eq!(input_sizes.len(), 2, "fixture: exactly two inputs");
    for size in &input_sizes {
        assert!(
            *size < config.min_file_size_bytes,
            "fixture: both inputs must be SUB-MIN so both are candidates \
             (measured {size}, resolved min {})",
            config.min_file_size_bytes
        );
    }
    assert!(
        2 >= config.min_input_files,
        "fixture: a bin of two must clear the count floor"
    );
    assert!(
        c > config.write_max_file_size,
        "fixture: the bin ({c}) must exceed the roll bound ({}) or nothing splits and every \
         split assertion below is vacuous",
        config.write_max_file_size
    );

    let before = scan_y_values(&table).await;
    let result = action().execute(&catalog).await.expect("the split run");

    (catalog, temp, SplitFixture {
        table,
        input_sizes,
        input_concat,
        input_pairs,
        before,
        config,
        result,
    })
}

/// C-009 pin 4. With an explicit SMALL band the bin exceeds `write_max` and the rolling writer
/// really rolls, so ONE bin produces MORE THAN ONE file — which the pre-G3 action could not do at
/// any configuration, because it wrote a single whole-bin batch to a writer bounded by the 512 MiB
/// DATA default.
///
/// MUTATION COVERAGE: revert `RollingFileWriterBuilder::new(.., write_max, ..)` to
/// `new_with_default_file_size` and the whole bin lands in ONE file. Revert the chunked feed to a
/// single `writer.write(batch)` and it also lands in one file, because `should_roll` is evaluated
/// once per `write`.
#[tokio::test]
async fn test_output_splits_into_multiple_files_at_a_small_explicit_config() {
    let (catalog, _temp, fixture) = split_fixture_run().await;

    assert_eq!(
        fixture.result.rewritten_delete_files_count, 2,
        "both inputs form ONE bin and both are rewritten"
    );
    assert!(
        fixture.result.added_delete_files_count >= 2,
        "ONE bin, MORE THAN ONE output file — got {}",
        fixture.result.added_delete_files_count
    );

    let reloaded = catalog
        .load_table(fixture.table.identifier())
        .await
        .unwrap();
    let outputs = live_pos_delete_files(&reloaded).await;
    assert_eq!(outputs.len(), fixture.result.added_delete_files_count);
    assert!(
        outputs
            .iter()
            .any(|f| f.file_size_in_bytes > fixture.config.target_file_size_bytes),
        "at least one output must be larger than the resolved TARGET — so this fixture would ALSO \
         red under the pin-3 mutant (roll at the target), not merely under a revert to the 512 MiB \
         data default"
    );
}

// ------------------------------------------------------------------------------------------------
// C-025 — the BOUNDED CHUNK FEED. Three FORK-AUTHORED literals with no Java analogue (R-11 / R-14):
// CHUNK_PAIRS = 256, CHUNK_MAX_SERIALIZED_BYTES = 16384, and the /2 footer reservation.
//
// R-16 is the shape of this section: the CONFIG-TIME assert inside `resolve_config` is trivially
// true by construction and is kept only as intent documentation with a tripwire; the RUNTIME
// clearance is established by `test_no_split_output_exceeds_max_file_size`'s MEASURED OUTPUTS.
// ------------------------------------------------------------------------------------------------

/// C-025 pin 1 (C-036 recipe 6 — no fixture). `chunk_budget` is
/// `min(CHUNK_MAX_SERIALIZED_BYTES, (max - write_max) / 2)`, and BOTH limbs are exercised:
///
/// - at the DELETE DEFAULTS the headroom is 26843546, so half of it (13421773) is far above the cap
///   and the CAP binds: `chunk_budget == 16384`;
/// - on a NARROW band the headroom half binds instead: `min 100 / target 200 / max 1000` gives
///   `write_max = 600`, headroom 400, and `chunk_budget == 200`.
///
/// MUTATION COVERAGE: drop the `min(CHUNK_MAX_SERIALIZED_BYTES, ..)` cap and the defaults case reads
/// 13421773. Drop the `/ CHUNK_HEADROOM_FOOTER_SHARE` footer reservation and the narrow case reads
/// 400. Neither mutant is caught by the other case, which is why both are here.
#[tokio::test]
async fn test_feed_chunk_budget_is_the_ruled_constant() {
    let (_temp, table) = config_table(&[]).await;

    let defaults = RewritePositionDeleteFiles::new(table.clone())
        .resolve_config()
        .expect("the delete defaults are a legal config");
    assert_eq!(
        defaults.max_file_size_bytes - defaults.write_max_file_size,
        26_843_546,
        "the candidate-filter headroom at the delete defaults (120795955 - 93952409)"
    );
    assert_eq!(
        defaults.chunk_budget, 16_384,
        "the wide default band lets the fork-authored CAP bind; half the headroom would be 13421773"
    );

    let narrow = RewritePositionDeleteFiles::new(table)
        .min_file_size_bytes(100)
        .target_file_size_bytes(200)
        .max_file_size_bytes(1_000)
        .resolve_config()
        .expect("legal knobs");
    assert_eq!(narrow.write_max_file_size, 600, "200 + (1000 - 200) * 0.5");
    assert_eq!(
        narrow.chunk_budget,
        (narrow.max_file_size_bytes - narrow.write_max_file_size) / 2,
        "on a narrow band the HEADROOM HALF binds, not the 16 KiB cap"
    );
    assert_eq!(
        narrow.chunk_budget, 200,
        "(1000 - 600) / 2 — the footer keeps the other half"
    );

    // RES-8's THROUGHPUT half, pinned rather than left as prose: a band of `max - target <= 2`
    // resolves `chunk_budget` to ZERO, which is below ANY pair's serialized size, so the one-pair
    // floor governs and the feed degrades to one Arrow batch per pair. Legal config, named residue —
    // see `write_compacted_file`'s RES-8 section.
    let (_degenerate_temp, degenerate_table) = config_table(&[]).await;
    let degenerate = RewritePositionDeleteFiles::new(degenerate_table)
        .min_file_size_bytes(100)
        .target_file_size_bytes(200)
        .max_file_size_bytes(202)
        .resolve_config()
        .expect("legal knobs: min < target < max");
    assert_eq!(
        degenerate.write_max_file_size, 201,
        "200 + (202 - 200) * 0.5"
    );
    assert_eq!(
        degenerate.chunk_budget, 0,
        "a two-byte band leaves a one-byte headroom, whose half is ZERO"
    );
}

/// C-025 pin 2 — the CHUNKING RULE itself, white-box on [`chunk_end`], because neither the pair cap
/// nor the one-pair floor is observable end to end (raising `CHUNK_PAIRS` only changes how often
/// `should_roll` runs, and a zero-pair chunk HANGS rather than failing an assertion).
///
/// MUTATION COVERAGE, one per element:
/// - raise or lower `CHUNK_PAIRS` ⇒ the "count cap binds" case reds;
/// - drop the `next > chunk_budget` break ⇒ the "byte cap binds" case reds;
/// - drop the `end > start` one-pair floor ⇒ the floor case returns `start` (0 pairs), which is
///   caught HERE as a failed assertion instead of spinning the feed loop forever;
/// - change `pair_serialized_bytes` from `len + 8` ⇒ the byte-cap boundary moves.
#[test]
fn test_chunk_end_takes_at_least_one_pair_and_respects_both_caps() {
    let pairs: Vec<(String, i64)> = (0..1_000i64).map(|pos| ("aaaa".to_string(), pos)).collect();
    assert_eq!(
        pair_serialized_bytes(&pairs[0]),
        12,
        "one pair measures `file_path.len() + 8` — 4 UTF-8 bytes plus the int64 pos"
    );

    assert_eq!(
        chunk_end(&pairs, 0, u64::MAX),
        CHUNK_PAIRS,
        "with an unbounded byte budget the PAIR CAP binds"
    );
    assert_eq!(
        chunk_end(&pairs, 0, 60),
        5,
        "with a 60-byte budget the BYTE CAP binds: 5 pairs measure exactly 60, a 6th would be 72"
    );
    assert_eq!(
        chunk_end(&pairs, 0, 0),
        1,
        "the ONE-PAIR FLOOR: a zero budget still takes one pair, or the feed loop never terminates"
    );
    assert_eq!(
        chunk_end(&pairs, 998, u64::MAX),
        1_000,
        "a tail shorter than the pair cap ends at the input length, never past it"
    );
}

/// C-025 pin 3 (C-036 RECIPE 3) — THE RUNTIME CLEARANCE PIN, which is where R-16 puts the weight
/// that the config-time assert must NOT be credited with carrying.
///
/// It measures, on real outputs, the two things the config assert cannot see:
/// 1. that a chunk's PARQUET contribution really does fit inside the raw-byte budget it was
///    denominated in (the stated assumption on `write_compacted_file`), and
/// 2. that the Parquet FOOTER — which `current_written_size()` EXCLUDES, and which this action
///    inflates by writing FULL untruncated `file_path` bounds — fits inside its reserved half.
///
/// Together those are what keep every run-1 output inside `[min, max]`, where the candidate filter
/// declines it forever. That containment is C-026's convergence argument, MEASURED here rather than
/// asserted.
#[tokio::test]
async fn test_no_split_output_exceeds_max_file_size() {
    let (catalog, _temp, table, s, t) = recipe_3_lone_oversized_fixture().await;
    let action = || RewritePositionDeleteFiles::new(table.clone()).target_file_size_bytes(t);
    let config = action().resolve_config().expect("legal knobs");
    assert_recipe_3_preconditions(s, &config);

    let before = scan_y_values(&table).await;
    let result = action().execute(&catalog).await.expect("run 1");
    assert_eq!(
        result.rewritten_delete_files_count, 1,
        "the LONE oversized file is admitted by too_much_content, which carries no `size > 1` guard"
    );
    assert_eq!(
        result.added_delete_files_count, 2,
        "and it is rewritten into EXACTLY TWO outputs — one roll at write_max = 1.4T, then a tail"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    let outputs = outputs_in_write_order(&reloaded).await;
    assert_eq!(outputs.len(), 2);

    for (file, _) in &outputs {
        assert!(
            file.file_size_in_bytes >= config.min_file_size_bytes
                && file.file_size_in_bytes <= config.max_file_size_bytes,
            "every run-1 output must land INSIDE [min, max] ({} .. {}) — an output outside it is a \
             candidate again, and the fixed point is gone (measured {})",
            config.min_file_size_bytes,
            config.max_file_size_bytes,
            file.file_size_in_bytes
        );
    }

    let first = outputs[0].0.file_size_in_bytes;
    assert!(
        first > config.write_max_file_size,
        "the FIRST output must EXCEED the roll bound — `should_roll` is a PRE-check, so a rolled \
         file has already passed it (measured {first}, write_max {})",
        config.write_max_file_size
    );
    let overshoot = first - config.write_max_file_size;
    assert!(
        overshoot <= 2 * config.chunk_budget,
        "the overshoot ({overshoot}) must fit inside one chunk PLUS its reserved footer half \
         (2 * chunk_budget = {}) — this is the raw-vs-output assumption and the footer allowance, \
         both MEASURED",
        2 * config.chunk_budget
    );

    let o: u64 = outputs.iter().map(|(f, _)| f.file_size_in_bytes).sum();
    assert!(
        o >= 215 * t / 100 + 2 * config.chunk_budget && o <= 28 * t / 10,
        "the measured TOTAL output O = {o} must sit inside the two-output window \
         [2.15T + 2 * chunk_budget, 2.8T] = [{}, {}]. Above 2.8T the SECOND output itself reaches \
         write_max and rolls, leaving a sub-min THIRD that reds the `[min, max]` assertion above",
        215 * t / 100 + 2 * config.chunk_budget,
        28 * t / 10
    );
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity across the split"
    );
}

// ------------------------------------------------------------------------------------------------
// C-026 — THE FIXED POINT, and the three counterexamples that BOUND it.
//
// This closes the non-convergence G2 deliberately opened: at G2's state a lone file above
// `max_file_size_bytes` was admitted, rewritten to about the same size, and RE-ADMITTED on every
// subsequent run — unbounded churn. The convergence does NOT come from the roll bound directly. It
// comes from the CANDIDATE FILTER: a run-1 output that lands inside [min, max] fails
// `outsideDesiredFileSizeRange`, is therefore never a candidate, and no bin forms.
//
// The claim is CONDITIONAL, not universal, and the two counterexamples below are PARITY-CORRECT —
// Java behaves identically — so they are pinned as EXPECTED behaviour. They are not defects and no
// later change should "fix" them.
// ------------------------------------------------------------------------------------------------

/// C-026 — ACCEPTANCE 3, in its ruled fixed-point form (R-3): the no-op is the three-way
/// CONJUNCTION `rewritten == 0 && added == 0 && current_snapshot_id() unchanged`.
///
/// The MANDATORY pre-assertions on run 1's state come FIRST, so a size drift fails loudly on a
/// precondition instead of reddening the fixed point for an unrelated reason. Per C-029, a red on
/// the three conjuncts THEMSELVES is a DESIGN failure to escalate — never an assert to weaken.
///
/// MUTATION COVERAGE: revert the roll bound to the resolved target and output 1 is ~T, still inside
/// [0.75T, 1.8T], so this test alone would still pass — which is exactly why C-009 pin 3 exists.
/// What this test kills is the far more dangerous class: any change that lets a run-1 output land
/// OUTSIDE [min, max] (drop the chunk cap, drop the footer reservation, roll at `max` instead of
/// `write_max`) re-admits it forever and reds the conjuncts.
#[tokio::test]
async fn test_second_run_is_a_no_op_after_split() {
    let (catalog, _temp, table, s, t) = recipe_3_lone_oversized_fixture().await;
    let action = |t_: Table| RewritePositionDeleteFiles::new(t_).target_file_size_bytes(t);
    let config = action(table.clone()).resolve_config().expect("legal knobs");
    assert_recipe_3_preconditions(s, &config);

    let before = scan_y_values(&table).await;
    let run1 = action(table.clone())
        .execute(&catalog)
        .await
        .expect("run 1");

    // PRE-ASSERTIONS ON RUN 1's STATE — every one on a MEASURED quantity.
    assert_eq!(
        run1.rewritten_delete_files_count, 1,
        "run 1 must admit EXACTLY ONE bin (the lone oversized file), or a counterexample below is \
         firing on this fixture instead of the fixed point"
    );
    assert_eq!(
        run1.added_delete_files_count, 2,
        "run 1 produced EXACTLY 2 outputs"
    );

    let after_run1 = catalog.load_table(table.identifier()).await.unwrap();
    let outputs = outputs_in_write_order(&after_run1).await;
    assert_eq!(outputs.len(), 2);
    for (file, _) in &outputs {
        assert!(
            file.file_size_in_bytes >= config.min_file_size_bytes
                && file.file_size_in_bytes <= config.max_file_size_bytes,
            "THE CONVERGENCE CONDITION: every run-1 output must be inside [min, max] ({} .. {}), \
             where outsideDesiredFileSizeRange declines it forever (measured {})",
            config.min_file_size_bytes,
            config.max_file_size_bytes,
            file.file_size_in_bytes
        );
    }
    let overshoot = outputs[0].0.file_size_in_bytes - config.write_max_file_size;
    assert!(
        outputs[0].0.file_size_in_bytes > config.write_max_file_size
            && overshoot <= 2 * config.chunk_budget,
        "run 1's first output overshoots the roll bound by {overshoot}, which must fit in \
         2 * chunk_budget = {}",
        2 * config.chunk_budget
    );
    let o: u64 = outputs.iter().map(|(f, _)| f.file_size_in_bytes).sum();
    assert!(
        o >= 215 * t / 100 + 2 * config.chunk_budget && o <= 28 * t / 10,
        "run 1's measured total O = {o} must sit inside [2.15T + 2 * chunk_budget, 2.8T] = [{}, {}]",
        215 * t / 100 + 2 * config.chunk_budget,
        28 * t / 10
    );

    // THE FIXED POINT, all three conjuncts.
    let snapshot_before_run2 = after_run1.metadata().current_snapshot_id();
    let run2 = action(after_run1.clone())
        .execute(&catalog)
        .await
        .expect("run 2");
    assert_eq!(
        run2.rewritten_delete_files_count, 0,
        "run 2 rewrites NOTHING — neither run-1 output is a candidate any more"
    );
    assert_eq!(run2.added_delete_files_count, 0, "run 2 adds NOTHING");
    let after_run2 = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        after_run2.metadata().current_snapshot_id(),
        snapshot_before_run2,
        "run 2 commits NO snapshot — zero counts alone would not rule out an empty Replace"
    );
    assert_eq!(
        scan_y_values(&after_run2).await,
        before,
        "read identity across both runs"
    );
}

/// C-026 COUNTEREXAMPLE 2, PINNED AS EXPECTED BEHAVIOUR — **not** a defect, and **not** to be
/// "fixed": Java behaves identically.
///
/// **C-036 RECIPE 11 (NEW — not one of the ledger's nine; see the section banner's disclosure).**
/// C-026 requires this counterexample pinned but C-036 describes no fixture that can build it.
///
/// `B >= min_input_files` bins in one partition each leave a SUB-MIN TAIL. Those tails co-bin on the
/// next run and are admitted by `enough_input_files` — the COUNT clause — so the second run is not a
/// no-op even though every non-tail output converged.
///
/// The fixture ISOLATES the count clause: five ~118 KB files, knobs at `min = 0.80C`,
/// `target = 0.85C`, `max = 0.95C` (so `write_max = 0.90C`) and `max_file_group_size_bytes = 1.05C`,
/// which forces FIVE bins of one. Each bin is admitted by `too_much_content` (its file exceeds max)
/// and splits into a ~0.9C output that lands IN RANGE plus a ~0.11C tail that does not. On run 2 the
/// five tails sum to well UNDER the target, so `enough_content` is false and `too_much_content` is
/// false — the admission is `enough_input_files` and nothing else, which the assertions below pin.
#[tokio::test]
async fn test_multi_bin_tails_are_readmitted_on_second_run() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    let mut deletes = Vec::new();
    for k in 0..5i64 {
        deletes.push(write_sized_pos_delete(&table, &x_path, 1 + k * 20_000, 12_000).await);
    }
    let c = deletes[0].file_size_in_bytes;
    let table = add_deletes(&catalog, &table, deletes).await;

    let action = |t: Table| {
        RewritePositionDeleteFiles::new(t)
            .min_file_size_bytes(c * 80 / 100)
            .target_file_size_bytes(c * 85 / 100)
            .max_file_size_bytes(c * 95 / 100)
            .max_file_group_size_bytes(c * 105 / 100)
    };
    let config = action(table.clone()).resolve_config().expect("legal knobs");

    let before = scan_y_values(&table).await;
    let run1 = action(table.clone())
        .execute(&catalog)
        .await
        .expect("run 1");
    assert_eq!(run1.rewritten_delete_files_count, 5);
    assert_eq!(
        run1.added_delete_files_count, 10,
        "FIVE bins of one, each splitting into an in-range output plus a sub-min tail"
    );

    let after_run1 = catalog.load_table(table.identifier()).await.unwrap();
    let sizes: Vec<u64> = live_pos_delete_files(&after_run1)
        .await
        .iter()
        .map(|f| f.file_size_in_bytes)
        .collect();
    let tails: Vec<u64> = sizes
        .iter()
        .copied()
        .filter(|s| *s < config.min_file_size_bytes)
        .collect();
    assert_eq!(
        tails.len(),
        5,
        "exactly five SUB-MIN tails, one per bin (measured {sizes:?})"
    );
    assert_eq!(
        sizes.len() - tails.len(),
        5,
        "and five outputs INSIDE [min, max], which are no longer candidates"
    );

    // THE ISOLATION: on run 2 the five tails form ONE bin, and ONLY the count clause admits it.
    let tail_sum: u64 = tails.iter().sum();
    assert!(
        tail_sum <= config.max_file_group_size_bytes,
        "the five tails ({tail_sum}) must co-bin under the group-size cap"
    );
    assert!(
        tail_sum <= config.target_file_size_bytes,
        "enough_content must be FALSE ({tail_sum} <= target {}) — otherwise this fixture does not \
         isolate the count clause and duplicates counterexample 3",
        config.target_file_size_bytes
    );
    assert!(
        tail_sum <= config.max_file_size_bytes,
        "too_much_content must be FALSE ({tail_sum} <= max {})",
        config.max_file_size_bytes
    );
    assert!(
        tails.len() >= config.min_input_files,
        "enough_input_files is the ONLY admitter: {} tails >= min_input_files {}",
        tails.len(),
        config.min_input_files
    );

    let run2 = action(after_run1).execute(&catalog).await.expect("run 2");
    assert_eq!(
        run2.rewritten_delete_files_count, 5,
        "PARITY-CORRECT, not a defect: the five sub-min tails ARE re-admitted, by the count clause"
    );
    assert_eq!(run2.added_delete_files_count, 1);

    let after_run2 = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&after_run2).await,
        before,
        "read identity across both runs"
    );
}

/// **C-036 RECIPE 12 (NEW — not one of the ledger's nine; see the section banner's disclosure).**
/// As with recipe 11, C-026 requires this counterexample pinned and C-036 describes no fixture for it.
///
/// C-026 COUNTEREXAMPLE 3, DISTINCT from counterexample 2 and likewise PINNED AS EXPECTED BEHAVIOUR:
/// as few as **TWO** bins whose sub-min tails SUM ABOVE THE TARGET are re-admitted, by
/// `enough_content`, which needs only `size > 1`. Two is far below the default count floor of five,
/// which is what makes this a separate bound on the fixed-point claim rather than a weaker case of
/// counterexample 2.
///
/// The fixture ISOLATES the content clause: two ~118 KB files, knobs at `min = 0.55C`,
/// `target = 0.60C`, `max = 0.75C` (so `write_max = 0.675C`) and `max_file_group_size_bytes = 1.05C`,
/// forcing TWO bins of one. Each splits into an in-range ~0.675C output plus a ~0.32C sub-min tail.
/// On run 2 the two tails sum ABOVE the target but BELOW the max, and two is below the DEFAULT count
/// floor, so `enough_content` is the sole admitter.
#[tokio::test]
async fn test_two_bin_tails_over_target_are_readmitted() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    let mut deletes = Vec::new();
    for k in 0..2i64 {
        deletes.push(write_sized_pos_delete(&table, &x_path, 1 + k * 20_000, 12_000).await);
    }
    let c = deletes[0].file_size_in_bytes;
    let table = add_deletes(&catalog, &table, deletes).await;

    let action = |t: Table| {
        RewritePositionDeleteFiles::new(t)
            .min_file_size_bytes(c * 55 / 100)
            .target_file_size_bytes(c * 60 / 100)
            .max_file_size_bytes(c * 75 / 100)
            .max_file_group_size_bytes(c * 105 / 100)
    };
    let config = action(table.clone()).resolve_config().expect("legal knobs");

    let before = scan_y_values(&table).await;
    let run1 = action(table.clone())
        .execute(&catalog)
        .await
        .expect("run 1");
    assert_eq!(run1.rewritten_delete_files_count, 2);
    assert_eq!(
        run1.added_delete_files_count, 4,
        "TWO bins of one, each splitting in two"
    );

    let after_run1 = catalog.load_table(table.identifier()).await.unwrap();
    let sizes: Vec<u64> = live_pos_delete_files(&after_run1)
        .await
        .iter()
        .map(|f| f.file_size_in_bytes)
        .collect();
    let tails: Vec<u64> = sizes
        .iter()
        .copied()
        .filter(|s| *s < config.min_file_size_bytes)
        .collect();
    assert_eq!(
        tails.len(),
        2,
        "exactly two SUB-MIN tails (measured {sizes:?})"
    );

    let tail_sum: u64 = tails.iter().sum();
    assert!(
        tails.len() < config.min_input_files,
        "enough_input_files must be FALSE: {} tails < the DEFAULT floor {}",
        tails.len(),
        config.min_input_files
    );
    assert!(
        tail_sum > config.target_file_size_bytes,
        "enough_content is the ONLY admitter: the two tails sum to {tail_sum} > target {}",
        config.target_file_size_bytes
    );
    assert!(
        tail_sum <= config.max_file_size_bytes,
        "too_much_content must be FALSE ({tail_sum} <= max {})",
        config.max_file_size_bytes
    );

    let run2 = action(after_run1).execute(&catalog).await.expect("run 2");
    assert_eq!(
        run2.rewritten_delete_files_count, 2,
        "PARITY-CORRECT, not a defect: TWO bins are enough — the content clause carries no count floor"
    );
    assert_eq!(run2.added_delete_files_count, 1);

    let after_run2 = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&after_run2).await,
        before,
        "read identity across both runs"
    );
}

// ------------------------------------------------------------------------------------------------
// C-010 — `write_compacted_file` returns `Vec<DataFile>`, and EVERY output of a bin is stamped with
// THAT BIN's own max rewritten data-seq.
//
// The stamp is a function of exactly two variables: which ENTRIES the max ranges over, and which
// OUTPUTS receive it. `test_compacted_file_carries_bin_max_rewritten_seq` (above) fixes the base
// value on a one-output bin; the two pins below fix the other two dimensions.
//
// DIRECTION OF DANGER, against the fork's OWN rule (`delete_file_index.rs`'s applicable_pos_deletes
// keeps `delete_seq >= data_seq`): an OVER-HIGH stamp OVER-APPLIES; an UNDER-LOW stamp stops
// applying and RESURRECTS rows. Stamping bin 2 from bin 1's entries is the UNDER-stamp path whenever
// bin 1's max is the lower one — the resurrection direction, and the reason the ranging dimension
// gets its own pin rather than being folded into the fan-out one.
// ------------------------------------------------------------------------------------------------

/// C-010 element 2 — the FAN-OUT dimension. A bin that SPLITS must stamp EVERY output with the bin
/// max, not just the first one the old `.next()` shape happened to return.
///
/// MUTATION COVERAGE: stamp only `new_files[0]` and let the rest inherit (i.e. `add_delete_file`)
/// and the second output carries the rewrite snapshot's seq (4) instead of 3 — an OVER-HIGH stamp,
/// which over-applies. Asserting only `pos_entries[0]` would miss it, which is why this iterates.
#[tokio::test]
async fn test_every_split_output_carries_bin_max_rewritten_seq() {
    let (catalog, _temp, fixture) = split_fixture_run().await;
    assert!(
        fixture.result.added_delete_files_count >= 2,
        "fixture: the bin must SPLIT, or this pin tests the same thing as the one-output pin"
    );

    let reloaded = catalog
        .load_table(fixture.table.identifier())
        .await
        .unwrap();
    let stamped: Vec<Option<i64>> = live_delete_entries_with_seq(&reloaded)
        .await
        .into_iter()
        .filter(|(f, _)| f.content_type() == DataContentType::PositionDeletes)
        .map(|(_, seq)| seq)
        .collect();
    assert_eq!(stamped.len(), fixture.result.added_delete_files_count);
    for seq in &stamped {
        assert_eq!(
            *seq,
            Some(3),
            "EVERY output of the bin carries the bin max (3) — the inputs are at seqs 2 and 3, and \
             the rewrite snapshot's own seq is 4 (measured {stamped:?})"
        );
    }
    assert_eq!(
        scan_y_values(&reloaded).await,
        fixture.before,
        "read identity"
    );
}

/// C-010 element 3 — the RANGING dimension, and the pin that kills the two most dangerous mutants.
///
/// ONE partition, FOUR position-delete files committed in FOUR separate snapshots so their data
/// sequence numbers are 2, 3, 5 and 6 (a dummy data-file append sits at seq 4). The group-size cap
/// is `2 * max(S_i)`, with `< 3 * min(S_i)` asserted, so the sequential first-fit packer puts
/// exactly TWO files in each bin — whatever the manifest order turns out to be.
///
/// The expected stamp per bin is DERIVED from what each output actually contains (each input masks a
/// disjoint 1000-block of positions, so an output's positions name its bin's members), never assumed
/// from an unpinned manifest ordering. The test then asserts the two bins' expected maxima DIFFER,
/// which is what makes both mutants lethal:
///
/// MUTATIONS, APPLIED:
/// - range the max over the PARTITION's whole entry list ⇒ both outputs carry 6 ⇒ the lower bin reds;
/// - compute bin 2's max from bin 1's entries ⇒ the second output carries the first bin's value ⇒
///   reds (and that is the UNDER-stamp, row-resurrection direction).
#[tokio::test]
async fn test_each_bin_output_carries_its_own_bin_max_not_the_partition_max() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    // Four pos-deletes over DISJOINT 1000-blocks, with distinct sizes, each in its own snapshot.
    let mut table = table;
    let mut by_block: HashMap<i64, i64> = HashMap::new(); // pos/1000 block -> data seq
    for (index, count) in [2i64, 3, 4, 5].into_iter().enumerate() {
        let block = 1 + index as i64;
        let pd = write_sized_pos_delete(&table, &x_path, block * 1_000, count).await;
        table = add_deletes(&catalog, &table, vec![pd]).await;
        if index == 1 {
            // A dummy data-file append between the bins, so the seqs are 2, 3, 5, 6 and the two
            // bins' maxima cannot coincide by accident.
            let w = write_data_file(&table, "w.parquet", 0, &[(0, 60, 600)]).await;
            table = append_files(&catalog, &table, vec![w]).await;
        }
        let seq = live_delete_entries_with_seq(&table)
            .await
            .into_iter()
            .filter(|(f, _)| f.content_type() == DataContentType::PositionDeletes)
            .filter_map(|(_, seq)| seq)
            .max()
            .expect("the just-committed pos-delete carries a seq");
        by_block.insert(block, seq);
    }
    let mut seqs: Vec<i64> = by_block.values().copied().collect();
    seqs.sort();
    assert_eq!(seqs, vec![2, 3, 5, 6], "fixture: the four input seqs");

    let sizes: Vec<u64> = live_pos_delete_files(&table)
        .await
        .iter()
        .map(|f| f.file_size_in_bytes)
        .collect();
    assert_eq!(sizes.len(), 4);
    let group_size = 2 * sizes.iter().copied().max().expect("four sizes");
    assert!(
        group_size < 3 * sizes.iter().copied().min().expect("four sizes"),
        "fixture: TWO files must always fit a bin and THREE must never — this makes the bin \
         MEMBERSHIP independent of the manifest order (measured {sizes:?})"
    );

    let action = || {
        RewritePositionDeleteFiles::new(table.clone())
            .min_input_files(2)
            .min_file_size_bytes(100_000)
            .target_file_size_bytes(200_000)
            .max_file_size_bytes(400_000)
            .max_file_group_size_bytes(group_size)
    };
    let config = action().resolve_config().expect("legal knobs");
    for size in &sizes {
        assert!(
            *size < config.min_file_size_bytes,
            "fixture: all four are sub-min candidates"
        );
    }

    let before = scan_y_values(&table).await;
    let result = action().execute(&catalog).await.expect("execute");
    assert_eq!(result.rewritten_delete_files_count, 4);
    assert_eq!(result.added_delete_files_count, 2, "two bins ⇒ two outputs");

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    let mut observed: Vec<(i64, i64)> = Vec::new(); // (expected bin max, stamped seq)
    for (file, seq) in live_delete_entries_with_seq(&reloaded).await {
        if file.content_type() != DataContentType::PositionDeletes {
            continue;
        }
        let pairs = read_pos_delete_pairs(&reloaded, &file).await;
        let mut blocks: Vec<i64> = pairs.iter().map(|(_, pos)| pos / 1_000).collect();
        blocks.sort();
        blocks.dedup();
        let expected = blocks
            .iter()
            .map(|block| by_block[block])
            .max()
            .expect("an output covers at least one input block");
        observed.push((
            expected,
            seq.expect("the compacted file carries an explicit seq"),
        ));
    }
    assert_eq!(observed.len(), 2);
    assert_ne!(
        observed[0].0, observed[1].0,
        "NON-VACUITY: the two bins' expected maxima must DIFFER, or neither mutant is lethal \
         (observed {observed:?})"
    );
    for (expected, stamped) in &observed {
        assert_eq!(
            stamped, expected,
            "each bin's output carries ITS OWN bin max — not the partition's, and not the other \
             bin's (observed {observed:?})"
        );
    }
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

/// C-010's counting half: `added_delete_files_count` is the REAL number of files added across the
/// split, and `added_bytes_count` is the CHECKED sum of their sizes — not the hard-coded `+= 1` and
/// single `file_size_in_bytes` the one-output shape carried.
///
/// MUTATION COVERAGE: restore `added_delete_files_count += 1` and the count reds against the live
/// pos-delete count; sum only the first output's bytes and the byte assertion reds.
#[tokio::test]
async fn test_result_counts_added_files_and_bytes_across_split_outputs() {
    let (catalog, _temp, fixture) = split_fixture_run().await;

    let reloaded = catalog
        .load_table(fixture.table.identifier())
        .await
        .unwrap();
    let outputs = live_pos_delete_files(&reloaded).await;
    assert!(outputs.len() >= 2, "fixture: the bin must SPLIT");

    assert_eq!(
        fixture.result.added_delete_files_count,
        outputs.len(),
        "the added COUNT is the real number of files the split produced"
    );
    assert_eq!(
        fixture.result.added_bytes_count,
        outputs.iter().map(|f| f.file_size_in_bytes).sum::<u64>(),
        "the added BYTES are the sum across EVERY output, not just the first"
    );
    assert_eq!(
        fixture.result.rewritten_bytes_count,
        fixture.input_sizes.iter().sum::<u64>(),
        "and the rewritten side is unchanged by the fan-out"
    );
}

// ------------------------------------------------------------------------------------------------
// C-044 — the GLOBAL SORT happens BEFORE any split, and the split preserves it.
// ------------------------------------------------------------------------------------------------

/// C-044. A bin's pairs are sorted by `(file_path, pos)` ONCE, globally, and the feed then chunks
/// that already-sorted `Vec` in order — so output *k*'s range lies entirely below output *k+1*'s and
/// the union of the outputs is EXACTLY the input multiset.
///
/// Java does not dedup within a group either (the reader bitmap dedups), so the multiset equality is
/// the right relation, not set equality.
///
/// MUTATION COVERAGE: move `pairs.sort()` below the chunking, or sort per chunk instead of globally,
/// and the ranges interleave — output *k*'s max exceeds output *k+1*'s min. The multiset equality
/// and the read-identity assertion both survive that mutation, which is precisely why the ORDERING
/// assertion has to be here: the masked row set alone cannot see it.
#[tokio::test]
async fn test_split_outputs_have_disjoint_ascending_ranges() {
    let (catalog, _temp, fixture) = split_fixture_run().await;
    assert!(
        fixture.result.added_delete_files_count >= 2,
        "fixture: the bin must SPLIT, or there are no ranges to be disjoint"
    );

    // NON-VACUITY, ON THE MEASURED INPUT SEQUENCE — not on the fixture's own construction.
    //
    // Everything below can only distinguish a GLOBAL sort from no sort at all if what the writer is
    // FED is not already ascending. `input_concat` is the two inputs read back off disk and
    // concatenated in MANIFEST ORDER, i.e. exactly the sequence `compact_group` builds before it
    // sorts; asserting that THAT is unsorted is the property, and a fixture edit that makes the two
    // inputs consecutive-and-disjoint again reds HERE.
    //
    // This replaces an earlier guard that asserted `input_pairs.len() == 24_000` while
    // `input_pairs` was built four lines above as `(1..=24_000).map(..)` — an unconditional
    // identity in the fixture's own construction that could never red. Reverting only the two
    // stride arguments and touching no assertion left the suite green
    // under the sort mutation. Same over-claim class R-16 corrected on the config-time assert.
    //
    // The witness form is deliberate: ONE descending adjacent pair is logically equivalent to "not
    // already ascending", and it fails in one line. Comparing the whole `Vec` against its own sorted
    // clone would emit a ~3.9 MB panic message for the same information. Not asserted, because it
    // would be the SAME unkillable class this round deleted from `assert_recipe_3_preconditions`:
    // `input_pairs` IS `sort(clone(input_concat))` by construction (see `split_fixture_run`), so
    // re-deriving it here and comparing could never red. The multiset relation that CAN red is the
    // union-vs-`input_pairs` assertion at the end of this test, against the real split OUTPUTS.
    assert!(
        fixture
            .input_concat
            .windows(2)
            .any(|pair| pair[0] > pair[1]),
        "fixture: the concatenated input MUST NOT already be ascending, or a global sort and no \
         sort are indistinguishable and every ordering assertion below is vacuous"
    );

    let reloaded = catalog
        .load_table(fixture.table.identifier())
        .await
        .unwrap();
    let outputs = outputs_in_write_order(&reloaded).await;
    assert_eq!(outputs.len(), fixture.result.added_delete_files_count);

    for (file, pairs) in &outputs {
        assert!(!pairs.is_empty(), "no output is empty");
        let mut sorted = pairs.clone();
        sorted.sort();
        assert_eq!(
            *pairs,
            sorted,
            "each output's pairs are ASCENDING in (file_path, pos) — file {}",
            file.file_path()
        );
    }
    for window in outputs.windows(2) {
        let left_max = window[0].1.last().expect("non-empty");
        let right_min = window[1].1.first().expect("non-empty");
        assert!(
            left_max <= right_min,
            "output ranges must be DISJOINT and ASCENDING: {left_max:?} then {right_min:?}"
        );
    }

    let mut union: Vec<(String, i64)> = outputs.iter().flat_map(|(_, p)| p.clone()).collect();
    union.sort();
    assert_eq!(
        union, fixture.input_pairs,
        "the union of the split outputs is EXACTLY the input multiset — no pair lost, none invented"
    );
    assert_eq!(
        scan_y_values(&reloaded).await,
        fixture.before,
        "read identity across the split"
    );
}

// ------------------------------------------------------------------------------------------------
// C-046 — the FAIL-CLOSED guard SURVIVES the `Vec<DataFile>` change at the new arity.
// ------------------------------------------------------------------------------------------------

/// C-046. A bin whose `pairs` are NON-EMPTY must produce at least one file. "No file" is a NORMAL
/// return from the parquet writer, not an error — `ParquetWriter::close` returns `Ok(vec![])` and
/// DELETES the output whenever `current_row_num == 0` — so nothing below this guard would object.
///
/// With the guard removed, `execute` goes on to commit a `Replace` snapshot that REMOVES live
/// position-delete files and adds none: silent UNDER-masking, the row-RESURRECTION direction. Nothing
/// downstream rejects it either, because `RewriteFilesAction::validate` early-returns when
/// `deleted_data_files` is empty, which is always true for this action.
///
/// WHITE-BOX on the extracted [`require_non_empty`], because the state is unreachable end to end:
/// the guard's whole point is that it fires where no fixture can put the writer.
///
/// MUTATION COVERAGE: delete the guard (return `Ok(files)` unconditionally) and the empty case
/// returns `Ok(vec![])`.
#[tokio::test]
async fn test_bin_with_pairs_but_no_output_file_is_a_hard_error() {
    let error = require_non_empty(Vec::new()).expect_err("an empty file list is a hard error");
    assert_eq!(error.kind(), ErrorKind::Unexpected);
    assert!(
        error
            .message()
            .contains("Position-delete writer produced no file for a non-empty input"),
        "the guard keeps its exact message: {error}"
    );

    let (_catalog, _temp, table, x_path) = gate_table().await;
    let file = write_sized_pos_delete(&table, &x_path, 1, 1).await;
    let passed = require_non_empty(vec![file.clone()]).expect("a non-empty list passes through");
    assert_eq!(passed.len(), 1);
    assert_eq!(
        passed[0].file_path(),
        file.file_path(),
        "and is returned UNCHANGED"
    );
}

// =================================================================================================
// THE COMMIT LOOP — one `RewriteFiles` per admitted BIN (C-011), the abort contract (C-037), and
// the per-bin zero-pairs skip (C-040).
//
// C-011 and C-037 both run on C-036 RECIPE 7, so the fixture is built once here. Every window the
// knob derivation makes tautological is RECORDED as true by construction below and its falsifiable
// content moved onto MEASURED quantities — the bin COMPOSITION is read back off the outputs rather
// than assumed from the packer's arithmetic.
// =================================================================================================

/// C-036 RECIPE 7 — TWO ADMITTED BINS in ONE partition.
///
/// FOUR position-delete files A, B, C, D in MANIFEST order, each masking a DISJOINT 1000-block of
/// positions so an output's pairs NAME the bin members that produced it. Sizes are MEASURED, never
/// predicted, and the knobs are set around them.
struct Recipe7 {
    table: Table,
    /// `S_A .. S_D`, MEASURED, in MANIFEST order.
    sizes: [u64; 4],
    /// A, B, C, D's file paths, in MANIFEST order.
    paths: [String; 4],
    /// `W` — the `max_file_group_size_bytes` knob, derived from the measured sizes.
    group_size: u64,
}

/// `m` — recipe 7's `min_file_size_bytes` knob. FIXED, not derived: every measured `S_i` is asserted
/// BELOW it, so a size drift that pushed a position-delete file past 100 KB reds on the precondition
/// instead of silently emptying the candidate set. `m < target` (200_000) closes C-006's (3).
const RECIPE_7_MIN: u64 = 100_000;

/// Build C-036 recipe 7. `W := max(S_A + S_B, S_C + S_D)`.
///
/// RECORDED as TRUE BY CONSTRUCTION and deliberately NOT asserted (C-036: no window assert may be an
/// algebraic identity in the fixture size): `S_A + S_B <= W` and `S_C + S_D <= W`, both immediate
/// from `W := max(..)`. The remaining leg, `W < S_A + S_B + S_C`, is FALSIFIABLE — it holds only
/// while `S_D < S_A + S_B` — and IS asserted. The bin composition the three legs are supposed to
/// force is then read back off the MEASURED outputs by the pins themselves.
async fn recipe_7_two_bin_fixture() -> (impl Catalog, TempDir, Recipe7) {
    let (catalog, temp, table, x_path) = gate_table().await;

    // Disjoint 1000-blocks (block = pos / 1000) with distinct pair counts, so the sizes differ and
    // each output's pairs identify its bin's members.
    let pd_a = write_sized_pos_delete(&table, &x_path, 1_000, 1).await;
    let pd_b = write_sized_pos_delete(&table, &x_path, 2_000, 2).await;
    let pd_c = write_sized_pos_delete(&table, &x_path, 3_000, 3).await;
    let pd_d = write_sized_pos_delete(&table, &x_path, 4_000, 4).await;
    let sizes = [
        pd_a.file_size_in_bytes,
        pd_b.file_size_in_bytes,
        pd_c.file_size_in_bytes,
        pd_d.file_size_in_bytes,
    ];
    let paths = [
        pd_a.file_path().to_string(),
        pd_b.file_path().to_string(),
        pd_c.file_path().to_string(),
        pd_d.file_path().to_string(),
    ];
    let table = add_deletes(&catalog, &table, vec![pd_a, pd_b, pd_c, pd_d]).await;

    assert_eq!(
        live_pos_delete_paths(&table).await,
        paths.to_vec(),
        "fixture: the manifest order must be A, B, C, D — the bin boundaries depend on it"
    );

    let group_size = (sizes[0] + sizes[1]).max(sizes[2] + sizes[3]);
    (catalog, temp, Recipe7 {
        table,
        sizes,
        paths,
        group_size,
    })
}

/// Recipe 7's knobs, as a fresh action every call (`execute` consumes `self`).
fn recipe_7_action(fixture: &Recipe7) -> RewritePositionDeleteFiles {
    RewritePositionDeleteFiles::new(fixture.table.clone())
        .min_input_files(2)
        .min_file_size_bytes(RECIPE_7_MIN)
        .target_file_size_bytes(200_000)
        .max_file_size_bytes(400_000)
        .max_file_group_size_bytes(fixture.group_size)
}

/// Recipe 7's mandatory PRE-`execute` preconditions, all over MEASURED sizes in MANIFEST order.
fn assert_recipe_7_preconditions(fixture: &Recipe7, config: &ResolvedConfig) {
    for size in &fixture.sizes {
        assert!(
            *size < config.min_file_size_bytes,
            "fixture: every file must be SUB-MIN so all four are candidates \
             (measured {size}, resolved min {})",
            config.min_file_size_bytes
        );
    }
    assert!(
        2 >= config.min_input_files,
        "fixture: a bin of two must clear the count floor (resolved {})",
        config.min_input_files
    );
    assert!(
        config.max_file_group_size_bytes < fixture.sizes[0] + fixture.sizes[1] + fixture.sizes[2],
        "fixture: C must NOT fit alongside A and B — this is what forces the split (W {}, \
         S_A + S_B + S_C {})",
        config.max_file_group_size_bytes,
        fixture.sizes[0] + fixture.sizes[1] + fixture.sizes[2]
    );
    // RECORDED, not asserted: `S_A + S_B <= W` and `S_C + S_D <= W` are identities in `W := max(..)`.
    assert!(
        config.write_max_file_size > fixture.sizes.iter().sum::<u64>(),
        "fixture: write_max must sit far above BOTH bins, so each bin emits exactly ONE output and \
         the split arity cannot confound the commit count (write_max {}, total input {})",
        config.write_max_file_size,
        fixture.sizes.iter().sum::<u64>()
    );
}

/// The 1000-BLOCKS each live position-delete file covers, one sorted+deduped `Vec` per output,
/// themselves sorted so the comparison does not depend on manifest order.
async fn output_blocks(table: &Table) -> Vec<Vec<i64>> {
    let mut all = Vec::new();
    for file in live_pos_delete_files(table).await {
        let mut blocks: Vec<i64> = read_pos_delete_pairs(table, &file)
            .await
            .iter()
            .map(|(_, pos)| pos / 1_000)
            .collect();
        blocks.sort();
        blocks.dedup();
        all.push(blocks);
    }
    all.sort();
    all
}

/// The snapshots appended to `table`'s history AFTER index `from`, oldest first.
fn snapshots_after(table: &Table, from: usize) -> Vec<SnapshotRef> {
    table.metadata().history()[from..]
        .iter()
        .map(|log| {
            table
                .metadata()
                .snapshot_by_id(log.snapshot_id)
                .expect("every history entry resolves to a snapshot")
                .clone()
        })
        .collect()
}

/// One snapshot's `added-delete-files` / `removed-delete-files` summary counters.
fn delete_file_counters(snapshot: &Snapshot) -> (Option<String>, Option<String>) {
    let props = &snapshot.summary().additional_properties;
    (
        props.get("added-delete-files").cloned(),
        props.get("removed-delete-files").cloned(),
    )
}

// ------------------------------------------------------------------------------------------------
// C-011 — exactly ONE `RewriteFiles` (one Replace snapshot) per admitted BIN.
// ------------------------------------------------------------------------------------------------

/// C-011. `execute` iterates BINS, not partitions, and commits ONE `RewriteFiles` per admitted bin:
/// a single partition packed into TWO bins produces TWO `Replace` snapshots, chained (each built on
/// the previous bin's committed tip), each removing exactly its OWN bin's inputs and adding exactly
/// its own output.
///
/// The correctness of committing sequentially against a FIXED `starting_snapshot_id` rests on two
/// facts verified at source: the bins replace DISJOINT delete-file sets (`pack_bins` partitions the
/// candidate list; group keys are disjoint), and `RewriteFilesAction::validate` early-returns when
/// `deleted_data_files` is empty — always true here, because this action passes
/// `rewrite_files(Vec::new(), Vec::new())` and only `delete_delete_files`.
///
/// NON-VACUITY: the two bins' MEMBERSHIP is not assumed from the packer's arithmetic — it is read
/// back off the outputs' `(file_path, pos)` pairs, each input file masking a disjoint 1000-block. A
/// commit shape that merged the partition into one bin, or split it differently, would show up in
/// the block coverage as well as in the snapshot count. The read-identity assertion at the end is a
/// REGRESSION guard, not the discriminating one: this fixture's inputs mask positions 1000..4004 of
/// a FIVE-row data file, so no live row is masked and the row set is full either way (the same
/// disclosure `test_bin_commit_failure_leaves_earlier_bins_committed` carries). `output_blocks` is
/// what does the work — design call 2's "never read identity alone" is satisfied by it.
///
/// RECORDED, NOT PINNED — the base-advance conjunct. C-011's proposition also says "the base table
/// is advanced after each bin commit". That leg is UNPINNABLE BY OBSERVATION: pointing the loop at
/// `self.table` instead of the advanced `table` leaves the WHOLE lib suite green, because
/// `Transaction::do_commit` refreshes a stale base and re-applies, so the two commits chain either
/// way. The advance is a re-apply-COST optimisation, exactly as the merged comment above the loop
/// says ("not required for CAS correctness under the retry/refresh loop"). The `parent_snapshot_id`
/// assertion below therefore pins only what it CAN — that the two commits CHAIN rather than FORK —
/// and deliberately does not imply the optimisation, the same discipline applied to recipe 7's
/// `S_A + S_B <= W` window.
///
/// MUTATION, APPLIED: replace `plan_bins`'s S5 + S6 block with `admitted.push((key, candidates))`
/// (one bin per PARTITION, the pre-unit shape). RED — one snapshot instead of two, one added file
/// instead of two, and the surviving output covers all four blocks.
#[tokio::test]
async fn test_one_rewrite_files_commit_per_bin() {
    let (catalog, _temp, fixture) = recipe_7_two_bin_fixture().await;
    let config = recipe_7_action(&fixture)
        .resolve_config()
        .expect("recipe 7's knobs are legal");
    assert_recipe_7_preconditions(&fixture, &config);

    let before = scan_y_values(&fixture.table).await;
    let history_before = fixture.table.metadata().history().len();

    let result = recipe_7_action(&fixture)
        .execute(&catalog)
        .await
        .expect("both bins commit");
    assert_eq!(
        result.rewritten_delete_files_count, 4,
        "all four candidates are rewritten, across TWO bins"
    );
    assert_eq!(
        result.added_delete_files_count, 2,
        "one compacted output per BIN"
    );

    let reloaded = catalog
        .load_table(fixture.table.identifier())
        .await
        .expect("reload");
    let new_snapshots = snapshots_after(&reloaded, history_before);
    assert_eq!(
        new_snapshots.len(),
        2,
        "TWO admitted bins ⇒ TWO commits, never one batched `RewriteFiles`"
    );
    for snapshot in &new_snapshots {
        assert_eq!(
            snapshot.summary().operation,
            Operation::Replace,
            "every bin commit is a Replace snapshot (Java `newRewrite()`)"
        );
        assert_eq!(
            delete_file_counters(snapshot),
            (Some("1".to_string()), Some("2".to_string())),
            "each snapshot replaces exactly ITS OWN bin: 2 position-deletes out, 1 in"
        );
    }
    assert_eq!(
        new_snapshots[1].parent_snapshot_id(),
        Some(new_snapshots[0].snapshot_id()),
        "the two bin commits CHAIN — the second's parent is the first, so the bins do not FORK \
         from a common base. This does NOT pin the base-advance optimisation; see the rustdoc."
    );

    // The bins the packer actually formed, MEASURED off the outputs.
    assert_eq!(
        output_blocks(&reloaded).await,
        vec![vec![1, 2], vec![3, 4]],
        "the two outputs cover {{A, B}} and {{C, D}} — one output per bin, and the bin boundary is \
         where the group-size knob put it"
    );
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

// ------------------------------------------------------------------------------------------------
// C-037 — the abort contract: earlier bins STAND, no partial result reaches the caller.
// ------------------------------------------------------------------------------------------------

/// C-037. A bin commit failure aborts `execute` with `Err`; the bins committed BEFORE it STAND and
/// are NOT rolled back; the caller receives NO partial [`RewritePositionDeleteFilesResult`].
///
/// The failure is injected by DELETING bin 2's first input file off disk after the fixture is
/// committed — deterministic, because within a partition the entry order is manifest order, so the
/// packer's bins are `{A, B}` then `{C, D}` and `execute` reaches `{C, D}` only after `{A, B}` has
/// already committed. The sabotage is asserted to have APPLIED before `execute` runs; a fixture in
/// which the file is not where the manifest says it is HARD-FAILS rather than proving nothing.
///
/// This RETAINS the pre-unit behaviour and makes it explicit — the bin change only multiplies the
/// windows in which the state arises, from one per partition to one per bin. The same contract
/// covers the `DataInvalid` raised mid-loop when a bin's `spec_id` is absent from table metadata.
///
/// MUTATION, APPLIED: swallow the per-bin error in `execute` (`match .. { Ok(t) => table = t, Err(_)
/// => continue }`). RED — `execute` returns `Ok` with a partial result. A rollback of bin 1 would
/// red the "bin 1 STANDS" assertions instead.
#[tokio::test]
async fn test_bin_commit_failure_leaves_earlier_bins_committed() {
    let (catalog, _temp, fixture) = recipe_7_two_bin_fixture().await;
    let config = recipe_7_action(&fixture)
        .resolve_config()
        .expect("recipe 7's knobs are legal");
    assert_recipe_7_preconditions(&fixture, &config);

    let before = scan_y_values(&fixture.table).await;
    let history_before = fixture.table.metadata().history().len();

    // SABOTAGE — bin 2's first member (C, the third file in manifest order). HARD-FAIL if it cannot
    // be applied: a sabotage that corrupted nothing has proven nothing.
    let victim = fixture.paths[2]
        .strip_prefix("file://")
        .unwrap_or(&fixture.paths[2])
        .to_string();
    assert!(
        std::path::Path::new(&victim).is_file(),
        "sabotage precondition: bin 2's input must be on disk at the path the manifest names \
         ({victim})"
    );
    std::fs::remove_file(&victim).expect("remove bin 2's input file");
    assert!(
        !std::path::Path::new(&victim).exists(),
        "sabotage must have APPLIED before execute runs"
    );

    let error = recipe_7_action(&fixture)
        .execute(&catalog)
        .await
        .expect_err("bin 2 cannot be read, so execute ABORTS");
    assert_eq!(
        error.kind(),
        ErrorKind::DataInvalid,
        "the abort carries the READ failure's typed kind: {error}"
    );
    assert!(
        error.to_string().contains(&victim),
        "and it names the file the sabotage removed, so the test cannot pass on some unrelated \
         failure (victim {victim}, error {error})"
    );

    let reloaded = catalog
        .load_table(fixture.table.identifier())
        .await
        .expect("reload");
    let new_snapshots = snapshots_after(&reloaded, history_before);
    assert_eq!(
        new_snapshots.len(),
        1,
        "bin 1 committed and STANDS; bin 2 never did, and NOTHING was rolled back"
    );
    assert_eq!(
        new_snapshots[0].summary().operation,
        Operation::Replace,
        "the surviving commit is bin 1's Replace snapshot"
    );
    assert_eq!(
        delete_file_counters(&new_snapshots[0]),
        (Some("1".to_string()), Some("2".to_string())),
        "and it replaced exactly BIN 1: A and B out, one compacted output in"
    );

    let live = live_pos_delete_paths(&reloaded).await;
    assert_eq!(
        live.len(),
        3,
        "bin 1's output plus bin 2's two untouched inputs (live: {live:?})"
    );
    assert!(
        !live.contains(&fixture.paths[0]) && !live.contains(&fixture.paths[1]),
        "bin 1's rewritten files are GONE — its commit was not undone (live: {live:?})"
    );
    assert!(
        live.contains(&fixture.paths[2]) && live.contains(&fixture.paths[3]),
        "bin 2's inputs are still live — the failed bin changed nothing (live: {live:?})"
    );
    let survivor = live
        .iter()
        .find(|path| !fixture.paths.contains(path))
        .expect("bin 1's new file is live");
    let survivor_file = live_pos_delete_files(&reloaded)
        .await
        .into_iter()
        .find(|f| f.file_path() == survivor)
        .expect("the new file resolves");
    let mut blocks: Vec<i64> = read_pos_delete_pairs(&reloaded, &survivor_file)
        .await
        .iter()
        .map(|(_, pos)| pos / 1_000)
        .collect();
    blocks.sort();
    blocks.dedup();
    assert_eq!(
        blocks,
        vec![1, 2],
        "the live new file is BIN 1's output — it carries A's and B's blocks and nothing else"
    );
    assert_eq!(
        read_pos_delete_pairs(&reloaded, &survivor_file).await.len(),
        3,
        "and it carries EVERY pair bin 1's two inputs held — A's 1 plus B's 2 — so bin 1's commit \
         masks exactly what it replaced"
    );

    // A post-`execute` read-identity scan is deliberately NOT asserted: the sabotage physically
    // removed a delete file that is still LIVE in metadata, so any scan of this table now fails on
    // the missing bytes. `before` is captured pre-sabotage only to document what the fixture masked.
    assert_eq!(
        before,
        HashSet::from([10, 20, 30, 40, 50]),
        "fixture: the inputs mask positions no row occupies, so the pre-execute row set is full"
    );
}

// ------------------------------------------------------------------------------------------------
// C-040 — an admitted BIN yielding ZERO pairs is skipped PER BIN.
// ------------------------------------------------------------------------------------------------

/// Write a genuinely ZERO-ROW parquet position-delete file into `table`'s data directory, in
/// partition `part_value`, and return the [`DataFile`] that describes it.
///
/// It cannot be produced through [`write_position_delete_file`]: `ParquetWriter::close` DELETES a
/// zero-row output and returns `Ok(vec![])`, which is precisely why this action's own writer can
/// never put such a file in a manifest. So the parquet bytes are written directly with the parquet
/// crate's `ArrowWriter` over the position-delete ARROW schema — field ids included, so the
/// reserved-column lookup still resolves if the reader hands back an empty batch — and the
/// manifest-side `DataFile` is taken from a real one-pair write (for the content type, format,
/// partition tuple and spec id) and re-pointed at the empty file with its row count, size and
/// per-column metrics corrected.
async fn write_zero_row_pos_delete(table: &Table, part_value: i64, name: &str) -> DataFile {
    use parquet::arrow::ArrowWriter;

    let config = PositionDeleteWriterConfig::new().expect("position-delete writer config");
    let empty = RecordBatch::new_empty(config.arrow_schema().clone());

    let mut buffer: Vec<u8> = Vec::new();
    let mut writer = ArrowWriter::try_new(
        &mut buffer,
        config.arrow_schema().clone(),
        Some(position_delete_writer_properties()),
    )
    .expect("open an arrow parquet writer over the position-delete schema");
    writer.write(&empty).expect("write the empty batch");
    writer.close().expect("close the zero-row parquet file");

    let path = format!("{}/data/{name}", table.metadata().location());
    table
        .file_io()
        .new_output(&path)
        .expect("new output for the zero-row file")
        .write(bytes::Bytes::from(buffer.clone()))
        .await
        .expect("write the zero-row parquet bytes");

    let mut file =
        write_position_delete_file(table, Some(part_value), &[("scaffold.parquet", 0)]).await;
    file.file_path = path;
    file.record_count = 0;
    file.file_size_in_bytes =
        u64::try_from(buffer.len()).expect("the zero-row file's length fits a u64");
    file.column_sizes = HashMap::new();
    file.value_counts = HashMap::new();
    file.null_value_counts = HashMap::new();
    file.nan_value_counts = HashMap::new();
    file.lower_bounds = HashMap::new();
    file.upper_bounds = HashMap::new();
    file
}

/// C-040. An admitted BIN whose files yield ZERO `(file_path, pos)` pairs is SKIPPED **per bin**: it
/// contributes zero to all four counts, commits nothing, and does not abort the loop over the
/// remaining bins.
///
/// The state is REACHED, not simulated: partition 0 holds EXACTLY five zero-row parquet
/// position-deletes and nothing else — so they cannot co-bin with non-empty files — and the bin is
/// admitted by `enough_input_files` at the DEFAULT floor of five. Partition 1 carries a normal
/// admissible group of five. The admission precondition (all five sub-min, count >= the resolved
/// floor) is asserted BEFORE `execute`, so a fixture that stopped reaching the branch would red
/// rather than pass vacuously.
///
/// The branch is DEFENSIVE and NOT a parity claim: Java cannot reach this state at all
/// (`RewritePositionDeletesGroup`'s ctor `Preconditions`-checks `!tasks.isEmpty()`), and the fork's
/// own writer cannot emit a zero-row position-delete, so only an externally-written file gets here.
///
/// WHY THE WHITE-BOX LEG EXISTS — do not "simplify" it away. MEASURED BY THE G4 REVIEW (not by this
/// author): against the hazard mutant — branch deleted from `compact_group` AND an early
/// `return Ok(result)` hoisted into `execute` — 10 runs went 6 RED on the end-to-end leg (zero bin
/// scheduled first) and 4 RED on the white-box leg (normal bin first): 10/10 lethal, the two legs
/// EXACT COMPLEMENTS. So the end-to-end leg ALONE is ~50% flaky against that mutant, and the
/// white-box leg is not a weaker substitute for it — it is the half that removes the coin flip. No
/// end-to-end-only construction can close it: for ANY mix of k zero bins and m normal bins,
/// `HashMap` iteration order may place every zero bin LAST.
///
/// MUTATION, APPLIED: delete the `if pairs.is_empty()` branch. RED — but note the CAUSE: the mutant
/// does NOT raise the error itself. `ParquetWriter::close` returns `Ok(vec![])` for zero rows and
/// deletes the empty output, so "no file" is a normal return; it is `require_non_empty` that turns
/// the empty `Vec` into `Err(Unexpected)`, and the mutant reds THROUGH that guard. Without the guard
/// the mutant would return `Ok`, add nothing, and drop five live delete files.
#[tokio::test]
async fn test_admitted_bin_with_zero_pairs_is_skipped() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let p0 = write_data_file(&table, "p0.parquet", 0, &[(0, 10, 1), (0, 20, 2)]).await;
    let p1 = write_data_file(&table, "p1.parquet", 1, &[(1, 30, 3), (1, 40, 4)]).await;
    let p1_path = p1.file_path().to_string();
    let table = append_files(&catalog, &table, vec![p0, p1]).await;

    // Partition 0: FIVE zero-row position-deletes and NOTHING else.
    let mut empties = Vec::new();
    for index in 0..5 {
        empties.push(write_zero_row_pos_delete(&table, 0, &format!("empty-{index}.parquet")).await);
    }
    let empty_paths: Vec<String> = empties.iter().map(|f| f.file_path().to_string()).collect();
    let empty_sizes: Vec<u64> = empties.iter().map(|f| f.file_size_in_bytes).collect();

    // Partition 1: a normal admissible group of FIVE, masking p1's row at position 1 (y = 40).
    let mut normals = Vec::new();
    for _ in 0..5 {
        normals.push(write_position_delete_file(&table, Some(1), &[(&p1_path, 1)]).await);
    }
    let normal_paths: Vec<String> = normals.iter().map(|f| f.file_path().to_string()).collect();
    let normal_sizes: Vec<u64> = normals.iter().map(|f| f.file_size_in_bytes).collect();

    let mut all = empties;
    all.extend(normals);
    let table = add_deletes(&catalog, &table, all).await;

    // ADMISSION PRECONDITIONS at the DEFAULT config — both bins must actually be admitted, or the
    // zero-pairs branch is never reached and this pin proves nothing.
    let action = || RewritePositionDeleteFiles::new(table.clone());
    let config = action().resolve_config().expect("the defaults are legal");
    for size in empty_sizes.iter().chain(normal_sizes.iter()) {
        assert!(
            *size < config.min_file_size_bytes,
            "fixture: every input must be a SUB-MIN candidate (measured {size}, resolved min {})",
            config.min_file_size_bytes
        );
    }
    assert_eq!(
        config.min_input_files, 5,
        "fixture: the zero-pairs bin must be admitted at Java's DEFAULT floor of FIVE, not a \
         lowered one — the literal, so a moved constant reds here rather than silently re-shaping \
         the fixture"
    );
    assert!(
        empty_paths.len() >= config.min_input_files,
        "fixture: the zero-pairs partition must clear `enough_input_files` on its own \
         ({} files, floor {})",
        empty_paths.len(),
        config.min_input_files
    );
    assert!(
        normal_paths.len() >= config.min_input_files,
        "fixture: the second partition must be admissible too"
    );

    // Acceptance item 4's read-identity leg, and SUBSTANTIVE here unlike recipe 7's: partition 1's
    // deletes genuinely mask a live row (p1's position 1, y = 40), so the pre/post comparison can
    // actually detect a compaction that changed the masked set. It also proves the production scan
    // path READS the five surviving zero-row position-deletes without error.
    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 20, 30]),
        "fixture: y = 40 is really masked by partition 1's deletes, so read identity is not \
         vacuous here"
    );

    let history_before = table.metadata().history().len();
    let result = action()
        .execute(&catalog)
        .await
        .expect("the zero-pairs bin is SKIPPED, not an error — and the other bin still commits");

    // All four counts reflect ONLY the second partition.
    assert_eq!(
        result.rewritten_delete_files_count, 5,
        "only the SECOND partition's five files are rewritten"
    );
    assert_eq!(result.added_delete_files_count, 1, "one compacted output");
    assert_eq!(
        result.rewritten_bytes_count,
        normal_sizes.iter().sum::<u64>(),
        "the rewritten BYTES are the second partition's alone — the skipped bin adds none"
    );

    let reloaded = catalog
        .load_table(table.identifier())
        .await
        .expect("reload");
    let new_snapshots = snapshots_after(&reloaded, history_before);
    assert_eq!(
        new_snapshots.len(),
        1,
        "exactly ONE commit: the skipped bin commits nothing, and the other bin is unaffected"
    );
    assert_eq!(new_snapshots[0].summary().operation, Operation::Replace);

    let live = live_pos_delete_paths(&reloaded).await;
    for path in &empty_paths {
        assert!(
            live.contains(path),
            "every zero-row file is STILL LIVE — the skipped bin was left untouched, not dropped"
        );
    }
    for path in &normal_paths {
        assert!(
            !live.contains(path),
            "the second partition's inputs were replaced"
        );
    }
    assert_eq!(
        live.len(),
        empty_paths.len() + 1,
        "five untouched zero-row files plus the second partition's one output (live: {live:?})"
    );
    let added_bytes: u64 = live_pos_delete_files(&reloaded)
        .await
        .into_iter()
        .filter(|f| !empty_paths.contains(&f.file_path().to_string()))
        .map(|f| f.file_size_in_bytes)
        .sum();
    assert_eq!(
        result.added_bytes_count, added_bytes,
        "the added BYTES are the one real output's, and nothing from the skipped bin"
    );
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity — and the scan still reads the five untouched zero-row position-deletes"
    );

    // WHITE BOX, the leg the end-to-end shape cannot force: `plan_bins` iterates a `HashMap`, so
    // which of the two bins `execute` reaches FIRST is not fixed. Driving `compact_group` directly
    // on the zero-pairs bin pins what the loop actually depends on — the skip RETURNS the table
    // unchanged (so the caller's loop CONTINUES) and leaves the result counters alone, rather than
    // signalling an abort.
    let entries: Vec<LiveDeleteEntry> = live_pos_delete_files(&reloaded)
        .await
        .into_iter()
        .filter(|f| empty_paths.contains(&f.file_path().to_string()))
        .map(|data_file| LiveDeleteEntry {
            data_file,
            sequence_number: 1,
        })
        .collect();
    assert_eq!(
        entries.len(),
        5,
        "the five zero-row entries are still there"
    );
    let bin: AdmittedBin = ((0, Struct::from_iter([Some(Literal::long(0))])), entries);
    let starting = reloaded
        .metadata()
        .current_snapshot()
        .expect("a current snapshot")
        .snapshot_id();
    let mut counters = RewritePositionDeleteFilesResult::default();
    let returned = action()
        .compact_group(&catalog, &reloaded, &bin, &config, starting, &mut counters)
        .await
        .expect("the zero-pairs bin returns Ok so the bin loop CONTINUES");
    assert_eq!(
        returned.metadata().current_snapshot_id(),
        reloaded.metadata().current_snapshot_id(),
        "the skip returns the table UNCHANGED — nothing was committed for this bin"
    );
    assert_eq!(
        counters,
        RewritePositionDeleteFilesResult::default(),
        "and it contributed zero to ALL FOUR counts"
    );
}

// =================================================================================================
// THE V3 ARM — legacy PARQUET position deletes become Puffin DELETION VECTORS. Every pin here is a
// read-identity proof plus a SHAPE assertion, because a V3 arm that declined to act would satisfy
// read identity on its own. The evidence class is read identity, not Java parity: `iceberg-core`
// 1.10.0 has no runner for this action.
// =================================================================================================

/// Upgrade `table` to format version 3 — how a table acquires legacy parquet position deletes it
/// can no longer write.
async fn upgrade_to_v3(catalog: &impl Catalog, table: &Table) -> Table {
    let tx = Transaction::new(table);
    let action = tx
        .upgrade_table_version()
        .set_format_version(FormatVersion::V3);
    let tx = action.apply(tx).unwrap();
    tx.commit(catalog).await.unwrap()
}

/// Write ONE Puffin file holding a deletion vector per `(target_path, positions)` entry, in
/// partition x=`part_value`. The multi-blob shape a real engine writes, and the fixture the Puffin
/// closure is pinned on.
async fn write_deletion_vectors_in_one_puffin(
    table: &Table,
    part_value: i64,
    targets: &[(&str, &[u64])],
) -> Vec<DataFile> {
    use crate::writer::base_writer::deletion_vector_writer::DVFileWriter;

    let dv_path = format!(
        "{}/data/dv-{}.puffin",
        table.metadata().location(),
        uuid::Uuid::now_v7()
    );
    let output = table.file_io().new_output(&dv_path).unwrap();
    let partition_key = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        Struct::from_iter([Some(Literal::long(part_value))]),
    )
    .expect("PartitionKey::new: valid partition tuple");
    let mut writer = DVFileWriter::new(output);
    for (target_path, positions) in targets {
        for &pos in *positions {
            writer
                .delete(target_path, pos, Some(&partition_key))
                .expect("record DV position");
        }
    }
    writer.close().await.expect("close DV writer")
}

/// Swap delete files through `RewriteFiles`, each added file stamped with `sequence_number`.
///
/// `RowDelta`'s `validate_fresh_dvs_only` refuses to add a DV for a data file a live position delete
/// still covers, so this is the only in-tree route to the half-migrated shape those pins need.
async fn swap_delete_files(
    catalog: &impl Catalog,
    table: &Table,
    removed: Vec<DataFile>,
    added: Vec<DataFile>,
    sequence_number: i64,
) -> Table {
    let tx = Transaction::new(table);
    let mut action = tx
        .rewrite_files(Vec::new(), Vec::new())
        .delete_delete_files(removed);
    for file in added {
        action = action.add_delete_file_with_sequence_number(file, sequence_number);
    }
    let tx = action.apply(tx).unwrap();
    tx.commit(catalog).await.unwrap()
}

/// The data sequence number of the live delete entry at `path`.
async fn live_delete_seq(table: &Table, path: &str) -> i64 {
    live_delete_entries_with_seq(table)
        .await
        .into_iter()
        .find(|(file, _)| file.file_path() == path)
        .and_then(|(_, seq)| seq)
        .expect("a live delete entry at that path, carrying a sequence number")
}

/// CROWN JEWEL of the V3 arm. A table upgraded from V2 holds TWO parquet position deletes for one
/// data file. On V3 those cannot be compacted into a third parquet file, so the arm converts them
/// into ONE deletion vector. Read identity plus SHAPE: the live rows are unchanged, both parquet
/// deletes are gone, and exactly one Puffin DV referencing that data file is live.
///
/// MUTATION COVERAGE — APPLIED: delete the version dispatch in `execute` and the run dies on the
/// commit's "Must use DVs for position deletes in V3". RED.
#[tokio::test]
async fn test_v3_converts_parquet_position_deletes_into_one_deletion_vector() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 7, &[
        (7, 10, 100),
        (7, 20, 200),
        (7, 30, 300),
        (7, 40, 400),
        (7, 50, 500),
    ])
    .await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await;

    let pd1 = write_position_delete_file(&table, Some(7), &[(&x_path, 1)]).await;
    let table = add_deletes(&catalog, &table, vec![pd1]).await;
    let pd2 = write_position_delete_file(&table, Some(7), &[(&x_path, 3)]).await;
    let table = add_deletes(&catalog, &table, vec![pd2]).await;
    let table = upgrade_to_v3(&catalog, &table).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 30, 50]),
        "before: the two parquet position deletes mask y=20 and y=40"
    );
    assert_eq!(
        count_pos(&live_delete_files(&table).await),
        2,
        "fixture: two live parquet position deletes on a V3 table"
    );

    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(result.rewritten_delete_files_count, 2);
    assert_eq!(result.added_delete_files_count, 1);
    assert!(result.rewritten_bytes_count > 0);
    assert!(result.added_bytes_count > 0);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity: the deletion vector masks exactly the rows the parquet deletes masked"
    );

    let after = live_delete_files(&reloaded).await;
    assert_eq!(after.len(), 1, "exactly one delete file survives");
    assert_eq!(
        after[0].file_format(),
        DataFileFormat::Puffin,
        "the survivor is a Puffin deletion vector, not a parquet position delete"
    );
    assert_eq!(
        after[0].referenced_data_file().as_deref(),
        Some(x_path.as_str()),
        "the deletion vector references the data file the position deletes named"
    );
    assert_eq!(after[0].record_count, 2, "both positions landed in the DV");
}

/// THE `(None, None)` HAZARD (row R114 bound (c)). Every `DVFileWriter::delete` call the arm makes
/// carries the referenced data file's OWN `PartitionKey`, so `resolve_partition_spec_id` always
/// takes its key arm and never the keyless one that stamps spec 0 with an EMPTY partition tuple.
///
/// MUTATION COVERAGE — APPLIED: pass `None` instead of `Some(&partition_key)` in
/// `write_deletion_vectors` and the DV lands with an empty partition. RED on the partition
/// assertion. A partition-keyed maintenance action then reasons over a field that lies.
#[tokio::test]
async fn test_v3_deletion_vector_carries_its_data_file_partition_and_spec() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 7, &[(7, 10, 1), (7, 20, 2)]).await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await;
    let pd = write_position_delete_file(&table, Some(7), &[(&x_path, 1)]).await;
    let table = add_deletes(&catalog, &table, vec![pd]).await;
    let table = upgrade_to_v3(&catalog, &table).await;

    RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    let after = live_delete_files(&reloaded).await;
    assert_eq!(after.len(), 1, "one deletion vector");
    assert_eq!(
        after[0].partition(),
        &Struct::from_iter([Some(Literal::long(7))]),
        "the DV carries the data file's OWN partition tuple, not an empty one"
    );
    assert_eq!(
        after[0].partition_spec_id, 0,
        "and the spec the data file was written under"
    );
}

/// THE MERGE. One data file carries BOTH a legacy parquet position delete and a deletion vector, and
/// the converted DV must union both sets. The DV here ABSORBS the parquet position, the shape Java's
/// `BaseDVFileWriter.loadPreviousDeletes` produces, so the union preserves read identity.
///
/// MUTATION COVERAGE — APPLIED: drop the previous-DV load in `plan_deletion_vectors` and the DV's
/// own masked row comes back. RED on read identity, with y=40 resurrected.
#[tokio::test]
async fn test_v3_conversion_merges_the_data_file_existing_deletion_vector() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 7, &[
        (7, 10, 100),
        (7, 20, 200),
        (7, 30, 300),
        (7, 40, 400),
        (7, 50, 500),
    ])
    .await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await;

    let keep = write_position_delete_file(&table, Some(7), &[(&x_path, 1)]).await;
    let table = add_deletes(&catalog, &table, vec![keep]).await;
    let replaced = write_position_delete_file(&table, Some(7), &[(&x_path, 3)]).await;
    let replaced_path = replaced.file_path().to_string();
    let table = add_deletes(&catalog, &table, vec![replaced]).await;
    let table = upgrade_to_v3(&catalog, &table).await;

    // Turn the SECOND parquet delete into a DV, leaving the first behind — the shape Java produces
    // when a DV lands beside a partition-scoped position delete it may not discard.
    let replaced_seq = live_delete_seq(&table, &replaced_path).await;
    let existing_dv = write_deletion_vectors_in_one_puffin(&table, 7, &[(&x_path, &[1, 3])]).await;
    let removed: Vec<DataFile> = live_delete_files(&table)
        .await
        .into_iter()
        .filter(|f| f.file_path() == replaced_path)
        .collect();
    let table = swap_delete_files(&catalog, &table, removed, existing_dv, replaced_seq).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 30, 50]),
        "fixture: the DV shadows the parquet delete and masks y=20 and y=40"
    );
    let live = live_delete_files(&table).await;
    assert_eq!(live.len(), 2, "fixture: one parquet delete AND one DV");
    assert_eq!(
        live.iter()
            .filter(|f| f.file_format() == DataFileFormat::Puffin)
            .count(),
        1,
        "fixture: exactly one of the two is a deletion vector"
    );

    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(
        result.rewritten_delete_files_count, 2,
        "the parquet delete AND the superseded DV are both consumed"
    );
    assert_eq!(
        result.added_delete_files_count, 1,
        "one merged DV replaces them"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity: the merged DV masks BOTH y=20 and y=40"
    );
    let after = live_delete_files(&reloaded).await;
    assert_eq!(after.len(), 1, "exactly one delete file survives");
    assert_eq!(
        after[0].record_count, 2,
        "the merged DV holds both positions"
    );
}

/// THE PUFFIN CLOSURE — the delete-correctness pin of this arm. Delete-file removal is PATH-keyed,
/// so superseding one DV blob removes every sibling blob in the same Puffin. Each sibling must be
/// rewritten or its deleted rows come back.
///
/// MUTATION COVERAGE — APPLIED: delete the sibling loop in `plan_deletion_vectors` and data file
/// Y's deletion vector is removed without a replacement. RED on read identity, with y=41
/// resurrected — a silent, irreversible resurrection in committed metadata.
#[tokio::test]
async fn test_v3_rewrite_keeps_sibling_deletion_vectors_of_the_same_puffin() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 7, &[(7, 10, 1), (7, 11, 2)]).await;
    let y = write_data_file(&table, "y.parquet", 7, &[(7, 40, 1), (7, 41, 2)]).await;
    let x_path = x.file_path().to_string();
    let y_path = y.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x, y]).await;

    // A throwaway parquet delete, present only so the `RewriteFiles` below has something to remove.
    let scratch = write_position_delete_file(&table, Some(7), &[(&y_path, 0)]).await;
    let scratch_path = scratch.file_path().to_string();
    let table = add_deletes(&catalog, &table, vec![scratch]).await;
    // The legacy delete this arm will consume: it masks X's position 1 only.
    let legacy = write_position_delete_file(&table, Some(7), &[(&x_path, 1)]).await;
    let table = add_deletes(&catalog, &table, vec![legacy]).await;
    let table = upgrade_to_v3(&catalog, &table).await;

    // ONE Puffin: X's vector absorbed the legacy delete's position 1, Y's masks position 1.
    let scratch_seq = live_delete_seq(&table, &scratch_path).await;
    let puffin =
        write_deletion_vectors_in_one_puffin(&table, 7, &[(&x_path, &[0, 1]), (&y_path, &[1])])
            .await;
    let puffin_path = puffin[0].file_path().to_string();
    assert!(
        puffin.iter().all(|f| f.file_path() == puffin_path),
        "fixture: both deletion vectors live in ONE Puffin file"
    );
    let removed: Vec<DataFile> = live_delete_files(&table)
        .await
        .into_iter()
        .filter(|f| f.file_path() == scratch_path)
        .collect();
    let table = swap_delete_files(&catalog, &table, removed, puffin, scratch_seq).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([40]),
        "fixture: X's rows 10 and 11 and Y's row 41 are all masked"
    );

    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity: the sibling vector's masked row did NOT come back"
    );
    assert_eq!(
        result.rewritten_delete_files_count, 3,
        "the legacy parquet delete plus BOTH deletion vectors of the superseded Puffin"
    );
    assert_eq!(
        result.added_delete_files_count, 2,
        "X's merged vector AND Y's rewritten sibling"
    );
    let after = live_delete_files(&reloaded).await;
    assert_eq!(after.len(), 2, "one deletion vector per data file");
    assert!(
        after.iter().all(|f| f.file_path() != puffin_path),
        "the superseded Puffin is gone; both vectors were rewritten into a new one"
    );
    // Two DV entries, ONE physical Puffin: the added bytes are that file's size, not twice it.
    assert_eq!(
        result.added_bytes_count, after[0].file_size_in_bytes,
        "added bytes are summed over DISTINCT file paths"
    );
}

/// The user filter restricts which partitions the V3 arm converts, exactly as it restricts which
/// the bin-pack arm compacts.
#[tokio::test]
async fn test_v3_filter_restricts_the_converted_partitions() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let a = write_data_file(&table, "a.parquet", 0, &[(0, 10, 1), (0, 11, 2)]).await;
    let b = write_data_file(&table, "b.parquet", 1, &[(1, 20, 1), (1, 21, 2)]).await;
    let a_path = a.file_path().to_string();
    let b_path = b.file_path().to_string();
    let table = append_files(&catalog, &table, vec![a, b]).await;

    let pda = write_position_delete_file(&table, Some(0), &[(&a_path, 1)]).await;
    let pdb = write_position_delete_file(&table, Some(1), &[(&b_path, 1)]).await;
    let table = add_deletes(&catalog, &table, vec![pda, pdb]).await;
    let table = upgrade_to_v3(&catalog, &table).await;

    let before = scan_y_values(&table).await;
    let result = RewritePositionDeleteFiles::new(table.clone())
        .filter(Reference::new("x").equal_to(Datum::long(0)))
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(result.rewritten_delete_files_count, 1, "partition 0 only");
    assert_eq!(result.added_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
    let after = live_delete_files(&reloaded).await;
    assert_eq!(after.len(), 2, "partition 1's parquet delete is untouched");
    assert_eq!(
        after
            .iter()
            .filter(|f| f.file_format() == DataFileFormat::Parquet)
            .count(),
        1,
        "exactly one parquet position delete remains — partition 1's"
    );
}

/// A position naming a data file the snapshot no longer holds is DROPPED, not refused — it can
/// delete nothing, which is what the V1/V2 arm effectively does too. REFUSING dead-ends the table:
/// `RemoveDanglingDeleteFiles` keys on `(spec_id, partition)` and the partition's live data
/// sequence, so it cannot clear a delete file that still names ONE live data file.
///
/// MUTATION COVERAGE — APPLIED: restore the refusal (error on the liveness miss) and this test reds.
#[tokio::test]
async fn test_v3_position_naming_a_non_live_data_file_is_dropped() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 7, &[(7, 10, 1), (7, 11, 2)]).await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await;
    // ONE delete file naming a LIVE data file and a GHOST one — the post-compaction shape
    // `RemoveDanglingDeleteFiles` cannot classify as dangling.
    let ghost = format!("{}/data/ghost.parquet", table.metadata().location());
    let pd = write_position_delete_file(&table, Some(7), &[
        (x_path.as_str(), 1),
        (ghost.as_str(), 0),
    ])
    .await;
    let table = add_deletes(&catalog, &table, vec![pd]).await;
    let table = upgrade_to_v3(&catalog, &table).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10]),
        "fixture: the live position masks y=11"
    );

    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .expect("a stale reference must not dead-end the arm");
    assert_eq!(result.rewritten_delete_files_count, 1);
    assert_eq!(
        result.added_delete_files_count, 1,
        "one DV for the LIVE data file; the ghost position is dropped"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity: dropping a position that could delete nothing changes nothing"
    );
    let after = live_delete_files(&reloaded).await;
    assert_eq!(after.len(), 1, "one deletion vector");
    assert_eq!(
        after[0].referenced_data_file().as_deref(),
        Some(x_path.as_str()),
        "and it references the LIVE data file, not the ghost"
    );
}

/// THE SHADOW CLOSURE — the second half of the delete-correctness line. A file-scoped position
/// delete is routed BY PATH with no partition condition, so one stamped `x=1` still applies to a
/// data file in `x=0`. Convert only the `x=0` delete and the new DV SHADOWS the other, which goes
/// inert.
///
/// MUTATION COVERAGE — APPLIED: drop the `refuse_shadowed_deletes` call and the run returns
/// `Ok {rewritten: 1, added: 1}`, live rows `{12}` becoming `{11, 12}` — y=11 resurrected.
#[tokio::test]
async fn test_v3_refuses_when_a_filtered_out_delete_would_be_shadowed() {
    let (catalog, temp) = local_fs_catalog().await;
    let table = create_short_path_partitioned_table(&catalog, temp.path(), FormatVersion::V2).await;

    let a = write_data_file(&table, "a.parquet", 0, &[
        (0, 10, 1),
        (0, 11, 2),
        (0, 12, 3),
    ])
    .await;
    let a_path = a.file_path().to_string();
    let table = append_files(&catalog, &table, vec![a]).await;

    // Both name ONLY a.parquet, so both are FILE-scoped (equal `file_path` bounds) and both apply by
    // path — but they are stamped in different partitions, and the filter judges them by that stamp.
    let in_scope = write_file_scoped_position_delete_file(&table, 0, &a_path, &[0]).await;
    let out_of_scope = write_file_scoped_position_delete_file(&table, 1, &a_path, &[1]).await;
    let table = add_deletes(&catalog, &table, vec![in_scope, out_of_scope]).await;
    let table = upgrade_to_v3(&catalog, &table).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([12]),
        "fixture NON-VACUITY: BOTH file-scoped deletes apply to a.parquet despite their stamps"
    );

    let error = RewritePositionDeleteFiles::new(table.clone())
        .filter(Reference::new("x").equal_to(Datum::long(0)))
        .execute(&catalog)
        .await
        .expect_err("a delete that would be shadowed must fail the run closed");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error.to_string().contains("SHADOWS"),
        "the refusal says what would happen: {error}"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "fail CLOSED: nothing was committed, so no row came back"
    );
}

/// THE PER-DV SEQ STAMP — the V3 twin of `test_each_bin_output_carries_its_own_bin_max_not_the_partition_max`.
/// Each deletion vector must carry ITS OWN plan maximum, not the run-wide one. A run-wide stamp
/// reads the same today but writes false metadata that `RemoveDanglingDeleteFiles` and conflict
/// detection then reason over.
///
/// MUTATION COVERAGE — APPLIED: stamp every DV with `plans.values().map(..).max()` and A's vector
/// carries B's 3 instead of its own 2. RED.
#[tokio::test]
async fn test_v3_each_deletion_vector_carries_its_own_source_max_seq() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let a = write_data_file(&table, "a.parquet", 0, &[(0, 10, 1), (0, 11, 2)]).await;
    let b = write_data_file(&table, "b.parquet", 0, &[(0, 20, 1), (0, 21, 2)]).await;
    let a_path = a.file_path().to_string();
    let b_path = b.file_path().to_string();
    let table = append_files(&catalog, &table, vec![a, b]).await; // data seq 1

    // SEPARATE commits, so the two legacy deletes carry DIFFERENT data sequence numbers.
    let pd_a = write_position_delete_file(&table, Some(0), &[(a_path.as_str(), 1)]).await;
    let pd_a_file = pd_a.file_path().to_string();
    let table = add_deletes(&catalog, &table, vec![pd_a]).await; // seq 2
    let pd_b = write_position_delete_file(&table, Some(0), &[(b_path.as_str(), 1)]).await;
    let pd_b_file = pd_b.file_path().to_string();
    let table = add_deletes(&catalog, &table, vec![pd_b]).await; // seq 3
    let table = upgrade_to_v3(&catalog, &table).await;

    let a_source_seq = live_delete_seq(&table, &pd_a_file).await;
    let b_source_seq = live_delete_seq(&table, &pd_b_file).await;
    assert_ne!(
        a_source_seq, b_source_seq,
        "fixture NON-VACUITY: the two plan maxima must DIFFER, or a run-wide stamp is unkillable"
    );

    let before = scan_y_values(&table).await;
    RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
    let stamped = live_delete_entries_with_seq(&reloaded).await;
    let seq_for = |data_file_path: &str| -> i64 {
        stamped
            .iter()
            .find(|(file, _)| file.referenced_data_file().as_deref() == Some(data_file_path))
            .and_then(|(_, seq)| *seq)
            .expect("a deletion vector referencing that data file")
    };
    assert_eq!(
        seq_for(&a_path),
        a_source_seq,
        "A's vector carries A's own source max, not the run-wide max"
    );
    assert_eq!(seq_for(&b_path), b_source_seq, "B's vector carries B's own");
}

/// A live ORC position delete on a V3 table is REFUSED, where the V1/V2 arm silently skips it. That
/// refusal is what makes `Ok(zeros)` total on this arm: an input it cannot express is an error.
#[tokio::test]
async fn test_v3_non_parquet_position_delete_is_refused_not_skipped() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 1), (0, 20, 2)]).await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await;
    let (table, _sizes) =
        add_fabricated_non_parquet_pos_deletes(&catalog, &table, &x_path, DataFileFormat::Orc)
            .await;
    let table = upgrade_to_v3(&catalog, &table).await;

    let error = RewritePositionDeleteFiles::new(table)
        .execute(&catalog)
        .await
        .expect_err("an unreadable position-delete format must fail closed on V3");
    assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
}

/// THE SHADOW CLOSURE, PARTITION LEG — the COMMON shape, not the rare one: a fork-written position
/// delete carries `truncate(16)` path bounds, so partition-scoped is its default. Data file A sits
/// in `x=0`; a FILE-scoped delete stamped `x=1` is admitted by `filter(x = 1)` and plans A, while a
/// PARTITION-scoped delete in `x=0` is excluded and still applies to A.
///
/// MUTATION COVERAGE — APPLIED: replace the partition arm with `None => None` and the run returns
/// `Ok {rewritten: 1, added: 1}`, live rows `{10}` becoming `{10, 12}` — y=12 resurrected.
#[tokio::test]
async fn test_v3_refuses_when_a_partition_scoped_delete_would_be_shadowed() {
    let (catalog, temp) = local_fs_catalog().await;
    let table = create_short_path_partitioned_table(&catalog, temp.path(), FormatVersion::V2).await;

    let a = write_data_file(&table, "a.parquet", 0, &[
        (0, 10, 1),
        (0, 11, 2),
        (0, 12, 3),
    ])
    .await;
    let a_path = a.file_path().to_string();
    let table = append_files(&catalog, &table, vec![a]).await;

    // Admitted by `filter(x = 1)`: file-scoped, so it applies to A by PATH despite its `x=1` stamp.
    let admitted = write_file_scoped_position_delete_file(&table, 1, &a_path, &[1]).await;
    // Excluded by that filter: partition-scoped in `x=0`, so it applies to A by (spec, partition).
    let excluded = write_position_delete_file(&table, Some(0), &[(a_path.as_str(), 2)]).await;
    assert!(
        referenced_data_file_location(&excluded).is_none(),
        "fixture NON-VACUITY: the excluded delete must be PARTITION-scoped, or this pins the path leg"
    );
    let table = add_deletes(&catalog, &table, vec![admitted, excluded]).await;
    let table = upgrade_to_v3(&catalog, &table).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10]),
        "fixture NON-VACUITY: BOTH deletes apply to a.parquet, by different routes"
    );

    let error = RewritePositionDeleteFiles::new(table.clone())
        .filter(Reference::new("x").equal_to(Datum::long(1)))
        .execute(&catalog)
        .await
        .expect_err("a partition-scoped delete that would be shadowed must fail the run closed");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(error.to_string().contains("SHADOWS"), "{error}");

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "fail CLOSED: nothing committed, so no row came back"
    );
}

/// THE ALREADY-SHADOWED DIRECTION. A deletion vector that does NOT cover a legacy delete it already
/// suppresses: merging makes those positions effective and DELETES rows the table returns today.
/// Java 1.10.0's own rewrite writes this shape — its `loadPreviousDeletes` is `path -> null`.
///
/// MUTATION COVERAGE — APPLIED: drop the superset check and the run returns
/// `Ok {rewritten: 2, added: 1}`, live rows `{11, 12}` becoming `{11}` — y=12 silently gone.
#[tokio::test]
async fn test_v3_refuses_when_the_existing_vector_does_not_cover_the_legacy_delete() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let a = write_data_file(&table, "a.parquet", 7, &[
        (7, 10, 1),
        (7, 11, 2),
        (7, 12, 3),
    ])
    .await;
    let a_path = a.file_path().to_string();
    let table = append_files(&catalog, &table, vec![a]).await;

    // The legacy delete the DV will shadow without covering: it masks position 2 (y=12).
    let shadowed = write_position_delete_file(&table, Some(7), &[(a_path.as_str(), 2)]).await;
    let table = add_deletes(&catalog, &table, vec![shadowed]).await;
    // A throwaway, present only so the `RewriteFiles` below has something to remove.
    let scratch = write_position_delete_file(&table, Some(7), &[(a_path.as_str(), 0)]).await;
    let scratch_path = scratch.file_path().to_string();
    let table = add_deletes(&catalog, &table, vec![scratch]).await;
    let table = upgrade_to_v3(&catalog, &table).await;

    let scratch_seq = live_delete_seq(&table, &scratch_path).await;
    let vector = write_deletion_vectors_in_one_puffin(&table, 7, &[(&a_path, &[0])]).await;
    let removed: Vec<DataFile> = live_delete_files(&table)
        .await
        .into_iter()
        .filter(|f| f.file_path() == scratch_path)
        .collect();
    let table = swap_delete_files(&catalog, &table, removed, vector, scratch_seq).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([11, 12]),
        "fixture NON-VACUITY: the DV masks position 0 and SHADOWS the delete masking position 2"
    );

    // NO FILTER — so the shadow closure is silent and only the superset check can catch this.
    let error = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .expect_err("a non-superset vector must fail the run closed");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error
            .to_string()
            .contains("DELETE rows the table returns today"),
        "the refusal names the loss direction: {error}"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "fail CLOSED: y=12 is still live"
    );
}

/// The Puffin closure's LIVENESS guard: a sibling blob whose data file is no longer live is dropped
/// with its Puffin, not planned. Without the guard it reaches `live_data_file` and errors.
///
/// MUTATION COVERAGE — APPLIED: drop `&& inventory.data_files.contains_key(data_file_path)` from the
/// sibling loop and the run returns `Unexpected`. RED on the success expectation.
#[tokio::test]
async fn test_v3_puffin_closure_skips_a_sibling_whose_data_file_is_gone() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 7, &[(7, 10, 1), (7, 11, 2)]).await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await;

    let legacy = write_position_delete_file(&table, Some(7), &[(x_path.as_str(), 1)]).await;
    let table = add_deletes(&catalog, &table, vec![legacy]).await;
    let scratch = write_position_delete_file(&table, Some(7), &[(x_path.as_str(), 0)]).await;
    let scratch_path = scratch.file_path().to_string();
    let table = add_deletes(&catalog, &table, vec![scratch]).await;
    let table = upgrade_to_v3(&catalog, &table).await;

    // ONE Puffin: X's vector (a SUPERSET of the legacy delete) beside a sibling for a data file the
    // snapshot no longer holds.
    let ghost = format!("{}/data/ghost.parquet", table.metadata().location());
    let scratch_seq = live_delete_seq(&table, &scratch_path).await;
    let puffin =
        write_deletion_vectors_in_one_puffin(&table, 7, &[(&x_path, &[0, 1]), (&ghost, &[0])])
            .await;
    let removed: Vec<DataFile> = live_delete_files(&table)
        .await
        .into_iter()
        .filter(|f| f.file_path() == scratch_path)
        .collect();
    let table = swap_delete_files(&catalog, &table, removed, puffin, scratch_seq).await;

    let before = scan_y_values(&table).await;
    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .expect("a sibling naming a dead data file must not fail the run");
    assert_eq!(
        result.added_delete_files_count, 1,
        "only X's vector is rewritten; the ghost sibling is dropped with its Puffin"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
    let after = live_delete_files(&reloaded).await;
    assert_eq!(after.len(), 1, "one deletion vector survives");
    assert_eq!(
        after[0].referenced_data_file().as_deref(),
        Some(x_path.as_str())
    );
}

/// A CAPABILITY LIMIT, stated rather than given a remedy that cannot work. A file-scoped ORC delete
/// the filter excluded still references a planned data file, so the closure refuses — but "widen the
/// filter" would only route it to `FeatureUnsupported`. No filter width converts such a table.
///
/// MUTATION COVERAGE — APPLIED: collapse the remedy to the unconditional "Widen the filter" string
/// and this test reds on the message.
#[tokio::test]
async fn test_v3_shadowed_unreadable_delete_says_no_filter_width_helps() {
    let (catalog, temp) = local_fs_catalog().await;
    let table = create_short_path_partitioned_table(&catalog, temp.path(), FormatVersion::V2).await;

    let a = write_data_file(&table, "a.parquet", 0, &[
        (0, 10, 1),
        (0, 11, 2),
        (0, 12, 3),
    ])
    .await;
    let a_path = a.file_path().to_string();
    let table = append_files(&catalog, &table, vec![a]).await;

    let admitted = write_position_delete_file(&table, Some(0), &[(a_path.as_str(), 2)]).await;
    let mut unreadable = write_file_scoped_position_delete_file(&table, 1, &a_path, &[1]).await;
    unreadable.file_format = DataFileFormat::Orc;
    assert!(
        referenced_data_file_location(&unreadable).is_some(),
        "fixture NON-VACUITY: the ORC delete must be FILE-scoped, or it never reaches the path leg"
    );
    let table = add_deletes(&catalog, &table, vec![admitted, unreadable]).await;
    let table = upgrade_to_v3(&catalog, &table).await;

    let error = RewritePositionDeleteFiles::new(table)
        .filter(Reference::new("x").equal_to(Datum::long(0)))
        .execute(&catalog)
        .await
        .expect_err("an unreadable shadowed delete must fail the run closed");
    assert!(
        error
            .to_string()
            .contains("NO filter setting converts this table"),
        "the refusal states the capability limit instead of an unreachable remedy: {error}"
    );
}
