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

//! Tests for RewritePositionDeleteFiles.
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
use crate::maintenance::RewriteDataFiles;
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

// Helpers (table build / data + position-delete writers / scan) — same shape as the convert tests.

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

/// A partitioned table at a DELIBERATELY SHORT location.
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

/// Write a DATA file in partition `part_value` holding `rows` (the file path is returned for use as the position-delete `file_path` target).
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

/// Write a real PARQUET position-delete file masking the given `(target_path, pos)` pairs in partition `part_value`.
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
        .with_partition_spec(table.metadata().default_partition_spec().as_ref().clone())
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

/// Write a FILE-SCOPED parquet position delete.
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

// CROWN JEWEL — read-identity over a data file masked by 2+ parquet position-delete files.

/// THE CROWN JEWEL (read-identity). One data file is masked by TWO parquet position-delete files (pos 1 = y=20, pos 3 = y=40). After.
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

// STALLER (seq stamping) — the compacted file must carry the group MAX rewritten data seq.

/// SEQ STAMPING. Data X is at seq 1; two position deletes mask it at seqs 2 and 3. The compacted file must carry the BIN MAX.
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

/// SEQ STAMPING, the resurrection guard. Data X at seq 1 is masked by two position deletes. A second data file W at seq 4 also.
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

// GROUPING + PARTITION ISOLATION.

/// MULTI-FILE GROUPING across data files in one partition.
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

/// PARTITION ISOLATION and multi-group table advance.
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
    assert_eq!(
        reloaded.metadata().history().len(),
        history_before + 2,
        "two group commits must each produce a Replace snapshot"
    );
}

/// Filter restriction. `filter(x == 0)` compacts only partition 0. Partition 1 stays untouched.
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

/// Honest zeros: five Puffin DVs on V3. File-scoped, so nothing to convert and no commit.
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

/// Write a single-data-file Puffin DELETION VECTOR masking the given absolute positions of `target_path`, in partition x=0.
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
    let mut writer = DVFileWriter::new(output).unpartitioned();
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

/// Commit FIVE live position-delete manifest entries in partition 0 whose `file_format` is `format`, and return their measured sizes.
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

/// The shared body of C-008's ORC and Avro elements: five live position-delete entries in one partition whose `file_format` is `format` are dropped at collection, so the action is a no-op.
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

/// C-008 ORC: five live ORC position deletes on V2 are skipped, not compacted.
#[tokio::test]
async fn test_non_parquet_position_deletes_skipped_at_collection_orc() {
    assert_non_parquet_pos_deletes_are_skipped(DataFileFormat::Orc).await;
}

/// C-008, `Avro` element — the same fixture and mutation as the ORC pin.
#[tokio::test]
async fn test_non_parquet_position_deletes_skipped_at_collection_avro() {
    assert_non_parquet_pos_deletes_are_skipped(DataFileFormat::Avro).await;
}

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

/// Unpartitioned table: two position deletes in the single unpartitioned group compact into one.
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

/// A minimal unpartitioned table carrying `properties` — the fixture for the white-box config pins.
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

/// `write.delete.target-file-size-bytes` as a `HashMap`, for the pins that exercise the parse function alone (no table needed).
fn delete_target_property(value: &str) -> std::collections::HashMap<String, String> {
    std::collections::HashMap::from([(
        TableProperties::PROPERTY_WRITE_DELETE_TARGET_FILE_SIZE_BYTES.to_string(),
        value.to_string(),
    )])
}

// C-035 — the parse: one function whose accept/reject domain is `Long.parseLong`'s, EXACTLY.

/// C-035 element 1 (ABSENT). No property ⇒ the 64 MiB delete default.
#[test]
fn test_parse_delete_target_absent_is_64_mib() {
    let properties = std::collections::HashMap::new();
    assert_eq!(
        parse_delete_target_file_size(&properties).expect("an absent property takes the default"),
        67108864
    );
}

/// C-035 element 2 (a well-formed decimal in `[2, i64::MAX - 1]`) — the only class that can also SURVIVE the preconditions.
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

/// C-035: `i64::MAX` parses, matching `Long.parseLong`. Downstream preconditions may still reject it.
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

/// C-035: unparsable and empty strings fail, matching `NumberFormatException`.
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

/// C-035 element 6 (magnitude outside the `long` range) — THE anti-`u64` pin.
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

/// C-035: `"0"` parses, then precondition (1) rejects it with Java's verbatim message.
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

/// C-035: `"-1"` parses as `i64`, matching `Long.parseLong`. Then precondition (1) rejects it. a `u64` parse would reject it.
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

// C-002 / C-032 — the target resolves from the DELETE property, never from the data-file one.

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

/// C-002: the data-file target property must not change this action's delete-file target. move the delete target. Reds if.
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

// C-007 — the ratio defaults through the shared `d2l` helper.

/// C-007: at the 64 MiB default, `d2l(target * 0.75)` is exactly 50331648. so no rounding is involved)..
#[tokio::test]
async fn test_config_min_file_size_default_is_three_quarters_target() {
    let (_temp, table) = config_table(&[]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect("the defaults satisfy every precondition");
    assert_eq!(config.min_file_size_bytes, 50331648);
}

/// C-007: at the 64 MiB default, `d2l(target * 1.8)` truncates as Java does.
#[tokio::test]
async fn test_config_max_file_size_default_is_one_point_eight_target() {
    let (_temp, table) = config_table(&[]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect("the defaults satisfy every precondition");
    assert_eq!(config.max_file_size_bytes, 120795955);
}

/// C-007: clamp `d2l` at `i64::MAX`, matching Java. Unclamped Rust saturates at `u64::MAX`. `u64::MAX`, so `.min(i64::MAX as u64)`.
#[tokio::test]
async fn test_config_max_file_size_clamps_to_java_long_max() {
    const TARGET: u64 = 6_000_000_000_000_000_000;
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

// C-001 — the remaining two ported options and their defaults.

/// C-001: `min_input_files` defaults to Java's 5.
#[tokio::test]
async fn test_config_min_input_files_default_is_five() {
    let (_temp, table) = config_table(&[]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect("the defaults satisfy every precondition");
    assert_eq!(config.min_input_files, 5);
}

/// C-001: `max_file_group_size_bytes` defaults to Java's 100 GiB. `MAX_FILE_GROUP_SIZE_BYTES_DEFAULT = 107374182400` (100 GiB).
#[tokio::test]
async fn test_config_max_file_group_size_default_is_100_gib() {
    let (_temp, table) = config_table(&[]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect("the defaults satisfy every precondition");
    assert_eq!(config.max_file_group_size_bytes, 107374182400);
    assert_eq!(config.max_file_group_size_bytes, 100 * 1024 * 1024 * 1024);
}

/// C-001: all five ported builders land on the resolved config. builder wired to the wrong field.
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

// C-006 — Java's `sizeThresholds` preconditions, each with Java's verbatim message.

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

/// C-006 precondition (1) at a NEGATIVE target — the ORDER-discriminating pin.
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

/// C-006 precondition (7) — ONE LEG PER KNOB.
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

    RewritePositionDeleteFiles::new(table)
        .max_file_size_bytes(i64::MAX as u64)
        .resolve_config()
        .expect("i64::MAX is inside Java's long domain");
}

/// C-035: `i64::MAX` parses, then precondition (4) rejects the defaulted max. CLAMPS to `i64::MAX`, so `target.
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

/// Write a position-delete file masking `count` consecutive positions of `target_path` from `first_pos`, in partition 0.
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

/// A partitioned table with ONE five-row data file (y = 10,20,30,40,50 at positions 0.
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

/// The live position-delete files in MANIFEST ORDER — the same order `collect_position_delete_groups` walks, so a fixture whose packing depends on order can assert the order it actually gets instead of assuming it.
async fn live_pos_delete_paths(table: &Table) -> Vec<String> {
    live_delete_files(table)
        .await
        .iter()
        .filter(|f| f.content_type() == DataContentType::PositionDeletes)
        .map(|f| f.file_path().to_string())
        .collect()
}

// C-003 element 1 — `enough_input_files`'s `size > 1` conjunct.

/// C-003: a lone sub-min file with `min_input_files(1)` is a candidate but still declined.
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

// C-003 element 2 (+ C-001, and the execute-altitude proof that the resolved config is CONSULTED).

/// C-003 element 2, TWO-SIDED at the DEFAULT config.
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

// C-003 element 4 — `enough_content` and its STRICTNESS at the target.

/// C-003: two sub-min files whose sizes sum just over target are admitted. target. They are below the count.
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

/// C-003 element 4, THE BOUNDARY. The same two sub-min files with `target := S_A + S_B` exactly.
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

// C-003 element 5 — `too_much_content` EXISTS and carries NO `size > 1` guard.

/// C-003: a lone file above `max_file_size_bytes` is admitted (`too_much_content` has no size>1 guard). `too_much_content` is `input_size >.
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

/// C-021: a lone sub-min file is a candidate and still declined by the group filter.
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

    // PRECONDITION FIRST — the measured size against the RESOLVED thresholds. An identity in `S`
    assert!(
        size < config.min_file_size_bytes,
        "fixture: the lone file must be SUB-MIN (measured {size}, resolved min {})",
        config.min_file_size_bytes
    );
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

/// C-021: a lone in-range file is not a candidate, so it never forms a bin. element 1.
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

    assert!(
        config.min_file_size_bytes <= size && size <= config.max_file_size_bytes,
        "fixture: the lone file must be IN RANGE (measured {size}, resolved band [{}, {}])",
        config.min_file_size_bytes,
        config.max_file_size_bytes
    );
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

// C-004 — the size-only candidate filter, and its two STRICT boundaries.

/// C-004 element 1. FIVE position-delete files in one partition, every one strictly IN RANGE, so `outsideDesiredFileSizeRange`.
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

/// C-004: `length < min` is strict. A file whose size equals min is not a candidate.
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

/// C-004: `length > max` is strict. A file whose size equals max is not a candidate.
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

// C-005 — the pipeline ORDER (S4 strictly before S5) and the filter STAGE.

/// C-005's ORDER pin, discriminating BY BIN COUNT.
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

/// C-005: the user filter binds once, after the no-snapshot return and before the walk. BEFORE the manifest walk, where.
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

// C-027 — the SHARED bin packer, reused through a weight closure.

/// C-027: `max_file_group_size_bytes` splits one partition into two bins. Each commits separately. commits its own Replace snapshot..
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

// C-041 — the group input-size sum SATURATES where Java's `long` sum wraps.

/// C-041. The bin input-size sum is `saturating_add`, so an overflowing bin resolves to `u64::MAX`
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

// C-003 elements 3 and 5b — the two gate leaves that are unreachable END TO END, pinned WHITE BOX.

/// A [`LiveDeleteEntry`] whose `file_size_in_bytes` is SET to `size` — the same white-box seam `test_group_input_size_saturates_not_wraps` uses.
async fn entry_of_size(table: &Table, target_path: &str, size: u64) -> LiveDeleteEntry {
    let mut data_file = write_sized_pos_delete(table, target_path, 1, 1).await;
    data_file.file_size_in_bytes = size;
    LiveDeleteEntry {
        data_file,
        sequence_number: 1,
    }
}

/// The thresholds both white-box gate pins run against: `min < target < max` (C-006 preconditions (3) and (4)), with `min_input_files` well above the bin size so `enough_input_files` cannot be the admitter under any of the mutants below.
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

/// C-003 element 5b — `too_much_content`'s BOUNDARY STRICTNESS.
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

/// C-003 element 3 — `enough_content`'s `size > 1` conjunct.
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

/// Every live PARQUET position-delete file in the current snapshot.
async fn live_pos_delete_files(table: &Table) -> Vec<DataFile> {
    live_delete_files(table)
        .await
        .into_iter()
        .filter(|f| f.content_type() == DataContentType::PositionDeletes)
        .collect()
}

/// Read one position-delete file's `(file_path, pos)` pairs back off disk, by RESERVED FIELD ID — the same way the action reads them.
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

/// Live position-delete files with their pairs, sorted by each file's first pair.
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
async fn recipe_3_lone_oversized_fixture() -> (impl Catalog, TempDir, Table, u64, u64) {
    let (catalog, temp, table, _x_path) = gate_table().await;
    let pd = write_wide_path_pos_delete(&table, 34_000).await;
    let s = pd.file_size_in_bytes;
    let t = s * 10 / 24;
    let table = add_deletes(&catalog, &table, vec![pd]).await;
    (catalog, temp, table, s, t)
}

/// Recipe 3's shared PRE-ASSERTIONS, run before `execute` in both of its tests: the fixture really is in the size class the recipe claims, and the chunk budget really is the ruled constant.
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

// C-009 — the roll bound is Java's `writeMaxFileSize()`, NOT the resolved target.

/// C-009: `write_max_file_size` at delete defaults, white-box through `resolve_config`. `write_max_file_size` is exactly Java's.
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

/// C-009 pin 2. `write_max_file_size` gets its OWN clamp pair, never a borrowed one. Java's `d2l`
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

/// C-009 pin 3 — the CALL-SITE DISCRIMINATOR, and the only pin that tells the two candidate bounds apart.
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

/// Write a position-delete file masking `count` positions of `target_path` STRIDED by `stride` from `first_pos`.
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
    /// The two inputs' pairs READ BACK OFF DISK and concatenated in MANIFEST ORDER — the exact sequence `compact_group` builds before it sorts.
    input_concat: Vec<(String, i64)>,
    /// `input_concat` sorted — the multiset the split outputs must reproduce exactly.
    input_pairs: Vec<(String, i64)>,
    before: HashSet<i64>,
    config: ResolvedConfig,
    result: RewritePositionDeleteFilesResult,
}

/// The SMALL-EXPLICIT-BAND SPLIT fixture (recipe 10), built and run once: two ~118 KB position-delete files over disjoint position ranges of one data file, committed in two snapshots so their data sequence numbers are 2 and 3 (bin max = 3).
async fn split_fixture_run() -> (impl Catalog, TempDir, SplitFixture) {
    let (catalog, temp, table, x_path) = gate_table().await;

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

/// C-009: an explicit small band makes the bin exceed `write_max`, so the rolling writer splits. rolls, so ONE bin produces MORE THAN.
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

/// C-025 pin 1 (C-036 recipe 6 — no fixture).
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

/// C-025 pin 2 — the CHUNKING RULE itself, white-box on [`chunk_end`], because neither the pair cap nor the one-pair floor is observable end to end (raising `CHUNK_PAIRS` only changes how often `should_roll` runs, and a zero-pair chunk HANGS rather than failing an assertion).
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

/// C-025 pin 3 (RECIPE 3) — THE RUNTIME CLEARANCE PIN.
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

// C-026 — THE FIXED POINT, and the three counterexamples that BOUND it.

/// C-026 — the no-op in its ruled fixed-point form: the three-way conjunction `rewritten == 0 && added == 0 && current_snapshot_id() unchanged`.
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

/// C-026 COUNTEREXAMPLE 2, PINNED AS EXPECTED BEHAVIOUR — not a defect and not to be "fixed": Java behaves identically.
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

/// C-026 COUNTEREXAMPLE 3, DISTINCT from counterexample 2 and likewise PINNED AS EXPECTED BEHAVIOUR.
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

/// C-010 element 2 — the FAN-OUT dimension.
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

/// C-010's counting half: `added_delete_files_count` is the REAL number of files added across the split, and `added_bytes_count` is the CHECKED sum of their sizes — not the hard-coded `+= 1` and single `file_size_in_bytes` the one-output shape carried.
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

// C-044 — the GLOBAL SORT happens BEFORE any split, and the split preserves it.

/// C-044: pairs are sorted globally once. Chunks are contiguous slices of that order. sorted `Vec` in order — so output.
#[tokio::test]
async fn test_split_outputs_have_disjoint_ascending_ranges() {
    let (catalog, _temp, fixture) = split_fixture_run().await;
    assert!(
        fixture.result.added_delete_files_count >= 2,
        "fixture: the bin must SPLIT, or there are no ranges to be disjoint"
    );

    // NON-VACUITY, ON THE MEASURED INPUT SEQUENCE — not on the fixture's own construction.
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

// C-046 — the FAIL-CLOSED guard SURVIVES the `Vec<DataFile>` change at the new arity.

/// C-046: a non-empty bin must produce at least one file. The parquet writer treats empty as normal. return from the parquet writer —.
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

/// C-036 RECIPE 7 — TWO ADMITTED BINS in ONE partition.
struct Recipe7 {
    table: Table,
    /// `S_A .. S_D`, MEASURED, in MANIFEST order.
    sizes: [u64; 4],
    /// A, B, C, D's file paths, in MANIFEST order.
    paths: [String; 4],
    /// `W` — the `max_file_group_size_bytes` knob, derived from the measured sizes.
    group_size: u64,
}

/// `m` — recipe 7's `min_file_size_bytes` knob.
const RECIPE_7_MIN: u64 = 100_000;

/// Build C-036 recipe 7.
async fn recipe_7_two_bin_fixture() -> (impl Catalog, TempDir, Recipe7) {
    let (catalog, temp, table, x_path) = gate_table().await;

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

/// The 1000-BLOCKS each live position-delete file covers, one sorted+deduped `Vec` per output, themselves sorted so the comparison does not depend on manifest order.
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

// C-011 — exactly ONE `RewriteFiles` (one Replace snapshot) per admitted BIN.

/// C-011: `execute` iterates bins, not partitions, and commits one `RewriteFiles` per bin. one partition packed into two.
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

// C-037 — the abort contract: earlier bins STAND, no partial result reaches the caller.

/// C-037: a bin commit failure aborts `execute`. Earlier bins stay committed. No rollback.
#[tokio::test]
async fn test_bin_commit_failure_leaves_earlier_bins_committed() {
    let (catalog, _temp, fixture) = recipe_7_two_bin_fixture().await;
    let config = recipe_7_action(&fixture)
        .resolve_config()
        .expect("recipe 7's knobs are legal");
    assert_recipe_7_preconditions(&fixture, &config);

    let before = scan_y_values(&fixture.table).await;
    let history_before = fixture.table.metadata().history().len();

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

    assert_eq!(
        before,
        HashSet::from([10, 20, 30, 40, 50]),
        "fixture: the inputs mask positions no row occupies, so the pre-execute row set is full"
    );
}

// C-040 — an admitted BIN yielding ZERO pairs is skipped PER BIN.

/// Write a genuinely ZERO-ROW parquet position-delete file into `table`'s data directory, in partition `part_value`, and return the [`DataFile`] describing it.
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

/// C-040: an admitted bin with zero pairs is skipped. Later bins still run. zero to all four counts, commits.
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

/// Upgrade `table` to format version 3 — how a table acquires legacy parquet position deletes it can no longer write.
async fn upgrade_to_v3(catalog: &impl Catalog, table: &Table) -> Table {
    let tx = Transaction::new(table);
    let action = tx
        .upgrade_table_version()
        .set_format_version(FormatVersion::V3);
    let tx = action.apply(tx).unwrap();
    tx.commit(catalog).await.unwrap()
}

/// Write ONE Puffin file holding a deletion vector per `(target_path, positions)` entry, in partition x=`part_value`.
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
    let mut writer = DVFileWriter::new(output).unpartitioned();
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

/// Crown jewel: two parquet position deletes on a V2 table upgraded to V3 convert into one Puffin DV.
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

/// Each `DVFileWriter::delete` carries that data file's own `PartitionKey`.
/// `with_partition_spec` is not used: one Puffin spans every partition.
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

/// A data file with both a legacy parquet delete and a DV gets one merged DV.
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

/// Superseding one DV blob must leave every sibling of that Puffin live and applying.
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

/// The user filter restricts which partitions the V3 arm converts, exactly as it restricts which the bin-pack arm compacts.
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

/// A position naming a data file the snapshot no longer holds is DROPPED, not refused — it can delete nothing, which is what the V1/V2 arm effectively does too.
#[tokio::test]
async fn test_v3_position_naming_a_non_live_data_file_is_dropped() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 7, &[(7, 10, 1), (7, 11, 2)]).await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await;
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

/// A file-scoped delete is routed by path. Converting only the matching partition would shadow it.
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

/// Each DV carries its own plan max. A run-wide stamp writes false metadata.
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

/// A live ORC position delete on a V3 table is REFUSED, where the V1/V2 arm silently skips it.
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

/// Partition-scoped is the fork default (`truncate(16)` path bounds). The path-only guard is not enough.
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

/// Merging a non-superset DV would make a shadowed position effective and delete live rows.
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
    assert!(
        error
            .to_string()
            .contains("THIS ACTION CANNOT CLEAR THAT STATE"),
        "and says the arm cannot clear it: {error}"
    );
    assert!(
        error
            .to_string()
            .contains("RewriteDataFiles with remove_dangling_deletes(true)"),
        "and names the escape that DOES clear it, pinned by \
         test_v3_non_superset_refusal_is_cleared_by_rewrite_data_files: {error}"
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

/// An excluded ORC delete still shadows. No filter width converts that table.
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

/// Build limit (k)'s shape: data file A whose deletion vector masks position 0 while a live legacy position delete masks position 2 and is SHADOWED by that vector.
async fn build_non_superset_vector_table(catalog: &impl Catalog) -> (Table, HashSet<i64>) {
    let table = create_partitioned_table(catalog, FormatVersion::V2).await;
    let a = write_data_file(&table, "a.parquet", 7, &[
        (7, 10, 1),
        (7, 11, 2),
        (7, 12, 3),
    ])
    .await;
    let a_path = a.file_path().to_string();
    let table = append_files(catalog, &table, vec![a]).await;

    let shadowed = write_position_delete_file(&table, Some(7), &[(a_path.as_str(), 2)]).await;
    let table = add_deletes(catalog, &table, vec![shadowed]).await;
    let scratch = write_position_delete_file(&table, Some(7), &[(a_path.as_str(), 0)]).await;
    let scratch_path = scratch.file_path().to_string();
    let table = add_deletes(catalog, &table, vec![scratch]).await;
    let table = upgrade_to_v3(catalog, &table).await;

    let scratch_seq = live_delete_seq(&table, &scratch_path).await;
    let vector = write_deletion_vectors_in_one_puffin(&table, 7, &[(&a_path, &[0])]).await;
    let removed: Vec<DataFile> = live_delete_files(&table)
        .await
        .into_iter()
        .filter(|f| f.file_path() == scratch_path)
        .collect();
    let table = swap_delete_files(catalog, &table, removed, vector, scratch_seq).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([11, 12]),
        "fixture NON-VACUITY: the DV masks position 0 and SHADOWS the delete masking position 2"
    );
    (table, before)
}

/// Escape: `RewriteDataFiles` with `remove_dangling_deletes(true)`. Default delete-ratio 0.3 admits this file.
#[tokio::test]
async fn test_v3_non_superset_refusal_is_cleared_by_rewrite_data_files() {
    let (catalog, _temp) = local_fs_catalog().await;
    let (table, before) = build_non_superset_vector_table(&catalog).await;

    let refusal = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .expect_err("the arm refuses this shape");
    assert!(
        refusal
            .to_string()
            .contains("THIS ACTION CANNOT CLEAR THAT STATE")
    );

    let without_gc = RewriteDataFiles::new(table.clone())
        .execute(&catalog)
        .await
        .expect("the default delete-ratio admits this 1/3 file");
    assert_eq!(without_gc.rewritten_data_files_count, 1);
    assert_eq!(
        without_gc.removed_delete_files_count, 1,
        "the rewrite drops the DV; the shadowed parquet delete stays until GC"
    );
    let after_rewrite = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(scan_y_values(&after_rewrite).await, before);
    assert_eq!(
        live_delete_files(&after_rewrite).await.len(),
        1,
        "the parquet position delete is still live without remove_dangling_deletes"
    );

    let (table, before) = build_non_superset_vector_table(&catalog).await;
    let rewrite = RewriteDataFiles::new(table.clone())
        .remove_dangling_deletes(true)
        .execute(&catalog)
        .await
        .expect("RewriteDataFiles clears the shadowed state at default ratio");
    assert_eq!(rewrite.rewritten_data_files_count, 1);
    assert_eq!(rewrite.added_data_files_count, 1);
    assert_eq!(
        rewrite.removed_delete_files_count, 2,
        "the DV drops with the rewritten file; the shadowed parquet delete is GC'd"
    );

    let cleared = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&cleared).await,
        before,
        "the escape preserves the live rows exactly — it does not resolve the shadow by deleting"
    );
    assert!(
        live_delete_files(&cleared).await.is_empty(),
        "no delete file survives, so nothing is left to refuse"
    );

    // And the arm now runs clean on the same table: honest zeros, not a refusal.
    let second = RewritePositionDeleteFiles::new(cleared.clone())
        .execute(&catalog)
        .await
        .expect("the cleared table converts without refusing");
    assert_eq!(second, RewritePositionDeleteFilesResult::default());

    for knob in ["remove_dangling_deletes(true)", "delete-ratio-threshold"] {
        assert!(
            refusal.to_string().contains(knob),
            "the refusal names '{knob}', the escape this test actually runs: {refusal}"
        );
    }
}
