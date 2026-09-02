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

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch};
use futures::TryStreamExt;
use iceberg::io::LocalFsStorageFactory;
use iceberg::maintenance::{RewriteDataFiles, RewritePositionDeleteFiles};
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::metadata_columns::RESERVED_COL_NAME_ROW_ID;
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, Literal, ManifestContentType,
    NestedField, Operation, PartitionKey, PartitionSpec, PrimitiveType, Schema, SortOrder, Struct,
    Transform, Type,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
};
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use tempfile::TempDir;

type Row = (i64, i64, i64);

fn maintenance_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_V3_MAINTENANCE_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn maintenance_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "grp", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(3, "y", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("build the {id, grp, y} schema")
}

async fn build_catalog(name: &str, warehouse: &str) -> impl Catalog + use<> {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            name,
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_string())]),
        )
        .await
        .expect("build the local-fs memory catalog")
}

async fn create_seed_table(catalog: &impl Catalog, name: &str, location: &str) -> Table {
    let schema = maintenance_schema();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("grp", "grp", Transform::Identity)
        .expect("identity(grp)")
        .build()
        .expect("build identity(grp) spec");
    let namespace = NamespaceIdent::new("interop".to_string());
    let _ = catalog.create_namespace(&namespace, HashMap::new()).await;
    let creation = TableCreation::builder()
        .name(name.to_string())
        .location(location.to_string())
        .schema(schema)
        .partition_spec(spec)
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create the partitioned seed table")
}

fn partition_key_for(table: &Table, grp: i64) -> PartitionKey {
    PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        Struct::from_iter([Some(Literal::long(grp))]),
    )
    .expect("build the identity(grp) partition key")
}

async fn write_data_file(table: &Table, tag: &str, grp: i64, rows: &[Row]) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema to arrow"));
    let ids: Vec<i64> = rows.iter().map(|(id, _, _)| *id).collect();
    let grps: Vec<i64> = rows.iter().map(|(_, g, _)| *g).collect();
    let ys: Vec<i64> = rows.iter().map(|(_, _, y)| *y).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(Int64Array::from(grps)) as ArrayRef,
        Arc::new(Int64Array::from(ys)) as ArrayRef,
    ])
    .expect("build the {id, grp, y} batch");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        format!("data-{tag}"),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(Some(partition_key_for(table, grp)))
        .await
        .expect("build the data file writer");
    writer.write(batch).await.expect("write the data batch");
    writer
        .close()
        .await
        .expect("close the data writer")
        .into_iter()
        .next()
        .expect("exactly one data file")
}

async fn write_position_delete(
    table: &Table,
    tag: &str,
    grp: i64,
    pairs: &[(&str, i64)],
) -> DataFile {
    let config = PositionDeleteWriterConfig::new().expect("position-delete config");
    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        format!("posdel-{tag}"),
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
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(Some(partition_key_for(table, grp)))
        .await
        .expect("build the position-delete writer");
    let paths: Vec<&str> = pairs.iter().map(|(path, _)| *path).collect();
    let positions: Vec<i64> = pairs.iter().map(|(_, pos)| *pos).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(arrow_array::StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions)) as ArrayRef,
    ])
    .expect("build the position-delete batch");
    writer
        .write(batch)
        .await
        .expect("write the position-delete batch");
    writer
        .close()
        .await
        .expect("close the position-delete writer")
        .into_iter()
        .next()
        .expect("exactly one position-delete file")
}

async fn fast_append(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    tx.fast_append()
        .add_data_files(files)
        .apply(tx)
        .expect("apply fast_append")
        .commit(catalog)
        .await
        .expect("commit fast_append")
}

async fn add_deletes(catalog: &impl Catalog, table: &Table, deletes: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    tx.row_delta()
        .add_deletes(deletes)
        .apply(tx)
        .expect("apply row_delta")
        .commit(catalog)
        .await
        .expect("commit row_delta")
}

async fn upgrade_to_v3(catalog: &impl Catalog, table: &Table) -> Table {
    let tx = Transaction::new(table);
    tx.upgrade_table_version()
        .set_format_version(FormatVersion::V3)
        .apply(tx)
        .expect("apply the format-version upgrade")
        .commit(catalog)
        .await
        .expect("commit the format-version upgrade")
}

async fn scan_rows(table: &Table) -> Vec<Row> {
    let batches: Vec<RecordBatch> = table
        .scan()
        .select(["id", "grp", "y"])
        .build()
        .expect("build the scan")
        .to_arrow()
        .await
        .expect("scan to arrow")
        .try_collect()
        .await
        .expect("collect the batches");
    let mut rows = Vec::new();
    for batch in &batches {
        let ids = batch
            .column_by_name("id")
            .expect("id column")
            .as_primitive::<Int64Type>();
        let grps = batch
            .column_by_name("grp")
            .expect("grp column")
            .as_primitive::<Int64Type>();
        let ys = batch
            .column_by_name("y")
            .expect("y column")
            .as_primitive::<Int64Type>();
        for index in 0..batch.num_rows() {
            rows.push((ids.value(index), grps.value(index), ys.value(index)));
        }
    }
    rows.sort_unstable();
    rows
}

async fn scan_row_ids(table: &Table) -> BTreeMap<i64, Option<i64>> {
    let batches: Vec<RecordBatch> = table
        .scan()
        .select(["id", RESERVED_COL_NAME_ROW_ID])
        .build()
        .expect("build the lineage scan")
        .to_arrow()
        .await
        .expect("lineage scan to arrow")
        .try_collect()
        .await
        .expect("collect the lineage batches");
    let mut rows = BTreeMap::new();
    for batch in &batches {
        let ids = batch
            .column_by_name("id")
            .expect("id column")
            .as_primitive::<Int64Type>();
        let row_ids = batch
            .column_by_name(RESERVED_COL_NAME_ROW_ID)
            .expect("_row_id column")
            .as_primitive::<Int64Type>();
        for index in 0..batch.num_rows() {
            let row_id = if row_ids.is_valid(index) {
                Some(row_ids.value(index))
            } else {
                None
            };
            rows.insert(ids.value(index), row_id);
        }
    }
    rows
}

async fn live_files(table: &Table) -> (Vec<DataFile>, Vec<DataFile>) {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("the table has a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load the manifest list");
    let mut data = Vec::new();
    let mut deletes = Vec::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("load the manifest");
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            if entry.content_type() == DataContentType::Data {
                data.push(entry.data_file().clone());
            } else {
                deletes.push(entry.data_file().clone());
            }
        }
    }
    (data, deletes)
}

async fn manifest_counts(table: &Table) -> (usize, usize) {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("the table has a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load the manifest list");
    let data = manifest_list
        .entries()
        .iter()
        .filter(|manifest| manifest.content == ManifestContentType::Data)
        .count();
    let deletes = manifest_list
        .entries()
        .iter()
        .filter(|manifest| manifest.content == ManifestContentType::Deletes)
        .count();
    (data, deletes)
}

async fn assigned_row_ranges(table: &Table) -> BTreeMap<String, (i64, u64)> {
    let (data, _) = live_files(table).await;
    let mut ranges = BTreeMap::new();
    for file in data {
        if let Some(first) = file.first_row_id() {
            ranges.insert(file.file_path().to_string(), (first, file.record_count()));
        }
    }
    ranges
}

fn data_paths(files: &[DataFile]) -> BTreeSet<String> {
    files
        .iter()
        .map(|file| file.file_path().to_string())
        .collect()
}

fn parquet_position_deletes(files: &[DataFile]) -> usize {
    files
        .iter()
        .filter(|file| {
            file.content_type() == DataContentType::PositionDeletes
                && file.file_format() == DataFileFormat::Parquet
        })
        .count()
}

fn puffin_deletes(files: &[DataFile]) -> usize {
    files
        .iter()
        .filter(|file| file.file_format() == DataFileFormat::Puffin)
        .count()
}

async fn write_stage(table: &Table, out_root: Option<&Path>, stage: &str) {
    let Some(root) = out_root else {
        return;
    };
    let dir = root.join(stage).join("metadata");
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    let path = dir.join("final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), path.to_string_lossy().to_string())
        .await
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

async fn seed_plain(catalog: &impl Catalog, table: Table) -> Table {
    let f1 = write_data_file(&table, "g1a", 1, &[
        (100, 1, 10),
        (110, 1, 10),
        (120, 1, 20),
    ])
    .await;
    let f3 = write_data_file(&table, "g2a", 2, &[(200, 2, 10), (210, 2, 20)]).await;
    let table = fast_append(catalog, &table, vec![f1, f3]).await;
    let f2 = write_data_file(&table, "g1b", 1, &[(130, 1, 20), (140, 1, 10)]).await;
    let f4 = write_data_file(&table, "g2b", 2, &[(220, 2, 20), (230, 2, 10)]).await;
    fast_append(catalog, &table, vec![f2, f4]).await
}

async fn seed_with_deletes(catalog: &impl Catalog, table: Table) -> Table {
    let f1 = write_data_file(&table, "g1a", 1, &[
        (100, 1, 10),
        (110, 1, 10),
        (120, 1, 20),
    ])
    .await;
    let f3 = write_data_file(&table, "g2a", 2, &[(200, 2, 10), (210, 2, 20)]).await;
    let f1_path = f1.file_path().to_string();
    let f3_path = f3.file_path().to_string();
    let table = fast_append(catalog, &table, vec![f1, f3]).await;
    let f2 = write_data_file(&table, "g1b", 1, &[(130, 1, 20), (140, 1, 10)]).await;
    let f4 = write_data_file(&table, "g2b", 2, &[(220, 2, 20), (230, 2, 10)]).await;
    let table = fast_append(catalog, &table, vec![f2, f4]).await;
    let d1 = write_position_delete(&table, "g1", 1, &[(&f1_path, 1)]).await;
    let d2 = write_position_delete(&table, "g2", 2, &[(&f3_path, 0)]).await;
    add_deletes(catalog, &table, vec![d1, d2]).await
}

async fn prepare_v3(catalog: &impl Catalog, table: &Table) -> Table {
    let table = upgrade_to_v3(catalog, table).await;
    let a = write_data_file(&table, "g1c", 1, &[(150, 1, 10)]).await;
    let b = write_data_file(&table, "g2c", 2, &[(250, 2, 20)]).await;
    fast_append(catalog, &table, vec![a, b]).await
}

fn current_operation(table: &Table) -> Operation {
    table
        .metadata()
        .current_snapshot()
        .expect("the table has a current snapshot")
        .summary()
        .operation
        .clone()
}

fn assert_distinct_row_ids(row_ids: &BTreeMap<i64, Option<i64>>) {
    let assigned: BTreeSet<i64> = row_ids.values().filter_map(|row_id| *row_id).collect();
    assert_eq!(
        assigned.len(),
        row_ids.len(),
        "the first V3 commit must give every live row a distinct row id"
    );
}

async fn run_rewrite_matrix(catalog: &impl Catalog, table: Table, out_root: Option<&Path>) {
    let ident = table.identifier().clone();
    let table = prepare_v3(catalog, &table).await;
    let before_rows = scan_rows(&table).await;
    let before_ids = scan_row_ids(&table).await;
    assert_distinct_row_ids(&before_ids);
    write_expected(out_root, &before_rows, &before_ids);
    let (before_data, _) = live_files(&table).await;
    let before_paths = data_paths(&before_data);
    assert_eq!(table.metadata().format_version(), FormatVersion::V3);
    assert_eq!(
        current_operation(&table),
        Operation::Append,
        "the seed stage must be an append, so a Replace assertion below cannot pass on a no-op"
    );
    write_stage(&table, out_root, "m0").await;

    let result = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(4_000_000)
        .min_input_files(2)
        .execute(catalog)
        .await
        .expect("M1 RewriteDataFiles on the current spec");
    assert!(result.rewritten_data_files_count >= 2);
    assert!(result.added_data_files_count >= 1);
    let m1 = catalog.load_table(&ident).await.expect("reload after M1");
    let (m1_data, m1_deletes) = live_files(&m1).await;
    assert_eq!(scan_rows(&m1).await, before_rows);
    assert_ne!(data_paths(&m1_data), before_paths);
    assert_eq!(scan_row_ids(&m1).await, before_ids);
    assert!(m1_deletes.is_empty());
    assert_eq!(current_operation(&m1), Operation::Replace);
    write_stage(&m1, out_root, "m1").await;

    let tx = Transaction::new(&m1);
    let evolved = tx
        .update_partition_spec()
        .remove_field("grp")
        .add_field("y")
        .apply(tx)
        .expect("apply the spec evolution")
        .commit(catalog)
        .await
        .expect("commit the spec evolution");
    let evolved_spec_id = evolved.metadata().default_partition_spec().spec_id();
    let result = RewriteDataFiles::new(evolved.clone())
        .target_file_size_bytes(4_000_000)
        .min_input_files(2)
        .execute(catalog)
        .await
        .expect("M2 RewriteDataFiles after the spec evolution");
    assert!(result.rewritten_data_files_count >= 2);
    let m2 = catalog.load_table(&ident).await.expect("reload after M2");
    let (m2_data, _) = live_files(&m2).await;
    assert_eq!(scan_rows(&m2).await, before_rows);
    assert_eq!(scan_row_ids(&m2).await, before_ids);
    for file in &m2_data {
        assert_eq!(file.partition_spec_id(), evolved_spec_id);
    }
    assert_eq!(current_operation(&m2), Operation::Replace);
    write_stage(&m2, out_root, "m2").await;
}

async fn run_delete_matrix(catalog: &impl Catalog, table: Table, out_root: Option<&Path>) {
    let ident = table.identifier().clone();
    let table = prepare_v3(catalog, &table).await;
    let before_rows = scan_rows(&table).await;
    let before_ids = scan_row_ids(&table).await;
    assert_distinct_row_ids(&before_ids);
    write_expected(out_root, &before_rows, &before_ids);
    let (_, before_deletes) = live_files(&table).await;
    assert_eq!(parquet_position_deletes(&before_deletes), 2);
    assert_eq!(
        current_operation(&table),
        Operation::Append,
        "the seed stage must be an append, so a Replace assertion below cannot pass on a no-op"
    );
    write_stage(&table, out_root, "m0").await;

    let result = RewritePositionDeleteFiles::new(table.clone())
        .min_input_files(1)
        .execute(catalog)
        .await
        .expect("M3 RewritePositionDeleteFiles converting to deletion vectors");
    assert_eq!(result.rewritten_delete_files_count, 2);
    assert!(result.added_delete_files_count >= 1);
    let m3 = catalog.load_table(&ident).await.expect("reload after M3");
    let (_, m3_deletes) = live_files(&m3).await;
    assert_eq!(scan_rows(&m3).await, before_rows);
    assert_eq!(parquet_position_deletes(&m3_deletes), 0);
    assert!(puffin_deletes(&m3_deletes) >= 1);
    assert_eq!(scan_row_ids(&m3).await, before_ids);
    assert_eq!(current_operation(&m3), Operation::Replace);
    write_stage(&m3, out_root, "m3").await;

    let (m3_data_manifests, m3_delete_manifests) = manifest_counts(&m3).await;
    assert!(m3_data_manifests > 1);
    assert!(m3_delete_manifests >= 1);
    let m3_ranges = assigned_row_ranges(&m3).await;
    assert_eq!(
        m3_ranges.len(),
        6,
        "every live data file must carry a row-id range, or the m4 range comparison is vacuous"
    );
    let next_row_id = i64::try_from(m3.metadata().next_row_id()).expect("next_row_id fits i64");

    let tx = Transaction::new(&m3);
    let m4 = tx
        .rewrite_manifests()
        .cluster_by(|_| "all".to_string())
        .apply(tx)
        .expect("apply M4 RewriteManifests")
        .commit(catalog)
        .await
        .expect("commit M4 RewriteManifests");
    let (m4_data_manifests, m4_delete_manifests) = manifest_counts(&m4).await;
    assert_eq!(m4_data_manifests, 1);
    assert_eq!(m4_delete_manifests, m3_delete_manifests);
    assert_eq!(scan_rows(&m4).await, before_rows);
    assert_eq!(scan_row_ids(&m4).await, before_ids);
    assert_eq!(assigned_row_ranges(&m4).await, m3_ranges);
    assert_eq!(
        i64::try_from(m4.metadata().next_row_id()).expect("next_row_id fits i64"),
        next_row_id
    );
    for (first, count) in m4_ranges_values(&assigned_row_ranges(&m4).await) {
        assert!(first + i64::try_from(count).expect("record count fits i64") <= next_row_id);
    }
    assert_eq!(current_operation(&m4), Operation::Replace);
    write_stage(&m4, out_root, "m4").await;

    let before_snapshots = m4.metadata().snapshots().count();
    let current = m4
        .metadata()
        .current_snapshot()
        .expect("current snapshot")
        .snapshot_id();
    let oldest = m4
        .metadata()
        .snapshots()
        .map(|snapshot| snapshot.timestamp_ms())
        .max()
        .expect("a snapshot timestamp");
    let tx = Transaction::new(&m4);
    let m5 = tx
        .expire_snapshots()
        .expire_older_than(oldest)
        .retain_last(1)
        .apply(tx)
        .expect("apply M5 ExpireSnapshots")
        .commit(catalog)
        .await
        .expect("commit M5 ExpireSnapshots");
    assert!(m5.metadata().snapshots().count() < before_snapshots);
    assert_eq!(
        m5.metadata()
            .current_snapshot()
            .expect("current snapshot")
            .snapshot_id(),
        current
    );
    assert_eq!(scan_rows(&m5).await, before_rows);
    assert_eq!(scan_row_ids(&m5).await, before_ids);
    assert_eq!(
        current_operation(&m5),
        Operation::Replace,
        "expiry keeps the current snapshot, so its operation is still the clustering commit's"
    );
    write_stage(&m5, out_root, "m5").await;
}

fn m4_ranges_values(ranges: &BTreeMap<String, (i64, u64)>) -> Vec<(i64, u64)> {
    ranges.values().copied().collect()
}

#[tokio::test]
async fn test_v3_rewrite_matrix_preserves_rows_and_lineage() {
    let temp = TempDir::new().expect("temp dir");
    let warehouse = temp.path().to_string_lossy().to_string();
    let catalog = build_catalog("v3_maintenance_rewrite", &warehouse).await;
    let table = create_seed_table(&catalog, "plain", &format!("{warehouse}/plain")).await;
    let table = seed_plain(&catalog, table).await;
    run_rewrite_matrix(&catalog, table, None).await;
}

#[tokio::test]
async fn test_v3_delete_matrix_converts_and_reclusters_without_losing_lineage() {
    let temp = TempDir::new().expect("temp dir");
    let warehouse = temp.path().to_string_lossy().to_string();
    let catalog = build_catalog("v3_maintenance_delete", &warehouse).await;
    let table = create_seed_table(&catalog, "deletes", &format!("{warehouse}/deletes")).await;
    let table = seed_with_deletes(&catalog, table).await;
    run_delete_matrix(&catalog, table, None).await;
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct MaintenanceRow {
    id: i64,
    grp: i64,
    y: i64,
    row_id: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct MaintenanceExpectation {
    rows: Vec<MaintenanceRow>,
}

fn expectation_rows(rows: &[Row], row_ids: &BTreeMap<i64, Option<i64>>) -> Vec<MaintenanceRow> {
    rows.iter()
        .map(|(id, grp, y)| MaintenanceRow {
            id: *id,
            grp: *grp,
            y: *y,
            row_id: *row_ids
                .get(id)
                .expect("a row id entry for every scanned id"),
        })
        .collect()
}

fn write_expected(out_root: Option<&Path>, rows: &[Row], row_ids: &BTreeMap<i64, Option<i64>>) {
    let Some(root) = out_root else {
        return;
    };
    std::fs::create_dir_all(root).unwrap_or_else(|e| panic!("create {}: {e}", root.display()));
    let path = root.join("expected.json");
    let expectation = MaintenanceExpectation {
        rows: expectation_rows(rows, row_ids),
    };
    std::fs::write(
        &path,
        serde_json::to_string(&expectation).expect("serialize the expectation"),
    )
    .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn read_java_rows(path: &Path) -> Vec<MaintenanceRow> {
    let json =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&json).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

async fn register_java_seed(catalog: &impl Catalog, dir: &Path, name: &str) -> Table {
    let meta = dir.join("metadata").join("final.metadata.json");
    assert!(
        meta.is_file(),
        "missing Java fixture {} — the Java generate step must run first",
        meta.display()
    );
    let staged = dir
        .join("metadata")
        .join(format!("99999-{}.metadata.json", uuid::Uuid::now_v7()));
    std::fs::copy(&meta, &staged).unwrap_or_else(|e| panic!("copy {}: {e}", meta.display()));
    let namespace = NamespaceIdent::new("interop".to_string());
    let _ = catalog.create_namespace(&namespace, HashMap::new()).await;
    catalog
        .register_table(
            &TableIdent::new(namespace, name.to_string()),
            staged.to_string_lossy().to_string(),
        )
        .await
        .unwrap_or_else(|e| panic!("register {}: {e}", staged.display()))
}

async fn assert_matches_java_seed(table: &Table, java_rows: &Path) {
    let expected = read_java_rows(java_rows);
    let rows = scan_rows(table).await;
    let row_ids = scan_row_ids(table).await;
    assert_eq!(
        expectation_rows(&rows, &row_ids),
        expected,
        "Rust must read the Java V2 seed exactly as Java read it"
    );
}

#[tokio::test]
async fn gen_rust_runs_the_v3_rewrite_matrix_over_the_java_seed() {
    let Some(dir) = maintenance_dir() else {
        return;
    };
    let warehouse = dir.to_string_lossy().to_string();
    let catalog = build_catalog("v3_maintenance_plain_gen", &warehouse).await;
    let seed = dir.join("java_v2_plain");
    let table = register_java_seed(&catalog, &seed, "plain").await;
    assert_eq!(table.metadata().format_version(), FormatVersion::V2);
    assert_matches_java_seed(&table, &seed.join("java_rows.json")).await;
    let out = dir.join("plain");
    run_rewrite_matrix(&catalog, table, Some(&out)).await;
}

#[tokio::test]
async fn gen_rust_runs_the_v3_delete_matrix_over_the_java_seed() {
    let Some(dir) = maintenance_dir() else {
        return;
    };
    let warehouse = dir.to_string_lossy().to_string();
    let catalog = build_catalog("v3_maintenance_deletes_gen", &warehouse).await;
    let seed = dir.join("java_v2_deletes");
    let table = register_java_seed(&catalog, &seed, "deletes").await;
    assert_eq!(table.metadata().format_version(), FormatVersion::V2);
    assert_matches_java_seed(&table, &seed.join("java_rows.json")).await;
    let out = dir.join("deletes");
    run_delete_matrix(&catalog, table, Some(&out)).await;
}
