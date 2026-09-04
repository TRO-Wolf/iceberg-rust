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
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use iceberg::io::LocalFsStorageFactory;
use iceberg::maintenance::RewritePositionDeleteFiles;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::metadata_columns::RESERVED_COL_NAME_ROW_ID;
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, NestedField, PrimitiveType, Schema,
    SortOrder, Type,
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

fn upgrade_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_V3_UPGRADE_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn upgrade_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "val", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build the {id long, val string} schema")
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

async fn create_table(
    catalog: &impl Catalog,
    name: &str,
    location: &str,
    format_version: FormatVersion,
) -> Table {
    let namespace = NamespaceIdent::new("interop".to_string());
    let _ = catalog.create_namespace(&namespace, HashMap::new()).await;
    let creation = TableCreation::builder()
        .name(name.to_string())
        .location(location.to_string())
        .schema(upgrade_schema())
        .partition_spec(iceberg::spec::PartitionSpec::unpartition_spec().into_unbound())
        .sort_order(SortOrder::unsorted_order())
        .format_version(format_version)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create the unpartitioned table")
}

async fn write_data_file(table: &Table, tag: &str, rows: &[(i64, &str)]) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema to arrow"));
    let ids: Vec<i64> = rows.iter().map(|(id, _)| *id).collect();
    let vals: Vec<String> = rows.iter().map(|(_, val)| (*val).to_string()).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(vals)) as ArrayRef,
    ])
    .expect("build the {id, val} batch");

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
        .unpartitioned()
        .build(None)
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

async fn write_position_delete(table: &Table, tag: &str, pairs: &[(&str, i64)]) -> DataFile {
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
        .unpartitioned()
        .build(None)
        .await
        .expect("build the position-delete writer");
    let paths: Vec<&str> = pairs.iter().map(|(path, _)| *path).collect();
    let positions: Vec<i64> = pairs.iter().map(|(_, pos)| *pos).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
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

async fn scan_rows(table: &Table) -> Vec<(i64, String)> {
    let batches: Vec<RecordBatch> = table
        .scan()
        .select(["id", "val"])
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
        let vals = batch.column_by_name("val").expect("val column");
        let vals = vals.as_string::<i32>();
        for index in 0..batch.num_rows() {
            rows.push((ids.value(index), vals.value(index).to_string()));
        }
    }
    rows.sort();
    rows
}

async fn scan_row_ids(table: &Table) -> Vec<(i64, Option<i64>)> {
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
    let mut rows = Vec::new();
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
            rows.push((ids.value(index), row_id));
        }
    }
    rows.sort();
    rows
}

async fn live_delete_files(table: &Table) -> Vec<DataFile> {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("the table has a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load the manifest list");
    let mut files = Vec::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("load the manifest");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.content_type() != DataContentType::Data {
                files.push(entry.data_file().clone());
            }
        }
    }
    files
}

fn snapshot_sequence_numbers(table: &Table) -> Vec<i64> {
    let mut seqs: Vec<i64> = table
        .metadata()
        .snapshots()
        .map(|snapshot| snapshot.sequence_number())
        .collect();
    seqs.sort_unstable();
    seqs
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct ExpectedRow {
    id: i64,
    val: String,
    row_id: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct UpgradeExpectation {
    format_version: u8,
    next_row_id: i64,
    snapshot_sequence_numbers: Vec<i64>,
    rows: Vec<ExpectedRow>,
}

async fn expectation(table: &Table) -> UpgradeExpectation {
    let values: BTreeMap<i64, String> = scan_rows(table).await.into_iter().collect();
    let rows = scan_row_ids(table)
        .await
        .into_iter()
        .map(|(id, row_id)| ExpectedRow {
            id,
            val: values
                .get(&id)
                .cloned()
                .expect("a value for every scanned id"),
            row_id,
        })
        .collect();
    UpgradeExpectation {
        format_version: match table.metadata().format_version() {
            FormatVersion::V1 => 1,
            FormatVersion::V2 => 2,
            FormatVersion::V3 => 3,
        },
        next_row_id: i64::try_from(table.metadata().next_row_id()).expect("next_row_id fits i64"),
        snapshot_sequence_numbers: snapshot_sequence_numbers(table),
        rows,
    }
}

fn read_expectation(path: &Path) -> UpgradeExpectation {
    let json =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&json).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn write_expectation(path: &Path, expectation: &UpgradeExpectation) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("create {}: {e}", parent.display()));
    }
    std::fs::write(
        path,
        serde_json::to_string(expectation).expect("serialize the expectation"),
    )
    .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

async fn write_final(table: &Table, out_dir: &Path) {
    let dir = out_dir.join("metadata");
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    let path = dir.join("final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), path.to_string_lossy().to_string())
        .await
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

async fn register_java_table(catalog: &impl Catalog, dir: &Path, name: &str) -> Table {
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

fn load_metadata_table(meta: &Path, name: &str) -> Table {
    let json =
        std::fs::read_to_string(meta).unwrap_or_else(|e| panic!("read {}: {e}", meta.display()));
    let metadata: iceberg::spec::TableMetadata =
        serde_json::from_str(&json).unwrap_or_else(|e| panic!("parse {}: {e}", meta.display()));
    Table::builder()
        .metadata(metadata)
        .metadata_location(meta.to_string_lossy().to_string())
        .identifier(TableIdent::from_strs(["interop", name]).expect("table identifier"))
        .file_io(iceberg::io::FileIO::new_with_fs())
        .build()
        .expect("build the read-only table")
}

#[tokio::test]
async fn test_v2_upgrade_to_v3_assigns_row_ids_to_appends_after_the_upgrade_only() {
    let temp = TempDir::new().expect("temp dir");
    let warehouse = temp.path().to_string_lossy().to_string();
    let catalog = build_catalog("v3_upgrade_offline", &warehouse).await;
    let table = create_table(
        &catalog,
        "seed",
        &format!("{warehouse}/seed"),
        FormatVersion::V2,
    )
    .await;
    let file = write_data_file(&table, "seed", &[(1, "a"), (2, "b"), (3, "c")]).await;
    let table = fast_append(&catalog, &table, vec![file]).await;
    assert_eq!(table.metadata().format_version(), FormatVersion::V2);

    let table = upgrade_to_v3(&catalog, &table).await;
    assert_eq!(table.metadata().format_version(), FormatVersion::V3);
    assert_eq!(table.metadata().next_row_id(), 0);
    assert_eq!(scan_row_ids(&table).await, vec![
        (1, None),
        (2, None),
        (3, None)
    ]);
    assert_eq!(snapshot_sequence_numbers(&table), vec![1]);

    let file = write_data_file(&table, "post", &[(4, "d"), (5, "e")]).await;
    let table = fast_append(&catalog, &table, vec![file]).await;
    assert_eq!(table.metadata().next_row_id(), 5);
    assert_eq!(snapshot_sequence_numbers(&table), vec![1, 2]);
    let row_ids = scan_row_ids(&table).await;
    assert_eq!(row_ids.len(), 5);
    let assigned: BTreeSet<i64> = row_ids.iter().filter_map(|(_, row_id)| *row_id).collect();
    assert_eq!(
        assigned.len(),
        5,
        "the first V3 commit assigns a distinct row id to every live row, carried files included"
    );
    assert_eq!(assigned.iter().copied().max(), Some(4));
    assert_eq!(
        row_ids
            .iter()
            .filter(|(id, _)| *id >= 4)
            .filter_map(|(_, row_id)| *row_id)
            .collect::<BTreeSet<i64>>(),
        BTreeSet::from([0, 1]),
        "the appended manifest is assigned first, so the new rows take the lowest ids"
    );
}

#[tokio::test]
async fn test_upgraded_v3_converts_legacy_parquet_deletes_and_keeps_no_parquet_delete() {
    let temp = TempDir::new().expect("temp dir");
    let warehouse = temp.path().to_string_lossy().to_string();
    let catalog = build_catalog("v3_upgrade_dv_offline", &warehouse).await;
    let table = create_table(
        &catalog,
        "seed",
        &format!("{warehouse}/seed"),
        FormatVersion::V2,
    )
    .await;
    let file = write_data_file(&table, "seed", &[(1, "a"), (2, "b"), (3, "c")]).await;
    let path = file.file_path().to_string();
    let table = fast_append(&catalog, &table, vec![file]).await;
    let delete = write_position_delete(&table, "one", &[(&path, 1)]).await;
    let table = add_deletes(&catalog, &table, vec![delete]).await;
    let table = upgrade_to_v3(&catalog, &table).await;
    let before = scan_rows(&table).await;
    assert_eq!(before, vec![(1, "a".to_string()), (3, "c".to_string())]);
    assert_eq!(live_delete_files(&table).await.len(), 1);

    let result = RewritePositionDeleteFiles::new(table.clone())
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("convert the legacy parquet position deletes");
    assert_eq!(result.rewritten_delete_files_count, 1);
    assert_eq!(result.added_delete_files_count, 1);
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload after the conversion");
    assert_eq!(scan_rows(&table).await, before);
    let deletes = live_delete_files(&table).await;
    assert_eq!(deletes.len(), 1);
    assert_eq!(deletes[0].file_format(), DataFileFormat::Puffin);
    assert_eq!(
        deletes
            .iter()
            .filter(|file| file.file_format() == DataFileFormat::Parquet)
            .count(),
        0
    );
}

const U1_APPEND: &[(i64, &str)] = &[(4, "d"), (5, "e")];
const SEED_ROWS: &[(i64, &str)] = &[(1, "a"), (2, "b"), (3, "c")];
const DELETE_SEED_ROWS: &[(i64, &str)] = &[(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")];

#[tokio::test]
async fn gen_rust_upgrades_java_v2_and_appends() {
    let Some(dir) = upgrade_dir() else {
        return;
    };
    let warehouse = dir.to_string_lossy().to_string();
    let catalog = build_catalog("v3_upgrade_u1", &warehouse).await;
    let table = register_java_table(&catalog, &dir.join("u1").join("java_v2"), "u1").await;
    assert_eq!(table.metadata().format_version(), FormatVersion::V2);
    assert_eq!(scan_rows(&table).await.len(), SEED_ROWS.len());

    let table = upgrade_to_v3(&catalog, &table).await;
    assert_eq!(table.metadata().format_version(), FormatVersion::V3);
    let file = write_data_file(&table, "u1-rust", U1_APPEND).await;
    let table = fast_append(&catalog, &table, vec![file]).await;

    let expectation = expectation(&table).await;
    assert_eq!(expectation.format_version, 3);
    assert_eq!(expectation.rows.len(), 5);
    assert!(expectation.rows.iter().all(|row| row.row_id.is_some()));
    assert_eq!(
        expectation.snapshot_sequence_numbers.len(),
        2,
        "the upgrade is metadata-only: only the two appends carry a sequence number"
    );
    let assigned: BTreeSet<i64> = expectation
        .rows
        .iter()
        .filter_map(|row| row.row_id)
        .collect();
    assert_eq!(
        assigned.len(),
        5,
        "every live row carries a distinct row id"
    );
    assert!(assigned.iter().all(|id| *id < expectation.next_row_id));

    write_final(&table, &dir.join("u1").join("rust_v3")).await;
    write_expectation(&dir.join("u1").join("rust_expected.json"), &expectation);
}

#[tokio::test]
async fn gen_rust_writes_v2_seed_for_the_java_upgrade() {
    let Some(dir) = upgrade_dir() else {
        return;
    };
    let warehouse = dir.to_string_lossy().to_string();
    let catalog = build_catalog("v3_upgrade_u2", &warehouse).await;
    let location = format!("{warehouse}/u2/rust_v2");
    let table = create_table(&catalog, "u2", &location, FormatVersion::V2).await;
    let file = write_data_file(&table, "u2-seed", SEED_ROWS).await;
    let table = fast_append(&catalog, &table, vec![file]).await;
    assert_eq!(table.metadata().format_version(), FormatVersion::V2);
    assert_eq!(scan_rows(&table).await.len(), 3);
    write_final(&table, Path::new(&location)).await;
}

#[tokio::test]
async fn gen_rust_converts_java_v2_position_deletes_after_upgrade() {
    let Some(dir) = upgrade_dir() else {
        return;
    };
    let warehouse = dir.to_string_lossy().to_string();
    let catalog = build_catalog("v3_upgrade_u3", &warehouse).await;
    let table = register_java_table(&catalog, &dir.join("u3").join("java_v2"), "u3").await;
    assert_eq!(table.metadata().format_version(), FormatVersion::V2);
    let java_pre = read_expectation(&dir.join("u3").join("java_pre_rows.json"));
    let rust_pre = expectation(&table).await;
    assert_eq!(
        rust_pre, java_pre,
        "Rust must read the Java V2 table with its parquet position delete applied"
    );
    let deletes = live_delete_files(&table).await;
    assert_eq!(
        deletes
            .iter()
            .filter(|file| file.file_format() == DataFileFormat::Parquet)
            .count(),
        1
    );

    let table = upgrade_to_v3(&catalog, &table).await;
    let result = RewritePositionDeleteFiles::new(table.clone())
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("convert the Java parquet position delete into a deletion vector");
    assert_eq!(result.rewritten_delete_files_count, 1);
    assert_eq!(result.added_delete_files_count, 1);
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload after the conversion");
    assert_eq!(scan_rows(&table).await, scan_rows_of(&java_pre));
    let deletes = live_delete_files(&table).await;
    assert_eq!(
        deletes
            .iter()
            .filter(|file| file.file_format() == DataFileFormat::Parquet)
            .count(),
        0,
        "the V3 table must carry no parquet position delete after the conversion"
    );
    assert!(
        deletes
            .iter()
            .any(|file| file.file_format() == DataFileFormat::Puffin)
    );
    write_final(&table, &dir.join("u3").join("rust_v3_dv")).await;
}

#[tokio::test]
async fn gen_rust_writes_v2_position_deletes_for_the_java_conversion() {
    let Some(dir) = upgrade_dir() else {
        return;
    };
    let warehouse = dir.to_string_lossy().to_string();
    let catalog = build_catalog("v3_upgrade_u4", &warehouse).await;
    let location = format!("{warehouse}/u4/rust_v2");
    let table = create_table(&catalog, "u4", &location, FormatVersion::V2).await;
    let file = write_data_file(&table, "u4-seed", DELETE_SEED_ROWS).await;
    let path = file.file_path().to_string();
    let table = fast_append(&catalog, &table, vec![file]).await;
    let delete = write_position_delete(&table, "u4", &[(&path, 3)]).await;
    let table = add_deletes(&catalog, &table, vec![delete]).await;
    assert_eq!(scan_rows(&table).await.len(), 4);
    assert_eq!(live_delete_files(&table).await.len(), 1);
    write_final(&table, Path::new(&location)).await;
    write_expectation(
        &dir.join("u4").join("rust_pre_rows.json"),
        &expectation(&table).await,
    );
}

#[tokio::test]
async fn verify_rust_reads_the_java_upgraded_v3_table() {
    let Some(dir) = upgrade_dir() else {
        return;
    };
    let meta = dir
        .join("u2")
        .join("java_v3")
        .join("metadata")
        .join("final.metadata.json");
    assert!(
        meta.is_file(),
        "missing {} — the Java upgrade step must run first",
        meta.display()
    );
    let table = load_metadata_table(&meta, "u2");
    let expected = read_expectation(&dir.join("u2").join("java_expected.json"));
    let actual = expectation(&table).await;
    assert_eq!(actual.format_version, 3);
    assert_eq!(
        actual, expected,
        "Rust and Java must agree on rows, format version, snapshot sequence numbers and row ids"
    );
    assert!(actual.rows.iter().all(|row| row.row_id.is_some()));
}

#[tokio::test]
async fn verify_rust_reads_the_java_converted_deletion_vectors() {
    let Some(dir) = upgrade_dir() else {
        return;
    };
    let meta = dir
        .join("u4")
        .join("java_v3_dv")
        .join("metadata")
        .join("final.metadata.json");
    assert!(
        meta.is_file(),
        "missing {} — the Java conversion step must run first",
        meta.display()
    );
    let table = load_metadata_table(&meta, "u4");
    let pre = read_expectation(&dir.join("u4").join("rust_pre_rows.json"));
    let expected = read_expectation(&dir.join("u4").join("java_expected.json"));
    let actual = expectation(&table).await;
    assert_eq!(actual.format_version, 3);
    assert_eq!(
        actual, expected,
        "Rust and Java must agree on the converted V3 table"
    );
    assert_eq!(
        actual.rows.iter().map(|row| row.id).collect::<Vec<_>>(),
        pre.rows.iter().map(|row| row.id).collect::<Vec<_>>(),
        "the Java conversion must mask exactly the rows the parquet position delete masked"
    );
    let deletes = live_delete_files(&table).await;
    assert_eq!(
        deletes
            .iter()
            .filter(|file| file.file_format() == DataFileFormat::Parquet)
            .count(),
        0
    );
    assert!(
        deletes
            .iter()
            .any(|file| file.file_format() == DataFileFormat::Puffin)
    );
    let assigned: BTreeSet<i64> = actual.rows.iter().filter_map(|row| row.row_id).collect();
    assert!(
        assigned.iter().all(|id| *id < actual.next_row_id),
        "every assigned row id stays below next_row_id"
    );
}

fn scan_rows_of(expectation: &UpgradeExpectation) -> Vec<(i64, String)> {
    let mut rows: Vec<(i64, String)> = expectation
        .rows
        .iter()
        .map(|row| (row.id, row.val.clone()))
        .collect();
    rows.sort();
    rows
}
