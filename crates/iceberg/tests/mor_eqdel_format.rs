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

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use iceberg::arrow::{arrow_schema_to_schema, schema_to_arrow_schema};
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, NestedField, PrimitiveType, Schema, SchemaRef,
    SortOrder, Type,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::base_writer::equality_delete_writer::{
    EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
};
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::file_writer::{
    AvroWriterBuilder, FileWriterBuilder, OrcWriterBuilder, ParquetWriterBuilder,
};
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation};
use tempfile::TempDir;

fn two_col_schema(key_name: &str, value_name: &str) -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, key_name, Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, value_name, Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build the two-column schema")
}

async fn setup(schema: Schema) -> (MemoryCatalog, Table, TempDir) {
    let warehouse = TempDir::new().expect("create warehouse");
    let warehouse_path = warehouse
        .path()
        .to_str()
        .expect("warehouse path is UTF-8")
        .to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
        )
        .await
        .expect("build memory catalog");
    let namespace = NamespaceIdent::new("mor_eqdel_format".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("target".to_string())
        .location(format!("{warehouse_path}/target"))
        .schema(schema)
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table");
    (catalog, table, warehouse)
}

async fn write_data_file(table: &Table, keys: Vec<i64>, values: Vec<&str>) -> DataFile {
    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("schema to arrow"));
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(keys)) as ArrayRef,
        Arc::new(StringArray::from(values)) as ArrayRef,
    ])
    .expect("build data batch");
    let location_gen = DefaultLocationGenerator::new(table.metadata()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "eqdeldata".to_string(),
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
        .expect("build data file writer");
    writer.write(batch).await.expect("write data batch");
    writer
        .close()
        .await
        .expect("close data file writer")
        .into_iter()
        .next()
        .expect("one data file")
}

async fn write_eq_delete_file<B>(
    table: &Table,
    format: DataFileFormat,
    prefix: &str,
    keys: Vec<i64>,
    make_builder: impl FnOnce(SchemaRef) -> B,
) -> DataFile
where
    B: FileWriterBuilder,
{
    let schema = table.metadata().current_schema();
    let config = EqualityDeleteWriterConfig::new(vec![1], schema.clone())
        .expect("equality-delete writer config");
    let location_gen = DefaultLocationGenerator::new(table.metadata()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        prefix.to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        format,
    );
    let projected = Arc::new(
        arrow_schema_to_schema(config.projected_arrow_schema_ref())
            .expect("projected arrow schema to iceberg schema"),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        make_builder(projected),
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .unpartitioned()
        .build(None)
        .await
        .expect("build equality-delete writer");
    let width = keys.len();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("schema to arrow"));
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(keys)) as ArrayRef,
        Arc::new(StringArray::from(vec!["x"; width])) as ArrayRef,
    ])
    .expect("build equality-delete key batch");
    writer
        .write(batch)
        .await
        .expect("write equality-delete batch");
    writer
        .close()
        .await
        .expect("close equality-delete writer")
        .into_iter()
        .next()
        .expect("one equality-delete file")
}

async fn append_data(catalog: &MemoryCatalog, table: &Table, file: DataFile) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file])
        .apply(tx)
        .expect("apply fast append");
    tx.commit(catalog).await.expect("commit fast append")
}

async fn commit_eq_delete(catalog: &MemoryCatalog, table: &Table, file: DataFile) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![file])
        .apply(tx)
        .expect("apply row delta");
    tx.commit(catalog).await.expect("commit row delta")
}

async fn drop_column(catalog: &MemoryCatalog, table: &Table, name: &str) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .update_schema()
        .delete_column(name)
        .apply(tx)
        .expect("apply update schema");
    tx.commit(catalog).await.expect("commit update schema")
}

async fn scan_batches_result(table: &Table, column: &str) -> iceberg::Result<Vec<RecordBatch>> {
    let stream = table
        .scan()
        .select([column])
        .build()?
        .to_arrow()
        .await?;
    stream.try_collect().await
}

async fn scan_longs(table: &Table, column: &str) -> Vec<i64> {
    let batches = scan_batches_result(table, column)
        .await
        .expect("scan to arrow");
    let mut values = Vec::new();
    for batch in &batches {
        let col = batch
            .column_by_name(column)
            .expect("key column")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("key column is Int64");
        for row in 0..col.len() {
            values.push(col.value(row));
        }
    }
    values.sort_unstable();
    values
}

async fn scan_strings(table: &Table, column: &str) -> Vec<String> {
    let batches = scan_batches_result(table, column)
        .await
        .expect("scan to arrow");
    let mut values = Vec::new();
    for batch in &batches {
        let col = batch
            .column_by_name(column)
            .expect("value column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("value column is Utf8");
        for row in 0..col.len() {
            values.push(col.value(row).to_string());
        }
    }
    values.sort();
    values
}

#[tokio::test]
async fn test_avro_equality_delete_with_dropped_key_refuses() {
    let (catalog, table, _warehouse) = setup(two_col_schema("id", "data")).await;
    let data = write_data_file(&table, vec![10, 20, 30, 40, 50], vec!["a", "b", "c", "d", "e"]).await;
    let table = append_data(&catalog, &table, data).await;
    let delete =
        write_eq_delete_file(&table, DataFileFormat::Avro, "eqdel", vec![20, 40], AvroWriterBuilder::new)
            .await;
    assert_eq!(delete.file_format(), DataFileFormat::Avro);
    let delete_path = delete.file_path().to_string();
    assert!(
        delete_path.ends_with(".avro"),
        "delete file {delete_path} keeps the Avro suffix"
    );
    let table = commit_eq_delete(&catalog, &table, delete).await;
    let table = drop_column(&catalog, &table, "id").await;
    match scan_batches_result(&table, "data").await {
        Ok(batches) => {
            let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
            panic!("a dropped equality key must refuse, got {rows} rows");
        }
        Err(err) => {
            assert_eq!(err.kind(), ErrorKind::DataInvalid, "refusal kind: {err:?}");
            assert!(
                err.message().contains(&delete_path),
                "refusal names the delete file: {err:?}"
            );
            assert!(
                err.message().contains("field id 1"),
                "refusal names the missing field id: {err:?}"
            );
        }
    }
}

#[tokio::test]
async fn test_orc_equality_delete_with_dropped_key_refuses() {
    let (catalog, table, _warehouse) = setup(two_col_schema("id", "data")).await;
    let data = write_data_file(&table, vec![10, 20, 30, 40, 50], vec!["a", "b", "c", "d", "e"]).await;
    let table = append_data(&catalog, &table, data).await;
    let delete =
        write_eq_delete_file(&table, DataFileFormat::Orc, "eqdel", vec![20, 40], OrcWriterBuilder::new)
            .await;
    assert_eq!(delete.file_format(), DataFileFormat::Orc);
    let delete_path = delete.file_path().to_string();
    assert!(
        delete_path.ends_with(".orc"),
        "delete file {delete_path} keeps the ORC suffix"
    );
    let table = commit_eq_delete(&catalog, &table, delete).await;
    let table = drop_column(&catalog, &table, "id").await;
    match scan_batches_result(&table, "data").await {
        Ok(batches) => {
            let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
            panic!("a dropped equality key must refuse, got {rows} rows");
        }
        Err(err) => {
            assert_eq!(err.kind(), ErrorKind::DataInvalid, "refusal kind: {err:?}");
            assert!(
                err.message().contains(&delete_path),
                "refusal names the delete file: {err:?}"
            );
            assert!(
                err.message().contains("field id 1"),
                "refusal names the missing field id: {err:?}"
            );
        }
    }
}

#[tokio::test]
async fn test_avro_equality_delete_with_live_key_applies_without_key_projection() {
    let (catalog, table, _warehouse) = setup(two_col_schema("y", "s")).await;
    let data = write_data_file(&table, vec![0, 10, 20, 30, 40], vec!["a", "b", "c", "d", "e"]).await;
    let table = append_data(&catalog, &table, data).await;
    let delete =
        write_eq_delete_file(&table, DataFileFormat::Avro, "eqdel", vec![20, 30], AvroWriterBuilder::new)
            .await;
    assert_eq!(delete.file_format(), DataFileFormat::Avro);
    let table = commit_eq_delete(&catalog, &table, delete).await;
    assert_eq!(
        scan_longs(&table, "y").await,
        vec![0, 10, 40],
        "the Avro delete removes y in {{20, 30}}"
    );
    assert_eq!(
        scan_strings(&table, "s").await,
        vec!["a".to_string(), "b".to_string(), "e".to_string()],
        "the Avro delete applies under a key-less projection"
    );
}

#[tokio::test]
async fn test_orc_equality_delete_with_live_key_applies_without_key_projection() {
    let (catalog, table, _warehouse) = setup(two_col_schema("y", "s")).await;
    let data = write_data_file(&table, vec![0, 10, 20, 30, 40], vec!["a", "b", "c", "d", "e"]).await;
    let table = append_data(&catalog, &table, data).await;
    let delete =
        write_eq_delete_file(&table, DataFileFormat::Orc, "eqdel", vec![20, 30], OrcWriterBuilder::new)
            .await;
    assert_eq!(delete.file_format(), DataFileFormat::Orc);
    let table = commit_eq_delete(&catalog, &table, delete).await;
    assert_eq!(
        scan_longs(&table, "y").await,
        vec![0, 10, 40],
        "the ORC delete removes y in {{20, 30}}"
    );
    assert_eq!(
        scan_strings(&table, "s").await,
        vec!["a".to_string(), "b".to_string(), "e".to_string()],
        "the ORC delete applies under a key-less projection"
    );
}
