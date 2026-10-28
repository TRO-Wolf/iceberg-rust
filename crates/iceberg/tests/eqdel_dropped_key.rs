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
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, FormatVersion, NestedField, PrimitiveType, Schema, SchemaRef,
    SortOrder, Struct, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::equality_delete_writer::{
    EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
};
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};

fn gen_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build the {id long, data string} schema")
}

async fn create_rust_table(catalog: &impl Catalog, table_location: &str) -> Table {
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");

    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location.to_string())
        .schema(gen_schema())
        .partition_spec(UnboundPartitionSpec::builder().build())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();

    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create rust_table")
}

fn batch_for_schema(schema: SchemaRef, ids: &[i64], data: &[&str]) -> RecordBatch {
    let arrow_schema =
        Arc::new(schema_to_arrow_schema(&schema).expect("iceberg schema converts to arrow"));
    let mut columns: Vec<ArrayRef> = Vec::new();
    for field in schema.as_struct().fields() {
        if field.name == "id" {
            columns.push(Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef);
        } else if field.name == "data" {
            columns.push(Arc::new(StringArray::from(data.to_vec())) as ArrayRef);
        } else {
            panic!("test fixture covers only id and data columns");
        }
    }
    RecordBatch::try_new(arrow_schema, columns).expect("build the fixture batch")
}

async fn write_data_file(table: &Table, schema: SchemaRef, ids: &[i64], data: &[&str]) -> DataFile {
    let batch = batch_for_schema(schema.clone(), ids, data);
    let file_path = format!(
        "{}/data/{}-rust-data.parquet",
        table.metadata().location(),
        uuid::Uuid::now_v7()
    );
    let output = table
        .file_io()
        .new_output(file_path)
        .expect("new parquet output");
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema,
    );
    let mut writer = parquet_builder
        .build(output)
        .await
        .expect("build parquet writer");
    writer.write(&batch).await.expect("write data batch");
    let data_file_builders = writer.close().await.expect("close parquet writer");

    let mut builder = data_file_builders
        .into_iter()
        .next()
        .expect("one data file builder");
    builder
        .content(DataContentType::Data)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .expect("build unpartitioned data file")
}

async fn write_equality_delete(
    table: &Table,
    schema: SchemaRef,
    equality_id: i32,
    ids: &[i64],
) -> DataFile {
    let config = EqualityDeleteWriterConfig::new(vec![equality_id], schema.clone())
        .expect("equality-delete writer config");

    let location_gen = DefaultLocationGenerator::new(table.metadata()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "eq-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        iceberg::spec::DataFileFormat::Parquet,
    );
    let projected_iceberg_schema = Arc::new(
        iceberg::arrow::arrow_schema_to_schema(config.projected_arrow_schema_ref())
            .expect("projected arrow schema converts to iceberg schema"),
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        projected_iceberg_schema,
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );

    let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .unpartitioned()
        .build(None)
        .await
        .expect("build equality-delete writer");

    let filler: Vec<&str> = std::iter::repeat_n("x", ids.len()).collect();
    let batch = batch_for_schema(schema, ids, &filler);
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

async fn append_5row_table(catalog: &impl Catalog, table_location: &str) -> Table {
    let table = create_rust_table(catalog, table_location).await;
    let schema = table.metadata().current_schema().clone();
    let data_file = write_data_file(&table, schema, &[10, 20, 30, 40, 50], &[
        "a", "b", "c", "d", "e",
    ])
    .await;
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![data_file])
        .apply(tx)
        .expect("apply fast append");
    tx.commit(catalog).await.expect("commit fast append")
}

async fn commit_eq_delete(catalog: &impl Catalog, table: &Table, file: DataFile) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![file])
        .apply(tx)
        .expect("apply deletes");
    tx.commit(catalog).await.expect("commit deletes")
}

async fn scan_strings(table: &Table, column: &str) -> Vec<String> {
    let batches = table
        .scan()
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to arrow")
        .try_collect::<Vec<_>>()
        .await
        .expect("collect scan batches");
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name(column)
                .expect("projected column present")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("string column")
                .iter()
                .map(|value| value.expect("non-null test value").to_owned())
                .collect::<Vec<_>>()
        })
        .collect()
}

async fn scan_row_count(table: &Table) -> usize {
    table
        .scan()
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to arrow")
        .try_collect::<Vec<_>>()
        .await
        .expect("collect scan batches")
        .iter()
        .map(RecordBatch::num_rows)
        .sum()
}

async fn load_catalog(warehouse: &str) -> impl Catalog {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "audit",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_owned())]),
        )
        .await
        .expect("load memory catalog")
}

#[tokio::test]
async fn equality_delete_survives_drop_of_key_column() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let catalog = load_catalog(&warehouse).await;
    let table = append_5row_table(&catalog, &format!("{warehouse}/table")).await;
    let schema = table.metadata().current_schema().clone();
    let file = write_equality_delete(&table, schema, 1, &[20, 40]).await;
    let table = commit_eq_delete(&catalog, &table, file).await;
    assert_eq!(scan_row_count(&table).await, 3);
    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .delete_column("id")
        .apply(tx)
        .expect("apply drop id");
    let table = tx.commit(&catalog).await.expect("commit drop id");
    assert_eq!(scan_strings(&table, "data").await, vec!["a", "c", "e"]);
}

#[tokio::test]
async fn equality_delete_survives_rename_of_key_column() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let catalog = load_catalog(&warehouse).await;
    let table = append_5row_table(&catalog, &format!("{warehouse}/table")).await;
    let schema = table.metadata().current_schema().clone();
    let file = write_equality_delete(&table, schema, 1, &[20, 40]).await;
    let table = commit_eq_delete(&catalog, &table, file).await;
    assert_eq!(scan_row_count(&table).await, 3);
    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .rename_column("id", "identifier")
        .apply(tx)
        .expect("apply rename id");
    let table = tx.commit(&catalog).await.expect("commit rename id");
    assert_eq!(scan_strings(&table, "data").await, vec!["a", "c", "e"]);
}

#[tokio::test]
async fn equality_delete_survives_readd_and_new_delete_scopes_to_new_rows() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let catalog = load_catalog(&warehouse).await;
    let table = append_5row_table(&catalog, &format!("{warehouse}/table")).await;
    let schema = table.metadata().current_schema().clone();
    let old_id = schema
        .field_by_name("id")
        .expect("id present before drop")
        .id;
    let file = write_equality_delete(&table, schema, old_id, &[20, 40]).await;
    let table = commit_eq_delete(&catalog, &table, file).await;
    assert_eq!(scan_row_count(&table).await, 3);

    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .delete_column("id")
        .apply(tx)
        .expect("apply drop id");
    let table = tx.commit(&catalog).await.expect("commit drop id");

    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .add_column("id", Type::Primitive(PrimitiveType::Long))
        .apply(tx)
        .expect("apply re-add id");
    let table = tx.commit(&catalog).await.expect("commit re-add id");
    assert_eq!(scan_strings(&table, "data").await, vec!["a", "c", "e"]);

    let schema = table.metadata().current_schema().clone();
    let new_id = schema
        .field_by_name("id")
        .expect("id present after re-add")
        .id;
    assert_ne!(old_id, new_id);

    let new_file = write_data_file(&table, schema.clone(), &[20, 60], &["f", "g"]).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![new_file])
        .apply(tx)
        .expect("apply append of post-re-add rows");
    let table = tx.commit(&catalog).await.expect("commit post-re-add rows");

    let new_delete = write_equality_delete(&table, schema, new_id, &[60, 10]).await;
    let table = commit_eq_delete(&catalog, &table, new_delete).await;

    let mut values = scan_strings(&table, "data").await;
    values.sort();
    assert_eq!(values, vec!["a", "c", "e", "f"]);
}
