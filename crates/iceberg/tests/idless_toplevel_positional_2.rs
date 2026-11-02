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

use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, Int32Array, ListArray, RecordBatch, StringArray, StructArray};
use arrow_buffer::NullBuffer;
use arrow_schema::{Field, Fields, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::expr::{Predicate, Reference};
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, ListType, MapType,
    NestedField, PrimitiveType, Schema, SortOrder, Struct, StructType, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::equality_delete_writer::{
    EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
};
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

fn int_field(id: i32, name: &str) -> NestedField {
    NestedField::optional(id, name, Type::Primitive(PrimitiveType::Int))
}

fn struct_type(fields: Vec<NestedField>) -> Type {
    Type::Struct(StructType::new(fields.into_iter().map(Arc::new).collect()))
}

fn int_list() -> Type {
    Type::List(ListType::new(
        NestedField::list_element(3, Type::Primitive(PrimitiveType::Int), false).into(),
    ))
}

fn schema_of(fields: Vec<NestedField>) -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(fields.into_iter().map(Arc::new).collect::<Vec<_>>())
        .build()
        .expect("build the test schema")
}

async fn new_table(warehouse: &TempDir, schema: Schema) -> (impl Catalog, Table) {
    let warehouse = warehouse.path().to_string_lossy().to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "pins",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("load memory catalog");
    let namespace = NamespaceIdent::new("pins".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("positional_table".to_string())
        .location(format!("{warehouse}/t"))
        .schema(schema)
        .partition_spec(UnboundPartitionSpec::builder().build())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create positional_table");
    (catalog, table)
}

fn ints(values: &[Option<i32>]) -> ArrayRef {
    Arc::new(Int32Array::from(values.to_vec()))
}

fn strings(values: &[Option<&str>]) -> ArrayRef {
    Arc::new(StringArray::from(values.to_vec()))
}

fn plain_struct(children: Vec<(&str, ArrayRef)>) -> ArrayRef {
    let fields: Vec<Field> = children
        .iter()
        .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
        .collect();
    let len = children.first().map_or(0, |(_, array)| array.len());
    let arrays: Vec<ArrayRef> = children.into_iter().map(|(_, array)| array).collect();
    Arc::new(
        StructArray::try_new_with_length(
            Fields::from(fields),
            arrays,
            Some(NullBuffer::new_valid(len)),
            len,
        )
        .expect("build the id-less fixture struct"),
    )
}

fn int_lists(values: Vec<Vec<i32>>) -> ArrayRef {
    Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(
        values
            .into_iter()
            .map(|list| Some(list.into_iter().map(Some).collect::<Vec<_>>())),
    ))
}

fn idless_batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
        .collect();
    let arrays: Vec<ArrayRef> = columns.into_iter().map(|(_, array)| array).collect();
    RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), arrays)
        .expect("build the id-less batch")
}

fn write_idless(table: &Table, batch: &RecordBatch) -> DataFile {
    let dir = format!("{}/data", table.metadata().location());
    std::fs::create_dir_all(&dir).expect("create the data directory");
    let path = format!("{dir}/idless-{}.parquet", uuid::Uuid::now_v7());
    let file = std::fs::File::create(&path).expect("create the id-less parquet file");
    let mut writer =
        ArrowWriter::try_new(file, batch.schema(), None).expect("build the arrow writer");
    writer.write(batch).expect("write the id-less batch");
    writer.close().expect("close the arrow writer");
    let size = std::fs::metadata(&path)
        .expect("stat the parquet file")
        .len();
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path)
        .file_format(DataFileFormat::Parquet)
        .partition(Struct::empty())
        .partition_spec_id(0)
        .record_count(u64::try_from(batch.num_rows()).expect("row count fits u64"))
        .file_size_in_bytes(size)
        .build()
        .expect("build the data file")
}

async fn table_with(
    warehouse: &TempDir,
    schema: Schema,
    batch: RecordBatch,
) -> (impl Catalog, Table) {
    let (catalog, table) = new_table(warehouse, schema).await;
    let file = write_idless(&table, &batch);
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");
    (catalog, table)
}

async fn scan_result(
    table: &Table,
    columns: Option<&[&str]>,
    filter: Option<Predicate>,
) -> iceberg::Result<Vec<RecordBatch>> {
    let mut builder = match columns {
        Some(columns) => table.scan().select(columns.iter().copied()),
        None => table.scan(),
    };
    if let Some(filter) = filter {
        builder = builder.with_filter(filter);
    }
    builder.build()?.to_arrow().await?.try_collect().await
}

fn shown(batches: &[RecordBatch]) -> Vec<String> {
    batches
        .iter()
        .flat_map(|batch| {
            (0..batch.num_rows()).map(move |row| {
                batch
                    .columns()
                    .iter()
                    .map(|column| {
                        if column.is_null(row) {
                            "NULL".to_string()
                        } else {
                            arrow_cast::display::array_value_to_string(column.as_ref(), row)
                                .expect("render the cell")
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            })
        })
        .collect()
}

async fn assert_refuses(
    table: &Table,
    columns: Option<&[&str]>,
    filter: Option<Predicate>,
    needles: &[&str],
) {
    let error = match scan_result(table, columns, filter).await {
        Ok(batches) => panic!("the scan read {:?} instead of refusing", shown(&batches)),
        Err(error) => error,
    };
    let message = error.to_string();
    assert_eq!(error.kind(), ErrorKind::DataInvalid, "{message}");
    for needle in needles {
        assert!(message.contains(needle), "missing {needle:?} in {message}");
    }
}

async fn struct_over_list_table(tmp: &TempDir) -> Table {
    let schema = schema_of(vec![
        int_field(1, "a"),
        NestedField::optional(2, "xs", int_list()),
    ]);
    let batch = idless_batch(vec![
        ("a", ints(&[Some(1), Some(2)])),
        ("deep", plain_struct(vec![("i", ints(&[Some(7), Some(8)]))])),
    ]);
    let (_catalog, table) = table_with(tmp, schema, batch).await;
    table
}

const STRUCT_OVER_LIST: &[&str] = &["'deep'", "position 1", "struct", "'xs'", "id 2", "list"];

#[tokio::test]
async fn a_positional_struct_column_landing_on_a_list_field_refuses() {
    let tmp = TempDir::new().expect("temp dir");
    let table = struct_over_list_table(&tmp).await;
    assert_refuses(&table, None, None, STRUCT_OVER_LIST).await;
}

#[tokio::test]
async fn a_filter_on_a_list_field_fed_by_a_positional_struct_refuses() {
    let tmp = TempDir::new().expect("temp dir");
    let table = struct_over_list_table(&tmp).await;
    let is_null = Reference::new("xs").is_null();
    assert_refuses(&table, Some(&["a"]), Some(is_null), STRUCT_OVER_LIST).await;
    let is_not_null = Reference::new("xs").is_not_null();
    assert_refuses(&table, Some(&["a"]), Some(is_not_null), STRUCT_OVER_LIST).await;
}

async fn list_over_struct_table(tmp: &TempDir) -> Table {
    let schema = schema_of(vec![
        int_field(1, "a"),
        NestedField::optional(2, "s", struct_type(vec![int_field(3, "i")])),
    ]);
    let batch = idless_batch(vec![
        ("a", ints(&[Some(1), Some(2)])),
        ("xs", int_lists(vec![vec![7], vec![8, 9]])),
    ]);
    let (_catalog, table) = table_with(tmp, schema, batch).await;
    table
}

const LIST_OVER_STRUCT: &[&str] = &["'xs'", "position 1", "list", "'s'", "id 2", "struct"];

#[tokio::test]
async fn a_positional_list_column_landing_on_a_struct_field_refuses() {
    let tmp = TempDir::new().expect("temp dir");
    let table = list_over_struct_table(&tmp).await;
    assert_refuses(&table, None, None, LIST_OVER_STRUCT).await;
    let is_null = Reference::new("s").is_null();
    assert_refuses(&table, Some(&["a"]), Some(is_null), LIST_OVER_STRUCT).await;
}

#[tokio::test]
async fn a_positional_struct_column_landing_on_a_map_field_refuses() {
    let tmp = TempDir::new().expect("temp dir");
    let schema = schema_of(vec![
        int_field(1, "a"),
        NestedField::optional(
            2,
            "mp",
            Type::Map(MapType::new(
                NestedField::map_key_element(3, Type::Primitive(PrimitiveType::String)).into(),
                NestedField::map_value_element(4, Type::Primitive(PrimitiveType::Int), false)
                    .into(),
            )),
        ),
    ]);
    let batch = idless_batch(vec![
        ("a", ints(&[Some(1), Some(2)])),
        ("deep", plain_struct(vec![("i", ints(&[Some(7), Some(8)]))])),
    ]);
    let (_catalog, table) = table_with(&tmp, schema, batch).await;
    assert_refuses(&table, None, None, &[
        "'deep'",
        "position 1",
        "struct",
        "'mp'",
        "id 2",
        "map",
    ])
    .await;
}

#[tokio::test]
async fn a_positional_struct_over_a_struct_with_other_children_reads_by_name_control() {
    let tmp = TempDir::new().expect("temp dir");
    let schema = schema_of(vec![
        int_field(1, "a"),
        NestedField::optional(2, "s", struct_type(vec![int_field(3, "i")])),
    ]);
    let batch = idless_batch(vec![
        ("a", ints(&[Some(1), Some(2)])),
        ("t", plain_struct(vec![("j", ints(&[Some(7), Some(8)]))])),
    ]);
    let (_catalog, table) = table_with(&tmp, schema, batch).await;
    let batches = scan_result(&table, None, None)
        .await
        .expect("the scan reads");
    assert_eq!(shown(&batches), vec!["1, {i: }", "2, {i: }"]);
}

async fn write_equality_delete(table: &Table, key: i32) -> DataFile {
    let schema = table.metadata().current_schema().clone();
    let config = EqualityDeleteWriterConfig::new(vec![1], schema.clone())
        .expect("equality-delete writer config");
    let projected = Arc::new(
        iceberg::arrow::arrow_schema_to_schema(config.projected_arrow_schema_ref())
            .expect("the projected arrow schema converts"),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            projected,
        ),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).expect("location generator"),
        DefaultFileNameGenerator::new(
            "eq-del".to_string(),
            Some(uuid::Uuid::now_v7().to_string()),
            DataFileFormat::Parquet,
        ),
    );
    let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .unpartitioned()
        .build(None)
        .await
        .expect("build the equality-delete writer");
    let arrow_schema = Arc::new(schema_to_arrow_schema(&schema).expect("arrow schema"));
    let batch = RecordBatch::try_new(arrow_schema, vec![
        ints(&[Some(key)]),
        strings(&[Some("x")]),
    ])
    .expect("build the equality-delete batch");
    writer.write(batch).await.expect("write the delete batch");
    writer
        .close()
        .await
        .expect("close the equality-delete writer")
        .into_iter()
        .next()
        .expect("one equality-delete file")
}

#[tokio::test]
async fn an_equality_delete_key_fed_by_a_type_changing_positional_column_refuses() {
    let tmp = TempDir::new().expect("temp dir");
    let schema = schema_of(vec![
        int_field(1, "a"),
        NestedField::optional(2, "b", Type::Primitive(PrimitiveType::String)),
    ]);
    let batch = idless_batch(vec![
        ("b", strings(&[Some("p"), Some("q")])),
        ("a", ints(&[Some(1), Some(2)])),
    ]);
    let (catalog, table) = table_with(&tmp, schema, batch).await;
    let delete = write_equality_delete(&table, 1).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete])
        .apply(tx)
        .expect("apply the equality delete");
    let table = tx
        .commit(&catalog)
        .await
        .expect("commit the equality delete");
    assert_refuses(&table, Some(&[]), None, &[
        "'b'",
        "position 0",
        "'a'",
        "id 1",
    ])
    .await;
}
