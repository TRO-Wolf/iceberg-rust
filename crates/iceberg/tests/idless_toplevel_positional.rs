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

use arrow_array::{
    Array, ArrayRef, Date32Array, Int32Array, Int64Array, RecordBatch, StringArray, StructArray,
};
use arrow_buffer::NullBuffer;
use arrow_cast::display::array_value_to_string;
use arrow_schema::{Field, Fields, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, NestedField,
    PrimitiveType, Schema, SortOrder, Struct, StructType, TableProperties, Type,
    UnboundPartitionSpec, create_name_mapping,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation};
use parquet::arrow::ArrowWriter;
use serde_json::{Value, json};
use tempfile::TempDir;

fn field(id: i32, name: &str, primitive: PrimitiveType) -> NestedField {
    NestedField::optional(id, name, Type::Primitive(primitive))
}

fn int_field(id: i32, name: &str) -> NestedField {
    field(id, name, PrimitiveType::Int)
}

fn struct_type(fields: Vec<NestedField>) -> Type {
    Type::Struct(StructType::new(fields.into_iter().map(Arc::new).collect()))
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

fn longs(values: &[Option<i64>]) -> ArrayRef {
    Arc::new(Int64Array::from(values.to_vec()))
}

fn strings(values: &[Option<&str>]) -> ArrayRef {
    Arc::new(StringArray::from(values.to_vec()))
}

fn dates(values: &[Option<i32>]) -> ArrayRef {
    Arc::new(Date32Array::from(values.to_vec()))
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

async fn append(catalog: &impl Catalog, table: &Table, batch: &RecordBatch) -> Table {
    let file = write_idless(table, batch);
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file])
        .apply(tx)
        .expect("apply fast append");
    tx.commit(catalog).await.expect("commit fast append")
}

async fn table_with(
    warehouse: &TempDir,
    schema: Schema,
    batch: RecordBatch,
) -> (impl Catalog, Table) {
    let (catalog, table) = new_table(warehouse, schema).await;
    let table = append(&catalog, &table, &batch).await;
    (catalog, table)
}

async fn set_mapping(catalog: &impl Catalog, table: &Table, mapping: &Value) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .update_table_properties()
        .set(
            TableProperties::PROPERTY_DEFAULT_NAME_MAPPING.to_string(),
            mapping.to_string(),
        )
        .apply(tx)
        .expect("apply the name mapping property");
    tx.commit(catalog).await.expect("commit the name mapping")
}

async fn install_mapping(catalog: &impl Catalog, table: &Table) -> Table {
    let mapping = create_name_mapping(table.metadata().current_schema())
        .expect("build the mapping of the current schema");
    let mapping = serde_json::to_value(&mapping).expect("serialize the mapping");
    set_mapping(catalog, table, &mapping).await
}

async fn drop_column(catalog: &impl Catalog, table: &Table, name: &str) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .update_schema()
        .delete_column(name)
        .apply(tx)
        .expect("apply drop column");
    tx.commit(catalog).await.expect("commit drop column")
}

async fn add_int_column(
    catalog: &impl Catalog,
    table: &Table,
    parent: Option<&str>,
    name: &str,
) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .update_schema()
        .add_column_to(parent, name, Type::Primitive(PrimitiveType::Int), None)
        .apply(tx)
        .expect("apply add column");
    tx.commit(catalog).await.expect("commit add column")
}

fn field_id(table: &Table, name: &str) -> i32 {
    table
        .metadata()
        .current_schema()
        .field_by_name(name)
        .unwrap_or_else(|| panic!("field {name} is in the table schema"))
        .id
}

async fn scan_result(table: &Table, columns: Option<&[&str]>) -> iceberg::Result<RecordBatch> {
    let builder = match columns {
        Some(columns) => table.scan().select(columns.iter().copied()),
        None => table.scan(),
    };
    let batches = builder
        .build()?
        .to_arrow()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let schema = batches.first().expect("at least one batch").schema();
    Ok(arrow_select::concat::concat_batches(&schema, &batches).expect("concat scan batches"))
}

async fn scan(table: &Table, columns: Option<&[&str]>) -> RecordBatch {
    scan_result(table, columns).await.expect("the scan reads")
}

fn cell(array: &dyn Array, row: usize) -> Option<String> {
    if array.is_null(row) {
        return None;
    }
    Some(array_value_to_string(array, row).expect("render the cell"))
}

fn rows(batch: &RecordBatch) -> Vec<Vec<Option<String>>> {
    (0..batch.num_rows())
        .map(|row| {
            batch
                .columns()
                .iter()
                .map(|column| cell(column.as_ref(), row))
                .collect()
        })
        .collect()
}

fn expected(values: &[&[Option<&str>]]) -> Vec<Vec<Option<String>>> {
    values
        .iter()
        .map(|row| row.iter().map(|cell| cell.map(str::to_string)).collect())
        .collect()
}

fn column_names(batch: &RecordBatch) -> Vec<String> {
    batch
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect()
}

async fn assert_refuses(table: &Table, columns: Option<&[&str]>, needles: &[&str]) {
    let message = match scan_result(table, columns).await {
        Ok(batch) => panic!(
            "the scan read {:?} {:?} instead of refusing",
            column_names(&batch),
            rows(&batch)
        ),
        Err(error) => {
            let message = error.to_string();
            assert_eq!(error.kind(), ErrorKind::DataInvalid, "{message}");
            message
        }
    };
    for needle in needles {
        assert!(message.contains(needle), "missing {needle:?} in {message}");
    }
    for forbidden in ["delete", "drop", "remove", "purge", "truncate"] {
        assert!(
            !message.to_lowercase().contains(forbidden),
            "the refusal names {forbidden:?}: {message}"
        );
    }
}

fn int_string_table() -> Schema {
    schema_of(vec![
        int_field(1, "a"),
        field(2, "b", PrimitiveType::String),
    ])
}

#[tokio::test]
async fn a_positional_string_column_landing_on_an_int_field_refuses() {
    let tmp = TempDir::new().expect("temp dir");
    let batch = idless_batch(vec![
        ("b", strings(&[Some("p"), Some("q")])),
        ("a", ints(&[Some(1), Some(2)])),
    ]);
    let (_catalog, table) = table_with(&tmp, int_string_table(), batch).await;
    assert_refuses(&table, None, &[
        "'b'",
        "position 0",
        "string",
        "'a'",
        "id 1",
        "int",
        "no field ids",
        "name mapping",
    ])
    .await;
}

#[tokio::test]
async fn a_positional_read_that_casts_into_a_wrong_value_refuses() {
    let tmp = TempDir::new().expect("temp dir");
    let batch = idless_batch(vec![
        ("b", strings(&[Some("7"), Some("8")])),
        ("a", ints(&[Some(1), Some(2)])),
    ]);
    let (_catalog, table) = table_with(&tmp, int_string_table(), batch).await;
    assert_refuses(&table, None, &["'b'", "position 0", "'a'", "id 1"]).await;
    assert_refuses(&table, Some(&["b"]), &["'a'", "position 1", "'b'", "id 2"]).await;
}

fn int_long_table() -> Schema {
    schema_of(vec![int_field(1, "a"), field(2, "b", PrimitiveType::Long)])
}

fn long_int_file() -> RecordBatch {
    idless_batch(vec![
        ("b", longs(&[Some(40), Some(30)])),
        ("a", ints(&[Some(1), Some(2)])),
    ])
}

#[tokio::test]
async fn a_positional_long_column_landing_on_an_int_field_refuses() {
    let tmp = TempDir::new().expect("temp dir");
    let (_catalog, table) = table_with(&tmp, int_long_table(), long_int_file()).await;
    assert_refuses(&table, None, &[
        "'b'",
        "position 0",
        "long",
        "'a'",
        "id 1",
        "int",
    ])
    .await;
}

#[tokio::test]
async fn a_positional_int_column_landing_on_a_long_field_reads_promoted() {
    let tmp = TempDir::new().expect("temp dir");
    let (_catalog, table) = table_with(&tmp, int_long_table(), long_int_file()).await;
    let batch = scan(&table, Some(&["b"])).await;
    assert_eq!(column_names(&batch), vec!["b".to_string()]);
    let b = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("b reads as a long column");
    assert_eq!(b.iter().collect::<Vec<_>>(), vec![Some(1), Some(2)]);
}

#[tokio::test]
async fn a_positional_date_column_landing_on_an_int_field_refuses() {
    let tmp = TempDir::new().expect("temp dir");
    let schema = schema_of(vec![int_field(1, "a"), field(2, "b", PrimitiveType::Date)]);
    let batch = idless_batch(vec![("b", dates(&[Some(18263)])), ("a", ints(&[Some(1)]))]);
    let (_catalog, table) = table_with(&tmp, schema, batch).await;
    assert_refuses(&table, None, &[
        "'b'",
        "position 0",
        "date",
        "'a'",
        "id 1",
        "int",
    ])
    .await;
}

#[tokio::test]
async fn a_positional_struct_column_landing_on_an_int_field_refuses() {
    let tmp = TempDir::new().expect("temp dir");
    let schema = schema_of(vec![
        int_field(1, "a"),
        NestedField::optional(2, "s", struct_type(vec![int_field(3, "d")])),
    ]);
    let batch = idless_batch(vec![
        ("s", plain_struct(vec![("d", ints(&[Some(9)]))])),
        ("a", ints(&[Some(1)])),
    ]);
    let (_catalog, table) = table_with(&tmp, schema, batch).await;
    assert_refuses(&table, None, &["'s'", "position 0", "'a'", "id 1", "int"]).await;
}

async fn re_added_top_level_table(tmp: &TempDir, mapped: bool) -> Table {
    let schema = schema_of(vec![int_field(1, "id"), int_field(2, "c")]);
    let batch = idless_batch(vec![("id", ints(&[Some(1)])), ("c", ints(&[Some(5)]))]);
    let (catalog, table) = table_with(tmp, schema, batch).await;
    let table = if mapped {
        install_mapping(&catalog, &table).await
    } else {
        table
    };
    let old_c = field_id(&table, "c");
    let table = drop_column(&catalog, &table, "c").await;
    let table = add_int_column(&catalog, &table, None, "c").await;
    assert_ne!(field_id(&table, "c"), old_c);
    table
}

#[tokio::test]
async fn a_re_added_top_level_name_reads_null_on_the_no_mapping_route() {
    let tmp = TempDir::new().expect("temp dir");
    let table = re_added_top_level_table(&tmp, false).await;
    let batch = scan(&table, None).await;
    assert_eq!(column_names(&batch), vec![
        "id".to_string(),
        "c".to_string()
    ]);
    assert_eq!(rows(&batch), expected(&[&[Some("1"), None]]));
}

#[tokio::test]
async fn a_re_added_top_level_name_reads_its_value_on_the_mapping_route() {
    let tmp = TempDir::new().expect("temp dir");
    let table = re_added_top_level_table(&tmp, true).await;
    let batch = scan(&table, None).await;
    assert_eq!(column_names(&batch), vec![
        "id".to_string(),
        "c".to_string()
    ]);
    assert_eq!(rows(&batch), expected(&[&[Some("1"), Some("5")]]));
}

#[tokio::test]
async fn an_unmapped_top_level_column_under_a_partial_mapping_reads_null() {
    let tmp = TempDir::new().expect("temp dir");
    let schema = schema_of(vec![int_field(1, "a"), int_field(2, "c")]);
    let batch = idless_batch(vec![("a", ints(&[Some(1)])), ("c", ints(&[Some(2)]))]);
    let (catalog, table) = table_with(&tmp, schema, batch).await;
    let table = set_mapping(&catalog, &table, &json!([{"field-id": 9, "names": ["zz"]}])).await;
    let batch = scan(&table, None).await;
    assert_eq!(column_names(&batch), vec!["a".to_string(), "c".to_string()]);
    assert_eq!(rows(&batch), expected(&[&[None, None]]));
}

#[tokio::test]
async fn a_file_with_fewer_columns_reads_by_position_control() {
    let tmp = TempDir::new().expect("temp dir");
    let schema = schema_of(vec![
        int_field(1, "x"),
        int_field(2, "a"),
        int_field(3, "b"),
    ]);
    let batch = idless_batch(vec![
        ("a", ints(&[Some(1), Some(2)])),
        ("b", ints(&[Some(40), Some(30)])),
    ]);
    let (_catalog, table) = table_with(&tmp, schema, batch).await;
    let batch = scan(&table, None).await;
    assert_eq!(
        rows(&batch),
        expected(&[&[Some("1"), Some("40"), None], &[
            Some("2"),
            Some("30"),
            None
        ]])
    );
}

#[tokio::test]
async fn a_same_type_swap_reads_by_position_control() {
    let tmp = TempDir::new().expect("temp dir");
    let schema = schema_of(vec![int_field(1, "a"), int_field(2, "b")]);
    let batch = idless_batch(vec![
        ("b", ints(&[Some(10), Some(20)])),
        ("a", ints(&[Some(1), Some(2)])),
    ]);
    let (_catalog, table) = table_with(&tmp, schema, batch).await;
    let batch = scan(&table, None).await;
    assert_eq!(column_names(&batch), vec!["a".to_string(), "b".to_string()]);
    assert_eq!(
        rows(&batch),
        expected(&[&[Some("10"), Some("1")], &[Some("20"), Some("2")]])
    );
}

fn struct_ints(batch: &RecordBatch, column: &str, child: &str) -> Vec<Option<i32>> {
    batch
        .column_by_name(column)
        .unwrap_or_else(|| panic!("column {column} present"))
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("a struct column")
        .column_by_name(child)
        .unwrap_or_else(|| panic!("child {child} present"))
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("an int child")
        .iter()
        .collect()
}

#[tokio::test]
async fn nested_children_read_by_name_on_the_no_mapping_route_control() {
    let tmp = TempDir::new().expect("temp dir");
    let schema = schema_of(vec![
        int_field(1, "id"),
        NestedField::optional(
            2,
            "s",
            struct_type(vec![int_field(3, "d"), int_field(4, "e")]),
        ),
    ]);
    let batch = idless_batch(vec![
        ("id", ints(&[Some(1), Some(2)])),
        (
            "s",
            plain_struct(vec![
                ("d", ints(&[Some(10), Some(20)])),
                ("e", ints(&[Some(30), Some(40)])),
            ]),
        ),
    ]);
    let (_catalog, table) = table_with(&tmp, schema, batch).await;
    let batch = scan(&table, None).await;
    assert_eq!(struct_ints(&batch, "s", "d"), vec![Some(10), Some(20)]);
    assert_eq!(struct_ints(&batch, "s", "e"), vec![Some(30), Some(40)]);
}

#[tokio::test]
async fn a_re_added_nested_child_keeps_its_old_value_on_the_no_mapping_route_control() {
    let tmp = TempDir::new().expect("temp dir");
    let schema = schema_of(vec![
        int_field(1, "id"),
        NestedField::optional(
            2,
            "s",
            struct_type(vec![int_field(3, "a"), int_field(4, "b")]),
        ),
    ]);
    let batch = idless_batch(vec![
        ("id", ints(&[Some(1)])),
        (
            "s",
            plain_struct(vec![("a", ints(&[Some(1)])), ("b", ints(&[Some(2)]))]),
        ),
    ]);
    let (catalog, table) = table_with(&tmp, schema, batch).await;
    let old_b = field_id(&table, "s.b");
    let table = drop_column(&catalog, &table, "s.b").await;
    let table = add_int_column(&catalog, &table, Some("s"), "b").await;
    assert_ne!(field_id(&table, "s.b"), old_b);
    let batch = scan(&table, None).await;
    assert_eq!(struct_ints(&batch, "s", "a"), vec![Some(1)]);
    assert_eq!(struct_ints(&batch, "s", "b"), vec![Some(2)]);
}
