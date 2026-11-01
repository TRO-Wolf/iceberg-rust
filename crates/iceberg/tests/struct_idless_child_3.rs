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

use arrow_array::builder::{Int32Builder, MapBuilder};
use arrow_array::{Array, ArrayRef, Int32Array, ListArray, MapArray, RecordBatch, StructArray};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, ListType, MapType,
    NestedField, PrimitiveType, Schema, SortOrder, Struct, StructType, TableProperties, Type,
    UnboundPartitionSpec, create_name_mapping,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation};
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use serde_json::{Value, json};
use tempfile::TempDir;

#[derive(Clone, Copy, Debug)]
enum Route {
    Fallback,
    Mapping,
}

#[derive(Clone, Copy, Debug)]
enum FileIds {
    None,
    TopLevelOnly,
}

fn int_field(id: i32, name: &str) -> NestedField {
    NestedField::optional(id, name, Type::Primitive(PrimitiveType::Int))
}

fn struct_type(fields: Vec<NestedField>) -> Type {
    Type::Struct(StructType::new(fields.into_iter().map(Arc::new).collect()))
}

fn with_id(column: NestedField) -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Int),
            )),
            Arc::new(column),
        ])
        .build()
        .expect("build the test schema")
}

async fn new_table(warehouse: &str, schema: Schema) -> (impl Catalog, Table) {
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "pins",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_owned())]),
        )
        .await
        .expect("load memory catalog");
    let namespace = NamespaceIdent::new("pins".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("idless_table".to_string())
        .location(format!("{warehouse}/t"))
        .schema(schema)
        .partition_spec(UnboundPartitionSpec::builder().build())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create idless_table");
    (catalog, table)
}

fn ints(values: &[Option<i32>]) -> ArrayRef {
    Arc::new(Int32Array::from(values.to_vec()))
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
        .expect("build the fixture struct"),
    )
}

fn list_of(values: ArrayRef, offsets: Vec<i32>) -> ArrayRef {
    let item = Arc::new(Field::new("item", values.data_type().clone(), true));
    Arc::new(
        ListArray::try_new(item, OffsetBuffer::new(offsets.into()), values, None)
            .expect("build the fixture list"),
    )
}

fn int_map(rows: &[&[(i32, i32)]]) -> ArrayRef {
    let mut builder = MapBuilder::new(None, Int32Builder::new(), Int32Builder::new());
    for row in rows {
        for (key, value) in row.iter() {
            builder.keys().append_value(*key);
            builder.values().append_value(*value);
        }
        builder.append(true).expect("append a map row");
    }
    Arc::new(builder.finish())
}

fn with_field_id(field: Field, id: i32) -> Field {
    field.with_metadata(HashMap::from([(
        PARQUET_FIELD_ID_META_KEY.to_string(),
        id.to_string(),
    )]))
}

fn write_file(table: &Table, column: (&str, ArrayRef), ids: FileIds) -> DataFile {
    let (name, array) = column;
    let rows = array.len();
    let id_field = Field::new("id", DataType::Int32, false);
    let column_field = Field::new(name, array.data_type().clone(), true);
    let fields = match ids {
        FileIds::None => vec![id_field, column_field],
        FileIds::TopLevelOnly => vec![with_field_id(id_field, 1), with_field_id(column_field, 2)],
    };
    let schema = Arc::new(ArrowSchema::new(fields));
    let id_values: Vec<Option<i32>> = (1..=rows as i32).map(Some).collect();
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![ints(&id_values), array])
        .expect("build the file batch");
    let dir = format!("{}/data", table.metadata().location());
    std::fs::create_dir_all(&dir).expect("create the data directory");
    let path = format!("{dir}/file-{}.parquet", uuid::Uuid::now_v7());
    let file = std::fs::File::create(&path).expect("create the parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("build the writer");
    writer.write(&batch).expect("write the batch");
    writer.close().expect("close the arrow writer");
    let size = std::fs::metadata(&path).expect("stat the file").len();
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path)
        .file_format(DataFileFormat::Parquet)
        .partition(Struct::empty())
        .partition_spec_id(0)
        .record_count(rows as u64)
        .file_size_in_bytes(size)
        .build()
        .expect("build the data file")
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

async fn table_with(
    schema: Schema,
    column: (&str, ArrayRef),
    ids: FileIds,
    mapping: Option<Value>,
) -> (TempDir, Table) {
    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, table) = new_table(&warehouse, schema).await;
    let file = write_file(&table, column, ids);
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");
    let table = match mapping {
        Some(mapping) => set_mapping(&catalog, &table, &mapping).await,
        None => table,
    };
    (tmp, table)
}

fn mapping_of(schema: &Schema) -> Value {
    serde_json::to_value(create_name_mapping(schema).expect("build mapping"))
        .expect("serialize the mapping")
}

fn route_mapping(route: Route, schema: &Schema) -> Option<Value> {
    match route {
        Route::Fallback => None,
        Route::Mapping => Some(mapping_of(schema)),
    }
}

async fn scan_result(table: &Table) -> iceberg::Result<RecordBatch> {
    let batches = table
        .scan()
        .build()?
        .to_arrow()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let schema = batches.first().expect("at least one batch").schema();
    Ok(arrow_select::concat::concat_batches(&schema, &batches).expect("concat scan batches"))
}

async fn scan(table: &Table) -> RecordBatch {
    scan_result(table).await.expect("the scan reads")
}

async fn assert_refuses(table: &Table, needles: &[&str], forbidden: &[&str]) {
    let error = scan_result(table).await.expect_err("the scan refuses");
    let message = error.to_string();
    assert_eq!(error.kind(), ErrorKind::DataInvalid, "{message}");
    for needle in needles {
        assert!(message.contains(needle), "missing {needle:?} in {message}");
    }
    for word in forbidden {
        assert!(!message.contains(word), "unexpected {word:?} in {message}");
    }
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> &'a dyn Array {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("column {name} present"))
        .as_ref()
}

fn as_struct(array: &dyn Array) -> &StructArray {
    array
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("column is a StructArray")
}

fn child_ints(parent: &StructArray, name: &str) -> Vec<Option<i32>> {
    parent
        .column_by_name(name)
        .unwrap_or_else(|| panic!("child {name} present"))
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("child is an int column")
        .iter()
        .collect()
}

fn child_names(parent: &StructArray) -> Vec<String> {
    parent.fields().iter().map(|f| f.name().clone()).collect()
}

fn validity(array: &dyn Array) -> Vec<bool> {
    (0..array.len()).map(|row| array.is_valid(row)).collect()
}

fn list_rows(array: &dyn Array) -> Vec<Vec<Option<i32>>> {
    let list = array
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("column is a ListArray");
    (0..list.len())
        .map(|row| {
            let values = list.value(row);
            let values = values
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("list element is an int column");
            values.iter().collect()
        })
        .collect()
}

fn map_rows(array: &dyn Array) -> Vec<Vec<(i32, Option<i32>)>> {
    let map = array
        .as_any()
        .downcast_ref::<MapArray>()
        .expect("column is a MapArray");
    (0..map.len())
        .map(|row| {
            let entries = map.value(row);
            let keys = entries.column(0).as_any().downcast_ref::<Int32Array>();
            let values = entries.column(1).as_any().downcast_ref::<Int32Array>();
            let (keys, values) = (keys.expect("int keys"), values.expect("int values"));
            (0..entries.len())
                .map(|entry| {
                    (
                        keys.value(entry),
                        values.is_valid(entry).then(|| values.value(entry)),
                    )
                })
                .collect()
        })
        .collect()
}

fn d_e_table() -> Schema {
    with_id(NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "d"), int_field(4, "e")]),
    ))
}

fn d_e_file() -> (&'static str, ArrayRef) {
    (
        "s",
        plain_struct(vec![("d", ints(&[Some(10)])), ("e", ints(&[Some(20)]))]),
    )
}

fn a_b_table() -> Schema {
    with_id(NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "a"), int_field(4, "b")]),
    ))
}

fn b_a_file() -> (&'static str, ArrayRef) {
    (
        "s",
        plain_struct(vec![("b", ints(&[Some(10)])), ("a", ints(&[Some(20)]))]),
    )
}

fn d_table() -> Schema {
    with_id(NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "d")]),
    ))
}

fn d_x_file() -> (&'static str, ArrayRef) {
    (
        "s",
        plain_struct(vec![("d", ints(&[Some(1)])), ("x", ints(&[Some(99)]))]),
    )
}

#[tokio::test]
async fn top_level_ids_only_file_with_same_named_children_refuses() {
    let (_tmp, table) = table_with(d_e_table(), d_e_file(), FileIds::TopLevelOnly, None).await;
    assert_refuses(&table, &["'s'", "no field ids"], &["mapping"]).await;
}

#[tokio::test]
async fn top_level_ids_only_file_with_swapped_children_refuses() {
    let (_tmp, table) = table_with(a_b_table(), b_a_file(), FileIds::TopLevelOnly, None).await;
    assert_refuses(&table, &["'s'", "no field ids"], &["mapping"]).await;
}

#[tokio::test]
async fn top_level_ids_only_file_with_an_extra_child_refuses() {
    let (_tmp, table) = table_with(d_table(), d_x_file(), FileIds::TopLevelOnly, None).await;
    assert_refuses(&table, &["'s'", "no field ids"], &["mapping"]).await;
}

async fn idless_same_named_children(route: Route) {
    let schema = d_e_table();
    let mapping = route_mapping(route, &schema);
    let (_tmp, table) = table_with(schema, d_e_file(), FileIds::None, mapping).await;
    let batch = scan(&table).await;
    let s = as_struct(column(&batch, "s"));
    assert_eq!(validity(s), vec![true]);
    assert_eq!(child_ints(s, "d"), vec![Some(10)]);
    assert_eq!(child_ints(s, "e"), vec![Some(20)]);
}

#[tokio::test]
async fn idless_file_with_same_named_children_reads_its_values_fallback() {
    idless_same_named_children(Route::Fallback).await;
}

#[tokio::test]
async fn idless_file_with_same_named_children_reads_its_values_mapping() {
    idless_same_named_children(Route::Mapping).await;
}

async fn idless_swapped_children(route: Route) {
    let schema = a_b_table();
    let mapping = route_mapping(route, &schema);
    let (_tmp, table) = table_with(schema, b_a_file(), FileIds::None, mapping).await;
    let batch = scan(&table).await;
    let s = as_struct(column(&batch, "s"));
    assert_eq!(child_names(s), vec!["a", "b"]);
    assert_eq!(child_ints(s, "a"), vec![Some(20)]);
    assert_eq!(child_ints(s, "b"), vec![Some(10)]);
}

#[tokio::test]
async fn idless_file_with_swapped_children_reads_by_name_fallback() {
    idless_swapped_children(Route::Fallback).await;
}

#[tokio::test]
async fn idless_file_with_swapped_children_reads_by_name_mapping() {
    idless_swapped_children(Route::Mapping).await;
}

async fn idless_extra_child(route: Route) {
    let schema = d_table();
    let mapping = route_mapping(route, &schema);
    let (_tmp, table) = table_with(schema, d_x_file(), FileIds::None, mapping).await;
    let batch = scan(&table).await;
    let s = as_struct(column(&batch, "s"));
    assert_eq!(child_names(s), vec!["d"]);
    assert_eq!(child_ints(s, "d"), vec![Some(1)]);
}

#[tokio::test]
async fn idless_file_with_an_extra_child_ignores_it_fallback() {
    idless_extra_child(Route::Fallback).await;
}

#[tokio::test]
async fn idless_file_with_an_extra_child_ignores_it_mapping() {
    idless_extra_child(Route::Mapping).await;
}

fn int_list_table() -> Schema {
    with_id(NestedField::optional(
        2,
        "l",
        Type::List(ListType::new(Arc::new(NestedField::list_element(
            3,
            Type::Primitive(PrimitiveType::Int),
            false,
        )))),
    ))
}

fn int_list_file() -> (&'static str, ArrayRef) {
    (
        "l",
        list_of(ints(&[Some(1), Some(2), Some(3)]), vec![0, 2, 3]),
    )
}

fn int_map_table() -> Schema {
    with_id(NestedField::optional(
        2,
        "m",
        Type::Map(MapType::new(
            Arc::new(NestedField::map_key_element(
                3,
                Type::Primitive(PrimitiveType::Int),
            )),
            Arc::new(NestedField::map_value_element(
                4,
                Type::Primitive(PrimitiveType::Int),
                false,
            )),
        )),
    ))
}

fn int_map_file() -> (&'static str, ArrayRef) {
    ("m", int_map(&[&[(1, 10), (2, 20)], &[(3, 30)]]))
}

fn top_mapping(column: Value) -> Value {
    json!([{"field-id": 1, "names": ["id"]}, column])
}

#[tokio::test]
async fn mapping_naming_a_list_without_its_element_entry_refuses() {
    let mapping = top_mapping(json!({"field-id": 2, "names": ["l"]}));
    let (_tmp, table) = table_with(
        int_list_table(),
        int_list_file(),
        FileIds::None,
        Some(mapping),
    )
    .await;
    assert_refuses(&table, &["'l'", "'element'"], &[]).await;
}

#[tokio::test]
async fn mapping_naming_a_map_without_its_key_and_value_entries_refuses() {
    let mapping = top_mapping(json!({"field-id": 2, "names": ["m"]}));
    let (_tmp, table) = table_with(
        int_map_table(),
        int_map_file(),
        FileIds::None,
        Some(mapping),
    )
    .await;
    assert_refuses(&table, &["'m'", "'key'"], &[]).await;
}

#[tokio::test]
async fn mapping_naming_a_map_with_its_key_entry_only_refuses() {
    let mapping = top_mapping(json!({
        "field-id": 2,
        "names": ["m"],
        "fields": [{"field-id": 3, "names": ["key"]}],
    }));
    let (_tmp, table) = table_with(
        int_map_table(),
        int_map_file(),
        FileIds::None,
        Some(mapping),
    )
    .await;
    assert_refuses(&table, &["'m'", "'value'"], &[]).await;
}

#[tokio::test]
async fn fully_mapped_list_and_map_read_their_values() {
    let schema = int_list_table();
    let mapping = mapping_of(&schema);
    let (_tmp, table) = table_with(schema, int_list_file(), FileIds::None, Some(mapping)).await;
    let batch = scan(&table).await;
    assert_eq!(list_rows(column(&batch, "l")), vec![
        vec![Some(1), Some(2)],
        vec![Some(3)]
    ]);
    let schema = int_map_table();
    let mapping = mapping_of(&schema);
    let (_tmp, table) = table_with(schema, int_map_file(), FileIds::None, Some(mapping)).await;
    let batch = scan(&table).await;
    assert_eq!(map_rows(column(&batch, "m")), vec![
        vec![(1, Some(10)), (2, Some(20))],
        vec![(3, Some(30))]
    ]);
}

#[tokio::test]
async fn list_of_structs_mapping_naming_the_element_and_one_child_reads_null_for_the_rest() {
    let schema = with_id(NestedField::optional(
        2,
        "l",
        Type::List(ListType::new(Arc::new(NestedField::list_element(
            3,
            struct_type(vec![int_field(4, "a"), int_field(5, "b")]),
            false,
        )))),
    ));
    let element = plain_struct(vec![
        ("a", ints(&[Some(1), Some(2), Some(3)])),
        ("b", ints(&[Some(10), Some(20), Some(30)])),
    ]);
    let mapping = top_mapping(json!({
        "field-id": 2,
        "names": ["l"],
        "fields": [{
            "field-id": 3,
            "names": ["element"],
            "fields": [{"field-id": 4, "names": ["a"]}],
        }],
    }));
    let file = ("l", list_of(element, vec![0, 2, 3]));
    let (_tmp, table) = table_with(schema, file, FileIds::None, Some(mapping)).await;
    let batch = scan(&table).await;
    let list = column(&batch, "l")
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("column is a ListArray");
    assert_eq!(validity(list), vec![true, true]);
    let elements = as_struct(list.values().as_ref());
    assert_eq!(child_ints(elements, "a"), vec![Some(1), Some(2), Some(3)]);
    assert_eq!(child_ints(elements, "b"), vec![None, None, None]);
}
