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

use arrow_array::{Array, ArrayRef, Int32Array, ListArray, RecordBatch, StructArray};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, ListType,
    NestedField, PrimitiveType, Schema, SortOrder, Struct, StructType, TableProperties, Type,
    UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use parquet::arrow::ArrowWriter;
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug)]
enum Route {
    Fallback,
    Mapping,
}

fn int_field(id: i32, name: &str) -> NestedField {
    NestedField::optional(id, name, Type::Primitive(PrimitiveType::Int))
}

fn struct_type(fields: Vec<NestedField>) -> Type {
    Type::Struct(StructType::new(fields.into_iter().map(Arc::new).collect()))
}

fn table_schema(column: NestedField) -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            column.into(),
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

fn plain_struct(children: Vec<(&str, ArrayRef)>, valid: &[bool]) -> ArrayRef {
    let fields: Vec<Field> = children
        .iter()
        .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
        .collect();
    let arrays: Vec<ArrayRef> = children.into_iter().map(|(_, array)| array).collect();
    Arc::new(
        StructArray::try_new_with_length(
            Fields::from(fields),
            arrays,
            Some(NullBuffer::from(valid.to_vec())),
            valid.len(),
        )
        .expect("build the id-less fixture struct"),
    )
}

async fn write_idless(table: &Table, ids: &[i32], name: &str, column: ArrayRef) -> DataFile {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(name, column.data_type().clone(), true),
    ]));
    let id_values: Vec<Option<i32>> = ids.iter().copied().map(Some).collect();
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![ints(&id_values), column])
        .expect("build the id-less fixture batch");
    let dir = format!("{}/data", table.metadata().location());
    std::fs::create_dir_all(&dir).expect("create the data directory");
    let path = format!("{dir}/idless-{}.parquet", uuid::Uuid::now_v7());
    let file = std::fs::File::create(&path).expect("create the id-less parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("build the arrow writer");
    writer.write(&batch).expect("write the id-less batch");
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
        .record_count(ids.len() as u64)
        .file_size_in_bytes(size)
        .build()
        .expect("build the data file")
}

async fn append(catalog: &impl Catalog, table: &Table, file: DataFile) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file])
        .apply(tx)
        .expect("apply fast append");
    tx.commit(catalog).await.expect("commit fast append")
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

async fn add_child(catalog: &impl Catalog, table: &Table, parent: &str, name: &str) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .update_schema()
        .add_column_to(
            Some(parent),
            name,
            Type::Primitive(PrimitiveType::Int),
            None,
        )
        .apply(tx)
        .expect("apply add nested column");
    tx.commit(catalog).await.expect("commit add nested column")
}

async fn set_mapping(catalog: &impl Catalog, table: &Table, mapping: Value) -> Table {
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

async fn with_route(catalog: &impl Catalog, table: Table, route: Route, mapping: Value) -> Table {
    match route {
        Route::Fallback => table,
        Route::Mapping => set_mapping(catalog, &table, mapping).await,
    }
}

fn mapped(id: i32, name: &str, fields: Vec<Value>) -> Value {
    if fields.is_empty() {
        json!({"field-id": id, "names": [name]})
    } else {
        json!({"field-id": id, "names": [name], "fields": fields})
    }
}

fn field_id(table: &Table, name: &str) -> i32 {
    table
        .metadata()
        .current_schema()
        .field_by_name(name)
        .unwrap_or_else(|| panic!("field {name} is in the table schema"))
        .id
}

async fn scan_batches(table: &Table) -> Vec<RecordBatch> {
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
}

async fn scan(table: &Table) -> RecordBatch {
    let batches = scan_batches(table).await;
    let schema = batches.first().expect("at least one batch").schema();
    for batch in &batches {
        for (column, field) in batch.columns().iter().zip(batch.schema().fields()) {
            assert_eq!(
                without_metadata(column.data_type()),
                without_metadata(field.data_type()),
                "batch column type equals its schema field type for {}",
                field.name()
            );
        }
    }
    arrow_select::concat::concat_batches(&schema, &batches).expect("concat scan batches")
}

fn without_metadata(data_type: &DataType) -> DataType {
    let strip = |field: &Arc<Field>| {
        Arc::new(Field::new(
            field.name(),
            without_metadata(field.data_type()),
            field.is_nullable(),
        ))
    };
    match data_type {
        DataType::Struct(fields) => DataType::Struct(fields.iter().map(strip).collect()),
        DataType::List(element) => DataType::List(strip(element)),
        DataType::Map(entries, sorted) => DataType::Map(strip(entries), *sorted),
        other => other.clone(),
    }
}

fn ids_of(batch: &RecordBatch) -> Vec<i32> {
    batch
        .column_by_name("id")
        .expect("id column present")
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("id is an int column")
        .iter()
        .map(|id| id.expect("id is never null"))
        .collect()
}

fn validity(array: &dyn Array) -> Vec<bool> {
    (0..array.len()).map(|row| array.is_valid(row)).collect()
}

fn as_struct(array: &dyn Array) -> &StructArray {
    array
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("column is a StructArray")
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> &'a StructArray {
    as_struct(
        batch
            .column_by_name(name)
            .unwrap_or_else(|| panic!("column {name} present"))
            .as_ref(),
    )
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

async fn two_leaf_dropped_for_new_child(route: Route) {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let schema = table_schema(NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "d"), int_field(4, "e")]),
    ));
    let (catalog, table) = new_table(&warehouse, schema).await;
    let (d, e) = (field_id(&table, "s.d"), field_id(&table, "s.e"));
    let s = plain_struct(
        vec![("d", ints(&[Some(1), None])), ("e", ints(&[Some(2), None]))],
        &[true, false],
    );
    let file = write_idless(&table, &[1, 2], "s", s).await;
    let table = append(&catalog, &table, file).await;
    let table = add_child(&catalog, &table, "s", "k").await;
    let k = field_id(&table, "s.k");
    let table = drop_column(&catalog, &table, "s.d").await;
    let table = drop_column(&catalog, &table, "s.e").await;
    let mapping = json!([
        mapped(1, "id", vec![]),
        mapped(2, "s", vec![
            mapped(d, "d", vec![]),
            mapped(e, "e", vec![]),
            mapped(k, "k", vec![])
        ]),
    ]);
    let table = with_route(&catalog, table, route, mapping).await;

    let batch = scan(&table).await;
    assert_eq!(ids_of(&batch), vec![1, 2]);
    let s = column(&batch, "s");
    assert_eq!(child_names(s), vec!["k".to_string()]);
    assert_eq!(validity(s), vec![true, false]);
    assert_eq!(child_ints(s, "k"), vec![None, None]);
}

#[tokio::test]
async fn idless_two_leaf_struct_dropped_children_new_child_reads_null_fallback() {
    two_leaf_dropped_for_new_child(Route::Fallback).await;
}

#[tokio::test]
async fn idless_two_leaf_struct_dropped_children_new_child_reads_null_mapping() {
    two_leaf_dropped_for_new_child(Route::Mapping).await;
}

async fn one_leaf_dropped_for_new_child(route: Route) {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let schema = table_schema(NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "d")]),
    ));
    let (catalog, table) = new_table(&warehouse, schema).await;
    let d = field_id(&table, "s.d");
    let s = plain_struct(vec![("d", ints(&[Some(1), None]))], &[true, false]);
    let file = write_idless(&table, &[1, 2], "s", s).await;
    let table = append(&catalog, &table, file).await;
    let table = add_child(&catalog, &table, "s", "k").await;
    let k = field_id(&table, "s.k");
    let table = drop_column(&catalog, &table, "s.d").await;
    let mapping = json!([
        mapped(1, "id", vec![]),
        mapped(2, "s", vec![mapped(d, "d", vec![]), mapped(k, "k", vec![])]),
    ]);
    let table = with_route(&catalog, table, route, mapping).await;

    let batch = scan(&table).await;
    assert_eq!(ids_of(&batch), vec![1, 2]);
    let s = column(&batch, "s");
    assert_eq!(child_names(s), vec!["k".to_string()]);
    assert_eq!(validity(s), vec![true, false]);
    assert_eq!(child_ints(s, "k"), vec![None, None]);
}

#[tokio::test]
async fn idless_one_leaf_struct_dropped_child_new_child_matches_schema_fallback() {
    one_leaf_dropped_for_new_child(Route::Fallback).await;
}

#[tokio::test]
async fn idless_one_leaf_struct_dropped_child_new_child_matches_schema_mapping() {
    one_leaf_dropped_for_new_child(Route::Mapping).await;
}

async fn one_leaf_with_added_child(route: Route) {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let schema = table_schema(NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "d")]),
    ));
    let (catalog, table) = new_table(&warehouse, schema).await;
    let d = field_id(&table, "s.d");
    let s = plain_struct(vec![("d", ints(&[Some(1), None]))], &[true, false]);
    let file = write_idless(&table, &[1, 2], "s", s).await;
    let table = append(&catalog, &table, file).await;
    let table = add_child(&catalog, &table, "s", "k").await;
    let k = field_id(&table, "s.k");
    let mapping = json!([
        mapped(1, "id", vec![]),
        mapped(2, "s", vec![mapped(d, "d", vec![]), mapped(k, "k", vec![])]),
    ]);
    let table = with_route(&catalog, table, route, mapping).await;

    let batch = scan(&table).await;
    assert_eq!(ids_of(&batch), vec![1, 2]);
    let s = column(&batch, "s");
    assert_eq!(child_names(s), vec!["d".to_string(), "k".to_string()]);
    assert_eq!(validity(s), vec![true, false]);
    assert_eq!(child_ints(s, "d"), vec![Some(1), None]);
    assert_eq!(child_ints(s, "k"), vec![None, None]);
}

#[tokio::test]
async fn idless_struct_with_added_child_reads_file_child_and_null_fallback() {
    one_leaf_with_added_child(Route::Fallback).await;
}

#[tokio::test]
async fn idless_struct_with_added_child_reads_file_child_and_null_mapping() {
    one_leaf_with_added_child(Route::Mapping).await;
}

async fn emptied_struct(route: Route) {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let schema = table_schema(NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "d")]),
    ));
    let (catalog, table) = new_table(&warehouse, schema).await;
    let d = field_id(&table, "s.d");
    let s = plain_struct(vec![("d", ints(&[Some(1), None]))], &[true, false]);
    let file = write_idless(&table, &[1, 2], "s", s).await;
    let table = append(&catalog, &table, file).await;
    let table = drop_column(&catalog, &table, "s.d").await;
    let mapping = json!([
        mapped(1, "id", vec![]),
        mapped(2, "s", vec![mapped(d, "d", vec![])]),
    ]);
    let table = with_route(&catalog, table, route, mapping).await;

    let batch = scan(&table).await;
    assert_eq!(ids_of(&batch), vec![1, 2]);
    let s = column(&batch, "s");
    assert_eq!(s.data_type(), &DataType::Struct(Fields::empty()));
    assert_eq!(validity(s), vec![true, false]);
}

#[tokio::test]
async fn idless_emptied_struct_reads_empty_struct_per_row_validity_fallback() {
    emptied_struct(Route::Fallback).await;
}

#[tokio::test]
async fn idless_emptied_struct_reads_empty_struct_per_row_validity_mapping() {
    emptied_struct(Route::Mapping).await;
}

async fn unchanged_struct(route: Route) {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let schema = table_schema(NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "d"), int_field(4, "e")]),
    ));
    let (catalog, table) = new_table(&warehouse, schema).await;
    let (d, e) = (field_id(&table, "s.d"), field_id(&table, "s.e"));
    let s = plain_struct(
        vec![
            ("d", ints(&[Some(1), None, Some(5)])),
            ("e", ints(&[Some(2), None, None])),
        ],
        &[true, false, true],
    );
    let file = write_idless(&table, &[1, 2, 3], "s", s).await;
    let table = append(&catalog, &table, file).await;
    let mapping = json!([
        mapped(1, "id", vec![]),
        mapped(2, "s", vec![mapped(d, "d", vec![]), mapped(e, "e", vec![])]),
    ]);
    let table = with_route(&catalog, table, route, mapping).await;

    let batches = scan_batches(&table).await;
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(ids_of(batch), vec![1, 2, 3]);
    let s = column(batch, "s");
    assert_eq!(child_names(s), vec!["d".to_string(), "e".to_string()]);
    assert_eq!(validity(s), vec![true, false, true]);
    assert_eq!(child_ints(s, "d"), vec![Some(1), None, Some(5)]);
    assert_eq!(child_ints(s, "e"), vec![Some(2), None, None]);
}

#[tokio::test]
async fn idless_struct_with_every_child_in_the_task_schema_is_unchanged_fallback() {
    unchanged_struct(Route::Fallback).await;
}

#[tokio::test]
async fn idless_struct_with_every_child_in_the_task_schema_is_unchanged_mapping() {
    unchanged_struct(Route::Mapping).await;
}

async fn nested_struct_dropped_for_new_child(route: Route) {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let inner = NestedField::optional(3, "t", struct_type(vec![int_field(4, "d")]));
    let schema = table_schema(NestedField::optional(2, "s", struct_type(vec![inner])));
    let (catalog, table) = new_table(&warehouse, schema).await;
    let (t, d) = (field_id(&table, "s.t"), field_id(&table, "s.t.d"));
    let inner = plain_struct(vec![("d", ints(&[Some(1), None, None]))], &[
        true, false, false,
    ]);
    let s = plain_struct(vec![("t", inner)], &[true, false, true]);
    let file = write_idless(&table, &[1, 2, 3], "s", s).await;
    let table = append(&catalog, &table, file).await;
    let table = add_child(&catalog, &table, "s.t", "k").await;
    let k = field_id(&table, "s.t.k");
    let table = drop_column(&catalog, &table, "s.t.d").await;
    let mapping = json!([
        mapped(1, "id", vec![]),
        mapped(2, "s", vec![mapped(t, "t", vec![
            mapped(d, "d", vec![]),
            mapped(k, "k", vec![])
        ])]),
    ]);
    let table = with_route(&catalog, table, route, mapping).await;

    let batch = scan(&table).await;
    assert_eq!(ids_of(&batch), vec![1, 2, 3]);
    let s = column(&batch, "s");
    assert_eq!(validity(s), vec![true, false, true]);
    let t = as_struct(s.column_by_name("t").expect("s.t present").as_ref());
    assert_eq!(child_names(t), vec!["k".to_string()]);
    assert_eq!(validity(t), vec![true, false, false]);
    assert_eq!(child_ints(t, "k"), vec![None, None, None]);
}

#[tokio::test]
async fn idless_nested_struct_dropped_child_new_child_reads_null_per_row_fallback() {
    nested_struct_dropped_for_new_child(Route::Fallback).await;
}

#[tokio::test]
async fn idless_nested_struct_dropped_child_new_child_reads_null_per_row_mapping() {
    nested_struct_dropped_for_new_child(Route::Mapping).await;
}

async fn list_element_dropped_for_new_child(route: Route) {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let element = NestedField::list_element(3, struct_type(vec![int_field(4, "d")]), false);
    let schema = table_schema(NestedField::optional(
        2,
        "l",
        Type::List(ListType {
            element_field: element.into(),
        }),
    ));
    let (catalog, table) = new_table(&warehouse, schema).await;
    let (element, d) = (
        field_id(&table, "l.element"),
        field_id(&table, "l.element.d"),
    );
    let values = plain_struct(vec![("d", ints(&[Some(1), Some(2)]))], &[true, true]);
    let item = Arc::new(Field::new("item", values.data_type().clone(), false));
    let l = ListArray::try_new(
        item,
        OffsetBuffer::new(vec![0, 2, 2, 2].into()),
        values,
        Some(NullBuffer::from(vec![true, false, true])),
    )
    .expect("build the id-less fixture list");
    let file = write_idless(&table, &[1, 2, 3], "l", Arc::new(l)).await;
    let table = append(&catalog, &table, file).await;
    let table = add_child(&catalog, &table, "l.element", "k").await;
    let k = field_id(&table, "l.element.k");
    let table = drop_column(&catalog, &table, "l.element.d").await;
    let mapping = json!([
        mapped(1, "id", vec![]),
        mapped(2, "l", vec![mapped(element, "element", vec![
            mapped(d, "d", vec![]),
            mapped(k, "k", vec![])
        ])]),
    ]);
    let table = with_route(&catalog, table, route, mapping).await;

    let batch = scan(&table).await;
    assert_eq!(ids_of(&batch), vec![1, 2, 3]);
    let l = batch
        .column_by_name("l")
        .expect("l present")
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("l is a list");
    assert_eq!(validity(l), vec![true, false, true]);
    assert_eq!(l.value_length(0), 2);
    assert_eq!(l.value_length(2), 0);
    let elements = as_struct(l.values().as_ref());
    assert_eq!(child_names(elements), vec!["k".to_string()]);
    assert_eq!(validity(elements), vec![true, true]);
    assert_eq!(child_ints(elements, "k"), vec![None, None]);
}

#[tokio::test]
async fn idless_list_element_struct_dropped_child_new_child_reads_null_fallback() {
    list_element_dropped_for_new_child(Route::Fallback).await;
}

#[tokio::test]
async fn idless_list_element_struct_dropped_child_new_child_reads_null_mapping() {
    list_element_dropped_for_new_child(Route::Mapping).await;
}

async fn unchanged_struct_concatenates(route: Route) {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let schema = table_schema(NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "d")]),
    ));
    let (catalog, table) = new_table(&warehouse, schema).await;
    let d = field_id(&table, "s.d");
    let s = plain_struct(vec![("d", ints(&[Some(1), None]))], &[true, false]);
    let file = write_idless(&table, &[1, 2], "s", s).await;
    let table = append(&catalog, &table, file).await;
    let mapping = json!([
        mapped(1, "id", vec![]),
        mapped(2, "s", vec![mapped(d, "d", vec![])]),
    ]);
    let table = with_route(&catalog, table, route, mapping).await;

    let batch = scan(&table).await;
    let s = column(&batch, "s");
    assert_eq!(s.data_type(), batch.schema().field(1).data_type());
    assert_eq!(validity(s), vec![true, false]);
    assert_eq!(child_ints(s, "d"), vec![Some(1), None]);
}

#[tokio::test]
async fn idless_struct_batches_carry_the_schema_type_and_concatenate_fallback() {
    unchanged_struct_concatenates(Route::Fallback).await;
}

#[tokio::test]
async fn idless_struct_batches_carry_the_schema_type_and_concatenate_mapping() {
    unchanged_struct_concatenates(Route::Mapping).await;
}

async fn top_level_only_mapping_table(warehouse: &str) -> (impl Catalog, Table) {
    let schema = table_schema(NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "d")]),
    ));
    let (catalog, table) = new_table(warehouse, schema).await;
    let s = plain_struct(vec![("d", ints(&[Some(1), None]))], &[true, false]);
    let file = write_idless(&table, &[1, 2], "s", s).await;
    let table = append(&catalog, &table, file).await;
    let mapping = json!([mapped(1, "id", vec![]), mapped(2, "s", vec![])]);
    let table = set_mapping(&catalog, &table, mapping).await;
    (catalog, table)
}

#[tokio::test]
async fn idless_struct_without_child_mapping_reads_its_unmapped_child_as_null() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (_catalog, table) = top_level_only_mapping_table(&warehouse).await;

    let batches = scan_batches(&table).await;
    assert_eq!(batches.len(), 1);
    let s = column(&batches[0], "s");
    assert_eq!(validity(s), vec![true, false]);
    assert_eq!(child_ints(s, "d"), vec![None, None]);
}

#[tokio::test]
async fn idless_struct_without_child_mapping_and_an_added_child_reads_null() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, table) = top_level_only_mapping_table(&warehouse).await;
    let table = add_child(&catalog, &table, "s", "k").await;

    let batch = scan(&table).await;
    let s = column(&batch, "s");
    assert_eq!(child_names(s), vec!["d".to_string(), "k".to_string()]);
    assert_eq!(validity(s), vec![true, false]);
    assert_eq!(child_ints(s, "d"), vec![None, None]);
    assert_eq!(child_ints(s, "k"), vec![None, None]);
}
