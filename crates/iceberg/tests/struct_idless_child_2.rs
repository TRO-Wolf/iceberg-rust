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

use arrow_array::{Array, ArrayRef, Int32Array, ListArray, RecordBatch, StringArray, StructArray};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{Field, Fields, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::expr::{Predicate, Reference};
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, FormatVersion, ListType,
    NestedField, PrimitiveType, Schema, SortOrder, Struct, StructType, TableProperties, Type,
    UnboundPartitionSpec, create_name_mapping,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use serde_json::{Value, json};
use tempfile::TempDir;

#[derive(Clone, Copy, Debug)]
enum Route {
    Fallback,
    Mapping,
}

enum Change<'a> {
    Rename(&'a str, &'a str),
    MoveFirst(&'a str),
    Add(&'a str, &'a str),
    Drop(&'a str),
}

fn int_field(id: i32, name: &str) -> NestedField {
    NestedField::optional(id, name, Type::Primitive(PrimitiveType::Int))
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

fn with_id(column: NestedField) -> Schema {
    schema_of(vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)),
        column,
    ])
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

fn write_columns(table: &Table, columns: Vec<(&str, ArrayRef)>, group_rows: usize) -> DataFile {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, array)| Field::new(*name, array.data_type().clone(), *name != "id"))
        .collect();
    let schema = Arc::new(ArrowSchema::new(fields));
    let arrays: Vec<ArrayRef> = columns.into_iter().map(|(_, array)| array).collect();
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).expect("build the id-less batch");
    let dir = format!("{}/data", table.metadata().location());
    std::fs::create_dir_all(&dir).expect("create the data directory");
    let path = format!("{dir}/idless-{}.parquet", uuid::Uuid::now_v7());
    let file = std::fs::File::create(&path).expect("create the id-less parquet file");
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(group_rows))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("build the writer");
    writer.write(&batch).expect("write the id-less batch");
    writer.close().expect("close the arrow writer");
    let size = std::fs::metadata(&path).expect("stat the file").len();
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path)
        .file_format(DataFileFormat::Parquet)
        .partition(Struct::empty())
        .partition_spec_id(0)
        .record_count(batch.num_rows() as u64)
        .file_size_in_bytes(size)
        .build()
        .expect("build the data file")
}

async fn commit_file(catalog: &impl Catalog, table: &Table, file: DataFile) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file])
        .apply(tx)
        .expect("apply fast append");
    tx.commit(catalog).await.expect("commit fast append")
}

async fn evolve(catalog: &impl Catalog, table: Table, changes: &[Change<'_>]) -> Table {
    let mut table = table;
    for change in changes {
        let tx = Transaction::new(&table);
        let update = tx.update_schema();
        let update = match change {
            Change::Rename(name, new_name) => update.rename_column(name, new_name),
            Change::MoveFirst(name) => update.move_first(name),
            Change::Add(parent, name) => update.add_column_to(
                Some(parent),
                name,
                Type::Primitive(PrimitiveType::Int),
                None,
            ),
            Change::Drop(name) => update.delete_column(name),
        };
        let tx = update.apply(tx).expect("apply the schema change");
        table = tx.commit(catalog).await.expect("commit the schema change");
    }
    table
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

fn mapping_at_add_time(table: &Table) -> Value {
    let mapping = create_name_mapping(table.metadata().current_schema()).expect("build mapping");
    serde_json::to_value(&mapping).expect("serialize the mapping")
}

async fn with_route(catalog: &impl Catalog, table: Table, route: Route, mapping: &Value) -> Table {
    match route {
        Route::Fallback => table,
        Route::Mapping => set_mapping(catalog, &table, mapping).await,
    }
}

async fn read_after(
    schema: Schema,
    columns: Vec<(&str, ArrayRef)>,
    changes: &[Change<'_>],
    route: Route,
) -> (TempDir, Table) {
    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, table) = new_table(&warehouse, schema).await;
    let mapping = mapping_at_add_time(&table);
    let file = write_columns(&table, columns, 1024);
    let table = commit_file(&catalog, &table, file).await;
    let table = evolve(&catalog, table, changes).await;
    let table = with_route(&catalog, table, route, &mapping).await;
    (tmp, table)
}

async fn collect(table: &Table, filter: Option<Predicate>, row_groups: bool) -> Vec<RecordBatch> {
    let mut builder = table.scan().with_row_group_filtering_enabled(row_groups);
    if let Some(filter) = filter {
        builder = builder.with_filter(filter);
    }
    builder
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
    let batches = collect(table, None, true).await;
    let schema = batches.first().expect("at least one batch").schema();
    arrow_select::concat::concat_batches(&schema, &batches).expect("concat scan batches")
}

async fn scan_error(table: &Table) -> iceberg::Error {
    let stream = table.scan().build().expect("build scan").to_arrow().await;
    match stream {
        Err(error) => error,
        Ok(stream) => stream
            .try_collect::<Vec<_>>()
            .await
            .expect_err("the scan refuses"),
    }
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

fn child<'a>(parent: &'a StructArray, name: &str) -> &'a StructArray {
    as_struct(
        parent
            .column_by_name(name)
            .unwrap_or_else(|| panic!("child {name} present"))
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

fn nested_table() -> Schema {
    with_id(NestedField::optional(
        2,
        "s",
        struct_type(vec![
            NestedField::optional(3, "t", struct_type(vec![int_field(6, "d")])),
            int_field(4, "u"),
            int_field(5, "w"),
        ]),
    ))
}

fn nested_file() -> Vec<(&'static str, ArrayRef)> {
    let t = plain_struct(vec![("d", ints(&[Some(1)]))], &[true]);
    let s = plain_struct(
        vec![("t", t), ("u", ints(&[Some(7)])), ("w", ints(&[Some(8)]))],
        &[true],
    );
    vec![("id", ints(&[Some(1)])), ("s", s)]
}

async fn nested_after(changes: &[Change<'_>], route: Route) -> (TempDir, RecordBatch) {
    let (tmp, table) = read_after(nested_table(), nested_file(), changes, route).await;
    let batch = scan(&table).await;
    (tmp, batch)
}

async fn single_rename(route: Route, expected_v: Option<i32>) {
    let (_tmp, batch) = nested_after(&[Change::Rename("s.u", "v")], route).await;
    let s = column(&batch, "s");
    assert_eq!(child_names(s), vec!["t", "v", "w"]);
    assert_eq!(child_ints(child(s, "t"), "d"), vec![Some(1)]);
    assert_eq!(child_ints(s, "v"), vec![expected_v]);
    assert_eq!(child_ints(s, "w"), vec![Some(8)]);
}

#[tokio::test]
async fn idless_renamed_child_after_a_nested_sibling_reads_null_fallback() {
    single_rename(Route::Fallback, None).await;
}

#[tokio::test]
async fn idless_renamed_child_after_a_nested_sibling_keeps_its_value_mapping() {
    single_rename(Route::Mapping, Some(7)).await;
}

async fn double_rename(route: Route, expected: (Option<i32>, Option<i32>)) {
    let changes = [Change::Rename("s.u", "uu"), Change::Rename("s.w", "ww")];
    let (_tmp, batch) = nested_after(&changes, route).await;
    let s = column(&batch, "s");
    assert_eq!(child_names(s), vec!["t", "uu", "ww"]);
    assert_eq!(child_ints(child(s, "t"), "d"), vec![Some(1)]);
    assert_eq!(child_ints(s, "uu"), vec![expected.0]);
    assert_eq!(child_ints(s, "ww"), vec![expected.1]);
}

#[tokio::test]
async fn idless_two_renamed_children_never_read_a_neighbour_fallback() {
    double_rename(Route::Fallback, (None, None)).await;
}

#[tokio::test]
async fn idless_two_renamed_children_keep_their_values_mapping() {
    double_rename(Route::Mapping, (Some(7), Some(8))).await;
}

async fn moved_child(route: Route) {
    let (_tmp, batch) = nested_after(&[Change::MoveFirst("s.w")], route).await;
    let s = column(&batch, "s");
    assert_eq!(child_names(s), vec!["w", "t", "u"]);
    assert_eq!(child_ints(s, "w"), vec![Some(8)]);
    assert_eq!(child_ints(child(s, "t"), "d"), vec![Some(1)]);
    assert_eq!(child_ints(s, "u"), vec![Some(7)]);
}

#[tokio::test]
async fn idless_moved_child_reads_by_name_fallback() {
    moved_child(Route::Fallback).await;
}

#[tokio::test]
async fn idless_moved_child_reads_by_name_mapping() {
    moved_child(Route::Mapping).await;
}

async fn added_and_dropped_after_a_nested_sibling(route: Route) {
    let changes = [Change::Add("s", "k"), Change::Drop("s.u")];
    let (_tmp, batch) = nested_after(&changes, route).await;
    let s = column(&batch, "s");
    assert_eq!(child_names(s), vec!["t", "w", "k"]);
    assert_eq!(child_ints(child(s, "t"), "d"), vec![Some(1)]);
    assert_eq!(child_ints(s, "w"), vec![Some(8)]);
    assert_eq!(child_ints(s, "k"), vec![None]);
}

#[tokio::test]
async fn idless_dropped_child_never_lends_its_value_to_a_sibling_fallback() {
    added_and_dropped_after_a_nested_sibling(Route::Fallback).await;
}

#[tokio::test]
async fn idless_dropped_child_never_lends_its_value_to_a_sibling_mapping() {
    added_and_dropped_after_a_nested_sibling(Route::Mapping).await;
}

async fn reordered_file_children(route: Route) {
    let schema = with_id(NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "a"), int_field(4, "b")]),
    ));
    let s = plain_struct(vec![("b", ints(&[Some(10)])), ("a", ints(&[Some(20)]))], &[
        true,
    ]);
    let columns = vec![("id", ints(&[Some(1)])), ("s", s)];
    let (_tmp, table) = read_after(schema, columns, &[], route).await;
    let s = column(&scan(&table).await, "s").clone();
    assert_eq!(child_names(&s), vec!["a", "b"]);
    assert_eq!(child_ints(&s, "a"), vec![Some(20)]);
    assert_eq!(child_ints(&s, "b"), vec![Some(10)]);
}

#[tokio::test]
async fn idless_file_children_in_another_order_read_by_name_fallback() {
    reordered_file_children(Route::Fallback).await;
}

#[tokio::test]
async fn idless_file_children_in_another_order_read_by_name_mapping() {
    reordered_file_children(Route::Mapping).await;
}

async fn extra_file_child_then_added(route: Route) {
    let schema = with_id(NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "d")]),
    ));
    let s = plain_struct(vec![("d", ints(&[Some(1)])), ("x", ints(&[Some(99)]))], &[
        true,
    ]);
    let columns = vec![("id", ints(&[Some(1)])), ("s", s)];
    let (_tmp, table) = read_after(schema, columns, &[Change::Add("s", "k")], route).await;
    let s = column(&scan(&table).await, "s").clone();
    assert_eq!(child_names(&s), vec!["d", "k"]);
    assert_eq!(child_ints(&s, "d"), vec![Some(1)]);
    assert_eq!(child_ints(&s, "k"), vec![None]);
}

#[tokio::test]
async fn idless_extra_file_child_never_reads_under_an_added_child_fallback() {
    extra_file_child_then_added(Route::Fallback).await;
}

#[tokio::test]
async fn idless_extra_file_child_never_reads_under_an_added_child_mapping() {
    extra_file_child_then_added(Route::Mapping).await;
}

async fn coincidence_shape(route: Route) {
    let schema = with_id(NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "a")]),
    ));
    let s = plain_struct(vec![("x", ints(&[Some(10)])), ("y", ints(&[Some(20)]))], &[
        true,
    ]);
    let columns = vec![("id", ints(&[Some(1)])), ("s", s)];
    let (_tmp, table) = read_after(schema, columns, &[], route).await;
    let s = column(&scan(&table).await, "s").clone();
    assert_eq!(validity(&s), vec![true]);
    assert_eq!(child_names(&s), vec!["a"]);
    assert_eq!(child_ints(&s, "a"), vec![None]);
}

#[tokio::test]
async fn idless_file_children_with_no_matching_name_read_null_fallback() {
    coincidence_shape(Route::Fallback).await;
}

#[tokio::test]
async fn idless_file_children_with_no_matching_name_read_null_mapping() {
    coincidence_shape(Route::Mapping).await;
}

#[tokio::test]
async fn idless_partial_nested_mapping_reads_the_unmapped_child_as_null() {
    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let schema = with_id(NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "d"), int_field(4, "e")]),
    ));
    let (catalog, table) = new_table(&warehouse, schema).await;
    let s = plain_struct(vec![("d", ints(&[Some(10)])), ("e", ints(&[Some(20)]))], &[
        true,
    ]);
    let file = write_columns(&table, vec![("id", ints(&[Some(1)])), ("s", s)], 1024);
    let table = commit_file(&catalog, &table, file).await;
    let mapping = json!([
        {"field-id": 1, "names": ["id"]},
        {"field-id": 2, "names": ["s"], "fields": [{"field-id": 3, "names": ["d"]}]},
    ]);
    let table = set_mapping(&catalog, &table, &mapping).await;
    let s = column(&scan(&table).await, "s").clone();
    assert_eq!(child_names(&s), vec!["d", "e"]);
    assert_eq!(child_ints(&s, "d"), vec![Some(10)]);
    assert_eq!(child_ints(&s, "e"), vec![None]);
}

async fn two_deep_by_name(route: Route, expected_bb: Option<i32>) {
    let schema = with_id(NestedField::optional(
        2,
        "s",
        struct_type(vec![NestedField::optional(
            3,
            "t",
            struct_type(vec![int_field(4, "a"), int_field(5, "b")]),
        )]),
    ));
    let t = plain_struct(vec![("b", ints(&[Some(2)])), ("a", ints(&[Some(1)]))], &[
        true,
    ]);
    let s = plain_struct(vec![("t", t)], &[true]);
    let columns = vec![("id", ints(&[Some(1)])), ("s", s)];
    let changes = [Change::Rename("s.t.b", "bb")];
    let (_tmp, table) = read_after(schema, columns, &changes, route).await;
    let batch = scan(&table).await;
    let t = child(column(&batch, "s"), "t");
    assert_eq!(child_names(t), vec!["a", "bb"]);
    assert_eq!(child_ints(t, "a"), vec![Some(1)]);
    assert_eq!(child_ints(t, "bb"), vec![expected_bb]);
}

#[tokio::test]
async fn idless_struct_two_deep_reads_by_name_and_a_rename_reads_null_fallback() {
    two_deep_by_name(Route::Fallback, None).await;
}

#[tokio::test]
async fn idless_struct_two_deep_reads_by_name_and_a_rename_keeps_its_value_mapping() {
    two_deep_by_name(Route::Mapping, Some(2)).await;
}

async fn list_element_by_name(route: Route) {
    let element = struct_type(vec![int_field(4, "a"), int_field(5, "b")]);
    let schema = with_id(NestedField::optional(
        2,
        "l",
        Type::List(ListType {
            element_field: NestedField::list_element(3, element, false).into(),
        }),
    ));
    let element = plain_struct(
        vec![
            ("b", ints(&[Some(10), Some(11)])),
            ("a", ints(&[Some(20), Some(21)])),
        ],
        &[true, true],
    );
    let item = Arc::new(Field::new("item", element.data_type().clone(), false));
    let list = ListArray::try_new(item, OffsetBuffer::from_lengths([2]), element, None)
        .expect("build the id-less list");
    let columns = vec![("id", ints(&[Some(1)])), ("l", Arc::new(list) as ArrayRef)];
    let (_tmp, table) = read_after(schema, columns, &[], route).await;
    let batch = scan(&table).await;
    let list = batch
        .column_by_name("l")
        .expect("l present")
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("l is a list");
    let element = as_struct(list.values().as_ref());
    assert_eq!(child_names(element), vec!["a", "b"]);
    assert_eq!(child_ints(element, "a"), vec![Some(20), Some(21)]);
    assert_eq!(child_ints(element, "b"), vec![Some(10), Some(11)]);
}

#[tokio::test]
async fn idless_list_element_children_read_by_name_fallback() {
    list_element_by_name(Route::Fallback).await;
}

#[tokio::test]
async fn idless_list_element_children_read_by_name_mapping() {
    list_element_by_name(Route::Mapping).await;
}

async fn null_free(route: Route, changes: &[Change<'_>]) -> (TempDir, RecordBatch) {
    let schema = with_id(NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "d")]),
    ));
    let s = plain_struct(vec![("d", ints(&[Some(1), Some(2)]))], &[true, true]);
    let columns = vec![("id", ints(&[Some(1), Some(2)])), ("s", s)];
    let (tmp, table) = read_after(schema, columns, changes, route).await;
    let batch = scan(&table).await;
    (tmp, batch)
}

async fn null_free_dropped_for_new_child(route: Route) {
    let changes = [Change::Add("s", "k"), Change::Drop("s.d")];
    let (_tmp, batch) = null_free(route, &changes).await;
    let s = column(&batch, "s");
    assert_eq!(child_names(s), vec!["k"]);
    assert_eq!(validity(s), vec![true, true]);
    assert_eq!(child_ints(s, "k"), vec![None, None]);
}

#[tokio::test]
async fn idless_null_free_struct_dropped_child_new_child_reads_null_fallback() {
    null_free_dropped_for_new_child(Route::Fallback).await;
}

#[tokio::test]
async fn idless_null_free_struct_dropped_child_new_child_reads_null_mapping() {
    null_free_dropped_for_new_child(Route::Mapping).await;
}

async fn null_free_added_child(route: Route) {
    let (_tmp, batch) = null_free(route, &[Change::Add("s", "k")]).await;
    let s = column(&batch, "s");
    assert_eq!(child_names(s), vec!["d", "k"]);
    assert_eq!(validity(s), vec![true, true]);
    assert_eq!(child_ints(s, "d"), vec![Some(1), Some(2)]);
    assert_eq!(child_ints(s, "k"), vec![None, None]);
}

#[tokio::test]
async fn idless_null_free_struct_added_child_reads_file_child_and_null_fallback() {
    null_free_added_child(Route::Fallback).await;
}

#[tokio::test]
async fn idless_null_free_struct_added_child_reads_file_child_and_null_mapping() {
    null_free_added_child(Route::Mapping).await;
}

async fn incompatible_nested_type(route: Route, table_type: PrimitiveType, file: ArrayRef) {
    let schema = with_id(NestedField::optional(
        2,
        "s",
        struct_type(vec![NestedField::optional(
            3,
            "d",
            Type::Primitive(table_type),
        )]),
    ));
    let s = plain_struct(vec![("d", file)], &[true, false]);
    let columns = vec![("id", ints(&[Some(1), Some(2)])), ("s", s)];
    let (_tmp, table) = read_after(schema, columns, &[], route).await;
    let error = scan_error(&table).await;
    assert_eq!(error.kind(), ErrorKind::DataInvalid, "{error}");
    let message = error.to_string();
    assert!(message.contains("'s.d'"), "{message}");
    assert!(
        message.contains("int") && message.contains("string"),
        "{message}"
    );
}

fn file_strings() -> ArrayRef {
    Arc::new(StringArray::from(vec![Some("1"), None]))
}

#[tokio::test]
async fn idless_nested_int_under_a_string_child_refuses_fallback() {
    incompatible_nested_type(
        Route::Fallback,
        PrimitiveType::String,
        ints(&[Some(1), None]),
    )
    .await;
}

#[tokio::test]
async fn idless_nested_int_under_a_string_child_refuses_mapping() {
    incompatible_nested_type(
        Route::Mapping,
        PrimitiveType::String,
        ints(&[Some(1), None]),
    )
    .await;
}

#[tokio::test]
async fn idless_nested_string_under_an_int_child_refuses_fallback() {
    incompatible_nested_type(Route::Fallback, PrimitiveType::Int, file_strings()).await;
}

#[tokio::test]
async fn idless_nested_string_under_an_int_child_refuses_mapping() {
    incompatible_nested_type(Route::Mapping, PrimitiveType::Int, file_strings()).await;
}

fn count(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

async fn assert_one_row_both_ways(table: &Table, column: &str, value: i32) {
    let filter = || Reference::new(column).equal_to(Datum::int(value));
    let pruned = collect(table, Some(filter()), true).await;
    let unpruned = collect(table, Some(filter()), false).await;
    assert_eq!(
        count(&unpruned),
        1,
        "{column} = {value} without row-group filtering"
    );
    assert_eq!(
        count(&pruned),
        1,
        "{column} = {value} with row-group filtering"
    );
}

#[tokio::test]
async fn idless_row_group_pruning_uses_the_mapped_ids() {
    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let schema = schema_of(vec![int_field(1, "a"), int_field(2, "b")]);
    let (catalog, table) = new_table(&warehouse, schema).await;
    let mapping = mapping_at_add_time(&table);
    let columns = vec![
        ("b", ints(&[Some(40), Some(30), Some(20), Some(10)])),
        ("a", ints(&[Some(1), Some(2), Some(3), Some(4)])),
    ];
    let file = write_columns(&table, columns, 2);
    let table = commit_file(&catalog, &table, file).await;
    let table = set_mapping(&catalog, &table, &mapping).await;
    assert_eq!(count(&collect(&table, None, true).await), 4);
    assert_one_row_both_ways(&table, "a", 1).await;
    assert_one_row_both_ways(&table, "b", 10).await;
}

#[tokio::test]
async fn idless_row_group_pruning_uses_the_fallback_ids_after_a_leading_drop() {
    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let schema = schema_of(vec![
        int_field(1, "x"),
        int_field(2, "a"),
        int_field(3, "b"),
    ]);
    let (catalog, table) = new_table(&warehouse, schema).await;
    let columns = vec![
        ("a", ints(&[Some(1), Some(2), Some(3), Some(4)])),
        ("b", ints(&[Some(40), Some(30), Some(20), Some(10)])),
    ];
    let file = write_columns(&table, columns, 2);
    let table = commit_file(&catalog, &table, file).await;
    let table = evolve(&catalog, table, &[Change::Drop("x")]).await;
    assert_eq!(count(&collect(&table, None, true).await), 4);
    assert_one_row_both_ways(&table, "a", 1).await;
    assert_one_row_both_ways(&table, "b", 10).await;
}
