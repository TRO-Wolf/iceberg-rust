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
use arrow_array::{Array, ArrayRef, Int32Array, ListArray, RecordBatch, StructArray};
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

const READER_MARK: &str = "iceberg.reader.left-unstamped";

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

fn schema_of(fields: Vec<NestedField>) -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(fields.into_iter().map(Arc::new).collect::<Vec<_>>())
        .build()
        .expect("build the test schema")
}

fn id_column() -> NestedField {
    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int))
}

fn int_list_column() -> NestedField {
    NestedField::optional(
        2,
        "l",
        Type::List(ListType::new(Arc::new(NestedField::list_element(
            3,
            Type::Primitive(PrimitiveType::Int),
            false,
        )))),
    )
}

fn struct_list_column() -> NestedField {
    NestedField::optional(
        2,
        "l",
        Type::List(ListType::new(Arc::new(NestedField::list_element(
            3,
            struct_type(vec![int_field(4, "a"), int_field(5, "b")]),
            false,
        )))),
    )
}

fn int_map_column() -> NestedField {
    NestedField::optional(
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
    )
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

fn id_meta(id: i32) -> HashMap<String, String> {
    HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())])
}

fn struct_of(children: Vec<(Field, ArrayRef)>) -> ArrayRef {
    let len = children.first().map_or(0, |(_, array)| array.len());
    let (fields, arrays): (Vec<Field>, Vec<ArrayRef>) = children.into_iter().unzip();
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

fn int_map() -> ArrayRef {
    let mut builder = MapBuilder::new(None, Int32Builder::new(), Int32Builder::new());
    for row in [&[(1, 10), (2, 20)][..], &[(3, 30)][..]] {
        for (key, value) in row {
            builder.keys().append_value(*key);
            builder.values().append_value(*value);
        }
        builder.append(true).expect("append a map row");
    }
    Arc::new(builder.finish())
}

fn column_field(name: &str, array: &ArrayRef, id: Option<i32>) -> Field {
    let field = Field::new(name, array.data_type().clone(), true);
    match id {
        Some(id) => field.with_metadata(id_meta(id)),
        None => field,
    }
}

fn write_file_with_fields(table: &Table, fields: Vec<Field>, columns: Vec<ArrayRef>) -> DataFile {
    let rows = columns.first().map_or(0, |column| column.len());
    let schema = Arc::new(ArrowSchema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).expect("build the file batch");
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

async fn table_with(
    schema: Schema,
    fields: Vec<Field>,
    columns: Vec<ArrayRef>,
    mapping: Option<Value>,
) -> (TempDir, Table) {
    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, table) = new_table(&warehouse, schema).await;
    let file = write_file_with_fields(&table, fields, columns);
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");
    let Some(mapping) = mapping else {
        return (tmp, table);
    };
    let tx = Transaction::new(&table);
    let tx = tx
        .update_table_properties()
        .set(
            TableProperties::PROPERTY_DEFAULT_NAME_MAPPING.to_string(),
            mapping.to_string(),
        )
        .apply(tx)
        .expect("apply the name mapping property");
    let table = tx.commit(&catalog).await.expect("commit the name mapping");
    (tmp, table)
}

fn route_mapping(route: Route, schema: &Schema) -> Option<Value> {
    match route {
        Route::Fallback => None,
        Route::Mapping => Some(
            serde_json::to_value(create_name_mapping(schema).expect("build mapping"))
                .expect("serialize the mapping"),
        ),
    }
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

async fn assert_refuses(table: &Table, columns: Option<&[&str]>, needles: &[&str]) {
    let error = scan_result(table, columns)
        .await
        .expect_err("the scan refuses");
    let message = error.to_string();
    assert_eq!(error.kind(), ErrorKind::DataInvalid, "{message}");
    for needle in needles {
        assert!(message.contains(needle), "missing {needle:?} in {message}");
    }
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> &'a dyn Array {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("column {name} present"))
        .as_ref()
}

fn as_ints(array: &dyn Array) -> Vec<Option<i32>> {
    array
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("an int column")
        .iter()
        .collect()
}

fn child_ints(parent: &dyn Array, name: &str) -> Vec<Option<i32>> {
    let parent = parent
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("column is a StructArray");
    as_ints(
        parent
            .column_by_name(name)
            .unwrap_or_else(|| panic!("child {name} present"))
            .as_ref(),
    )
}

fn marked_paths(fields: &Fields, prefix: &str, found: &mut Vec<String>) {
    for field in fields {
        let path = format!("{prefix}{}", field.name());
        if field.metadata().contains_key(READER_MARK) {
            found.push(path.clone());
        }
        match field.data_type() {
            DataType::Struct(children) => marked_paths(children, &format!("{path}."), found),
            DataType::List(element) | DataType::LargeList(element) => marked_paths(
                &Fields::from(vec![element.clone()]),
                &format!("{path}."),
                found,
            ),
            DataType::Map(entries, _) => marked_paths(
                &Fields::from(vec![entries.clone()]),
                &format!("{path}."),
                found,
            ),
            _ => {}
        }
    }
}

fn assert_no_reader_mark(batch: &RecordBatch) {
    let mut found = Vec::new();
    marked_paths(batch.schema().fields(), "", &mut found);
    assert!(
        found.is_empty(),
        "the reader mark reached the output at {found:?}"
    );
}

fn embedded_ids_table() -> Schema {
    schema_of(vec![
        int_field(1, "x"),
        NestedField::optional(
            2,
            "s",
            struct_type(vec![int_field(3, "e"), int_field(4, "d")]),
        ),
    ])
}

fn embedded_ids_file() -> (Vec<Field>, Vec<ArrayRef>) {
    let s = struct_of(vec![
        (
            Field::new("d", DataType::Int32, true).with_metadata(id_meta(3)),
            ints(&[Some(10)]),
        ),
        (
            Field::new("e", DataType::Int32, true).with_metadata(id_meta(4)),
            ints(&[Some(20)]),
        ),
    ]);
    let x = ints(&[Some(7)]);
    (
        vec![column_field("x", &x, None), column_field("s", &s, Some(2))],
        vec![x, s],
    )
}

async fn first_field_without_an_id_keeps_embedded_nested_ids(route: Route) {
    let schema = embedded_ids_table();
    let mapping = route_mapping(route, &schema);
    let (fields, columns) = embedded_ids_file();
    let (_tmp, table) = table_with(schema, fields, columns, mapping).await;
    let batch = scan(&table, None).await;
    let s = column(&batch, "s");
    assert_eq!(child_ints(s, "e"), vec![Some(10)]);
    assert_eq!(child_ints(s, "d"), vec![Some(20)]);
    assert_eq!(as_ints(column(&batch, "x")), vec![None]);
}

#[tokio::test]
async fn a_first_field_without_an_id_keeps_the_embedded_nested_ids_fallback() {
    first_field_without_an_id_keeps_embedded_nested_ids(Route::Fallback).await;
}

#[tokio::test]
async fn a_first_field_without_an_id_keeps_the_embedded_nested_ids_mapping() {
    first_field_without_an_id_keeps_embedded_nested_ids(Route::Mapping).await;
}

fn unmapped_mapping() -> Value {
    json!([{"field-id": 9, "names": ["zz"]}])
}

fn idless_ints(names: &[&str]) -> (Vec<Field>, Vec<ArrayRef>) {
    let columns: Vec<ArrayRef> = (1..=names.len() as i32)
        .map(|value| ints(&[Some(value)]))
        .collect();
    let fields = names
        .iter()
        .zip(&columns)
        .map(|(name, array)| column_field(name, array, None))
        .collect();
    (fields, columns)
}

async fn two_unmapped_columns(route: Route) {
    let schema = schema_of(vec![int_field(1, "a"), int_field(2, "c")]);
    let mapping = match route {
        Route::Fallback => None,
        Route::Mapping => Some(unmapped_mapping()),
    };
    let (fields, columns) = idless_ints(&["a", "c"]);
    let (_tmp, table) = table_with(schema, fields, columns, mapping).await;
    let batch = scan(&table, None).await;
    assert_no_reader_mark(&batch);
    assert_eq!(as_ints(column(&batch, "a")), vec![Some(1)]);
    assert_eq!(as_ints(column(&batch, "c")), vec![Some(2)]);
}

#[tokio::test]
async fn the_reader_mark_never_reaches_the_output_with_two_unmapped_columns_fallback() {
    two_unmapped_columns(Route::Fallback).await;
}

#[tokio::test]
async fn the_reader_mark_never_reaches_the_output_with_two_unmapped_columns_mapping() {
    two_unmapped_columns(Route::Mapping).await;
}

async fn one_unmapped_column(route: Route) {
    let schema = schema_of(vec![int_field(1, "c")]);
    let mapping = match route {
        Route::Fallback => None,
        Route::Mapping => Some(unmapped_mapping()),
    };
    let (fields, columns) = idless_ints(&["c"]);
    let (_tmp, table) = table_with(schema, fields, columns, mapping).await;
    let batch = scan(&table, None).await;
    assert_no_reader_mark(&batch);
    assert_eq!(as_ints(column(&batch, "c")), vec![Some(1)]);
}

#[tokio::test]
async fn the_reader_mark_never_reaches_the_output_with_one_unmapped_column_fallback() {
    one_unmapped_column(Route::Fallback).await;
}

#[tokio::test]
async fn the_reader_mark_never_reaches_the_output_with_one_unmapped_column_mapping() {
    one_unmapped_column(Route::Mapping).await;
}

#[tokio::test]
async fn the_reader_mark_on_an_unmapped_struct_child_never_reaches_the_output() {
    let schema = schema_of(vec![
        id_column(),
        NestedField::optional(
            2,
            "s",
            struct_type(vec![int_field(3, "d"), int_field(4, "e")]),
        ),
    ]);
    let mapping = json!([
        {"field-id": 1, "names": ["id"]},
        {"field-id": 2, "names": ["s"], "fields": [{"field-id": 3, "names": ["d"]}]},
    ]);
    let s = struct_of(vec![
        (Field::new("d", DataType::Int32, true), ints(&[Some(10)])),
        (Field::new("e", DataType::Int32, true), ints(&[Some(20)])),
    ]);
    let id = ints(&[Some(1)]);
    let fields = vec![
        Field::new("id", DataType::Int32, false),
        column_field("s", &s, None),
    ];
    let (_tmp, table) = table_with(schema, fields, vec![id, s], Some(mapping)).await;
    let batch = scan(&table, None).await;
    assert_no_reader_mark(&batch);
    assert_eq!(child_ints(column(&batch, "s"), "d"), vec![Some(10)]);
    assert_eq!(child_ints(column(&batch, "s"), "e"), vec![None]);
}

fn two_ids() -> ArrayRef {
    ints(&[Some(1), Some(2)])
}

fn incomplete_map_mapping() -> Value {
    json!([
        {"field-id": 1, "names": ["id"]},
        {"field-id": 2, "names": ["m"]},
    ])
}

fn incomplete_list_mapping() -> Value {
    json!([
        {"field-id": 1, "names": ["id"]},
        {"field-id": 2, "names": ["l"]},
    ])
}

fn idless_id_and(name: &str, array: ArrayRef) -> (Vec<Field>, Vec<ArrayRef>) {
    let fields = vec![
        Field::new("id", DataType::Int32, false),
        column_field(name, &array, None),
    ];
    (fields, vec![two_ids(), array])
}

#[tokio::test]
async fn an_unprojected_map_with_an_incomplete_mapping_reads_the_selected_column() {
    let schema = schema_of(vec![id_column(), int_map_column()]);
    let (fields, columns) = idless_id_and("m", int_map());
    let (_tmp, table) = table_with(schema, fields, columns, Some(incomplete_map_mapping())).await;
    let batch = scan(&table, Some(&["id"])).await;
    assert_eq!(as_ints(column(&batch, "id")), vec![Some(1), Some(2)]);
}

#[tokio::test]
async fn a_dropped_map_with_an_incomplete_mapping_never_refuses() {
    let schema = schema_of(vec![id_column()]);
    let (fields, columns) = idless_id_and("m", int_map());
    let (_tmp, table) = table_with(schema, fields, columns, Some(incomplete_map_mapping())).await;
    let batch = scan(&table, Some(&["id"])).await;
    assert_eq!(as_ints(column(&batch, "id")), vec![Some(1), Some(2)]);
    let batch = scan(&table, None).await;
    assert_eq!(as_ints(column(&batch, "id")), vec![Some(1), Some(2)]);
}

#[tokio::test]
async fn an_unprojected_list_with_an_incomplete_mapping_reads_the_selected_column() {
    let schema = schema_of(vec![id_column(), int_list_column()]);
    let list = list_of(ints(&[Some(1), Some(2), Some(3)]), vec![0, 2, 3]);
    let (fields, columns) = idless_id_and("l", list);
    let (_tmp, table) = table_with(schema, fields, columns, Some(incomplete_list_mapping())).await;
    let batch = scan(&table, Some(&["id"])).await;
    assert_eq!(as_ints(column(&batch, "id")), vec![Some(1), Some(2)]);
}

#[tokio::test]
async fn a_projected_map_with_an_incomplete_mapping_still_refuses() {
    let schema = schema_of(vec![id_column(), int_map_column()]);
    let (fields, columns) = idless_id_and("m", int_map());
    let (_tmp, table) = table_with(schema, fields, columns, Some(incomplete_map_mapping())).await;
    assert_refuses(&table, Some(&["m"]), &["'m'", "'key'"]).await;
    assert_refuses(&table, Some(&["id", "m"]), &["'m'", "'key'"]).await;
}

fn top_level_ids_file(name: &str, array: ArrayRef) -> (Vec<Field>, Vec<ArrayRef>) {
    let fields = vec![
        Field::new("id", DataType::Int32, false).with_metadata(id_meta(1)),
        column_field(name, &array, Some(2)),
    ];
    (fields, vec![two_ids(), array])
}

fn int_list() -> ArrayRef {
    list_of(ints(&[Some(1), Some(2), Some(3)]), vec![0, 2, 3])
}

fn struct_list() -> ArrayRef {
    let element = struct_of(vec![
        (
            Field::new("a", DataType::Int32, true),
            ints(&[Some(1), Some(2), Some(3)]),
        ),
        (
            Field::new("b", DataType::Int32, true),
            ints(&[Some(10), Some(20), Some(30)]),
        ),
    ]);
    list_of(element, vec![0, 2, 3])
}

#[tokio::test]
async fn a_list_of_ints_without_an_element_id_in_a_file_with_ids_refuses() {
    let schema = schema_of(vec![id_column(), int_list_column()]);
    let (fields, columns) = top_level_ids_file("l", int_list());
    let (_tmp, table) = table_with(schema, fields, columns, None).await;
    assert_refuses(&table, None, &["list 'l'", "no field ids"]).await;
}

#[tokio::test]
async fn a_list_of_structs_without_nested_ids_in_a_file_with_ids_refuses() {
    let schema = schema_of(vec![id_column(), struct_list_column()]);
    let (fields, columns) = top_level_ids_file("l", struct_list());
    let (_tmp, table) = table_with(schema, fields, columns, None).await;
    assert_refuses(&table, None, &["list 'l'", "no field ids"]).await;
}

#[tokio::test]
async fn a_map_without_key_and_value_ids_in_a_file_with_ids_refuses() {
    let schema = schema_of(vec![id_column(), int_map_column()]);
    let (fields, columns) = top_level_ids_file("m", int_map());
    let (_tmp, table) = table_with(schema, fields, columns, None).await;
    assert_refuses(&table, None, &["map 'm'", "no field ids"]).await;
}

#[tokio::test]
async fn an_unprojected_list_without_nested_ids_in_a_file_with_ids_never_refuses() {
    let schema = schema_of(vec![id_column(), int_list_column()]);
    let (fields, columns) = top_level_ids_file("l", int_list());
    let (_tmp, table) = table_with(schema, fields, columns, None).await;
    let batch = scan(&table, Some(&["id"])).await;
    assert_eq!(as_ints(column(&batch, "id")), vec![Some(1), Some(2)]);
}

#[tokio::test]
async fn an_idless_list_of_ints_still_reads_its_values_on_the_fallback_route() {
    let schema = schema_of(vec![id_column(), int_list_column()]);
    let (fields, columns) = idless_id_and("l", int_list());
    let (_tmp, table) = table_with(schema, fields, columns, None).await;
    let batch = scan(&table, None).await;
    let list = column(&batch, "l")
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("column is a ListArray");
    assert_eq!(as_ints(list.value(0).as_ref()), vec![Some(1), Some(2)]);
    assert_eq!(as_ints(list.value(1).as_ref()), vec![Some(3)]);
}
