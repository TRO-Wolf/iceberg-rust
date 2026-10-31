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

use arrow_array::{Array, ArrayRef, Int32Array, ListArray, MapArray, RecordBatch, StructArray};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, FieldRef, Fields};
use futures::TryStreamExt;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, FormatVersion, ListType, MapType, NestedField, PrimitiveType,
    Schema, SortOrder, Struct, StructType, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};

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

async fn create_table(catalog: &impl Catalog, location: &str, schema: Schema) -> Table {
    let namespace = NamespaceIdent::new("pins".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("struct_table".to_string())
        .location(location.to_string())
        .schema(schema)
        .partition_spec(UnboundPartitionSpec::builder().build())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create struct_table")
}

fn column_type(table: &Table, index: usize) -> DataType {
    let schema = table.metadata().current_schema();
    let arrow_schema = schema_to_arrow_schema(schema).expect("schema converts to arrow");
    arrow_schema.field(index).data_type().clone()
}

fn struct_fields(data_type: &DataType) -> Fields {
    match data_type {
        DataType::Struct(fields) => fields.clone(),
        other => panic!("expected a struct type, got {other}"),
    }
}

fn list_element(data_type: &DataType) -> FieldRef {
    match data_type {
        DataType::List(element) => element.clone(),
        other => panic!("expected a list type, got {other}"),
    }
}

fn ints(values: &[Option<i32>]) -> ArrayRef {
    Arc::new(Int32Array::from(values.to_vec()))
}

fn struct_array(data_type: &DataType, children: Vec<ArrayRef>, valid: &[bool]) -> ArrayRef {
    Arc::new(
        StructArray::try_new_with_length(
            struct_fields(data_type),
            children,
            Some(NullBuffer::from(valid.to_vec())),
            valid.len(),
        )
        .expect("build the fixture struct"),
    )
}

async fn write_file(table: &Table, ids: &[i32], column: ArrayRef) -> DataFile {
    let schema = table.metadata().current_schema().clone();
    let arrow_schema =
        Arc::new(schema_to_arrow_schema(&schema).expect("iceberg schema converts to arrow"));
    let batch = RecordBatch::try_new(arrow_schema, vec![ints(&ids_opt(ids)), column])
        .expect("build the fixture batch");
    let file_path = format!(
        "{}/data/{}-data.parquet",
        table.metadata().location(),
        uuid::Uuid::now_v7()
    );
    let output = table.file_io().new_output(file_path).expect("new output");
    let builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema,
    );
    let mut writer = builder.build(output).await.expect("build parquet writer");
    writer.write(&batch).await.expect("write data batch");
    let mut builders = writer.close().await.expect("close parquet writer");
    let mut data_file = builders.pop().expect("one data file builder");
    data_file
        .content(DataContentType::Data)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .expect("build data file")
}

fn ids_opt(ids: &[i32]) -> Vec<Option<i32>> {
    ids.iter().copied().map(Some).collect()
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

async fn add_column(catalog: &impl Catalog, table: &Table, name: &str, ty: Type) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .update_schema()
        .add_column(name, ty)
        .apply(tx)
        .expect("apply add column");
    tx.commit(catalog).await.expect("commit add column")
}

async fn scan(table: &Table, columns: Option<&[&str]>) -> RecordBatch {
    let builder = table.scan();
    let builder = match columns {
        Some(columns) => builder.select(columns.iter().copied()),
        None => builder,
    };
    let batches = builder
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to arrow")
        .try_collect::<Vec<_>>()
        .await
        .expect("collect scan batches");
    let schema = batches.first().expect("at least one batch").schema();
    let mut batches = batches;
    batches.sort_by_key(first_id);
    arrow_select::concat::concat_batches(&schema, &batches).expect("concat batches")
}

fn first_id(batch: &RecordBatch) -> i32 {
    batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .map(|ids| ids.value(0))
        .unwrap_or_default()
}

fn id_values(batch: &RecordBatch) -> Vec<i32> {
    let ids = batch
        .column_by_name("id")
        .expect("id column present")
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("id is an int column");
    ids.iter().map(|id| id.expect("id is never null")).collect()
}

fn validity(array: &dyn Array) -> Vec<bool> {
    (0..array.len()).map(|row| array.is_valid(row)).collect()
}

fn empty_struct_column<'a>(batch: &'a RecordBatch, name: &str) -> &'a StructArray {
    let column = batch.column_by_name(name).expect("struct column present");
    assert_eq!(column.data_type(), &DataType::Struct(Fields::empty()));
    column
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("struct column is a StructArray")
}

async fn one_child_table(catalog: &impl Catalog, location: &str) -> Table {
    let schema = table_schema(NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "k")]),
    ));
    let table = create_table(catalog, location, schema).await;
    let s_type = column_type(&table, 1);
    let s = struct_array(&s_type, vec![ints(&[Some(10)])], &[true]);
    let file = write_file(&table, &[1], s).await;
    append(catalog, &table, file).await
}

#[tokio::test]
async fn one_child_struct_dropped_reads_non_null_empty_struct() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let catalog = load_catalog(&warehouse).await;
    let table = one_child_table(&catalog, &format!("{warehouse}/t")).await;
    let table = drop_column(&catalog, &table, "s.k").await;

    let batch = scan(&table, None).await;
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(id_values(&batch), vec![1]);
    let s = empty_struct_column(&batch, "s");
    assert_eq!(validity(s), vec![true]);

    let ids = scan(&table, Some(&["id"])).await;
    assert_eq!(ids.num_columns(), 1);
    assert_eq!(id_values(&ids), vec![1]);

    let only_s = scan(&table, Some(&["s"])).await;
    assert_eq!(only_s.num_columns(), 1);
    assert_eq!(validity(empty_struct_column(&only_s, "s")), vec![true]);
}

#[tokio::test]
async fn two_child_struct_dropped_one_at_a_time_reads_non_null_empty_struct() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let catalog = load_catalog(&warehouse).await;
    let schema = table_schema(NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "k"), int_field(4, "j")]),
    ));
    let table = create_table(&catalog, &format!("{warehouse}/t"), schema).await;
    let s_type = column_type(&table, 1);
    let s = struct_array(&s_type, vec![ints(&[Some(10)]), ints(&[Some(20)])], &[true]);
    let file = write_file(&table, &[1], s).await;
    let table = append(&catalog, &table, file).await;
    let table = drop_column(&catalog, &table, "s.k").await;
    let table = drop_column(&catalog, &table, "s.j").await;

    let batch = scan(&table, None).await;
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(id_values(&batch), vec![1]);
    assert_eq!(validity(empty_struct_column(&batch, "s")), vec![true]);
    assert_eq!(id_values(&scan(&table, Some(&["id"])).await), vec![1]);
}

#[tokio::test]
async fn null_struct_row_stays_null_beside_non_null_empty_struct() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let catalog = load_catalog(&warehouse).await;
    let schema = table_schema(NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "k")]),
    ));
    let table = create_table(&catalog, &format!("{warehouse}/t"), schema).await;
    let s_type = column_type(&table, 1);
    let s = struct_array(&s_type, vec![ints(&[Some(10), None])], &[true, false]);
    let file = write_file(&table, &[1, 2], s).await;
    let table = append(&catalog, &table, file).await;
    let table = drop_column(&catalog, &table, "s.k").await;

    let batch = scan(&table, None).await;
    assert_eq!(id_values(&batch), vec![1, 2]);
    assert_eq!(validity(empty_struct_column(&batch, "s")), vec![
        true, false
    ]);
}

#[tokio::test]
async fn file_written_between_drops_scans_with_the_old_file() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let catalog = load_catalog(&warehouse).await;
    let schema = table_schema(NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "k"), int_field(4, "j")]),
    ));
    let table = create_table(&catalog, &format!("{warehouse}/t"), schema).await;
    let s_type = column_type(&table, 1);
    let s = struct_array(&s_type, vec![ints(&[Some(10)]), ints(&[Some(20)])], &[true]);
    let file = write_file(&table, &[1], s).await;
    let table = append(&catalog, &table, file).await;
    let table = drop_column(&catalog, &table, "s.k").await;

    let s_type = column_type(&table, 1);
    let s = struct_array(&s_type, vec![ints(&[Some(30), None])], &[true, false]);
    let file = write_file(&table, &[2, 3], s).await;
    let table = append(&catalog, &table, file).await;
    let table = drop_column(&catalog, &table, "s.j").await;

    let batch = scan(&table, None).await;
    assert_eq!(id_values(&batch), vec![1, 2, 3]);
    assert_eq!(validity(empty_struct_column(&batch, "s")), vec![
        true, true, false
    ]);
}

#[tokio::test]
async fn struct_added_after_the_file_reads_null_per_row() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let catalog = load_catalog(&warehouse).await;
    let table = one_child_table(&catalog, &format!("{warehouse}/t")).await;
    let added = struct_type(vec![int_field(0, "x")]);
    let table = add_column(&catalog, &table, "added", added).await;

    let batch = scan(&table, None).await;
    let added = batch.column_by_name("added").expect("added column present");
    assert_eq!(validity(added.as_ref()), vec![false]);

    let table = drop_column(&catalog, &table, "added.x").await;
    let batch = scan(&table, None).await;
    assert_eq!(id_values(&batch), vec![1]);
    let s = batch.column_by_name("s").expect("s column present");
    assert_eq!(validity(s.as_ref()), vec![true]);
    assert_eq!(validity(empty_struct_column(&batch, "added")), vec![false]);
}

#[tokio::test]
async fn struct_nested_in_struct_with_children_dropped_keeps_file_validity() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let catalog = load_catalog(&warehouse).await;
    let inner = NestedField::optional(4, "i", struct_type(vec![int_field(5, "x")]));
    let schema = table_schema(NestedField::optional(
        2,
        "o",
        struct_type(vec![int_field(3, "a"), inner]),
    ));
    let table = create_table(&catalog, &format!("{warehouse}/t"), schema).await;
    let o_type = column_type(&table, 1);
    let i_type = struct_fields(&o_type)[1].data_type().clone();
    let i = struct_array(&i_type, vec![ints(&[Some(5), None, None])], &[
        true, false, false,
    ]);
    let o = struct_array(&o_type, vec![ints(&[Some(1), Some(2), None]), i], &[
        true, true, false,
    ]);
    let file = write_file(&table, &[1, 2, 3], o).await;
    let table = append(&catalog, &table, file).await;
    let table = drop_column(&catalog, &table, "o.i.x").await;

    let batch = scan(&table, None).await;
    assert_eq!(id_values(&batch), vec![1, 2, 3]);
    let o = batch
        .column_by_name("o")
        .expect("o present")
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("o is a struct");
    assert_eq!(validity(o), vec![true, true, false]);
    let a = o
        .column_by_name("a")
        .expect("o.a present")
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("o.a is an int column");
    assert_eq!(a.value(0), 1);
    assert_eq!(a.value(1), 2);
    let i = o.column_by_name("i").expect("o.i present");
    assert_eq!(i.data_type(), &DataType::Struct(Fields::empty()));
    assert_eq!(validity(i.as_ref()), vec![true, false, false]);

    let table = drop_column(&catalog, &table, "o.a").await;
    let batch = scan(&table, None).await;
    let o = batch
        .column_by_name("o")
        .expect("o present")
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("o is a struct");
    assert_eq!(o.num_columns(), 1);
    assert_eq!(validity(o), vec![true, true, false]);
    let i = o.column_by_name("i").expect("o.i present");
    assert_eq!(i.data_type(), &DataType::Struct(Fields::empty()));
    assert_eq!(validity(i.as_ref()), vec![true, false, false]);
}

#[tokio::test]
async fn list_of_struct_with_children_dropped_keeps_elements() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let catalog = load_catalog(&warehouse).await;
    let element = NestedField::list_element(3, struct_type(vec![int_field(4, "k")]), false);
    let schema = table_schema(NestedField::optional(
        2,
        "l",
        Type::List(ListType {
            element_field: element.into(),
        }),
    ));
    let table = create_table(&catalog, &format!("{warehouse}/t"), schema).await;
    let l_type = column_type(&table, 1);
    let element_field = list_element(&l_type);
    let values = struct_array(
        element_field.data_type(),
        vec![ints(&[Some(1), Some(2)])],
        &[true, true],
    );
    let l = ListArray::try_new(
        element_field,
        OffsetBuffer::new(vec![0, 2, 2, 2].into()),
        values,
        Some(NullBuffer::from(vec![true, true, false])),
    )
    .expect("build the fixture list");
    let file = write_file(&table, &[1, 2, 3], Arc::new(l)).await;
    let table = append(&catalog, &table, file).await;
    let table = drop_column(&catalog, &table, "l.element.k").await;

    let batch = scan(&table, None).await;
    assert_eq!(id_values(&batch), vec![1, 2, 3]);
    let l = batch
        .column_by_name("l")
        .expect("l present")
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("l is a list");
    assert_eq!(validity(l), vec![true, true, false]);
    assert_eq!(l.value_length(0), 2);
    assert_eq!(l.value_length(1), 0);
    assert_eq!(l.values().data_type(), &DataType::Struct(Fields::empty()));
    assert_eq!(validity(l.values().as_ref()), vec![true, true]);
}

#[tokio::test]
async fn map_of_struct_with_children_dropped_keeps_entries() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let catalog = load_catalog(&warehouse).await;
    let map = MapType::new(
        NestedField::map_key_element(3, Type::Primitive(PrimitiveType::Int)).into(),
        NestedField::map_value_element(4, struct_type(vec![int_field(5, "v")]), false).into(),
    );
    let schema = table_schema(NestedField::optional(2, "m", Type::Map(map)));
    let table = create_table(&catalog, &format!("{warehouse}/t"), schema).await;
    let m_type = column_type(&table, 1);
    let DataType::Map(entries_field, ordered) = &m_type else {
        panic!("m is a map type");
    };
    let entries_fields = struct_fields(entries_field.data_type());
    let value_type = entries_fields[1].data_type().clone();
    let entries = StructArray::try_new(
        entries_fields,
        vec![
            ints(&[Some(7)]),
            struct_array(&value_type, vec![ints(&[Some(1)])], &[true]),
        ],
        None,
    )
    .expect("build the fixture entries");
    let m = MapArray::try_new(
        entries_field.clone(),
        OffsetBuffer::new(vec![0, 1, 1].into()),
        entries,
        Some(NullBuffer::from(vec![true, false])),
        *ordered,
    )
    .expect("build the fixture map");
    let file = write_file(&table, &[1, 2], Arc::new(m)).await;
    let table = append(&catalog, &table, file).await;
    let table = drop_column(&catalog, &table, "m.value.v").await;

    let batch = scan(&table, None).await;
    assert_eq!(id_values(&batch), vec![1, 2]);
    let m = batch
        .column_by_name("m")
        .expect("m present")
        .as_any()
        .downcast_ref::<MapArray>()
        .expect("m is a map");
    assert_eq!(validity(m), vec![true, false]);
    assert_eq!(m.value_length(0), 1);
    assert_eq!(m.values().data_type(), &DataType::Struct(Fields::empty()));
    assert_eq!(validity(m.values().as_ref()), vec![true]);
}

#[tokio::test]
async fn writing_a_struct_with_no_children_refuses_without_panic() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let catalog = load_catalog(&warehouse).await;
    let table = one_child_table(&catalog, &format!("{warehouse}/t")).await;
    let table = drop_column(&catalog, &table, "s.k").await;
    let schema = table.metadata().current_schema().clone();
    let arrow_schema =
        Arc::new(schema_to_arrow_schema(&schema).expect("iceberg schema converts to arrow"));
    let s = struct_array(&column_type(&table, 1), vec![], &[true]);
    let batch = RecordBatch::try_new(arrow_schema, vec![ints(&[Some(2)]), s])
        .expect("build the post-drop batch");
    let file_path = format!("{}/data/post-drop.parquet", table.metadata().location());
    let output = table.file_io().new_output(file_path).expect("new output");
    let builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema,
    );
    let mut writer = builder.build(output).await.expect("build parquet writer");
    let outcome = writer.write(&batch).await;
    assert!(
        outcome.is_err(),
        "a Parquet file cannot hold a struct with no children"
    );
}
