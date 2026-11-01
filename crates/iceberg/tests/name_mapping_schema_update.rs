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
    Array, ArrayRef, Int32Array, ListArray, RecordBatch, StringArray, StructArray, UInt32Array,
};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};
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
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use parquet::arrow::ArrowWriter;
use serde_json::{Value, json};
use tempfile::TempDir;

fn int_field(id: i32, name: &str) -> NestedField {
    NestedField::optional(id, name, Type::Primitive(PrimitiveType::Int))
}

fn struct_type(fields: Vec<NestedField>) -> Type {
    Type::Struct(StructType::new(fields.into_iter().map(Arc::new).collect()))
}

fn schema_of(columns: Vec<NestedField>) -> Schema {
    let mut fields = vec![Arc::new(NestedField::required(
        1,
        "id",
        Type::Primitive(PrimitiveType::Int),
    ))];
    fields.extend(columns.into_iter().map(Arc::new));
    Schema::builder()
        .with_schema_id(0)
        .with_fields(fields)
        .build()
        .expect("build the test schema")
}

fn s_of_d() -> Schema {
    schema_of(vec![NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "d")]),
    )])
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
        .name("mapped_table".to_string())
        .location(format!("{warehouse}/t"))
        .schema(schema)
        .partition_spec(UnboundPartitionSpec::builder().build())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create mapped_table");
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

fn list_of(values: ArrayRef, offsets: Vec<i32>) -> ArrayRef {
    let item = Arc::new(Field::new("item", values.data_type().clone(), true));
    Arc::new(
        ListArray::try_new(item, OffsetBuffer::new(offsets.into()), values, None)
            .expect("build the id-less fixture list"),
    )
}

fn write_idless(table: &Table, ids: &[i32], columns: Vec<(&str, ArrayRef)>) -> DataFile {
    let mut fields = vec![Field::new("id", DataType::Int32, false)];
    let id_values: Vec<Option<i32>> = ids.iter().copied().map(Some).collect();
    let mut arrays = vec![ints(&id_values)];
    for (name, array) in columns {
        fields.push(Field::new(name, array.data_type().clone(), true));
        arrays.push(array);
    }
    let schema = Arc::new(ArrowSchema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).expect("build the id-less batch");
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
        .record_count(u64::try_from(ids.len()).expect("row count fits u64"))
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

async fn install_mapping(catalog: &impl Catalog, table: &Table) -> Table {
    let mapping = create_name_mapping(table.metadata().current_schema())
        .expect("build the mapping of the current schema");
    let tx = Transaction::new(table);
    let tx = tx
        .update_table_properties()
        .set(
            TableProperties::PROPERTY_DEFAULT_NAME_MAPPING.to_string(),
            serde_json::to_string(&mapping).expect("serialize the mapping"),
        )
        .apply(tx)
        .expect("apply the name mapping property");
    tx.commit(catalog).await.expect("commit the name mapping")
}

fn mapping_property(table: &Table) -> Option<String> {
    table
        .metadata()
        .properties()
        .get(TableProperties::PROPERTY_DEFAULT_NAME_MAPPING)
        .cloned()
}

fn mapping_json(table: &Table) -> Value {
    let property = mapping_property(table).expect("the table carries a name mapping");
    serde_json::from_str(&property).expect("the name mapping property is JSON")
}

fn mapped(id: i32, names: &[&str], fields: Vec<Value>) -> Value {
    if fields.is_empty() {
        json!({"field-id": id, "names": names})
    } else {
        json!({"field-id": id, "names": names, "fields": fields})
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

async fn scan_where(table: &Table, filter: Option<Predicate>) -> RecordBatch {
    let mut builder = table.scan();
    if let Some(filter) = filter {
        builder = builder.with_filter(filter);
    }
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
    let batch = arrow_select::concat::concat_batches(&schema, &batches).expect("concat batches");
    let mut order: Vec<u32> = (0..batch.num_rows())
        .map(|row| u32::try_from(row).expect("row index fits u32"))
        .collect();
    let ids = ids_of(&batch);
    order.sort_by_key(|row| ids[usize::try_from(*row).expect("row index fits usize")]);
    arrow_select::take::take_record_batch(&batch, &UInt32Array::from(order))
        .expect("order the rows by id")
}

async fn scan(table: &Table) -> RecordBatch {
    scan_where(table, None).await
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

fn as_struct(array: &dyn Array) -> &StructArray {
    array
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("column is a StructArray")
}

fn struct_column<'a>(batch: &'a RecordBatch, name: &str) -> &'a StructArray {
    as_struct(
        batch
            .column_by_name(name)
            .unwrap_or_else(|| panic!("column {name} present"))
            .as_ref(),
    )
}

fn int_values(array: &dyn Array) -> Vec<Option<i32>> {
    array
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("an int column")
        .iter()
        .collect()
}

fn child_ints(parent: &StructArray, name: &str) -> Vec<Option<i32>> {
    int_values(
        parent
            .column_by_name(name)
            .unwrap_or_else(|| panic!("child {name} present"))
            .as_ref(),
    )
}

fn validity(array: &dyn Array) -> Vec<bool> {
    (0..array.len()).map(|row| array.is_valid(row)).collect()
}

#[tokio::test]
async fn rename_of_a_nested_child_maps_the_new_name_and_keeps_the_old_one() {
    let tmp = TempDir::new().expect("temp dir");
    let (catalog, table) = new_table(&tmp, s_of_d()).await;
    let first = write_idless(&table, &[1], vec![(
        "s",
        plain_struct(vec![("d", ints(&[Some(1)]))]),
    )]);
    let table = append(&catalog, &table, first).await;
    let table = install_mapping(&catalog, &table).await;
    let (id, s, d) = (
        field_id(&table, "id"),
        field_id(&table, "s"),
        field_id(&table, "s.d"),
    );
    assert_eq!(
        mapping_json(&table),
        json!([
            mapped(id, &["id"], vec![]),
            mapped(s, &["s"], vec![mapped(d, &["d"], vec![])]),
        ])
    );

    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .rename_column("s.d", "v")
        .apply(tx)
        .expect("apply the rename");
    let table = tx.commit(&catalog).await.expect("commit the rename");

    let second = write_idless(&table, &[2], vec![(
        "s",
        plain_struct(vec![("v", ints(&[Some(2)]))]),
    )]);
    let table = append(&catalog, &table, second).await;
    let batch = scan(&table).await;
    assert_eq!(ids_of(&batch), vec![1, 2]);
    assert_eq!(child_ints(struct_column(&batch, "s"), "v"), vec![
        Some(1),
        Some(2)
    ]);

    let filter = Reference::new("s.v").equal_to(Datum::int(2));
    let filtered = scan_where(&table, Some(filter)).await;
    assert_eq!(ids_of(&filtered), vec![2]);
    assert_eq!(
        mapping_json(&table),
        json!([
            mapped(id, &["id"], vec![]),
            mapped(s, &["s"], vec![mapped(d, &["d", "v"], vec![])]),
        ])
    );
}

#[tokio::test]
async fn added_nested_child_maps_to_its_new_id() {
    let tmp = TempDir::new().expect("temp dir");
    let (catalog, table) = new_table(&tmp, s_of_d()).await;
    let first = write_idless(&table, &[1], vec![(
        "s",
        plain_struct(vec![("d", ints(&[Some(1)]))]),
    )]);
    let table = append(&catalog, &table, first).await;
    let table = install_mapping(&catalog, &table).await;

    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .add_column_to(Some("s"), "k", Type::Primitive(PrimitiveType::Int), None)
        .apply(tx)
        .expect("apply the nested add");
    let table = tx.commit(&catalog).await.expect("commit the nested add");
    let (id, s, d, k) = (
        field_id(&table, "id"),
        field_id(&table, "s"),
        field_id(&table, "s.d"),
        field_id(&table, "s.k"),
    );

    let with_k = write_idless(&table, &[2], vec![(
        "s",
        plain_struct(vec![("d", ints(&[Some(2)])), ("k", ints(&[Some(22)]))]),
    )]);
    let table = append(&catalog, &table, with_k).await;
    let without_k = write_idless(&table, &[3], vec![(
        "s",
        plain_struct(vec![("d", ints(&[Some(3)]))]),
    )]);
    let table = append(&catalog, &table, without_k).await;
    let batch = scan(&table).await;
    assert_eq!(ids_of(&batch), vec![1, 2, 3]);
    let s_values = struct_column(&batch, "s");
    assert_eq!(child_ints(s_values, "d"), vec![Some(1), Some(2), Some(3)]);
    assert_eq!(child_ints(s_values, "k"), vec![None, Some(22), None]);
    assert_eq!(
        mapping_json(&table),
        json!([
            mapped(id, &["id"], vec![]),
            mapped(s, &["s"], vec![
                mapped(d, &["d"], vec![]),
                mapped(k, &["k"], vec![])
            ]),
        ])
    );
}

#[tokio::test]
async fn added_top_level_column_maps_to_its_new_id() {
    let tmp = TempDir::new().expect("temp dir");
    let (catalog, table) = new_table(&tmp, s_of_d()).await;
    let first = write_idless(&table, &[1], vec![(
        "s",
        plain_struct(vec![("d", ints(&[Some(1)]))]),
    )]);
    let table = append(&catalog, &table, first).await;
    let table = install_mapping(&catalog, &table).await;
    let before = mapping_json(&table);

    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .add_column("y", Type::Primitive(PrimitiveType::Int))
        .apply(tx)
        .expect("apply the top-level add");
    let table = tx.commit(&catalog).await.expect("commit the top-level add");
    let mut expected = before.as_array().expect("the mapping is an array").clone();
    expected.push(mapped(field_id(&table, "y"), &["y"], vec![]));

    let second = write_idless(&table, &[2], vec![
        ("s", plain_struct(vec![("d", ints(&[Some(2)]))])),
        ("y", ints(&[Some(7)])),
    ]);
    let table = append(&catalog, &table, second).await;
    let batch = scan(&table).await;
    assert_eq!(ids_of(&batch), vec![1, 2]);
    let y = batch.column_by_name("y").expect("y column present");
    assert_eq!(int_values(y.as_ref()), vec![None, Some(7)]);
    assert_eq!(mapping_json(&table), Value::Array(expected));
}

async fn evolve_drop(catalog: &impl Catalog, table: &Table, name: &str) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .update_schema()
        .delete_column(name)
        .apply(tx)
        .expect("apply the drop");
    tx.commit(catalog).await.expect("commit the drop")
}

#[tokio::test]
async fn dropped_then_re_added_name_moves_to_the_new_id() {
    let tmp = TempDir::new().expect("temp dir");
    let schema = schema_of(vec![
        NestedField::optional(
            2,
            "s",
            struct_type(vec![int_field(3, "a"), int_field(4, "b")]),
        ),
        int_field(5, "c"),
    ]);
    let (catalog, table) = new_table(&tmp, schema).await;
    let file = write_idless(&table, &[1], vec![
        (
            "s",
            plain_struct(vec![("a", ints(&[Some(1)])), ("b", ints(&[Some(2)]))]),
        ),
        ("c", ints(&[Some(5)])),
    ]);
    let table = append(&catalog, &table, file).await;
    let table = install_mapping(&catalog, &table).await;
    let (id, s, a, old_b, old_c) = (
        field_id(&table, "id"),
        field_id(&table, "s"),
        field_id(&table, "s.a"),
        field_id(&table, "s.b"),
        field_id(&table, "c"),
    );

    let table = evolve_drop(&catalog, &table, "s.b").await;
    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .add_column_to(Some("s"), "b", Type::Primitive(PrimitiveType::Int), None)
        .apply(tx)
        .expect("apply the nested re-add");
    let table = tx.commit(&catalog).await.expect("commit the nested re-add");
    let table = evolve_drop(&catalog, &table, "c").await;
    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .add_column("c", Type::Primitive(PrimitiveType::Int))
        .apply(tx)
        .expect("apply the top-level re-add");
    let table = tx
        .commit(&catalog)
        .await
        .expect("commit the top-level re-add");
    let (new_b, new_c) = (field_id(&table, "s.b"), field_id(&table, "c"));
    assert_ne!(new_b, old_b);
    assert_ne!(new_c, old_c);

    let batch = scan(&table).await;
    assert_eq!(ids_of(&batch), vec![1]);
    let s_values = struct_column(&batch, "s");
    assert_eq!(child_ints(s_values, "a"), vec![Some(1)]);
    assert_eq!(child_ints(s_values, "b"), vec![Some(2)]);
    let c = batch.column_by_name("c").expect("c column present");
    assert_eq!(int_values(c.as_ref()), vec![Some(5)]);
    assert_eq!(
        mapping_json(&table),
        json!([
            mapped(id, &["id"], vec![]),
            mapped(s, &["s"], vec![
                mapped(a, &["a"], vec![]),
                mapped(old_b, &[], vec![]),
                mapped(new_b, &["b"], vec![]),
            ]),
            mapped(old_c, &[], vec![]),
            mapped(new_c, &["c"], vec![]),
        ])
    );
}

#[tokio::test]
async fn added_struct_column_maps_its_children() {
    let tmp = TempDir::new().expect("temp dir");
    let (catalog, table) = new_table(&tmp, s_of_d()).await;
    let first = write_idless(&table, &[1], vec![(
        "s",
        plain_struct(vec![("d", ints(&[Some(1)]))]),
    )]);
    let table = append(&catalog, &table, first).await;
    let table = install_mapping(&catalog, &table).await;
    let before = mapping_json(&table);

    let t = struct_type(vec![
        int_field(100, "p"),
        NestedField::optional(101, "q", Type::Primitive(PrimitiveType::String)),
    ]);
    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .add_column("t", t)
        .apply(tx)
        .expect("apply the struct add");
    let table = tx.commit(&catalog).await.expect("commit the struct add");
    let mut expected = before.as_array().expect("the mapping is an array").clone();
    expected.push(mapped(field_id(&table, "t"), &["t"], vec![
        mapped(field_id(&table, "t.p"), &["p"], vec![]),
        mapped(field_id(&table, "t.q"), &["q"], vec![]),
    ]));

    let second = write_idless(&table, &[2], vec![
        ("s", plain_struct(vec![("d", ints(&[Some(2)]))])),
        (
            "t",
            plain_struct(vec![("p", ints(&[Some(9)])), ("q", strings(&[Some("x")]))]),
        ),
    ]);
    let table = append(&catalog, &table, second).await;
    let batch = scan(&table).await;
    assert_eq!(ids_of(&batch), vec![1, 2]);
    let t = struct_column(&batch, "t");
    assert_eq!(validity(t), vec![false, true]);
    assert_eq!(child_ints(t, "p")[1], Some(9));
    let q = t
        .column_by_name("q")
        .expect("child q present")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("q is a string column")
        .iter()
        .map(|value| value.map(str::to_string))
        .collect::<Vec<_>>();
    assert_eq!(q[1], Some("x".to_string()));
    assert_eq!(mapping_json(&table), Value::Array(expected));
}

#[tokio::test]
async fn added_child_of_a_list_element_maps_under_the_element() {
    let tmp = TempDir::new().expect("temp dir");
    let schema = schema_of(vec![NestedField::optional(
        2,
        "l",
        Type::List(ListType::new(Arc::new(NestedField::list_element(
            3,
            struct_type(vec![int_field(4, "a")]),
            false,
        )))),
    )]);
    let (catalog, table) = new_table(&tmp, schema).await;
    let first = write_idless(&table, &[1], vec![(
        "l",
        list_of(plain_struct(vec![("a", ints(&[Some(1)]))]), vec![0, 1]),
    )]);
    let table = append(&catalog, &table, first).await;
    let table = install_mapping(&catalog, &table).await;

    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .add_column_to(Some("l"), "b", Type::Primitive(PrimitiveType::Int), None)
        .apply(tx)
        .expect("apply the element add");
    let table = tx.commit(&catalog).await.expect("commit the element add");

    let second = write_idless(&table, &[2], vec![(
        "l",
        list_of(
            plain_struct(vec![("a", ints(&[Some(2)])), ("b", ints(&[Some(20)]))]),
            vec![0, 1],
        ),
    )]);
    let table = append(&catalog, &table, second).await;
    let batch = scan(&table).await;
    assert_eq!(ids_of(&batch), vec![1, 2]);
    let list = batch
        .column_by_name("l")
        .expect("l column present")
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("l is a ListArray");
    let elements = as_struct(list.values().as_ref());
    assert_eq!(child_ints(elements, "a"), vec![Some(1), Some(2)]);
    assert_eq!(child_ints(elements, "b"), vec![None, Some(20)]);
    assert_eq!(
        mapping_json(&table),
        json!([
            mapped(field_id(&table, "id"), &["id"], vec![]),
            mapped(field_id(&table, "l"), &["l"], vec![mapped(
                field_id(&table, "l.element"),
                &["element"],
                vec![
                    mapped(field_id(&table, "l.element.a"), &["a"], vec![]),
                    mapped(field_id(&table, "l.element.b"), &["b"], vec![]),
                ]
            )]),
        ])
    );
}

#[tokio::test]
async fn table_without_a_mapping_gets_none_from_a_schema_update() {
    let tmp = TempDir::new().expect("temp dir");
    let (catalog, table) = new_table(&tmp, s_of_d()).await;
    assert_eq!(mapping_property(&table), None);
    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .rename_column("s.d", "v")
        .add_column("y", Type::Primitive(PrimitiveType::Int))
        .apply(tx)
        .expect("apply the evolution");
    let table = tx.commit(&catalog).await.expect("commit the evolution");
    assert!(
        table
            .metadata()
            .current_schema()
            .field_by_name("s.v")
            .is_some()
    );
    assert_eq!(mapping_property(&table), None);
}

#[tokio::test]
async fn type_promotion_and_move_leave_the_mapping_unchanged() {
    let tmp = TempDir::new().expect("temp dir");
    let (catalog, table) = new_table(&tmp, s_of_d()).await;
    let table = install_mapping(&catalog, &table).await;
    let before = mapping_property(&table).expect("the mapping is installed");

    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .update_column("s.d", PrimitiveType::Long)
        .apply(tx)
        .expect("apply the promotion");
    let table = tx.commit(&catalog).await.expect("commit the promotion");
    let d = table
        .metadata()
        .current_schema()
        .field_by_name("s.d")
        .expect("s.d present");
    assert_eq!(*d.field_type, Type::Primitive(PrimitiveType::Long));
    assert_eq!(mapping_property(&table), Some(before.clone()));

    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .move_first("s")
        .apply(tx)
        .expect("apply the move");
    let table = tx.commit(&catalog).await.expect("commit the move");
    let first = &table.metadata().current_schema().as_struct().fields()[0];
    assert_eq!(first.name, "s");
    assert_eq!(mapping_property(&table), Some(before));
}
