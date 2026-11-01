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

use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch, StructArray, UInt32Array};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};
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
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use parquet::arrow::ArrowWriter;
use serde_json::{Value, json};
use tempfile::TempDir;

const NAMESPACE: &str = "pins";

const TABLE: &str = "mapped_table";

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
    let namespace = NamespaceIdent::new(NAMESPACE.to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name(TABLE.to_string())
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

async fn evolve_drop(catalog: &impl Catalog, table: &Table, name: &str) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .update_schema()
        .delete_column(name)
        .apply(tx)
        .expect("apply the drop");
    tx.commit(catalog).await.expect("commit the drop")
}

async fn evolve_rename(catalog: &impl Catalog, table: &Table, name: &str, new_name: &str) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .update_schema()
        .rename_column(name, new_name)
        .apply(tx)
        .expect("apply the rename");
    tx.commit(catalog).await.expect("commit the rename")
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

async fn scan(table: &Table) -> RecordBatch {
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

fn int_values(array: &dyn Array) -> Vec<Option<i32>> {
    array
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("an int column")
        .iter()
        .collect()
}

fn child_ints(batch: &RecordBatch, parent: &str, name: &str) -> Vec<Option<i32>> {
    let parent = batch
        .column_by_name(parent)
        .unwrap_or_else(|| panic!("column {parent} present"))
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("column is a StructArray");
    int_values(
        parent
            .column_by_name(name)
            .unwrap_or_else(|| panic!("child {name} present"))
            .as_ref(),
    )
}

#[tokio::test]
async fn rename_onto_a_dropped_sibling_name_takes_the_name_from_the_sibling() {
    let tmp = TempDir::new().expect("temp dir");
    let schema = schema_of(vec![int_field(2, "x"), int_field(3, "y")]);
    let (catalog, table) = new_table(&tmp, schema).await;
    let first = write_idless(&table, &[1], vec![
        ("x", ints(&[Some(1)])),
        ("y", ints(&[Some(2)])),
    ]);
    let table = append(&catalog, &table, first).await;
    let table = install_mapping(&catalog, &table).await;
    let table = evolve_drop(&catalog, &table, "x").await;
    let table = evolve_rename(&catalog, &table, "y", "x").await;
    assert_eq!(
        mapping_json(&table),
        json!([
            mapped(1, &["id"], vec![]),
            mapped(2, &[], vec![]),
            mapped(3, &["y", "x"], vec![]),
        ])
    );

    let second = write_idless(&table, &[2], vec![("x", ints(&[Some(9)]))]);
    let table = append(&catalog, &table, second).await;
    let batch = scan(&table).await;
    assert_eq!(ids_of(&batch), vec![1, 2]);
    let x = batch.column_by_name("x").expect("x column present");
    assert_eq!(int_values(x.as_ref()), vec![Some(2), Some(9)]);
}

#[tokio::test]
async fn nested_rename_onto_a_dropped_sibling_name_takes_the_name_from_the_sibling() {
    let tmp = TempDir::new().expect("temp dir");
    let schema = schema_of(vec![NestedField::optional(
        2,
        "s",
        struct_type(vec![int_field(3, "x"), int_field(4, "y")]),
    )]);
    let (catalog, table) = new_table(&tmp, schema).await;
    let first = write_idless(&table, &[1], vec![(
        "s",
        plain_struct(vec![("x", ints(&[Some(1)])), ("y", ints(&[Some(2)]))]),
    )]);
    let table = append(&catalog, &table, first).await;
    let table = install_mapping(&catalog, &table).await;
    let table = evolve_drop(&catalog, &table, "s.x").await;
    let table = evolve_rename(&catalog, &table, "s.y", "x").await;
    assert_eq!(
        mapping_json(&table),
        json!([
            mapped(1, &["id"], vec![]),
            mapped(2, &["s"], vec![
                mapped(3, &[], vec![]),
                mapped(4, &["y", "x"], vec![]),
            ]),
        ])
    );

    let second = write_idless(&table, &[2], vec![(
        "s",
        plain_struct(vec![("x", ints(&[Some(9)]))]),
    )]);
    let table = append(&catalog, &table, second).await;
    let batch = scan(&table).await;
    assert_eq!(ids_of(&batch), vec![1, 2]);
    assert_eq!(child_ints(&batch, "s", "x"), vec![Some(2), Some(9)]);
}

#[tokio::test]
async fn rename_onto_a_live_sibling_name_is_refused_and_the_mapping_is_untouched() {
    let tmp = TempDir::new().expect("temp dir");
    let schema = schema_of(vec![int_field(2, "x"), int_field(3, "y")]);
    let (catalog, table) = new_table(&tmp, schema).await;
    let table = install_mapping(&catalog, &table).await;
    let before = mapping_property(&table).expect("the mapping is installed");
    let schema_id = table.metadata().current_schema_id();

    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .rename_column("y", "x")
        .apply(tx)
        .expect("apply the rename");
    let error = tx
        .commit(&catalog)
        .await
        .expect_err("a rename onto a live sibling name is refused");
    assert!(
        error.to_string().contains("multiple fields for name x"),
        "the refusal names the duplicate: {error}"
    );

    let ident = TableIdent::new(
        NamespaceIdent::new(NAMESPACE.to_string()),
        TABLE.to_string(),
    );
    let reloaded = catalog.load_table(&ident).await.expect("reload the table");
    assert_eq!(reloaded.metadata().current_schema_id(), schema_id);
    assert_eq!(mapping_property(&reloaded), Some(before));
}
