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
    Array, ArrayRef, Decimal128Array, Int32Array, LargeBinaryArray, RecordBatch, StringArray,
};
use arrow_schema::{Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::expr::{Predicate, Reference};
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, FormatVersion, NestedField,
    PrimitiveType, Schema, SortOrder, Struct, TableProperties, Type, UnboundPartitionSpec,
    create_name_mapping,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

fn field(id: i32, name: &str, primitive: PrimitiveType) -> NestedField {
    NestedField::optional(id, name, Type::Primitive(primitive))
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

fn data_file(path: String, rows: u64) -> DataFile {
    let size = std::fs::metadata(&path)
        .expect("stat the parquet file")
        .len();
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path)
        .file_format(DataFileFormat::Parquet)
        .partition(Struct::empty())
        .partition_spec_id(0)
        .record_count(rows)
        .file_size_in_bytes(size)
        .build()
        .expect("build the data file")
}

fn file_path(table: &Table) -> String {
    let dir = format!("{}/data", table.metadata().location());
    std::fs::create_dir_all(&dir).expect("create the data directory");
    format!("{dir}/idless-{}.parquet", uuid::Uuid::now_v7())
}

fn write_idless(table: &Table, columns: Vec<(&str, ArrayRef)>) -> DataFile {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
        .collect();
    let arrays: Vec<ArrayRef> = columns.into_iter().map(|(_, array)| array).collect();
    let batch = RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), arrays)
        .expect("build the id-less batch");
    let path = file_path(table);
    let file = std::fs::File::create(&path).expect("create the id-less parquet file");
    let mut writer =
        ArrowWriter::try_new(file, batch.schema(), None).expect("build the arrow writer");
    writer.write(&batch).expect("write the id-less batch");
    writer.close().expect("close the arrow writer");
    data_file(
        path,
        u64::try_from(batch.num_rows()).expect("row count fits u64"),
    )
}

fn write_non_utf8_string_file(table: &Table) -> DataFile {
    use parquet::basic::{LogicalType, Repetition, Type as PhysicalType};
    use parquet::data_type::{ByteArray, ByteArrayType, Int32Type};
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::types::Type as SchemaType;

    let k = SchemaType::primitive_type_builder("k", PhysicalType::INT32)
        .with_repetition(Repetition::OPTIONAL)
        .build()
        .expect("the int column");
    let v = SchemaType::primitive_type_builder("v", PhysicalType::BYTE_ARRAY)
        .with_repetition(Repetition::OPTIONAL)
        .with_logical_type(Some(LogicalType::String))
        .build()
        .expect("the string column");
    let schema = SchemaType::group_type_builder("schema")
        .with_fields(vec![Arc::new(k), Arc::new(v)])
        .build()
        .expect("the file schema");
    let path = file_path(table);
    let file = std::fs::File::create(&path).expect("create the parquet file");
    let mut writer = SerializedFileWriter::new(file, Arc::new(schema), Default::default())
        .expect("build the parquet writer");
    let mut row_group = writer.next_row_group().expect("a row group");
    let mut column = row_group
        .next_column()
        .expect("the k column")
        .expect("k is present");
    column
        .typed::<Int32Type>()
        .write_batch(&[1], Some(&[1]), None)
        .expect("write k");
    column.close().expect("close k");
    let mut column = row_group
        .next_column()
        .expect("the v column")
        .expect("v is present");
    column
        .typed::<ByteArrayType>()
        .write_batch(&[ByteArray::from(vec![0xFF, 0xFE, 0x61])], Some(&[1]), None)
        .expect("write v");
    column.close().expect("close v");
    row_group.close().expect("close the row group");
    writer.close().expect("close the parquet writer");
    data_file(path, 1)
}

async fn table_with_file(
    tmp: &TempDir,
    schema: Schema,
    write: impl FnOnce(&Table) -> DataFile,
    mapped: bool,
) -> Table {
    let (catalog, table) = new_table(tmp, schema).await;
    let file = write(&table);
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");
    if !mapped {
        return table;
    }
    let mapping = create_name_mapping(table.metadata().current_schema()).expect("the mapping");
    let tx = Transaction::new(&table);
    let tx = tx
        .update_table_properties()
        .set(
            TableProperties::PROPERTY_DEFAULT_NAME_MAPPING.to_string(),
            serde_json::to_string(&mapping).expect("serialize the mapping"),
        )
        .apply(tx)
        .expect("apply the name mapping property");
    tx.commit(&catalog).await.expect("commit the name mapping")
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

fn ks(batches: &[RecordBatch]) -> Vec<i32> {
    let mut ks: Vec<i32> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name("k")
                .expect("k present")
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("k is an int column")
                .iter()
                .flatten()
                .collect::<Vec<_>>()
        })
        .collect();
    ks.sort_unstable();
    ks
}

async fn filtered_ks(table: &Table, filter: Predicate) -> Vec<i32> {
    let shown = filter.to_string();
    let batches = scan_result(table, Some(&["k"]), Some(filter))
        .await
        .unwrap_or_else(|e| panic!("WHERE {shown} reads: {e}"));
    ks(&batches)
}

async fn filtered_count(table: &Table, filter: Predicate) -> usize {
    let shown = filter.to_string();
    scan_result(table, Some(&[]), Some(filter))
        .await
        .unwrap_or_else(|e| panic!("count(*) WHERE {shown} reads: {e}"))
        .iter()
        .map(RecordBatch::num_rows)
        .sum()
}

async fn assert_refuses(
    table: &Table,
    columns: Option<&[&str]>,
    filter: Option<Predicate>,
    needles: &[impl AsRef<str>],
) {
    let shown = filter.as_ref().map(ToString::to_string);
    let error = match scan_result(table, columns, filter).await {
        Ok(batches) => panic!(
            "the scan WHERE {shown:?} read k = {:?} instead of refusing",
            ks_or_rows(&batches)
        ),
        Err(error) => error,
    };
    let message = error.to_string();
    assert_eq!(error.kind(), ErrorKind::DataInvalid, "{message}");
    for needle in needles {
        let needle = needle.as_ref();
        assert!(message.contains(needle), "missing {needle:?} in {message}");
    }
}

fn ks_or_rows(batches: &[RecordBatch]) -> String {
    match batches.first().and_then(|batch| batch.column_by_name("k")) {
        Some(_) => format!("{:?}", ks(batches)),
        None => format!(
            "{} rows",
            batches.iter().map(RecordBatch::num_rows).sum::<usize>()
        ),
    }
}

fn with_op(needles: [String; 6]) -> Vec<String> {
    let [column, position, file, id, table, op] = needles;
    vec![
        column,
        position,
        file,
        id,
        table,
        format!("filtered with `{op}`"),
    ]
}

fn three_rows(values: ArrayRef) -> Vec<(&'static str, ArrayRef)> {
    vec![
        (
            "k",
            Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(3)])) as ArrayRef,
        ),
        ("v", values),
    ]
}

async fn binary_read_as_string(tmp: &TempDir) -> Table {
    let schema = schema_of(vec![
        field(1, "k", PrimitiveType::Int),
        field(2, "v", PrimitiveType::String),
    ]);
    let bytes: ArrayRef = Arc::new(arrow_array::BinaryArray::from(vec![
        Some(b"abc".as_ref()),
        Some(b"xyz".as_ref()),
        None,
    ]));
    table_with_file(
        tmp,
        schema,
        |table| write_idless(table, three_rows(bytes)),
        false,
    )
    .await
}

#[tokio::test]
async fn a_binary_column_read_as_string_answers_null_tests_a_prefix_an_inequality_and_a_count() {
    let tmp = TempDir::new().expect("temp dir");
    let table = binary_read_as_string(&tmp).await;
    let v = || Reference::new("v");
    assert_eq!(filtered_ks(&table, v().is_not_null()).await, vec![1, 2]);
    assert_eq!(filtered_ks(&table, v().is_null()).await, vec![3]);
    assert_eq!(
        filtered_ks(&table, v().starts_with(Datum::string("a"))).await,
        vec![1]
    );
    let not_abc = filtered_ks(&table, v().not_equal_to(Datum::string("abc"))).await;
    assert!(not_abc.contains(&2) && !not_abc.contains(&1), "{not_abc:?}");
    assert_eq!(filtered_count(&table, v().is_not_null()).await, 2);
}

#[tokio::test]
async fn a_binary_column_read_as_string_refuses_comparisons() {
    let tmp = TempDir::new().expect("temp dir");
    let table = binary_read_as_string(&tmp).await;
    let v = || Reference::new("v");
    let needles = |op: &'static str| ["'v'", "position 1", "binary", "id 2", "string", op];
    let needles = |op: &'static str| needles(op).map(str::to_string);
    let eq = v().equal_to(Datum::string("abc"));
    assert_refuses(&table, Some(&["k"]), Some(eq), &with_op(needles("="))).await;
    let gt = v().greater_than(Datum::string("b"));
    assert_refuses(&table, Some(&["k"]), Some(gt), &with_op(needles(">"))).await;
    let within = v().is_in([Datum::string("abc"), Datum::string("xyz")]);
    assert_refuses(&table, Some(&["k"]), Some(within), &with_op(needles("IN"))).await;
}

async fn string_read_as_binary(tmp: &TempDir) -> Table {
    let schema = schema_of(vec![
        field(1, "k", PrimitiveType::Int),
        field(2, "v", PrimitiveType::Binary),
    ]);
    let text: ArrayRef = Arc::new(StringArray::from(vec![Some("abc"), Some("xyz"), None]));
    table_with_file(
        tmp,
        schema,
        |table| write_idless(table, three_rows(text)),
        false,
    )
    .await
}

#[tokio::test]
async fn a_string_column_read_as_binary_refuses_comparisons() {
    let tmp = TempDir::new().expect("temp dir");
    let table = string_read_as_binary(&tmp).await;
    let v = || Reference::new("v");
    let needles = |op: &'static str| ["'v'", "position 1", "string", "id 2", "binary", op];
    let needles = |op: &'static str| needles(op).map(str::to_string);
    let eq = || v().equal_to(Datum::binary(b"abc".to_vec()));
    assert_refuses(&table, Some(&["k"]), Some(eq()), &with_op(needles("="))).await;
    let gt = v().greater_than(Datum::binary(b"b".to_vec()));
    assert_refuses(&table, Some(&["k"]), Some(gt), &with_op(needles(">"))).await;
    let within = v().is_in([
        Datum::binary(b"abc".to_vec()),
        Datum::binary(b"xyz".to_vec()),
    ]);
    assert_refuses(&table, Some(&["k"]), Some(within), &with_op(needles("IN"))).await;
    assert_refuses(&table, Some(&[]), Some(eq()), &with_op(needles("="))).await;
}

#[tokio::test]
async fn a_string_column_read_as_binary_answers_a_null_test_and_an_inequality() {
    let tmp = TempDir::new().expect("temp dir");
    let table = string_read_as_binary(&tmp).await;
    let v = || Reference::new("v");
    assert_eq!(filtered_ks(&table, v().is_not_null()).await, vec![1, 2]);
    let not_abc = filtered_ks(&table, v().not_equal_to(Datum::binary(b"abc".to_vec()))).await;
    assert!(not_abc.contains(&2) && !not_abc.contains(&1), "{not_abc:?}");
}

async fn overflowing_decimal(tmp: &TempDir, file_precision: u8, unscaled: i128) -> Table {
    let schema = schema_of(vec![
        field(1, "k", PrimitiveType::Int),
        field(2, "d", PrimitiveType::Decimal {
            precision: 10,
            scale: 2,
        }),
    ]);
    let decimals: ArrayRef = Arc::new(
        Decimal128Array::from(vec![Some(unscaled), Some(12345)])
            .with_precision_and_scale(file_precision, 2)
            .expect("a valid decimal type"),
    );
    let columns = vec![
        (
            "k",
            Arc::new(Int32Array::from(vec![Some(1), Some(2)])) as ArrayRef,
        ),
        ("d", decimals),
    ];
    table_with_file(tmp, schema, |table| write_idless(table, columns), false).await
}

async fn assert_overflow_is_loud(table: &Table) {
    let read = scan_result(table, None, None).await;
    assert!(
        read.is_err(),
        "the read answered {:?}",
        read.map(|b| ks(&b))
    );
    let d = || Reference::new("d");
    let filters = [
        d().is_not_null(),
        d().greater_than(Datum::decimal_from_str("0.00").expect("a decimal literal")),
        d().equal_to(Datum::decimal_from_str("123.45").expect("a decimal literal")),
    ];
    for filter in filters {
        assert_refuses(table, Some(&["k"]), Some(filter), &[""; 0]).await;
    }
}

#[tokio::test]
async fn an_overflowing_narrowed_decimal_is_loud_under_a_filter() {
    let tmp = TempDir::new().expect("temp dir");
    let table = overflowing_decimal(&tmp, 12, 123_456_789_012).await;
    assert_overflow_is_loud(&table).await;
}

#[tokio::test]
async fn an_overflowing_narrowed_wide_decimal_is_loud_under_a_filter() {
    let tmp = TempDir::new().expect("temp dir");
    let table = overflowing_decimal(&tmp, 20, 12_345_678_901_234_567).await;
    assert_overflow_is_loud(&table).await;
}

async fn non_utf8_string_read_as_binary(mapped: bool) {
    let tmp = TempDir::new().expect("temp dir");
    let schema = schema_of(vec![
        field(1, "k", PrimitiveType::Int),
        field(2, "v", PrimitiveType::Binary),
    ]);
    let table = table_with_file(&tmp, schema, write_non_utf8_string_file, mapped).await;
    let batches = scan_result(&table, None, None)
        .await
        .expect("the scan reads the raw bytes");
    let v = batches[0]
        .column_by_name("v")
        .expect("v present")
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .expect("v reads as binary");
    assert_eq!(v.value(0), [0xFF, 0xFE, 0x61]);
    assert_eq!(
        filtered_ks(&table, Reference::new("v").is_not_null()).await,
        vec![1]
    );
}

#[tokio::test]
async fn a_non_utf8_string_column_read_as_binary_keeps_its_raw_bytes_no_mapping() {
    non_utf8_string_read_as_binary(false).await;
}

#[tokio::test]
async fn a_non_utf8_string_column_read_as_binary_keeps_its_raw_bytes_mapping() {
    non_utf8_string_read_as_binary(true).await;
}
