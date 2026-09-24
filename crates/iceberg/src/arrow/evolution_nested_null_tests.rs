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

use std::collections::BTreeMap;
use std::fs::File;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Int32Array, Int64Array, ListArray, MapArray, RecordBatch, StringArray,
    StructArray,
};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_cast::display::{ArrayFormatter, FormatOptions};
use arrow_schema::{DataType, Field, Fields};
use futures::TryStreamExt;
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use crate::arrow::{ArrowReaderBuilder, create_primitive_array_repeated, schema_to_arrow_schema};
use crate::io::FileIO;
use crate::scan::{FileScanTask, FileScanTaskStream};
use crate::spec::{
    DataFileFormat, ListType, MapType, NestedField, PrimitiveType, Schema, SchemaRef, StructType,
    Type,
};

const EVOLVED_COLUMNS: [&str; 4] = ["s2", "m2", "a2", "m"];

fn string_type() -> Type {
    Type::Primitive(PrimitiveType::String)
}

fn int_type() -> Type {
    Type::Primitive(PrimitiveType::Int)
}

fn base_fields() -> Vec<Arc<NestedField>> {
    vec![
        NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
        NestedField::optional(2, "data", string_type()).into(),
        NestedField::optional(3, "cat", string_type()).into(),
    ]
}

fn build_schema(fields: Vec<Arc<NestedField>>) -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(fields)
            .build()
            .expect("schema builds"),
    )
}

fn evolved_schema() -> SchemaRef {
    let mut fields = base_fields();
    fields.extend([
        NestedField::optional(
            4,
            "s2",
            Type::Struct(StructType::new(vec![
                NestedField::optional(5, "p", int_type()).into(),
                NestedField::optional(6, "q", string_type()).into(),
            ])),
        )
        .into(),
        NestedField::optional(
            7,
            "m2",
            Type::Map(MapType::new(
                NestedField::map_key_element(8, string_type()).into(),
                NestedField::map_value_element(9, int_type(), false).into(),
            )),
        )
        .into(),
        NestedField::optional(
            10,
            "a2",
            Type::List(ListType::new(
                NestedField::list_element(11, string_type(), false).into(),
            )),
        )
        .into(),
        NestedField::optional(
            12,
            "m",
            Type::Map(MapType::new(
                NestedField::map_key_element(
                    13,
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(14, "a", int_type()).into(),
                    ])),
                )
                .into(),
                NestedField::map_value_element(15, string_type(), false).into(),
            )),
        )
        .into(),
    ]);
    build_schema(fields)
}

fn base_columns(ids: &[i64]) -> Vec<ArrayRef> {
    let data: Vec<String> = ids.iter().map(|id| format!("d{id}")).collect();
    let cat: Vec<String> = ids.iter().map(|id| format!("c{id}")).collect();
    vec![
        Arc::new(Int64Array::from(ids.to_vec())),
        Arc::new(StringArray::from(data)),
        Arc::new(StringArray::from(cat)),
    ]
}

fn write_parquet(dir: &TempDir, name: &str, schema: &Schema, columns: Vec<ArrayRef>) -> String {
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("arrow schema"));
    let batch = RecordBatch::try_new(arrow_schema.clone(), columns).expect("batch builds");
    let path = format!("{}/{name}", dir.path().to_str().expect("utf8"));
    let file = File::create(&path).expect("create file");
    let mut writer = ArrowWriter::try_new(file, arrow_schema, None).expect("writer");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");
    path
}

fn write_pre_alter_file(dir: &TempDir) -> String {
    let schema = build_schema(base_fields());
    write_parquet(dir, "pre_alter.parquet", &schema, base_columns(&[1, 2, 3]))
}

fn evolved_field(name: &str) -> Field {
    let arrow_schema = schema_to_arrow_schema(&evolved_schema()).expect("arrow schema");
    arrow_schema
        .field_with_name(name)
        .expect("evolved field")
        .clone()
}

fn struct_fields(data_type: &DataType) -> Fields {
    match data_type {
        DataType::Struct(fields) => fields.clone(),
        other => panic!("expected struct, got {other}"),
    }
}

fn entries_field(name: &str) -> Arc<Field> {
    match evolved_field(name).data_type() {
        DataType::Map(entries, _) => entries.clone(),
        other => panic!("expected map, got {other}"),
    }
}

fn post_alter_columns() -> Vec<ArrayRef> {
    let mut columns = base_columns(&[4, 5]);
    let s2_fields = struct_fields(evolved_field("s2").data_type());
    columns.push(Arc::new(StructArray::new(
        s2_fields,
        vec![
            Arc::new(Int32Array::from(vec![Some(1), None])),
            Arc::new(StringArray::from(vec![Some("x"), None])),
        ],
        Some(NullBuffer::from(vec![true, false])),
    )));
    let m2_entries = entries_field("m2");
    columns.push(Arc::new(
        MapArray::try_new(
            m2_entries.clone(),
            OffsetBuffer::from_lengths([1, 0]),
            StructArray::new(
                struct_fields(m2_entries.data_type()),
                vec![
                    Arc::new(StringArray::from(vec!["k"])),
                    Arc::new(Int32Array::from(vec![7])),
                ],
                None,
            ),
            None,
            false,
        )
        .expect("m2 map"),
    ));
    let a2_element = match evolved_field("a2").data_type() {
        DataType::List(element) => element.clone(),
        other => panic!("expected list, got {other}"),
    };
    columns.push(Arc::new(
        ListArray::try_new(
            a2_element,
            OffsetBuffer::from_lengths([2, 0]),
            Arc::new(StringArray::from(vec!["a", "b"])),
            Some(NullBuffer::from(vec![true, false])),
        )
        .expect("a2 list"),
    ));
    let m_entries = entries_field("m");
    let m_fields = struct_fields(m_entries.data_type());
    let key = StructArray::new(
        struct_fields(m_fields[0].data_type()),
        vec![Arc::new(Int32Array::from(vec![9]))],
        None,
    );
    columns.push(Arc::new(
        MapArray::try_new(
            m_entries,
            OffsetBuffer::from_lengths([1, 0]),
            StructArray::new(
                m_fields,
                vec![Arc::new(key), Arc::new(StringArray::from(vec!["v"]))],
                None,
            ),
            Some(NullBuffer::from(vec![true, false])),
            false,
        )
        .expect("m map"),
    ));
    columns
}

fn scan_task(path: &str, schema: SchemaRef, project: &[i32]) -> FileScanTask {
    FileScanTask {
        file_size_in_bytes: std::fs::metadata(path).expect("stat").len(),
        start: 0,
        length: 0,
        record_count: None,
        file_record_count: None,
        data_file_path: Arc::from(path.to_string()),
        data_file_format: DataFileFormat::Parquet,
        schema,
        project_field_ids: Arc::from(project.to_vec()),
        predicate: None,
        deletes: Arc::from(vec![]),
        partition: None,
        partition_spec: None,
        name_mapping: None,
        case_sensitive: false,
        split_offsets: None,
        first_row_id: None,
        file_sequence_number: None,
    }
}

async fn scan(
    paths: &[&str],
    schema: SchemaRef,
    project: &[i32],
) -> crate::Result<Vec<RecordBatch>> {
    let tasks: Vec<crate::Result<FileScanTask>> = paths
        .iter()
        .map(|path| Ok(scan_task(path, schema.clone(), project)))
        .collect();
    let stream = Box::pin(futures::stream::iter(tasks)) as FileScanTaskStream;
    ArrowReaderBuilder::new(FileIO::new_with_fs())
        .build()
        .read(stream)?
        .try_collect::<Vec<RecordBatch>>()
        .await
}

fn rendered_rows(batches: &[RecordBatch], column: &str) -> BTreeMap<i64, String> {
    let options = FormatOptions::default().with_null("NULL");
    let mut rows = BTreeMap::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .expect("id column")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("id is Int64");
        let values = batch.column_by_name(column).expect("evolved column");
        let formatter = ArrayFormatter::try_new(values.as_ref(), &options).expect("formatter");
        for row in 0..batch.num_rows() {
            rows.insert(ids.value(row), formatter.value(row).to_string());
        }
    }
    rows
}

fn assert_all_null(batches: &[RecordBatch], column: &str) {
    let expected_type = evolved_field(column).data_type().clone();
    for batch in batches {
        let array = batch.column_by_name(column).expect("evolved column");
        assert_eq!(array.data_type(), &expected_type, "{column} type");
        assert_eq!(array.null_count(), batch.num_rows(), "{column} null count");
        assert!(
            (0..batch.num_rows()).all(|row| array.is_null(row)),
            "{column} has a non-null row"
        );
    }
}

async fn pre_alter_batches(column: &str) -> Vec<RecordBatch> {
    let dir = TempDir::new().expect("temp dir");
    let path = write_pre_alter_file(&dir);
    let schema = evolved_schema();
    let column_id = schema.field_by_name(column).expect("evolved column").id;
    let batches = scan(&[&path], schema, &[1, 2, 3, column_id])
        .await
        .expect("pre-ALTER file reads under the evolved schema");
    assert_eq!(
        batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        3,
        "seed rows"
    );
    batches
}

#[tokio::test]
async fn evolved_struct_reads_null_for_pre_alter_file() {
    assert_all_null(&pre_alter_batches("s2").await, "s2");
}

#[tokio::test]
async fn evolved_map_reads_null_for_pre_alter_file() {
    assert_all_null(&pre_alter_batches("m2").await, "m2");
}

#[tokio::test]
async fn evolved_list_reads_null_for_pre_alter_file() {
    assert_all_null(&pre_alter_batches("a2").await, "a2");
}

#[tokio::test]
async fn evolved_struct_keyed_map_reads_null_for_pre_alter_file() {
    let batches = pre_alter_batches("m").await;
    assert_all_null(&batches, "m");
    let DataType::Map(entries, _) = batches[0].column_by_name("m").expect("m").data_type() else {
        panic!("m is not a map");
    };
    let key_fields = struct_fields(entries.data_type());
    assert!(!entries.is_nullable(), "map entries are non-null");
    assert!(!key_fields[0].is_nullable(), "map key is non-null");
}

#[tokio::test]
async fn evolved_columns_read_values_after_alter_and_nulls_before() {
    let dir = TempDir::new().expect("temp dir");
    let pre = write_pre_alter_file(&dir);
    let post = write_parquet(
        &dir,
        "post_alter.parquet",
        &evolved_schema(),
        post_alter_columns(),
    );
    let batches = scan(&[&pre, &post], evolved_schema(), &[1, 2, 3, 4, 7, 10, 12])
        .await
        .expect("mixed files read under the evolved schema");
    let expected: [(&str, [&str; 2]); 4] = [
        ("s2", ["{p: 1, q: x}", "NULL"]),
        ("m2", ["{k: 7}", "{}"]),
        ("a2", ["[a, b]", "NULL"]),
        ("m", ["{{a: 9}: v}", "NULL"]),
    ];
    for (column, post_rows) in expected {
        let rows = rendered_rows(&batches, column);
        let want: BTreeMap<i64, String> = [
            (1, "NULL"),
            (2, "NULL"),
            (3, "NULL"),
            (4, post_rows[0]),
            (5, post_rows[1]),
        ]
        .into_iter()
        .map(|(id, value)| (id, value.to_string()))
        .collect();
        assert_eq!(rows, want, "{column}");
    }
}

#[test]
fn null_fill_covers_every_evolved_container_type() {
    for column in EVOLVED_COLUMNS {
        let data_type = evolved_field(column).data_type().clone();
        let array = create_primitive_array_repeated(&data_type, &None, 4)
            .unwrap_or_else(|e| panic!("{column}: {e}"));
        assert_eq!(array.data_type(), &data_type, "{column}");
        assert_eq!(array.len(), 4, "{column}");
        assert_eq!(array.null_count(), 4, "{column}");
    }
}

#[test]
fn null_fill_nests_a_map_inside_a_struct() {
    let data_type = DataType::Struct(
        vec![
            evolved_field("m2").with_name("inner_map"),
            evolved_field("a2").with_name("inner_list"),
        ]
        .into(),
    );
    let array = create_primitive_array_repeated(&data_type, &None, 3).expect("struct of map");
    let array = array
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("struct array");
    assert_eq!(array.null_count(), 3);
    assert_eq!(array.column(0).null_count(), 3);
    assert_eq!(array.column(1).null_count(), 3);
}
