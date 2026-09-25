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

use arrow_array::{Array, ArrayRef, Int64Array, NullArray, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use arrow_select::concat::concat_batches;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use super::ParquetWriterBuilder;
use crate::arrow::record_batch_transformer::RecordBatchTransformerBuilder;
use crate::arrow::schema_to_arrow_schema;
use crate::io::FileIO;
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, NestedField, PrimitiveType, Schema, Struct,
    StructType, Type,
};
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator, FileNameGenerator, LocationGenerator,
};
use crate::writer::file_writer::{FileWriter, FileWriterBuilder};

fn unknown_top_level_schema() -> Arc<Schema> {
    Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "c", Type::Primitive(PrimitiveType::Unknown)).into(),
            ])
            .build()
            .expect("unknown top-level schema"),
    )
}

fn unknown_nested_schema() -> Arc<Schema> {
    Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(
                    2,
                    "s",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(3, "a", Type::Primitive(PrimitiveType::Unknown))
                            .into(),
                        NestedField::optional(4, "b", Type::Primitive(PrimitiveType::Long))
                            .into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .expect("unknown nested schema"),
    )
}

fn batch_with_null_unknown(ids: Vec<i64>) -> RecordBatch {
    let rows = ids.len();
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("c", DataType::Null, true),
    ]));
    RecordBatch::try_new(
        arrow_schema,
        vec![
            Arc::new(Int64Array::from(ids)) as ArrayRef,
            Arc::new(NullArray::new(rows)) as ArrayRef,
        ],
    )
    .expect("batch with Null unknown column")
}

fn batch_with_null_nested_unknown(ids: Vec<i64>, b_values: Vec<i64>) -> RecordBatch {
    let struct_fields: arrow_schema::Fields = vec![
        Field::new("a", DataType::Null, true),
        Field::new("b", DataType::Int64, true),
    ]
    .into();
    let struct_column = Arc::new(StructArray::new(
        struct_fields.clone(),
        vec![
            Arc::new(NullArray::new(ids.len())) as ArrayRef,
            Arc::new(Int64Array::from(b_values)) as ArrayRef,
        ],
        None,
    )) as ArrayRef;
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("s", DataType::Struct(struct_fields), true),
    ]));
    RecordBatch::try_new(
        arrow_schema,
        vec![Arc::new(Int64Array::from(ids)) as ArrayRef, struct_column],
    )
    .expect("batch with Null nested unknown field")
}

async fn write_single_file(schema: &Arc<Schema>, batch: &RecordBatch) -> (TempDir, DataFile) {
    let temp_dir = TempDir::new().expect("temp dir");
    let file_io = FileIO::new_with_fs();
    let location_gen = DefaultLocationGenerator::with_data_location(
        temp_dir.path().to_str().expect("temp path").to_string(),
    );
    let file_name_gen =
        DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);
    let output_file = file_io
        .new_output(location_gen.generate_location(None, &file_name_gen.generate_file_name()))
        .expect("output file");
    let mut writer = ParquetWriterBuilder::new(WriterProperties::builder().build(), schema.clone())
        .build(output_file)
        .await
        .expect("unknown schema builds without refusal");
    writer.write(batch).await.expect("write batch");
    let data_file = writer
        .close()
        .await
        .expect("close writer")
        .into_iter()
        .next()
        .expect("one data file builder")
        .content(DataContentType::Data)
        .partition(Struct::empty())
        .partition_spec_id(0)
        .build()
        .expect("data file");
    (temp_dir, data_file)
}

async fn read_file_batches(file_io: &FileIO, data_file: &DataFile) -> Vec<RecordBatch> {
    let bytes = file_io
        .new_input(data_file.file_path.clone())
        .expect("input file")
        .read()
        .await
        .expect("read file bytes");
    ParquetRecordBatchReaderBuilder::try_new(bytes)
        .expect("open parquet file")
        .build()
        .expect("parquet reader")
        .map(|batch| batch.expect("file batch"))
        .collect::<Vec<_>>()
}

fn concat_file_batches(batches: &[RecordBatch]) -> RecordBatch {
    concat_batches(&batches[0].schema(), batches).expect("concat file batches")
}

fn stamped_id_column(file_batch: &RecordBatch, table_schema: &Schema) -> (Field, ArrayRef) {
    let table_arrow = schema_to_arrow_schema(table_schema).expect("table arrow schema");
    let field = table_arrow
        .field_with_name("id")
        .expect("table has id column")
        .as_ref()
        .clone();
    let column = file_batch
        .column(
            file_batch
                .schema()
                .index_of("id")
                .expect("file has id column"),
        )
        .clone();
    (field, column)
}

#[tokio::test]
async fn unknown_top_level_column_writes_no_parquet_column_and_reads_back_null() {
    let schema = unknown_top_level_schema();
    let (_temp_dir, data_file) =
        write_single_file(&schema, &batch_with_null_unknown(vec![1, 2, 3])).await;

    assert_eq!(data_file.record_count(), 3);
    assert_eq!(data_file.value_counts(), &HashMap::from([(1, 3u64)]));
    assert_eq!(data_file.null_value_counts(), &HashMap::from([(1, 0u64)]));
    assert!(
        data_file.column_sizes().contains_key(&1),
        "the written column keeps its size entry"
    );
    for (map_name, map) in [
        ("column_sizes", data_file.column_sizes()),
        ("value_counts", data_file.value_counts()),
        ("null_value_counts", data_file.null_value_counts()),
        ("nan_value_counts", data_file.nan_value_counts()),
    ] {
        assert!(
            !map.contains_key(&2),
            "{map_name} must carry no entry for the unknown field id"
        );
    }
    assert!(
        !data_file.lower_bounds().contains_key(&2),
        "lower_bounds must carry no entry for the unknown field id"
    );
    assert!(
        !data_file.upper_bounds().contains_key(&2),
        "upper_bounds must carry no entry for the unknown field id"
    );

    let file_io = FileIO::new_with_fs();
    let batches = read_file_batches(&file_io, &data_file).await;
    let file_batch = concat_file_batches(&batches);
    let file_schema = file_batch.schema();
    let names: Vec<&str> = file_schema
        .fields()
        .iter()
        .map(|field| field.name().as_str())
        .collect();
    assert_eq!(
        names,
        vec!["id"],
        "the parquet file holds no column for the unknown field"
    );

    let (id_field, id_column) = stamped_id_column(&file_batch, &schema);
    let stamped = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![id_field])),
        vec![id_column],
    )
    .expect("id-stamped file batch");
    let mut transformer = RecordBatchTransformerBuilder::new(schema.clone(), &[1, 2]).build();
    let out = transformer
        .process_record_batch(stamped)
        .expect("transform file batch");
    assert_eq!(out.num_columns(), 2);
    let out_ids = out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("id column");
    assert_eq!(out_ids.values(), &[1, 2, 3]);
    let out_unknown = out.column(1);
    assert_eq!(out_unknown.data_type(), &DataType::Null);
    assert_eq!(out_unknown.len(), 3);
    assert_eq!(out_unknown.logical_null_count(), 3);
}

#[tokio::test]
async fn unknown_nested_struct_field_is_written_without_that_field() {
    let schema = unknown_nested_schema();
    let (_temp_dir, data_file) =
        write_single_file(&schema, &batch_with_null_nested_unknown(vec![1, 2], vec![10, 20]))
            .await;

    assert_eq!(data_file.record_count(), 2);
    assert!(
        !data_file.value_counts().contains_key(&3),
        "value_counts must carry no entry for the nested unknown field id"
    );
    assert_eq!(
        data_file.value_counts().get(&4),
        Some(&2u64),
        "the nested sibling keeps its counts"
    );
    assert!(
        !data_file.column_sizes().contains_key(&3),
        "column_sizes must carry no entry for the nested unknown field id"
    );

    let file_io = FileIO::new_with_fs();
    let batches = read_file_batches(&file_io, &data_file).await;
    let file_batch = concat_file_batches(&batches);
    let file_struct = file_batch
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("struct column");
    assert_eq!(
        file_struct.num_columns(),
        1,
        "the written struct holds only the known child"
    );

    let table_arrow = schema_to_arrow_schema(&schema).expect("table arrow schema");
    let id_field = table_arrow
        .field_with_name("id")
        .expect("table has id column")
        .as_ref()
        .clone();
    let struct_field = table_arrow
        .field_with_name("s")
        .expect("table has struct column")
        .as_ref()
        .clone();
    let DataType::Struct(table_children) = struct_field.data_type().clone() else {
        panic!("table struct column must stay a struct");
    };
    let rebuilt_struct = Arc::new(StructArray::new(
        table_children,
        vec![
            Arc::new(NullArray::new(2)) as ArrayRef,
            file_struct.column(0).clone(),
        ],
        file_struct.nulls().cloned(),
    )) as ArrayRef;
    let stamped = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![id_field, struct_field])),
        vec![file_batch.column(0).clone(), rebuilt_struct],
    )
    .expect("struct-stamped file batch");
    let mut transformer = RecordBatchTransformerBuilder::new(schema.clone(), &[1, 2]).build();
    let out = transformer
        .process_record_batch(stamped)
        .expect("transform file batch");
    let out_struct = out
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("out struct column");
    let out_a = out_struct.column(0);
    assert_eq!(out_a.data_type(), &DataType::Null);
    assert_eq!(out_a.logical_null_count(), 2);
    let out_b = out_struct
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("sibling column");
    assert_eq!(out_b.values(), &[10, 20]);
}

#[tokio::test]
async fn rewritten_file_after_row_delete_still_omits_unknown_column() {
    let schema = unknown_top_level_schema();
    let (_first_dir, first_file) =
        write_single_file(&schema, &batch_with_null_unknown(vec![1, 2])).await;

    let file_io = FileIO::new_with_fs();
    let batches = read_file_batches(&file_io, &first_file).await;
    let file_batch = concat_file_batches(&batches);
    let file_ids = file_batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("id column");
    let kept: Vec<i64> = file_ids
        .iter()
        .flatten()
        .filter(|id| *id != 1)
        .collect();
    assert_eq!(kept, vec![2]);

    let (_second_dir, second_file) =
        write_single_file(&schema, &batch_with_null_unknown(kept)).await;
    assert_eq!(second_file.record_count(), 1);
    assert_eq!(second_file.value_counts(), &HashMap::from([(1, 1u64)]));
    assert!(
        !second_file.value_counts().contains_key(&2),
        "the rewritten file carries no entry for the unknown field id"
    );
    let second_batches = read_file_batches(&file_io, &second_file).await;
    let second_batch = concat_file_batches(&second_batches);
    let second_schema = second_batch.schema();
    let names: Vec<&str> = second_schema
        .fields()
        .iter()
        .map(|field| field.name().as_str())
        .collect();
    assert_eq!(
        names,
        vec!["id"],
        "the rewritten file holds no column for the unknown field"
    );
}

#[tokio::test]
async fn batch_missing_a_written_column_fails_loud_instead_of_writing_short() {
    let schema = Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "c", Type::Primitive(PrimitiveType::Unknown)).into(),
                NestedField::optional(3, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("schema"),
    );
    let temp_dir = TempDir::new().expect("temp dir");
    let file_io = FileIO::new_with_fs();
    let output_file = file_io
        .new_output(temp_dir.path().join("out.parquet").to_string_lossy())
        .expect("output file");
    let mut writer = ParquetWriterBuilder::new(WriterProperties::builder().build(), schema)
        .build(output_file)
        .await
        .expect("build");
    let error = writer
        .write(&batch_with_null_unknown(vec![1]))
        .await
        .expect_err("a batch missing the written name column must fail loud");
    assert!(
        error.message().contains("name"),
        "the error must name the missing column, got: {}",
        error.message()
    );
}
