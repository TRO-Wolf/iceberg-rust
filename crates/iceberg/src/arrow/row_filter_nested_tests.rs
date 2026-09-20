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
use std::fs::File;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StructArray};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use crate::arrow::ArrowReaderBuilder;
use crate::expr::{Bind, Predicate, Reference};
use crate::io::FileIO;
use crate::scan::{FileScanTask, FileScanTaskStream};
use crate::spec::{DataFileFormat, NestedField, PrimitiveType, Schema, SchemaRef, Type};

fn field_with_id(name: &str, data_type: DataType, nullable: bool, id: i32) -> Field {
    Field::new(name, data_type, nullable).with_metadata(HashMap::from([(
        PARQUET_FIELD_ID_META_KEY.to_string(),
        id.to_string(),
    )]))
}

fn struct_fixture_schema() -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    2,
                    "xs",
                    Type::Struct(crate::spec::StructType::new(vec![
                        NestedField::optional(3, "a", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .expect("fixture schema builds"),
    )
}

fn write_struct_fixture(dir: &TempDir) -> String {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        field_with_id("id", DataType::Int32, true, 1),
        field_with_id(
            "xs",
            DataType::Struct(vec![field_with_id("a", DataType::Int32, true, 3)].into()),
            true,
            2,
        ),
    ]));
    let id = Arc::new(Int32Array::from(vec![1, 2, 3, 4])) as ArrayRef;
    let a = Arc::new(Int32Array::from(vec![Some(1), None, None, Some(4)])) as ArrayRef;
    let xs = Arc::new(StructArray::new(
        vec![Arc::new(field_with_id("a", DataType::Int32, true, 3))].into(),
        vec![a],
        Some(NullBuffer::from(vec![true, false, true, true])),
    )) as ArrayRef;
    let to_write = RecordBatch::try_new(arrow_schema.clone(), vec![id, xs]).expect("batch builds");
    let path = format!("{}/struct.parquet", dir.path().to_str().expect("utf8"));
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let file = File::create(&path).expect("create fixture file");
    let mut writer = ArrowWriter::try_new(file, arrow_schema, Some(props)).expect("fixture writer");
    writer.write(&to_write).expect("write fixture batch");
    writer.close().expect("close fixture writer");
    path
}

async fn scan_ids(
    path: &str,
    schema: SchemaRef,
    project: Vec<i32>,
    predicate: Option<Predicate>,
) -> Vec<i32> {
    let bound = predicate
        .map(|expression| {
            expression
                .bind(schema.clone(), true)
                .expect("predicate binds")
        })
        .map(Arc::new);
    let tasks = Box::pin(futures::stream::iter(vec![Ok(FileScanTask {
        file_size_in_bytes: std::fs::metadata(path).expect("stat").len(),
        start: 0,
        length: 0,
        record_count: None,
        file_record_count: None,
        data_file_path: Arc::from(path.to_string()),
        data_file_format: DataFileFormat::Parquet,
        schema,
        project_field_ids: Arc::from(project),
        predicate: bound,
        deletes: Arc::from(vec![]),
        partition: None,
        partition_spec: None,
        name_mapping: None,
        case_sensitive: false,
        split_offsets: None,
        first_row_id: None,
        file_sequence_number: None,
    })])) as FileScanTaskStream;
    let reader = ArrowReaderBuilder::new(FileIO::new_with_fs()).build();
    reader
        .read(tasks)
        .expect("scan starts")
        .try_collect::<Vec<RecordBatch>>()
        .await
        .expect("scan completes")
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("id column")
                .values()
                .iter()
                .copied()
        })
        .collect()
}

#[tokio::test]
async fn row_filter_is_null_on_optional_struct_returns_only_null_struct_row() {
    let dir = TempDir::new().expect("temp dir");
    let path = write_struct_fixture(&dir);
    let schema = struct_fixture_schema();
    let ids = scan_ids(
        &path,
        schema,
        vec![1, 2],
        Some(Reference::new("xs").is_null()),
    )
    .await;
    assert_eq!(ids, vec![2]);
}
