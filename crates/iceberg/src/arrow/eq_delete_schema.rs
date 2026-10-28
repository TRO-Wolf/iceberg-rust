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

use std::collections::HashSet;
use std::io::Read;
use std::sync::Arc;

use apache_avro::Reader as AvroReader;
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::arrow_schema_to_schema;
use crate::arrow::avro_reader::read_avro_data_bytes;
use crate::arrow::orc_reader::{orc_top_level_schema, read_orc_data_bytes};
use crate::io::FileIO;
use crate::scan::FileScanTaskDeleteFile;
use crate::spec::{DataFileFormat, NestedFieldRef, Schema, SchemaRef};
use crate::{Error, ErrorKind, Result};

const DELETE_FILE_READ_BATCH_SIZE: usize = 1024;

pub(crate) fn equality_delete_schema(
    current_schema: &Schema,
    file_schema: &Schema,
    equality_ids: &[i32],
    delete_file_path: &str,
) -> Result<Schema> {
    let mut fields: Vec<NestedFieldRef> = Vec::with_capacity(equality_ids.len());
    let mut used_names: HashSet<String> = HashSet::new();
    let mut seen: HashSet<i32> = HashSet::new();
    for &id in equality_ids {
        if !seen.insert(id) {
            continue;
        }
        let field = match current_schema.field_by_id(id) {
            Some(field) => field.clone(),
            None => file_schema.field_by_id(id).cloned().ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Equality delete file '{delete_file_path}' is missing its key column: \
                         equality field id {id} is absent from the delete file's own schema"
                    ),
                )
            })?,
        };
        let mut name = field.name.clone();
        while used_names.contains(&name.to_lowercase()) {
            name.push('_');
        }
        used_names.insert(name.to_lowercase());
        if name == field.name {
            fields.push(field);
        } else {
            let mut renamed = field.as_ref().clone();
            renamed.name = name;
            fields.push(Arc::new(renamed));
        }
    }
    Schema::builder().with_fields(fields).build()
}

pub(crate) fn parquet_key_schema(
    arrow_schema: &ArrowSchema,
    equality_ids: &[i32],
) -> Result<Schema> {
    let wanted: HashSet<i32> = equality_ids.iter().copied().collect();
    let fields: Vec<Field> = arrow_schema
        .fields()
        .iter()
        .filter(|field| arrow_field_holds_any(field, &wanted))
        .map(|field| field.as_ref().clone())
        .collect();
    arrow_schema_to_schema(&ArrowSchema::new(fields))
}

pub(crate) async fn whole_file_equality_delete_batches(
    file_io: &FileIO,
    task: &FileScanTaskDeleteFile,
    current_schema: &Schema,
    equality_ids: &[i32],
) -> Result<(Vec<RecordBatch>, SchemaRef)> {
    let input = file_io.new_input(&task.file_path)?;
    let bytes = input.read().await?;
    let file_schema = match task.file_format {
        DataFileFormat::Avro => avro_file_schema(&bytes)?,
        _ => orc_top_level_schema(&bytes)?.0,
    };
    let delete_schema = Arc::new(equality_delete_schema(
        current_schema,
        &file_schema,
        equality_ids,
        &task.file_path,
    )?);
    let expected = delete_schema.clone();
    let is_avro = task.file_format == DataFileFormat::Avro;
    let batches = crate::runtime::spawn_blocking(move || {
        if is_avro {
            read_avro_data_bytes(&bytes, &expected, DELETE_FILE_READ_BATCH_SIZE)
        } else {
            read_orc_data_bytes(bytes, &expected, DELETE_FILE_READ_BATCH_SIZE)
        }
    })
    .try_join()
    .await
    .map_err(|error| error.with_context("path", task.file_path.clone()))??;
    Ok((batches, delete_schema))
}

fn avro_file_schema(bytes: &[u8]) -> Result<Schema> {
    let (head, body) = crate::avro::ocf::repaired_ocf_parts(bytes)?;
    let reader = AvroReader::new(head.as_ref().chain(body)).map_err(|e| {
        Error::new(
            ErrorKind::DataInvalid,
            "Failed to open Avro delete file (could not read the OCF header)",
        )
        .with_source(e)
    })?;
    crate::avro::avro_schema_to_schema(reader.writer_schema())
}

fn arrow_field_holds_any(field: &Field, wanted: &HashSet<i32>) -> bool {
    let own = field
        .metadata()
        .get(PARQUET_FIELD_ID_META_KEY)
        .and_then(|id| id.parse::<i32>().ok())
        .is_some_and(|id| wanted.contains(&id));
    own || match field.data_type() {
        DataType::Struct(children) => children
            .iter()
            .any(|child| arrow_field_holds_any(child, wanted)),
        _ => false,
    }
}
