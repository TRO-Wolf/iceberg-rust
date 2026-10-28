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

use std::collections::{HashMap, HashSet};
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
use crate::spec::{
    DataFileFormat, NestedField, NestedFieldRef, Schema, SchemaRef, StructType, Type,
};
use crate::{Error, ErrorKind, Result};

const DELETE_FILE_READ_BATCH_SIZE: usize = 1024;

pub(crate) fn equality_delete_schema(
    current_schema: &Schema,
    file_schema: &Schema,
    equality_ids: &[i32],
    delete_file_path: &str,
) -> Result<Schema> {
    let current_paths = struct_paths(current_schema);
    let file_paths = struct_paths(file_schema);
    let mut kept: HashSet<i32> = HashSet::new();
    let mut top_level: Vec<NestedFieldRef> = Vec::new();
    let mut seen: HashSet<i32> = HashSet::new();
    for &id in equality_ids {
        if !seen.insert(id) {
            continue;
        }
        let Some(file_path) = file_paths.get(&id) else {
            if file_schema.field_by_id(id).is_some() {
                return Err(unsupported_key_path(delete_file_path, id));
            }
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Equality delete file '{delete_file_path}' is missing its key column: \
                     equality field id {id} is absent from the delete file's own schema"
                ),
            ));
        };
        let current_field = current_schema.field_by_id(id);
        match (current_paths.get(&id), current_field) {
            (Some(current_path), _) if current_path == file_path => {
                kept.insert(id);
            }
            (Some(_), Some(field)) if file_path.len() == 1 => {
                top_level.push(field.clone());
            }
            (None, None) if file_path.len() == 1 => {
                let recovered = file_schema
                    .field_by_id(id)
                    .ok_or_else(|| unsupported_key_path(delete_file_path, id))?;
                top_level.push(recovered.clone());
            }
            _ => return Err(unsupported_key_path(delete_file_path, id)),
        }
    }
    let mut fields = prune_struct(current_schema.as_struct(), &kept);
    let mut used_names: HashSet<String> = fields.iter().map(|f| f.name.to_lowercase()).collect();
    for field in top_level {
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

pub(crate) fn top_level_field_ids(schema: &Schema) -> Vec<i32> {
    schema.as_struct().fields().iter().map(|f| f.id).collect()
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
        _ => {
            let (schema, all_ids) = orc_top_level_schema(&bytes)?;
            if let Some(id) = equality_ids
                .iter()
                .find(|id| schema.field_by_id(**id).is_none() && all_ids.contains(id))
            {
                return Err(unsupported_key_path(&task.file_path, *id));
            }
            schema
        }
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

fn unsupported_key_path(delete_file_path: &str, id: i32) -> Error {
    Error::new(
        ErrorKind::FeatureUnsupported,
        format!(
            "Equality delete file '{delete_file_path}' keys on field id {id} at a struct path \
             the current schema cannot resolve (a nested key dropped from the table, a nested key \
             under a different struct path, or a key inside a list or map); applying it is not \
             supported"
        ),
    )
}

fn struct_paths(schema: &Schema) -> HashMap<i32, Vec<i32>> {
    let mut paths = HashMap::new();
    let mut prefix = Vec::new();
    collect_struct_paths(schema.as_struct(), &mut prefix, &mut paths);
    paths
}

fn collect_struct_paths(
    struct_type: &StructType,
    prefix: &mut Vec<i32>,
    paths: &mut HashMap<i32, Vec<i32>>,
) {
    for field in struct_type.fields() {
        prefix.push(field.id);
        paths.insert(field.id, prefix.clone());
        if let Type::Struct(child) = field.field_type.as_ref() {
            collect_struct_paths(child, prefix, paths);
        }
        prefix.pop();
    }
}

fn prune_struct(struct_type: &StructType, kept: &HashSet<i32>) -> Vec<NestedFieldRef> {
    let mut fields = Vec::new();
    for field in struct_type.fields() {
        if kept.contains(&field.id) {
            fields.push(field.clone());
            continue;
        }
        let Type::Struct(child) = field.field_type.as_ref() else {
            continue;
        };
        let children = prune_struct(child, kept);
        if children.is_empty() {
            continue;
        }
        let mut pruned: NestedField = field.as_ref().clone();
        pruned.field_type = Box::new(Type::Struct(StructType::new(children)));
        fields.push(Arc::new(pruned));
    }
    fields
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
