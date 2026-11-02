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
use std::str::FromStr;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::basic::Type as PhysicalType;
use parquet::file::metadata::{FileMetaData, ParquetMetaData, ParquetMetaDataBuilder};
use parquet::schema::types::{SchemaDescriptor, Type as ParquetType, TypePtr};

use crate::arrow::arrow_type_to_type;
use crate::arrow::field_id_stamping::idless_read_allowed;
use crate::arrow::row_filter_plan::top_level_ancestor_id;
use crate::spec::{PrimitiveType, Schema, Type};
use crate::{Error, ErrorKind, Result};

fn stamped_id(field: &Field) -> Option<i32> {
    field
        .metadata()
        .get(PARQUET_FIELD_ID_META_KEY)
        .and_then(|id| i32::from_str(id).ok())
}

fn table_primitive(table_schema: &Schema, field: &Field) -> Option<PrimitiveType> {
    let table_field = table_schema.as_struct().field_by_id(stamped_id(field)?)?;
    match table_field.field_type.as_ref() {
        Type::Primitive(primitive) => Some(primitive.clone()),
        _ => None,
    }
}

fn string_read_as_binary(table_schema: &Schema, field: &Field, parquet_field: &TypePtr) -> bool {
    matches!(
        field.data_type(),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    ) && parquet_field.is_primitive()
        && parquet_field.get_physical_type() == PhysicalType::BYTE_ARRAY
        && table_primitive(table_schema, field) == Some(PrimitiveType::Binary)
}

fn without_annotation(parquet_field: &TypePtr) -> Result<TypePtr> {
    let info = parquet_field.get_basic_info();
    let rebuilt = ParquetType::primitive_type_builder(info.name(), PhysicalType::BYTE_ARRAY)
        .with_repetition(info.repetition())
        .with_id(info.has_id().then(|| info.id()))
        .build()
        .map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                format!(
                    "cannot read the data file column '{}' as bytes",
                    info.name()
                ),
            )
            .with_source(e)
        })?;
    Ok(Arc::new(rebuilt))
}

pub(crate) fn idless_read_options(
    metadata: &Arc<ParquetMetaData>,
    arrow_schema: ArrowSchemaRef,
    table_schema: &Schema,
) -> Result<(Arc<ParquetMetaData>, ArrowReaderOptions)> {
    let file_metadata = metadata.file_metadata();
    let root = file_metadata.schema_descr().root_schema();
    let parquet_fields = root.get_fields();
    let as_bytes: Vec<bool> = arrow_schema
        .fields()
        .iter()
        .zip(parquet_fields)
        .map(|(field, parquet_field)| string_read_as_binary(table_schema, field, parquet_field))
        .collect();
    if !as_bytes.contains(&true) {
        let options = ArrowReaderOptions::new().with_schema(arrow_schema);
        return Ok((Arc::clone(metadata), options));
    }
    let mut rebuilt_fields = Vec::with_capacity(parquet_fields.len());
    let mut arrow_fields = Vec::with_capacity(parquet_fields.len());
    for ((field, parquet_field), bytes) in arrow_schema
        .fields()
        .iter()
        .zip(parquet_fields)
        .zip(&as_bytes)
    {
        match bytes {
            true => {
                rebuilt_fields.push(without_annotation(parquet_field)?);
                arrow_fields.push(field.as_ref().clone().with_data_type(DataType::Binary));
            }
            false => {
                rebuilt_fields.push(Arc::clone(parquet_field));
                arrow_fields.push(field.as_ref().clone());
            }
        }
    }
    let root = ParquetType::group_type_builder(root.name())
        .with_fields(rebuilt_fields)
        .build()
        .map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                "cannot rebuild the data file schema to read a string column as bytes",
            )
            .with_source(e)
        })?;
    let rebuilt_metadata = FileMetaData::new(
        file_metadata.version(),
        file_metadata.num_rows(),
        file_metadata.created_by().map(str::to_string),
        file_metadata.key_value_metadata().cloned(),
        Arc::new(SchemaDescriptor::new(Arc::new(root))),
        file_metadata.column_orders().cloned(),
    );
    let rebuilt = ParquetMetaDataBuilder::new(rebuilt_metadata)
        .set_row_groups(metadata.row_groups().to_vec())
        .set_column_index(metadata.column_index().cloned())
        .set_offset_index(metadata.offset_index().cloned())
        .build();
    let arrow_schema = Arc::new(ArrowSchema::new_with_metadata(
        arrow_fields,
        arrow_schema.metadata().clone(),
    ));
    Ok((
        Arc::new(rebuilt),
        ArrowReaderOptions::new().with_schema(arrow_schema),
    ))
}

pub(crate) fn filter_reads_a_converted_column(
    table_schema: &Schema,
    parquet_schema: &SchemaDescriptor,
    stamped_arrow_schema: &ArrowSchemaRef,
    iceberg_field_ids: &HashSet<i32>,
) -> bool {
    let filtered: HashSet<i32> = iceberg_field_ids
        .iter()
        .filter_map(|id| top_level_ancestor_id(table_schema, *id))
        .collect();
    stamped_arrow_schema
        .fields()
        .iter()
        .zip(parquet_schema.root_schema().get_fields())
        .any(|(field, parquet_field)| {
            let Some(table_type) = table_primitive(table_schema, field) else {
                return false;
            };
            let Ok(Type::Primitive(file_type)) = arrow_type_to_type(field.data_type()) else {
                return false;
            };
            stamped_id(field).is_some_and(|id| filtered.contains(&id))
                && parquet_field.is_primitive()
                && file_type != table_type
                && idless_read_allowed(parquet_field.get_physical_type(), &file_type, &table_type)
        })
}
