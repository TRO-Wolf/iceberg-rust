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

use parquet::arrow::async_reader::AsyncFileReader;
use parquet::file::metadata::ParquetMetaData;
use parquet::schema::types::Type as ParquetType;

use crate::arrow::ArrowFileReader;
use crate::io::FileIO;
use crate::spec::{
    DataContentType, DataFile, Literal, MetricsByFieldId, MetricsConfig, NameMapping, NestedField,
    PartitionSpecRef, PrimitiveType, Schema, SchemaRef, Struct, StructType, Type,
};
use crate::writer::file_writer::ParquetWriter;
use crate::{Error, ErrorKind, Result};

pub(super) const HIVE_DEFAULT_PARTITION: &str = "__HIVE_DEFAULT_PARTITION__";

pub(super) fn unescape_hive_path_name(segment: &str) -> String {
    let bytes = segment.as_bytes();
    let Some(first) = segment.find('%') else {
        return segment.to_string();
    };
    let mut out = String::with_capacity(segment.len());
    out.push_str(&segment[..first]);
    let mut index = first;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            match hex_pair(bytes[index + 1], bytes[index + 2]) {
                Some(code) => {
                    out.push(char::from(code));
                    index += 3;
                    continue;
                }
                None => {
                    out.push('%');
                    index += 1;
                    continue;
                }
            }
        }
        let rest = &segment[index..];
        let character = rest.chars().next().unwrap_or('\u{0}');
        out.push(character);
        index += character.len_utf8();
    }
    out
}

fn hex_pair(high: u8, low: u8) -> Option<u8> {
    Some((hex_digit(high)? << 4) | hex_digit(low)?)
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

pub(super) struct AdoptionContext {
    pub(super) schema: SchemaRef,
    pub(super) metrics_config: MetricsConfig,
    pub(super) name_mapping: Option<Arc<NameMapping>>,
    pub(super) spec: PartitionSpecRef,
    pub(super) partition_type: StructType,
}

pub(super) async fn adopt_parquet_file(
    file_io: &FileIO,
    context: &AdoptionContext,
    path: &str,
    file_size_in_bytes: u64,
    partition_values: &[(String, String)],
) -> Result<DataFile> {
    let partition = partition_struct(context, partition_values)?;

    let input_file = file_io.new_input(path)?;
    let file_metadata = input_file.metadata().await?;
    let reader = input_file.reader().await?;
    let mut parquet_reader = ArrowFileReader::new(file_metadata, reader);
    let parquet_metadata = parquet_reader.get_metadata(None).await.map_err(|err| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Cannot read the parquet footer of the file to import: {path}"),
        )
        .with_source(err)
    })?;

    let resolved = resolved_file_schema(context, &parquet_metadata, path)?;
    let metrics = MetricsByFieldId::new(&resolved, &context.metrics_config);
    let mut builder = ParquetWriter::parquet_to_data_file_builder(
        Arc::new(resolved),
        parquet_metadata,
        usize::try_from(file_size_in_bytes).map_err(|err| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("File size {file_size_in_bytes} of {path} does not fit in usize"),
            )
            .with_source(err)
        })?,
        path.to_string(),
        HashMap::new(),
        &metrics,
    )?;
    builder
        .content(DataContentType::Data)
        .partition_spec_id(context.spec.spec_id())
        .partition(partition)
        .split_offsets(None)
        .sort_order_id(0);
    builder.build().map_err(|err| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Cannot build a data file entry for the imported file {path}"),
        )
        .with_source(err)
    })
}

fn resolved_file_schema(
    context: &AdoptionContext,
    parquet_metadata: &ParquetMetaData,
    path: &str,
) -> Result<Schema> {
    let root = parquet_metadata
        .file_metadata()
        .schema_descr()
        .root_schema();
    let columns = root.get_fields();
    let embedded = has_embedded_field_ids(root);
    if embedded {
        refuse_partial_field_ids(columns, path)?;
    }

    let mut fields: Vec<Arc<NestedField>> = Vec::with_capacity(columns.len());
    for (position, column) in columns.iter().enumerate() {
        let Some(field_id) = resolve_field_id(context, column, position, embedded) else {
            continue;
        };
        let Some(table_field) = context.schema.field_by_id(field_id) else {
            continue;
        };
        fields.push(Arc::new(NestedField {
            id: field_id,
            name: column.name().to_string(),
            required: table_field.required,
            field_type: table_field.field_type.clone(),
            doc: None,
            initial_default: None,
            write_default: None,
        }));
    }

    Schema::builder()
        .with_fields(fields)
        .build()
        .map_err(|err| {
            Error::new(
                ErrorKind::DataInvalid,
                "Cannot resolve the field ids of the parquet file to import",
            )
            .with_source(err)
        })
}

fn resolve_field_id(
    context: &AdoptionContext,
    column: &ParquetType,
    position: usize,
    embedded: bool,
) -> Option<i32> {
    if embedded {
        return column
            .get_basic_info()
            .has_id()
            .then(|| column.get_basic_info().id());
    }
    match &context.name_mapping {
        Some(name_mapping) => name_mapping
            .fields()
            .iter()
            .find(|field| field.names().iter().any(|name| name == column.name()))
            .and_then(|field| field.field_id()),
        None => i32::try_from(position + 1).ok(),
    }
}

fn refuse_partial_field_ids(columns: &[Arc<ParquetType>], path: &str) -> Result<()> {
    let without: Vec<&str> = columns
        .iter()
        .filter(|column| !column.get_basic_info().has_id())
        .map(|column| column.name())
        .collect();
    if without.is_empty() {
        return Ok(());
    }
    Err(Error::new(
        ErrorKind::DataInvalid,
        format!(
            "Cannot import the parquet file {path} because only some of its columns carry Iceberg field ids. These columns carry none: {}. Rewrite the file with a field id on every column, or with none at all.",
            without.join(", ")
        ),
    ))
}

fn has_embedded_field_ids(node: &ParquetType) -> bool {
    if node.get_basic_info().has_id() && !node.is_schema() {
        return true;
    }
    node.is_group()
        && node
            .get_fields()
            .iter()
            .any(|child| has_embedded_field_ids(child))
}

fn partition_struct(
    context: &AdoptionContext,
    partition_values: &[(String, String)],
) -> Result<Struct> {
    if context.spec.is_unpartitioned() {
        return Ok(Struct::empty());
    }
    let types = context.partition_type.fields();
    let mut literals: Vec<Option<Literal>> = Vec::with_capacity(types.len());
    for (index, field) in context.spec.fields().iter().enumerate() {
        let raw = partition_values
            .iter()
            .find(|(name, _)| name == &field.name)
            .map(|(_, value)| value.as_str());
        let field_type = types.get(index).ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                format!("Partition type has no field at position {index}"),
            )
        })?;
        literals.push(partition_literal(&field_type.field_type, raw, &field.name)?);
    }
    Ok(Struct::from_iter(literals))
}

fn partition_literal(
    field_type: &Type,
    raw: Option<&str>,
    field_name: &str,
) -> Result<Option<Literal>> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    if raw == HIVE_DEFAULT_PARTITION {
        return Ok(None);
    }
    let Type::Primitive(primitive) = field_type else {
        return Err(unsupported_partition_type(field_type, field_name));
    };
    let literal = match primitive {
        PrimitiveType::Boolean => Literal::bool_from_str(raw)?,
        PrimitiveType::Int => Literal::int(parse_partition_value::<i32>(raw, field_name)?),
        PrimitiveType::Long => Literal::long(parse_partition_value::<i64>(raw, field_name)?),
        PrimitiveType::Float => Literal::float(parse_partition_value::<f32>(raw, field_name)?),
        PrimitiveType::Double => Literal::double(parse_partition_value::<f64>(raw, field_name)?),
        PrimitiveType::String => Literal::string(raw),
        PrimitiveType::Uuid => Literal::uuid_from_str(raw)?,
        PrimitiveType::Fixed(length) => {
            let mut bytes = raw.as_bytes().to_vec();
            bytes.resize(usize::try_from(*length).unwrap_or(usize::MAX), 0);
            Literal::fixed(bytes)
        }
        PrimitiveType::Binary => Literal::binary(raw.as_bytes().to_vec()),
        PrimitiveType::Decimal { .. } => Literal::decimal_from_str(raw)?,
        PrimitiveType::Date => Literal::date_from_str(raw)?,
        _ => return Err(unsupported_partition_type(field_type, field_name)),
    };
    Ok(Some(literal))
}

fn parse_partition_value<T: std::str::FromStr>(raw: &str, field_name: &str) -> Result<T> {
    raw.parse::<T>().map_err(|_| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Cannot parse the partition value '{raw}' of column {field_name}"),
        )
    })
}

fn unsupported_partition_type(field_type: &Type, field_name: &str) -> Error {
    Error::new(
        ErrorKind::FeatureUnsupported,
        format!(
            "Unsupported type for fromPartitionString: {field_type} (partition column {field_name})"
        ),
    )
}
