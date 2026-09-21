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

use crate::spec::{NestedField, PrimitiveType, Schema, StructType, Type};
use crate::{Error, ErrorKind, Result};

pub(crate) const ICEBERG_ID_ATTRIBUTE: &str = "iceberg.id";
pub(crate) const ICEBERG_REQUIRED_ATTRIBUTE: &str = "iceberg.required";
pub(crate) const ICEBERG_LONG_TYPE_ATTRIBUTE: &str = "iceberg.long-type";
pub(crate) const ICEBERG_BINARY_TYPE_ATTRIBUTE: &str = "iceberg.binary-type";
pub(crate) const ICEBERG_FIELD_LENGTH: &str = "iceberg.length";
pub(crate) const ICEBERG_TIMESTAMP_UNIT: &str = "iceberg.timestamp-unit";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OrcKind {
    Boolean = 0,
    Int = 3,
    Long = 4,
    Float = 5,
    Double = 6,
    String = 7,
    Binary = 8,
    Timestamp = 9,
    List = 10,
    Map = 11,
    Struct = 12,
    Decimal = 14,
    Date = 15,
    TimestampInstant = 18,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct OrcType {
    pub(crate) kind: Option<OrcKind>,
    pub(crate) subtypes: Vec<u32>,
    pub(crate) field_names: Vec<String>,
    pub(crate) maximum_length: Option<u32>,
    pub(crate) precision: Option<u32>,
    pub(crate) scale: Option<u32>,
    pub(crate) attributes: Vec<(String, String)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColumnShape {
    Boolean,
    Int,
    Long,
    Float,
    Double,
    Date,
    Timestamp,
    Decimal,
    Bytes,
    Utf8,
    List,
    Map,
    Struct,
}

#[derive(Debug, Clone)]
pub(crate) struct OrcColumn {
    pub(crate) shape: ColumnShape,
    pub(crate) children: Vec<usize>,
    pub(crate) iceberg_type: Option<Type>,
}

#[derive(Debug, Clone)]
pub(crate) struct OrcSchema {
    pub(crate) types: Vec<OrcType>,
    pub(crate) columns: Vec<OrcColumn>,
}

pub(crate) fn build_orc_schema(schema: &Schema) -> Result<OrcSchema> {
    let mut out = OrcSchema {
        types: Vec::new(),
        columns: Vec::new(),
    };
    push_struct(&mut out, schema.as_struct(), None, None)?;
    Ok(out)
}

fn push_struct(
    out: &mut OrcSchema,
    struct_type: &StructType,
    field_id: Option<i32>,
    required: Option<bool>,
) -> Result<usize> {
    let index = reserve(out, ColumnShape::Struct, None);
    out.types[index].kind = Some(OrcKind::Struct);
    out.types[index].attributes = base_attributes(field_id, required);
    let mut children = Vec::with_capacity(struct_type.fields().len());
    let mut names = Vec::with_capacity(struct_type.fields().len());
    for field in struct_type.fields() {
        names.push(field.name.clone());
        children.push(push_field(out, field)?);
    }
    out.types[index].subtypes = children.iter().map(|c| *c as u32).collect();
    out.types[index].field_names = names;
    out.columns[index].children = children;
    Ok(index)
}

fn push_field(out: &mut OrcSchema, field: &NestedField) -> Result<usize> {
    let id = Some(field.id);
    let required = Some(field.required);
    match field.field_type.as_ref() {
        Type::Primitive(primitive) => push_primitive(out, primitive, field, id, required),
        Type::Struct(inner) => push_struct(out, inner, id, required),
        Type::List(list) => {
            let index = reserve(out, ColumnShape::List, None);
            out.types[index].kind = Some(OrcKind::List);
            out.types[index].attributes = base_attributes(id, required);
            let element = push_field(out, &list.element_field)?;
            out.types[index].subtypes = vec![element as u32];
            out.columns[index].children = vec![element];
            Ok(index)
        }
        Type::Map(map) => {
            let index = reserve(out, ColumnShape::Map, None);
            out.types[index].kind = Some(OrcKind::Map);
            out.types[index].attributes = base_attributes(id, required);
            let key = push_field(out, &map.key_field)?;
            let value = push_field(out, &map.value_field)?;
            out.types[index].subtypes = vec![key as u32, value as u32];
            out.columns[index].children = vec![key, value];
            Ok(index)
        }
        Type::Variant => Err(unsupported(field, "variant")),
    }
}

fn push_primitive(
    out: &mut OrcSchema,
    primitive: &PrimitiveType,
    field: &NestedField,
    id: Option<i32>,
    required: Option<bool>,
) -> Result<usize> {
    let mut extra: Vec<(String, String)> = Vec::new();
    let (kind, shape) = match primitive {
        PrimitiveType::Boolean => (OrcKind::Boolean, ColumnShape::Boolean),
        PrimitiveType::Int => (OrcKind::Int, ColumnShape::Int),
        PrimitiveType::Long => {
            extra.push((ICEBERG_LONG_TYPE_ATTRIBUTE.to_string(), "LONG".to_string()));
            (OrcKind::Long, ColumnShape::Long)
        }
        PrimitiveType::Float => (OrcKind::Float, ColumnShape::Float),
        PrimitiveType::Double => (OrcKind::Double, ColumnShape::Double),
        PrimitiveType::Date => (OrcKind::Date, ColumnShape::Date),
        PrimitiveType::Time => {
            extra.push((ICEBERG_LONG_TYPE_ATTRIBUTE.to_string(), "TIME".to_string()));
            (OrcKind::Long, ColumnShape::Long)
        }
        PrimitiveType::Timestamp => {
            extra.push((ICEBERG_TIMESTAMP_UNIT.to_string(), "MICROS".to_string()));
            (OrcKind::Timestamp, ColumnShape::Timestamp)
        }
        PrimitiveType::Timestamptz => {
            extra.push((ICEBERG_TIMESTAMP_UNIT.to_string(), "MICROS".to_string()));
            (OrcKind::TimestampInstant, ColumnShape::Timestamp)
        }
        PrimitiveType::TimestampNs => {
            extra.push((ICEBERG_TIMESTAMP_UNIT.to_string(), "NANOS".to_string()));
            (OrcKind::Timestamp, ColumnShape::Timestamp)
        }
        PrimitiveType::TimestamptzNs => {
            extra.push((ICEBERG_TIMESTAMP_UNIT.to_string(), "NANOS".to_string()));
            (OrcKind::TimestampInstant, ColumnShape::Timestamp)
        }
        PrimitiveType::String => (OrcKind::String, ColumnShape::Utf8),
        PrimitiveType::Uuid => {
            extra.push((
                ICEBERG_BINARY_TYPE_ATTRIBUTE.to_string(),
                "UUID".to_string(),
            ));
            (OrcKind::Binary, ColumnShape::Bytes)
        }
        PrimitiveType::Fixed(length) => {
            extra.push((
                ICEBERG_BINARY_TYPE_ATTRIBUTE.to_string(),
                "FIXED".to_string(),
            ));
            extra.push((ICEBERG_FIELD_LENGTH.to_string(), length.to_string()));
            (OrcKind::Binary, ColumnShape::Bytes)
        }
        PrimitiveType::Binary => {
            extra.push((
                ICEBERG_BINARY_TYPE_ATTRIBUTE.to_string(),
                "BINARY".to_string(),
            ));
            (OrcKind::Binary, ColumnShape::Bytes)
        }
        PrimitiveType::Decimal { .. } => (OrcKind::Decimal, ColumnShape::Decimal),
        PrimitiveType::Unknown => return Err(unsupported(field, "unknown")),
    };

    let index = reserve(out, shape, Some(field.field_type.as_ref().clone()));
    out.types[index].kind = Some(kind);
    if let PrimitiveType::Decimal { precision, scale } = primitive {
        out.types[index].precision = Some(*precision);
        out.types[index].scale = Some(*scale);
    }
    let mut attributes = base_attributes(id, required);
    attributes.extend(extra);
    out.types[index].attributes = attributes;
    Ok(index)
}

fn reserve(out: &mut OrcSchema, shape: ColumnShape, iceberg_type: Option<Type>) -> usize {
    out.types.push(OrcType::default());
    out.columns.push(OrcColumn {
        shape,
        children: Vec::new(),
        iceberg_type,
    });
    out.types.len() - 1
}

fn base_attributes(field_id: Option<i32>, required: Option<bool>) -> Vec<(String, String)> {
    let mut attributes = Vec::new();
    if let Some(id) = field_id {
        attributes.push((ICEBERG_ID_ATTRIBUTE.to_string(), id.to_string()));
    }
    if let Some(required) = required {
        attributes.push((ICEBERG_REQUIRED_ATTRIBUTE.to_string(), required.to_string()));
    }
    attributes
}

fn unsupported(field: &NestedField, type_name: &str) -> Error {
    Error::new(
        ErrorKind::FeatureUnsupported,
        format!("ORC data writer does not support the {type_name} type"),
    )
    .with_context("field_id", field.id.to_string())
    .with_context("field_name", field.name.clone())
}

#[cfg(test)]
mod tests {
    include!("orc_type_tests.rs");
}
