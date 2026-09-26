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

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};

use crate::spec::{PrimitiveType, Schema, Type};
use crate::{Error, ErrorKind, Result};

const MAX_NESTING_DEPTH: usize = 128;

pub(crate) fn schema_has_unknown(schema: &Schema) -> bool {
    schema
        .as_struct()
        .fields()
        .iter()
        .any(|field| type_has_unknown(field.field_type.as_ref()))
}

fn type_has_unknown(root: &Type) -> bool {
    let mut stack: Vec<&Type> = vec![root];
    while let Some(current) = stack.pop() {
        match current {
            Type::Primitive(PrimitiveType::Unknown) => return true,
            Type::Struct(struct_type) => stack.extend(
                struct_type
                    .fields()
                    .iter()
                    .map(|field| field.field_type.as_ref()),
            ),
            Type::List(list_type) => stack.push(list_type.element_field.field_type.as_ref()),
            Type::Map(map_type) => {
                stack.push(map_type.key_field.field_type.as_ref());
                stack.push(map_type.value_field.field_type.as_ref());
            }
            Type::Primitive(_) | Type::Variant => {}
        }
    }
    false
}

fn is_unknown(ty: &Type) -> bool {
    matches!(ty, Type::Primitive(PrimitiveType::Unknown))
}

pub(crate) fn writer_arrow_schema(schema: &Schema) -> Result<ArrowSchema> {
    for field in schema.as_struct().fields() {
        refuse_unwritable_unknown(&field.name, field.field_type.as_ref(), 0)?;
    }
    let full: ArrowSchema = schema.try_into()?;
    Ok(ArrowSchema::new_with_metadata(
        strip_fields(full.fields(), 0)?,
        full.metadata().clone(),
    ))
}

fn refuse_unwritable_unknown(path: &str, ty: &Type, depth: usize) -> Result<()> {
    if depth > MAX_NESTING_DEPTH {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("write schema exceeds nesting depth {MAX_NESTING_DEPTH}"),
        ));
    }
    match ty {
        Type::Struct(struct_type) => {
            let fields = struct_type.fields();
            if !fields.is_empty() && fields.iter().all(|field| is_unknown(&field.field_type)) {
                return Err(Error::new(
                    ErrorKind::FeatureUnsupported,
                    format!(
                        "Cannot write struct '{path}': every field is unknown, and Parquet \
                         refuses an empty group"
                    ),
                ));
            }
            for field in fields {
                refuse_unwritable_unknown(
                    &format!("{path}.{}", field.name),
                    field.field_type.as_ref(),
                    depth + 1,
                )?;
            }
            Ok(())
        }
        Type::List(list_type) => {
            refuse_unknown_under(path, "element", list_type.element_field.field_type.as_ref())
        }
        Type::Map(map_type) => {
            refuse_unknown_under(path, "key", map_type.key_field.field_type.as_ref())?;
            refuse_unknown_under(path, "value", map_type.value_field.field_type.as_ref())
        }
        Type::Primitive(_) | Type::Variant => Ok(()),
    }
}

fn refuse_unknown_under(path: &str, role: &str, ty: &Type) -> Result<()> {
    if is_unknown(ty) {
        return Err(Error::new(
            ErrorKind::FeatureUnsupported,
            format!("Cannot convert {role} Parquet: unknown (column '{path}.{role}')"),
        ));
    }
    if type_has_unknown(ty) {
        return Err(Error::new(
            ErrorKind::FeatureUnsupported,
            format!(
                "Writing an unknown field under the {role} of '{path}' is not supported: \
                 only top-level and struct unknown fields are omitted from the Parquet file"
            ),
        ));
    }
    Ok(())
}

fn strip_fields(fields: &Fields, depth: usize) -> Result<Vec<Field>> {
    if depth > MAX_NESTING_DEPTH {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("write schema exceeds nesting depth {MAX_NESTING_DEPTH}"),
        ));
    }
    fields
        .iter()
        .filter(|field| !matches!(field.data_type(), DataType::Null))
        .map(|field| strip_field(field, depth))
        .collect()
}

fn strip_field(field: &Field, depth: usize) -> Result<Field> {
    match field.data_type() {
        DataType::Struct(children) => Ok(field
            .clone()
            .with_data_type(DataType::Struct(strip_fields(children, depth + 1)?.into()))),
        _ => Ok(field.clone()),
    }
}

pub(crate) fn project_batch(
    batch: &RecordBatch,
    writer_schema: &ArrowSchema,
) -> Result<RecordBatch> {
    let batch_schema = batch.schema();
    let mut columns = Vec::with_capacity(writer_schema.fields().len());
    for writer_field in writer_schema.fields() {
        let (index, batch_field) = batch_schema
            .fields()
            .iter()
            .enumerate()
            .find(|(_, field)| field.name() == writer_field.name())
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("write batch is missing column '{}'", writer_field.name()),
                )
            })?;
        columns.push(project_column(
            batch_field,
            writer_field,
            batch.column(index),
            0,
        )?);
    }
    RecordBatch::try_new(Arc::new(writer_schema.clone()), columns).map_err(|err| {
        Error::new(
            ErrorKind::Unexpected,
            "failed to build unknown-stripped write batch",
        )
        .with_source(err)
    })
}

fn project_column(
    batch_field: &Field,
    writer_field: &Field,
    column: &ArrayRef,
    depth: usize,
) -> Result<ArrayRef> {
    if depth > MAX_NESTING_DEPTH {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("write batch exceeds nesting depth {MAX_NESTING_DEPTH}"),
        ));
    }
    match (batch_field.data_type(), writer_field.data_type()) {
        (DataType::Struct(batch_children), DataType::Struct(writer_children)) => {
            let struct_array = column
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "write batch column '{}' is not a struct array",
                            writer_field.name()
                        ),
                    )
                })?;
            let mut children = Vec::with_capacity(writer_children.len());
            for writer_child in writer_children {
                let (index, batch_child) = batch_children
                    .iter()
                    .enumerate()
                    .find(|(_, field)| field.name() == writer_child.name())
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "write batch struct column '{}' is missing child '{}'",
                                writer_field.name(),
                                writer_child.name()
                            ),
                        )
                    })?;
                children.push(project_column(
                    batch_child,
                    writer_child,
                    struct_array.column(index),
                    depth + 1,
                )?);
            }
            let projected = StructArray::try_new(
                writer_children.clone(),
                children,
                struct_array.nulls().cloned(),
            )
            .map_err(|err| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "failed to project struct column '{}' for writing",
                        writer_field.name()
                    ),
                )
                .with_source(err)
            })?;
            Ok(Arc::new(projected))
        }
        _ => Ok(column.clone()),
    }
}
