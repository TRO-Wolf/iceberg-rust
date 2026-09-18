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

use datafusion::arrow::array::{
    Array, ArrayRef, ListArray, MapArray, RecordBatch, RecordBatchOptions, StructArray, make_array,
    new_null_array,
};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{
    DataType, Field as ArrowField, Fields, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef,
};
use datafusion::error::{DataFusionError, Result as DFResult};
use iceberg::{Error, ErrorKind};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::to_datafusion_error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ColumnSource {
    Scanned(String),
    Absent,
}

pub(crate) fn advertised_field_id(field: &ArrowField) -> DFResult<i32> {
    let raw = field
        .metadata()
        .get(PARQUET_FIELD_ID_META_KEY)
        .ok_or_else(|| {
            to_datafusion_error(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "column '{}' carries no `{PARQUET_FIELD_ID_META_KEY}` metadata, so it cannot be \
                     bound to a table field: an Iceberg column is identified by its field id, and \
                     matching on the name instead would read the wrong column after a rename",
                    field.name()
                ),
            ))
        })?;
    raw.parse::<i32>().map_err(|e| {
        to_datafusion_error(
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "column '{}' carries an unparsable `{PARQUET_FIELD_ID_META_KEY}` metadata value '{raw}'",
                    field.name()
                ),
            )
            .with_source(e),
        )
    })
}

pub(crate) fn is_arrow_promotion_allowed(from: &DataType, to: &DataType) -> bool {
    if from == to {
        return true;
    }
    match (from, to) {
        (DataType::Int32, DataType::Int64) => true,
        (DataType::Float32, DataType::Float64) => true,
        (
            DataType::Decimal128(from_precision, from_scale),
            DataType::Decimal128(to_precision, to_scale),
        )
        | (
            DataType::Decimal256(from_precision, from_scale),
            DataType::Decimal256(to_precision, to_scale),
        ) => from_scale == to_scale && from_precision <= to_precision,
        _ => false,
    }
}

pub(crate) fn conform_batch(
    batch: RecordBatch,
    advertised: &ArrowSchemaRef,
    sources: &[ColumnSource],
) -> DFResult<RecordBatch> {
    if batch.schema_ref() == advertised {
        return Ok(batch);
    }
    if sources.len() != advertised.fields().len() {
        return Err(datafusion::error::DataFusionError::Internal(format!(
            "the scan bound {} columns but advertises {}",
            sources.len(),
            advertised.fields().len()
        )));
    }

    let num_rows = batch.num_rows();
    let mut columns = Vec::with_capacity(advertised.fields().len());
    for (field, source) in advertised.fields().iter().zip(sources) {
        match source {
            ColumnSource::Scanned(name) => {
                let column = batch.column_by_name(name).ok_or_else(|| {
                    datafusion::error::DataFusionError::Internal(format!(
                        "the scan selected column '{name}' for advertised column '{}' but the \
                         scanned batch does not carry it",
                        field.name()
                    ))
                })?;
                columns.push(conform_column(column, field, field.name())?);
            }
            ColumnSource::Absent if field.is_nullable() => {
                columns.push(new_null_array(field.data_type(), num_rows))
            }
            ColumnSource::Absent => {
                return Err(to_datafusion_error(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "required column '{}' has no field with its id in the snapshot being \
                         scanned, so there is no data to read for it and a required column cannot \
                         be null-filled",
                        field.name()
                    ),
                )));
            }
        }
    }

    RecordBatch::try_new_with_options(
        advertised.clone(),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(num_rows)),
    )
    .map_err(|e| {
        datafusion::error::DataFusionError::ArrowError(
            Box::new(e),
            Some("failed to conform a scanned batch to the schema the plan advertised".to_string()),
        )
    })
}

pub(crate) fn conform_column(
    column: &ArrayRef,
    target: &ArrowField,
    path: &str,
) -> DFResult<ArrayRef> {
    if column.data_type() == target.data_type() {
        return Ok(column.clone());
    }
    if is_arrow_promotion_allowed(column.data_type(), target.data_type()) {
        return cast(column, target.data_type()).map_err(|e| {
            datafusion::error::DataFusionError::ArrowError(
                Box::new(e),
                Some(format!("promoting column '{path}'")),
            )
        });
    }

    match (column.data_type(), target.data_type()) {
        (DataType::Struct(scanned_fields), DataType::Struct(target_fields)) => {
            let scanned = downcast::<StructArray>(column, path)?;
            let len = scanned.len();

            let scanned_ids = scanned_fields
                .iter()
                .map(|field| advertised_field_id(field))
                .collect::<DFResult<Vec<_>>>()?;

            let mut children = Vec::with_capacity(target_fields.len());
            for target_child in target_fields {
                let child_path = format!("{path}.{}", target_child.name());
                let target_id = advertised_field_id(target_child)?;
                match scanned_ids.iter().position(|id| *id == target_id) {
                    Some(index) => children.push(conform_column(
                        scanned.column(index),
                        target_child,
                        &child_path,
                    )?),
                    None if target_child.is_nullable() => {
                        children.push(new_null_array(target_child.data_type(), len))
                    }
                    None => {
                        return Err(to_datafusion_error(Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "required field '{child_path}' has no field with its id in the \
                                 snapshot being scanned, so there is no data to read for it and a \
                                 required field cannot be null-filled"
                            ),
                        )));
                    }
                }
            }
            Ok(Arc::new(
                StructArray::try_new_with_length(
                    target_fields.clone(),
                    children,
                    scanned.nulls().cloned(),
                    len,
                )
                .map_err(|e| {
                    datafusion::error::DataFusionError::ArrowError(
                        Box::new(e),
                        Some(format!("conforming struct column '{path}'")),
                    )
                })?,
            ))
        }
        (DataType::List(_), DataType::List(target_element)) => {
            let scanned = downcast::<ListArray>(column, path)?;
            let values =
                conform_column(scanned.values(), target_element, &format!("{path}.element"))?;
            Ok(Arc::new(
                ListArray::try_new(
                    target_element.clone(),
                    scanned.offsets().clone(),
                    values,
                    scanned.nulls().cloned(),
                )
                .map_err(|e| {
                    datafusion::error::DataFusionError::ArrowError(
                        Box::new(e),
                        Some(format!("conforming list column '{path}'")),
                    )
                })?,
            ))
        }
        (DataType::Map(_, _), DataType::Map(target_entries, ordered)) => {
            let scanned = downcast::<MapArray>(column, path)?;
            let entries: ArrayRef = Arc::new(scanned.entries().clone());
            let conformed = conform_column(&entries, target_entries, path)?;
            let conformed = downcast::<StructArray>(&conformed, path)?.clone();
            Ok(Arc::new(
                MapArray::try_new(
                    target_entries.clone(),
                    scanned.offsets().clone(),
                    conformed,
                    scanned.nulls().cloned(),
                    *ordered,
                )
                .map_err(|e| {
                    datafusion::error::DataFusionError::ArrowError(
                        Box::new(e),
                        Some(format!("conforming map column '{path}'")),
                    )
                })?,
            ))
        }
        (scanned_type, target_type) => Err(to_datafusion_error(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "column '{path}' is {scanned_type} in the snapshot being scanned but {target_type} \
                 in the schema this query was planned against, and that is not a legal Iceberg type \
                 promotion — the data cannot be read as the planned type"
            ),
        ))),
    }
}

fn downcast<'a, T: 'static>(column: &'a ArrayRef, path: &str) -> DFResult<&'a T> {
    column.as_any().downcast_ref::<T>().ok_or_else(|| {
        datafusion::error::DataFusionError::Internal(format!(
            "column '{path}' has type {} but is not backed by the matching array kind",
            column.data_type()
        ))
    })
}

const MAX_STRIP_DEPTH: usize = 128;

pub(crate) fn strip_nested_metadata_from_schema(schema: &ArrowSchema) -> ArrowSchema {
    ArrowSchema::new_with_metadata(
        schema
            .fields()
            .iter()
            .map(|field| {
                Arc::new(
                    field
                        .as_ref()
                        .clone()
                        .with_data_type(strip_nested_metadata_from_type(field.data_type(), 0)),
                )
            })
            .collect::<Fields>(),
        schema.metadata().clone(),
    )
}

pub(crate) fn strip_nested_metadata_from_record_batch(batch: RecordBatch) -> DFResult<RecordBatch> {
    let num_rows = batch.num_rows();
    let schema = Arc::new(strip_nested_metadata_from_schema(&batch.schema()));
    let mut columns = Vec::with_capacity(schema.fields().len());
    for (column, field) in batch.columns().iter().zip(schema.fields()) {
        columns.push(strip_nested_metadata_from_column(
            column,
            field.data_type(),
            0,
        )?);
    }
    RecordBatch::try_new_with_options(
        schema,
        columns,
        &RecordBatchOptions::new().with_row_count(Some(num_rows)),
    )
    .map_err(|e| {
        datafusion::error::DataFusionError::ArrowError(
            Box::new(e),
            Some("failed to strip nested field metadata from a scanned batch".to_string()),
        )
    })
}

fn strip_nested_metadata_from_field(field: &ArrowField, depth: usize) -> ArrowField {
    ArrowField::new(
        field.name().clone(),
        strip_nested_metadata_from_type(field.data_type(), depth),
        field.is_nullable(),
    )
}

fn strip_nested_metadata_from_type(data_type: &DataType, depth: usize) -> DataType {
    if depth > MAX_STRIP_DEPTH {
        return data_type.clone();
    }
    match data_type {
        DataType::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|field| Arc::new(strip_nested_metadata_from_field(field, depth + 1)))
                .collect(),
        ),
        DataType::List(field) => {
            DataType::List(Arc::new(strip_nested_metadata_from_field(field, depth + 1)))
        }
        DataType::LargeList(field) => {
            DataType::LargeList(Arc::new(strip_nested_metadata_from_field(field, depth + 1)))
        }
        DataType::FixedSizeList(field, width) => DataType::FixedSizeList(
            Arc::new(strip_nested_metadata_from_field(field, depth + 1)),
            *width,
        ),
        DataType::Map(field, ordered) => DataType::Map(
            Arc::new(strip_nested_metadata_from_field(field, depth + 1)),
            *ordered,
        ),
        _ => data_type.clone(),
    }
}

fn strip_nested_metadata_from_column(
    column: &ArrayRef,
    target: &DataType,
    depth: usize,
) -> DFResult<ArrayRef> {
    if column.data_type() == target {
        return Ok(column.clone());
    }
    if depth > MAX_STRIP_DEPTH {
        return Err(DataFusionError::Internal(format!(
            "a nested column deeper than {MAX_STRIP_DEPTH} levels cannot be relabelled"
        )));
    }
    let target_fields = nested_metadata_fields(target);
    let data = column.to_data();
    if data.child_data().len() != target_fields.len() {
        return Err(DataFusionError::Internal(format!(
            "a scanned column of type {} does not carry the child arrays its declared type implies",
            column.data_type()
        )));
    }
    let mut children = Vec::with_capacity(target_fields.len());
    for (child_data, target_field) in data.child_data().iter().zip(target_fields) {
        children.push(
            strip_nested_metadata_from_column(
                &make_array(child_data.clone()),
                target_field.data_type(),
                depth + 1,
            )?
            .to_data(),
        );
    }
    Ok(make_array(unsafe {
        data.into_builder()
            .data_type(target.clone())
            .child_data(children)
            .build_unchecked()
    }))
}

fn nested_metadata_fields(data_type: &DataType) -> Vec<&ArrowField> {
    match data_type {
        DataType::Struct(fields) => fields.iter().map(|field| field.as_ref()).collect(),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => vec![field.as_ref()],
        _ => vec![],
    }
}
