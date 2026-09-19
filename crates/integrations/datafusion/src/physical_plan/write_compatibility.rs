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

use datafusion::arrow::datatypes::{DataType, Field, Fields, SchemaRef};
use datafusion::common::Result as DFResult;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{CastExpr, Column};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::projection::ProjectionExec;

pub(super) const MAX_WRITE_COMPATIBILITY_DEPTH: usize = 64;

pub(super) fn field_is_write_compatible(input: &Field, expected: &Field, depth: usize) -> bool {
    input.name() == expected.name()
        && (!input.is_nullable() || expected.is_nullable())
        && data_type_is_write_compatible(input.data_type(), expected.data_type(), depth)
}

fn data_type_is_write_compatible(input: &DataType, expected: &DataType, depth: usize) -> bool {
    if depth >= MAX_WRITE_COMPATIBILITY_DEPTH {
        return input == expected;
    }
    let depth = depth + 1;
    match (input, expected) {
        (DataType::Struct(input_fields), DataType::Struct(expected_fields)) => {
            input_fields.len() == expected_fields.len()
                && input_fields
                    .iter()
                    .zip(expected_fields.iter())
                    .all(|(input, expected)| field_is_write_compatible(input, expected, depth))
        }
        (DataType::List(input_element), DataType::List(expected_element))
        | (DataType::LargeList(input_element), DataType::LargeList(expected_element)) => {
            field_is_write_compatible(input_element, expected_element, depth)
        }
        (
            DataType::FixedSizeList(input_element, input_len),
            DataType::FixedSizeList(expected_element, expected_len),
        ) => {
            input_len == expected_len
                && field_is_write_compatible(input_element, expected_element, depth)
        }
        (
            DataType::Map(input_entries, input_sorted),
            DataType::Map(expected_entries, expected_sorted),
        ) => {
            input_sorted == expected_sorted
                && field_is_write_compatible(input_entries, expected_entries, depth)
        }
        (input, expected) => input == expected || leaf_layout_compatible(input, expected),
    }
}

pub(super) fn leaf_layout_compatible(input: &DataType, expected: &DataType) -> bool {
    matches!(
        (input, expected),
        (
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        ) | (
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView,
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView
        )
    )
}

pub(super) fn canonical_layout_input(
    input: Arc<dyn ExecutionPlan>,
    expected_fields: &Fields,
) -> DFResult<(Arc<dyn ExecutionPlan>, SchemaRef)> {
    let input_schema = input.schema();
    let needs_cast = |input_field: &Arc<Field>, expected_field: &Arc<Field>| {
        input_field.data_type() != expected_field.data_type()
            && leaf_layout_compatible(input_field.data_type(), expected_field.data_type())
    };
    if !input_schema
        .fields()
        .iter()
        .zip(expected_fields.iter())
        .any(|(input, expected)| needs_cast(input, expected))
    {
        return Ok((input, input_schema));
    }
    let canonical_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = input_schema
        .fields()
        .iter()
        .zip(expected_fields.iter())
        .enumerate()
        .map(|(index, (input_field, expected_field))| {
            let expr: Arc<dyn PhysicalExpr> = if needs_cast(input_field, expected_field) {
                Arc::new(CastExpr::new(
                    Arc::new(Column::new(input_field.name(), index)),
                    expected_field.data_type().clone(),
                    None,
                ))
            } else {
                Arc::new(Column::new(input_field.name(), index))
            };
            (expr, input_field.name().clone())
        })
        .collect();
    let input = Arc::new(ProjectionExec::try_new(canonical_exprs, input)?);
    let input_schema = input.schema();
    Ok((input, input_schema))
}
