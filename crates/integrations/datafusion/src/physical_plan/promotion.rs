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

use datafusion::arrow::array::{ArrayRef, RecordBatch};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{DataType, SchemaRef};
use datafusion::arrow::error::ArrowError;

pub(crate) fn widened_batch(
    table_schema: &SchemaRef,
    columns: Vec<ArrayRef>,
) -> Result<RecordBatch, ArrowError> {
    let columns = columns
        .into_iter()
        .zip(table_schema.fields())
        .map(
            |(column, field)| match (column.data_type(), field.data_type()) {
                (DataType::Int32, DataType::Int64) | (DataType::Float32, DataType::Float64) => {
                    cast(&column, field.data_type())
                }
                (
                    DataType::Decimal128(precision, scale),
                    DataType::Decimal128(to_precision, to_scale),
                ) if scale == to_scale && precision < to_precision => {
                    cast(&column, field.data_type())
                }
                _ => Ok(column),
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(Arc::clone(table_schema), columns)
}
