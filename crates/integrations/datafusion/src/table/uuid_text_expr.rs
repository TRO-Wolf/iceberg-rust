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

use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use datafusion::arrow::array::{Array, BooleanArray, RecordBatch};
use datafusion::arrow::compute::nullif;
use datafusion::arrow::datatypes::{DataType, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use datafusion::catalog::Session;
use datafusion::common::DFSchema;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::Expr;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::ColumnarValue;
use iceberg::spec::Schema as IcebergSchema;

use super::uuid_text::{
    arrow_schema_with_uuid_as_text, collect_uuid_field_ids, convert_column_uuid_text_to_bytes,
    render_batch_uuid_as_text,
};

#[derive(Debug, Clone)]
pub(crate) struct UuidTextExpr {
    inner: Arc<dyn PhysicalExpr>,
    text_schema: ArrowSchemaRef,
    byte_type: Option<DataType>,
    row_filter: Option<Arc<dyn PhysicalExpr>>,
}

impl PartialEq for UuidTextExpr {
    fn eq(&self, other: &Self) -> bool {
        self.inner.eq(&other.inner)
            && self.text_schema == other.text_schema
            && self.byte_type == other.byte_type
            && self.row_filter == other.row_filter
    }
}

impl Eq for UuidTextExpr {}

impl Hash for UuidTextExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
        self.byte_type.hash(state);
        self.row_filter.hash(state);
    }
}

impl Display for UuidTextExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "uuid_text({})", self.inner)
    }
}

impl PhysicalExpr for UuidTextExpr {
    fn data_type(&self, _input_schema: &ArrowSchema) -> DFResult<DataType> {
        match &self.byte_type {
            Some(byte_type) => Ok(byte_type.clone()),
            None => self.inner.data_type(&self.text_schema),
        }
    }

    fn nullable(&self, _input_schema: &ArrowSchema) -> DFResult<bool> {
        self.inner.nullable(&self.text_schema)
    }

    fn evaluate(&self, batch: &RecordBatch) -> DFResult<ColumnarValue> {
        let text_batch = render_batch_uuid_as_text(batch.clone(), &self.text_schema)?;
        let value = self.inner.evaluate(&text_batch)?;
        match &self.byte_type {
            None => Ok(value),
            Some(byte_type) => {
                let num_rows = batch.num_rows();
                let mut array = value.into_array(num_rows)?;
                if let Some(row_filter) = &self.row_filter {
                    let matched = row_filter.evaluate(&text_batch)?.into_array(num_rows)?;
                    let matched =
                        matched
                            .as_any()
                            .downcast_ref::<BooleanArray>()
                            .ok_or_else(|| {
                                DataFusionError::Internal(
                                    "UPDATE filter did not evaluate to a boolean".to_string(),
                                )
                            })?;
                    let unmatched: BooleanArray = (0..num_rows)
                        .map(|row| Some(!(matched.is_valid(row) && matched.value(row))))
                        .collect();
                    array = nullif(&array, &unmatched)
                        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
                }
                Ok(ColumnarValue::Array(convert_column_uuid_text_to_bytes(
                    &array, byte_type,
                )?))
            }
        }
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> DFResult<Arc<dyn PhysicalExpr>> {
        Ok(self)
    }

    fn fmt_sql(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

pub(crate) fn uuid_text_schema(
    byte_schema: &ArrowSchemaRef,
    schema: &IcebergSchema,
) -> ArrowSchemaRef {
    Arc::new(arrow_schema_with_uuid_as_text(
        byte_schema,
        &collect_uuid_field_ids(schema),
    ))
}

pub(crate) fn text_physical_expr(
    state: &dyn Session,
    expr: Expr,
    text_schema: &ArrowSchemaRef,
) -> DFResult<Arc<dyn PhysicalExpr>> {
    let df_schema = DFSchema::try_from(text_schema.as_ref().clone())?;
    state.create_physical_expr(expr, &df_schema)
}

pub(crate) fn uuid_text_expr(
    inner: Arc<dyn PhysicalExpr>,
    text_schema: &ArrowSchemaRef,
    byte_type: Option<DataType>,
    row_filter: Option<Arc<dyn PhysicalExpr>>,
) -> Arc<dyn PhysicalExpr> {
    Arc::new(UuidTextExpr {
        inner,
        text_schema: text_schema.clone(),
        byte_type,
        row_filter,
    })
}
