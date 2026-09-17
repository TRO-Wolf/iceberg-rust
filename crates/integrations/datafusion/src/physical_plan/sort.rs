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

//! Partition-based sorting for Iceberg tables.

use std::sync::Arc;

use datafusion::arrow::array::{Array, RecordBatch};
use datafusion::arrow::compute::{SortOptions, cast};
use datafusion::arrow::datatypes::{DataType, Schema as ArrowSchema};
use datafusion::common::Result as DFResult;
use datafusion::error::DataFusionError;
use datafusion::physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
use datafusion::physical_plan::expressions::Column;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::{ColumnarValue, ExecutionPlan};
use iceberg::arrow::{PROJECTED_PARTITION_VALUE_COLUMN, type_to_arrow_type};
use iceberg::spec::{NullOrder, SortDirection, TableProperties, Transform};
use iceberg::table::Table;
use iceberg::transform::create_transform_function;
use iceberg::{Error, ErrorKind};

use crate::to_datafusion_error;

/// Sorts an ExecutionPlan by partition values for Iceberg tables.
///
/// This function takes an input ExecutionPlan that has been extended with partition values
/// (via `project_with_partition`) and returns a SortExec that sorts by the partition column.
/// The partition values are expected to be in a struct column named `PROJECTED_PARTITION_VALUE_COLUMN`.
///
/// For unpartitioned tables or plans without the partition column, returns an error.
///
/// # Arguments
/// * `input` - The input ExecutionPlan with projected partition values
///
/// # Returns
/// * `Ok(Arc<dyn ExecutionPlan>)` - A SortExec that sorts by partition values
/// * `Err` - If the partition column is not found
pub(crate) fn sort_by_partition(input: Arc<dyn ExecutionPlan>) -> DFResult<Arc<dyn ExecutionPlan>> {
    let schema = input.schema();

    // Find the partition column in the schema
    let (partition_column_index, _partition_field) = schema
        .column_with_name(PROJECTED_PARTITION_VALUE_COLUMN)
        .ok_or_else(|| {
            DataFusionError::Plan(format!(
                "Partition column '{PROJECTED_PARTITION_VALUE_COLUMN}' not found in schema. Ensure the plan has been extended with partition values using project_with_partition."
            ))
        })?;

    // Create a single sort expression for the partition column
    let column_expr = Arc::new(Column::new(
        PROJECTED_PARTITION_VALUE_COLUMN,
        partition_column_index,
    ));

    let sort_expr = PhysicalSortExpr {
        expr: column_expr,
        options: SortOptions::default(), // Ascending, nulls last
    };

    // Create a SortExec with preserve_partitioning=true to ensure the output partitioning
    // is the same as the input partitioning, and the data is sorted within each partition
    let lex_ordering = LexOrdering::new(vec![sort_expr]).ok_or_else(|| {
        DataFusionError::Plan("Failed to create LexOrdering from sort expression".to_string())
    })?;

    let sort_exec = SortExec::new(lex_ordering, input).with_preserve_partitioning(true);

    Ok(Arc::new(sort_exec))
}

#[derive(Debug, Clone)]
struct SortTransformExpr {
    source: Arc<dyn PhysicalExpr>,
    transform: Transform,
    source_type: DataType,
    result_type: DataType,
}

impl PartialEq for SortTransformExpr {
    fn eq(&self, other: &Self) -> bool {
        self.source.eq(&other.source)
            && self.transform == other.transform
            && self.source_type == other.source_type
            && self.result_type == other.result_type
    }
}

impl Eq for SortTransformExpr {}

impl std::fmt::Display for SortTransformExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", self.transform, self.source)
    }
}

impl std::hash::Hash for SortTransformExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.source.hash(state);
        self.transform.hash(state);
        self.source_type.hash(state);
        self.result_type.hash(state);
    }
}

impl PhysicalExpr for SortTransformExpr {
    fn data_type(&self, _input_schema: &ArrowSchema) -> DFResult<DataType> {
        Ok(self.result_type.clone())
    }

    fn nullable(&self, _input_schema: &ArrowSchema) -> DFResult<bool> {
        Ok(true)
    }

    fn evaluate(&self, batch: &RecordBatch) -> DFResult<ColumnarValue> {
        let array = self.source.evaluate(batch)?.into_array(batch.num_rows())?;
        let array = if array.data_type() == &self.source_type {
            array
        } else {
            cast(&array, &self.source_type)?
        };
        let transformed = create_transform_function(&self.transform)
            .map_err(to_datafusion_error)?
            .transform(array)
            .map_err(to_datafusion_error)?;
        Ok(ColumnarValue::Array(transformed))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.source]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> DFResult<Arc<dyn PhysicalExpr>> {
        let [source] = children.try_into().map_err(|children: Vec<_>| {
            DataFusionError::Internal(format!(
                "SortTransformExpr expects exactly one child, got {}",
                children.len()
            ))
        })?;
        Ok(Arc::new(SortTransformExpr {
            source,
            transform: self.transform,
            source_type: self.source_type.clone(),
            result_type: self.result_type.clone(),
        }))
    }

    fn fmt_sql(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

pub(crate) struct WriteSort {
    pub(crate) exprs: Option<Vec<PhysicalSortExpr>>,
    pub(crate) sort_order_id: Option<i32>,
}

pub(crate) fn write_sort_plan(table: &Table, input_schema: &ArrowSchema) -> WriteSort {
    let order = table.metadata().default_sort_order();
    if order.is_unsorted() {
        return WriteSort {
            exprs: None,
            sort_order_id: Some(0),
        };
    }
    let Ok(order_id) = i32::try_from(order.order_id) else {
        return WriteSort {
            exprs: None,
            sort_order_id: None,
        };
    };
    let iceberg_schema = table.metadata().current_schema();
    let mut keys = Vec::with_capacity(order.fields.len());
    for field in &order.fields {
        if field.transform == Transform::Void {
            continue;
        }
        let Some(source) = iceberg_schema
            .as_struct()
            .fields()
            .iter()
            .find(|candidate| candidate.id == field.source_id)
        else {
            return WriteSort {
                exprs: None,
                sort_order_id: None,
            };
        };
        let Ok(index) = input_schema.index_of(&source.name) else {
            return WriteSort {
                exprs: None,
                sort_order_id: None,
            };
        };
        let expr: Arc<dyn PhysicalExpr> = if field.transform == Transform::Identity {
            Arc::new(Column::new(&source.name, index))
        } else {
            let Ok(source_type) = type_to_arrow_type(source.field_type.as_ref()) else {
                return WriteSort {
                    exprs: None,
                    sort_order_id: None,
                };
            };
            let Ok(transformed_type) = field.transform.result_type(source.field_type.as_ref())
            else {
                return WriteSort {
                    exprs: None,
                    sort_order_id: None,
                };
            };
            let Ok(result_type) = type_to_arrow_type(&transformed_type) else {
                return WriteSort {
                    exprs: None,
                    sort_order_id: None,
                };
            };
            if create_transform_function(&field.transform).is_err() {
                return WriteSort {
                    exprs: None,
                    sort_order_id: None,
                };
            }
            Arc::new(SortTransformExpr {
                source: Arc::new(Column::new(&source.name, index)),
                transform: field.transform,
                source_type,
                result_type,
            })
        };
        keys.push(PhysicalSortExpr {
            expr,
            options: SortOptions {
                descending: field.direction == SortDirection::Descending,
                nulls_first: field.null_order == NullOrder::First,
            },
        });
    }
    if keys.is_empty() {
        return WriteSort {
            exprs: None,
            sort_order_id: Some(order_id),
        };
    }
    if let Ok(partition_index) = input_schema.index_of(PROJECTED_PARTITION_VALUE_COLUMN) {
        keys.insert(0, PhysicalSortExpr {
            expr: Arc::new(Column::new(
                PROJECTED_PARTITION_VALUE_COLUMN,
                partition_index,
            )),
            options: SortOptions::default(),
        });
    }
    WriteSort {
        exprs: Some(keys),
        sort_order_id: Some(order_id),
    }
}

pub(crate) fn sort_for_write(
    input: Arc<dyn ExecutionPlan>,
    table: &Table,
) -> DFResult<(Arc<dyn ExecutionPlan>, Option<i32>)> {
    let sort = write_sort_plan(table, input.schema().as_ref());
    let Some(exprs) = sort.exprs else {
        return Ok((
            write_input_without_default_sort(input, table)?,
            sort.sort_order_id,
        ));
    };
    let lex_ordering = LexOrdering::new(exprs).ok_or_else(|| {
        DataFusionError::Plan("default sort order produced no sort expressions".to_string())
    })?;
    let sort_exec = SortExec::new(lex_ordering, input).with_preserve_partitioning(true);
    Ok((Arc::new(sort_exec), sort.sort_order_id))
}

fn write_input_without_default_sort(
    input: Arc<dyn ExecutionPlan>,
    table: &Table,
) -> DFResult<Arc<dyn ExecutionPlan>> {
    let fanout_enabled = table
        .metadata()
        .properties()
        .get(TableProperties::PROPERTY_DATAFUSION_WRITE_FANOUT_ENABLED)
        .map(|value| {
            value
                .parse::<bool>()
                .map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Invalid value for {}, expected 'true' or 'false'",
                            TableProperties::PROPERTY_DATAFUSION_WRITE_FANOUT_ENABLED
                        ),
                    )
                    .with_source(e)
                })
                .map_err(to_datafusion_error)
        })
        .transpose()?
        .unwrap_or(TableProperties::PROPERTY_DATAFUSION_WRITE_FANOUT_ENABLED_DEFAULT);
    if fanout_enabled {
        Ok(input)
    } else {
        sort_by_partition(input)
    }
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::{Int32Array, RecordBatch, StringArray, StructArray};
    use datafusion::arrow::datatypes::{DataType, Field, Fields, Schema as ArrowSchema};
    use datafusion::datasource::{MemTable, TableProvider};
    use datafusion::prelude::SessionContext;

    use super::*;

    #[tokio::test]
    async fn test_sort_by_partition_basic() {
        // Create a schema with a partition column
        let partition_fields =
            Fields::from(vec![Field::new("id_partition", DataType::Int32, false)]);

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            Field::new(
                PROJECTED_PARTITION_VALUE_COLUMN,
                DataType::Struct(partition_fields.clone()),
                false,
            ),
        ]));

        // Create test data with partition values
        let id_array = Arc::new(Int32Array::from(vec![3, 1, 2]));
        let name_array = Arc::new(StringArray::from(vec!["c", "a", "b"]));
        let partition_array = Arc::new(StructArray::from(vec![(
            Arc::new(Field::new("id_partition", DataType::Int32, false)),
            Arc::new(Int32Array::from(vec![3, 1, 2])) as _,
        )]));

        let batch =
            RecordBatch::try_new(schema.clone(), vec![id_array, name_array, partition_array])
                .unwrap();

        let ctx = SessionContext::new();
        let mem_table = MemTable::try_new(schema.clone(), vec![vec![batch]]).unwrap();
        let input = mem_table.scan(&ctx.state(), None, &[], None).await.unwrap();

        // Apply sort
        let sorted_plan = sort_by_partition(input).unwrap();

        // Execute and verify
        let result = datafusion::physical_plan::collect(sorted_plan, ctx.task_ctx())
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let result_batch = &result[0];

        let id_col = result_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

        // Verify data is sorted by partition value
        assert_eq!(id_col.value(0), 1);
        assert_eq!(id_col.value(1), 2);
        assert_eq!(id_col.value(2), 3);
    }

    #[tokio::test]
    async fn test_sort_by_partition_missing_column() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ])
        .unwrap();

        let ctx = SessionContext::new();
        let mem_table = MemTable::try_new(schema.clone(), vec![vec![batch]]).unwrap();
        let input = mem_table.scan(&ctx.state(), None, &[], None).await.unwrap();

        let result = sort_by_partition(input);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Partition column '_partition' not found")
        );
    }

    #[tokio::test]
    async fn test_sort_by_partition_multi_field() {
        // Test with multiple partition fields in the struct
        let partition_fields = Fields::from(vec![
            Field::new("year", DataType::Int32, false),
            Field::new("month", DataType::Int32, false),
        ]);

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("data", DataType::Utf8, false),
            Field::new(
                PROJECTED_PARTITION_VALUE_COLUMN,
                DataType::Struct(partition_fields.clone()),
                false,
            ),
        ]));

        // Create test data with partition values (year, month)
        let id_array = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
        let data_array = Arc::new(StringArray::from(vec!["a", "b", "c", "d"]));

        // Partition values: (2024, 2), (2024, 1), (2023, 12), (2024, 1)
        let year_array = Arc::new(Int32Array::from(vec![2024, 2024, 2023, 2024]));
        let month_array = Arc::new(Int32Array::from(vec![2, 1, 12, 1]));

        let partition_array = Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("year", DataType::Int32, false)),
                year_array as _,
            ),
            (
                Arc::new(Field::new("month", DataType::Int32, false)),
                month_array as _,
            ),
        ]));

        let batch =
            RecordBatch::try_new(schema.clone(), vec![id_array, data_array, partition_array])
                .unwrap();

        let ctx = SessionContext::new();
        let mem_table = MemTable::try_new(schema.clone(), vec![vec![batch]]).unwrap();
        let input = mem_table.scan(&ctx.state(), None, &[], None).await.unwrap();

        // Apply sort
        let sorted_plan = sort_by_partition(input).unwrap();

        // Execute and verify
        let result = datafusion::physical_plan::collect(sorted_plan, ctx.task_ctx())
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let result_batch = &result[0];

        let id_col = result_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

        // Verify data is sorted by partition value (struct comparison)
        // Expected order: (2023, 12), (2024, 1), (2024, 1), (2024, 2)
        // Which corresponds to ids: 3, 2, 4, 1
        assert_eq!(id_col.value(0), 3);
        assert_eq!(id_col.value(1), 2);
        assert_eq!(id_col.value(2), 4);
        assert_eq!(id_col.value(3), 1);
    }
}
