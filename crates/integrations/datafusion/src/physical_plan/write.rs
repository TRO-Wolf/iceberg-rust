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

use std::fmt::{Debug, Formatter};
use std::str::FromStr;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, RecordBatch, StringArray, UInt64Array};
use datafusion::arrow::compute::SortOptions;
use datafusion::arrow::datatypes::{
    DataType, Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef,
};
use datafusion::common::Result as DFResult;
use datafusion::error::DataFusionError;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{
    EquivalenceProperties, LexOrdering, OrderingRequirements, Partitioning, PhysicalSortExpr,
};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::expressions::Column;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, ExecutionPlanProperties,
    PlanProperties, execute_input_stream,
};
use futures::StreamExt;
use iceberg::arrow::{FieldMatchMode, PROJECTED_PARTITION_VALUE_COLUMN};
use iceberg::spec::{
    DataFileFormat, MetricsConfig, PartitionSpecRef, TableProperties, serialize_data_file_to_json,
};
use iceberg::table::Table;
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::AnyFileWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, TableLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use uuid::Uuid;

use crate::physical_plan::sort::write_sort_plan;
use crate::physical_plan::{DATA_FILES_COL_NAME, WRITE_PARTITION_INDEX_COL_NAME};
use crate::task_writer::TaskWriter;
use crate::to_datafusion_error;

/// An execution plan node that writes data to an Iceberg table.
///
/// This execution plan takes input data from a child execution plan and writes it to an Iceberg table.
/// It handles the creation of data files in the appropriate format and returns information about the written files as its output.
#[derive(Debug)]
pub(crate) struct IcebergWriteExec {
    table: Table,
    input: Arc<dyn ExecutionPlan>,
    input_distribution: Distribution,
    input_ordering: Option<OrderingRequirements>,
    partition_spec: PartitionSpecRef,
    sort_order_id: Option<i32>,
    result_schema: ArrowSchemaRef,
    plan_properties: Arc<PlanProperties>,
}

impl IcebergWriteExec {
    /// Creates the write node.
    ///
    /// The node's advertised schema is its RESULT schema (the serialized data files), not the
    /// table's: `execute` emits result batches, and a node whose parents are planned against a
    /// schema it never emits is the BUG-011 skew in the write path. It also removes the last
    /// consumer of the provider's cached table schema from this branch of the plan.
    pub fn new(
        table: Table,
        input: Arc<dyn ExecutionPlan>,
        partition_spec: PartitionSpecRef,
        sort_order_id: Option<i32>,
    ) -> Self {
        let (input_distribution, input_ordering) =
            Self::input_requirements(&table, &input, &partition_spec);
        Self::new_with_requirements(
            table,
            input,
            input_distribution,
            input_ordering,
            partition_spec,
            sort_order_id,
        )
    }

    fn new_with_requirements(
        table: Table,
        input: Arc<dyn ExecutionPlan>,
        input_distribution: Distribution,
        input_ordering: Option<OrderingRequirements>,
        partition_spec: PartitionSpecRef,
        sort_order_id: Option<i32>,
    ) -> Self {
        let result_schema = Self::make_result_schema();
        let plan_properties = Self::compute_properties(&input, Arc::clone(&result_schema));

        Self {
            table,
            input,
            input_distribution,
            input_ordering,
            partition_spec,
            sort_order_id,
            result_schema,
            plan_properties,
        }
    }

    fn input_requirements(
        table: &Table,
        input: &Arc<dyn ExecutionPlan>,
        partition_spec: &PartitionSpecRef,
    ) -> (Distribution, Option<OrderingRequirements>) {
        let sort = write_sort_plan(table, input.schema().as_ref());
        let sort_ordering = sort
            .exprs
            .and_then(|exprs| LexOrdering::new(exprs).map(OrderingRequirements::from));
        if partition_spec.is_unpartitioned() {
            return (Distribution::UnspecifiedDistribution, sort_ordering);
        }

        let Ok(partition_column_index) = input.schema().index_of(PROJECTED_PARTITION_VALUE_COLUMN)
        else {
            return (Distribution::UnspecifiedDistribution, sort_ordering);
        };
        let partition_column = Arc::new(Column::new(
            PROJECTED_PARTITION_VALUE_COLUMN,
            partition_column_index,
        ));
        let distribution = Distribution::HashPartitioned(vec![partition_column.clone()]);
        let ordering = sort_ordering.or_else(|| {
            let fanout_enabled = table
                .metadata()
                .properties()
                .get(TableProperties::PROPERTY_DATAFUSION_WRITE_FANOUT_ENABLED)
                .and_then(|value| value.parse::<bool>().ok())
                .unwrap_or(TableProperties::PROPERTY_DATAFUSION_WRITE_FANOUT_ENABLED_DEFAULT);
            (!fanout_enabled)
                .then(|| {
                    LexOrdering::new(vec![PhysicalSortExpr {
                        expr: partition_column,
                        options: SortOptions::default(),
                    }])
                    .map(OrderingRequirements::from)
                })
                .flatten()
        });

        (distribution, ordering)
    }

    fn compute_properties(
        input: &Arc<dyn ExecutionPlan>,
        schema: ArrowSchemaRef,
    ) -> Arc<PlanProperties> {
        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(input.output_partitioning().partition_count()),
            EmissionType::Final,
            Boundedness::Bounded,
        ))
    }

    // Create a record batch with serialized data files
    fn make_result_batch(data_files: Vec<String>, partition: u64) -> DFResult<RecordBatch> {
        let len = data_files.len();
        let files_array = Arc::new(StringArray::from(data_files)) as ArrayRef;
        let index_array = Arc::new(UInt64Array::from_value(partition, len)) as ArrayRef;

        RecordBatch::try_new(Self::make_result_schema(), vec![files_array, index_array]).map_err(
            |e| {
                DataFusionError::ArrowError(
                    Box::new(e),
                    Some("Failed to make result batch".to_string()),
                )
            },
        )
    }

    fn make_result_schema() -> ArrowSchemaRef {
        // Define a schema.
        Arc::new(ArrowSchema::new(vec![
            Field::new(DATA_FILES_COL_NAME, DataType::Utf8, false),
            Field::new(WRITE_PARTITION_INDEX_COL_NAME, DataType::UInt64, false),
        ]))
    }
}

impl DisplayAs for IcebergWriteExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                write!(f, "IcebergWriteExec: table={}", self.table.identifier())
            }
            DisplayFormatType::Verbose => {
                write!(
                    f,
                    "IcebergWriteExec: table={}, result_schema={:?}",
                    self.table.identifier(),
                    self.result_schema
                )
            }
            DisplayFormatType::TreeRender => {
                write!(f, "IcebergWriteExec: table={}", self.table.identifier())
            }
        }
    }
}

impl ExecutionPlan for IcebergWriteExec {
    fn name(&self) -> &str {
        "IcebergWriteExec"
    }

    /// Prevents the introduction of additional `RepartitionExec` and processing input in parallel.
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![self.input_distribution.clone()]
    }

    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        vec![self.input_ordering.clone()]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        // Maintains ordering in the sense that the written file will reflect the ordering of the input.
        vec![true; self.children().len()]
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "IcebergWriteExec expects exactly one child, but provided {}",
                children.len()
            )));
        }

        Ok(Arc::new(Self::new_with_requirements(
            self.table.clone(),
            Arc::clone(&children[0]),
            self.input_distribution.clone(),
            self.input_ordering.clone(),
            self.partition_spec.clone(),
            self.sort_order_id,
        )))
    }

    /// Executes the write operation for the given partition.
    ///
    /// This function:
    /// 1. Sets up a data file writer based on the table's configuration
    /// 2. Processes input data from the child execution plan
    /// 3. Writes the data to files using the configured writer
    /// 4. Returns a stream containing information about the written data files
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let format_version = self.table.metadata().format_version();

        // Get typed table properties
        let table_props = self
            .table
            .metadata()
            .table_properties()
            .map_err(to_datafusion_error)?;

        // Check data file format
        let file_format = DataFileFormat::from_str(&table_props.write_format_default)
            .map_err(to_datafusion_error)?;
        let file_writer_builder = AnyFileWriterBuilder::for_format(
            file_format,
            self.table.metadata().current_schema().clone(),
            self.table.metadata().properties(),
            MetricsConfig::for_table(self.table.metadata()).map_err(to_datafusion_error)?,
            FieldMatchMode::Name,
        )
        .map_err(to_datafusion_error)?;
        let target_file_size = table_props.write_target_file_size_bytes;

        let file_io = self.table.file_io().clone();
        // todo location_gen and file_name_gen should be configurable
        let location_generator =
            TableLocationGenerator::new(self.table.metadata()).map_err(to_datafusion_error)?;
        // todo filename prefix/suffix should be configurable
        let file_name_generator =
            DefaultFileNameGenerator::new(Uuid::now_v7().to_string(), None, file_format);
        let rolling_writer_builder = RollingFileWriterBuilder::new(
            file_writer_builder,
            target_file_size,
            file_io,
            location_generator,
            file_name_generator,
        );
        let fanout_enabled = table_props.write_datafusion_fanout_enabled;
        let schema = self.table.metadata().current_schema().clone();
        let partition_spec = self.partition_spec.clone();
        let partition_type = partition_spec
            .partition_type(&schema)
            .map_err(to_datafusion_error)?;
        let mut data_file_writer_builder = DataFileWriterBuilder::new(rolling_writer_builder)
            .with_partition_spec(partition_spec.as_ref().clone());
        if let Some(sort_order_id) = self.sort_order_id {
            data_file_writer_builder = data_file_writer_builder.with_sort_order_id(sort_order_id);
        }
        let task_writer = TaskWriter::try_new(
            data_file_writer_builder,
            fanout_enabled,
            schema.clone(),
            partition_spec,
        )
        .map_err(to_datafusion_error)?;

        // Get input data
        let data = execute_input_stream(
            Arc::clone(&self.input),
            self.input.schema(), // input schema may have projected column `_partition`
            partition,
            Arc::clone(&context),
        )?;

        // Create write stream
        let stream = futures::stream::once(async move {
            let mut task_writer = task_writer;
            let mut input_stream = data;

            while let Some(batch) = input_stream.next().await {
                let batch = batch?;
                task_writer
                    .write(batch)
                    .await
                    .map_err(to_datafusion_error)?;
            }

            let data_files = task_writer.close().await.map_err(to_datafusion_error)?;

            // Convert builders to data files and then to JSON strings
            let data_files_strs: Vec<String> = data_files
                .into_iter()
                .map(|data_file| {
                    serialize_data_file_to_json(data_file, &partition_type, format_version)
                        .map_err(to_datafusion_error)
                })
                .collect::<DFResult<Vec<String>>>()?;

            Self::make_result_batch(data_files_strs, partition as u64)
        })
        .boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.result_schema),
            stream,
        )))
    }
}

#[cfg(test)]
#[path = "write_tests.rs"]
mod tests;
