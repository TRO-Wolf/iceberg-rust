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

use datafusion::arrow::array::{Array, ArrayRef, BooleanArray, Int64Builder, RecordBatch};
use datafusion::arrow::compute::filter;
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema, SchemaRef};
use datafusion::common::{DataFusionError, Result as DFResult};
use iceberg::arrow::{FieldMatchMode, PROJECTED_PARTITION_VALUE_COLUMN, PartitionValueCalculator};
use iceberg::expr::Predicate;
use iceberg::metadata_columns::{
    RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER,
    RESERVED_COL_NAME_ROW_ID, RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER,
    RESERVED_FIELD_ID_ROW_ID, format_supports_row_lineage, schema_with_row_lineage,
};
use iceberg::spec::{DataFile, DataFileFormat, FormatVersion, SchemaRef as IcebergSchemaRef};
use iceberg::table::Table;
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::task_writer::TaskWriter;
use crate::to_datafusion_error;

pub(super) fn table_write_schema(table: &Table) -> DFResult<IcebergSchemaRef> {
    let schema = table.metadata().current_schema().clone();
    if format_supports_row_lineage(table.metadata().format_version()) {
        Ok(Arc::new(
            schema_with_row_lineage(schema.as_ref()).map_err(crate::to_datafusion_error)?,
        ))
    } else {
        Ok(schema)
    }
}

pub(super) fn push_lineage_scan_columns(projection: &mut Vec<String>, version: FormatVersion) {
    if format_supports_row_lineage(version) {
        projection.push(RESERVED_COL_NAME_ROW_ID.to_string());
        projection.push(RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER.to_string());
    }
}

pub(super) fn filter_lineage_columns(
    batch: &RecordBatch,
    mask: &BooleanArray,
) -> DFResult<Option<(ArrayRef, ArrayRef)>> {
    let Some(row_id) = batch.column_by_name(RESERVED_COL_NAME_ROW_ID) else {
        return Ok(None);
    };
    let seq = batch
        .column_by_name(RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER)
        .ok_or_else(|| {
            DataFusionError::Internal(
                "scan projected _row_id without _last_updated_sequence_number".to_string(),
            )
        })?;
    let row_id =
        filter(row_id, mask).map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
    let seq = filter(seq, mask).map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
    Ok(Some((row_id, seq)))
}

pub(super) fn null_last_updated_where_true(
    last_updated: ArrayRef,
    mask: &BooleanArray,
) -> DFResult<ArrayRef> {
    let longs = last_updated
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int64Array>()
        .ok_or_else(|| {
            DataFusionError::Internal("_last_updated_sequence_number is not Int64".to_string())
        })?;
    let mut builder = Int64Builder::with_capacity(longs.len());
    for index in 0..longs.len() {
        if mask.value(index) {
            builder.append_null();
        } else if longs.is_valid(index) {
            builder.append_value(longs.value(index));
        } else {
            builder.append_null();
        }
    }
    Ok(Arc::new(builder.finish()))
}

pub(super) fn table_prefix_batch(
    batch: &RecordBatch,
    table_field_count: usize,
) -> DFResult<RecordBatch> {
    if batch.num_columns() < table_field_count {
        return Err(DataFusionError::Internal(format!(
            "write batch has {} columns, expected at least {table_field_count} table columns",
            batch.num_columns()
        )));
    }
    let fields: Vec<Arc<Field>> = batch
        .schema()
        .fields()
        .iter()
        .take(table_field_count)
        .cloned()
        .collect();
    let columns: Vec<ArrayRef> = batch.columns()[..table_field_count].to_vec();
    RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

pub(super) fn attach_update_lineage(
    scan_batch: &RecordBatch,
    keep: &BooleanArray,
    rewritten: RecordBatch,
    null_last_updated_where: Option<&BooleanArray>,
) -> DFResult<RecordBatch> {
    let Some((row_id, last_updated)) = filter_lineage_columns(scan_batch, keep)? else {
        return Ok(rewritten);
    };
    let last_updated = match null_last_updated_where {
        Some(mask) => null_last_updated_where_true(last_updated, mask)?,
        None => {
            let all_updated = BooleanArray::from(vec![true; last_updated.len()]);
            null_last_updated_where_true(last_updated, &all_updated)?
        }
    };
    attach_lineage(rewritten, row_id, last_updated)
}

pub(super) async fn cow_scan_stream(
    table: &Table,
    table_schema: &SchemaRef,
    scan_snapshot_id: Option<i64>,
    prune: Option<Predicate>,
) -> DFResult<iceberg::scan::ArrowRecordBatchStream> {
    let mut projection: Vec<String> = table_schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    projection.push(RESERVED_COL_NAME_FILE.to_string());
    push_lineage_scan_columns(&mut projection, table.metadata().format_version());

    let mut builder = table.scan().select(projection);
    if let Some(snapshot_id) = scan_snapshot_id {
        builder = builder.snapshot_id(snapshot_id);
    }
    if let Some(prune) = prune {
        builder = builder.with_file_prune_only(prune);
    }
    builder
        .build()
        .map_err(crate::to_datafusion_error)?
        .to_arrow()
        .await
        .map_err(crate::to_datafusion_error)
}

pub(super) fn attach_lineage(
    table_batch: RecordBatch,
    row_id: ArrayRef,
    last_updated: ArrayRef,
) -> DFResult<RecordBatch> {
    let mut fields: Vec<Arc<Field>> = table_batch.schema().fields().iter().cloned().collect();
    let mut columns = table_batch.columns().to_vec();
    fields.push(Arc::new(lineage_arrow_field(
        RESERVED_COL_NAME_ROW_ID,
        RESERVED_FIELD_ID_ROW_ID,
    )));
    fields.push(Arc::new(lineage_arrow_field(
        RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER,
        RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER,
    )));
    columns.push(row_id);
    columns.push(last_updated);
    RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

fn lineage_arrow_field(name: &'static str, field_id: i32) -> Field {
    Field::new(name, DataType::Int64, true).with_metadata(HashMap::from([(
        PARQUET_FIELD_ID_META_KEY.to_string(),
        field_id.to_string(),
    )]))
}

type DmlDataFileWriterBuilder =
    DataFileWriterBuilder<ParquetWriterBuilder, DefaultLocationGenerator, DefaultFileNameGenerator>;

pub(super) struct StreamingDataFileWriter {
    writer: Option<TaskWriter<DmlDataFileWriterBuilder>>,
    schema: IcebergSchemaRef,
    table_field_count: usize,
    partition_spec: iceberg::spec::PartitionSpecRef,
    calculator: Option<PartitionValueCalculator>,
    builder: Option<DmlDataFileWriterBuilder>,
}

impl StreamingDataFileWriter {
    pub(super) fn try_new(table: &Table) -> DFResult<Self> {
        let table_schema = table.metadata().current_schema().clone();
        let table_field_count = table_schema.as_struct().fields().len();
        let schema = table_write_schema(table)?;
        let partition_spec = table.metadata().default_partition_spec().clone();

        let parquet_builder = ParquetWriterBuilder::new_with_match_mode(
            parquet::file::properties::WriterProperties::default(),
            schema.clone(),
            FieldMatchMode::Name,
        );
        let location_gen =
            DefaultLocationGenerator::new(table.metadata().clone()).map_err(to_datafusion_error)?;
        let file_name_gen = DefaultFileNameGenerator::new(
            uuid::Uuid::now_v7().to_string(),
            None,
            DataFileFormat::Parquet,
        );
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            table.file_io().clone(),
            location_gen,
            file_name_gen,
        );
        let builder = DataFileWriterBuilder::new(rolling)
            .with_partition_spec(partition_spec.as_ref().clone());

        let calculator = if partition_spec.is_unpartitioned() {
            None
        } else {
            Some(
                PartitionValueCalculator::try_new(&partition_spec, &table_schema)
                    .map_err(to_datafusion_error)?,
            )
        };

        Ok(Self {
            writer: None,
            schema,
            table_field_count,
            partition_spec,
            calculator,
            builder: Some(builder),
        })
    }

    fn ensure_writer(&mut self) -> DFResult<&mut TaskWriter<DmlDataFileWriterBuilder>> {
        if self.writer.is_none() {
            let builder = self.builder.take().ok_or_else(|| {
                DataFusionError::Internal(
                    "StreamingDataFileWriter builder already consumed".to_string(),
                )
            })?;
            let writer = TaskWriter::try_new(
                builder,
                true,
                self.schema.clone(),
                self.partition_spec.clone(),
            )
            .map_err(to_datafusion_error)?;
            self.writer = Some(writer);
        }
        self.writer.as_mut().ok_or_else(|| {
            DataFusionError::Internal("StreamingDataFileWriter not initialized".into())
        })
    }

    pub(super) async fn write_batch(&mut self, batch: RecordBatch) -> DFResult<()> {
        if self.partition_spec.is_unpartitioned() {
            self.ensure_writer()?
                .write(batch)
                .await
                .map_err(to_datafusion_error)
        } else {
            let calculator = self.calculator.as_ref().ok_or_else(|| {
                DataFusionError::Internal(
                    "StreamingDataFileWriter partition calculator missing".to_string(),
                )
            })?;
            let partition_source = table_prefix_batch(&batch, self.table_field_count)?;
            let partition_array = calculator
                .calculate(&partition_source)
                .map_err(to_datafusion_error)?;

            let partition_field = Field::new(
                PROJECTED_PARTITION_VALUE_COLUMN,
                partition_array.data_type().clone(),
                false,
            );
            let extended_schema = Arc::new(ArrowSchema::new(
                batch
                    .schema()
                    .fields()
                    .iter()
                    .cloned()
                    .chain(std::iter::once(Arc::new(partition_field)))
                    .collect::<Vec<_>>(),
            ));
            let mut extended_columns: Vec<ArrayRef> = batch.columns().to_vec();
            extended_columns.push(partition_array);
            let extended_batch = RecordBatch::try_new(extended_schema, extended_columns)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;

            self.ensure_writer()?
                .write(extended_batch)
                .await
                .map_err(to_datafusion_error)
        }
    }

    pub(super) async fn finish(self) -> DFResult<Vec<DataFile>> {
        match self.writer {
            None => Ok(Vec::new()),
            Some(writer) => writer.close().await.map_err(to_datafusion_error),
        }
    }
}
