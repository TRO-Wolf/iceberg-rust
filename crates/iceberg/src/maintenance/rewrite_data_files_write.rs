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

use futures::TryStreamExt;

use crate::arrow::{ArrowReaderBuilder, RecordBatchPartitionSplitter};
use crate::error::{Error, ErrorKind, Result};
use crate::maintenance::rewrite_data_files_router::BoundedPartitionRouter;
use crate::metadata_columns::{
    RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER, RESERVED_FIELD_ID_ROW_ID,
    format_supports_row_lineage, schema_with_row_lineage,
};
use crate::scan::FileScanTask;
use crate::spec::{DataFile, DataFileFormat, SchemaRef};
use crate::table::Table;
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};

pub(crate) struct CompactedWrite {
    pub files: Vec<DataFile>,
    #[allow(dead_code)]
    pub peak_open_partition_writers: usize,
}

pub(crate) async fn write_compacted_files(
    table: &Table,
    group: &[FileScanTask],
    target_file_size_bytes: u64,
    max_open_partition_writers: usize,
) -> Result<CompactedWrite> {
    if max_open_partition_writers == 0 {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            "'max-open-partition-writers' is set to 0 but must be > 0",
        ));
    }

    let schema = rewrite_write_schema(table)?;
    let spec = table.metadata().default_partition_spec().as_ref().clone();

    let location_generator = DefaultLocationGenerator::new(table.metadata().clone())?;
    let file_name_generator = DefaultFileNameGenerator::new(
        "compacted".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let rolling_builder = RollingFileWriterBuilder::new(
        parquet_builder,
        usize::try_from(target_file_size_bytes).unwrap_or(usize::MAX),
        table.file_io().clone(),
        location_generator,
        file_name_generator,
    );
    let writer_builder =
        DataFileWriterBuilder::new(rolling_builder).with_partition_spec(spec.clone());

    let carry_lineage = format_supports_row_lineage(table.metadata().format_version());
    let tasks: Vec<Result<FileScanTask>> = group
        .iter()
        .cloned()
        .map(|mut task| {
            task.predicate = None;
            if carry_lineage {
                project_row_lineage(&mut task);
            }
            Ok(task)
        })
        .collect();
    let task_stream = Box::pin(futures::stream::iter(tasks)) as crate::scan::FileScanTaskStream;
    let mut batch_stream = ArrowReaderBuilder::new(table.file_io().clone())
        .build()
        .read(task_stream)?;

    if spec.fields().is_empty() {
        let mut writer = writer_builder.build(None).await?;
        while let Some(batch) = batch_stream.try_next().await? {
            writer.write(batch).await?;
        }
        let files = writer.close().await?;
        return Ok(CompactedWrite {
            files,
            peak_open_partition_writers: 1,
        });
    }

    let splitter = RecordBatchPartitionSplitter::try_new_with_computed_values(
        schema.clone(),
        table.metadata().default_partition_spec().clone(),
    )?;
    let mut router = BoundedPartitionRouter::new(writer_builder, max_open_partition_writers)?;
    while let Some(batch) = batch_stream.try_next().await? {
        for (partition_key, partition_batch) in splitter.split(&batch)? {
            router.write(partition_key, partition_batch).await?;
        }
    }
    let peak_open_partition_writers = router.peak_open_partition_writers();
    let files = router.close().await?;
    Ok(CompactedWrite {
        files,
        peak_open_partition_writers,
    })
}

fn rewrite_write_schema(table: &Table) -> Result<SchemaRef> {
    let schema = table.metadata().current_schema();
    if format_supports_row_lineage(table.metadata().format_version()) {
        Ok(Arc::new(schema_with_row_lineage(schema)?))
    } else {
        Ok(schema.clone())
    }
}

fn project_row_lineage(task: &mut FileScanTask) {
    let mut ids = task.project_field_ids.to_vec();
    if !ids.contains(&RESERVED_FIELD_ID_ROW_ID) {
        ids.push(RESERVED_FIELD_ID_ROW_ID);
    }
    if !ids.contains(&RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER) {
        ids.push(RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER);
    }
    task.project_field_ids = Arc::from(ids);
}
