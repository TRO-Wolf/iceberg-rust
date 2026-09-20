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

use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, FieldRef, Schema as ArrowSchema};
use futures::TryStreamExt;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::ArrowReader;
use crate::metadata_columns::{
    RESERVED_COL_NAME_CHANGE_ORDINAL, RESERVED_COL_NAME_CHANGE_TYPE,
    RESERVED_COL_NAME_COMMIT_SNAPSHOT_ID, RESERVED_FIELD_ID_CHANGE_ORDINAL,
    RESERVED_FIELD_ID_CHANGE_TYPE, RESERVED_FIELD_ID_COMMIT_SNAPSHOT_ID,
};
use crate::scan::{
    ArrowRecordBatchStream, ChangelogOperation, ChangelogScanTask, ChangelogScanTaskStream,
    ChangelogTaskKind, FileScanTaskStream,
};
use crate::{Error, ErrorKind, Result};

#[derive(Clone)]
pub struct ChangelogReader {
    reader: ArrowReader,
}

impl ChangelogReader {
    #[must_use]
    pub fn new(reader: ArrowReader) -> Self {
        Self { reader }
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn read(self, tasks: ChangelogScanTaskStream) -> Result<ArrowRecordBatchStream> {
        let reader = self.reader;
        let stream = tasks
            .and_then(move |task| {
                let reader = reader.clone();
                async move { read_one_task(reader, task) }
            })
            .try_flatten();
        Ok(Box::pin(stream))
    }
}

#[must_use]
pub fn changelog_arrow_fields() -> Vec<FieldRef> {
    vec![
        reserved_field(
            RESERVED_COL_NAME_CHANGE_TYPE,
            DataType::Utf8,
            RESERVED_FIELD_ID_CHANGE_TYPE,
        ),
        reserved_field(
            RESERVED_COL_NAME_CHANGE_ORDINAL,
            DataType::Int32,
            RESERVED_FIELD_ID_CHANGE_ORDINAL,
        ),
        reserved_field(
            RESERVED_COL_NAME_COMMIT_SNAPSHOT_ID,
            DataType::Int64,
            RESERVED_FIELD_ID_COMMIT_SNAPSHOT_ID,
        ),
    ]
}

#[must_use]
pub fn changelog_arrow_schema(base: &ArrowSchema) -> ArrowSchema {
    let mut fields: Vec<FieldRef> = base.fields().iter().map(Arc::clone).collect();
    fields.extend(changelog_arrow_fields());
    ArrowSchema::new(fields)
}

fn reserved_field(name: &str, data_type: DataType, field_id: i32) -> FieldRef {
    Arc::new(
        Field::new(name, data_type, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            field_id.to_string(),
        )])),
    )
}

fn read_one_task(reader: ArrowReader, task: ChangelogScanTask) -> Result<ArrowRecordBatchStream> {
    if task.kind == ChangelogTaskKind::DeletedRows {
        return Err(Error::new(
            ErrorKind::FeatureUnsupported,
            "DeletedRows tasks are currently not supported in changelog scans",
        ));
    }
    let operation = task.operation();
    let change_ordinal = task.change_ordinal;
    let commit_snapshot_id = task.commit_snapshot_id;
    let file_scan_task = task.file_scan_task;
    let tasks: FileScanTaskStream =
        Box::pin(futures::stream::once(async move { Ok(file_scan_task) }));
    let batches = reader.read(tasks)?;
    Ok(Box::pin(batches.and_then(move |batch| {
        futures::future::ready(with_change_columns(
            &batch,
            operation,
            change_ordinal,
            commit_snapshot_id,
        ))
    })))
}

fn with_change_columns(
    batch: &RecordBatch,
    operation: ChangelogOperation,
    change_ordinal: i32,
    commit_snapshot_id: i64,
) -> Result<RecordBatch> {
    let rows = batch.num_rows();
    let schema = Arc::new(changelog_arrow_schema(batch.schema_ref()));
    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    columns.push(Arc::new(StringArray::from(vec![
        operation_name(operation);
        rows
    ])));
    columns.push(Arc::new(Int32Array::from(vec![change_ordinal; rows])));
    columns.push(Arc::new(Int64Array::from(vec![commit_snapshot_id; rows])));
    RecordBatch::try_new(schema, columns).map_err(|error| {
        Error::new(
            ErrorKind::Unexpected,
            "failed to append the changelog columns to a record batch",
        )
        .with_source(error)
    })
}

fn operation_name(operation: ChangelogOperation) -> &'static str {
    match operation {
        ChangelogOperation::Insert => "INSERT",
        ChangelogOperation::Delete => "DELETE",
        ChangelogOperation::UpdateBefore => "UPDATE_BEFORE",
        ChangelogOperation::UpdateAfter => "UPDATE_AFTER",
    }
}
