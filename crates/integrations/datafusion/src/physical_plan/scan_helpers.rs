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

use std::collections::{HashMap, HashSet};

use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::error::Result as DFResult;
use iceberg::expr::BoundPredicate;
use iceberg::metadata_columns::is_metadata_column_name;
use iceberg::scan::PartitionWork;
use iceberg::table::Table;

use super::conform::{ColumnSource, advertised_field_id};
use crate::to_datafusion_error;

pub(crate) fn exact_table_row_count(partitions: &[PartitionWork]) -> Option<usize> {
    let task_count: usize = partitions.iter().map(|work| work.tasks().count()).sum();
    let mut counted_paths: HashSet<&str> = HashSet::with_capacity(task_count);
    let mut total = 0usize;
    for task in partitions.iter().flat_map(PartitionWork::tasks) {
        if !task.deletes.is_empty()
            || !matches!(task.predicate(), None | Some(BoundPredicate::AlwaysTrue))
        {
            return None;
        }
        if counted_paths.insert(task.data_file_path()) {
            let file_records = usize::try_from(task.file_record_count?).ok()?;
            total = total.checked_add(file_records)?;
        }
    }
    Some(total)
}

pub(crate) fn resolve_bindings(
    table: &Table,
    snapshot_id: Option<i64>,
    schema: &ArrowSchemaRef,
) -> DFResult<HashMap<String, Option<String>>> {
    let metadata = table.metadata();
    let snapshot = match snapshot_id {
        Some(snapshot_id) => metadata.snapshot_by_id(snapshot_id),
        None => metadata.current_snapshot(),
    };
    let Some(snapshot) = snapshot else {
        return Ok(schema
            .fields()
            .iter()
            .map(|field| (field.name().clone(), Some(field.name().clone())))
            .collect());
    };
    let snapshot_schema = snapshot.schema(metadata).map_err(to_datafusion_error)?;

    let mut bindings = HashMap::with_capacity(schema.fields().len());
    for field in schema.fields() {
        if is_metadata_column_name(field.name()) {
            bindings.insert(field.name().clone(), Some(field.name().clone()));
            continue;
        }
        let field_id = advertised_field_id(field)?;
        bindings.insert(
            field.name().clone(),
            snapshot_schema
                .name_by_field_id(field_id)
                .map(str::to_string),
        );
    }
    Ok(bindings)
}

pub(crate) fn project_bindings(
    output_schema: &ArrowSchemaRef,
    bindings: &HashMap<String, Option<String>>,
) -> DFResult<(Vec<String>, Vec<ColumnSource>)> {
    let mut scan_columns = Vec::with_capacity(output_schema.fields().len());
    let mut sources = Vec::with_capacity(output_schema.fields().len());
    for field in output_schema.fields() {
        match bindings.get(field.name()) {
            Some(Some(name)) => {
                scan_columns.push(name.clone());
                sources.push(ColumnSource::Scanned(name.clone()));
            }
            Some(None) => sources.push(ColumnSource::Absent),
            None => {
                return Err(datafusion::error::DataFusionError::Internal(format!(
                    "projected column '{}' is not part of the schema the scan was built from",
                    field.name()
                )));
            }
        }
    }
    Ok((scan_columns, sources))
}
