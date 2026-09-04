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
use std::sync::{Arc, Mutex};

use datafusion::common::{DataFusionError, Result as DFResult};
use iceberg::expr::Predicate;
use iceberg::scan::ArrowRecordBatchStream;
use iceberg::spec::Struct;
use iceberg::table::Table;

use super::MergeOnReadDeleteKind;
use crate::to_datafusion_error;

pub(super) async fn mor_scan_stream(
    table: &Table,
    projection: Vec<String>,
    prune: Option<Predicate>,
    scan_snapshot_id: Option<i64>,
) -> DFResult<(
    ArrowRecordBatchStream,
    Arc<Mutex<HashMap<String, (i32, Struct)>>>,
)> {
    let mut builder = table.scan().select(projection);
    if let Some(snapshot_id) = scan_snapshot_id {
        builder = builder.snapshot_id(snapshot_id);
    }
    if let Some(prune) = prune {
        builder = builder.with_file_prune_only(prune);
    }
    let scan = builder.build().map_err(to_datafusion_error)?;
    scan.to_arrow_with_file_partitions()
        .await
        .map_err(to_datafusion_error)
}

pub(super) fn dv_partitions_for(
    kind: MergeOnReadDeleteKind,
    pairs: &[(String, i64)],
    shared: &Mutex<HashMap<String, (i32, Struct)>>,
) -> DFResult<HashMap<String, (i32, Struct)>> {
    match kind {
        MergeOnReadDeleteKind::PositionDeletes => Ok(HashMap::new()),
        MergeOnReadDeleteKind::DeletionVectors => {
            let touched: HashSet<&str> =
                pairs.iter().map(|(path, _)| path.as_str()).collect();
            let mut known = shared
                .lock()
                .map_err(|error| {
                    DataFusionError::Internal(format!("partition map lock failed: {error}"))
                })?
                .clone();
            known.retain(|path, _| touched.contains(path.as_str()));
            Ok(known)
        }
    }
}
