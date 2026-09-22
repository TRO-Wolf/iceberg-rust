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

use datafusion::common::config::ConfigOptions;
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::execution::config::SessionConfig;
use iceberg::spec::DataFile;

#[allow(clippy::type_complexity)]
pub struct DataFileCommitOrder(
    pub Arc<dyn Fn(Vec<DataFile>, &ConfigOptions) -> Vec<DataFile> + Send + Sync>,
);

impl std::fmt::Debug for DataFileCommitOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("DataFileCommitOrder")
            .field(&std::any::type_name::<Self>())
            .finish()
    }
}

impl DataFileCommitOrder {
    pub fn new(
        f: impl Fn(Vec<DataFile>, &ConfigOptions) -> Vec<DataFile> + Send + Sync + 'static,
    ) -> Self {
        Self(Arc::new(f))
    }
}

pub(crate) fn apply_commit_order(
    files: Vec<DataFile>,
    config: &SessionConfig,
) -> DFResult<Vec<DataFile>> {
    let Some(order) = config.get_extension::<DataFileCommitOrder>() else {
        return Ok(files);
    };
    let mut expected: HashMap<String, usize> = HashMap::new();
    for file in &files {
        *expected.entry(file.file_path().to_string()).or_insert(0) += 1;
    }
    let reordered = (order.0)(files, config.options());
    if reordered.len() != expected.values().sum::<usize>() {
        return Err(DataFusionError::Internal(
            "data file commit order hook must return a permutation of its input files".to_string(),
        ));
    }
    for file in &reordered {
        match expected.get_mut(file.file_path()) {
            Some(count) if *count > 0 => *count -= 1,
            _ => {
                return Err(DataFusionError::Internal(
                    "data file commit order hook must return a permutation of its input files"
                        .to_string(),
                ));
            }
        }
    }
    Ok(reordered)
}

#[cfg(test)]
#[path = "commit_order_tests.rs"]
mod tests;
