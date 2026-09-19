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

pub(crate) mod commit;
pub(crate) mod conform;
pub(crate) mod cow_affected;
#[cfg(test)]
mod dangling_dv_delete_tests;
pub(crate) mod delete;
pub(crate) mod delete_legacy_merge;
pub(crate) mod expr_to_predicate;
#[cfg(test)]
mod list_null_tests;
pub(crate) mod metadata_scan;
#[cfg(test)]
mod occ_exec_tests;
#[cfg(test)]
mod page_prune_tests;
pub(crate) mod project;
pub(crate) mod promotion;
pub(crate) mod repartition;
pub(crate) mod row_lineage;
pub(crate) mod scan;
pub(crate) mod scan_helpers;
pub(crate) mod scan_knobs;
pub(crate) mod snapshot_target;
pub(crate) mod sort;
#[cfg(test)]
mod spark_fixture_tests;
pub(crate) mod update;
pub(crate) mod write;

pub(crate) const DATA_FILES_COL_NAME: &str = "data_files";
pub(crate) const WRITE_PARTITION_INDEX_COL_NAME: &str = "write_partition_index";

pub use project::project_with_partition;
pub use scan::{IcebergScanOptions, IcebergTableScan, ensure_iceberg_scan_options};
