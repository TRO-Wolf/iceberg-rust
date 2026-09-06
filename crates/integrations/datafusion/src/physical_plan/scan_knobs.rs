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

use datafusion::common::config::{ConfigEntry, ConfigExtension, ExtensionOptions};
use datafusion::error::DataFusionError;
use datafusion::execution::TaskContext;

/// Iceberg-specific scan knobs registered on DataFusion [`ConfigOptions`], prefix `iceberg.`.
///
/// | Knob | Default | Meaning |
/// |---|---|---|
/// | `multi_partition_scan` | `true` | `false` forces `T = 1` without touching `target_partitions` |
/// | `data_file_concurrency` | `0` | the budget `L`; `0` derives it from `target_partitions` |
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct IcebergScanOptions {
    /// `false` disables multi-partition output, whatever `target_partitions` says.
    pub multi_partition_scan: bool,
    /// Total data-file concurrency budget `L`. Zero → use `target_partitions`.
    pub data_file_concurrency: usize,
}

impl Default for IcebergScanOptions {
    fn default() -> Self {
        Self {
            multi_partition_scan: true,
            data_file_concurrency: 0,
        }
    }
}

impl ExtensionOptions for IcebergScanOptions {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, key: &str, value: &str) -> datafusion::common::Result<()> {
        match key {
            "multi_partition_scan" => {
                self.multi_partition_scan = value.parse().map_err(|e| {
                    DataFusionError::Configuration(format!(
                        "invalid iceberg.multi_partition_scan={value}: {e}"
                    ))
                })?;
            }
            "data_file_concurrency" => {
                self.data_file_concurrency = value.parse().map_err(|e| {
                    DataFusionError::Configuration(format!(
                        "invalid iceberg.data_file_concurrency={value}: {e}"
                    ))
                })?;
            }
            _ => {
                return Err(DataFusionError::Configuration(format!(
                    "unknown iceberg config key: {key}"
                )));
            }
        }
        Ok(())
    }

    fn entries(&self) -> Vec<ConfigEntry> {
        vec![
            ConfigEntry {
                key: "multi_partition_scan".to_string(),
                value: Some(self.multi_partition_scan.to_string()),
                description: "When false, force T=1 multi-partition off-switch without collapsing session target_partitions",
            },
            ConfigEntry {
                key: "data_file_concurrency".to_string(),
                value: Some(self.data_file_concurrency.to_string()),
                description: "Total data-file concurrency budget L (0 = derive from target_partitions)",
            },
        ]
    }
}

impl ConfigExtension for IcebergScanOptions {
    const PREFIX: &'static str = "iceberg";
}

/// Session-derived knobs for building an Iceberg core `TableScan` and its partition assignment.
/// DataFusion's `TaskContext` supplies them. Row selection stays at the core default, off, because
/// parsing the Parquet page index can outweigh the gain.
///
/// | Symbol | Value |
/// |---|---|
/// | `T` | the output partition budget: `target_partitions`, or `1` with `multi_partition_scan` off |
/// | `L` | the data-file concurrency: `data_file_concurrency`, else `target_partitions` |
/// | `P` | the per-partition concurrency `max(1, ceil(L/N))` |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScanKnobs {
    pub batch_size: Option<usize>,
    /// The budget `L`, for `TableScanBuilder::with_data_file_concurrency_limit`.
    pub data_file_concurrency: Option<usize>,
    /// Session target partition count (raw `T` input before off-switch).
    pub target_partitions: usize,
    /// Dedicated multi-partition off-switch (pin 13). Default true.
    pub multi_partition_scan: bool,
}

impl Default for ScanKnobs {
    fn default() -> Self {
        Self {
            batch_size: None,
            data_file_concurrency: None,
            target_partitions: 1,
            multi_partition_scan: true,
        }
    }
}

/// Floor the session-derived knobs. `ParquetRecordBatchReader` reads `batch_size == 0` as
/// end-of-stream, which looks like a successful empty scan, and `try_buffer_unordered(0)` hangs.
pub(crate) fn clamp_scan_knob(value: usize) -> usize {
    value.max(1)
}

pub(crate) fn scan_knobs_from_context(context: &TaskContext) -> ScanKnobs {
    let config = context.session_config();
    // DataFusion does not normalize 0, and Parquet returns an empty stream for it.
    let batch_size = clamp_scan_knob(config.batch_size());
    let target_partitions = clamp_scan_knob(config.target_partitions());

    let iceberg_opts = config
        .options()
        .extensions
        .get::<IcebergScanOptions>()
        .cloned()
        .unwrap_or_default();
    let multi_partition_scan = iceberg_opts.multi_partition_scan;
    // L falls back to target_partitions when the dedicated surface is zero.
    let data_file_concurrency = if iceberg_opts.data_file_concurrency > 0 {
        clamp_scan_knob(iceberg_opts.data_file_concurrency)
    } else {
        target_partitions
    };

    ScanKnobs {
        batch_size: Some(batch_size),
        data_file_concurrency: Some(data_file_concurrency),
        target_partitions,
        multi_partition_scan,
    }
}

/// Register default [`IcebergScanOptions`] on a session config, if absent.
pub fn ensure_iceberg_scan_options(config: &mut datafusion::prelude::SessionConfig) {
    if config
        .options()
        .extensions
        .get::<IcebergScanOptions>()
        .is_none()
    {
        config
            .options_mut()
            .extensions
            .insert(IcebergScanOptions::default());
    }
}
