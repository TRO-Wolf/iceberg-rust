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

use std::collections::BTreeSet;

use crate::error::{Error, ErrorKind, Result};
use crate::maintenance::rewrite_data_files::RewriteDataFiles;
use crate::maintenance::rewrite_data_files_plan::format_java_double;
use crate::maintenance::rewrite_data_files_zorder::column_kind;
use crate::spec::{Schema, SortOrder, Transform};
use crate::table::Table;

pub(super) const SHUFFLE_PARTITIONS_PER_FILE_DEFAULT: usize = 1;
pub(super) const COMPRESSION_FACTOR_DEFAULT: f64 = 1.0;
pub(super) const VAR_LENGTH_CONTRIBUTION_DEFAULT: u32 = 8;
pub(super) const MAX_OUTPUT_SIZE_DEFAULT: u32 = i32::MAX as u32;
pub(super) const SORT_MEMORY_BUDGET_BYTES_DEFAULT: u64 = 128 * 1024 * 1024;
const Z_COLUMN: &str = "ICEZVALUE";

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct ZOrderSpec {
    columns: Vec<String>,
    var_length_contribution: u32,
    max_output_size: u32,
}

impl ZOrderSpec {
    #[allow(missing_docs)]
    pub fn new(columns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        ZOrderSpec {
            columns: columns.into_iter().map(Into::into).collect(),
            var_length_contribution: VAR_LENGTH_CONTRIBUTION_DEFAULT,
            max_output_size: MAX_OUTPUT_SIZE_DEFAULT,
        }
    }

    #[allow(missing_docs)]
    pub fn var_length_contribution(mut self, bytes: u32) -> Self {
        self.var_length_contribution = bytes;
        self
    }

    #[allow(missing_docs)]
    pub fn max_output_size(mut self, bytes: u32) -> Self {
        self.max_output_size = bytes;
        self
    }

    #[allow(missing_docs)]
    pub fn columns(&self) -> &[String] {
        &self.columns
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
#[allow(missing_docs)]
pub enum RewriteStrategy {
    #[default]
    BinPack,
    SortByTableOrder,
    Sort(SortOrder),
    ZOrder(ZOrderSpec),
}

impl RewriteStrategy {
    #[allow(missing_docs)]
    pub fn description(&self) -> &'static str {
        match self {
            RewriteStrategy::BinPack => "BIN-PACK",
            RewriteStrategy::SortByTableOrder | RewriteStrategy::Sort(_) => "SORT",
            RewriteStrategy::ZOrder(_) => "Z-ORDER",
        }
    }

    #[allow(missing_docs)]
    pub fn valid_option_names(&self) -> &'static [&'static str] {
        match self {
            RewriteStrategy::BinPack => &[],
            RewriteStrategy::SortByTableOrder | RewriteStrategy::Sort(_) => {
                &["shuffle-partitions-per-file", "compression-factor"]
            }
            RewriteStrategy::ZOrder(_) => &[
                "shuffle-partitions-per-file",
                "compression-factor",
                "var-length-contribution",
                "max-output-size",
            ],
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct StrategyConfig {
    pub(super) strategy: RewriteStrategy,
    pub(super) shuffle_partitions_per_file: usize,
    pub(super) compression_factor: f64,
    pub(super) sort_memory_budget_bytes: u64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        StrategyConfig {
            strategy: RewriteStrategy::BinPack,
            shuffle_partitions_per_file: SHUFFLE_PARTITIONS_PER_FILE_DEFAULT,
            compression_factor: COMPRESSION_FACTOR_DEFAULT,
            sort_memory_budget_bytes: SORT_MEMORY_BUDGET_BYTES_DEFAULT,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) enum ResolvedStrategy {
    BinPack,
    Sort {
        order: SortOrder,
        stamp: i32,
    },
    ZOrder {
        columns: Vec<String>,
        var_length_contribution: usize,
        max_output_size: usize,
    },
}

impl RewriteDataFiles {
    #[allow(missing_docs)]
    pub fn strategy(mut self, strategy: RewriteStrategy) -> Self {
        self.strategy_config.strategy = strategy;
        self
    }

    #[allow(missing_docs)]
    pub fn shuffle_partitions_per_file(mut self, shuffle_partitions_per_file: usize) -> Self {
        self.strategy_config.shuffle_partitions_per_file = shuffle_partitions_per_file;
        self
    }

    #[allow(missing_docs)]
    pub fn compression_factor(mut self, compression_factor: f64) -> Self {
        self.strategy_config.compression_factor = compression_factor;
        self
    }

    #[allow(missing_docs)]
    pub fn sort_memory_budget_bytes(mut self, sort_memory_budget_bytes: u64) -> Self {
        self.strategy_config.sort_memory_budget_bytes = sort_memory_budget_bytes;
        self
    }
}

pub(super) fn resolve_strategy(table: &Table, config: &StrategyConfig) -> Result<ResolvedStrategy> {
    if config.shuffle_partitions_per_file == 0 {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            "'shuffle-partitions-per-file' is set to 0 but must be > 0",
        ));
    }
    if config.compression_factor.is_nan() || config.compression_factor <= 0.0 {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "'compression-factor' is set to {} but must be > 0",
                format_java_double(config.compression_factor)
            ),
        ));
    }
    if config.sort_memory_budget_bytes == 0 {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            "'sort-memory-budget-bytes' is set to 0 but must be > 0",
        ));
    }

    let mut layout_options: BTreeSet<&str> = BTreeSet::new();
    if config.shuffle_partitions_per_file != SHUFFLE_PARTITIONS_PER_FILE_DEFAULT {
        layout_options.insert("shuffle-partitions-per-file");
    }
    if config.compression_factor != COMPRESSION_FACTOR_DEFAULT {
        layout_options.insert("compression-factor");
    }
    let accepted = config.strategy.valid_option_names();
    let refused: Vec<&str> = layout_options
        .into_iter()
        .filter(|option| !accepted.contains(option))
        .collect();
    if !refused.is_empty() {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Cannot use options [{}], they are not supported by the action or the rewriter {}",
                refused.join(", "),
                config.strategy.description()
            ),
        ));
    }

    let schema = table.metadata().current_schema();
    match &config.strategy {
        RewriteStrategy::BinPack => Ok(ResolvedStrategy::BinPack),
        RewriteStrategy::SortByTableOrder => {
            let order = table.metadata().default_sort_order().as_ref().clone();
            if order.is_unsorted() {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot sort data without a valid sort order, table '{}' is unsorted and no sort order is provided",
                        table.identifier()
                    ),
                ));
            }
            check_order_columns(schema, &order)?;
            let stamp = sort_order_stamp(table, &order);
            Ok(ResolvedStrategy::Sort { order, stamp })
        }
        RewriteStrategy::Sort(order) => {
            if order.is_unsorted() {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Cannot sort data without a valid sort order, the provided sort order is null or empty",
                ));
            }
            check_order_columns(schema, order)?;
            let stamp = sort_order_stamp(table, order);
            Ok(ResolvedStrategy::Sort {
                order: order.clone(),
                stamp,
            })
        }
        RewriteStrategy::ZOrder(spec) => {
            let columns = valid_z_order_columns(table, spec)?;
            if spec.var_length_contribution == 0 {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Cannot use less than 1 byte for variable length types with ZOrder, 'var-length-contribution' was set to 0",
                ));
            }
            if spec.max_output_size == 0 {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Cannot have the interleaved ZOrder value use less than 1 byte, 'max-output-size' was set to 0",
                ));
            }
            Ok(ResolvedStrategy::ZOrder {
                columns,
                var_length_contribution: spec.var_length_contribution as usize,
                max_output_size: spec.max_output_size as usize,
            })
        }
    }
}

fn check_order_columns(schema: &Schema, order: &SortOrder) -> Result<()> {
    for field in &order.fields {
        let source = schema.field_by_id(field.source_id).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot find source column for sort field: {field} in {}",
                    java_struct_display(schema)
                ),
            )
        })?;
        if field.transform != Transform::Identity
            && field
                .transform
                .result_type(source.field_type.as_ref())
                .is_err()
        {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot sort by transform {} of column '{}' of type {}",
                    field.transform, source.name, source.field_type
                ),
            ));
        }
    }
    Ok(())
}

pub(super) fn sort_order_stamp(table: &Table, order: &SortOrder) -> i32 {
    let mut ids: Vec<&crate::spec::SortOrderRef> = table.metadata().sort_orders_iter().collect();
    ids.sort_by_key(|candidate| candidate.order_id);
    for candidate in ids {
        if candidate.fields == order.fields {
            return i32::try_from(candidate.order_id).unwrap_or(0);
        }
    }
    0
}

fn valid_z_order_columns(table: &Table, spec: &ZOrderSpec) -> Result<Vec<String>> {
    if spec.columns.is_empty() {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            "Cannot ZOrder when no columns are specified",
        ));
    }
    let schema = table.metadata().current_schema();
    if schema.field_by_name_case_insensitive(Z_COLUMN).is_some() {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Cannot zorder because the table has a column named '{Z_COLUMN}', which conflicts with Iceberg's internal Z-order column name"
            ),
        ));
    }
    let identity_sources: BTreeSet<i32> = table
        .metadata()
        .default_partition_spec()
        .fields()
        .iter()
        .filter(|field| field.transform == Transform::Identity)
        .map(|field| field.source_id)
        .collect();

    let mut kept = Vec::with_capacity(spec.columns.len());
    for name in &spec.columns {
        let field = schema.field_by_name(name).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot find column '{name}' in table schema (case sensitive = false): {}",
                    java_struct_display(schema)
                ),
            )
        })?;
        column_kind(
            name,
            field.field_type.as_ref(),
            spec.var_length_contribution.max(1) as usize,
        )?;
        if identity_sources.contains(&field.id) {
            continue;
        }
        kept.push(field.name.clone());
    }
    if kept.is_empty() {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            "Cannot ZOrder, all columns provided were identity partition columns and cannot be used",
        ));
    }
    Ok(kept)
}

fn java_struct_display(schema: &Schema) -> String {
    let fields: Vec<String> = schema
        .as_struct()
        .fields()
        .iter()
        .map(|field| {
            let optionality = if field.required {
                "required"
            } else {
                "optional"
            };
            format!(
                "{}: {}: {optionality} {}",
                field.id, field.name, field.field_type
            )
        })
        .collect();
    format!("struct<{}>", fields.join(", "))
}
