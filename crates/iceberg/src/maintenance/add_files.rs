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
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt, stream};

use super::add_files_datafile::{AdoptionContext, adopt_parquet_file, unescape_hive_path_name};
use crate::scan::context::parse_name_mapping;
use crate::spec::{
    DEFAULT_SCHEMA_NAME_MAPPING, DataFile, MetricsConfig, PartitionSpecRef, Transform,
    create_name_mapping,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{Catalog, Error, ErrorKind, Result};

const DUPLICATE_FILE_SAMPLE: usize = 10;

#[allow(missing_docs)]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct AddFilesResult {
    #[allow(missing_docs)]
    pub added_files_count: u64,
    #[allow(missing_docs)]
    pub changed_partition_count: Option<u64>,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddFilesEntry {
    #[allow(missing_docs)]
    pub path: String,
    #[allow(missing_docs)]
    pub partition: Vec<(String, String)>,
}

impl AddFilesEntry {
    #[allow(missing_docs)]
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            partition: Vec::new(),
        }
    }

    #[allow(missing_docs)]
    pub fn with_partition(mut self, partition: Vec<(String, String)>) -> Self {
        self.partition = partition;
        self
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AddFilesSource {
    #[allow(missing_docs)]
    Directory(String),
    #[allow(missing_docs)]
    Files(Vec<AddFilesEntry>),
}

#[allow(missing_docs)]
pub struct AddFiles {
    table: Table,
    source: AddFilesSource,
    partition_filter: HashMap<String, String>,
    check_duplicate_files: bool,
    parallelism: usize,
}

struct SourceFile {
    path: String,
    size: u64,
    partition: Vec<(String, String)>,
}

impl AddFiles {
    #[allow(missing_docs)]
    pub fn new(table: Table, source: AddFilesSource) -> Self {
        Self {
            table,
            source,
            partition_filter: HashMap::new(),
            check_duplicate_files: true,
            parallelism: 1,
        }
    }

    #[allow(missing_docs)]
    pub fn partition_filter(mut self, partition_filter: HashMap<String, String>) -> Self {
        self.partition_filter = partition_filter;
        self
    }

    #[allow(missing_docs)]
    pub fn check_duplicate_files(mut self, check_duplicate_files: bool) -> Self {
        self.check_duplicate_files = check_duplicate_files;
        self
    }

    #[allow(missing_docs)]
    pub fn parallelism(mut self, parallelism: usize) -> Self {
        self.parallelism = parallelism;
        self
    }

    #[allow(missing_docs)]
    pub async fn execute(self, catalog: &dyn Catalog) -> Result<AddFilesResult> {
        if self.parallelism == 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Parallelism should be larger than 0",
            ));
        }

        let table = ensure_name_mapping_present(&self.table, catalog).await?;
        let table_name = table.identifier().to_string();

        let (files, partition_names) = discover(&table, &self.source, self.parallelism).await?;
        if files.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot find any file to import under {}",
                    describe(&self.source)
                ),
            ));
        }
        let spec = find_compatible_spec(&partition_names, &table)?;
        validate_partition_filter(&spec, &self.partition_filter, &table_name)?;

        let files = filter_partitions(files, &self.partition_filter);
        if !spec.is_unpartitioned() && files.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot find any matching partitions in table {table_name}"),
            ));
        }

        if self.check_duplicate_files {
            refuse_duplicates(&table, &files).await?;
        }

        let data_files = self.build_data_files(&table, spec, &files).await?;

        let transaction = Transaction::new(&table);
        let action = transaction
            .merge_append()
            .with_check_duplicate(self.check_duplicate_files)
            .add_data_files(data_files);
        let transaction = action.apply(transaction)?;
        let committed = transaction.commit(catalog).await?;

        Ok(result_of(&committed))
    }

    async fn build_data_files(
        &self,
        table: &Table,
        spec: PartitionSpecRef,
        files: &[SourceFile],
    ) -> Result<Vec<DataFile>> {
        let metadata = table.metadata();
        let context = Arc::new(AdoptionContext {
            schema: metadata.current_schema().clone(),
            metrics_config: MetricsConfig::for_table(metadata)?,
            name_mapping: parse_name_mapping(metadata)?,
            partition_type: spec.partition_type(metadata.current_schema())?,
            spec,
        });
        let file_io = table.file_io().clone();

        stream::iter(files.iter())
            .map(|file| {
                let context = Arc::clone(&context);
                let file_io = file_io.clone();
                async move {
                    adopt_parquet_file(&file_io, &context, &file.path, file.size, &file.partition)
                        .await
                }
            })
            .buffered(self.parallelism)
            .try_collect()
            .await
    }
}

fn describe(source: &AddFilesSource) -> String {
    match source {
        AddFilesSource::Directory(root) => root.clone(),
        AddFilesSource::Files(entries) => format!("the given list of {} files", entries.len()),
    }
}

fn result_of(table: &Table) -> AddFilesResult {
    let Some(snapshot) = table.metadata().current_snapshot() else {
        return AddFilesResult::default();
    };
    let summary = &snapshot.summary().additional_properties;
    AddFilesResult {
        added_files_count: summary
            .get("added-data-files")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0),
        changed_partition_count: summary
            .get("changed-partition-count")
            .and_then(|value| value.parse::<u64>().ok()),
    }
}

async fn ensure_name_mapping_present(table: &Table, catalog: &dyn Catalog) -> Result<Table> {
    if table
        .metadata()
        .properties()
        .contains_key(DEFAULT_SCHEMA_NAME_MAPPING)
    {
        return Ok(table.clone());
    }
    let mapping = create_name_mapping(table.metadata().current_schema())?;
    let json = serde_json::to_string(&mapping).map_err(|err| {
        Error::new(
            ErrorKind::Unexpected,
            "Cannot serialize the default name mapping",
        )
        .with_source(err)
    })?;
    let transaction = Transaction::new(table);
    let action = transaction
        .update_table_properties()
        .set(DEFAULT_SCHEMA_NAME_MAPPING.to_string(), json);
    let transaction = action.apply(transaction)?;
    transaction.commit(catalog).await
}

async fn discover(
    table: &Table,
    source: &AddFilesSource,
    parallelism: usize,
) -> Result<(Vec<SourceFile>, Vec<String>)> {
    match source {
        AddFilesSource::Directory(root) => discover_directory(table, root).await,
        AddFilesSource::Files(entries) => {
            let file_io = table.file_io().clone();
            let files: Vec<SourceFile> = stream::iter(entries.iter())
                .map(|entry| {
                    let file_io = file_io.clone();
                    async move {
                        let size = file_io.new_input(&entry.path)?.metadata().await?.size;
                        Ok::<SourceFile, Error>(SourceFile {
                            path: entry.path.clone(),
                            size,
                            partition: entry.partition.clone(),
                        })
                    }
                })
                .buffered(parallelism)
                .try_collect()
                .await?;
            let names = partition_names_of(&files)?;
            Ok((files, names))
        }
    }
}

async fn discover_directory(table: &Table, root: &str) -> Result<(Vec<SourceFile>, Vec<String>)> {
    let prefix = format!("{}/", root.trim_end_matches('/'));
    let listed = table.file_io().list(&prefix).await.map_err(|err| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Cannot list the source location {root} to import"),
        )
        .with_source(err)
    })?;

    let mut files = Vec::new();
    for info in listed {
        let Some(relative) = info.location.strip_prefix(&prefix) else {
            continue;
        };
        let segments: Vec<&str> = relative.split('/').collect();
        if segments.iter().any(|segment| is_hidden(segment)) {
            continue;
        }
        let mut partition = Vec::with_capacity(segments.len().saturating_sub(1));
        for segment in &segments[..segments.len().saturating_sub(1)] {
            let Some((name, value)) = segment.split_once('=') else {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot infer the partition columns of {root}: the directory '{segment}' is not a 'name=value' partition directory"
                    ),
                ));
            };
            partition.push((
                unescape_hive_path_name(name),
                unescape_hive_path_name(value),
            ));
        }
        files.push(SourceFile {
            path: info.location.clone(),
            size: info.size,
            partition,
        });
    }
    files.sort_by(|left, right| left.path.cmp(&right.path));

    let names = partition_names_of(&files)?;
    Ok((files, names))
}

fn is_hidden(segment: &str) -> bool {
    segment.starts_with('_') || segment.starts_with('.')
}

fn partition_names_of(files: &[SourceFile]) -> Result<Vec<String>> {
    let mut names: Option<Vec<String>> = None;
    for file in files {
        let own: Vec<String> = file
            .partition
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        match &names {
            None => names = Some(own),
            Some(known) if known == &own => {}
            Some(known) => {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Conflicting directory structures in the source to import: {known:?} and {own:?}"
                    ),
                ));
            }
        }
    }
    Ok(names.unwrap_or_default())
}

pub(super) fn find_compatible_spec(
    partition_names: &[String],
    table: &Table,
) -> Result<PartitionSpecRef> {
    let wanted: Vec<String> = partition_names
        .iter()
        .map(|name| name.to_lowercase())
        .collect();
    let mut specs: Vec<&PartitionSpecRef> = table.metadata().partition_specs_iter().collect();
    specs.sort_by_key(|spec| spec.spec_id());
    for spec in specs {
        if !spec
            .fields()
            .iter()
            .all(|field| field.transform == Transform::Identity)
        {
            continue;
        }
        let names: Vec<String> = spec
            .fields()
            .iter()
            .map(|field| field.name.to_lowercase())
            .collect();
        if names == wanted {
            return Ok(spec.clone());
        }
    }
    Err(Error::new(
        ErrorKind::DataInvalid,
        format!(
            "Cannot find a partition spec in Iceberg table {} that matches the partition columns ([{}]) in input table",
            table.identifier(),
            partition_names.join(", ")
        ),
    ))
}

pub(super) fn validate_partition_filter(
    spec: &PartitionSpecRef,
    partition_filter: &HashMap<String, String>,
    table_name: &str,
) -> Result<()> {
    if spec.is_unpartitioned() {
        if partition_filter.is_empty() {
            return Ok(());
        }
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Cannot use partition filter with an unpartitioned table {table_name}"),
        ));
    }
    if partition_filter.is_empty() {
        return Ok(());
    }
    if spec.fields().len() < partition_filter.len() {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Cannot add data files to target table {table_name} because that table is partitioned, but the number of columns in the provided partition filter ({}) is greater than the number of partitioned columns in table ({})",
                partition_filter.len(),
                spec.fields().len()
            ),
        ));
    }
    let names: HashSet<&str> = spec
        .fields()
        .iter()
        .map(|field| field.name.as_str())
        .collect();
    let mut unknown: Vec<&str> = partition_filter
        .keys()
        .map(String::as_str)
        .filter(|key| !names.contains(key))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    unknown.sort_unstable();
    let mut valid: Vec<&str> = names.into_iter().collect();
    valid.sort_unstable();
    Err(Error::new(
        ErrorKind::DataInvalid,
        format!(
            "Cannot add files to target table {table_name}. {table_name} is partitioned but the specified partition filter refers to columns that are not partitioned: {} . Valid partition columns: [{}]",
            unknown.join(", "),
            valid.join(",")
        ),
    ))
}

fn filter_partitions(
    files: Vec<SourceFile>,
    partition_filter: &HashMap<String, String>,
) -> Vec<SourceFile> {
    if partition_filter.is_empty() {
        return files;
    }
    files
        .into_iter()
        .filter(|file| {
            partition_filter.iter().all(|(key, value)| {
                file.partition
                    .iter()
                    .any(|(name, own)| name == key && own == value)
            })
        })
        .collect()
}

async fn refuse_duplicates(table: &Table, files: &[SourceFile]) -> Result<()> {
    let live = live_entry_paths(table).await?;
    let mut duplicates: Vec<&str> = files
        .iter()
        .map(|file| file.path.as_str())
        .filter(|path| live.contains(*path))
        .collect();
    if duplicates.is_empty() {
        return Ok(());
    }
    duplicates.sort_unstable();
    duplicates.truncate(DUPLICATE_FILE_SAMPLE);
    Err(Error::new(
        ErrorKind::DataInvalid,
        format!(
            "Cannot complete import because data files to be imported already exist within the target table: {}.  This is disabled by default as Iceberg is not designed for multiple references to the same file within the same table.  If you are sure, you may set 'check_duplicate_files' to false to force the import.",
            duplicates.join(",")
        ),
    ))
}

async fn live_entry_paths(table: &Table) -> Result<HashSet<String>> {
    let metadata = table.metadata();
    let Some(snapshot) = metadata.current_snapshot() else {
        return Ok(HashSet::new());
    };
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), metadata)
        .await?;
    let mut paths = HashSet::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if entry.is_alive() {
                paths.insert(entry.data_file().file_path().to_string());
            }
        }
    }
    Ok(paths)
}
