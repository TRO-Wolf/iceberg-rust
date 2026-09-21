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

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::error::{Error, ErrorKind, Result};
use crate::spec::{
    DataFile, Literal, ManifestContentType, ManifestEntry, ManifestFile, PrimitiveLiteral,
    TableMetadata, apply_manifest_list_context,
};
use crate::transaction::rewrite_manifests::{RewriteManifestsAction, RewriteOutcome};
use crate::transaction::snapshot::SnapshotProducer;

type SortKey = Vec<Option<Literal>>;

type SortEntry = (SortKey, ManifestEntry);

impl RewriteManifestsAction {
    #[allow(missing_docs)]
    pub fn sort_by_columns(mut self, columns: Vec<String>) -> Result<Self> {
        if columns.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "sort_by must not be empty when provided",
            ));
        }
        self.sort_by_columns = Some(columns);
        self.cluster_by = None;
        self.cluster_by_columns = None;
        Ok(self)
    }

    pub(super) async fn perform_selected_rewrite(
        &self,
        snapshot_producer: &mut SnapshotProducer<'_>,
        current_manifests: &[ManifestFile],
        target_size_bytes: u64,
    ) -> Result<RewriteOutcome> {
        let cluster_set = self.cluster_by.is_some() || self.cluster_by_columns.is_some();
        if let Some(columns) = self.sort_by_columns.as_ref()
            && !cluster_set
        {
            return self
                .perform_sort_rewrite(
                    snapshot_producer,
                    current_manifests,
                    target_size_bytes,
                    columns,
                )
                .await;
        }
        self.perform_rewrite(snapshot_producer, current_manifests, target_size_bytes)
            .await
    }

    async fn perform_sort_rewrite(
        &self,
        snapshot_producer: &mut SnapshotProducer<'_>,
        current_manifests: &[ManifestFile],
        target_size_bytes: u64,
        columns: &[String],
    ) -> Result<RewriteOutcome> {
        if target_size_bytes == 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "commit.manifest.target-size-bytes must be positive, got 0",
            ));
        }
        let deleted_paths: HashSet<&str> = self
            .deleted_manifests
            .iter()
            .map(|manifest| manifest.manifest_path.as_str())
            .collect();

        let mut kept_manifests: Vec<ManifestFile> = Vec::new();
        let mut rewritten_manifests: Vec<ManifestFile> = Vec::new();
        let mut sort_runs = SortRuns::new(target_size_bytes);
        let mut entries_processed: u64 = 0;
        let table = snapshot_producer.table;

        for manifest_file in current_manifests {
            if deleted_paths.contains(manifest_file.manifest_path.as_str()) {
                continue;
            }

            let content_matches =
                manifest_file.content == ManifestContentType::Data || self.rewrite_delete_manifests;
            let should_rewrite = content_matches
                && self
                    .rewrite_if
                    .as_ref()
                    .map(|predicate| predicate(manifest_file))
                    .unwrap_or(true);

            if !should_rewrite {
                kept_manifests.push(manifest_file.clone());
                continue;
            }

            let (_, mut entries) = manifest_file
                .load_manifest_parts_with_schema_fallback(snapshot_producer.table.file_io(), None)
                .await?;
            apply_manifest_list_context(&mut entries, manifest_file)?;
            sort_runs.add_manifest(
                manifest_file.partition_spec_id,
                manifest_file.content,
                manifest_file.manifest_length.max(0) as u64,
            );

            for entry in entries {
                if !entry.is_alive() {
                    continue;
                }
                let key = sort_key_for_columns(columns, entry.data_file(), table.metadata())?;
                sort_runs.push(
                    manifest_file.partition_spec_id,
                    manifest_file.content,
                    key,
                    entry,
                );
                entries_processed += 1;
            }

            rewritten_manifests.push(manifest_file.clone());
        }

        let new_manifests = sort_runs.finish(snapshot_producer).await?;
        let new_manifest_count = new_manifests.len();
        let kept_count = kept_manifests.len();

        Ok(RewriteOutcome {
            new_manifests,
            new_manifest_count,
            rewritten_manifests,
            kept_manifests,
            kept_count,
            entries_processed,
        })
    }
}

struct SortRuns {
    target_size_bytes: u64,
    runs: HashMap<(i32, ManifestContentType), SortRun>,
}

struct SortRun {
    bytes: u64,
    entries: Vec<SortEntry>,
}

impl SortRuns {
    fn new(target_size_bytes: u64) -> Self {
        Self {
            target_size_bytes,
            runs: HashMap::new(),
        }
    }

    fn run(&mut self, partition_spec_id: i32, content: ManifestContentType) -> &mut SortRun {
        self.runs
            .entry((partition_spec_id, content))
            .or_insert_with(|| SortRun {
                bytes: 0,
                entries: Vec::new(),
            })
    }

    fn add_manifest(&mut self, partition_spec_id: i32, content: ManifestContentType, bytes: u64) {
        let run = self.run(partition_spec_id, content);
        run.bytes = run.bytes.saturating_add(bytes);
    }

    fn push(
        &mut self,
        partition_spec_id: i32,
        content: ManifestContentType,
        key: SortKey,
        entry: ManifestEntry,
    ) {
        self.run(partition_spec_id, content)
            .entries
            .push((key, entry));
    }

    async fn finish(
        self,
        snapshot_producer: &mut SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        let mut runs: Vec<((i32, ManifestContentType), SortRun)> = self.runs.into_iter().collect();
        runs.sort_by_key(|(key, _)| *key);
        let mut finished = Vec::new();
        for ((partition_spec_id, content), run) in runs {
            if run.entries.is_empty() {
                continue;
            }
            for group in run.groups(self.target_size_bytes) {
                let mut writer =
                    snapshot_producer.new_cluster_manifest_writer(partition_spec_id, content)?;
                for (_, entry) in group {
                    writer.add_existing_entry(entry)?;
                }
                finished.push(writer.write_manifest_file().await?);
            }
        }
        Ok(finished)
    }
}

impl SortRun {
    fn groups(mut self, target_size_bytes: u64) -> Vec<Vec<SortEntry>> {
        self.entries.sort_by(|left, right| {
            cmp_sort_key(&left.0, &right.0)
                .then_with(|| left.1.file_path().cmp(right.1.file_path()))
        });
        let count = self.entries.len();
        let count_u64 = count as u64;
        let target_manifests = self
            .bytes
            .div_ceil(target_size_bytes)
            .clamp(1, count_u64.max(1));
        let mut cuts: Vec<usize> = Vec::new();
        for bound in 1..target_manifests {
            let probe =
                (u128::from(bound) * u128::from(count_u64) / u128::from(target_manifests)) as usize;
            let value = self.entries[probe].0.clone();
            let mut cut = probe;
            while cut + 1 < count && self.entries[cut + 1].0 == value {
                cut += 1;
            }
            if cut + 1 < count && cuts.last().is_none_or(|&last| last < cut) {
                cuts.push(cut);
            }
        }
        let mut groups: Vec<Vec<SortEntry>> = Vec::new();
        let mut current: Vec<SortEntry> = Vec::new();
        let mut cut_at = cuts.into_iter().peekable();
        for (index, entry) in self.entries.into_iter().enumerate() {
            current.push(entry);
            if cut_at.peek() == Some(&index) {
                cut_at.next();
                groups.push(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            groups.push(current);
        }
        groups
    }
}

fn sort_key_for_columns(
    columns: &[String],
    file: &DataFile,
    metadata: &TableMetadata,
) -> Result<SortKey> {
    let spec_id = file.partition_spec_id();
    let spec = metadata.partition_spec_by_id(spec_id).ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Cannot sort by columns: partition spec {spec_id} not found"),
        )
    })?;
    let fields = spec.fields();
    let partition = file.partition();
    let mut selected = SortKey::with_capacity(columns.len());
    for column in columns {
        let index = fields
            .iter()
            .position(|field| field.name == *column)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot sort by column '{column}': not a partition field of spec {}",
                        spec.spec_id()
                    ),
                )
            })?;
        selected.push(partition.fields().get(index).cloned().flatten());
    }
    Ok(selected)
}

fn cmp_sort_key(left: &SortKey, right: &SortKey) -> Ordering {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| cmp_sort_value(left, right))
        .find(|ordering| *ordering != Ordering::Equal)
        .unwrap_or(Ordering::Equal)
}

fn cmp_sort_value(left: &Option<Literal>, right: &Option<Literal>) -> Ordering {
    match (left, right) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(left), Some(right)) => cmp_literal(left, right),
    }
}

fn cmp_literal(left: &Literal, right: &Literal) -> Ordering {
    match (left, right) {
        (Literal::Primitive(left), Literal::Primitive(right)) => cmp_primitive(left, right),
        _ => format!("{left:?}").cmp(&format!("{right:?}")),
    }
}

fn cmp_primitive(left: &PrimitiveLiteral, right: &PrimitiveLiteral) -> Ordering {
    match (left, right) {
        (PrimitiveLiteral::Boolean(left), PrimitiveLiteral::Boolean(right)) => left.cmp(right),
        (PrimitiveLiteral::Int(left), PrimitiveLiteral::Int(right)) => left.cmp(right),
        (PrimitiveLiteral::Long(left), PrimitiveLiteral::Long(right)) => left.cmp(right),
        (PrimitiveLiteral::Float(left), PrimitiveLiteral::Float(right)) => {
            left.0.total_cmp(&right.0)
        }
        (PrimitiveLiteral::Double(left), PrimitiveLiteral::Double(right)) => {
            left.0.total_cmp(&right.0)
        }
        (PrimitiveLiteral::String(left), PrimitiveLiteral::String(right)) => left.cmp(right),
        (PrimitiveLiteral::Binary(left), PrimitiveLiteral::Binary(right)) => left.cmp(right),
        (PrimitiveLiteral::Int128(left), PrimitiveLiteral::Int128(right)) => left.cmp(right),
        (PrimitiveLiteral::UInt128(left), PrimitiveLiteral::UInt128(right)) => left.cmp(right),
        (PrimitiveLiteral::BelowMin, PrimitiveLiteral::BelowMin) => Ordering::Equal,
        (PrimitiveLiteral::BelowMin, _) => Ordering::Less,
        (_, PrimitiveLiteral::BelowMin) => Ordering::Greater,
        (PrimitiveLiteral::AboveMax, PrimitiveLiteral::AboveMax) => Ordering::Equal,
        (PrimitiveLiteral::AboveMax, _) => Ordering::Greater,
        (_, PrimitiveLiteral::AboveMax) => Ordering::Less,
        _ => format!("{left:?}").cmp(&format!("{right:?}")),
    }
}
