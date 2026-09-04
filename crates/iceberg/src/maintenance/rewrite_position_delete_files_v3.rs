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

use super::*;

impl RewritePositionDeleteFiles {
    /// Convert live filter-matching parquet position deletes into one Puffin DV per data file.
    ///
    /// # Errors
    ///
    /// Unreadable format, a live excluded delete a written vector would shadow, two live DVs
    /// for one data file, or a DV seq below its data file. `Ok` with zeros means nothing to convert.
    pub(super) async fn rewrite_to_deletion_vectors(
        &self,
        catalog: &dyn Catalog,
        snapshot: &Snapshot,
        partition_filter: &mut PartitionFilter,
        starting_snapshot_id: i64,
        config: &ResolvedConfig,
    ) -> Result<RewritePositionDeleteFilesResult> {
        let mut inventory = self
            .collect_v3_delete_inventory(snapshot, partition_filter)
            .await?;
        if inventory.legacy_position_deletes.is_empty() {
            return Ok(RewritePositionDeleteFilesResult::default());
        }

        let mut groups: HashMap<GroupKey, Vec<LiveDeleteEntry>> = HashMap::new();
        for entry in std::mem::take(&mut inventory.legacy_position_deletes) {
            groups
                .entry((
                    entry.data_file.partition_spec_id,
                    entry.data_file.partition().clone(),
                ))
                .or_default()
                .push(entry);
        }
        let (bins, declined) = plan_bins(groups, config);
        inventory.gate_declined_position_deletes.extend(declined);
        inventory.legacy_position_deletes = bins.into_iter().flat_map(|(_, bin)| bin).collect();
        if inventory.legacy_position_deletes.is_empty() {
            return Ok(RewritePositionDeleteFilesResult::default());
        }

        let (plans, superseded_puffin_paths) = self.plan_deletion_vectors(&inventory).await?;
        refuse_shadowed_deletes(&inventory, &plans)?;
        let new_deletion_vectors = self.write_deletion_vectors(&plans, &inventory).await?;

        // Superseded blobs plus the siblings this path rewrites alongside them.
        let mut rewritten_files: Vec<DataFile> = inventory
            .legacy_position_deletes
            .iter()
            .map(|entry| entry.data_file.clone())
            .collect();
        rewritten_files.extend(
            inventory
                .deletion_vectors
                .values()
                .filter(|entry| superseded_puffin_paths.contains(entry.data_file.file_path()))
                .map(|entry| entry.data_file.clone()),
        );

        let result = summarize_v3_rewrite(&rewritten_files, &new_deletion_vectors)?;

        let transaction = Transaction::new(&self.table);
        let mut action = transaction
            .rewrite_files(Vec::new(), Vec::new())
            .delete_delete_files(rewritten_files);
        for delete_file in new_deletion_vectors {
            let sequence_number = deletion_vector_sequence_number(&delete_file, &plans)?;
            action = action.add_delete_file_with_sequence_number(delete_file, sequence_number);
        }
        let action = action.validate_from_snapshot(starting_snapshot_id);
        action.apply(transaction)?.commit(catalog).await?;

        Ok(result)
    }

    /// One walk: live data files, admitted parquet position deletes, and Puffin DVs by data file.
    ///
    /// # Errors
    ///
    /// A position delete that is neither Parquet nor Puffin, a DV with no referenced data file,
    /// or two live DVs for one data file.
    async fn collect_v3_delete_inventory(
        &self,
        snapshot: &Snapshot,
        partition_filter: &mut PartitionFilter,
    ) -> Result<V3DeleteInventory> {
        let metadata = self.table.metadata();
        let manifest_list = snapshot
            .load_manifest_list(self.table.file_io(), metadata)
            .await?;

        let mut inventory = V3DeleteInventory::default();
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file.load_manifest(self.table.file_io()).await?;
            for entry in manifest.entries() {
                if !entry.is_alive() {
                    continue;
                }
                let data_file = entry.data_file();
                let sequence_number = entry.sequence_number().unwrap_or(0);
                match data_file.content_type() {
                    DataContentType::Data => {
                        inventory.data_files.insert(
                            data_file.file_path().to_string(),
                            LiveDataFile {
                                partition_spec_id: data_file.partition_spec_id,
                                partition: data_file.partition().clone(),
                                sequence_number,
                            },
                        );
                    }
                    DataContentType::PositionDeletes => {
                        inventory.admit_position_delete(
                            metadata,
                            partition_filter,
                            data_file,
                            sequence_number,
                        )?;
                    }
                    DataContentType::EqualityDeletes => {}
                }
            }
        }

        Ok(inventory)
    }

    /// Plan one merged DV per data file. Rewrite every sibling blob in a superseded Puffin.
    /// Merging a non-superset DV would make a shadowed position effective and delete live rows.
    async fn plan_deletion_vectors(
        &self,
        inventory: &V3DeleteInventory,
    ) -> Result<(HashMap<String, DeletionVectorPlan>, HashSet<String>)> {
        let mut plans: HashMap<String, DeletionVectorPlan> = HashMap::new();
        for entry in &inventory.legacy_position_deletes {
            let mut pairs: Vec<(String, i64)> = Vec::new();
            self.read_position_pairs(&self.table, &entry.data_file, &mut pairs)
                .await?;
            for (data_file_path, position) in pairs {
                let position = u64::try_from(position).map_err(|error| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Position delete '{}' has a negative position {position} for data file \
                             '{data_file_path}'",
                            entry.data_file.file_path()
                        ),
                    )
                    .with_source(error)
                })?;
                // Drop positions for data files the snapshot no longer holds. Refusing dead-ends the table.
                if !inventory.data_files.contains_key(&data_file_path) {
                    continue;
                }
                let plan = plans.entry(data_file_path).or_default();
                plan.positions.push(position);
                plan.sequence_number = plan.sequence_number.max(entry.sequence_number);
            }
        }

        let mut superseded_puffin_paths: HashSet<String> = plans
            .keys()
            .filter_map(|path| inventory.deletion_vectors.get(path))
            .map(|entry| entry.data_file.file_path().to_string())
            .collect();
        for (data_file_path, entry) in &inventory.deletion_vectors {
            if superseded_puffin_paths.contains(entry.data_file.file_path())
                && inventory.data_files.contains_key(data_file_path)
            {
                plans.entry(data_file_path.clone()).or_default();
            }
        }

        for (data_file_path, plan) in &mut plans {
            let data_file = inventory.live_data_file(data_file_path)?;
            if let Some(entry) = inventory.deletion_vectors.get(data_file_path) {
                let previous = load_delete_vector(self.table.file_io(), &entry.data_file).await?;
                // This DV already suppresses the legacy delete. Folding in a position it does not
                // hold would silently delete a live row. Refusal writes nothing.
                if let Some(unshadowed) = plan
                    .positions
                    .iter()
                    .find(|position| !previous.contains(**position))
                {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Data file '{data_file_path}' holds a deletion vector that does not \
                             cover position {unshadowed} of a legacy position delete it already \
                             suppresses. Converting would DELETE rows the table returns today, so \
                             this run refuses. THIS ACTION CANNOT CLEAR THAT STATE at any filter \
                             width. RewriteDataFiles with remove_dangling_deletes(true) clears it \
                             when the planner admits the file. The default delete-ratio-threshold \
                             0.3 admits a file whose file-scoped deletes cover at least 30% of its \
                             rows."
                        ),
                    ));
                }
                plan.positions.extend(previous.iter());
                plan.sequence_number = plan.sequence_number.max(entry.sequence_number);
                superseded_puffin_paths.insert(entry.data_file.file_path().to_string());
            }
            if plan.sequence_number < data_file.sequence_number {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Deletion vector for data file '{data_file_path}' would carry data sequence \
                         number {} but the data file is at {}",
                        plan.sequence_number, data_file.sequence_number
                    ),
                ));
            }
        }

        Ok((plans, superseded_puffin_paths))
    }

    /// Write every planned DV into one Puffin. Each `delete` carries that data file's PartitionKey.
    /// Do not use `with_partition_spec`: one Puffin spans every partition this arm touches.
    async fn write_deletion_vectors(
        &self,
        plans: &HashMap<String, DeletionVectorPlan>,
        inventory: &V3DeleteInventory,
    ) -> Result<Vec<DataFile>> {
        let metadata = self.table.metadata();
        let schema = metadata.current_schema().clone();
        let location_generator = DefaultLocationGenerator::new(metadata.clone())?;
        let file_name_generator = DefaultFileNameGenerator::new(
            "rewritten-dv".to_string(),
            Some(uuid::Uuid::now_v7().to_string()),
            DataFileFormat::Puffin,
        );
        let location =
            location_generator.generate_location(None, &file_name_generator.generate_file_name());
        let mut writer =
            DVFileWriter::new(self.table.file_io().new_output(location)?).unpartitioned();

        for (data_file_path, plan) in plans {
            let data_file = inventory.live_data_file(data_file_path)?;
            let spec = metadata
                .partition_spec_by_id(data_file.partition_spec_id)
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Data file '{data_file_path}' references unknown partition spec {}",
                            data_file.partition_spec_id
                        ),
                    )
                })?
                .as_ref()
                .clone();
            let partition_key =
                PartitionKey::new(spec, schema.clone(), data_file.partition.clone())?;
            for position in &plan.positions {
                writer.delete(data_file_path, *position, Some(&partition_key))?;
            }
        }

        writer.close().await
    }
}

/// Live data file: the partition a covering DV must carry, and the seq it must not fall below.
struct LiveDataFile {
    partition_spec_id: i32,
    partition: Struct,
    sequence_number: i64,
}

/// Live delete inventory of a V3 table, taken in one manifest walk.
#[derive(Default)]
struct V3DeleteInventory {
    data_files: HashMap<String, LiveDataFile>,
    /// Parquet position deletes the filter admits.
    legacy_position_deletes: Vec<LiveDeleteEntry>,
    /// Puffin DVs keyed by the data file each one references.
    deletion_vectors: HashMap<String, LiveDeleteEntry>,
    /// Position deletes the filter rejected. A new DV covering their data file would shadow them.
    unconverted_position_deletes: Vec<LiveDeleteEntry>,
    /// Filter-admitted deletes the size gate declined. They stay live like the rejected ones.
    gate_declined_position_deletes: Vec<LiveDeleteEntry>,
}

impl V3DeleteInventory {
    /// Route one live position-delete entry into the inventory.
    ///
    /// # Errors
    ///
    /// Neither Parquet nor Puffin, a DV with no referenced data file, or a second live DV for one file.
    fn admit_position_delete(
        &mut self,
        metadata: &TableMetadata,
        partition_filter: &mut PartitionFilter,
        data_file: &DataFile,
        sequence_number: i64,
    ) -> Result<()> {
        let entry = LiveDeleteEntry {
            data_file: data_file.clone(),
            sequence_number,
        };
        match data_file.file_format() {
            DataFileFormat::Puffin => {
                let referenced = referenced_data_file_location(data_file).ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Deletion vector '{}' names no referenced data file",
                            data_file.file_path()
                        ),
                    )
                })?;
                if self
                    .deletion_vectors
                    .insert(referenced.clone(), entry)
                    .is_some()
                {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!("Data file '{referenced}' has more than one live deletion vector"),
                    ));
                }
                Ok(())
            }
            DataFileFormat::Parquet => {
                if partition_filter.matches(metadata, data_file)? {
                    self.legacy_position_deletes.push(entry);
                } else {
                    self.unconverted_position_deletes.push(entry);
                }
                Ok(())
            }
            // Refuse, do not skip. Skipping would make zero counts mean "did not look".
            format => {
                if partition_filter.matches(metadata, data_file)? {
                    return Err(Error::new(
                        ErrorKind::FeatureUnsupported,
                        format!(
                            "Position delete '{}' is {format}: only Parquet position deletes and \
                             Puffin deletion vectors are supported on format version 3",
                            data_file.file_path()
                        ),
                    ));
                }
                self.unconverted_position_deletes.push(entry);
                Ok(())
            }
        }
    }

    /// The live data file at `data_file_path`. A miss is a planner bug, not a table state.
    fn live_data_file(&self, data_file_path: &str) -> Result<&LiveDataFile> {
        self.data_files.get(data_file_path).ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                format!(
                    "Planned deletion vector names data file '{data_file_path}', which is not live"
                ),
            )
        })
    }
}

/// Merged positions and the data sequence number to stamp on one written DV.
#[derive(Default)]
struct DeletionVectorPlan {
    positions: Vec<u64>,
    sequence_number: i64,
}

/// Refuse a run that would leave a live position delete shadowed by a DV this run writes.
/// Re-derives reader routing; it does not share `PopulatedDeleteFileIndex` keying.
///
/// # Errors
///
/// `DataInvalid`. Widen the filter, unless the delete is ORC or Avro, which no width converts.
fn refuse_shadowed_deletes(
    inventory: &V3DeleteInventory,
    plans: &HashMap<String, DeletionVectorPlan>,
) -> Result<()> {
    if plans.is_empty() {
        return Ok(());
    }
    let mut planned_partitions: HashSet<(i32, &Struct)> = HashSet::new();
    for data_file_path in plans.keys() {
        let data_file = inventory.live_data_file(data_file_path)?;
        planned_partitions.insert((data_file.partition_spec_id, &data_file.partition));
    }

    for entry in &inventory.unconverted_position_deletes {
        let delete_file = &entry.data_file;
        let Some(shadowed_data_file) = shadowed_data_file(plans, &planned_partitions, delete_file)
        else {
            continue;
        };
        let remedy = if delete_file.file_format() == DataFileFormat::Parquet {
            "Widen the filter so the same run converts it."
        } else {
            "This arm cannot read that format, so NO filter setting converts this table."
        };
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Position delete '{}' still applies to {shadowed_data_file} but the filter excluded \
                 it: the deletion vector this run would write there SHADOWS it and its deleted rows \
                 would come back. {remedy}",
                delete_file.file_path()
            ),
        ));
    }
    for entry in &inventory.gate_declined_position_deletes {
        let delete_file = &entry.data_file;
        let Some(shadowed_data_file) = shadowed_data_file(plans, &planned_partitions, delete_file)
        else {
            continue;
        };
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Position delete '{}' still applies to {shadowed_data_file} but the size gate \
                 declined it: the deletion vector this run would write there SHADOWS it and its \
                 deleted rows would come back. Rerun once its partition holds enough files, or set \
                 rewrite-all.",
                delete_file.file_path()
            ),
        ));
    }
    Ok(())
}

fn shadowed_data_file(
    plans: &HashMap<String, DeletionVectorPlan>,
    planned_partitions: &HashSet<(i32, &Struct)>,
    delete_file: &DataFile,
) -> Option<String> {
    match referenced_data_file_location(delete_file) {
        Some(referenced) => plans.contains_key(&referenced).then_some(referenced),
        // Partition-scoped: do not apply the seq rule. A false alarm beats losing rows.
        None => planned_partitions
            .contains(&(delete_file.partition_spec_id, delete_file.partition()))
            .then(|| "a data file in the same partition".to_string()),
    }
}

/// Stamp for one written DV, read back from the plan it came from.
fn deletion_vector_sequence_number(
    delete_file: &DataFile,
    plans: &HashMap<String, DeletionVectorPlan>,
) -> Result<i64> {
    delete_file
        .referenced_data_file()
        .and_then(|path| plans.get(&path))
        .map(|plan| plan.sequence_number)
        .ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                format!(
                    "Written deletion vector '{}' has no planned sequence number",
                    delete_file.file_path()
                ),
            )
        })
}

/// Result counts of one V3 rewrite. Sum bytes over distinct paths: each DV `DataFile` carries the
/// whole Puffin size, so a per-entry sum would count the same bytes once per blob.
fn summarize_v3_rewrite(
    rewritten_files: &[DataFile],
    added_files: &[DataFile],
) -> Result<RewritePositionDeleteFilesResult> {
    let distinct_bytes = |files: &[DataFile]| -> Result<u64> {
        let mut seen: HashSet<&str> = HashSet::new();
        let mut total: u64 = 0;
        for file in files {
            if seen.insert(file.file_path()) {
                total = total.checked_add(file.file_size_in_bytes).ok_or_else(|| {
                    Error::new(ErrorKind::Unexpected, "rewrite bytes count overflow")
                })?;
            }
        }
        Ok(total)
    };

    Ok(RewritePositionDeleteFilesResult {
        rewritten_delete_files_count: rewritten_files.len(),
        added_delete_files_count: added_files.len(),
        rewritten_bytes_count: distinct_bytes(rewritten_files)?,
        added_bytes_count: distinct_bytes(added_files)?,
    })
}
