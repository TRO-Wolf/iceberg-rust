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
    pub(super) async fn rewrite_bin(
        &self,
        table: &Table,
        bin: AdmittedBin,
        live_paths: Option<&HashSet<Arc<str>>>,
        config: &ResolvedConfig,
    ) -> Result<RewrittenBin> {
        let (key, entries) = bin;

        let mut pairs: Vec<(String, i64)> = Vec::new();
        for entry in &entries {
            self.read_position_pairs(table, &entry.data_file, &mut pairs)
                .await?;
        }

        pairs.retain(|(path, _)| live_paths.is_some_and(|live| live.contains(path.as_str())));
        pairs.sort_unstable();

        let added = self
            .write_group_outputs(table, &key, &pairs, config)
            .await?;

        let max_seq = entries
            .iter()
            .map(|e| e.sequence_number)
            .max()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    "rewrite_bin called with an empty group (no sequence numbers)",
                )
            })?;

        Ok(RewrittenBin {
            deleted: entries.into_iter().map(|e| e.data_file).collect(),
            added: added.into_iter().map(|file| (file, max_seq)).collect(),
        })
    }

    async fn write_group_outputs(
        &self,
        table: &Table,
        key: &GroupKey,
        pairs: &[(String, i64)],
        config: &ResolvedConfig,
    ) -> Result<Vec<DataFile>> {
        let mut written: Vec<DataFile> = Vec::new();
        if pairs.is_empty() {
            return Ok(written);
        }
        let factory = Self::group_writer_factory(table, key, config)?;
        let mut start = 0;
        while start < pairs.len() {
            let mut end = pairs.len();
            if config.delete_granularity == DeleteGranularity::File {
                end = start + 1;
                while end < pairs.len() && pairs[end].0 == pairs[start].0 {
                    end += 1;
                }
            }
            match self
                .write_compacted_file(&factory, &pairs[start..end])
                .await
            {
                Ok(files) => written.extend(files),
                Err(error) => {
                    delete_paths_quietly(
                        table,
                        written.iter().map(|file| file.file_path().to_string()),
                    )
                    .await;
                    return Err(error);
                }
            }
            start = end;
        }
        Ok(written)
    }

    pub(super) async fn commit_bins(
        &self,
        catalog: &dyn Catalog,
        table: &mut Table,
        pending: &mut Vec<RewrittenBin>,
        starting_snapshot_id: i64,
        result: &mut RewritePositionDeleteFilesResult,
    ) -> Result<()> {
        if pending.is_empty() {
            return Ok(());
        }
        let bins = std::mem::take(pending);
        let added_paths: Vec<String> = bins
            .iter()
            .flat_map(|bin| {
                bin.added
                    .iter()
                    .map(|(file, _)| file.file_path().to_string())
            })
            .collect();

        let mut batch = RewritePositionDeleteFilesResult::default();
        let transaction = Transaction::new(&*table);
        let mut action = transaction.rewrite_files(Vec::new(), Vec::new());
        for bin in bins {
            batch.rewritten_delete_files_count += bin.deleted.len();
            batch.rewritten_bytes_count = batch
                .rewritten_bytes_count
                .checked_add(
                    bin.deleted
                        .iter()
                        .map(|file| file.file_size_in_bytes)
                        .sum::<u64>(),
                )
                .ok_or_else(|| {
                    Error::new(ErrorKind::Unexpected, "rewritten bytes count overflow")
                })?;
            batch.added_delete_files_count += bin.added.len();
            batch.added_bytes_count = batch
                .added_bytes_count
                .checked_add(
                    bin.added
                        .iter()
                        .map(|(file, _)| file.file_size_in_bytes)
                        .sum::<u64>(),
                )
                .ok_or_else(|| Error::new(ErrorKind::Unexpected, "added bytes count overflow"))?;
            action = action.delete_delete_files(bin.deleted);
            for (file, sequence_number) in bin.added {
                action = action.add_delete_file_with_sequence_number(file, sequence_number);
            }
        }

        let transaction = match action
            .validate_from_snapshot(starting_snapshot_id)
            .apply(transaction)
        {
            Ok(transaction) => transaction,
            Err(error) => {
                delete_paths_quietly(table, added_paths.iter().cloned()).await;
                return Err(error);
            }
        };
        match transaction.commit(catalog).await {
            Ok(committed) => {
                *table = committed;
                result.rewritten_delete_files_count += batch.rewritten_delete_files_count;
                result.added_delete_files_count += batch.added_delete_files_count;
                result.rewritten_bytes_count = result
                    .rewritten_bytes_count
                    .checked_add(batch.rewritten_bytes_count)
                    .ok_or_else(|| {
                        Error::new(ErrorKind::Unexpected, "rewritten bytes count overflow")
                    })?;
                result.added_bytes_count = result
                    .added_bytes_count
                    .checked_add(batch.added_bytes_count)
                    .ok_or_else(|| {
                        Error::new(ErrorKind::Unexpected, "added bytes count overflow")
                    })?;
                Ok(())
            }
            Err(error) => {
                delete_paths_quietly(table, added_paths.iter().cloned()).await;
                Err(error)
            }
        }
    }
}

pub(super) struct RewrittenBin {
    pub(super) deleted: Vec<DataFile>,
    pub(super) added: Vec<(DataFile, i64)>,
}

pub(super) async fn delete_uncommitted_files(table: &Table, bins: &[RewrittenBin]) {
    delete_paths_quietly(
        table,
        bins.iter().flat_map(|bin| {
            bin.added
                .iter()
                .map(|(file, _)| file.file_path().to_string())
        }),
    )
    .await;
}

async fn delete_paths_quietly(table: &Table, paths: impl Iterator<Item = String>) {
    for path in paths {
        let _ = table.file_io().delete(&path).await;
    }
}
