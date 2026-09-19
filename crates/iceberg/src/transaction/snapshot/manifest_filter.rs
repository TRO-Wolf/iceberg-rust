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

use std::collections::HashSet;

use super::SnapshotProducer;
use super::removal_targets::{RemovalHits, RemovalTargets};
use crate::delete_file_index::{is_deletion_vector, referenced_data_file_location};
use crate::error::Result;
use crate::spec::{
    DataFile, Manifest, ManifestContentType, ManifestEntry, ManifestFile,
    UNASSIGNED_SEQUENCE_NUMBER,
};
use crate::{Error, ErrorKind};

struct DeleteFileExpiry {
    min_data_sequence_number: i64,
    every_manifest: bool,
}

fn is_dangling_dv(data_file: &DataFile, removed_data_paths: &HashSet<String>) -> bool {
    is_deletion_vector(data_file)
        && referenced_data_file_location(data_file)
            .is_some_and(|referenced| removed_data_paths.contains(&referenced))
}

impl DeleteFileExpiry {
    fn new<'m>(
        filtered_data_manifests: impl IntoIterator<Item = &'m ManifestFile>,
        last_sequence_number: i64,
        every_manifest: bool,
    ) -> Self {
        let min_data_sequence_number = filtered_data_manifests
            .into_iter()
            .map(|manifest_file| manifest_file.min_sequence_number)
            .filter(|sequence_number| *sequence_number != UNASSIGNED_SEQUENCE_NUMBER)
            .fold(last_sequence_number, i64::min);
        Self {
            min_data_sequence_number,
            every_manifest,
        }
    }

    fn may_expire_in(&self, manifest_file: &ManifestFile) -> bool {
        (manifest_file.has_added_files() || manifest_file.has_existing_files())
            && manifest_file.min_sequence_number < self.min_data_sequence_number
    }

    fn expires(&self, entry: &ManifestEntry) -> bool {
        entry.is_alive()
            && entry.sequence_number().is_some_and(|sequence_number| {
                sequence_number > 0 && sequence_number < self.min_data_sequence_number
            })
    }
}

impl SnapshotProducer<'_> {
    pub(super) async fn process_deletes(
        &mut self,
        existing_manifests: Vec<ManifestFile>,
        removed_data_files: &[DataFile],
        removed_delete_files: &[DataFile],
        drops_old_delete_files: bool,
    ) -> Result<(Vec<ManifestFile>, Vec<DataFile>)> {
        if removed_data_files.is_empty() && removed_delete_files.is_empty() {
            return Ok((existing_manifests, vec![]));
        }

        let targets = RemovalTargets::new(removed_data_files, removed_delete_files);
        let mut hits = RemovalHits::default();
        let (data_manifests, delete_manifests): (Vec<_>, Vec<_>) = existing_manifests
            .into_iter()
            .partition(|manifest_file| manifest_file.content == ManifestContentType::Data);

        let mut filtered = Vec::with_capacity(data_manifests.len() + delete_manifests.len());
        for manifest_file in data_manifests {
            filtered.push(
                self.filter_manifest(manifest_file, &targets, None, None, &mut hits)
                    .await?,
            );
        }
        let expiry = drops_old_delete_files.then(|| {
            DeleteFileExpiry::new(
                filtered.iter().map(|(manifest_file, _)| manifest_file),
                self.table.metadata().last_sequence_number(),
                !removed_data_files.is_empty(),
            )
        });
        let dangling_dv_paths =
            (drops_old_delete_files && !removed_data_files.is_empty()).then(|| {
                removed_data_files
                    .iter()
                    .map(|data_file| data_file.file_path().to_string())
                    .collect::<HashSet<String>>()
            });
        for manifest_file in delete_manifests {
            filtered.push(
                self.filter_manifest(
                    manifest_file,
                    &targets,
                    expiry.as_ref(),
                    dangling_dv_paths.as_ref(),
                    &mut hits,
                )
                .await?,
            );
        }

        let missing = targets.missing_data_paths(&hits);
        if !missing.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Missing required files to delete: {}", missing.join(", ")),
            ));
        }

        let result_manifests = filtered
            .into_iter()
            .filter(|(manifest_file, rewritten)| {
                *rewritten || manifest_file.has_added_files() || manifest_file.has_existing_files()
            })
            .map(|(manifest_file, _)| manifest_file)
            .collect();
        Ok((result_manifests, hits.into_expired_delete_files()))
    }

    async fn filter_manifest(
        &mut self,
        manifest_file: ManifestFile,
        targets: &RemovalTargets<'_>,
        expiry: Option<&DeleteFileExpiry>,
        dangling_dv_paths: Option<&HashSet<String>>,
        hits: &mut RemovalHits,
    ) -> Result<(ManifestFile, bool)> {
        let content = manifest_file.content;
        let expiry = expiry.filter(|expiry| expiry.may_expire_in(&manifest_file));
        let may_hold_dangling = dangling_dv_paths.is_some()
            && (manifest_file.has_added_files() || manifest_file.has_existing_files());
        if !targets.wants(content) && expiry.is_none() && !may_hold_dangling {
            return Ok((manifest_file, false));
        }
        let manifest = manifest_file.load_manifest(self.table.file_io()).await?;

        let has_removal = manifest
            .entries()
            .iter()
            .any(|entry| entry.is_alive() && targets.matches(content, entry.data_file()));
        let expiry = expiry.filter(|expiry| expiry.every_manifest || has_removal);
        let has_expired = expiry
            .is_some_and(|expiry| manifest.entries().iter().any(|entry| expiry.expires(entry)));
        let has_dangling = dangling_dv_paths.is_some_and(|removed_data_paths| {
            manifest.entries().iter().any(|entry| {
                entry.is_alive() && is_dangling_dv(entry.data_file(), removed_data_paths)
            })
        });
        if !has_removal && !has_expired && !has_dangling {
            return Ok((manifest_file, false));
        }

        let rewritten = self
            .rewrite_manifest_with_deletes(
                &manifest_file,
                &manifest,
                targets,
                expiry,
                dangling_dv_paths,
                hits,
            )
            .await?;
        Ok((rewritten, true))
    }

    async fn rewrite_manifest_with_deletes(
        &mut self,
        manifest_file: &ManifestFile,
        manifest: &Manifest,
        targets: &RemovalTargets<'_>,
        expiry: Option<&DeleteFileExpiry>,
        dangling_dv_paths: Option<&HashSet<String>>,
        hits: &mut RemovalHits,
    ) -> Result<ManifestFile> {
        let content = manifest_file.content;
        let mut writer = self.new_filtering_manifest_writer(manifest_file)?;

        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }

            let entry = entry.as_ref().clone();
            if targets.matches(content, entry.data_file()) {
                hits.record(content, entry.data_file());
                writer.add_delete_entry(entry)?;
            } else if dangling_dv_paths.is_some_and(|removed_data_paths| {
                is_dangling_dv(entry.data_file(), removed_data_paths)
            }) || expiry.is_some_and(|expiry| expiry.expires(&entry))
            {
                hits.expire(entry.data_file());
                writer.add_delete_entry(entry)?;
            } else {
                writer.add_existing_entry(entry)?;
            }
        }

        writer.write_manifest_file().await
    }
}
