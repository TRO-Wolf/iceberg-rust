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

use futures::{StreamExt, stream};

use super::SnapshotProducer;
use super::removal_targets::{RemovalHits, RemovalTargets};
use crate::delete_file_index::{is_deletion_vector, referenced_data_file_location};
use crate::error::Result;
use crate::io::FileIO;
use crate::spec::{
    DataContentType, DataFile, Manifest, ManifestContentType, ManifestEntry, ManifestFile,
    UNASSIGNED_SEQUENCE_NUMBER,
};
use crate::{Error, ErrorKind};

const DELETE_MANIFEST_SCAN_CONCURRENCY: usize = 8;

#[derive(Clone, Copy)]
struct DeleteFileExpiry {
    min_data_sequence_number: i64,
    every_manifest: bool,
}

enum ManifestScan {
    Carry(ManifestFile),
    Rewrite {
        manifest_file: ManifestFile,
        manifest: Manifest,
        expiry: Option<DeleteFileExpiry>,
    },
}

fn is_dangling_dv(data_file: &DataFile, removed_data_paths: &HashSet<&str>) -> bool {
    if !is_deletion_vector(data_file)
        || data_file.content_type() == DataContentType::EqualityDeletes
    {
        return false;
    }
    match data_file.referenced_data_file_ref() {
        Some(referenced) => removed_data_paths.contains(referenced),
        None => referenced_data_file_location(data_file)
            .is_some_and(|referenced| removed_data_paths.contains(referenced.as_str())),
    }
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

async fn scan_manifest(
    file_io: &FileIO,
    manifest_file: ManifestFile,
    targets: &RemovalTargets<'_>,
    expiry: Option<&DeleteFileExpiry>,
    dangling_dv_paths: Option<&HashSet<&str>>,
) -> Result<ManifestScan> {
    let content = manifest_file.content;
    let expiry = expiry.filter(|expiry| expiry.may_expire_in(&manifest_file));
    let may_hold_dangling = dangling_dv_paths.is_some()
        && (manifest_file.has_added_files() || manifest_file.has_existing_files());
    if !targets.wants(content) && expiry.is_none() && !may_hold_dangling {
        return Ok(ManifestScan::Carry(manifest_file));
    }
    let manifest = manifest_file.load_manifest(file_io).await?;
    let mut has_removal = false;
    let mut has_expired = false;
    let mut has_dangling = false;
    for entry in manifest.entries().iter().filter(|entry| entry.is_alive()) {
        has_removal = has_removal || targets.matches(content, entry.data_file());
        has_expired = has_expired || expiry.is_some_and(|expiry| expiry.expires(entry));
        has_dangling = has_dangling
            || dangling_dv_paths.is_some_and(|paths| is_dangling_dv(entry.data_file(), paths));
    }
    let expiry = expiry
        .filter(|expiry| expiry.every_manifest || has_removal)
        .copied();
    if !(has_removal || has_dangling || has_expired && expiry.is_some()) {
        return Ok(ManifestScan::Carry(manifest_file));
    }
    Ok(ManifestScan::Rewrite {
        manifest_file,
        manifest,
        expiry,
    })
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
            (drops_old_delete_files && targets.has_data_targets()).then(|| targets.data_paths());
        let delete_manifest_count = delete_manifests.len();
        let file_io = self.table.file_io().clone();
        let mut delete_scans = stream::iter(delete_manifests.into_iter().enumerate().map(
            |(index, manifest_file)| {
                let file_io = file_io.clone();
                let targets = &targets;
                let expiry = expiry.as_ref();
                async move {
                    scan_manifest(&file_io, manifest_file, targets, expiry, dangling_dv_paths)
                        .await
                        .map(|scan| (index, scan))
                }
            },
        ))
        .buffer_unordered(DELETE_MANIFEST_SCAN_CONCURRENCY);
        let mut scans: Vec<Option<ManifestScan>> =
            (0..delete_manifest_count).map(|_| None).collect();
        while let Some(result) = delete_scans.next().await {
            let (index, scan) = result?;
            scans[index] = Some(scan);
        }
        for scan in scans.into_iter().flatten() {
            match scan {
                ManifestScan::Carry(manifest_file) => filtered.push((manifest_file, false)),
                ManifestScan::Rewrite {
                    manifest_file,
                    manifest,
                    expiry,
                } => filtered.push((
                    self.rewrite_manifest_with_deletes(
                        &manifest_file,
                        &manifest,
                        &targets,
                        expiry.as_ref(),
                        dangling_dv_paths,
                        &mut hits,
                    )
                    .await?,
                    true,
                )),
            }
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
        dangling_dv_paths: Option<&HashSet<&str>>,
        hits: &mut RemovalHits,
    ) -> Result<(ManifestFile, bool)> {
        match scan_manifest(
            self.table.file_io(),
            manifest_file,
            targets,
            expiry,
            dangling_dv_paths,
        )
        .await?
        {
            ManifestScan::Carry(manifest_file) => Ok((manifest_file, false)),
            ManifestScan::Rewrite {
                manifest_file,
                manifest,
                expiry,
            } => {
                let rewritten = self
                    .rewrite_manifest_with_deletes(
                        &manifest_file,
                        &manifest,
                        targets,
                        expiry.as_ref(),
                        dangling_dv_paths,
                        hits,
                    )
                    .await?;
                Ok((rewritten, true))
            }
        }
    }

    async fn rewrite_manifest_with_deletes(
        &mut self,
        manifest_file: &ManifestFile,
        manifest: &Manifest,
        targets: &RemovalTargets<'_>,
        expiry: Option<&DeleteFileExpiry>,
        dangling_dv_paths: Option<&HashSet<&str>>,
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
