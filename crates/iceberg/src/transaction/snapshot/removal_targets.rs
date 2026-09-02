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

use crate::spec::{DataFile, ManifestContentType};

pub(super) type OwnedDeleteFileKey = (String, Option<i64>, Option<i64>);

type DeleteFileKey<'a> = (&'a str, Option<i64>, Option<i64>);

pub(super) fn owned_delete_key(file: &DataFile) -> OwnedDeleteFileKey {
    (
        file.file_path().to_string(),
        file.content_offset(),
        file.content_size_in_bytes(),
    )
}

fn delete_key(file: &DataFile) -> DeleteFileKey<'_> {
    (
        file.file_path(),
        file.content_offset(),
        file.content_size_in_bytes(),
    )
}

pub(super) struct RemovalTargets<'a> {
    data_paths: HashSet<&'a str>,
    delete_keys: HashSet<DeleteFileKey<'a>>,
}

#[derive(Default)]
pub(super) struct RemovalHits {
    data_paths: HashSet<String>,
    delete_keys: HashSet<OwnedDeleteFileKey>,
}

impl<'a> RemovalTargets<'a> {
    pub(super) fn new(
        removed_data_files: &'a [DataFile],
        removed_delete_files: &'a [DataFile],
    ) -> Self {
        Self {
            data_paths: removed_data_files
                .iter()
                .map(|file| file.file_path())
                .collect(),
            delete_keys: removed_delete_files.iter().map(delete_key).collect(),
        }
    }

    pub(super) fn matches(&self, content: ManifestContentType, file: &DataFile) -> bool {
        match content {
            ManifestContentType::Deletes => self.delete_keys.contains(&delete_key(file)),
            ManifestContentType::Data => self.data_paths.contains(file.file_path()),
        }
    }

    pub(super) fn missing(&self, hits: &RemovalHits) -> Vec<&str> {
        let mut missing: Vec<&str> = self
            .data_paths
            .iter()
            .filter(|path| !hits.data_paths.contains(**path))
            .copied()
            .collect();
        missing.extend(
            self.delete_keys
                .iter()
                .filter(|key| {
                    !hits
                        .delete_keys
                        .contains(&(key.0.to_string(), key.1, key.2))
                })
                .map(|key| key.0),
        );
        missing
    }
}

impl RemovalHits {
    pub(super) fn record(&mut self, content: ManifestContentType, file: &DataFile) {
        match content {
            ManifestContentType::Deletes => {
                self.delete_keys.insert(owned_delete_key(file));
            }
            ManifestContentType::Data => {
                self.data_paths.insert(file.file_path().to_string());
            }
        }
    }
}
