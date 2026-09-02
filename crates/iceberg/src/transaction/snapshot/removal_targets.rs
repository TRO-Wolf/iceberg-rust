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

use crate::spec::{DataFile, ManifestContentType};

pub(super) type DeleteFileKey<'a> = (&'a str, Option<i64>, Option<i64>);

pub(super) fn delete_key(file: &DataFile) -> DeleteFileKey<'_> {
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

    pub(super) fn has_data_targets(&self) -> bool {
        !self.data_paths.is_empty()
    }

    pub(super) fn has_delete_targets(&self) -> bool {
        !self.delete_keys.is_empty()
    }

    pub(super) fn wants(&self, content: ManifestContentType) -> bool {
        match content {
            ManifestContentType::Deletes => self.has_delete_targets(),
            ManifestContentType::Data => self.has_data_targets(),
        }
    }

    pub(super) fn missing_data_paths(&self, hits: &RemovalHits) -> Vec<&str> {
        self.data_paths
            .iter()
            .filter(|path| !hits.data_paths.contains(**path))
            .copied()
            .collect()
    }
}

impl RemovalHits {
    pub(super) fn record(&mut self, content: ManifestContentType, file: &DataFile) {
        if content == ManifestContentType::Data {
            self.data_paths.insert(file.file_path().to_string());
        }
    }
}

pub(super) struct DeleteFileMatcher<'a> {
    requested: &'a [DataFile],
    wanted: HashMap<DeleteFileKey<'a>, usize>,
    found: Vec<bool>,
}

impl<'a> DeleteFileMatcher<'a> {
    pub(super) fn new(requested: &'a [DataFile]) -> Self {
        let mut wanted = HashMap::with_capacity(requested.len());
        for (index, file) in requested.iter().enumerate() {
            wanted.insert(delete_key(file), index);
        }
        Self {
            requested,
            wanted,
            found: vec![false; requested.len()],
        }
    }

    pub(super) fn hit(&mut self, file: &DataFile) -> bool {
        match self.wanted.get(&delete_key(file)) {
            Some(index) => {
                self.found[*index] = true;
                true
            }
            None => false,
        }
    }

    pub(super) fn missing(&self) -> Vec<&'a str> {
        self.requested
            .iter()
            .enumerate()
            .filter(|(index, _)| !self.found[*index])
            .map(|(_, file)| file.file_path())
            .collect()
    }
}
