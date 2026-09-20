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

use std::collections::HashMap;

use crate::table::Table;
use crate::{Catalog, Error, ErrorKind, Result};

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
    pub async fn execute(self, _catalog: &dyn Catalog) -> Result<AddFilesResult> {
        let _ = (
            &self.table,
            &self.source,
            &self.partition_filter,
            self.check_duplicate_files,
            self.parallelism,
        );
        Err(Error::new(
            ErrorKind::FeatureUnsupported,
            "add_files is not implemented yet",
        ))
    }
}
