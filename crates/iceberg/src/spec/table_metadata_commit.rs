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

use std::str::FromStr;

use bytes::Bytes;

use super::table_metadata::TableMetadata;
use super::view_metadata::ViewMetadata;
use crate::catalog::MetadataLocation;
use crate::io::FileIO;
use crate::{Error, ErrorKind, Result};

impl TableMetadata {
    /// Write staged commit output, creating a Hadoop `vN` file exclusively.
    pub async fn write_commit_metadata(
        &self,
        file_io: &FileIO,
        metadata_location: impl AsRef<str>,
    ) -> Result<()> {
        let location = metadata_location.as_ref();
        if !is_hadoop_location(location) {
            return self.write_to(file_io, location).await;
        }
        write_hadoop_version_bytes(file_io, location, serde_json::to_vec(self)?.into()).await
    }
}

impl ViewMetadata {
    /// Write staged view commit output, creating a Hadoop `vN` file exclusively.
    pub async fn write_commit_metadata(
        &self,
        file_io: &FileIO,
        metadata_location: impl AsRef<str>,
    ) -> Result<()> {
        let location = metadata_location.as_ref();
        if !is_hadoop_location(location) {
            return self.write_to(file_io, location).await;
        }
        write_hadoop_version_bytes(file_io, location, serde_json::to_vec(self)?.into()).await
    }
}

fn is_hadoop_location(location: &str) -> bool {
    MetadataLocation::from_str(location).is_ok_and(|parsed| parsed.is_hadoop_convention())
}

fn version_exists_conflict(location: &str, existing: &str) -> Error {
    Error::new(
        ErrorKind::CatalogCommitConflicts,
        format!(
            "Cannot commit table metadata to {location}: version file already exists ({existing})"
        ),
    )
    .with_retryable(true)
}

async fn write_hadoop_version_bytes(
    file_io: &FileIO,
    location: &str,
    payload: Bytes,
) -> Result<()> {
    if let Ok(parsed) = MetadataLocation::from_str(location)
        && let Some(siblings) = parsed.hadoop_version_siblings()
    {
        for sibling in &siblings {
            if file_io.exists(sibling).await? {
                return Err(version_exists_conflict(location, sibling));
            }
        }
    }
    match file_io.write_new(location, payload).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::PreconditionFailed => {
            Err(version_exists_conflict(location, location).with_source(error))
        }
        Err(error) => Err(error),
    }
}
