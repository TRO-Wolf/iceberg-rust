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

use super::MEMORY_CATALOG_METADATA_NAMING;
use crate::io::FileIO;
use crate::spec::TableMetadata;
use crate::{Error, ErrorKind, MetadataLocation, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum MetadataNaming {
    #[default]
    Uuid,
    Hadoop,
}

impl MetadataNaming {
    pub(crate) fn from_props(props: &HashMap<String, String>) -> Result<Self> {
        match props
            .get(MEMORY_CATALOG_METADATA_NAMING)
            .map(String::as_str)
        {
            None | Some("uuid") => Ok(Self::Uuid),
            Some("hadoop") => Ok(Self::Hadoop),
            Some(other) => Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Invalid value {other:?} for catalog property {MEMORY_CATALOG_METADATA_NAMING}: expected \"uuid\" or \"hadoop\""
                ),
            )),
        }
    }

    pub(crate) fn first_location(self, metadata: &TableMetadata) -> Result<MetadataLocation> {
        match self {
            Self::Uuid => MetadataLocation::for_metadata(metadata),
            Self::Hadoop => MetadataLocation::for_hadoop_metadata(metadata),
        }
    }

    pub(crate) fn ensure_rename_supported(self) -> Result<()> {
        match self {
            Self::Uuid => Ok(()),
            Self::Hadoop => Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "Cannot rename Hadoop tables",
            )),
        }
    }

    pub(crate) async fn write_first_metadata(
        self,
        file_io: &FileIO,
        metadata: &TableMetadata,
        location: &MetadataLocation,
    ) -> Result<()> {
        let path = location.to_string();
        match self {
            Self::Uuid => metadata.write_to(file_io, &path).await,
            Self::Hadoop => {
                metadata.write_commit_metadata(file_io, &path).await?;
                let result = publish_first_version_hint(file_io, location).await;
                if result.is_err() {
                    remove_after_failed_create(file_io, &path).await;
                }
                result
            }
        }
    }

    pub(crate) async fn advance_version_hint(self, file_io: &FileIO, metadata_location: &str) {
        if self != Self::Hadoop {
            return;
        }
        let Ok(location) = MetadataLocation::from_file_path(metadata_location) else {
            return;
        };
        if let Err(error) = write_version_hint(file_io, &location).await {
            tracing::warn!(
                ?error,
                metadata_location,
                "committed Hadoop metadata but failed to write version-hint.text"
            );
        }
    }
}

async fn write_version_hint(file_io: &FileIO, location: &MetadataLocation) -> Result<()> {
    let Some((path, version)) = location.hadoop_version_hint() else {
        return Ok(());
    };
    file_io.new_output(path)?.write(version.into()).await
}

async fn publish_first_version_hint(file_io: &FileIO, location: &MetadataLocation) -> Result<()> {
    let Some((path, version)) = location.hadoop_version_hint() else {
        return Ok(());
    };
    let existed = file_io.exists(&path).await?;
    let result = file_io.new_output(&path)?.write(version.into()).await;
    if result.is_err() && !existed {
        remove_after_failed_create(file_io, &path).await;
    }
    result
}

async fn remove_after_failed_create(file_io: &FileIO, path: &str) {
    if let Err(delete_error) = file_io.delete(path).await {
        tracing::warn!(
            ?delete_error,
            path,
            "failed to remove a file written by a failed Hadoop create"
        );
    }
}
