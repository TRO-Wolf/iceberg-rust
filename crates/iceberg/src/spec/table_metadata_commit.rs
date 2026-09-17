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

use super::table_metadata::TableMetadata;
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
        let hadoop =
            MetadataLocation::from_str(location).is_ok_and(|parsed| parsed.is_hadoop_convention());
        if !hadoop {
            return self.write_to(file_io, location).await;
        }
        match file_io
            .write_new(location, serde_json::to_vec(self)?.into())
            .await
        {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == ErrorKind::PreconditionFailed => Err(Error::new(
                ErrorKind::CatalogCommitConflicts,
                format!("Cannot commit table metadata to {location}: version file already exists"),
            )
            .with_retryable(true)
            .with_source(error)),
            Err(error) => Err(error),
        }
    }
}
