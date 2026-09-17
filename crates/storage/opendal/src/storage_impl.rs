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

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use iceberg::io::{FileInfo, FileMetadata, FileRead, FileWrite, InputFile, OutputFile, Storage};
use iceberg::{Error, ErrorKind, Result};

use super::utils::{from_opendal_error, join_list_location};
use super::{
    OpenDalReader, OpenDalStorage, OpenDalWriter, buffer_to_bytes,
    file_meta_from_complete_list_entry, list_entry_metadata_complete, stat_incomplete_list_entries,
};

#[typetag::serde(name = "OpenDalStorage")]
#[async_trait]
impl Storage for OpenDalStorage {
    async fn exists(&self, path: &str) -> Result<bool> {
        let (op, relative_path) = self.create_operator(&path)?;
        Ok(op.exists(relative_path).await.map_err(from_opendal_error)?)
    }

    async fn metadata(&self, path: &str) -> Result<FileMetadata> {
        let (op, relative_path) = self.create_operator(&path)?;
        let meta = op.stat(relative_path).await.map_err(from_opendal_error)?;
        Ok(FileMetadata {
            size: meta.content_length(),
        })
    }

    async fn read(&self, path: &str) -> Result<Bytes> {
        let (op, relative_path) = self.create_operator(&path)?;
        Ok(buffer_to_bytes(
            op.read(relative_path).await.map_err(from_opendal_error)?,
        ))
    }

    async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
        let (op, relative_path) = self.create_operator(&path)?;
        Ok(Box::new(OpenDalReader(
            op.reader(relative_path).await.map_err(from_opendal_error)?,
        )))
    }

    async fn write(&self, path: &str, bs: Bytes) -> Result<()> {
        let (op, relative_path) = self.create_operator(&path)?;
        op.write(relative_path, bs)
            .await
            .map_err(from_opendal_error)?;
        Ok(())
    }

    async fn write_new(&self, path: &str, bs: Bytes) -> Result<()> {
        let (op, relative_path) = self.create_operator(&path)?;
        match op
            .write_with(relative_path, bs.clone())
            .if_not_exists(true)
            .await
        {
            Ok(_) => Ok(()),
            Err(error)
                if matches!(
                    error.kind(),
                    opendal::ErrorKind::AlreadyExists | opendal::ErrorKind::ConditionNotMatch
                ) =>
            {
                Err(Error::new(
                    ErrorKind::PreconditionFailed,
                    format!("Cannot create {path}: file already exists"),
                )
                .with_source(error))
            }
            Err(error) if error.kind() == opendal::ErrorKind::Unsupported => {
                if self.exists(path).await? {
                    return Err(Error::new(
                        ErrorKind::PreconditionFailed,
                        format!("Cannot create {path}: file already exists"),
                    ));
                }
                self.write(path, bs).await
            }
            Err(error) => Err(from_opendal_error(error)),
        }
    }

    async fn writer(&self, path: &str) -> Result<Box<dyn FileWrite>> {
        let (op, relative_path) = self.create_operator(&path)?;
        Ok(Box::new(OpenDalWriter(
            op.writer(relative_path).await.map_err(from_opendal_error)?,
        )))
    }

    async fn delete(&self, path: &str) -> Result<()> {
        let (op, relative_path) = self.create_operator(&path)?;
        Ok(op.delete(relative_path).await.map_err(from_opendal_error)?)
    }

    async fn delete_prefix(&self, path: &str) -> Result<()> {
        let (op, relative_path) = self.create_operator(&path)?;
        let path = if relative_path.ends_with('/') {
            relative_path.to_string()
        } else {
            format!("{relative_path}/")
        };
        Ok(op.remove_all(&path).await.map_err(from_opendal_error)?)
    }

    /// Recursively list every file under `prefix`, as Java `HadoopFileIO.listPrefix` does. # Notes
    /// The prefix is normalized to a trailing `/`, so prefix `ab` never reports a sibling key
    /// `ab2/...`. Size and last-modified come from the list entry when
    /// [`list_entry_metadata_complete`] allows it, and from `stat` otherwise, so size 0 stays
    /// authoritative.
    async fn list(&self, path: &str) -> Result<Vec<FileInfo>> {
        let (op, relative_path) = self.create_operator(&path)?;
        // The base re-prefixes each entry back into the scheme-qualified location the
        // caller knows.
        let base = &path[..path.len() - relative_path.len()];

        let list_root = if relative_path.is_empty() || relative_path.ends_with('/') {
            relative_path.to_string()
        } else {
            format!("{relative_path}/")
        };

        let entries = op
            .list_with(&list_root)
            .recursive(true)
            .await
            .map_err(from_opendal_error)?;

        // Incomplete entries queue a `stat` keyed by slot index, so the result order does
        // not follow HEAD completion order.
        let mut locations: Vec<String> = Vec::with_capacity(entries.len());
        let mut ready_meta: Vec<Option<(u64, i64)>> = Vec::with_capacity(entries.len());
        let mut need_stat: Vec<(usize, String)> = Vec::new();

        for entry in entries {
            let list_meta = entry.metadata();
            // Skip directory markers and delete-marker entries (not live files).
            if !list_meta.is_file() || list_meta.is_deleted() {
                continue;
            }

            let location = join_list_location(base, entry.path());
            if list_entry_metadata_complete(list_meta) {
                locations.push(location);
                ready_meta.push(Some(file_meta_from_complete_list_entry(list_meta)));
            } else {
                let slot_idx = locations.len();
                need_stat.push((slot_idx, entry.path().to_string()));
                locations.push(location);
                ready_meta.push(None);
            }
        }

        stat_incomplete_list_entries(
            &op,
            &need_stat,
            self.list_stat_concurrency(),
            &mut ready_meta,
        )
        .await?;

        let mut files = Vec::with_capacity(locations.len());
        for (location, meta) in locations.into_iter().zip(ready_meta) {
            let (size, created_at_millis) = meta.ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("list stat did not produce metadata for {location}"),
                )
            })?;
            files.push(FileInfo::new(location, size, created_at_millis));
        }
        Ok(files)
    }

    #[allow(unreachable_code, unused_variables)]
    fn new_input(&self, path: &str) -> Result<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    #[allow(unreachable_code, unused_variables)]
    fn new_output(&self, path: &str) -> Result<OutputFile> {
        Ok(OutputFile::new(Arc::new(self.clone()), path.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "opendal-memory")]
    fn memory_storage() -> OpenDalStorage {
        OpenDalStorage::Memory {
            operator: crate::memory::memory_config_build().expect("memory operator builds"),
            operator_cache: crate::OperatorCache::default(),
        }
    }

    #[cfg(feature = "opendal-memory")]
    #[tokio::test]
    async fn test_opendal_memory_write_new_refuses_existing() {
        let storage = memory_storage();
        let path = "memory:/exclusive/new.txt";
        let first = Bytes::from("first");
        storage
            .write_new(path, first.clone())
            .await
            .expect("create-new on absent path succeeds");
        assert_eq!(storage.read(path).await.expect("read back"), first);
        let err = storage
            .write_new(path, Bytes::from("second"))
            .await
            .expect_err("create-new on existing path fails");
        assert_eq!(err.kind(), ErrorKind::PreconditionFailed);
        assert_eq!(
            storage.read(path).await.expect("winner bytes intact"),
            first
        );
    }
}
