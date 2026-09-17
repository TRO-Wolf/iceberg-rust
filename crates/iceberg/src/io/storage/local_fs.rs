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

//! Local filesystem storage implementation for testing.
//!
//! This module provides a `LocalFsStorage` implementation that uses standard
//! Rust filesystem operations. It is primarily intended for unit testing
//! scenarios where tests need to read/write files on the local filesystem.

use std::fs;
use std::io::Write;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::io::{
    FileInfo, FileMetadata, FileRead, FileWrite, InputFile, OutputFile, Storage, StorageConfig,
    StorageFactory,
};
use crate::{Error, ErrorKind, Result};

/// Convert a filesystem modification time into milliseconds since the Unix epoch.
///
/// Mirrors how Java's `HadoopFileIO.listPrefix` reports `FileStatus.getModificationTime()` as
/// the `FileInfo.createdAtMillis` value. A modification time at or before the epoch (only
/// reachable on clocks set far in the past) clamps to `0` so the value stays non-negative.
fn modified_time_to_millis(modified: SystemTime) -> i64 {
    match modified.duration_since(UNIX_EPOCH) {
        Ok(duration) => i64::try_from(duration.as_millis()).unwrap_or(i64::MAX),
        Err(_) => 0,
    }
}

/// Local filesystem storage implementation.
///
/// This storage implementation uses standard Rust filesystem operations,
/// making it suitable for unit tests that need to read/write files on disk.
///
/// # Path Normalization
///
/// The storage normalizes paths to handle various formats:
/// - `file:///path/to/file` -> `/path/to/file`
/// - `file:/path/to/file` -> `/path/to/file`
/// - `/path/to/file` -> `/path/to/file`
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LocalFsStorage;

impl LocalFsStorage {
    /// Create a new `LocalFsStorage` instance.
    pub fn new() -> Self {
        Self
    }

    /// Normalize a path by removing scheme prefixes.
    ///
    /// This handles the following formats:
    /// - `file:///path` -> `/path`
    /// - `file://path` -> `/path` (treats as absolute)
    /// - `file:/path` -> `/path`
    /// - `/path` -> `/path`
    pub(crate) fn normalize_path(path: &str) -> PathBuf {
        let path = if let Some(stripped) = path.strip_prefix("file://") {
            // file:///path -> /path or file://path -> /path
            if stripped.starts_with('/') {
                stripped.to_string()
            } else {
                format!("/{stripped}")
            }
        } else if let Some(stripped) = path.strip_prefix("file:") {
            // file:/path -> /path
            if stripped.starts_with('/') {
                stripped.to_string()
            } else {
                format!("/{stripped}")
            }
        } else {
            path.to_string()
        };
        PathBuf::from(path)
    }
}

#[async_trait]
#[typetag::serde]
impl Storage for LocalFsStorage {
    async fn exists(&self, path: &str) -> Result<bool> {
        let path = Self::normalize_path(path);
        Ok(path.exists())
    }

    async fn metadata(&self, path: &str) -> Result<FileMetadata> {
        let path = Self::normalize_path(path);
        let metadata = fs::metadata(&path).map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Failed to get metadata for {}: {}", path.display(), e),
            )
        })?;
        Ok(FileMetadata {
            size: metadata.len(),
        })
    }

    async fn read(&self, path: &str) -> Result<Bytes> {
        let path = Self::normalize_path(path);
        let content = fs::read(&path).map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Failed to read file {}: {}", path.display(), e),
            )
        })?;
        Ok(Bytes::from(content))
    }

    async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
        let path = Self::normalize_path(path);
        let file = fs::File::open(&path).map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Failed to open file {}: {}", path.display(), e),
            )
        })?;
        Ok(Box::new(LocalFsFileRead::new(file)))
    }

    async fn write(&self, path: &str, bs: Bytes) -> Result<()> {
        let path = Self::normalize_path(path);

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Failed to create directory {}: {}", parent.display(), e),
                )
            })?;
        }

        fs::write(&path, &bs).map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                format!("Failed to write file {}: {}", path.display(), e),
            )
        })?;
        Ok(())
    }

    async fn write_new(&self, path: &str, bs: Bytes) -> Result<()> {
        let path = Self::normalize_path(path);

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Failed to create directory {}: {}", parent.display(), e),
                )
            })?;
        }

        let temp = staged_temp_path(&path)?;
        stage_and_publish(&temp, &path, &bs)
    }

    async fn writer(&self, path: &str) -> Result<Box<dyn FileWrite>> {
        let path = Self::normalize_path(path);

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Failed to create directory {}: {}", parent.display(), e),
                )
            })?;
        }

        let file = fs::File::create(&path).map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                format!("Failed to create file {}: {}", path.display(), e),
            )
        })?;
        Ok(Box::new(LocalFsFileWrite::new(file)))
    }

    async fn delete(&self, path: &str) -> Result<()> {
        let path = Self::normalize_path(path);
        if path.exists() {
            fs::remove_file(&path).map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Failed to delete file {}: {}", path.display(), e),
                )
            })?;
        }
        Ok(())
    }

    async fn delete_prefix(&self, path: &str) -> Result<()> {
        let path = Self::normalize_path(path);
        if path.is_dir() {
            fs::remove_dir_all(&path).map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Failed to delete directory {}: {}", path.display(), e),
                )
            })?;
        }
        Ok(())
    }

    /// Recursively list every file under `prefix`.
    ///
    /// # Prefix semantics: DIRECTORY
    ///
    /// `prefix` names a directory; the listing is the set of files in that directory tree
    /// (matching `delete_prefix`, which removes a directory tree, and Java's
    /// `HadoopFileIO.listPrefix` over a hierarchical filesystem, which walks
    /// `FileSystem.listFiles(prefix, recursive = true)`). Because the prefix is a directory
    /// boundary, a sibling directory `ab2/` is NOT reported for prefix `ab` — only entries
    /// genuinely nested under the `ab` directory are returned. A `prefix` that does not name
    /// an existing directory (a plain file, or a path that does not exist) yields an empty
    /// list, never an error — a legitimately empty listing must not be confused with a
    /// failure.
    ///
    /// Only files are reported; directories themselves are descended into but never emitted
    /// as entries. The walk uses an explicit stack (not recursion) so an arbitrarily deep
    /// tree cannot overflow the call stack.
    ///
    /// # Symlinks are skipped (deliberate, safety-load-bearing)
    ///
    /// Each entry is classified with [`std::fs::DirEntry::metadata`], which does NOT follow
    /// symlinks — so a symlink is neither descended into nor emitted as a file. This is the
    /// conservative posture an orphan-file sweep needs: a symlinked directory cannot form a
    /// walk cycle (`a/loop -> a` terminates), and a symlink that escapes the prefix
    /// (`table/out -> /elsewhere`) cannot pull files OUTSIDE the table root into the listing —
    /// which would otherwise become deletion of live data outside the table. It also matches
    /// `delete_prefix`: `remove_dir_all` removes a directory symlink itself but never follows
    /// it to delete the target, so `list` and `delete_prefix` agree to leave symlink targets
    /// alone. Do NOT change this to follow symlinks without re-opening that hole.
    async fn list(&self, prefix: &str) -> Result<Vec<FileInfo>> {
        let root = Self::normalize_path(prefix);
        if !root.is_dir() {
            return Ok(Vec::new());
        }

        let mut files = Vec::new();
        let mut directories = vec![root];
        while let Some(directory) = directories.pop() {
            let entries = fs::read_dir(&directory).map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Failed to read directory {}: {}", directory.display(), e),
                )
            })?;
            for entry in entries {
                let entry = entry.map_err(|e| {
                    Error::new(
                        ErrorKind::Unexpected,
                        format!(
                            "Failed to read directory entry under {}: {}",
                            directory.display(),
                            e
                        ),
                    )
                })?;
                let entry_path = entry.path();
                let metadata = entry.metadata().map_err(|e| {
                    Error::new(
                        ErrorKind::Unexpected,
                        format!("Failed to stat {}: {}", entry_path.display(), e),
                    )
                })?;

                if metadata.is_dir() {
                    directories.push(entry_path);
                    continue;
                }
                // Symlinks and other special entries are not regular files; skip them so the
                // listing reports only real data files (Java's listFiles likewise yields file
                // statuses, and a symlink target is not an orphan candidate here).
                if !metadata.is_file() {
                    continue;
                }

                let modified = metadata.modified().map_err(|e| {
                    Error::new(
                        ErrorKind::Unexpected,
                        format!(
                            "Failed to read modification time for {}: {}",
                            entry_path.display(),
                            e
                        ),
                    )
                })?;
                let location = entry_path
                    .to_str()
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!("Path {} is not valid UTF-8", entry_path.display()),
                        )
                    })?
                    .to_string();
                files.push(FileInfo::new(
                    location,
                    metadata.len(),
                    modified_time_to_millis(modified),
                ));
            }
        }
        Ok(files)
    }

    fn new_input(&self, path: &str) -> Result<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    fn new_output(&self, path: &str) -> Result<OutputFile> {
        Ok(OutputFile::new(Arc::new(self.clone()), path.to_string()))
    }
}

fn staged_temp_path(dest: &std::path::Path) -> Result<PathBuf> {
    let name = dest.file_name().ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Cannot stage temp file for directory path {}",
                dest.display()
            ),
        )
    })?;
    let mut temp = dest.to_path_buf();
    temp.set_file_name(format!(
        ".{}-tmp-{}",
        name.to_string_lossy(),
        Uuid::new_v4()
    ));
    Ok(temp)
}

fn stage_and_publish(temp: &std::path::Path, dest: &std::path::Path, bs: &[u8]) -> Result<()> {
    let staged = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temp)
        .and_then(|mut file| file.write_all(bs));
    if let Err(io_err) = staged {
        let _ = fs::remove_file(temp);
        return Err(Error::new(
            ErrorKind::Unexpected,
            format!("Failed to stage temp file {}: {io_err}", temp.display()),
        ));
    }
    match fs::hard_link(temp, dest) {
        Ok(()) => {}
        Err(io_err) if io_err.kind() == std::io::ErrorKind::AlreadyExists => {
            let _ = fs::remove_file(temp);
            return Err(Error::new(
                ErrorKind::PreconditionFailed,
                format!("Cannot create {}: file already exists", dest.display()),
            )
            .with_source(io_err));
        }
        Err(io_err) => {
            let _ = fs::remove_file(temp);
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!("Failed to publish {}: {io_err}", dest.display()),
            ));
        }
    }
    fs::remove_file(temp).map_err(|io_err| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to remove temp file {}: {io_err}", temp.display()),
        )
    })?;
    Ok(())
}

/// File reader for local filesystem storage.
///
/// # Concurrency
///
/// The file handle is held behind an [`Arc`] (no [`Mutex`](std::sync::Mutex), no shared seek
/// cursor) and every read is a *positioned* read — `read_exact_at` on unix, `seek_read` on
/// Windows — so any number of ranges can be read concurrently from the same handle without
/// serializing on a lock or racing a shared cursor. Each blocking syscall runs inside
/// [`tokio::task::spawn_blocking`] so it never stalls a tokio worker thread.
#[derive(Debug)]
pub struct LocalFsFileRead {
    file: Arc<fs::File>,
}

impl LocalFsFileRead {
    /// Create a new `LocalFsFileRead` with the given file.
    pub fn new(file: fs::File) -> Self {
        Self {
            file: Arc::new(file),
        }
    }
}

/// Perform a positioned `read_exact` at `offset` for `len` bytes without disturbing any shared
/// file cursor.
///
/// This is the synchronous, blocking core shared by every range read; callers invoke it from
/// inside [`tokio::task::spawn_blocking`]. On unix it uses
/// [`std::os::unix::fs::FileExt::read_exact_at`]; on Windows it loops over
/// [`std::os::windows::fs::FileExt::seek_read`] (which performs a positioned read and does not
/// move the handle's cursor) until the buffer is full, surfacing the same `UnexpectedEof`-class
/// short-read error as `read_exact`.
fn read_exact_at(file: &fs::File, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
    let mut buffer = vec![0u8; len];

    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(&mut buffer, offset)?;
    }

    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut filled = 0usize;
        while filled < len {
            match file.seek_read(&mut buffer[filled..], offset + filled as u64) {
                Ok(0) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "failed to fill whole buffer",
                    ));
                }
                Ok(n) => filled += n,
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
    }

    Ok(buffer)
}

#[async_trait]
impl FileRead for LocalFsFileRead {
    async fn read(&self, range: Range<u64>) -> Result<Bytes> {
        let len = usize::try_from(range.end - range.start).map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Read length {} does not fit in usize: {e}",
                    range.end - range.start
                ),
            )
        })?;
        let offset = range.start;
        let file = Arc::clone(&self.file);

        // Run the blocking positioned read off the async executor so the tokio worker is not
        // stalled by the `pread`/`seek_read` syscall.
        let buffer = tokio::task::spawn_blocking(move || read_exact_at(&file, offset, len))
            .await
            .map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Failed to join blocking read task: {e}"),
                )
            })?
            .map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Failed to read {len} bytes at offset {offset}: {e}"),
                )
            })?;

        Ok(Bytes::from(buffer))
    }
}

/// File writer for local filesystem storage.
///
/// This struct implements `FileWrite` for writing to local files.
#[derive(Debug)]
pub struct LocalFsFileWrite {
    file: Option<fs::File>,
}

impl LocalFsFileWrite {
    /// Create a new `LocalFsFileWrite` for the given file.
    pub fn new(file: fs::File) -> Self {
        Self { file: Some(file) }
    }
}

#[async_trait]
impl FileWrite for LocalFsFileWrite {
    async fn write(&mut self, bs: Bytes) -> Result<()> {
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| Error::new(ErrorKind::DataInvalid, "Cannot write to closed file"))?;

        file.write_all(&bs).map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                format!("Failed to write to file: {e}"),
            )
        })?;

        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        let file = self
            .file
            .take()
            .ok_or_else(|| Error::new(ErrorKind::DataInvalid, "File already closed"))?;

        file.sync_all()
            .map_err(|e| Error::new(ErrorKind::Unexpected, format!("Failed to sync file: {e}")))?;

        Ok(())
    }
}

/// Factory for creating `LocalFsStorage` instances.
///
/// This factory implements `StorageFactory` and creates `LocalFsStorage`
/// instances for the "file" scheme.
///
/// # Example
///
/// ```rust,ignore
/// use iceberg::io::{StorageConfig, StorageFactory, LocalFsStorageFactory};
///
/// let factory = LocalFsStorageFactory;
/// let config = StorageConfig::new();
/// let storage = factory.build(&config)?;
/// ```
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LocalFsStorageFactory;

#[typetag::serde]
impl StorageFactory for LocalFsStorageFactory {
    fn build(&self, _config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        Ok(Arc::new(LocalFsStorage::new()))
    }
}

#[cfg(test)]
#[path = "local_fs_tests.rs"]
mod tests;
