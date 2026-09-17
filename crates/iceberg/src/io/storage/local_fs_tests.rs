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

use tempfile::TempDir;

use super::*;

/// Risk: a secs/millis confusion or a wrong epoch base in the mtime->createdAtMillis
/// conversion would feed A2 wrong timestamps. Pins EXACT values at the boundaries: epoch ->
/// 0, epoch + 1500 ms -> 1500 (proving milliseconds, not seconds), and a pre-epoch
/// SystemTime clamps to 0 rather than going negative.
#[test]
fn test_modified_time_to_millis_is_exact_and_clamps_pre_epoch() {
    use std::time::Duration;

    assert_eq!(modified_time_to_millis(UNIX_EPOCH), 0);
    assert_eq!(
        modified_time_to_millis(UNIX_EPOCH + Duration::from_millis(1500)),
        1500,
        "must report milliseconds since the epoch, not seconds"
    );
    assert_eq!(
        modified_time_to_millis(UNIX_EPOCH - Duration::from_secs(1)),
        0,
        "a pre-epoch modification time must clamp to 0, never go negative"
    );
}

#[test]
fn test_normalize_path() {
    // Test file:/// prefix
    assert_eq!(
        LocalFsStorage::normalize_path("file:///path/to/file"),
        PathBuf::from("/path/to/file")
    );

    // Test file:// prefix (without leading slash in path)
    assert_eq!(
        LocalFsStorage::normalize_path("file://path/to/file"),
        PathBuf::from("/path/to/file")
    );

    // Test file:/ prefix
    assert_eq!(
        LocalFsStorage::normalize_path("file:/path/to/file"),
        PathBuf::from("/path/to/file")
    );

    // Test bare path
    assert_eq!(
        LocalFsStorage::normalize_path("/path/to/file"),
        PathBuf::from("/path/to/file")
    );
}

#[tokio::test]
async fn test_local_fs_storage_write_read() {
    let tmp_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let path = tmp_dir.path().join("test.txt");
    let path_str = path.to_str().unwrap();
    let content = Bytes::from("Hello, World!");

    // Write
    storage.write(path_str, content.clone()).await.unwrap();

    // Read
    let read_content = storage.read(path_str).await.unwrap();
    assert_eq!(read_content, content);
}

#[tokio::test]
async fn test_local_fs_storage_exists() {
    let tmp_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let path = tmp_dir.path().join("test.txt");
    let path_str = path.to_str().unwrap();

    // File doesn't exist initially
    assert!(!storage.exists(path_str).await.unwrap());

    // Write file
    storage.write(path_str, Bytes::from("test")).await.unwrap();

    // File exists now
    assert!(storage.exists(path_str).await.unwrap());
}

#[tokio::test]
async fn test_local_fs_storage_metadata() {
    let tmp_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let path = tmp_dir.path().join("test.txt");
    let path_str = path.to_str().unwrap();
    let content = Bytes::from("Hello, World!");

    storage.write(path_str, content.clone()).await.unwrap();

    let metadata = storage.metadata(path_str).await.unwrap();
    assert_eq!(metadata.size, content.len() as u64);
}

#[tokio::test]
async fn test_local_fs_storage_delete() {
    let tmp_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let path = tmp_dir.path().join("test.txt");
    let path_str = path.to_str().unwrap();

    storage.write(path_str, Bytes::from("test")).await.unwrap();
    assert!(storage.exists(path_str).await.unwrap());

    storage.delete(path_str).await.unwrap();
    assert!(!storage.exists(path_str).await.unwrap());
}

#[tokio::test]
async fn test_local_fs_storage_delete_prefix() {
    let tmp_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let dir_path = tmp_dir.path().join("subdir");
    let file1 = dir_path.join("file1.txt");
    let file2 = dir_path.join("file2.txt");

    // Create files in subdirectory
    storage
        .write(file1.to_str().unwrap(), Bytes::from("1"))
        .await
        .unwrap();
    storage
        .write(file2.to_str().unwrap(), Bytes::from("2"))
        .await
        .unwrap();

    // Delete prefix (directory)
    storage
        .delete_prefix(dir_path.to_str().unwrap())
        .await
        .unwrap();

    // Directory should be deleted
    assert!(!dir_path.exists());
}

#[tokio::test]
async fn test_local_fs_storage_reader() {
    let tmp_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let path = tmp_dir.path().join("test.txt");
    let path_str = path.to_str().unwrap();
    let content = Bytes::from("Hello, World!");

    storage.write(path_str, content.clone()).await.unwrap();

    let reader = storage.reader(path_str).await.unwrap();
    let read_content = reader.read(0..content.len() as u64).await.unwrap();
    assert_eq!(read_content, content);

    // Test partial read
    let partial = reader.read(0..5).await.unwrap();
    assert_eq!(partial, Bytes::from("Hello"));
}

/// Risk: the positioned-read rewrite (no shared cursor, no `Mutex`) must return byte-exact
/// results for overlapping ranges issued CONCURRENTLY against one shared reader — a
/// shared-cursor seek-then-read would corrupt interleaved reads. Reads run via spawned
/// tasks (current-thread runtime, so this also proves the blocking reads join without
/// deadlock); every slice is checked against the source bytes.
#[tokio::test]
async fn test_local_fs_concurrent_positioned_reads_are_byte_exact() {
    let tmp_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let path = tmp_dir.path().join("ranges.bin");
    let path_str = path.to_str().unwrap();

    // Deterministic, non-uniform content so a wrong offset/length is detectable.
    let content: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
    storage
        .write(path_str, Bytes::from(content.clone()))
        .await
        .unwrap();

    let reader: Arc<dyn FileRead> = storage.reader(path_str).await.unwrap().into();

    // A mix of overlapping, adjacent, full-span and single-byte ranges.
    let ranges: Vec<Range<u64>> = vec![
        0..4096,
        0..1,
        10..20,
        15..25, // overlaps 10..20
        100..2100,
        2000..4096, // overlaps 100..2100
        4095..4096,
        512..512, // empty range
    ];

    let mut handles = Vec::new();
    for range in ranges.clone() {
        let reader = Arc::clone(&reader);
        handles.push(tokio::spawn(async move {
            let bytes = reader.read(range.clone()).await.unwrap();
            (range, bytes)
        }));
    }

    for handle in handles {
        let (range, bytes) = handle.await.unwrap();
        let start = range.start as usize;
        let end = range.end as usize;
        assert_eq!(
            bytes.as_ref(),
            &content[start..end],
            "range {range:?} returned wrong bytes"
        );
    }
}

/// Risk: short-read / out-of-bounds behavior must remain a loud error (matching the prior
/// `read_exact` contract), not a silent truncation or a panic — a range past EOF must fail.
#[tokio::test]
async fn test_local_fs_read_past_eof_errors() {
    let tmp_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let path = tmp_dir.path().join("small.bin");
    let path_str = path.to_str().unwrap();

    storage
        .write(path_str, Bytes::from_static(b"hello"))
        .await
        .unwrap();

    let reader = storage.reader(path_str).await.unwrap();
    // Request more bytes than exist.
    let result = reader.read(0..100).await;
    let error = result.expect_err("reading past EOF must error, not truncate");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
}

#[tokio::test]
async fn test_local_fs_storage_writer() {
    let tmp_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let path = tmp_dir.path().join("test.txt");
    let path_str = path.to_str().unwrap();

    let mut writer = storage.writer(path_str).await.unwrap();
    writer.write(Bytes::from("Hello, ")).await.unwrap();
    writer.write(Bytes::from("World!")).await.unwrap();
    writer.close().await.unwrap();

    let content = storage.read(path_str).await.unwrap();
    assert_eq!(content, Bytes::from("Hello, World!"));
}

#[tokio::test]
async fn test_local_fs_file_write_double_close() {
    let tmp_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let path = tmp_dir.path().join("test.txt");
    let path_str = path.to_str().unwrap();

    let mut writer = storage.writer(path_str).await.unwrap();
    writer.write(Bytes::from("test")).await.unwrap();
    writer.close().await.unwrap();

    // Second close should fail
    let result = writer.close().await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_local_fs_file_write_after_close() {
    let tmp_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let path = tmp_dir.path().join("test.txt");
    let path_str = path.to_str().unwrap();

    let mut writer = storage.writer(path_str).await.unwrap();
    writer.close().await.unwrap();

    // Write after close should fail
    let result = writer.write(Bytes::from("test")).await;
    assert!(result.is_err());
}

#[test]
fn test_local_fs_storage_factory() {
    let factory = LocalFsStorageFactory;
    let config = StorageConfig::new();
    let storage = factory.build(&config).unwrap();

    // Verify we got a valid storage instance
    assert!(format!("{storage:?}").contains("LocalFsStorage"));
}

#[tokio::test]
async fn test_local_fs_creates_parent_directories() {
    let tmp_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let path = tmp_dir.path().join("a/b/c/test.txt");
    let path_str = path.to_str().unwrap();

    // Write should create parent directories
    storage.write(path_str, Bytes::from("test")).await.unwrap();

    assert!(path.exists());
}

/// Risk: a missed file at depth, or a directory leaking in as a "file", would make A2's
/// orphan set wrong. Pins the EXACT recursive file set across nested subdirectories,
/// correct sizes, and plausible (>0, <= now) timestamps.
#[tokio::test]
async fn test_list_returns_exact_recursive_file_set_with_sizes_and_times() {
    let tmp_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let root = tmp_dir.path();

    // Build a nested tree: top-level file + two levels of subdirectories.
    let top = root.join("top.txt");
    let nested = root.join("sub/nested.txt");
    let deep = root.join("sub/deeper/deep.txt");
    storage
        .write(top.to_str().unwrap(), Bytes::from("a"))
        .await
        .unwrap();
    storage
        .write(nested.to_str().unwrap(), Bytes::from("bb"))
        .await
        .unwrap();
    storage
        .write(deep.to_str().unwrap(), Bytes::from("ccc"))
        .await
        .unwrap();
    // An empty directory must contribute nothing (no directories-as-files).
    fs::create_dir_all(root.join("sub/empty_dir")).unwrap();

    let now_millis = modified_time_to_millis(SystemTime::now());
    let listed = storage.list(root.to_str().unwrap()).await.unwrap();

    // Exactly the three files, no directories.
    let mut locations: Vec<String> = listed.iter().map(|f| f.location.clone()).collect();
    locations.sort();
    let mut expected = vec![
        top.to_str().unwrap().to_string(),
        nested.to_str().unwrap().to_string(),
        deep.to_str().unwrap().to_string(),
    ];
    expected.sort();
    assert_eq!(locations, expected);

    // Sizes match the written bytes, by location.
    let size_of = |path: &std::path::Path| {
        listed
            .iter()
            .find(|f| f.location == path.to_str().unwrap())
            .unwrap()
            .size
    };
    assert_eq!(size_of(&top), 1);
    assert_eq!(size_of(&nested), 2);
    assert_eq!(size_of(&deep), 3);

    // Timestamps are plausible: strictly positive and not in the future.
    for file in &listed {
        assert!(
            file.created_at_millis > 0,
            "expected a positive mtime, got {}",
            file.created_at_millis
        );
        assert!(
            file.created_at_millis <= now_millis,
            "mtime {} should not exceed now {now_millis}",
            file.created_at_millis
        );
    }
}

/// Risk: over-listing is over-deletion in A2. A sibling directory `ab2/` must NEVER appear
/// when listing the `ab/` directory prefix (directory-boundary semantics).
#[tokio::test]
async fn test_list_excludes_sibling_directory_outside_prefix() {
    let tmp_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let root = tmp_dir.path();

    let inside = root.join("ab/inside.txt");
    let sibling = root.join("ab2/outside.txt");
    storage
        .write(inside.to_str().unwrap(), Bytes::from("in"))
        .await
        .unwrap();
    storage
        .write(sibling.to_str().unwrap(), Bytes::from("out"))
        .await
        .unwrap();

    let listed = storage
        .list(root.join("ab").to_str().unwrap())
        .await
        .unwrap();

    let locations: Vec<&str> = listed.iter().map(|f| f.location.as_str()).collect();
    assert_eq!(locations, vec![inside.to_str().unwrap()]);
    assert!(
        !locations.contains(&sibling.to_str().unwrap()),
        "sibling ab2/ leaked into the ab/ listing"
    );
}

/// Risk: an empty directory must be a legitimate empty answer, not an error (a downstream
/// caller distinguishes "no files" from "could not list").
#[tokio::test]
async fn test_list_empty_directory_is_empty_not_error() {
    let tmp_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let empty = tmp_dir.path().join("empty");
    fs::create_dir_all(&empty).unwrap();

    let listed = storage.list(empty.to_str().unwrap()).await.unwrap();
    assert!(listed.is_empty());
}

/// Risk: a prefix that is not an existing directory (here, never created) must yield an
/// empty list rather than erroring, matching `delete_prefix`'s no-op-on-missing behavior.
#[tokio::test]
async fn test_list_nonexistent_prefix_is_empty() {
    let tmp_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let missing = tmp_dir.path().join("does_not_exist");

    let listed = storage.list(missing.to_str().unwrap()).await.unwrap();
    assert!(listed.is_empty());
}

/// Risk: a plain file given as the prefix must not be reported as if it were its own
/// directory listing (local_fs uses directory semantics — a file is not a directory).
#[tokio::test]
async fn test_list_file_path_as_prefix_is_empty() {
    let tmp_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let file = tmp_dir.path().join("a.txt");
    storage
        .write(file.to_str().unwrap(), Bytes::from("x"))
        .await
        .unwrap();

    let listed = storage.list(file.to_str().unwrap()).await.unwrap();
    assert!(listed.is_empty());
}

/// Risk: a symlinked directory forming a cycle (`a/loop -> a`) must NOT make the walk loop
/// forever or error — it must terminate. The walk inspects each entry with `lstat`-style
/// metadata (`DirEntry::metadata`, which does not follow symlinks), so a symlink is neither
/// descended into nor emitted. Pins termination + that the symlink contributes no entry.
#[cfg(unix)]
#[tokio::test]
async fn test_list_symlink_cycle_terminates_and_skips_symlink() {
    use std::os::unix::fs::symlink;

    let tmp_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let root = tmp_dir.path();

    let real = root.join("a/real.txt");
    storage
        .write(real.to_str().unwrap(), Bytes::from("x"))
        .await
        .unwrap();
    // a/loop -> a  (a directory symlink that closes a cycle).
    symlink(root.join("a"), root.join("a/loop")).unwrap();

    let listed = storage
        .list(root.join("a").to_str().unwrap())
        .await
        .unwrap();
    let locations: Vec<&str> = listed.iter().map(|f| f.location.as_str()).collect();
    // Only the real file; the cycle did not hang and the symlink emitted nothing.
    assert_eq!(locations, vec![real.to_str().unwrap()]);
}

/// Risk: over-listing is over-deletion in A2. A symlink that escapes the prefix — pointing
/// at a directory OR a file OUTSIDE the table root — must never surface outside paths in the
/// listing, or the orphan sweep could delete live data outside the table. Pins that neither
/// a directory-symlink nor a file-symlink to an outside location leaks into the listing.
#[cfg(unix)]
#[tokio::test]
async fn test_list_symlink_escaping_prefix_does_not_leak_outside_files() {
    use std::os::unix::fs::symlink;

    let tmp_dir = TempDir::new().unwrap();
    let outside_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let root = tmp_dir.path();

    let inside = root.join("table/inside.txt");
    storage
        .write(inside.to_str().unwrap(), Bytes::from("in"))
        .await
        .unwrap();
    let outside_file = outside_dir.path().join("secret.txt");
    storage
        .write(outside_file.to_str().unwrap(), Bytes::from("secret"))
        .await
        .unwrap();
    // A directory-symlink and a file-symlink that both escape the `table` prefix.
    symlink(outside_dir.path(), root.join("table/out_dir")).unwrap();
    symlink(&outside_file, root.join("table/out_file")).unwrap();

    let listed = storage
        .list(root.join("table").to_str().unwrap())
        .await
        .unwrap();
    let locations: Vec<&str> = listed.iter().map(|f| f.location.as_str()).collect();
    assert_eq!(locations, vec![inside.to_str().unwrap()]);
    assert!(
        !locations.iter().any(|location| location.contains("secret")),
        "an outside file leaked through a symlink into the listing: {locations:?}"
    );
}

/// Risk: a mid-walk failure (here an unreadable subdirectory) must be PROPAGATED, not
/// silently swallowed — a silent skip makes the listing quietly incomplete, and A2 would
/// then treat genuinely-live files it never saw as already gone (or, with later inversions,
/// fail to protect them). Java's Hadoop `RemoteIterator` throws on such I/O errors; this
/// pins the same loud posture: an `Unexpected` error naming the unreadable directory.
#[cfg(unix)]
#[tokio::test]
async fn test_list_unreadable_subdirectory_errors_loudly_not_silently_skipped() {
    use std::os::unix::fs::PermissionsExt;

    let tmp_dir = TempDir::new().unwrap();
    let storage = LocalFsStorage::new();
    let root = tmp_dir.path();

    storage
        .write(
            root.join("readable.txt").to_str().unwrap(),
            Bytes::from("ok"),
        )
        .await
        .unwrap();
    let locked = root.join("locked");
    storage
        .write(
            locked.join("hidden.txt").to_str().unwrap(),
            Bytes::from("secret"),
        )
        .await
        .unwrap();
    // Drop read+exec so `read_dir` on `locked` fails mid-walk.
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();

    let result = storage.list(root.to_str().unwrap()).await;

    // Restore permissions first so the TempDir can be cleaned up regardless of assertions.
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).unwrap();

    let error = result.expect_err("an unreadable subdirectory must surface as an error");
    assert_eq!(error.kind(), ErrorKind::Unexpected);
    assert!(
        error.message().contains("locked"),
        "error must name the unreadable directory, got: {}",
        error.message()
    );
}

#[tokio::test]
async fn test_local_fs_storage_write_new_refuses_existing() {
    let tmp_dir = TempDir::new().expect("tempdir");
    let storage = LocalFsStorage::new();
    let path = tmp_dir.path().join("nested").join("new.txt");
    let path_str = path.to_str().expect("utf8 path");
    let first = Bytes::from("first");
    storage
        .write_new(path_str, first.clone())
        .await
        .expect("create-new on absent path succeeds");
    assert_eq!(storage.read(path_str).await.expect("read back"), first);
    let err = storage
        .write_new(path_str, Bytes::from("second"))
        .await
        .expect_err("create-new on existing path fails");
    assert_eq!(err.kind(), ErrorKind::PreconditionFailed);
    assert_eq!(
        storage.read(path_str).await.expect("winner bytes intact"),
        first
    );
    let residue: Vec<String> = std::fs::read_dir(tmp_dir.path().join("nested"))
        .expect("list dir")
        .map(|entry| {
            entry
                .expect("dir entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(residue, vec!["new.txt".to_string()]);
}

#[test]
fn test_stage_and_publish_success_removes_temp() {
    let tmp_dir = TempDir::new().expect("tempdir");
    let dest = tmp_dir.path().join("v3.metadata.json");
    let temp = tmp_dir.path().join("staged.tmp");
    super::stage_and_publish(&temp, &dest, b"winner").expect("publish lands");
    assert_eq!(std::fs::read(&dest).expect("read dest"), b"winner");
    assert!(!temp.exists(), "temp is removed after publish");
}

#[test]
fn test_stage_and_publish_collision_keeps_dest_removes_temp() {
    let tmp_dir = TempDir::new().expect("tempdir");
    let dest = tmp_dir.path().join("v3.metadata.json");
    let temp = tmp_dir.path().join("staged.tmp");
    std::fs::write(&dest, b"winner").expect("seed dest");
    let err = super::stage_and_publish(&temp, &dest, b"loser").expect_err("collision fails");
    assert_eq!(err.kind(), ErrorKind::PreconditionFailed);
    assert_eq!(std::fs::read(&dest).expect("dest intact"), b"winner");
    assert!(!temp.exists(), "temp is removed after collision");
}

#[test]
fn test_stage_and_publish_body_failure_leaves_no_dest() {
    let tmp_dir = TempDir::new().expect("tempdir");
    let dest = tmp_dir.path().join("v3.metadata.json");
    let temp = tmp_dir.path().join("blocked.tmp");
    std::fs::create_dir(&temp).expect("temp path is a directory");
    super::stage_and_publish(&temp, &dest, b"loser").expect_err("body write fails");
    assert!(!dest.exists(), "no partial final file remains");
}
