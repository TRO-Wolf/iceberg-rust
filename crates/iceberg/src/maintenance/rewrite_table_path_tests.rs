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

//! Offline unit tests for [`RewriteTablePath`](super::RewriteTablePath) — the FULL-rewrite metadata-path
//! rewrite. The pure path helpers and `replace_paths` are tested in isolation (a wide mutation-bait
//! net), and the full action is run against a real committed local-fs table carrying data, a
//! position-delete, and an equality-delete to exercise EVERY branch (the rewritten path GRAPH and the
//! per-class copy-plan direction).

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};

use super::*;
use crate::io::{FileIOBuilder, LocalFsStorageFactory};
use crate::memory::MemoryCatalogBuilder;
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, NestedField, PartitionSpec,
    PrimitiveType, Schema, Struct, TableMetadata, Type,
};
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
};
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};

// ---- pure path helpers (Java RewriteTablePathUtil parity) -------------------------------------

#[test]
fn maybe_append_file_separator_appends_only_when_missing() {
    assert_eq!(maybe_append_file_separator("s3://b/x"), "s3://b/x/");
    // MUTATION-BAIT: a double-append (always appending) would yield "s3://b/x//".
    assert_eq!(maybe_append_file_separator("s3://b/x/"), "s3://b/x/");
}

#[test]
fn relativize_strips_the_separator_appended_prefix() {
    assert_eq!(
        relativize("s3://b/src/data/d.parquet", "s3://b/src").expect("relativize"),
        "data/d.parquet"
    );
    // A trailing slash on the prefix is idempotent (maybeAppendFileSeparator).
    assert_eq!(
        relativize("s3://b/src/data/d.parquet", "s3://b/src/").expect("relativize"),
        "data/d.parquet"
    );
}

#[test]
fn relativize_errors_when_path_not_under_prefix() {
    // Java's relativize throws "Path %s does not start with %s".
    let err = relativize("s3://other/data/d.parquet", "s3://b/src").unwrap_err();
    assert!(
        err.to_string().contains("does not start with"),
        "expected a not-under-prefix error, got: {err}"
    );
}

#[test]
fn new_path_swaps_the_prefix_exactly() {
    assert_eq!(
        new_path("s3://b/src/data/d.parquet", "s3://b/src", "s3://b/tgt").expect("new_path"),
        "s3://b/tgt/data/d.parquet"
    );
    // MUTATION-BAIT: a literal `replace` (not prefix-anchored) would also rewrite an inner "src"
    // occurrence; newPath must only touch the leading prefix.
    assert_eq!(
        new_path("s3://b/src/src-data/d.parquet", "s3://b/src", "s3://b/tgt").expect("new_path"),
        "s3://b/tgt/src-data/d.parquet"
    );
}

#[test]
fn staging_path_mirrors_source_relative_layout_under_staging() {
    assert_eq!(
        staging_path("s3://b/src/metadata/m.avro", "s3://b/src", "file:///stage").expect("staging"),
        "file:///stage/metadata/m.avro"
    );
}

#[test]
fn replace_first_prefix_is_first_occurrence_only_unlike_new_path() {
    // The Java `location` field uses replaceFirst (FIRST occurrence), NOT newPath. So an inner repeat
    // of the prefix string is NOT rewritten (only the first occurrence is) — and crucially, the
    // location replace is NOT prefix-anchored the way newPath is. This asymmetry is the whole point.
    assert_eq!(
        replace_first_prefix("s3://b/src/t/src-suffix", "src", "TGT"),
        "s3://b/TGT/t/src-suffix",
        "replaceFirst rewrites only the FIRST occurrence of the (unanchored) source"
    );
}

// ---- replace_paths field-by-field (the metadata.json rewrite) ---------------------------------

/// Build a real committed multi-snapshot table, then assert `replace_paths` rewrites exactly the
/// path-bearing metadata fields (location, snapshot manifest_list, metadata-log, the four properties,
/// statistics) and PASSES THROUGH partition_statistics, refs, schemas, specs verbatim.
#[tokio::test]
async fn replace_paths_rewrites_only_path_fields() {
    let (catalog, file_io, _tmp) = local_fs_catalog().await;
    let table = create_table(&catalog).await;
    let location = table.metadata().location().to_string();
    let d1 = real_data_file(&file_io, &format!("{location}/data/d1.parquet"), b"d1").await;
    let table = append(&catalog, &table, vec![d1]).await;

    let metadata = table.metadata();
    let source = location.clone();
    let target = format!("{location}-TARGET");

    let rewritten = replace_paths(metadata, &source, &target).expect("replace_paths");

    // (1) location via replaceFirst.
    assert_eq!(rewritten.location(), format!("{location}-TARGET"));

    // (2) each snapshot's manifest_list via newPath.
    for snapshot in rewritten.snapshots() {
        assert!(
            snapshot.manifest_list().starts_with(&target),
            "manifest_list must be rewritten to the target prefix: {}",
            snapshot.manifest_list()
        );
        assert!(
            !snapshot.manifest_list().starts_with(&format!("{source}/")),
            "no snapshot manifest_list may retain the source prefix"
        );
    }

    // (3) metadata-log entries via newPath.
    for entry in rewritten.metadata_log() {
        assert!(
            entry.metadata_file.starts_with(&target),
            "metadata-log entry must be rewritten: {}",
            entry.metadata_file
        );
    }

    // The schemas/specs are carried verbatim (count + current id unchanged).
    assert_eq!(
        rewritten.current_schema().schema_id(),
        metadata.current_schema().schema_id(),
        "the current schema id is carried verbatim"
    );
    assert_eq!(
        rewritten.current_snapshot_id(),
        metadata.current_snapshot_id(),
        "currentSnapshotId is carried"
    );
}

/// MUTATION-BAIT (the four path properties): exactly the four `write.*.path` keys are rewritten; every
/// other property is left untouched.
#[tokio::test]
async fn replace_paths_rewrites_exactly_the_four_path_properties() {
    let (catalog, file_io, _tmp) = local_fs_catalog().await;
    let table = create_table(&catalog).await;
    let location = table.metadata().location().to_string();
    let d1 = real_data_file(&file_io, &format!("{location}/data/d1.parquet"), b"d1").await;
    let table = append(&catalog, &table, vec![d1]).await;

    // Inject the four path properties + one unrelated property directly into a cloned metadata.
    let mut metadata = table.metadata().clone();
    let source = location.clone();
    let target = format!("{location}-T");
    metadata.properties.insert(
        "write.object-storage.path".to_string(),
        format!("{source}/obj"),
    );
    metadata.properties.insert(
        "write.folder-storage.path".to_string(),
        format!("{source}/folder"),
    );
    metadata
        .properties
        .insert("write.data.path".to_string(), format!("{source}/data"));
    metadata.properties.insert(
        "write.metadata.path".to_string(),
        format!("{source}/metadata"),
    );
    metadata
        .properties
        .insert("commit.retry.num-retries".to_string(), "7".to_string());

    let rewritten = replace_paths(&metadata, &source, &target).expect("replace_paths");

    for key in [
        "write.object-storage.path",
        "write.folder-storage.path",
        "write.data.path",
        "write.metadata.path",
    ] {
        let value = rewritten
            .properties()
            .get(key)
            .expect("path property present");
        assert!(
            value.starts_with(&target),
            "path property {key} must be rewritten: {value}"
        );
    }
    // The unrelated property is UNTOUCHED.
    assert_eq!(
        rewritten.properties().get("commit.retry.num-retries"),
        Some(&"7".to_string()),
        "a non-path property must NOT be rewritten"
    );
}

/// MUTATION-BAIT (the 1.10.0 divergence): `statistics` IS rewritten but `partition_statistics` is
/// PASSED THROUGH un-rewritten. A port that symmetrically rewrote partition stats fails here.
#[tokio::test]
async fn replace_paths_rewrites_statistics_but_passes_partition_statistics_through() {
    use crate::spec::{PartitionStatisticsFile, StatisticsFile};

    let (catalog, file_io, _tmp) = local_fs_catalog().await;
    let table = create_table(&catalog).await;
    let location = table.metadata().location().to_string();
    let d1 = real_data_file(&file_io, &format!("{location}/data/d1.parquet"), b"d1").await;
    let table = append(&catalog, &table, vec![d1]).await;

    let mut metadata = table.metadata().clone();
    let snapshot_id = metadata.current_snapshot_id().expect("snapshot");
    let source = location.clone();
    let target = format!("{location}-T");

    let stats_path = format!("{source}/metadata/stats.puffin");
    let part_stats_path = format!("{source}/metadata/partition-stats.parquet");
    metadata.statistics.insert(snapshot_id, StatisticsFile {
        snapshot_id,
        statistics_path: stats_path.clone(),
        file_size_in_bytes: 1,
        file_footer_size_in_bytes: 1,
        key_metadata: None,
        blob_metadata: Vec::new(),
    });
    metadata
        .partition_statistics
        .insert(snapshot_id, PartitionStatisticsFile {
            snapshot_id,
            statistics_path: part_stats_path.clone(),
            file_size_in_bytes: 1,
        });

    let rewritten = replace_paths(&metadata, &source, &target).expect("replace_paths");

    // statistics IS rewritten.
    let rewritten_stats = rewritten
        .statistics_iter()
        .next()
        .expect("a statistics file");
    assert!(
        rewritten_stats.statistics_path.starts_with(&target),
        "the statistics file path MUST be rewritten: {}",
        rewritten_stats.statistics_path
    );

    // partition_statistics is PASSED THROUGH un-rewritten (the 1.10.0 divergence).
    let rewritten_part = rewritten
        .partition_statistics_iter()
        .next()
        .expect("a partition statistics file");
    assert_eq!(
        rewritten_part.statistics_path, part_stats_path,
        "partition_statistics MUST be passed through un-rewritten (Java 1.10.0 divergence)"
    );
}

// ---- the full action: the rewritten path GRAPH + the copy-plan --------------------------------

/// The end-to-end action over a table carrying data + a position-delete + an equality-delete: every
/// rewritten location is under the target (NOTHING retains the source prefix), the staged metadata
/// re-reads, and the copy-plan carries the per-class direction.
#[tokio::test]
async fn execute_rewrites_the_graph_and_emits_the_copy_plan() {
    let (catalog, file_io, _tmp) = local_fs_catalog().await;
    let table = create_table(&catalog).await;
    let location = table.metadata().location().to_string();

    // s1: a data file.
    let d1 = real_data_file(&file_io, &format!("{location}/data/d1.parquet"), b"d1").await;
    let table = append(&catalog, &table, vec![d1.clone()]).await;

    // s2: a row-delta carrying a REAL position-delete (on d1) + a real equality-delete.
    let pos_delete = write_real_pos_delete(
        &table,
        &file_io,
        &format!("{location}/data"),
        d1.file_path(),
    )
    .await;
    let eq_delete = real_eq_delete(&file_io, &format!("{location}/data/eq-deletes.parquet")).await;
    let table = row_delta(&catalog, &table, vec![pos_delete, eq_delete]).await;

    let source = location.clone();
    let target = "s3://bucket/relocated".to_string();
    let staging = format!("{location}-staging");

    let result = super::RewriteTablePath::new(table.clone())
        .rewrite_location_prefix(&source, &target)
        .staging_location(&staging)
        .execute(&file_io)
        .await
        .expect("execute rewrite table path");

    // The staged metadata.json re-reads and its location is rewritten.
    let staged = TableMetadata::read_from(&file_io, &result.staged_metadata_location)
        .await
        .expect("read staged metadata");
    assert!(
        staged.location().starts_with(&target),
        "the staged metadata.json location must be rewritten: {}",
        staged.location()
    );

    // Every rewritten TARGET path in the copy-plan is under the target prefix; NONE retains the
    // source prefix (the no-source-leak invariant).
    let source_with_sep = format!("{source}/");
    assert!(
        !result.copy_plan.is_empty(),
        "the copy-plan must be non-empty"
    );
    for (from, to) in &result.copy_plan {
        assert!(
            to.starts_with(&target),
            "every copy-plan TARGET must be under the target prefix: {to}"
        );
        assert!(
            !to.starts_with(&source_with_sep),
            "no copy-plan target may retain the source prefix: {to}"
        );
        assert!(!from.is_empty(), "every copy-plan source must be non-empty");
    }

    // The latest version is the current snapshot id (FULL rewrite endVersion = current).
    assert_eq!(
        result.latest_version,
        table.metadata().current_snapshot_id().expect("snapshot")
    );
}

/// MUTATION-BAIT (the copy-plan direction by class): a DATA file copies FROM its ORIGINAL SOURCE
/// location; the MANIFEST-LIST / MANIFEST / POSITION-DELETE copy FROM the STAGING location; the
/// EQUALITY-delete copies FROM its ORIGINAL SOURCE location. Flipping any direction fails here.
#[tokio::test]
async fn copy_plan_direction_is_staged_vs_source_per_class() {
    let (catalog, file_io, _tmp) = local_fs_catalog().await;
    let table = create_table(&catalog).await;
    let location = table.metadata().location().to_string();

    let d1 = real_data_file(&file_io, &format!("{location}/data/d1.parquet"), b"d1").await;
    let table = append(&catalog, &table, vec![d1.clone()]).await;
    let pos_delete = write_real_pos_delete(
        &table,
        &file_io,
        &format!("{location}/data"),
        d1.file_path(),
    )
    .await;
    let eq_delete = real_eq_delete(&file_io, &format!("{location}/data/eq-deletes.parquet")).await;
    let table = row_delta(&catalog, &table, vec![
        pos_delete.clone(),
        eq_delete.clone(),
    ])
    .await;

    let source = location.clone();
    let target = "s3://bucket/relocated".to_string();
    let staging = format!("{location}-staging");

    let result = super::RewriteTablePath::new(table.clone())
        .rewrite_location_prefix(&source, &target)
        .staging_location(&staging)
        .execute(&file_io)
        .await
        .expect("execute rewrite table path");

    let staging_with_sep = format!("{staging}/");

    // The DATA file: its copy-plan entry copies FROM the ORIGINAL SOURCE location (verbatim).
    let data_target = new_path(d1.file_path(), &source, &target).expect("data target");
    let data_entry = result
        .copy_plan
        .iter()
        .find(|(_, to)| to == &data_target)
        .expect("a copy-plan entry for the data file");
    assert_eq!(
        data_entry.0,
        d1.file_path(),
        "a DATA file must copy FROM its ORIGINAL SOURCE location (verbatim), not from staging"
    );
    assert!(
        !data_entry.0.starts_with(&staging_with_sep),
        "a DATA file must NOT copy from staging"
    );

    // The EQUALITY-delete: copies FROM its ORIGINAL SOURCE location (verbatim).
    let eq_target = new_path(eq_delete.file_path(), &source, &target).expect("eq target");
    let eq_entry = result
        .copy_plan
        .iter()
        .find(|(_, to)| to == &eq_target)
        .expect("a copy-plan entry for the equality-delete");
    assert_eq!(
        eq_entry.0,
        eq_delete.file_path(),
        "an EQUALITY-delete must copy FROM its ORIGINAL SOURCE location (verbatim)"
    );

    // The POSITION-delete: copies FROM the STAGING location (content-rewritten).
    let pos_target = new_path(pos_delete.file_path(), &source, &target).expect("pos target");
    let pos_entry = result
        .copy_plan
        .iter()
        .find(|(_, to)| to == &pos_target)
        .expect("a copy-plan entry for the position-delete");
    assert!(
        pos_entry.0.starts_with(&staging_with_sep),
        "a POSITION-delete (content-rewritten) must copy FROM the STAGING location, got: {}",
        pos_entry.0
    );

    // At least one manifest-list + manifest entry copies FROM staging.
    let staged_entries = result
        .copy_plan
        .iter()
        .filter(|(from, _)| from.starts_with(&staging_with_sep))
        .count();
    assert!(
        staged_entries >= 2,
        "the manifest-list + manifests + position-delete are STAGED (copy from staging): got {staged_entries}"
    );
}

/// The position-delete CONTENT is physically rewritten into staging: the staged file re-reads and its
/// `file_path` column points at the TARGET (not the source). The ONLY content-rewritten payload.
#[tokio::test]
async fn position_delete_content_is_rewritten_into_staging() {
    use futures::StreamExt;

    use crate::arrow::delete_file_loader::BasicDeleteFileLoader;

    let (catalog, file_io, _tmp) = local_fs_catalog().await;
    let table = create_table(&catalog).await;
    let location = table.metadata().location().to_string();

    let d1 = real_data_file(&file_io, &format!("{location}/data/d1.parquet"), b"d1").await;
    let table = append(&catalog, &table, vec![d1.clone()]).await;
    let pos_delete = write_real_pos_delete(
        &table,
        &file_io,
        &format!("{location}/data"),
        d1.file_path(),
    )
    .await;
    let table = row_delta(&catalog, &table, vec![pos_delete.clone()]).await;

    let source = location.clone();
    let target = "s3://bucket/relocated".to_string();
    let staging = format!("{location}-staging");

    let result = super::RewriteTablePath::new(table.clone())
        .rewrite_location_prefix(&source, &target)
        .staging_location(&staging)
        .execute(&file_io)
        .await
        .expect("execute rewrite table path");

    // The staged pos-delete content lives under the staging location; its copy-plan "to" is the
    // pos-delete's target path. Read the staged content back and assert the file_path column now
    // points at the TARGET (the referenced data file was rewritten).
    let pos_target = new_path(pos_delete.file_path(), &source, &target).expect("pos target");
    let staging_with_sep = format!("{staging}/");
    let staged_pos_path = result
        .copy_plan
        .iter()
        .find(|(from, to)| to == &pos_target && from.starts_with(&staging_with_sep))
        .map(|(from, _)| from.clone())
        .expect("a staged pos-delete copy-plan entry");
    let loader = BasicDeleteFileLoader::new(file_io.clone());
    let size = file_io
        .new_input(&staged_pos_path)
        .expect("input")
        .metadata()
        .await
        .expect("metadata")
        .size;
    let mut stream = loader
        .parquet_to_batch_stream(&staged_pos_path, size)
        .await
        .expect("read staged pos-delete");
    let mut saw_rewritten_path = false;
    while let Some(batch) = stream.next().await {
        let batch = batch.expect("batch");
        let (path_col, _pos) =
            super::locate_reserved_columns(&batch, &staged_pos_path).expect("reserved columns");
        for row in 0..batch.num_rows() {
            let referenced = path_col.value(row);
            assert!(
                referenced.starts_with(&target),
                "the rewritten pos-delete's file_path column must point at the TARGET: {referenced}"
            );
            assert!(
                !referenced.starts_with(&format!("{source}/")),
                "the rewritten pos-delete's file_path must NOT retain the source prefix"
            );
            saw_rewritten_path = true;
        }
    }
    assert!(
        saw_rewritten_path,
        "the staged pos-delete file must carry at least one rewritten file_path record"
    );
}

/// IDEMPOTENCE / NO-DOUBLE-REWRITE proxy: re-running the action on the SAME source table produces the
/// SAME copy-plan (the plan is a pure function of the metadata graph + prefixes). And a SECOND prefix
/// applied to a graph already rewritten once would fail the source-prefix precondition (proving the
/// action does not silently re-rewrite an already-rewritten path).
#[tokio::test]
async fn execute_is_deterministic_and_rejects_double_rewrite() {
    let (catalog, file_io, _tmp) = local_fs_catalog().await;
    let table = create_table(&catalog).await;
    let location = table.metadata().location().to_string();
    let d1 = real_data_file(&file_io, &format!("{location}/data/d1.parquet"), b"d1").await;
    let table = append(&catalog, &table, vec![d1]).await;

    let source = location.clone();
    let target = "s3://bucket/relocated".to_string();
    let staging1 = format!("{location}-staging1");
    let staging2 = format!("{location}-staging2");

    let plan1 = super::RewriteTablePath::new(table.clone())
        .rewrite_location_prefix(&source, &target)
        .staging_location(&staging1)
        .execute(&file_io)
        .await
        .expect("first run")
        .copy_plan;
    let plan2 = super::RewriteTablePath::new(table.clone())
        .rewrite_location_prefix(&source, &target)
        .staging_location(&staging2)
        .execute(&file_io)
        .await
        .expect("second run")
        .copy_plan;

    // The (source, target) pairs are identical between runs (staging dirs differ, so we compare the
    // source-relative + target structure by stripping the staging prefix).
    let normalize = |plan: &[(String, String)], staging: &str| -> Vec<(String, String)> {
        let with_sep = format!("{staging}/");
        let mut v: Vec<(String, String)> = plan
            .iter()
            .map(|(from, to)| {
                let from_norm = from
                    .strip_prefix(&with_sep)
                    .map(|rest| format!("<STAGING>/{rest}"))
                    .unwrap_or_else(|| from.clone());
                (from_norm, to.clone())
            })
            .collect();
        v.sort();
        v
    };
    assert_eq!(
        normalize(&plan1, &staging1),
        normalize(&plan2, &staging2),
        "the copy-plan is a deterministic function of the graph + prefixes"
    );

    // A double-rewrite (applying the action expecting the WRONG source prefix) hard-fails: the data
    // files no longer start with a bogus source prefix.
    let err = super::RewriteTablePath::new(table.clone())
        .rewrite_location_prefix("s3://never/matches", &target)
        .staging_location(&staging1)
        .execute(&file_io)
        .await
        .expect_err("a source prefix that matches nothing must fail the precondition");
    assert!(
        err.to_string().contains("does not start with")
            || err.to_string().contains("not under the source prefix"),
        "expected a source-prefix precondition error, got: {err}"
    );
}

/// PRECONDITION: missing prefixes / staging location are typed errors (not a panic, not a silent
/// no-op).
#[tokio::test]
async fn execute_requires_prefixes_and_staging() {
    let (catalog, _file_io, _tmp) = local_fs_catalog().await;
    let table = create_table(&catalog).await;
    let file_io = FileIOBuilder::new(Arc::new(LocalFsStorageFactory)).build();

    // No prefixes set.
    let err = super::RewriteTablePath::new(table.clone())
        .staging_location("file:///stage")
        .execute(&file_io)
        .await
        .expect_err("missing prefixes must error");
    assert!(err.to_string().contains("prefixes must be set"));

    // No staging location set.
    let err = super::RewriteTablePath::new(table.clone())
        .rewrite_location_prefix("s3://a", "s3://b")
        .execute(&file_io)
        .await
        .expect_err("missing staging must error");
    assert!(err.to_string().contains("staging_location"));
}

// ---- replace_path_bounds (the file_path-column bound rewrite) ----------------------------------

/// `replace_path_bounds` rewrites the file_path-column bounds when lower == upper (single referenced
/// file), and CLEARS them when lower != upper (spans multiple files) — Java's `metricsWithoutPathBounds`.
#[test]
fn replace_path_bounds_rewrites_single_file_clears_multi() {
    use crate::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH as PATH_ID;

    let source = "s3://b/src";
    let target = "s3://b/tgt";

    // Single referenced file: lower == upper ⇒ both rewritten.
    let mut single = pos_delete_with_bounds(
        "s3://b/src/data/pos.parquet",
        Some("s3://b/src/data/d1.parquet"),
        Some("s3://b/src/data/d1.parquet"),
    );
    super::replace_path_bounds(&mut single, source, target).expect("replace bounds single");
    let lower = single.lower_bounds.get(&PATH_ID).expect("lower present");
    assert_eq!(
        super::datum_as_string(lower).as_deref(),
        Some("s3://b/tgt/data/d1.parquet"),
        "a single-file bound must be rewritten to the target"
    );

    // Multi-file: lower != upper ⇒ both CLEARED (Java metricsWithoutPathBounds).
    let mut multi = pos_delete_with_bounds(
        "s3://b/src/data/pos.parquet",
        Some("s3://b/src/data/d1.parquet"),
        Some("s3://b/src/data/d2.parquet"),
    );
    super::replace_path_bounds(&mut multi, source, target).expect("replace bounds multi");
    assert!(
        !multi.lower_bounds.contains_key(&PATH_ID) && !multi.upper_bounds.contains_key(&PATH_ID),
        "a multi-file bound must be CLEARED (metricsWithoutPathBounds)"
    );
}

// ============================================================================================
// Fixtures.
// ============================================================================================

async fn local_fs_catalog() -> (impl Catalog, FileIO, tempfile::TempDir) {
    let temp_dir = tempfile::TempDir::new().expect("temp dir");
    let warehouse = temp_dir
        .path()
        .to_str()
        .expect("utf8 temp path")
        .to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([("warehouse".to_string(), warehouse)]),
        )
        .await
        .expect("load local-fs memory catalog");
    let file_io = FileIOBuilder::new(Arc::new(LocalFsStorageFactory)).build();
    (catalog, file_io, temp_dir)
}

fn two_long_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "x",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                2,
                "y",
                Type::Primitive(PrimitiveType::Long),
            )),
        ])
        .build()
        .expect("build schema")
}

async fn create_table(catalog: &impl Catalog) -> Table {
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("t".to_string())
        .schema(two_long_schema())
        .partition_spec(PartitionSpec::unpartition_spec())
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

async fn write_real_file(file_io: &FileIO, path: &str, content: &[u8]) {
    file_io
        .new_output(path)
        .expect("new output")
        .write(bytes::Bytes::copy_from_slice(content))
        .await
        .expect("write file");
}

async fn real_data_file(file_io: &FileIO, path: &str, content: &[u8]) -> DataFile {
    write_real_file(file_io, path, content).await;
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(content.len() as u64)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .expect("build data file")
}

async fn real_eq_delete(file_io: &FileIO, path: &str) -> DataFile {
    write_real_file(file_io, path, b"eq").await;
    DataFileBuilder::default()
        .content(DataContentType::EqualityDeletes)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(2)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .equality_ids(Some(vec![2]))
        .build()
        .expect("build eq-delete file")
}

/// Write a REAL parquet position-delete file (via the same writer the action uses) deleting `pos 0` of
/// `referenced_path`, returning its [`DataFile`].
async fn write_real_pos_delete(
    table: &Table,
    _file_io: &FileIO,
    _data_dir: &str,
    referenced_path: &str,
) -> DataFile {
    let metadata = table.metadata();
    let config = PositionDeleteWriterConfig::new().expect("pos delete config");
    let location_gen = DefaultLocationGenerator::new(metadata.clone()).expect("location gen");
    let file_name_gen = DefaultFileNameGenerator::new(
        "pos-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        config.schema().clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(None)
        .await
        .expect("build pos delete writer");
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(vec![referenced_path])) as ArrayRef,
        Arc::new(Int64Array::from(vec![0_i64])) as ArrayRef,
    ])
    .expect("pos delete batch");
    writer.write(batch).await.expect("write pos delete");
    writer
        .close()
        .await
        .expect("close pos delete writer")
        .into_iter()
        .next()
        .expect("a pos delete file")
}

/// A POSITION-delete [`DataFile`] carrying file_path-column lower/upper bounds (for the
/// `replace_path_bounds` unit test).
fn pos_delete_with_bounds(path: &str, lower: Option<&str>, upper: Option<&str>) -> DataFile {
    use crate::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH as PATH_ID;

    let mut builder = DataFileBuilder::default();
    builder
        .content(DataContentType::PositionDeletes)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(4)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::empty());
    let mut lower_bounds = HashMap::new();
    let mut upper_bounds = HashMap::new();
    if let Some(l) = lower {
        lower_bounds.insert(PATH_ID, Datum::string(l));
    }
    if let Some(u) = upper {
        upper_bounds.insert(PATH_ID, Datum::string(u));
    }
    builder
        .lower_bounds(lower_bounds)
        .upper_bounds(upper_bounds);
    builder.build().expect("build pos delete with bounds")
}

async fn append(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(files)
        .apply(tx)
        .expect("apply fast append");
    tx.commit(catalog).await.expect("commit fast append")
}

async fn row_delta(catalog: &impl Catalog, table: &Table, deletes: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .row_delta()
        .add_deletes(deletes)
        .apply(tx)
        .expect("apply row delta");
    tx.commit(catalog).await.expect("commit row delta")
}
