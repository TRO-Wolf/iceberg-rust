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

use super::*;

async fn oracle_shape_fixture() -> (impl Catalog, TempDir, Table, Vec<String>, HashSet<i64>) {
    let (catalog, temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let mut data_paths = Vec::new();
    let mut data_files = Vec::new();
    for p in 0..2i64 {
        for f in 0..4i64 {
            let base = p * 200 + f * 50;
            let rows: Vec<(i64, i64, i64)> = (0..50)
                .map(|i| (p, base + i, base + i))
                .collect();
            let file = write_data_file(&table, &format!("p{p}-f{f}.parquet"), p, &rows).await;
            data_paths.push(file.file_path().to_string());
            data_files.push(file);
        }
    }
    let table = append_files(&catalog, &table, data_files).await;

    let mut deletes = Vec::new();
    for (index, path) in data_paths.iter().enumerate() {
        let part = (index / 4) as i64;
        let positions: Vec<i64> = (0..50).step_by(2).collect();
        deletes.push(write_file_scoped_position_delete_file(&table, part, path, &positions).await);
    }
    assert_eq!(deletes.len(), 8);
    assert!(
        deletes
            .iter()
            .all(|f| referenced_data_file_location(f).is_some()),
        "fixture: every input delete is FILE-scoped, as the oracle's are"
    );
    let table = add_deletes(&catalog, &table, deletes).await;

    let live_y: HashSet<i64> = (0..400i64).filter(|id| id % 2 == 1).collect();
    assert_eq!(live_y.len(), 200);
    assert_eq!(
        scan_y_values(&table).await,
        live_y,
        "fixture: the 25 even positions of each of the 8 files are masked"
    );
    (catalog, temp, table, data_paths, live_y)
}

async fn live_delete_entry_seqs(table: &Table) -> Vec<(DataFile, Option<i64>, Option<i64>)> {
    let snapshot = table.metadata().current_snapshot().unwrap();
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .unwrap();
    let mut out = Vec::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
        for entry in manifest.entries() {
            if entry.is_alive() {
                out.push((
                    entry.data_file().clone(),
                    entry.sequence_number(),
                    entry.file_sequence_number,
                ));
            }
        }
    }
    out
}

async fn assert_oracle_outputs(
    table: &Table,
    data_paths: &[String],
    live_y: &HashSet<i64>,
    expected_data_seq: i64,
    new_snapshot: &Snapshot,
) {
    let entries = live_delete_entry_seqs(table).await;
    assert_eq!(
        entries.len(),
        8,
        "eight live output delete files, one per data file"
    );
    let mut referenced = HashSet::new();
    for (file, data_seq, file_seq) in &entries {
        assert_eq!(
            *data_seq,
            Some(expected_data_seq),
            "output data sequence number preserved from the input deletes"
        );
        assert_eq!(
            *file_seq,
            Some(new_snapshot.sequence_number()),
            "output file sequence number is the rewrite snapshot's"
        );
        let path = referenced_data_file_location(file)
            .expect("every output is FILE-scoped (references exactly one data file)");
        referenced.insert(path);
        let pairs = read_pos_delete_pairs(table, file).await;
        assert_eq!(pairs.len(), 25, "each output holds 25 positions");
    }
    assert_eq!(referenced.len(), 8, "the 8 outputs cover 8 distinct data files");
    for path in data_paths {
        assert!(
            referenced.contains(path),
            "output set covers data file {path}"
        );
    }
    assert_eq!(
        scan_y_values(table).await,
        *live_y,
        "read identity: the same 200 rows stay live"
    );
}

async fn assert_single_commit_cell(
    catalog: &impl Catalog,
    table: &Table,
    data_paths: &[String],
    live_y: &HashSet<i64>,
    action: RewritePositionDeleteFiles,
) {
    let input_entries = live_delete_entries_with_seq(table).await;
    assert_eq!(input_entries.len(), 8);
    let input_seq = input_entries[0].1.expect("input deletes carry a seq");
    assert!(
        input_entries.iter().all(|(_, seq)| *seq == Some(input_seq)),
        "fixture: all eight inputs share one data sequence number"
    );
    let input_bytes: u64 = input_entries
        .iter()
        .map(|(f, _)| f.file_size_in_bytes)
        .sum();
    let history_before = table.metadata().history().len();

    let result = action.execute(catalog).await.unwrap();
    assert_eq!(result.rewritten_delete_files_count, 8);
    assert_eq!(result.added_delete_files_count, 8);
    assert_eq!(
        result.rewritten_bytes_count, input_bytes,
        "rewritten bytes are the INPUT delete files' sizes"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    let new_snapshots = snapshots_after(&reloaded, history_before);
    assert_eq!(
        new_snapshots.len(),
        1,
        "exactly ONE new snapshot for the whole rewrite"
    );
    assert_eq!(
        new_snapshots[0].summary().operation,
        Operation::Replace,
        "the single commit is a replace snapshot"
    );

    let outputs = live_delete_files(&reloaded).await;
    let added_bytes: u64 = outputs.iter().map(|f| f.file_size_in_bytes).sum();
    assert_eq!(
        result.added_bytes_count, added_bytes,
        "added bytes are the OUTPUT delete files' sizes"
    );
    assert!(result.added_bytes_count > 0);

    assert_oracle_outputs(
        &reloaded,
        data_paths,
        live_y,
        input_seq,
        &new_snapshots[0],
    )
    .await;
}

#[tokio::test]
async fn test_rewrite_all_commits_once_and_keeps_file_scope() {
    let (catalog, _temp, table, data_paths, live_y) = oracle_shape_fixture().await;
    assert_single_commit_cell(
        &catalog,
        &table,
        &data_paths,
        &live_y,
        RewritePositionDeleteFiles::new(table.clone()).rewrite_all(true),
    )
    .await;
}

#[tokio::test]
async fn test_min_input_files_1_commits_once_and_keeps_file_scope() {
    let (catalog, _temp, table, data_paths, live_y) = oracle_shape_fixture().await;
    assert_single_commit_cell(
        &catalog,
        &table,
        &data_paths,
        &live_y,
        RewritePositionDeleteFiles::new(table.clone()).min_input_files(1),
    )
    .await;
}

#[tokio::test]
async fn test_baseline_declines_every_bin_and_commits_nothing() {
    let (catalog, _temp, table, _data_paths, live_y) = oracle_shape_fixture().await;
    let history_before = table.metadata().history().len();

    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "four files per partition are below the default floor of five"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert!(
        snapshots_after(&reloaded, history_before).is_empty(),
        "baseline: no new snapshot"
    );
    assert_eq!(live_delete_files(&reloaded).await.len(), 8);
    assert_eq!(scan_y_values(&reloaded).await, live_y);
}

#[tokio::test]
async fn test_partial_progress_commits_one_batch_per_commit() {
    let (catalog, _temp, table, data_paths, live_y) = oracle_shape_fixture().await;
    let input_seq = live_delete_entries_with_seq(&table).await[0]
        .1
        .expect("input seq");
    let history_before = table.metadata().history().len();

    let result = RewritePositionDeleteFiles::new(table.clone())
        .rewrite_all(true)
        .partial_progress(true)
        .partial_progress_max_commits(2)
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(result.rewritten_delete_files_count, 8);
    assert_eq!(result.added_delete_files_count, 8);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    let new_snapshots = snapshots_after(&reloaded, history_before);
    assert_eq!(
        new_snapshots.len(),
        2,
        "two bins at ceil(2/2) = 1 bin per commit gives TWO replace snapshots"
    );
    for snapshot in &new_snapshots {
        assert_eq!(snapshot.summary().operation, Operation::Replace);
        assert_eq!(
            delete_file_counters(snapshot),
            (Some("4".to_string()), Some("4".to_string())),
            "each batch replaces its own bin: 4 deletes out, 4 file-scoped outputs in"
        );
    }
    assert_oracle_outputs(&reloaded, &data_paths, &live_y, input_seq, &new_snapshots[1]).await;
}

#[tokio::test]
async fn test_partial_progress_max_commits_1_batches_all_bins_into_one_commit() {
    let (catalog, _temp, table, data_paths, live_y) = oracle_shape_fixture().await;
    let history_before = table.metadata().history().len();

    let result = RewritePositionDeleteFiles::new(table.clone())
        .rewrite_all(true)
        .partial_progress(true)
        .partial_progress_max_commits(1)
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(result.rewritten_delete_files_count, 8);
    assert_eq!(result.added_delete_files_count, 8);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    let new_snapshots = snapshots_after(&reloaded, history_before);
    assert_eq!(
        new_snapshots.len(),
        1,
        "ceil(2/1) = 2 bins per commit folds the whole rewrite into one snapshot"
    );
    assert_eq!(new_snapshots[0].summary().operation, Operation::Replace);
    assert_eq!(
        scan_y_values(&reloaded).await,
        live_y,
        "read identity"
    );
    assert_eq!(data_paths.len(), 8);
}

#[tokio::test]
async fn test_partial_progress_max_commits_zero_is_rejected() {
    let (catalog, _temp, table, _data_paths, _live_y) = oracle_shape_fixture().await;
    let error = RewritePositionDeleteFiles::new(table.clone())
        .partial_progress(true)
        .partial_progress_max_commits(0)
        .execute(&catalog)
        .await
        .expect_err("max-commits must be positive when partial progress is enabled");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
}

#[tokio::test]
async fn test_dangling_positions_are_dropped_not_rewritten() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let a = write_data_file(&table, "a.parquet", 7, &[
        (7, 10, 1),
        (7, 20, 2),
        (7, 30, 3),
        (7, 40, 4),
        (7, 50, 5),
        (7, 60, 6),
        (7, 70, 7),
    ])
    .await;
    let a_path = a.file_path().to_string();
    let table = append_files(&catalog, &table, vec![a]).await;

    let dead_path = format!("{}/data/dead.parquet", table.metadata().location());
    let pd_live = write_file_scoped_position_delete_file(&table, 7, &a_path, &[1, 3]).await;
    let pd_dead = write_file_scoped_position_delete_file(&table, 7, &dead_path, &[0, 5]).await;
    let pd_mixed = write_position_delete_file(&table, Some(7), &[
        (a_path.as_str(), 5),
        (dead_path.as_str(), 2),
    ])
    .await;
    let table = add_deletes(&catalog, &table, vec![pd_live, pd_dead, pd_mixed]).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 30, 50, 70]),
        "fixture: positions 1, 3, 5 of the live file are masked; the dead path masks nothing"
    );

    let result = RewritePositionDeleteFiles::new(table.clone())
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(
        result.rewritten_delete_files_count, 3,
        "all three input deletes are rewritten away, dead-path ones included"
    );
    assert_eq!(
        result.added_delete_files_count, 1,
        "only the live data file earns an output delete file"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    let outputs = live_delete_files(&reloaded).await;
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        referenced_data_file_location(&outputs[0]),
        Some(a_path.clone()),
        "the one output is file-scoped on the live data file"
    );
    let pairs = read_pos_delete_pairs(&reloaded, &outputs[0]).await;
    let positions: HashSet<i64> = pairs.iter().map(|(_, pos)| *pos).collect();
    assert_eq!(
        positions,
        HashSet::from([1, 3, 5]),
        "positions for the dead data file are dropped, never rewritten"
    );
    assert!(
        pairs.iter().all(|(path, _)| path == &a_path),
        "no output row names the dead path"
    );
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity"
    );
}

async fn create_partitioned_table_with_props(
    catalog: &impl Catalog,
    properties: &[(&str, &str)],
) -> Table {
    let schema = three_long_schema();
    let spec = PartitionSpec::builder(schema.clone())
        .add_partition_field("x", "x", Transform::Identity)
        .expect("add partition field")
        .build()
        .expect("build spec");
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, std::collections::HashMap::new())
        .await
        .expect("create namespace");
    let mut props: Vec<(String, String)> = vec![(
        "write.parquet.compression-codec".to_string(),
        "uncompressed".to_string(),
    )];
    props.extend(
        properties
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string())),
    );
    let creation = TableCreation::builder()
        .name("t".to_string())
        .schema(schema)
        .partition_spec(spec)
        .format_version(FormatVersion::V2)
        .properties(props)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

#[tokio::test]
async fn test_partition_granularity_writes_partition_scoped_outputs_in_one_commit() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table_with_props(&catalog, &[(
        "write.delete.granularity",
        "partition",
    )])
    .await;

    let mut data_paths = Vec::new();
    let mut data_files = Vec::new();
    for p in 0..2i64 {
        for f in 0..4i64 {
            let base = p * 200 + f * 50;
            let rows: Vec<(i64, i64, i64)> = (0..50)
                .map(|i| (p, base + i, base + i))
                .collect();
            let file = write_data_file(&table, &format!("p{p}-f{f}.parquet"), p, &rows).await;
            data_paths.push(file.file_path().to_string());
            data_files.push(file);
        }
    }
    let table = append_files(&catalog, &table, data_files).await;
    let mut deletes = Vec::new();
    for (index, path) in data_paths.iter().enumerate() {
        let part = (index / 4) as i64;
        let positions: Vec<i64> = (0..50).step_by(2).collect();
        deletes.push(write_file_scoped_position_delete_file(&table, part, path, &positions).await);
    }
    let table = add_deletes(&catalog, &table, deletes).await;
    let live_y: HashSet<i64> = (0..400i64).filter(|id| id % 2 == 1).collect();
    let history_before = table.metadata().history().len();

    let result = RewritePositionDeleteFiles::new(table.clone())
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(result.rewritten_delete_files_count, 8);
    assert_eq!(
        result.added_delete_files_count, 2,
        "partition granularity: one output per partition"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    let new_snapshots = snapshots_after(&reloaded, history_before);
    assert_eq!(new_snapshots.len(), 1, "still ONE replace snapshot");
    let outputs = live_delete_files(&reloaded).await;
    assert_eq!(outputs.len(), 2);
    for file in &outputs {
        assert!(
            referenced_data_file_location(file).is_none(),
            "a partition-granularity output is PARTITION-scoped"
        );
        let pairs = read_pos_delete_pairs(&reloaded, file).await;
        assert_eq!(pairs.len(), 100, "each output holds the partition's 100 positions");
    }
    assert_eq!(scan_y_values(&reloaded).await, live_y, "read identity");
}
