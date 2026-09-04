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

/// Honest zeros: five Puffin DVs on V3. File-scoped, so nothing to convert and no commit.
#[tokio::test]
async fn test_v3_deletion_vectors_are_not_compacted() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V3).await;

    // FIVE data files in partition 0, each holding two rows; the DV below masks the SECOND.
    let mut data_paths = Vec::new();
    let mut files = Vec::new();
    for (index, y) in [10i64, 20, 30, 40, 50].into_iter().enumerate() {
        let file = write_data_file(&table, &format!("dv-target-{index}.parquet"), 0, &[
            (0, y, 1),
            (0, y + 1, 2),
        ])
        .await;
        data_paths.push(file.file_path().to_string());
        files.push(file);
    }
    let table = append_files(&catalog, &table, files).await;

    // FIVE Puffin DVs in the SAME partition 0 — one per data file, each masking its position 1.
    let mut table = table;
    for path in &data_paths {
        let dv = write_deletion_vector(&table, path, &[1]).await;
        table = add_deletes(&catalog, &table, vec![dv]).await;
    }

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 20, 30, 40, 50]),
        "before: each DV masks its data file's second row (y = 11, 21, 31, 41, 51)"
    );

    let action = || RewritePositionDeleteFiles::new(table.clone());
    let config = action().resolve_config().expect("the defaults are legal");
    let dvs = live_delete_files(&table).await;
    assert_eq!(dvs.len(), 5, "fixture: FIVE live deletion vectors");
    assert!(
        dvs.iter()
            .all(|f| f.file_format() == DataFileFormat::Puffin),
        "fixture: every live delete is a Puffin DV"
    );
    assert!(
        dvs.iter()
            .all(|f| f.content_type() == DataContentType::PositionDeletes),
        "fixture NON-VACUITY: a DV carries content PositionDeletes, so it clears the CONTENT \
         filter and reaches the FORMAT skip — this test would prove nothing if it were dropped one \
         line earlier"
    );
    assert!(
        dvs.iter()
            .all(|f| f.file_size_in_bytes < config.min_file_size_bytes),
        "fixture: every DV is SUB-MIN, so the mutated build makes all five candidates"
    );
    assert_eq!(
        config.min_input_files, 5,
        "fixture: FIVE files clear the DEFAULT floor with no knob — the literal, so a moved \
         constant reds here rather than silently re-shaping the fixture"
    );

    let snapshot_before = table.metadata().current_snapshot_id();
    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "DVs are NOT compacted by this action — zero counts, no commit, even at five same-partition DVs"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "post-execute SHAPE: a skipped group must NOT commit a new snapshot"
    );
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity: the DVs are untouched, the live set unchanged"
    );
    // All five Puffin DVs are still live, and none became a parquet pos-delete.
    let deletes = live_delete_files(&reloaded).await;
    assert_eq!(deletes.len(), 5, "all five DVs remain live");
    assert!(
        deletes
            .iter()
            .all(|f| f.file_format() == DataFileFormat::Puffin),
        "every surviving delete is a Puffin DV (none was compacted into a parquet pos-delete)"
    );
}

/// Write a single-data-file Puffin DELETION VECTOR masking the given absolute positions of `target_path`, in partition x=0.
async fn write_deletion_vector(table: &Table, target_path: &str, positions: &[u64]) -> DataFile {
    use crate::writer::base_writer::deletion_vector_writer::DVFileWriter;

    let dv_path = format!(
        "{}/data/dv-{}.puffin",
        table.metadata().location(),
        uuid::Uuid::now_v7()
    );
    let output = table.file_io().new_output(&dv_path).unwrap();
    let partition_key = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        Struct::from_iter([Some(Literal::long(0))]),
    )
    .expect("PartitionKey::new: valid partition tuple");
    let mut writer = DVFileWriter::new(output).unpartitioned();
    for &pos in positions {
        writer
            .delete(target_path, pos, Some(&partition_key))
            .expect("record DV position");
    }
    writer
        .close()
        .await
        .expect("close DV writer")
        .into_iter()
        .next()
        .expect("one DV delete file")
}

#[tokio::test]
async fn test_v3_two_parquet_deletes_below_the_five_floor_stay_parquet() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 7, &[
        (7, 10, 100),
        (7, 20, 200),
        (7, 30, 300),
        (7, 40, 400),
        (7, 50, 500),
    ])
    .await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await;

    let pd1 = write_position_delete_file(&table, Some(7), &[(&x_path, 1)]).await;
    let table = add_deletes(&catalog, &table, vec![pd1]).await;
    let pd2 = write_position_delete_file(&table, Some(7), &[(&x_path, 3)]).await;
    let table = add_deletes(&catalog, &table, vec![pd2]).await;
    let table = upgrade_to_v3(&catalog, &table).await;

    let before = scan_y_values(&table).await;
    assert_eq!(before, HashSet::from([10, 30, 50]));
    let snapshot_before = table.metadata().current_snapshot_id();

    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "two files are below Java's min-input-files floor of five: four zeros"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "a declined group must NOT commit"
    );
    let after = live_delete_files(&reloaded).await;
    assert_eq!(after.len(), 2, "both parquet deletes stay live");
    assert!(
        after
            .iter()
            .all(|f| f.file_format() == DataFileFormat::Parquet),
        "no deletion vector is written below the floor"
    );
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

#[tokio::test]
async fn test_v3_one_parquet_delete_below_the_five_floor_stays_parquet() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 7, &[(7, 10, 1), (7, 20, 2)]).await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await;
    let pd = write_position_delete_file(&table, Some(7), &[(&x_path, 1)]).await;
    let table = add_deletes(&catalog, &table, vec![pd]).await;
    let table = upgrade_to_v3(&catalog, &table).await;

    let before = scan_y_values(&table).await;
    assert_eq!(before, HashSet::from([10]));
    let snapshot_before = table.metadata().current_snapshot_id();

    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "one file is below Java's min-input-files floor of five: four zeros"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "a declined group must NOT commit"
    );
    let after = live_delete_files(&reloaded).await;
    assert_eq!(after.len(), 1, "the parquet delete stays live");
    assert_eq!(after[0].file_format(), DataFileFormat::Parquet);
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

#[tokio::test]
async fn test_v3_one_partition_scoped_delete_covering_two_files_stays_parquet() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let a = write_data_file(&table, "a.parquet", 7, &[(7, 10, 1), (7, 11, 2)]).await;
    let b = write_data_file(&table, "b.parquet", 7, &[(7, 20, 1), (7, 21, 2)]).await;
    let a_path = a.file_path().to_string();
    let b_path = b.file_path().to_string();
    let table = append_files(&catalog, &table, vec![a, b]).await;

    let pd = write_position_delete_file(&table, Some(7), &[
        (a_path.as_str(), 1),
        (b_path.as_str(), 1),
    ])
    .await;
    assert!(
        referenced_data_file_location(&pd).is_none(),
        "fixture: a delete naming two data files is PARTITION-scoped"
    );
    let table = add_deletes(&catalog, &table, vec![pd]).await;
    let table = upgrade_to_v3(&catalog, &table).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 20]),
        "fixture: the partition-scoped delete masks one row per data file"
    );
    let snapshot_before = table.metadata().current_snapshot_id();

    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "one partition-scoped delete is one file below the floor even though it covers two data files: four zeros"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "a declined group must NOT commit"
    );
    let after = live_delete_files(&reloaded).await;
    assert_eq!(after.len(), 1, "the parquet delete stays live");
    assert_eq!(after[0].file_format(), DataFileFormat::Parquet);
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

#[tokio::test]
async fn test_v3_five_file_scoped_deletes_at_the_floor_convert_to_one_dv_per_data_file() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let mut paths = Vec::new();
    let mut data_files = Vec::new();
    for index in 0..5 {
        let data = write_data_file(&table, &format!("d{index}.parquet"), 7, &[
            (7, 10 + index * 10, 1),
            (7, 11 + index * 10, 2),
        ])
        .await;
        paths.push(data.file_path().to_string());
        data_files.push(data);
    }
    let table = append_files(&catalog, &table, data_files).await;

    let mut deletes = Vec::new();
    for path in &paths {
        deletes.push(write_file_scoped_position_delete_file(&table, 7, path, &[1]).await);
    }
    assert!(
        deletes
            .iter()
            .all(|f| referenced_data_file_location(f).is_some()),
        "fixture: every delete is FILE-scoped"
    );
    let table = add_deletes(&catalog, &table, deletes).await;
    let table = upgrade_to_v3(&catalog, &table).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before.len(),
        5,
        "fixture: each delete masks one of two rows"
    );

    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(
        result.rewritten_delete_files_count, 5,
        "five files meet the floor: the whole group converts"
    );
    assert_eq!(
        result.added_delete_files_count, 5,
        "one deletion vector per referenced data file"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
    let after = live_delete_files(&reloaded).await;
    assert_eq!(
        after
            .iter()
            .filter(|f| f.file_format() == DataFileFormat::Parquet)
            .count(),
        0,
        "no parquet position delete survives"
    );
    assert_eq!(
        after
            .iter()
            .filter(|f| f.file_format() == DataFileFormat::Puffin)
            .count(),
        5,
        "one Puffin deletion vector per data file"
    );

    let second = RewritePositionDeleteFiles::new(reloaded.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(
        second,
        RewritePositionDeleteFilesResult::default(),
        "second run: nothing legacy left, honest zeros"
    );
}

#[tokio::test]
async fn test_v3_rewrite_all_converts_a_lone_parquet_delete_below_the_floor() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 7, &[(7, 10, 1), (7, 20, 2)]).await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await;
    let pd = write_position_delete_file(&table, Some(7), &[(&x_path, 1)]).await;
    let table = add_deletes(&catalog, &table, vec![pd]).await;
    let table = upgrade_to_v3(&catalog, &table).await;

    let before = scan_y_values(&table).await;

    let result = RewritePositionDeleteFiles::new(table.clone())
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(result.rewritten_delete_files_count, 1);
    assert_eq!(result.added_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
    let after = live_delete_files(&reloaded).await;
    assert_eq!(
        after.len(),
        1,
        "one deletion vector replaces the lone delete"
    );
    assert_eq!(after[0].file_format(), DataFileFormat::Puffin);
}

#[tokio::test]
async fn test_v3_refuses_when_an_admitted_vector_would_shadow_a_gate_declined_delete() {
    let (catalog, temp) = local_fs_catalog().await;
    let table = create_short_path_partitioned_table(&catalog, temp.path(), FormatVersion::V2).await;

    let a = write_data_file(&table, "a.parquet", 0, &[
        (0, 10, 1),
        (0, 11, 2),
        (0, 12, 3),
        (0, 13, 4),
        (0, 14, 5),
        (0, 15, 6),
        (0, 16, 7),
    ])
    .await;
    let a_path = a.file_path().to_string();
    let table = append_files(&catalog, &table, vec![a]).await;

    let mut admitted = Vec::new();
    for position in 0..5 {
        admitted
            .push(write_file_scoped_position_delete_file(&table, 1, &a_path, &[position]).await);
    }
    let declined = write_position_delete_file(&table, Some(0), &[(&a_path, 5)]).await;
    assert!(
        referenced_data_file_location(&declined).is_none(),
        "fixture: the declined delete is PARTITION-scoped"
    );
    let table = add_deletes(&catalog, &table, vec![declined]).await;
    let table = add_deletes(&catalog, &table, admitted).await;
    let table = upgrade_to_v3(&catalog, &table).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([16]),
        "fixture: all six deletes apply to the same data file by different routes"
    );

    let error = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .expect_err("a gate-declined delete shadowed by an admitted vector must fail closed");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error.to_string().contains("size gate"),
        "the refusal names the gate, not the filter: {error}"
    );
    assert!(
        error.to_string().contains("rewrite-all"),
        "the refusal names the bypass: {error}"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "fail CLOSED: nothing committed, so no row came back"
    );
}

#[tokio::test]
async fn test_admission_rewrite_all_admits_lone_sub_min_file() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    let pd = write_sized_pos_delete(&table, &x_path, 1, 1).await;
    let size = pd.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd]).await;

    let action = || {
        RewritePositionDeleteFiles::new(table.clone())
            .min_input_files(1)
            .rewrite_all(true)
    };
    let config = action().resolve_config().expect("legal knobs");
    assert!(
        size < config.min_file_size_bytes,
        "fixture: the lone file is SUB-MIN (measured {size}, resolved min {})",
        config.min_file_size_bytes
    );
    assert!(config.rewrite_all, "fixture: the bypass is on");

    let before = scan_y_values(&table).await;
    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result.rewritten_delete_files_count, 1,
        "rewrite-all bypasses the candidate filter and the group gate on the bin-pack arm too"
    );
    assert_eq!(result.added_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}
