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

async fn rewrite_and_reload(
    catalog: &impl Catalog,
    action: RewriteDataFiles,
    ident_table: &Table,
) -> Table {
    action.execute(catalog).await.expect("rewrite data files");
    catalog
        .load_table(ident_table.identifier())
        .await
        .expect("reload table")
}

#[tokio::test]
async fn rdf_replace_summary_counts_added_files_and_live_totals() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = seed_mor_table(&catalog, FormatVersion::V2).await;

    let table = rewrite_and_reload(&catalog, RewriteDataFiles::new(table.clone()), &table).await;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .clone();

    assert_replace_totals_match_live_files(&table, &snapshot).await;
    let props = props_of(&snapshot);
    assert_rdf_key_set(&props, true, "rdf default");
    assert_eq!(
        prop_u64(&props, "removed-delete-files"),
        2,
        "the seq-GC expired two of the three accumulated position deletes"
    );
    assert_manifest_counts(
        &props,
        (7, 1, 6),
        "rdf default (oracle merges_rdf_only shape)",
    );
}

#[tokio::test]
async fn rdf_rewrite_all_replace_summary_matches_java_keys() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = seed_mor_table(&catalog, FormatVersion::V2).await;

    let action = RewriteDataFiles::new(table.clone()).rewrite_all(true);
    let table = rewrite_and_reload(&catalog, action, &table).await;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .clone();

    assert_replace_totals_match_live_files(&table, &snapshot).await;
    assert_rdf_key_set(&props_of(&snapshot), true, "rdf rewrite-all");
}

#[tokio::test]
async fn rdf_fresh_sequence_numbers_replace_summary_matches_java_keys() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = seed_mor_table(&catalog, FormatVersion::V2).await;

    let action = RewriteDataFiles::new(table.clone()).use_starting_sequence_number(false);
    let table = rewrite_and_reload(&catalog, action, &table).await;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .clone();

    assert_replace_totals_match_live_files(&table, &snapshot).await;
    assert_rdf_key_set(&props_of(&snapshot), true, "rdf fresh-seq");
}

#[tokio::test]
async fn rdf_partial_progress_chains_totals_across_commits() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let mut data_files = Vec::new();
    for index in 0..4i64 {
        data_files.push(
            write_data_file(
                &table,
                &format!("d0-{index}.parquet"),
                0,
                &rows(0, index * 250, 250),
            )
            .await,
        );
    }
    for index in 0..3i64 {
        data_files.push(
            write_data_file(
                &table,
                &format!("d1-{index}.parquet"),
                1,
                &rows(1, index * 250, 250),
            )
            .await,
        );
    }
    let path_x0 = data_files[0].file_path().to_string();
    let path_x1 = data_files[4].file_path().to_string();
    let table = append_files(&catalog, &table, data_files).await;

    let pd_x1 = write_position_delete_file(&table, 1, &[(path_x1, 0)]).await;
    let table = add_deletes(&catalog, &table, vec![pd_x1]).await;
    let pd_x0 = write_position_delete_file(&table, 0, &[(path_x0, 0)]).await;
    let table = add_deletes(&catalog, &table, vec![pd_x0]).await;

    RewriteDataFiles::new(table.clone())
        .rewrite_all(true)
        .partial_progress(true)
        .partial_progress_max_commits(2)
        .rewrite_job_order(RewriteJobOrder::FilesDesc)
        .execute(&catalog)
        .await
        .expect("rewrite data files");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");

    let last = table
        .metadata()
        .current_snapshot()
        .expect("head snapshot")
        .as_ref()
        .clone();
    let batch_one = table
        .metadata()
        .snapshot_by_id(last.parent_snapshot_id().expect("batch-2 parent id"))
        .expect("batch-1 snapshot")
        .as_ref()
        .clone();
    let seed_head = table
        .metadata()
        .snapshot_by_id(batch_one.parent_snapshot_id().expect("batch-1 parent id"))
        .expect("seed head snapshot");
    assert_eq!(
        seed_head.summary().operation,
        Operation::Delete,
        "exactly two partial-progress commits follow the seeded delete commit"
    );

    assert_eq!(batch_one.summary().operation, Operation::Replace);
    assert_replace_totals_match_live_files(&table, &batch_one).await;
    assert_rdf_key_set(&props_of(&batch_one), false, "rdf partial batch-1");

    assert_replace_totals_match_live_files(&table, &last).await;
    assert_rdf_key_set(&props_of(&last), true, "rdf partial batch-2");
    assert_eq!(
        prop_u64(&props_of(&last), "removed-position-delete-files"),
        1,
        "batch-2 expires the seq-2 position delete; the seq-3 delete survives because \
         batch-1's compacted file carries the starting sequence number"
    );
    assert_eq!(
        prop_u64(&props_of(&last), "removed-delete-files"),
        1,
        "exactly one delete file expires in batch-2"
    );

    let before = props_of(&batch_one);
    let after = props_of(&last);
    for (total, added, removed) in [
        ("total-data-files", "added-data-files", "deleted-data-files"),
        (
            "total-delete-files",
            "added-delete-files",
            "removed-delete-files",
        ),
        ("total-records", "added-records", "deleted-records"),
        ("total-files-size", "added-files-size", "removed-files-size"),
        (
            "total-position-deletes",
            "added-position-deletes",
            "removed-position-deletes",
        ),
        (
            "total-equality-deletes",
            "added-equality-deletes",
            "removed-equality-deletes",
        ),
    ] {
        assert_eq!(
            prop_or_zero(&after, total),
            prop_or_zero(&before, total) + prop_or_zero(&after, added)
                - prop_or_zero(&after, removed),
            "chain {total} = parent + added - removed through batch-2"
        );
    }
}

#[tokio::test]
async fn rdf_bucketed_table_replace_summary_matches_java_keys() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_bucketed_table(&catalog, FormatVersion::V2).await;

    let mut data_files = Vec::new();
    for index in 0..4i64 {
        data_files.push(
            write_current_spec_file(
                &table,
                &format!("bdata-{index}"),
                &rows(PARTITION, index * 250, 250),
            )
            .await,
        );
    }
    let deleted_path = data_files[0].file_path().to_string();
    let table = append_files(&catalog, &table, data_files).await;

    let bucket_partition = Struct::from_iter([Some(literal_from_long_transform(
        Transform::Bucket(8),
        PARTITION,
    ))]);
    let mut table = table;
    for merge in 0..3i64 {
        let merge_file = write_current_spec_file(
            &table,
            &format!("bmerge-{merge}"),
            &rows(PARTITION, 1000 + merge * 200, 200),
        )
        .await;
        let delete_file =
            write_pos_del_in_partition(&table, &bucket_partition, &[(deleted_path.clone(), merge)])
                .await;
        table = merge_commit(&catalog, &table, vec![merge_file], vec![delete_file]).await;
    }

    let action = RewriteDataFiles::new(table.clone()).rewrite_all(true);
    let table = rewrite_and_reload(&catalog, action, &table).await;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .clone();

    assert_replace_totals_match_live_files(&table, &snapshot).await;
    assert_rdf_key_set(&props_of(&snapshot), true, "rdf bucketed");
    assert_eq!(
        prop_u64(&props_of(&snapshot), "changed-partition-count"),
        1,
        "one bucket partition was rewritten"
    );
}

async fn write_pos_del_in_partition(
    table: &Table,
    partition: &Struct,
    deletes: &[(String, i64)],
) -> DataFile {
    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};

    use crate::spec::PartitionKey;
    use crate::writer::base_writer::position_delete_writer::{
        PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
    };
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
    use crate::writer::{IcebergWriter, IcebergWriterBuilder};

    let config = PositionDeleteWriterConfig::new().expect("pos-del config");
    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location gen");
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
    let partition_key = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        partition.clone(),
    )
    .expect("partition key");
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(Some(partition_key))
        .await
        .expect("pos-del writer");

    let paths: Vec<&str> = deletes.iter().map(|(path, _)| path.as_str()).collect();
    let positions: Vec<i64> = deletes.iter().map(|(_, pos)| *pos).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions)) as ArrayRef,
    ])
    .expect("pos-del batch");
    writer.write(batch).await.expect("write pos-del");
    writer
        .close()
        .await
        .expect("close pos-del")
        .into_iter()
        .next()
        .expect("pos-del file")
}

#[tokio::test]
async fn rdf_v3_dv_replace_summary_reports_removed_dvs() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V3).await;

    let mut data_files = Vec::new();
    for index in 0..4i64 {
        data_files.push(
            write_data_file(
                &table,
                &format!("dvdata-{index}.parquet"),
                PARTITION,
                &rows(PARTITION, index * 250, 250),
            )
            .await,
        );
    }
    let positions: Vec<u64> = (0..25).collect();
    let paths: Vec<String> = data_files
        .iter()
        .map(|file| file.file_path().to_string())
        .collect();
    let targets: Vec<(&str, &[u64])> = paths
        .iter()
        .map(|path| (path.as_str(), positions.as_slice()))
        .collect();
    let table = append_files(&catalog, &table, data_files).await;
    let dvs = write_dv(&table, PARTITION, &targets).await;
    let table = add_deletes(&catalog, &table, dvs).await;

    let action = RewriteDataFiles::new(table.clone()).rewrite_all(true);
    let table = rewrite_and_reload(&catalog, action, &table).await;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .clone();
    let props = props_of(&snapshot);

    assert_replace_totals_match_live_files(&table, &snapshot).await;
    let mut required = rdf_required_keys(false);
    required.extend([
        "removed-dvs",
        "removed-delete-files",
        "removed-position-deletes",
    ]);
    assert_summary_keys(
        &props,
        &required,
        &[
            "removed-position-delete-files",
            "added-delete-files",
            "added-dvs",
            "added-position-delete-files",
            "added-position-deletes",
            "added-equality-delete-files",
            "added-equality-deletes",
            "removed-equality-delete-files",
            "removed-equality-deletes",
            "entries-processed",
        ],
        "rdf v3 dv",
    );
    assert_eq!(prop_u64(&props, "removed-dvs"), 4, "four DV blobs dropped");
    assert_eq!(
        prop_u64(&props, "removed-position-deletes"),
        100,
        "100 positions were deleted across the four DVs"
    );
    assert_eq!(prop_u64(&props, "total-records"), 900);
    assert_eq!(prop_u64(&props, "total-delete-files"), 0);
    assert_eq!(prop_u64(&props, "total-position-deletes"), 0);
    assert_manifest_counts(&props, (3, 0, 2), "rdf v3 dv (oracle v3_dv_rdf)");
}

#[tokio::test]
async fn rdf_v3_rewrite_all_removes_equality_and_dv_delete_files() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V3).await;

    let mut data_files = Vec::new();
    for index in 0..4i64 {
        data_files.push(
            write_data_file(
                &table,
                &format!("edata-{index}.parquet"),
                PARTITION,
                &rows(PARTITION, index * 250, 250),
            )
            .await,
        );
    }
    let positions: Vec<u64> = (0..25).collect();
    let paths: Vec<String> = data_files
        .iter()
        .map(|file| file.file_path().to_string())
        .collect();
    let targets: Vec<(&str, &[u64])> = paths
        .iter()
        .map(|path| (path.as_str(), positions.as_slice()))
        .collect();
    let table = append_files(&catalog, &table, data_files).await;
    let eq_del = write_equality_delete_file(&table, PARTITION, &[1000, 1001]).await;
    let table = add_deletes(&catalog, &table, vec![eq_del]).await;
    let dvs = write_dv(&table, PARTITION, &targets).await;
    let table = add_deletes(&catalog, &table, dvs).await;

    let action = RewriteDataFiles::new(table.clone()).rewrite_all(true);
    let table = rewrite_and_reload(&catalog, action, &table).await;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .clone();
    let props = props_of(&snapshot);

    assert_replace_totals_match_live_files(&table, &snapshot).await;
    let mut required = rdf_required_keys(false);
    required.extend([
        "removed-dvs",
        "removed-delete-files",
        "removed-position-deletes",
        "removed-equality-delete-files",
        "removed-equality-deletes",
    ]);
    assert_summary_keys(
        &props,
        &required,
        &[
            "removed-position-delete-files",
            "added-delete-files",
            "added-dvs",
            "added-position-delete-files",
            "added-position-deletes",
            "added-equality-delete-files",
            "added-equality-deletes",
            "entries-processed",
        ],
        "rdf v3 dv+eq",
    );
    assert_eq!(prop_u64(&props, "removed-dvs"), 4);
    assert_eq!(prop_u64(&props, "removed-equality-delete-files"), 1);
    assert_eq!(prop_u64(&props, "removed-delete-files"), 5);
    assert_eq!(prop_u64(&props, "total-equality-deletes"), 0);
}

async fn seed_rpd_table(catalog: &impl Catalog, delete_commits: i64) -> Table {
    let table = create_partitioned_table(catalog, FormatVersion::V2).await;
    let data_a =
        write_data_file(&table, "rpd-a.parquet", PARTITION, &rows(PARTITION, 0, 250)).await;
    let data_b = write_data_file(
        &table,
        "rpd-b.parquet",
        PARTITION,
        &rows(PARTITION, 250, 250),
    )
    .await;
    let deleted_path = data_a.file_path().to_string();
    let mut table = append_files(catalog, &table, vec![data_a, data_b]).await;
    for commit in 0..delete_commits {
        let delete_file =
            write_position_delete_file(&table, PARTITION, &[(deleted_path.clone(), commit)]).await;
        table = add_deletes(catalog, &table, vec![delete_file]).await;
    }
    table
}

const RPD_BASE_KEYS: &[&str] = &[
    "added-delete-files",
    "added-files-size",
    "added-position-deletes",
    "changed-partition-count",
    "removed-delete-files",
    "removed-files-size",
    "removed-position-deletes",
    "manifests-created",
    "manifests-kept",
    "manifests-replaced",
    "total-data-files",
    "total-delete-files",
    "total-records",
    "total-files-size",
    "total-position-deletes",
    "total-equality-deletes",
];

const RPD_FORBIDDEN: &[&str] = &[
    "added-data-files",
    "added-records",
    "deleted-data-files",
    "deleted-records",
    "added-equality-delete-files",
    "added-equality-deletes",
    "removed-equality-delete-files",
    "removed-equality-deletes",
    "entries-processed",
];

#[tokio::test]
async fn rpd_v2_replace_summary_matches_java_keys() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = seed_rpd_table(&catalog, 3).await;

    RewritePositionDeleteFiles::new(table.clone())
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("rewrite position deletes");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .clone();
    let props = props_of(&snapshot);

    assert_eq!(snapshot.summary().operation, Operation::Replace);
    let mut required = RPD_BASE_KEYS.to_vec();
    required.push("added-position-delete-files");
    required.push("removed-position-delete-files");
    let mut forbidden = RPD_FORBIDDEN.to_vec();
    forbidden.extend(["added-dvs", "removed-dvs"]);
    assert_summary_keys(&props, &required, &forbidden, "rpd v2");

    assert_eq!(prop_u64(&props, "added-delete-files"), 1);
    assert_eq!(prop_u64(&props, "added-position-delete-files"), 1);
    assert_eq!(prop_u64(&props, "added-position-deletes"), 3);
    assert_eq!(prop_u64(&props, "removed-delete-files"), 3);
    assert_eq!(prop_u64(&props, "removed-position-delete-files"), 3);
    assert_eq!(prop_u64(&props, "removed-position-deletes"), 3);
    assert_eq!(prop_u64(&props, "total-delete-files"), 1);
    assert_eq!(prop_u64(&props, "total-position-deletes"), 3);
    assert_eq!(prop_u64(&props, "total-data-files"), 2);
    assert_eq!(prop_u64(&props, "total-records"), 500);
}

#[tokio::test]
async fn rpd_v3_replace_summary_reports_added_dv() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = seed_rpd_table(&catalog, 2).await;
    let table = upgrade_to_v3(&catalog, &table).await;

    RewritePositionDeleteFiles::new(table.clone())
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("rewrite position deletes");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .clone();
    let props = props_of(&snapshot);

    assert_eq!(snapshot.summary().operation, Operation::Replace);
    let mut required = RPD_BASE_KEYS.to_vec();
    required.extend(["added-dvs", "removed-position-delete-files"]);
    let mut forbidden = RPD_FORBIDDEN.to_vec();
    forbidden.extend(["added-position-delete-files", "removed-dvs"]);
    assert_summary_keys(&props, &required, &forbidden, "rpd v3");

    assert_eq!(prop_u64(&props, "added-delete-files"), 1);
    assert_eq!(prop_u64(&props, "added-dvs"), 1);
    assert_eq!(prop_u64(&props, "added-position-deletes"), 2);
    assert_eq!(prop_u64(&props, "removed-delete-files"), 2);
    assert_eq!(prop_u64(&props, "removed-position-delete-files"), 2);
    assert_eq!(prop_u64(&props, "removed-position-deletes"), 2);
    assert_eq!(prop_u64(&props, "total-delete-files"), 1);
    assert_eq!(prop_u64(&props, "total-position-deletes"), 2);
}

const RM_REQUIRED_KEYS: &[&str] = &[
    "changed-partition-count",
    "entries-processed",
    "manifests-created",
    "manifests-kept",
    "manifests-replaced",
    "total-data-files",
    "total-delete-files",
    "total-records",
    "total-files-size",
    "total-position-deletes",
    "total-equality-deletes",
];

const RM_FORBIDDEN_KEYS: &[&str] = &[
    "added-data-files",
    "added-records",
    "added-files-size",
    "added-delete-files",
    "added-dvs",
    "added-position-delete-files",
    "added-position-deletes",
    "added-equality-delete-files",
    "added-equality-deletes",
    "deleted-data-files",
    "deleted-records",
    "removed-delete-files",
    "removed-dvs",
    "removed-files-size",
    "removed-position-delete-files",
    "removed-position-deletes",
    "removed-equality-delete-files",
    "removed-equality-deletes",
];

fn assert_rm_totals_carried(snapshot: &Snapshot, parent: &Snapshot) {
    let props = props_of(snapshot);
    let parent_props = props_of(parent);
    for key in [
        "total-data-files",
        "total-delete-files",
        "total-records",
        "total-files-size",
        "total-position-deletes",
        "total-equality-deletes",
    ] {
        assert_eq!(
            prop_u64(&props, key),
            prop_u64(&parent_props, key),
            "{key} carried unchanged through manifest rewrite"
        );
    }
    assert_eq!(snapshot.summary().operation, Operation::Replace);
    assert_eq!(
        prop_u64(&props, "changed-partition-count"),
        0,
        "manifest rewrite changes no partitions"
    );
}

#[tokio::test]
async fn rm_data_only_replace_summary_matches_java_keys() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let mut table = table;
    for index in 0..3i64 {
        let file = write_data_file(
            &table,
            &format!("rm-{index}.parquet"),
            PARTITION,
            &rows(PARTITION, index * 250, 250),
        )
        .await;
        table = append_files(&catalog, &table, vec![file]).await;
    }

    let tx = Transaction::new(&table);
    let action = tx.rewrite_manifests().cluster_by(|_| "all".to_string());
    let tx = action.apply(tx).expect("apply rewrite manifests");
    let table = tx.commit(&catalog).await.expect("commit rewrite manifests");

    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("head snapshot")
        .as_ref()
        .clone();
    let parent = table
        .metadata()
        .snapshot_by_id(snapshot.parent_snapshot_id().expect("parent id"))
        .expect("parent snapshot")
        .as_ref()
        .clone();
    let props = props_of(&snapshot);

    assert_summary_keys(&props, RM_REQUIRED_KEYS, RM_FORBIDDEN_KEYS, "rm data-only");
    assert_rm_totals_carried(&snapshot, &parent);
    assert_eq!(prop_u64(&props, "manifests-created"), 1);
    assert_eq!(prop_u64(&props, "manifests-kept"), 0);
    assert_eq!(prop_u64(&props, "manifests-replaced"), 3);
    assert_eq!(prop_u64(&props, "entries-processed"), 3);
}

#[tokio::test]
async fn rm_delete_manifests_replace_summary_matches_java_keys() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let data_a = write_data_file(
        &table,
        "rmdel-a.parquet",
        PARTITION,
        &rows(PARTITION, 0, 250),
    )
    .await;
    let deleted_path = data_a.file_path().to_string();
    let mut table = append_files(&catalog, &table, vec![data_a]).await;
    let data_b = write_data_file(
        &table,
        "rmdel-b.parquet",
        PARTITION,
        &rows(PARTITION, 250, 250),
    )
    .await;
    table = append_files(&catalog, &table, vec![data_b]).await;
    let delete_file = write_position_delete_file(&table, PARTITION, &[(deleted_path, 0)]).await;
    let table = add_deletes(&catalog, &table, vec![delete_file]).await;

    let tx = Transaction::new(&table);
    let action = tx
        .rewrite_manifests()
        .cluster_by(|_| "all".to_string())
        .rewrite_delete_manifests(true);
    let tx = action.apply(tx).expect("apply rewrite manifests");
    let table = tx.commit(&catalog).await.expect("commit rewrite manifests");

    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("head snapshot")
        .as_ref()
        .clone();
    let parent = table
        .metadata()
        .snapshot_by_id(snapshot.parent_snapshot_id().expect("parent id"))
        .expect("parent snapshot")
        .as_ref()
        .clone();
    let props = props_of(&snapshot);

    assert_summary_keys(
        &props,
        RM_REQUIRED_KEYS,
        RM_FORBIDDEN_KEYS,
        "rm with deletes",
    );
    assert_rm_totals_carried(&snapshot, &parent);
    assert_eq!(prop_u64(&props, "manifests-created"), 2);
    assert_eq!(prop_u64(&props, "manifests-kept"), 0);
    assert_eq!(prop_u64(&props, "manifests-replaced"), 3);
    assert_eq!(prop_u64(&props, "entries-processed"), 3);
}
