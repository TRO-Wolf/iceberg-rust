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

use std::collections::HashSet;

use crate::memory::tests::new_memory_catalog;
use crate::spec::{DataFile, Literal, ManifestContentType, ManifestFile, Operation, Struct};
use crate::table::Table;
use crate::transaction::rewrite_manifests::tests::{
    append_files, current_manifests, live_entry, scan_y_values, summary_prop,
    write_real_data_file, write_real_position_delete,
};
use crate::transaction::tests::{make_v2_minimal_table_in_catalog, make_v3_minimal_table_in_catalog};
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::Catalog;

async fn manifests_of(table: &Table, content: ManifestContentType) -> Vec<ManifestFile> {
    current_manifests(table)
        .await
        .into_iter()
        .filter(|m| m.content == content)
        .collect()
}

async fn live_delete_data_file(table: &Table, path: &str) -> DataFile {
    for manifest in current_manifests(table).await {
        if manifest.content != ManifestContentType::Deletes {
            continue;
        }
        let entries = manifest
            .load_manifest(table.file_io())
            .await
            .unwrap()
            .entries()
            .iter()
            .filter(|entry| entry.is_alive())
            .map(|entry| entry.data_file().clone())
            .collect::<Vec<_>>();
        for entry in entries {
            if entry.file_path() == path {
                return entry;
            }
        }
    }
    panic!("no live delete entry for {path}");
}

async fn live_delete_paths(table: &Table) -> Vec<String> {
    let mut out = Vec::new();
    for manifest in current_manifests(table).await {
        if manifest.content != ManifestContentType::Deletes {
            continue;
        }
        for entry in manifest
            .load_manifest(table.file_io())
            .await
            .unwrap()
            .entries()
        {
            if entry.is_alive() {
                out.push(entry.data_file().file_path().to_string());
            }
        }
    }
    out
}

async fn raw_delete_entry(
    manifest: &ManifestFile,
    table: &Table,
    path: &str,
) -> (Option<i64>, Option<i64>, Option<i64>) {
    use crate::spec::Manifest;
    let bytes = table
        .file_io()
        .new_input(&manifest.manifest_path)
        .unwrap()
        .read()
        .await
        .unwrap();
    let (_, raw_entries) = Manifest::try_from_avro_bytes(&bytes).unwrap();
    raw_entries
        .iter()
        .find(|entry| entry.file_path() == path)
        .map(|entry| {
            (
                entry.snapshot_id(),
                entry.sequence_number(),
                entry.file_sequence_number,
            )
        })
        .unwrap()
}

async fn write_real_dv_file(
    table: &Table,
    file_name: &str,
    part_value: i64,
    deletes: &[(String, u64)],
) -> DataFile {
    use crate::spec::PartitionKey;
    use crate::writer::base_writer::deletion_vector_writer::DVFileWriter;

    let partition_key = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        Struct::from_iter([Some(Literal::long(part_value))]),
    )
    .expect("PartitionKey::new: valid partition tuple");
    let dv_path = format!("{}/data/{file_name}", table.metadata().location());
    let output_file = table.file_io().new_output(&dv_path).unwrap();
    let mut dv_writer = DVFileWriter::new(output_file).unpartitioned();
    for (data_file_path, pos) in deletes {
        dv_writer
            .delete(data_file_path, *pos, Some(&partition_key))
            .unwrap();
    }
    dv_writer.close().await.unwrap().into_iter().next().unwrap()
}

async fn rewrite_deletes(table: &Table, catalog: &impl Catalog) -> Table {
    let tx = Transaction::new(table);
    let action = tx
        .rewrite_manifests()
        .cluster_by(|_| String::new())
        .rewrite_delete_manifests(true);
    let tx = action.apply(tx).unwrap();
    tx.commit(catalog).await.unwrap()
}

#[tokio::test]
async fn test_rewrite_delete_manifests_unpartitioned_v2() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;

    let f1 = write_real_data_file(&table, "u1.parquet", 0, &[(0, 1)]).await;
    let table = append_files(&catalog, &table, vec![f1.clone()]).await;
    let f2 = write_real_data_file(&table, "u2.parquet", 0, &[(0, 2), (0, 3)]).await;
    let table = append_files(&catalog, &table, vec![f2.clone()]).await;
    let f3 = write_real_data_file(&table, "u3.parquet", 0, &[(0, 4), (0, 5), (0, 6)]).await;
    let table = append_files(&catalog, &table, vec![f3.clone()]).await;

    let d1 = write_real_position_delete(&table, 0, &[(f1.file_path().to_string(), 0)]).await;
    let tx = Transaction::new(&table);
    let tx = tx.row_delta().add_deletes(vec![d1]).apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    let d2 = write_real_position_delete(
        &table,
        0,
        &[
            (f2.file_path().to_string(), 0),
            (f3.file_path().to_string(), 1),
        ],
    )
    .await;
    let tx = Transaction::new(&table);
    let tx = tx.row_delta().add_deletes(vec![d2]).apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    assert_eq!(3, manifests_of(&table, ManifestContentType::Data).await.len());
    assert_eq!(
        2,
        manifests_of(&table, ManifestContentType::Deletes)
            .await
            .len()
    );

    let delete_paths = live_delete_paths(&table).await;
    assert_eq!(2, delete_paths.len());
    let mut before_provenance = Vec::new();
    for path in &delete_paths {
        before_provenance.push(live_entry(&table, path).await);
    }

    let table = rewrite_deletes(&table, &catalog).await;

    let data_manifests = manifests_of(&table, ManifestContentType::Data).await;
    let delete_manifests = manifests_of(&table, ManifestContentType::Deletes).await;
    assert_eq!(1, data_manifests.len());
    assert_eq!(1, delete_manifests.len());
    assert_eq!(Some(3), data_manifests[0].existing_files_count);
    assert_eq!(Some(0), data_manifests[0].added_files_count);
    assert_eq!(Some(0), data_manifests[0].deleted_files_count);
    assert_eq!(Some(2), delete_manifests[0].existing_files_count);
    assert_eq!(Some(0), delete_manifests[0].added_files_count);
    assert_eq!(Some(0), delete_manifests[0].deleted_files_count);

    assert_eq!(Some("2".into()), summary_prop(&table, "manifests-created"));
    assert_eq!(Some("0".into()), summary_prop(&table, "manifests-kept"));
    assert_eq!(Some("5".into()), summary_prop(&table, "manifests-replaced"));
    assert_eq!(
        Operation::Replace,
        table.metadata().current_snapshot().unwrap().summary().operation
    );

    for (i, path) in delete_paths.iter().enumerate() {
        assert_eq!(before_provenance[i], live_entry(&table, path).await);
        let (_, snap, seq, fseq) = before_provenance[i];
        assert_eq!(
            (snap, seq, fseq),
            raw_delete_entry(&delete_manifests[0], &table, path).await,
            "{path}: delete provenance must be stored explicitly, not re-inherited"
        );
        assert_ne!(
            None, seq,
            "{path}: data sequence number must survive the rewrite"
        );
    }

    assert_eq!(HashSet::from([3, 4, 6]), scan_y_values(&table).await);
}

#[tokio::test]
async fn test_rewrite_delete_manifests_unpartitioned_v3_dv() {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;

    let f1 = write_real_data_file(&table, "u1.parquet", 0, &[(0, 1)]).await;
    let table = append_files(&catalog, &table, vec![f1.clone()]).await;
    let f2 = write_real_data_file(&table, "u2.parquet", 0, &[(0, 2), (0, 3)]).await;
    let table = append_files(&catalog, &table, vec![f2.clone()]).await;
    let f3 = write_real_data_file(&table, "u3.parquet", 0, &[(0, 4), (0, 5), (0, 6)]).await;
    let table = append_files(&catalog, &table, vec![f3.clone()]).await;

    let d1 = write_real_dv_file(&table, "dv1.puffin", 0, &[(f1.file_path().to_string(), 0)]).await;
    let tx = Transaction::new(&table);
    let tx = tx.row_delta().add_deletes(vec![d1]).apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    let d2 = write_real_dv_file(
        &table,
        "dv2.puffin",
        0,
        &[
            (f2.file_path().to_string(), 0),
            (f3.file_path().to_string(), 1),
        ],
    )
    .await;
    let tx = Transaction::new(&table);
    let tx = tx.row_delta().add_deletes(vec![d2]).apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    assert_eq!(3, manifests_of(&table, ManifestContentType::Data).await.len());
    assert_eq!(
        2,
        manifests_of(&table, ManifestContentType::Deletes)
            .await
            .len()
    );

    let delete_paths = live_delete_paths(&table).await;
    assert_eq!(2, delete_paths.len());
    let mut before_provenance = Vec::new();
    for path in &delete_paths {
        before_provenance.push(live_entry(&table, path).await);
    }

    let table = rewrite_deletes(&table, &catalog).await;

    let data_manifests = manifests_of(&table, ManifestContentType::Data).await;
    let delete_manifests = manifests_of(&table, ManifestContentType::Deletes).await;
    assert_eq!(1, data_manifests.len());
    assert_eq!(1, delete_manifests.len());
    assert_eq!(Some(3), data_manifests[0].existing_files_count);
    assert_eq!(Some(2), delete_manifests[0].existing_files_count);
    assert_eq!(Some("2".into()), summary_prop(&table, "manifests-created"));
    assert_eq!(Some("5".into()), summary_prop(&table, "manifests-replaced"));

    for (i, path) in delete_paths.iter().enumerate() {
        assert_eq!(before_provenance[i], live_entry(&table, path).await);
        let (_, snap, seq, fseq) = before_provenance[i];
        assert_eq!(
            (snap, seq, fseq),
            raw_delete_entry(&delete_manifests[0], &table, path).await
        );
    }

    assert_eq!(HashSet::from([3, 4, 6]), scan_y_values(&table).await);
}

#[tokio::test]
async fn test_rewrite_delete_manifests_drops_empty_delete_manifest() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;

    let f1 = write_real_data_file(&table, "p1.parquet", 0, &[(0, 1), (0, 2)]).await;
    let table = append_files(&catalog, &table, vec![f1.clone()]).await;
    let f2 = write_real_data_file(&table, "p2.parquet", 0, &[(0, 3), (0, 4)]).await;
    let table = append_files(&catalog, &table, vec![f2.clone()]).await;

    let d1 = write_real_position_delete(&table, 0, &[(f1.file_path().to_string(), 0)]).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![d1.clone()])
        .apply(tx)
        .unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    let d2 = write_real_position_delete(&table, 0, &[(f2.file_path().to_string(), 0)]).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![d2.clone()])
        .apply(tx)
        .unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    assert_eq!(2, manifests_of(&table, ManifestContentType::Data).await.len());
    assert_eq!(
        2,
        manifests_of(&table, ManifestContentType::Deletes)
            .await
            .len()
    );

    let committed_d2 = live_delete_data_file(&table, d2.file_path()).await;
    let tx = Transaction::new(&table);
    let action = tx
        .rewrite_files(
            Vec::<DataFile>::new(),
            Vec::<DataFile>::new(),
        )
        .delete_delete_file(committed_d2);
    let tx = action.apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    let delete_manifests = manifests_of(&table, ManifestContentType::Deletes).await;
    assert_eq!(2, delete_manifests.len());
    let live_total: u32 = delete_manifests
        .iter()
        .map(|m| m.existing_files_count.unwrap_or(0) + m.added_files_count.unwrap_or(0))
        .sum();
    assert_eq!(1, live_total);

    let table = rewrite_deletes(&table, &catalog).await;

    let data_manifests = manifests_of(&table, ManifestContentType::Data).await;
    let delete_manifests = manifests_of(&table, ManifestContentType::Deletes).await;
    assert_eq!(1, data_manifests.len());
    assert_eq!(1, delete_manifests.len());
    assert_eq!(Some(2), data_manifests[0].existing_files_count);
    assert_eq!(Some(1), delete_manifests[0].existing_files_count);
    assert_eq!(Some("2".into()), summary_prop(&table, "manifests-created"));
    assert_eq!(Some("4".into()), summary_prop(&table, "manifests-replaced"));

    let (_, _, seq, fseq) = live_entry(&table, d1.file_path()).await;
    assert!(seq.is_some());
    assert!(fseq.is_some());

    assert_eq!(HashSet::from([2, 3, 4]), scan_y_values(&table).await);
}

#[tokio::test]
async fn test_rewrite_delete_manifests_explicit_false_keeps_java_default() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;

    let f1 = write_real_data_file(&table, "q1.parquet", 0, &[(0, 1)]).await;
    let table = append_files(&catalog, &table, vec![f1.clone()]).await;
    let f2 = write_real_data_file(&table, "q2.parquet", 0, &[(0, 2)]).await;
    let table = append_files(&catalog, &table, vec![f2.clone()]).await;

    let d1 = write_real_position_delete(&table, 0, &[(f1.file_path().to_string(), 0)]).await;
    let tx = Transaction::new(&table);
    let tx = tx.row_delta().add_deletes(vec![d1]).apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    let tx = Transaction::new(&table);
    let action = tx
        .rewrite_manifests()
        .cluster_by(|_| String::new())
        .rewrite_delete_manifests(false);
    let tx = action.apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    assert_eq!(1, manifests_of(&table, ManifestContentType::Data).await.len());
    let delete_manifests = manifests_of(&table, ManifestContentType::Deletes).await;
    assert_eq!(1, delete_manifests.len());
    assert_eq!(Some("1".into()), summary_prop(&table, "manifests-created"));
    assert_eq!(Some("1".into()), summary_prop(&table, "manifests-kept"));
    assert_eq!(Some("2".into()), summary_prop(&table, "manifests-replaced"));
    assert_eq!(HashSet::from([2]), scan_y_values(&table).await);
}

#[tokio::test]
async fn test_rewrite_delete_manifests_respects_rewrite_if() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;

    let f1 = write_real_data_file(&table, "r1.parquet", 0, &[(0, 1)]).await;
    let table = append_files(&catalog, &table, vec![f1.clone()]).await;
    let f2 = write_real_data_file(&table, "r2.parquet", 0, &[(0, 2)]).await;
    let table = append_files(&catalog, &table, vec![f2.clone()]).await;

    let d1 = write_real_position_delete(&table, 0, &[(f1.file_path().to_string(), 0)]).await;
    let tx = Transaction::new(&table);
    let tx = tx.row_delta().add_deletes(vec![d1]).apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();
    let d2 = write_real_position_delete(&table, 0, &[(f2.file_path().to_string(), 0)]).await;
    let tx = Transaction::new(&table);
    let tx = tx.row_delta().add_deletes(vec![d2]).apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    let mut delete_paths: Vec<String> = manifests_of(&table, ManifestContentType::Deletes)
        .await
        .iter()
        .map(|m| m.manifest_path.clone())
        .collect();
    delete_paths.sort();
    let keep_path = delete_paths[1].clone();
    let rewrite_path = delete_paths[0].clone();

    let tx = Transaction::new(&table);
    let action = tx
        .rewrite_manifests()
        .cluster_by(|_| String::new())
        .rewrite_delete_manifests(true)
        .rewrite_if(move |m| m.manifest_path == rewrite_path);
    let tx = action.apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    let after_delete_paths: HashSet<String> = manifests_of(&table, ManifestContentType::Deletes)
        .await
        .iter()
        .map(|m| m.manifest_path.clone())
        .collect();
    assert_eq!(2, after_delete_paths.len());
    assert!(after_delete_paths.contains(&keep_path));
    assert!(!after_delete_paths.contains(&delete_paths[0]));
}
