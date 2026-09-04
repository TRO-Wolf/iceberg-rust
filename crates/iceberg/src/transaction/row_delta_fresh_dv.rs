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

//! Fresh-DV commit door extracted so `row_delta.rs` can stay under its legacy ceiling.

use std::collections::{HashMap, HashSet};

use futures::stream::{FuturesOrdered, TryStreamExt};

use crate::delete_file_index::{is_deletion_vector, referenced_data_file_location};
use crate::delete_vector_container::DV_IO_CONCURRENCY;
use crate::spec::{DataContentType, DataFile, ManifestContentType, ManifestFile};
use crate::table::Table;
use crate::transaction::snapshot::{dv_desc, latest_snapshot};
use crate::{Error, ErrorKind, Result};

/// Reject a DV over a live file-scoped parquet position delete unless this commit removes that delete.
pub(crate) async fn validate_fresh_dvs_only(
    table: &Table,
    added_dvs: &HashMap<String, &DataFile>,
    removed_delete_files: &[DataFile],
    branch: &str,
) -> Result<()> {
    if added_dvs.is_empty() {
        return Ok(());
    }

    let removed_delete_paths: HashSet<&str> = removed_delete_files
        .iter()
        .map(|file| file.file_path())
        .collect();

    let Some(snapshot) = latest_snapshot(table.metadata(), branch) else {
        return Ok(());
    };
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), &table.metadata_ref())
        .await?;

    let mut live_data_entry_by_path: HashMap<String, Option<i64>> = HashMap::new();
    let data_manifests: Vec<&ManifestFile> = manifest_list
        .entries()
        .iter()
        .filter(|manifest_file| manifest_file.content == ManifestContentType::Data)
        .collect();
    let file_io = table.file_io();
    let mut pending = FuturesOrdered::new();
    let mut issued = 0usize;
    while live_data_entry_by_path.len() < added_dvs.len() {
        let budget = if issued == 0 { 1 } else { DV_IO_CONCURRENCY };
        while pending.len() < budget && issued < data_manifests.len() {
            let manifest_file = data_manifests[issued];
            pending.push_back(async move { manifest_file.load_manifest(file_io).await });
            issued += 1;
        }
        let Some(manifest) = pending.try_next().await? else {
            break;
        };
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            let file = entry.data_file();
            if added_dvs.contains_key(file.file_path()) {
                live_data_entry_by_path
                    .insert(file.file_path().to_string(), entry.sequence_number());
                if live_data_entry_by_path.len() == added_dvs.len() {
                    break;
                }
            }
        }
    }

    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_file.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            let existing = entry.data_file();
            if existing.content_type() != DataContentType::PositionDeletes {
                continue;
            }
            if removed_delete_paths.contains(existing.file_path()) {
                continue;
            }

            if is_deletion_vector(existing) {
                if let Some(referenced) = existing.referenced_data_file()
                    && added_dvs.contains_key(&referenced)
                {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Cannot commit deletion vector for {}: the current snapshot already \
                             carries a live deletion vector for that data file ({}). Read it \
                             back with delete_vector::load_delete_vector, merge it through \
                             DVFileWriter::with_previous_deletes, and pass the superseded file \
                             to RowDelta::remove_deletes_many in THIS commit (Java \
                             BaseDVFileWriter.loadPreviousDeletes + RowDelta.removeDeletes). \
                             Committing as-is would leave two DVs for one data file, which the \
                             scan rejects",
                            referenced,
                            dv_desc(existing)
                        ),
                    ));
                }
            } else {
                let Some(referenced_path) = referenced_data_file_location(existing) else {
                    continue;
                };
                if !added_dvs.contains_key(&referenced_path) {
                    continue;
                }
                let Some(data_seq) = live_data_entry_by_path.get(&referenced_path) else {
                    continue;
                };
                let applies = match (entry.sequence_number(), *data_seq) {
                    (Some(delete_seq), Some(data_seq)) => delete_seq >= data_seq,
                    _ => true,
                };
                if applies {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Cannot commit deletion vector for {}: live position delete file \
                             {} still applies to that data file and would be silently \
                             superseded by the DV at read time. Read it back and merge it \
                             through DVFileWriter::with_previous_deletes, and pass the \
                             superseded file to RowDelta::remove_deletes_many in THIS commit \
                             (Java BaseDVFileWriter.loadPreviousDeletes + \
                             RowDelta.removeDeletes)",
                            referenced_path,
                            existing.file_path()
                        ),
                    ));
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    use tempfile::TempDir;

    use crate::delete_vector_container::close_touched_dv_containers_with_partitions;
    use crate::delete_vector_container::counting::{
        CountingStorageFactory, append, synthetic_data_file as counting_data_file,
    };
    use crate::memory::tests::new_memory_catalog;
    use crate::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, Literal, Struct,
    };
    use crate::table::Table;
    use crate::transaction::tests::{
        make_v2_minimal_table_in_catalog, make_v3_minimal_table_in_catalog,
    };
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{CatalogBuilder, ErrorKind};

    fn synthetic_data_file(path: &str, part_value: i64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(part_value))]))
            .build()
            .unwrap()
    }

    fn synthetic_file_scoped_delete(path: &str, referenced: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .referenced_data_file(Some(referenced.to_string()))
            .build()
            .unwrap()
    }

    fn synthetic_dv_file(path: &str, referenced: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Puffin)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .referenced_data_file(Some(referenced.to_string()))
            .content_offset(Some(4))
            .content_size_in_bytes(Some(40))
            .build()
            .unwrap()
    }

    async fn append_files(
        catalog: &impl crate::Catalog,
        table: &crate::table::Table,
        files: Vec<DataFile>,
    ) -> crate::table::Table {
        let tx = Transaction::new(table);
        let tx = tx.fast_append().add_data_files(files).apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    struct CommitFixture {
        catalog: MemoryCatalog,
        table: Table,
        paths: Vec<String>,
        data_manifest_reads: Arc<AtomicU64>,
        latent: Arc<AtomicBool>,
        _warehouse: TempDir,
    }

    async fn commit_fixture(n: usize) -> CommitFixture {
        let warehouse = TempDir::new().expect("warehouse");
        let data_manifest_reads = Arc::new(AtomicU64::new(0));
        let data_manifest_paths = Arc::new(Mutex::new(HashSet::new()));
        let latent = Arc::new(AtomicBool::new(false));
        let catalog = MemoryCatalogBuilder::default()
            .with_storage_factory(Arc::new(CountingStorageFactory {
                manifest_reads: Arc::new(AtomicU64::new(0)),
                bytes_read: Arc::new(AtomicU64::new(0)),
                data_manifest_paths: data_manifest_paths.clone(),
                data_manifest_reads: data_manifest_reads.clone(),
                latent: latent.clone(),
                ..Default::default()
            }))
            .load(
                "memory",
                HashMap::from([(
                    MEMORY_CATALOG_WAREHOUSE.to_string(),
                    warehouse.path().to_str().expect("utf8").to_string(),
                )]),
            )
            .await
            .expect("catalog");
        let mut table = make_v3_minimal_table_in_catalog(&catalog).await;
        let mut paths = Vec::with_capacity(n);
        for index in 0..n {
            let path = format!("{}/data/f{index}.parquet", table.metadata().location());
            table = append(&catalog, &table, counting_data_file(&path)).await;
            paths.push(path);
        }
        let snapshot = table
            .metadata()
            .current_snapshot()
            .expect("current snapshot");
        let list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .expect("manifest list");
        let data_paths: HashSet<String> = list
            .entries()
            .iter()
            .filter(|manifest| manifest.content == crate::spec::ManifestContentType::Data)
            .map(|manifest| manifest.manifest_path.clone())
            .collect();
        *data_manifest_paths.lock().expect("data-manifest path set") = data_paths;
        data_manifest_reads.store(0, Ordering::Relaxed);
        CommitFixture {
            catalog,
            table,
            paths,
            data_manifest_reads,
            latent,
            _warehouse: warehouse,
        }
    }

    async fn commit_dv_for(
        catalog: &MemoryCatalog,
        table: &Table,
        dv_path: &str,
        referenced: &str,
    ) -> Table {
        let tx = Transaction::new(table);
        let tx = tx
            .row_delta()
            .add_deletes(vec![synthetic_dv_file(dv_path, referenced)])
            .apply(tx)
            .expect("apply row delta");
        tx.commit(catalog).await.expect("commit row delta")
    }

    #[tokio::test]
    async fn fresh_dv_commit_for_newest_file_reads_one_data_manifest() {
        let fixture = commit_fixture(192).await;
        let newest = fixture.paths.last().expect("n > 0").clone();
        commit_dv_for(
            &fixture.catalog,
            &fixture.table,
            "test/newest-dv.puffin",
            &newest,
        )
        .await;
        assert_eq!(
            fixture.data_manifest_reads.load(Ordering::Relaxed),
            1,
            "validation stops once every added DV file is found"
        );
    }

    #[tokio::test]
    async fn fresh_dv_commit_for_oldest_file_reads_every_data_manifest() {
        let fixture = commit_fixture(192).await;
        let oldest = fixture.paths.first().expect("n > 0").clone();
        commit_dv_for(
            &fixture.catalog,
            &fixture.table,
            "test/oldest-dv.puffin",
            &oldest,
        )
        .await;
        assert_eq!(
            fixture.data_manifest_reads.load(Ordering::Relaxed),
            u64::try_from(fixture.paths.len()).expect("count fits"),
            "a file in the oldest data manifest keeps the full walk and validates"
        );
    }

    #[tokio::test]
    async fn fresh_dv_commit_for_unknown_file_reads_every_data_manifest() {
        let fixture = commit_fixture(192).await;
        commit_dv_for(
            &fixture.catalog,
            &fixture.table,
            "test/ghost-dv.puffin",
            "test/ghost.parquet",
        )
        .await;
        assert_eq!(
            fixture.data_manifest_reads.load(Ordering::Relaxed),
            u64::try_from(fixture.paths.len()).expect("count fits"),
            "a key never found keeps the full walk"
        );
    }

    #[tokio::test]
    async fn fresh_dv_commit_for_newest_file_reads_one_data_manifest_on_latent_store() {
        let fixture = commit_fixture(192).await;
        fixture.latent.store(true, Ordering::Relaxed);
        let newest = fixture.paths.last().expect("n > 0").clone();
        commit_dv_for(
            &fixture.catalog,
            &fixture.table,
            "test/newest-dv.puffin",
            &newest,
        )
        .await;
        fixture.latent.store(false, Ordering::Relaxed);
        assert_eq!(
            fixture.data_manifest_reads.load(Ordering::Relaxed),
            1,
            "a latent store must issue one GET, not DV_IO_CONCURRENCY"
        );
    }

    #[tokio::test]
    async fn test_row_delta_dv_refuses_file_scoped_parquet_unless_removed() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;
        let parquet = synthetic_file_scoped_delete("test/a-pos.parquet", "test/a.parquet");
        let tx = Transaction::new(&table);
        let tx = tx
            .row_delta()
            .add_deletes(vec![parquet.clone()])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let tx = Transaction::new(&table);
        let tx = tx
            .upgrade_table_version()
            .set_format_version(FormatVersion::V3)
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let tx = Transaction::new(&table);
        let tx = tx
            .row_delta()
            .add_deletes(vec![synthetic_dv_file(
                "test/a-dv.puffin",
                "test/a.parquet",
            )])
            .apply(tx)
            .unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("a file-scoped parquet delete still blocks a DV");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("test/a.parquet"),
            "door names the referenced file, got: {}",
            err.message()
        );
        let tx = Transaction::new(&table);
        let tx = tx
            .row_delta()
            .add_deletes(vec![synthetic_dv_file(
                "test/a-dv-2.puffin",
                "test/a.parquet",
            )])
            .remove_deletes(parquet)
            .apply(tx)
            .unwrap();
        tx.commit(&catalog)
            .await
            .expect("removing the file-scoped parquet in the same commit lets the DV land");
    }

    #[tokio::test]
    async fn test_row_delta_dv_commits_when_file_scoped_delete_predates_data_file() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let parquet = synthetic_file_scoped_delete("test/a-pos.parquet", "test/a.parquet");
        let tx = Transaction::new(&table);
        let tx = tx.row_delta().add_deletes(vec![parquet]).apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;
        let tx = Transaction::new(&table);
        let tx = tx
            .upgrade_table_version()
            .set_format_version(FormatVersion::V3)
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let tx = Transaction::new(&table);
        let tx = tx
            .row_delta()
            .add_deletes(vec![synthetic_dv_file(
                "test/a-dv.puffin",
                "test/a.parquet",
            )])
            .apply(tx)
            .unwrap();
        tx.commit(&catalog)
            .await
            .expect("a file-scoped delete older than the data file does not block the DV");
    }

    #[tokio::test]
    #[ignore = "measurement, not a CI pin"]
    async fn measure_commit_at_8_48_192_data_manifests() {
        for n in [8usize, 48usize, 192usize] {
            let fixture = commit_fixture(n).await;
            let newest = fixture.paths.last().expect("n > 0").clone();
            let known = HashMap::from([(
                newest.clone(),
                (0i32, Struct::from_iter([Some(Literal::long(0))])),
            )]);
            let new_positions = HashMap::from([(newest.clone(), vec![0u64])]);
            let start = std::time::Instant::now();
            let close = close_touched_dv_containers_with_partitions(
                &fixture.table,
                &new_positions,
                None,
                &known,
                None,
            )
            .await
            .expect("close");
            let close_elapsed = start.elapsed();
            let start = std::time::Instant::now();
            commit_dv_for(
                &fixture.catalog,
                &fixture.table,
                "test/measure-dv.puffin",
                &newest,
            )
            .await;
            let commit_elapsed = start.elapsed();
            println!(
                "F-25 n={n} close={close_elapsed:?} commit={commit_elapsed:?} added={} commit_data_manifest_reads={}",
                close.added.len(),
                fixture.data_manifest_reads.load(Ordering::Relaxed),
            );
            assert_eq!(close.added.len(), 1);
        }
    }
}
