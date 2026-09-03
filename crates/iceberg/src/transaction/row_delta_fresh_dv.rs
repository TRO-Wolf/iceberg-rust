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

use crate::delete_file_index::{is_deletion_vector, referenced_data_file_location};
use crate::spec::{DataContentType, DataFile, ManifestContentType};
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
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_file.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            let file = entry.data_file();
            if added_dvs.contains_key(file.file_path()) {
                live_data_entry_by_path
                    .insert(file.file_path().to_string(), entry.sequence_number());
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
    use crate::ErrorKind;
    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, Literal, Struct,
    };
    use crate::transaction::tests::make_v2_minimal_table_in_catalog;
    use crate::transaction::{ApplyTransactionAction, Transaction};

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
}
