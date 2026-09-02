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

//! The add-seam rule of [`FirstRowIdPolicy`], driven end to end through every producer that can
//! add a data file (Java `MergingSnapshotProducer.add(DataFile)` →
//! `Delegates.suppressFirstRowId`, and the `FastAppend` that does not call it).
//!
//! The probe file carries a `first_row_id` no reader would compute for it. The assertions read
//! the manifest bytes back with [`Manifest::parse_avro`], which skips read-side inheritance, so
//! they see the value the producer STORED rather than the one a reader derives.

use std::collections::HashMap;

use crate::memory::tests::new_memory_catalog;
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, Manifest,
    ManifestContentType, ManifestStatus, Struct,
};
use crate::table::Table;
use crate::transaction::tests::make_v3_minimal_table_in_catalog;
use crate::transaction::{ApplyTransactionAction, Transaction};

/// The value the probe file arrives with. No manifest range in these fixtures reaches it, so a
/// stored `Some(PROBE_FIRST_ROW_ID)` can only have come from the caller.
const PROBE_FIRST_ROW_ID: i64 = 90_000;

/// A data file under the fixture's spec 0 (`identity(x)`), partition `(x = 0)`.
fn data_file(path: &str, first_row_id: Option<i64>) -> DataFile {
    let mut file = DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(3)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::long(0))]))
        .build()
        .expect("build the fixture data file");
    file.first_row_id = first_row_id;
    file
}

/// Every live DATA entry's STORED `first_row_id`, keyed by file path.
///
/// Reads the Avro directly: `ManifestFile::load_manifest` runs `assign_first_row_ids`, which
/// overwrites an absent value and would hide the difference this module measures.
async fn stored_first_row_ids(table: &Table) -> HashMap<String, Option<i64>> {
    let metadata = table.metadata();
    let snapshot = metadata
        .current_snapshot()
        .expect("the committed table has a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), metadata)
        .await
        .expect("load the manifest list");

    let mut stored = HashMap::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue;
        }
        let bytes = table
            .file_io()
            .new_input(&manifest_file.manifest_path)
            .expect("open the manifest")
            .read()
            .await
            .expect("read the manifest bytes");
        let manifest = Manifest::parse_avro(&bytes).expect("parse the manifest avro");
        for entry in manifest.entries() {
            if entry.status() == ManifestStatus::Deleted {
                continue;
            }
            stored.insert(
                entry.file_path().to_string(),
                entry.data_file().first_row_id(),
            );
        }
    }
    stored
}

/// The producers of the charter's partition that can add a data file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Producer {
    FastAppend,
    MergeAppend,
    OverwriteFiles,
    ReplacePartitions,
    RewriteFiles,
    RowDelta,
}

impl Producer {
    /// Java `FastAppend` extends `SnapshotProducer`; every other arm extends
    /// `MergingSnapshotProducer`, whose `add(DataFile)` suppresses.
    fn suppresses(self) -> bool {
        self != Producer::FastAppend
    }
}

/// Seed the table with `seed`, then add `probe` through `producer`, and return what each file's
/// `first_row_id` was STORED as.
async fn commit_probe(producer: Producer) -> HashMap<String, Option<i64>> {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;

    let seed = data_file("test/seed.parquet", None);
    let transaction = Transaction::new(&table);
    let transaction = transaction
        .fast_append()
        .add_data_files(vec![seed.clone()])
        .apply(transaction)
        .expect("apply the seed append");
    let table = transaction
        .commit(&catalog)
        .await
        .expect("commit the seed append");

    let probe = data_file("test/probe.parquet", Some(PROBE_FIRST_ROW_ID));
    let transaction = Transaction::new(&table);
    let transaction = match producer {
        Producer::FastAppend => transaction
            .fast_append()
            .add_data_files(vec![probe])
            .apply(transaction),
        Producer::MergeAppend => transaction
            .merge_append()
            .add_data_files(vec![probe])
            .apply(transaction),
        Producer::OverwriteFiles => transaction
            .overwrite_files()
            .add_file(probe)
            .delete_file(seed.file_path().to_string())
            .apply(transaction),
        Producer::ReplacePartitions => transaction
            .replace_partitions()
            .add_file(probe)
            .apply(transaction),
        Producer::RewriteFiles => transaction
            .rewrite_files(vec![seed.clone()], vec![probe])
            .apply(transaction),
        Producer::RowDelta => transaction
            .row_delta()
            .add_data_files(vec![probe])
            .apply(transaction),
    }
    .expect("apply the probe action");
    let table = transaction
        .commit(&catalog)
        .await
        .expect("commit the probe action");

    stored_first_row_ids(&table).await
}

/// The domain table: one row per producer that can add a data file.
///
/// Risk pinned: a stale `first_row_id` survives read-side inheritance, so the added file claims
/// a row-id range that describes other rows. `FastAppend` is the deliberate exception — Java
/// does not suppress there, and matching that asymmetry is the point of the seam.
#[tokio::test]
async fn every_merging_producer_suppresses_first_row_id_and_fast_append_does_not() {
    for producer in [
        Producer::FastAppend,
        Producer::MergeAppend,
        Producer::OverwriteFiles,
        Producer::ReplacePartitions,
        Producer::RewriteFiles,
        Producer::RowDelta,
    ] {
        let stored = commit_probe(producer).await;
        let probe = stored
            .get("test/probe.parquet")
            .copied()
            .unwrap_or_else(|| panic!("{producer:?} committed no probe entry"));
        let expected = if producer.suppresses() {
            None
        } else {
            Some(PROBE_FIRST_ROW_ID)
        };
        assert_eq!(
            probe, expected,
            "{producer:?} stored the wrong first_row_id for the added file"
        );
    }
}

/// The seed file's stored `first_row_id` must stay absent whatever the producer does to it, so
/// the domain table above cannot pass by suppressing every entry in the manifest.
#[tokio::test]
async fn suppression_reaches_only_the_added_file() {
    let stored = commit_probe(Producer::MergeAppend).await;
    assert_eq!(
        stored.get("test/seed.parquet").copied(),
        Some(None),
        "the carried-forward seed entry must still be present and unassigned"
    );
}

/// `DeleteFiles` is the seventh producer of the partition. It passes `Suppress` like every other
/// merging producer, but it hands the producer no data file at all, so the rule is vacuous
/// there. Asserted through the commit rather than by reading the call site.
///
/// The survivor's stored id is its INHERITED one, not an absent value: the rewrite that
/// tombstones the deleted file reads the source manifest through the assigning reader.
#[tokio::test]
async fn delete_files_adds_no_data_file_to_suppress() {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;

    let seed = data_file("test/seed.parquet", None);
    let other = data_file("test/other.parquet", None);
    let transaction = Transaction::new(&table);
    let transaction = transaction
        .fast_append()
        .add_data_files(vec![seed.clone(), other])
        .apply(transaction)
        .expect("apply the seed append");
    let table = transaction
        .commit(&catalog)
        .await
        .expect("commit the seed append");

    let transaction = Transaction::new(&table);
    let transaction = transaction
        .delete_files()
        .delete_file(seed.file_path().to_string())
        .apply(transaction)
        .expect("apply the delete");
    let table = transaction
        .commit(&catalog)
        .await
        .expect("commit the delete");

    let stored = stored_first_row_ids(&table).await;
    assert!(
        !stored.contains_key("test/seed.parquet"),
        "the deleted file must not survive as a live entry"
    );
    assert_eq!(
        stored.len(),
        1,
        "a delete-only commit adds no data file, so it has none to suppress: {stored:?}"
    );
    assert_eq!(
        stored.get("test/other.parquet").copied(),
        Some(Some(3)),
        "the survivor keeps the id it inherited behind the 3-row seed"
    );
}

#[tokio::test]
async fn filtered_manifest_copies_existing_and_deleted_first_row_id() {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;

    let seed = data_file("test/seed.parquet", None);
    let other = data_file("test/other.parquet", None);
    let transaction = Transaction::new(&table);
    let transaction = transaction
        .fast_append()
        .add_data_files(vec![seed.clone(), other])
        .apply(transaction)
        .expect("apply the seed append");
    let table = transaction
        .commit(&catalog)
        .await
        .expect("commit the seed append");

    let transaction = Transaction::new(&table);
    let transaction = transaction
        .delete_files()
        .delete_file(seed.file_path().to_string())
        .apply(transaction)
        .expect("apply the delete");
    let table = transaction
        .commit(&catalog)
        .await
        .expect("commit the delete");

    let metadata = table.metadata();
    let snapshot = metadata
        .current_snapshot()
        .expect("the committed table has a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), metadata)
        .await
        .expect("load the manifest list");
    let mut by_path = HashMap::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue;
        }
        let bytes = table
            .file_io()
            .new_input(&manifest_file.manifest_path)
            .expect("open the manifest")
            .read()
            .await
            .expect("read the manifest bytes");
        let manifest = Manifest::parse_avro(&bytes).expect("parse the manifest avro");
        for entry in manifest.entries() {
            by_path.insert(
                entry.file_path().to_string(),
                (entry.status(), entry.data_file().first_row_id()),
            );
        }
    }
    assert_eq!(
        by_path.get("test/other.parquet").copied(),
        Some((ManifestStatus::Existing, Some(3))),
        "EXISTING survivor keeps the stored file first_row_id"
    );
    assert_eq!(
        by_path.get("test/seed.parquet").copied(),
        Some((ManifestStatus::Deleted, Some(0))),
        "DELETED entry keeps the stored file first_row_id"
    );
}
