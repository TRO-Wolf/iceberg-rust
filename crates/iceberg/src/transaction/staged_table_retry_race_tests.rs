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

use std::collections::HashMap;
use std::sync::Arc;

use super::version_tests::{
    NoOverwriteStorageFactory, assert_metadata_keys_written_once, replace_creation,
    seed_single_write_table, single_write_catalog, single_write_storage, version_data_file,
};
use crate::memory::{
    MEMORY_CATALOG_METADATA_NAMING, MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder,
};
use crate::table::Table;
use crate::transaction::StagedTableTransaction;
use crate::{Catalog, CatalogBuilder, ErrorKind};

async fn single_write_hadoop_catalog(factory: NoOverwriteStorageFactory) -> MemoryCatalog {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(factory))
        .load(
            "mem",
            HashMap::from([
                (
                    MEMORY_CATALOG_WAREHOUSE.to_string(),
                    "memory://warehouse".to_string(),
                ),
                (
                    MEMORY_CATALOG_METADATA_NAMING.to_string(),
                    "hadoop".to_string(),
                ),
            ]),
        )
        .await
        .expect("load hadoop memory catalog")
}

async fn assert_metadata_version_count(table: &Table, metadata_dir: &str, expected: usize) {
    let listed = table
        .file_io()
        .list(metadata_dir)
        .await
        .expect("list metadata dir");
    let versions: Vec<_> = listed
        .iter()
        .filter(|entry| entry.location.ends_with(".metadata.json"))
        .collect();
    assert_eq!(
        versions.len(),
        expected,
        "expected {expected} metadata versions, got {:?}",
        versions
            .iter()
            .map(|entry| &entry.location)
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn retry_after_a_lost_uuid_pointer_cas_stages_a_fresh_key() {
    let (factory, writes) = single_write_storage();
    let catalog = single_write_catalog(factory).await;
    let table = seed_single_write_table(&catalog).await;
    let base_location = table
        .metadata_location_result()
        .expect("base location")
        .to_string();
    let metadata_dir = format!("{}/metadata", table.metadata().location());

    let loser = StagedTableTransaction::begin_replace(&table, replace_creation(table.identifier()))
        .await
        .expect("begin loser");
    let loser_location = loser
        .table()
        .metadata_location_result()
        .expect("loser staged location")
        .to_string();
    let winner =
        StagedTableTransaction::begin_replace(&table, replace_creation(table.identifier()))
            .await
            .expect("begin winner");
    let winner_location = winner
        .table()
        .metadata_location_result()
        .expect("winner staged location")
        .to_string();
    assert_ne!(
        loser_location, winner_location,
        "concurrent replaces must stage distinct keys"
    );

    let committed_winner = winner.commit(&catalog).await.expect("winner commits");
    assert_eq!(
        committed_winner
            .metadata_location_result()
            .expect("winner pointer"),
        winner_location.as_str()
    );

    let err = match loser.commit(&catalog).await {
        Ok(_) => panic!("a replace on a moved pointer must fail the CAS"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
    assert!(
        err.retryable(),
        "a lost pointer CAS must be retryable: {err}"
    );

    let reloaded = catalog
        .load_table(table.identifier())
        .await
        .expect("reload after conflict");
    assert_eq!(
        reloaded
            .metadata_location_result()
            .expect("reloaded pointer"),
        winner_location.as_str()
    );
    let retry =
        StagedTableTransaction::begin_replace(&reloaded, replace_creation(table.identifier()))
            .await
            .expect("re-begin after conflict");
    let retry_location = retry
        .table()
        .metadata_location_result()
        .expect("retry staged location")
        .to_string();
    assert_ne!(
        retry_location, loser_location,
        "a retry must stage a fresh key, not reuse the loser's written key"
    );
    assert_ne!(
        retry_location, winner_location,
        "a retry must stage a fresh key, not reuse the winner's key"
    );
    assert!(
        retry_location.starts_with(&format!("{metadata_dir}/00002-")),
        "the retry must continue the winner's version, got {retry_location}"
    );

    let committed_retry = retry.commit(&catalog).await.expect("retry commits");
    assert_eq!(
        committed_retry
            .metadata_location_result()
            .expect("retry pointer"),
        retry_location.as_str()
    );
    assert_metadata_keys_written_once(&writes, &[
        base_location,
        winner_location,
        loser_location,
        retry_location,
    ]);
    assert_metadata_version_count(&committed_retry, &metadata_dir, 4).await;
}

#[tokio::test]
async fn concurrent_uuid_replaces_keep_the_winners_file_byte_identical() {
    let (factory, writes) = single_write_storage();
    let catalog = single_write_catalog(factory).await;
    let table = seed_single_write_table(&catalog).await;
    let base_location = table
        .metadata_location_result()
        .expect("base location")
        .to_string();

    let loser = StagedTableTransaction::begin_replace(&table, replace_creation(table.identifier()))
        .await
        .expect("begin loser")
        .add_data_files(vec![version_data_file(
            "memory://warehouse/ns/t/data/loser.parquet",
            3,
        )]);
    let loser_location = loser
        .table()
        .metadata_location_result()
        .expect("loser staged location")
        .to_string();
    let winner =
        StagedTableTransaction::begin_replace(&table, replace_creation(table.identifier()))
            .await
            .expect("begin winner")
            .add_data_files(vec![version_data_file(
                "memory://warehouse/ns/t/data/winner.parquet",
                5,
            )]);
    let winner_location = winner
        .table()
        .metadata_location_result()
        .expect("winner staged location")
        .to_string();
    assert_ne!(
        loser_location, winner_location,
        "concurrent replaces must stage distinct keys"
    );

    let committed = winner.commit(&catalog).await.expect("winner commits");
    let winner_bytes = committed
        .file_io()
        .new_input(&winner_location)
        .expect("open winner")
        .read()
        .await
        .expect("read winner bytes");

    let err = match loser.commit(&catalog).await {
        Ok(_) => panic!("a replace on a moved pointer must fail the CAS"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
    assert!(
        err.retryable(),
        "a lost pointer CAS must be retryable: {err}"
    );
    assert_eq!(
        committed
            .file_io()
            .new_input(&winner_location)
            .expect("reopen winner")
            .read()
            .await
            .expect("reread winner"),
        winner_bytes,
        "the loser must not touch the winner's file"
    );
    let current = catalog
        .load_table(table.identifier())
        .await
        .expect("reload after race");
    assert_eq!(
        current.metadata_location_result().expect("pointer"),
        winner_location.as_str()
    );
    assert_metadata_keys_written_once(&writes, &[base_location, winner_location, loser_location]);
}

#[tokio::test]
async fn hadoop_retry_after_a_lost_next_version_stages_a_fresh_version() {
    let (factory, writes) = single_write_storage();
    let catalog = single_write_hadoop_catalog(factory).await;
    let table = seed_single_write_table(&catalog).await;
    let base_location = table
        .metadata_location_result()
        .expect("base location")
        .to_string();
    let metadata_dir = format!("{}/metadata", table.metadata().location());
    assert_eq!(
        base_location,
        format!("{metadata_dir}/v1.metadata.json"),
        "a hadoop seed must publish v1, got {base_location}"
    );

    let loser = StagedTableTransaction::begin_replace(&table, replace_creation(table.identifier()))
        .await
        .expect("begin loser");
    let loser_location = loser
        .table()
        .metadata_location_result()
        .expect("loser staged location")
        .to_string();
    assert_eq!(
        loser_location,
        format!("{metadata_dir}/v2.metadata.json"),
        "a hadoop replace must stage v2, got {loser_location}"
    );
    let winner =
        StagedTableTransaction::begin_replace(&table, replace_creation(table.identifier()))
            .await
            .expect("begin winner");
    let winner_location = winner
        .table()
        .metadata_location_result()
        .expect("winner staged location")
        .to_string();
    assert_eq!(
        winner_location, loser_location,
        "hadoop replaces from one base share one deterministic target"
    );

    let committed = winner.commit(&catalog).await.expect("winner lands v2");
    let winner_bytes = committed
        .file_io()
        .new_input(&winner_location)
        .expect("open winner")
        .read()
        .await
        .expect("read winner bytes");

    let err = match loser.commit(&catalog).await {
        Ok(_) => panic!("a replace onto a landed v2 must fail"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
    assert!(
        err.retryable(),
        "a lost next version must be retryable: {err}"
    );
    assert_eq!(
        committed
            .file_io()
            .new_input(&winner_location)
            .expect("reopen winner")
            .read()
            .await
            .expect("reread winner"),
        winner_bytes,
        "the loser must not touch the winner's file"
    );

    let reloaded = catalog
        .load_table(table.identifier())
        .await
        .expect("reload after conflict");
    assert_eq!(
        reloaded
            .metadata_location_result()
            .expect("reloaded pointer"),
        winner_location.as_str()
    );
    let retry =
        StagedTableTransaction::begin_replace(&reloaded, replace_creation(table.identifier()))
            .await
            .expect("re-begin after conflict");
    let retry_location = retry
        .table()
        .metadata_location_result()
        .expect("retry staged location")
        .to_string();
    assert_eq!(
        retry_location,
        format!("{metadata_dir}/v3.metadata.json"),
        "the retry must stage v3, got {retry_location}"
    );
    let committed_retry = retry.commit(&catalog).await.expect("retry commits");
    assert_eq!(
        committed_retry
            .metadata_location_result()
            .expect("retry pointer"),
        retry_location.as_str()
    );
    assert_metadata_keys_written_once(&writes, &[base_location, winner_location, retry_location]);
    assert_metadata_version_count(&committed_retry, &metadata_dir, 3).await;
}
