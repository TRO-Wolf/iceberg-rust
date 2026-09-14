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

use iceberg::spec::{FormatVersion, TableMetadataBuilder};
use iceberg::table::Table;
use iceberg::transaction::StagedTableTransaction;
use iceberg::{Catalog, ErrorKind, TableCreation, TableIdent};

use crate::commit_outcome_tests::{catalog_with, data_file, schema};
use crate::commit_transport::GlueCommitScript;

async fn begin_replace(table: &Table, ident: &TableIdent) -> StagedTableTransaction {
    StagedTableTransaction::begin_replace(
        table,
        TableCreation::builder()
            .name(ident.name().to_string())
            .schema(schema())
            .build(),
    )
    .await
    .expect("begin replace")
}

#[tokio::test]
async fn staged_replace_commit_swaps_glue_pointer_and_retains_uuid() {
    let (catalog, table, _, file_io, ident) =
        catalog_with([GlueCommitScript::Success], FormatVersion::V2).await;
    let base_location = table
        .metadata_location_result()
        .expect("base location")
        .to_string();
    let base_uuid = table.metadata().uuid();

    let published = begin_replace(&table, &ident)
        .await
        .add_data_files(vec![data_file("memory://pr5a/files/replace.parquet")])
        .commit(&catalog)
        .await
        .expect("staged replace commit");

    let published_location = published
        .metadata_location_result()
        .expect("published location")
        .to_string();
    assert!(file_io.exists(&published_location).await.expect("exists"));
    let loaded = catalog.load_table(&ident).await.expect("load");
    assert_eq!(
        loaded.metadata_location(),
        Some(published_location.as_str())
    );
    assert_eq!(published.metadata().uuid(), base_uuid);
    assert!(
        published
            .metadata()
            .metadata_log()
            .iter()
            .any(|entry| entry.metadata_file == base_location),
        "the base metadata file must be appended to the metadata log"
    );
    assert!(published.metadata().current_snapshot().is_some());
    assert_eq!(catalog.catalog_commit_attempts(), 1);
}

#[tokio::test]
async fn replace_publish_stale_base_conflicts_retryable_before_any_send() {
    let (catalog, table, _, _, ident) =
        catalog_with([GlueCommitScript::Success], FormatVersion::V2).await;

    let loser = begin_replace(&table, &ident).await;
    let winner = begin_replace(&table, &ident).await;
    winner.commit(&catalog).await.expect("winner publish");

    let error = loser.commit(&catalog).await.expect_err("stale base");
    assert_eq!(error.kind(), ErrorKind::CatalogCommitConflicts);
    assert!(error.retryable(), "a stale replace base must be retryable");
    assert_eq!(catalog.catalog_commit_attempts(), 1);
}

#[tokio::test]
async fn replace_publish_unreadable_staged_metadata_refuses_before_send() {
    let (catalog, table, _, file_io, ident) =
        catalog_with([GlueCommitScript::Success], FormatVersion::V2).await;
    let staged = begin_replace(&table, &ident).await;
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location")
        .to_string();
    file_io
        .delete(&staged_location)
        .await
        .expect("delete staged");

    let error = staged
        .commit(&catalog)
        .await
        .expect_err("unreadable staged");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(catalog.catalog_commit_attempts(), 0);
    let still = catalog.load_table(&ident).await.expect("load");
    assert_eq!(still.metadata_location(), table.metadata_location());
}

#[tokio::test]
async fn replace_publish_foreign_uuid_staged_metadata_refuses_before_send() {
    let (catalog, table, _, file_io, ident) =
        catalog_with([GlueCommitScript::Success], FormatVersion::V2).await;
    let staged = begin_replace(&table, &ident).await;
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location")
        .to_string();
    let foreign = TableMetadataBuilder::from_table_creation(
        TableCreation::builder()
            .name(ident.name().to_string())
            .schema(schema())
            .location(table.metadata().location().to_string())
            .build(),
    )
    .expect("foreign metadata builder")
    .build()
    .expect("foreign metadata")
    .metadata;
    assert_ne!(foreign.uuid(), table.metadata().uuid());
    foreign
        .write_to(&file_io, &staged_location)
        .await
        .expect("overwrite staged file");

    let error = staged.commit(&catalog).await.expect_err("foreign uuid");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(catalog.catalog_commit_attempts(), 0);
    let still = catalog.load_table(&ident).await.expect("load");
    assert_eq!(still.metadata_location(), table.metadata_location());
}

#[tokio::test]
async fn replace_publish_maybe_sent_lost_is_unknown_and_keeps_pointer() {
    let (catalog, table, _, file_io, ident) =
        catalog_with([GlueCommitScript::MaybeSentLost], FormatVersion::V2).await;
    let staged = begin_replace(&table, &ident).await;
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location")
        .to_string();

    let error = staged.commit(&catalog).await.expect_err("maybe sent lost");
    assert_eq!(error.kind(), ErrorKind::CommitStateUnknown);
    assert!(!error.retryable());
    assert_eq!(catalog.catalog_commit_attempts(), 1);
    assert!(file_io.exists(&staged_location).await.expect("exists"));
    let still = catalog.load_table(&ident).await.expect("load");
    assert_eq!(still.metadata_location(), table.metadata_location());
}

#[tokio::test]
async fn replace_publish_accept_then_lose_is_unknown_and_pointer_moved() {
    let (catalog, table, scripted, file_io, ident) =
        catalog_with([GlueCommitScript::AcceptThenLose], FormatVersion::V2).await;
    let staged = begin_replace(&table, &ident).await;
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location")
        .to_string();

    let error = staged.commit(&catalog).await.expect_err("accept then lose");
    assert_eq!(error.kind(), ErrorKind::CommitStateUnknown);
    assert!(!error.retryable());
    assert_eq!(catalog.catalog_commit_attempts(), 1);
    assert!(scripted.observed_accepted_response_lost());
    assert!(file_io.exists(&staged_location).await.expect("exists"));
    let moved = catalog.load_table(&ident).await.expect("load");
    assert_eq!(moved.metadata_location(), Some(staged_location.as_str()));
}

#[tokio::test]
async fn replace_publish_concurrent_modification_is_retryable_conflict() {
    let (catalog, table, _, _, ident) = catalog_with(
        [GlueCommitScript::ConcurrentModification],
        FormatVersion::V2,
    )
    .await;

    let error = begin_replace(&table, &ident)
        .await
        .commit(&catalog)
        .await
        .expect_err("concurrent modification");
    assert_eq!(error.kind(), ErrorKind::CatalogCommitConflicts);
    assert!(error.retryable());
    assert_eq!(catalog.catalog_commit_attempts(), 1);
}

#[tokio::test]
async fn replace_publish_access_denied_is_terminal() {
    let (catalog, table, _, _, ident) =
        catalog_with([GlueCommitScript::AccessDenied], FormatVersion::V2).await;

    let error = begin_replace(&table, &ident)
        .await
        .commit(&catalog)
        .await
        .expect_err("access denied");
    assert_eq!(error.kind(), ErrorKind::Unexpected);
    assert_ne!(error.kind(), ErrorKind::CommitStateUnknown);
    assert!(!error.retryable());
    assert!(error.message().contains("Authorization denied"));
    assert_eq!(catalog.catalog_commit_attempts(), 1);
}
