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

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use iceberg::io::FileIO;
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, NestedField,
    PrimitiveType, Schema, Struct, TableMetadataBuilder, Type,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{
    Catalog, CatalogBuilder, ErrorKind, MetadataLocation, NamespaceIdent, Result, TableCreation,
    TableIdent,
};

use crate::catalog::GlueCatalog;
use crate::commit_transport::{
    DiscardingGlueCommitTransport, GlueCommitScript, GlueCommitTransport,
    ScriptedGlueCommitTransport,
};
use crate::{AWS_REGION_NAME, GLUE_CATALOG_PROP_WAREHOUSE, GlueCatalogBuilder};

const CREDENTIALED_ENV: &str = "ICEBERG_PR5A_CREDENTIALED";
const COMMIT_CLASSES: [CommitClass; 7] = [
    CommitClass::SnapshotAppend,
    CommitClass::RowDeltaV3Dv,
    CommitClass::RewriteFiles,
    CommitClass::SchemaUpdate,
    CommitClass::PropertyUpdate,
    CommitClass::V2ToV3Upgrade,
    CommitClass::SnapshotReferenceUpdate,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommitClass {
    SnapshotAppend,
    RowDeltaV3Dv,
    RewriteFiles,
    SchemaUpdate,
    PropertyUpdate,
    V2ToV3Upgrade,
    SnapshotReferenceUpdate,
}

impl CommitClass {
    fn name(self) -> &'static str {
        match self {
            Self::SnapshotAppend => "snapshot-append",
            Self::RowDeltaV3Dv => "row-delta-v3-dv",
            Self::RewriteFiles => "rewrite-files",
            Self::SchemaUpdate => "schema-update",
            Self::PropertyUpdate => "property-update",
            Self::V2ToV3Upgrade => "v2-to-v3-upgrade",
            Self::SnapshotReferenceUpdate => "snapshot-reference-update",
        }
    }

    fn format_version(self) -> FormatVersion {
        match self {
            Self::RowDeltaV3Dv => FormatVersion::V3,
            _ => FormatVersion::V2,
        }
    }

    fn needs_seed_append(self) -> bool {
        matches!(
            self,
            Self::RowDeltaV3Dv | Self::RewriteFiles | Self::SnapshotReferenceUpdate
        )
    }
}

static TABLE_SEQ: AtomicU64 = AtomicU64::new(0);

fn schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("schema")
}

fn data_file(path: &str) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .expect("data file")
}

fn dv_file(path: &str, referenced: &str) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::PositionDeletes)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Puffin)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .referenced_data_file(Some(referenced.to_string()))
        .content_offset(Some(4))
        .content_size_in_bytes(Some(40))
        .build()
        .expect("dv file")
}

fn unique_ident() -> TableIdent {
    let n = TABLE_SEQ.fetch_add(1, Ordering::SeqCst);
    TableIdent::new(NamespaceIdent::new("pr5a".to_string()), format!("t{n}"))
}

async fn dummy_glue_client() -> aws_sdk_glue::Client {
    let cfg = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .credentials_provider(aws_sdk_glue::config::Credentials::new(
            "pr5a", "pr5a", None, None, "pr5a",
        ))
        .region(aws_config::Region::new("us-east-1"))
        .load()
        .await;
    aws_sdk_glue::Client::new(&cfg)
}

async fn seed_table(file_io: &FileIO, ident: &TableIdent, format: FormatVersion) -> Table {
    let location = format!(
        "memory://pr5a/{}/{}",
        ident.namespace().to_url_string(),
        ident.name()
    );
    let creation = TableCreation::builder()
        .name(ident.name().to_string())
        .location(location.clone())
        .schema(schema())
        .format_version(format)
        .properties([
            ("commit.retry.num-retries".to_string(), "3".to_string()),
            ("commit.retry.min-wait-ms".to_string(), "1".to_string()),
            ("commit.retry.max-wait-ms".to_string(), "5".to_string()),
            (
                "commit.status-check.num-retries".to_string(),
                "1".to_string(),
            ),
            (
                "commit.status-check.min-wait-ms".to_string(),
                "1".to_string(),
            ),
            (
                "commit.status-check.max-wait-ms".to_string(),
                "5".to_string(),
            ),
            (
                "commit.status-check.total-timeout-ms".to_string(),
                "50".to_string(),
            ),
        ])
        .build();
    let metadata = TableMetadataBuilder::from_table_creation(creation)
        .expect("metadata builder")
        .build()
        .expect("metadata")
        .metadata;
    let metadata_location = MetadataLocation::new_with_table_location(location).to_string();
    metadata
        .write_to(file_io, &metadata_location)
        .await
        .expect("write seed metadata");
    Table::builder()
        .identifier(ident.clone())
        .metadata(metadata)
        .metadata_location(metadata_location)
        .file_io(file_io.clone())
        .build()
        .expect("seed table")
}

async fn catalog_with(
    scripts: impl IntoIterator<Item = GlueCommitScript>,
    format: FormatVersion,
) -> (
    GlueCatalog,
    Table,
    Arc<ScriptedGlueCommitTransport>,
    FileIO,
    TableIdent,
) {
    let file_io = FileIO::new_with_memory();
    let ident = unique_ident();
    let table = seed_table(&file_io, &ident, format).await;
    let scripted = ScriptedGlueCommitTransport::new(scripts);
    let client = dummy_glue_client().await;
    let catalog = GlueCatalog::for_commit_outcome_tests(
        file_io.clone(),
        Arc::clone(&scripted) as Arc<dyn GlueCommitTransport>,
        table.clone(),
        client,
    );
    (catalog, table, scripted, file_io, ident)
}

fn apply_class(table: &Table, class: CommitClass, seed_path: &str) -> Transaction {
    let tx = Transaction::new(table);
    match class {
        CommitClass::SnapshotAppend => {
            let action = tx
                .fast_append()
                .add_data_files(vec![data_file(&format!("{seed_path}/append.parquet"))]);
            action.apply(tx).expect("apply append")
        }
        CommitClass::PropertyUpdate => tx
            .update_table_properties()
            .set("pr5a.class".to_string(), class.name().to_string())
            .apply(tx)
            .expect("apply properties"),
        CommitClass::SchemaUpdate => tx
            .update_schema()
            .add_column("pr5a_extra", Type::Primitive(PrimitiveType::Long))
            .apply(tx)
            .expect("apply schema"),
        CommitClass::V2ToV3Upgrade => tx
            .upgrade_table_version()
            .set_format_version(FormatVersion::V3)
            .apply(tx)
            .expect("apply upgrade"),
        CommitClass::SnapshotReferenceUpdate => {
            let snapshot_id = table
                .metadata()
                .current_snapshot()
                .expect("snapshot for ref update")
                .snapshot_id();
            tx.manage_snapshots()
                .create_branch("audit", snapshot_id)
                .apply(tx)
                .expect("apply branch")
        }
        CommitClass::RowDeltaV3Dv => {
            let action = tx.row_delta().add_deletes(vec![dv_file(
                &format!("{seed_path}/dv.puffin"),
                &format!("{seed_path}/data.parquet"),
            )]);
            action.apply(tx).expect("apply row delta")
        }
        CommitClass::RewriteFiles => {
            let original = data_file(&format!("{seed_path}/data.parquet"));
            let rewritten = data_file(&format!("{seed_path}/rewritten.parquet"));
            tx.rewrite_files(vec![original], vec![rewritten])
                .apply(tx)
                .expect("apply rewrite")
        }
    }
}

async fn seed_append_if_needed(
    catalog: &GlueCatalog,
    table: Table,
    class: CommitClass,
    seed_path: &str,
) -> Table {
    if !class.needs_seed_append() {
        return table;
    }
    let tx = Transaction::new(&table);
    let action = tx
        .fast_append()
        .add_data_files(vec![data_file(&format!("{seed_path}/data.parquet"))]);
    let tx = action.apply(tx).expect("apply seed append");
    tx.commit(catalog).await.expect("seed append")
}

async fn commit_class(catalog: &GlueCatalog, table: Table, class: CommitClass) -> Result<Table> {
    let seed_path = format!("memory://pr5a/files/{}", class.name());
    let table = seed_append_if_needed(catalog, table, class, &seed_path).await;
    let tx = apply_class(&table, class, &seed_path);
    tx.commit(catalog).await
}

#[tokio::test]
async fn never_sent_is_terminal_for_every_commit_class_on_the_shared_glue_path() {
    for class in COMMIT_CLASSES {
        let scripts = if class.needs_seed_append() {
            vec![GlueCommitScript::Success, GlueCommitScript::StopBeforeSend]
        } else {
            vec![GlueCommitScript::StopBeforeSend]
        };
        let (catalog, table, scripted, file_io, _) =
            catalog_with(scripts, class.format_version()).await;
        let before_files = count_metadata_files(&file_io, &table).await;
        let error = commit_class(&catalog, table.clone(), class)
            .await
            .expect_err(class.name());
        assert_eq!(
            error.kind(),
            ErrorKind::Unexpected,
            "{} never-sent must stay terminal",
            class.name()
        );
        assert_ne!(error.kind(), ErrorKind::CommitStateUnknown);
        assert!(
            !error.retryable(),
            "{} never-sent must not retry",
            class.name()
        );
        assert!(
            error.message().contains("before the update request"),
            "{} message: {}",
            class.name(),
            error.message()
        );
        let expected_attempts = if class.needs_seed_append() { 2 } else { 1 };
        assert_eq!(
            catalog.catalog_commit_attempts(),
            expected_attempts,
            "{} catalog attempts",
            class.name()
        );
        assert_eq!(
            scripted.catalog_commit_attempts(),
            expected_attempts,
            "{} transport attempts",
            class.name()
        );
        let after_files = count_metadata_files(&file_io, &table).await;
        assert!(
            after_files > before_files,
            "{} staged metadata must remain after never-sent",
            class.name()
        );
    }
}

#[tokio::test]
async fn maybe_sent_is_unknown_without_a_second_commit_for_every_commit_class() {
    for class in COMMIT_CLASSES {
        let scripts = if class.needs_seed_append() {
            vec![GlueCommitScript::Success, GlueCommitScript::MaybeSentLost]
        } else {
            vec![GlueCommitScript::MaybeSentLost]
        };
        let (catalog, table, _, _, _) = catalog_with(scripts, class.format_version()).await;
        let error = commit_class(&catalog, table, class)
            .await
            .expect_err(class.name());
        assert_eq!(
            error.kind(),
            ErrorKind::CommitStateUnknown,
            "{} maybe-sent",
            class.name()
        );
        assert!(!error.retryable());
        let expected_attempts = if class.needs_seed_append() { 2 } else { 1 };
        assert_eq!(
            catalog.catalog_commit_attempts(),
            expected_attempts,
            "{} must not blind-retry an unknown outcome",
            class.name()
        );
    }
}

#[tokio::test]
async fn accepted_then_lost_append_reconciles_without_a_duplicate_commit() {
    let (catalog, table, scripted, _, _) =
        catalog_with([GlueCommitScript::AcceptThenLose], FormatVersion::V2).await;
    let before = table.metadata().snapshots().count();
    let committed = commit_class(&catalog, table, CommitClass::SnapshotAppend)
        .await
        .expect("accepted-then-lost append must reconcile to success");
    assert!(
        scripted.observed_accepted_response_lost(),
        "the response-loss arm must have fired"
    );
    assert_eq!(
        catalog.catalog_commit_attempts(),
        1,
        "reconciliation must not send a second catalog commit"
    );
    assert_eq!(
        committed.metadata().snapshots().count(),
        before + 1,
        "the intended snapshot must be current after reconcile"
    );
}

#[tokio::test]
async fn accepted_then_lost_without_the_snapshot_stays_unknown() {
    let (catalog, table, scripted, _, _) =
        catalog_with([GlueCommitScript::MaybeSentLost], FormatVersion::V2).await;
    let before = table.metadata().snapshots().count();
    let error = commit_class(&catalog, table.clone(), CommitClass::SnapshotAppend)
        .await
        .expect_err("exhaustion");
    assert_eq!(error.kind(), ErrorKind::CommitStateUnknown);
    assert_eq!(catalog.catalog_commit_attempts(), 1);
    let loaded = catalog
        .load_table(table.identifier())
        .await
        .expect("load after exhaustion");
    assert_eq!(
        loaded.metadata().snapshots().count(),
        before,
        "an unconfirmed snapshot must not appear as committed"
    );
    assert!(!scripted.observed_accepted_response_lost());
}

#[tokio::test]
async fn metadata_only_accepted_then_lost_is_typed_unknown_never_success() {
    let (catalog, table, scripted, _, _) =
        catalog_with([GlueCommitScript::AcceptThenLose], FormatVersion::V2).await;
    let error = commit_class(&catalog, table, CommitClass::PropertyUpdate)
        .await
        .expect_err("metadata-only unknown");
    assert!(scripted.observed_accepted_response_lost());
    assert_eq!(error.kind(), ErrorKind::CommitStateUnknown);
    assert!(!error.retryable());
    assert_eq!(catalog.catalog_commit_attempts(), 1);
}

#[tokio::test]
async fn cas_conflict_rebases_when_the_append_validation_contract_permits() {
    let (catalog, table, _, _, _) = catalog_with(
        [
            GlueCommitScript::ConcurrentModification,
            GlueCommitScript::Success,
        ],
        FormatVersion::V2,
    )
    .await;
    let committed = commit_class(&catalog, table, CommitClass::SnapshotAppend)
        .await
        .expect("conflict then success is a permitted rebase");
    assert_eq!(catalog.catalog_commit_attempts(), 2);
    assert!(committed.metadata().current_snapshot().is_some());
}

#[tokio::test]
async fn permanent_authorization_denial_is_terminal_and_does_not_clean_staged_files() {
    let (catalog, table, _, file_io, _) =
        catalog_with([GlueCommitScript::AccessDenied], FormatVersion::V2).await;
    let before_files = count_metadata_files(&file_io, &table).await;
    let error = commit_class(&catalog, table.clone(), CommitClass::SnapshotAppend)
        .await
        .expect_err("auth denial");
    assert_eq!(error.kind(), ErrorKind::Unexpected);
    assert_ne!(error.kind(), ErrorKind::CommitStateUnknown);
    assert!(!error.retryable());
    assert!(error.message().contains("Authorization denied"));
    assert_eq!(catalog.catalog_commit_attempts(), 1);
    let after_files = count_metadata_files(&file_io, &table).await;
    assert!(
        after_files > before_files,
        "authorization denial must not clean possibly-live staged metadata"
    );
}

async fn count_metadata_files(file_io: &FileIO, table: &Table) -> usize {
    let location = table.metadata().location();
    let prefix = format!("{location}/metadata");
    file_io.list(&prefix).await.unwrap_or_default().len()
}

fn credentialed_requested() -> bool {
    std::env::var(CREDENTIALED_ENV).ok().as_deref() == Some("1")
}

fn require_credential_source() {
    assert!(
        std::env::var("AWS_ACCESS_KEY_ID").is_ok()
            || std::env::var("AWS_PROFILE").is_ok()
            || std::env::var("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI").is_ok()
            || std::env::var("AWS_WEB_IDENTITY_TOKEN_FILE").is_ok(),
        "ICEBERG_PR5A_CREDENTIALED=1 requires an AWS credential source"
    );
}

#[tokio::test]
async fn credentialed_glue_commit_class_smokes_and_one_accepted_then_lost_append() {
    if !credentialed_requested() {
        assert_ne!(std::env::var(CREDENTIALED_ENV).ok().as_deref(), Some("1"));
        return;
    }
    let warehouse = std::env::var("ICEBERG_PR5A_GLUE_WAREHOUSE").unwrap_or_default();
    assert!(
        !warehouse.is_empty(),
        "ICEBERG_PR5A_CREDENTIALED=1 requires ICEBERG_PR5A_GLUE_WAREHOUSE"
    );
    require_credential_source();
    let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());
    let props = std::collections::HashMap::from([
        (GLUE_CATALOG_PROP_WAREHOUSE.to_string(), warehouse),
        (AWS_REGION_NAME.to_string(), region),
    ]);
    let catalog = GlueCatalogBuilder::default()
        .load("pr5a-glue", props)
        .await
        .expect("load glue catalog");
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time")
        .as_millis();
    let ident = TableIdent::new(NamespaceIdent::new(format!("pr5a{stamp}")), "t".to_string());
    catalog
        .create_namespace(ident.namespace(), std::collections::HashMap::new())
        .await
        .expect("create unique namespace");
    let creation = TableCreation::builder()
        .name(ident.name().to_string())
        .schema(schema())
        .format_version(FormatVersion::V2)
        .build();
    let mut table = match catalog.create_table(ident.namespace(), creation).await {
        Ok(table) => table,
        Err(error) => {
            let _ = catalog.drop_namespace(ident.namespace()).await;
            panic!("create table: {error}");
        }
    };
    let order = [
        CommitClass::SnapshotAppend,
        CommitClass::SchemaUpdate,
        CommitClass::PropertyUpdate,
        CommitClass::SnapshotReferenceUpdate,
        CommitClass::RewriteFiles,
        CommitClass::V2ToV3Upgrade,
        CommitClass::RowDeltaV3Dv,
    ];
    for class in order {
        table = match commit_class(&catalog, table, class).await {
            Ok(table) => table,
            Err(error) => {
                let _ = catalog.drop_table(&ident).await;
                let _ = catalog.drop_namespace(ident.namespace()).await;
                panic!("{} smoke: {error}", class.name());
            }
        };
    }
    let discarding = Arc::new(DiscardingGlueCommitTransport::new(
        catalog.live_commit_transport(),
    ));
    let discarded =
        catalog.with_commit_transport(Arc::clone(&discarding) as Arc<dyn GlueCommitTransport>);
    let lost = commit_class(&discarded, table, CommitClass::SnapshotAppend).await;
    let _ = discarded.drop_table(&ident).await;
    let _ = discarded.drop_namespace(ident.namespace()).await;
    assert!(
        discarding.observed_accepted_response_lost(),
        "credentialed accepted-then-lost must use discard mode"
    );
    assert_eq!(
        discarding.catalog_commit_attempts(),
        1,
        "catalog_attempts must stay 1 after response loss"
    );
    assert!(
        lost.is_ok()
            || lost
                .as_ref()
                .is_err_and(|error| error.kind() == ErrorKind::CommitStateUnknown),
        "accepted-then-lost is success after reconcile or typed unknown"
    );
}

#[tokio::test]
async fn discarding_transport_marks_accepted_response_lost() {
    let (catalog, _, _, _, _) = catalog_with([GlueCommitScript::Success], FormatVersion::V2).await;
    let discarding = Arc::new(DiscardingGlueCommitTransport::new(
        catalog.live_commit_transport(),
    ));
    let catalog =
        catalog.with_commit_transport(Arc::clone(&discarding) as Arc<dyn GlueCommitTransport>);
    assert_eq!(catalog.catalog_commit_attempts(), 0);
    let discarding = DiscardingGlueCommitTransport::new(ScriptedGlueCommitTransport::new([
        GlueCommitScript::Success,
    ]) as Arc<dyn GlueCommitTransport>);
    let send = discarding
        .send_update_table(crate::commit_transport::GlueUpdateTableCall {
            database_name: "pr5a".to_string(),
            table_input: dummy_table_input(),
            version_id: None,
            catalog_id: None,
        })
        .await;
    assert!(matches!(
        send,
        crate::commit_transport::GlueCommitSend::AcceptedResponseLost
    ));
    assert!(discarding.observed_accepted_response_lost());
    assert_eq!(discarding.catalog_commit_attempts(), 1);
}

fn dummy_table_input() -> aws_sdk_glue::types::TableInput {
    aws_sdk_glue::types::TableInput::builder()
        .name("t")
        .build()
        .expect("table input")
}
