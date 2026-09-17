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

use tempfile::TempDir;

use super::*;
use crate::io::{FileIOBuilder, LocalFsStorageFactory, MemoryStorageFactory};
use crate::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use crate::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, NestedField, PrimitiveType, Schema, Struct,
    Type,
};
use crate::{Catalog, CatalogBuilder};

fn schema_id_name() -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                2,
                "name",
                Type::Primitive(PrimitiveType::String),
            )),
        ])
        .build()
        .unwrap()
}

/// A genuinely different column set from [`schema_id_name`] with high, caller-chosen field-ids
/// (50/51) — used to prove that replace takes the caller's ids AS-IS (D5).
fn schema_sku_price() -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                50,
                "sku",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                51,
                "price",
                Type::Primitive(PrimitiveType::Long),
            )),
        ])
        .build()
        .unwrap()
}

fn data_file(path: &str, records: u64) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(records)
        .partition(Struct::empty())
        .partition_spec_id(0)
        .build()
        .expect("build data file")
}

async fn shared_fs_catalog(warehouse: &str) -> (impl Catalog, FileIO) {
    // LocalFs so catalog FileIO and staged FileIO share the same on-disk store
    // (MemoryStorageFactory builds a fresh HashMap per FileIO).
    let factory = Arc::new(LocalFsStorageFactory);
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(factory.clone())
        .load(
            "mem",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_string())]),
        )
        .await
        .unwrap();
    let file_io = FileIOBuilder::new(factory).build();
    (catalog, file_io)
}

#[tokio::test]
async fn create_abort_before_commit_leaves_no_table() {
    let tmp = TempDir::new().unwrap();
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, file_io) = shared_fs_catalog(&warehouse).await;
    let ns = NamespaceIdent::new("sales".into());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let ident = TableIdent::new(ns, "orders".into());
    let location = format!("{warehouse}/sales/orders");
    let creation = TableCreation::builder()
        .name("orders".into())
        .location(location)
        .schema(schema_id_name())
        .build();
    let staged = StagedTableTransaction::begin_create(file_io, ident.clone(), creation)
        .await
        .unwrap();
    assert_eq!(staged.mode(), StagedTableMode::Create);
    drop(staged);
    assert!(!catalog.table_exists(&ident).await.unwrap());
}

#[tokio::test]
async fn create_commit_publishes_once() {
    let tmp = TempDir::new().unwrap();
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, file_io) = shared_fs_catalog(&warehouse).await;
    let ns = NamespaceIdent::new("sales".into());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let ident = TableIdent::new(ns, "orders".into());
    let location = format!("{warehouse}/sales/orders");
    let creation = TableCreation::builder()
        .name("orders".into())
        .location(location.clone())
        .schema(schema_id_name())
        .build();
    let staged = StagedTableTransaction::begin_create(file_io, ident.clone(), creation)
        .await
        .unwrap();
    let file_path = format!("{location}/data/f1.parquet");
    let published = staged
        .add_data_files(vec![data_file(&file_path, 2)])
        .commit(&catalog)
        .await
        .unwrap();
    assert_eq!(published.identifier(), &ident);
    assert!(catalog.table_exists(&ident).await.unwrap());
    let loaded = catalog.load_table(&ident).await.unwrap();
    assert!(loaded.metadata().current_snapshot().is_some());
    assert_eq!(
        loaded
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .operation,
        crate::spec::Operation::Append
    );
}

#[tokio::test]
async fn create_publish_reload_failure_leaves_no_catalog_entry() {
    // Finding N1: the catalog and the staged writer use SEPARATE in-memory stores
    // (`MemoryStorageFactory` builds a fresh HashMap per FileIO), so the staged metadata is
    // UNREADABLE by the catalog's FileIO — the real staged-CTAS reload failure during publish.
    // Risk pinned: an insert-then-read publish leaves a half-created table (`table_exists`
    // true, `load_table` errors) and breaks `IF NOT EXISTS` idempotency on retry.
    let tmp = TempDir::new().unwrap();
    let warehouse = tmp.path().to_string_lossy().to_string();
    let catalog = MemoryCatalogBuilder::default()
        .load(
            "mem",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("load catalog");
    let staged_file_io = FileIOBuilder::new(Arc::new(MemoryStorageFactory)).build();

    let ns = NamespaceIdent::new("sales".into());
    catalog
        .create_namespace(&ns, HashMap::new())
        .await
        .expect("create namespace");
    let ident = TableIdent::new(ns.clone(), "orders".into());
    let location = format!("{warehouse}/sales/orders");
    let creation = TableCreation::builder()
        .name("orders".into())
        .location(location.clone())
        .schema(schema_id_name())
        .build();

    let staged = StagedTableTransaction::begin_create(staged_file_io, ident.clone(), creation)
        .await
        .expect("begin create");

    // Publish MUST error: the catalog's FileIO cannot read the staged metadata.
    staged
        .commit(&catalog)
        .await
        .expect_err("publish must fail when the catalog cannot read the staged metadata");

    // ... and MUST leave no catalog pointer behind (the atomicity guarantee).
    assert!(
        !catalog.table_exists(&ident).await.expect("table_exists"),
        "a failed create-publish must leave no catalog entry (half-created table)"
    );

    // Idempotency intact: the identifier is free, so re-creating the SAME table succeeds
    // (the pre-fix insert-then-read path fails here with TableAlreadyExists).
    let recreated = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("orders".into())
                .location(location)
                .schema(schema_id_name())
                .build(),
        )
        .await
        .expect("re-create of the same identifier after a failed publish must succeed");
    assert_eq!(recreated.identifier(), &ident);
    assert!(
        catalog
            .table_exists(&ident)
            .await
            .expect("table_exists after recreate"),
        "the re-created table must be present"
    );
}

#[tokio::test]
async fn replace_abort_before_commit_keeps_original() {
    let tmp = TempDir::new().unwrap();
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, _) = shared_fs_catalog(&warehouse).await;
    let ns = NamespaceIdent::new("sales".into());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let original = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("orders".into())
                .schema(schema_id_name())
                .build(),
        )
        .await
        .unwrap();
    let original_meta = original.metadata_location_result().unwrap().to_string();

    let creation = TableCreation::builder()
        .name("orders".into())
        .schema(schema_id_name())
        .build();
    let staged = StagedTableTransaction::begin_replace(&original, creation)
        .await
        .unwrap();
    let staged = staged.add_data_files(vec![data_file(&format!("{warehouse}/stage/f.parquet"), 9)]);
    drop(staged);

    let still = catalog.load_table(original.identifier()).await.unwrap();
    assert_eq!(
        still.metadata_location_result().unwrap(),
        original_meta.as_str()
    );
    assert!(still.metadata().current_snapshot().is_none());
}

#[tokio::test]
async fn replace_commit_swaps_pointer() {
    let tmp = TempDir::new().unwrap();
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, _) = shared_fs_catalog(&warehouse).await;
    let ns = NamespaceIdent::new("sales".into());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let original = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("orders".into())
                .schema(schema_id_name())
                .build(),
        )
        .await
        .unwrap();
    let original_meta = original.metadata_location_result().unwrap().to_string();

    let creation = TableCreation::builder()
        .name("orders".into())
        .schema(schema_id_name())
        .build();
    let staged = StagedTableTransaction::begin_replace(&original, creation)
        .await
        .unwrap();
    let published = staged
        .add_data_files(vec![data_file(&format!("{warehouse}/stage/f.parquet"), 2)])
        .commit(&catalog)
        .await
        .unwrap();
    assert_ne!(
        published.metadata_location_result().unwrap(),
        original_meta.as_str()
    );
    let loaded = catalog.load_table(original.identifier()).await.unwrap();
    assert!(loaded.metadata().current_snapshot().is_some());
}

#[tokio::test]
async fn replace_stale_base_is_rejected() {
    let tmp = TempDir::new().unwrap();
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, _) = shared_fs_catalog(&warehouse).await;
    let ns = NamespaceIdent::new("sales".into());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let original = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("orders".into())
                .schema(schema_id_name())
                .build(),
        )
        .await
        .unwrap();

    let creation_a = TableCreation::builder()
        .name("orders".into())
        .schema(schema_id_name())
        .build();
    let staged_a = StagedTableTransaction::begin_replace(&original, creation_a)
        .await
        .unwrap();

    let creation_b = TableCreation::builder()
        .name("orders".into())
        .schema(schema_id_name())
        .build();
    let staged_b = StagedTableTransaction::begin_replace(&original, creation_b)
        .await
        .unwrap();
    let _winner = staged_b.commit(&catalog).await.unwrap();

    let err = staged_a.commit(&catalog).await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
    assert!(err.retryable(), "stale replace must be retryable: {err}");
}

#[tokio::test]
async fn replace_cycle_keeps_location_stable_and_reads_latest() {
    // Finding N2: a published replace must NOT relocate the table. Across repeated CREATE OR
    // REPLACE cycles `metadata().location()` stays equal to the ORIGINAL location (no
    // `__staged_replace` suffix, no compounding), and each replace's read surface reflects
    // ONLY that replace's data (a replace builds fresh metadata, so `total-records` is this
    // cycle's count, never an accumulation).
    let tmp = TempDir::new().unwrap();
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, _) = shared_fs_catalog(&warehouse).await;
    let ns = NamespaceIdent::new("sales".into());
    catalog
        .create_namespace(&ns, HashMap::new())
        .await
        .expect("create namespace");

    let table_location = format!("{warehouse}/sales/orders");
    let original = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("orders".into())
                .location(table_location.clone())
                .schema(schema_id_name())
                .build(),
        )
        .await
        .expect("create original");
    assert_eq!(original.metadata().location(), table_location.as_str());

    let mut current = original;
    for (cycle, records) in [11_u64, 22, 33].into_iter().enumerate() {
        let creation = TableCreation::builder()
            .name("orders".into())
            .schema(schema_id_name())
            .build();
        let staged = StagedTableTransaction::begin_replace(&current, creation)
            .await
            .expect("begin replace");
        let data_path = format!("{table_location}/data/cycle-{cycle}.parquet");
        let published = staged
            .add_data_files(vec![data_file(&data_path, records)])
            .commit(&catalog)
            .await
            .expect("publish replace");

        // The published table's root location is the ORIGINAL — every cycle, no drift.
        assert_eq!(
            published.metadata().location(),
            table_location.as_str(),
            "replace cycle {cycle} drifted the table location"
        );

        // The catalog read surface agrees, and exposes exactly THIS replace's data.
        let reloaded = catalog
            .load_table(current.identifier())
            .await
            .expect("reload after replace");
        assert_eq!(
            reloaded.metadata().location(),
            table_location.as_str(),
            "reloaded location drifted at cycle {cycle}"
        );
        let snapshot = reloaded
            .metadata()
            .current_snapshot()
            .expect("current snapshot after replace");
        assert_eq!(
            snapshot
                .summary()
                .additional_properties
                .get("total-records"),
            Some(&records.to_string()),
            "replace cycle {cycle} did not expose the latest data"
        );
        current = reloaded;
    }
}

#[tokio::test]
async fn replace_retains_uuid_history_and_metadata_log() {
    // D1: a replace is built ON TOP OF the existing metadata (Java `buildReplacement`), not from
    // scratch. Over a CREATE then two REPLACE cycles this pins that (a) the table UUID is
    // retained, (b) the pre-replace snapshot survives in the published metadata (time-travel raw
    // material) while the `main` branch exposes ONLY the latest replace's data, and (c) the
    // metadata log grows (retained + appended) rather than being truncated.
    let tmp = TempDir::new().unwrap();
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, _) = shared_fs_catalog(&warehouse).await;
    let ns = NamespaceIdent::new("sales".into());
    catalog
        .create_namespace(&ns, HashMap::new())
        .await
        .expect("create namespace");

    let table_location = format!("{warehouse}/sales/orders");
    let original = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("orders".into())
                .location(table_location.clone())
                .schema(schema_id_name())
                .build(),
        )
        .await
        .expect("create original");
    let original_uuid = original.metadata().uuid();

    // First replace: establishes snapshot S1 with 5 records.
    let creation1 = TableCreation::builder()
        .name("orders".into())
        .schema(schema_id_name())
        .build();
    StagedTableTransaction::begin_replace(&original, creation1)
        .await
        .expect("begin replace 1")
        .add_data_files(vec![data_file(
            &format!("{table_location}/data/r1.parquet"),
            5,
        )])
        .commit(&catalog)
        .await
        .expect("publish replace 1");
    let after1 = catalog
        .load_table(original.identifier())
        .await
        .expect("reload 1");
    assert_eq!(
        after1.metadata().uuid(),
        original_uuid,
        "replace 1 regenerated the table UUID"
    );
    let s1 = after1
        .metadata()
        .current_snapshot()
        .expect("snapshot after replace 1")
        .snapshot_id();
    let metadata_log_len_after1 = after1.metadata().metadata_log().len();

    // Second replace: establishes snapshot S2 with 7 records.
    let creation2 = TableCreation::builder()
        .name("orders".into())
        .schema(schema_id_name())
        .build();
    StagedTableTransaction::begin_replace(&after1, creation2)
        .await
        .expect("begin replace 2")
        .add_data_files(vec![data_file(
            &format!("{table_location}/data/r2.parquet"),
            7,
        )])
        .commit(&catalog)
        .await
        .expect("publish replace 2");
    let after2 = catalog
        .load_table(original.identifier())
        .await
        .expect("reload 2");

    // (a) UUID retained across BOTH replaces (from_table_creation would mint a fresh v7 UUID).
    assert_eq!(
        after2.metadata().uuid(),
        original_uuid,
        "replace 2 regenerated the table UUID"
    );

    // (b) the pre-replace snapshot S1 is still present (history retained) ...
    assert!(
        after2.metadata().snapshot_by_id(s1).is_some(),
        "replace 2 truncated the snapshot history (S1 lost)"
    );
    assert!(
        after2.metadata().snapshots().len() >= 2,
        "history not retained: both snapshots should survive the replace"
    );
    // ... while `main` exposes ONLY the latest replace's data.
    let current = after2
        .metadata()
        .current_snapshot()
        .expect("current snapshot after replace 2");
    assert_ne!(
        current.snapshot_id(),
        s1,
        "main still points at the pre-replace snapshot"
    );
    assert_eq!(
        current.summary().additional_properties.get("total-records"),
        Some(&"7".to_string()),
        "main did not expose exactly the latest replace's data"
    );

    // (c) the metadata log GREW (retained + appended), never truncated.
    assert!(
        after2.metadata().metadata_log().len() > metadata_log_len_after1,
        "metadata log was truncated rather than appended-to across replace 2 \
         (after1={metadata_log_len_after1}, after2={})",
        after2.metadata().metadata_log().len()
    );
}

#[tokio::test]
async fn replace_default_creation_preserves_v1_format_version() {
    // D4 regression pin: a CREATE OR REPLACE of a V1 table with a DEFAULT-built TableCreation
    // (no `format-version` property; the builder defaults `format_version` to V2) must NOT
    // upgrade the on-disk format version. Java `buildReplacement` derives the version ONLY from
    // a `format-version` property directive — absent ⇒ keep the existing version. The prior
    // `max(previous, creation.format_version)` derivation silently upgraded V1 → V2 here.
    let tmp = TempDir::new().unwrap();
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, _) = shared_fs_catalog(&warehouse).await;
    let ns = NamespaceIdent::new("sales".into());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let table_location = format!("{warehouse}/sales/orders");
    let original = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("orders".into())
                .location(table_location.clone())
                .schema(schema_id_name())
                .format_version(FormatVersion::V1)
                .build(),
        )
        .await
        .unwrap();
    assert_eq!(original.metadata().format_version(), FormatVersion::V1);

    let creation = TableCreation::builder()
        .name("orders".into())
        .schema(schema_id_name())
        .build();
    // The default-built TableCreation carries the V2 default — this is precisely the trap: it is
    // indistinguishable from an explicit request, so the replace path must ignore it.
    assert_eq!(creation.format_version, FormatVersion::V2);

    let published = StagedTableTransaction::begin_replace(&original, creation)
        .await
        .unwrap()
        .add_data_files(vec![data_file(
            &format!("{table_location}/data/f.parquet"),
            3,
        )])
        .commit(&catalog)
        .await
        .unwrap();
    assert_eq!(
        published.metadata().format_version(),
        FormatVersion::V1,
        "replace with a default TableCreation silently upgraded the format version V1 -> V2"
    );
    let reloaded = catalog.load_table(original.identifier()).await.unwrap();
    assert_eq!(reloaded.metadata().format_version(), FormatVersion::V1);
}

#[tokio::test]
async fn replace_upgrades_format_version_by_property() {
    // D4: an explicit `format-version` property directs the upgrade (V1 -> V2), and the key is
    // consumed as a DIRECTIVE — it must NOT appear in the published property map (Java
    // `persistedProperties` filters reserved properties out of the persisted map).
    let tmp = TempDir::new().unwrap();
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, _) = shared_fs_catalog(&warehouse).await;
    let ns = NamespaceIdent::new("sales".into());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let table_location = format!("{warehouse}/sales/orders");
    let original = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("orders".into())
                .location(table_location.clone())
                .schema(schema_id_name())
                .format_version(FormatVersion::V1)
                .build(),
        )
        .await
        .unwrap();

    let creation = TableCreation::builder()
        .name("orders".into())
        .schema(schema_id_name())
        .properties(HashMap::from([(
            TableProperties::PROPERTY_FORMAT_VERSION.to_string(),
            "2".to_string(),
        )]))
        .build();
    let published = StagedTableTransaction::begin_replace(&original, creation)
        .await
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    assert_eq!(
        published.metadata().format_version(),
        FormatVersion::V2,
        "property-directed upgrade to V2 did not take"
    );
    assert!(
        !published
            .metadata()
            .properties()
            .contains_key(TableProperties::PROPERTY_FORMAT_VERSION),
        "the format-version directive leaked into the persisted property map"
    );
}

#[tokio::test]
async fn replace_downgrade_attempt_errors_and_keeps_original() {
    // D4: an explicit LOWER `format-version` ("1" on a V2 table) is a hard `DataInvalid` error
    // (Java `Builder.upgradeFormatVersion` throws on downgrade), and the failed `begin_replace`
    // leaves the original table current & unchanged in the catalog.
    let tmp = TempDir::new().unwrap();
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, _) = shared_fs_catalog(&warehouse).await;
    let ns = NamespaceIdent::new("sales".into());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let table_location = format!("{warehouse}/sales/orders");
    let original = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("orders".into())
                .location(table_location.clone())
                .schema(schema_id_name())
                .format_version(FormatVersion::V2)
                .build(),
        )
        .await
        .unwrap();
    let original_meta = original.metadata_location_result().unwrap().to_string();

    let creation = TableCreation::builder()
        .name("orders".into())
        .schema(schema_id_name())
        .properties(HashMap::from([(
            TableProperties::PROPERTY_FORMAT_VERSION.to_string(),
            "1".to_string(),
        )]))
        .build();
    let err = match StagedTableTransaction::begin_replace(&original, creation).await {
        Ok(_) => panic!("explicit format-version downgrade must error, not succeed"),
        Err(e) => e,
    };
    assert_eq!(
        err.kind(),
        ErrorKind::DataInvalid,
        "explicit format-version downgrade must be DataInvalid: {err}"
    );

    // The original table is still current & unchanged (V2, same metadata pointer).
    let still = catalog.load_table(original.identifier()).await.unwrap();
    assert_eq!(
        still.metadata_location_result().unwrap(),
        original_meta.as_str()
    );
    assert_eq!(still.metadata().format_version(), FormatVersion::V2);
}

#[tokio::test]
async fn replace_with_different_schema_keeps_caller_ids() {
    // D5: replace takes the caller's schema field-ids AS-IS (NOT Java's name-based
    // `assignFreshIds`); `last_column_id` only advances monotonically (max, never reduced below
    // the base). A replace with a genuinely different column set pins that the caller ids survive
    // verbatim in the published current schema, and pre-replace snapshots stay readable via their
    // own schema (per-snapshot schema binding).
    let tmp = TempDir::new().unwrap();
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, _) = shared_fs_catalog(&warehouse).await;
    let ns = NamespaceIdent::new("sales".into());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let table_location = format!("{warehouse}/sales/orders");
    let original = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("orders".into())
                .location(table_location.clone())
                .schema(schema_id_name())
                .build(),
        )
        .await
        .unwrap();

    // Replace 1: establish snapshot S1 under the id/name schema.
    StagedTableTransaction::begin_replace(
        &original,
        TableCreation::builder()
            .name("orders".into())
            .schema(schema_id_name())
            .build(),
    )
    .await
    .unwrap()
    .add_data_files(vec![data_file(
        &format!("{table_location}/data/r1.parquet"),
        5,
    )])
    .commit(&catalog)
    .await
    .unwrap();
    let after1 = catalog.load_table(original.identifier()).await.unwrap();
    let s1_snapshot = after1
        .metadata()
        .current_snapshot()
        .expect("snapshot after replace 1");
    let s1 = s1_snapshot.snapshot_id();
    let s1_schema_id = s1_snapshot.schema_id();

    // Replace 2: a genuinely DIFFERENT column set with high, caller-chosen ids (50/51).
    let published = StagedTableTransaction::begin_replace(
        &after1,
        TableCreation::builder()
            .name("orders".into())
            .schema(schema_sku_price())
            .build(),
    )
    .await
    .unwrap()
    .add_data_files(vec![data_file(
        &format!("{table_location}/data/r2.parquet"),
        7,
    )])
    .commit(&catalog)
    .await
    .unwrap();

    // Caller ids preserved AS-IS in the published current schema (no name-based reassignment).
    let current = published.metadata().current_schema();
    assert!(
        current.field_by_id(50).is_some_and(|f| f.name == "sku"),
        "caller id 50 (sku) was not preserved as-is"
    );
    assert!(
        current.field_by_id(51).is_some_and(|f| f.name == "price"),
        "caller id 51 (price) was not preserved as-is"
    );
    // The genuinely different column set: base ids 1/2 are not part of the new current schema.
    assert!(current.field_by_id(1).is_none() && current.field_by_id(2).is_none());

    // last_column_id advanced to the caller's highest id, never reduced below the base.
    assert_eq!(
        published.metadata().last_column_id(),
        51,
        "last_column_id must advance to the caller's max field-id"
    );
    assert!(published.metadata().last_column_id() >= after1.metadata().last_column_id());

    // Pre-replace snapshot S1 survives AND is still readable via its OWN schema (id/name),
    // unaffected by the current-schema swap.
    assert!(
        published.metadata().snapshot_by_id(s1).is_some(),
        "pre-replace snapshot S1 was lost"
    );
    let s1_schema = s1_schema_id
        .and_then(|id| published.metadata().schema_by_id(id))
        .expect("S1's schema must still be readable after the schema swap");
    assert!(
        s1_schema.field_by_id(1).is_some_and(|f| f.name == "id"),
        "S1's schema is no longer readable as id/name"
    );
    assert!(s1_schema.field_by_id(2).is_some_and(|f| f.name == "name"));
}
