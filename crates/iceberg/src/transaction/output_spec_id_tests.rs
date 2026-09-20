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

use crate::io::LocalFsStorageFactory;
use crate::memory::MemoryCatalogBuilder;
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, Literal,
    ManifestEntryRef, NestedField, PartitionSpec, PrimitiveType, Schema, Struct, Transform, Type,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation};

pub(crate) fn oracle_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "cat", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("oracle schema")
}

pub(crate) fn unpartitioned_spec(schema: &Schema) -> PartitionSpec {
    PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .build()
        .expect("unpartitioned spec")
}

pub(crate) fn cat_spec(schema: &Schema) -> PartitionSpec {
    PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("cat", "cat", Transform::Identity)
        .expect("identity(cat) partition field")
        .build()
        .expect("cat spec")
}

pub(crate) async fn local_catalog() -> (impl Catalog, TempDir) {
    let temp_dir = TempDir::new().expect("temp dir");
    let warehouse = temp_dir.path().to_str().expect("utf8 path").to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([("warehouse".to_string(), warehouse)]),
        )
        .await
        .expect("local-fs memory catalog");
    (catalog, temp_dir)
}

pub(crate) async fn create_oracle_table(
    catalog: &impl Catalog,
    spec: PartitionSpec,
    format_version: FormatVersion,
) -> Table {
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("t".to_string())
        .schema(oracle_schema())
        .partition_spec(spec)
        .format_version(format_version)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

pub(crate) async fn evolve_spec(
    catalog: &impl Catalog,
    table: &Table,
    update: impl FnOnce(
        crate::transaction::update_partition_spec::UpdatePartitionSpecAction,
    ) -> crate::transaction::update_partition_spec::UpdatePartitionSpecAction,
) -> Table {
    let tx = Transaction::new(table);
    let tx = update(tx.update_partition_spec()).apply(tx).unwrap();
    tx.commit(catalog).await.expect("spec evolution commit")
}

pub(crate) fn cat_partition(cat: &str) -> Struct {
    Struct::from_iter([Some(Literal::string(cat.to_string()))])
}

pub(crate) fn oracle_file(
    table: &Table,
    name: &str,
    spec_id: i32,
    partition: Struct,
    record_count: u64,
) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(format!("{}/data/{name}", table.metadata().location()))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(record_count)
        .partition_spec_id(spec_id)
        .partition(partition)
        .build()
        .expect("data file")
}

pub(crate) async fn snapshot_manifests(table: &Table) -> Vec<(i32, Vec<ManifestEntryRef>)> {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("table has a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list loads");
    let mut manifests = Vec::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest loads");
        manifests.push((manifest_file.partition_spec_id, manifest.entries().to_vec()));
    }
    manifests
}

pub(crate) fn assert_manifest_specs(manifests: &[(i32, Vec<ManifestEntryRef>)]) {
    for (spec_id, entries) in manifests {
        for entry in entries {
            assert_eq!(
                entry.data_file().partition_spec_id,
                *spec_id,
                "manifest stamped spec {spec_id} holds an entry stamped spec {}",
                entry.data_file().partition_spec_id,
            );
        }
    }
}

pub(crate) fn live_files(manifests: &[(i32, Vec<ManifestEntryRef>)]) -> Vec<(i32, Struct, u64)> {
    let mut files = Vec::new();
    for (spec_id, entries) in manifests {
        for entry in entries {
            if entry.is_alive() && entry.content_type() == DataContentType::Data {
                files.push((
                    *spec_id,
                    entry.data_file().partition().clone(),
                    entry.data_file().record_count(),
                ));
            }
        }
    }
    files
}

async fn add_cat_then_commit_mixed(
    catalog: &impl Catalog,
    format_version: FormatVersion,
    commit: impl FnOnce(Transaction, Vec<DataFile>) -> crate::Result<Transaction>,
) -> Table {
    let table = create_oracle_table(
        catalog,
        unpartitioned_spec(&oracle_schema()),
        format_version,
    )
    .await;
    let table = evolve_spec(catalog, &table, |action| action.add_field("cat")).await;
    assert_eq!(table.metadata().default_partition_spec_id(), 1);

    let spec0 = oracle_file(&table, "os-spec0.parquet", 0, Struct::empty(), 2);
    let spec1 = oracle_file(&table, "os-spec1.parquet", 1, cat_partition("w"), 1);
    let tx = Transaction::new(&table);
    let tx = commit(tx, vec![spec0, spec1]).expect("apply");
    tx.commit(catalog).await.expect("mixed-spec commit")
}

async fn mixed_spec_table(format_version: FormatVersion) -> (impl Catalog, TempDir, Table) {
    let (catalog, guard) = local_catalog().await;
    let table = create_oracle_table(
        &catalog,
        unpartitioned_spec(&oracle_schema()),
        format_version,
    )
    .await;
    let table = evolve_spec(&catalog, &table, |action| action.add_field("cat")).await;
    assert_eq!(table.metadata().default_partition_spec_id(), 1);
    (catalog, guard, table)
}

async fn measure_mixed_spec_commit(
    format_version: FormatVersion,
    commit: impl FnOnce(Transaction, Vec<DataFile>) -> crate::Result<Transaction>,
) -> Vec<(i32, Vec<ManifestEntryRef>)> {
    let (catalog, _guard) = local_catalog().await;
    let table = add_cat_then_commit_mixed(&catalog, format_version, commit).await;
    snapshot_manifests(&table).await
}

#[tokio::test]
async fn measure_fast_append_mixed_specs_groups_manifests_per_spec() {
    let manifests = measure_mixed_spec_commit(FormatVersion::V2, |tx, files| {
        tx.fast_append().add_data_files(files).apply(tx)
    })
    .await;
    assert_manifest_specs(&manifests);
    assert_eq!(manifests.len(), 2);

    let mut files = live_files(&manifests);
    files.sort_by_key(|(spec_id, _, _)| *spec_id);
    assert_eq!(files, vec![
        (0, Struct::empty(), 2),
        (1, cat_partition("w"), 1),
    ]);
}

#[tokio::test]
async fn measure_merge_append_mixed_specs_groups_manifests_per_spec() {
    let manifests = measure_mixed_spec_commit(FormatVersion::V2, |tx, files| {
        tx.merge_append().add_data_files(files).apply(tx)
    })
    .await;
    assert_manifest_specs(&manifests);
    assert_eq!(manifests.len(), 2);

    let mut files = live_files(&manifests);
    files.sort_by_key(|(spec_id, _, _)| *spec_id);
    assert_eq!(files, vec![
        (0, Struct::empty(), 2),
        (1, cat_partition("w"), 1),
    ]);
}

#[tokio::test]
async fn measure_overwrite_files_mixed_specs_groups_manifests_per_spec() {
    let (catalog, _guard, table) = mixed_spec_table(FormatVersion::V2).await;
    let seed = oracle_file(&table, "seed.parquet", 1, cat_partition("y"), 1);
    let seed_path = seed.file_path.clone();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![seed])
        .apply(tx)
        .unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    let spec0 = oracle_file(&table, "os-spec0.parquet", 0, Struct::empty(), 2);
    let spec1 = oracle_file(&table, "os-spec1.parquet", 1, cat_partition("w"), 1);
    let tx = Transaction::new(&table);
    let tx = tx
        .overwrite_files()
        .delete_file(seed_path)
        .add_files(vec![spec0, spec1])
        .apply(tx)
        .unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    let manifests = snapshot_manifests(&table).await;
    assert_manifest_specs(&manifests);
    let mut files = live_files(&manifests);
    files.sort_by_key(|(spec_id, _, _)| *spec_id);
    assert_eq!(files, vec![
        (0, Struct::empty(), 2),
        (1, cat_partition("w"), 1),
    ]);
}

#[tokio::test]
async fn measure_replace_partitions_mixed_specs_groups_manifests_per_spec() {
    let (catalog, _guard, table) = mixed_spec_table(FormatVersion::V2).await;
    let seed = oracle_file(&table, "seed.parquet", 1, cat_partition("y"), 1);
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![seed])
        .apply(tx)
        .unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    let spec0 = oracle_file(&table, "os-spec0.parquet", 0, Struct::empty(), 2);
    let spec1 = oracle_file(&table, "os-spec1.parquet", 1, cat_partition("y"), 3);
    let tx = Transaction::new(&table);
    let tx = tx
        .replace_partitions()
        .add_files(vec![spec0, spec1])
        .apply(tx)
        .unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    let manifests = snapshot_manifests(&table).await;
    assert_manifest_specs(&manifests);
    let mut files = live_files(&manifests);
    files.sort_by_key(|(spec_id, _, _)| *spec_id);
    assert_eq!(files, vec![
        (0, Struct::empty(), 2),
        (1, cat_partition("y"), 3),
    ]);
}

#[tokio::test]
async fn measure_older_partitioned_spec_file_validates_against_its_own_spec() {
    let (catalog, _guard) = local_catalog().await;
    let table = create_oracle_table(&catalog, cat_spec(&oracle_schema()), FormatVersion::V2).await;
    let table = evolve_spec(&catalog, &table, |action| action.remove_field("cat")).await;
    assert_eq!(table.metadata().default_partition_spec_id(), 1);
    assert!(
        table.metadata().default_partition_spec().is_unpartitioned(),
        "dropping the only field leaves an empty default spec"
    );

    let spec0 = oracle_file(&table, "os-spec0.parquet", 0, cat_partition("w"), 1);
    let spec1 = oracle_file(&table, "os-spec1.parquet", 1, Struct::empty(), 1);
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![spec0, spec1])
        .apply(tx)
        .unwrap();
    let table = tx.commit(&catalog).await.expect(
        "a spec-0 file's arity-1 tuple must validate against spec 0, not the empty default",
    );

    let manifests = snapshot_manifests(&table).await;
    assert_manifest_specs(&manifests);
    let mut files = live_files(&manifests);
    files.sort_by_key(|(spec_id, _, _)| *spec_id);
    assert_eq!(files, vec![
        (0, cat_partition("w"), 1),
        (1, Struct::empty(), 1),
    ]);
}

#[tokio::test]
async fn measure_added_file_partition_validated_against_its_own_spec() {
    let (catalog, _guard, table) = mixed_spec_table(FormatVersion::V2).await;
    let file = oracle_file(&table, "bad-arity.parquet", 0, cat_partition("w"), 1);
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file])
        .apply(tx)
        .unwrap();
    let err = tx
        .commit(&catalog)
        .await
        .expect_err("a spec-0 file with a spec-1 tuple must fail its own spec's arity");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
}

#[tokio::test]
async fn measure_added_file_unknown_spec_id_fails() {
    let (catalog, _guard, table) = mixed_spec_table(FormatVersion::V2).await;
    let file = oracle_file(&table, "bad-spec.parquet", 9, cat_partition("w"), 1);
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file])
        .apply(tx)
        .unwrap();
    let err = tx
        .commit(&catalog)
        .await
        .expect_err("a file stamped with a spec id the table does not have must fail");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(
        err.message()
            .contains("Cannot find partition spec 9 for data file"),
        "unexpected message: {}",
        err.message()
    );
}

#[tokio::test]
async fn resolve_output_spec_none_resolves_table_default() {
    let (_catalog, _guard, table) = mixed_spec_table(FormatVersion::V2).await;
    let spec = crate::writer::resolve_output_spec(&table, None).expect("default resolves");
    assert_eq!(spec.spec_id(), table.metadata().default_partition_spec_id());
}

#[tokio::test]
async fn resolve_output_spec_returns_older_spec() {
    let (_catalog, _guard, table) = mixed_spec_table(FormatVersion::V2).await;
    let spec = crate::writer::resolve_output_spec(&table, Some(0)).expect("spec 0 resolves");
    assert_eq!(spec.spec_id(), 0);
    assert!(spec.is_unpartitioned());
}

#[tokio::test]
async fn resolve_output_spec_unknown_id_is_data_invalid() {
    let (_catalog, _guard, table) = mixed_spec_table(FormatVersion::V2).await;
    let err =
        crate::writer::resolve_output_spec(&table, Some(9)).expect_err("an unknown spec id fails");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        err.message(),
        "Output spec id 9 is not a valid spec id for table"
    );
}
