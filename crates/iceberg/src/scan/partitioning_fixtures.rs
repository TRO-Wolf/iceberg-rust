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

//! Java-legal multi-spec fixtures for the PT unification campaign.
//!
//! Lives here (not in `scan/mod.rs`) so that 7055-line file does not grow.
//! Re-exported constructors hang off [`super::tests::TableTestFixture`].
//!
//! `TableTestFixture::new_with_two_identity_specs` is a Java-invalid unifier
//! input (duplicate field id 1000, different sources) and is NOT built on here.

#![allow(missing_docs)]

use std::fs;
use std::sync::Arc;

use minijinja::value::Value;
use minijinja::{AutoEscape, Environment, context};
use tempfile::TempDir;
use uuid::Uuid;

use super::tests::TableTestFixture;
use crate::TableIdent;
use crate::io::FileIO;
use crate::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, Literal, ManifestEntry, ManifestListWriter,
    ManifestStatus, ManifestWriterBuilder, PartitionSpec, Struct, StructType, TableMetadata,
    Transform, UnboundPartitionField,
};
use crate::table::Table;

fn render_template(template: &str, ctx: Value) -> String {
    let mut env = Environment::new();
    env.set_auto_escape_callback(|_| AutoEscape::None);
    env.render_str(template, ctx).unwrap()
}

fn load_template_metadata(table_location: &std::path::Path) -> (TableMetadata, std::path::PathBuf) {
    let manifest_list1_location = table_location.join("metadata/manifests_list_1.avro");
    let manifest_list2_location = table_location.join("metadata/manifests_list_2.avro");
    let table_metadata1_location = table_location.join("metadata/v1.json");
    let template_json_str = fs::read_to_string(format!(
        "{}/testdata/example_table_metadata_v2.json",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("example_table_metadata_v2.json");
    let metadata_json = render_template(&template_json_str, context! {
        table_location => &table_location,
        manifest_list_1_location => &manifest_list1_location,
        manifest_list_2_location => &manifest_list2_location,
        table_metadata_1_location => &table_metadata1_location,
    });
    let metadata = serde_json::from_str::<TableMetadata>(&metadata_json)
        .expect("parse example_table_metadata_v2");
    (metadata, table_metadata1_location)
}

fn build_fixture(
    table_location: std::path::PathBuf,
    table_metadata: TableMetadata,
) -> TableTestFixture {
    let file_io = FileIO::new_with_fs();
    let table = Table::builder()
        .metadata(table_metadata)
        .identifier(TableIdent::from_strs(["db", "table1"]).expect("ident"))
        .file_io(file_io)
        .metadata_location(table_metadata1_location_str(&table_location))
        .build()
        .expect("build table");
    TableTestFixture {
        table_location: table_location.to_str().expect("utf8 path").to_string(),
        table,
    }
}

fn table_metadata1_location_str(table_location: &std::path::Path) -> String {
    table_location
        .join("metadata/v1.json")
        .to_str()
        .expect("utf8 metadata path")
        .to_string()
}

impl TableTestFixture {
    /// Fixture 1 — Java-legal widening evolution.
    ///
    /// Spec 0: `identity(x)` @ field id 1000. Spec 1 (default): `identity(x)` @
    /// 1000 plus `bucket(y, 8)` @ 1001. Unified type is two fields; a spec-0
    /// file's 1-tuple coerces to `{x, null}`.
    pub fn new_with_widening_spec_evolution() -> Self {
        let tmp_dir = TempDir::new().expect("temp dir");
        let table_location = tmp_dir.path().join("table1");
        let (mut table_metadata, _meta_loc) = load_template_metadata(&table_location);

        let spec1 = Arc::new(
            PartitionSpec::builder(table_metadata.current_schema().clone())
                .with_spec_id(1)
                .add_unbound_fields(vec![
                    UnboundPartitionField {
                        source_id: 1,
                        field_id: Some(1000),
                        name: "x".to_string(),
                        transform: Transform::Identity,
                    },
                    UnboundPartitionField {
                        source_id: 2,
                        field_id: Some(1001),
                        name: "y_bucket_8".to_string(),
                        transform: Transform::Bucket(8),
                    },
                ])
                .expect("add spec 1 fields")
                .build()
                .expect("bind spec 1"),
        );
        let spec1_type = spec1
            .partition_type(table_metadata.current_schema())
            .expect("spec 1 partition type");
        table_metadata.default_spec = spec1.clone();
        table_metadata.default_partition_type = spec1_type;
        table_metadata.partition_specs.insert(1, spec1);
        table_metadata.last_partition_id = 1001;

        build_fixture(table_location, table_metadata)
    }

    /// Writes one file per spec for fixture 1, each in its own manifest.
    ///
    /// * `1.parquet` under spec 0, partition `x == 7`.
    /// * `2.parquet` under spec 1, partition `(x == 7, y_bucket_8 == 3)`.
    pub async fn setup_widening_spec_manifests(&mut self) {
        let current_snapshot = self
            .table
            .metadata()
            .current_snapshot()
            .expect("current snapshot")
            .clone();
        let current_schema = current_snapshot
            .schema(self.table.metadata())
            .expect("snapshot schema");
        let spec_zero = self
            .table
            .metadata()
            .partition_spec_by_id(0)
            .expect("spec 0")
            .clone();
        let spec_one = self
            .table
            .metadata()
            .partition_spec_by_id(1)
            .expect("spec 1")
            .clone();

        let manifest_a = {
            let mut writer = ManifestWriterBuilder::new(
                next_manifest_file(self),
                Some(current_snapshot.snapshot_id()),
                None,
                current_schema.clone(),
                spec_zero.as_ref().clone(),
            )
            .build_v2_data();
            writer
                .add_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Added)
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(0)
                                .content(DataContentType::Data)
                                .file_path(format!("{}/1.parquet", &self.table_location))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(1024)
                                .record_count(1)
                                .partition(Struct::from_iter([Some(Literal::long(7))]))
                                .key_metadata(None)
                                .build()
                                .expect("spec 0 data file"),
                        )
                        .build(),
                )
                .expect("add spec 0 entry");
            writer
                .write_manifest_file()
                .await
                .expect("write manifest a")
        };

        let manifest_b = {
            let mut writer = ManifestWriterBuilder::new(
                next_manifest_file(self),
                Some(current_snapshot.snapshot_id()),
                None,
                current_schema.clone(),
                spec_one.as_ref().clone(),
            )
            .build_v2_data();
            writer
                .add_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Added)
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(1)
                                .content(DataContentType::Data)
                                .file_path(format!("{}/2.parquet", &self.table_location))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(1024)
                                .record_count(1)
                                .partition(Struct::from_iter([
                                    Some(Literal::long(7)),
                                    Some(Literal::int(3)),
                                ]))
                                .key_metadata(None)
                                .build()
                                .expect("spec 1 data file"),
                        )
                        .build(),
                )
                .expect("add spec 1 entry");
            writer
                .write_manifest_file()
                .await
                .expect("write manifest b")
        };

        let mut manifest_list_write = ManifestListWriter::v2(
            self.table
                .file_io()
                .new_output(current_snapshot.manifest_list())
                .expect("manifest list output"),
            current_snapshot.snapshot_id(),
            current_snapshot.parent_snapshot_id(),
            current_snapshot.sequence_number(),
        );
        manifest_list_write
            .add_manifests(vec![manifest_a, manifest_b].into_iter())
            .expect("add manifests");
        manifest_list_write
            .close()
            .await
            .expect("close manifest list");
    }

    /// Fixture 5 — evolved to unpartitioned, with a live file under spec 0.
    ///
    /// Spec 0: `identity(x)` @ 1000. Spec 1 (default): unpartitioned. Unified
    /// type is non-empty, so `partitions` must keep `partition` / `spec_id`.
    pub fn new_evolved_to_unpartitioned() -> Self {
        let tmp_dir = TempDir::new().expect("temp dir");
        let table_location = tmp_dir.path().join("table1");
        let (mut table_metadata, _meta_loc) = load_template_metadata(&table_location);

        let unpartitioned_spec = Arc::new(
            PartitionSpec::builder(table_metadata.current_schema().clone())
                .with_spec_id(1)
                .build()
                .expect("unpartitioned spec 1"),
        );
        table_metadata.default_spec = unpartitioned_spec.clone();
        table_metadata.default_partition_type = StructType::new(vec![]);
        table_metadata.partition_specs.insert(1, unpartitioned_spec);

        build_fixture(table_location, table_metadata)
    }

    /// Writes one live data file under spec 0 (`x == 7`) for fixture 5.
    pub async fn setup_evolved_to_unpartitioned_file(&mut self) {
        let current_snapshot = self
            .table
            .metadata()
            .current_snapshot()
            .expect("current snapshot")
            .clone();
        let current_schema = current_snapshot
            .schema(self.table.metadata())
            .expect("snapshot schema");
        let spec_zero = self
            .table
            .metadata()
            .partition_spec_by_id(0)
            .expect("spec 0")
            .clone();

        let mut writer = ManifestWriterBuilder::new(
            next_manifest_file(self),
            Some(current_snapshot.snapshot_id()),
            None,
            current_schema,
            spec_zero.as_ref().clone(),
        )
        .build_v2_data();
        writer
            .add_entry(
                ManifestEntry::builder()
                    .status(ManifestStatus::Added)
                    .data_file(
                        DataFileBuilder::default()
                            .partition_spec_id(0)
                            .content(DataContentType::Data)
                            .file_path(format!("{}/1.parquet", &self.table_location))
                            .file_format(DataFileFormat::Parquet)
                            .file_size_in_bytes(1024)
                            .record_count(1)
                            .partition(Struct::from_iter([Some(Literal::long(7))]))
                            .key_metadata(None)
                            .build()
                            .expect("spec 0 data file"),
                    )
                    .build(),
            )
            .expect("add entry");
        let manifest = writer.write_manifest_file().await.expect("write manifest");

        let mut manifest_list_write = ManifestListWriter::v2(
            self.table
                .file_io()
                .new_output(current_snapshot.manifest_list())
                .expect("manifest list output"),
            current_snapshot.snapshot_id(),
            current_snapshot.parent_snapshot_id(),
            current_snapshot.sequence_number(),
        );
        manifest_list_write
            .add_manifests(vec![manifest].into_iter())
            .expect("add manifest");
        manifest_list_write
            .close()
            .await
            .expect("close manifest list");
    }
}

fn next_manifest_file(fixture: &TableTestFixture) -> crate::io::OutputFile {
    fixture
        .table
        .file_io()
        .new_output(format!(
            "{}/metadata/manifest_{}.avro",
            fixture.table_location,
            Uuid::new_v4()
        ))
        .expect("manifest output")
}
