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

use std::fs;
use std::sync::Arc;

use bimap::BiHashMap;
use minijinja::value::Value;
use minijinja::{AutoEscape, Environment, context};
use tempfile::TempDir;
use uuid::Uuid;

use super::*;
use crate::TableIdent;
use crate::io::{FileIO, OutputFile};
use crate::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, Datum, FieldSummary, INITIAL_SEQUENCE_NUMBER,
    Literal, ManifestContentType, ManifestEntry, ManifestListWriter, ManifestStatus,
    ManifestWriterBuilder, Map, NestedField, PartitionSpec, PrimitiveType, Schema, Snapshot,
    Struct, TableMetadata, Transform, Type,
};
use crate::table::Table;

fn render_template(template: &str, ctx: Value) -> String {
    let mut env = Environment::new();
    env.set_auto_escape_callback(|_| AutoEscape::None);
    env.render_str(template, ctx).unwrap()
}

struct TableTestFixture {
    table_location: String,
    table: Table,
}

impl TableTestFixture {
    fn new() -> Self {
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().join("table1");
        let manifest_list1_location = table_location.join("metadata/manifests_list_1.avro");
        let manifest_list2_location = table_location.join("metadata/manifests_list_2.avro");
        let table_metadata1_location = table_location.join("metadata/v1.json");

        let file_io = FileIO::new_with_fs();

        let table_metadata = {
            let template_json_str = fs::read_to_string(format!(
                "{}/testdata/example_table_metadata_v2.json",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            let metadata_json = render_template(&template_json_str, context! {
                table_location => &table_location,
                manifest_list_1_location => &manifest_list1_location,
                manifest_list_2_location => &manifest_list2_location,
                table_metadata_1_location => &table_metadata1_location,
            });
            serde_json::from_str::<TableMetadata>(&metadata_json).unwrap()
        };

        let table = Table::builder()
            .metadata(table_metadata)
            .identifier(TableIdent::from_strs(["db", "table1"]).unwrap())
            .file_io(file_io.clone())
            .metadata_location(table_metadata1_location.as_os_str().to_str().unwrap())
            .build()
            .unwrap();

        Self {
            table_location: table_location.to_str().unwrap().to_string(),
            table,
        }
    }

    fn next_manifest_file(&self) -> OutputFile {
        self.table
            .file_io()
            .new_output(format!(
                "{}/metadata/manifest_{}.avro",
                self.table_location,
                Uuid::new_v4()
            ))
            .unwrap()
    }

    async fn setup_manifest_files(&mut self) {
        let current_snapshot = self.table.metadata().current_snapshot().unwrap();
        let current_schema = current_snapshot.schema(self.table.metadata()).unwrap();
        let current_partition_spec = self.table.metadata().default_partition_spec();

        // Write data files
        let mut writer = ManifestWriterBuilder::new(
            self.next_manifest_file(),
            Some(current_snapshot.snapshot_id()),
            None,
            current_schema.clone(),
            current_partition_spec.as_ref().clone(),
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
                            .file_size_in_bytes(100)
                            .record_count(1)
                            .partition(Struct::from_iter([Some(Literal::long(100))]))
                            .build()
                            .unwrap(),
                    )
                    .build(),
            )
            .unwrap();
        let data_file_manifest = writer.write_manifest_file().await.unwrap();

        // Write to manifest list
        let mut manifest_list_write = ManifestListWriter::v2(
            self.table
                .file_io()
                .new_output(current_snapshot.manifest_list())
                .unwrap(),
            current_snapshot.snapshot_id(),
            current_snapshot.parent_snapshot_id(),
            current_snapshot.sequence_number(),
        );
        manifest_list_write
            .add_manifests(vec![data_file_manifest].into_iter())
            .unwrap();
        manifest_list_write.close().await.unwrap();
    }

    async fn write_two_entry_manifest(&self) -> ManifestFile {
        let current_snapshot = self
            .table
            .metadata()
            .current_snapshot()
            .expect("fixture must have a current snapshot");
        let current_schema = current_snapshot
            .schema(self.table.metadata())
            .expect("fixture snapshot must resolve its schema");
        let current_partition_spec = self.table.metadata().default_partition_spec();

        let mut writer = ManifestWriterBuilder::new(
            self.next_manifest_file(),
            None,
            None,
            current_schema.clone(),
            current_partition_spec.as_ref().clone(),
        )
        .build_v2_data();
        for (name, record_count) in [("10.parquet", 3u64), ("11.parquet", 5u64)] {
            writer
                .add_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Added)
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(0)
                                .content(DataContentType::Data)
                                .file_path(format!("{}/{name}", &self.table_location))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(100)
                                .record_count(record_count)
                                .partition(Struct::from_iter([Some(Literal::long(100))]))
                                .build()
                                .expect("data file must build"),
                        )
                        .build(),
                )
                .expect("entry must append");
        }
        writer
            .write_manifest_file()
            .await
            .expect("manifest must write")
    }
}

#[tokio::test]
async fn test_get_manifest_list_and_manifest_from_disabled_cache() {
    let mut fixture = TableTestFixture::new();
    fixture.setup_manifest_files().await;

    let object_cache = ObjectCache::with_disabled_cache(fixture.table.file_io().clone());

    let result_manifest_list = object_cache
        .get_manifest_list(
            fixture.table.metadata().current_snapshot().unwrap(),
            &fixture.table.metadata_ref(),
        )
        .await
        .unwrap();

    assert_eq!(result_manifest_list.entries().len(), 1);

    let manifest_file = result_manifest_list.entries().first().unwrap();
    let result_manifest = object_cache
        .get_manifest(manifest_file, None)
        .await
        .unwrap();

    assert_eq!(
        result_manifest
            .entries()
            .first()
            .unwrap()
            .file_path()
            .split("/")
            .last()
            .unwrap(),
        "1.parquet"
    );
}

/// SAF-001: a V1/legacy snapshot may omit `schema_id` (`Snapshot::schema_id` is `Option`).
/// The default (enabled) `ObjectCache` must build its manifest-list key and load the list
/// without panicking — matching the cache-disabled path, which already handles `None`.
///
/// MUTATION (restore the `SchemaId` key element and `snapshot.schema_id().unwrap()` in
/// `get_manifest_list`): key construction panics on this schema-id-less snapshot, before any
/// I/O, so this test aborts instead of returning the manifest list.
#[tokio::test]
async fn test_get_manifest_list_from_default_cache_with_schemaless_snapshot() {
    let mut fixture = TableTestFixture::new();
    fixture.setup_manifest_files().await;

    let current = fixture.table.metadata().current_snapshot().unwrap();
    // A legacy snapshot that omits `schema_id`, reusing the real manifest-list path.
    let schemaless = Arc::new(
        Snapshot::builder()
            .with_snapshot_id(current.snapshot_id())
            .with_sequence_number(current.sequence_number())
            .with_timestamp_ms(current.timestamp_ms())
            .with_manifest_list(current.manifest_list())
            .with_summary(current.summary().clone())
            .build(),
    );
    assert_eq!(schemaless.schema_id(), None);

    let object_cache = ObjectCache::new(fixture.table.file_io().clone());
    let manifest_list = object_cache
        .get_manifest_list(&schemaless, &fixture.table.metadata_ref())
        .await
        .unwrap();

    assert_eq!(manifest_list.entries().len(), 1);
}

#[test]
fn test_clamp_cache_weight_floor_and_cap() {
    assert_eq!(clamp_cache_weight(0), 1);
    assert_eq!(clamp_cache_weight(1), 1);
    assert_eq!(clamp_cache_weight(u32::MAX as u64), u32::MAX);
    assert_eq!(clamp_cache_weight(u32::MAX as u64 + 1), u32::MAX);
    assert_eq!(clamp_cache_weight(100), 100);
    assert_eq!(
        clamp_cache_weight(sequence_charge::<u128>(usize::MAX)),
        u32::MAX
    );
}

#[test]
fn test_capacity_api_preserves_zero_and_values_above_item_weight_width() {
    let file_io = FileIO::new_with_fs();
    let large_capacity = u64::from(u32::MAX) + 1;
    let large = ObjectCache::new_with_capacity(file_io.clone(), large_capacity);
    assert_eq!(large.cache.policy().max_capacity(), Some(large_capacity));

    let disabled = ObjectCache::new_with_capacity(file_io, 0);
    assert!(disabled.cache_disabled);
    assert_eq!(disabled.cache.policy().max_capacity(), Some(0));
}

#[tokio::test]
async fn test_loaded_manifest_and_list_charges_include_their_graphs() {
    let mut fixture = TableTestFixture::new();
    fixture.setup_manifest_files().await;

    let object_cache = ObjectCache::new(fixture.table.file_io().clone());
    let manifest_list = object_cache
        .get_manifest_list(
            fixture
                .table
                .metadata()
                .current_snapshot()
                .expect("fixture must have a current snapshot"),
            &fixture.table.metadata_ref(),
        )
        .await
        .expect("manifest list must load");

    let list_charge = manifest_list_charge(&manifest_list);
    assert!(list_charge > shallow_charge::<ManifestList>());
    let entry = manifest_list
        .entries()
        .first()
        .expect("fixture list has one entry");

    let manifest = object_cache
        .get_manifest(entry, None)
        .await
        .expect("manifest must load");
    let manifest_charge = manifest_charge(&manifest);
    let empty_manifest = Manifest::new(manifest.metadata().clone(), Vec::new());
    assert!(manifest_charge > shallow_charge::<Manifest>());
    assert!(manifest_charge > super::manifest_charge(&empty_manifest));
    assert!(manifest_charge > list_charge);
}

#[tokio::test]
async fn test_a_budget_below_retained_charge_evicts_entries() {
    let mut fixture = TableTestFixture::new();
    fixture.setup_manifest_files().await;

    let fixture_cache = ObjectCache::new(fixture.table.file_io().clone());
    let manifest_list = fixture_cache
        .get_manifest_list(
            fixture.table.metadata().current_snapshot().unwrap(),
            &fixture.table.metadata_ref(),
        )
        .await
        .unwrap();
    let manifest = fixture_cache
        .get_manifest(manifest_list.entries().first().unwrap(), None)
        .await
        .unwrap();

    let object_cache = ObjectCache::new_with_capacity(fixture.table.file_io().clone(), 280_000);
    for index in 0..256 {
        object_cache
            .cache
            .insert(
                CachedObjectKey::ManifestList((
                    format!("manifest-list-{index}"),
                    FormatVersion::V2,
                    Some(0),
                )),
                CachedItem::ManifestList(Arc::new(manifest_list.as_ref().clone())),
            )
            .await;
        object_cache
            .cache
            .insert(
                CachedObjectKey::Manifest((format!("manifest-{index}"), None)),
                CachedItem::RawManifest(Arc::new(manifest.as_ref().clone())),
            )
            .await;
    }
    object_cache.cache.run_pending_tasks().await;

    assert!(object_cache.cache.entry_count() > 0);
    assert!(object_cache.cache.entry_count() < 512);
    assert!(object_cache.cache.weighted_size() <= 280_000);
}

#[test]
fn test_cache_key_charge_tracks_path_allocation() {
    let short = CachedObjectKey::Manifest((String::from("m"), None));
    let long = CachedObjectKey::Manifest(("m".repeat(4096), None));

    assert!(cache_key_charge(&long) > cache_key_charge(&short));
}

#[tokio::test]
async fn test_manifest_charge_tracks_variable_data_file_payloads() {
    let mut fixture = TableTestFixture::new();
    fixture.setup_manifest_files().await;
    let object_cache = ObjectCache::new(fixture.table.file_io().clone());
    let manifest_list = object_cache
        .get_manifest_list(
            fixture.table.metadata().current_snapshot().unwrap(),
            &fixture.table.metadata_ref(),
        )
        .await
        .unwrap();
    let manifest = object_cache
        .get_manifest(manifest_list.entries().first().unwrap(), None)
        .await
        .unwrap();
    let baseline_charge = manifest_charge(&manifest);
    let baseline_entry = manifest.entries().first().unwrap().as_ref();

    let mut path = baseline_entry.data_file().clone();
    path.file_path = "p".repeat(4096);

    let mut column_sizes = baseline_entry.data_file().clone();
    let mut value_counts = baseline_entry.data_file().clone();
    let mut null_value_counts = baseline_entry.data_file().clone();
    let mut nan_value_counts = baseline_entry.data_file().clone();
    for field_id in 0..64 {
        column_sizes.column_sizes.insert(field_id, 1);
        value_counts.value_counts.insert(field_id, 1);
        null_value_counts.null_value_counts.insert(field_id, 1);
        nan_value_counts.nan_value_counts.insert(field_id, 1);
    }

    let mut lower_bounds = baseline_entry.data_file().clone();
    let mut upper_bounds = baseline_entry.data_file().clone();
    for field_id in 0..64 {
        lower_bounds
            .lower_bounds
            .insert(field_id, Datum::string("l".repeat(128)));
        upper_bounds
            .upper_bounds
            .insert(field_id, Datum::binary(vec![1; 128]));
    }

    let mut string_partition = baseline_entry.data_file().clone();
    string_partition.partition = Struct::from_iter([Some(Literal::string("s".repeat(4096)))]);
    let mut list_partition = baseline_entry.data_file().clone();
    list_partition.partition =
        Struct::from_iter([Some(Literal::List(vec![Some(Literal::binary(vec![
            1;
            4096
        ]))]))]);
    let mut map_partition = baseline_entry.data_file().clone();
    map_partition.partition = Struct::from_iter([Some(Literal::Map(Map::from([(
        Literal::string("k".repeat(4096)),
        Some(Literal::string("v".repeat(4096))),
    )])))]);

    let mut key_metadata = baseline_entry.data_file().clone();
    key_metadata.key_metadata = Some(vec![1; 4096]);
    let mut split_offsets = baseline_entry.data_file().clone();
    split_offsets.split_offsets = Some(vec![1; 512]);
    let mut equality_ids = baseline_entry.data_file().clone();
    equality_ids.equality_ids = Some(vec![1; 1024]);
    let mut referenced_data_file = baseline_entry.data_file().clone();
    referenced_data_file.referenced_data_file = Some("r".repeat(4096));

    for data_file in [
        path,
        column_sizes,
        value_counts,
        null_value_counts,
        nan_value_counts,
        lower_bounds,
        upper_bounds,
        string_partition,
        list_partition,
        map_partition,
        key_metadata,
        split_offsets,
        equality_ids,
        referenced_data_file,
    ] {
        let mut entry = baseline_entry.clone();
        entry.data_file = data_file;
        let changed = Manifest::new(manifest.metadata().clone(), vec![entry]);
        assert!(manifest_charge(&changed) > baseline_charge);
    }
}

#[tokio::test]
async fn test_manifest_charge_tracks_schema_payloads() {
    let mut fixture = TableTestFixture::new();
    fixture.setup_manifest_files().await;
    let object_cache = ObjectCache::new(fixture.table.file_io().clone());
    let manifest_list = object_cache
        .get_manifest_list(
            fixture.table.metadata().current_snapshot().unwrap(),
            &fixture.table.metadata_ref(),
        )
        .await
        .unwrap();
    let manifest = object_cache
        .get_manifest(manifest_list.entries().first().unwrap(), None)
        .await
        .unwrap();
    let build_schema = |name: String, doc: String, initial: String, write: String| {
        let field = NestedField::optional(1, name, Type::Primitive(PrimitiveType::String))
            .with_doc(doc)
            .with_initial_default(Literal::string(initial))
            .with_write_default(Literal::string(write));
        Schema::builder()
            .with_fields([Arc::new(field)])
            .build()
            .unwrap()
    };
    let short = build_schema(
        String::from("n"),
        String::from("d"),
        String::from("i"),
        String::from("w"),
    );
    for schema in [
        build_schema(
            "n".repeat(4096),
            String::from("d"),
            String::from("i"),
            String::from("w"),
        ),
        build_schema(
            String::from("n"),
            "d".repeat(4096),
            String::from("i"),
            String::from("w"),
        ),
        build_schema(
            String::from("n"),
            String::from("d"),
            "i".repeat(4096),
            String::from("w"),
        ),
        build_schema(
            String::from("n"),
            String::from("d"),
            String::from("i"),
            "w".repeat(4096),
        ),
    ] {
        assert!(schema_charge(&schema) > schema_charge(&short));
    }
    let aliased_schema = short
        .clone()
        .into_builder()
        .with_alias(BiHashMap::from_iter([("a".repeat(4096), 1)]))
        .build()
        .unwrap();
    assert!(schema_charge(&aliased_schema) > schema_charge(&short));
    let entries = manifest
        .entries()
        .iter()
        .map(|entry| entry.as_ref().clone())
        .collect::<Vec<_>>();
    let mut short_metadata = manifest.metadata().clone();
    short_metadata.schema = Arc::new(short);
    let short_manifest = Manifest::new(short_metadata, entries.clone());
    let mut changed_metadata = manifest.metadata().clone();
    changed_metadata.schema = Arc::new(aliased_schema);
    let changed = Manifest::new(changed_metadata, entries);

    assert!(manifest_charge(&changed) > manifest_charge(&short_manifest));

    let schema = manifest.metadata().schema.clone();
    let source_name = &schema.as_struct().fields()[0].name;
    let short_spec = PartitionSpec::builder(schema.clone())
        .add_partition_field(source_name, "p", Transform::Identity)
        .unwrap()
        .build()
        .unwrap();
    let long_spec = PartitionSpec::builder(schema.clone())
        .add_partition_field(source_name, "p".repeat(4096), Transform::Identity)
        .unwrap()
        .build()
        .unwrap();
    let mut short_spec_metadata = manifest.metadata().clone();
    short_spec_metadata.partition_spec = short_spec;
    let short_spec_manifest = Manifest::new(short_spec_metadata, Vec::new());
    let mut long_spec_metadata = manifest.metadata().clone();
    long_spec_metadata.partition_spec = long_spec;
    let long_spec_manifest = Manifest::new(long_spec_metadata, Vec::new());
    assert!(manifest_charge(&long_spec_manifest) > manifest_charge(&short_spec_manifest));
}

#[tokio::test]
async fn test_manifest_file_charge_tracks_list_payloads() {
    let mut fixture = TableTestFixture::new();
    fixture.setup_manifest_files().await;
    let object_cache = ObjectCache::new(fixture.table.file_io().clone());
    let manifest_list = object_cache
        .get_manifest_list(
            fixture.table.metadata().current_snapshot().unwrap(),
            &fixture.table.metadata_ref(),
        )
        .await
        .unwrap();
    let baseline = manifest_list.entries().first().unwrap();
    let baseline_charge = manifest_file_payload_charge(baseline);

    let mut path = baseline.clone();
    path.manifest_path = "p".repeat(4096);
    assert!(manifest_file_payload_charge(&path) > baseline_charge);

    let mut child_length = baseline.clone();
    let unchanged_charge = manifest_file_payload_charge(&child_length);
    child_length.manifest_length = i64::MAX;
    assert_eq!(
        manifest_file_payload_charge(&child_length),
        unchanged_charge
    );

    let mut summaries = baseline.clone();
    summaries.partitions = Some(vec![FieldSummary {
        contains_null: false,
        contains_nan: Some(false),
        lower_bound: Some(vec![1; 4096].into()),
        upper_bound: Some(vec![2; 4096].into()),
    }]);
    assert!(manifest_file_payload_charge(&summaries) > baseline_charge);

    let mut metadata = baseline.clone();
    metadata.key_metadata = Some(vec![1; 4096]);
    assert!(manifest_file_payload_charge(&metadata) > baseline_charge);
}

#[tokio::test]
async fn test_oversized_charge_is_rejected_and_a_smaller_item_remains_cacheable() {
    let mut fixture = TableTestFixture::new();
    fixture.setup_manifest_files().await;
    let fixture_cache = ObjectCache::new(fixture.table.file_io().clone());
    let manifest_list = fixture_cache
        .get_manifest_list(
            fixture.table.metadata().current_snapshot().unwrap(),
            &fixture.table.metadata_ref(),
        )
        .await
        .unwrap();
    let item = CachedItem::ManifestList(manifest_list);
    let small_key =
        CachedObjectKey::ManifestList((String::from("small"), FormatVersion::V2, Some(0)));
    let capacity = cached_object_charge(&small_key, &item);
    let object_cache =
        ObjectCache::new_with_capacity(fixture.table.file_io().clone(), capacity.into());
    let large_key =
        CachedObjectKey::ManifestList(("large".repeat(4096), FormatVersion::V2, Some(0)));

    object_cache.cache.insert(large_key, item.clone()).await;
    object_cache.cache.run_pending_tasks().await;
    assert_eq!(object_cache.cache.entry_count(), 0);

    object_cache.cache.insert(small_key, item).await;
    object_cache.cache.run_pending_tasks().await;
    assert_eq!(object_cache.cache.entry_count(), 1);
    assert_eq!(object_cache.cache.weighted_size(), u64::from(capacity));
}

#[tokio::test]
async fn test_get_manifest_list_and_manifest_from_default_cache() {
    let mut fixture = TableTestFixture::new();
    fixture.setup_manifest_files().await;

    let object_cache = ObjectCache::new(fixture.table.file_io().clone());

    // not in cache
    let result_manifest_list = object_cache
        .get_manifest_list(
            fixture.table.metadata().current_snapshot().unwrap(),
            &fixture.table.metadata_ref(),
        )
        .await
        .unwrap();

    assert_eq!(result_manifest_list.entries().len(), 1);

    // retrieve cached version
    let result_manifest_list = object_cache
        .get_manifest_list(
            fixture.table.metadata().current_snapshot().unwrap(),
            &fixture.table.metadata_ref(),
        )
        .await
        .unwrap();

    assert_eq!(result_manifest_list.entries().len(), 1);

    let manifest_file = result_manifest_list.entries().first().unwrap();

    // not in cache
    let result_manifest = object_cache
        .get_manifest(manifest_file, None)
        .await
        .unwrap();

    assert_eq!(
        result_manifest
            .entries()
            .first()
            .unwrap()
            .file_path()
            .split("/")
            .last()
            .unwrap(),
        "1.parquet"
    );

    // retrieve cached version
    let result_manifest = object_cache
        .get_manifest(manifest_file, None)
        .await
        .unwrap();

    assert_eq!(
        result_manifest
            .entries()
            .first()
            .unwrap()
            .file_path()
            .split("/")
            .last()
            .unwrap(),
        "1.parquet"
    );
}

#[tokio::test]
async fn test_get_manifest_rangeless_then_ranged_serves_each_callers_context() {
    let fixture = TableTestFixture::new();
    let base = fixture.write_two_entry_manifest().await;

    let v2_entry = ManifestFile {
        sequence_number: 11,
        added_snapshot_id: 1001,
        first_row_id: None,
        ..base.clone()
    };
    let v3_entry = ManifestFile {
        sequence_number: 22,
        added_snapshot_id: 2002,
        first_row_id: Some(100),
        ..base
    };

    let object_cache = ObjectCache::new(fixture.table.file_io().clone());

    let first = object_cache
        .get_manifest(&v2_entry, None)
        .await
        .expect("rangeless read must load");
    assert_eq!(first.entries().len(), 2);
    for entry in first.entries() {
        assert_eq!(entry.data_file().first_row_id(), None);
        assert_eq!(entry.snapshot_id(), Some(1001));
        assert_eq!(entry.sequence_number(), Some(11));
        assert_eq!(entry.file_sequence_number, Some(11));
    }

    let second = object_cache
        .get_manifest(&v3_entry, None)
        .await
        .expect("ranged read must load");
    assert_eq!(second.entries().len(), 2);
    let ids: Vec<Option<i64>> = second
        .entries()
        .iter()
        .map(|entry| entry.data_file().first_row_id())
        .collect();
    assert_eq!(ids, vec![Some(100), Some(103)]);
    for entry in second.entries() {
        assert_eq!(entry.snapshot_id(), Some(2002));
        assert_eq!(entry.sequence_number(), Some(22));
        assert_eq!(entry.file_sequence_number, Some(22));
    }

    let third = object_cache
        .get_manifest(&v2_entry, None)
        .await
        .expect("second rangeless read must load");
    for entry in third.entries() {
        assert_eq!(entry.data_file().first_row_id(), None);
        assert_eq!(entry.snapshot_id(), Some(1001));
        assert_eq!(entry.sequence_number(), Some(11));
    }
}

#[tokio::test]
async fn test_get_manifest_ranged_then_rangeless_serves_each_callers_context() {
    let fixture = TableTestFixture::new();
    let base = fixture.write_two_entry_manifest().await;

    let v3_entry = ManifestFile {
        sequence_number: 22,
        added_snapshot_id: 2002,
        first_row_id: Some(100),
        ..base.clone()
    };
    let v2_entry = ManifestFile {
        sequence_number: 11,
        added_snapshot_id: 1001,
        first_row_id: None,
        ..base
    };

    let object_cache = ObjectCache::new(fixture.table.file_io().clone());

    let first = object_cache
        .get_manifest(&v3_entry, None)
        .await
        .expect("ranged read must load");
    let ids: Vec<Option<i64>> = first
        .entries()
        .iter()
        .map(|entry| entry.data_file().first_row_id())
        .collect();
    assert_eq!(ids, vec![Some(100), Some(103)]);
    for entry in first.entries() {
        assert_eq!(entry.snapshot_id(), Some(2002));
        assert_eq!(entry.sequence_number(), Some(22));
    }

    let second = object_cache
        .get_manifest(&v2_entry, None)
        .await
        .expect("rangeless read must load");
    assert_eq!(second.entries().len(), 2);
    for entry in second.entries() {
        assert_eq!(entry.data_file().first_row_id(), None);
        assert_eq!(entry.snapshot_id(), Some(1001));
        assert_eq!(entry.sequence_number(), Some(11));
        assert_eq!(entry.file_sequence_number, Some(11));
    }
}

#[tokio::test]
async fn test_get_manifest_concurrent_contexts_stay_isolated() {
    let fixture = TableTestFixture::new();
    let base = fixture.write_two_entry_manifest().await;
    let rangeless = ManifestFile {
        sequence_number: 11,
        added_snapshot_id: 1001,
        first_row_id: None,
        ..base.clone()
    };
    let ranged = ManifestFile {
        sequence_number: 22,
        added_snapshot_id: 2002,
        first_row_id: Some(100),
        ..base
    };
    let cache = Arc::new(ObjectCache::new(fixture.table.file_io().clone()));
    let mut handles = Vec::with_capacity(16);
    for _ in 0..16 {
        let cache = Arc::clone(&cache);
        let rangeless = rangeless.clone();
        let ranged = ranged.clone();
        handles.push(tokio::spawn(async move {
            for _ in 0..20 {
                let plain = cache
                    .get_manifest(&rangeless, None)
                    .await
                    .expect("rangeless read must load");
                assert_eq!(plain.entries().len(), 2);
                for entry in plain.entries() {
                    assert_eq!(entry.data_file().first_row_id(), None);
                    assert_eq!(entry.snapshot_id(), Some(1001));
                    assert_eq!(entry.sequence_number(), Some(11));
                    assert_eq!(entry.file_sequence_number, Some(11));
                }
                let lined = cache
                    .get_manifest(&ranged, None)
                    .await
                    .expect("ranged read must load");
                let ids: Vec<Option<i64>> = lined
                    .entries()
                    .iter()
                    .map(|entry| entry.data_file().first_row_id())
                    .collect();
                assert_eq!(ids, vec![Some(100), Some(103)]);
                for entry in lined.entries() {
                    assert_eq!(entry.snapshot_id(), Some(2002));
                    assert_eq!(entry.sequence_number(), Some(22));
                    assert_eq!(entry.file_sequence_number, Some(22));
                }
            }
        }));
    }
    for handle in handles {
        handle.await.expect("task must complete");
    }
}

#[tokio::test]
async fn test_get_manifest_deletes_content_ignores_first_row_id_range() {
    let fixture = TableTestFixture::new();
    let base = fixture.write_two_entry_manifest().await;
    let warm = ManifestFile {
        sequence_number: 22,
        added_snapshot_id: 2002,
        first_row_id: Some(100),
        ..base.clone()
    };
    let deletes = ManifestFile {
        content: ManifestContentType::Deletes,
        sequence_number: 33,
        added_snapshot_id: 3003,
        first_row_id: Some(77),
        ..base
    };
    let cache = ObjectCache::new(fixture.table.file_io().clone());
    cache
        .get_manifest(&warm, None)
        .await
        .expect("warm read must load");
    let cached = cache
        .get_manifest(&deletes, None)
        .await
        .expect("deletes read must load");
    assert_eq!(cached.entries().len(), 2);
    for entry in cached.entries() {
        assert_eq!(entry.data_file().first_row_id(), None);
    }
    let direct = ObjectCache::with_disabled_cache(fixture.table.file_io().clone())
        .get_manifest(&deletes, None)
        .await
        .expect("direct read must load");
    assert_eq!(cached.entries(), direct.entries());
}

#[tokio::test]
async fn test_get_manifest_v1_and_v2_entries_match_direct_load() {
    let fixture = TableTestFixture::new();
    let base = fixture.write_two_entry_manifest().await;
    let v1 = ManifestFile {
        sequence_number: INITIAL_SEQUENCE_NUMBER,
        added_snapshot_id: 1001,
        first_row_id: None,
        ..base.clone()
    };
    let v2 = ManifestFile {
        sequence_number: 11,
        added_snapshot_id: 1001,
        first_row_id: None,
        ..base
    };
    let direct_v1 = ObjectCache::with_disabled_cache(fixture.table.file_io().clone())
        .get_manifest(&v1, None)
        .await
        .expect("direct v1 read must load");
    let direct_v2 = ObjectCache::with_disabled_cache(fixture.table.file_io().clone())
        .get_manifest(&v2, None)
        .await
        .expect("direct v2 read must load");
    let cache = ObjectCache::new(fixture.table.file_io().clone());
    cache
        .get_manifest(&v2, None)
        .await
        .expect("v2 warm read must load");
    let cached_v1 = cache
        .get_manifest(&v1, None)
        .await
        .expect("v1 read must load");
    assert_eq!(cached_v1.entries(), direct_v1.entries());
    let cache = ObjectCache::new(fixture.table.file_io().clone());
    cache
        .get_manifest(&v1, None)
        .await
        .expect("v1 warm read must load");
    let cached_v2 = cache
        .get_manifest(&v2, None)
        .await
        .expect("v2 read must load");
    assert_eq!(cached_v2.entries(), direct_v2.entries());
}

#[tokio::test]
async fn test_get_manifest_first_row_id_overflow_is_data_invalid() {
    let fixture = TableTestFixture::new();
    let base = fixture.write_two_entry_manifest().await;
    let valid = ManifestFile {
        sequence_number: 22,
        added_snapshot_id: 2002,
        first_row_id: Some(100),
        ..base.clone()
    };
    let overflowing = ManifestFile {
        sequence_number: 22,
        added_snapshot_id: 2002,
        first_row_id: Some(u64::MAX),
        ..base
    };
    let cache = ObjectCache::new(fixture.table.file_io().clone());
    cache
        .get_manifest(&valid, None)
        .await
        .expect("warm read must load");
    let cached_err = cache
        .get_manifest(&overflowing, None)
        .await
        .expect_err("overflowing range must fail");
    assert_eq!(cached_err.kind(), ErrorKind::DataInvalid);
    let direct_err = ObjectCache::with_disabled_cache(fixture.table.file_io().clone())
        .get_manifest(&overflowing, None)
        .await
        .expect_err("direct overflowing range must fail");
    assert_eq!(direct_err.kind(), ErrorKind::DataInvalid);
    let again = cache
        .get_manifest(&valid, None)
        .await
        .expect("valid read after overflow must load");
    assert_eq!(again.entries().len(), 2);
}
