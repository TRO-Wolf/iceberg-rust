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

use minijinja::value::Value;
use minijinja::{AutoEscape, Environment, context};
use tempfile::TempDir;
use uuid::Uuid;

use super::*;
use crate::TableIdent;
use crate::io::{FileIO, OutputFile};
use crate::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, INITIAL_SEQUENCE_NUMBER, Literal,
    ManifestContentType, ManifestEntry, ManifestListWriter, ManifestStatus, ManifestWriterBuilder,
    Snapshot, Struct, TableMetadata,
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

/// Wave A: weigher clamp floors at 1 and caps at `u32::MAX`.
#[test]
fn test_clamp_cache_weight_floor_and_cap() {
    assert_eq!(clamp_cache_weight(0), 1);
    assert_eq!(clamp_cache_weight(1), 1);
    assert_eq!(clamp_cache_weight(u32::MAX as u64), u32::MAX);
    assert_eq!(clamp_cache_weight(u32::MAX as u64 + 1), u32::MAX);
    assert_eq!(clamp_cache_weight(100), 100);

    // Relative scale of the entry-count estimate used for manifests.
    let one = clamp_cache_weight(1u64.saturating_mul(ROUGH_MANIFEST_ENTRY_BYTES));
    let ten = clamp_cache_weight(10u64.saturating_mul(ROUGH_MANIFEST_ENTRY_BYTES));
    assert!(
        ten > one,
        "more entries must weigh more: one={one} ten={ten}"
    );
    assert_eq!(one, ROUGH_MANIFEST_ENTRY_BYTES as u32);

    // Saturating multiply before clamp must not panic and must cap at u32::MAX.
    let overflow_weight =
        clamp_cache_weight(u64::MAX.saturating_mul(ROUGH_MANIFEST_LIST_ENTRY_BYTES));
    assert_eq!(overflow_weight, u32::MAX);
}

/// Wave A: loaded manifest / manifest-list weights use real estimates (≥ 1, and
/// manifest entry weight scales with the entry count for a 1-entry fixture).
#[tokio::test]
async fn test_estimate_weights_on_loaded_manifests() {
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

    let list_weight = estimate_manifest_list_weight(&manifest_list);
    assert!(
        list_weight >= 1,
        "manifest list weight must floor at 1, got {list_weight}"
    );
    // List weight is entry_count × list-entry estimate — never the sum of child
    // manifest_length values (those are separate cached objects).
    let entry = manifest_list
        .entries()
        .first()
        .expect("fixture list has one entry");
    let expected_list = clamp_cache_weight(
        (manifest_list.entries().len() as u64)
            .max(1)
            .saturating_mul(ROUGH_MANIFEST_LIST_ENTRY_BYTES),
    );
    assert_eq!(
        list_weight, expected_list,
        "list weight must be entry_count × list-entry estimate, not child manifest_length"
    );
    // Sanity: when the child has a declared length, it must NOT equal the list weight
    // (unless by coincidence the length equals the list-entry constant).
    if entry.manifest_length > 0 && entry.manifest_length as u64 != ROUGH_MANIFEST_LIST_ENTRY_BYTES
    {
        assert_ne!(
            list_weight,
            clamp_cache_weight(entry.manifest_length as u64),
            "list weight must not use child manifest_length"
        );
    }

    let manifest = object_cache
        .get_manifest(entry, None)
        .await
        .expect("manifest must load");
    let manifest_weight = estimate_manifest_weight(&manifest);
    let expected = clamp_cache_weight(
        (manifest.entries().len() as u64)
            .max(1)
            .saturating_mul(ROUGH_MANIFEST_ENTRY_BYTES),
    );
    assert_eq!(
        manifest_weight, expected,
        "manifest weight must be entry_count × rough bytes (floored)"
    );
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
