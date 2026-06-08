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

//! The `files` family of metadata tables: `files`, `data_files`, `delete_files`.
//!
//! Each exposes the data/delete files referenced by the table's **current snapshot** as rows, with the
//! data-file column set (content, file path/format, partition, record/size counts, the metrics maps,
//! and the V3 deletion-vector fields). The three tables share one schema, one read, and one row builder
//! and differ ONLY by which manifests they read — mirroring Java `BaseFilesTable`:
//!
//! - [`FilesTable`]       → all manifests          (Java `FilesTable` / `snapshot().allManifests()`)
//! - [`DataFilesTable`]   → DATA-content manifests  (Java `DataFilesTable` / `snapshot().dataManifests()`)
//! - [`DeleteFilesTable`] → DELETE-content manifests (Java `DeleteFilesTable` / `snapshot().deleteManifests()`)
//!
//! Within a selected manifest only LIVE entries (Added/Existing, [`ManifestEntry::is_alive`]) are rows.
//!
//! The data-file column set (schema + row builder) is the shared [`crate::inspect::data_file`] projection
//! — the `files` family flattens it to top-level columns, the `entries` table nests it under a `data_file`
//! struct. See that module (Rule of Three).
//!
//! References:
//! - <https://github.com/apache/iceberg/blob/main/core/src/main/java/org/apache/iceberg/BaseFilesTable.java>
//! - <https://github.com/apache/iceberg/blob/main/api/src/main/java/org/apache/iceberg/DataFile.java>
//!
//! Deferred column: `readable_metrics` (Java `MetricsUtil.readableMetricsStruct` — a virtual per-data-column
//! struct of human-readable min/max/counts). All raw columns, including the metrics maps, are present.

use std::sync::Arc;

use arrow_array::RecordBatch;
use futures::{StreamExt, stream};

use super::data_file::{DataFileStructBuilder, data_file_fields};
use crate::Result;
use crate::arrow::schema_to_arrow_schema;
use crate::scan::ArrowRecordBatchStream;
use crate::spec::{ManifestContentType, Schema};
use crate::table::Table;

/// Which files a [`FilesTable`] exposes — the only thing that differs across the three tables.
///
/// Mirrors the Java `BaseFilesTableScan.manifests()` override on each concrete table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilesTableKind {
    /// All manifests (Java `FilesTable`).
    All,
    /// DATA-content manifests only (Java `DataFilesTable`).
    Data,
    /// DELETE-content manifests only (Java `DeleteFilesTable`).
    Deletes,
}

impl FilesTableKind {
    /// Returns whether a manifest of the given content type should be read for this table.
    fn includes_manifest(&self, content: ManifestContentType) -> bool {
        match self {
            FilesTableKind::All => true,
            FilesTableKind::Data => content == ManifestContentType::Data,
            FilesTableKind::Deletes => content == ManifestContentType::Deletes,
        }
    }
}

/// The shared base for the `files` / `data_files` / `delete_files` metadata tables (Java
/// `BaseFilesTable`). The three concrete tables wrap this with a fixed [`FilesTableKind`].
pub struct FilesTable<'a> {
    table: &'a Table,
    kind: FilesTableKind,
}

impl<'a> FilesTable<'a> {
    fn new(table: &'a Table, kind: FilesTableKind) -> Self {
        Self { table, kind }
    }

    /// Create a `files` table (all data + delete files in the current snapshot).
    pub fn all(table: &'a Table) -> Self {
        Self::new(table, FilesTableKind::All)
    }

    /// Create a `data_files` table (only DATA-content files in the current snapshot).
    pub fn data(table: &'a Table) -> Self {
        Self::new(table, FilesTableKind::Data)
    }

    /// Create a `delete_files` table (only position/equality delete files in the current snapshot).
    pub fn deletes(table: &'a Table) -> Self {
        Self::new(table, FilesTableKind::Deletes)
    }

    /// Returns the iceberg schema of the files metadata table.
    ///
    /// Mirrors Java `DataFile.getType(partitionType).fields()` — the field ids are the canonical
    /// `DataFile` ids from `api/DataFile.java`, built from the shared [`data_file_fields`] projection (the
    /// `files` family exposes them FLAT as the table's top-level columns). The partition column carries the
    /// table's DEFAULT partition type. `readable_metrics` is deferred.
    pub fn schema(&self) -> Schema {
        let partition_type = self.table.metadata().default_partition_type();
        Schema::builder()
            .with_fields(data_file_fields(partition_type))
            .build()
            .expect("files metadata table schema is statically valid")
    }

    /// Scans the files metadata table.
    ///
    /// Reads the current snapshot's manifest list, selects the manifests whose content passes this
    /// table's [`FilesTableKind`] filter, and emits one row per LIVE manifest entry built from its
    /// [`crate::spec::DataFile`]. An empty table (no current snapshot) yields a single empty batch.
    pub async fn scan(&self) -> Result<ArrowRecordBatchStream> {
        // The flattened files-table Arrow schema IS the `data_file` struct's child fields (top-level), so
        // the same `DataFileStructBuilder` that builds the `entries` nested column builds these rows; we
        // then split its `StructArray` into the top-level columns.
        let arrow_schema = Arc::new(schema_to_arrow_schema(&self.schema())?);
        let partition_type = self.table.metadata().default_partition_type().clone();
        let data_file_arrow_fields = arrow_schema.fields().clone();

        let mut builder = DataFileStructBuilder::new(&data_file_arrow_fields, &partition_type);

        if let Some(snapshot) = self.table.metadata().current_snapshot() {
            let manifest_list = snapshot
                .load_manifest_list(self.table.file_io(), self.table.metadata())
                .await?;
            for manifest_file in manifest_list.entries() {
                if !self.kind.includes_manifest(manifest_file.content) {
                    continue;
                }
                let manifest = manifest_file.load_manifest(self.table.file_io()).await?;
                for entry in manifest.entries() {
                    if entry.is_alive() {
                        builder.append(entry.data_file())?;
                    }
                }
            }
        }

        let data_file_struct = builder.finish();
        let batch = RecordBatch::try_new(arrow_schema, data_file_struct.columns().to_vec())?;
        Ok(stream::iter(vec![Ok(batch)]).boxed())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use futures::TryStreamExt;

    use crate::scan::tests::TableTestFixture;
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Datum, Literal, ManifestContentType,
        ManifestEntry, ManifestListWriter, ManifestStatus, ManifestWriterBuilder, Struct,
    };

    /// A known, fixed file size used for every file in the fixtures (the metadata table reads only the
    /// manifest metadata, so no real parquet data file is needed).
    const FILE_SIZE: u64 = 1024;

    /// Builds the current snapshot's manifest list with one DATA manifest (3 data files:
    /// Added/Deleted/Existing across partitions 100/200/300) AND one DELETE manifest (1 Added
    /// position-delete file in partition 100). Returns nothing — the fixture's current snapshot is wired.
    ///
    /// This drives only public crate APIs (`ManifestWriterBuilder`, `ManifestListWriter`, the fixture's
    /// public `table`/`table_location`), so it does not depend on the scan fixture's private helpers.
    async fn setup_data_and_delete_manifests(fixture: &TableTestFixture) {
        let metadata = fixture.table.metadata().clone();
        let current_snapshot = metadata.current_snapshot().unwrap();
        let parent_snapshot = current_snapshot.parent_snapshot(&metadata).unwrap();
        let current_schema = current_snapshot.schema(&metadata).unwrap();
        let current_partition_spec = metadata.default_partition_spec();

        let manifest_output = |fixture: &TableTestFixture| {
            fixture
                .table
                .file_io()
                .new_output(format!(
                    "{}/metadata/manifest_{}.avro",
                    fixture.table_location,
                    uuid::Uuid::new_v4()
                ))
                .unwrap()
        };

        // DATA manifest.
        let mut data_writer = ManifestWriterBuilder::new(
            manifest_output(fixture),
            Some(current_snapshot.snapshot_id()),
            None,
            current_schema.clone(),
            current_partition_spec.as_ref().clone(),
        )
        .build_v2_data();
        data_writer
            .add_entry(
                ManifestEntry::builder()
                    .status(ManifestStatus::Added)
                    .data_file(
                        DataFileBuilder::default()
                            .partition_spec_id(0)
                            .content(DataContentType::Data)
                            .file_path(format!("{}/1.parquet", &fixture.table_location))
                            .file_format(DataFileFormat::Parquet)
                            .file_size_in_bytes(FILE_SIZE)
                            .record_count(1)
                            .partition(Struct::from_iter([Some(Literal::long(100))]))
                            .column_sizes(HashMap::from([(1, 42u64)]))
                            .lower_bounds(HashMap::from([(1, Datum::long(1))]))
                            .key_metadata(None)
                            .build()
                            .unwrap(),
                    )
                    .build(),
            )
            .unwrap();
        data_writer
            .add_delete_entry(
                ManifestEntry::builder()
                    .status(ManifestStatus::Deleted)
                    .snapshot_id(parent_snapshot.snapshot_id())
                    .sequence_number(parent_snapshot.sequence_number())
                    .file_sequence_number(parent_snapshot.sequence_number())
                    .data_file(
                        DataFileBuilder::default()
                            .partition_spec_id(0)
                            .content(DataContentType::Data)
                            .file_path(format!("{}/2.parquet", &fixture.table_location))
                            .file_format(DataFileFormat::Parquet)
                            .file_size_in_bytes(FILE_SIZE)
                            .record_count(1)
                            .partition(Struct::from_iter([Some(Literal::long(200))]))
                            .build()
                            .unwrap(),
                    )
                    .build(),
            )
            .unwrap();
        data_writer
            .add_existing_entry(
                ManifestEntry::builder()
                    .status(ManifestStatus::Existing)
                    .snapshot_id(parent_snapshot.snapshot_id())
                    .sequence_number(parent_snapshot.sequence_number())
                    .file_sequence_number(parent_snapshot.sequence_number())
                    .data_file(
                        DataFileBuilder::default()
                            .partition_spec_id(0)
                            .content(DataContentType::Data)
                            .file_path(format!("{}/3.parquet", &fixture.table_location))
                            .file_format(DataFileFormat::Parquet)
                            .file_size_in_bytes(FILE_SIZE)
                            .record_count(1)
                            .partition(Struct::from_iter([Some(Literal::long(300))]))
                            .build()
                            .unwrap(),
                    )
                    .build(),
            )
            .unwrap();
        let data_manifest = data_writer.write_manifest_file().await.unwrap();

        // DELETE manifest: one Added position-delete file in partition 100.
        let mut delete_writer = ManifestWriterBuilder::new(
            manifest_output(fixture),
            Some(current_snapshot.snapshot_id()),
            None,
            current_schema.clone(),
            current_partition_spec.as_ref().clone(),
        )
        .build_v2_deletes();
        delete_writer
            .add_entry(
                ManifestEntry::builder()
                    .status(ManifestStatus::Added)
                    .data_file(
                        DataFileBuilder::default()
                            .partition_spec_id(0)
                            .content(DataContentType::PositionDeletes)
                            .file_path(format!("{}/delete-1.parquet", &fixture.table_location))
                            .file_format(DataFileFormat::Parquet)
                            .file_size_in_bytes(FILE_SIZE)
                            .record_count(1)
                            .partition(Struct::from_iter([Some(Literal::long(100))]))
                            .build()
                            .unwrap(),
                    )
                    .build(),
            )
            .unwrap();
        let delete_manifest = delete_writer.write_manifest_file().await.unwrap();

        let mut manifest_list_write = ManifestListWriter::v2(
            fixture
                .table
                .file_io()
                .new_output(current_snapshot.manifest_list())
                .unwrap(),
            current_snapshot.snapshot_id(),
            current_snapshot.parent_snapshot_id(),
            current_snapshot.sequence_number(),
        );
        manifest_list_write
            .add_manifests(vec![data_manifest, delete_manifest].into_iter())
            .unwrap();
        manifest_list_write.close().await.unwrap();

        // Sanity: the manifest list now carries exactly one DATA and one DELETE manifest.
        let manifest_list = current_snapshot
            .load_manifest_list(fixture.table.file_io(), &metadata)
            .await
            .unwrap();
        let contents: Vec<ManifestContentType> =
            manifest_list.entries().iter().map(|m| m.content).collect();
        assert!(contents.contains(&ManifestContentType::Data));
        assert!(contents.contains(&ManifestContentType::Deletes));
    }

    /// Collects the sorted `file_path` set of a files-table scan.
    async fn scan_paths(stream: crate::scan::ArrowRecordBatchStream) -> Vec<String> {
        let batches: Vec<_> = stream.try_collect().await.unwrap();
        let mut paths = Vec::new();
        for batch in &batches {
            let column = batch
                .column_by_name("file_path")
                .unwrap()
                .as_string::<i32>();
            for index in 0..column.len() {
                paths.push(column.value(index).to_string());
            }
        }
        paths.sort();
        paths
    }

    /// Concatenates a files-table scan into a single batch.
    async fn scan_single_batch(
        stream: crate::scan::ArrowRecordBatchStream,
    ) -> arrow_array::RecordBatch {
        let batches: Vec<_> = stream.try_collect().await.unwrap();
        arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap()
    }

    #[tokio::test]
    async fn test_files_table_lists_live_data_and_delete_files() {
        // RISK: wrong file set — `files` must list every LIVE data + delete file (Added/Existing),
        // never the Deleted tombstone (2.parquet).
        let fixture = TableTestFixture::new();
        setup_data_and_delete_manifests(&fixture).await;

        let stream = fixture.table.inspect().files().scan().await.unwrap();
        let paths = scan_paths(stream).await;

        assert_eq!(paths, vec![
            format!("{}/1.parquet", fixture.table_location),
            format!("{}/3.parquet", fixture.table_location),
            format!("{}/delete-1.parquet", fixture.table_location),
        ]);
    }

    #[tokio::test]
    async fn test_data_files_table_excludes_delete_files() {
        // RISK: wrong content filter — `data_files` reads DATA manifests only, so the position-delete
        // file must NOT appear, and the Deleted 2.parquet stays excluded as a tombstone.
        let fixture = TableTestFixture::new();
        setup_data_and_delete_manifests(&fixture).await;

        let stream = fixture.table.inspect().data_files().scan().await.unwrap();
        let paths = scan_paths(stream).await;

        assert_eq!(paths, vec![
            format!("{}/1.parquet", fixture.table_location),
            format!("{}/3.parquet", fixture.table_location),
        ]);
    }

    #[tokio::test]
    async fn test_delete_files_table_lists_only_delete_files() {
        // RISK: wrong content filter — `delete_files` reads DELETE manifests only; exactly the one
        // position-delete file, none of the data files.
        let fixture = TableTestFixture::new();
        setup_data_and_delete_manifests(&fixture).await;

        let stream = fixture.table.inspect().delete_files().scan().await.unwrap();
        let paths = scan_paths(stream).await;

        assert_eq!(paths, vec![format!(
            "{}/delete-1.parquet",
            fixture.table_location
        )]);
    }

    #[tokio::test]
    async fn test_files_table_content_column_distinguishes_data_and_deletes() {
        // RISK: wrong `content` value — DATA files must report content 0, the position-delete file 1.
        let fixture = TableTestFixture::new();
        setup_data_and_delete_manifests(&fixture).await;

        let batch = scan_single_batch(fixture.table.inspect().files().scan().await.unwrap()).await;

        let paths = batch
            .column_by_name("file_path")
            .unwrap()
            .as_string::<i32>();
        let content = batch
            .column_by_name("content")
            .unwrap()
            .as_primitive::<arrow_array::types::Int32Type>();
        let mut content_by_suffix = HashMap::new();
        for index in 0..paths.len() {
            let suffix = paths.value(index).rsplit('/').next().unwrap().to_string();
            content_by_suffix.insert(suffix, content.value(index));
        }
        assert_eq!(content_by_suffix["1.parquet"], 0);
        assert_eq!(content_by_suffix["3.parquet"], 0);
        assert_eq!(content_by_suffix["delete-1.parquet"], 1);
    }

    #[tokio::test]
    async fn test_files_table_record_count_and_size_match_committed_metadata() {
        // RISK: wrong column mapping — record_count / file_size_in_bytes must reflect the committed
        // DataFile values (record_count == 1; file_size == FILE_SIZE).
        let fixture = TableTestFixture::new();
        setup_data_and_delete_manifests(&fixture).await;

        let batch = scan_single_batch(fixture.table.inspect().files().scan().await.unwrap()).await;

        let record_count = batch
            .column_by_name("record_count")
            .unwrap()
            .as_primitive::<arrow_array::types::Int64Type>();
        let file_size = batch
            .column_by_name("file_size_in_bytes")
            .unwrap()
            .as_primitive::<arrow_array::types::Int64Type>();
        assert_eq!(record_count.len(), 3);
        for index in 0..record_count.len() {
            assert_eq!(record_count.value(index), 1);
            assert_eq!(file_size.value(index), FILE_SIZE as i64);
        }
    }

    #[tokio::test]
    async fn test_files_table_partition_struct_and_metrics_map_present() {
        // RISK: wrong column — the partition column must be the partition struct (long `x`), and the
        // metrics maps must be populated for the Added file (column_sizes {1: 42}).
        let fixture = TableTestFixture::new();
        setup_data_and_delete_manifests(&fixture).await;

        let batch = scan_single_batch(fixture.table.inspect().files().scan().await.unwrap()).await;

        let partition = batch.column_by_name("partition").unwrap().as_struct();
        assert_eq!(partition.num_columns(), 1);
        let partition_values = partition
            .column(0)
            .as_primitive::<arrow_array::types::Int64Type>();
        let mut partitions: Vec<i64> = (0..partition_values.len())
            .map(|index| partition_values.value(index))
            .collect();
        partitions.sort();
        assert_eq!(partitions, vec![100, 100, 300]);

        let column_sizes = batch.column_by_name("column_sizes").unwrap().as_map();
        let mut found_added_metrics = false;
        for index in 0..column_sizes.len() {
            let entries = column_sizes.value(index);
            let keys = entries
                .column(0)
                .as_primitive::<arrow_array::types::Int32Type>();
            let values = entries
                .column(1)
                .as_primitive::<arrow_array::types::Int64Type>();
            if keys.len() == 1 && keys.value(0) == 1 && values.value(0) == 42 {
                found_added_metrics = true;
            }
        }
        assert!(
            found_added_metrics,
            "expected column_sizes {{1: 42}} on the Added file"
        );
    }

    #[tokio::test]
    async fn test_files_table_arrow_schema_columns_and_types() {
        // RISK: wrong column set / type — assert the Arrow schema is the DataFile column set with the
        // expected leading types (content Int32, file_path Utf8, partition Struct, the metrics Maps).
        let fixture = TableTestFixture::new();
        let schema = fixture.table.inspect().files().schema();
        let arrow = crate::arrow::schema_to_arrow_schema(&schema).unwrap();

        let names: Vec<&str> = arrow.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec![
            "content",
            "file_path",
            "file_format",
            "spec_id",
            "partition",
            "record_count",
            "file_size_in_bytes",
            "column_sizes",
            "value_counts",
            "null_value_counts",
            "nan_value_counts",
            "lower_bounds",
            "upper_bounds",
            "key_metadata",
            "split_offsets",
            "equality_ids",
            "sort_order_id",
            "first_row_id",
            "referenced_data_file",
            "content_offset",
            "content_size_in_bytes",
        ]);

        use arrow_schema::DataType;
        assert_eq!(
            arrow.field_with_name("content").unwrap().data_type(),
            &DataType::Int32
        );
        assert_eq!(
            arrow.field_with_name("file_path").unwrap().data_type(),
            &DataType::Utf8
        );
        assert_eq!(
            arrow.field_with_name("record_count").unwrap().data_type(),
            &DataType::Int64
        );
        assert!(matches!(
            arrow.field_with_name("partition").unwrap().data_type(),
            DataType::Struct(_)
        ));
        assert!(matches!(
            arrow.field_with_name("column_sizes").unwrap().data_type(),
            DataType::Map(_, _)
        ));
        assert!(matches!(
            arrow.field_with_name("lower_bounds").unwrap().data_type(),
            DataType::Map(_, _)
        ));
    }

    #[tokio::test]
    async fn test_files_table_empty_table_yields_empty_batch() {
        // RISK: panic / non-empty on an empty table — no current snapshot must yield zero rows.
        let fixture = TableTestFixture::new_empty();
        let batches: Vec<_> = fixture
            .table
            .inspect()
            .files()
            .scan()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0);
    }

    #[tokio::test]
    async fn test_files_table_unpartitioned_keeps_empty_partition_struct_known_divergence() {
        // RISK / KNOWN DIVERGENCE from Java: for an UNPARTITIONED table Java `BaseFilesTable.schema()`
        // DROPS the `partition` field entirely ("avoid returning an empty struct, which is not always
        // supported. instead, drop the partition field" — `TypeUtil.selectNot(schema, PARTITION_ID)`).
        // The Rust port currently KEEPS a `partition` column typed as an empty struct (`Struct([])`).
        // This is non-corrupting (the file rows + every other column are correct, the row count is
        // right) but is a schema-shape divergence that matters for eventual Java interop — tracked in
        // GAP_MATRIX/todo as a deferral, NOT silently wrong. This test PINS the current behavior so the
        // divergence cannot change unnoticed; when the Java drop-empty-partition rule is implemented,
        // this test flips to assert the `partition` column is ABSENT.
        let fixture = TableTestFixture::new_unpartitioned();
        let metadata = fixture.table.metadata().clone();
        let current_snapshot = metadata.current_snapshot().unwrap();
        let current_schema = current_snapshot.schema(&metadata).unwrap();
        let current_partition_spec = metadata.default_partition_spec();

        let output = fixture
            .table
            .file_io()
            .new_output(format!(
                "{}/metadata/manifest_unp_{}.avro",
                fixture.table_location,
                uuid::Uuid::new_v4()
            ))
            .unwrap();
        let mut data_writer = ManifestWriterBuilder::new(
            output,
            Some(current_snapshot.snapshot_id()),
            None,
            current_schema.clone(),
            current_partition_spec.as_ref().clone(),
        )
        .build_v2_data();
        data_writer
            .add_entry(
                ManifestEntry::builder()
                    .status(ManifestStatus::Added)
                    .data_file(
                        DataFileBuilder::default()
                            .partition_spec_id(0)
                            .content(DataContentType::Data)
                            .file_path(format!("{}/u1.parquet", &fixture.table_location))
                            .file_format(DataFileFormat::Parquet)
                            .file_size_in_bytes(FILE_SIZE)
                            .record_count(1)
                            .partition(Struct::empty())
                            .build()
                            .unwrap(),
                    )
                    .build(),
            )
            .unwrap();
        let data_manifest = data_writer.write_manifest_file().await.unwrap();

        let mut manifest_list_write = ManifestListWriter::v2(
            fixture
                .table
                .file_io()
                .new_output(current_snapshot.manifest_list())
                .unwrap(),
            current_snapshot.snapshot_id(),
            current_snapshot.parent_snapshot_id(),
            current_snapshot.sequence_number(),
        );
        manifest_list_write
            .add_manifests(vec![data_manifest].into_iter())
            .unwrap();
        manifest_list_write.close().await.unwrap();

        let batch = scan_single_batch(fixture.table.inspect().files().scan().await.unwrap()).await;

        // Does not panic; the single data file is listed.
        assert_eq!(batch.num_rows(), 1);
        // CURRENT (divergent) behavior: the partition column is present as an empty struct.
        let partition = batch.column_by_name("partition").unwrap().as_struct();
        assert_eq!(
            partition.num_columns(),
            0,
            "unpartitioned files table currently keeps an empty-struct partition column \
             (Java drops it) — see the GAP_MATRIX deferral"
        );
    }
}
