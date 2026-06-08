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

//! The `entries` metadata table — the raw manifest-entry view of the table's **current snapshot**.
//!
//! Unlike the `files` family (which lists only LIVE data/delete files), `entries` exposes EVERY manifest
//! entry of EVERY manifest (data AND delete) of the current snapshot — including the `Deleted` tombstones
//! (`status == 2`). It is the diagnostic view: Java's `ManifestEntriesTable` javadoc warns it "exposes
//! internal details, like files that have been deleted." A bug that drops the Deleted entries makes the
//! `entries` table wrong.
//!
//! Each row is one [`crate::spec::ManifestEntry`]:
//! - `status` (id 0, int) — `0`=Existing / `1`=Added / `2`=Deleted (Java `ManifestEntry.Status` ids).
//! - `snapshot_id` (id 1, long, nullable) — the entry's snapshot id; null when inherited.
//! - `sequence_number` (id 3, long, nullable) — the data sequence number; null when inherited.
//! - `file_sequence_number` (id 4, long, nullable) — the file sequence number; null when inherited.
//! - `data_file` (id 2, struct) — the SAME [`crate::inspect::data_file`] projection the `files` table
//!   exposes flat, here NESTED under one struct column.
//!
//! The schema mirrors Java `ManifestEntry.getSchema(partitionType)` = `wrapFileSchema(DataFile.getType(...))`
//! (`core/.../ManifestEntry.java`) with the canonical field ids (status 0 / snapshot_id 1 / sequence_number
//! 3 / file_sequence_number 4 / data_file 2).
//!
//! References:
//! - <https://github.com/apache/iceberg/blob/main/core/src/main/java/org/apache/iceberg/BaseEntriesTable.java>
//! - <https://github.com/apache/iceberg/blob/main/core/src/main/java/org/apache/iceberg/ManifestEntriesTable.java>
//! - <https://github.com/apache/iceberg/blob/main/core/src/main/java/org/apache/iceberg/ManifestEntry.java>
//!
//! Deferred column: `readable_metrics` (Java `MetricsUtil.readableMetricsStruct`), exactly as the `files`
//! table defers it.

use std::sync::Arc;

use arrow_array::builder::{Int32Builder, Int64Builder};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::DataType;
use futures::{StreamExt, stream};

use super::data_file::{DataFileStructBuilder, data_file_fields};
use crate::arrow::schema_to_arrow_schema;
use crate::scan::ArrowRecordBatchStream;
use crate::spec::{NestedField, PrimitiveType, Schema, Type};
use crate::table::Table;
use crate::{Error, ErrorKind, Result};

/// The `entries` metadata table (Java `ManifestEntriesTable` / `BaseEntriesTable`).
pub struct EntriesTable<'a> {
    table: &'a Table,
}

impl<'a> EntriesTable<'a> {
    /// Create a new `entries` table instance.
    pub fn new(table: &'a Table) -> Self {
        Self { table }
    }

    /// Returns the iceberg schema of the `entries` metadata table.
    ///
    /// Mirrors Java `ManifestEntry.getSchema(partitionType)`: `status`(0), `snapshot_id`(1),
    /// `sequence_number`(3), `file_sequence_number`(4), `data_file`(2, struct = the shared `data_file`
    /// projection over the table's DEFAULT partition type). `readable_metrics` is deferred.
    pub fn schema(&self) -> Schema {
        let partition_type = self.table.metadata().default_partition_type();
        let data_file_type = Type::Struct(crate::spec::StructType::new(data_file_fields(
            partition_type,
        )));
        let fields = vec![
            NestedField::required(0, "status", Type::Primitive(PrimitiveType::Int)),
            NestedField::optional(1, "snapshot_id", Type::Primitive(PrimitiveType::Long)),
            NestedField::optional(3, "sequence_number", Type::Primitive(PrimitiveType::Long)),
            NestedField::optional(
                4,
                "file_sequence_number",
                Type::Primitive(PrimitiveType::Long),
            ),
            NestedField::required(2, "data_file", data_file_type),
        ];
        Schema::builder()
            .with_fields(fields.into_iter().map(Arc::new))
            .build()
            .expect("entries metadata table schema is statically valid")
    }

    /// Scans the `entries` metadata table.
    ///
    /// Reads the current snapshot's manifest list, then EVERY manifest (data AND delete — Java
    /// `snapshot().allManifests`), then EVERY entry (NO `is_alive` filter — the Deleted tombstones are
    /// rows here). An empty table (no current snapshot) yields a single empty batch.
    pub async fn scan(&self) -> Result<ArrowRecordBatchStream> {
        let arrow_schema = Arc::new(schema_to_arrow_schema(&self.schema())?);
        let partition_type = self.table.metadata().default_partition_type().clone();
        let data_file_arrow_fields = data_file_struct_fields(&arrow_schema)?;

        let mut status = Int32Builder::new();
        let mut snapshot_id = Int64Builder::new();
        let mut sequence_number = Int64Builder::new();
        let mut file_sequence_number = Int64Builder::new();
        let mut data_file = DataFileStructBuilder::new(&data_file_arrow_fields, &partition_type);

        if let Some(snapshot) = self.table.metadata().current_snapshot() {
            let manifest_list = snapshot
                .load_manifest_list(self.table.file_io(), self.table.metadata())
                .await?;
            for manifest_file in manifest_list.entries() {
                let manifest = manifest_file.load_manifest(self.table.file_io()).await?;
                for entry in manifest.entries() {
                    // ALL entries — incl. Deleted tombstones (status 2). This is the difference from the
                    // `files` family, which filters on `entry.is_alive()`.
                    status.append_value(entry.status() as i32);
                    snapshot_id.append_option(entry.snapshot_id());
                    sequence_number.append_option(entry.sequence_number());
                    // `ManifestEntry` exposes `file_sequence_number` as a public field, not a method.
                    file_sequence_number.append_option(entry.file_sequence_number);
                    data_file.append(entry.data_file())?;
                }
            }
        }

        let columns: Vec<ArrayRef> = vec![
            Arc::new(status.finish()),
            Arc::new(snapshot_id.finish()),
            Arc::new(sequence_number.finish()),
            Arc::new(file_sequence_number.finish()),
            Arc::new(data_file.finish()),
        ];
        let batch = RecordBatch::try_new(arrow_schema, columns)?;
        Ok(stream::iter(vec![Ok(batch)]).boxed())
    }
}

/// Extracts the Arrow child `Fields` of the `data_file` struct column from the converted entries-table
/// Arrow schema (used to build the nested `DataFileStructBuilder`).
fn data_file_struct_fields(arrow_schema: &arrow_schema::Schema) -> Result<arrow_schema::Fields> {
    match arrow_schema.field_with_name("data_file")?.data_type() {
        DataType::Struct(fields) => Ok(fields.clone()),
        other => Err(Error::new(
            ErrorKind::Unexpected,
            format!("entries metadata table data_file column must be a struct, got {other:?}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow_array::cast::AsArray;
    use arrow_array::{Array, StructArray};
    use futures::TryStreamExt;

    use crate::scan::tests::TableTestFixture;
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Datum, Literal, ManifestEntry,
        ManifestListWriter, ManifestStatus, ManifestWriterBuilder, Struct,
    };

    /// A known, fixed file size used for every file in the fixtures (the metadata table reads only the
    /// manifest metadata, so no real parquet data file is needed).
    const FILE_SIZE: u64 = 1024;

    /// Builds the current snapshot's manifest list with one DATA manifest carrying a MIX of entry statuses
    /// — Added (1.parquet), Deleted (2.parquet, a tombstone) and Existing (3.parquet) — AND one DELETE
    /// manifest with one Added position-delete file (delete-1.parquet). The Deleted + Existing entries
    /// carry the PARENT snapshot's id + sequence numbers (the committed inherited values); the Added
    /// entries carry NONE (inherited at read time → reported as the parent snapshot's values once the
    /// manifest list is loaded). Mirrors the Increment-1 `files.rs` fixture shape.
    ///
    /// Uses only public crate APIs, so it does not depend on the scan fixture's private helpers.
    async fn setup_mixed_status_manifests(fixture: &TableTestFixture) {
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

        // DATA manifest: Added / Deleted / Existing.
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
                            .record_count(2)
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
                            .record_count(3)
                            .partition(Struct::from_iter([Some(Literal::long(300))]))
                            .build()
                            .unwrap(),
                    )
                    .build(),
            )
            .unwrap();
        let data_manifest = data_writer.write_manifest_file().await.unwrap();

        // DELETE manifest: one Added position-delete file.
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
                            .record_count(4)
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
    }

    /// Concatenates an entries-table scan into a single batch.
    async fn scan_single_batch(
        stream: crate::scan::ArrowRecordBatchStream,
    ) -> arrow_array::RecordBatch {
        let batches: Vec<_> = stream.try_collect().await.unwrap();
        arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap()
    }

    /// One entry row projected for assertions (the top-level entry columns + a couple of `data_file`
    /// children), so tests can compare per-entry against the committed metadata without depending on order.
    #[derive(Debug, Clone, Copy)]
    struct EntryRow {
        status: i32,
        snapshot_id: Option<i64>,
        sequence_number: Option<i64>,
        file_sequence_number: Option<i64>,
        record_count: i64,
        content: i32,
    }

    /// Extracts each entry row keyed by the data_file's file_path leaf (so a test can assert per-entry
    /// without depending on row order).
    fn rows_by_suffix(batch: &arrow_array::RecordBatch) -> HashMap<String, EntryRow> {
        let status = batch
            .column_by_name("status")
            .unwrap()
            .as_primitive::<arrow_array::types::Int32Type>();
        let snapshot_id = batch
            .column_by_name("snapshot_id")
            .unwrap()
            .as_primitive::<arrow_array::types::Int64Type>();
        let sequence_number = batch
            .column_by_name("sequence_number")
            .unwrap()
            .as_primitive::<arrow_array::types::Int64Type>();
        let file_sequence_number = batch
            .column_by_name("file_sequence_number")
            .unwrap()
            .as_primitive::<arrow_array::types::Int64Type>();
        let data_file = batch
            .column_by_name("data_file")
            .unwrap()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let file_path = data_file
            .column_by_name("file_path")
            .unwrap()
            .as_string::<i32>();
        let record_count = data_file
            .column_by_name("record_count")
            .unwrap()
            .as_primitive::<arrow_array::types::Int64Type>();
        let content = data_file
            .column_by_name("content")
            .unwrap()
            .as_primitive::<arrow_array::types::Int32Type>();

        let opt = |a: &arrow_array::Int64Array, i: usize| {
            if a.is_null(i) { None } else { Some(a.value(i)) }
        };

        let mut out = HashMap::new();
        for i in 0..batch.num_rows() {
            let suffix = file_path.value(i).rsplit('/').next().unwrap().to_string();
            out.insert(suffix, EntryRow {
                status: status.value(i),
                snapshot_id: opt(snapshot_id, i),
                sequence_number: opt(sequence_number, i),
                file_sequence_number: opt(file_sequence_number, i),
                record_count: record_count.value(i),
                content: content.value(i),
            });
        }
        out
    }

    #[tokio::test]
    async fn test_entries_table_includes_deleted_tombstone() {
        // RISK (the headline difference from `files`): `entries` shows ALL entries, INCLUDING the Deleted
        // tombstone (2.parquet, status 2). Dropping it = the entries table is wrong.
        let fixture = TableTestFixture::new();
        setup_mixed_status_manifests(&fixture).await;

        let batch =
            scan_single_batch(fixture.table.inspect().entries().scan().await.unwrap()).await;
        let rows = rows_by_suffix(&batch);

        // All four entries present: Added data, Deleted data, Existing data, Added delete.
        assert_eq!(batch.num_rows(), 4);
        let mut suffixes: Vec<&String> = rows.keys().collect();
        suffixes.sort();
        assert_eq!(suffixes, vec![
            "1.parquet",
            "2.parquet",
            "3.parquet",
            "delete-1.parquet",
        ]);
        // The Deleted tombstone IS present (this is the bug-pinning assertion).
        assert!(
            rows.contains_key("2.parquet"),
            "the Deleted tombstone 2.parquet must be a row in the entries table"
        );
        assert_eq!(
            rows["2.parquet"].status, 2,
            "2.parquet must carry status Deleted=2"
        );
    }

    #[tokio::test]
    async fn test_entries_table_status_values_per_entry() {
        // RISK: wrong `status` mapping — Added=1, Existing=0, Deleted=2 (Java `ManifestEntry.Status` ids).
        let fixture = TableTestFixture::new();
        setup_mixed_status_manifests(&fixture).await;

        let batch =
            scan_single_batch(fixture.table.inspect().entries().scan().await.unwrap()).await;
        let rows = rows_by_suffix(&batch);

        assert_eq!(rows["1.parquet"].status, 1, "Added data file → status 1");
        assert_eq!(rows["2.parquet"].status, 2, "Deleted data file → status 2");
        assert_eq!(rows["3.parquet"].status, 0, "Existing data file → status 0");
        assert_eq!(
            rows["delete-1.parquet"].status, 1,
            "Added delete file → status 1"
        );
    }

    #[tokio::test]
    async fn test_entries_table_sequence_and_snapshot_ids_match_committed() {
        // RISK: wrong snapshot_id / sequence_number / file_sequence_number — they must reflect the
        // values ACTUALLY committed to the manifest (incl. the writer's overrides + read-time inheritance),
        // NOT an idealized guess. The committed shapes (per the `ManifestWriter` contract):
        //  - Existing entry: snapshot_id + both seq numbers preserved AS WRITTEN (= the PARENT snapshot's).
        //  - Deleted entry: status + snapshot_id are STAMPED with the manifest's snapshot id (= CURRENT)
        //    by `add_delete_entry`, but the data/file sequence numbers are PRESERVED (= the PARENT's).
        //  - Added entry: had NO snapshot_id/seq → INHERITED from the manifest-list entry at read time
        //    (= the CURRENT snapshot's id + seq).
        let fixture = TableTestFixture::new();
        setup_mixed_status_manifests(&fixture).await;

        let metadata = fixture.table.metadata().clone();
        let current = metadata.current_snapshot().unwrap();
        let parent = current.parent_snapshot(&metadata).unwrap();

        let batch =
            scan_single_batch(fixture.table.inspect().entries().scan().await.unwrap()).await;
        let rows = rows_by_suffix(&batch);

        // Existing entry: snapshot id + seq preserved as written (the parent snapshot's).
        let existing = rows["3.parquet"];
        assert_eq!(existing.snapshot_id, Some(parent.snapshot_id()));
        assert_eq!(existing.sequence_number, Some(parent.sequence_number()));
        assert_eq!(
            existing.file_sequence_number,
            Some(parent.sequence_number())
        );

        // Deleted entry: snapshot id STAMPED to the manifest's snapshot (current) by `add_delete_entry`;
        // the data/file sequence numbers are PRESERVED (the parent's). This pins the genuine committed
        // value — a code that mis-read it (e.g. inherited everything) would fail here.
        let deleted = rows["2.parquet"];
        assert_eq!(deleted.snapshot_id, Some(current.snapshot_id()));
        assert_eq!(deleted.sequence_number, Some(parent.sequence_number()));
        assert_eq!(deleted.file_sequence_number, Some(parent.sequence_number()));

        // Added entry: snapshot_id + seq inherited from the CURRENT manifest-list entry at read time.
        let added = rows["1.parquet"];
        assert_eq!(added.snapshot_id, Some(current.snapshot_id()));
        assert_eq!(added.sequence_number, Some(current.sequence_number()));
        assert_eq!(added.file_sequence_number, Some(current.sequence_number()));
    }

    #[tokio::test]
    async fn test_entries_table_data_file_struct_carries_right_file() {
        // RISK: row misalignment / wrong data_file projection — the nested data_file struct column must
        // carry THIS entry's file (record_count + content per file_path). Spot-checked via the struct's
        // child arrays.
        let fixture = TableTestFixture::new();
        setup_mixed_status_manifests(&fixture).await;

        let batch =
            scan_single_batch(fixture.table.inspect().entries().scan().await.unwrap()).await;
        let rows = rows_by_suffix(&batch);

        assert_eq!(rows["1.parquet"].record_count, 1);
        assert_eq!(rows["1.parquet"].content, 0); // content == DATA(0)
        assert_eq!(rows["2.parquet"].record_count, 2);
        assert_eq!(rows["2.parquet"].content, 0);
        assert_eq!(rows["3.parquet"].record_count, 3);
        assert_eq!(rows["3.parquet"].content, 0);
        assert_eq!(rows["delete-1.parquet"].record_count, 4);
        assert_eq!(rows["delete-1.parquet"].content, 1); // content == PositionDeletes(1)

        // The data_file struct also carries the Added file's metrics map (column_sizes {1: 42}).
        let data_file = batch
            .column_by_name("data_file")
            .unwrap()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let column_sizes = data_file.column_by_name("column_sizes").unwrap().as_map();
        let mut found = false;
        for i in 0..column_sizes.len() {
            let entries = column_sizes.value(i);
            let keys = entries
                .column(0)
                .as_primitive::<arrow_array::types::Int32Type>();
            let values = entries
                .column(1)
                .as_primitive::<arrow_array::types::Int64Type>();
            if keys.len() == 1 && keys.value(0) == 1 && values.value(0) == 42 {
                found = true;
            }
        }
        assert!(
            found,
            "expected column_sizes {{1: 42}} on the Added data file's data_file struct"
        );
    }

    #[tokio::test]
    async fn test_entries_table_arrow_schema_columns_field_ids_and_nested_struct() {
        // RISK: wrong schema — 5 top-level columns with Java field ids (status 0, snapshot_id 1,
        // sequence_number 3, file_sequence_number 4, data_file 2), and data_file is a Struct of the
        // data_file projection.
        use arrow_schema::DataType;
        let fixture = TableTestFixture::new();
        let schema = fixture.table.inspect().entries().schema();
        let arrow = crate::arrow::schema_to_arrow_schema(&schema).unwrap();

        let names: Vec<&str> = arrow.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec![
            "status",
            "snapshot_id",
            "sequence_number",
            "file_sequence_number",
            "data_file",
        ]);

        let field_id = |name: &str| -> &str {
            arrow
                .field_with_name(name)
                .unwrap()
                .metadata()
                .get(parquet::arrow::PARQUET_FIELD_ID_META_KEY)
                .unwrap()
                .as_str()
        };
        assert_eq!(field_id("status"), "0");
        assert_eq!(field_id("snapshot_id"), "1");
        assert_eq!(field_id("sequence_number"), "3");
        assert_eq!(field_id("file_sequence_number"), "4");
        assert_eq!(field_id("data_file"), "2");

        assert_eq!(
            arrow.field_with_name("status").unwrap().data_type(),
            &DataType::Int32
        );
        // status is required (non-null); the seq/snapshot columns are nullable.
        assert!(!arrow.field_with_name("status").unwrap().is_nullable());
        assert!(arrow.field_with_name("snapshot_id").unwrap().is_nullable());

        // data_file is a Struct carrying the data_file projection (spot-check a couple of nested ids).
        let data_file_fields = match arrow.field_with_name("data_file").unwrap().data_type() {
            DataType::Struct(fields) => fields.clone(),
            other => panic!("data_file must be a struct, got {other:?}"),
        };
        let nested_names: Vec<&str> = data_file_fields.iter().map(|f| f.name().as_str()).collect();
        assert!(nested_names.contains(&"file_path"));
        assert!(nested_names.contains(&"record_count"));
        assert!(nested_names.contains(&"partition"));
        // The nested file_path keeps its canonical DataFile field id 100.
        let file_path_id = data_file_fields
            .iter()
            .find(|f| f.name() == "file_path")
            .unwrap()
            .metadata()
            .get(parquet::arrow::PARQUET_FIELD_ID_META_KEY)
            .unwrap()
            .clone();
        assert_eq!(file_path_id, "100");
    }

    #[tokio::test]
    async fn test_entries_table_empty_table_yields_empty_batch() {
        // RISK: panic / non-empty on an empty table — no current snapshot must yield zero rows.
        let fixture = TableTestFixture::new_empty();
        let batches: Vec<_> = fixture
            .table
            .inspect()
            .entries()
            .scan()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0);
    }
}
