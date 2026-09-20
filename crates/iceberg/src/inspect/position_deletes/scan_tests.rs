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
use std::fs::File;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use parquet::file::properties::WriterProperties;
use uuid::Uuid;

use super::*;
use crate::arrow::schema_to_arrow_schema;
use crate::delete_vector::DeleteVector;
use crate::metadata_columns::RESERVED_FIELD_ID_POS;
use crate::puffin::{Blob, CompressionCodec, DELETION_VECTOR_V1, PuffinWriter};
use crate::scan::tests::TableTestFixture;
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, ManifestEntry,
    ManifestFile, ManifestListWriter, ManifestStatus, ManifestWriterBuilder, PartitionSpec,
    Struct,
};

fn field_id_meta(id: i32) -> HashMap<String, String> {
    HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())])
}

fn write_posdel_parquet(
    path: &str,
    rows: &[(String, i64)],
    row_batch: Option<&RecordBatch>,
) -> u64 {
    let mut fields = vec![
        Field::new("file_path", DataType::Utf8, false)
            .with_metadata(field_id_meta(DELETE_FILE_PATH_ID)),
        Field::new("pos", DataType::Int64, false)
            .with_metadata(field_id_meta(DELETE_FILE_POS_ID)),
    ];
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(rows.iter().map(|row| row.0.clone()))),
        Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.1))),
    ];
    if let Some(row_batch) = row_batch {
        let row_fields = row_batch.schema().fields().clone();
        fields.push(
            Field::new("row", DataType::Struct(row_fields), true)
                .with_metadata(field_id_meta(DELETE_FILE_ROW_FIELD_ID)),
        );
        columns.push(Arc::new(arrow_array::StructArray::from(row_batch.clone())));
    }
    let arrow_schema = Arc::new(ArrowSchema::new(fields));
    let batch = RecordBatch::try_new(arrow_schema.clone(), columns).expect("posdel batch");
    let file = File::create(path).expect("create posdel parquet");
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, arrow_schema, Some(props)).expect("arrow writer");
    writer.write(&batch).expect("write posdel batch");
    writer.close().expect("close posdel parquet");
    std::fs::metadata(path).expect("posdel parquet stat").len()
}

fn table_row_batch(table_schema: &Schema, x: i64) -> RecordBatch {
    let arrow = Arc::new(schema_to_arrow_schema(table_schema).expect("table arrow schema"));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from_iter_values([x])),
        Arc::new(Int64Array::from_iter_values([x + 100])),
        Arc::new(Int64Array::from_iter_values([x + 200])),
        Arc::new(StringArray::from_iter_values(["row-string"])),
        Arc::new(Float64Array::from_iter_values([1.5])),
        Arc::new(Int32Array::from_iter_values([42])),
        Arc::new(Int64Array::from_iter_values([x + 300])),
        Arc::new(BooleanArray::from(vec![true])),
    ];
    RecordBatch::try_new(arrow, columns).expect("row batch")
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

fn posdel_data_file(
    path: &str,
    file_size: u64,
    record_count: u64,
    spec_id: i32,
    partition: Struct,
) -> DataFile {
    DataFileBuilder::default()
        .partition_spec_id(spec_id)
        .content(DataContentType::PositionDeletes)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(file_size)
        .record_count(record_count)
        .partition(partition)
        .key_metadata(None)
        .build()
        .expect("posdel data file")
}

fn added_entry(data_file: DataFile) -> ManifestEntry {
    ManifestEntry::builder()
        .status(ManifestStatus::Added)
        .data_file(data_file)
        .build()
}

async fn write_manifest(
    fixture: &TableTestFixture,
    spec: &PartitionSpec,
    v3: bool,
    entries: Vec<ManifestEntry>,
    deletes: bool,
) -> ManifestFile {
    let current_snapshot = fixture
        .table
        .metadata()
        .current_snapshot()
        .expect("current snapshot");
    let current_schema = current_snapshot
        .schema(fixture.table.metadata())
        .expect("snapshot schema");
    let builder = ManifestWriterBuilder::new(
        next_manifest_file(fixture),
        Some(current_snapshot.snapshot_id()),
        None,
        current_schema,
        spec.clone(),
    );
    let mut writer = match (v3, deletes) {
        (false, false) => builder.build_v2_data(),
        (false, true) => builder.build_v2_deletes(),
        (true, false) => builder.build_v3_data(),
        (true, true) => builder.build_v3_deletes(),
    };
    for entry in entries {
        match entry.status() {
            ManifestStatus::Added => writer.add_entry(entry).expect("add entry"),
            ManifestStatus::Deleted => writer.add_delete_entry(entry).expect("add delete entry"),
            ManifestStatus::Existing => writer
                .add_existing_entry(entry)
                .expect("add existing entry"),
        }
    }
    writer.write_manifest_file().await.expect("write manifest")
}

async fn write_manifest_list(
    fixture: &TableTestFixture,
    manifests: Vec<ManifestFile>,
    v3: bool,
) {
    let current_snapshot = fixture
        .table
        .metadata()
        .current_snapshot()
        .expect("current snapshot")
        .clone();
    let output = fixture
        .table
        .file_io()
        .new_output(current_snapshot.manifest_list())
        .expect("manifest list output");
    let mut writer = if v3 {
        ManifestListWriter::v3(
            output,
            current_snapshot.snapshot_id(),
            current_snapshot.parent_snapshot_id(),
            current_snapshot.sequence_number(),
            None,
        )
    } else {
        ManifestListWriter::v2(
            output,
            current_snapshot.snapshot_id(),
            current_snapshot.parent_snapshot_id(),
            current_snapshot.sequence_number(),
        )
    };
    writer.add_manifests(manifests.into_iter()).expect("add manifests");
    writer.close().await.expect("close manifest list");
}

fn deleted_entry(fixture: &TableTestFixture, data_file: DataFile) -> ManifestEntry {
    let current_snapshot = fixture
        .table
        .metadata()
        .current_snapshot()
        .expect("current snapshot");
    ManifestEntry::builder()
        .status(ManifestStatus::Deleted)
        .snapshot_id(current_snapshot.snapshot_id())
        .sequence_number(current_snapshot.sequence_number())
        .file_sequence_number(current_snapshot.sequence_number())
        .data_file(data_file)
        .build()
}

fn default_spec(fixture: &TableTestFixture) -> PartitionSpec {
    fixture.table.metadata().default_partition_spec().as_ref().clone()
}

async fn collect_posdel_batches(fixture: &TableTestFixture) -> Vec<RecordBatch> {
    let stream = fixture
        .table
        .inspect()
        .position_deletes()
        .scan()
        .expect("position_deletes scan");
    stream.try_collect().await.expect("collect posdel batches")
}

fn string_column<'a>(batch: &'a RecordBatch, name: &str) -> &'a StringArray {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("column {name} must exist"))
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap_or_else(|| panic!("column {name} must be a string array"))
}

fn int64_column<'a>(batch: &'a RecordBatch, name: &str) -> &'a Int64Array {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("column {name} must exist"))
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap_or_else(|| panic!("column {name} must be an int64 array"))
}

fn int32_column<'a>(batch: &'a RecordBatch, name: &str) -> &'a Int32Array {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("column {name} must exist"))
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap_or_else(|| panic!("column {name} must be an int32 array"))
}

fn single_batch(batches: Vec<RecordBatch>) -> RecordBatch {
    let total: usize = batches.iter().map(RecordBatch::num_rows).sum();
    assert!(!batches.is_empty(), "scan must emit at least one batch");
    let merged = if batches.len() == 1 {
        batches.into_iter().next().expect("one batch")
    } else {
        let schema = batches[0].schema();
        let mut columns: Vec<ArrayRef> = Vec::new();
        for index in 0..batches[0].num_columns() {
            let refs: Vec<&dyn arrow_array::Array> = batches
                .iter()
                .map(|batch| batch.column(index).as_ref())
                .collect();
            columns.push(arrow_select::concat::concat(&refs).expect("concat"));
        }
        RecordBatch::try_new(schema, columns).expect("merged batch")
    };
    assert_eq!(merged.num_rows(), total, "merged row count");
    merged
}

fn partition_struct<'a>(batch: &'a RecordBatch) -> &'a arrow_array::StructArray {
    batch
        .column_by_name("partition")
        .expect("partition column")
        .as_any()
        .downcast_ref::<arrow_array::StructArray>()
        .expect("partition must be a struct")
}

fn partition_i64(partition: &arrow_array::StructArray, name: &str, row: usize) -> Option<i64> {
    let column = partition
        .column_by_name(name)
        .unwrap_or_else(|| panic!("partition child {name}"))
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap_or_else(|| panic!("partition child {name} must be int64"));
    if column.is_null(row) {
        None
    } else {
        Some(column.value(row))
    }
}

fn partition_i32(partition: &arrow_array::StructArray, name: &str, row: usize) -> Option<i32> {
    let column = partition
        .column_by_name(name)
        .unwrap_or_else(|| panic!("partition child {name}"))
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap_or_else(|| panic!("partition child {name} must be int32"));
    if column.is_null(row) {
        None
    } else {
        Some(column.value(row))
    }
}

fn upgrade_fixture_to_v3(fixture: &mut TableTestFixture) {
    let upgraded = fixture
        .table
        .metadata()
        .clone()
        .into_builder(Some(
            fixture
                .table
                .metadata_location()
                .unwrap_or_default()
                .to_string(),
        ))
        .upgrade_format_version(crate::spec::FormatVersion::V3)
        .expect("upgrade to v3")
        .build()
        .expect("v3 metadata")
        .metadata;
    fixture.table = crate::table::Table::builder()
        .metadata(upgraded)
        .identifier(fixture.table.identifier().clone())
        .file_io(fixture.table.file_io().clone())
        .metadata_location(
            fixture
                .table
                .metadata_location()
                .unwrap_or_default()
                .to_string(),
        )
        .build()
        .expect("v3 table");
}

async fn write_dv_puffin(
    fixture: &TableTestFixture,
    path: &str,
    dvs: &[(&str, &[u64])],
) -> (u64, Vec<(i64, i64)>) {
    let output = fixture.table.file_io().new_output(path).expect("puffin output");
    let mut writer = PuffinWriter::new(&output, HashMap::new(), false)
        .await
        .expect("puffin writer");
    let mut coordinates = Vec::new();
    for (data_path, positions) in dvs {
        let mut delete_vector = DeleteVector::default();
        for position in *positions {
            delete_vector.insert(*position);
        }
        let blob = Blob::builder()
            .r#type(DELETION_VECTOR_V1.to_string())
            .fields(vec![RESERVED_FIELD_ID_POS])
            .snapshot_id(-1)
            .sequence_number(-1)
            .data(
                delete_vector
                    .serialize_deletion_vector_v1()
                    .expect("serialize dv"),
            )
            .properties(HashMap::from([(
                "referenced-data-file".to_string(),
                data_path.to_string(),
            )]))
            .build();
        let metadata = writer.add(blob, CompressionCodec::None).await.expect("add blob");
        coordinates.push((
            i64::try_from(metadata.offset()).expect("offset fits i64"),
            i64::try_from(metadata.length()).expect("length fits i64"),
        ));
    }
    let size = writer.close().await.expect("close puffin");
    (size, coordinates)
}

#[tokio::test]
async fn scan_partitioned_v2_rows_match_oracle() {
    let fixture = TableTestFixture::new();
    std::fs::create_dir_all(format!("{}/metadata", fixture.table_location))
        .expect("metadata dir");

    let data_a = format!("{}/data/x=1/a.parquet", fixture.table_location);
    let data_b = format!("{}/data/x=2/b.parquet", fixture.table_location);
    let del_a = format!("{}/del_a.parquet", fixture.table_location);
    let del_b = format!("{}/del_b.parquet", fixture.table_location);
    let del_leaked = format!("{}/del_leaked.parquet", fixture.table_location);
    let del_eq = format!("{}/del_eq.parquet", fixture.table_location);

    let size_a = write_posdel_parquet(&del_a, &[(data_a.clone(), 1)], None);
    let size_b = write_posdel_parquet(&del_b, &[(data_b.clone(), 1)], None);
    let size_leaked = write_posdel_parquet(&del_leaked, &[(data_a.clone(), 9)], None);
    let size_eq = write_posdel_parquet(&del_eq, &[(data_a.clone(), 5)], None);

    let spec = default_spec(&fixture);
    let data_manifest = write_manifest(
        &fixture,
        &spec,
        false,
        vec![added_entry(
            DataFileBuilder::default()
                .partition_spec_id(0)
                .content(DataContentType::Data)
                .file_path(data_a.clone())
                .file_format(DataFileFormat::Parquet)
                .file_size_in_bytes(100)
                .record_count(2)
                .partition(Struct::from_iter([Some(Literal::long(1))]))
                .key_metadata(None)
                .build()
                .expect("data file"),
        )],
        false,
    )
    .await;
    let delete_manifest = write_manifest(
        &fixture,
        &spec,
        false,
        vec![
            added_entry(posdel_data_file(
                &del_a,
                size_a,
                1,
                0,
                Struct::from_iter([Some(Literal::long(1))]),
            )),
            added_entry(posdel_data_file(
                &del_b,
                size_b,
                1,
                0,
                Struct::from_iter([Some(Literal::long(2))]),
            )),
            deleted_entry(
                &fixture,
                posdel_data_file(
                    &del_leaked,
                    size_leaked,
                    1,
                    0,
                    Struct::from_iter([Some(Literal::long(1))]),
                ),
            ),
            added_entry(
                DataFileBuilder::default()
                    .partition_spec_id(0)
                    .content(DataContentType::EqualityDeletes)
                    .file_path(del_eq.clone())
                    .file_format(DataFileFormat::Parquet)
                    .file_size_in_bytes(size_eq)
                    .record_count(1)
                    .partition(Struct::from_iter([Some(Literal::long(1))]))
                    .equality_ids(Some(vec![1]))
                    .key_metadata(None)
                    .build()
                    .expect("equality delete file"),
            ),
        ],
        true,
    )
    .await;
    write_manifest_list(&fixture, vec![data_manifest, delete_manifest], false).await;

    let batch = single_batch(collect_posdel_batches(&fixture).await);
    assert_eq!(batch.num_rows(), 2, "only live position deletes");

    let file_paths = string_column(&batch, "file_path");
    let positions = int64_column(&batch, "pos");
    let spec_ids = int32_column(&batch, "spec_id");
    let delete_file_paths = string_column(&batch, "delete_file_path");
    let partition = partition_struct(&batch);
    let rows = batch
        .column_by_name("row")
        .expect("row column")
        .as_any()
        .downcast_ref::<arrow_array::StructArray>()
        .expect("row must be a struct");

    let mut pairs: Vec<(String, i64, Option<i64>, i32, String)> = (0..2)
        .map(|index| {
            (
                file_paths.value(index).to_string(),
                positions.value(index),
                partition_i64(partition, "x", index),
                spec_ids.value(index),
                delete_file_paths.value(index).to_string(),
            )
        })
        .collect();
    pairs.sort();
    assert_eq!(
        pairs,
        vec![
            (data_a.clone(), 1, Some(1), 0, del_a.clone()),
            (data_b.clone(), 1, Some(2), 0, del_b.clone()),
        ]
    );
    for index in 0..2 {
        assert!(rows.is_null(index), "row must be NULL when the file stores none");
    }
}

#[tokio::test]
async fn scan_unpartitioned_v2_drops_partition_column() {
    let fixture = TableTestFixture::new_unpartitioned();
    std::fs::create_dir_all(format!("{}/metadata", fixture.table_location))
        .expect("metadata dir");

    let data_a = format!("{}/data/a.parquet", fixture.table_location);
    let del_a = format!("{}/del_a.parquet", fixture.table_location);
    let size_a = write_posdel_parquet(&del_a, &[(data_a.clone(), 3)], None);

    let spec = default_spec(&fixture);
    let delete_manifest = write_manifest(
        &fixture,
        &spec,
        false,
        vec![added_entry(posdel_data_file(
            &del_a,
            size_a,
            1,
            0,
            Struct::empty(),
        ))],
        true,
    )
    .await;
    write_manifest_list(&fixture, vec![delete_manifest], false).await;

    let batch = single_batch(collect_posdel_batches(&fixture).await);
    assert_eq!(batch.num_rows(), 1);
    assert!(batch.column_by_name("partition").is_none());
    assert_eq!(string_column(&batch, "file_path").value(0), data_a);
    assert_eq!(int64_column(&batch, "pos").value(0), 3);
    assert_eq!(string_column(&batch, "delete_file_path").value(0), del_a);
    assert_eq!(int32_column(&batch, "spec_id").value(0), 0);
}

#[tokio::test]
async fn scan_partitioned_v3_dv_rows_match_oracle() {
    let mut fixture = TableTestFixture::new();
    std::fs::create_dir_all(format!("{}/metadata", fixture.table_location))
        .expect("metadata dir");
    upgrade_fixture_to_v3(&mut fixture);

    let data_a = format!("{}/data/x=1/a.parquet", fixture.table_location);
    let data_b = format!("{}/data/x=2/b.parquet", fixture.table_location);
    let puffin_path = format!("{}/data/deletes.puffin", fixture.table_location);
    let (puffin_size, coordinates) = write_dv_puffin(
        &fixture,
        &puffin_path,
        &[(&data_a, &[1_u64]), (&data_b, &[7_u64])],
    )
    .await;

    let spec = default_spec(&fixture);
    let dv_file = |data_path: &str, partition: i64, coordinate: (i64, i64)| {
        DataFileBuilder::default()
            .partition_spec_id(0)
            .content(DataContentType::PositionDeletes)
            .file_path(puffin_path.clone())
            .file_format(DataFileFormat::Puffin)
            .file_size_in_bytes(puffin_size)
            .record_count(1)
            .referenced_data_file(Some(data_path.to_string()))
            .content_offset(Some(coordinate.0))
            .content_size_in_bytes(Some(coordinate.1))
            .partition(Struct::from_iter([Some(Literal::long(partition))]))
            .key_metadata(None)
            .build()
            .expect("dv data file")
    };
    let delete_manifest = write_manifest(
        &fixture,
        &spec,
        true,
        vec![
            added_entry(dv_file(&data_a, 1, coordinates[0])),
            added_entry(dv_file(&data_b, 2, coordinates[1])),
        ],
        true,
    )
    .await;
    write_manifest_list(&fixture, vec![delete_manifest], true).await;

    let batch = single_batch(collect_posdel_batches(&fixture).await);
    assert_eq!(batch.num_rows(), 2);

    let file_paths = string_column(&batch, "file_path");
    let positions = int64_column(&batch, "pos");
    let offsets = int64_column(&batch, "content_offset");
    let sizes = int64_column(&batch, "content_size_in_bytes");
    let rows = batch
        .column_by_name("row")
        .expect("row column")
        .as_any()
        .downcast_ref::<arrow_array::StructArray>()
        .expect("row must be a struct");

    let mut seen: Vec<(String, i64, i64, i64)> = (0..2)
        .map(|index| {
            (
                file_paths.value(index).to_string(),
                positions.value(index),
                offsets.value(index),
                sizes.value(index),
            )
        })
        .collect();
    seen.sort();
    assert_eq!(
        seen,
        vec![
            (data_a.clone(), 1, coordinates[0].0, coordinates[0].1),
            (data_b.clone(), 7, coordinates[1].0, coordinates[1].1),
        ]
    );
    for index in 0..2 {
        assert!(rows.is_null(index), "DV rows carry no row payload");
        assert!(
            !offsets.is_null(index) && !sizes.is_null(index),
            "DV coordinates must be populated"
        );
    }
}

#[tokio::test]
async fn scan_evolved_two_specs_null_fills_unified_partition() {
    let fixture = TableTestFixture::new_with_widening_spec_evolution();
    std::fs::create_dir_all(format!("{}/metadata", fixture.table_location))
        .expect("metadata dir");

    let data_a = format!("{}/data/x=5/a.parquet", fixture.table_location);
    let data_b = format!("{}/data/x=5/b.parquet", fixture.table_location);
    let del_a = format!("{}/del_a.parquet", fixture.table_location);
    let del_b = format!("{}/del_b.parquet", fixture.table_location);
    let size_a = write_posdel_parquet(&del_a, &[(data_a.clone(), 1)], None);
    let size_b = write_posdel_parquet(&del_b, &[(data_b.clone(), 4)], None);

    let spec_zero = fixture
        .table
        .metadata()
        .partition_spec_by_id(0)
        .expect("spec 0")
        .as_ref()
        .clone();
    let spec_one = fixture
        .table
        .metadata()
        .partition_spec_by_id(1)
        .expect("spec 1")
        .as_ref()
        .clone();

    let manifest_spec_zero = write_manifest(
        &fixture,
        &spec_zero,
        false,
        vec![added_entry(posdel_data_file(
            &del_a,
            size_a,
            1,
            0,
            Struct::from_iter([Some(Literal::long(5))]),
        ))],
        true,
    )
    .await;
    let manifest_spec_one = write_manifest(
        &fixture,
        &spec_one,
        false,
        vec![added_entry(posdel_data_file(
            &del_b,
            size_b,
            1,
            1,
            Struct::from_iter([Some(Literal::long(5)), Some(Literal::int(3))]),
        ))],
        true,
    )
    .await;
    write_manifest_list(
        &fixture,
        vec![manifest_spec_zero, manifest_spec_one],
        false,
    )
    .await;

    let batch = single_batch(collect_posdel_batches(&fixture).await);
    assert_eq!(batch.num_rows(), 2);
    let partition = partition_struct(&batch);
    let spec_ids = int32_column(&batch, "spec_id");
    let delete_file_paths = string_column(&batch, "delete_file_path");
    let positions = int64_column(&batch, "pos");

    let mut seen: Vec<(String, i64, i32, Option<i64>, Option<i32>)> = (0..2)
        .map(|index| {
            (
                delete_file_paths.value(index).to_string(),
                positions.value(index),
                spec_ids.value(index),
                partition_i64(partition, "x", index),
                partition_i32(partition, "y_bucket_8", index),
            )
        })
        .collect();
    seen.sort();
    assert_eq!(
        seen,
        vec![
            (del_a.clone(), 1, 0, Some(5), None),
            (del_b.clone(), 4, 1, Some(5), Some(3)),
        ]
    );
}

#[tokio::test]
async fn scan_row_column_is_read_when_the_file_carries_it() {
    let fixture = TableTestFixture::new();
    std::fs::create_dir_all(format!("{}/metadata", fixture.table_location))
        .expect("metadata dir");

    let data_a = format!("{}/data/x=1/a.parquet", fixture.table_location);
    let data_b = format!("{}/data/x=1/b.parquet", fixture.table_location);
    let del_plain = format!("{}/del_plain.parquet", fixture.table_location);
    let del_rowed = format!("{}/del_rowed.parquet", fixture.table_location);

    let row_batch = table_row_batch(fixture.table.metadata().current_schema(), 7);
    let size_plain = write_posdel_parquet(&del_plain, &[(data_a.clone(), 1)], None);
    let size_rowed = write_posdel_parquet(
        &del_rowed,
        &[(data_b.clone(), 2)],
        Some(&row_batch),
    );

    let spec = default_spec(&fixture);
    let delete_manifest = write_manifest(
        &fixture,
        &spec,
        false,
        vec![
            added_entry(posdel_data_file(
                &del_plain,
                size_plain,
                1,
                0,
                Struct::from_iter([Some(Literal::long(1))]),
            )),
            added_entry(posdel_data_file(
                &del_rowed,
                size_rowed,
                1,
                0,
                Struct::from_iter([Some(Literal::long(1))]),
            )),
        ],
        true,
    )
    .await;
    write_manifest_list(&fixture, vec![delete_manifest], false).await;

    let batch = single_batch(collect_posdel_batches(&fixture).await);
    assert_eq!(batch.num_rows(), 2);
    let file_paths = string_column(&batch, "file_path");
    let rows = batch
        .column_by_name("row")
        .expect("row column")
        .as_any()
        .downcast_ref::<arrow_array::StructArray>()
        .expect("row must be a struct");

    let mut saw_plain_null = false;
    let mut saw_rowed = false;
    for index in 0..2 {
        match file_paths.value(index) {
            path if path == data_a => {
                assert!(rows.is_null(index));
                saw_plain_null = true;
            }
            path if path == data_b => {
                assert!(rows.is_valid(index), "stored row must be surfaced");
                let x = rows
                    .column_by_name("x")
                    .expect("row.x")
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("row.x int64");
                assert_eq!(x.value(index), 7);
                let a = rows
                    .column_by_name("a")
                    .expect("row.a")
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("row.a utf8");
                assert_eq!(a.value(index), "row-string");
                saw_rowed = true;
            }
            other => panic!("unexpected file_path {other}"),
        }
    }
    assert!(saw_plain_null && saw_rowed);
}

#[tokio::test]
async fn scan_v2_output_has_no_dv_columns() {
    let fixture = TableTestFixture::new();
    std::fs::create_dir_all(format!("{}/metadata", fixture.table_location))
        .expect("metadata dir");

    let del_a = format!("{}/del_a.parquet", fixture.table_location);
    let size_a = write_posdel_parquet(
        &del_a,
        &[(format!("{}/data/a.parquet", fixture.table_location), 1)],
        None,
    );
    let spec = default_spec(&fixture);
    let delete_manifest = write_manifest(
        &fixture,
        &spec,
        false,
        vec![added_entry(posdel_data_file(
            &del_a,
            size_a,
            1,
            0,
            Struct::from_iter([Some(Literal::long(1))]),
        ))],
        true,
    )
    .await;
    write_manifest_list(&fixture, vec![delete_manifest], false).await;

    let batches = collect_posdel_batches(&fixture).await;
    for batch in &batches {
        assert!(batch.column_by_name("content_offset").is_none());
        assert!(batch.column_by_name("content_size_in_bytes").is_none());
    }
}

#[tokio::test]
async fn scan_empty_table_emits_no_rows() {
    let fixture = TableTestFixture::new();
    std::fs::create_dir_all(format!("{}/metadata", fixture.table_location))
        .expect("metadata dir");

    let spec = default_spec(&fixture);
    let data_manifest = write_manifest(
        &fixture,
        &spec,
        false,
        vec![added_entry(
            DataFileBuilder::default()
                .partition_spec_id(0)
                .content(DataContentType::Data)
                .file_path(format!("{}/data/a.parquet", fixture.table_location))
                .file_format(DataFileFormat::Parquet)
                .file_size_in_bytes(100)
                .record_count(1)
                .partition(Struct::from_iter([Some(Literal::long(1))]))
                .key_metadata(None)
                .build()
                .expect("data file"),
        )],
        false,
    )
    .await;
    write_manifest_list(&fixture, vec![data_manifest], false).await;

    let total: usize = collect_posdel_batches(&fixture)
        .await
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert_eq!(total, 0);
}
