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

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use futures::TryStreamExt;
use parquet::basic::Compression;
use parquet::file::reader::{FileReader, SerializedFileReader};

use crate::arrow::{FieldMatchMode, schema_to_arrow_schema};
use crate::maintenance::RewriteDataFilesResult;
use crate::maintenance::rewrite_data_files::RewriteDataFiles;
use crate::maintenance::rewrite_data_files::tests::{
    append_files, local_fs_catalog, scan_rows, write_data_file,
};
use crate::scan::FileScanTask;
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, Literal, MetricsConfig, NestedField,
    PartitionSpec, PrimitiveType, Schema, SchemaRef, Struct, TableProperties, Transform, Type,
};
use crate::table::Table;
use crate::writer::file_writer::{AnyFileWriterBuilder, FileWriter, FileWriterBuilder};
use crate::{Catalog, ErrorKind, NamespaceIdent, TableCreation};

fn three_long_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "x",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                2,
                "y",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                3,
                "z",
                Type::Primitive(PrimitiveType::Long),
            )),
        ])
        .build()
        .expect("build the three-long schema")
}

async fn create_codec_table(
    catalog: &impl Catalog,
    format_version: FormatVersion,
    properties: HashMap<String, String>,
) -> Table {
    let schema = three_long_schema();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("x", "x", Transform::Identity)
        .expect("add the partition field")
        .build()
        .expect("build the spec");
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation {
        name: "t".to_string(),
        location: None,
        schema,
        partition_spec: Some(spec.into_unbound()),
        sort_order: None,
        properties,
        format_version,
    };
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create the codec table")
}

fn codec_properties(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect()
}

async fn write_batch_in_format(
    table: &Table,
    file_name: &str,
    batch: &RecordBatch,
    schema: SchemaRef,
    format: DataFileFormat,
    partition: Struct,
) -> DataFile {
    let file_path = format!("{}/data/{file_name}", table.metadata().location());
    let output = table.file_io().new_output(file_path).expect("output file");
    let builder = AnyFileWriterBuilder::for_format(
        format,
        schema,
        table.metadata().properties(),
        MetricsConfig::for_table(table.metadata()).expect("metrics config"),
        FieldMatchMode::Id,
    )
    .expect("route the fixture format");
    let mut writer = builder.build(output).await.expect("build the writer");
    writer.write(batch).await.expect("write the rows");
    let mut data_file = writer
        .close()
        .await
        .expect("close the writer")
        .into_iter()
        .next()
        .expect("one data file builder");
    data_file
        .content(DataContentType::Data)
        .partition_spec_id(0)
        .partition(partition);
    data_file.build().expect("build the data file")
}

async fn write_data_file_in_format(
    table: &Table,
    file_name: &str,
    part_value: i64,
    rows: &[(i64, i64, i64)],
    format: DataFileFormat,
) -> DataFile {
    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("arrow schema"));
    let xs: Vec<i64> = rows.iter().map(|(x, _, _)| *x).collect();
    let ys: Vec<i64> = rows.iter().map(|(_, y, _)| *y).collect();
    let zs: Vec<i64> = rows.iter().map(|(_, _, z)| *z).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(xs)) as ArrayRef,
        Arc::new(Int64Array::from(ys)) as ArrayRef,
        Arc::new(Int64Array::from(zs)) as ArrayRef,
    ])
    .expect("build the rows batch");
    write_batch_in_format(
        table,
        file_name,
        &batch,
        schema.clone(),
        format,
        Struct::from_iter([Some(Literal::long(part_value))]),
    )
    .await
}

async fn write_numbered_format_files(
    table: &Table,
    prefix: &str,
    format: DataFileFormat,
    row_sets: Vec<Vec<(i64, i64, i64)>>,
) -> Vec<DataFile> {
    let mut files = Vec::with_capacity(row_sets.len());
    for (index, rows) in row_sets.iter().enumerate() {
        files.push(
            write_data_file_in_format(
                table,
                &format!("{prefix}-{index}.{format}"),
                0,
                rows,
                format,
            )
            .await,
        );
    }
    files
}

async fn compact_and_reload(
    catalog: &impl Catalog,
    table: &Table,
    target: u64,
) -> (RewriteDataFilesResult, Table) {
    let result = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(target)
        .execute(catalog)
        .await
        .expect("execute the compaction");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload the table");
    (result, table)
}

async fn live_data_files(table: &Table) -> Vec<DataFile> {
    let mut files = Vec::new();
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load the manifest list");
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("load the manifest");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.data_file().content_type() == DataContentType::Data {
                files.push(entry.data_file().clone());
            }
        }
    }
    files.sort_by(|left, right| left.file_path().cmp(right.file_path()));
    files
}

async fn planned_group(table: &Table) -> Vec<FileScanTask> {
    table
        .scan()
        .build()
        .expect("scan builder")
        .plan_files()
        .await
        .expect("plan files")
        .try_collect()
        .await
        .expect("collect tasks")
}

async fn parquet_column_compressions(table: &Table, path: &str) -> Vec<Compression> {
    let bytes = table
        .file_io()
        .new_input(path)
        .expect("open the parquet file")
        .read()
        .await
        .expect("read the parquet file");
    let reader = SerializedFileReader::new(bytes).expect("read the parquet footer");
    let mut codecs = Vec::new();
    for group in reader.metadata().row_groups() {
        for column in group.columns() {
            codecs.push(column.compression());
        }
    }
    assert!(!codecs.is_empty(), "the output must hold column chunks");
    codecs
}

fn read_uvarint(rest: &[u8]) -> (u64, &[u8]) {
    let mut value = 0u64;
    let mut shift = 0u32;
    let mut consumed = 0usize;
    loop {
        assert!(
            shift < 70,
            "the postscript holds a malformed varint past ten bytes"
        );
        let byte = *rest
            .get(consumed)
            .expect("the postscript ends inside a varint");
        consumed += 1;
        value |= u64::from(byte & 0x7f) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            return (value, &rest[consumed..]);
        }
    }
}

fn postscript_compression_kind(postscript: &[u8]) -> u64 {
    let mut rest = postscript;
    loop {
        assert!(!rest.is_empty(), "the postscript must carry field 2");
        let (tag, tail) = read_uvarint(rest);
        rest = tail;
        match tag & 7 {
            0 => {
                let (value, tail) = read_uvarint(rest);
                rest = tail;
                if tag >> 3 == 2 {
                    return value;
                }
            }
            2 => {
                let (length, tail) = read_uvarint(rest);
                let length =
                    usize::try_from(length).expect("the postscript holds a length past usize");
                assert!(
                    tail.len() >= length,
                    "the postscript ends inside a length-prefixed field"
                );
                rest = &tail[length..];
            }
            5 => {
                assert!(rest.len() >= 4, "the postscript ends inside a fixed32");
                rest = &rest[4..];
            }
            wire => panic!("the postscript carries unexpected wire type {wire}"),
        }
    }
}

async fn orc_postscript_compression(table: &Table, path: &str) -> u64 {
    let bytes = table
        .file_io()
        .new_input(path)
        .expect("open the orc file")
        .read()
        .await
        .expect("read the orc file");
    assert!(
        bytes.len() > 8,
        "an orc file must hold a postscript, got {} bytes",
        bytes.len()
    );
    assert_eq!(
        &bytes[bytes.len() - 4..bytes.len() - 1],
        b"ORC",
        "the file must close with the ORC magic"
    );
    let ps_len = usize::from(bytes[bytes.len() - 1]);
    assert!(
        ps_len + 1 < bytes.len(),
        "the postscript length must fit the file"
    );
    let postscript = &bytes[bytes.len() - 1 - ps_len..bytes.len() - 1];
    assert_eq!(
        &postscript[postscript.len() - 3..],
        b"ORC",
        "the postscript must end with the ORC magic"
    );
    postscript_compression_kind(postscript)
}

async fn compact_six_parquet_files(
    catalog: &impl Catalog,
    properties: HashMap<String, String>,
) -> Table {
    let table = create_codec_table(catalog, FormatVersion::V2, properties).await;
    let mut files = Vec::new();
    for index in 0..6i64 {
        files.push(
            write_data_file(&table, &format!("small-{index}.parquet"), 0, &[(
                0,
                100 + index,
                1000 + index,
            )])
            .await,
        );
    }
    let table = append_files(catalog, &table, files).await;
    assert_eq!(
        scan_rows(&table).await.len(),
        6,
        "fixture: six rows before compaction"
    );
    let (result, table) = compact_and_reload(catalog, &table, 1_000_000).await;
    assert_eq!(result.rewritten_data_files_count, 6);
    assert!(result.added_data_files_count >= 1);
    table
}

async fn compact_six_orc_files(
    catalog: &impl Catalog,
    properties: HashMap<String, String>,
) -> Table {
    let table = create_codec_table(catalog, FormatVersion::V2, properties).await;
    let row_sets: Vec<Vec<(i64, i64, i64)>> = (0..6i64)
        .map(|index| vec![(0, 100 + index, 1000 + index)])
        .collect();
    let files = write_numbered_format_files(&table, "small", DataFileFormat::Orc, row_sets).await;
    let table = append_files(catalog, &table, files).await;
    assert_eq!(
        scan_rows(&table).await.len(),
        6,
        "fixture: six rows before compaction"
    );
    let (result, table) = compact_and_reload(catalog, &table, 1_000_000).await;
    assert_eq!(result.rewritten_data_files_count, 6);
    assert!(result.added_data_files_count >= 1);
    table
}

#[tokio::test]
async fn parquet_compaction_defaults_every_column_to_zstd() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = compact_six_parquet_files(&catalog, HashMap::new()).await;
    let files = live_data_files(&table).await;
    assert!(
        !files.is_empty() && files.len() < 6,
        "compaction must leave fewer files, got {}",
        files.len()
    );
    for file in &files {
        assert_eq!(file.file_format(), DataFileFormat::Parquet);
        for codec in parquet_column_compressions(&table, file.file_path()).await {
            assert!(
                matches!(codec, Compression::ZSTD(_)),
                "the default parquet codec must be zstd, got {codec:?}"
            );
        }
    }
    assert_eq!(
        scan_rows(&table).await.len(),
        6,
        "compaction must conserve every row"
    );
}

#[tokio::test]
async fn parquet_compaction_honours_the_uncompressed_codec() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = compact_six_parquet_files(
        &catalog,
        codec_properties(&[(
            TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC,
            "uncompressed",
        )]),
    )
    .await;
    let files = live_data_files(&table).await;
    assert!(
        !files.is_empty() && files.len() < 6,
        "compaction must leave fewer files, got {}",
        files.len()
    );
    for file in &files {
        assert_eq!(file.file_format(), DataFileFormat::Parquet);
        for codec in parquet_column_compressions(&table, file.file_path()).await {
            assert_eq!(
                codec,
                Compression::UNCOMPRESSED,
                "the codec property must reach the compacted columns"
            );
        }
    }
    assert_eq!(
        scan_rows(&table).await.len(),
        6,
        "compaction must conserve every row"
    );
}

#[tokio::test]
async fn orc_compaction_records_lower_bounds_for_field_two() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = compact_six_orc_files(
        &catalog,
        codec_properties(&[("write.format.default", "orc")]),
    )
    .await;
    let files = live_data_files(&table).await;
    assert!(
        !files.is_empty() && files.len() < 6,
        "compaction must leave fewer files, got {}",
        files.len()
    );
    for file in &files {
        assert_eq!(file.file_format(), DataFileFormat::Orc);
        assert!(
            file.lower_bounds().contains_key(&2),
            "the compacted orc file must bound field 2, got {:?}",
            file.file_path()
        );
        assert!(
            file.upper_bounds().contains_key(&2),
            "the compacted orc file must bound field 2 above, got {:?}",
            file.file_path()
        );
    }
    assert_eq!(
        scan_rows(&table).await.len(),
        6,
        "compaction must conserve every row"
    );
}

#[tokio::test]
async fn orc_compaction_with_metrics_none_writes_no_bounds() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = compact_six_orc_files(
        &catalog,
        codec_properties(&[
            ("write.format.default", "orc"),
            ("write.metadata.metrics.default", "none"),
        ]),
    )
    .await;
    let files = live_data_files(&table).await;
    assert!(
        !files.is_empty() && files.len() < 6,
        "compaction must leave fewer files, got {}",
        files.len()
    );
    for file in &files {
        assert_eq!(file.file_format(), DataFileFormat::Orc);
        assert!(
            file.lower_bounds().is_empty(),
            "mode none must leave lower bounds empty, got {:?}",
            file.file_path()
        );
        assert!(
            file.upper_bounds().is_empty(),
            "mode none must leave upper bounds empty, got {:?}",
            file.file_path()
        );
    }
    assert_eq!(
        scan_rows(&table).await.len(),
        6,
        "compaction must conserve every row"
    );
}

#[tokio::test]
async fn orc_compaction_honours_the_none_compression_codec() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = compact_six_orc_files(
        &catalog,
        codec_properties(&[
            ("write.format.default", "orc"),
            (TableProperties::PROPERTY_ORC_COMPRESSION_CODEC, "none"),
        ]),
    )
    .await;
    let files = live_data_files(&table).await;
    assert!(
        !files.is_empty() && files.len() < 6,
        "compaction must leave fewer files, got {}",
        files.len()
    );
    for file in &files {
        assert_eq!(file.file_format(), DataFileFormat::Orc);
        assert_eq!(
            orc_postscript_compression(&table, file.file_path()).await,
            0,
            "the codec property must reach the compacted postscript"
        );
    }
    assert_eq!(
        scan_rows(&table).await.len(),
        6,
        "compaction must conserve every row"
    );
}

#[tokio::test]
async fn orc_compaction_refuses_a_bad_stripe_size() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = create_codec_table(
        &catalog,
        FormatVersion::V2,
        codec_properties(&[
            ("write.format.default", "orc"),
            (TableProperties::PROPERTY_ORC_STRIPE_SIZE_BYTES, "bogus"),
        ]),
    )
    .await;
    let file = write_data_file(&table, "small-0.parquet", 0, &[(0, 100, 1000)]).await;
    let table = append_files(&catalog, &table, vec![file]).await;
    let group = planned_group(&table).await;
    assert_eq!(group.len(), 1, "fixture: one planned task");
    let outcome = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(1_000_000)
        .write_group_for_test(&table, &group)
        .await;
    let error = match outcome {
        Ok(_) => panic!("a bad orc stripe size must refuse the write"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "Invalid value for write.orc.stripe-size-bytes: bogus",
        "the refusal must carry the typed bare message"
    );
}

#[tokio::test]
async fn parquet_compaction_records_lower_bounds_for_field_two() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = compact_six_parquet_files(&catalog, HashMap::new()).await;
    let files = live_data_files(&table).await;
    assert!(
        !files.is_empty() && files.len() < 6,
        "compaction must leave fewer files, got {}",
        files.len()
    );
    for file in &files {
        assert_eq!(file.file_format(), DataFileFormat::Parquet);
        assert!(
            file.lower_bounds().contains_key(&2),
            "the compacted parquet file must bound field 2, got {:?}",
            file.file_path()
        );
        assert!(
            file.upper_bounds().contains_key(&2),
            "the compacted parquet file must bound field 2 above, got {:?}",
            file.file_path()
        );
    }
    assert_eq!(
        scan_rows(&table).await.len(),
        6,
        "compaction must conserve every row"
    );
}

#[tokio::test]
async fn parquet_compaction_with_metrics_none_writes_no_bounds() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = compact_six_parquet_files(
        &catalog,
        codec_properties(&[("write.metadata.metrics.default", "none")]),
    )
    .await;
    let files = live_data_files(&table).await;
    assert!(
        !files.is_empty() && files.len() < 6,
        "compaction must leave fewer files, got {}",
        files.len()
    );
    for file in &files {
        assert_eq!(file.file_format(), DataFileFormat::Parquet);
        assert!(
            file.lower_bounds().is_empty(),
            "mode none must leave lower bounds empty, got {:?}",
            file.file_path()
        );
        assert!(
            file.upper_bounds().is_empty(),
            "mode none must leave upper bounds empty, got {:?}",
            file.file_path()
        );
    }
    assert_eq!(
        scan_rows(&table).await.len(),
        6,
        "compaction must conserve every row"
    );
}
