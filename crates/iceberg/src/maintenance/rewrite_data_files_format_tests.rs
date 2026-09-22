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

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use parquet::basic::Encoding;
use parquet::file::reader::{FileReader, SerializedFileReader};

use crate::arrow::{FieldMatchMode, schema_to_arrow_schema};
use crate::maintenance::rewrite_data_files::RewriteDataFiles;
use crate::maintenance::rewrite_data_files::tests::{
    append_files, local_fs_catalog, scan_rows, write_data_file,
};
use crate::maintenance::rewrite_data_files_sort::ResolvedStrategy;
use crate::maintenance::rewrite_data_files_sort_key::KeyPlan;
use crate::maintenance::rewrite_data_files_sort_run::{ExternalSorter, SortedBatchSink};
use crate::maintenance::{RewriteDataFilesResult, RewriteStrategy};
use crate::metadata_columns::{
    RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER, RESERVED_COL_NAME_ROW_ID,
};
use crate::scan::FileScanTask;
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, Literal, MetricsConfig, NestedField,
    NullOrder, PartitionSpec, PrimitiveType, Schema, SchemaRef, SortDirection, SortField,
    SortOrder, Struct, Transform, Type,
};
use crate::table::Table;
use crate::writer::file_writer::{AnyFileWriterBuilder, FileWriter, FileWriterBuilder};
use crate::{Catalog, Error, ErrorKind, NamespaceIdent, Result, TableCreation};

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

async fn create_format_table(
    catalog: &impl Catalog,
    format_version: FormatVersion,
    format: Option<&str>,
    sort_order: Option<SortOrder>,
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
    let properties = format
        .map(|name| HashMap::from([("write.format.default".to_string(), name.to_string())]))
        .unwrap_or_default();
    let creation = TableCreation {
        name: "t".to_string(),
        location: None,
        schema,
        partition_spec: Some(spec.into_unbound()),
        sort_order,
        properties,
        format_version,
    };
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create the format table")
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

fn assert_all_format(files: &[DataFile], format: DataFileFormat, reason: &str) {
    let suffix = format!(".{format}");
    for file in files {
        assert_eq!(
            file.file_format(),
            format,
            "{reason}, got {}",
            file.file_path()
        );
        assert!(
            file.file_path().ends_with(suffix.as_str()),
            "an output must carry the {format} extension, got {}",
            file.file_path()
        );
    }
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

async fn parquet_column_dictionary_flags(table: &Table, path: &str) -> Vec<bool> {
    let bytes = table
        .file_io()
        .new_input(path)
        .expect("open the parquet file")
        .read()
        .await
        .expect("read the parquet file");
    let reader = SerializedFileReader::new(bytes).expect("read the parquet footer");
    let mut flags = Vec::new();
    for group in reader.metadata().row_groups() {
        for column in group.columns() {
            flags.push(column.encodings().any(|encoding| {
                matches!(
                    encoding,
                    Encoding::PLAIN_DICTIONARY | Encoding::RLE_DICTIONARY
                )
            }));
        }
    }
    flags
}

fn spill_files_under(table: &Table) -> Vec<String> {
    let location = table
        .metadata()
        .location()
        .trim_start_matches("file:/")
        .to_string();
    let mut found = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(format!("/{location}"))];
    while let Some(directory) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.to_string_lossy().contains("rewrite-sort-spill-") {
                found.push(path.to_string_lossy().to_string());
            }
        }
    }
    found.sort();
    found
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

async fn scan_lineage(table: &Table) -> Vec<(i64, i64, i64)> {
    let stream = table
        .scan()
        .select([
            "y",
            RESERVED_COL_NAME_ROW_ID,
            RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER,
        ])
        .build()
        .expect("lineage scan")
        .to_arrow()
        .await
        .expect("lineage batches");
    let batches: Vec<RecordBatch> = stream.try_collect().await.expect("collect lineage");
    let mut rows = Vec::new();
    for batch in batches {
        let ys = batch
            .column_by_name("y")
            .expect("y column")
            .as_primitive::<Int64Type>();
        let row_ids = batch
            .column_by_name(RESERVED_COL_NAME_ROW_ID)
            .expect("_row_id column")
            .as_primitive::<Int64Type>();
        let seqs = batch
            .column_by_name(RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER)
            .expect("sequence column")
            .as_primitive::<Int64Type>();
        for index in 0..batch.num_rows() {
            assert!(row_ids.is_valid(index), "a v3 row must carry a _row_id");
            assert!(seqs.is_valid(index), "a v3 row must carry a sequence");
            rows.push((ys.value(index), row_ids.value(index), seqs.value(index)));
        }
    }
    rows.sort_unstable();
    rows
}

fn sort_probe_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            Arc::new(NestedField::optional(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                2,
                "s",
                Type::Primitive(PrimitiveType::String),
            )),
        ])
        .build()
        .expect("build the sort probe schema")
}

fn id_ascending() -> SortOrder {
    SortOrder {
        order_id: 1,
        fields: vec![SortField {
            source_id: 1,
            transform: Transform::Identity,
            direction: SortDirection::Ascending,
            null_order: NullOrder::First,
        }],
    }
}

fn scattered_ids(count: i64) -> Vec<i64> {
    (0..count).map(|row| (row * 7919) % count).collect()
}

fn sort_probe_batch(arrow_schema: &Arc<arrow_schema::Schema>, ids: &[i64]) -> RecordBatch {
    let strings: Vec<String> = ids
        .iter()
        .map(|id| format!("row-{id:08}-padding"))
        .collect();
    RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
        Arc::new(StringArray::from(strings)) as ArrayRef,
    ])
    .expect("build the sort probe batch")
}

fn probe_sorter(table: &Table) -> (ExternalSorter, Arc<arrow_schema::Schema>) {
    let iceberg_schema = sort_probe_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(&iceberg_schema).expect("arrow schema"));
    let strategy = ResolvedStrategy::Sort {
        order: id_ascending(),
        stamp: 1,
    };
    let plan = KeyPlan::build(&strategy, &iceberg_schema, &arrow_schema)
        .expect("build the key plan")
        .expect("a sort strategy resolves a key plan");
    let spill_prefix = format!(
        "{}/data/rewrite-sort-spill-probe-{}",
        table.metadata().location(),
        uuid::Uuid::now_v7()
    );
    let sorter = ExternalSorter::new(
        table.file_io().clone(),
        spill_prefix,
        32 * 1024,
        arrow_schema.clone(),
        plan,
    );
    (sorter, arrow_schema)
}

struct CollectSink {
    batches: Vec<RecordBatch>,
}

impl SortedBatchSink for CollectSink {
    async fn write_sorted(&mut self, batch: RecordBatch) -> Result<()> {
        self.batches.push(batch);
        Ok(())
    }
}

struct RefuseSink;

impl SortedBatchSink for RefuseSink {
    async fn write_sorted(&mut self, _batch: RecordBatch) -> Result<()> {
        Err(Error::new(
            ErrorKind::Unexpected,
            "the probe sink refuses the merge",
        ))
    }
}

fn batch_ids(batch: &RecordBatch) -> Vec<i64> {
    let column = batch.column_by_name("id").expect("id column").clone();
    let mut ids = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        assert!(!column.is_null(row), "the probe ids are never null");
        ids.push(column.as_primitive::<Int64Type>().value(row));
    }
    ids
}

#[tokio::test]
async fn test_compaction_keeps_table_format() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = create_format_table(&catalog, FormatVersion::V2, Some("orc"), None).await;
    let row_sets: Vec<Vec<(i64, i64, i64)>> = (0..6i64)
        .map(|index| vec![(0, 100 + index, 1000 + index)])
        .collect();
    let files = write_numbered_format_files(&table, "small", DataFileFormat::Orc, row_sets).await;
    let table = append_files(&catalog, &table, files).await;
    let rows_before = scan_rows(&table).await;
    assert_eq!(rows_before.len(), 6, "fixture: six rows before compaction");

    let (result, table) = compact_and_reload(&catalog, &table, 1_000_000).await;
    assert_eq!(result.rewritten_data_files_count, 6);
    assert!(result.added_data_files_count >= 1);

    let files = live_data_files(&table).await;
    assert!(
        !files.is_empty() && files.len() < 6,
        "compaction must leave fewer files, got {}",
        files.len()
    );
    assert_all_format(
        &files,
        DataFileFormat::Orc,
        "compaction on an orc table must leave orc files",
    );
    assert_eq!(
        scan_rows(&table).await,
        rows_before,
        "compaction must conserve every row"
    );
}

#[tokio::test]
async fn avro_compaction_keeps_table_format() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = create_format_table(&catalog, FormatVersion::V2, Some("avro"), None).await;
    let row_sets: Vec<Vec<(i64, i64, i64)>> = (0..6i64)
        .map(|index| vec![(0, 200 + index, 2000 + index)])
        .collect();
    let files = write_numbered_format_files(&table, "small", DataFileFormat::Avro, row_sets).await;
    let table = append_files(&catalog, &table, files).await;
    let rows_before = scan_rows(&table).await;
    assert_eq!(rows_before.len(), 6, "fixture: six rows before compaction");

    let (result, table) = compact_and_reload(&catalog, &table, 1_000_000).await;
    assert_eq!(result.rewritten_data_files_count, 6);
    assert!(result.added_data_files_count >= 1);

    let files = live_data_files(&table).await;
    assert!(
        !files.is_empty() && files.len() < 6,
        "compaction must leave fewer files, got {}",
        files.len()
    );
    assert_all_format(
        &files,
        DataFileFormat::Avro,
        "compaction on an avro table must leave avro files",
    );
    assert_eq!(
        scan_rows(&table).await,
        rows_before,
        "compaction must conserve every row"
    );
}

async fn compact_mixed_cardinality_parquet(catalog: &impl Catalog) -> Table {
    let table = create_format_table(catalog, FormatVersion::V2, None, None).await;
    let mut files = Vec::new();
    for file in 0..6i64 {
        let rows: Vec<(i64, i64, i64)> = (0..2000i64)
            .map(|row| {
                let ordinal = file * 2000 + row;
                (0, ordinal % 8, ordinal)
            })
            .collect();
        files.push(write_data_file(&table, &format!("mixed-{file}.parquet"), 0, &rows).await);
    }
    let table = append_files(catalog, &table, files).await;
    let (_, table) = compact_and_reload(catalog, &table, 100_000_000).await;
    table
}

#[tokio::test]
async fn parquet_compaction_keeps_dictionary_on_for_low_cardinality_columns() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = compact_mixed_cardinality_parquet(&catalog).await;
    let files = live_data_files(&table).await;
    assert!(!files.is_empty(), "compaction must leave output files");
    assert_all_format(
        &files,
        DataFileFormat::Parquet,
        "a parquet table must compact to parquet",
    );
    for file in &files {
        let flags = parquet_column_dictionary_flags(&table, file.file_path()).await;
        assert!(!flags.is_empty(), "the output must hold column chunks");
        assert_eq!(
            flags.len() % 3,
            0,
            "three columns per row group, got {} flags",
            flags.len()
        );
        for group in flags.chunks_exact(3) {
            assert!(
                group[0] && group[1],
                "the constant and low-cardinality columns must stay dictionary-encoded, got {group:?}"
            );
        }
    }
    assert_eq!(
        scan_rows(&table).await.len(),
        12_000,
        "compaction must conserve every row"
    );
}

#[tokio::test]
async fn parquet_compaction_disables_dictionary_on_fallback_columns() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = compact_mixed_cardinality_parquet(&catalog).await;
    let files = live_data_files(&table).await;
    assert!(!files.is_empty(), "compaction must leave output files");
    for file in &files {
        let flags = parquet_column_dictionary_flags(&table, file.file_path()).await;
        assert!(!flags.is_empty(), "the output must hold column chunks");
        assert_eq!(
            flags.len() % 3,
            0,
            "three columns per row group, got {} flags",
            flags.len()
        );
        for group in flags.chunks_exact(3) {
            assert!(
                !group[2],
                "the unique-per-row column must hit the dictionary fallback, got {group:?}"
            );
        }
    }
}

#[tokio::test]
async fn sort_spill_files_are_parquet_and_vanish_on_success() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = create_format_table(&catalog, FormatVersion::V2, None, None).await;
    let (mut sorter, arrow_schema) = probe_sorter(&table);
    for round in 0..3 {
        let ids: Vec<i64> = scattered_ids(2000)
            .into_iter()
            .map(|id| id + round * 2000)
            .collect();
        sorter
            .push(sort_probe_batch(&arrow_schema, &ids))
            .await
            .expect("push the probe batch");
    }
    let spills = spill_files_under(&table);
    assert!(
        !spills.is_empty(),
        "the probe budget must force at least one spill"
    );
    for spill in &spills {
        let bytes = std::fs::read(spill).expect("read the spill file");
        assert!(
            bytes.len() > 8,
            "a spill file must hold parquet pages, got {} bytes",
            bytes.len()
        );
        assert_eq!(&bytes[0..4], b"PAR1", "a spill file must open with PAR1");
        assert_eq!(
            &bytes[bytes.len() - 4..],
            b"PAR1",
            "a spill file must close with PAR1"
        );
    }
    let mut sink = CollectSink {
        batches: Vec::new(),
    };
    let stats = sorter.finish(&mut sink).await.expect("finish the merge");
    assert_eq!(
        stats.spilled_runs, 3,
        "the pin needs three spilled runs, got {}",
        stats.spilled_runs
    );
    let mut merged = Vec::new();
    for batch in &sink.batches {
        merged.extend(batch_ids(batch));
    }
    assert_eq!(merged.len(), 6000, "the merge must conserve every row");
    let mut ordered = merged.clone();
    ordered.sort_unstable();
    assert_eq!(merged, ordered, "the merge must emit one ascending run");
    assert_eq!(
        spill_files_under(&table),
        Vec::<String>::new(),
        "a spill file survived the successful merge"
    );
}

#[tokio::test]
async fn sort_spill_files_vanish_when_the_sink_fails() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = create_format_table(&catalog, FormatVersion::V2, None, None).await;
    let (mut sorter, arrow_schema) = probe_sorter(&table);
    sorter
        .push(sort_probe_batch(&arrow_schema, &scattered_ids(2000)))
        .await
        .expect("push the probe batch");
    assert!(
        !spill_files_under(&table).is_empty(),
        "the probe budget must force a spill"
    );
    let mut sink = RefuseSink;
    let error = sorter
        .finish(&mut sink)
        .await
        .expect_err("the refusing sink must fail the merge");
    assert_eq!(
        error.message(),
        "the probe sink refuses the merge",
        "the drain error must surface, not the cleanup"
    );
    assert_eq!(
        spill_files_under(&table),
        Vec::<String>::new(),
        "a spill file survived the failed merge"
    );
}

#[tokio::test]
async fn sort_rewrite_on_an_orc_table_writes_orc_and_cleans_its_spills() {
    let (catalog, _guard) = local_fs_catalog().await;
    let schema = sort_probe_schema();
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation {
        name: "t".to_string(),
        location: None,
        schema: schema.clone(),
        partition_spec: None,
        sort_order: None,
        properties: HashMap::from([("write.format.default".to_string(), "orc".to_string())]),
        format_version: FormatVersion::V2,
    };
    let mut table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create the orc sort table");
    let arrow_schema = Arc::new(schema_to_arrow_schema(&schema).expect("arrow schema"));
    for file in 0..4i64 {
        let ids: Vec<i64> = scattered_ids(6000)
            .into_iter()
            .map(|id| id + file * 6000)
            .collect();
        let batch = sort_probe_batch(&arrow_schema, &ids);
        let data_file = write_batch_in_format(
            &table,
            &format!("sort-in-{file}.orc"),
            &batch,
            Arc::new(schema.clone()),
            DataFileFormat::Orc,
            Struct::empty(),
        )
        .await;
        table = append_files(&catalog, &table, vec![data_file]).await;
    }

    let group = planned_group(&table).await;
    let written = RewriteDataFiles::new(table.clone())
        .strategy(RewriteStrategy::Sort(id_ascending()))
        .sort_memory_budget_bytes(64 * 1024)
        .rewrite_all(true)
        .write_group_for_test(&table, &group)
        .await
        .expect("run the sorted write");
    assert!(
        written.sort_stats.spilled_runs > 1,
        "the budget must force several runs, got {}",
        written.sort_stats.spilled_runs
    );
    assert!(
        !written.files.is_empty(),
        "the sorted write must leave output files"
    );
    let mut rows = 0u64;
    for file in &written.files {
        assert_eq!(
            file.file_format(),
            DataFileFormat::Orc,
            "the sorted arm must honor the table format, got {}",
            file.file_path()
        );
        assert!(
            file.file_path().ends_with(".orc"),
            "an orc output must carry the orc extension, got {}",
            file.file_path()
        );
        rows += file.record_count();
    }
    assert_eq!(rows, 24_000, "the sorted write must conserve every row");
    assert_eq!(
        spill_files_under(&table),
        Vec::<String>::new(),
        "a spill file survived the sorted write"
    );
}

#[tokio::test]
async fn legacy_sorted_run_arm_writes_orc_and_stamps_the_table_order() {
    let (catalog, _guard) = local_fs_catalog().await;
    let order = SortOrder {
        order_id: 1,
        fields: vec![SortField {
            source_id: 2,
            transform: Transform::Identity,
            direction: SortDirection::Ascending,
            null_order: NullOrder::First,
        }],
    };
    let table = create_format_table(&catalog, FormatVersion::V2, Some("orc"), Some(order)).await;
    let row_sets: Vec<Vec<(i64, i64, i64)>> = (0..6i64)
        .map(|file| {
            scattered_ids(4)
                .into_iter()
                .map(|id| (0, id + file * 100, id))
                .collect::<Vec<(i64, i64, i64)>>()
        })
        .collect();
    let files =
        write_numbered_format_files(&table, "unsorted", DataFileFormat::Orc, row_sets).await;
    let table = append_files(&catalog, &table, files).await;
    let rows_before = scan_rows(&table).await;
    assert_eq!(rows_before.len(), 24, "fixture: twenty-four rows");

    let (result, table) = compact_and_reload(&catalog, &table, 1_000_000).await;
    assert_eq!(result.rewritten_data_files_count, 6);
    assert!(result.added_data_files_count >= 1);
    let files = live_data_files(&table).await;
    assert!(!files.is_empty(), "compaction must leave output files");
    assert_all_format(
        &files,
        DataFileFormat::Orc,
        "the legacy sorted arm must honor the table format",
    );
    for file in &files {
        assert_eq!(
            file.sort_order_id(),
            Some(1),
            "the legacy sorted arm must stamp the table order id, got {}",
            file.file_path()
        );
    }
    assert_eq!(
        scan_rows(&table).await,
        rows_before,
        "compaction must conserve every row"
    );
}

#[tokio::test]
async fn orc_compaction_rolls_every_output_file_through_the_format() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = create_format_table(&catalog, FormatVersion::V2, Some("orc"), None).await;
    let row_sets: Vec<Vec<(i64, i64, i64)>> = (0..6i64)
        .map(|file| {
            (0..500i64)
                .map(|row| (0, file * 500 + row, row))
                .collect::<Vec<(i64, i64, i64)>>()
        })
        .collect();
    let files = write_numbered_format_files(&table, "roll", DataFileFormat::Orc, row_sets).await;
    let table = append_files(&catalog, &table, files).await;
    let rows_before = scan_rows(&table).await;
    assert_eq!(rows_before.len(), 3000, "fixture: three thousand rows");

    let (_, table) = compact_and_reload(&catalog, &table, 8 * 1024).await;
    let files = live_data_files(&table).await;
    assert!(
        files.len() > 1,
        "the tiny target must roll more than one output file, got {}",
        files.len()
    );
    assert_all_format(
        &files,
        DataFileFormat::Orc,
        "every rolled file must stay orc",
    );
    assert_eq!(
        scan_rows(&table).await,
        rows_before,
        "compaction must conserve every row"
    );
}

#[tokio::test]
async fn orc_table_with_parquet_inputs_compacts_to_orc() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = create_format_table(&catalog, FormatVersion::V2, Some("orc"), None).await;
    let mut files = Vec::new();
    for index in 0..6i64 {
        files.push(
            write_data_file(&table, &format!("mixed-{index}.parquet"), 0, &[(
                0,
                300 + index,
                3000 + index,
            )])
            .await,
        );
    }
    let table = append_files(&catalog, &table, files).await;
    let rows_before = scan_rows(&table).await;
    assert_eq!(rows_before.len(), 6, "fixture: six rows before compaction");

    let (result, table) = compact_and_reload(&catalog, &table, 1_000_000).await;
    assert_eq!(result.rewritten_data_files_count, 6);
    assert!(result.added_data_files_count >= 1);

    let files = live_data_files(&table).await;
    assert!(
        !files.is_empty() && files.len() < 6,
        "compaction must leave fewer files, got {}",
        files.len()
    );
    assert_all_format(
        &files,
        DataFileFormat::Orc,
        "parquet inputs under an orc default must compact to orc",
    );
    assert_eq!(
        scan_rows(&table).await,
        rows_before,
        "compaction must conserve every row"
    );
}

#[tokio::test]
async fn garbage_write_format_default_is_refused_with_the_typed_message() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = create_format_table(&catalog, FormatVersion::V2, Some("csv"), None).await;
    let file = write_data_file(&table, "small-0.parquet", 0, &[(0, 100, 1000)]).await;
    let table = append_files(&catalog, &table, vec![file]).await;
    let group = planned_group(&table).await;
    assert_eq!(group.len(), 1, "fixture: one planned task");
    let outcome = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(1_000_000)
        .write_group_for_test(&table, &group)
        .await;
    let error = match outcome {
        Ok(_) => panic!("a garbage data format must refuse the write"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "Unsupported data file format: csv",
        "the refusal must carry the typed bare message"
    );
}

#[tokio::test]
async fn puffin_write_format_default_is_refused_as_a_sidecar() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = create_format_table(&catalog, FormatVersion::V2, Some("puffin"), None).await;
    let file = write_data_file(&table, "small-0.parquet", 0, &[(0, 100, 1000)]).await;
    let table = append_files(&catalog, &table, vec![file]).await;
    let group = planned_group(&table).await;
    assert_eq!(group.len(), 1, "fixture: one planned task");
    let outcome = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(1_000_000)
        .write_group_for_test(&table, &group)
        .await;
    let error = match outcome {
        Ok(_) => panic!("a puffin data format must refuse the write"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "Cannot build a data-file writer for format puffin: a sidecar is never a data file",
        "the refusal must carry the typed bare message"
    );
}

#[tokio::test]
async fn v3_orc_compaction_keeps_row_lineage() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = create_format_table(&catalog, FormatVersion::V3, Some("orc"), None).await;
    let row_sets: Vec<Vec<(i64, i64, i64)>> = (0..6i64)
        .map(|index| vec![(0, 100 + index, 1000 + index)])
        .collect();
    let files = write_numbered_format_files(&table, "small", DataFileFormat::Orc, row_sets).await;
    let table = append_files(&catalog, &table, files).await;
    let before = scan_lineage(&table).await;
    assert_eq!(before.len(), 6, "fixture: six rows before compaction");

    let (result, table) = compact_and_reload(&catalog, &table, 1_000_000).await;
    assert_eq!(result.rewritten_data_files_count, 6);
    assert!(result.added_data_files_count >= 1);

    let files = live_data_files(&table).await;
    assert!(
        !files.is_empty() && files.len() < 6,
        "compaction must leave fewer files, got {}",
        files.len()
    );
    assert_all_format(
        &files,
        DataFileFormat::Orc,
        "compaction on a v3 orc table must leave orc files",
    );
    assert_eq!(
        scan_lineage(&table).await,
        before,
        "compaction must keep _row_id and last_updated_seq for every live row"
    );
}
