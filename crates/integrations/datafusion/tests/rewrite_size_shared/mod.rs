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

#![allow(dead_code)]

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray};
use datafusion::arrow::compute::{SortColumn, concat_batches, lexsort_to_indices, take};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema, TimeUnit};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use futures::TryStreamExt;
use iceberg::arrow::{ArrowReaderBuilder, FieldMatchMode, RecordBatchPartitionSplitter};
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::scan::{FileScanTask, FileScanTaskStream};
use iceberg::spec::{
    DataContentType, FormatVersion, ManifestContentType, NestedField, PrimitiveType, Schema,
    Struct, TableProperties, Transform, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::file_writer::{
    FileWriter, FileWriterBuilder, ParquetWriter, ParquetWriterBuilder,
    parquet_compression_from_properties,
};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;
use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::serialized_reader::ReadOptionsBuilder;
use parquet::schema::types::ColumnPath;
use tempfile::TempDir;

pub const BATCHES: usize = 206;
pub const BATCH_ROWS: i64 = 2000;
pub const GROUP_COUNT: usize = 20;
pub const BASE_US: i64 = 1_672_531_200_000_000;
pub const SPAN_US: i64 = 730 * 86_400 * 1_000_000;

pub struct ProbeFixture {
    pub context: SessionContext,
    pub catalog: Arc<MemoryCatalog>,
    pub table_ident: TableIdent,
    _warehouse: TempDir,
    pub scratch: TempDir,
}

pub async fn create_fixture(compression_level: Option<&str>) -> ProbeFixture {
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::optional(1, "ts", Type::Primitive(PrimitiveType::Timestamp)).into(),
            NestedField::optional(2, "grp", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "id", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("build schema");
    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "grp", Transform::Identity)
        .expect("partition field")
        .build();
    create_fixture_inner(schema, partition_spec, "probe", "bed", compression_level).await
}

pub const LOW_CARD_BATCHES: usize = 100;
pub const LOW_CARD_VALUES: i64 = 8;

pub async fn create_low_cardinality_fixture(compression_level: Option<&str>) -> ProbeFixture {
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "grp", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build schema");
    let partition_spec = UnboundPartitionSpec::builder().with_spec_id(0).build();
    create_fixture_inner(
        schema,
        partition_spec,
        "probe_low",
        "bed",
        compression_level,
    )
    .await
}

async fn create_fixture_inner(
    schema: Schema,
    partition_spec: UnboundPartitionSpec,
    namespace: &str,
    table_name: &str,
    compression_level: Option<&str>,
) -> ProbeFixture {
    let warehouse = TempDir::new().expect("create warehouse");
    let warehouse_path = warehouse
        .path()
        .to_str()
        .expect("warehouse path is UTF-8")
        .to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
        )
        .await
        .expect("build memory catalog");
    let namespace = NamespaceIdent::new(namespace.to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let mut properties = HashMap::new();
    if let Some(level) = compression_level {
        properties.insert(
            TableProperties::PROPERTY_PARQUET_COMPRESSION_LEVEL.to_string(),
            level.to_string(),
        );
    }
    let table_ident = TableIdent::new(namespace.clone(), table_name.to_string());
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name(table_name.to_string())
                .location(format!("{warehouse_path}/{table_name}"))
                .schema(schema)
                .partition_spec(partition_spec)
                .properties(properties)
                .format_version(FormatVersion::V2)
                .build(),
        )
        .await
        .expect("create table");
    let catalog = Arc::new(catalog);
    let provider = Arc::new(
        IcebergCatalogProvider::try_new(catalog.clone())
            .await
            .expect("catalog provider"),
    );
    let context = SessionContext::new();
    context.register_catalog("catalog", provider);
    ProbeFixture {
        context,
        catalog,
        table_ident,
        _warehouse: warehouse,
        scratch: TempDir::new().expect("create scratch"),
    }
}

fn seed_batch(batch: usize) -> RecordBatch {
    let total = i128::from(BATCHES as i64) * i128::from(BATCH_ROWS);
    let base = i64::try_from(batch).expect("batch index") * BATCH_ROWS;
    let group = format!("g{:02}", batch % GROUP_COUNT);
    let ids: Vec<i64> = (0..BATCH_ROWS).map(|row| base + row).collect();
    let stamps: Vec<i64> = (0..BATCH_ROWS)
        .map(|row| {
            i64::try_from(
                i128::from(base + row) * i128::from(SPAN_US) / total + i128::from(BASE_US),
            )
            .expect("timestamp fits i64")
        })
        .collect();
    let groups: Vec<&str> = (0..BATCH_ROWS).map(|_| group.as_str()).collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("grp", DataType::Utf8, true),
        Field::new("id", DataType::Int64, true),
    ]));
    RecordBatch::try_new(schema, vec![
        Arc::new(TimestampMicrosecondArray::from(stamps)),
        Arc::new(StringArray::from(groups)),
        Arc::new(Int64Array::from(ids)),
    ])
    .expect("build seed batch")
}

async fn run_sql(context: &SessionContext, sql: &str) {
    context
        .sql(sql)
        .await
        .unwrap_or_else(|error| panic!("plan `{sql}`: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("execute `{sql}`: {error}"));
}

fn table_sql_name(fixture: &ProbeFixture) -> String {
    format!(
        "catalog.{}.{}",
        fixture.table_ident.namespace()[0],
        fixture.table_ident.name()
    )
}

pub async fn build_bed(fixture: &ProbeFixture) {
    build_bed_n(fixture, BATCHES).await
}

pub async fn build_bed_n(fixture: &ProbeFixture, batches: usize) {
    let target = table_sql_name(fixture);
    for batch in 0..batches {
        let seed = seed_batch(batch);
        let source =
            MemTable::try_new(seed.schema(), vec![vec![seed]]).expect("build seed MemTable");
        fixture
            .context
            .register_table(format!("seed_{batch}"), Arc::new(source))
            .expect("register seed table");
        run_sql(
            &fixture.context,
            &format!("INSERT INTO {target} SELECT ts, grp, id FROM seed_{batch}"),
        )
        .await;
    }
}

fn low_cardinality_seed_batch(batch: usize) -> RecordBatch {
    let base = i64::try_from(batch).expect("batch index") * BATCH_ROWS;
    let ids: Vec<i64> = (0..BATCH_ROWS).map(|row| base + row).collect();
    let groups: Vec<String> = ids
        .iter()
        .map(|id| format!("g{:02}", id % LOW_CARD_VALUES))
        .collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("grp", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(schema, vec![
        Arc::new(Int64Array::from(ids)),
        Arc::new(StringArray::from(groups)),
    ])
    .expect("build low-cardinality seed batch")
}

pub async fn build_low_cardinality_bed(fixture: &ProbeFixture) {
    let target = table_sql_name(fixture);
    for batch in 0..LOW_CARD_BATCHES {
        let seed = low_cardinality_seed_batch(batch);
        let source =
            MemTable::try_new(seed.schema(), vec![vec![seed]]).expect("build seed MemTable");
        fixture
            .context
            .register_table(format!("seed_low_{batch}"), Arc::new(source))
            .expect("register seed table");
        run_sql(
            &fixture.context,
            &format!("INSERT INTO {target} SELECT id, grp FROM seed_low_{batch}"),
        )
        .await;
    }
}

pub async fn live_file_paths(catalog: &MemoryCatalog, table_ident: &TableIdent) -> Vec<String> {
    let table = catalog.load_table(table_ident).await.expect("load table");
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("a snapshot is committed");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load manifest list");
    let mut paths = Vec::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("load manifest");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.data_file().content_type() == DataContentType::Data {
                paths.push(entry.data_file().file_path().to_string());
            }
        }
    }
    paths.sort();
    paths
}

pub async fn remove_table_property(
    catalog: &MemoryCatalog,
    table_ident: &TableIdent,
    key: &str,
) -> Table {
    let table = catalog.load_table(table_ident).await.expect("load table");
    let transaction = Transaction::new(&table);
    let transaction = transaction
        .update_table_properties()
        .remove(key.to_string())
        .apply(transaction)
        .expect("stage property removal");
    transaction
        .commit(catalog)
        .await
        .expect("commit property removal")
}

pub async fn scan_tasks(table: &Table) -> Vec<FileScanTask> {
    table
        .scan()
        .build()
        .expect("build scan")
        .plan_files()
        .await
        .expect("plan files")
        .try_collect()
        .await
        .expect("collect scan tasks")
}

fn local_fs_path(file_path: &str) -> &str {
    file_path.strip_prefix("file://").unwrap_or(file_path)
}

struct ChunkReport {
    column: String,
    codec: String,
    level: String,
    encodings: Vec<String>,
    dictionary_page: bool,
    data_page_offset: i64,
    pages: usize,
    rows: i64,
    compressed: i64,
    uncompressed: i64,
    statistics: bool,
}

struct FileReport {
    path: String,
    file_bytes: u64,
    created_by: String,
    row_groups: usize,
    rows: i64,
    compressed: i64,
    uncompressed: i64,
    offset_index: bool,
    key_value_keys: Vec<String>,
    chunks: Vec<ChunkReport>,
}

fn inspect_file(path: &str) -> FileReport {
    let local = local_fs_path(path);
    let file_bytes = std::fs::metadata(local)
        .unwrap_or_else(|error| panic!("stat {local}: {error}"))
        .len();
    let page_index_file = File::open(local).unwrap_or_else(|error| panic!("open {local}: {error}"));
    let options = ReadOptionsBuilder::new().with_page_index().build();
    let reader = match SerializedFileReader::new_with_options(page_index_file, options) {
        Ok(reader) => reader,
        Err(_) => {
            let file = File::open(local).unwrap_or_else(|error| panic!("open {local}: {error}"));
            SerializedFileReader::new(file)
                .unwrap_or_else(|error| panic!("read footer {local}: {error}"))
        }
    };
    let metadata = reader.metadata();
    let file_metadata = metadata.file_metadata();
    let offset_index = metadata.offset_index();
    let mut report = FileReport {
        path: local.to_string(),
        file_bytes,
        created_by: file_metadata.created_by().unwrap_or("<none>").to_string(),
        row_groups: metadata.num_row_groups(),
        rows: 0,
        compressed: 0,
        uncompressed: 0,
        offset_index: offset_index.is_some(),
        key_value_keys: file_metadata
            .key_value_metadata()
            .map(|entries| entries.iter().map(|entry| entry.key.clone()).collect())
            .unwrap_or_default(),
        chunks: Vec::new(),
    };
    for (row_group_index, row_group) in metadata.row_groups().iter().enumerate() {
        report.rows += row_group.num_rows();
        for (column_index, column) in row_group.columns().iter().enumerate() {
            let pages = offset_index
                .map(|index| index[row_group_index][column_index].page_locations().len())
                .unwrap_or(0);
            let level = match column.compression() {
                Compression::ZSTD(value) => format!("{}", value.compression_level()),
                _ => String::new(),
            };
            let mut encodings: Vec<String> = column
                .encodings()
                .map(|encoding| format!("{encoding:?}"))
                .collect();
            encodings.sort();
            report.chunks.push(ChunkReport {
                column: column.column_descr().name().to_string(),
                codec: format!("{:?}", column.compression())
                    .split('(')
                    .next()
                    .unwrap_or("?")
                    .to_string(),
                level,
                encodings,
                dictionary_page: column.dictionary_page_offset().is_some(),
                data_page_offset: column.data_page_offset(),
                pages,
                rows: column.num_values(),
                compressed: column.compressed_size(),
                uncompressed: column.uncompressed_size(),
                statistics: column.statistics().is_some(),
            });
            report.compressed += column.compressed_size();
            report.uncompressed += column.uncompressed_size();
        }
    }
    report
}

fn print_file_report(report: &FileReport) {
    println!(
        "file {} bytes={} rows={} row_groups={} compressed={} uncompressed={} offset_index={} created_by={} kv={:?}",
        report.path,
        report.file_bytes,
        report.rows,
        report.row_groups,
        report.compressed,
        report.uncompressed,
        report.offset_index,
        report.created_by,
        report.key_value_keys,
    );
    for chunk in &report.chunks {
        println!(
            "  chunk col={} codec={} level={} encodings={:?} dict_page={} data_page_offset={} pages={} rows={} compressed={} uncompressed={} stats={}",
            chunk.column,
            chunk.codec,
            chunk.level,
            chunk.encodings,
            chunk.dictionary_page,
            chunk.data_page_offset,
            chunk.pages,
            chunk.rows,
            chunk.compressed,
            chunk.uncompressed,
            chunk.statistics,
        );
    }
}

pub struct SetTotals {
    pub files: usize,
    pub file_bytes: u64,
    pub rows: i64,
    pub compressed: i64,
    pub uncompressed: i64,
    pub per_column: Vec<(String, i64, i64)>,
}

pub fn measure_set(label: &str, paths: &[String], detail_files: usize) -> SetTotals {
    let mut totals = SetTotals {
        files: paths.len(),
        file_bytes: 0,
        rows: 0,
        compressed: 0,
        uncompressed: 0,
        per_column: Vec::new(),
    };
    let mut columns: Vec<(String, i64, i64)> = Vec::new();
    for (index, path) in paths.iter().enumerate() {
        let report = inspect_file(path);
        totals.file_bytes += report.file_bytes;
        totals.rows += report.rows;
        totals.compressed += report.compressed;
        totals.uncompressed += report.uncompressed;
        for chunk in &report.chunks {
            if let Some(entry) = columns
                .iter_mut()
                .find(|(name, _, _)| name == &chunk.column)
            {
                entry.1 += chunk.compressed;
                entry.2 += chunk.uncompressed;
            } else {
                columns.push((chunk.column.clone(), chunk.compressed, chunk.uncompressed));
            }
        }
        if index < detail_files {
            print_file_report(&report);
        }
    }
    totals.per_column = columns;
    println!(
        "== {label}: files={} file_bytes={} rows={} sigma_compressed={} sigma_uncompressed={} ratio={:.6}",
        totals.files,
        totals.file_bytes,
        totals.rows,
        totals.compressed,
        totals.uncompressed,
        totals.compressed as f64 / totals.uncompressed as f64,
    );
    for (name, compressed, uncompressed) in &totals.per_column {
        println!("  column {name}: compressed={compressed} uncompressed={uncompressed}");
    }
    totals
}

pub fn column_dictionary_evidence(paths: &[String], column: &str) -> (usize, usize, usize) {
    let mut chunks = 0;
    let mut with_dict_page = 0;
    let mut dict_only_data = 0;
    for path in paths {
        let local = local_fs_path(path);
        let file = File::open(local).unwrap_or_else(|error| panic!("open {local}: {error}"));
        let reader = SerializedFileReader::new(file)
            .unwrap_or_else(|error| panic!("read footer {local}: {error}"));
        for row_group in reader.metadata().row_groups() {
            for column_chunk in row_group.columns() {
                if column_chunk.column_descr().name() != column {
                    continue;
                }
                chunks += 1;
                if column_chunk.dictionary_page_offset().is_some() {
                    with_dict_page += 1;
                }
                if column_chunk.page_encoding_stats_mask().is_some_and(|mask| {
                    mask.is_only(Encoding::PLAIN_DICTIONARY)
                        || mask.is_only(Encoding::RLE_DICTIONARY)
                }) {
                    dict_only_data += 1;
                }
            }
        }
    }
    (chunks, with_dict_page, dict_only_data)
}

pub fn print_first_rows(path: &str, label: &str) {
    let local = local_fs_path(path);
    let file = File::open(local).unwrap_or_else(|error| panic!("open {local}: {error}"));
    let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap_or_else(|error| panic!("reader builder {local}: {error}"));
    let mut reader = builder
        .build()
        .unwrap_or_else(|error| panic!("build reader {local}: {error}"));
    let Some(batch) = reader.next() else {
        println!("{label}: {local} is empty");
        return;
    };
    let batch = batch.expect("read first batch");
    let id_index = batch.schema().index_of("id").expect("id column");
    let grp_index = batch.schema().index_of("grp").expect("grp column");
    let ids = batch
        .column(id_index)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("id is Int64");
    let grps = batch
        .column(grp_index)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("grp is Utf8");
    let count = ids.len().min(50);
    let id_values: Vec<i64> = (0..count).map(|row| ids.value(row)).collect();
    let grp_values: Vec<&str> = (0..count).map(|row| grps.value(row)).collect();
    println!("{label}: first file {local}");
    println!("{label}: first {count} ids: {id_values:?}");
    println!("{label}: first {count} grps: {grp_values:?}");
}

pub fn print_writer_properties_table(table: &Table) {
    let compression = parquet_compression_from_properties(table.metadata().properties())
        .expect("parse compression properties");
    let path_props = WriterProperties::builder()
        .set_compression(compression)
        .build();
    let column = ColumnPath::from("id");
    println!(
        "== WriterProperties both paths build (insert: write.rs, rewrite: rewrite_data_files_write.rs) =="
    );
    println!("{:<44} value on BOTH paths", "property");
    let rows: Vec<(String, String)> = vec![
        (
            "compression(id)".to_string(),
            format!("{:?}", path_props.compression(&column)),
        ),
        (
            "dictionary_enabled(id)".to_string(),
            format!("{}", path_props.dictionary_enabled(&column)),
        ),
        (
            "encoding(id)".to_string(),
            format!("{:?}", path_props.encoding(&column)),
        ),
        (
            "dictionary_page_size_limit".to_string(),
            format!("{}", path_props.dictionary_page_size_limit()),
        ),
        (
            "data_page_size_limit".to_string(),
            format!("{}", path_props.data_page_size_limit()),
        ),
        (
            "data_page_row_count_limit".to_string(),
            format!("{}", path_props.data_page_row_count_limit()),
        ),
        (
            "write_batch_size".to_string(),
            format!("{}", path_props.write_batch_size()),
        ),
        (
            "max_row_group_row_count".to_string(),
            format!("{:?}", path_props.max_row_group_row_count()),
        ),
        (
            "max_row_group_bytes".to_string(),
            format!("{:?}", path_props.max_row_group_bytes()),
        ),
        (
            "statistics_enabled(id)".to_string(),
            format!("{:?}", path_props.statistics_enabled(&column)),
        ),
        (
            "write_page_header_statistics(id)".to_string(),
            format!("{}", path_props.write_page_header_statistics(&column)),
        ),
        (
            "statistics_truncate_length".to_string(),
            format!("{:?}", path_props.statistics_truncate_length()),
        ),
        (
            "column_index_truncate_length".to_string(),
            format!("{:?}", path_props.column_index_truncate_length()),
        ),
        (
            "writer_version".to_string(),
            format!("{:?}", path_props.writer_version()),
        ),
        (
            "created_by".to_string(),
            path_props.created_by().to_string(),
        ),
        (
            "offset_index_disabled".to_string(),
            format!("{}", path_props.offset_index_disabled()),
        ),
        (
            "sorting_columns".to_string(),
            format!("{:?}", path_props.sorting_columns()),
        ),
        (
            "bloom_filter_position".to_string(),
            format!("{:?}", path_props.bloom_filter_position()),
        ),
        (
            "bloom_filter_properties(id)".to_string(),
            format!("{:?}", path_props.bloom_filter_properties(&column)),
        ),
        (
            "coerce_types".to_string(),
            format!("{}", path_props.coerce_types()),
        ),
        (
            "data_page_v2_compression_ratio_threshold".to_string(),
            format!("{}", path_props.data_page_v2_compression_ratio_threshold()),
        ),
        (
            "key_value_metadata".to_string(),
            format!("{:?}", path_props.key_value_metadata()),
        ),
    ];
    for (name, value) in rows {
        println!("{name:<44} {value}");
    }
    println!(
        "{:<28} {:<50} rewrite path (rewrite_data_files_write.rs)",
        "construction detail", "insert path (write.rs)"
    );
    println!(
        "{:<28} {:<50} new(FieldMatchMode::Id)",
        "ParquetWriterBuilder", "new_with_match_mode(FieldMatchMode::Name)"
    );
    println!(
        "{:<28} {:<50} ArrowReader stream -> RecordBatchPartitionSplitter -> BoundedPartitionRouter",
        "routing", "TaskWriter fanout, rows clustered by partition"
    );
    println!(
        "{:<28} {:<50} parquet reader batches (8192-row default)",
        "input batches", "DataFusion child exec batches (2000 rows)"
    );
}

fn sort_batch_by_id(batch: &RecordBatch) -> RecordBatch {
    let index = batch.schema().index_of("id").expect("id column for sort");
    let indices = lexsort_to_indices(
        &[SortColumn {
            values: batch.column(index).clone(),
            options: None,
        }],
        None,
    )
    .expect("lexsort_to_indices");
    let columns = batch
        .columns()
        .iter()
        .map(|column| take(column.as_ref(), &indices, None).expect("take"))
        .collect();
    RecordBatch::try_new(batch.schema(), columns).expect("sorted batch")
}

pub async fn manual_rewrite(
    table: &Table,
    tasks: Vec<FileScanTask>,
    props: &WriterProperties,
    match_mode: FieldMatchMode,
    sort_rows_by_id: bool,
    scratch: &Path,
    tag: &str,
) -> Vec<String> {
    let schema = table.metadata().current_schema().clone();
    let spec = table.metadata().default_partition_spec().clone();
    let file_io = table.file_io().clone();
    let splitter =
        RecordBatchPartitionSplitter::try_new_with_computed_values(schema.clone(), spec.clone())
            .expect("build partition splitter");
    let task_stream =
        Box::pin(futures::stream::iter(tasks.into_iter().map(Ok))) as FileScanTaskStream;
    let mut batch_stream = ArrowReaderBuilder::new(file_io.clone())
        .build()
        .read(task_stream)
        .expect("read task stream");
    let mut by_partition: HashMap<Struct, Vec<RecordBatch>> = HashMap::new();
    let mut order: Vec<Struct> = Vec::new();
    while let Some(batch) = batch_stream.try_next().await.expect("read batch") {
        for (key, part) in splitter.split(&batch).expect("split batch") {
            let data = key.data().clone();
            if !by_partition.contains_key(&data) {
                order.push(data.clone());
            }
            by_partition.entry(data).or_default().push(part);
        }
    }
    let mut paths = Vec::new();
    for (index, data) in order.iter().enumerate() {
        let mut parts = by_partition.remove(data).expect("partition batches");
        if sort_rows_by_id {
            let concatenated =
                concat_batches(&parts[0].schema(), &parts).expect("concat partition batches");
            parts = vec![sort_batch_by_id(&concatenated)];
        }
        let path = format!("file://{}/{tag}-{index}.parquet", scratch.display());
        let output = file_io.new_output(&path).expect("create output");
        let mut writer: ParquetWriter =
            ParquetWriterBuilder::new_with_match_mode(props.clone(), schema.clone(), match_mode)
                .build(output)
                .await
                .expect("build parquet writer");
        for part in &parts {
            writer.write(part).await.expect("write batch");
        }
        writer.close().await.expect("close writer");
        paths.push(path);
    }
    paths
}

fn zstd_props(level: i32) -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::ZSTD(
            ZstdLevel::try_new(level).expect("zstd level"),
        ))
        .build()
}

pub async fn flip_table(
    table: &Table,
    input_tasks: &[FileScanTask],
    input_compressed: i64,
    scratch: &Path,
) {
    let baseline = zstd_props(1);
    let cases: Vec<(&str, WriterProperties, FieldMatchMode, bool)> = vec![
        (
            "baseline (zstd-1, dict on, scan order)",
            baseline.clone(),
            FieldMatchMode::Id,
            false,
        ),
        ("zstd level 3", zstd_props(3), FieldMatchMode::Id, false),
        ("zstd level 9", zstd_props(9), FieldMatchMode::Id, false),
        (
            "dictionary OFF",
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(
                    ZstdLevel::try_new(1).expect("zstd level"),
                ))
                .set_dictionary_enabled(false)
                .build(),
            FieldMatchMode::Id,
            false,
        ),
        (
            "zstd level 3 + dictionary OFF",
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(
                    ZstdLevel::try_new(3).expect("zstd level"),
                ))
                .set_dictionary_enabled(false)
                .build(),
            FieldMatchMode::Id,
            false,
        ),
        (
            "data_page_size_limit 256KiB",
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(
                    ZstdLevel::try_new(1).expect("zstd level"),
                ))
                .set_data_page_size_limit(256 * 1024)
                .build(),
            FieldMatchMode::Id,
            false,
        ),
        (
            "data_page_size_limit 8MiB",
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(
                    ZstdLevel::try_new(1).expect("zstd level"),
                ))
                .set_data_page_size_limit(8 * 1024 * 1024)
                .build(),
            FieldMatchMode::Id,
            false,
        ),
        (
            "data_page_row_count_limit 1Mi",
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(
                    ZstdLevel::try_new(1).expect("zstd level"),
                ))
                .set_data_page_row_count_limit(1024 * 1024)
                .build(),
            FieldMatchMode::Id,
            false,
        ),
        (
            "max_row_group_row_count 200k",
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(
                    ZstdLevel::try_new(1).expect("zstd level"),
                ))
                .set_max_row_group_row_count(Some(200_000))
                .build(),
            FieldMatchMode::Id,
            false,
        ),
        (
            "write_batch_size 8192",
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(
                    ZstdLevel::try_new(1).expect("zstd level"),
                ))
                .set_write_batch_size(8192)
                .build(),
            FieldMatchMode::Id,
            false,
        ),
        (
            "statistics NONE",
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(
                    ZstdLevel::try_new(1).expect("zstd level"),
                ))
                .set_statistics_enabled(EnabledStatistics::None)
                .build(),
            FieldMatchMode::Id,
            false,
        ),
        (
            "encoding PLAIN",
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(
                    ZstdLevel::try_new(1).expect("zstd level"),
                ))
                .set_encoding(Encoding::PLAIN)
                .build(),
            FieldMatchMode::Id,
            false,
        ),
        (
            "offset_index disabled",
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(
                    ZstdLevel::try_new(1).expect("zstd level"),
                ))
                .set_offset_index_disabled(true)
                .build(),
            FieldMatchMode::Id,
            false,
        ),
        (
            "match_mode Name",
            baseline.clone(),
            FieldMatchMode::Name,
            false,
        ),
        ("sorted by id", baseline.clone(), FieldMatchMode::Id, true),
        (
            "sorted by id + dictionary OFF",
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(
                    ZstdLevel::try_new(1).expect("zstd level"),
                ))
                .set_dictionary_enabled(false)
                .build(),
            FieldMatchMode::Id,
            true,
        ),
    ];
    println!("== one-at-a-time flips (input compressed = {input_compressed}) ==");
    println!(
        "{:<40} {:>14} {:>14} {:>8}",
        "case", "compressed", "file_bytes", "out/in"
    );
    for (name, props, match_mode, sort) in &cases {
        let paths = manual_rewrite(
            table,
            input_tasks.to_vec(),
            props,
            *match_mode,
            *sort,
            scratch,
            "flip",
        )
        .await;
        let totals = measure_set(name, &paths, 0);
        let case_ratio = totals.compressed as f64 / input_compressed as f64;
        println!(
            "{name:<40} {:>14} {:>14} {:>8.4}",
            totals.compressed, totals.file_bytes, case_ratio
        );
    }
}
