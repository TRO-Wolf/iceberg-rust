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

//! Java interop tests for data-level scan execution with merge-on-read deletes. Direction 1 loads a
//! table the Java oracle wrote and compares against the rows Java emitted.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use iceberg::arrow::DeleteFilter;
use iceberg::io::{FileIO, LocalFsStorageFactory};
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, FormatVersion, Literal, NestedField, PartitionKey, PartitionSpec,
    PrimitiveType, Schema, SchemaRef, SortOrder, Struct, TableMetadata, Transform, Type,
    UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::base_writer::equality_delete_writer::{
    EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
};
use iceberg::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
};
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Deserialize;

// The Java oracle row model, deserialized from a JSON array of {id, data}.

/// One live row of Java's merge-on-read read: the `id` and the nullable `data` string.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ScanRow {
    id: i64,
    data: Option<String>,
}

/// Sort rows by id, so the comparison is order-independent.
fn sorted_by_id(mut rows: Vec<ScanRow>) -> Vec<ScanRow> {
    rows.sort_by(|a, b| a.id.cmp(&b.id).then_with(|| cmp_opt(&a.data, &b.data)));
    rows
}

fn cmp_opt(a: &Option<String>, b: &Option<String>) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(x), Some(y)) => x.cmp(y),
    }
}

// Fixture loading and Table construction.

/// The dir the Java oracle wrote the table and JSON rows into. `None` when the env var is unset.
fn scan_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_SCAN_DIR").map(PathBuf::from)
}

/// The dir the direction-2 GEN path writes a Rust-authored table into, for Java to read.
fn scan_gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_SCAN_GEN_DIR").map(PathBuf::from)
}

/// The dir the Java oracle wrote the equality-delete table and JSON rows into.
fn eq_scan_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_EQ_SCAN_DIR").map(PathBuf::from)
}

/// The dir the direction-2 GEN path writes a Rust-authored equality-delete table into.
fn eq_scan_gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_EQ_SCAN_GEN_DIR").map(PathBuf::from)
}

/// The dir the Java oracle wrote the partitioned table and JSON rows into.
fn part_scan_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_PART_SCAN_DIR").map(PathBuf::from)
}

/// The dir the direction-2 GEN path writes a Rust-authored partitioned table into.
fn part_scan_gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_PART_SCAN_GEN_DIR").map(PathBuf::from)
}

/// Load the Java ground-truth partitioned rows.
fn read_java_part_rows(dir: &std::path::Path) -> Vec<ScanRow> {
    let path = dir.join("java_part_scan_rows.json");
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str::<Vec<ScanRow>>(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// The dir the Java oracle wrote the multi-file-per-partition table and JSON rows into.
fn multifile_scan_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_MULTIFILE_SCAN_DIR").map(PathBuf::from)
}

/// The dir the direction-2 GEN path writes a Rust-authored multi-file table into.
fn multifile_scan_gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_MULTIFILE_SCAN_GEN_DIR").map(PathBuf::from)
}

/// The trailing path component of a file location. The plan assertions compare basenames because
/// the fixture's absolute paths depend on the temp dir the runner script picked.
fn file_name(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_string()
}

/// The dir the Java oracle wrote the file-scoped position-delete table and JSON rows into.
fn file_scoped_deletes_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_FILE_SCOPED_DELETES_DIR").map(PathBuf::from)
}

/// The dir holding the R117 cross-task variant of the file-scoped fixture, whose control delete is
/// stamped `category=b` instead of the empty `category=c`.
fn file_scoped_deletes_crosstask_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_FILE_SCOPED_DELETES_CROSSTASK_DIR").map(PathBuf::from)
}

/// Load the Java ground-truth file-scoped-delete rows.
fn read_java_file_scoped_rows(dir: &std::path::Path) -> Vec<ScanRow> {
    let path = dir.join("java_file_scoped_delete_rows.json");
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str::<Vec<ScanRow>>(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// Load the Java ground-truth multi-file rows.
fn read_java_multifile_rows(dir: &std::path::Path) -> Vec<ScanRow> {
    let path = dir.join("java_multifile_scan_rows.json");
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str::<Vec<ScanRow>>(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// Load the Java ground-truth equality-delete rows.
fn read_java_eq_rows(dir: &std::path::Path) -> Vec<ScanRow> {
    let path = dir.join("java_eq_scan_rows.json");
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str::<Vec<ScanRow>>(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// Load the Java ground-truth rows.
fn read_java_rows(dir: &std::path::Path) -> Vec<ScanRow> {
    let path = dir.join("java_scan_rows.json");
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str::<Vec<ScanRow>>(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// Build a `Table` over the Java-written `final.metadata.json`. A local-filesystem `FileIO` lets the
/// absolute manifest and parquet paths the commits wrote resolve directly.
fn load_table(dir: &std::path::Path) -> Table {
    let metadata_path = dir.join("table/metadata/final.metadata.json");
    let json = fs::read_to_string(&metadata_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", metadata_path.display()));
    let metadata: TableMetadata = serde_json::from_str(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", metadata_path.display()));

    Table::builder()
        .metadata(metadata)
        .metadata_location(metadata_path.to_string_lossy().to_string())
        .identifier(TableIdent::from_strs(["interop", "scan_exec"]).expect("valid identifier"))
        .file_io(FileIO::new_with_fs())
        .build()
        .expect("build table from Java-written final.metadata.json")
}

// Arrow column extraction: a scan batch into comparable [`ScanRow`]s, by column name.

/// Extract the `id` and `data` columns from one scan batch. The `data` column reads as either
/// i32- or i64-offset Utf8, because `to_arrow` may emit either width.
fn extract_rows(batch: &RecordBatch) -> Vec<ScanRow> {
    let id = batch
        .column_by_name("id")
        .expect("id column present")
        .as_primitive::<Int64Type>();
    let data = batch.column_by_name("data").expect("data column present");

    (0..batch.num_rows())
        .map(|i| ScanRow {
            id: id.value(i),
            data: string_value(data, i),
        })
        .collect()
}

/// Read row `i` of a nullable string column, as either Utf8 or LargeUtf8.
fn string_value(array: &arrow_array::ArrayRef, i: usize) -> Option<String> {
    use arrow_schema::DataType;
    if array.is_null(i) {
        return None;
    }
    match array.data_type() {
        DataType::Utf8 => Some(array.as_string::<i32>().value(i).to_string()),
        DataType::LargeUtf8 => Some(array.as_string::<i64>().value(i).to_string()),
        other => panic!("unexpected data column arrow type: {other:?}"),
    }
}

// The single env-gated interop test.

#[tokio::test]
async fn test_scan_exec_merge_on_read_matches_java_read() {
    let Some(dir) = scan_dir() else {
        println!(
            "skipping interop_scan_exec — set ICEBERG_INTEROP_SCAN_DIR \
             (run dev/java-interop/run-interop-scan-exec.sh)"
        );
        return;
    };

    let table = load_table(&dir);

    // The scan applies the position deletes against a Java-written table.
    let batch_stream = table
        .scan()
        .build()
        .expect("build table scan")
        .to_arrow()
        .await
        .expect("scan to_arrow");
    let batches: Vec<RecordBatch> = batch_stream
        .try_collect()
        .await
        .expect("collect scan batches");

    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_rows(batch));
    }
    let rust_rows = sorted_by_id(rust_rows);
    let java_rows = sorted_by_id(read_java_rows(&dir));

    // -- The merge-on-read proof. ---------------------------------------------------------------

    // 3 live rows survive: 5 written, 2 deleted.
    assert_eq!(
        rust_rows.len(),
        3,
        "exactly 3 rows survive merge-on-read (5 written, positions 1 and 3 deleted)"
    );

    // The deleted rows, id 20 at position 1 and id 40 at position 3, must be absent.
    assert!(
        !rust_rows.iter().any(|r| r.id == 20),
        "id 20 (deleted at position 1) must be ABSENT after merge-on-read"
    );
    assert!(
        !rust_rows.iter().any(|r| r.id == 40),
        "id 40 (deleted at position 3) must be ABSENT after merge-on-read"
    );

    assert_eq!(
        rust_rows, java_rows,
        "Rust scan→Arrow (merge-on-read) rows must equal Java's IcebergGenerics read field-for-field"
    );

    // Pin the exact live set, so it cannot drift unnoticed.
    let live_ids: Vec<i64> = rust_rows.iter().map(|r| r.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 50],
        "the live id set after merge-on-read is exactly {{10, 30, 50}}"
    );
    let live_data: Vec<Option<&str>> = rust_rows.iter().map(|r| r.data.as_deref()).collect();
    assert_eq!(
        live_data,
        vec![Some("a"), Some("c"), Some("e")],
        "the live data column matches the committed values for ids 10/30/50"
    );

    println!(
        "interop_scan_exec OK — Rust scan→Arrow merge-on-read = Java read: 3 live rows {{10,30,50}}, \
         deleted 20/40 absent"
    );
}

// EQUALITY-DELETE, DIRECTION 1 — Java writes the equality delete, Rust reads it.
//
// Java wrote an unpartitioned V2 table: a 5-row data file at sequence 1, then an equality delete
// on field id 1 for ids 20 and 40 at sequence 2. The delete applies because 1 < 2, so the live
// set is {10,30,50}. Java emitted its own read into `java_eq_scan_rows.json`.

#[tokio::test]
async fn test_scan_exec_equality_delete_matches_java_read() {
    let Some(dir) = eq_scan_dir() else {
        println!(
            "skipping interop_scan_exec equality-delete — set ICEBERG_INTEROP_EQ_SCAN_DIR \
             (run dev/java-interop/run-interop-eq-delete.sh)"
        );
        return;
    };

    let table = load_table(&dir);

    // The seq-2 equality delete drops every seq-1 row whose `id` is 20 or 40.
    let batch_stream = table
        .scan()
        .build()
        .expect("build table scan")
        .to_arrow()
        .await
        .expect("scan to_arrow");
    let batches: Vec<RecordBatch> = batch_stream
        .try_collect()
        .await
        .expect("collect scan batches");

    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_rows(batch));
    }
    let rust_rows = sorted_by_id(rust_rows);
    let java_rows = sorted_by_id(read_java_eq_rows(&dir));

    // -- The equality-delete merge-on-read proof. -----------------------------------------------

    // 3 live rows survive: 5 written, 2 deleted by value.
    assert_eq!(
        rust_rows.len(),
        3,
        "exactly 3 rows survive merge-on-read (5 written, ids 20 and 40 deleted by VALUE)"
    );

    // The equality delete on field id 1 dropped id 20 and id 40.
    assert!(
        !rust_rows.iter().any(|r| r.id == 20),
        "id 20 (equality-deleted by value) must be ABSENT after merge-on-read"
    );
    assert!(
        !rust_rows.iter().any(|r| r.id == 40),
        "id 40 (equality-deleted by value) must be ABSENT after merge-on-read"
    );

    assert_eq!(
        rust_rows, java_rows,
        "Rust scan→Arrow (equality merge-on-read) rows must equal Java's IcebergGenerics read field-for-field"
    );

    let live_ids: Vec<i64> = rust_rows.iter().map(|r| r.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 50],
        "the live id set after equality merge-on-read is exactly {{10, 30, 50}}"
    );
    let live_data: Vec<Option<&str>> = rust_rows.iter().map(|r| r.data.as_deref()).collect();
    assert_eq!(
        live_data,
        vec![Some("a"), Some("c"), Some("e")],
        "the live data column matches the committed values for ids 10/30/50"
    );

    println!(
        "interop_scan_exec equality-delete OK — Rust scan→Arrow equality merge-on-read = Java read: \
         3 live rows {{10,30,50}}, deleted 20/40 absent"
    );
}

// PARTITIONED merge-on-read, DIRECTION 1 — Java writes the partitioned table and the
// partition-scoped delete, Rust reads it.
//
// Java wrote a V2 table partitioned by identity(category), one data file per partition at
// sequence 1, then a position delete in partition a for position 1 (id=20) at sequence 2. The
// live set is {10,30,40,50}. Java emitted its own read into `java_part_scan_rows.json`.
//
// Rust's delete_file_index keys deletes by partition and spec id, so the cat=a delete must reach
// only the cat=a data file. Applying it to cat=b, or dropping a partition, fails this test.

#[tokio::test]
async fn test_part_scan_exec_partition_scoped_merge_on_read_matches_java_read() {
    let Some(dir) = part_scan_dir() else {
        println!(
            "skipping interop_scan_exec partitioned — set ICEBERG_INTEROP_PART_SCAN_DIR \
             (run dev/java-interop/run-interop-part.sh)"
        );
        return;
    };

    let table = load_table(&dir);

    // The partition-scoped delete drops position 1 of the cat=a data file. cat=b is untouched.
    let batch_stream = table
        .scan()
        .build()
        .expect("build table scan")
        .to_arrow()
        .await
        .expect("scan to_arrow");
    let batches: Vec<RecordBatch> = batch_stream
        .try_collect()
        .await
        .expect("collect scan batches");

    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_rows(batch));
    }
    let rust_rows = sorted_by_id(rust_rows);
    let java_rows = sorted_by_id(read_java_part_rows(&dir));

    // -- The partition-aware merge-on-read proof. -----------------------------------------------

    // 4 live rows survive: 5 written across both partitions, position 1 of cat=a deleted.
    assert_eq!(
        rust_rows.len(),
        4,
        "exactly 4 rows survive partition-aware merge-on-read (5 written, cat=a position 1 deleted)"
    );

    // The deleted row, id 20 at position 1 of the cat=a data file, must be absent.
    assert!(
        !rust_rows.iter().any(|r| r.id == 20),
        "id 20 (partition-scoped delete at cat=a position 1) must be ABSENT after merge-on-read"
    );

    // Both partitions stay otherwise intact: cat=a keeps 10 and 30, cat=b keeps 40 and 50.
    for id in [10_i64, 30, 40, 50] {
        assert!(
            rust_rows.iter().any(|r| r.id == id),
            "id {id} must be present — both partitions intact except cat=a's deleted id=20"
        );
    }

    assert_eq!(
        rust_rows, java_rows,
        "Rust scan→Arrow (partition-aware merge-on-read) rows must equal Java's IcebergGenerics read \
         field-for-field"
    );

    // Pin the exact live set, so it cannot drift unnoticed.
    let live_ids: Vec<i64> = rust_rows.iter().map(|r| r.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 40, 50],
        "the live id set after partition-aware merge-on-read is exactly {{10, 30, 40, 50}}"
    );
    let live_data: Vec<Option<&str>> = rust_rows.iter().map(|r| r.data.as_deref()).collect();
    assert_eq!(
        live_data,
        vec![Some("x"), Some("z"), Some("p"), Some("q")],
        "the live data column matches the committed values for ids 10/30 (cat=a) and 40/50 (cat=b)"
    );

    println!(
        "interop_scan_exec partitioned OK — Rust scan→Arrow partition-aware merge-on-read = Java read: \
         4 live rows {{10,30,40,50}}, cat=a's id=20 deleted, cat=b intact"
    );
}

// DIRECTION 2 — the GEN path: Rust writes a real on-disk table; Java reads it back.
//
// Commits through a `MemoryCatalog` backed by `LocalFsStorageFactory`, so metadata, manifests and
// parquet land on the real local filesystem, and writes `final.metadata.json` at a known path.
// Java loads that metadata, reads with `IcebergGenerics`, and asserts {10,30,50}.

/// The unpartitioned V2 schema Java expects: {1 id long required, 2 data string optional}.
fn gen_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build the {id long, data string} schema")
}

/// Create the unpartitioned V2 table at exactly `<gen_dir>/rust_table`, so the on-disk layout is
/// the deterministic one Java loads.
async fn create_rust_table(catalog: &impl Catalog, table_location: &str) -> Table {
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");

    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location.to_string())
        .schema(gen_schema())
        .partition_spec(UnboundPartitionSpec::builder().build())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();

    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create rust_table")
}

/// Write a real parquet data file of 5 rows into the table's location, through the production
/// `ParquetWriterBuilder`. No hand-rolled parquet.
async fn write_gen_data_file(table: &Table) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));

    let ids = Int64Array::from(vec![10_i64, 20, 30, 40, 50]);
    let data = StringArray::from(vec!["a", "b", "c", "d", "e"]);
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(ids) as ArrayRef,
        Arc::new(data) as ArrayRef,
    ])
    .expect("build the 5-row data batch");

    // Write the parquet under the table location, so Java's FileIO resolves it from the manifest
    // entry.
    let file_path = format!(
        "{}/data/00000-rust-data.parquet",
        table.metadata().location()
    );
    let output = table
        .file_io()
        .new_output(file_path)
        .expect("new parquet output");
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let mut writer = parquet_builder
        .build(output)
        .await
        .expect("build parquet writer");
    writer.write(&batch).await.expect("write data batch");
    let data_file_builders = writer.close().await.expect("close parquet writer");

    // The parquet writer leaves content and partition unstamped. Finish as an unpartitioned data
    // file with the default spec id 0.
    let mut builder = data_file_builders
        .into_iter()
        .next()
        .expect("one data file builder");
    builder
        .content(DataContentType::Data)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .expect("build unpartitioned data file")
}

#[tokio::test]
async fn test_scan_exec_gen_rust_writes_java_readable_table() {
    let Some(gen_dir) = scan_gen_dir() else {
        println!(
            "skipping interop_scan_exec GEN — set ICEBERG_INTEROP_SCAN_GEN_DIR \
             (run dev/java-interop/run-interop-scan-exec-d2.sh)"
        );
        return;
    };

    let warehouse = gen_dir.to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_gen",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let table = create_rust_table(&catalog, &table_location).await;

    let data_file = write_gen_data_file(&table).await;
    let data_file_path = data_file.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![data_file])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");

    // 3. Engine step: scan (_file, _pos) to discover the identity of ids 20 and 40, then row_delta
    //    a real position delete built from the pairs.
    let mut pairs = discover_row_identities(&table, &[20, 40]).await;
    pairs.sort();
    assert_eq!(
        pairs,
        vec![
            (data_file_path.clone(), 1_i64),
            (data_file_path.clone(), 3_i64),
        ],
        "scan(_file,_pos) discovered ids 20/40 at their true positions 1/3 in the committed data file"
    );
    let delete_file = write_pos_delete_from_pairs(&table, &pairs).await;
    assert_eq!(delete_file.content_type(), DataContentType::PositionDeletes);
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // 4. Confirm the table is internally consistent before Java reads it.
    let batches: Vec<RecordBatch> = table
        .scan()
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect batches");
    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_rows(batch));
    }
    let rust_rows = sorted_by_id(rust_rows);
    let live_ids: Vec<i64> = rust_rows.iter().map(|r| r.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 50],
        "Rust's own scan of the written table must already be {{10,30,50}} (20/40 deleted)"
    );

    let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), &final_metadata_path)
        .await
        .expect("write final.metadata.json");

    println!(
        "interop_scan_exec GEN OK — Rust wrote {table_location} (parquet data + position-delete + \
         final.metadata.json); Rust scan = {{10,30,50}}. Java verify-interop-scan-exec reads it next."
    );
}

// ENGINE-BOUNDARY OFFLINE PROOF — plays the downstream engine's role over the public core-crate
// surface only: scan `(_file, _pos)` for row identity, write a position delete from the pairs,
// commit it via `RowDelta`, and confirm merge-on-read omits exactly those rows. This is the
// executable contract a DataFusion-wrapped engine builds DELETE, UPDATE and MERGE on.

/// Decode the reserved `_file` column at row `i`. The scan emits it as a per-file constant, which
/// the transformer materializes run-end-encoded. Both the encoded and plain `Utf8` forms decode.
fn decode_file_path(col: &ArrayRef, i: usize) -> String {
    use arrow_array::RunArray;
    use arrow_array::types::Int32Type;

    if let Some(plain) = col.as_any().downcast_ref::<StringArray>() {
        return plain.value(i).to_string();
    }
    if let Some(run) = col.as_any().downcast_ref::<RunArray<Int32Type>>() {
        let physical = run.get_physical_index(i);
        return run
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("_file REE values are Utf8")
            .value(physical)
            .to_string();
    }
    panic!("unexpected _file column type: {:?}", col.data_type());
}

/// Return the `(data_file_path, position)` of every row whose id is in `target_ids`. This is how a
/// downstream engine discovers row identity before it writes position deletes.
async fn discover_row_identities(table: &Table, target_ids: &[i64]) -> Vec<(String, i64)> {
    use iceberg::metadata_columns::{RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_POS};

    let batches: Vec<RecordBatch> = table
        .scan()
        .select(["id", RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_POS])
        .build()
        .expect("build identity scan")
        .to_arrow()
        .await
        .expect("identity scan to_arrow")
        .try_collect()
        .await
        .expect("collect identity batches");

    let mut pairs = Vec::new();
    for batch in &batches {
        let id_col = batch
            .column_by_name("id")
            .expect("id column")
            .as_primitive::<Int64Type>();
        let file_col = batch
            .column_by_name(RESERVED_COL_NAME_FILE)
            .expect("_file column");
        let pos_col = batch
            .column_by_name(RESERVED_COL_NAME_POS)
            .expect("_pos column")
            .as_primitive::<Int64Type>();
        for i in 0..batch.num_rows() {
            if target_ids.contains(&id_col.value(i)) {
                pairs.push((decode_file_path(file_col, i), pos_col.value(i)));
            }
        }
    }
    pairs
}

/// Write a real parquet position-delete file from discovered `(data_file_path, position)` pairs.
/// The spec requires the pairs to arrive sorted by `(file_path, pos)`.
async fn write_pos_delete_from_pairs(table: &Table, pairs: &[(String, i64)]) -> DataFile {
    let config = PositionDeleteWriterConfig::new().expect("position-delete writer config");
    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "pos-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        iceberg::spec::DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        config.schema().clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(None)
        .await
        .expect("build position-delete writer");

    let paths: Vec<&str> = pairs.iter().map(|(path, _)| path.as_str()).collect();
    let positions: Vec<i64> = pairs.iter().map(|(_, pos)| *pos).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions)) as ArrayRef,
    ])
    .expect("build position-delete batch");
    writer
        .write(batch)
        .await
        .expect("write position-delete batch");
    writer
        .close()
        .await
        .expect("close position-delete writer")
        .into_iter()
        .next()
        .expect("one position-delete file")
}

#[tokio::test]
async fn test_engine_boundary_scan_pos_then_row_delta() {
    use tempfile::TempDir;

    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "engine_boundary",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let table = create_rust_table(&catalog, &table_location).await;

    let data_file = write_gen_data_file(&table).await;
    let data_file_path = data_file.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![data_file])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");

    // 3. Engine step: scan `(_file, _pos)` to discover the identity of ids 20 and 40. This also
    //    asserts `_pos` reports the true physical ordinal, 1 and 3.
    let mut pairs = discover_row_identities(&table, &[20, 40]).await;
    pairs.sort(); // spec: position deletes sorted by (file_path, pos)
    assert_eq!(
        pairs,
        vec![
            (data_file_path.clone(), 1_i64),
            (data_file_path.clone(), 3_i64),
        ],
        "scan(_file,_pos) must report ids 20/40 at their TRUE physical positions 1 and 3"
    );

    let delete_file = write_pos_delete_from_pairs(&table, &pairs).await;
    assert_eq!(delete_file.content_type(), DataContentType::PositionDeletes);
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    let batches: Vec<RecordBatch> = table
        .scan()
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect batches");
    let mut rows = Vec::new();
    for batch in &batches {
        rows.extend(extract_rows(batch));
    }
    let live_ids: Vec<i64> = sorted_by_id(rows).iter().map(|row| row.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 50],
        "after committing the scan-discovered position deletes, ids 20/40 must be gone"
    );
}

// EQUALITY-DELETE, DIRECTION 2 — the GEN path: Rust writes the equality delete, Java reads it.
//
// Sequence ordering is the correctness point. The data is `fast_append`ed first at sequence 1 and
// the equality delete is `row_delta`ed second at sequence 2, so the delete applies. Java reads the
// resulting `final.metadata.json` and asserts {10,30,50}.

/// Write a real parquet equality-delete file keyed on field id 1, deleting ids 20 and 40. The
/// Java-readable fixture's case of [`write_equality_delete_for_ids`]. Only `id` lands on disk.
async fn write_gen_equality_delete_file(table: &Table) -> DataFile {
    write_equality_delete_for_ids(table, &[20, 40]).await
}

#[tokio::test]
async fn test_scan_exec_gen_rust_writes_java_readable_equality_delete_table() {
    let Some(gen_dir) = eq_scan_gen_dir() else {
        println!(
            "skipping interop_scan_exec equality-delete GEN — set ICEBERG_INTEROP_EQ_SCAN_GEN_DIR \
             (run dev/java-interop/run-interop-eq-delete-d2.sh)"
        );
        return;
    };

    let warehouse = gen_dir.to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_eq_gen",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let table = create_rust_table(&catalog, &table_location).await;

    let data_file = write_gen_data_file(&table).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![data_file])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");

    // 3. row_delta a real equality delete for ids 20 and 40 at sequence 2. The data committed
    //    first at sequence 1, so the delete applies to it.
    let delete_file = write_gen_equality_delete_file(&table).await;
    assert_eq!(delete_file.content_type(), DataContentType::EqualityDeletes);
    assert_eq!(
        delete_file.equality_ids(),
        Some(vec![1]),
        "the equality delete must carry equality_ids = [1] (field id of `id`)"
    );
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // 4. Confirm our own scan already applies the equality delete before Java reads it.
    let batches: Vec<RecordBatch> = table
        .scan()
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect batches");
    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_rows(batch));
    }
    let rust_rows = sorted_by_id(rust_rows);
    let live_ids: Vec<i64> = rust_rows.iter().map(|r| r.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 50],
        "Rust's own scan of the written table must already be {{10,30,50}} (20/40 equality-deleted)"
    );

    let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), &final_metadata_path)
        .await
        .expect("write final.metadata.json");

    println!(
        "interop_scan_exec equality-delete GEN OK — Rust wrote {table_location} (parquet data seq 1 + \
         equality-delete seq 2 + final.metadata.json); Rust scan = {{10,30,50}}. Java verify-interop-eq-delete \
         reads it next."
    );
}

// PARTITIONED merge-on-read, DIRECTION 2 — the GEN path: Rust writes the partitioned table and the
// partition-scoped delete, Java reads it back. The partition-write proof.
//
// The production `DataFileWriter` and `PositionDeleteFileWriter` are built with a `PartitionKey`,
// which stamps the partition Struct and spec id onto each file and routes the parquet under the
// partition path. Data commits at sequence 1, the delete at sequence 2. Java asserts {10,30,40,50}.

/// The partitioned V2 schema Java expects. `category` is a required top-level field, and the spec
/// partitions by identity(category).
fn part_gen_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "category", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build the {id long, category string, data string} schema")
}

/// Build the identity(category) unbound partition spec (spec id 0) the partitioned table is created with.
fn part_gen_unbound_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "category".to_string(), Transform::Identity)
        .expect("add identity(category) partition field")
        .build()
}

/// Create the partitioned V2 table at exactly `<gen_dir>/rust_table`, so the on-disk layout is the
/// deterministic one Java loads.
async fn create_partitioned_rust_table(catalog: &impl Catalog, table_location: &str) -> Table {
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");

    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location.to_string())
        .schema(part_gen_schema())
        .partition_spec(part_gen_unbound_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();

    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create partitioned rust_table")
}

/// Build the `PartitionKey` for one identity(category) partition value, bound to the table's
/// default partition spec.
fn category_partition_key(schema: SchemaRef, spec: PartitionSpec, category: &str) -> PartitionKey {
    PartitionKey::new(
        spec,
        schema,
        Struct::from_iter([Some(Literal::string(category))]),
    )
    .expect("PartitionKey::new: valid partition tuple")
}

/// Write a real parquet data file for one partition. The writer stamps the partition Struct and
/// spec id onto the `DataFile` and routes the parquet under the partition path. Each row's
/// `category` matches the partition, so the data agrees with the stamp.
async fn write_partitioned_gen_data_file(
    table: &Table,
    partition_key: &PartitionKey,
    category: &str,
    ids: Vec<i64>,
    data_values: Vec<&str>,
) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));

    let row_count = ids.len();
    let categories: Vec<&str> = std::iter::repeat_n(category, row_count).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(categories)) as ArrayRef,
        Arc::new(StringArray::from(data_values)) as ArrayRef,
    ])
    .expect("build the per-partition data batch");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "rust-data".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        iceberg::spec::DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );

    // Built with the PartitionKey, `close()` stamps `partition` and `partition_spec_id` from the
    // key, and the location generator routes the parquet under the partition path.
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(Some(partition_key.clone()))
        .await
        .expect("build partitioned data file writer");
    writer
        .write(batch)
        .await
        .expect("write per-partition batch");
    writer
        .close()
        .await
        .expect("close partitioned data file writer")
        .into_iter()
        .next()
        .expect("one data file per partition")
}

/// Write a real parquet partition-scoped position-delete file, deleting position 1 of
/// `data_file_path`. The writer stamps the partition Struct and spec id onto the delete file, and
/// the delete-file index keys by partition and spec id, so the delete reaches only that partition.
async fn write_partitioned_gen_position_delete_file(
    table: &Table,
    partition_key: &PartitionKey,
    data_file_path: &str,
) -> DataFile {
    let config = PositionDeleteWriterConfig::new().expect("position-delete writer config");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "pos-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        iceberg::spec::DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        config.schema().clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );

    // Build with the caller's partition key, so the delete carries that partition Struct and spec
    // id. Reused by the identity(category), multi-file, and truncate fixtures.
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(Some(partition_key.clone()))
        .await
        .expect("build partition-scoped position-delete writer");

    let paths = StringArray::from(vec![data_file_path]);
    let positions = Int64Array::from(vec![1_i64]);
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(paths) as ArrayRef,
        Arc::new(positions) as ArrayRef,
    ])
    .expect("build the partition-scoped position-delete batch");
    writer
        .write(batch)
        .await
        .expect("write partition-scoped position-delete batch");
    writer
        .close()
        .await
        .expect("close partition-scoped position-delete writer")
        .into_iter()
        .next()
        .expect("one partition-scoped position-delete file")
}

#[tokio::test]
async fn test_part_scan_exec_gen_rust_writes_java_readable_partitioned_table() {
    let Some(gen_dir) = part_scan_gen_dir() else {
        println!(
            "skipping interop_scan_exec partitioned GEN — set ICEBERG_INTEROP_PART_SCAN_GEN_DIR \
             (run dev/java-interop/run-interop-part-d2.sh)"
        );
        return;
    };

    // 1. A MemoryCatalog over the local FS, table pinned to <gen_dir>/rust_table and partitioned
    //    by identity(category).
    let warehouse = gen_dir.to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_part_gen",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let table = create_partitioned_rust_table(&catalog, &table_location).await;

    let schema = table.metadata().current_schema().clone();
    let spec = table.metadata().default_partition_spec().as_ref().clone();
    let partition_key_a = category_partition_key(schema.clone(), spec.clone(), "a");
    let partition_key_b = category_partition_key(schema.clone(), spec.clone(), "b");

    // 2. Write one real parquet data file per partition, each stamped with its partition value,
    //    then fast_append both at sequence 1.
    let data_file_a =
        write_partitioned_gen_data_file(&table, &partition_key_a, "a", vec![10, 20, 30], vec![
            "x", "y", "z",
        ])
        .await;
    let data_file_b =
        write_partitioned_gen_data_file(&table, &partition_key_b, "b", vec![40, 50], vec![
            "p", "q",
        ])
        .await;

    // Each data file must carry the right partition value and spec id 0.
    assert_eq!(data_file_a.content_type(), DataContentType::Data);
    assert_eq!(data_file_b.content_type(), DataContentType::Data);
    assert_eq!(
        data_file_a.partition(),
        &Struct::from_iter([Some(Literal::string("a"))]),
        "cat=a data file must carry the category=a partition value"
    );
    assert_eq!(
        data_file_b.partition(),
        &Struct::from_iter([Some(Literal::string("b"))]),
        "cat=b data file must carry the category=b partition value"
    );

    let data_file_a_path = data_file_a.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![data_file_a, data_file_b])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");

    // 3. row_delta a partition-scoped position delete in partition a at sequence 2. The data
    //    committed first at sequence 1, so the delete applies.
    let delete_file =
        write_partitioned_gen_position_delete_file(&table, &partition_key_a, &data_file_a_path)
            .await;
    assert_eq!(delete_file.content_type(), DataContentType::PositionDeletes);
    assert_eq!(
        delete_file.partition(),
        &Struct::from_iter([Some(Literal::string("a"))]),
        "the position-delete must be PARTITION-SCOPED to category=a"
    );
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // 4. Confirm the table is internally consistent before Java reads it.
    let batches: Vec<RecordBatch> = table
        .scan()
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect batches");
    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_rows(batch));
    }
    let rust_rows = sorted_by_id(rust_rows);
    let live_ids: Vec<i64> = rust_rows.iter().map(|r| r.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 40, 50],
        "Rust's own scan of the written partitioned table must already be {{10,30,40,50}} (cat=a id=20 deleted)"
    );

    let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), &final_metadata_path)
        .await
        .expect("write final.metadata.json");

    println!(
        "interop_scan_exec partitioned GEN OK — Rust wrote {table_location} (per-partition parquet data + \
         partition-scoped position-delete + final.metadata.json); Rust scan = {{10,30,40,50}}. Java \
         verify-interop-part-scan reads it next."
    );
}

// ENGINE CUSTOM-SCAN EQUIVALENCE — the public `DeleteFilter` (Java
// `org.apache.iceberg.data.DeleteFilter`) reproduces the built-in scan exactly. An engine plans the
// files, reads each data file with the `parquet` crate directly, then reuses iceberg's delete
// resolution to drop deleted rows. These tests assert that path equals `to_arrow()` for position
// deletes, equality deletes, and the two combined. That equality is the seam's contract: an engine
// can bring its own physical scan and still get iceberg-correct merge-on-read rows.
//
// The unit test in `delete_filter.rs` proves `load` and `apply` in isolation. These resolve the same
// deletes, on a real committed table, two ways and assert equality.
//
// Non-vacuity: the equality predicate resolves `id` by Iceberg field id from the raw parquet
// `PARQUET_FIELD_ID_META_KEY`. A broken round-trip reads the column as absent, the predicate keeps
// every row, and the engine path diverges from ground truth, so these assertions fail loud.

/// Strip a `file://` scheme, so a `FileScanTask` data-file path opens as a local path. Either form
/// can arrive.
fn strip_file_scheme(path: &str) -> &str {
    path.strip_prefix("file://").unwrap_or(path)
}

/// Write a real parquet equality-delete file keyed on field id 1, deleting the given `ids`. The
/// writer projects the table schema down to `id`, so only `id` lands on disk.
async fn write_equality_delete_for_ids(table: &Table, ids: &[i64]) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    // The config builds a projector from the full table schema down to `id`, so it takes a
    // full-schema batch and extracts the `id` values.
    let config = EqualityDeleteWriterConfig::new(vec![1], schema.clone())
        .expect("equality-delete writer config (equality_ids = [1])");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "eq-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        iceberg::spec::DataFileFormat::Parquet,
    );
    // The parquet writer must use the projected schema, because that is what lands on disk.
    let projected_iceberg_schema = Arc::new(
        iceberg::arrow::arrow_schema_to_schema(config.projected_arrow_schema_ref())
            .expect("projected arrow schema → iceberg schema"),
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        projected_iceberg_schema,
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );

    let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .build(None)
        .await
        .expect("build equality-delete writer");

    // A full-schema batch carrying the delete keys. The projector keeps only `id`, but the batch
    // must match the full table schema for the column-index projection to resolve.
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));
    let data: Vec<&str> = std::iter::repeat_n("x", ids.len()).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
        Arc::new(StringArray::from(data)) as ArrayRef,
    ])
    .expect("build the equality-delete key batch");
    writer
        .write(batch)
        .await
        .expect("write equality-delete batch");
    writer
        .close()
        .await
        .expect("close equality-delete writer")
        .into_iter()
        .next()
        .expect("one equality-delete file")
}

/// The downstream-engine custom-scan path, over the public surface. Plans the files, reads each
/// data file with the `parquet` crate directly, then drops deleted rows through [`DeleteFilter`].
/// The returned rows must equal the built-in `to_arrow()` scan.
async fn engine_custom_scan_rows(table: &Table) -> Vec<ScanRow> {
    let tasks: Vec<_> = table
        .scan()
        .build()
        .expect("build scan")
        .plan_files()
        .await
        .expect("plan files")
        .try_collect()
        .await
        .expect("collect scan tasks");

    let mut rows = Vec::new();
    for task in &tasks {
        // Resolve the task's deletes once. Position deletes load eagerly, the predicate lazily.
        let deletes = DeleteFilter::load(task, table.file_io().clone())
            .await
            .expect("load DeleteFilter");
        let equality_predicate = deletes
            .equality_delete_predicate(task)
            .await
            .expect("resolve equality-delete predicate");

        // The engine's own read: every physical row, in file order, with no deletes applied.
        let file = fs::File::open(strip_file_scheme(task.data_file_path()))
            .expect("open data-file parquet for the engine's own read");
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("parquet reader builder")
            .build()
            .expect("build parquet record-batch reader");

        // Apply the deletes batch by batch, tracking the file position of each batch's row 0.
        let mut row_base = 0u64;
        for batch in reader {
            let batch = batch.expect("read a data batch");
            let row_count = batch.num_rows() as u64;
            let survivors = deletes
                .apply(task, batch, row_base, equality_predicate.as_ref())
                .expect("apply merge-on-read deletes");
            rows.extend(extract_rows(&survivors));
            row_base += row_count;
        }
    }
    rows
}

/// The built-in scan, the ground truth. `to_arrow()` applies the same deletes internally.
async fn builtin_scan_rows(table: &Table) -> Vec<ScanRow> {
    let batches: Vec<RecordBatch> = table
        .scan()
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect scan batches");
    batches.iter().flat_map(extract_rows).collect()
}

/// Build a fresh unpartitioned 5-row table and `fast_append` it at sequence 1. The shared setup for
/// the equivalence scenarios below.
async fn append_5row_table(catalog: &impl Catalog, table_location: &str) -> (Table, String) {
    let table = create_rust_table(catalog, table_location).await;
    let data_file = write_gen_data_file(&table).await;
    let data_file_path = data_file.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![data_file])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(catalog).await.expect("commit fast append");
    (table, data_file_path)
}

#[tokio::test]
async fn test_engine_deletefilter_equivalence_position_deletes() {
    use tempfile::TempDir;

    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "equivalence_pos",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let (table, data_file_path) = append_5row_table(&catalog, &table_location).await;

    // Commit a position delete of ids 20 and 40, discovered the way an engine would.
    let mut pairs = discover_row_identities(&table, &[20, 40]).await;
    pairs.sort();
    assert_eq!(
        pairs,
        vec![
            (data_file_path.clone(), 1_i64),
            (data_file_path.clone(), 3_i64),
        ],
        "ids 20/40 sit at file positions 1/3"
    );
    let delete_file = write_pos_delete_from_pairs(&table, &pairs).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // The public `deleted_row_positions`, Java `deletedRowPositions()`, reports exactly {1, 3}.
    {
        let tasks: Vec<_> = table
            .scan()
            .build()
            .expect("build scan")
            .plan_files()
            .await
            .expect("plan files")
            .try_collect()
            .await
            .expect("collect tasks");
        let deletes = DeleteFilter::load(&tasks[0], table.file_io().clone())
            .await
            .expect("load DeleteFilter");
        let positions = deletes
            .deleted_row_positions(&tasks[0])
            .expect("a positional delete vector is present");
        assert_eq!(positions.len(), 2, "exactly two positions are deleted");
        assert!(
            positions.contains(1) && positions.contains(3),
            "positions 1 and 3 (ids 20, 40) are the deleted positions"
        );
    }

    // The equivalence: the engine DeleteFilter path equals the built-in scan.
    let ground_truth = sorted_by_id(builtin_scan_rows(&table).await);
    let engine = sorted_by_id(engine_custom_scan_rows(&table).await);
    assert_eq!(
        engine, ground_truth,
        "engine DeleteFilter (raw read + apply) must equal the built-in delete-applying scan"
    );
    let live_ids: Vec<i64> = engine.iter().map(|row| row.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 50],
        "position-delete survivors are exactly {{10, 30, 50}}"
    );
}

#[tokio::test]
async fn test_engine_deletefilter_equivalence_equality_deletes() {
    use tempfile::TempDir;

    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "equivalence_eq",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let (table, _data_file_path) = append_5row_table(&catalog, &table_location).await;

    // Commit an equality delete of ids 20 and 40, keyed on field id 1, at sequence 2.
    let delete_file = write_equality_delete_for_ids(&table, &[20, 40]).await;
    assert_eq!(delete_file.content_type(), DataContentType::EqualityDeletes);
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // An equality-only task carries a predicate, Java `eqDeletedRowFilter()`, and no positional
    // deletes.
    {
        let tasks: Vec<_> = table
            .scan()
            .build()
            .expect("build scan")
            .plan_files()
            .await
            .expect("plan files")
            .try_collect()
            .await
            .expect("collect tasks");
        let deletes = DeleteFilter::load(&tasks[0], table.file_io().clone())
            .await
            .expect("load DeleteFilter");
        assert!(
            deletes
                .equality_delete_predicate(&tasks[0])
                .await
                .expect("resolve equality-delete predicate")
                .is_some(),
            "an equality-delete predicate is present"
        );
        let positions = deletes.deleted_row_positions(&tasks[0]);
        assert!(
            positions.is_none_or(|dv| dv.is_empty()),
            "an equality-only task has no positional deletes"
        );
    }

    // The equivalence: the engine equality path equals the built-in scan.
    let ground_truth = sorted_by_id(builtin_scan_rows(&table).await);
    let engine = sorted_by_id(engine_custom_scan_rows(&table).await);
    assert_eq!(
        engine, ground_truth,
        "engine equality DeleteFilter (raw read + predicate apply) must equal the built-in equality scan"
    );
    let live_ids: Vec<i64> = engine.iter().map(|row| row.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 50],
        "equality-delete survivors are exactly {{10, 30, 50}}"
    );
}

#[tokio::test]
async fn test_engine_deletefilter_equivalence_position_and_equality_deletes() {
    use tempfile::TempDir;

    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "equivalence_combined",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let (table, data_file_path) = append_5row_table(&catalog, &table_location).await;

    // Commit a position delete and an equality delete in one row_delta at sequence 2, which
    // exercises the combined mask path of `DeleteFilter::apply`.
    let pos_delete = write_pos_delete_from_pairs(&table, &[(data_file_path.clone(), 0)]).await;
    let eq_delete = write_equality_delete_for_ids(&table, &[30]).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![pos_delete, eq_delete])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // The equivalence: the engine combined path equals the built-in scan.
    let ground_truth = sorted_by_id(builtin_scan_rows(&table).await);
    let engine = sorted_by_id(engine_custom_scan_rows(&table).await);
    assert_eq!(
        engine, ground_truth,
        "engine combined DeleteFilter (positional AND equality mask) must equal the built-in scan"
    );
    let live_ids: Vec<i64> = engine.iter().map(|row| row.id).collect();
    assert_eq!(
        live_ids,
        vec![20, 40, 50],
        "combined survivors are exactly {{20, 40, 50}} (position 0 = id 10, and id 30, deleted)"
    );
}

// MULTI-FILE-PER-PARTITION merge-on-read. When one partition holds more than one data file, the
// `DeleteFileIndex` routes a partition-scoped position delete to EVERY data-file task in that
// partition. The delete must still apply only to the rows of the data file it references by path.
// The path-keyed loader `parse_positional_deletes_record_batch_stream` is what makes that hold.
// A loader that partition-broadcast positions instead would make the sibling row vanish.

#[tokio::test]
async fn test_engine_deletefilter_multifile_partition_position_delete_spares_sibling() {
    use tempfile::TempDir;

    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "equivalence_multifile",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let table = create_partitioned_rust_table(&catalog, &table_location).await;

    let schema = table.metadata().current_schema().clone();
    let spec = table.metadata().default_partition_spec().as_ref().clone();
    let key_a = category_partition_key(schema.clone(), spec.clone(), "a");

    // Two data files in the same partition: file1 ids {10,20,30}, file2 ids {40,50,60}, both at
    // positions 0..2 and both stamped category=a.
    let file1 =
        write_partitioned_gen_data_file(&table, &key_a, "a", vec![10, 20, 30], vec!["x", "y", "z"])
            .await;
    let file2 =
        write_partitioned_gen_data_file(&table, &key_a, "a", vec![40, 50, 60], vec!["p", "q", "r"])
            .await;
    let file1_path = file1.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file1, file2])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");

    // A partition-scoped position delete referencing file1 at position 1. The DeleteFileIndex routes
    // it to both tasks, so the loader must apply it to file1 by path only. Id 50 must survive.
    let delete_file = write_partitioned_gen_position_delete_file(&table, &key_a, &file1_path).await;
    assert_eq!(delete_file.content_type(), DataContentType::PositionDeletes);
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // Both paths must agree, and both must drop file1's id 20 only.
    let ground_truth = sorted_by_id(builtin_scan_rows(&table).await);
    let engine = sorted_by_id(engine_custom_scan_rows(&table).await);
    assert_eq!(
        engine, ground_truth,
        "engine DeleteFilter path == built-in scan for a multi-file partition"
    );
    let live_ids: Vec<i64> = engine.iter().map(|row| row.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 40, 50, 60],
        "only file1's position 1 (id 20) is deleted; file2's same-ordinal position 1 (id 50) is SPARED"
    );
    assert!(
        engine.iter().any(|row| row.id == 50),
        "id 50 (file2 position 1) MUST survive — position deletes are path-keyed, not partition-broadcast"
    );
    assert!(
        !engine.iter().any(|row| row.id == 20),
        "id 20 (file1 position 1) must be deleted"
    );
}

/// Write a real parquet partition-scoped equality-delete file keyed on field id 1, for the
/// partitioned schema. The writer stamps the partition Struct and spec id onto the delete file, so
/// the `DeleteFileIndex` routes it to that partition's tasks. Only `id` lands on disk.
async fn write_partitioned_equality_delete_for_ids(
    table: &Table,
    partition_key: &PartitionKey,
    category: &str,
    ids: &[i64],
) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let config = EqualityDeleteWriterConfig::new(vec![1], schema.clone())
        .expect("equality-delete writer config (equality_ids = [1])");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "eq-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        iceberg::spec::DataFileFormat::Parquet,
    );
    let projected_iceberg_schema = Arc::new(
        iceberg::arrow::arrow_schema_to_schema(config.projected_arrow_schema_ref())
            .expect("projected arrow schema → iceberg schema"),
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        projected_iceberg_schema,
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    // Build with the partition key, so the delete carries the partition Struct and spec id.
    let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .build(Some(partition_key.clone()))
        .await
        .expect("build partition-scoped equality-delete writer");

    // A full-schema batch carrying the delete keys. The projector keeps only `id`.
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));
    let categories: Vec<&str> = std::iter::repeat_n(category, ids.len()).collect();
    let data: Vec<&str> = std::iter::repeat_n("x", ids.len()).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
        Arc::new(StringArray::from(categories)) as ArrayRef,
        Arc::new(StringArray::from(data)) as ArrayRef,
    ])
    .expect("build the partition-scoped equality-delete key batch");
    writer
        .write(batch)
        .await
        .expect("write equality-delete batch");
    writer
        .close()
        .await
        .expect("close equality-delete writer")
        .into_iter()
        .next()
        .expect("one equality-delete file")
}

#[tokio::test]
async fn test_engine_deletefilter_multifile_partition_equality_delete_applies_across_files() {
    use tempfile::TempDir;

    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "equivalence_multifile_eq",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let table = create_partitioned_rust_table(&catalog, &table_location).await;

    let schema = table.metadata().current_schema().clone();
    let spec = table.metadata().default_partition_spec().as_ref().clone();
    let key_a = category_partition_key(schema.clone(), spec.clone(), "a");

    // Two data files in the same partition: file1 ids {10,20,30}, file2 ids {40,50,60}.
    let file1 =
        write_partitioned_gen_data_file(&table, &key_a, "a", vec![10, 20, 30], vec!["x", "y", "z"])
            .await;
    let file2 =
        write_partitioned_gen_data_file(&table, &key_a, "a", vec![40, 50, 60], vec!["p", "q", "r"])
            .await;
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file1, file2])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");

    // A partition-scoped equality delete for {20, 50}. Id 20 lives in file1 and id 50 in file2, so
    // the delete must apply to both data files of partition a.
    let delete_file =
        write_partitioned_equality_delete_for_ids(&table, &key_a, "a", &[20, 50]).await;
    assert_eq!(delete_file.content_type(), DataContentType::EqualityDeletes);
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // Both paths must agree, and both must drop id 20 from file1 and id 50 from file2.
    let ground_truth = sorted_by_id(builtin_scan_rows(&table).await);
    let engine = sorted_by_id(engine_custom_scan_rows(&table).await);
    assert_eq!(
        engine, ground_truth,
        "engine equality DeleteFilter path == built-in scan across a multi-file partition"
    );
    let live_ids: Vec<i64> = engine.iter().map(|row| row.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 40, 60],
        "the partition eq-delete drops id 20 (file1) AND id 50 (file2) — it applies to BOTH files"
    );
}

// MULTI-FILE-PER-PARTITION merge-on-read INTEROP — the bidirectional proof. Direction 1: Java writes
// two data files in one partition plus a position delete on file1, and Rust must drop id 20 (file1
// position 1) while sparing id 50 (file2's same ordinal). Direction 2 writes the same shape from
// Rust for Java to read back.

#[tokio::test]
async fn test_multifile_scan_exec_matches_java_read() {
    let Some(dir) = multifile_scan_dir() else {
        println!(
            "skipping interop_scan_exec multi-file — set ICEBERG_INTEROP_MULTIFILE_SCAN_DIR \
             (run dev/java-interop/run-interop-multifile-scan.sh)"
        );
        return;
    };

    let table = load_table(&dir);

    // The scan applies the partition-scoped position delete, path-keyed to file1 only.
    let batches: Vec<RecordBatch> = table
        .scan()
        .build()
        .expect("build table scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect scan batches");
    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_rows(batch));
    }
    let rust_rows = sorted_by_id(rust_rows);
    let java_rows = sorted_by_id(read_java_multifile_rows(&dir));

    // 5 live rows: 6 written across two files in one partition, file1's position 1 deleted.
    assert_eq!(
        rust_rows.len(),
        5,
        "exactly 5 rows survive (6 written in 2 files, file1 position 1 deleted)"
    );
    // File1's id 20 is deleted. File2's id 50, at its own position 1, must survive.
    assert!(
        !rust_rows.iter().any(|r| r.id == 20),
        "id 20 (file1 position 1) must be ABSENT"
    );
    assert!(
        rust_rows.iter().any(|r| r.id == 50),
        "id 50 (file2 position 1, the SIBLING) must be PRESENT — the delete is path-keyed, not broadcast"
    );
    assert_eq!(
        rust_rows, java_rows,
        "Rust multi-file scan→Arrow rows must equal Java's IcebergGenerics read"
    );
    let live_ids: Vec<i64> = rust_rows.iter().map(|r| r.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 40, 50, 60],
        "the live id set after multi-file merge-on-read is exactly {{10, 30, 40, 50, 60}}"
    );

    println!(
        "interop_scan_exec multi-file OK — Rust scan = Java read: 5 live rows {{10,30,40,50,60}}, \
         file1 id 20 deleted, file2 sibling id 50 spared"
    );
}

// FILE-SCOPED position-delete routing INTEROP — Java's `DeleteFileIndex` routes a position delete
// with a derivable referenced data file into a path-keyed map, consulted with no spec and no
// partition condition. Java writes deletes stamped with a spec and partition that match neither
// data file: the field leg through `referenced_data_file`, the bounds leg through equal `file_path`
// bounds, which is the shape Java's own `PositionDeleteWriter` emits. A partition-scoped control
// delete must not apply.

#[tokio::test]
async fn test_file_scoped_delete_scan_matches_java_read() {
    let Some(dir) = file_scoped_deletes_dir() else {
        println!(
            "skipping interop_scan_exec file-scoped deletes — set \
             ICEBERG_INTEROP_FILE_SCOPED_DELETES_DIR \
             (run dev/java-interop/run-interop-file-scoped-deletes.sh)"
        );
        return;
    };

    let table = load_table(&dir);

    // Plan-level parity with Java `DeleteFileIndex.forDataFile`: which delete files attach to which
    // data file. Asserted before the rows, because a row-level result can match by coincidence while
    // the wrong delete files are attached.
    let mut planned: Vec<(String, Vec<String>)> = table
        .scan()
        .build()
        .expect("build table scan for planning")
        .plan_files()
        .await
        .expect("plan files")
        .try_collect::<Vec<_>>()
        .await
        .expect("collect file scan tasks")
        .into_iter()
        .map(|task| {
            let mut deletes: Vec<String> = task
                .deletes
                .iter()
                .map(|delete| file_name(&delete.file_path))
                .collect();
            deletes.sort();
            (file_name(&task.data_file_path), deletes)
        })
        .collect();
    planned.sort();
    assert_eq!(
        planned,
        vec![
            ("00000-a.parquet".to_string(), vec![
                "00000-field-leg-deletes.parquet".to_string()
            ]),
            ("00000-b.parquet".to_string(), vec![
                "00000-bounds-leg-deletes.parquet".to_string()
            ]),
        ],
        "each data file must receive EXACTLY the file-scoped delete that references it — and the \
         partition-scoped control (stamped an empty partition) must reach neither"
    );

    let batches: Vec<RecordBatch> = table
        .scan()
        .build()
        .expect("build table scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect scan batches");
    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_rows(batch));
    }
    let rust_rows = sorted_by_id(rust_rows);
    let java_rows = sorted_by_id(read_java_file_scoped_rows(&dir));

    let live_ids: Vec<i64> = rust_rows.iter().map(|row| row.id).collect();
    // Id 20 is deleted by the field leg, `referenced_data_file` = file A, even though that delete is
    // stamped spec 0 and an empty partition while file A is spec 1 and category=a.
    assert!(
        !live_ids.contains(&20),
        "id 20 must be deleted by the file-scoped delete carrying referenced_data_file, live: \
         {live_ids:?}"
    );
    // Id 50 is deleted by the bounds leg, which has no `referenced_data_file` and only equal
    // `file_path` bounds naming file B. Java's `PositionDeleteWriter` emits this shape.
    assert!(
        !live_ids.contains(&50),
        "id 50 must be deleted by the file-scoped delete identified by its file_path bounds, live: \
         {live_ids:?}"
    );
    // Id 30 is named by the control delete, which is partition-scoped and stamped the empty
    // category=c while the row lives in category=a. Java does not apply it, so Rust must not either.
    assert!(
        live_ids.contains(&30),
        "id 30 must SURVIVE — the control delete is partition-scoped and its partition does not \
         match the data file's, live: {live_ids:?}"
    );
    assert_eq!(
        live_ids,
        vec![10, 30, 40, 60],
        "the live id set after file-scoped merge-on-read is exactly {{10, 30, 40, 60}}"
    );
    assert_eq!(
        rust_rows, java_rows,
        "Rust scan→Arrow rows must equal Java's own IcebergGenerics read of the same table"
    );

    println!(
        "interop_scan_exec file-scoped deletes OK — Rust scan = Java read: live rows \
         {{10,30,40,60}}, field-leg id 20 and bounds-leg id 50 deleted across the spec/partition \
         mismatch, partition-scoped control id 30 spared"
    );
}

/// The cross-task over-delete pin. The control delete is stamped `category=b`, so the plan attaches
/// it to file B's task, but its rows name file A's position 2.
#[tokio::test]
async fn test_file_scoped_delete_crosstask_control_does_not_leak() {
    let Some(dir) = file_scoped_deletes_crosstask_dir() else {
        println!(
            "skipping interop_scan_exec cross-task file-scoped deletes — set \
             ICEBERG_INTEROP_FILE_SCOPED_DELETES_CROSSTASK_DIR \
             (run dev/java-interop/run-interop-file-scoped-deletes.sh)"
        );
        return;
    };

    let table = load_table(&dir);

    // The control must attach to file B's task first. That attachment is the hazard, so a fixture
    // whose control attached to nothing would pass the row assertions vacuously.
    let mut planned: Vec<(String, Vec<String>)> = table
        .scan()
        .build()
        .expect("build table scan for planning")
        .plan_files()
        .await
        .expect("plan files")
        .try_collect::<Vec<_>>()
        .await
        .expect("collect file scan tasks")
        .into_iter()
        .map(|task| {
            let mut deletes: Vec<String> = task
                .deletes
                .iter()
                .map(|delete| file_name(&delete.file_path))
                .collect();
            deletes.sort();
            (file_name(&task.data_file_path), deletes)
        })
        .collect();
    planned.sort();
    assert_eq!(
        planned,
        vec![
            ("00000-a.parquet".to_string(), vec![
                "00000-field-leg-deletes.parquet".to_string()
            ]),
            ("00000-b.parquet".to_string(), vec![
                "00000-bounds-leg-deletes.parquet".to_string(),
                "00000-control-deletes.parquet".to_string(),
            ]),
        ],
        "the cross-task control (partition-scoped, category=b) must attach to file B's task and \
         ONLY there — its rows naming file A must not re-route it"
    );

    let batches: Vec<RecordBatch> = table
        .scan()
        .build()
        .expect("build table scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect scan batches");
    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_rows(batch));
    }
    let rust_rows = sorted_by_id(rust_rows);
    let java_rows = sorted_by_id(read_java_file_scoped_rows(&dir));

    let live_ids: Vec<i64> = rust_rows.iter().map(|row| row.id).collect();
    // The pin: id 30 must survive. Shared delete state across tasks deletes it.
    assert!(
        live_ids.contains(&30),
        "id 30 must SURVIVE — the control delete belongs to file B's task and its rows for file A \
         must not leak across tasks (the R117 over-delete), live: {live_ids:?}"
    );
    // The task's legitimate deletes still apply: field-leg id 20, bounds-leg id 50.
    assert_eq!(
        live_ids,
        vec![10, 30, 40, 60],
        "the live id set must match Java's per-task merge-on-read exactly"
    );
    assert_eq!(
        rust_rows, java_rows,
        "Rust scan→Arrow rows must equal Java's own IcebergGenerics read of the same table"
    );

    println!(
        "interop_scan_exec cross-task file-scoped deletes OK — Rust scan = Java read: live rows \
         {{10,30,40,60}}; the category=b control stayed scoped to file B's task and id 30 survived"
    );
}

#[tokio::test]
async fn test_multifile_scan_exec_gen_rust_writes_java_readable_table() {
    let Some(gen_dir) = multifile_scan_gen_dir() else {
        println!(
            "skipping interop_scan_exec multi-file GEN — set ICEBERG_INTEROP_MULTIFILE_SCAN_GEN_DIR \
             (run dev/java-interop/run-interop-multifile-scan-d2.sh)"
        );
        return;
    };

    let warehouse = gen_dir.to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_multifile_gen",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let table = create_partitioned_rust_table(&catalog, &table_location).await;

    let schema = table.metadata().current_schema().clone();
    let spec = table.metadata().default_partition_spec().as_ref().clone();
    let key_a = category_partition_key(schema.clone(), spec.clone(), "a");

    // Two data files in the same partition: file1 ids {10,20,30}, file2 ids {40,50,60}.
    let file1 =
        write_partitioned_gen_data_file(&table, &key_a, "a", vec![10, 20, 30], vec!["x", "y", "z"])
            .await;
    let file2 =
        write_partitioned_gen_data_file(&table, &key_a, "a", vec![40, 50, 60], vec!["p", "q", "r"])
            .await;
    let file1_path = file1.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file1, file2])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");

    // A partition-scoped position delete on file1 position 1. File2's id 50 must survive.
    let delete_file = write_partitioned_gen_position_delete_file(&table, &key_a, &file1_path).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // Confirm our own scan reads {10,30,40,50,60} before Java reads it.
    let batches: Vec<RecordBatch> = table
        .scan()
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect batches");
    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_rows(batch));
    }
    let live_ids: Vec<i64> = sorted_by_id(rust_rows).iter().map(|r| r.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 40, 50, 60],
        "Rust's own scan of the written multi-file table must be {{10,30,40,50,60}} (file1 id 20 deleted)"
    );

    let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), &final_metadata_path)
        .await
        .expect("write final.metadata.json");

    println!(
        "interop_scan_exec multi-file GEN OK — Rust wrote {table_location} (two data files in one \
         partition + partition-scoped position-delete + final.metadata.json); Rust scan = \
         {{10,30,40,50,60}}. Java verify-interop-multifile-scan reads it next."
    );
}

// NON-IDENTITY TRANSFORM merge-on-read INTEROP — the bidirectional proof. The table is partitioned
// by `truncate[10](id)`, so no raw id equals its partition value. This proves the `DeleteFileIndex`
// matches a partition-scoped delete to the transformed partition Struct, not to a raw column value.
// truncate=10 holds ids 11/13/15 and truncate=20 holds 21/23. A delete in partition 10 removes
// position 1, so the live set is {11,15,21,23}.

/// The dir the Java oracle wrote the truncate-partitioned table and JSON rows into.
fn nonidentity_scan_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_NONIDENTITY_SCAN_DIR").map(PathBuf::from)
}

/// The dir the direction-2 GEN path writes a Rust-authored truncate-partitioned table into.
fn nonidentity_scan_gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_NONIDENTITY_SCAN_GEN_DIR").map(PathBuf::from)
}

/// Load the Java ground-truth non-identity rows.
fn read_java_nonidentity_rows(dir: &std::path::Path) -> Vec<ScanRow> {
    let path = dir.join("java_nonidentity_scan_rows.json");
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str::<Vec<ScanRow>>(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// The `truncate[10](id)` unbound partition spec, a non-identity transform over [`gen_schema`].
fn truncate_gen_unbound_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(1, "id_trunc".to_string(), Transform::Truncate(10))
        .expect("add truncate[10](id) partition field")
        .build()
}

/// Create the `{id, data}` V2 table at `<gen_dir>/rust_table` partitioned by `truncate[10](id)`.
async fn create_truncate_partitioned_rust_table(
    catalog: &impl Catalog,
    table_location: &str,
) -> Table {
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");

    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location.to_string())
        .schema(gen_schema())
        .partition_spec(truncate_gen_unbound_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();

    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create truncate-partitioned rust_table")
}

/// Build the `PartitionKey` for one `truncate[10](id)` partition value, bound to the table's
/// default spec.
fn truncate_partition_key(schema: SchemaRef, spec: PartitionSpec, value: i64) -> PartitionKey {
    PartitionKey::new(
        spec,
        schema,
        Struct::from_iter([Some(Literal::long(value))]),
    )
    .expect("PartitionKey::new: valid partition tuple")
}

/// Write a real parquet data file for one `truncate[10](id)` partition. The writer stamps the
/// transformed partition value onto the `DataFile`. Every id the caller passes must truncate to the
/// key's value, so the data agrees with the stamp.
async fn write_truncate_gen_data_file(
    table: &Table,
    partition_key: &PartitionKey,
    ids: Vec<i64>,
    data_values: Vec<&str>,
) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(data_values)) as ArrayRef,
    ])
    .expect("build the per-partition data batch");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "rust-data".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        iceberg::spec::DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(Some(partition_key.clone()))
        .await
        .expect("build truncate-partitioned data file writer");
    writer
        .write(batch)
        .await
        .expect("write per-partition batch");
    writer
        .close()
        .await
        .expect("close truncate-partitioned data file writer")
        .into_iter()
        .next()
        .expect("one data file per partition")
}

#[tokio::test]
async fn test_nonidentity_scan_exec_matches_java_read() {
    let Some(dir) = nonidentity_scan_dir() else {
        println!(
            "skipping interop_scan_exec non-identity — set ICEBERG_INTEROP_NONIDENTITY_SCAN_DIR \
             (run dev/java-interop/run-interop-nonidentity-scan.sh)"
        );
        return;
    };

    let table = load_table(&dir);

    let batches: Vec<RecordBatch> = table
        .scan()
        .build()
        .expect("build table scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect scan batches");
    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_rows(batch));
    }
    let rust_rows = sorted_by_id(rust_rows);
    let java_rows = sorted_by_id(read_java_nonidentity_rows(&dir));

    // 4 live rows: 5 written across two truncate partitions, truncate=10's position 1 deleted.
    assert_eq!(
        rust_rows.len(),
        4,
        "exactly 4 rows survive (5 written, truncate=10 position 1 deleted)"
    );
    // truncate=10 loses id 13. truncate=20 stays intact.
    assert!(
        !rust_rows.iter().any(|r| r.id == 13),
        "id 13 (truncate=10 partition, position 1) must be ABSENT"
    );
    assert!(
        rust_rows.iter().any(|r| r.id == 21) && rust_rows.iter().any(|r| r.id == 23),
        "the truncate=20 partition (ids 21, 23) must be intact — the delete matched only truncate=10"
    );
    assert_eq!(
        rust_rows, java_rows,
        "Rust non-identity (truncate) scan→Arrow rows must equal Java's IcebergGenerics read"
    );
    let live_ids: Vec<i64> = rust_rows.iter().map(|r| r.id).collect();
    assert_eq!(
        live_ids,
        vec![11, 15, 21, 23],
        "the live id set after truncate-partition merge-on-read is exactly {{11, 15, 21, 23}}"
    );

    println!(
        "interop_scan_exec non-identity OK — Rust scan = Java read: {{11,15,21,23}}, \
         truncate=10 id 13 deleted, truncate=20 intact"
    );
}

#[tokio::test]
async fn test_nonidentity_scan_exec_gen_rust_writes_java_readable_table() {
    let Some(gen_dir) = nonidentity_scan_gen_dir() else {
        println!(
            "skipping interop_scan_exec non-identity GEN — set ICEBERG_INTEROP_NONIDENTITY_SCAN_GEN_DIR \
             (run dev/java-interop/run-interop-nonidentity-scan-d2.sh)"
        );
        return;
    };

    let warehouse = gen_dir.to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_nonidentity_gen",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let table = create_truncate_partitioned_rust_table(&catalog, &table_location).await;

    let schema = table.metadata().current_schema().clone();
    let spec = table.metadata().default_partition_spec().as_ref().clone();
    let key_10 = truncate_partition_key(schema.clone(), spec.clone(), 10);
    let key_20 = truncate_partition_key(schema.clone(), spec.clone(), 20);

    // truncate=10 holds ids 11/13/15. truncate=20 holds 21/23.
    let file_10 =
        write_truncate_gen_data_file(&table, &key_10, vec![11, 13, 15], vec!["x", "y", "z"]).await;
    let file_20 = write_truncate_gen_data_file(&table, &key_20, vec![21, 23], vec!["p", "q"]).await;
    let file_10_path = file_10.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file_10, file_20])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");

    // A partition-scoped position delete in partition truncate=10, at position 1.
    let delete_file =
        write_partitioned_gen_position_delete_file(&table, &key_10, &file_10_path).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // Confirm our own scan reads {11,15,21,23} before Java reads it.
    let batches: Vec<RecordBatch> = table
        .scan()
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect batches");
    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_rows(batch));
    }
    let live_ids: Vec<i64> = sorted_by_id(rust_rows).iter().map(|r| r.id).collect();
    assert_eq!(
        live_ids,
        vec![11, 15, 21, 23],
        "Rust's own scan of the written truncate-partitioned table must be {{11,15,21,23}} (id 13 deleted)"
    );

    let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), &final_metadata_path)
        .await
        .expect("write final.metadata.json");

    println!(
        "interop_scan_exec non-identity GEN OK — Rust wrote {table_location} (truncate[10](id) \
         partitioned + partition-scoped position-delete on truncate=10 + final.metadata.json); Rust \
         scan = {{11,15,21,23}}. Java verify-interop-nonidentity-scan reads it next."
    );
}

/// Write a real parquet partition-scoped equality-delete file for [`gen_schema`], keyed on field
/// id 1. The writer stamps the transformed partition Struct and spec id onto the delete file, so the
/// `DeleteFileIndex` routes it by the truncate value.
async fn write_truncate_partitioned_equality_delete_for_ids(
    table: &Table,
    partition_key: &PartitionKey,
    ids: &[i64],
) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let config = EqualityDeleteWriterConfig::new(vec![1], schema.clone())
        .expect("equality-delete writer config (equality_ids = [1])");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "eq-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        iceberg::spec::DataFileFormat::Parquet,
    );
    let projected_iceberg_schema = Arc::new(
        iceberg::arrow::arrow_schema_to_schema(config.projected_arrow_schema_ref())
            .expect("projected arrow schema → iceberg schema"),
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        projected_iceberg_schema,
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    // Build with the truncate partition key, so the delete carries the transformed Struct.
    let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .build(Some(partition_key.clone()))
        .await
        .expect("build truncate-partition-scoped equality-delete writer");

    // A full-schema batch carrying the delete keys. The projector keeps only `id`.
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));
    let data: Vec<&str> = std::iter::repeat_n("x", ids.len()).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
        Arc::new(StringArray::from(data)) as ArrayRef,
    ])
    .expect("build the truncate-scoped equality-delete key batch");
    writer
        .write(batch)
        .await
        .expect("write equality-delete batch");
    writer
        .close()
        .await
        .expect("close equality-delete writer")
        .into_iter()
        .next()
        .expect("one equality-delete file")
}

/// The non-identity `DeleteFilter`-equivalence proof (ENGINE_CONTRACT §2). The layout is
/// `truncate[10](id)`: truncate=10 holds ids {11,13,15}, truncate=20 holds {21,23}.
#[tokio::test]
async fn test_engine_deletefilter_nonidentity_partition_equivalence() {
    use tempfile::TempDir;

    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "equivalence_nonidentity",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let table = create_truncate_partitioned_rust_table(&catalog, &table_location).await;

    let schema = table.metadata().current_schema().clone();
    let spec = table.metadata().default_partition_spec().as_ref().clone();
    let key_10 = truncate_partition_key(schema.clone(), spec.clone(), 10);
    let key_20 = truncate_partition_key(schema.clone(), spec.clone(), 20);

    // truncate=10 holds ids 11/13/15. truncate=20 holds 21/23.
    let file_10 =
        write_truncate_gen_data_file(&table, &key_10, vec![11, 13, 15], vec!["x", "y", "z"]).await;
    let file_20 = write_truncate_gen_data_file(&table, &key_20, vec![21, 23], vec!["p", "q"]).await;
    let file_10_path = file_10.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file_10, file_20])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");

    // A position delete scoped to truncate=10 and an equality delete scoped to truncate=20, both
    // stamped with transformed partition values.
    let pos_delete =
        write_partitioned_gen_position_delete_file(&table, &key_10, &file_10_path).await;
    assert_eq!(pos_delete.content_type(), DataContentType::PositionDeletes);
    let eq_delete =
        write_truncate_partitioned_equality_delete_for_ids(&table, &key_20, &[21]).await;
    assert_eq!(eq_delete.content_type(), DataContentType::EqualityDeletes);
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![pos_delete, eq_delete])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // The engine path must equal the built-in scan over the non-identity layout.
    let ground_truth = sorted_by_id(builtin_scan_rows(&table).await);
    let engine = sorted_by_id(engine_custom_scan_rows(&table).await);
    assert_eq!(
        engine, ground_truth,
        "engine DeleteFilter path == built-in scan over a truncate[10](id) (non-identity) layout"
    );

    // Pin the live set too. The equivalence alone tolerates a regression that breaks both paths.
    let live_ids: Vec<i64> = engine.iter().map(|row| row.id).collect();
    assert_eq!(
        live_ids,
        vec![11, 15, 23],
        "live set is exactly {{11, 15, 23}}: id 13 position-deleted (truncate=10), id 21 \
         equality-deleted (truncate=20)"
    );
    assert!(
        engine.iter().any(|row| row.id == 23),
        "id 23 (truncate=20, position 1) MUST survive — the truncate=10 position delete is \
         path-keyed, not broadcast across transform partitions"
    );
    assert!(
        !engine.iter().any(|row| row.id == 21),
        "id 21 must be equality-deleted via the TRANSFORMED (truncate=20) partition routing"
    );
}
