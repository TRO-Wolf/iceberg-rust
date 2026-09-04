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

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use datafusion::arrow::array::{
    ArrayRef, DictionaryArray, Int32Array, Int64Array, RecordBatch, RunArray, StringArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Int32Type, Schema, SchemaRef};
use datafusion::common::ScalarValue;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, Column, Literal};
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::io::{
    FileInfo, FileMetadata, FileRead, FileWrite, InputFile, LocalFsStorageFactory, OutputFile,
    Storage, StorageConfig, StorageFactory,
};
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::metadata_columns::{RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_POS};
use iceberg::spec::{
    DataContentType, ManifestContentType, NestedField, PrimitiveType, Schema as IcebergSchema, Type,
};
use iceberg::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
use iceberg::{CatalogBuilder, NamespaceIdent, Result, TableCreation};
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

use super::delete_position_deletes::group_pairs_by_partition;
use super::{
    IsolationLevel, apply_assignments, decode_file_path, decode_file_paths_batch, decode_position,
    position_delete_unpartitioned_fast_path, sort_position_delete_pairs, *,
};

// An assignment must never smuggle a NULL into a REQUIRED column. A dictionary or REE array
// whose VALUES hold a NULL reports `null_count() == 0`, and `RecordBatch::try_new`'s own check
// is physical too, so the NULL passes both gates and is written.

/// A single-column table schema for `d`, `nullable` as given, dictionary-encoded Utf8.
fn dict_column_schema(nullable: bool) -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "d",
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        nullable,
    )]))
}

/// Dictionary array with a NULL in the VALUES: null-free keys, logically NULL at row 1.
fn dict_with_null_value() -> ArrayRef {
    let values = StringArray::from(vec![Some("x"), None]);
    let keys = Int32Array::from(vec![0, 1]);
    Arc::new(
        DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).expect("dictionary array"),
    )
}

#[test]
fn test_dictionary_encoded_null_cannot_be_assigned_to_a_required_column() {
    let column = dict_with_null_value();
    // The premise of the whole test: physically clean, logically NULL.
    assert_eq!(column.null_count(), 0, "physical null count must be 0");
    assert_eq!(column.logical_null_count(), 1, "row 1 is logically NULL");

    let schema = dict_column_schema(false);
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::clone(&column)]).expect("batch");
    let assignment: Arc<dyn PhysicalExpr> = Arc::new(Column::new("d", 0));

    let err = apply_assignments(&batch, &[(0, assignment)], &schema, None)
        .expect_err("a dictionary-encoded NULL must not reach a required column");
    assert!(
        err.to_string()
            .contains("UPDATE cannot assign NULL to required column 'd'"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_dictionary_encoded_null_is_fine_for_an_optional_column() {
    // The negative pin: the guard must reject only REQUIRED columns.
    let column = dict_with_null_value();
    let schema = dict_column_schema(true);
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::clone(&column)]).expect("batch");
    let assignment: Arc<dyn PhysicalExpr> = Arc::new(Column::new("d", 0));

    let out = apply_assignments(&batch, &[(0, assignment)], &schema, None)
        .expect("an optional column may take a NULL");
    assert_eq!(out.column(0).logical_null_count(), 1);
}

#[test]
fn test_null_free_assignment_to_a_required_column_still_succeeds() {
    // The other negative pin: `logical_null_count` must not reject clean data.
    let values = StringArray::from(vec![Some("x"), Some("y")]);
    let keys = Int32Array::from(vec![0, 1]);
    let column: ArrayRef = Arc::new(
        DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).expect("dictionary array"),
    );
    let schema = dict_column_schema(false);
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::clone(&column)]).expect("batch");
    let assignment: Arc<dyn PhysicalExpr> = Arc::new(Column::new("d", 0));

    let out = apply_assignments(&batch, &[(0, assignment)], &schema, None)
        .expect("a NULL-free assignment to a required column must succeed");
    assert_eq!(out.column(0).logical_null_count(), 0);
}

// Arrow's `value()` on a NULL slot returns a well-formed lie: `""` for a string, `0` for an i64.
// Both feed a position-delete tuple, so a NULL `_file` deletes against an empty path and a NULL
// `_pos` deletes ROW 0 of a real data file.

#[test]
fn test_decode_file_path_rejects_a_null_path() {
    let col: ArrayRef = Arc::new(StringArray::from(vec![Some("s3://b/a.parquet"), None]));
    assert!(
        decode_file_path(&col, 0).is_ok(),
        "the live row must still decode"
    );
    let err = decode_file_path(&col, 1).expect_err("a NULL _file must not decode to \"\"");
    assert!(err.to_string().contains("_file"), "unexpected error: {err}");
}

#[test]
fn test_decode_file_paths_batch_rejects_a_null_path() {
    let col: ArrayRef = Arc::new(StringArray::from(vec![Some("s3://b/a.parquet"), None]));
    let err = decode_file_paths_batch(&col).expect_err("a NULL _file must not decode to \"\"");
    assert!(err.to_string().contains("_file"), "unexpected error: {err}");
}

#[test]
fn test_decode_file_path_rejects_a_null_ree_value() {
    // The REE shape the COW scan actually produces, with a NULL in the run VALUES.
    let run_ends = Int32Array::from(vec![2, 4]);
    let values = StringArray::from(vec![Some("f/a.parquet"), None]);
    let ree = RunArray::<Int32Type>::try_new(&run_ends, &values).expect("build REE");
    let col: ArrayRef = Arc::new(ree);
    assert!(decode_file_path(&col, 0).is_ok(), "run 0 is live");
    let err = decode_file_path(&col, 3).expect_err("a NULL REE _file value must not decode");
    assert!(err.to_string().contains("_file"), "unexpected error: {err}");
    let err = decode_file_paths_batch(&col).expect_err("batch decode must reject it too");
    assert!(err.to_string().contains("_file"), "unexpected error: {err}");
}

#[test]
fn test_decode_position_rejects_a_null_position() {
    let col = Int64Array::from(vec![Some(7), None]);
    assert_eq!(
        decode_position(&col, 0).expect("the live row must decode"),
        7
    );
    let err = decode_position(&col, 1).expect_err("a NULL _pos must not decode to 0");
    assert!(err.to_string().contains("_pos"), "unexpected error: {err}");
}

/// `decode_file_paths_batch` must produce, for every row, EXACTLY the string `decode_file_path`
/// would: plain, run-end-encoded, and sliced REE. Byte-identical per-row results are the
/// correctness contract for COW affected-file detection and keep-masks.
fn assert_batch_matches_per_row(col: &ArrayRef) {
    let batch = decode_file_paths_batch(col).expect("batch decode");
    assert_eq!(batch.len(), col.len(), "one decoded path per row");
    for (row, decoded) in batch.iter().enumerate() {
        let per_row = decode_file_path(col, row).expect("per-row decode");
        assert_eq!(
            *decoded, per_row,
            "row {row}: batch decode must equal per-row decode"
        );
    }
}

#[test]
fn test_decode_file_paths_batch_plain_string_array() {
    let col: ArrayRef = Arc::new(StringArray::from(vec![
        "s3://b/a.parquet",
        "s3://b/a.parquet",
        "s3://b/c.parquet",
    ]));
    assert_batch_matches_per_row(&col);
}

#[test]
fn test_decode_file_paths_batch_ree_with_runs() {
    let run_ends = Int32Array::from(vec![3, 4, 6]);
    let values = StringArray::from(vec!["f/a.parquet", "f/b.parquet", "f/a.parquet"]);
    let ree = RunArray::<Int32Type>::try_new(&run_ends, &values).expect("build REE");
    let col: ArrayRef = Arc::new(ree);
    assert_eq!(col.len(), 6);
    assert_batch_matches_per_row(&col);
}

#[test]
fn test_decode_file_paths_batch_ree_single_run() {
    let run_ends = Int32Array::from(vec![5]);
    let values = StringArray::from(vec!["only/file.parquet"]);
    let ree = RunArray::<Int32Type>::try_new(&run_ends, &values).expect("build REE");
    let col: ArrayRef = Arc::new(ree);
    assert_batch_matches_per_row(&col);
}

#[test]
fn test_decode_file_paths_batch_sliced_ree_offset_fallback() {
    // offset != 0 exercises the `get_physical_index` fallback branch.
    let run_ends = Int32Array::from(vec![3, 4, 7]);
    let values = StringArray::from(vec!["f/a.parquet", "f/b.parquet", "f/c.parquet"]);
    let ree = RunArray::<Int32Type>::try_new(&run_ends, &values).expect("build REE");
    let sliced = ree.slice(2, 3);
    let col: ArrayRef = Arc::new(sliced);
    assert_eq!(col.len(), 3);
    assert_batch_matches_per_row(&col);
}

/// `sort_position_delete_pairs` MUST produce ascending `(file_path, pos)` order for ANY input.
/// The concurrent scan interleaves files, so an integration test cannot pin the spec order
/// deterministically.
///
/// MUTATION PROOF: make `sort_position_delete_pairs` a no-op (delete the `pairs.sort()`) and this
/// test goes RED, because the deliberately-unsorted input stays unsorted.
#[test]
fn test_sort_position_delete_pairs_orders_by_path_then_pos() {
    // Files interleaved, positions descending within a file: the shape a concurrent scan gives.
    let mut pairs: Vec<(String, i64)> = vec![
        ("s3://b/file_b.parquet".to_string(), 5),
        ("s3://b/file_a.parquet".to_string(), 2),
        ("s3://b/file_b.parquet".to_string(), 1),
        ("s3://b/file_a.parquet".to_string(), 0),
        ("s3://b/file_a.parquet".to_string(), 10),
    ];
    sort_position_delete_pairs(&mut pairs);
    let expected: Vec<(String, i64)> = vec![
        ("s3://b/file_a.parquet".to_string(), 0),
        ("s3://b/file_a.parquet".to_string(), 2),
        ("s3://b/file_a.parquet".to_string(), 10),
        ("s3://b/file_b.parquet".to_string(), 1),
        ("s3://b/file_b.parquet".to_string(), 5),
    ];
    assert_eq!(
        pairs, expected,
        "position-delete pairs must be sorted ascending by (file_path, pos) — spec order"
    );
    // Form-agnostic: catch any sort that is not a true ascending `(path, pos)` order.
    for window in pairs.windows(2) {
        assert!(
            window[0] <= window[1],
            "pairs must be non-decreasing by (file_path, pos): {:?} then {:?}",
            window[0],
            window[1]
        );
    }
}

/// Parse parity with Java `IsolationLevel.fromName`: case-insensitive accept, and a LOUD
/// `"Invalid isolation level: <name>"` on an unknown name, never a silent default.
///
/// MUTATION: make the parse default instead of erroring and this test goes RED.
#[test]
fn test_isolation_level_parse_java_parity() {
    for accepted in ["serializable", "SERIALIZABLE", "Serializable"] {
        assert_eq!(
            IsolationLevel::parse(accepted).expect("parse serializable spelling"),
            IsolationLevel::Serializable,
            "'{accepted}' must parse as serializable"
        );
    }
    for accepted in ["snapshot", "SNAPSHOT", "Snapshot"] {
        assert_eq!(
            IsolationLevel::parse(accepted).expect("parse snapshot spelling"),
            IsolationLevel::Snapshot,
            "'{accepted}' must parse as snapshot"
        );
    }

    // An unknown name fails loud, carrying Java's message shape and the offending name.
    let err = IsolationLevel::parse("read-committed")
        .expect_err("an unknown isolation level must fail loud, not default");
    assert!(
        err.to_string()
            .contains("Invalid isolation level: read-committed"),
        "error must carry Java's message + the offending name, got: {err}"
    );
    // Java cannot disable row-level validation, so 'none' is not a row-level isolation level.
    assert!(
        IsolationLevel::parse("none").is_err(),
        "'none' must be rejected for row-level operations"
    );
}

// BUG-001 — the unpartitioned fast-path predicate (mutation-proven).

#[test]
fn test_pos_delete_fast_path_only_for_single_empty_spec() {
    // A never-evolved empty partition type.
    assert!(position_delete_unpartitioned_fast_path(1, 0));
    // Partitioned or all-Void: always walk the manifests.
    assert!(!position_delete_unpartitioned_fast_path(1, 1));
    // Evolved: multi-spec with an empty default MUST NOT fast-path.
    assert!(
        !position_delete_unpartitioned_fast_path(2, 0),
        "BUG-001: multi-spec with empty default must take the manifest walk"
    );
    assert!(!position_delete_unpartitioned_fast_path(2, 1));
    // Zero specs is not a real table shape; refuse the fast path.
    assert!(!position_delete_unpartitioned_fast_path(0, 0));
}

/// Mutation twin: weakening the rule to "the default is empty" alone fails this assert.
#[test]
fn test_pos_delete_fast_path_mutation_field_count_only_is_wrong() {
    let evolved_empty_default = position_delete_unpartitioned_fast_path(2, 0);
    assert!(
        !evolved_empty_default,
        "mutation RED: field_count-only condition would take the fast path here"
    );
}

/// C1-L-002: an all-Void spec is unpartitioned but has fields, so it must NOT fast-path.
#[test]
fn test_pos_delete_fast_path_rejects_all_void_single_spec() {
    // One void field.
    assert!(
        !position_delete_unpartitioned_fast_path(1, 1),
        "all-Void needs a null-tuple PartitionKey, not the empty fast path"
    );
}

// The grouping resolves every pair's real partition, instead of fabricating an empty tuple.

/// `path → (spec_id, partition)` for two files of a one-field partitioned spec.
fn partition_map() -> std::collections::HashMap<String, (i32, iceberg::spec::Struct)> {
    use iceberg::spec::{Literal, Struct};

    let mut map = std::collections::HashMap::new();
    map.insert(
        "s3://b/x0.parquet".to_string(),
        (1, Struct::from_iter([Some(Literal::long(0))])),
    );
    map.insert(
        "s3://b/x1.parquet".to_string(),
        (1, Struct::from_iter([Some(Literal::long(1))])),
    );
    map
}

/// The normal path: pairs are grouped by their data file's own `(spec_id, partition)`, so each
/// delete file is stamped with the spec + partition of the file it deletes from.
#[test]
fn test_group_pairs_by_partition_groups_by_the_target_files_partition() {
    let map = partition_map();
    let pairs = vec![
        ("s3://b/x0.parquet".to_string(), 3),
        ("s3://b/x1.parquet".to_string(), 7),
        ("s3://b/x0.parquet".to_string(), 1),
    ];

    let groups = group_pairs_by_partition(&pairs, &map).expect("every pair resolves");
    assert_eq!(
        groups.len(),
        2,
        "one group per distinct partition: {groups:?}"
    );
    let x0 = groups
        .get(&map["s3://b/x0.parquet"])
        .expect("the x=0 group must exist");
    assert_eq!(x0.len(), 2, "both x=0 pairs land in the same group");
    assert_eq!(
        groups
            .get(&map["s3://b/x1.parquet"])
            .expect("the x=1 group must exist")
            .len(),
        1
    );
}

/// A pair whose data file is not live in the map's snapshot must FAIL. The old fallback paired
/// a partitioned spec with an empty tuple, writing a delete file under a `field=null` path that
/// no reader matches — a silent under-delete, so the rows come back.
///
/// MUTATION: restore the `unwrap_or_else(|| (default_spec.spec_id(), Struct::empty()))` fallback
/// and this test goes RED.
#[test]
fn test_group_pairs_by_partition_rejects_an_unmatched_data_file() {
    let map = partition_map();
    let pairs = vec![
        ("s3://b/x0.parquet".to_string(), 3),
        ("s3://b/ghost.parquet".to_string(), 0),
    ];

    let err = group_pairs_by_partition(&pairs, &map)
        .expect_err("an unresolvable data file must fail loudly");
    assert!(
        err.to_string().contains("s3://b/ghost.parquet")
            && err.to_string().contains("is not a live file"),
        "the error must name the offending file: {err}"
    );
}
fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn is_snapshot_list(path: &str) -> bool {
    let name = file_name(path);
    name.starts_with("snap-") && name.ends_with(".avro")
}

fn is_manifest(path: &str) -> bool {
    let name = file_name(path);
    name.ends_with(".avro") && !name.starts_with("snap-")
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CountingStorage {
    #[serde(skip)]
    inner: Option<Arc<dyn Storage>>,
    #[serde(skip)]
    manifest_reads: Arc<AtomicU64>,
    #[serde(skip)]
    data_manifest_paths: Arc<Mutex<HashSet<String>>>,
    #[serde(skip)]
    data_manifest_reads: Arc<AtomicU64>,
    #[serde(skip)]
    opens: Arc<AtomicU64>,
}

impl CountingStorage {
    fn inner(&self) -> &Arc<dyn Storage> {
        self.inner.as_ref().expect("counting storage was built")
    }

    fn count(&self, path: &str) {
        if path.ends_with(".parquet") {
            self.opens.fetch_add(1, Ordering::Relaxed);
        }
        if is_snapshot_list(path) {
            return;
        }
        if is_manifest(path) {
            self.manifest_reads.fetch_add(1, Ordering::Relaxed);
            if self
                .data_manifest_paths
                .lock()
                .expect("data-manifest path set")
                .contains(path)
            {
                self.data_manifest_reads.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[async_trait]
#[typetag::serde]
impl Storage for CountingStorage {
    async fn exists(&self, path: &str) -> Result<bool> {
        self.inner().exists(path).await
    }

    async fn metadata(&self, path: &str) -> Result<FileMetadata> {
        self.inner().metadata(path).await
    }

    async fn read(&self, path: &str) -> Result<Bytes> {
        self.count(path);
        self.inner().read(path).await
    }

    async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
        self.count(path);
        self.inner().reader(path).await
    }

    async fn write(&self, path: &str, bs: Bytes) -> Result<()> {
        self.inner().write(path, bs).await
    }

    async fn writer(&self, path: &str) -> Result<Box<dyn FileWrite>> {
        self.inner().writer(path).await
    }

    async fn delete(&self, path: &str) -> Result<()> {
        self.inner().delete(path).await
    }

    async fn delete_prefix(&self, path: &str) -> Result<()> {
        self.inner().delete_prefix(path).await
    }

    async fn list(&self, prefix: &str) -> Result<Vec<FileInfo>> {
        self.inner().list(prefix).await
    }

    fn new_input(&self, path: &str) -> Result<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    fn new_output(&self, path: &str) -> Result<OutputFile> {
        self.inner().new_output(path)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CountingStorageFactory {
    #[serde(skip)]
    manifest_reads: Arc<AtomicU64>,
    #[serde(skip)]
    data_manifest_paths: Arc<Mutex<HashSet<String>>>,
    #[serde(skip)]
    data_manifest_reads: Arc<AtomicU64>,
    #[serde(skip)]
    opens: Arc<AtomicU64>,
}

#[typetag::serde]
impl StorageFactory for CountingStorageFactory {
    fn build(&self, config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        Ok(Arc::new(CountingStorage {
            inner: Some(LocalFsStorageFactory.build(config)?),
            manifest_reads: self.manifest_reads.clone(),
            data_manifest_paths: self.data_manifest_paths.clone(),
            data_manifest_reads: self.data_manifest_reads.clone(),
            opens: self.opens.clone(),
        }))
    }
}

struct MorCloseFixture {
    catalog: MemoryCatalog,
    table: Table,
    paths: Vec<String>,
    factory: Arc<CountingStorageFactory>,
    _warehouse: TempDir,
}

async fn mor_close_fixture() -> MorCloseFixture {
    mor_close_fixture_at(48).await
}

async fn mor_close_fixture_at(n: usize) -> MorCloseFixture {
    let warehouse = TempDir::new().expect("warehouse");
    let factory = Arc::new(CountingStorageFactory {
        data_manifest_paths: Arc::new(Mutex::new(HashSet::new())),
        data_manifest_reads: Arc::new(AtomicU64::new(0)),
        ..Default::default()
    });
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(factory.clone())
        .load(
            "memory",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                warehouse.path().to_str().expect("utf8").to_string(),
            )]),
        )
        .await
        .expect("catalog");
    let namespace = NamespaceIdent::new("ns".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let schema = IcebergSchema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "val", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("schema");
    let mut properties = HashMap::new();
    properties.insert("write.delete.mode".to_string(), "merge-on-read".to_string());
    properties.insert("write.update.mode".to_string(), "merge-on-read".to_string());
    let creation = TableCreation::builder()
        .name("t".to_string())
        .location(format!("{}/mor", warehouse.path().to_str().expect("utf8")))
        .schema(schema)
        .format_version(FormatVersion::V3)
        .properties(properties)
        .build();
    let mut table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("table");
    let arrow_schema =
        Arc::new(schema_to_arrow_schema(table.metadata().current_schema()).expect("arrow schema"));
    let mut paths = Vec::with_capacity(n);
    for index in 0..i32::try_from(n).expect("n fits") {
        let file_path = format!("{}/data/f{index}.parquet", table.metadata().location());
        let batch = RecordBatch::try_new(Arc::clone(&arrow_schema), vec![
            Arc::new(Int32Array::from(vec![index])) as ArrayRef,
            Arc::new(StringArray::from(vec![format!("v{index}")])) as ArrayRef,
        ])
        .expect("batch");
        let output = table
            .file_io()
            .new_output(file_path.clone())
            .expect("output");
        let parquet_builder = ParquetWriterBuilder::new(
            WriterProperties::builder().build(),
            table.metadata().current_schema().clone(),
        );
        let mut writer = parquet_builder.build(output).await.expect("writer");
        writer.write(&batch).await.expect("write");
        let mut file_builder = writer
            .close()
            .await
            .expect("close")
            .into_iter()
            .next()
            .expect("file");
        file_builder
            .content(DataContentType::Data)
            .partition_spec_id(0)
            .partition(Struct::empty());
        let file = file_builder.build().expect("data file");
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![file])
            .apply(tx)
            .expect("append");
        table = tx.commit(&catalog).await.expect("commit");
        paths.push(file_path);
    }
    let snapshot = table.metadata().current_snapshot().expect("snapshot");
    let list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    let data_paths: HashSet<String> = list
        .entries()
        .iter()
        .filter(|manifest| manifest.content == ManifestContentType::Data)
        .map(|manifest| manifest.manifest_path.clone())
        .collect();
    assert_eq!(data_paths.len(), n, "one data manifest per append");
    *factory
        .data_manifest_paths
        .lock()
        .expect("data-manifest path set") = data_paths;
    factory.data_manifest_reads.store(0, Ordering::Relaxed);
    factory.manifest_reads.store(0, Ordering::Relaxed);
    factory.opens.store(0, Ordering::Relaxed);
    MorCloseFixture {
        catalog,
        table,
        paths,
        factory,
        _warehouse: warehouse,
    }
}

fn oldest_eq_predicate() -> Arc<dyn PhysicalExpr> {
    Arc::new(BinaryExpr::new(
        Arc::new(Column::new("id", 0)),
        Operator::Eq,
        Arc::new(Literal::new(ScalarValue::Int32(Some(0)))),
    ))
}

async fn scan_oldest_pair(
    table: &Table,
    projection: Vec<String>,
    table_schema: &SchemaRef,
    scan_snapshot_id: Option<i64>,
) -> (Vec<(String, i64)>, HashMap<String, (i32, Struct)>) {
    let predicate = oldest_eq_predicate();
    let (mut stream, shared) = mor_scan::mor_scan_stream(table, projection, None, scan_snapshot_id)
        .await
        .expect("scan");
    let mut pairs: Vec<(String, i64)> = Vec::new();
    while let Some(batch) = stream.try_next().await.expect("batch") {
        let table_batch = table_column_batch(&batch, table_schema).expect("columns");
        let mask = match_mask(&Some(Arc::clone(&predicate)), &table_batch).expect("mask");
        let file_col = batch
            .column_by_name(RESERVED_COL_NAME_FILE)
            .expect("file column");
        let pos_col = batch
            .column_by_name(RESERVED_COL_NAME_POS)
            .expect("pos column")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("pos type");
        for row in 0..batch.num_rows() {
            if mask.value(row) {
                pairs.push((
                    decode_file_path(file_col, row).expect("path"),
                    decode_position(pos_col, row).expect("pos"),
                ));
            }
        }
    }
    sort_position_delete_pairs(&mut pairs);
    let known = shared.lock().expect("partition map").clone();
    (pairs, known)
}

#[tokio::test]
async fn mor_delete_close_with_complete_partitions_reads_no_data_manifest() {
    let fixture = mor_close_fixture().await;
    let table = &fixture.table;
    let oldest = fixture.paths.first().expect("paths").clone();
    let table_schema =
        Arc::new(schema_to_arrow_schema(table.metadata().current_schema()).expect("arrow schema"));
    let mut projection: Vec<String> = table_schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    projection.push(RESERVED_COL_NAME_FILE.to_string());
    projection.push(RESERVED_COL_NAME_POS.to_string());
    let scan_snapshot_id = table
        .metadata()
        .current_snapshot()
        .map(|snapshot| snapshot.snapshot_id());
    let (pairs, known) = scan_oldest_pair(table, projection, &table_schema, scan_snapshot_id).await;
    assert_eq!(pairs, vec![(oldest.clone(), 0)]);
    assert_eq!(known.len(), 48);
    assert_eq!(known.get(&oldest), Some(&(0, Struct::empty())));
    fixture
        .factory
        .data_manifest_reads
        .store(0, Ordering::Relaxed);
    let close = write_merge_on_read_deletes(
        table,
        MergeOnReadDeleteKind::DeletionVectors,
        &pairs,
        &known,
        scan_snapshot_id,
    )
    .await
    .expect("close");
    assert_eq!(close.added.len(), 1);
    assert_eq!(close.added[0].partition_spec_id(), 0);
    assert_eq!(close.added[0].partition(), &Struct::empty());
    assert_eq!(
        close
            .referenced_data_files()
            .into_iter()
            .collect::<Vec<_>>(),
        vec![oldest]
    );
    assert_eq!(
        fixture.factory.data_manifest_reads.load(Ordering::Relaxed),
        0,
        "a complete known_partitions map must skip the data-manifest walk"
    );
}

#[tokio::test]
async fn mor_update_close_with_complete_partitions_reads_no_data_manifest() {
    let fixture = mor_close_fixture().await;
    let table = &fixture.table;
    let oldest = fixture.paths.first().expect("paths").clone();
    let table_schema =
        Arc::new(schema_to_arrow_schema(table.metadata().current_schema()).expect("arrow schema"));
    let mut projection: Vec<String> = table_schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    projection.push(RESERVED_COL_NAME_FILE.to_string());
    projection.push(RESERVED_COL_NAME_POS.to_string());
    push_lineage_scan_columns(&mut projection, table.metadata().format_version());
    let scan_snapshot_id = table
        .metadata()
        .current_snapshot()
        .map(|snapshot| snapshot.snapshot_id());
    let (pairs, known) = scan_oldest_pair(table, projection, &table_schema, scan_snapshot_id).await;
    assert_eq!(pairs, vec![(oldest.clone(), 0)]);
    assert_eq!(known.len(), 48);
    fixture
        .factory
        .data_manifest_reads
        .store(0, Ordering::Relaxed);
    let close = write_merge_on_read_deletes(
        table,
        MergeOnReadDeleteKind::DeletionVectors,
        &pairs,
        &known,
        scan_snapshot_id,
    )
    .await
    .expect("close");
    assert_eq!(close.added.len(), 1);
    assert_eq!(
        fixture.factory.data_manifest_reads.load(Ordering::Relaxed),
        0,
        "a complete known_partitions map must skip the data-manifest walk"
    );
}

#[tokio::test]
async fn mor_delete_close_with_partial_partitions_still_walks_and_matches() {
    let fixture = mor_close_fixture().await;
    let table = &fixture.table;
    let oldest = fixture.paths.first().expect("paths").clone();
    let table_schema =
        Arc::new(schema_to_arrow_schema(table.metadata().current_schema()).expect("arrow schema"));
    let mut projection: Vec<String> = table_schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    projection.push(RESERVED_COL_NAME_FILE.to_string());
    projection.push(RESERVED_COL_NAME_POS.to_string());
    let scan_snapshot_id = table
        .metadata()
        .current_snapshot()
        .map(|snapshot| snapshot.snapshot_id());
    let (pairs, known) = scan_oldest_pair(table, projection, &table_schema, scan_snapshot_id).await;
    assert_eq!(pairs, vec![(oldest.clone(), 0)]);
    let mut partial = known.clone();
    partial.remove(&oldest);
    assert_eq!(partial.len(), 47);
    fixture
        .factory
        .data_manifest_reads
        .store(0, Ordering::Relaxed);
    let close = write_merge_on_read_deletes(
        table,
        MergeOnReadDeleteKind::DeletionVectors,
        &pairs,
        &partial,
        scan_snapshot_id,
    )
    .await
    .expect("close");
    assert_eq!(close.added.len(), 1);
    assert_eq!(
        close
            .referenced_data_files()
            .into_iter()
            .collect::<Vec<_>>(),
        vec![oldest]
    );
    assert_eq!(
        fixture.factory.data_manifest_reads.load(Ordering::Relaxed),
        48,
        "a touched file the map misses still walks every data manifest"
    );
}

#[tokio::test]
async fn mor_delete_threads_scan_partitions_to_the_close() {
    let fixture = mor_close_fixture().await;
    let table = &fixture.table;
    let table_schema =
        Arc::new(schema_to_arrow_schema(table.metadata().current_schema()).expect("arrow schema"));
    let (deleted, close) = merge_on_read_delete(
        table,
        &fixture.catalog,
        Some(oldest_eq_predicate()),
        None,
        &table_schema,
        IsolationLevel::Serializable,
        None,
    )
    .await
    .expect("delete");
    assert_eq!(deleted, 1);
    assert_eq!(close.added.len(), 1);
    assert!(
        close.data_sequence_numbers.is_empty(),
        "threaded partitions must skip the data-manifest walk"
    );
}

#[tokio::test]
async fn mor_update_threads_scan_partitions_to_the_close() {
    let fixture = mor_close_fixture().await;
    let table = &fixture.table;
    let table_schema =
        Arc::new(schema_to_arrow_schema(table.metadata().current_schema()).expect("arrow schema"));
    let assignments: Vec<(usize, Arc<dyn PhysicalExpr>)> = vec![(
        1,
        Arc::new(Literal::new(ScalarValue::Utf8(Some("w".to_string())))),
    )];
    let (updated, close) = merge_on_read_update(
        table,
        &fixture.catalog,
        Some(oldest_eq_predicate()),
        None,
        &assignments,
        &table_schema,
        IsolationLevel::Serializable,
        None,
    )
    .await
    .expect("update");
    assert_eq!(updated, 1);
    assert_eq!(close.added.len(), 1);
    assert!(
        close.data_sequence_numbers.is_empty(),
        "threaded partitions must skip the data-manifest walk"
    );
}

#[tokio::test]
#[ignore = "measurement, not a CI pin"]
async fn measure_mor_delete_close_at_8_48_192() {
    for n in [8usize, 48, 192] {
        let fixture = mor_close_fixture_at(n).await;
        let table = &fixture.table;
        let table_schema = Arc::new(
            schema_to_arrow_schema(table.metadata().current_schema()).expect("arrow schema"),
        );
        let mut projection: Vec<String> = table_schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        projection.push(RESERVED_COL_NAME_FILE.to_string());
        projection.push(RESERVED_COL_NAME_POS.to_string());
        let scan_snapshot_id = table
            .metadata()
            .current_snapshot()
            .map(|snapshot| snapshot.snapshot_id());
        let (pairs, known) =
            scan_oldest_pair(table, projection, &table_schema, scan_snapshot_id).await;
        assert_eq!(pairs.len(), 1);
        assert_eq!(known.len(), n);
        let runs = if n == 48 { 5 } else { 3 };
        for run in 1..=runs {
            fixture
                .factory
                .data_manifest_reads
                .store(0, Ordering::Relaxed);
            fixture.factory.opens.store(0, Ordering::Relaxed);
            let start = std::time::Instant::now();
            let before = write_merge_on_read_deletes(
                table,
                MergeOnReadDeleteKind::DeletionVectors,
                &pairs,
                &HashMap::new(),
                scan_snapshot_id,
            )
            .await
            .expect("close");
            let before_wall = start.elapsed();
            let before_reads = fixture.factory.data_manifest_reads.load(Ordering::Relaxed);
            let before_opens = fixture.factory.opens.load(Ordering::Relaxed);
            fixture
                .factory
                .data_manifest_reads
                .store(0, Ordering::Relaxed);
            fixture.factory.opens.store(0, Ordering::Relaxed);
            let start = std::time::Instant::now();
            let after = write_merge_on_read_deletes(
                table,
                MergeOnReadDeleteKind::DeletionVectors,
                &pairs,
                &known,
                scan_snapshot_id,
            )
            .await
            .expect("close");
            let after_wall = start.elapsed();
            let after_reads = fixture.factory.data_manifest_reads.load(Ordering::Relaxed);
            let after_opens = fixture.factory.opens.load(Ordering::Relaxed);
            println!(
                "F-26 n={n} run={run} before_reads={before_reads} before_opens={before_opens} before_wall={before_wall:?} after_reads={after_reads} after_opens={after_opens} after_wall={after_wall:?} added={}",
                after.added.len() + before.added.len(),
            );
            assert_eq!(before.added.len(), 1);
            assert_eq!(after.added.len(), 1);
        }
    }
}
