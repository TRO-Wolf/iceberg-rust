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

use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{Array, RecordBatch};
use async_trait::async_trait;
use bytes::Bytes;
use futures::TryStreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::page_prune_fixture::{field_id_map, file_metadata, selected_rows};
use crate::arrow::reader::ArrowReader;
use crate::expr::{Bind, Predicate, Reference};
use crate::io::{
    FileIO, FileIOBuilder, FileInfo, FileMetadata, FileRead, FileWrite, InputFile, LocalFsStorage,
    OutputFile, Storage, StorageConfig, StorageFactory,
};
use crate::spec::{Datum, TableMetadata};
use crate::table::Table;
use crate::{Result, TableIdent};

const FIXTURE_PREFIX: &str = "/iceberg-fixtures/page-prune";
const STRING_PREFIX: &str =
    "pppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp";

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/interop/page_prune")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PrefixStorage {
    root: String,
}

impl PrefixStorage {
    fn map(&self, path: &str) -> String {
        match path.strip_prefix(FIXTURE_PREFIX) {
            Some(rest) => format!("{}{rest}", self.root),
            None => path.to_string(),
        }
    }
}

#[async_trait]
#[typetag::serde]
impl Storage for PrefixStorage {
    async fn exists(&self, path: &str) -> Result<bool> {
        LocalFsStorage::new().exists(&self.map(path)).await
    }
    async fn metadata(&self, path: &str) -> Result<FileMetadata> {
        LocalFsStorage::new().metadata(&self.map(path)).await
    }
    async fn read(&self, path: &str) -> Result<Bytes> {
        LocalFsStorage::new().read(&self.map(path)).await
    }
    async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
        LocalFsStorage::new().reader(&self.map(path)).await
    }
    async fn write(&self, path: &str, bs: Bytes) -> Result<()> {
        LocalFsStorage::new().write(&self.map(path), bs).await
    }
    async fn writer(&self, path: &str) -> Result<Box<dyn FileWrite>> {
        LocalFsStorage::new().writer(&self.map(path)).await
    }
    async fn delete(&self, path: &str) -> Result<()> {
        LocalFsStorage::new().delete(&self.map(path)).await
    }
    async fn delete_prefix(&self, path: &str) -> Result<()> {
        LocalFsStorage::new().delete_prefix(&self.map(path)).await
    }
    async fn list(&self, prefix: &str) -> Result<Vec<FileInfo>> {
        LocalFsStorage::new().list(&self.map(prefix)).await
    }
    fn new_input(&self, path: &str) -> Result<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }
    fn new_output(&self, path: &str) -> Result<OutputFile> {
        Ok(OutputFile::new(Arc::new(self.clone()), path.to_string()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PrefixStorageFactory {
    root: String,
}

#[typetag::serde]
impl StorageFactory for PrefixStorageFactory {
    fn build(&self, _config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        Ok(Arc::new(PrefixStorage {
            root: self.root.clone(),
        }))
    }
}

pub(crate) fn fixture_io() -> FileIO {
    FileIOBuilder::new(Arc::new(PrefixStorageFactory {
        root: fixture_root().to_str().expect("utf8").to_string(),
    }))
    .build()
}

fn truth() -> Value {
    serde_json::from_str(
        &std::fs::read_to_string(fixture_root().join("page_prune_truth.json")).expect("truth"),
    )
    .expect("parse truth")
}

fn load_table(truth: &Value, name: &str) -> Table {
    let location = truth["tables"][name]["metadata_file"]
        .as_str()
        .expect("metadata_file");
    let rel = location
        .strip_prefix(FIXTURE_PREFIX)
        .expect("fixture prefix");
    let json = std::fs::read_to_string(fixture_root().join(rel.trim_start_matches('/')))
        .expect("metadata json");
    let metadata: TableMetadata = serde_json::from_str(&json).expect("parse metadata");
    Table::builder()
        .metadata(metadata)
        .metadata_location(location.to_string())
        .identifier(TableIdent::from_strs(["ns", name]).expect("ident"))
        .file_io(fixture_io())
        .build()
        .expect("table")
}

fn datum_string(s: &str) -> Datum {
    Datum::string(format!("{STRING_PREFIX}{s}"))
}

fn base_queries() -> Vec<(&'static str, Option<Predicate>)> {
    let id = || Reference::new("id");
    vec![
        (
            "id_range",
            Some(
                id().greater_than_or_equal_to(Datum::long(400))
                    .and(id().less_than_or_equal_to(Datum::long(520))),
            ),
        ),
        ("id_eq", Some(id().equal_to(Datum::long(1234)))),
        ("id_lt", Some(id().less_than(Datum::long(150)))),
        (
            "i_gt",
            Some(Reference::new("i").greater_than(Datum::int(1800))),
        ),
        (
            "s_eq",
            Some(Reference::new("s").equal_to(datum_string("001234"))),
        ),
        (
            "s_ge",
            Some(Reference::new("s").greater_than_or_equal_to(datum_string("001900"))),
        ),
        (
            "s_starts",
            Some(Reference::new("s").starts_with(datum_string("0019"))),
        ),
        (
            "s_not_starts",
            Some(Reference::new("s").not_starts_with(datum_string("0"))),
        ),
        ("n_is_null", Some(Reference::new("n").is_null())),
        ("n_not_null", Some(Reference::new("n").is_not_null())),
        ("n_eq", Some(Reference::new("n").equal_to(Datum::int(900)))),
        (
            "n_ne",
            Some(Reference::new("n").not_equal_to(Datum::int(900))),
        ),
        (
            "n_not_in",
            Some(Reference::new("n").is_not_in([Datum::int(900), Datum::int(901)])),
        ),
        ("d_isnan", Some(Reference::new("d").is_nan())),
        ("d_not_nan", Some(Reference::new("d").is_not_nan())),
        (
            "d_lt",
            Some(Reference::new("d").less_than(Datum::double(100.0))),
        ),
        (
            "d_gt",
            Some(Reference::new("d").greater_than(Datum::double(1000.0))),
        ),
        (
            "d_not_lt",
            Some(
                Reference::new("d")
                    .less_than(Datum::double(1000.0))
                    .negate(),
            ),
        ),
        (
            "f_gt",
            Some(Reference::new("f").greater_than(Datum::float(2000.0))),
        ),
        (
            "ts_range",
            Some(
                Reference::new("ts")
                    .greater_than_or_equal_to(Datum::timestamptz_micros(1_767_226_200_000_000))
                    .and(
                        Reference::new("ts")
                            .less_than(Datum::timestamptz_micros(1_767_226_320_000_000)),
                    ),
            ),
        ),
        ("_unfiltered", None),
    ]
}

fn evo_queries() -> Vec<(&'static str, Option<Predicate>)> {
    vec![
        (
            "i_promoted_gt",
            Some(Reference::new("i").greater_than(Datum::long(1800))),
        ),
        (
            "i_promoted_big",
            Some(Reference::new("i").greater_than(Datum::long(3_000_000_000i64))),
        ),
        (
            "f_promoted_gt",
            Some(Reference::new("f").greater_than(Datum::double(2000.0))),
        ),
        (
            "dec_promoted_gt",
            Some(
                Reference::new("dec")
                    .greater_than(Datum::decimal_from_str("1500.00").expect("decimal")),
            ),
        ),
        (
            "renamed_eq",
            Some(Reference::new("s2").equal_to(datum_string("001234"))),
        ),
        ("added_is_null", Some(Reference::new("addc").is_null())),
        (
            "added_eq",
            Some(Reference::new("addc").equal_to(Datum::int(7))),
        ),
        ("added_not_null", Some(Reference::new("addc").is_not_null())),
        ("readded_is_null", Some(Reference::new("n").is_null())),
        (
            "readded_eq",
            Some(Reference::new("n").equal_to(Datum::int(900))),
        ),
        ("_unfiltered", None),
    ]
}

fn truth_rows(truth: &Value, table: &str, query: &str) -> Vec<Vec<i64>> {
    truth["tables"][table]["answers"][query]["rows"]
        .as_array()
        .map(|rows| rows.as_slice())
        .unwrap_or(&[])
        .iter()
        .map(|row| {
            row.as_array()
                .expect("row")
                .iter()
                .map(|v| v.as_i64().expect("i64"))
                .collect()
        })
        .collect()
}

fn expected_rows(truth: &Value, table: &str, query: &str) -> Vec<Vec<i64>> {
    let mut rows = truth_rows(truth, table, query);
    if matches!(query, "n_ne" | "n_not_in") {
        rows.extend(truth_rows(truth, table, "n_is_null"));
    }
    rows.sort();
    rows
}

async fn scan_rows(
    table: &Table,
    columns: &[&str],
    predicate: Option<Predicate>,
    row_selection: bool,
) -> Vec<Vec<i64>> {
    let mut builder = table
        .scan()
        .select(columns.iter().copied())
        .with_row_selection_enabled(row_selection);
    if let Some(predicate) = predicate {
        builder = builder.with_filter(predicate);
    }
    let batches: Vec<RecordBatch> = builder
        .build()
        .expect("scan")
        .to_arrow()
        .await
        .expect("to_arrow")
        .try_collect()
        .await
        .expect("collect");
    let mut rows = Vec::new();
    for batch in &batches {
        for row in 0..batch.num_rows() {
            rows.push(
                (0..batch.num_columns())
                    .map(|col| {
                        let array = batch
                            .column(col)
                            .as_any()
                            .downcast_ref::<arrow_array::Int64Array>()
                            .expect("int64 column");
                        assert!(!array.is_null(row), "fixture columns are non-null");
                        array.value(row)
                    })
                    .collect(),
            );
        }
    }
    rows.sort();
    rows
}

async fn assert_table_queries(
    truth: &Value,
    table_name: &str,
    columns: &[&str],
    queries: Vec<(&'static str, Option<Predicate>)>,
) {
    let table = load_table(truth, table_name);
    for (query, predicate) in queries {
        let expected = expected_rows(truth, table_name, query);
        let on = scan_rows(&table, columns, predicate.clone(), true).await;
        assert_eq!(
            on.len(),
            expected.len(),
            "{table_name}.{query}: row-selection ON row count"
        );
        assert_eq!(on, expected, "{table_name}.{query}: ON rows != Spark");
        let off = scan_rows(&table, columns, predicate, false).await;
        assert_eq!(off, expected, "{table_name}.{query}: OFF rows != Spark");
    }
}

#[tokio::test]
async fn spark_base_v2_queries_match() {
    let truth = truth();
    assert_table_queries(&truth, "base_v2", &["id"], base_queries()).await;
}

#[tokio::test]
async fn spark_base_v3_queries_match_with_lineage() {
    let truth = truth();
    assert_table_queries(
        &truth,
        "base_v3",
        &["id", "_row_id", "_last_updated_sequence_number"],
        base_queries(),
    )
    .await;
}

#[tokio::test]
async fn spark_del_v2_queries_match_position_deletes() {
    let truth = truth();
    assert_table_queries(&truth, "del_v2", &["id"], base_queries()).await;
}

#[tokio::test]
async fn spark_del_v3_queries_match_dvs_and_lineage() {
    let truth = truth();
    assert_table_queries(
        &truth,
        "del_v3",
        &["id", "_row_id", "_last_updated_sequence_number"],
        base_queries(),
    )
    .await;
}

#[tokio::test]
async fn spark_evo_v2_queries_match_evolved_schema() {
    let truth = truth();
    assert_table_queries(&truth, "evo_v2", &["id"], evo_queries()).await;
}

async fn assert_kept_rows_subset(truth: &Value, table_name: &str, query: &str) {
    let table = load_table(truth, table_name);
    let predicate = base_queries()
        .into_iter()
        .find(|(name, _)| *name == query)
        .and_then(|(_, pred)| pred)
        .expect("query predicate");
    let schema = table.metadata().current_schema().clone();
    let bound = predicate.bind(schema.clone(), false).expect("bind");

    let tasks: Vec<_> = table
        .scan()
        .build()
        .expect("scan")
        .plan_files()
        .await
        .expect("plan")
        .try_collect()
        .await
        .expect("tasks");

    let expected = expected_rows(truth, table_name, query).len();
    let mut total_rows = 0usize;
    let mut kept_rows = 0usize;
    for task in &tasks {
        let local = task
            .data_file_path
            .strip_prefix(FIXTURE_PREFIX)
            .map(|rest| fixture_root().join(rest.trim_start_matches('/')))
            .expect("fixture path");
        let metadata = file_metadata(local.to_str().expect("utf8"));
        let map = field_id_map(&metadata);
        let selection = ArrowReader::get_row_selection_for_filter_predicate(
            &bound, &metadata, &None, &map, &schema,
        )
        .expect("selection")
        .expect("index present on all columns this predicate uses");
        total_rows += selection.iter().map(|s| s.row_count).sum::<usize>();
        kept_rows += selected_rows(&selection);
    }
    assert!(
        kept_rows >= expected,
        "{table_name}.{query}: kept {kept_rows} must cover {expected} matching rows"
    );
    assert!(
        kept_rows < total_rows,
        "{table_name}.{query}: kept {kept_rows} of {total_rows} — the scan did not prune"
    );
}

#[tokio::test]
async fn spark_base_v2_selective_predicates_skip_pages() {
    let truth = truth();
    for query in ["id_eq", "id_range", "ts_range", "i_gt", "f_gt"] {
        assert_kept_rows_subset(&truth, "base_v2", query).await;
    }
}

#[tokio::test]
async fn spark_base_v3_selective_predicates_skip_pages() {
    let truth = truth();
    for query in ["id_eq", "id_range", "ts_range"] {
        assert_kept_rows_subset(&truth, "base_v3", query).await;
    }
}

#[tokio::test]
async fn spark_base_v2_per_file_selection_prunes() {
    let truth = truth();
    let table = load_table(&truth, "base_v2");
    let predicate = base_queries()
        .into_iter()
        .find(|(name, _)| *name == "id_eq")
        .and_then(|(_, pred)| pred)
        .expect("predicate");
    let schema = table.metadata().current_schema().clone();
    let bound = predicate.bind(schema.clone(), false).expect("bind");

    let tasks: Vec<_> = table
        .scan()
        .build()
        .expect("scan")
        .plan_files()
        .await
        .expect("plan")
        .try_collect()
        .await
        .expect("tasks");
    let mut saw_skip = false;
    let mut saw_keep = false;
    for task in &tasks {
        let local = task
            .data_file_path
            .strip_prefix(FIXTURE_PREFIX)
            .map(|rest| fixture_root().join(rest.trim_start_matches('/')))
            .expect("fixture path");
        let metadata = file_metadata(local.to_str().expect("utf8"));
        let map = field_id_map(&metadata);
        let selection = ArrowReader::get_row_selection_for_filter_predicate(
            &bound, &metadata, &None, &map, &schema,
        )
        .expect("selection")
        .expect("index");
        for selector in selection.iter() {
            saw_skip |= selector.skip;
            saw_keep |= !selector.skip;
        }
    }
    assert!(saw_skip, "id_eq must skip pages on at least one file");
    assert!(saw_keep, "id_eq must keep pages on at least one file");
}
