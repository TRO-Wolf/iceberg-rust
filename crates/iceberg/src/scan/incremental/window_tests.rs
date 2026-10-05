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

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;

use futures::TryStreamExt;

use super::IncrementalAppendScanBuilder;
use super::window::{NonAppendPolicy, appends_between};
use crate::memory::tests::new_memory_catalog;
use crate::scan::FileScanTask;
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, Literal, Operation,
    Struct, TableMetadata,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{Catalog, Error, ErrorKind, TableCreation, TableIdent};

async fn make_minimal_table(catalog: &impl Catalog) -> Table {
    let table_ident =
        TableIdent::from_strs([format!("ns-{}", uuid::Uuid::new_v4()), "t".to_string()]).unwrap();
    catalog
        .create_namespace(table_ident.namespace(), HashMap::new())
        .await
        .unwrap();
    let file = File::open(format!(
        "{}/testdata/table_metadata/TableMetadataV3ValidMinimal.json",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let base_metadata = serde_json::from_reader::<_, TableMetadata>(BufReader::new(file)).unwrap();
    let table_creation = TableCreation::builder()
        .schema((**base_metadata.current_schema()).clone())
        .partition_spec((**base_metadata.default_partition_spec()).clone())
        .sort_order((**base_metadata.default_sort_order()).clone())
        .name(table_ident.name().to_string())
        .format_version(FormatVersion::V3)
        .build();
    catalog
        .create_table(table_ident.namespace(), table_creation)
        .await
        .unwrap()
}

fn data_file(path: &str) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::long(1))]))
        .build()
        .unwrap()
}

struct Lineage {
    table: Table,
    s0: i64,
    s1_append: i64,
    s2_overwrite: i64,
    s3_append: i64,
    s4_delete: i64,
    s5_replace: i64,
    s6_append: i64,
}

async fn commit_and_check(
    catalog: &impl Catalog,
    tx: Transaction,
    expected: Operation,
) -> (Table, i64) {
    let table = tx.commit(catalog).await.unwrap();
    let snapshot = table.metadata().current_snapshot().unwrap();
    assert_eq!(snapshot.summary().operation, expected);
    let snapshot_id = snapshot.snapshot_id();
    (table, snapshot_id)
}

async fn append(catalog: &impl Catalog, table: &Table, path: &str) -> (Table, i64) {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![data_file(path)])
        .apply(tx)
        .unwrap();
    commit_and_check(catalog, tx, Operation::Append).await
}

async fn lineage(catalog: &impl Catalog) -> Lineage {
    let table = make_minimal_table(catalog).await;
    let (table, s0) = append(catalog, &table, "s0.parquet").await;
    let (table, s1_append) = append(catalog, &table, "s1.parquet").await;

    let tx = Transaction::new(&table);
    let tx = tx
        .overwrite_files()
        .delete_file("s0.parquet")
        .add_file(data_file("s2.parquet"))
        .apply(tx)
        .unwrap();
    let (table, s2_overwrite) = commit_and_check(catalog, tx, Operation::Overwrite).await;

    let (table, s3_append) = append(catalog, &table, "s3.parquet").await;

    let tx = Transaction::new(&table);
    let tx = tx
        .overwrite_files()
        .delete_file("s1.parquet")
        .apply(tx)
        .unwrap();
    let (table, s4_delete) = commit_and_check(catalog, tx, Operation::Delete).await;

    let tx = Transaction::new(&table);
    let tx = tx
        .rewrite_files(vec![data_file("s2.parquet")], vec![data_file("s5.parquet")])
        .apply(tx)
        .unwrap();
    let (table, s5_replace) = commit_and_check(catalog, tx, Operation::Replace).await;

    let (table, s6_append) = append(catalog, &table, "s6.parquet").await;

    Lineage {
        table,
        s0,
        s1_append,
        s2_overwrite,
        s3_append,
        s4_delete,
        s5_replace,
        s6_append,
    }
}

fn window(
    lineage: &Lineage,
    from_exclusive: i64,
    to: i64,
    configure: impl FnOnce(IncrementalAppendScanBuilder<'_>) -> IncrementalAppendScanBuilder<'_>,
) -> super::IncrementalAppendScan {
    configure(
        lineage
            .table
            .incremental_append_scan()
            .from_snapshot_id_exclusive(from_exclusive)
            .to_snapshot_id(to),
    )
    .build()
    .unwrap()
}

async fn planned(scan: &super::IncrementalAppendScan) -> Result<HashSet<String>, Error> {
    let tasks: Vec<FileScanTask> = scan.plan_files().await?.try_collect().await?;
    Ok(tasks
        .into_iter()
        .map(|task| task.data_file_path.to_string())
        .collect())
}

fn paths(names: &[&str]) -> HashSet<String> {
    names.iter().map(|name| name.to_string()).collect()
}

fn assert_refused(error: Error, snapshot_id: i64, operation: &str, from: i64, to: i64) {
    assert_eq!(error.kind(), ErrorKind::PreconditionFailed, "{error}");
    let message = error.message().to_string();
    assert!(message.contains(&snapshot_id.to_string()), "{message}");
    assert!(message.contains(operation), "{message}");
    assert!(message.contains(&format!("({from}, {to}]")), "{message}");
}

#[tokio::test]
async fn fail_on_non_append_refuses_an_overwrite_between_appends() {
    let catalog = new_memory_catalog().await;
    let lineage = lineage(&catalog).await;
    let (from, to) = (lineage.s0, lineage.s3_append);

    let scan = window(&lineage, from, to, |b| b.with_fail_on_non_append(true));
    let error = planned(&scan).await.unwrap_err();
    assert_refused(error, lineage.s2_overwrite, "overwrite", from, to);
}

#[tokio::test]
async fn fail_on_non_append_refuses_a_delete_between_appends() {
    let catalog = new_memory_catalog().await;
    let lineage = lineage(&catalog).await;
    let (from, to) = (lineage.s2_overwrite, lineage.s6_append);

    let scan = window(&lineage, from, to, |b| b.with_fail_on_non_append(true));
    let error = planned(&scan).await.unwrap_err();
    assert_refused(error, lineage.s4_delete, "delete", from, to);
}

#[tokio::test]
async fn fail_on_non_append_skips_a_replace_silently_as_spark_does() {
    let catalog = new_memory_catalog().await;
    let lineage = lineage(&catalog).await;

    let scan = window(&lineage, lineage.s4_delete, lineage.s6_append, |b| {
        b.with_fail_on_non_append(true)
    });
    assert_eq!(planned(&scan).await.unwrap(), paths(&["s6.parquet"]));
}

#[tokio::test]
async fn fail_on_non_append_names_the_oldest_refused_snapshot_first() {
    let catalog = new_memory_catalog().await;
    let lineage = lineage(&catalog).await;
    let (from, to) = (lineage.s0, lineage.s6_append);

    let scan = window(&lineage, from, to, |b| b.with_fail_on_non_append(true));
    let error = planned(&scan).await.unwrap_err();
    assert_refused(error, lineage.s2_overwrite, "overwrite", from, to);
}

#[tokio::test]
async fn skip_overwrite_skips_only_overwrites() {
    let catalog = new_memory_catalog().await;
    let lineage = lineage(&catalog).await;

    let scan = window(&lineage, lineage.s0, lineage.s3_append, |b| {
        b.with_fail_on_non_append(true)
            .with_skip_overwrite_snapshots(true)
    });
    assert_eq!(
        planned(&scan).await.unwrap(),
        paths(&["s1.parquet", "s3.parquet"])
    );

    let (from, to) = (lineage.s0, lineage.s6_append);
    let scan = window(&lineage, from, to, |b| {
        b.with_fail_on_non_append(true)
            .with_skip_overwrite_snapshots(true)
    });
    let error = planned(&scan).await.unwrap_err();
    assert_refused(error, lineage.s4_delete, "delete", from, to);
}

#[tokio::test]
async fn skip_delete_skips_only_deletes() {
    let catalog = new_memory_catalog().await;
    let lineage = lineage(&catalog).await;

    let scan = window(&lineage, lineage.s2_overwrite, lineage.s6_append, |b| {
        b.with_fail_on_non_append(true)
            .with_skip_delete_snapshots(true)
    });
    assert_eq!(
        planned(&scan).await.unwrap(),
        paths(&["s3.parquet", "s6.parquet"])
    );

    let (from, to) = (lineage.s0, lineage.s6_append);
    let scan = window(&lineage, from, to, |b| {
        b.with_fail_on_non_append(true)
            .with_skip_delete_snapshots(true)
    });
    let error = planned(&scan).await.unwrap_err();
    assert_refused(error, lineage.s2_overwrite, "overwrite", from, to);
}

#[tokio::test]
async fn both_skips_yield_every_append_in_the_window() {
    let catalog = new_memory_catalog().await;
    let lineage = lineage(&catalog).await;

    let scan = window(&lineage, lineage.s0, lineage.s6_append, |b| {
        b.with_fail_on_non_append(true)
            .with_skip_overwrite_snapshots(true)
            .with_skip_delete_snapshots(true)
    });
    assert_eq!(
        planned(&scan).await.unwrap(),
        paths(&["s1.parquet", "s3.parquet", "s6.parquet"])
    );
}

#[tokio::test]
async fn option_off_keeps_todays_silent_skip() {
    let catalog = new_memory_catalog().await;
    let lineage = lineage(&catalog).await;
    let expected = paths(&["s1.parquet", "s3.parquet", "s6.parquet"]);

    let default = window(&lineage, lineage.s0, lineage.s6_append, |b| b);
    assert_eq!(planned(&default).await.unwrap(), expected);

    let explicit_off = window(&lineage, lineage.s0, lineage.s6_append, |b| {
        b.with_fail_on_non_append(false)
            .with_skip_overwrite_snapshots(true)
            .with_skip_delete_snapshots(false)
    });
    assert_eq!(planned(&explicit_off).await.unwrap(), expected);
}

#[tokio::test]
async fn fail_on_non_append_keeps_the_exclusive_from_and_inclusive_to_bounds() {
    let catalog = new_memory_catalog().await;
    let lineage = lineage(&catalog).await;

    let from_overwrite_exclusive = window(&lineage, lineage.s2_overwrite, lineage.s3_append, |b| {
        b.with_fail_on_non_append(true)
    });
    assert_eq!(
        planned(&from_overwrite_exclusive).await.unwrap(),
        paths(&["s3.parquet"])
    );

    let inclusive = lineage
        .table
        .incremental_append_scan()
        .from_snapshot_id_inclusive(lineage.s2_overwrite)
        .to_snapshot_id(lineage.s3_append)
        .with_fail_on_non_append(true)
        .build()
        .unwrap();
    let error = planned(&inclusive).await.unwrap_err();
    assert_refused(
        error,
        lineage.s2_overwrite,
        "overwrite",
        lineage.s1_append,
        lineage.s3_append,
    );

    let to_on_delete = window(&lineage, lineage.s3_append, lineage.s4_delete, |b| {
        b.with_fail_on_non_append(true)
    });
    let error = planned(&to_on_delete).await.unwrap_err();
    assert_refused(
        error,
        lineage.s4_delete,
        "delete",
        lineage.s3_append,
        lineage.s4_delete,
    );
}

#[tokio::test]
async fn fail_on_non_append_keeps_the_empty_range_empty() {
    let catalog = new_memory_catalog().await;
    let lineage = lineage(&catalog).await;
    let policy = NonAppendPolicy {
        fail_loud: true,
        skip_overwrite: false,
        skip_delete: false,
    };

    let empty = appends_between(
        lineage.table.metadata(),
        Some(lineage.s2_overwrite),
        lineage.s2_overwrite,
        policy,
    )
    .unwrap();
    assert!(empty.is_empty());

    let rejected = lineage
        .table
        .incremental_append_scan()
        .from_snapshot_id_exclusive(lineage.s2_overwrite)
        .to_snapshot_id(lineage.s2_overwrite)
        .with_fail_on_non_append(true)
        .build();
    assert!(rejected.is_err());

    let replace_only = window(&lineage, lineage.s4_delete, lineage.s5_replace, |b| {
        b.with_fail_on_non_append(true)
    });
    assert!(planned(&replace_only).await.unwrap().is_empty());
}
