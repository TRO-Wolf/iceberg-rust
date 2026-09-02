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
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::arrow::array::{Array, AsArray};
use datafusion::execution::context::SessionContext;
use futures::TryStreamExt;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::metadata_columns::{
    RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER, RESERVED_COL_NAME_ROW_ID,
};
use iceberg::spec::{FormatVersion, MAIN_BRANCH, NestedField, PrimitiveType, Schema, Type};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergTableProvider;

const BRANCH: &str = "b";
const UPDATED_ID: i32 = 10;

fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_MOR_BRANCH_LINEAGE_GEN_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn compare_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_MOR_BRANCH_LINEAGE_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn current_hadoop_metadata(meta_dir: &Path) -> PathBuf {
    let mut best: Option<(u64, PathBuf)> = None;
    for entry in fs::read_dir(meta_dir).expect("java metadata dir") {
        let path = entry.expect("dirent").path();
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("");
        let Some(rest) = name.strip_prefix('v') else {
            continue;
        };
        let Some(digits) = rest.strip_suffix(".metadata.json") else {
            continue;
        };
        let Ok(version) = digits.parse::<u64>() else {
            continue;
        };
        match &best {
            Some((current, _)) if version <= *current => {}
            _ => best = Some((version, path)),
        }
    }
    best.map(|(_, path)| path)
        .expect("the Java table writes vN.metadata.json")
}

fn schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "val", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("schema")
}

fn mor_properties() -> HashMap<String, String> {
    HashMap::from([
        ("write.delete.mode".to_string(), "merge-on-read".to_string()),
        ("write.update.mode".to_string(), "merge-on-read".to_string()),
    ])
}

async fn catalog_at(warehouse: &str) -> MemoryCatalog {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "mor_branch_lineage",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_string())]),
        )
        .await
        .expect("memory catalog")
}

async fn namespace_in(catalog: &MemoryCatalog) -> NamespaceIdent {
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    namespace
}

async fn load(catalog: &dyn Catalog, namespace: &NamespaceIdent, name: &str) -> Table {
    catalog
        .load_table(&TableIdent::new(namespace.clone(), name.to_string()))
        .await
        .expect("load table")
}

async fn branch_ctx(
    catalog: Arc<dyn Catalog>,
    namespace: NamespaceIdent,
    name: &str,
    branch: Option<&str>,
) -> SessionContext {
    let provider = IcebergTableProvider::try_new(catalog, namespace, name.to_string())
        .await
        .expect("provider");
    let provider = match branch {
        Some(branch_name) => provider.with_commit_branch(branch_name),
        None => provider,
    };
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(provider))
        .expect("register");
    ctx
}

async fn run_sql(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql)
        .await
        .unwrap_or_else(|error| panic!("plan `{sql}`: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("exec `{sql}`: {error}"));
}

async fn lineage_on_ref(table: &Table, ref_name: &str) -> Vec<(i32, i64, i64)> {
    let batches: Vec<_> = table
        .scan()
        .use_ref(ref_name)
        .select([
            "id",
            RESERVED_COL_NAME_ROW_ID,
            RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER,
        ])
        .build()
        .unwrap_or_else(|error| panic!("scan {ref_name}: {error}"))
        .to_arrow()
        .await
        .unwrap_or_else(|error| panic!("to_arrow {ref_name}: {error}"))
        .try_collect()
        .await
        .unwrap_or_else(|error| panic!("collect {ref_name}: {error}"));
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .expect("id")
            .as_primitive::<datafusion::arrow::datatypes::Int32Type>();
        let row_ids = batch
            .column_by_name(RESERVED_COL_NAME_ROW_ID)
            .expect("_row_id")
            .as_primitive::<datafusion::arrow::datatypes::Int64Type>();
        let seqs = batch
            .column_by_name(RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER)
            .expect("_last_updated_sequence_number")
            .as_primitive::<datafusion::arrow::datatypes::Int64Type>();
        for index in 0..batch.num_rows() {
            assert!(row_ids.is_valid(index), "live row must carry _row_id");
            assert!(
                seqs.is_valid(index),
                "live row must carry a sequence number"
            );
            rows.push((ids.value(index), row_ids.value(index), seqs.value(index)));
        }
    }
    rows.sort_unstable();
    rows
}

fn lineage_text(rows: &[(i32, i64, i64)]) -> String {
    let mut text = String::new();
    for (id, row_id, seq) in rows {
        text.push_str(&format!("{id}={row_id}={seq}\n"));
    }
    text
}

fn seq_of(rows: &[(i32, i64, i64)], id: i32) -> i64 {
    rows.iter()
        .find(|(row, _, _)| *row == id)
        .unwrap_or_else(|| panic!("id {id} not live"))
        .2
}

fn row_id_of(rows: &[(i32, i64, i64)], id: i32) -> i64 {
    rows.iter()
        .find(|(row, _, _)| *row == id)
        .unwrap_or_else(|| panic!("id {id} not live"))
        .1
}

async fn file_basenames(table: &Table, ref_name: &str) -> Vec<String> {
    let tasks: Vec<_> = table
        .scan()
        .use_ref(ref_name)
        .build()
        .unwrap_or_else(|error| panic!("scan {ref_name}: {error}"))
        .plan_files()
        .await
        .unwrap_or_else(|error| panic!("plan_files {ref_name}: {error}"))
        .try_collect()
        .await
        .unwrap_or_else(|error| panic!("collect {ref_name}: {error}"));
    let mut names: Vec<String> = tasks
        .iter()
        .map(|task| {
            Path::new(task.data_file_path())
                .file_name()
                .expect("basename")
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names.dedup();
    names
}

fn parse_lineage(path: &Path) -> Vec<(i32, i64, i64)> {
    let text = fs::read_to_string(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    let mut rows = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split('=').collect();
        assert_eq!(parts.len(), 3, "malformed lineage line `{trimmed}`");
        rows.push((
            parts[0].parse::<i32>().expect("id"),
            parts[1].parse::<i64>().expect("row id"),
            parts[2].parse::<i64>().expect("sequence number"),
        ));
    }
    rows.sort_unstable();
    rows
}

async fn create_branch(catalog: &dyn Catalog, table: &Table, branch: &str) -> Table {
    let main_id = table
        .metadata()
        .current_snapshot_id()
        .expect("main head for the branch");
    let tx = Transaction::new(table);
    let tx = tx
        .manage_snapshots()
        .create_branch(branch, main_id)
        .apply(tx)
        .expect("apply create_branch");
    tx.commit(catalog).await.expect("commit create_branch")
}

async fn seed_diverged_v3(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
    name: &str,
) -> Table {
    let ctx = branch_ctx(catalog.clone(), namespace.clone(), name, None).await;
    run_sql(&ctx, "INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')").await;
    let table = load(catalog.as_ref(), namespace, name).await;
    create_branch(catalog.as_ref(), &table, BRANCH).await;
    let ctx = branch_ctx(catalog.clone(), namespace.clone(), name, Some(BRANCH)).await;
    run_sql(&ctx, "INSERT INTO t VALUES (10, 'x'), (11, 'y')").await;
    load(catalog.as_ref(), namespace, name).await
}

#[tokio::test]
async fn mor_update_on_branch_keeps_row_id_and_advances_seq_twice() {
    let warehouse = tempfile::TempDir::new().expect("warehouse");
    let warehouse_path = warehouse.path().to_str().expect("utf8").to_string();
    let catalog = catalog_at(&warehouse_path).await;
    let namespace = namespace_in(&catalog).await;
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("t".to_string())
                .location(format!("{warehouse_path}/t"))
                .schema(schema())
                .format_version(FormatVersion::V3)
                .properties(mor_properties())
                .build(),
        )
        .await
        .expect("create table");
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let table = seed_diverged_v3(&catalog, &namespace, "t").await;

    let main_id = table.metadata().current_snapshot_id().expect("main");
    let main_files = file_basenames(&table, MAIN_BRANCH).await;
    let main_lineage = lineage_on_ref(&table, MAIN_BRANCH).await;
    let next_row_id_before = table.metadata().next_row_id();
    let branch_seed = lineage_on_ref(&table, BRANCH).await;
    assert_eq!(branch_seed.len(), 5, "seed branch rows: {branch_seed:?}");
    let seed_row_id = row_id_of(&branch_seed, UPDATED_ID);
    let seed_seq = seq_of(&branch_seed, UPDATED_ID);

    let ctx = branch_ctx(catalog.clone(), namespace.clone(), "t", Some(BRANCH)).await;
    run_sql(&ctx, "UPDATE t SET val = 'X' WHERE id = 10").await;
    let table = load(catalog.as_ref(), &namespace, "t").await;
    let after_first = lineage_on_ref(&table, BRANCH).await;
    let first_seq = seq_of(&after_first, UPDATED_ID);
    assert_eq!(row_id_of(&after_first, UPDATED_ID), seed_row_id);
    assert!(
        first_seq > seed_seq,
        "first update must advance the sequence"
    );

    run_sql(&ctx, "UPDATE t SET val = 'XX' WHERE id = 10").await;
    let table = load(catalog.as_ref(), &namespace, "t").await;
    let after_second = lineage_on_ref(&table, BRANCH).await;
    assert_eq!(
        row_id_of(&after_second, UPDATED_ID),
        seed_row_id,
        "the updated row keeps one _row_id across both branch updates"
    );
    assert!(
        seq_of(&after_second, UPDATED_ID) > first_seq,
        "second update must advance the sequence again"
    );
    for (id, row_id, seq) in &branch_seed {
        if *id == UPDATED_ID {
            continue;
        }
        assert_eq!(
            (row_id_of(&after_second, *id), seq_of(&after_second, *id)),
            (*row_id, *seq),
            "unmatched branch row {id} lineage moved"
        );
    }
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    assert_eq!(file_basenames(&table, MAIN_BRANCH).await, main_files);
    assert_eq!(lineage_on_ref(&table, MAIN_BRANCH).await, main_lineage);
    assert_eq!(table.metadata().next_row_id(), next_row_id_before);
    let branch_files = file_basenames(&table, BRANCH).await;
    assert_eq!(
        branch_files.len(),
        main_files.len() + 3,
        "branch holds the seed pair plus the two UPDATE outputs: main={main_files:?} branch={branch_files:?}"
    );
}

#[tokio::test]
async fn rust_reads_java_branch_lineage() {
    let Some(dir) = compare_dir() else {
        println!(
            "skipping rust_reads_java_branch_lineage — set ICEBERG_INTEROP_MOR_BRANCH_LINEAGE_DIR"
        );
        return;
    };
    let metadata = dir
        .join("branch_table")
        .join("metadata")
        .join("final.metadata.json");
    assert!(
        metadata.is_file(),
        "missing Java fixture {}",
        metadata.display()
    );
    let warehouse = tempfile::TempDir::new().expect("warehouse");
    let catalog = catalog_at(warehouse.path().to_str().expect("utf8")).await;
    let namespace = namespace_in(&catalog).await;
    catalog
        .register_table(
            &TableIdent::new(namespace.clone(), "branch_table".to_string()),
            metadata.to_string_lossy().to_string(),
        )
        .await
        .expect("register the Java table");
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let table = load(catalog.as_ref(), &namespace, "branch_table").await;
    assert_eq!(table.metadata().format_version(), FormatVersion::V3);
    assert_eq!(
        lineage_on_ref(&table, BRANCH).await,
        parse_lineage(&dir.join("java_seed_branch_lineage.txt")),
        "Rust must read the branch lineage Java wrote"
    );
    assert_eq!(
        lineage_on_ref(&table, MAIN_BRANCH).await,
        parse_lineage(&dir.join("java_seed_main_lineage.txt")),
        "Rust must read the main lineage Java wrote"
    );
    let branch_files = file_basenames(&table, BRANCH).await;
    let main_files = file_basenames(&table, MAIN_BRANCH).await;
    assert_eq!(
        branch_files.len(),
        main_files.len() + 1,
        "Java branch seed is main's file plus one branch file: main={main_files:?} branch={branch_files:?}"
    );
}

#[tokio::test]
async fn rust_updates_java_branch_lineage_gen() {
    let Some(dir) = gen_dir() else {
        println!(
            "skipping rust_updates_java_branch_lineage_gen — set ICEBERG_INTEROP_MOR_BRANCH_LINEAGE_GEN_DIR"
        );
        return;
    };
    let metadata = current_hadoop_metadata(&dir.join("branch_table").join("metadata"));
    assert!(
        metadata.is_file(),
        "missing Java fixture {}",
        metadata.display()
    );
    let warehouse = tempfile::TempDir::new().expect("warehouse");
    let catalog = catalog_at(warehouse.path().to_str().expect("utf8")).await;
    let namespace = namespace_in(&catalog).await;
    catalog
        .register_table(
            &TableIdent::new(namespace.clone(), "branch_table".to_string()),
            metadata.to_string_lossy().to_string(),
        )
        .await
        .expect("register the Java table");
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);

    let table = load(catalog.as_ref(), &namespace, "branch_table").await;
    let main_id = table.metadata().current_snapshot_id().expect("main");
    let main_files = file_basenames(&table, MAIN_BRANCH).await;
    let main_lineage = lineage_on_ref(&table, MAIN_BRANCH).await;
    let next_row_id_before = table.metadata().next_row_id();
    let branch_seed = lineage_on_ref(&table, BRANCH).await;
    let seed_row_id = row_id_of(&branch_seed, UPDATED_ID);
    let seed_seq = seq_of(&branch_seed, UPDATED_ID);

    let ctx = branch_ctx(
        catalog.clone(),
        namespace.clone(),
        "branch_table",
        Some(BRANCH),
    )
    .await;
    run_sql(&ctx, "UPDATE t SET val = 'X' WHERE id = 10").await;
    let table = load(catalog.as_ref(), &namespace, "branch_table").await;
    let after_first = lineage_on_ref(&table, BRANCH).await;
    let first_seq = seq_of(&after_first, UPDATED_ID);
    assert_eq!(row_id_of(&after_first, UPDATED_ID), seed_row_id);
    assert!(
        first_seq > seed_seq,
        "first update must advance the sequence"
    );

    run_sql(&ctx, "UPDATE t SET val = 'XX' WHERE id = 10").await;
    let table = load(catalog.as_ref(), &namespace, "branch_table").await;
    let after_second = lineage_on_ref(&table, BRANCH).await;
    assert_eq!(row_id_of(&after_second, UPDATED_ID), seed_row_id);
    assert!(seq_of(&after_second, UPDATED_ID) > first_seq);
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    assert_eq!(file_basenames(&table, MAIN_BRANCH).await, main_files);
    assert_eq!(lineage_on_ref(&table, MAIN_BRANCH).await, main_lineage);
    assert_eq!(table.metadata().next_row_id(), next_row_id_before);

    let after_dir = dir.join("rust_after");
    let rust_metadata_dir = after_dir.join("rust_table").join("metadata");
    fs::create_dir_all(&rust_metadata_dir).expect("rust_after metadata dir");
    table
        .metadata()
        .write_to(
            table.file_io(),
            rust_metadata_dir
                .join("final.metadata.json")
                .to_str()
                .expect("utf8 metadata path"),
        )
        .await
        .expect("write final.metadata.json");
    fs::write(
        after_dir.join("expected_branch_lineage.txt"),
        lineage_text(&after_second),
    )
    .expect("expected_branch_lineage.txt");
    fs::write(
        after_dir.join("expected_branch_files.txt"),
        file_basenames(&table, BRANCH).await.join("\n") + "\n",
    )
    .expect("expected_branch_files.txt");
    fs::write(after_dir.join("updated_id.txt"), format!("{UPDATED_ID}\n")).expect("updated_id.txt");
    fs::write(
        after_dir.join("first_update_seq.txt"),
        format!("{first_seq}\n"),
    )
    .expect("first_update_seq.txt");
}
