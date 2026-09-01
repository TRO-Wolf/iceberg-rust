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
use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::arrow::array::{Int32Array, RecordBatch, StringArray};
use datafusion::execution::context::SessionContext;
use futures::TryStreamExt;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, MAIN_BRANCH, NestedField, PrimitiveType, Schema, Type,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{
    Catalog, CatalogBuilder, MemoryCatalog, NamespaceIdent, Result, TableCreation, TableIdent,
};
use iceberg_datafusion::IcebergTableProvider;

const BRANCH: &str = "b";
const TAG: &str = "t";

fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_BRANCH_GEN_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn compare_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_BRANCH_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("schema")
}

async fn catalog_at(warehouse: &str) -> MemoryCatalog {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "branch_dml",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_string())]),
        )
        .await
        .expect("memory catalog")
}

async fn create_named_table(
    catalog: &MemoryCatalog,
    name: &str,
    location: &str,
    format_version: FormatVersion,
    properties: HashMap<String, String>,
) -> NamespaceIdent {
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name(name.to_string())
                .location(location.to_string())
                .schema(schema())
                .format_version(format_version)
                .properties(properties)
                .build(),
        )
        .await
        .expect("create table");
    namespace
}

async fn provider(
    catalog: Arc<dyn Catalog>,
    namespace: NamespaceIdent,
    name: &str,
    branch: Option<&str>,
) -> IcebergTableProvider {
    let provider = IcebergTableProvider::try_new(catalog, namespace, name.to_string())
        .await
        .expect("provider");
    match branch {
        Some(branch_name) => provider.with_commit_branch(branch_name),
        None => provider,
    }
}

async fn register(provider: IcebergTableProvider) -> SessionContext {
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

async fn load(catalog: &dyn Catalog, namespace: &NamespaceIdent, name: &str) -> Table {
    catalog
        .load_table(&TableIdent::new(namespace.clone(), name.to_string()))
        .await
        .expect("load table")
}

fn ref_id(table: &Table, name: &str) -> Option<i64> {
    table
        .metadata()
        .snapshot_for_ref(name)
        .map(|snapshot| snapshot.snapshot_id())
}

async fn create_named_branch(catalog: &dyn Catalog, table: &Table, branch: &str) -> Table {
    let main_id = table
        .metadata()
        .current_snapshot_id()
        .expect("main head for branch");
    let tx = Transaction::new(table);
    let tx = tx
        .manage_snapshots()
        .create_branch(branch, main_id)
        .apply(tx)
        .expect("apply create_branch");
    tx.commit(catalog).await.expect("commit create_branch")
}

fn sorted_ids(batches: &[RecordBatch]) -> Vec<i32> {
    let mut ids = Vec::new();
    for batch in batches {
        let column = batch
            .column_by_name("id")
            .expect("id")
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("id int");
        for row in 0..column.len() {
            ids.push(column.value(row));
        }
    }
    ids.sort_unstable();
    ids
}

async fn query_ids(ctx: &SessionContext, sql: &str) -> Vec<i32> {
    let batches = ctx
        .sql(sql)
        .await
        .unwrap_or_else(|error| panic!("plan `{sql}`: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("exec `{sql}`: {error}"));
    sorted_ids(&batches)
}

async fn file_basenames(table: &Table, ref_name: &str) -> Vec<String> {
    let scan = table
        .scan()
        .use_ref(ref_name)
        .build()
        .unwrap_or_else(|error| panic!("scan {ref_name}: {error}"));
    let tasks: Vec<_> = scan
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

fn assert_file_sets_diverged(main_files: &[String], branch_files: &[String]) {
    assert!(
        !main_files.is_empty(),
        "main file set must be non-empty, got {main_files:?}"
    );
    assert!(
        branch_files.len() > main_files.len(),
        "branch file set must be a strict superset of main: main={main_files:?} branch={branch_files:?}"
    );
    for name in main_files {
        assert!(
            branch_files.contains(name),
            "branch file set missing main file {name}: main={main_files:?} branch={branch_files:?}"
        );
    }
}

async fn write_final(table: &Table, location: &str) {
    let path = format!("{location}/metadata/final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), &path)
        .await
        .unwrap_or_else(|error| panic!("write {path}: {error}"));
}

async fn seed_diverged(
    catalog: &Arc<dyn Catalog>,
    namespace: NamespaceIdent,
    name: &str,
) -> (i64, Vec<String>, Vec<String>) {
    let ctx = register(provider(catalog.clone(), namespace.clone(), name, None).await).await;
    run_sql(&ctx, "INSERT INTO t VALUES (1, 'main-a'), (2, 'main-b')").await;
    let table = load(catalog.as_ref(), &namespace, name).await;
    let table = create_named_branch(catalog.as_ref(), &table, BRANCH).await;
    let main_id = table.metadata().current_snapshot_id().expect("main");
    let ctx =
        register(provider(catalog.clone(), namespace.clone(), name, Some(BRANCH)).await).await;
    run_sql(
        &ctx,
        "INSERT INTO t VALUES (10, 'branch-x'), (11, 'branch-y')",
    )
    .await;
    let table = load(catalog.as_ref(), &namespace, name).await;
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    let main_files = file_basenames(&table, MAIN_BRANCH).await;
    let branch_files = file_basenames(&table, BRANCH).await;
    assert_file_sets_diverged(&main_files, &branch_files);
    (main_id, main_files, branch_files)
}

async fn write_id_file(table: &Table, id: i32, data: &str) -> DataFile {
    let iceberg_schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(iceberg_schema).expect("iceberg to arrow"));
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int32Array::from(vec![id])),
        Arc::new(StringArray::from(vec![data])),
    ])
    .expect("batch");
    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        format!("retry-{id}"),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        iceberg_schema.clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = DataFileWriterBuilder::new(rolling)
        .unpartitioned()
        .build(None)
        .await
        .expect("writer");
    writer.write(batch).await.expect("write");
    writer
        .close()
        .await
        .expect("close")
        .into_iter()
        .next()
        .expect("one file")
}

#[tokio::test]
async fn rust_reproduces_java_diverged_branch_table() -> Result<()> {
    let warehouse = tempfile::TempDir::new().expect("warehouse");
    let warehouse_path = warehouse.path().to_str().expect("utf8").to_string();
    let location = format!("{warehouse_path}/offline");
    let catalog = catalog_at(&warehouse_path).await;
    let namespace =
        create_named_table(&catalog, "t", &location, FormatVersion::V2, HashMap::new()).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let (main_id, main_files, branch_files) = seed_diverged(&catalog, namespace.clone(), "t").await;
    let table = load(catalog.as_ref(), &namespace, "t").await;
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    let branch_id = ref_id(&table, BRANCH).expect("branch");
    assert_ne!(branch_id, main_id);
    let parent = table
        .metadata()
        .snapshot_by_id(branch_id)
        .expect("branch snapshot")
        .parent_snapshot_id();
    assert_eq!(parent, Some(main_id));
    assert!(table.metadata().snapshot_for_ref(BRANCH).is_some());
    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", None).await).await;
    assert_eq!(query_ids(&ctx, "SELECT id FROM t").await, vec![1, 2]);
    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", Some(BRANCH)).await).await;
    assert_eq!(query_ids(&ctx, "SELECT id FROM t").await, vec![
        1, 2, 10, 11
    ]);
    assert_eq!(file_basenames(&table, MAIN_BRANCH).await, main_files);
    assert_eq!(file_basenames(&table, BRANCH).await, branch_files);
    Ok(())
}

#[tokio::test]
async fn v3_merge_on_read_delete_on_diverged_branch_uses_branch_files() -> Result<()> {
    let warehouse = tempfile::TempDir::new().expect("warehouse");
    let warehouse_path = warehouse.path().to_str().expect("utf8").to_string();
    let location = format!("{warehouse_path}/v3_mor");
    let properties = HashMap::from([
        ("write.delete.mode".to_string(), "merge-on-read".to_string()),
        ("write.update.mode".to_string(), "merge-on-read".to_string()),
    ]);
    let catalog = catalog_at(&warehouse_path).await;
    let namespace =
        create_named_table(&catalog, "t", &location, FormatVersion::V3, properties).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let (main_id, main_files, _) = seed_diverged(&catalog, namespace.clone(), "t").await;
    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", Some(BRANCH)).await).await;
    run_sql(&ctx, "DELETE FROM t WHERE id = 10").await;
    let table = load(catalog.as_ref(), &namespace, "t").await;
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    assert_eq!(file_basenames(&table, MAIN_BRANCH).await, main_files);
    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", None).await).await;
    assert_eq!(query_ids(&ctx, "SELECT id FROM t").await, vec![1, 2]);
    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", Some(BRANCH)).await).await;
    assert_eq!(query_ids(&ctx, "SELECT id FROM t").await, vec![1, 2, 11]);
    Ok(())
}

#[tokio::test]
async fn missing_branch_fails_on_read_and_update() -> Result<()> {
    let warehouse = tempfile::TempDir::new().expect("warehouse");
    let warehouse_path = warehouse.path().to_str().expect("utf8").to_string();
    let location = format!("{warehouse_path}/missing");
    let catalog = catalog_at(&warehouse_path).await;
    let namespace =
        create_named_table(&catalog, "t", &location, FormatVersion::V2, HashMap::new()).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", None).await).await;
    run_sql(&ctx, "INSERT INTO t VALUES (1, 'main-a')").await;
    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", Some(BRANCH)).await).await;
    let read_err = ctx
        .sql("SELECT id FROM t")
        .await
        .expect("plan")
        .collect()
        .await
        .expect_err("missing branch must fail SELECT");
    let read_message = read_err.to_string();
    assert!(
        read_message.contains(BRANCH) && read_message.contains("not found"),
        "SELECT must name the ref, got: {read_message}"
    );
    let update_err = ctx
        .sql("UPDATE t SET data = 'z' WHERE id = 1")
        .await
        .expect("plan")
        .collect()
        .await
        .expect_err("missing branch must fail UPDATE");
    let update_message = update_err.to_string();
    assert!(
        update_message.contains(BRANCH) && update_message.contains("not found"),
        "UPDATE must name the ref, got: {update_message}"
    );
    let table = load(catalog.as_ref(), &namespace, "t").await;
    assert!(table.metadata().snapshot_for_ref(BRANCH).is_none());
    Ok(())
}

#[tokio::test]
async fn insert_creates_missing_branch_per_snapshot_producer() -> Result<()> {
    let warehouse = tempfile::TempDir::new().expect("warehouse");
    let warehouse_path = warehouse.path().to_str().expect("utf8").to_string();
    let location = format!("{warehouse_path}/insert_create");
    let catalog = catalog_at(&warehouse_path).await;
    let namespace =
        create_named_table(&catalog, "t", &location, FormatVersion::V2, HashMap::new()).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", None).await).await;
    run_sql(&ctx, "INSERT INTO t VALUES (1, 'main-a'), (2, 'main-b')").await;
    let table = load(catalog.as_ref(), &namespace, "t").await;
    let main_id = table.metadata().current_snapshot_id().expect("main");
    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", Some(BRANCH)).await).await;
    run_sql(&ctx, "INSERT INTO t VALUES (20, 'created')").await;
    let table = load(catalog.as_ref(), &namespace, "t").await;
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    let branch_id = ref_id(&table, BRANCH).expect("created branch");
    assert_ne!(branch_id, main_id);
    let parent = table
        .metadata()
        .snapshot_by_id(branch_id)
        .expect("snapshot")
        .parent_snapshot_id();
    assert_eq!(parent, Some(main_id));
    Ok(())
}

#[tokio::test]
async fn tag_target_refuses_writes() -> Result<()> {
    let warehouse = tempfile::TempDir::new().expect("warehouse");
    let warehouse_path = warehouse.path().to_str().expect("utf8").to_string();
    let location = format!("{warehouse_path}/tag");
    let catalog = catalog_at(&warehouse_path).await;
    let namespace =
        create_named_table(&catalog, "t", &location, FormatVersion::V2, HashMap::new()).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", None).await).await;
    run_sql(&ctx, "INSERT INTO t VALUES (1, 'main-a')").await;
    let table = load(catalog.as_ref(), &namespace, "t").await;
    let main_id = table.metadata().current_snapshot_id().expect("main");
    let tx = Transaction::new(&table);
    let tx = tx
        .manage_snapshots()
        .create_tag(TAG, main_id)
        .apply(tx)
        .expect("apply tag");
    tx.commit(catalog.as_ref()).await.expect("commit tag");
    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", Some(TAG)).await).await;
    let err = ctx
        .sql("INSERT INTO t VALUES (2, 'nope')")
        .await
        .expect("plan")
        .collect()
        .await
        .expect_err("tag must refuse writes");
    let message = err.to_string();
    assert!(
        message
            .contains("t is a tag, not a branch. Tags cannot be targets for producing snapshots"),
        "unexpected message: {message}"
    );
    Ok(())
}

#[tokio::test]
async fn rust_reads_java_diverged_branch() -> Result<()> {
    let Some(dir) = compare_dir() else {
        println!("skipping rust_reads_java_diverged_branch — set ICEBERG_INTEROP_BRANCH_DIR");
        return Ok(());
    };
    let warehouse = tempfile::TempDir::new().expect("warehouse");
    let catalog = catalog_at(warehouse.path().to_str().expect("utf8")).await;
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let metadata = dir
        .join("diverged")
        .join("table")
        .join("metadata")
        .join("final.metadata.json");
    assert!(
        metadata.is_file(),
        "missing Java diverged fixture {}",
        metadata.display()
    );
    catalog
        .register_table(
            &TableIdent::new(namespace.clone(), "diverged".to_string()),
            metadata.to_string_lossy().to_string(),
        )
        .await
        .expect("register java diverged");
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let table = load(catalog.as_ref(), &namespace, "diverged").await;
    let main_files = file_basenames(&table, MAIN_BRANCH).await;
    let branch_files = file_basenames(&table, BRANCH).await;
    assert_file_sets_diverged(&main_files, &branch_files);
    let ctx = register(provider(catalog.clone(), namespace.clone(), "diverged", None).await).await;
    assert_eq!(query_ids(&ctx, "SELECT id FROM t").await, vec![1, 2]);
    let ctx =
        register(provider(catalog.clone(), namespace.clone(), "diverged", Some(BRANCH)).await)
            .await;
    assert_eq!(query_ids(&ctx, "SELECT id FROM t").await, vec![
        1, 2, 10, 11
    ]);
    let main_id = table.metadata().current_snapshot_id().expect("main");
    let branch_id = ref_id(&table, BRANCH).expect("java branch");
    assert_ne!(branch_id, main_id);
    assert_eq!(
        table
            .metadata()
            .snapshot_by_id(branch_id)
            .expect("snap")
            .parent_snapshot_id(),
        Some(main_id)
    );
    Ok(())
}

#[tokio::test]
async fn rust_reads_java_v3_diverged_branch() -> Result<()> {
    let Some(dir) = compare_dir() else {
        println!("skipping rust_reads_java_v3_diverged_branch — set ICEBERG_INTEROP_BRANCH_DIR");
        return Ok(());
    };
    let warehouse = tempfile::TempDir::new().expect("warehouse");
    let catalog = catalog_at(warehouse.path().to_str().expect("utf8")).await;
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let metadata = dir
        .join("v3_diverged")
        .join("table")
        .join("metadata")
        .join("final.metadata.json");
    catalog
        .register_table(
            &TableIdent::new(namespace.clone(), "v3".to_string()),
            metadata.to_string_lossy().to_string(),
        )
        .await
        .expect("register java v3");
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let table = load(catalog.as_ref(), &namespace, "v3").await;
    assert_eq!(table.metadata().format_version(), FormatVersion::V3);
    assert_file_sets_diverged(
        &file_basenames(&table, MAIN_BRANCH).await,
        &file_basenames(&table, BRANCH).await,
    );
    let ctx = register(provider(catalog.clone(), namespace.clone(), "v3", None).await).await;
    assert_eq!(query_ids(&ctx, "SELECT id FROM t").await, vec![1, 2]);
    let ctx =
        register(provider(catalog.clone(), namespace.clone(), "v3", Some(BRANCH)).await).await;
    assert_eq!(query_ids(&ctx, "SELECT id FROM t").await, vec![
        1, 2, 10, 11
    ]);
    Ok(())
}

#[tokio::test]
async fn java_missing_branch_and_tag_match_producer() -> Result<()> {
    let Some(dir) = compare_dir() else {
        println!(
            "skipping java_missing_branch_and_tag_match_producer — set ICEBERG_INTEROP_BRANCH_DIR"
        );
        return Ok(());
    };
    let warehouse = tempfile::TempDir::new().expect("warehouse");
    let catalog = catalog_at(warehouse.path().to_str().expect("utf8")).await;
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    catalog
        .register_table(
            &TableIdent::new(namespace.clone(), "no_branch".to_string()),
            dir.join("no_branch")
                .join("table")
                .join("metadata")
                .join("final.metadata.json")
                .to_string_lossy()
                .to_string(),
        )
        .await
        .expect("register no_branch");
    catalog
        .register_table(
            &TableIdent::new(namespace.clone(), "tag".to_string()),
            dir.join("tag")
                .join("table")
                .join("metadata")
                .join("final.metadata.json")
                .to_string_lossy()
                .to_string(),
        )
        .await
        .expect("register tag");
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let ctx = register(
        provider(
            catalog.clone(),
            namespace.clone(),
            "no_branch",
            Some(BRANCH),
        )
        .await,
    )
    .await;
    let read_err = ctx
        .sql("SELECT id FROM t")
        .await
        .expect("plan")
        .collect()
        .await
        .expect_err("java no_branch SELECT");
    assert!(
        read_err.to_string().contains(BRANCH),
        "missing-ref must name {BRANCH}: {}",
        read_err
    );
    let update_err = ctx
        .sql("UPDATE t SET data = 'z' WHERE id = 1")
        .await
        .expect("plan")
        .collect()
        .await
        .expect_err("java no_branch UPDATE");
    assert!(
        update_err.to_string().contains(BRANCH),
        "missing-ref UPDATE must name {BRANCH}: {}",
        update_err
    );
    let ctx = register(provider(catalog.clone(), namespace.clone(), "tag", Some(TAG)).await).await;
    let tag_err = ctx
        .sql("INSERT INTO t VALUES (9, 'nope')")
        .await
        .expect("plan")
        .collect()
        .await
        .expect_err("java tag INSERT");
    assert!(
        tag_err
            .to_string()
            .contains("t is a tag, not a branch. Tags cannot be targets for producing snapshots"),
        "unexpected tag message: {tag_err}"
    );
    Ok(())
}

async fn gen_table(
    gen_path: &Path,
    folder: &str,
    format_version: FormatVersion,
    properties: HashMap<String, String>,
) -> (Arc<dyn Catalog>, NamespaceIdent, String) {
    let warehouse = gen_path.to_string_lossy().to_string();
    let location = format!("{warehouse}/{folder}");
    let catalog = catalog_at(&warehouse).await;
    let namespace =
        create_named_table(&catalog, folder, &location, format_version, properties).await;
    (Arc::new(catalog), namespace, location)
}

#[tokio::test]
async fn gen_rust_append() {
    let Some(gen_path) = gen_dir() else {
        println!("skipping gen_rust_append — set ICEBERG_INTEROP_BRANCH_GEN_DIR");
        return;
    };
    let (catalog, namespace, location) =
        gen_table(&gen_path, "rust_append", FormatVersion::V2, HashMap::new()).await;
    let (main_id, main_files, _) = seed_diverged(&catalog, namespace.clone(), "rust_append").await;
    let ctx = register(
        provider(
            catalog.clone(),
            namespace.clone(),
            "rust_append",
            Some(BRANCH),
        )
        .await,
    )
    .await;
    run_sql(&ctx, "INSERT INTO t VALUES (20, 'rust-append')").await;
    let table = load(catalog.as_ref(), &namespace, "rust_append").await;
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    let ctx =
        register(provider(catalog.clone(), namespace.clone(), "rust_append", None).await).await;
    assert_eq!(query_ids(&ctx, "SELECT id FROM t").await, vec![1, 2]);
    let ctx = register(
        provider(
            catalog.clone(),
            namespace.clone(),
            "rust_append",
            Some(BRANCH),
        )
        .await,
    )
    .await;
    assert_eq!(query_ids(&ctx, "SELECT id FROM t").await, vec![
        1, 2, 10, 11, 20
    ]);
    let after_main = file_basenames(&table, MAIN_BRANCH).await;
    assert_eq!(after_main, main_files);
    assert_file_sets_diverged(&after_main, &file_basenames(&table, BRANCH).await);
    write_final(&table, &location).await;
}

#[tokio::test]
async fn gen_rust_cow() {
    let Some(gen_path) = gen_dir() else {
        println!("skipping gen_rust_cow — set ICEBERG_INTEROP_BRANCH_GEN_DIR");
        return;
    };
    let (catalog, namespace, location) =
        gen_table(&gen_path, "rust_cow", FormatVersion::V2, HashMap::new()).await;
    let (main_id, main_files, _) = seed_diverged(&catalog, namespace.clone(), "rust_cow").await;
    let ctx =
        register(provider(catalog.clone(), namespace.clone(), "rust_cow", Some(BRANCH)).await)
            .await;
    run_sql(&ctx, "DELETE FROM t WHERE id = 10").await;
    run_sql(&ctx, "UPDATE t SET data = 'z' WHERE id = 11").await;
    let table = load(catalog.as_ref(), &namespace, "rust_cow").await;
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    let ctx = register(provider(catalog.clone(), namespace.clone(), "rust_cow", None).await).await;
    assert_eq!(query_ids(&ctx, "SELECT id FROM t").await, vec![1, 2]);
    let ctx =
        register(provider(catalog.clone(), namespace.clone(), "rust_cow", Some(BRANCH)).await)
            .await;
    assert_eq!(query_ids(&ctx, "SELECT id FROM t").await, vec![1, 2, 11]);
    assert_eq!(file_basenames(&table, MAIN_BRANCH).await, main_files);
    write_final(&table, &location).await;
}

#[tokio::test]
async fn gen_rust_mor() {
    let Some(gen_path) = gen_dir() else {
        println!("skipping gen_rust_mor — set ICEBERG_INTEROP_BRANCH_GEN_DIR");
        return;
    };
    let properties = HashMap::from([
        ("write.delete.mode".to_string(), "merge-on-read".to_string()),
        ("write.update.mode".to_string(), "merge-on-read".to_string()),
    ]);
    let (catalog, namespace, location) =
        gen_table(&gen_path, "rust_mor", FormatVersion::V3, properties).await;
    let (main_id, main_files, _) = seed_diverged(&catalog, namespace.clone(), "rust_mor").await;
    let ctx =
        register(provider(catalog.clone(), namespace.clone(), "rust_mor", Some(BRANCH)).await)
            .await;
    run_sql(&ctx, "DELETE FROM t WHERE id = 10").await;
    run_sql(&ctx, "UPDATE t SET data = 'z' WHERE id = 11").await;
    let table = load(catalog.as_ref(), &namespace, "rust_mor").await;
    assert_eq!(table.metadata().format_version(), FormatVersion::V3);
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    let ctx = register(provider(catalog.clone(), namespace.clone(), "rust_mor", None).await).await;
    assert_eq!(query_ids(&ctx, "SELECT id FROM t").await, vec![1, 2]);
    let ctx =
        register(provider(catalog.clone(), namespace.clone(), "rust_mor", Some(BRANCH)).await)
            .await;
    assert_eq!(query_ids(&ctx, "SELECT id FROM t").await, vec![1, 2, 11]);
    assert_eq!(file_basenames(&table, MAIN_BRANCH).await, main_files);
    write_final(&table, &location).await;
}

#[tokio::test]
async fn gen_rust_created() {
    let Some(gen_path) = gen_dir() else {
        println!("skipping gen_rust_created — set ICEBERG_INTEROP_BRANCH_GEN_DIR");
        return;
    };
    let (catalog, namespace, location) =
        gen_table(&gen_path, "rust_created", FormatVersion::V2, HashMap::new()).await;
    let (main_id, main_files, branch_files) =
        seed_diverged(&catalog, namespace.clone(), "rust_created").await;
    let table = load(catalog.as_ref(), &namespace, "rust_created").await;
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    assert_file_sets_diverged(&main_files, &branch_files);
    write_final(&table, &location).await;
}

#[tokio::test]
async fn gen_rust_insert_create() {
    let Some(gen_path) = gen_dir() else {
        println!("skipping gen_rust_insert_create — set ICEBERG_INTEROP_BRANCH_GEN_DIR");
        return;
    };
    let (catalog, namespace, location) = gen_table(
        &gen_path,
        "rust_insert_create",
        FormatVersion::V2,
        HashMap::new(),
    )
    .await;
    let ctx = register(
        provider(
            catalog.clone(),
            namespace.clone(),
            "rust_insert_create",
            None,
        )
        .await,
    )
    .await;
    run_sql(&ctx, "INSERT INTO t VALUES (1, 'main-a'), (2, 'main-b')").await;
    let table = load(catalog.as_ref(), &namespace, "rust_insert_create").await;
    let main_id = table.metadata().current_snapshot_id().expect("main");
    let main_files = file_basenames(&table, MAIN_BRANCH).await;
    let ctx = register(
        provider(
            catalog.clone(),
            namespace.clone(),
            "rust_insert_create",
            Some(BRANCH),
        )
        .await,
    )
    .await;
    run_sql(&ctx, "INSERT INTO t VALUES (20, 'created')").await;
    let table = load(catalog.as_ref(), &namespace, "rust_insert_create").await;
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    assert_eq!(file_basenames(&table, MAIN_BRANCH).await, main_files);
    assert_file_sets_diverged(&main_files, &file_basenames(&table, BRANCH).await);
    write_final(&table, &location).await;
}

#[tokio::test]
async fn gen_rust_retry() {
    let Some(gen_path) = gen_dir() else {
        println!("skipping gen_rust_retry — set ICEBERG_INTEROP_BRANCH_GEN_DIR");
        return;
    };
    let (catalog, namespace, location) =
        gen_table(&gen_path, "rust_retry", FormatVersion::V2, HashMap::new()).await;
    let (main_id, main_files, _) = seed_diverged(&catalog, namespace.clone(), "rust_retry").await;
    let table = load(catalog.as_ref(), &namespace, "rust_retry").await;
    let pending_file = write_id_file(&table, 31, "pending").await;
    let winner_file = write_id_file(&table, 30, "winner").await;
    let pending = Transaction::new(&table);
    let pending = pending
        .fast_append()
        .add_data_files(vec![pending_file])
        .to_branch(BRANCH)
        .apply(pending)
        .expect("apply pending");
    let winner = Transaction::new(&table);
    let winner = winner
        .fast_append()
        .add_data_files(vec![winner_file])
        .to_branch(BRANCH)
        .apply(winner)
        .expect("apply winner");
    let table = winner.commit(catalog.as_ref()).await.expect("winner");
    let winner_id = ref_id(&table, BRANCH).expect("winner head");
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    let table = pending
        .commit(catalog.as_ref())
        .await
        .expect("pending retry");
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    let pending_id = ref_id(&table, BRANCH).expect("pending head");
    assert_ne!(pending_id, winner_id);
    assert_eq!(
        table
            .metadata()
            .snapshot_by_id(pending_id)
            .expect("pending snap")
            .parent_snapshot_id(),
        Some(winner_id)
    );
    assert_eq!(file_basenames(&table, MAIN_BRANCH).await, main_files);
    write_final(&table, &location).await;
}
