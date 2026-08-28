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

//! F-13 U4 — V3 merge-on-read SQL DML interop: Rust deletes via DataFusion SQL, Java reads back.
//!
//! The sibling of [`interop_partitioned_dml`] (copy-on-write) and the SQL-driven sibling of
//! `iceberg/tests/interop_dv_table.rs` (hand-driven DVs). This one proves the whole U4 chain:
//! a SQL `DELETE` on a V3 table routes to `DVFileWriter`, commits Puffin deletion vectors, and
//! Java's production scan reads the result correctly.
//!
//! The third DELETE is the point of the fixture. It targets a data file that ALREADY carries a
//! DV, so it exercises the load-merge-supersede path that hand-driven harnesses never reach.
//!
//! What the Java leg actually proves, measured by sabotage: its ROW comparison is load-bearing —
//! shifting every DV position by one yields a table Java reads as `{1,4,5}` instead of `{3,4,6}`.
//! Its two SHAPE checks (no position-delete file, one DV per data file) are belt-and-braces: the
//! fork's own `RowDelta` already refuses both, with "Must use DVs for position deletes in V3" and
//! the fresh-DV door. They are kept because they cost nothing and would catch a regression in
//! those doors, but they are not the reason this leg exists.
//!
//! **When `ICEBERG_INTEROP_DV_SQL_GEN_DIR` is unset this test is a clean no-op**, so the offline
//! `cargo test` gate needs no Java or Maven. `dev/java-interop/run-interop-dv-sql.sh` sets it.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, Int64Array, RunArray, StringArray};
use datafusion::arrow::datatypes::Int32Type;
use datafusion::execution::context::SessionContext;
use futures::TryStreamExt;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::metadata_columns::{RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_POS};
use iceberg::spec::{
    DataFileFormat, FormatVersion, ManifestContentType, NestedField, PartitionKey, PrimitiveType,
    Schema, Transform, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::deletion_vector_writer::DVFileWriter;
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;

/// Return the GEN dir from the environment variable, or `None` when unset.
fn dv_sql_gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_DV_SQL_GEN_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// Hadoop `vN.metadata.json` with the highest N. `final.metadata.json` is a copy, not a commit
/// pointer.
fn current_hadoop_metadata(meta_dir: &std::path::Path) -> PathBuf {
    let mut best: Option<(u64, PathBuf)> = None;
    for entry in fs::read_dir(meta_dir).expect("java metadata dir") {
        let path = entry.expect("dirent").path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
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
        .expect("Java table writes vN.metadata.json")
}

async fn run_sql(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql)
        .await
        .unwrap_or_else(|error| panic!("plan `{sql}`: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("execute `{sql}`: {error}"));
}

/// Risk pinned (with the Java `verify-interop-dv-sql` step): the whole V3 merge-on-read SQL path.
/// A DELETE that wrote Parquet position deletes would produce a table Java rejects at V3; a DV that
/// was dropped, wrongly keyed, or left un-superseded resurrects or double-counts rows.
#[tokio::test]
async fn test_dv_sql_gen_rust_deletes_java_readable_v3_dv_table() {
    let Some(gen_dir) = dv_sql_gen_dir() else {
        println!(
            "skipping interop_dv_sql GEN — set ICEBERG_INTEROP_DV_SQL_GEN_DIR \
             (run dev/java-interop/run-interop-dv-sql.sh)"
        );
        return;
    };
    fs::create_dir_all(&gen_dir).expect("create interop dir");

    // 1. MemoryCatalog over the LOCAL FS; the table sits at a path Java can predict.
    let warehouse = gen_dir.to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_dv_sql",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS for dv-sql interop");

    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");

    // 2. A V3 table, partitioned by identity(category), both row-level modes merge-on-read.
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "category", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build schema");
    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(3, "category", Transform::Identity)
        .expect("add identity(category) partition field")
        .build();

    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location.clone())
        .schema(schema)
        .partition_spec(partition_spec)
        .format_version(FormatVersion::V3)
        .properties(HashMap::from([
            ("write.delete.mode".to_string(), "merge-on-read".to_string()),
            ("write.update.mode".to_string(), "merge-on-read".to_string()),
        ]))
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create rust_table");

    let client = Arc::new(catalog);
    let provider = IcebergCatalogProvider::try_new(client.clone())
        .await
        .expect("build IcebergCatalogProvider");
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));

    // 3. Six rows across two partitions.
    run_sql(
        &ctx,
        "INSERT INTO catalog.interop.rust_table VALUES \
         (1, 'a', 'electronics'), (2, 'b', 'electronics'), (3, 'c', 'electronics'), \
         (4, 'd', 'books'), (5, 'e', 'books'), (6, 'f', 'books')",
    )
    .await;

    // 4. Three `DELETE` statements in three commits. The first two touch a different partition each; the
    //    THIRD re-deletes from the electronics file, which already carries the first DV, so the
    //    writer must load it, merge, and supersede it.
    run_sql(&ctx, "DELETE FROM catalog.interop.rust_table WHERE id = 2").await;
    run_sql(&ctx, "DELETE FROM catalog.interop.rust_table WHERE id = 5").await;
    run_sql(&ctx, "DELETE FROM catalog.interop.rust_table WHERE id = 1").await;

    // 5. Rust's own read is the first check; Java's is the one that matters.
    let table_ident = TableIdent::new(namespace.clone(), "rust_table".to_string());
    let table = client
        .load_table(&table_ident)
        .await
        .expect("reload the table after the deletes");

    let mut live_delete_files = Vec::new();
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("the table has commits");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load manifest list");
    for manifest_entry in manifest_list.entries() {
        if manifest_entry.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_entry
            .load_manifest(table.file_io())
            .await
            .expect("load delete manifest");
        for entry in manifest.entries() {
            if entry.is_alive() {
                live_delete_files.push(entry.data_file().clone());
            }
        }
    }

    assert_eq!(
        live_delete_files.len(),
        2,
        "one live DV per touched data file — the third DELETE supersedes the first DV rather \
         than adding a third"
    );
    for delete_file in &live_delete_files {
        assert_eq!(
            delete_file.file_format(),
            DataFileFormat::Puffin,
            "V3 forbids new position-delete files"
        );
    }

    // 6. The ground truth Java compares against, and the metadata at a path Java can find.
    // Written by hand rather than through serde_json: this crate does not carry it as a
    // dev-dependency, and three rows do not justify adding one.
    let expected_rows = "[\n  {\"id\": 3, \"data\": \"c\"},\n                           {\"id\": 4, \"data\": \"d\"},\n                           {\"id\": 6, \"data\": \"f\"}\n]\n";
    fs::write(gen_dir.join("expected_rows.json"), expected_rows).expect("write expected_rows.json");

    let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), &final_metadata_path)
        .await
        .expect("write final.metadata.json");

    println!(
        "interop_dv_sql GEN OK — three SQL `DELETE` statements on a V3 table left {} live Puffin DVs; \
         final.metadata.json → {final_metadata_path}",
        live_delete_files.len()
    );
}

/// F-17: Java reads a Rust SQL DELETE that closed a shared two-blob Puffin.
#[tokio::test]
async fn test_dv_sql_gen_shared_puffin_delete_java_readable() {
    let Some(root) = dv_sql_gen_dir() else {
        println!("skipping interop_dv_sql shared-puffin GEN — set ICEBERG_INTEROP_DV_SQL_GEN_DIR");
        return;
    };
    let gen_dir = root.join("shared_puffin");
    fs::create_dir_all(&gen_dir).expect("create shared-puffin interop dir");
    let warehouse = gen_dir.to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_dv_sql_shared",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("catalog");
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "category", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("schema");
    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(3, "category", Transform::Identity)
        .expect("identity(category)")
        .build();
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("rust_table".to_string())
                .location(table_location.clone())
                .schema(schema)
                .partition_spec(partition_spec)
                .format_version(FormatVersion::V3)
                .properties(HashMap::from([
                    ("write.delete.mode".to_string(), "merge-on-read".to_string()),
                    ("write.update.mode".to_string(), "merge-on-read".to_string()),
                ]))
                .build(),
        )
        .await
        .expect("create table");
    let client = Arc::new(catalog);
    let provider = IcebergCatalogProvider::try_new(client.clone())
        .await
        .expect("provider");
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));
    run_sql(
        &ctx,
        "INSERT INTO catalog.interop.rust_table VALUES \
         (1, 'a', 'electronics'), (2, 'b', 'electronics'), (3, 'c', 'electronics'), \
         (4, 'd', 'books'), (5, 'e', 'books'), (6, 'f', 'books')",
    )
    .await;

    let ident = TableIdent::new(namespace.clone(), "rust_table".to_string());
    let table = client.load_table(&ident).await.expect("load after insert");
    commit_shared_puffin_for_ids(&table, client.as_ref(), 2, 5).await;
    run_sql(&ctx, "DELETE FROM catalog.interop.rust_table WHERE id = 1").await;
    let table = client.load_table(&ident).await.expect("load after delete");
    let expected_rows = "[\n  {\"id\": 3, \"data\": \"c\"},\n  {\"id\": 4, \"data\": \"d\"},\n  {\"id\": 6, \"data\": \"f\"}\n]\n";
    fs::write(gen_dir.join("expected_rows.json"), expected_rows).expect("expected_rows.json");
    let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), &final_metadata_path)
        .await
        .expect("write final.metadata.json");
    println!("interop_dv_sql shared-puffin GEN OK → {final_metadata_path}");
}

/// F-17: Java reads a Rust SQL UPDATE that closed a shared two-blob Puffin.
#[tokio::test]
async fn test_dv_sql_gen_shared_puffin_update_java_readable() {
    let Some(root) = dv_sql_gen_dir() else {
        println!(
            "skipping interop_dv_sql shared-puffin UPDATE GEN — set ICEBERG_INTEROP_DV_SQL_GEN_DIR"
        );
        return;
    };
    let gen_dir = root.join("shared_puffin_update");
    fs::create_dir_all(&gen_dir).expect("create shared-puffin update interop dir");
    let warehouse = gen_dir.to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_dv_sql_shared_update",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("catalog");
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "category", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("schema");
    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(3, "category", Transform::Identity)
        .expect("identity(category)")
        .build();
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("rust_table".to_string())
                .location(table_location.clone())
                .schema(schema)
                .partition_spec(partition_spec)
                .format_version(FormatVersion::V3)
                .properties(HashMap::from([
                    ("write.delete.mode".to_string(), "merge-on-read".to_string()),
                    ("write.update.mode".to_string(), "merge-on-read".to_string()),
                ]))
                .build(),
        )
        .await
        .expect("create table");
    let client = Arc::new(catalog);
    let provider = IcebergCatalogProvider::try_new(client.clone())
        .await
        .expect("provider");
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));
    run_sql(
        &ctx,
        "INSERT INTO catalog.interop.rust_table VALUES \
         (1, 'a', 'electronics'), (2, 'b', 'electronics'), (3, 'c', 'electronics'), \
         (4, 'd', 'books'), (5, 'e', 'books'), (6, 'f', 'books')",
    )
    .await;
    let ident = TableIdent::new(namespace.clone(), "rust_table".to_string());
    let table = client.load_table(&ident).await.expect("load after insert");
    commit_shared_puffin_for_ids(&table, client.as_ref(), 2, 5).await;
    run_sql(
        &ctx,
        "UPDATE catalog.interop.rust_table SET data = 'z' WHERE id = 1",
    )
    .await;
    let table = client.load_table(&ident).await.expect("load after update");
    let expected_rows = "[\n  {\"id\": 1, \"data\": \"z\"},\n  {\"id\": 3, \"data\": \"c\"},\n  {\"id\": 4, \"data\": \"d\"},\n  {\"id\": 6, \"data\": \"f\"}\n]\n";
    fs::write(gen_dir.join("expected_rows.json"), expected_rows).expect("expected_rows.json");
    let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), &final_metadata_path)
        .await
        .expect("write final.metadata.json");
    println!("interop_dv_sql shared-puffin UPDATE GEN OK → {final_metadata_path}");
}

/// F-17: Rust SQL DELETE against a Java BaseDVFileWriter shared Puffin; Java reads the result.
#[tokio::test]
async fn test_dv_sql_consume_java_written_shared_puffin() {
    let Some(java_dir) = std::env::var_os("ICEBERG_INTEROP_DV_SQL_JAVA_SHARED")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
    else {
        println!(
            "skipping Java-written shared-Puffin consume — set ICEBERG_INTEROP_DV_SQL_JAVA_SHARED"
        );
        return;
    };
    let metadata_location = current_hadoop_metadata(&java_dir.join("table").join("metadata"));
    assert!(
        metadata_location.is_file(),
        "missing Java Hadoop metadata at {}",
        metadata_location.display()
    );
    let out_dir = java_dir.join("after_delete");
    fs::create_dir_all(out_dir.join("rust_table").join("metadata")).expect("after_delete dir");
    let warehouse = java_dir.to_string_lossy().to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_dv_sql_java_shared",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse)]),
        )
        .await
        .expect("catalog");
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let ident = TableIdent::new(namespace.clone(), "rust_table".to_string());
    catalog
        .register_table(&ident, metadata_location.to_string_lossy().to_string())
        .await
        .expect("register Java table");
    let table = catalog.load_table(&ident).await.expect("load Java table");
    let tx = Transaction::new(&table);
    tx.update_table_properties()
        .set("write.delete.mode".to_string(), "merge-on-read".to_string())
        .set("write.update.mode".to_string(), "merge-on-read".to_string())
        .apply(tx)
        .expect("apply MoR properties")
        .commit(&catalog)
        .await
        .expect("commit MoR properties");
    let client = Arc::new(catalog);
    let provider = IcebergCatalogProvider::try_new(client.clone())
        .await
        .expect("provider");
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));
    run_sql(&ctx, "DELETE FROM catalog.interop.rust_table WHERE id = 10").await;
    let table = client.load_table(&ident).await.expect("load after delete");
    let expected_rows = "[\n  {\"id\": 30, \"data\": \"z\"},\n  {\"id\": 50, \"data\": \"q\"}\n]\n";
    fs::write(out_dir.join("expected_rows.json"), expected_rows).expect("expected_rows.json");
    let final_metadata_path = out_dir
        .join("rust_table")
        .join("metadata")
        .join("final.metadata.json");
    table
        .metadata()
        .write_to(
            table.file_io(),
            final_metadata_path.to_str().expect("utf8 metadata path"),
        )
        .await
        .expect("write final.metadata.json");
    println!(
        "interop_dv_sql Java-written shared-Puffin DELETE GEN OK → {}",
        final_metadata_path.display()
    );
}

fn decode_file_path(col: &ArrayRef, row: usize) -> String {
    if let Some(plain) = col.as_any().downcast_ref::<StringArray>() {
        return plain.value(row).to_string();
    }
    if let Some(run) = col.as_any().downcast_ref::<RunArray<Int32Type>>() {
        let values = run
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("_file REE values utf8");
        return values.value(run.get_physical_index(row)).to_string();
    }
    panic!("unexpected _file column type: {:?}", col.data_type());
}

async fn commit_shared_puffin_for_ids(table: &Table, catalog: &dyn Catalog, id_a: i64, id_b: i64) {
    let mut stream = table
        .scan()
        .select([
            "id".to_string(),
            RESERVED_COL_NAME_FILE.to_string(),
            RESERVED_COL_NAME_POS.to_string(),
        ])
        .build()
        .expect("scan")
        .to_arrow()
        .await
        .expect("arrow");
    let mut deletes = Vec::new();
    while let Some(batch) = stream.try_next().await.expect("batch") {
        let ids = batch
            .column_by_name("id")
            .expect("id")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("id i64");
        let file_col = batch.column_by_name(RESERVED_COL_NAME_FILE).expect("_file");
        let pos = batch
            .column_by_name(RESERVED_COL_NAME_POS)
            .expect("_pos")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("_pos i64");
        for row in 0..batch.num_rows() {
            let id = ids.value(row);
            if id == id_a || id == id_b {
                deletes.push((
                    decode_file_path(file_col, row),
                    u64::try_from(pos.value(row)).expect("pos"),
                ));
            }
        }
    }
    assert_eq!(
        deletes.len(),
        2,
        "need two row positions for the shared Puffin"
    );
    let data_files = {
        let mut files = HashMap::new();
        let snapshot = table.metadata().current_snapshot().expect("snapshot");
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .expect("manifest list");
        for manifest_file in manifest_list.entries() {
            if manifest_file.content != iceberg::spec::ManifestContentType::Data {
                continue;
            }
            let manifest = manifest_file
                .load_manifest(table.file_io())
                .await
                .expect("data manifest");
            for entry in manifest.entries() {
                if entry.is_alive() {
                    let file = entry.data_file().clone();
                    files.insert(file.file_path().to_string(), file);
                }
            }
        }
        files
    };
    let puffin = format!(
        "{}/data/shared-dv-{}.puffin",
        table.metadata().location(),
        uuid::Uuid::now_v7()
    );
    let mut writer =
        DVFileWriter::new(table.file_io().new_output(&puffin).expect("output")).unpartitioned();
    let schema = table.metadata().current_schema().clone();
    for (path, position) in &deletes {
        let data_file = data_files.get(path).expect("live data file");
        let spec = table
            .metadata()
            .partition_spec_by_id(data_file.partition_spec_id())
            .expect("spec")
            .as_ref()
            .clone();
        let key = PartitionKey::new(spec, schema.clone(), data_file.partition().clone())
            .expect("partition key");
        writer
            .delete(path, *position, Some(&key))
            .expect("record position");
    }
    let files = writer.close().await.expect("close shared puffin");
    let tx = Transaction::new(table);
    tx.row_delta()
        .add_deletes(files)
        .apply(tx)
        .expect("apply")
        .commit(catalog)
        .await
        .expect("commit shared puffin");
}
