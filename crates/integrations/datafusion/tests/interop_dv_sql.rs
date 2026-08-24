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

use datafusion::execution::context::SessionContext;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataFileFormat, FormatVersion, ManifestContentType, NestedField, PrimitiveType, Schema,
    Transform, Type, UnboundPartitionSpec,
};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;

/// Return the GEN dir from the environment variable, or `None` when unset.
fn dv_sql_gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_DV_SQL_GEN_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
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
