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

//! U4 — Partitioned copy-on-write DML interop: Rust writes via DataFusion SQL DML → Java reads.
//!
//! # Scenario (MUST-HAVE): partitioned COW DELETE
//!
//! Table schema: `{id int, category string, value string}` partitioned by `identity(category)`.
//! Two partitions are inserted:
//!   * `electronics`: ids 1 ("laptop"), 2 ("phone")
//!   * `books`: ids 3 ("novel"), 4 ("textbook")
//!
//! SQL `DELETE FROM … WHERE category = 'electronics'` runs in copy-on-write mode (no
//! `write.delete.mode` property → COW). The `electronics` data file is entirely removed from the
//! manifest; the `books` data file is left untouched.
//!
//! Expected survivors: ids 3 ("novel") and 4 ("textbook") in the `books` partition.
//!
//! The test writes the mutated table under `<gen_dir>/rust_table` and lands a canonical
//! `final.metadata.json` so the Java oracle (`verify-interop-part-dml`) can load it, read it with
//! `IcebergGenerics`, and assert the exact survivor set.
//!
//! # ENV gate
//!
//! When `ICEBERG_INTEROP_PART_DML_GEN_DIR` is **unset** the test returns immediately — a clean
//! no-op that keeps the offline `cargo test` gate green without any Java/Maven dependency.
//! The `dev/java-interop/run-interop-partitioned-dml.sh` script sets the variable and runs the
//! end-to-end chain.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use datafusion::arrow::array::{Array, Int32Array, StringArray, UInt64Array};
use datafusion::execution::context::SessionContext;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{NestedField, PrimitiveType, Schema, Transform, Type, UnboundPartitionSpec};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;

// ===========================================================================
// ENV gate
// ===========================================================================

/// Return the GEN dir from the environment variable, or `None` when unset.
fn part_dml_gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_PART_DML_GEN_DIR").map(PathBuf::from)
}

// ===========================================================================
// U4 GEN test — partitioned COW DELETE: Rust writes via SQL DML → Java reads
// ===========================================================================

/// GEN direction for U4: build a partitioned `identity(category)` V2 table, insert rows into
/// two partitions, run a SQL `DELETE FROM … WHERE category = 'electronics'` (copy-on-write),
/// reload the table from the catalog, and write `final.metadata.json` so Java can verify it.
///
/// **When `ICEBERG_INTEROP_PART_DML_GEN_DIR` is unset this test is a clean no-op.**
#[tokio::test]
async fn test_part_dml_gen_rust_writes_java_readable_partitioned_cow_table() {
    let Some(gen_dir) = part_dml_gen_dir() else {
        println!(
            "skipping interop_partitioned_dml GEN — set ICEBERG_INTEROP_PART_DML_GEN_DIR \
             (run dev/java-interop/run-interop-partitioned-dml.sh)"
        );
        return;
    };

    // ------------------------------------------------------------------
    // 1. MemoryCatalog over the LOCAL FS, warehouse = <gen_dir>.
    //    Fixed table location: <gen_dir>/rust_table — deterministic for Java.
    //    The namespace and table are created BEFORE building the DataFusion
    //    IcebergCatalogProvider, matching the pattern used in integration tests
    //    (the provider snapshots the catalog at construction time).
    // ------------------------------------------------------------------
    let warehouse = gen_dir.to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");

    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_part_dml",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS for part-dml interop");

    // ------------------------------------------------------------------
    // 2. Create namespace + the partitioned table at the fixed location.
    //    identity(category) partition, no write.delete.mode → COW.
    // ------------------------------------------------------------------
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "category", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "value", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build schema");

    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "category", Transform::Identity)
        .expect("add identity(category) partition field")
        .build();

    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location.clone())
        .schema(schema)
        .partition_spec(partition_spec)
        .properties(HashMap::new()) // no write.delete.mode → COW
        .build();

    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create rust_table");

    // ------------------------------------------------------------------
    // 3. Build the DataFusion catalog provider + SessionContext.
    //    The provider is built AFTER table creation so the table is visible.
    // ------------------------------------------------------------------
    let client = Arc::new(catalog);
    let provider = IcebergCatalogProvider::try_new(client.clone())
        .await
        .expect("build IcebergCatalogProvider");
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));

    // ------------------------------------------------------------------
    // 4. INSERT rows into TWO partitions via DataFusion SQL.
    //    electronics: ids 1/2 ("laptop"/"phone")
    //    books:       ids 3/4 ("novel"/"textbook")
    // ------------------------------------------------------------------
    let insert_batches = ctx
        .sql(
            "INSERT INTO catalog.interop.rust_table VALUES \
             (1, 'electronics', 'laptop'), \
             (2, 'electronics', 'phone'), \
             (3, 'books', 'novel'), \
             (4, 'books', 'textbook')",
        )
        .await
        .expect("INSERT SQL plan")
        .collect()
        .await
        .expect("collect INSERT result");
    let inserted: u64 = insert_batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("insert count column")
        .value(0);
    assert_eq!(inserted, 4, "INSERT must write exactly 4 rows");

    // ------------------------------------------------------------------
    // 5. SQL DELETE WHERE category = 'electronics' — copy-on-write path.
    //    The electronics partition file is dropped; books partition
    //    file is left entirely untouched in the manifest.
    // ------------------------------------------------------------------
    let delete_batches = ctx
        .sql("DELETE FROM catalog.interop.rust_table WHERE category = 'electronics'")
        .await
        .expect("DELETE SQL plan")
        .collect()
        .await
        .expect("collect DELETE result");
    let deleted: u64 = delete_batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("delete count column")
        .value(0);
    assert_eq!(
        deleted, 2,
        "DELETE must report exactly 2 electronics rows deleted"
    );

    // ------------------------------------------------------------------
    // 6. Sanity: re-read the table via DataFusion to confirm survivors.
    //    Expected: ids {3, 4} in books partition only.
    // ------------------------------------------------------------------
    let select_batches = ctx
        .sql("SELECT id, category, value FROM catalog.interop.rust_table ORDER BY id")
        .await
        .expect("SELECT SQL plan")
        .collect()
        .await
        .expect("collect SELECT result");
    let total_rows: usize = select_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 2, "exactly 2 rows survive after COW DELETE");

    // Verify the surviving ids are {3, 4}.
    let mut surviving_ids: Vec<i32> = Vec::new();
    for batch in &select_batches {
        let id_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("id column is Int32");
        for i in 0..id_col.len() {
            surviving_ids.push(id_col.value(i));
        }
    }
    surviving_ids.sort();
    assert_eq!(
        surviving_ids,
        vec![3, 4],
        "COW DELETE must leave exactly ids {{3,4}} (books partition)"
    );

    // Verify categories and values for completeness.
    let mut surviving_cats: Vec<String> = Vec::new();
    let mut surviving_vals: Vec<String> = Vec::new();
    for batch in &select_batches {
        let cat_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("category column is Utf8");
        let val_col = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("value column is Utf8");
        for i in 0..batch.num_rows() {
            surviving_cats.push(cat_col.value(i).to_string());
            surviving_vals.push(val_col.value(i).to_string());
        }
    }
    assert!(
        surviving_cats.iter().all(|c| c == "books"),
        "all surviving rows must be in the books partition"
    );
    assert_eq!(
        surviving_vals,
        vec!["novel", "textbook"],
        "surviving values must be {{novel, textbook}} in id order"
    );

    // ------------------------------------------------------------------
    // 7. Reload the table from the catalog and write final.metadata.json
    //    at the deterministic path Java reads.
    // ------------------------------------------------------------------
    let tbl_ident = TableIdent::new(namespace, "rust_table".to_string());
    let table = client
        .load_table(&tbl_ident)
        .await
        .expect("reload rust_table after DML");

    let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), &final_metadata_path)
        .await
        .expect("write final.metadata.json");

    println!(
        "interop_partitioned_dml GEN OK — Rust wrote {table_location} (partitioned COW DELETE: \
         2 partitions inserted, electronics deleted, books partition untouched); \
         survivor ids = {{3, 4}} (books: novel, textbook). \
         final.metadata.json → {final_metadata_path}. \
         Java verify-interop-part-dml reads it next."
    );
}
