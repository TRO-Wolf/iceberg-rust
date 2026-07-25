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

//! Partitioned `INSERT INTO … SELECT` correctness: the MANIFEST partition tuple
//! must be computed from the values actually written, not from whatever batch
//! DataFusion's `ProjectionPushdown` re-parented the partition expression onto.
//!
//! Every test here asserts the committed `DataFile.partition()` tuple through the
//! core API (manifest-level truth). A `WHERE`-count assertion alone is forbidden
//! for this defect class: it goes falsely green whenever scan pruning changes.
//!
//! Test map (task/back-to-goal-2026-07-25-brief.md, Unit 1):
//! * T2 — `INSERT … SELECT CASE WHEN … THEN NULL` → the tuple slot must be NULL.
//! * T3 — same with a non-NULL divergent value (`'zzz'`) — kills the
//!   "just add a null check" false fix.
//! * T4 — same-typed column permutation (`SELECT b, a`) — the tuple must carry
//!   the value written to the partition-source column, not the same-named source
//!   column of the scan batch.
//! * T5 — plain passthrough `SELECT` control (green before the fix too).
//! * T6 — `VALUES` control (green before the fix too).
//! * T10 — FROM-less literal `INSERT INTO … SELECT <literals>` (no `FROM`):
//!   plans over `PlaceholderRowExec` (1 row, 0 columns); panicked before the fix,
//!   must succeed with the correct tuple after it.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{Int32Array, RecordBatch, StringArray, UInt64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    Literal, NestedField, PrimitiveLiteral, PrimitiveType, Schema, Transform, Type,
    UnboundPartitionSpec,
};
use iceberg::{
    Catalog, CatalogBuilder, MemoryCatalog, NamespaceIdent, Result, TableCreation, TableIdent,
};
use iceberg_datafusion::IcebergCatalogProvider;
use tempfile::TempDir;

// ===========================================================================
// Fixture helpers
// ===========================================================================

fn temp_path() -> String {
    let temp_dir = TempDir::new().expect("create temp dir");
    temp_dir
        .path()
        .to_str()
        .expect("temp dir path is valid UTF-8")
        .to_string()
}

async fn get_iceberg_catalog() -> MemoryCatalog {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), temp_path())]),
        )
        .await
        .expect("build memory catalog")
}

/// Create `catalog.<ns>.t` = `{id int required, category string optional,
/// value string required}` partitioned by `identity(category)`, plus a plain
/// DataFusion `src` table `{id, category, value}` with rows
/// `(1, 'books', 'x')` and `(2, 'electronics', 'y')`.
async fn make_case_ctx(ns: &str) -> Result<(SessionContext, Arc<MemoryCatalog>, TableIdent)> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new(ns.to_string());
    iceberg_catalog
        .create_namespace(&namespace, HashMap::new())
        .await?;

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "category", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "value", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()?;
    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "category", Transform::Identity)?
        .build();
    let creation = TableCreation::builder()
        .name("t".to_string())
        .location(temp_path())
        .schema(schema)
        .partition_spec(partition_spec)
        .properties(HashMap::new())
        .build();
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let provider = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", provider);

    // Plain DataFusion source table with the same column names/types.
    // `category` is nullable to line up with the optional target column.
    let src_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
        Field::new("value", DataType::Utf8, false),
    ]));
    let src_batch = RecordBatch::try_new(src_schema.clone(), vec![
        Arc::new(Int32Array::from(vec![1, 2])),
        Arc::new(StringArray::from(vec![Some("books"), Some("electronics")])),
        Arc::new(StringArray::from(vec!["x", "y"])),
    ])
    .expect("build src batch");
    let src = MemTable::try_new(src_schema, vec![vec![src_batch]]).expect("build src MemTable");
    ctx.register_table("src", Arc::new(src))
        .expect("register src table");

    let ident = TableIdent::new(namespace, "t".to_string());
    Ok((ctx, client, ident))
}

/// Create `catalog.<ns>.t` = `{a string required (partition source), b string
/// required}` partitioned by `identity(a)`, plus a `src` table `{a, b}` with the
/// single row `('A1', 'B1')`.
async fn make_permutation_ctx(
    ns: &str,
) -> Result<(SessionContext, Arc<MemoryCatalog>, TableIdent)> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new(ns.to_string());
    iceberg_catalog
        .create_namespace(&namespace, HashMap::new())
        .await?;

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "a", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(2, "b", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()?;
    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(1, "a", Transform::Identity)?
        .build();
    let creation = TableCreation::builder()
        .name("t".to_string())
        .location(temp_path())
        .schema(schema)
        .partition_spec(partition_spec)
        .properties(HashMap::new())
        .build();
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let provider = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", provider);

    let src_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("a", DataType::Utf8, false),
        Field::new("b", DataType::Utf8, false),
    ]));
    let src_batch = RecordBatch::try_new(src_schema.clone(), vec![
        Arc::new(StringArray::from(vec!["A1"])),
        Arc::new(StringArray::from(vec!["B1"])),
    ])
    .expect("build src batch");
    let src = MemTable::try_new(src_schema, vec![vec![src_batch]]).expect("build src MemTable");
    ctx.register_table("src", Arc::new(src))
        .expect("register src table");

    let ident = TableIdent::new(namespace, "t".to_string());
    Ok((ctx, client, ident))
}

/// Read the committed manifest through the core API and return the sorted list
/// of single-field partition tuples (`None` = NULL slot) of all live data
/// files, plus the summed record count.
async fn manifest_partition_tuples(
    client: &Arc<MemoryCatalog>,
    ident: &TableIdent,
) -> Result<(Vec<Option<String>>, u64)> {
    let table = client.load_table(ident).await?;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("table has a committed snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await?;

    let mut tuples: Vec<Option<String>> = Vec::new();
    let mut total_records: u64 = 0;
    for mf in manifest_list.entries() {
        let manifest = mf.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if entry.is_alive() {
                let partition = entry.data_file().partition();
                assert_eq!(
                    partition.fields().len(),
                    1,
                    "single-field partition spec expected"
                );
                let slot = match partition.fields()[0].as_ref() {
                    Some(Literal::Primitive(PrimitiveLiteral::String(s))) => Some(s.clone()),
                    None => None,
                    other => panic!("unexpected partition literal kind: {other:?}"),
                };
                tuples.push(slot);
                total_records += entry.data_file().record_count();
            }
        }
    }
    tuples.sort();
    Ok((tuples, total_records))
}

/// Run an INSERT statement and return the reported row count.
async fn run_insert(ctx: &SessionContext, sql: &str) -> u64 {
    let batches = ctx
        .sql(sql)
        .await
        .expect("plan INSERT statement")
        .collect()
        .await
        .expect("execute INSERT statement");
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("insert count column")
        .value(0)
}

// ===========================================================================
// T2 — CASE WHEN … THEN NULL: the manifest tuple slot must be NULL
// ===========================================================================

/// T2: `INSERT INTO t SELECT id, CASE WHEN id = 2 THEN NULL ELSE category END,
/// value FROM src`. Row 2's partition-source value is NULL, so its manifest
/// partition tuple must carry a NULL slot; row 1 stays in `books`.
#[tokio::test]
async fn test_insert_select_case_null_partition_source_manifest_tuple() -> Result<()> {
    let (ctx, client, ident) = make_case_ctx("t2_case_null").await?;

    let inserted = run_insert(
        &ctx,
        "INSERT INTO catalog.t2_case_null.t \
         SELECT id, CASE WHEN id = 2 THEN NULL ELSE category END AS category, value FROM src",
    )
    .await;
    assert_eq!(inserted, 2, "INSERT must report 2 rows written");

    let (tuples, total_records) = manifest_partition_tuples(&client, &ident).await?;
    assert_eq!(total_records, 2, "manifest record counts must sum to 2");
    // Option sorts None first.
    assert_eq!(
        tuples,
        vec![None, Some("books".to_string())],
        "manifest partition tuples must be computed from the projected CASE \
         values: one NULL slot (id=2) and one 'books' slot (id=1)"
    );
    Ok(())
}

// ===========================================================================
// T3 — CASE WHEN … THEN 'zzz': non-NULL divergent value (null-check killer)
// ===========================================================================

/// T3: same shape as T2 but the divergent branch produces the non-NULL value
/// `'zzz'`. This is the test that kills the "just add a null check" false fix:
/// the corrupted tuple (`electronics`) is non-NULL and perfectly well-formed.
#[tokio::test]
async fn test_insert_select_case_zzz_partition_source_manifest_tuple() -> Result<()> {
    let (ctx, client, ident) = make_case_ctx("t3_case_zzz").await?;

    let inserted = run_insert(
        &ctx,
        "INSERT INTO catalog.t3_case_zzz.t \
         SELECT id, CASE WHEN id = 2 THEN 'zzz' ELSE category END AS category, value FROM src",
    )
    .await;
    assert_eq!(inserted, 2, "INSERT must report 2 rows written");

    let (tuples, total_records) = manifest_partition_tuples(&client, &ident).await?;
    assert_eq!(total_records, 2, "manifest record counts must sum to 2");
    assert_eq!(
        tuples,
        vec![Some("books".to_string()), Some("zzz".to_string())],
        "manifest partition tuples must be computed from the projected CASE \
         values: 'books' (id=1) and 'zzz' (id=2) — never the raw source column"
    );
    Ok(())
}

// ===========================================================================
// T4 — same-typed column permutation: SELECT b, a
// ===========================================================================

/// T4: `INSERT INTO t SELECT b, a FROM src` writes src.b into table column `a`
/// (the partition source) and src.a into table column `b`. The manifest tuple
/// must carry the value actually written to `a` (`'B1'`), not the value of the
/// same-named column of the scan batch (`'A1'`).
#[tokio::test]
async fn test_insert_select_permuted_columns_manifest_tuple() -> Result<()> {
    let (ctx, client, ident) = make_permutation_ctx("t4_permutation").await?;

    let inserted = run_insert(
        &ctx,
        "INSERT INTO catalog.t4_permutation.t SELECT b, a FROM src",
    )
    .await;
    assert_eq!(inserted, 1, "INSERT must report 1 row written");

    let (tuples, total_records) = manifest_partition_tuples(&client, &ident).await?;
    assert_eq!(total_records, 1, "manifest record counts must sum to 1");
    assert_eq!(
        tuples,
        vec![Some("B1".to_string())],
        "identity(a) partition tuple must be src.b's value 'B1' (the value \
         written to column a), not src.a's value 'A1'"
    );

    // The written data row must be (a='B1', b='A1') — tuple and data agree.
    let rows = ctx
        .sql("SELECT a, b FROM catalog.t4_permutation.t")
        .await
        .expect("plan SELECT")
        .collect()
        .await
        .expect("execute SELECT");
    assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    let a_col = rows[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("a column is Utf8");
    let b_col = rows[0]
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("b column is Utf8");
    assert_eq!(a_col.value(0), "B1", "column a must hold src.b's value");
    assert_eq!(b_col.value(0), "A1", "column b must hold src.a's value");
    Ok(())
}

// ===========================================================================
// T5 — plain passthrough SELECT control (green before the fix too)
// ===========================================================================

/// T5 control: a plain `INSERT INTO t SELECT id, category, value FROM src`
/// (no computed items, no permutation) must produce correct tuples — and did
/// so even before the fix (the re-parented batch was positionally identical).
#[tokio::test]
async fn test_insert_select_passthrough_control_manifest_tuple() -> Result<()> {
    let (ctx, client, ident) = make_case_ctx("t5_passthrough").await?;

    let inserted = run_insert(
        &ctx,
        "INSERT INTO catalog.t5_passthrough.t SELECT id, category, value FROM src",
    )
    .await;
    assert_eq!(inserted, 2, "INSERT must report 2 rows written");

    let (tuples, total_records) = manifest_partition_tuples(&client, &ident).await?;
    assert_eq!(total_records, 2, "manifest record counts must sum to 2");
    assert_eq!(
        tuples,
        vec![Some("books".to_string()), Some("electronics".to_string())],
        "passthrough insert must land one file per source category"
    );
    Ok(())
}

// ===========================================================================
// T6 — VALUES control (green before the fix too)
// ===========================================================================

/// T6 control: `INSERT INTO t VALUES …` plans over a leaf values node (no
/// SELECT-list projection to fuse with), so it was correct even before the fix.
#[tokio::test]
async fn test_insert_values_control_manifest_tuple() -> Result<()> {
    let (ctx, client, ident) = make_case_ctx("t6_values").await?;

    let inserted = run_insert(
        &ctx,
        "INSERT INTO catalog.t6_values.t VALUES \
         (1, 'books', 'x'), (2, 'electronics', 'y')",
    )
    .await;
    assert_eq!(inserted, 2, "INSERT must report 2 rows written");

    let (tuples, total_records) = manifest_partition_tuples(&client, &ident).await?;
    assert_eq!(total_records, 2, "manifest record counts must sum to 2");
    assert_eq!(
        tuples,
        vec![Some("books".to_string()), Some("electronics".to_string())],
        "VALUES insert must land one file per category"
    );
    Ok(())
}

// ===========================================================================
// T10 — FROM-less literal INSERT (panicked before the fix)
// ===========================================================================

/// T10: `INSERT INTO t SELECT 1, 'books', 'x'` (no `FROM`) plans the literals
/// over `PlaceholderRowExec` — a 1-row, ZERO-column batch. Before the fix the
/// fused partition expression read that batch positionally and panicked in
/// `record_batch_projector`; after the fix the substituted literal children
/// evaluate correctly and the manifest tuple is `'books'`.
///
/// The table uses REQUIRED fields: literal SELECT items are non-nullable, and
/// `project_with_partition`'s pre-existing strict schema validation (nullability
/// included) must pass for the plan to reach the partition machinery at all.
/// (On an optional-column table the same statement is rejected by that
/// validation today — a separate, loud, pre-existing provider limitation.)
#[tokio::test]
async fn test_insert_fromless_literal_select_partitioned() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("t10_fromless".to_string());
    iceberg_catalog
        .create_namespace(&namespace, HashMap::new())
        .await?;

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "category", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "value", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()?;
    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "category", Transform::Identity)?
        .build();
    let creation = TableCreation::builder()
        .name("t".to_string())
        .location(temp_path())
        .schema(schema)
        .partition_spec(partition_spec)
        .properties(HashMap::new())
        .build();
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let provider = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", provider);

    let inserted = run_insert(
        &ctx,
        "INSERT INTO catalog.t10_fromless.t SELECT 1, 'books', 'x'",
    )
    .await;
    assert_eq!(inserted, 1, "INSERT must report 1 row written");

    let ident = TableIdent::new(namespace, "t".to_string());
    let (tuples, total_records) = manifest_partition_tuples(&client, &ident).await?;
    assert_eq!(total_records, 1, "manifest record counts must sum to 1");
    assert_eq!(
        tuples,
        vec![Some("books".to_string())],
        "FROM-less literal insert must land in the 'books' partition"
    );
    Ok(())
}
