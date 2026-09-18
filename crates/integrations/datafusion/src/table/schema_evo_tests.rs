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
use std::sync::Arc;

use datafusion::prelude::SessionContext;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use tempfile::TempDir;

use super::tests::*;
use super::*;

/// A provider built before an evolution advertises the OLD schema, while the scan reloads the
/// table and would otherwise `select_all()` the NEW column set.
#[tokio::test]
async fn test_stale_provider_scan_is_self_consistent() {
    let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
    let ident = TableIdent::new(namespace.clone(), table_name.clone());

    let stale_provider = Arc::new(
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
            .await
            .expect("construct the provider BEFORE the evolution"),
    );

    // The current snapshot carries 3 columns while `stale_provider` advertises 2.
    evolve_schema(&catalog, &ident, SchemaOp::AddOptionalInt("extra")).await;
    {
        let fresh =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .expect("construct a fresh provider for the out-of-band write");
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(fresh))
            .expect("register for the out-of-band write");
        ctx.sql("INSERT INTO t VALUES (1, 'a', 7)")
            .await
            .expect("plan the out-of-band insert")
            .collect()
            .await
            .expect("execute the out-of-band insert");
    }

    let ctx = SessionContext::new();
    ctx.register_table("t", stale_provider.clone() as Arc<dyn TableProvider>)
        .expect("register the stale provider");

    let batches = ctx
        .sql("SELECT * FROM t")
        .await
        .expect("plan SELECT * through the stale provider")
        .collect()
        .await
        .expect("execute SELECT * through the stale provider");
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 1, "the row written out of band must be visible");
    for batch in &batches {
        assert_eq!(
            batch.num_columns(),
            2,
            "the batch must match the 2-column schema the plan advertised, got {:?}",
            batch.schema()
        );
        assert!(
            batch.column_by_name("extra").is_none(),
            "a column the plan never advertised must not appear in the batch"
        );
    }

    // The advertised schema does not move, so a second query through the SAME provider is
    // planned and answered identically.
    assert_eq!(
        stale_provider.schema().fields().len(),
        2,
        "the advertised schema must not move under the plans built on it"
    );
    let batches = ctx
        .sql("SELECT * FROM t")
        .await
        .expect("plan the follow-up SELECT *")
        .collect()
        .await
        .expect("execute the follow-up SELECT *");
    for batch in &batches {
        assert_eq!(batch.num_columns(), 2, "got {:?}", batch.schema());
    }
}

/// The silent path: with NO projection the scan once asked the reloaded table for
/// `select_all()`. DataFusion addresses batch columns by ordinal, so an extra one corrupts.
#[tokio::test]
async fn test_unprojected_scan_advertises_the_schema_it_emits() {
    let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
    let ident = TableIdent::new(namespace.clone(), table_name.clone());

    let stale_provider =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
            .await
            .expect("construct the provider BEFORE the evolution");

    evolve_schema(&catalog, &ident, SchemaOp::AddOptionalInt("extra")).await;
    {
        let fresh =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .expect("construct a fresh provider for the out-of-band write");
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(fresh))
            .expect("register for the out-of-band write");
        ctx.sql("INSERT INTO t VALUES (1, 'a', 7)")
            .await
            .expect("plan the out-of-band insert")
            .collect()
            .await
            .expect("execute the out-of-band insert");
    }

    let ctx = SessionContext::new();
    let state = ctx.state();
    // `projection: None` is the path that once became `select_all()`.
    let plan = stale_provider
        .scan(&state, None, &[], None)
        .await
        .expect("plan an unprojected scan through the stale provider");
    let advertised = plan.schema();
    assert_eq!(
        advertised.fields().len(),
        2,
        "the plan advertises the schema it was planned against"
    );

    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx())
        .await
        .expect("execute the unprojected scan");
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 1, "the out-of-band row must be visible");
    for batch in &batches {
        assert_eq!(
            batch.schema(),
            advertised,
            "an emitted batch must match the schema the plan advertised"
        );
    }
}

/// Builds a table whose snapshot carries an out-of-band `extra` column, plus a provider
/// constructed BEFORE that evolution.
async fn stale_provider_over_evolved_table(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
    table_name: &str,
) -> IcebergTableProvider {
    let ident = TableIdent::new(namespace.clone(), table_name.to_string());
    let stale_provider =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.to_string())
            .await
            .expect("construct the provider BEFORE the evolution");

    evolve_schema(catalog, &ident, SchemaOp::AddOptionalInt("extra")).await;

    let fresh =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.to_string())
            .await
            .expect("construct a fresh provider for the out-of-band write");
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(fresh))
        .expect("register for the out-of-band write");
    ctx.sql("INSERT INTO t VALUES (1, 'a', 7), (2, 'b', 8)")
        .await
        .expect("plan the out-of-band insert")
        .collect()
        .await
        .expect("execute the out-of-band insert");

    stale_provider
}

fn dml_row_count(batches: &[datafusion::arrow::array::RecordBatch]) -> u64 {
    batches
        .iter()
        .map(|batch| {
            let array = batch
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::UInt64Array>()
                .expect("the DML result column must be UInt64");
            (0..array.len()).map(|i| array.value(i)).sum::<u64>()
        })
        .sum()
}

/// `delete_from` binds its row filter and projection base to the CURRENT schema. A filter over
/// a column added out of band cannot bind against the cached one.
#[tokio::test]
async fn test_delete_binds_to_current_schema_not_the_cached_one() {
    use datafusion::prelude::{col, lit};

    let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
    let stale_provider = stale_provider_over_evolved_table(&catalog, &namespace, &table_name).await;

    let ctx = SessionContext::new();
    let state = ctx.state();
    let plan = stale_provider
        .delete_from(&state, vec![col("extra").eq(lit(7))])
        .await
        .expect("plan a DELETE filtered on the out-of-band column");
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx())
        .await
        .expect("execute the DELETE");
    assert_eq!(
        dml_row_count(&batches),
        1,
        "exactly the row matching `extra = 7` must be deleted"
    );
}

/// `update` resolves each `SET` target against the CURRENT schema, so the column index and the
/// projection base describe one state. The cached schema calls a new column unknown.
#[tokio::test]
async fn test_update_binds_to_current_schema_not_the_cached_one() {
    use datafusion::prelude::{col, lit};

    let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
    let stale_provider = stale_provider_over_evolved_table(&catalog, &namespace, &table_name).await;

    let ctx = SessionContext::new();
    let state = ctx.state();
    let plan = stale_provider
        .update(&state, vec![("extra".to_string(), lit(99))], vec![
            col("id").eq(lit(1)),
        ])
        .await
        .expect("plan an UPDATE assigning to the out-of-band column");
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx())
        .await
        .expect("execute the UPDATE");
    assert_eq!(
        dml_row_count(&batches),
        1,
        "exactly the row matching `id = 1` must be updated"
    );
}

/// `RENAME COLUMN` keeps the field id and creates NO snapshot, so the advertised name and the
/// snapshot's name differ. Name binding null-fills over live data; field-id binding does not.
#[tokio::test]
async fn test_rename_preserves_values_under_the_new_name() {
    let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
    let ident = TableIdent::new(namespace.clone(), table_name.clone());

    evolve_schema(&catalog, &ident, SchemaOp::AddOptionalInt("opt")).await;
    seed(
        &catalog,
        &namespace,
        &table_name,
        "INSERT INTO t VALUES (1, 'a', 7)",
    )
    .await;

    // The rename lands AFTER the write, so the data sits under the old name.
    evolve_schema(&catalog, &ident, SchemaOp::Rename("opt", "opt2")).await;

    let provider =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
            .await
            .expect("construct a provider on the renamed table");
    let batches = query_through(Arc::new(provider), "SELECT * FROM t").await;

    let mut seen = 0;
    for batch in &batches {
        let renamed = batch
            .column_by_name("opt2")
            .expect("the renamed column must be present under its NEW name");
        assert_eq!(
            renamed.null_count(),
            0,
            "the renamed column must carry its data, not NULLs: {:?}",
            batch.schema()
        );
        let values = renamed
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int32Array>()
            .expect("opt2 must be Int32");
        for i in 0..values.len() {
            assert_eq!(
                values.value(i),
                7,
                "the stored value must survive the rename"
            );
            seen += 1;
        }
    }
    assert_eq!(seen, 1, "the seeded row must be readable");
}

/// A renamed REQUIRED column has no null-fill escape, so name binding fails the query outright.
#[tokio::test]
async fn test_required_column_rename_preserves_values() {
    let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
    let ident = TableIdent::new(namespace.clone(), table_name.clone());

    seed(
        &catalog,
        &namespace,
        &table_name,
        "INSERT INTO t VALUES (1, 'a')",
    )
    .await;
    // `name` is REQUIRED in the fixture schema.
    evolve_schema(&catalog, &ident, SchemaOp::Rename("name", "full_name")).await;

    let provider =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
            .await
            .expect("construct a provider on the renamed table");
    let batches = query_through(Arc::new(provider), "SELECT * FROM t").await;

    let mut seen = 0;
    for batch in &batches {
        use datafusion::arrow::array::Array;
        let renamed = batch
            .column_by_name("full_name")
            .expect("the renamed required column must be present")
            .as_any()
            .downcast_ref::<datafusion::arrow::array::StringArray>()
            .expect("full_name must be Utf8");
        for i in 0..renamed.len() {
            assert_eq!(renamed.value(i), "a");
            seen += 1;
        }
    }
    assert_eq!(seen, 1, "the seeded row must be readable");
}

/// A VIEW captures the provider and its projection ORDINALS. A schema that shrank under it
/// makes `TableScan::try_new` index it with a stale ordinal and PANIC.
#[tokio::test]
async fn test_view_survives_an_out_of_band_column_drop() {
    let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
    let ident = TableIdent::new(namespace.clone(), table_name.clone());

    seed(
        &catalog,
        &namespace,
        &table_name,
        "INSERT INTO t VALUES (1, 'a')",
    )
    .await;

    let provider =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
            .await
            .expect("construct the provider the view will capture");
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(provider))
        .expect("register the table");
    ctx.sql("CREATE VIEW v AS SELECT name FROM t")
        .await
        .expect("plan CREATE VIEW")
        .collect()
        .await
        .expect("create the view");

    // The drop lands BEFORE round 1, so a provider that republished its schema there leaves
    // round 2 indexing a SHORTER schema with round-1 ordinals, which panics.
    evolve_schema(&catalog, &ident, SchemaOp::Drop("id")).await;

    let round1 = ctx
        .sql("SELECT * FROM v")
        .await
        .expect("plan round 1")
        .collect()
        .await
        .expect("execute round 1");
    assert_eq!(round1.iter().map(|b| b.num_rows()).sum::<usize>(), 1);

    let round2 = ctx
        .sql("SELECT * FROM v")
        .await
        .expect("plan round 2 (must not panic)")
        .collect()
        .await
        .expect("execute round 2 (must not panic)");
    assert_eq!(
        round2.iter().map(|b| b.num_rows()).sum::<usize>(),
        1,
        "the view must keep answering against the schema it was created with"
    );
    for batch in &round2 {
        assert_eq!(batch.num_columns(), 1);
        assert!(batch.column_by_name("name").is_some());
    }
}

/// A DataFrame collected after an evolution holds planning-time ordinals, and must not panic.
#[tokio::test]
async fn test_deferred_dataframe_survives_an_out_of_band_evolution() {
    let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
    let ident = TableIdent::new(namespace.clone(), table_name.clone());

    seed(
        &catalog,
        &namespace,
        &table_name,
        "INSERT INTO t VALUES (1, 'a')",
    )
    .await;

    let provider =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
            .await
            .expect("construct the provider");
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(provider))
        .expect("register the table");

    let df = ctx.sql("SELECT name FROM t").await.expect("plan the query");

    evolve_schema(&catalog, &ident, SchemaOp::Drop("id")).await;

    // Execute twice: the second physical-planning round re-indexes the provider schema with
    // the plan's stored ordinals.
    let first = df
        .clone()
        .collect()
        .await
        .expect("the deferred plan must execute, not panic");
    assert_eq!(first.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    let second = df
        .collect()
        .await
        .expect("re-executing the deferred plan must not panic either");
    assert_eq!(second.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    for batch in &second {
        assert_eq!(batch.num_columns(), 1);
    }
}

/// `int` to `long` is a legal promotion and creates no snapshot, so the scanned data is still
/// `int` while the plan advertises `long`. The values must be read, widened.
#[tokio::test]
async fn test_legal_int_to_long_promotion_reads_widened_values() {
    let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
    let ident = TableIdent::new(namespace.clone(), table_name.clone());

    seed(
        &catalog,
        &namespace,
        &table_name,
        "INSERT INTO t VALUES (42, 'a')",
    )
    .await;
    evolve_schema(&catalog, &ident, SchemaOp::PromoteToLong("id")).await;

    let provider =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
            .await
            .expect("construct a provider on the promoted table");
    assert_eq!(
        provider.schema().field(0).data_type(),
        &datafusion::arrow::datatypes::DataType::Int64,
        "the provider advertises the promoted type"
    );
    let batches = query_through(Arc::new(provider), "SELECT * FROM t").await;

    let mut seen = 0;
    for batch in &batches {
        let ids = batch
            .column_by_name("id")
            .expect("id must be present")
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .expect("id must be read as the PROMOTED Int64");
        for i in 0..ids.len() {
            assert_eq!(
                ids.value(i),
                42,
                "the stored value must survive the promotion"
            );
            seen += 1;
        }
    }
    assert_eq!(seen, 1, "the seeded row must be readable");
}

/// In the steady state the scanned batch's schema is IDENTICAL to the advertised one, metadata
/// included, so `conform_batch` rebuilds nothing. A reader change would move every scan onto
/// the rebuild path.
#[tokio::test]
async fn test_steady_state_batch_schema_is_identical_to_the_advertised_schema() {
    let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
    seed(
        &catalog,
        &namespace,
        &table_name,
        "INSERT INTO t VALUES (1, 'a')",
    )
    .await;

    let provider =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
            .await
            .expect("construct the provider");
    let ctx = SessionContext::new();
    let state = ctx.state();
    let plan = provider
        .scan(&state, None, &[], None)
        .await
        .expect("plan the scan");
    let advertised = plan.schema();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx())
        .await
        .expect("execute the scan");
    assert!(!batches.is_empty(), "the seeded row must produce a batch");
    for batch in &batches {
        assert_eq!(
            batch.schema(),
            advertised,
            "the reader's schema must be identical to the advertised one in the steady state"
        );
    }
}

/// The nested-evolution fixture `(id int, s struct<a int>)`. `s.a` is field 3, so an added
/// `s.b` takes field 4, as Iceberg assigns.
async fn get_test_catalog_and_struct_table() -> (Arc<dyn Catalog>, NamespaceIdent, String, TempDir)
{
    use iceberg::spec::StructType;

    let temp_dir = TempDir::new().expect("temp dir");
    let warehouse_path = temp_dir.path().to_str().expect("utf-8 path").to_string();
    let catalog = MemoryCatalogBuilder::default()
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
        )
        .await
        .expect("memory catalog");
    let namespace = NamespaceIdent::new("test_ns".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(
                2,
                "s",
                Type::Struct(StructType::new(vec![
                    NestedField::optional(3, "a", Type::Primitive(PrimitiveType::Int)).into(),
                ])),
            )
            .into(),
        ])
        .build()
        .expect("nested schema");

    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("nested_table".to_string())
                .location(format!("{warehouse_path}/nested_table"))
                .schema(schema)
                .properties(HashMap::new())
                .build(),
        )
        .await
        .expect("create nested table");

    (
        Arc::new(catalog),
        namespace,
        "nested_table".to_string(),
        temp_dir,
    )
}

/// Adds a NESTED column out of band: `ALTER TABLE ADD COLUMN <parent>.<name> int`.
async fn evolve_add_nested_column(
    catalog: &Arc<dyn Catalog>,
    ident: &TableIdent,
    parent: &str,
    name: &str,
) {
    use iceberg::transaction::{ApplyTransactionAction, Transaction};

    let table = catalog
        .load_table(ident)
        .await
        .expect("load table for out-of-band evolution");
    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .add_column_to(
            Some(parent),
            name,
            Type::Primitive(PrimitiveType::Int),
            None,
        )
        .apply(tx)
        .expect("queue the nested add-column");
    tx.commit(catalog.as_ref())
        .await
        .expect("commit the nested add-column");
}

/// `ADD COLUMN s.b` creates no snapshot, so the scanned struct holds only `a` while the plan
/// advertises `{a, b}`. Conforming must recurse into the struct, as Spark does.
#[tokio::test]
async fn test_nested_add_column_reads_null_for_the_new_field() {
    use datafusion::arrow::array::{Array, Int32Array, StructArray};

    let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_struct_table().await;
    let ident = TableIdent::new(namespace.clone(), table_name.clone());

    seed(
        &catalog,
        &namespace,
        &table_name,
        "INSERT INTO t VALUES (1, named_struct('a', 5))",
    )
    .await;

    evolve_add_nested_column(&catalog, &ident, "s", "b").await;

    let provider =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
            .await
            .expect("construct a provider on the nested-evolved table");
    let batches = query_through(Arc::new(provider), "SELECT * FROM t").await;

    let mut seen = 0;
    for batch in &batches {
        let structs = batch
            .column_by_name("s")
            .expect("the struct column must be present")
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("s must be a struct");
        assert_eq!(
            structs.num_columns(),
            2,
            "the struct must carry the advertised child set, got {:?}",
            structs.data_type()
        );
        let a = structs
            .column_by_name("a")
            .expect("s.a must be present")
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("s.a must be Int32");
        let b = structs.column_by_name("b").expect("s.b must be present");
        for i in 0..a.len() {
            assert_eq!(a.value(i), 5, "the stored nested value must survive");
            assert!(b.is_null(i), "a nested column added later reads as NULL");
            seen += 1;
        }
    }
    assert_eq!(seen, 1, "the seeded row must be readable");
}

/// Renames a column out of band; `name` may be a dotted path for a nested field.
async fn evolve_rename(catalog: &Arc<dyn Catalog>, ident: &TableIdent, name: &str, to: &str) {
    use iceberg::transaction::{ApplyTransactionAction, Transaction};

    let table = catalog
        .load_table(ident)
        .await
        .expect("load table for out-of-band evolution");
    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .rename_column(name, to)
        .apply(tx)
        .expect("queue the rename");
    tx.commit(catalog.as_ref())
        .await
        .expect("commit the rename");
}

/// The nested rename: the child keeps its field id, so its value comes back under the new name.
#[tokio::test]
async fn test_nested_rename_preserves_values() {
    use datafusion::arrow::array::{Array, Int32Array, StructArray};

    let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_struct_table().await;
    let ident = TableIdent::new(namespace.clone(), table_name.clone());

    seed(
        &catalog,
        &namespace,
        &table_name,
        "INSERT INTO t VALUES (1, named_struct('a', 5))",
    )
    .await;

    evolve_rename(&catalog, &ident, "s.a", "renamed_a").await;

    let provider =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
            .await
            .expect("construct a provider on the nested-renamed table");
    let batches = query_through(Arc::new(provider), "SELECT * FROM t").await;

    let mut seen = 0;
    for batch in &batches {
        let structs = batch
            .column_by_name("s")
            .expect("the struct column must be present")
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("s must be a struct");
        let renamed = structs
            .column_by_name("renamed_a")
            .expect("the renamed child must be present under its NEW name")
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("s.renamed_a must be Int32");
        for i in 0..renamed.len() {
            assert!(
                renamed.is_valid(i),
                "the renamed nested column must carry its data, not NULLs"
            );
            assert_eq!(renamed.value(i), 5);
            seen += 1;
        }
    }
    assert_eq!(seen, 1, "the seeded row must be readable");
}

// ===== Pushed-down-filter rebinding =====
//
// A pushed filter PRUNES rows before DataFusion sees them, and `Inexact` pushdown only discards
// false positives. These three pin how a name-keyed filter fails once names and ids disagree.

/// A table `(a int, b int)` for the name-swap case: one shared type, so a filter bound to the
/// wrong column type-checks and returns the wrong rows.
async fn get_test_catalog_and_two_int_table() -> (Arc<dyn Catalog>, NamespaceIdent, String, TempDir)
{
    let temp_dir = TempDir::new().expect("temp dir");
    let warehouse_path = temp_dir.path().to_str().expect("utf-8 path").to_string();
    let catalog = MemoryCatalogBuilder::default()
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
        )
        .await
        .expect("memory catalog");
    let namespace = NamespaceIdent::new("test_ns".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::optional(1, "a", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "b", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()
        .expect("two-int schema");

    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("swap_table".to_string())
                .location(format!("{warehouse_path}/swap_table"))
                .schema(schema)
                .properties(HashMap::new())
                .build(),
        )
        .await
        .expect("create two-int table");

    (
        Arc::new(catalog),
        namespace,
        "swap_table".to_string(),
        temp_dir,
    )
}

/// After a rename, a filter over the NEW name must go down under the snapshot's name. The
/// advertised name fails to bind, so the query dies instead of returning the row.
#[tokio::test]
async fn test_pushdown_after_a_rename_binds_the_snapshot_name() {
    let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
    let ident = TableIdent::new(namespace.clone(), table_name.clone());

    evolve_schema(&catalog, &ident, SchemaOp::AddOptionalInt("opt")).await;
    seed(
        &catalog,
        &namespace,
        &table_name,
        "INSERT INTO t VALUES (1, 'a', 7), (2, 'b', 8)",
    )
    .await;
    evolve_rename(&catalog, &ident, "opt", "opt2").await;

    let provider =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
            .await
            .expect("construct a provider on the renamed table");
    let batches = query_through(Arc::new(provider), "SELECT id FROM t WHERE opt2 = 7").await;
    assert_eq!(
        batches.iter().map(|b| b.num_rows()).sum::<usize>(),
        1,
        "exactly the row with opt2 = 7 must come back"
    );
}

/// After DROP and re-ADD the advertised column has a FRESH field id the snapshot lacks, so
/// every row reads NULL. Pushing under the old name prunes rows DataFusion cannot get back.
#[tokio::test]
async fn test_pushdown_after_drop_and_readd_keeps_the_rows() {
    let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
    let ident = TableIdent::new(namespace.clone(), table_name.clone());

    evolve_schema(&catalog, &ident, SchemaOp::AddOptionalInt("opt")).await;
    seed(
        &catalog,
        &namespace,
        &table_name,
        "INSERT INTO t VALUES (1, 'a', 7), (2, 'b', 8)",
    )
    .await;
    evolve_schema(&catalog, &ident, SchemaOp::Drop("opt")).await;
    evolve_schema(&catalog, &ident, SchemaOp::AddOptionalInt("opt")).await;

    let provider =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
            .await
            .expect("construct a provider on the re-added column");
    let batches = query_through(Arc::new(provider), "SELECT id FROM t WHERE opt IS NULL").await;
    assert_eq!(
        batches.iter().map(|b| b.num_rows()).sum::<usize>(),
        2,
        "the re-added column reads NULL for every row, so both must match"
    );
}

/// The silent one: swap two names, and a filter pushed under the advertised name binds to the
/// OTHER column's data. Same type, no error, wrong rows.
#[tokio::test]
async fn test_pushdown_after_a_name_swap_filters_the_right_column() {
    let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_two_int_table().await;
    let ident = TableIdent::new(namespace.clone(), table_name.clone());

    seed(
        &catalog,
        &namespace,
        &table_name,
        "INSERT INTO t VALUES (1, 2)",
    )
    .await;

    // a -> tmp, b -> a, tmp -> b: field 1 is now called `b` and field 2 is now called `a`.
    evolve_rename(&catalog, &ident, "a", "tmp").await;
    evolve_rename(&catalog, &ident, "b", "a").await;
    evolve_rename(&catalog, &ident, "tmp", "b").await;

    let provider =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
            .await
            .expect("construct a provider on the swapped table");
    // Only the NAMES moved. The column advertised as `a` is field id 2, which the snapshot
    // still calls `b`, holding the value 2.
    assert_eq!(provider.schema().field(0).name(), "b");
    assert_eq!(provider.schema().field(1).name(), "a");

    let batches = query_through(Arc::new(provider), "SELECT * FROM t WHERE a = 2").await;
    assert_eq!(
        batches.iter().map(|b| b.num_rows()).sum::<usize>(),
        1,
        "`a` now holds the value 2, so the row must match"
    );
}

/// The scan must read ONLY the projected column. It drives
/// [`crate::physical_plan::scan::get_batch_stream`] with the resolved column set, so a revert
/// to `select_all()` widens the pre-conform batch and REDs here. `conform_batch` would
/// otherwise hide it, because it narrows the batch again.
#[tokio::test]
async fn test_scan_reads_only_the_projected_column() {
    use futures::TryStreamExt;

    use crate::physical_plan::scan::get_batch_stream;

    let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
    seed(
        &catalog,
        &namespace,
        &table_name,
        "INSERT INTO t VALUES (1, 'a')",
    )
    .await;

    let table = catalog
        .load_table(&TableIdent::new(namespace.clone(), table_name.clone()))
        .await
        .expect("load the seeded table");
    let batches: Vec<_> = get_batch_stream(
        table,
        None,
        vec!["name".to_string()],
        None,
        crate::physical_plan::scan::ScanKnobs::default(),
    )
    .await
    .expect("open the scan stream")
    .try_collect()
    .await
    .expect("read the scan stream");

    assert!(!batches.is_empty(), "the seeded row must produce a batch");
    for batch in &batches {
        assert_eq!(
            batch.num_columns(),
            1,
            "only the projected column may be read, got {:?}",
            batch.schema()
        );
        assert_eq!(batch.schema().field(0).name(), "name");
    }
}

#[tokio::test]
async fn test_no_limit_pushdown() {
    use datafusion::datasource::TableProvider;

    let (_catalog, _ns, _name, table, _tmp) = get_static_test_table().await;
    let table_provider = IcebergStaticTableProvider::try_new_from_table(table)
        .await
        .unwrap();

    let ctx = SessionContext::new();
    let state = ctx.state();

    let scan_plan = table_provider.scan(&state, None, &[], None).await.unwrap();

    let iceberg_scan = scan_plan
        .downcast_ref::<IcebergTableScan>()
        .expect("Expected IcebergTableScan");

    assert_eq!(
        iceberg_scan.limit(),
        None,
        "Limit should be None when not specified"
    );
}

/// Incomplete metadata must surface a planning error, not demote to `UnknownPartitioning(1)`.
#[tokio::test]
async fn test_plan_tasks_failure_fail_closed_not_n1_demote() {
    use datafusion::datasource::TableProvider;

    let table = get_test_table_from_metadata_file().await;
    let table_provider = IcebergStaticTableProvider::try_new_from_table(table)
        .await
        .unwrap();
    let ctx = SessionContext::new();
    let state = ctx.state();
    let err = table_provider
        .scan(&state, None, &[], None)
        .await
        .expect_err("incomplete fixture must fail plan, not demote to N=1");
    let msg = err.to_string();
    assert!(
        msg.contains("manifest")
            || msg.contains("file")
            || msg.contains("Failed")
            || msg.contains("not found")
            || msg.contains("No such"),
        "expected planning/IO root cause, got: {msg}"
    );
}
