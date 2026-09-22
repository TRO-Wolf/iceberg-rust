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

use datafusion::arrow::array::{Array, AsArray};
use datafusion::assert_batches_eq;
use datafusion::execution::context::SessionContext;
use futures::TryStreamExt;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::metadata_columns::{
    RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER, RESERVED_COL_NAME_ROW_ID,
};
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, NestedField, PrimitiveType, Schema,
    TableProperties, Transform, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;
use parquet::basic::Encoding;
use parquet::file::reader::{FileReader, SerializedFileReader};
use tempfile::TempDir;

fn leak_temp_path() -> String {
    let temp_dir = TempDir::new().expect("temp dir");
    let path = temp_dir.path().to_str().expect("utf8").to_string();
    std::mem::forget(temp_dir);
    path
}

async fn catalog() -> MemoryCatalog {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(iceberg::io::LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), leak_temp_path())]),
        )
        .await
        .expect("load catalog")
}

async fn v3_cow_ctx(ns: &str, tbl: &str) -> (SessionContext, Arc<MemoryCatalog>) {
    v3_cow_ctx_inner(ns, tbl, false, HashMap::new()).await
}

async fn v3_cow_ctx_with_format(
    ns: &str,
    tbl: &str,
    format: &str,
) -> (SessionContext, Arc<MemoryCatalog>) {
    let prop = TableProperties::PROPERTY_DEFAULT_FILE_FORMAT.to_string();
    v3_cow_ctx_inner(ns, tbl, false, HashMap::from([(prop, format.to_string())])).await
}

async fn v3_cow_ctx_with_dict(
    ns: &str,
    tbl: &str,
    dict: &str,
) -> (SessionContext, Arc<MemoryCatalog>) {
    v3_cow_ctx_inner(
        ns,
        tbl,
        false,
        HashMap::from([("parquet.enable.dictionary".to_string(), dict.to_string())]),
    )
    .await
}

async fn v3_cow_partitioned_ctx(ns: &str, tbl: &str) -> (SessionContext, Arc<MemoryCatalog>) {
    v3_cow_ctx_inner(ns, tbl, true, HashMap::new()).await
}

async fn v3_cow_ctx_inner(
    ns: &str,
    tbl: &str,
    partitioned: bool,
    properties: HashMap<String, String>,
) -> (SessionContext, Arc<MemoryCatalog>) {
    let iceberg_catalog = catalog().await;
    let namespace = NamespaceIdent::new(ns.to_string());
    iceberg_catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");

    let mut fields =
        vec![NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into()];
    if partitioned {
        fields.push(
            NestedField::required(2, "category", Type::Primitive(PrimitiveType::String)).into(),
        );
        fields.push(NestedField::required(3, "val", Type::Primitive(PrimitiveType::String)).into());
    } else {
        fields.push(NestedField::required(2, "val", Type::Primitive(PrimitiveType::String)).into());
    }
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(fields)
        .build()
        .expect("schema");

    let location = leak_temp_path();
    let creation = if partitioned {
        let partition_spec = UnboundPartitionSpec::builder()
            .with_spec_id(0)
            .add_partition_field(2, "category", Transform::Identity)
            .expect("identity(category)")
            .build();
        TableCreation::builder()
            .name(tbl.to_string())
            .location(location)
            .schema(schema)
            .partition_spec(partition_spec)
            .properties(properties)
            .format_version(FormatVersion::V3)
            .build()
    } else {
        TableCreation::builder()
            .name(tbl.to_string())
            .location(location)
            .schema(schema)
            .properties(properties)
            .format_version(FormatVersion::V3)
            .build()
    };
    iceberg_catalog
        .create_table(&namespace, creation)
        .await
        .expect("create v3 table");

    let client = Arc::new(iceberg_catalog);
    let provider = Arc::new(
        IcebergCatalogProvider::try_new(client.clone())
            .await
            .expect("provider"),
    );
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", provider);
    (ctx, client)
}

async fn lineage_rows(table: &Table) -> Vec<(i32, i64, i64)> {
    let batches: Vec<_> = table
        .scan()
        .select([
            "id",
            RESERVED_COL_NAME_ROW_ID,
            RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER,
        ])
        .build()
        .expect("scan")
        .to_arrow()
        .await
        .expect("to_arrow")
        .try_collect()
        .await
        .expect("collect");
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
            .expect("seq")
            .as_primitive::<datafusion::arrow::datatypes::Int64Type>();
        for index in 0..batch.num_rows() {
            assert!(row_ids.is_valid(index));
            assert!(seqs.is_valid(index));
            rows.push((ids.value(index), row_ids.value(index), seqs.value(index)));
        }
    }
    rows.sort_unstable();
    rows
}

#[tokio::test]
async fn cow_delete_keeps_survivor_row_ids() {
    let ns = "lineage_cow_delete";
    let tbl = "t";
    let (ctx, client) = v3_cow_ctx(ns, tbl).await;
    ctx.sql(&format!(
        "INSERT INTO catalog.{ns}.{tbl} VALUES (1, 'a'), (2, 'b')"
    ))
    .await
    .expect("insert 1")
    .collect()
    .await
    .expect("insert 1 collect");
    ctx.sql(&format!("INSERT INTO catalog.{ns}.{tbl} VALUES (3, 'c')"))
        .await
        .expect("insert 2")
        .collect()
        .await
        .expect("insert 2 collect");

    let ident = TableIdent::new(NamespaceIdent::new(ns.to_string()), tbl.to_string());
    let table = client.load_table(&ident).await.expect("load");
    let before = lineage_rows(&table).await;
    assert_eq!(before, vec![(1, 0, 1), (2, 1, 1), (3, 2, 2)]);

    ctx.sql(&format!("DELETE FROM catalog.{ns}.{tbl} WHERE id = 2"))
        .await
        .expect("delete")
        .collect()
        .await
        .expect("delete collect");

    let table = client.load_table(&ident).await.expect("reload");
    let after = lineage_rows(&table).await;
    assert_eq!(
        after,
        vec![(1, 0, 1), (3, 2, 2)],
        "COW DELETE must keep survivor _row_id and last_updated_seq"
    );
}

#[tokio::test]
async fn cow_update_keeps_row_id_and_bumps_matched_seq() {
    let ns = "lineage_cow_update";
    let tbl = "t";
    let (ctx, client) = v3_cow_ctx(ns, tbl).await;
    ctx.sql(&format!(
        "INSERT INTO catalog.{ns}.{tbl} VALUES (1, 'a'), (2, 'b')"
    ))
    .await
    .expect("insert 1")
    .collect()
    .await
    .expect("insert 1 collect");
    ctx.sql(&format!("INSERT INTO catalog.{ns}.{tbl} VALUES (3, 'c')"))
        .await
        .expect("insert 2")
        .collect()
        .await
        .expect("insert 2 collect");

    let ident = TableIdent::new(NamespaceIdent::new(ns.to_string()), tbl.to_string());
    let table = client.load_table(&ident).await.expect("load");
    let before = lineage_rows(&table).await;
    assert_eq!(before, vec![(1, 0, 1), (2, 1, 1), (3, 2, 2)]);

    ctx.sql(&format!(
        "UPDATE catalog.{ns}.{tbl} SET val = 'B' WHERE id = 2"
    ))
    .await
    .expect("update")
    .collect()
    .await
    .expect("update collect");

    let df = ctx
        .sql(&format!(
            "SELECT id, val FROM catalog.{ns}.{tbl} ORDER BY id"
        ))
        .await
        .expect("select")
        .collect()
        .await
        .expect("select collect");
    assert_batches_eq!(
        &[
            "+----+-----+",
            "| id | val |",
            "+----+-----+",
            "| 1  | a   |",
            "| 2  | B   |",
            "| 3  | c   |",
            "+----+-----+",
        ],
        &df
    );

    let table = client.load_table(&ident).await.expect("reload");
    let after = lineage_rows(&table).await;
    let by_id: HashMap<i32, (i64, i64)> = after
        .into_iter()
        .map(|(id, row_id, seq)| (id, (row_id, seq)))
        .collect();
    assert_eq!(
        by_id[&1],
        (0, 1),
        "unmatched survivor keeps _row_id and seq"
    );
    assert_eq!(
        by_id[&3],
        (2, 2),
        "unmatched later row keeps _row_id and seq"
    );
    assert_eq!(by_id[&2].0, 1, "updated row keeps _row_id");
    assert!(
        by_id[&2].1 > 1,
        "updated row last_updated_seq must advance, got {}",
        by_id[&2].1
    );
}

async fn load_v3_partitioned(client: &MemoryCatalog, ns: &str, tbl: &str) -> Table {
    let ident = TableIdent::new(NamespaceIdent::new(ns.to_string()), tbl.to_string());
    let table = client.load_table(&ident).await.expect("load");
    assert_eq!(table.metadata().format_version(), FormatVersion::V3);
    assert!(
        !table.metadata().default_partition_spec().is_unpartitioned(),
        "partitioned v3 pin must not silently create an unpartitioned table"
    );
    table
}

#[tokio::test]
async fn cow_delete_keeps_survivor_row_ids_across_partitions() {
    let ns = "lineage_cow_delete_part";
    let tbl = "t";
    let (ctx, client) = v3_cow_partitioned_ctx(ns, tbl).await;
    ctx.sql(&format!(
        "INSERT INTO catalog.{ns}.{tbl} VALUES (1, 'a', 'x'), (2, 'a', 'y')"
    ))
    .await
    .expect("insert a")
    .collect()
    .await
    .expect("insert a collect");
    ctx.sql(&format!(
        "INSERT INTO catalog.{ns}.{tbl} VALUES (3, 'b', 'z')"
    ))
    .await
    .expect("insert b")
    .collect()
    .await
    .expect("insert b collect");

    let table = load_v3_partitioned(client.as_ref(), ns, tbl).await;
    let before = lineage_rows(&table).await;
    assert_eq!(before, vec![(1, 0, 1), (2, 1, 1), (3, 2, 2)]);

    ctx.sql(&format!("DELETE FROM catalog.{ns}.{tbl} WHERE id = 2"))
        .await
        .expect("delete")
        .collect()
        .await
        .expect("delete collect");

    let table = load_v3_partitioned(client.as_ref(), ns, tbl).await;
    let after = lineage_rows(&table).await;
    assert_eq!(
        after,
        vec![(1, 0, 1), (3, 2, 2)],
        "partitioned COW DELETE must keep survivor _row_id/seq in the rewritten partition and the untouched one"
    );
}

#[tokio::test]
async fn cow_update_keeps_row_id_and_bumps_matched_seq_across_partitions() {
    let ns = "lineage_cow_update_part";
    let tbl = "t";
    let (ctx, client) = v3_cow_partitioned_ctx(ns, tbl).await;
    ctx.sql(&format!(
        "INSERT INTO catalog.{ns}.{tbl} VALUES (1, 'a', 'x'), (2, 'a', 'y')"
    ))
    .await
    .expect("insert a")
    .collect()
    .await
    .expect("insert a collect");
    ctx.sql(&format!(
        "INSERT INTO catalog.{ns}.{tbl} VALUES (3, 'b', 'z')"
    ))
    .await
    .expect("insert b")
    .collect()
    .await
    .expect("insert b collect");

    let table = load_v3_partitioned(client.as_ref(), ns, tbl).await;
    let before = lineage_rows(&table).await;
    assert_eq!(before, vec![(1, 0, 1), (2, 1, 1), (3, 2, 2)]);

    ctx.sql(&format!(
        "UPDATE catalog.{ns}.{tbl} SET val = 'Y' WHERE id = 2"
    ))
    .await
    .expect("update")
    .collect()
    .await
    .expect("update collect");

    let df = ctx
        .sql(&format!(
            "SELECT id, category, val FROM catalog.{ns}.{tbl} ORDER BY id"
        ))
        .await
        .expect("select")
        .collect()
        .await
        .expect("select collect");
    assert_batches_eq!(
        &[
            "+----+----------+-----+",
            "| id | category | val |",
            "+----+----------+-----+",
            "| 1  | a        | x   |",
            "| 2  | a        | Y   |",
            "| 3  | b        | z   |",
            "+----+----------+-----+",
        ],
        &df
    );

    let table = load_v3_partitioned(client.as_ref(), ns, tbl).await;
    let after = lineage_rows(&table).await;
    let by_id: HashMap<i32, (i64, i64)> = after
        .into_iter()
        .map(|(id, row_id, seq)| (id, (row_id, seq)))
        .collect();
    assert_eq!(
        by_id[&1],
        (0, 1),
        "same-partition unmatched survivor keeps _row_id and seq"
    );
    assert_eq!(
        by_id[&3],
        (2, 2),
        "other-partition unmatched row keeps _row_id and seq"
    );
    assert_eq!(by_id[&2].0, 1, "updated row keeps _row_id");
    assert!(
        by_id[&2].1 > 1,
        "updated row last_updated_seq must advance, got {}",
        by_id[&2].1
    );
}

fn next_row_id(table: &Table) -> u64 {
    table.metadata().next_row_id()
}

async fn state_rows(table: &Table) -> Vec<(i32, String, i64, i64)> {
    let batches: Vec<_> = table
        .scan()
        .select([
            "id",
            "val",
            RESERVED_COL_NAME_ROW_ID,
            RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER,
        ])
        .build()
        .expect("scan")
        .to_arrow()
        .await
        .expect("to_arrow")
        .try_collect()
        .await
        .expect("collect");
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .expect("id")
            .as_primitive::<datafusion::arrow::datatypes::Int32Type>();
        let vals = batch.column_by_name("val").expect("val").as_string::<i32>();
        let row_ids = batch
            .column_by_name(RESERVED_COL_NAME_ROW_ID)
            .expect("_row_id")
            .as_primitive::<datafusion::arrow::datatypes::Int64Type>();
        let seqs = batch
            .column_by_name(RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER)
            .expect("seq")
            .as_primitive::<datafusion::arrow::datatypes::Int64Type>();
        for index in 0..batch.num_rows() {
            assert!(row_ids.is_valid(index));
            assert!(seqs.is_valid(index));
            rows.push((
                ids.value(index),
                vals.value(index).to_string(),
                row_ids.value(index),
                seqs.value(index),
            ));
        }
    }
    rows.sort_unstable();
    rows
}

async fn assert_state(table: &Table, rows: &[(i32, &str, i64, i64)], next: u64) {
    let expected: Vec<(i32, String, i64, i64)> = rows
        .iter()
        .map(|(id, val, row_id, seq)| (*id, (*val).to_string(), *row_id, *seq))
        .collect();
    assert_eq!(state_rows(table).await, expected);
    assert_eq!(next_row_id(table), next);
}

async fn run_sql(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql)
        .await
        .unwrap_or_else(|error| panic!("plan `{sql}`: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("execute `{sql}`: {error}"));
}

async fn spark_insert3(ns: &str) -> (SessionContext, Arc<MemoryCatalog>, TableIdent) {
    let tbl = "t";
    let (ctx, client) = v3_cow_ctx(ns, tbl).await;
    run_sql(
        &ctx,
        &format!("INSERT INTO catalog.{ns}.{tbl} VALUES (1, 'a'), (2, 'b'), (3, 'c')"),
    )
    .await;
    let ident = TableIdent::new(NamespaceIdent::new(ns.to_string()), tbl.to_string());
    let table = client.load_table(&ident).await.expect("load");
    assert_state(&table, &[(1, "a", 0, 1), (2, "b", 1, 1), (3, "c", 2, 1)], 3).await;
    (ctx, client, ident)
}

#[tokio::test]
async fn spark_delete_id_2() {
    let ns = "spark_del_2";
    let (ctx, client, ident) = spark_insert3(ns).await;
    run_sql(&ctx, &format!("DELETE FROM catalog.{ns}.t WHERE id = 2")).await;
    let table = client.load_table(&ident).await.expect("reload");
    assert_state(&table, &[(1, "a", 0, 1), (3, "c", 2, 1)], 5).await;
}

#[tokio::test]
async fn spark_delete_id_3() {
    let ns = "spark_del_3";
    let (ctx, client, ident) = spark_insert3(ns).await;
    run_sql(&ctx, &format!("DELETE FROM catalog.{ns}.t WHERE id = 3")).await;
    let table = client.load_table(&ident).await.expect("reload");
    assert_state(&table, &[(1, "a", 0, 1), (2, "b", 1, 1)], 5).await;
}

#[tokio::test]
async fn spark_delete_id_1() {
    let ns = "spark_del_1";
    let (ctx, client, ident) = spark_insert3(ns).await;
    run_sql(&ctx, &format!("DELETE FROM catalog.{ns}.t WHERE id = 1")).await;
    let table = client.load_table(&ident).await.expect("reload");
    assert_state(&table, &[(2, "b", 1, 1), (3, "c", 2, 1)], 5).await;
}

#[tokio::test]
async fn spark_delete_id_2_then_id_1() {
    let ns = "spark_del_2_1";
    let (ctx, client, ident) = spark_insert3(ns).await;
    run_sql(&ctx, &format!("DELETE FROM catalog.{ns}.t WHERE id = 2")).await;
    let table = client.load_table(&ident).await.expect("reload");
    assert_state(&table, &[(1, "a", 0, 1), (3, "c", 2, 1)], 5).await;
    run_sql(&ctx, &format!("DELETE FROM catalog.{ns}.t WHERE id = 1")).await;
    let table = client.load_table(&ident).await.expect("reload");
    assert_state(&table, &[(3, "c", 2, 1)], 6).await;
}

#[tokio::test]
async fn spark_update_id_2_then_delete_id_1() {
    let ns = "spark_upd_del_1";
    let (ctx, client, ident) = spark_insert3(ns).await;
    run_sql(
        &ctx,
        &format!("UPDATE catalog.{ns}.t SET val = 'B' WHERE id = 2"),
    )
    .await;
    let table = client.load_table(&ident).await.expect("reload");
    assert_state(&table, &[(1, "a", 0, 1), (2, "B", 1, 2), (3, "c", 2, 1)], 6).await;
    run_sql(&ctx, &format!("DELETE FROM catalog.{ns}.t WHERE id = 1")).await;
    let table = client.load_table(&ident).await.expect("reload");
    assert_state(&table, &[(2, "B", 1, 2), (3, "c", 2, 1)], 8).await;
}

#[tokio::test]
async fn spark_update_id_2_then_delete_id_2() {
    let ns = "spark_upd_del_2";
    let (ctx, client, ident) = spark_insert3(ns).await;
    run_sql(
        &ctx,
        &format!("UPDATE catalog.{ns}.t SET val = 'B' WHERE id = 2"),
    )
    .await;
    let table = client.load_table(&ident).await.expect("reload");
    assert_state(&table, &[(1, "a", 0, 1), (2, "B", 1, 2), (3, "c", 2, 1)], 6).await;
    run_sql(&ctx, &format!("DELETE FROM catalog.{ns}.t WHERE id = 2")).await;
    let table = client.load_table(&ident).await.expect("reload");
    assert_state(&table, &[(1, "a", 0, 1), (3, "c", 2, 1)], 8).await;
}

#[tokio::test]
async fn spark_insert_overwrite_then_delete_id_2() {
    let ns = "spark_ow_del_2";
    let (ctx, client, ident) = spark_insert3(ns).await;
    run_sql(
        &ctx,
        &format!("INSERT OVERWRITE catalog.{ns}.t VALUES (1, 'a'), (2, 'b'), (3, 'c')"),
    )
    .await;
    let table = client.load_table(&ident).await.expect("reload");
    assert_state(&table, &[(1, "a", 3, 2), (2, "b", 4, 2), (3, "c", 5, 2)], 6).await;
    run_sql(&ctx, &format!("DELETE FROM catalog.{ns}.t WHERE id = 2")).await;
    let table = client.load_table(&ident).await.expect("reload");
    assert_state(&table, &[(1, "a", 3, 2), (3, "c", 5, 2)], 8).await;
}

#[tokio::test]
async fn spark_update_id_le_2_then_delete_id_3() {
    let ns = "spark_upd_le2_del_3";
    let (ctx, client, ident) = spark_insert3(ns).await;
    run_sql(
        &ctx,
        &format!("UPDATE catalog.{ns}.t SET val = 'B' WHERE id <= 2"),
    )
    .await;
    let table = client.load_table(&ident).await.expect("reload");
    assert_state(&table, &[(1, "B", 0, 2), (2, "B", 1, 2), (3, "c", 2, 1)], 6).await;
    run_sql(&ctx, &format!("DELETE FROM catalog.{ns}.t WHERE id = 3")).await;
    let table = client.load_table(&ident).await.expect("reload");
    assert_state(&table, &[(1, "B", 0, 2), (2, "B", 1, 2)], 8).await;
}

#[tokio::test]
async fn spark_delete_id_2_insert_4_then_delete_id_1() {
    let ns = "spark_del_ins_del";
    let (ctx, client, ident) = spark_insert3(ns).await;
    run_sql(&ctx, &format!("DELETE FROM catalog.{ns}.t WHERE id = 2")).await;
    let table = client.load_table(&ident).await.expect("reload");
    assert_state(&table, &[(1, "a", 0, 1), (3, "c", 2, 1)], 5).await;
    run_sql(&ctx, &format!("INSERT INTO catalog.{ns}.t VALUES (4, 'd')")).await;
    let table = client.load_table(&ident).await.expect("reload");
    assert_state(&table, &[(1, "a", 0, 1), (3, "c", 2, 1), (4, "d", 5, 3)], 6).await;
    run_sql(&ctx, &format!("DELETE FROM catalog.{ns}.t WHERE id = 1")).await;
    let table = client.load_table(&ident).await.expect("reload");
    assert_state(&table, &[(3, "c", 2, 1), (4, "d", 5, 3)], 7).await;
}

#[tokio::test]
async fn spark_three_single_row_inserts_then_delete_id_2_then_id_1() {
    let ns = "spark_3ins_del";
    let tbl = "t";
    let (ctx, client) = v3_cow_ctx(ns, tbl).await;
    run_sql(
        &ctx,
        &format!("INSERT INTO catalog.{ns}.{tbl} VALUES (1, 'a')"),
    )
    .await;
    run_sql(
        &ctx,
        &format!("INSERT INTO catalog.{ns}.{tbl} VALUES (2, 'b')"),
    )
    .await;
    run_sql(
        &ctx,
        &format!("INSERT INTO catalog.{ns}.{tbl} VALUES (3, 'c')"),
    )
    .await;
    let ident = TableIdent::new(NamespaceIdent::new(ns.to_string()), tbl.to_string());
    let table = client.load_table(&ident).await.expect("load");
    assert_state(&table, &[(1, "a", 0, 1), (2, "b", 1, 2), (3, "c", 2, 3)], 3).await;
    run_sql(
        &ctx,
        &format!("DELETE FROM catalog.{ns}.{tbl} WHERE id = 2"),
    )
    .await;
    let table = client.load_table(&ident).await.expect("reload");
    assert_state(&table, &[(1, "a", 0, 1), (3, "c", 2, 3)], 3).await;
    run_sql(
        &ctx,
        &format!("DELETE FROM catalog.{ns}.{tbl} WHERE id = 1"),
    )
    .await;
    let table = client.load_table(&ident).await.expect("reload");
    assert_state(&table, &[(3, "c", 2, 3)], 3).await;
}

async fn live_data_files(table: &Table) -> Vec<DataFile> {
    let snapshot = table.metadata().current_snapshot().expect("snapshot");
    let list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    let mut files = Vec::new();
    for manifest_file in list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.content_type() == DataContentType::Data {
                files.push(entry.data_file().clone());
            }
        }
    }
    files
}

async fn assert_rewrite_format(ns: &str, format: DataFileFormat, check_bytes: impl Fn(&[u8])) {
    let tbl = "t";
    let name = format.to_string();
    let (ctx, client) = v3_cow_ctx_with_format(ns, tbl, &name).await;
    run_sql(
        &ctx,
        &format!("INSERT INTO catalog.{ns}.{tbl} VALUES (1, 'a'), (2, 'b'), (3, 'c')"),
    )
    .await;
    run_sql(
        &ctx,
        &format!("DELETE FROM catalog.{ns}.{tbl} WHERE id = 2"),
    )
    .await;
    let ident = TableIdent::new(NamespaceIdent::new(ns.to_string()), tbl.to_string());
    let table = client.load_table(&ident).await.expect("reload");
    let files = live_data_files(&table).await;
    assert!(!files.is_empty(), "delete leaves live data files");
    for file in &files {
        assert_eq!(file.content_type(), DataContentType::Data);
        assert_eq!(file.file_format(), format, "rewritten file keeps {name}");
        let path = file.file_path();
        let expected = format!(".{format}");
        assert!(
            path.ends_with(&expected),
            "rewritten file path {path} ends with {expected}"
        );
        if format != DataFileFormat::Parquet {
            assert!(
                !path.ends_with(".parquet"),
                "non-parquet rewritten path {path} keeps its own suffix"
            );
        }
        let bytes = table
            .file_io()
            .new_input(file.file_path())
            .expect("input")
            .read()
            .await
            .expect("read");
        check_bytes(&bytes);
    }
    let df = ctx
        .sql(&format!(
            "SELECT id, val FROM catalog.{ns}.{tbl} ORDER BY id"
        ))
        .await
        .expect("select")
        .collect()
        .await
        .expect("collect");
    assert_batches_eq!(
        &[
            "+----+-----+",
            "| id | val |",
            "+----+-----+",
            "| 1  | a   |",
            "| 3  | c   |",
            "+----+-----+",
        ],
        &df
    );
}

#[tokio::test]
async fn cow_delete_rewrites_orc_as_orc() {
    assert_rewrite_format("lineage_cow_rewrite_orc", DataFileFormat::Orc, |bytes| {
        assert!(
            bytes.len() > 4 && &bytes[bytes.len() - 4..bytes.len() - 1] == b"ORC",
            "orc tail magic"
        );
    })
    .await;
}

#[tokio::test]
async fn cow_delete_rewrites_avro_as_avro() {
    assert_rewrite_format("lineage_cow_rewrite_avro", DataFileFormat::Avro, |bytes| {
        assert!(bytes.starts_with(b"Obj\x01"), "avro OCF header");
    })
    .await;
}

fn parquet_columns_use_dictionary(bytes: bytes::Bytes) -> Vec<bool> {
    let reader = SerializedFileReader::new(bytes).expect("read the parquet footer");
    let metadata = reader.metadata();
    assert!(
        metadata.num_row_groups() > 0,
        "the rewritten file must hold a row group"
    );
    let mut flags = Vec::new();
    for row_group in metadata.row_groups() {
        for column in row_group.columns() {
            flags.push(column.encodings().any(|encoding| {
                matches!(
                    encoding,
                    Encoding::PLAIN_DICTIONARY | Encoding::RLE_DICTIONARY
                )
            }));
        }
    }
    assert!(
        !flags.is_empty(),
        "the rewritten file must hold column chunks"
    );
    flags
}

async fn cow_rewritten_parquet_dictionary(ns: &str, dict: Option<&str>) -> Vec<bool> {
    let tbl = "t";
    let (ctx, client) = match dict {
        None => v3_cow_ctx(ns, tbl).await,
        Some(value) => v3_cow_ctx_with_dict(ns, tbl, value).await,
    };
    let values = (1..=300)
        .map(|id| format!("({id}, 'v{}')", id % 4))
        .collect::<Vec<_>>()
        .join(", ");
    run_sql(
        &ctx,
        &format!("INSERT INTO catalog.{ns}.{tbl} VALUES {values}"),
    )
    .await;
    run_sql(
        &ctx,
        &format!("DELETE FROM catalog.{ns}.{tbl} WHERE id = 1"),
    )
    .await;
    let ident = TableIdent::new(NamespaceIdent::new(ns.to_string()), tbl.to_string());
    let table = client.load_table(&ident).await.expect("reload");
    let files = live_data_files(&table).await;
    assert_eq!(files.len(), 1, "one delete rewrites one data file");
    assert_eq!(files[0].file_format(), DataFileFormat::Parquet);
    let bytes = table
        .file_io()
        .new_input(files[0].file_path())
        .expect("input")
        .read()
        .await
        .expect("read");
    parquet_columns_use_dictionary(bytes)
}

#[tokio::test]
async fn cow_delete_rewrite_honors_metrics_default_none() {
    let ns = "lineage_cow_metrics_none";
    let tbl = "t";
    let (ctx, client) = v3_cow_ctx_inner(
        ns,
        tbl,
        false,
        HashMap::from([(
            "write.metadata.metrics.default".to_string(),
            "none".to_string(),
        )]),
    )
    .await;
    run_sql(
        &ctx,
        &format!("INSERT INTO catalog.{ns}.{tbl} VALUES (1, 'a'), (2, 'b'), (3, 'c')"),
    )
    .await;
    run_sql(
        &ctx,
        &format!("DELETE FROM catalog.{ns}.{tbl} WHERE id = 2"),
    )
    .await;
    let ident = TableIdent::new(NamespaceIdent::new(ns.to_string()), tbl.to_string());
    let table = client.load_table(&ident).await.expect("reload");
    let files = live_data_files(&table).await;
    assert_eq!(files.len(), 1, "one delete rewrites one data file");
    assert!(
        files[0].column_sizes().is_empty(),
        "metrics.default=none must write no column_sizes"
    );
    assert!(
        files[0].value_counts().is_empty(),
        "metrics.default=none must write no value_counts"
    );
    assert!(
        files[0].null_value_counts().is_empty(),
        "metrics.default=none must write no null_value_counts"
    );
    assert!(
        files[0].nan_value_counts().is_empty(),
        "metrics.default=none must write no nan_value_counts"
    );
    assert!(
        files[0].lower_bounds().is_empty(),
        "metrics.default=none must write no lower_bounds"
    );
    assert!(
        files[0].upper_bounds().is_empty(),
        "metrics.default=none must write no upper_bounds"
    );
}

#[tokio::test]
async fn cow_delete_rewrite_parquet_defaults_dictionary_off() {
    let flags = cow_rewritten_parquet_dictionary("lineage_cow_dict_off", None).await;
    assert!(
        flags.iter().all(|flag| !flag),
        "a default rewrite leaves no dictionary encoding"
    );
}

#[tokio::test]
async fn cow_delete_rewrite_parquet_property_enables_dictionary() {
    let flags = cow_rewritten_parquet_dictionary("lineage_cow_dict_on", Some("true")).await;
    assert!(
        flags.iter().all(|flag| *flag),
        "a property rewrite keeps dictionary encoding"
    );
}
