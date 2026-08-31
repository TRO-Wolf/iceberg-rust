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
    FormatVersion, NestedField, PrimitiveType, Schema, Transform, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;
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
    v3_cow_ctx_inner(ns, tbl, false).await
}

async fn v3_cow_partitioned_ctx(ns: &str, tbl: &str) -> (SessionContext, Arc<MemoryCatalog>) {
    v3_cow_ctx_inner(ns, tbl, true).await
}

async fn v3_cow_ctx_inner(
    ns: &str,
    tbl: &str,
    partitioned: bool,
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
            .format_version(FormatVersion::V3)
            .build()
    } else {
        TableCreation::builder()
            .name(tbl.to_string())
            .location(location)
            .schema(schema)
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
