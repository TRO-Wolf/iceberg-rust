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
use datafusion::physical_plan::collect;
use futures::TryStreamExt;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::metadata_columns::{
    RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER, RESERVED_COL_NAME_ROW_ID,
};
use iceberg::spec::{
    DataFileFormat, FormatVersion, NestedField, PrimitiveType, Schema, Transform, Type,
    UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::ApplyTransactionAction;
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

async fn ctx_with(
    ns: &str,
    tbl: &str,
    version: FormatVersion,
    merge_on_read: bool,
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

    let mut properties = HashMap::new();
    if merge_on_read {
        properties.insert("write.delete.mode".to_string(), "merge-on-read".to_string());
        properties.insert("write.update.mode".to_string(), "merge-on-read".to_string());
    }

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
            .format_version(version)
            .properties(properties)
            .partition_spec(partition_spec)
            .build()
    } else {
        TableCreation::builder()
            .name(tbl.to_string())
            .location(location)
            .schema(schema)
            .format_version(version)
            .properties(properties)
            .build()
    };
    iceberg_catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table");

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

async fn run_sql(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql)
        .await
        .unwrap_or_else(|error| panic!("plan `{sql}`: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("execute `{sql}`: {error}"));
}

async fn load(client: &MemoryCatalog, ns: &str, tbl: &str) -> Table {
    client
        .load_table(&TableIdent::new(
            NamespaceIdent::new(ns.to_string()),
            tbl.to_string(),
        ))
        .await
        .expect("load")
}

async fn lineage_rows(table: &Table) -> Vec<(i32, String, i64, i64)> {
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
            assert!(
                row_ids.is_valid(index),
                "live row must have _row_id at {index}"
            );
            assert!(
                seqs.is_valid(index),
                "live row must have _last_updated_sequence_number at {index}"
            );
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

fn next_row_id(table: &Table) -> u64 {
    table.metadata().next_row_id()
}

async fn live_delete_formats(table: &Table) -> Vec<DataFileFormat> {
    let snapshot = table.metadata().current_snapshot().expect("snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    let mut formats = Vec::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != iceberg::spec::ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest");
        for entry in manifest.entries() {
            if entry.is_alive() {
                formats.push(entry.data_file().file_format());
            }
        }
    }
    formats
}

async fn seed_three_rows(ctx: &SessionContext, ns: &str, tbl: &str) {
    run_sql(
        ctx,
        &format!("INSERT INTO catalog.{ns}.{tbl} VALUES (1, 'a'), (2, 'b'), (3, 'c')"),
    )
    .await;
}

#[tokio::test]
async fn mor_update_keeps_row_id_and_advances_matched_seq() {
    let ns = "lineage_mor_update";
    let tbl = "t";
    let (ctx, client) = ctx_with(ns, tbl, FormatVersion::V3, true, false).await;
    seed_three_rows(&ctx, ns, tbl).await;

    let table = load(client.as_ref(), ns, tbl).await;
    let before = lineage_rows(&table).await;
    assert_eq!(before, vec![
        (1, "a".to_string(), 0, 1),
        (2, "b".to_string(), 1, 1),
        (3, "c".to_string(), 2, 1)
    ]);
    let next_before = next_row_id(&table);
    assert_eq!(next_before, 3);

    run_sql(
        &ctx,
        &format!("UPDATE catalog.{ns}.{tbl} SET val = 'B' WHERE id = 2"),
    )
    .await;

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
            "| 2  | B   |",
            "| 3  | c   |",
            "+----+-----+",
        ],
        &df
    );

    let table = load(client.as_ref(), ns, tbl).await;
    let after = lineage_rows(&table).await;
    let by_id: HashMap<i32, (String, i64, i64)> = after
        .into_iter()
        .map(|(id, val, row_id, seq)| (id, (val, row_id, seq)))
        .collect();
    assert_eq!(by_id[&1], ("a".to_string(), 0, 1), "unmatched keeps both");
    assert_eq!(by_id[&3], ("c".to_string(), 2, 1), "unmatched keeps both");
    assert_eq!(by_id[&2].1, 1, "updated row keeps _row_id");
    assert!(
        by_id[&2].2 > 1,
        "updated row last_updated_seq must advance, got {}",
        by_id[&2].2
    );
    assert_eq!(by_id[&2].0, "B");
    assert_eq!(
        next_row_id(&table),
        next_before + 1,
        "MoR replacement is an unassigned DATA manifest; Java += added"
    );
}

#[tokio::test]
async fn sequential_mor_update_keeps_one_row_id_and_advances_seq_twice() {
    let ns = "lineage_mor_update_seq";
    let tbl = "t";
    let (ctx, client) = ctx_with(ns, tbl, FormatVersion::V3, true, false).await;
    seed_three_rows(&ctx, ns, tbl).await;

    let table = load(client.as_ref(), ns, tbl).await;
    let next_before = next_row_id(&table);

    run_sql(
        &ctx,
        &format!("UPDATE catalog.{ns}.{tbl} SET val = 'B' WHERE id = 2"),
    )
    .await;
    let table = load(client.as_ref(), ns, tbl).await;
    let mid = lineage_rows(&table).await;
    let mid_by_id: HashMap<i32, (String, i64, i64)> = mid
        .into_iter()
        .map(|(id, val, row_id, seq)| (id, (val, row_id, seq)))
        .collect();
    let first_seq = mid_by_id[&2].2;
    assert_eq!(mid_by_id[&2].1, 1);
    assert!(first_seq > 1);
    assert_eq!(next_row_id(&table), next_before + 1);

    run_sql(
        &ctx,
        &format!("UPDATE catalog.{ns}.{tbl} SET val = 'BB' WHERE id = 2"),
    )
    .await;
    let table = load(client.as_ref(), ns, tbl).await;
    let after = lineage_rows(&table).await;
    let by_id: HashMap<i32, (String, i64, i64)> = after
        .into_iter()
        .map(|(id, val, row_id, seq)| (id, (val, row_id, seq)))
        .collect();
    assert_eq!(by_id[&2].1, 1, "the same row keeps one _row_id");
    assert!(
        by_id[&2].2 > first_seq,
        "second UPDATE must advance last_updated_seq again, first={first_seq} second={}",
        by_id[&2].2
    );
    assert_eq!(by_id[&2].0, "BB");
    assert_eq!(by_id[&1], ("a".to_string(), 0, 1));
    assert_eq!(by_id[&3], ("c".to_string(), 2, 1));
    assert_eq!(next_row_id(&table), next_before + 2);
}

#[tokio::test]
async fn partitioned_mor_update_keeps_lineage_across_partitions() {
    let ns = "lineage_mor_update_part";
    let tbl = "t";
    let (ctx, client) = ctx_with(ns, tbl, FormatVersion::V3, true, true).await;
    run_sql(
        &ctx,
        &format!(
            "INSERT INTO catalog.{ns}.{tbl} VALUES (1, 'a', 'x'), (2, 'a', 'y'), (3, 'b', 'z')"
        ),
    )
    .await;

    let table = load(client.as_ref(), ns, tbl).await;
    let before = lineage_rows(&table).await;
    let before_by_id: HashMap<i32, (String, i64, i64)> = before
        .into_iter()
        .map(|(id, val, row_id, seq)| (id, (val, row_id, seq)))
        .collect();
    assert_eq!(before_by_id.len(), 3);
    let next_before = next_row_id(&table);

    run_sql(
        &ctx,
        &format!("UPDATE catalog.{ns}.{tbl} SET val = 'Y' WHERE id = 2"),
    )
    .await;

    let table = load(client.as_ref(), ns, tbl).await;
    let after = lineage_rows(&table).await;
    let by_id: HashMap<i32, (String, i64, i64)> = after
        .into_iter()
        .map(|(id, val, row_id, seq)| (id, (val, row_id, seq)))
        .collect();
    assert_eq!(
        by_id[&1], before_by_id[&1],
        "partition a unmatched row keeps lineage"
    );
    assert_eq!(
        by_id[&3], before_by_id[&3],
        "partition b unmatched row keeps lineage"
    );
    assert_eq!(by_id[&2].1, before_by_id[&2].1, "updated row keeps _row_id");
    assert!(
        by_id[&2].2 > before_by_id[&2].2,
        "updated row last_updated_seq must advance"
    );
    assert_eq!(by_id[&2].0, "Y");
    assert_eq!(next_row_id(&table), next_before + 1);
}

#[tokio::test]
async fn v2_mor_update_writes_position_deletes_and_has_no_v3_lineage() {
    let ns = "lineage_mor_v2";
    let tbl = "t";
    let (ctx, client) = ctx_with(ns, tbl, FormatVersion::V2, true, false).await;
    seed_three_rows(&ctx, ns, tbl).await;
    run_sql(
        &ctx,
        &format!("UPDATE catalog.{ns}.{tbl} SET val = 'B' WHERE id = 2"),
    )
    .await;

    let table = load(client.as_ref(), ns, tbl).await;
    assert_eq!(table.metadata().format_version(), FormatVersion::V2);
    let formats = live_delete_formats(&table).await;
    assert!(
        formats.contains(&DataFileFormat::Parquet),
        "V2 merge-on-read UPDATE must write parquet position deletes, got {formats:?}"
    );
    assert!(
        !formats.contains(&DataFileFormat::Puffin),
        "V2 must not write a deletion vector, got {formats:?}"
    );
    assert_eq!(next_row_id(&table), 0, "V2 has no row-lineage counter");

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
    for batch in batches {
        let row_ids = batch
            .column_by_name(RESERVED_COL_NAME_ROW_ID)
            .expect("_row_id");
        let seqs = batch
            .column_by_name(RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER)
            .expect("seq");
        assert_eq!(row_ids.null_count(), batch.num_rows());
        assert_eq!(seqs.null_count(), batch.num_rows());
    }
}

#[tokio::test]
async fn mor_update_conflict_on_removed_data_file_publishes_no_replacement_dv() {
    let ns = "lineage_mor_conflict";
    let tbl = "t";
    let (ctx, client) = ctx_with(ns, tbl, FormatVersion::V3, true, false).await;
    seed_three_rows(&ctx, ns, tbl).await;
    let table = load(client.as_ref(), ns, tbl).await;
    let data_files_before = {
        let snapshot = table.metadata().current_snapshot().expect("snapshot");
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .expect("manifest list");
        let mut files = Vec::new();
        for manifest_file in manifest_list.entries() {
            if manifest_file.content != iceberg::spec::ManifestContentType::Data {
                continue;
            }
            let manifest = manifest_file
                .load_manifest(table.file_io())
                .await
                .expect("manifest");
            for entry in manifest.entries() {
                if entry.is_alive() {
                    files.push(entry.data_file().clone());
                }
            }
        }
        files
    };
    assert_eq!(data_files_before.len(), 1);
    let target = data_files_before[0].clone();

    let plan = ctx
        .sql(&format!(
            "UPDATE catalog.{ns}.{tbl} SET val = 'B' WHERE id = 2"
        ))
        .await
        .expect("plan update")
        .create_physical_plan()
        .await
        .expect("frozen physical plan");
    let tx = iceberg::transaction::Transaction::new(&table);
    tx.delete_files()
        .delete_files([target.file_path().to_string()])
        .apply(tx)
        .expect("apply delete_files")
        .commit(client.as_ref())
        .await
        .expect("concurrent remove of the referenced data file");

    let err = collect(plan, ctx.task_ctx())
        .await
        .expect_err("UPDATE must refuse a concurrent removal of its referenced data file");
    let message = err.to_string();
    assert!(
        message.contains("missing data files") || message.contains("conflicting delete"),
        "expected a files-exist or deleted-files refusal, got {message}"
    );

    let table = load(client.as_ref(), ns, tbl).await;
    let dv_formats = live_delete_formats(&table).await;
    assert!(
        dv_formats.is_empty(),
        "a refused UPDATE must not leave a replacement DV live, got {dv_formats:?}"
    );
}
