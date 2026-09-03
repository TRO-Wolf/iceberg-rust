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

use datafusion::execution::context::SessionContext;
use datafusion::prelude::SessionConfig;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, FormatVersion, Literal, ManifestContentType, NestedField, PrimitiveLiteral,
    PrimitiveType, Schema, Transform, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;
use tempfile::TempDir;

const SHUFFLES: [[i32; 5]; 10] = [
    [3, 1, 4, 0, 2],
    [4, 2, 0, 3, 1],
    [2, 4, 1, 3, 0],
    [0, 4, 2, 1, 3],
    [1, 3, 0, 4, 2],
    [4, 0, 1, 2, 3],
    [2, 0, 3, 4, 1],
    [1, 4, 3, 0, 2],
    [3, 2, 4, 1, 0],
    [0, 2, 4, 1, 3],
];

fn leak_temp_path() -> String {
    let temp_dir = TempDir::new().expect("temp dir");
    let path = temp_dir.path().to_str().expect("utf8").to_string();
    std::mem::forget(temp_dir);
    path
}

async fn catalog() -> MemoryCatalog {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), leak_temp_path())]),
        )
        .await
        .expect("load catalog")
}

async fn partitioned_ctx(ns: &str, tbl: &str) -> (SessionContext, Arc<MemoryCatalog>) {
    let iceberg_catalog = catalog().await;
    let namespace = NamespaceIdent::new(ns.to_string());
    iceberg_catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "part", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()
        .expect("schema");
    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "part", Transform::Identity)
        .expect("identity(part)")
        .build();
    iceberg_catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name(tbl.to_string())
                .location(leak_temp_path())
                .schema(schema)
                .partition_spec(partition_spec)
                .format_version(FormatVersion::V3)
                .properties(HashMap::from([(
                    "write.datafusion.fanout.enabled".to_string(),
                    "true".to_string(),
                )]))
                .build(),
        )
        .await
        .expect("create table");
    let client = Arc::new(iceberg_catalog);
    let provider = Arc::new(
        IcebergCatalogProvider::try_new(client.clone())
            .await
            .expect("provider"),
    );
    let config = SessionConfig::new().with_target_partitions(1);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_catalog("catalog", provider);
    (ctx, client)
}

async fn manifest_partition_order(table: &Table) -> Vec<i32> {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    let mut order = Vec::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest");
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            if entry.data_file().content_type() != DataContentType::Data {
                continue;
            }
            match entry.data_file().partition().fields() {
                [Some(Literal::Primitive(PrimitiveLiteral::Int(value)))] => order.push(*value),
                other => panic!("expected identity-int partition, got {other:?}"),
            }
        }
    }
    order
}

#[tokio::test]
async fn partitioned_insert_manifest_order_is_ascending_across_ten_shuffles() {
    for (run, shuffle) in SHUFFLES.iter().enumerate() {
        let ns = format!("fanout_order_{run}");
        let tbl = "t";
        let (ctx, catalog) = partitioned_ctx(&ns, tbl).await;
        let values = shuffle
            .iter()
            .enumerate()
            .map(|(index, part)| format!("({}, {part})", index as i32 + 1))
            .collect::<Vec<_>>()
            .join(", ");
        ctx.sql(&format!("INSERT INTO catalog.{ns}.{tbl} VALUES {values}"))
            .await
            .unwrap_or_else(|error| panic!("plan insert run {run}: {error}"))
            .collect()
            .await
            .unwrap_or_else(|error| panic!("execute insert run {run}: {error}"));
        let table = catalog
            .load_table(&TableIdent::new(
                NamespaceIdent::new(ns.clone()),
                tbl.to_string(),
            ))
            .await
            .expect("load table");
        let order = manifest_partition_order(&table).await;
        assert_eq!(
            order,
            vec![0, 1, 2, 3, 4],
            "run {run} shuffle {shuffle:?} wrote manifest data-file order {order:?}"
        );
    }
}
