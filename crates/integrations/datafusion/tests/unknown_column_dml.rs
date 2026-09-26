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
use std::fs::File;
use std::sync::Arc;

use datafusion::execution::context::SessionContext;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, FormatVersion, ManifestContentType, NestedField, PrimitiveType,
    Schema, Type,
};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use tempfile::TempDir;

async fn run(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql)
        .await
        .unwrap_or_else(|error| panic!("plan {sql}: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("run {sql}: {error}"));
}

async fn v3_unknown_table() -> (SessionContext, TempDir, Arc<dyn Catalog>, TableIdent) {
    let warehouse = TempDir::new().expect("warehouse");
    let path = warehouse.path().to_str().expect("utf-8 path").to_string();
    let catalog: Arc<dyn Catalog> = Arc::new(
        MemoryCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), path)]),
            )
            .await
            .expect("memory catalog"),
    );
    let namespace = NamespaceIdent::new("ns".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "c", Type::Primitive(PrimitiveType::Unknown)).into(),
            NestedField::optional(3, "v", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("unknown schema");
    let creation = TableCreation::builder()
        .name("t".to_string())
        .schema(schema)
        .format_version(FormatVersion::V3)
        .properties(HashMap::from([(
            "write.delete.mode".to_string(),
            "copy-on-write".to_string(),
        )]))
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create v3 table");
    let provider = IcebergCatalogProvider::try_new(Arc::clone(&catalog))
        .await
        .expect("catalog provider");
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));
    (
        ctx,
        warehouse,
        catalog,
        TableIdent::new(namespace, "t".to_string()),
    )
}

async fn live_data_files(catalog: &Arc<dyn Catalog>, ident: &TableIdent) -> Vec<DataFile> {
    let table = catalog.load_table(ident).await.expect("load table");
    let snapshot = table.metadata().current_snapshot().expect("snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    let mut files = Vec::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.data_file().content_type() == DataContentType::Data {
                files.push(entry.data_file().clone());
            }
        }
    }
    files
}

fn parquet_column_names(file_path: &str) -> Vec<String> {
    let local = file_path.strip_prefix("file://").unwrap_or(file_path);
    let metadata = ArrowReaderMetadata::load(
        &File::open(local).expect("open data file"),
        ArrowReaderOptions::default(),
    )
    .expect("parquet footer");
    metadata
        .parquet_schema()
        .columns()
        .iter()
        .map(|column| column.path().string())
        .collect()
}

#[tokio::test]
async fn copy_on_write_delete_rewrites_a_file_without_the_unknown_column() {
    let (ctx, _warehouse, catalog, ident) = v3_unknown_table().await;
    run(
        &ctx,
        "INSERT INTO catalog.ns.t VALUES (1, NULL, 'a'), (2, NULL, 'b'), (3, NULL, 'c')",
    )
    .await;
    let inserted = live_data_files(&catalog, &ident).await;
    assert_eq!(inserted.len(), 1);

    run(&ctx, "DELETE FROM catalog.ns.t WHERE id = 1").await;
    let rewritten = live_data_files(&catalog, &ident).await;
    assert_eq!(rewritten.len(), 1, "the delete rewrites one file");
    let file = &rewritten[0];
    assert_ne!(file.file_path(), inserted[0].file_path());
    assert_eq!(file.record_count(), 2);
    assert_eq!(parquet_column_names(inserted[0].file_path()), vec![
        "id", "v"
    ]);
    assert_eq!(parquet_column_names(file.file_path()), vec![
        "id",
        "v",
        "_row_id",
        "_last_updated_sequence_number"
    ]);
    for (map_name, has_unknown_id) in [
        ("column_sizes", file.column_sizes().contains_key(&2)),
        ("value_counts", file.value_counts().contains_key(&2)),
        (
            "null_value_counts",
            file.null_value_counts().contains_key(&2),
        ),
        ("nan_value_counts", file.nan_value_counts().contains_key(&2)),
        ("lower_bounds", file.lower_bounds().contains_key(&2)),
        ("upper_bounds", file.upper_bounds().contains_key(&2)),
    ] {
        assert!(!has_unknown_id, "{map_name} must omit the unknown field id");
    }
    assert_eq!(file.value_counts().get(&1), Some(&2));
    assert_eq!(file.value_counts().get(&3), Some(&2));
}
