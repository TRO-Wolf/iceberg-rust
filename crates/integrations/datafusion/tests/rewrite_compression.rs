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
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, FormatVersion, ManifestContentType, NestedField, PrimitiveType, Schema, Type,
};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;
use parquet::basic::Compression;
use parquet::file::reader::{FileReader, SerializedFileReader};
use tempfile::TempDir;

async fn create_dml_fixture(
    namespace: &str,
    table_name: &str,
    properties: HashMap<String, String>,
    format_version: FormatVersion,
) -> (SessionContext, Arc<MemoryCatalog>, TableIdent, TempDir) {
    let warehouse = TempDir::new().expect("create warehouse");
    let warehouse_path = warehouse
        .path()
        .to_str()
        .expect("warehouse path is UTF-8")
        .to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
        )
        .await
        .expect("build memory catalog");
    let namespace = NamespaceIdent::new(namespace.to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "label", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build schema");
    let table_ident = TableIdent::new(namespace.clone(), table_name.to_string());
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name(table_name.to_string())
                .location(format!("{warehouse_path}/{table_name}"))
                .schema(schema)
                .properties(properties)
                .format_version(format_version)
                .build(),
        )
        .await
        .expect("create table");
    let catalog = Arc::new(catalog);
    let provider = Arc::new(
        IcebergCatalogProvider::try_new(catalog.clone())
            .await
            .expect("catalog provider"),
    );
    let context = SessionContext::new();
    context.register_catalog("catalog", provider);
    (context, catalog, table_ident, warehouse)
}

async fn run_sql(context: &SessionContext, sql: &str) {
    context
        .sql(sql)
        .await
        .unwrap_or_else(|error| panic!("plan `{sql}`: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("execute `{sql}`: {error}"));
}

async fn insert_rows(context: &SessionContext, table_ident: &TableIdent) {
    let sql = format!(
        "INSERT INTO catalog.{}.{} VALUES (1, 'alpha'), (2, 'beta'), (3, 'gamma')",
        table_ident.namespace()[0],
        table_ident.name()
    );
    run_sql(context, &sql).await;
}

async fn delete_row(context: &SessionContext, table_ident: &TableIdent, id: i32) {
    let sql = format!(
        "DELETE FROM catalog.{}.{} WHERE id = {id}",
        table_ident.namespace()[0],
        table_ident.name()
    );
    run_sql(context, &sql).await;
}

async fn live_file_paths(
    catalog: &MemoryCatalog,
    table_ident: &TableIdent,
    manifest_content: ManifestContentType,
    data_content: DataContentType,
) -> Vec<String> {
    let table = catalog.load_table(table_ident).await.expect("load table");
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("a snapshot is committed");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load manifest list");
    let mut paths = Vec::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != manifest_content {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("load manifest");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.data_file().content_type() == data_content {
                paths.push(entry.data_file().file_path().to_string());
            }
        }
    }
    paths
}

fn local_fs_path(file_path: &str) -> &str {
    file_path.strip_prefix("file://").unwrap_or(file_path)
}

fn assert_every_chunk_zstd(file_path: &str) {
    let local = local_fs_path(file_path);
    let file = File::open(local).unwrap_or_else(|error| panic!("open file {local}: {error}"));
    let reader = SerializedFileReader::new(file)
        .unwrap_or_else(|error| panic!("read parquet footer {local}: {error}"));
    let mut chunks = 0;
    for row_group in reader.metadata().row_groups() {
        for column in row_group.columns() {
            chunks += 1;
            assert!(
                matches!(column.compression(), Compression::ZSTD(_)),
                "unexpected column-chunk compression {:?} in {local}",
                column.compression()
            );
        }
    }
    assert!(chunks > 0, "parquet file {local} has no column chunks");
}

#[tokio::test]
async fn rewrite_data_files_output_carries_table_codec() {
    let (context, catalog, table_ident, _warehouse) = create_dml_fixture(
        "rewrite_compression_compaction",
        "target",
        HashMap::new(),
        FormatVersion::V2,
    )
    .await;
    insert_rows(&context, &table_ident).await;
    insert_rows(&context, &table_ident).await;

    let table = catalog
        .load_table(&table_ident)
        .await
        .expect("load table for rewrite");
    let result = iceberg::maintenance::RewriteDataFiles::new(table)
        .min_input_files(2)
        .execute(catalog.as_ref())
        .await
        .expect("run rewrite_data_files");
    assert!(
        result.rewritten_data_files_count > 0,
        "the two inserted files must be rewritten, got {result:?}"
    );

    let paths = live_file_paths(
        &catalog,
        &table_ident,
        ManifestContentType::Data,
        DataContentType::Data,
    )
    .await;
    assert!(
        !paths.is_empty(),
        "rewrite_data_files must leave live data files"
    );
    for path in &paths {
        assert_every_chunk_zstd(path);
    }
}

#[tokio::test]
async fn cow_delete_rewrite_output_carries_table_codec() {
    let (context, catalog, table_ident, _warehouse) = create_dml_fixture(
        "rewrite_compression_cow_delete",
        "target",
        HashMap::new(),
        FormatVersion::V2,
    )
    .await;
    insert_rows(&context, &table_ident).await;

    let before = live_file_paths(
        &catalog,
        &table_ident,
        ManifestContentType::Data,
        DataContentType::Data,
    )
    .await;
    delete_row(&context, &table_ident, 2).await;
    let after = live_file_paths(
        &catalog,
        &table_ident,
        ManifestContentType::Data,
        DataContentType::Data,
    )
    .await;

    assert!(
        !after.is_empty(),
        "copy-on-write delete must leave survivor data files"
    );
    for path in &after {
        assert!(
            !before.contains(path),
            "copy-on-write delete must rewrite the file; {path} predates the delete"
        );
        assert_every_chunk_zstd(path);
    }
}

#[tokio::test]
async fn mor_delete_position_delete_file_carries_table_codec() {
    let (context, catalog, table_ident, _warehouse) = create_dml_fixture(
        "rewrite_compression_mor_delete",
        "target",
        HashMap::from([("write.delete.mode".to_string(), "merge-on-read".to_string())]),
        FormatVersion::V2,
    )
    .await;
    insert_rows(&context, &table_ident).await;
    delete_row(&context, &table_ident, 2).await;

    let paths = live_file_paths(
        &catalog,
        &table_ident,
        ManifestContentType::Deletes,
        DataContentType::PositionDeletes,
    )
    .await;
    assert!(
        !paths.is_empty(),
        "merge-on-read delete must write a position-delete file"
    );
    for path in &paths {
        assert_every_chunk_zstd(path);
    }
}
