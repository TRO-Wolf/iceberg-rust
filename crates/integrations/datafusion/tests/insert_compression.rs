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
    DataContentType, ManifestContentType, NestedField, PrimitiveType, Schema, TableProperties, Type,
};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;
use parquet::basic::Compression;
use parquet::file::reader::{FileReader, SerializedFileReader};
use tempfile::TempDir;

const CODEC_KEY: &str = TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC;
const LEVEL_KEY: &str = TableProperties::PROPERTY_PARQUET_COMPRESSION_LEVEL;

async fn create_insert_fixture(
    namespace: &str,
    table_name: &str,
    properties: HashMap<String, String>,
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

async fn insert_rows(context: &SessionContext, table_ident: &TableIdent) {
    let sql = format!(
        "INSERT INTO catalog.{}.{} VALUES (1, 'alpha'), (2, 'beta'), (3, 'gamma')",
        table_ident.namespace()[0],
        table_ident.name()
    );
    context
        .sql(&sql)
        .await
        .expect("plan insert")
        .collect()
        .await
        .expect("execute insert");
}

async fn live_data_file_paths(catalog: &MemoryCatalog, table_ident: &TableIdent) -> Vec<String> {
    let table = catalog
        .load_table(table_ident)
        .await
        .expect("load table after insert");
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("insert commits one snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load manifest list");
    let mut paths = Vec::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("load manifest");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.data_file().content_type() == DataContentType::Data {
                paths.push(entry.data_file().file_path().to_string());
            }
        }
    }
    assert!(
        !paths.is_empty(),
        "insert must produce at least one live data file"
    );
    paths
}

fn local_fs_path(file_path: &str) -> &str {
    file_path.strip_prefix("file://").unwrap_or(file_path)
}

fn column_chunk_compressions(file_path: &str) -> Vec<Compression> {
    let local = local_fs_path(file_path);
    let file = File::open(local).unwrap_or_else(|error| {
        panic!("open data file {local}: {error}");
    });
    let reader = SerializedFileReader::new(file).unwrap_or_else(|error| {
        panic!("read parquet footer {local}: {error}");
    });
    let mut compressions = Vec::new();
    for row_group in reader.metadata().row_groups() {
        for column in row_group.columns() {
            compressions.push(column.compression());
        }
    }
    assert!(
        !compressions.is_empty(),
        "parquet file {local} has no column chunks"
    );
    compressions
}

async fn insert_column_chunk_compressions(
    namespace: &str,
    table_name: &str,
    properties: HashMap<String, String>,
) -> Vec<Compression> {
    let (context, catalog, table_ident, _warehouse) =
        create_insert_fixture(namespace, table_name, properties).await;
    insert_rows(&context, &table_ident).await;
    let mut compressions = Vec::new();
    for path in live_data_file_paths(&catalog, &table_ident).await {
        compressions.extend(column_chunk_compressions(&path));
    }
    compressions
}

fn assert_every_chunk_matches(
    compressions: &[Compression],
    expected: impl Fn(&Compression) -> bool,
) {
    assert!(
        !compressions.is_empty(),
        "expected at least one column chunk"
    );
    for compression in compressions {
        assert!(
            expected(compression),
            "unexpected column-chunk compression {compression:?}"
        );
    }
}

#[tokio::test]
async fn insert_into_default_table_writes_zstd() {
    let compressions =
        insert_column_chunk_compressions("insert_compression_default", "target", HashMap::new())
            .await;
    assert_every_chunk_matches(&compressions, |compression| {
        matches!(compression, Compression::ZSTD(_))
    });
}

#[tokio::test]
async fn insert_into_honours_codec_property() {
    for (namespace, codec) in [
        ("insert_compression_uncompressed", "uncompressed"),
        ("insert_compression_snappy", "snappy"),
        ("insert_compression_gzip", "gzip"),
    ] {
        let compressions = insert_column_chunk_compressions(
            namespace,
            "target",
            HashMap::from([(CODEC_KEY.to_string(), codec.to_string())]),
        )
        .await;
        match codec {
            "snappy" => assert_every_chunk_matches(&compressions, |compression| {
                matches!(compression, Compression::SNAPPY)
            }),
            "gzip" => assert_every_chunk_matches(&compressions, |compression| {
                matches!(compression, Compression::GZIP(_))
            }),
            "uncompressed" => assert_every_chunk_matches(&compressions, |compression| {
                matches!(compression, Compression::UNCOMPRESSED)
            }),
            other => panic!("unexpected codec fixture {other}"),
        }
    }
}

#[tokio::test]
async fn insert_into_honours_zstd_level() {
    let properties = HashMap::from([
        (CODEC_KEY.to_string(), "zstd".to_string()),
        (LEVEL_KEY.to_string(), "3".to_string()),
    ]);
    match iceberg::writer::file_writer::parquet_compression_from_properties(&properties)
        .expect("parse zstd level 3")
    {
        Compression::ZSTD(level) => {
            assert_eq!(level.compression_level(), 3);
        }
        other => panic!("expected ZSTD(3), got {other:?}"),
    }
    let compressions =
        insert_column_chunk_compressions("insert_compression_zstd_level", "target", properties)
            .await;
    assert_every_chunk_matches(&compressions, |compression| {
        matches!(compression, Compression::ZSTD(_))
    });
}

#[tokio::test]
async fn insert_into_unknown_codec_fails_loud() {
    let (context, _catalog, table_ident, _warehouse) = create_insert_fixture(
        "insert_compression_brotli",
        "target",
        HashMap::from([(CODEC_KEY.to_string(), "brotli".to_string())]),
    )
    .await;
    let sql = format!(
        "INSERT INTO catalog.{}.{} VALUES (1, 'alpha')",
        table_ident.namespace()[0],
        table_ident.name()
    );
    let error = match context.sql(&sql).await {
        Err(error) => error,
        Ok(dataframe) => dataframe
            .collect()
            .await
            .expect_err("brotli insert must fail"),
    };
    let message = error.to_string();
    assert!(
        message.contains(CODEC_KEY),
        "error must name the property key, got {message}"
    );
    assert!(
        message.contains("brotli"),
        "error must name the bad value, got {message}"
    );
}
