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

use iceberg::io::{FileIO, LocalFsStorageFactory};
use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation, TableIdent};
use sqlx::migrate::MigrateDatabase;
use tempfile::TempDir;

use crate::catalog::{
    SQL_CATALOG_PROP_BIND_STYLE, SQL_CATALOG_PROP_URI, SQL_CATALOG_PROP_WAREHOUSE,
};
use crate::{SqlBindStyle, SqlCatalogBuilder};

async fn new_sql_catalog(db_path: &str, warehouse: &str) -> impl Catalog {
    let uri = format!("sqlite:{db_path}");
    sqlx::Sqlite::create_database(&uri)
        .await
        .expect("create sqlite database");
    let props = HashMap::from_iter([
        (SQL_CATALOG_PROP_URI.to_string(), uri),
        (
            SQL_CATALOG_PROP_WAREHOUSE.to_string(),
            warehouse.to_string(),
        ),
        (
            SQL_CATALOG_PROP_BIND_STYLE.to_string(),
            SqlBindStyle::DollarNumeric.to_string(),
        ),
    ]);
    SqlCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load("iceberg", props)
        .await
        .expect("sql catalog loads")
}

fn test_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "foo", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()
        .expect("schema builds")
}

#[tokio::test]
async fn hadoop_planted_v3_conflicts_through_sql_commit() {
    let dir = TempDir::new().expect("tempdir");
    let warehouse = dir.path().join("wh");
    let warehouse_str = warehouse.to_str().expect("utf8 path").to_string();
    let db_path = dir
        .path()
        .join("catalog.db")
        .to_str()
        .expect("utf8 path")
        .to_string();
    let file_io = FileIO::new_with_fs();

    let catalog = new_sql_catalog(&db_path, &warehouse_str).await;
    let ns = NamespaceIdent::new("ns".to_string());
    catalog
        .create_namespace(&ns, HashMap::new())
        .await
        .expect("namespace creates");
    let table_location = dir
        .path()
        .join("wh")
        .join("ns")
        .join("src")
        .to_str()
        .expect("utf8 path")
        .to_string();
    let source = catalog
        .create_table(
            &ns,
            TableCreation::builder()
                .name("src".to_string())
                .schema(test_schema())
                .location(table_location.clone())
                .build(),
        )
        .await
        .expect("source table creates");

    let v2 = format!("{table_location}/metadata/v2.metadata.json");
    source
        .metadata()
        .write_to(&file_io, &v2)
        .await
        .expect("seed v2");
    let ident = TableIdent::new(ns, "hadoop".to_string());
    catalog
        .register_table(&ident, v2.clone())
        .await
        .expect("register v2");

    let v3 = format!("{table_location}/metadata/v3.metadata.json");
    std::fs::copy(&v2, &v3).expect("plant v3");
    let planted_bytes = std::fs::read(&v3).expect("read planted v3");

    let table = catalog.load_table(&ident).await.expect("load base");
    let tx = Transaction::new(&table);
    let err = tx
        .update_table_properties()
        .set("writer".to_string(), "one".to_string())
        .apply(tx)
        .expect("apply stages")
        .commit(&catalog)
        .await
        .expect_err("planted v3 must conflict through the sql commit path");
    assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
    assert!(err.retryable());
    assert_eq!(
        std::fs::read(&v3).expect("planted v3 intact"),
        planted_bytes
    );
    let stuck = catalog.load_table(&ident).await.expect("loads");
    assert_eq!(stuck.metadata_location().expect("pointer"), v2.as_str());
}
