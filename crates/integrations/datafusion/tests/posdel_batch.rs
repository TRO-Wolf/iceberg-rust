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

use datafusion::arrow::array::Int32Array;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::execution::context::SessionContext;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, FormatVersion, NestedField, PrimitiveType, Schema, Type,
};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;
use tempfile::TempDir;

fn batch_properties(batch_rows: &str) -> HashMap<String, String> {
    HashMap::from([
        ("write.delete.mode".to_string(), "merge-on-read".to_string()),
        (
            "write.delete.position-delete-batch-rows".to_string(),
            batch_rows.to_string(),
        ),
        (
            "write.delete.target-file-size-bytes".to_string(),
            "1024".to_string(),
        ),
    ])
}

async fn create_fixture(
    namespace: &str,
    table_name: &str,
    batch_rows: &str,
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
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path)]),
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
            NestedField::required(2, "grp", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build schema");
    let table_ident = TableIdent::new(namespace.clone(), table_name.to_string());
    let location = format!(
        "{}/{table_name}",
        warehouse.path().to_str().expect("warehouse path is UTF-8")
    );
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name(table_name.to_string())
                .location(location)
                .schema(schema)
                .properties(batch_properties(batch_rows))
                .format_version(FormatVersion::V2)
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

fn table_sql(table_ident: &TableIdent) -> String {
    format!(
        "catalog.{}.{}",
        table_ident.namespace()[0],
        table_ident.name()
    )
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

async fn query_batches(context: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    context
        .sql(sql)
        .await
        .unwrap_or_else(|error| panic!("plan `{sql}`: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("execute `{sql}`: {error}"))
}

async fn insert_twenty_five(context: &SessionContext, table_ident: &TableIdent) {
    let values: Vec<String> = (1..=25).map(|id| format!("({id}, 'g{id}')")).collect();
    run_sql(
        context,
        &format!(
            "INSERT INTO {} VALUES {}",
            table_sql(table_ident),
            values.join(", ")
        ),
    )
    .await;
}

async fn live_files_by_content(
    catalog: &Arc<MemoryCatalog>,
    table_ident: &TableIdent,
    content: DataContentType,
) -> Vec<DataFile> {
    let table = catalog
        .load_table(table_ident)
        .await
        .expect("load the table");
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load the manifest list");
    let mut files = Vec::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("load the manifest");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.data_file().content_type() == content {
                files.push(entry.data_file().clone());
            }
        }
    }
    files.sort_by(|left, right| left.file_path().cmp(right.file_path()));
    files
}

async fn surviving_ids(context: &SessionContext, table_ident: &TableIdent) -> Vec<i32> {
    let sql = format!("SELECT id FROM {} ORDER BY id", table_sql(table_ident));
    let batches = query_batches(context, &sql).await;
    let mut ids = Vec::new();
    for batch in batches {
        let column = batch
            .column_by_name("id")
            .expect("id column")
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("id is Int32");
        for row in 0..column.len() {
            ids.push(column.value(row));
        }
    }
    ids.sort_unstable();
    ids
}

#[tokio::test]
async fn test_posdel_batch_rows_delete_all_reads_empty() {
    let (context, catalog, table_ident, _warehouse) =
        create_fixture("posdel_batch_delete_all", "target", "10").await;
    insert_twenty_five(&context, &table_ident).await;
    assert_eq!(
        live_files_by_content(&catalog, &table_ident, DataContentType::Data)
            .await
            .len(),
        1,
        "the fixture holds one data file"
    );
    run_sql(
        &context,
        &format!("DELETE FROM {} WHERE id > 0", table_sql(&table_ident)),
    )
    .await;
    assert_eq!(
        live_files_by_content(&catalog, &table_ident, DataContentType::PositionDeletes)
            .await
            .len(),
        1,
        "batched writes keep one position-delete file"
    );
    assert!(
        surviving_ids(&context, &table_ident).await.is_empty(),
        "no row survives a full delete"
    );
}

#[tokio::test]
async fn test_posdel_batch_rows_partial_delete_survivors_by_value() {
    let (context, catalog, table_ident, _warehouse) =
        create_fixture("posdel_batch_partial", "target", "10").await;
    insert_twenty_five(&context, &table_ident).await;
    assert_eq!(
        live_files_by_content(&catalog, &table_ident, DataContentType::Data)
            .await
            .len(),
        1,
        "the fixture holds one data file"
    );
    run_sql(
        &context,
        &format!("DELETE FROM {} WHERE id <= 13", table_sql(&table_ident)),
    )
    .await;
    assert_eq!(
        live_files_by_content(&catalog, &table_ident, DataContentType::PositionDeletes)
            .await
            .len(),
        1,
        "batched writes keep one position-delete file"
    );
    assert_eq!(
        surviving_ids(&context, &table_ident).await,
        (14..=25).collect::<Vec<i32>>(),
        "exactly the unpredicated rows survive"
    );
}

#[tokio::test]
async fn test_posdel_batch_rows_zero_refuses_delete() {
    let (context, _catalog, table_ident, _warehouse) =
        create_fixture("posdel_batch_zero", "target", "0").await;
    insert_twenty_five(&context, &table_ident).await;
    let result = context
        .sql(&format!(
            "DELETE FROM {} WHERE id > 0",
            table_sql(&table_ident)
        ))
        .await
        .expect("plan the delete")
        .collect()
        .await;
    let message = match result {
        Ok(_) => panic!("a zero batch-rows value must refuse the delete"),
        Err(error) => error.to_string(),
    };
    assert!(
        message.contains("write.delete.position-delete-batch-rows"),
        "the refusal names the property, got `{message}`"
    );
}
