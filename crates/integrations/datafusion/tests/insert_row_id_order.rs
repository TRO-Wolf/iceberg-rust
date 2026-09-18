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

use datafusion::arrow::array::{Float64Array, Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use datafusion::prelude::SessionConfig;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, FormatVersion, Literal, ManifestContentType, NestedField, PrimitiveLiteral,
    PrimitiveType, Schema, Transform, Type, UnboundPartitionSpec,
};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;
use tempfile::TempDir;

const RUNS: usize = 20;
const TARGET_PARTITIONS: usize = 4;

struct WriteFixture {
    ctx: SessionContext,
    catalog: Arc<MemoryCatalog>,
    namespace: String,
    _warehouse: TempDir,
}

async fn write_fixture(namespace: &str, partitioned: bool) -> WriteFixture {
    let warehouse = TempDir::new().expect("warehouse");
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
        .expect("load catalog");
    let namespace_ident = NamespaceIdent::new(namespace.to_string());
    catalog
        .create_namespace(&namespace_ident, HashMap::new())
        .await
        .expect("namespace");
    let catalog = Arc::new(catalog);
    for run in 0..RUNS {
        create_table(
            &catalog,
            &namespace_ident,
            &warehouse_path,
            &format!("t{run}"),
            partitioned,
        )
        .await;
    }
    let provider = Arc::new(
        IcebergCatalogProvider::try_new(catalog.clone())
            .await
            .expect("provider"),
    );
    let ctx = SessionContext::new_with_config(
        SessionConfig::new().with_target_partitions(TARGET_PARTITIONS),
    );
    ctx.register_catalog("catalog", provider);
    WriteFixture {
        ctx,
        catalog,
        namespace: namespace.to_string(),
        _warehouse: warehouse,
    }
}

fn table_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "cat", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "v", Type::Primitive(PrimitiveType::Double)).into(),
        ])
        .build()
        .expect("schema")
}

async fn create_table(
    catalog: &Arc<MemoryCatalog>,
    namespace: &NamespaceIdent,
    warehouse_path: &str,
    name: &str,
    partitioned: bool,
) {
    let partition_spec = if partitioned {
        UnboundPartitionSpec::builder()
            .with_spec_id(0)
            .add_partition_field(2, "cat", Transform::Identity)
            .expect("identity(cat)")
            .build()
    } else {
        UnboundPartitionSpec::builder().with_spec_id(0).build()
    };
    catalog
        .create_table(
            namespace,
            TableCreation::builder()
                .name(name.to_string())
                .location(format!("{warehouse_path}/{name}"))
                .schema(table_schema())
                .partition_spec(partition_spec)
                .format_version(FormatVersion::V3)
                .build(),
        )
        .await
        .expect("create table");
}

fn arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("cat", DataType::Utf8, false),
        Field::new("v", DataType::Float64, false),
    ]))
}

fn cat_for(id: i64, modulus: i64) -> String {
    ((b'a' + (id.rem_euclid(modulus)) as u8) as char).to_string()
}

fn source_table(rows: i64, modulus: i64) -> MemTable {
    let per_partition = rows / TARGET_PARTITIONS as i64;
    let partitions = (0..TARGET_PARTITIONS as i64)
        .map(|partition| {
            let start = partition * per_partition;
            let ids: Vec<i64> = (start..start + per_partition).collect();
            let batch = RecordBatch::try_new(arrow_schema(), vec![
                Arc::new(Int64Array::from(ids.clone())),
                Arc::new(StringArray::from(
                    ids.iter()
                        .map(|id| cat_for(*id, modulus))
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    ids.iter().map(|id| *id as f64 + 0.5).collect::<Vec<_>>(),
                )),
            ])
            .expect("batch");
            vec![batch]
        })
        .collect();
    MemTable::try_new(arrow_schema(), partitions).expect("memtable")
}

async fn run_sql(fixture: &WriteFixture, sql: &str) {
    fixture
        .ctx
        .sql(sql)
        .await
        .unwrap_or_else(|error| panic!("plan `{sql}`: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("execute `{sql}`: {error}"));
}

async fn committed_files(fixture: &WriteFixture, table: &str) -> Vec<(String, u64, i64)> {
    let table = fixture
        .catalog
        .load_table(&TableIdent::new(
            NamespaceIdent::new(fixture.namespace.clone()),
            table.to_string(),
        ))
        .await
        .expect("load table");
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("insert commits one snapshot");
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
            if !entry.is_alive() || entry.data_file().content_type() != DataContentType::Data {
                continue;
            }
            let cat = match entry.data_file().partition().fields() {
                [Some(Literal::Primitive(PrimitiveLiteral::String(cat)))] => cat.clone(),
                [] => String::new(),
                other => panic!("unexpected partition shape {other:?}"),
            };
            files.push((
                cat,
                entry.data_file().record_count(),
                entry
                    .data_file()
                    .first_row_id()
                    .expect("v3 data file carries first_row_id"),
            ));
        }
    }
    files
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn insert_select_first_row_id_follows_ascending_partition_order() {
    let fixture = write_fixture("rowid_select", true).await;
    fixture
        .ctx
        .register_table("source", Arc::new(source_table(300, 3)))
        .expect("register source");
    let expected = vec![
        ("a".to_string(), 100, 0),
        ("b".to_string(), 100, 100),
        ("c".to_string(), 100, 200),
    ];
    for run in 0..RUNS {
        let table = format!("t{run}");
        run_sql(
            &fixture,
            &format!("INSERT INTO catalog.rowid_select.{table} SELECT id, cat, v FROM source"),
        )
        .await;
        let files = committed_files(&fixture, &table).await;
        assert_eq!(files, expected, "run {run} committed files out of order");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn insert_select_eight_partitions_first_row_id_ascending() {
    let fixture = write_fixture("rowid_eight", true).await;
    fixture
        .ctx
        .register_table("source", Arc::new(source_table(240, 8)))
        .expect("register source");
    let expected: Vec<(String, u64, i64)> = (0..8_i64)
        .map(|index| (cat_for(index, 8), 30, index * 30))
        .collect();
    for run in 0..RUNS {
        let table = format!("t{run}");
        run_sql(
            &fixture,
            &format!("INSERT INTO catalog.rowid_eight.{table} SELECT id, cat, v FROM source"),
        )
        .await;
        let files = committed_files(&fixture, &table).await;
        assert_eq!(files, expected, "run {run} committed files out of order");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn insert_values_first_row_id_follows_ascending_partition_order() {
    let fixture = write_fixture("rowid_values", true).await;
    let expected = vec![
        ("a".to_string(), 2, 0),
        ("b".to_string(), 2, 2),
        ("c".to_string(), 2, 4),
    ];
    for run in 0..RUNS {
        let table = format!("t{run}");
        run_sql(
            &fixture,
            &format!(
                "INSERT INTO catalog.rowid_values.{table} VALUES \
                 (1,'b',1.0),(2,'a',2.0),(3,'c',3.0),(4,'a',4.0),(5,'b',5.0),(6,'c',6.0)"
            ),
        )
        .await;
        let files = committed_files(&fixture, &table).await;
        assert_eq!(files, expected, "run {run} committed files out of order");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn insert_overwrite_first_row_id_follows_ascending_partition_order() {
    let fixture = write_fixture("rowid_overwrite", true).await;
    fixture
        .ctx
        .register_table("source", Arc::new(source_table(300, 3)))
        .expect("register source");
    let expected = vec![
        ("a".to_string(), 100, 0),
        ("b".to_string(), 100, 100),
        ("c".to_string(), 100, 200),
    ];
    for run in 0..RUNS {
        let table = format!("t{run}");
        run_sql(
            &fixture,
            &format!(
                "INSERT OVERWRITE catalog.rowid_overwrite.{table} SELECT id, cat, v FROM source"
            ),
        )
        .await;
        let files = committed_files(&fixture, &table).await;
        assert_eq!(files, expected, "run {run} committed files out of order");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn insert_unpartitioned_first_row_id_tiles_the_write() {
    let fixture = write_fixture("rowid_unpart", false).await;
    fixture
        .ctx
        .register_table("source", Arc::new(source_table(300, 3)))
        .expect("register source");
    for run in 0..RUNS {
        let table = format!("t{run}");
        run_sql(
            &fixture,
            &format!("INSERT INTO catalog.rowid_unpart.{table} SELECT id, cat, v FROM source"),
        )
        .await;
        let files = committed_files(&fixture, &table).await;
        assert_eq!(files.len(), 4, "run {run} committed file count");
        assert!(
            files.iter().all(|(_, count, _)| *count == 75),
            "run {run} per-file counts {files:?}"
        );
        let row_ids: Vec<i64> = files.iter().map(|(_, _, row_id)| *row_id).collect();
        assert_eq!(row_ids, vec![0, 75, 150, 225], "run {run} row-id tiling");
    }
}
