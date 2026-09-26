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

use datafusion::arrow::array::{Array, Int64Array, StringArray};
use datafusion::error::Result as DFResult;
use datafusion::prelude::SessionContext;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    FormatVersion, ManifestContentType, ManifestStatus, NestedField, PartitionSpec, PrimitiveType,
    Schema, Type,
};
use iceberg::table::Table;
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use tempfile::TempDir;

use super::IcebergTableProvider;

fn oracle_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "cat", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("oracle schema")
}

fn unpartitioned_spec() -> PartitionSpec {
    PartitionSpec::builder(oracle_schema())
        .with_spec_id(0)
        .build()
        .expect("unpartitioned spec")
}

async fn catalog_with_table(
    properties: HashMap<String, String>,
) -> (Arc<dyn Catalog>, NamespaceIdent, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let catalog = MemoryCatalogBuilder::default()
        .load(
            "memory",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                temp_dir.path().to_str().unwrap().to_string(),
            )]),
        )
        .await
        .unwrap();
    let namespace = NamespaceIdent::new("merge_ns".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .unwrap();
    let creation = TableCreation::builder()
        .name("t".to_string())
        .schema(oracle_schema())
        .partition_spec(unpartitioned_spec())
        .format_version(FormatVersion::V2)
        .properties(properties)
        .build();
    catalog.create_table(&namespace, creation).await.unwrap();
    (Arc::new(catalog), namespace, temp_dir)
}

async fn load_table(catalog: &Arc<dyn Catalog>, namespace: &NamespaceIdent) -> Table {
    catalog
        .load_table(&TableIdent::new(namespace.clone(), "t".to_string()))
        .await
        .unwrap()
}

async fn insert_values(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
    values: &str,
) -> DFResult<()> {
    let provider =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), "t".to_string())
            .await
            .unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(provider)).unwrap();
    let df = ctx.sql(&format!("INSERT INTO t VALUES {values}")).await?;
    df.collect().await?;
    Ok(())
}

async fn three_inserts(catalog: &Arc<dyn Catalog>, namespace: &NamespaceIdent) {
    insert_values(catalog, namespace, "(1, 'a', 'x'), (2, 'b', 'y')")
        .await
        .unwrap();
    insert_values(catalog, namespace, "(3, 'c', 'x')")
        .await
        .unwrap();
    insert_values(catalog, namespace, "(4, 'd', 'z')")
        .await
        .unwrap();
}

async fn data_manifest_count(catalog: &Arc<dyn Catalog>, namespace: &NamespaceIdent) -> usize {
    let table = load_table(catalog, namespace).await;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("table has a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), &table.metadata_ref())
        .await
        .expect("manifest list loads");
    manifest_list
        .entries()
        .iter()
        .filter(|manifest| manifest.content == ManifestContentType::Data)
        .count()
}

async fn data_manifest_entries(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
) -> Vec<(ManifestStatus, Option<i64>, Option<i64>)> {
    let table = load_table(catalog, namespace).await;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("table has a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), &table.metadata_ref())
        .await
        .expect("manifest list loads");
    let data: Vec<_> = manifest_list
        .entries()
        .iter()
        .filter(|manifest| manifest.content == ManifestContentType::Data)
        .collect();
    assert_eq!(data.len(), 1);
    let manifest = data[0]
        .load_manifest(table.file_io())
        .await
        .expect("manifest loads");
    manifest
        .entries()
        .iter()
        .map(|entry| (entry.status(), entry.sequence_number(), entry.snapshot_id()))
        .collect()
}

async fn current_snapshot_id(catalog: &Arc<dyn Catalog>, namespace: &NamespaceIdent) -> i64 {
    load_table(catalog, namespace)
        .await
        .metadata()
        .current_snapshot()
        .expect("table has a current snapshot")
        .snapshot_id()
}

async fn current_summary_triple(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
) -> (Option<String>, Option<String>, Option<String>) {
    let table = load_table(catalog, namespace).await;
    let summary = table
        .metadata()
        .current_snapshot()
        .expect("table has a current snapshot")
        .summary();
    (
        summary
            .additional_properties
            .get("manifests-created")
            .cloned(),
        summary.additional_properties.get("manifests-kept").cloned(),
        summary
            .additional_properties
            .get("manifests-replaced")
            .cloned(),
    )
}

async fn rows_answer(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
) -> Vec<(i64, String, String)> {
    let provider =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), "t".to_string())
            .await
            .unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(provider)).unwrap();
    let batches = ctx
        .sql("SELECT id, data, cat FROM t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("id is Int64");
        let data = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("data is Utf8");
        let cats = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("cat is Utf8");
        for i in 0..batch.num_rows() {
            rows.push((
                ids.value(i),
                data.value(i).to_string(),
                cats.value(i).to_string(),
            ));
        }
    }
    rows
}

fn oracle_rows() -> Vec<(i64, String, String)> {
    vec![
        (1, "a".to_string(), "x".to_string()),
        (2, "b".to_string(), "y".to_string()),
        (3, "c".to_string(), "x".to_string()),
        (4, "d".to_string(), "z".to_string()),
    ]
}

fn min_count_properties(value: &str) -> HashMap<String, String> {
    HashMap::from([(
        "commit.manifest.min-count-to-merge".to_string(),
        value.to_string(),
    )])
}

#[tokio::test]
async fn min_count_2_three_inserts_leave_one_manifest() {
    let (catalog, namespace, _guard) = catalog_with_table(min_count_properties("2")).await;
    three_inserts(&catalog, &namespace).await;
    assert_eq!(data_manifest_count(&catalog, &namespace).await, 1);
    assert_eq!(rows_answer(&catalog, &namespace).await, oracle_rows());
}

#[tokio::test]
async fn min_count_2_merged_manifest_keeps_provenance() {
    let (catalog, namespace, _guard) = catalog_with_table(min_count_properties("2")).await;
    insert_values(&catalog, &namespace, "(1, 'a', 'x'), (2, 'b', 'y')")
        .await
        .unwrap();
    let first_snapshot = current_snapshot_id(&catalog, &namespace).await;
    insert_values(&catalog, &namespace, "(3, 'c', 'x')")
        .await
        .unwrap();
    let second_snapshot = current_snapshot_id(&catalog, &namespace).await;
    insert_values(&catalog, &namespace, "(4, 'd', 'z')")
        .await
        .unwrap();
    let third_snapshot = current_snapshot_id(&catalog, &namespace).await;
    assert_eq!(data_manifest_entries(&catalog, &namespace).await, vec![
        (ManifestStatus::Added, Some(3), Some(third_snapshot)),
        (ManifestStatus::Existing, Some(2), Some(second_snapshot)),
        (ManifestStatus::Existing, Some(1), Some(first_snapshot)),
    ]);
}

#[tokio::test]
async fn default_properties_three_inserts_keep_three_manifests() {
    let (catalog, namespace, _guard) = catalog_with_table(HashMap::new()).await;
    three_inserts(&catalog, &namespace).await;
    assert_eq!(data_manifest_count(&catalog, &namespace).await, 3);
    assert_eq!(rows_answer(&catalog, &namespace).await, oracle_rows());
}

#[tokio::test]
async fn merge_disabled_keeps_every_manifest() {
    let properties = HashMap::from([
        (
            "commit.manifest-merge.enabled".to_string(),
            "false".to_string(),
        ),
        (
            "commit.manifest.min-count-to-merge".to_string(),
            "2".to_string(),
        ),
    ]);
    let (catalog, namespace, _guard) = catalog_with_table(properties).await;
    three_inserts(&catalog, &namespace).await;
    assert_eq!(data_manifest_count(&catalog, &namespace).await, 3);
    assert_eq!(rows_answer(&catalog, &namespace).await, oracle_rows());
}

#[tokio::test]
async fn tiny_target_size_keeps_three_manifests() {
    let properties = HashMap::from([(
        "commit.manifest.target-size-bytes".to_string(),
        "100".to_string(),
    )]);
    let (catalog, namespace, _guard) = catalog_with_table(properties).await;
    three_inserts(&catalog, &namespace).await;
    assert_eq!(data_manifest_count(&catalog, &namespace).await, 3);
    assert_eq!(rows_answer(&catalog, &namespace).await, oracle_rows());
}

#[tokio::test]
async fn min_count_5_series_matches_spark() {
    let (catalog, namespace, _guard) = catalog_with_table(min_count_properties("5")).await;
    let mut counts = Vec::new();
    let mut triples = Vec::new();
    for index in 1..=20 {
        insert_values(&catalog, &namespace, &format!("({index}, 'v', 'x')"))
            .await
            .unwrap();
        counts.push(data_manifest_count(&catalog, &namespace).await);
        triples.push(current_summary_triple(&catalog, &namespace).await);
    }
    assert_eq!(counts[0], 1);
    assert_eq!(counts[4], 1);
    assert_eq!(counts[19], 4);
    assert_eq!(
        triples[0],
        (
            Some("1".to_string()),
            Some("0".to_string()),
            Some("0".to_string())
        )
    );
    assert_eq!(
        triples[4],
        (
            Some("1".to_string()),
            Some("0".to_string()),
            Some("4".to_string())
        )
    );
    assert_eq!(
        triples[19],
        (
            Some("1".to_string()),
            Some("3".to_string()),
            Some("0".to_string())
        )
    );
}
