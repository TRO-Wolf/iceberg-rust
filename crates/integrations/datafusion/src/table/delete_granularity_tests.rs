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
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result as DFResult;
use datafusion::prelude::SessionContext;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH;
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, ManifestContentType, ManifestStatus,
    NestedField, PartitionSpec, PrimitiveLiteral, PrimitiveType, Schema, Transform, Type,
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

fn cat_spec() -> PartitionSpec {
    PartitionSpec::builder(oracle_schema())
        .with_spec_id(0)
        .add_partition_field("cat", "cat", Transform::Identity)
        .expect("identity(cat) partition field")
        .build()
        .expect("cat spec")
}

async fn catalog_with_table(
    format_version: FormatVersion,
    spec: PartitionSpec,
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
    let namespace = NamespaceIdent::new("os_ns".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .unwrap();
    let creation = TableCreation::builder()
        .name("t".to_string())
        .schema(oracle_schema())
        .partition_spec(spec)
        .format_version(format_version)
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

async fn sql(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
    query: &str,
) -> DFResult<Vec<RecordBatch>> {
    let provider =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), "t".to_string())
            .await
            .expect("table provider");
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(provider))
        .expect("register");
    ctx.sql(query).await?.collect().await
}

async fn insert_values(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
    values: &str,
) -> DFResult<()> {
    sql(
        catalog,
        namespace,
        &format!("INSERT INTO t VALUES {values}"),
    )
    .await?;
    Ok(())
}

fn render_partition_field(field: &Option<iceberg::spec::Literal>) -> Option<String> {
    field
        .as_ref()
        .map(|literal| match literal.as_primitive_literal() {
            Some(PrimitiveLiteral::String(value)) => value.clone(),
            Some(PrimitiveLiteral::Int(value)) => value.to_string(),
            Some(PrimitiveLiteral::Long(value)) => value.to_string(),
            other => format!("{other:?}"),
        })
}

fn partition_values(file: &DataFile) -> Vec<Option<String>> {
    file.partition()
        .fields()
        .iter()
        .map(render_partition_field)
        .collect()
}

async fn added_delete_files(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
) -> Vec<DataFile> {
    let table = load_table(catalog, namespace).await;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("table has a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list loads");
    let mut files = Vec::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest loads");
        for entry in manifest.entries() {
            if entry.status() == ManifestStatus::Added {
                files.push(entry.data_file().clone());
            }
        }
    }
    files
}

async fn live_data_files(catalog: &Arc<dyn Catalog>, namespace: &NamespaceIdent) -> Vec<DataFile> {
    let table = load_table(catalog, namespace).await;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("table has a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list loads");
    let mut files = Vec::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest loads");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.content_type() == DataContentType::Data {
                files.push(entry.data_file().clone());
            }
        }
    }
    files
}

fn bound_path(file: &DataFile) -> String {
    let bound = file
        .lower_bounds()
        .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH)
        .expect("file_path lower bound")
        .to_bytes()
        .expect("bound bytes");
    String::from_utf8(bound.as_ref().to_vec()).expect("utf8 bound")
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
        .sql("SELECT id, data, cat FROM t")
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
    rows.sort();
    rows
}

async fn summary_value(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
    key: &str,
) -> Option<String> {
    let table = load_table(catalog, namespace).await;
    table
        .metadata()
        .current_snapshot()
        .expect("table has a current snapshot")
        .summary()
        .additional_properties
        .get(key)
        .cloned()
}

async fn snapshot_count(catalog: &Arc<dyn Catalog>, namespace: &NamespaceIdent) -> usize {
    load_table(catalog, namespace)
        .await
        .metadata()
        .snapshots()
        .len()
}

async fn files_under_data(catalog: &Arc<dyn Catalog>, namespace: &NamespaceIdent) -> Vec<String> {
    let table = load_table(catalog, namespace).await;
    let mut locations: Vec<String> = table
        .file_io()
        .list(format!("{}/data/", table.metadata().location()))
        .await
        .expect("list data dir")
        .into_iter()
        .map(|info| info.location)
        .collect();
    locations.sort();
    locations
}

fn mor_delete_properties() -> HashMap<String, String> {
    HashMap::from([("write.delete.mode".to_string(), "merge-on-read".to_string())])
}

fn mor_update_properties() -> HashMap<String, String> {
    HashMap::from([("write.update.mode".to_string(), "merge-on-read".to_string())])
}

async fn seed_two_files(catalog: &Arc<dyn Catalog>, namespace: &NamespaceIdent) {
    insert_values(catalog, namespace, "(1, 'a', 'x'), (2, 'b', 'y')")
        .await
        .expect("first insert");
    insert_values(catalog, namespace, "(3, 'c', 'x')")
        .await
        .expect("second insert");
}

#[tokio::test]
async fn file_granularity_writes_one_delete_file_per_data_file() {
    let mut properties = mor_delete_properties();
    properties.insert("write.delete.granularity".to_string(), "file".to_string());
    let (catalog, namespace, _guard) =
        catalog_with_table(FormatVersion::V2, unpartitioned_spec(), properties).await;
    seed_two_files(&catalog, &namespace).await;
    sql(&catalog, &namespace, "DELETE FROM t WHERE id IN (1, 3)")
        .await
        .expect("delete");
    let deletes = added_delete_files(&catalog, &namespace).await;
    assert_eq!(deletes.len(), 2);
    for file in &deletes {
        assert_eq!(file.record_count(), 1);
    }
    let mut bounds: Vec<String> = deletes.iter().map(bound_path).collect();
    bounds.sort();
    let mut data_paths: Vec<String> = live_data_files(&catalog, &namespace)
        .await
        .iter()
        .map(|file| file.file_path().to_string())
        .collect();
    data_paths.sort();
    assert_eq!(data_paths.len(), 2);
    assert_eq!(bounds, data_paths);
    for file in &deletes {
        assert_eq!(
            file.lower_bounds().get(&RESERVED_FIELD_ID_DELETE_FILE_PATH),
            file.upper_bounds().get(&RESERVED_FIELD_ID_DELETE_FILE_PATH)
        );
    }
    assert_eq!(
        summary_value(&catalog, &namespace, "added-delete-files")
            .await
            .as_deref(),
        Some("2")
    );
    assert_eq!(
        summary_value(&catalog, &namespace, "added-position-deletes")
            .await
            .as_deref(),
        Some("2")
    );
    assert_eq!(
        summary_value(&catalog, &namespace, "total-delete-files")
            .await
            .as_deref(),
        Some("2")
    );
    assert_eq!(
        summary_value(&catalog, &namespace, "total-position-deletes")
            .await
            .as_deref(),
        Some("2")
    );
    assert_eq!(rows_answer(&catalog, &namespace).await, vec![(
        2,
        "b".to_string(),
        "y".to_string()
    )]);
}

#[tokio::test]
async fn unset_granularity_defaults_to_file() {
    let (catalog, namespace, _guard) = catalog_with_table(
        FormatVersion::V2,
        unpartitioned_spec(),
        mor_delete_properties(),
    )
    .await;
    seed_two_files(&catalog, &namespace).await;
    sql(&catalog, &namespace, "DELETE FROM t WHERE id IN (1, 3)")
        .await
        .expect("delete");
    let deletes = added_delete_files(&catalog, &namespace).await;
    assert_eq!(deletes.len(), 2);
    for file in &deletes {
        assert_eq!(file.record_count(), 1);
    }
    assert_eq!(rows_answer(&catalog, &namespace).await, vec![(
        2,
        "b".to_string(),
        "y".to_string()
    )]);
}

#[tokio::test]
async fn partition_granularity_keeps_one_file() {
    let mut properties = mor_delete_properties();
    properties.insert(
        "write.delete.granularity".to_string(),
        "partition".to_string(),
    );
    let (catalog, namespace, _guard) =
        catalog_with_table(FormatVersion::V2, unpartitioned_spec(), properties).await;
    seed_two_files(&catalog, &namespace).await;
    sql(&catalog, &namespace, "DELETE FROM t WHERE id IN (1, 3)")
        .await
        .expect("delete");
    let deletes = added_delete_files(&catalog, &namespace).await;
    assert_eq!(deletes.len(), 1);
    assert_eq!(deletes[0].record_count(), 2);
    assert_eq!(rows_answer(&catalog, &namespace).await, vec![(
        2,
        "b".to_string(),
        "y".to_string()
    )]);
}

#[tokio::test]
async fn granularity_is_case_insensitive() {
    let mut properties = mor_delete_properties();
    properties.insert("write.delete.granularity".to_string(), "FILE".to_string());
    let (catalog, namespace, _guard) =
        catalog_with_table(FormatVersion::V2, unpartitioned_spec(), properties).await;
    seed_two_files(&catalog, &namespace).await;
    sql(&catalog, &namespace, "DELETE FROM t WHERE id IN (1, 3)")
        .await
        .expect("delete");
    let deletes = added_delete_files(&catalog, &namespace).await;
    assert_eq!(deletes.len(), 2);
    for file in &deletes {
        assert_eq!(file.record_count(), 1);
    }
}

#[tokio::test]
async fn unknown_granularity_refuses_before_writing() {
    let mut properties = mor_delete_properties();
    properties.insert("write.delete.granularity".to_string(), "row".to_string());
    let (catalog, namespace, _guard) =
        catalog_with_table(FormatVersion::V2, unpartitioned_spec(), properties).await;
    seed_two_files(&catalog, &namespace).await;
    let files_before = files_under_data(&catalog, &namespace).await;
    assert!(!files_before.is_empty());
    let snapshots_before = snapshot_count(&catalog, &namespace).await;
    let rows_before = rows_answer(&catalog, &namespace).await;
    let error = sql(&catalog, &namespace, "DELETE FROM t WHERE id IN (1, 3)")
        .await
        .expect_err("delete with unknown granularity must refuse");
    assert!(
        error.to_string().contains("write.delete.granularity"),
        "unexpected error: {error}"
    );
    assert!(
        error.to_string().contains("row"),
        "unexpected error: {error}"
    );
    assert_eq!(snapshot_count(&catalog, &namespace).await, snapshots_before);
    assert_eq!(rows_answer(&catalog, &namespace).await, rows_before);
    assert_eq!(files_under_data(&catalog, &namespace).await, files_before);
}

#[tokio::test]
async fn single_data_file_delete_is_unchanged() {
    let (catalog, namespace, _guard) = catalog_with_table(
        FormatVersion::V2,
        unpartitioned_spec(),
        mor_delete_properties(),
    )
    .await;
    seed_two_files(&catalog, &namespace).await;
    sql(&catalog, &namespace, "DELETE FROM t WHERE id = 2")
        .await
        .expect("delete");
    let deletes = added_delete_files(&catalog, &namespace).await;
    assert_eq!(deletes.len(), 1);
    assert_eq!(deletes[0].record_count(), 1);
    assert_eq!(rows_answer(&catalog, &namespace).await, vec![
        (1, "a".to_string(), "x".to_string()),
        (3, "c".to_string(), "x".to_string()),
    ]);
}

#[tokio::test]
async fn partitioned_file_granularity_stamps_each_file_with_its_partition() {
    let (catalog, namespace, _guard) =
        catalog_with_table(FormatVersion::V2, cat_spec(), mor_delete_properties()).await;
    seed_two_files(&catalog, &namespace).await;
    let data_files = live_data_files(&catalog, &namespace).await;
    assert_eq!(data_files.len(), 3);
    sql(&catalog, &namespace, "DELETE FROM t WHERE id IN (1, 2, 3)")
        .await
        .expect("delete");
    let deletes = added_delete_files(&catalog, &namespace).await;
    assert_eq!(deletes.len(), 3);
    let mut partitions: Vec<Vec<Option<String>>> = deletes.iter().map(partition_values).collect();
    partitions.sort();
    assert_eq!(partitions, vec![
        vec![Some("x".to_string())],
        vec![Some("x".to_string())],
        vec![Some("y".to_string())],
    ]);
    let expected: HashMap<String, Vec<Option<String>>> = data_files
        .iter()
        .map(|file| (file.file_path().to_string(), partition_values(file)))
        .collect();
    for file in &deletes {
        assert_eq!(file.record_count(), 1);
        assert_eq!(
            file.lower_bounds().get(&RESERVED_FIELD_ID_DELETE_FILE_PATH),
            file.upper_bounds().get(&RESERVED_FIELD_ID_DELETE_FILE_PATH)
        );
        assert_eq!(
            expected.get(&bound_path(file)),
            Some(&partition_values(file))
        );
    }
    assert!(rows_answer(&catalog, &namespace).await.is_empty());
}

#[tokio::test]
async fn mor_update_honours_file_granularity() {
    let (catalog, namespace, _guard) = catalog_with_table(
        FormatVersion::V2,
        unpartitioned_spec(),
        mor_update_properties(),
    )
    .await;
    seed_two_files(&catalog, &namespace).await;
    sql(
        &catalog,
        &namespace,
        "UPDATE t SET data = 'u' WHERE id IN (1, 3)",
    )
    .await
    .expect("update");
    let deletes = added_delete_files(&catalog, &namespace).await;
    assert_eq!(deletes.len(), 2);
    for file in &deletes {
        assert_eq!(file.record_count(), 1);
    }
    let id_data: Vec<(i64, String)> = rows_answer(&catalog, &namespace)
        .await
        .into_iter()
        .map(|(id, data, _)| (id, data))
        .collect();
    assert_eq!(id_data, vec![
        (1, "u".to_string()),
        (2, "b".to_string()),
        (3, "u".to_string()),
    ]);
}

#[tokio::test]
async fn v3_delete_still_writes_deletion_vectors() {
    let mut properties = mor_delete_properties();
    properties.insert("write.delete.granularity".to_string(), "file".to_string());
    let (catalog, namespace, _guard) =
        catalog_with_table(FormatVersion::V3, unpartitioned_spec(), properties).await;
    seed_two_files(&catalog, &namespace).await;
    sql(&catalog, &namespace, "DELETE FROM t WHERE id IN (1, 3)")
        .await
        .expect("delete");
    let deletes = added_delete_files(&catalog, &namespace).await;
    assert_eq!(deletes.len(), 2);
    for file in &deletes {
        assert_eq!(file.file_format(), DataFileFormat::Puffin);
    }
    assert_eq!(rows_answer(&catalog, &namespace).await, vec![(
        2,
        "b".to_string(),
        "y".to_string()
    )]);
}
