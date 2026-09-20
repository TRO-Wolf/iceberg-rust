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
    DataContentType, FormatVersion, NestedField, PartitionSpec, PrimitiveLiteral, PrimitiveType,
    Schema, Transform, Type,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
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

async fn evolve_spec(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
    add_fields: &[(&str, Transform)],
    remove_fields: &[&str],
) {
    let table = load_table(catalog, namespace).await;
    let tx = Transaction::new(&table);
    let mut action = tx.update_partition_spec();
    for &(source_name, ref transform) in add_fields {
        action = if matches!(transform, Transform::Identity) {
            action.add_field(source_name)
        } else {
            action.add_field_with_transform(None, source_name, *transform)
        };
    }
    for name in remove_fields {
        action = action.remove_field(name);
    }
    let tx = action.apply(tx).unwrap();
    tx.commit(catalog.as_ref()).await.unwrap();
}

async fn insert_values(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
    output_spec_id: Option<i32>,
    values: &str,
) -> DFResult<()> {
    let mut provider =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), "t".to_string())
            .await
            .unwrap();
    if let Some(spec_id) = output_spec_id {
        provider = provider.with_output_spec_id(spec_id);
    }
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(provider)).unwrap();
    let df = ctx.sql(&format!("INSERT INTO t VALUES {values}")).await?;
    df.collect().await?;
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

async fn files_answer(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
) -> Vec<(i32, Vec<Option<String>>, u64)> {
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
                let data_file = entry.data_file();
                files.push((
                    data_file.partition_spec_id(),
                    data_file
                        .partition()
                        .fields()
                        .iter()
                        .map(render_partition_field)
                        .collect(),
                    data_file.record_count(),
                ));
            }
        }
    }
    files.sort();
    files
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

fn oracle_rows() -> Vec<(i64, String, String)> {
    vec![
        (1, "a".to_string(), "x".to_string()),
        (2, "b".to_string(), "y".to_string()),
        (7, "g".to_string(), "x".to_string()),
        (8, "h".to_string(), "w".to_string()),
    ]
}

async fn pin_insert_into_targets_older_spec(format_version: FormatVersion) {
    let (catalog, namespace, _guard) =
        catalog_with_table(format_version, unpartitioned_spec()).await;
    evolve_spec(&catalog, &namespace, &[("cat", Transform::Identity)], &[]).await;
    let table = load_table(&catalog, &namespace).await;
    assert_eq!(table.metadata().default_partition_spec_id(), 1);

    insert_values(
        &catalog,
        &namespace,
        Some(0),
        "(1, 'a', 'x'), (2, 'b', 'y')",
    )
    .await
    .unwrap();
    insert_values(
        &catalog,
        &namespace,
        Some(0),
        "(7, 'g', 'x'), (8, 'h', 'w')",
    )
    .await
    .unwrap();

    assert_eq!(files_answer(&catalog, &namespace).await, vec![
        (0, vec![], 2),
        (0, vec![], 2),
    ]);
    assert_eq!(rows_answer(&catalog, &namespace).await, oracle_rows());
}

#[tokio::test]
async fn pin_insert_into_targets_older_spec_v2() {
    pin_insert_into_targets_older_spec(FormatVersion::V2).await;
}

#[tokio::test]
async fn pin_insert_into_targets_older_spec_v3() {
    pin_insert_into_targets_older_spec(FormatVersion::V3).await;
}

async fn pin_insert_into_targets_new_spec(format_version: FormatVersion) {
    let (catalog, namespace, _guard) =
        catalog_with_table(format_version, unpartitioned_spec()).await;
    evolve_spec(&catalog, &namespace, &[("cat", Transform::Identity)], &[]).await;

    insert_values(
        &catalog,
        &namespace,
        Some(0),
        "(1, 'a', 'x'), (2, 'b', 'y')",
    )
    .await
    .unwrap();
    insert_values(
        &catalog,
        &namespace,
        Some(1),
        "(7, 'g', 'x'), (8, 'h', 'w')",
    )
    .await
    .unwrap();

    assert_eq!(files_answer(&catalog, &namespace).await, vec![
        (0, vec![], 2),
        (1, vec![Some("w".to_string())], 1),
        (1, vec![Some("x".to_string())], 1),
    ]);
    assert_eq!(rows_answer(&catalog, &namespace).await, oracle_rows());
}

#[tokio::test]
async fn pin_insert_into_targets_new_spec_v2() {
    pin_insert_into_targets_new_spec(FormatVersion::V2).await;
}

#[tokio::test]
async fn pin_insert_into_targets_new_spec_v3() {
    pin_insert_into_targets_new_spec(FormatVersion::V3).await;
}

async fn pin_insert_into_partitioned_older_spec(format_version: FormatVersion) {
    let (catalog, namespace, _guard) = catalog_with_table(format_version, cat_spec()).await;
    insert_values(&catalog, &namespace, None, "(1, 'a', 'x'), (2, 'b', 'y')")
        .await
        .unwrap();
    evolve_spec(&catalog, &namespace, &[], &["cat"]).await;
    let table = load_table(&catalog, &namespace).await;
    assert!(table.metadata().default_partition_spec().is_unpartitioned());

    insert_values(
        &catalog,
        &namespace,
        Some(0),
        "(7, 'g', 'x'), (8, 'h', 'w')",
    )
    .await
    .unwrap();

    assert_eq!(files_answer(&catalog, &namespace).await, vec![
        (0, vec![Some("w".to_string())], 1),
        (0, vec![Some("x".to_string())], 1),
        (0, vec![Some("x".to_string())], 1),
        (0, vec![Some("y".to_string())], 1),
    ]);
    assert_eq!(rows_answer(&catalog, &namespace).await, oracle_rows());
}

#[tokio::test]
async fn pin_insert_into_partitioned_older_spec_v2() {
    pin_insert_into_partitioned_older_spec(FormatVersion::V2).await;
}

#[tokio::test]
async fn pin_insert_into_partitioned_older_spec_v3() {
    pin_insert_into_partitioned_older_spec(FormatVersion::V3).await;
}

async fn pin_insert_into_two_field_spec(format_version: FormatVersion) {
    let (catalog, namespace, _guard) =
        catalog_with_table(format_version, unpartitioned_spec()).await;
    evolve_spec(
        &catalog,
        &namespace,
        &[("cat", Transform::Identity), ("id", Transform::Bucket(2))],
        &[],
    )
    .await;
    let table = load_table(&catalog, &namespace).await;
    assert_eq!(table.metadata().default_partition_spec().fields().len(), 2);

    insert_values(
        &catalog,
        &namespace,
        Some(0),
        "(1, 'a', 'x'), (2, 'b', 'y')",
    )
    .await
    .unwrap();
    insert_values(
        &catalog,
        &namespace,
        Some(1),
        "(7, 'g', 'x'), (8, 'h', 'w')",
    )
    .await
    .unwrap();

    let files = files_answer(&catalog, &namespace).await;
    assert_eq!(files.len(), 3);
    assert_eq!(files[0], (0, vec![], 2));
    let mut partitioned: Vec<&Vec<Option<String>>> =
        files[1..].iter().map(|(_, tuple, _)| tuple).collect();
    partitioned.sort();
    assert_eq!(partitioned.len(), 2);
    for (index, tuple) in partitioned.iter().enumerate() {
        assert_eq!(files[index + 1].0, 1);
        assert_eq!(files[index + 1].2, 1);
        assert_eq!(tuple.len(), 2);
        let bucket = tuple[1]
            .as_ref()
            .expect("bucket value present")
            .parse::<i32>()
            .expect("bucket is an int");
        assert!((0..2).contains(&bucket));
    }
    assert_eq!(
        partitioned
            .iter()
            .map(|tuple| tuple[0].clone())
            .collect::<Vec<_>>(),
        vec![Some("w".to_string()), Some("x".to_string())]
    );
    assert_eq!(rows_answer(&catalog, &namespace).await, oracle_rows());
}

#[tokio::test]
async fn pin_insert_into_two_field_spec_v2() {
    pin_insert_into_two_field_spec(FormatVersion::V2).await;
}

#[tokio::test]
async fn pin_insert_into_two_field_spec_v3() {
    pin_insert_into_two_field_spec(FormatVersion::V3).await;
}

async fn dml_values(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
    output_spec_id: Option<i32>,
    sql: &str,
) -> DFResult<()> {
    let mut provider =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), "t".to_string())
            .await
            .unwrap();
    if let Some(spec_id) = output_spec_id {
        provider = provider.with_output_spec_id(spec_id);
    }
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(provider)).unwrap();
    let df = ctx.sql(sql).await?;
    df.collect().await?;
    Ok(())
}

async fn pin_delete_copy_on_write_targets_older_spec(format_version: FormatVersion) {
    let (catalog, namespace, _guard) =
        catalog_with_table(format_version, unpartitioned_spec()).await;
    evolve_spec(&catalog, &namespace, &[("cat", Transform::Identity)], &[]).await;
    let table = load_table(&catalog, &namespace).await;
    assert_eq!(table.metadata().default_partition_spec_id(), 1);

    insert_values(
        &catalog,
        &namespace,
        Some(0),
        "(1, 'a', 'x'), (2, 'b', 'y')",
    )
    .await
    .unwrap();

    dml_values(&catalog, &namespace, Some(0), "DELETE FROM t WHERE id = 1")
        .await
        .unwrap();

    assert_eq!(files_answer(&catalog, &namespace).await, vec![(
        0,
        vec![],
        1
    )]);
    assert_eq!(rows_answer(&catalog, &namespace).await, vec![(
        2,
        "b".to_string(),
        "y".to_string()
    )]);
}

#[tokio::test]
async fn pin_delete_copy_on_write_targets_older_spec_v2() {
    pin_delete_copy_on_write_targets_older_spec(FormatVersion::V2).await;
}

#[tokio::test]
async fn pin_delete_copy_on_write_targets_older_spec_v3() {
    pin_delete_copy_on_write_targets_older_spec(FormatVersion::V3).await;
}

async fn pin_update_copy_on_write_targets_older_spec(format_version: FormatVersion) {
    let (catalog, namespace, _guard) =
        catalog_with_table(format_version, unpartitioned_spec()).await;
    evolve_spec(&catalog, &namespace, &[("cat", Transform::Identity)], &[]).await;
    let table = load_table(&catalog, &namespace).await;
    assert_eq!(table.metadata().default_partition_spec_id(), 1);

    insert_values(
        &catalog,
        &namespace,
        Some(0),
        "(1, 'a', 'x'), (2, 'b', 'y')",
    )
    .await
    .unwrap();

    dml_values(
        &catalog,
        &namespace,
        Some(0),
        "UPDATE t SET data = 'z' WHERE id = 1",
    )
    .await
    .unwrap();

    assert_eq!(files_answer(&catalog, &namespace).await, vec![(
        0,
        vec![],
        2
    )]);
    assert_eq!(rows_answer(&catalog, &namespace).await, vec![
        (1, "z".to_string(), "x".to_string()),
        (2, "b".to_string(), "y".to_string()),
    ]);
}

#[tokio::test]
async fn pin_update_copy_on_write_targets_older_spec_v2() {
    pin_update_copy_on_write_targets_older_spec(FormatVersion::V2).await;
}

#[tokio::test]
async fn pin_update_copy_on_write_targets_older_spec_v3() {
    pin_update_copy_on_write_targets_older_spec(FormatVersion::V3).await;
}

async fn pin_insert_into_bad_spec_id(format_version: FormatVersion) {
    let (catalog, namespace, _guard) =
        catalog_with_table(format_version, unpartitioned_spec()).await;
    evolve_spec(&catalog, &namespace, &[("cat", Transform::Identity)], &[]).await;

    let err = insert_values(
        &catalog,
        &namespace,
        Some(9),
        "(7, 'g', 'x'), (8, 'h', 'w')",
    )
    .await
    .expect_err("an unknown output spec id must fail");
    assert!(
        err.to_string()
            .contains("Output spec id 9 is not a valid spec id for table"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn pin_insert_into_bad_spec_id_v2() {
    pin_insert_into_bad_spec_id(FormatVersion::V2).await;
}

#[tokio::test]
async fn pin_insert_into_bad_spec_id_v3() {
    pin_insert_into_bad_spec_id(FormatVersion::V3).await;
}
