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

//! Integration tests for Iceberg Datafusion with Hive Metastore.

use std::collections::HashMap;
use std::sync::Arc;
use std::vec;

use datafusion::arrow::array::{Array, Int32Array, StringArray, UInt64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::execution::context::SessionContext;
use datafusion::parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use expect_test::expect;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    NestedField, PrimitiveType, Schema, StructType, Transform, Type, UnboundPartitionSpec,
};
use iceberg::test_utils::check_record_batches;
use iceberg::{
    Catalog, CatalogBuilder, MemoryCatalog, NamespaceIdent, Result, TableCreation, TableIdent,
};
use iceberg_datafusion::IcebergCatalogProvider;
use tempfile::TempDir;

fn temp_path() -> String {
    let temp_dir = TempDir::new().unwrap();
    temp_dir.path().to_str().unwrap().to_string()
}

async fn get_iceberg_catalog() -> MemoryCatalog {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), temp_path())]),
        )
        .await
        .unwrap()
}

fn get_struct_type() -> StructType {
    StructType::new(vec![
        NestedField::required(4, "s_foo1", Type::Primitive(PrimitiveType::Int)).into(),
        NestedField::required(5, "s_foo2", Type::Primitive(PrimitiveType::String)).into(),
    ])
}

async fn set_test_namespace(catalog: &MemoryCatalog, namespace: &NamespaceIdent) -> Result<()> {
    let properties = HashMap::new();

    catalog.create_namespace(namespace, properties).await?;

    Ok(())
}

fn get_table_creation(
    location: impl ToString,
    name: impl ToString,
    schema: Option<Schema>,
) -> Result<TableCreation> {
    let schema = match schema {
        None => Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "foo1", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "foo2", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()?,
        Some(schema) => schema,
    };

    let creation = TableCreation::builder()
        .location(location.to_string())
        .name(name.to_string())
        .properties(HashMap::new())
        .schema(schema)
        .build();

    Ok(creation)
}

/// A `{foo1 int, foo2 string}` table creation with `write.delete.mode` and `write.update.mode` both set
/// to `merge-on-read`, so `DELETE`/`UPDATE` use the position-delete (`RowDelta`) path.
fn get_merge_on_read_table_creation(
    location: impl ToString,
    name: impl ToString,
) -> Result<TableCreation> {
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "foo1", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "foo2", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()?;
    Ok(TableCreation::builder()
        .location(location.to_string())
        .name(name.to_string())
        .properties(HashMap::from([
            ("write.delete.mode".to_string(), "merge-on-read".to_string()),
            ("write.update.mode".to_string(), "merge-on-read".to_string()),
        ]))
        .schema(schema)
        .build())
}

#[tokio::test]
async fn test_provider_plan_stream_schema() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_provider_get_table_schema".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let creation = get_table_creation(temp_path(), "my_table", None)?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    let provider = ctx.catalog("catalog").unwrap();
    let schema = provider.schema("test_provider_get_table_schema").unwrap();

    let table = schema.table("my_table").await.unwrap().unwrap();
    let table_schema = table.schema();

    let expected = [("foo1", &DataType::Int32), ("foo2", &DataType::Utf8)];

    for (field, exp) in table_schema.fields().iter().zip(expected.iter()) {
        assert_eq!(field.name(), exp.0);
        assert_eq!(field.data_type(), exp.1);
        assert!(!field.is_nullable())
    }

    let df = ctx
        .sql("select foo2 from catalog.test_provider_get_table_schema.my_table")
        .await
        .unwrap();

    let task_ctx = Arc::new(df.task_ctx());
    let plan = df.create_physical_plan().await.unwrap();
    let stream = plan.execute(1, task_ctx).unwrap();

    // Ensure both the plan and the stream conform to the same schema
    assert_eq!(plan.schema(), stream.schema());
    assert_eq!(
        stream.schema().as_ref(),
        &ArrowSchema::new(vec![
            Field::new("foo2", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )]))
        ]),
    );

    Ok(())
}

#[tokio::test]
async fn test_provider_list_table_names() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_provider_list_table_names".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let creation = get_table_creation(temp_path(), "my_table", None)?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    let provider = ctx.catalog("catalog").unwrap();
    let schema = provider.schema("test_provider_list_table_names").unwrap();

    let result = schema.table_names();

    expect![[r#"
        [
            "my_table",
            "my_table$snapshots",
            "my_table$manifests",
            "my_table$files",
            "my_table$data_files",
            "my_table$delete_files",
            "my_table$entries",
            "my_table$all_files",
            "my_table$all_data_files",
            "my_table$all_delete_files",
            "my_table$all_entries",
            "my_table$history",
            "my_table$refs",
            "my_table$metadata_log_entries",
            "my_table$partitions",
            "my_table$all_manifests",
        ]
    "#]]
    .assert_debug_eq(&result);

    Ok(())
}

#[tokio::test]
async fn test_provider_list_schema_names() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_provider_list_schema_names".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    let provider = ctx.catalog("catalog").unwrap();

    let expected = ["test_provider_list_schema_names"];
    let result = provider.schema_names();

    assert!(
        expected
            .iter()
            .all(|item| result.contains(&item.to_string()))
    );
    Ok(())
}

#[tokio::test]
async fn test_table_projection() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("ns".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "foo1", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "foo2", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "foo3", Type::Struct(get_struct_type())).into(),
        ])
        .build()?;
    let creation = get_table_creation(temp_path(), "t1", Some(schema))?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);
    let table_df = ctx.table("catalog.ns.t1").await.unwrap();

    let records = table_df
        .clone()
        .explain(false, false)
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(1, records.len());
    let record = &records[0];
    // the first column is plan_type, the second column plan string.
    let s = record
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(2, s.len());
    // the first row is logical_plan, the second row is physical_plan
    assert!(s.value(1).contains("projection:[foo1,foo2,foo3]"));

    // datafusion doesn't support query foo3.s_foo1, use foo3 instead
    let records = table_df
        .select_columns(&["foo1", "foo3"])
        .unwrap()
        .explain(false, false)
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(1, records.len());
    let record = &records[0];
    let s = record
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(2, s.len());
    assert!(
        s.value(1)
            .contains("IcebergTableScan projection:[foo1,foo3]")
    );

    Ok(())
}

#[tokio::test]
async fn test_table_predict_pushdown() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("ns".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "foo", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "bar", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()?;
    let creation = get_table_creation(temp_path(), "t1", Some(schema))?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);
    let records = ctx
        .sql("select * from catalog.ns.t1 where (foo > 1 and length(bar) = 1 ) or bar is null")
        .await
        .unwrap()
        .explain(false, false)
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(1, records.len());
    let record = &records[0];
    // the first column is plan_type, the second column plan string.
    let s = record
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(2, s.len());
    // the first row is logical_plan, the second row is physical_plan
    let expected = "predicate:[(foo > 1) OR (bar IS NULL)]";
    assert!(s.value(1).trim().contains(expected));
    Ok(())
}

#[tokio::test]
async fn test_metadata_table() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("ns".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "foo", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "bar", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()?;
    let creation = get_table_creation(temp_path(), "t1", Some(schema))?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);
    let snapshots = ctx
        .sql("select * from catalog.ns.t1$snapshots")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    check_record_batches(
        snapshots,
        expect![[r#"
            Field { "committed_at": Timestamp(µs, "+00:00"), metadata: {"PARQUET:field_id": "1"} },
            Field { "snapshot_id": Int64, metadata: {"PARQUET:field_id": "2"} },
            Field { "parent_id": nullable Int64, metadata: {"PARQUET:field_id": "3"} },
            Field { "operation": nullable Utf8, metadata: {"PARQUET:field_id": "4"} },
            Field { "manifest_list": nullable Utf8, metadata: {"PARQUET:field_id": "5"} },
            Field { "summary": nullable Map("key_value": non-null Struct("key": non-null Utf8, metadata: {"PARQUET:field_id": "7"}, "value": Utf8, metadata: {"PARQUET:field_id": "8"}), unsorted), metadata: {"PARQUET:field_id": "6"} }"#]],
        expect![[r#"
            committed_at: PrimitiveArray<Timestamp(µs, "+00:00")>
            [
            ],
            snapshot_id: PrimitiveArray<Int64>
            [
            ],
            parent_id: PrimitiveArray<Int64>
            [
            ],
            operation: StringArray
            [
            ],
            manifest_list: StringArray
            [
            ],
            summary: MapArray
            [
            ]"#]],
        &[],
        None,
    );

    let manifests = ctx
        .sql("select * from catalog.ns.t1$manifests")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    check_record_batches(
        manifests,
        expect![[r#"
            Field { "content": Int32, metadata: {"PARQUET:field_id": "14"} },
            Field { "path": Utf8, metadata: {"PARQUET:field_id": "1"} },
            Field { "length": Int64, metadata: {"PARQUET:field_id": "2"} },
            Field { "partition_spec_id": Int32, metadata: {"PARQUET:field_id": "3"} },
            Field { "added_snapshot_id": Int64, metadata: {"PARQUET:field_id": "4"} },
            Field { "added_data_files_count": Int32, metadata: {"PARQUET:field_id": "5"} },
            Field { "existing_data_files_count": Int32, metadata: {"PARQUET:field_id": "6"} },
            Field { "deleted_data_files_count": Int32, metadata: {"PARQUET:field_id": "7"} },
            Field { "added_delete_files_count": Int32, metadata: {"PARQUET:field_id": "15"} },
            Field { "existing_delete_files_count": Int32, metadata: {"PARQUET:field_id": "16"} },
            Field { "deleted_delete_files_count": Int32, metadata: {"PARQUET:field_id": "17"} },
            Field { "partition_summaries": List(non-null Struct("contains_null": non-null Boolean, metadata: {"PARQUET:field_id": "10"}, "contains_nan": Boolean, metadata: {"PARQUET:field_id": "11"}, "lower_bound": Utf8, metadata: {"PARQUET:field_id": "12"}, "upper_bound": Utf8, metadata: {"PARQUET:field_id": "13"}), metadata: {"PARQUET:field_id": "9"}), metadata: {"PARQUET:field_id": "8"} }"#]],
        expect![[r#"
            content: PrimitiveArray<Int32>
            [
            ],
            path: StringArray
            [
            ],
            length: PrimitiveArray<Int64>
            [
            ],
            partition_spec_id: PrimitiveArray<Int32>
            [
            ],
            added_snapshot_id: PrimitiveArray<Int64>
            [
            ],
            added_data_files_count: PrimitiveArray<Int32>
            [
            ],
            existing_data_files_count: PrimitiveArray<Int32>
            [
            ],
            deleted_data_files_count: PrimitiveArray<Int32>
            [
            ],
            added_delete_files_count: PrimitiveArray<Int32>
            [
            ],
            existing_delete_files_count: PrimitiveArray<Int32>
            [
            ],
            deleted_delete_files_count: PrimitiveArray<Int32>
            [
            ],
            partition_summaries: ListArray
            [
            ]"#]],
        &[],
        None,
    );

    Ok(())
}

#[tokio::test]
async fn test_insert_into() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_insert_into".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let creation = get_table_creation(temp_path(), "my_table", None)?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    // Verify table schema
    let provider = ctx.catalog("catalog").unwrap();
    let schema = provider.schema("test_insert_into").unwrap();
    let table = schema.table("my_table").await.unwrap().unwrap();
    let table_schema = table.schema();

    let expected = [("foo1", &DataType::Int32), ("foo2", &DataType::Utf8)];
    for (field, exp) in table_schema.fields().iter().zip(expected.iter()) {
        assert_eq!(field.name(), exp.0);
        assert_eq!(field.data_type(), exp.1);
        assert!(!field.is_nullable())
    }

    // Insert data into the table
    let df = ctx
        .sql("INSERT INTO catalog.test_insert_into.my_table VALUES (1, 'alan'), (2, 'turing')")
        .await
        .unwrap();

    // Verify the insert operation result
    let batches = df.collect().await.unwrap();
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert!(
        batch.num_rows() == 1 && batch.num_columns() == 1,
        "Results should only have one row and one column that has the number of rows inserted"
    );
    // Verify the number of rows inserted
    let rows_inserted = batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(rows_inserted.value(0), 2);

    // Query the table to verify the inserted data
    let df = ctx
        .sql("SELECT * FROM catalog.test_insert_into.my_table")
        .await
        .unwrap();

    let batches = df.collect().await.unwrap();

    // Use check_record_batches to verify the data
    check_record_batches(
        batches,
        expect![[r#"
            Field { "foo1": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "foo2": Utf8, metadata: {"PARQUET:field_id": "2"} }"#]],
        expect![[r#"
            foo1: PrimitiveArray<Int32>
            [
              1,
              2,
            ],
            foo2: StringArray
            [
              "alan",
              "turing",
            ]"#]],
        &[],
        Some("foo1"),
    );

    Ok(())
}

#[tokio::test]
async fn test_insert_overwrite() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_insert_overwrite".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let creation = get_table_creation(temp_path(), "my_table", None)?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    // Seed the table with two rows via INSERT INTO (append).
    ctx.sql("INSERT INTO catalog.test_insert_overwrite.my_table VALUES (1, 'alan'), (2, 'turing')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // INSERT OVERWRITE replaces ALL existing data with the new rows (DataFusion maps the
    // `overwrite` flag to `InsertOp::Overwrite`, which `IcebergCommitExec` commits via
    // `overwrite_files().overwrite_by_row_filter(AlwaysTrue)` — delete-all + add-new in one snapshot).
    let df = ctx
        .sql(
            "INSERT OVERWRITE catalog.test_insert_overwrite.my_table VALUES (9, 'replaced'), (10, 'fresh')",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let rows_written = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(
        rows_written.value(0),
        2,
        "INSERT OVERWRITE reports the 2 rows it wrote"
    );

    // SELECT * must return ONLY the overwrite rows — the original (1,alan),(2,turing) are GONE.
    // (An append would leave 4 rows; a correct overwrite leaves exactly the 2 new ones.)
    let df = ctx
        .sql("SELECT * FROM catalog.test_insert_overwrite.my_table")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let total_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
    assert_eq!(
        total_rows, 2,
        "INSERT OVERWRITE must REPLACE all data: exactly the 2 new rows remain, not 4 (append)"
    );
    check_record_batches(
        batches,
        expect![[r#"
            Field { "foo1": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "foo2": Utf8, metadata: {"PARQUET:field_id": "2"} }"#]],
        expect![[r#"
            foo1: PrimitiveArray<Int32>
            [
              9,
              10,
            ],
            foo2: StringArray
            [
              "replaced",
              "fresh",
            ]"#]],
        &[],
        Some("foo1"),
    );

    Ok(())
}

#[tokio::test]
async fn test_delete_from_merge_on_read() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_delete_merge_read".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    // Create the table in MERGE-ON-READ delete mode (the default is copy-on-write).
    let creation = get_merge_on_read_table_creation(temp_path(), "my_table")?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql(
        "INSERT INTO catalog.test_delete_merge_read.my_table VALUES (1, 'alan'), (2, 'turing'), (3, 'ALAN')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // DELETE WHERE foo1 > 0 AND lower(foo2) = 'alan'. The `lower(foo2)` branch is NOT convertible to an
    // Iceberg predicate, so a buggy delete relying on inexact pushdown would LOOSEN the filter to
    // `foo1 > 0` and OVER-DELETE all three rows. Our exact-filter delete removes only rows 1 and 3
    // (foo2 case-insensitively equal to "alan"), leaving (2, 'turing').
    let df = ctx
        .sql("DELETE FROM catalog.test_delete_merge_read.my_table WHERE foo1 > 0 AND lower(foo2) = 'alan'")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let deleted = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(
        deleted.value(0),
        2,
        "exactly the 2 rows matching the EXACT filter (rows 1 and 3) are deleted"
    );

    // Row (2, 'turing') MUST survive — the inexact-pushdown bug would have wrongly deleted it.
    let df = ctx
        .sql("SELECT * FROM catalog.test_delete_merge_read.my_table")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let total: usize = batches.iter().map(|batch| batch.num_rows()).sum();
    assert_eq!(
        total, 1,
        "row (2,'turing') must SURVIVE: the exact filter deletes only foo2~='alan' (rows 1,3), not all"
    );
    check_record_batches(
        batches,
        expect![[r#"
            Field { "foo1": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "foo2": Utf8, metadata: {"PARQUET:field_id": "2"} }"#]],
        expect![[r#"
            foo1: PrimitiveArray<Int32>
            [
              2,
            ],
            foo2: StringArray
            [
              "turing",
            ]"#]],
        &[],
        Some("foo1"),
    );

    Ok(())
}

#[tokio::test]
async fn test_delete_all_rows_no_where() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_delete_all".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = get_merge_on_read_table_creation(temp_path(), "my_table")?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql("INSERT INTO catalog.test_delete_all.my_table VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // `DELETE FROM t` with no WHERE (predicate = None) deletes every row.
    let df = ctx
        .sql("DELETE FROM catalog.test_delete_all.my_table")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let deleted = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(
        deleted.value(0),
        3,
        "DELETE FROM t (no WHERE) deletes every row"
    );

    let df = ctx
        .sql("SELECT * FROM catalog.test_delete_all.my_table")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let total: usize = batches.iter().map(|batch| batch.num_rows()).sum();
    assert_eq!(total, 0, "the table is empty after DELETE FROM t");

    Ok(())
}

#[tokio::test]
async fn test_delete_across_data_files() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_delete_multifile".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = get_merge_on_read_table_creation(temp_path(), "my_table")?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    // Two separate INSERT statements → two separate data files; each row's `_pos` is file-local (0,1 in each).
    ctx.sql("INSERT INTO catalog.test_delete_multifile.my_table VALUES (1, 'a'), (2, 'b')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    ctx.sql("INSERT INTO catalog.test_delete_multifile.my_table VALUES (3, 'c'), (4, 'd')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // DELETE one row from EACH file (foo1=2 lives in file 1 at pos 1; foo1=3 in file 2 at pos 0). The
    // position deletes must be PATH-keyed per file — a bug that confused per-file `_pos` would delete the
    // wrong rows. Survivors must be exactly {1, 4}.
    let df = ctx
        .sql("DELETE FROM catalog.test_delete_multifile.my_table WHERE foo1 = 2 OR foo1 = 3")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let deleted = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(
        deleted.value(0),
        2,
        "one row deleted from each of the two data files"
    );

    let df = ctx
        .sql("SELECT * FROM catalog.test_delete_multifile.my_table")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let total: usize = batches.iter().map(|batch| batch.num_rows()).sum();
    assert_eq!(total, 2, "exactly two rows survive across the two files");
    check_record_batches(
        batches,
        expect![[r#"
            Field { "foo1": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "foo2": Utf8, metadata: {"PARQUET:field_id": "2"} }"#]],
        expect![[r#"
            foo1: PrimitiveArray<Int32>
            [
              1,
              4,
            ],
            foo2: StringArray
            [
              "a",
              "d",
            ]"#]],
        &[],
        Some("foo1"),
    );

    Ok(())
}

#[tokio::test]
async fn test_delete_from_copy_on_write() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_delete_cow".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    // A default table (no write.delete.mode property) resolves to copy-on-write.
    let creation = get_table_creation(temp_path(), "my_table", None)?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql(
        "INSERT INTO catalog.test_delete_cow.my_table VALUES (1, 'alan'), (2, 'turing'), (3, 'ALAN')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // The SAME discriminating filter as the MoR test: `lower(foo2)` is unconvertible, so an inexact
    // pushdown would over-delete. Copy-on-write must rewrite the data file keeping ONLY (2,'turing').
    let df = ctx
        .sql("DELETE FROM catalog.test_delete_cow.my_table WHERE foo1 > 0 AND lower(foo2) = 'alan'")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let deleted = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(
        deleted.value(0),
        2,
        "copy-on-write deletes exactly rows 1 and 3 (the EXACT filter, not the loosened pushdown)"
    );

    let df = ctx
        .sql("SELECT * FROM catalog.test_delete_cow.my_table")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let total: usize = batches.iter().map(|batch| batch.num_rows()).sum();
    assert_eq!(
        total, 1,
        "copy-on-write rewrote the data file keeping only the surviving row (2,'turing')"
    );
    check_record_batches(
        batches,
        expect![[r#"
            Field { "foo1": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "foo2": Utf8, metadata: {"PARQUET:field_id": "2"} }"#]],
        expect![[r#"
            foo1: PrimitiveArray<Int32>
            [
              2,
            ],
            foo2: StringArray
            [
              "turing",
            ]"#]],
        &[],
        Some("foo1"),
    );

    // COW `DELETE FROM t` (no WHERE) on the last row → empty table (the survivors-empty / replace-all-
    // with-no-files path).
    let df = ctx
        .sql("DELETE FROM catalog.test_delete_cow.my_table")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let deleted = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(
        deleted.value(0),
        1,
        "COW DELETE FROM t deletes the last remaining row"
    );
    let df = ctx
        .sql("SELECT * FROM catalog.test_delete_cow.my_table")
        .await
        .unwrap();
    let total: usize = df
        .collect()
        .await
        .unwrap()
        .iter()
        .map(|batch| batch.num_rows())
        .sum();
    assert_eq!(
        total, 0,
        "the table is empty after copy-on-write DELETE FROM t"
    );

    Ok(())
}

#[tokio::test]
async fn test_update_merge_on_read() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_update_merge_read".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = get_merge_on_read_table_creation(temp_path(), "my_table")?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql(
        "INSERT INTO catalog.test_update_merge_read.my_table VALUES (1, 'alan'), (2, 'turing'), (3, 'ALAN')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // The discriminating filter again: only rows with foo2 case-insensitively 'alan' (rows 1, 3) are
    // updated; row 2 is untouched. `foo1 = foo1 + 100` proves the assignment expression reads the OLD
    // value. Merge-on-read writes the new rows + position-deletes the old in one RowDelta.
    let df = ctx
        .sql(
            "UPDATE catalog.test_update_merge_read.my_table SET foo2 = 'X', foo1 = foo1 + 100 \
             WHERE foo1 > 0 AND lower(foo2) = 'alan'",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let updated = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(updated.value(0), 2, "exactly rows 1 and 3 are updated");

    let df = ctx
        .sql("SELECT * FROM catalog.test_update_merge_read.my_table")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    // Row 2 (2,'turing') is unchanged; rows 1,3 become (101,'X'),(103,'X').
    check_record_batches(
        batches,
        expect![[r#"
            Field { "foo1": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "foo2": Utf8, metadata: {"PARQUET:field_id": "2"} }"#]],
        expect![[r#"
            foo1: PrimitiveArray<Int32>
            [
              2,
              101,
              103,
            ],
            foo2: StringArray
            [
              "turing",
              "X",
              "X",
            ]"#]],
        &[],
        Some("foo1"),
    );

    Ok(())
}

#[tokio::test]
async fn test_update_copy_on_write() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_update_cow".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    // Default table (no write.update.mode) → copy-on-write UPDATE.
    let creation = get_table_creation(temp_path(), "my_table", None)?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql(
        "INSERT INTO catalog.test_update_cow.my_table VALUES (1, 'alan'), (2, 'turing'), (3, 'ALAN')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // Same SET + discriminating WHERE; copy-on-write rewrites the data file (matching rows take the new
    // values via `zip`, non-matching keep the old).
    let df = ctx
        .sql(
            "UPDATE catalog.test_update_cow.my_table SET foo2 = 'X', foo1 = foo1 + 100 \
             WHERE foo1 > 0 AND lower(foo2) = 'alan'",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let updated = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(updated.value(0), 2, "exactly rows 1 and 3 are updated");

    let df = ctx
        .sql("SELECT * FROM catalog.test_update_cow.my_table")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    check_record_batches(
        batches,
        expect![[r#"
            Field { "foo1": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "foo2": Utf8, metadata: {"PARQUET:field_id": "2"} }"#]],
        expect![[r#"
            foo1: PrimitiveArray<Int32>
            [
              2,
              101,
              103,
            ],
            foo2: StringArray
            [
              "turing",
              "X",
              "X",
            ]"#]],
        &[],
        Some("foo1"),
    );

    Ok(())
}

// ============================================================================
// COW UPDATE — partitioned table tests (U2)
// ============================================================================

/// COW UPDATE on a partitioned table: update a non-partition column WHERE matches rows in one
/// partition; assert updated values, untouched rows in the other partition survive unchanged.
///
/// Table: `{id int, category string, value string}` partitioned by `identity(category)`.
/// Two partitions: `electronics` (ids 1,2) and `books` (ids 3,4).
/// UPDATE sets `value = 'UPDATED'` WHERE `category = 'electronics'`.
/// Post-UPDATE: electronics rows have the new value; books rows are unchanged.
#[tokio::test]
async fn test_update_cow_partitioned() -> Result<()> {
    let (ctx, _client) = make_partitioned_delete_ctx("test_upd_cow_part", "items").await?;

    ctx.sql(
        "INSERT INTO catalog.test_upd_cow_part.items VALUES \
         (1, 'electronics', 'laptop'), \
         (2, 'electronics', 'phone'), \
         (3, 'books', 'novel'), \
         (4, 'books', 'textbook')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // UPDATE only the electronics rows — value column gets new text.
    let batches = ctx
        .sql(
            "UPDATE catalog.test_upd_cow_part.items \
             SET value = 'UPDATED' WHERE category = 'electronics'",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let upd_count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(upd_count, 2, "exactly 2 electronics rows updated");

    // SELECT the full table; books rows must be unchanged, electronics rows have new value.
    let batches = ctx
        .sql("SELECT * FROM catalog.test_upd_cow_part.items ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 4, "all 4 rows survive (only values updated)");

    check_record_batches(
        batches,
        expect![[r#"
            Field { "id": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "category": Utf8, metadata: {"PARQUET:field_id": "2"} },
            Field { "value": Utf8, metadata: {"PARQUET:field_id": "3"} }"#]],
        expect![[r#"
            id: PrimitiveArray<Int32>
            [
              1,
              2,
              3,
              4,
            ],
            category: StringArray
            [
              "electronics",
              "electronics",
              "books",
              "books",
            ],
            value: StringArray
            [
              "UPDATED",
              "UPDATED",
              "novel",
              "textbook",
            ]"#]],
        &[],
        Some("id"),
    );

    Ok(())
}

/// COW UPDATE on a partitioned table where the UPDATE CHANGES the partition-key column.
///
/// Table: `{id int, category string, value string}` partitioned by `identity(category)`.
/// Row (id=1, category='electronics', value='laptop') is in the `electronics` partition.
/// `UPDATE … SET category = 'books' WHERE id = 1` changes its partition key.
/// Post-UPDATE: id=1 must appear with `category='books'` (in the books partition); all other
/// rows are unchanged.
#[tokio::test]
async fn test_update_cow_partitioned_moves_partition() -> Result<()> {
    let (ctx, _client) = make_partitioned_delete_ctx("test_upd_cow_move", "items").await?;

    ctx.sql(
        "INSERT INTO catalog.test_upd_cow_move.items VALUES \
         (1, 'electronics', 'laptop'), \
         (2, 'electronics', 'phone'), \
         (3, 'books', 'novel')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // UPDATE changes the partition-key column for id=1.
    let batches = ctx
        .sql(
            "UPDATE catalog.test_upd_cow_move.items \
             SET category = 'books' WHERE id = 1",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let upd_count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(upd_count, 1, "exactly 1 row updated");

    // id=1 must now appear with category='books'.
    let batches = ctx
        .sql("SELECT * FROM catalog.test_upd_cow_move.items ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 3, "all 3 rows survive after partition-move UPDATE");

    check_record_batches(
        batches,
        expect![[r#"
            Field { "id": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "category": Utf8, metadata: {"PARQUET:field_id": "2"} },
            Field { "value": Utf8, metadata: {"PARQUET:field_id": "3"} }"#]],
        expect![[r#"
            id: PrimitiveArray<Int32>
            [
              1,
              2,
              3,
            ],
            category: StringArray
            [
              "books",
              "electronics",
              "books",
            ],
            value: StringArray
            [
              "laptop",
              "phone",
              "novel",
            ]"#]],
        &[],
        Some("id"),
    );

    // Also verify via a partition-filtered query that id=1 is now found in books.
    let batches = ctx
        .sql(
            "SELECT id FROM catalog.test_upd_cow_move.items \
             WHERE category = 'books' ORDER BY id",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let ids: Vec<i32> = batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    assert!(ids.contains(&1), "id=1 is now in the books partition");
    assert!(!ids.contains(&2), "id=2 stays in electronics, not books");

    Ok(())
}

/// Confirm the existing unpartitioned COW UPDATE still passes under the new file-level path.
/// The discriminating `lower(foo2) = 'alan'` filter is unconvertible — an inexact pushdown
/// would over-update. The exact PhysicalExpr eval and the assignment expression (`foo1 + 100`)
/// must be preserved end-to-end.
#[tokio::test]
async fn test_update_cow_unpartitioned_exact_filter_preserved() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_upd_cow_exact".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    // Default table (no write.update.mode) → copy-on-write UPDATE.
    let creation = get_table_creation(temp_path(), "my_table", None)?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql(
        "INSERT INTO catalog.test_upd_cow_exact.my_table VALUES \
         (1, 'alan'), (2, 'turing'), (3, 'ALAN')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // The lower(foo2) = 'alan' filter matches rows 1 and 3, NOT 2.
    // Assignment foo1 + 100 tests expression eval.
    let batches = ctx
        .sql(
            "UPDATE catalog.test_upd_cow_exact.my_table \
             SET foo2 = 'X', foo1 = foo1 + 100 \
             WHERE foo1 > 0 AND lower(foo2) = 'alan'",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let upd_count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(upd_count, 2, "exactly rows 1 and 3 are updated");

    let batches = ctx
        .sql("SELECT * FROM catalog.test_upd_cow_exact.my_table")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    check_record_batches(
        batches,
        expect![[r#"
            Field { "foo1": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "foo2": Utf8, metadata: {"PARQUET:field_id": "2"} }"#]],
        expect![[r#"
            foo1: PrimitiveArray<Int32>
            [
              2,
              101,
              103,
            ],
            foo2: StringArray
            [
              "turing",
              "X",
              "X",
            ]"#]],
        &[],
        Some("foo1"),
    );

    Ok(())
}

#[tokio::test]
async fn test_update_no_where_updates_all_rows() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_update_all".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = get_merge_on_read_table_creation(temp_path(), "my_table")?;
    iceberg_catalog.create_table(&namespace, creation).await?;
    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql("INSERT INTO catalog.test_update_all.my_table VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // UPDATE with no WHERE (predicate = None) updates every row.
    let df = ctx
        .sql("UPDATE catalog.test_update_all.my_table SET foo1 = foo1 + 10")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let updated = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(updated.value(0), 3, "UPDATE with no WHERE updates all rows");

    let total: usize = ctx
        .sql("SELECT * FROM catalog.test_update_all.my_table WHERE foo1 IN (11, 12, 13)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
        .iter()
        .map(|batch| batch.num_rows())
        .sum();
    assert_eq!(total, 3, "every row's foo1 was incremented by 10");
    Ok(())
}

#[tokio::test]
async fn test_update_zero_match_is_noop() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_update_noop".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = get_merge_on_read_table_creation(temp_path(), "my_table")?;
    iceberg_catalog.create_table(&namespace, creation).await?;
    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql("INSERT INTO catalog.test_update_noop.my_table VALUES (1, 'a'), (2, 'b')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // UPDATE matching zero rows is a no-op (count 0, no commit, table unchanged).
    let df = ctx
        .sql("UPDATE catalog.test_update_noop.my_table SET foo2 = 'z' WHERE foo1 = 999")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let updated = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(
        updated.value(0),
        0,
        "UPDATE matching no rows reports 0 updated"
    );
    let total: usize = ctx
        .sql("SELECT * FROM catalog.test_update_noop.my_table")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
        .iter()
        .map(|batch| batch.num_rows())
        .sum();
    assert_eq!(total, 2, "the table is unchanged after a zero-match UPDATE");
    Ok(())
}

#[tokio::test]
async fn test_update_null_into_required_is_rejected() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_update_null".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = get_merge_on_read_table_creation(temp_path(), "my_table")?;
    iceberg_catalog.create_table(&namespace, creation).await?;
    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql("INSERT INTO catalog.test_update_null.my_table VALUES (1, 'a'), (2, 'b')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // `foo1` is a REQUIRED column; assigning NULL must be rejected (not silently written).
    let outcome = ctx
        .sql("UPDATE catalog.test_update_null.my_table SET foo1 = NULL WHERE foo2 = 'a'")
        .await;
    let errored = match outcome {
        Err(_) => true,
        Ok(df) => df.collect().await.is_err(),
    };
    assert!(
        errored,
        "UPDATE assigning NULL to the required column foo1 must error, not write a null"
    );
    Ok(())
}

fn get_nested_struct_type() -> StructType {
    // Create a nested struct type with:
    // - address: STRUCT<street: STRING, city: STRING, zip: INT>
    // - contact: STRUCT<email: STRING, phone: STRING>
    StructType::new(vec![
        NestedField::optional(
            10,
            "address",
            Type::Struct(StructType::new(vec![
                NestedField::required(11, "street", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(12, "city", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(13, "zip", Type::Primitive(PrimitiveType::Int)).into(),
            ])),
        )
        .into(),
        NestedField::optional(
            20,
            "contact",
            Type::Struct(StructType::new(vec![
                NestedField::optional(21, "email", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(22, "phone", Type::Primitive(PrimitiveType::String)).into(),
            ])),
        )
        .into(),
    ])
}

#[tokio::test]
async fn test_insert_into_nested() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_insert_nested".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let table_name = "nested_table";

    // Create a schema with nested fields
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "profile", Type::Struct(get_nested_struct_type())).into(),
        ])
        .build()?;

    // Create the table with the nested schema
    let creation = get_table_creation(temp_path(), table_name, Some(schema))?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    // Verify table schema
    let provider = ctx.catalog("catalog").unwrap();
    let schema = provider.schema("test_insert_nested").unwrap();
    let table = schema.table("nested_table").await.unwrap().unwrap();
    let table_schema = table.schema();

    // Verify the schema has the expected structure
    assert_eq!(table_schema.fields().len(), 3);
    assert_eq!(table_schema.field(0).name(), "id");
    assert_eq!(table_schema.field(1).name(), "name");
    assert_eq!(table_schema.field(2).name(), "profile");
    assert!(matches!(
        table_schema.field(2).data_type(),
        DataType::Struct(_)
    ));

    // In DataFusion, we need to use named_struct to create struct values
    // Insert data with nested structs
    let insert_sql = r#"
    INSERT INTO catalog.test_insert_nested.nested_table
    SELECT 
        1 as id, 
        'Alice' as name,
        named_struct(
            'address', named_struct(
                'street', '123 Main St',
                'city', 'San Francisco',
                'zip', 94105
            ),
            'contact', named_struct(
                'email', 'alice@example.com',
                'phone', '555-1234'
            )
        ) as profile
    UNION ALL
    SELECT 
        2 as id, 
        'Bob' as name,
        named_struct(
            'address', named_struct(
                'street', '456 Market St',
                'city', 'San Jose',
                'zip', 95113
            ),
            'contact', named_struct(
                'email', 'bob@example.com',
                'phone', NULL
            )
        ) as profile
    "#;

    // Execute the insert
    let df = ctx.sql(insert_sql).await.unwrap();
    let batches = df.collect().await.unwrap();

    // Verify the insert operation result
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert!(batch.num_rows() == 1 && batch.num_columns() == 1);

    // Verify the number of rows inserted
    let rows_inserted = batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(rows_inserted.value(0), 2);

    // Query the table to verify the inserted data
    let df = ctx
        .sql("SELECT * FROM catalog.test_insert_nested.nested_table ORDER BY id")
        .await
        .unwrap();

    let batches = df.collect().await.unwrap();

    // Use check_record_batches to verify the data
    check_record_batches(
        batches,
        expect![[r#"
            Field { "id": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "name": Utf8, metadata: {"PARQUET:field_id": "2"} },
            Field { "profile": nullable Struct("address": Struct("street": non-null Utf8, metadata: {"PARQUET:field_id": "6"}, "city": non-null Utf8, metadata: {"PARQUET:field_id": "7"}, "zip": non-null Int32, metadata: {"PARQUET:field_id": "8"}), metadata: {"PARQUET:field_id": "4"}, "contact": Struct("email": Utf8, metadata: {"PARQUET:field_id": "9"}, "phone": Utf8, metadata: {"PARQUET:field_id": "10"}), metadata: {"PARQUET:field_id": "5"}), metadata: {"PARQUET:field_id": "3"} }"#]],
        expect![[r#"
            id: PrimitiveArray<Int32>
            [
              1,
              2,
            ],
            name: StringArray
            [
              "Alice",
              "Bob",
            ],
            profile: StructArray
            -- validity:
            [
              valid,
              valid,
            ]
            [
            -- child 0: "address" (Struct([Field { name: "street", data_type: Utf8, metadata: {"PARQUET:field_id": "6"} }, Field { name: "city", data_type: Utf8, metadata: {"PARQUET:field_id": "7"} }, Field { name: "zip", data_type: Int32, metadata: {"PARQUET:field_id": "8"} }]))
            StructArray
            -- validity:
            [
              valid,
              valid,
            ]
            [
            -- child 0: "street" (Utf8)
            StringArray
            [
              "123 Main St",
              "456 Market St",
            ]
            -- child 1: "city" (Utf8)
            StringArray
            [
              "San Francisco",
              "San Jose",
            ]
            -- child 2: "zip" (Int32)
            PrimitiveArray<Int32>
            [
              94105,
              95113,
            ]
            ]
            -- child 1: "contact" (Struct([Field { name: "email", data_type: Utf8, nullable: true, metadata: {"PARQUET:field_id": "9"} }, Field { name: "phone", data_type: Utf8, nullable: true, metadata: {"PARQUET:field_id": "10"} }]))
            StructArray
            -- validity:
            [
              valid,
              valid,
            ]
            [
            -- child 0: "email" (Utf8)
            StringArray
            [
              "alice@example.com",
              "bob@example.com",
            ]
            -- child 1: "phone" (Utf8)
            StringArray
            [
              "555-1234",
              null,
            ]
            ]
            ]"#]],
        &[],
        Some("id"),
    );

    // Query with explicit field access to verify nested data
    let df = ctx
        .sql(
            r#"
            SELECT 
                id, 
                name,
                profile.address.street,
                profile.address.city,
                profile.address.zip,
                profile.contact.email,
                profile.contact.phone
            FROM catalog.test_insert_nested.nested_table 
            ORDER BY id
        "#,
        )
        .await
        .unwrap();

    let batches = df.collect().await.unwrap();

    // Use check_record_batches to verify the flattened data
    check_record_batches(
        batches,
        expect![[r#"
            Field { "id": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "name": Utf8, metadata: {"PARQUET:field_id": "2"} },
            Field { "catalog.test_insert_nested.nested_table.profile[address][street]": nullable Utf8, metadata: {"PARQUET:field_id": "6"} },
            Field { "catalog.test_insert_nested.nested_table.profile[address][city]": nullable Utf8, metadata: {"PARQUET:field_id": "7"} },
            Field { "catalog.test_insert_nested.nested_table.profile[address][zip]": nullable Int32, metadata: {"PARQUET:field_id": "8"} },
            Field { "catalog.test_insert_nested.nested_table.profile[contact][email]": nullable Utf8, metadata: {"PARQUET:field_id": "9"} },
            Field { "catalog.test_insert_nested.nested_table.profile[contact][phone]": nullable Utf8, metadata: {"PARQUET:field_id": "10"} }"#]],
        expect![[r#"
            id: PrimitiveArray<Int32>
            [
              1,
              2,
            ],
            name: StringArray
            [
              "Alice",
              "Bob",
            ],
            catalog.test_insert_nested.nested_table.profile[address][street]: StringArray
            [
              "123 Main St",
              "456 Market St",
            ],
            catalog.test_insert_nested.nested_table.profile[address][city]: StringArray
            [
              "San Francisco",
              "San Jose",
            ],
            catalog.test_insert_nested.nested_table.profile[address][zip]: PrimitiveArray<Int32>
            [
              94105,
              95113,
            ],
            catalog.test_insert_nested.nested_table.profile[contact][email]: StringArray
            [
              "alice@example.com",
              "bob@example.com",
            ],
            catalog.test_insert_nested.nested_table.profile[contact][phone]: StringArray
            [
              "555-1234",
              null,
            ]"#]],
        &[],
        Some("id"),
    );

    Ok(())
}

#[tokio::test]
async fn test_insert_into_partitioned() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_partitioned_write".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    // Create a schema with a partition column
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "category", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "value", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()?;

    // Create partition spec with identity transform on category
    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "category", Transform::Identity)?
        .build();

    // Create the partitioned table
    let creation = TableCreation::builder()
        .name("partitioned_table".to_string())
        .location(temp_path())
        .schema(schema)
        .partition_spec(partition_spec)
        .properties(HashMap::new())
        .build();

    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    // Insert data with multiple partition values in a single batch
    let df = ctx
        .sql(
            r#"
            INSERT INTO catalog.test_partitioned_write.partitioned_table 
            VALUES 
                (1, 'electronics', 'laptop'),
                (2, 'electronics', 'phone'),
                (3, 'books', 'novel'),
                (4, 'books', 'textbook'),
                (5, 'clothing', 'shirt')
            "#,
        )
        .await
        .unwrap();

    let batches = df.collect().await.unwrap();
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    let rows_inserted = batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(rows_inserted.value(0), 5);

    // Query the table to verify data
    let df = ctx
        .sql("SELECT * FROM catalog.test_partitioned_write.partitioned_table ORDER BY id")
        .await
        .unwrap();

    let batches = df.collect().await.unwrap();

    // Verify the data - note that _partition column should NOT be present
    check_record_batches(
        batches,
        expect![[r#"
            Field { "id": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "category": Utf8, metadata: {"PARQUET:field_id": "2"} },
            Field { "value": Utf8, metadata: {"PARQUET:field_id": "3"} }"#]],
        expect![[r#"
            id: PrimitiveArray<Int32>
            [
              1,
              2,
              3,
              4,
              5,
            ],
            category: StringArray
            [
              "electronics",
              "electronics",
              "books",
              "books",
              "clothing",
            ],
            value: StringArray
            [
              "laptop",
              "phone",
              "novel",
              "textbook",
              "shirt",
            ]"#]],
        &[],
        Some("id"),
    );

    // Verify that data files exist under correct partition paths
    let table_ident = TableIdent::new(namespace.clone(), "partitioned_table".to_string());
    let table = client.load_table(&table_ident).await?;
    let table_location = table.metadata().location();
    let file_io = table.file_io();

    // List files under each expected partition path
    let electronics_path = format!("{table_location}/data/category=electronics");
    let books_path = format!("{table_location}/data/category=books");
    let clothing_path = format!("{table_location}/data/category=clothing");

    // Verify partition directories exist and contain data files
    assert!(
        file_io.exists(&electronics_path).await?,
        "Expected partition directory: {electronics_path}"
    );
    assert!(
        file_io.exists(&books_path).await?,
        "Expected partition directory: {books_path}"
    );
    assert!(
        file_io.exists(&clothing_path).await?,
        "Expected partition directory: {clothing_path}"
    );

    Ok(())
}

/// Helper that builds a partitioned `{id int, category string, value string}` table partitioned by
/// identity(category), registers it in a fresh `SessionContext`, and returns the context + catalog
/// client so tests can issue SQL and inspect the catalog. The table namespace and name are supplied
/// by the caller so multiple tests can coexist without namespace collisions.
async fn make_partitioned_delete_ctx(
    ns: &str,
    tbl: &str,
) -> Result<(SessionContext, Arc<MemoryCatalog>)> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new(ns.to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "category", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "value", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()?;

    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "category", Transform::Identity)?
        .build();

    let creation = TableCreation::builder()
        .name(tbl.to_string())
        .location(temp_path())
        .schema(schema)
        .partition_spec(partition_spec)
        .properties(HashMap::new())
        .build();

    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    Ok((ctx, client))
}

/// COW DELETE on a partitioned table: delete rows in ONE partition, verify the other is untouched.
///
/// Table: `{id int, category string, value string}` partitioned by `identity(category)`.
/// Two partitions: `electronics` (ids 1,2) and `books` (ids 3,4).
/// DELETE removes all rows WHERE category = 'electronics'.
/// Post-DELETE: only the `books` rows remain.
#[tokio::test]
async fn test_delete_cow_partitioned() -> Result<()> {
    let (ctx, _client) = make_partitioned_delete_ctx("test_del_cow_part", "items").await?;

    ctx.sql(
        "INSERT INTO catalog.test_del_cow_part.items VALUES \
         (1, 'electronics', 'laptop'), \
         (2, 'electronics', 'phone'), \
         (3, 'books', 'novel'), \
         (4, 'books', 'textbook')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // DELETE all electronics rows — only the books partition must survive.
    let batches = ctx
        .sql("DELETE FROM catalog.test_del_cow_part.items WHERE category = 'electronics'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let deleted = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(deleted, 2, "exactly 2 electronics rows deleted");

    // SELECT must return ONLY the books rows.
    let batches = ctx
        .sql("SELECT * FROM catalog.test_del_cow_part.items ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 2, "two books rows survive");

    check_record_batches(
        batches,
        expect![[r#"
            Field { "id": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "category": Utf8, metadata: {"PARQUET:field_id": "2"} },
            Field { "value": Utf8, metadata: {"PARQUET:field_id": "3"} }"#]],
        expect![[r#"
            id: PrimitiveArray<Int32>
            [
              3,
              4,
            ],
            category: StringArray
            [
              "books",
              "books",
            ],
            value: StringArray
            [
              "novel",
              "textbook",
            ]"#]],
        &[],
        Some("id"),
    );

    Ok(())
}

/// COW DELETE FROM (no WHERE) on a partitioned table empties it.
#[tokio::test]
async fn test_delete_cow_partitioned_delete_from_all() -> Result<()> {
    let (ctx, _client) = make_partitioned_delete_ctx("test_del_cow_part_all", "items").await?;

    ctx.sql(
        "INSERT INTO catalog.test_del_cow_part_all.items VALUES \
         (1, 'electronics', 'laptop'), \
         (2, 'books', 'novel')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let batches = ctx
        .sql("DELETE FROM catalog.test_del_cow_part_all.items")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let deleted = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(deleted, 2, "both rows deleted");

    let total: usize = ctx
        .sql("SELECT * FROM catalog.test_del_cow_part_all.items")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
        .iter()
        .map(|b| b.num_rows())
        .sum();
    assert_eq!(total, 0, "table is empty after DELETE FROM");

    Ok(())
}

/// Confirm the existing unpartitioned COW test still passes under the new file-level path.
/// The discriminating `lower(foo2) = 'alan'` filter is unconvertible — an inexact pushdown
/// would over-delete (also remove row 2 'turing'). The exact PhysicalExpr eval must be preserved.
#[tokio::test]
async fn test_delete_cow_unpartitioned_exact_filter_preserved() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_del_cow_exact".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = get_table_creation(temp_path(), "my_table", None)?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql(
        "INSERT INTO catalog.test_del_cow_exact.my_table VALUES \
         (1, 'alan'), (2, 'turing'), (3, 'ALAN')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let batches = ctx
        .sql(
            "DELETE FROM catalog.test_del_cow_exact.my_table \
             WHERE foo1 > 0 AND lower(foo2) = 'alan'",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let deleted = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(
        deleted, 2,
        "exact filter deletes rows 1 and 3 only, not 2 ('turing')"
    );

    let batches = ctx
        .sql("SELECT * FROM catalog.test_del_cow_exact.my_table")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 1, "only (2,'turing') survives");

    check_record_batches(
        batches,
        expect![[r#"
            Field { "foo1": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "foo2": Utf8, metadata: {"PARQUET:field_id": "2"} }"#]],
        expect![[r#"
            foo1: PrimitiveArray<Int32>
            [
              2,
            ],
            foo2: StringArray
            [
              "turing",
            ]"#]],
        &[],
        Some("foo1"),
    );

    Ok(())
}

// ============================================================================
// ADDITIONAL EDGE-CASE PROBES — U1 COW DELETE adversarial verification
// ============================================================================

/// EDGE-CASE PROBE 1: Verify affected-path matching at the manifest level.
/// After COW DELETE, inspect the post-commit snapshot's manifest data files
/// directly (not just SELECT row counts). Confirm:
/// a) The deleted source file is gone from the live manifest set.
/// b) A new rewritten file was added (distinct path).
/// c) The unaffected file is still present with its ORIGINAL path.
/// This distinguishes silent no-op (old file stays + new file added = DUPLICATE rows)
/// from correct behavior.
#[tokio::test]
async fn test_delete_cow_path_matching_and_manifest_inspection() -> Result<()> {
    let (ctx, client) = make_partitioned_delete_ctx("dml_probe1", "items").await?;

    // Insert two batches into two separate transactions to get TWO distinct data files.
    ctx.sql("INSERT INTO catalog.dml_probe1.items VALUES (1, 'electronics', 'laptop')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    ctx.sql("INSERT INTO catalog.dml_probe1.items VALUES (2, 'books', 'novel')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // Load the table BEFORE delete to record existing file paths.
    let ns = NamespaceIdent::new("dml_probe1".to_string());
    let tbl_id = iceberg::TableIdent::new(ns.clone(), "items".to_string());
    let table_before = client.load_table(&tbl_id).await?;
    let snap_before = table_before.metadata().current_snapshot().unwrap();
    let ml_before = snap_before
        .load_manifest_list(table_before.file_io(), table_before.metadata())
        .await?;
    let mut paths_before: std::collections::HashSet<String> = std::collections::HashSet::new();
    for mf in ml_before.entries() {
        let m = mf.load_manifest(table_before.file_io()).await?;
        for entry in m.entries() {
            if entry.is_alive() {
                paths_before.insert(entry.file_path().to_string());
            }
        }
    }
    assert_eq!(
        paths_before.len(),
        2,
        "should have exactly 2 source files before DELETE"
    );

    // DELETE electronics row — only that file is affected.
    let batches = ctx
        .sql("DELETE FROM catalog.dml_probe1.items WHERE category = 'electronics'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let deleted = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(deleted, 1, "1 row deleted");

    // Load the table AFTER delete; inspect the live manifest set.
    let table_after = client.load_table(&tbl_id).await?;
    let snap_after = table_after.metadata().current_snapshot().unwrap();
    let ml_after = snap_after
        .load_manifest_list(table_after.file_io(), table_after.metadata())
        .await?;
    let mut paths_after: std::collections::HashSet<String> = std::collections::HashSet::new();
    for mf in ml_after.entries() {
        let m = mf.load_manifest(table_after.file_io()).await?;
        for entry in m.entries() {
            if entry.is_alive() {
                paths_after.insert(entry.file_path().to_string());
            }
        }
    }

    // CRITICAL assertions:
    // 1. Exactly ONE live file after (the books file, untouched, plus a rewritten... wait:
    //    the electronics file had 1 row deleted (all rows), so the rewrite produces NO survivors
    //    for that file. The books file was unaffected and stays. So exactly 1 live file.
    assert_eq!(
        paths_after.len(),
        1,
        "after full-file DELETE, live set must be exactly 1 (unaffected books file); got {paths_after:?}"
    );

    // 2. The surviving file MUST be the original books file (unchanged path).
    // We can't distinguish paths before delete by content alone, so we check that the surviving
    // path is one of the original paths (i.e., it is the unaffected original books file).
    let rows = ctx
        .sql("SELECT id, category FROM catalog.dml_probe1.items")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = rows.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 1, "exactly 1 row survives");

    // 3. No path in paths_after should be the SAME as any path_before AND also the electronics path.
    //    The unaffected books file must still be present with its original path.
    let surviving_path = paths_after.iter().next().unwrap().clone();
    assert!(
        paths_before.contains(&surviving_path),
        "the surviving file must be the original unaffected books file (same path); \
         surviving={surviving_path}; paths_before={paths_before:?}"
    );

    Ok(())
}

/// EDGE-CASE PROBE 2: Multi-file-per-partition — DELETE hits only FILE A; FILE B must be untouched.
/// Partition has 2 data files (two INSERT transactions into the same partition).
/// DELETE predicate matches rows in file A only.
/// After commit: file A is gone, file B has its ORIGINAL path (not rewritten).
#[tokio::test]
async fn test_delete_cow_multi_file_per_partition_only_affected_rewritten() -> Result<()> {
    let (ctx, client) = make_partitioned_delete_ctx("dml_probe2", "items").await?;

    // Two separate INSERT statements into the SAME partition → two distinct data files for 'electronics'.
    ctx.sql("INSERT INTO catalog.dml_probe2.items VALUES (1, 'electronics', 'laptop')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    ctx.sql("INSERT INTO catalog.dml_probe2.items VALUES (2, 'electronics', 'tablet')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ns = NamespaceIdent::new("dml_probe2".to_string());
    let tbl_id = iceberg::TableIdent::new(ns.clone(), "items".to_string());
    let table_before = client.load_table(&tbl_id).await?;

    // Collect paths before
    let snap_before = table_before.metadata().current_snapshot().unwrap();
    let ml_before = snap_before
        .load_manifest_list(table_before.file_io(), table_before.metadata())
        .await?;
    let mut paths_before: std::collections::HashSet<String> = std::collections::HashSet::new();
    for mf in ml_before.entries() {
        let m = mf.load_manifest(table_before.file_io()).await?;
        for entry in m.entries() {
            if entry.is_alive() {
                paths_before.insert(entry.file_path().to_string());
            }
        }
    }
    assert_eq!(paths_before.len(), 2, "two files before delete");

    // DELETE WHERE id = 1 — only file A (containing id=1) is affected.
    // File B (containing id=2) must be left untouched.
    ctx.sql("DELETE FROM catalog.dml_probe2.items WHERE id = 1")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // Inspect post-commit live paths.
    let table_after = client.load_table(&tbl_id).await?;
    let snap_after = table_after.metadata().current_snapshot().unwrap();
    let ml_after = snap_after
        .load_manifest_list(table_after.file_io(), table_after.metadata())
        .await?;
    let mut paths_after: std::collections::HashSet<String> = std::collections::HashSet::new();
    for mf in ml_after.entries() {
        let m = mf.load_manifest(table_after.file_io()).await?;
        for entry in m.entries() {
            if entry.is_alive() {
                paths_after.insert(entry.file_path().to_string());
            }
        }
    }

    // After deleting id=1 (which was the only row in file A, fully deleted):
    // - file A: fully deleted, no survivors → 0 rewritten files for A
    // - file B: unaffected → still has original path
    // Expected: exactly 1 live file, and it must be file B's original path.
    assert_eq!(
        paths_after.len(),
        1,
        "one file must survive (file B, untouched); got {paths_after:?}"
    );
    let survivor_path = paths_after.iter().next().unwrap();
    assert!(
        paths_before.contains(survivor_path),
        "file B must retain its ORIGINAL path (not rewritten); got {survivor_path}"
    );

    // Row content: only id=2 survives.
    let rows = ctx
        .sql("SELECT id FROM catalog.dml_probe2.items ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let ids: Vec<i32> = rows
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(ids, vec![2i32], "only id=2 survives");

    Ok(())
}

/// EDGE-CASE PROBE 3: Non-identity partition transform (truncate[4] on category string).
/// This tests PartitionValueCalculator with a real transform, not identity.
/// If the partition column calculation is wrong, the rewritten file would be assigned
/// to the wrong partition → DataFile.partition() would carry the wrong value.
#[tokio::test]
async fn test_delete_cow_non_identity_transform_truncate() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("dml_probe3".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    // Table: {id int, category string, value int} partitioned by truncate[4](category)
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "category", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "value", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()?;

    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "category_trunc", Transform::Truncate(4))?
        .build();

    let creation = TableCreation::builder()
        .name("trunc_table".to_string())
        .location(temp_path())
        .schema(schema)
        .partition_spec(partition_spec)
        .properties(HashMap::new())
        .build();

    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    // Insert: "electronics" truncates to "elec", "books" truncates to "book"
    ctx.sql(
        "INSERT INTO catalog.dml_probe3.trunc_table VALUES \
         (1, 'electronics', 100), \
         (2, 'electronics', 200), \
         (3, 'books', 300), \
         (4, 'books', 400)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // DELETE rows where id = 1 — file containing electronics rows affected.
    // After DELETE: rows 2,3,4 survive; file must be correctly placed in 'elec' partition.
    let batches = ctx
        .sql("DELETE FROM catalog.dml_probe3.trunc_table WHERE id = 1")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let deleted = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(deleted, 1);

    // Verify surviving rows include id=2 (rewritten in electronics/elec partition) and 3,4.
    let rows = ctx
        .sql("SELECT id FROM catalog.dml_probe3.trunc_table ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let ids: Vec<i32> = rows
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(
        ids,
        vec![2i32, 3, 4],
        "rows 2,3,4 survive after truncate-partitioned DELETE"
    );

    // Inspect post-commit DataFile.partition() for the rewritten file to confirm the transform
    // was applied correctly (category_trunc should be "elec" for the rewritten electronics file).
    let tbl_id = iceberg::TableIdent::new(namespace, "trunc_table".to_string());
    let table = client.load_table(&tbl_id).await?;
    let snap = table.metadata().current_snapshot().unwrap();
    let ml = snap
        .load_manifest_list(table.file_io(), table.metadata())
        .await?;
    let mut partition_values: Vec<String> = Vec::new();
    for mf in ml.entries() {
        let m = mf.load_manifest(table.file_io()).await?;
        for entry in m.entries() {
            if entry.is_alive() {
                let pv = entry.data_file().partition();
                // partition struct has one field: category_trunc (String, truncated to 4 chars)
                let field = pv.fields()[0].as_ref();
                if let Some(lit) = field
                    && let iceberg::spec::Literal::Primitive(
                        iceberg::spec::PrimitiveLiteral::String(s),
                    ) = lit
                {
                    partition_values.push(s.clone());
                }
            }
        }
    }
    partition_values.sort();
    // Two files survive: 1 file for "book" partition (ids 3,4), 1 file for "elec" partition (id=2
    // rewritten). After DELETE, the affected file is rewritten (dropping id=1), producing 1 new
    // "elec"-keyed file. The "book" file is unaffected (or, if both partitions were in a single
    // source file, both get correctly routed on rewrite).
    assert_eq!(
        partition_values,
        vec!["book".to_string(), "elec".to_string()],
        "rewritten file must be in 'elec' partition; all partition values: {partition_values:?}"
    );

    Ok(())
}

/// EDGE-CASE PROBE 4: Verify a DELETE WHERE predicate that hits NO rows is a no-op (no new snapshot).
#[tokio::test]
async fn test_delete_cow_no_match_is_noop() -> Result<()> {
    let (ctx, client) = make_partitioned_delete_ctx("dml_probe4", "items").await?;

    ctx.sql("INSERT INTO catalog.dml_probe4.items VALUES (1, 'electronics', 'laptop')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ns = NamespaceIdent::new("dml_probe4".to_string());
    let tbl_id = iceberg::TableIdent::new(ns, "items".to_string());
    let table_before = client.load_table(&tbl_id).await?;
    let snap_id_before = table_before
        .metadata()
        .current_snapshot()
        .map(|s| s.snapshot_id());

    // DELETE where nothing matches.
    let batches = ctx
        .sql("DELETE FROM catalog.dml_probe4.items WHERE category = 'books'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let deleted = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(deleted, 0, "no rows deleted");

    let table_after = client.load_table(&tbl_id).await?;
    let snap_id_after = table_after
        .metadata()
        .current_snapshot()
        .map(|s| s.snapshot_id());
    assert_eq!(
        snap_id_before, snap_id_after,
        "no-op DELETE must not create a new snapshot"
    );

    Ok(())
}

// ============================================================================
// ADDITIONAL EDGE-CASE PROBES — U2 COW UPDATE adversarial verification
// ============================================================================

/// COW UPDATE PROBE 1: Row-conservation + manifest-level inspection for COW UPDATE.
///
/// Two files: file_A (id=1 matches, id=2 does not — in the same file because they're inserted
/// together) and file_B (id=3, unaffected partition).
/// After UPDATE SET value='NEW' WHERE id=1:
///   - File_A must be DELETED from the live manifest and replaced by a NEW file with 2 rows
///     (row1 updated, row2 carried unchanged).
///   - File_B must remain with its ORIGINAL path (not rewritten).
///   - Total row count = 3 (row conservation).
///   - Row content: id=1 → 'NEW', id=2 → 'phone', id=3 → 'novel'.
///
/// This probe specifically catches:
///   a) Phantom deletion (row count drops after UPDATE)
///   b) Duplication (row count rises after UPDATE)
///   c) Incorrect path matching (old file stays AND new file added = double rows)
///   d) Over-rewrite (unaffected file_B rewritten when it should not be)
#[tokio::test]
async fn test_update_cow_row_conservation_and_manifest_inspection() -> Result<()> {
    let (ctx, client) = make_partitioned_delete_ctx("upd_probe_u2_1", "items").await?;

    // Insert two separate transactions so file_A (electronics) and file_B (books) are distinct.
    // file_A: rows id=1 AND id=2 in the electronics partition (single INSERT = one file).
    ctx.sql(
        "INSERT INTO catalog.upd_probe_u2_1.items VALUES \
         (1, 'electronics', 'laptop'), (2, 'electronics', 'phone')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // file_B: row id=3 in the books partition (separate INSERT = separate file).
    ctx.sql("INSERT INTO catalog.upd_probe_u2_1.items VALUES (3, 'books', 'novel')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // Record pre-UPDATE manifest state.
    let ns = NamespaceIdent::new("upd_probe_u2_1".to_string());
    let tbl_id = iceberg::TableIdent::new(ns.clone(), "items".to_string());
    let table_before = client.load_table(&tbl_id).await?;
    let snap_before = table_before.metadata().current_snapshot().unwrap();
    let ml_before = snap_before
        .load_manifest_list(table_before.file_io(), table_before.metadata())
        .await?;
    let mut paths_before: std::collections::HashSet<String> = std::collections::HashSet::new();
    for mf in ml_before.entries() {
        let m = mf.load_manifest(table_before.file_io()).await?;
        for entry in m.entries() {
            if entry.is_alive() {
                paths_before.insert(entry.file_path().to_string());
            }
        }
    }
    assert_eq!(paths_before.len(), 2, "2 source files before UPDATE");

    // UPDATE: only id=1 matches WHERE; id=2 in the same file must be carried unchanged.
    let batches = ctx
        .sql(
            "UPDATE catalog.upd_probe_u2_1.items \
             SET value = 'NEW' WHERE id = 1",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let upd_count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(upd_count, 1, "exactly 1 row updated (only id=1 matched)");

    // Inspect manifest after UPDATE.
    let table_after = client.load_table(&tbl_id).await?;
    let snap_after = table_after.metadata().current_snapshot().unwrap();
    let ml_after = snap_after
        .load_manifest_list(table_after.file_io(), table_after.metadata())
        .await?;
    let mut live_files_after: Vec<(String, u64)> = Vec::new(); // (path, record_count)
    for mf in ml_after.entries() {
        let m = mf.load_manifest(table_after.file_io()).await?;
        for entry in m.entries() {
            if entry.is_alive() {
                let rc = entry.data_file().record_count();
                live_files_after.push((entry.file_path().to_string(), rc));
            }
        }
    }

    // After UPDATE: file_A is replaced (2 rows: 1 updated + 1 unchanged), file_B unchanged.
    // So exactly 2 live files, with total 3 rows.
    assert_eq!(
        live_files_after.len(),
        2,
        "exactly 2 live files after UPDATE (1 rewritten + 1 unaffected); got {live_files_after:?}"
    );
    let total_manifest_rows: u64 = live_files_after.iter().map(|(_, rc)| rc).sum();
    assert_eq!(
        total_manifest_rows, 3,
        "manifest record counts must sum to 3 (row conservation); got {total_manifest_rows}"
    );

    // File_B (books, unaffected) must still carry its ORIGINAL path.
    let paths_after: std::collections::HashSet<String> =
        live_files_after.iter().map(|(p, _)| p.clone()).collect();
    let original_surviving: Vec<&String> = paths_before
        .iter()
        .filter(|p| paths_after.contains(*p))
        .collect();
    assert_eq!(
        original_surviving.len(),
        1,
        "exactly one original file (file_B books) must survive unchanged; \
         paths_before={paths_before:?} paths_after={paths_after:?}"
    );

    // The NEW rewritten file (electronics) must have a DIFFERENT path than any pre-UPDATE file.
    let new_paths: Vec<&String> = paths_after
        .iter()
        .filter(|p| !paths_before.contains(*p))
        .collect();
    assert_eq!(
        new_paths.len(),
        1,
        "exactly one NEW file (rewritten electronics) must appear; \
         new_paths={new_paths:?}"
    );

    // Row content: exact row values post-UPDATE.
    let batches = ctx
        .sql("SELECT * FROM catalog.upd_probe_u2_1.items ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 3,
        "SELECT must return exactly 3 rows (no drop, no dup)"
    );

    check_record_batches(
        batches,
        expect![[r#"
            Field { "id": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "category": Utf8, metadata: {"PARQUET:field_id": "2"} },
            Field { "value": Utf8, metadata: {"PARQUET:field_id": "3"} }"#]],
        expect![[r#"
            id: PrimitiveArray<Int32>
            [
              1,
              2,
              3,
            ],
            category: StringArray
            [
              "electronics",
              "electronics",
              "books",
            ],
            value: StringArray
            [
              "NEW",
              "phone",
              "novel",
            ]"#]],
        &[],
        Some("id"),
    );

    Ok(())
}

/// COW UPDATE PROBE 2: Multi-file-per-partition UPDATE — only the affected file is rewritten;
/// the second file in the same partition is untouched (verified at manifest path level).
///
/// Partition 'electronics' has 2 files: file_A (id=1) and file_B (id=2).
/// UPDATE SET value='NEW' WHERE id=1.
/// After UPDATE:
///   - file_A is replaced by a new file (1 row, updated).
///   - file_B retains its ORIGINAL path and content (not rewritten).
///   - Total rows = 2.
#[tokio::test]
async fn test_update_cow_multi_file_per_partition_only_affected_rewritten() -> Result<()> {
    let (ctx, client) = make_partitioned_delete_ctx("upd_probe_u2_2", "items").await?;

    // Two separate INSERT statements → two distinct files in the electronics partition.
    ctx.sql("INSERT INTO catalog.upd_probe_u2_2.items VALUES (1, 'electronics', 'laptop')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    ctx.sql("INSERT INTO catalog.upd_probe_u2_2.items VALUES (2, 'electronics', 'tablet')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ns = NamespaceIdent::new("upd_probe_u2_2".to_string());
    let tbl_id = iceberg::TableIdent::new(ns.clone(), "items".to_string());
    let table_before = client.load_table(&tbl_id).await?;
    let snap_before = table_before.metadata().current_snapshot().unwrap();
    let ml_before = snap_before
        .load_manifest_list(table_before.file_io(), table_before.metadata())
        .await?;
    let mut paths_before: std::collections::HashSet<String> = std::collections::HashSet::new();
    for mf in ml_before.entries() {
        let m = mf.load_manifest(table_before.file_io()).await?;
        for entry in m.entries() {
            if entry.is_alive() {
                paths_before.insert(entry.file_path().to_string());
            }
        }
    }
    assert_eq!(paths_before.len(), 2, "2 files before UPDATE");

    // UPDATE WHERE id=1 — only file_A (containing id=1) is affected.
    let batches = ctx
        .sql("UPDATE catalog.upd_probe_u2_2.items SET value = 'NEW' WHERE id = 1")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let upd_count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(upd_count, 1, "exactly 1 row updated");

    // Inspect post-commit live paths.
    let table_after = client.load_table(&tbl_id).await?;
    let snap_after = table_after.metadata().current_snapshot().unwrap();
    let ml_after = snap_after
        .load_manifest_list(table_after.file_io(), table_after.metadata())
        .await?;
    let mut paths_after: std::collections::HashSet<String> = std::collections::HashSet::new();
    for mf in ml_after.entries() {
        let m = mf.load_manifest(table_after.file_io()).await?;
        for entry in m.entries() {
            if entry.is_alive() {
                paths_after.insert(entry.file_path().to_string());
            }
        }
    }

    // Exactly 2 live files: 1 new (rewritten file_A) + 1 original (file_B unchanged).
    assert_eq!(
        paths_after.len(),
        2,
        "still 2 files after UPDATE; got {paths_after:?}"
    );

    // Exactly one original file must survive (file_B).
    let surviving_original: Vec<&String> = paths_before
        .iter()
        .filter(|p| paths_after.contains(*p))
        .collect();
    assert_eq!(
        surviving_original.len(),
        1,
        "exactly one ORIGINAL file (file_B) must survive; \
         paths_before={paths_before:?} paths_after={paths_after:?}"
    );

    // Exactly one new file must have been added (rewritten file_A).
    let new_files: Vec<&String> = paths_after
        .iter()
        .filter(|p| !paths_before.contains(*p))
        .collect();
    assert_eq!(
        new_files.len(),
        1,
        "exactly one NEW file (rewritten file_A) must appear; new_files={new_files:?}"
    );

    // Row content correct.
    let rows = ctx
        .sql("SELECT id, value FROM catalog.upd_probe_u2_2.items ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = rows.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 2, "both rows must survive");

    let ids: Vec<i32> = rows
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    let values: Vec<&str> = rows
        .iter()
        .flat_map(|b| {
            let col = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            (0..col.len()).map(|i| col.value(i)).collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(ids, vec![1i32, 2], "ids must be 1,2");
    assert_eq!(
        values,
        vec!["NEW", "tablet"],
        "id=1 updated, id=2 unchanged"
    );

    Ok(())
}

/// COW UPDATE PROBE 3: COW UPDATE with no WHERE on a PARTITIONED table.
///
/// `predicate = None` path in a partitioned table — all files are affected and all rows updated.
/// Table: 2 partitions, 2 rows each. UPDATE SET value='ALL' (no WHERE).
/// Post-UPDATE:
///   - All 4 rows must have value='ALL'.
///   - Total row count = 4 (no loss, no dup).
///   - Updated count returned = 4.
///   - New snapshot must have been created.
#[tokio::test]
async fn test_update_cow_partitioned_no_where_updates_all() -> Result<()> {
    let (ctx, client) = make_partitioned_delete_ctx("upd_probe_u2_3", "items").await?;

    ctx.sql(
        "INSERT INTO catalog.upd_probe_u2_3.items VALUES \
         (1, 'electronics', 'laptop'), \
         (2, 'electronics', 'phone'), \
         (3, 'books', 'novel'), \
         (4, 'books', 'textbook')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let ns = NamespaceIdent::new("upd_probe_u2_3".to_string());
    let tbl_id = iceberg::TableIdent::new(ns.clone(), "items".to_string());
    let table_before = client.load_table(&tbl_id).await?;
    let snap_id_before = table_before
        .metadata()
        .current_snapshot()
        .map(|s| s.snapshot_id());

    // UPDATE with no WHERE — should update ALL rows in ALL partitions.
    let batches = ctx
        .sql("UPDATE catalog.upd_probe_u2_3.items SET value = 'ALL'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let upd_count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(upd_count, 4, "all 4 rows updated (no WHERE = all match)");

    // A new snapshot must have been created.
    let table_after = client.load_table(&tbl_id).await?;
    let snap_id_after = table_after
        .metadata()
        .current_snapshot()
        .map(|s| s.snapshot_id());
    assert_ne!(
        snap_id_before, snap_id_after,
        "UPDATE must create a new snapshot"
    );

    // All 4 rows must have value='ALL'.
    let batches = ctx
        .sql("SELECT * FROM catalog.upd_probe_u2_3.items ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 4, "all 4 rows must survive (no row loss)");

    check_record_batches(
        batches,
        expect![[r#"
            Field { "id": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "category": Utf8, metadata: {"PARQUET:field_id": "2"} },
            Field { "value": Utf8, metadata: {"PARQUET:field_id": "3"} }"#]],
        expect![[r#"
            id: PrimitiveArray<Int32>
            [
              1,
              2,
              3,
              4,
            ],
            category: StringArray
            [
              "electronics",
              "electronics",
              "books",
              "books",
            ],
            value: StringArray
            [
              "ALL",
              "ALL",
              "ALL",
              "ALL",
            ]"#]],
        &[],
        Some("id"),
    );

    Ok(())
}

/// COW UPDATE PROBE 4: COW UPDATE zero-match is a no-op (no snapshot created).
///
/// UPDATE WHERE predicate matches zero rows → updated=0, no snapshot.
/// This verifies the no-op early-exit path.
#[tokio::test]
async fn test_update_cow_partitioned_no_match_is_noop() -> Result<()> {
    let (ctx, client) = make_partitioned_delete_ctx("upd_probe_u2_4", "items").await?;

    ctx.sql("INSERT INTO catalog.upd_probe_u2_4.items VALUES (1, 'electronics', 'laptop')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ns = NamespaceIdent::new("upd_probe_u2_4".to_string());
    let tbl_id = iceberg::TableIdent::new(ns.clone(), "items".to_string());
    let table_before = client.load_table(&tbl_id).await?;
    let snap_id_before = table_before
        .metadata()
        .current_snapshot()
        .map(|s| s.snapshot_id());

    let batches = ctx
        .sql("UPDATE catalog.upd_probe_u2_4.items SET value = 'X' WHERE category = 'books'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let upd_count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(upd_count, 0, "no rows match, updated=0");

    // No new snapshot must be created.
    let table_after = client.load_table(&tbl_id).await?;
    let snap_id_after = table_after
        .metadata()
        .current_snapshot()
        .map(|s| s.snapshot_id());
    assert_eq!(
        snap_id_before, snap_id_after,
        "no-op UPDATE must not create a new snapshot"
    );

    Ok(())
}

/// COW UPDATE PROBE 5: Partition-move verified at manifest/DataFile level, not just SELECT.
///
/// File in partition 'electronics' with rows r1 (id=1, moves to 'books') and r2 (id=2, stays in
/// 'electronics'). After UPDATE SET category='books' WHERE id=1:
///   - The old electronics file is DELETED from the manifest.
///   - Two new files appear: one in 'books' partition (r1), one in 'electronics' partition (r2).
///   - Each new file's DataFile.partition() must carry the correct partition struct.
///   - Total row count = 2 (unchanged).
///
/// This probe attacks the partition-move correctness at a deeper level than the existing SELECT
/// test, verifying that the file-level partition metadata is correct, not just query results.
#[tokio::test]
async fn test_update_cow_partition_move_manifest_level_verification() -> Result<()> {
    let (ctx, client) = make_partitioned_delete_ctx("cow_update_move_probe", "items").await?;

    // Single file in 'electronics' with 2 rows.
    ctx.sql(
        "INSERT INTO catalog.cow_update_move_probe.items VALUES \
         (1, 'electronics', 'laptop'), (2, 'electronics', 'phone')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let ns = NamespaceIdent::new("cow_update_move_probe".to_string());
    let tbl_id = iceberg::TableIdent::new(ns.clone(), "items".to_string());

    // UPDATE: move id=1 to 'books' partition; id=2 stays in 'electronics'.
    let batches = ctx
        .sql(
            "UPDATE catalog.cow_update_move_probe.items \
             SET category = 'books' WHERE id = 1",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let upd_count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(upd_count, 1, "1 row updated (id=1 moved to books)");

    // Inspect post-UPDATE manifest: collect (path, partition_value, record_count).
    let table_after = client.load_table(&tbl_id).await?;
    let snap_after = table_after.metadata().current_snapshot().unwrap();
    let ml_after = snap_after
        .load_manifest_list(table_after.file_io(), table_after.metadata())
        .await?;

    let mut partition_vals: Vec<String> = Vec::new();
    let mut total_records: u64 = 0;
    for mf in ml_after.entries() {
        let m = mf.load_manifest(table_after.file_io()).await?;
        for entry in m.entries() {
            if entry.is_alive() {
                let pv = entry.data_file().partition();
                // The partition spec has one field: identity(category) → the value is the category string.
                let field_val = pv.fields()[0].as_ref();
                if let Some(iceberg::spec::Literal::Primitive(
                    iceberg::spec::PrimitiveLiteral::String(s),
                )) = field_val
                {
                    partition_vals.push(s.clone());
                }
                total_records += entry.data_file().record_count();
            }
        }
    }

    // Row conservation at manifest level.
    assert_eq!(
        total_records, 2,
        "manifest record counts must sum to 2 (row conservation at DataFile level); \
         partition_vals={partition_vals:?}"
    );

    // Both partition values must appear: 'books' (id=1 moved) and 'electronics' (id=2 stayed).
    partition_vals.sort();
    assert_eq!(
        partition_vals,
        vec!["books".to_string(), "electronics".to_string()],
        "exactly one DataFile in 'books' partition and one in 'electronics' partition; \
         partition_vals={partition_vals:?}"
    );

    // Verify via SELECT that row content is also correct.
    let batches = ctx
        .sql("SELECT * FROM catalog.cow_update_move_probe.items ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 2, "SELECT must return 2 rows");

    check_record_batches(
        batches,
        expect![[r#"
            Field { "id": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "category": Utf8, metadata: {"PARQUET:field_id": "2"} },
            Field { "value": Utf8, metadata: {"PARQUET:field_id": "3"} }"#]],
        expect![[r#"
            id: PrimitiveArray<Int32>
            [
              1,
              2,
            ],
            category: StringArray
            [
              "books",
              "electronics",
            ],
            value: StringArray
            [
              "laptop",
              "phone",
            ]"#]],
        &[],
        Some("id"),
    );

    Ok(())
}

// ============================================================================
// MoR UPDATE — partitioned table tests (U3)
// ============================================================================

/// Helper that builds a partitioned V2 `{id int, category string, value string}` table with
/// `write.delete.mode = merge-on-read` and `write.update.mode = merge-on-read`, partitioned by
/// `identity(category)`. The format version defaults to V2 (required for MoR position deletes).
async fn make_partitioned_mread_ctx(
    ns: &str,
    tbl: &str,
) -> Result<(SessionContext, Arc<MemoryCatalog>)> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new(ns.to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "category", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "value", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()?;

    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "category", Transform::Identity)?
        .build();

    let creation = TableCreation::builder()
        .name(tbl.to_string())
        .location(temp_path())
        .schema(schema)
        .partition_spec(partition_spec)
        .properties(HashMap::from([
            ("write.delete.mode".to_string(), "merge-on-read".to_string()),
            ("write.update.mode".to_string(), "merge-on-read".to_string()),
        ]))
        .build();

    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    Ok((ctx, client))
}

/// Prerequisite: confirm that partitioned MoR DELETE already works (no guard exists for DELETE).
///
/// Table: `{id int, category string, value string}` partitioned by `identity(category)`,
/// `write.delete.mode = merge-on-read`, V2.
/// Two partitions: `electronics` (ids 1,2) and `books` (ids 3,4).
/// DELETE rows WHERE category = 'electronics'.
/// Post-DELETE: only the books rows survive (confirms the position-delete RowDelta path works
/// partitioned — the prerequisite for the MoR UPDATE path).
#[tokio::test]
async fn test_delete_mread_partitioned() -> Result<()> {
    let (ctx, _client) = make_partitioned_mread_ctx("test_del_mread_part", "items").await?;

    ctx.sql(
        "INSERT INTO catalog.test_del_mread_part.items VALUES \
         (1, 'electronics', 'laptop'), \
         (2, 'electronics', 'phone'), \
         (3, 'books', 'novel'), \
         (4, 'books', 'textbook')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // MoR DELETE: write position-delete file covering both electronics rows; commit RowDelta.
    let batches = ctx
        .sql("DELETE FROM catalog.test_del_mread_part.items WHERE category = 'electronics'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let deleted = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(
        deleted, 2,
        "MoR DELETE must remove exactly the 2 electronics rows"
    );

    // The books rows must survive untouched.
    let batches = ctx
        .sql("SELECT * FROM catalog.test_del_mread_part.items ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total, 2,
        "exactly 2 books rows survive after MoR DELETE on partitioned table"
    );

    check_record_batches(
        batches,
        expect![[r#"
            Field { "id": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "category": Utf8, metadata: {"PARQUET:field_id": "2"} },
            Field { "value": Utf8, metadata: {"PARQUET:field_id": "3"} }"#]],
        expect![[r#"
            id: PrimitiveArray<Int32>
            [
              3,
              4,
            ],
            category: StringArray
            [
              "books",
              "books",
            ],
            value: StringArray
            [
              "novel",
              "textbook",
            ]"#]],
        &[],
        Some("id"),
    );

    Ok(())
}

/// MoR UPDATE on a partitioned table: update a non-partition column WHERE matches rows in one
/// partition; assert updated values, untouched rows in the other partition survive unchanged.
///
/// Table: `{id int, category string, value string}` partitioned by `identity(category)`,
/// `write.update.mode = merge-on-read`, V2.
/// Two partitions: `electronics` (ids 1,2) and `books` (ids 3,4).
/// UPDATE sets `value = 'UPDATED'` WHERE `category = 'electronics'`.
/// MoR: position-deletes for old rows + new data file with updated rows, in one RowDelta.
/// Post-UPDATE: electronics rows have the new value; books rows are unchanged.
#[tokio::test]
async fn test_update_mread_partitioned() -> Result<()> {
    let (ctx, _client) = make_partitioned_mread_ctx("test_upd_mread_part", "items").await?;

    ctx.sql(
        "INSERT INTO catalog.test_upd_mread_part.items VALUES \
         (1, 'electronics', 'laptop'), \
         (2, 'electronics', 'phone'), \
         (3, 'books', 'novel'), \
         (4, 'books', 'textbook')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // MoR UPDATE: update only the electronics rows' value column.
    let batches = ctx
        .sql(
            "UPDATE catalog.test_upd_mread_part.items \
             SET value = 'UPDATED' WHERE category = 'electronics'",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let upd_count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(upd_count, 2, "exactly 2 electronics rows updated via MoR");

    // All 4 rows survive; electronics have new value; books unchanged.
    let batches = ctx
        .sql("SELECT * FROM catalog.test_upd_mread_part.items ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total, 4,
        "all 4 rows survive after MoR UPDATE (only values changed)"
    );

    check_record_batches(
        batches,
        expect![[r#"
            Field { "id": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "category": Utf8, metadata: {"PARQUET:field_id": "2"} },
            Field { "value": Utf8, metadata: {"PARQUET:field_id": "3"} }"#]],
        expect![[r#"
            id: PrimitiveArray<Int32>
            [
              1,
              2,
              3,
              4,
            ],
            category: StringArray
            [
              "electronics",
              "electronics",
              "books",
              "books",
            ],
            value: StringArray
            [
              "UPDATED",
              "UPDATED",
              "novel",
              "textbook",
            ]"#]],
        &[],
        Some("id"),
    );

    Ok(())
}

/// MoR UPDATE that changes the partition-key column: the old row is position-deleted (in the
/// original partition's data file) and the new row is inserted into the NEW partition's data file,
/// all in one RowDelta.
///
/// Table: `{id int, category string, value string}` partitioned by `identity(category)`,
/// `write.update.mode = merge-on-read`, V2.
/// `UPDATE … SET category = 'books' WHERE id = 1` moves id=1 from `electronics` to `books`.
/// Post-UPDATE: id=1 appears with category='books'; id=2 stays in electronics unchanged.
#[tokio::test]
async fn test_update_mread_partitioned_moves_partition() -> Result<()> {
    let (ctx, _client) = make_partitioned_mread_ctx("test_upd_mread_move", "items").await?;

    ctx.sql(
        "INSERT INTO catalog.test_upd_mread_move.items VALUES \
         (1, 'electronics', 'laptop'), \
         (2, 'electronics', 'phone'), \
         (3, 'books', 'novel')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // MoR UPDATE changes the partition-key column for id=1.
    let batches = ctx
        .sql(
            "UPDATE catalog.test_upd_mread_move.items \
             SET category = 'books' WHERE id = 1",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let upd_count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(upd_count, 1, "exactly 1 row updated (partition-move)");

    // All 3 rows survive; id=1 now has category='books'.
    let batches = ctx
        .sql("SELECT * FROM catalog.test_upd_mread_move.items ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total, 3,
        "all 3 rows survive after MoR partition-move UPDATE"
    );

    check_record_batches(
        batches,
        expect![[r#"
            Field { "id": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "category": Utf8, metadata: {"PARQUET:field_id": "2"} },
            Field { "value": Utf8, metadata: {"PARQUET:field_id": "3"} }"#]],
        expect![[r#"
            id: PrimitiveArray<Int32>
            [
              1,
              2,
              3,
            ],
            category: StringArray
            [
              "books",
              "electronics",
              "books",
            ],
            value: StringArray
            [
              "laptop",
              "phone",
              "novel",
            ]"#]],
        &[],
        Some("id"),
    );

    // Verify via a partition-filtered query that id=1 is now in the books partition.
    let batches = ctx
        .sql(
            "SELECT id FROM catalog.test_upd_mread_move.items \
             WHERE category = 'books' ORDER BY id",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let ids: Vec<i32> = batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    assert!(
        ids.contains(&1),
        "id=1 must now be in the books partition after MoR partition-move UPDATE"
    );
    assert!(
        !ids.contains(&2),
        "id=2 must stay in the electronics partition"
    );

    Ok(())
}

// ============================================================================
// MoR PARTITION PROBES — manifest-level position-delete partition-stamp verification
// ============================================================================

/// MoR PARTITION PROBE 1: cross-partition MoR DELETE — both partitions.
///
/// DELETE WHERE id > 0 matches rows in BOTH partitions.  The implementation
/// must produce TWO position-delete files (one per partition) each stamped
/// with the correct `(spec_id, partition Struct)` of their data file.
///
/// This is the hardest case for `write_position_deletes`: cross-partition
/// grouping.  A single delete file stamped with the wrong partition would
/// either be rejected at commit time or silently scope the deletes incorrectly.
#[tokio::test]
async fn test_delete_mread_cross_partition_manifest_stamp() -> Result<()> {
    let (ctx, client) = make_partitioned_mread_ctx("mread_partition_probe1", "items").await?;

    ctx.sql(
        "INSERT INTO catalog.mread_partition_probe1.items VALUES \
         (1, 'electronics', 'laptop'), \
         (2, 'electronics', 'phone'), \
         (3, 'books', 'novel'), \
         (4, 'books', 'textbook')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // Single INSERT creates ONE data file (all 4 rows in one batch) but the
    // partition-aware writer will split it into two files (one per partition).
    // Record the data-file partition structs so we can compare against delete files.
    let ns = NamespaceIdent::new("mread_partition_probe1".to_string());
    let tbl_id = iceberg::TableIdent::new(ns.clone(), "items".to_string());

    // DELETE all rows — hits every partition.
    let batches = ctx
        .sql("DELETE FROM catalog.mread_partition_probe1.items WHERE id > 0")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let deleted = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(deleted, 4, "all 4 rows deleted");

    // Inspect delete-file manifests.
    let table_after = client.load_table(&tbl_id).await?;
    let snap_after = table_after.metadata().current_snapshot().unwrap();
    let ml_after = snap_after
        .load_manifest_list(table_after.file_io(), table_after.metadata())
        .await?;

    let mut del_partitions: Vec<String> = Vec::new();
    let mut data_partitions: Vec<String> = Vec::new();
    for mf in ml_after.entries() {
        let m = mf.load_manifest(table_after.file_io()).await?;
        for entry in m.entries() {
            if !entry.is_alive() {
                continue;
            }
            let df = entry.data_file();
            let pv = df.partition();
            match df.content_type() {
                iceberg::spec::DataContentType::Data => {
                    if let Some(iceberg::spec::Literal::Primitive(
                        iceberg::spec::PrimitiveLiteral::String(s),
                    )) = pv.fields().first().and_then(|f| f.as_ref())
                    {
                        data_partitions.push(s.clone());
                    }
                }
                iceberg::spec::DataContentType::PositionDeletes => {
                    // Each delete file must carry a non-empty partition struct (identity spec).
                    assert!(
                        !pv.fields().is_empty(),
                        "delete file partition struct must not be empty for a partitioned table; \
                         delete_file={:?}",
                        df.file_path()
                    );
                    // The partition_spec_id must point to the table's partitioned spec.
                    assert_ne!(
                        df.partition_spec_id(),
                        -1,
                        "delete file must have a valid partition_spec_id"
                    );
                    if let Some(iceberg::spec::Literal::Primitive(
                        iceberg::spec::PrimitiveLiteral::String(s),
                    )) = pv.fields().first().and_then(|f| f.as_ref())
                    {
                        del_partitions.push(s.clone());
                    }
                }
                _ => {}
            }
        }
    }

    // Data files: both partitions present.
    data_partitions.sort();
    assert_eq!(
        data_partitions,
        vec!["books".to_string(), "electronics".to_string()],
        "expected data files in both partitions; got {data_partitions:?}"
    );

    // Delete files: one per partition, matching data-file partitions exactly.
    del_partitions.sort();
    assert_eq!(
        del_partitions,
        vec!["books".to_string(), "electronics".to_string()],
        "expected exactly one delete file per partition; got {del_partitions:?}"
    );

    // Post-delete SELECT must be empty.
    let batches = ctx
        .sql("SELECT COUNT(*) as c FROM catalog.mread_partition_probe1.items")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let count: i64 = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 0, "all rows deleted; SELECT must return 0");

    Ok(())
}

/// MoR PARTITION PROBE 2: MoR UPDATE — manifest-level partition stamp on delete files.
///
/// The position-delete files committed by a MoR UPDATE must be stamped with
/// the EXACT `(spec_id, partition Struct)` of the data file they delete from.
/// This test inspects the committed delete files at the manifest level — not
/// just the SELECT result — so a wrong stamp (e.g. empty Struct) would fail
/// here even if the scan happens to still resolve correctly in some engine.
#[tokio::test]
async fn test_update_mread_partitioned_delete_file_stamp() -> Result<()> {
    let (ctx, client) = make_partitioned_mread_ctx("mread_partition_probe2", "items").await?;

    ctx.sql(
        "INSERT INTO catalog.mread_partition_probe2.items VALUES \
         (1, 'electronics', 'laptop'), \
         (2, 'electronics', 'phone'), \
         (3, 'books', 'novel')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // UPDATE electronics rows only.
    let batches = ctx
        .sql(
            "UPDATE catalog.mread_partition_probe2.items \
             SET value = 'UPDATED' WHERE category = 'electronics'",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let upd_count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(upd_count, 2, "2 electronics rows updated");

    // Inspect delete-file partition stamps in the post-UPDATE snapshot.
    let ns = NamespaceIdent::new("mread_partition_probe2".to_string());
    let tbl_id = iceberg::TableIdent::new(ns.clone(), "items".to_string());
    let table_after = client.load_table(&tbl_id).await?;
    let snap_after = table_after.metadata().current_snapshot().unwrap();
    let ml_after = snap_after
        .load_manifest_list(table_after.file_io(), table_after.metadata())
        .await?;

    let mut del_partitions: Vec<String> = Vec::new();
    for mf in ml_after.entries() {
        let m = mf.load_manifest(table_after.file_io()).await?;
        for entry in m.entries() {
            if !entry.is_alive() {
                continue;
            }
            let df = entry.data_file();
            if df.content_type() != iceberg::spec::DataContentType::PositionDeletes {
                continue;
            }
            let pv = df.partition();
            assert!(
                !pv.fields().is_empty(),
                "delete file partition struct must not be empty; file={:?} partition={:?}",
                df.file_path(),
                pv
            );
            if let Some(iceberg::spec::Literal::Primitive(
                iceberg::spec::PrimitiveLiteral::String(s),
            )) = pv.fields().first().and_then(|f| f.as_ref())
            {
                del_partitions.push(s.clone());
            }
        }
    }

    // The UPDATE touched only electronics rows → delete file must be in 'electronics' partition.
    assert_eq!(
        del_partitions,
        vec!["electronics".to_string()],
        "delete file must be stamped with the 'electronics' partition; got {del_partitions:?}"
    );

    // Verify the SELECT result as a second sanity check.
    let batches = ctx
        .sql("SELECT * FROM catalog.mread_partition_probe2.items ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 3, "all 3 rows survive; 2 updated + 1 unchanged");

    Ok(())
}

/// MoR PARTITION PROBE 3: MoR UPDATE spanning two partitions — two delete files, each correctly stamped.
///
/// UPDATE touches rows in BOTH partitions (updates the `value` column).
/// The implementation must produce two delete files (one per partition),
/// each stamped with its partition's `category` value.  A merged or incorrectly-stamped
/// delete file would fail here.
#[tokio::test]
async fn test_update_mread_cross_partition_delete_stamps() -> Result<()> {
    let (ctx, client) = make_partitioned_mread_ctx("mread_partition_probe3", "items").await?;

    ctx.sql(
        "INSERT INTO catalog.mread_partition_probe3.items VALUES \
         (1, 'electronics', 'laptop'), \
         (2, 'books', 'novel')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // UPDATE all rows (both partitions).
    let batches = ctx
        .sql("UPDATE catalog.mread_partition_probe3.items SET value = 'UPDATED' WHERE id > 0")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let upd_count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(upd_count, 2, "both rows updated");

    let ns = NamespaceIdent::new("mread_partition_probe3".to_string());
    let tbl_id = iceberg::TableIdent::new(ns.clone(), "items".to_string());
    let table_after = client.load_table(&tbl_id).await?;
    let snap_after = table_after.metadata().current_snapshot().unwrap();
    let ml_after = snap_after
        .load_manifest_list(table_after.file_io(), table_after.metadata())
        .await?;

    let mut del_partitions: Vec<String> = Vec::new();
    for mf in ml_after.entries() {
        let m = mf.load_manifest(table_after.file_io()).await?;
        for entry in m.entries() {
            if !entry.is_alive() {
                continue;
            }
            let df = entry.data_file();
            if df.content_type() != iceberg::spec::DataContentType::PositionDeletes {
                continue;
            }
            let pv = df.partition();
            assert!(
                !pv.fields().is_empty(),
                "delete file partition struct must not be empty for a partitioned table; \
                 file={:?} partition={:?}",
                df.file_path(),
                pv
            );
            if let Some(iceberg::spec::Literal::Primitive(
                iceberg::spec::PrimitiveLiteral::String(s),
            )) = pv.fields().first().and_then(|f| f.as_ref())
            {
                del_partitions.push(s.clone());
            }
        }
    }

    del_partitions.sort();
    assert_eq!(
        del_partitions,
        vec!["books".to_string(), "electronics".to_string()],
        "must have one delete file per partition; got {del_partitions:?}"
    );

    // SELECT must show updated values for both rows.
    let batches = ctx
        .sql("SELECT * FROM catalog.mread_partition_probe3.items ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 2, "both rows survive after UPDATE");

    Ok(())
}

/// MoR PARTITION PROBE 4: MoR UPDATE on a partitioned table where rows in one
/// partition come from TWO distinct data files — confirm both sets of positions
/// are grouped into a SINGLE delete file for that partition (same Struct → same group).
///
/// This probes `Struct` equality/hashing: two data files with the same partition
/// value must produce identical `Struct` keys, collapsing into one group and one
/// delete file.  A hash/equality bug would produce two delete files for the same
/// partition.
#[tokio::test]
async fn test_update_mread_two_files_same_partition_single_delete() -> Result<()> {
    let (ctx, client) = make_partitioned_mread_ctx("mread_partition_probe4", "items").await?;

    // Two separate INSERT statements → two data files, both in 'electronics' partition.
    ctx.sql("INSERT INTO catalog.mread_partition_probe4.items VALUES (1, 'electronics', 'laptop')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    ctx.sql("INSERT INTO catalog.mread_partition_probe4.items VALUES (2, 'electronics', 'phone')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // UPDATE both rows (both files, same partition).
    let batches = ctx
        .sql(
            "UPDATE catalog.mread_partition_probe4.items \
             SET value = 'UPDATED' WHERE category = 'electronics'",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let upd_count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(upd_count, 2, "2 rows updated");

    let ns = NamespaceIdent::new("mread_partition_probe4".to_string());
    let tbl_id = iceberg::TableIdent::new(ns.clone(), "items".to_string());
    let table_after = client.load_table(&tbl_id).await?;
    let snap_after = table_after.metadata().current_snapshot().unwrap();
    let ml_after = snap_after
        .load_manifest_list(table_after.file_io(), table_after.metadata())
        .await?;

    let mut del_files: Vec<(String, String)> = Vec::new(); // (file_path, partition_val)
    for mf in ml_after.entries() {
        let m = mf.load_manifest(table_after.file_io()).await?;
        for entry in m.entries() {
            if !entry.is_alive() {
                continue;
            }
            let df = entry.data_file();
            if df.content_type() != iceberg::spec::DataContentType::PositionDeletes {
                continue;
            }
            let pv = df.partition();
            assert!(
                !pv.fields().is_empty(),
                "delete file partition struct must not be empty; file={:?}",
                df.file_path()
            );
            if let Some(iceberg::spec::Literal::Primitive(
                iceberg::spec::PrimitiveLiteral::String(s),
            )) = pv.fields().first().and_then(|f| f.as_ref())
            {
                del_files.push((df.file_path().to_string(), s.clone()));
            }
        }
    }

    // Both data files are in 'electronics' → must collapse to exactly ONE delete file.
    assert_eq!(
        del_files.len(),
        1,
        "two data files in the SAME partition must produce exactly ONE delete file; \
         got {del_files:?}"
    );
    assert_eq!(
        del_files[0].1, "electronics",
        "the single delete file must be stamped with 'electronics'; got {:?}",
        del_files[0].1
    );

    // Both rows must survive with updated value.
    let batches = ctx
        .sql("SELECT * FROM catalog.mread_partition_probe4.items ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 2, "both rows survive");

    Ok(())
}

// =================================================================================================
// NULL three-valued-logic — a predicate evaluating to NULL is NOT a match (the row is neither
// deleted nor updated). The implementation enforces this with `mask.is_valid(row) && mask.value(row)`
// in every DML path; these tests make that guard load-bearing (inverting it goes RED here).
// =================================================================================================

/// Copy-on-write DELETE: `foo2 = 'alan'` is NULL for the NULL-`foo2` row, so that row must SURVIVE.
#[tokio::test]
async fn test_delete_cow_null_predicate_three_valued_logic() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_delete_cow_null".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = get_table_creation(temp_path(), "my_table", Some(nullable_foo_schema()))?;
    iceberg_catalog.create_table(&namespace, creation).await?;
    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql("INSERT INTO catalog.test_delete_cow_null.my_table VALUES (1, 'alan'), (2, NULL), (3, 'bob')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let df = ctx
        .sql("DELETE FROM catalog.test_delete_cow_null.my_table WHERE foo2 = 'alan'")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let deleted = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(
        deleted.value(0),
        1,
        "only the foo2='alan' row is deleted; the NULL-foo2 row is NOT a match (NULL != TRUE)"
    );

    let ids = select_foo1_sorted(&ctx, "catalog.test_delete_cow_null.my_table").await;
    assert_eq!(
        ids,
        vec![2, 3],
        "the NULL-foo2 row (foo1=2) SURVIVES — a NULL predicate result is not a delete match"
    );
    Ok(())
}

/// Merge-on-read DELETE: same three-valued-logic contract on the position-delete path.
#[tokio::test]
async fn test_delete_mread_null_predicate_three_valued_logic() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_delete_mread_null".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = nullable_merge_on_read_table_creation(temp_path(), "my_table");
    iceberg_catalog.create_table(&namespace, creation).await?;
    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql("INSERT INTO catalog.test_delete_mread_null.my_table VALUES (1, 'alan'), (2, NULL), (3, 'bob')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let df = ctx
        .sql("DELETE FROM catalog.test_delete_mread_null.my_table WHERE foo2 = 'alan'")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let deleted = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(
        deleted.value(0),
        1,
        "only foo2='alan' deleted; NULL-foo2 row is not a match"
    );

    let ids = select_foo1_sorted(&ctx, "catalog.test_delete_mread_null.my_table").await;
    assert_eq!(
        ids,
        vec![2, 3],
        "the NULL-foo2 row (foo1=2) SURVIVES the merge-on-read delete"
    );
    Ok(())
}

/// Copy-on-write UPDATE (exercises `match_mask`, shared by both UPDATE modes): the NULL-`foo2` row
/// must NOT be updated — its `foo1` stays 2, not 99.
#[tokio::test]
async fn test_update_cow_null_predicate_three_valued_logic() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_update_cow_null".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = get_table_creation(temp_path(), "my_table", Some(nullable_foo_schema()))?;
    iceberg_catalog.create_table(&namespace, creation).await?;
    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql("INSERT INTO catalog.test_update_cow_null.my_table VALUES (1, 'alan'), (2, NULL), (3, 'bob')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let df = ctx
        .sql("UPDATE catalog.test_update_cow_null.my_table SET foo1 = 99 WHERE foo2 = 'alan'")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let updated = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(
        updated.value(0),
        1,
        "only foo2='alan' updated; NULL-foo2 row is not a match"
    );

    // Rows are now (99,'alan'), (2,NULL), (3,'bob'); the NULL row's foo1 must be UNCHANGED at 2.
    let ids = select_foo1_sorted(&ctx, "catalog.test_update_cow_null.my_table").await;
    assert_eq!(
        ids,
        vec![2, 3, 99],
        "the NULL-foo2 row keeps foo1=2 (not updated to 99) — NULL predicate is not an update match"
    );
    Ok(())
}

/// Collect `foo1` from a table, ascending, as a plain `Vec<i32>` for order-independent assertions.
async fn select_foo1_sorted(ctx: &SessionContext, table: &str) -> Vec<i32> {
    let df = ctx
        .sql(&format!("SELECT foo1 FROM {table} ORDER BY foo1"))
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let mut ids: Vec<i32> = Vec::new();
    for batch in &batches {
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for row in 0..batch.num_rows() {
            ids.push(col.value(row));
        }
    }
    ids
}

/// `{foo1 required int, foo2 OPTIONAL string}` — a nullable `foo2` so a row can carry a NULL operand
/// for the three-valued-logic tests above.
fn nullable_foo_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "foo1", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "foo2", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .unwrap()
}

/// A merge-on-read `{foo1, foo2-nullable}` table creation for the NULL three-valued-logic test.
fn nullable_merge_on_read_table_creation(
    location: impl ToString,
    name: impl ToString,
) -> TableCreation {
    TableCreation::builder()
        .location(location.to_string())
        .name(name.to_string())
        .properties(HashMap::from([
            ("write.delete.mode".to_string(), "merge-on-read".to_string()),
            ("write.update.mode".to_string(), "merge-on-read".to_string()),
        ]))
        .schema(nullable_foo_schema())
        .build()
}
