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

use datafusion::arrow::array::{Array, StringArray, UInt64Array};
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
// CRITIC PROBE TESTS — U1 COW DELETE adversarial verification
// ============================================================================

/// CRITIC PROBE 1: Verify affected-path matching at the manifest level.
/// After COW DELETE, inspect the post-commit snapshot's manifest data files
/// directly (not just SELECT row counts). Confirm:
/// a) The deleted source file is gone from the live manifest set.
/// b) A new rewritten file was added (distinct path).
/// c) The unaffected file is still present with its ORIGINAL path.
/// This distinguishes silent no-op (old file stays + new file added = DUPLICATE rows)
/// from correct behavior.
#[tokio::test]
async fn test_delete_cow_path_matching_and_manifest_inspection() -> Result<()> {
    let (ctx, client) = make_partitioned_delete_ctx("critic_probe1", "items").await?;

    // Insert two batches into two separate transactions to get TWO distinct data files.
    ctx.sql("INSERT INTO catalog.critic_probe1.items VALUES (1, 'electronics', 'laptop')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    ctx.sql("INSERT INTO catalog.critic_probe1.items VALUES (2, 'books', 'novel')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // Load the table BEFORE delete to record existing file paths.
    let ns = NamespaceIdent::new("critic_probe1".to_string());
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
        .sql("DELETE FROM catalog.critic_probe1.items WHERE category = 'electronics'")
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
        .sql("SELECT id, category FROM catalog.critic_probe1.items")
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

/// CRITIC PROBE 2: Multi-file-per-partition — DELETE hits only FILE A; FILE B must be untouched.
/// Partition has 2 data files (two INSERT transactions into the same partition).
/// DELETE predicate matches rows in file A only.
/// After commit: file A is gone, file B has its ORIGINAL path (not rewritten).
#[tokio::test]
async fn test_delete_cow_multi_file_per_partition_only_affected_rewritten() -> Result<()> {
    let (ctx, client) = make_partitioned_delete_ctx("critic_probe2", "items").await?;

    // Two separate INSERT statements into the SAME partition → two distinct data files for 'electronics'.
    ctx.sql("INSERT INTO catalog.critic_probe2.items VALUES (1, 'electronics', 'laptop')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    ctx.sql("INSERT INTO catalog.critic_probe2.items VALUES (2, 'electronics', 'tablet')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ns = NamespaceIdent::new("critic_probe2".to_string());
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
    ctx.sql("DELETE FROM catalog.critic_probe2.items WHERE id = 1")
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
        .sql("SELECT id FROM catalog.critic_probe2.items ORDER BY id")
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

/// CRITIC PROBE 3: Non-identity partition transform (truncate[4] on category string).
/// This tests PartitionValueCalculator with a real transform, not identity.
/// If the partition column calculation is wrong, the rewritten file would be assigned
/// to the wrong partition → DataFile.partition() would carry the wrong value.
#[tokio::test]
async fn test_delete_cow_non_identity_transform_truncate() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("critic_probe3".to_string());
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
        "INSERT INTO catalog.critic_probe3.trunc_table VALUES \
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
        .sql("DELETE FROM catalog.critic_probe3.trunc_table WHERE id = 1")
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
        .sql("SELECT id FROM catalog.critic_probe3.trunc_table ORDER BY id")
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

/// CRITIC PROBE 4: Verify a DELETE WHERE predicate that hits NO rows is a no-op (no new snapshot).
#[tokio::test]
async fn test_delete_cow_no_match_is_noop() -> Result<()> {
    let (ctx, client) = make_partitioned_delete_ctx("critic_probe4", "items").await?;

    ctx.sql("INSERT INTO catalog.critic_probe4.items VALUES (1, 'electronics', 'laptop')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ns = NamespaceIdent::new("critic_probe4".to_string());
    let tbl_id = iceberg::TableIdent::new(ns, "items".to_string());
    let table_before = client.load_table(&tbl_id).await?;
    let snap_id_before = table_before
        .metadata()
        .current_snapshot()
        .map(|s| s.snapshot_id());

    // DELETE where nothing matches.
    let batches = ctx
        .sql("DELETE FROM catalog.critic_probe4.items WHERE category = 'books'")
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
