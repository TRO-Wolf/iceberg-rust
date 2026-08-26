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

/// A `{foo1 int, foo2 string}` table with both DML modes set to merge-on-read, so DELETE and
/// UPDATE take the position-delete path.
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
    // An empty table plans one partition, and `execute(i)` past it is a typed error.
    let stream = plan.execute(0, task_ctx).unwrap();

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
            "my_table$position_deletes",
        ]
    "#]]
    .assert_debug_eq(&result);

    Ok(())
}

#[tokio::test]
async fn test_dollar_in_base_table_name_sql_read_and_metadata_twin() -> Result<()> {
    // `split_once('$')` makes `table_exist("a$b")` false and `a$b$files` unresolvable.
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_dollar_name".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let creation = get_table_creation(temp_path(), "a$b", None)?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    let provider = ctx.catalog("catalog").unwrap();
    let schema = provider.schema("test_dollar_name").unwrap();

    assert!(schema.table_exist("a$b"));
    assert!(schema.table_exist("a$b$snapshots"));
    assert!(schema.table_exist("a$b$files"));
    let names = schema.table_names();
    assert!(names.contains(&"a$b".to_string()));
    assert!(names.contains(&"a$b$files".to_string()));

    // Plain a$b read (empty table).
    let batches = ctx
        .sql("SELECT * FROM catalog.test_dollar_name.\"a$b\"")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
    assert_eq!(rows, 0);

    // Metadata twin of a table whose name already contains `$`.
    let snapshots = ctx
        .sql("SELECT * FROM catalog.test_dollar_name.\"a$b$snapshots\"")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert!(
        snapshots[0].schema().field_with_name("snapshot_id").is_ok(),
        "a$b$snapshots must resolve as the snapshots metadata table"
    );

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
    let s = record
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(2, s.len());
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
    let s = record
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(2, s.len());
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
            Field { "committed_at": Timestamp(µs, "UTC"), metadata: {"PARQUET:field_id": "1"} },
            Field { "snapshot_id": Int64, metadata: {"PARQUET:field_id": "2"} },
            Field { "parent_id": nullable Int64, metadata: {"PARQUET:field_id": "3"} },
            Field { "operation": nullable Utf8, metadata: {"PARQUET:field_id": "4"} },
            Field { "manifest_list": nullable Utf8, metadata: {"PARQUET:field_id": "5"} },
            Field { "summary": nullable Map("key_value": non-null Struct("key": non-null Utf8, metadata: {"PARQUET:field_id": "7"}, "value": Utf8, metadata: {"PARQUET:field_id": "8"}), unsorted), metadata: {"PARQUET:field_id": "6"} }"#]],
        expect![[r#"
            committed_at: PrimitiveArray<Timestamp(µs, "UTC")>
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

    let df = ctx
        .sql("INSERT INTO catalog.test_insert_into.my_table VALUES (1, 'alan'), (2, 'turing')")
        .await
        .unwrap();

    let batches = df.collect().await.unwrap();
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert!(
        batch.num_rows() == 1 && batch.num_columns() == 1,
        "Results should only have one row and one column that has the number of rows inserted"
    );
    let rows_inserted = batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(rows_inserted.value(0), 2);

    let df = ctx
        .sql("SELECT * FROM catalog.test_insert_into.my_table")
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

    // INSERT OVERWRITE replaces all data: DataFusion's `InsertOp::Overwrite` becomes
    // `overwrite_files().overwrite_by_row_filter(AlwaysTrue)`, one snapshot.
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

    // Only the overwrite rows survive. An append would leave 4 rows.
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

    // `lower(foo2)` is not convertible to an Iceberg predicate. A delete that trusts inexact
    // pushdown loosens the filter to `foo1 > 0` and deletes all three rows. The exact filter
    // removes rows 1 and 3 only.
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

    // Two INSERT statements give two data files, and `_pos` is file-local in each.
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

    // Delete one row from each file. The position deletes must be path-keyed per file, because
    // `_pos` is file-local. Survivors are exactly {1, 4}.
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

    // The same discriminating filter as the MoR test. Copy-on-write must rewrite the file
    // keeping only (2,'turing').
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

    // Only rows 1 and 3 match. `foo1 = foo1 + 100` proves the assignment reads the old value.
    // Merge-on-read writes the new rows and position-deletes the old, in one RowDelta.
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

/// COW UPDATE of a non-partition column, matching one partition of two. The electronics rows take
/// the new value and the books rows stay unchanged.
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

/// COW UPDATE that changes the partition-key column. `SET category = 'books' WHERE id = 1` moves
/// id 1 into the books partition. Every other row stays unchanged.
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

/// The unpartitioned COW UPDATE under the file-level path. `lower(foo2) = 'alan'` is
/// unconvertible, so an inexact pushdown over-updates. The exact eval and the assignment
/// expression must both survive.
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

/// The copy-on-write twin of the test above.
///
/// Streaming changed the failure timing: a later batch can trip the NULL guard while batch 1 is
/// already inside an open writer, which is then dropped without `close()` and leaves staged files.
/// What must not change is that the statement errors and the table stays exactly as it was.
#[tokio::test]
async fn test_update_cow_null_into_required_is_rejected() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_update_cow_null_req".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    // Empty properties ⇒ copy-on-write for UPDATE.
    let creation = get_table_creation(temp_path(), "my_table", None)?;
    iceberg_catalog.create_table(&namespace, creation).await?;
    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql("INSERT INTO catalog.test_update_cow_null_req.my_table VALUES (1, 'a'), (2, 'b')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let outcome = ctx
        .sql("UPDATE catalog.test_update_cow_null_req.my_table SET foo1 = NULL WHERE foo2 = 'a'")
        .await;
    let errored = match outcome {
        Err(_) => true,
        Ok(df) => df.collect().await.is_err(),
    };
    assert!(
        errored,
        "copy-on-write UPDATE assigning NULL to the required column foo1 must error, not write a null"
    );

    // Nothing was committed: both original rows are intact and unchanged.
    let ids = select_foo1_sorted(&ctx, "catalog.test_update_cow_null_req.my_table").await;
    assert_eq!(
        ids,
        vec![1, 2],
        "a rejected copy-on-write UPDATE must leave the table exactly as it was — no commit, no \
         partial rewrite"
    );
    Ok(())
}

fn get_nested_struct_type() -> StructType {
    // The leaf fields under `address` are OPTIONAL, not REQUIRED. DataFusion's nested-struct
    // cast validation rejects a cast from a nullable SQL `named_struct` field to a non-nullable
    // target field. A required-nested insert through `named_struct` needs an engine path that
    // checks non-null at run time instead of rejecting at planning time.
    StructType::new(vec![
        NestedField::optional(
            10,
            "address",
            Type::Struct(StructType::new(vec![
                NestedField::optional(11, "street", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(12, "city", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(13, "zip", Type::Primitive(PrimitiveType::Int)).into(),
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

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "profile", Type::Struct(get_nested_struct_type())).into(),
        ])
        .build()?;

    let creation = get_table_creation(temp_path(), table_name, Some(schema))?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);

    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    let provider = ctx.catalog("catalog").unwrap();
    let schema = provider.schema("test_insert_nested").unwrap();
    let table = schema.table("nested_table").await.unwrap().unwrap();
    let table_schema = table.schema();

    assert_eq!(table_schema.fields().len(), 3);
    assert_eq!(table_schema.field(0).name(), "id");
    assert_eq!(table_schema.field(1).name(), "name");
    assert_eq!(table_schema.field(2).name(), "profile");
    assert!(matches!(
        table_schema.field(2).data_type(),
        DataType::Struct(_)
    ));

    let insert_sql = r#"
    INSERT INTO catalog.test_insert_nested.nested_table
    SELECT 
        1 as id, 
        'Alice' as name,
        named_struct(
            'address', named_struct(
                'street', '123 Main St',
                'city', 'San Francisco',
                'zip', CAST(94105 AS INT)
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
                'zip', CAST(95113 AS INT)
            ),
            'contact', named_struct(
                'email', 'bob@example.com',
                'phone', NULL
            )
        ) as profile
    "#;

    let df = ctx.sql(insert_sql).await.unwrap();
    let batches = df.collect().await.unwrap();

    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert!(batch.num_rows() == 1 && batch.num_columns() == 1);

    let rows_inserted = batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(rows_inserted.value(0), 2);

    let df = ctx
        .sql("SELECT * FROM catalog.test_insert_nested.nested_table ORDER BY id")
        .await
        .unwrap();

    let batches = df.collect().await.unwrap();

    check_record_batches(
        batches,
        expect![[r#"
            Field { "id": Int32, metadata: {"PARQUET:field_id": "1"} },
            Field { "name": Utf8, metadata: {"PARQUET:field_id": "2"} },
            Field { "profile": nullable Struct("address": Struct("street": Utf8, metadata: {"PARQUET:field_id": "6"}, "city": Utf8, metadata: {"PARQUET:field_id": "7"}, "zip": Int32, metadata: {"PARQUET:field_id": "8"}), metadata: {"PARQUET:field_id": "4"}, "contact": Struct("email": Utf8, metadata: {"PARQUET:field_id": "9"}, "phone": Utf8, metadata: {"PARQUET:field_id": "10"}), metadata: {"PARQUET:field_id": "5"}), metadata: {"PARQUET:field_id": "3"} }"#]],
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
            -- child 0: "address" (Struct([Field { name: "street", data_type: Utf8, nullable: true, metadata: {"PARQUET:field_id": "6"} }, Field { name: "city", data_type: Utf8, nullable: true, metadata: {"PARQUET:field_id": "7"} }, Field { name: "zip", data_type: Int32, nullable: true, metadata: {"PARQUET:field_id": "8"} }]))
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

    let table_ident = TableIdent::new(namespace.clone(), "partitioned_table".to_string());
    let table = client.load_table(&table_ident).await?;
    let table_location = table.metadata().location();
    let file_io = table.file_io();

    let electronics_path = format!("{table_location}/data/category=electronics");
    let books_path = format!("{table_location}/data/category=books");
    let clothing_path = format!("{table_location}/data/category=clothing");

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

/// Builds an `identity(category)`-partitioned `{id, category, value}` table in a fresh
/// `SessionContext` and returns the context and the catalog. The caller names the namespace, so
/// tests do not collide.
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

/// COW DELETE of one partition of two. Only the books rows remain.
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

/// The unpartitioned COW DELETE under the file-level path. `lower(foo2) = 'alan'` is
/// unconvertible, so an inexact pushdown also removes row 2.
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

/// Affected-path matching, checked at the manifest level rather than by row count. The deleted
/// source file must leave the live set, a rewritten file must appear, and the unaffected file must
/// keep its original path. A silent no-op leaves the old file beside the new one and duplicates
/// rows.
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

    // The electronics file loses every row, so its rewrite produces no survivors. The books file
    // is unaffected. Exactly one live file remains.
    assert_eq!(
        paths_after.len(),
        1,
        "after full-file DELETE, live set must be exactly 1 (unaffected books file); got {paths_after:?}"
    );

    // The survivor must be the original books file. Content alone cannot tell the two pre-delete
    // paths apart, so assert the survivor is one of them.
    let rows = ctx
        .sql("SELECT id, category FROM catalog.dml_probe1.items")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = rows.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 1, "exactly 1 row survives");

    // The unaffected books file keeps its original path.
    let surviving_path = paths_after.iter().next().unwrap().clone();
    assert!(
        paths_before.contains(&surviving_path),
        "the surviving file must be the original unaffected books file (same path); \
         surviving={surviving_path}; paths_before={paths_before:?}"
    );

    Ok(())
}

/// A partition with two data files, where the DELETE matches file A only. File A goes and file B
/// keeps its original path.
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

    // File A loses its only row, so it produces no rewritten file. File B is unaffected. Exactly
    // one live file remains, at file B's original path.
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

/// A non-identity partition transform, `truncate[4]` on a string. A wrong partition calculation
/// puts the rewritten file in the wrong partition, so `DataFile.partition()` carries the wrong
/// value.
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
    // Two files survive: the untouched "book" file, and the rewritten "elec" file without id 1.
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

/// Row conservation for COW UPDATE, checked at the manifest level. file_A holds a matching and a
/// non-matching row, file_B sits in another partition.
///
/// file_A must be replaced by one new file of two rows, one updated and one carried. file_B must
/// keep its original path. The total stays at three rows. A wrong result shows as a dropped row,
/// a duplicated row, an old file left beside the new one, or a rewritten file_B.
#[tokio::test]
async fn test_update_cow_row_conservation_and_manifest_inspection() -> Result<()> {
    let (ctx, client) = make_partitioned_delete_ctx("upd_probe_u2_1", "items").await?;

    // Two transactions make file_A and file_B distinct. file_A holds ids 1 and 2.
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

/// A partition with two files, where the UPDATE matches file_A only. file_A is replaced by a new
/// one-row file and file_B keeps its original path and content.
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

/// COW UPDATE with no WHERE on a partitioned table, so every file is affected. All four rows take
/// the new value, the count returned is 4, and a new snapshot appears.
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

/// A zero-match COW UPDATE is a no-op: the count is 0 and no snapshot appears.
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

/// A partition move, checked at the `DataFile` level rather than by SELECT. One file holds a row
/// that moves to books and a row that stays in electronics.
///
/// The old file leaves the manifest, two new files appear, and each carries the right partition
/// struct. The row count stays at two.
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

/// Builds a V2 `identity(category)`-partitioned `{id, category, value}` table with both DML modes
/// set to merge-on-read. V2 is required for position deletes.
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

/// Partitioned MoR DELETE of one partition of two. Only the books rows survive. This is the
/// prerequisite for the MoR UPDATE path.
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

/// Partitioned MoR UPDATE of a non-partition column, matching one partition of two. One RowDelta
/// carries the position deletes and the new data file. The books rows stay unchanged.
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

/// MoR UPDATE that changes the partition-key column. One RowDelta position-deletes the old row in
/// its partition and writes the new row into the books partition. id 2 stays unchanged.
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

/// Cross-partition MoR DELETE. The write must produce one position-delete file per partition,
/// each stamped with its data file's `(spec_id, partition)`. A single file with the wrong stamp is
/// either rejected at commit or silently scoped wrong.
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

    // One INSERT, but the partition-aware writer splits it into one file per partition. Record
    // the partition structs to compare against the delete files.
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

/// The position-delete files a MoR UPDATE commits must carry the exact `(spec_id, partition)` of
/// the data file they delete from. This reads the manifest, not the SELECT result, so a wrong
/// stamp fails here even when a scan still resolves.
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

/// A MoR UPDATE spanning both partitions must produce one delete file per partition, each stamped
/// with its own `category`. A merged file, or one with the wrong stamp, fails here.
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

/// Two data files in one partition must group into a single delete file. This pins `Struct`
/// equality and hashing: equal partition values must give equal keys. A hashing defect produces
/// two delete files for one partition.
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
// An evolved table whose default spec is unpartitioned can still hold data under older
// partitioned specs. The fast path may stamp `partition_key = None` only when the table has one
// spec and that spec is unpartitioned.
// =================================================================================================

/// After DROP PARTITION FIELD (default becomes unpartitioned) a MoR DELETE must still stamp
/// each position-delete file with the data file's own `(spec_id, partition)` so the read-side
/// attach does not miss and resurrect rows.
#[tokio::test]
async fn test_delete_mread_after_drop_partition_field_no_resurrection() -> Result<()> {
    use iceberg::transaction::{ApplyTransactionAction, Transaction};

    let (ctx, client) = make_partitioned_mread_ctx("bug001_evolved_unpart", "items").await?;

    ctx.sql(
        "INSERT INTO catalog.bug001_evolved_unpart.items VALUES \
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

    let ns = NamespaceIdent::new("bug001_evolved_unpart".to_string());
    let tbl_id = TableIdent::new(ns.clone(), "items".to_string());

    // Removing identity(category) leaves an unpartitioned default, over partitioned data files.
    let table = client.load_table(&tbl_id).await?;
    assert_eq!(
        table.metadata().partition_specs_iter().len(),
        1,
        "fixture: one spec before evolution"
    );
    assert!(
        !table.metadata().default_partition_spec().is_unpartitioned(),
        "fixture: default is partitioned before DROP"
    );
    let tx = Transaction::new(&table);
    let tx = tx
        .update_partition_spec()
        .remove_field("category")
        .apply(tx)
        .expect("apply remove_field(category)");
    let table = tx.commit(client.as_ref()).await.expect("commit evolution");
    assert!(
        table.metadata().default_partition_spec().is_unpartitioned(),
        "fixture: default is unpartitioned after DROP PARTITION FIELD"
    );
    assert!(
        table.metadata().partition_specs_iter().len() > 1,
        "fixture: multi-spec after evolution (old partitioned + new unpartitioned default)"
    );

    // Re-register so DF sees the evolved metadata (provider caches per-load schema).
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let _ = ctx.register_catalog("catalog", catalog);

    // Collect pre-evolution data-file (spec_id, partition) for equality assert after DELETE.
    let mut data_stamps: HashMap<String, (i32, iceberg::spec::Struct)> = HashMap::new();
    {
        let snap = table.metadata().current_snapshot().expect("snapshot");
        let ml = snap
            .load_manifest_list(table.file_io(), table.metadata())
            .await?;
        for mf in ml.entries() {
            let m = mf.load_manifest(table.file_io()).await?;
            for entry in m.entries() {
                if !entry.is_alive() {
                    continue;
                }
                let df = entry.data_file();
                if df.content_type() == iceberg::spec::DataContentType::Data {
                    data_stamps.insert(
                        df.file_path().to_string(),
                        (df.partition_spec_id(), df.partition().clone()),
                    );
                }
            }
        }
    }

    // An unconditional fast path stamps `partition_key = None` here, the deletes never attach,
    // and every row comes back.
    let batches = ctx
        .sql("DELETE FROM catalog.bug001_evolved_unpart.items WHERE id IN (1, 3)")
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
    assert_eq!(deleted, 2, "two rows matched for delete");

    // Live scan: zero resurrection.
    let live = ctx
        .sql("SELECT id FROM catalog.bug001_evolved_unpart.items ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let ids: Vec<i32> = live
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(
        ids,
        vec![2, 4],
        "ids 1 and 3 must stay deleted (no resurrection under evolved unpartitioned default)"
    );

    // Manifest-level: every position-delete file must equal some data file's (spec_id, partition).
    let table_after = client.load_table(&tbl_id).await?;
    let snap = table_after
        .metadata()
        .current_snapshot()
        .expect("snapshot after delete");
    let ml = snap
        .load_manifest_list(table_after.file_io(), table_after.metadata())
        .await?;
    let mut pos_del_count = 0usize;
    for mf in ml.entries() {
        let m = mf.load_manifest(table_after.file_io()).await?;
        for entry in m.entries() {
            if !entry.is_alive() {
                continue;
            }
            let df = entry.data_file();
            if df.content_type() != iceberg::spec::DataContentType::PositionDeletes {
                continue;
            }
            pos_del_count += 1;
            let stamp = (df.partition_spec_id(), df.partition().clone());
            assert!(
                data_stamps.values().any(|d| d == &stamp),
                "pos-delete stamp (spec_id={}, partition={:?}) must equal a live data file's stamp; \
                 data_stamps={data_stamps:?}",
                stamp.0,
                stamp.1
            );
        }
    }
    assert!(
        pos_del_count >= 1,
        "at least one position-delete file must have been written"
    );

    // INSERT under the new unpartitioned default, then DELETE. The stamps must match the new
    // file's non-zero empty-spec id, not a fabricated spec 0.
    ctx.sql(
        "INSERT INTO catalog.bug001_evolved_unpart.items VALUES \
         (10, 'post', 'row-a'), (11, 'post', 'row-b')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let table_mid = client.load_table(&tbl_id).await?;
    let default_spec_id = table_mid.metadata().default_partition_spec_id();
    assert!(
        table_mid
            .metadata()
            .default_partition_spec()
            .is_unpartitioned(),
        "default still unpartitioned"
    );

    let batches = ctx
        .sql("DELETE FROM catalog.bug001_evolved_unpart.items WHERE id = 10")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0),
        1
    );

    let live2 = ctx
        .sql("SELECT id FROM catalog.bug001_evolved_unpart.items ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let ids2: Vec<i32> = live2
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(
        ids2,
        vec![2, 4, 11],
        "id 10 deleted under post-DROP unpartitioned default; 11 survives"
    );

    // At least one pos-delete must claim the current default_spec_id (empty tuple under that id).
    let table_final = client.load_table(&tbl_id).await?;
    let snap_f = table_final
        .metadata()
        .current_snapshot()
        .expect("final snap");
    let ml_f = snap_f
        .load_manifest_list(table_final.file_io(), table_final.metadata())
        .await?;
    let mut saw_default_spec_delete = false;
    for mf in ml_f.entries() {
        let m = mf.load_manifest(table_final.file_io()).await?;
        for entry in m.entries() {
            if !entry.is_alive() {
                continue;
            }
            let df = entry.data_file();
            if df.content_type() == iceberg::spec::DataContentType::PositionDeletes
                && df.partition_spec_id() == default_spec_id
            {
                saw_default_spec_delete = true;
                assert!(
                    df.partition().fields().is_empty(),
                    "post-DROP empty-spec delete must carry empty partition tuple"
                );
            }
        }
    }
    assert!(
        saw_default_spec_delete,
        "C1-L-001: post-DROP DELETE must stamp pos-deletes with default_spec_id={default_spec_id}, \
         not fabricated 0"
    );

    Ok(())
}

// =================================================================================================
// A predicate that evaluates to NULL is not a match, so the row is neither deleted nor updated.
// Every DML path enforces it with `mask.is_valid(row) && mask.value(row)`. These tests make that
// guard load-bearing.
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

// =================================================================================================
// Non-vacuous three-valued logic for the copy-on-write paths.
//
// The `=`-only tests above cannot falsify the `is_valid` guard: on a NULL operand Arrow gives
// (valid=false, value=false), so the guard is redundant there. Both stayed green when the guard
// was dropped. One `match_mask` line now governs three of the four DML paths, and these two `<>`
// tests turn red when it goes.
// =================================================================================================

/// COW DELETE with a `<>` predicate over a NULL operand: the NULL-`foo2` row evaluates to
/// (valid=false, value=TRUE), so ONLY the `is_valid` guard in `match_mask` keeps it out of the
/// affected/deleted set.
#[tokio::test]
async fn test_delete_cow_null_neq_predicate_isvalid_guard() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_del_cow_null_neq".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = get_table_creation(temp_path(), "my_table", Some(nullable_foo_schema()))?;
    iceberg_catalog.create_table(&namespace, creation).await?;
    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql("INSERT INTO catalog.test_del_cow_null_neq.my_table VALUES (1, 'alan'), (2, NULL), (3, 'bob')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let batches = ctx
        .sql("DELETE FROM catalog.test_del_cow_null_neq.my_table WHERE foo2 <> 'zzz'")
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
        "only the two non-NULL rows ('alan','bob') are deleted; the NULL row is UNKNOWN, not a match"
    );

    let survivors = select_foo1_sorted(&ctx, "catalog.test_del_cow_null_neq.my_table").await;
    assert_eq!(
        survivors,
        vec![2],
        "the NULL-foo2 row (foo1=2) is REWRITTEN as a survivor by the copy-on-write DELETE — the \
         `is_valid` guard is load-bearing under `<>`"
    );
    Ok(())
}

/// COW UPDATE with a `<>` predicate over a NULL operand. The guard is load-bearing TWICE here: in
/// pass 1 (would the NULL row make its file affected / bump the count) and in pass 2 (would the row
/// take the SET value instead of being carried unchanged).
#[tokio::test]
async fn test_update_cow_null_neq_predicate_isvalid_guard() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_upd_cow_null_neq".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = get_table_creation(temp_path(), "my_table", Some(nullable_foo_schema()))?;
    iceberg_catalog.create_table(&namespace, creation).await?;
    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql("INSERT INTO catalog.test_upd_cow_null_neq.my_table VALUES (1, 'alan'), (2, NULL), (3, 'bob')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let batches = ctx
        .sql("UPDATE catalog.test_upd_cow_null_neq.my_table SET foo1 = 99 WHERE foo2 <> 'zzz'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let updated = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(
        updated, 2,
        "only the two non-NULL rows are updated; the NULL row is UNKNOWN, not an update match"
    );

    let ids = select_foo1_sorted(&ctx, "catalog.test_upd_cow_null_neq.my_table").await;
    assert_eq!(
        ids,
        vec![2, 99, 99],
        "the NULL-foo2 row keeps foo1=2 through the copy-on-write rewrite — `is_valid` guards it"
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

/// `{foo1 required int, foo2 optional string}`, so a row can carry a NULL operand.
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

// =================================================================================================
// Merge-on-read DML streaming coverage.
//
// The MoR executors consume the live scan batch by batch. Three invariants must hold. The survivor
// set stays the same over a multi-file table whose scan interleaves batches. A failure at the
// single commit leaves zero new snapshots. A NULL predicate result is not a match.
// =================================================================================================

use std::fmt::Debug;

use iceberg::{Namespace, TableCommit};

/// A [`Catalog`] that delegates everything except `update_table`, which always fails. The DML
/// executors write their files and then attempt one commit. A failed commit must leave the table
/// untouched, with the staged files orphaned.
#[derive(Debug)]
struct FailingCommitCatalog {
    inner: Arc<dyn Catalog>,
}

#[async_trait::async_trait]
impl Catalog for FailingCommitCatalog {
    async fn list_namespaces(
        &self,
        parent: Option<&NamespaceIdent>,
    ) -> Result<Vec<NamespaceIdent>> {
        self.inner.list_namespaces(parent).await
    }

    async fn create_namespace(
        &self,
        namespace: &NamespaceIdent,
        properties: HashMap<String, String>,
    ) -> Result<Namespace> {
        self.inner.create_namespace(namespace, properties).await
    }

    async fn get_namespace(&self, namespace: &NamespaceIdent) -> Result<Namespace> {
        self.inner.get_namespace(namespace).await
    }

    async fn namespace_exists(&self, namespace: &NamespaceIdent) -> Result<bool> {
        self.inner.namespace_exists(namespace).await
    }

    async fn update_namespace(
        &self,
        namespace: &NamespaceIdent,
        properties: HashMap<String, String>,
    ) -> Result<()> {
        self.inner.update_namespace(namespace, properties).await
    }

    async fn drop_namespace(&self, namespace: &NamespaceIdent) -> Result<()> {
        self.inner.drop_namespace(namespace).await
    }

    async fn list_tables(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>> {
        self.inner.list_tables(namespace).await
    }

    async fn create_table(
        &self,
        namespace: &NamespaceIdent,
        creation: TableCreation,
    ) -> Result<iceberg::table::Table> {
        self.inner.create_table(namespace, creation).await
    }

    async fn load_table(&self, table: &TableIdent) -> Result<iceberg::table::Table> {
        self.inner.load_table(table).await
    }

    async fn drop_table(&self, table: &TableIdent) -> Result<()> {
        self.inner.drop_table(table).await
    }

    async fn table_exists(&self, table: &TableIdent) -> Result<bool> {
        self.inner.table_exists(table).await
    }

    async fn rename_table(&self, src: &TableIdent, dest: &TableIdent) -> Result<()> {
        self.inner.rename_table(src, dest).await
    }

    async fn register_table(
        &self,
        table: &TableIdent,
        metadata_location: String,
    ) -> Result<iceberg::table::Table> {
        self.inner.register_table(table, metadata_location).await
    }

    /// The injected fault: the single commit at the end of every MoR DML op fails here.
    async fn update_table(&self, _commit: TableCommit) -> Result<iceberg::table::Table> {
        Err(iceberg::Error::new(
            iceberg::ErrorKind::Unexpected,
            "injected commit failure (H-COMMIT fault injection)",
        ))
    }
}

/// A MoR DELETE over four data files, whose scan interleaves batches, must delete exactly the
/// matched rows. Streaming changes how rows reach the writer, never which rows go.
#[tokio::test]
async fn test_delete_mread_streaming_multifile_survivor_set() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_mread_stream_del".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = get_merge_on_read_table_creation(temp_path(), "my_table")?;
    iceberg_catalog.create_table(&namespace, creation).await?;
    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    // Four separate inserts ⇒ four data files; the default scan interleaves their batches.
    for chunk in [
        "(1, 'a'), (2, 'b')",
        "(3, 'c'), (4, 'd')",
        "(5, 'e'), (6, 'f')",
        "(7, 'g'), (8, 'h')",
    ] {
        ctx.sql(&format!(
            "INSERT INTO catalog.test_mread_stream_del.my_table VALUES {chunk}"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    }

    // Delete the even foo1 rows (spread across every data file) — the oracle survivors are the odds.
    let batches = ctx
        .sql("DELETE FROM catalog.test_mread_stream_del.my_table WHERE foo1 % 2 = 0")
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
    assert_eq!(deleted, 4, "exactly the 4 even-foo1 rows are deleted");

    let survivors = select_foo1_sorted(&ctx, "catalog.test_mread_stream_del.my_table").await;
    assert_eq!(
        survivors,
        vec![1, 3, 5, 7],
        "streaming MoR DELETE leaves EXACTLY the odd-foo1 oracle survivor set across all files"
    );
    Ok(())
}

/// The same over a MoR UPDATE. The new rows stream per batch and the delete side buffers only the
/// matched pairs. Updated and untouched rows must both match the oracle.
#[tokio::test]
async fn test_update_mread_streaming_multifile_survivor_set() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_mread_stream_upd".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = get_merge_on_read_table_creation(temp_path(), "my_table")?;
    iceberg_catalog.create_table(&namespace, creation).await?;
    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    for chunk in [
        "(1, 'a'), (2, 'b')",
        "(3, 'c'), (4, 'd')",
        "(5, 'e'), (6, 'f')",
    ] {
        ctx.sql(&format!(
            "INSERT INTO catalog.test_mread_stream_upd.my_table VALUES {chunk}"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    }

    // Set foo1 = foo1 + 100 for the even rows (2,4,6) — spread across every file.
    let batches = ctx
        .sql("UPDATE catalog.test_mread_stream_upd.my_table SET foo1 = foo1 + 100 WHERE foo1 % 2 = 0")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let updated = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(updated, 3, "exactly the 3 even-foo1 rows are updated");

    let ids = select_foo1_sorted(&ctx, "catalog.test_mread_stream_upd.my_table").await;
    assert_eq!(
        ids,
        vec![1, 3, 5, 102, 104, 106],
        "odd rows unchanged; even rows +100 — streaming MoR UPDATE matches the oracle exactly"
    );
    Ok(())
}

/// With a catalog whose commit fails, a MoR DELETE must error and leave the snapshot id and count
/// unchanged. The position-delete file is staged, then orphaned.
#[tokio::test]
async fn test_delete_mread_commit_failure_leaves_snapshot_unchanged() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_mread_del_fault".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = get_merge_on_read_table_creation(temp_path(), "my_table")?;
    iceberg_catalog.create_table(&namespace, creation).await?;
    let client = Arc::new(iceberg_catalog);

    // First seed data through a WORKING catalog provider so there is a baseline snapshot.
    let good_provider = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let good_ctx = SessionContext::new();
    good_ctx.register_catalog("catalog", good_provider);
    good_ctx
        .sql(
            "INSERT INTO catalog.test_mread_del_fault.my_table VALUES (1, 'a'), (2, 'b'), (3, 'c')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let tbl_id = TableIdent::new(namespace.clone(), "my_table".to_string());
    let before = client.load_table(&tbl_id).await?;
    let snapshot_id_before = before.metadata().current_snapshot_id();
    let snapshot_count_before = before.metadata().snapshots().count();

    // Now run the DELETE through a catalog whose commit ALWAYS fails.
    let failing: Arc<dyn Catalog> = Arc::new(FailingCommitCatalog {
        inner: client.clone(),
    });
    let failing_provider = Arc::new(IcebergCatalogProvider::try_new(failing).await?);
    let bad_ctx = SessionContext::new();
    bad_ctx.register_catalog("catalog", failing_provider);

    let result = bad_ctx
        .sql("DELETE FROM catalog.test_mread_del_fault.my_table WHERE foo1 >= 1")
        .await
        .unwrap()
        .collect()
        .await;
    let err = result.expect_err("the DELETE must surface the injected commit failure as an error");
    // Non-vacuity: the error is the injected commit fault, not an earlier failure.
    assert!(
        err.to_string().contains("injected commit failure"),
        "the error must be the injected commit fault, got: {err}"
    );

    // Reload from the real catalog: the commit failed after the writes, so no snapshot moved.
    let after = client.load_table(&tbl_id).await?;
    assert_eq!(
        after.metadata().current_snapshot_id(),
        snapshot_id_before,
        "commit failure must NOT advance current_snapshot_id (commit-once-after-close atomicity)"
    );
    assert_eq!(
        after.metadata().snapshots().count(),
        snapshot_count_before,
        "commit failure must NOT create a new snapshot — staged delete file is orphaned, not committed"
    );

    // And the data is fully intact — no rows were actually deleted.
    let good_ctx2 = SessionContext::new();
    good_ctx2.register_catalog(
        "catalog",
        Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?),
    );
    let survivors = select_foo1_sorted(&good_ctx2, "catalog.test_mread_del_fault.my_table").await;
    assert_eq!(
        survivors,
        vec![1, 2, 3],
        "all rows survive the failed DELETE — the table is not torn"
    );
    Ok(())
}

/// The same invariant for MoR UPDATE, which also writes new data files before the one commit.
#[tokio::test]
async fn test_update_mread_commit_failure_leaves_snapshot_unchanged() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_mread_upd_fault".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = get_merge_on_read_table_creation(temp_path(), "my_table")?;
    iceberg_catalog.create_table(&namespace, creation).await?;
    let client = Arc::new(iceberg_catalog);

    let good_provider = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let good_ctx = SessionContext::new();
    good_ctx.register_catalog("catalog", good_provider);
    good_ctx
        .sql(
            "INSERT INTO catalog.test_mread_upd_fault.my_table VALUES (1, 'a'), (2, 'b'), (3, 'c')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let tbl_id = TableIdent::new(namespace.clone(), "my_table".to_string());
    let before = client.load_table(&tbl_id).await?;
    let snapshot_id_before = before.metadata().current_snapshot_id();
    let snapshot_count_before = before.metadata().snapshots().count();

    let failing: Arc<dyn Catalog> = Arc::new(FailingCommitCatalog {
        inner: client.clone(),
    });
    let failing_provider = Arc::new(IcebergCatalogProvider::try_new(failing).await?);
    let bad_ctx = SessionContext::new();
    bad_ctx.register_catalog("catalog", failing_provider);

    let result = bad_ctx
        .sql("UPDATE catalog.test_mread_upd_fault.my_table SET foo2 = 'z' WHERE foo1 >= 1")
        .await
        .unwrap()
        .collect()
        .await;
    let err = result.expect_err("the UPDATE must surface the injected commit failure as an error");
    // Non-vacuity: the error is the injected commit fault, not an earlier failure.
    assert!(
        err.to_string().contains("injected commit failure"),
        "the error must be the injected commit fault, got: {err}"
    );

    let after = client.load_table(&tbl_id).await?;
    assert_eq!(
        after.metadata().current_snapshot_id(),
        snapshot_id_before,
        "commit failure must NOT advance current_snapshot_id on the UPDATE path"
    );
    assert_eq!(
        after.metadata().snapshots().count(),
        snapshot_count_before,
        "commit failure must NOT create a new snapshot — staged delete + data files are orphaned"
    );

    let good_ctx2 = SessionContext::new();
    good_ctx2.register_catalog(
        "catalog",
        Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?),
    );
    let ids = select_foo1_sorted(&good_ctx2, "catalog.test_mread_upd_fault.my_table").await;
    assert_eq!(
        ids,
        vec![1, 2, 3],
        "the failed UPDATE left the table exactly as before (no torn state)"
    );
    Ok(())
}

/// `UPDATE ... WHERE nullable_col = X` must not update a row whose operand is NULL. This guards
/// the `match_mask` NULL collapse on the update path.
#[tokio::test]
async fn test_update_mread_null_predicate_three_valued_logic() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_update_mread_null".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = nullable_merge_on_read_table_creation(temp_path(), "my_table");
    iceberg_catalog.create_table(&namespace, creation).await?;
    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql("INSERT INTO catalog.test_update_mread_null.my_table VALUES (1, 'alan'), (2, NULL), (3, 'bob')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let batches = ctx
        .sql("UPDATE catalog.test_update_mread_null.my_table SET foo1 = 99 WHERE foo2 = 'alan'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let updated = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(
        updated, 1,
        "only foo2='alan' updated; the NULL-foo2 row is NOT a match (three-valued logic)"
    );

    let ids = select_foo1_sorted(&ctx, "catalog.test_update_mread_null.my_table").await;
    assert_eq!(
        ids,
        vec![2, 3, 99],
        "MoR UPDATE: the NULL-foo2 row keeps foo1=2 — NULL predicate is not an update match"
    );
    Ok(())
}

// =================================================================================================
// Non-vacuous three-valued logic, zero-file, and `(file_path, pos)` sort coverage.
//
// On a NULL operand `=` gives (valid=false, value=false), so the `is_valid` guard is redundant
// there. `<>` gives (valid=false, value=TRUE), where the guard alone keeps the NULL row out. The
// tests below use `<>`.
// =================================================================================================

/// MoR DELETE with a `<>` predicate over a NULL operand. The NULL row gives
/// (valid=false, value=TRUE), so only `mask.is_valid(row)` keeps it. Dropping that guard turns
/// this test red.
#[tokio::test]
async fn test_delete_mread_null_neq_predicate_isvalid_guard() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_del_mread_null_neq".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = nullable_merge_on_read_table_creation(temp_path(), "my_table");
    iceberg_catalog.create_table(&namespace, creation).await?;
    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql("INSERT INTO catalog.test_del_mread_null_neq.my_table VALUES (1, 'alan'), (2, NULL), (3, 'bob')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // `foo2 <> 'zzz'` is TRUE for 'alan' and 'bob', and UNKNOWN for the NULL row. The NULL row's
    // value bit is TRUE, so only the `is_valid` guard keeps it alive.
    let batches = ctx
        .sql("DELETE FROM catalog.test_del_mread_null_neq.my_table WHERE foo2 <> 'zzz'")
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
        "only the two non-NULL rows ('alan','bob') are deleted; the NULL row is UNKNOWN, not a match"
    );

    let survivors = select_foo1_sorted(&ctx, "catalog.test_del_mread_null_neq.my_table").await;
    assert_eq!(
        survivors,
        vec![2],
        "the NULL-foo2 row (foo1=2) SURVIVES a `<>` MoR DELETE — the `is_valid` guard is load-bearing"
    );
    Ok(())
}

/// MoR UPDATE with a `<>` predicate over a NULL operand. The same guard, this time in
/// `match_mask`. Dropping it turns this test red.
#[tokio::test]
async fn test_update_mread_null_neq_predicate_isvalid_guard() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_upd_mread_null_neq".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = nullable_merge_on_read_table_creation(temp_path(), "my_table");
    iceberg_catalog.create_table(&namespace, creation).await?;
    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql("INSERT INTO catalog.test_upd_mread_null_neq.my_table VALUES (1, 'alan'), (2, NULL), (3, 'bob')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // `foo2 <> 'zzz'` matches 'alan' and 'bob' (updated to foo1=99); the NULL row is UNKNOWN and must
    // NOT be updated. Its value-bit is TRUE, so only the `is_valid` guard in `match_mask` spares it.
    let batches = ctx
        .sql("UPDATE catalog.test_upd_mread_null_neq.my_table SET foo1 = 99 WHERE foo2 <> 'zzz'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let updated = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(
        updated, 2,
        "only the two non-NULL rows are updated; the NULL row is UNKNOWN, not an update match"
    );

    let ids = select_foo1_sorted(&ctx, "catalog.test_upd_mread_null_neq.my_table").await;
    assert_eq!(
        ids,
        vec![2, 99, 99],
        "the NULL-foo2 row keeps foo1=2 (NOT updated to 99) under a `<>` MoR UPDATE — `is_valid` guards it"
    );
    Ok(())
}

/// A MoR UPDATE matching no rows must add no data file and no snapshot. Disabling the
/// `updated == 0` guard in `merge_on_read_update` turns this test red.
///
/// A writer opened but never fed emits zero files on `close()`, so the underlying writer already
/// holds the zero-file guarantee. The `updated == 0` guard is what this test pins.
#[tokio::test]
async fn test_update_mread_zero_match_writes_no_file_no_snapshot() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_upd_mread_zero".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;
    let creation = get_merge_on_read_table_creation(temp_path(), "my_table")?;
    iceberg_catalog.create_table(&namespace, creation).await?;
    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    ctx.sql("INSERT INTO catalog.test_upd_mread_zero.my_table VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let tbl_id = TableIdent::new(namespace.clone(), "my_table".to_string());
    let before = client.load_table(&tbl_id).await?;
    let snapshot_id_before = before.metadata().current_snapshot_id();
    let snapshot_count_before = before.metadata().snapshots().count();

    // WHERE foo1 > 1000 matches no rows.
    let batches = ctx
        .sql("UPDATE catalog.test_upd_mread_zero.my_table SET foo2 = 'z' WHERE foo1 > 1000")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let updated = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(updated, 0, "no rows match WHERE foo1 > 1000 — zero updated");

    // No new snapshot: a zero-match UPDATE is a no-op, so no empty data file and no RowDelta commit.
    let after = client.load_table(&tbl_id).await?;
    assert_eq!(
        after.metadata().current_snapshot_id(),
        snapshot_id_before,
        "a zero-match MoR UPDATE must NOT advance the snapshot (no empty data file, no commit)"
    );
    assert_eq!(
        after.metadata().snapshots().count(),
        snapshot_count_before,
        "a zero-match MoR UPDATE must NOT create a new snapshot"
    );

    // Walk the latest snapshot's manifests: the no-op added no delete file and no data file.
    let snap = after.metadata().current_snapshot().unwrap();
    let ml = snap
        .load_manifest_list(after.file_io(), after.metadata())
        .await?;
    let mut data_files = 0usize;
    let mut delete_files = 0usize;
    for mf in ml.entries() {
        let m = mf.load_manifest(after.file_io()).await?;
        for entry in m.entries() {
            if !entry.is_alive() {
                continue;
            }
            match entry.data_file().content_type() {
                iceberg::spec::DataContentType::Data => data_files += 1,
                iceberg::spec::DataContentType::PositionDeletes
                | iceberg::spec::DataContentType::EqualityDeletes => delete_files += 1,
            }
        }
    }
    assert_eq!(
        data_files, 1,
        "exactly one data file (the seed) — the zero-match UPDATE added none"
    );
    assert_eq!(
        delete_files, 0,
        "the zero-match UPDATE added no position-delete file"
    );
    Ok(())
}

/// `isnan(col)` becomes the Iceberg `is_nan` predicate and pushes down `Inexact` into the parquet
/// `RowFilter`. When `not_nan` compiled to constant false there, `WHERE NOT isnan(x)` returned
/// zero rows: DataFusion's re-filter can mask over-inclusion, never restore an over-dropped row.
///
/// `isnan` returns the NaN row, `NOT isnan` the finite row. Iceberg keeps the NULL row, per Java
/// `NaNUtil.isNaN(null) == false`, and DataFusion's re-filter then drops it.
#[tokio::test]
async fn test_isnan_filter_pushdown_returns_correct_rows() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_isnan_pushdown".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "dbl", Type::Primitive(PrimitiveType::Double)).into(),
            NestedField::optional(3, "flt", Type::Primitive(PrimitiveType::Float)).into(),
        ])
        .build()?;
    let creation = get_table_creation(temp_path(), "my_table", Some(schema))?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    // Rows: id 1 = NaN in both float columns, id 2 = finite, id 3 = NULL.
    ctx.sql(
        "INSERT INTO catalog.test_isnan_pushdown.my_table VALUES \
         (1, CAST('NaN' AS DOUBLE), CAST('NaN' AS FLOAT)), \
         (2, 2.5, CAST(3.5 AS FLOAT)), \
         (3, NULL, NULL)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    for (filter, expected_ids) in [
        ("isnan(dbl)", vec![1]),
        ("NOT isnan(dbl)", vec![2]),
        ("isnan(flt)", vec![1]),
        ("NOT isnan(flt)", vec![2]),
    ] {
        let batches = ctx
            .sql(&format!(
                "SELECT id FROM catalog.test_isnan_pushdown.my_table WHERE {filter}"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        let mut ids: Vec<i32> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("id column is Int32")
                    .iter()
                    .map(|value| value.expect("id is a required column"))
                    .collect::<Vec<_>>()
            })
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, expected_ids, "WHERE {filter}");
    }

    Ok(())
}

/// DataFusion pushes `<>` down as an `Inexact` Iceberg `not_eq` filter. The scan keeps NULL cells
/// under it, because Java `Evaluator` makes `notEq(null, lit)` TRUE. DataFusion's re-filter then
/// drops those rows again, since `NULL <> 100.5` is SQL NULL.
///
/// So SQL consumers keep SQL semantics while library consumers get the Java contract.
/// `IS NULL OR <>` shows the NULL rows are still reachable from SQL.
#[tokio::test]
async fn test_neq_pushdown_sql_3vl_refilter_drops_null_rows() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_neq_3vl_refilter".to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "dbl", Type::Primitive(PrimitiveType::Double)).into(),
        ])
        .build()?;
    let creation = get_table_creation(temp_path(), "my_table", Some(schema))?;
    iceberg_catalog.create_table(&namespace, creation).await?;

    let client = Arc::new(iceberg_catalog);
    let catalog = Arc::new(IcebergCatalogProvider::try_new(client).await?);
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);

    // Rows: id 1 matches the literal, id 2 is NULL, id 3 differs.
    ctx.sql(
        "INSERT INTO catalog.test_neq_3vl_refilter.my_table VALUES \
         (1, 100.5), (2, NULL), (3, -7.25)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    for (filter, expected_ids) in [
        // The Iceberg scan keeps the NULL row and DataFusion's re-filter drops it again.
        ("dbl <> 100.5", vec![3]),
        ("dbl < 200.0", vec![1, 3]),
        // The NULL row remains reachable through SQL when requested explicitly.
        ("dbl IS NULL OR dbl <> 100.5", vec![2, 3]),
    ] {
        let batches = ctx
            .sql(&format!(
                "SELECT id FROM catalog.test_neq_3vl_refilter.my_table WHERE {filter}"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        let mut ids: Vec<i32> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("id column is Int32")
                    .iter()
                    .map(|value| value.expect("id is a required column"))
                    .collect::<Vec<_>>()
            })
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, expected_ids, "WHERE {filter}");
    }

    Ok(())
}

// =================================================================================================
// ENGINE_CONTRACT §5 and §8 — DML optimistic-concurrency validations and the operation id.
//
// Every test below uses one two-handle race. The DML statement's physical plan freezes a table
// handle at plan time. A second engine handle then commits and moves the catalog head. The frozen
// plan executes last, refreshes to the new head, and must run the §5 validations over the window
// between the two. A true conflict is rejected loudly and non-retryably, with Java's message.
//
// Oracles: Java `SparkWrite`, `SparkPositionDeltaWrite`, and `SparkRowLevelOperationBuilder`.
// =================================================================================================

/// A `{foo1 int, foo2 string}` table creation with caller-chosen table properties.
fn get_table_creation_with_props(
    location: impl ToString,
    name: impl ToString,
    properties: HashMap<String, String>,
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
        .properties(properties)
        .schema(schema)
        .build())
}

/// A fresh `SessionContext` over the shared catalog. Its provider loads the table at its own plan
/// time, so it sees the current head.
async fn s5_new_ctx(client: &Arc<MemoryCatalog>) -> SessionContext {
    let provider = Arc::new(
        IcebergCatalogProvider::try_new(client.clone())
            .await
            .expect("catalog provider"),
    );
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", provider);
    ctx
}

/// Creates `catalog.<namespace>.t` with `properties`, seeds it with one INSERT, and returns the
/// shared catalog and a context.
async fn s5_fixture(
    namespace: &str,
    properties: HashMap<String, String>,
    seed_values: &str,
) -> (Arc<MemoryCatalog>, SessionContext) {
    let iceberg_catalog = get_iceberg_catalog().await;
    let ns = NamespaceIdent::new(namespace.to_string());
    set_test_namespace(&iceberg_catalog, &ns)
        .await
        .expect("create namespace");
    let creation =
        get_table_creation_with_props(temp_path(), "t", properties).expect("table creation");
    iceberg_catalog
        .create_table(&ns, creation)
        .await
        .expect("create table");
    let client = Arc::new(iceberg_catalog);
    let ctx = s5_new_ctx(&client).await;
    ctx.sql(&format!(
        "INSERT INTO catalog.{namespace}.t VALUES {seed_values}"
    ))
    .await
    .expect("seed insert plan")
    .collect()
    .await
    .expect("seed insert");
    (client, ctx)
}

/// Build the DML statement's PHYSICAL plan — the provider loads (freezes) its table handle here.
async fn s5_freeze_plan(
    ctx: &SessionContext,
    sql: &str,
) -> Arc<dyn datafusion::physical_plan::ExecutionPlan> {
    ctx.sql(sql)
        .await
        .expect("logical plan")
        .create_physical_plan()
        .await
        .expect("physical plan (freezes the table handle)")
}

/// Execute a frozen DML plan; `Ok(row_count)` from the count batch, or the loud commit error.
async fn s5_execute_frozen(
    ctx: &SessionContext,
    plan: Arc<dyn datafusion::physical_plan::ExecutionPlan>,
) -> std::result::Result<u64, datafusion::error::DataFusionError> {
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;
    Ok(batches
        .first()
        .and_then(|batch| batch.column(0).as_any().downcast_ref::<UInt64Array>())
        .map_or(0, |arr| if arr.is_empty() { 0 } else { arr.value(0) }))
}

/// All rows of `catalog.<namespace>.t` as sorted `(foo1, foo2)` pairs, read through a FRESH handle.
async fn s5_table_rows(client: &Arc<MemoryCatalog>, namespace: &str) -> Vec<(i32, String)> {
    let ctx = s5_new_ctx(client).await;
    let batches = ctx
        .sql(&format!("SELECT * FROM catalog.{namespace}.t"))
        .await
        .expect("select plan")
        .collect()
        .await
        .expect("select rows");
    let mut rows = Vec::new();
    for batch in &batches {
        let foo1 = batch
            .column_by_name("foo1")
            .and_then(|col| col.as_any().downcast_ref::<Int32Array>().cloned())
            .expect("foo1 column");
        let foo2 = batch
            .column_by_name("foo2")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>().cloned())
            .expect("foo2 column");
        for row in 0..batch.num_rows() {
            rows.push((foo1.value(row), foo2.value(row).to_string()));
        }
    }
    rows.sort();
    rows
}

/// The `engine.operation-id` stamp of every snapshot of `catalog.<namespace>.t` (one entry per
/// snapshot; `None` for an unstamped snapshot), plus the current head's stamp.
async fn s5_operation_ids(
    client: &Arc<MemoryCatalog>,
    namespace: &str,
) -> (Vec<Option<String>>, Option<String>) {
    let table = client
        .load_table(&TableIdent::from_strs([namespace, "t"]).expect("table ident"))
        .await
        .expect("load table");
    let all: Vec<Option<String>> = table
        .metadata()
        .snapshots()
        .map(|snapshot| {
            snapshot
                .summary()
                .additional_properties
                .get("engine.operation-id")
                .cloned()
        })
        .collect();
    let head = table.metadata().current_snapshot().and_then(|snapshot| {
        snapshot
            .summary()
            .additional_properties
            .get("engine.operation-id")
            .cloned()
    });
    (all, head)
}

/// The live DATA-file paths of `catalog.<namespace>.t`'s current snapshot (manifest walk through a
/// fresh handle) — used to name the file a rejection message must cite.
async fn s5_live_data_paths(client: &Arc<MemoryCatalog>, namespace: &str) -> Vec<String> {
    let table = client
        .load_table(&TableIdent::from_strs([namespace, "t"]).expect("table ident"))
        .await
        .expect("load table");
    let metadata = table.metadata();
    let mut paths = Vec::new();
    if let Some(snapshot) = metadata.current_snapshot() {
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), metadata)
            .await
            .expect("load manifest list");
        for manifest_entry in manifest_list.entries() {
            if manifest_entry.content != iceberg::spec::ManifestContentType::Data {
                continue;
            }
            let manifest = manifest_entry
                .load_manifest(table.file_io())
                .await
                .expect("load manifest");
            for entry in manifest.entries() {
                if entry.is_alive() {
                    paths.push(entry.file_path().to_string());
                }
            }
        }
    }
    paths.sort();
    paths
}

/// A manifest-only position-delete file for an unpartitioned table, used where a real
/// merge-on-read DELETE is unavailable. It carries no metrics, so it applies to every data file.
///
/// # Notes
///
/// Never select the table's content after committing one: the parquet is not on disk.
fn s5_synthetic_position_delete(path: &str) -> iceberg::spec::DataFile {
    iceberg::spec::DataFileBuilder::default()
        .content(iceberg::spec::DataContentType::PositionDeletes)
        .file_path(path.to_string())
        .file_format(iceberg::spec::DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(0)
        .partition(iceberg::spec::Struct::empty())
        .build()
        .expect("synthetic position-delete file")
}

/// Commit a synthetic position-delete against the CURRENT head through a fresh handle (a real
/// concurrent `RowDelta` recording `Operation::Delete`).
async fn s5_commit_synthetic_delete(client: &Arc<MemoryCatalog>, namespace: &str, path: &str) {
    use iceberg::transaction::{ApplyTransactionAction, Transaction};
    let table = client
        .load_table(&TableIdent::from_strs([namespace, "t"]).expect("table ident"))
        .await
        .expect("load table");
    let tx = Transaction::new(&table);
    let action = tx
        .row_delta()
        .add_deletes(vec![s5_synthetic_position_delete(path)]);
    let tx = action.apply(tx).expect("apply row delta");
    tx.commit(client.as_ref() as &dyn Catalog)
        .await
        .expect("concurrent synthetic delete commit");
}

/// CoW DELETE at the serializable default, against a concurrent append. Java
/// `SparkWrite.commitWithSerializableIsolation` rejects it through `validate_no_conflicting_data`.
/// The table must not change. It used to commit over the concurrent writer.
#[tokio::test]
async fn test_s5_cow_delete_serializable_default_rejects_concurrent_append()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let (client, ctx) = s5_fixture("s5_cow_del_ser", HashMap::new(), "(1, 'a'), (2, 'b')").await;

    let plan = s5_freeze_plan(&ctx, "DELETE FROM catalog.s5_cow_del_ser.t WHERE foo1 = 1").await;

    // Under serializable, a concurrent insert matching the AlwaysTrue conflict filter breaks the
    // isolation contract.
    let ctx2 = s5_new_ctx(&client).await;
    ctx2.sql("INSERT INTO catalog.s5_cow_del_ser.t VALUES (3, 'c')")
        .await?
        .collect()
        .await?;

    let err = s5_execute_frozen(&ctx, plan)
        .await
        .expect_err("serializable CoW DELETE must reject the concurrent append");
    assert!(
        err.to_string()
            .contains("Found conflicting files that can contain records matching"),
        "must carry Java's conflicting-data ValidationException message, got: {err}"
    );

    // The delete must NOT have applied: base rows AND the concurrent row are all present.
    let rows = s5_table_rows(&client, "s5_cow_del_ser").await;
    assert_eq!(
        rows,
        vec![
            (1, "a".to_string()),
            (2, "b".to_string()),
            (3, "c".to_string())
        ],
        "rejected DELETE must leave the table exactly as the concurrent writer left it"
    );
    Ok(())
}

/// CoW DELETE at snapshot isolation, against a concurrent position-delete file on the same data
/// file. Java `SparkWrite.commitWithSnapshotIsolation` rejects it through
/// `validate_no_conflicting_deletes`, naming the file.
///
/// The check works only because the removals carry full `DataFile` metadata. Path-only removals
/// made it inert.
#[tokio::test]
async fn test_s5_cow_delete_snapshot_rejects_concurrent_delete_on_affected_file()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let props = HashMap::from([(
        "write.delete.isolation-level".to_string(),
        "snapshot".to_string(),
    )]);
    let (client, ctx) = s5_fixture("s5_cow_del_snap_del", props, "(1, 'a'), (2, 'b')").await;
    let seeded_paths = s5_live_data_paths(&client, "s5_cow_del_snap_del").await;
    assert_eq!(seeded_paths.len(), 1, "seed must land in exactly one file");

    let plan = s5_freeze_plan(
        &ctx,
        "DELETE FROM catalog.s5_cow_del_snap_del.t WHERE foo1 = 1",
    )
    .await;

    // The concurrent delete lands a position-delete file on the data file the frozen CoW DELETE
    // is about to remove.
    s5_commit_synthetic_delete(
        &client,
        "s5_cow_del_snap_del",
        "mem/s5-concurrent-pos-del.parquet",
    )
    .await;

    let err = s5_execute_frozen(&ctx, plan)
        .await
        .expect_err("snapshot CoW DELETE must reject dropping a file under a concurrent delete");
    let message = err.to_string();
    assert!(
        message.contains("found new delete for replaced data file"),
        "must carry Java's conflicting-deletes message, got: {message}"
    );
    assert!(
        message.contains(&seeded_paths[0]),
        "must NAME the replaced data file {}, got: {message}",
        seeded_paths[0]
    );

    // Metadata only, because the synthetic parquet is not on disk. Two snapshots must remain.
    let table = client
        .load_table(&TableIdent::from_strs(["s5_cow_del_snap_del", "t"]).unwrap())
        .await?;
    assert_eq!(
        table.metadata().snapshots().len(),
        2,
        "rejected CoW DELETE must not commit a snapshot"
    );
    Ok(())
}

/// False-positive guard: CoW DELETE at snapshot isolation, against a concurrent append, commits.
/// Snapshot isolation checks deletes only. The deleted row goes and the concurrent row stays.
#[tokio::test]
async fn test_s5_cow_delete_snapshot_allows_concurrent_append()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let props = HashMap::from([(
        "write.delete.isolation-level".to_string(),
        "snapshot".to_string(),
    )]);
    let (client, ctx) = s5_fixture("s5_cow_del_snap_ok", props, "(1, 'a'), (2, 'b')").await;

    let plan = s5_freeze_plan(
        &ctx,
        "DELETE FROM catalog.s5_cow_del_snap_ok.t WHERE foo1 = 1",
    )
    .await;

    let ctx2 = s5_new_ctx(&client).await;
    ctx2.sql("INSERT INTO catalog.s5_cow_del_snap_ok.t VALUES (3, 'c')")
        .await?
        .collect()
        .await?;

    let deleted = s5_execute_frozen(&ctx, plan)
        .await
        .expect("snapshot CoW DELETE must tolerate a non-conflicting concurrent append");
    assert_eq!(deleted, 1, "exactly the one matching row is deleted");

    let rows = s5_table_rows(&client, "s5_cow_del_snap_ok").await;
    assert_eq!(
        rows,
        vec![(2, "b".to_string()), (3, "c".to_string())],
        "deleted row gone; concurrent append survives"
    );
    Ok(())
}

/// CoW UPDATE at the serializable default, against a concurrent append, is rejected. Java's
/// isolation switch does not branch on the command, so UPDATE follows the DELETE recipe.
#[tokio::test]
async fn test_s5_cow_update_serializable_default_rejects_concurrent_append()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let (client, ctx) = s5_fixture("s5_cow_upd_ser", HashMap::new(), "(1, 'a'), (2, 'b')").await;

    let plan = s5_freeze_plan(
        &ctx,
        "UPDATE catalog.s5_cow_upd_ser.t SET foo2 = 'x' WHERE foo1 = 1",
    )
    .await;

    let ctx2 = s5_new_ctx(&client).await;
    ctx2.sql("INSERT INTO catalog.s5_cow_upd_ser.t VALUES (3, 'c')")
        .await?
        .collect()
        .await?;

    let err = s5_execute_frozen(&ctx, plan)
        .await
        .expect_err("serializable CoW UPDATE must reject the concurrent append");
    assert!(
        err.to_string()
            .contains("Found conflicting files that can contain records matching"),
        "must carry Java's conflicting-data message, got: {err}"
    );

    let rows = s5_table_rows(&client, "s5_cow_upd_ser").await;
    assert_eq!(
        rows,
        vec![
            (1, "a".to_string()),
            (2, "b".to_string()),
            (3, "c".to_string())
        ],
        "rejected UPDATE must not have changed any row"
    );
    Ok(())
}

/// False-positive guard: CoW UPDATE at snapshot isolation, against a concurrent append, commits.
/// The matched row takes the new value and the concurrent row survives.
#[tokio::test]
async fn test_s5_cow_update_snapshot_allows_concurrent_append()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let props = HashMap::from([(
        "write.update.isolation-level".to_string(),
        "snapshot".to_string(),
    )]);
    let (client, ctx) = s5_fixture("s5_cow_upd_snap_ok", props, "(1, 'a'), (2, 'b')").await;

    let plan = s5_freeze_plan(
        &ctx,
        "UPDATE catalog.s5_cow_upd_snap_ok.t SET foo2 = 'x' WHERE foo1 = 1",
    )
    .await;

    let ctx2 = s5_new_ctx(&client).await;
    ctx2.sql("INSERT INTO catalog.s5_cow_upd_snap_ok.t VALUES (3, 'c')")
        .await?
        .collect()
        .await?;

    let updated = s5_execute_frozen(&ctx, plan)
        .await
        .expect("snapshot CoW UPDATE must tolerate a non-conflicting concurrent append");
    assert_eq!(updated, 1);

    let rows = s5_table_rows(&client, "s5_cow_upd_snap_ok").await;
    assert_eq!(
        rows,
        vec![
            (1, "x".to_string()),
            (2, "b".to_string()),
            (3, "c".to_string())
        ],
        "updated value applied; concurrent append survives"
    );
    Ok(())
}

/// MoR DELETE against a concurrent copy-on-write rewrite of the data file its position deletes
/// reference. Java `SparkPositionDeltaWrite.commit` rejects it through `validate_data_files_exist`,
/// which is unconditional.
///
/// The rewrite is a real CoW UPDATE, so it removes the referenced file. Isolation stays at
/// snapshot, so the files-exist check is what rejects, not the serializable data check.
#[tokio::test]
async fn test_s5_merge_on_read_delete_rejects_concurrent_rewrite_of_target_file()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let props = HashMap::from([
        ("write.delete.mode".to_string(), "merge-on-read".to_string()),
        (
            "write.delete.isolation-level".to_string(),
            "snapshot".to_string(),
        ),
    ]);
    let (client, ctx) = s5_fixture("s5_mr_del_rewrite", props, "(1, 'a'), (2, 'b')").await;

    let plan = s5_freeze_plan(
        &ctx,
        "DELETE FROM catalog.s5_mr_del_rewrite.t WHERE foo1 = 1",
    )
    .await;

    // Concurrent CoW UPDATE rewrites the single seeded data file (removes it, adds a rewritten one).
    let ctx2 = s5_new_ctx(&client).await;
    ctx2.sql("UPDATE catalog.s5_mr_del_rewrite.t SET foo2 = 'rewritten' WHERE foo1 = 2")
        .await?
        .collect()
        .await?;

    let err = s5_execute_frozen(&ctx, plan)
        .await
        .expect_err("MoR DELETE must reject: its position deletes reference a rewritten-away file");
    assert!(
        err.to_string()
            .contains("Cannot commit, missing data files"),
        "must carry Java's files-exist ValidationException message, got: {err}"
    );

    // The rewrite stands; the delete did not apply (row 1 survives with its original value).
    let rows = s5_table_rows(&client, "s5_mr_del_rewrite").await;
    assert_eq!(
        rows,
        vec![(1, "a".to_string()), (2, "rewritten".to_string())],
        "concurrent rewrite must stand; the rejected MoR DELETE must not have applied"
    );
    Ok(())
}

/// False-positive guard: MoR DELETE at snapshot isolation, against a concurrent append, commits.
/// The position deletes do not reference the appended file, and the `RowDelta` reads back.
#[tokio::test]
async fn test_s5_merge_on_read_delete_snapshot_allows_concurrent_append()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let props = HashMap::from([
        ("write.delete.mode".to_string(), "merge-on-read".to_string()),
        (
            "write.delete.isolation-level".to_string(),
            "snapshot".to_string(),
        ),
    ]);
    let (client, ctx) = s5_fixture("s5_mr_del_snap_ok", props, "(1, 'a'), (2, 'b')").await;

    let plan = s5_freeze_plan(
        &ctx,
        "DELETE FROM catalog.s5_mr_del_snap_ok.t WHERE foo1 = 1",
    )
    .await;

    let ctx2 = s5_new_ctx(&client).await;
    ctx2.sql("INSERT INTO catalog.s5_mr_del_snap_ok.t VALUES (3, 'c')")
        .await?
        .collect()
        .await?;

    let deleted = s5_execute_frozen(&ctx, plan)
        .await
        .expect("snapshot MoR DELETE must tolerate a non-conflicting concurrent append");
    assert_eq!(deleted, 1);

    let rows = s5_table_rows(&client, "s5_mr_del_snap_ok").await;
    assert_eq!(
        rows,
        vec![(2, "b".to_string()), (3, "c".to_string())],
        "position delete applied on read; concurrent append survives"
    );
    Ok(())
}

/// MoR DELETE at the serializable default, against a concurrent append, is rejected through
/// `validate_no_conflicting_data_files`. The per-operation default governs the MoR path too.
#[tokio::test]
async fn test_s5_merge_on_read_delete_serializable_default_rejects_concurrent_append()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let props = HashMap::from([("write.delete.mode".to_string(), "merge-on-read".to_string())]);
    let (client, ctx) = s5_fixture("s5_mr_del_ser", props, "(1, 'a'), (2, 'b')").await;

    let plan = s5_freeze_plan(&ctx, "DELETE FROM catalog.s5_mr_del_ser.t WHERE foo1 = 1").await;

    let ctx2 = s5_new_ctx(&client).await;
    ctx2.sql("INSERT INTO catalog.s5_mr_del_ser.t VALUES (3, 'c')")
        .await?
        .collect()
        .await?;

    let err = s5_execute_frozen(&ctx, plan)
        .await
        .expect_err("serializable (default) MoR DELETE must reject the concurrent append");
    assert!(
        err.to_string()
            .contains("Found conflicting files that can contain records matching"),
        "must carry Java's conflicting-data message, got: {err}"
    );

    let rows = s5_table_rows(&client, "s5_mr_del_ser").await;
    assert_eq!(
        rows,
        vec![
            (1, "a".to_string()),
            (2, "b".to_string()),
            (3, "c".to_string())
        ],
        "rejected MoR DELETE must not have applied"
    );
    Ok(())
}

/// The UPDATE-only §5 arms, `validate_deleted_files` and `validate_no_conflicting_delete_files`,
/// which Java runs for UPDATE and MERGE. A MoR UPDATE against a concurrent MoR DELETE of the rows
/// it read is rejected, even at snapshot isolation: the arm is isolation-independent.
#[tokio::test]
async fn test_s5_merge_on_read_update_rejects_concurrent_delete_of_read_rows()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let props = HashMap::from([
        ("write.delete.mode".to_string(), "merge-on-read".to_string()),
        ("write.update.mode".to_string(), "merge-on-read".to_string()),
        (
            "write.update.isolation-level".to_string(),
            "snapshot".to_string(),
        ),
    ]);
    let (client, ctx) = s5_fixture("s5_mr_upd_del", props, "(1, 'a'), (2, 'b')").await;

    let plan = s5_freeze_plan(
        &ctx,
        "UPDATE catalog.s5_mr_upd_del.t SET foo2 = 'x' WHERE foo1 = 1",
    )
    .await;

    // Concurrent REAL merge-on-read DELETE of the same row lands a position-delete file.
    let ctx2 = s5_new_ctx(&client).await;
    ctx2.sql("DELETE FROM catalog.s5_mr_upd_del.t WHERE foo1 = 1")
        .await?
        .collect()
        .await?;

    let err = s5_execute_frozen(&ctx, plan)
        .await
        .expect_err("MoR UPDATE must reject: a concurrent delete removed rows it read");
    assert!(
        err.to_string()
            .contains("Found new conflicting delete files that can apply to records matching"),
        "must carry Java's conflicting-delete-files message, got: {err}"
    );

    // The concurrent delete stands; the update did not apply.
    let rows = s5_table_rows(&client, "s5_mr_upd_del").await;
    assert_eq!(
        rows,
        vec![(2, "b".to_string())],
        "concurrent delete stands; rejected UPDATE must not have applied"
    );
    Ok(())
}

/// The DELETE and UPDATE asymmetry: two MoR deletes of the same rows commit. Java arms the
/// delete-conflict checks for UPDATE and MERGE only, and the files-exist check skips
/// `Operation::Delete` snapshots by default. Two deletes of one row are idempotent.
#[tokio::test]
async fn test_s5_merge_on_read_delete_tolerates_concurrent_delete_same_rows()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let props = HashMap::from([
        ("write.delete.mode".to_string(), "merge-on-read".to_string()),
        (
            "write.delete.isolation-level".to_string(),
            "snapshot".to_string(),
        ),
    ]);
    let (client, ctx) = s5_fixture("s5_mr_del_del", props, "(1, 'a'), (2, 'b')").await;

    let plan = s5_freeze_plan(&ctx, "DELETE FROM catalog.s5_mr_del_del.t WHERE foo1 = 1").await;

    let ctx2 = s5_new_ctx(&client).await;
    ctx2.sql("DELETE FROM catalog.s5_mr_del_del.t WHERE foo1 = 1")
        .await?
        .collect()
        .await?;

    let deleted = s5_execute_frozen(&ctx, plan)
        .await
        .expect("a MoR DELETE must tolerate a concurrent DELETE of the same rows (Java asymmetry)");
    assert_eq!(
        deleted, 1,
        "the frozen handle still saw the row at scan time"
    );

    let rows = s5_table_rows(&client, "s5_mr_del_del").await;
    assert_eq!(
        rows,
        vec![(2, "b".to_string())],
        "both deletes applied idempotently — the row is gone once"
    );
    Ok(())
}

/// Every INSERT commit stamps `engine.operation-id` into the snapshot summary. That marker is
/// what reconciles an ambiguous commit outcome. A bare `fast_append` left a retry unreconcilable.
#[tokio::test]
async fn test_s8_insert_stamps_operation_id() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    let (client, _ctx) = s5_fixture("s8_insert_stamp", HashMap::new(), "(1, 'a'), (2, 'b')").await;

    let (all_ids, head_id) = s5_operation_ids(&client, "s8_insert_stamp").await;
    assert_eq!(all_ids.len(), 1, "one snapshot (the seed INSERT)");
    let id = head_id.expect("the INSERT snapshot must carry engine.operation-id");
    assert!(
        uuid::Uuid::parse_str(&id).is_ok(),
        "the stamp must be a parseable UUID, got: {id}"
    );
    Ok(())
}

/// An INSERT that must refresh and re-apply over a concurrent commit lands exactly one snapshot
/// carrying its operation id. The re-applied attempt reuses that id, rows are not duplicated, and
/// ids stay unique per statement, so a reconciler cannot false-match another operation.
#[tokio::test]
async fn test_s8_insert_retry_single_stamp_no_duplicate()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let (client, ctx) = s5_fixture("s8_insert_retry", HashMap::new(), "(1, 'a'), (2, 'b')").await;

    // Freeze the INSERT's physical plan against S1.
    let plan = s5_freeze_plan(
        &ctx,
        "INSERT INTO catalog.s8_insert_retry.t VALUES (5, 'e')",
    )
    .await;

    // Move the head with a concurrent INSERT.
    let ctx2 = s5_new_ctx(&client).await;
    ctx2.sql("INSERT INTO catalog.s8_insert_retry.t VALUES (3, 'c')")
        .await?
        .collect()
        .await?;

    // The frozen commit then refreshes to S2 and re-applies, landing S3.
    let written = s5_execute_frozen(&ctx, plan)
        .await
        .expect("append never conflicts — the refresh-re-apply must land");
    assert_eq!(written, 1);

    let rows = s5_table_rows(&client, "s8_insert_retry").await;
    assert_eq!(
        rows,
        vec![
            (1, "a".to_string()),
            (2, "b".to_string()),
            (3, "c".to_string()),
            (5, "e".to_string())
        ],
        "no duplicated rows across the refresh-re-apply"
    );

    let (all_ids, head_id) = s5_operation_ids(&client, "s8_insert_retry").await;
    assert_eq!(all_ids.len(), 3, "seed + concurrent + frozen = 3 snapshots");
    let mut ids: Vec<String> = all_ids
        .into_iter()
        .map(|id| id.expect("every INSERT snapshot must be stamped"))
        .collect();
    let head = head_id.expect("head (the frozen INSERT) must be stamped");
    assert_eq!(
        ids.iter().filter(|id| **id == head).count(),
        1,
        "the frozen INSERT's id must appear in EXACTLY one snapshot (no double-stamp on re-apply)"
    );
    let before = ids.len();
    ids.sort();
    ids.dedup();
    assert_eq!(
        ids.len(),
        before,
        "operation ids must be pairwise distinct across statements"
    );
    for id in &ids {
        assert!(
            uuid::Uuid::parse_str(id).is_ok(),
            "every stamp must be a parseable UUID, got: {id}"
        );
    }
    Ok(())
}

/// INSERT OVERWRITE at the engine default of snapshot isolation, against a concurrent MoR DELETE.
/// Java `SparkWrite.OverwriteByFilter.commit` rejects it through `validate_no_conflicting_deletes`,
/// with the row filter as the conflict filter. A bare `overwrite_by_row_filter(AlwaysTrue)`
/// silently discarded the concurrent deleter's intent.
#[tokio::test]
async fn test_s5_insert_overwrite_default_rejects_concurrent_delete()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let props = HashMap::from([("write.delete.mode".to_string(), "merge-on-read".to_string())]);
    let (client, ctx) = s5_fixture("s5_ow_del", props, "(1, 'a'), (2, 'b')").await;

    let plan = s5_freeze_plan(&ctx, "INSERT OVERWRITE catalog.s5_ow_del.t VALUES (9, 'z')").await;

    let ctx2 = s5_new_ctx(&client).await;
    ctx2.sql("DELETE FROM catalog.s5_ow_del.t WHERE foo1 = 1")
        .await?
        .collect()
        .await?;

    let err = s5_execute_frozen(&ctx, plan)
        .await
        .expect_err("default INSERT OVERWRITE must reject the concurrent delete-file add");
    assert!(
        err.to_string()
            .contains("Found new conflicting delete files that can apply to records matching"),
        "must carry Java's conflicting-delete-files message, got: {err}"
    );

    let rows = s5_table_rows(&client, "s5_ow_del").await;
    assert_eq!(
        rows,
        vec![(2, "b".to_string())],
        "concurrent delete stands; the rejected overwrite must not have replaced the table"
    );
    Ok(())
}

/// INSERT OVERWRITE at serializable isolation, against a concurrent append, is rejected through
/// `validate_no_conflicting_data`. The row filter is the conflict filter, so any insert conflicts.
#[tokio::test]
async fn test_s5_insert_overwrite_serializable_rejects_concurrent_append()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let props = HashMap::from([(
        "write.overwrite.isolation-level".to_string(),
        "serializable".to_string(),
    )]);
    let (client, ctx) = s5_fixture("s5_ow_ser", props, "(1, 'a'), (2, 'b')").await;

    let plan = s5_freeze_plan(&ctx, "INSERT OVERWRITE catalog.s5_ow_ser.t VALUES (9, 'z')").await;

    let ctx2 = s5_new_ctx(&client).await;
    ctx2.sql("INSERT INTO catalog.s5_ow_ser.t VALUES (3, 'c')")
        .await?
        .collect()
        .await?;

    let err = s5_execute_frozen(&ctx, plan)
        .await
        .expect_err("serializable INSERT OVERWRITE must reject the concurrent append");
    assert!(
        err.to_string()
            .contains("Found conflicting files that can contain records matching"),
        "must carry Java's conflicting-data message, got: {err}"
    );

    let rows = s5_table_rows(&client, "s5_ow_ser").await;
    assert_eq!(
        rows,
        vec![
            (1, "a".to_string()),
            (2, "b".to_string()),
            (3, "c".to_string())
        ],
        "rejected overwrite must leave the concurrent writer's state intact"
    );
    Ok(())
}

/// False-positive guard at snapshot isolation: INSERT OVERWRITE against a concurrent append
/// commits, and the table holds exactly the overwrite's rows. The refreshed re-apply replaces the
/// append too, which is Spark's static-overwrite behavior.
#[tokio::test]
async fn test_s5_insert_overwrite_snapshot_allows_concurrent_append_and_replaces_it()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let (client, ctx) = s5_fixture("s5_ow_snap_ok", HashMap::new(), "(1, 'a'), (2, 'b')").await;

    let plan = s5_freeze_plan(
        &ctx,
        "INSERT OVERWRITE catalog.s5_ow_snap_ok.t VALUES (9, 'z')",
    )
    .await;

    let ctx2 = s5_new_ctx(&client).await;
    ctx2.sql("INSERT INTO catalog.s5_ow_snap_ok.t VALUES (3, 'c')")
        .await?
        .collect()
        .await?;

    let written = s5_execute_frozen(&ctx, plan)
        .await
        .expect("snapshot-level INSERT OVERWRITE must tolerate a concurrent append");
    assert_eq!(written, 1);

    let rows = s5_table_rows(&client, "s5_ow_snap_ok").await;
    assert_eq!(
        rows,
        vec![(9, "z".to_string())],
        "INSERT OVERWRITE replaces ALL data — including the concurrent append"
    );

    // The overwrite snapshot is stamped too (§8 applies to both IcebergCommitExec arms).
    let (_, head_id) = s5_operation_ids(&client, "s5_ow_snap_ok").await;
    let id = head_id.expect("the overwrite snapshot must carry engine.operation-id");
    assert!(uuid::Uuid::parse_str(&id).is_ok());
    Ok(())
}

/// The `none` escape hatch restores Spark's unvalidated default: Java runs no validations on
/// `SparkWrite.OverwriteByFilter.commit` unless a per-write option asks for them. The
/// concurrent-delete race rejected above now commits,
/// and the overwrite replaces the table.
#[tokio::test]
async fn test_s5_insert_overwrite_none_restores_unvalidated_java_default()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let props = HashMap::from([
        ("write.delete.mode".to_string(), "merge-on-read".to_string()),
        (
            "write.overwrite.isolation-level".to_string(),
            "none".to_string(),
        ),
    ]);
    let (client, ctx) = s5_fixture("s5_ow_none", props, "(1, 'a'), (2, 'b')").await;

    let plan = s5_freeze_plan(
        &ctx,
        "INSERT OVERWRITE catalog.s5_ow_none.t VALUES (9, 'z')",
    )
    .await;

    let ctx2 = s5_new_ctx(&client).await;
    ctx2.sql("DELETE FROM catalog.s5_ow_none.t WHERE foo1 = 1")
        .await?
        .collect()
        .await?;

    let written = s5_execute_frozen(&ctx, plan)
        .await
        .expect("'none' must restore the unvalidated Java-default overwrite");
    assert_eq!(written, 1);

    let rows = s5_table_rows(&client, "s5_ow_none").await;
    assert_eq!(
        rows,
        vec![(9, "z".to_string())],
        "unvalidated overwrite replaces the table (the dangling concurrent delete is inert)"
    );
    Ok(())
}

/// P14b — an invalid `write.delete.isolation-level` fails LOUD at PLAN time (Java resolves the
/// row-level isolation in the operation-builder constructor and `IsolationLevel.fromName` throws
/// `"Invalid isolation level: %s"`) — never a silent default.
#[tokio::test]
async fn test_s5_invalid_isolation_property_fails_plan_loud()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let props = HashMap::from([(
        "write.delete.isolation-level".to_string(),
        "read-committed".to_string(),
    )]);
    let (_client, ctx) = s5_fixture("s5_bad_iso", props, "(1, 'a')").await;

    let err = ctx
        .sql("DELETE FROM catalog.s5_bad_iso.t WHERE foo1 = 1")
        .await?
        .create_physical_plan()
        .await
        .expect_err("an invalid isolation level must fail the statement at plan time");
    assert!(
        err.to_string()
            .contains("Invalid isolation level: read-committed"),
        "must carry Java's fromName message + the offending value, got: {err}"
    );
    Ok(())
}

/// P14c — an invalid `write.overwrite.isolation-level` fails INSERT OVERWRITE loud, while a plain
/// INSERT INTO on the same table is UNAFFECTED (the property is consulted only by the Overwrite
/// arm, mirroring Java's per-operation option scoping).
#[tokio::test]
async fn test_s5_invalid_overwrite_isolation_only_gates_overwrite()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let props = HashMap::from([(
        "write.overwrite.isolation-level".to_string(),
        "read-committed".to_string(),
    )]);
    let (client, ctx) = s5_fixture("s5_bad_ow_iso", props, "(1, 'a')").await;

    // Plain INSERT INTO ignores the overwrite property entirely.
    ctx.sql("INSERT INTO catalog.s5_bad_ow_iso.t VALUES (2, 'b')")
        .await?
        .collect()
        .await
        .expect("INSERT INTO must not consult the overwrite isolation property");

    // INSERT OVERWRITE consults it and must fail loud.
    let plan = s5_freeze_plan(
        &ctx,
        "INSERT OVERWRITE catalog.s5_bad_ow_iso.t VALUES (9, 'z')",
    )
    .await;
    let err = s5_execute_frozen(&ctx, plan)
        .await
        .expect_err("an invalid overwrite isolation level must fail the statement");
    assert!(
        err.to_string()
            .contains("Invalid isolation level: read-committed"),
        "must carry Java's fromName message + the offending value, got: {err}"
    );

    let rows = s5_table_rows(&client, "s5_bad_ow_iso").await;
    assert_eq!(
        rows,
        vec![(1, "a".to_string()), (2, "b".to_string())],
        "the failed overwrite must not have replaced anything"
    );
    Ok(())
}

/// A V3 `{id int, val string}` table with both row-level modes set to merge-on-read.
async fn make_v3_mread_ctx(ns: &str, tbl: &str) -> Result<(SessionContext, Arc<MemoryCatalog>)> {
    make_versioned_mread_ctx(ns, tbl, iceberg::spec::FormatVersion::V3).await
}

async fn make_versioned_mread_ctx(
    ns: &str,
    tbl: &str,
    format_version: iceberg::spec::FormatVersion,
) -> Result<(SessionContext, Arc<MemoryCatalog>)> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new(ns.to_string());
    set_test_namespace(&iceberg_catalog, &namespace).await?;

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "val", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()?;

    let creation = TableCreation::builder()
        .name(tbl.to_string())
        .location(temp_path())
        .schema(schema)
        .format_version(format_version)
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

/// Every live delete file of the table's current snapshot.
async fn live_delete_files(
    client: &MemoryCatalog,
    ns: &str,
    tbl: &str,
) -> Result<Vec<iceberg::spec::DataFile>> {
    let table_ident =
        iceberg::TableIdent::new(NamespaceIdent::new(ns.to_string()), tbl.to_string());
    let table = client.load_table(&table_ident).await?;
    let mut delete_files = Vec::new();
    let Some(snapshot) = table.metadata().current_snapshot() else {
        return Ok(delete_files);
    };
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await?;
    for manifest_entry in manifest_list.entries() {
        if manifest_entry.content != iceberg::spec::ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_entry.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if entry.is_alive() {
                delete_files.push(entry.data_file().clone());
            }
        }
    }
    Ok(delete_files)
}

async fn delete_count(ctx: &SessionContext, sql: &str) -> u64 {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0)
}

async fn surviving_ids(ctx: &SessionContext, sql: &str) -> Vec<i32> {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let mut ids = Vec::new();
    for batch in &batches {
        let column = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        ids.extend((0..batch.num_rows()).map(|row| column.value(row)));
    }
    ids
}

/// Risk pinned: a V3 merge-on-read DELETE writing a Parquet position-delete file. The V3 spec
/// FORBIDS new position-delete files, so a table written that way is invalid to a Java reader.
#[tokio::test]
async fn test_delete_mread_v3_writes_a_deletion_vector() -> Result<()> {
    let (ctx, client) = make_v3_mread_ctx("test_del_mread_v3", "items").await?;
    ctx.sql("INSERT INTO catalog.test_del_mread_v3.items VALUES (1,'a'),(2,'b'),(3,'c'),(4,'d')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let deleted = delete_count(
        &ctx,
        "DELETE FROM catalog.test_del_mread_v3.items WHERE id = 2",
    )
    .await;
    assert_eq!(deleted, 1, "exactly one row matches id = 2");

    let ids = surviving_ids(
        &ctx,
        "SELECT id FROM catalog.test_del_mread_v3.items ORDER BY id",
    )
    .await;
    assert_eq!(ids, vec![1, 3, 4], "the V3 DV must hide exactly row id = 2");

    let delete_files = live_delete_files(&client, "test_del_mread_v3", "items").await?;
    assert_eq!(delete_files.len(), 1, "one DV per touched data file");
    let dv = &delete_files[0];
    assert_eq!(
        dv.file_format(),
        iceberg::spec::DataFileFormat::Puffin,
        "V3 forbids new position-delete files, so this must be a Puffin DV"
    );
    assert!(
        dv.referenced_data_file().is_some(),
        "a DV is file-scoped and names the data file it covers"
    );
    assert_eq!(dv.record_count(), 1, "the DV carries the one deleted row");
    Ok(())
}

/// Risk pinned: a second DELETE on a data file that already has a DV leaving BOTH live. V3 allows
/// at most one DV per data file, and two live DVs over one file double-count its positions.
#[tokio::test]
async fn test_delete_mread_v3_second_delete_supersedes_the_first_dv() -> Result<()> {
    let (ctx, client) = make_v3_mread_ctx("test_del_mread_v3_merge", "items").await?;
    ctx.sql(
        "INSERT INTO catalog.test_del_mread_v3_merge.items VALUES (1,'a'),(2,'b'),(3,'c'),(4,'d')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    delete_count(
        &ctx,
        "DELETE FROM catalog.test_del_mread_v3_merge.items WHERE id = 2",
    )
    .await;
    delete_count(
        &ctx,
        "DELETE FROM catalog.test_del_mread_v3_merge.items WHERE id = 3",
    )
    .await;

    let ids = surviving_ids(
        &ctx,
        "SELECT id FROM catalog.test_del_mread_v3_merge.items ORDER BY id",
    )
    .await;
    assert_eq!(ids, vec![1, 4], "both deletes must apply");

    let delete_files = live_delete_files(&client, "test_del_mread_v3_merge", "items").await?;
    assert_eq!(
        delete_files.len(),
        1,
        "the second DV supersedes the first, so exactly one stays live"
    );
    assert_eq!(
        delete_files[0].record_count(),
        2,
        "the surviving DV carries BOTH deleted positions, so the first was merged not dropped"
    );
    Ok(())
}

/// Risk pinned: the UPDATE arm still writing position deletes on V3 after the DELETE arm was
/// converted. Both arms call the same guard, and both must dispatch.
#[tokio::test]
async fn test_update_mread_v3_writes_a_deletion_vector() -> Result<()> {
    let (ctx, client) = make_v3_mread_ctx("test_upd_mread_v3", "items").await?;
    ctx.sql("INSERT INTO catalog.test_upd_mread_v3.items VALUES (1,'a'),(2,'b'),(3,'c')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    ctx.sql("UPDATE catalog.test_upd_mread_v3.items SET val = 'z' WHERE id = 2")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let batches = ctx
        .sql("SELECT id, val FROM catalog.test_upd_mread_v3.items ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 3, "an UPDATE changes rows, it does not remove them");

    let delete_files = live_delete_files(&client, "test_upd_mread_v3", "items").await?;
    assert_eq!(delete_files.len(), 1, "the updated row is hidden by one DV");
    assert_eq!(
        delete_files[0].file_format(),
        iceberg::spec::DataFileFormat::Puffin,
        "the UPDATE arm must dispatch to deletion vectors on V3 too"
    );
    Ok(())
}

/// Risk pinned: the version dispatch admitting V1, which has no delete files of any kind.
#[tokio::test]
async fn test_mread_v1_is_still_refused() -> Result<()> {
    let (ctx, _client) = make_versioned_mread_ctx(
        "test_del_mread_v1",
        "items",
        iceberg::spec::FormatVersion::V1,
    )
    .await?;
    ctx.sql("INSERT INTO catalog.test_del_mread_v1.items VALUES (1,'a'),(2,'b')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let error = ctx
        .sql("DELETE FROM catalog.test_del_mread_v1.items WHERE id = 2")
        .await
        .unwrap()
        .collect()
        .await
        .expect_err("a V1 table has no delete files, so merge-on-read cannot run");
    assert!(
        error.to_string().contains("copy-on-write"),
        "the refusal must name the alternative; got: {error}"
    );
    Ok(())
}

/// A V3 `{id int, category string, val string}` table partitioned by `identity(category)`, both
/// row-level modes merge-on-read.
async fn make_v3_partitioned_mread_ctx(
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
            NestedField::required(3, "val", Type::Primitive(PrimitiveType::String)).into(),
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
        .format_version(iceberg::spec::FormatVersion::V3)
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

/// Risk pinned: a V3 DV written with NO partition context. An unpartitioned table cannot see this
/// — a missing key and the real key both resolve to spec 0 with an empty tuple. On a partitioned
/// table the DV would carry an empty partition tuple, which groups it into the wrong per-spec
/// manifest and can prune it out of the scan that must apply it.
#[tokio::test]
async fn test_delete_mread_v3_partitioned_dv_carries_its_data_file_partition() -> Result<()> {
    let (ctx, client) = make_v3_partitioned_mread_ctx("test_del_mread_v3_part", "items").await?;
    ctx.sql(
        "INSERT INTO catalog.test_del_mread_v3_part.items VALUES \
         (1,'electronics','laptop'),(2,'electronics','phone'),(3,'books','novel')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let deleted = delete_count(
        &ctx,
        "DELETE FROM catalog.test_del_mread_v3_part.items WHERE id = 1",
    )
    .await;
    assert_eq!(deleted, 1, "exactly one row matches id = 1");

    let ids = surviving_ids(
        &ctx,
        "SELECT id FROM catalog.test_del_mread_v3_part.items ORDER BY id",
    )
    .await;
    assert_eq!(ids, vec![2, 3], "the DV must hide exactly row id = 1");

    let delete_files = live_delete_files(&client, "test_del_mread_v3_part", "items").await?;
    assert_eq!(delete_files.len(), 1, "one DV covers the one touched file");
    let dv = &delete_files[0];
    assert_eq!(
        dv.file_format(),
        iceberg::spec::DataFileFormat::Puffin,
        "V3 forbids new position-delete files"
    );

    // The DV must carry the SAME (spec_id, partition) as the data file it covers, not an empty
    // tuple under a fabricated spec.
    let table_ident = iceberg::TableIdent::new(
        NamespaceIdent::new("test_del_mread_v3_part".to_string()),
        "items".to_string(),
    );
    let table = client.load_table(&table_ident).await?;
    let snapshot = table.metadata().current_snapshot().unwrap();
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await?;
    let referenced = dv.referenced_data_file().expect("a DV names its data file");
    let mut data_file_partition = None;
    for manifest_entry in manifest_list.entries() {
        if manifest_entry.content != iceberg::spec::ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_entry.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if entry.is_alive() && entry.data_file().file_path() == referenced {
                data_file_partition = Some((
                    entry.data_file().partition_spec_id(),
                    entry.data_file().partition().clone(),
                ));
            }
        }
    }
    let (spec_id, partition) = data_file_partition.expect("the covered data file is live");
    assert_eq!(
        dv.partition_spec_id(),
        spec_id,
        "the DV must be stamped with its data file's spec id"
    );
    assert_eq!(
        dv.partition(),
        &partition,
        "the DV must carry its data file's partition tuple, not an empty one"
    );
    Ok(())
}

/// Risk pinned: the UPDATE arm never removing a superseded DV. The DELETE arm's removal is pinned
/// by its own second-delete test; this is the same path on the other arm, and the two are wired
/// separately.
///
/// The second UPDATE must target a row that is still in the ORIGINAL data file. A row that was
/// already updated has moved to a new file and would need no merge.
#[tokio::test]
async fn test_update_mread_v3_second_update_supersedes_the_first_dv() -> Result<()> {
    let (ctx, client) = make_v3_mread_ctx("test_upd_mread_v3_merge", "items").await?;
    ctx.sql("INSERT INTO catalog.test_upd_mread_v3_merge.items VALUES (1,'a'),(2,'b'),(3,'c')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    ctx.sql("UPDATE catalog.test_upd_mread_v3_merge.items SET val = 'y' WHERE id = 1")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    ctx.sql("UPDATE catalog.test_upd_mread_v3_merge.items SET val = 'z' WHERE id = 2")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ids = surviving_ids(
        &ctx,
        "SELECT id FROM catalog.test_upd_mread_v3_merge.items ORDER BY id",
    )
    .await;
    assert_eq!(
        ids,
        vec![1, 2, 3],
        "an UPDATE must not lose or duplicate rows"
    );

    let delete_files = live_delete_files(&client, "test_upd_mread_v3_merge", "items").await?;
    assert_eq!(
        delete_files.len(),
        1,
        "both updates hide rows of the SAME original data file, so the second DV must supersede \
         the first rather than joining it"
    );
    assert_eq!(
        delete_files[0].record_count(),
        2,
        "the surviving DV carries BOTH hidden positions, so the first was merged not dropped"
    );
    Ok(())
}

/// Risk pinned: one `PartitionKey` stamped on every DV of a multi-file delete. A statement matching
/// rows in two data files writes two DVs in ONE Puffin, and each must carry its OWN data file's
/// partition. Every other V3 test touches a single data file, where any per-path lookup is an
/// identity.
#[tokio::test]
async fn test_delete_mread_v3_multi_file_dvs_each_carry_their_own_partition() -> Result<()> {
    let (ctx, client) = make_v3_partitioned_mread_ctx("test_del_mread_v3_multi", "items").await?;
    ctx.sql(
        "INSERT INTO catalog.test_del_mread_v3_multi.items VALUES \
         (1,'electronics','laptop'),(2,'electronics','phone'),\
         (3,'books','novel'),(4,'books','textbook')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // One statement, two partitions, two data files.
    let deleted = delete_count(
        &ctx,
        "DELETE FROM catalog.test_del_mread_v3_multi.items WHERE id = 1 OR id = 3",
    )
    .await;
    assert_eq!(deleted, 2, "one row matches in each partition");

    let ids = surviving_ids(
        &ctx,
        "SELECT id FROM catalog.test_del_mread_v3_multi.items ORDER BY id",
    )
    .await;
    assert_eq!(ids, vec![2, 4], "exactly one row survives per partition");

    let delete_files = live_delete_files(&client, "test_del_mread_v3_multi", "items").await?;
    assert_eq!(delete_files.len(), 2, "one DV per touched data file");

    // The two DVs must carry DIFFERENT partition tuples — their own, not a shared one.
    let mut partitions: Vec<iceberg::spec::Struct> = delete_files
        .iter()
        .map(|dv| dv.partition().clone())
        .collect();
    partitions.dedup();
    assert_eq!(
        partitions.len(),
        2,
        "each DV must carry its OWN data file's partition; a shared key collapses them to one"
    );
    Ok(())
}

/// Risk pinned: a V2 table with Parquet position deletes upgraded to V3, then deleted from again.
/// Java's `loadPreviousDeletes` would union those positions into the new DV; this port reads DVs
/// only, so it must refuse — and refuse BEFORE the Puffin is written. The commit door catches it
/// either way, but only after a fully written, unreferenced Puffin has reached storage.
#[tokio::test]
async fn test_delete_mread_v3_refuses_a_file_still_covered_by_position_deletes() -> Result<()> {
    let (ctx, client) = make_versioned_mread_ctx(
        "test_del_mread_upgrade",
        "items",
        iceberg::spec::FormatVersion::V2,
    )
    .await?;
    ctx.sql("INSERT INTO catalog.test_del_mread_upgrade.items VALUES (1,'a'),(2,'b'),(3,'c')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    // V2: this writes a Parquet position-delete file.
    delete_count(
        &ctx,
        "DELETE FROM catalog.test_del_mread_upgrade.items WHERE id = 1",
    )
    .await;

    let table_ident = iceberg::TableIdent::new(
        NamespaceIdent::new("test_del_mread_upgrade".to_string()),
        "items".to_string(),
    );
    let table = client.load_table(&table_ident).await?;
    let tx = iceberg::transaction::Transaction::new(&table);
    let action = tx
        .upgrade_table_version()
        .set_format_version(iceberg::spec::FormatVersion::V3);
    let tx = iceberg::transaction::ApplyTransactionAction::apply(action, tx)?;
    tx.commit(client.as_ref()).await?;

    // The provider snapshots the catalog at construction, so rebuild it to see V3.
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "catalog",
        Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?),
    );

    let error = ctx
        .sql("DELETE FROM catalog.test_del_mread_upgrade.items WHERE id = 2")
        .await
        .unwrap()
        .collect()
        .await
        .expect_err("the data file is still covered by a Parquet position delete");
    assert!(
        error.to_string().contains("Parquet position-delete"),
        "the refusal must name the cause; got: {error}"
    );

    // Nothing was written: the refusal runs before the Puffin is opened.
    let delete_files = live_delete_files(&client, "test_del_mread_upgrade", "items").await?;
    assert_eq!(
        delete_files.len(),
        1,
        "only the original V2 position-delete file is live — no orphaned DV was committed"
    );
    assert_eq!(
        delete_files[0].file_format(),
        iceberg::spec::DataFileFormat::Parquet,
        "the live delete file is still the V2 position delete"
    );
    Ok(())
}

/// Risk pinned: the V3 arm stamping every DV with the table's CURRENT default spec instead of the
/// data file's own. The V2 arm carries this pin already
/// (`test_delete_mread_after_drop_partition_field_no_resurrection`), added because the same mistake
/// on the position-delete path was a real shipped bug. Every other V3 test uses a single-spec
/// table, where the two expressions are the same value.
#[tokio::test]
async fn test_delete_mread_v3_after_drop_partition_field_stamps_the_files_own_spec() -> Result<()> {
    use iceberg::transaction::{ApplyTransactionAction, Transaction};

    let (ctx, client) = make_v3_partitioned_mread_ctx("test_del_mread_v3_evolved", "items").await?;
    ctx.sql(
        "INSERT INTO catalog.test_del_mread_v3_evolved.items VALUES \
         (1,'electronics','laptop'),(2,'electronics','phone')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let ns = NamespaceIdent::new("test_del_mread_v3_evolved".to_string());
    let tbl_id = iceberg::TableIdent::new(ns.clone(), "items".to_string());

    // Drop identity(category): the new default spec is unpartitioned, but the data file stays
    // under the original partitioned spec.
    let table = client.load_table(&tbl_id).await?;
    let original_spec_id = table.metadata().default_partition_spec().spec_id();
    let tx = Transaction::new(&table);
    let tx = tx
        .update_partition_spec()
        .remove_field("category")
        .apply(tx)
        .expect("apply remove_field(category)");
    let table = tx.commit(client.as_ref()).await.expect("commit evolution");
    let default_spec_id = table.metadata().default_partition_spec().spec_id();
    assert_ne!(
        original_spec_id, default_spec_id,
        "fixture: the default spec must have moved, or the mutation is unobservable"
    );

    let ctx2 = SessionContext::new();
    ctx2.register_catalog(
        "catalog",
        Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?),
    );
    let deleted = delete_count(
        &ctx2,
        "DELETE FROM catalog.test_del_mread_v3_evolved.items WHERE id = 1",
    )
    .await;
    assert_eq!(deleted, 1, "one row matches");

    let delete_files = live_delete_files(&client, "test_del_mread_v3_evolved", "items").await?;
    assert_eq!(delete_files.len(), 1, "one DV covers the one touched file");
    assert_eq!(
        delete_files[0].partition_spec_id(),
        original_spec_id,
        "the DV must carry the DATA FILE's spec, not the table's new default"
    );

    let ids = surviving_ids(
        &ctx2,
        "SELECT id FROM catalog.test_del_mread_v3_evolved.items ORDER BY id",
    )
    .await;
    assert_eq!(ids, vec![2], "the deleted row must not resurrect");
    Ok(())
}

/// The PARTITIONED form of the legacy-position-delete refusal.
///
/// It does NOT isolate the partition-tuple carry, though an earlier version of this comment claimed
/// it did. The fork's own V2 position delete satisfies BOTH legs at once — it carries a derivable
/// name AND its data file's own `(spec_id, partition)` — so killing either leg alone leaves the
/// other holding this test up; only killing both reddens it. The per-leg pins live in the seam
/// tests. This one pins that the whole rule still refuses on a partitioned table.
#[tokio::test]
async fn test_delete_mread_v3_partitioned_refuses_a_file_still_covered_by_position_deletes()
-> Result<()> {
    use iceberg::transaction::{ApplyTransactionAction, Transaction};

    let (ctx, client) = make_partitioned_mread_ctx("test_del_mread_part_upgrade", "items").await?;
    ctx.sql(
        "INSERT INTO catalog.test_del_mread_part_upgrade.items VALUES \
         (1, 'electronics', 'laptop'), (2, 'electronics', 'phone'), (3, 'books', 'novel')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // V2: writes a Parquet position-delete file, stamped in the electronics partition.
    delete_count(
        &ctx,
        "DELETE FROM catalog.test_del_mread_part_upgrade.items WHERE id = 1",
    )
    .await;

    let tbl_id = iceberg::TableIdent::new(
        NamespaceIdent::new("test_del_mread_part_upgrade".to_string()),
        "items".to_string(),
    );
    let table = client.load_table(&tbl_id).await?;
    let tx = Transaction::new(&table);
    let action = tx
        .upgrade_table_version()
        .set_format_version(iceberg::spec::FormatVersion::V3);
    let tx = ApplyTransactionAction::apply(action, tx)?;
    tx.commit(client.as_ref()).await?;

    let ctx = SessionContext::new();
    ctx.register_catalog(
        "catalog",
        Arc::new(IcebergCatalogProvider::try_new(client.clone()).await?),
    );

    // The other electronics row lives in the file that delete still covers.
    let error = ctx
        .sql("DELETE FROM catalog.test_del_mread_part_upgrade.items WHERE id = 2")
        .await
        .unwrap()
        .collect()
        .await
        .expect_err("the data file is still covered by a Parquet position delete");
    assert!(
        error.to_string().contains("Parquet position-delete"),
        "the refusal must name the cause; got: {error}"
    );

    let delete_files = live_delete_files(&client, "test_del_mread_part_upgrade", "items").await?;
    assert_eq!(
        delete_files.len(),
        1,
        "only the original V2 position-delete file is live — no orphaned DV was committed"
    );
    Ok(())
}
