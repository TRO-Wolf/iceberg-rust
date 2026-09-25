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

use tempfile::TempDir;

use super::*;
use crate::io::LocalFsStorageFactory;
use crate::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use crate::spec::{
    NestedField, NullOrder, PrimitiveType, Schema, SortDirection, SortField, StructType, Transform,
    Type, UnboundPartitionSpec,
};
use crate::{Catalog, CatalogBuilder};

fn long(id: i32, name: &str) -> NestedField {
    NestedField::required(id, name, Type::Primitive(PrimitiveType::Long))
}

fn schema(fields: Vec<NestedField>) -> Schema {
    Schema::builder()
        .with_fields(fields.into_iter().map(Arc::new))
        .build()
        .unwrap()
}

fn location(id: i32, lat: i32, lon: i32, extra: Vec<NestedField>) -> NestedField {
    let mut fields = vec![long(lat, "lat"), long(lon, "lon")];
    fields.extend(extra);
    NestedField::required(
        id,
        "location",
        Type::Struct(StructType::new(fields.into_iter().map(Arc::new).collect())),
    )
}

fn replacement(schema: Schema) -> TableCreation {
    TableCreation::builder()
        .name("orders".into())
        .schema(schema)
        .build()
}

async fn table_with(tmp: &TempDir, creation: TableCreation) -> (impl Catalog, Table) {
    let warehouse = tmp.path().to_string_lossy().to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "mem",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .unwrap();
    let ns = NamespaceIdent::new("sales".into());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let creation = TableCreation {
        location: Some(format!("{warehouse}/sales/orders")),
        ..creation
    };
    let table = catalog.create_table(&ns, creation).await.unwrap();
    (catalog, table)
}

async fn staged_schema(existing: &Table, creation: TableCreation) -> (Schema, i32) {
    let staged = StagedTableTransaction::begin_replace(existing, creation)
        .await
        .unwrap();
    let metadata = staged.table().metadata();
    (
        metadata.current_schema().as_ref().clone(),
        metadata.last_column_id(),
    )
}

fn id_of(schema: &Schema, name: &str) -> Option<i32> {
    schema.field_id_by_name(name)
}

#[tokio::test]
async fn replace_reordered_columns_keep_their_ids_by_name() {
    let tmp = TempDir::new().unwrap();
    let base = schema(vec![long(1, "id"), long(2, "name")]);
    let (_catalog, table) = table_with(&tmp, replacement(base)).await;

    let reordered = schema(vec![long(1, "name"), long(2, "id")]);
    let (fresh, last_column_id) = staged_schema(&table, replacement(reordered)).await;

    assert_eq!(id_of(&fresh, "id"), Some(1));
    assert_eq!(id_of(&fresh, "name"), Some(2));
    assert_eq!(fresh.as_struct().fields()[0].name, "name");
    assert_eq!(last_column_id, 2);
}

#[tokio::test]
async fn replace_renamed_and_added_columns_take_ids_above_last_column_id() {
    let tmp = TempDir::new().unwrap();
    let base = schema(vec![long(1, "id"), long(2, "name")]);
    let (_catalog, table) = table_with(&tmp, replacement(base)).await;

    let renamed = schema(vec![long(1, "id"), long(2, "full_name"), long(3, "extra")]);
    let (fresh, last_column_id) = staged_schema(&table, replacement(renamed)).await;

    assert_eq!(id_of(&fresh, "id"), Some(1));
    assert_eq!(id_of(&fresh, "full_name"), Some(3));
    assert_eq!(id_of(&fresh, "extra"), Some(4));
    assert_eq!(last_column_id, 4);
}

#[tokio::test]
async fn replace_never_reuses_a_dropped_id_when_the_name_returns() {
    let tmp = TempDir::new().unwrap();
    let base = schema(vec![long(1, "id"), long(2, "name")]);
    let (catalog, table) = table_with(&tmp, replacement(base)).await;

    let dropped = schema(vec![long(1, "id")]);
    let after_drop = StagedTableTransaction::begin_replace(&table, replacement(dropped))
        .await
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    assert_eq!(after_drop.metadata().last_column_id(), 2);

    let readded = schema(vec![long(1, "id"), long(2, "name")]);
    let (fresh, last_column_id) = staged_schema(&after_drop, replacement(readded)).await;

    assert_eq!(id_of(&fresh, "id"), Some(1));
    assert_eq!(id_of(&fresh, "name"), Some(3));
    assert_eq!(last_column_id, 3);
}

#[tokio::test]
async fn replace_nested_struct_fields_keep_ids_by_dotted_name() {
    let tmp = TempDir::new().unwrap();
    let base = schema(vec![long(1, "id"), location(2, 3, 4, vec![])]);
    let (_catalog, table) = table_with(&tmp, replacement(base)).await;

    let moved = schema(vec![
        NestedField::required(
            1,
            "location",
            Type::Struct(StructType::new(vec![
                Arc::new(long(2, "lon")),
                Arc::new(long(3, "lat")),
                Arc::new(long(4, "id")),
            ])),
        ),
        long(5, "id"),
    ]);
    let (fresh, last_column_id) = staged_schema(&table, replacement(moved)).await;

    assert_eq!(id_of(&fresh, "id"), Some(1));
    assert_eq!(id_of(&fresh, "location"), Some(2));
    assert_eq!(id_of(&fresh, "location.lat"), Some(3));
    assert_eq!(id_of(&fresh, "location.lon"), Some(4));
    assert_eq!(id_of(&fresh, "location.id"), Some(5));
    assert_eq!(last_column_id, 5);
}

#[tokio::test]
async fn replace_rebinds_the_partition_spec_source_by_name() {
    let tmp = TempDir::new().unwrap();
    let base = schema(vec![long(1, "id"), long(2, "name")]);
    let base_spec = UnboundPartitionSpec::builder()
        .add_partition_field(2, "name_part", Transform::Identity)
        .unwrap()
        .build();
    let base_creation = TableCreation {
        partition_spec: Some(base_spec),
        ..replacement(base)
    };
    let (_catalog, table) = table_with(&tmp, base_creation).await;
    let base_field_id = table.metadata().default_partition_spec().fields()[0].field_id;

    let moved = schema(vec![long(1, "name"), long(2, "id")]);
    let moved_spec = UnboundPartitionSpec::builder()
        .add_partition_field(1, "name_part", Transform::Identity)
        .unwrap()
        .build();
    let creation = TableCreation {
        partition_spec: Some(moved_spec),
        ..replacement(moved)
    };
    let staged = StagedTableTransaction::begin_replace(&table, creation)
        .await
        .unwrap();
    let metadata = staged.table().metadata();
    let field = &metadata.default_partition_spec().fields()[0];

    assert_eq!(field.source_id, 2);
    assert_eq!(
        metadata.current_schema().name_by_field_id(field.source_id),
        Some("name")
    );
    assert_eq!(field.field_id, base_field_id);
}

#[tokio::test]
async fn replace_rebinds_the_sort_order_source_by_name() {
    let tmp = TempDir::new().unwrap();
    let base = schema(vec![long(1, "id"), long(2, "name")]);
    let (_catalog, table) = table_with(&tmp, replacement(base)).await;

    let moved = schema(vec![long(1, "name"), long(2, "id")]);
    let order = SortOrder::builder()
        .with_sort_field(SortField {
            source_id: 1,
            transform: Transform::Identity,
            direction: SortDirection::Descending,
            null_order: NullOrder::Last,
        })
        .build_unbound()
        .unwrap();
    let creation = TableCreation {
        sort_order: Some(order),
        ..replacement(moved)
    };
    let staged = StagedTableTransaction::begin_replace(&table, creation)
        .await
        .unwrap();
    let metadata = staged.table().metadata();
    let field = &metadata.default_sort_order().fields[0];

    assert_eq!(field.source_id, 2);
    assert_eq!(field.direction, SortDirection::Descending);
    assert_eq!(
        metadata.current_schema().name_by_field_id(field.source_id),
        Some("name")
    );
}

#[tokio::test]
async fn replace_rejects_a_partition_source_missing_from_the_replacement_schema() {
    let tmp = TempDir::new().unwrap();
    let base = schema(vec![long(1, "id"), long(2, "name")]);
    let (_catalog, table) = table_with(&tmp, replacement(base)).await;

    let spec = UnboundPartitionSpec::builder()
        .add_partition_field(9, "ghost", Transform::Identity)
        .unwrap()
        .build();
    let creation = TableCreation {
        partition_spec: Some(spec),
        ..replacement(schema(vec![long(1, "id"), long(2, "name")]))
    };
    let err = match StagedTableTransaction::begin_replace(&table, creation).await {
        Ok(_) => panic!("a partition source outside the replacement schema must fail"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
}

#[tokio::test]
async fn replace_keeps_identifier_fields_by_name() {
    let tmp = TempDir::new().unwrap();
    let base = Schema::builder()
        .with_fields(vec![Arc::new(long(1, "id")), Arc::new(long(2, "name"))])
        .with_identifier_field_ids(vec![1])
        .build()
        .unwrap();
    let (_catalog, table) = table_with(&tmp, replacement(base)).await;

    let moved = Schema::builder()
        .with_fields(vec![Arc::new(long(1, "name")), Arc::new(long(2, "id"))])
        .with_identifier_field_ids(vec![2])
        .build()
        .unwrap();
    let (fresh, _) = staged_schema(&table, replacement(moved)).await;

    assert_eq!(fresh.identifier_field_ids().collect::<Vec<_>>(), vec![1]);
    assert_eq!(id_of(&fresh, "id"), Some(1));
}

#[tokio::test]
async fn replace_leaves_caller_supplied_by_name_ids_unchanged() {
    let tmp = TempDir::new().unwrap();
    let base = schema(vec![
        long(1, "id"),
        long(2, "name"),
        location(3, 4, 5, vec![]),
    ]);
    let (_catalog, table) = table_with(&tmp, replacement(base)).await;

    let by_name = schema(vec![
        location(3, 4, 5, vec![long(7, "alt")]),
        long(2, "name"),
        long(1, "id"),
        long(6, "extra"),
    ]);
    let (fresh, last_column_id) = staged_schema(&table, replacement(by_name.clone())).await;

    assert_eq!(fresh.as_struct(), by_name.as_struct());
    assert_eq!(last_column_id, 7);
}
