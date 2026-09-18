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

use datafusion::arrow::array::{Int32Array, UInt64Array};
use datafusion::prelude::SessionContext;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    FormatVersion, ListType, MapType, NestedField, PrimitiveType, Schema as IcebergSchema,
    StructType, Type,
};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use tempfile::TempDir;

use crate::IcebergCatalogProvider;

struct NullFixture {
    ctx: SessionContext,
    _warehouse: TempDir,
}

#[derive(Clone, Copy)]
enum NullShape {
    ListInt,
    ListStruct,
    MapStrInt,
    StructInt,
}

impl NullShape {
    const ALL: [NullShape; 4] = [
        NullShape::ListInt,
        NullShape::ListStruct,
        NullShape::MapStrInt,
        NullShape::StructInt,
    ];

    fn name(self) -> &'static str {
        match self {
            NullShape::ListInt => "list<int>",
            NullShape::ListStruct => "list<struct<a:int>>",
            NullShape::MapStrInt => "map<string,int>",
            NullShape::StructInt => "struct<a:int>",
        }
    }

    fn xs_type(self) -> Type {
        match self {
            NullShape::ListInt => Type::List(ListType::new(
                NestedField::list_element(3, Type::Primitive(PrimitiveType::Int), false).into(),
            )),
            NullShape::ListStruct => Type::List(ListType::new(
                NestedField::list_element(
                    3,
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(4, "a", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                    false,
                )
                .into(),
            )),
            NullShape::MapStrInt => Type::Map(MapType::new(
                NestedField::map_key_element(3, Type::Primitive(PrimitiveType::String)).into(),
                NestedField::map_value_element(4, Type::Primitive(PrimitiveType::Int), false)
                    .into(),
            )),
            NullShape::StructInt => Type::Struct(StructType::new(vec![
                NestedField::optional(3, "a", Type::Primitive(PrimitiveType::Int)).into(),
            ])),
        }
    }

    fn seed(self) -> &'static str {
        match self {
            NullShape::ListInt => "(1, [1, 2]), (2, NULL), (3, []), (4, [NULL])",
            NullShape::ListStruct => {
                "(1, [named_struct('a', CAST(1 AS INT))]), (2, NULL), (3, []), \
                 (4, [CAST(NULL AS STRUCT<a INT>)])"
            }
            NullShape::MapStrInt => {
                "(1, map('k', CAST(1 AS INT))), (2, NULL), (3, MAP {}), \
                 (4, map('k', CAST(NULL AS INT)))"
            }
            NullShape::StructInt => {
                "(1, named_struct('a', CAST(1 AS INT))), (2, NULL), \
                 (3, named_struct('a', CAST(NULL AS INT))), \
                 (4, named_struct('a', CAST(4 AS INT)))"
            }
        }
    }
}

async fn null_fixture(
    merge_on_read: bool,
    format_version: FormatVersion,
    shape: NullShape,
) -> NullFixture {
    let warehouse = TempDir::new().expect("warehouse");
    let catalog = Arc::new(
        MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(
                    MEMORY_CATALOG_WAREHOUSE.to_string(),
                    warehouse.path().to_str().expect("utf8").to_string(),
                )]),
            )
            .await
            .expect("catalog"),
    );
    let namespace = NamespaceIdent::new("ns".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let schema = IcebergSchema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "xs", shape.xs_type()).into(),
        ])
        .build()
        .expect("schema");
    let mode = if merge_on_read {
        "merge-on-read"
    } else {
        "copy-on-write"
    };
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("t".to_string())
                .location(format!("{}/t", warehouse.path().to_str().expect("utf8")))
                .schema(schema)
                .format_version(format_version)
                .properties(HashMap::from([
                    ("write.delete.mode".to_string(), mode.to_string()),
                    ("write.update.mode".to_string(), mode.to_string()),
                ]))
                .build(),
        )
        .await
        .expect("table");

    let catalog_provider = IcebergCatalogProvider::try_new(catalog.clone())
        .await
        .expect("catalog provider");
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(catalog_provider));
    ctx.sql(&format!("INSERT INTO catalog.ns.t VALUES {}", shape.seed()))
        .await
        .expect("plan seed insert")
        .collect()
        .await
        .expect("seed insert");

    NullFixture {
        ctx,
        _warehouse: warehouse,
    }
}

async fn ids(ctx: &SessionContext) -> Vec<i64> {
    let batches = ctx
        .sql("SELECT id FROM catalog.ns.t ORDER BY id")
        .await
        .expect("plan select ids")
        .collect()
        .await
        .expect("select ids");
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("id column")
                .values()
                .iter()
                .map(|id| i64::from(*id))
                .collect::<Vec<i64>>()
        })
        .collect()
}

async fn dml_count(ctx: &SessionContext, sql: &str) -> u64 {
    let batches = ctx
        .sql(sql)
        .await
        .expect("plan DML")
        .collect()
        .await
        .expect("DML commits");
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("count column")
                .values()
                .iter()
                .copied()
        })
        .sum()
}

async fn select_ids(ctx: &SessionContext, sql: &str) -> Vec<i64> {
    let batches = ctx
        .sql(sql)
        .await
        .expect("plan select")
        .collect()
        .await
        .expect("select");
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("id column")
                .values()
                .iter()
                .map(|id| i64::from(*id))
                .collect::<Vec<i64>>()
        })
        .collect()
}

#[tokio::test]
async fn delete_where_xs_is_null_removes_only_the_null_row() {
    for shape in NullShape::ALL {
        for format_version in [FormatVersion::V2, FormatVersion::V3] {
            for merge_on_read in [true, false] {
                let fixture = null_fixture(merge_on_read, format_version, shape).await;
                let deleted =
                    dml_count(&fixture.ctx, "DELETE FROM catalog.ns.t WHERE xs IS NULL").await;
                assert_eq!(
                    deleted,
                    1,
                    "{} {format_version:?} merge_on_read={merge_on_read}: only id 2 is null",
                    shape.name()
                );
                assert_eq!(
                    ids(&fixture.ctx).await,
                    vec![1, 3, 4],
                    "{} {format_version:?} merge_on_read={merge_on_read}",
                    shape.name()
                );
            }
        }
    }
}

#[tokio::test]
async fn delete_where_xs_is_not_null_keeps_only_the_null_row() {
    for shape in NullShape::ALL {
        for format_version in [FormatVersion::V2, FormatVersion::V3] {
            for merge_on_read in [true, false] {
                let fixture = null_fixture(merge_on_read, format_version, shape).await;
                let deleted = dml_count(
                    &fixture.ctx,
                    "DELETE FROM catalog.ns.t WHERE xs IS NOT NULL",
                )
                .await;
                assert_eq!(
                    deleted,
                    3,
                    "{} {format_version:?} merge_on_read={merge_on_read}: the non-null rows delete",
                    shape.name()
                );
                assert_eq!(
                    ids(&fixture.ctx).await,
                    vec![2],
                    "{} {format_version:?} merge_on_read={merge_on_read}",
                    shape.name()
                );
            }
        }
    }
}

#[tokio::test]
async fn delete_where_id_and_xs_is_null_composes_with_a_primitive_conjunct() {
    for shape in NullShape::ALL {
        for format_version in [FormatVersion::V2, FormatVersion::V3] {
            for merge_on_read in [true, false] {
                let fixture = null_fixture(merge_on_read, format_version, shape).await;
                let deleted = dml_count(
                    &fixture.ctx,
                    "DELETE FROM catalog.ns.t WHERE id > 1 AND xs IS NULL",
                )
                .await;
                assert_eq!(
                    deleted,
                    1,
                    "{} {format_version:?} merge_on_read={merge_on_read}: only id 2 is null",
                    shape.name()
                );
                assert_eq!(
                    ids(&fixture.ctx).await,
                    vec![1, 3, 4],
                    "{} {format_version:?} merge_on_read={merge_on_read}",
                    shape.name()
                );
            }
        }
    }
}

#[tokio::test]
async fn delete_where_xs_is_null_or_id_eq_1_keeps_matching_rows() {
    for shape in NullShape::ALL {
        for format_version in [FormatVersion::V2, FormatVersion::V3] {
            for merge_on_read in [true, false] {
                let fixture = null_fixture(merge_on_read, format_version, shape).await;
                let deleted = dml_count(
                    &fixture.ctx,
                    "DELETE FROM catalog.ns.t WHERE xs IS NULL OR id = 1",
                )
                .await;
                assert_eq!(
                    deleted,
                    2,
                    "{} {format_version:?} merge_on_read={merge_on_read}: ids 1 and 2 match",
                    shape.name()
                );
                assert_eq!(
                    ids(&fixture.ctx).await,
                    vec![3, 4],
                    "{} {format_version:?} merge_on_read={merge_on_read}",
                    shape.name()
                );
            }
        }
    }
}

#[tokio::test]
async fn update_where_xs_is_null_updates_only_the_null_row() {
    for shape in NullShape::ALL {
        for format_version in [FormatVersion::V2, FormatVersion::V3] {
            for merge_on_read in [true, false] {
                let fixture = null_fixture(merge_on_read, format_version, shape).await;
                let updated = dml_count(
                    &fixture.ctx,
                    "UPDATE catalog.ns.t SET id = id + 100 WHERE xs IS NULL",
                )
                .await;
                assert_eq!(
                    updated,
                    1,
                    "{} {format_version:?} merge_on_read={merge_on_read}: only id 2 is null",
                    shape.name()
                );
                assert_eq!(
                    ids(&fixture.ctx).await,
                    vec![1, 3, 4, 102],
                    "{} {format_version:?} merge_on_read={merge_on_read}",
                    shape.name()
                );
            }
        }
    }
}

#[tokio::test]
async fn update_where_xs_is_not_null_updates_every_non_null_row() {
    for shape in NullShape::ALL {
        for format_version in [FormatVersion::V2, FormatVersion::V3] {
            for merge_on_read in [true, false] {
                let fixture = null_fixture(merge_on_read, format_version, shape).await;
                let updated = dml_count(
                    &fixture.ctx,
                    "UPDATE catalog.ns.t SET id = id + 100 WHERE xs IS NOT NULL",
                )
                .await;
                assert_eq!(
                    updated,
                    3,
                    "{} {format_version:?} merge_on_read={merge_on_read}: the non-null rows update",
                    shape.name()
                );
                assert_eq!(
                    ids(&fixture.ctx).await,
                    vec![2, 101, 103, 104],
                    "{} {format_version:?} merge_on_read={merge_on_read}",
                    shape.name()
                );
            }
        }
    }
}

#[tokio::test]
async fn update_where_id_and_xs_is_null_composes_with_a_primitive_conjunct() {
    for shape in NullShape::ALL {
        for format_version in [FormatVersion::V2, FormatVersion::V3] {
            for merge_on_read in [true, false] {
                let fixture = null_fixture(merge_on_read, format_version, shape).await;
                let updated = dml_count(
                    &fixture.ctx,
                    "UPDATE catalog.ns.t SET id = id + 100 WHERE id > 1 AND xs IS NULL",
                )
                .await;
                assert_eq!(
                    updated,
                    1,
                    "{} {format_version:?} merge_on_read={merge_on_read}: only id 2 is null",
                    shape.name()
                );
                assert_eq!(
                    ids(&fixture.ctx).await,
                    vec![1, 3, 4, 102],
                    "{} {format_version:?} merge_on_read={merge_on_read}",
                    shape.name()
                );
            }
        }
    }
}

#[tokio::test]
async fn update_where_xs_is_null_or_id_eq_1_updates_matching_rows() {
    for shape in NullShape::ALL {
        for format_version in [FormatVersion::V2, FormatVersion::V3] {
            for merge_on_read in [true, false] {
                let fixture = null_fixture(merge_on_read, format_version, shape).await;
                let updated = dml_count(
                    &fixture.ctx,
                    "UPDATE catalog.ns.t SET id = id + 100 WHERE xs IS NULL OR id = 1",
                )
                .await;
                assert_eq!(
                    updated,
                    2,
                    "{} {format_version:?} merge_on_read={merge_on_read}: ids 1 and 2 match",
                    shape.name()
                );
                assert_eq!(
                    ids(&fixture.ctx).await,
                    vec![3, 4, 101, 102],
                    "{} {format_version:?} merge_on_read={merge_on_read}",
                    shape.name()
                );
            }
        }
    }
}

#[tokio::test]
async fn select_where_xs_dot_a_is_null_returns_the_null_leaf_rows() {
    for format_version in [FormatVersion::V2, FormatVersion::V3] {
        for merge_on_read in [true, false] {
            let fixture = null_fixture(merge_on_read, format_version, NullShape::StructInt).await;
            assert_eq!(
                select_ids(
                    &fixture.ctx,
                    "SELECT id FROM catalog.ns.t WHERE xs.a IS NULL ORDER BY id"
                )
                .await,
                vec![2, 3],
                "{format_version:?} merge_on_read={merge_on_read}: a NULL struct and {{a:NULL}} both have xs.a NULL"
            );
        }
    }
}

#[tokio::test]
async fn select_where_xs_is_null_returns_the_null_row() {
    for shape in NullShape::ALL {
        for format_version in [FormatVersion::V2, FormatVersion::V3] {
            for merge_on_read in [true, false] {
                let fixture = null_fixture(merge_on_read, format_version, shape).await;
                assert_eq!(
                    select_ids(&fixture.ctx, "SELECT id FROM catalog.ns.t WHERE xs IS NULL").await,
                    vec![2],
                    "{} {format_version:?} merge_on_read={merge_on_read}",
                    shape.name()
                );
            }
        }
    }
}
