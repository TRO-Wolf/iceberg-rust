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
use futures::TryStreamExt;
use iceberg::expr::{Predicate, Reference};
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::spec::{
    Datum, FormatVersion, ListType, MapType, NestedField, PrimitiveType, Schema as IcebergSchema,
    StructType, Type,
};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use tempfile::TempDir;

use crate::IcebergCatalogProvider;

struct NullFixture {
    ctx: SessionContext,
    catalog: Arc<MemoryCatalog>,
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

    fn rows(self) -> [&'static str; 4] {
        match self {
            NullShape::ListInt => ["(1, [1, 2])", "(2, NULL)", "(3, [])", "(4, [NULL])"],
            NullShape::ListStruct => [
                "(1, [named_struct('a', CAST(1 AS INT))])",
                "(2, NULL)",
                "(3, [])",
                "(4, [CAST(NULL AS STRUCT<a INT>)])",
            ],
            NullShape::MapStrInt => [
                "(1, map('k', CAST(1 AS INT)))",
                "(2, NULL)",
                "(3, MAP {})",
                "(4, map('k', CAST(NULL AS INT)))",
            ],
            NullShape::StructInt => [
                "(1, named_struct('a', CAST(1 AS INT)))",
                "(2, NULL)",
                "(3, named_struct('a', CAST(NULL AS INT)))",
                "(4, named_struct('a', CAST(4 AS INT)))",
            ],
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

#[derive(Clone, Copy, Default)]
struct FixtureOpts {
    seeded_per_row: bool,
    partition_on_id: bool,
    target_partitions: Option<usize>,
    metrics_default: Option<&'static str>,
}

async fn null_fixture_opts(
    merge_on_read: bool,
    format_version: FormatVersion,
    shape: NullShape,
    opts: FixtureOpts,
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
    let mut properties = HashMap::from([
        ("write.delete.mode".to_string(), mode.to_string()),
        ("write.update.mode".to_string(), mode.to_string()),
    ]);
    if let Some(metrics) = opts.metrics_default {
        properties.insert(
            "write.metadata.metrics.default".to_string(),
            metrics.to_string(),
        );
    }
    let creation = TableCreation::builder()
        .name("t".to_string())
        .location(format!("{}/t", warehouse.path().to_str().expect("utf8")))
        .schema(schema)
        .format_version(format_version)
        .properties(properties);
    if opts.partition_on_id {
        let creation = creation
            .partition_spec(
                iceberg::spec::UnboundPartitionSpec::builder()
                    .with_spec_id(0)
                    .add_partition_field(1, "id", iceberg::spec::Transform::Identity)
                    .expect("identity(id)")
                    .build(),
            )
            .build();
        catalog
            .create_table(&namespace, creation)
            .await
            .expect("table");
    } else {
        catalog
            .create_table(&namespace, creation.build())
            .await
            .expect("table");
    }

    let catalog_provider = IcebergCatalogProvider::try_new(catalog.clone())
        .await
        .expect("catalog provider");
    let ctx = match opts.target_partitions {
        Some(n) => SessionContext::new_with_config(
            datafusion::execution::config::SessionConfig::new().with_target_partitions(n),
        ),
        None => SessionContext::new(),
    };
    ctx.register_catalog("catalog", Arc::new(catalog_provider));
    if opts.seeded_per_row {
        for row in shape.rows() {
            ctx.sql(&format!("INSERT INTO catalog.ns.t VALUES {row}"))
                .await
                .expect("plan seed insert")
                .collect()
                .await
                .expect("seed insert");
        }
    } else {
        ctx.sql(&format!("INSERT INTO catalog.ns.t VALUES {}", shape.seed()))
            .await
            .expect("plan seed insert")
            .collect()
            .await
            .expect("seed insert");
    }

    NullFixture {
        ctx,
        catalog,
        _warehouse: warehouse,
    }
}

async fn null_fixture(
    merge_on_read: bool,
    format_version: FormatVersion,
    shape: NullShape,
) -> NullFixture {
    null_fixture_opts(merge_on_read, format_version, shape, FixtureOpts::default()).await
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

async fn plan_task_count(table: &iceberg::table::Table, predicate: Predicate) -> usize {
    let tasks: Vec<iceberg::scan::FileScanTask> = table
        .scan()
        .with_file_prune_only(predicate)
        .build()
        .expect("a prune-only scan must never bind an unbindable term")
        .plan_files()
        .await
        .expect("plan files")
        .try_collect()
        .await
        .expect("collect tasks");
    tasks.len()
}

#[tokio::test]
async fn cow_prune_scan_drops_the_unbindable_null_term() {
    for shape in NullShape::ALL {
        for format_version in [FormatVersion::V2, FormatVersion::V3] {
            let fixture = null_fixture_opts(false, format_version, shape, FixtureOpts {
                seeded_per_row: true,
                ..FixtureOpts::default()
            })
            .await;
            let table = fixture
                .catalog
                .load_table(&TableIdent::from_strs(["ns", "t"]).expect("ident"))
                .await
                .expect("load table");

            let and_prune = Reference::new("id")
                .greater_than(Datum::int(1))
                .and(Reference::new("xs").is_null());
            let planned = plan_task_count(&table, and_prune).await;
            assert_eq!(
                planned,
                3,
                "{} {format_version:?}: the sound conjunct prunes the id=1 file only",
                shape.name()
            );

            let or_prune = Reference::new("xs")
                .is_null()
                .or(Reference::new("id").equal_to(Datum::int(1)));
            let planned = plan_task_count(&table, or_prune).await;
            assert_eq!(
                planned,
                4,
                "{} {format_version:?}: an unbindable disjunct widens the prune to every file",
                shape.name()
            );
        }
    }
}

#[tokio::test]
async fn filtered_row_filter_stays_loud_on_an_unbindable_term() {
    for shape in NullShape::ALL {
        for format_version in [FormatVersion::V2, FormatVersion::V3] {
            let fixture = null_fixture_opts(false, format_version, shape, FixtureOpts {
                seeded_per_row: true,
                ..FixtureOpts::default()
            })
            .await;
            let table = fixture
                .catalog
                .load_table(&TableIdent::from_strs(["ns", "t"]).expect("ident"))
                .await
                .expect("load table");

            let compound = Reference::new("id")
                .greater_than(Datum::int(1))
                .and(Reference::new("xs").is_null());
            let err = table
                .scan()
                .with_filter(compound)
                .build()
                .expect_err("a row filter binds exactly or fails loudly");
            assert_eq!(
                err.kind(),
                iceberg::ErrorKind::DataInvalid,
                "{} {format_version:?}: the unbindable term must fail the row-filter bind",
                shape.name()
            );

            let err = table
                .scan()
                .with_filter(Reference::new("xs").is_null())
                .build()
                .expect_err("a row filter binds exactly or fails loudly");
            assert_eq!(
                err.kind(),
                iceberg::ErrorKind::DataInvalid,
                "{} {format_version:?}: the unbindable null test must fail the row-filter bind",
                shape.name()
            );
        }
    }
}

#[tokio::test]
async fn a_row_filter_never_returns_the_widened_row_set() {
    for shape in NullShape::ALL {
        for format_version in [FormatVersion::V2, FormatVersion::V3] {
            let fixture = null_fixture_opts(false, format_version, shape, FixtureOpts {
                seeded_per_row: true,
                ..FixtureOpts::default()
            })
            .await;
            let table = fixture
                .catalog
                .load_table(&TableIdent::from_strs(["ns", "t"]).expect("ident"))
                .await
                .expect("load table");

            match table
                .scan()
                .with_filter(Reference::new("xs").is_null())
                .build()
            {
                Err(err) => assert_eq!(
                    err.kind(),
                    iceberg::ErrorKind::DataInvalid,
                    "{} {format_version:?}: loud is the current row-filter contract",
                    shape.name()
                ),
                Ok(scan) => {
                    let batches: Vec<datafusion::arrow::record_batch::RecordBatch> = scan
                        .to_arrow()
                        .await
                        .expect("arrow stream")
                        .try_collect()
                        .await
                        .expect("collect batches");
                    let mut ids = vec![];
                    for batch in &batches {
                        let column = batch
                            .column_by_name("id")
                            .expect("id column")
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .expect("int32 id column");
                        ids.extend_from_slice(column.values());
                    }
                    ids.sort_unstable();
                    assert_eq!(
                        ids,
                        vec![2],
                        "{} {format_version:?}: Java returns only the NULL row; a widened residual returns every row",
                        shape.name()
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn incremental_scan_stays_loud_on_an_unbindable_term() {
    for shape in NullShape::ALL {
        for format_version in [FormatVersion::V2, FormatVersion::V3] {
            let fixture = null_fixture_opts(false, format_version, shape, FixtureOpts {
                seeded_per_row: true,
                ..FixtureOpts::default()
            })
            .await;
            let table = fixture
                .catalog
                .load_table(&TableIdent::from_strs(["ns", "t"]).expect("ident"))
                .await
                .expect("load table");

            let filter = Reference::new("id")
                .greater_than(Datum::int(1))
                .and(Reference::new("xs").is_null());
            let err = table
                .incremental_append_scan()
                .with_filter(filter)
                .build()
                .expect_err("incremental residuals always apply; the filter stays loud");
            assert_eq!(
                err.kind(),
                iceberg::ErrorKind::DataInvalid,
                "{} {format_version:?}: the unbindable term must fail the incremental bind",
                shape.name()
            );
        }
    }
}
