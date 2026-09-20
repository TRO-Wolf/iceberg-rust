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

use datafusion::arrow::datatypes::{DataType, TimeUnit};
use datafusion::logical_expr::expr::Cast;
use datafusion::prelude::{Expr, col, lit};
use datafusion::scalar::ScalarValue;
use iceberg::expr::{Predicate, Reference};
use iceberg::spec::{Datum, NestedField, PrimitiveType, Schema as IcebergSchema, SchemaRef, Type};

use super::convert_filters_to_predicate;

const V2_MICROS: i64 = 1_703_000_000_000_000;
const V2_NANOS: i64 = 1_703_000_000_000_000_500;

fn ts_tz_schema() -> SchemaRef {
    IcebergSchema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "ts", Type::Primitive(PrimitiveType::Timestamptz)).into(),
            NestedField::optional(3, "tsn", Type::Primitive(PrimitiveType::Timestamp)).into(),
            NestedField::optional(4, "tsz", Type::Primitive(PrimitiveType::TimestamptzNs)).into(),
            NestedField::optional(5, "tsnn", Type::Primitive(PrimitiveType::TimestampNs)).into(),
        ])
        .build()
        .expect("ts tz schema")
        .into()
}

fn push(expr: Expr) -> Option<Predicate> {
    convert_filters_to_predicate(&[expr], &ts_tz_schema())
}

fn us(value: i64, tz: Option<&str>) -> ScalarValue {
    ScalarValue::TimestampMicrosecond(Some(value), tz.map(Into::into))
}

fn ns(value: i64, tz: Option<&str>) -> ScalarValue {
    ScalarValue::TimestampNanosecond(Some(value), tz.map(Into::into))
}

fn ms(value: i64, tz: Option<&str>) -> ScalarValue {
    ScalarValue::TimestampMillisecond(Some(value), tz.map(Into::into))
}

fn sec(value: i64, tz: Option<&str>) -> ScalarValue {
    ScalarValue::TimestampSecond(Some(value), tz.map(Into::into))
}

fn ts_cast(name: &str, unit: TimeUnit, tz: Option<&str>) -> Expr {
    Expr::Cast(Cast::new(
        Box::new(col(name)),
        DataType::Timestamp(unit, tz.map(Into::into)),
    ))
}

#[test]
fn zoned_timestamp_literals_map_to_timestamptz_datums() {
    for tz in ["UTC", "+00:00", "America/New_York"] {
        assert_eq!(
            super::scalar_value_to_datum(&us(V2_MICROS, Some(tz))),
            Some(Datum::timestamptz_micros(V2_MICROS)),
            "zone {tz} is a display label; the micros are the UTC instant"
        );
        assert_eq!(
            super::scalar_value_to_datum(&ns(V2_NANOS, Some(tz))),
            Some(Datum::timestamptz_nanos(V2_NANOS)),
            "zone {tz} is a display label; the nanos are the UTC instant"
        );
    }
}

#[test]
fn zoneless_timestamp_literals_keep_zoneless_datums() {
    assert_eq!(
        super::scalar_value_to_datum(&us(V2_MICROS, None)),
        Some(Datum::timestamp_micros(V2_MICROS))
    );
    assert_eq!(
        super::scalar_value_to_datum(&ns(V2_NANOS, None)),
        Some(Datum::timestamp_nanos(V2_NANOS))
    );
    assert_eq!(
        super::scalar_value_to_datum(&ScalarValue::TimestampMicrosecond(None, Some("UTC".into()))),
        None
    );
}

#[test]
fn milli_and_second_literals_widen_exactly() {
    assert_eq!(
        super::scalar_value_to_datum(&ms(1_703_000_000_000, Some("UTC"))),
        Some(Datum::timestamptz_micros(1_703_000_000_000_000))
    );
    assert_eq!(
        super::scalar_value_to_datum(&ms(1_703_000_000_000, None)),
        Some(Datum::timestamp_micros(1_703_000_000_000_000))
    );
    assert_eq!(
        super::scalar_value_to_datum(&sec(1_703_000_000, Some("+00:00"))),
        Some(Datum::timestamptz_micros(1_703_000_000_000_000))
    );
    assert_eq!(
        super::scalar_value_to_datum(&sec(1_703_000_000, None)),
        Some(Datum::timestamp_micros(1_703_000_000_000_000))
    );
}

#[test]
fn milli_and_second_literals_past_the_micros_range_are_not_pushed() {
    for value in [i64::MAX, i64::MIN] {
        assert_eq!(super::scalar_value_to_datum(&ms(value, None)), None);
        assert_eq!(super::scalar_value_to_datum(&ms(value, Some("UTC"))), None);
        assert_eq!(super::scalar_value_to_datum(&sec(value, None)), None);
        assert_eq!(super::scalar_value_to_datum(&sec(value, Some("UTC"))), None);
    }
}

#[test]
fn zoned_literals_push_onto_timestamptz_columns() {
    for tz in ["UTC", "+00:00", "America/New_York"] {
        assert_eq!(
            push(col("ts").gt_eq(lit(us(V2_MICROS, Some(tz))))),
            Some(
                Reference::new("ts").greater_than_or_equal_to(Datum::timestamptz_micros(V2_MICROS))
            ),
            "ts >= lit@{tz}"
        );
    }
    assert_eq!(
        push(lit(us(V2_MICROS, Some("UTC"))).lt_eq(col("ts"))),
        Some(Reference::new("ts").greater_than_or_equal_to(Datum::timestamptz_micros(V2_MICROS)))
    );
    assert_eq!(
        push(col("ts").eq(lit(us(V2_MICROS, Some("+00:00"))))),
        Some(Reference::new("ts").equal_to(Datum::timestamptz_micros(V2_MICROS)))
    );
    assert_eq!(
        push(col("ts").lt(lit(us(V2_MICROS, Some("America/New_York"))))),
        Some(Reference::new("ts").less_than(Datum::timestamptz_micros(V2_MICROS)))
    );
    assert_eq!(
        push(col("ts").in_list(
            vec![
                lit(us(V2_MICROS, Some("UTC"))),
                lit(us(V2_MICROS + 1, Some("+00:00")))
            ],
            false
        )),
        Some(Reference::new("ts").is_in([
            Datum::timestamptz_micros(V2_MICROS),
            Datum::timestamptz_micros(V2_MICROS + 1)
        ]))
    );
    assert_eq!(
        push(
            col("ts")
                .gt_eq(lit(us(V2_MICROS, Some("UTC"))))
                .and(col("ts").lt_eq(lit(us(V2_MICROS + 10, Some("UTC")))))
        ),
        Some(
            Reference::new("ts")
                .greater_than_or_equal_to(Datum::timestamptz_micros(V2_MICROS))
                .and(
                    Reference::new("ts")
                        .less_than_or_equal_to(Datum::timestamptz_micros(V2_MICROS + 10))
                )
        )
    );
    assert_eq!(
        push(col("tsz").gt_eq(lit(ns(V2_NANOS, Some("UTC"))))),
        Some(Reference::new("tsz").greater_than_or_equal_to(Datum::timestamptz_nanos(V2_NANOS)))
    );
    assert_eq!(
        push(col("ts").gt_eq(lit(us(-86_400_000_000, Some("UTC"))))),
        Some(
            Reference::new("ts")
                .greater_than_or_equal_to(Datum::timestamptz_micros(-86_400_000_000))
        )
    );
}

#[test]
fn zoneless_literals_still_push_onto_zoneless_columns() {
    assert_eq!(
        push(col("tsn").gt_eq(lit(us(V2_MICROS, None)))),
        Some(Reference::new("tsn").greater_than_or_equal_to(Datum::timestamp_micros(V2_MICROS)))
    );
    assert_eq!(
        push(col("tsnn").gt_eq(lit(ns(V2_NANOS, None)))),
        Some(Reference::new("tsnn").greater_than_or_equal_to(Datum::timestamp_nanos(V2_NANOS)))
    );
}

#[test]
fn cross_zone_timestamp_comparisons_stay_unpushed() {
    let cases = vec![
        col("ts").gt_eq(lit(us(V2_MICROS, None))),
        col("tsn").gt_eq(lit(us(V2_MICROS, Some("UTC")))),
        col("ts").gt_eq(lit(ns(V2_NANOS, Some("UTC")))),
        col("tsn").gt_eq(lit(ns(V2_NANOS, Some("UTC")))),
        col("tsz").gt_eq(lit(us(V2_MICROS, Some("UTC")))),
        col("tsnn").gt_eq(lit(us(V2_MICROS, Some("UTC")))),
        col("ts").gt_eq(lit(ns(V2_NANOS, None))),
        col("tsn").in_list(vec![lit(us(1, None)), lit(us(2, Some("UTC")))], false),
    ];
    for expr in cases {
        assert_eq!(push(expr.clone()), None, "{expr}");
    }
}

#[test]
fn zone_string_only_timestamp_cast_on_a_column_strips() {
    for target_tz in ["+00:00", "America/New_York"] {
        let expr = ts_cast("ts", TimeUnit::Microsecond, Some(target_tz))
            .gt_eq(lit(us(V2_MICROS, Some(target_tz))));
        assert_eq!(
            push(expr),
            Some(
                Reference::new("ts").greater_than_or_equal_to(Datum::timestamptz_micros(V2_MICROS))
            ),
            "CAST(ts AS Timestamp(us, {target_tz})) keeps the same instants"
        );
    }
    let expr = ts_cast("tsz", TimeUnit::Nanosecond, Some("+00:00"))
        .gt_eq(lit(ns(V2_NANOS, Some("+00:00"))));
    assert_eq!(
        push(expr),
        Some(Reference::new("tsz").greater_than_or_equal_to(Datum::timestamptz_nanos(V2_NANOS)))
    );
}

#[test]
fn zone_or_unit_changing_timestamp_casts_stay_unpushed() {
    let cases = vec![
        ts_cast("ts", TimeUnit::Microsecond, None).gt_eq(lit(us(V2_MICROS, None))),
        ts_cast("tsn", TimeUnit::Microsecond, Some("UTC")).gt_eq(lit(us(V2_MICROS, Some("UTC")))),
        ts_cast("ts", TimeUnit::Nanosecond, Some("UTC")).gt_eq(lit(ns(V2_NANOS, Some("UTC")))),
        ts_cast("ts", TimeUnit::Millisecond, Some("UTC")).gt_eq(lit(us(V2_MICROS, Some("UTC")))),
        ts_cast("tsn", TimeUnit::Nanosecond, None).gt_eq(lit(ns(V2_NANOS, None))),
        ts_cast("tsz", TimeUnit::Microsecond, Some("UTC")).gt_eq(lit(us(V2_MICROS, Some("UTC")))),
        ts_cast("tsz", TimeUnit::Microsecond, Some("+00:00"))
            .gt_eq(lit(us(V2_MICROS, Some("+00:00")))),
    ];
    for expr in cases {
        assert_eq!(push(expr.clone()), None, "{expr}");
    }
}

#[test]
fn cast_wrapped_timestamp_literal_converts_then_pushes() {
    let inner = Expr::Literal(us(V2_MICROS, Some("+00:00")), None);
    let expr = col("ts").gt_eq(Expr::Cast(Cast::new(
        Box::new(inner),
        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
    )));
    assert_eq!(
        push(expr),
        Some(Reference::new("ts").greater_than_or_equal_to(Datum::timestamptz_micros(V2_MICROS)))
    );
}

#[tokio::test]
async fn single_stream_scan_reads_rows_behind_a_zoned_literal() {
    use std::collections::HashMap;
    use std::sync::Arc;

    use datafusion::arrow::array::{Array, Int64Array, RecordBatch, TimestampMicrosecondArray};
    use datafusion::arrow::datatypes::{
        DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
    };
    use datafusion::datasource::MemTable;
    use datafusion::execution::TaskContext;
    use datafusion::physical_plan::ExecutionPlan;
    use datafusion::prelude::SessionContext;
    use futures::TryStreamExt;
    use iceberg::io::LocalFsStorageFactory;
    use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
    use tempfile::TempDir;

    use crate::IcebergCatalogProvider;
    use crate::physical_plan::scan::IcebergTableScan;

    let warehouse = TempDir::new().expect("warehouse");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                warehouse
                    .path()
                    .to_str()
                    .expect("warehouse path is UTF-8")
                    .to_string(),
            )]),
        )
        .await
        .expect("memory catalog");
    let namespace = NamespaceIdent::new("ts_tz_single".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let schema = IcebergSchema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "ts", Type::Primitive(PrimitiveType::Timestamptz)).into(),
        ])
        .build()
        .expect("schema");
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("tt".to_string())
                .location(format!("{}/tt", warehouse.path().to_str().expect("utf8")))
                .schema(schema)
                .build(),
        )
        .await
        .expect("create table");
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "catalog",
        Arc::new(
            IcebergCatalogProvider::try_new(catalog.clone())
                .await
                .expect("catalog provider"),
        ),
    );
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDataType::Int64, false),
        ArrowField::new(
            "ts",
            ArrowDataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
    ]));
    for (id, micros) in [(1_i64, 1_700_000_000_000_000_i64), (2, V2_MICROS)] {
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int64Array::from(vec![id])),
            Arc::new(TimestampMicrosecondArray::from(vec![micros]).with_timezone("UTC")),
        ])
        .expect("row batch");
        ctx.register_table(
            "src",
            Arc::new(MemTable::try_new(arrow_schema.clone(), vec![vec![batch]]).expect("memtable")),
        )
        .expect("register src");
        ctx.sql("INSERT INTO catalog.ts_tz_single.tt SELECT id, ts FROM src")
            .await
            .expect("insert sql")
            .collect()
            .await
            .expect("insert");
        ctx.deregister_table("src").expect("deregister src");
    }
    let table = catalog
        .load_table(&TableIdent::new(namespace, "tt".to_string()))
        .await
        .expect("load table");
    let scan_schema = Arc::new(
        iceberg::arrow::schema_to_arrow_schema(table.metadata().current_schema())
            .expect("arrow schema"),
    );
    let filter = col("ts").gt_eq(lit(us(V2_MICROS, Some("UTC"))));
    let scan = IcebergTableScan::new(table, None, false, scan_schema, None, &[filter], None)
        .expect("scan builds");
    assert_eq!(
        scan.predicates(),
        Some(&Reference::new("ts").greater_than_or_equal_to(Datum::timestamptz_micros(V2_MICROS)))
    );
    let batches: Vec<RecordBatch> = scan
        .execute(0, Arc::new(TaskContext::default()))
        .expect("execute")
        .try_collect()
        .await
        .expect("collect");
    let ids: Vec<i64> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id is Int64")
                .iter()
                .map(|value| value.expect("id is a required column"))
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(ids, vec![2]);
}
