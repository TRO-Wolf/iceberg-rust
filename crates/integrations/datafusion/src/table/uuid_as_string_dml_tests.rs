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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, Int32Array, ListArray, MapArray, RecordBatch, StringArray, UInt64Array,
};
use datafusion::catalog::TableProvider;
use datafusion::common::Column;
use datafusion::logical_expr::{BinaryExpr, Expr, Operator, col, lit};
use datafusion::prelude::SessionContext;
use datafusion::scalar::ScalarValue;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{ListType, MapType, NestedField, PrimitiveType, Schema, Type};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use tempfile::TempDir;

use super::uuid_text::parse_uuid_text;
use super::*;

const U0: &str = "00000000-0000-0000-0000-000000000001";
const U1: &str = "123e4567-e89b-12d3-a456-426614174000";
const UP: &str = "123E4567-E89B-12D3-A456-426614174000";
const U2: &str = "123e4567-e89b-12d3-a456-4266141740ff";
const SHORT: &str = "00000001-0002-0003-0004-000000000005";
const MODES: [&str; 2] = ["copy-on-write", "merge-on-read"];

struct Fixture {
    provider: Arc<IcebergTableProvider>,
    _warehouse: TempDir,
}

async fn fixture(mode: &str, fields: Vec<NestedField>, as_string: bool) -> Fixture {
    let warehouse = TempDir::new().expect("warehouse");
    let root = warehouse.path().to_str().expect("utf8").to_string();
    let catalog: Arc<dyn Catalog> = Arc::new(
        MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), root.clone())]),
            )
            .await
            .expect("catalog"),
    );
    let namespace = NamespaceIdent::new("ns".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(fields.into_iter().map(Arc::new).collect::<Vec<_>>())
        .build()
        .expect("schema");
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("t".to_string())
                .location(format!("{root}/t"))
                .schema(schema)
                .properties(HashMap::from([
                    ("write.delete.mode".to_string(), mode.to_string()),
                    ("write.update.mode".to_string(), mode.to_string()),
                ]))
                .build(),
        )
        .await
        .expect("table");
    let provider = Arc::new(
        IcebergTableProvider::try_new(catalog, namespace, "t")
            .await
            .expect("provider")
            .with_uuid_as_string(as_string),
    );
    Fixture {
        provider,
        _warehouse: warehouse,
    }
}

fn scalar_fields() -> Vec<NestedField> {
    vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)),
        NestedField::optional(2, "u", Type::Primitive(PrimitiveType::Uuid)),
        NestedField::optional(3, "t", Type::Primitive(PrimitiveType::String)),
    ]
}

async fn spark_rows(mode: &str) -> Fixture {
    let fixture = fixture(mode, scalar_fields(), true).await;
    sql(
        &fixture,
        &format!(
            "INSERT INTO t VALUES (1, '{U1}', '{UP}'), (2, '{U2}', 'x'), (3, NULL, 'y'), (4, '{U0}', '{U1}')"
        ),
    )
    .await
    .expect("seed insert");
    fixture
}

async fn sql(fixture: &Fixture, query: &str) -> std::result::Result<Vec<RecordBatch>, String> {
    let ctx = SessionContext::new();
    ctx.register_table("t", fixture.provider.clone() as Arc<dyn TableProvider>)
        .expect("register");
    let frame = ctx.sql(query).await.map_err(|e| e.to_string())?;
    frame.collect().await.map_err(|e| e.to_string())
}

async fn count(fixture: &Fixture, query: &str) -> u64 {
    sql(fixture, query)
        .await
        .unwrap_or_else(|e| panic!("`{query}`: {e}"))
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("count column")
                .values()
                .to_vec()
        })
        .sum()
}

async fn ids(fixture: &Fixture, query: &str) -> Vec<i32> {
    sql(fixture, query)
        .await
        .unwrap_or_else(|e| panic!("`{query}`: {e}"))
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("id column")
                .values()
                .to_vec()
        })
        .collect()
}

async fn texts(fixture: &Fixture, query: &str) -> Vec<Option<String>> {
    let mut out = Vec::new();
    for batch in sql(fixture, query)
        .await
        .unwrap_or_else(|e| panic!("`{query}`: {e}"))
    {
        let strings = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("text column");
        for row in 0..strings.len() {
            out.push(
                strings
                    .is_valid(row)
                    .then(|| strings.value(row).to_string()),
            );
        }
    }
    out
}

async fn surviving(fixture: &Fixture) -> Vec<i32> {
    ids(fixture, "SELECT id FROM t ORDER BY id").await
}

#[tokio::test]
async fn uuid_delete_follows_spark_string_semantics() {
    let cases: [(String, u64, Vec<i32>); 18] = [
        (format!("u = '{UP}'"), 0, vec![1, 2, 3, 4]),
        (format!("u <> '{UP}'"), 3, vec![3]),
        (format!("u <> '{U1}'"), 2, vec![1, 3]),
        (format!("u IN ('{UP}')"), 0, vec![1, 2, 3, 4]),
        (format!("u NOT IN ('{UP}')"), 3, vec![3]),
        (format!("u NOT IN ('{U1}')"), 2, vec![1, 3]),
        (format!("u < '{UP}'"), 1, vec![1, 2, 3]),
        (format!("u > '{UP}'"), 2, vec![3, 4]),
        (format!("u < '{U2}'"), 2, vec![2, 3]),
        ("u IS NULL".to_string(), 1, vec![1, 2, 4]),
        ("u <=> NULL".to_string(), 1, vec![1, 2, 4]),
        ("u LIKE '%4266%'".to_string(), 2, vec![3, 4]),
        ("u = '1-2-3-4-5'".to_string(), 0, vec![1, 2, 3, 4]),
        ("u < '1-2-3-4-5'".to_string(), 1, vec![1, 2, 3]),
        (format!("upper(u) = '{UP}'"), 1, vec![2, 3, 4]),
        ("u = 'abc' OR upper(t) = 'X'".to_string(), 1, vec![1, 3, 4]),
        ("u LIKE '123e%' OR upper(t) = 'X'".to_string(), 2, vec![
            3, 4,
        ]),
        ("u < 'abc' AND upper(t) = 'X'".to_string(), 1, vec![1, 3, 4]),
    ];
    for mode in MODES {
        for (predicate, deleted, kept) in &cases {
            let fixture = spark_rows(mode).await;
            let query = format!("DELETE FROM t WHERE {predicate}");
            assert_eq!(count(&fixture, &query).await, *deleted, "{mode}: {query}");
            assert_eq!(surviving(&fixture).await, *kept, "{mode}: {query}");
            let select = format!("SELECT id FROM t WHERE {predicate} ORDER BY id");
            assert!(ids(&fixture, &select).await.is_empty(), "{mode}: {select}");
        }
    }
}

#[tokio::test]
async fn uuid_delete_refuses_literals_iceberg_cannot_bind() {
    let cases = [
        ("u = 'abc'", "Invalid UUID string: abc"),
        ("u > '2'", "Invalid UUID string: 2"),
        ("u = 'abc' OR id = 1", "Invalid UUID string: abc"),
        ("NOT (u = 'abc')", "Invalid UUID string: abc"),
        ("u IN ('abc')", "Invalid UUID string: abc"),
        ("u LIKE 'abc'", "Invalid UUID string: abc"),
        ("u NOT LIKE 'abc'", "Invalid UUID string: abc"),
        ("u NOT IN ('abc')", "Invalid UUID string: abc"),
        ("u <=> 'abc'", "Invalid UUID string: abc"),
        ("u IS NOT DISTINCT FROM 'abc'", "Invalid UUID string: abc"),
        ("u = 'abc' AND NOT (u <=> NULL)", "Invalid UUID string: abc"),
        (
            "u LIKE '123e%'",
            "Term for STARTS_WITH or NOT_STARTS_WITH must produce a string: ref(id=2, accessor-type=uuid): uuid",
        ),
        (
            "u NOT LIKE '123e%'",
            "Term for STARTS_WITH or NOT_STARTS_WITH must produce a string: ref(id=2, accessor-type=uuid): uuid",
        ),
    ];
    for mode in MODES {
        for (predicate, message) in cases {
            let fixture = spark_rows(mode).await;
            let query = format!("DELETE FROM t WHERE {predicate}");
            let err = sql(&fixture, &query)
                .await
                .expect_err("Spark refuses this DELETE");
            assert!(err.contains(message), "{mode}: {query} -> {err}");
            assert_eq!(surviving(&fixture).await, vec![1, 2, 3, 4]);
        }
    }
}

#[tokio::test]
async fn uuid_delete_conversion_follows_this_engines_optimizer_shape() {
    let kept = [
        "u = 'abc' AND id + 0 = 1",
        "u = 'abc' AND id = 1.0",
        "u = 'abc' AND id IN (1, 2.5)",
    ];
    for mode in MODES {
        let fixture = spark_rows(mode).await;
        let query = "DELETE FROM t WHERE CAST(id AS STRING) = '1' AND u = 'abc'";
        let err = sql(&fixture, query)
            .await
            .expect_err("the unwrapped cast converts here");
        assert!(
            err.contains("Invalid UUID string: abc"),
            "{mode}: {query} -> {err}"
        );
        assert_eq!(surviving(&fixture).await, vec![1, 2, 3, 4]);
        for predicate in kept {
            let fixture = spark_rows(mode).await;
            let query = format!("DELETE FROM t WHERE {predicate}");
            assert_eq!(count(&fixture, &query).await, 0, "{mode}: {query}");
            assert_eq!(
                surviving(&fixture).await,
                vec![1, 2, 3, 4],
                "{mode}: {query}"
            );
        }
    }
}

#[tokio::test]
async fn uuid_update_where_follows_spark_string_semantics() {
    let cases: [(String, u64, Vec<i32>); 7] = [
        ("u = 'abc'".to_string(), 0, vec![]),
        ("u > '2'".to_string(), 0, vec![]),
        ("u LIKE '123e%'".to_string(), 2, vec![1, 2]),
        (format!("u = '{UP}'"), 0, vec![]),
        (format!("u <> '{UP}'"), 3, vec![1, 2, 4]),
        (format!("u IN ('{U1}', '{UP}')"), 1, vec![1]),
        ("u IS NULL".to_string(), 1, vec![3]),
    ];
    for mode in MODES {
        for (predicate, updated, hit) in &cases {
            let fixture = spark_rows(mode).await;
            let query = format!("UPDATE t SET t = 'hit' WHERE {predicate}");
            assert_eq!(count(&fixture, &query).await, *updated, "{mode}: {query}");
            let hits = ids(&fixture, "SELECT id FROM t WHERE t = 'hit' ORDER BY id").await;
            assert_eq!(hits, *hit, "{mode}: {query}");
        }
    }
}

#[tokio::test]
async fn uuid_update_assigns_null_columns_and_expressions() {
    let cases: [(String, Vec<Option<&str>>); 5] = [
        ("u = NULL".to_string(), vec![None, Some(U2), None, None]),
        ("u = t".to_string(), vec![
            Some(U1),
            Some(U2),
            None,
            Some(U1),
        ]),
        ("u = upper(u)".to_string(), vec![
            Some(U1),
            Some(U2),
            None,
            Some(U0),
        ]),
        (format!("u = '{UP}'"), vec![
            Some(U1),
            Some(U2),
            None,
            Some(U1),
        ]),
        ("u = '1-2-3-4-5'".to_string(), vec![
            Some(SHORT),
            Some(U2),
            None,
            Some(SHORT),
        ]),
    ];
    for mode in MODES {
        for (assignment, expected) in &cases {
            let fixture = spark_rows(mode).await;
            let query = format!("UPDATE t SET {assignment} WHERE id IN (1, 4)");
            assert_eq!(count(&fixture, &query).await, 2, "{mode}: {query}");
            let expected: Vec<Option<String>> = expected
                .iter()
                .map(|value| value.map(str::to_string))
                .collect();
            assert_eq!(
                texts(&fixture, "SELECT u FROM t ORDER BY id").await,
                expected,
                "{mode}: {query}"
            );
        }
        let fixture = spark_rows(mode).await;
        for (assignment, message) in [
            ("u = concat(t, 'z')", "UUID string too large"),
            ("u = 'not-a-uuid'", "Invalid UUID string: not-a-uuid"),
        ] {
            let query = format!("UPDATE t SET {assignment} WHERE id IN (1, 4)");
            let err = sql(&fixture, &query)
                .await
                .expect_err("invalid uuid refuses");
            assert!(err.contains(message), "{mode}: {query} -> {err}");
        }
        assert_eq!(texts(&fixture, "SELECT u FROM t ORDER BY id").await, vec![
            Some(U1.to_string()),
            Some(U2.to_string()),
            None,
            Some(U0.to_string())
        ]);
    }
}

#[tokio::test]
async fn uuid_select_keeps_case_sensitive_answers_for_negated_and_range_literals() {
    let fixture = fixture("copy-on-write", scalar_fields(), true).await;
    sql(&fixture, &format!("INSERT INTO t VALUES (1, '{U1}', 'a')"))
        .await
        .expect("first file");
    sql(&fixture, &format!("INSERT INTO t VALUES (2, '{U2}', 'b')"))
        .await
        .expect("second file");
    sql(&fixture, "INSERT INTO t VALUES (3, NULL, 'c')")
        .await
        .expect("third file");
    let u2_upper = U2.to_uppercase();
    let cases = [
        (format!("u <> '{UP}'"), vec![1, 2]),
        (format!("NOT (u = '{UP}')"), vec![1, 2]),
        (format!("u NOT IN ('{UP}')"), vec![1, 2]),
        (format!("u NOT IN ('{UP}', '{u2_upper}')"), vec![1, 2]),
        (format!("u > '{UP}'"), vec![1, 2]),
        (format!("u >= '{UP}'"), vec![1, 2]),
        (format!("u < '{UP}'"), vec![]),
        (format!("u <= '{u2_upper}'"), vec![]),
        (format!("'{UP}' < u"), vec![1, 2]),
    ];
    for (predicate, expected) in cases {
        let query = format!("SELECT id FROM t WHERE {predicate} ORDER BY id");
        assert_eq!(ids(&fixture, &query).await, expected, "{query}");
    }
}

async fn planned_files(provider: &IcebergTableProvider, filters: &[Expr]) -> HashSet<String> {
    let ctx = SessionContext::new();
    let plan = provider
        .scan(&ctx.state(), None, filters, None)
        .await
        .expect("scan plans");
    plan.downcast_ref::<crate::physical_plan::IcebergTableScan>()
        .expect("IcebergTableScan")
        .partition_work()
        .iter()
        .flat_map(|work| work.tasks())
        .map(|task| task.data_file_path().to_string())
        .collect()
}

fn text_literal(text: &str) -> Expr {
    Expr::Literal(ScalarValue::Utf8(Some(text.to_string())), None)
}

fn compare(op: Operator, text: &str) -> Expr {
    Expr::BinaryExpr(BinaryExpr::new(
        Box::new(Expr::Column(Column::from_name("u"))),
        op,
        Box::new(text_literal(text)),
    ))
}

async fn two_file_fixture(as_string: bool) -> Fixture {
    let fixture = fixture("copy-on-write", scalar_fields(), true).await;
    sql(&fixture, &format!("INSERT INTO t VALUES (1, '{U1}', 'a')"))
        .await
        .expect("first file");
    sql(&fixture, &format!("INSERT INTO t VALUES (2, '{U2}', 'b')"))
        .await
        .expect("second file");
    if as_string {
        fixture
    } else {
        let catalog_provider = (*fixture.provider).clone().with_uuid_as_string(false);
        Fixture {
            provider: Arc::new(catalog_provider),
            _warehouse: fixture._warehouse,
        }
    }
}

#[tokio::test]
async fn uuid_delete_evaluates_rows_where_spark_drops_whole_files_by_bytes() {
    let upper = two_file_fixture(true).await;
    let query = format!("DELETE FROM t WHERE u = '{UP}'");
    assert_eq!(count(&upper, &query).await, 0, "{query}");
    assert_eq!(surviving(&upper).await, vec![1, 2]);
    let fixture = fixture("copy-on-write", scalar_fields(), true).await;
    for (id, value) in [(1, SHORT), (2, U2)] {
        sql(
            &fixture,
            &format!("INSERT INTO t VALUES ({id}, '{value}', 'a')"),
        )
        .await
        .expect("one file per row");
    }
    let query = "DELETE FROM t WHERE u = '1-2-3-4-5'";
    assert_eq!(count(&fixture, query).await, 0, "{query}");
    assert_eq!(surviving(&fixture).await, vec![1, 2]);
}

#[tokio::test]
async fn uuid_in_and_range_literals_prune_files() {
    let fixture = two_file_fixture(true).await;
    let all = planned_files(&fixture.provider, &[]).await;
    assert_eq!(all.len(), 2);
    let in_list = col("u").in_list(vec![text_literal(U1)], false);
    let cases = [
        (in_list, 1),
        (compare(Operator::Lt, U2), 1),
        (compare(Operator::Gt, U1), 1),
        (compare(Operator::NotEq, U1), 1),
        (compare(Operator::NotEq, UP), 2),
        (compare(Operator::Gt, UP), 2),
        (col("u").in_list(vec![text_literal(UP)], true), 2),
    ];
    for (filter, files) in cases {
        let planned = planned_files(&fixture.provider, std::slice::from_ref(&filter)).await;
        assert_eq!(planned.len(), files, "{filter}");
        assert!(planned.is_subset(&all));
    }
    let batches = ids(&fixture, &format!("SELECT id FROM t WHERE u IN ('{U1}')")).await;
    assert_eq!(batches, vec![1]);
    let batches = ids(&fixture, &format!("SELECT id FROM t WHERE u < '{U2}'")).await;
    assert_eq!(batches, vec![1]);
}

#[tokio::test]
async fn uuid_option_off_byte_literal_prunes_with_unchanged_rows() {
    let fixture = two_file_fixture(false).await;
    let all = planned_files(&fixture.provider, &[]).await;
    assert_eq!(all.len(), 2);
    let bytes = parse_uuid_text(U1).expect("uuid parses").to_vec();
    let filter = col("u").eq(lit(ScalarValue::FixedSizeBinary(16, Some(bytes))));
    let planned = planned_files(&fixture.provider, std::slice::from_ref(&filter)).await;
    assert_eq!(planned.len(), 1);
    let ctx = SessionContext::new();
    let frame = ctx
        .read_table(fixture.provider.clone() as Arc<dyn TableProvider>)
        .expect("frame")
        .filter(filter)
        .expect("filter")
        .select_columns(&["id"])
        .expect("project");
    let rows: Vec<i32> = frame
        .collect()
        .await
        .expect("rows")
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("id")
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(rows, vec![1]);
}

fn nested_fields() -> Vec<NestedField> {
    vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)),
        NestedField::optional(
            2,
            "us",
            Type::List(ListType::new(Arc::new(NestedField::list_element(
                3,
                Type::Primitive(PrimitiveType::Uuid),
                false,
            )))),
        ),
        NestedField::optional(
            4,
            "m",
            Type::Map(MapType::new(
                Arc::new(NestedField::map_key_element(
                    5,
                    Type::Primitive(PrimitiveType::Uuid),
                )),
                Arc::new(NestedField::map_value_element(
                    6,
                    Type::Primitive(PrimitiveType::Uuid),
                    false,
                )),
            )),
        ),
    ]
}

#[tokio::test]
async fn uuid_list_and_map_round_trip_as_lowercase_text() {
    let fixture = fixture("copy-on-write", nested_fields(), true).await;
    sql(
        &fixture,
        &format!("INSERT INTO t VALUES (1, make_array('{UP}', '{U2}'), MAP {{'{UP}': '{U2}'}})"),
    )
    .await
    .expect("nested insert");
    let batches = sql(&fixture, "SELECT us, m FROM t").await.expect("select");
    let list = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("list column");
    let elements = list.value(0);
    let elements = elements
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("list elements render as text");
    assert_eq!(
        (0..elements.len())
            .map(|row| elements.value(row).to_string())
            .collect::<Vec<_>>(),
        vec![U1.to_string(), U2.to_string()]
    );
    let map = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<MapArray>()
        .expect("map column");
    let keys = map
        .keys()
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("map keys render as text");
    let values = map
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("map values render as text");
    assert_eq!(keys.value(0), U1);
    assert_eq!(values.value(0), U2);
    let err = sql(
        &fixture,
        "INSERT INTO t VALUES (2, make_array('bad'), NULL)",
    )
    .await
    .expect_err("invalid list element refuses");
    assert!(err.contains("Invalid UUID string: bad"), "{err}");
    let err = sql(
        &fixture,
        &format!("INSERT INTO t VALUES (3, NULL, MAP {{'{U1}': 'bad'}})"),
    )
    .await
    .expect_err("invalid map value refuses");
    assert!(err.contains("Invalid UUID string: bad"), "{err}");
    assert_eq!(surviving(&fixture).await, vec![1]);
}

#[test]
fn uuid_parser_mirrors_java_uuid_from_string() {
    let render =
        |text: &str| parse_uuid_text(text).map(|bytes| super::uuid_text::render_uuid_text(&bytes));
    assert_eq!(render("1-2-3-4-5"), Ok(SHORT.to_string()));
    assert_eq!(render("+1-2-3-4-5"), Ok(SHORT.to_string()));
    assert_eq!(
        render("0-0-0-0-0"),
        Ok("00000000-0000-0000-0000-000000000000".to_string())
    );
    assert_eq!(
        render("123456789-2-3-4-5"),
        Ok("23456789-0002-0003-0004-000000000005".to_string())
    );
    assert_eq!(render(UP), Ok(U1.to_string()));
    for (input, message) in [
        (
            "123e4567-e89b-12d3-a456-4266141740000",
            "UUID string too large",
        ),
        (
            " 123e4567-e89b-12d3-a456-426614174000",
            "UUID string too large",
        ),
        ("1--3-4-5", "NumberFormatException: "),
        (
            "1-2-3-4-z",
            "NumberFormatException: Error at index 0 in: \"z\"",
        ),
        (
            "1-2-3-4-1234567890abcdef1",
            "NumberFormatException: Error at index 16 in: \"1234567890abcdef1\"",
        ),
        (
            " 1-2-3-4-5",
            "NumberFormatException: Error at index 0 in: \" 1\"",
        ),
        (
            "+-2-3-4-5",
            "NumberFormatException: Error at index 1 in: \"+\"",
        ),
        (
            "123e4567e89b12d3a456426614174000",
            "Invalid UUID string: 123e4567e89b12d3a456426614174000",
        ),
        ("1-2-3-4", "Invalid UUID string: 1-2-3-4"),
        ("1-2-3-4-5-6", "Invalid UUID string: 1-2-3-4-5-6"),
        ("abc", "Invalid UUID string: abc"),
        ("", "Invalid UUID string: "),
    ] {
        assert_eq!(render(input), Err(message.to_string()), "{input}");
    }
}

#[tokio::test]
async fn uuid_insert_accepts_java_short_groups_and_refuses_too_large() {
    let fixture = fixture("copy-on-write", scalar_fields(), true).await;
    sql(&fixture, "INSERT INTO t VALUES (1, '1-2-3-4-5', NULL)")
        .await
        .expect("Java-lenient short groups insert");
    assert_eq!(texts(&fixture, "SELECT u FROM t").await, vec![Some(
        SHORT.to_string()
    )]);
    let err = sql(
        &fixture,
        "INSERT INTO t VALUES (2, '123e4567-e89b-12d3-a456-4266141740000', NULL)",
    )
    .await
    .expect_err("37 characters refuse");
    assert!(err.contains("UUID string too large"), "{err}");
    let err = sql(&fixture, "INSERT INTO t VALUES (3, '1-2-3-4-z', NULL)")
        .await
        .expect_err("bad hex refuses");
    assert!(
        err.contains("NumberFormatException: Error at index 0 in: \"z\""),
        "{err}"
    );
    assert_eq!(surviving(&fixture).await, vec![1]);
}
