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

use std::sync::Arc;

use super::*;
use crate::spec::{
    ListType, MapType, NestedField, PrimitiveLiteral, PrimitiveType, Schema, Type,
    METRICS_MODE_COLUMN_CONF_PREFIX, METRICS_MODE_DEFAULT_KEY,
};

fn wide_schema() -> Schema {
    Schema::builder()
        .with_schema_id(1)
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Int),
            )),
            Arc::new(NestedField::optional(
                2,
                "f",
                Type::Primitive(PrimitiveType::Float),
            )),
            Arc::new(NestedField::optional(
                3,
                "s",
                Type::Primitive(PrimitiveType::String),
            )),
            Arc::new(NestedField::optional(
                4,
                "b",
                Type::Primitive(PrimitiveType::Binary),
            )),
            Arc::new(NestedField::optional(
                5,
                "l",
                Type::List(ListType {
                    element_field: NestedField::list_element(
                        6,
                        Type::Primitive(PrimitiveType::Int),
                        false,
                    )
                    .into(),
                }),
            )),
            Arc::new(NestedField::optional(
                7,
                "m",
                Type::Map(MapType {
                    key_field: NestedField::map_key_element(
                        8,
                        Type::Primitive(PrimitiveType::String),
                    )
                    .into(),
                    value_field: NestedField::map_value_element(
                        9,
                        Type::Primitive(PrimitiveType::Int),
                        false,
                    )
                    .into(),
                }),
            )),
            Arc::new(NestedField::optional(
                10,
                "st",
                Type::Struct(StructType::new(vec![
                    NestedField::optional(11, "x", Type::Primitive(PrimitiveType::Int)).into(),
                ])),
            )),
        ])
        .build()
        .expect("build the wide schema")
}

fn primitive(literal: PrimitiveLiteral) -> Option<Literal> {
    Some(Literal::Primitive(literal))
}

fn wide_row(
    id: i32,
    float_value: Option<f32>,
    string_value: Option<&str>,
    binary_value: Option<Vec<u8>>,
    nested_x: Option<Option<i32>>,
) -> Literal {
    let nested = nested_x.map(|x| {
        Literal::Struct(Struct::from_iter(vec![
            x.map(|value| Literal::Primitive(PrimitiveLiteral::Int(value))),
        ]))
    });
    Literal::Struct(Struct::from_iter(vec![
        primitive(PrimitiveLiteral::Int(id)),
        float_value.map(|v| Literal::Primitive(PrimitiveLiteral::Float(v.into()))),
        string_value.map(|v| Literal::Primitive(PrimitiveLiteral::String(v.to_string()))),
        binary_value.map(|v| Literal::Primitive(PrimitiveLiteral::Binary(v))),
        None,
        None,
        nested,
    ]))
}

#[test]
fn test_only_struct_descended_field_ids_carry_counts() {
    let schema = wide_schema();
    let mut collector = OrcMetricsCollector::new(&schema, &MetricsConfig::default());
    collector.observe_row(&schema, Some(&wide_row(1, Some(1.5), Some("a"), None, Some(Some(7)))));
    let metrics = collector.build();

    let mut ids: Vec<i32> = metrics.value_counts.keys().copied().collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![1, 2, 3, 4, 5, 7, 10, 11],
        "list elements (6), map keys (8) and map values (9) carry no counts"
    );
    assert!(metrics.value_counts.values().all(|count| *count == 1));
}

#[test]
fn test_null_counts_include_rows_whose_ancestor_struct_is_null() {
    let schema = wide_schema();
    let mut collector = OrcMetricsCollector::new(&schema, &MetricsConfig::default());
    collector.observe_row(&schema, Some(&wide_row(1, None, None, None, Some(Some(7)))));
    collector.observe_row(&schema, Some(&wide_row(2, None, None, None, Some(None))));
    collector.observe_row(&schema, Some(&wide_row(3, None, None, None, None)));
    let metrics = collector.build();

    assert_eq!(metrics.value_counts.get(&11), Some(&3));
    assert_eq!(
        metrics.null_value_counts.get(&11),
        Some(&2),
        "one explicit null plus one whose parent struct is null"
    );
    assert_eq!(metrics.null_value_counts.get(&10), Some(&1));
    assert_eq!(metrics.null_value_counts.get(&1), Some(&0));
}

#[test]
fn test_binary_columns_carry_no_bounds_because_orc_stats_have_none() {
    let schema = wide_schema();
    let mut collector = OrcMetricsCollector::new(&schema, &MetricsConfig::default());
    collector.observe_row(
        &schema,
        Some(&wide_row(1, None, Some("a"), Some(vec![1, 2, 3]), None)),
    );
    let metrics = collector.build();

    assert!(metrics.lower_bounds.contains_key(&3), "string keeps bounds");
    assert!(
        !metrics.lower_bounds.contains_key(&4),
        "ORC BinaryColumnStatistics carries no minimum"
    );
    assert!(!metrics.upper_bounds.contains_key(&4));
}

#[test]
fn test_containers_carry_counts_but_never_bounds() {
    let schema = wide_schema();
    let mut collector = OrcMetricsCollector::new(&schema, &MetricsConfig::default());
    collector.observe_row(&schema, Some(&wide_row(1, None, None, None, Some(Some(1)))));
    let metrics = collector.build();

    assert!(metrics.value_counts.contains_key(&5));
    assert!(metrics.value_counts.contains_key(&7));
    assert!(metrics.value_counts.contains_key(&10));
    for id in [5, 7, 10] {
        assert!(!metrics.lower_bounds.contains_key(&id));
        assert!(!metrics.upper_bounds.contains_key(&id));
    }
}

#[test]
fn test_nan_is_counted_and_excluded_from_the_float_bounds() {
    let schema = wide_schema();
    let mut collector = OrcMetricsCollector::new(&schema, &MetricsConfig::default());
    collector.observe_row(&schema, Some(&wide_row(1, Some(3.5), None, None, None)));
    collector.observe_row(&schema, Some(&wide_row(2, Some(f32::NAN), None, None, None)));
    collector.observe_row(&schema, Some(&wide_row(3, Some(-1.5), None, None, None)));
    let metrics = collector.build();

    assert_eq!(metrics.nan_value_counts.get(&2), Some(&1));
    assert_eq!(
        metrics.lower_bounds.get(&2).map(Datum::literal),
        Some(&PrimitiveLiteral::Float((-1.5f32).into()))
    );
    assert_eq!(
        metrics.upper_bounds.get(&2).map(Datum::literal),
        Some(&PrimitiveLiteral::Float(3.5f32.into()))
    );
}

#[test]
fn test_nan_counts_are_zero_for_a_float_column_with_no_nan_and_absent_for_other_types() {
    let schema = wide_schema();
    let mut collector = OrcMetricsCollector::new(&schema, &MetricsConfig::default());
    collector.observe_row(&schema, Some(&wide_row(1, Some(3.5), None, None, None)));
    let metrics = collector.build();

    assert_eq!(metrics.nan_value_counts.get(&2), Some(&0));
    assert_eq!(metrics.nan_value_counts.len(), 1);
}

#[test]
fn test_metrics_mode_none_drops_the_column_entirely() {
    let schema = wide_schema();
    let config = MetricsConfig::from_properties(&HashMap::from([(
        format!("{METRICS_MODE_COLUMN_CONF_PREFIX}s"),
        "none".to_string(),
    )]))
    .expect("parse the metrics config");
    let mut collector = OrcMetricsCollector::new(&schema, &config);
    collector.observe_row(&schema, Some(&wide_row(1, None, Some("abc"), None, None)));
    let metrics = collector.build();

    assert!(!metrics.value_counts.contains_key(&3));
    assert!(!metrics.null_value_counts.contains_key(&3));
    assert!(!metrics.lower_bounds.contains_key(&3));
    assert!(metrics.value_counts.contains_key(&1));
}

#[test]
fn test_metrics_mode_counts_keeps_counts_and_drops_bounds() {
    let schema = wide_schema();
    let config = MetricsConfig::from_properties(&HashMap::from([(
        METRICS_MODE_DEFAULT_KEY.to_string(),
        "counts".to_string(),
    )]))
    .expect("parse the metrics config");
    let mut collector = OrcMetricsCollector::new(&schema, &config);
    collector.observe_row(&schema, Some(&wide_row(1, Some(1.5), Some("abc"), None, None)));
    let metrics = collector.build();

    assert_eq!(metrics.value_counts.get(&3), Some(&1));
    assert!(metrics.lower_bounds.is_empty());
    assert!(metrics.upper_bounds.is_empty());
}

#[test]
fn test_string_bounds_are_truncated_by_the_configured_mode() {
    let schema = wide_schema();
    let long = "a".repeat(40);
    let mut collector = OrcMetricsCollector::new(&schema, &MetricsConfig::default());
    collector.observe_row(&schema, Some(&wide_row(1, None, Some(&long), None, None)));
    let metrics = collector.build();

    let lower = metrics.lower_bounds.get(&3).expect("a string lower bound");
    match lower.literal() {
        PrimitiveLiteral::String(value) => assert_eq!(
            value.len(),
            16,
            "the default metrics mode truncates string bounds to 16"
        ),
        other => panic!("expected a string bound, got {other:?}"),
    }
}

#[test]
fn test_column_sizes_map_orc_column_indices_back_to_field_ids() {
    let schema = wide_schema();
    let orc_schema = crate::writer::file_writer::orc_writer::orc_type::build_orc_schema(&schema)
        .expect("map the schema");
    let mut collector = OrcMetricsCollector::new(&schema, &MetricsConfig::default());
    collector.observe_row(&schema, Some(&wide_row(1, None, None, None, None)));
    collector.observe_stripe_column_sizes(&orc_schema, &HashMap::from([(1usize, 11u64)]));
    collector.observe_stripe_column_sizes(&orc_schema, &HashMap::from([(1usize, 4u64)]));
    let metrics = collector.build();

    assert_eq!(
        metrics.column_sizes.get(&1),
        Some(&15),
        "stripe sizes accumulate per field id"
    );
}
