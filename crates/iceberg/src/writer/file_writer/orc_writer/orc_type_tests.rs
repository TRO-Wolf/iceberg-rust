use std::sync::Arc;

use super::*;
use crate::spec::{ListType, MapType, NestedField, Schema};

fn schema_of(fields: Vec<NestedField>) -> Schema {
    Schema::builder()
        .with_schema_id(1)
        .with_fields(fields.into_iter().map(Arc::new).collect::<Vec<_>>())
        .build()
        .expect("build the fixture schema")
}

fn attribute<'a>(orc_type: &'a OrcType, key: &str) -> Option<&'a str> {
    orc_type
        .attributes
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

type PrimitiveCase = (PrimitiveType, OrcKind, Vec<(&'static str, &'static str)>);

#[test]
fn test_every_iceberg_primitive_maps_to_java_s_orc_kind_and_attributes() {
    let cases: Vec<PrimitiveCase> = vec![
        (PrimitiveType::Boolean, OrcKind::Boolean, vec![]),
        (PrimitiveType::Int, OrcKind::Int, vec![]),
        (PrimitiveType::Long, OrcKind::Long, vec![(
            ICEBERG_LONG_TYPE_ATTRIBUTE,
            "LONG",
        )]),
        (PrimitiveType::Float, OrcKind::Float, vec![]),
        (PrimitiveType::Double, OrcKind::Double, vec![]),
        (PrimitiveType::Date, OrcKind::Date, vec![]),
        (PrimitiveType::Time, OrcKind::Long, vec![(
            ICEBERG_LONG_TYPE_ATTRIBUTE,
            "TIME",
        )]),
        (PrimitiveType::Timestamp, OrcKind::Timestamp, vec![(
            ICEBERG_TIMESTAMP_UNIT,
            "MICROS",
        )]),
        (
            PrimitiveType::Timestamptz,
            OrcKind::TimestampInstant,
            vec![(ICEBERG_TIMESTAMP_UNIT, "MICROS")],
        ),
        (PrimitiveType::TimestampNs, OrcKind::Timestamp, vec![(
            ICEBERG_TIMESTAMP_UNIT,
            "NANOS",
        )]),
        (
            PrimitiveType::TimestamptzNs,
            OrcKind::TimestampInstant,
            vec![(ICEBERG_TIMESTAMP_UNIT, "NANOS")],
        ),
        (PrimitiveType::String, OrcKind::String, vec![]),
        (PrimitiveType::Uuid, OrcKind::Binary, vec![(
            ICEBERG_BINARY_TYPE_ATTRIBUTE,
            "UUID",
        )]),
        (PrimitiveType::Fixed(12), OrcKind::Binary, vec![
            (ICEBERG_BINARY_TYPE_ATTRIBUTE, "FIXED"),
            (ICEBERG_FIELD_LENGTH, "12"),
        ]),
        (PrimitiveType::Binary, OrcKind::Binary, vec![(
            ICEBERG_BINARY_TYPE_ATTRIBUTE,
            "BINARY",
        )]),
        (
            PrimitiveType::Decimal {
                precision: 10,
                scale: 2,
            },
            OrcKind::Decimal,
            vec![],
        ),
    ];

    for (primitive, expected_kind, extras) in cases {
        let schema = schema_of(vec![NestedField::optional(
            7,
            "v",
            Type::Primitive(primitive.clone()),
        )]);
        let orc = build_orc_schema(&schema).expect("map the schema");
        assert_eq!(orc.types.len(), 2, "root + one column for {primitive}");
        let column = &orc.types[1];
        assert_eq!(column.kind, Some(expected_kind), "kind for {primitive}");
        assert_eq!(
            attribute(column, ICEBERG_ID_ATTRIBUTE),
            Some("7"),
            "iceberg.id for {primitive}"
        );
        assert_eq!(
            attribute(column, ICEBERG_REQUIRED_ATTRIBUTE),
            Some("false"),
            "iceberg.required for {primitive}"
        );
        for (key, value) in extras {
            assert_eq!(
                attribute(column, key),
                Some(value),
                "{key} for {primitive}"
            );
        }
        if let PrimitiveType::Decimal { precision, scale } = &primitive {
            assert_eq!(column.precision, Some(*precision));
            assert_eq!(column.scale, Some(*scale));
        }
    }
}

#[test]
fn test_required_attribute_follows_the_iceberg_schema() {
    let schema = schema_of(vec![NestedField::required(
        1,
        "id",
        Type::Primitive(PrimitiveType::Long),
    )]);
    let orc = build_orc_schema(&schema).expect("map the schema");
    assert_eq!(
        attribute(&orc.types[1], ICEBERG_REQUIRED_ATTRIBUTE),
        Some("true")
    );
}

#[test]
fn test_the_root_struct_carries_field_names_and_no_iceberg_attributes() {
    let schema = schema_of(vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)),
        NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)),
    ]);
    let orc = build_orc_schema(&schema).expect("map the schema");
    assert_eq!(orc.types[0].kind, Some(OrcKind::Struct));
    assert!(
        orc.types[0].attributes.is_empty(),
        "Java stamps no iceberg attribute on the ORC root"
    );
    assert_eq!(orc.types[0].field_names, vec![
        "id".to_string(),
        "data".to_string()
    ]);
    assert_eq!(orc.types[0].subtypes, vec![1, 2]);
}

#[test]
fn test_nested_types_are_laid_out_pre_order_like_java() {
    let schema = schema_of(vec![
        NestedField::optional(
            13,
            "c_arr",
            Type::List(ListType {
                element_field: NestedField::list_element(
                    16,
                    Type::Primitive(PrimitiveType::Int),
                    false,
                )
                .into(),
            }),
        ),
        NestedField::optional(
            14,
            "c_map",
            Type::Map(MapType {
                key_field: NestedField::map_key_element(17, Type::Primitive(PrimitiveType::String))
                    .into(),
                value_field: NestedField::map_value_element(
                    18,
                    Type::Primitive(PrimitiveType::Int),
                    false,
                )
                .into(),
            }),
        ),
        NestedField::optional(
            15,
            "c_struct",
            Type::Struct(crate::spec::StructType::new(vec![
                NestedField::optional(19, "x", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(20, "y", Type::Primitive(PrimitiveType::String)).into(),
            ])),
        ),
    ]);
    let orc = build_orc_schema(&schema).expect("map the schema");

    let kinds: Vec<Option<OrcKind>> = orc.types.iter().map(|t| t.kind).collect();
    assert_eq!(kinds, vec![
        Some(OrcKind::Struct),
        Some(OrcKind::List),
        Some(OrcKind::Int),
        Some(OrcKind::Map),
        Some(OrcKind::String),
        Some(OrcKind::Int),
        Some(OrcKind::Struct),
        Some(OrcKind::Int),
        Some(OrcKind::String),
    ]);
    assert_eq!(orc.types[0].subtypes, vec![1, 3, 6]);
    assert_eq!(orc.types[1].subtypes, vec![2]);
    assert_eq!(orc.types[3].subtypes, vec![4, 5]);
    assert_eq!(orc.types[6].subtypes, vec![7, 8]);
    assert_eq!(orc.types[6].field_names, vec![
        "x".to_string(),
        "y".to_string()
    ]);

    let ids: Vec<Option<&str>> = orc
        .types
        .iter()
        .map(|t| attribute(t, ICEBERG_ID_ATTRIBUTE))
        .collect();
    assert_eq!(ids, vec![
        None,
        Some("13"),
        Some("16"),
        Some("14"),
        Some("17"),
        Some("18"),
        Some("15"),
        Some("19"),
        Some("20"),
    ]);
    assert_eq!(
        attribute(&orc.types[4], ICEBERG_REQUIRED_ATTRIBUTE),
        Some("true"),
        "a map key is always required"
    );
}

#[test]
fn test_variant_is_refused_by_name_and_never_panics() {
    let schema = schema_of(vec![NestedField::optional(3, "v", Type::Variant)]);
    let error = build_orc_schema(&schema).expect_err("variant has no ORC mapping");
    assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
    assert!(
        error.message().contains("variant"),
        "the refusal must name the type: {error}"
    );
}

#[test]
fn test_unknown_is_refused_by_name_and_never_panics() {
    let schema = schema_of(vec![NestedField::optional(
        4,
        "u",
        Type::Primitive(PrimitiveType::Unknown),
    )]);
    let error = build_orc_schema(&schema).expect_err("unknown has no ORC mapping");
    assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
    assert!(
        error.message().contains("unknown"),
        "the refusal must name the type: {error}"
    );
}

#[test]
fn test_a_refused_type_nested_in_a_list_is_still_refused() {
    let schema = schema_of(vec![NestedField::optional(
        5,
        "l",
        Type::List(ListType {
            element_field: NestedField::list_element(6, Type::Variant, false).into(),
        }),
    )]);
    let error = build_orc_schema(&schema).expect_err("a variant element has no ORC mapping");
    assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
}
