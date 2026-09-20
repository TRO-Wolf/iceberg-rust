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

use super::*;
use crate::spec::{Literal, PrimitiveType, Type};

#[test]
fn test_partition_spec() {
    let spec = r#"
        {
        "spec-id": 1,
        "fields": [ {
            "source-id": 4,
            "field-id": 1000,
            "name": "ts_day",
            "transform": "day"
            }, {
            "source-id": 1,
            "field-id": 1001,
            "name": "id_bucket",
            "transform": "bucket[16]"
            }, {
            "source-id": 2,
            "field-id": 1002,
            "name": "id_truncate",
            "transform": "truncate[4]"
            } ]
        }
        "#;

    let partition_spec: PartitionSpec = serde_json::from_str(spec).unwrap();
    assert_eq!(4, partition_spec.fields[0].source_id);
    assert_eq!(1000, partition_spec.fields[0].field_id);
    assert_eq!("ts_day", partition_spec.fields[0].name);
    assert_eq!(Transform::Day, partition_spec.fields[0].transform);

    assert_eq!(1, partition_spec.fields[1].source_id);
    assert_eq!(1001, partition_spec.fields[1].field_id);
    assert_eq!("id_bucket", partition_spec.fields[1].name);
    assert_eq!(Transform::Bucket(16), partition_spec.fields[1].transform);

    assert_eq!(2, partition_spec.fields[2].source_id);
    assert_eq!(1002, partition_spec.fields[2].field_id);
    assert_eq!("id_truncate", partition_spec.fields[2].name);
    assert_eq!(Transform::Truncate(4), partition_spec.fields[2].transform);
}

#[test]
fn test_table_metadata_with_invalid_transform_parameter_fails_deserialization() {
    fn metadata_json(transform: &str) -> String {
        format!(
            r#"
                {{
                    "format-version": 2,
                    "table-uuid": "9c12d441-03fe-4693-9a96-a0705ddf69c1",
                    "location": "s3://bucket/test/location",
                    "last-sequence-number": 1,
                    "last-updated-ms": 1602638573590,
                    "last-column-id": 1,
                    "current-schema-id": 0,
                    "schemas": [
                        {{
                            "type": "struct",
                            "schema-id": 0,
                            "fields": [
                                {{
                                    "id": 1,
                                    "name": "x",
                                    "required": true,
                                    "type": "long"
                                }}
                            ]
                        }}
                    ],
                    "default-spec-id": 0,
                    "partition-specs": [
                        {{
                            "spec-id": 0,
                            "fields": [
                                {{
                                    "source-id": 1,
                                    "field-id": 1000,
                                    "name": "x_partition",
                                    "transform": "{transform}"
                                }}
                            ]
                        }}
                    ],
                    "last-partition-id": 1000,
                    "default-sort-order-id": 0,
                    "sort-orders": [
                        {{
                            "order-id": 0,
                            "fields": []
                        }}
                    ],
                    "properties": {{}},
                    "snapshots": [],
                    "statistics": [],
                    "snapshot-log": [],
                    "metadata-log": []
                }}
                "#
        )
    }

    let control = serde_json::from_str::<crate::spec::TableMetadata>(&metadata_json("bucket[16]"))
        .expect("control metadata with bucket[16] must deserialize");
    assert_eq!(
        control.default_partition_spec().fields()[0].transform,
        Transform::Bucket(16)
    );

    for sabotaged in ["bucket[0]", "truncate[0]", "bucket[2147483648]"] {
        let serde_error =
            serde_json::from_str::<crate::spec::TableMetadata>(&metadata_json(sabotaged))
                .unwrap_err();
        let error = Error::from(serde_error);
        assert_eq!(error.kind(), ErrorKind::DataInvalid, "{sabotaged}");
    }

    let serde_error = serde_json::from_str::<PartitionSpec>(
        r#"{
                "spec-id": 0,
                "fields": [
                    {
                        "source-id": 1,
                        "field-id": 1000,
                        "name": "x_partition",
                        "transform": "bucket[0]"
                    }
                ]
            }"#,
    )
    .unwrap_err();
    assert!(
        serde_error
            .to_string()
            .contains("Invalid number of buckets: 0 (must be > 0)"),
        "expected the Java precondition text, got: {serde_error}"
    );
}

#[test]
fn test_partition_spec_builder_rejects_zero_parameter_transforms() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()
        .expect("valid schema");

    let error = PartitionSpec::builder(schema.clone())
        .add_partition_field("id", "id_bucket", Transform::Bucket(0))
        .expect_err("bucket[0] must be rejected by the bound builder");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error
            .message()
            .contains("Invalid number of buckets: 0 (must be > 0)"),
        "message must match the Java precondition text, got: {}",
        error.message()
    );

    let error = PartitionSpec::builder(schema.clone())
        .add_partition_field("id", "id_truncate", Transform::Truncate(0))
        .expect_err("truncate[0] must be rejected by the bound builder");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);

    PartitionSpec::builder(schema)
        .add_partition_field("id", "id_bucket", Transform::Bucket(16))
        .expect("bucket[16] is legal")
        .build()
        .expect("legal spec must build");
}

#[test]
fn test_unbound_partition_spec_builder_rejects_zero_parameter_transforms() {
    let error = UnboundPartitionSpec::builder()
        .add_partition_field(1, "id_bucket", Transform::Bucket(0))
        .expect_err("bucket[0] must be rejected by the unbound builder");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);

    let error = UnboundPartitionSpec::builder()
        .add_partition_field(1, "id_truncate", Transform::Truncate(0))
        .expect_err("truncate[0] must be rejected by the unbound builder");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);

    UnboundPartitionSpec::builder()
        .add_partition_field(1, "id_bucket", Transform::Bucket(16))
        .expect("bucket[16] is legal");
}

#[test]
fn test_is_unpartitioned() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
            NestedField::required(
                2,
                "name",
                Type::Primitive(crate::spec::PrimitiveType::String),
            )
            .into(),
        ])
        .build()
        .unwrap();
    let partition_spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(1)
        .build()
        .unwrap();
    assert!(
        partition_spec.is_unpartitioned(),
        "Empty partition spec should be unpartitioned"
    );

    let partition_spec = PartitionSpec::builder(schema.clone())
        .add_unbound_fields(vec![
            UnboundPartitionField::builder()
                .source_id(1)
                .name("id".to_string())
                .transform(Transform::Identity)
                .build(),
            UnboundPartitionField::builder()
                .source_id(2)
                .name("name_string".to_string())
                .transform(Transform::Void)
                .build(),
        ])
        .unwrap()
        .with_spec_id(1)
        .build()
        .unwrap();
    assert!(
        !partition_spec.is_unpartitioned(),
        "Partition spec with one non void transform should not be unpartitioned"
    );

    let partition_spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(1)
        .add_unbound_fields(vec![
            UnboundPartitionField::builder()
                .source_id(1)
                .name("id_void".to_string())
                .transform(Transform::Void)
                .build(),
            UnboundPartitionField::builder()
                .source_id(2)
                .name("name_void".to_string())
                .transform(Transform::Void)
                .build(),
        ])
        .unwrap()
        .build()
        .unwrap();
    assert!(
        partition_spec.is_unpartitioned(),
        "Partition spec with all void field should be unpartitioned"
    );
}

#[test]
fn test_unbound_partition_spec() {
    let spec = r#"
		{
		"spec-id": 1,
		"fields": [ {
			"source-id": 4,
			"field-id": 1000,
			"name": "ts_day",
			"transform": "day"
			}, {
			"source-id": 1,
			"field-id": 1001,
			"name": "id_bucket",
			"transform": "bucket[16]"
			}, {
			"source-id": 2,
			"field-id": 1002,
			"name": "id_truncate",
			"transform": "truncate[4]"
			} ]
		}
		"#;

    let partition_spec: UnboundPartitionSpec = serde_json::from_str(spec).unwrap();
    assert_eq!(Some(1), partition_spec.spec_id);

    assert_eq!(4, partition_spec.fields[0].source_id);
    assert_eq!(Some(1000), partition_spec.fields[0].field_id);
    assert_eq!("ts_day", partition_spec.fields[0].name);
    assert_eq!(Transform::Day, partition_spec.fields[0].transform);

    assert_eq!(1, partition_spec.fields[1].source_id);
    assert_eq!(Some(1001), partition_spec.fields[1].field_id);
    assert_eq!("id_bucket", partition_spec.fields[1].name);
    assert_eq!(Transform::Bucket(16), partition_spec.fields[1].transform);

    assert_eq!(2, partition_spec.fields[2].source_id);
    assert_eq!(Some(1002), partition_spec.fields[2].field_id);
    assert_eq!("id_truncate", partition_spec.fields[2].name);
    assert_eq!(Transform::Truncate(4), partition_spec.fields[2].transform);

    let spec = r#"
		{
		"fields": [ {
			"source-id": 4,
			"name": "ts_day",
			"transform": "day"
			} ]
		}
		"#;
    let partition_spec: UnboundPartitionSpec = serde_json::from_str(spec).unwrap();
    assert_eq!(None, partition_spec.spec_id);

    assert_eq!(4, partition_spec.fields[0].source_id);
    assert_eq!(None, partition_spec.fields[0].field_id);
    assert_eq!("ts_day", partition_spec.fields[0].name);
    assert_eq!(Transform::Day, partition_spec.fields[0].transform);
}

#[test]
fn test_new_unpartition() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
            NestedField::required(
                2,
                "name",
                Type::Primitive(crate::spec::PrimitiveType::String),
            )
            .into(),
        ])
        .build()
        .unwrap();
    let partition_spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .build()
        .unwrap();
    let partition_type = partition_spec.partition_type(&schema).unwrap();
    assert_eq!(0, partition_type.fields().len());

    let unpartition_spec = PartitionSpec::unpartition_spec();
    assert_eq!(partition_spec, unpartition_spec);
}

#[test]
fn test_partition_type() {
    let spec = r#"
            {
            "spec-id": 1,
            "fields": [ {
                "source-id": 4,
                "field-id": 1000,
                "name": "ts_day",
                "transform": "day"
                }, {
                "source-id": 1,
                "field-id": 1001,
                "name": "id_bucket",
                "transform": "bucket[16]"
                }, {
                "source-id": 2,
                "field-id": 1002,
                "name": "id_truncate",
                "transform": "truncate[4]"
                } ]
            }
            "#;

    let partition_spec: PartitionSpec = serde_json::from_str(spec).unwrap();
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
            NestedField::required(
                2,
                "name",
                Type::Primitive(crate::spec::PrimitiveType::String),
            )
            .into(),
            NestedField::required(
                3,
                "ts",
                Type::Primitive(crate::spec::PrimitiveType::Timestamp),
            )
            .into(),
            NestedField::required(
                4,
                "ts_day",
                Type::Primitive(crate::spec::PrimitiveType::Timestamp),
            )
            .into(),
            NestedField::required(
                5,
                "id_bucket",
                Type::Primitive(crate::spec::PrimitiveType::Int),
            )
            .into(),
            NestedField::required(
                6,
                "id_truncate",
                Type::Primitive(crate::spec::PrimitiveType::Int),
            )
            .into(),
        ])
        .build()
        .unwrap();

    let partition_type = partition_spec.partition_type(&schema).unwrap();
    assert_eq!(3, partition_type.fields().len());
    assert_eq!(
        *partition_type.fields()[0],
        NestedField::optional(
            partition_spec.fields[0].field_id,
            &partition_spec.fields[0].name,
            Type::Primitive(crate::spec::PrimitiveType::Date)
        )
    );
    assert_eq!(
        *partition_type.fields()[1],
        NestedField::optional(
            partition_spec.fields[1].field_id,
            &partition_spec.fields[1].name,
            Type::Primitive(crate::spec::PrimitiveType::Int)
        )
    );
    assert_eq!(
        *partition_type.fields()[2],
        NestedField::optional(
            partition_spec.fields[2].field_id,
            &partition_spec.fields[2].name,
            Type::Primitive(crate::spec::PrimitiveType::String)
        )
    );
}

#[test]
fn test_partition_empty() {
    let spec = r#"
            {
            "spec-id": 1,
            "fields": []
            }
            "#;

    let partition_spec: PartitionSpec = serde_json::from_str(spec).unwrap();
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
            NestedField::required(
                2,
                "name",
                Type::Primitive(crate::spec::PrimitiveType::String),
            )
            .into(),
            NestedField::required(
                3,
                "ts",
                Type::Primitive(crate::spec::PrimitiveType::Timestamp),
            )
            .into(),
            NestedField::required(
                4,
                "ts_day",
                Type::Primitive(crate::spec::PrimitiveType::Timestamp),
            )
            .into(),
            NestedField::required(
                5,
                "id_bucket",
                Type::Primitive(crate::spec::PrimitiveType::Int),
            )
            .into(),
            NestedField::required(
                6,
                "id_truncate",
                Type::Primitive(crate::spec::PrimitiveType::Int),
            )
            .into(),
        ])
        .build()
        .unwrap();

    let partition_type = partition_spec.partition_type(&schema).unwrap();
    assert_eq!(0, partition_type.fields().len());
}

#[test]
fn test_partition_error() {
    let spec = r#"
        {
        "spec-id": 1,
        "fields": [ {
            "source-id": 4,
            "field-id": 1000,
            "name": "ts_day",
            "transform": "day"
            }, {
            "source-id": 1,
            "field-id": 1001,
            "name": "id_bucket",
            "transform": "bucket[16]"
            }, {
            "source-id": 2,
            "field-id": 1002,
            "name": "id_truncate",
            "transform": "truncate[4]"
            } ]
        }
        "#;

    let partition_spec: PartitionSpec = serde_json::from_str(spec).unwrap();
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
            NestedField::required(
                2,
                "name",
                Type::Primitive(crate::spec::PrimitiveType::String),
            )
            .into(),
        ])
        .build()
        .unwrap();

    assert!(partition_spec.partition_type(&schema).is_err());
}

#[test]
fn test_builder_disallow_duplicate_names() {
    UnboundPartitionSpec::builder()
        .add_partition_field(1, "ts_day".to_string(), Transform::Day)
        .unwrap()
        .add_partition_field(2, "ts_day".to_string(), Transform::Day)
        .unwrap_err();
}

fn schema_with_variant_column() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
            NestedField::optional(2, "v", Type::Variant).into(),
        ])
        .build()
        .unwrap()
}

#[test]
fn test_variant_rejected_as_partition_source_for_identity_and_bucket() {
    for transform in [
        Transform::Identity,
        Transform::Bucket(16),
        Transform::Truncate(4),
        Transform::Year,
        Transform::Month,
        Transform::Day,
        Transform::Hour,
    ] {
        let error = PartitionSpec::builder(schema_with_variant_column())
            .add_unbound_field(UnboundPartitionField {
                source_id: 2,
                field_id: None,
                name: "v_part".to_string(),
                transform,
            })
            .expect_err("a variant partition source must be rejected");
        assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
        assert!(
            error
                .message()
                .contains("Cannot partition by non-primitive source field: 'variant'"),
            "{transform} must reject variant at the non-primitive door (Java fires it before \
                 canTransform), got: {}",
            error.message()
        );
    }
}

#[test]
fn test_variant_accepted_as_void_partition_source() {
    let spec = PartitionSpec::builder(schema_with_variant_column())
        .add_unbound_field(UnboundPartitionField {
            source_id: 2,
            field_id: None,
            name: "v_void".to_string(),
            transform: Transform::Void,
        })
        .expect("void on a variant source is legal (Java skips alwaysNull)")
        .build()
        .expect("build the spec");
    assert_eq!(spec.fields().len(), 1);
    assert_eq!(spec.fields()[0].transform, Transform::Void);
}

#[test]
fn test_builder_disallow_duplicate_field_ids() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
            NestedField::required(
                2,
                "name",
                Type::Primitive(crate::spec::PrimitiveType::String),
            )
            .into(),
        ])
        .build()
        .unwrap();
    PartitionSpec::builder(schema.clone())
        .add_unbound_field(UnboundPartitionField {
            source_id: 1,
            field_id: Some(1000),
            name: "id".to_string(),
            transform: Transform::Identity,
        })
        .unwrap()
        .add_unbound_field(UnboundPartitionField {
            source_id: 2,
            field_id: Some(1000),
            name: "id_bucket".to_string(),
            transform: Transform::Bucket(16),
        })
        .unwrap_err();
}

#[test]
fn test_builder_auto_assign_field_ids() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
            NestedField::required(
                2,
                "name",
                Type::Primitive(crate::spec::PrimitiveType::String),
            )
            .into(),
            NestedField::required(
                3,
                "ts",
                Type::Primitive(crate::spec::PrimitiveType::Timestamp),
            )
            .into(),
        ])
        .build()
        .unwrap();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(1)
        .add_unbound_field(UnboundPartitionField {
            source_id: 1,
            name: "id".to_string(),
            transform: Transform::Identity,
            field_id: Some(1012),
        })
        .unwrap()
        .add_unbound_field(UnboundPartitionField {
            source_id: 2,
            name: "name_void".to_string(),
            transform: Transform::Void,
            field_id: None,
        })
        .unwrap()
        .add_unbound_field(UnboundPartitionField {
            source_id: 3,
            name: "year".to_string(),
            transform: Transform::Year,
            field_id: Some(1),
        })
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(1012, spec.fields[0].field_id);
    assert_eq!(1013, spec.fields[1].field_id);
    assert_eq!(1, spec.fields[2].field_id);
}

#[test]
fn test_builder_valid_schema() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
            NestedField::required(
                2,
                "name",
                Type::Primitive(crate::spec::PrimitiveType::String),
            )
            .into(),
        ])
        .build()
        .unwrap();

    PartitionSpec::builder(schema.clone())
        .with_spec_id(1)
        .build()
        .unwrap();

    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(1)
        .add_partition_field("id", "id_bucket[16]", Transform::Bucket(16))
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(spec, PartitionSpec {
        spec_id: 1,
        fields: vec![PartitionField {
            source_id: 1,
            field_id: 1000,
            name: "id_bucket[16]".to_string(),
            transform: Transform::Bucket(16),
        }],
    });
    assert_eq!(
        spec.partition_type(&schema).unwrap(),
        StructType::new(vec![
            NestedField::optional(1000, "id_bucket[16]", Type::Primitive(PrimitiveType::Int))
                .into()
        ])
    )
}

#[test]
fn test_collision_with_schema_name() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
        ])
        .build()
        .unwrap();

    PartitionSpec::builder(schema.clone())
        .with_spec_id(1)
        .build()
        .unwrap();

    let err = PartitionSpec::builder(schema)
        .with_spec_id(1)
        .add_unbound_field(UnboundPartitionField {
            source_id: 1,
            field_id: None,
            name: "id".to_string(),
            transform: Transform::Bucket(16),
        })
        .unwrap_err();
    assert!(err.message().contains("conflicts with schema"))
}

#[test]
fn test_builder_collision_is_ok_for_identity_transforms() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
            NestedField::required(
                2,
                "number",
                Type::Primitive(crate::spec::PrimitiveType::Int),
            )
            .into(),
        ])
        .build()
        .unwrap();

    PartitionSpec::builder(schema.clone())
        .with_spec_id(1)
        .build()
        .unwrap();

    PartitionSpec::builder(schema.clone())
        .with_spec_id(1)
        .add_unbound_field(UnboundPartitionField {
            source_id: 1,
            field_id: None,
            name: "id".to_string(),
            transform: Transform::Identity,
        })
        .unwrap()
        .build()
        .unwrap();

    PartitionSpec::builder(schema)
        .with_spec_id(1)
        .add_unbound_field(UnboundPartitionField {
            source_id: 2,
            field_id: None,
            name: "id".to_string(),
            transform: Transform::Identity,
        })
        .unwrap_err();
}

#[test]
fn test_builder_collision_is_ok_for_void_named_after_its_own_source() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
            NestedField::required(
                2,
                "number",
                Type::Primitive(crate::spec::PrimitiveType::Int),
            )
            .into(),
        ])
        .build()
        .unwrap();

    PartitionSpec::builder(schema.clone())
        .with_spec_id(1)
        .add_unbound_field(UnboundPartitionField {
            source_id: 1,
            field_id: Some(1000),
            name: "id".to_string(),
            transform: Transform::Void,
        })
        .unwrap()
        .build()
        .unwrap();

    PartitionSpec::builder(schema)
        .with_spec_id(1)
        .add_unbound_field(UnboundPartitionField {
            source_id: 2,
            field_id: Some(1000),
            name: "id".to_string(),
            transform: Transform::Void,
        })
        .unwrap_err();
}

#[test]
fn test_builder_all_source_ids_must_exist() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
            NestedField::required(
                2,
                "name",
                Type::Primitive(crate::spec::PrimitiveType::String),
            )
            .into(),
            NestedField::required(
                3,
                "ts",
                Type::Primitive(crate::spec::PrimitiveType::Timestamp),
            )
            .into(),
        ])
        .build()
        .unwrap();

    PartitionSpec::builder(schema.clone())
        .with_spec_id(1)
        .add_unbound_fields(vec![
            UnboundPartitionField {
                source_id: 1,
                field_id: None,
                name: "id_bucket".to_string(),
                transform: Transform::Bucket(16),
            },
            UnboundPartitionField {
                source_id: 2,
                field_id: None,
                name: "name".to_string(),
                transform: Transform::Identity,
            },
        ])
        .unwrap()
        .build()
        .unwrap();

    PartitionSpec::builder(schema)
        .with_spec_id(1)
        .add_unbound_fields(vec![
            UnboundPartitionField {
                source_id: 1,
                field_id: None,
                name: "id_bucket".to_string(),
                transform: Transform::Bucket(16),
            },
            UnboundPartitionField {
                source_id: 4,
                field_id: None,
                name: "name".to_string(),
                transform: Transform::Identity,
            },
        ])
        .unwrap_err();
}

#[test]
fn test_builder_disallows_redundant() {
    let err = UnboundPartitionSpec::builder()
        .with_spec_id(1)
        .add_partition_field(1, "id_bucket[16]".to_string(), Transform::Bucket(16))
        .unwrap()
        .add_partition_field(
            1,
            "id_bucket_with_other_name".to_string(),
            Transform::Bucket(16),
        )
        .unwrap_err();
    assert!(err.message().contains("redundant partition"));
}

#[test]
fn test_builder_incompatible_transforms_disallowed() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
        ])
        .build()
        .unwrap();

    PartitionSpec::builder(schema)
        .with_spec_id(1)
        .add_unbound_field(UnboundPartitionField {
            source_id: 1,
            field_id: None,
            name: "id_year".to_string(),
            transform: Transform::Year,
        })
        .unwrap_err();
}

#[test]
fn test_build_unbound_specs_without_partition_id() {
    let spec = UnboundPartitionSpec::builder()
        .with_spec_id(1)
        .add_partition_fields(vec![UnboundPartitionField {
            source_id: 1,
            field_id: None,
            name: "id_bucket[16]".to_string(),
            transform: Transform::Bucket(16),
        }])
        .unwrap()
        .build();

    assert_eq!(spec, UnboundPartitionSpec {
        spec_id: Some(1),
        fields: vec![UnboundPartitionField {
            source_id: 1,
            field_id: None,
            name: "id_bucket[16]".to_string(),
            transform: Transform::Bucket(16),
        }]
    });
}

#[test]
fn test_is_compatible_with() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
            NestedField::required(
                2,
                "name",
                Type::Primitive(crate::spec::PrimitiveType::String),
            )
            .into(),
        ])
        .build()
        .unwrap();

    let partition_spec_1 = PartitionSpec::builder(schema.clone())
        .with_spec_id(1)
        .add_unbound_field(UnboundPartitionField {
            source_id: 1,
            field_id: None,
            name: "id_bucket".to_string(),
            transform: Transform::Bucket(16),
        })
        .unwrap()
        .build()
        .unwrap();

    let partition_spec_2 = PartitionSpec::builder(schema)
        .with_spec_id(1)
        .add_unbound_field(UnboundPartitionField {
            source_id: 1,
            field_id: None,
            name: "id_bucket".to_string(),
            transform: Transform::Bucket(16),
        })
        .unwrap()
        .build()
        .unwrap();

    assert!(partition_spec_1.is_compatible_with(&partition_spec_2));
}

#[test]
fn test_not_compatible_with_transform_different() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
        ])
        .build()
        .unwrap();

    let partition_spec_1 = PartitionSpec::builder(schema.clone())
        .with_spec_id(1)
        .add_unbound_field(UnboundPartitionField {
            source_id: 1,
            field_id: None,
            name: "id_bucket".to_string(),
            transform: Transform::Bucket(16),
        })
        .unwrap()
        .build()
        .unwrap();

    let partition_spec_2 = PartitionSpec::builder(schema)
        .with_spec_id(1)
        .add_unbound_field(UnboundPartitionField {
            source_id: 1,
            field_id: None,
            name: "id_bucket".to_string(),
            transform: Transform::Bucket(32),
        })
        .unwrap()
        .build()
        .unwrap();

    assert!(!partition_spec_1.is_compatible_with(&partition_spec_2));
}

#[test]
fn test_not_compatible_with_source_id_different() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
            NestedField::required(
                2,
                "name",
                Type::Primitive(crate::spec::PrimitiveType::String),
            )
            .into(),
        ])
        .build()
        .unwrap();

    let partition_spec_1 = PartitionSpec::builder(schema.clone())
        .with_spec_id(1)
        .add_unbound_field(UnboundPartitionField {
            source_id: 1,
            field_id: None,
            name: "id_bucket".to_string(),
            transform: Transform::Bucket(16),
        })
        .unwrap()
        .build()
        .unwrap();

    let partition_spec_2 = PartitionSpec::builder(schema)
        .with_spec_id(1)
        .add_unbound_field(UnboundPartitionField {
            source_id: 2,
            field_id: None,
            name: "id_bucket".to_string(),
            transform: Transform::Bucket(16),
        })
        .unwrap()
        .build()
        .unwrap();

    assert!(!partition_spec_1.is_compatible_with(&partition_spec_2));
}

#[test]
fn test_not_compatible_with_order_different() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
            NestedField::required(
                2,
                "name",
                Type::Primitive(crate::spec::PrimitiveType::String),
            )
            .into(),
        ])
        .build()
        .unwrap();

    let partition_spec_1 = PartitionSpec::builder(schema.clone())
        .with_spec_id(1)
        .add_unbound_field(UnboundPartitionField {
            source_id: 1,
            field_id: None,
            name: "id_bucket".to_string(),
            transform: Transform::Bucket(16),
        })
        .unwrap()
        .add_unbound_field(UnboundPartitionField {
            source_id: 2,
            field_id: None,
            name: "name".to_string(),
            transform: Transform::Identity,
        })
        .unwrap()
        .build()
        .unwrap();

    let partition_spec_2 = PartitionSpec::builder(schema)
        .with_spec_id(1)
        .add_unbound_field(UnboundPartitionField {
            source_id: 2,
            field_id: None,
            name: "name".to_string(),
            transform: Transform::Identity,
        })
        .unwrap()
        .add_unbound_field(UnboundPartitionField {
            source_id: 1,
            field_id: None,
            name: "id_bucket".to_string(),
            transform: Transform::Bucket(16),
        })
        .unwrap()
        .build()
        .unwrap();

    assert!(!partition_spec_1.is_compatible_with(&partition_spec_2));
}

#[test]
fn test_highest_field_id_unpartitioned() {
    let spec = PartitionSpec::builder(Schema::builder().with_fields(vec![]).build().unwrap())
        .with_spec_id(1)
        .build()
        .unwrap();

    assert!(spec.highest_field_id().is_none());
}

#[test]
fn test_highest_field_id() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
            NestedField::required(
                2,
                "name",
                Type::Primitive(crate::spec::PrimitiveType::String),
            )
            .into(),
        ])
        .build()
        .unwrap();

    let spec = PartitionSpec::builder(schema)
        .with_spec_id(1)
        .add_unbound_field(UnboundPartitionField {
            source_id: 1,
            field_id: Some(1001),
            name: "id".to_string(),
            transform: Transform::Identity,
        })
        .unwrap()
        .add_unbound_field(UnboundPartitionField {
            source_id: 2,
            field_id: Some(1000),
            name: "name".to_string(),
            transform: Transform::Identity,
        })
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(Some(1001), spec.highest_field_id());
}

#[test]
fn test_has_sequential_ids() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
            NestedField::required(
                2,
                "name",
                Type::Primitive(crate::spec::PrimitiveType::String),
            )
            .into(),
        ])
        .build()
        .unwrap();

    let spec = PartitionSpec::builder(schema)
        .with_spec_id(1)
        .add_unbound_field(UnboundPartitionField {
            source_id: 1,
            field_id: Some(1000),
            name: "id".to_string(),
            transform: Transform::Identity,
        })
        .unwrap()
        .add_unbound_field(UnboundPartitionField {
            source_id: 2,
            field_id: Some(1001),
            name: "name".to_string(),
            transform: Transform::Identity,
        })
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(1000, spec.fields[0].field_id);
    assert_eq!(1001, spec.fields[1].field_id);
    assert!(spec.has_sequential_ids());
}

#[test]
fn test_sequential_ids_must_start_at_1000() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
            NestedField::required(
                2,
                "name",
                Type::Primitive(crate::spec::PrimitiveType::String),
            )
            .into(),
        ])
        .build()
        .unwrap();

    let spec = PartitionSpec::builder(schema)
        .with_spec_id(1)
        .add_unbound_field(UnboundPartitionField {
            source_id: 1,
            field_id: Some(999),
            name: "id".to_string(),
            transform: Transform::Identity,
        })
        .unwrap()
        .add_unbound_field(UnboundPartitionField {
            source_id: 2,
            field_id: Some(1000),
            name: "name".to_string(),
            transform: Transform::Identity,
        })
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(999, spec.fields[0].field_id);
    assert_eq!(1000, spec.fields[1].field_id);
    assert!(!spec.has_sequential_ids());
}

#[test]
fn test_sequential_ids_must_have_no_gaps() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int)).into(),
            NestedField::required(
                2,
                "name",
                Type::Primitive(crate::spec::PrimitiveType::String),
            )
            .into(),
        ])
        .build()
        .unwrap();

    let spec = PartitionSpec::builder(schema)
        .with_spec_id(1)
        .add_unbound_field(UnboundPartitionField {
            source_id: 1,
            field_id: Some(1000),
            name: "id".to_string(),
            transform: Transform::Identity,
        })
        .unwrap()
        .add_unbound_field(UnboundPartitionField {
            source_id: 2,
            field_id: Some(1002),
            name: "name".to_string(),
            transform: Transform::Identity,
        })
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(1000, spec.fields[0].field_id);
    assert_eq!(1002, spec.fields[1].field_id);
    assert!(!spec.has_sequential_ids());
}

#[test]
fn test_partition_to_path() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "timestamp", Type::Primitive(PrimitiveType::Timestamp)).into(),
            NestedField::required(4, "empty", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .unwrap();

    let spec = PartitionSpec::builder(schema.clone())
        .add_partition_field("id", "id", Transform::Identity)
        .unwrap()
        .add_partition_field("name", "name", Transform::Identity)
        .unwrap()
        .add_partition_field("timestamp", "ts_hour", Transform::Hour)
        .unwrap()
        .add_partition_field("empty", "empty_void", Transform::Void)
        .unwrap()
        .build()
        .unwrap();

    let data = Struct::from_iter([
        Some(Literal::int(42)),
        Some(Literal::string("alice")),
        Some(Literal::int(1000)),
        Some(Literal::string("empty")),
    ]);

    assert_eq!(
        spec.partition_to_path(&data, schema.into()),
        "id=42/name=alice/ts_hour=1000/empty_void=null"
    );
}
