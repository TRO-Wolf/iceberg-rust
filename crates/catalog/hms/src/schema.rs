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

use hive_metastore::FieldSchema;
use iceberg::spec::{PrimitiveType, Schema, SchemaVisitor, visit_schema};
use iceberg::{Error, ErrorKind, Result};

use crate::catalog::HiveVersion;

type HiveSchema = Vec<FieldSchema>;

/// Converts an Iceberg [`Schema`] into a Hive column list, mirroring Java
/// `HiveSchemaUtil.convertToTypeString` (behavior bytecode-verified against
/// iceberg-hive-metastore-1.10.0.jar with javap; offsets cited inline below).
#[derive(Debug, Default)]
pub(crate) struct HiveSchemaBuilder {
    schema: HiveSchema,
    depth: usize,
    hive_version: HiveVersion,
}

impl HiveSchemaBuilder {
    /// Creates a new `HiveSchemaBuilder` from iceberg `Schema`.
    ///
    /// `hive_version` gates the Hive-version-dependent type strings (currently
    /// only the timestamptz column type); see [`HiveVersion`].
    pub fn from_iceberg(schema: &Schema, hive_version: HiveVersion) -> Result<HiveSchemaBuilder> {
        let mut builder = Self {
            hive_version,
            ..Self::default()
        };
        visit_schema(schema, &mut builder)?;
        Ok(builder)
    }

    /// Returns the newly converted `HiveSchema`
    pub fn build(self) -> HiveSchema {
        self.schema
    }

    /// Check if is in `StructType` while traversing schema
    fn is_inside_struct(&self) -> bool {
        self.depth > 0
    }
}

impl SchemaVisitor for HiveSchemaBuilder {
    type T = String;

    fn schema(
        &mut self,
        _schema: &iceberg::spec::Schema,
        value: String,
    ) -> iceberg::Result<String> {
        Ok(value)
    }

    fn before_struct_field(
        &mut self,
        _field: &iceberg::spec::NestedFieldRef,
    ) -> iceberg::Result<()> {
        self.depth += 1;
        Ok(())
    }

    fn r#struct(
        &mut self,
        r#_struct: &iceberg::spec::StructType,
        results: Vec<String>,
    ) -> iceberg::Result<String> {
        // Java joins struct fields with a bare "," (no space) — bytecode-verified
        // (javap, iceberg-hive-metastore-1.10.0.jar, HiveSchemaUtil.convertToTypeString
        // offsets 204-206: Collectors.joining(",") into "struct<%s>" at 219).
        Ok(format!("struct<{}>", results.join(",")))
    }

    fn after_struct_field(
        &mut self,
        _field: &iceberg::spec::NestedFieldRef,
    ) -> iceberg::Result<()> {
        self.depth -= 1;
        Ok(())
    }

    fn field(
        &mut self,
        field: &iceberg::spec::NestedFieldRef,
        value: String,
    ) -> iceberg::Result<String> {
        if self.is_inside_struct() {
            // Java renders struct fields as "%s:%s" (name:type, no case folding) —
            // bytecode-verified (javap, iceberg-hive-metastore-1.10.0.jar,
            // HiveSchemaUtil.lambda$convertToTypeString$2 offset 0).
            return Ok(format!("{}:{}", field.name, value));
        }

        self.schema.push(FieldSchema {
            name: Some(field.name.clone().into()),
            r#type: Some(value.clone().into()),
            comment: field.doc.clone().map(|doc| doc.into()),
        });

        Ok(value)
    }

    fn list(&mut self, _list: &iceberg::spec::ListType, value: String) -> iceberg::Result<String> {
        Ok(format!("array<{value}>"))
    }

    fn map(
        &mut self,
        _map: &iceberg::spec::MapType,
        key_value: String,
        value: String,
    ) -> iceberg::Result<String> {
        Ok(format!("map<{key_value},{value}>"))
    }

    fn primitive(&mut self, p: &iceberg::spec::PrimitiveType) -> iceberg::Result<String> {
        let hive_type = match p {
            PrimitiveType::Boolean => "boolean".to_string(),
            PrimitiveType::Int => "int".to_string(),
            PrimitiveType::Long => "bigint".to_string(),
            PrimitiveType::Float => "float".to_string(),
            PrimitiveType::Double => "double".to_string(),
            PrimitiveType::Date => "date".to_string(),
            PrimitiveType::Timestamp => "timestamp".to_string(),
            // Java's TIMESTAMP case is Hive-version-gated: `if HiveVersion.min(HIVE_3)
            // && ts.shouldAdjustToUTC() -> "timestamp with local time zone" else ->
            // "timestamp"` — bytecode-verified (javap, iceberg-hive-metastore-1.10.0.jar,
            // HiveSchemaUtil.convertToTypeString offsets 113-139). So timestamptz (µs)
            // is a string on every Hive version, never an error.
            PrimitiveType::Timestamptz => match self.hive_version {
                HiveVersion::Hive3Plus => "timestamp with local time zone".to_string(),
                HiveVersion::Hive2 => "timestamp".to_string(),
            },
            // Java's TypeID $SwitchMap (HiveSchemaUtil$1) has no TIMESTAMP_NANO or
            // UNKNOWN entry, so both nano variants (regardless of zone) and `unknown`
            // fall to the default arm: `throw new UnsupportedOperationException(type +
            // " is not supported")` — bytecode-verified (javap,
            // iceberg-hive-metastore-1.10.0.jar, convertToTypeString offsets 303-319;
            // concat recipe "\u{1} is not supported"). Java never emits a
            // "timestamp_ns" column type; mirror the throw rather than fabricate one.
            PrimitiveType::TimestampNs | PrimitiveType::TimestamptzNs | PrimitiveType::Unknown => {
                return Err(Error::new(
                    ErrorKind::FeatureUnsupported,
                    format!("{p} is not supported"),
                ));
            }
            PrimitiveType::Time | PrimitiveType::String | PrimitiveType::Uuid => {
                "string".to_string()
            }
            PrimitiveType::Binary | PrimitiveType::Fixed(_) => "binary".to_string(),
            PrimitiveType::Decimal { precision, scale } => {
                format!("decimal({precision},{scale})")
            }
        };

        Ok(hive_type)
    }
}

#[cfg(test)]
mod tests {
    use iceberg::Result;
    use iceberg::spec::Schema;

    use super::*;

    #[test]
    fn test_schema_with_nested_maps() -> Result<()> {
        let record = r#"
            {
                "schema-id": 1,
                "type": "struct",
                "fields": [
                    {
                        "id": 1,
                        "name": "quux",
                        "required": true,
                        "type": {
                            "type": "map",
                            "key-id": 2,
                            "key": "string",
                            "value-id": 3,
                            "value-required": true,
                            "value": {
                                "type": "map",
                                "key-id": 4,
                                "key": "string",
                                "value-id": 5,
                                "value-required": true,
                                "value": "int"
                            }
                        }
                    }
                ]
            }
        "#;

        let schema = serde_json::from_str::<Schema>(record)?;

        let result = HiveSchemaBuilder::from_iceberg(&schema, HiveVersion::default())?.build();

        let expected = vec![FieldSchema {
            name: Some("quux".into()),
            r#type: Some("map<string,map<string,int>>".into()),
            comment: None,
        }];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_schema_with_struct_inside_list() -> Result<()> {
        let record = r#"
        {
            "schema-id": 1,
            "type": "struct",
            "fields": [
                {
                    "id": 1,
                    "name": "location",
                    "required": true,
                    "type": {
                        "type": "list",
                        "element-id": 2,
                        "element-required": true,
                        "element": {
                            "type": "struct",
                            "fields": [
                                {
                                    "id": 3,
                                    "name": "latitude",
                                    "required": false,
                                    "type": "float"
                                },
                                {
                                    "id": 4,
                                    "name": "longitude",
                                    "required": false,
                                    "type": "float"
                                }
                            ]
                        }
                    }
                }
            ]
        }
        "#;

        let schema = serde_json::from_str::<Schema>(record)?;

        let result = HiveSchemaBuilder::from_iceberg(&schema, HiveVersion::default())?.build();

        let expected = vec![FieldSchema {
            name: Some("location".into()),
            r#type: Some("array<struct<latitude:float,longitude:float>>".into()),
            comment: None,
        }];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_schema_with_structs() -> Result<()> {
        let record = r#"{
            "type": "struct",
            "schema-id": 1,
            "fields": [
                {
                    "id": 1,
                    "name": "person",
                    "required": true,
                    "type": {
                        "type": "struct",
                        "fields": [
                            {
                                "id": 2,
                                "name": "name",
                                "required": true,
                                "type": "string"
                            },
                            {
                                "id": 3,
                                "name": "age",
                                "required": false,
                                "type": "int"
                            }
                        ]
                    }
                }
            ]
        }"#;

        let schema = serde_json::from_str::<Schema>(record)?;

        let result = HiveSchemaBuilder::from_iceberg(&schema, HiveVersion::default())?.build();

        let expected = vec![FieldSchema {
            name: Some("person".into()),
            r#type: Some("struct<name:string,age:int>".into()),
            comment: None,
        }];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_schema_with_simple_fields() -> Result<()> {
        let record = r#"{
            "type": "struct",
            "schema-id": 1,
            "fields": [
                {
                    "id": 1,
                    "name": "c1",
                    "required": true,
                    "type": "boolean"
                },
                {
                    "id": 2,
                    "name": "c2",
                    "required": true,
                    "type": "int"
                },
                {
                    "id": 3,
                    "name": "c3",
                    "required": true,
                    "type": "long"
                },
                {
                    "id": 4,
                    "name": "c4",
                    "required": true,
                    "type": "float"
                },
                {
                    "id": 5,
                    "name": "c5",
                    "required": true,
                    "type": "double"
                },
                {
                    "id": 6,
                    "name": "c6",
                    "required": true,
                    "type": "decimal(2,2)"
                },
                {
                    "id": 7,
                    "name": "c7",
                    "required": true,
                    "type": "date"
                },
                {
                    "id": 8,
                    "name": "c8",
                    "required": true,
                    "type": "time"
                },
                {
                    "id": 9,
                    "name": "c9",
                    "required": true,
                    "type": "timestamp"
                },
                {
                    "id": 10,
                    "name": "c10",
                    "required": true,
                    "type": "string"
                },
                {
                    "id": 11,
                    "name": "c11",
                    "required": true,
                    "type": "uuid"
                },
                {
                    "id": 12,
                    "name": "c12",
                    "required": true,
                    "type": "fixed[4]"
                },
                {
                    "id": 13,
                    "name": "c13",
                    "required": true,
                    "type": "binary"
                }
            ]
        }"#;

        let schema = serde_json::from_str::<Schema>(record)?;

        let result = HiveSchemaBuilder::from_iceberg(&schema, HiveVersion::default())?.build();

        let expected = vec![
            FieldSchema {
                name: Some("c1".into()),
                r#type: Some("boolean".into()),
                comment: None,
            },
            FieldSchema {
                name: Some("c2".into()),
                r#type: Some("int".into()),
                comment: None,
            },
            FieldSchema {
                name: Some("c3".into()),
                r#type: Some("bigint".into()),
                comment: None,
            },
            FieldSchema {
                name: Some("c4".into()),
                r#type: Some("float".into()),
                comment: None,
            },
            FieldSchema {
                name: Some("c5".into()),
                r#type: Some("double".into()),
                comment: None,
            },
            FieldSchema {
                name: Some("c6".into()),
                r#type: Some("decimal(2,2)".into()),
                comment: None,
            },
            FieldSchema {
                name: Some("c7".into()),
                r#type: Some("date".into()),
                comment: None,
            },
            FieldSchema {
                name: Some("c8".into()),
                r#type: Some("string".into()),
                comment: None,
            },
            FieldSchema {
                name: Some("c9".into()),
                r#type: Some("timestamp".into()),
                comment: None,
            },
            FieldSchema {
                name: Some("c10".into()),
                r#type: Some("string".into()),
                comment: None,
            },
            FieldSchema {
                name: Some("c11".into()),
                r#type: Some("string".into()),
                comment: None,
            },
            FieldSchema {
                name: Some("c12".into()),
                r#type: Some("binary".into()),
                comment: None,
            },
            FieldSchema {
                name: Some("c13".into()),
                r#type: Some("binary".into()),
                comment: None,
            },
        ];

        assert_eq!(result, expected);

        Ok(())
    }

    // RISK: `unknown` has no Hive column type (it is an always-null column with no physical
    // storage). Java's Hive type conversion has no UNKNOWN mapping and throws; the Rust converter
    // must fail loudly with FeatureUnsupported rather than fabricate a column type.
    #[test]
    fn test_unknown_type_conversion_is_rejected() {
        let mut builder = HiveSchemaBuilder::default();
        let error = builder
            .primitive(&iceberg::spec::PrimitiveType::Unknown)
            .expect_err("unknown has no Hive type");
        assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
        // Java message shape: type + " is not supported" (UnknownType.toString() = "unknown").
        assert!(
            error.to_string().contains("unknown is not supported"),
            "error must name the type, got: {error}"
        );
    }

    /// Builds a single-field schema around `field_type` and returns the converted
    /// Hive column type string.
    fn convert_single_field(
        field_type: iceberg::spec::PrimitiveType,
        hive_version: HiveVersion,
    ) -> Result<String> {
        let schema = Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                iceberg::spec::NestedField::required(
                    1,
                    "c1",
                    iceberg::spec::Type::Primitive(field_type),
                )
                .into(),
            ])
            .build()?;
        let result = HiveSchemaBuilder::from_iceberg(&schema, hive_version)?.build();
        result[0]
            .r#type
            .as_ref()
            .map(|t| t.to_string())
            .ok_or_else(|| Error::new(ErrorKind::Unexpected, "converted column has no type"))
    }

    // RISK: the version gate decides whether zone semantics survive into the Hive
    // column type. Java: HiveVersion.min(HIVE_3) && shouldAdjustToUTC() →
    // "timestamp with local time zone" (bytecode offsets 113-139). Pins BOTH the
    // default (Hive 3+) branch byte-exactly and that the default IS Hive 3+.
    #[test]
    fn test_timestamptz_micros_default_hive3_plus() -> Result<()> {
        let result = convert_single_field(PrimitiveType::Timestamptz, HiveVersion::default())?;
        assert_eq!(result, "timestamp with local time zone");
        Ok(())
    }

    // The explicit Hive 2 branch: below HIVE_3 the gate short-circuits and
    // timestamptz (µs) degrades to plain "timestamp" (never an error).
    #[test]
    fn test_timestamptz_micros_hive2_is_timestamp() -> Result<()> {
        let result = convert_single_field(PrimitiveType::Timestamptz, HiveVersion::Hive2)?;
        assert_eq!(result, "timestamp");
        Ok(())
    }

    // Naive timestamp (µs) is "timestamp" on EVERY Hive version — shouldAdjustToUTC()
    // is false, so the version gate must not affect it.
    #[test]
    fn test_naive_timestamp_micros_is_timestamp_on_both_versions() -> Result<()> {
        for hive_version in [HiveVersion::Hive3Plus, HiveVersion::Hive2] {
            let result = convert_single_field(PrimitiveType::Timestamp, hive_version)?;
            assert_eq!(result, "timestamp", "hive_version: {hive_version:?}");
        }
        Ok(())
    }

    // RISK: Java REJECTS both nano variants (TIMESTAMP_NANO has no $SwitchMap entry →
    // default throw, bytecode offsets 303-319) and never emits a "timestamp_ns"
    // column type. Emitting one would write a column type no Hive metastore knows.
    #[test]
    fn test_timestamp_nanos_rejected_both_variants() {
        for (field_type, type_name) in [
            (PrimitiveType::TimestampNs, "timestamp_ns"),
            (PrimitiveType::TimestamptzNs, "timestamptz_ns"),
        ] {
            for hive_version in [HiveVersion::Hive3Plus, HiveVersion::Hive2] {
                let error = convert_single_field(field_type.clone(), hive_version)
                    .expect_err("nano timestamps have no Hive type");
                assert_eq!(
                    error.kind(),
                    ErrorKind::FeatureUnsupported,
                    "type: {type_name}, hive_version: {hive_version:?}"
                );
                assert!(
                    error
                        .to_string()
                        .contains(&format!("{type_name} is not supported")),
                    "error must name the type, got: {error}"
                );
            }
        }
    }

    // RISK: separator-sensitive nested pin — Java joins struct fields with a bare ","
    // (bytecode offset 204) at EVERY nesting depth; a ", " join would emit column
    // type strings that differ from every Java-written Hive table.
    #[test]
    fn test_nested_struct_join_is_bare_comma() -> Result<()> {
        let record = r#"{
            "type": "struct",
            "schema-id": 1,
            "fields": [
                {
                    "id": 1,
                    "name": "outer",
                    "required": true,
                    "type": {
                        "type": "struct",
                        "fields": [
                            {"id": 2, "name": "a", "required": true, "type": "int"},
                            {
                                "id": 3,
                                "name": "b",
                                "required": true,
                                "type": {
                                    "type": "struct",
                                    "fields": [
                                        {"id": 4, "name": "c", "required": true, "type": "string"},
                                        {"id": 5, "name": "d", "required": true, "type": "long"}
                                    ]
                                }
                            },
                            {"id": 6, "name": "e", "required": true, "type": "boolean"}
                        ]
                    }
                }
            ]
        }"#;

        let schema = serde_json::from_str::<Schema>(record)?;
        let result = HiveSchemaBuilder::from_iceberg(&schema, HiveVersion::default())?.build();

        let expected = vec![FieldSchema {
            name: Some("outer".into()),
            r#type: Some("struct<a:int,b:struct<c:string,d:bigint>,e:boolean>".into()),
            comment: None,
        }];

        assert_eq!(result, expected);

        Ok(())
    }
}
