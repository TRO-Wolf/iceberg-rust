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

/// Property `iceberg.field.id` for `Column`
pub(crate) const ICEBERG_FIELD_ID: &str = "iceberg.field.id";
/// Property `iceberg.field.optional` for `Column`
pub(crate) const ICEBERG_FIELD_OPTIONAL: &str = "iceberg.field.optional";
/// Property `iceberg.field.current` for `Column`
pub(crate) const ICEBERG_FIELD_CURRENT: &str = "iceberg.field.current";

use std::collections::HashMap;

use aws_sdk_glue::types::Column;
use iceberg::Result;
use iceberg::spec::{PrimitiveType, SchemaVisitor, TableMetadata, visit_schema};

use crate::error::from_aws_build_error;

type GlueSchema = Vec<Column>;

#[derive(Debug, Default)]
pub(crate) struct GlueSchemaBuilder {
    schema: GlueSchema,
    is_current: bool,
    depth: usize,
}

impl GlueSchemaBuilder {
    /// Creates a new `GlueSchemaBuilder` from iceberg `Schema`
    pub fn from_iceberg(metadata: &TableMetadata) -> Result<GlueSchemaBuilder> {
        let current_schema = metadata.current_schema();

        let mut builder = Self {
            schema: Vec::new(),
            is_current: true,
            depth: 0,
        };

        visit_schema(current_schema, &mut builder)?;

        builder.is_current = false;

        for schema in metadata.schemas_iter() {
            if schema.schema_id() == current_schema.schema_id() {
                continue;
            }

            visit_schema(schema, &mut builder)?;
        }

        Ok(builder)
    }

    /// Returns the newly converted `GlueSchema`
    pub fn build(self) -> GlueSchema {
        self.schema
    }

    /// Check if is in `StructType` while traversing schema
    fn is_inside_struct(&self) -> bool {
        self.depth > 0
    }
}

impl SchemaVisitor for GlueSchemaBuilder {
    type T = String;

    fn schema(
        &mut self,
        _schema: &iceberg::spec::Schema,
        value: Self::T,
    ) -> iceberg::Result<String> {
        Ok(value)
    }

    fn before_struct_field(&mut self, _field: &iceberg::spec::NestedFieldRef) -> Result<()> {
        self.depth += 1;
        Ok(())
    }

    fn r#struct(
        &mut self,
        r#_struct: &iceberg::spec::StructType,
        results: Vec<String>,
    ) -> iceberg::Result<String> {
        // Java joins struct fields with a BARE comma, no space:
        // `fields.stream().map(...).collect(Collectors.joining(","))` wrapped in
        // `String.format("struct<%s>", ...)` — bytecode-verified (javap, iceberg-aws-1.10.0.jar):
        // `IcebergToGlueConverter.toTypeString` offsets 190 (`ldc "," → Collectors.joining`) and
        // 205 (`ldc "struct<%s>"`).
        Ok(format!("struct<{}>", results.join(",")))
    }

    fn after_struct_field(&mut self, _field: &iceberg::spec::NestedFieldRef) -> Result<()> {
        self.depth -= 1;
        Ok(())
    }

    fn field(
        &mut self,
        field: &iceberg::spec::NestedFieldRef,
        value: String,
    ) -> iceberg::Result<String> {
        if self.is_inside_struct() {
            // Java renders each struct field as `name:type` via the struct-field lambda
            // `String.format("%s:%s", field.name(), toTypeString(field.type()))` — bytecode-verified
            // (javap, iceberg-aws-1.10.0.jar): `IcebergToGlueConverter.lambda$toTypeString$2` (`ldc
            // "%s:%s"`, `NestedField.name()`, recursive `toTypeString(field.type())`). The name is
            // emitted verbatim (no case-folding) and the type is the recursive conversion of `value`.
            return Ok(format!("{}:{}", field.name, &value));
        }

        let parameters = HashMap::from([
            (ICEBERG_FIELD_ID.to_string(), format!("{}", field.id)),
            (
                ICEBERG_FIELD_OPTIONAL.to_string(),
                format!("{}", !field.required).to_lowercase(),
            ),
            (
                ICEBERG_FIELD_CURRENT.to_string(),
                format!("{}", self.is_current).to_lowercase(),
            ),
        ]);

        let mut builder = Column::builder()
            .name(field.name.clone())
            .r#type(&value)
            .set_parameters(Some(parameters));

        if let Some(comment) = field.doc.as_ref() {
            builder = builder.comment(comment);
        }

        let column = builder.build().map_err(from_aws_build_error)?;

        self.schema.push(column);

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

    fn primitive(&mut self, p: &iceberg::spec::PrimitiveType) -> iceberg::Result<Self::T> {
        let glue_type = match p {
            PrimitiveType::Boolean => "boolean".to_string(),
            PrimitiveType::Int => "int".to_string(),
            PrimitiveType::Long => "bigint".to_string(),
            PrimitiveType::Float => "float".to_string(),
            PrimitiveType::Double => "double".to_string(),
            PrimitiveType::Date => "date".to_string(),
            // Both micros timestamp variants share Java `Type.TypeID.TIMESTAMP`
            // (`TimestampType.typeId()` returns TIMESTAMP for BOTH zone variants), so Java's Glue
            // converter maps them via `case TIMESTAMP: return "timestamp"` — bytecode-verified
            // (javap, iceberg-aws-1.10.0.jar): `IcebergToGlueConverter.toTypeString` tableswitch
            // index 10 → offset 120 (`ldc "timestamp"`). The Glue column type is informational only
            // — the Iceberg metadata JSON is the source of truth — so the with/without-zone
            // distinction is not represented, exactly as in Java.
            PrimitiveType::Timestamp | PrimitiveType::Timestamptz => "timestamp".to_string(),
            // Both nanos timestamp variants share Java `Type.TypeID.TIMESTAMP_NANO`
            // (`TimestampNanoType.typeId()`), which has NO explicit case in
            // `IcebergToGlueConverter.toTypeString`, so Java falls through to `default: return
            // type.typeId().name().toLowerCase(Locale.ENGLISH)` = "timestamp_nano" — bytecode-verified
            // (javap, iceberg-aws-1.10.0.jar): the `$SwitchMap` references exactly 16 TypeIDs and
            // TIMESTAMP_NANO is not among them, so `toTypeString` hits the default at offsets
            // 291-306 (`typeId().name().toLowerCase(ENGLISH)`); `Type$TypeID.TIMESTAMP_NANO.name()`
            // is "TIMESTAMP_NANO" → "timestamp_nano". Java does NOT throw for either nano variant.
            PrimitiveType::TimestampNs | PrimitiveType::TimestamptzNs => {
                "timestamp_nano".to_string()
            }
            PrimitiveType::Time | PrimitiveType::String | PrimitiveType::Uuid => {
                "string".to_string()
            }
            PrimitiveType::Binary | PrimitiveType::Fixed(_) => "binary".to_string(),
            // `unknown` has no explicit case in `IcebergToGlueConverter.toTypeString`, so Java falls
            // through to the same `default: return typeId().name().toLowerCase(Locale.ENGLISH)` as
            // the nano variants — bytecode-verified (javap, iceberg-aws-1.10.0.jar): UNKNOWN is not
            // among the 16 `$SwitchMap` TypeIDs, and `Type$TypeID.UNKNOWN.name()` is "UNKNOWN" →
            // "unknown". Java does NOT throw in `toTypeString`; strict byte-parity renders "unknown".
            // (This is a display-only Glue column type — nothing in the crate parses it back — and
            // `unknown` remains an always-null column with no physical storage, as in Java.)
            PrimitiveType::Unknown => "unknown".to_string(),
            PrimitiveType::Decimal { precision, scale } => {
                format!("decimal({precision},{scale})")
            }
        };

        Ok(glue_type)
    }
}

#[cfg(test)]
mod tests {
    use iceberg::TableCreation;
    use iceberg::spec::{Schema, TableMetadataBuilder};

    use super::*;

    fn create_metadata(schema: Schema) -> Result<TableMetadata> {
        let table_creation = TableCreation::builder()
            .name("my_table".to_string())
            .location("my_location".to_string())
            .schema(schema)
            .build();
        let metadata = TableMetadataBuilder::from_table_creation(table_creation)?
            .build()?
            .metadata;

        Ok(metadata)
    }

    fn create_column(
        name: impl Into<String>,
        r#type: impl Into<String>,
        id: impl Into<String>,
        optional: bool,
    ) -> Result<Column> {
        let parameters = HashMap::from([
            (ICEBERG_FIELD_ID.to_string(), id.into()),
            (ICEBERG_FIELD_OPTIONAL.to_string(), optional.to_string()),
            (ICEBERG_FIELD_CURRENT.to_string(), "true".to_string()),
        ]);

        Column::builder()
            .name(name)
            .r#type(r#type)
            .set_comment(None)
            .set_parameters(Some(parameters))
            .build()
            .map_err(from_aws_build_error)
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
        let metadata = create_metadata(schema)?;

        let result = GlueSchemaBuilder::from_iceberg(&metadata)?.build();

        let expected = vec![
            create_column("c1", "boolean", "1", false)?,
            create_column("c2", "int", "2", false)?,
            create_column("c3", "bigint", "3", false)?,
            create_column("c4", "float", "4", false)?,
            create_column("c5", "double", "5", false)?,
            create_column("c6", "decimal(2,2)", "6", false)?,
            create_column("c7", "date", "7", false)?,
            create_column("c8", "string", "8", false)?,
            create_column("c9", "timestamp", "9", false)?,
            create_column("c10", "string", "10", false)?,
            create_column("c11", "string", "11", false)?,
            create_column("c12", "binary", "12", false)?,
            create_column("c13", "binary", "13", false)?,
        ];

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
        let metadata = create_metadata(schema)?;

        let result = GlueSchemaBuilder::from_iceberg(&metadata)?.build();

        let expected = vec![create_column(
            "person",
            "struct<name:string,age:int>",
            "1",
            false,
        )?];

        assert_eq!(result, expected);

        Ok(())
    }

    // C-6: struct-directly-inside-struct full-string pin. The bare-comma separator is load-bearing
    // at TWO nesting depths — between the outer struct's fields AND between the inner struct's
    // fields. Java oracle (bytecode-verified, iceberg-aws-1.10.0.jar): `struct<%s>` with fields
    // joined by `Collectors.joining(",")` and each field rendered `%s:%s` (name:type). Mutation
    // proof (C-6): reverting the struct join to `", "` reds this pin at both depths.
    #[test]
    fn test_schema_with_struct_inside_struct() -> Result<()> {
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
                            { "id": 2, "name": "a", "required": true, "type": "int" },
                            {
                                "id": 3,
                                "name": "inner",
                                "required": true,
                                "type": {
                                    "type": "struct",
                                    "fields": [
                                        { "id": 4, "name": "x", "required": true, "type": "string" },
                                        { "id": 5, "name": "y", "required": false, "type": "long" }
                                    ]
                                }
                            },
                            { "id": 6, "name": "b", "required": false, "type": "double" }
                        ]
                    }
                }
            ]
        }"#;

        let schema = serde_json::from_str::<Schema>(record)?;
        let metadata = create_metadata(schema)?;

        let result = GlueSchemaBuilder::from_iceberg(&metadata)?.build();

        let expected = vec![create_column(
            "outer",
            "struct<a:int,inner:struct<x:string,y:bigint>,b:double>",
            "1",
            false,
        )?];

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
        let metadata = create_metadata(schema)?;

        let result = GlueSchemaBuilder::from_iceberg(&metadata)?.build();

        let expected = vec![create_column(
            "location",
            "array<struct<latitude:float,longitude:float>>",
            "1",
            false,
        )?];

        assert_eq!(result, expected);

        Ok(())
    }

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
        let metadata = create_metadata(schema)?;

        let result = GlueSchemaBuilder::from_iceberg(&metadata)?.build();

        let expected = vec![create_column(
            "quux",
            "map<string,map<string,int>>",
            "1",
            false,
        )?];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_schema_with_optional_fields() -> Result<()> {
        let record = r#"{
            "type": "struct",
            "schema-id": 1,
            "fields": [
                {
                    "id": 1,
                    "name": "required_field",
                    "required": true,
                    "type": "string"
                },
                {
                    "id": 2,
                    "name": "optional_field",
                    "required": false,
                    "type": "int"
                }
            ]
        }"#;

        let schema = serde_json::from_str::<Schema>(record)?;
        let metadata = create_metadata(schema)?;

        let result = GlueSchemaBuilder::from_iceberg(&metadata)?.build();

        let expected = vec![
            create_column("required_field", "string", "1", false)?,
            create_column("optional_field", "int", "2", true)?,
        ];

        assert_eq!(result, expected);
        Ok(())
    }

    // Byte-parity pin (C-3): Java `IcebergToGlueConverter.toTypeString` has NO explicit UNKNOWN case,
    // so it falls to `default: typeId().name().toLowerCase(Locale.ENGLISH)` = "unknown" and NEVER
    // throws — bytecode-verified (javap, iceberg-aws-1.10.0.jar): UNKNOWN is not among the 16
    // `$SwitchMap` TypeIDs; `Type$TypeID.UNKNOWN.name()` = "UNKNOWN". The Glue column type is
    // display-only (nothing parses it back), so the fork renders "unknown" for strict byte-parity
    // rather than rejecting. Mutation proof (C-6): reverting this arm to a `FeatureUnsupported`
    // reject reds this test.
    #[test]
    fn test_unknown_type_maps_to_glue_unknown_string() -> Result<()> {
        let mut builder = GlueSchemaBuilder {
            schema: Vec::new(),
            is_current: true,
            depth: 0,
        };
        assert_eq!(builder.primitive(&PrimitiveType::Unknown)?, "unknown");
        Ok(())
    }

    // ENTRY / REPRO PIN (F-A2-4). A Spark-written table carries `timestamptz` columns (Spark
    // timestamps are timestamp-with-zone). Before the fix, `GlueSchemaBuilder::from_iceberg`
    // returned `FeatureUnsupported` for such a column; that error propagates through
    // `utils::convert_to_glue_table` (the sole `create_table` schema-build funnel, `utils.rs:149`)
    // and aborts table creation for EVERY Spark-shaped schema. Java parity:
    // `IcebergToGlueConverter.toTypeString` maps a timestamp-with-zone (Java `Type.TypeID.TIMESTAMP`,
    // shared by both zone variants — `TimestampType.typeId()`, `Types.java:271`) via
    // `case TIMESTAMP: return "timestamp"` (`IcebergToGlueConverter.java:315-316`, tag
    // apache-iceberg-1.10.0). The Glue column type is therefore exactly "timestamp".
    #[test]
    fn test_timestamptz_maps_to_glue_timestamp_string() -> Result<()> {
        let record = r#"{
            "type": "struct",
            "schema-id": 1,
            "fields": [
                {
                    "id": 1,
                    "name": "event_ts",
                    "required": true,
                    "type": "timestamptz"
                }
            ]
        }"#;

        let schema = serde_json::from_str::<Schema>(record)?;
        let metadata = create_metadata(schema)?;

        let result = GlueSchemaBuilder::from_iceberg(&metadata)?.build();

        let expected = vec![create_column("event_ts", "timestamp", "1", false)?];

        assert_eq!(result, expected);

        Ok(())
    }

    // Java-oracle type strings for the WHOLE timestamp family, asserted at the primitive funnel.
    // Oracle: `IcebergToGlueConverter.toTypeString` — bytecode-verified (javap, iceberg-aws-1.10.0.jar).
    //   * micros, both zone variants → Java `Type.TypeID.TIMESTAMP` → tableswitch index 10 → offset
    //     120 (`ldc "timestamp"`).
    //   * nanos, both zone variants → Java `Type.TypeID.TIMESTAMP_NANO`, which is NOT among the 16
    //     `$SwitchMap` TypeIDs → `default: typeId().name().toLowerCase(Locale.ENGLISH)` (offsets
    //     291-306) = "timestamp_nano" (`TIMESTAMP_NANO.name()` lowercased).
    // C-2: both nano variants now render "timestamp_nano" (Java's shared `TypeID.TIMESTAMP_NANO`
    // forces byte-identical output — `typeId()` has no zone/precision branch). This pin is
    // separator-agnostic; it locks the exact primitive strings across the whole timestamp family.
    // Mutation proof (C-6): reverting either nano arm to "timestamp_ns" reds this matrix.
    #[test]
    fn test_timestamp_family_glue_type_strings() -> Result<()> {
        let mut builder = GlueSchemaBuilder {
            schema: Vec::new(),
            is_current: true,
            depth: 0,
        };

        assert_eq!(builder.primitive(&PrimitiveType::Timestamp)?, "timestamp");
        assert_eq!(builder.primitive(&PrimitiveType::Timestamptz)?, "timestamp");
        assert_eq!(
            builder.primitive(&PrimitiveType::TimestampNs)?,
            "timestamp_nano"
        );
        assert_eq!(
            builder.primitive(&PrimitiveType::TimestamptzNs)?,
            "timestamp_nano"
        );

        Ok(())
    }

    // Acceptance-shaped schema (F-A2-4): a Spark-written CTAS target — `timestamptz` + string +
    // double + long at top level, PLUS a nested struct carrying a `timestamptz` leaf AND a list
    // whose element is a `timestamptz` — round-trips the FULL `from_iceberg` create_table
    // schema-build path (not just the primitive fn), proving nested timestamptz leaves normalize.
    // Matches Java `IcebergToGlueConverter.toTypeString`'s recursive struct/list handling —
    // bytecode-verified (javap, iceberg-aws-1.10.0.jar): the struct field lambda emits
    // `String.format("%s:%s", name, toTypeString(type))` joined by a BARE comma
    // (`Collectors.joining(",")`) inside `struct<%s>`, and lists render `array<%s>` — so the exact
    // strings are `struct<name:type,...>` (no space) / `array<type>`.
    #[test]
    fn test_acceptance_shaped_schema_with_nested_timestamptz() -> Result<()> {
        let record = r#"{
            "type": "struct",
            "schema-id": 1,
            "fields": [
                { "id": 1, "name": "event_ts", "required": true, "type": "timestamptz" },
                { "id": 2, "name": "name", "required": true, "type": "string" },
                { "id": 3, "name": "score", "required": false, "type": "double" },
                { "id": 4, "name": "seq", "required": true, "type": "long" },
                {
                    "id": 5,
                    "name": "meta",
                    "required": true,
                    "type": {
                        "type": "struct",
                        "fields": [
                            { "id": 6, "name": "created_at", "required": true, "type": "timestamptz" },
                            { "id": 7, "name": "label", "required": false, "type": "string" }
                        ]
                    }
                },
                {
                    "id": 8,
                    "name": "history",
                    "required": false,
                    "type": {
                        "type": "list",
                        "element-id": 9,
                        "element-required": true,
                        "element": "timestamptz"
                    }
                }
            ]
        }"#;

        let schema = serde_json::from_str::<Schema>(record)?;
        let metadata = create_metadata(schema)?;

        let result = GlueSchemaBuilder::from_iceberg(&metadata)?.build();

        let expected = vec![
            create_column("event_ts", "timestamp", "1", false)?,
            create_column("name", "string", "2", false)?,
            create_column("score", "double", "3", true)?,
            create_column("seq", "bigint", "4", false)?,
            create_column(
                "meta",
                "struct<created_at:timestamp,label:string>",
                "5",
                false,
            )?,
            // `from_table_creation` reassigns fresh sequential field ids: the six top-level fields
            // take ids 1..=6 in declaration order (so `history` is id 6, not the JSON-declared 8),
            // and the nested `created_at` / `label` / list-element take 7..=9.
            create_column("history", "array<timestamp>", "6", true)?,
        ];

        assert_eq!(result, expected);

        Ok(())
    }
}
