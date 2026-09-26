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

use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use super::ParquetWriterBuilder;
use crate::ErrorKind;
use crate::io::FileIO;
use crate::spec::{ListType, MapType, NestedField, PrimitiveType, Schema, StructType, Type};
use crate::writer::file_writer::FileWriterBuilder;

/// A variant-bearing schema is refused BEFORE any bytes are written, at every depth. Without
/// the guard the refusal lands in `close()`, leaving an orphan file.
#[tokio::test]
async fn a_variant_schema_is_refused_before_any_bytes_are_written() {
    for (label, variant_field) in [
        ("top level", NestedField::optional(2, "v", Type::Variant)),
        (
            "in a struct",
            NestedField::optional(
                2,
                "v",
                Type::Struct(StructType::new(vec![
                    NestedField::optional(3, "inner", Type::Variant).into(),
                ])),
            ),
        ),
        (
            "in a list",
            NestedField::optional(
                2,
                "v",
                Type::List(ListType {
                    element_field: NestedField::list_element(3, Type::Variant, true).into(),
                }),
            ),
        ),
        (
            "as a map key",
            NestedField::optional(
                2,
                "v",
                Type::Map(MapType {
                    key_field: NestedField::map_key_element(3, Type::Variant).into(),
                    value_field: NestedField::map_value_element(
                        4,
                        Type::Primitive(PrimitiveType::String),
                        true,
                    )
                    .into(),
                }),
            ),
        ),
        (
            "as a map value",
            NestedField::optional(
                2,
                "v",
                Type::Map(MapType {
                    key_field: NestedField::map_key_element(
                        3,
                        Type::Primitive(PrimitiveType::String),
                    )
                    .into(),
                    value_field: NestedField::map_value_element(4, Type::Variant, true).into(),
                }),
            ),
        ),
    ] {
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    variant_field.into(),
                ])
                .build()
                .expect("schema"),
        );

        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let path = temp_dir
            .path()
            .join("out.parquet")
            .to_string_lossy()
            .to_string();
        let output = file_io.new_output(&path).expect("output file");

        let error = match ParquetWriterBuilder::new(WriterProperties::builder().build(), schema)
            .build(output)
            .await
        {
            Ok(_) => panic!("a variant schema must be refused at BUILD time ({label})"),
            Err(error) => error,
        };
        assert_eq!(
            error.kind(),
            ErrorKind::FeatureUnsupported,
            "variant {label} must be refused"
        );
        assert!(
            error.message().contains("Writing the variant column"),
            "the error must name the write refusal for {label}, got: {}",
            error.message()
        );
        assert!(
            !std::path::Path::new(&path).exists(),
            "refusing at build time must leave NO file behind for {label}"
        );
    }
}

#[tokio::test]
async fn an_unknown_top_level_schema_builds_without_refusal() {
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "u", Type::Primitive(PrimitiveType::Unknown)).into(),
            ])
            .build()
            .expect("schema"),
    );

    let temp_dir = TempDir::new().expect("temp dir");
    let file_io = FileIO::new_with_fs();
    let path = temp_dir
        .path()
        .join("out.parquet")
        .to_string_lossy()
        .to_string();
    let output = file_io.new_output(&path).expect("output file");

    ParquetWriterBuilder::new(WriterProperties::builder().build(), schema)
        .build(output)
        .await
        .expect("an unknown schema must build without refusal");
}

fn unknown() -> Type {
    Type::Primitive(PrimitiveType::Unknown)
}

#[tokio::test]
async fn unknown_shapes_java_refuses_are_refused_before_any_bytes_are_written() {
    for (label, unknown_field, expected) in [
        (
            "in a struct",
            NestedField::optional(
                2,
                "u",
                Type::Struct(StructType::new(vec![
                    NestedField::optional(3, "inner", unknown()).into(),
                ])),
            ),
            "Cannot write struct 'u': every field is unknown, and Parquet refuses an empty group",
        ),
        (
            "in a list",
            NestedField::optional(
                2,
                "u",
                Type::List(ListType {
                    element_field: NestedField::list_element(3, unknown(), true).into(),
                }),
            ),
            "Cannot convert element Parquet: unknown (column 'u.element')",
        ),
        (
            "as a map key",
            NestedField::optional(
                2,
                "u",
                Type::Map(MapType {
                    key_field: NestedField::map_key_element(3, unknown()).into(),
                    value_field: NestedField::map_value_element(
                        4,
                        Type::Primitive(PrimitiveType::String),
                        true,
                    )
                    .into(),
                }),
            ),
            "Cannot convert key Parquet: unknown (column 'u.key')",
        ),
        (
            "as a map value",
            NestedField::optional(
                2,
                "u",
                Type::Map(MapType {
                    key_field: NestedField::map_key_element(
                        3,
                        Type::Primitive(PrimitiveType::String),
                    )
                    .into(),
                    value_field: NestedField::map_value_element(4, unknown(), true).into(),
                }),
            ),
            "Cannot convert value Parquet: unknown (column 'u.value')",
        ),
        (
            "in a struct inside a list",
            NestedField::optional(
                2,
                "u",
                Type::List(ListType {
                    element_field: NestedField::list_element(
                        3,
                        Type::Struct(StructType::new(vec![
                            NestedField::optional(4, "a", unknown()).into(),
                            NestedField::optional(5, "b", Type::Primitive(PrimitiveType::Long))
                                .into(),
                        ])),
                        true,
                    )
                    .into(),
                }),
            ),
            "Writing an unknown field under the element of 'u' is not supported",
        ),
    ] {
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    unknown_field.into(),
                ])
                .build()
                .expect("schema"),
        );
        let temp_dir = TempDir::new().expect("temp dir");
        let path = temp_dir
            .path()
            .join("out.parquet")
            .to_string_lossy()
            .to_string();
        let output = FileIO::new_with_fs()
            .new_output(&path)
            .expect("output file");

        let error = match ParquetWriterBuilder::new(WriterProperties::builder().build(), schema)
            .build(output)
            .await
        {
            Ok(_) => panic!("unknown {label} must be refused at BUILD time"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), ErrorKind::FeatureUnsupported, "{label}");
        assert!(
            error.message().contains(expected),
            "unknown {label}: expected '{expected}', got: {}",
            error.message()
        );
        assert!(
            !std::path::Path::new(&path).exists(),
            "refusing at build time must leave NO file behind for {label}"
        );
    }
}
