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

/// An `unknown`-bearing schema is refused at build, at every depth, with no file left behind.
#[tokio::test]
async fn an_unknown_schema_is_refused_before_any_bytes_are_written() {
    for (label, unknown_field) in [
        (
            "top level",
            NestedField::optional(2, "u", Type::Primitive(PrimitiveType::Unknown)),
        ),
        (
            "in a struct",
            NestedField::optional(
                2,
                "u",
                Type::Struct(StructType::new(vec![
                    NestedField::optional(3, "inner", Type::Primitive(PrimitiveType::Unknown))
                        .into(),
                ])),
            ),
        ),
        (
            "in a list",
            NestedField::optional(
                2,
                "u",
                Type::List(ListType {
                    element_field: NestedField::list_element(
                        3,
                        Type::Primitive(PrimitiveType::Unknown),
                        true,
                    )
                    .into(),
                }),
            ),
        ),
        (
            "as a map key",
            NestedField::optional(
                2,
                "u",
                Type::Map(MapType {
                    key_field: NestedField::map_key_element(
                        3,
                        Type::Primitive(PrimitiveType::Unknown),
                    )
                    .into(),
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
                "u",
                Type::Map(MapType {
                    key_field: NestedField::map_key_element(
                        3,
                        Type::Primitive(PrimitiveType::String),
                    )
                    .into(),
                    value_field: NestedField::map_value_element(
                        4,
                        Type::Primitive(PrimitiveType::Unknown),
                        true,
                    )
                    .into(),
                }),
            ),
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
            Ok(_) => panic!("an unknown schema must be refused at BUILD time ({label})"),
            Err(error) => error,
        };
        assert_eq!(
            error.kind(),
            ErrorKind::FeatureUnsupported,
            "unknown {label} must be refused"
        );
        assert!(
            error.message().contains("Writing the unknown column"),
            "the error must name the unknown write refusal for {label}, got: {}",
            error.message()
        );
        assert!(
            error.message().contains("unknown"),
            "the error must name the type for {label}, got: {}",
            error.message()
        );
        assert!(
            !std::path::Path::new(&path).exists(),
            "refusing at build time must leave NO file behind for {label}"
        );
    }
}
