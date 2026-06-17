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

//! METADATA-LEVEL interop for the V3 `unknown` primitive type (BLOCK-2 G2).
//!
//! `unknown` (Java `org.apache.iceberg.types.Types.UnknownType`, typeId `UNKNOWN`, `toString()`
//! `"unknown"`) is an always-null column with **no physical storage** — Java `TypeToMessageType`
//! returns null for it, so no parquet column is emitted. Consequently the WHOLE contract for the
//! type is the schema/metadata round-trip: there are NO data files, and NO Docker. This test is the
//! byte-/field-id-level parity evidence that flips the GAP_MATRIX `unknown` row.
//!
//! It exercises BOTH directions against the Java `iceberg-core` 1.10.0 reference oracle in
//! `dev/java-interop/` (`UnknownTypeOracle`):
//!
//! - **Direction 1 (Rust reads what Java wrote).** Parse the committed Java-written
//!   `java.metadata.json` (produced by `TableMetadataParser.toJson` over a V3 schema carrying an
//!   `unknown` column at every placement — top-level, nested in a struct, as a list element, and as
//!   a map value) and assert the parsed Rust schema is **structurally equal** (recursive field id /
//!   name / type / required / doc / default via `StructType: PartialEq`) to the canonical schema this
//!   test builds. This runs in the normal offline suite — it reads the committed fixture, no Java /
//!   Docker needed. If Rust wrongly parsed `"unknown"` (dropping the column or reading a different
//!   type), this fails.
//! - **Direction 2 (Java reads what Rust writes).** When `ICEBERG_INTEROP_UNKNOWN_GEN_DIR` is set,
//!   write the Rust `TableMetadata` (serialized via `serde_json`) to `rust.metadata.json` in that dir
//!   for the Java oracle's `verify-interop-unknown` step. A normal `cargo test` run does NOT write
//!   files (the env var is unset); only `dev/java-interop/run-interop-unknown.sh` sets it.
//!
//! Comparison is by PARSING into the Rust model and asserting structural equality — NOT by comparing
//! raw JSON bytes (Jackson and serde_json differ in key order / whitespace). Logical table identity
//! *including field ids* is the contract.

use std::fs;
use std::path::{Path, PathBuf};

use iceberg::spec::{
    ListType, MapType, NestedField, PrimitiveType, Schema, StructType, TableMetadata, Type,
};

/// Root of the committed interop fixture, relative to the `iceberg` crate manifest.
fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/interop/unknown_type")
}

/// The canonical V3 schema with `unknown` at every placement, in the EXACT field-id layout Java's
/// `TableMetadata.newTableMetadata` assigns (top-level 1..5, then nested struct field=6, list
/// element=7, map key=8, map value=9). This MUST mirror `UnknownTypeOracle.unknownSchema()` in
/// `dev/java-interop/.../InteropOracle.java`.
fn canonical_unknown_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "u", Type::Primitive(PrimitiveType::Unknown)).into(),
            NestedField::optional(
                3,
                "payload",
                Type::Struct(StructType::new(vec![
                    NestedField::optional(6, "nested_u", Type::Primitive(PrimitiveType::Unknown))
                        .into(),
                ])),
            )
            .into(),
            NestedField::optional(
                4,
                "events",
                Type::List(ListType {
                    element_field: NestedField::list_element(
                        7,
                        Type::Primitive(PrimitiveType::Unknown),
                        false,
                    )
                    .into(),
                }),
            )
            .into(),
            NestedField::optional(
                5,
                "tags",
                Type::Map(MapType {
                    key_field: NestedField::map_key_element(
                        8,
                        Type::Primitive(PrimitiveType::String),
                    )
                    .into(),
                    value_field: NestedField::map_value_element(
                        9,
                        Type::Primitive(PrimitiveType::Unknown),
                        false,
                    )
                    .into(),
                }),
            )
            .into(),
        ])
        .build()
        .expect("build canonical unknown schema")
}

/// Load a `TableMetadata` fixture by file name.
fn load_metadata(file_name: &str) -> TableMetadata {
    let path = fixture_dir().join(file_name);
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str::<TableMetadata>(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// Direction 1: Rust parses the Java-written `unknown` metadata and the schema struct matches the
/// canonical one (field id / name / type / required, recursively).
///
/// RISK: a dropped `unknown` column, a wrong-typed parse (e.g. resolving `"unknown"` to some other
/// primitive), or a lost nested placement would silently break round-trip interop with Java-written
/// V3 tables. `StructType: PartialEq` compares the full recursive shape, so an equal `as_struct()`
/// pins all of it at once. Also asserts the metadata is V3 (the gate keeps `unknown` off older
/// tables) — a V2 fixture would itself be rejected by `check_compatibility` on the add path.
#[test]
fn test_unknown_metadata_read_parity_with_java() {
    use iceberg::spec::FormatVersion;

    let java = load_metadata("java.metadata.json");
    assert_eq!(
        java.format_version(),
        FormatVersion::V3,
        "the unknown fixture must be a V3 table (Java MIN_FORMAT_VERSIONS gates unknown at v3)",
    );

    let java_schema = java.current_schema();
    let canonical = canonical_unknown_schema();
    assert_eq!(
        java_schema.as_struct(),
        canonical.as_struct(),
        "the Java-written unknown schema must parse to the canonical struct (field id / name / \
         type / required, recursively)",
    );

    // The schema must demand V3 through the SAME gate as timestamp_ns / variant.
    assert_eq!(
        java_schema.min_format_version(),
        FormatVersion::V3,
        "an unknown schema must demand v3",
    );

    // Direction 2 generator: only when the env var is set (the run-script sets it; a plain
    // `cargo test` leaves it unset and writes nothing).
    if let Ok(gen_dir) = std::env::var("ICEBERG_INTEROP_UNKNOWN_GEN_DIR") {
        // Re-serialize the parsed Java metadata so the bytes are Rust's own serializer output; the
        // schema (with the unknown column) is what the Java verify step parses and compares.
        let json = serde_json::to_string_pretty(&java)
            .expect("serialize unknown metadata for the Java verify step");
        let out_path = Path::new(&gen_dir).join("rust.metadata.json");
        fs::write(&out_path, json)
            .unwrap_or_else(|error| panic!("write {}: {error}", out_path.display()));
    }
}

/// A bare-schema JSON round-trip: `unknown` serializes to the bare lowercase string in every
/// placement and re-parses to `PrimitiveType::Unknown`. This is the most direct evidence of the
/// on-disk schema-token contract, independent of the surrounding table metadata.
///
/// RISK: a serializer that emitted an object wrapper or a different token for `unknown` would write
/// schema JSON Java cannot read. Parsing the Java-written `java.schema.json` and re-serializing it,
/// then asserting structural equality, pins the token both directions.
#[test]
fn test_unknown_schema_json_round_trip() {
    let path = fixture_dir().join("java.schema.json");
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));

    let schema: Schema = serde_json::from_str(&json)
        .unwrap_or_else(|error| panic!("parse java.schema.json: {error}"));
    assert_eq!(
        schema.as_struct(),
        canonical_unknown_schema().as_struct(),
        "the Java-written bare schema JSON must parse to the canonical unknown struct",
    );

    // Re-serialize and re-parse: Rust's own emitted token must round-trip back to the same struct.
    let reser = serde_json::to_string(&schema).expect("re-serialize schema");
    let reparsed: Schema =
        serde_json::from_str(&reser).expect("re-parse Rust-serialized unknown schema");
    assert_eq!(
        reparsed.as_struct(),
        canonical_unknown_schema().as_struct(),
        "Rust must round-trip the unknown token through its own serializer",
    );
    // The raw token must be the bare lowercase string `"unknown"` (not an object wrapper).
    assert!(
        reser.contains(r#""unknown""#),
        "the serialized schema must carry the bare \"unknown\" token, got: {reser}",
    );
}
