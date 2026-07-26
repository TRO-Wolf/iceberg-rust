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

//! Partition-path URL-escaping interop (GAP_MATRIX row R161).
//!
//! `PartitionSpec::partition_to_path` is a PURE function (spec + schema + tuple -> String) with no
//! on-disk artifact, so this is a CROSS-IMPL CONFORMANCE oracle rather than a byte round-trip: the
//! Java half (`InteropOracle$PartitionPathOracle`) emits its OWN
//! `org.apache.iceberg.PartitionSpec.partitionToPath` output for each NAMED case, and this test
//! rebuilds the SAME case INDEPENDENTLY from the battery below — keyed by the same id — and
//! byte-compares. No input travels across the boundary, so a match cannot be an echo.
//!
//! This file is the RUST half of `dev/java-interop/run-interop-partition-path.sh`. It is env-gated
//! and a clean no-op under the offline `cargo test` gate: without
//! `ICEBERG_INTEROP_PARTITION_PATH_DIR` it asserts only that the battery is internally consistent.
//!
//! Scope: the battery stays on the types whose HUMAN STRING is identical on both engines
//! (string/int identity, bucket, truncate, void), so a failure here is an ESCAPING divergence. The
//! temporal/binary human-string renderings diverge for reasons that predate R161 and are recorded
//! as a named residue on that row; they are deliberately not exercised here.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use iceberg::spec::{
    Literal, NestedField, PartitionSpec, PrimitiveType, Schema, SchemaRef, Struct, Transform, Type,
};

fn interop_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_PARTITION_PATH_DIR").map(PathBuf::from)
}

/// The two-column schema every case below binds to — the same shape the Java oracle declares.
fn schema() -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, "s", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(2, "i", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .expect("the two-column interop schema must build"),
    )
}

/// A one-field spec over `source` under partition-field name `name`.
fn one_field(source: &str, name: &str, transform: Transform) -> PartitionSpec {
    PartitionSpec::builder(schema())
        .add_partition_field(source, name, transform)
        .expect("the interop partition field must be legal")
        .build()
        .expect("the one-field interop spec must build")
}

/// `identity(s)` under `name`, over a single-slot tuple holding `value` (`None` = NULL).
fn identity_string(name: &str, value: Option<&str>) -> String {
    let spec = one_field("s", name, Transform::Identity);
    let data = Struct::from_iter([value.map(Literal::string)]);
    spec.partition_to_path(&data, schema())
}

/// The battery, keyed by the SAME case ids the Java oracle uses. Each entry is built here from
/// first principles — nothing is read from Java's JSON to construct it.
fn battery() -> BTreeMap<&'static str, String> {
    let mut cases: BTreeMap<&'static str, String> = BTreeMap::new();

    // Safe-set controls — byte-identical to an unescaped rendering.
    cases.insert("plain_string", identity_string("s", Some("alice")));
    cases.insert("safe_dashes", identity_string("dt", Some("2024-01-31")));
    cases.insert("safe_punctuation", identity_string("s", Some("-_.*")));

    // Value-side escaping.
    cases.insert("slash_value", identity_string("s", Some("a/b")));
    cases.insert("space_value", identity_string("s", Some("a b")));
    cases.insert("plus_value", identity_string("s", Some("a+b")));
    cases.insert("percent_value", identity_string("s", Some("a%b")));
    cases.insert("equals_value", identity_string("s", Some("a=b")));
    cases.insert("ampersand_value", identity_string("s", Some("a&b")));
    cases.insert("empty_value", identity_string("s", Some("")));
    cases.insert("newline_value", identity_string("s", Some("a\nb")));
    cases.insert("unicode_2byte", identity_string("s", Some("\u{e9}")));
    cases.insert(
        "unicode_3byte",
        identity_string("s", Some("\u{4e2d}\u{6587}")),
    );
    cases.insert("unicode_4byte", identity_string("s", Some("\u{1f600}")));

    // Name-side escaping, and the separate NULL branch.
    cases.insert("nasty_name", identity_string("a/b c=d", Some("v")));
    cases.insert("null_value", identity_string("s", None));
    cases.insert("nasty_name_null", identity_string("a/b c=d", None));

    // Multi-field: the `/` between pairs and the `=` inside a pair stay raw.
    let multi = PartitionSpec::builder(schema())
        .add_partition_field("s", "a b", Transform::Identity)
        .expect("identity(s) as `a b` is legal")
        .add_partition_field("i", "c/d", Transform::Identity)
        .expect("identity(i) as `c/d` is legal")
        .build()
        .expect("the two-field interop spec must build");
    cases.insert(
        "multi_field",
        multi.partition_to_path(
            &Struct::from_iter([Some(Literal::string("x/y")), Some(Literal::int(5))]),
            schema(),
        ),
    );

    // Non-identity transforms on the types whose human string matches on both engines.
    cases.insert(
        "int_identity",
        one_field("i", "i", Transform::Identity)
            .partition_to_path(&Struct::from_iter([Some(Literal::int(42))]), schema()),
    );
    cases.insert(
        "bucket_int",
        one_field("s", "s_bucket", Transform::Bucket(16))
            .partition_to_path(&Struct::from_iter([Some(Literal::int(7))]), schema()),
    );
    cases.insert(
        "truncate_string",
        one_field("s", "s_trunc", Transform::Truncate(4)).partition_to_path(
            &Struct::from_iter([Some(Literal::string("a/b c"))]),
            schema(),
        ),
    );
    cases.insert(
        "void_null",
        one_field("s", "s_void", Transform::Void)
            .partition_to_path(&Struct::from_iter([None]), schema()),
    );

    cases
}

/// Offline guard: the battery builds and every case renders a non-empty `name=value` pair. Keeps
/// the file honest in the default `cargo test` gate, where no Java fixture exists.
#[test]
fn partition_path_battery_is_well_formed() {
    let cases = battery();
    assert_eq!(cases.len(), 22, "the battery lost cases");
    for (id, path) in &cases {
        assert!(
            path.contains('='),
            "case {id} must render at least one `name=value` pair, got {path:?}"
        );
    }
    // A slash inside a VALUE never survives into the path; the multi-field case is the only one
    // carrying a raw `/`, as the pair separator.
    for (id, path) in &cases {
        let expected_slashes = usize::from(*id == "multi_field");
        assert_eq!(
            path.matches('/').count(),
            expected_slashes,
            "case {id} has an unexpected number of raw `/` in {path:?}"
        );
    }
}

/// DIRECTION 1 — "Rust renders what JAVA renders". Env-gated on
/// `ICEBERG_INTEROP_PARTITION_PATH_DIR`; a clean no-op without it.
#[test]
fn interop_partition_path_matches_java() {
    let Some(dir) = interop_dir() else {
        return;
    };

    let fixture = dir.join("java_partition_paths.json");
    let json = std::fs::read_to_string(&fixture)
        .unwrap_or_else(|error| panic!("read {}: {error}", fixture.display()));
    let parsed: serde_json::Value = serde_json::from_str(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", fixture.display()));
    let java_cases = parsed
        .get("cases")
        .and_then(serde_json::Value::as_object)
        .unwrap_or_else(|| panic!("{} has no `cases` object", fixture.display()));

    let rust_cases = battery();

    // The id SETS must match exactly — otherwise a silently-dropped case would look like a pass.
    let java_ids: Vec<&str> = java_cases.keys().map(String::as_str).collect();
    let mut rust_ids: Vec<&str> = rust_cases.keys().copied().collect();
    rust_ids.sort_unstable();
    let mut sorted_java_ids = java_ids.clone();
    sorted_java_ids.sort_unstable();
    assert_eq!(
        rust_ids, sorted_java_ids,
        "the Rust battery and the Java oracle disagree on the case set"
    );
    assert!(
        !java_ids.is_empty(),
        "the Java fixture carries no cases — the comparison would be vacuous"
    );

    let mut compared = 0usize;
    for (id, rust_path) in &rust_cases {
        let java_path = java_cases
            .get(*id)
            .and_then(serde_json::Value::as_str)
            .unwrap_or_else(|| panic!("Java fixture has no string path for case {id}"));
        assert_eq!(
            rust_path, java_path,
            "case {id}: Rust `partition_to_path` must byte-match Java `partitionToPath`"
        );
        compared += 1;
    }
    assert_eq!(compared, 22, "every battery case must have been compared");

    println!("interop_partition_path: {compared} cases byte-match Java partitionToPath");
}
