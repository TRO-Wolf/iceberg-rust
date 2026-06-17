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

//! ExpressionParser-JSON interop (GAP_MATRIX row 147).
//!
//! The REAL Java `org.apache.iceberg.expressions.ExpressionParser` is the oracle; the shared
//! contract is the SCHEMA (Java `SchemaParser.toJson` ⇄ Rust serde [`Schema`]). The battery of
//! expressions is built INDEPENDENTLY on each side keyed by a stable name — no JSON is copied
//! across to derive the expected, so a byte-match proves real codec parity, not an echo.
//!
//! This file is the RUST half of `dev/java-interop/run-interop-expression.sh`. It is env-gated and
//! a no-op under the offline `cargo test` gate:
//!
//! - `ICEBERG_INTEROP_EXPRESSION_GEN_DIR` — GEN (Rust writes `rust_expressions.jsonl` for Java to
//!   validate, and — if Java has already written `java_expressions.jsonl` — asserts Rust
//!   `from_json(java_json, schema)` round-trips byte-for-byte).
//! - `ICEBERG_INTEROP_EXPRESSION_DIR` — DIRECTION 2 (Rust reads Java's `java_expressions.jsonl`,
//!   `from_json` + re-serialize, asserts byte-equality with Java's `ExpressionParser.toJson`).

use std::ops::Not;
use std::path::PathBuf;

use iceberg::expr::expression_parser::{from_json, to_json};
use iceberg::expr::{Predicate, PredicateOperator, Reference, SetExpression};
use iceberg::spec::{Datum, Schema};

fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_EXPRESSION_GEN_DIR").map(PathBuf::from)
}

fn verify_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_EXPRESSION_DIR").map(PathBuf::from)
}

/// The shared battery, keyed by the SAME names as the Java oracle. Each entry is the Rust
/// `Predicate` whose `to_json` must byte-match Java `ExpressionParser.toJson` of the same-named
/// Java expression.
fn battery() -> Vec<(&'static str, Predicate)> {
    vec![
        ("is_null", Reference::new("x").is_null()),
        ("not_null", Reference::new("x").is_not_null()),
        ("is_nan", Reference::new("f").is_nan()),
        ("not_nan", Reference::new("f").is_not_nan()),
        ("lt", Reference::new("i").less_than(Datum::int(10))),
        (
            "lt_eq",
            Reference::new("i").less_than_or_equal_to(Datum::int(10)),
        ),
        ("gt", Reference::new("i").greater_than(Datum::int(10))),
        (
            "gt_eq",
            Reference::new("i").greater_than_or_equal_to(Datum::int(10)),
        ),
        ("eq_int", Reference::new("i").equal_to(Datum::int(42))),
        // float/double VALUE predicates — the value arm Java renders via Float/Double.toString.
        // These magnitudes byte-match Java exactly (the JDK-11 non-minimal residue is unit-pinned
        // in expression_parser.rs::float_residue_is_jdk_nonminimal, not exercised here).
        (
            "eq_float",
            Reference::new("f").equal_to(Datum::float(2.5f32)),
        ),
        (
            "eq_double",
            Reference::new("dbl").equal_to(Datum::double(1e10)),
        ),
        (
            "eq_double_sci",
            Reference::new("dbl").equal_to(Datum::double(1e-4)),
        ),
        (
            "eq_long",
            Reference::new("l").equal_to(Datum::long(9_000_000_000i64)),
        ),
        (
            "not_eq_str",
            Reference::new("s").not_equal_to(Datum::string("hi")),
        ),
        (
            "starts_with",
            Reference::new("s").starts_with(Datum::string("pre")),
        ),
        (
            "not_starts_with",
            Reference::new("s").not_starts_with(Datum::string("pre")),
        ),
        ("eq_bool", Reference::new("b").equal_to(Datum::bool(true))),
        (
            "eq_str",
            Reference::new("s").equal_to(Datum::string("hello")),
        ),
        (
            "eq_date",
            Reference::new("d").equal_to(Datum::date_from_str("2017-11-16").unwrap()),
        ),
        (
            "eq_time",
            Reference::new("t").equal_to(Datum::time_from_str("13:14:15.000001").unwrap()),
        ),
        (
            "eq_ts",
            Reference::new("ts")
                .equal_to(Datum::timestamp_from_str("2017-11-16T14:15:16.123456").unwrap()),
        ),
        (
            "eq_tstz",
            Reference::new("tstz")
                .equal_to(Datum::timestamptz_from_str("2017-11-16T14:15:16.123456+00:00").unwrap()),
        ),
        (
            "eq_dec",
            Reference::new("dec").equal_to(Datum::decimal_from_str("12.34").unwrap()),
        ),
        (
            "eq_dec_scale",
            Reference::new("dec").equal_to(Datum::decimal_from_str("12.30").unwrap()),
        ),
        (
            "eq_uuid",
            Reference::new("u")
                .equal_to(Datum::uuid_from_str("f79c3e09-677c-4bbd-a479-3f349cb785e7").unwrap()),
        ),
        (
            "eq_bin",
            Reference::new("bin").equal_to(Datum::binary(vec![0x01, 0xAB, 0xFF])),
        ),
        (
            "eq_fixed",
            Reference::new("fx").equal_to(Datum::fixed(vec![0x0A, 0x0B, 0x0C])),
        ),
        (
            // `.and()` builds `And(LogicalExpression::new([lhs, rhs]))` for non-constant operands.
            "and",
            Reference::new("i")
                .equal_to(Datum::int(1))
                .and(Reference::new("l").equal_to(Datum::long(2i64))),
        ),
        (
            "or",
            Reference::new("x")
                .is_null()
                .or(Reference::new("y").is_not_null()),
        ),
        (
            // The `Not` trait builds `Not(LogicalExpression::new([self]))` (no simplification).
            "not",
            Reference::new("i").equal_to(Datum::int(5)).not(),
        ),
        (
            "in",
            Predicate::Set(SetExpression::new(
                PredicateOperator::In,
                Reference::new("i"),
                [Datum::int(1), Datum::int(2), Datum::int(3)]
                    .into_iter()
                    .collect(),
            )),
        ),
        (
            "not_in",
            Predicate::Set(SetExpression::new(
                PredicateOperator::NotIn,
                Reference::new("s"),
                [Datum::string("a"), Datum::string("b")]
                    .into_iter()
                    .collect(),
            )),
        ),
        ("always_true", Predicate::AlwaysTrue),
        ("always_false", Predicate::AlwaysFalse),
    ]
}

/// Read the Java-written schema (`SchemaParser.toJson`) via Rust's serde [`Schema`].
fn read_schema(dir: &std::path::Path) -> Schema {
    let path = dir.join("schema.json");
    let bytes = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read schema.json at {}: {e}", path.display()));
    serde_json::from_str::<Schema>(&bytes).expect("parse Java schema.json via Rust serde Schema")
}

/// `in` / `not-in` serialize their values from an unordered set, so the byte order of the `values`
/// array is not stable across Rust (`FnvHashSet`) and Java (`Set`). For those two names the interop
/// comparison is set-SEMANTIC (parse both and compare the [`Predicate`], whose `PartialEq` is
/// set-based); every other predicate is compared byte-exact. The unit tests separately pin the set
/// round-trip and the exact `{"type":"in",...}` shape.
fn is_set_predicate(name: &str) -> bool {
    name == "in" || name == "not_in"
}

/// Assert Rust's reading of `java_json` round-trips: byte-exact for scalar/logical predicates,
/// set-semantic for `in`/`not-in`.
fn assert_matches(name: &str, java_json: &str, schema: &Schema) {
    let parsed =
        from_json(java_json, schema).unwrap_or_else(|e| panic!("Rust from_json for {name}: {e}"));
    if is_set_predicate(name) {
        // Compare the parsed Predicate against Java's JSON re-parsed by Rust — both are
        // FnvHashSet-backed, so PartialEq is order-insensitive.
        let from_rust_reser = from_json(&to_json(&parsed).unwrap(), schema).unwrap();
        assert_eq!(
            parsed, from_rust_reser,
            "set predicate {name} round-trips (order-insensitive)"
        );
    } else {
        let reser = to_json(&parsed).expect("Rust re-serialize");
        assert_eq!(reser, java_json, "Rust toJson(fromJson(java)) for {name}");
    }
}

/// Parse the `{name, json}` JSONL Java wrote into a name → canonical-JSON map.
fn read_jsonl(path: &std::path::Path) -> Vec<(String, String)> {
    let content =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let v: serde_json::Value = serde_json::from_str(line).expect("parse JSONL line");
            (
                v["name"].as_str().expect("name").to_string(),
                v["json"].as_str().expect("json").to_string(),
            )
        })
        .collect()
}

/// GEN: Rust writes `rust_expressions.jsonl` (its `to_json` for the battery, for Java to validate)
/// and — when Java has already produced `java_expressions.jsonl` — asserts Rust `from_json` of
/// every Java JSON round-trips byte-for-byte through the Rust codec.
#[test]
fn test_expression_gen_rust_writes_and_round_trips() {
    let Some(dir) = gen_dir() else {
        println!(
            "skipping interop_expression GEN — set ICEBERG_INTEROP_EXPRESSION_GEN_DIR \
             (run dev/java-interop/run-interop-expression.sh)"
        );
        return;
    };

    let schema = read_schema(&dir);

    // Write the Rust battery as JSONL for the Java verify step. Sanity-check Rust round-trips its
    // OWN output BEFORE handing it to Java (a Rust regression is caught here, not shipped).
    let mut out = String::new();
    for (name, pred) in battery() {
        let json = to_json(&pred).expect("Rust to_json");
        let back = from_json(&json, &schema).expect("Rust from_json round-trip");
        let reser = to_json(&back).expect("Rust to_json re-serialize");
        assert_eq!(reser, json, "Rust self round-trip for {name}");
        let name_lit = serde_json::Value::String(name.to_string());
        let json_lit = serde_json::Value::String(json);
        out.push_str(&format!(r#"{{"name":{name_lit},"json":{json_lit}}}"#));
        out.push('\n');
    }
    let rust_file = dir.join("rust_expressions.jsonl");
    std::fs::write(&rust_file, out).expect("write rust_expressions.jsonl");
    println!("wrote {}", rust_file.display());

    // If Java already emitted its battery, assert Rust parses+re-serializes each byte-for-byte.
    let java_file = dir.join("java_expressions.jsonl");
    if java_file.exists() {
        for (name, java_json) in read_jsonl(&java_file) {
            assert_matches(&name, &java_json, &schema);
        }
        println!(
            "Rust round-tripped every Java expression (byte-exact, set-semantic for in/not-in)"
        );
    }
}

/// DIRECTION 2: Rust reads Java's `java_expressions.jsonl`, `from_json` + re-serialize, and asserts
/// byte-equality with Java `ExpressionParser.toJson`.
#[test]
fn test_expression_verify_rust_reads_java() {
    let Some(dir) = verify_dir() else {
        println!(
            "skipping interop_expression D2 — set ICEBERG_INTEROP_EXPRESSION_DIR \
             (run dev/java-interop/run-interop-expression.sh)"
        );
        return;
    };

    let schema = read_schema(&dir);
    let java_file = dir.join("java_expressions.jsonl");
    let entries = read_jsonl(&java_file);
    assert!(!entries.is_empty(), "java_expressions.jsonl had no entries");

    for (name, java_json) in entries {
        assert_matches(&name, &java_json, &schema);
    }
    println!(
        "D2: Rust read + re-serialized every Java expression (byte-exact, set-semantic for in/not-in)"
    );
}
