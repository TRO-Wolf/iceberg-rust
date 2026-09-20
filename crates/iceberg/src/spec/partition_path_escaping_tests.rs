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

//! R161 pins: BOTH sides of every `name=value` pair are escaped, exactly as Java does.
//!
//! Java `partitionToPath` appends `escape(name)`, `"="`, `escape(humanString)` per field and
//! joins the pairs with a raw `"/"`; those two separators are STRUCTURE and stay raw. `escape`
//! is `java.net.URLEncoder.encode(s, "UTF-8")`. Every expectation below is a verbatim jar-oracle
//! result against `iceberg-api-1.10.0` (`dev/java-interop/run-interop-partition-path.sh`).

use std::sync::Arc;

use super::*;
use crate::spec::{Literal, PrimitiveType, Type};

/// A one-column `s: string` schema — the binding target for every one-field spec below.
fn string_schema() -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, "s", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("one-column schema must build"),
    )
}

/// `identity(s)` exposed under `field_name`.
fn string_spec(field_name: &str) -> PartitionSpec {
    PartitionSpec::builder(string_schema())
        .add_partition_field("s", field_name, Transform::Identity)
        .expect("identity(s) under an arbitrary partition-field name is legal")
        .build()
        .expect("the one-field spec must build")
}

/// Render `field_name=value` through EVERY public entry point and assert they agree — the
/// total path, the fallible path, and `PartitionKey::to_path`.
fn render(field_name: &str, value: Option<&str>) -> String {
    let schema = string_schema();
    let spec = string_spec(field_name);
    let data = Struct::from_iter([value.map(Literal::string)]);

    let total = spec.partition_to_path(&data, schema.clone());
    let fallible = spec
        .try_partition_to_path(&data, schema.clone())
        .expect("a well-formed (spec, schema, tuple) triple must not error");
    let via_key = PartitionKey::new(spec, schema, data)
        .expect("PartitionKey::new: valid partition tuple")
        .to_path();

    assert_eq!(
        total, fallible,
        "the total and fallible paths must render identically"
    );
    assert_eq!(
        total, via_key,
        "`PartitionKey::to_path` must render identically"
    );
    total
}

// The escaper itself — a full printable-ASCII sweep against Java's `URLEncoder`.

/// The printable-ASCII characters `URLEncoder.encode(s, "UTF-8")` leaves untouched, verbatim
/// from the jar sweep over `0x20..=0x7E` (note: a space is NOT here — it maps to `+`).
const JAVA_SAFE_ASCII: &str = "*-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";

/// Every printable-ASCII partition value renders exactly as Java's `URLEncoder` renders it: the
/// 66 safe characters pass through, a space becomes `+`, and the remaining 28 become `%XX`.
#[test]
fn printable_ascii_sweep_matches_java_url_encoder() {
    let mut passed_through = 0usize;
    let mut percent_encoded = 0usize;
    for byte in 0x20u8..=0x7Eu8 {
        let ch = char::from(byte);
        let expected = if JAVA_SAFE_ASCII.contains(ch) {
            passed_through += 1;
            ch.to_string()
        } else if ch == ' ' {
            "+".to_string()
        } else {
            percent_encoded += 1;
            format!("%{byte:02X}")
        };
        assert_eq!(
            render("s", Some(&ch.to_string())),
            format!("s={expected}"),
            "ASCII 0x{byte:02X} ({ch:?}) must render as Java's URLEncoder renders it"
        );
    }
    assert_eq!(
        passed_through, 66,
        "the URLEncoder safe set is `A-Z a-z 0-9 - _ . *` — 66 printable-ASCII characters"
    );
    assert_eq!(
        percent_encoded, 28,
        "95 printable ASCII = 66 safe + 1 space + 28 percent-encoded"
    );
}

// The VALUE side — jar-oracle table.

/// `identity(s: string)` named `s`: (partition value, Java `partitionToPath`).
const JAVA_IDENTITY_STRING_PATHS: &[(&str, &str)] = &[
    ("plain", "s=plain"),
    ("AZaz09", "s=AZaz09"),
    ("-_.*", "s=-_.*"),
    ("a/b", "s=a%2Fb"),
    ("a b", "s=a+b"),
    ("a+b", "s=a%2Bb"),
    ("a%b", "s=a%25b"),
    ("a=b", "s=a%3Db"),
    ("a&b", "s=a%26b"),
    ("a?b", "s=a%3Fb"),
    ("a#b", "s=a%23b"),
    ("a:b", "s=a%3Ab"),
    ("a~b", "s=a%7Eb"),
    ("a!b", "s=a%21b"),
    ("a'b", "s=a%27b"),
    ("a(b)c", "s=a%28b%29c"),
    ("a,b", "s=a%2Cb"),
    ("a;b", "s=a%3Bb"),
    ("a@b", "s=a%40b"),
    ("a$b", "s=a%24b"),
    ("", "s="),
    ("  ", "s=++"),
    ("\u{e9}", "s=%C3%A9"),
    ("\u{4e2d}\u{6587}", "s=%E4%B8%AD%E6%96%87"),
    ("\u{1f600}", "s=%F0%9F%98%80"),
    ("x\u{e9} / y", "s=x%C3%A9+%2F+y"),
    ("%2F", "s=%252F"),
    ("a\nb", "s=a%0Ab"),
    ("..", "s=.."),
    (".", "s=."),
    ("null", "s=null"),
];

/// Every value in the jar-oracle table renders byte-identically to Java, including multi-byte
/// UTF-8: one `%XX` group per UTF-8 byte, never per `char`.
#[test]
fn identity_string_values_match_java() {
    for (value, expected) in JAVA_IDENTITY_STRING_PATHS {
        assert_eq!(
            &render("s", Some(value)),
            expected,
            "partition value {value:?} must render exactly as Java does"
        );
    }
    assert_eq!(
        JAVA_IDENTITY_STRING_PATHS.len(),
        31,
        "the jar-oracle value table lost rows"
    );
}

// The NAME side — Java escapes it too.

/// `identity(s)` under a tricky partition-field NAME, value `"v"`: (field name, Java path).
const JAVA_FIELD_NAME_PATHS: &[(&str, &str)] = &[
    ("weird name", "weird+name=v"),
    ("a/b", "a%2Fb=v"),
    ("a=b", "a%3Db=v"),
    ("a%b", "a%25b=v"),
    ("s_bucket", "s_bucket=v"),
    ("x\u{e9}", "x%C3%A9=v"),
    ("a+b", "a%2Bb=v"),
    ("*star*", "*star*=v"),
];

/// The partition-field NAME goes through the same escaper as the value (Java escapes both
/// sides; escaping only the value would still let a `/` in a field name forge a directory).
#[test]
fn field_names_match_java() {
    for (field_name, expected) in JAVA_FIELD_NAME_PATHS {
        assert_eq!(
            &render(field_name, Some("v")),
            expected,
            "partition-field name {field_name:?} must render exactly as Java does"
        );
    }
    assert_eq!(
        JAVA_FIELD_NAME_PATHS.len(),
        8,
        "the jar-oracle field-name table lost rows"
    );
}

/// A NULL partition value stays the literal `null`, and the NAME is still escaped on that
/// branch. The `name=null` fallbacks are a separate code path and need their own pin.
#[test]
fn null_values_keep_rendering_null_with_an_escaped_name() {
    const JAVA_FIELD_NAME_NULL_PATHS: &[(&str, &str)] = &[
        ("a/b", "a%2Fb=null"),
        ("weird name", "weird+name=null"),
        ("a%b", "a%25b=null"),
    ];
    for (field_name, expected) in JAVA_FIELD_NAME_NULL_PATHS {
        assert_eq!(
            &render(field_name, None),
            expected,
            "a NULL value under field name {field_name:?} must render exactly as Java does"
        );
    }
    // A string value that literally reads "null" is indistinguishable from a NULL value — the
    // same ambiguity Java has, pinned so nobody "fixes" it into a divergence.
    assert_eq!(render("s", Some("null")), render("s", None));
}

/// `name=null` is emitted from THREE sites, and each needs its own pin: a mutation of one is
/// invisible to the others. Java `Transform.toHumanString` returns the literal `"null"` before
/// it switches on the type.
///
/// Site 1 is the lenient fallback in `partition_to_path`. The commit path pairs a file's older
/// spec with the current schema, so an unescaped name puts a raw `/` into a `partitions.` key.
#[test]
fn the_lenient_fallback_null_still_escapes_the_field_name() {
    let spec = string_spec("a/b");
    // A schema without source id 1: the field's partition type is not derivable, so the total
    // path falls back to `null` for it.
    let evolved: SchemaRef = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::optional(2, "other", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("the evolved schema must build"),
    );
    // A value that WOULD have rendered, to prove the fallback is what emits the pair.
    let data = Struct::from_iter([Some(Literal::string("x/y"))]);

    let path = spec.partition_to_path(&data, evolved.clone());
    assert_eq!(
        path, "a%2Fb=null",
        "the lenient fallback must escape the field name exactly as Java does"
    );
    assert_eq!(
        path.matches('/').count(),
        0,
        "a `/` in the field name must not forge a directory level on the fallback branch"
    );
    // Fixture sanity: this branch is reached only because the triple is inconsistent.
    let err = spec
        .try_partition_to_path(&data, evolved)
        .expect_err("a dropped source column must be a typed error on the fallible path");
    assert_eq!(err.kind(), crate::ErrorKind::Unexpected);
}

/// Site 2 of three: the `void`-past-the-end-of-tuple branch in `render_partition_field`. An
/// all-`void` spec reports `is_unpartitioned()`, so an empty tuple reaches `name=null` here.
#[test]
fn the_void_past_end_null_still_escapes_the_field_name() {
    let schema: SchemaRef = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, "s", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("one-column schema must build"),
    );
    let spec = PartitionSpec::builder(schema.clone())
        .add_partition_field("s", "c/d", Transform::Void)
        .expect("void(s) under an arbitrary partition-field name is legal")
        .build()
        .expect("the one-field void spec must build");
    assert!(
        spec.is_unpartitioned(),
        "fixture sanity: an all-void spec reports unpartitioned, which is why callers pair it \
             with an empty tuple"
    );
    let data = Struct::empty();

    let path = spec.partition_to_path(&data, schema.clone());
    assert_eq!(
        path, "c%2Fd=null",
        "the void-past-end branch must escape the field name exactly as Java does"
    );
    assert_eq!(
        path.matches('/').count(),
        0,
        "a `/` in the field name must not forge a directory level on the void branch"
    );
    assert_eq!(
        spec.try_partition_to_path(&data, schema)
            .expect("an all-void spec paired with an empty tuple is legitimate"),
        path,
        "the fallible path must render the same escaped pair"
    );
}

// Structure vs. content.

/// The `/` between pairs and the `=` inside a pair are STRUCTURE — they stay raw — while a `/`
/// or `=` inside a name or a value is CONTENT and is escaped.
#[test]
fn pair_and_field_separators_stay_raw() {
    let schema: SchemaRef = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, "s", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(2, "i", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .expect("two-column schema must build"),
    );
    let spec = PartitionSpec::builder(schema.clone())
        .add_partition_field("s", "a b", Transform::Identity)
        .expect("identity(s) as `a b` is legal")
        .add_partition_field("i", "c/d", Transform::Identity)
        .expect("identity(i) as `c/d` is legal")
        .build()
        .expect("the two-field spec must build");
    let data = Struct::from_iter([Some(Literal::string("x/y")), Some(Literal::int(5))]);

    // Jar oracle: `a+b=x%2Fy/c%2Fd=5`.
    let path = spec.partition_to_path(&data, schema);
    assert_eq!(path, "a+b=x%2Fy/c%2Fd=5");
    assert_eq!(
        path.matches('/').count(),
        1,
        "exactly ONE raw `/` — the separator between the two pairs"
    );
    assert_eq!(
        path.matches('=').count(),
        2,
        "exactly TWO raw `=` — one per pair"
    );
}

/// The headline safety property: a `/` inside a partition VALUE can no longer forge an extra
/// directory level in a data file's location (nor a bogus `partitions.` summary key).
#[test]
fn a_slash_in_a_value_cannot_forge_a_directory_level() {
    let path = render("s", Some("a/b/c"));
    assert_eq!(path, "s=a%2Fb%2Fc");
    assert_eq!(
        path.matches('/').count(),
        0,
        "a single-field path must contain no raw `/` whatever the value holds"
    );
}

/// A space and a `+` must not collide: Java maps space to `+` and `+` to `%2B`, so the two
/// values keep distinct paths (a naive "escape `/` only" fix would collapse them).
#[test]
fn space_and_plus_stay_distinct() {
    assert_eq!(render("s", Some("a b")), "s=a+b");
    assert_eq!(render("s", Some("a+b")), "s=a%2Bb");
    assert_ne!(render("s", Some("a b")), render("s", Some("a+b")));
}

// The no-churn invariant — the overwhelmingly common case must be BYTE-IDENTICAL to pre-R161.

/// Every partition value inside the URLEncoder safe set renders EXACTLY as it did before R161:
/// no `%XX`, no `+`. This keeps ordinary table layouts unchanged, and it fails loudly under an
/// over-eager escaper such as RFC-3986 `NON_ALPHANUMERIC`, which mangles `-`, `_`, `.` and `*`.
#[test]
fn safe_partition_values_are_byte_identical_to_the_unescaped_rendering() {
    const COMMON: &[(&str, &str)] = &[
        ("dt", "2024-01-31"),
        ("category", "electronics"),
        ("id", "42"),
        ("region", "us-east-1"),
        ("s_bucket", "7"),
        ("amount", "-12.34"),
        ("uu", "f79c3e09-677c-4bbd-a479-3f349cb785e7"),
        ("star.name_1", "*star.name_1*"),
        ("empty_void", "null"),
    ];
    for (field_name, value) in COMMON {
        assert_eq!(
            &render(field_name, Some(value)),
            &format!("{field_name}={value}"),
            "a safe-set partition value must render byte-identically to pre-R161"
        );
    }
}

/// The pre-R161 fixture from `tests::test_partition_to_path` is byte-stable: a realistic
/// four-field path over plain values is untouched by the escaper.
#[test]
fn the_pre_r161_multi_field_fixture_is_byte_stable() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "timestamp", Type::Primitive(PrimitiveType::Timestamp)).into(),
            NestedField::required(4, "empty", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("the four-column schema must build");
    let spec = PartitionSpec::builder(schema.clone())
        .add_partition_field("id", "id", Transform::Identity)
        .expect("identity(id) is legal")
        .add_partition_field("name", "name", Transform::Identity)
        .expect("identity(name) is legal")
        .add_partition_field("timestamp", "ts_hour", Transform::Hour)
        .expect("hour(timestamp) is legal")
        .add_partition_field("empty", "empty_void", Transform::Void)
        .expect("void(empty) is legal")
        .build()
        .expect("the four-field spec must build");
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

/// Which value CLASSES move under R161. FIVE fork-supported column types render a human string
/// containing `:`, four of them a space too, so their path changes for EVERY value, the V3
/// nanosecond pair included. `date` is the byte-stable control in the same tuple.
///
/// The four `assert_ne!`s ALARM on the named human-string residue: when one becomes equal, that
/// residue is closed and row R161 must change in the same commit.
#[test]
fn the_five_always_moving_temporal_types_move_for_every_value() {
    let schema: SchemaRef = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, "ts", Type::Primitive(PrimitiveType::Timestamp)).into(),
                NestedField::optional(2, "tz", Type::Primitive(PrimitiveType::Timestamptz)).into(),
                NestedField::optional(3, "tm", Type::Primitive(PrimitiveType::Time)).into(),
                NestedField::optional(4, "tsn", Type::Primitive(PrimitiveType::TimestampNs)).into(),
                NestedField::optional(5, "tzn", Type::Primitive(PrimitiveType::TimestamptzNs))
                    .into(),
                NestedField::optional(6, "dt", Type::Primitive(PrimitiveType::Date)).into(),
            ])
            .build()
            .expect("the six-column temporal schema must build"),
    );
    let spec = PartitionSpec::builder(schema.clone())
        .add_partition_field("ts", "ts", Transform::Identity)
        .expect("identity(ts) is legal")
        .add_partition_field("tz", "tz", Transform::Identity)
        .expect("identity(tz) is legal")
        .add_partition_field("tm", "tm", Transform::Identity)
        .expect("identity(tm) is legal")
        .add_partition_field("tsn", "tsn", Transform::Identity)
        .expect("identity(tsn) is legal")
        .add_partition_field("tzn", "tzn", Transform::Identity)
        .expect("identity(tzn) is legal")
        .add_partition_field("dt", "dt", Transform::Identity)
        .expect("identity(dt) is legal")
        .build()
        .expect("the six-field temporal spec must build");
    // 2017-11-16T22:31:08 in micros (and the same instant in nanos); 22:31:08 in micros;
    // 2022-01-08 in days.
    let data = Struct::from_iter([
        Some(Literal::timestamp(1_510_871_468_000_000)),
        Some(Literal::timestamptz(1_510_871_468_000_000)),
        Some(Literal::time(81_068_000_000)),
        Some(Literal::timestamp_nano(1_510_871_468_000_000_000)),
        Some(Literal::timestamptz_nano(1_510_871_468_000_000_000)),
        Some(Literal::date(19_000)),
    ]);

    let path = spec.partition_to_path(&data, schema);
    let pairs: Vec<&str> = path.split('/').collect();
    assert_eq!(
        pairs,
        vec![
            "ts=2017-11-16+22%3A31%3A08",
            "tz=2017-11-16+22%3A31%3A08+UTC",
            "tm=22%3A31%3A08",
            "tsn=2017-11-16+22%3A31%3A08",
            "tzn=2017-11-16+22%3A31%3A08+UTC",
            "dt=2022-01-08",
        ],
        "the five temporal types whose human string holds a `:` move under the escaper; \
             `date` does not"
    );

    // `time` MATCHES Java post-R161: this change closes that divergence.
    assert_eq!(
        pairs[2], "tm=22%3A31%3A08",
        "Java: `22:31:08` escapes to `22%3A31%3A08`"
    );
    // `date` is byte-stable: its human string holds no character outside the safe set.
    assert_eq!(
        pairs[5], "dt=2022-01-08",
        "Java: `2022-01-08`, untouched by the escaper"
    );
    // The four remaining divergences, pinned as an alarm. Each expected form is Java's own,
    // measured on the JVM, so none is a dead comparison.
    assert_ne!(
        pairs[0], "ts=2017-11-16T22%3A31%3A08",
        "residue R161: Java renders ISO `T`, the fork renders a space (escaped `+`)"
    );
    assert_ne!(
        pairs[1], "tz=2017-11-16T22%3A31%3A08%2B00%3A00",
        "residue R161: Java renders ISO `T` and `+00:00`, the fork renders a space and ` UTC`"
    );
    assert_ne!(
        pairs[3], "tsn=2017-11-16T22%3A31%3A08",
        "residue R161: the nanosecond pair diverges exactly like the microsecond pair"
    );
    assert_ne!(
        pairs[4], "tzn=2017-11-16T22%3A31%3A08%2B00%3A00",
        "residue R161: the nanosecond pair diverges exactly like the microsecond pair"
    );
}

/// Render a one-field `transform(column)` spec over `column: ty` holding `value`.
fn render_one(
    column: &str,
    field_name: &str,
    ty: PrimitiveType,
    transform: Transform,
    value: Literal,
) -> String {
    let schema: SchemaRef = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, column, Type::Primitive(ty)).into(),
            ])
            .build()
            .expect("the one-column schema must build"),
    );
    let spec = PartitionSpec::builder(schema.clone())
        .add_partition_field(column, field_name, transform)
        .expect("the transform must be legal for this column type")
        .build()
        .expect("the one-field spec must build");
    spec.partition_to_path(&Struct::from_iter([Some(value)]), schema)
}

/// Byte-stability is a property of the OUTPUT type, not of the transform name.
/// `Transform::result_type` returns the input type for `Truncate`, so `truncate(string, N)`
/// renders a STRING and is as escaper-sensitive as `identity(string)`. Over int, long, decimal
/// and binary it stays in the safe set: binary and fixed render as Java's standard Base64
/// (`TransformUtil.base64encode`, no `Truncate` override), so a source byte such as `0x2F` never
/// reaches the path. Base64's `+`, `/` and `=` are then escaped like any other character.
#[test]
fn truncate_is_byte_stable_except_over_string() {
    let moving = [
        (
            render_one(
                "s",
                "s_trunc",
                PrimitiveType::String,
                Transform::Truncate(4),
                Literal::string("a/b c"),
            ),
            "s_trunc=a%2Fb+c",
        ),
        (
            render_one(
                "s",
                "t5",
                PrimitiveType::String,
                Transform::Truncate(5),
                Literal::string("east 1x"),
            ),
            "t5=east+1x",
        ),
    ];
    for (rendered, expected) in &moving {
        assert_eq!(
            rendered, expected,
            "truncate over `string` renders a string and MUST be escaped"
        );
    }

    // CLOSED 2026-07-31 (QC / R161): Java base64 for binary partition values.
    let truncated_binary = render_one(
        "bn",
        "tb",
        PrimitiveType::Binary,
        Transform::Truncate(2),
        Literal::binary(vec![0x61, 0x2F, 0x62]),
    );
    assert_eq!(
        truncated_binary, "tb=YS9i",
        "Java `partitionToPath` emits `tb=YS9i` for truncate(binary,2) over bytes 61 2F 62 \
             (TransformUtil.base64encode / java.util.Base64.getEncoder)"
    );

    let stable = [
        (
            render_one(
                "s",
                "tsafe",
                PrimitiveType::String,
                Transform::Truncate(16),
                Literal::string("us-east-1"),
            ),
            "tsafe=us-east-1",
        ),
        (
            render_one(
                "i",
                "ti",
                PrimitiveType::Int,
                Transform::Truncate(10),
                Literal::int(25),
            ),
            "ti=25",
        ),
        (
            render_one(
                "l",
                "tl",
                PrimitiveType::Long,
                Transform::Truncate(10),
                Literal::long(-25),
            ),
            "tl=-25",
        ),
        (
            render_one(
                "d",
                "td",
                PrimitiveType::Decimal {
                    precision: 9,
                    scale: 2,
                },
                Transform::Truncate(50),
                Literal::decimal(12345),
            ),
            "td=123.45",
        ),
        // Base64 `YS9i` is inside the URLEncoder safe set — no further escaping.
        (truncated_binary.clone(), "tb=YS9i"),
    ];
    for (rendered, expected) in &stable {
        assert_eq!(
            rendered, expected,
            "this truncate output holds no character outside the safe set after the human \
                 string is formed"
        );
    }
}

/// `identity(binary)` and `identity(fixed[N])` use the same Base64 human string as Java, not
/// UPPERCASE hex. `Display for Datum` still renders hex; only the partition-path seam is base64.
#[test]
fn identity_binary_and_fixed_render_java_base64() {
    let bytes = vec![0x61, 0x2F, 0x62]; // ASCII "a/b" → base64 "YS9i"
    assert_eq!(
        render_one(
            "bn",
            "b",
            PrimitiveType::Binary,
            Transform::Identity,
            Literal::binary(bytes.clone()),
        ),
        "b=YS9i",
        "identity(binary) must match Java TransformUtil.base64encode"
    );
    assert_eq!(
        render_one(
            "fx",
            "f",
            PrimitiveType::Fixed(3),
            Transform::Identity,
            Literal::fixed(bytes.clone()),
        ),
        "f=YS9i",
        "identity(fixed[3]) must match Java TransformUtil.base64encode"
    );

    // Standard Base64, NOT URL-safe: bytes that produce `+`, `/` or `=` are then URL-escaped.
    // 0xFB 0xFF -> base64 "+/8=" -> escaped "%2B%2F8%3D".
    assert_eq!(
        render_one(
            "bn",
            "b",
            PrimitiveType::Binary,
            Transform::Identity,
            Literal::binary(vec![0xFB, 0xFF]),
        ),
        "b=%2B%2F8%3D",
        "base64 alphabet chars outside the URLEncoder safe set must be escaped"
    );

    // Display stays hex (orthogonal surface).
    let datum = crate::spec::Datum::new(
        PrimitiveType::Binary,
        crate::spec::PrimitiveLiteral::Binary(vec![0x61, 0x2F, 0x62]),
    );
    assert_eq!(
        datum.to_string(),
        "612F62",
        "Display for Binary stays UPPERCASE hex; only to_human_string is base64"
    );
    assert_eq!(datum.to_human_string(), "YS9i");

    // Empty binary → empty human string (Java: `toHumanString(Binary, empty ByteBuffer)` → "").
    assert_eq!(
        render_one(
            "bn",
            "b",
            PrimitiveType::Binary,
            Transform::Identity,
            Literal::binary(Vec::<u8>::new()),
        ),
        "b=",
        "empty binary human string is empty (Java jar-oracle)"
    );

    // UUID is NOT base64 — Java uses UUID.toString(); fork uses Display for UInt128.
    assert_eq!(
        render_one(
            "u",
            "u",
            PrimitiveType::Uuid,
            Transform::Identity,
            Literal::uuid(uuid::Uuid::from_u128(1)),
        ),
        "u=00000000-0000-0000-0000-000000000001",
        "identity(uuid) must not be routed through base64"
    );
}

/// R161 restores INJECTIVITY of partition tuple to directory, the data-trust half of the defect.
///
/// Before R161 the pair was `format!("{name}={value}")` with both sides raw, so a `/` or an `=`
/// inside a VALUE made two DISTINCT tuples render the SAME path. Colliding paths put two
/// partitions' data files in one directory and merge their `partitions.<path>` summary entries,
/// so the per-partition record counts are silently summed.
#[test]
fn two_distinct_tuples_can_no_longer_collide_on_one_directory() {
    let schema: SchemaRef = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, "a", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(2, "b", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("the two-column schema must build"),
    );
    let two_field = PartitionSpec::builder(schema.clone())
        .add_partition_field("a", "a", Transform::Identity)
        .expect("identity(a) is legal")
        .add_partition_field("b", "b", Transform::Identity)
        .expect("identity(b) is legal")
        .build()
        .expect("the two-field spec must build");

    // Same spec, two distinct tuples — both rendered `a=1/b=2/b=3` before R161.
    let x = two_field.partition_to_path(
        &Struct::from_iter([Some(Literal::string("1/b=2")), Some(Literal::string("3"))]),
        schema.clone(),
    );
    let y = two_field.partition_to_path(
        &Struct::from_iter([Some(Literal::string("1")), Some(Literal::string("2/b=3"))]),
        schema.clone(),
    );
    // Injectivity FIRST, so a regression's failure message displays the collision itself
    // rather than a byte mismatch on one side of it.
    assert_ne!(
        x, y,
        "two distinct partition tuples of ONE spec must never share a directory"
    );
    assert_eq!(x, "a=1%2Fb%3D2/b=3");
    assert_eq!(y, "a=1/b=2%2Fb%3D3");

    // Cross-arity: a 1-field spec and a 2-field spec of the same evolving table — both
    // rendered `a=1/b=2` before R161.
    let one_field = PartitionSpec::builder(schema.clone())
        .add_partition_field("a", "a", Transform::Identity)
        .expect("identity(a) is legal")
        .build()
        .expect("the one-field spec must build");
    let narrow = one_field.partition_to_path(
        &Struct::from_iter([Some(Literal::string("1/b=2"))]),
        schema.clone(),
    );
    let wide = two_field.partition_to_path(
        &Struct::from_iter([Some(Literal::string("1")), Some(Literal::string("2"))]),
        schema,
    );
    assert_ne!(
        narrow, wide,
        "tuples under two specs of ONE table must never share a directory"
    );
    assert_eq!(narrow, "a=1%2Fb%3D2");
    assert_eq!(wide, "a=1/b=2");
}
