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

//! Java `Float.toString` / `Double.toString` text. Used by [`Datum::to_human_string`]
//! (row R161) and by expression JSON (row R149). `Display for Datum` stays Rust.

use std::fmt::Write;

/// Java `Float.toString`.
pub(crate) fn java_to_string_f32(v: f32) -> String {
    java_to_string_float(f64::from(v), true)
}

/// Java `Double.toString`.
pub(crate) fn java_to_string_f64(v: f64) -> String {
    java_to_string_float(v, false)
}

/// Finite algorithm matches `expression_parser::format_java_float`. Non-finite
/// forms are Java `toString`, not the JSON path that rejects them.
pub(crate) fn java_to_string_float(v: f64, is_float: bool) -> String {
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        return if v.is_sign_negative() {
            "-Infinity".to_string()
        } else {
            "Infinity".to_string()
        };
    }
    if v == 0.0 {
        return if v.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }

    let neg = v < 0.0;
    let sci = if is_float {
        format!("{:e}", (v as f32).abs())
    } else {
        format!("{:e}", v.abs())
    };
    let (mantissa, exp_str) = sci.split_once('e').expect("Rust {:e} always contains e");
    let exp: i32 = exp_str
        .parse()
        .expect("Rust {:e} exponent is a signed integer");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let ndigits = digits.len();

    let mut out = String::new();
    if neg {
        out.push('-');
    }
    // Java FloatingDecimal: scientific when the leading digit's power is >= 7 or <= -4.
    if exp >= 7 || exp <= -4 {
        out.push_str(&digits[..1]);
        out.push('.');
        if ndigits == 1 {
            out.push('0');
        } else {
            out.push_str(&digits[1..]);
        }
        out.push('E');
        let _ = write!(out, "{exp}");
    } else if exp >= 0 {
        let int_len = (exp + 1) as usize;
        if ndigits <= int_len {
            out.push_str(&digits);
            for _ in 0..(int_len - ndigits) {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            out.push_str(&digits[..int_len]);
            out.push('.');
            out.push_str(&digits[int_len..]);
        }
    } else {
        out.push_str("0.");
        for _ in 0..(-exp - 1) {
            out.push('0');
        }
        out.push_str(&digits);
    }
    out
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::spec::{
        Datum, Literal, NestedField, PartitionSpec, PrimitiveType, Schema, Transform, Type,
    };

    /// R161 jar-oracle forms. `1.5` and `NaN` already agreed with Rust Display.
    const CASES: &[(&str, f32, f64, &str)] = &[
        ("1.0", 1.0, 1.0, "1.0"),
        ("1.0E10", 1.0e10, 1.0e10, "1.0E10"),
        ("-0.0", -0.0, -0.0, "-0.0"),
        ("Infinity", f32::INFINITY, f64::INFINITY, "Infinity"),
        (
            "-Infinity",
            f32::NEG_INFINITY,
            f64::NEG_INFINITY,
            "-Infinity",
        ),
        ("NaN", f32::NAN, f64::NAN, "NaN"),
        ("1.5", 1.5, 1.5, "1.5"),
    ];

    #[test]
    fn java_to_string_matches_r161_forms() {
        for (name, f, d, expected) in CASES {
            assert_eq!(java_to_string_f32(*f), *expected, "f32 {name}");
            assert_eq!(java_to_string_f64(*d), *expected, "f64 {name}");
        }
    }

    #[test]
    fn datum_to_human_string_uses_java_form_not_display() {
        let one = Datum::float(1.0f32);
        assert_eq!(one.to_string(), "1");
        assert_eq!(one.to_human_string(), "1.0");

        let inf = Datum::float(f32::INFINITY);
        assert_eq!(inf.to_string(), "inf");
        assert_eq!(inf.to_human_string(), "Infinity");

        let d1 = Datum::double(1.0f64);
        assert_eq!(d1.to_string(), "1");
        assert_eq!(d1.to_human_string(), "1.0");

        let dinf = Datum::double(f64::INFINITY);
        assert_eq!(dinf.to_string(), "inf");
        assert_eq!(dinf.to_human_string(), "Infinity");
    }

    fn identity_path(ty: PrimitiveType, literal: Literal) -> String {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "f", Type::Primitive(ty.clone())).into(),
            ])
            .build()
            .expect("schema");
        let schema = Arc::new(schema);
        let spec = PartitionSpec::builder(schema.clone())
            .add_partition_field("f", "f", Transform::Identity)
            .expect("identity field")
            .build()
            .expect("spec");
        let data = crate::spec::Struct::from_iter([Some(literal)]);
        spec.partition_to_path(&data, schema)
    }

    #[test]
    fn identity_partition_path_uses_java_human_string() {
        for (name, f, d, expected) in CASES {
            let fty = Type::Primitive(PrimitiveType::Float);
            let dty = Type::Primitive(PrimitiveType::Double);
            assert_eq!(
                Transform::Identity.to_human_string(&fty, Some(&Literal::float(*f))),
                *expected,
                "transform f32 {name}"
            );
            assert_eq!(
                Transform::Identity.to_human_string(&dty, Some(&Literal::double(*d))),
                *expected,
                "transform f64 {name}"
            );
            assert_eq!(
                identity_path(PrimitiveType::Float, Literal::float(*f)),
                format!("f={expected}"),
                "path f32 {name}"
            );
            assert_eq!(
                identity_path(PrimitiveType::Double, Literal::double(*d)),
                format!("f={expected}"),
                "path f64 {name}"
            );
        }
    }
}
