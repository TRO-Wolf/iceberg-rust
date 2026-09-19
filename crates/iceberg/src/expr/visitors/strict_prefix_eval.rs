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

use std::cmp::Ordering;

use crate::expr::BoundReference;
use crate::spec::{DataFile, Datum, PrimitiveLiteral};
use crate::{Error, ErrorKind, Result};

fn may_contain_null(data_file: &DataFile, field_id: i32) -> bool {
    match data_file.null_value_counts.get(&field_id) {
        Some(&null_count) => null_count > 0,
        None => true,
    }
}

fn contains_nulls_only(data_file: &DataFile, field_id: i32) -> bool {
    let null_count = data_file.null_value_counts.get(&field_id);
    let value_count = data_file.value_counts.get(&field_id);
    null_count.is_some() && null_count == value_count
}

fn cmp_utf16_prefix(bound: &str, prefix: &str) -> Ordering {
    let limit = bound
        .encode_utf16()
        .count()
        .min(prefix.encode_utf16().count());
    let mut bound_chars = bound.chars();
    let mut prefix_chars = prefix.chars();
    let (mut bound_units, mut prefix_units) = (0usize, 0usize);
    loop {
        let bound_char = (bound_units < limit).then(|| bound_chars.next()).flatten();
        let prefix_char = (prefix_units < limit)
            .then(|| prefix_chars.next())
            .flatten();
        match (bound_char, prefix_char) {
            (Some(bound), Some(prefix)) => {
                bound_units += bound.len_utf16();
                prefix_units += prefix.len_utf16();
                match bound.cmp(&prefix) {
                    Ordering::Equal => {}
                    ordering => return ordering,
                }
            }
            (None, None) => return Ordering::Equal,
            (Some(_), None) => return Ordering::Greater,
            (None, Some(_)) => return Ordering::Less,
        }
    }
}

fn string_literal<'a>(datum: &'a Datum, operator: &str) -> Result<&'a str> {
    let PrimitiveLiteral::String(value) = datum.literal() else {
        return Err(Error::new(
            ErrorKind::Unexpected,
            format!("Cannot use {operator} operator on non-string values"),
        ));
    };
    Ok(value.as_str())
}

pub(crate) fn eval_starts_with(
    data_file: &DataFile,
    reference: &BoundReference,
    datum: &Datum,
) -> Result<bool> {
    let field_id = reference.field().id;

    if reference.accessor().is_nested() || may_contain_null(data_file, field_id) {
        return Ok(false);
    }

    let prefix = string_literal(datum, "StartsWith")?;

    let (Some(lower), Some(upper)) = (
        data_file.promoted_lower_bound(reference),
        data_file.promoted_upper_bound(reference),
    ) else {
        return Ok(false);
    };
    let lower = string_literal_of(lower.literal(), "StartsWith", "lower_bound")?;
    let upper = string_literal_of(upper.literal(), "StartsWith", "upper_bound")?;

    Ok(lower.starts_with(prefix) && upper.starts_with(prefix))
}

fn string_literal_of<'a>(
    literal: &'a PrimitiveLiteral,
    operator: &str,
    bound: &str,
) -> Result<&'a str> {
    let PrimitiveLiteral::String(value) = literal else {
        return Err(Error::new(
            ErrorKind::Unexpected,
            format!("Cannot use {operator} operator on non-string {bound} value"),
        ));
    };
    Ok(value.as_str())
}

pub(crate) fn eval_not_starts_with(
    data_file: &DataFile,
    reference: &BoundReference,
    datum: &Datum,
) -> Result<bool> {
    let field_id = reference.field().id;

    if reference.accessor().is_nested() {
        return Ok(false);
    }
    if contains_nulls_only(data_file, field_id) {
        return Ok(true);
    }

    let prefix = string_literal(datum, "NotStartsWith")?;

    if let Some(lower) = data_file.promoted_lower_bound(reference)
        && cmp_utf16_prefix(
            string_literal_of(lower.literal(), "NotStartsWith", "lower_bound")?,
            prefix,
        ) == Ordering::Greater
    {
        return Ok(true);
    }

    if let Some(upper) = data_file.promoted_upper_bound(reference)
        && cmp_utf16_prefix(
            string_literal_of(upper.literal(), "NotStartsWith", "upper_bound")?,
            prefix,
        ) == Ordering::Less
    {
        return Ok(true);
    }

    Ok(false)
}
