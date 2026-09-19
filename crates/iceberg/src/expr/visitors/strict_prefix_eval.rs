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

fn utf16(value: &str) -> Vec<u16> {
    value.encode_utf16().collect()
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

    let PrimitiveLiteral::String(prefix) = datum.literal() else {
        return Err(Error::new(
            ErrorKind::Unexpected,
            "Cannot use StartsWith operator on non-string values",
        ));
    };

    let (Some(lower), Some(upper)) = (
        data_file.promoted_lower_bound(reference),
        data_file.promoted_upper_bound(reference),
    ) else {
        return Ok(false);
    };
    let PrimitiveLiteral::String(lower) = lower.literal() else {
        return Err(Error::new(
            ErrorKind::Unexpected,
            "Cannot use StartsWith operator on non-string lower_bound value",
        ));
    };
    let PrimitiveLiteral::String(upper) = upper.literal() else {
        return Err(Error::new(
            ErrorKind::Unexpected,
            "Cannot use StartsWith operator on non-string upper_bound value",
        ));
    };

    let prefix_units = utf16(prefix);
    let lower_units = utf16(lower);
    if lower_units.len() < prefix_units.len()
        || lower_units[..prefix_units.len()] != prefix_units[..]
    {
        return Ok(false);
    }

    let upper_units = utf16(upper);
    if upper_units.len() < prefix_units.len()
        || upper_units[..prefix_units.len()] != prefix_units[..]
    {
        return Ok(false);
    }

    Ok(true)
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

    let PrimitiveLiteral::String(prefix) = datum.literal() else {
        return Err(Error::new(
            ErrorKind::Unexpected,
            "Cannot use NotStartsWith operator on non-string values",
        ));
    };
    let prefix_units = utf16(prefix);

    if let Some(lower) = data_file.promoted_lower_bound(reference) {
        let PrimitiveLiteral::String(lower) = lower.literal() else {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "Cannot use NotStartsWith operator on non-string lower_bound value",
            ));
        };
        let lower_units = utf16(lower);
        let trunc = prefix_units.len().min(lower_units.len());
        if lower_units[..trunc] > prefix_units[..] {
            return Ok(true);
        }
    }

    if let Some(upper) = data_file.promoted_upper_bound(reference) {
        let PrimitiveLiteral::String(upper) = upper.literal() else {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "Cannot use NotStartsWith operator on non-string upper_bound value",
            ));
        };
        let upper_units = utf16(upper);
        let trunc = prefix_units.len().min(upper_units.len());
        if upper_units[..trunc] < prefix_units[..] {
            return Ok(true);
        }
    }

    Ok(false)
}
