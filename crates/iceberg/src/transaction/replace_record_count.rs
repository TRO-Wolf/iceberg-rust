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

use std::collections::HashMap;

use crate::error::{Error, ErrorKind, Result};
use crate::spec::{Operation, Summary};

const ADDED_RECORDS: &str = "added-records";
const DELETED_RECORDS: &str = "deleted-records";

pub(crate) fn validate_replace_record_counts(summary: &Summary) -> Result<()> {
    if summary.operation != Operation::Replace {
        return Ok(());
    }
    let added = property_as_long(&summary.additional_properties, ADDED_RECORDS)?;
    let deleted = property_as_long(&summary.additional_properties, DELETED_RECORDS)?;
    if added > deleted {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Invalid REPLACE operation: {added} added records > {deleted} replaced records"
            ),
        ));
    }
    Ok(())
}

fn property_as_long(properties: &HashMap<String, String>, key: &str) -> Result<i64> {
    match properties.get(key) {
        None => Ok(0),
        Some(value) => value.parse::<i64>().map_err(|err| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Invalid snapshot summary property {key}: {value}"),
            )
            .with_source(err)
        }),
    }
}

#[cfg(test)]
#[path = "replace_record_count_tests.rs"]
mod tests;
