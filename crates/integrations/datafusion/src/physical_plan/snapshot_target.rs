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

use iceberg::table::Table;
use iceberg::{Error, ErrorKind, Result};

pub(crate) fn maybe_to_branch<A>(
    action: A,
    branch: Option<&str>,
    to_branch: impl FnOnce(A, &str) -> A,
) -> A {
    match branch {
        Some(name) => to_branch(action, name),
        None => action,
    }
}

pub(crate) fn maybe_validate_from_snapshot<A>(
    action: A,
    snapshot_id: Option<i64>,
    validate: impl FnOnce(A, i64) -> A,
) -> A {
    match snapshot_id {
        Some(id) => validate(action, id),
        None => action,
    }
}

pub(crate) fn resolve_scan_snapshot_id(
    table: &Table,
    commit_branch: Option<&str>,
) -> Result<Option<i64>> {
    match commit_branch {
        None => Ok(table.metadata().current_snapshot_id()),
        Some(name) => table
            .metadata()
            .snapshot_for_ref(name)
            .map(|snapshot| Some(snapshot.snapshot_id()))
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("snapshot ref '{name}' not found"),
                )
            }),
    }
}

pub(crate) fn optional_ref_snapshot_id(table: &Table, commit_branch: Option<&str>) -> Option<i64> {
    match commit_branch {
        None => table.metadata().current_snapshot_id(),
        Some(name) => table
            .metadata()
            .snapshot_for_ref(name)
            .map(|snapshot| snapshot.snapshot_id()),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn maybe_validate_from_snapshot_applies_the_scan_snapshot_when_set() {
        let out = super::maybe_validate_from_snapshot(0_i64, Some(7), |_, id| id);
        assert_eq!(
            out, 7,
            "validate_from_snapshot must arm with the scanned snapshot"
        );
    }

    #[test]
    fn maybe_validate_from_snapshot_skips_when_no_snapshot() {
        let out = super::maybe_validate_from_snapshot(0_i64, None, |_, id| id);
        assert_eq!(
            out, 0,
            "no scan snapshot means no validate_from_snapshot pin"
        );
    }
}
