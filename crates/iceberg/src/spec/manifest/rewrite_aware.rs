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
use std::sync::{LazyLock, Mutex};

use super::{DataFile, ManifestEntry};
use crate::spec::PrimitiveLiteral;

static UNASSIGNED_ROW_COUNTS: LazyLock<Mutex<HashMap<String, u64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub(crate) fn note_unassigned_row_count(path: String, n: u64) {
    if let Ok(mut map) = UNASSIGNED_ROW_COUNTS.lock() {
        map.insert(path, n);
    }
}

pub(crate) fn take_unassigned_row_count(path: &str) -> Option<u64> {
    UNASSIGNED_ROW_COUNTS
        .lock()
        .ok()
        .and_then(|mut map| map.remove(path))
}

/// Result of rewrite-aware first-row-id recovery after `FirstRowIdPolicy::Suppress`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RewriteAwareFirstRowIds {
    pub manifest_first_row_id: Option<u64>,
    pub unassigned_row_count: Option<u64>,
}

fn stored_row_id_bounds(file: &DataFile) -> Option<(i64, i64)> {
    if file.first_row_id.is_some() {
        let first = file.first_row_id?;
        let last = first.checked_add(i64::try_from(file.record_count.saturating_sub(1)).ok()?)?;
        return Some((first, last));
    }
    let field_id = crate::metadata_columns::RESERVED_FIELD_ID_ROW_ID;
    let values = file.value_counts.get(&field_id).copied()?;
    if values != file.record_count {
        return None;
    }
    let nulls = file.null_value_counts.get(&field_id).copied().unwrap_or(0);
    if nulls != 0 {
        return None;
    }
    let lower = match file.lower_bounds.get(&field_id)?.literal() {
        PrimitiveLiteral::Long(value) => *value,
        _ => return None,
    };
    let upper = match file.upper_bounds.get(&field_id).map(|d| d.literal()) {
        Some(PrimitiveLiteral::Long(value)) => *value,
        _ if file.record_count <= 1 => lower,
        _ => return None,
    };
    Some((lower, upper))
}

pub(crate) fn apply_rewrite_aware_first_row_ids(
    manifest_first_row_id: Option<u64>,
    entries: &mut [ManifestEntry],
) -> RewriteAwareFirstRowIds {
    if manifest_first_row_id.is_some() {
        return RewriteAwareFirstRowIds {
            manifest_first_row_id,
            unassigned_row_count: None,
        };
    }
    let mut min_id: Option<u64> = None;
    let mut max_id: Option<u64> = None;
    let mut assigned_rows: u64 = 0;
    let mut unassigned_rows: u64 = 0;
    let mut stamped: u64 = 0;
    let mut live: u64 = 0;
    for entry in entries.iter_mut() {
        if !entry.is_alive() {
            continue;
        }
        live += 1;
        let Some((lower, upper)) = stored_row_id_bounds(&entry.data_file) else {
            unassigned_rows = unassigned_rows.saturating_add(entry.data_file.record_count);
            continue;
        };
        if entry.data_file.first_row_id.is_none() {
            entry.data_file.first_row_id = Some(lower);
        }
        stamped += 1;
        assigned_rows = assigned_rows.saturating_add(entry.data_file.record_count);
        if let Ok(lo) = u64::try_from(lower) {
            min_id = Some(min_id.map_or(lo, |current| current.min(lo)));
        }
        if let Ok(hi) = u64::try_from(upper) {
            max_id = Some(max_id.map_or(hi, |current| current.max(hi)));
        }
    }
    let contiguous = match (min_id, max_id) {
        (Some(lo), Some(hi)) => hi.saturating_sub(lo).saturating_add(1) == assigned_rows,
        _ => false,
    };
    let all_stored = live > 0 && stamped == live && unassigned_rows == 0;
    let mixed = stamped > 0 && unassigned_rows > 0;
    RewriteAwareFirstRowIds {
        manifest_first_row_id: if all_stored && contiguous {
            min_id
        } else {
            None
        },
        unassigned_row_count: mixed.then_some(unassigned_rows),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::manifest::{DataContentType, DataFileBuilder, DataFileFormat, ManifestStatus};
    use crate::spec::{Datum, Struct};

    fn entry_with_stored_row_ids(
        path: &str,
        record_count: u64,
        min_row_id: i64,
        max_row_id: i64,
        nulls: u64,
    ) -> ManifestEntry {
        use std::collections::HashMap;

        use crate::metadata_columns::RESERVED_FIELD_ID_ROW_ID;

        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .partition(Struct::empty())
            .record_count(record_count)
            .file_size_in_bytes(100)
            .value_counts(HashMap::from([(RESERVED_FIELD_ID_ROW_ID, record_count)]))
            .null_value_counts(HashMap::from([(RESERVED_FIELD_ID_ROW_ID, nulls)]))
            .lower_bounds(HashMap::from([(
                RESERVED_FIELD_ID_ROW_ID,
                Datum::long(min_row_id),
            )]))
            .upper_bounds(HashMap::from([(
                RESERVED_FIELD_ID_ROW_ID,
                Datum::long(max_row_id),
            )]));
        ManifestEntry {
            status: ManifestStatus::Added,
            snapshot_id: None,
            sequence_number: None,
            file_sequence_number: None,
            data_file: builder.build().expect("build data file"),
        }
    }

    fn entry_without_row_ids(path: &str, record_count: u64) -> ManifestEntry {
        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .partition(Struct::empty())
            .record_count(record_count)
            .file_size_in_bytes(100);
        ManifestEntry {
            status: ManifestStatus::Added,
            snapshot_id: None,
            sequence_number: None,
            file_sequence_number: None,
            data_file: builder.build().expect("build data file"),
        }
    }

    #[test]
    fn contiguous_stored_row_ids_claim_the_manifest_range() {
        let mut entries = vec![
            entry_with_stored_row_ids("a.parquet", 2, 4, 5, 0),
            entry_with_stored_row_ids("b.parquet", 1, 1, 1, 0),
        ];
        let assigned = apply_rewrite_aware_first_row_ids(None, &mut entries);
        assert_eq!(assigned.manifest_first_row_id, None);
        assert_eq!(assigned.unassigned_row_count, None);
        assert_eq!(entries[0].data_file.first_row_id, Some(4));
        assert_eq!(entries[1].data_file.first_row_id, Some(1));

        let mut contiguous = vec![
            entry_with_stored_row_ids("a.parquet", 2, 0, 1, 0),
            entry_with_stored_row_ids("b.parquet", 1, 2, 2, 0),
        ];
        let assigned = apply_rewrite_aware_first_row_ids(None, &mut contiguous);
        assert_eq!(assigned.manifest_first_row_id, Some(0));
        assert_eq!(assigned.unassigned_row_count, None);
        assert_eq!(contiguous[0].data_file.first_row_id, Some(0));
        assert_eq!(contiguous[1].data_file.first_row_id, Some(2));
    }

    #[test]
    fn non_contiguous_stored_ids_do_not_claim_the_manifest_range() {
        let mut entries = vec![entry_with_stored_row_ids("a.parquet", 2, 0, 2, 0)];
        let assigned = apply_rewrite_aware_first_row_ids(None, &mut entries);
        assert_eq!(assigned.manifest_first_row_id, None);
        assert_eq!(assigned.unassigned_row_count, None);
        assert_eq!(entries[0].data_file.first_row_id, Some(0));
    }

    #[test]
    fn mixed_manifest_stamps_stored_files_and_counts_only_new_rows() {
        let mut entries = vec![
            entry_without_row_ids("new.parquet", 3),
            entry_with_stored_row_ids("rewritten.parquet", 2, 0, 2, 0),
        ];
        let assigned = apply_rewrite_aware_first_row_ids(None, &mut entries);
        assert_eq!(assigned.manifest_first_row_id, None);
        assert_eq!(assigned.unassigned_row_count, Some(3));
        assert!(entries[0].data_file.first_row_id.is_none());
        assert_eq!(entries[1].data_file.first_row_id, Some(0));
    }

    #[test]
    fn a_null_stored_row_id_does_not_claim_the_manifest_range() {
        let mut entries = vec![entry_with_stored_row_ids("a.parquet", 2, 4, 5, 1)];
        let assigned = apply_rewrite_aware_first_row_ids(None, &mut entries);
        assert_eq!(assigned.manifest_first_row_id, None);
        assert!(entries[0].data_file.first_row_id.is_none());
    }

    #[test]
    fn an_already_assigned_manifest_range_is_kept() {
        let mut entries = vec![entry_with_stored_row_ids("a.parquet", 2, 4, 5, 0)];
        assert_eq!(
            apply_rewrite_aware_first_row_ids(Some(90), &mut entries).manifest_first_row_id,
            Some(90)
        );
    }
}
