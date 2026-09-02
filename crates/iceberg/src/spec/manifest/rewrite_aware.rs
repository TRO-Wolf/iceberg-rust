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

use super::{DataFile, ManifestEntry};
use crate::spec::PrimitiveLiteral;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RewriteAwareFirstRowIds {
    pub manifest_first_row_id: Option<u64>,
    pub unassigned_row_count: Option<u64>,
}

pub(crate) fn data_file_has_complete_stored_row_ids(file: &DataFile) -> bool {
    stored_row_id_bounds(file).is_some()
}

pub(crate) fn stored_row_id_bounds(file: &DataFile) -> Option<(i64, i64)> {
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
    source_has_stored_row_ids: Option<bool>,
) -> RewriteAwareFirstRowIds {
    if manifest_first_row_id.is_some() {
        return RewriteAwareFirstRowIds {
            manifest_first_row_id,
            unassigned_row_count: None,
        };
    }
    let mut min_id: Option<u64> = None;
    let mut max_id: Option<u64> = None;
    let mut stored_rows: u64 = 0;
    let mut unassigned_rows: u64 = 0;
    let mut stamped: u64 = 0;
    for entry in entries.iter_mut() {
        if !entry.is_alive() {
            continue;
        }
        let Some((lower, upper)) = stored_row_id_bounds(&entry.data_file) else {
            unassigned_rows = unassigned_rows.saturating_add(entry.data_file.record_count);
            continue;
        };
        if entry.data_file.first_row_id.is_none() {
            entry.data_file.first_row_id = Some(lower);
        }
        stamped += 1;
        stored_rows = stored_rows.saturating_add(entry.data_file.record_count);
        if let Ok(lo) = u64::try_from(lower) {
            min_id = Some(min_id.map_or(lo, |current| current.min(lo)));
        }
        if let Ok(hi) = u64::try_from(upper) {
            max_id = Some(max_id.map_or(hi, |current| current.max(hi)));
        }
    }
    let holes = match (min_id, max_id) {
        (Some(lo), Some(hi)) => hi
            .saturating_sub(lo)
            .saturating_add(1)
            .saturating_sub(stored_rows),
        _ => 0,
    };
    let mixed = stamped > 0 && unassigned_rows > 0;
    let all_stored = stamped > 0 && unassigned_rows == 0;
    let stored_source = source_has_stored_row_ids == Some(true);
    let unassigned_row_count = if mixed {
        Some(unassigned_rows.saturating_add(if stored_source { holes } else { 0 }))
    } else if all_stored && stored_source {
        Some(holes)
    } else if all_stored && source_has_stored_row_ids.is_none() {
        Some(0)
    } else {
        None
    };
    RewriteAwareFirstRowIds {
        manifest_first_row_id: None,
        unassigned_row_count,
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
    fn first_materialization_does_not_override_java_increment() {
        let mut entries = vec![entry_with_stored_row_ids("a.parquet", 2, 0, 2, 0)];
        let assigned = apply_rewrite_aware_first_row_ids(None, &mut entries, Some(false));
        assert_eq!(assigned.manifest_first_row_id, None);
        assert_eq!(assigned.unassigned_row_count, None);
        assert_eq!(entries[0].data_file.first_row_id, Some(0));
    }

    #[test]
    fn stored_source_contiguous_survivors_consume_zero() {
        let mut entries = vec![entry_with_stored_row_ids("a.parquet", 2, 1, 2, 0)];
        let assigned = apply_rewrite_aware_first_row_ids(None, &mut entries, Some(true));
        assert_eq!(assigned.unassigned_row_count, Some(0));
    }

    #[test]
    fn stored_source_gapped_survivors_consume_holes() {
        let mut entries = vec![entry_with_stored_row_ids("a.parquet", 2, 0, 2, 0)];
        let assigned = apply_rewrite_aware_first_row_ids(None, &mut entries, Some(true));
        assert_eq!(assigned.unassigned_row_count, Some(1));
        assert_eq!(entries[0].data_file.first_row_id, Some(0));
    }

    #[test]
    fn no_removed_files_all_stored_consumes_zero() {
        let mut entries = vec![entry_with_stored_row_ids("a.parquet", 1, 1, 1, 0)];
        let assigned = apply_rewrite_aware_first_row_ids(None, &mut entries, None);
        assert_eq!(assigned.unassigned_row_count, Some(0));
        assert_eq!(entries[0].data_file.first_row_id, Some(1));
    }

    #[test]
    fn mixed_manifest_counts_only_new_rows() {
        let mut entries = vec![
            entry_without_row_ids("new.parquet", 3),
            entry_with_stored_row_ids("rewritten.parquet", 2, 0, 2, 0),
        ];
        let assigned = apply_rewrite_aware_first_row_ids(None, &mut entries, None);
        assert_eq!(assigned.manifest_first_row_id, None);
        assert_eq!(assigned.unassigned_row_count, Some(3));
        assert!(entries[0].data_file.first_row_id.is_none());
        assert_eq!(entries[1].data_file.first_row_id, Some(0));
    }

    #[test]
    fn a_null_stored_row_id_is_unassigned() {
        let mut entries = vec![entry_with_stored_row_ids("a.parquet", 2, 4, 5, 1)];
        let assigned = apply_rewrite_aware_first_row_ids(None, &mut entries, None);
        assert_eq!(assigned.unassigned_row_count, None);
        assert!(entries[0].data_file.first_row_id.is_none());
    }

    #[test]
    fn an_already_assigned_manifest_range_is_kept() {
        let mut entries = vec![entry_with_stored_row_ids("a.parquet", 2, 4, 5, 0)];
        assert_eq!(
            apply_rewrite_aware_first_row_ids(Some(90), &mut entries, None).manifest_first_row_id,
            Some(90)
        );
    }

    #[tokio::test]
    async fn mixed_manifest_list_writer_advances_only_by_new_rows() {
        use tempfile::TempDir;

        use crate::io::FileIO;
        use crate::spec::manifest::ManifestWriterBuilder;
        use crate::spec::{ManifestListWriter, NestedField, PrimitiveType, Schema, Type};

        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let schema = std::sync::Arc::new(
            Schema::builder()
                .with_fields(vec![std::sync::Arc::new(NestedField::optional(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Long),
                ))])
                .build()
                .expect("schema"),
        );
        let partition_spec = crate::spec::PartitionSpec::builder(schema.clone())
            .with_spec_id(0)
            .build()
            .expect("spec");
        let manifest_path = temp_dir.path().join("mixed.avro");
        let output = file_io
            .new_output(manifest_path.to_str().expect("utf8"))
            .expect("output");
        let mut writer = ManifestWriterBuilder::new(output, Some(1), None, schema, partition_spec)
            .build_v3_data();
        writer
            .add_entry(entry_without_row_ids("new.parquet", 3))
            .expect("add new");
        writer
            .add_entry(entry_with_stored_row_ids("rewritten.parquet", 2, 0, 2, 0))
            .expect("add stored");
        let manifest = writer.write_manifest_file().await.expect("write");
        assert_eq!(manifest.unassigned_row_count, Some(3));

        let list_path = temp_dir.path().join("list.avro");
        let list_out = file_io
            .new_output(list_path.to_str().expect("utf8"))
            .expect("list out");
        let mut list_writer = ManifestListWriter::v3(list_out, 1, Some(0), 1, Some(3));
        list_writer
            .add_manifests(std::iter::once(manifest))
            .expect("add manifests");
        assert_eq!(list_writer.next_row_id(), Some(6));
        list_writer.close().await.expect("close list");

        let bytes = std::fs::read(&list_path).expect("read list");
        let list =
            crate::spec::ManifestList::parse_with_version(&bytes, crate::spec::FormatVersion::V3)
                .expect("parse list");
        assert_eq!(list.entries()[0].first_row_id, Some(3));
        let loaded = list.entries()[0]
            .load_manifest(&file_io)
            .await
            .expect("load_manifest");
        let ids: Vec<Option<i64>> = loaded
            .entries()
            .iter()
            .map(|entry| entry.data_file.first_row_id)
            .collect();
        assert_eq!(ids, vec![Some(3), Some(0)]);
    }
}
