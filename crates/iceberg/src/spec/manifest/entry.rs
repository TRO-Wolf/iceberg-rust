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

use std::sync::Arc;

use apache_avro::Schema as AvroSchema;
use once_cell::sync::Lazy;
use typed_builder::TypedBuilder;

use crate::avro::schema_to_avro_schema;
use crate::error::Result;
use crate::spec::{
    DataContentType, DataFile, INITIAL_SEQUENCE_NUMBER, ListType, Literal, ManifestContentType,
    ManifestFile, MapType, NestedField, NestedFieldRef, PrimitiveLiteral, PrimitiveType, Schema,
    StructType, Type,
};
use crate::{Error, ErrorKind};

/// Reference to [`ManifestEntry`].
pub type ManifestEntryRef = Arc<ManifestEntry>;

/// A manifest is an immutable Avro file that lists data files or delete
/// files, along with each file’s partition data tuple, metrics, and tracking
/// information.
#[derive(Debug, PartialEq, Eq, Clone, TypedBuilder)]
pub struct ManifestEntry {
    /// field: 0
    ///
    /// Used to track additions and deletions.
    pub status: ManifestStatus,
    /// field id: 1
    ///
    /// Snapshot id where the file was added, or deleted if status is 2.
    /// Inherited when null.
    #[builder(default, setter(strip_option(fallback = snapshot_id_opt)))]
    pub snapshot_id: Option<i64>,
    /// field id: 3
    ///
    /// Data sequence number of the file.
    /// Inherited when null and status is 1 (added).
    #[builder(default, setter(strip_option(fallback = sequence_number_opt)))]
    pub sequence_number: Option<i64>,
    /// field id: 4
    ///
    /// File sequence number indicating when the file was added.
    /// Inherited when null and status is 1 (added).
    #[builder(default, setter(strip_option(fallback = file_sequence_number_opt)))]
    pub file_sequence_number: Option<i64>,
    /// field id: 2
    ///
    /// File path, partition tuple, metrics, …
    pub data_file: DataFile,
}

impl ManifestEntry {
    /// Check if this manifest entry is deleted.
    pub fn is_alive(&self) -> bool {
        matches!(
            self.status,
            ManifestStatus::Added | ManifestStatus::Existing
        )
    }

    /// Status of this manifest entry
    pub fn status(&self) -> ManifestStatus {
        self.status
    }

    /// Content type of this manifest entry.
    #[inline]
    pub fn content_type(&self) -> DataContentType {
        self.data_file.content
    }

    /// File format of this manifest entry.
    #[inline]
    pub fn file_format(&self) -> DataFileFormat {
        self.data_file.file_format
    }

    /// Data file path of this manifest entry.
    #[inline]
    pub fn file_path(&self) -> &str {
        &self.data_file.file_path
    }

    /// Data file record count of the manifest entry.
    #[inline]
    pub fn record_count(&self) -> u64 {
        self.data_file.record_count
    }

    /// Inherit data from manifest list, such as snapshot id, sequence number.
    pub(crate) fn inherit_data(&mut self, snapshot_entry: &ManifestFile) {
        if self.snapshot_id.is_none() {
            self.snapshot_id = Some(snapshot_entry.added_snapshot_id);
        }

        if self.sequence_number.is_none()
            && (self.status == ManifestStatus::Added
                || snapshot_entry.sequence_number == INITIAL_SEQUENCE_NUMBER)
        {
            self.sequence_number = Some(snapshot_entry.sequence_number);
        }

        if self.file_sequence_number.is_none()
            && (self.status == ManifestStatus::Added
                || snapshot_entry.sequence_number == INITIAL_SEQUENCE_NUMBER)
        {
            self.file_sequence_number = Some(snapshot_entry.sequence_number);
        }
    }

    /// Snapshot id
    #[inline]
    pub fn snapshot_id(&self) -> Option<i64> {
        self.snapshot_id
    }

    /// Data sequence number.
    #[inline]
    pub fn sequence_number(&self) -> Option<i64> {
        self.sequence_number
    }

    /// File size in bytes.
    #[inline]
    pub fn file_size_in_bytes(&self) -> u64 {
        self.data_file.file_size_in_bytes
    }

    /// get a reference to the actual data file
    #[inline]
    pub fn data_file(&self) -> &DataFile {
        &self.data_file
    }
}

/// Assign each entry's `first_row_id` from the manifest's own range, mirroring Java
/// `ManifestReader.idAssigner`. Stateful, so it takes the whole slice in manifest order.
///
/// An absent `manifest_first_row_id` forces every file to `None`. A present one is a running
/// counter that skips `Deleted` and already-assigned entries without advancing.
///
/// # Errors
///
/// [`ErrorKind::DataInvalid`] on counter overflow. Java wraps; a wrapped id aliases live rows.
pub(crate) fn assign_first_row_ids(
    entries: &mut [ManifestEntry],
    manifest_first_row_id: Option<u64>,
) -> Result<()> {
    let Some(manifest_first_row_id) = manifest_first_row_id else {
        for entry in entries.iter_mut() {
            entry.data_file.first_row_id = None;
        }
        return Ok(());
    };

    let mut next_row_id = manifest_first_row_id;
    for entry in entries.iter_mut() {
        if entry.status == ManifestStatus::Deleted || entry.data_file.first_row_id.is_some() {
            continue;
        }
        // The counter is `u64`; `DataFile::first_row_id` is `i64` (Java's `Long`). Fail closed on
        // the narrowing rather than wrap into a negative row id.
        entry.data_file.first_row_id = Some(
            i64::try_from(next_row_id)
                .map_err(|_| overflow_error(manifest_first_row_id, entry, "exceeds i64::MAX"))?,
        );
        next_row_id = next_row_id
            .checked_add(entry.data_file.record_count)
            .ok_or_else(|| overflow_error(manifest_first_row_id, entry, "overflowed u64"))?;
    }
    Ok(())
}

pub(crate) fn apply_manifest_list_context(
    entries: &mut [ManifestEntry],
    manifest_file: &ManifestFile,
) -> Result<()> {
    // Let entries inherit values from the manifest list entry.
    for entry in entries.iter_mut() {
        entry.inherit_data(manifest_file);
    }

    // V3 row lineage: assign `first_row_id` from this manifest's range (Java
    // `ManifestReader.idAssigner`). The ids are a running total, so entries must arrive in
    // manifest order. A delete manifest never lends row ids and Java ignores a range on one, so
    // `None` here matches; erroring would reject a table Java scans fine.
    let manifest_range = match manifest_file.content {
        ManifestContentType::Deletes => None,
        ManifestContentType::Data => manifest_file.first_row_id,
    };
    assign_first_row_ids(entries, manifest_range)
}

/// Used to track additions and deletions in ManifestEntry.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ManifestStatus {
    /// Value: 0
    Existing = 0,
    /// Value: 1
    Added = 1,
    /// Value: 2
    ///
    /// Deletes are informational only and not used in scans.
    Deleted = 2,
}

impl TryFrom<i32> for ManifestStatus {
    type Error = Error;

    fn try_from(v: i32) -> Result<ManifestStatus> {
        match v {
            0 => Ok(ManifestStatus::Existing),
            1 => Ok(ManifestStatus::Added),
            2 => Ok(ManifestStatus::Deleted),
            _ => Err(Error::new(
                ErrorKind::DataInvalid,
                format!("manifest status {v} is invalid"),
            )),
        }
    }
}

use super::DataFileFormat;

static STATUS: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::required(
            0,
            "status",
            Type::Primitive(PrimitiveType::Int),
        ))
    })
};

static SNAPSHOT_ID_V1: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::required(
            1,
            "snapshot_id",
            Type::Primitive(PrimitiveType::Long),
        ))
    })
};

static SNAPSHOT_ID_V2: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::optional(
            1,
            "snapshot_id",
            Type::Primitive(PrimitiveType::Long),
        ))
    })
};

static SEQUENCE_NUMBER: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::optional(
            3,
            "sequence_number",
            Type::Primitive(PrimitiveType::Long),
        ))
    })
};

static FILE_SEQUENCE_NUMBER: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::optional(
            4,
            "file_sequence_number",
            Type::Primitive(PrimitiveType::Long),
        ))
    })
};

static CONTENT: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(
            NestedField::required(134, "content", Type::Primitive(PrimitiveType::Int))
                // 0 refers to DataContentType::DATA
                .with_initial_default(Literal::Primitive(PrimitiveLiteral::Int(0))),
        )
    })
};

static FILE_PATH: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::required(
            100,
            "file_path",
            Type::Primitive(PrimitiveType::String),
        ))
    })
};

static FILE_FORMAT: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::required(
            101,
            "file_format",
            Type::Primitive(PrimitiveType::String),
        ))
    })
};

static RECORD_COUNT: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::required(
            103,
            "record_count",
            Type::Primitive(PrimitiveType::Long),
        ))
    })
};

static FILE_SIZE_IN_BYTES: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::required(
            104,
            "file_size_in_bytes",
            Type::Primitive(PrimitiveType::Long),
        ))
    })
};

// Deprecated. Always write a default in v1. Do not write in v2.
static BLOCK_SIZE_IN_BYTES: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::required(
            105,
            "block_size_in_bytes",
            Type::Primitive(PrimitiveType::Long),
        ))
    })
};

static COLUMN_SIZES: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::optional(
            108,
            "column_sizes",
            Type::Map(MapType {
                key_field: Arc::new(NestedField::required(
                    117,
                    "key",
                    Type::Primitive(PrimitiveType::Int),
                )),
                value_field: Arc::new(NestedField::required(
                    118,
                    "value",
                    Type::Primitive(PrimitiveType::Long),
                )),
            }),
        ))
    })
};

static VALUE_COUNTS: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::optional(
            109,
            "value_counts",
            Type::Map(MapType {
                key_field: Arc::new(NestedField::required(
                    119,
                    "key",
                    Type::Primitive(PrimitiveType::Int),
                )),
                value_field: Arc::new(NestedField::required(
                    120,
                    "value",
                    Type::Primitive(PrimitiveType::Long),
                )),
            }),
        ))
    })
};

static NULL_VALUE_COUNTS: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::optional(
            110,
            "null_value_counts",
            Type::Map(MapType {
                key_field: Arc::new(NestedField::required(
                    121,
                    "key",
                    Type::Primitive(PrimitiveType::Int),
                )),
                value_field: Arc::new(NestedField::required(
                    122,
                    "value",
                    Type::Primitive(PrimitiveType::Long),
                )),
            }),
        ))
    })
};

static NAN_VALUE_COUNTS: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::optional(
            137,
            "nan_value_counts",
            Type::Map(MapType {
                key_field: Arc::new(NestedField::required(
                    138,
                    "key",
                    Type::Primitive(PrimitiveType::Int),
                )),
                value_field: Arc::new(NestedField::required(
                    139,
                    "value",
                    Type::Primitive(PrimitiveType::Long),
                )),
            }),
        ))
    })
};

static LOWER_BOUNDS: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::optional(
            125,
            "lower_bounds",
            Type::Map(MapType {
                key_field: Arc::new(NestedField::required(
                    126,
                    "key",
                    Type::Primitive(PrimitiveType::Int),
                )),
                value_field: Arc::new(NestedField::required(
                    127,
                    "value",
                    Type::Primitive(PrimitiveType::Binary),
                )),
            }),
        ))
    })
};

static UPPER_BOUNDS: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::optional(
            128,
            "upper_bounds",
            Type::Map(MapType {
                key_field: Arc::new(NestedField::required(
                    129,
                    "key",
                    Type::Primitive(PrimitiveType::Int),
                )),
                value_field: Arc::new(NestedField::required(
                    130,
                    "value",
                    Type::Primitive(PrimitiveType::Binary),
                )),
            }),
        ))
    })
};

static KEY_METADATA: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::optional(
            131,
            "key_metadata",
            Type::Primitive(PrimitiveType::Binary),
        ))
    })
};

static SPLIT_OFFSETS: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::optional(
            132,
            "split_offsets",
            Type::List(ListType {
                element_field: Arc::new(NestedField::required(
                    133,
                    "element",
                    Type::Primitive(PrimitiveType::Long),
                )),
            }),
        ))
    })
};

static EQUALITY_IDS: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::optional(
            135,
            "equality_ids",
            Type::List(ListType {
                element_field: Arc::new(NestedField::required(
                    136,
                    "element",
                    Type::Primitive(PrimitiveType::Int),
                )),
            }),
        ))
    })
};

static SORT_ORDER_ID: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::optional(
            140,
            "sort_order_id",
            Type::Primitive(PrimitiveType::Int),
        ))
    })
};

static FIRST_ROW_ID: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::optional(
            142,
            "first_row_id",
            Type::Primitive(PrimitiveType::Long),
        ))
    })
};

static REFERENCE_DATA_FILE: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::optional(
            143,
            "referenced_data_file",
            Type::Primitive(PrimitiveType::String),
        ))
    })
};

static CONTENT_OFFSET: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::optional(
            144,
            "content_offset",
            Type::Primitive(PrimitiveType::Long),
        ))
    })
};

static CONTENT_SIZE_IN_BYTES: Lazy<NestedFieldRef> = {
    Lazy::new(|| {
        Arc::new(NestedField::optional(
            145,
            "content_size_in_bytes",
            Type::Primitive(PrimitiveType::Long),
        ))
    })
};

fn data_file_fields_v3(partition_type: &StructType) -> Vec<NestedFieldRef> {
    vec![
        CONTENT.clone(),
        FILE_PATH.clone(),
        FILE_FORMAT.clone(),
        Arc::new(NestedField::required(
            102,
            "partition",
            Type::Struct(partition_type.clone()),
        )),
        RECORD_COUNT.clone(),
        FILE_SIZE_IN_BYTES.clone(),
        COLUMN_SIZES.clone(),
        VALUE_COUNTS.clone(),
        NULL_VALUE_COUNTS.clone(),
        NAN_VALUE_COUNTS.clone(),
        LOWER_BOUNDS.clone(),
        UPPER_BOUNDS.clone(),
        KEY_METADATA.clone(),
        SPLIT_OFFSETS.clone(),
        EQUALITY_IDS.clone(),
        SORT_ORDER_ID.clone(),
        FIRST_ROW_ID.clone(),
        REFERENCE_DATA_FILE.clone(),
        CONTENT_OFFSET.clone(),
        CONTENT_SIZE_IN_BYTES.clone(),
    ]
}

pub(super) fn data_file_schema_v3(partition_type: &StructType) -> Result<AvroSchema> {
    let schema = Schema::builder()
        .with_fields(data_file_fields_v3(partition_type))
        .build()?;
    schema_to_avro_schema("data_file", &schema)
}

fn data_file_fields_v2(partition_type: &StructType) -> Vec<NestedFieldRef> {
    vec![
        CONTENT.clone(),
        FILE_PATH.clone(),
        FILE_FORMAT.clone(),
        Arc::new(NestedField::required(
            102,
            "partition",
            Type::Struct(partition_type.clone()),
        )),
        RECORD_COUNT.clone(),
        FILE_SIZE_IN_BYTES.clone(),
        COLUMN_SIZES.clone(),
        VALUE_COUNTS.clone(),
        NULL_VALUE_COUNTS.clone(),
        NAN_VALUE_COUNTS.clone(),
        LOWER_BOUNDS.clone(),
        UPPER_BOUNDS.clone(),
        KEY_METADATA.clone(),
        SPLIT_OFFSETS.clone(),
        EQUALITY_IDS.clone(),
        SORT_ORDER_ID.clone(),
        FIRST_ROW_ID.clone(),
        REFERENCE_DATA_FILE.clone(),
        // Why are the following two fields here in the existing v2 schema?
        // In the spec, they are not even listed as optional for v2.
        CONTENT_OFFSET.clone(),
        CONTENT_SIZE_IN_BYTES.clone(),
    ]
}

pub(super) fn data_file_schema_v2(partition_type: &StructType) -> Result<AvroSchema> {
    let schema = Schema::builder()
        .with_fields(data_file_fields_v2(partition_type))
        .build()?;
    schema_to_avro_schema("data_file", &schema)
}

pub(super) fn manifest_schema_v2(partition_type: &StructType) -> Result<AvroSchema> {
    let fields = vec![
        STATUS.clone(),
        SNAPSHOT_ID_V2.clone(),
        SEQUENCE_NUMBER.clone(),
        FILE_SEQUENCE_NUMBER.clone(),
        Arc::new(NestedField::required(
            2,
            "data_file",
            Type::Struct(StructType::new(data_file_fields_v2(partition_type))),
        )),
    ];
    let schema = Schema::builder().with_fields(fields).build()?;
    schema_to_avro_schema("manifest_entry", &schema)
}

fn data_file_fields_v1(partition_type: &StructType) -> Vec<NestedFieldRef> {
    vec![
        FILE_PATH.clone(),
        FILE_FORMAT.clone(),
        Arc::new(NestedField::required(
            102,
            "partition",
            Type::Struct(partition_type.clone()),
        )),
        RECORD_COUNT.clone(),
        FILE_SIZE_IN_BYTES.clone(),
        BLOCK_SIZE_IN_BYTES.clone(),
        COLUMN_SIZES.clone(),
        VALUE_COUNTS.clone(),
        NULL_VALUE_COUNTS.clone(),
        NAN_VALUE_COUNTS.clone(),
        LOWER_BOUNDS.clone(),
        UPPER_BOUNDS.clone(),
        KEY_METADATA.clone(),
        SPLIT_OFFSETS.clone(),
        SORT_ORDER_ID.clone(),
    ]
}

pub(super) fn data_file_schema_v1(partition_type: &StructType) -> Result<AvroSchema> {
    let schema = Schema::builder()
        .with_fields(data_file_fields_v1(partition_type))
        .build()?;
    schema_to_avro_schema("data_file", &schema)
}

pub(super) fn manifest_schema_v1(partition_type: &StructType) -> Result<AvroSchema> {
    let fields = vec![
        STATUS.clone(),
        SNAPSHOT_ID_V1.clone(),
        Arc::new(NestedField::required(
            2,
            "data_file",
            Type::Struct(StructType::new(data_file_fields_v1(partition_type))),
        )),
    ];
    let schema = Schema::builder().with_fields(fields).build()?;
    schema_to_avro_schema("manifest_entry", &schema)
}

/// The shared error for both overflow doors of [`assign_first_row_ids`]. The `i64` door is the
/// reachable one (`i64::MAX` < `u64::MAX`); the `u64` door exists so the addition itself can never
/// wrap silently.
fn overflow_error(manifest_first_row_id: u64, entry: &ManifestEntry, what: &str) -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        format!("row-lineage first_row_id assignment {what}"),
    )
    .with_context("manifest_first_row_id", manifest_first_row_id.to_string())
    .with_context("file_path", entry.data_file.file_path.clone())
    .with_context("record_count", entry.data_file.record_count.to_string())
}

#[cfg(test)]
mod first_row_id_tests {
    use super::*;
    use crate::spec::{DataContentType, DataFileBuilder, DataFileFormat, Struct};

    /// A minimal data-file entry; fixtures vary `record_count` to keep the counter's steps
    /// distinguishable.
    fn entry(
        status: ManifestStatus,
        path: &str,
        record_count: u64,
        first_row_id: Option<i64>,
    ) -> ManifestEntry {
        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .partition(Struct::empty())
            .record_count(record_count)
            .file_size_in_bytes(100);
        let mut data_file = builder.build().expect("build data file");
        data_file.first_row_id = first_row_id;
        ManifestEntry {
            status,
            snapshot_id: None,
            sequence_number: None,
            file_sequence_number: None,
            data_file,
        }
    }

    fn assigned(entries: &[ManifestEntry]) -> Vec<Option<i64>> {
        entries.iter().map(|e| e.data_file.first_row_id).collect()
    }

    // The closed domain of `assign_first_row_ids`. Every skip cell is pinned by the id the next
    // assignable file receives.
    //
    // | Manifest `first_row_id` | Entry state | Java behaviour |
    // |---|---|---|
    // | absent | any | force `None` (OVERWRITING a stored value) |
    // | present | `Deleted` | skip; counter does NOT advance |
    // | present | already has an id | skip; counter does NOT advance |
    // | present | `Added`, no id | assign; counter += `record_count` |
    // | present | `Existing`, no id | assign; counter += `record_count` (the guard is `!= Deleted`) |

    /// The absent arm applies to EVERY entry state — Java's carries no status guard — so pinning
    /// it on `Added` alone would let a skip-mutation survive.
    #[test]
    fn absent_manifest_row_id_forces_every_file_to_none() {
        let mut entries = vec![
            entry(ManifestStatus::Added, "a.parquet", 10, None),
            entry(ManifestStatus::Added, "b.parquet", 10, Some(4242)),
            entry(ManifestStatus::Deleted, "c.parquet", 10, Some(77)),
            entry(ManifestStatus::Existing, "d.parquet", 10, Some(88)),
            entry(ManifestStatus::Existing, "e.parquet", 10, None),
        ];
        assign_first_row_ids(&mut entries, None).expect("assign");
        assert_eq!(
            assigned(&entries),
            vec![None; 5],
            "an unassigned manifest lends no ids, and OVERWRITES a stored one — in EVERY entry \
             state, since Java's absent arm has no status guard"
        );
    }

    #[test]
    fn present_manifest_row_id_assigns_a_running_total() {
        let mut entries = vec![
            entry(ManifestStatus::Added, "a.parquet", 10, None),
            entry(ManifestStatus::Added, "b.parquet", 3, None),
            entry(ManifestStatus::Added, "c.parquet", 7, None),
        ];
        assign_first_row_ids(&mut entries, Some(100)).expect("assign");
        assert_eq!(
            assigned(&entries),
            vec![Some(100), Some(110), Some(113)],
            "each file starts where the previous one's rows ended"
        );
    }

    #[test]
    fn a_deleted_entry_is_skipped_and_does_not_consume_ids() {
        let mut entries = vec![
            entry(ManifestStatus::Added, "a.parquet", 10, None),
            entry(ManifestStatus::Deleted, "gone.parquet", 999, None),
            entry(ManifestStatus::Added, "c.parquet", 7, None),
        ];
        assign_first_row_ids(&mut entries, Some(100)).expect("assign");
        assert_eq!(
            assigned(&entries),
            vec![Some(100), None, Some(110)],
            "the deleted entry gets no id AND does not advance the counter — `c` takes 110, not \
             1109, which is the discriminator between skipping and merely not-assigning"
        );
    }

    #[test]
    fn an_already_assigned_entry_is_skipped_and_does_not_consume_ids() {
        let mut entries = vec![
            entry(ManifestStatus::Added, "a.parquet", 10, None),
            entry(ManifestStatus::Existing, "kept.parquet", 999, Some(50)),
            entry(ManifestStatus::Added, "c.parquet", 7, None),
        ];
        assign_first_row_ids(&mut entries, Some(100)).expect("assign");
        assert_eq!(
            assigned(&entries),
            vec![Some(100), Some(50), Some(110)],
            "the stored id is PRESERVED (unlike the absent-manifest arm) and its record_count does \
             not advance the counter"
        );
    }

    /// The guard is `status != Deleted`, not `status == Added`: reusing `inherit_data`'s shape
    /// would leave every carried-forward file without a row id.
    #[test]
    fn an_existing_entry_without_an_id_is_still_assigned() {
        let mut entries = vec![
            entry(ManifestStatus::Existing, "a.parquet", 10, None),
            entry(ManifestStatus::Existing, "b.parquet", 5, None),
        ];
        assign_first_row_ids(&mut entries, Some(0)).expect("assign");
        assert_eq!(assigned(&entries), vec![Some(0), Some(10)]);
    }

    // Two non-interchangeable overflow doors: a test reaching only the `u64` one leaves the `i64`
    // narrowing — the reachable door — unpinned.

    /// The `i64` narrowing door: a manifest range already past `i64::MAX` cannot lend any id.
    #[test]
    fn a_row_id_beyond_i64_max_is_rejected() {
        let mut entries = vec![entry(ManifestStatus::Added, "a.parquet", 1, None)];
        let beyond_i64 = i64::MAX as u64 + 1;
        let error = assign_first_row_ids(&mut entries, Some(beyond_i64))
            .expect_err("a range starting past i64::MAX has no representable id");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains("exceeds i64::MAX"),
            "the i64 door must be the one that fires, got: {}",
            error.message()
        );
        assert_eq!(
            assigned(&entries),
            vec![None],
            "nothing is assigned on the failing entry — no wrapped, negative id is left behind"
        );
    }

    /// The `u64` addition door: the counter itself cannot wrap.
    #[test]
    fn an_overflowing_counter_is_rejected() {
        let mut entries = vec![
            entry(ManifestStatus::Added, "a.parquet", u64::MAX, None),
            entry(ManifestStatus::Added, "b.parquet", 1, None),
        ];
        let error = assign_first_row_ids(&mut entries, Some(1))
            .expect_err("the running total wraps u64 on the first file's record_count");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains("overflowed u64"),
            "the u64 door must be the one that fires, got: {}",
            error.message()
        );
    }

    /// An empty manifest is not an error in either arm.
    #[test]
    fn no_entries_is_a_no_op_in_both_arms() {
        let mut entries: Vec<ManifestEntry> = Vec::new();
        assign_first_row_ids(&mut entries, None).expect("absent arm");
        assign_first_row_ids(&mut entries, Some(7)).expect("present arm");
    }
}
