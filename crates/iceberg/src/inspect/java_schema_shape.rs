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

//! Java inspect schema-shape battery.
//!
//! Ports the schema assertions implied by Java
//! `core/.../MetadataTableUtils.createMetadataTableInstance`,
//! `core/.../BaseFilesTable.schema`, `core/.../PartitionsTable.schema`,
//! `core/.../BaseEntriesTable.schema` and `core/.../MetricsUtil.readableMetricsSchema`.
//! Each test names the Java class/method it cites. Additive only — no production
//! behaviour change.
//!
//! Increment 1 covered `MetadataTableUtils`, `BaseFilesTable` and `PartitionsTable`.
//! Increment 2 adds the `BaseEntriesTable` / `entries` *shape* and the
//! `readable_metrics` STRUCTURE recon.
//!
//! Out of the battery so far (conductor-13F A4): `PositionDeletesTable` row
//! behaviour and cross-spec `Partitioning.partitionType` unification. The
//! `readable_metrics` interior field-id ORDER is a *documented divergence*, not a
//! gap: Java assigns interior ids while walking `dataTableSchema.idToName()` —
//! Java-`HashMap` order, which is bucket-dependent and not portable across JVMs —
//! so there is no Java order to port. The fork assigns ids in ascending
//! data-table field-id order and then applies Java's by-name sort to the emitted
//! columns. These tests pin the fork's documented behaviour (see
//! [`crate::inspect::readable_metrics`] module docs) and deliberately do NOT
//! chase Java's HashMap order.

use crate::inspect::MetadataTableType;
use crate::scan::tests::TableTestFixture;
use crate::spec::{NestedFieldRef, PrimitiveType, Schema, Type};
use crate::table::Table;

/// Java `DataFile.getType(partitionType).fields()` — `api/DataFile.java` ids.
/// `BaseFilesTable.schema()` uses this list, then optionally
/// `TypeUtil.selectNot(..., DataFile.PARTITION_ID=102)`, then joins
/// `readable_metrics`.
const DATA_FILE_GET_TYPE_FIELDS: &[(i32, &str, bool)] = &[
    (134, "content", false),
    (100, "file_path", true),
    (101, "file_format", true),
    (141, "spec_id", false),
    (102, "partition", true),
    (103, "record_count", true),
    (104, "file_size_in_bytes", true),
    (108, "column_sizes", false),
    (109, "value_counts", false),
    (110, "null_value_counts", false),
    (137, "nan_value_counts", false),
    (125, "lower_bounds", false),
    (128, "upper_bounds", false),
    (131, "key_metadata", false),
    (132, "split_offsets", false),
    (135, "equality_ids", false),
    (140, "sort_order_id", false),
    (142, "first_row_id", false),
    (143, "referenced_data_file", false),
    (144, "content_offset", false),
    (145, "content_size_in_bytes", false),
];

/// Java `DataFile.PARTITION_ID` (`api/DataFile.java`).
const DATA_FILE_PARTITION_ID: i32 = 102;

/// Java `PartitionsTable` constructor field list (`core/.../PartitionsTable.java`).
/// `schema()` returns this list, or `TypeUtil.select` of the count/timestamp ids
/// when the table is unpartitioned.
const PARTITIONS_TABLE_FIELDS: &[(i32, &str, bool)] = &[
    (1, "partition", true),
    (4, "spec_id", true),
    (2, "record_count", true),
    (3, "file_count", true),
    (11, "total_data_file_size_in_bytes", true),
    (5, "position_delete_record_count", true),
    (6, "position_delete_file_count", true),
    (7, "equality_delete_record_count", true),
    (8, "equality_delete_file_count", true),
    (9, "last_updated_at", false),
    (10, "last_updated_snapshot_id", false),
];

/// Java `PartitionsTable.schema()` `TypeUtil.select` id set when unpartitioned.
const PARTITIONS_UNPARTITIONED_FIELD_IDS: &[i32] = &[2, 3, 11, 5, 6, 7, 8, 9, 10];

/// Java `ManifestEntry.wrapFileSchema(fileType)` (`core/.../ManifestEntry.java`):
/// `STATUS = required(0, ...)`, `SNAPSHOT_ID = optional(1, ...)`,
/// `SEQUENCE_NUMBER = optional(3, ...)`, `FILE_SEQUENCE_NUMBER = optional(4, ...)`,
/// then `required(DATA_FILE_ID = 2, "data_file", fileType)`. Note the ids are
/// declared 0/1/3/4 and the `data_file` row lands LAST while holding id 2 — the
/// interleave is load-bearing for Spark readers.
const MANIFEST_ENTRY_WRAP_FIELDS: &[(i32, &str, bool)] = &[
    (0, "status", true),
    (1, "snapshot_id", false),
    (3, "sequence_number", false),
    (4, "file_sequence_number", false),
    (2, "data_file", true),
];

/// Java `MetricsUtil.READABLE_METRIC_COLS` order (`core/.../MetricsUtil.java`).
/// `true` = the sub-field is a fixed `Types.LongType`; `false` = its `colType` is
/// `Types.NestedField::type`, i.e. the data COLUMN's own primitive type.
const READABLE_METRIC_COLS: &[(&str, bool)] = &[
    ("column_size", true),
    ("value_count", true),
    ("null_value_count", true),
    ("nan_value_count", true),
    ("lower_bound", false),
    ("upper_bound", false),
];

/// The leaf (primitive) columns of [`TableTestFixture`]'s current schema, in
/// ASCENDING data-table field-id order. Java's `readableMetricsSchema` walks
/// `dataTableSchema.idToName().keySet()` (HashMap order) to ASSIGN ids, then sorts
/// the emitted fields by NAME. This fixture is chosen so the two orders differ,
/// which is what makes the by-name sort observable.
const FIXTURE_LEAF_COLUMNS_BY_FIELD_ID: &[(&str, PrimitiveType)] = &[
    ("x", PrimitiveType::Long),
    ("y", PrimitiveType::Long),
    ("z", PrimitiveType::Long),
    ("a", PrimitiveType::String),
    ("dbl", PrimitiveType::Double),
    ("i32", PrimitiveType::Int),
    ("i64", PrimitiveType::Long),
    ("bool", PrimitiveType::Boolean),
];

/// Java seeds the readable-metrics id counter at the HOST metadata table's
/// `highestFieldId()`. Partitioned, the `entries`/`files` schema nests the
/// partition struct whose field ids start at `PARTITION_DATA_ID_START = 1000`, so
/// the seed is 1000; unpartitioned the partition field is dropped first and the
/// seed falls back to `DataFile` id 145 (`content_size_in_bytes`).
const READABLE_METRICS_SEED_PARTITIONED: i32 = 1000;
const READABLE_METRICS_SEED_UNPARTITIONED: i32 = 145;

/// The two Rust analogues of Java `ManifestEntriesTable` / `AllEntriesTable` —
/// both inherit `BaseEntriesTable.schema()`.
fn entries_family_schemas(table: &Table) -> [Schema; 2] {
    let inspect = table.inspect();
    [inspect.entries().schema(), inspect.all_entries().schema()]
}

/// The child fields of a top-level STRUCT column (`data_file` / `readable_metrics`).
fn struct_fields<'a>(schema: &'a Schema, name: &str) -> &'a [NestedFieldRef] {
    let field = schema
        .field_by_name(name)
        .unwrap_or_else(|| panic!("schema has no field {name}"));
    match field.field_type.as_ref() {
        Type::Struct(struct_type) => struct_type.fields(),
        other => panic!("field {name} must be a struct, got {other}"),
    }
}

/// The child fields of a nested struct, addressed by its parent's child list.
fn nested_struct_fields<'a>(fields: &'a [NestedFieldRef], name: &str) -> &'a [NestedFieldRef] {
    let field = fields
        .iter()
        .find(|field| field.name == name)
        .unwrap_or_else(|| panic!("struct has no child {name}"));
    match field.field_type.as_ref() {
        Type::Struct(struct_type) => struct_type.fields(),
        other => panic!("child {name} must be a struct, got {other}"),
    }
}

fn assert_nested_field_row(
    fields: &[NestedFieldRef],
    index: usize,
    id: i32,
    name: &str,
    required: bool,
    context: &str,
) {
    let field = &fields[index];
    assert_eq!(
        field.id, id,
        "{context} field[{index}] id: expected {id}, got {}",
        field.id
    );
    assert_eq!(
        field.name, name,
        "{context} field[{index}] name: expected {name}, got {}",
        field.name
    );
    assert_eq!(
        field.required, required,
        "{context} field[{index}] ({name}) required: expected {required}, got {}",
        field.required
    );
}

/// The six Rust analogues of Java `FilesTable` / `DataFilesTable` /
/// `DeleteFilesTable` / `AllFilesTable` / `AllDataFilesTable` /
/// `AllDeleteFilesTable` — all inherit `BaseFilesTable.schema()`.
fn files_family_schemas(table: &Table) -> [Schema; 6] {
    let inspect = table.inspect();
    [
        inspect.files().schema(),
        inspect.data_files().schema(),
        inspect.delete_files().schema(),
        inspect.all_files().schema(),
        inspect.all_data_files().schema(),
        inspect.all_delete_files().schema(),
    ]
}

/// Java `MetadataTableUtils.createMetadataTableInstance` switch — every type
/// the Rust enum currently exposes (POSITION_DELETES landed schema-only in FB-2).
fn schema_for_metadata_table_type(table: &Table, table_type: MetadataTableType) -> Schema {
    let inspect = table.inspect();
    match table_type {
        MetadataTableType::Snapshots => inspect.snapshots().schema(),
        MetadataTableType::Manifests => inspect.manifests().schema(),
        MetadataTableType::Files => inspect.files().schema(),
        MetadataTableType::DataFiles => inspect.data_files().schema(),
        MetadataTableType::DeleteFiles => inspect.delete_files().schema(),
        MetadataTableType::Entries => inspect.entries().schema(),
        MetadataTableType::AllFiles => inspect.all_files().schema(),
        MetadataTableType::AllDataFiles => inspect.all_data_files().schema(),
        MetadataTableType::AllDeleteFiles => inspect.all_delete_files().schema(),
        MetadataTableType::AllEntries => inspect.all_entries().schema(),
        MetadataTableType::History => inspect.history().schema(),
        MetadataTableType::Refs => inspect.refs().schema(),
        MetadataTableType::MetadataLogEntries => inspect.metadata_log_entries().schema(),
        MetadataTableType::Partitions => inspect.partitions().schema(),
        MetadataTableType::AllManifests => inspect.all_manifests().schema(),
        MetadataTableType::PositionDeletes => inspect.position_deletes().schema(),
    }
}

fn assert_field_row(schema: &Schema, index: usize, id: i32, name: &str, required: bool) {
    let field = &schema.as_struct().fields()[index];
    assert_eq!(
        field.id, id,
        "schema field[{index}] id: expected {id}, got {}",
        field.id
    );
    assert_eq!(
        field.name, name,
        "schema field[{index}] name: expected {name}, got {}",
        field.name
    );
    assert_eq!(
        field.required, required,
        "schema field[{index}] ({name}) required: expected {required}, got {}",
        field.required
    );
    let by_id = schema
        .field_by_id(id)
        .unwrap_or_else(|| panic!("schema has no field id {id} ({name})"));
    assert_eq!(by_id.name, name);
    assert_eq!(by_id.required, required);
}

#[test]
fn metadata_table_utils_create_instance_types_are_constructible() {
    // RISK: a MetadataTableUtils switch arm whose Rust analogue cannot produce a
    // schema means the increment-1 battery is not locking the Java type set.
    // Cite: MetadataTableUtils.createMetadataTableInstance (the private
    // switch on MetadataTableType). POSITION_DELETES joined schema-only in FB-2.
    let fixture = TableTestFixture::new();
    for table_type in MetadataTableType::all_types() {
        let schema = schema_for_metadata_table_type(&fixture.table, table_type.clone());
        assert!(
            !schema.as_struct().fields().is_empty(),
            "{} schema must not be empty",
            table_type.as_str()
        );
    }
    let constructed: Vec<String> = MetadataTableType::all_types()
        .map(|table_type| table_type.as_str().to_string())
        .collect();

    assert_eq!(constructed, vec![
        "snapshots",
        "manifests",
        "files",
        "data_files",
        "delete_files",
        "entries",
        "all_files",
        "all_data_files",
        "all_delete_files",
        "all_entries",
        "history",
        "refs",
        "metadata_log_entries",
        "partitions",
        "all_manifests",
        "position_deletes",
    ]);
}

#[test]
fn metadata_table_utils_has_metadata_table_name_vocabulary() {
    // RISK: Java MetadataTableUtils.hasMetadataTableName is
    // `MetadataTableType.from(identifier.name()) != null`. A silent rename of
    // as_str()/TryFrom would desync the Spark `$suffix` vocabulary (#194 F-3).
    // Cite: MetadataTableUtils.hasMetadataTableName + MetadataTableType.from.
    for table_type in MetadataTableType::all_types() {
        let name = table_type.as_str();
        let parsed = MetadataTableType::try_from(name)
            .unwrap_or_else(|err| panic!("TryFrom({name:?}) must succeed: {err}"));
        assert_eq!(parsed.as_str(), name);
    }
    assert!(
        MetadataTableType::try_from("not_a_metadata_table").is_err(),
        "unknown suffix must not parse as a metadata table type"
    );
}

#[test]
fn base_files_table_schema_matches_data_file_get_type_then_readable_metrics() {
    // RISK: field-id / name / required drift on the files-family schema vs
    // Java DataFile.getType + BaseFilesTable.schema (join readable_metrics last).
    // Cite: BaseFilesTable.schema(); DataFile.getType(partitionType).
    let fixture = TableTestFixture::new();
    let schema = fixture.table.inspect().files().schema();
    let fields = schema.as_struct().fields();

    assert_eq!(
        fields.len(),
        DATA_FILE_GET_TYPE_FIELDS.len() + 1,
        "partitioned BaseFilesTable schema is DataFile.getType + readable_metrics"
    );
    for (index, &(id, name, required)) in DATA_FILE_GET_TYPE_FIELDS.iter().enumerate() {
        assert_field_row(&schema, index, id, name, required);
    }
    let readable = fields
        .last()
        .expect("readable_metrics is appended after the DataFile projection");
    assert_eq!(readable.name, "readable_metrics");
    assert!(
        !readable.required,
        "Java MetricsUtil.readableMetricsSchema field is optional"
    );
    assert!(
        matches!(readable.field_type.as_ref(), Type::Struct(_)),
        "readable_metrics must be a struct (interior field-id order is OUT of this increment)"
    );

    let partition = schema
        .field_by_id(DATA_FILE_PARTITION_ID)
        .expect("partitioned schema keeps DataFile.PARTITION_ID=102");
    assert!(matches!(partition.field_type.as_ref(), Type::Struct(_)));
    assert!(partition.required);

    let content = schema
        .field_by_name("content")
        .expect("content is DataFile field 134");
    assert_eq!(
        content.field_type.as_ref(),
        &Type::Primitive(PrimitiveType::Int)
    );
    let file_path = schema
        .field_by_name("file_path")
        .expect("file_path is DataFile field 100");
    assert_eq!(
        file_path.field_type.as_ref(),
        &Type::Primitive(PrimitiveType::String)
    );
    let record_count = schema
        .field_by_name("record_count")
        .expect("record_count is DataFile field 103");
    assert_eq!(
        record_count.field_type.as_ref(),
        &Type::Primitive(PrimitiveType::Long)
    );
}

#[test]
fn base_files_table_schema_drops_only_partition_id_when_unpartitioned() {
    // RISK: Java BaseFilesTable.schema() does TypeUtil.selectNot(schema,
    // DataFile.PARTITION_ID) when partitionType.fields() is empty — it does
    // NOT drop spec_id, and it still joins readable_metrics afterwards.
    // Cite: BaseFilesTable.schema() empty-partition branch.
    let fixture = TableTestFixture::new_unpartitioned();
    let expected: Vec<(i32, &str, bool)> = DATA_FILE_GET_TYPE_FIELDS
        .iter()
        .copied()
        .filter(|(id, _, _)| *id != DATA_FILE_PARTITION_ID)
        .collect();

    for schema in files_family_schemas(&fixture.table) {
        let fields = schema.as_struct().fields();
        assert_eq!(
            fields.len(),
            expected.len() + 1,
            "unpartitioned files-family schema is DataFile.getType minus 102, plus readable_metrics"
        );
        for (index, &(id, name, required)) in expected.iter().enumerate() {
            assert_field_row(&schema, index, id, name, required);
        }
        assert!(
            schema.field_by_id(DATA_FILE_PARTITION_ID).is_none(),
            "unpartitioned BaseFilesTable must drop field 102"
        );
        assert!(
            schema.field_by_name("spec_id").is_some(),
            "Java drops only PARTITION_ID, not spec_id"
        );
        assert_eq!(
            fields.last().expect("readable_metrics still appended").name,
            "readable_metrics"
        );
    }
}

#[test]
fn base_files_table_six_analogues_share_one_schema() {
    // RISK: FilesTable / DataFilesTable / DeleteFilesTable / All* inherit
    // BaseFilesTable.schema() — a per-analogue drift is a silent Java break.
    // Cite: BaseFilesTable.schema() (inherited; the six tables differ only by
    // which manifests they scan).
    let fixture = TableTestFixture::new();
    let schemas = files_family_schemas(&fixture.table);
    let reference = schemas[0].as_struct().fields();
    for (index, schema) in schemas.iter().enumerate().skip(1) {
        let fields = schema.as_struct().fields();
        assert_eq!(
            fields.len(),
            reference.len(),
            "files-family analogue {index} field count drifted from files"
        );
        for (field_index, (got, want)) in fields.iter().zip(reference.iter()).enumerate() {
            assert_eq!(got.id, want.id, "analogue {index} field[{field_index}] id");
            assert_eq!(
                got.name, want.name,
                "analogue {index} field[{field_index}] name"
            );
            assert_eq!(
                got.required, want.required,
                "analogue {index} field[{field_index}] required"
            );
        }
    }
}

#[test]
fn partitions_table_schema_matches_java_constructor_field_list() {
    // RISK: PartitionsTable field ids are non-sequential (partition/1, spec_id/4,
    // record_count/2, …). A sequential rewrite would break Spark readers.
    // Cite: PartitionsTable constructor field list + schema() (partitioned path).
    let fixture = TableTestFixture::new();
    let schema = fixture.table.inspect().partitions().schema();
    let fields = schema.as_struct().fields();
    assert_eq!(fields.len(), PARTITIONS_TABLE_FIELDS.len());
    for (index, &(id, name, required)) in PARTITIONS_TABLE_FIELDS.iter().enumerate() {
        assert_field_row(&schema, index, id, name, required);
    }
    let partition = schema
        .field_by_id(1)
        .expect("partitioned PartitionsTable keeps field 1");
    assert!(matches!(partition.field_type.as_ref(), Type::Struct(_)));
    let spec_id = schema.field_by_name("spec_id").expect("spec_id / 4");
    assert_eq!(
        spec_id.field_type.as_ref(),
        &Type::Primitive(PrimitiveType::Int)
    );
    let record_count = schema
        .field_by_name("record_count")
        .expect("record_count / 2");
    assert_eq!(
        record_count.field_type.as_ref(),
        &Type::Primitive(PrimitiveType::Long)
    );
    let last_updated_at = schema
        .field_by_name("last_updated_at")
        .expect("last_updated_at / 9");
    assert_eq!(
        last_updated_at.field_type.as_ref(),
        &Type::Primitive(PrimitiveType::Timestamptz)
    );
}

#[test]
fn partitions_table_schema_type_util_select_when_unpartitioned() {
    // RISK: Java PartitionsTable.schema() TypeUtil.selects ids
    // {2,3,11,5,6,7,8,9,10} — dropping partition/1 AND spec_id/4.
    // Cite: PartitionsTable.schema() unpartitioned branch.
    let fixture = TableTestFixture::new_unpartitioned();
    let schema = fixture.table.inspect().partitions().schema();
    let fields = schema.as_struct().fields();
    assert_eq!(fields.len(), PARTITIONS_UNPARTITIONED_FIELD_IDS.len());
    let expected: Vec<(i32, &str, bool)> = PARTITIONS_TABLE_FIELDS
        .iter()
        .copied()
        .filter(|(id, _, _)| PARTITIONS_UNPARTITIONED_FIELD_IDS.contains(id))
        .collect();
    for (index, &(id, name, required)) in expected.iter().enumerate() {
        assert_field_row(&schema, index, id, name, required);
    }
    assert!(schema.field_by_id(1).is_none(), "drops partition / 1");
    assert!(schema.field_by_id(4).is_none(), "drops spec_id / 4");
}

#[test]
fn base_entries_table_schema_wraps_manifest_entry_get_schema() {
    // RISK: `entries` is NOT a flat DataFile projection — Java wraps it:
    // BaseEntriesTable.schema() = ManifestEntry.getSchema(partitionType)
    //   = wrapFileSchema(DataFile.getType(partitionType)),
    // i.e. four scalar rows (ids 0/1/3/4) plus a REQUIRED `data_file` struct that
    // holds id 2 while sitting last. Flattening it, or renumbering the
    // 0/1/3/4-then-2 interleave, silently breaks every Spark `entries` reader.
    // Cite: BaseEntriesTable.schema(); ManifestEntry.wrapFileSchema + the
    // STATUS/SNAPSHOT_ID/SEQUENCE_NUMBER/FILE_SEQUENCE_NUMBER/DATA_FILE_ID constants.
    let fixture = TableTestFixture::new();
    for schema in entries_family_schemas(&fixture.table) {
        let fields = schema.as_struct().fields();
        assert_eq!(
            fields.len(),
            MANIFEST_ENTRY_WRAP_FIELDS.len() + 1,
            "entries schema is ManifestEntry.wrapFileSchema + readable_metrics"
        );
        for (index, &(id, name, required)) in MANIFEST_ENTRY_WRAP_FIELDS.iter().enumerate() {
            assert_field_row(&schema, index, id, name, required);
        }

        let status = schema.field_by_id(0).expect("ManifestEntry.STATUS / 0");
        assert_eq!(
            status.field_type.as_ref(),
            &Type::Primitive(PrimitiveType::Int),
            "Java ManifestEntry.STATUS is IntegerType"
        );
        for id in [1, 3, 4] {
            let field = schema
                .field_by_id(id)
                .unwrap_or_else(|| panic!("ManifestEntry field {id}"));
            assert_eq!(
                field.field_type.as_ref(),
                &Type::Primitive(PrimitiveType::Long),
                "ManifestEntry field {id} ({}) is LongType",
                field.name
            );
        }

        // `data_file` carries the SAME DataFile.getType projection BaseFilesTable
        // exposes flat — here nested one level down.
        let data_file = struct_fields(&schema, "data_file");
        assert_eq!(
            data_file.len(),
            DATA_FILE_GET_TYPE_FIELDS.len(),
            "entries.data_file is exactly DataFile.getType(partitionType)"
        );
        for (index, &(id, name, required)) in DATA_FILE_GET_TYPE_FIELDS.iter().enumerate() {
            assert_nested_field_row(data_file, index, id, name, required, "entries.data_file");
        }

        let readable = fields
            .last()
            .expect("TypeUtil.join appends readable_metrics last");
        assert_eq!(readable.name, "readable_metrics");
        assert!(
            !readable.required,
            "Java MetricsUtil.readableMetricsSchema emits an optional struct"
        );
        assert!(matches!(readable.field_type.as_ref(), Type::Struct(_)));
    }
}

#[test]
fn base_entries_table_schema_drops_only_nested_partition_when_unpartitioned() {
    // RISK: Java's empty-partition branch is TypeUtil.selectNot(schema,
    // {DataFile.PARTITION_ID}) applied to the WRAPPED schema — so it removes
    // `data_file.partition` (nested, id 102) and nothing else. The five top-level
    // rows survive, `data_file` stays required, spec_id/141 stays, and
    // readable_metrics is still joined AFTER the drop (which is what moves the id
    // seed from 1000 down to 145). Dropping the whole `data_file` row, or
    // short-circuiting the join, are both silent Java breaks. Landed in FW-3 (#194).
    // Cite: BaseEntriesTable.schema() empty-partition branch.
    let fixture = TableTestFixture::new_unpartitioned();
    let expected: Vec<(i32, &str, bool)> = DATA_FILE_GET_TYPE_FIELDS
        .iter()
        .copied()
        .filter(|(id, _, _)| *id != DATA_FILE_PARTITION_ID)
        .collect();

    for schema in entries_family_schemas(&fixture.table) {
        let fields = schema.as_struct().fields();
        assert_eq!(
            fields.len(),
            MANIFEST_ENTRY_WRAP_FIELDS.len() + 1,
            "the unpartitioned drop is nested — the top-level row count is unchanged"
        );
        for (index, &(id, name, required)) in MANIFEST_ENTRY_WRAP_FIELDS.iter().enumerate() {
            assert_field_row(&schema, index, id, name, required);
        }

        let data_file = struct_fields(&schema, "data_file");
        assert_eq!(data_file.len(), expected.len());
        for (index, &(id, name, required)) in expected.iter().enumerate() {
            assert_nested_field_row(data_file, index, id, name, required, "entries.data_file");
        }
        assert!(
            !data_file
                .iter()
                .any(|field| field.id == DATA_FILE_PARTITION_ID),
            "unpartitioned entries must drop nested field 102"
        );
        assert!(
            data_file.iter().any(|field| field.name == "spec_id"),
            "Java drops only PARTITION_ID, not spec_id"
        );
        assert_eq!(
            fields
                .last()
                .expect("readable_metrics still joined after the drop")
                .name,
            "readable_metrics"
        );
    }
}

#[test]
fn base_entries_table_two_analogues_share_one_schema() {
    // RISK: ManifestEntriesTable and AllEntriesTable both inherit
    // BaseEntriesTable.schema() and differ ONLY in which manifests they scan. A
    // per-analogue schema drift is a silent Java break.
    // Cite: BaseEntriesTable.schema() (inherited by ManifestEntriesTable /
    // AllEntriesTable).
    for fixture in [
        TableTestFixture::new(),
        TableTestFixture::new_unpartitioned(),
    ] {
        let [entries, all_entries] = entries_family_schemas(&fixture.table);
        assert_eq!(
            entries.as_struct(),
            all_entries.as_struct(),
            "all_entries must inherit the entries schema verbatim"
        );
    }
}

#[test]
fn base_entries_table_data_file_projection_equals_base_files_table_projection() {
    // RISK: BaseFilesTable.schema() and BaseEntriesTable.schema() both build on
    // DataFile.getType(partitionType) — files exposes it FLAT, entries NESTS it
    // under `data_file`. Both then apply the SAME selectNot(102) empty-partition
    // rule. If the two projections ever diverge, one of the two tables has drifted
    // from DataFile.getType.
    // Cite: BaseFilesTable.schema(); BaseEntriesTable.schema();
    // DataFile.getType(partitionType).
    for fixture in [
        TableTestFixture::new(),
        TableTestFixture::new_unpartitioned(),
    ] {
        let files = fixture.table.inspect().files().schema();
        let files_fields = files.as_struct().fields();
        // BaseFilesTable is the projection with readable_metrics appended; strip it.
        let (files_readable, files_projection) = files_fields
            .split_last()
            .expect("files schema ends with readable_metrics");
        assert_eq!(files_readable.name, "readable_metrics");

        let entries = fixture.table.inspect().entries().schema();
        let data_file = struct_fields(&entries, "data_file");

        assert_eq!(
            data_file.len(),
            files_projection.len(),
            "entries.data_file and the files projection are the same DataFile.getType list"
        );
        for (index, (nested, flat)) in data_file.iter().zip(files_projection.iter()).enumerate() {
            assert_eq!(nested.id, flat.id, "DataFile field[{index}] id");
            assert_eq!(nested.name, flat.name, "DataFile field[{index}] name");
            assert_eq!(
                nested.required, flat.required,
                "DataFile field[{index}] required"
            );
            assert_eq!(
                nested.field_type, flat.field_type,
                "DataFile field[{index}] type"
            );
        }
    }
}

#[test]
fn readable_metrics_has_one_optional_struct_per_primitive_leaf_column() {
    // RISK: Java emits a per-column struct ONLY for ids whose
    // dataTableSchema.findField(id).type().isPrimitiveType(); each is
    // NestedField.optional(...).withDoc("Metrics for column %s"), and the whole
    // column is optional with doc "Column metrics in readable form". A required
    // struct, a missing/renamed doc, or a column set drawn from the METADATA table
    // instead of the DATA table are all Java breaks.
    // Cite: MetricsUtil.readableMetricsSchema.
    let fixture = TableTestFixture::new();
    for schema in entries_family_schemas(&fixture.table) {
        let readable = schema
            .field_by_name("readable_metrics")
            .expect("readable_metrics is joined onto the entries schema");
        assert!(!readable.required);
        assert_eq!(
            readable.doc.as_deref(),
            Some("Column metrics in readable form"),
            "Java passes this doc to the top-level readable_metrics field"
        );

        let columns = struct_fields(&schema, "readable_metrics");
        assert_eq!(
            columns.len(),
            FIXTURE_LEAF_COLUMNS_BY_FIELD_ID.len(),
            "one struct per PRIMITIVE data-table column"
        );
        for column in columns {
            assert!(
                !column.required,
                "per-column metric struct {} must be optional",
                column.name
            );
            assert_eq!(
                column.doc.as_deref(),
                Some(format!("Metrics for column {}", column.name).as_str()),
                "Java doc for column {}",
                column.name
            );
            assert!(matches!(column.field_type.as_ref(), Type::Struct(_)));
        }
    }
}

#[test]
fn readable_metrics_column_struct_is_the_six_java_metric_cols() {
    // RISK: the interior of each per-column struct is fixed by
    // MetricsUtil.READABLE_METRIC_COLS: six optional sub-fields, in that list
    // order, the first four LongType and the last two typed by the DATA column
    // (Types.NestedField::type as colType). The bound typing is the whole point of
    // `readable_metrics` — it is the decoded inverse of the raw
    // lower_bounds/upper_bounds byte maps.
    // Cite: MetricsUtil.READABLE_METRIC_COLS; MetricsUtil.readableMetricsSchema.
    let fixture = TableTestFixture::new();
    let schema = fixture.table.inspect().entries().schema();
    let columns = struct_fields(&schema, "readable_metrics");

    for &(column_name, ref column_type) in FIXTURE_LEAF_COLUMNS_BY_FIELD_ID {
        let metrics = nested_struct_fields(columns, column_name);
        assert_eq!(
            metrics.len(),
            READABLE_METRIC_COLS.len(),
            "column {column_name} must carry the six READABLE_METRIC_COLS"
        );
        for (index, &(metric_name, is_long)) in READABLE_METRIC_COLS.iter().enumerate() {
            let field = &metrics[index];
            assert_eq!(
                field.name, metric_name,
                "column {column_name} metric[{index}] name"
            );
            assert!(
                !field.required,
                "column {column_name} metric {metric_name} must be optional"
            );
            let expected = if is_long {
                Type::Primitive(PrimitiveType::Long)
            } else {
                Type::Primitive(column_type.clone())
            };
            assert_eq!(
                field.field_type.as_ref(),
                &expected,
                "column {column_name} metric {metric_name} type"
            );
        }
    }
}

#[test]
fn readable_metrics_columns_are_emitted_in_name_order_with_id_order_divergence() {
    // RISK: Java's LAST act in readableMetricsSchema is
    // `fields.sort(Comparator.comparing(NestedField::name))` — the emitted column
    // ORDER is by NAME, and that is the portable, Java-citable rule. The ids,
    // however, are assigned BEFORE that sort while walking
    // dataTableSchema.idToName() — Java-HashMap order, which is bucket-dependent
    // and NOT portable across JVMs, so there is no Java order to port. The fork
    // assigns in ascending data-table field-id order instead; this test pins BOTH
    // halves of that documented divergence, deliberately without chasing Java's
    // HashMap order.
    // Cite: MetricsUtil.readableMetricsSchema (the sort, and the idToName walk);
    // crate::inspect::readable_metrics module docs (the divergence).
    let fixture = TableTestFixture::new();
    let schema = fixture.table.inspect().entries().schema();
    let columns = struct_fields(&schema, "readable_metrics");

    // Half 1 — Java's rule: emitted in name order.
    let emitted: Vec<&str> = columns.iter().map(|field| field.name.as_str()).collect();
    let mut by_name: Vec<&str> = FIXTURE_LEAF_COLUMNS_BY_FIELD_ID
        .iter()
        .map(|&(name, _)| name)
        .collect();
    by_name.sort_unstable();
    assert_eq!(
        emitted, by_name,
        "readable_metrics columns are sorted by name (Java fields.sort)"
    );
    // The fixture must actually discriminate: its field-id order is NOT its name
    // order, otherwise this test would pass vacuously.
    let by_field_id: Vec<&str> = FIXTURE_LEAF_COLUMNS_BY_FIELD_ID
        .iter()
        .map(|&(name, _)| name)
        .collect();
    assert_ne!(
        by_name, by_field_id,
        "fixture must have differing name and field-id orders for this pin to bite"
    );

    // Half 2 — the fork's documented id assignment: ascending DATA-table field id,
    // so the id order follows `x, y, z, a, dbl, i32, i64, bool`, NOT the name order.
    let id_of = |name: &str| -> i32 {
        columns
            .iter()
            .find(|field| field.name == name)
            .unwrap_or_else(|| panic!("readable_metrics has no column {name}"))
            .id
    };
    let ids_in_field_id_order: Vec<i32> = by_field_id.iter().map(|name| id_of(name)).collect();
    let mut ascending = ids_in_field_id_order.clone();
    ascending.sort_unstable();
    assert_eq!(
        ids_in_field_id_order, ascending,
        "ids are assigned in ascending data-table field-id order (documented divergence)"
    );
    assert!(
        id_of("x") < id_of("a"),
        "x (field id 1) is numbered before a (field id 4) despite sorting after it by name"
    );
}

#[test]
fn readable_metrics_id_counter_seeds_at_the_host_metadata_highest_field_id() {
    // RISK: Java seeds `new AtomicInteger(metadataTableSchema.highestFieldId())`
    // and PRE-increments, assigning, per column, the column-struct id then its six
    // sub-field ids, and finally the top-level readable_metrics id last. The seed
    // is the HOST schema's highest id, so it is 1000 partitioned (the nested
    // partition struct's ids start at PARTITION_DATA_ID_START = 1000) and 145
    // unpartitioned (DataFile.CONTENT_SIZE_IN_BYTES, after selectNot dropped 102).
    // Seeding off a constant, or joining readable_metrics BEFORE the partition
    // drop, would collide readable-metrics ids with real DataFile ids.
    // Cite: MetricsUtil.readableMetricsSchema; BaseEntriesTable.schema() ordering.
    const PER_COLUMN_IDS: i32 = 1 + 6;
    for (fixture, seed) in [
        (TableTestFixture::new(), READABLE_METRICS_SEED_PARTITIONED),
        (
            TableTestFixture::new_unpartitioned(),
            READABLE_METRICS_SEED_UNPARTITIONED,
        ),
    ] {
        let schema = fixture.table.inspect().entries().schema();
        let readable = schema.field_by_name("readable_metrics").expect("joined");
        let columns = struct_fields(&schema, "readable_metrics");
        let column_count = FIXTURE_LEAF_COLUMNS_BY_FIELD_ID.len() as i32;

        // Per column: struct id, then six contiguous sub-field ids.
        let mut next = seed;
        for &(column_name, _) in FIXTURE_LEAF_COLUMNS_BY_FIELD_ID {
            next += 1;
            let column = columns
                .iter()
                .find(|field| field.name == column_name)
                .unwrap_or_else(|| panic!("readable_metrics has no column {column_name}"));
            assert_eq!(
                column.id, next,
                "column {column_name} struct id (seed {seed}, pre-increment counter)"
            );
            for metric in nested_struct_fields(columns, column_name) {
                next += 1;
                assert_eq!(
                    metric.id, next,
                    "column {column_name} sub-field {} id",
                    metric.name
                );
            }
        }
        // The top-level field takes the LAST id.
        next += 1;
        assert_eq!(
            readable.id, next,
            "readable_metrics takes the final pre-incremented id"
        );
        assert_eq!(
            readable.id,
            seed + column_count * PER_COLUMN_IDS + 1,
            "total ids consumed = 1 + 6 per column, plus the top-level field"
        );
        // No readable-metrics id may collide with a host-schema id.
        assert!(
            columns.iter().all(|column| column.id > seed),
            "every readable_metrics id must exceed the host highest field id"
        );
    }
}

#[test]
fn readable_metrics_shape_is_identical_across_files_and_entries() {
    // RISK: Java calls the SAME MetricsUtil.readableMetricsSchema from
    // BaseFilesTable.schema() and BaseEntriesTable.schema(), with the same data
    // table schema and hosts whose highestFieldId coincides. A `files`-only or
    // `entries`-only readable_metrics implementation would drift the two apart.
    // Cite: BaseFilesTable.schema(); BaseEntriesTable.schema(); both end in
    // TypeUtil.join(schema, MetricsUtil.readableMetricsSchema(table().schema(), schema)).
    for fixture in [
        TableTestFixture::new(),
        TableTestFixture::new_unpartitioned(),
    ] {
        let files = fixture.table.inspect().files().schema();
        let entries = fixture.table.inspect().entries().schema();
        let files_readable = files.field_by_name("readable_metrics").expect("files");
        let entries_readable = entries.field_by_name("readable_metrics").expect("entries");
        assert_eq!(
            files_readable, entries_readable,
            "files and entries must join the identical readable_metrics field"
        );
    }
}
