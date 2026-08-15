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

//! Increment-1 Java inspect schema-shape battery.
//!
//! Ports the schema assertions implied by Java
//! `core/.../MetadataTableUtils.createMetadataTableInstance`,
//! `core/.../BaseFilesTable.schema`, and `core/.../PartitionsTable.schema`.
//! Each test names the Java class/method it cites. Additive only — no production
//! behaviour change.
//!
//! Out of this increment (conductor-13F A4): `PositionDeletesTable`,
//! `BaseEntriesTable` / `entries` *shape*, cross-spec `Partitioning.partitionType`
//! unification, and `readable_metrics` interior field-id order.

use crate::inspect::MetadataTableType;
use crate::scan::tests::TableTestFixture;
use crate::spec::{PrimitiveType, Schema, Type};
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
/// the Rust enum currently exposes (POSITION_DELETES is not in the enum; A4 OUT).
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
    // switch on MetadataTableType). POSITION_DELETES is OUT of this increment.
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
