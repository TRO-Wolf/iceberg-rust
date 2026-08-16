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

//! `position_deletes` metadata table — SCHEMA ONLY (scan is not ported).
//!
//! Mirrors Java `PositionDeletesTable.calculateSchema` (`core/.../PositionDeletesTable.java`):
//! the fixed metadata columns (`MetadataColumns.DELETE_FILE_PATH` / `DELETE_FILE_POS` /
//! `DELETE_FILE_ROW_*` / `PARTITION_COLUMN_ID` / `SPEC_ID_COLUMN_ID` / `FILE_PATH_COLUMN_ID`,
//! plus the v3 DV columns `CONTENT_OFFSET_COLUMN_ID` / `CONTENT_SIZE_IN_BYTES_COLUMN_ID`), the
//! partition-field id reassignment (smallest positive ids not used by ANY table schema nor the
//! metadata columns), and the empty-partition `TypeUtil.selectNot(PARTITION_COLUMN_ID)` drop.
//!
//! Two deliberate bounds, both tracked in GAP_MATRIX R142:
//! - **Scan is refused loud** (`FeatureUnsupported`): Java backs this table with a dedicated
//!   `PositionDeletesBatchScan` over delete manifests; that port is a separate campaign.
//!   Increment D does **not** un-refuse the scan.
//! - **Partition type** is [`TableMetadata::unified_partition_type`] (Java
//!   `Partitioning.partitionType`) — increment D. That automatically corrects the
//!   remapped-child-id set and the empty-partition drop predicate.

use crate::scan::ArrowRecordBatchStream;
use crate::spec::{NestedField, NestedFieldRef, PrimitiveType, Schema, StructType, Type};
use crate::table::Table;
use crate::{Error, ErrorKind, Result};

/// Java `Integer.MAX_VALUE` — the anchor for `MetadataColumns` reserved field ids.
const JAVA_INT_MAX: i32 = i32::MAX;

/// `MetadataColumns.DELETE_FILE_PATH` (`file_path`, required string).
const DELETE_FILE_PATH_ID: i32 = JAVA_INT_MAX - 101;
/// `MetadataColumns.DELETE_FILE_POS` (`pos`, required long).
const DELETE_FILE_POS_ID: i32 = JAVA_INT_MAX - 102;
/// `MetadataColumns.DELETE_FILE_ROW_FIELD_ID` (`row`, optional struct of the table schema).
const DELETE_FILE_ROW_FIELD_ID: i32 = JAVA_INT_MAX - 103;
/// `MetadataColumns.PARTITION_COLUMN_ID` (`partition`, required struct; dropped when empty).
const PARTITION_COLUMN_ID: i32 = JAVA_INT_MAX - 5;
/// `MetadataColumns.SPEC_ID_COLUMN_ID` (`spec_id`, required int).
const SPEC_ID_COLUMN_ID: i32 = JAVA_INT_MAX - 4;
/// `MetadataColumns.FILE_PATH_COLUMN_ID` (`delete_file_path`, required string).
const FILE_PATH_COLUMN_ID: i32 = JAVA_INT_MAX - 1;
/// `MetadataColumns.CONTENT_OFFSET_COLUMN_ID` (`content_offset`, optional long, v3+).
const CONTENT_OFFSET_COLUMN_ID: i32 = JAVA_INT_MAX - 6;
/// `MetadataColumns.CONTENT_SIZE_IN_BYTES_COLUMN_ID` (`content_size_in_bytes`, optional long, v3+).
const CONTENT_SIZE_IN_BYTES_COLUMN_ID: i32 = JAVA_INT_MAX - 7;

/// PositionDeletes table (schema only — see the module doc for the scan bound).
pub struct PositionDeletesTable<'a> {
    table: &'a Table,
    /// Java `Partitioning.partitionType(table)` — stored so [`Self::schema`] stays infallible.
    unified_partition_type: StructType,
}

impl<'a> PositionDeletesTable<'a> {
    /// Fallible constructor: resolves the unified partition type up front.
    ///
    /// The DataFusion `IcebergMetadataTableProvider::try_new` is the public
    /// fallible seam (A5).
    ///
    /// # Errors
    ///
    /// Propagates [`crate::spec::TableMetadata::unified_partition_type`].
    pub fn try_new(table: &'a Table) -> Result<Self> {
        let unified_partition_type = table.metadata().unified_partition_type()?;
        Ok(Self {
            table,
            unified_partition_type,
        })
    }

    /// Create a new PositionDeletes table instance.
    ///
    /// Signature stays infallible (A5). On a G1/G2 table this falls back to
    /// [`TableMetadata::default_partition_type`] so `inspect().position_deletes().schema()`
    /// cannot panic; [`Self::try_new`] is the loud refuse path.
    pub fn new(table: &'a Table) -> Self {
        match Self::try_new(table) {
            Ok(this) => this,
            Err(_) => Self {
                table,
                unified_partition_type: table.metadata().default_partition_type().clone(),
            },
        }
    }

    /// Returns the iceberg schema of the `position_deletes` table.
    ///
    /// Transcribed from Java `PositionDeletesTable.calculateSchema`: column list order is
    /// `file_path`, `pos`, `row`, `partition`, `spec_id`, `delete_file_path` (+ `content_offset`,
    /// `content_size_in_bytes` on format v3+); partition child ids are reassigned to the smallest
    /// positive ids unused by any table schema or metadata column; an EMPTY *unified* partition
    /// type drops the `partition` column entirely (`TypeUtil.selectNot(PARTITION_COLUMN_ID)`).
    pub fn schema(&self) -> Schema {
        let metadata = self.table.metadata();
        let partition_type = &self.unified_partition_type;
        let table_struct = metadata.current_schema().as_struct().clone();
        let format_version = metadata.format_version() as u8;

        let mut fields: Vec<NestedFieldRef> = vec![
            NestedField::required(
                DELETE_FILE_PATH_ID,
                "file_path",
                Type::Primitive(PrimitiveType::String),
            )
            .into(),
            NestedField::required(
                DELETE_FILE_POS_ID,
                "pos",
                Type::Primitive(PrimitiveType::Long),
            )
            .into(),
            NestedField::optional(DELETE_FILE_ROW_FIELD_ID, "row", Type::Struct(table_struct))
                .into(),
        ];

        // Java: partition child ids are reassigned before the schema is built (the remap
        // callback passed to `new Schema(...)` touches ONLY `idsToReassign`).
        if !partition_type.fields().is_empty() {
            let remapped = remap_partition_field_ids(self.table, partition_type);
            fields.push(
                NestedField::required(PARTITION_COLUMN_ID, "partition", Type::Struct(remapped))
                    .into(),
            );
        }

        fields.push(
            NestedField::required(
                SPEC_ID_COLUMN_ID,
                "spec_id",
                Type::Primitive(PrimitiveType::Int),
            )
            .into(),
        );
        fields.push(
            NestedField::required(
                FILE_PATH_COLUMN_ID,
                "delete_file_path",
                Type::Primitive(PrimitiveType::String),
            )
            .into(),
        );

        if format_version >= 3 {
            fields.push(
                NestedField::optional(
                    CONTENT_OFFSET_COLUMN_ID,
                    "content_offset",
                    Type::Primitive(PrimitiveType::Long),
                )
                .into(),
            );
            fields.push(
                NestedField::optional(
                    CONTENT_SIZE_IN_BYTES_COLUMN_ID,
                    "content_size_in_bytes",
                    Type::Primitive(PrimitiveType::Long),
                )
                .into(),
            );
        }

        Schema::builder()
            .with_fields(fields)
            .build()
            .expect("position_deletes metadata table schema is structurally valid")
    }

    /// Scanning `position_deletes` is not ported — refused loud, never a wrong/empty result.
    ///
    /// Java backs this table with `PositionDeletesTable.PositionDeletesBatchScan` (a dedicated
    /// scan over DELETE manifests; `newScan()` on the base table type is unsupported there too).
    pub fn scan(&self) -> Result<ArrowRecordBatchStream> {
        Err(Error::new(
            ErrorKind::FeatureUnsupported,
            "position_deletes metadata table scan is not yet ported: only its schema is \
             available (Java PositionDeletesTable.PositionDeletesBatchScan has no Rust analogue)",
        ))
    }
}

/// Java `calculateSchema` id reassignment: partition child ids move to the smallest positive
/// ids not used by any table schema (all schema versions) nor by the metadata columns / the
/// embedded `row` struct.
fn remap_partition_field_ids(table: &Table, partition_type: &StructType) -> StructType {
    let mut used: std::collections::HashSet<i32> = [
        DELETE_FILE_PATH_ID,
        DELETE_FILE_POS_ID,
        DELETE_FILE_ROW_FIELD_ID,
        PARTITION_COLUMN_ID,
        SPEC_ID_COLUMN_ID,
        FILE_PATH_COLUMN_ID,
        CONTENT_OFFSET_COLUMN_ID,
        CONTENT_SIZE_IN_BYTES_COLUMN_ID,
    ]
    .into();
    for schema in table.metadata().schemas_iter() {
        collect_struct_field_ids(schema.as_struct(), &mut used);
    }

    let mut next_id = 0_i32;
    let mut fresh = || {
        loop {
            next_id += 1;
            if !used.contains(&next_id) {
                return next_id;
            }
        }
    };

    StructType::new(
        partition_type
            .fields()
            .iter()
            .map(|field| {
                NestedField::new(
                    fresh(),
                    field.name.clone(),
                    (*field.field_type).clone(),
                    field.required,
                )
                .into()
            })
            .collect(),
    )
}

/// Recursively collect every field id reachable in `struct_type` (Java `TypeUtil.indexById`).
fn collect_struct_field_ids(struct_type: &StructType, used: &mut std::collections::HashSet<i32>) {
    for field in struct_type.fields() {
        used.insert(field.id);
        if let Type::Struct(nested) = field.field_type.as_ref() {
            collect_struct_field_ids(nested, used);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inspect::MetadataTableType;
    use crate::scan::tests::TableTestFixture;

    fn assert_field(schema: &Schema, index: usize, id: i32, name: &str, required: bool) {
        let field = &schema.as_struct().fields()[index];
        assert_eq!(field.id, id, "field[{index}] id ({name})");
        assert_eq!(field.name, name, "field[{index}] name");
        assert_eq!(field.required, required, "field[{index}] ({name}) required");
    }

    /// Cite: Java `PositionDeletesTable.calculateSchema` — fixed columns in builder order,
    /// v2 table (no DV columns), partition child ids reassigned off the table-schema id space.
    #[test]
    fn partitioned_schema_matches_java_calculate_schema() {
        let fixture = TableTestFixture::new();
        let schema = fixture.table.inspect().position_deletes().schema();
        let fields = schema.as_struct().fields();
        assert_eq!(fields.len(), 6, "v2 partitioned column count");
        assert_field(&schema, 0, JAVA_INT_MAX - 101, "file_path", true);
        assert_field(&schema, 1, JAVA_INT_MAX - 102, "pos", true);
        assert_field(&schema, 2, JAVA_INT_MAX - 103, "row", false);
        assert_field(&schema, 3, JAVA_INT_MAX - 5, "partition", true);
        assert_field(&schema, 4, JAVA_INT_MAX - 4, "spec_id", true);
        assert_field(&schema, 5, JAVA_INT_MAX - 1, "delete_file_path", true);

        // Cite: `row` embeds the table schema struct as-is.
        let row = &fields[2];
        let Type::Struct(row_struct) = row.field_type.as_ref() else {
            panic!("row must be a struct");
        };
        assert_eq!(
            row_struct,
            fixture.table.metadata().current_schema().as_struct(),
            "row struct is the current table schema"
        );

        // Cite: the id-reassignment lambda — partition child ids move to the smallest positive
        // ids not used by ANY table schema nor the metadata columns.
        let Type::Struct(partition_struct) = fields[3].field_type.as_ref() else {
            panic!("partition must be a struct");
        };
        let mut used = std::collections::HashSet::new();
        for s in fixture.table.metadata().schemas_iter() {
            collect_struct_field_ids(s.as_struct(), &mut used);
        }
        for child in partition_struct.fields() {
            assert!(child.id > 0, "reassigned partition child id is positive");
            assert!(
                !used.contains(&child.id),
                "reassigned partition child id {} must not collide with any table schema id",
                child.id
            );
        }
    }

    /// Cite: Java `calculateSchema` tail — empty partition type returns
    /// `TypeUtil.selectNot(result, PARTITION_COLUMN_ID)` (drop, not empty struct).
    #[test]
    fn unpartitioned_schema_drops_partition_column() {
        let fixture = TableTestFixture::new_unpartitioned();
        let schema = fixture.table.inspect().position_deletes().schema();
        let fields = schema.as_struct().fields();
        assert_eq!(fields.len(), 5, "v2 unpartitioned column count");
        assert!(
            schema.field_by_id(PARTITION_COLUMN_ID).is_none(),
            "unpartitioned position_deletes must drop field {PARTITION_COLUMN_ID}"
        );
        assert_field(&schema, 0, JAVA_INT_MAX - 101, "file_path", true);
        assert_field(&schema, 3, JAVA_INT_MAX - 4, "spec_id", true);
        assert_field(&schema, 4, JAVA_INT_MAX - 1, "delete_file_path", true);
    }

    /// The FB-2 bound: schema-only. The scan refusal must be loud and name the table.
    #[test]
    fn scan_is_refused_loud() {
        let fixture = TableTestFixture::new();
        let err = fixture
            .table
            .inspect()
            .position_deletes()
            .scan()
            .err()
            .expect("position_deletes scan must refuse");
        assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
        let message = err.to_string();
        assert!(
            message.contains("position_deletes") && message.contains("not yet ported"),
            "refusal must name the table and the bound: {message}"
        );
    }

    /// Cite: Java `MetadataTableType.from` — vocabulary + `$`-suffix resolution key.
    #[test]
    fn metadata_table_type_round_trips() {
        let ty = MetadataTableType::try_from("position_deletes").expect("vocabulary");
        assert_eq!(ty.as_str(), "position_deletes");
        assert!(
            MetadataTableType::all_types().any(|t| t.as_str() == "position_deletes"),
            "all_types must include position_deletes"
        );
    }

    /// Increment D: unified type has two children under widening evolution,
    /// so the remapped `partition` struct has two fields (default spec would
    /// also be two here; the next test is the discriminating one).
    #[test]
    fn widening_schema_partition_has_two_remapped_children() {
        let fixture = TableTestFixture::new_with_widening_spec_evolution();
        let schema = fixture.table.inspect().position_deletes().schema();
        let partition = schema
            .field_by_id(PARTITION_COLUMN_ID)
            .expect("widening table keeps partition");
        let Type::Struct(partition_struct) = partition.field_type.as_ref() else {
            panic!("partition must be a struct");
        };
        assert_eq!(
            partition_struct.fields().len(),
            2,
            "unified type {{x, y_bucket_8}} remaps two children"
        );
        assert_eq!(partition_struct.fields()[0].name, "x");
        assert_eq!(partition_struct.fields()[1].name, "y_bucket_8");
    }

    /// Discriminator vs default_partition_type: current spec is unpartitioned
    /// but unified type is `{x}`, so `partition` stays (Java empty-drop fires
    /// on `Partitioning.partitionType`, not the default spec).
    #[test]
    fn evolved_to_unpartitioned_keeps_partition_column() {
        let fixture = TableTestFixture::new_evolved_to_unpartitioned();
        let schema = fixture.table.inspect().position_deletes().schema();
        assert!(
            schema.field_by_id(PARTITION_COLUMN_ID).is_some(),
            "historical spec 0 keeps the partition column"
        );
        let partition = schema.field_by_id(PARTITION_COLUMN_ID).unwrap();
        let Type::Struct(partition_struct) = partition.field_type.as_ref() else {
            panic!("partition must be a struct");
        };
        assert_eq!(partition_struct.fields().len(), 1);
        assert_eq!(partition_struct.fields()[0].name, "x");
    }

    /// Increment D does not un-refuse the scan (FB-2 bound).
    #[test]
    fn scan_still_refused_after_unified_schema() {
        let fixture = TableTestFixture::new_with_widening_spec_evolution();
        let err = fixture
            .table
            .inspect()
            .position_deletes()
            .scan()
            .err()
            .expect("scan stays refused");
        assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
    }

    /// `try_new` is the G2 refuse path. `new_with_two_identity_specs` is a
    /// Java-invalid unifier input (field id 1000 reused for two sources) and
    /// is used here ONLY as a refusal pin, never as a successful unifier input.
    #[test]
    fn try_new_refuses_conflicting_field_ids() {
        let fixture = TableTestFixture::new_with_two_identity_specs();
        let error = match PositionDeletesTable::try_new(&fixture.table) {
            Ok(_) => panic!("G2: conflicting field ids must refuse try_new"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().starts_with("Conflicting partition fields"),
            "message was: {}",
            error.message()
        );
    }
}
