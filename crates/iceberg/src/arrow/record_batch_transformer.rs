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
use std::sync::Arc;

use arrow_array::{
    Array as ArrowArray, ArrayRef, Int32Array, Int64Array, RecordBatch, RecordBatchOptions,
    RunArray,
};
use arrow_cast::cast;
use arrow_schema::{
    DataType, Field, FieldRef, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef, SchemaRef,
};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::value::{create_primitive_array_repeated, create_primitive_array_single_element};
use crate::arrow::{datum_to_arrow_type_with_ree, schema_to_arrow_schema};
use crate::metadata_columns::{
    RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER, RESERVED_FIELD_ID_POS,
    RESERVED_FIELD_ID_ROW_ID, get_metadata_field, is_row_lineage_field,
};
use crate::spec::{
    Datum, Literal, PartitionSpec, PrimitiveLiteral, Schema as IcebergSchema, Struct, Transform,
};
use crate::{Error, ErrorKind, Result};

/// Build a map of field ID to constant value (as Datum) for identity-partitioned fields.
///
/// Implements Iceberg spec "Column Projection" rule #1: use partition metadata constants
/// only for identity-transformed fields. Non-identity transforms (bucket, truncate, year, etc.)
/// store derived values in partition metadata, so source columns must be read from data files.
///
/// Example: For `bucket(4, id)`, partition metadata has `id_bucket = 2` (bucket number),
/// but the actual `id` values (100, 200, 300) are only in the data file.
///
/// Matches Java's `PartitionUtil.constantsMap()` which filters `if (field.transform().isIdentity())`.
///
/// # References
/// - Spec: https://iceberg.apache.org/spec/#column-projection
/// - Java: core/src/main/java/org/apache/iceberg/util/PartitionUtil.java:constantsMap()
fn constants_map(
    partition_spec: &PartitionSpec,
    partition_data: &Struct,
    schema: &IcebergSchema,
) -> Result<HashMap<i32, Datum>> {
    let mut constants = HashMap::new();

    for (pos, field) in partition_spec.fields().iter().enumerate() {
        // Only identity transforms should use constant values from partition metadata
        if matches!(field.transform, Transform::Identity) {
            // Get the field from schema to extract its type
            let iceberg_field = schema.field_by_id(field.source_id).ok_or(Error::new(
                ErrorKind::Unexpected,
                format!("Field {} not found in schema", field.source_id),
            ))?;

            // Ensure the field type is primitive
            let prim_type = match &*iceberg_field.field_type {
                crate::spec::Type::Primitive(prim_type) => prim_type,
                _ => {
                    return Err(Error::new(
                        ErrorKind::Unexpected,
                        format!(
                            "Partition field {} has non-primitive type {:?}",
                            field.source_id, iceberg_field.field_type
                        ),
                    ));
                }
            };

            // Get the partition value for this field.
            //
            // The tuple can be SHORTER than the spec — corrupt metadata, or a file's tuple
            // paired with a different spec. Java resolves that to null rather than failing:
            // `PartitionUtil.constantsMap` reads `partitionData.get(pos)`, and
            // `PartitionData.get(int)` opens with `if (pos >= data.length) { return null; }`
            // (iceberg-core 1.10.0, decoded from the shipped jar), after which
            // `IdentityPartitionConverters.convertConstant` maps a null value back to null.
            // Match that — warn, then fall through to the same "absent from the constants map"
            // resolution the explicit-null case uses — instead of indexing past the end and
            // aborting the scan task.
            let Some(partition_value) = partition_data.fields().get(pos) else {
                tracing::warn!(
                    source_id = field.source_id,
                    position = pos,
                    tuple_len = partition_data.fields().len(),
                    spec_id = partition_spec.spec_id(),
                    "partition tuple is shorter than its partition spec; resolving the \
                     identity-partitioned column as null (Java PartitionData.get returns null \
                     past the end of the tuple)"
                );
                continue;
            };

            // Handle both None (null) and Some(Literal::Primitive) cases
            match partition_value {
                None => {
                    // Skip null partition values - they will be resolved as null per Iceberg spec rule #4.
                    // When a partition value is null, we don't add it to the constants map,
                    // allowing downstream column resolution to handle it correctly.
                    continue;
                }
                Some(Literal::Primitive(value)) => {
                    // Build the constant from the partition value, COERCED to the source field's
                    // Iceberg type. The partition tuple can carry a literal whose in-memory variant
                    // is NARROWER than the (possibly type-promoted) column — e.g. an `Int(i32)`
                    // partition value for a column promoted to `Long`. Without coercion the array
                    // builder sees `(Int64, Int(19))` and errors ("Unsupported constant type
                    // combination: Int64 with Some(Int(19))", `test_evolved_schema`).
                    //
                    // `Datum::to(field_type)` is the canonical Iceberg coercion (the same table used
                    // throughout `arrow/`): `Int->Long`, `Int->Date`, `Long->Timestamp/Timestamptz`,
                    // with equal types passing through. This mirrors Java
                    // `IdentityPartitionConverters.convertConstant(partitionType.field(pos).type(),
                    // value)`, where the TYPE comes from the (schema-derived) partition type, not the
                    // literal's stored representation.
                    let datum = Datum::new(prim_type.clone(), value.clone())
                        .to(&iceberg_field.field_type)
                        .map_err(|e| {
                            Error::new(
                                ErrorKind::DataInvalid,
                                format!(
                                    "Failed to coerce identity-partition value for field {} to its column type {:?}",
                                    field.source_id, iceberg_field.field_type
                                ),
                            )
                            .with_source(e)
                        })?;
                    constants.insert(field.source_id, datum);
                }
                Some(literal) => {
                    return Err(Error::new(
                        ErrorKind::Unexpected,
                        format!(
                            "Partition field {} has non-primitive value: {:?}",
                            field.source_id, literal
                        ),
                    ));
                }
            }
        }
    }

    Ok(constants)
}

/// Indicates how a particular column in a processed RecordBatch should
/// be sourced.
#[derive(Debug)]
pub(crate) enum ColumnSource {
    // signifies that a column should be passed through unmodified
    // from the file's RecordBatch
    PassThrough {
        source_index: usize,
    },

    // signifies that a column from the file's RecordBatch has undergone
    // type promotion so the source column with the given index needs
    // to be promoted to the specified type
    Promote {
        target_type: DataType,
        source_index: usize,
    },

    // Signifies that a new column has been inserted before the column
    // with index `index`. (we choose "before" rather than "after" so
    // that we can use usize; if we insert after, then we need to
    // be able to store -1 here to signify that a new
    // column is to be added at the front of the column list).
    // If multiple columns need to be inserted at a given
    // location, they should all be given the same index, as the index
    // here refers to the original RecordBatch, not the interim state after
    // a preceding operation.
    Add {
        target_type: DataType,
        value: Option<PrimitiveLiteral>,
    },

    // The reserved `_pos` metadata column: a new Int64 column holding each row's absolute
    // physical ordinal in the data file (0-based). The values are threaded from the read
    // position by `process_record_batch`, not read from the file, so the read path MUST feed
    // batches in file order with no rows skipped (no Parquet `RowSelection` / row-group pruning)
    // for the positions to be correct — enforced by the callers that project `_pos`.
    RowPosition,

    // The reserved `_row_id` metadata column when the data file does NOT carry one: each row's
    // row id is `first_row_id + physical ordinal` (Java `ValueReaders$RowIdReader`, whose null
    // arm returns `firstRowId + pos`). `first_row_id` is the data file's assigned range start,
    // inherited at manifest read by `assign_first_row_ids`. Carries the same in-order, no-skip
    // decode requirement as `RowPosition`, since it is computed FROM the physical ordinal.
    RowId {
        first_row_id: i64,
    },

    // The reserved `_row_id` column when the data file DOES carry one: pass the stored value
    // through, filling NULLs with `first_row_id + physical ordinal` (Java
    // `ValueReaders$RowIdReader.read`: `idReader` first, `firstRowId + pos` only when it is null).
    RowIdFromFile {
        source_index: usize,
        first_row_id: i64,
    },

    // The reserved `_last_updated_sequence_number` column when the data file carries one: pass the
    // stored value through, filling NULLs with the file's own sequence number (Java
    // `ValueReaders$LastUpdatedSeqReader.read`). When the file does NOT carry the column the value
    // is a plain constant and takes the `Add` path instead.
    LastUpdatedSeqFromFile {
        source_index: usize,
        file_sequence_number: i64,
    },
    // The iceberg spec refers to other permissible schema evolution actions
    // (see https://iceberg.apache.org/spec/#schema-evolution):
    // renaming fields, deleting fields and reordering fields.
    // Renames only affect the schema of the RecordBatch rather than the
    // columns themselves, so a single updated cached schema can
    // be re-used and no per-column actions are required.
    // Deletion and Reorder can be achieved without needing this
    // post-processing step by using the projection mask.
}

#[derive(Debug)]
enum BatchTransform {
    // Indicates that no changes need to be performed to the RecordBatches
    // coming in from the stream and that they can be passed through
    // unmodified
    PassThrough,

    Modify {
        // Every transformed RecordBatch will have the same schema. We create the
        // target just once and cache it here. Helpfully, Arc<Schema> is needed in
        // the constructor for RecordBatch, so we don't need an expensive copy
        // each time we build a new RecordBatch
        target_schema: Arc<ArrowSchema>,

        // Indicates how each column in the target schema is derived.
        operations: Vec<ColumnSource>,
    },

    // Sometimes only the schema will need modifying, for example when
    // the column names have changed vs the file, but not the column types.
    // we can avoid a heap allocation per RecordBach in this case by retaining
    // the existing column Vec.
    ModifySchema {
        target_schema: Arc<ArrowSchema>,
    },
}

#[derive(Debug)]
enum SchemaComparison {
    Equivalent,
    NameChangesOnly,
    Different,
}

/// Builder for RecordBatchTransformer to improve ergonomics when constructing with optional parameters.
///
/// Constant fields are pre-computed for both virtual/metadata fields (like _file) and
/// identity-partitioned fields to avoid duplicate work during batch processing.
#[derive(Debug)]
pub(crate) struct RecordBatchTransformerBuilder {
    snapshot_schema: Arc<IcebergSchema>,
    projected_iceberg_field_ids: Vec<i32>,
    constant_fields: HashMap<i32, Datum>,
    /// V3 row lineage: the data file's assigned `first_row_id` and its file sequence number,
    /// threaded from the manifest entry. `None` when the table is not V3 or the file has no
    /// assigned range — projecting a row-lineage column is then an error rather than a guess.
    first_row_id: Option<i64>,
    file_sequence_number: Option<i64>,
}

impl RecordBatchTransformerBuilder {
    pub(crate) fn new(
        snapshot_schema: Arc<IcebergSchema>,
        projected_iceberg_field_ids: &[i32],
    ) -> Self {
        Self {
            snapshot_schema,
            projected_iceberg_field_ids: projected_iceberg_field_ids.to_vec(),
            constant_fields: HashMap::new(),
            first_row_id: None,
            file_sequence_number: None,
        }
    }

    /// Add a constant value for a specific field ID.
    /// This is used for virtual/metadata fields like _file that have constant values per batch.
    ///
    /// # Arguments
    /// * `field_id` - The field ID to associate with the constant
    /// * `datum` - The constant value (with type) for this field
    pub(crate) fn with_constant(mut self, field_id: i32, datum: Datum) -> Self {
        self.constant_fields.insert(field_id, datum);
        self
    }

    /// Supply the V3 row-lineage inputs for this data file: its assigned `first_row_id` (the
    /// output of `assign_first_row_ids` at manifest read) and its file sequence number.
    ///
    /// Both are `Option` because a file in a V1/V2 table — or a V3 file in a manifest with no
    /// assigned range — has neither. Projecting `_row_id` or `_last_updated_sequence_number`
    /// without the corresponding value yields an ALL-NULL column, matching Java exactly
    /// (`ValueReaders.rowIds(null, …)` and `lastUpdated` with either constant null both return
    /// `constant(null)`). It is never defaulted to zero, which would mint colliding row ids.
    pub(crate) fn with_row_lineage(
        mut self,
        first_row_id: Option<i64>,
        file_sequence_number: Option<i64>,
    ) -> Self {
        self.first_row_id = first_row_id;
        self.file_sequence_number = file_sequence_number;
        self
    }

    /// Set partition spec and data together for identifying identity-transformed partition columns.
    ///
    /// Both partition_spec and partition_data must be provided together since the spec defines
    /// which fields are identity-partitioned, and the data provides their constant values.
    /// This method computes the partition constants and merges them into constant_fields.
    pub(crate) fn with_partition(
        mut self,
        partition_spec: Arc<PartitionSpec>,
        partition_data: Struct,
    ) -> Result<Self> {
        // Compute partition constants for identity-transformed fields (already returns Datum)
        let partition_constants =
            constants_map(&partition_spec, &partition_data, &self.snapshot_schema)?;

        // Add partition constants to constant_fields
        for (field_id, datum) in partition_constants {
            self.constant_fields.insert(field_id, datum);
        }

        Ok(self)
    }

    pub(crate) fn build(self) -> RecordBatchTransformer {
        RecordBatchTransformer {
            snapshot_schema: self.snapshot_schema,
            projected_iceberg_field_ids: self.projected_iceberg_field_ids,
            constant_fields: self.constant_fields,
            first_row_id: self.first_row_id,
            file_sequence_number: self.file_sequence_number,
            batch_transform: None,
            next_row_position: 0,
        }
    }
}

/// Transforms RecordBatches from Parquet files to match the Iceberg table schema.
///
/// Handles schema evolution, column reordering, type promotion, and implements the Iceberg spec's
/// "Column Projection" rules for resolving field IDs "not present" in data files:
/// 1. Return the value from partition metadata if an Identity Transform exists
/// 2. Use schema.name-mapping.default metadata to map field id to columns without field id (applied in ArrowReader)
/// 3. Return the default value if it has a defined initial-default
/// 4. Return null in all other cases
///
/// # Field ID Resolution
///
/// Field ID resolution happens in ArrowReader before data is read (matching Java's ReadConf):
/// - If file has embedded field IDs: trust them (ParquetSchemaUtil.hasIds() = true)
/// - If file lacks IDs and name_mapping exists: apply name mapping (ParquetSchemaUtil.applyNameMapping())
/// - If file lacks IDs and no name_mapping: use position-based fallback (ParquetSchemaUtil.addFallbackIds())
///
/// By the time RecordBatchTransformer processes data, all field IDs are trustworthy.
/// This transformer only handles remaining projection rules (#1, #3, #4) for fields still "not present".
///
/// # Partition Spec and Data
///
/// **Bucket partitioning**: Distinguish identity transforms (use partition metadata constants)
/// from non-identity transforms like bucket (read from data file) to enable runtime filtering on
/// bucket-partitioned columns. For example, `bucket(4, id)` stores only the bucket number in
/// partition metadata, so actual `id` values must be read from the data file.
///
/// # References
/// - Spec: https://iceberg.apache.org/spec/#column-projection
/// - Java: parquet/src/main/java/org/apache/iceberg/parquet/ReadConf.java (field ID resolution)
/// - Java: core/src/main/java/org/apache/iceberg/util/PartitionUtil.java (partition constants)
#[derive(Debug)]
pub(crate) struct RecordBatchTransformer {
    snapshot_schema: Arc<IcebergSchema>,
    projected_iceberg_field_ids: Vec<i32>,
    // Pre-computed constant field information: field_id -> Datum
    // Includes both virtual/metadata fields (like _file) and identity-partitioned fields
    // Datum holds both the Iceberg type and the value
    constant_fields: HashMap<i32, Datum>,

    // V3 row lineage inputs for the data file being read (see
    // `RecordBatchTransformerBuilder::with_row_lineage`).
    first_row_id: Option<i64>,
    file_sequence_number: Option<i64>,

    // BatchTransform gets lazily constructed based on the schema of
    // the first RecordBatch we receive from the file
    batch_transform: Option<BatchTransform>,

    // The absolute physical row position (0-based) of the NEXT row to process, advanced by each
    // batch's row count. Sourced into the reserved `_pos` column (`ColumnSource::RowPosition`)
    // when projected. Correct only under an in-order, no-skip decode — see that variant.
    next_row_position: u64,
}

/// The shared overflow error for the `_row_id` computation.
fn row_id_overflow(first_row_id: i64, start_row_position: u64, num_rows: usize) -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        "row-lineage `_row_id` computation overflowed i64",
    )
    .with_context("first_row_id", first_row_id.to_string())
    .with_context("start_row_position", start_row_position.to_string())
    .with_context("num_rows", num_rows.to_string())
}

impl RecordBatchTransformer {
    pub(crate) fn process_record_batch(
        &mut self,
        record_batch: RecordBatch,
    ) -> Result<RecordBatch> {
        // Lazily build the transform from the first batch's schema.
        if self.batch_transform.is_none() {
            let transform = Self::generate_batch_transform(
                record_batch.schema_ref(),
                self.snapshot_schema.as_ref(),
                &self.projected_iceberg_field_ids,
                &self.constant_fields,
                self.first_row_id,
                self.file_sequence_number,
            )?;
            self.batch_transform = Some(transform);
        }

        // The absolute physical position of this batch's first row (for `_pos`), captured before
        // the immutable borrow of `batch_transform` below.
        let start_row_position = self.next_row_position;
        let row_count = record_batch.num_rows();

        let result = match self
            .batch_transform
            .as_ref()
            .expect("batch_transform was just initialized")
        {
            BatchTransform::PassThrough => record_batch,
            BatchTransform::Modify {
                target_schema,
                operations,
            } => {
                let options = RecordBatchOptions::default()
                    .with_match_field_names(false)
                    .with_row_count(Some(row_count));
                RecordBatch::try_new_with_options(
                    Arc::clone(target_schema),
                    Self::transform_columns(
                        record_batch.columns(),
                        operations,
                        row_count,
                        start_row_position,
                    )?,
                    &options,
                )?
            }
            BatchTransform::ModifySchema { target_schema } => {
                let options = RecordBatchOptions::default()
                    .with_match_field_names(false)
                    .with_row_count(Some(row_count));
                RecordBatch::try_new_with_options(
                    Arc::clone(target_schema),
                    record_batch.columns().to_vec(),
                    &options,
                )?
            }
        };

        // Advance the running position by the FULL batch (before any later delete / predicate mask
        // filters rows out) so the next batch's `_pos` continues from the correct physical ordinal.
        self.next_row_position = self.next_row_position.saturating_add(row_count as u64);

        Ok(result)
    }

    // Compare the schema of the incoming RecordBatches to the schema of
    // the Iceberg snapshot to determine what, if any, transformation
    // needs to be applied. If the schemas match, we return BatchTransform::PassThrough
    // to indicate that no changes need to be made. Otherwise, we return a
    // BatchTransform::Modify containing the target RecordBatch schema and
    // the list of `ColumnSource`s that indicate how to source each column in
    // the resulting RecordBatches.
    fn generate_batch_transform(
        source_schema: &ArrowSchemaRef,
        snapshot_schema: &IcebergSchema,
        projected_iceberg_field_ids: &[i32],
        constant_fields: &HashMap<i32, Datum>,
        first_row_id: Option<i64>,
        file_sequence_number: Option<i64>,
    ) -> Result<BatchTransform> {
        let mapped_unprojected_arrow_schema = Arc::new(schema_to_arrow_schema(snapshot_schema)?);
        let field_id_to_mapped_schema_map =
            Self::build_field_id_to_arrow_schema_map(&mapped_unprojected_arrow_schema)?;

        // Create a new arrow schema by selecting fields from mapped_unprojected,
        // in the order of the field ids in projected_iceberg_field_ids
        let fields: Result<Vec<_>> = projected_iceberg_field_ids
            .iter()
            .map(|field_id| {
                // Check if this is a constant field
                if constant_fields.contains_key(field_id) {
                    // For metadata/virtual fields (like _file), get name from metadata_columns
                    // For partition fields, get name from schema (they exist in schema)
                    if let Ok(iceberg_field) = get_metadata_field(*field_id) {
                        // This is a metadata/virtual field - convert Iceberg field to Arrow
                        let datum = constant_fields.get(field_id).ok_or(Error::new(
                            ErrorKind::Unexpected,
                            "constant field not found",
                        ))?;
                        let arrow_type = datum_to_arrow_type_with_ree(datum)?;
                        let arrow_field =
                            Field::new(&iceberg_field.name, arrow_type, !iceberg_field.required)
                                .with_metadata(HashMap::from([(
                                    PARQUET_FIELD_ID_META_KEY.to_string(),
                                    iceberg_field.id.to_string(),
                                )]));
                        Ok(Arc::new(arrow_field))
                    } else {
                        // This is an identity-partition constant field. It EXISTS in the table
                        // schema, so its declared scan-schema field (plain Arrow type, nullability
                        // and field-id metadata) is authoritative — the materialized constant must
                        // match it EXACTLY, not a Run-End-Encoded variant. Emitting REE here is the
                        // bug that broke `test_insert_into_partitioned` ("expected Utf8 but found
                        // RunEndEncoded"): the output batch schema would declare REE where the
                        // projected scan schema (and DataFusion) require a plain `Utf8`/`Int64`.
                        // Java's `PartitionUtil.constantsMap` is encoding-agnostic; REE was a
                        // Rust-only storage optimization that violated the schema contract for
                        // columns the reader must hand back in their declared physical type.
                        Ok(field_id_to_mapped_schema_map
                            .get(field_id)
                            .ok_or(Error::new(ErrorKind::Unexpected, "field not found"))?
                            .0
                            .clone())
                    }
                } else if *field_id == RESERVED_FIELD_ID_ROW_ID
                    || *field_id == RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER
                {
                    // V3 row-lineage reserved columns: absent from the table schema like `_pos`,
                    // so their Arrow field comes from the reserved-column definition (Iceberg
                    // `long` => Arrow Int64). Values are synthesized or null-filled below.
                    let meta = get_metadata_field(*field_id)?;
                    Ok(Arc::new(
                        Field::new(&meta.name, DataType::Int64, !meta.required).with_metadata(
                            HashMap::from([(
                                PARQUET_FIELD_ID_META_KEY.to_string(),
                                meta.id.to_string(),
                            )]),
                        ),
                    ))
                } else if *field_id == RESERVED_FIELD_ID_POS {
                    // `_pos` reserved metadata column: not a value constant and absent from the
                    // table schema (so the regular lookup below would fail). Build its Arrow field
                    // from the reserved-column definition (Iceberg `long` => Arrow Int64); the
                    // values are synthesized from the read position by `ColumnSource::RowPosition`.
                    let pos_meta = get_metadata_field(*field_id)?;
                    Ok(Arc::new(
                        Field::new(&pos_meta.name, DataType::Int64, !pos_meta.required)
                            .with_metadata(HashMap::from([(
                                PARQUET_FIELD_ID_META_KEY.to_string(),
                                pos_meta.id.to_string(),
                            )])),
                    ))
                } else {
                    // Regular field - use schema as-is
                    Ok(field_id_to_mapped_schema_map
                        .get(field_id)
                        .ok_or(Error::new(ErrorKind::Unexpected, "field not found"))?
                        .0
                        .clone())
                }
            })
            .collect();

        let target_schema = Arc::new(ArrowSchema::new(fields?));

        // A constant field (identity-partition or metadata/virtual) is AUTHORITATIVE and
        // must OVERRIDE any column of the same field id physically present in the data file
        // (Java: partition metadata wins over file data; `BaseParquetReaders` consults
        // `idToConstant` before the file column). If such a field is also in the source
        // file, the `PassThrough` / `ModifySchema` fast paths would hand back the FILE
        // value instead of the constant — so we must take the column-rebuilding `Modify`
        // path. (In the common Hive-migration case the partition column is NOT in the file,
        // and `compare_schemas` already reports `Different` because the field is missing.)
        let constant_overrides_file_column = !constant_fields.is_empty() && {
            let source_field_ids = Self::build_field_id_to_arrow_schema_map(source_schema)?;
            constant_fields
                .keys()
                .any(|field_id| source_field_ids.contains_key(field_id))
        };

        let comparison = if constant_overrides_file_column {
            SchemaComparison::Different
        } else {
            Self::compare_schemas(source_schema, &target_schema)
        };

        match comparison {
            SchemaComparison::Equivalent => Ok(BatchTransform::PassThrough),
            SchemaComparison::NameChangesOnly => Ok(BatchTransform::ModifySchema { target_schema }),
            SchemaComparison::Different => Ok(BatchTransform::Modify {
                operations: Self::generate_transform_operations(
                    source_schema,
                    snapshot_schema,
                    projected_iceberg_field_ids,
                    field_id_to_mapped_schema_map,
                    constant_fields,
                    first_row_id,
                    file_sequence_number,
                )?,
                target_schema,
            }),
        }
    }

    /// Compares the source and target schemas
    /// Determines if they have changed in any meaningful way:
    ///  * If they have different numbers of fields, then we need to modify
    ///    the incoming RecordBatch schema AND columns
    ///  * If they have the same number of fields, but some of them differ in
    ///    either data type or nullability, then we need to modify the
    ///    incoming RecordBatch schema AND columns
    ///  * If the schemas differ only in the column names, then we need
    ///    to modify the RecordBatch schema BUT we can keep the
    ///    original column data unmodified
    ///  * If the schemas are identical (or differ only in inconsequential
    ///    ways) then we can pass through the original RecordBatch unmodified
    fn compare_schemas(
        source_schema: &ArrowSchemaRef,
        target_schema: &ArrowSchemaRef,
    ) -> SchemaComparison {
        if source_schema.fields().len() != target_schema.fields().len() {
            return SchemaComparison::Different;
        }

        let mut names_changed = false;

        for (source_field, target_field) in source_schema
            .fields()
            .iter()
            .zip(target_schema.fields().iter())
        {
            if source_field.data_type() != target_field.data_type()
                || source_field.is_nullable() != target_field.is_nullable()
            {
                return SchemaComparison::Different;
            }

            // A positional FIELD-ID mismatch means the file's physical column order differs from
            // the projected (target) order — e.g. a name-mapped `add_files` file whose columns
            // are stored in a different order than the table schema, or a projection that reads a
            // subset out of physical order. The `NameChangesOnly` / `Equivalent` fast paths relabel
            // or pass columns THROUGH by position, which would MISLABEL them (hand back the wrong
            // column under a field's name — the wrong-column class). When both fields carry a
            // parseable field id and they differ, force the field-id-based `Modify` path
            // (`generate_transform_operations`), which sources each output column by field id.
            if let (Some(source_id), Some(target_id)) = (
                Self::field_id_of(source_field),
                Self::field_id_of(target_field),
            ) && source_id != target_id
            {
                return SchemaComparison::Different;
            }

            // A V3 ROW-LINEAGE column is NEVER a pass-through, even when the file's column matches
            // the target field exactly. Its value is stored-else-fallback PER ROW (Java
            // `ValueReaders$RowIdReader` / `$LastUpdatedSeqReader`), so a null in the stored column
            // must be replaced with `first_row_id + pos` / the file sequence number. The fast
            // paths hand the column back verbatim, nulls included — which is precisely what the
            // fallback exists to prevent. Force the field-id-based `Modify` path.
            if Self::field_id_of(source_field).is_some_and(is_row_lineage_field)
                || Self::field_id_of(target_field).is_some_and(is_row_lineage_field)
            {
                return SchemaComparison::Different;
            }

            if source_field.name() != target_field.name() {
                names_changed = true;
            }
        }

        if names_changed {
            SchemaComparison::NameChangesOnly
        } else {
            SchemaComparison::Equivalent
        }
    }

    fn generate_transform_operations(
        source_schema: &ArrowSchemaRef,
        snapshot_schema: &IcebergSchema,
        projected_iceberg_field_ids: &[i32],
        field_id_to_mapped_schema_map: HashMap<i32, (FieldRef, usize)>,
        constant_fields: &HashMap<i32, Datum>,
        first_row_id: Option<i64>,
        file_sequence_number: Option<i64>,
    ) -> Result<Vec<ColumnSource>> {
        let field_id_to_source_schema_map =
            Self::build_field_id_to_arrow_schema_map(source_schema)?;

        projected_iceberg_field_ids
            .iter()
            .map(|field_id| {
                // Check if this is a constant field (metadata/virtual or identity-partitioned)
                // Constant fields always use their pre-computed constant values, regardless of whether
                // they exist in the Parquet file. This is per Iceberg spec rule #1: partition metadata
                // is authoritative and should be preferred over file data.
                if let Some(datum) = constant_fields.get(field_id) {
                    // The column's physical Arrow type MUST equal what the target schema declares
                    // for it (built in `generate_batch_transform` above), or the produced batch
                    // fails `RecordBatch::try_new` against that schema.
                    //
                    //  * Metadata/virtual fields (`_file`, ...) have no entry in the table schema,
                    //    so the target schema declares them Run-End-Encoded — keep REE here.
                    //  * Identity-partition fields EXIST in the table schema; the target schema
                    //    declares their plain physical type (e.g. `Int64`/`Utf8`), so the constant
                    //    is a PLAIN repeated array of that type. Emitting REE for these is the bug
                    //    that broke `test_insert_into_partitioned`.
                    let target_type = if get_metadata_field(*field_id).is_ok() {
                        datum_to_arrow_type_with_ree(datum)?
                    } else {
                        field_id_to_mapped_schema_map
                            .get(field_id)
                            .ok_or(Error::new(
                                ErrorKind::Unexpected,
                                "could not find constant field in schema",
                            ))?
                            .0
                            .data_type()
                            .clone()
                    };
                    return Ok(ColumnSource::Add {
                        value: Some(datum.literal().clone()),
                        target_type,
                    });
                }

                // `_pos` reserved metadata column. Absent from the table schema, so the lookup
                // below would fail. If the file carries `_pos` (the Avro reader emits it as a
                // running counter) pass it through; otherwise (Parquet / ORC) synthesize it from
                // the read position via `RowPosition`.
                if *field_id == RESERVED_FIELD_ID_POS {
                    return Ok(match field_id_to_source_schema_map.get(field_id) {
                        Some((_, source_index)) => ColumnSource::PassThrough {
                            source_index: *source_index,
                        },
                        None => ColumnSource::RowPosition,
                    });
                }

                // V3 row-lineage reserved columns. Java `ValueReaders.fileFieldReader` dispatches
                // on whether the FILE carries the field: a present field gets a dedicated reader
                // that falls back per NULL row, an absent one gets the computed/constant value.
                if *field_id == RESERVED_FIELD_ID_ROW_ID {
                    // No assigned range => an ALL-NULL column, exactly as Java's
                    // `ValueReaders.rowIds(null, reader)`. A V1/V2 file simply has no row
                    // identity, which is a fact about the row, not a failure to read it.
                    let Some(first_row_id) = first_row_id else {
                        return Ok(ColumnSource::Add {
                            target_type: DataType::Int64,
                            value: None,
                        });
                    };
                    return Ok(match field_id_to_source_schema_map.get(field_id) {
                        Some((_, source_index)) => ColumnSource::RowIdFromFile {
                            source_index: *source_index,
                            first_row_id,
                        },
                        None => ColumnSource::RowId { first_row_id },
                    });
                }

                if *field_id == RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER {
                    // Java gates this column on BOTH inputs, not just the sequence number:
                    // `ValueReaders.lastUpdated(rowIdConst, fileSeq, reader)` is a null constant
                    // if EITHER is null. So a V1/V2 file — which HAS a
                    // sequence number but no `first_row_id` — reports NULL here, not its sequence
                    // number. Gating on the sequence number alone would fabricate a
                    // last-updated value for every row of every pre-V3 table.
                    let (Some(_), Some(file_sequence_number)) =
                        (first_row_id, file_sequence_number)
                    else {
                        return Ok(ColumnSource::Add {
                            target_type: DataType::Int64,
                            value: None,
                        });
                    };
                    return Ok(match field_id_to_source_schema_map.get(field_id) {
                        Some((_, source_index)) => ColumnSource::LastUpdatedSeqFromFile {
                            source_index: *source_index,
                            file_sequence_number,
                        },
                        // Absent from the file: a plain per-file constant.
                        None => ColumnSource::Add {
                            target_type: DataType::Int64,
                            value: Some(PrimitiveLiteral::Long(file_sequence_number)),
                        },
                    });
                }

                let (target_field, _) =
                    field_id_to_mapped_schema_map
                        .get(field_id)
                        .ok_or(Error::new(
                            ErrorKind::Unexpected,
                            "could not find field in schema",
                        ))?;
                let target_type = target_field.data_type();

                let iceberg_field = snapshot_schema.field_by_id(*field_id).ok_or(Error::new(
                    ErrorKind::Unexpected,
                    "Field not found in snapshot schema",
                ))?;

                // Iceberg spec's "Column Projection" rules (https://iceberg.apache.org/spec/#column-projection).
                // For fields "not present" in data files:
                // 1. Use partition metadata (identity transforms only)
                // 2. Use name mapping
                // 3. Use initial_default
                // 4. Return null
                //
                // Why check partition constants before Parquet field IDs (Java: BaseParquetReaders.java:299):
                // In add_files scenarios, partition columns may exist in BOTH Parquet AND partition metadata.
                // Partition metadata is authoritative - it defines which partition this file belongs to.

                // Field ID resolution now happens in ArrowReader via:
                // 1. Embedded field IDs (ParquetSchemaUtil.hasIds() = true) - trust them
                // 2. Name mapping (ParquetSchemaUtil.applyNameMapping()) - applied upfront
                // 3. Position-based fallback (ParquetSchemaUtil.addFallbackIds()) - applied upfront
                //
                // At this point, all field IDs in the source schema are trustworthy.
                // No conflict detection needed - schema resolution happened in reader.rs.
                let field_by_id = field_id_to_source_schema_map.get(field_id).map(
                    |(source_field, source_index)| {
                        if source_field.data_type().equals_datatype(target_type) {
                            ColumnSource::PassThrough {
                                source_index: *source_index,
                            }
                        } else {
                            ColumnSource::Promote {
                                target_type: target_type.clone(),
                                source_index: *source_index,
                            }
                        }
                    },
                );

                // Apply spec's fallback steps for "not present" fields.
                // Rule #1 (constants) is handled at the beginning of this function
                let column_source = if let Some(source) = field_by_id {
                    source
                } else {
                    // Rules #2, #3 and #4:
                    // Rule #2 (name mapping) was already applied in reader.rs if needed.
                    // If field_id is still not found, the column doesn't exist in the Parquet file.
                    // Fall through to rule #3 (initial_default) or rule #4 (null).
                    let default_value = iceberg_field.initial_default.as_ref().and_then(|lit| {
                        if let Literal::Primitive(prim) = lit {
                            Some(prim.clone())
                        } else {
                            None
                        }
                    });

                    ColumnSource::Add {
                        value: default_value,
                        target_type: target_type.clone(),
                    }
                };

                Ok(column_source)
            })
            .collect()
    }

    /// The Iceberg field id stamped on an Arrow field (`PARQUET:field_id` metadata), parsed as an
    /// `i32`, or `None` when the field carries no (parseable) id. Used by [`Self::compare_schemas`]
    /// to detect a physical-vs-projected reordering.
    fn field_id_of(field: &FieldRef) -> Option<i32> {
        field
            .metadata()
            .get(PARQUET_FIELD_ID_META_KEY)
            .and_then(|id| id.parse().ok())
    }

    fn build_field_id_to_arrow_schema_map(
        source_schema: &SchemaRef,
    ) -> Result<HashMap<i32, (FieldRef, usize)>> {
        let mut field_id_to_source_schema = HashMap::new();
        for (source_field_idx, source_field) in source_schema.fields.iter().enumerate() {
            // Check if field has a field ID in metadata
            if let Some(field_id_str) = source_field.metadata().get(PARQUET_FIELD_ID_META_KEY) {
                let this_field_id = field_id_str.parse().map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("field id not parseable as an i32: {e}"),
                    )
                })?;

                field_id_to_source_schema
                    .insert(this_field_id, (source_field.clone(), source_field_idx));
            }
            // If field doesn't have a field ID, skip it - name mapping will handle it
        }

        Ok(field_id_to_source_schema)
    }

    /// `first_row_id + physical ordinal` for each of `num_rows` rows starting at
    /// `start_row_position` — the fallback arm of Java `ValueReaders$RowIdReader`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::DataInvalid`] if any row's id would overflow `i64`. Java's `long`
    /// addition wraps here; the fork fails closed, because a wrapped row id aliases another live
    /// row's identity instead of surfacing as a read failure.
    fn row_ids_from_positions(
        first_row_id: i64,
        start_row_position: u64,
        num_rows: usize,
    ) -> Result<Int64Array> {
        if num_rows == 0 {
            return Ok(Int64Array::from_iter_values(std::iter::empty()));
        }
        let overflow = || row_id_overflow(first_row_id, start_row_position, num_rows);

        // Ids are monotonic in position, so checking the LAST row covers the batch. Its offset is
        // `start + num_rows - 1`, NOT `start + num_rows` — the latter is one PAST the last row and
        // would reject a batch whose final id is exactly `i64::MAX`.
        let first = first_row_id
            .checked_add(i64::try_from(start_row_position).map_err(|_| overflow())?)
            .ok_or_else(overflow)?;
        let last_offset = i64::try_from(num_rows - 1).map_err(|_| overflow())?;
        first.checked_add(last_offset).ok_or_else(overflow)?;

        // Every id in `[first, first + num_rows - 1]` is now proven representable, so the
        // per-row addition below cannot overflow.
        Ok(Int64Array::from_iter_values(
            (0..last_offset + 1).map(|offset| first + offset),
        ))
    }

    fn transform_columns(
        columns: &[Arc<dyn ArrowArray>],
        operations: &[ColumnSource],
        num_rows: usize,
        start_row_position: u64,
    ) -> Result<Vec<Arc<dyn ArrowArray>>> {
        operations
            .iter()
            .map(|op| {
                Ok(match op {
                    ColumnSource::PassThrough { source_index } => columns[*source_index].clone(),

                    ColumnSource::Promote {
                        target_type,
                        source_index,
                    } => cast(&*columns[*source_index], target_type)?,

                    ColumnSource::Add { target_type, value } => {
                        Self::create_column(target_type, value, num_rows)?
                    }

                    ColumnSource::RowPosition => {
                        // Each row's absolute physical ordinal: [start, start + num_rows).
                        let end = start_row_position.saturating_add(num_rows as u64);
                        Arc::new(Int64Array::from_iter_values(
                            (start_row_position..end).map(|p| p as i64),
                        ))
                    }

                    ColumnSource::RowId { first_row_id } => {
                        // Java `ValueReaders$RowIdReader` with no stored column: every row is
                        // `firstRowId + pos`. Computed over the PHYSICAL ordinal, so this shares
                        // `RowPosition`'s in-order, no-skip decode requirement.
                        Arc::new(Self::row_ids_from_positions(
                            *first_row_id,
                            start_row_position,
                            num_rows,
                        )?)
                    }

                    ColumnSource::RowIdFromFile {
                        source_index,
                        first_row_id,
                    } => {
                        // Java `ValueReaders$RowIdReader.read`: the stored id wins; only a NULL
                        // falls back to `firstRowId + pos`.
                        let stored = columns[*source_index].as_ref();
                        let stored =
                            stored
                                .as_any()
                                .downcast_ref::<Int64Array>()
                                .ok_or_else(|| {
                                    Error::new(
                                        ErrorKind::DataInvalid,
                                        "the data file's `_row_id` column is not an Int64 array",
                                    )
                                })?;
                        if stored.null_count() == 0 {
                            columns[*source_index].clone()
                        } else {
                            let computed = Self::row_ids_from_positions(
                                *first_row_id,
                                start_row_position,
                                num_rows,
                            )?;
                            Arc::new(Int64Array::from_iter_values((0..num_rows).map(|row| {
                                if stored.is_null(row) {
                                    computed.value(row)
                                } else {
                                    stored.value(row)
                                }
                            })))
                        }
                    }

                    ColumnSource::LastUpdatedSeqFromFile {
                        source_index,
                        file_sequence_number,
                    } => {
                        // Java `ValueReaders$LastUpdatedSeqReader.read`: the stored value wins;
                        // only a NULL falls back to the file's own sequence number.
                        let stored = columns[*source_index].as_ref();
                        let stored = stored.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                            Error::new(
                                ErrorKind::DataInvalid,
                                "the data file's `_last_updated_sequence_number` column is not an \
                                 Int64 array",
                            )
                        })?;
                        if stored.null_count() == 0 {
                            columns[*source_index].clone()
                        } else {
                            Arc::new(Int64Array::from_iter_values((0..num_rows).map(|row| {
                                if stored.is_null(row) {
                                    *file_sequence_number
                                } else {
                                    stored.value(row)
                                }
                            })))
                        }
                    }
                })
            })
            .collect()
    }

    fn create_column(
        target_type: &DataType,
        prim_lit: &Option<PrimitiveLiteral>,
        num_rows: usize,
    ) -> Result<ArrayRef> {
        // Check if this is a RunEndEncoded type (for constant fields)
        if let DataType::RunEndEncoded(_, values_field) = target_type {
            // Helper to create a Run-End Encoded array
            let create_ree_array = |values_array: ArrayRef| -> Result<ArrayRef> {
                let run_ends = if num_rows == 0 {
                    Int32Array::from(Vec::<i32>::new())
                } else {
                    Int32Array::from(vec![num_rows as i32])
                };
                Ok(Arc::new(
                    RunArray::try_new(&run_ends, &values_array).map_err(|e| {
                        Error::new(
                            ErrorKind::Unexpected,
                            "Failed to create RunArray for constant value",
                        )
                        .with_source(e)
                    })?,
                ))
            };

            // Create the values array using the helper function
            let values_array =
                create_primitive_array_single_element(values_field.data_type(), prim_lit)?;

            // Wrap in Run-End Encoding
            create_ree_array(values_array)
        } else {
            // Non-REE type (simple arrays for non-constant fields)
            create_primitive_array_repeated(target_type, prim_lit, num_rows)
        }
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{
        Array, Date32Array, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
        StringArray,
    };
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

    use crate::ErrorKind;
    use crate::arrow::record_batch_transformer::{
        RecordBatchTransformer, RecordBatchTransformerBuilder, constants_map,
    };
    use crate::metadata_columns::{
        RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER, RESERVED_FIELD_ID_ROW_ID,
    };
    use crate::spec::{Literal, NestedField, PrimitiveType, Schema, Struct, Type};

    /// Helper to extract string values from either StringArray or RunEndEncoded<StringArray>
    /// Returns empty string for null values
    fn get_string_value(array: &dyn Array, index: usize) -> String {
        if let Some(string_array) = array.as_any().downcast_ref::<StringArray>() {
            if string_array.is_null(index) {
                String::new()
            } else {
                string_array.value(index).to_string()
            }
        } else if let Some(run_array) = array
            .as_any()
            .downcast_ref::<arrow_array::RunArray<arrow_array::types::Int32Type>>()
        {
            let values = run_array.values();
            let string_values = values
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("REE values should be StringArray");
            // For REE, all rows have the same value (index 0 in the values array)
            if string_values.is_null(0) {
                String::new()
            } else {
                string_values.value(0).to_string()
            }
        } else {
            panic!("Expected StringArray or RunEndEncoded<StringArray>");
        }
    }

    /// Helper to extract int values from either Int32Array or RunEndEncoded<Int32Array>
    fn get_int_value(array: &dyn Array, index: usize) -> i32 {
        if let Some(int_array) = array.as_any().downcast_ref::<Int32Array>() {
            int_array.value(index)
        } else if let Some(run_array) = array
            .as_any()
            .downcast_ref::<arrow_array::RunArray<arrow_array::types::Int32Type>>()
        {
            let values = run_array.values();
            let int_values = values
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("REE values should be Int32Array");
            int_values.value(0)
        } else {
            panic!("Expected Int32Array or RunEndEncoded<Int32Array>");
        }
    }

    #[test]
    fn build_field_id_to_source_schema_map_works() {
        let arrow_schema = arrow_schema_already_same_as_target();

        let result =
            RecordBatchTransformer::build_field_id_to_arrow_schema_map(&arrow_schema).unwrap();

        let expected = HashMap::from_iter([
            (10, (arrow_schema.fields()[0].clone(), 0)),
            (11, (arrow_schema.fields()[1].clone(), 1)),
            (12, (arrow_schema.fields()[2].clone(), 2)),
            (14, (arrow_schema.fields()[3].clone(), 3)),
            (15, (arrow_schema.fields()[4].clone(), 4)),
        ]);

        assert!(result.eq(&expected));
    }

    #[test]
    fn processor_returns_properly_shaped_record_batch_when_no_schema_migration_required() {
        let snapshot_schema = Arc::new(iceberg_table_schema());
        let projected_iceberg_field_ids = [13, 14];

        let mut inst =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_iceberg_field_ids)
                .build();

        let result = inst
            .process_record_batch(source_record_batch_no_migration_required())
            .unwrap();

        let expected = source_record_batch_no_migration_required();

        assert_eq!(result, expected);
    }

    #[test]
    fn processor_returns_properly_shaped_record_batch_when_schema_migration_required() {
        let snapshot_schema = Arc::new(iceberg_table_schema());
        let projected_iceberg_field_ids = [10, 11, 12, 14, 15]; // a, b, c, e, f

        let mut inst =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_iceberg_field_ids)
                .build();

        let result = inst.process_record_batch(source_record_batch()).unwrap();

        let expected = expected_record_batch_migration_required();

        assert_eq!(result, expected);
    }

    #[test]
    fn schema_evolution_adds_date_column_with_nulls() {
        // Reproduces TestSelect.readAndWriteWithBranchAfterSchemaChange from iceberg-spark.
        // When reading old snapshots after adding a DATE column, the transformer must
        // populate the new column with NULL values since old files lack this field.
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(3, "date_col", Type::Primitive(PrimitiveType::Date))
                        .into(),
                ])
                .build()
                .unwrap(),
        );
        let projected_iceberg_field_ids = [1, 2, 3];

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_iceberg_field_ids)
                .build();

        let file_schema = Arc::new(ArrowSchema::new(vec![
            simple_field("id", DataType::Int32, false, "1"),
            simple_field("name", DataType::Utf8, true, "2"),
        ]));

        let file_batch = RecordBatch::try_new(file_schema, vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![
                Some("Alice"),
                Some("Bob"),
                Some("Charlie"),
            ])),
        ])
        .unwrap();

        let result = transformer.process_record_batch(file_batch).unwrap();

        assert_eq!(result.num_columns(), 3);
        assert_eq!(result.num_rows(), 3);

        let id_column = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id_column.values(), &[1, 2, 3]);

        let name_column = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(name_column.value(0), "Alice");
        assert_eq!(name_column.value(1), "Bob");
        assert_eq!(name_column.value(2), "Charlie");

        let date_column = result
            .column(2)
            .as_any()
            .downcast_ref::<Date32Array>()
            .unwrap();
        assert!(date_column.is_null(0));
        assert!(date_column.is_null(1));
        assert!(date_column.is_null(2));
    }

    #[test]
    fn row_position_metadata_column_counts_physical_ordinal_across_batches() {
        // Projecting the reserved `_pos` column must inject an Int64 column of each row's absolute
        // physical ordinal in the data file (0-based), and the counter must CONTINUE across batches
        // — it is the contract a downstream engine relies on to write position deletes.
        use arrow_array::Int64Array;

        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );
        // Project the data column plus the reserved `_pos` metadata column.
        let projected_iceberg_field_ids = [1, crate::metadata_columns::RESERVED_FIELD_ID_POS];

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_iceberg_field_ids)
                .build();

        let file_schema = Arc::new(ArrowSchema::new(vec![simple_field(
            "id",
            DataType::Int32,
            false,
            "1",
        )]));

        // First batch (3 rows) => _pos 0, 1, 2.
        let batch1 =
            RecordBatch::try_new(file_schema.clone(), vec![Arc::new(Int32Array::from(vec![
                10, 20, 30,
            ]))])
            .unwrap();
        let out1 = transformer.process_record_batch(batch1).unwrap();
        assert_eq!(out1.num_columns(), 2);
        assert_eq!(
            out1.column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[0, 1, 2]
        );

        // Second batch (2 rows) => _pos 3, 4 — the counter continues, it does NOT restart.
        let batch2 =
            RecordBatch::try_new(file_schema, vec![Arc::new(Int32Array::from(vec![40, 50]))])
                .unwrap();
        let out2 = transformer.process_record_batch(batch2).unwrap();
        assert_eq!(
            out2.column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[3, 4]
        );
    }

    #[test]
    fn schema_evolution_adds_struct_column_with_nulls() {
        // Test that when a struct column is added after data files are written,
        // the transformer can materialize the missing struct column with null values.
        // This reproduces the scenario from Iceberg 1.10.0 TestSparkReaderDeletes tests
        // where binaryData and structData columns were added to the schema.
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(
                        3,
                        "struct_col",
                        Type::Struct(crate::spec::StructType::new(vec![
                            NestedField::optional(
                                100,
                                "inner_field",
                                Type::Primitive(PrimitiveType::String),
                            )
                            .into(),
                        ])),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        let projected_iceberg_field_ids = [1, 2, 3];

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_iceberg_field_ids)
                .build();

        let file_schema = Arc::new(ArrowSchema::new(vec![
            simple_field("id", DataType::Int32, false, "1"),
            simple_field("data", DataType::Utf8, false, "2"),
        ]));

        let file_batch = RecordBatch::try_new(file_schema, vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ])
        .unwrap();

        let result = transformer.process_record_batch(file_batch).unwrap();

        assert_eq!(result.num_columns(), 3);
        assert_eq!(result.num_rows(), 3);

        let id_column = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id_column.values(), &[1, 2, 3]);

        let data_column = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(data_column.value(0), "a");
        assert_eq!(data_column.value(1), "b");
        assert_eq!(data_column.value(2), "c");

        let struct_column = result
            .column(2)
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .unwrap();
        assert!(struct_column.is_null(0));
        assert!(struct_column.is_null(1));
        assert!(struct_column.is_null(2));
    }

    pub fn source_record_batch() -> RecordBatch {
        RecordBatch::try_new(
            arrow_schema_promotion_addition_and_renaming_required(),
            vec![
                Arc::new(Int32Array::from(vec![Some(1001), Some(1002), Some(1003)])), // b
                Arc::new(Float32Array::from(vec![
                    Some(12.125),
                    Some(23.375),
                    Some(34.875),
                ])), // c
                Arc::new(Int32Array::from(vec![Some(2001), Some(2002), Some(2003)])), // d
                Arc::new(StringArray::from(vec![
                    Some("Apache"),
                    Some("Iceberg"),
                    Some("Rocks"),
                ])), // e
            ],
        )
        .unwrap()
    }

    pub fn source_record_batch_no_migration_required() -> RecordBatch {
        RecordBatch::try_new(
            arrow_schema_no_promotion_addition_or_renaming_required(),
            vec![
                Arc::new(Int32Array::from(vec![Some(2001), Some(2002), Some(2003)])), // d
                Arc::new(StringArray::from(vec![
                    Some("Apache"),
                    Some("Iceberg"),
                    Some("Rocks"),
                ])), // e
            ],
        )
        .unwrap()
    }

    pub fn expected_record_batch_migration_required() -> RecordBatch {
        RecordBatch::try_new(arrow_schema_already_same_as_target(), vec![
            Arc::new(StringArray::from(Vec::<Option<String>>::from([
                None, None, None,
            ]))), // a
            Arc::new(Int64Array::from(vec![Some(1001), Some(1002), Some(1003)])), // b
            Arc::new(Float64Array::from(vec![
                Some(12.125),
                Some(23.375),
                Some(34.875),
            ])), // c
            Arc::new(StringArray::from(vec![
                Some("Apache"),
                Some("Iceberg"),
                Some("Rocks"),
            ])), // e (d skipped by projection)
            Arc::new(StringArray::from(vec![
                Some("(╯°□°）╯"),
                Some("(╯°□°）╯"),
                Some("(╯°□°）╯"),
            ])), // f
        ])
        .unwrap()
    }

    pub fn iceberg_table_schema() -> Schema {
        Schema::builder()
            .with_schema_id(2)
            .with_fields(vec![
                NestedField::optional(10, "a", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(11, "b", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(12, "c", Type::Primitive(PrimitiveType::Double)).into(),
                NestedField::required(13, "d", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(14, "e", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(15, "f", Type::Primitive(PrimitiveType::String))
                    .with_initial_default(Literal::string("(╯°□°）╯"))
                    .into(),
            ])
            .build()
            .unwrap()
    }

    fn arrow_schema_already_same_as_target() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            simple_field("a", DataType::Utf8, true, "10"),
            simple_field("b", DataType::Int64, false, "11"),
            simple_field("c", DataType::Float64, false, "12"),
            simple_field("e", DataType::Utf8, true, "14"),
            simple_field("f", DataType::Utf8, false, "15"),
        ]))
    }

    fn arrow_schema_promotion_addition_and_renaming_required() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            simple_field("b", DataType::Int32, false, "11"),
            simple_field("c", DataType::Float32, false, "12"),
            simple_field("d", DataType::Int32, false, "13"),
            simple_field("e_old", DataType::Utf8, true, "14"),
        ]))
    }

    fn arrow_schema_no_promotion_addition_or_renaming_required() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            simple_field("d", DataType::Int32, false, "13"),
            simple_field("e", DataType::Utf8, true, "14"),
        ]))
    }

    /// Create a simple arrow field with metadata.
    fn simple_field(name: &str, ty: DataType, nullable: bool, value: &str) -> Field {
        Field::new(name, ty, nullable).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            value.to_string(),
        )]))
    }

    /// Test for add_files with Parquet files that have NO field IDs (Hive tables).
    ///
    /// This reproduces the scenario from Iceberg spec where:
    /// - Hive-style partitioned Parquet files are imported via add_files procedure
    /// - Parquet files originally DO NOT have field IDs (typical for Hive tables)
    /// - ArrowReader applies name mapping to assign correct Iceberg field IDs
    /// - Iceberg schema assigns field IDs: id (1), name (2), dept (3), subdept (4)
    /// - Partition columns (id, dept) have initial_default values
    ///
    /// Per the Iceberg spec (https://iceberg.apache.org/spec/#column-projection),
    /// this scenario requires `schema.name-mapping.default` from table metadata
    /// to correctly map Parquet columns by name to Iceberg field IDs.
    /// This mapping is now applied in ArrowReader before data is processed.
    ///
    /// Expected behavior:
    /// 1. id=1 (from initial_default) - spec rule #3
    /// 2. name="John Doe" (from Parquet with field_id=2 assigned by reader) - found by field ID
    /// 3. dept="hr" (from initial_default) - spec rule #3
    /// 4. subdept="communications" (from Parquet with field_id=4 assigned by reader) - found by field ID
    #[test]
    fn add_files_with_name_mapping_applied_in_reader() {
        // Iceberg schema after add_files: id (partition), name, dept (partition), subdept
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int))
                        .with_initial_default(Literal::int(1))
                        .into(),
                    NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(3, "dept", Type::Primitive(PrimitiveType::String))
                        .with_initial_default(Literal::string("hr"))
                        .into(),
                    NestedField::optional(4, "subdept", Type::Primitive(PrimitiveType::String))
                        .into(),
                ])
                .build()
                .unwrap(),
        );

        // Simulate ArrowReader having applied name mapping:
        // Original Parquet: name, subdept (NO field IDs)
        // After reader.rs applies name mapping: name (field_id=2), subdept (field_id=4)
        //
        // Note: Partition columns (id, dept) are NOT in the Parquet file - they're in directory paths
        use std::collections::HashMap;
        let parquet_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("name", DataType::Utf8, true).with_metadata(HashMap::from([(
                "PARQUET:field_id".to_string(),
                "2".to_string(),
            )])),
            Field::new("subdept", DataType::Utf8, true).with_metadata(HashMap::from([(
                "PARQUET:field_id".to_string(),
                "4".to_string(),
            )])),
        ]));

        let projected_field_ids = [1, 2, 3, 4]; // id, name, dept, subdept

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_field_ids).build();

        // Create a Parquet RecordBatch with data for: name="John Doe", subdept="communications"
        let parquet_batch = RecordBatch::try_new(parquet_schema, vec![
            Arc::new(StringArray::from(vec!["John Doe"])),
            Arc::new(StringArray::from(vec!["communications"])),
        ])
        .unwrap();

        let result = transformer.process_record_batch(parquet_batch).unwrap();

        // Verify the transformed RecordBatch has:
        // - id=1 (from initial_default, not from Parquet)
        // - name="John Doe" (from Parquet with correct field_id=2)
        // - dept="hr" (from initial_default, not from Parquet)
        // - subdept="communications" (from Parquet with correct field_id=4)
        assert_eq!(result.num_columns(), 4);
        assert_eq!(result.num_rows(), 1);

        let id_column = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id_column.value(0), 1);

        let name_column = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(name_column.value(0), "John Doe");

        let dept_column = result
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(dept_column.value(0), "hr");

        let subdept_column = result
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(subdept_column.value(0), "communications");
    }

    /// Test for bucket partitioning where source columns must be read from data files.
    ///
    /// This test verifies correct implementation of the Iceberg spec's "Column Projection" rules:
    /// > "Return the value from partition metadata if an **Identity Transform** exists for the field"
    ///
    /// # Why this test is critical
    ///
    /// The key insight is that partition metadata stores TRANSFORMED values, not source values:
    /// - For `bucket(4, id)`, partition metadata has `id_bucket = 2` (the bucket number)
    /// - The actual `id` column values (100, 200, 300) are ONLY in the data file
    ///
    /// If iceberg-rust incorrectly treated bucket-partitioned fields as constants, it would:
    /// 1. Replace all `id` values with the constant `2` from partition metadata
    /// 2. Break runtime filtering (e.g., `WHERE id = 100` would match no rows)
    /// 3. Return incorrect query results
    ///
    /// # What this test verifies
    ///
    /// - Bucket-partitioned fields (e.g., `bucket(4, id)`) are read from the data file
    /// - The source column `id` contains actual values (100, 200, 300), not constants
    /// - Java's `PartitionUtil.constantsMap()` behavior is correctly replicated:
    ///   ```java
    ///   if (field.transform().isIdentity()) {  // FALSE for bucket transforms
    ///       idToConstant.put(field.sourceId(), converted);
    ///   }
    ///   ```
    ///
    /// # Real-world impact
    ///
    /// This reproduces the failure scenario from Iceberg Java's TestRuntimeFiltering:
    /// - Tables partitioned by `bucket(N, col)` are common for load balancing
    /// - Queries filter on the source column: `SELECT * FROM tbl WHERE col = value`
    /// - Runtime filtering pushes predicates down to Iceberg file scans
    /// - Without this fix, the filter would match against constant partition values instead of data
    ///
    /// # References
    /// - Iceberg spec: format/spec.md "Column Projection" + "Partition Transforms"
    /// - Java impl: core/src/main/java/org/apache/iceberg/util/PartitionUtil.java
    /// - Java test: spark/src/test/java/.../TestRuntimeFiltering.java
    #[test]
    fn bucket_partitioning_reads_source_column_from_file() {
        use crate::spec::{Struct, Transform};

        // Table schema: id (data column), name (data column), id_bucket (partition column)
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        // Partition spec: bucket(4, id) - the id field is bucketed
        let partition_spec = Arc::new(
            crate::spec::PartitionSpec::builder(snapshot_schema.clone())
                .with_spec_id(0)
                .add_partition_field("id", "id_bucket", Transform::Bucket(4))
                .unwrap()
                .build()
                .unwrap(),
        );

        // Partition data: bucket value is 2
        // In Iceberg, partition data is a Struct where each field corresponds to a partition field
        let partition_data = Struct::from_iter(vec![Some(Literal::int(2))]);

        // Parquet file contains both id and name columns
        let parquet_schema = Arc::new(ArrowSchema::new(vec![
            simple_field("id", DataType::Int32, false, "1"),
            simple_field("name", DataType::Utf8, true, "2"),
        ]));

        let projected_field_ids = [1, 2]; // id, name

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_field_ids)
                .with_partition(partition_spec, partition_data)
                .expect("Failed to add partition constants")
                .build();

        // Create a Parquet RecordBatch with actual data
        // The id column MUST be read from here, not treated as a constant
        let parquet_batch = RecordBatch::try_new(parquet_schema, vec![
            Arc::new(Int32Array::from(vec![100, 200, 300])),
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
        ])
        .unwrap();

        let result = transformer.process_record_batch(parquet_batch).unwrap();

        // Verify the transformed RecordBatch correctly reads id from the file
        // (NOT as a constant from partition metadata)
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.num_rows(), 3);

        let id_column = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        // These values MUST come from the Parquet file, not be replaced by constants
        assert_eq!(id_column.value(0), 100);
        assert_eq!(id_column.value(1), 200);
        assert_eq!(id_column.value(2), 300);

        let name_column = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(name_column.value(0), "Alice");
        assert_eq!(name_column.value(1), "Bob");
        assert_eq!(name_column.value(2), "Charlie");
    }

    /// Test that identity-transformed partition fields ARE treated as constants.
    ///
    /// This is the complement to `bucket_partitioning_reads_source_column_from_file`,
    /// verifying that constants_map() correctly identifies identity-transformed
    /// partition fields per the Iceberg spec.
    ///
    /// # Spec requirement (format/spec.md "Column Projection")
    ///
    /// > "Return the value from partition metadata if an Identity Transform exists for the field
    /// >  and the partition value is present in the `partition` struct on `data_file` object
    /// >  in the manifest. This allows for metadata only migrations of Hive tables."
    ///
    /// # Why identity transforms use constants
    ///
    /// Unlike bucket/truncate/year/etc., identity transforms don't modify the value:
    /// - `identity(dept)` stores the actual `dept` value in partition metadata
    /// - Partition metadata has `dept = "engineering"` (the real value, not a hash/bucket)
    /// - This value can be used directly without reading the data file
    ///
    /// # Performance benefit
    ///
    /// For Hive migrations where partition columns aren't in data files:
    /// - Partition metadata provides the column values
    /// - No need to read from data files (metadata-only query optimization)
    /// - Common pattern: `dept=engineering/subdept=backend/file.parquet`
    ///   - `dept` and `subdept` are in directory structure, not in `file.parquet`
    ///   - Iceberg populates these from partition metadata as constants
    ///
    /// # What this test verifies
    ///
    /// - Identity-partitioned fields use constants from partition metadata
    /// - The `dept` column is populated with `"engineering"` (not read from file)
    /// - Java's `PartitionUtil.constantsMap()` behavior is matched:
    ///   ```java
    ///   if (field.transform().isIdentity()) {  // TRUE for identity
    ///       idToConstant.put(field.sourceId(), converted);
    ///   }
    ///   ```
    ///
    /// # References
    /// - Iceberg spec: format/spec.md "Column Projection"
    /// - Java impl: core/src/main/java/org/apache/iceberg/util/PartitionUtil.java
    #[test]
    fn identity_partition_uses_constant_from_metadata() {
        use crate::spec::{Struct, Transform};

        // Table schema: id (data column), dept (partition column), name (data column)
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "dept", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(3, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        // Partition spec: identity(dept) - the dept field uses identity transform
        let partition_spec = Arc::new(
            crate::spec::PartitionSpec::builder(snapshot_schema.clone())
                .with_spec_id(0)
                .add_partition_field("dept", "dept", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );

        // Partition data: dept="engineering"
        let partition_data = Struct::from_iter(vec![Some(Literal::string("engineering"))]);

        // Parquet file contains only id and name (dept is in partition path)
        let parquet_schema = Arc::new(ArrowSchema::new(vec![
            simple_field("id", DataType::Int32, false, "1"),
            simple_field("name", DataType::Utf8, true, "3"),
        ]));

        let projected_field_ids = [1, 2, 3]; // id, dept, name

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_field_ids)
                .with_partition(partition_spec, partition_data)
                .expect("Failed to add partition constants")
                .build();

        let parquet_batch = RecordBatch::try_new(parquet_schema, vec![
            Arc::new(Int32Array::from(vec![100, 200])),
            Arc::new(StringArray::from(vec!["Alice", "Bob"])),
        ])
        .unwrap();

        let result = transformer.process_record_batch(parquet_batch).unwrap();

        // Verify the dept column is populated with the constant from partition metadata
        assert_eq!(result.num_columns(), 3);
        assert_eq!(result.num_rows(), 2);

        // Use helpers to handle both simple and REE arrays
        assert_eq!(get_int_value(result.column(0).as_ref(), 0), 100);
        assert_eq!(get_int_value(result.column(0).as_ref(), 1), 200);

        // dept column comes from partition metadata (constant) - will be REE
        assert_eq!(
            get_string_value(result.column(1).as_ref(), 0),
            "engineering"
        );
        assert_eq!(
            get_string_value(result.column(1).as_ref(), 1),
            "engineering"
        );

        // name column comes from file
        assert_eq!(get_string_value(result.column(2).as_ref(), 0), "Alice");
        assert_eq!(get_string_value(result.column(2).as_ref(), 1), "Bob");
    }

    /// A partition tuple SHORTER than its partition spec must not abort the read.
    ///
    /// `constants_map` walks the spec's fields by position and reads the tuple at that position.
    /// A tuple that is too short — corrupt metadata, or a file's tuple paired with a different
    /// spec (the shape `SnapshotProducer::summary` produces on the commit path) — used to index
    /// past the end of `Struct` and panic, killing the scan task.
    ///
    /// Java resolves the same input to null: `PartitionUtil.constantsMap` reads
    /// `partitionData.get(pos)`, and `PartitionData.get(int)` opens with
    /// `if (pos >= data.length) { return null; }` (iceberg-core 1.10.0, decoded from the shipped
    /// jar), after which `IdentityPartitionConverters.convertConstant` returns null for a null
    /// value. Leaving the field out of the constants map is how this module represents "resolve
    /// as null" — the same path the explicit-null case takes.
    #[test]
    fn constants_map_tolerates_a_partition_tuple_shorter_than_the_spec() {
        use crate::spec::{Datum, PartitionSpec, Struct, Transform};

        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "dept", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(3, "region", Type::Primitive(PrimitiveType::String))
                        .into(),
                ])
                .build()
                .expect("build snapshot schema"),
        );

        // Two identity partition fields...
        let partition_spec = PartitionSpec::builder(snapshot_schema.clone())
            .with_spec_id(0)
            .add_partition_field("dept", "dept", Transform::Identity)
            .expect("add dept partition field")
            .add_partition_field("region", "region", Transform::Identity)
            .expect("add region partition field")
            .build()
            .expect("build partition spec");

        // ... but a tuple carrying only the first value.
        let partition_data = Struct::from_iter(vec![Some(Literal::string("engineering"))]);

        let constants = constants_map(&partition_spec, &partition_data, &snapshot_schema)
            .expect("a short partition tuple must not fail the read");

        assert_eq!(
            constants.get(&2),
            Some(&Datum::string("engineering")),
            "the value the tuple DOES carry must still be used as a constant"
        );
        assert!(
            !constants.contains_key(&3),
            "the missing position must resolve as null (absent from the constants map), \
             not as some other field's value: {constants:?}"
        );
    }

    /// Test bucket partitioning with renamed source column.
    ///
    /// This verifies correct behavior for TestRuntimeFiltering.testRenamedSourceColumnTable() in Iceberg Java.
    /// When a source column is renamed after partitioning is established, field-ID-based mapping
    /// must still correctly identify the column in Parquet files.
    ///
    /// # Scenario
    ///
    /// 1. Table created with `bucket(4, id)` partitioning
    /// 2. Data written to Parquet files (field_id=1, name="id")
    /// 3. Column renamed: `ALTER TABLE ... RENAME COLUMN id TO row_id`
    /// 4. Iceberg schema now has: field_id=1, name="row_id"
    /// 5. Parquet files still have: field_id=1, name="id"
    ///
    /// # Expected Behavior Per Iceberg Spec
    ///
    /// Per the Iceberg spec "Column Projection" section and Java's PartitionUtil.constantsMap():
    /// - Bucket transforms are NON-identity, so partition metadata stores bucket numbers (0-3), not source values
    /// - Source columns for non-identity transforms MUST be read from data files
    /// - Field-ID-based mapping should find the column by field_id=1 (ignoring name mismatch)
    /// - Runtime filtering on `row_id` should work correctly
    ///
    /// # What This Tests
    ///
    /// This test ensures that when FileScanTask provides partition_spec and partition_data:
    /// - constants_map() correctly identifies that bucket(4, row_id) is NOT an identity transform
    /// - The source column (field_id=1) is NOT added to constants_map
    /// - Field-ID-based mapping reads actual values from the Parquet file
    /// - Values [100, 200, 300] are read, not replaced with bucket constant 2
    ///
    /// # References
    /// - Java test: spark/src/test/java/.../TestRuntimeFiltering.java::testRenamedSourceColumnTable
    /// - Java impl: core/src/main/java/org/apache/iceberg/util/PartitionUtil.java::constantsMap()
    /// - Iceberg spec: format/spec.md "Column Projection" section
    #[test]
    fn test_bucket_partitioning_with_renamed_source_column() {
        use crate::spec::{Struct, Transform};

        // Iceberg schema after rename: row_id (was id), name
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "row_id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        // Partition spec: bucket(4, row_id) - but source_id still points to field_id=1
        let partition_spec = Arc::new(
            crate::spec::PartitionSpec::builder(snapshot_schema.clone())
                .with_spec_id(0)
                .add_partition_field("row_id", "row_id_bucket", Transform::Bucket(4))
                .unwrap()
                .build()
                .unwrap(),
        );

        // Partition data: bucket value is 2
        let partition_data = Struct::from_iter(vec![Some(Literal::int(2))]);

        // Parquet file has OLD column name "id" but SAME field_id=1
        // Field-ID-based mapping should find this despite name mismatch
        let parquet_schema = Arc::new(ArrowSchema::new(vec![
            simple_field("id", DataType::Int32, false, "1"),
            simple_field("name", DataType::Utf8, true, "2"),
        ]));

        let projected_field_ids = [1, 2]; // row_id (field_id=1), name (field_id=2)

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_field_ids)
                .with_partition(partition_spec, partition_data)
                .expect("Failed to add partition constants")
                .build();

        // Create a Parquet RecordBatch with actual data
        // Despite column rename, data should be read via field_id=1
        let parquet_batch = RecordBatch::try_new(parquet_schema, vec![
            Arc::new(Int32Array::from(vec![100, 200, 300])),
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
        ])
        .unwrap();

        let result = transformer.process_record_batch(parquet_batch).unwrap();

        // Verify the transformed RecordBatch correctly reads data despite name mismatch
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.num_rows(), 3);

        let row_id_column = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        // These values MUST come from the Parquet file via field_id=1,
        // not be replaced by the bucket constant (2)
        assert_eq!(row_id_column.value(0), 100);
        assert_eq!(row_id_column.value(1), 200);
        assert_eq!(row_id_column.value(2), 300);

        let name_column = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(name_column.value(0), "Alice");
        assert_eq!(name_column.value(1), "Bob");
        assert_eq!(name_column.value(2), "Charlie");
    }

    /// Comprehensive integration test that verifies all 4 Iceberg spec rules work correctly.
    ///
    /// Per the Iceberg spec (https://iceberg.apache.org/spec/#column-projection),
    /// "Values for field ids which are not present in a data file must be resolved
    /// according the following rules:"
    ///
    /// This test creates a scenario where each rule is exercised:
    /// - Rule #1: dept (identity-partitioned) -> constant from partition metadata
    /// - Rule #2: data (via name mapping) -> read from Parquet file by name
    /// - Rule #3: category (initial_default) -> use default value
    /// - Rule #4: notes (no default) -> return null
    ///
    /// # References
    /// - Iceberg spec: format/spec.md "Column Projection" section
    #[test]
    fn test_all_four_spec_rules() {
        use crate::spec::Transform;

        // Iceberg schema with columns designed to exercise each spec rule
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    // Field in Parquet by field ID (normal case)
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    // Rule #1: Identity-partitioned field - should use partition metadata
                    NestedField::required(2, "dept", Type::Primitive(PrimitiveType::String)).into(),
                    // Rule #2: Field resolved by name mapping (ArrowReader already applied)
                    NestedField::required(3, "data", Type::Primitive(PrimitiveType::String)).into(),
                    // Rule #3: Field with initial_default
                    NestedField::optional(4, "category", Type::Primitive(PrimitiveType::String))
                        .with_initial_default(Literal::string("default_category"))
                        .into(),
                    // Rule #4: Field with no default - should be null
                    NestedField::optional(5, "notes", Type::Primitive(PrimitiveType::String))
                        .into(),
                ])
                .build()
                .unwrap(),
        );

        // Partition spec: identity transform on dept
        let partition_spec = Arc::new(
            crate::spec::PartitionSpec::builder(snapshot_schema.clone())
                .with_spec_id(0)
                .add_partition_field("dept", "dept", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );

        // Partition data: dept="engineering"
        let partition_data = Struct::from_iter(vec![Some(Literal::string("engineering"))]);

        // Parquet schema: simulates post-ArrowReader state where name mapping already applied
        // Has id (field_id=1) and data (field_id=3, assigned by ArrowReader via name mapping)
        // Missing: dept (in partition), category (has default), notes (no default)
        let parquet_schema = Arc::new(ArrowSchema::new(vec![
            simple_field("id", DataType::Int32, false, "1"),
            simple_field("data", DataType::Utf8, false, "3"),
        ]));

        let projected_field_ids = [1, 2, 3, 4, 5]; // id, dept, data, category, notes

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_field_ids)
                .with_partition(partition_spec, partition_data)
                .expect("Failed to add partition constants")
                .build();

        let parquet_batch = RecordBatch::try_new(parquet_schema, vec![
            Arc::new(Int32Array::from(vec![100, 200])),
            Arc::new(StringArray::from(vec!["value1", "value2"])),
        ])
        .unwrap();

        let result = transformer.process_record_batch(parquet_batch).unwrap();

        assert_eq!(result.num_columns(), 5);
        assert_eq!(result.num_rows(), 2);

        // Verify each column demonstrates the correct spec rule:

        // Normal case: id from Parquet by field ID
        // Use helpers to handle both simple and REE arrays
        assert_eq!(get_int_value(result.column(0).as_ref(), 0), 100);
        assert_eq!(get_int_value(result.column(0).as_ref(), 1), 200);

        // Rule #1: dept from partition metadata (identity transform) - will be REE
        assert_eq!(
            get_string_value(result.column(1).as_ref(), 0),
            "engineering"
        );
        assert_eq!(
            get_string_value(result.column(1).as_ref(), 1),
            "engineering"
        );

        // Rule #2: data from Parquet via name mapping - will be regular array
        assert_eq!(get_string_value(result.column(2).as_ref(), 0), "value1");
        assert_eq!(get_string_value(result.column(2).as_ref(), 1), "value2");

        // Rule #3: category from initial_default - will be REE
        assert_eq!(
            get_string_value(result.column(3).as_ref(), 0),
            "default_category"
        );
        assert_eq!(
            get_string_value(result.column(3).as_ref(), 1),
            "default_category"
        );

        // Rule #4: notes is null (no default, not in Parquet, not in partition) - will be REE with null
        // For null REE arrays, we still use the helper which handles extraction
        assert_eq!(get_string_value(result.column(4).as_ref(), 0), "");
        assert_eq!(get_string_value(result.column(4).as_ref(), 1), "");
    }

    /// Test handling of null values in identity-partitioned columns.
    ///
    /// Reproduces TestPartitionValues.testNullPartitionValue() from iceberg-java, which
    /// writes records where the partition column has null values. Before the fix in #1922,
    /// this would error with "Partition field X has null value for identity transform".
    #[test]
    fn null_identity_partition_value() {
        use crate::spec::{Struct, Transform};

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        let partition_spec = Arc::new(
            crate::spec::PartitionSpec::builder(schema.clone())
                .with_spec_id(0)
                .add_partition_field("data", "data", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );

        // Partition has null value for the data column
        let partition_data = Struct::from_iter(vec![None]);

        let file_schema = Arc::new(ArrowSchema::new(vec![simple_field(
            "id",
            DataType::Int32,
            true,
            "1",
        )]));

        let projected_field_ids = [1, 2];

        let mut transformer = RecordBatchTransformerBuilder::new(schema, &projected_field_ids)
            .with_partition(partition_spec, partition_data)
            .expect("Should handle null partition values")
            .build();

        let file_batch =
            RecordBatch::try_new(file_schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])
                .unwrap();

        let result = transformer.process_record_batch(file_batch).unwrap();

        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.num_rows(), 3);

        let id_col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id_col.values(), &[1, 2, 3]);

        // Partition column with null value should produce nulls
        let data_col = result.column(1);
        assert!(data_col.is_null(0));
        assert!(data_col.is_null(1));
        assert!(data_col.is_null(2));
    }

    /// Regression pin for the REE-leak bug that triggered the 2026-06-08 revert
    /// (`test_insert_into_partitioned` — "expected Utf8 but found RunEndEncoded").
    ///
    /// An identity-partition column EXISTS in the table schema, so the transformer's
    /// output batch schema must declare it with the SAME plain physical Arrow type the
    /// scan schema declares (`Utf8` here) — NOT a `RunEndEncoded` variant. Materializing
    /// the constant as REE made the output schema disagree with the projected scan schema,
    /// and DataFusion (and `RecordBatch::try_new` against the declared schema) rejected it.
    ///
    /// This test asserts the EXACT physical type (plain `StringArray`, declared `Utf8`),
    /// not just the value — a `get_string_value`-style helper would pass under REE too.
    /// Java's `PartitionUtil.constantsMap` is encoding-agnostic; the column is handed back
    /// in its declared type.
    #[test]
    fn identity_partition_constant_is_plain_array_not_run_end_encoded() {
        use arrow_schema::DataType;

        use crate::spec::Transform;

        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "category", Type::Primitive(PrimitiveType::String))
                        .into(),
                ])
                .build()
                .unwrap(),
        );

        let partition_spec = Arc::new(
            crate::spec::PartitionSpec::builder(snapshot_schema.clone())
                .with_spec_id(0)
                .add_partition_field("category", "category", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );

        let partition_data = Struct::from_iter(vec![Some(Literal::string("electronics"))]);

        // The parquet file lacks the partition column (Hive-style migration): only `id`.
        let parquet_schema = Arc::new(ArrowSchema::new(vec![simple_field(
            "id",
            DataType::Int32,
            false,
            "1",
        )]));

        let projected_field_ids = [1, 2];

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_field_ids)
                .with_partition(partition_spec, partition_data)
                .expect("partition constants")
                .build();

        let parquet_batch =
            RecordBatch::try_new(parquet_schema, vec![Arc::new(Int32Array::from(vec![
                1, 2, 3,
            ]))])
            .unwrap();

        let result = transformer.process_record_batch(parquet_batch).unwrap();

        // The output batch's declared schema field for `category` must be PLAIN Utf8,
        // matching the table schema — never RunEndEncoded.
        assert_eq!(
            result.schema().field(1).data_type(),
            &DataType::Utf8,
            "identity-partition constant must declare its plain scan-schema type, not REE"
        );

        // And the materialized column must be a plain StringArray (not a RunArray).
        let category = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("category column must be a plain StringArray, not RunEndEncoded");
        assert_eq!(category.value(0), "electronics");
        assert_eq!(category.value(1), "electronics");
        assert_eq!(category.value(2), "electronics");
    }

    /// Regression pin for the int->long widening bug that triggered the 2026-06-08 revert
    /// (`test_evolved_schema` — "Unsupported constant type combination: Int64 with Some(Int(19))").
    ///
    /// A partition tuple can carry a literal whose in-memory variant is NARROWER than the
    /// (type-promoted) column: an `Int(i32)` partition value for a column whose Iceberg type
    /// is `Long`. The constant must be materialized into an `Int64` (`Long`) column by
    /// coercing the value to the FIELD's Iceberg type via `Datum::to` — the same coercion
    /// Java applies in `IdentityPartitionConverters.convertConstant(partitionType.field(pos)
    /// .type(), value)`, where the type comes from the schema-derived partition type, not the
    /// literal's stored representation. Without the coercion the array builder sees
    /// `(Int64, Int(19))` and errors.
    #[test]
    fn identity_partition_widens_int_literal_to_long_column() {
        use arrow_schema::DataType;

        use crate::spec::Transform;

        // Column `p` has Iceberg type Long (e.g. promoted from Int) but the partition
        // tuple still stores the value as the narrower Int(19) variant.
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "p", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .unwrap(),
        );

        let partition_spec = Arc::new(
            crate::spec::PartitionSpec::builder(snapshot_schema.clone())
                .with_spec_id(0)
                .add_partition_field("p", "p", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );

        // Partition value stored as the NARROW Int(19) variant (the revert's exact case).
        let partition_data = Struct::from_iter(vec![Some(Literal::int(19))]);

        let parquet_schema = Arc::new(ArrowSchema::new(vec![simple_field(
            "id",
            DataType::Int32,
            false,
            "1",
        )]));

        let projected_field_ids = [1, 2];

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_field_ids)
                .with_partition(partition_spec, partition_data)
                .expect("partition constants must widen Int(i32) to a Long column")
                .build();

        let parquet_batch =
            RecordBatch::try_new(parquet_schema, vec![Arc::new(Int32Array::from(vec![7, 8]))])
                .unwrap();

        let result = transformer.process_record_batch(parquet_batch).unwrap();

        // The materialized `p` column must be a plain Int64 column carrying the widened value.
        assert_eq!(result.schema().field(1).data_type(), &DataType::Int64);
        let p_col = result
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("p column must be a plain Int64Array");
        assert_eq!(p_col.value(0), 19);
        assert_eq!(p_col.value(1), 19);
    }

    /// Schema-contract pin (reviewer-added): a projection that REORDERS and SUBSETS the
    /// columns — the identity-partition constant column projected FIRST and a non-partition
    /// data column SECOND, while a third schema column is dropped — must produce an output
    /// batch whose schema EXACTLY equals the declared projection (field names, plain physical
    /// types, nullability, AND order), with the constant column a plain array (never REE) and
    /// carrying the PARTITION value (overriding the differing file value).
    ///
    /// Scope note: the reordered/subset shape already forces the column-rebuilding `Modify`
    /// path (source has 3 fields, target 2), so this test does NOT isolate the
    /// `constant_overrides_file_column` flag — the constant-vs-file OVERRIDE in isolation is
    /// pinned by the scan test `test_identity_partition_column_value_comes_from_metadata_not_file`
    /// (in-order full projection, where only the override forces `Modify`). What this pin adds
    /// is that the OVERRIDE-path schema build emits the declared plain physical type in the
    /// requested column ORDER for a reordered/subset projection.
    #[test]
    fn identity_partition_reordered_subset_projection_matches_declared_schema() {
        use arrow_schema::DataType;

        use crate::spec::Transform;

        // Schema order: id(1, Int), category(2, String, identity-partition), extra(3, Long).
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "category", Type::Primitive(PrimitiveType::String))
                        .into(),
                    NestedField::optional(3, "extra", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .unwrap(),
        );

        let partition_spec = Arc::new(
            crate::spec::PartitionSpec::builder(snapshot_schema.clone())
                .with_spec_id(0)
                .add_partition_field("category", "category", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );

        // Partition value DIFFERS from the file `category` so the override is observable.
        let partition_data = Struct::from_iter(vec![Some(Literal::string("books"))]);

        // The file physically carries ALL THREE columns (add_files style), including the
        // partition column with a DIFFERENT value ("ignored_file_value").
        let parquet_schema = Arc::new(ArrowSchema::new(vec![
            simple_field("id", DataType::Int32, false, "1"),
            simple_field("category", DataType::Utf8, false, "2"),
            simple_field("extra", DataType::Int64, true, "3"),
        ]));

        // Project REORDERED + SUBSET: category(2) FIRST, id(1) SECOND, drop extra(3).
        let projected_field_ids = [2, 1];

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_field_ids)
                .with_partition(partition_spec, partition_data)
                .expect("partition constants")
                .build();

        let parquet_batch = RecordBatch::try_new(parquet_schema, vec![
            Arc::new(Int32Array::from(vec![10, 11])),
            Arc::new(StringArray::from(vec![
                "ignored_file_value",
                "ignored_file_value",
            ])),
            Arc::new(Int64Array::from(vec![100, 200])),
        ])
        .unwrap();

        let result = transformer.process_record_batch(parquet_batch).unwrap();

        // Output schema must be exactly [category: Utf8 (non-null), id: Int32 (non-null)].
        assert_eq!(result.num_columns(), 2);
        let sch = result.schema();
        assert_eq!(sch.field(0).name(), "category");
        assert_eq!(sch.field(0).data_type(), &DataType::Utf8);
        assert!(!sch.field(0).is_nullable());
        assert_eq!(sch.field(1).name(), "id");
        assert_eq!(sch.field(1).data_type(), &DataType::Int32);
        assert!(!sch.field(1).is_nullable());

        // category is the constant (plain StringArray), OVERRIDING the file value.
        let category = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("category must be a plain StringArray, not REE");
        assert_eq!(category.value(0), "books");
        assert_eq!(category.value(1), "books");

        // id is read from the file unchanged.
        let id = result
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id.value(0), 10);
        assert_eq!(id.value(1), 11);
    }

    // ---- V3 row lineage: `_row_id` / `_last_updated_sequence_number` (F-13 V2) ----------------
    //
    // Java `ValueReaders.fileFieldReader` dispatches on whether the FILE carries the reserved
    // field, and each of the two dedicated readers then dispatches per ROW on whether the stored
    // value is null. That is a 2x2 domain per column, and both axes are pinned below:
    //
    // | | file lacks the column | file has it, no nulls | file has it, some nulls |
    // |---|---|---|---|
    // | `_row_id` | `first_row_id + pos` for every row | stored value verbatim | stored wins per row; NULL -> `first_row_id + pos` |
    // | `_last_updated_sequence_number` | the file's sequence number, constant | stored value verbatim | stored wins per row; NULL -> file sequence number |
    //
    // Plus the two refusal cells (projected without the corresponding input) and the overflow
    // door. The mixed-null cells are the discriminating ones: an implementation that ignores the
    // stored column, and one that ignores the fallback, both pass the two pure cells.

    fn row_lineage_schema() -> Arc<Schema> {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .unwrap(),
        )
    }

    /// A file batch of `id` values, optionally carrying a reserved row-lineage column.
    fn row_lineage_batch(
        ids: Vec<i64>,
        extra: Option<(i32, &str, Vec<Option<i64>>)>,
    ) -> RecordBatch {
        let mut fields = vec![simple_field("id", DataType::Int64, false, "1")];
        let mut columns: Vec<arrow_array::ArrayRef> = vec![Arc::new(Int64Array::from(ids))];
        if let Some((field_id, name, values)) = extra {
            fields.push(simple_field(
                name,
                DataType::Int64,
                true,
                &field_id.to_string(),
            ));
            columns.push(Arc::new(Int64Array::from(values)));
        }
        RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns).unwrap()
    }

    fn int64_col(batch: &RecordBatch, index: usize) -> Vec<Option<i64>> {
        let array = batch
            .column(index)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 column");
        (0..array.len())
            .map(|row| {
                if array.is_null(row) {
                    None
                } else {
                    Some(array.value(row))
                }
            })
            .collect()
    }

    #[test]
    fn row_id_is_computed_from_first_row_id_and_position_when_absent_from_the_file() {
        let projected = [1, RESERVED_FIELD_ID_ROW_ID];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(Some(100), Some(7))
            .build();

        let first = transformer
            .process_record_batch(row_lineage_batch(vec![1, 2, 3], None))
            .unwrap();
        assert_eq!(int64_col(&first, 1), vec![Some(100), Some(101), Some(102)]);

        // The position counter CONTINUES across batches — it does not restart, or every batch
        // after the first would repeat the same row ids.
        let second = transformer
            .process_record_batch(row_lineage_batch(vec![4, 5], None))
            .unwrap();
        assert_eq!(int64_col(&second, 1), vec![Some(103), Some(104)]);
    }

    #[test]
    fn row_id_stored_in_the_file_wins_over_the_computed_value() {
        let projected = [1, RESERVED_FIELD_ID_ROW_ID];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(Some(100), Some(7))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(
                vec![1, 2, 3],
                Some((RESERVED_FIELD_ID_ROW_ID, "_row_id", vec![
                    Some(900),
                    Some(901),
                    Some(902),
                ])),
            ))
            .unwrap();
        assert_eq!(
            int64_col(&batch, 1),
            vec![Some(900), Some(901), Some(902)],
            "a file that carries row ids keeps them — they are the rows' durable identity, and \
             recomputing would renumber rows that were carried through a rewrite"
        );
    }

    /// The discriminating cell: stored and computed values INTERLEAVE within one batch.
    #[test]
    fn a_null_row_id_in_the_file_falls_back_to_first_row_id_plus_position() {
        let projected = [1, RESERVED_FIELD_ID_ROW_ID];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(Some(100), Some(7))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(
                vec![1, 2, 3, 4],
                Some((RESERVED_FIELD_ID_ROW_ID, "_row_id", vec![
                    Some(900),
                    None,
                    Some(902),
                    None,
                ])),
            ))
            .unwrap();
        assert_eq!(
            int64_col(&batch, 1),
            vec![Some(900), Some(101), Some(902), Some(103)],
            "each NULL takes `first_row_id + ITS OWN position` (101 at row 1, 103 at row 3) — not \
             a running count of the nulls, and not the whole column recomputed"
        );
    }

    /// Java returns an ALL-NULL column here, NOT an error: `ValueReaders.rowIds(null, reader)`
    /// is a null constant (see `task/f13-v3-row-lineage-ledger.md`).
    /// A V1/V2 file simply has no row identity — that is a fact about the rows, not a failure to
    /// read them, and erroring would make `SELECT _row_id` unusable on any mixed-version table.
    ///
    /// This REPLACES `projecting_row_id_without_an_assigned_range_is_refused`, which pinned an
    /// invented refusal (bundle-Critic F-C).
    #[test]
    fn projecting_row_id_without_an_assigned_range_yields_nulls() {
        let projected = [1, RESERVED_FIELD_ID_ROW_ID];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(None, Some(7))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(vec![1, 2], None))
            .expect("no assigned range is not an error");
        assert_eq!(int64_col(&batch, 1), vec![None, None]);
    }

    #[test]
    fn last_updated_sequence_number_is_the_files_own_when_absent_from_the_file() {
        let projected = [1, RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(Some(100), Some(7))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(vec![1, 2], None))
            .unwrap();
        assert_eq!(
            int64_col(&batch, 1),
            vec![Some(7), Some(7)],
            "a constant per file — the file's own sequence number, NOT the row position"
        );
    }

    /// The discriminating cell for the sequence column.
    #[test]
    fn a_null_last_updated_sequence_number_falls_back_to_the_files_own() {
        let projected = [1, RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(Some(100), Some(7))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(
                vec![1, 2, 3],
                Some((
                    RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER,
                    "_last_updated_sequence_number",
                    vec![Some(3), None, Some(5)],
                )),
            ))
            .unwrap();
        assert_eq!(
            int64_col(&batch, 1),
            vec![Some(3), Some(7), Some(5)],
            "the stored per-row value wins; only the NULL takes the file's sequence number"
        );
    }

    /// The DISCRIMINATING cell for the absent-range arm. Java's `ValueReaders.rowIds(null, …)` is
    /// a null constant, which means the stored column is **ignored**, not preferred — the arm is
    /// chosen before the file is ever consulted. The sibling test feeds a batch with NO stored
    /// column, so it cannot tell "discard the stored value" from "there was nothing to discard":
    /// reordering the match to prefer a stored column whenever one exists passed the whole suite.
    ///
    /// Reachable in practice — a V3 file carrying `_row_id` read under a manifest with no
    /// assigned range, which is every DELETE manifest.
    #[test]
    fn a_stored_row_id_is_discarded_when_there_is_no_assigned_range() {
        let projected = [1, RESERVED_FIELD_ID_ROW_ID];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(None, Some(7))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(
                vec![1, 2, 3],
                Some((RESERVED_FIELD_ID_ROW_ID, "_row_id", vec![
                    Some(900),
                    Some(901),
                    Some(902),
                ])),
            ))
            .expect("no assigned range is not an error");
        assert_eq!(
            int64_col(&batch, 1),
            vec![None, None, None],
            "no assigned range means NO row identity — the stored column is discarded, not \
             preferred. Java reaches `constant(null)` without consulting the file at all."
        );
    }

    /// The same cell for the sequence column: a stored value is discarded when the gate fails.
    #[test]
    fn a_stored_last_updated_sequence_number_is_discarded_without_a_first_row_id() {
        let projected = [1, RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(None, Some(5))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(
                vec![1, 2],
                Some((
                    RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER,
                    "_last_updated_sequence_number",
                    vec![Some(31), Some(33)],
                )),
            ))
            .expect("a missing first_row_id is not an error");
        assert_eq!(
            int64_col(&batch, 1),
            vec![None, None],
            "Java gates on BOTH inputs BEFORE reading the column, so a stored value is discarded"
        );
    }

    /// Java gates `_last_updated_sequence_number` on BOTH inputs:
    /// `ValueReaders.lastUpdated(Long rowIdConst, Long fileSeq, reader)` returns `constant(null)`
    /// if EITHER is null. So a V1/V2 file —
    /// which HAS a sequence number but no `first_row_id` — reports NULL, not its sequence number.
    ///
    /// This is the cell that matters most in practice: gating on the sequence number alone
    /// fabricates a last-updated value for every row of every pre-V3 table, and does it silently.
    /// (bundle-Critic F-B.)
    #[test]
    fn last_updated_sequence_number_is_null_without_a_first_row_id() {
        let projected = [1, RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(None, Some(5))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(vec![1, 2], None))
            .expect("a missing first_row_id is not an error");
        assert_eq!(
            int64_col(&batch, 1),
            vec![None, None],
            "NULL, not the file's sequence number — Java gates on BOTH inputs"
        );
    }

    /// The other half of the same gate: no file sequence number is also NULL.
    #[test]
    fn last_updated_sequence_number_is_null_without_a_file_sequence_number() {
        let projected = [1, RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(Some(100), None)
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(vec![1], None))
            .expect("a missing file sequence number is not an error");
        assert_eq!(int64_col(&batch, 1), vec![None]);
    }

    /// The boundary the overflow check must NOT reject: a batch whose LAST id is exactly
    /// `i64::MAX` is representable and must succeed. Checking `start + num_rows` (one PAST the
    /// last row) instead of `start + num_rows - 1` passes every other test in this module and
    /// fails only here.
    #[test]
    fn a_row_id_of_exactly_i64_max_is_allowed() {
        let projected = [1, RESERVED_FIELD_ID_ROW_ID];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(Some(i64::MAX - 1), Some(7))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(vec![1, 2], None))
            .expect("the last id is exactly i64::MAX, which is representable");
        assert_eq!(int64_col(&batch, 1), vec![
            Some(i64::MAX - 1),
            Some(i64::MAX)
        ]);
    }

    /// Fail closed instead of wrapping into a negative row id (Java's `long` addition wraps).
    #[test]
    fn a_row_id_computation_that_overflows_i64_is_refused() {
        let projected = [1, RESERVED_FIELD_ID_ROW_ID];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(Some(i64::MAX), Some(7))
            .build();

        let error = transformer
            .process_record_batch(row_lineage_batch(vec![1, 2], None))
            .expect_err("i64::MAX + 2 has no representable row id");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains("overflowed i64"),
            "got: {}",
            error.message()
        );
    }
}
