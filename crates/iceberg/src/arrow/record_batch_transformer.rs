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

/// Builds the field id to constant map for identity-partitioned fields. Java
/// `PartitionUtil.constantsMap`.
///
/// # Notes
///
/// Only identity transforms qualify. A bucket, truncate, or year transform stores a DERIVED value
/// in partition metadata, so the reader must take the source column from the data file.
fn constants_map(
    partition_spec: &PartitionSpec,
    partition_data: &Struct,
    schema: &IcebergSchema,
) -> Result<HashMap<i32, Datum>> {
    let mut constants = HashMap::new();

    for (pos, field) in partition_spec.fields().iter().enumerate() {
        if matches!(field.transform, Transform::Identity) {
            let iceberg_field = schema.field_by_id(field.source_id).ok_or(Error::new(
                ErrorKind::Unexpected,
                format!("Field {} not found in schema", field.source_id),
            ))?;

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

            // The tuple can be SHORTER than the spec, from corrupt metadata or a tuple paired
            // with a different spec. Java `PartitionData.get` returns null past the end, so match
            // that. Warn and leave the field out of the map, which resolves it as null. Indexing
            // past the end would abort the scan task.
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

            match partition_value {
                None => {
                    // A field absent from the constants map resolves as null downstream.
                    continue;
                }
                Some(Literal::Primitive(value)) => {
                    // Coerce the value to the FIELD's Iceberg type, like Java
                    // `IdentityPartitionConverters.convertConstant`. A partition tuple can carry a
                    // literal narrower than a type-promoted column, such as `Int(i32)` for a
                    // column promoted to `Long`. Without the coercion the array builder sees
                    // `(Int64, Int(19))` and errors.
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

/// How a column in a processed RecordBatch is sourced.
#[derive(Debug)]
pub(crate) enum ColumnSource {
    // Pass the file's column through unmodified.
    PassThrough {
        source_index: usize,
    },

    /// Promote the file's column to the type the table schema now declares.
    Promote {
        target_type: DataType,
        source_index: usize,
    },

    /// Insert a new constant column that the file does not carry.
    Add {
        target_type: DataType,
        value: Option<PrimitiveLiteral>,
    },

    /// The reserved `_pos` column: each row's 0-based physical ordinal in the data file.
    ///
    /// `process_record_batch` threads the value from the read position. The read path MUST
    /// therefore feed batches in file order with no rows skipped, so no Parquet `RowSelection`
    /// and no row-group pruning. The callers that project `_pos` enforce that.
    RowPosition,

    // The reserved `_row_id` column when the file does NOT carry one. Java
    // `ValueReaders$RowIdReader`. Computed from the ordinal, so it shares `RowPosition`'s
    // in-order, no-skip decode requirement.
    RowId {
        first_row_id: i64,
    },

    /// The reserved `_row_id` column when the file DOES carry one. The stored value wins, and a
    /// NULL falls back to `first_row_id + ordinal`. Java `ValueReaders$RowIdReader.read`.
    RowIdFromFile {
        source_index: usize,
        first_row_id: i64,
    },

    /// The reserved `_last_updated_sequence_number` column when the file carries one. The stored
    /// value wins, and a NULL falls back to the file's sequence number.
    LastUpdatedSeqFromFile {
        source_index: usize,
        file_sequence_number: i64,
    },
    // A rename, a delete, and a reorder need no variant here. A rename only changes the batch
    // schema, and the projection mask already handles a delete and a reorder.
}

#[derive(Debug)]
enum BatchTransform {
    /// The incoming batches already match. Pass them through.
    PassThrough,

    Modify {
        // Every transformed batch shares this schema, so build it once and cache it.
        target_schema: Arc<ArrowSchema>,

        operations: Vec<ColumnSource>,
    },

    // Only the schema changes, such as a rename. Keep the existing column `Vec` and save a heap
    // allocation per batch.
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

/// Builds a [`RecordBatchTransformer`] from its optional parameters.
///
/// The constant fields are pre-computed once, for both metadata fields such as `_file` and
/// identity-partitioned fields, so batch processing does not repeat the work.
#[derive(Debug)]
pub(crate) struct RecordBatchTransformerBuilder {
    snapshot_schema: Arc<IcebergSchema>,
    projected_iceberg_field_ids: Vec<i32>,
    constant_fields: HashMap<i32, Datum>,
    /// V3 row lineage: the data file's assigned `first_row_id` and its file sequence number. `None`
    /// when the table is not V3 or the file has no assigned range.
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

    /// Adds the constant `datum` for `field_id`. Metadata fields such as `_file` use it.
    pub(crate) fn with_constant(mut self, field_id: i32, datum: Datum) -> Self {
        self.constant_fields.insert(field_id, datum);
        self
    }

    /// Supply the V3 row-lineage inputs for this data file.
    ///
    /// Both are `Option`; without them a projected row-lineage column is all-NULL, as in Java.
    /// Never defaulted to zero, which would mint colliding row ids.
    pub(crate) fn with_row_lineage(
        mut self,
        first_row_id: Option<i64>,
        file_sequence_number: Option<i64>,
    ) -> Self {
        self.first_row_id = first_row_id;
        self.file_sequence_number = file_sequence_number;
        self
    }

    /// Sets the partition spec and its tuple, then merges the identity-partition constants into
    /// the constant fields. The spec names the identity fields, and the tuple holds their values,
    /// so the two arrive together.
    pub(crate) fn with_partition(
        mut self,
        partition_spec: Arc<PartitionSpec>,
        partition_data: Struct,
    ) -> Result<Self> {
        let partition_constants =
            constants_map(&partition_spec, &partition_data, &self.snapshot_schema)?;

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

/// Transforms a data file's RecordBatches to match the Iceberg table schema. It handles schema
/// evolution, column reordering, type promotion, and the spec's Column Projection rules.
///
/// | Rule | Source for a field id the file does not carry |
/// |---|---|
/// | 1 | the partition metadata constant, for an identity transform |
/// | 2 | the name mapping, applied earlier by `ArrowReader` |
/// | 3 | the field's `initial-default` |
/// | 4 | null |
///
/// # Notes
///
/// `ArrowReader` resolves every field id before the read, like Java `ReadConf`, so the ids here
/// are already trustworthy. This transformer applies rules 1, 3, and 4 only.
///
/// A non-identity transform stores a derived value, so its source column comes from the data file.
/// `bucket(4, id)` stores the bucket number, and runtime filtering on `id` needs the real values.
#[derive(Debug)]
pub(crate) struct RecordBatchTransformer {
    snapshot_schema: Arc<IcebergSchema>,
    projected_iceberg_field_ids: Vec<i32>,
    // Metadata fields such as `_file`, plus the identity-partitioned fields.
    constant_fields: HashMap<i32, Datum>,

    // See `RecordBatchTransformerBuilder::with_row_lineage`.
    first_row_id: Option<i64>,
    file_sequence_number: Option<i64>,

    // Built lazily from the first batch's schema.
    batch_transform: Option<BatchTransform>,

    // The 0-based physical position of the NEXT row. It feeds `ColumnSource::RowPosition`, and it
    // is correct only under an in-order, no-skip decode. See that variant.
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

        // Captured before the immutable borrow of `batch_transform` below.
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

        // Advance by the FULL batch, before any later delete or predicate mask drops rows, so the
        // next batch's `_pos` continues from the correct physical ordinal.
        self.next_row_position = self.next_row_position.saturating_add(row_count as u64);

        Ok(result)
    }

    /// Compares the incoming batch schema with the snapshot schema and picks the transform to
    /// apply.
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

        // Select fields in the order of `projected_iceberg_field_ids`.
        let fields: Result<Vec<_>> = projected_iceberg_field_ids
            .iter()
            .map(|field_id| {
                if constant_fields.contains_key(field_id) {
                    if let Ok(iceberg_field) = get_metadata_field(*field_id) {
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
                        // An identity-partition field EXISTS in the table schema, so its declared
                        // scan-schema field is authoritative. The constant must match that field
                        // exactly, never a Run-End-Encoded variant. REE here makes the output
                        // schema declare REE where the scan schema requires a plain `Utf8`.
                        Ok(field_id_to_mapped_schema_map
                            .get(field_id)
                            .ok_or(Error::new(ErrorKind::Unexpected, "field not found"))?
                            .0
                            .clone())
                    }
                } else if *field_id == RESERVED_FIELD_ID_ROW_ID
                    || *field_id == RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER
                {
                    // Row-lineage columns are absent from the table schema, like `_pos`, so the
                    // field comes from the reserved-column definition.
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
                    // `_pos` is absent from the table schema, so the lookup below would fail.
                    // `ColumnSource::RowPosition` synthesizes the values from the read position.
                    let pos_meta = get_metadata_field(*field_id)?;
                    Ok(Arc::new(
                        Field::new(&pos_meta.name, DataType::Int64, !pos_meta.required)
                            .with_metadata(HashMap::from([(
                                PARQUET_FIELD_ID_META_KEY.to_string(),
                                pos_meta.id.to_string(),
                            )])),
                    ))
                } else {
                    Ok(field_id_to_mapped_schema_map
                        .get(field_id)
                        .ok_or(Error::new(ErrorKind::Unexpected, "field not found"))?
                        .0
                        .clone())
                }
            })
            .collect();

        let target_schema = Arc::new(ArrowSchema::new(fields?));

        // A constant field is AUTHORITATIVE and must override a file column of the same field id,
        // as in Java `BaseParquetReaders`. The `PassThrough` and `ModifySchema` fast paths would
        // hand back the FILE value, so force the column-rebuilding `Modify` path.
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

    /// Compares the source and target schemas.
    ///
    /// | Difference | Result |
    /// |---|---|
    /// | field count, data type, or nullability | `Different`: rebuild schema and columns |
    /// | column names only | `NameChangesOnly`: rebuild the schema, keep the columns |
    /// | none | `Equivalent`: pass the batch through |
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

            // A positional field-id mismatch means the file's column order differs from the
            // projected order. The fast paths relabel or pass columns through BY POSITION, which
            // hands back the wrong column under a field's name. Force the `Modify` path, which
            // sources each output column by field id.
            if let (Some(source_id), Some(target_id)) = (
                Self::field_id_of(source_field),
                Self::field_id_of(target_field),
            ) && source_id != target_id
            {
                return SchemaComparison::Different;
            }

            // A row-lineage column is never a pass-through. Its value is stored-else-fallback per
            // ROW, so force the `Modify` path. The source half is defensive: the target half alone
            // decides every case reachable today.
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
                // A constant field wins over a file column of the same id, per spec rule 1.
                if let Some(datum) = constant_fields.get(field_id) {
                    // The physical Arrow type MUST equal what the target schema declares, or
                    // `RecordBatch::try_new` rejects the batch. A metadata field has no table
                    // schema entry, so the target declares it Run-End-Encoded. An
                    // identity-partition field has one, so the target declares its plain type.
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

                // `_pos` is absent from the table schema, so the lookup below would fail. The
                // Avro reader emits `_pos` as a running counter, so pass a stored column through.
                // Parquet and ORC have none, so synthesize it from the read position.
                if *field_id == RESERVED_FIELD_ID_POS {
                    return Ok(match field_id_to_source_schema_map.get(field_id) {
                        Some((_, source_index)) => ColumnSource::PassThrough {
                            source_index: *source_index,
                        },
                        None => ColumnSource::RowPosition,
                    });
                }

                // Java `ValueReaders.fileFieldReader` dispatches on whether the FILE carries the
                // field. Present gets a per-row fallback reader, absent gets a computed value.
                if *field_id == RESERVED_FIELD_ID_ROW_ID {
                    // No assigned range gives an all-NULL column, as in Java
                    // `ValueReaders.rowIds(null, reader)`. A V1 or V2 file has no row identity.
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
                    // Java gates this column on BOTH inputs, so a V1 or V2 file reports NULL.
                    // The sequence number alone fabricates a value for every pre-V3 row.
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

                // A constant field wins over a file column of the same id, per spec rule 1.
                // `generate_batch_transform` already handled that above.

                // Every field id in the source schema is already resolved and trustworthy.
                // `reader.rs` applied the embedded ids, the name mapping, or the position
                // fallback, so no conflict detection is needed here.
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

                let column_source = if let Some(source) = field_by_id {
                    source
                } else {
                    // The file has no such column, so fall to rule 3 then rule 4.
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
            // A field with no field id is left to the name mapping.
        }

        Ok(field_id_to_source_schema)
    }

    /// `first_row_id + physical ordinal` for `num_rows` rows from `start_row_position`. The
    /// fallback arm of Java `ValueReaders$RowIdReader`.
    ///
    /// # Errors
    ///
    /// On `i64` overflow. Java wraps, but a wrapped id aliases another live row's identity.
    fn row_ids_from_positions(
        first_row_id: i64,
        start_row_position: u64,
        num_rows: usize,
    ) -> Result<Int64Array> {
        if num_rows == 0 {
            return Ok(Int64Array::from_iter_values(std::iter::empty()));
        }
        let overflow = || row_id_overflow(first_row_id, start_row_position, num_rows);

        // Ids rise with position, so the LAST row bounds the batch. Its offset is
        // `start + num_rows - 1`. `start + num_rows` would reject a batch ending at `i64::MAX`.
        let first = first_row_id
            .checked_add(i64::try_from(start_row_position).map_err(|_| overflow())?)
            .ok_or_else(overflow)?;
        let last_offset = i64::try_from(num_rows - 1).map_err(|_| overflow())?;
        first.checked_add(last_offset).ok_or_else(overflow)?;

        // Every id in `[first, first + num_rows - 1]` is proven representable, so the per-row
        // addition below cannot overflow.
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
                        let end = start_row_position.saturating_add(num_rows as u64);
                        Arc::new(Int64Array::from_iter_values(
                            (start_row_position..end).map(|p| p as i64),
                        ))
                    }

                    ColumnSource::RowId { first_row_id } => {
                        // No stored column, so every row is `firstRowId + pos`.
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
                        // Java `ValueReaders$RowIdReader.read`: the stored id wins, and only a
                        // NULL falls back to `firstRowId + pos`.
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
                        // Java `ValueReaders$LastUpdatedSeqReader.read`: the stored value wins,
                        // and only a NULL falls back to the file's own sequence number.
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
        if let DataType::RunEndEncoded(_, values_field) = target_type {
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

            let values_array =
                create_primitive_array_single_element(values_field.data_type(), prim_lit)?;

            create_ree_array(values_array)
        } else {
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

    /// Reads a string from a `StringArray` or a run-end-encoded one. A null gives `""`.
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
            // Every row of an REE constant column shares the value at index 0.
            if string_values.is_null(0) {
                String::new()
            } else {
                string_values.value(0).to_string()
            }
        } else {
            panic!("Expected StringArray or RunEndEncoded<StringArray>");
        }
    }

    /// Reads an int from an `Int32Array` or a run-end-encoded one.
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
        // A DATE column added after the file was written must materialize as NULLs.
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
        // Risk pinned: the `_pos` counter must CONTINUE across batches. A downstream engine
        // writes position deletes against it.
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

        // First batch of 3 rows gives _pos 0, 1, 2.
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

        // Second batch of 2 rows gives _pos 3, 4. The counter does NOT restart.
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
        // A struct column added after the data files were written must materialize as nulls.
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

    /// `add_files` over a Hive-style Parquet file with no field ids. `ArrowReader` has already
    /// applied the name mapping, so `name` and `subdept` arrive with ids 2 and 4.
    ///
    /// Risk pinned: the partition columns `id` and `dept` are absent from the file and must come
    /// from `initial_default`, spec rule 3. The two mapped columns must come from the file.
    #[test]
    fn add_files_with_name_mapping_applied_in_reader() {
        // Schema after add_files: id (partition), name, dept (partition), subdept.
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

        // The file held name and subdept with no ids. `reader.rs` mapped them to 2 and 4. The
        // partition columns id and dept live in the directory path, not the file.
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

        // The file's two columns.
        let parquet_batch = RecordBatch::try_new(parquet_schema, vec![
            Arc::new(StringArray::from(vec!["John Doe"])),
            Arc::new(StringArray::from(vec!["communications"])),
        ])
        .unwrap();

        let result = transformer.process_record_batch(parquet_batch).unwrap();

        // id and dept come from initial_default, name and subdept from the file.
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

    /// Risk pinned: treating a bucket-partitioned field as a constant. Partition metadata holds
    /// `id_bucket = 2`, while the real `id` values 100, 200, 300 live only in the data file.
    /// Replacing the column with the bucket number breaks runtime filtering, so `WHERE id = 100`
    /// would match no rows. Java `PartitionUtil.constantsMap` filters on `isIdentity()`, which is
    /// false for a bucket transform.
    #[test]
    fn bucket_partitioning_reads_source_column_from_file() {
        use crate::spec::{Struct, Transform};

        // Schema: id and name are data columns, id_bucket is the partition column.
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

        // Partition spec: bucket(4, id).
        let partition_spec = Arc::new(
            crate::spec::PartitionSpec::builder(snapshot_schema.clone())
                .with_spec_id(0)
                .add_partition_field("id", "id_bucket", Transform::Bucket(4))
                .unwrap()
                .build()
                .unwrap(),
        );

        // Partition tuple: bucket value 2.
        let partition_data = Struct::from_iter(vec![Some(Literal::int(2))]);

        // The file carries both id and name.
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

        // The id column MUST be read from here, not treated as a constant.
        let parquet_batch = RecordBatch::try_new(parquet_schema, vec![
            Arc::new(Int32Array::from(vec![100, 200, 300])),
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
        ])
        .unwrap();

        let result = transformer.process_record_batch(parquet_batch).unwrap();

        // id must come from the file, not from partition metadata.
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.num_rows(), 3);

        let id_column = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        // These values MUST come from the file, not from a constant.
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

    /// The complement to `bucket_partitioning_reads_source_column_from_file`. An identity
    /// transform stores the real value, so `dept = "engineering"` comes from partition metadata
    /// and the file is never consulted. This is what makes a metadata-only Hive migration work.
    #[test]
    fn identity_partition_uses_constant_from_metadata() {
        use crate::spec::{Struct, Transform};

        // Schema: id and name are data columns, dept is the partition column.
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

        // Partition spec: identity(dept).
        let partition_spec = Arc::new(
            crate::spec::PartitionSpec::builder(snapshot_schema.clone())
                .with_spec_id(0)
                .add_partition_field("dept", "dept", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );

        // Partition tuple: dept="engineering".
        let partition_data = Struct::from_iter(vec![Some(Literal::string("engineering"))]);

        // The file carries only id and name. dept lives in the partition path.
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

        // dept must carry the partition-metadata constant.
        assert_eq!(result.num_columns(), 3);
        assert_eq!(result.num_rows(), 2);

        assert_eq!(get_int_value(result.column(0).as_ref(), 0), 100);
        assert_eq!(get_int_value(result.column(0).as_ref(), 1), 200);

        // dept comes from partition metadata, so it is REE.
        assert_eq!(
            get_string_value(result.column(1).as_ref(), 0),
            "engineering"
        );
        assert_eq!(
            get_string_value(result.column(1).as_ref(), 1),
            "engineering"
        );

        // name comes from the file.
        assert_eq!(get_string_value(result.column(2).as_ref(), 0), "Alice");
        assert_eq!(get_string_value(result.column(2).as_ref(), 1), "Bob");
    }

    /// Risk pinned: a partition tuple SHORTER than its spec used to index past the end of
    /// `Struct` and panic, which killed the scan task. Java `PartitionData.get` returns null past
    /// the end, so the field must instead stay out of the constants map and resolve as null.
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

        // Two identity partition fields.
        let partition_spec = PartitionSpec::builder(snapshot_schema.clone())
            .with_spec_id(0)
            .add_partition_field("dept", "dept", Transform::Identity)
            .expect("add dept partition field")
            .add_partition_field("region", "region", Transform::Identity)
            .expect("add region partition field")
            .build()
            .expect("build partition spec");

        // A tuple carrying only the first value.
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

    /// The bucket case after `RENAME COLUMN id TO row_id`. The file still names the column `id`,
    /// and both sides keep field id 1.
    ///
    /// Risk pinned: the lookup must match on field id, not on name, and must still read 100, 200,
    /// 300 from the file rather than the bucket constant 2.
    #[test]
    fn test_bucket_partitioning_with_renamed_source_column() {
        use crate::spec::{Struct, Transform};

        // Schema after the rename: row_id, name.
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

        // Partition spec: bucket(4, row_id), with source_id still 1.
        let partition_spec = Arc::new(
            crate::spec::PartitionSpec::builder(snapshot_schema.clone())
                .with_spec_id(0)
                .add_partition_field("row_id", "row_id_bucket", Transform::Bucket(4))
                .unwrap()
                .build()
                .unwrap(),
        );

        // Partition tuple: bucket value 2.
        let partition_data = Struct::from_iter(vec![Some(Literal::int(2))]);

        // The file keeps the OLD name "id" and the SAME field id 1.
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

        // The data must be read through field id 1.
        let parquet_batch = RecordBatch::try_new(parquet_schema, vec![
            Arc::new(Int32Array::from(vec![100, 200, 300])),
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
        ])
        .unwrap();

        let result = transformer.process_record_batch(parquet_batch).unwrap();

        // The name mismatch must not change the result.
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.num_rows(), 3);

        let row_id_column = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        // These values MUST come from the file through field id 1, not from the bucket constant.
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

    /// All four Column Projection rules in one batch.
    ///
    /// | Column | Rule | Source |
    /// |---|---|---|
    /// | dept | 1 | the identity-partition constant |
    /// | data | 2 | the file, through the name mapping |
    /// | category | 3 | `initial_default` |
    /// | notes | 4 | null |
    #[test]
    fn test_all_four_spec_rules() {
        use crate::spec::Transform;

        // One column per spec rule.
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    // The normal case: found in the file by field id.
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    // Rule 1: identity-partitioned.
                    NestedField::required(2, "dept", Type::Primitive(PrimitiveType::String)).into(),
                    // Rule 2: resolved by the name mapping.
                    NestedField::required(3, "data", Type::Primitive(PrimitiveType::String)).into(),
                    // Rule 3: has an initial_default.
                    NestedField::optional(4, "category", Type::Primitive(PrimitiveType::String))
                        .with_initial_default(Literal::string("default_category"))
                        .into(),
                    // Rule 4: no default, so null.
                    NestedField::optional(5, "notes", Type::Primitive(PrimitiveType::String))
                        .into(),
                ])
                .build()
                .unwrap(),
        );

        // Partition spec: identity(dept).
        let partition_spec = Arc::new(
            crate::spec::PartitionSpec::builder(snapshot_schema.clone())
                .with_spec_id(0)
                .add_partition_field("dept", "dept", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );

        // Partition tuple: dept="engineering".
        let partition_data = Struct::from_iter(vec![Some(Literal::string("engineering"))]);

        // The post-ArrowReader file schema: id (1) and data (3). dept, category, and notes are
        // absent.
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

        // Each column below demonstrates one spec rule.

        // The normal case: id from the file by field id.
        assert_eq!(get_int_value(result.column(0).as_ref(), 0), 100);
        assert_eq!(get_int_value(result.column(0).as_ref(), 1), 200);

        // Rule 1: dept from partition metadata, so REE.
        assert_eq!(
            get_string_value(result.column(1).as_ref(), 0),
            "engineering"
        );
        assert_eq!(
            get_string_value(result.column(1).as_ref(), 1),
            "engineering"
        );

        // Rule 2: data from the file, so a plain array.
        assert_eq!(get_string_value(result.column(2).as_ref(), 0), "value1");
        assert_eq!(get_string_value(result.column(2).as_ref(), 1), "value2");

        // Rule 3: category from initial_default, so REE.
        assert_eq!(
            get_string_value(result.column(3).as_ref(), 0),
            "default_category"
        );
        assert_eq!(
            get_string_value(result.column(3).as_ref(), 1),
            "default_category"
        );

        // Rule 4: notes is a null REE column.
        assert_eq!(get_string_value(result.column(4).as_ref(), 0), "");
        assert_eq!(get_string_value(result.column(4).as_ref(), 1), "");
    }

    /// Risk pinned: a null value in an identity-partitioned column used to error. It must
    /// materialize as a null column.
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

        // The partition tuple holds a null.
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

        // The partition column must produce nulls.
        let data_col = result.column(1);
        assert!(data_col.is_null(0));
        assert!(data_col.is_null(1));
        assert!(data_col.is_null(2));
    }

    /// Risk pinned: the REE leak. An identity-partition column exists in the table schema, so the
    /// output batch must declare its plain physical type, `Utf8` here, never `RunEndEncoded`.
    /// Materializing the constant as REE made the output schema disagree with the scan schema.
    ///
    /// The test asserts the EXACT physical type, not just the value. A `get_string_value` helper
    /// would pass under REE too.
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

        // The file lacks the partition column, as in a Hive migration. It carries only `id`.
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

        // The declared field for `category` must be plain Utf8, never RunEndEncoded.
        assert_eq!(
            result.schema().field(1).data_type(),
            &DataType::Utf8,
            "identity-partition constant must declare its plain scan-schema type, not REE"
        );

        // The materialized column must be a plain StringArray, not a RunArray.
        let category = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("category column must be a plain StringArray, not RunEndEncoded");
        assert_eq!(category.value(0), "electronics");
        assert_eq!(category.value(1), "electronics");
        assert_eq!(category.value(2), "electronics");
    }

    /// Risk pinned: the int-to-long widening bug. A partition tuple can carry a literal narrower
    /// than a type-promoted column, such as `Int(19)` for a `Long` column. `Datum::to` must
    /// coerce the value to the FIELD's type, like Java `IdentityPartitionConverters
    /// .convertConstant`. Without it the array builder sees `(Int64, Int(19))` and errors.
    #[test]
    fn identity_partition_widens_int_literal_to_long_column() {
        use arrow_schema::DataType;

        use crate::spec::Transform;

        // Column `p` is Long, but the tuple still stores the narrower Int(19) variant.
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

        // The partition value in its NARROW Int(19) form.
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

        // `p` must materialize as a plain Int64 column with the widened value.
        assert_eq!(result.schema().field(1).data_type(), &DataType::Int64);
        let p_col = result
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("p column must be a plain Int64Array");
        assert_eq!(p_col.value(0), 19);
        assert_eq!(p_col.value(1), 19);
    }

    /// Risk pinned: a REORDERED and SUBSET projection must give an output schema exactly equal to
    /// the declared projection, in names, plain physical types, nullability, AND order. The
    /// constant column must be a plain array carrying the PARTITION value.
    ///
    /// The reordered shape already forces the `Modify` path, so this does not isolate the
    /// `constant_overrides_file_column` flag. The scan test
    /// `test_identity_partition_column_value_comes_from_metadata_not_file` pins that alone.
    #[test]
    fn identity_partition_reordered_subset_projection_matches_declared_schema() {
        use arrow_schema::DataType;

        use crate::spec::Transform;

        // Schema order: id(1, Int), category(2, String, partitioned), extra(3, Long).
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

        // The partition value DIFFERS from the file's, so the override is observable.
        let partition_data = Struct::from_iter(vec![Some(Literal::string("books"))]);

        // The file carries ALL THREE columns, add_files style, with a different `category`.
        let parquet_schema = Arc::new(ArrowSchema::new(vec![
            simple_field("id", DataType::Int32, false, "1"),
            simple_field("category", DataType::Utf8, false, "2"),
            simple_field("extra", DataType::Int64, true, "3"),
        ]));

        // Project category(2) first, id(1) second, and drop extra(3).
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

        // The output schema must be exactly [category: Utf8, id: Int32], both non-null.
        assert_eq!(result.num_columns(), 2);
        let sch = result.schema();
        assert_eq!(sch.field(0).name(), "category");
        assert_eq!(sch.field(0).data_type(), &DataType::Utf8);
        assert!(!sch.field(0).is_nullable());
        assert_eq!(sch.field(1).name(), "id");
        assert_eq!(sch.field(1).data_type(), &DataType::Int32);
        assert!(!sch.field(1).is_nullable());

        // category is the constant plain StringArray, and it OVERRIDES the file value.
        let category = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("category must be a plain StringArray, not REE");
        assert_eq!(category.value(0), "books");
        assert_eq!(category.value(1), "books");

        // id comes from the file unchanged.
        let id = result
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id.value(0), 10);
        assert_eq!(id.value(1), 11);
    }

    // ---- V3 row lineage: `_row_id` / `_last_updated_sequence_number` -------------------------
    //
    // Java dispatches on whether the FILE carries the field, then per ROW on whether the stored
    // value is null. Both axes are pinned below, and the mixed-null cells discriminate.
    //
    // | | file lacks the column | file has it, no nulls | file has it, some nulls |
    // |---|---|---|---|
    // | `_row_id` | `first_row_id + pos` for every row | stored value verbatim | stored wins per row; NULL -> `first_row_id + pos` |
    // | `_last_updated_sequence_number` | the file's sequence number, constant | stored value verbatim | stored wins per row; NULL -> file sequence number |

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

    /// A batch of `id` values, optionally carrying a reserved row-lineage column.
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

        // The counter CONTINUES across batches. A restart repeats the same row ids.
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

    /// Java returns an all-NULL column here, not an error. An error would make `SELECT _row_id`
    /// unusable on a mixed-version table.
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

    /// The discriminating cell for the absent-range arm. The stored column is IGNORED, because
    /// the arm is chosen before the file is consulted.
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

    /// Java gates `_last_updated_sequence_number` on BOTH inputs, so a V1 or V2 file reports
    /// NULL. The sequence number alone fabricates a value for every pre-V3 row.
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

    /// The `num_rows == 0` guard in `row_ids_from_positions` is load-bearing. Without it
    /// `num_rows - 1` underflows on an ordinary empty batch.
    #[test]
    fn an_empty_batch_yields_an_empty_row_id_column() {
        let projected = [1, RESERVED_FIELD_ID_ROW_ID];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(Some(100), Some(7))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(Vec::new(), None))
            .expect("an empty batch is not an error");
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(int64_col(&batch, 1), Vec::<Option<i64>>::new());

        // The counter is unmoved, so the NEXT batch starts at the range's first id.
        let next = transformer
            .process_record_batch(row_lineage_batch(vec![1, 2], None))
            .expect("second batch");
        assert_eq!(int64_col(&next, 1), vec![Some(100), Some(101)]);
    }

    /// The boundary the overflow check must NOT reject: a batch whose last id is exactly
    /// `i64::MAX`. Only here does `start + num_rows` differ from `start + num_rows - 1`.
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
