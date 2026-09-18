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

use arrow_array::{Array as ArrowArray, Int64Array, RecordBatch, RecordBatchOptions};
use arrow_cast::cast;
use arrow_schema::{
    DataType, Field, FieldRef, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef, SchemaRef,
};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::nested_projection::{
    NestedProjectionPlan, create_constant_column, nested_projection_applies,
};
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
/// `PartitionUtil.constantsMap`. # Notes Only identity transforms qualify.
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

    NestedProject(NestedProjectionPlan, usize),

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
pub(crate) enum BatchTransform {
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

    batch_transform: Option<(SchemaRef, BatchTransform)>,

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
        if self.batch_transform.as_ref().is_none_or(|(schema, _)| {
            !Arc::ptr_eq(schema, record_batch.schema_ref())
                && **schema != **record_batch.schema_ref()
        }) {
            let transform = Self::generate_batch_transform(
                record_batch.schema_ref(),
                self.snapshot_schema.as_ref(),
                &self.projected_iceberg_field_ids,
                &self.constant_fields,
                self.first_row_id,
                self.file_sequence_number,
            )?;
            self.batch_transform = Some((record_batch.schema_ref().clone(), transform));
        }

        // Captured before the immutable borrow of `batch_transform` below.
        let start_row_position = self.next_row_position;
        let row_count = record_batch.num_rows();

        let (_, transform) = self
            .batch_transform
            .as_mut()
            .expect("batch_transform was just initialized");
        let result = match transform {
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
    pub(crate) fn generate_batch_transform(
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

                let target_field = &field_id_to_mapped_schema_map
                    .get(field_id)
                    .ok_or(Error::new(
                        ErrorKind::Unexpected,
                        "could not find field in schema",
                    ))?
                    .0;
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
                let field_by_id = field_id_to_source_schema_map
                    .get(field_id)
                    .map(|(source_field, source_index)| {
                        let source_type = source_field.data_type();
                        if nested_projection_applies(source_type, target_type)
                            && source_type != target_type
                        {
                            NestedProjectionPlan::build(
                                source_type,
                                target_type,
                                snapshot_schema,
                                target_field.name(),
                            )
                            .map(|plan| ColumnSource::NestedProject(plan, *source_index))
                        } else if source_type.equals_datatype(target_type) {
                            Ok(ColumnSource::PassThrough {
                                source_index: *source_index,
                            })
                        } else {
                            Ok(ColumnSource::Promote {
                                target_type: target_type.clone(),
                                source_index: *source_index,
                            })
                        }
                    })
                    .transpose()?;

                let column_source = if let Some(source) = field_by_id {
                    source
                } else {
                    // The file has no such column, so fall to rule 3 then rule 4.
                    let default_value =
                        iceberg_field
                            .initial_default
                            .as_ref()
                            .and_then(|lit| match lit {
                                Literal::Primitive(prim) => Some(prim.clone()),
                                _ => None,
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
        operations: &mut [ColumnSource],
        num_rows: usize,
        start_row_position: u64,
    ) -> Result<Vec<Arc<dyn ArrowArray>>> {
        operations
            .iter_mut()
            .map(|op| {
                Ok(match op {
                    ColumnSource::PassThrough { source_index } => columns[*source_index].clone(),

                    ColumnSource::Promote {
                        target_type,
                        source_index,
                    } => cast(&*columns[*source_index], target_type)?,

                    ColumnSource::NestedProject(plan, source_index) => {
                        plan.apply(columns[*source_index].clone())?
                    }

                    ColumnSource::Add { target_type, value } => {
                        create_constant_column(target_type, value, num_rows)?
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

    include!("record_batch_transformer_tests.rs");
    include!("record_batch_transformer_partition_tests.rs");

    // ---- V3 row lineage: `_row_id` / `_last_updated_sequence_number` -------------------------
    //
    // Java dispatches on whether the FILE carries the field, then per ROW on whether the stored
    // value is null. Both axes are pinned below, and the mixed-null cells discriminate.
    //
    // | | file lacks the column | file has it, no nulls | file has it, some nulls |
    // |---|---|---|---|
    // | `_row_id` | `first_row_id + pos` for every row | stored value verbatim | stored wins per row; NULL -> `first_row_id + pos` |
    // | `_last_updated_sequence_number` | the file's sequence number, constant | stored value verbatim | stored wins per row; NULL -> file sequence number |

    include!("record_batch_transformer_row_lineage_tests.rs");
}
