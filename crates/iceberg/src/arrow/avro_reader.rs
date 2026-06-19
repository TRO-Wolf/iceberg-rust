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

//! Avro **data-file** value reader: an `apache_avro` datum stream → Arrow [`RecordBatch`]
//! converter that mirrors Java 1.10.0 `org.apache.iceberg.data.avro.PlannedDataReader`
//! (`org.apache.iceberg.avro.ValueReaders.buildReadPlan`).
//!
//! # What this is (and is not)
//!
//! This is the **engine core** for reading Iceberg data files written in Avro. It is a *pure*
//! module: given the raw bytes of an Avro Object Container File plus the **expected** Iceberg
//! schema (the projection), it produces Arrow record batches. The scan read path wires it in via
//! [`crate::arrow::reader`]'s `process_avro_file_scan_task` (U2): that path feeds the batches
//! produced here through the same `RecordBatchTransformer` the Parquet path uses (schema evolution
//! + `_file` / partition constants) and applies merge-on-read deletes post-materialization.
//!
//! # The read plan (parity with `ValueReaders.buildReadPlan`)
//!
//! Resolution is **by field-id**, never by name — **at every struct level**. Java's
//! `PlannedDataReader.ReadBuilder.record` calls `ValueReaders.buildReadPlan` for *each* record it
//! visits (the partner visitor recurses the `(expected Iceberg, file Avro)` tree), so a nested
//! struct resolves its children by id with the full skip + missing-default machinery just like the
//! top level. We mirror that: the writer Avro schema is read from the OCF header and a recursive
//! [`StructPlan`] is compiled over the `(writer record, expected struct)` pair, descending into
//! nested structs, list elements, and map keys/values.
//!
//! For each writer record field (each carrying the `field-id` Avro property, the same property
//! [`crate::avro`]'s schema conversion round-trips):
//!
//! - if the id maps to a position in the expected (projection) struct → **project** it there;
//! - otherwise → **skip** it (decode-and-discard the value).
//!
//! Every expected id that no writer field claimed is a **missing column**, filled with a constant
//! in Java's exact priority order:
//!
//! 1. an id→constant mapping (partition / metadata constants) — *not exposed by this v1 entry
//!    point* (see [`ReadPlanInput::id_to_constant`]); when empty this rung never fires;
//! 2. the field's V3 `initial-default` → constant;
//! 3. id == `IS_DELETED` (`RESERVED_FIELD_ID_DELETED`) → constant `false`;
//! 4. id == `ROW_POSITION` (`RESERVED_FIELD_ID_POS`) → the running row-position counter;
//! 5. the field is optional → constant `null`;
//! 6. otherwise → error `Missing required field: <name>`.
//!
//! # Type promotion
//!
//! Read-time promotion is driven by the **expected** Iceberg type, exactly as Java does it:
//! an Avro `int` read into an expected `long` is widened to `long`; an Avro `float` read into an
//! expected `double` is widened to `double`. No other promotions.
//!
//! # Timestamp tz-ness
//!
//! tz-ness is decided by the **expected** Iceberg type, not the file. This is *not* a literal
//! transcription of Java (Java's `ReadBuilder.primitive` reads the file's `adjust-to-utc` schema
//! prop via `AvroSchemaUtil.isTimestamptz`), but the outcome is parity-correct here because (1) the
//! Rust writer ([`crate::avro`]) emits no `adjust-to-utc` and collapses both `timestamp` and
//! `timestamptz` to Avro `timestamp-micros` (and the `_ns` pair to `timestamp-nanos`), and (2)
//! `apache_avro` 0.21 keys tz purely off the logical-type *name* (`timestamp-micros` vs
//! `local-timestamp-micros`) and ignores `adjust-to-utc` entirely — so even a Java-written
//! `timestamptz` file (`timestamp-micros` + `adjust-to-utc=true`) decodes to
//! [`AvroValue::TimestampMicros`]. Expected-type dispatch is therefore the only viable strategy and
//! produces the same Arrow tz-ness Java would.
//!
//! # Known gaps (v1)
//!
//! - Files **without** Avro `field-id` properties (name-mapping fallback,
//!   `org.apache.iceberg.mapping.NameMapping`) are **not** supported yet: such a file errors loudly
//!   rather than silently resolving by name (wrong slot).
//! - V3 **row-lineage** present-field readers — Java `ValueReaders.fileFieldReader` special-cases a
//!   *present* file field whose id is `ROW_ID` or `LAST_UPDATED_SEQUENCE_NUMBER` with dedicated
//!   readers. Here those ids, when present in the file, read as plain columns (their default-fill
//!   path is unaffected). This is niche V3 metadata and is deferred with U2.
//! - The `variant` type read is deferred (the Avro→Iceberg schema converter rejects variant on the
//!   read path anyway).

// U2 (scan-path wiring) is DONE: `arrow::reader`'s `process_avro_file_scan_task` consumes
// `read_avro_data_file` in the production scan path, and `read_avro_data_bytes` (the sync core) is
// called both by `read_avro_data_file` and by the offline tests. `ReadPlanInput`'s `id_to_constant`
// rung is exercised only by the offline tests below — the scan supplies its partition constants
// through the `RecordBatchTransformer` (as the Parquet path does), not through the reader's
// id→constant map. This file carries no module-level `dead_code` allow; any genuinely-dead item is
// flagged at the item with a targeted allow and a one-line reason.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use apache_avro::schema::{RecordSchema, UnionSchema};
use apache_avro::types::Value as AvroValue;
use apache_avro::{Reader as AvroReader, Schema as AvroSchema};
use arrow_array::builder::{FixedSizeBinaryBuilder, LargeBinaryBuilder, StringBuilder};
use arrow_array::{
    ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array, Int32Array,
    Int64Array, ListArray, MapArray, RecordBatch, StructArray, Time64MicrosecondArray,
    TimestampMicrosecondArray, TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};

use crate::arrow::{schema_to_arrow_schema, type_to_arrow_type};
use crate::io::InputFile;
use crate::metadata_columns::{RESERVED_FIELD_ID_DELETED, RESERVED_FIELD_ID_POS};
use crate::spec::{
    Literal, NestedField, PrimitiveLiteral, PrimitiveType, Schema, StructType, Type,
};
use crate::{Error, ErrorKind, Result};

/// The Avro record-field property holding an Iceberg field id (matches the `FIELD_ID_PROP`
/// stamped by [`crate::avro`]'s schema conversion).
const FIELD_ID_PROP: &str = "field-id";

// =================================================================================================
// Public(crate) entry points
// =================================================================================================

/// Read an Avro data file from an [`InputFile`] and return its rows as Arrow [`RecordBatch`]es.
///
/// The full file is fetched into memory (Avro OCF reads are inherently whole-block and the
/// `apache_avro` reader is synchronous), then decoded off the async executor via
/// [`crate::runtime::spawn_blocking`] so CPU-bound decode never blocks the runtime.
///
/// `expected` is the projection: only its fields appear in the output, in its field order, and the
/// output [`RecordBatch`] schema equals [`schema_to_arrow_schema(expected)`](schema_to_arrow_schema).
pub(crate) async fn read_avro_data_file(
    input: &InputFile,
    expected: Arc<Schema>,
    batch_size: usize,
) -> Result<Vec<RecordBatch>> {
    let bytes = input.read().await?;
    // Decode off the async executor: Avro datum decode is CPU-bound. The crate's `JoinHandle`
    // yields the closure's `Result` directly (it propagates a join panic internally).
    crate::runtime::spawn_blocking(move || {
        read_avro_data_bytes(&bytes, expected.as_ref(), batch_size)
    })
    .await
}

/// Decode Avro OCF `bytes` into Arrow [`RecordBatch`]es against the `expected` projection schema.
///
/// This is the synchronous core; [`read_avro_data_file`] is the async/[`InputFile`] wrapper. It is
/// exposed `pub(crate)` so callers that already hold the bytes (and offline tests) can drive it
/// directly without an [`InputFile`].
pub(crate) fn read_avro_data_bytes(
    bytes: &[u8],
    expected: &Schema,
    batch_size: usize,
) -> Result<Vec<RecordBatch>> {
    if batch_size == 0 {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            "Avro data-file batch_size must be greater than zero",
        ));
    }

    let reader = AvroReader::new(Cursor::new(bytes)).map_err(|e| {
        Error::new(
            ErrorKind::DataInvalid,
            "Failed to open Avro data file (could not read the OCF header)",
        )
        .with_source(e)
    })?;

    // The writer schema, as parsed from the OCF header; it carries the `field-id` props.
    let writer_avro_schema = reader.writer_schema().clone();
    let writer_record = match &writer_avro_schema {
        AvroSchema::Record(record) => record,
        other => {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Avro data file top-level schema must be a record, found {other:?}"),
            ));
        }
    };

    let plan = StructPlan::build(
        writer_record,
        expected.as_struct(),
        &ReadPlanInput::default(),
    )?;
    let arrow_schema = Arc::new(schema_to_arrow_schema(expected)?);

    // Row-oriented decode → per-column accumulation → columnar finalize, batched by `batch_size`.
    let mut batches = Vec::new();
    // Absolute row position within the file (for `_pos`); advanced across batches.
    let mut file_row_base: i64 = 0;
    let mut rows: Vec<Vec<(String, AvroValue)>> = Vec::with_capacity(batch_size.min(1024));

    for datum in reader {
        let value = datum.map_err(|e| {
            Error::new(ErrorKind::DataInvalid, "Failed to decode an Avro data row").with_source(e)
        })?;
        let AvroValue::Record(fields) = value else {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Avro data row was not a record",
            ));
        };
        rows.push(fields);

        if rows.len() == batch_size {
            let batch = plan.finish_batch(&rows, &arrow_schema, file_row_base)?;
            file_row_base += i64::try_from(rows.len()).unwrap_or(i64::MAX);
            batches.push(batch);
            rows.clear();
        }
    }

    if !rows.is_empty() {
        batches.push(plan.finish_batch(&rows, &arrow_schema, file_row_base)?);
    }

    Ok(batches)
}

/// Optional inputs to the read plan that have no v1 caller surface yet but exist so the missing
/// column default priority is faithful to Java.
#[derive(Default)]
pub(crate) struct ReadPlanInput {
    /// id → constant value (partition / metadata constants). Highest-priority missing-column
    /// default rung. Empty in v1: the scan layer (U2) supplies it.
    pub(crate) id_to_constant: HashMap<i32, Literal>,
}

// =================================================================================================
// The recursive read plan (id→pos at EVERY struct level — parity with ValueReaders.buildReadPlan)
// =================================================================================================

/// The compiled read plan for one struct level over `(writer Avro record, expected Iceberg struct)`.
///
/// Built recursively: a File-sourced child whose type is itself a struct / list / map carries its
/// own nested plan, so resolution is by field-id all the way down. This is the Rust analogue of
/// Java building a `ValueReaders.StructReader` whose per-field readers are themselves the result of
/// `buildReadPlan` at the nested level.
struct StructPlan {
    /// Projected output children, in expected field order.
    children: Vec<PlanColumn>,
}

/// One projected output column and how to source it.
struct PlanColumn {
    /// The expected Iceberg field for this output position.
    field: Arc<NestedField>,
    /// How to fill this column.
    source: ColumnSource,
}

/// Where a projected column's values come from.
enum ColumnSource {
    /// Read writer-record position `pos` of each row, decoding it with `reader`.
    File { pos: usize, reader: ValueReader },
    /// Fill with a constant [`Literal`] (or all-null when `None`).
    Constant(Option<Literal>),
    /// Fill with the running row position counter (`ROW_POSITION`, `_pos`).
    RowPosition,
}

impl StructPlan {
    /// Build the id→pos plan for one struct level, mirroring `ValueReaders.buildReadPlan`.
    fn build(
        writer_record: &RecordSchema,
        expected: &StructType,
        input: &ReadPlanInput,
    ) -> Result<Self> {
        // id → position in the EXPECTED struct (projection order). We `remove` ids as writer fields
        // claim them so the leftovers are exactly the missing columns.
        let mut id_to_pos: HashMap<i32, usize> = HashMap::with_capacity(expected.fields().len());
        for (pos, field) in expected.fields().iter().enumerate() {
            id_to_pos.insert(field.id, pos);
        }

        // For each expected position, the writer position + its writer Avro subschema (so nested
        // structs/lists/maps can build their own plans).
        let mut file_for_expected: Vec<Option<(usize, &AvroSchema)>> =
            vec![None; expected.fields().len()];

        for (writer_pos, writer_field) in writer_record.fields.iter().enumerate() {
            let id = avro_id(&writer_field.custom_attributes, FIELD_ID_PROP).ok_or_else(|| {
                Error::new(
                    ErrorKind::FeatureUnsupported,
                    format!(
                        "Avro data file field '{}' has no `field-id` property; name-mapping \
                         fallback is not supported yet",
                        writer_field.name
                    ),
                )
            })?;
            if let Some(expected_pos) = id_to_pos.remove(&id) {
                file_for_expected[expected_pos] = Some((writer_pos, &writer_field.schema));
            }
            // else: not projected → decode-and-discard. `apache_avro` already decoded the value
            // into the row's `Vec<(String, Value)>`; we simply never read that position.
        }

        // Assemble projected children in expected order, resolving missing-column defaults.
        let mut children = Vec::with_capacity(expected.fields().len());
        for (expected_pos, field) in expected.fields().iter().enumerate() {
            let source = match file_for_expected[expected_pos] {
                Some((pos, writer_schema)) => {
                    let reader = ValueReader::build(writer_schema, &field.field_type, input)?;
                    ColumnSource::File { pos, reader }
                }
                None => missing_column_source(field, input)?,
            };
            children.push(PlanColumn {
                field: field.clone(),
                source,
            });
        }

        Ok(StructPlan { children })
    }

    /// Read the children of one decoded struct row (`fields`), in plan order, into `child_acc` (one
    /// `Vec<AvroValue>` per File-sourced child, aligned with [`Self::child_arrays`]). A null parent
    /// row (`None`) pushes `Null` so the child arrays stay row-aligned; the parent's null buffer
    /// masks them out. Constant / row-position children are not accumulated here — they are
    /// materialised at finish time from the row count.
    fn push_row(&self, fields: Option<&[(String, AvroValue)]>, child_acc: &mut [Vec<AvroValue>]) {
        let mut file_idx = 0;
        for col in &self.children {
            if let ColumnSource::File { pos, .. } = &col.source {
                let value = match fields {
                    Some(fields) => fields
                        .get(*pos)
                        .map(|(_, v)| v.clone())
                        .unwrap_or(AvroValue::Null),
                    None => AvroValue::Null,
                };
                child_acc[file_idx].push(value);
                file_idx += 1;
            }
        }
    }

    /// Number of File-sourced children (the width of a `push_row` accumulator).
    fn num_file_children(&self) -> usize {
        self.children
            .iter()
            .filter(|c| matches!(c.source, ColumnSource::File { .. }))
            .count()
    }

    /// Convert all File-sourced children to Arrow arrays. `child_acc` must align with
    /// [`Self::push_row`] (one entry per File-sourced child, in plan order).
    fn child_arrays(&self, child_acc: &[Vec<AvroValue>]) -> Result<Vec<ArrayRef>> {
        let mut file_idx = 0;
        let mut out = Vec::with_capacity(child_acc.len());
        for col in &self.children {
            if let ColumnSource::File { reader, .. } = &col.source {
                out.push(reader.to_arrow(&child_acc[file_idx])?);
                file_idx += 1;
            }
        }
        Ok(out)
    }

    /// Finalise the top-level rows into a [`RecordBatch`], materialising constant / row-position
    /// columns from `row_base` and `num_rows`.
    fn finish_batch(
        &self,
        rows: &[Vec<(String, AvroValue)>],
        arrow_schema: &Arc<ArrowSchema>,
        row_base: i64,
    ) -> Result<RecordBatch> {
        let num_rows = rows.len();

        // Decode File-sourced children row by row.
        let mut child_acc: Vec<Vec<AvroValue>> = (0..self.num_file_children())
            .map(|_| Vec::with_capacity(num_rows))
            .collect();
        for row in rows {
            self.push_row(Some(row), &mut child_acc);
        }
        let mut file_arrays = self.child_arrays(&child_acc)?.into_iter();

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(self.children.len());
        for col in &self.children {
            let array = match &col.source {
                ColumnSource::File { .. } => file_arrays
                    .next()
                    .expect("one File array per File-sourced child"),
                ColumnSource::Constant(lit) => {
                    constant_array(&col.field.field_type, lit.as_ref(), num_rows)?
                }
                ColumnSource::RowPosition => {
                    let values: Vec<i64> = (0..num_rows)
                        .map(|i| row_base + i64::try_from(i).unwrap_or(i64::MAX))
                        .collect();
                    Arc::new(Int64Array::from(values))
                }
            };
            arrays.push(array);
        }

        RecordBatch::try_new(arrow_schema.clone(), arrays).map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                "Failed to assemble Arrow RecordBatch from decoded Avro rows",
            )
            .with_source(e)
        })
    }
}

/// Resolve a missing column's fill, in Java's exact priority order.
fn missing_column_source(field: &NestedField, input: &ReadPlanInput) -> Result<ColumnSource> {
    // (1) id → constant (partition / metadata constants).
    if let Some(lit) = input.id_to_constant.get(&field.id) {
        return Ok(ColumnSource::Constant(Some(lit.clone())));
    }
    // (2) V3 initial-default.
    if let Some(default) = &field.initial_default {
        return Ok(ColumnSource::Constant(Some(default.clone())));
    }
    // (3) IS_DELETED → false.
    if field.id == RESERVED_FIELD_ID_DELETED {
        return Ok(ColumnSource::Constant(Some(Literal::bool(false))));
    }
    // (4) ROW_POSITION → running counter.
    if field.id == RESERVED_FIELD_ID_POS {
        return Ok(ColumnSource::RowPosition);
    }
    // (5) optional → null.
    if !field.required {
        return Ok(ColumnSource::Constant(None));
    }
    // (6) required-and-absent → error.
    Err(Error::new(
        ErrorKind::DataInvalid,
        format!("Missing required field: {}", field.name),
    ))
}

/// Read an Iceberg field id out of an Avro record field's / array's / map's attribute map.
fn avro_id(
    attrs: &std::collections::BTreeMap<String, serde_json::Value>,
    key: &str,
) -> Option<i32> {
    attrs
        .get(key)
        .and_then(serde_json::Value::as_i64)
        .and_then(|v| i32::try_from(v).ok())
}

/// Strip an optional union wrapper from a writer Avro schema, yielding the inner branch.
///
/// Iceberg writes optional columns as a 2-branch union with a `null` branch; the inner non-null
/// branch is what carries a nested record / array / map. A non-union schema is returned as-is.
fn unwrap_schema_union(schema: &AvroSchema) -> &AvroSchema {
    if let AvroSchema::Union(union) = schema {
        return non_null_branch(union).unwrap_or(schema);
    }
    schema
}

/// The single non-null branch of a union, if there is exactly one.
fn non_null_branch(union: &UnionSchema) -> Option<&AvroSchema> {
    let mut non_null = union
        .variants()
        .iter()
        .filter(|v| !matches!(v, AvroSchema::Null));
    let first = non_null.next()?;
    if non_null.next().is_some() {
        return None;
    }
    Some(first)
}

// =================================================================================================
// ValueReader: decode a column of Avro values to an Arrow array (schema-aware, recursive)
// =================================================================================================

/// A compiled reader for one (writer subschema, expected Iceberg type) pair. Built once during
/// planning, then applied to a column of decoded Avro values. Nested readers carry their own plans
/// so field-id resolution holds at every level.
enum ValueReader {
    /// A leaf Iceberg primitive (driven by the expected type for promotion / tz-ness).
    Primitive(PrimitiveType),
    /// A nested struct: its own recursive [`StructPlan`] over the writer record.
    Struct {
        plan: Box<StructPlan>,
        arrow_fields: Fields,
    },
    /// A list: a reader for the element type.
    List {
        element_field: Arc<NestedField>,
        element_reader: Box<ValueReader>,
        element_arrow_field: Arc<Field>,
    },
    /// A map: readers for key and value types, and whether the writer wrote the Avro `map` form
    /// (string keys) or the array-of-records form (non-string keys).
    Map {
        key_field: Arc<NestedField>,
        value_field: Arc<NestedField>,
        key_reader: Box<ValueReader>,
        value_reader: Box<ValueReader>,
        key_arrow_field: Arc<Field>,
        value_arrow_field: Arc<Field>,
    },
    /// The `unknown` type: always-null, no physical bytes.
    Unknown,
}

impl ValueReader {
    /// Build a reader from a writer Avro subschema and the expected Iceberg type.
    fn build(writer_schema: &AvroSchema, expected: &Type, input: &ReadPlanInput) -> Result<Self> {
        let writer_schema = unwrap_schema_union(writer_schema);
        match expected {
            Type::Primitive(PrimitiveType::Unknown) => Ok(ValueReader::Unknown),
            Type::Primitive(p) => Ok(ValueReader::Primitive(p.clone())),
            Type::Struct(s) => {
                let AvroSchema::Record(record) = writer_schema else {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!("expected Avro record for Iceberg struct, found {writer_schema:?}"),
                    ));
                };
                let plan = StructPlan::build(record, s, input)?;
                let arrow_fields = struct_arrow_fields(s)?;
                Ok(ValueReader::Struct {
                    plan: Box::new(plan),
                    arrow_fields,
                })
            }
            Type::List(l) => {
                let element_writer = list_element_schema(writer_schema)?;
                let element_reader =
                    ValueReader::build(element_writer, &l.element_field.field_type, input)?;
                Ok(ValueReader::List {
                    element_field: l.element_field.clone(),
                    element_reader: Box::new(element_reader),
                    element_arrow_field: Arc::new(nested_field_to_arrow(&l.element_field)?),
                })
            }
            Type::Map(m) => {
                let (key_reader, value_reader) =
                    map_kv_readers(writer_schema, &m.key_field, &m.value_field, input)?;
                Ok(ValueReader::Map {
                    key_field: m.key_field.clone(),
                    value_field: m.value_field.clone(),
                    key_reader: Box::new(key_reader),
                    value_reader: Box::new(value_reader),
                    key_arrow_field: Arc::new(nested_field_to_arrow(&m.key_field)?),
                    value_arrow_field: Arc::new(nested_field_to_arrow(&m.value_field)?),
                })
            }
            Type::Variant => Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "Avro data-file read of the variant type is not supported yet",
            )),
        }
    }

    /// Convert a column of decoded Avro values to an Arrow array.
    fn to_arrow(&self, values: &[AvroValue]) -> Result<ArrayRef> {
        match self {
            ValueReader::Primitive(p) => primitive_values_to_arrow(p, values),
            ValueReader::Unknown => Ok(Arc::new(arrow_array::NullArray::new(values.len()))),
            ValueReader::Struct { plan, arrow_fields } => {
                struct_values_to_arrow(plan, arrow_fields, values)
            }
            ValueReader::List {
                element_field,
                element_reader,
                element_arrow_field,
            } => list_values_to_arrow(element_field, element_reader, element_arrow_field, values),
            ValueReader::Map {
                key_field,
                value_field,
                key_reader,
                value_reader,
                key_arrow_field,
                value_arrow_field,
            } => map_values_to_arrow(
                key_field,
                value_field,
                key_reader,
                value_reader,
                key_arrow_field,
                value_arrow_field,
                values,
            ),
        }
    }
}

/// The writer element schema of an Avro array (lists). Unwraps an optional-element union.
fn list_element_schema(schema: &AvroSchema) -> Result<&AvroSchema> {
    match schema {
        AvroSchema::Array(array) => Ok(unwrap_schema_union(&array.items)),
        other => Err(Error::new(
            ErrorKind::DataInvalid,
            format!("expected Avro array for Iceberg list, found {other:?}"),
        )),
    }
}

/// Build the (key, value) [`ValueReader`]s of an Iceberg map. Iceberg writes string-keyed maps as
/// an Avro `map` (key is implicitly `string`, value is `MapSchema.types`) and non-string-keyed maps
/// as an Avro `array` of `{key, value}` records.
fn map_kv_readers(
    schema: &AvroSchema,
    key_field: &NestedField,
    value_field: &NestedField,
    input: &ReadPlanInput,
) -> Result<(ValueReader, ValueReader)> {
    match schema {
        AvroSchema::Map(map) => {
            let key_reader = ValueReader::build(&AvroSchema::String, &key_field.field_type, input)?;
            let value_reader = ValueReader::build(
                unwrap_schema_union(&map.types),
                &value_field.field_type,
                input,
            )?;
            Ok((key_reader, value_reader))
        }
        AvroSchema::Array(array) => {
            let AvroSchema::Record(entry) = array.items.as_ref() else {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "expected Avro array-of-records for a non-string-keyed Iceberg map",
                ));
            };
            let key = entry.fields.first().ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "map entry record is missing its key field",
                )
            })?;
            let value = entry.fields.get(1).ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "map entry record is missing its value field",
                )
            })?;
            let key_reader = ValueReader::build(
                unwrap_schema_union(&key.schema),
                &key_field.field_type,
                input,
            )?;
            let value_reader = ValueReader::build(
                unwrap_schema_union(&value.schema),
                &value_field.field_type,
                input,
            )?;
            Ok((key_reader, value_reader))
        }
        other => Err(Error::new(
            ErrorKind::DataInvalid,
            format!("expected Avro map or array for Iceberg map, found {other:?}"),
        )),
    }
}

// =================================================================================================
// Optional-union unwrapping (value side) + type mismatch
// =================================================================================================

/// Strip a union wrapper, yielding `None` for the null branch and `Some(inner)` otherwise.
///
/// Iceberg writes optional columns as a 2-branch union with a `null` branch in either position.
/// `apache_avro` decodes a union to [`AvroValue::Union(idx, box)`]; a bare `Null` (no union) is
/// also treated as `None`.
fn unwrap_optional(value: &AvroValue) -> Option<&AvroValue> {
    match value {
        AvroValue::Union(_, inner) => match inner.as_ref() {
            AvroValue::Null => None,
            other => Some(other),
        },
        AvroValue::Null => None,
        other => Some(other),
    }
}

/// A type error: the decoded Avro value did not match the expected Iceberg type.
fn type_mismatch(expected: &impl std::fmt::Debug, value: &AvroValue) -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        format!("Avro value {value:?} does not match expected Iceberg type {expected:?}"),
    )
}

// =================================================================================================
// Primitive Value → Arrow array (with int→long / float→double promotion, logical types)
// =================================================================================================

fn primitive_values_to_arrow(p: &PrimitiveType, values: &[AvroValue]) -> Result<ArrayRef> {
    macro_rules! collect_opt {
        ($arr:ty, $pat:pat => $val:expr) => {{
            let mut out: Vec<Option<_>> = Vec::with_capacity(values.len());
            for v in values {
                match unwrap_optional(v) {
                    None => out.push(None),
                    Some($pat) => out.push(Some($val)),
                    Some(other) => return Err(type_mismatch(&p, other)),
                }
            }
            Arc::new(<$arr>::from(out)) as ArrayRef
        }};
    }

    let array: ArrayRef = match p {
        PrimitiveType::Boolean => {
            collect_opt!(BooleanArray, AvroValue::Boolean(b) => *b)
        }
        PrimitiveType::Int => {
            collect_opt!(Int32Array, AvroValue::Int(i) => *i)
        }
        // int→long read-time promotion: an Avro INT read into an expected LONG widens.
        PrimitiveType::Long => {
            let mut out: Vec<Option<i64>> = Vec::with_capacity(values.len());
            for v in values {
                match unwrap_optional(v) {
                    None => out.push(None),
                    Some(AvroValue::Long(l)) => out.push(Some(*l)),
                    Some(AvroValue::Int(i)) => out.push(Some(i64::from(*i))),
                    Some(other) => return Err(type_mismatch(&p, other)),
                }
            }
            Arc::new(Int64Array::from(out))
        }
        PrimitiveType::Float => {
            collect_opt!(Float32Array, AvroValue::Float(f) => *f)
        }
        // float→double read-time promotion.
        PrimitiveType::Double => {
            let mut out: Vec<Option<f64>> = Vec::with_capacity(values.len());
            for v in values {
                match unwrap_optional(v) {
                    None => out.push(None),
                    Some(AvroValue::Double(d)) => out.push(Some(*d)),
                    Some(AvroValue::Float(f)) => out.push(Some(f64::from(*f))),
                    Some(other) => return Err(type_mismatch(&p, other)),
                }
            }
            Arc::new(Float64Array::from(out))
        }
        PrimitiveType::Date => {
            // Avro `date` = INT days since epoch → Arrow Date32 (days).
            let mut out: Vec<Option<i32>> = Vec::with_capacity(values.len());
            for v in values {
                match unwrap_optional(v) {
                    None => out.push(None),
                    Some(AvroValue::Date(d)) => out.push(Some(*d)),
                    Some(AvroValue::Int(d)) => out.push(Some(*d)),
                    Some(other) => return Err(type_mismatch(&p, other)),
                }
            }
            Arc::new(Date32Array::from(out))
        }
        PrimitiveType::Time => {
            // Avro `time-micros` = LONG micros since midnight → Arrow Time64(µs).
            let mut out: Vec<Option<i64>> = Vec::with_capacity(values.len());
            for v in values {
                match unwrap_optional(v) {
                    None => out.push(None),
                    Some(AvroValue::TimeMicros(t)) => out.push(Some(*t)),
                    Some(AvroValue::Long(t)) => out.push(Some(*t)),
                    Some(other) => return Err(type_mismatch(&p, other)),
                }
            }
            Arc::new(Time64MicrosecondArray::from(out))
        }
        PrimitiveType::Timestamp => {
            let out = timestamp_micros(p, values)?;
            Arc::new(TimestampMicrosecondArray::from(out))
        }
        PrimitiveType::Timestamptz => {
            let out = timestamp_micros(p, values)?;
            Arc::new(
                TimestampMicrosecondArray::from(out).with_timezone(crate::arrow::UTC_TIME_ZONE),
            )
        }
        PrimitiveType::TimestampNs => {
            let out = timestamp_nanos(p, values)?;
            Arc::new(TimestampNanosecondArray::from(out))
        }
        PrimitiveType::TimestamptzNs => {
            let out = timestamp_nanos(p, values)?;
            Arc::new(TimestampNanosecondArray::from(out).with_timezone(crate::arrow::UTC_TIME_ZONE))
        }
        PrimitiveType::String => {
            let mut builder = StringBuilder::new();
            for v in values {
                match unwrap_optional(v) {
                    None => builder.append_null(),
                    Some(AvroValue::String(s)) => builder.append_value(s),
                    Some(AvroValue::Enum(_, s)) => builder.append_value(s),
                    Some(other) => return Err(type_mismatch(&p, other)),
                }
            }
            Arc::new(builder.finish())
        }
        PrimitiveType::Uuid => {
            // Avro `uuid` = FIXED[16] big-endian → Arrow FixedSizeBinary(16).
            let mut builder = FixedSizeBinaryBuilder::new(16);
            for v in values {
                match unwrap_optional(v) {
                    None => builder.append_null(),
                    Some(AvroValue::Uuid(u)) => builder
                        .append_value(u.as_bytes())
                        .map_err(|e| fixed_err(e, 16))?,
                    Some(AvroValue::Fixed(16, bytes)) => {
                        builder.append_value(bytes).map_err(|e| fixed_err(e, 16))?
                    }
                    Some(other) => return Err(type_mismatch(&p, other)),
                }
            }
            Arc::new(builder.finish())
        }
        PrimitiveType::Fixed(len) => {
            let width = i32::try_from(*len).map_err(|_| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("fixed length {len} exceeds i32 range"),
                )
            })?;
            let mut builder = FixedSizeBinaryBuilder::new(width);
            for v in values {
                match unwrap_optional(v) {
                    None => builder.append_null(),
                    Some(AvroValue::Fixed(_, bytes)) => builder
                        .append_value(bytes)
                        .map_err(|e| fixed_err(e, width))?,
                    Some(AvroValue::Bytes(bytes)) => builder
                        .append_value(bytes)
                        .map_err(|e| fixed_err(e, width))?,
                    Some(other) => return Err(type_mismatch(&p, other)),
                }
            }
            Arc::new(builder.finish())
        }
        PrimitiveType::Binary => {
            let mut builder = LargeBinaryBuilder::new();
            for v in values {
                match unwrap_optional(v) {
                    None => builder.append_null(),
                    Some(AvroValue::Bytes(bytes)) => builder.append_value(bytes),
                    Some(AvroValue::Fixed(_, bytes)) => builder.append_value(bytes),
                    Some(other) => return Err(type_mismatch(&p, other)),
                }
            }
            Arc::new(builder.finish())
        }
        PrimitiveType::Decimal { precision, scale } => {
            decimal_values_to_arrow(*precision, *scale, values)?
        }
        PrimitiveType::Unknown => {
            // `unknown` has no physical bytes; an always-null Null array. (Normally routed via
            // `ValueReader::Unknown`, but kept here so the match is exhaustive.)
            Arc::new(arrow_array::NullArray::new(values.len()))
        }
    };
    Ok(array)
}

/// Extract micros from `timestamp(tz)`-typed Avro values.
fn timestamp_micros(p: &PrimitiveType, values: &[AvroValue]) -> Result<Vec<Option<i64>>> {
    let mut out: Vec<Option<i64>> = Vec::with_capacity(values.len());
    for v in values {
        match unwrap_optional(v) {
            None => out.push(None),
            Some(AvroValue::TimestampMicros(t)) => out.push(Some(*t)),
            Some(AvroValue::LocalTimestampMicros(t)) => out.push(Some(*t)),
            Some(AvroValue::Long(t)) => out.push(Some(*t)),
            Some(other) => return Err(type_mismatch(&p, other)),
        }
    }
    Ok(out)
}

/// Extract nanos from `timestamp_ns(tz)`-typed Avro values.
fn timestamp_nanos(p: &PrimitiveType, values: &[AvroValue]) -> Result<Vec<Option<i64>>> {
    let mut out: Vec<Option<i64>> = Vec::with_capacity(values.len());
    for v in values {
        match unwrap_optional(v) {
            None => out.push(None),
            Some(AvroValue::TimestampNanos(t)) => out.push(Some(*t)),
            Some(AvroValue::LocalTimestampNanos(t)) => out.push(Some(*t)),
            Some(AvroValue::Long(t)) => out.push(Some(*t)),
            Some(other) => return Err(type_mismatch(&p, other)),
        }
    }
    Ok(out)
}

/// Build a Decimal128 array. Avro decimal = two's-complement **big-endian** unscaled integer
/// (FIXED or BYTES); the scale comes from the logical type.
fn decimal_values_to_arrow(precision: u32, scale: u32, values: &[AvroValue]) -> Result<ArrayRef> {
    let arrow_precision = u8::try_from(precision).map_err(|_| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("decimal precision {precision} exceeds Arrow Decimal128 range"),
        )
    })?;
    let arrow_scale = i8::try_from(scale).map_err(|_| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("decimal scale {scale} exceeds Arrow Decimal128 range"),
        )
    })?;

    let mut out: Vec<Option<i128>> = Vec::with_capacity(values.len());
    for v in values {
        match unwrap_optional(v) {
            None => out.push(None),
            Some(AvroValue::Decimal(d)) => {
                // `Vec<u8>::try_from(&Decimal)` yields the sign-extended two's-complement
                // big-endian unscaled integer (Avro decimal wire form).
                let be: Vec<u8> = d.try_into().map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        "Failed to extract Avro decimal unscaled bytes",
                    )
                    .with_source(e)
                })?;
                out.push(Some(be_bytes_to_i128(&be)?));
            }
            Some(AvroValue::Fixed(_, bytes)) | Some(AvroValue::Bytes(bytes)) => {
                out.push(Some(be_bytes_to_i128(bytes)?));
            }
            Some(other) => {
                return Err(type_mismatch(
                    &PrimitiveType::Decimal { precision, scale },
                    other,
                ));
            }
        }
    }

    let array = Decimal128Array::from(out)
        .with_precision_and_scale(arrow_precision, arrow_scale)
        .map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Failed to build Decimal128 array (precision {precision}, scale {scale})"),
            )
            .with_source(e)
        })?;
    Ok(Arc::new(array))
}

/// Two's-complement big-endian bytes → i128 (with sign extension).
fn be_bytes_to_i128(bytes: &[u8]) -> Result<i128> {
    if bytes.len() > 16 {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "decimal unscaled value is {} bytes, exceeds the 16-byte Decimal128 limit",
                bytes.len()
            ),
        ));
    }
    if bytes.is_empty() {
        return Ok(0);
    }
    let negative = bytes[0] & 0x80 != 0;
    let mut buf = if negative { [0xFFu8; 16] } else { [0u8; 16] };
    let start = 16 - bytes.len();
    buf[start..].copy_from_slice(bytes);
    Ok(i128::from_be_bytes(buf))
}

fn fixed_err(e: arrow_schema::ArrowError, width: i32) -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        format!("Avro fixed/bytes value did not match Arrow FixedSizeBinary({width})"),
    )
    .with_source(e)
}

// =================================================================================================
// Nested: struct / list / map (all by-field-id via the recursive plan / nested readers)
// =================================================================================================

/// Convert a column of struct values using the nested [`StructPlan`] (field-id resolution at this
/// level), producing a [`StructArray`].
fn struct_values_to_arrow(
    plan: &StructPlan,
    arrow_fields: &Fields,
    values: &[AvroValue],
) -> Result<ArrayRef> {
    // Per File-sourced child, the per-row Avro value (null parent row → Null for every child).
    let mut child_acc: Vec<Vec<AvroValue>> = (0..plan.num_file_children())
        .map(|_| Vec::with_capacity(values.len()))
        .collect();
    let mut null_buffer = Vec::with_capacity(values.len());

    for v in values {
        match unwrap_optional(v) {
            None => {
                null_buffer.push(false);
                plan.push_row(None, &mut child_acc);
            }
            Some(AvroValue::Record(row)) => {
                null_buffer.push(true);
                plan.push_row(Some(row), &mut child_acc);
            }
            Some(other) => return Err(type_mismatch(arrow_fields, other)),
        }
    }

    // Assemble all children (File-sourced + constant / row-position) in plan order.
    let num_rows = values.len();
    let mut file_arrays = plan.child_arrays(&child_acc)?.into_iter();
    let mut child_arrays: Vec<ArrayRef> = Vec::with_capacity(plan.children.len());
    for col in &plan.children {
        let array = match &col.source {
            ColumnSource::File { .. } => file_arrays
                .next()
                .expect("one File array per File-sourced child"),
            ColumnSource::Constant(lit) => {
                constant_array(&col.field.field_type, lit.as_ref(), num_rows)?
            }
            ColumnSource::RowPosition => {
                // A nested struct has no independent row position; Java fills nested `_pos` from 0.
                let values: Vec<i64> = (0..num_rows)
                    .map(|i| i64::try_from(i).unwrap_or(i64::MAX))
                    .collect();
                Arc::new(Int64Array::from(values))
            }
        };
        child_arrays.push(array);
    }

    let nulls = arrow_buffer::NullBuffer::from(null_buffer);
    let array =
        StructArray::try_new(arrow_fields.clone(), child_arrays, Some(nulls)).map_err(|e| {
            Error::new(ErrorKind::DataInvalid, "Failed to build Arrow StructArray").with_source(e)
        })?;
    Ok(Arc::new(array))
}

fn list_values_to_arrow(
    element_field: &NestedField,
    element_reader: &ValueReader,
    element_arrow_field: &Arc<Field>,
    values: &[AvroValue],
) -> Result<ArrayRef> {
    // Flatten elements, recording offsets and the list-level null buffer.
    let mut flat_elements: Vec<AvroValue> = Vec::new();
    let mut offsets: Vec<i32> = vec![0];
    let mut null_buffer = Vec::with_capacity(values.len());
    let mut running: i32 = 0;

    for v in values {
        match unwrap_optional(v) {
            None => {
                null_buffer.push(false);
                offsets.push(running);
            }
            Some(AvroValue::Array(items)) => {
                null_buffer.push(true);
                for item in items {
                    flat_elements.push(item.clone());
                    running = running.checked_add(1).ok_or_else(|| {
                        Error::new(ErrorKind::DataInvalid, "list offset overflowed i32")
                    })?;
                }
                offsets.push(running);
            }
            Some(other) => return Err(type_mismatch(&element_field.name, other)),
        }
    }

    let element_array = element_reader.to_arrow(&flat_elements)?;
    let offset_buffer = arrow_buffer::OffsetBuffer::new(offsets.into());
    let nulls = arrow_buffer::NullBuffer::from(null_buffer);
    let array = ListArray::try_new(
        element_arrow_field.clone(),
        offset_buffer,
        element_array,
        Some(nulls),
    )
    .map_err(|e| {
        Error::new(ErrorKind::DataInvalid, "Failed to build Arrow ListArray").with_source(e)
    })?;
    Ok(Arc::new(array))
}

#[allow(clippy::too_many_arguments)]
fn map_values_to_arrow(
    key_field: &NestedField,
    value_field: &NestedField,
    key_reader: &ValueReader,
    value_reader: &ValueReader,
    key_arrow_field: &Arc<Field>,
    value_arrow_field: &Arc<Field>,
    values: &[AvroValue],
) -> Result<ArrayRef> {
    let mut flat_keys: Vec<AvroValue> = Vec::new();
    let mut flat_values: Vec<AvroValue> = Vec::new();
    let mut offsets: Vec<i32> = vec![0];
    let mut null_buffer = Vec::with_capacity(values.len());
    let mut running: i32 = 0;

    let push_entry = |running: &mut i32| -> Result<()> {
        *running = running
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorKind::DataInvalid, "map offset overflowed i32"))?;
        Ok(())
    };

    for v in values {
        match unwrap_optional(v) {
            None => {
                null_buffer.push(false);
                offsets.push(running);
            }
            // Iceberg string-keyed maps are written as an Avro `map`.
            Some(AvroValue::Map(entries)) => {
                null_buffer.push(true);
                for (k, val) in entries {
                    flat_keys.push(AvroValue::String(k.clone()));
                    flat_values.push(val.clone());
                    push_entry(&mut running)?;
                }
                offsets.push(running);
            }
            // Iceberg non-string-keyed maps are written as an Avro array of key/value records;
            // the two record fields carry the key-id / value-id, read positionally as Java does
            // (entry record = exactly [key, value]).
            Some(AvroValue::Array(entries)) => {
                null_buffer.push(true);
                for entry in entries {
                    let AvroValue::Record(kv) = entry else {
                        return Err(type_mismatch(&value_field.name, entry));
                    };
                    let key = kv
                        .first()
                        .map(|(_, v)| v.clone())
                        .ok_or_else(|| type_mismatch(&key_field.name, entry))?;
                    let value = kv
                        .get(1)
                        .map(|(_, v)| v.clone())
                        .ok_or_else(|| type_mismatch(&value_field.name, entry))?;
                    flat_keys.push(key);
                    flat_values.push(value);
                    push_entry(&mut running)?;
                }
                offsets.push(running);
            }
            Some(other) => return Err(type_mismatch(&value_field.name, other)),
        }
    }

    let key_array = key_reader.to_arrow(&flat_keys)?;
    let value_array = value_reader.to_arrow(&flat_values)?;

    let entry_struct = StructArray::try_new(
        Fields::from(vec![
            key_arrow_field.as_ref().clone(),
            value_arrow_field.as_ref().clone(),
        ]),
        vec![key_array, value_array],
        None,
    )
    .map_err(|e| {
        Error::new(
            ErrorKind::DataInvalid,
            "Failed to build map entry StructArray",
        )
        .with_source(e)
    })?;

    let entries_field = Arc::new(Field::new(
        crate::arrow::DEFAULT_MAP_FIELD_NAME,
        DataType::Struct(Fields::from(vec![
            key_arrow_field.as_ref().clone(),
            value_arrow_field.as_ref().clone(),
        ])),
        false,
    ));
    let offset_buffer = arrow_buffer::OffsetBuffer::new(offsets.into());
    let nulls = arrow_buffer::NullBuffer::from(null_buffer);
    let array = MapArray::try_new(
        entries_field,
        offset_buffer,
        entry_struct,
        Some(nulls),
        false,
    )
    .map_err(|e| {
        Error::new(ErrorKind::DataInvalid, "Failed to build Arrow MapArray").with_source(e)
    })?;
    Ok(Arc::new(array))
}

/// The Arrow [`Fields`] of an Iceberg struct (via the canonical schema converter).
fn struct_arrow_fields(s: &StructType) -> Result<Fields> {
    match type_to_arrow_type(&Type::Struct(s.clone()))? {
        DataType::Struct(fields) => Ok(fields),
        other => Err(Error::new(
            ErrorKind::Unexpected,
            format!("expected struct arrow type, found {other:?}"),
        )),
    }
}

/// Convert an Iceberg [`NestedField`] to its Arrow [`Field`] (id metadata + nullability), reusing
/// the canonical schema converter so list/map child fields match the rest of the crate.
fn nested_field_to_arrow(field: &NestedField) -> Result<Field> {
    let data_type = type_to_arrow_type(&field.field_type)?;
    let mut arrow_field = Field::new(&field.name, data_type, !field.required);
    arrow_field.set_metadata(HashMap::from([(
        parquet::arrow::PARQUET_FIELD_ID_META_KEY.to_string(),
        field.id.to_string(),
    )]));
    Ok(arrow_field)
}

// =================================================================================================
// Constant (missing-column) arrays
// =================================================================================================

/// Build a constant Arrow array of `num_rows` for a missing column: either a repeated literal or,
/// when `lit` is `None`, all-null.
fn constant_array(expected: &Type, lit: Option<&Literal>, num_rows: usize) -> Result<ArrayRef> {
    let data_type = type_to_arrow_type(expected)?;
    let prim_lit: Option<PrimitiveLiteral> = match lit {
        None => None,
        Some(Literal::Primitive(p)) => Some(p.clone()),
        Some(other) => {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "constant default for nested type is not supported in the Avro data reader \
                     (got {other:?})"
                ),
            ));
        }
    };
    crate::arrow::create_primitive_array_repeated(&data_type, &prim_lit, num_rows)
}

#[cfg(test)]
mod tests {
    include!("avro_reader_tests.rs");
}
