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

//! ORC **data-file** reader: an [`orc-rust`](orc_rust) Arrow stream → Iceberg-typed Arrow
//! [`RecordBatch`] converter that mirrors Java 1.10.0 `org.apache.iceberg.orc` —
//! `ORCSchemaUtil.buildOrcProjection` (the by-field-id projection + promotion contract) and
//! `GenericOrcReader` (the per-type physical read dispatch).
//!
//! # Why we parse the footer ourselves
//!
//! `orc-rust` decodes ORC data in any codec, but it **discards ORC type attributes** (its `proto`
//! module is private and `RootDataType::from_proto` drops `attributes`). Iceberg stores the
//! `field-id` of each column as an ORC type attribute (`iceberg.id`), exactly as Java writes it via
//! `ORCSchemaUtil`. So to resolve **by field-id** — the only correct resolution, never by name — we
//! hand-parse the ORC Footer ([`footer`]) to recover the `iceberg.id` (and `iceberg.required`)
//! attribute of every type, then build a `field-id → ORC column index` map.
//!
//! # The projection (parity with `ORCSchemaUtil.buildOrcProjection`)
//!
//! For each **expected** (projection) field, we look it up by id in the file's id→type map:
//!
//! - **present** → project that ORC column index. The expected primitive must be the *same* ORC
//!   category, OR a Java-sanctioned read-time promotion: Iceberg `long` over ORC `int`, Iceberg
//!   `double` over ORC `float`, Iceberg `decimal` over ORC `decimal` **iff same scale and expected
//!   precision > file precision**. Anything else is a hard `Can not promote …` error.
//! - **missing** (id not in file): if the field is **required** and has no `initial-default` → hard
//!   `Missing required field` error; if it has an `initial-default` → `FeatureUnsupported` (Java
//!   throws `UnsupportedOperationException` — ORC cannot read non-null defaults); else (optional) →
//!   synthesize an **all-null** column.
//!
//! Output columns are in **expected** field order (Java reorders to the projection), and every
//! output `Field` is stamped with `parquet::arrow::PARQUET_FIELD_ID_META_KEY` (via
//! [`schema_to_arrow_schema`]) so the U2 `RecordBatchTransformer` and the delete-predicate evaluator
//! can resolve it by id.
//!
//! # Physical read dispatch (parity with `GenericOrcReader`)
//!
//! Dispatch is keyed on the **resolved Iceberg type**, not the ORC category — because several
//! Iceberg types share one ORC physical category and are disambiguated only by an Iceberg attribute
//! Java writes (`iceberg.long-type=TIME`, `iceberg.binary-type=UUID|FIXED`, the timestamp tz-ness).
//! The expected Iceberg type already carries that distinction, so we drive off it (the same strategy
//! [`crate::arrow::avro_reader`] uses). Concretely we convert the `orc-rust` Arrow output to the
//! canonical Iceberg Arrow type ([`crate::arrow::type_to_arrow_type`]):
//!
//! | Iceberg type | orc-rust Arrow out | converted to |
//! |---|---|---|
//! | `long` / `int` / `float` / `double` / `boolean` / `string` / `date` | matching primitive | as-is (+ int→long / float→double promotion) |
//! | `time` | `Int64` (ORC LONG micros) | `Time64(µs)` |
//! | `timestamp` | `Timestamp(ns, None)` | `Timestamp(µs, None)` (ns→µs down-cast) |
//! | `timestamptz` | `Timestamp(ns, Some(UTC))` | `Timestamp(µs, Some("+00:00"))` |
//! | `decimal(p,s)` | `Decimal128(p',s)` | `Decimal128(p,s)` (precision widened on promotion) |
//! | `uuid` | `Binary` | `FixedSizeBinary(16)` |
//! | `fixed[L]` | `Binary` | `FixedSizeBinary(L)` |
//! | `binary` | `Binary` | `LargeBinary` |
//!
//! # Scope (v1)
//!
//! - **Footer codecs:** only uncompressed and **ZLIB** (ORC ZLIB = *raw* DEFLATE). The actual data
//!   decode is `orc-rust`'s and supports every codec; only **our footer parse** is codec-limited —
//!   a SNAPPY/ZSTD/LZ4/LZO-compressed *footer* yields a clear [`ErrorKind::FeatureUnsupported`].
//! - **Top-level structs of primitives + logical types** are covered. Nested struct/list/map
//!   schema-evolution-by-id and the V3 `variant` / `geometry` / `geography` types are deferred.
//! - Files **without** `iceberg.id` attributes (name-mapping fallback) error loudly rather than
//!   resolving by name.
//! - Reserved `_pos` / `_file` / partition-constant columns and id→constant precedence are the U2
//!   `RecordBatchTransformer`'s job (as for the Parquet/Avro paths); this reader only supplies file
//!   columns and optional-missing nulls.

// U2 (scan-path wiring) is DONE: `arrow::reader`'s `process_orc_file_scan_task` consumes
// `read_orc_data_file` in the production scan path, and `read_orc_data_bytes` (the sync core) is
// called both by `read_orc_data_file` and by the offline tests. Both stay `pub(crate)`; this file
// carries no module-level `dead_code` allow — the compiler is left to flag any genuinely-dead item.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{FixedSizeBinaryBuilder, LargeBinaryBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, TimestampNanosecondType};
use arrow_array::{
    Array, ArrayRef, BinaryArray, Decimal128Array, Int64Array, NullArray, RecordBatch,
    Time64MicrosecondArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Schema as ArrowSchema, TimeUnit};
use bytes::Bytes;
use orc_rust::ArrowReaderBuilder;
use orc_rust::projection::ProjectionMask;
use orc_rust::schema::RootDataType;

use crate::arrow::{UTC_TIME_ZONE, schema_to_arrow_schema, type_to_arrow_type};
use crate::io::InputFile;
use crate::spec::{NestedField, PrimitiveType, Schema, Type};
use crate::{Error, ErrorKind, Result};

mod footer;

use footer::{OrcCategory, OrcFileType, parse_footer};

// =================================================================================================
// Public(crate) entry points
// =================================================================================================

/// Read an ORC data file from an [`InputFile`] and return its rows as Arrow [`RecordBatch`]es.
///
/// The full file is fetched into memory, then decoded off the async executor via
/// [`crate::runtime::spawn_blocking`] (ORC decode is CPU-bound), exactly as
/// [`crate::arrow::avro_reader::read_avro_data_file`] does.
///
/// `expected` is the projection: only its fields appear in the output, in its field order, and the
/// output [`RecordBatch`] schema equals [`schema_to_arrow_schema(expected)`](schema_to_arrow_schema).
pub(crate) async fn read_orc_data_file(
    input: &InputFile,
    expected: Arc<Schema>,
    batch_size: usize,
) -> Result<Vec<RecordBatch>> {
    let bytes = input.read().await?;
    crate::runtime::spawn_blocking(move || {
        read_orc_data_bytes(&bytes, expected.as_ref(), batch_size)
    })
    .await
}

/// Decode ORC `bytes` into Arrow [`RecordBatch`]es against the `expected` projection schema.
///
/// The synchronous core; [`read_orc_data_file`] is the async/[`InputFile`] wrapper. Exposed
/// `pub(crate)` so callers holding the bytes (and offline tests) drive it without an [`InputFile`].
pub(crate) fn read_orc_data_bytes(
    bytes: &[u8],
    expected: &Schema,
    batch_size: usize,
) -> Result<Vec<RecordBatch>> {
    if batch_size == 0 {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            "ORC data-file batch_size must be greater than zero",
        ));
    }

    // (1) Hand-parse the footer to recover the `field-id → ORC type` map (with `iceberg.id`).
    let file_types = parse_footer(bytes)?;

    // (2) Open `orc-rust` over the same bytes (it re-reads the footer itself for the data streams).
    let builder = ArrowReaderBuilder::try_new(Bytes::copy_from_slice(bytes)).map_err(|e| {
        Error::new(
            ErrorKind::DataInvalid,
            "Failed to open ORC data file (orc-rust could not read the file metadata)",
        )
        .with_source(e)
    })?;
    let root_data_type = builder.file_metadata().root_data_type().clone();

    // (3) Build the by-field-id projection plan (parity with `ORCSchemaUtil.buildOrcProjection`).
    let plan = ProjectionPlan::build(expected, &file_types)?;
    let arrow_schema = Arc::new(schema_to_arrow_schema(expected)?);

    // (4) Project `orc-rust` by the ORC column indices we need, then decode.
    let projection = ProjectionMask::roots(&root_data_type, plan.projected_orc_indices());
    let reader = builder
        .with_batch_size(batch_size)
        .with_projection(projection)
        .build();

    // The `orc-rust` output batch's field order is the *file* (ORC) column order filtered by the
    // projection, named by ORC field name. We index it by ORC column index → output column.
    let mut out_batches = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                "Failed to decode an ORC record batch",
            )
            .with_source(e)
        })?;
        let orc_index_to_array = orc_batch_by_index(&batch, &root_data_type, &plan)?;
        out_batches.push(plan.assemble(&orc_index_to_array, batch.num_rows(), &arrow_schema)?);
    }

    // An empty file still yields a single empty batch so the caller sees the schema.
    if out_batches.is_empty() {
        out_batches.push(plan.assemble(&HashMap::new(), 0, &arrow_schema)?);
    }

    Ok(out_batches)
}

// =================================================================================================
// Projection plan (id→orc-index, with Java promotion rules)
// =================================================================================================

/// One projected output column and how to source it.
struct PlanColumn {
    /// The expected Iceberg field for this output position.
    field: Arc<NestedField>,
    source: ColumnSource,
}

/// Where a projected output column's values come from.
enum ColumnSource {
    /// Read the `orc-rust` array at this ORC column index and convert it to `expected`.
    Orc {
        orc_index: usize,
        expected: PrimitiveType,
    },
    /// Synthesize an all-null column (optional field absent from the file).
    AllNull,
}

/// The compiled by-field-id projection over the top-level struct.
struct ProjectionPlan {
    columns: Vec<PlanColumn>,
}

impl ProjectionPlan {
    /// Build the projection, mirroring `ORCSchemaUtil.buildOrcProjection` for a top-level struct of
    /// primitives. Nested compound types are rejected (deferred to a later increment).
    fn build(expected: &Schema, file_types: &HashMap<i32, OrcFileType>) -> Result<Self> {
        let mut columns = Vec::with_capacity(expected.as_struct().fields().len());
        for field in expected.as_struct().fields() {
            let Type::Primitive(expected_prim) = field.field_type.as_ref() else {
                return Err(Error::new(
                    ErrorKind::FeatureUnsupported,
                    format!(
                        "ORC data-file read of nested type for field '{}' is not supported yet \
                         (only top-level primitive/logical columns)",
                        field.name
                    ),
                ));
            };

            let source = match file_types.get(&field.id) {
                Some(file_type) => {
                    // PRIMITIVE PRESENT: require same type or a sanctioned promotion.
                    check_read_compatible(expected_prim, file_type, &field.name)?;
                    ColumnSource::Orc {
                        orc_index: file_type.orc_column_index,
                        expected: expected_prim.clone(),
                    }
                }
                None => missing_column_source(field)?,
            };
            columns.push(PlanColumn {
                field: field.clone(),
                source,
            });
        }
        Ok(ProjectionPlan { columns })
    }

    /// The set of ORC column indices to hand `ProjectionMask::roots`.
    fn projected_orc_indices(&self) -> Vec<usize> {
        self.columns
            .iter()
            .filter_map(|c| match &c.source {
                ColumnSource::Orc { orc_index, .. } => Some(*orc_index),
                ColumnSource::AllNull => None,
            })
            .collect()
    }

    /// Assemble the expected-order output batch from the `orc_index → array` map.
    fn assemble(
        &self,
        orc_index_to_array: &HashMap<usize, ArrayRef>,
        num_rows: usize,
        arrow_schema: &Arc<ArrowSchema>,
    ) -> Result<RecordBatch> {
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(self.columns.len());
        for col in &self.columns {
            let array = match &col.source {
                ColumnSource::Orc {
                    orc_index,
                    expected,
                } => {
                    let raw = orc_index_to_array.get(orc_index).ok_or_else(|| {
                        Error::new(
                            ErrorKind::Unexpected,
                            format!(
                                "ORC column index {orc_index} (field '{}') missing from the \
                                 decoded batch",
                                col.field.name
                            ),
                        )
                    })?;
                    convert_orc_array(expected, raw, &col.field.name)?
                }
                ColumnSource::AllNull => {
                    let data_type = type_to_arrow_type(&col.field.field_type)?;
                    arrow_array::new_null_array(&data_type, num_rows)
                }
            };
            arrays.push(array);
        }

        RecordBatch::try_new(arrow_schema.clone(), arrays).map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                "Failed to assemble Arrow RecordBatch from decoded ORC columns",
            )
            .with_source(e)
        })
    }
}

/// Resolve a missing column's fill, in Java's exact priority order
/// (`ORCSchemaUtil.buildOrcProjection`, the `else` branch).
fn missing_column_source(field: &NestedField) -> Result<ColumnSource> {
    // Java: `checkArgument(isRequired ? initialDefault != null : true, "Missing required field …")`.
    if field.required && field.initial_default.is_none() {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Missing required field: {}", field.name),
        ));
    }
    // Java: a non-null `initialDefault` → UnsupportedOperationException (ORC can't read defaults).
    if field.initial_default.is_some() {
        return Err(Error::new(
            ErrorKind::FeatureUnsupported,
            format!(
                "ORC cannot read default value for field {} (non-null initial-default)",
                field.name
            ),
        ));
    }
    // Optional, no default → synthesize an all-null column.
    Ok(ColumnSource::AllNull)
}

/// Enforce the Java read-compatibility contract for a present primitive
/// (`ORCSchemaUtil.getPromotedType` / `isSameType`).
fn check_read_compatible(
    expected: &PrimitiveType,
    file_type: &OrcFileType,
    field_name: &str,
) -> Result<()> {
    if is_read_compatible(expected, file_type) {
        Ok(())
    } else {
        Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Can not promote {:?} type to {} (field '{}')",
                file_type.category, expected, field_name
            ),
        ))
    }
}

/// Whether the expected Iceberg primitive can read the ORC physical category — same physical type or
/// a Java-sanctioned promotion (`int→long`, `float→double`, decimal precision-widen at same scale).
fn is_read_compatible(expected: &PrimitiveType, file: &OrcFileType) -> bool {
    use OrcCategory as C;
    match expected {
        // Promotions.
        PrimitiveType::Long => matches!(file.category, C::Long | C::Int),
        PrimitiveType::Double => matches!(file.category, C::Double | C::Float),
        PrimitiveType::Decimal { precision, scale } => match file.category {
            C::Decimal => {
                // Java is laxer here: `buildOrcProjection`'s `isSameType` accepts ANY DECIMAL→DECIMAL
                // (TYPE_MAPPING has no precision/scale check) and `DecimalReader` then `setScale()`s
                // to the FILE scale, silently ignoring a requested scale mismatch (a latent Java bug).
                // We are deliberately stricter in the SAFE direction: require the same scale, and
                // accept identity (`==`) or a precision-widen promotion (`getPromotedType`: same scale
                // AND expected precision > file precision; `>=` folds identity with promotion).
                // Unreachable on real Iceberg-written ORC, where the file always carries the table's
                // declared precision/scale (see `test_decimal_scale_mismatch_is_stricter_than_java`).
                let same_scale = file.scale == *scale;
                same_scale && *precision >= file.precision
            }
            _ => false,
        },
        // Same-type reads.
        PrimitiveType::Boolean => matches!(file.category, C::Boolean),
        // Java's TYPE_MAPPING maps INTEGER → {BYTE, SHORT, INT}, so Java accepts reading an ORC BYTE
        // or SHORT column as Iceberg `int`. We accept only ORC INT: Iceberg always writes `int` as
        // ORC INT (and the reader requires `iceberg.id`, so non-Iceberg ORC is out of scope), making
        // the BYTE/SHORT→int widening unreachable on real Iceberg files. Narrower than Java but never
        // observably divergent on an Iceberg-written file.
        PrimitiveType::Int => matches!(file.category, C::Int),
        PrimitiveType::Float => matches!(file.category, C::Float),
        PrimitiveType::String => matches!(file.category, C::String | C::Varchar | C::Char),
        PrimitiveType::Date => matches!(file.category, C::Date),
        // ORC LONG, disambiguated by the expected Iceberg type (Java writes iceberg.long-type=TIME).
        PrimitiveType::Time => matches!(file.category, C::Long),
        PrimitiveType::Timestamp => matches!(file.category, C::Timestamp),
        PrimitiveType::Timestamptz => matches!(file.category, C::TimestampInstant),
        // ns timestamps share the ORC physical category; tz-ness drives the split.
        PrimitiveType::TimestampNs => matches!(file.category, C::Timestamp),
        PrimitiveType::TimestamptzNs => matches!(file.category, C::TimestampInstant),
        // ORC BINARY, disambiguated by the expected Iceberg type (iceberg.binary-type).
        PrimitiveType::Binary => matches!(file.category, C::Binary),
        PrimitiveType::Uuid => matches!(file.category, C::Binary),
        PrimitiveType::Fixed(_) => matches!(file.category, C::Binary),
        // `unknown` has no physical ORC column; it never resolves to a present file type.
        PrimitiveType::Unknown => false,
    }
}

// =================================================================================================
// orc-rust Arrow output → Iceberg-typed Arrow
// =================================================================================================

/// Index the `orc-rust` output batch by ORC column index, restricted to the columns the plan
/// projected. `orc-rust` orders/names its output by the *file* (ORC) schema (filtered by the
/// projection), so we walk the projected root's children to recover each child's ORC column index.
fn orc_batch_by_index(
    batch: &RecordBatch,
    root_data_type: &RootDataType,
    plan: &ProjectionPlan,
) -> Result<HashMap<usize, ArrayRef>> {
    // The projected root preserves file order; its children carry their ORC `column_index`.
    let projected = root_data_type.project(&ProjectionMask::roots(
        root_data_type,
        plan.projected_orc_indices(),
    ));
    let children = projected.children();
    if children.len() != batch.num_columns() {
        return Err(Error::new(
            ErrorKind::Unexpected,
            format!(
                "ORC projected schema has {} columns but decoded batch has {}",
                children.len(),
                batch.num_columns()
            ),
        ));
    }

    let mut map = HashMap::with_capacity(children.len());
    for (pos, child) in children.iter().enumerate() {
        map.insert(child.data_type().column_index(), batch.column(pos).clone());
    }
    Ok(map)
}

/// Convert one `orc-rust` Arrow array to the canonical Iceberg Arrow type for `expected`.
fn convert_orc_array(
    expected: &PrimitiveType,
    raw: &ArrayRef,
    field_name: &str,
) -> Result<ArrayRef> {
    match expected {
        // Identity reads — `orc-rust` already produced the right Arrow primitive.
        PrimitiveType::Boolean
        | PrimitiveType::Int
        | PrimitiveType::Float
        | PrimitiveType::String
        | PrimitiveType::Date => Ok(raw.clone()),

        // int→long promotion: ORC may have stored INT; widen to Int64 if needed.
        PrimitiveType::Long => match raw.data_type() {
            DataType::Int64 => Ok(raw.clone()),
            DataType::Int32 => {
                let ints = raw.as_primitive::<arrow_array::types::Int32Type>();
                let widened: Int64Array = ints.iter().map(|v| v.map(i64::from)).collect();
                Ok(Arc::new(widened))
            }
            other => Err(physical_mismatch("long", other, field_name)),
        },

        // float→double promotion: ORC may have stored FLOAT; widen to Float64 if needed.
        PrimitiveType::Double => match raw.data_type() {
            DataType::Float64 => Ok(raw.clone()),
            DataType::Float32 => {
                let floats = raw.as_primitive::<arrow_array::types::Float32Type>();
                let widened: arrow_array::Float64Array =
                    floats.iter().map(|v| v.map(f64::from)).collect();
                Ok(Arc::new(widened))
            }
            other => Err(physical_mismatch("double", other, field_name)),
        },

        // ORC LONG micros tagged iceberg.long-type=TIME → Arrow Time64(µs).
        PrimitiveType::Time => match raw.data_type() {
            DataType::Int64 => {
                let longs = raw.as_primitive::<arrow_array::types::Int64Type>();
                let times: Time64MicrosecondArray = longs.iter().collect();
                Ok(Arc::new(times))
            }
            other => Err(physical_mismatch("time", other, field_name)),
        },

        // ns→µs down-cast (Iceberg is micros; `orc-rust` yields nanos), tz None.
        PrimitiveType::Timestamp => Ok(timestamp_ns_to_us(raw, None, field_name)?),
        // ns→µs down-cast, tz "+00:00".
        PrimitiveType::Timestamptz => Ok(timestamp_ns_to_us(raw, Some(UTC_TIME_ZONE), field_name)?),
        // ns timestamps keep nanosecond precision; only re-stamp the tz to the canonical literal.
        PrimitiveType::TimestampNs | PrimitiveType::TimestamptzNs => Ok(raw.clone()),

        PrimitiveType::Decimal { precision, scale } => {
            convert_decimal(raw, *precision, *scale, field_name)
        }

        // ORC BINARY tagged iceberg.binary-type=UUID → FixedSizeBinary(16).
        PrimitiveType::Uuid => binary_to_fixed(raw, 16, field_name),
        // ORC BINARY tagged iceberg.binary-type=FIXED → FixedSizeBinary(L).
        PrimitiveType::Fixed(len) => {
            let width = i32::try_from(*len).map_err(|_| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("fixed length {len} exceeds i32 range (field '{field_name}')"),
                )
            })?;
            binary_to_fixed(raw, width, field_name)
        }
        // ORC BINARY → Arrow LargeBinary (the canonical Iceberg binary Arrow type).
        PrimitiveType::Binary => binary_to_large_binary(raw, field_name),

        // `unknown` never has a physical ORC column (handled in `is_read_compatible`).
        PrimitiveType::Unknown => Ok(Arc::new(NullArray::new(raw.len()))),
    }
}

/// Down-cast an `orc-rust` `Timestamp(Nanosecond, _)` array to `Timestamp(Microsecond, tz)`,
/// matching Java's micros reconstruction (floor-division toward negative infinity is preserved by
/// `div_euclid`).
fn timestamp_ns_to_us(raw: &ArrayRef, tz: Option<&str>, field_name: &str) -> Result<ArrayRef> {
    let DataType::Timestamp(TimeUnit::Nanosecond, _) = raw.data_type() else {
        return Err(physical_mismatch("timestamp", raw.data_type(), field_name));
    };
    let nanos = raw.as_primitive::<TimestampNanosecondType>();
    let micros: TimestampMicrosecondArray = nanos
        .iter()
        .map(|v| v.map(|ns| ns.div_euclid(1_000)))
        .collect();
    let array = match tz {
        Some(tz) => micros.with_timezone(tz),
        None => micros,
    };
    Ok(Arc::new(array))
}

/// Re-scale / re-precision an `orc-rust` `Decimal128` array to the expected `(precision, scale)`.
/// `orc-rust` already produced the file's `(p', s)`; on a precision-widen promotion the unscaled
/// values are identical and only the declared precision changes (same scale is enforced upstream).
fn convert_decimal(
    raw: &ArrayRef,
    precision: u32,
    scale: u32,
    field_name: &str,
) -> Result<ArrayRef> {
    let DataType::Decimal128(_, raw_scale) = raw.data_type() else {
        return Err(physical_mismatch("decimal", raw.data_type(), field_name));
    };
    let arrow_precision = u8::try_from(precision).map_err(|_| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("decimal precision {precision} exceeds Decimal128 (field '{field_name}')"),
        )
    })?;
    let arrow_scale = i8::try_from(scale).map_err(|_| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("decimal scale {scale} exceeds Decimal128 (field '{field_name}')"),
        )
    })?;
    if *raw_scale < 0 {
        // ORC never writes negative scale via Iceberg; guard anyway.
        return Err(physical_mismatch("decimal", raw.data_type(), field_name));
    }
    let values = raw.as_primitive::<Decimal128Type>();
    let rescaled = Decimal128Array::from_iter(values.iter())
        .with_precision_and_scale(arrow_precision, arrow_scale)
        .map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Failed to set Decimal128 precision/scale ({precision}, {scale}) for field \
                     '{field_name}'"
                ),
            )
            .with_source(e)
        })?;
    Ok(Arc::new(rescaled))
}

/// Convert an `orc-rust` `Binary` array to `FixedSizeBinary(width)`, enforcing the width.
fn binary_to_fixed(raw: &ArrayRef, width: i32, field_name: &str) -> Result<ArrayRef> {
    let bin = downcast_binary(raw, field_name)?;
    let mut builder = FixedSizeBinaryBuilder::new(width);
    for i in 0..bin.len() {
        if bin.is_null(i) {
            builder.append_null();
        } else {
            builder.append_value(bin.value(i)).map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("ORC binary value for field '{field_name}' is not {width} bytes wide"),
                )
                .with_source(e)
            })?;
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// Convert an `orc-rust` `Binary` array to the canonical Iceberg `LargeBinary` Arrow type.
fn binary_to_large_binary(raw: &ArrayRef, field_name: &str) -> Result<ArrayRef> {
    let bin = downcast_binary(raw, field_name)?;
    let mut builder = LargeBinaryBuilder::new();
    for i in 0..bin.len() {
        if bin.is_null(i) {
            builder.append_null();
        } else {
            builder.append_value(bin.value(i));
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// Downcast an `orc-rust` array to `BinaryArray` (ORC binary is yielded as Arrow `Binary`).
fn downcast_binary<'a>(raw: &'a ArrayRef, field_name: &str) -> Result<&'a BinaryArray> {
    raw.as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| physical_mismatch("binary", raw.data_type(), field_name))
}

/// A physical-type mismatch error (the `orc-rust` Arrow output did not match the resolved Iceberg
/// type's expected physical shape — should only happen on a corrupt file / orc-rust contract drift).
fn physical_mismatch(iceberg_type: &str, found: &DataType, field_name: &str) -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        format!(
            "ORC physical type {found:?} for field '{field_name}' is incompatible with the \
             resolved Iceberg type '{iceberg_type}'"
        ),
    )
}

#[cfg(test)]
mod tests {
    include!("orc_reader_tests.rs");
}
