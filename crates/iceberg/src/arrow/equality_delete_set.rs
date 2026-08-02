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

//! A hashed set-membership accelerator for equality deletes — the O(R) fast path.
//!
//! The equality-delete READ contract is otherwise expressed as a survival [`Predicate`] built one
//! leaf-per-delete-row (`crate::arrow::caching_delete_file_loader`): a tree of `E` leaves evaluated
//! against every data batch, so applying a file of `E` delete rows to `R` data rows costs `O(E·R)`.
//! Java instead hashes the delete keys into a `StructLikeSet` and tests membership per data row —
//! `O(R)`. This module is the Rust analogue of that hashed set, BUT it is deliberately scoped to the
//! cases where set-membership is provably byte-identical to the predicate path (the real oracle —
//! see the equivalence harness in `delete_filter.rs`).
//!
//! ## Why this is gated by key-column type (the soundness boundary)
//!
//! The predicate path compares a data column against each delete literal with the **Arrow**
//! comparison kernel (`arrow_ord::cmp::eq`), whose FLOAT kernels use *total ordering*:
//! `NaN == NaN` is TRUE and `-0.0 != 0.0`. A hash key built from [`Datum`]/`OrderedFloat` agrees on
//! `NaN` but COLLAPSES `-0.0` and `+0.0` into one key — so for a `Float`/`Double` key column the set
//! path would delete a `-0.0` row that the predicate path keeps. That divergence is proven by
//! `delete_filter::tests::test_h6_naive_set_diverges_on_negative_zero`.
//!
//! Therefore [`EqDeleteKeySet::is_eligible_type`] admits ONLY the primitive types that satisfy BOTH
//! (a) [`Datum`] equality byte-identical to the Arrow `eq` kernel AND (b) evaluability by the
//! predicate fallback (`get_arrow_datum`) — so a per-batch bail to the predicate path can never land
//! on an unsupported-type error. Only `Float`, `Double`, `Decimal` (a cast-rescale hazard when the
//! delete-file and data-file scales differ), and `Unknown` are excluded: the floats fail (a), and
//! `Unknown` is not a real value type. `Time` (compared as its `i64` micros-from-midnight backing)
//! and `Fixed` (compared as a fixed-width byte string) BOTH satisfy (a) — their equality is integer-
//! / byte-identical under the two kernels — AND, since `get_arrow_datum` now has arms for them,
//! satisfy (b); they are admitted. An eq-delete file with ANY excluded key column routes the whole
//! task back to the untouched predicate path. The matrix of admitted types is proven identical to
//! the predicate path in `delete_filter.rs`'s harness.
//!
//! ## Columnar encoding (FK1 / scout #3)
//!
//! Keys are stored as a compact byte encoding (or a specialized `HashSet<i64>` for a single
//! integer-like column), not as `HashSet<Vec<Option<Datum>>>`. Probe-side `delete_mask` hashes /
//! encodes directly from Arrow arrays without a per-cell `Datum` decode + clone storm. Build still
//! accepts the loader's `Vec<Option<Datum>>` rows (decoded once at parse) and re-encodes them into
//! the compact form. NULL data cells still bail to the predicate path (conservative 3VL boundary —
//! null-aware set membership is a follow-up seed).

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Date32Array, FixedSizeBinaryArray, Int32Array,
    Int64Array, LargeBinaryArray, LargeStringArray, RecordBatch, StringArray,
    Time64MicrosecondArray, TimestampMicrosecondArray, TimestampNanosecondArray,
};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::spec::{Datum, PrimitiveLiteral, PrimitiveType};
use crate::{Error, ErrorKind, Result};

/// One equality-delete file represented as a hashed set of its delete-key tuples, for `O(R)`
/// membership application. A data row is DELETED by this file iff, for the file's ordered key
/// columns, the row's value tuple is present in the set — exactly the condition the
/// per-delete-row survival predicate encodes (`NOT(AND col_i = v_i)`), but tested by hash lookup
/// instead of by evaluating an `E`-leaf boolean tree.
///
/// Construction is gated: a set is built ONLY when every key column's type is
/// [`is_eligible_type`](Self::is_eligible_type) (non-float, Datum-Eq == Arrow-eq). When any key
/// column is ineligible the caller keeps the predicate path.
#[derive(Debug, Clone)]
pub(crate) struct EqDeleteKeySet {
    /// The key columns in file order: `(iceberg field id, iceberg field name, primitive type)`.
    key_columns: Vec<(i32, String, PrimitiveType)>,
    /// Compact membership store — see [`KeyStore`].
    store: KeyStore,
    /// True when at least one delete tuple carried a NULL cell that the I64 store could not
    /// retain (nulls are dropped there). The Bytes store encodes null tags, so this flag stays
    /// false for Bytes builds. Callers must not treat an I64-empty set as “deletes nothing” when
    /// this is true — null data still needs the predicate path (`col IS NULL`).
    i64_dropped_null_deletes: bool,
}

/// Membership backend. Single integer-like keys use `I64` (no alloc per key); everything else
/// uses length-tagged byte encodings compatible with [`encode_literal`] / Arrow probe encoding.
#[derive(Debug, Clone)]
enum KeyStore {
    /// Single-column boolean / int32 / int64-family key.
    I64(HashSet<i64>),
    /// Multi-column keys, or string/binary/uuid/fixed single-column keys.
    Bytes(HashSet<Vec<u8>>),
}

impl EqDeleteKeySet {
    /// Whether `ty` may participate in the hashed fast path: `true` iff (a) [`Datum`] equality for
    /// this type is byte-identical to the Arrow `eq` kernel the predicate path uses, AND (b) the
    /// predicate fallback (`get_arrow_datum`) can actually evaluate the type — so a per-batch bail
    /// to the predicate path (e.g. on a key-column NULL) never lands on an unsupported-type error.
    /// Floats are excluded (total-ordering / signed-zero divergence — proven), `Decimal` is excluded
    /// (the predicate path's `try_cast_literal` may rescale a literal to the column scale, which a
    /// raw-`i128` key does not), and `Unknown` is not a real value type. `Time` is admitted (it
    /// compares as its `i64` micros-from-midnight backing — integer-identical under both kernels) and
    /// `Fixed(_)` is admitted (a fixed-width byte string — byte-identical under both kernels); both
    /// gained a `get_arrow_datum` arm, so a key-null bail to the predicate path now succeeds rather
    /// than erroring. Every admitted type compares as an integer, byte string, or UTF-8 string under
    /// both Arrow `eq` and `Datum` `Eq`, and is convertible by `get_arrow_datum`.
    pub(crate) fn is_eligible_type(ty: &PrimitiveType) -> bool {
        match ty {
            PrimitiveType::Boolean
            | PrimitiveType::Int
            | PrimitiveType::Long
            | PrimitiveType::Date
            | PrimitiveType::Time
            | PrimitiveType::Timestamp
            | PrimitiveType::Timestamptz
            | PrimitiveType::TimestampNs
            | PrimitiveType::TimestamptzNs
            | PrimitiveType::String
            | PrimitiveType::Uuid
            | PrimitiveType::Binary
            | PrimitiveType::Fixed(_) => true,
            // Excluded: equality diverges (Float/Double — total-ordering / signed-zero), a rescale
            // hazard (Decimal), or not a value type (Unknown) — see the doc above.
            PrimitiveType::Float
            | PrimitiveType::Double
            | PrimitiveType::Decimal { .. }
            | PrimitiveType::Unknown => false,
        }
    }

    /// `true` when a single key column can use the specialized `HashSet<i64>` store.
    fn is_i64_family(ty: &PrimitiveType) -> bool {
        matches!(
            ty,
            PrimitiveType::Boolean
                | PrimitiveType::Int
                | PrimitiveType::Long
                | PrimitiveType::Date
                | PrimitiveType::Time
                | PrimitiveType::Timestamp
                | PrimitiveType::Timestamptz
                | PrimitiveType::TimestampNs
                | PrimitiveType::TimestamptzNs
        )
    }

    /// Build a set from the ordered key columns and the per-row delete tuples (each inner `Vec` has
    /// one entry per key column, in `key_columns` order). Returns `None` — signalling "use the
    /// predicate path" — if ANY key column type is ineligible. Duplicate tuples collapse (a set),
    /// matching the predicate path where duplicate delete rows are redundant.
    ///
    /// **Null cells:** the Bytes store retains them (null tag) for a future null-aware probe. The
    /// I64 specialized store cannot represent null — those cells are dropped and an internal
    /// `i64_dropped_null_deletes` flag is set so [`is_empty`](Self::is_empty) stays false and
    /// callers still route null data to the predicate path. Probe still bails on any null in the
    /// data batch (conservative 3VL boundary).
    pub(crate) fn try_build(
        key_columns: Vec<(i32, String, PrimitiveType)>,
        rows: Vec<Vec<Option<Datum>>>,
    ) -> Option<Self> {
        if key_columns.is_empty() {
            return None;
        }
        if !key_columns
            .iter()
            .all(|(_, _, ty)| Self::is_eligible_type(ty))
        {
            return None;
        }

        let mut i64_dropped_null_deletes = false;
        let store = if key_columns.len() == 1 && Self::is_i64_family(&key_columns[0].2) {
            let mut set = HashSet::with_capacity(rows.len());
            for row in rows {
                if row.len() != 1 {
                    return None;
                }
                match &row[0] {
                    // Null delete keys cannot hit the i64 specialized store; drop them and
                    // remember — a non-null data batch never matches a null delete key, and null
                    // data batches must still reach the predicate path (null-delete leaf).
                    None => {
                        i64_dropped_null_deletes = true;
                    }
                    Some(d) => {
                        set.insert(literal_as_i64(d.literal())?);
                    }
                }
            }
            KeyStore::I64(set)
        } else {
            let mut set = HashSet::with_capacity(rows.len());
            let mut buf = Vec::with_capacity(64);
            for row in rows {
                if row.len() != key_columns.len() {
                    return None;
                }
                buf.clear();
                for (cell, (_, _, ty)) in row.iter().zip(key_columns.iter()) {
                    match cell {
                        None => encode_null(&mut buf),
                        Some(d) => encode_literal(ty, d.literal(), &mut buf)?,
                    }
                }
                set.insert(buf.clone());
            }
            KeyStore::Bytes(set)
        };

        Some(Self {
            key_columns,
            store,
            i64_dropped_null_deletes,
        })
    }

    /// The ordered key field ids — used to confirm a task's eq-delete files share a key schema before
    /// the per-file masks are OR-combined.
    pub(crate) fn key_field_ids(&self) -> Vec<i32> {
        self.key_columns.iter().map(|(id, _, _)| *id).collect()
    }

    /// `true` if this file cannot delete any data row — empty membership store **and** no I64-dropped
    /// null deletes. A null-only I64 file has an empty store but `i64_dropped_null_deletes`, so this
    /// is `false`: null data rows can still be deleted via the predicate fallback. Matches the
    /// predicate path where a pure-`IS NULL` delete file is not a no-op.
    pub(crate) fn is_empty(&self) -> bool {
        if self.i64_dropped_null_deletes {
            return false;
        }
        match &self.store {
            KeyStore::I64(s) => s.is_empty(),
            KeyStore::Bytes(s) => s.is_empty(),
        }
    }

    /// Per-row DELETE mask over `batch`: `out[i] == true` ⇒ row `i` matches some delete tuple (is
    /// deleted by this file). Resolves each key column in `batch` by Iceberg field id
    /// (`PARQUET_FIELD_ID_META_KEY`) and probes the compact key store via columnar accessors —
    /// no per-cell [`Datum`] allocation on the probe path.
    ///
    /// Returns `Ok(None)` — meaning "fall back to the predicate path for this batch" — when ANY key
    /// column has a NULL in `batch`. This is the soundness boundary: null-key rows are governed by
    /// the predicate path's Java nulls-first semantics (unit A2: a NULL cell survives a value
    /// delete — `NULL != v` is TRUE — and is deleted only by a matching NULL delete tuple via
    /// `not_null`, the Java `StructLikeSet` verdict). The bail keeps this path conservative: the
    /// predicate path is the oracle for every null-carrying batch. Null-aware columnar membership
    /// is a deliberate follow-up (FK1 residual seed).
    ///
    /// A key column ABSENT from the batch returns an error rather than silently disagreeing with the
    /// predicate path (the apply seam guarantees the eq-delete columns are projected).
    pub(crate) fn delete_mask(&self, batch: &RecordBatch) -> Result<Option<Vec<bool>>> {
        let num_rows = batch.num_rows();

        // Resolve columns + null bail BEFORE the empty short-circuit. The I64 store drops null
        // delete cells (they cannot be represented as i64); a file whose only deletes are NULL
        // therefore builds an empty I64 set. Empty-first would return `Some(all-false)` and skip
        // the predicate fallback, under-deleting null data rows that the survival predicate's
        // `col IS NULL` leaf must remove. Null-carrying batches always route to the oracle.
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.key_columns.len());
        for (field_id, field_name, _) in &self.key_columns {
            let column = resolve_column_by_field_id(batch, *field_id).ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!(
                        "equality-delete set fast path: key column '{field_name}' (field id \
                         {field_id}) is absent from the data batch"
                    ),
                )
            })?;
            if column.null_count() > 0 {
                return Ok(None);
            }
            columns.push(column);
        }

        // No key-column NULLs: an empty set deletes nothing among non-null data (null-only
        // delete files also correctly delete nothing here — null deletes never match non-null).
        if self.is_empty() {
            return Ok(Some(vec![false; num_rows]));
        }

        match &self.store {
            KeyStore::I64(set) => {
                debug_assert_eq!(self.key_columns.len(), 1);
                let ty = &self.key_columns[0].2;
                let values = i64_column_values(&columns[0], ty)?;
                if values.len() != num_rows {
                    return Err(Error::new(
                        ErrorKind::Unexpected,
                        format!(
                            "equality-delete set fast path: i64 key column length {} != batch \
                             num_rows {num_rows}",
                            values.len()
                        ),
                    ));
                }
                let mut mask = Vec::with_capacity(num_rows);
                for v in values {
                    mask.push(set.contains(&v));
                }
                Ok(Some(mask))
            }
            KeyStore::Bytes(set) => {
                let mut mask = Vec::with_capacity(num_rows);
                let mut buf = Vec::with_capacity(64);
                for row in 0..num_rows {
                    buf.clear();
                    for (col, (_, _, ty)) in columns.iter().zip(self.key_columns.iter()) {
                        encode_arrow_cell(col, ty, row, &mut buf)?;
                    }
                    mask.push(set.contains(&buf));
                }
                Ok(Some(mask))
            }
        }
    }
}

// ===========================================================================
// Encoding — Datum/Literal side and Arrow probe side share the same layout
// ===========================================================================

/// Null cell tag (only used when encoding delete-side nulls into the Bytes store).
const TAG_NULL: u8 = 0;
const TAG_PRESENT: u8 = 1;

fn encode_null(out: &mut Vec<u8>) {
    out.push(TAG_NULL);
}

/// Encode a present primitive literal into `out`. Returns `None` if the literal shape does not
/// match the declared column type (build-time refuse → fall back to predicate path).
fn encode_literal(ty: &PrimitiveType, lit: &PrimitiveLiteral, out: &mut Vec<u8>) -> Option<()> {
    out.push(TAG_PRESENT);
    match (ty, lit) {
        (PrimitiveType::Boolean, PrimitiveLiteral::Boolean(v)) => {
            out.push(u8::from(*v));
            Some(())
        }
        (PrimitiveType::Int | PrimitiveType::Date, PrimitiveLiteral::Int(v)) => {
            out.extend_from_slice(&v.to_le_bytes());
            Some(())
        }
        (
            PrimitiveType::Long
            | PrimitiveType::Time
            | PrimitiveType::Timestamp
            | PrimitiveType::Timestamptz
            | PrimitiveType::TimestampNs
            | PrimitiveType::TimestamptzNs,
            PrimitiveLiteral::Long(v),
        ) => {
            out.extend_from_slice(&v.to_le_bytes());
            Some(())
        }
        (PrimitiveType::String, PrimitiveLiteral::String(s)) => {
            let bytes = s.as_bytes();
            let len = u32::try_from(bytes.len()).ok()?;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(bytes);
            Some(())
        }
        (PrimitiveType::Binary | PrimitiveType::Fixed(_), PrimitiveLiteral::Binary(b)) => {
            let len = u32::try_from(b.len()).ok()?;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(b);
            Some(())
        }
        (PrimitiveType::Uuid, PrimitiveLiteral::UInt128(v)) => {
            out.extend_from_slice(&v.to_le_bytes());
            Some(())
        }
        // Int128 path for UUID is not expected from Datum::new on Uuid, but refuse rather than
        // silently mis-encode.
        _ => None,
    }
}

fn literal_as_i64(lit: &PrimitiveLiteral) -> Option<i64> {
    match lit {
        PrimitiveLiteral::Boolean(v) => Some(i64::from(*v)),
        PrimitiveLiteral::Int(v) => Some(i64::from(*v)),
        PrimitiveLiteral::Long(v) => Some(*v),
        _ => None,
    }
}

/// Extract non-null i64-family values from an Arrow column (nulls already refused by caller).
fn i64_column_values(column: &ArrayRef, ty: &PrimitiveType) -> Result<Vec<i64>> {
    match ty {
        PrimitiveType::Boolean => {
            let a = column
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| type_mismatch("BooleanArray", column))?;
            Ok((0..a.len()).map(|i| i64::from(a.value(i))).collect())
        }
        PrimitiveType::Int => {
            let a = column
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| type_mismatch("Int32Array", column))?;
            Ok((0..a.len()).map(|i| i64::from(a.value(i))).collect())
        }
        PrimitiveType::Date => {
            let a = column
                .as_any()
                .downcast_ref::<Date32Array>()
                .ok_or_else(|| type_mismatch("Date32Array", column))?;
            Ok((0..a.len()).map(|i| i64::from(a.value(i))).collect())
        }
        PrimitiveType::Long => {
            let a = column
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| type_mismatch("Int64Array", column))?;
            Ok((0..a.len()).map(|i| a.value(i)).collect())
        }
        PrimitiveType::Time => {
            if let Some(a) = column.as_any().downcast_ref::<Time64MicrosecondArray>() {
                Ok((0..a.len()).map(|i| a.value(i)).collect())
            } else if let Some(a) = column.as_any().downcast_ref::<Int64Array>() {
                Ok((0..a.len()).map(|i| a.value(i)).collect())
            } else {
                Err(type_mismatch("Time64MicrosecondArray|Int64Array", column))
            }
        }
        PrimitiveType::Timestamp | PrimitiveType::Timestamptz => {
            if let Some(a) = column.as_any().downcast_ref::<TimestampMicrosecondArray>() {
                Ok((0..a.len()).map(|i| a.value(i)).collect())
            } else if let Some(a) = column.as_any().downcast_ref::<Int64Array>() {
                Ok((0..a.len()).map(|i| a.value(i)).collect())
            } else {
                Err(type_mismatch(
                    "TimestampMicrosecondArray|Int64Array",
                    column,
                ))
            }
        }
        PrimitiveType::TimestampNs | PrimitiveType::TimestamptzNs => {
            if let Some(a) = column.as_any().downcast_ref::<TimestampNanosecondArray>() {
                Ok((0..a.len()).map(|i| a.value(i)).collect())
            } else if let Some(a) = column.as_any().downcast_ref::<Int64Array>() {
                Ok((0..a.len()).map(|i| a.value(i)).collect())
            } else {
                Err(type_mismatch("TimestampNanosecondArray|Int64Array", column))
            }
        }
        other => Err(Error::new(
            ErrorKind::Unexpected,
            format!("equality-delete set: i64 store used for non-i64 type {other:?}"),
        )),
    }
}

/// Encode one non-null Arrow cell at `row` into `out` with the same layout as [`encode_literal`].
fn encode_arrow_cell(
    column: &ArrayRef,
    ty: &PrimitiveType,
    row: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    out.push(TAG_PRESENT);
    match ty {
        PrimitiveType::Boolean => {
            let a = column
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| type_mismatch("BooleanArray", column))?;
            out.push(u8::from(a.value(row)));
        }
        PrimitiveType::Int => {
            let a = column
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| type_mismatch("Int32Array", column))?;
            out.extend_from_slice(&a.value(row).to_le_bytes());
        }
        PrimitiveType::Date => {
            let a = column
                .as_any()
                .downcast_ref::<Date32Array>()
                .ok_or_else(|| type_mismatch("Date32Array", column))?;
            out.extend_from_slice(&a.value(row).to_le_bytes());
        }
        PrimitiveType::Long => {
            let a = column
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| type_mismatch("Int64Array", column))?;
            out.extend_from_slice(&a.value(row).to_le_bytes());
        }
        PrimitiveType::Time => {
            let v = if let Some(a) = column.as_any().downcast_ref::<Time64MicrosecondArray>() {
                a.value(row)
            } else if let Some(a) = column.as_any().downcast_ref::<Int64Array>() {
                a.value(row)
            } else {
                return Err(type_mismatch("Time64MicrosecondArray|Int64Array", column));
            };
            out.extend_from_slice(&v.to_le_bytes());
        }
        PrimitiveType::Timestamp | PrimitiveType::Timestamptz => {
            let v = if let Some(a) = column.as_any().downcast_ref::<TimestampMicrosecondArray>() {
                a.value(row)
            } else if let Some(a) = column.as_any().downcast_ref::<Int64Array>() {
                a.value(row)
            } else {
                return Err(type_mismatch(
                    "TimestampMicrosecondArray|Int64Array",
                    column,
                ));
            };
            out.extend_from_slice(&v.to_le_bytes());
        }
        PrimitiveType::TimestampNs | PrimitiveType::TimestamptzNs => {
            let v = if let Some(a) = column.as_any().downcast_ref::<TimestampNanosecondArray>() {
                a.value(row)
            } else if let Some(a) = column.as_any().downcast_ref::<Int64Array>() {
                a.value(row)
            } else {
                return Err(type_mismatch("TimestampNanosecondArray|Int64Array", column));
            };
            out.extend_from_slice(&v.to_le_bytes());
        }
        PrimitiveType::String => {
            let bytes = if let Some(a) = column.as_any().downcast_ref::<StringArray>() {
                a.value(row).as_bytes()
            } else if let Some(a) = column.as_any().downcast_ref::<LargeStringArray>() {
                a.value(row).as_bytes()
            } else {
                return Err(type_mismatch("StringArray|LargeStringArray", column));
            };
            let len = u32::try_from(bytes.len()).map_err(|_| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "equality-delete set: string key longer than u32::MAX",
                )
            })?;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(bytes);
        }
        PrimitiveType::Binary | PrimitiveType::Fixed(_) => {
            let bytes = if let Some(a) = column.as_any().downcast_ref::<BinaryArray>() {
                a.value(row)
            } else if let Some(a) = column.as_any().downcast_ref::<LargeBinaryArray>() {
                a.value(row)
            } else if let Some(a) = column.as_any().downcast_ref::<FixedSizeBinaryArray>() {
                a.value(row)
            } else {
                return Err(type_mismatch(
                    "BinaryArray|LargeBinaryArray|FixedSizeBinaryArray",
                    column,
                ));
            };
            let len = u32::try_from(bytes.len()).map_err(|_| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "equality-delete set: binary key longer than u32::MAX",
                )
            })?;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(bytes);
        }
        PrimitiveType::Uuid => {
            // UUID is FixedSizeBinary(16) on the Arrow side; Datum stores `uuid.as_u128()`.
            let bytes = if let Some(a) = column.as_any().downcast_ref::<FixedSizeBinaryArray>() {
                a.value(row)
            } else if let Some(a) = column.as_any().downcast_ref::<BinaryArray>() {
                a.value(row)
            } else {
                return Err(type_mismatch(
                    "FixedSizeBinaryArray|BinaryArray (uuid)",
                    column,
                ));
            };
            let arr: [u8; 16] = bytes.try_into().map_err(|_| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "equality-delete set: uuid key must be 16 bytes, got {}",
                        bytes.len()
                    ),
                )
            })?;
            let v = uuid::Uuid::from_bytes(arr).as_u128();
            out.extend_from_slice(&v.to_le_bytes());
        }
        other => {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!(
                    "equality-delete set: encode_arrow_cell on ineligible/unhandled type {other:?}"
                ),
            ));
        }
    }
    Ok(())
}

fn type_mismatch(expected: &str, column: &ArrayRef) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        format!(
            "equality-delete set fast path: expected {expected}, got {:?}",
            column.data_type()
        ),
    )
}

/// Resolve a batch column by its Iceberg field id (`PARQUET_FIELD_ID_META_KEY` field metadata),
/// mirroring `record_batch_predicate::RecordBatchPredicateEvaluator`'s resolution so the set path and
/// the predicate path read the SAME column for a given key.
fn resolve_column_by_field_id(batch: &RecordBatch, field_id: i32) -> Option<ArrayRef> {
    for (idx, field) in batch.schema().fields().iter().enumerate() {
        if let Some(id_str) = field.metadata().get(PARQUET_FIELD_ID_META_KEY)
            && let Ok(id) = id_str.parse::<i32>()
            && id == field_id
        {
            return Some(Arc::clone(batch.column(idx)));
        }
    }
    None
}

#[cfg(test)]
mod fk1_microbench {
    use std::sync::Arc;
    use std::time::Instant;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

    use super::EqDeleteKeySet;
    use crate::spec::{Datum, PrimitiveType};

    fn long_batch(values: Vec<i64>, field_id: i32) -> RecordBatch {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(PARQUET_FIELD_ID_META_KEY.to_string(), field_id.to_string());
        let field = Field::new("id", DataType::Int64, false).with_metadata(metadata);
        let schema = Arc::new(ArrowSchema::new(vec![field]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values)) as ArrayRef])
            .expect("batch")
    }

    /// Hour-0 / after wall for single-column Long keyset apply.
    /// Run: `cargo test -p iceberg --lib fk1_eq_delete_apply_microbench -- --nocapture --ignored`
    #[test]
    #[ignore = "manual hour-0 / after microbench — not part of the default gate"]
    fn fk1_eq_delete_apply_microbench() {
        let n_data: i64 = 1_000_000;
        for n_del in [100_000i64, 1_000_000i64] {
            let delete_rows: Vec<Vec<Option<Datum>>> =
                (0..n_del).map(|i| vec![Some(Datum::long(i * 2))]).collect();
            let set = EqDeleteKeySet::try_build(
                vec![(1, "id".to_string(), PrimitiveType::Long)],
                delete_rows,
            )
            .expect("Long set builds");

            let data: Vec<i64> = (0..n_data).collect();
            let batch = long_batch(data, 1);

            // Warmup
            let _ = set.delete_mask(&batch).expect("mask");

            let t0 = Instant::now();
            let mask = set.delete_mask(&batch).expect("mask").expect("non-null");
            let elapsed = t0.elapsed();
            let ns_per_row = elapsed.as_nanos() as f64 / n_data as f64;
            let deleted: usize = mask.iter().filter(|d| **d).count();
            eprintln!(
                "FK1 microbench: data={n_data} deletes={n_del} wall={elapsed:?} \
                 ns/row={ns_per_row:.2} deleted={deleted}"
            );
            assert_eq!(mask.len(), n_data as usize);
            // Every even key in [0, n_del*2) that is also < n_data is deleted.
            assert!(deleted > 0);
        }
    }
}
