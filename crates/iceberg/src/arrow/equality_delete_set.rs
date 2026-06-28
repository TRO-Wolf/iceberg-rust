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

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::arrow_primitive_to_literal;
use crate::spec::{Datum, PrimitiveType, Type};
use crate::{Error, ErrorKind, Result};

/// One equality-delete file represented as a hashed set of its delete-key tuples, for `O(R)`
/// membership application. A data row is DELETED by this file iff, for the file's ordered key
/// columns, the row's value tuple is present in [`tuples`](Self::tuples) — exactly the condition the
/// per-delete-row survival predicate encodes (`NOT(AND col_i = v_i)`), but tested by hash lookup
/// instead of by evaluating an `E`-leaf boolean tree.
///
/// Construction is gated: a set is built ONLY when every key column's type is
/// [`is_eligible_type`](Self::is_eligible_type) (non-float, Datum-Eq == Arrow-eq). When any key
/// column is ineligible the caller keeps the predicate path.
#[derive(Debug, Clone)]
pub(crate) struct EqDeleteKeySet {
    /// The key columns in file order: `(iceberg field id, iceberg field name, primitive type)`. The
    /// field id resolves the data column at apply time; the name/type are retained for diagnostics
    /// and to re-decode the data column with the SAME `arrow_primitive_to_literal` conversion the
    /// delete tuples were built with (so both sides are compared as identically-produced [`Datum`]s).
    key_columns: Vec<(i32, String, PrimitiveType)>,
    /// The distinct delete-key tuples. Each tuple has one entry per key column, in `key_columns`
    /// order; `None` is a NULL delete cell (which the predicate path encodes as `col IS NULL` and so
    /// deletes data rows that are NULL in that column).
    tuples: HashSet<Vec<Option<Datum>>>,
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

    /// Build a set from the ordered key columns and the per-row delete tuples (each inner `Vec` has
    /// one entry per key column, in `key_columns` order). Returns `None` — signalling "use the
    /// predicate path" — if ANY key column type is ineligible. Duplicate tuples collapse (a set),
    /// matching the predicate path where duplicate delete rows are redundant.
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
        let tuples: HashSet<Vec<Option<Datum>>> = rows.into_iter().collect();
        Some(Self {
            key_columns,
            tuples,
        })
    }

    /// The ordered key field ids — used to confirm a task's eq-delete files share a key schema before
    /// the per-file masks are OR-combined.
    pub(crate) fn key_field_ids(&self) -> Vec<i32> {
        self.key_columns.iter().map(|(id, _, _)| *id).collect()
    }

    /// `true` if the set has no delete tuples (an eq-delete file that deletes nothing). Such a file
    /// never deletes a row, exactly like its `AlwaysTrue` survival predicate.
    pub(crate) fn is_empty(&self) -> bool {
        self.tuples.is_empty()
    }

    /// Per-row DELETE mask over `batch`: `out[i] == true` ⇒ row `i` matches some delete tuple (is
    /// deleted by this file). Resolves each key column in `batch` by Iceberg field id
    /// (`PARQUET_FIELD_ID_META_KEY`), decodes it to [`Datum`]s with the SAME
    /// `arrow_primitive_to_literal` conversion the delete tuples used, assembles each row's tuple,
    /// and tests membership.
    ///
    /// Returns `Ok(None)` — meaning "fall back to the predicate path for this batch" — when ANY key
    /// column has a NULL in `batch`. This is the soundness boundary: the predicate path's survival
    /// mask is `coerce_nulls_to_false(eval(NOT(AND col=v)))`, so a data row that is NULL in a key
    /// column is governed by Arrow three-valued logic + the null→false coercion (it may be deleted
    /// EVEN WITHOUT a matching NULL delete tuple). Set membership cannot reproduce that 3VL exactly,
    /// so a batch carrying a key-column NULL defers to the proven predicate path. For batches with NO
    /// key-column NULLs, set membership is byte-identical to the predicate deletion (proven in
    /// `delete_filter.rs`'s harness): a fully-non-null row's tuple equals a stored delete tuple iff
    /// the predicate `OR_j (AND_i col_i = v_i)` is TRUE for it.
    ///
    /// A key column ABSENT from the batch returns an error rather than silently disagreeing with the
    /// predicate path (the apply seam guarantees the eq-delete columns are projected).
    pub(crate) fn delete_mask(&self, batch: &RecordBatch) -> Result<Option<Vec<bool>>> {
        let num_rows = batch.num_rows();
        if self.tuples.is_empty() {
            return Ok(Some(vec![false; num_rows]));
        }

        // Decode each key column to per-row Datums, in key_columns order. If any key column carries a
        // NULL, bail to the predicate path (3VL boundary above).
        let mut decoded_columns: Vec<Vec<Option<Datum>>> =
            Vec::with_capacity(self.key_columns.len());
        for (field_id, field_name, primitive_type) in &self.key_columns {
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

            let literals =
                arrow_primitive_to_literal(&column, &Type::Primitive(primitive_type.clone()))?;
            let datums: Vec<Option<Datum>> = literals
                .into_iter()
                .map(|maybe_literal| {
                    maybe_literal
                        .map(|literal| {
                            literal
                                .as_primitive_literal()
                                .map(|prim| Datum::new(primitive_type.clone(), prim))
                                .ok_or_else(|| {
                                    Error::new(
                                        ErrorKind::Unexpected,
                                        "equality-delete set fast path: data cell is not a \
                                         primitive literal",
                                    )
                                })
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?;
            decoded_columns.push(datums);
        }

        // Assemble each row's key tuple and test membership. No key column has NULLs here, so every
        // decoded cell is `Some` and a row's tuple matches a stored tuple only when that stored tuple
        // is itself all-non-null and equal — exactly the predicate path's non-null deletion.
        let mut mask = Vec::with_capacity(num_rows);
        let mut tuple_buf: Vec<Option<Datum>> = Vec::with_capacity(self.key_columns.len());
        for row in 0..num_rows {
            tuple_buf.clear();
            for column in &decoded_columns {
                tuple_buf.push(column[row].clone());
            }
            mask.push(self.tuples.contains(&tuple_buf));
        }
        Ok(Some(mask))
    }
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
