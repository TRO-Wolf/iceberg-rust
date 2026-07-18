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

//! Evaluate a [`BoundPredicate`] over ONE already-materialized Arrow [`RecordBatch`] to a
//! [`BooleanArray`] mask, resolving each [`BoundReference`] to a column by its Iceberg
//! **field-id** (`PARQUET_FIELD_ID_META_KEY` metadata).
//!
//! This is the post-materialization analogue of the Parquet read path's `PredicateConverter`
//! (in [`crate::arrow::reader`]). The Parquet path pushes the predicate DOWN into the parquet
//! reader as a `RowFilter` (it resolves a reference to a parquet-projection-mask leaf index);
//! callers that already hold a decoded batch (the equality-delete materialization in
//! [`crate::maintenance::convert_equality_delete_files`], and the **Avro** scan read path, which
//! cannot push down) instead resolve references against the batch's field-id metadata. The arrow
//! comparison kernels are identical to the Parquet path's, so a row that survives here survives
//! there — the two delete-application strategies are behaviorally 1:1.
//!
//! # Null semantics: Java nulls-first total order, NOT SQL three-valued logic
//!
//! Every leaf mask is TWO-valued (`null_count() == 0`), carrying the verdict Java's row-level
//! `Evaluator$EvalVisitor` (iceberg-api 1.10.0) produces. Java computes each comparison as
//! `literal.comparator().compare(term.eval(struct), literal.value())` where the comparator is
//! `Comparators.nullsFirst().thenComparing(naturalOrder)` (`Literals$ComparableLiteral` static
//! init, offsets 0-11) — a TOTAL order with null smallest (`Comparators$NullsFirst.compare`:
//! null-vs-non-null ⇒ `iconst_m1` at offsets 19-20, non-null-vs-null ⇒ `iconst_1` at 17-18).
//! Hence for a NULL value (or a column ABSENT from the batch — schema evolution reads it as a
//! NULL column, exactly like Java's `term.eval(struct) == null`) vs a non-null literal:
//!
//! | op | verdict | op | verdict |
//! |---|---|---|---|
//! | `is_null`, `<`, `<=`, `!=`, `not_in`, `not_starts_with`, `not_nan` | `true` | `not_null`, `>`, `>=`, `==`, `in`, `starts_with`, `is_nan` | `false` |
//!
//! (`in`: Java `literalSet.contains(null)` is `false` for both the `HashSet` and the
//! `CharSequenceSet` — the latter's `contains` fails its `instanceof CharSequence` check at
//! offsets 1-4 and returns `false` at 43-44; `startsWith` has an explicit null guard —
//! `ifnull 38` at offsets 11-12 of `Evaluator$EvalVisitor.startsWith` — and `notStartsWith`/
//! `notIn`/`notEq` are the plain negations.) Two-valued masks keep `not`/`and`/`or`
//! composition plain boolean algebra (Java's), and make the downstream null→false coercions
//! (`RowFilter` semantics, `coerce_nulls_to_false`) no-ops. SQL-3VL consumers are unaffected:
//! DataFusion's `Inexact` pushdown re-applies its own three-valued filter on top.

use std::collections::HashMap;

use arrow_arith::boolean::{and, and_kleene, is_not_null, is_null, not, or, or_kleene};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Datum as ArrowDatum, Float32Array, Float64Array, RecordBatch,
};
use arrow_ord::cmp::{eq, gt, gt_eq, lt, lt_eq, neq};
use arrow_schema::ArrowError;
use arrow_string::like::starts_with;
use fnv::FnvHashSet;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::{get_arrow_datum, try_cast_literal};
use crate::expr::visitors::bound_predicate_visitor::{BoundPredicateVisitor, visit};
use crate::expr::{BoundPredicate, BoundReference};
use crate::spec::Datum;
use crate::{Error, ErrorKind, Result};

/// Evaluate `predicate` over `batch` to a per-row [`BooleanArray`] mask (`true` ⇒ the row matches
/// the predicate). Columns are resolved by Iceberg field id (the batch's
/// `PARQUET_FIELD_ID_META_KEY` field metadata); NULL cells and absent columns evaluate under
/// Java's nulls-first total order to a TWO-valued mask (see the module docs).
pub(crate) fn evaluate_predicate_to_mask(
    predicate: &BoundPredicate,
    batch: &RecordBatch,
) -> Result<BooleanArray> {
    let mut evaluator = RecordBatchPredicateEvaluator::new(batch)?;
    visit(&mut evaluator, predicate)
}

/// A [`BoundPredicateVisitor`] that evaluates the predicate against ONE [`RecordBatch`] to a
/// [`BooleanArray`], mapping each [`BoundReference`] to a column by its field-id
/// (`PARQUET_FIELD_ID_META_KEY`) metadata. Mirrors the read path's `PredicateConverter` arrow
/// kernels, but resolves columns by field id (the batch is already schema-evolved) rather than by
/// a parquet projection-mask leaf index.
pub(crate) struct RecordBatchPredicateEvaluator<'a> {
    /// field id -> column index in the batch.
    field_id_to_col: HashMap<i32, usize>,
    batch: &'a RecordBatch,
}

impl<'a> RecordBatchPredicateEvaluator<'a> {
    pub(crate) fn new(batch: &'a RecordBatch) -> Result<Self> {
        let mut field_id_to_col = HashMap::new();
        for (idx, field) in batch.schema().fields().iter().enumerate() {
            if let Some(id_str) = field.metadata().get(PARQUET_FIELD_ID_META_KEY)
                && let Ok(id) = id_str.parse::<i32>()
            {
                field_id_to_col.insert(id, idx);
            }
        }
        Ok(Self {
            field_id_to_col,
            batch,
        })
    }

    /// The batch column for a reference, by field id (None when the column is absent — schema
    /// evolution).
    fn column_for(&self, reference: &BoundReference) -> Option<ArrayRef> {
        self.field_id_to_col
            .get(&reference.field().id)
            .map(|idx| self.batch.column(*idx).clone())
    }

    fn all_true(&self) -> Result<BooleanArray> {
        Ok(BooleanArray::from(vec![true; self.batch.num_rows()]))
    }

    fn all_false(&self) -> Result<BooleanArray> {
        Ok(BooleanArray::from(vec![false; self.batch.num_rows()]))
    }

    /// Cast the literal to the column's arrow type (the read-side `try_cast_literal`), run
    /// `kernel`, and resolve NULL cells to `null_verdict` — the verdict Java's nulls-first
    /// total-order comparison produces for a NULL value (see the module docs). A missing column
    /// is a NULL column, so it takes the same verdict for every row.
    fn binary_cmp(
        &self,
        reference: &BoundReference,
        literal: &Datum,
        null_verdict: bool,
        kernel: impl Fn(&ArrayRef, &dyn ArrowDatum) -> std::result::Result<BooleanArray, ArrowError>,
    ) -> Result<BooleanArray> {
        match self.column_for(reference) {
            Some(col) => {
                let lit = get_arrow_datum(literal)?;
                let cast = try_cast_literal(&lit, col.data_type()).map_err(arrow_err)?;
                Ok(null_filled(
                    kernel(&col, cast.as_ref()).map_err(arrow_err)?,
                    null_verdict,
                ))
            }
            None if null_verdict => self.all_true(),
            None => self.all_false(),
        }
    }
}

/// Resolve a possibly-three-valued comparison mask to the TWO-valued mask Java's nulls-first
/// total order dictates: every NULL slot (an arrow kernel's "one side was NULL" outcome) becomes
/// the op's `null_verdict` from the module-doc truth table; valid slots keep their kernel value.
/// The only NULL sources in these masks are NULL data cells — the literal side is never null
/// (Java binds a null literal to `IS NULL`/`IS NOT NULL`, and `Datum` cannot hold one).
///
/// Consults `is_valid` per slot, NEVER the underlying value buffer of an invalid slot (arrow
/// leaves those bytes arbitrary).
pub(crate) fn null_filled(mask: BooleanArray, null_verdict: bool) -> BooleanArray {
    if mask.null_count() == 0 {
        return mask;
    }
    BooleanArray::from_iter((0..mask.len()).map(|row| {
        Some(if mask.is_valid(row) {
            mask.value(row)
        } else {
            null_verdict
        })
    }))
}

/// Per-row `is_nan` mask over `column`, mirroring Java `NaNUtil.isNaN(Object)`
/// (iceberg-api 1.10.0 bytecode: a NULL value is NOT NaN — `ifnonnull 6` / `iconst_0` at offsets
/// 0-5; `Double.isNaN` at 13-23; `Float.isNaN` at 31-41; any non-floating value falls through to
/// `iconst_0` at 42-43). The mask is TWO-valued (`null_count() == 0`): Java's NaN predicates
/// return a plain boolean, never SQL three-valued-logic NULL, so a NULL cell yields literal
/// `false` here (and literal `true` under [`not_nan_row_mask`]) — a validity-propagating mask
/// would make the `RowFilter` / `coerce_nulls_to_false` consumers DROP null cells under
/// `not_nan`, diverging from Java. A non-float column (unreachable through `Predicate::bind`,
/// which rejects NaN predicates on non-float terms) degrades to the same all-`false` constant
/// Java produces.
pub(crate) fn is_nan_row_mask(column: &dyn Array) -> BooleanArray {
    nan_row_mask(column, false)
}

/// Per-row `not_nan` mask over `column`: the elementwise negation of [`is_nan_row_mask`],
/// matching Java `Evaluator$EvalVisitor.notNaN` (`!NaNUtil.isNaN(term.eval(struct))`, bytecode
/// offsets 10-21). A NULL cell is NOT NaN, so it yields literal `true` (the row is KEPT).
pub(crate) fn not_nan_row_mask(column: &dyn Array) -> BooleanArray {
    nan_row_mask(column, true)
}

/// Shared core of [`is_nan_row_mask`] / [`not_nan_row_mask`]: elementwise
/// `is_valid(row) && value(row).is_nan()`, XOR-flipped by `negate`. See [`is_nan_row_mask`] for
/// the Java `NaNUtil` truth table this mirrors (null → not NaN; non-float → not NaN).
fn nan_row_mask(column: &dyn Array, negate: bool) -> BooleanArray {
    if let Some(floats) = column.as_any().downcast_ref::<Float32Array>() {
        BooleanArray::from_iter(
            (0..floats.len())
                .map(|row| Some((floats.is_valid(row) && floats.value(row).is_nan()) != negate)),
        )
    } else if let Some(floats) = column.as_any().downcast_ref::<Float64Array>() {
        BooleanArray::from_iter(
            (0..floats.len())
                .map(|row| Some((floats.is_valid(row) && floats.value(row).is_nan()) != negate)),
        )
    } else {
        // Java `NaNUtil.isNaN` returns `false` for any non-Double/Float value (bytecode 42-43),
        // so `is_nan` is all-`false` and `not_nan` all-`true` on a non-float column.
        BooleanArray::from(vec![negate; column.len()])
    }
}

fn arrow_err(e: ArrowError) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        "Failed to evaluate a bound predicate over a record batch",
    )
    .with_source(e)
}

impl BoundPredicateVisitor for RecordBatchPredicateEvaluator<'_> {
    type T = BooleanArray;

    fn always_true(&mut self) -> Result<BooleanArray> {
        self.all_true()
    }

    fn always_false(&mut self) -> Result<BooleanArray> {
        self.all_false()
    }

    fn and(&mut self, lhs: BooleanArray, rhs: BooleanArray) -> Result<BooleanArray> {
        and_kleene(&lhs, &rhs).map_err(arrow_err)
    }

    fn or(&mut self, lhs: BooleanArray, rhs: BooleanArray) -> Result<BooleanArray> {
        or_kleene(&lhs, &rhs).map_err(arrow_err)
    }

    fn not(&mut self, inner: BooleanArray) -> Result<BooleanArray> {
        not(&inner).map_err(arrow_err)
    }

    fn is_null(&mut self, reference: &BoundReference, _p: &BoundPredicate) -> Result<BooleanArray> {
        match self.column_for(reference) {
            Some(col) => is_null(&col).map_err(arrow_err),
            None => self.all_true(),
        }
    }

    fn not_null(
        &mut self,
        reference: &BoundReference,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        match self.column_for(reference) {
            Some(col) => is_not_null(&col).map_err(arrow_err),
            None => self.all_false(),
        }
    }

    fn is_nan(&mut self, reference: &BoundReference, _p: &BoundPredicate) -> Result<BooleanArray> {
        match self.column_for(reference) {
            Some(col) => Ok(is_nan_row_mask(&col)),
            // A missing column is a NULL column: Java `NaNUtil.isNaN(null)` == false.
            None => self.all_false(),
        }
    }

    fn not_nan(&mut self, reference: &BoundReference, _p: &BoundPredicate) -> Result<BooleanArray> {
        match self.column_for(reference) {
            Some(col) => Ok(not_nan_row_mask(&col)),
            // A missing column is a NULL column: `!NaNUtil.isNaN(null)` == true.
            None => self.all_true(),
        }
    }

    // Null verdicts below: Java `Evaluator$EvalVisitor` compare-then-branch shapes over the
    // nulls-first comparator (compare(null, lit) == -1): `lt` is `< 0` ⇒ true, `ltEq` `<= 0` ⇒
    // true, `gt` `> 0` ⇒ false, `gtEq` `>= 0` ⇒ false, `eq` `== 0` ⇒ false, `notEq` = !eq ⇒
    // true (each method's branch instruction sits at bytecode offset 29, jumping to 36).

    fn less_than(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        self.binary_cmp(reference, literal, true, |c, l| lt(c, l))
    }

    fn less_than_or_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        self.binary_cmp(reference, literal, true, |c, l| lt_eq(c, l))
    }

    fn greater_than(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        self.binary_cmp(reference, literal, false, |c, l| gt(c, l))
    }

    fn greater_than_or_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        self.binary_cmp(reference, literal, false, |c, l| gt_eq(c, l))
    }

    fn eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        self.binary_cmp(reference, literal, false, |c, l| eq(c, l))
    }

    fn not_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        // Java `notEq` = !eq, so a NULL value (compare == -1, != 0) ⇒ TRUE — including on a
        // missing column (pre-fix this arm answered `false` there, audit BUG-002).
        self.binary_cmp(reference, literal, true, |c, l| neq(c, l))
    }

    fn starts_with(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        // Java `startsWith` has an explicit null guard (`ifnull 38`, offsets 11-12) ⇒ false.
        self.binary_cmp(reference, literal, false, |c, l| starts_with(c, l))
    }

    fn not_starts_with(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        // Java `notStartsWith` = !startsWith ⇒ true on a NULL value.
        self.binary_cmp(reference, literal, true, |c, l| not(&starts_with(c, l)?))
    }

    fn r#in(
        &mut self,
        reference: &BoundReference,
        literals: &FnvHashSet<Datum>,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        match self.column_for(reference) {
            Some(col) => {
                let mut acc = BooleanArray::from(vec![false; col.len()]);
                for literal in literals {
                    let lit = get_arrow_datum(literal)?;
                    let cast = try_cast_literal(&lit, col.data_type()).map_err(arrow_err)?;
                    let matches = eq(&col, cast.as_ref()).map_err(arrow_err)?;
                    acc = or(&acc, &matches).map_err(arrow_err)?;
                }
                // Java `in` = `literalSet.contains(value)`: `contains(null)` is false for both
                // set impls (module docs) ⇒ a NULL cell is NOT in any literal set.
                Ok(null_filled(acc, false))
            }
            None => self.all_false(),
        }
    }

    fn not_in(
        &mut self,
        reference: &BoundReference,
        literals: &FnvHashSet<Datum>,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        match self.column_for(reference) {
            Some(col) => {
                let mut acc = BooleanArray::from(vec![true; col.len()]);
                for literal in literals {
                    let lit = get_arrow_datum(literal)?;
                    let cast = try_cast_literal(&lit, col.data_type()).map_err(arrow_err)?;
                    let nonmatch = neq(&col, cast.as_ref()).map_err(arrow_err)?;
                    acc = and(&acc, &nonmatch).map_err(arrow_err)?;
                }
                // Java `notIn` = !in ⇒ a NULL cell IS "not in" every literal set.
                Ok(null_filled(acc, true))
            }
            None => self.all_true(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Float32Array, Float64Array, RecordBatch};
    use arrow_buffer::NullBuffer;
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

    use super::*;
    use crate::expr::{Bind, Reference};
    use crate::spec::{NestedField, PrimitiveType, Schema, SchemaRef, Type};

    /// Iceberg schema for the NaN pins: `dbl`/`flt` are float columns PRESENT in the test batch;
    /// `missing_dbl` binds but is ABSENT from the batch (schema evolution / projection).
    fn nan_test_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(1, "dbl", Type::Primitive(PrimitiveType::Double)).into(),
                    NestedField::optional(2, "flt", Type::Primitive(PrimitiveType::Float)).into(),
                    NestedField::optional(3, "missing_dbl", Type::Primitive(PrimitiveType::Double))
                        .into(),
                ])
                .build()
                .expect("build the NaN pin schema"),
        )
    }

    /// A batch with rows `[NaN, finite, NULL]` in both a Float64 (`dbl`, field id 1) and a
    /// Float32 (`flt`, field id 2) column.
    fn nan_finite_null_batch() -> RecordBatch {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("dbl", DataType::Float64, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("flt", DataType::Float32, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]));
        let dbl = Arc::new(Float64Array::from(vec![Some(f64::NAN), Some(1.5), None])) as ArrayRef;
        let flt = Arc::new(Float32Array::from(vec![Some(f32::NAN), Some(2.5), None])) as ArrayRef;
        RecordBatch::try_new(arrow_schema, vec![dbl, flt]).expect("build the NaN pin batch")
    }

    /// Collects a mask to plain bools, first asserting it is TWO-valued (`null_count() == 0`) —
    /// the Java parity invariant: `NaNUtil.isNaN` returns a boolean, never a 3VL NULL, so a
    /// validity-propagating mask (which would make `RowFilter`/`coerce_nulls_to_false` DROP null
    /// cells under `not_nan`) is itself a divergence.
    fn two_valued(mask: &BooleanArray) -> Vec<bool> {
        assert_eq!(
            mask.null_count(),
            0,
            "NaN predicate masks must be two-valued (Java NaNUtil returns a plain boolean)"
        );
        (0..mask.len()).map(|i| mask.value(i)).collect()
    }

    /// Pins: `is_nan` on a present float column evaluates PER ROW — the pre-fix constant-`true`
    /// shape returned finite (and NULL) rows as if they were NaN (audit BUG-001). NULL cell ⇒
    /// `false` per Java `NaNUtil.isNaN(null)` (bytecode offsets 0-5: `ifnonnull`/`iconst_0`).
    #[test]
    fn is_nan_evaluates_per_row_f64_and_f32_null_is_not_nan() {
        let schema = nan_test_schema();
        let batch = nan_finite_null_batch();

        for column_name in ["dbl", "flt"] {
            let predicate = Reference::new(column_name)
                .is_nan()
                .bind(schema.clone(), true)
                .expect("bind is_nan");
            let mask = evaluate_predicate_to_mask(&predicate, &batch).expect("evaluate is_nan");
            assert_eq!(
                two_valued(&mask),
                vec![true, false, false],
                "is_nan({column_name}) over [NaN, finite, NULL]"
            );
        }
    }

    /// Pins: `not_nan` on a present float column KEEPS finite AND NULL rows — the pre-fix
    /// constant-`false` shape silently dropped EVERY row (the corruption class of audit
    /// BUG-001). NULL cell ⇒ `true` per Java `Evaluator$EvalVisitor.notNaN` =
    /// `!NaNUtil.isNaN(value)` (bytecode offsets 10-21 negating the null⇒false of `NaNUtil`).
    /// Also pins `NOT(is_nan)` ≡ `not_nan` through the visitor's `not()` — only a two-valued
    /// `is_nan` mask keeps that equivalence for NULL cells.
    #[test]
    fn not_nan_keeps_finite_and_null_rows_f64_and_f32() {
        let schema = nan_test_schema();
        let batch = nan_finite_null_batch();

        for column_name in ["dbl", "flt"] {
            let not_nan = Reference::new(column_name)
                .is_not_nan()
                .bind(schema.clone(), true)
                .expect("bind not_nan");
            let mask = evaluate_predicate_to_mask(&not_nan, &batch).expect("evaluate not_nan");
            assert_eq!(
                two_valued(&mask),
                vec![false, true, true],
                "not_nan({column_name}) over [NaN, finite, NULL]"
            );

            let negated_is_nan = (!Reference::new(column_name).is_nan())
                .bind(schema.clone(), true)
                .expect("bind NOT(is_nan)");
            let negated_mask =
                evaluate_predicate_to_mask(&negated_is_nan, &batch).expect("evaluate NOT(is_nan)");
            assert_eq!(
                two_valued(&negated_mask),
                vec![false, true, true],
                "NOT(is_nan({column_name})) must equal not_nan({column_name})"
            );
        }
    }

    /// Pins the `is_valid` guard directly: a NULL slot whose UNDERLYING BUFFER value is NaN
    /// (legal in arrow — values under invalid slots are arbitrary) must still evaluate as
    /// NOT NaN. An implementation that reads the buffer without consulting validity would
    /// report `true` for `is_nan` on row 2.
    #[test]
    fn nan_masks_ignore_buffer_values_under_null_slots() {
        let floats = Float64Array::new(
            vec![f64::NAN, 1.5, f64::NAN].into(),
            Some(NullBuffer::from(vec![true, true, false])),
        );
        assert!(floats.is_null(2), "row 2 must be a NULL slot");

        let is_nan_mask = is_nan_row_mask(&floats);
        assert_eq!(two_valued(&is_nan_mask), vec![true, false, false]);

        let not_nan_mask = not_nan_row_mask(&floats);
        assert_eq!(two_valued(&not_nan_mask), vec![false, true, true]);
    }

    /// Pins the missing-column (schema evolution) arms: a bound float column ABSENT from the
    /// batch is a NULL column, so `is_nan` ⇒ all-`false` and `not_nan` ⇒ all-`true`
    /// (Java: `term.eval(struct)` yields null ⇒ `NaNUtil.isNaN(null)` == false).
    #[test]
    fn nan_predicates_on_missing_column_treat_column_as_null() {
        let schema = nan_test_schema();
        let batch = nan_finite_null_batch();

        let is_nan = Reference::new("missing_dbl")
            .is_nan()
            .bind(schema.clone(), true)
            .expect("bind is_nan on the missing column");
        let mask = evaluate_predicate_to_mask(&is_nan, &batch).expect("evaluate is_nan");
        assert_eq!(two_valued(&mask), vec![false, false, false]);

        let not_nan = Reference::new("missing_dbl")
            .is_not_nan()
            .bind(schema, true)
            .expect("bind not_nan on the missing column");
        let mask = evaluate_predicate_to_mask(&not_nan, &batch).expect("evaluate not_nan");
        assert_eq!(two_valued(&mask), vec![true, true, true]);
    }

    /// Iceberg schema for the nulls-first pins: `n` (long) and `s` (string) are PRESENT in the
    /// test batch; `missing_n` / `missing_s` bind but are ABSENT from it (schema evolution).
    fn nulls_first_test_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(1, "n", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(2, "s", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(3, "missing_n", Type::Primitive(PrimitiveType::Long))
                        .into(),
                    NestedField::optional(4, "missing_s", Type::Primitive(PrimitiveType::String))
                        .into(),
                ])
                .build()
                .expect("build the nulls-first pin schema"),
        )
    }

    /// A batch with rows `n = [1, 5, NULL]`, `s = ["apple", "banana", NULL]`.
    fn nulls_first_batch() -> RecordBatch {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("n", DataType::Int64, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("s", DataType::Utf8, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]));
        let n = Arc::new(arrow_array::Int64Array::from(vec![
            Some(1i64),
            Some(5),
            None,
        ])) as ArrayRef;
        let s = Arc::new(arrow_array::StringArray::from(vec![
            Some("apple"),
            Some("banana"),
            None,
        ])) as ArrayRef;
        RecordBatch::try_new(arrow_schema, vec![n, s]).expect("build the nulls-first pin batch")
    }

    fn nulls_first_mask(predicate: &crate::expr::Predicate, label: &str) -> Vec<bool> {
        let schema = nulls_first_test_schema();
        let batch = nulls_first_batch();
        let bound = predicate
            .clone()
            .bind(schema, true)
            .unwrap_or_else(|e| panic!("bind {label}: {e}"));
        let mask = evaluate_predicate_to_mask(&bound, &batch)
            .unwrap_or_else(|e| panic!("evaluate {label}: {e}"));
        two_valued(&mask)
    }

    /// Pins the PRESENT-column NULL-cell truth table for every comparison op against Java's
    /// nulls-first total order (audit BUG-003): `Evaluator$EvalVisitor` compares through
    /// `Comparators.nullsFirst().thenComparing(naturalOrder)` (compare(null, lit) == -1), so a
    /// NULL cell is TRUE under `<`/`<=`/`!=`/`NOT IN`/`NOT STARTS WITH` and FALSE under the
    /// rest. Every mask must be TWO-valued — a 3VL-null slot would be dropped by the
    /// `RowFilter`/`coerce_nulls_to_false` consumers regardless of the Java verdict. The
    /// already-coinciding FALSE-verdict ops are pinned too, guarding future kernel churn.
    #[test]
    fn nulls_first_present_column_truth_table() {
        // Row layout: n = [1, 5, NULL] — literal 5 makes rows GO left/eq/null.
        let cases: Vec<(crate::expr::Predicate, Vec<bool>, &str)> = vec![
            (
                Reference::new("n").less_than(Datum::long(5)),
                vec![true, false, true],
                "n < 5",
            ),
            (
                Reference::new("n").less_than_or_equal_to(Datum::long(1)),
                vec![true, false, true],
                "n <= 1",
            ),
            (
                Reference::new("n").greater_than(Datum::long(1)),
                vec![false, true, false],
                "n > 1",
            ),
            (
                Reference::new("n").greater_than_or_equal_to(Datum::long(5)),
                vec![false, true, false],
                "n >= 5",
            ),
            (
                Reference::new("n").equal_to(Datum::long(5)),
                vec![false, true, false],
                "n == 5",
            ),
            (
                Reference::new("n").not_equal_to(Datum::long(5)),
                vec![true, false, true],
                "n != 5",
            ),
            (
                Reference::new("n").is_in([Datum::long(1), Datum::long(99)]),
                vec![true, false, false],
                "n IN (1, 99)",
            ),
            (
                Reference::new("n").is_not_in([Datum::long(1), Datum::long(99)]),
                vec![false, true, true],
                "n NOT IN (1, 99)",
            ),
            (
                Reference::new("s").starts_with(Datum::string("app")),
                vec![true, false, false],
                "s STARTS WITH 'app'",
            ),
            (
                Reference::new("s").not_starts_with(Datum::string("app")),
                vec![false, true, true],
                "s NOT STARTS WITH 'app'",
            ),
        ];
        for (predicate, expected, label) in cases {
            assert_eq!(nulls_first_mask(&predicate, label), expected, "{label}");
        }
    }

    /// Pins NOT-composition parity: `bind` PRESERVES `Predicate::Not` (`BoundPredicate::Not`),
    /// so the visitor's `not()` runs over the leaf masks. Only TWO-valued leaves keep
    /// `NOT(n == 5)` ≡ `n != 5` for NULL cells — a 3VL `eq` mask would give
    /// `not(NULL) = NULL` → dropped, where Java's plain-boolean `notEq` says TRUE. This is the
    /// pin that REFUTES leaving the Java-null=FALSE kernels three-valued.
    #[test]
    fn nulls_first_not_composition_matches_java_negation() {
        assert_eq!(
            nulls_first_mask(
                &!Reference::new("n").equal_to(Datum::long(5)),
                "NOT(n == 5)"
            ),
            vec![true, false, true],
            "NOT(n == 5) must equal Java notEq — NULL cell => TRUE"
        );
        assert_eq!(
            nulls_first_mask(
                &!Reference::new("n").greater_than(Datum::long(1)),
                "NOT(n > 1)"
            ),
            vec![true, false, true],
            "NOT(n > 1) must equal Java !gt — NULL cell => TRUE"
        );
    }

    /// Pins the MISSING-column (schema evolution) arm for every op: an absent column is a NULL
    /// column, taking the same nulls-first verdict for all rows. `missing_n != 5` all-true is
    /// the audit BUG-002 headline (pre-fix: all-false ⇒ a schema-evolved file returned ZERO
    /// rows under `!=`).
    #[test]
    fn nulls_first_missing_column_truth_table() {
        let cases: Vec<(crate::expr::Predicate, bool, &str)> = vec![
            (
                Reference::new("missing_n").less_than(Datum::long(5)),
                true,
                "missing_n < 5",
            ),
            (
                Reference::new("missing_n").less_than_or_equal_to(Datum::long(5)),
                true,
                "missing_n <= 5",
            ),
            (
                Reference::new("missing_n").greater_than(Datum::long(5)),
                false,
                "missing_n > 5",
            ),
            (
                Reference::new("missing_n").greater_than_or_equal_to(Datum::long(5)),
                false,
                "missing_n >= 5",
            ),
            (
                Reference::new("missing_n").equal_to(Datum::long(5)),
                false,
                "missing_n == 5",
            ),
            (
                Reference::new("missing_n").not_equal_to(Datum::long(5)),
                true,
                "missing_n != 5",
            ),
            (
                Reference::new("missing_n").is_in([Datum::long(5), Datum::long(99)]),
                false,
                "missing_n IN (5, 99)",
            ),
            (
                Reference::new("missing_n").is_not_in([Datum::long(5), Datum::long(99)]),
                true,
                "missing_n NOT IN (5, 99)",
            ),
            (
                Reference::new("missing_n").is_null(),
                true,
                "missing_n IS NULL",
            ),
            (
                Reference::new("missing_n").is_not_null(),
                false,
                "missing_n IS NOT NULL",
            ),
            (
                Reference::new("missing_s").starts_with(Datum::string("app")),
                false,
                "missing_s STARTS WITH 'app'",
            ),
            (
                Reference::new("missing_s").not_starts_with(Datum::string("app")),
                true,
                "missing_s NOT STARTS WITH 'app'",
            ),
        ];
        for (predicate, expected, label) in cases {
            assert_eq!(
                nulls_first_mask(&predicate, label),
                vec![expected; 3],
                "{label}"
            );
        }
    }

    /// Pins [`null_filled`]'s `is_valid` guard (the A1 pattern): buffer values under INVALID
    /// slots are arbitrary in arrow, so the fill must consult validity, never the value buffer.
    /// Also pins the kernel side: an `n == 5` over a NULL slot whose buffer holds `5` must NOT
    /// match.
    #[test]
    fn null_filled_ignores_buffer_values_under_null_slots() {
        // A mask whose invalid slot carries `true` in the value buffer: fill=false must win.
        let mask = BooleanArray::new(
            vec![true, true].into(),
            Some(NullBuffer::from(vec![true, false])),
        );
        assert_eq!(two_valued(&null_filled(mask, false)), vec![true, false]);
        // ... and an invalid slot carrying `false`: fill=true must win.
        let mask = BooleanArray::new(
            vec![false, false].into(),
            Some(NullBuffer::from(vec![true, false])),
        );
        assert_eq!(two_valued(&null_filled(mask, true)), vec![false, true]);

        // End-to-end: Int64 [5, NULL(buffer=5)] under `n == 5` → [true, false].
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("n", DataType::Int64, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));
        let n = arrow_array::Int64Array::new(
            vec![5i64, 5].into(),
            Some(NullBuffer::from(vec![true, false])),
        );
        assert!(n.is_null(1), "row 1 must be a NULL slot");
        let batch = RecordBatch::try_new(arrow_schema, vec![Arc::new(n) as ArrayRef])
            .expect("build the buffer-under-null batch");
        let bound = Reference::new("n")
            .equal_to(Datum::long(5))
            .bind(nulls_first_test_schema(), true)
            .expect("bind n == 5");
        let mask = evaluate_predicate_to_mask(&bound, &batch).expect("evaluate n == 5");
        assert_eq!(two_valued(&mask), vec![true, false]);
    }
}
