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
//! A reference to a column ABSENT from the batch (schema evolution / projection) is treated as a
//! NULL column under three-valued logic, matching `PredicateConverter`'s "missing column → null"
//! branches: `is_null` / `<` / `<=` / `not_starts_with` / `not_in` are `true` on a missing column,
//! every other leaf is `false`.

use std::collections::HashMap;

use arrow_arith::boolean::{and, and_kleene, is_not_null, is_null, not, or, or_kleene};
use arrow_array::{Array, ArrayRef, BooleanArray, Datum as ArrowDatum, RecordBatch};
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
/// `PARQUET_FIELD_ID_META_KEY` field metadata); a reference to an absent column is treated as a
/// NULL column under three-valued logic (see the module docs).
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

    /// Cast the literal to the column's arrow type (the read-side `try_cast_literal`) and run `kernel`.
    fn binary_cmp(
        &self,
        reference: &BoundReference,
        literal: &Datum,
        on_missing_true: bool,
        kernel: impl Fn(&ArrayRef, &dyn ArrowDatum) -> std::result::Result<BooleanArray, ArrowError>,
    ) -> Result<BooleanArray> {
        match self.column_for(reference) {
            Some(col) => {
                let lit = get_arrow_datum(literal)?;
                let cast = try_cast_literal(&lit, col.data_type()).map_err(arrow_err)?;
                kernel(&col, cast.as_ref()).map_err(arrow_err)
            }
            None if on_missing_true => self.all_true(),
            None => self.all_false(),
        }
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
        if self.column_for(reference).is_some() {
            self.all_true()
        } else {
            self.all_false()
        }
    }

    fn not_nan(&mut self, reference: &BoundReference, _p: &BoundPredicate) -> Result<BooleanArray> {
        if self.column_for(reference).is_some() {
            self.all_false()
        } else {
            self.all_true()
        }
    }

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
        self.binary_cmp(reference, literal, false, |c, l| neq(c, l))
    }

    fn starts_with(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        self.binary_cmp(reference, literal, false, |c, l| starts_with(c, l))
    }

    fn not_starts_with(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
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
                Ok(acc)
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
                Ok(acc)
            }
            None => self.all_true(),
        }
    }
}
