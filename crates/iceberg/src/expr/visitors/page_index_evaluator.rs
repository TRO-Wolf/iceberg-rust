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

//! Evaluates predicates against a Parquet Page Index

use std::collections::HashMap;

use fnv::FnvHashSet;
use ordered_float::OrderedFloat;
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};
use parquet::file::metadata::RowGroupMetaData;
use parquet::file::page_index::column_index::ColumnIndexMetaData;
use parquet::file::page_index::offset_index::OffsetIndexMetaData;

use crate::expr::visitors::bound_predicate_visitor::{BoundPredicateVisitor, visit};
use crate::expr::{BoundPredicate, BoundReference};
use crate::spec::{Datum, PrimitiveLiteral, PrimitiveType, Schema};
use crate::{Error, ErrorKind, Result};

type OffsetIndex = Vec<OffsetIndexMetaData>;

const IN_PREDICATE_LIMIT: usize = 200;

enum MissingColBehavior {
    CantMatch,
    MightMatch,
}

pub(crate) enum PageNullCount {
    AllNull,
    NoneNull,
    SomeNull,
    Unknown,
}

impl PageNullCount {
    fn from_row_and_null_counts(num_rows: usize, null_count: Option<i64>) -> Self {
        match (num_rows, null_count) {
            (x, Some(y)) if x == y as usize => PageNullCount::AllNull,
            (_, Some(0)) => PageNullCount::NoneNull,
            (_, Some(_)) => PageNullCount::SomeNull,
            _ => PageNullCount::Unknown,
        }
    }
}

pub(crate) struct PageIndexEvaluator<'a> {
    column_index: &'a [ColumnIndexMetaData],
    offset_index: &'a OffsetIndex,
    row_group_metadata: &'a RowGroupMetaData,
    iceberg_field_id_to_parquet_column_index: &'a HashMap<i32, usize>,
    snapshot_schema: &'a Schema,
    row_count_cache: HashMap<usize, Vec<usize>>,
}

impl<'a> PageIndexEvaluator<'a> {
    pub(crate) fn new(
        column_index: &'a [ColumnIndexMetaData],
        offset_index: &'a OffsetIndex,
        row_group_metadata: &'a RowGroupMetaData,
        field_id_map: &'a HashMap<i32, usize>,
        snapshot_schema: &'a Schema,
    ) -> Self {
        Self {
            column_index,
            offset_index,
            row_group_metadata,
            iceberg_field_id_to_parquet_column_index: field_id_map,
            snapshot_schema,
            row_count_cache: HashMap::new(),
        }
    }

    /// Evaluate this `PageIndexEvaluator`'s filter predicate against a
    /// specific page's column index entry in a parquet file's page index.
    /// [`ArrowReader`] uses the resulting [`RowSelection`] to reject
    /// pages within a parquet file's row group that cannot contain rows
    /// matching the filter predicate.
    pub(crate) fn eval(
        filter: &'a BoundPredicate,
        column_index: &'a [ColumnIndexMetaData],
        offset_index: &'a OffsetIndex,
        row_group_metadata: &'a RowGroupMetaData,
        field_id_map: &'a HashMap<i32, usize>,
        snapshot_schema: &'a Schema,
    ) -> Result<Vec<RowSelector>> {
        if row_group_metadata.num_rows() == 0 {
            return Ok(vec![]);
        }

        let mut evaluator = Self::new(
            column_index,
            offset_index,
            row_group_metadata,
            field_id_map,
            snapshot_schema,
        );

        Ok(visit(&mut evaluator, filter)?.iter().copied().collect())
    }

    fn select_all_rows(&self) -> Result<RowSelection> {
        Ok(vec![RowSelector::select(
            self.row_group_metadata.num_rows() as usize
        )]
        .into())
    }

    fn skip_all_rows(&self) -> Result<RowSelection> {
        Ok(vec![RowSelector::skip(
            self.row_group_metadata.num_rows() as usize
        )]
        .into())
    }

    fn calc_row_selection<F>(
        &mut self,
        field_id: i32,
        predicate: F,
        missing_col_behavior: MissingColBehavior,
    ) -> Result<RowSelection>
    where
        F: Fn(Option<Datum>, Option<Datum>, PageNullCount) -> Result<bool>,
    {
        let Some(&parquet_column_index) =
            self.iceberg_field_id_to_parquet_column_index.get(&field_id)
        else {
            // if the snapshot's column is not present in the row group,
            // exit early
            return match missing_col_behavior {
                MissingColBehavior::CantMatch => self.skip_all_rows(),
                MissingColBehavior::MightMatch => self.select_all_rows(),
            };
        };

        let Some(field) = self.snapshot_schema.field_by_id(field_id) else {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!("Field with id {field_id} missing from snapshot schema"),
            ));
        };

        let Some(field_type) = field.field_type.as_primitive_type() else {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!("Field with id {field_id} not convertible to primitive type"),
            ));
        };

        let Some(column_index) = self.column_index.get(parquet_column_index) else {
            // This should not happen, but we fail soft anyway so that the scan is still
            // successful, just a bit slower
            return self.select_all_rows();
        };

        let row_counts = {
            // Caches row count calculations for columns that appear multiple times in
            // the predicate
            if !self.row_count_cache.contains_key(&parquet_column_index) {
                let Some(offset_index) = self.offset_index.get(parquet_column_index) else {
                    // if we have a column index, we should always have an offset index.
                    return Err(Error::new(
                        ErrorKind::Unexpected,
                        format!("Missing offset index for field id {field_id}"),
                    ));
                };

                let count = self.calc_row_counts(offset_index);
                self.row_count_cache.insert(parquet_column_index, count);
            }
            self.row_count_cache
                .get(&parquet_column_index)
                .expect("row counts just cached")
        };

        let Some(page_filter) = Self::apply_predicate_to_column_index(
            predicate,
            field_type,
            column_index,
            row_counts,
        )?
        else {
            return self.select_all_rows();
        };

        let row_selectors: Vec<_> = row_counts
            .iter()
            .zip(page_filter.iter())
            .map(|(&row_count, &is_selected)| {
                if is_selected {
                    RowSelector::select(row_count)
                } else {
                    RowSelector::skip(row_count)
                }
            })
            .collect();

        Ok(row_selectors.into())
    }

    /// Returns a list of row counts per page
    fn calc_row_counts(&self, offset_index: &OffsetIndexMetaData) -> Vec<usize> {
        let mut remaining_rows = self.row_group_metadata.num_rows() as usize;
        let mut row_counts = Vec::with_capacity(offset_index.page_locations().len());

        let page_locations = offset_index.page_locations();
        for (idx, page_location) in page_locations.iter().enumerate() {
            let row_count = if idx < page_locations.len() - 1 {
                let row_count = (page_locations[idx + 1].first_row_index
                    - page_location.first_row_index) as usize;
                remaining_rows -= row_count;
                row_count
            } else {
                remaining_rows
            };

            row_counts.push(row_count);
        }

        row_counts
    }

    fn apply_predicate_to_column_index<F>(
        predicate: F,
        field_type: &PrimitiveType,
        column_index: &ColumnIndexMetaData,
        row_counts: &[usize],
    ) -> Result<Option<Vec<bool>>>
    where
        F: Fn(Option<Datum>, Option<Datum>, PageNullCount) -> Result<bool>,
    {
        let result: Result<Vec<bool>> = match column_index {
            ColumnIndexMetaData::NONE | ColumnIndexMetaData::INT96(_) => {
                return Ok(None);
            }
            ColumnIndexMetaData::BOOLEAN(idx) => Self::apply_to_pages(
                &predicate,
                idx.min_values_iter(),
                idx.max_values_iter(),
                |i| idx.null_count(i),
                row_counts,
                |&val| Self::bound_datum(field_type, PrimitiveLiteral::Boolean(val)),
            ),
            ColumnIndexMetaData::INT32(idx) => Self::apply_to_pages(
                &predicate,
                idx.min_values_iter(),
                idx.max_values_iter(),
                |i| idx.null_count(i),
                row_counts,
                |&val| Self::bound_datum(field_type, PrimitiveLiteral::Int(val)),
            ),
            ColumnIndexMetaData::INT64(idx) => Self::apply_to_pages(
                &predicate,
                idx.min_values_iter(),
                idx.max_values_iter(),
                |i| idx.null_count(i),
                row_counts,
                |&val| Self::bound_datum(field_type, PrimitiveLiteral::Long(val)),
            ),
            ColumnIndexMetaData::FLOAT(idx) => Self::apply_to_pages(
                &predicate,
                idx.min_values_iter(),
                idx.max_values_iter(),
                |i| idx.null_count(i),
                row_counts,
                |&val| {
                    Self::bound_datum(field_type, PrimitiveLiteral::Float(OrderedFloat::from(val)))
                },
            ),
            ColumnIndexMetaData::DOUBLE(idx) => Self::apply_to_pages(
                &predicate,
                idx.min_values_iter(),
                idx.max_values_iter(),
                |i| idx.null_count(i),
                row_counts,
                |&val| {
                    Self::bound_datum(
                        field_type,
                        PrimitiveLiteral::Double(OrderedFloat::from(val)),
                    )
                },
            ),
            ColumnIndexMetaData::BYTE_ARRAY(idx)
            | ColumnIndexMetaData::FIXED_LEN_BYTE_ARRAY(idx) => Self::apply_to_pages(
                &predicate,
                idx.min_values_iter(),
                idx.max_values_iter(),
                |i| idx.null_count(i),
                row_counts,
                |val| Self::bound_datum_bytes(field_type, val),
            ),
        };

        Ok(Some(result?))
    }

    fn apply_to_pages<'v, F, V>(
        predicate: &F,
        min_values: impl Iterator<Item = Option<&'v V>>,
        max_values: impl Iterator<Item = Option<&'v V>>,
        null_count: impl Fn(usize) -> Option<i64>,
        row_counts: &[usize],
        bound: impl Fn(&'v V) -> Option<Datum>,
    ) -> Result<Vec<bool>>
    where
        F: Fn(Option<Datum>, Option<Datum>, PageNullCount) -> Result<bool>,
        V: ?Sized + 'v,
    {
        min_values
            .zip(max_values)
            .enumerate()
            .zip(row_counts.iter())
            .map(|((i, (min, max)), &row_count)| {
                predicate(
                    min.and_then(&bound),
                    max.and_then(&bound),
                    PageNullCount::from_row_and_null_counts(row_count, null_count(i)),
                )
            })
            .collect()
    }

    fn bound_datum(field_type: &PrimitiveType, literal: PrimitiveLiteral) -> Option<Datum> {
        match (field_type, &literal) {
            (PrimitiveType::Decimal { .. }, PrimitiveLiteral::Int(value)) => Some(Datum::new(
                field_type.clone(),
                PrimitiveLiteral::Int128(i128::from(*value)),
            )),
            (PrimitiveType::Decimal { .. }, PrimitiveLiteral::Long(value)) => Some(Datum::new(
                field_type.clone(),
                PrimitiveLiteral::Int128(i128::from(*value)),
            )),
            _ if field_type.compatible(&literal) => Some(Datum::new(field_type.clone(), literal)),
            _ => {
                let promoted = literal.promote_to(field_type);
                field_type
                    .compatible(&promoted)
                    .then(|| Datum::new(field_type.clone(), promoted))
            }
        }
    }

    fn bound_datum_bytes(field_type: &PrimitiveType, bytes: &[u8]) -> Option<Datum> {
        Datum::try_from_bytes(bytes, field_type.clone()).ok()
    }

    pub(crate) fn eq_keeps_page(
        min: Option<Datum>,
        max: Option<Datum>,
        nulls: PageNullCount,
        datum: &Datum,
    ) -> bool {
        if matches!(nulls, PageNullCount::AllNull) {
            return false;
        }

        if datum.is_nan()
            || min.as_ref().is_some_and(|bound| bound.is_nan())
            || max.as_ref().is_some_and(|bound| bound.is_nan())
        {
            return true;
        }

        if let Some(min) = min
            && min.gt(datum)
        {
            return false;
        }

        if let Some(max) = max
            && max.lt(datum)
        {
            return false;
        }

        true
    }

    pub(crate) fn in_keeps_page(
        min: Option<Datum>,
        max: Option<Datum>,
        nulls: PageNullCount,
        literals: &FnvHashSet<Datum>,
    ) -> bool {
        if matches!(nulls, PageNullCount::AllNull) {
            return false;
        }

        if min.as_ref().is_some_and(|bound| bound.is_nan())
            || max.as_ref().is_some_and(|bound| bound.is_nan())
            || literals.iter().any(|literal| literal.is_nan())
        {
            return true;
        }

        match (min, max) {
            (Some(min), Some(max))
                if literals
                    .iter()
                    .all(|datum| datum.lt(&min) || datum.gt(&max)) =>
            {
                return false;
            }
            (Some(min), _) if !literals.iter().any(|datum| datum.ge(&min)) => {
                return false;
            }
            (_, Some(max)) if !literals.iter().any(|datum| datum.le(&max)) => {
                return false;
            }

            _ => {}
        }

        true
    }

    fn visit_inequality(
        &mut self,
        reference: &BoundReference,
        datum: &Datum,
        cmp_fn: fn(&Datum, &Datum) -> bool,
        use_lower_bound: bool,
    ) -> Result<RowSelection> {
        let field_id = reference.field().id;

        self.calc_row_selection(
            field_id,
            |min, max, null_count| {
                if matches!(null_count, PageNullCount::AllNull) {
                    return Ok(false);
                }

                if datum.is_nan() {
                    // NaN indicates unreliable bounds.
                    return Ok(true);
                }

                let bound = if use_lower_bound { min } else { max };

                if let Some(bound) = bound {
                    if bound.partial_cmp(datum).is_none() || cmp_fn(&bound, datum) {
                        return Ok(true);
                    }

                    return Ok(false);
                }

                Ok(true)
            },
            MissingColBehavior::MightMatch,
        )
    }
}

impl BoundPredicateVisitor for PageIndexEvaluator<'_> {
    type T = RowSelection;

    fn always_true(&mut self) -> Result<RowSelection> {
        self.select_all_rows()
    }

    fn always_false(&mut self) -> Result<RowSelection> {
        self.skip_all_rows()
    }

    fn and(&mut self, lhs: RowSelection, rhs: RowSelection) -> Result<RowSelection> {
        Ok(lhs.intersection(&rhs))
    }

    fn or(&mut self, lhs: RowSelection, rhs: RowSelection) -> Result<RowSelection> {
        Ok(lhs.union(&rhs))
    }

    fn not(&mut self, _: RowSelection) -> Result<RowSelection> {
        Err(Error::new(
            ErrorKind::Unexpected,
            "NOT unsupported at this point. NOT-rewrite should be performed first",
        ))
    }

    fn is_null(
        &mut self,
        reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        let field_id = reference.field().id;

        self.calc_row_selection(
            field_id,
            |_max, _min, null_count| Ok(!matches!(null_count, PageNullCount::NoneNull)),
            MissingColBehavior::MightMatch,
        )
    }

    fn not_null(
        &mut self,
        reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        let field_id = reference.field().id;

        self.calc_row_selection(
            field_id,
            |_max, _min, null_count| Ok(!matches!(null_count, PageNullCount::AllNull)),
            MissingColBehavior::CantMatch,
        )
    }

    fn is_nan(
        &mut self,
        reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        // NaN counts not present in ColumnChunkMetadata Statistics.
        // Only float columns can be NaN.
        if reference.field().field_type.is_floating_type() {
            self.select_all_rows()
        } else {
            self.skip_all_rows()
        }
    }

    fn not_nan(
        &mut self,
        _reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        // NaN counts not present in ColumnChunkMetadata Statistics
        self.select_all_rows()
    }

    fn less_than(
        &mut self,
        reference: &BoundReference,
        datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        self.visit_inequality(reference, datum, PartialOrd::lt, true)
    }

    fn less_than_or_eq(
        &mut self,
        reference: &BoundReference,
        datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        self.visit_inequality(reference, datum, PartialOrd::le, true)
    }

    fn greater_than(
        &mut self,
        reference: &BoundReference,
        datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        self.visit_inequality(reference, datum, PartialOrd::gt, false)
    }

    fn greater_than_or_eq(
        &mut self,
        reference: &BoundReference,
        datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        self.visit_inequality(reference, datum, PartialOrd::ge, false)
    }

    fn eq(
        &mut self,
        reference: &BoundReference,
        datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        let field_id = reference.field().id;

        self.calc_row_selection(
            field_id,
            |min, max, nulls| Ok(Self::eq_keeps_page(min, max, nulls, datum)),
            MissingColBehavior::CantMatch,
        )
    }

    fn not_eq(
        &mut self,
        _reference: &BoundReference,
        _datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        // Because the bounds are not necessarily a min or max value,
        // this cannot be answered using them. notEq(col, X) with (X, Y)
        // doesn't guarantee that X is a value in col.
        self.select_all_rows()
    }

    fn starts_with(
        &mut self,
        reference: &BoundReference,
        datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        let field_id = reference.field().id;

        let PrimitiveLiteral::String(datum) = datum.literal() else {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "Cannot use StartsWith operator on non-string values",
            ));
        };

        self.calc_row_selection(
            field_id,
            |min, max, nulls| {
                if matches!(nulls, PageNullCount::AllNull) {
                    return Ok(false);
                }

                if let Some(lower_bound) = min {
                    let PrimitiveLiteral::String(lower_bound) = lower_bound.literal() else {
                        return Err(Error::new(
                            ErrorKind::Unexpected,
                            "Cannot use StartsWith operator on non-string lower_bound value",
                        ));
                    };

                    let prefix_length = lower_bound.chars().count().min(datum.chars().count());

                    // truncate lower bound so that its length
                    // is not greater than the length of prefix
                    let truncated_lower_bound =
                        lower_bound.chars().take(prefix_length).collect::<String>();
                    if datum < &truncated_lower_bound {
                        return Ok(false);
                    }
                }

                if let Some(upper_bound) = max {
                    let PrimitiveLiteral::String(upper_bound) = upper_bound.literal() else {
                        return Err(Error::new(
                            ErrorKind::Unexpected,
                            "Cannot use StartsWith operator on non-string upper_bound value",
                        ));
                    };

                    let prefix_length = upper_bound.chars().count().min(datum.chars().count());

                    // truncate upper bound so that its length
                    // is not greater than the length of prefix
                    let truncated_upper_bound =
                        upper_bound.chars().take(prefix_length).collect::<String>();
                    if datum > &truncated_upper_bound {
                        return Ok(false);
                    }
                }

                Ok(true)
            },
            MissingColBehavior::CantMatch,
        )
    }

    fn not_starts_with(
        &mut self,
        reference: &BoundReference,
        datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        let field_id = reference.field().id;

        // notStartsWith will match unless all values must start with the prefix.
        // This happens when the lower and upper bounds both start with the prefix.

        let PrimitiveLiteral::String(prefix) = datum.literal() else {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "Cannot use StartsWith operator on non-string values",
            ));
        };

        self.calc_row_selection(
            field_id,
            |min, max, nulls| {
                if !matches!(nulls, PageNullCount::NoneNull) {
                    return Ok(true);
                }

                let Some(lower_bound) = min else {
                    return Ok(true);
                };

                let PrimitiveLiteral::String(lower_bound_str) = lower_bound.literal() else {
                    return Err(Error::new(
                        ErrorKind::Unexpected,
                        "Cannot use NotStartsWith operator on non-string lower_bound value",
                    ));
                };

                if lower_bound_str < prefix {
                    // if lower is shorter than the prefix then lower doesn't start with the prefix
                    return Ok(true);
                }

                let prefix_len = prefix.chars().count();

                if lower_bound_str.chars().take(prefix_len).collect::<String>() == *prefix {
                    // lower bound matches the prefix

                    let Some(upper_bound) = max else {
                        return Ok(true);
                    };

                    let PrimitiveLiteral::String(upper_bound) = upper_bound.literal() else {
                        return Err(Error::new(
                            ErrorKind::Unexpected,
                            "Cannot use NotStartsWith operator on non-string upper_bound value",
                        ));
                    };

                    // if upper is shorter than the prefix then upper can't start with the prefix
                    if upper_bound.chars().count() < prefix_len {
                        return Ok(true);
                    }

                    if upper_bound.chars().take(prefix_len).collect::<String>() == *prefix {
                        // both bounds match the prefix, so all rows must match the
                        // prefix and therefore do not satisfy the predicate
                        return Ok(false);
                    }
                }

                Ok(true)
            },
            MissingColBehavior::MightMatch,
        )
    }

    fn r#in(
        &mut self,
        reference: &BoundReference,
        literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        let field_id = reference.field().id;

        if literals.len() > IN_PREDICATE_LIMIT {
            // skip evaluating the predicate if the number of values is too big
            return self.select_all_rows();
        }
        self.calc_row_selection(
            field_id,
            |min, max, nulls| Ok(Self::in_keeps_page(min, max, nulls, literals)),
            MissingColBehavior::CantMatch,
        )
    }

    fn not_in(
        &mut self,
        _reference: &BoundReference,
        _literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        // Because the bounds are not necessarily a min or max value,
        // this cannot be answered using them. notIn(col, {X, ...})
        // with (X, Y) doesn't guarantee that X is a value in col.
        self.select_all_rows()
    }
}

struct PrunableLeafVisitor;

impl BoundPredicateVisitor for PrunableLeafVisitor {
    type T = bool;

    fn always_true(&mut self) -> Result<bool> {
        Ok(false)
    }

    fn always_false(&mut self) -> Result<bool> {
        Ok(true)
    }

    fn and(&mut self, lhs: bool, rhs: bool) -> Result<bool> {
        Ok(lhs || rhs)
    }

    fn or(&mut self, lhs: bool, rhs: bool) -> Result<bool> {
        Ok(lhs && rhs)
    }

    fn not(&mut self, _inner: bool) -> Result<bool> {
        Ok(true)
    }

    fn is_null(
        &mut self,
        _reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(true)
    }

    fn not_null(
        &mut self,
        _reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(true)
    }

    fn is_nan(&mut self, reference: &BoundReference, _predicate: &BoundPredicate) -> Result<bool> {
        Ok(!reference.field().field_type.is_floating_type())
    }

    fn not_nan(
        &mut self,
        _reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(false)
    }

    fn less_than(
        &mut self,
        _reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(true)
    }

    fn less_than_or_eq(
        &mut self,
        _reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(true)
    }

    fn greater_than(
        &mut self,
        _reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(true)
    }

    fn greater_than_or_eq(
        &mut self,
        _reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(true)
    }

    fn eq(
        &mut self,
        _reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(true)
    }

    fn not_eq(
        &mut self,
        _reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(false)
    }

    fn starts_with(
        &mut self,
        _reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(true)
    }

    fn not_starts_with(
        &mut self,
        _reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(false)
    }

    fn r#in(
        &mut self,
        _reference: &BoundReference,
        literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(literals.len() <= IN_PREDICATE_LIMIT)
    }

    fn not_in(
        &mut self,
        _reference: &BoundReference,
        _literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(false)
    }
}

pub(crate) fn predicate_can_prune_pages(predicate: &BoundPredicate) -> bool {
    visit(&mut PrunableLeafVisitor, predicate).unwrap_or(true)
}
