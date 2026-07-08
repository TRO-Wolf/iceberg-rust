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

//! Aggregate push-down over manifest [`DataFile`] metrics.
//!
//! This mirrors the Java `org.apache.iceberg.expressions` aggregate stack
//! (`Aggregate`, `UnboundAggregate`, `BoundAggregate`, `CountStar`,
//! `CountNonNull`, `MinAggregate`, `MaxAggregate`, and `AggregateEvaluator`).
//! It computes `count(*)` / `count(col)` / `min(col)` / `max(col)` from the
//! per-file metrics carried in manifest entries **without scanning any data**.
//!
//! # The fold (matches Java `AggregateEvaluator` exactly)
//!
//! Over a sequence of [`DataFile`]s:
//!
//! - `count(*)` = `Σ record_count`
//! - `count(col_id)` = `Σ (value_count[col_id] − null_value_count[col_id])`
//! - `min(col_id)` = `min` over `lower_bounds[col_id]` (typed [`Datum`] order)
//! - `max(col_id)` = `max` over `upper_bounds[col_id]` (typed [`Datum`] order)
//!
//! # The not-pushable invalidation (the critical safety property)
//!
//! When a required metric is **missing for any data file**, the aggregator is
//! permanently *invalidated* — exactly as Java's `NullSafeAggregator.update`
//! flips `isValid` to `false` the first time `hasValue(file)` returns false.
//! An invalidated aggregator reports `is_valid() == false` and yields `None`
//! from [`AggregateEvaluator::result`], signalling the engine to fall back to a
//! real scan. Returning a silently-wrong partial aggregate is the failure mode
//! this guards against.
//!
//! `count(*)` always has a value (every file carries a `record_count`).
//! `count(col)` requires BOTH `value_counts` and `null_value_counts` to contain
//! the column id. `min`/`max` require `lower_bounds`/`upper_bounds` to contain
//! the id, OR the column to be entirely null in that file (in which case there
//! is no bound to contribute and the running value is left unchanged).
//!
//! # Scope: `Extract` is intentionally not implemented
//!
//! The Java surface also exposes `UnboundExtract` / `BoundExtract` (the
//! `extract(col, path, type)` term used for variant shredding). That capability
//! is frontier-parked; this module deliberately does **not** build it. The only
//! aggregate term this module models is a plain column [`Reference`] — there is
//! no `Extract` term to construct, so a shredding/extract path cannot enter the
//! aggregate tree here at all. If the broader expression layer later grows an
//! `Extract` term, binding it through an aggregate must be rejected
//! (`FeatureUnsupported`) rather than silently treated as a column.

// This is parity scaffolding: the aggregate tree + evaluator are built and
// unit-tested ahead of the scan-side push-down wiring that will consume them
// (GAP_MATRIX row R150, 🟡). Until that wiring lands the public-to-crate surface
// has no in-crate caller, so the production graph reports dead code. The same
// pattern is used by `strict_metrics_evaluator.rs`.
#![allow(dead_code)]

use std::fmt::{Display, Formatter};

use crate::expr::{Bind, BoundReference, Reference};
use crate::spec::{DataFile, Datum, PrimitiveType, SchemaRef, Type};
use crate::{Error, ErrorKind, Result};

/// The aggregate operation, mirroring the aggregate arms of Java
/// `Expression.Operation` (`COUNT`, `COUNT_STAR`, `MAX`, `MIN`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AggregateOperation {
    /// `count(*)` — `Σ record_count`. Has no column term.
    CountStar,
    /// `count(col)` — non-null count from value/null counts.
    Count,
    /// `max(col)` — maximum of the per-file upper bounds.
    Max,
    /// `min(col)` — minimum of the per-file lower bounds.
    Min,
}

impl Display for AggregateOperation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AggregateOperation::CountStar => "count_star",
            AggregateOperation::Count => "count",
            AggregateOperation::Max => "max",
            AggregateOperation::Min => "min",
        };
        write!(f, "{s}")
    }
}

/// An unbound aggregate expression — an [`AggregateOperation`] over an optional
/// column term, mirroring Java `UnboundAggregate`.
///
/// `count(*)` carries no term; `count` / `min` / `max` each carry a column
/// [`Reference`]. Use [`UnboundAggregate::count_star`], [`UnboundAggregate::count`],
/// [`UnboundAggregate::min`], or [`UnboundAggregate::max`] to construct one.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct UnboundAggregate {
    op: AggregateOperation,
    term: Option<Reference>,
}

impl UnboundAggregate {
    /// `count(*)`.
    pub(crate) fn count_star() -> Self {
        Self {
            op: AggregateOperation::CountStar,
            term: None,
        }
    }

    /// `count(col)`.
    pub(crate) fn count(term: Reference) -> Self {
        Self {
            op: AggregateOperation::Count,
            term: Some(term),
        }
    }

    /// `min(col)`.
    pub(crate) fn min(term: Reference) -> Self {
        Self {
            op: AggregateOperation::Min,
            term: Some(term),
        }
    }

    /// `max(col)`.
    pub(crate) fn max(term: Reference) -> Self {
        Self {
            op: AggregateOperation::Max,
            term: Some(term),
        }
    }

    /// The aggregate operation.
    pub(crate) fn op(&self) -> AggregateOperation {
        self.op
    }
}

impl Bind for UnboundAggregate {
    type Bound = BoundAggregate;

    /// Bind the aggregate term to `schema`, mirroring Java
    /// `UnboundAggregate.bind` → `boundTerm`.
    ///
    /// `count(*)` has no term and binds trivially. The other operations require
    /// a column [`Reference`] (`Invalid aggregate term: null` in Java) and the
    /// referenced field must be a primitive type — the only kind that carries
    /// comparable metrics.
    fn bind(&self, schema: SchemaRef, case_sensitive: bool) -> Result<BoundAggregate> {
        match self.op {
            AggregateOperation::CountStar => Ok(BoundAggregate {
                op: self.op,
                term: None,
            }),
            AggregateOperation::Count | AggregateOperation::Min | AggregateOperation::Max => {
                let term = self.term.as_ref().ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Invalid aggregate term: null for {} aggregate", self.op),
                    )
                })?;

                let bound_ref = term.bind(schema, case_sensitive)?;

                // Metrics-driven aggregates only make sense over primitive
                // columns (bounds are serialized primitive single-values).
                if !matches!(bound_ref.field().field_type.as_ref(), Type::Primitive(_)) {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Cannot compute {} over non-primitive column {}",
                            self.op,
                            bound_ref.field().name
                        ),
                    ));
                }

                Ok(BoundAggregate {
                    op: self.op,
                    term: Some(bound_ref),
                })
            }
        }
    }
}

/// A bound aggregate expression — an [`AggregateOperation`] over a resolved
/// column reference, mirroring Java `BoundAggregate` / `CountStar` /
/// `CountNonNull` / `MinAggregate` / `MaxAggregate`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundAggregate {
    op: AggregateOperation,
    term: Option<BoundReference>,
}

impl BoundAggregate {
    /// The aggregate operation.
    pub(crate) fn op(&self) -> AggregateOperation {
        self.op
    }

    /// The bound column reference, or `None` for `count(*)`.
    pub(crate) fn term(&self) -> Option<&BoundReference> {
        self.term.as_ref()
    }

    /// The field id of the aggregate's column term.
    ///
    /// Returns `None` for `count(*)`, which has no column.
    fn field_id(&self) -> Option<i32> {
        self.term.as_ref().map(|t| t.field().id)
    }

    /// The primitive type of the aggregate's column term, used to type the
    /// `min`/`max` result. Returns `None` for `count(*)`.
    fn primitive_type(&self) -> Option<&PrimitiveType> {
        match self.term.as_ref()?.field().field_type.as_ref() {
            Type::Primitive(p) => Some(p),
            _ => None,
        }
    }

    /// Create a fresh, valid [`Aggregator`] for this aggregate.
    fn new_aggregator(&self) -> Aggregator {
        Aggregator {
            aggregate: self.clone(),
            is_valid: true,
            count: 0,
            value: None,
        }
    }

    /// Whether `file` carries the metric(s) this aggregate needs.
    ///
    /// Mirrors the per-aggregate `hasValue(DataFile)` in Java. When this returns
    /// `false`, the owning [`Aggregator`] is invalidated (the not-pushable case).
    fn has_value(&self, file: &DataFile) -> bool {
        match self.op {
            // CountStar.hasValue == recordCount() >= 0; record_count is u64.
            AggregateOperation::CountStar => true,
            AggregateOperation::Count => {
                let Some(id) = self.field_id() else {
                    return false;
                };
                // CountNonNull.hasValue: BOTH value_counts AND null_value_counts
                // must contain the column id.
                file.value_counts().contains_key(&id) && file.null_value_counts().contains_key(&id)
            }
            AggregateOperation::Min => {
                let Some(id) = self.field_id() else {
                    return false;
                };
                file.lower_bounds().contains_key(&id) || self.column_all_null(file, id)
            }
            AggregateOperation::Max => {
                let Some(id) = self.field_id() else {
                    return false;
                };
                file.upper_bounds().contains_key(&id) || self.column_all_null(file, id)
            }
        }
    }

    /// Whether the column `id` is entirely null in `file`.
    ///
    /// Mirrors the `allNull` short-circuit in Java `MinAggregate`/`MaxAggregate`
    /// `hasValue`: `valueCount != null && valueCount > 0 && nullCount == valueCount`.
    /// An all-null column has a value (it just contributes no bound), so the
    /// missing bound must not invalidate the aggregator.
    fn column_all_null(&self, file: &DataFile, id: i32) -> bool {
        match (
            file.value_counts().get(&id),
            file.null_value_counts().get(&id),
        ) {
            (Some(&value_count), Some(&null_count)) => value_count > 0 && null_count == value_count,
            _ => false,
        }
    }
}

/// A running aggregation over a stream of [`DataFile`]s for one
/// [`BoundAggregate`], mirroring Java `BoundAggregate.NullSafeAggregator`.
///
/// `count(*)` / `count(col)` accumulate into `count`; `min` / `max` accumulate
/// into `value`. `is_valid` latches to `false` the first time a file lacks the
/// required metric and never recovers — once invalid, [`Aggregator::result`]
/// yields `None`.
#[derive(Debug)]
struct Aggregator {
    aggregate: BoundAggregate,
    is_valid: bool,
    count: u64,
    value: Option<Datum>,
}

impl Aggregator {
    /// Fold one [`DataFile`]'s metrics into the running aggregate.
    ///
    /// Mirrors Java `NullSafeAggregator.update(DataFile)`: once invalid, do
    /// nothing; if the file lacks the required metric, invalidate; otherwise
    /// fold the per-file contribution.
    fn update(&mut self, file: &DataFile) -> Result<()> {
        if !self.is_valid {
            return Ok(());
        }

        if !self.aggregate.has_value(file) {
            // The staller: a missing required metric makes the whole aggregate
            // not pushable. The engine must fall back to a real scan.
            self.is_valid = false;
            return Ok(());
        }

        match self.aggregate.op {
            AggregateOperation::CountStar => {
                self.count = self
                    .count
                    .checked_add(file.record_count())
                    .ok_or_else(count_overflow)?;
            }
            AggregateOperation::Count => {
                // value_count − null_count; both keys are guaranteed present by
                // has_value. null_count defaults to 0 to match Java safeGet.
                let id = self
                    .aggregate
                    .field_id()
                    .expect("count(col) aggregate has a bound column term");
                let value_count = *file.value_counts().get(&id).unwrap_or(&0);
                let null_count = *file.null_value_counts().get(&id).unwrap_or(&0);
                let non_null = value_count.checked_sub(null_count).ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Invalid metrics for column {id}: null_value_count ({null_count}) \
                             exceeds value_count ({value_count})"
                        ),
                    )
                })?;
                self.count = self
                    .count
                    .checked_add(non_null)
                    .ok_or_else(count_overflow)?;
            }
            AggregateOperation::Min => {
                let id = self
                    .aggregate
                    .field_id()
                    .expect("min aggregate has a bound column term");
                if let Some(bound) = file.lower_bounds().get(&id) {
                    self.merge_value(bound, true)?;
                }
            }
            AggregateOperation::Max => {
                let id = self
                    .aggregate
                    .field_id()
                    .expect("max aggregate has a bound column term");
                if let Some(bound) = file.upper_bounds().get(&id) {
                    self.merge_value(bound, false)?;
                }
            }
        }

        Ok(())
    }

    /// Merge one bound into the running min/max value using typed [`Datum`]
    /// ordering. `keep_lesser` selects `min` (keep the smaller) vs `max`.
    fn merge_value(&mut self, candidate: &Datum, keep_lesser: bool) -> Result<()> {
        match &self.value {
            None => {
                self.value = Some(candidate.clone());
            }
            Some(current) => {
                let ordering = current.partial_cmp(candidate).ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Cannot compare bounds for {} aggregate: {current} vs {candidate}",
                            self.aggregate.op
                        ),
                    )
                })?;
                let replace = if keep_lesser {
                    ordering == std::cmp::Ordering::Greater
                } else {
                    ordering == std::cmp::Ordering::Less
                };
                if replace {
                    self.value = Some(candidate.clone());
                }
            }
        }
        Ok(())
    }

    /// Whether this aggregator is still pushable (no required metric was missing).
    fn is_valid(&self) -> bool {
        self.is_valid
    }

    /// The aggregate result, or `None` if the aggregate is not pushable.
    ///
    /// Mirrors Java `NullSafeAggregator.result`: `null` when invalid.
    fn result(&self) -> Option<AggregateValue> {
        if !self.is_valid {
            return None;
        }
        match self.aggregate.op {
            AggregateOperation::CountStar | AggregateOperation::Count => {
                Some(AggregateValue::Count(self.count))
            }
            AggregateOperation::Min | AggregateOperation::Max => {
                Some(AggregateValue::Bound(self.value.clone()))
            }
        }
    }
}

fn count_overflow() -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        "Aggregate count overflowed u64 while folding record counts",
    )
}

/// The result of one aggregator after folding over a sequence of data files.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AggregateValue {
    /// A `count(*)` / `count(col)` result.
    Count(u64),
    /// A `min` / `max` result; `None` when every contributing file had an
    /// all-null column (no bound to report) yet the aggregate is still valid.
    Bound(Option<Datum>),
}

/// Folds a set of [`BoundAggregate`]s over a sequence of [`DataFile`] metrics,
/// mirroring Java `AggregateEvaluator`.
///
/// Construct with [`AggregateEvaluator::new`], feed files via
/// [`AggregateEvaluator::update`], then read [`AggregateEvaluator::all_valid`]
/// and [`AggregateEvaluator::results`]. If any aggregator became invalid the
/// whole evaluation is not pushable and the engine must scan.
pub(crate) struct AggregateEvaluator {
    aggregators: Vec<Aggregator>,
}

impl AggregateEvaluator {
    /// Build an evaluator from a set of bound aggregates.
    pub(crate) fn new(aggregates: &[BoundAggregate]) -> Self {
        Self {
            aggregators: aggregates.iter().map(|a| a.new_aggregator()).collect(),
        }
    }

    /// Build an evaluator by binding a set of unbound aggregates to `schema`.
    pub(crate) fn create(
        aggregates: &[UnboundAggregate],
        schema: SchemaRef,
        case_sensitive: bool,
    ) -> Result<Self> {
        let bound = aggregates
            .iter()
            .map(|a| a.bind(schema.clone(), case_sensitive))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::new(&bound))
    }

    /// Fold one data file's metrics into every aggregator.
    pub(crate) fn update(&mut self, file: &DataFile) -> Result<()> {
        for aggregator in &mut self.aggregators {
            aggregator.update(file)?;
        }
        Ok(())
    }

    /// Whether every aggregator is still pushable.
    ///
    /// When this is `false`, at least one aggregate could not be computed from
    /// metrics alone and the engine must fall back to a real scan.
    pub(crate) fn all_valid(&self) -> bool {
        self.aggregators.iter().all(Aggregator::is_valid)
    }

    /// The per-aggregate results in construction order.
    ///
    /// Each entry is `None` exactly when that aggregator is not pushable.
    pub(crate) fn results(&self) -> Vec<Option<AggregateValue>> {
        self.aggregators.iter().map(Aggregator::result).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::*;
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, NestedField, PrimitiveType, Schema,
        SchemaRef, Struct, Type,
    };

    fn schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .expect("schema builds"),
        )
    }

    /// Build a data file with explicit metrics maps.
    fn data_file(
        record_count: u64,
        value_counts: HashMap<i32, u64>,
        null_counts: HashMap<i32, u64>,
        lower: HashMap<i32, Datum>,
        upper: HashMap<i32, Datum>,
    ) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("memory://data.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .partition(Struct::empty())
            .record_count(record_count)
            .file_size_in_bytes(1)
            .value_counts(value_counts)
            .null_value_counts(null_counts)
            .lower_bounds(lower)
            .upper_bounds(upper)
            .build()
            .expect("data file builds")
    }

    fn bound(agg: UnboundAggregate) -> BoundAggregate {
        agg.bind(schema(), true).expect("aggregate binds")
    }

    fn run(aggs: &[BoundAggregate], files: &[DataFile]) -> AggregateEvaluator {
        let mut eval = AggregateEvaluator::new(aggs);
        for f in files {
            eval.update(f).expect("update succeeds");
        }
        eval
    }

    // count(*) = Σ record_count.
    #[test]
    fn count_star_sums_record_counts() {
        let agg = bound(UnboundAggregate::count_star());
        let files = vec![
            data_file(
                10,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
            ),
            data_file(
                25,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
            ),
            data_file(
                7,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
            ),
        ];
        let eval = run(std::slice::from_ref(&agg), &files);
        assert!(eval.all_valid());
        assert_eq!(eval.results(), vec![Some(AggregateValue::Count(42))]);
    }

    // count(*) over zero files is 0 and still valid.
    #[test]
    fn count_star_no_files_is_zero() {
        let agg = bound(UnboundAggregate::count_star());
        let eval = run(std::slice::from_ref(&agg), &[]);
        assert!(eval.all_valid());
        assert_eq!(eval.results(), vec![Some(AggregateValue::Count(0))]);
    }

    // count(col) = Σ (value_count − null_count), no-null case.
    #[test]
    fn count_col_no_nulls() {
        let agg = bound(UnboundAggregate::count(Reference::new("id")));
        let files = vec![
            data_file(
                10,
                HashMap::from([(1, 10)]),
                HashMap::from([(1, 0)]),
                HashMap::new(),
                HashMap::new(),
            ),
            data_file(
                5,
                HashMap::from([(1, 5)]),
                HashMap::from([(1, 0)]),
                HashMap::new(),
                HashMap::new(),
            ),
        ];
        let eval = run(std::slice::from_ref(&agg), &files);
        assert!(eval.all_valid());
        assert_eq!(eval.results(), vec![Some(AggregateValue::Count(15))]);
    }

    // count(col) subtracts nulls; all-null file contributes 0.
    #[test]
    fn count_col_with_and_all_nulls() {
        let agg = bound(UnboundAggregate::count(Reference::new("id")));
        let files = vec![
            // 10 values, 3 null -> 7 non-null
            data_file(
                10,
                HashMap::from([(1, 10)]),
                HashMap::from([(1, 3)]),
                HashMap::new(),
                HashMap::new(),
            ),
            // 8 values, all null -> 0 non-null
            data_file(
                8,
                HashMap::from([(1, 8)]),
                HashMap::from([(1, 8)]),
                HashMap::new(),
                HashMap::new(),
            ),
        ];
        let eval = run(std::slice::from_ref(&agg), &files);
        assert!(eval.all_valid());
        assert_eq!(eval.results(), vec![Some(AggregateValue::Count(7))]);
    }

    // count(col): missing null_value_counts[id] for ANY file => NOT pushable.
    // This is the staller. Dropping the invalidation would return a wrong number.
    #[test]
    fn count_col_missing_null_counts_not_pushable() {
        let agg = bound(UnboundAggregate::count(Reference::new("id")));
        let files = vec![
            data_file(
                10,
                HashMap::from([(1, 10)]),
                HashMap::from([(1, 0)]),
                HashMap::new(),
                HashMap::new(),
            ),
            // value_counts present, null_value_counts MISSING for id=1
            data_file(
                5,
                HashMap::from([(1, 5)]),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
            ),
        ];
        let eval = run(std::slice::from_ref(&agg), &files);
        assert!(!eval.all_valid(), "missing null counts must invalidate");
        assert_eq!(
            eval.results(),
            vec![None],
            "must report not-pushable, never a wrong partial count"
        );
    }

    // count(col): missing value_counts[id] for ANY file => NOT pushable.
    #[test]
    fn count_col_missing_value_counts_not_pushable() {
        let agg = bound(UnboundAggregate::count(Reference::new("id")));
        let files = vec![
            // null_value_counts present, value_counts MISSING for id=1
            data_file(
                5,
                HashMap::new(),
                HashMap::from([(1, 0)]),
                HashMap::new(),
                HashMap::new(),
            ),
        ];
        let eval = run(std::slice::from_ref(&agg), &files);
        assert!(!eval.all_valid());
        assert_eq!(eval.results(), vec![None]);
    }

    // min over multi-file lower bounds, typed comparison, including reversed order.
    #[test]
    fn min_over_multiple_files() {
        let agg = bound(UnboundAggregate::min(Reference::new("id")));
        let files = vec![
            data_file(
                3,
                HashMap::from([(1, 3)]),
                HashMap::from([(1, 0)]),
                HashMap::from([(1, Datum::long(50))]),
                HashMap::from([(1, Datum::long(90))]),
            ),
            // smaller lower bound appears second (reversed order)
            data_file(
                3,
                HashMap::from([(1, 3)]),
                HashMap::from([(1, 0)]),
                HashMap::from([(1, Datum::long(10))]),
                HashMap::from([(1, Datum::long(40))]),
            ),
        ];
        let eval = run(std::slice::from_ref(&agg), &files);
        assert!(eval.all_valid());
        assert_eq!(eval.results(), vec![Some(AggregateValue::Bound(Some(
            Datum::long(10)
        )))]);
    }

    // max over multi-file upper bounds, single + multi.
    #[test]
    fn max_over_multiple_files() {
        let agg = bound(UnboundAggregate::max(Reference::new("id")));
        let files = vec![
            data_file(
                3,
                HashMap::from([(1, 3)]),
                HashMap::from([(1, 0)]),
                HashMap::from([(1, Datum::long(10))]),
                HashMap::from([(1, Datum::long(40))]),
            ),
            data_file(
                3,
                HashMap::from([(1, 3)]),
                HashMap::from([(1, 0)]),
                HashMap::from([(1, Datum::long(50))]),
                HashMap::from([(1, Datum::long(90))]),
            ),
        ];
        let eval = run(std::slice::from_ref(&agg), &files);
        assert!(eval.all_valid());
        assert_eq!(eval.results(), vec![Some(AggregateValue::Bound(Some(
            Datum::long(90)
        )))]);
    }

    // min over a single file.
    #[test]
    fn min_single_file() {
        let agg = bound(UnboundAggregate::min(Reference::new("name")));
        let files = vec![data_file(
            2,
            HashMap::from([(2, 2)]),
            HashMap::from([(2, 0)]),
            HashMap::from([(2, Datum::string("apple"))]),
            HashMap::from([(2, Datum::string("mango"))]),
        )];
        let eval = run(std::slice::from_ref(&agg), &files);
        assert!(eval.all_valid());
        assert_eq!(eval.results(), vec![Some(AggregateValue::Bound(Some(
            Datum::string("apple")
        )))]);
    }

    // min/max: missing lower/upper bound for a file with a NON-null column
    // => NOT pushable. The staller for value aggregates.
    #[test]
    fn min_missing_lower_bound_not_pushable() {
        let agg = bound(UnboundAggregate::min(Reference::new("id")));
        let files = vec![
            data_file(
                3,
                HashMap::from([(1, 3)]),
                HashMap::from([(1, 0)]),
                HashMap::from([(1, Datum::long(10))]),
                HashMap::from([(1, Datum::long(40))]),
            ),
            // a file with non-null data but NO lower bound for id=1
            data_file(
                4,
                HashMap::from([(1, 4)]),
                HashMap::from([(1, 1)]),
                HashMap::new(),
                HashMap::new(),
            ),
        ];
        let eval = run(std::slice::from_ref(&agg), &files);
        assert!(!eval.all_valid(), "missing lower bound must invalidate");
        assert_eq!(eval.results(), vec![None]);
    }

    #[test]
    fn max_missing_upper_bound_not_pushable() {
        let agg = bound(UnboundAggregate::max(Reference::new("id")));
        let files = vec![data_file(
            4,
            HashMap::from([(1, 4)]),
            HashMap::from([(1, 1)]),
            HashMap::new(),
            HashMap::new(),
        )];
        let eval = run(std::slice::from_ref(&agg), &files);
        assert!(!eval.all_valid());
        assert_eq!(eval.results(), vec![None]);
    }

    // min/max: an all-null column with NO bound is still pushable and
    // contributes no value (matches Java's allNull short-circuit in hasValue).
    #[test]
    fn min_all_null_column_keeps_valid_no_bound() {
        let agg = bound(UnboundAggregate::min(Reference::new("id")));
        let files = vec![
            // bound present, normal contribution
            data_file(
                5,
                HashMap::from([(1, 5)]),
                HashMap::from([(1, 0)]),
                HashMap::from([(1, Datum::long(20))]),
                HashMap::from([(1, Datum::long(80))]),
            ),
            // all-null column, no bound -> stays valid, no contribution
            data_file(
                6,
                HashMap::from([(1, 6)]),
                HashMap::from([(1, 6)]),
                HashMap::new(),
                HashMap::new(),
            ),
        ];
        let eval = run(std::slice::from_ref(&agg), &files);
        assert!(eval.all_valid(), "all-null column must NOT invalidate");
        assert_eq!(eval.results(), vec![Some(AggregateValue::Bound(Some(
            Datum::long(20)
        )))]);
    }

    // A fully all-null column across all files: valid, but no bound to report.
    #[test]
    fn min_all_files_all_null_yields_no_bound() {
        let agg = bound(UnboundAggregate::min(Reference::new("id")));
        let files = vec![data_file(
            6,
            HashMap::from([(1, 6)]),
            HashMap::from([(1, 6)]),
            HashMap::new(),
            HashMap::new(),
        )];
        let eval = run(std::slice::from_ref(&agg), &files);
        assert!(eval.all_valid());
        assert_eq!(eval.results(), vec![Some(AggregateValue::Bound(None))]);
    }

    // Multiple aggregates together: one invalid does not corrupt the others'
    // values, but all_valid() reports the whole evaluation as not pushable.
    #[test]
    fn mixed_aggregates_one_invalid() {
        let count_star = bound(UnboundAggregate::count_star());
        let count_id = bound(UnboundAggregate::count(Reference::new("id")));
        let aggs = vec![count_star, count_id];
        let files = vec![
            data_file(
                10,
                HashMap::from([(1, 10)]),
                HashMap::from([(1, 2)]),
                HashMap::new(),
                HashMap::new(),
            ),
            // count(id) loses its null counts -> invalid; count(*) still fine.
            data_file(
                5,
                HashMap::from([(1, 5)]),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
            ),
        ];
        let eval = run(&aggs, &files);
        assert!(!eval.all_valid());
        let results = eval.results();
        assert_eq!(results[0], Some(AggregateValue::Count(15)));
        assert_eq!(results[1], None);
    }

    // Binding a non-count aggregate without a term is an error.
    #[test]
    fn bind_min_requires_term() {
        let unbound = UnboundAggregate {
            op: AggregateOperation::Min,
            term: None,
        };
        assert!(unbound.bind(schema(), true).is_err());
    }

    // Binding a reference to a missing column is an error.
    #[test]
    fn bind_unknown_column_errors() {
        let agg = UnboundAggregate::min(Reference::new("does_not_exist"));
        assert!(agg.bind(schema(), true).is_err());
    }

    // count(col): null_value_count exceeding value_count is invalid metrics.
    #[test]
    fn count_col_null_exceeds_value_errors() {
        let agg = bound(UnboundAggregate::count(Reference::new("id")));
        let file = data_file(
            5,
            HashMap::from([(1, 5)]),
            HashMap::from([(1, 9)]),
            HashMap::new(),
            HashMap::new(),
        );
        let mut eval = AggregateEvaluator::new(std::slice::from_ref(&agg));
        let err = eval.update(&file);
        assert!(err.is_err(), "null > value must be a hard metrics error");
    }

    /// Cross-impl interop (G1) — the metrics fold against Java 1.10.0
    /// `AggregateEvaluator.result()`. Skipped in the offline gate; the harness
    /// `dev/java-interop/run-interop-aggregate.sh` sets `ICEBERG_INTEROP_AGGREGATE_DIR`.
    ///
    /// The fixture metrics + the aggregate order are hand-declared identically
    /// here and in `InteropOracle.generateAggregate` (anti-circular). Java emits
    /// its OWN fold as `java_aggregate.json`; this asserts the Rust fold matches
    /// BOTH the hand-computed expected AND Java's emitted values.
    #[test]
    fn interop_aggregate_matches_java() {
        let Ok(dir) = std::env::var("ICEBERG_INTEROP_AGGREGATE_DIR") else {
            return; // offline gate: no Java fixture present, nothing to compare
        };

        // --- the shared fixture (must mirror InteropOracle.generateAggregate) ---
        let schema: SchemaRef = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(3, "score", Type::Primitive(PrimitiveType::Double))
                        .into(),
                ])
                .build()
                .expect("schema builds"),
        );

        let files = vec![
            data_file(
                3,
                HashMap::from([(1, 3), (2, 3), (3, 3)]),
                HashMap::from([(1, 0), (2, 1), (3, 0)]),
                HashMap::from([
                    (1, Datum::long(10)),
                    (2, Datum::string("apple")),
                    (3, Datum::double(1.5)),
                ]),
                HashMap::from([
                    (1, Datum::long(40)),
                    (2, Datum::string("mango")),
                    (3, Datum::double(9.5)),
                ]),
            ),
            data_file(
                2,
                HashMap::from([(1, 2), (2, 2), (3, 2)]),
                HashMap::from([(1, 0), (2, 0), (3, 1)]),
                HashMap::from([
                    (1, Datum::long(5)),
                    (2, Datum::string("banana")),
                    (3, Datum::double(2.0)),
                ]),
                HashMap::from([
                    (1, Datum::long(50)),
                    (2, Datum::string("cherry")),
                    (3, Datum::double(8.0)),
                ]),
            ),
            data_file(
                4,
                HashMap::from([(1, 4), (2, 4), (3, 4)]),
                HashMap::from([(1, 1), (2, 2), (3, 0)]),
                HashMap::from([
                    (1, Datum::long(1)),
                    (2, Datum::string("avocado")),
                    (3, Datum::double(0.5)),
                ]),
                HashMap::from([
                    (1, Datum::long(90)),
                    (2, Datum::string("zucchini")),
                    (3, Datum::double(7.0)),
                ]),
            ),
        ];

        // Aggregate order — MUST match the Java emit order.
        let aggs: Vec<BoundAggregate> = [
            UnboundAggregate::count_star(),
            UnboundAggregate::count(Reference::new("id")),
            UnboundAggregate::count(Reference::new("name")),
            UnboundAggregate::count(Reference::new("score")),
            UnboundAggregate::min(Reference::new("id")),
            UnboundAggregate::max(Reference::new("id")),
            UnboundAggregate::min(Reference::new("name")),
            UnboundAggregate::max(Reference::new("name")),
            UnboundAggregate::min(Reference::new("score")),
            UnboundAggregate::max(Reference::new("score")),
        ]
        .iter()
        .map(|a| a.bind(schema.clone(), true).expect("aggregate binds"))
        .collect();

        let eval = run(&aggs, &files);
        assert!(eval.all_valid(), "all metrics present ⇒ pushable");
        let results = eval.results();

        // --- anchor 1: hand-computed expected (anti-circular) ---
        let expected = vec![
            Some(AggregateValue::Count(9)),                    // count(*)  = 3+2+4
            Some(AggregateValue::Count(8)),                    // count(id) = (3-0)+(2-0)+(4-1)
            Some(AggregateValue::Count(6)),                    // count(name) = (3-1)+(2-0)+(4-2)
            Some(AggregateValue::Count(8)),                    // count(score) = (3-0)+(2-1)+(4-0)
            Some(AggregateValue::Bound(Some(Datum::long(1)))), // min(id)
            Some(AggregateValue::Bound(Some(Datum::long(90)))), // max(id)
            Some(AggregateValue::Bound(Some(Datum::string("apple")))), // min(name)
            Some(AggregateValue::Bound(Some(Datum::string("zucchini")))), // max(name)
            Some(AggregateValue::Bound(Some(Datum::double(0.5)))), // min(score)
            Some(AggregateValue::Bound(Some(Datum::double(9.5)))), // max(score)
        ];
        assert_eq!(
            results, expected,
            "Rust fold must match the hand-computed expected"
        );

        // --- anchor 2: Java's own fold over the same fixture (cross-impl) ---
        let java_path = std::path::Path::new(&dir).join("java_aggregate.json");
        let java_raw = std::fs::read_to_string(&java_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", java_path.display()));
        let java: serde_json::Value = serde_json::from_str(&java_raw).expect("valid java json");
        assert_eq!(
            java["all_valid"],
            serde_json::json!(true),
            "Java agrees: pushable"
        );
        let java_results = java["results"].as_array().expect("results array");
        assert_eq!(java_results.len(), results.len(), "same aggregate count");

        for (i, (rust, jr)) in results.iter().zip(java_results).enumerate() {
            let rust = rust
                .as_ref()
                .unwrap_or_else(|| panic!("aggregate {i} unexpectedly invalid"));
            match jr["kind"].as_str().expect("kind") {
                "count" => {
                    let jv = jr["value"].as_u64().expect("count value");
                    assert_eq!(
                        *rust,
                        AggregateValue::Count(jv),
                        "count aggregate {i} == Java"
                    );
                }
                "bound" => {
                    let datum = match jr["type"].as_str().expect("bound type") {
                        "long" => Datum::long(jr["value"].as_i64().expect("long")),
                        "string" => Datum::string(jr["value"].as_str().expect("string")),
                        "double" => Datum::double(jr["value"].as_f64().expect("double")),
                        other => panic!("unhandled bound type {other}"),
                    };
                    assert_eq!(
                        *rust,
                        AggregateValue::Bound(Some(datum)),
                        "bound aggregate {i} == Java"
                    );
                }
                other => panic!("unhandled result kind {other}"),
            }
        }

        // --- the not-pushable path: drop value_counts[id] on one file ⇒ count(id) un-foldable ---
        let not_pushable = vec![
            files[0].clone(),
            data_file(
                2,
                HashMap::from([(2, 2), (3, 2)]), // no id (field 1) value-count
                HashMap::from([(1, 0), (2, 0), (3, 1)]),
                HashMap::new(),
                HashMap::new(),
            ),
            files[2].clone(),
        ];
        assert!(
            !run(&aggs, &not_pushable).all_valid(),
            "a missing metric ⇒ not pushable (engine must scan)"
        );
        let np_path = std::path::Path::new(&dir).join("java_aggregate_not_pushable.json");
        let np_raw = std::fs::read_to_string(&np_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", np_path.display()));
        let np: serde_json::Value = serde_json::from_str(&np_raw).expect("valid java json");
        assert_eq!(
            np["all_valid"],
            serde_json::json!(false),
            "Java also reports the fixture not pushable"
        );
    }
}
