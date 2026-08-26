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

//! The residual evaluator computes the part of a row filter that the partition
//! values do not decide. Rust port of Java
//! `org.apache.iceberg.expressions.ResidualEvaluator`.
//!
//! Scan planning calls [`ResidualEvaluator::residual_bound_for`] per
//! [`crate::scan::FileScanTask`]. Filter-based conflict validation is the
//! second consumer.
//!
//! For `day(utc_timestamp)` partitioning under `utc_timestamp >= a AND
//! utc_timestamp <= b`, partition value `d` gives:
//!
//! | partition value | residual |
//! |---|---|
//! | `day(a) < d < day(b)` | always true |
//! | `d == day(a)`, `d != day(b)` | `utc_timestamp >= a` |
//! | `d == day(b)`, `d != day(a)` | `utc_timestamp <= b` |
//! | `d == day(a) == day(b)` | both bounds |

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use fnv::FnvHashSet;

use crate::Result;
use crate::expr::visitors::bound_predicate_visitor::{BoundPredicateVisitor, visit};
use crate::expr::visitors::expression_evaluator::ExpressionEvaluatorVisitor;
use crate::expr::{
    BinaryExpression, Bind, BoundPredicate, BoundReference, Predicate, Reference, SetExpression,
    UnaryExpression,
};
use crate::spec::{Datum, PartitionSpecRef, Schema, SchemaRef, Struct};

/// Computes the residual of a row filter for a given partition's values.
///
/// Mirrors Java `ResidualEvaluator`. The filter is a [`BoundPredicate`] and the
/// residual is an unbound [`Predicate`]. Partition source columns are top-level
/// schema fields, so a kept leaf rebuilds its reference from the bound field
/// name.
///
/// # Memo contract
///
/// [`residual_bound_for`](Self::residual_bound_for) keys its memo on the
/// partition tuple alone. Never reuse one evaluator across snapshot schemas or
/// bind-case settings. `bind_case_sensitive` must match the stored
/// `case_sensitive` field. [`RESIDUAL_BOUND_MEMO_SOFT_CAP`] bounds memo growth
/// on a high-cardinality scan.
#[derive(Debug)]
pub(crate) struct ResidualEvaluator {
    /// `Some(spec, partition_schema)` for a partitioned spec; `None` for an
    /// unpartitioned spec (the residual is then always the whole filter).
    partitioned: Option<PartitionedState>,
    filter: BoundPredicate,
    /// Case sensitivity used to bind projected predicates to the partition type.
    case_sensitive: bool,
    /// Memo of partition tuple → bound residual. A poisoned lock is recovered:
    /// the values are pure derived data (crate-wide scan-cache policy).
    residual_bound_memo: RwLock<HashMap<Struct, Arc<BoundPredicate>>>,
}

/// Soft cap on `residual_bound_memo` entries. Past it a residual still computes
/// but is not inserted, so a high-cardinality scan cannot grow the memo without
/// bound.
const RESIDUAL_BOUND_MEMO_SOFT_CAP: usize = 8192;

/// The state needed to evaluate residuals against a non-empty partition spec.
#[derive(Debug, Clone)]
struct PartitionedState {
    spec: PartitionSpecRef,
    /// Java `spec.partitionType()`, as a [`Schema`] to bind projections against.
    partition_schema: SchemaRef,
}

impl ResidualEvaluator {
    /// Returns an evaluator for an unpartitioned spec. Every residual is the
    /// whole filter (Java `ResidualEvaluator.unpartitioned`). `case_sensitive`
    /// is stored to keep the memo contract.
    pub(crate) fn unpartitioned(filter: BoundPredicate, case_sensitive: bool) -> Self {
        Self {
            partitioned: None,
            filter,
            case_sensitive,
            residual_bound_memo: RwLock::new(HashMap::new()),
        }
    }

    /// Returns a residual evaluator for a partition spec and a bound filter.
    ///
    /// An empty spec degrades to the unpartitioned form (Java
    /// `ResidualEvaluator.of`). `schema` is the table schema the filter binds
    /// against. It gives the partition type, which the Rust `PartitionSpec`
    /// does not carry.
    pub(crate) fn of(
        spec: PartitionSpecRef,
        schema: &Schema,
        filter: BoundPredicate,
        case_sensitive: bool,
    ) -> Result<Self> {
        if spec.fields().is_empty() {
            return Ok(Self::unpartitioned(filter, case_sensitive));
        }

        let partition_type = spec.partition_type(schema)?;
        let partition_schema = Schema::builder()
            .with_schema_id(spec.spec_id())
            .with_fields(partition_type.fields().to_vec())
            .build()?;

        Ok(Self {
            partitioned: Some(PartitionedState {
                spec,
                partition_schema: SchemaRef::new(partition_schema),
            }),
            filter,
            case_sensitive,
            residual_bound_memo: RwLock::new(HashMap::new()),
        })
    }

    /// Returns the residual of the filter for the given partition values.
    ///
    /// `partition` is the partition tuple (Java `StructLike partitionData`). An
    /// unpartitioned evaluator returns the filter verbatim.
    pub(crate) fn residual_for(&self, partition: &Struct) -> Result<Predicate> {
        let Some(state) = &self.partitioned else {
            return bound_to_unbound(&self.filter);
        };

        let mut visitor = ResidualVisitor {
            partition,
            spec: state,
            case_sensitive: self.case_sensitive,
        };
        visit(&mut visitor, &self.filter)
    }

    /// Returns the residual of the filter for `partition`, already bound to
    /// `snapshot_schema` under `bind_case_sensitive`.
    ///
    /// The memo keys on the partition tuple only. It is valid only while the
    /// filter and partition state stay fixed and `bind_case_sensitive` matches
    /// the first insert. Past [`RESIDUAL_BOUND_MEMO_SOFT_CAP`] a partition
    /// still computes but is not inserted.
    pub(crate) fn residual_bound_for(
        &self,
        partition: &Struct,
        snapshot_schema: SchemaRef,
        bind_case_sensitive: bool,
    ) -> Result<Arc<BoundPredicate>> {
        self.residual_bound_for_with_cap(
            partition,
            snapshot_schema,
            bind_case_sensitive,
            RESIDUAL_BOUND_MEMO_SOFT_CAP,
        )
    }

    /// Same as [`residual_bound_for`](Self::residual_bound_for) with an explicit
    /// insert soft-cap. Tests pass a small cap to pin the gate without filling
    /// 8192 partitions.
    fn residual_bound_for_with_cap(
        &self,
        partition: &Struct,
        snapshot_schema: SchemaRef,
        bind_case_sensitive: bool,
        soft_cap: usize,
    ) -> Result<Arc<BoundPredicate>> {
        // The memo does not key on the bind flag. A caller that changes it
        // between calls gets wrong memo hits.
        debug_assert_eq!(
            bind_case_sensitive, self.case_sensitive,
            "residual_bound_for bind_case_sensitive must match ResidualEvaluator.case_sensitive"
        );

        {
            let read = self
                .residual_bound_memo
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(cached) = read.get(partition) {
                return Ok(cached.clone());
            }
        }

        let residual = self.residual_for(partition)?;
        let bound = Arc::new(residual.bind(snapshot_schema, bind_case_sensitive)?);

        let mut write = self
            .residual_bound_memo
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Double-check under the write lock. A concurrent writer can insert this
        // partition, and fill the map to the cap, between the read miss and this
        // acquire. Prefer the memoized Arc so Arc identity holds at that edge.
        if let Some(cached) = write.get(partition) {
            return Ok(cached.clone());
        }
        // Soft cap: skip insert when full so cardinality spikes cannot OOM the memo.
        if write.len() >= soft_cap {
            return Ok(bound);
        }
        Ok(write.entry(partition.clone()).or_insert(bound).clone())
    }

    /// Memo size, for the soft-cap and poison pins.
    #[cfg(test)]
    fn residual_memo_len(&self) -> usize {
        self.residual_bound_memo
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// Panics while holding the memo write guard to leave the lock poisoned.
    #[cfg(test)]
    fn poison_residual_memo_for_test(&self) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = self
                .residual_bound_memo
                .write()
                .expect("test setup: residual memo must not already be poisoned");
            panic!("deliberately poison residual_bound_memo");
        }));
        assert!(result.is_err(), "the poisoning closure must panic");
        assert!(
            self.residual_bound_memo.is_poisoned(),
            "test setup: residual_bound_memo must be poisoned"
        );
    }
}

/// Java `ResidualEvaluator.ResidualVisitor`: reduces each leaf against the
/// partition values.
struct ResidualVisitor<'a> {
    partition: &'a Struct,
    spec: &'a PartitionedState,
    case_sensitive: bool,
}

impl ResidualVisitor<'_> {
    /// Applies Java `ResidualVisitor.predicate(BoundPredicate)`. For each
    /// partition field on the predicate's source column, a true strict
    /// projection gives `AlwaysTrue` and a false inclusive projection gives
    /// `AlwaysFalse`. No conclusive field keeps the original predicate.
    fn reduce_leaf(
        &self,
        reference: &BoundReference,
        predicate: &BoundPredicate,
    ) -> Result<Predicate> {
        let source_id = reference.field().id;
        let parts: Vec<_> = self
            .spec
            .spec
            .fields()
            .iter()
            .filter(|field| field.source_id == source_id)
            .collect();
        if parts.is_empty() {
            return bound_to_unbound(predicate);
        }

        for part in parts {
            // Strict projection is true only if every row in the partition
            // satisfies the predicate, so the predicate drops.
            if let Some(strict) = part.transform.strict_project(&part.name, predicate)?
                && self.evaluate_projection(strict)?
            {
                return Ok(Predicate::AlwaysTrue);
            }

            // Inclusive projection is false only if no row in the partition
            // satisfies the predicate, so the residual is false.
            if let Some(inclusive) = part.transform.project(&part.name, predicate)?
                && !self.evaluate_projection(inclusive)?
            {
                return Ok(Predicate::AlwaysFalse);
            }
        }

        bound_to_unbound(predicate)
    }

    /// Binds a projected predicate to the partition type and evaluates it
    /// against the partition values. A constant projection short-circuits.
    fn evaluate_projection(&self, projection: Predicate) -> Result<bool> {
        let bound = projection.bind(self.spec.partition_schema.clone(), self.case_sensitive)?;
        match bound {
            BoundPredicate::AlwaysTrue => Ok(true),
            BoundPredicate::AlwaysFalse => Ok(false),
            other => {
                // The expression evaluator cannot run a `Not`. The transform
                // projections never emit one, but normalize anyway.
                let mut evaluator = ExpressionEvaluatorVisitor::new(self.partition);
                visit(&mut evaluator, &other.rewrite_not())
            }
        }
    }
}

impl BoundPredicateVisitor for ResidualVisitor<'_> {
    type T = Predicate;

    fn always_true(&mut self) -> Result<Predicate> {
        Ok(Predicate::AlwaysTrue)
    }

    fn always_false(&mut self) -> Result<Predicate> {
        Ok(Predicate::AlwaysFalse)
    }

    fn and(&mut self, lhs: Predicate, rhs: Predicate) -> Result<Predicate> {
        Ok(lhs.and(rhs))
    }

    fn or(&mut self, lhs: Predicate, rhs: Predicate) -> Result<Predicate> {
        Ok(lhs.or(rhs))
    }

    fn not(&mut self, inner: Predicate) -> Result<Predicate> {
        Ok(simplifying_not(inner))
    }

    fn is_null(
        &mut self,
        reference: &BoundReference,
        predicate: &BoundPredicate,
    ) -> Result<Predicate> {
        self.reduce_leaf(reference, predicate)
    }

    fn not_null(
        &mut self,
        reference: &BoundReference,
        predicate: &BoundPredicate,
    ) -> Result<Predicate> {
        self.reduce_leaf(reference, predicate)
    }

    fn is_nan(
        &mut self,
        reference: &BoundReference,
        predicate: &BoundPredicate,
    ) -> Result<Predicate> {
        self.reduce_leaf(reference, predicate)
    }

    fn not_nan(
        &mut self,
        reference: &BoundReference,
        predicate: &BoundPredicate,
    ) -> Result<Predicate> {
        self.reduce_leaf(reference, predicate)
    }

    fn less_than(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        predicate: &BoundPredicate,
    ) -> Result<Predicate> {
        self.reduce_leaf(reference, predicate)
    }

    fn less_than_or_eq(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        predicate: &BoundPredicate,
    ) -> Result<Predicate> {
        self.reduce_leaf(reference, predicate)
    }

    fn greater_than(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        predicate: &BoundPredicate,
    ) -> Result<Predicate> {
        self.reduce_leaf(reference, predicate)
    }

    fn greater_than_or_eq(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        predicate: &BoundPredicate,
    ) -> Result<Predicate> {
        self.reduce_leaf(reference, predicate)
    }

    fn eq(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        predicate: &BoundPredicate,
    ) -> Result<Predicate> {
        self.reduce_leaf(reference, predicate)
    }

    fn not_eq(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        predicate: &BoundPredicate,
    ) -> Result<Predicate> {
        self.reduce_leaf(reference, predicate)
    }

    fn starts_with(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        predicate: &BoundPredicate,
    ) -> Result<Predicate> {
        self.reduce_leaf(reference, predicate)
    }

    fn not_starts_with(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        predicate: &BoundPredicate,
    ) -> Result<Predicate> {
        self.reduce_leaf(reference, predicate)
    }

    fn r#in(
        &mut self,
        reference: &BoundReference,
        _literals: &FnvHashSet<Datum>,
        predicate: &BoundPredicate,
    ) -> Result<Predicate> {
        self.reduce_leaf(reference, predicate)
    }

    fn not_in(
        &mut self,
        reference: &BoundReference,
        _literals: &FnvHashSet<Datum>,
        predicate: &BoundPredicate,
    ) -> Result<Predicate> {
        self.reduce_leaf(reference, predicate)
    }
}

/// Reconstructs an unbound [`Predicate`] from a bound one, for a kept leaf and
/// for the unpartitioned evaluator. A leaf rebuilds its reference from the bound
/// field name, which is valid because partition sources are top-level fields.
fn bound_to_unbound(predicate: &BoundPredicate) -> Result<Predicate> {
    Ok(match predicate {
        BoundPredicate::AlwaysTrue => Predicate::AlwaysTrue,
        BoundPredicate::AlwaysFalse => Predicate::AlwaysFalse,
        BoundPredicate::And(expr) => {
            let [left, right] = expr.inputs();
            bound_to_unbound(left)?.and(bound_to_unbound(right)?)
        }
        BoundPredicate::Or(expr) => {
            let [left, right] = expr.inputs();
            bound_to_unbound(left)?.or(bound_to_unbound(right)?)
        }
        BoundPredicate::Not(expr) => {
            let [inner] = expr.inputs();
            simplifying_not(bound_to_unbound(inner)?)
        }
        BoundPredicate::Unary(expr) => Predicate::Unary(UnaryExpression::new(
            expr.op(),
            unbound_reference(expr.term()),
        )),
        BoundPredicate::Binary(expr) => Predicate::Binary(BinaryExpression::new(
            expr.op(),
            unbound_reference(expr.term()),
            expr.literal().clone(),
        )),
        BoundPredicate::Set(expr) => Predicate::Set(SetExpression::new(
            expr.op(),
            unbound_reference(expr.term()),
            expr.literals().clone(),
        )),
    })
}

/// Builds an unbound [`Reference`] from a bound reference's field name.
fn unbound_reference(reference: &BoundReference) -> Reference {
    Reference::new(reference.field().name.clone())
}

/// Negates a residual like Java `Expressions.not`, folding constants and double
/// negation. The `Predicate` `!` operator does not fold constants, so this must.
fn simplifying_not(inner: Predicate) -> Predicate {
    match inner {
        Predicate::AlwaysTrue => Predicate::AlwaysFalse,
        Predicate::AlwaysFalse => Predicate::AlwaysTrue,
        Predicate::Not(expr) => {
            let [child] = expr.inputs();
            child.clone()
        }
        other => !other,
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Not;
    use std::sync::Arc;

    use super::*;
    use crate::spec::{
        Literal, NestedField, PartitionSpec, PrimitiveLiteral, PrimitiveType, Transform, Type,
        UnboundPartitionField,
    };

    /// Columns `ts` (id 1), `id` (id 2), and optional `name` (id 3).
    fn day_example_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_fields(vec![
                    Arc::new(NestedField::required(
                        1,
                        "ts",
                        Type::Primitive(PrimitiveType::Timestamp),
                    )),
                    Arc::new(NestedField::required(
                        2,
                        "id",
                        Type::Primitive(PrimitiveType::Int),
                    )),
                    Arc::new(NestedField::optional(
                        3,
                        "name",
                        Type::Primitive(PrimitiveType::String),
                    )),
                ])
                .build()
                .expect("schema builds"),
        )
    }

    /// `PARTITIONED BY day(ts)` over [`day_example_schema`].
    fn day_partition_spec(schema: SchemaRef) -> PartitionSpecRef {
        Arc::new(
            PartitionSpec::builder(schema)
                .with_spec_id(1)
                .add_unbound_field(
                    UnboundPartitionField::builder()
                        .source_id(1)
                        .name("ts_day".to_string())
                        .field_id(1000)
                        .transform(Transform::Day)
                        .build(),
                )
                .expect("add day field")
                .build()
                .expect("spec builds"),
        )
    }

    /// Microseconds since the Unix epoch for a UTC date-time.
    fn micros(datetime: &str) -> i64 {
        match Datum::timestamp_from_str(datetime)
            .expect("valid timestamp")
            .literal()
        {
            PrimitiveLiteral::Long(value) => *value,
            other => panic!("expected a Long timestamp literal, got {other:?}"),
        }
    }

    /// The `day` partition tuple for a UTC date string.
    fn day_partition(date: &str) -> Struct {
        let day_value = match Datum::date_from_str(date).expect("valid date").literal() {
            PrimitiveLiteral::Int(value) => *value,
            other => panic!("expected an Int date literal, got {other:?}"),
        };
        Struct::from_iter([Some(Literal::date(day_value))])
    }

    /// Builds the bound `ts >= a AND ts <= b` filter from the day-example javadoc.
    fn ts_between_filter(schema: SchemaRef, a: &str, b: &str) -> BoundPredicate {
        Reference::new("ts")
            .greater_than_or_equal_to(Datum::timestamp_micros(micros(a)))
            .and(Reference::new("ts").less_than_or_equal_to(Datum::timestamp_micros(micros(b))))
            .bind(schema, true)
            .expect("filter binds")
    }

    // ---- The Javadoc day(utc_timestamp) worked example: 4 residual cases ----

    #[test]
    fn test_day_example_partition_strictly_between_bounds_reduces_to_always_true() {
        // d > day(a) and d < day(b) => residual is always true.
        let schema = day_example_schema();
        let spec = day_partition_spec(schema.clone());
        let filter =
            ts_between_filter(schema.clone(), "2021-01-01T10:00:00", "2021-01-31T10:00:00");
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        let residual = evaluator
            .residual_for(&day_partition("2021-01-15"))
            .unwrap();
        assert_eq!(residual, Predicate::AlwaysTrue);
    }

    #[test]
    fn test_day_example_partition_equals_lower_bound_keeps_lower_predicate_only() {
        // d == day(a) and d != day(b) => residual is ts >= a (lower kept, upper dropped).
        let schema = day_example_schema();
        let spec = day_partition_spec(schema.clone());
        let filter =
            ts_between_filter(schema.clone(), "2021-01-01T10:00:00", "2021-01-31T10:00:00");
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        let residual = evaluator
            .residual_for(&day_partition("2021-01-01"))
            .unwrap();

        let expected = Reference::new("ts")
            .greater_than_or_equal_to(Datum::timestamp_micros(micros("2021-01-01T10:00:00")));
        assert_eq!(residual, expected);
    }

    #[test]
    fn test_day_example_partition_equals_upper_bound_keeps_upper_predicate_only() {
        // d == day(b) and d != day(a) => residual is ts <= b (upper kept, lower dropped).
        let schema = day_example_schema();
        let spec = day_partition_spec(schema.clone());
        let filter =
            ts_between_filter(schema.clone(), "2021-01-01T10:00:00", "2021-01-31T10:00:00");
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        let residual = evaluator
            .residual_for(&day_partition("2021-01-31"))
            .unwrap();

        let expected = Reference::new("ts")
            .less_than_or_equal_to(Datum::timestamp_micros(micros("2021-01-31T10:00:00")));
        assert_eq!(residual, expected);
    }

    #[test]
    fn test_day_example_partition_equals_both_bounds_keeps_both_predicates() {
        // d == day(a) == day(b) => residual is ts >= a AND ts <= b (both kept).
        let schema = day_example_schema();
        let spec = day_partition_spec(schema.clone());
        let filter =
            ts_between_filter(schema.clone(), "2021-01-10T08:00:00", "2021-01-10T20:00:00");
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        let residual = evaluator
            .residual_for(&day_partition("2021-01-10"))
            .unwrap();

        let expected = Reference::new("ts")
            .greater_than_or_equal_to(Datum::timestamp_micros(micros("2021-01-10T08:00:00")))
            .and(
                Reference::new("ts")
                    .less_than_or_equal_to(Datum::timestamp_micros(micros("2021-01-10T20:00:00"))),
            );
        assert_eq!(residual, expected);
    }

    // ---- Identity partition ----

    fn identity_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_fields(vec![Arc::new(NestedField::required(
                    1,
                    "category",
                    Type::Primitive(PrimitiveType::Int),
                ))])
                .build()
                .expect("schema builds"),
        )
    }

    fn identity_spec(schema: SchemaRef) -> PartitionSpecRef {
        Arc::new(
            PartitionSpec::builder(schema)
                .with_spec_id(1)
                .add_unbound_field(
                    UnboundPartitionField::builder()
                        .source_id(1)
                        .name("category".to_string())
                        .field_id(1000)
                        .transform(Transform::Identity)
                        .build(),
                )
                .expect("add identity field")
                .build()
                .expect("spec builds"),
        )
    }

    #[test]
    fn test_identity_partition_eq_matching_value_reduces_to_always_true() {
        let schema = identity_schema();
        let spec = identity_spec(schema.clone());
        let filter = Reference::new("category")
            .equal_to(Datum::int(5))
            .bind(schema.clone(), true)
            .unwrap();
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        let residual = evaluator
            .residual_for(&Struct::from_iter([Some(Literal::int(5))]))
            .unwrap();
        assert_eq!(residual, Predicate::AlwaysTrue);
    }

    #[test]
    fn test_identity_partition_eq_non_matching_value_reduces_to_always_false() {
        let schema = identity_schema();
        let spec = identity_spec(schema.clone());
        let filter = Reference::new("category")
            .equal_to(Datum::int(5))
            .bind(schema.clone(), true)
            .unwrap();
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        let residual = evaluator
            .residual_for(&Struct::from_iter([Some(Literal::int(7))]))
            .unwrap();
        assert_eq!(residual, Predicate::AlwaysFalse);
    }

    // ---- Bucket partition (non-invertible) ----

    #[test]
    fn test_bucket_partition_keeps_predicate_unchanged() {
        use crate::transform::create_transform_function;

        let schema = identity_schema();
        let spec = Arc::new(
            PartitionSpec::builder(schema.clone())
                .with_spec_id(1)
                .add_unbound_field(
                    UnboundPartitionField::builder()
                        .source_id(1)
                        .name("category_bucket".to_string())
                        .field_id(1000)
                        .transform(Transform::Bucket(16))
                        .build(),
                )
                .expect("add bucket field")
                .build()
                .expect("spec builds"),
        );
        let filter = Reference::new("category")
            .equal_to(Datum::int(5))
            .bind(schema.clone(), true)
            .unwrap();
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        // Use the partition a `category = 5` row lands in. The inclusive
        // projection is then true, and bucket has no strict projection for
        // `eq`, so the predicate survives.
        let bucket_value = match create_transform_function(&Transform::Bucket(16))
            .unwrap()
            .transform_literal(&Datum::int(5))
            .unwrap()
            .expect("bucket of 5")
            .literal()
        {
            PrimitiveLiteral::Int(value) => *value,
            other => panic!("expected an Int bucket literal, got {other:?}"),
        };

        let residual = evaluator
            .residual_for(&Struct::from_iter([Some(Literal::int(bucket_value))]))
            .unwrap();
        let expected = Reference::new("category").equal_to(Datum::int(5));
        assert_eq!(residual, expected);
    }

    // ---- Predicate on a non-partition column ----

    #[test]
    fn test_predicate_on_non_partition_column_is_kept() {
        let schema = day_example_schema();
        let spec = day_partition_spec(schema.clone());
        // `id` (column 2) is not a partition source column.
        let filter = Reference::new("id")
            .greater_than(Datum::int(100))
            .bind(schema.clone(), true)
            .unwrap();
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        let residual = evaluator
            .residual_for(&day_partition("2021-01-01"))
            .unwrap();
        let expected = Reference::new("id").greater_than(Datum::int(100));
        assert_eq!(residual, expected);
    }

    // ---- Unpartitioned spec ----

    #[test]
    fn test_unpartitioned_spec_returns_filter_verbatim() {
        let schema = day_example_schema();
        let filter = Reference::new("id")
            .greater_than(Datum::int(100))
            .and(
                Reference::new("ts")
                    .less_than_or_equal_to(Datum::timestamp_micros(micros("2021-01-31T10:00:00"))),
            )
            .bind(schema.clone(), true)
            .unwrap();
        // An empty spec degrades to unpartitioned via `of`.
        let empty_spec = Arc::new(PartitionSpec::unpartition_spec());
        let evaluator = ResidualEvaluator::of(empty_spec, &schema, filter, true).unwrap();

        let residual = evaluator
            .residual_for(&Struct::from_iter(Vec::<Option<Literal>>::new()))
            .unwrap();
        let expected = Reference::new("id").greater_than(Datum::int(100)).and(
            Reference::new("ts")
                .less_than_or_equal_to(Datum::timestamp_micros(micros("2021-01-31T10:00:00"))),
        );
        assert_eq!(residual, expected);
    }

    #[test]
    fn test_unpartitioned_constructor_returns_filter_verbatim() {
        let schema = day_example_schema();
        let filter = Reference::new("id")
            .greater_than(Datum::int(100))
            .bind(schema.clone(), true)
            .unwrap();
        let evaluator = ResidualEvaluator::unpartitioned(filter, true);

        let residual = evaluator
            .residual_for(&Struct::from_iter(Vec::<Option<Literal>>::new()))
            .unwrap();
        assert_eq!(residual, Reference::new("id").greater_than(Datum::int(100)));
    }

    #[test]
    fn test_unpartitioned_spec_round_trips_a_not_filter_keeping_the_negation() {
        // Only the unpartitioned path reaches the And/Or/Not arms of
        // `bound_to_unbound`. Pin the `Not` arm: dropping the negation would
        // return `id > 100`.
        let schema = day_example_schema();
        let filter = Reference::new("id")
            .greater_than(Datum::int(100))
            .not()
            .bind(schema.clone(), true)
            .unwrap();
        let evaluator = ResidualEvaluator::unpartitioned(filter, true);

        let residual = evaluator
            .residual_for(&Struct::from_iter(Vec::<Option<Literal>>::new()))
            .unwrap();
        assert_eq!(
            residual,
            Reference::new("id").greater_than(Datum::int(100)).not()
        );
    }

    // ---- AlwaysTrue / AlwaysFalse filters ----

    #[test]
    fn test_always_true_filter_passes_through() {
        let schema = day_example_schema();
        let spec = day_partition_spec(schema.clone());
        let filter = Predicate::AlwaysTrue.bind(schema.clone(), true).unwrap();
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        let residual = evaluator
            .residual_for(&day_partition("2021-01-01"))
            .unwrap();
        assert_eq!(residual, Predicate::AlwaysTrue);
    }

    #[test]
    fn test_always_false_filter_passes_through() {
        let schema = day_example_schema();
        let spec = day_partition_spec(schema.clone());
        let filter = Predicate::AlwaysFalse.bind(schema.clone(), true).unwrap();
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        let residual = evaluator
            .residual_for(&day_partition("2021-01-01"))
            .unwrap();
        assert_eq!(residual, Predicate::AlwaysFalse);
    }

    // ---- Mixed reducible + non-reducible leaves under and / or / not ----

    #[test]
    fn test_and_of_reducible_and_non_reducible_keeps_only_the_non_reducible() {
        // The identity partition leaf reduces to AlwaysTrue. AND with the
        // surviving non-partition leaf simplifies to that survivor.
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    Arc::new(NestedField::required(
                        1,
                        "category",
                        Type::Primitive(PrimitiveType::Int),
                    )),
                    Arc::new(NestedField::required(
                        2,
                        "id",
                        Type::Primitive(PrimitiveType::Int),
                    )),
                ])
                .build()
                .unwrap(),
        );
        let spec = identity_spec_with_source(schema.clone());
        let filter = Reference::new("category")
            .equal_to(Datum::int(5))
            .and(Reference::new("id").greater_than(Datum::int(100)))
            .bind(schema.clone(), true)
            .unwrap();
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        let residual = evaluator
            .residual_for(&Struct::from_iter([Some(Literal::int(5))]))
            .unwrap();
        // category==5 → AlwaysTrue; AND AlwaysTrue id>100 → id>100.
        assert_eq!(residual, Reference::new("id").greater_than(Datum::int(100)));
    }

    #[test]
    fn test_and_short_circuits_to_false_when_partition_excludes_the_partition_leaf() {
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    Arc::new(NestedField::required(
                        1,
                        "category",
                        Type::Primitive(PrimitiveType::Int),
                    )),
                    Arc::new(NestedField::required(
                        2,
                        "id",
                        Type::Primitive(PrimitiveType::Int),
                    )),
                ])
                .build()
                .unwrap(),
        );
        let spec = identity_spec_with_source(schema.clone());
        let filter = Reference::new("category")
            .equal_to(Datum::int(5))
            .and(Reference::new("id").greater_than(Datum::int(100)))
            .bind(schema.clone(), true)
            .unwrap();
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        // partition category=7 → the partition leaf is AlwaysFalse → AND is AlwaysFalse.
        let residual = evaluator
            .residual_for(&Struct::from_iter([Some(Literal::int(7))]))
            .unwrap();
        assert_eq!(residual, Predicate::AlwaysFalse);
    }

    #[test]
    fn test_or_of_reducible_true_and_non_reducible_short_circuits_to_true() {
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    Arc::new(NestedField::required(
                        1,
                        "category",
                        Type::Primitive(PrimitiveType::Int),
                    )),
                    Arc::new(NestedField::required(
                        2,
                        "id",
                        Type::Primitive(PrimitiveType::Int),
                    )),
                ])
                .build()
                .unwrap(),
        );
        let spec = identity_spec_with_source(schema.clone());
        let filter = Reference::new("category")
            .equal_to(Datum::int(5))
            .or(Reference::new("id").greater_than(Datum::int(100)))
            .bind(schema.clone(), true)
            .unwrap();
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        // category=5 → AlwaysTrue; OR anything → AlwaysTrue.
        let residual = evaluator
            .residual_for(&Struct::from_iter([Some(Literal::int(5))]))
            .unwrap();
        assert_eq!(residual, Predicate::AlwaysTrue);
    }

    /// An identity spec over `category` (id 1) for a 2-column schema.
    fn identity_spec_with_source(schema: SchemaRef) -> PartitionSpecRef {
        Arc::new(
            PartitionSpec::builder(schema)
                .with_spec_id(1)
                .add_unbound_field(
                    UnboundPartitionField::builder()
                        .source_id(1)
                        .name("category".to_string())
                        .field_id(1000)
                        .transform(Transform::Identity)
                        .build(),
                )
                .expect("add identity field")
                .build()
                .expect("spec builds"),
        )
    }

    // ---- not(...) over a partition leaf ----

    #[test]
    fn test_not_over_partition_leaf_negates_the_reduced_constant() {
        let schema = identity_schema();
        let spec = identity_spec(schema.clone());
        // NOT(category == 5): the inner leaf reduces, then `not` negates the constant.
        let filter = Reference::new("category")
            .equal_to(Datum::int(5))
            .not()
            .bind(schema.clone(), true)
            .unwrap();
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        // partition category=5: inner is AlwaysTrue → NOT → AlwaysFalse.
        let residual_match = evaluator
            .residual_for(&Struct::from_iter([Some(Literal::int(5))]))
            .unwrap();
        assert_eq!(residual_match, Predicate::AlwaysFalse);

        // partition category=7: inner is AlwaysFalse → NOT → AlwaysTrue.
        let residual_miss = evaluator
            .residual_for(&Struct::from_iter([Some(Literal::int(7))]))
            .unwrap();
        assert_eq!(residual_miss, Predicate::AlwaysTrue);
    }

    // ---- Truncate partition ----

    /// A schema with a single required `amount` int column (id 1).
    fn amount_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_fields(vec![Arc::new(NestedField::required(
                    1,
                    "amount",
                    Type::Primitive(PrimitiveType::Int),
                ))])
                .build()
                .expect("schema builds"),
        )
    }

    /// `PARTITIONED BY truncate(amount, 10)` over [`amount_schema`].
    fn truncate_spec(schema: SchemaRef) -> PartitionSpecRef {
        Arc::new(
            PartitionSpec::builder(schema)
                .with_spec_id(1)
                .add_unbound_field(
                    UnboundPartitionField::builder()
                        .source_id(1)
                        .name("amount_trunc".to_string())
                        .field_id(1000)
                        .transform(Transform::Truncate(10))
                        .build(),
                )
                .expect("add truncate field")
                .build()
                .expect("spec builds"),
        )
    }

    #[test]
    fn test_truncate_partition_reduces_all_three_ways() {
        // filter: amount >= 15, partitioned by truncate(amount, 10).
        let schema = amount_schema();
        let spec = truncate_spec(schema.clone());
        let filter = Reference::new("amount")
            .greater_than_or_equal_to(Datum::int(15))
            .bind(schema.clone(), true)
            .unwrap();
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        // truncate=10 holds 10..=19, which straddles 15. Neither projection is
        // conclusive, so the predicate is kept.
        let straddling = evaluator
            .residual_for(&Struct::from_iter([Some(Literal::int(10))]))
            .unwrap();
        assert_eq!(
            straddling,
            Reference::new("amount").greater_than_or_equal_to(Datum::int(15))
        );

        // truncate=20 holds 20..=29, all >= 15, so strict projection is true.
        let all_match = evaluator
            .residual_for(&Struct::from_iter([Some(Literal::int(20))]))
            .unwrap();
        assert_eq!(all_match, Predicate::AlwaysTrue);

        // truncate=0 holds 0..=9, all < 15, so inclusive projection is false.
        let none_match = evaluator
            .residual_for(&Struct::from_iter([Some(Literal::int(0))]))
            .unwrap();
        assert_eq!(none_match, Predicate::AlwaysFalse);
    }

    // ---- Temporal (year) partition ----

    /// The `year(ts)` partition value for a UTC date-time.
    fn year_partition_value(datetime: &str) -> i32 {
        use crate::transform::create_transform_function;
        match create_transform_function(&Transform::Year)
            .unwrap()
            .transform_literal(&Datum::timestamp_micros(micros(datetime)))
            .unwrap()
            .expect("year of timestamp")
            .literal()
        {
            PrimitiveLiteral::Int(value) => *value,
            other => panic!("expected an Int year literal, got {other:?}"),
        }
    }

    #[test]
    fn test_year_partition_keeps_both_bounds_inside_the_year_and_excludes_other_years() {
        // filter: ts >= 2021-06-01 AND ts <= 2021-06-30 (both inside year 2021),
        // partitioned by year(ts).
        let schema = day_example_schema();
        let spec = Arc::new(
            PartitionSpec::builder(schema.clone())
                .with_spec_id(1)
                .add_unbound_field(
                    UnboundPartitionField::builder()
                        .source_id(1)
                        .name("ts_year".to_string())
                        .field_id(1000)
                        .transform(Transform::Year)
                        .build(),
                )
                .expect("add year field")
                .build()
                .expect("spec builds"),
        );
        let filter =
            ts_between_filter(schema.clone(), "2021-06-01T00:00:00", "2021-06-30T00:00:00");
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        // year=2021 contains the whole range, but the year alone decides
        // neither bound, so both predicates are kept.
        let in_year = evaluator
            .residual_for(&Struct::from_iter([Some(Literal::int(
                year_partition_value("2021-06-15T00:00:00"),
            ))]))
            .unwrap();
        let expected = Reference::new("ts")
            .greater_than_or_equal_to(Datum::timestamp_micros(micros("2021-06-01T00:00:00")))
            .and(
                Reference::new("ts")
                    .less_than_or_equal_to(Datum::timestamp_micros(micros("2021-06-30T00:00:00"))),
            );
        assert_eq!(in_year, expected);

        // year=2020 is entirely before the range, so the inclusive projection of
        // `ts <= b` is false.
        let before = evaluator
            .residual_for(&Struct::from_iter([Some(Literal::int(
                year_partition_value("2020-06-15T00:00:00"),
            ))]))
            .unwrap();
        assert_eq!(before, Predicate::AlwaysFalse);

        // year=2022 is entirely after the range.
        let after = evaluator
            .residual_for(&Struct::from_iter([Some(Literal::int(
                year_partition_value("2022-06-15T00:00:00"),
            ))]))
            .unwrap();
        assert_eq!(after, Predicate::AlwaysFalse);
    }

    // ---- Void partition ----

    #[test]
    fn test_void_partition_keeps_predicate_unchanged() {
        // Java `VoidTransform.project` and `projectStrict` both return null, so
        // a void-partitioned predicate never reduces. The null partition value
        // must not panic.
        let schema = identity_schema();
        let spec = Arc::new(
            PartitionSpec::builder(schema.clone())
                .with_spec_id(1)
                .add_unbound_field(
                    UnboundPartitionField::builder()
                        .source_id(1)
                        .name("category_void".to_string())
                        .field_id(1000)
                        .transform(Transform::Void)
                        .build(),
                )
                .expect("add void field")
                .build()
                .expect("spec builds"),
        );
        let filter = Reference::new("category")
            .equal_to(Datum::int(5))
            .bind(schema.clone(), true)
            .unwrap();
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        // The void partition value is always null.
        let residual = evaluator.residual_for(&Struct::from_iter([None])).unwrap();
        assert_eq!(residual, Reference::new("category").equal_to(Datum::int(5)));
    }

    // ---- Leaf reconstruction round-trip ----
    //
    // `bound_to_unbound` could drop a negation, swap an operator, or collapse a
    // set, all silently. These pin the set and unary shapes; the other tests
    // cover only the binary shape.

    #[test]
    fn test_in_predicate_on_non_partition_column_round_trips_as_same_in() {
        let schema = day_example_schema();
        let spec = day_partition_spec(schema.clone());
        // `id` is not a partition source. `in (1, 2, 3)` must come back with the
        // same set, operator, and column.
        let filter = Reference::new("id")
            .is_in([Datum::int(1), Datum::int(2), Datum::int(3)])
            .bind(schema.clone(), true)
            .unwrap();
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        let residual = evaluator
            .residual_for(&day_partition("2021-01-01"))
            .unwrap();
        let expected = Reference::new("id").is_in([Datum::int(1), Datum::int(2), Datum::int(3)]);
        assert_eq!(residual, expected);
    }

    #[test]
    fn test_not_in_predicate_on_non_partition_column_round_trips_as_same_not_in() {
        let schema = day_example_schema();
        let spec = day_partition_spec(schema.clone());
        let filter = Reference::new("id")
            .is_in([Datum::int(1), Datum::int(2)])
            .not()
            .bind(schema.clone(), true)
            .unwrap();
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        let residual = evaluator
            .residual_for(&day_partition("2021-01-01"))
            .unwrap();
        // The binder keeps `not(in (1, 2))` as `Not(In {1, 2})`. The residual
        // must round-trip that shape, with the negation and the set intact.
        let expected = Reference::new("id")
            .is_in([Datum::int(1), Datum::int(2)])
            .not();
        assert_eq!(residual, expected);
    }

    #[test]
    fn test_is_null_unary_on_non_partition_column_round_trips_as_same_is_null() {
        let schema = day_example_schema();
        let spec = day_partition_spec(schema.clone());
        // `name` (column 3, optional) is not a partition source column.
        let filter = Reference::new("name")
            .is_null()
            .bind(schema.clone(), true)
            .unwrap();
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        let residual = evaluator
            .residual_for(&day_partition("2021-01-01"))
            .unwrap();
        assert_eq!(residual, Reference::new("name").is_null());
    }

    /// Files that share a partition reuse one residual, proven by Arc identity
    /// on the second call.
    #[test]
    fn test_residual_bound_for_memoizes_by_partition() {
        let schema = day_example_schema();
        let spec = day_partition_spec(schema.clone());
        let filter =
            ts_between_filter(schema.clone(), "2021-01-01T10:00:00", "2021-01-31T10:00:00");
        let evaluator =
            ResidualEvaluator::of(spec, &schema, filter, true).expect("evaluator must build");

        let partition = day_partition("2021-01-15");
        let first = evaluator
            .residual_bound_for(&partition, schema.clone(), true)
            .expect("first residual_bound_for must succeed");
        let second = evaluator
            .residual_bound_for(&partition, schema.clone(), true)
            .expect("memoized residual_bound_for must succeed");
        assert!(
            Arc::ptr_eq(&first, &second),
            "same partition must reuse the memoized Arc"
        );
        assert!(
            matches!(first.as_ref(), BoundPredicate::AlwaysTrue),
            "strictly-between day residual must be AlwaysTrue, got {first:?}"
        );

        // A different partition must not share the AlwaysTrue memo entry.
        let other = evaluator
            .residual_bound_for(&day_partition("2021-01-01"), schema, true)
            .expect("residual for boundary day must succeed");
        assert!(
            !Arc::ptr_eq(&first, &other),
            "different partitions must not share the same memo entry"
        );
    }

    /// For a boundary day residual, `residual_bound_for` equals
    /// `bind(residual_for(..))`, and the second call is a memo hit.
    #[test]
    fn test_residual_bound_for_boundary_day_matches_residual_for_bind() {
        let schema = day_example_schema();
        let spec = day_partition_spec(schema.clone());
        let filter =
            ts_between_filter(schema.clone(), "2021-01-01T10:00:00", "2021-01-31T10:00:00");
        let evaluator =
            ResidualEvaluator::of(spec, &schema, filter, true).expect("evaluator must build");

        // Lower-bound day keeps `ts >= a` — a non-trivial residual.
        let partition = day_partition("2021-01-01");
        let expected_unbound = evaluator
            .residual_for(&partition)
            .expect("residual_for on boundary day must succeed");
        assert_ne!(
            expected_unbound,
            Predicate::AlwaysTrue,
            "boundary day must not collapse to AlwaysTrue"
        );
        let expected_bound = expected_unbound
            .bind(schema.clone(), true)
            .expect("binding residual must succeed");

        let first = evaluator
            .residual_bound_for(&partition, schema.clone(), true)
            .expect("first residual_bound_for must succeed");
        assert_eq!(
            first.as_ref(),
            &expected_bound,
            "first residual_bound_for must equal bind(residual_for)"
        );

        let second = evaluator
            .residual_bound_for(&partition, schema, true)
            .expect("second residual_bound_for must succeed");
        assert!(
            Arc::ptr_eq(&first, &second),
            "second call must hit the memo (same Arc)"
        );
        assert_eq!(
            second.as_ref(),
            &expected_bound,
            "memo hit must retain the same bound residual"
        );
    }

    /// The `case_sensitive = false` path still matches `residual_for` + bind
    /// under the same flag.
    #[test]
    fn test_residual_bound_for_case_insensitive_matches_residual_for_bind() {
        let schema = day_example_schema();
        let spec = day_partition_spec(schema.clone());
        // Bind the filter case-insensitively so it matches evaluator.case_sensitive.
        let filter = Reference::new("ts")
            .greater_than_or_equal_to(Datum::timestamp_micros(micros("2021-01-01T10:00:00")))
            .and(
                Reference::new("ts")
                    .less_than_or_equal_to(Datum::timestamp_micros(micros("2021-01-31T10:00:00"))),
            )
            .bind(schema.clone(), false)
            .expect("case-insensitive filter binds");
        let evaluator =
            ResidualEvaluator::of(spec, &schema, filter, false).expect("evaluator must build");

        let partition = day_partition("2021-01-01");
        let expected = evaluator
            .residual_for(&partition)
            .expect("residual_for")
            .bind(schema.clone(), false)
            .expect("bind residual case-insensitively");

        let bound = evaluator
            .residual_bound_for(&partition, schema, false)
            .expect("residual_bound_for case-insensitive");
        assert_eq!(
            bound.as_ref(),
            &expected,
            "case_sensitive=false residual_bound_for must match residual_for+bind"
        );
        assert!(
            !matches!(bound.as_ref(), BoundPredicate::AlwaysTrue),
            "boundary residual under case_insensitive bind must stay non-trivial"
        );
    }

    /// Past `soft_cap` a residual still computes but the memo does not grow.
    /// Removing the cap check fails this test. It also pins the production
    /// constant, so a change to the ceiling must be deliberate.
    #[test]
    fn test_residual_bound_memo_soft_cap_skips_insert_past_ceiling() {
        assert_eq!(
            RESIDUAL_BOUND_MEMO_SOFT_CAP, 8192,
            "production residual memo soft-cap must stay 8192 (C1-SEC-001); \
             change only with an intentional capacity review"
        );

        let schema = day_example_schema();
        let spec = day_partition_spec(schema.clone());
        let filter =
            ts_between_filter(schema.clone(), "2021-01-01T10:00:00", "2021-01-31T10:00:00");
        let evaluator =
            ResidualEvaluator::of(spec, &schema, filter, true).expect("evaluator must build");

        // Cap of 2: first two distinct partitions insert; third+ still compute.
        const TEST_CAP: usize = 2;
        let partitions = [
            day_partition("2021-01-01"),
            day_partition("2021-01-15"),
            day_partition("2021-01-31"),
            day_partition("2021-01-20"),
        ];

        let mut computed: Vec<Arc<BoundPredicate>> = Vec::with_capacity(partitions.len());
        for partition in &partitions {
            let bound = evaluator
                .residual_bound_for_with_cap(partition, schema.clone(), true, TEST_CAP)
                .expect("residual_bound_for_with_cap must succeed past soft-cap");
            // Semantic oracle: still matches residual_for + bind even when not inserted.
            let expected = evaluator
                .residual_for(partition)
                .expect("residual_for")
                .bind(schema.clone(), true)
                .expect("bind residual");
            assert_eq!(
                bound.as_ref(),
                &expected,
                "soft-cap skip must still return a correct residual for {partition:?}"
            );
            computed.push(bound);
        }

        assert_eq!(
            evaluator.residual_memo_len(),
            TEST_CAP,
            "memo must stop growing at the soft-cap; if this grows past {TEST_CAP}, \
             the insert gate was removed or bypassed"
        );

        // First two inserts remain memo hits (same Arc on re-query).
        let first_again = evaluator
            .residual_bound_for_with_cap(&partitions[0], schema.clone(), true, TEST_CAP)
            .expect("memo hit for first insert");
        let second_again = evaluator
            .residual_bound_for_with_cap(&partitions[1], schema.clone(), true, TEST_CAP)
            .expect("memo hit for second insert");
        assert!(
            Arc::ptr_eq(&computed[0], &first_again),
            "partition under the cap must remain a memo hit"
        );
        assert!(
            Arc::ptr_eq(&computed[1], &second_again),
            "partition under the cap must remain a memo hit"
        );

        // Past-cap partitions are recomputed each time (not Arc-identical across calls).
        let third_again = evaluator
            .residual_bound_for_with_cap(&partitions[2], schema.clone(), true, TEST_CAP)
            .expect("past-cap residual must still compute");
        assert_eq!(
            third_again.as_ref(),
            computed[2].as_ref(),
            "past-cap recompute must stay semantically equal"
        );
        assert!(
            !Arc::ptr_eq(&computed[2], &third_again),
            "past-cap partitions must not be memoized (fresh Arc each call)"
        );
        assert_eq!(
            evaluator.residual_memo_len(),
            TEST_CAP,
            "re-query of past-cap partitions must not grow the memo"
        );
    }

    /// The memo recovers from a poisoned `RwLock`. Hit and miss paths both
    /// return correct residuals after the poisoning panic.
    #[test]
    fn test_residual_bound_memo_recovers_from_poisoned_lock() {
        let schema = day_example_schema();
        let spec = day_partition_spec(schema.clone());
        let filter =
            ts_between_filter(schema.clone(), "2021-01-01T10:00:00", "2021-01-31T10:00:00");
        let evaluator =
            ResidualEvaluator::of(spec, &schema, filter, true).expect("evaluator must build");

        // Seed a memo entry, then poison — hit path must recover and serve it.
        let partition = day_partition("2021-01-15");
        let seeded = evaluator
            .residual_bound_for(&partition, schema.clone(), true)
            .expect("seed residual_bound_for must succeed");
        assert!(
            matches!(seeded.as_ref(), BoundPredicate::AlwaysTrue),
            "strictly-between day residual must be AlwaysTrue for the seed"
        );
        evaluator.poison_residual_memo_for_test();

        let hit = evaluator
            .residual_bound_for(&partition, schema.clone(), true)
            .expect("poisoned residual memo must recover on hit, not panic");
        assert!(
            Arc::ptr_eq(&seeded, &hit),
            "recovery must serve the entry cached before the panic"
        );
        assert_eq!(
            hit.as_ref(),
            seeded.as_ref(),
            "recovered hit must retain the bound residual value"
        );

        // Miss path (compute + insert) on a poisoned lock that still holds prior data.
        let cold_partition = day_partition("2021-01-01");
        let expected = evaluator
            .residual_for(&cold_partition)
            .expect("residual_for")
            .bind(schema.clone(), true)
            .expect("bind residual");
        evaluator.poison_residual_memo_for_test();
        let miss = evaluator
            .residual_bound_for(&cold_partition, schema.clone(), true)
            .expect("poisoned residual memo must recover on miss-compute, not panic");
        assert_eq!(
            miss.as_ref(),
            &expected,
            "miss after poison must equal residual_for + bind"
        );
        assert!(
            !matches!(miss.as_ref(), BoundPredicate::AlwaysTrue),
            "boundary day residual after poison recovery must stay non-trivial"
        );

        // Second call after miss-insert is a memo hit despite the earlier poison.
        let miss_again = evaluator
            .residual_bound_for(&cold_partition, schema, true)
            .expect("memo hit after poison-recovered insert must succeed");
        assert!(
            Arc::ptr_eq(&miss, &miss_again),
            "insert performed under poison recovery must be memoized"
        );
    }

    // ---- Multiple partition fields on one source column ----

    #[test]
    fn test_multiple_partition_fields_on_one_source_reduce_via_the_conclusive_field() {
        // Java loops over every partition field with a matching source id.
        // `category` sources both bucket(category, 16), never conclusive for
        // `eq`, and identity(category), which is. The loop must reach the
        // identity field. No other test has two fields on one source.
        let schema = identity_schema();
        let spec = Arc::new(
            PartitionSpec::builder(schema.clone())
                .with_spec_id(1)
                .add_unbound_field(
                    UnboundPartitionField::builder()
                        .source_id(1)
                        .name("category_bucket".to_string())
                        .field_id(1000)
                        .transform(Transform::Bucket(16))
                        .build(),
                )
                .expect("add bucket field")
                .add_unbound_field(
                    UnboundPartitionField::builder()
                        .source_id(1)
                        .name("category_id".to_string())
                        .field_id(1001)
                        .transform(Transform::Identity)
                        .build(),
                )
                .expect("add identity field")
                .build()
                .expect("spec builds"),
        );
        let filter = Reference::new("category")
            .equal_to(Datum::int(5))
            .bind(schema.clone(), true)
            .unwrap();
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).unwrap();

        // Partition struct order follows spec field order: [bucket(category), category].
        let bucket_value = {
            use crate::transform::create_transform_function;
            match create_transform_function(&Transform::Bucket(16))
                .unwrap()
                .transform_literal(&Datum::int(5))
                .unwrap()
                .expect("bucket of 5")
                .literal()
            {
                PrimitiveLiteral::Int(value) => *value,
                other => panic!("expected an Int bucket literal, got {other:?}"),
            }
        };

        // identity=5 has a true strict projection, reached past the
        // inconclusive bucket field.
        let matching = evaluator
            .residual_for(&Struct::from_iter([
                Some(Literal::int(bucket_value)),
                Some(Literal::int(5)),
            ]))
            .unwrap();
        assert_eq!(matching, Predicate::AlwaysTrue);

        // identity=7 has a false inclusive projection, again only reachable by
        // continuing the loop.
        let non_matching = evaluator
            .residual_for(&Struct::from_iter([
                Some(Literal::int(bucket_value)),
                Some(Literal::int(7)),
            ]))
            .unwrap();
        assert_eq!(non_matching, Predicate::AlwaysFalse);
    }

    // ---- Mutation guard: strict vs inclusive direction ----
    //
    // Swapping `strict_project` and `project` in `reduce_leaf` breaks the
    // day-example tests: the between-bounds case stops reducing to AlwaysTrue,
    // and the equals-one-bound cases stop dropping the satisfied half. Making
    // `residual_for` ignore the partition breaks every reduction test.
}
