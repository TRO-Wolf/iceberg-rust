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

use fnv::FnvHashSet;

use crate::expr::predicate::MAX_PREDICATE_DEPTH;
use crate::expr::{BoundPredicate, BoundReference, PredicateOperator};
use crate::spec::Datum;
use crate::{Error, ErrorKind, Result};

/// A visitor for [`BoundPredicate`]s. Visits in post-order.
pub trait BoundPredicateVisitor {
    /// The return type of this visitor
    type T;

    /// Called after an `AlwaysTrue` predicate is visited
    fn always_true(&mut self) -> Result<Self::T>;

    /// Called after an `AlwaysFalse` predicate is visited
    fn always_false(&mut self) -> Result<Self::T>;

    /// Called after an `And` predicate is visited
    fn and(&mut self, lhs: Self::T, rhs: Self::T) -> Result<Self::T>;

    /// Called after an `Or` predicate is visited
    fn or(&mut self, lhs: Self::T, rhs: Self::T) -> Result<Self::T>;

    /// Called after a `Not` predicate is visited
    fn not(&mut self, inner: Self::T) -> Result<Self::T>;

    /// Called after a predicate with an `IsNull` operator is visited
    fn is_null(
        &mut self,
        reference: &BoundReference,
        predicate: &BoundPredicate,
    ) -> Result<Self::T>;

    /// Called after a predicate with a `NotNull` operator is visited
    fn not_null(
        &mut self,
        reference: &BoundReference,
        predicate: &BoundPredicate,
    ) -> Result<Self::T>;

    /// Called after a predicate with an `IsNan` operator is visited
    fn is_nan(&mut self, reference: &BoundReference, predicate: &BoundPredicate)
    -> Result<Self::T>;

    /// Called after a predicate with a `NotNan` operator is visited
    fn not_nan(
        &mut self,
        reference: &BoundReference,
        predicate: &BoundPredicate,
    ) -> Result<Self::T>;

    /// Called after a predicate with a `LessThan` operator is visited
    fn less_than(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        predicate: &BoundPredicate,
    ) -> Result<Self::T>;

    /// Called after a predicate with a `LessThanOrEq` operator is visited
    fn less_than_or_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        predicate: &BoundPredicate,
    ) -> Result<Self::T>;

    /// Called after a predicate with a `GreaterThan` operator is visited
    fn greater_than(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        predicate: &BoundPredicate,
    ) -> Result<Self::T>;

    /// Called after a predicate with a `GreaterThanOrEq` operator is visited
    fn greater_than_or_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        predicate: &BoundPredicate,
    ) -> Result<Self::T>;

    /// Called after a predicate with an `Eq` operator is visited
    fn eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        predicate: &BoundPredicate,
    ) -> Result<Self::T>;

    /// Called after a predicate with a `NotEq` operator is visited
    fn not_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        predicate: &BoundPredicate,
    ) -> Result<Self::T>;

    /// Called after a predicate with a `StartsWith` operator is visited
    fn starts_with(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        predicate: &BoundPredicate,
    ) -> Result<Self::T>;

    /// Called after a predicate with a `NotStartsWith` operator is visited
    fn not_starts_with(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        predicate: &BoundPredicate,
    ) -> Result<Self::T>;

    /// Called after a predicate with an `In` operator is visited
    fn r#in(
        &mut self,
        reference: &BoundReference,
        literals: &FnvHashSet<Datum>,
        predicate: &BoundPredicate,
    ) -> Result<Self::T>;

    /// Called after a predicate with a `NotIn` operator is visited
    fn not_in(
        &mut self,
        reference: &BoundReference,
        literals: &FnvHashSet<Datum>,
        predicate: &BoundPredicate,
    ) -> Result<Self::T>;
}

/// Visits a [`BoundPredicate`] with the provided visitor,
/// in post-order
pub(crate) fn visit<V: BoundPredicateVisitor>(
    visitor: &mut V,
    predicate: &BoundPredicate,
) -> Result<V::T> {
    visit_at_depth(visitor, predicate, 0)
}

fn visit_at_depth<V: BoundPredicateVisitor>(
    visitor: &mut V,
    predicate: &BoundPredicate,
    depth: usize,
) -> Result<V::T> {
    if depth > MAX_PREDICATE_DEPTH {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Bound predicate nesting exceeds maximum depth {MAX_PREDICATE_DEPTH}"),
        ));
    }

    match predicate {
        BoundPredicate::And(expr) => {
            let [left_pred, right_pred] = expr.inputs();

            let left_result = visit_at_depth(visitor, left_pred, depth + 1)?;
            let right_result = visit_at_depth(visitor, right_pred, depth + 1)?;

            visitor.and(left_result, right_result)
        }
        BoundPredicate::Or(expr) => {
            let [left_pred, right_pred] = expr.inputs();

            let left_result = visit_at_depth(visitor, left_pred, depth + 1)?;
            let right_result = visit_at_depth(visitor, right_pred, depth + 1)?;

            visitor.or(left_result, right_result)
        }
        BoundPredicate::Not(expr) => {
            let [inner_pred] = expr.inputs();

            let inner_result = visit_at_depth(visitor, inner_pred, depth + 1)?;

            visitor.not(inner_result)
        }
        leaf => visit_leaf(visitor, leaf),
    }
}

/// The non-recursive arms of [`visit_at_depth`], deliberately **not inlined** — see the
/// companion `predicate_visitor::visit_leaf` for why the recursive frame is kept small.
#[inline(never)]
fn visit_leaf<V: BoundPredicateVisitor>(
    visitor: &mut V,
    predicate: &BoundPredicate,
) -> Result<V::T> {
    match predicate {
        BoundPredicate::And(_) | BoundPredicate::Or(_) | BoundPredicate::Not(_) => Err(Error::new(
            ErrorKind::Unexpected,
            "visit_leaf reached a logical predicate node",
        )),
        BoundPredicate::AlwaysTrue => visitor.always_true(),
        BoundPredicate::AlwaysFalse => visitor.always_false(),
        BoundPredicate::Unary(expr) => match expr.op() {
            PredicateOperator::IsNull => visitor.is_null(expr.term(), predicate),
            PredicateOperator::NotNull => visitor.not_null(expr.term(), predicate),
            PredicateOperator::IsNan => visitor.is_nan(expr.term(), predicate),
            PredicateOperator::NotNan => visitor.not_nan(expr.term(), predicate),
            op => Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Unexpected op for unary predicate: {op}"),
            )),
        },
        BoundPredicate::Binary(expr) => {
            let reference = expr.term();
            let literal = expr.literal();
            match expr.op() {
                PredicateOperator::LessThan => visitor.less_than(reference, literal, predicate),
                PredicateOperator::LessThanOrEq => {
                    visitor.less_than_or_eq(reference, literal, predicate)
                }
                PredicateOperator::GreaterThan => {
                    visitor.greater_than(reference, literal, predicate)
                }
                PredicateOperator::GreaterThanOrEq => {
                    visitor.greater_than_or_eq(reference, literal, predicate)
                }
                PredicateOperator::Eq => visitor.eq(reference, literal, predicate),
                PredicateOperator::NotEq => visitor.not_eq(reference, literal, predicate),
                PredicateOperator::StartsWith => visitor.starts_with(reference, literal, predicate),
                PredicateOperator::NotStartsWith => {
                    visitor.not_starts_with(reference, literal, predicate)
                }
                op => Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!("Unexpected op for binary predicate: {op}"),
                )),
            }
        }
        BoundPredicate::Set(expr) => {
            let reference = expr.term();
            let literals = expr.literals();
            match expr.op() {
                PredicateOperator::In => visitor.r#in(reference, literals, predicate),
                PredicateOperator::NotIn => visitor.not_in(reference, literals, predicate),
                op => Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!("Unexpected op for set predicate: {op}"),
                )),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Not;
    use std::sync::Arc;

    use fnv::FnvHashSet;

    use super::MAX_PREDICATE_DEPTH;
    use crate::ErrorKind;
    use crate::expr::visitors::bound_predicate_visitor::{BoundPredicateVisitor, visit};
    use crate::expr::{
        BinaryExpression, Bind, BoundPredicate, BoundReference, LogicalExpression, Predicate,
        PredicateOperator, Reference, SetExpression, UnaryExpression,
    };
    use crate::spec::{Datum, NestedField, PrimitiveType, Schema, SchemaRef, Type};

    struct TestEvaluator {}
    impl BoundPredicateVisitor for TestEvaluator {
        type T = bool;

        fn always_true(&mut self) -> crate::Result<Self::T> {
            Ok(true)
        }

        fn always_false(&mut self) -> crate::Result<Self::T> {
            Ok(false)
        }

        fn and(&mut self, lhs: Self::T, rhs: Self::T) -> crate::Result<Self::T> {
            Ok(lhs && rhs)
        }

        fn or(&mut self, lhs: Self::T, rhs: Self::T) -> crate::Result<Self::T> {
            Ok(lhs || rhs)
        }

        fn not(&mut self, inner: Self::T) -> crate::Result<Self::T> {
            Ok(!inner)
        }

        fn is_null(
            &mut self,
            _reference: &BoundReference,
            _predicate: &BoundPredicate,
        ) -> crate::Result<bool> {
            Ok(true)
        }

        fn not_null(
            &mut self,
            _reference: &BoundReference,
            _predicate: &BoundPredicate,
        ) -> crate::Result<bool> {
            Ok(false)
        }

        fn is_nan(
            &mut self,
            _reference: &BoundReference,
            _predicate: &BoundPredicate,
        ) -> crate::Result<bool> {
            Ok(true)
        }

        fn not_nan(
            &mut self,
            _reference: &BoundReference,
            _predicate: &BoundPredicate,
        ) -> crate::Result<bool> {
            Ok(false)
        }

        fn less_than(
            &mut self,
            _reference: &BoundReference,
            _literal: &Datum,
            _predicate: &BoundPredicate,
        ) -> crate::Result<bool> {
            Ok(true)
        }

        fn less_than_or_eq(
            &mut self,
            _reference: &BoundReference,
            _literal: &Datum,
            _predicate: &BoundPredicate,
        ) -> crate::Result<bool> {
            Ok(false)
        }

        fn greater_than(
            &mut self,
            _reference: &BoundReference,
            _literal: &Datum,
            _predicate: &BoundPredicate,
        ) -> crate::Result<bool> {
            Ok(true)
        }

        fn greater_than_or_eq(
            &mut self,
            _reference: &BoundReference,
            _literal: &Datum,
            _predicate: &BoundPredicate,
        ) -> crate::Result<bool> {
            Ok(false)
        }

        fn eq(
            &mut self,
            _reference: &BoundReference,
            _literal: &Datum,
            _predicate: &BoundPredicate,
        ) -> crate::Result<bool> {
            Ok(true)
        }

        fn not_eq(
            &mut self,
            _reference: &BoundReference,
            _literal: &Datum,
            _predicate: &BoundPredicate,
        ) -> crate::Result<bool> {
            Ok(false)
        }

        fn starts_with(
            &mut self,
            _reference: &BoundReference,
            _literal: &Datum,
            _predicate: &BoundPredicate,
        ) -> crate::Result<bool> {
            Ok(true)
        }

        fn not_starts_with(
            &mut self,
            _reference: &BoundReference,
            _literal: &Datum,
            _predicate: &BoundPredicate,
        ) -> crate::Result<bool> {
            Ok(false)
        }

        fn r#in(
            &mut self,
            _reference: &BoundReference,
            _literals: &FnvHashSet<Datum>,
            _predicate: &BoundPredicate,
        ) -> crate::Result<bool> {
            Ok(true)
        }

        fn not_in(
            &mut self,
            _reference: &BoundReference,
            _literals: &FnvHashSet<Datum>,
            _predicate: &BoundPredicate,
        ) -> crate::Result<bool> {
            Ok(false)
        }
    }

    fn create_test_schema() -> SchemaRef {
        let schema = Schema::builder()
            .with_fields(vec![
                Arc::new(NestedField::required(
                    1,
                    "a",
                    Type::Primitive(PrimitiveType::Int),
                )),
                Arc::new(NestedField::required(
                    2,
                    "b",
                    Type::Primitive(PrimitiveType::Float),
                )),
                Arc::new(NestedField::optional(
                    3,
                    "c",
                    Type::Primitive(PrimitiveType::Float),
                )),
            ])
            .build()
            .unwrap();

        let schema_arc = Arc::new(schema);
        schema_arc.clone()
    }

    fn nested_logical_predicate(depth: usize) -> BoundPredicate {
        let mut predicate = BoundPredicate::AlwaysTrue;
        for level in 0..depth {
            predicate = match level % 3 {
                0 => BoundPredicate::Not(LogicalExpression::new([Box::new(predicate)])),
                1 => BoundPredicate::And(LogicalExpression::new([
                    Box::new(predicate),
                    Box::new(BoundPredicate::AlwaysTrue),
                ])),
                _ => BoundPredicate::Or(LogicalExpression::new([
                    Box::new(predicate),
                    Box::new(BoundPredicate::AlwaysFalse),
                ])),
            };
        }
        predicate
    }

    /// The bound walk accepts exactly `MAX_PREDICATE_DEPTH` levels and rejects one more with a
    /// typed `DataInvalid`. Catches `depth > MAX` drifting to `depth >= MAX`.
    ///
    /// It runs on an explicitly sized thread: at the measured 4,650 bytes per level (unoptimized
    /// build; see [`MAX_PREDICATE_DEPTH`]) the walk needs ~4.6 MiB, which is more than a default
    /// thread gets, and sizing it here keeps the test a *test failure* rather than an abort.
    #[test]
    fn logical_depth_limit_is_inclusive() {
        const DEV_BYTES_PER_LEVEL: usize = 4_650;

        std::thread::Builder::new()
            .stack_size(3 * DEV_BYTES_PER_LEVEL * (MAX_PREDICATE_DEPTH + 2))
            .spawn(|| {
                let at_limit = nested_logical_predicate(MAX_PREDICATE_DEPTH);
                assert!(visit(&mut TestEvaluator {}, &at_limit).is_ok());

                let beyond_limit = nested_logical_predicate(MAX_PREDICATE_DEPTH + 1);
                let error = visit(&mut TestEvaluator {}, &beyond_limit).expect_err(
                    "a bound predicate beyond the logical nesting limit must be rejected",
                );
                assert_eq!(error.kind(), ErrorKind::DataInvalid);
                assert!(error.to_string().contains("maximum depth"));
            })
            .expect("spawning the depth-test thread must succeed")
            .join()
            .expect("the bound depth walk must not overflow its sized stack");
    }

    #[test]
    fn test_always_true() {
        let predicate = Predicate::AlwaysTrue;
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(result.unwrap());
    }

    #[test]
    fn test_always_false() {
        let predicate = Predicate::AlwaysFalse;
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(!result.unwrap());
    }

    #[test]
    fn test_logical_and() {
        let predicate = Predicate::AlwaysTrue.and(Predicate::AlwaysFalse);
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(!result.unwrap());

        let predicate = Predicate::AlwaysFalse.and(Predicate::AlwaysFalse);
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(!result.unwrap());

        let predicate = Predicate::AlwaysTrue.and(Predicate::AlwaysTrue);
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(result.unwrap());
    }

    #[test]
    fn test_logical_or() {
        let predicate = Predicate::AlwaysTrue.or(Predicate::AlwaysFalse);
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(result.unwrap());

        let predicate = Predicate::AlwaysFalse.or(Predicate::AlwaysFalse);
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(!result.unwrap());

        let predicate = Predicate::AlwaysTrue.or(Predicate::AlwaysTrue);
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(result.unwrap());
    }

    #[test]
    fn test_not() {
        let predicate = Predicate::AlwaysFalse.not();
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(result.unwrap());

        let predicate = Predicate::AlwaysTrue.not();
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(!result.unwrap());
    }

    #[test]
    fn test_is_null() {
        let predicate = Predicate::Unary(UnaryExpression::new(
            PredicateOperator::IsNull,
            Reference::new("c"),
        ));
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(result.unwrap());
    }

    #[test]
    fn test_not_null() {
        let predicate = Predicate::Unary(UnaryExpression::new(
            PredicateOperator::NotNull,
            Reference::new("a"),
        ));
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(result.unwrap());
    }

    #[test]
    fn test_is_nan() {
        let predicate = Predicate::Unary(UnaryExpression::new(
            PredicateOperator::IsNan,
            Reference::new("b"),
        ));
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(result.unwrap());
    }

    #[test]
    fn test_not_nan() {
        let predicate = Predicate::Unary(UnaryExpression::new(
            PredicateOperator::NotNan,
            Reference::new("b"),
        ));
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(!result.unwrap());
    }

    #[test]
    fn test_less_than() {
        let predicate = Predicate::Binary(BinaryExpression::new(
            PredicateOperator::LessThan,
            Reference::new("a"),
            Datum::int(10),
        ));
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(result.unwrap());
    }

    #[test]
    fn test_less_than_or_eq() {
        let predicate = Predicate::Binary(BinaryExpression::new(
            PredicateOperator::LessThanOrEq,
            Reference::new("a"),
            Datum::int(10),
        ));
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(!result.unwrap());
    }

    #[test]
    fn test_greater_than() {
        let predicate = Predicate::Binary(BinaryExpression::new(
            PredicateOperator::GreaterThan,
            Reference::new("a"),
            Datum::int(10),
        ));
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(result.unwrap());
    }

    #[test]
    fn test_greater_than_or_eq() {
        let predicate = Predicate::Binary(BinaryExpression::new(
            PredicateOperator::GreaterThanOrEq,
            Reference::new("a"),
            Datum::int(10),
        ));
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(!result.unwrap());
    }

    #[test]
    fn test_eq() {
        let predicate = Predicate::Binary(BinaryExpression::new(
            PredicateOperator::Eq,
            Reference::new("a"),
            Datum::int(10),
        ));
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(result.unwrap());
    }

    #[test]
    fn test_not_eq() {
        let predicate = Predicate::Binary(BinaryExpression::new(
            PredicateOperator::NotEq,
            Reference::new("a"),
            Datum::int(10),
        ));
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(!result.unwrap());
    }

    #[test]
    fn test_starts_with() {
        let predicate = Predicate::Binary(BinaryExpression::new(
            PredicateOperator::StartsWith,
            Reference::new("a"),
            Datum::int(10),
        ));
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(result.unwrap());
    }

    #[test]
    fn test_not_starts_with() {
        let predicate = Predicate::Binary(BinaryExpression::new(
            PredicateOperator::NotStartsWith,
            Reference::new("a"),
            Datum::int(10),
        ));
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(!result.unwrap());
    }

    #[test]
    fn test_in() {
        let predicate = Predicate::Set(SetExpression::new(
            PredicateOperator::In,
            Reference::new("a"),
            FnvHashSet::from_iter(vec![Datum::int(1)]),
        ));
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(result.unwrap());
    }

    #[test]
    fn test_not_in() {
        let predicate = Predicate::Set(SetExpression::new(
            PredicateOperator::NotIn,
            Reference::new("a"),
            FnvHashSet::from_iter(vec![Datum::int(1)]),
        ));
        let bound_predicate = predicate.bind(create_test_schema(), false).unwrap();

        let mut test_evaluator = TestEvaluator {};

        let result = visit(&mut test_evaluator, &bound_predicate);

        assert!(!result.unwrap());
    }

    // Audit SAF-004, layer 2 (defense in depth): a mismatched-arity operator
    // reaching the bound dispatcher in a release build (constructor
    // `debug_assert!` compiled out) must yield a typed `DataInvalid` error, NOT
    // `panic!`. `new_unchecked` builds the otherwise unconstructable invalid
    // value; restoring any `panic!` turns these RED.

    fn bound_ref() -> BoundReference {
        Reference::new("a")
            .bind(create_test_schema(), false)
            .expect("bind reference a")
    }

    #[test]
    fn visit_bound_unary_with_non_unary_op_errors_not_panics() {
        let predicate = BoundPredicate::Unary(UnaryExpression::new_unchecked(
            PredicateOperator::LessThan,
            bound_ref(),
        ));
        let mut test_evaluator = TestEvaluator {};
        let err = visit(&mut test_evaluator, &predicate)
            .expect_err("non-unary op in bound unary shape must error, not panic");
        assert!(
            err.to_string()
                .contains("Unexpected op for unary predicate"),
            "message: {err}"
        );
    }

    #[test]
    fn visit_bound_binary_with_non_binary_op_errors_not_panics() {
        let predicate = BoundPredicate::Binary(BinaryExpression::new_unchecked(
            PredicateOperator::IsNull,
            bound_ref(),
            Datum::int(10),
        ));
        let mut test_evaluator = TestEvaluator {};
        let err = visit(&mut test_evaluator, &predicate)
            .expect_err("non-binary op in bound binary shape must error, not panic");
        assert!(
            err.to_string()
                .contains("Unexpected op for binary predicate"),
            "message: {err}"
        );
    }

    #[test]
    fn visit_bound_set_with_non_set_op_errors_not_panics() {
        let predicate = BoundPredicate::Set(SetExpression::new_unchecked(
            PredicateOperator::LessThan,
            bound_ref(),
            FnvHashSet::from_iter(vec![Datum::int(1)]),
        ));
        let mut test_evaluator = TestEvaluator {};
        let err = visit(&mut test_evaluator, &predicate)
            .expect_err("non-set op in bound set shape must error, not panic");
        assert!(
            err.to_string().contains("Unexpected op for set predicate"),
            "message: {err}"
        );
    }
}
