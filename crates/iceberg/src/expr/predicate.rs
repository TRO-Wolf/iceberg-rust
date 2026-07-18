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

//! This module contains predicate expressions.
//! Predicate expressions are used to filter data, and evaluates to a boolean value. For example,
//! `a > 10` is a predicate expression, and it evaluates to `true` if `a` is greater than `10`,

use std::fmt::{Debug, Display, Formatter};
use std::ops::Not;

use array_init::array_init;
use fnv::FnvHashSet;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::expr::visitors::bound_predicate_visitor::visit as visit_bound;
use crate::expr::visitors::predicate_visitor::visit;
use crate::expr::visitors::rewrite_not::RewriteNotVisitor;
use crate::expr::{Bind, BoundReference, PredicateOperator, Reference};
use crate::spec::{Datum, PrimitiveLiteral, SchemaRef};
use crate::{Error, ErrorKind};

/// Logical expression, such as `AND`, `OR`, `NOT`.
#[derive(PartialEq, Clone)]
pub struct LogicalExpression<T, const N: usize> {
    inputs: [Box<T>; N],
}

impl<T: Serialize, const N: usize> Serialize for LogicalExpression<T, N> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where S: serde::Serializer {
        self.inputs.serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for LogicalExpression<T, N> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        let inputs = Vec::<Box<T>>::deserialize(deserializer)?;
        Ok(LogicalExpression::new(
            array_init::from_iter(inputs.into_iter()).ok_or_else(|| {
                serde::de::Error::custom(format!("Failed to deserialize LogicalExpression: the len of inputs is not match with the len of LogicalExpression {N}"))
            })?,
        ))
    }
}

impl<T: Debug, const N: usize> Debug for LogicalExpression<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogicalExpression")
            .field("inputs", &self.inputs)
            .finish()
    }
}

impl<T, const N: usize> LogicalExpression<T, N> {
    pub(crate) fn new(inputs: [Box<T>; N]) -> Self {
        Self { inputs }
    }

    /// Return inputs of this logical expression.
    pub fn inputs(&self) -> [&T; N] {
        let mut ret: [&T; N] = [self.inputs[0].as_ref(); N];
        for (i, item) in ret.iter_mut().enumerate() {
            *item = &self.inputs[i];
        }
        ret
    }
}

impl<T: Bind, const N: usize> Bind for LogicalExpression<T, N>
where T::Bound: Sized
{
    type Bound = LogicalExpression<T::Bound, N>;

    fn bind(&self, schema: SchemaRef, case_sensitive: bool) -> Result<Self::Bound> {
        let mut outputs: [Option<Box<T::Bound>>; N] = array_init(|_| None);
        for (i, input) in self.inputs.iter().enumerate() {
            outputs[i] = Some(Box::new(input.bind(schema.clone(), case_sensitive)?));
        }

        // It's safe to use `unwrap` here since they are all `Some`.
        let bound_inputs = array_init::from_iter(outputs.into_iter().map(Option::unwrap)).unwrap();
        Ok(LogicalExpression::new(bound_inputs))
    }
}

/// Unary predicate, for example, `a IS NULL`.
///
/// Deserialization goes through `SerdeUnaryExpression` so that the operator's
/// arity is validated at the wire boundary — [`UnaryExpression::new`] only
/// `debug_assert!`s it, so without this a crafted/corrupt serialized predicate
/// could smuggle a non-unary operator past construction (see the audit note on
/// the `SerdeUnaryExpression` shadow).
#[derive(PartialEq, Clone, Serialize, Deserialize)]
#[serde(
    try_from = "SerdeUnaryExpression<T>",
    bound(serialize = "T: Serialize", deserialize = "T: Deserialize<'de>")
)]
pub struct UnaryExpression<T> {
    /// Operator of this predicate, must be single operand operator.
    op: PredicateOperator,
    /// Term of this predicate, for example, `a` in `a IS NULL`.
    term: T,
}

/// Serde deserialization shadow for [`UnaryExpression`]: it captures the raw
/// wire fields, then the [`TryFrom`] impl rejects a non-unary operator BEFORE a
/// [`UnaryExpression`] value exists.
///
/// The public constructor [`UnaryExpression::new`] guards arity with a
/// `debug_assert!` only, which is compiled out in release. Because
/// `UnaryExpression`/[`Predicate`]/[`BoundPredicate`] derive `Deserialize` and a
/// `BoundPredicate` is reachable over the wire (it is a field of a
/// `Serialize`/`Deserialize` `FileScanTask`), a serialized predicate encoding a
/// mismatched op (e.g. a binary op in a unary shape) would otherwise bypass
/// `new` entirely and later panic the visitor dispatch in a release build
/// (audit SAF-004). Validating here closes that hole; `Serialize` stays derived
/// on the real type because we never emit an invalid shape.
#[derive(Deserialize)]
#[serde(bound(deserialize = "T: Deserialize<'de>"))]
struct SerdeUnaryExpression<T> {
    op: PredicateOperator,
    term: T,
}

impl<T> TryFrom<SerdeUnaryExpression<T>> for UnaryExpression<T> {
    type Error = String;

    fn try_from(raw: SerdeUnaryExpression<T>) -> std::result::Result<Self, Self::Error> {
        if !raw.op.is_unary() {
            return Err(format!(
                "Cannot deserialize unary predicate: {:?} is not a unary operator",
                raw.op
            ));
        }
        Ok(Self {
            op: raw.op,
            term: raw.term,
        })
    }
}

impl<T: Debug> Debug for UnaryExpression<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnaryExpression")
            .field("op", &self.op)
            .field("term", &self.term)
            .finish()
    }
}

impl<T: Display> Display for UnaryExpression<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.term, self.op)
    }
}

impl<T: Bind> Bind for UnaryExpression<T> {
    type Bound = UnaryExpression<T::Bound>;

    fn bind(&self, schema: SchemaRef, case_sensitive: bool) -> Result<Self::Bound> {
        let bound_term = self.term.bind(schema, case_sensitive)?;
        Ok(UnaryExpression::new(self.op, bound_term))
    }
}

impl<T> UnaryExpression<T> {
    /// Creates a unary expression with the given operator and term.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iceberg::expr::{PredicateOperator, Reference, UnaryExpression};
    ///
    /// UnaryExpression::new(PredicateOperator::IsNull, Reference::new("c"));
    /// ```
    pub fn new(op: PredicateOperator, term: T) -> Self {
        debug_assert!(op.is_unary());
        Self { op, term }
    }

    /// Test-only constructor that bypasses the `is_unary` guard, so a test can
    /// build an invalid-arity value and exercise the visitor's defensive error
    /// path. The serde boundary and `debug_assert!` make such a value otherwise
    /// unconstructable.
    #[cfg(test)]
    pub(crate) fn new_unchecked(op: PredicateOperator, term: T) -> Self {
        Self { op, term }
    }

    /// Return the operator of this predicate.
    pub fn op(&self) -> PredicateOperator {
        self.op
    }

    /// Return the term of this predicate.
    pub fn term(&self) -> &T {
        &self.term
    }
}

/// Binary predicate, for example, `a > 10`.
///
/// Deserialization goes through `SerdeBinaryExpression` so that a non-binary
/// operator is rejected at the wire boundary; see the `SerdeUnaryExpression`
/// shadow for why the derived `Deserialize` alone is unsafe (audit SAF-004).
#[derive(PartialEq, Clone, Serialize, Deserialize)]
#[serde(
    try_from = "SerdeBinaryExpression<T>",
    bound(serialize = "T: Serialize", deserialize = "T: Deserialize<'de>")
)]
pub struct BinaryExpression<T> {
    /// Operator of this predicate, must be binary operator, such as `=`, `>`, `<`, etc.
    op: PredicateOperator,
    /// Term of this predicate, for example, `a` in `a > 10`.
    term: T,
    /// Literal of this predicate, for example, `10` in `a > 10`.
    literal: Datum,
}

/// Serde deserialization shadow for [`BinaryExpression`]; the [`TryFrom`] impl
/// rejects a non-binary operator before the value exists. See
/// [`SerdeUnaryExpression`] for the rationale.
#[derive(Deserialize)]
#[serde(bound(deserialize = "T: Deserialize<'de>"))]
struct SerdeBinaryExpression<T> {
    op: PredicateOperator,
    term: T,
    literal: Datum,
}

impl<T> TryFrom<SerdeBinaryExpression<T>> for BinaryExpression<T> {
    type Error = String;

    fn try_from(raw: SerdeBinaryExpression<T>) -> std::result::Result<Self, Self::Error> {
        if !raw.op.is_binary() {
            return Err(format!(
                "Cannot deserialize binary predicate: {:?} is not a binary operator",
                raw.op
            ));
        }
        Ok(Self {
            op: raw.op,
            term: raw.term,
            literal: raw.literal,
        })
    }
}

impl<T: Debug> Debug for BinaryExpression<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinaryExpression")
            .field("op", &self.op)
            .field("term", &self.term)
            .field("literal", &self.literal)
            .finish()
    }
}

impl<T> BinaryExpression<T> {
    /// Creates a binary expression with the given operator, term and literal.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iceberg::expr::{BinaryExpression, PredicateOperator, Reference};
    /// use iceberg::spec::Datum;
    ///
    /// BinaryExpression::new(
    ///     PredicateOperator::LessThanOrEq,
    ///     Reference::new("a"),
    ///     Datum::int(10),
    /// );
    /// ```
    pub fn new(op: PredicateOperator, term: T, literal: Datum) -> Self {
        debug_assert!(op.is_binary());
        Self { op, term, literal }
    }

    /// Test-only constructor that bypasses the `is_binary` guard; see
    /// [`UnaryExpression::new_unchecked`].
    #[cfg(test)]
    pub(crate) fn new_unchecked(op: PredicateOperator, term: T, literal: Datum) -> Self {
        Self { op, term, literal }
    }

    /// Return the operator used by this predicate expression.
    pub fn op(&self) -> PredicateOperator {
        self.op
    }

    /// Return the literal of this predicate.
    pub fn literal(&self) -> &Datum {
        &self.literal
    }

    /// Return the term of this predicate.
    pub fn term(&self) -> &T {
        &self.term
    }
}

impl<T: Display> Display for BinaryExpression<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {}", self.term, self.op, self.literal)
    }
}

impl<T: Bind> Bind for BinaryExpression<T> {
    type Bound = BinaryExpression<T::Bound>;

    fn bind(&self, schema: SchemaRef, case_sensitive: bool) -> Result<Self::Bound> {
        let bound_term = self.term.bind(schema.clone(), case_sensitive)?;
        Ok(BinaryExpression::new(
            self.op,
            bound_term,
            self.literal.clone(),
        ))
    }
}

/// Set predicates, for example, `a in (1, 2, 3)`.
///
/// Deserialization goes through `SerdeSetExpression` so that a non-set operator
/// is rejected at the wire boundary; see the `SerdeUnaryExpression` shadow for
/// why the derived `Deserialize` alone is unsafe (audit SAF-004).
#[derive(PartialEq, Clone, Serialize, Deserialize)]
#[serde(
    try_from = "SerdeSetExpression<T>",
    bound(serialize = "T: Serialize", deserialize = "T: Deserialize<'de>")
)]
pub struct SetExpression<T> {
    /// Operator of this predicate, must be set operator, such as `IN`, `NOT IN`, etc.
    op: PredicateOperator,
    /// Term of this predicate, for example, `a` in `a in (1, 2, 3)`.
    term: T,
    /// Literals of this predicate, for example, `(1, 2, 3)` in `a in (1, 2, 3)`.
    literals: FnvHashSet<Datum>,
}

/// Serde deserialization shadow for [`SetExpression`]; the [`TryFrom`] impl
/// rejects a non-set operator before the value exists. See
/// [`SerdeUnaryExpression`] for the rationale.
#[derive(Deserialize)]
#[serde(bound(deserialize = "T: Deserialize<'de>"))]
struct SerdeSetExpression<T> {
    op: PredicateOperator,
    term: T,
    literals: FnvHashSet<Datum>,
}

impl<T> TryFrom<SerdeSetExpression<T>> for SetExpression<T> {
    type Error = String;

    fn try_from(raw: SerdeSetExpression<T>) -> std::result::Result<Self, Self::Error> {
        if !raw.op.is_set() {
            return Err(format!(
                "Cannot deserialize set predicate: {:?} is not a set operator",
                raw.op
            ));
        }
        Ok(Self {
            op: raw.op,
            term: raw.term,
            literals: raw.literals,
        })
    }
}

impl<T: Debug> Debug for SetExpression<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SetExpression")
            .field("op", &self.op)
            .field("term", &self.term)
            .field("literal", &self.literals)
            .finish()
    }
}

impl<T> SetExpression<T> {
    /// Creates a set expression with the given operator, term and literal.
    ///
    /// # Example
    ///
    /// ```rust
    /// use fnv::FnvHashSet;
    /// use iceberg::expr::{PredicateOperator, Reference, SetExpression};
    /// use iceberg::spec::Datum;
    ///
    /// SetExpression::new(
    ///     PredicateOperator::In,
    ///     Reference::new("a"),
    ///     FnvHashSet::from_iter(vec![Datum::int(1)]),
    /// );
    /// ```
    pub fn new(op: PredicateOperator, term: T, literals: FnvHashSet<Datum>) -> Self {
        debug_assert!(op.is_set());
        Self { op, term, literals }
    }

    /// Test-only constructor that bypasses the `is_set` guard; see
    /// [`UnaryExpression::new_unchecked`].
    #[cfg(test)]
    pub(crate) fn new_unchecked(
        op: PredicateOperator,
        term: T,
        literals: FnvHashSet<Datum>,
    ) -> Self {
        Self { op, term, literals }
    }

    /// Return the operator of this predicate.
    pub fn op(&self) -> PredicateOperator {
        self.op
    }

    /// Return the hash set of values compared against the term in this expression.
    pub fn literals(&self) -> &FnvHashSet<Datum> {
        &self.literals
    }

    /// Return the term of this predicate.
    pub fn term(&self) -> &T {
        &self.term
    }
}

impl<T: Bind> Bind for SetExpression<T> {
    type Bound = SetExpression<T::Bound>;

    fn bind(&self, schema: SchemaRef, case_sensitive: bool) -> Result<Self::Bound> {
        let bound_term = self.term.bind(schema.clone(), case_sensitive)?;
        Ok(SetExpression::new(
            self.op,
            bound_term,
            self.literals.clone(),
        ))
    }
}

impl<T: Display + Debug> Display for SetExpression<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut literal_strs = self.literals.iter().map(|l| format!("{l}"));

        write!(f, "{} {} ({})", self.term, self.op, literal_strs.join(", "))
    }
}

/// Unbound predicate expression before binding to a schema.
#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub enum Predicate {
    /// AlwaysTrue predicate, for example, `TRUE`.
    AlwaysTrue,
    /// AlwaysFalse predicate, for example, `FALSE`.
    AlwaysFalse,
    /// And predicate, for example, `a > 10 AND b < 20`.
    And(LogicalExpression<Predicate, 2>),
    /// Or predicate, for example, `a > 10 OR b < 20`.
    Or(LogicalExpression<Predicate, 2>),
    /// Not predicate, for example, `NOT (a > 10)`.
    Not(LogicalExpression<Predicate, 1>),
    /// Unary expression, for example, `a IS NULL`.
    Unary(UnaryExpression<Reference>),
    /// Binary expression, for example, `a > 10`.
    Binary(BinaryExpression<Reference>),
    /// Set predicates, for example, `a in (1, 2, 3)`.
    Set(SetExpression<Reference>),
}

impl Bind for Predicate {
    type Bound = BoundPredicate;

    fn bind(&self, schema: SchemaRef, case_sensitive: bool) -> Result<BoundPredicate> {
        match self {
            Predicate::And(expr) => {
                let bound_expr = expr.bind(schema, case_sensitive)?;

                let [left, right] = bound_expr.inputs;
                Ok(match (left, right) {
                    (_, r) if matches!(&*r, &BoundPredicate::AlwaysFalse) => {
                        BoundPredicate::AlwaysFalse
                    }
                    (l, _) if matches!(&*l, &BoundPredicate::AlwaysFalse) => {
                        BoundPredicate::AlwaysFalse
                    }
                    (left, r) if matches!(&*r, &BoundPredicate::AlwaysTrue) => *left,
                    (l, right) if matches!(&*l, &BoundPredicate::AlwaysTrue) => *right,
                    (left, right) => BoundPredicate::And(LogicalExpression::new([left, right])),
                })
            }
            Predicate::Not(expr) => {
                let bound_expr = expr.bind(schema, case_sensitive)?;
                let [inner] = bound_expr.inputs;
                Ok(match inner {
                    e if matches!(&*e, &BoundPredicate::AlwaysTrue) => BoundPredicate::AlwaysFalse,
                    e if matches!(&*e, &BoundPredicate::AlwaysFalse) => BoundPredicate::AlwaysTrue,
                    e => BoundPredicate::Not(LogicalExpression::new([e])),
                })
            }
            Predicate::Or(expr) => {
                let bound_expr = expr.bind(schema, case_sensitive)?;
                let [left, right] = bound_expr.inputs;
                Ok(match (left, right) {
                    (l, r)
                        if matches!(&*r, &BoundPredicate::AlwaysTrue)
                            || matches!(&*l, &BoundPredicate::AlwaysTrue) =>
                    {
                        BoundPredicate::AlwaysTrue
                    }
                    (left, r) if matches!(&*r, &BoundPredicate::AlwaysFalse) => *left,
                    (l, right) if matches!(&*l, &BoundPredicate::AlwaysFalse) => *right,
                    (left, right) => BoundPredicate::Or(LogicalExpression::new([left, right])),
                })
            }
            Predicate::Unary(expr) => {
                let bound_expr = expr.bind(schema, case_sensitive)?;

                match &bound_expr.op {
                    &PredicateOperator::IsNull => {
                        if bound_expr.term.field().required {
                            return Ok(BoundPredicate::AlwaysFalse);
                        }
                    }
                    &PredicateOperator::NotNull => {
                        if bound_expr.term.field().required {
                            return Ok(BoundPredicate::AlwaysTrue);
                        }
                    }
                    &PredicateOperator::IsNan | &PredicateOperator::NotNan => {
                        if !bound_expr.term.field().field_type.is_floating_type() {
                            return Err(Error::new(
                                ErrorKind::DataInvalid,
                                format!(
                                    "Expecting floating point type, but found {}",
                                    bound_expr.term.field().field_type
                                ),
                            ));
                        }
                    }
                    op => {
                        return Err(Error::new(
                            ErrorKind::Unexpected,
                            format!("Expecting unary operator, but found {op}"),
                        ));
                    }
                }

                Ok(BoundPredicate::Unary(bound_expr))
            }
            Predicate::Binary(expr) => {
                let bound_expr = expr.bind(schema, case_sensitive)?;
                let bound_literal = bound_expr.literal.to(&bound_expr.term.field().field_type)?;

                match bound_literal.literal() {
                    PrimitiveLiteral::AboveMax => match &bound_expr.op {
                        &PredicateOperator::LessThan
                        | &PredicateOperator::LessThanOrEq
                        | &PredicateOperator::NotEq => {
                            return Ok(BoundPredicate::AlwaysTrue);
                        }
                        &PredicateOperator::GreaterThan
                        | &PredicateOperator::GreaterThanOrEq
                        | &PredicateOperator::Eq => {
                            return Ok(BoundPredicate::AlwaysFalse);
                        }
                        _ => {}
                    },
                    PrimitiveLiteral::BelowMin => match &bound_expr.op {
                        &PredicateOperator::GreaterThan
                        | &PredicateOperator::GreaterThanOrEq
                        | &PredicateOperator::NotEq => {
                            return Ok(BoundPredicate::AlwaysTrue);
                        }
                        &PredicateOperator::LessThan
                        | &PredicateOperator::LessThanOrEq
                        | &PredicateOperator::Eq => {
                            return Ok(BoundPredicate::AlwaysFalse);
                        }
                        _ => {}
                    },
                    _ => {}
                }

                Ok(BoundPredicate::Binary(BinaryExpression::new(
                    bound_expr.op,
                    bound_expr.term,
                    bound_literal,
                )))
            }
            Predicate::Set(expr) => {
                let bound_expr = expr.bind(schema, case_sensitive)?;
                let bound_literals = bound_expr
                    .literals
                    .into_iter()
                    .map(|l| l.to(&bound_expr.term.field().field_type))
                    .collect::<Result<FnvHashSet<Datum>>>()?;

                match &bound_expr.op {
                    &PredicateOperator::In => {
                        if bound_literals.is_empty() {
                            return Ok(BoundPredicate::AlwaysFalse);
                        }
                        if bound_literals.len() == 1 {
                            return Ok(BoundPredicate::Binary(BinaryExpression::new(
                                PredicateOperator::Eq,
                                bound_expr.term,
                                bound_literals.into_iter().next().unwrap(),
                            )));
                        }
                    }
                    &PredicateOperator::NotIn => {
                        if bound_literals.is_empty() {
                            return Ok(BoundPredicate::AlwaysTrue);
                        }
                        if bound_literals.len() == 1 {
                            return Ok(BoundPredicate::Binary(BinaryExpression::new(
                                PredicateOperator::NotEq,
                                bound_expr.term,
                                bound_literals.into_iter().next().unwrap(),
                            )));
                        }
                    }
                    op => {
                        return Err(Error::new(
                            ErrorKind::Unexpected,
                            format!("Expecting unary operator,but found {op}"),
                        ));
                    }
                }

                Ok(BoundPredicate::Set(SetExpression::new(
                    bound_expr.op,
                    bound_expr.term,
                    bound_literals,
                )))
            }
            Predicate::AlwaysTrue => Ok(BoundPredicate::AlwaysTrue),
            Predicate::AlwaysFalse => Ok(BoundPredicate::AlwaysFalse),
        }
    }
}

impl Display for Predicate {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Predicate::AlwaysTrue => {
                write!(f, "TRUE")
            }
            Predicate::AlwaysFalse => {
                write!(f, "FALSE")
            }
            Predicate::And(expr) => {
                write!(f, "({}) AND ({})", expr.inputs()[0], expr.inputs()[1])
            }
            Predicate::Or(expr) => {
                write!(f, "({}) OR ({})", expr.inputs()[0], expr.inputs()[1])
            }
            Predicate::Not(expr) => {
                write!(f, "NOT ({})", expr.inputs()[0])
            }
            Predicate::Unary(expr) => {
                write!(f, "{expr}")
            }
            Predicate::Binary(expr) => {
                write!(f, "{expr}")
            }
            Predicate::Set(expr) => {
                write!(f, "{expr}")
            }
        }
    }
}

impl Predicate {
    /// Combines two predicates with `AND`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::ops::Bound::Unbounded;
    ///
    /// use iceberg::expr::BoundPredicate::Unary;
    /// use iceberg::expr::Reference;
    /// use iceberg::spec::Datum;
    /// let expr1 = Reference::new("a").less_than(Datum::long(10));
    ///
    /// let expr2 = Reference::new("b").less_than(Datum::long(20));
    ///
    /// let expr = expr1.and(expr2);
    ///
    /// assert_eq!(&format!("{expr}"), "(a < 10) AND (b < 20)");
    /// ```
    pub fn and(self, other: Predicate) -> Predicate {
        match (self, other) {
            (Predicate::AlwaysFalse, _) => Predicate::AlwaysFalse,
            (_, Predicate::AlwaysFalse) => Predicate::AlwaysFalse,
            (Predicate::AlwaysTrue, rhs) => rhs,
            (lhs, Predicate::AlwaysTrue) => lhs,
            (lhs, rhs) => Predicate::And(LogicalExpression::new([Box::new(lhs), Box::new(rhs)])),
        }
    }

    /// Combines two predicates with `OR`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::ops::Bound::Unbounded;
    ///
    /// use iceberg::expr::BoundPredicate::Unary;
    /// use iceberg::expr::Reference;
    /// use iceberg::spec::Datum;
    /// let expr1 = Reference::new("a").less_than(Datum::long(10));
    ///
    /// let expr2 = Reference::new("b").less_than(Datum::long(20));
    ///
    /// let expr = expr1.or(expr2);
    ///
    /// assert_eq!(&format!("{expr}"), "(a < 10) OR (b < 20)");
    /// ```
    pub fn or(self, other: Predicate) -> Predicate {
        match (self, other) {
            (Predicate::AlwaysTrue, _) => Predicate::AlwaysTrue,
            (_, Predicate::AlwaysTrue) => Predicate::AlwaysTrue,
            (Predicate::AlwaysFalse, rhs) => rhs,
            (lhs, Predicate::AlwaysFalse) => lhs,
            (lhs, rhs) => Predicate::Or(LogicalExpression::new([Box::new(lhs), Box::new(rhs)])),
        }
    }

    /// Returns a predicate representing the negation ('NOT') of this one,
    /// by using inverse predicates rather than wrapping in a `NOT`.
    /// Used for `NOT` elimination.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::ops::Bound::Unbounded;
    ///
    /// use iceberg::expr::BoundPredicate::Unary;
    /// use iceberg::expr::{LogicalExpression, Predicate, Reference};
    /// use iceberg::spec::Datum;
    /// let expr1 = Reference::new("a").less_than(Datum::long(10));
    /// let expr2 = Reference::new("b")
    ///     .less_than(Datum::long(5))
    ///     .and(Reference::new("c").less_than(Datum::long(10)));
    ///
    /// let result = expr1.negate();
    /// assert_eq!(&format!("{result}"), "a >= 10");
    ///
    /// let result = expr2.negate();
    /// assert_eq!(&format!("{result}"), "(b >= 5) OR (c >= 10)");
    /// ```
    pub fn negate(self) -> Predicate {
        match self {
            Predicate::AlwaysTrue => Predicate::AlwaysFalse,
            Predicate::AlwaysFalse => Predicate::AlwaysTrue,
            Predicate::And(expr) => Predicate::Or(LogicalExpression::new(
                expr.inputs.map(|expr| Box::new(expr.negate())),
            )),
            Predicate::Or(expr) => Predicate::And(LogicalExpression::new(
                expr.inputs.map(|expr| Box::new(expr.negate())),
            )),
            Predicate::Not(expr) => {
                let LogicalExpression { inputs: [input_0] } = expr;
                *input_0
            }
            Predicate::Unary(expr) => {
                Predicate::Unary(UnaryExpression::new(expr.op.negate(), expr.term))
            }
            Predicate::Binary(expr) => Predicate::Binary(BinaryExpression::new(
                expr.op.negate(),
                expr.term,
                expr.literal,
            )),
            Predicate::Set(expr) => Predicate::Set(SetExpression::new(
                expr.op.negate(),
                expr.term,
                expr.literals,
            )),
        }
    }
    /// Simplifies the expression by removing `NOT` predicates,
    /// directly negating the inner expressions instead. The transformation
    /// applies logical laws (such as De Morgan's laws) to
    /// recursively negate and simplify inner expressions within `NOT`
    /// predicates.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::ops::Not;
    ///
    /// use iceberg::expr::{LogicalExpression, Predicate, Reference};
    /// use iceberg::spec::Datum;
    ///
    /// let expression = Reference::new("a").less_than(Datum::long(5)).not();
    /// let result = expression.rewrite_not();
    ///
    /// assert_eq!(&format!("{result}"), "a >= 5");
    /// ```
    pub fn rewrite_not(self) -> Predicate {
        visit(&mut RewriteNotVisitor::new(), &self)
            .expect("RewriteNotVisitor guarantees always success")
    }
}

impl Not for Predicate {
    type Output = Predicate;

    /// Create a predicate which is the reverse of this predicate. For example: `NOT (a > 10)`.
    ///
    /// This is different from [`Predicate::negate()`] since it doesn't rewrite expression, but
    /// just adds a `NOT` operator.
    ///
    /// # Example
    ///     
    ///```rust
    /// use std::ops::Bound::Unbounded;
    ///
    /// use iceberg::expr::BoundPredicate::Unary;
    /// use iceberg::expr::Reference;
    /// use iceberg::spec::Datum;
    /// let expr1 = Reference::new("a").less_than(Datum::long(10));
    ///
    /// let expr = !expr1;
    ///
    /// assert_eq!(&format!("{expr}"), "NOT (a < 10)");
    /// ```
    fn not(self) -> Self::Output {
        Predicate::Not(LogicalExpression::new([Box::new(self)]))
    }
}

/// Bound predicate expression after binding to a schema.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BoundPredicate {
    /// An expression always evaluates to true.
    AlwaysTrue,
    /// An expression always evaluates to false.
    AlwaysFalse,
    /// An expression combined by `AND`, for example, `a > 10 AND b < 20`.
    And(LogicalExpression<BoundPredicate, 2>),
    /// An expression combined by `OR`, for example, `a > 10 OR b < 20`.
    Or(LogicalExpression<BoundPredicate, 2>),
    /// An expression combined by `NOT`, for example, `NOT (a > 10)`.
    Not(LogicalExpression<BoundPredicate, 1>),
    /// Unary expression, for example, `a IS NULL`.
    Unary(UnaryExpression<BoundReference>),
    /// Binary expression, for example, `a > 10`.
    Binary(BinaryExpression<BoundReference>),
    /// Set predicates, for example, `a IN (1, 2, 3)`.
    Set(SetExpression<BoundReference>),
}

impl BoundPredicate {
    pub(crate) fn and(self, other: BoundPredicate) -> BoundPredicate {
        BoundPredicate::And(LogicalExpression::new([Box::new(self), Box::new(other)]))
    }

    pub(crate) fn or(self, other: BoundPredicate) -> BoundPredicate {
        BoundPredicate::Or(LogicalExpression::new([Box::new(self), Box::new(other)]))
    }

    pub(crate) fn negate(self) -> BoundPredicate {
        match self {
            BoundPredicate::AlwaysTrue => BoundPredicate::AlwaysFalse,
            BoundPredicate::AlwaysFalse => BoundPredicate::AlwaysTrue,
            BoundPredicate::And(expr) => BoundPredicate::Or(LogicalExpression::new(
                expr.inputs.map(|expr| Box::new(expr.negate())),
            )),
            BoundPredicate::Or(expr) => BoundPredicate::And(LogicalExpression::new(
                expr.inputs.map(|expr| Box::new(expr.negate())),
            )),
            BoundPredicate::Not(expr) => {
                let LogicalExpression { inputs: [input_0] } = expr;
                *input_0
            }
            BoundPredicate::Unary(expr) => {
                BoundPredicate::Unary(UnaryExpression::new(expr.op.negate(), expr.term))
            }
            BoundPredicate::Binary(expr) => BoundPredicate::Binary(BinaryExpression::new(
                expr.op.negate(),
                expr.term,
                expr.literal,
            )),
            BoundPredicate::Set(expr) => BoundPredicate::Set(SetExpression::new(
                expr.op.negate(),
                expr.term,
                expr.literals,
            )),
        }
    }

    /// Simplifies the expression by removing `NOT` predicates,
    /// directly negating the inner expressions instead. The transformation
    /// applies logical laws (such as De Morgan's laws) to
    /// recursively negate and simplify inner expressions within `NOT`
    /// predicates.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::ops::Not;
    ///
    /// use iceberg::expr::{Bind, BoundPredicate, Reference};
    /// use iceberg::spec::Datum;
    ///
    /// // This would need to be bound first, but the concept is:
    /// // let expression = bound_predicate.not();
    /// // let result = expression.rewrite_not();
    /// ```
    pub fn rewrite_not(self) -> BoundPredicate {
        visit_bound(&mut RewriteNotVisitor::new(), &self)
            .expect("RewriteNotVisitor guarantees always success")
    }
}

impl Display for BoundPredicate {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BoundPredicate::AlwaysTrue => {
                write!(f, "True")
            }
            BoundPredicate::AlwaysFalse => {
                write!(f, "False")
            }
            BoundPredicate::And(expr) => {
                write!(f, "({}) AND ({})", expr.inputs()[0], expr.inputs()[1])
            }
            BoundPredicate::Or(expr) => {
                write!(f, "({}) OR ({})", expr.inputs()[0], expr.inputs()[1])
            }
            BoundPredicate::Not(expr) => {
                write!(f, "NOT ({})", expr.inputs()[0])
            }
            BoundPredicate::Unary(expr) => {
                write!(f, "{expr}")
            }
            BoundPredicate::Binary(expr) => {
                write!(f, "{expr}")
            }
            BoundPredicate::Set(expr) => {
                write!(f, "{expr}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Not;
    use std::sync::Arc;

    use crate::expr::Predicate::{AlwaysFalse, AlwaysTrue};
    use crate::expr::{Bind, BoundPredicate, Reference};
    use crate::spec::{Datum, NestedField, PrimitiveType, Schema, SchemaRef, Type};

    #[test]
    fn test_logical_or_rewrite_not() {
        let expression = Reference::new("b")
            .less_than(Datum::long(5))
            .or(Reference::new("c").less_than(Datum::long(10)))
            .not();

        let expected = Reference::new("b")
            .greater_than_or_equal_to(Datum::long(5))
            .and(Reference::new("c").greater_than_or_equal_to(Datum::long(10)));

        let result = expression.rewrite_not();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_logical_and_rewrite_not() {
        let expression = Reference::new("b")
            .less_than(Datum::long(5))
            .and(Reference::new("c").less_than(Datum::long(10)))
            .not();

        let expected = Reference::new("b")
            .greater_than_or_equal_to(Datum::long(5))
            .or(Reference::new("c").greater_than_or_equal_to(Datum::long(10)));

        let result = expression.rewrite_not();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_set_rewrite_not() {
        let expression = Reference::new("a")
            .is_in([Datum::int(5), Datum::int(6)])
            .not();

        let expected = Reference::new("a").is_not_in([Datum::int(5), Datum::int(6)]);

        let result = expression.rewrite_not();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_binary_rewrite_not() {
        let expression = Reference::new("a").less_than(Datum::long(5)).not();

        let expected = Reference::new("a").greater_than_or_equal_to(Datum::long(5));

        let result = expression.rewrite_not();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_unary_rewrite_not() {
        let expression = Reference::new("a").is_null().not();

        let expected = Reference::new("a").is_not_null();

        let result = expression.rewrite_not();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_predicate_and_reduce_always_true_false() {
        let true_or_expr = AlwaysTrue.and(Reference::new("b").less_than(Datum::long(5)));
        assert_eq!(&format!("{true_or_expr}"), "b < 5");

        let expr_or_true = Reference::new("b")
            .less_than(Datum::long(5))
            .and(AlwaysTrue);
        assert_eq!(&format!("{expr_or_true}"), "b < 5");

        let false_or_expr = AlwaysFalse.and(Reference::new("b").less_than(Datum::long(5)));
        assert_eq!(&format!("{false_or_expr}"), "FALSE");

        let expr_or_false = Reference::new("b")
            .less_than(Datum::long(5))
            .and(AlwaysFalse);
        assert_eq!(&format!("{expr_or_false}"), "FALSE");
    }

    #[test]
    fn test_predicate_or_reduce_always_true_false() {
        let true_or_expr = AlwaysTrue.or(Reference::new("b").less_than(Datum::long(5)));
        assert_eq!(&format!("{true_or_expr}"), "TRUE");

        let expr_or_true = Reference::new("b").less_than(Datum::long(5)).or(AlwaysTrue);
        assert_eq!(&format!("{expr_or_true}"), "TRUE");

        let false_or_expr = AlwaysFalse.or(Reference::new("b").less_than(Datum::long(5)));
        assert_eq!(&format!("{false_or_expr}"), "b < 5");

        let expr_or_false = Reference::new("b")
            .less_than(Datum::long(5))
            .or(AlwaysFalse);
        assert_eq!(&format!("{expr_or_false}"), "b < 5");
    }

    #[test]
    fn test_predicate_negate_and() {
        let expression = Reference::new("b")
            .less_than(Datum::long(5))
            .and(Reference::new("c").less_than(Datum::long(10)));

        let expected = Reference::new("b")
            .greater_than_or_equal_to(Datum::long(5))
            .or(Reference::new("c").greater_than_or_equal_to(Datum::long(10)));

        let result = expression.negate();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_predicate_negate_or() {
        let expression = Reference::new("b")
            .greater_than_or_equal_to(Datum::long(5))
            .or(Reference::new("c").greater_than_or_equal_to(Datum::long(10)));

        let expected = Reference::new("b")
            .less_than(Datum::long(5))
            .and(Reference::new("c").less_than(Datum::long(10)));

        let result = expression.negate();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_predicate_negate_not() {
        let expression = Reference::new("b")
            .greater_than_or_equal_to(Datum::long(5))
            .not();

        let expected = Reference::new("b").greater_than_or_equal_to(Datum::long(5));

        let result = expression.negate();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_predicate_negate_unary() {
        let expression = Reference::new("b").is_not_null();

        let expected = Reference::new("b").is_null();

        let result = expression.negate();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_predicate_negate_binary() {
        let expression = Reference::new("a").less_than(Datum::long(5));

        let expected = Reference::new("a").greater_than_or_equal_to(Datum::long(5));

        let result = expression.negate();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_predicate_negate_set() {
        let expression = Reference::new("a").is_in([Datum::long(5), Datum::long(6)]);

        let expected = Reference::new("a").is_not_in([Datum::long(5), Datum::long(6)]);

        let result = expression.negate();

        assert_eq!(result, expected);
    }

    pub fn table_schema_simple() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_identifier_field_ids(vec![2])
                .with_fields(vec![
                    NestedField::optional(1, "foo", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(2, "bar", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(3, "baz", Type::Primitive(PrimitiveType::Boolean)).into(),
                    NestedField::optional(4, "qux", Type::Primitive(PrimitiveType::Float)).into(),
                ])
                .build()
                .unwrap(),
        )
    }

    fn test_bound_predicate_serialize_diserialize(bound_predicate: BoundPredicate) {
        let serialized = serde_json::to_string(&bound_predicate).unwrap();
        let deserialized: BoundPredicate = serde_json::from_str(&serialized).unwrap();
        assert_eq!(bound_predicate, deserialized);
    }

    #[test]
    fn test_bind_is_null() {
        let schema = table_schema_simple();
        let expr = Reference::new("foo").is_null();
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "foo IS NULL");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_is_null_required() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").is_null();
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "False");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_is_not_null() {
        let schema = table_schema_simple();
        let expr = Reference::new("foo").is_not_null();
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "foo IS NOT NULL");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_is_not_null_required() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").is_not_null();
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "True");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_is_nan() {
        let schema = table_schema_simple();
        let expr = Reference::new("qux").is_nan();
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "qux IS NAN");

        let schema_string = table_schema_simple();
        let expr_string = Reference::new("foo").is_nan();
        let bound_expr_string = expr_string.bind(schema_string, true);
        assert!(bound_expr_string.is_err());
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_is_nan_wrong_type() {
        let schema = table_schema_simple();
        let expr = Reference::new("foo").is_nan();
        let bound_expr = expr.bind(schema, true);
        assert!(bound_expr.is_err());
    }

    #[test]
    fn test_bind_is_not_nan() {
        let schema = table_schema_simple();
        let expr = Reference::new("qux").is_not_nan();
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "qux IS NOT NAN");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_is_not_nan_wrong_type() {
        let schema = table_schema_simple();
        let expr = Reference::new("foo").is_not_nan();
        let bound_expr = expr.bind(schema, true);
        assert!(bound_expr.is_err());
    }

    #[test]
    fn test_bind_less_than() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").less_than(Datum::int(10));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "bar < 10");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_less_than_wrong_type() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").less_than(Datum::string("abcd"));
        let bound_expr = expr.bind(schema, true);
        assert!(bound_expr.is_err());
    }

    #[test]
    fn test_bind_less_than_or_eq() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").less_than_or_equal_to(Datum::int(10));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "bar <= 10");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_less_than_or_eq_wrong_type() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").less_than_or_equal_to(Datum::string("abcd"));
        let bound_expr = expr.bind(schema, true);
        assert!(bound_expr.is_err());
    }

    #[test]
    fn test_bind_greater_than() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").greater_than(Datum::int(10));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "bar > 10");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_greater_than_wrong_type() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").greater_than(Datum::string("abcd"));
        let bound_expr = expr.bind(schema, true);
        assert!(bound_expr.is_err());
    }

    #[test]
    fn test_bind_greater_than_or_eq() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").greater_than_or_equal_to(Datum::int(10));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "bar >= 10");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_greater_than_or_eq_wrong_type() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").greater_than_or_equal_to(Datum::string("abcd"));
        let bound_expr = expr.bind(schema, true);
        assert!(bound_expr.is_err());
    }

    #[test]
    fn test_bind_equal_to() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").equal_to(Datum::int(10));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "bar = 10");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_equal_to_above_max() {
        let schema = table_schema_simple();
        // int32 can hold up to 2147483647
        let expr = Reference::new("bar").equal_to(Datum::long(2147483648i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "False");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_equal_to_below_min() {
        let schema = table_schema_simple();
        // int32 can hold up to -2147483647
        let expr = Reference::new("bar").equal_to(Datum::long(-2147483649i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "False");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_not_equal_to_above_max() {
        let schema = table_schema_simple();
        // int32 can hold up to 2147483647
        let expr = Reference::new("bar").not_equal_to(Datum::long(2147483648i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "True");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_not_equal_to_below_min() {
        let schema = table_schema_simple();
        // int32 can hold up to -2147483647
        let expr = Reference::new("bar").not_equal_to(Datum::long(-2147483649i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "True");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_less_than_above_max() {
        let schema = table_schema_simple();
        // int32 can hold up to 2147483647
        let expr = Reference::new("bar").less_than(Datum::long(2147483648i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "True");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_less_than_below_min() {
        let schema = table_schema_simple();
        // int32 can hold up to -2147483647
        let expr = Reference::new("bar").less_than(Datum::long(-2147483649i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "False");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_less_than_or_equal_to_above_max() {
        let schema = table_schema_simple();
        // int32 can hold up to 2147483647
        let expr = Reference::new("bar").less_than_or_equal_to(Datum::long(2147483648i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "True");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_less_than_or_equal_to_below_min() {
        let schema = table_schema_simple();
        // int32 can hold up to -2147483647
        let expr = Reference::new("bar").less_than_or_equal_to(Datum::long(-2147483649i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "False");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_great_than_above_max() {
        let schema = table_schema_simple();
        // int32 can hold up to 2147483647
        let expr = Reference::new("bar").greater_than(Datum::long(2147483648i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "False");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_great_than_below_min() {
        let schema = table_schema_simple();
        // int32 can hold up to -2147483647
        let expr = Reference::new("bar").greater_than(Datum::long(-2147483649i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "True");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_great_than_or_equal_to_above_max() {
        let schema = table_schema_simple();
        // int32 can hold up to 2147483647
        let expr = Reference::new("bar").greater_than_or_equal_to(Datum::long(2147483648i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "False");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_great_than_or_equal_to_below_min() {
        let schema = table_schema_simple();
        // int32 can hold up to -2147483647
        let expr = Reference::new("bar").greater_than_or_equal_to(Datum::long(-2147483649i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "True");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_equal_to_wrong_type() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").equal_to(Datum::string("abcd"));
        let bound_expr = expr.bind(schema, true);
        assert!(bound_expr.is_err());
    }

    #[test]
    fn test_bind_not_equal_to() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").not_equal_to(Datum::int(10));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "bar != 10");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_not_equal_to_wrong_type() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").not_equal_to(Datum::string("abcd"));
        let bound_expr = expr.bind(schema, true);
        assert!(bound_expr.is_err());
    }

    #[test]
    fn test_bind_starts_with() {
        let schema = table_schema_simple();
        let expr = Reference::new("foo").starts_with(Datum::string("abcd"));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), r#"foo STARTS WITH "abcd""#);
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_starts_with_wrong_type() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").starts_with(Datum::string("abcd"));
        let bound_expr = expr.bind(schema, true);
        assert!(bound_expr.is_err());
    }

    #[test]
    fn test_bind_not_starts_with() {
        let schema = table_schema_simple();
        let expr = Reference::new("foo").not_starts_with(Datum::string("abcd"));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), r#"foo NOT STARTS WITH "abcd""#);
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_not_starts_with_wrong_type() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").not_starts_with(Datum::string("abcd"));
        let bound_expr = expr.bind(schema, true);
        assert!(bound_expr.is_err());
    }

    #[test]
    fn test_bind_in() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").is_in([Datum::int(10), Datum::int(20)]);
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "bar IN (20, 10)");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_in_empty() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").is_in(vec![]);
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "False");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_in_one_literal() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").is_in(vec![Datum::int(10)]);
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "bar = 10");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_in_wrong_type() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").is_in(vec![Datum::int(10), Datum::string("abcd")]);
        let bound_expr = expr.bind(schema, true);
        assert!(bound_expr.is_err());
    }

    #[test]
    fn test_bind_not_in() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").is_not_in([Datum::int(10), Datum::int(20)]);
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "bar NOT IN (20, 10)");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_not_in_empty() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").is_not_in(vec![]);
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "True");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_not_in_one_literal() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").is_not_in(vec![Datum::int(10)]);
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "bar != 10");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_not_in_wrong_type() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").is_not_in([Datum::int(10), Datum::string("abcd")]);
        let bound_expr = expr.bind(schema, true);
        assert!(bound_expr.is_err());
    }

    #[test]
    fn test_bind_and() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar")
            .less_than(Datum::int(10))
            .and(Reference::new("foo").is_null());
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "(bar < 10) AND (foo IS NULL)");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_and_always_false() {
        let schema = table_schema_simple();
        let expr = Reference::new("foo")
            .less_than(Datum::string("abcd"))
            .and(Reference::new("bar").is_null());
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "False");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_and_always_true() {
        let schema = table_schema_simple();
        let expr = Reference::new("foo")
            .less_than(Datum::string("abcd"))
            .and(Reference::new("bar").is_not_null());
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), r#"foo < "abcd""#);
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_or() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar")
            .less_than(Datum::int(10))
            .or(Reference::new("foo").is_null());
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "(bar < 10) OR (foo IS NULL)");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_or_always_true() {
        let schema = table_schema_simple();
        let expr = Reference::new("foo")
            .less_than(Datum::string("abcd"))
            .or(Reference::new("bar").is_not_null());
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "True");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_or_always_false() {
        let schema = table_schema_simple();
        let expr = Reference::new("foo")
            .less_than(Datum::string("abcd"))
            .or(Reference::new("bar").is_null());
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), r#"foo < "abcd""#);
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_not() {
        let schema = table_schema_simple();
        let expr = !Reference::new("bar").less_than(Datum::int(10));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "NOT (bar < 10)");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_not_always_true() {
        let schema = table_schema_simple();
        let expr = !Reference::new("bar").is_not_null();
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "False");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_not_always_false() {
        let schema = table_schema_simple();
        let expr = !Reference::new("bar").is_null();
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), r#"True"#);
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bound_predicate_rewrite_not_binary() {
        let schema = table_schema_simple();

        // Test NOT elimination on binary predicates: NOT(bar < 10) => bar >= 10
        let predicate = Reference::new("bar").less_than(Datum::int(10)).not();
        let bound_predicate = predicate.bind(schema.clone(), true).unwrap();
        let result = bound_predicate.rewrite_not();

        // The result should be bar >= 10
        let expected_predicate = Reference::new("bar").greater_than_or_equal_to(Datum::int(10));
        let expected_bound = expected_predicate.bind(schema, true).unwrap();

        assert_eq!(result, expected_bound);
        assert_eq!(&format!("{result}"), "bar >= 10");
    }

    #[test]
    fn test_bound_predicate_rewrite_not_unary() {
        let schema = table_schema_simple();

        // Test NOT elimination on unary predicates: NOT(foo IS NULL) => foo IS NOT NULL
        let predicate = Reference::new("foo").is_null().not();
        let bound_predicate = predicate.bind(schema.clone(), true).unwrap();
        let result = bound_predicate.rewrite_not();

        // The result should be foo IS NOT NULL
        let expected_predicate = Reference::new("foo").is_not_null();
        let expected_bound = expected_predicate.bind(schema, true).unwrap();

        assert_eq!(result, expected_bound);
        assert_eq!(&format!("{result}"), "foo IS NOT NULL");
    }

    #[test]
    fn test_bound_predicate_rewrite_not_set() {
        let schema = table_schema_simple();

        // Test NOT elimination on set predicates: NOT(bar IN (10, 20)) => bar NOT IN (10, 20)
        let predicate = Reference::new("bar")
            .is_in([Datum::int(10), Datum::int(20)])
            .not();
        let bound_predicate = predicate.bind(schema.clone(), true).unwrap();
        let result = bound_predicate.rewrite_not();

        // The result should be bar NOT IN (10, 20)
        let expected_predicate = Reference::new("bar").is_not_in([Datum::int(10), Datum::int(20)]);
        let expected_bound = expected_predicate.bind(schema, true).unwrap();

        assert_eq!(result, expected_bound);
        // Note: HashSet order may vary, so we check that it contains the expected format
        let result_str = format!("{result}");
        assert!(
            result_str.contains("bar NOT IN")
                && result_str.contains("10")
                && result_str.contains("20")
        );
    }

    #[test]
    fn test_bound_predicate_rewrite_not_and_demorgan() {
        let schema = table_schema_simple();

        // Test De Morgan's law: NOT(A AND B) = (NOT A) OR (NOT B)
        // NOT((bar < 10) AND (foo IS NULL)) => (bar >= 10) OR (foo IS NOT NULL)
        let predicate = Reference::new("bar")
            .less_than(Datum::int(10))
            .and(Reference::new("foo").is_null())
            .not();

        let bound_predicate = predicate.bind(schema.clone(), true).unwrap();
        let result = bound_predicate.rewrite_not();

        // Expected: (bar >= 10) OR (foo IS NOT NULL)
        let expected_predicate = Reference::new("bar")
            .greater_than_or_equal_to(Datum::int(10))
            .or(Reference::new("foo").is_not_null());

        let expected_bound = expected_predicate.bind(schema, true).unwrap();

        assert_eq!(result, expected_bound);
        assert_eq!(&format!("{result}"), "(bar >= 10) OR (foo IS NOT NULL)");
    }

    #[test]
    fn test_bound_predicate_rewrite_not_or_demorgan() {
        let schema = table_schema_simple();

        // Test De Morgan's law: NOT(A OR B) = (NOT A) AND (NOT B)
        // NOT((bar < 10) OR (foo IS NULL)) => (bar >= 10) AND (foo IS NOT NULL)
        let predicate = Reference::new("bar")
            .less_than(Datum::int(10))
            .or(Reference::new("foo").is_null())
            .not();

        let bound_predicate = predicate.bind(schema.clone(), true).unwrap();
        let result = bound_predicate.rewrite_not();

        // Expected: (bar >= 10) AND (foo IS NOT NULL)
        let expected_predicate = Reference::new("bar")
            .greater_than_or_equal_to(Datum::int(10))
            .and(Reference::new("foo").is_not_null());

        let expected_bound = expected_predicate.bind(schema, true).unwrap();

        assert_eq!(result, expected_bound);
        assert_eq!(&format!("{result}"), "(bar >= 10) AND (foo IS NOT NULL)");
    }

    #[test]
    fn test_bound_predicate_rewrite_not_double_negative() {
        let schema = table_schema_simple();

        // Test double negative elimination: NOT(NOT(bar < 10)) => bar < 10
        let predicate = Reference::new("bar").less_than(Datum::int(10)).not().not();
        let bound_predicate = predicate.bind(schema.clone(), true).unwrap();
        let result = bound_predicate.rewrite_not();

        // The result should be bar < 10 (original predicate)
        let expected_predicate = Reference::new("bar").less_than(Datum::int(10));
        let expected_bound = expected_predicate.bind(schema, true).unwrap();

        assert_eq!(result, expected_bound);
        assert_eq!(&format!("{result}"), "bar < 10");
    }

    #[test]
    fn test_bound_predicate_rewrite_not_always_true_false() {
        let schema = table_schema_simple();

        // Test NOT(AlwaysTrue) => AlwaysFalse
        let predicate = Reference::new("bar").is_not_null().not(); // This becomes NOT(AlwaysTrue) since bar is required
        let bound_predicate = predicate.bind(schema.clone(), true).unwrap();
        let result = bound_predicate.rewrite_not();

        assert_eq!(result, BoundPredicate::AlwaysFalse);
        assert_eq!(&format!("{result}"), "False");

        // Test NOT(AlwaysFalse) => AlwaysTrue
        let predicate2 = Reference::new("bar").is_null().not(); // This becomes NOT(AlwaysFalse) since bar is required
        let bound_predicate2 = predicate2.bind(schema, true).unwrap();
        let result2 = bound_predicate2.rewrite_not();

        assert_eq!(result2, BoundPredicate::AlwaysTrue);
        assert_eq!(&format!("{result2}"), "True");
    }

    #[test]
    fn test_bound_predicate_rewrite_not_complex_nested() {
        let schema = table_schema_simple();

        // Test complex nested expression:
        // NOT(NOT((bar >= 10) AND (foo IS NOT NULL)) OR (bar < 5))
        // Should become: ((bar >= 10) AND (foo IS NOT NULL)) AND (bar >= 5)
        let inner_predicate = Reference::new("bar")
            .greater_than_or_equal_to(Datum::int(10))
            .and(Reference::new("foo").is_not_null())
            .not();

        let complex_predicate = inner_predicate
            .or(Reference::new("bar").less_than(Datum::int(5)))
            .not();

        let bound_predicate = complex_predicate.bind(schema.clone(), true).unwrap();
        let result = bound_predicate.rewrite_not();

        // Expected: ((bar >= 10) AND (foo IS NOT NULL)) AND (bar >= 5)
        // This is because NOT(NOT(A) OR B) = A AND NOT(B)
        let expected_predicate = Reference::new("bar")
            .greater_than_or_equal_to(Datum::int(10))
            .and(Reference::new("foo").is_not_null())
            .and(Reference::new("bar").greater_than_or_equal_to(Datum::int(5)));

        let expected_bound = expected_predicate.bind(schema, true).unwrap();

        assert_eq!(result, expected_bound);
        assert_eq!(
            &format!("{result}"),
            "((bar >= 10) AND (foo IS NOT NULL)) AND (bar >= 5)"
        );
    }

    /// Pins for audit SAF-004 — op/arity validation on the predicate serde
    /// boundary. They lock three properties: (1) valid payloads keep their exact
    /// on-disk bytes (wire-format stability), (2) a wrong-class operator is
    /// rejected at deserialize time with a typed message for every shape, bound
    /// and unbound, and (3) the constructors' `is_*` guards still admit the
    /// valid shapes. The visitor's defensive error path is pinned in the two
    /// visitor modules.
    mod serde_arity_pins {
        use fnv::FnvHashSet;

        use super::*;
        use crate::expr::{
            BinaryExpression, Predicate, PredicateOperator, SetExpression, UnaryExpression,
        };

        // -- Wire-format STABILITY --------------------------------------------
        // Frozen JSON captured from the pre-change serializer. Each payload must
        // still deserialize to the same value AND re-serialize to identical
        // bytes now that the deserialize path routes through the shadow structs
        // — proving already-serialized `FileScanTask` predicates round-trip
        // unchanged (the serialized FORMAT of valid predicates is load-bearing).

        const FROZEN_UNARY: &str = r#"{"Unary":{"op":"IsNull","term":{"name":"bar"}}}"#;
        const FROZEN_BINARY: &str = r#"{"Binary":{"op":"LessThan","term":{"name":"bar"},"literal":{"type":"int","literal":10}}}"#;
        const FROZEN_SET: &str =
            r#"{"Set":{"op":"In","term":{"name":"bar"},"literals":[{"type":"int","literal":10}]}}"#;

        #[test]
        fn wire_format_stable_unary() {
            let expected = Reference::new("bar").is_null();
            let decoded: Predicate =
                serde_json::from_str(FROZEN_UNARY).expect("frozen unary payload must deserialize");
            assert_eq!(decoded, expected);
            assert_eq!(
                serde_json::to_string(&expected).expect("serialize unary"),
                FROZEN_UNARY
            );
        }

        #[test]
        fn wire_format_stable_binary() {
            let expected = Reference::new("bar").less_than(Datum::int(10));
            let decoded: Predicate = serde_json::from_str(FROZEN_BINARY)
                .expect("frozen binary payload must deserialize");
            assert_eq!(decoded, expected);
            assert_eq!(
                serde_json::to_string(&expected).expect("serialize binary"),
                FROZEN_BINARY
            );
        }

        #[test]
        fn wire_format_stable_set() {
            let expected = Reference::new("bar").is_in([Datum::int(10)]);
            let decoded: Predicate =
                serde_json::from_str(FROZEN_SET).expect("frozen set payload must deserialize");
            assert_eq!(decoded, expected);
            assert_eq!(
                serde_json::to_string(&expected).expect("serialize set"),
                FROZEN_SET
            );
        }

        // -- Round-trip per class (unbound + bound) ---------------------------

        #[test]
        fn round_trip_unbound_all_classes() {
            for predicate in [
                Reference::new("bar").is_null(),
                Reference::new("bar").less_than(Datum::int(10)),
                Reference::new("bar").is_in([Datum::int(10), Datum::int(20)]),
            ] {
                let json = serde_json::to_string(&predicate).expect("serialize unbound");
                let decoded: Predicate = serde_json::from_str(&json).expect("deserialize unbound");
                assert_eq!(decoded, predicate);
            }
        }

        #[test]
        fn round_trip_bound_all_classes() {
            let schema = table_schema_simple();
            for predicate in [
                Reference::new("foo").is_null(),
                Reference::new("bar").less_than(Datum::int(10)),
                Reference::new("bar").is_in([Datum::int(10), Datum::int(20)]),
            ] {
                let bound = predicate.bind(schema.clone(), true).expect("bind");
                let json = serde_json::to_string(&bound).expect("serialize bound");
                let decoded: BoundPredicate =
                    serde_json::from_str(&json).expect("deserialize bound");
                assert_eq!(decoded, bound);
            }
        }

        // -- Rejection at the serde boundary (unbound), typed message ---------

        #[test]
        fn reject_unbound_unary_with_binary_op() {
            let json = r#"{"Unary":{"op":"LessThan","term":{"name":"bar"}}}"#;
            let err = serde_json::from_str::<Predicate>(json)
                .expect_err("binary op in unary shape must be rejected");
            let msg = err.to_string();
            assert!(msg.contains("not a unary operator"), "message: {msg}");
            assert!(msg.contains("LessThan"), "message: {msg}");
        }

        #[test]
        fn reject_unbound_binary_with_unary_op() {
            let json = r#"{"Binary":{"op":"IsNull","term":{"name":"bar"},"literal":{"type":"int","literal":10}}}"#;
            let err = serde_json::from_str::<Predicate>(json)
                .expect_err("unary op in binary shape must be rejected");
            let msg = err.to_string();
            assert!(msg.contains("not a binary operator"), "message: {msg}");
            assert!(msg.contains("IsNull"), "message: {msg}");
        }

        #[test]
        fn reject_unbound_binary_with_set_op() {
            let json = r#"{"Binary":{"op":"In","term":{"name":"bar"},"literal":{"type":"int","literal":10}}}"#;
            let err = serde_json::from_str::<Predicate>(json)
                .expect_err("set op in binary shape must be rejected");
            assert!(
                err.to_string().contains("not a binary operator"),
                "message: {err}"
            );
        }

        #[test]
        fn reject_unbound_set_with_binary_op() {
            let json = r#"{"Set":{"op":"LessThan","term":{"name":"bar"},"literals":[{"type":"int","literal":10}]}}"#;
            let err = serde_json::from_str::<Predicate>(json)
                .expect_err("binary op in set shape must be rejected");
            let msg = err.to_string();
            assert!(msg.contains("not a set operator"), "message: {msg}");
            assert!(msg.contains("LessThan"), "message: {msg}");
        }

        // -- Rejection at the serde boundary (bound), typed message -----------
        // A valid bound predicate is serialized, then only its `op` token is
        // rewritten to a wrong-class operator — so the payload is byte-identical
        // to a real one except for the smuggled op, mirroring the wire attack.

        fn corrupt_op(json: &str, from: &str, to: &str) -> String {
            let needle = format!(r#""op":"{from}""#);
            let replacement = format!(r#""op":"{to}""#);
            assert!(
                json.contains(&needle),
                "op token {needle} not present in {json}"
            );
            json.replacen(&needle, &replacement, 1)
        }

        #[test]
        fn reject_bound_unary_with_binary_op() {
            let schema = table_schema_simple();
            let bound = Reference::new("foo")
                .is_null()
                .bind(schema, true)
                .expect("bind");
            let json = serde_json::to_string(&bound).expect("serialize");
            let corrupted = corrupt_op(&json, "IsNull", "LessThan");
            let err = serde_json::from_str::<BoundPredicate>(&corrupted)
                .expect_err("binary op in bound unary shape must be rejected");
            assert!(
                err.to_string().contains("not a unary operator"),
                "message: {err}"
            );
        }

        #[test]
        fn reject_bound_binary_with_unary_op() {
            let schema = table_schema_simple();
            let bound = Reference::new("bar")
                .less_than(Datum::int(10))
                .bind(schema, true)
                .expect("bind");
            let json = serde_json::to_string(&bound).expect("serialize");
            let corrupted = corrupt_op(&json, "LessThan", "IsNull");
            let err = serde_json::from_str::<BoundPredicate>(&corrupted)
                .expect_err("unary op in bound binary shape must be rejected");
            assert!(
                err.to_string().contains("not a binary operator"),
                "message: {err}"
            );
        }

        #[test]
        fn reject_bound_set_with_binary_op() {
            let schema = table_schema_simple();
            let bound = Reference::new("bar")
                .is_in([Datum::int(10), Datum::int(20)])
                .bind(schema, true)
                .expect("bind");
            let json = serde_json::to_string(&bound).expect("serialize");
            let corrupted = corrupt_op(&json, "In", "LessThan");
            let err = serde_json::from_str::<BoundPredicate>(&corrupted)
                .expect_err("binary op in bound set shape must be rejected");
            assert!(
                err.to_string().contains("not a set operator"),
                "message: {err}"
            );
        }

        // -- The valid shapes still deserialize through each shadow -----------

        #[test]
        fn accept_each_valid_class() {
            let unary = r#"{"Unary":{"op":"NotNull","term":{"name":"bar"}}}"#;
            let binary = r#"{"Binary":{"op":"GreaterThan","term":{"name":"bar"},"literal":{"type":"int","literal":10}}}"#;
            let set = r#"{"Set":{"op":"NotIn","term":{"name":"bar"},"literals":[{"type":"int","literal":10}]}}"#;
            assert_eq!(
                serde_json::from_str::<Predicate>(unary).expect("valid unary"),
                Reference::new("bar").is_not_null()
            );
            assert_eq!(
                serde_json::from_str::<Predicate>(binary).expect("valid binary"),
                Reference::new("bar").greater_than(Datum::int(10))
            );
            assert_eq!(
                serde_json::from_str::<Predicate>(set).expect("valid set"),
                Reference::new("bar").is_not_in([Datum::int(10)])
            );
        }

        // Silence unused-import lint for the unchecked constructors, which are
        // exercised by the visitor-module pins rather than here.
        #[test]
        fn unchecked_constructors_build_invalid_shapes() {
            let unary =
                UnaryExpression::new_unchecked(PredicateOperator::LessThan, Reference::new("a"));
            assert_eq!(unary.op(), PredicateOperator::LessThan);
            let binary = BinaryExpression::new_unchecked(
                PredicateOperator::IsNull,
                Reference::new("a"),
                Datum::int(1),
            );
            assert_eq!(binary.op(), PredicateOperator::IsNull);
            let set = SetExpression::new_unchecked(
                PredicateOperator::GreaterThan,
                Reference::new("a"),
                FnvHashSet::from_iter([Datum::int(1)]),
            );
            assert_eq!(set.op(), PredicateOperator::GreaterThan);
        }
    }
}
