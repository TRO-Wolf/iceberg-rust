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
use crate::expr::{Bind, BoundReference, PredicateOperator, Reference};
use crate::spec::{Datum, PrimitiveLiteral, SchemaRef};
use crate::{Error, ErrorKind};

/// The deepest logical nesting that [`Predicate::bind`] and the predicate-tree visitors accept
/// before they return `DataInvalid`. It bounds the RECURSIVE walks only: `bind` and the two
/// visitors. `negate`, `rewrite_not` and both `Display` impls cost O(1) stack at any depth. The
/// DERIVED `Drop`, `Clone` and `PartialEq` glue still recurses, and no depth check can intercept
/// `Drop`, because the tree is destroyed after every gate has run. The number below is a measured
/// stack budget: the bytes one level costs, bisected on a thread of exact `stack_size`.
///
/// | recursive walk                   | dev profile | release profile |
/// |----------------------------------|-------------|-----------------|
/// | `Predicate::bind`                | 5,504 B     | 964 B           |
/// | `bound_predicate_visitor::visit` | 4,650 B     | 787 B           |
/// | `predicate_visitor::visit`       | 4,245 B     | 771 B           |
///
/// Half a 2 MiB tokio worker stack, divided by 964 B, gives 1,087 levels. The `bind` figures hold
/// only while the leaf arms stay in the `#[inline(never)]` [`Predicate::bind_leaf`]. A dev build
/// aborts near level 380 before the limit rejects anything; raise `RUST_MIN_STACK` there. The limit
/// must stay far above real workloads: Java has no counterpart, and the previous value of 100
/// turned working tables into scan failures, at one conjunct per equality-delete file.
pub(crate) const MAX_PREDICATE_DEPTH: usize = 1000;

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
            array_init::from_iter(inputs).ok_or_else(|| {
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

/// Unary predicate, for example `a IS NULL`. `SerdeUnaryExpression` validates its arity.
#[derive(PartialEq, Clone, Serialize, Deserialize)]
#[serde(
    try_from = "SerdeUnaryExpression<T>",
    bound(serialize = "T: Serialize", deserialize = "T: Deserialize<'de>")
)]
pub struct UnaryExpression<T> {
    op: PredicateOperator,
    term: T,
}

/// Serde shadow for [`UnaryExpression`]: [`TryFrom`] rejects a non-unary operator BEFORE a value
/// exists. `new` guards arity with a `debug_assert!` that release builds compile out, and a
/// `BoundPredicate` is reachable over the wire, so a mismatched op would panic the visitor.
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
    /// ```rust
    /// use iceberg::expr::{PredicateOperator, Reference, UnaryExpression};
    ///
    /// UnaryExpression::new(PredicateOperator::IsNull, Reference::new("c"));
    /// ```
    pub fn new(op: PredicateOperator, term: T) -> Self {
        debug_assert!(op.is_unary());
        Self { op, term }
    }

    /// Test-only constructor that bypasses `is_unary`, to build an invalid-arity value.
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

/// Binary predicate, for example `a > 10`. `SerdeBinaryExpression` validates its arity.
#[derive(PartialEq, Clone, Serialize, Deserialize)]
#[serde(
    try_from = "SerdeBinaryExpression<T>",
    bound(serialize = "T: Serialize", deserialize = "T: Deserialize<'de>")
)]
pub struct BinaryExpression<T> {
    op: PredicateOperator,
    term: T,
    literal: Datum,
}

/// Serde shadow for [`BinaryExpression`]; see [`SerdeUnaryExpression`].
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

    /// Test-only constructor that bypasses `is_binary`; see [`UnaryExpression::new_unchecked`].
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

/// Set predicate, for example `a in (1, 2, 3)`. `SerdeSetExpression` validates its arity.
#[derive(PartialEq, Clone, Serialize, Deserialize)]
#[serde(
    try_from = "SerdeSetExpression<T>",
    bound(serialize = "T: Serialize", deserialize = "T: Deserialize<'de>")
)]
pub struct SetExpression<T> {
    op: PredicateOperator,
    term: T,
    literals: FnvHashSet<Datum>,
}

/// Serde shadow for [`SetExpression`]; see [`SerdeUnaryExpression`].
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

    /// Test-only constructor that bypasses `is_set`; see [`UnaryExpression::new_unchecked`].
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
        self.bind_at_depth(schema, case_sensitive, 0)
    }
}

impl Predicate {
    fn bind_at_depth(
        &self,
        schema: SchemaRef,
        case_sensitive: bool,
        depth: usize,
    ) -> Result<BoundPredicate> {
        if depth > MAX_PREDICATE_DEPTH {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Predicate binding exceeds maximum depth {MAX_PREDICATE_DEPTH}"),
            ));
        }

        match self {
            Predicate::And(expr) => {
                let [left_predicate, right_predicate] = expr.inputs();
                let left = Box::new(left_predicate.bind_at_depth(
                    schema.clone(),
                    case_sensitive,
                    depth + 1,
                )?);
                let right =
                    Box::new(right_predicate.bind_at_depth(schema, case_sensitive, depth + 1)?);
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
                let [inner_predicate] = expr.inputs();
                let inner =
                    Box::new(inner_predicate.bind_at_depth(schema, case_sensitive, depth + 1)?);
                Ok(match inner {
                    e if matches!(&*e, &BoundPredicate::AlwaysTrue) => BoundPredicate::AlwaysFalse,
                    e if matches!(&*e, &BoundPredicate::AlwaysFalse) => BoundPredicate::AlwaysTrue,
                    e => BoundPredicate::Not(LogicalExpression::new([e])),
                })
            }
            Predicate::Or(expr) => {
                let [left_predicate, right_predicate] = expr.inputs();
                let left = Box::new(left_predicate.bind_at_depth(
                    schema.clone(),
                    case_sensitive,
                    depth + 1,
                )?);
                let right =
                    Box::new(right_predicate.bind_at_depth(schema, case_sensitive, depth + 1)?);
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
            leaf => leaf.bind_leaf(schema, case_sensitive),
        }
    }

    /// The non-recursive arms of `bind_at_depth`, deliberately NOT inlined: folding them back in
    /// costs `bind` 12,336 bytes per level instead of 5,504, and [`MAX_PREDICATE_DEPTH`] uses it.
    #[inline(never)]
    fn bind_leaf(&self, schema: SchemaRef, case_sensitive: bool) -> Result<BoundPredicate> {
        match self {
            // Unreachable: only leaf arms come here. Typed, so a broken split cannot panic.
            Predicate::And(_) | Predicate::Or(_) | Predicate::Not(_) => Err(Error::new(
                ErrorKind::Unexpected,
                "bind_leaf reached a logical predicate node",
            )),
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

/// One pending item of a `Display` walk: a subtree, or a scheduled separator.
enum DisplayToken<'a, P> {
    Node(&'a P),
    Text(&'static str),
}

impl Display for Predicate {
    /// Renders the predicate with an EXPLICIT STACK, never by recursion. `Display` cannot report an
    /// error and is reachable from any `{}` on a deserialized predicate, so a recursive impl would
    /// abort the process on exactly the trees the depth limit exists to reject.
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut pending = vec![DisplayToken::Node(self)];
        while let Some(token) = pending.pop() {
            let node = match token {
                DisplayToken::Text(text) => {
                    f.write_str(text)?;
                    continue;
                }
                DisplayToken::Node(node) => node,
            };
            match node {
                Predicate::AlwaysTrue => f.write_str("TRUE")?,
                Predicate::AlwaysFalse => f.write_str("FALSE")?,
                // Pushed in reverse: "(" left ") AND (" right ")".
                Predicate::And(expr) => {
                    let [left, right] = expr.inputs();
                    pending.push(DisplayToken::Text(")"));
                    pending.push(DisplayToken::Node(right));
                    pending.push(DisplayToken::Text(") AND ("));
                    pending.push(DisplayToken::Node(left));
                    pending.push(DisplayToken::Text("("));
                }
                Predicate::Or(expr) => {
                    let [left, right] = expr.inputs();
                    pending.push(DisplayToken::Text(")"));
                    pending.push(DisplayToken::Node(right));
                    pending.push(DisplayToken::Text(") OR ("));
                    pending.push(DisplayToken::Node(left));
                    pending.push(DisplayToken::Text("("));
                }
                Predicate::Not(expr) => {
                    let [inner] = expr.inputs();
                    pending.push(DisplayToken::Text(")"));
                    pending.push(DisplayToken::Node(inner));
                    pending.push(DisplayToken::Text("NOT ("));
                }
                Predicate::Unary(expr) => write!(f, "{expr}")?,
                Predicate::Binary(expr) => write!(f, "{expr}")?,
                Predicate::Set(expr) => write!(f, "{expr}")?,
            }
        }
        Ok(())
    }
}

impl Predicate {
    /// Combines two predicates with `AND`.
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

    /// Returns the negation of this predicate, using inverse predicates rather than a `NOT`.
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
    /// De Morgan's laws push the negation through every level, so the walk is as deep as the tree.
    /// It uses an in-place EXPLICIT STACK: `negate` is infallible and public, so it cannot report a
    /// depth error, and a recursive form would abort the process on a deep tree.
    pub fn negate(mut self) -> Predicate {
        // Correctness needs `PredicateOperator::negate` to preserve arity: no node changes shape.
        let mut pending: Vec<&mut Predicate> = vec![&mut self];
        while let Some(node) = pending.pop() {
            // `AlwaysTrue` is a placeholder: it is overwritten before the loop can observe it.
            let (negated, descend) = match std::mem::replace(node, Predicate::AlwaysTrue) {
                Predicate::AlwaysTrue => (Predicate::AlwaysFalse, false),
                Predicate::AlwaysFalse => (Predicate::AlwaysTrue, false),
                Predicate::And(expr) => (Predicate::Or(expr), true),
                Predicate::Or(expr) => (Predicate::And(expr), true),
                // `NOT` cancels: the spliced subtree is already correct, so it is NOT queued.
                Predicate::Not(expr) => {
                    let LogicalExpression { inputs: [input_0] } = expr;
                    (*input_0, false)
                }
                Predicate::Unary(expr) => (
                    Predicate::Unary(UnaryExpression::new(expr.op.negate(), expr.term)),
                    false,
                ),
                Predicate::Binary(expr) => (
                    Predicate::Binary(BinaryExpression::new(
                        expr.op.negate(),
                        expr.term,
                        expr.literal,
                    )),
                    false,
                ),
                Predicate::Set(expr) => (
                    Predicate::Set(SetExpression::new(
                        expr.op.negate(),
                        expr.term,
                        expr.literals,
                    )),
                    false,
                ),
            };
            *node = negated;

            if descend {
                match node {
                    Predicate::And(expr) | Predicate::Or(expr) => {
                        let [left, right] = &mut expr.inputs;
                        pending.push(right);
                        pending.push(left);
                    }
                    // Unreachable: `descend` is only set by the two arms above.
                    _ => {}
                }
            }
        }
        self
    }

    /// Removes `NOT` predicates by negating the inner expressions, through De Morgan's laws.
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
    ///
    /// An EXPLICIT-STACK post-order rewrite. It does NOT route through the recursive,
    /// depth-limited `predicate_visitor::visit`, which would make this infallible `pub` method
    /// abort on a deep filter. The depth limit still applies at [`Predicate::bind`], where it can
    /// be reported. So a node whose operator has the wrong arity passes through untouched here.
    pub fn rewrite_not(self) -> Predicate {
        /// A parent suspended on the child now being rewritten; it holds the sibling result.
        enum Frame {
            AndRight(Predicate),
            AndCombine(Predicate),
            OrRight(Predicate),
            OrCombine(Predicate),
            Negate,
        }

        let mut node = self;
        let mut frames: Vec<Frame> = Vec::new();
        let value: Predicate;

        'descend: loop {
            // Walk down the left spine, suspending each logical node, until a leaf is reached.
            let mut leaf = loop {
                match node {
                    Predicate::And(expr) => {
                        let LogicalExpression {
                            inputs: [left, right],
                        } = expr;
                        frames.push(Frame::AndRight(*right));
                        node = *left;
                    }
                    Predicate::Or(expr) => {
                        let LogicalExpression {
                            inputs: [left, right],
                        } = expr;
                        frames.push(Frame::OrRight(*right));
                        node = *left;
                    }
                    Predicate::Not(expr) => {
                        let LogicalExpression { inputs: [inner] } = expr;
                        frames.push(Frame::Negate);
                        node = *inner;
                    }
                    // Every leaf rewrites to itself, as `RewriteNotVisitor` did.
                    other => break other,
                }
            };

            // The loop returns when the frame stack empties, so it is total without an `unwrap`.
            loop {
                match frames.pop() {
                    None => {
                        value = leaf;
                        break 'descend;
                    }
                    Some(Frame::AndRight(right)) => {
                        frames.push(Frame::AndCombine(leaf));
                        node = right;
                        continue 'descend;
                    }
                    Some(Frame::AndCombine(left)) => leaf = left.and(leaf),
                    Some(Frame::OrRight(right)) => {
                        frames.push(Frame::OrCombine(leaf));
                        node = right;
                        continue 'descend;
                    }
                    Some(Frame::OrCombine(left)) => leaf = left.or(leaf),
                    Some(Frame::Negate) => leaf = leaf.negate(),
                }
            }
        }

        value
    }
}

impl Not for Predicate {
    type Output = Predicate;

    /// Creates the reverse of this predicate. Unlike [`Predicate::negate()`], it rewrites nothing.
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

    /// In-place, explicit-stack negation; see [`Predicate::negate`] for why it does not recurse.
    pub(crate) fn negate(mut self) -> BoundPredicate {
        let mut pending: Vec<&mut BoundPredicate> = vec![&mut self];
        while let Some(node) = pending.pop() {
            // `AlwaysTrue` is a placeholder: it is overwritten before the loop can observe it.
            let (negated, descend) = match std::mem::replace(node, BoundPredicate::AlwaysTrue) {
                BoundPredicate::AlwaysTrue => (BoundPredicate::AlwaysFalse, false),
                BoundPredicate::AlwaysFalse => (BoundPredicate::AlwaysTrue, false),
                BoundPredicate::And(expr) => (BoundPredicate::Or(expr), true),
                BoundPredicate::Or(expr) => (BoundPredicate::And(expr), true),
                // `NOT` cancels: the inner tree is spliced in UNCHANGED and must not be queued.
                BoundPredicate::Not(expr) => {
                    let LogicalExpression { inputs: [input_0] } = expr;
                    (*input_0, false)
                }
                BoundPredicate::Unary(expr) => (
                    BoundPredicate::Unary(UnaryExpression::new(expr.op.negate(), expr.term)),
                    false,
                ),
                BoundPredicate::Binary(expr) => (
                    BoundPredicate::Binary(BinaryExpression::new(
                        expr.op.negate(),
                        expr.term,
                        expr.literal,
                    )),
                    false,
                ),
                BoundPredicate::Set(expr) => (
                    BoundPredicate::Set(SetExpression::new(
                        expr.op.negate(),
                        expr.term,
                        expr.literals,
                    )),
                    false,
                ),
            };
            *node = negated;

            if descend {
                match node {
                    BoundPredicate::And(expr) | BoundPredicate::Or(expr) => {
                        let [left, right] = &mut expr.inputs;
                        pending.push(right);
                        pending.push(left);
                    }
                    // Unreachable: `descend` is only set by the two arms above.
                    _ => {}
                }
            }
        }
        self
    }

    /// Removes `NOT` predicates by negating the inner expressions, through De Morgan's laws.
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
    ///
    /// The explicit-stack bound twin of [`Predicate::rewrite_not`]. See it for why this does not
    /// route through the recursive, depth-limited bound visitor.
    pub fn rewrite_not(self) -> BoundPredicate {
        /// A parent suspended on the child now being rewritten; it holds the sibling result.
        enum Frame {
            AndRight(BoundPredicate),
            AndCombine(BoundPredicate),
            OrRight(BoundPredicate),
            OrCombine(BoundPredicate),
            Negate,
        }

        let mut node = self;
        let mut frames: Vec<Frame> = Vec::new();
        let value: BoundPredicate;

        'descend: loop {
            let mut leaf = loop {
                match node {
                    BoundPredicate::And(expr) => {
                        let LogicalExpression {
                            inputs: [left, right],
                        } = expr;
                        frames.push(Frame::AndRight(*right));
                        node = *left;
                    }
                    BoundPredicate::Or(expr) => {
                        let LogicalExpression {
                            inputs: [left, right],
                        } = expr;
                        frames.push(Frame::OrRight(*right));
                        node = *left;
                    }
                    BoundPredicate::Not(expr) => {
                        let LogicalExpression { inputs: [inner] } = expr;
                        frames.push(Frame::Negate);
                        node = *inner;
                    }
                    other => break other,
                }
            };

            loop {
                match frames.pop() {
                    None => {
                        value = leaf;
                        break 'descend;
                    }
                    Some(Frame::AndRight(right)) => {
                        frames.push(Frame::AndCombine(leaf));
                        node = right;
                        continue 'descend;
                    }
                    Some(Frame::AndCombine(left)) => leaf = left.and(leaf),
                    Some(Frame::OrRight(right)) => {
                        frames.push(Frame::OrCombine(leaf));
                        node = right;
                        continue 'descend;
                    }
                    Some(Frame::OrCombine(left)) => leaf = left.or(leaf),
                    Some(Frame::Negate) => leaf = leaf.negate(),
                }
            }
        }

        value
    }
}

impl Display for BoundPredicate {
    /// Explicit-stack rendering; see the [`Predicate`] impl for why `Display` must not recurse.
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut pending = vec![DisplayToken::Node(self)];
        while let Some(token) = pending.pop() {
            let node = match token {
                DisplayToken::Text(text) => {
                    f.write_str(text)?;
                    continue;
                }
                DisplayToken::Node(node) => node,
            };
            match node {
                BoundPredicate::AlwaysTrue => f.write_str("True")?,
                BoundPredicate::AlwaysFalse => f.write_str("False")?,
                BoundPredicate::And(expr) => {
                    let [left, right] = expr.inputs();
                    pending.push(DisplayToken::Text(")"));
                    pending.push(DisplayToken::Node(right));
                    pending.push(DisplayToken::Text(") AND ("));
                    pending.push(DisplayToken::Node(left));
                    pending.push(DisplayToken::Text("("));
                }
                BoundPredicate::Or(expr) => {
                    let [left, right] = expr.inputs();
                    pending.push(DisplayToken::Text(")"));
                    pending.push(DisplayToken::Node(right));
                    pending.push(DisplayToken::Text(") OR ("));
                    pending.push(DisplayToken::Node(left));
                    pending.push(DisplayToken::Text("("));
                }
                BoundPredicate::Not(expr) => {
                    let [inner] = expr.inputs();
                    pending.push(DisplayToken::Text(")"));
                    pending.push(DisplayToken::Node(inner));
                    pending.push(DisplayToken::Text("NOT ("));
                }
                BoundPredicate::Unary(expr) => write!(f, "{expr}")?,
                BoundPredicate::Binary(expr) => write!(f, "{expr}")?,
                BoundPredicate::Set(expr) => write!(f, "{expr}")?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Not;
    use std::sync::Arc;

    use fnv::FnvHashSet;

    use super::MAX_PREDICATE_DEPTH;
    use crate::expr::Predicate::{AlwaysFalse, AlwaysTrue};
    use crate::expr::visitors::bound_predicate_visitor::{
        BoundPredicateVisitor, visit as visit_bound,
    };
    use crate::expr::visitors::predicate_visitor::visit as visit_unbound;
    use crate::expr::visitors::rewrite_not::RewriteNotVisitor;
    use crate::expr::{
        Bind, BoundPredicate, BoundReference, LogicalExpression, Predicate, Reference,
    };
    use crate::spec::{Datum, NestedField, PrimitiveType, Schema, SchemaRef, Type};
    use crate::{ErrorKind, Result};

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

    fn nested_logical_predicate(depth: usize) -> Predicate {
        let mut predicate = Predicate::AlwaysTrue;
        for level in 0..depth {
            predicate = match level % 3 {
                0 => Predicate::Not(LogicalExpression::new([Box::new(predicate)])),
                1 => Predicate::And(LogicalExpression::new([
                    Box::new(predicate),
                    Box::new(Predicate::AlwaysTrue),
                ])),
                _ => Predicate::Or(LogicalExpression::new([
                    Box::new(predicate),
                    Box::new(Predicate::AlwaysFalse),
                ])),
            };
        }
        predicate
    }

    /// The measured cost of one `bind` level. A depth test sizes its thread from it, never guesses.
    const DEV_BYTES_PER_BIND_LEVEL: usize = 5_504;

    /// Runs `body` on a thread sized for `depth` levels, with a 3x margin over the measured cost.
    fn with_stack_for_depth<T: Send + 'static>(
        depth: usize,
        body: impl FnOnce() -> T + Send + 'static,
    ) -> T {
        let stack_size = (3 * DEV_BYTES_PER_BIND_LEVEL * depth).max(2 * 1024 * 1024);
        std::thread::Builder::new()
            .stack_size(stack_size)
            .spawn(body)
            .expect("spawning the depth-test thread must succeed")
            .join()
            .expect("the depth-test thread must not panic or overflow its stack")
    }

    /// A left-spine `AND` chain: the shape whose depth `MAX_PREDICATE_DEPTH` must clear.
    fn left_folded_conjunction(conjuncts: usize) -> Predicate {
        let mut predicate = Predicate::AlwaysTrue;
        for i in 0..conjuncts {
            let value = i32::try_from(i).expect("the test corpus stays inside i32");
            predicate = predicate.and(Reference::new("bar").equal_to(Datum::int(value)));
        }
        predicate
    }

    /// Regression pin: 512 equality-delete files, each with a 12-level fold, must BIND and VISIT.
    /// Java reads this table, so failing it is a parity break. RED below depth 524.
    #[test]
    fn eq_delete_fold_depth_binds_and_visits() {
        const EQ_DELETE_FILES: usize = 512;
        const PER_FILE_FOLD_LEVELS: usize = 12;
        let depth = EQ_DELETE_FILES + PER_FILE_FOLD_LEVELS;
        assert!(
            depth <= MAX_PREDICATE_DEPTH,
            "the eq-delete fold shape must stay inside the limit"
        );

        let schema = table_schema_simple();
        with_stack_for_depth(depth, move || {
            let predicate = left_folded_conjunction(depth);
            let bound = predicate
                .bind(schema, true)
                .expect("a realistic equality-delete fold must bind");

            let mut evaluator = DepthCountingVisitor { combinators: 0 };
            let leaves = visit_bound(&mut evaluator, &bound)
                .expect("a realistic equality-delete fold must visit");
            assert_eq!(leaves, depth, "every conjunct must reach the visitor");
        });
    }

    /// The limit is inclusive on both walks, and a rejection is typed. Catches `>` drifting to `>=`.
    #[test]
    fn bind_and_visit_depth_limit_is_inclusive() {
        let schema = table_schema_simple();
        with_stack_for_depth(MAX_PREDICATE_DEPTH + 2, move || {
            let at_limit = nested_logical_predicate(MAX_PREDICATE_DEPTH);
            let bound = at_limit
                .bind(schema.clone(), true)
                .expect("binding exactly at the nesting limit must succeed");
            let mut evaluator = DepthCountingVisitor { combinators: 0 };
            assert!(visit_bound(&mut evaluator, &bound).is_ok());

            let beyond_limit = nested_logical_predicate(MAX_PREDICATE_DEPTH + 1);
            let error = beyond_limit
                .bind(schema, true)
                .expect_err("binding beyond the logical nesting limit must be rejected");
            assert_eq!(error.kind(), ErrorKind::DataInvalid);
            assert!(error.to_string().contains("maximum depth"));
        });
    }

    /// Counts leaves, to prove a deep tree is really walked.
    struct DepthCountingVisitor {
        combinators: usize,
    }

    impl BoundPredicateVisitor for DepthCountingVisitor {
        type T = usize;

        fn always_true(&mut self) -> Result<usize> {
            Ok(0)
        }
        fn always_false(&mut self) -> Result<usize> {
            Ok(0)
        }
        fn and(&mut self, lhs: usize, rhs: usize) -> Result<usize> {
            self.combinators += 1;
            Ok(lhs + rhs)
        }
        fn or(&mut self, lhs: usize, rhs: usize) -> Result<usize> {
            self.combinators += 1;
            Ok(lhs + rhs)
        }
        fn not(&mut self, inner: usize) -> Result<usize> {
            Ok(inner)
        }
        fn is_null(&mut self, _r: &BoundReference, _p: &BoundPredicate) -> Result<usize> {
            Ok(1)
        }
        fn not_null(&mut self, _r: &BoundReference, _p: &BoundPredicate) -> Result<usize> {
            Ok(1)
        }
        fn is_nan(&mut self, _r: &BoundReference, _p: &BoundPredicate) -> Result<usize> {
            Ok(1)
        }
        fn not_nan(&mut self, _r: &BoundReference, _p: &BoundPredicate) -> Result<usize> {
            Ok(1)
        }
        fn less_than(
            &mut self,
            _r: &BoundReference,
            _l: &Datum,
            _p: &BoundPredicate,
        ) -> Result<usize> {
            Ok(1)
        }
        fn less_than_or_eq(
            &mut self,
            _r: &BoundReference,
            _l: &Datum,
            _p: &BoundPredicate,
        ) -> Result<usize> {
            Ok(1)
        }
        fn greater_than(
            &mut self,
            _r: &BoundReference,
            _l: &Datum,
            _p: &BoundPredicate,
        ) -> Result<usize> {
            Ok(1)
        }
        fn greater_than_or_eq(
            &mut self,
            _r: &BoundReference,
            _l: &Datum,
            _p: &BoundPredicate,
        ) -> Result<usize> {
            Ok(1)
        }
        fn eq(&mut self, _r: &BoundReference, _l: &Datum, _p: &BoundPredicate) -> Result<usize> {
            Ok(1)
        }
        fn not_eq(
            &mut self,
            _r: &BoundReference,
            _l: &Datum,
            _p: &BoundPredicate,
        ) -> Result<usize> {
            Ok(1)
        }
        fn starts_with(
            &mut self,
            _r: &BoundReference,
            _l: &Datum,
            _p: &BoundPredicate,
        ) -> Result<usize> {
            Ok(1)
        }
        fn not_starts_with(
            &mut self,
            _r: &BoundReference,
            _l: &Datum,
            _p: &BoundPredicate,
        ) -> Result<usize> {
            Ok(1)
        }
        fn r#in(
            &mut self,
            _r: &BoundReference,
            _l: &FnvHashSet<Datum>,
            _p: &BoundPredicate,
        ) -> Result<usize> {
            Ok(1)
        }
        fn not_in(
            &mut self,
            _r: &BoundReference,
            _l: &FnvHashSet<Datum>,
            _p: &BoundPredicate,
        ) -> Result<usize> {
            Ok(1)
        }
    }

    /// Every arm the rewrites take, including the constant-folding doors and stacked `NOT`s.
    fn rewrite_corpus() -> Vec<Predicate> {
        let lt = || Reference::new("bar").less_than(Datum::int(40));
        let gt = || Reference::new("bar").greater_than(Datum::int(3));
        let null = || Reference::new("foo").is_null();
        let set = || Reference::new("bar").is_in([Datum::int(1), Datum::int(2)]);

        vec![
            Predicate::AlwaysTrue,
            Predicate::AlwaysFalse,
            lt(),
            null(),
            set(),
            lt().not(),
            null().not(),
            set().not(),
            lt().not().not(),
            lt().not().not().not(),
            lt().and(gt()),
            lt().or(gt()),
            lt().and(gt()).not(),
            lt().or(gt()).not(),
            lt().and(gt()).not().not(),
            lt().and(null().or(set())).not(),
            lt().not().and(gt().not()).or(set().not()).not(),
            Predicate::And(LogicalExpression::new([
                Box::new(Predicate::AlwaysTrue),
                Box::new(lt()),
            ])),
            Predicate::Or(LogicalExpression::new([
                Box::new(Predicate::AlwaysFalse),
                Box::new(lt()),
            ])),
            Predicate::Not(LogicalExpression::new([Box::new(Predicate::And(
                LogicalExpression::new([Box::new(Predicate::AlwaysTrue), Box::new(lt())]),
            ))])),
            nested_logical_predicate(9),
            left_folded_conjunction(6),
        ]
    }

    /// Differential oracle for `Predicate::rewrite_not`. Catches swapping `and`/`or`, reversing the
    /// child order, dropping the `Negate` frame, and splicing a cancelled `NOT` wrongly.
    #[test]
    fn rewrite_not_matches_the_visitor_oracle() {
        for predicate in rewrite_corpus() {
            let oracle = visit_unbound(&mut RewriteNotVisitor::new(), &predicate)
                .expect("the corpus contains no malformed operators");
            let iterative = predicate.clone().rewrite_not();
            assert_eq!(
                iterative, oracle,
                "iterative rewrite_not diverged from the visitor on {predicate}"
            );
        }
    }

    /// Differential oracle for the explicit-stack `BoundPredicate::rewrite_not`.
    #[test]
    fn bound_rewrite_not_matches_the_visitor_oracle() {
        let schema = table_schema_simple();
        for predicate in rewrite_corpus() {
            let bound = predicate
                .bind(schema.clone(), true)
                .expect("the corpus binds against the simple test schema");
            let oracle = visit_bound(&mut RewriteNotVisitor::new(), &bound)
                .expect("the corpus contains no malformed operators");
            let iterative = bound.clone().rewrite_not();
            assert_eq!(
                iterative, oracle,
                "iterative bound rewrite_not diverged from the visitor on {bound}"
            );
        }
    }

    /// `negate` must splice a cancelled `NOT` in UNCHANGED, or it negates a correct subtree.
    #[test]
    fn negate_splices_a_cancelled_not_without_descending() {
        let inner = Reference::new("bar")
            .less_than(Datum::int(40))
            .and(Reference::new("foo").is_null());
        let negated = inner.clone().not().negate();
        assert_eq!(negated, inner);
        assert_eq!(&format!("{negated}"), "(bar < 40) AND (foo IS NULL)");
    }

    /// A recursive renderer, used only as the `Display` oracle.
    fn render_recursively(predicate: &Predicate) -> String {
        match predicate {
            Predicate::AlwaysTrue => "TRUE".to_string(),
            Predicate::AlwaysFalse => "FALSE".to_string(),
            Predicate::And(expr) => {
                let [l, r] = expr.inputs();
                format!(
                    "({}) AND ({})",
                    render_recursively(l),
                    render_recursively(r)
                )
            }
            Predicate::Or(expr) => {
                let [l, r] = expr.inputs();
                format!("({}) OR ({})", render_recursively(l), render_recursively(r))
            }
            Predicate::Not(expr) => {
                let [inner] = expr.inputs();
                format!("NOT ({})", render_recursively(inner))
            }
            Predicate::Unary(expr) => format!("{expr}"),
            Predicate::Binary(expr) => format!("{expr}"),
            Predicate::Set(expr) => format!("{expr}"),
        }
    }

    /// `Display` must be byte-identical to the recursive form. Catches an out-of-order token push,
    /// a dropped parenthesis, and a wrong separator on a logical arm.
    #[test]
    fn display_matches_the_recursive_rendering() {
        for predicate in rewrite_corpus() {
            assert_eq!(
                format!("{predicate}"),
                render_recursively(&predicate),
                "explicit-stack Display diverged from the recursive rendering"
            );
        }
    }

    /// `rewrite_not`, `negate` and `Display` must cost O(1) stack: they are infallible and `pub`,
    /// so they cannot report a depth error. Run 50x beyond the limit on a 2 MiB stack.
    #[test]
    fn deep_trees_are_rewritten_negated_and_rendered_without_recursion() {
        const DEPTH: usize = 50_000;
        const TOKIO_WORKER_STACK: usize = 2 * 1024 * 1024;

        let handle = std::thread::Builder::new()
            .stack_size(TOKIO_WORKER_STACK)
            .spawn(|| {
                // The DERIVED `Clone`, `PartialEq` and `Drop` recurse: never apply them here.
                let build_deep = || {
                    let mut deep = Reference::new("bar").less_than(Datum::int(40));
                    for _ in 0..DEPTH {
                        deep = deep.not();
                    }
                    deep
                };

                // Leak before asserting: a failure would unwind through `Drop` and abort.
                let subject = build_deep();
                let rendered = format!("{subject}");
                std::mem::forget(subject);

                // 50,000 stacked NOTs cancel in pairs, leaving the original leaf.
                let rewritten = build_deep().rewrite_not();
                let rewritten_text = format!("{rewritten}");
                std::mem::forget(rewritten);

                // One NOT peeled off the front, nothing else touched.
                let negated = build_deep().negate();
                let negated_text = format!("{negated}");
                std::mem::forget(negated);

                // A `NOT` chain never makes `negate` descend: its `Not` arm splices and stops.
                let spine = left_folded_conjunction(DEPTH);
                let spine_text = format!("{spine}");
                std::mem::forget(spine);

                let spine_negated = left_folded_conjunction(DEPTH).negate();
                let spine_negated_text = format!("{spine_negated}");
                std::mem::forget(spine_negated);

                let spine_rewritten = left_folded_conjunction(DEPTH).rewrite_not();
                let spine_rewritten_text = format!("{spine_rewritten}");
                std::mem::forget(spine_rewritten);

                assert!(rendered.starts_with("NOT (NOT ("));
                assert_eq!(rendered.matches("NOT (").count(), DEPTH);
                assert_eq!(rendered.matches("bar < 40").count(), 1);
                assert_eq!(rendered.matches(')').count(), DEPTH);
                assert_eq!(rewritten_text, "bar < 40");
                assert_eq!(negated_text, rendered[5..rendered.len() - 1]);

                // De Morgan turns every `AND` into an `OR` and every `=` leaf into a `!=`.
                assert_eq!(spine_text.matches(") AND (").count(), DEPTH - 1);
                assert_eq!(spine_negated_text.matches(") OR (").count(), DEPTH - 1);
                assert_eq!(spine_negated_text.matches(") AND (").count(), 0);
                assert_eq!(spine_negated_text.matches("bar != ").count(), DEPTH);
                assert_eq!(spine_rewritten_text, spine_text);

                // The derived `Drop` is the one walk no depth check can intercept, so leak it.
            })
            .expect("spawning the deep-tree thread must succeed");
        handle
            .join()
            .expect("the iterative walks must not overflow a 2 MiB stack");
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
        let expr = Reference::new("bar").equal_to(Datum::long(2147483648i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "False");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_equal_to_below_min() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").equal_to(Datum::long(-2147483649i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "False");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_not_equal_to_above_max() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").not_equal_to(Datum::long(2147483648i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "True");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_not_equal_to_below_min() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").not_equal_to(Datum::long(-2147483649i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "True");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_less_than_above_max() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").less_than(Datum::long(2147483648i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "True");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_less_than_below_min() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").less_than(Datum::long(-2147483649i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "False");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_less_than_or_equal_to_above_max() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").less_than_or_equal_to(Datum::long(2147483648i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "True");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_less_than_or_equal_to_below_min() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").less_than_or_equal_to(Datum::long(-2147483649i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "False");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_great_than_above_max() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").greater_than(Datum::long(2147483648i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "False");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_great_than_below_min() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").greater_than(Datum::long(-2147483649i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "True");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_great_than_or_equal_to_above_max() {
        let schema = table_schema_simple();
        let expr = Reference::new("bar").greater_than_or_equal_to(Datum::long(2147483648i64));
        let bound_expr = expr.bind(schema, true).unwrap();
        assert_eq!(&format!("{bound_expr}"), "False");
        test_bound_predicate_serialize_diserialize(bound_expr);
    }

    #[test]
    fn test_bind_great_than_or_equal_to_below_min() {
        let schema = table_schema_simple();
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

        let predicate = Reference::new("bar").less_than(Datum::int(10)).not();
        let bound_predicate = predicate.bind(schema.clone(), true).unwrap();
        let result = bound_predicate.rewrite_not();

        let expected_predicate = Reference::new("bar").greater_than_or_equal_to(Datum::int(10));
        let expected_bound = expected_predicate.bind(schema, true).unwrap();

        assert_eq!(result, expected_bound);
        assert_eq!(&format!("{result}"), "bar >= 10");
    }

    #[test]
    fn test_bound_predicate_rewrite_not_unary() {
        let schema = table_schema_simple();

        let predicate = Reference::new("foo").is_null().not();
        let bound_predicate = predicate.bind(schema.clone(), true).unwrap();
        let result = bound_predicate.rewrite_not();

        let expected_predicate = Reference::new("foo").is_not_null();
        let expected_bound = expected_predicate.bind(schema, true).unwrap();

        assert_eq!(result, expected_bound);
        assert_eq!(&format!("{result}"), "foo IS NOT NULL");
    }

    #[test]
    fn test_bound_predicate_rewrite_not_set() {
        let schema = table_schema_simple();

        let predicate = Reference::new("bar")
            .is_in([Datum::int(10), Datum::int(20)])
            .not();
        let bound_predicate = predicate.bind(schema.clone(), true).unwrap();
        let result = bound_predicate.rewrite_not();

        let expected_predicate = Reference::new("bar").is_not_in([Datum::int(10), Datum::int(20)]);
        let expected_bound = expected_predicate.bind(schema, true).unwrap();

        assert_eq!(result, expected_bound);
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

        let predicate = Reference::new("bar")
            .less_than(Datum::int(10))
            .and(Reference::new("foo").is_null())
            .not();

        let bound_predicate = predicate.bind(schema.clone(), true).unwrap();
        let result = bound_predicate.rewrite_not();

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

        let predicate = Reference::new("bar")
            .less_than(Datum::int(10))
            .or(Reference::new("foo").is_null())
            .not();

        let bound_predicate = predicate.bind(schema.clone(), true).unwrap();
        let result = bound_predicate.rewrite_not();

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

        let predicate = Reference::new("bar").less_than(Datum::int(10)).not().not();
        let bound_predicate = predicate.bind(schema.clone(), true).unwrap();
        let result = bound_predicate.rewrite_not();

        let expected_predicate = Reference::new("bar").less_than(Datum::int(10));
        let expected_bound = expected_predicate.bind(schema, true).unwrap();

        assert_eq!(result, expected_bound);
        assert_eq!(&format!("{result}"), "bar < 10");
    }

    #[test]
    fn test_bound_predicate_rewrite_not_always_true_false() {
        let schema = table_schema_simple();

        let predicate = Reference::new("bar").is_not_null().not(); // This becomes NOT(AlwaysTrue) since bar is required
        let bound_predicate = predicate.bind(schema.clone(), true).unwrap();
        let result = bound_predicate.rewrite_not();

        assert_eq!(result, BoundPredicate::AlwaysFalse);
        assert_eq!(&format!("{result}"), "False");

        let predicate2 = Reference::new("bar").is_null().not(); // This becomes NOT(AlwaysFalse) since bar is required
        let bound_predicate2 = predicate2.bind(schema, true).unwrap();
        let result2 = bound_predicate2.rewrite_not();

        assert_eq!(result2, BoundPredicate::AlwaysTrue);
        assert_eq!(&format!("{result2}"), "True");
    }

    #[test]
    fn test_bound_predicate_rewrite_not_complex_nested() {
        let schema = table_schema_simple();

        let inner_predicate = Reference::new("bar")
            .greater_than_or_equal_to(Datum::int(10))
            .and(Reference::new("foo").is_not_null())
            .not();

        let complex_predicate = inner_predicate
            .or(Reference::new("bar").less_than(Datum::int(5)))
            .not();

        let bound_predicate = complex_predicate.bind(schema.clone(), true).unwrap();
        let result = bound_predicate.rewrite_not();

        // NOT(NOT(A) OR B) = A AND NOT(B)
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

    /// Op and arity validation on the predicate serde boundary. Valid payloads keep their exact
    /// on-disk bytes, and a wrong-class operator is rejected for every shape.
    mod serde_arity_pins {
        use fnv::FnvHashSet;

        use super::*;
        use crate::expr::{
            BinaryExpression, Predicate, PredicateOperator, SetExpression, UnaryExpression,
        };

        // -- Wire-format STABILITY --------------------------------------------
        // Frozen JSON: each payload must re-serialize to identical bytes through the shadows.

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
        // Only the `op` token of a real payload is rewritten, mirroring the wire attack.

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

        // The unchecked constructors are exercised by the visitor-module pins, not here.
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
