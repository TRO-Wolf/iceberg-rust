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

use super::bound_predicate_visitor::{BoundPredicateVisitor, visit};
use crate::expr::{BoundPredicate, BoundReference};
use crate::spec::{DataFile, Datum, PrimitiveLiteral, Struct};
use crate::{Error, ErrorKind, Result};

/// Evaluates a [`DataFile`]'s partition [`Struct`] to check
/// if the partition tuples match the given [`BoundPredicate`].
///
/// Use within [`TableScan`] to prune the list of [`DataFile`]s
/// that could potentially match the TableScan's filter.
#[derive(Debug)]
pub(crate) struct ExpressionEvaluator {
    /// The provided partition filter.
    partition_filter: BoundPredicate,
}

impl ExpressionEvaluator {
    /// Creates a new [`ExpressionEvaluator`].
    pub(crate) fn new(partition_filter: BoundPredicate) -> Self {
        Self { partition_filter }
    }

    /// Evaluate this [`ExpressionEvaluator`]'s partition filter against
    /// the provided [`DataFile`]'s partition [`Struct`]. Used by [`TableScan`]
    /// to see if this [`DataFile`] could possibly contain data that matches
    /// the scan's filter.
    pub(crate) fn eval(&self, data_file: &DataFile) -> Result<bool> {
        let mut visitor = ExpressionEvaluatorVisitor::new(data_file.partition());

        visit(&mut visitor, &self.partition_filter)
    }
}

/// Acts as a visitor for [`ExpressionEvaluator`] to apply
/// evaluation logic to different parts of a data structure,
/// specifically for data file partitions.
///
/// Exposed as `pub(crate)` so the residual evaluator can reuse the exact
/// per-operator evaluation of a bound predicate against a partition [`Struct`]
/// (it evaluates the strict/inclusive *projected* predicate the same way).
#[derive(Debug)]
pub(crate) struct ExpressionEvaluatorVisitor<'a> {
    /// Reference to a [`DataFile`]'s partition [`Struct`].
    partition: &'a Struct,
}

impl<'a> ExpressionEvaluatorVisitor<'a> {
    /// Creates a new [`ExpressionEvaluatorVisitor`].
    pub(crate) fn new(partition: &'a Struct) -> Self {
        Self { partition }
    }
}

impl BoundPredicateVisitor for ExpressionEvaluatorVisitor<'_> {
    type T = bool;

    fn always_true(&mut self) -> Result<bool> {
        Ok(true)
    }

    fn always_false(&mut self) -> Result<bool> {
        Ok(false)
    }

    fn and(&mut self, lhs: bool, rhs: bool) -> Result<bool> {
        Ok(lhs && rhs)
    }

    fn or(&mut self, lhs: bool, rhs: bool) -> Result<bool> {
        Ok(lhs || rhs)
    }

    fn not(&mut self, _inner: bool) -> Result<bool> {
        Err(Error::new(
            ErrorKind::Unexpected,
            "The evaluation of expressions should not be performed against Predicates that contain a Not operator. Ensure that \"Rewrite Not\" gets applied to the originating Predicate before binding it.",
        ))
    }

    fn is_null(&mut self, reference: &BoundReference, _predicate: &BoundPredicate) -> Result<bool> {
        match reference.accessor().get(self.partition)? {
            Some(_) => Ok(false),
            None => Ok(true),
        }
    }

    fn not_null(
        &mut self,
        reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        match reference.accessor().get(self.partition)? {
            Some(_) => Ok(true),
            None => Ok(false),
        }
    }

    fn is_nan(&mut self, reference: &BoundReference, _predicate: &BoundPredicate) -> Result<bool> {
        match reference.accessor().get(self.partition)? {
            Some(datum) => Ok(datum.is_nan()),
            None => Ok(false),
        }
    }

    fn not_nan(&mut self, reference: &BoundReference, _predicate: &BoundPredicate) -> Result<bool> {
        match reference.accessor().get(self.partition)? {
            Some(datum) => Ok(!datum.is_nan()),
            None => Ok(true),
        }
    }

    // Null-partition (`None`) arms below port Java's nulls-first TOTAL order: iceberg-core
    // 1.10.0 `ManifestReader.evaluator()` builds `new Evaluator(spec.partitionType(),
    // inclusive-projection ∧ partFilter, caseSensitive)` (bytecode offsets 7-55) and applies it
    // to `entry.file().partition()` (`lambda$entries$0` offsets 4-16), and
    // `Evaluator$EvalVisitor` compares via `Comparators.nullsFirst().thenComparing(
    // naturalOrder)` — compare(null, lit) == -1. So for a null partition value: `<`/`<=`/`!=`/
    // `not_in`/`not_starts_with` ⇒ true (KEEP the file), `>`/`>=`/`==`/`in`/`starts_with` ⇒
    // false. The same verdicts flow into `ResidualEvaluator` (its Java `ResidualVisitor.lt`
    // etc. use the identical comparator chain, offsets 0-27), which reuses this visitor.

    fn less_than(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        match reference.accessor().get(self.partition)? {
            Some(datum) => Ok(&datum < literal),
            // null < lit is TRUE under nulls-first — `Ok(false)` here silently PRUNED files
            // whose partition value is null (audit BUG-011, lost rows).
            None => Ok(true),
        }
    }

    fn less_than_or_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        match reference.accessor().get(self.partition)? {
            Some(datum) => Ok(&datum <= literal),
            // null <= lit is TRUE under nulls-first (Java `ltEq`, `ifgt 36` over -1).
            None => Ok(true),
        }
    }

    fn greater_than(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        match reference.accessor().get(self.partition)? {
            Some(datum) => Ok(&datum > literal),
            None => Ok(false),
        }
    }

    fn greater_than_or_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        match reference.accessor().get(self.partition)? {
            Some(datum) => Ok(&datum >= literal),
            None => Ok(false),
        }
    }

    fn eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        match reference.accessor().get(self.partition)? {
            Some(datum) => Ok(&datum == literal),
            None => Ok(false),
        }
    }

    fn not_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        match reference.accessor().get(self.partition)? {
            Some(datum) => Ok(&datum != literal),
            None => Ok(true),
        }
    }

    fn starts_with(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        let Some(datum) = reference.accessor().get(self.partition)? else {
            return Ok(false);
        };

        match (datum.literal(), literal.literal()) {
            (PrimitiveLiteral::String(d), PrimitiveLiteral::String(l)) => Ok(d.starts_with(l)),
            _ => Ok(false),
        }
    }

    fn not_starts_with(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(!self.starts_with(reference, literal, _predicate)?)
    }

    fn r#in(
        &mut self,
        reference: &BoundReference,
        literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        match reference.accessor().get(self.partition)? {
            Some(datum) => Ok(literals.contains(&datum)),
            None => Ok(false),
        }
    }

    fn not_in(
        &mut self,
        reference: &BoundReference,
        literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        match reference.accessor().get(self.partition)? {
            Some(datum) => Ok(!literals.contains(&datum)),
            None => Ok(true),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use fnv::FnvHashSet;
    use predicate::SetExpression;

    use super::ExpressionEvaluator;
    use crate::Result;
    use crate::expr::visitors::inclusive_projection::InclusiveProjection;
    use crate::expr::{
        BinaryExpression, Bind, BoundPredicate, Predicate, PredicateOperator, Reference,
        UnaryExpression, predicate,
    };
    use crate::spec::{
        DataContentType, DataFile, DataFileFormat, Datum, Literal, NestedField, PartitionSpec,
        PartitionSpecRef, PrimitiveType, Schema, SchemaRef, Struct, Transform, Type,
        UnboundPartitionField,
    };

    fn create_partition_spec(r#type: PrimitiveType) -> Result<(PartitionSpecRef, SchemaRef)> {
        let schema = Schema::builder()
            .with_fields(vec![Arc::new(NestedField::optional(
                1,
                "a",
                Type::Primitive(r#type),
            ))])
            .build()?;

        let spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(1)
            .add_unbound_field(
                UnboundPartitionField::builder()
                    .source_id(1)
                    .name("a".to_string())
                    .field_id(1)
                    .transform(Transform::Identity)
                    .build(),
            )
            .unwrap()
            .build()
            .unwrap();

        Ok((Arc::new(spec), schema.into()))
    }

    fn create_partition_filter(
        partition_spec: PartitionSpecRef,
        schema: &Schema,
        predicate: &BoundPredicate,
        case_sensitive: bool,
    ) -> Result<BoundPredicate> {
        let partition_type = partition_spec.partition_type(schema).unwrap();
        let partition_fields = partition_type.fields().to_owned();

        let partition_schema = Schema::builder()
            .with_schema_id(partition_spec.spec_id())
            .with_fields(partition_fields)
            .build()?;

        let mut inclusive_projection = InclusiveProjection::new((*partition_spec).clone().into());

        let partition_filter = inclusive_projection
            .project(predicate)?
            .rewrite_not()
            .bind(Arc::new(partition_schema), case_sensitive)?;

        Ok(partition_filter)
    }

    fn create_expression_evaluator(
        partition_spec: PartitionSpecRef,
        schema: &Schema,
        predicate: &BoundPredicate,
        case_sensitive: bool,
    ) -> Result<ExpressionEvaluator> {
        let partition_filter =
            create_partition_filter(partition_spec, schema, predicate, case_sensitive)?;

        Ok(ExpressionEvaluator::new(partition_filter))
    }

    fn create_data_file_float() -> DataFile {
        let partition = Struct::from_iter([Some(Literal::float(1.0))]);

        DataFile {
            content: DataContentType::Data,
            file_path: "/test/path".to_string(),
            file_format: DataFileFormat::Parquet,
            partition,
            record_count: 1,
            file_size_in_bytes: 1,
            column_sizes: HashMap::new(),
            value_counts: HashMap::new(),
            null_value_counts: HashMap::new(),
            nan_value_counts: HashMap::new(),
            lower_bounds: HashMap::new(),
            upper_bounds: HashMap::new(),
            key_metadata: None,
            split_offsets: None,
            equality_ids: None,
            sort_order_id: None,
            partition_spec_id: 0,
            first_row_id: None,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
        }
    }

    fn create_data_file_string() -> DataFile {
        let partition = Struct::from_iter([Some(Literal::string("test str"))]);

        DataFile {
            content: DataContentType::Data,
            file_path: "/test/path".to_string(),
            file_format: DataFileFormat::Parquet,
            partition,
            record_count: 1,
            file_size_in_bytes: 1,
            column_sizes: HashMap::new(),
            value_counts: HashMap::new(),
            null_value_counts: HashMap::new(),
            nan_value_counts: HashMap::new(),
            lower_bounds: HashMap::new(),
            upper_bounds: HashMap::new(),
            key_metadata: None,
            split_offsets: None,
            equality_ids: None,
            sort_order_id: None,
            partition_spec_id: 0,
            first_row_id: None,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
        }
    }

    #[test]
    fn test_expr_or() -> Result<()> {
        let case_sensitive = true;
        let (partition_spec, schema) = create_partition_spec(PrimitiveType::Float)?;

        let predicate = Predicate::Binary(BinaryExpression::new(
            PredicateOperator::LessThan,
            Reference::new("a"),
            Datum::float(1.0),
        ))
        .or(Predicate::Binary(BinaryExpression::new(
            PredicateOperator::GreaterThanOrEq,
            Reference::new("a"),
            Datum::float(0.4),
        )))
        .bind(schema.clone(), case_sensitive)?;

        let expression_evaluator =
            create_expression_evaluator(partition_spec, &schema, &predicate, case_sensitive)?;

        let data_file = create_data_file_float();

        let result = expression_evaluator.eval(&data_file)?;

        assert!(result);

        Ok(())
    }

    #[test]
    fn test_expr_and() -> Result<()> {
        let case_sensitive = true;
        let (partition_spec, schema) = create_partition_spec(PrimitiveType::Float)?;

        let predicate = Predicate::Binary(BinaryExpression::new(
            PredicateOperator::LessThan,
            Reference::new("a"),
            Datum::float(1.1),
        ))
        .and(Predicate::Binary(BinaryExpression::new(
            PredicateOperator::GreaterThanOrEq,
            Reference::new("a"),
            Datum::float(0.4),
        )))
        .bind(schema.clone(), case_sensitive)?;

        let expression_evaluator =
            create_expression_evaluator(partition_spec, &schema, &predicate, case_sensitive)?;

        let data_file = create_data_file_float();

        let result = expression_evaluator.eval(&data_file)?;

        assert!(result);

        Ok(())
    }

    #[test]
    fn test_expr_not_in() -> Result<()> {
        let case_sensitive = true;
        let (partition_spec, schema) = create_partition_spec(PrimitiveType::Float)?;

        let predicate = Predicate::Set(SetExpression::new(
            PredicateOperator::NotIn,
            Reference::new("a"),
            FnvHashSet::from_iter([Datum::float(0.9), Datum::float(1.2), Datum::float(2.4)]),
        ))
        .bind(schema.clone(), case_sensitive)?;

        let expression_evaluator =
            create_expression_evaluator(partition_spec, &schema, &predicate, case_sensitive)?;

        let data_file = create_data_file_float();

        let result = expression_evaluator.eval(&data_file)?;

        assert!(result);

        Ok(())
    }

    #[test]
    fn test_expr_in() -> Result<()> {
        let case_sensitive = true;
        let (partition_spec, schema) = create_partition_spec(PrimitiveType::Float)?;

        let predicate = Predicate::Set(SetExpression::new(
            PredicateOperator::In,
            Reference::new("a"),
            FnvHashSet::from_iter([Datum::float(1.0), Datum::float(1.2), Datum::float(2.4)]),
        ))
        .bind(schema.clone(), case_sensitive)?;

        let expression_evaluator =
            create_expression_evaluator(partition_spec, &schema, &predicate, case_sensitive)?;

        let data_file = create_data_file_float();

        let result = expression_evaluator.eval(&data_file)?;

        assert!(result);

        Ok(())
    }

    #[test]
    fn test_expr_not_starts_with() -> Result<()> {
        let case_sensitive = true;
        let (partition_spec, schema) = create_partition_spec(PrimitiveType::String)?;

        let predicate = Predicate::Binary(BinaryExpression::new(
            PredicateOperator::NotStartsWith,
            Reference::new("a"),
            Datum::string("not"),
        ))
        .bind(schema.clone(), case_sensitive)?;

        let expression_evaluator =
            create_expression_evaluator(partition_spec, &schema, &predicate, case_sensitive)?;

        let data_file = create_data_file_string();

        let result = expression_evaluator.eval(&data_file)?;

        assert!(result);

        Ok(())
    }

    #[test]
    fn test_expr_starts_with() -> Result<()> {
        let case_sensitive = true;
        let (partition_spec, schema) = create_partition_spec(PrimitiveType::String)?;

        let predicate = Predicate::Binary(BinaryExpression::new(
            PredicateOperator::StartsWith,
            Reference::new("a"),
            Datum::string("test"),
        ))
        .bind(schema.clone(), case_sensitive)?;

        let expression_evaluator =
            create_expression_evaluator(partition_spec, &schema, &predicate, case_sensitive)?;

        let data_file = create_data_file_string();

        let result = expression_evaluator.eval(&data_file)?;

        assert!(result);

        Ok(())
    }

    #[test]
    fn test_expr_not_eq() -> Result<()> {
        let case_sensitive = true;
        let (partition_spec, schema) = create_partition_spec(PrimitiveType::Float)?;

        let predicate = Predicate::Binary(BinaryExpression::new(
            PredicateOperator::NotEq,
            Reference::new("a"),
            Datum::float(0.9),
        ))
        .bind(schema.clone(), case_sensitive)?;

        let expression_evaluator =
            create_expression_evaluator(partition_spec, &schema, &predicate, case_sensitive)?;

        let data_file = create_data_file_float();

        let result = expression_evaluator.eval(&data_file)?;

        assert!(result);

        Ok(())
    }

    #[test]
    fn test_expr_eq() -> Result<()> {
        let case_sensitive = true;
        let (partition_spec, schema) = create_partition_spec(PrimitiveType::Float)?;

        let predicate = Predicate::Binary(BinaryExpression::new(
            PredicateOperator::Eq,
            Reference::new("a"),
            Datum::float(1.0),
        ))
        .bind(schema.clone(), case_sensitive)?;

        let expression_evaluator =
            create_expression_evaluator(partition_spec, &schema, &predicate, case_sensitive)?;

        let data_file = create_data_file_float();

        let result = expression_evaluator.eval(&data_file)?;

        assert!(result);

        Ok(())
    }

    #[test]
    fn test_expr_greater_than_or_eq() -> Result<()> {
        let case_sensitive = true;
        let (partition_spec, schema) = create_partition_spec(PrimitiveType::Float)?;

        let predicate = Predicate::Binary(BinaryExpression::new(
            PredicateOperator::GreaterThanOrEq,
            Reference::new("a"),
            Datum::float(1.0),
        ))
        .bind(schema.clone(), case_sensitive)?;

        let expression_evaluator =
            create_expression_evaluator(partition_spec, &schema, &predicate, case_sensitive)?;

        let data_file = create_data_file_float();

        let result = expression_evaluator.eval(&data_file)?;

        assert!(result);

        Ok(())
    }

    #[test]
    fn test_expr_greater_than() -> Result<()> {
        let case_sensitive = true;
        let (partition_spec, schema) = create_partition_spec(PrimitiveType::Float)?;

        let predicate = Predicate::Binary(BinaryExpression::new(
            PredicateOperator::GreaterThan,
            Reference::new("a"),
            Datum::float(0.9),
        ))
        .bind(schema.clone(), case_sensitive)?;

        let expression_evaluator =
            create_expression_evaluator(partition_spec, &schema, &predicate, case_sensitive)?;

        let data_file = create_data_file_float();

        let result = expression_evaluator.eval(&data_file)?;

        assert!(result);

        Ok(())
    }

    #[test]
    fn test_expr_less_than_or_eq() -> Result<()> {
        let case_sensitive = true;
        let (partition_spec, schema) = create_partition_spec(PrimitiveType::Float)?;

        let predicate = Predicate::Binary(BinaryExpression::new(
            PredicateOperator::LessThanOrEq,
            Reference::new("a"),
            Datum::float(1.0),
        ))
        .bind(schema.clone(), case_sensitive)?;

        let expression_evaluator =
            create_expression_evaluator(partition_spec, &schema, &predicate, case_sensitive)?;

        let data_file = create_data_file_float();

        let result = expression_evaluator.eval(&data_file)?;

        assert!(result);

        Ok(())
    }

    #[test]
    fn test_expr_less_than() -> Result<()> {
        let case_sensitive = true;
        let (partition_spec, schema) = create_partition_spec(PrimitiveType::Float)?;

        let predicate = Predicate::Binary(BinaryExpression::new(
            PredicateOperator::LessThan,
            Reference::new("a"),
            Datum::float(1.1),
        ))
        .bind(schema.clone(), case_sensitive)?;

        let expression_evaluator =
            create_expression_evaluator(partition_spec, &schema, &predicate, case_sensitive)?;

        let data_file = create_data_file_float();

        let result = expression_evaluator.eval(&data_file)?;

        assert!(result);

        Ok(())
    }

    #[test]
    fn test_expr_is_not_nan() -> Result<()> {
        let case_sensitive = true;
        let (partition_spec, schema) = create_partition_spec(PrimitiveType::Float)?;
        let predicate = Predicate::Unary(UnaryExpression::new(
            PredicateOperator::NotNan,
            Reference::new("a"),
        ))
        .bind(schema.clone(), case_sensitive)?;

        let expression_evaluator =
            create_expression_evaluator(partition_spec, &schema, &predicate, case_sensitive)?;

        let data_file = create_data_file_float();

        let result = expression_evaluator.eval(&data_file)?;

        assert!(result);

        Ok(())
    }

    #[test]
    fn test_expr_is_nan() -> Result<()> {
        let case_sensitive = true;
        let (partition_spec, schema) = create_partition_spec(PrimitiveType::Float)?;
        let predicate = Predicate::Unary(UnaryExpression::new(
            PredicateOperator::IsNan,
            Reference::new("a"),
        ))
        .bind(schema.clone(), case_sensitive)?;

        let expression_evaluator =
            create_expression_evaluator(partition_spec, &schema, &predicate, case_sensitive)?;

        let data_file = create_data_file_float();

        let result = expression_evaluator.eval(&data_file)?;

        assert!(!result);

        Ok(())
    }

    #[test]
    fn test_expr_is_not_null() -> Result<()> {
        let case_sensitive = true;
        let (partition_spec, schema) = create_partition_spec(PrimitiveType::Float)?;
        let predicate = Predicate::Unary(UnaryExpression::new(
            PredicateOperator::NotNull,
            Reference::new("a"),
        ))
        .bind(schema.clone(), case_sensitive)?;

        let expression_evaluator =
            create_expression_evaluator(partition_spec, &schema, &predicate, case_sensitive)?;

        let data_file = create_data_file_float();

        let result = expression_evaluator.eval(&data_file)?;

        assert!(result);

        Ok(())
    }

    #[test]
    fn test_expr_is_null() -> Result<()> {
        let case_sensitive = true;
        let (partition_spec, schema) = create_partition_spec(PrimitiveType::Float)?;
        let predicate = Predicate::Unary(UnaryExpression::new(
            PredicateOperator::IsNull,
            Reference::new("a"),
        ))
        .bind(schema.clone(), case_sensitive)?;

        let expression_evaluator =
            create_expression_evaluator(partition_spec, &schema, &predicate, case_sensitive)?;

        let data_file = create_data_file_float();

        let result = expression_evaluator.eval(&data_file)?;

        assert!(!result);

        Ok(())
    }

    #[test]
    fn test_expr_always_false() -> Result<()> {
        let case_sensitive = true;
        let (partition_spec, schema) = create_partition_spec(PrimitiveType::Float)?;
        let predicate = Predicate::AlwaysFalse.bind(schema.clone(), case_sensitive)?;

        let expression_evaluator =
            create_expression_evaluator(partition_spec, &schema, &predicate, case_sensitive)?;

        let data_file = create_data_file_float();

        let result = expression_evaluator.eval(&data_file)?;

        assert!(!result);

        Ok(())
    }

    #[test]
    fn test_expr_always_true() -> Result<()> {
        let case_sensitive = true;
        let (partition_spec, schema) = create_partition_spec(PrimitiveType::Float)?;
        let predicate = Predicate::AlwaysTrue.bind(schema.clone(), case_sensitive)?;

        let expression_evaluator =
            create_expression_evaluator(partition_spec, &schema, &predicate, case_sensitive)?;

        let data_file = create_data_file_float();

        let result = expression_evaluator.eval(&data_file)?;

        assert!(result);

        Ok(())
    }

    /// A [`DataFile`] whose single identity-partition value is NULL, for the nulls-first sweep.
    fn create_data_file_null_partition() -> DataFile {
        DataFile {
            partition: Struct::from_iter([None]),
            ..create_data_file_float()
        }
    }

    /// Evaluates one bound predicate over a NULL partition value through the full
    /// [`ExpressionEvaluator`] (identity spec, so the row predicate projects to itself).
    fn eval_on_null_partition(
        predicate: Predicate,
        r#type: PrimitiveType,
        expected: bool,
        op_label: &str,
    ) {
        let (partition_spec, schema) =
            create_partition_spec(r#type).expect("build the partition spec");
        let bound = predicate
            .bind(schema.clone(), true)
            .expect("bind the sweep predicate");
        let evaluator = create_expression_evaluator(partition_spec, &schema, &bound, true)
            .expect("build the expression evaluator");
        let result = evaluator
            .eval(&create_data_file_null_partition())
            .expect("evaluate against the null partition");
        assert_eq!(
            result, expected,
            "{op_label} over a NULL partition value must be {expected} (Java nulls-first)"
        );
    }

    /// Pins the FULL null-partition truth table against Java's nulls-first total order (audit
    /// BUG-011): iceberg-core `ManifestReader.evaluator()` applies `Evaluator$EvalVisitor` —
    /// whose comparisons run `Comparators.nullsFirst().thenComparing(naturalOrder)`,
    /// compare(null, lit) == -1 — to `entry.file().partition()`. `false` on a keep-op PRUNES
    /// the file: pre-fix, `<` and `<=` returned `Ok(false)` for a null partition value and
    /// silently LOST every row of such files.
    #[test]
    fn test_null_partition_value_truth_table_nulls_first() {
        let long_cases: Vec<(Predicate, bool, &str)> = vec![
            // Java `lt`: compare(null, 5) == -1 < 0 => TRUE (keep). THE BUG-011 headline.
            (Reference::new("a").less_than(Datum::long(5)), true, "a < 5"),
            (
                Reference::new("a").less_than_or_equal_to(Datum::long(5)),
                true,
                "a <= 5",
            ),
            (
                Reference::new("a").greater_than(Datum::long(5)),
                false,
                "a > 5",
            ),
            (
                Reference::new("a").greater_than_or_equal_to(Datum::long(5)),
                false,
                "a >= 5",
            ),
            (
                Reference::new("a").equal_to(Datum::long(5)),
                false,
                "a == 5",
            ),
            (
                Reference::new("a").not_equal_to(Datum::long(5)),
                true,
                "a != 5",
            ),
            // Java `in`: literalSet.contains(null) == false for every set impl.
            (
                Reference::new("a").is_in([Datum::long(5), Datum::long(99)]),
                false,
                "a IN (5, 99)",
            ),
            (
                Reference::new("a").is_not_in([Datum::long(5), Datum::long(99)]),
                true,
                "a NOT IN (5, 99)",
            ),
            (Reference::new("a").is_null(), true, "a IS NULL"),
            (Reference::new("a").is_not_null(), false, "a IS NOT NULL"),
        ];
        for (predicate, expected, label) in long_cases {
            eval_on_null_partition(predicate, PrimitiveType::Long, expected, label);
        }

        // Java `startsWith` null-guards to false (`ifnull 38`, offsets 11-12);
        // `notStartsWith` negates it.
        let string_cases: Vec<(Predicate, bool, &str)> = vec![
            (
                Reference::new("a").starts_with(Datum::string("prefix")),
                false,
                "a STARTS WITH 'prefix'",
            ),
            (
                Reference::new("a").not_starts_with(Datum::string("prefix")),
                true,
                "a NOT STARTS WITH 'prefix'",
            ),
        ];
        for (predicate, expected, label) in string_cases {
            eval_on_null_partition(predicate, PrimitiveType::String, expected, label);
        }

        // Java `NaNUtil.isNaN(null)` == false; `notNaN` negates it.
        eval_on_null_partition(
            Reference::new("a").is_nan(),
            PrimitiveType::Double,
            false,
            "isnan(a)",
        );
        eval_on_null_partition(
            Reference::new("a").is_not_nan(),
            PrimitiveType::Double,
            true,
            "NOT isnan(a)",
        );
    }
}
