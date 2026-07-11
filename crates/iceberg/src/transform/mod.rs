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

//! Transform function used to compute partition values.

use std::fmt::Debug;

use arrow_array::ArrayRef;

use crate::spec::{Datum, Transform};
use crate::{Error, ErrorKind, Result};

mod bucket;
mod identity;
mod temporal;
mod truncate;
mod void;

/// TransformFunction is a trait that defines the interface for all transform functions.
pub trait TransformFunction: Send + Sync + Debug {
    /// transform will take an input array and transform it into a new array.
    /// The implementation of this function will need to check and downcast the input to specific
    /// type.
    fn transform(&self, input: ArrayRef) -> Result<ArrayRef>;
    /// transform_literal will take an input literal and transform it into a new literal.
    fn transform_literal(&self, input: &Datum) -> Result<Option<Datum>>;
    /// A thin wrapper around `transform_literal`
    /// to return an error even when it's `None`.
    fn transform_literal_result(&self, input: &Datum) -> Result<Datum> {
        self.transform_literal(input)?.ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                format!("Returns 'None' for literal {input}"),
            )
        })
    }
}

/// BoxedTransformFunction is a boxed trait object of TransformFunction.
pub type BoxedTransformFunction = Box<dyn TransformFunction>;

/// create_transform_function creates a boxed trait object of TransformFunction from a Transform.
pub fn create_transform_function(transform: &Transform) -> Result<BoxedTransformFunction> {
    match transform {
        Transform::Identity => Ok(Box::new(identity::Identity {})),
        Transform::Void => Ok(Box::new(void::Void {})),
        Transform::Year => Ok(Box::new(temporal::Year {})),
        Transform::Month => Ok(Box::new(temporal::Month {})),
        Transform::Day => Ok(Box::new(temporal::Day {})),
        Transform::Hour => Ok(Box::new(temporal::Hour {})),
        Transform::Bucket(mod_n) => Ok(Box::new(bucket::Bucket::new(*mod_n)?)),
        Transform::Truncate(width) => Ok(Box::new(truncate::Truncate::new(*width)?)),
        Transform::Unknown => Err(crate::error::Error::new(
            crate::ErrorKind::FeatureUnsupported,
            "Transform Unknown is not implemented",
        )),
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;
    use std::sync::Arc;

    use crate::Result;
    use crate::expr::accessor::StructAccessor;
    use crate::expr::{
        BinaryExpression, BoundPredicate, BoundReference, PredicateOperator, SetExpression,
    };
    use crate::spec::{Datum, NestedField, NestedFieldRef, PrimitiveType, Transform, Type};

    /// A utitily struct, test fixture
    /// used for testing the projection on `Transform`
    pub(crate) struct TestProjectionFixture {
        transform: Transform,
        name: String,
        field: NestedFieldRef,
    }

    impl TestProjectionFixture {
        pub(crate) fn new(
            transform: Transform,
            name: impl Into<String>,
            field: NestedField,
        ) -> Self {
            TestProjectionFixture {
                transform,
                name: name.into(),
                field: Arc::new(field),
            }
        }
        pub(crate) fn binary_predicate(
            &self,
            op: PredicateOperator,
            literal: Datum,
        ) -> BoundPredicate {
            BoundPredicate::Binary(BinaryExpression::new(
                op,
                BoundReference::new(
                    self.name.clone(),
                    self.field.clone(),
                    Arc::new(StructAccessor::new(1, PrimitiveType::Boolean)),
                ),
                literal,
            ))
        }
        pub(crate) fn set_predicate(
            &self,
            op: PredicateOperator,
            literals: Vec<Datum>,
        ) -> BoundPredicate {
            BoundPredicate::Set(SetExpression::new(
                op,
                BoundReference::new(
                    self.name.clone(),
                    self.field.clone(),
                    Arc::new(StructAccessor::new(1, PrimitiveType::Boolean)),
                ),
                HashSet::from_iter(literals),
            ))
        }
        pub(crate) fn assert_projection(
            &self,
            predicate: &BoundPredicate,
            expected: Option<&str>,
        ) -> Result<()> {
            let result = self.transform.project(&self.name, predicate)?;
            match expected {
                Some(exp) => assert_eq!(format!("{}", result.unwrap()), exp),
                None => assert!(result.is_none()),
            }
            Ok(())
        }
    }

    /// A utility struct, test fixture
    /// used for testing the transform on `Transform`
    pub(crate) struct TestTransformFixture {
        pub display: String,
        pub json: String,
        pub dedup_name: String,
        pub preserves_order: bool,
        pub satisfies_order_of: Vec<(Transform, bool)>,
        pub trans_types: Vec<(Type, Option<Type>)>,
    }

    // RISK: `Transform::Bucket(0)` / `Transform::Truncate(0)` can be constructed directly (the
    // enum payload is public and cannot be guarded), and pre-fix the apply path panicked with a
    // divide/modulo-by-zero — a hostile or corrupt table spec crashed the process. The apply door
    // must return a typed error instead. In Java the instance cannot exist at all
    // (Bucket.java:41-42 / Truncate.java:42, 1.10.0).
    #[test]
    fn test_create_transform_function_rejects_zero_parameter_transforms() {
        for transform in [Transform::Bucket(0), Transform::Truncate(0)] {
            let error = super::create_transform_function(&transform)
                .expect_err("zero-parameter transform must not yield a transform function");
            assert_eq!(error.kind(), crate::ErrorKind::DataInvalid, "{transform}");
        }
    }

    // RISK: pre-fix `bucket_n` cast `mod_n as i32`, so 2147483648 (= i32::MAX + 1) wrapped to
    // i32::MIN and `(v & i32::MAX) % i32::MIN` returned the masked hash itself — silently WRONG
    // bucket values (partition-routing divergence vs Java, where such a count is unrepresentable).
    #[test]
    fn test_create_transform_function_rejects_parameters_above_java_int_max() {
        for transform in [
            Transform::Bucket(2147483648),
            Transform::Truncate(2147483648),
        ] {
            let error = super::create_transform_function(&transform)
                .expect_err("parameter above the Java int maximum must be rejected");
            assert_eq!(error.kind(), crate::ErrorKind::DataInvalid, "{transform}");
        }
    }

    // RISK (end-to-end, the original panic site): projecting a predicate through bucket[0]
    // reached `% 0` via transform_literal. It must surface as Err, never a panic.
    #[test]
    fn test_project_with_zero_bucket_returns_error_instead_of_panicking() {
        let fixture = TestProjectionFixture::new(
            Transform::Bucket(0),
            "name",
            NestedField::required(1, "value", Type::Primitive(PrimitiveType::Int)),
        );
        let error = Transform::Bucket(0)
            .project(
                "name",
                &fixture.binary_predicate(PredicateOperator::Eq, Datum::int(10)),
            )
            .expect_err("projection through bucket[0] must fail, not panic");
        assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
    }

    impl TestTransformFixture {
        #[track_caller]
        pub(crate) fn assert_transform(&self, trans: Transform) {
            assert_eq!(self.display, format!("{trans}"));
            assert_eq!(self.json, serde_json::to_string(&trans).unwrap());
            assert_eq!(trans, serde_json::from_str(self.json.as_str()).unwrap());
            assert_eq!(self.dedup_name, trans.dedup_name());
            assert_eq!(self.preserves_order, trans.preserves_order());

            for (other_trans, satisfies_order_of) in &self.satisfies_order_of {
                assert_eq!(
                    satisfies_order_of,
                    &trans.satisfies_order_of(other_trans),
                    "Failed to check satisfies order {trans}, {other_trans}, {satisfies_order_of}"
                );
            }

            for (i, (input_type, result_type)) in self.trans_types.iter().enumerate() {
                let actual = trans.result_type(input_type).ok();
                assert_eq!(
                    result_type, &actual,
                    "type mismatch at index {i}, input: {input_type}, expected: {result_type:?}, actual: {actual:?}"
                );
            }
        }
    }
}
