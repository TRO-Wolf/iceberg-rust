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

use std::sync::Arc;

use serde_derive::{Deserialize, Serialize};

use crate::spec::{Datum, Literal, PrimitiveType, Struct};
use crate::{Error, ErrorKind, Result};

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct StructAccessor {
    position: usize,
    r#type: PrimitiveType,
    inner: Option<Box<StructAccessor>>,
}

pub(crate) type StructAccessorRef = Arc<StructAccessor>;

impl StructAccessor {
    pub(crate) fn new(position: usize, r#type: PrimitiveType) -> Self {
        StructAccessor {
            position,
            r#type,
            inner: None,
        }
    }

    pub(crate) fn wrap(position: usize, inner: Box<StructAccessor>) -> Self {
        StructAccessor {
            position,
            r#type: inner.r#type().clone(),
            inner: Some(inner),
        }
    }

    pub(crate) fn position(&self) -> usize {
        self.position
    }

    pub(crate) fn r#type(&self) -> &PrimitiveType {
        &self.r#type
    }

    pub(crate) fn get<'a>(&'a self, container: &'a Struct) -> Result<Option<Datum>> {
        let value = container.fields().get(self.position).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot access field at position {} from struct with {} fields",
                    self.position,
                    container.fields().len()
                ),
            )
        })?;

        match &self.inner {
            None => match value {
                None => Ok(None),
                // PrimitiveLiteral records the physical representation, not a separate semantic
                // type tag. Compatibility therefore intentionally accepts representation-sharing
                // families such as int/date, long/time/timestamps, and binary/fixed.
                Some(Literal::Primitive(literal)) if self.r#type().compatible(literal) => {
                    Ok(Some(Datum::new(self.r#type().clone(), literal.clone())))
                }
                Some(Literal::Primitive(literal)) => Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Literal {literal:?} at position {} is not compatible with accessor type {}",
                        self.position,
                        self.r#type()
                    ),
                )),
                Some(_) => Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Expected Literal to be Primitive",
                )),
            },
            Some(inner) => match value {
                None => Ok(None),
                Some(Literal::Struct(wrapped)) => inner.get(wrapped),
                Some(_) => Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Nested accessor should only be wrapping a Struct",
                )),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ErrorKind;
    use crate::expr::accessor::StructAccessor;
    use crate::spec::{Datum, Literal, PrimitiveType, Struct};

    #[test]
    fn test_single_level_accessor() {
        let accessor = StructAccessor::new(1, PrimitiveType::Boolean);

        assert_eq!(accessor.r#type(), &PrimitiveType::Boolean);
        assert_eq!(accessor.position(), 1);

        let test_struct =
            Struct::from_iter(vec![Some(Literal::bool(false)), Some(Literal::bool(true))]);

        assert_eq!(accessor.get(&test_struct).unwrap(), Some(Datum::bool(true)));
    }

    #[test]
    fn test_single_level_accessor_null() {
        let accessor = StructAccessor::new(1, PrimitiveType::Boolean);

        assert_eq!(accessor.r#type(), &PrimitiveType::Boolean);
        assert_eq!(accessor.position(), 1);

        let test_struct = Struct::from_iter(vec![Some(Literal::bool(false)), None]);

        assert_eq!(accessor.get(&test_struct).unwrap(), None);
    }

    #[test]
    fn test_nested_accessor() {
        let nested_accessor = StructAccessor::new(1, PrimitiveType::Boolean);
        let accessor = StructAccessor::wrap(2, Box::new(nested_accessor));

        assert_eq!(accessor.r#type(), &PrimitiveType::Boolean);
        //assert_eq!(accessor.position(), 1);

        let nested_test_struct =
            Struct::from_iter(vec![Some(Literal::bool(false)), Some(Literal::bool(true))]);

        let test_struct = Struct::from_iter(vec![
            Some(Literal::bool(false)),
            Some(Literal::bool(false)),
            Some(Literal::Struct(nested_test_struct)),
        ]);

        assert_eq!(accessor.get(&test_struct).unwrap(), Some(Datum::bool(true)));
    }

    #[test]
    fn test_nested_accessor_null() {
        let nested_accessor = StructAccessor::new(0, PrimitiveType::Boolean);
        let accessor = StructAccessor::wrap(2, Box::new(nested_accessor));

        assert_eq!(accessor.r#type(), &PrimitiveType::Boolean);
        //assert_eq!(accessor.position(), 1);

        let nested_test_struct = Struct::from_iter(vec![None, Some(Literal::bool(true))]);

        let test_struct = Struct::from_iter(vec![
            Some(Literal::bool(false)),
            Some(Literal::bool(false)),
            Some(Literal::Struct(nested_test_struct)),
        ]);

        assert_eq!(accessor.get(&test_struct).unwrap(), None);
    }

    #[test]
    fn test_single_level_accessor_rejects_short_struct() {
        let accessor = StructAccessor::new(1, PrimitiveType::Boolean);
        let test_struct = Struct::from_iter([Some(Literal::bool(false))]);

        let error = accessor
            .get(&test_struct)
            .expect_err("a short struct must return a typed error");

        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert_eq!(
            error.message(),
            "Cannot access field at position 1 from struct with 1 fields"
        );
    }

    #[test]
    fn test_nested_accessor_rejects_short_inner_struct() {
        let accessor =
            StructAccessor::wrap(0, Box::new(StructAccessor::new(0, PrimitiveType::Boolean)));
        let test_struct = Struct::from_iter([Some(Literal::Struct(Struct::empty()))]);

        let error = accessor
            .get(&test_struct)
            .expect_err("a short nested struct must return a typed error");

        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert_eq!(
            error.message(),
            "Cannot access field at position 0 from struct with 0 fields"
        );
    }

    #[test]
    fn test_accessor_rejects_out_of_bounds_positions_without_panicking() {
        let cases = [
            (0, Struct::empty(), 0),
            (1, Struct::from_iter([Some(Literal::bool(false))]), 1),
            (
                usize::MAX,
                Struct::from_iter([Some(Literal::bool(false))]),
                1,
            ),
        ];

        for (position, test_struct, field_count) in cases {
            let error = StructAccessor::new(position, PrimitiveType::Boolean)
                .get(&test_struct)
                .expect_err("an out-of-bounds position must return a typed error");

            assert_eq!(error.kind(), ErrorKind::DataInvalid);
            assert_eq!(
                error.message(),
                format!(
                    "Cannot access field at position {position} from struct with {field_count} fields"
                )
            );
        }
    }

    #[test]
    fn test_accessor_rejects_primitive_type_mismatch_at_outer_and_nested_leaves() {
        let outer_error = StructAccessor::new(0, PrimitiveType::Boolean)
            .get(&Struct::from_iter([Some(Literal::int(7))]))
            .expect_err("an outer leaf with the wrong primitive kind must fail");
        assert_eq!(outer_error.kind(), ErrorKind::DataInvalid);
        assert!(outer_error.message().contains("accessor type boolean"));

        let nested_accessor =
            StructAccessor::wrap(0, Box::new(StructAccessor::new(0, PrimitiveType::Boolean)));
        let nested_error = nested_accessor
            .get(&Struct::from_iter([Some(Literal::Struct(
                Struct::from_iter([Some(Literal::int(7))]),
            ))]))
            .expect_err("a nested leaf with the wrong primitive kind must fail");
        assert_eq!(nested_error.kind(), ErrorKind::DataInvalid);
        assert!(nested_error.message().contains("accessor type boolean"));
    }

    #[test]
    fn test_accessor_accepts_representation_compatible_primitive_families() {
        // Literal constructors for these semantic types collapse to the same PrimitiveLiteral
        // representation. The accessor can validate that representation, but cannot recover the
        // constructor's semantic identity from the literal.
        let cases = [
            (PrimitiveType::Int, Literal::date(7)),
            (PrimitiveType::Date, Literal::int(7)),
            (PrimitiveType::Long, Literal::time(7)),
            (PrimitiveType::Time, Literal::timestamp(7)),
            (PrimitiveType::Timestamp, Literal::timestamptz(7)),
            (PrimitiveType::Timestamptz, Literal::long(7)),
            (PrimitiveType::Binary, Literal::fixed([1, 2])),
            (PrimitiveType::Fixed(2), Literal::binary([1, 2])),
            (
                PrimitiveType::Decimal {
                    precision: 9,
                    scale: 2,
                },
                Literal::decimal(12345),
            ),
        ];

        for (accessor_type, literal) in cases {
            let value = StructAccessor::new(0, accessor_type.clone())
                .get(&Struct::from_iter([Some(literal)]))
                .expect("a representation-compatible primitive must be accepted")
                .expect("the present primitive must produce a datum");

            assert_eq!(value.data_type(), &accessor_type);
            assert!(accessor_type.compatible(value.literal()));
        }
    }

    #[test]
    fn test_accessor_rejects_representation_incompatible_primitives() {
        let cases = [
            (PrimitiveType::Int, Literal::long(7)),
            (PrimitiveType::Long, Literal::int(7)),
            (
                PrimitiveType::Decimal {
                    precision: 9,
                    scale: 2,
                },
                Literal::long(12345),
            ),
            (PrimitiveType::Binary, Literal::string("bytes")),
            (PrimitiveType::String, Literal::binary([1, 2])),
        ];

        for (accessor_type, literal) in cases {
            let error = StructAccessor::new(0, accessor_type.clone())
                .get(&Struct::from_iter([Some(literal)]))
                .expect_err("a representation-incompatible primitive must fail");

            assert_eq!(error.kind(), ErrorKind::DataInvalid);
            assert!(
                error
                    .message()
                    .contains(&format!("accessor type {accessor_type}"))
            );
        }
    }

    #[test]
    fn test_accessor_distinguishes_nested_null_parent_from_present_wrong_shape() {
        let primitive_leaf_error = StructAccessor::new(0, PrimitiveType::Boolean)
            .get(&Struct::from_iter([Some(Literal::Struct(Struct::empty()))]))
            .expect_err("a leaf accessor must reject a struct literal");
        assert_eq!(primitive_leaf_error.kind(), ErrorKind::DataInvalid);

        let nested_accessor =
            StructAccessor::wrap(0, Box::new(StructAccessor::new(0, PrimitiveType::Boolean)));
        let nested_shape_error = nested_accessor
            .get(&Struct::from_iter([Some(Literal::bool(true))]))
            .expect_err("a nested accessor must reject a primitive outer literal");
        assert_eq!(nested_shape_error.kind(), ErrorKind::DataInvalid);

        assert_eq!(
            nested_accessor
                .get(&Struct::from_iter([None]))
                .expect("an optional null parent must propagate through a nested accessor"),
            None
        );

        assert_eq!(
            StructAccessor::new(0, PrimitiveType::Boolean)
                .get(&Struct::from_iter([None]))
                .expect("a null leaf is valid"),
            None
        );
    }
}
