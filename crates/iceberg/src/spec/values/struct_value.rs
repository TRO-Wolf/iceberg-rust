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

//! Struct type for Iceberg values

use std::hash::{Hash, Hasher};
use std::ops::Index;

use super::{Literal, PrimitiveLiteral};

/// The partition struct stores the tuple of partition values for each file.
/// Its type is derived from the partition fields of the partition spec used to write the manifest file.
/// In v2, the partition struct's field ids must match the ids from the partition spec.
#[derive(Debug, Clone)]
pub struct Struct {
    /// Vector to store the field values
    fields: Vec<Option<Literal>>,
}

impl Struct {
    /// Create a empty struct.
    pub fn empty() -> Self {
        Self { fields: Vec::new() }
    }

    /// Create a iterator to read the field in order of field_value.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = Option<&Literal>> {
        self.fields.iter().map(|field| field.as_ref())
    }

    /// returns true if the field at position `index` is null
    pub fn is_null_at_index(&self, index: usize) -> bool {
        self.fields[index].is_none()
    }

    /// Return fields in the struct.
    pub fn fields(&self) -> &[Option<Literal>] {
        &self.fields
    }
}

impl PartialEq for Struct {
    fn eq(&self, other: &Self) -> bool {
        fields_eq(&self.fields, &other.fields)
    }
}

impl Eq for Struct {}

impl Hash for Struct {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_fields(&self.fields, state);
    }
}

fn float_key(value: f32) -> u32 {
    if value.is_nan() {
        f32::NAN.to_bits()
    } else {
        value.to_bits()
    }
}

fn double_key(value: f64) -> u64 {
    if value.is_nan() {
        f64::NAN.to_bits()
    } else {
        value.to_bits()
    }
}

fn fields_eq(left: &[Option<Literal>], right: &[Option<Literal>]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| match (left, right) {
                (None, None) => true,
                (Some(left), Some(right)) => literal_eq(left, right),
                _ => false,
            })
}

fn literal_eq(left: &Literal, right: &Literal) -> bool {
    match (left, right) {
        (
            Literal::Primitive(PrimitiveLiteral::Float(left)),
            Literal::Primitive(PrimitiveLiteral::Float(right)),
        ) => float_key(left.0) == float_key(right.0),
        (
            Literal::Primitive(PrimitiveLiteral::Double(left)),
            Literal::Primitive(PrimitiveLiteral::Double(right)),
        ) => double_key(left.0) == double_key(right.0),
        (Literal::List(left), Literal::List(right)) => fields_eq(left, right),
        _ => left == right,
    }
}

fn hash_fields<H: Hasher>(fields: &[Option<Literal>], state: &mut H) {
    fields.len().hash(state);
    for field in fields {
        match field {
            None => 0u8.hash(state),
            Some(literal) => {
                1u8.hash(state);
                hash_literal(literal, state);
            }
        }
    }
}

fn hash_literal<H: Hasher>(literal: &Literal, state: &mut H) {
    match literal {
        Literal::Primitive(PrimitiveLiteral::Float(value)) => {
            0u8.hash(state);
            float_key(value.0).hash(state);
        }
        Literal::Primitive(PrimitiveLiteral::Double(value)) => {
            1u8.hash(state);
            double_key(value.0).hash(state);
        }
        Literal::List(items) => {
            2u8.hash(state);
            hash_fields(items, state);
        }
        other => {
            3u8.hash(state);
            other.hash(state);
        }
    }
}

impl Index<usize> for Struct {
    type Output = Option<Literal>;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.fields[idx]
    }
}

impl IntoIterator for Struct {
    type Item = Option<Literal>;

    type IntoIter = std::vec::IntoIter<Option<Literal>>;

    fn into_iter(self) -> Self::IntoIter {
        self.fields.into_iter()
    }
}

impl FromIterator<Option<Literal>> for Struct {
    fn from_iter<I: IntoIterator<Item = Option<Literal>>>(iter: I) -> Self {
        Struct {
            fields: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::spec::PrimitiveLiteral;

    fn one(value: Literal) -> Struct {
        Struct::from_iter([Some(value)])
    }

    fn distinct(keys: &[Struct]) -> usize {
        keys.iter().collect::<HashSet<_>>().len()
    }

    #[test]
    fn signed_zeros_are_two_partition_keys() {
        let doubles = [one(Literal::double(-0.0)), one(Literal::double(0.0))];
        assert_ne!(doubles[0], doubles[1]);
        assert_eq!(distinct(&doubles), 2);
        let floats = [one(Literal::float(-0.0)), one(Literal::float(0.0))];
        assert_ne!(floats[0], floats[1]);
        assert_eq!(distinct(&floats), 2);
        assert_eq!(one(Literal::double(-0.0)), one(Literal::double(-0.0)));
        assert_eq!(one(Literal::float(0.0)), one(Literal::float(0.0)));
    }

    #[test]
    fn every_nan_is_one_partition_key() {
        let doubles = [
            one(Literal::double(f64::NAN)),
            one(Literal::double(-f64::NAN)),
            one(Literal::double(f64::from_bits(0x7ff0_0000_0000_0001))),
            one(Literal::double(f64::from_bits(0xfff8_dead_beef_0001))),
        ];
        assert!(doubles.iter().all(|key| key == &doubles[0]));
        assert_eq!(distinct(&doubles), 1);
        let floats = [
            one(Literal::float(f32::NAN)),
            one(Literal::float(-f32::NAN)),
            one(Literal::float(f32::from_bits(0x7f80_0001))),
        ];
        assert!(floats.iter().all(|key| key == &floats[0]));
        assert_eq!(distinct(&floats), 1);
        assert_ne!(one(Literal::double(f64::NAN)), one(Literal::double(0.0)));
    }

    #[test]
    fn other_leaves_keep_their_equality() {
        let keys = [
            Struct::from_iter([Some(Literal::long(7)), None, Some(Literal::string("a"))]),
            Struct::from_iter([Some(Literal::long(7)), None, Some(Literal::string("a"))]),
            Struct::from_iter([Some(Literal::long(7)), Some(Literal::int(0)), None]),
            Struct::from_iter([Some(Literal::double(1.5))]),
            Struct::from_iter([Some(Literal::double(1.5))]),
            Struct::from_iter([Some(Literal::float(1.5))]),
            Struct::from_iter([Some(Literal::double(1.5)), None]),
            Struct::empty(),
        ];
        assert_eq!(keys[0], keys[1]);
        assert_ne!(keys[0], keys[2]);
        assert_eq!(keys[3], keys[4]);
        assert_ne!(keys[3], keys[5]);
        assert_ne!(keys[3], keys[6]);
        assert_eq!(distinct(&keys), 6);
        assert_ne!(
            one(Literal::Primitive(PrimitiveLiteral::Double(
                ordered_float::OrderedFloat(1.0)
            ))),
            one(Literal::Primitive(PrimitiveLiteral::Long(1)))
        );
    }

    #[test]
    fn nested_float_leaves_follow_the_same_rule() {
        let nested = |value: f64| one(Literal::Struct(one(Literal::double(value))));
        assert_ne!(nested(-0.0), nested(0.0));
        assert_eq!(nested(f64::NAN), nested(-f64::NAN));
        assert_eq!(distinct(&[nested(-0.0), nested(0.0), nested(-0.0)]), 2);
        let list = |value: f32| one(Literal::List(vec![Some(Literal::float(value)), None]));
        assert_ne!(list(-0.0), list(0.0));
        assert_eq!(list(f32::NAN), list(-f32::NAN));
        assert_eq!(distinct(&[list(-0.0), list(0.0), list(0.0)]), 2);
    }
}
