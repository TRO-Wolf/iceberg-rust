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

use crate::error::Result;
use crate::expr::{Bind, BoundPredicate, MAX_PREDICATE_DEPTH, Predicate, Reference};
use crate::spec::SchemaRef;

impl Predicate {
    #[allow(missing_docs)]
    pub fn drop_unbindable_terms(&self, schema: &SchemaRef, case_sensitive: bool) -> Predicate {
        self.drop_unbindable_terms_at(schema, case_sensitive, 0)
    }

    #[allow(missing_docs)]
    pub fn bind_pruning(&self, schema: SchemaRef, case_sensitive: bool) -> Result<BoundPredicate> {
        self.drop_unbindable_terms(&schema, case_sensitive)
            .bind(schema, case_sensitive)
    }

    fn drop_unbindable_terms_at(
        &self,
        schema: &SchemaRef,
        case_sensitive: bool,
        depth: usize,
    ) -> Predicate {
        if depth > MAX_PREDICATE_DEPTH {
            return Predicate::AlwaysTrue;
        }
        match self {
            Predicate::And(expr) => {
                let [left, right] = expr.inputs();
                left.drop_unbindable_terms_at(schema, case_sensitive, depth + 1)
                    .and(right.drop_unbindable_terms_at(schema, case_sensitive, depth + 1))
            }
            Predicate::Or(expr) => {
                let [left, right] = expr.inputs();
                left.drop_unbindable_terms_at(schema, case_sensitive, depth + 1)
                    .or(right.drop_unbindable_terms_at(schema, case_sensitive, depth + 1))
            }
            Predicate::Not(expr) => {
                let [inner] = expr.inputs();
                if inner.contains_unbindable_term(schema, case_sensitive, 0) {
                    Predicate::AlwaysTrue
                } else {
                    self.clone()
                }
            }
            leaf => {
                if leaf_is_unbindable(leaf, schema, case_sensitive) {
                    Predicate::AlwaysTrue
                } else {
                    leaf.clone()
                }
            }
        }
    }

    fn contains_unbindable_term(
        &self,
        schema: &SchemaRef,
        case_sensitive: bool,
        depth: usize,
    ) -> bool {
        if depth > MAX_PREDICATE_DEPTH {
            return false;
        }
        match self {
            Predicate::And(expr) | Predicate::Or(expr) => {
                let [left, right] = expr.inputs();
                left.contains_unbindable_term(schema, case_sensitive, depth + 1)
                    || right.contains_unbindable_term(schema, case_sensitive, depth + 1)
            }
            Predicate::Not(expr) => {
                let [inner] = expr.inputs();
                inner.contains_unbindable_term(schema, case_sensitive, depth + 1)
            }
            leaf => leaf_is_unbindable(leaf, schema, case_sensitive),
        }
    }
}

fn leaf_is_unbindable(leaf: &Predicate, schema: &SchemaRef, case_sensitive: bool) -> bool {
    match leaf {
        Predicate::Unary(expr) => term_is_unbindable(expr.term(), schema, case_sensitive),
        Predicate::Binary(expr) => term_is_unbindable(expr.term(), schema, case_sensitive),
        Predicate::Set(expr) => term_is_unbindable(expr.term(), schema, case_sensitive),
        _ => false,
    }
}

fn term_is_unbindable(term: &Reference, schema: &SchemaRef, case_sensitive: bool) -> bool {
    let field = if case_sensitive {
        schema.field_by_name(term.name())
    } else {
        schema.field_by_name_case_insensitive(term.name())
    };
    match field {
        Some(field) => schema.accessor_by_field_id(field.id).is_none(),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Not;
    use std::sync::Arc;

    use crate::expr::{Predicate, Reference};
    use crate::spec::{Datum, ListType, NestedField, PrimitiveType, Schema, SchemaRef, Type};

    fn schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(
                        2,
                        "xs",
                        Type::List(ListType::new(
                            NestedField::list_element(
                                3,
                                Type::Primitive(PrimitiveType::Int),
                                false,
                            )
                            .into(),
                        )),
                    )
                    .into(),
                ])
                .build()
                .expect("schema"),
        )
    }

    #[test]
    fn and_keeps_the_bindable_conjunct() {
        let predicate = Reference::new("id")
            .greater_than(Datum::int(1))
            .and(Reference::new("xs").is_null());
        assert_eq!(
            predicate.drop_unbindable_terms(&schema(), true),
            Reference::new("id").greater_than(Datum::int(1))
        );
    }

    #[test]
    fn or_widens_to_always_true() {
        let predicate = Reference::new("xs")
            .is_null()
            .or(Reference::new("id").equal_to(Datum::int(1)));
        assert_eq!(
            predicate.drop_unbindable_terms(&schema(), true),
            Predicate::AlwaysTrue
        );
    }

    #[test]
    fn an_or_branch_inside_an_and_widens_to_the_other_conjunct() {
        let predicate = Reference::new("id").equal_to(Datum::int(2)).and(
            Reference::new("xs")
                .is_null()
                .or(Reference::new("id").greater_than(Datum::int(1))),
        );
        assert_eq!(
            predicate.drop_unbindable_terms(&schema(), true),
            Reference::new("id").equal_to(Datum::int(2))
        );
    }

    #[test]
    fn not_over_an_unbindable_term_widens() {
        let predicate = Reference::new("xs").is_null().not();
        assert_eq!(
            predicate.drop_unbindable_terms(&schema(), true),
            Predicate::AlwaysTrue
        );
    }

    #[test]
    fn not_over_a_compound_with_an_unbindable_term_widens() {
        let predicate = Reference::new("id")
            .greater_than(Datum::int(1))
            .and(Reference::new("xs").is_null())
            .not();
        assert_eq!(
            predicate.drop_unbindable_terms(&schema(), true),
            Predicate::AlwaysTrue
        );
    }

    #[test]
    fn a_missing_field_stays_a_bind_error() {
        let predicate = Reference::new("nope")
            .equal_to(Datum::int(1))
            .and(Reference::new("id").equal_to(Datum::int(2)));
        assert_eq!(predicate.drop_unbindable_terms(&schema(), true), predicate);
        assert!(predicate.bind_pruning(schema(), true).is_err());
    }

    #[test]
    fn a_wrong_cased_name_stays_a_bind_error() {
        let predicate = Reference::new("ID").equal_to(Datum::int(1));
        assert_eq!(predicate.drop_unbindable_terms(&schema(), true), predicate);
        assert!(predicate.bind_pruning(schema(), true).is_err());
        assert!(predicate.bind_pruning(schema(), false).is_ok());
    }

    #[test]
    fn a_bindable_predicate_is_unchanged() {
        let predicate = Reference::new("id")
            .greater_than(Datum::int(1))
            .and(Reference::new("id").less_than(Datum::int(4)));
        assert_eq!(predicate.drop_unbindable_terms(&schema(), true), predicate);
    }

    #[test]
    fn bind_pruning_binds_the_sanitized_predicate() {
        let predicate = Reference::new("id")
            .greater_than(Datum::int(1))
            .and(Reference::new("xs").is_null());
        assert!(predicate.bind_pruning(schema(), true).is_ok());
        assert!(
            Reference::new("xs")
                .is_null()
                .bind_pruning(schema(), true)
                .is_ok()
        );
    }
}
