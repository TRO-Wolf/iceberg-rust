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

use super::*;

/// A post order schema visitor.
///
/// For order of methods called, please refer to [`visit_schema`].
pub trait SchemaVisitor {
    /// Return type of this visitor.
    type T;

    /// Called before struct field.
    fn before_struct_field(&mut self, _field: &NestedFieldRef) -> Result<()> {
        Ok(())
    }
    /// Called after struct field.
    fn after_struct_field(&mut self, _field: &NestedFieldRef) -> Result<()> {
        Ok(())
    }
    /// Called before list field.
    fn before_list_element(&mut self, _field: &NestedFieldRef) -> Result<()> {
        Ok(())
    }
    /// Called after list field.
    fn after_list_element(&mut self, _field: &NestedFieldRef) -> Result<()> {
        Ok(())
    }
    /// Called before map key field.
    fn before_map_key(&mut self, _field: &NestedFieldRef) -> Result<()> {
        Ok(())
    }
    /// Called after map key field.
    fn after_map_key(&mut self, _field: &NestedFieldRef) -> Result<()> {
        Ok(())
    }
    /// Called before map value field.
    fn before_map_value(&mut self, _field: &NestedFieldRef) -> Result<()> {
        Ok(())
    }
    /// Called after map value field.
    fn after_map_value(&mut self, _field: &NestedFieldRef) -> Result<()> {
        Ok(())
    }

    /// Called after schema's type visited.
    fn schema(&mut self, schema: &Schema, value: Self::T) -> Result<Self::T>;
    /// Called after struct's field type visited.
    fn field(&mut self, field: &NestedFieldRef, value: Self::T) -> Result<Self::T>;
    /// Called after struct's fields visited.
    fn r#struct(&mut self, r#struct: &StructType, results: Vec<Self::T>) -> Result<Self::T>;
    /// Called after list fields visited.
    fn list(&mut self, list: &ListType, value: Self::T) -> Result<Self::T>;
    /// Called after map's key and value fields visited.
    fn map(&mut self, map: &MapType, key_value: Self::T, value: Self::T) -> Result<Self::T>;
    /// Called when see a primitive type.
    fn primitive(&mut self, p: &PrimitiveType) -> Result<Self::T>;
    /// Called when a `variant` type is visited.
    ///
    /// The default mirrors Java 1.10.0 `TypeUtil.SchemaVisitor.variant(VariantType)`, which
    /// throws `UnsupportedOperationException("Unsupported type: variant")` — a visitor that does
    /// not opt in fails loudly instead of silently mishandling the type. Structural visitors
    /// (indexing, pruning, id reassignment) override this as a leaf, exactly like their Java
    /// counterparts (`IndexByName`/`IndexById`/`IndexParents`/`PruneColumns` all override it).
    fn variant(&mut self) -> Result<Self::T> {
        Err(Error::new(
            ErrorKind::FeatureUnsupported,
            "Unsupported type: variant",
        ))
    }
}

/// Maximum schema-type nesting depth either `visit_type` walk will descend.
///
/// A schema can come from a partner/attacker-influenced metadata file; without a bound a
/// deeply-nested type (struct/list/map-within-…) would overflow the thread stack on this
/// recursive post-order walk. Mirrors the variant parser's
/// [`crate::variant::MAX_NESTING_DEPTH`] (`128`) — far above any real Iceberg schema's nesting.
const MAX_SCHEMA_NESTING_DEPTH: usize = 128;

/// Visiting a type in post order.
pub(crate) fn visit_type<V: SchemaVisitor>(r#type: &Type, visitor: &mut V) -> Result<V::T> {
    visit_type_at_depth(r#type, visitor, 0)
}

/// Depth-bounded body of [`visit_type`]. `depth` is the current nesting level (root at `0`); each
/// nested element/key/value/field recurses at `depth + 1`, and exceeding
/// [`MAX_SCHEMA_NESTING_DEPTH`] returns a typed error instead of overflowing the stack.
fn visit_type_at_depth<V: SchemaVisitor>(
    r#type: &Type,
    visitor: &mut V,
    depth: usize,
) -> Result<V::T> {
    if depth > MAX_SCHEMA_NESTING_DEPTH {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Schema type nesting exceeds maximum depth {MAX_SCHEMA_NESTING_DEPTH}"),
        ));
    }
    match r#type {
        Type::Primitive(p) => visitor.primitive(p),
        Type::Variant => visitor.variant(),
        Type::List(list) => {
            visitor.before_list_element(&list.element_field)?;
            let value = visit_type_at_depth(&list.element_field.field_type, visitor, depth + 1)?;
            visitor.after_list_element(&list.element_field)?;
            visitor.list(list, value)
        }
        Type::Map(map) => {
            let key_result = {
                visitor.before_map_key(&map.key_field)?;
                let ret = visit_type_at_depth(&map.key_field.field_type, visitor, depth + 1)?;
                visitor.after_map_key(&map.key_field)?;
                ret
            };

            let value_result = {
                visitor.before_map_value(&map.value_field)?;
                let ret = visit_type_at_depth(&map.value_field.field_type, visitor, depth + 1)?;
                visitor.after_map_value(&map.value_field)?;
                ret
            };

            visitor.map(map, key_result, value_result)
        }
        Type::Struct(s) => visit_struct_at_depth(s, visitor, depth),
    }
}

/// Visit struct type in post order.
pub fn visit_struct<V: SchemaVisitor>(s: &StructType, visitor: &mut V) -> Result<V::T> {
    visit_struct_at_depth(s, visitor, 0)
}

/// Depth-bounded body of [`visit_struct`]; see [`visit_type_at_depth`].
fn visit_struct_at_depth<V: SchemaVisitor>(
    s: &StructType,
    visitor: &mut V,
    depth: usize,
) -> Result<V::T> {
    if depth > MAX_SCHEMA_NESTING_DEPTH {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Schema type nesting exceeds maximum depth {MAX_SCHEMA_NESTING_DEPTH}"),
        ));
    }
    let mut results = Vec::with_capacity(s.fields().len());
    for field in s.fields() {
        visitor.before_struct_field(field)?;
        let result = visit_type_at_depth(&field.field_type, visitor, depth + 1)?;
        visitor.after_struct_field(field)?;
        let result = visitor.field(field, result)?;
        results.push(result);
    }

    visitor.r#struct(s, results)
}

/// Visit schema in post order.
pub fn visit_schema<V: SchemaVisitor>(schema: &Schema, visitor: &mut V) -> Result<V::T> {
    let result = visit_struct(&schema.r#struct, visitor)?;
    visitor.schema(schema, result)
}

/// A post order schema visitor with partner.
///
/// For order of methods called, please refer to [`visit_schema_with_partner`].
pub trait SchemaWithPartnerVisitor<P> {
    /// Return type of this visitor.
    type T;

    /// Called before struct field.
    fn before_struct_field(&mut self, _field: &NestedFieldRef, _partner: &P) -> Result<()> {
        Ok(())
    }
    /// Called after struct field.
    fn after_struct_field(&mut self, _field: &NestedFieldRef, _partner: &P) -> Result<()> {
        Ok(())
    }
    /// Called before list field.
    fn before_list_element(&mut self, _field: &NestedFieldRef, _partner: &P) -> Result<()> {
        Ok(())
    }
    /// Called after list field.
    fn after_list_element(&mut self, _field: &NestedFieldRef, _partner: &P) -> Result<()> {
        Ok(())
    }
    /// Called before map key field.
    fn before_map_key(&mut self, _field: &NestedFieldRef, _partner: &P) -> Result<()> {
        Ok(())
    }
    /// Called after map key field.
    fn after_map_key(&mut self, _field: &NestedFieldRef, _partner: &P) -> Result<()> {
        Ok(())
    }
    /// Called before map value field.
    fn before_map_value(&mut self, _field: &NestedFieldRef, _partner: &P) -> Result<()> {
        Ok(())
    }
    /// Called after map value field.
    fn after_map_value(&mut self, _field: &NestedFieldRef, _partner: &P) -> Result<()> {
        Ok(())
    }

    /// Called after schema's type visited.
    fn schema(&mut self, schema: &Schema, partner: &P, value: Self::T) -> Result<Self::T>;
    /// Called after struct's field type visited.
    fn field(&mut self, field: &NestedFieldRef, partner: &P, value: Self::T) -> Result<Self::T>;
    /// Called after struct's fields visited.
    fn r#struct(
        &mut self,
        r#struct: &StructType,
        partner: &P,
        results: Vec<Self::T>,
    ) -> Result<Self::T>;
    /// Called after list fields visited.
    fn list(&mut self, list: &ListType, partner: &P, value: Self::T) -> Result<Self::T>;
    /// Called after map's key and value fields visited.
    fn map(
        &mut self,
        map: &MapType,
        partner: &P,
        key_value: Self::T,
        value: Self::T,
    ) -> Result<Self::T>;
    /// Called when see a primitive type.
    fn primitive(&mut self, p: &PrimitiveType, partner: &P) -> Result<Self::T>;
    /// Called when a `variant` type is visited.
    ///
    /// Default mirrors Java 1.10.0 `TypeUtil.SchemaVisitor.variant`'s
    /// `UnsupportedOperationException("Unsupported type: variant")` — see
    /// [`SchemaVisitor::variant`]. Variant column DATA handling (Arrow value conversion, NaN
    /// counting, equality-delete projection) is file-level variant I/O, which is deferred — the
    /// loud default is the correct behavior for every current partner visitor.
    fn variant(&mut self, _partner: &P) -> Result<Self::T> {
        Err(Error::new(
            ErrorKind::FeatureUnsupported,
            "Unsupported type: variant",
        ))
    }
}

/// Accessor used to get child partner from parent partner.
pub trait PartnerAccessor<P> {
    /// Get the struct partner from schema partner.
    fn struct_partner<'a>(&self, schema_partner: &'a P) -> Result<&'a P>;
    /// Get the field partner from struct partner.
    fn field_partner<'a>(&self, struct_partner: &'a P, field: &NestedField) -> Result<&'a P>;
    /// Get the list element partner from list partner.
    fn list_element_partner<'a>(&self, list_partner: &'a P) -> Result<&'a P>;
    /// Get the map key partner from map partner.
    fn map_key_partner<'a>(&self, map_partner: &'a P) -> Result<&'a P>;
    /// Get the map value partner from map partner.
    fn map_value_partner<'a>(&self, map_partner: &'a P) -> Result<&'a P>;
}

/// Visiting a type in post order.
pub(crate) fn visit_type_with_partner<P, V: SchemaWithPartnerVisitor<P>, A: PartnerAccessor<P>>(
    r#type: &Type,
    partner: &P,
    visitor: &mut V,
    accessor: &A,
) -> Result<V::T> {
    visit_type_with_partner_at_depth(r#type, partner, visitor, accessor, 0)
}

/// Depth-bounded body of [`visit_type_with_partner`]. Mirrors [`visit_type_at_depth`]: `depth` is
/// the current nesting level (root at `0`), each nested element/key/value/field recurses at
/// `depth + 1`, and exceeding [`MAX_SCHEMA_NESTING_DEPTH`] returns a typed error instead of
/// overflowing the stack on a partner-influenced (and thus possibly hostile) schema.
fn visit_type_with_partner_at_depth<P, V: SchemaWithPartnerVisitor<P>, A: PartnerAccessor<P>>(
    r#type: &Type,
    partner: &P,
    visitor: &mut V,
    accessor: &A,
    depth: usize,
) -> Result<V::T> {
    if depth > MAX_SCHEMA_NESTING_DEPTH {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Schema type nesting exceeds maximum depth {MAX_SCHEMA_NESTING_DEPTH}"),
        ));
    }
    match r#type {
        Type::Primitive(p) => visitor.primitive(p, partner),
        Type::Variant => visitor.variant(partner),
        Type::List(list) => {
            let list_element_partner = accessor.list_element_partner(partner)?;
            visitor.before_list_element(&list.element_field, list_element_partner)?;
            let element_results = visit_type_with_partner_at_depth(
                &list.element_field.field_type,
                list_element_partner,
                visitor,
                accessor,
                depth + 1,
            )?;
            visitor.after_list_element(&list.element_field, list_element_partner)?;
            visitor.list(list, partner, element_results)
        }
        Type::Map(map) => {
            let key_partner = accessor.map_key_partner(partner)?;
            visitor.before_map_key(&map.key_field, key_partner)?;
            let key_result = visit_type_with_partner_at_depth(
                &map.key_field.field_type,
                key_partner,
                visitor,
                accessor,
                depth + 1,
            )?;
            visitor.after_map_key(&map.key_field, key_partner)?;

            let value_partner = accessor.map_value_partner(partner)?;
            visitor.before_map_value(&map.value_field, value_partner)?;
            let value_result = visit_type_with_partner_at_depth(
                &map.value_field.field_type,
                value_partner,
                visitor,
                accessor,
                depth + 1,
            )?;
            visitor.after_map_value(&map.value_field, value_partner)?;

            visitor.map(map, partner, key_result, value_result)
        }
        Type::Struct(s) => visit_struct_with_partner_at_depth(s, partner, visitor, accessor, depth),
    }
}

/// Visit struct type in post order.
pub fn visit_struct_with_partner<P, V: SchemaWithPartnerVisitor<P>, A: PartnerAccessor<P>>(
    s: &StructType,
    partner: &P,
    visitor: &mut V,
    accessor: &A,
) -> Result<V::T> {
    visit_struct_with_partner_at_depth(s, partner, visitor, accessor, 0)
}

/// Depth-bounded body of [`visit_struct_with_partner`]; see [`visit_type_with_partner_at_depth`].
fn visit_struct_with_partner_at_depth<P, V: SchemaWithPartnerVisitor<P>, A: PartnerAccessor<P>>(
    s: &StructType,
    partner: &P,
    visitor: &mut V,
    accessor: &A,
    depth: usize,
) -> Result<V::T> {
    if depth > MAX_SCHEMA_NESTING_DEPTH {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Schema type nesting exceeds maximum depth {MAX_SCHEMA_NESTING_DEPTH}"),
        ));
    }
    let mut results = Vec::with_capacity(s.fields().len());
    for field in s.fields() {
        let field_partner = accessor.field_partner(partner, field)?;
        visitor.before_struct_field(field, field_partner)?;
        let result = visit_type_with_partner_at_depth(
            &field.field_type,
            field_partner,
            visitor,
            accessor,
            depth + 1,
        )?;
        visitor.after_struct_field(field, field_partner)?;
        let result = visitor.field(field, field_partner, result)?;
        results.push(result);
    }

    visitor.r#struct(s, partner, results)
}

/// Visit schema in post order.
pub fn visit_schema_with_partner<P, V: SchemaWithPartnerVisitor<P>, A: PartnerAccessor<P>>(
    schema: &Schema,
    partner: &P,
    visitor: &mut V,
    accessor: &A,
) -> Result<V::T> {
    let result = visit_struct_with_partner(
        &schema.r#struct,
        accessor.struct_partner(partner)?,
        visitor,
        accessor,
    )?;
    visitor.schema(schema, partner, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial leaf-counting visitor: it asserts the walk reaches primitives and does no
    /// allocation, so the depth-limit error (raised before any visitor call past the bound) is
    /// the only thing under test.
    struct CountingVisitor;

    impl SchemaVisitor for CountingVisitor {
        type T = ();

        fn schema(&mut self, _schema: &Schema, _value: ()) -> Result<()> {
            Ok(())
        }
        fn field(&mut self, _field: &NestedFieldRef, _value: ()) -> Result<()> {
            Ok(())
        }
        fn r#struct(&mut self, _struct: &StructType, _results: Vec<()>) -> Result<()> {
            Ok(())
        }
        fn list(&mut self, _list: &ListType, _value: ()) -> Result<()> {
            Ok(())
        }
        fn map(&mut self, _map: &MapType, _key: (), _value: ()) -> Result<()> {
            Ok(())
        }
        fn primitive(&mut self, _p: &PrimitiveType) -> Result<()> {
            Ok(())
        }
    }

    /// Wrap `inner` in `depth` nested single-element lists.
    fn nested_list(depth: usize) -> Type {
        let mut ty = Type::Primitive(PrimitiveType::Int);
        for _ in 0..depth {
            ty = Type::List(ListType {
                element_field: NestedField::list_element(1, ty, true).into(),
            });
        }
        ty
    }

    #[test]
    fn test_visit_type_depth_limit_errors() {
        // Deeper than the bound must error (typed) rather than overflow the stack.
        let deep = nested_list(MAX_SCHEMA_NESTING_DEPTH + 5);
        let mut visitor = CountingVisitor;
        let err = visit_type(&deep, &mut visitor)
            .expect_err("over-deep type must error, not overflow the stack");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
    }

    #[test]
    fn test_visit_type_normal_nesting_succeeds() {
        // A normally-nested type still visits cleanly (behavior unchanged for valid inputs).
        let shallow = nested_list(8);
        let mut visitor = CountingVisitor;
        visit_type(&shallow, &mut visitor).expect("normally-nested type must visit successfully");
    }

    /// A trivial unit-partner visitor mirroring [`CountingVisitor`] for the partner walk. The
    /// partner is `()`; the accessor below hands the same unit partner to every child, which is
    /// enough to drive the recursion through struct fields, list elements, and map key/value.
    struct UnitPartnerVisitor;

    impl SchemaWithPartnerVisitor<()> for UnitPartnerVisitor {
        type T = ();

        fn schema(&mut self, _schema: &Schema, _partner: &(), _value: ()) -> Result<()> {
            Ok(())
        }
        fn field(&mut self, _field: &NestedFieldRef, _partner: &(), _value: ()) -> Result<()> {
            Ok(())
        }
        fn r#struct(
            &mut self,
            _struct: &StructType,
            _partner: &(),
            _results: Vec<()>,
        ) -> Result<()> {
            Ok(())
        }
        fn list(&mut self, _list: &ListType, _partner: &(), _value: ()) -> Result<()> {
            Ok(())
        }
        fn map(&mut self, _map: &MapType, _partner: &(), _key: (), _value: ()) -> Result<()> {
            Ok(())
        }
        fn primitive(&mut self, _p: &PrimitiveType, _partner: &()) -> Result<()> {
            Ok(())
        }
    }

    /// Hands the same `&()` partner to every child of the walk.
    struct UnitPartnerAccessor;

    impl PartnerAccessor<()> for UnitPartnerAccessor {
        fn struct_partner<'a>(&self, schema_partner: &'a ()) -> Result<&'a ()> {
            Ok(schema_partner)
        }
        fn field_partner<'a>(
            &self,
            struct_partner: &'a (),
            _field: &NestedField,
        ) -> Result<&'a ()> {
            Ok(struct_partner)
        }
        fn list_element_partner<'a>(&self, list_partner: &'a ()) -> Result<&'a ()> {
            Ok(list_partner)
        }
        fn map_key_partner<'a>(&self, map_partner: &'a ()) -> Result<&'a ()> {
            Ok(map_partner)
        }
        fn map_value_partner<'a>(&self, map_partner: &'a ()) -> Result<&'a ()> {
            Ok(map_partner)
        }
    }

    /// Wrap `inner` in `depth` nested single-key/value maps, so the bound must thread through BOTH
    /// the map key AND value child recursions of the partner walk (a list only exercises one
    /// child).
    fn nested_map(depth: usize) -> Type {
        let mut ty = Type::Primitive(PrimitiveType::Int);
        for _ in 0..depth {
            ty = Type::Map(MapType {
                key_field: NestedField::map_key_element(1, Type::Primitive(PrimitiveType::String))
                    .into(),
                value_field: NestedField::map_value_element(2, ty, true).into(),
            });
        }
        ty
    }

    #[test]
    fn test_visit_type_with_partner_depth_limit_errors() {
        // The partner walk must apply the SAME bound: deeper than the limit errors (typed) instead
        // of overflowing the stack on a partner-influenced schema.
        let deep = nested_list(MAX_SCHEMA_NESTING_DEPTH + 5);
        let mut visitor = UnitPartnerVisitor;
        let err = visit_type_with_partner(&deep, &(), &mut visitor, &UnitPartnerAccessor)
            .expect_err("over-deep partner walk must error, not overflow the stack");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);

        // Also prove the bound threads through the map key AND value child recursions.
        let deep_map = nested_map(MAX_SCHEMA_NESTING_DEPTH + 5);
        let mut visitor = UnitPartnerVisitor;
        let err = visit_type_with_partner(&deep_map, &(), &mut visitor, &UnitPartnerAccessor)
            .expect_err("over-deep partner map walk must error, not overflow the stack");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
    }

    #[test]
    fn test_visit_type_with_partner_normal_nesting_succeeds() {
        // A normally-nested type with a valid partner still visits cleanly (behavior unchanged).
        let shallow = nested_list(8);
        let mut visitor = UnitPartnerVisitor;
        visit_type_with_partner(&shallow, &(), &mut visitor, &UnitPartnerAccessor)
            .expect("normally-nested partner walk must visit successfully");
    }
}
