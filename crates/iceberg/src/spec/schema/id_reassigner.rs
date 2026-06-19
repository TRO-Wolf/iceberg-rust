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

//! Fresh field-id (re)assignment ported from Java `TypeUtil` / `AssignFreshIds`.
//!
//! Note for readers: `assign_fresh_ids_with_base` and `reassign_doc` are parity-complete and
//! unit-tested but have **no in-tree caller yet** — they will be wired when the create-table /
//! metadata-join consumers land. Do not assume they are load-bearing. (Status is tracked in
//! `docs/parity/GAP_MATRIX.md`, row "Type utilities".)

use std::cell::Cell;

use super::utils::try_insert_field;
use super::*;

pub struct ReassignFieldIds {
    next_field_id: i32,
    old_to_new_id: HashMap<i32, i32>,
}

// We are not using the visitor here, as post order traversal is not desired.
// Instead we want to re-assign all fields on one level first before diving deeper.
impl ReassignFieldIds {
    pub fn new(start_from: i32) -> Self {
        Self {
            next_field_id: start_from,
            old_to_new_id: HashMap::new(),
        }
    }

    pub fn reassign_field_ids(
        &mut self,
        fields: Vec<NestedFieldRef>,
    ) -> Result<Vec<NestedFieldRef>> {
        // Visit fields on the same level first
        let outer_fields = fields
            .into_iter()
            .map(|field| {
                try_insert_field(&mut self.old_to_new_id, field.id, self.next_field_id)?;
                let new_field = Arc::unwrap_or_clone(field).with_id(self.next_field_id);
                self.increase_next_field_id()?;
                Ok(Arc::new(new_field))
            })
            .collect::<Result<Vec<_>>>()?;

        // Now visit nested fields
        outer_fields
            .into_iter()
            .map(|field| {
                if field.field_type.is_primitive() {
                    Ok(field)
                } else {
                    let mut new_field = Arc::unwrap_or_clone(field);
                    *new_field.field_type = self.reassign_ids_visit_type(*new_field.field_type)?;
                    Ok(Arc::new(new_field))
                }
            })
            .collect()
    }

    fn reassign_ids_visit_type(&mut self, field_type: Type) -> Result<Type> {
        match field_type {
            Type::Primitive(s) => Ok(Type::Primitive(s)),
            // Leaf, like a primitive — Java 1.10.0 `AssignFreshIds.variant` returns the type
            // unchanged (it carries no nested ids to reassign).
            Type::Variant => Ok(Type::Variant),
            Type::Struct(s) => {
                let new_fields = self.reassign_field_ids(s.fields().to_vec())?;
                Ok(Type::Struct(StructType::new(new_fields)))
            }
            Type::List(l) => {
                self.old_to_new_id
                    .insert(l.element_field.id, self.next_field_id);
                let mut element_field = Arc::unwrap_or_clone(l.element_field);
                element_field.id = self.next_field_id;
                self.increase_next_field_id()?;
                *element_field.field_type =
                    self.reassign_ids_visit_type(*element_field.field_type)?;
                Ok(Type::List(ListType {
                    element_field: Arc::new(element_field),
                }))
            }
            Type::Map(m) => {
                self.old_to_new_id
                    .insert(m.key_field.id, self.next_field_id);
                let mut key_field = Arc::unwrap_or_clone(m.key_field);
                key_field.id = self.next_field_id;
                self.increase_next_field_id()?;
                *key_field.field_type = self.reassign_ids_visit_type(*key_field.field_type)?;

                self.old_to_new_id
                    .insert(m.value_field.id, self.next_field_id);
                let mut value_field = Arc::unwrap_or_clone(m.value_field);
                value_field.id = self.next_field_id;
                self.increase_next_field_id()?;
                *value_field.field_type = self.reassign_ids_visit_type(*value_field.field_type)?;

                Ok(Type::Map(MapType {
                    key_field: Arc::new(key_field),
                    value_field: Arc::new(value_field),
                }))
            }
        }
    }

    fn increase_next_field_id(&mut self) -> Result<()> {
        self.next_field_id = self.next_field_id.checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "Field ID overflowed, cannot add more fields",
            )
        })?;
        Ok(())
    }

    pub fn apply_to_identifier_fields(&self, field_ids: HashSet<i32>) -> Result<HashSet<i32>> {
        field_ids
            .into_iter()
            .map(|id| {
                self.old_to_new_id.get(&id).copied().ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Identifier Field ID {id} not found"),
                    )
                })
            })
            .collect()
    }

    pub fn apply_to_aliases(
        &self,
        alias: BiHashMap<String, i32>,
    ) -> Result<BiHashMap<String, i32>> {
        alias
            .into_iter()
            .map(|(name, id)| {
                self.old_to_new_id
                    .get(&id)
                    .copied()
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!("Field with id {id} for alias {name} not found"),
                        )
                    })
                    .map(|new_id| (name, new_id))
            })
            .collect()
    }
}

/// The `TypeUtil.NextID` seam: a stateful id source returning a fresh, strictly increasing id on
/// every call. Mirrors Java `org.apache.iceberg.types.TypeUtil.NextID` (a `@FunctionalInterface`
/// with `int get()`). All `assign_fresh_ids` entry points take one of these so the same recursion
/// services `UpdateSchema`-style closures, `assignIncreasingFreshIds`, and the
/// `reassignOrRefreshIds` "continue from the source's highest id" flow.
pub type NextId<'a> = dyn FnMut() -> Result<i32> + 'a;

// =====================================================================================
// assign-ids family — `TypeUtil.assignFreshIds` / `assignIds` / `assignIncreasingFreshIds`.
//
// Java's `AssignFreshIds` is a `CustomOrderSchemaVisitor`: a parent's ids are assigned BEFORE
// its children's (top-down / level-order), and within a struct ALL immediate field ids are
// assigned before any child type is recursed. We mirror that ordering explicitly here (the same
// two-pass shape as `ReassignFieldIds`) so ids match Java byte-for-byte on round trips.
// =====================================================================================

/// Assign fresh ids to every field in `field_type`, pulling each new id from `next_id` in the
/// Java `AssignFreshIds` (level-order) traversal order.
///
/// This is the Rust port of `TypeUtil.assignFreshIds(Type, NextID)` (the no-base-schema overload,
/// whose `idFor` always falls through to `nextId.get()`). A primitive — and `variant`, which Java
/// 1.10.0 `AssignFreshIds.variant` returns unchanged — carries no ids and passes through.
pub fn assign_fresh_ids(field_type: &Type, next_id: &mut NextId<'_>) -> Result<Type> {
    match field_type {
        Type::Primitive(p) => Ok(Type::Primitive(p.clone())),
        Type::Variant => Ok(Type::Variant),
        Type::Struct(s) => Ok(Type::Struct(assign_fresh_ids_to_fields(
            s.fields(),
            next_id,
        )?)),
        Type::List(l) => {
            // Level-order: the element id is the list's single immediate id; assign it first.
            let element_id = next_id()?;
            let element_type = assign_fresh_ids(&l.element_field.field_type, next_id)?;
            Ok(Type::List(ListType::new(Arc::new(
                NestedField::list_element(element_id, element_type, l.element_field.required),
            ))))
        }
        Type::Map(m) => {
            // Level-order: assign key id THEN value id (both immediate) before recursing either.
            let key_id = next_id()?;
            let value_id = next_id()?;
            let key_type = assign_fresh_ids(&m.key_field.field_type, next_id)?;
            let value_type = assign_fresh_ids(&m.value_field.field_type, next_id)?;
            Ok(Type::Map(MapType::new(
                Arc::new(NestedField::map_key_element(key_id, key_type)),
                Arc::new(NestedField::map_value_element(
                    value_id,
                    value_type,
                    m.value_field.required,
                )),
            )))
        }
    }
}

/// The struct body of [`assign_fresh_ids`]: assign fresh ids for ALL immediate fields first (pass
/// 1, level-order), then recurse into each field's type (pass 2). Doc and default attributes are
/// preserved (Java rebuilds with `NestedField.from(field).withId(id).ofType(type)`).
fn assign_fresh_ids_to_fields(
    fields: &[NestedFieldRef],
    next_id: &mut NextId<'_>,
) -> Result<StructType> {
    let new_ids = fields
        .iter()
        .map(|_| next_id())
        .collect::<Result<Vec<_>>>()?;
    let mut new_fields = Vec::with_capacity(fields.len());
    for (field, new_id) in fields.iter().zip(new_ids) {
        let new_type = assign_fresh_ids(&field.field_type, next_id)?;
        let mut rebuilt = NestedField::new(new_id, field.name.clone(), new_type, field.required);
        rebuilt.doc = field.doc.clone();
        rebuilt.initial_default = field.initial_default.clone();
        rebuilt.write_default = field.write_default.clone();
        new_fields.push(Arc::new(rebuilt));
    }
    Ok(StructType::new(new_fields))
}

/// Assign fresh ids to a whole schema, giving the result `schema_id` as its id. Rust port of
/// `TypeUtil.assignFreshIds(int baseId, Schema schema, NextID)` — the `baseId` is the schema id of
/// the produced schema (NOT a field id), and identifier fields are recomputed by name against the
/// freshly-assigned struct via [`refresh_identifier_fields`].
pub fn assign_fresh_ids_to_schema(
    schema_id: i32,
    schema: &Schema,
    next_id: &mut NextId<'_>,
) -> Result<Schema> {
    let fresh = assign_fresh_ids_to_fields(schema.as_struct().fields(), next_id)?;
    let identifier_field_ids = refresh_identifier_fields(&fresh, schema)?;
    Schema::builder()
        .with_schema_id(schema_id)
        .with_fields(fresh.fields().iter().cloned())
        .with_identifier_field_ids(identifier_field_ids)
        .build()
}

/// Assign monotonically increasing fresh ids starting at 1, returning a new schema with the same
/// schema id. Rust port of `TypeUtil.assignIncreasingFreshIds(Schema)`, which seeds an
/// `AtomicInteger(0)` and pulls ids via `incrementAndGet` (so the first id is 1).
pub fn assign_increasing_fresh_ids(schema: &Schema) -> Result<Schema> {
    let counter = Cell::new(0_i32);
    let mut next_id = || -> Result<i32> {
        let next = counter.get().checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "Field ID overflowed, cannot add more fields",
            )
        })?;
        counter.set(next);
        Ok(next)
    };
    assign_fresh_ids_to_schema(schema.schema_id(), schema, &mut next_id)
}

/// Assign fresh ids to a whole schema, but REUSE the id of any field whose full dotted name is also
/// present in `base` (the base schema). Rust port of `TypeUtil.assignFreshIds(Schema schema, Schema
/// base, NextID)`.
///
/// Java builds `AssignFreshIds(schema, base, nextId)` — a `CustomOrderSchemaVisitor` whose `idFor`
/// is the only place the id comes from:
/// ```text
/// idFor(name):  baseSchema != null && name != null && baseSchema.findField(name) != null
///                 ? baseSchema.findField(name).fieldId()
///                 : nextId.get()
/// name(id):     visitingSchema.findColumnName(id)   // the field's current full dotted name
/// ```
/// So for every field/element/key/value the new id is the base id of the SAME-NAMED field when one
/// exists, otherwise a fresh id from `next_id`. The struct walk is the same two-pass level-order as
/// [`assign_fresh_ids_to_fields`] (Java's `struct` collects ALL immediate ids via `idFor` first,
/// then recurses children lazily). The produced schema is built with the default schema id `0`
/// (Java's `new Schema(fields, refreshIdentifierFields(struct, schema))` — no id arg), and
/// identifier fields are recomputed by name against the freshly-id'd struct.
pub fn assign_fresh_ids_with_base(
    schema: &Schema,
    base: &Schema,
    next_id: &mut NextId<'_>,
) -> Result<Schema> {
    let fresh = assign_fresh_ids_with_base_struct(schema.as_struct(), schema, base, next_id)?;
    let identifier_field_ids = refresh_identifier_fields(&fresh, schema)?;
    Schema::builder()
        .with_schema_id(0)
        .with_fields(fresh.fields().iter().cloned())
        .with_identifier_field_ids(identifier_field_ids)
        .build()
}

/// `idFor(name(field_id))` for the base-schema overload: reuse the base schema's id for the field
/// whose current full dotted name matches, else pull a fresh id. `field_id` is the field's CURRENT
/// id in `visiting` (Java's `name(int)` = `visitingSchema.findColumnName(id)`).
fn base_id_for(
    field_id: i32,
    visiting: &Schema,
    base: &Schema,
    next_id: &mut NextId<'_>,
) -> Result<i32> {
    // Resolve the field's current full name in `visiting`, then reuse the base schema's id for that
    // same name (Java `idFor`: `baseSchema.findField(name)` reuse, else a fresh id). The name is
    // bound into an owned `String` so the lookup reads as two plain steps; a borrowed `&str` from
    // `visiting` would also be sound here (the `base` lookup borrows a different schema).
    let name: Option<String> = visiting.name_by_field_id(field_id).map(str::to_owned);
    if let Some(name) = name
        && let Some(reused) = base.field_id_by_name(&name)
    {
        return Ok(reused);
    }
    next_id()
}

/// Struct body of [`assign_fresh_ids_with_base`]: two-pass level-order (assign every immediate id
/// via [`base_id_for`] first, then recurse children) — the exact shape of Java's `AssignFreshIds.
/// struct`, where the lazy `Iterables.transform` over `VisitFieldFuture`s is forced only in pass 2.
fn assign_fresh_ids_with_base_struct(
    struct_type: &StructType,
    visiting: &Schema,
    base: &Schema,
    next_id: &mut NextId<'_>,
) -> Result<StructType> {
    let new_ids = struct_type
        .fields()
        .iter()
        .map(|field| base_id_for(field.id, visiting, base, next_id))
        .collect::<Result<Vec<_>>>()?;
    let mut new_fields = Vec::with_capacity(struct_type.fields().len());
    for (field, new_id) in struct_type.fields().iter().zip(new_ids) {
        let new_type = assign_fresh_ids_with_base_type(&field.field_type, visiting, base, next_id)?;
        let mut rebuilt = NestedField::new(new_id, field.name.clone(), new_type, field.required);
        rebuilt.doc = field.doc.clone();
        rebuilt.initial_default = field.initial_default.clone();
        rebuilt.write_default = field.write_default.clone();
        new_fields.push(Arc::new(rebuilt));
    }
    Ok(StructType::new(new_fields))
}

/// Recurse [`assign_fresh_ids_with_base_struct`] into a nested type. Element/key/value ids reuse the
/// base id of the same-named element/key/value (Java's `list`/`map` call `idFor(name(elementId))`
/// etc.), else pull fresh.
fn assign_fresh_ids_with_base_type(
    field_type: &Type,
    visiting: &Schema,
    base: &Schema,
    next_id: &mut NextId<'_>,
) -> Result<Type> {
    match field_type {
        Type::Primitive(p) => Ok(Type::Primitive(p.clone())),
        Type::Variant => Ok(Type::Variant),
        Type::Struct(s) => Ok(Type::Struct(assign_fresh_ids_with_base_struct(
            s, visiting, base, next_id,
        )?)),
        Type::List(l) => {
            // Level-order: the element id is the list's single immediate id; assign it first.
            let element_id = base_id_for(l.element_field.id, visiting, base, next_id)?;
            let element_type = assign_fresh_ids_with_base_type(
                &l.element_field.field_type,
                visiting,
                base,
                next_id,
            )?;
            Ok(Type::List(ListType::new(Arc::new(
                NestedField::list_element(element_id, element_type, l.element_field.required),
            ))))
        }
        Type::Map(m) => {
            // Level-order: assign key id THEN value id (both immediate) before recursing either.
            let key_id = base_id_for(m.key_field.id, visiting, base, next_id)?;
            let value_id = base_id_for(m.value_field.id, visiting, base, next_id)?;
            let key_type =
                assign_fresh_ids_with_base_type(&m.key_field.field_type, visiting, base, next_id)?;
            let value_type = assign_fresh_ids_with_base_type(
                &m.value_field.field_type,
                visiting,
                base,
                next_id,
            )?;
            Ok(Type::Map(MapType::new(
                Arc::new(NestedField::map_key_element(key_id, key_type)),
                Arc::new(NestedField::map_value_element(
                    value_id,
                    value_type,
                    m.value_field.required,
                )),
            )))
        }
    }
}

/// Re-key a type's ids through the `old_id -> new_id` map `get_id`. Rust port of
/// `TypeUtil.assignIds(Type, GetID)`: unlike [`assign_fresh_ids`], the structure is preserved and
/// EVERY id (including list element / map key+value) is rewritten by the caller-supplied function.
/// A missing mapping is the caller's contract to handle (the closure returns the id to use).
pub fn assign_ids(field_type: &Type, get_id: &mut dyn FnMut(i32) -> i32) -> Result<Type> {
    match field_type {
        Type::Primitive(p) => Ok(Type::Primitive(p.clone())),
        Type::Variant => Ok(Type::Variant),
        Type::Struct(s) => {
            let mut new_fields = Vec::with_capacity(s.fields().len());
            for field in s.fields() {
                let new_id = get_id(field.id);
                let new_type = assign_ids(&field.field_type, get_id)?;
                let mut rebuilt =
                    NestedField::new(new_id, field.name.clone(), new_type, field.required);
                rebuilt.doc = field.doc.clone();
                rebuilt.initial_default = field.initial_default.clone();
                rebuilt.write_default = field.write_default.clone();
                new_fields.push(Arc::new(rebuilt));
            }
            Ok(Type::Struct(StructType::new(new_fields)))
        }
        Type::List(l) => {
            let element_id = get_id(l.element_field.id);
            let element_type = assign_ids(&l.element_field.field_type, get_id)?;
            Ok(Type::List(ListType::new(Arc::new(
                NestedField::list_element(element_id, element_type, l.element_field.required),
            ))))
        }
        Type::Map(m) => {
            let key_id = get_id(m.key_field.id);
            let value_id = get_id(m.value_field.id);
            let key_type = assign_ids(&m.key_field.field_type, get_id)?;
            let value_type = assign_ids(&m.value_field.field_type, get_id)?;
            Ok(Type::Map(MapType::new(
                Arc::new(NestedField::map_key_element(key_id, key_type)),
                Arc::new(NestedField::map_value_element(
                    value_id,
                    value_type,
                    m.value_field.required,
                )),
            )))
        }
    }
}

// =====================================================================================
// reassign family — `TypeUtil.reassignIds` / `reassignOrRefreshIds` / `reassignDoc` /
// `refreshIdentifierFields`.
//
// Java's `ReassignIds` is a `CustomOrderSchemaVisitor` that walks `schema` while tracking the
// matching position in a `sourceSchema`, aligning ids BY NAME. The behavior at a name that is
// absent from the source is controlled by an optional id source: `None` (→ `reassign_ids`) makes
// the absence an error; a real `NextId` (→ `reassign_or_refresh_ids`) assigns fresh ids to the
// whole unmatched subtree.
// =====================================================================================

/// Align the ids of `schema` to `id_source` BY NAME (case-sensitive). Rust port of
/// `TypeUtil.reassignIds(Schema, Schema)` (= the `caseSensitive = true` overload). Every field in
/// `schema` must have a same-named field in `id_source`; a name not present in the source is a
/// hard error (`DataInvalid`), mirroring Java's `IllegalArgumentException("Field ... not found")`.
pub fn reassign_ids(schema: &Schema, id_source: &Schema) -> Result<Schema> {
    reassign_ids_with_case(schema, id_source, true)
}

/// Case-sensitivity-parameterized [`reassign_ids`]. Rust port of
/// `TypeUtil.reassignIds(Schema, Schema, boolean caseSensitive)`.
pub fn reassign_ids_with_case(
    schema: &Schema,
    id_source: &Schema,
    case_sensitive: bool,
) -> Result<Schema> {
    let mut visitor = ReassignIds::new(id_source, None, case_sensitive);
    let fresh = visitor.visit_schema(schema)?;
    let identifier_field_ids = refresh_identifier_fields(&fresh, schema)?;
    Schema::builder()
        .with_schema_id(schema.schema_id())
        .with_fields(fresh.fields().iter().cloned())
        .with_identifier_field_ids(identifier_field_ids)
        .build()
}

/// Align the ids of `schema` to `id_source` BY NAME, assigning FRESH ids (continuing from
/// `id_source.highest_field_id()`) to any name not present in the source. Rust port of
/// `TypeUtil.reassignOrRefreshIds(Schema, Schema)` (= the `caseSensitive = true` overload).
pub fn reassign_or_refresh_ids(schema: &Schema, id_source: &Schema) -> Result<Schema> {
    reassign_or_refresh_ids_with_case(schema, id_source, true)
}

/// Case-sensitivity-parameterized [`reassign_or_refresh_ids`]. Rust port of
/// `TypeUtil.reassignOrRefreshIds(Schema, Schema, boolean caseSensitive)`.
pub fn reassign_or_refresh_ids_with_case(
    schema: &Schema,
    id_source: &Schema,
    case_sensitive: bool,
) -> Result<Schema> {
    let counter = Cell::new(id_source.highest_field_id());
    let mut next_id = || -> Result<i32> {
        let next = counter.get().checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "Field ID overflowed, cannot add more fields",
            )
        })?;
        counter.set(next);
        Ok(next)
    };
    let fresh = {
        let mut visitor = ReassignIds::new(id_source, Some(&mut next_id), case_sensitive);
        visitor.visit_schema(schema)?
    };
    let identifier_field_ids = refresh_identifier_fields(&fresh, schema)?;
    Schema::builder()
        .with_schema_id(schema.schema_id())
        .with_fields(fresh.fields().iter().cloned())
        .with_identifier_field_ids(identifier_field_ids)
        .build()
}

/// Copy field docs from `doc_source` onto `schema` BY ID, leaving everything else unchanged. Rust
/// port of `TypeUtil.reassignDoc(Schema, Schema)` (Java's `ReassignDoc` visitor): for each field,
/// if `doc_source` has a field with the same id and a non-null doc, that doc replaces the field's
/// doc; otherwise the doc is cleared. The result keeps the original schema id (Java's `reassignDoc`
/// builds `new Schema(fields)`, i.e. the default id `0`); we preserve `schema.schema_id()` because
/// the `Schema(List)` Java ctor and our builder both default to 0 but callers expect identity.
pub fn reassign_doc(schema: &Schema, doc_source: &Schema) -> Result<Schema> {
    let new_struct = reassign_doc_struct(schema.as_struct(), doc_source);
    Schema::builder()
        .with_schema_id(schema.schema_id())
        .with_fields(new_struct.fields().iter().cloned())
        .with_identifier_field_ids(schema.identifier_field_ids())
        .build()
}

/// Recursive worker for [`reassign_doc`]: rebuild a struct copying docs from `doc_source` by id.
fn reassign_doc_struct(s: &StructType, doc_source: &Schema) -> StructType {
    let new_fields = s
        .fields()
        .iter()
        .map(|field| {
            let new_type = reassign_doc_type(&field.field_type, doc_source);
            let mut rebuilt =
                NestedField::new(field.id, field.name.clone(), new_type, field.required);
            // Java's ReassignDoc.field sets doc = sourceField != null ? sourceField.doc() : null.
            rebuilt.doc = doc_source
                .field_by_id(field.id)
                .and_then(|sf| sf.doc.clone());
            rebuilt.initial_default = field.initial_default.clone();
            rebuilt.write_default = field.write_default.clone();
            Arc::new(rebuilt)
        })
        .collect();
    StructType::new(new_fields)
}

/// Recurse [`reassign_doc_struct`] into nested types. Element/key/value field docs are NOT carried
/// by Java's `ReassignDoc` (its `list`/`map` only rebuild the container), so we leave them as-is.
fn reassign_doc_type(field_type: &Type, doc_source: &Schema) -> Type {
    match field_type {
        Type::Struct(s) => Type::Struct(reassign_doc_struct(s, doc_source)),
        Type::List(l) => {
            let element_type = reassign_doc_type(&l.element_field.field_type, doc_source);
            let mut element = (*l.element_field).clone();
            *element.field_type = element_type;
            Type::List(ListType::new(Arc::new(element)))
        }
        Type::Map(m) => {
            let key_type = reassign_doc_type(&m.key_field.field_type, doc_source);
            let value_type = reassign_doc_type(&m.value_field.field_type, doc_source);
            let mut key = (*m.key_field).clone();
            *key.field_type = key_type;
            let mut value = (*m.value_field).clone();
            *value.field_type = value_type;
            Type::Map(MapType::new(Arc::new(key), Arc::new(value)))
        }
        other => other.clone(),
    }
}

/// Recompute the identifier-field id set of a freshly-id'd struct by carrying the source schema's
/// identifier-field NAMES across. Rust port of `TypeUtil.refreshIdentifierFields(StructType,
/// Schema)`: index the new struct by name, then for each identifier name in `source_schema` look
/// up its id in the new struct. A name that no longer resolves is a hard error
/// (`DataInvalid`), mirroring Java's `Preconditions.checkArgument(..., "Cannot find ID for
/// identifier field %s in schema %s")`.
pub fn refresh_identifier_fields(
    new_struct: &StructType,
    source_schema: &Schema,
) -> Result<HashSet<i32>> {
    // Java uses indexByName (full dotted names) so a nested identifier field resolves correctly.
    let name_to_id = {
        let mut index = IndexByName::default();
        visit_struct(new_struct, &mut index)?;
        index.indexes().0
    };

    let mut identifier_field_ids = HashSet::new();
    for id in source_schema.identifier_field_ids() {
        let name = source_schema.name_by_field_id(id).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot find name for identifier field id {id} in source schema"),
            )
        })?;
        let new_id = name_to_id.get(name).copied().ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot find ID for identifier field {name} in schema {new_struct}"),
            )
        })?;
        identifier_field_ids.insert(new_id);
    }
    Ok(identifier_field_ids)
}

/// The Java `ReassignIds` `CustomOrderSchemaVisitor`, expressed as an explicit recursive walk over
/// `schema` that tracks the matching position in `source` (the id source). At each field the
/// matching same-named source field supplies the id and the source position for the child walk;
/// when there is no match, an `assign_id` source (if present) assigns fresh ids to the whole
/// unmatched subtree, otherwise the absence is a hard error.
struct ReassignIds<'a, 'b> {
    source: &'a Schema,
    assign_id: Option<&'a mut NextId<'b>>,
    case_sensitive: bool,
}

impl<'a, 'b> ReassignIds<'a, 'b> {
    fn new(
        source: &'a Schema,
        assign_id: Option<&'a mut NextId<'b>>,
        case_sensitive: bool,
    ) -> Self {
        Self {
            source,
            assign_id,
            case_sensitive,
        }
    }

    fn visit_schema(&mut self, schema: &Schema) -> Result<StructType> {
        let source_struct = self.source.as_struct().clone();
        self.reassign_struct(schema.as_struct(), &source_struct)
    }

    /// Look up `name` in `source_struct` honoring `case_sensitive`.
    fn source_field<'s>(
        &self,
        source_struct: &'s StructType,
        name: &str,
    ) -> Option<&'s NestedFieldRef> {
        if self.case_sensitive {
            source_struct.field_by_name(name)
        } else {
            source_struct
                .fields()
                .iter()
                .find(|f| f.name.eq_ignore_ascii_case(name))
        }
    }

    fn reassign_struct(
        &mut self,
        struct_type: &StructType,
        source_struct: &StructType,
    ) -> Result<StructType> {
        // Two-phase, matching Java `ReassignIds.struct`: the `CustomOrderSchemaVisitor` first forces
        // ALL child results (computing each field's reassigned TYPE, which pulls any fresh subtree
        // ids in field order), and only THEN loops over the fields to assign each field's OWN id.
        // Interleaving the two phases would change the fresh-id stream for unmatched subtrees.
        let matched: Vec<Option<NestedFieldRef>> = struct_type
            .fields()
            .iter()
            .map(|field| self.source_field(source_struct, &field.name).cloned())
            .collect();

        // Phase 1: child types (in field order).
        let mut new_types = Vec::with_capacity(struct_type.fields().len());
        for (field, source_field) in struct_type.fields().iter().zip(matched.iter()) {
            let new_type = match source_field {
                Some(source_field) => {
                    self.reassign_type(&field.field_type, &source_field.field_type)?
                }
                None => self.assign_fresh_or_fail(&field.field_type, &field.name)?,
            };
            new_types.push(new_type);
        }

        // Phase 2: each field's own id (in field order).
        let mut new_fields = Vec::with_capacity(struct_type.fields().len());
        for ((field, source_field), new_type) in struct_type
            .fields()
            .iter()
            .zip(matched.into_iter())
            .zip(new_types.into_iter())
        {
            let new_id = match source_field {
                Some(source_field) => source_field.id,
                None => self.next_assigned_id(&field.name)?,
            };
            let mut rebuilt =
                NestedField::new(new_id, field.name.clone(), new_type, field.required);
            rebuilt.doc = field.doc.clone();
            rebuilt.initial_default = field.initial_default.clone();
            rebuilt.write_default = field.write_default.clone();
            new_fields.push(Arc::new(rebuilt));
        }
        Ok(StructType::new(new_fields))
    }

    /// Reassign ids in `field_type`, aligning against `source_type` (the same-named source field's
    /// type). The id for a nested container's element/key/value comes from the source container.
    fn reassign_type(&mut self, field_type: &Type, source_type: &Type) -> Result<Type> {
        match (field_type, source_type) {
            (Type::Struct(s), Type::Struct(source_s)) => {
                Ok(Type::Struct(self.reassign_struct(s, source_s)?))
            }
            (Type::List(l), Type::List(source_l)) => {
                let element_type = self.reassign_type(
                    &l.element_field.field_type,
                    &source_l.element_field.field_type,
                )?;
                Ok(Type::List(ListType::new(Arc::new(
                    NestedField::list_element(
                        source_l.element_field.id,
                        element_type,
                        l.element_field.required,
                    ),
                ))))
            }
            (Type::Map(m), Type::Map(source_m)) => {
                let key_type =
                    self.reassign_type(&m.key_field.field_type, &source_m.key_field.field_type)?;
                let value_type = self
                    .reassign_type(&m.value_field.field_type, &source_m.value_field.field_type)?;
                Ok(Type::Map(MapType::new(
                    Arc::new(NestedField::map_key_element(
                        source_m.key_field.id,
                        key_type,
                    )),
                    Arc::new(NestedField::map_value_element(
                        source_m.value_field.id,
                        value_type,
                        m.value_field.required,
                    )),
                )))
            }
            // A primitive / variant visited field carries no nested ids and passes through
            // unchanged regardless of the source type — Java `ReassignIds.primitive`/`variant`
            // simply `return type` with no `sourceType` check.
            (Type::Primitive(_) | Type::Variant, _) => Ok(field_type.clone()),
            // A structural type mismatch at a MATCHED name (e.g. the visited field is a list but the
            // same-named source field is a struct). Java's `ReassignIds.struct`/`list`/`map` runs
            // `Preconditions.checkArgument(sourceType.isXType(), "Not a X: %s", sourceType)` and
            // THROWS `IllegalArgumentException` — it does NOT assign fresh ids here. We mirror the
            // exact "Not a struct/list/map: <source>" message (`DataInvalid`).
            (Type::Struct(_), _) => Err(Self::not_a_type_error("struct", source_type)),
            (Type::List(_), _) => Err(Self::not_a_type_error("list", source_type)),
            (Type::Map(_), _) => Err(Self::not_a_type_error("map", source_type)),
        }
    }

    /// Build the Java-faithful structural-mismatch error: `ReassignIds.struct`/`list`/`map` throws
    /// `IllegalArgumentException("Not a struct/list/map: <sourceType>")` via `Preconditions.
    /// checkArgument`. `kind` is the visited container kind ("struct"/"list"/"map") and `source` is
    /// the same-named source field's (mismatching) type, rendered like Java's `%s` (`Type.toString`).
    fn not_a_type_error(kind: &str, source: &Type) -> Error {
        Error::new(ErrorKind::DataInvalid, format!("Not a {kind}: {source}"))
    }

    /// Assign fresh ids to a whole subtree (used when a name has no match in the source), or fail
    /// when there is no id source — mirroring Java's `assignId != null ? assignFreshIds(..) :
    /// throw`.
    fn assign_fresh_or_fail(&mut self, field_type: &Type, name: &str) -> Result<Type> {
        if field_type.is_primitive() || matches!(field_type, Type::Variant) {
            return Ok(field_type.clone());
        }
        match self.assign_id.as_deref_mut() {
            Some(next_id) => assign_fresh_ids(field_type, next_id),
            None => Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Field {name} not found in source schema"),
            )),
        }
    }

    /// Produce the id for an unmatched field: a fresh id from the id source, or a hard error.
    fn next_assigned_id(&mut self, name: &str) -> Result<i32> {
        match self.assign_id.as_deref_mut() {
            Some(next_id) => next_id(),
            None => Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Field {name} not found in source schema"),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::schema::tests::table_schema_nested;

    #[test]
    fn test_reassign_ids() {
        let schema = Schema::builder()
            .with_schema_id(1)
            .with_identifier_field_ids(vec![3])
            .with_alias(BiHashMap::from_iter(vec![("bar_alias".to_string(), 3)]))
            .with_fields(vec![
                NestedField::optional(5, "foo", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(3, "bar", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(4, "baz", Type::Primitive(PrimitiveType::Boolean)).into(),
            ])
            .build()
            .unwrap();

        let reassigned_schema = schema
            .into_builder()
            .with_reassigned_field_ids(0)
            .build()
            .unwrap();

        let expected = Schema::builder()
            .with_schema_id(1)
            .with_identifier_field_ids(vec![1])
            .with_alias(BiHashMap::from_iter(vec![("bar_alias".to_string(), 1)]))
            .with_fields(vec![
                NestedField::optional(0, "foo", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(1, "bar", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "baz", Type::Primitive(PrimitiveType::Boolean)).into(),
            ])
            .build()
            .unwrap();

        pretty_assertions::assert_eq!(expected, reassigned_schema);
        assert_eq!(reassigned_schema.highest_field_id(), 2);
    }

    // RISK: id reassignment over a schema containing a variant column must treat variant as a
    // LEAF (Java 1.10.0 `AssignFreshIds.variant` returns the type unchanged): the column's own id
    // is reassigned like any field, the type passes through, and SIBLING ids after it stay in
    // sequence. A missing arm would make `with_reassigned_field_ids` (used by catalog
    // create-table flows) fail on every variant schema.
    #[test]
    fn test_reassign_ids_passes_variant_through() {
        let schema = Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::optional(7, "v", Type::Variant).into(),
                NestedField::required(5, "bar", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();

        let reassigned_schema = schema
            .into_builder()
            .with_reassigned_field_ids(0)
            .build()
            .unwrap();

        let expected = Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::optional(0, "v", Type::Variant).into(),
                NestedField::required(1, "bar", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();

        pretty_assertions::assert_eq!(expected, reassigned_schema);
        assert_eq!(reassigned_schema.highest_field_id(), 1);
    }

    #[test]
    fn test_reassigned_ids_nested() {
        let schema = table_schema_nested();
        let reassigned_schema = schema
            .into_builder()
            .with_alias(BiHashMap::from_iter(vec![("bar_alias".to_string(), 2)]))
            .with_reassigned_field_ids(0)
            .build()
            .unwrap();

        let expected = Schema::builder()
            .with_schema_id(1)
            .with_identifier_field_ids(vec![1])
            .with_alias(BiHashMap::from_iter(vec![("bar_alias".to_string(), 1)]))
            .with_fields(vec![
                NestedField::optional(0, "foo", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(1, "bar", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "baz", Type::Primitive(PrimitiveType::Boolean)).into(),
                NestedField::required(
                    3,
                    "qux",
                    Type::List(ListType {
                        element_field: NestedField::list_element(
                            7,
                            Type::Primitive(PrimitiveType::String),
                            true,
                        )
                        .into(),
                    }),
                )
                .into(),
                NestedField::required(
                    4,
                    "quux",
                    Type::Map(MapType {
                        key_field: NestedField::map_key_element(
                            8,
                            Type::Primitive(PrimitiveType::String),
                        )
                        .into(),
                        value_field: NestedField::map_value_element(
                            9,
                            Type::Map(MapType {
                                key_field: NestedField::map_key_element(
                                    10,
                                    Type::Primitive(PrimitiveType::String),
                                )
                                .into(),
                                value_field: NestedField::map_value_element(
                                    11,
                                    Type::Primitive(PrimitiveType::Int),
                                    true,
                                )
                                .into(),
                            }),
                            true,
                        )
                        .into(),
                    }),
                )
                .into(),
                NestedField::required(
                    5,
                    "location",
                    Type::List(ListType {
                        element_field: NestedField::list_element(
                            12,
                            Type::Struct(StructType::new(vec![
                                NestedField::optional(
                                    13,
                                    "latitude",
                                    Type::Primitive(PrimitiveType::Float),
                                )
                                .into(),
                                NestedField::optional(
                                    14,
                                    "longitude",
                                    Type::Primitive(PrimitiveType::Float),
                                )
                                .into(),
                            ])),
                            true,
                        )
                        .into(),
                    }),
                )
                .into(),
                NestedField::optional(
                    6,
                    "person",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(15, "name", Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::required(16, "age", Type::Primitive(PrimitiveType::Int))
                            .into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap();

        pretty_assertions::assert_eq!(expected, reassigned_schema);
        assert_eq!(reassigned_schema.highest_field_id(), 16);
        assert_eq!(reassigned_schema.field_by_id(6).unwrap().name, "person");
        assert_eq!(reassigned_schema.field_by_id(16).unwrap().name, "age");
    }

    #[test]
    fn test_reassign_ids_fails_with_duplicate_ids() {
        let reassigned_schema = Schema::builder()
            .with_schema_id(1)
            .with_identifier_field_ids(vec![5])
            .with_alias(BiHashMap::from_iter(vec![("bar_alias".to_string(), 3)]))
            .with_fields(vec![
                NestedField::required(5, "foo", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(3, "bar", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(3, "baz", Type::Primitive(PrimitiveType::Boolean)).into(),
            ])
            .with_reassigned_field_ids(0)
            .build()
            .unwrap_err();

        assert!(reassigned_schema.message().contains("'field.id' 3"));
    }

    // ===== assign-ids family =====

    /// `assign_fresh_ids` over a nested list-of-map type assigns ids in Java's level-order: the
    /// element id BEFORE the map's key/value ids, and the map's key id BEFORE its value id. Risk: a
    /// naive depth-first walk would interleave ids differently and break Java round-trip parity.
    #[test]
    fn test_assign_fresh_ids_level_order_nested() {
        // list< map< string, int > > with arbitrary original ids.
        let field_type = Type::List(ListType::new(
            NestedField::list_element(
                50,
                Type::Map(MapType::new(
                    NestedField::map_key_element(60, Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::map_value_element(70, Type::Primitive(PrimitiveType::Int), true)
                        .into(),
                )),
                true,
            )
            .into(),
        ));

        let counter = Cell::new(0_i32);
        let mut next = || -> Result<i32> {
            let n = counter.get() + 1;
            counter.set(n);
            Ok(n)
        };
        let result = assign_fresh_ids(&field_type, &mut next).unwrap();

        // Element id = 1 (assigned first), then key = 2, value = 3 (key before value).
        let Type::List(list) = result else {
            panic!("expected list")
        };
        assert_eq!(list.element_field.id, 1, "list element id assigned first");
        let Type::Map(map) = list.element_field.field_type.as_ref() else {
            panic!("expected map element")
        };
        assert_eq!(map.key_field.id, 2, "map key id before value id");
        assert_eq!(map.value_field.id, 3, "map value id after key id");
    }

    /// `assign_fresh_ids` over a struct with a nested struct FOLLOWED BY a sibling assigns ids in
    /// Java's level-order: ALL immediate struct field ids first (`a`, then `b`), and only THEN the
    /// nested child id (`x`). Risk: a depth-first walk (assign `a`, immediately recurse into `x`,
    /// then assign `b`) would yield a=1, x=2, b=3 instead of a=1, b=2, x=3 — exactly the regression
    /// that breaks Java `AssignFreshIds` round-trip parity. This test DISCRIMINATES the two-pass
    /// level-order from a depth-first walk (the list<map> test alone cannot, as it has no struct
    /// sibling). Mirrors Java `AssignFreshIds.struct` (pass 1: all immediate `idFor`; pass 2:
    /// recurse children).
    #[test]
    fn test_assign_fresh_ids_level_order_struct_siblings() {
        // struct< a: struct< x: int >, b: int > with arbitrary original ids.
        let field_type = Type::Struct(StructType::new(vec![
            NestedField::required(
                90,
                "a",
                Type::Struct(StructType::new(vec![
                    NestedField::required(91, "x", Type::Primitive(PrimitiveType::Int)).into(),
                ])),
            )
            .into(),
            NestedField::required(92, "b", Type::Primitive(PrimitiveType::Int)).into(),
        ]));

        let counter = Cell::new(0_i32);
        let mut next = || -> Result<i32> {
            let n = counter.get() + 1;
            counter.set(n);
            Ok(n)
        };
        let result = assign_fresh_ids(&field_type, &mut next).unwrap();

        let Type::Struct(s) = result else {
            panic!("expected struct")
        };
        // Level-order: a=1 and b=2 (BOTH immediate ids) before the nested x=3.
        assert_eq!(s.fields()[0].id, 1, "first immediate field `a` gets id 1");
        assert_eq!(
            s.fields()[1].id,
            2,
            "second immediate field `b` gets id 2 BEFORE descending into `a` (level-order, not 3)"
        );
        let Type::Struct(inner) = s.fields()[0].field_type.as_ref() else {
            panic!("expected nested struct")
        };
        assert_eq!(
            inner.fields()[0].id,
            3,
            "nested field `x` gets id 3 (assigned AFTER both siblings; depth-first would give 2)"
        );
    }

    /// `assign_fresh_ids_to_schema` recomputes identifier fields by NAME against the fresh struct,
    /// and stamps the given schema id. Risk: dropping the identifier-field carry-over would lose the
    /// table's primary-key declaration after a fresh-id pass.
    #[test]
    fn test_assign_fresh_ids_to_schema_carries_identifier_by_name() {
        let schema = Schema::builder()
            .with_schema_id(7)
            .with_identifier_field_ids(vec![100])
            .with_fields(vec![
                NestedField::required(100, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(101, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();

        let counter = Cell::new(0_i32);
        let mut next = || -> Result<i32> {
            let n = counter.get() + 1;
            counter.set(n);
            Ok(n)
        };
        let fresh = assign_fresh_ids_to_schema(9, &schema, &mut next).unwrap();

        assert_eq!(fresh.schema_id(), 9);
        // Fresh ids 1, 2 in field order; the identifier follows the `id` column to its new id (1).
        assert_eq!(fresh.field_by_name("id").unwrap().id, 1);
        assert_eq!(fresh.field_by_name("name").unwrap().id, 2);
        let ids: Vec<i32> = fresh.identifier_field_ids().collect();
        assert_eq!(ids, vec![1]);
    }

    /// `assign_increasing_fresh_ids` starts the id stream at 1 (Java seeds `AtomicInteger(0)` and
    /// pulls via `incrementAndGet`). Risk: an off-by-one (starting at 0) would shift every id.
    #[test]
    fn test_assign_increasing_fresh_ids_starts_at_one() {
        let schema = Schema::builder()
            .with_schema_id(3)
            .with_fields(vec![
                NestedField::required(40, "a", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(41, "b", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();

        let fresh = assign_increasing_fresh_ids(&schema).unwrap();
        assert_eq!(fresh.schema_id(), 3, "schema id preserved");
        assert_eq!(fresh.field_by_name("a").unwrap().id, 1, "first id is 1");
        assert_eq!(fresh.field_by_name("b").unwrap().id, 2);
        assert_eq!(fresh.highest_field_id(), 2);
    }

    /// `assign_ids` rewrites EVERY id through the supplied `old -> new` map (structure preserved),
    /// unlike `assign_fresh_ids`. Risk: missing the list-element / map id rewrite would leave stale
    /// ids dangling after a remap.
    #[test]
    fn test_assign_ids_remaps_all_ids() {
        let field_type = Type::Struct(StructType::new(vec![
            NestedField::required(
                1,
                "items",
                Type::List(ListType::new(
                    NestedField::list_element(2, Type::Primitive(PrimitiveType::Int), true).into(),
                )),
            )
            .into(),
        ]));

        // Map each old id to old + 100.
        let mut get_id = |old: i32| old + 100;
        let result = assign_ids(&field_type, &mut get_id).unwrap();

        let Type::Struct(s) = result else {
            panic!("expected struct")
        };
        assert_eq!(s.fields()[0].id, 101);
        let Type::List(list) = s.fields()[0].field_type.as_ref() else {
            panic!("expected list")
        };
        assert_eq!(list.element_field.id, 102, "list element id remapped too");
    }

    /// `assign_fresh_ids_with_base` REUSES the base schema's id for every field whose full dotted
    /// name also appears in the base, and assigns FRESH ids (from `next_id`) to the rest. Java's
    /// `idFor(name)` = `baseSchema.findField(name) != null ? found.fieldId() : nextId.get()`, where
    /// `name` comes from the visiting schema's `findColumnName(currentId)`. Risk: ignoring the base
    /// would re-number a matched column and break the "reuse base ids where structure matches"
    /// contract this overload exists for.
    #[test]
    fn test_assign_fresh_ids_with_base_reuses_base_ids_by_name() {
        // The schema to re-id (current ids are arbitrary / colliding).
        let schema = Schema::builder()
            .with_schema_id(5)
            .with_fields(vec![
                NestedField::required(50, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(51, "name", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(52, "added", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();
        // Base supplies canonical ids for `id` and `name`; `added` is absent.
        let base = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(10, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(20, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();

        let counter = Cell::new(0_i32);
        let mut next = || -> Result<i32> {
            let n = counter.get() + 1;
            counter.set(n);
            Ok(n)
        };
        let fresh = assign_fresh_ids_with_base(&schema, &base, &mut next).unwrap();

        // Default schema id 0 (Java's `new Schema(fields, ids)` ctor, no id arg).
        assert_eq!(fresh.schema_id(), 0);
        // `id` and `name` reuse the base ids; `added` gets a fresh id from `next_id`.
        assert_eq!(fresh.field_by_name("id").unwrap().id, 10, "reused base id");
        assert_eq!(
            fresh.field_by_name("name").unwrap().id,
            20,
            "reused base id"
        );
        assert_eq!(
            fresh.field_by_name("added").unwrap().id,
            1,
            "unmatched name gets the first fresh id"
        );
    }

    /// `assign_fresh_ids_with_base` walks a NESTED struct in level-order, reusing base ids by full
    /// dotted name and pulling fresh ids for unmatched nested names. Risk: looking up the wrong name
    /// (short vs dotted) or reusing position would assign the wrong nested base id.
    #[test]
    fn test_assign_fresh_ids_with_base_nested_reuse_and_fresh() {
        let schema = Schema::builder()
            .with_schema_id(7)
            .with_fields(vec![
                NestedField::optional(
                    70,
                    "point",
                    Type::Struct(StructType::new(vec![
                        NestedField::required(71, "x", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::required(72, "y", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap();
        // Base matches `point` and `point.x` but NOT `point.y`.
        let base = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::optional(
                    30,
                    "point",
                    Type::Struct(StructType::new(vec![
                        NestedField::required(31, "x", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap();

        let counter = Cell::new(0_i32);
        let mut next = || -> Result<i32> {
            let n = counter.get() + 1;
            counter.set(n);
            Ok(n)
        };
        let fresh = assign_fresh_ids_with_base(&schema, &base, &mut next).unwrap();

        assert_eq!(fresh.field_id_by_name("point"), Some(30), "reused base id");
        assert_eq!(
            fresh.field_id_by_name("point.x"),
            Some(31),
            "reused base id"
        );
        // `point.y` is unmatched -> the first fresh id (1). The struct walk is level-order, so the
        // two immediate ids (point reused=30, then point's child ids) precede the descent, but the
        // ONLY fresh id pulled is for `point.y`.
        assert_eq!(
            fresh.field_id_by_name("point.y"),
            Some(1),
            "unmatched nested name gets a fresh id"
        );
    }

    // ===== reassign family =====

    /// `reassign_ids` aligns ids to a source schema BY NAME (the source's ids win, even nested).
    /// Risk: aligning by position instead of name would assign the wrong id to any reordered or
    /// renamed column.
    #[test]
    fn test_reassign_ids_aligns_by_name() {
        // Source provides the canonical ids.
        let source = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(10, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(
                    20,
                    "point",
                    Type::Struct(StructType::new(vec![
                        NestedField::required(21, "x", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::required(22, "y", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap();

        // Same names, DIFFERENT ids and a different field order.
        let schema = Schema::builder()
            .with_schema_id(5)
            .with_fields(vec![
                NestedField::optional(
                    99,
                    "point",
                    Type::Struct(StructType::new(vec![
                        NestedField::required(98, "x", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::required(97, "y", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .into(),
                NestedField::required(96, "id", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap();

        let reassigned = reassign_ids(&schema, &source).unwrap();
        assert_eq!(reassigned.schema_id(), 5, "schema id preserved");
        assert_eq!(reassigned.field_by_name("id").unwrap().id, 10);
        assert_eq!(reassigned.field_by_name("point").unwrap().id, 20);
        // Nested fields align by their full names too.
        assert_eq!(reassigned.field_id_by_name("point.x"), Some(21));
        assert_eq!(reassigned.field_id_by_name("point.y"), Some(22));
    }

    /// `reassign_ids` is a HARD ERROR when a field has no same-named field in the source (no id
    /// source to fall back on). Risk: silently assigning some id would diverge from Java's throw and
    /// could collide with a real source id.
    #[test]
    fn test_reassign_ids_fails_on_unmatched_name() {
        let source = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(10, "id", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap();
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "extra", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();

        let err = reassign_ids(&schema, &source).expect_err("unmatched name must fail");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("extra") && err.message().contains("not found in source"),
            "message was: {}",
            err.message()
        );
    }

    /// `reassign_ids` is a HARD ERROR on a STRUCTURAL TYPE MISMATCH at a MATCHED name: the visited
    /// field `payload` is a list but the same-named source field is a struct. Java's
    /// `ReassignIds.list` runs `Preconditions.checkArgument(sourceType.isListType(), "Not a list:
    /// %s", sourceType)` and throws `IllegalArgumentException` — it does NOT silently reassign or
    /// fall through to fresh ids. Risk (the Critic's MEDIUM #2): the previous `(other, _)` arm
    /// routed this to `assign_fresh_or_fail`, which for `reassign_or_refresh_ids` SILENTLY assigned
    /// fresh ids (Java throws) and for `reassign_ids` errored with the WRONG "not found" message.
    #[test]
    fn test_reassign_ids_fails_on_matched_name_type_mismatch() {
        // Source: `payload` is a STRUCT.
        let source = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::optional(
                    10,
                    "payload",
                    Type::Struct(StructType::new(vec![
                        NestedField::required(11, "n", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap();
        // Schema: same name `payload`, but a LIST.
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::optional(
                    1,
                    "payload",
                    Type::List(ListType::new(
                        NestedField::list_element(2, Type::Primitive(PrimitiveType::Int), true)
                            .into(),
                    )),
                )
                .into(),
            ])
            .build()
            .unwrap();

        let err = reassign_ids(&schema, &source).expect_err("type mismatch must fail");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("Not a list"),
            "expected Java 'Not a list: %s' message, was: {}",
            err.message()
        );
    }

    /// The SAME structural mismatch under `reassign_or_refresh_ids` (which DOES carry an id source)
    /// must ALSO throw, NOT silently assign fresh ids — Java's `ReassignIds.struct`/`list`/`map`
    /// type guard fires regardless of whether `assignId` is set (the `assignId` fallback is only on
    /// the no-MATCH path in `field()`/`id()`, not on a matched-name type mismatch). Risk: the prior
    /// fall-through let `reassign_or_refresh_ids` accept a list-where-struct silently.
    #[test]
    fn test_reassign_or_refresh_ids_fails_on_matched_name_type_mismatch() {
        let source = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::optional(
                    10,
                    "m",
                    Type::Map(MapType::new(
                        NestedField::map_key_element(11, Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::map_value_element(
                            12,
                            Type::Primitive(PrimitiveType::Int),
                            true,
                        )
                        .into(),
                    )),
                )
                .into(),
            ])
            .build()
            .unwrap();
        // Same name `m`, but a STRUCT (not a map).
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::optional(
                    1,
                    "m",
                    Type::Struct(StructType::new(vec![
                        NestedField::required(2, "a", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap();

        let err = reassign_or_refresh_ids(&schema, &source)
            .expect_err("type mismatch must fail (no silent fresh)");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("Not a struct"),
            "expected Java 'Not a struct: %s' message, was: {}",
            err.message()
        );
    }

    /// `reassign_or_refresh_ids` reuses source ids by name where they match, and assigns FRESH ids
    /// (continuing from the source's highest id) to unmatched names. Risk: not continuing from the
    /// source high-water mark could reuse an id already present in the source (a collision).
    #[test]
    fn test_reassign_or_refresh_ids_assigns_fresh_for_unmatched() {
        let source = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(10, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(15, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();
        // highest_field_id of source is 15.
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "extra", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();

        let result = reassign_or_refresh_ids(&schema, &source).unwrap();
        // `id` matches the source -> id 10. `extra` is fresh, continuing from 15 -> 16.
        assert_eq!(result.field_by_name("id").unwrap().id, 10);
        assert_eq!(result.field_by_name("extra").unwrap().id, 16);
    }

    /// `reassign_doc` copies docs from the doc source BY ID and clears docs absent from the source.
    /// Risk: copying by name/position would attach the wrong comment to a column.
    #[test]
    fn test_reassign_doc_copies_by_id() {
        let doc_source = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long))
                    .with_doc("the primary key")
                    .into(),
                NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();
        // Same ids; field 1 has a stale doc, field 2 has a doc that should be CLEARED (absent in src).
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long))
                    .with_doc("stale")
                    .into(),
                NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String))
                    .with_doc("to be cleared")
                    .into(),
            ])
            .build()
            .unwrap();

        let result = reassign_doc(&schema, &doc_source).unwrap();
        assert_eq!(
            result.field_by_id(1).unwrap().doc.as_deref(),
            Some("the primary key"),
            "doc copied from source by id"
        );
        assert_eq!(
            result.field_by_id(2).unwrap().doc,
            None,
            "doc cleared when the source field has none"
        );
    }

    /// `refresh_identifier_fields` is a HARD ERROR when an identifier name no longer resolves in the
    /// freshly-id'd struct (Java's `Preconditions.checkArgument` "Cannot find ID for identifier
    /// field"). Risk: silently dropping the identifier would lose the primary-key constraint.
    #[test]
    fn test_refresh_identifier_fields_fails_when_name_missing() {
        let source = Schema::builder()
            .with_schema_id(0)
            .with_identifier_field_ids(vec![1])
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap();
        // A new struct that does NOT contain `id`.
        let new_struct = StructType::new(vec![
            NestedField::required(5, "other", Type::Primitive(PrimitiveType::Long)).into(),
        ]);

        let err = refresh_identifier_fields(&new_struct, &source)
            .expect_err("missing identifier name must fail");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message()
                .contains("Cannot find ID for identifier field id"),
            "message was: {}",
            err.message()
        );
    }
}
