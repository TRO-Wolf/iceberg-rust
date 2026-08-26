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
//! `assign_fresh_ids_with_base` and `reassign_doc` have no in-tree caller yet. The create-table
//! and metadata-join consumers will wire them.

use std::cell::Cell;

use super::utils::try_insert_field;
use super::*;

pub struct ReassignFieldIds {
    next_field_id: i32,
    old_to_new_id: HashMap<i32, i32>,
}

// Not the visitor: this walk must re-assign a whole level before it descends, not post-order.
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
        // Pass 1: same-level fields.
        let outer_fields = fields
            .into_iter()
            .map(|field| {
                try_insert_field(&mut self.old_to_new_id, field.id, self.next_field_id)?;
                let new_field = Arc::unwrap_or_clone(field).with_id(self.next_field_id);
                self.increase_next_field_id()?;
                Ok(Arc::new(new_field))
            })
            .collect::<Result<Vec<_>>>()?;

        // Pass 2: nested fields.
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
            // A variant is a leaf. Java `AssignFreshIds.variant` returns the type unchanged.
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

/// A stateful id source. Every call returns a fresh, strictly increasing id. Mirrors Java
/// `TypeUtil.NextID`. Every `assign_fresh_ids` entry point takes one, so one recursion serves
/// every flow.
pub type NextId<'a> = dyn FnMut() -> Result<i32> + 'a;

/// Maximum nesting depth the raw-`Type` entry points descend before they return a typed error.
///
/// A [`Schema`] argument is already bounded by the depth-checked builder. [`assign_fresh_ids`]
/// and [`assign_ids`] take an unvalidated caller-supplied `Type` instead, straight from
/// `UpdateSchemaAction::add_column`.
///
/// `128` matches every other nesting bound in the crate. Depth follows the schema visitor's
/// convention, so `SchemaBuilder::build` rejects one level earlier than this door. The bound can
/// never refuse an otherwise-legal column.
///
/// Java does not bound this recursion and raises `StackOverflowError`. The typed
/// [`ErrorKind::DataInvalid`] is a deliberate divergence, only where Java has no behavior.
const MAX_ASSIGN_IDS_NESTING_DEPTH: usize = 128;

/// The typed refusal for a `Type` nesting deeper than [`MAX_ASSIGN_IDS_NESTING_DEPTH`]. It is
/// worded like the schema visitor's depth error so the two are greppable together.
fn nesting_depth_exceeded() -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        format!(
            "Schema type nesting exceeds maximum depth {MAX_ASSIGN_IDS_NESTING_DEPTH} while assigning field ids"
        ),
    )
}

// assign-ids family: `TypeUtil.assignFreshIds` / `assignIds` / `assignIncreasingFreshIds`.
//
// Java `AssignFreshIds` assigns a parent's ids before its children's, and every immediate struct
// field id before any child type. Ids must match Java byte-for-byte on a round trip.

/// Assign fresh ids to every field in `field_type`, pulling each id from `next_id` in Java
/// `AssignFreshIds` level order. A primitive and a variant carry no ids and pass through.
/// `field_type` is unvalidated, so the walk is bounded by [`MAX_ASSIGN_IDS_NESTING_DEPTH`].
pub fn assign_fresh_ids(field_type: &Type, next_id: &mut NextId<'_>) -> Result<Type> {
    assign_fresh_ids_at_depth(field_type, next_id, 0)
}

/// Depth-bounded body of [`assign_fresh_ids`]. `depth` follows the schema visitor's convention.
fn assign_fresh_ids_at_depth(
    field_type: &Type,
    next_id: &mut NextId<'_>,
    depth: usize,
) -> Result<Type> {
    if depth > MAX_ASSIGN_IDS_NESTING_DEPTH {
        return Err(nesting_depth_exceeded());
    }
    match field_type {
        Type::Primitive(p) => Ok(Type::Primitive(p.clone())),
        Type::Variant => Ok(Type::Variant),
        Type::Struct(s) => Ok(Type::Struct(assign_fresh_ids_to_fields_at_depth(
            s.fields(),
            next_id,
            depth,
        )?)),
        Type::List(l) => {
            // Level-order: the element id is the list's single immediate id; assign it first.
            let element_id = next_id()?;
            let element_type =
                assign_fresh_ids_at_depth(&l.element_field.field_type, next_id, depth + 1)?;
            Ok(Type::List(ListType::new(Arc::new(
                NestedField::list_element(element_id, element_type, l.element_field.required),
            ))))
        }
        Type::Map(m) => {
            // Level-order: assign key id THEN value id (both immediate) before recursing either.
            let key_id = next_id()?;
            let value_id = next_id()?;
            let key_type = assign_fresh_ids_at_depth(&m.key_field.field_type, next_id, depth + 1)?;
            let value_type =
                assign_fresh_ids_at_depth(&m.value_field.field_type, next_id, depth + 1)?;
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

/// The struct body of [`assign_fresh_ids`]. Pass 1 assigns every immediate field id. Pass 2
/// recurses into each field's type. Doc and default attributes survive. It is entered at depth
/// `0` only from [`assign_fresh_ids_to_schema`], whose `Schema` is builder-validated.
fn assign_fresh_ids_to_fields(
    fields: &[NestedFieldRef],
    next_id: &mut NextId<'_>,
) -> Result<StructType> {
    assign_fresh_ids_to_fields_at_depth(fields, next_id, 0)
}

/// Depth-carrying body of [`assign_fresh_ids_to_fields`]. `depth` is the enclosing struct's level.
/// It runs no check: every path back into the recursion goes through
/// [`assign_fresh_ids_at_depth`], which owns the single bound.
fn assign_fresh_ids_to_fields_at_depth(
    fields: &[NestedFieldRef],
    next_id: &mut NextId<'_>,
    depth: usize,
) -> Result<StructType> {
    let new_ids = fields
        .iter()
        .map(|_| next_id())
        .collect::<Result<Vec<_>>>()?;
    let mut new_fields = Vec::with_capacity(fields.len());
    for (field, new_id) in fields.iter().zip(new_ids) {
        let new_type = assign_fresh_ids_at_depth(&field.field_type, next_id, depth + 1)?;
        let mut rebuilt = NestedField::new(new_id, field.name.clone(), new_type, field.required);
        rebuilt.doc = field.doc.clone();
        rebuilt.initial_default = field.initial_default.clone();
        rebuilt.write_default = field.write_default.clone();
        new_fields.push(Arc::new(rebuilt));
    }
    Ok(StructType::new(new_fields))
}

/// Assign fresh ids to a whole schema and stamp it with `schema_id`, which is a schema id, not a
/// field id. [`refresh_identifier_fields`] recomputes the identifier fields by name.
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

/// Assign increasing fresh ids starting at 1, keeping the schema id. Ports
/// `TypeUtil.assignIncreasingFreshIds`, whose first id is 1.
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

/// Assign fresh ids to a whole schema, but reuse the `base` id of any full dotted name present in
/// `base`. Ports `TypeUtil.assignFreshIds(Schema, Schema, NextID)`.
///
/// Java's `idFor` is the only id source:
/// ```text
/// idFor(name):  base.findField(name) != null ? base.findField(name).fieldId() : nextId.get()
/// name(id):     visitingSchema.findColumnName(id)   // the field's current full dotted name
/// ```
/// The struct walk is the same two-pass level order as [`assign_fresh_ids_to_fields`]. The result
/// carries the default schema id `0`, and identifier fields are recomputed by name.
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

/// `idFor(name(field_id))`: reuse the base id of the matching full dotted name, else pull a fresh
/// id. `field_id` is the field's current id in `visiting`.
fn base_id_for(
    field_id: i32,
    visiting: &Schema,
    base: &Schema,
    next_id: &mut NextId<'_>,
) -> Result<i32> {
    let name: Option<String> = visiting.name_by_field_id(field_id).map(str::to_owned);
    if let Some(name) = name
        && let Some(reused) = base.field_id_by_name(&name)
    {
        return Ok(reused);
    }
    next_id()
}

/// Struct body of [`assign_fresh_ids_with_base`]. Pass 1 assigns every immediate id through
/// [`base_id_for`]. Pass 2 recurses into the children. Java `AssignFreshIds.struct` has this shape.
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

/// Recurse [`assign_fresh_ids_with_base_struct`] into a nested type. An element, key, or value id
/// reuses the base id of the same name, else pulls fresh. It is deliberately not depth-bounded:
/// only [`assign_fresh_ids_with_base`] reaches it, and its `Schema` inputs are already checked.
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

/// Re-key a type's ids through `get_id`. Ports `TypeUtil.assignIds(Type, GetID)`. The structure
/// survives and every id goes through `get_id`, list element and map key and value included. The
/// caller handles a missing mapping. The walk is bounded by [`MAX_ASSIGN_IDS_NESTING_DEPTH`].
pub fn assign_ids(field_type: &Type, get_id: &mut dyn FnMut(i32) -> i32) -> Result<Type> {
    assign_ids_at_depth(field_type, get_id, 0)
}

/// Depth-bounded body of [`assign_ids`]; `depth` follows the same convention as
/// [`assign_fresh_ids_at_depth`].
fn assign_ids_at_depth(
    field_type: &Type,
    get_id: &mut dyn FnMut(i32) -> i32,
    depth: usize,
) -> Result<Type> {
    if depth > MAX_ASSIGN_IDS_NESTING_DEPTH {
        return Err(nesting_depth_exceeded());
    }
    match field_type {
        Type::Primitive(p) => Ok(Type::Primitive(p.clone())),
        Type::Variant => Ok(Type::Variant),
        Type::Struct(s) => {
            let mut new_fields = Vec::with_capacity(s.fields().len());
            for field in s.fields() {
                let new_id = get_id(field.id);
                let new_type = assign_ids_at_depth(&field.field_type, get_id, depth + 1)?;
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
            let element_type = assign_ids_at_depth(&l.element_field.field_type, get_id, depth + 1)?;
            Ok(Type::List(ListType::new(Arc::new(
                NestedField::list_element(element_id, element_type, l.element_field.required),
            ))))
        }
        Type::Map(m) => {
            let key_id = get_id(m.key_field.id);
            let value_id = get_id(m.value_field.id);
            let key_type = assign_ids_at_depth(&m.key_field.field_type, get_id, depth + 1)?;
            let value_type = assign_ids_at_depth(&m.value_field.field_type, get_id, depth + 1)?;
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

// reassign family: `TypeUtil.reassignIds` / `reassignOrRefreshIds` / `reassignDoc` /
// `refreshIdentifierFields`.
//
// Java `ReassignIds` walks `schema`, tracks the position in a source schema, and aligns ids by
// name. At an unmatched name, `None` is an error and a `NextId` assigns fresh subtree ids.

/// Align the ids of `schema` to `id_source` by name, case-sensitive. Ports
/// `TypeUtil.reassignIds(Schema, Schema)`. A name absent from the source is a hard error.
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

/// Align the ids of `schema` to `id_source` by name. An unmatched name gets a fresh id,
/// continuing from `id_source.highest_field_id()`. Ports `TypeUtil.reassignOrRefreshIds`.
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

/// Copy field docs from `doc_source` onto `schema` by id. A same-id source doc replaces the doc.
/// Every other doc is cleared. The result keeps `schema.schema_id()`, where Java would give the
/// default `0`, because callers here expect the schema id to survive.
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

/// Recurse [`reassign_doc_struct`] into nested types. Java `ReassignDoc` does not carry element,
/// key, or value docs, so they stay as they are.
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

/// Recompute a freshly-id'd struct's identifier-field ids by carrying the source identifier NAMES
/// across. Ports `TypeUtil.refreshIdentifierFields`. A name that no longer resolves is a hard
/// error.
pub fn refresh_identifier_fields(
    new_struct: &StructType,
    source_schema: &Schema,
) -> Result<HashSet<i32>> {
    // Index by full dotted name, as Java does, so a nested identifier field resolves.
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

/// Java `ReassignIds` as an explicit recursive walk over `schema`, tracking the position in
/// `source`. A matching same-named source field supplies the id and the child source position. At
/// an unmatched name, `assign_id` assigns fresh ids to the subtree, or the absence is an error.
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
        // Two phases, as Java `ReassignIds.struct` has. Phase 1 computes every child type, which
        // pulls the fresh subtree ids in field order. Phase 2 assigns each field's own id.
        // Interleaving the phases would change the fresh-id stream for an unmatched subtree.
        let matched: Vec<Option<NestedFieldRef>> = struct_type
            .fields()
            .iter()
            .map(|field| self.source_field(source_struct, &field.name).cloned())
            .collect();

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

        let mut new_fields = Vec::with_capacity(struct_type.fields().len());
        for ((field, source_field), new_type) in
            struct_type.fields().iter().zip(matched).zip(new_types)
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

    /// Reassign ids in `field_type` against `source_type`, the same-named source field's type. A
    /// nested container takes its element, key, and value ids from the source container.
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
            // A primitive or variant carries no nested ids and passes through whatever the
            // source type is. Java runs no `sourceType` check on those arms.
            (Type::Primitive(_) | Type::Variant, _) => Ok(field_type.clone()),
            // A structural mismatch at a MATCHED name, such as a list against a source struct.
            // Java throws here and does NOT assign fresh ids. Mirror its message exactly.
            (Type::Struct(_), _) => Err(Self::not_a_type_error("struct", source_type)),
            (Type::List(_), _) => Err(Self::not_a_type_error("list", source_type)),
            (Type::Map(_), _) => Err(Self::not_a_type_error("map", source_type)),
        }
    }

    /// Build the structural-mismatch error Java throws: `Not a struct/list/map: <sourceType>`.
    /// `kind` is the visited container kind. `source` is the mismatching source type.
    fn not_a_type_error(kind: &str, source: &Type) -> Error {
        Error::new(ErrorKind::DataInvalid, format!("Not a {kind}: {source}"))
    }

    /// Assign fresh ids to a whole unmatched subtree, or fail when there is no id source.
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

    // Risk: reassignment must treat variant as a LEAF. The column's own id is reassigned, the
    // type passes through, and sibling ids after it stay in sequence. A missing arm makes
    // `with_reassigned_field_ids` fail on every variant schema.
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

    /// `assign_fresh_ids` over list-of-map assigns the element id before the map key and value
    /// ids, and the key id before the value id. A depth-first walk would interleave them.
    #[test]
    fn test_assign_fresh_ids_level_order_nested() {
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

        // Element id 1 first, then key 2, then value 3.
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

    /// A nested struct followed by a sibling takes every immediate id first, then the child id.
    /// The mutation this discriminates: a depth-first walk yields a=1, x=2, b=3, not a=1, b=2,
    /// x=3. The list-of-map test has no struct sibling and cannot catch it.
    #[test]
    fn test_assign_fresh_ids_level_order_struct_siblings() {
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
        // Both immediate ids (a=1, b=2) precede the nested x=3.
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

    /// Recomputing identifier fields by name keeps the table's primary-key declaration.
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
        // Fresh ids 1 and 2 in field order. The identifier follows `id` to its new id 1.
        assert_eq!(fresh.field_by_name("id").unwrap().id, 1);
        assert_eq!(fresh.field_by_name("name").unwrap().id, 2);
        let ids: Vec<i32> = fresh.identifier_field_ids().collect();
        assert_eq!(ids, vec![1]);
    }

    /// `assign_increasing_fresh_ids` starts the id stream at 1. Starting at 0 shifts every id.
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

    /// `assign_ids` rewrites every id and keeps the structure. A missed list or map id dangles.
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

    /// The base id of a matching dotted name is reused; ignoring it re-numbers a matched column.
    #[test]
    fn test_assign_fresh_ids_with_base_reuses_base_ids_by_name() {
        let schema = Schema::builder()
            .with_schema_id(5)
            .with_fields(vec![
                NestedField::required(50, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(51, "name", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(52, "added", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();
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

        assert_eq!(fresh.schema_id(), 0);
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

    /// A short-name lookup or a positional match gives a nested field the wrong base id.
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
        // `point.y` is unmatched, so it takes the first fresh id (1). It is the only fresh id.
        assert_eq!(
            fresh.field_id_by_name("point.y"),
            Some(1),
            "unmatched nested name gets a fresh id"
        );
    }

    /// Aligning by position, not name, gives a reordered or renamed column the wrong id.
    #[test]
    fn test_reassign_ids_aligns_by_name() {
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
        assert_eq!(reassigned.field_id_by_name("point.x"), Some(21));
        assert_eq!(reassigned.field_id_by_name("point.y"), Some(22));
    }

    /// A name absent from the source is a hard error. A silent id could collide with a source id.
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

    /// A structural mismatch at a matched name is a hard error: `payload` is a list here and a
    /// struct in the source. The mutation this discriminates: route the mismatch to
    /// `assign_fresh_or_fail`, which assigns fresh ids or raises the wrong `not found` message.
    #[test]
    fn test_reassign_ids_fails_on_matched_name_type_mismatch() {
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

    /// The same mismatch must also fail under `reassign_or_refresh_ids`, which carries an id
    /// source. The `assignId` fallback covers the no-match path only.
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

    /// Fresh ids must continue from the source's highest id, or they collide with a source id.
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
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "extra", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();

        let result = reassign_or_refresh_ids(&schema, &source).unwrap();
        // `id` matches the source and takes 10. `extra` is fresh and continues from 15 to 16.
        assert_eq!(result.field_by_name("id").unwrap().id, 10);
        assert_eq!(result.field_by_name("extra").unwrap().id, 16);
    }

    /// Docs copy by id. Copying by name or position attaches the wrong comment to a column.
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

    /// An identifier name that no longer resolves is a hard error, not a dropped constraint.
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

    // ===== recursion safety: the caller-supplied raw `Type` doors =====

    /// `depth` nested single-field structs around a boolean leaf, built iteratively so the
    /// fixture never recurses. `UpdateSchemaAction::add_column` accepts this shape.
    fn deeply_nested_struct_type(depth: usize) -> Type {
        let mut field_type = Type::Primitive(PrimitiveType::Boolean);
        for level in (1..=depth).rev() {
            let id = i32::try_from(level).expect("test depth fits i32");
            field_type = Type::Struct(StructType::new(vec![
                NestedField::optional(id, "nested", field_type).into(),
            ]));
        }
        field_type
    }

    /// Run `call` on a thread with a known, bounded stack. `field_type` goes in and comes back
    /// out, so the deep fixture's recursive DROP runs on the harness stack, not this one.
    ///
    /// 3 MiB comes from a measurement. Bisecting `stack_size` in the dev profile, the bounded walk
    /// overflows at 1152 KiB and succeeds at 1280 KiB, so it needs about 1.25 MiB. The 4096 levels
    /// these tests feed in need about 40 MiB unbounded. 3 MiB therefore passes with the guard and
    /// aborts on `fatal runtime error: stack overflow` without it.
    fn on_bounded_stack(field_type: Type, call: fn(&Type) -> Result<Type>) -> (Type, Result<Type>) {
        std::thread::Builder::new()
            .name("assign-ids-bounded-stack".to_string())
            .stack_size(3 * 1024 * 1024)
            .spawn(move || {
                let result = call(&field_type);
                (field_type, result)
            })
            .expect("spawn the bounded-stack assign-ids thread")
            .join()
            .expect("assigning ids must not overflow or panic")
    }

    fn increasing_from_one(field_type: &Type) -> Result<Type> {
        let counter = Cell::new(0_i32);
        let mut next = || -> Result<i32> {
            let n = counter.get() + 1;
            counter.set(n);
            Ok(n)
        };
        assign_fresh_ids(field_type, &mut next)
    }

    fn identity_remap(field_type: &Type) -> Result<Type> {
        let mut get_id = |old: i32| old;
        assign_ids(field_type, &mut get_id)
    }

    /// Assert a depth refusal, whatever produced it.
    fn assert_depth_error(error: &Error) {
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains(
                "Schema type nesting exceeds maximum depth 128 while assigning field ids"
            ),
            "message was: {}",
            error.message()
        );
    }

    // Risk: `assign_fresh_ids` takes the unvalidated `Type` of `UpdateSchemaAction::add_column`.
    // Unbounded, a hostile struct chain overflows the thread stack. The mutation this
    // discriminates: delete the depth guard from `assign_fresh_ids_at_depth`, which aborts.
    #[test]
    fn assign_fresh_ids_rejects_hostile_nesting_instead_of_overflowing() {
        let (deep, result) = on_bounded_stack(deeply_nested_struct_type(4096), increasing_from_one);
        let error = result.expect_err("a 4096-deep struct chain must be rejected");

        assert_depth_error(&error);
        drop(deep);
    }

    // Risk: `assign_ids` is the second raw-`Type` door and needs the same bound.
    // The mutation this discriminates: delete the guard from `assign_ids_at_depth`.
    #[test]
    fn assign_ids_rejects_hostile_nesting_instead_of_overflowing() {
        let (deep, result) = on_bounded_stack(deeply_nested_struct_type(4096), identity_remap);
        let error = result.expect_err("a 4096-deep struct chain must be rejected");

        assert_depth_error(&error);
        drop(deep);
    }

    // Risk: the bound must sit at exactly `MAX_ASSIGN_IDS_NESTING_DEPTH`, so it never refuses a
    // column `SchemaBuilder::build` accepts. The mutations this discriminates: an off-by-one in
    // either direction (`>=` for `>`, or `depth + 1` into the fields walk), and a new constant.
    #[test]
    fn assign_fresh_ids_depth_bound_is_exactly_the_family_constant() {
        assert_eq!(
            MAX_ASSIGN_IDS_NESTING_DEPTH, 128,
            "the bound must stay in lockstep with MAX_SCHEMA_NESTING_DEPTH / \
             MAX_AVRO_SCHEMA_DEPTH / MAX_ARROW_SCHEMA_NESTING_DEPTH / MAX_NESTING_DEPTH"
        );

        let at_bound = deeply_nested_struct_type(MAX_ASSIGN_IDS_NESTING_DEPTH);
        let fresh = increasing_from_one(&at_bound)
            .expect("nesting exactly at the bound must still be assigned");
        let Type::Struct(outer) = fresh else {
            panic!("expected the outermost struct back")
        };
        assert_eq!(outer.fields()[0].id, 1, "ids are still assigned normally");

        let over_bound = deeply_nested_struct_type(MAX_ASSIGN_IDS_NESTING_DEPTH + 1);
        let error =
            increasing_from_one(&over_bound).expect_err("one level past the bound is rejected");
        assert_depth_error(&error);
    }

    // Risk: the bound must count LIST and MAP nesting. A container-only chain reaches the same
    // recursion through other match arms, and a lost `depth + 1` on one arm leaves it unbounded
    // while every struct test stays green. One chain per container arm pins each independently.
    //
    // The mutations this discriminates: drop the `depth + 1` on the list-element arm, the map-KEY
    // arm, or the map-VALUE arm. Each turns exactly one `expect_err` below RED.
    //
    // A map key may itself be a nested type. Neither `MapType::new` nor Java restricts it, so
    // `map<map<...>,v>` reaches the key arm from `UpdateSchemaAction::add_column`.
    #[test]
    fn assign_fresh_ids_bounds_list_and_map_nesting_too() {
        let mut list_chain = Type::Primitive(PrimitiveType::Boolean);
        for level in (1..=(MAX_ASSIGN_IDS_NESTING_DEPTH + 1)).rev() {
            let id = i32::try_from(level).expect("test depth fits i32");
            list_chain = Type::List(ListType::new(
                NestedField::list_element(id, list_chain, false).into(),
            ));
        }
        let error = increasing_from_one(&list_chain).expect_err("deep list chain must be rejected");
        assert_depth_error(&error);

        // Nested through the value position. Every key is a shallow primitive.
        let mut map_value_chain = Type::Primitive(PrimitiveType::Boolean);
        for level in (1..=(MAX_ASSIGN_IDS_NESTING_DEPTH + 1)).rev() {
            let key_id = i32::try_from(2 * level).expect("test depth fits i32");
            map_value_chain = Type::Map(MapType::new(
                NestedField::map_key_element(key_id, Type::Primitive(PrimitiveType::String)).into(),
                NestedField::map_value_element(key_id + 1, map_value_chain, false).into(),
            ));
        }
        let error = increasing_from_one(&map_value_chain)
            .expect_err("deep map chain nested through VALUES must be rejected");
        assert_depth_error(&error);

        // Nested through the key position. Without this chain the key arm is unpinned: the value
        // chain above has primitive keys that return before any depth accumulates.
        let mut map_key_chain = Type::Primitive(PrimitiveType::String);
        for level in (1..=(MAX_ASSIGN_IDS_NESTING_DEPTH + 1)).rev() {
            let key_id = i32::try_from(2 * level).expect("test depth fits i32");
            map_key_chain = Type::Map(MapType::new(
                NestedField::map_key_element(key_id, map_key_chain).into(),
                NestedField::map_value_element(
                    key_id + 1,
                    Type::Primitive(PrimitiveType::String),
                    false,
                )
                .into(),
            ));
        }
        let error = increasing_from_one(&map_key_chain)
            .expect_err("deep map chain nested through KEYS must be rejected");
        assert_depth_error(&error);
    }
}
