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

use std::collections::{HashMap, HashSet};

use super::prune_columns::prune_columns_struct;
use super::visitor::{SchemaVisitor, visit_struct};
use crate::spec::datatypes::{ListType, MapType, NestedFieldRef, PrimitiveType, StructType, Type};
use crate::spec::schema::Schema;
use crate::{Error, ErrorKind, Result};

pub fn try_insert_field<V>(map: &mut HashMap<i32, V>, field_id: i32, value: V) -> Result<()> {
    map.insert(field_id, value).map_or_else(
        || Ok(()),
        |_| {
            Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Found duplicate 'field.id' {field_id}. Field ids must be unique."),
            ))
        },
    )
}

// =====================================================================================
// Projection family — `TypeUtil.project` / `select` / `selectNot` / `getProjectedIds`.
//
// `project` and `select` differ in ONE knob (`PruneColumns.selectFullTypes`): both keep exactly
// the selected ids and the parents needed to reach them, but `select` additionally pulls in the
// FULL subtree of any selected nested field while `project` keeps only the selected ids. The
// engine is the existing `prune_columns` visitor; these are the thin `TypeUtil` wrappers.
// =====================================================================================

/// Project `struct_type` down to exactly the fields in `ids` (plus the parents needed to reach
/// them). Rust port of `TypeUtil.project(StructType, Set<Integer>)` — i.e. `PruneColumns` with
/// `selectFullTypes = false`. A nested field that is selected keeps only its selected descendants.
pub fn project_struct(struct_type: &StructType, ids: &HashSet<i32>) -> Result<StructType> {
    prune_columns_struct(struct_type, ids.iter().copied(), false)
}

/// Project `schema` down to exactly the fields in `ids`. Rust port of `TypeUtil.project(Schema,
/// Set<Integer>)`; the result carries the same schema id as `schema`.
pub fn project(schema: &Schema, ids: &HashSet<i32>) -> Result<Schema> {
    let projected = project_struct(schema.as_struct(), ids)?;
    Schema::builder()
        .with_schema_id(schema.schema_id())
        .with_fields(projected.fields().iter().cloned())
        .build()
}

/// Select the fields in `ids` from `struct_type`, including the FULL subtree of any selected nested
/// field. Rust port of `TypeUtil.select(StructType, Set<Integer>)` — i.e. `PruneColumns` with
/// `selectFullTypes = true`. This is the distinction from [`project_struct`]: selecting a struct id
/// here keeps the whole struct, whereas `project` keeps only the explicitly selected descendants.
pub fn select_struct(struct_type: &StructType, ids: &HashSet<i32>) -> Result<StructType> {
    prune_columns_struct(struct_type, ids.iter().copied(), true)
}

/// Select the fields in `ids` from `schema` (full subtrees for selected nested fields). Rust port
/// of `TypeUtil.select(Schema, Set<Integer>)`; the result carries the same schema id as `schema`.
pub fn select(schema: &Schema, ids: &HashSet<i32>) -> Result<Schema> {
    let selected = select_struct(schema.as_struct(), ids)?;
    Schema::builder()
        .with_schema_id(schema.schema_id())
        .with_fields(selected.fields().iter().cloned())
        .build()
}

/// Project `struct_type` keeping everything EXCEPT the ids in `excluded`. Rust port of
/// `TypeUtil.selectNot(StructType, Set<Integer>)`: compute all projected ids (without struct ids),
/// remove the excluded ids, then project. Because the struct ids are excluded from the base set, a
/// nested field is dropped only when ALL of its leaf descendants are excluded.
pub fn select_not_struct(struct_type: &StructType, excluded: &HashSet<i32>) -> Result<StructType> {
    // getIdsInternal(struct, includeStructIds = false): the leaf-and-container ids minus struct ids.
    let mut ids = get_ids_internal(&Type::Struct(struct_type.clone()), false)?;
    for id in excluded {
        ids.remove(id);
    }
    project_struct(struct_type, &ids)
}

/// Project `schema` keeping everything EXCEPT the ids in `excluded`. Rust port of
/// `TypeUtil.selectNot(Schema, Set<Integer>)`.
pub fn select_not(schema: &Schema, excluded: &HashSet<i32>) -> Result<Schema> {
    let mut ids = get_ids_internal(&Type::Struct(schema.as_struct().clone()), false)?;
    for id in excluded {
        ids.remove(id);
    }
    project(schema, &ids)
}

/// The full set of projected ids for `schema` (INCLUDING struct ids). Rust port of
/// `TypeUtil.getProjectedIds(Schema)` — `getIdsInternal(schema.asStruct(), includeStructIds=true)`.
pub fn get_projected_ids_schema(schema: &Schema) -> Result<HashSet<i32>> {
    get_ids_internal(&Type::Struct(schema.as_struct().clone()), true)
}

/// The full set of projected ids for a type (INCLUDING struct ids). Rust port of
/// `TypeUtil.getProjectedIds(Type)`: a primitive contributes no ids; any other type contributes the
/// ids of all of its fields (struct fields, list elements, map keys/values) recursively.
pub fn get_projected_ids(field_type: &Type) -> Result<HashSet<i32>> {
    if field_type.is_primitive() {
        return Ok(HashSet::new());
    }
    get_ids_internal(field_type, true)
}

/// Rust port of Java's `GetProjectedIds` (`TypeUtil.getIdsInternal`), faithful to the 1.10.0
/// bytecode (`javap -p -c org.apache.iceberg.types.GetProjectedIds`).
///
/// `GetProjectedIds` is an eager `SchemaVisitor<Set<Integer>>` whose every method returns the SAME
/// mutable set EXCEPT `primitive()` and `variant()`, which return `null`. The eager dispatch in
/// `TypeUtil.visit(Type, SchemaVisitor)` calls `field(field, visit(field.type))` for each struct
/// field, `list(listType, visit(elementType))`, and `map(mapType, visit(keyType), visit(valueType))`.
/// Two rules fall out of this:
///
/// - **`field`** adds the field's OWN id iff `(includeStructIds && type.isStructType()) ||
///   type.isPrimitiveType() || type.isVariantType()`. A LIST/MAP field's own id is therefore NEVER
///   added (the `&& isStructType` guard is load-bearing — finding mustFix #2). Its descendant ids are
///   contributed by the nested `list()`/`map()`.
/// - **`list`** adds the element FIELD id only when the element-type visit returned `null`, i.e. the
///   element type is a primitive or a variant; for a struct/list/map element type the element id is
///   omitted. **`map`** adds the key AND value field ids only when the VALUE-type visit returned
///   `null` (the value type is a primitive or variant) — gated on the value, not the key.
///
/// Verified against live Java 1.10.0: `getProjectedIds({1, 2:struct{3,4}, 5:list<6 str>})=[1,2,3,4,6]`
/// (struct id 2 in, list id 5 out, element id 6 in); `getProjectedIds({1, 6:map<7,8>})=[1,7,8]`;
/// `getProjectedIds({1, 2:list<3:struct{4}>})=[1,4]`; `getProjectedIds({1, 2:map<3 str, 4:struct{5}>})
/// =[1,5]`; `getProjectedIds({1, 2:map<3:struct{5}, 4 int>})=[1,3,4,5]`; `{1, 2:list<3 variant>}=
/// [1,3]`.
fn get_ids_internal(field_type: &Type, include_struct_ids: bool) -> Result<HashSet<i32>> {
    let mut ids = HashSet::new();
    collect_ids(field_type, include_struct_ids, &mut ids);
    Ok(ids)
}

/// Whether visiting `field_type` returns `null` in Java's `GetProjectedIds` — true only for
/// primitives and variants (their visitor methods `return null`); every nested type returns the
/// shared set (non-null). This is the gate `list()`/`map()` use to decide whether to add their
/// element / key+value field ids.
fn projected_visit_is_null(field_type: &Type) -> bool {
    field_type.is_primitive() || field_type.is_variant()
}

/// The eager recursion mirroring `TypeUtil.visit(Type, GetProjectedIds)`: walk `field_type`,
/// inserting ids into `ids` per the `field`/`list`/`map` rules above. Structs/lists/maps add no id of
/// their own here (only `field`/`list`/`map` add ids, and they are reached via the parent); the
/// top-level call therefore contributes nothing for a bare struct's own id, matching Java.
fn collect_ids(field_type: &Type, include_struct_ids: bool, ids: &mut HashSet<i32>) {
    match field_type {
        Type::Struct(s) => {
            for field in s.fields() {
                // Java: result = visit(field.type); then field(field, result).
                if (include_struct_ids && field.field_type.is_struct())
                    || field.field_type.is_primitive()
                    || field.field_type.is_variant()
                {
                    ids.insert(field.id);
                }
                collect_ids(&field.field_type, include_struct_ids, ids);
            }
        }
        Type::List(l) => {
            // Java list(): add the element FIELD id only when visit(elementType) returned null.
            if projected_visit_is_null(&l.element_field.field_type) {
                ids.insert(l.element_field.id);
            }
            collect_ids(&l.element_field.field_type, include_struct_ids, ids);
        }
        Type::Map(m) => {
            // Java map(): add BOTH key and value field ids only when visit(valueType) returned null.
            if projected_visit_is_null(&m.value_field.field_type) {
                ids.insert(m.key_field.id);
                ids.insert(m.value_field.id);
            }
            collect_ids(&m.key_field.field_type, include_struct_ids, ids);
            collect_ids(&m.value_field.field_type, include_struct_ids, ids);
        }
        Type::Primitive(_) | Type::Variant => {}
    }
}

// =====================================================================================
// `join`, `estimate_size`, `index_quoted_name_by_id` — the cheap pure `TypeUtil` peripherals.
// =====================================================================================

/// Concatenate the TOP-LEVEL columns of two schemas. Rust port of `TypeUtil.join(Schema, Schema)`:
/// keep all columns of `left`, then append each column of `right` whose id is not already present;
/// a shared id whose columns DIFFER is a hard error (Java's `IllegalArgumentException("Schemas have
/// different columns with same id: %s, %s")`). The result carries `left`'s schema id.
pub fn join(left: &Schema, right: &Schema) -> Result<Schema> {
    let mut columns: Vec<NestedFieldRef> = left.as_struct().fields().to_vec();
    for right_field in right.as_struct().fields() {
        match left.field_by_id(right_field.id) {
            None => columns.push(right_field.clone()),
            Some(left_field) => {
                if left_field != right_field {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Schemas have different columns with same id: {left_field:?}, {right_field:?}"
                        ),
                    ));
                }
            }
        }
    }
    Schema::builder()
        .with_schema_id(left.schema_id())
        .with_fields(columns)
        .build()
}

/// Estimate the in-memory byte size of a top-level field. Rust port of
/// `TypeUtil.estimateSize(NestedField)` — delegates to the per-type estimate of the field's type.
pub fn estimate_size(field: &NestedFieldRef) -> i32 {
    estimate_type_size(&field.field_type)
}

/// Per-type byte-size estimate. Rust port of the private `TypeUtil.estimateSize(Type)` switch
/// (decoded from the 1.10.0 `TypeUtil$1` `$SwitchMap` + the `estimateSize` `tableswitch`):
///
/// | type | bytes |
/// |---|---|
/// | boolean | 1 |
/// | int / float / date | 4 |
/// | long / double / time / timestamp / timestamp_ns | 8 |
/// | string | 54 |
/// | uuid | 28 |
/// | fixed(n) | n |
/// | binary / variant | 80 |
/// | decimal | 44 |
/// | unknown | 0 |
/// | struct | 12 + Σ field sizes |
/// | list | 12 + 5 * element size |
/// | map | 12 + 5 * (12 + key + value size) |
///
/// Rust's timestamp-with-tz arms (`timestamptz` / `timestamptz_ns`) share Java's single
/// `TIMESTAMP` / `TIMESTAMP_NANO` `TypeID` (Java models the tz via a flag, not a distinct id), so
/// they map to 8 like their no-tz counterparts. Java's `geometry` / `geography` map to explicit
/// cases (`TypeUtil$1` GEOMETRY/GEOGRAPHY ordinals -> `tableswitch` case 16 / case 17 -> bipush 80),
/// NOT the `default`; the `default` (16) is a dead fall-through unreachable for any real `TypeID`.
/// Geometry/geography have no Rust representation today, so neither arm appears here.
fn estimate_type_size(field_type: &Type) -> i32 {
    match field_type {
        Type::Primitive(p) => match p {
            PrimitiveType::Boolean => 1,
            PrimitiveType::Int | PrimitiveType::Float | PrimitiveType::Date => 4,
            PrimitiveType::Long
            | PrimitiveType::Double
            | PrimitiveType::Time
            | PrimitiveType::Timestamp
            | PrimitiveType::Timestamptz
            | PrimitiveType::TimestampNs
            | PrimitiveType::TimestamptzNs => 8,
            PrimitiveType::String => 54,
            PrimitiveType::Uuid => 28,
            PrimitiveType::Fixed(len) => i32::try_from(*len).unwrap_or(i32::MAX),
            PrimitiveType::Binary => 80,
            PrimitiveType::Decimal { .. } => 44,
            PrimitiveType::Unknown => 0,
        },
        // Java's VARIANT estimate is 80 (TypeUtil$1 case 15 -> bipush 80).
        Type::Variant => 80,
        Type::Struct(s) => {
            let mut size: i32 = 12;
            for field in s.fields() {
                size = size.saturating_add(estimate_type_size(&field.field_type));
            }
            size
        }
        Type::List(l) => {
            let element = estimate_type_size(&l.element_field.field_type);
            12_i32.saturating_add(element.saturating_mul(5))
        }
        Type::Map(m) => {
            let key = estimate_type_size(&m.key_field.field_type);
            let value = estimate_type_size(&m.value_field.field_type);
            // Java: 12 + 5 * (12 + key + value). NOTE the inner per-entry 12 — the list arm has NO
            // inner 12 (`12 + 5*element`); this map-vs-list asymmetry is faithful to the 1.10.0
            // `TypeUtil.estimateSize` bytecode (map arm: `bipush 12; +key; +value; bipush 12;
            // iconst_5; imul; iadd`).
            let inner = 12_i32.saturating_add(key).saturating_add(value);
            12_i32.saturating_add(inner.saturating_mul(5))
        }
    }
}

/// Build an id → quoted-full-name index for `struct_type`, applying `quote` to each name segment.
/// Rust port of `TypeUtil.indexQuotedNameById(StructType, Function<String, String>)` — like the
/// id→name index, but each dotted segment is passed through the `quote` function (so e.g. a SQL
/// quoting function can backtick-quote reserved names).
pub fn index_quoted_name_by_id(
    struct_type: &StructType,
    quote: impl Fn(&str) -> String,
) -> Result<HashMap<i32, String>> {
    struct IndexQuotedNameById<F: Fn(&str) -> String> {
        quote: F,
        segments: Vec<String>,
        result: HashMap<i32, String>,
    }

    impl<F: Fn(&str) -> String> IndexQuotedNameById<F> {
        fn add(&mut self, name: &str, field_id: i32) {
            let quoted = (self.quote)(name);
            let full = self
                .segments
                .iter()
                .cloned()
                .chain(std::iter::once(quoted))
                .collect::<Vec<_>>()
                .join(".");
            self.result.insert(field_id, full);
        }
    }

    impl<F: Fn(&str) -> String> SchemaVisitor for IndexQuotedNameById<F> {
        type T = ();

        fn before_struct_field(&mut self, field: &NestedFieldRef) -> Result<()> {
            self.segments.push((self.quote)(&field.name));
            Ok(())
        }

        fn after_struct_field(&mut self, _field: &NestedFieldRef) -> Result<()> {
            self.segments.pop();
            Ok(())
        }

        fn schema(&mut self, _schema: &Schema, _value: ()) -> Result<()> {
            Ok(())
        }

        fn field(&mut self, field: &NestedFieldRef, _value: ()) -> Result<()> {
            // `field` is the POST-order callback: `after_struct_field` has already popped this
            // field's segment, so `segments` now holds only the parent path. Add the entry using
            // the parent segments + this field's quoted name (mirrors `IndexByName::add_field`).
            self.add(&field.name, field.id);
            Ok(())
        }

        fn r#struct(&mut self, _struct: &StructType, _results: Vec<()>) -> Result<()> {
            Ok(())
        }

        fn list(&mut self, list: &ListType, _value: ()) -> Result<()> {
            self.add(
                crate::spec::datatypes::LIST_FIELD_NAME,
                list.element_field.id,
            );
            Ok(())
        }

        fn map(&mut self, map: &MapType, _key_value: (), _value: ()) -> Result<()> {
            self.add(crate::spec::datatypes::MAP_KEY_FIELD_NAME, map.key_field.id);
            self.add(
                crate::spec::datatypes::MAP_VALUE_FIELD_NAME,
                map.value_field.id,
            );
            Ok(())
        }

        fn primitive(&mut self, _: &PrimitiveType) -> Result<()> {
            Ok(())
        }

        fn variant(&mut self) -> Result<()> {
            Ok(())
        }
    }

    let mut visitor = IndexQuotedNameById {
        quote,
        segments: Vec::new(),
        result: HashMap::new(),
    };
    visit_struct(struct_type, &mut visitor)?;
    Ok(visitor.result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::datatypes::{ListType, NestedField, StructType};

    /// Build the nested test schema:
    /// `{ 1: id (req int), 2: point (opt struct< 3: x int, 4: y int >), 5: tags (opt list< string >) }`.
    fn nested_schema() -> Schema {
        Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    2,
                    "point",
                    Type::Struct(StructType::new(vec![
                        NestedField::required(3, "x", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::required(4, "y", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .into(),
                NestedField::optional(
                    5,
                    "tags",
                    Type::List(ListType::new(
                        NestedField::list_element(6, Type::Primitive(PrimitiveType::String), true)
                            .into(),
                    )),
                )
                .into(),
            ])
            .build()
            .unwrap()
    }

    /// `project` selecting the struct id alone keeps ONLY the necessary parent (an empty struct,
    /// since no descendant is selected) — the key project-vs-select distinction. Risk: returning the
    /// full subtree here would conflate `project` with `select`.
    #[test]
    fn test_project_struct_id_alone_keeps_empty_struct() {
        let schema = nested_schema();
        let ids = HashSet::from([2]); // point only, no descendant
        let projected = project(&schema, &ids).unwrap();
        let point = projected
            .field_by_name("point")
            .expect("point kept as parent");
        let Type::Struct(s) = point.field_type.as_ref() else {
            panic!("point must remain a struct")
        };
        assert!(
            s.fields().is_empty(),
            "project keeps only selected descendants; none selected => empty struct"
        );
        assert!(
            projected.field_by_name("id").is_none(),
            "unselected id dropped"
        );
    }

    /// `select` selecting the struct id pulls the FULL subtree (both x and y) — the contrast to
    /// `project`. Risk: pruning to an empty struct here would lose the selected nested columns.
    #[test]
    fn test_select_struct_id_keeps_full_subtree() {
        let schema = nested_schema();
        let ids = HashSet::from([2]); // point only
        let selected = select(&schema, &ids).unwrap();
        let point = selected.field_by_name("point").expect("point kept");
        let Type::Struct(s) = point.field_type.as_ref() else {
            panic!("point must be a struct")
        };
        assert_eq!(
            s.fields().len(),
            2,
            "select keeps the full subtree (x and y)"
        );
    }

    /// `project` selecting a nested leaf keeps the parent struct AND the selected leaf only. Risk:
    /// dropping the parent would orphan the leaf; keeping the sibling would over-select.
    #[test]
    fn test_project_nested_leaf_keeps_parent_and_leaf_only() {
        let schema = nested_schema();
        let ids = HashSet::from([3]); // point.x only
        let projected = project(&schema, &ids).unwrap();
        let point = projected.field_by_name("point").expect("parent kept");
        let Type::Struct(s) = point.field_type.as_ref() else {
            panic!("point must be a struct")
        };
        assert_eq!(s.fields().len(), 1, "only the selected leaf survives");
        assert_eq!(s.fields()[0].name, "x");
    }

    /// `select_not` drops exactly the excluded leaf ids (keeping everything else). Risk: excluding by
    /// struct id instead of leaf ids would drop a whole subtree unexpectedly.
    #[test]
    fn test_select_not_drops_excluded_leaves() {
        let schema = nested_schema();
        let excluded = HashSet::from([4]); // drop point.y
        let result = select_not(&schema, &excluded).unwrap();
        let point = result.field_by_name("point").expect("point kept");
        let Type::Struct(s) = point.field_type.as_ref() else {
            panic!("point must be a struct")
        };
        assert_eq!(s.fields().len(), 1, "only y dropped");
        assert_eq!(s.fields()[0].name, "x");
        assert!(result.field_by_name("id").is_some(), "id retained");
    }

    /// `get_projected_ids_schema` includes STRUCT field ids but NOT list/map container field ids
    /// (the `&& isStructType` guard in Java `GetProjectedIds.field`). Ground-truthed against live
    /// Java 1.10.0: `getProjectedIds({1, 2:struct{3,4}, 5:list<6 str>}) = [1,2,3,4,6]` — the struct
    /// field id 2 IS present, the list field id 5 is NOT, the element id 6 IS. Risk: dropping the
    /// `&& isStructType` guard (the prior bug) over-projects the list/map field id 5.
    #[test]
    fn test_get_projected_ids_includes_struct_but_not_list_field_id() {
        let schema = nested_schema();
        let ids = get_projected_ids_schema(&schema).unwrap();
        // id(1), point(2) struct id, x(3), y(4), element(6) — NOT the list field id 5.
        assert_eq!(ids, HashSet::from([1, 2, 3, 4, 6]));
    }

    /// Map container field id is NOT projected; the key+value ids ARE (gated on the value type being
    /// a primitive). Live Java: `getProjectedIds({1, 6:map<7 str, 8 int>}) = [1, 7, 8]` — map field
    /// id 6 absent. Risk: a visitor that always adds key/value (or the map field id) would over- or
    /// under-project the wrong ids.
    #[test]
    fn test_get_projected_ids_map_omits_map_field_id() {
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    6,
                    "m",
                    Type::Map(MapType::new(
                        NestedField::map_key_element(7, Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::map_value_element(
                            8,
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
        let ids = get_projected_ids_schema(&schema).unwrap();
        assert_eq!(ids, HashSet::from([1, 7, 8]));
    }

    /// A list of STRUCT omits BOTH the list field id and the struct element field id (the element
    /// visit returns non-null, so `list()` does not add the element id; and a struct field reached
    /// via `list` is never put through `field()`). Live Java: `getProjectedIds({1, 2:list<3:struct
    /// {4 int}>}) = [1, 4]`. Risk: a visitor that always adds the list element id would wrongly
    /// include id 3.
    #[test]
    fn test_get_projected_ids_list_of_struct_omits_element_id() {
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    2,
                    "arr",
                    Type::List(ListType::new(
                        NestedField::list_element(
                            3,
                            Type::Struct(StructType::new(vec![
                                NestedField::required(4, "a", Type::Primitive(PrimitiveType::Int))
                                    .into(),
                            ])),
                            true,
                        )
                        .into(),
                    )),
                )
                .into(),
            ])
            .build()
            .unwrap();
        let ids = get_projected_ids_schema(&schema).unwrap();
        assert_eq!(ids, HashSet::from([1, 4]));
    }

    /// A map whose VALUE is a struct omits key+value ids (value visit non-null); only the struct's
    /// leaf is projected. Live Java: `getProjectedIds({1, 2:map<3 str, 4:struct{5 int}>}) = [1, 5]`.
    /// The complementary case (struct KEY, primitive VALUE) DOES add key+value (gated on the value):
    /// `getProjectedIds({1, 2:map<3:struct{5}, 4 int>}) = [1, 3, 4, 5]`. Pins that the gate is the
    /// VALUE type, not the key.
    #[test]
    fn test_get_projected_ids_map_gate_is_value_type() {
        // value is a struct -> key/value ids omitted, only the leaf 5.
        let struct_value = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    2,
                    "m",
                    Type::Map(MapType::new(
                        NestedField::map_key_element(3, Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::map_value_element(
                            4,
                            Type::Struct(StructType::new(vec![
                                NestedField::required(5, "x", Type::Primitive(PrimitiveType::Int))
                                    .into(),
                            ])),
                            true,
                        )
                        .into(),
                    )),
                )
                .into(),
            ])
            .build()
            .unwrap();
        assert_eq!(
            get_projected_ids_schema(&struct_value).unwrap(),
            HashSet::from([1, 5])
        );

        // value is a primitive but key is a struct -> key+value ids added, plus the key's leaf.
        let struct_key = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    2,
                    "m",
                    Type::Map(MapType::new(
                        NestedField::map_key_element(
                            3,
                            Type::Struct(StructType::new(vec![
                                NestedField::required(5, "x", Type::Primitive(PrimitiveType::Int))
                                    .into(),
                            ])),
                        )
                        .into(),
                        NestedField::map_value_element(
                            4,
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
        assert_eq!(
            get_projected_ids_schema(&struct_key).unwrap(),
            HashSet::from([1, 3, 4, 5])
        );
    }

    /// `get_projected_ids(Type)` returns the empty set for a primitive (Java's primitive short-
    /// circuit). Risk: returning the field's own id would over-project a leaf type.
    #[test]
    fn test_get_projected_ids_primitive_is_empty() {
        let ids = get_projected_ids(&Type::Primitive(PrimitiveType::Int)).unwrap();
        assert!(ids.is_empty());
    }

    /// `join` concatenates disjoint columns and tolerates a shared id whose column is IDENTICAL.
    /// Risk: rejecting an identical shared column would break legitimate re-joins.
    #[test]
    fn test_join_concats_disjoint_and_allows_identical_shared() {
        let left = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "a", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();
        let right = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                // shared id 1, IDENTICAL column
                NestedField::required(1, "a", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "b", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();
        let joined = join(&left, &right).unwrap();
        assert_eq!(
            joined.as_struct().fields().len(),
            2,
            "a + b, a not duplicated"
        );
        assert!(joined.field_by_id(1).is_some() && joined.field_by_id(2).is_some());
    }

    /// `join` is a HARD ERROR when two schemas share an id with DIFFERENT columns (Java's
    /// IllegalArgumentException). Risk: silently keeping one would corrupt the merged schema's ids.
    #[test]
    fn test_join_rejects_conflicting_shared_id() {
        let left = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "a", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();
        let right = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                // same id 1, DIFFERENT type
                NestedField::required(1, "a", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap();
        let err = join(&left, &right).expect_err("conflicting shared id must fail");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("different columns with same id"),
            "message was: {}",
            err.message()
        );
    }

    /// `estimate_size` matches Java's per-type byte estimates and recursion. Risk: a wrong constant
    /// (decoded from the `TypeUtil$1` SwitchMap) would skew downstream split sizing.
    #[test]
    fn test_estimate_size_matches_java_constants() {
        let int_field: NestedFieldRef =
            NestedField::required(1, "i", Type::Primitive(PrimitiveType::Int)).into();
        assert_eq!(estimate_size(&int_field), 4);

        let str_field: NestedFieldRef =
            NestedField::required(2, "s", Type::Primitive(PrimitiveType::String)).into();
        assert_eq!(estimate_size(&str_field), 54);

        let dec_field: NestedFieldRef = NestedField::required(
            3,
            "d",
            Type::Primitive(PrimitiveType::Decimal {
                precision: 10,
                scale: 2,
            }),
        )
        .into();
        assert_eq!(estimate_size(&dec_field), 44);

        // struct< int, string > => 12 + 4 + 54 = 70.
        let struct_field: NestedFieldRef = NestedField::required(
            4,
            "st",
            Type::Struct(StructType::new(vec![
                NestedField::required(5, "i", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(6, "s", Type::Primitive(PrimitiveType::String)).into(),
            ])),
        )
        .into();
        assert_eq!(estimate_size(&struct_field), 70);

        // list< int > => 12 + 5 * 4 = 32.
        let list_field: NestedFieldRef = NestedField::required(
            7,
            "l",
            Type::List(ListType::new(
                NestedField::list_element(8, Type::Primitive(PrimitiveType::Int), true).into(),
            )),
        )
        .into();
        assert_eq!(estimate_size(&list_field), 32);

        // map< int, string > => 12 + 5 * (12 + 4 + 54) = 12 + 5*70 = 362. Live Java 1.10.0 = 362.
        // The inner per-entry 12 is the map-vs-list asymmetry (a naive 12 + 5*(4+54) = 302 is WRONG).
        let map_field: NestedFieldRef = NestedField::required(
            10,
            "m",
            Type::Map(MapType::new(
                NestedField::map_key_element(11, Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::map_value_element(12, Type::Primitive(PrimitiveType::String), true)
                    .into(),
            )),
        )
        .into();
        assert_eq!(estimate_size(&map_field), 362);

        // unknown => 0.
        let unknown_field: NestedFieldRef =
            NestedField::required(9, "u", Type::Primitive(PrimitiveType::Unknown)).into();
        assert_eq!(estimate_size(&unknown_field), 0);
    }

    /// `index_quoted_name_by_id` applies the quote function to each dotted name segment and indexes
    /// nested fields by their full quoted name. Risk: not quoting per-segment (or quoting the whole
    /// dotted path) would produce names a SQL layer cannot round-trip.
    #[test]
    fn test_index_quoted_name_by_id_quotes_each_segment() {
        let schema = nested_schema();
        let index = index_quoted_name_by_id(schema.as_struct(), |s| format!("`{s}`")).unwrap();
        assert_eq!(index.get(&1).map(String::as_str), Some("`id`"));
        assert_eq!(index.get(&3).map(String::as_str), Some("`point`.`x`"));
        assert_eq!(index.get(&4).map(String::as_str), Some("`point`.`y`"));
    }
}
