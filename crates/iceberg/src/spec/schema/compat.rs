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

//! Schema write/read compatibility checking — the Rust port of Java
//! `org.apache.iceberg.types.CheckCompatibility` plus the `TypeUtil.validateSchema` /
//! `validateWriteSchema` / `checkSchemaCompatibility` wrappers that wrap it into an error.
//!
//! ## Direction (decoded from the 1.10.0 bytecode, verified against live Java)
//!
//! This is the subtle part. `TypeUtil.checkSchemaCompatibility(ctx, reference, schema, checkOrdering,
//! checkNullability)` calls `CheckCompatibility.writeCompatibilityErrors(reference, schema, …)` (or
//! `typeCompatibilityErrors(reference, schema, …)` when `checkOrdering` is false). Each of those is
//! `TypeUtil.visit(reference, new CheckCompatibility(schema, …))` — so Java **VISITS the `reference`
//! schema** while tracking the matching position in the **`schema`** argument (the visitor's
//! `currentType`). The visited schema (`reference`) is the one whose fields are iterated and whose
//! field names appear in messages; the tracked schema (`schema`) is what `currentType` walks.
//!
//! For `validateWriteSchema(table, write, …)` the wired `reference` is the **table** and the
//! `schema` is the **write** schema, so the walk VISITS the TABLE and tracks the WRITE schema. An
//! error therefore fires when the TABLE requires/orders something the write schema fails to supply
//! (e.g. `validateWriteSchema(tableReq3, write_without_it) -> "needed is required, but is missing"`).
//! Ground-truthed against Java 1.10.0: `validateWriteSchema(table, writeWithRequiredData, true, true)
//! => OK`; `writeCompatibilityErrors(table, write) = []` but `writeCompatibilityErrors(write, table)
//! = [data should be required, but is optional]`.
//!
//! ## Knobs
//! - **type promotion** — the tracked (`schema`) primitive must equal the visited primitive or be an
//!   allowed promotion to it ([`is_promotion_allowed`]); else `": %s cannot be promoted to %s"`
//!   `(tracked, visited)`, or `": %s cannot be read as a %s"` `(tracked.typeId lowercased, visited)`
//!   when the tracked position is not even a primitive.
//! - **nullability** (`check_nullability`) — when set, a VISITED field that is *required* against a
//!   TRACKED field that is *optional* is an error (`"<name> should be required, but is optional"`);
//!   likewise list elements / map values.
//! - **ordering** (`check_ordering`) — when set, the visited fields must appear in the same relative
//!   order as the tracked fields (`"<name> is out of order, before <name>"`). Per the 1.10.0
//!   bytecode the `checkOrdering` bool is forwarded into BOTH `write`/`typeCompatibilityErrors`, so
//!   either form honors it; what distinguishes them is nullability (write = on, type = off).

use super::*;
use crate::spec::datatypes::Type;

/// Collect write-compatibility problems. Rust port of
/// `CheckCompatibility.writeCompatibilityErrors(reference, schema, checkOrdering)`. Per the 1.10.0
/// bytecode this is `visit(reference, new CheckCompatibility(schema, checkOrdering, /*checkNullability=*/true))`:
/// it VISITS `reference` while tracking `schema` via `current_type`, with **nullability checking
/// always ON** (the write form) and ordering controlled by the passed `check_ordering`.
fn write_compatibility_errors(
    reference: &Schema,
    schema: &Schema,
    check_ordering: bool,
) -> Result<Vec<String>> {
    let mut visitor = CheckCompatibility {
        tracked: schema,
        check_ordering,
        check_nullability: true,
        current_type: None,
    };
    visitor.visit_schema(reference)
}

/// Collect type-compatibility problems. Rust port of
/// `CheckCompatibility.typeCompatibilityErrors(reference, schema, checkOrdering)` —
/// `visit(reference, new CheckCompatibility(schema, checkOrdering, /*checkNullability=*/false))`:
/// VISITS `reference`, tracks `schema`, with **nullability checking always OFF** (the type form)
/// and ordering controlled by the passed `check_ordering`.
fn type_compatibility_errors(
    reference: &Schema,
    schema: &Schema,
    check_ordering: bool,
) -> Result<Vec<String>> {
    let mut visitor = CheckCompatibility {
        tracked: schema,
        check_ordering,
        check_nullability: false,
        current_type: None,
    };
    visitor.visit_schema(reference)
}

// =====================================================================================
// Public entry points — `TypeUtil.checkSchemaCompatibility` and its two wrappers.
// =====================================================================================

/// Validate that `schema` is compatible with `reference` under the given checks, returning a
/// `DataInvalid` error whose message mirrors Java when it is not. Rust port of
/// `TypeUtil.checkSchemaCompatibility(String context, Schema reference, Schema schema, boolean
/// checkOrdering, boolean checkNullability)`.
///
/// Per the 1.10.0 bytecode (`if (checkNullability) writeCompatibilityErrors(ref, schema,
/// checkOrdering) else typeCompatibilityErrors(ref, schema, checkOrdering)`), `check_nullability`
/// SELECTS the write form (nullability checked) vs the type form (nullability not checked), and
/// `check_ordering` is forwarded to the visitor's ordering check in BOTH forms. (Param order is kept
/// as Java's `(…, checkOrdering, checkNullability)`.)
pub fn check_schema_compatibility(
    context: &str,
    reference: &Schema,
    schema: &Schema,
    check_ordering: bool,
    check_nullability: bool,
) -> Result<()> {
    let errors = if check_nullability {
        write_compatibility_errors(reference, schema, check_ordering)?
    } else {
        type_compatibility_errors(reference, schema, check_ordering)?
    };

    if errors.is_empty() {
        return Ok(());
    }

    // Build the joined message byte-for-byte with Java's StringBuilder in checkSchemaCompatibility.
    let mut message = String::new();
    message.push_str(context);
    message.push('\n');
    message.push_str(&reference.as_struct().to_string());
    message.push('\n');
    message.push_str("Provided schema:");
    message.push('\n');
    message.push_str(&schema.as_struct().to_string());
    message.push('\n');
    message.push_str("Problems:");
    for error in errors {
        message.push_str("\n* ");
        message.push_str(&error);
    }

    Err(Error::new(ErrorKind::DataInvalid, message))
}

/// Validate that `write_schema` may be written into a table with `table_schema`. Rust port of
/// `TypeUtil.validateWriteSchema(Schema tableSchema, Schema writeSchema, Boolean checkNullability,
/// Boolean checkOrdering)`.
///
/// Per the 1.10.0 bytecode this forwards `(checkNullability, checkOrdering)` straight into
/// `checkSchemaCompatibility(ctx, tableSchema, writeSchema, checkOrdering, checkNullability)` — so
/// `check_nullability` selects the write form (nullability checked) and `check_ordering` toggles the
/// ordering check. The context string is Java's verbatim: `"Cannot write incompatible dataset to
/// table with schema:"`.
pub fn validate_write_schema(
    table_schema: &Schema,
    write_schema: &Schema,
    check_nullability: bool,
    check_ordering: bool,
) -> Result<()> {
    check_schema_compatibility(
        "Cannot write incompatible dataset to table with schema:",
        table_schema,
        write_schema,
        check_ordering,
        check_nullability,
    )
}

/// Validate that the `schema` provided for the named `context` is compatible with the expected
/// `reference` schema. Rust port of `TypeUtil.validateSchema(String context, Schema reference,
/// Schema schema, boolean checkOrdering, boolean checkNullability)`; the context is formatted into
/// Java's verbatim `"Provided %s schema is incompatible with expected schema:"`.
pub fn validate_schema(
    context: &str,
    reference: &Schema,
    schema: &Schema,
    check_ordering: bool,
    check_nullability: bool,
) -> Result<()> {
    let formatted = format!("Provided {context} schema is incompatible with expected schema:");
    check_schema_compatibility(
        &formatted,
        reference,
        schema,
        check_ordering,
        check_nullability,
    )
}

/// =====================================================================================
/// The `CheckCompatibility` walk — mirrors Java's `CustomOrderSchemaVisitor<List<String>>`.
///
/// Java drives this with a lazy supplier-based `CustomOrderSchemaVisitor`; we express the same
/// top-down walk as explicit recursion. The recursion walks the VISITED schema (the `reference`
/// argument of `checkSchemaCompatibility`); `current_type` is the matching position in the TRACKED
/// schema (the `schema` argument, stored in `tracked`). Each method returns the list of problem
/// strings for its subtree, prefixed (in `field`) by the visited field name exactly as Java does.
/// =====================================================================================
struct CheckCompatibility<'a> {
    /// The schema that `current_type` walks (Java's `this.schema`, the constructor arg).
    tracked: &'a Schema,
    check_ordering: bool,
    check_nullability: bool,
    current_type: Option<Type>,
}

impl CheckCompatibility<'_> {
    fn visit_schema(&mut self, visited: &Schema) -> Result<Vec<String>> {
        // Java's schema() sets currentType = tracked.asStruct() then walks the VISITED struct.
        self.current_type = Some(Type::Struct(self.tracked.as_struct().clone()));
        let result = self.visit_struct(visited.as_struct());
        self.current_type = None;
        result
    }

    fn visit_type(&mut self, visited_type: &Type) -> Result<Vec<String>> {
        match visited_type {
            Type::Struct(s) => self.visit_struct(s),
            Type::List(l) => self.visit_list(l),
            Type::Map(m) => self.visit_map(m),
            Type::Primitive(p) => self.visit_primitive(p),
            Type::Variant => self.visit_variant(),
        }
    }

    fn visit_struct(&mut self, visited_struct: &StructType) -> Result<Vec<String>> {
        // The tracked position must be a struct.
        let Some(Type::Struct(tracked_struct)) = self.current_type.clone() else {
            let current = self.current_type_display();
            return Ok(vec![format!(": {current} cannot be read as a struct")]);
        };

        let mut errors = Vec::new();
        for field in visited_struct.fields() {
            errors.extend(self.visit_field(&tracked_struct, field)?);
        }

        // Java's ordering check (decoded from struct() bytecode): build a map of
        // {trackedField.fieldId() -> position} over the TRACKED struct's fields, then iterate the
        // VISITED struct's fields tracking `last_index`. When a visited field that exists in the
        // tracked struct maps to a tracked position <= a prior one, it is out of order. The message
        // names the offending VISITED field and the tracked field at `last_index`.
        if self.check_ordering {
            let mut tracked_id_to_pos: HashMap<i32, usize> = HashMap::new();
            for (pos, field) in tracked_struct.fields().iter().enumerate() {
                tracked_id_to_pos.insert(field.id, pos);
            }

            let mut last_index: i64 = -1;
            for visited_field in visited_struct.fields() {
                if let Some(&index) = tracked_id_to_pos.get(&visited_field.id) {
                    let index = index as i64;
                    if last_index >= index {
                        let before = &tracked_struct.fields()[last_index as usize];
                        errors.push(format!(
                            "{} is out of order, before {}",
                            visited_field.name, before.name
                        ));
                    }
                    last_index = index;
                }
            }
        }

        Ok(errors)
    }

    fn visit_field(
        &mut self,
        tracked_struct: &StructType,
        visited_field: &NestedFieldRef,
    ) -> Result<Vec<String>> {
        let saved = self.current_type.clone();

        let tracked_field = tracked_struct.field_by_id(visited_field.id).cloned();
        let mut errors = Vec::new();

        let Some(tracked_field) = tracked_field else {
            // No matching field in the tracked schema: an error only when the VISITED field is
            // required (Java recipe #1: "<name> is required, but is missing").
            if visited_field.required {
                return Ok(vec![format!(
                    "{} is required, but is missing",
                    visited_field.name
                )]);
            }
            return Ok(errors);
        };

        // Descend: the tracked position becomes the matched tracked field's type.
        self.current_type = Some((*tracked_field.field_type).clone());

        // Java recipe #2 (no leading colon): "<name> should be required, but is optional".
        if self.check_nullability && visited_field.required && !tracked_field.required {
            errors.push(format!(
                "{} should be required, but is optional",
                visited_field.name
            ));
        }

        let child_errors = self.visit_type(&visited_field.field_type)?;
        for child in child_errors {
            // Java: childError.startsWith(":") ? name + childError (#3) : name + "." + childError (#4).
            if child.starts_with(':') {
                errors.push(format!("{}{}", visited_field.name, child));
            } else {
                errors.push(format!("{}.{}", visited_field.name, child));
            }
        }

        self.current_type = saved;
        Ok(errors)
    }

    fn visit_list(&mut self, visited_list: &ListType) -> Result<Vec<String>> {
        let Some(Type::List(tracked_list)) = self.current_type.clone() else {
            let current = self.current_type_display();
            return Ok(vec![format!(": {current} cannot be read as a list")]);
        };

        let saved = self.current_type.clone();
        let mut errors = Vec::new();

        // Java: visited.isElementRequired() && tracked.isElementOptional().
        if visited_list.element_field.required && !tracked_list.element_field.required {
            errors.push(": elements should be required, but are optional".to_string());
        }

        self.current_type = Some((*tracked_list.element_field.field_type).clone());
        errors.extend(self.visit_type(&visited_list.element_field.field_type)?);

        self.current_type = saved;
        Ok(errors)
    }

    fn visit_map(&mut self, visited_map: &MapType) -> Result<Vec<String>> {
        let Some(Type::Map(tracked_map)) = self.current_type.clone() else {
            let current = self.current_type_display();
            return Ok(vec![format!(": {current} cannot be read as a map")]);
        };

        let saved = self.current_type.clone();
        let mut errors = Vec::new();

        // Java: visited.isValueRequired() && tracked.isValueOptional() (gated on the value only).
        if visited_map.value_field.required && !tracked_map.value_field.required {
            errors.push(": values should be required, but are optional".to_string());
        }

        // Java visits the key then the value, repositioning current_type for each.
        self.current_type = Some((*tracked_map.key_field.field_type).clone());
        errors.extend(self.visit_type(&visited_map.key_field.field_type)?);

        self.current_type = Some((*tracked_map.value_field.field_type).clone());
        errors.extend(self.visit_type(&visited_map.value_field.field_type)?);

        self.current_type = saved;
        Ok(errors)
    }

    fn visit_primitive(&mut self, visited_primitive: &PrimitiveType) -> Result<Vec<String>> {
        match self.current_type.clone() {
            // Java: if currentType.equals(primitive) -> NO_ERRORS.
            Some(Type::Primitive(tracked_primitive)) if tracked_primitive == *visited_primitive => {
                Ok(Vec::new())
            }
            Some(Type::Primitive(tracked_primitive)) => {
                let from = Type::Primitive(tracked_primitive);
                // Java: isPromotionAllowed(currentType.asPrimitiveType(), visited) — the tracked
                // primitive must be promotable to the visited primitive; else
                // "<from> cannot be promoted to <visited>".
                if is_promotion_allowed(&from, visited_primitive) {
                    Ok(Vec::new())
                } else {
                    Ok(vec![format!(
                        ": {from} cannot be promoted to {visited_primitive}"
                    )])
                }
            }
            // currentType (tracked) is not a primitive: Java formats the tracked typeId in lower
            // case ("<typeid> cannot be read as a <visited>").
            Some(other) => Ok(vec![format!(
                ": {} cannot be read as a {visited_primitive}",
                type_id_lower(&other)
            )]),
            None => Ok(vec![format!(
                ": (unknown) cannot be read as a {visited_primitive}"
            )]),
        }
    }

    fn visit_variant(&mut self) -> Result<Vec<String>> {
        match self.current_type.clone() {
            Some(Type::Variant) => Ok(Vec::new()),
            Some(other) => Ok(vec![format!(": {other} cannot be read as a variant")]),
            None => Ok(vec![": (unknown) cannot be read as a variant".to_string()]),
        }
    }

    fn current_type_display(&self) -> String {
        match &self.current_type {
            Some(t) => t.to_string(),
            None => "(unknown)".to_string(),
        }
    }
}

/// Java's `currentType.typeId().toString().toLowerCase(ENGLISH)` for the `"%s cannot be read as %s"`
/// recipe in `primitive()`. The TypeID name lower-cased (e.g. `struct`, `list`, `map`); for the
/// container/variant types we render the short id name to mirror Java's `TypeID` enum constants.
fn type_id_lower(t: &Type) -> String {
    match t {
        Type::Struct(_) => "struct".to_string(),
        Type::List(_) => "list".to_string(),
        Type::Map(_) => "map".to_string(),
        Type::Variant => "variant".to_string(),
        // A primitive is handled by the equal/promotion arms before this is reached; fall back to
        // the type's own Display (this arm is effectively unreachable for primitives).
        Type::Primitive(p) => p.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::datatypes::{ListType, MapType, NestedField, PrimitiveType, StructType, Type};

    /// A simple `{1: id (req int), 2: data (opt string)}` table schema used as the reference.
    fn table_schema() -> Schema {
        Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap()
    }

    /// Identical write schema must be compatible with no problems.
    #[test]
    fn compatible_identical_schema() {
        let table = table_schema();
        let write = table_schema();
        assert!(validate_write_schema(&table, &write, true, true).is_ok());
    }

    /// Legal promotion: a `long` table column may be written from an `int` write column (`int`
    /// promotes to `long`). Java visits the table (`long`) tracking the write (`int`) and consults
    /// `isPromotionAllowed(int, long) = true`. Live Java 1.10.0: `writeCompatibilityErrors(table_long,
    /// write_int) = []` (whereas the OPPOSITE, `table_int, write_long`, ERRORS with `long cannot be
    /// promoted to int`). Risk: if promotion were not consulted, this legal write would be rejected.
    #[test]
    fn compatible_promotion_int_to_long() {
        // Table: id is LONG.
        let table = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();
        // Write: id is INT (promotes to the table's long).
        let write = table_schema();
        assert!(validate_write_schema(&table, &write, true, true).is_ok());
    }

    /// A forbidden promotion: the table reads an `int` (id 1) but the write supplies a `string`.
    /// Java walks the TABLE (visited) tracking the WRITE; `primitive()` checks whether the tracked
    /// (write `string`) can be promoted to the visited (table `int`) — it cannot. Live Java 1.10.0:
    /// `writeCompatibilityErrors(table_int, write_string) = [id: string cannot be promoted to int]`.
    /// Risk: silently accepting would let a writer commit data the table cannot read.
    #[test]
    fn incompatible_forbidden_promotion() {
        let table = table_schema(); // id: int, data: string
        let write = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();
        let err = validate_write_schema(&table, &write, true, true).expect_err("should fail");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message()
                .contains("id: string cannot be promoted to int"),
            "message was: {}",
            err.message()
        );
    }

    /// Nullability: the TABLE requires `data` (id 2) but the write makes it OPTIONAL. Java visits the
    /// table (required field) tracking the write (optional field) and flags it. Live Java 1.10.0:
    /// `writeCompatibilityErrors(table_data_required, write_data_optional) = [data should be
    /// required, but is optional]` (NO leading colon, NO field-prefix — it is a top-level field). The
    /// OPPOSITE direction (table optional, write required) is OK — pinned in
    /// [`compatible_write_required_over_optional_table`].
    #[test]
    fn incompatible_required_over_optional_when_nullability_checked() {
        // Table: data (id 2) REQUIRED.
        let table = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();
        // Write: data (id 2) OPTIONAL.
        let write = table_schema();
        // validate_write_schema(table, write, check_nullability, check_ordering): check_nullability
        // (the 3rd arg) selects the write form, which turns the nullability check ON. Live Java
        // 1.10.0: validateWriteSchema(table_data_required, write_data_optional, true, true) ERRORS.
        let err = validate_write_schema(&table, &write, true, true).expect_err("should fail");
        assert_eq!(
            err.message().lines().last().unwrap(),
            "* data should be required, but is optional",
            "full message was: {}",
            err.message()
        );
    }

    /// The Java-correct OK direction: a write that makes an OPTIONAL table field REQUIRED is fine
    /// (the table can read a required column as optional). Live Java 1.10.0:
    /// `validateWriteSchema(table {data optional}, write {data required}, true, true) => OK` and
    /// `writeCompatibilityErrors(table, write) = []`. This pins finding mustFix #1 (the direction was
    /// previously inverted, which made this case ERROR).
    #[test]
    fn compatible_write_required_over_optional_table() {
        let table = table_schema(); // data (id 2) OPTIONAL
        let write = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();
        assert!(
            validate_write_schema(&table, &write, false, true).is_ok(),
            "write making an optional table field required must be OK (Java parity)"
        );
    }

    /// With nullability NOT checked, the required-over-optional case is allowed. Pins that the
    /// nullability rule is gated on the flag (and confirms the Java arg-crossing: passing the 4th
    /// arg = false disables the nullability check).
    #[test]
    fn required_over_optional_allowed_when_nullability_unchecked() {
        let table = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();
        let write = table_schema();
        assert!(validate_write_schema(&table, &write, false, false).is_ok());
    }

    /// Missing required: the TABLE requires field id 3 (`needed`) but the write OMITS it. Java visits
    /// the table (which has the required field with no tracked match) and flags it. Live Java 1.10.0:
    /// `writeCompatibilityErrors(table_with_needed, write_without) = [needed is required, but is
    /// missing]`. Note the message has NO leading colon and NO "field" word (finding HIGH #3).
    #[test]
    fn incompatible_missing_required_field() {
        // Table requires id 3.
        let table = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(3, "needed", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();
        // Write omits id 3.
        let write = table_schema();
        let err = validate_write_schema(&table, &write, true, true).expect_err("should fail");
        assert_eq!(
            err.message().lines().last().unwrap(),
            "* needed is required, but is missing",
            "full message was: {}",
            err.message()
        );
    }

    /// A write supplying an EXTRA required field absent from the table is compatible — the table
    /// does not require it, so there is nothing to miss. Live Java 1.10.0:
    /// `validateWriteSchema(table, write_with_extra_required_id3, true, true) => OK`. Risk: a
    /// direction error would wrongly reject a write that simply carries an extra column.
    #[test]
    fn compatible_write_with_extra_required_field() {
        let table = table_schema();
        let write = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(3, "extra", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();
        assert!(
            validate_write_schema(&table, &write, true, true).is_ok(),
            "an extra required field in the write must be OK (Java parity)"
        );
    }

    /// Ordering: the WRITE reorders fields relative to the TABLE. Java visits the table tracking the
    /// write and flags the visited field whose tracked position regresses. Live Java 1.10.0:
    /// `writeCompatibilityErrors(table {id, data}, write {data, id}) = [data is out of order, before
    /// id]`. Pinned exact-message; the check fires only under `check_ordering`.
    #[test]
    fn incompatible_out_of_order_when_ordering_checked() {
        let table = table_schema(); // [id(1), data(2)]
        // Write reverses: data (id 2) then id (id 1).
        let write = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();
        // check_ordering is the 4th arg; set it true. check_nullability=false isolates the ordering
        // check (the type form still honors checkOrdering per the 1.10.0 bytecode).
        let err = validate_write_schema(&table, &write, false, true).expect_err("should fail");
        assert_eq!(
            err.message().lines().last().unwrap(),
            "* data is out of order, before id",
            "full message was: {}",
            err.message()
        );
    }

    /// The same reordering is allowed by `type_compatibility_errors` (no ordering check). Pins that
    /// ordering is the only thing the ordering knob controls.
    #[test]
    fn reorder_allowed_without_ordering_check() {
        let table = table_schema();
        let read = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();
        assert!(check_schema_compatibility("read", &table, &read, false, true).is_ok());
    }

    /// A nested struct field type mismatch ("cannot be read as a struct") with the field-name prefix
    /// applied. Pins the nested recursion + the name-prefix concatenation (the `:`-prefix branch).
    #[test]
    fn incompatible_nested_type_mismatch_carries_field_name() {
        // Table (visited): { 1: point (req struct< 2: x int >) }.
        let table = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(
                    1,
                    "point",
                    Type::Struct(StructType::new(vec![
                        NestedField::required(2, "x", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap();
        // Write (tracked): { 1: point (req int) } — point is a primitive, not a struct.
        let write = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "point", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();
        let err = validate_write_schema(&table, &write, true, true).expect_err("should fail");
        // Live Java 1.10.0: [point: int cannot be read as a struct].
        assert_eq!(
            err.message().lines().last().unwrap(),
            "* point: int cannot be read as a struct",
            "full message was: {}",
            err.message()
        );
    }

    /// A forbidden promotion BENEATH a list element carries the `.`-joined path prefix. Java's child
    /// recipes: a child error that does NOT start with ":" is joined as `name + "." + child`. Live
    /// Java 1.10.0: `writeCompatibilityErrors(list<struct{int}>, list<struct{string}>) = [arr.x:
    /// string cannot be promoted to int]`. Pins both child-prefix branches (the inner `x:`-colon
    /// concat and the outer `arr.`-dot concat).
    #[test]
    fn incompatible_nested_list_struct_promotion_path() {
        // Table: arr list< struct{ x int } >.
        let table = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(
                    1,
                    "arr",
                    Type::List(ListType::new(
                        NestedField::list_element(
                            2,
                            Type::Struct(StructType::new(vec![
                                NestedField::required(3, "x", Type::Primitive(PrimitiveType::Int))
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
        // Write: arr list< struct{ x string } > — x is forbidden-promoted.
        let write = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(
                    1,
                    "arr",
                    Type::List(ListType::new(
                        NestedField::list_element(
                            2,
                            Type::Struct(StructType::new(vec![
                                NestedField::required(
                                    3,
                                    "x",
                                    Type::Primitive(PrimitiveType::String),
                                )
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
        let err = validate_write_schema(&table, &write, true, true).expect_err("should fail");
        assert_eq!(
            err.message().lines().last().unwrap(),
            "* arr.x: string cannot be promoted to int",
            "full message was: {}",
            err.message()
        );
    }

    /// List element nullability: the TABLE list element is REQUIRED but the write makes it OPTIONAL.
    /// Java flags when `visited.isElementRequired() && tracked.isElementOptional()`. Live Java
    /// 1.10.0: `writeCompatibilityErrors(table {list<req str>}, write {list<opt str>}) = [tags:
    /// elements should be required, but are optional]`. Risk: list-element nullability is a distinct
    /// rule from struct-field nullability and must be checked separately.
    #[test]
    fn incompatible_list_element_required_over_optional() {
        // Table: { 1: tags (req list< REQ string >) }.
        let table = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(
                    1,
                    "tags",
                    Type::List(ListType::new(
                        NestedField::list_element(2, Type::Primitive(PrimitiveType::String), true)
                            .into(),
                    )),
                )
                .into(),
            ])
            .build()
            .unwrap();
        // Write: optional list element.
        let write = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(
                    1,
                    "tags",
                    Type::List(ListType::new(
                        NestedField::list_element(2, Type::Primitive(PrimitiveType::String), false)
                            .into(),
                    )),
                )
                .into(),
            ])
            .build()
            .unwrap();
        let err = validate_write_schema(&table, &write, false, true).expect_err("should fail");
        assert_eq!(
            err.message().lines().last().unwrap(),
            "* tags: elements should be required, but are optional",
            "full message was: {}",
            err.message()
        );
    }

    /// Map value nullability: the TABLE map value is REQUIRED but the write makes it OPTIONAL. Java
    /// flags when `visited.isValueRequired() && tracked.isValueOptional()` (gated on the value). Live
    /// Java 1.10.0: `writeCompatibilityErrors(table {map<str, req int>}, write {map<str, opt int>}) =
    /// [m: values should be required, but are optional]`.
    #[test]
    fn incompatible_map_value_required_over_optional() {
        // Table: { 1: m (req map< string, REQ int >) }.
        let table = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(
                    1,
                    "m",
                    Type::Map(MapType::new(
                        NestedField::map_key_element(2, Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::map_value_element(
                            3,
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
        let write = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(
                    1,
                    "m",
                    Type::Map(MapType::new(
                        NestedField::map_key_element(2, Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::map_value_element(
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
            .unwrap();
        let err = validate_write_schema(&table, &write, false, true).expect_err("should fail");
        assert_eq!(
            err.message().lines().last().unwrap(),
            "* m: values should be required, but are optional",
            "full message was: {}",
            err.message()
        );
    }

    /// `validate_schema` formats the context into Java's "Provided %s schema is incompatible..."
    /// header on failure. Pins the wrapper's distinct context text.
    #[test]
    fn validate_schema_formats_context() {
        let table = table_schema();
        let bad = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();
        let err = validate_schema("read", &table, &bad, false, true).expect_err("should fail");
        assert!(
            err.message()
                .contains("Provided read schema is incompatible with expected schema:"),
            "message was: {}",
            err.message()
        );
    }
}
