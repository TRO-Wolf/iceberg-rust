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

//! Cross-spec partition-type unification — the port of Java `Partitioning`
//! (`core/src/main/java/org/apache/iceberg/Partitioning.java`).
//!
//! The five public/crate functions here are the hoist of the analogue that
//! previously lived (with G1–G4 fidelity gaps) in
//! `maintenance/partition_stats.rs`. Inspect tables still project
//! [`TableMetadata::default_partition_type`] until increments B/C/D adopt
//! this module.

use std::collections::{HashMap, HashSet};

use super::{
    NestedField, PartitionField, PartitionSpec, PartitionSpecRef, PrimitiveType, Schema, Struct,
    StructType, TableMetadata, Transform, Type,
};
use crate::{Error, ErrorKind, Result};

/// Java `Partitioning.partitionType(Table)`.
///
/// Union of every partition field across `specs` whose source column is still
/// present in `schema`, keyed by partition field id, sorted by field id
/// ascending, every field optional.
///
/// # Errors
///
/// - [`ErrorKind::DataInvalid`] whose message starts with
///   `"Cannot build table partition type, unknown transforms"` when any spec
///   carries [`Transform::Unknown`] (Java L270–276).
/// - [`ErrorKind::DataInvalid`] whose message starts with
///   `"Conflicting partition fields"` when two specs reuse one field id with
///   incompatible source/transform (Java L305–310).
pub fn partition_type(schema: &Schema, specs: &[PartitionSpecRef]) -> Result<StructType> {
    build_partition_projection_type(
        "table partition",
        schema,
        specs,
        &all_active_field_ids(schema, specs),
    )
}

/// Java `Partitioning.groupingKeyType(Schema, Collection<PartitionSpec>)`.
///
/// Intersection of the non-void, live-source partition fields common to EVERY
/// spec. `schema == None` considers all partition fields (Java's
/// nullable-schema contract).
///
/// # Errors
///
/// Same G1/G2 refusals as [`partition_type`], with type name `"grouping key"`.
pub fn grouping_key_type(
    schema: Option<&Schema>,
    specs: &[PartitionSpecRef],
) -> Result<StructType> {
    // Java always passes a schema into `buildPartitionProjectionType` for type
    // resolution; when the public `schema` argument is null it only skips the
    // live-source filter. Use the first spec's sources via an empty schema
    // fallback only for the type-resolution argument — the projected id set
    // is computed with the nullable schema.
    let type_schema = match schema {
        Some(present) => present,
        None => {
            return build_partition_projection_type_nullable_schema(
                "grouping key",
                specs,
                &common_active_field_ids(None, specs),
            );
        }
    };
    build_partition_projection_type(
        "grouping key",
        type_schema,
        specs,
        &common_active_field_ids(Some(type_schema), specs),
    )
}

/// Java `Partitioning.unionPartitionTypes(Collection<PartitionSpec>)`
/// (package-private there).
///
/// Retains fields whose source column was dropped. `pub(crate)` until a
/// consumer exists.
///
/// # Errors
///
/// Same G1/G2 refusals as [`partition_type`], with type name `"union partition"`.
#[allow(dead_code)]
pub(crate) fn union_partition_types(
    schema: &Schema,
    specs: &[PartitionSpecRef],
) -> Result<StructType> {
    build_partition_projection_type("union partition", schema, specs, &all_field_ids(specs))
}

/// Java `Partitioning.isPartitioned(Table)` — any spec has a non-void field.
#[must_use]
pub fn is_partitioned(specs: &[PartitionSpecRef]) -> bool {
    specs.iter().any(|spec| !spec.is_unpartitioned())
}

/// Java `PartitionUtil.coercePartition(StructType, PartitionSpec, StructLike)`
/// = `StructProjection.createAllowMissing(spec.partitionType(), unified)`.
///
/// Field-id matched (NOT positional); a unified field absent from `spec`
/// reads null. The file tuple is aligned with [`PartitionSpec::fields`], so
/// this does **not** call [`PartitionSpec::partition_type`] (that hard-errors
/// on a dropped source column — G4).
///
/// `schema` is accepted for signature parity with the design (§3.2) and for
/// increment-B callers; matching does not consult it.
///
/// # Errors
///
/// [`ErrorKind::DataInvalid`] if a **required** unified field is missing from
/// `spec` (Java `createAllowMissing` throws when `!found && !optional`).
pub fn coerce_partition(
    unified: &StructType,
    spec: &PartitionSpec,
    schema: &Schema,
    partition: &Struct,
) -> Result<Struct> {
    let _schema = schema;
    let spec_fields = spec.fields();
    let file_values = partition.fields();
    let mut values = Vec::with_capacity(unified.fields().len());
    for unified_field in unified.fields() {
        match spec_fields
            .iter()
            .position(|field| field.field_id == unified_field.id)
        {
            Some(index) if index < file_values.len() => {
                values.push(file_values[index].clone());
            }
            _ if !unified_field.required => values.push(None),
            _ => {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot find field {} in spec {}",
                        unified_field.id,
                        spec.spec_id()
                    ),
                ));
            }
        }
    }
    Ok(Struct::from_iter(values))
}

impl TableMetadata {
    /// Java `Partitioning.partitionType(table)` over this table's specs +
    /// current schema.
    ///
    /// Unlike [`Self::default_partition_type`] this is NOT cached — it is
    /// O(specs × fields).
    ///
    /// # Errors
    ///
    /// Propagates [`partition_type`] G1/G2 refusals.
    pub fn unified_partition_type(&self) -> Result<StructType> {
        let specs: Vec<PartitionSpecRef> = self.partition_specs_iter().cloned().collect();
        partition_type(self.current_schema(), &specs)
    }
}

// =============================================================================
// Java `buildPartitionProjectionType`
// =============================================================================

fn build_partition_projection_type(
    type_name: &str,
    schema: &Schema,
    specs: &[PartitionSpecRef],
    projected_field_ids: &HashSet<i32>,
) -> Result<StructType> {
    refuse_unknown_transforms(type_name, specs)?;

    let mut sorted_specs: Vec<&PartitionSpec> = specs.iter().map(AsRef::as_ref).collect();
    sorted_specs.sort_by_key(|spec| std::cmp::Reverse(spec.spec_id()));

    let mut field_map: HashMap<i32, PartitionField> = HashMap::new();
    let mut type_map: HashMap<i32, Type> = HashMap::new();
    let mut name_map: HashMap<i32, String> = HashMap::new();

    for spec in sorted_specs {
        for field in spec.fields() {
            if !projected_field_ids.contains(&field.field_id) {
                continue;
            }
            // G4: resolve THIS field's type only after the id filter. Never
            // call `spec.partition_type(schema)` on the whole spec.
            let field_type = resolve_field_type(field, schema)?;
            match field_map.get(&field.field_id) {
                None => {
                    name_map.insert(field.field_id, field.name.clone());
                    type_map.insert(field.field_id, field_type);
                    field_map.insert(field.field_id, field.clone());
                }
                Some(existing) => {
                    if !equivalent_ignoring_names(field, existing) {
                        return Err(Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "Conflicting partition fields: ['{}', '{}']",
                                format_partition_field(field),
                                format_partition_field(existing)
                            ),
                        ));
                    }
                    // G3: newest spec wrote first; if it was void, take the
                    // non-void field's type. Name stays on the newest spec.
                    if is_void_transform(existing) && !is_void_transform(field) {
                        type_map.insert(field.field_id, field_type);
                        field_map.insert(field.field_id, field.clone());
                    }
                }
            }
        }
    }

    emit_optional_struct(&field_map, &type_map, &name_map)
}

/// `grouping_key_type(None, …)` has no schema for live-source filtering; type
/// resolution still needs a schema for live columns. Dead sources fall back
/// the same way as [`union_partition_types`].
fn build_partition_projection_type_nullable_schema(
    type_name: &str,
    specs: &[PartitionSpecRef],
    projected_field_ids: &HashSet<i32>,
) -> Result<StructType> {
    refuse_unknown_transforms(type_name, specs)?;

    let mut sorted_specs: Vec<&PartitionSpec> = specs.iter().map(AsRef::as_ref).collect();
    sorted_specs.sort_by_key(|spec| std::cmp::Reverse(spec.spec_id()));

    let mut field_map: HashMap<i32, PartitionField> = HashMap::new();
    let mut type_map: HashMap<i32, Type> = HashMap::new();
    let mut name_map: HashMap<i32, String> = HashMap::new();

    for spec in sorted_specs {
        for field in spec.fields() {
            if !projected_field_ids.contains(&field.field_id) {
                continue;
            }
            let field_type = resolve_field_type_without_schema(field);
            match field_map.get(&field.field_id) {
                None => {
                    name_map.insert(field.field_id, field.name.clone());
                    type_map.insert(field.field_id, field_type);
                    field_map.insert(field.field_id, field.clone());
                }
                Some(existing) => {
                    if !equivalent_ignoring_names(field, existing) {
                        return Err(Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "Conflicting partition fields: ['{}', '{}']",
                                format_partition_field(field),
                                format_partition_field(existing)
                            ),
                        ));
                    }
                    if is_void_transform(existing) && !is_void_transform(field) {
                        type_map.insert(field.field_id, field_type);
                        field_map.insert(field.field_id, field.clone());
                    }
                }
            }
        }
    }

    emit_optional_struct(&field_map, &type_map, &name_map)
}

fn emit_optional_struct(
    field_map: &HashMap<i32, PartitionField>,
    type_map: &HashMap<i32, Type>,
    name_map: &HashMap<i32, String>,
) -> Result<StructType> {
    let mut field_ids: Vec<i32> = field_map.keys().copied().collect();
    field_ids.sort_unstable();
    let mut fields = Vec::with_capacity(field_ids.len());
    for field_id in field_ids {
        let name = name_map.get(&field_id).ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                format!("unified name missing for partition field id {field_id}"),
            )
        })?;
        let field_type = type_map.get(&field_id).ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                format!("unified type missing for partition field id {field_id}"),
            )
        })?;
        fields.push(NestedField::optional(field_id, name, field_type.clone()).into());
    }
    Ok(StructType::new(fields))
}

fn refuse_unknown_transforms(type_name: &str, specs: &[PartitionSpecRef]) -> Result<()> {
    let unknown: Vec<String> = specs
        .iter()
        .flat_map(|spec| spec.fields())
        .filter(|field| field.transform == Transform::Unknown)
        .map(|field| field.transform.to_string())
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    Err(Error::new(
        ErrorKind::DataInvalid,
        format!(
            "Cannot build {type_name} type, unknown transforms: [{}]",
            unknown.join(", ")
        ),
    ))
}

fn all_field_ids(specs: &[PartitionSpecRef]) -> HashSet<i32> {
    specs
        .iter()
        .flat_map(|spec| spec.fields())
        .map(|field| field.field_id)
        .collect()
}

fn all_active_field_ids(schema: &Schema, specs: &[PartitionSpecRef]) -> HashSet<i32> {
    specs
        .iter()
        .flat_map(|spec| spec.fields())
        .filter(|field| schema.field_by_id(field.source_id).is_some())
        .map(|field| field.field_id)
        .collect()
}

fn common_active_field_ids(schema: Option<&Schema>, specs: &[PartitionSpecRef]) -> HashSet<i32> {
    let mut specs_iter = specs.iter();
    let Some(first) = specs_iter.next() else {
        return HashSet::new();
    };
    let mut common = active_field_ids(schema, first);
    for spec in specs_iter {
        let active = active_field_ids(schema, spec);
        common.retain(|field_id| active.contains(field_id));
    }
    common
}

fn active_field_ids(schema: Option<&Schema>, spec: &PartitionSpec) -> HashSet<i32> {
    spec.fields()
        .iter()
        .filter(|field| match schema {
            None => true,
            Some(present) => present.field_by_id(field.source_id).is_some(),
        })
        .filter(|field| !is_void_transform(field))
        .map(|field| field.field_id)
        .collect()
}

fn equivalent_ignoring_names(field: &PartitionField, existing: &PartitionField) -> bool {
    field.field_id == existing.field_id
        && field.source_id == existing.source_id
        && compatible_transforms(field.transform, existing.transform)
}

fn compatible_transforms(left: Transform, right: Transform) -> bool {
    left == right || left == Transform::Void || right == Transform::Void
}

fn is_void_transform(field: &PartitionField) -> bool {
    field.transform == Transform::Void
}

/// Resolve a partition field's result type against `schema`.
///
/// Live source → [`Transform::result_type`]. Dead source → Java
/// `PartitionSpec.partitionType` fallback: identity/void/truncate become
/// [`PrimitiveType::Unknown`]; bucket and the temporal transforms keep their
/// fixed `int` result; [`Transform::Unknown`] stays `string`.
fn resolve_field_type(field: &PartitionField, schema: &Schema) -> Result<Type> {
    match schema.field_by_id(field.source_id) {
        Some(source) => field.transform.result_type(&source.field_type),
        None => Ok(dead_source_result_type(field.transform)),
    }
}

fn resolve_field_type_without_schema(field: &PartitionField) -> Type {
    dead_source_result_type(field.transform)
}

fn dead_source_result_type(transform: Transform) -> Type {
    match transform {
        Transform::Bucket(_)
        | Transform::Year
        | Transform::Month
        | Transform::Day
        | Transform::Hour => Type::Primitive(PrimitiveType::Int),
        Transform::Unknown => Type::Primitive(PrimitiveType::String),
        Transform::Identity | Transform::Void | Transform::Truncate(_) => {
            Type::Primitive(PrimitiveType::Unknown)
        }
    }
}

/// Java `PartitionField.toString` analogue used in the G2 message.
fn format_partition_field(field: &PartitionField) -> String {
    format!(
        "{}: {}: {}({})",
        field.field_id, field.name, field.transform, field.source_id
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::spec::{Literal, NestedField, PartitionSpec, UnboundPartitionField};

    // ---------------------------------------------------------------------------
    // Test fixtures — Java `TestPartitioning` SCHEMA + named specs
    // ---------------------------------------------------------------------------

    fn test_schema() -> Schema {
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(3, "category", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("TestPartitioning SCHEMA")
    }

    fn long_xyz_schema() -> Schema {
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "x", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(3, "z", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .expect("x/y/z long schema")
    }

    fn bind(schema: &Schema, spec_id: i32, fields: Vec<UnboundPartitionField>) -> PartitionSpecRef {
        Arc::new(
            PartitionSpec::builder(schema.clone())
                .with_spec_id(spec_id)
                .add_unbound_fields(fields)
                .expect("add unbound fields")
                .build()
                .expect("bind partition spec"),
        )
    }

    fn identity_field(source_id: i32, field_id: i32, name: &str) -> UnboundPartitionField {
        UnboundPartitionField {
            source_id,
            field_id: Some(field_id),
            name: name.to_string(),
            transform: Transform::Identity,
        }
    }

    fn bucket_field(
        source_id: i32,
        field_id: i32,
        name: &str,
        buckets: u32,
    ) -> UnboundPartitionField {
        UnboundPartitionField {
            source_id,
            field_id: Some(field_id),
            name: name.to_string(),
            transform: Transform::Bucket(buckets),
        }
    }

    fn void_field(source_id: i32, field_id: i32, name: &str) -> UnboundPartitionField {
        UnboundPartitionField {
            source_id,
            field_id: Some(field_id),
            name: name.to_string(),
            transform: Transform::Void,
        }
    }

    fn assert_optional_field(
        struct_type: &StructType,
        index: usize,
        id: i32,
        name: &str,
        ty: Type,
    ) {
        let field = &struct_type.fields()[index];
        assert_eq!(field.id, id, "field {index} id");
        assert_eq!(field.name, name, "field {index} name");
        assert_eq!(field.field_type.as_ref(), &ty, "field {index} type");
        assert!(!field.required, "every unified field is optional");
    }

    // ---------------------------------------------------------------------------
    // §4.1 shape pins (Java `TestPartitioning`)
    // ---------------------------------------------------------------------------

    /// Java `testPartitionTypeWithSpecEvolutionInV1Tables`.
    #[test]
    fn partition_type_with_spec_evolution_v1() {
        let schema = test_schema();
        let specs = [
            bind(&schema, 0, vec![identity_field(2, 1000, "data")]),
            bind(&schema, 1, vec![
                identity_field(2, 1000, "data"),
                bucket_field(3, 1001, "category_bucket_8", 8),
            ]),
        ];
        let unified = partition_type(&schema, &specs).expect("v1 evolution unifies");
        assert_eq!(unified.fields().len(), 2);
        assert_optional_field(
            &unified,
            0,
            1000,
            "data",
            Type::Primitive(PrimitiveType::String),
        );
        assert_optional_field(
            &unified,
            1,
            1001,
            "category_bucket_8",
            Type::Primitive(PrimitiveType::Int),
        );
    }

    /// Java `testPartitionTypeWithSpecEvolutionInV2Tables`.
    #[test]
    fn partition_type_with_spec_evolution_v2() {
        let schema = test_schema();
        let specs = [
            bind(&schema, 0, vec![identity_field(2, 1000, "data")]),
            bind(&schema, 1, vec![identity_field(3, 1001, "category")]),
        ];
        let unified = partition_type(&schema, &specs).expect("v2 evolution unifies");
        assert_eq!(unified.fields().len(), 2);
        assert_optional_field(
            &unified,
            0,
            1000,
            "data",
            Type::Primitive(PrimitiveType::String),
        );
        assert_optional_field(
            &unified,
            1,
            1001,
            "category",
            Type::Primitive(PrimitiveType::String),
        );
    }

    /// Java `testPartitionTypeWithRenamesInV1Table` — newest spec's name wins.
    /// Mutation bait: sort specs ascending → this reds (`p1` instead of `p2`).
    #[test]
    fn partition_type_newest_spec_name_wins() {
        let schema = test_schema();
        let specs = [
            bind(&schema, 0, vec![identity_field(2, 1000, "p1")]),
            bind(&schema, 1, vec![
                identity_field(2, 1000, "p1"),
                identity_field(3, 1001, "category"),
            ]),
            bind(&schema, 2, vec![
                identity_field(2, 1000, "p2"),
                identity_field(3, 1001, "category"),
            ]),
        ];
        let unified = partition_type(&schema, &specs).expect("rename unifies");
        assert_eq!(unified.fields().len(), 2);
        assert_optional_field(
            &unified,
            0,
            1000,
            "p2",
            Type::Primitive(PrimitiveType::String),
        );
        assert_optional_field(
            &unified,
            1,
            1001,
            "category",
            Type::Primitive(PrimitiveType::String),
        );
    }

    /// Java `testPartitionTypeWithAddingBackSamePartitionFieldInV1Table`.
    #[test]
    fn partition_type_void_repair_v1_re_add() {
        let schema = test_schema();
        let specs = [
            bind(&schema, 0, vec![identity_field(2, 1000, "data")]),
            bind(&schema, 1, vec![void_field(2, 1000, "data_1000")]),
            bind(&schema, 2, vec![
                void_field(2, 1000, "data_1000"),
                identity_field(2, 1001, "data"),
            ]),
        ];
        let unified = partition_type(&schema, &specs).expect("v1 re-add unifies");
        assert_eq!(unified.fields().len(), 2);
        assert_optional_field(
            &unified,
            0,
            1000,
            "data_1000",
            Type::Primitive(PrimitiveType::String),
        );
        assert_optional_field(
            &unified,
            1,
            1001,
            "data",
            Type::Primitive(PrimitiveType::String),
        );
    }

    /// G3 type repair: newest spec is void (result = source `string`); older
    /// spec is `bucket` (result `int`). Name from newest; type from non-void.
    /// Mutation bait: delete the void-repair branch → type stays `string`.
    #[test]
    fn partition_type_void_repair_uses_non_void_result_type() {
        let schema = test_schema();
        let specs = [
            bind(&schema, 0, vec![bucket_field(
                3,
                1000,
                "category_bucket_8",
                8,
            )]),
            bind(&schema, 1, vec![
                void_field(3, 1000, "category_1000"),
                identity_field(3, 1001, "category"),
            ]),
        ];
        let unified = partition_type(&schema, &specs).expect("void repair unifies");
        assert_eq!(unified.fields().len(), 2);
        assert_optional_field(
            &unified,
            0,
            1000,
            "category_1000",
            Type::Primitive(PrimitiveType::Int),
        );
        assert_optional_field(
            &unified,
            1,
            1001,
            "category",
            Type::Primitive(PrimitiveType::String),
        );
    }

    /// Java `testPartitionTypeWithAddingBackSamePartitionFieldInV2Table`.
    #[test]
    fn partition_type_v2_re_add_reuses_field() {
        let schema = test_schema();
        let specs = [
            bind(&schema, 0, vec![identity_field(2, 1000, "data")]),
            bind(&schema, 1, vec![]),
            bind(&schema, 2, vec![identity_field(2, 1000, "data")]),
        ];
        let unified = partition_type(&schema, &specs).expect("v2 re-add unifies");
        assert_eq!(unified.fields().len(), 1);
        assert_optional_field(
            &unified,
            0,
            1000,
            "data",
            Type::Primitive(PrimitiveType::String),
        );
    }

    /// Java `testPartitionTypeWithIncompatibleSpecEvolution`.
    /// Mutation bait: replace the conflict check with `continue` → this reds.
    #[test]
    fn partition_type_conflicting_fields_refused() {
        let schema = test_schema();
        let specs = [
            bind(&schema, 0, vec![identity_field(2, 1000, "data")]),
            bind(&schema, 1, vec![identity_field(3, 1000, "category")]),
        ];
        let error = partition_type(&schema, &specs).expect_err("G2 must refuse");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().starts_with("Conflicting partition fields"),
            "message was: {}",
            error.message()
        );
    }

    /// Java `testPartitionTypeIgnoreInactiveFields` — the G4 regression pin.
    /// Mutation bait: call `spec.partition_type(schema)` before the id filter
    /// → stage 2/3 `Err` instead of dropping the dead field.
    #[test]
    fn partition_type_ignores_inactive_fields() {
        let full = test_schema();
        let spec0 = bind(&full, 0, vec![
            identity_field(2, 1000, "data"),
            bucket_field(3, 1001, "category_bucket", 8),
        ]);
        let spec1 = bind(&full, 1, vec![identity_field(2, 1000, "data")]);
        let spec2 = bind(&full, 2, vec![]);

        let stage1 = partition_type(&full, std::slice::from_ref(&spec0)).expect("stage 1");
        assert_eq!(stage1.fields().len(), 2);
        assert_optional_field(
            &stage1,
            0,
            1000,
            "data",
            Type::Primitive(PrimitiveType::String),
        );
        assert_optional_field(
            &stage1,
            1,
            1001,
            "category_bucket",
            Type::Primitive(PrimitiveType::Int),
        );

        let no_category = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("schema without category");
        let stage2 = partition_type(&no_category, &[spec0.clone(), spec1.clone()])
            .expect("G4: dropped source must be ignored, not Err");
        assert_eq!(stage2.fields().len(), 1);
        assert_optional_field(
            &stage2,
            0,
            1000,
            "data",
            Type::Primitive(PrimitiveType::String),
        );

        let no_data = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .expect("schema without data");
        let stage3 = partition_type(&no_data, &[spec0, spec1, spec2])
            .expect("G4: both sources dropped → empty, not Err");
        assert!(stage3.fields().is_empty());
    }

    /// Java `testUnionPartitionTypesRetainsDroppedSourceFields`.
    #[test]
    fn union_partition_types_retains_dropped_sources() {
        let full = test_schema();
        let spec0 = bind(&full, 0, vec![
            identity_field(2, 1000, "data"),
            bucket_field(3, 1001, "category_bucket", 8),
        ]);
        let empty = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .expect("schema with both partition sources dropped");
        let unified =
            union_partition_types(&empty, &[spec0]).expect("union retains dropped sources");
        assert_eq!(unified.fields().len(), 2);
        assert_optional_field(
            &unified,
            0,
            1000,
            "data",
            Type::Primitive(PrimitiveType::Unknown),
        );
        assert_optional_field(
            &unified,
            1,
            1001,
            "category_bucket",
            Type::Primitive(PrimitiveType::Int),
        );
    }

    /// Java `testGroupingKeyTypeWithSpecEvolutionInV1Tables` /
    /// `…InV2Tables` (same expected shape).
    #[test]
    fn grouping_key_type_with_spec_evolution() {
        let schema = test_schema();
        let specs = [
            bind(&schema, 0, vec![identity_field(2, 1000, "data")]),
            bind(&schema, 1, vec![
                identity_field(2, 1000, "data"),
                bucket_field(3, 1001, "category_bucket_8", 8),
            ]),
        ];
        let key = grouping_key_type(Some(&schema), &specs).expect("grouping key");
        assert_eq!(key.fields().len(), 1);
        assert_optional_field(
            &key,
            0,
            1000,
            "data",
            Type::Primitive(PrimitiveType::String),
        );
    }

    /// Java `testGroupingKeyTypeWithDroppedPartitionFieldInV1Tables`.
    #[test]
    fn grouping_key_type_with_dropped_partition_field_v1() {
        let schema = test_schema();
        let specs = [
            bind(&schema, 0, vec![
                identity_field(2, 1000, "data"),
                bucket_field(3, 1001, "category_bucket_8", 8),
            ]),
            bind(&schema, 1, vec![
                identity_field(2, 1000, "data"),
                void_field(3, 1001, "category_bucket_8"),
            ]),
        ];
        let key = grouping_key_type(Some(&schema), &specs).expect("v1 drop grouping key");
        assert_eq!(key.fields().len(), 1);
        assert_optional_field(
            &key,
            0,
            1000,
            "data",
            Type::Primitive(PrimitiveType::String),
        );
    }

    /// Java `testGroupingKeyTypeWithDroppedPartitionFieldInV2Tables`.
    #[test]
    fn grouping_key_type_with_dropped_partition_field_v2() {
        let schema = test_schema();
        let specs = [
            bind(&schema, 0, vec![
                identity_field(2, 1000, "data"),
                bucket_field(3, 1001, "category_bucket_8", 8),
            ]),
            bind(&schema, 1, vec![identity_field(2, 1000, "data")]),
        ];
        let key = grouping_key_type(Some(&schema), &specs).expect("v2 drop grouping key");
        assert_eq!(key.fields().len(), 1);
        assert_optional_field(
            &key,
            0,
            1000,
            "data",
            Type::Primitive(PrimitiveType::String),
        );
    }

    /// Java nullable-schema contract: `schema == None` keeps fields whose
    /// source would be dead under a projected schema.
    #[test]
    fn grouping_key_type_null_schema_considers_all_fields() {
        let schema = test_schema();
        let spec = bind(&schema, 0, vec![
            identity_field(2, 1000, "data"),
            identity_field(3, 1001, "category"),
        ]);
        let projected = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("projected schema");
        let with_schema =
            grouping_key_type(Some(&projected), std::slice::from_ref(&spec)).expect("projected");
        assert_eq!(with_schema.fields().len(), 1);
        assert_eq!(with_schema.fields()[0].id, 1000);

        let without = grouping_key_type(None, std::slice::from_ref(&spec)).expect("null schema");
        assert_eq!(without.fields().len(), 2);
        assert_eq!(without.fields()[0].id, 1000);
        assert_eq!(without.fields()[1].id, 1001);
    }

    /// Derived from Java L270–276 /
    /// `testPartitionTypeWithUnknownTransformAndDroppedSourceColumn`.
    /// Mutation bait: drop the unknown-transform gate → this reds.
    #[test]
    fn partition_type_refuses_unknown_transform() {
        let schema = test_schema();
        let spec = bind(&schema, 0, vec![UnboundPartitionField {
            source_id: 2,
            field_id: Some(1000),
            name: "data_custom".to_string(),
            transform: Transform::Unknown,
        }]);
        let error = partition_type(&schema, &[spec]).expect_err("G1 must refuse");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error
                .message()
                .starts_with("Cannot build table partition type, unknown transforms"),
            "message was: {}",
            error.message()
        );
    }

    #[test]
    fn is_partitioned_any_non_void() {
        let schema = test_schema();
        let partitioned = bind(&schema, 0, vec![identity_field(2, 1000, "data")]);
        let voids = bind(&schema, 1, vec![void_field(2, 1000, "data")]);
        let empty = bind(&schema, 2, vec![]);
        assert!(is_partitioned(std::slice::from_ref(&partitioned)));
        assert!(!is_partitioned(std::slice::from_ref(&voids)));
        assert!(!is_partitioned(std::slice::from_ref(&empty)));
        assert!(!is_partitioned(&[voids, empty]));
        assert!(!is_partitioned(&[]));
    }

    // ---------------------------------------------------------------------------
    // coerce_partition units (§1.5 StructProjection.createAllowMissing)
    // ---------------------------------------------------------------------------

    #[test]
    fn coerce_partition_id_matched_reorder() {
        let schema = long_xyz_schema();
        let spec = bind(&schema, 0, vec![
            identity_field(2, 1001, "y"),
            identity_field(1, 1000, "x"),
        ]);
        let unified = StructType::new(vec![
            NestedField::optional(1000, "x", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(1001, "y", Type::Primitive(PrimitiveType::Long)).into(),
        ]);
        let file = Struct::from_iter([Some(Literal::long(9)), Some(Literal::long(7))]);
        let coerced = coerce_partition(&unified, &spec, &schema, &file).expect("coerce");
        assert_eq!(
            coerced,
            Struct::from_iter([Some(Literal::long(7)), Some(Literal::long(9))]),
            "unified (x,y) remapped from spec order (y,x)"
        );
    }

    #[test]
    fn coerce_partition_null_fills_absent_field() {
        let schema = long_xyz_schema();
        let spec = bind(&schema, 0, vec![identity_field(1, 1000, "x")]);
        let unified = StructType::new(vec![
            NestedField::optional(1000, "x", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(1001, "y", Type::Primitive(PrimitiveType::Long)).into(),
        ]);
        let file = Struct::from_iter([Some(Literal::long(7))]);
        let coerced = coerce_partition(&unified, &spec, &schema, &file).expect("coerce");
        assert_eq!(
            coerced,
            Struct::from_iter([Some(Literal::long(7)), None]),
            "y absent from spec → null-fill"
        );
    }

    #[test]
    fn coerce_partition_spec_is_strict_subset() {
        let schema = long_xyz_schema();
        let spec = bind(&schema, 0, vec![
            identity_field(1, 1000, "x"),
            identity_field(2, 1001, "y"),
        ]);
        let unified = StructType::new(vec![
            NestedField::optional(1000, "x", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(1001, "y", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(1002, "z", Type::Primitive(PrimitiveType::Long)).into(),
        ]);
        let file = Struct::from_iter([Some(Literal::long(1)), Some(Literal::long(2))]);
        let coerced = coerce_partition(&unified, &spec, &schema, &file).expect("coerce");
        assert_eq!(
            coerced,
            Struct::from_iter([Some(Literal::long(1)), Some(Literal::long(2)), None])
        );
    }

    /// Mutation bait: replace the id match with a positional index → this
    /// reds (same-typed fields swapped across specs).
    #[test]
    fn coerce_partition_same_typed_fields_swapped_are_not_positional() {
        let schema = long_xyz_schema();
        let spec = bind(&schema, 0, vec![
            identity_field(2, 1001, "y"),
            identity_field(1, 1000, "x"),
        ]);
        // Unified order puts 1001 first so a positional read of spec's
        // (y, x) would look right for the first column and wrong for the
        // second if we only asserted one field — both are checked.
        let unified = StructType::new(vec![
            NestedField::optional(1001, "y", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(1000, "x", Type::Primitive(PrimitiveType::Long)).into(),
        ]);
        let file = Struct::from_iter([Some(Literal::long(9)), Some(Literal::long(7))]);
        let coerced = coerce_partition(&unified, &spec, &schema, &file).expect("coerce");
        assert_eq!(
            coerced,
            Struct::from_iter([Some(Literal::long(9)), Some(Literal::long(7))])
        );

        let unified_ascending = StructType::new(vec![
            NestedField::optional(1000, "x", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(1001, "y", Type::Primitive(PrimitiveType::Long)).into(),
        ]);
        let coerced_asc =
            coerce_partition(&unified_ascending, &spec, &schema, &file).expect("coerce asc");
        assert_eq!(
            coerced_asc,
            Struct::from_iter([Some(Literal::long(7)), Some(Literal::long(9))]),
            "positional impl would emit (9, 7)"
        );
    }

    /// Mutation bait: emit `NestedField::required` → null-fill of a missing
    /// field becomes `DataInvalid` instead of `None`.
    #[test]
    fn coerce_partition_required_missing_field_is_data_invalid() {
        let schema = long_xyz_schema();
        let spec = bind(&schema, 0, vec![identity_field(1, 1000, "x")]);
        let unified = StructType::new(vec![
            NestedField::required(1001, "y", Type::Primitive(PrimitiveType::Long)).into(),
        ]);
        let file = Struct::from_iter([Some(Literal::long(7))]);
        let error =
            coerce_partition(&unified, &spec, &schema, &file).expect_err("required missing field");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
    }
}
