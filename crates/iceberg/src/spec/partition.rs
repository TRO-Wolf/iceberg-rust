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

/*!
 * Partitioning
 */
use std::sync::Arc;

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

use super::transform::Transform;
use super::{NestedField, Schema, SchemaRef, StructType, Type};
use crate::spec::Struct;
use crate::{Error, ErrorKind, Result};

pub(crate) const UNPARTITIONED_LAST_ASSIGNED_ID: i32 = 999;
pub(crate) const DEFAULT_PARTITION_SPEC_ID: i32 = 0;

/// Partition fields capture the transform from table data to partition values.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone, TypedBuilder)]
#[serde(rename_all = "kebab-case")]
pub struct PartitionField {
    /// A source column id from the table’s schema
    pub source_id: i32,
    /// A partition field id that is used to identify a partition field and is unique within a partition spec.
    /// In v2 table metadata, it is unique across all partition specs.
    pub field_id: i32,
    /// A partition name.
    pub name: String,
    /// A transform that is applied to the source column to produce a partition value.
    pub transform: Transform,
}

impl PartitionField {
    /// To unbound partition field
    pub fn into_unbound(self) -> UnboundPartitionField {
        self.into()
    }
}

/// Reference to [`PartitionSpec`].
pub type PartitionSpecRef = Arc<PartitionSpec>;
/// Partition spec that defines how to produce a tuple of partition values from a record.
///
/// A [`PartitionSpec`] is originally obtained by binding an [`UnboundPartitionSpec`] to a schema and is
/// only guaranteed to be valid for that schema. The main difference between [`PartitionSpec`] and
/// [`UnboundPartitionSpec`] is that the former has field ids assigned,
/// while field ids are optional for [`UnboundPartitionSpec`].
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
#[serde(rename_all = "kebab-case")]
pub struct PartitionSpec {
    /// Identifier for PartitionSpec
    spec_id: i32,
    /// Details of the partition spec
    fields: Vec<PartitionField>,
}

impl PartitionSpec {
    /// Create a new partition spec builder with the given schema.
    pub fn builder(schema: impl Into<SchemaRef>) -> PartitionSpecBuilder {
        PartitionSpecBuilder::new(schema)
    }

    /// Fields of the partition spec
    pub fn fields(&self) -> &[PartitionField] {
        &self.fields
    }

    /// Spec id of the partition spec
    pub fn spec_id(&self) -> i32 {
        self.spec_id
    }

    /// Get a new unpartitioned partition spec
    pub fn unpartition_spec() -> Self {
        Self {
            spec_id: DEFAULT_PARTITION_SPEC_ID,
            fields: vec![],
        }
    }

    pub(crate) fn from_fields_unchecked(spec_id: i32, fields: Vec<PartitionField>) -> Self {
        Self { spec_id, fields }
    }

    /// Returns if the partition spec is unpartitioned.
    ///
    /// A [`PartitionSpec`] is unpartitioned if it has no fields or all fields are [`Transform::Void`] transform.
    pub fn is_unpartitioned(&self) -> bool {
        self.fields.is_empty() || self.fields.iter().all(|f| f.transform == Transform::Void)
    }

    /// Returns the partition type of this partition spec.
    pub fn partition_type(&self, schema: &Schema) -> Result<StructType> {
        PartitionSpecBuilder::partition_type(&self.fields, schema)
    }

    /// Convert to unbound partition spec
    pub fn into_unbound(self) -> UnboundPartitionSpec {
        self.into()
    }

    /// Change the spec id of the partition spec
    pub fn with_spec_id(self, spec_id: i32) -> Self {
        Self { spec_id, ..self }
    }

    /// Check if this partition spec has sequential partition ids.
    /// Sequential ids start from 1000 and increment by 1 for each field.
    /// This is required for spec version 1
    pub fn has_sequential_ids(&self) -> bool {
        has_sequential_ids(self.fields.iter().map(|f| f.field_id))
    }

    /// Get the highest field id in the partition spec.
    pub fn highest_field_id(&self) -> Option<i32> {
        self.fields.iter().map(|f| f.field_id).max()
    }

    /// Check if this partition spec is compatible with another partition spec. Returns true if the
    /// partition spec is equal to the other spec with partition field ids ignored and spec_id
    /// ignored. The following must be identical: * The number of fields * Field order * Field names
    /// * Source column ids * Transforms.
    pub fn is_compatible_with(&self, other: &PartitionSpec) -> bool {
        if self.fields.len() != other.fields.len() {
            return false;
        }

        for (this_field, other_field) in self.fields.iter().zip(other.fields.iter()) {
            if this_field.source_id != other_field.source_id
                || this_field.name != other_field.name
                || this_field.transform != other_field.transform
            {
                return false;
            }
        }

        true
    }

    /// Returns the partition path: `name=value` pairs joined by `/`. Both sides of each pair are
    /// URL-escaped like Java (see [`escape_partition_path_component`]), so a `/`, `=` or space in a
    /// name or a value cannot forge path structure. A NULL value renders `name=null`.
    pub fn partition_to_path(&self, data: &Struct, schema: SchemaRef) -> String {
        let field_types = self.lenient_partition_field_types(&schema);

        self.fields
            .iter()
            .enumerate()
            .map(|(index, field)| {
                match Self::render_partition_field(
                    field,
                    field_types.get(index).and_then(Option::as_ref),
                    data,
                    index,
                ) {
                    Ok(rendered) => rendered,
                    Err(error) => {
                        tracing::warn!(
                            ?error,
                            spec_id = self.spec_id,
                            partition_field = field.name.as_str(),
                            index,
                            "partition value is not renderable under this spec/schema; rendering \
                             `null` for it (Java renders `null` for an absent partition value)"
                        );
                        escaped_partition_pair(&field.name, "null")
                    }
                }
            })
            .join("/")
    }

    /// The fallible sibling of [`PartitionSpec::partition_to_path`]. Returns the SAME string on
    /// `Ok`, escaping included. # Errors Returns an error when the `(spec, schema, data)` triple is
    /// not self-consistent: the partition type is not derivable under `schema`, the tuple is
    /// shorter than the spec, a value is not a primitive literal, or a value's literal kind fails
    /// `PrimitiveType::compatible`.
    pub fn try_partition_to_path(&self, data: &Struct, schema: SchemaRef) -> Result<String> {
        let partition_type = self.partition_type(&schema)?;
        let mut rendered = Vec::with_capacity(self.fields.len());
        for (index, (field, field_type)) in
            self.fields.iter().zip(partition_type.fields()).enumerate()
        {
            rendered.push(Self::render_partition_field(
                field,
                Some(&field_type.field_type),
                data,
                index,
            )?);
        }
        Ok(rendered.join("/"))
    }

    /// Validates that `(self, schema, data)` is self-consistent, and builds no path string. Same
    /// rules as [`Self::try_partition_to_path`]. [`PartitionKey::new`] uses it, so key construction
    /// does not pay for human-string rendering and escaping.
    pub(crate) fn validate_partition_data(&self, data: &Struct, schema: &Schema) -> Result<()> {
        let partition_type = self.partition_type(schema)?;
        for (index, (field, field_type)) in
            self.fields.iter().zip(partition_type.fields()).enumerate()
        {
            Self::check_partition_field(field, Some(&field_type.field_type), data, index)?;
        }
        Ok(())
    }

    /// Per-field partition types for the TOTAL path. Lenient: a field whose source column is absent
    /// from `schema`, or whose transform rejects that column's type, yields `None` instead of
    /// failing the call. Java's `PartitionSpec.partitionType()` substitutes `UnknownType`.
    fn lenient_partition_field_types(&self, schema: &Schema) -> Vec<Option<Type>> {
        self.fields
            .iter()
            .map(|field| {
                schema
                    .field_by_id(field.source_id)
                    .and_then(|source| field.transform.result_type(&source.field_type).ok())
            })
            .collect()
    }

    /// Shared acceptance check for one partition field. Returns the checked `(field_type, literal)`
    /// when a non-null value is present, or `None` when the rendered human string is the literal
    /// `"null"` (missing void slot, NULL slot). Errors match [`Self::try_partition_to_path`].
    fn check_partition_field<'a>(
        field: &PartitionField,
        field_type: Option<&'a Type>,
        data: &'a Struct,
        index: usize,
    ) -> Result<Option<(&'a Type, &'a super::Literal)>> {
        let Some(slot) = data.fields().get(index) else {
            // Past the end of the tuple. A `void` field's value is always null, so a missing slot
            // carries no information. An all-`void` spec's `is_unpartitioned()` callers produce it.
            if field.transform == Transform::Void {
                return Ok(None);
            }
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Partition tuple has {} value(s) but partition field `{}` is at index {index}",
                    data.fields().len(),
                    field.name
                ),
            ));
        };

        let Some(field_type) = field_type else {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!(
                    "Cannot derive the partition type of field `{}`: no column with source column \
                     id {} in the schema in use",
                    field.name, field.source_id
                ),
            ));
        };

        // A NULL partition value is legal and renders `null` (Java `toHumanString(type, null)`).
        let Some(literal) = slot.as_ref() else {
            return Ok(None);
        };

        let Some(primitive_value) = literal.as_primitive_literal() else {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Partition value for field `{}` must be a primitive literal",
                    field.name
                ),
            ));
        };
        let Some(primitive_type) = field_type.as_primitive_type() else {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Partition field `{}` has non-primitive type `{field_type}` but its value is a \
                     primitive literal",
                    field.name
                ),
            ));
        };
        if !primitive_type.compatible(&primitive_value) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Partition value for field `{}` is not compatible with its partition type \
                     `{primitive_type}`",
                    field.name
                ),
            ));
        }

        Ok(Some((field_type, literal)))
    }

    /// Render one escaped `name=value` pair, or describe why it cannot render. `field_type` is
    /// `None` when the field's partition type is not derivable under the schema in use.
    ///
    /// Every `Ok` return renders without aborting: the value is absent or NULL, or a primitive
    /// literal whose kind `PrimitiveType::compatible` accepts. That set is a subset of what
    /// `Display for Datum` can format.
    fn render_partition_field(
        field: &PartitionField,
        field_type: Option<&Type>,
        data: &Struct,
        index: usize,
    ) -> Result<String> {
        match Self::check_partition_field(field, field_type, data, index)? {
            None => Ok(escaped_partition_pair(&field.name, "null")),
            Some((field_type, literal)) => Ok(escaped_partition_pair(
                &field.name,
                &field.transform.to_human_string(field_type, Some(literal)),
            )),
        }
    }
}

/// The UPPERCASE hex alphabet `java.net.URLEncoder` emits.
const UPPER_HEX: &[u8; 16] = b"0123456789ABCDEF";

/// Escape one side of a partition path's `name=value` pair exactly as Java escapes it.
///
/// Java `PartitionSpec.escape` is `java.net.URLEncoder.encode(s, "UTF-8")`. That is
/// `application/x-www-form-urlencoded`, NOT RFC-3986: `A-Z a-z 0-9 - _ . *` pass through, a space
/// becomes `+`, and every other character becomes one UPPERCASE `%XX` group per UTF-8 byte. A byte
/// loop matches Java's `char` loop, and Java's unpaired-surrogate case cannot occur on a Rust `str`.
fn escape_partition_path_component(component: &str) -> String {
    let mut escaped = String::with_capacity(component.len());
    for &byte in component.as_bytes() {
        match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'*' => {
                escaped.push(char::from(byte));
            }
            b' ' => escaped.push('+'),
            _ => {
                escaped.push('%');
                escaped.push(char::from(UPPER_HEX[usize::from(byte >> 4)]));
                escaped.push(char::from(UPPER_HEX[usize::from(byte & 0x0F)]));
            }
        }
    }
    escaped
}

/// Render one escaped `name=value` pair, the body of Java's `partitionToPath` loop. The `=` and the
/// joining `/` are path STRUCTURE and stay raw. Every pair the partition path emits is built here,
/// so no branch can miss the escaping.
///
/// Four call sites reach it: the rendered-value branch and three `name=null` branches. Each one is
/// pinned individually in `partition_path_escaping_tests`.
fn escaped_partition_pair(name: &str, value: &str) -> String {
    format!(
        "{}={}",
        escape_partition_path_component(name),
        escape_partition_path_component(value)
    )
}

/// A partition key represents a specific partition in a table, containing the partition spec,
/// schema, and the actual partition values.
#[derive(Clone, Debug)]
pub struct PartitionKey {
    /// The partition spec that contains the partition fields.
    spec: PartitionSpec,
    /// The schema to which the partition spec is bound.
    schema: SchemaRef,
    /// Partition fields' values in struct.
    data: Struct,
}

impl PartitionKey {
    /// Creates a new partition key with the given spec, schema, and data. # Errors Returns
    /// [`ErrorKind::DataInvalid`] or [`ErrorKind::Unexpected`] when `(spec, schema, data)` is not
    /// self-consistent, under the same rules as [`PartitionSpec::try_partition_to_path`]. An
    /// invalid partition tuple is unrepresentable as a [`PartitionKey`], matching Java's
    /// `StructTransform`.
    pub fn new(spec: PartitionSpec, schema: SchemaRef, data: Struct) -> Result<Self> {
        // Validate without building a path string, so the write path does not pay for rendering.
        let data = spec.validated_promoted_partition(data, schema.as_ref())?;
        Ok(Self { spec, schema, data })
    }

    /// Creates a new partition key from another key, with new data. Validates the new data against
    /// this key's spec and schema, like [`Self::new`].
    pub fn copy_with_data(&self, data: Struct) -> Result<Self> {
        Self::new(self.spec.clone(), self.schema.clone(), data)
    }

    /// Generates a partition path based on the partition values.
    pub fn to_path(&self) -> String {
        self.spec.partition_to_path(&self.data, self.schema.clone())
    }

    /// Returns `true` if the partition key is absent (`None`)
    /// or represents an unpartitioned spec.
    pub fn is_effectively_none(partition_key: Option<&PartitionKey>) -> bool {
        match partition_key {
            None => true,
            Some(pk) => pk.spec.is_unpartitioned(),
        }
    }

    /// Returns the associated [`PartitionSpec`].
    pub fn spec(&self) -> &PartitionSpec {
        &self.spec
    }

    /// Returns the associated [`SchemaRef`].
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Returns the associated [`Struct`].
    pub fn data(&self) -> &Struct {
        &self.data
    }
}

/// Reference to [`UnboundPartitionSpec`].
pub type UnboundPartitionSpecRef = Arc<UnboundPartitionSpec>;
/// Unbound partition field can be built without a schema and later bound to a schema.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone, TypedBuilder)]
#[serde(rename_all = "kebab-case")]
pub struct UnboundPartitionField {
    /// A source column id from the table’s schema
    pub source_id: i32,
    /// A partition field id that is used to identify a partition field and is unique within a partition spec.
    /// In v2 table metadata, it is unique across all partition specs.
    #[builder(default, setter(strip_option(fallback = field_id_opt)))]
    pub field_id: Option<i32>,
    /// A partition name.
    pub name: String,
    /// A transform that is applied to the source column to produce a partition value.
    pub transform: Transform,
}

/// Unbound partition spec can be built without a schema and later bound to a schema.
/// They are used to transport schema information as part of the REST specification.
/// The main difference to [`PartitionSpec`] is that the field ids are optional.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone, Default)]
#[serde(rename_all = "kebab-case")]
pub struct UnboundPartitionSpec {
    /// Identifier for PartitionSpec
    pub(crate) spec_id: Option<i32>,
    /// Details of the partition spec
    pub(crate) fields: Vec<UnboundPartitionField>,
}

impl UnboundPartitionSpec {
    /// Create unbound partition spec builder
    pub fn builder() -> UnboundPartitionSpecBuilder {
        UnboundPartitionSpecBuilder::default()
    }

    /// Bind this unbound partition spec to a schema.
    pub fn bind(self, schema: impl Into<SchemaRef>) -> Result<PartitionSpec> {
        PartitionSpecBuilder::new_from_unbound(self, schema)?.build()
    }

    /// Spec id of the partition spec
    pub fn spec_id(&self) -> Option<i32> {
        self.spec_id
    }

    /// Fields of the partition spec
    pub fn fields(&self) -> &[UnboundPartitionField] {
        &self.fields
    }

    /// Change the spec id of the partition spec
    pub fn with_spec_id(self, spec_id: i32) -> Self {
        Self {
            spec_id: Some(spec_id),
            ..self
        }
    }
}

fn has_sequential_ids(field_ids: impl Iterator<Item = i32>) -> bool {
    for (index, field_id) in field_ids.enumerate() {
        let expected_id = (UNPARTITIONED_LAST_ASSIGNED_ID as i64)
            .checked_add(1)
            .and_then(|id| id.checked_add(index as i64))
            .unwrap_or(i64::MAX);

        if field_id as i64 != expected_id {
            return false;
        }
    }

    true
}

impl From<PartitionField> for UnboundPartitionField {
    fn from(field: PartitionField) -> Self {
        UnboundPartitionField {
            source_id: field.source_id,
            field_id: Some(field.field_id),
            name: field.name,
            transform: field.transform,
        }
    }
}

impl From<PartitionSpec> for UnboundPartitionSpec {
    fn from(spec: PartitionSpec) -> Self {
        UnboundPartitionSpec {
            spec_id: Some(spec.spec_id),
            fields: spec.fields.into_iter().map(Into::into).collect(),
        }
    }
}

/// Create a new UnboundPartitionSpec
#[derive(Debug, Default)]
pub struct UnboundPartitionSpecBuilder {
    spec_id: Option<i32>,
    fields: Vec<UnboundPartitionField>,
}

impl UnboundPartitionSpecBuilder {
    /// Create a new partition spec builder with the given schema.
    pub fn new() -> Self {
        Self {
            spec_id: None,
            fields: vec![],
        }
    }

    /// Set the spec id for the partition spec.
    pub fn with_spec_id(mut self, spec_id: i32) -> Self {
        self.spec_id = Some(spec_id);
        self
    }

    /// Add a new partition field to the partition spec from an unbound partition field.
    pub fn add_partition_field(
        self,
        source_id: i32,
        target_name: impl ToString,
        transformation: Transform,
    ) -> Result<Self> {
        let field = UnboundPartitionField {
            source_id,
            field_id: None,
            name: target_name.to_string(),
            transform: transformation,
        };
        self.add_partition_field_internal(field)
    }

    /// Add multiple partition fields to the partition spec.
    pub fn add_partition_fields(
        self,
        fields: impl IntoIterator<Item = UnboundPartitionField>,
    ) -> Result<Self> {
        let mut builder = self;
        for field in fields {
            builder = builder.add_partition_field_internal(field)?;
        }
        Ok(builder)
    }

    fn add_partition_field_internal(mut self, field: UnboundPartitionField) -> Result<Self> {
        // Java parity: `Bucket.get` and `Truncate.get` reject an invalid parameter at construction,
        // so Java can never hold one. The builder is the earliest Rust door, because the `Transform`
        // enum payload is public and cannot be guarded.
        field.transform.validate()?;
        self.check_name_set_and_unique(&field.name)?;
        self.check_for_redundant_partitions(field.source_id, &field.transform)?;
        if let Some(partition_field_id) = field.field_id {
            self.check_partition_id_unique(partition_field_id)?;
        }
        self.fields.push(field);
        Ok(self)
    }

    /// Build the unbound partition spec.
    pub fn build(self) -> UnboundPartitionSpec {
        UnboundPartitionSpec {
            spec_id: self.spec_id,
            fields: self.fields,
        }
    }
}

/// Create valid partition specs for a given schema.
#[derive(Debug)]
pub struct PartitionSpecBuilder {
    spec_id: Option<i32>,
    last_assigned_field_id: i32,
    fields: Vec<UnboundPartitionField>,
    schema: SchemaRef,
}

impl PartitionSpecBuilder {
    /// Create a new partition spec builder with the given schema.
    pub fn new(schema: impl Into<SchemaRef>) -> Self {
        Self {
            spec_id: None,
            fields: vec![],
            last_assigned_field_id: UNPARTITIONED_LAST_ASSIGNED_ID,
            schema: schema.into(),
        }
    }

    /// Create a new partition spec builder from an existing unbound partition spec.
    pub fn new_from_unbound(
        unbound: UnboundPartitionSpec,
        schema: impl Into<SchemaRef>,
    ) -> Result<Self> {
        let mut builder =
            Self::new(schema).with_spec_id(unbound.spec_id.unwrap_or(DEFAULT_PARTITION_SPEC_ID));

        for field in unbound.fields {
            builder = builder.add_unbound_field(field)?;
        }
        Ok(builder)
    }

    /// Set the last assigned field id. Set it when a new spec is created for existing
    /// `TableMetadata`. A `field_id` must be unique in V2 metadata, so pass the highest id used
    /// before.
    pub fn with_last_assigned_field_id(mut self, last_assigned_field_id: i32) -> Self {
        self.last_assigned_field_id = last_assigned_field_id;
        self
    }

    /// Set the spec id for the partition spec.
    pub fn with_spec_id(mut self, spec_id: i32) -> Self {
        self.spec_id = Some(spec_id);
        self
    }

    /// Add a new partition field to the partition spec.
    pub fn add_partition_field(
        self,
        source_name: impl AsRef<str>,
        target_name: impl Into<String>,
        transform: Transform,
    ) -> Result<Self> {
        let source_id = self
            .schema
            .field_by_name(source_name.as_ref())
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot find source column with name: {} in schema",
                        source_name.as_ref()
                    ),
                )
            })?
            .id;
        let field = UnboundPartitionField {
            source_id,
            field_id: None,
            name: target_name.into(),
            transform,
        };

        self.add_unbound_field(field)
    }

    /// Add a new partition field to the partition spec. Uses the partition field id when it is set,
    /// otherwise assigns a new one.
    pub fn add_unbound_field(mut self, field: UnboundPartitionField) -> Result<Self> {
        // Java parity: see `UnboundPartitionSpecBuilder::add_partition_field_internal` — an
        // invalid bucket/truncate parameter is rejected before any other spec check.
        field.transform.validate()?;
        self.check_name_set_and_unique(&field.name)?;
        self.check_for_redundant_partitions(field.source_id, &field.transform)?;
        Self::check_name_does_not_collide_with_schema(&field, &self.schema)?;
        Self::check_transform_compatibility(&field, &self.schema)?;
        if let Some(partition_field_id) = field.field_id {
            self.check_partition_id_unique(partition_field_id)?;
        }

        // Non-fallible from here
        self.fields.push(field);
        Ok(self)
    }

    /// Wrapper around `with_unbound_fields` to add multiple partition fields.
    pub fn add_unbound_fields(
        self,
        fields: impl IntoIterator<Item = UnboundPartitionField>,
    ) -> Result<Self> {
        let mut builder = self;
        for field in fields {
            builder = builder.add_unbound_field(field)?;
        }
        Ok(builder)
    }

    /// Build a bound partition spec with the given schema.
    pub fn build(self) -> Result<PartitionSpec> {
        let fields = Self::set_field_ids(self.fields, self.last_assigned_field_id)?;
        Ok(PartitionSpec {
            spec_id: self.spec_id.unwrap_or(DEFAULT_PARTITION_SPEC_ID),
            fields,
        })
    }

    fn set_field_ids(
        fields: Vec<UnboundPartitionField>,
        last_assigned_field_id: i32,
    ) -> Result<Vec<PartitionField>> {
        let mut last_assigned_field_id = last_assigned_field_id;
        // Already assigned partition ids. Skip one of these when we see it during iteration.
        let assigned_ids = fields
            .iter()
            .filter_map(|f| f.field_id)
            .collect::<std::collections::HashSet<_>>();

        fn _check_add_1(prev: i32) -> Result<i32> {
            prev.checked_add(1).ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Cannot assign more partition ids. Overflow.",
                )
            })
        }

        let mut bound_fields = Vec::with_capacity(fields.len());
        for field in fields.into_iter() {
            let partition_field_id = if let Some(partition_field_id) = field.field_id {
                last_assigned_field_id = std::cmp::max(last_assigned_field_id, partition_field_id);
                partition_field_id
            } else {
                last_assigned_field_id = _check_add_1(last_assigned_field_id)?;
                while assigned_ids.contains(&last_assigned_field_id) {
                    last_assigned_field_id = _check_add_1(last_assigned_field_id)?;
                }
                last_assigned_field_id
            };

            bound_fields.push(PartitionField {
                source_id: field.source_id,
                field_id: partition_field_id,
                name: field.name,
                transform: field.transform,
            })
        }

        Ok(bound_fields)
    }

    /// Returns the partition type of this partition spec.
    fn partition_type(fields: &Vec<PartitionField>, schema: &Schema) -> Result<StructType> {
        let mut struct_fields = Vec::with_capacity(fields.len());
        for partition_field in fields {
            let field = schema
                .field_by_id(partition_field.source_id)
                .ok_or_else(|| {
                    Error::new(
                        // Unreachable: check_transform_compatibility proved the source field exists.
                        ErrorKind::Unexpected,
                        format!(
                            "No column with source column id {} in schema {:?}",
                            partition_field.source_id, schema
                        ),
                    )
                })?;
            let res_type = partition_field.transform.result_type(&field.field_type)?;
            let field =
                NestedField::optional(partition_field.field_id, &partition_field.name, res_type)
                    .into();
            struct_fields.push(field);
        }
        Ok(StructType::new(struct_fields))
    }

    /// Ensure the partition name is unique among schema column names. A duplicate is allowed only
    /// when the partition is sourced from that same column and the transform is identity or void.
    ///
    /// The `void` exception mirrors Java's `checkAndAddPartitionName(name, sourceId)`, where the
    /// rule is the name to source-id correspondence. A V1 removed field is re-added as `void(name)`
    /// under the same name and id; without the exception that replacement fails at bind time.
    fn check_name_does_not_collide_with_schema(
        field: &UnboundPartitionField,
        schema: &Schema,
    ) -> Result<()> {
        match schema.field_by_name(field.name.as_str()) {
            Some(schema_collision) => {
                let is_identity_or_void =
                    field.transform == Transform::Identity || field.transform == Transform::Void;
                if is_identity_or_void {
                    if schema_collision.id == field.source_id {
                        Ok(())
                    } else {
                        Err(Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "Cannot create identity partition sourced from different field in schema. Field name '{}' has id `{}` in schema but partition source id is `{}`",
                                field.name, schema_collision.id, field.source_id
                            ),
                        ))
                    }
                } else {
                    Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Cannot create partition with name: '{}' that conflicts with schema field and is not an identity transform.",
                            field.name
                        ),
                    ))
                }
            }
            None => Ok(()),
        }
    }

    /// Ensure that the transformation of the field is compatible with type of the field
    /// in the schema. Implicitly also checks if the source field exists in the schema.
    fn check_transform_compatibility(field: &UnboundPartitionField, schema: &Schema) -> Result<()> {
        let schema_field = schema.field_by_id(field.source_id).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot find partition source field with id `{}` in schema",
                    field.source_id
                ),
            )
        })?;

        if field.transform != Transform::Void {
            if !schema_field.field_type.is_primitive() {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot partition by non-primitive source field: '{}'.",
                        schema_field.field_type
                    ),
                ));
            }

            if field
                .transform
                .result_type(&schema_field.field_type)
                .is_err()
            {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Invalid source type: '{}' for transform: '{}'.",
                        schema_field.field_type,
                        field.transform.dedup_name()
                    ),
                ));
            }
        }

        Ok(())
    }
}

/// Contains checks that are common to both PartitionSpecBuilder and UnboundPartitionSpecBuilder
trait CorePartitionSpecValidator {
    /// Ensure that the partition name is unique among the partition fields and is not empty.
    fn check_name_set_and_unique(&self, name: &str) -> Result<()> {
        if name.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Cannot use empty partition name",
            ));
        }

        if self.fields().iter().any(|f| f.name == name) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot use partition name more than once: {name}"),
            ));
        }
        Ok(())
    }

    /// For a single source-column transformations must be unique.
    fn check_for_redundant_partitions(&self, source_id: i32, transform: &Transform) -> Result<()> {
        let collision = self.fields().iter().find(|f| {
            f.source_id == source_id && f.transform.dedup_name() == transform.dedup_name()
        });

        if let Some(collision) = collision {
            Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot add redundant partition with source id `{}` and transform `{}`. A partition with the same source id and transform already exists with name `{}`",
                    source_id,
                    transform.dedup_name(),
                    collision.name
                ),
            ))
        } else {
            Ok(())
        }
    }

    /// Check field / partition_id unique within the partition spec if set
    fn check_partition_id_unique(&self, field_id: i32) -> Result<()> {
        if self.fields().iter().any(|f| f.field_id == Some(field_id)) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot use field id more than once in one PartitionSpec: {field_id}"),
            ));
        }

        Ok(())
    }

    fn fields(&self) -> &Vec<UnboundPartitionField>;
}

impl CorePartitionSpecValidator for PartitionSpecBuilder {
    fn fields(&self) -> &Vec<UnboundPartitionField> {
        &self.fields
    }
}

impl CorePartitionSpecValidator for UnboundPartitionSpecBuilder {
    fn fields(&self) -> &Vec<UnboundPartitionField> {
        &self.fields
    }
}

#[cfg(test)]
#[path = "partition_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "partition_path_totalisation_tests.rs"]
mod partition_path_totalisation_tests;

#[cfg(test)]
#[path = "partition_path_escaping_tests.rs"]
mod partition_path_escaping_tests;
