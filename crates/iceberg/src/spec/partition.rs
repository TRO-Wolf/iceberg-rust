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

    /// Check if this partition spec is compatible with another partition spec.
    ///
    /// Returns true if the partition spec is equal to the other spec with partition field ids ignored and
    /// spec_id ignored. The following must be identical:
    /// * The number of fields
    /// * Field order
    /// * Field names
    /// * Source column ids
    /// * Transforms
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
    ///
    /// # Notes
    ///
    /// The call is TOTAL. An inconsistent `(spec, schema, data)` triple renders `null` for the
    /// offending field and warns. This is wider than Java, which throws for all but a short tuple.
    /// The write and commit callers cannot report failure and must not abort a long-running engine.
    /// [`PartitionSpec::try_partition_to_path`] returns this same string on `Ok`, with typed errors.
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
    /// `Ok`, escaping included.
    ///
    /// # Errors
    ///
    /// Returns an error when the `(spec, schema, data)` triple is not self-consistent: the
    /// partition type is not derivable under `schema`, the tuple is shorter than the spec, a value
    /// is not a primitive literal, or a value's literal kind fails `PrimitiveType::compatible`.
    /// A NULL value, and a missing value for a [`Transform::Void`] field, are NOT errors.
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
    /// Creates a new partition key with the given spec, schema, and data.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::DataInvalid`] or [`ErrorKind::Unexpected`] when `(spec, schema, data)`
    /// is not self-consistent, under the same rules as
    /// [`PartitionSpec::try_partition_to_path`]. An invalid partition tuple is unrepresentable as a
    /// [`PartitionKey`], matching Java's `StructTransform`. A NULL slot is legal, and an all-`void`
    /// spec may pair with an empty tuple.
    pub fn new(spec: PartitionSpec, schema: SchemaRef, data: Struct) -> Result<Self> {
        // Validate without building a path string, so the write path does not pay for rendering.
        spec.validate_partition_data(&data, schema.as_ref())?;
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
mod tests {
    use super::*;
    use crate::spec::{Literal, PrimitiveType, Type};

    #[test]
    fn test_partition_spec() {
        let spec = r#"
        {
        "spec-id": 1,
        "fields": [ {
            "source-id": 4,
            "field-id": 1000,
            "name": "ts_day",
            "transform": "day"
            }, {
            "source-id": 1,
            "field-id": 1001,
            "name": "id_bucket",
            "transform": "bucket[16]"
            }, {
            "source-id": 2,
            "field-id": 1002,
            "name": "id_truncate",
            "transform": "truncate[4]"
            } ]
        }
        "#;

        let partition_spec: PartitionSpec = serde_json::from_str(spec).unwrap();
        assert_eq!(4, partition_spec.fields[0].source_id);
        assert_eq!(1000, partition_spec.fields[0].field_id);
        assert_eq!("ts_day", partition_spec.fields[0].name);
        assert_eq!(Transform::Day, partition_spec.fields[0].transform);

        assert_eq!(1, partition_spec.fields[1].source_id);
        assert_eq!(1001, partition_spec.fields[1].field_id);
        assert_eq!("id_bucket", partition_spec.fields[1].name);
        assert_eq!(Transform::Bucket(16), partition_spec.fields[1].transform);

        assert_eq!(2, partition_spec.fields[2].source_id);
        assert_eq!(1002, partition_spec.fields[2].field_id);
        assert_eq!("id_truncate", partition_spec.fields[2].name);
        assert_eq!(Transform::Truncate(4), partition_spec.fields[2].transform);
    }

    // RISK (crown jewel): table-metadata JSON carrying bucket[0] used to deserialize fine and abort
    // later with a divide-by-zero at partition-value computation. Any hostile or corrupt metadata
    // file triggers it. It must fail AT DESERIALIZATION with DataInvalid, like Java's
    // `TableMetadataParser` -> `Transforms.fromString` -> `Bucket.get`.
    #[test]
    fn test_table_metadata_with_invalid_transform_parameter_fails_deserialization() {
        fn metadata_json(transform: &str) -> String {
            format!(
                r#"
                {{
                    "format-version": 2,
                    "table-uuid": "9c12d441-03fe-4693-9a96-a0705ddf69c1",
                    "location": "s3://bucket/test/location",
                    "last-sequence-number": 1,
                    "last-updated-ms": 1602638573590,
                    "last-column-id": 1,
                    "current-schema-id": 0,
                    "schemas": [
                        {{
                            "type": "struct",
                            "schema-id": 0,
                            "fields": [
                                {{
                                    "id": 1,
                                    "name": "x",
                                    "required": true,
                                    "type": "long"
                                }}
                            ]
                        }}
                    ],
                    "default-spec-id": 0,
                    "partition-specs": [
                        {{
                            "spec-id": 0,
                            "fields": [
                                {{
                                    "source-id": 1,
                                    "field-id": 1000,
                                    "name": "x_partition",
                                    "transform": "{transform}"
                                }}
                            ]
                        }}
                    ],
                    "last-partition-id": 1000,
                    "default-sort-order-id": 0,
                    "sort-orders": [
                        {{
                            "order-id": 0,
                            "fields": []
                        }}
                    ],
                    "properties": {{}},
                    "snapshots": [],
                    "statistics": [],
                    "snapshot-log": [],
                    "metadata-log": []
                }}
                "#
            )
        }

        // CONTROL first (docs/testing.md sabotage discipline): the same metadata with a legal
        // transform parses, so the sabotaged variants below fail on the transform bound.
        let control =
            serde_json::from_str::<crate::spec::TableMetadata>(&metadata_json("bucket[16]"))
                .expect("control metadata with bucket[16] must deserialize");
        assert_eq!(
            control.default_partition_spec().fields()[0].transform,
            Transform::Bucket(16)
        );

        for sabotaged in ["bucket[0]", "truncate[0]", "bucket[2147483648]"] {
            let serde_error =
                serde_json::from_str::<crate::spec::TableMetadata>(&metadata_json(sabotaged))
                    .unwrap_err();
            // Mirror the production conversion in TableMetadata::read_from
            // (`serde_json::from_slice(...)?` routes through `Error::from`).
            let error = Error::from(serde_error);
            assert_eq!(error.kind(), ErrorKind::DataInvalid, "{sabotaged}");
        }

        // The untagged TableMetadataEnum swallows the Java precondition text, so pin the message at
        // the partition-spec JSON door — the same bytes-on-disk shape the metadata carries.
        let serde_error = serde_json::from_str::<PartitionSpec>(
            r#"{
                "spec-id": 0,
                "fields": [
                    {
                        "source-id": 1,
                        "field-id": 1000,
                        "name": "x_partition",
                        "transform": "bucket[0]"
                    }
                ]
            }"#,
        )
        .unwrap_err();
        assert!(
            serde_error
                .to_string()
                .contains("Invalid number of buckets: 0 (must be > 0)"),
            "expected the Java precondition text, got: {serde_error}"
        );
    }

    // RISK: the bound builder is the programmatic route into a PartitionSpec. Java can never hold a
    // Bucket(0) or Truncate(0), so admitting one builds a spec that aborts the process at apply time.
    #[test]
    fn test_partition_spec_builder_rejects_zero_parameter_transforms() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .expect("valid schema");

        let error = PartitionSpec::builder(schema.clone())
            .add_partition_field("id", "id_bucket", Transform::Bucket(0))
            .expect_err("bucket[0] must be rejected by the bound builder");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error
                .message()
                .contains("Invalid number of buckets: 0 (must be > 0)"),
            "message must match the Java precondition text, got: {}",
            error.message()
        );

        let error = PartitionSpec::builder(schema.clone())
            .add_partition_field("id", "id_truncate", Transform::Truncate(0))
            .expect_err("truncate[0] must be rejected by the bound builder");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);

        // Over-broadened-guard pin: a legal parameter still passes every builder check.
        PartitionSpec::builder(schema)
            .add_partition_field("id", "id_bucket", Transform::Bucket(16))
            .expect("bucket[16] is legal")
            .build()
            .expect("legal spec must build");
    }

    // RISK: the unbound builder feeds catalog create-table requests, under the same
    // reject-at-construction contract as the bound builder.
    #[test]
    fn test_unbound_partition_spec_builder_rejects_zero_parameter_transforms() {
        let error = UnboundPartitionSpec::builder()
            .add_partition_field(1, "id_bucket", Transform::Bucket(0))
            .expect_err("bucket[0] must be rejected by the unbound builder");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);

        let error = UnboundPartitionSpec::builder()
            .add_partition_field(1, "id_truncate", Transform::Truncate(0))
            .expect_err("truncate[0] must be rejected by the unbound builder");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);

        // Over-broadened-guard pin: a legal parameter is still accepted.
        UnboundPartitionSpec::builder()
            .add_partition_field(1, "id_bucket", Transform::Bucket(16))
            .expect("bucket[16] is legal");
    }

    #[test]
    fn test_is_unpartitioned() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
                NestedField::required(
                    2,
                    "name",
                    Type::Primitive(crate::spec::PrimitiveType::String),
                )
                .into(),
            ])
            .build()
            .unwrap();
        let partition_spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(1)
            .build()
            .unwrap();
        assert!(
            partition_spec.is_unpartitioned(),
            "Empty partition spec should be unpartitioned"
        );

        let partition_spec = PartitionSpec::builder(schema.clone())
            .add_unbound_fields(vec![
                UnboundPartitionField::builder()
                    .source_id(1)
                    .name("id".to_string())
                    .transform(Transform::Identity)
                    .build(),
                UnboundPartitionField::builder()
                    .source_id(2)
                    .name("name_string".to_string())
                    .transform(Transform::Void)
                    .build(),
            ])
            .unwrap()
            .with_spec_id(1)
            .build()
            .unwrap();
        assert!(
            !partition_spec.is_unpartitioned(),
            "Partition spec with one non void transform should not be unpartitioned"
        );

        let partition_spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(1)
            .add_unbound_fields(vec![
                UnboundPartitionField::builder()
                    .source_id(1)
                    .name("id_void".to_string())
                    .transform(Transform::Void)
                    .build(),
                UnboundPartitionField::builder()
                    .source_id(2)
                    .name("name_void".to_string())
                    .transform(Transform::Void)
                    .build(),
            ])
            .unwrap()
            .build()
            .unwrap();
        assert!(
            partition_spec.is_unpartitioned(),
            "Partition spec with all void field should be unpartitioned"
        );
    }

    #[test]
    fn test_unbound_partition_spec() {
        let spec = r#"
		{
		"spec-id": 1,
		"fields": [ {
			"source-id": 4,
			"field-id": 1000,
			"name": "ts_day",
			"transform": "day"
			}, {
			"source-id": 1,
			"field-id": 1001,
			"name": "id_bucket",
			"transform": "bucket[16]"
			}, {
			"source-id": 2,
			"field-id": 1002,
			"name": "id_truncate",
			"transform": "truncate[4]"
			} ]
		}
		"#;

        let partition_spec: UnboundPartitionSpec = serde_json::from_str(spec).unwrap();
        assert_eq!(Some(1), partition_spec.spec_id);

        assert_eq!(4, partition_spec.fields[0].source_id);
        assert_eq!(Some(1000), partition_spec.fields[0].field_id);
        assert_eq!("ts_day", partition_spec.fields[0].name);
        assert_eq!(Transform::Day, partition_spec.fields[0].transform);

        assert_eq!(1, partition_spec.fields[1].source_id);
        assert_eq!(Some(1001), partition_spec.fields[1].field_id);
        assert_eq!("id_bucket", partition_spec.fields[1].name);
        assert_eq!(Transform::Bucket(16), partition_spec.fields[1].transform);

        assert_eq!(2, partition_spec.fields[2].source_id);
        assert_eq!(Some(1002), partition_spec.fields[2].field_id);
        assert_eq!("id_truncate", partition_spec.fields[2].name);
        assert_eq!(Transform::Truncate(4), partition_spec.fields[2].transform);

        let spec = r#"
		{
		"fields": [ {
			"source-id": 4,
			"name": "ts_day",
			"transform": "day"
			} ]
		}
		"#;
        let partition_spec: UnboundPartitionSpec = serde_json::from_str(spec).unwrap();
        assert_eq!(None, partition_spec.spec_id);

        assert_eq!(4, partition_spec.fields[0].source_id);
        assert_eq!(None, partition_spec.fields[0].field_id);
        assert_eq!("ts_day", partition_spec.fields[0].name);
        assert_eq!(Transform::Day, partition_spec.fields[0].transform);
    }

    #[test]
    fn test_new_unpartition() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
                NestedField::required(
                    2,
                    "name",
                    Type::Primitive(crate::spec::PrimitiveType::String),
                )
                .into(),
            ])
            .build()
            .unwrap();
        let partition_spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(0)
            .build()
            .unwrap();
        let partition_type = partition_spec.partition_type(&schema).unwrap();
        assert_eq!(0, partition_type.fields().len());

        let unpartition_spec = PartitionSpec::unpartition_spec();
        assert_eq!(partition_spec, unpartition_spec);
    }

    #[test]
    fn test_partition_type() {
        let spec = r#"
            {
            "spec-id": 1,
            "fields": [ {
                "source-id": 4,
                "field-id": 1000,
                "name": "ts_day",
                "transform": "day"
                }, {
                "source-id": 1,
                "field-id": 1001,
                "name": "id_bucket",
                "transform": "bucket[16]"
                }, {
                "source-id": 2,
                "field-id": 1002,
                "name": "id_truncate",
                "transform": "truncate[4]"
                } ]
            }
            "#;

        let partition_spec: PartitionSpec = serde_json::from_str(spec).unwrap();
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
                NestedField::required(
                    2,
                    "name",
                    Type::Primitive(crate::spec::PrimitiveType::String),
                )
                .into(),
                NestedField::required(
                    3,
                    "ts",
                    Type::Primitive(crate::spec::PrimitiveType::Timestamp),
                )
                .into(),
                NestedField::required(
                    4,
                    "ts_day",
                    Type::Primitive(crate::spec::PrimitiveType::Timestamp),
                )
                .into(),
                NestedField::required(
                    5,
                    "id_bucket",
                    Type::Primitive(crate::spec::PrimitiveType::Int),
                )
                .into(),
                NestedField::required(
                    6,
                    "id_truncate",
                    Type::Primitive(crate::spec::PrimitiveType::Int),
                )
                .into(),
            ])
            .build()
            .unwrap();

        let partition_type = partition_spec.partition_type(&schema).unwrap();
        assert_eq!(3, partition_type.fields().len());
        assert_eq!(
            *partition_type.fields()[0],
            NestedField::optional(
                partition_spec.fields[0].field_id,
                &partition_spec.fields[0].name,
                Type::Primitive(crate::spec::PrimitiveType::Date)
            )
        );
        assert_eq!(
            *partition_type.fields()[1],
            NestedField::optional(
                partition_spec.fields[1].field_id,
                &partition_spec.fields[1].name,
                Type::Primitive(crate::spec::PrimitiveType::Int)
            )
        );
        assert_eq!(
            *partition_type.fields()[2],
            NestedField::optional(
                partition_spec.fields[2].field_id,
                &partition_spec.fields[2].name,
                Type::Primitive(crate::spec::PrimitiveType::String)
            )
        );
    }

    #[test]
    fn test_partition_empty() {
        let spec = r#"
            {
            "spec-id": 1,
            "fields": []
            }
            "#;

        let partition_spec: PartitionSpec = serde_json::from_str(spec).unwrap();
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
                NestedField::required(
                    2,
                    "name",
                    Type::Primitive(crate::spec::PrimitiveType::String),
                )
                .into(),
                NestedField::required(
                    3,
                    "ts",
                    Type::Primitive(crate::spec::PrimitiveType::Timestamp),
                )
                .into(),
                NestedField::required(
                    4,
                    "ts_day",
                    Type::Primitive(crate::spec::PrimitiveType::Timestamp),
                )
                .into(),
                NestedField::required(
                    5,
                    "id_bucket",
                    Type::Primitive(crate::spec::PrimitiveType::Int),
                )
                .into(),
                NestedField::required(
                    6,
                    "id_truncate",
                    Type::Primitive(crate::spec::PrimitiveType::Int),
                )
                .into(),
            ])
            .build()
            .unwrap();

        let partition_type = partition_spec.partition_type(&schema).unwrap();
        assert_eq!(0, partition_type.fields().len());
    }

    #[test]
    fn test_partition_error() {
        let spec = r#"
        {
        "spec-id": 1,
        "fields": [ {
            "source-id": 4,
            "field-id": 1000,
            "name": "ts_day",
            "transform": "day"
            }, {
            "source-id": 1,
            "field-id": 1001,
            "name": "id_bucket",
            "transform": "bucket[16]"
            }, {
            "source-id": 2,
            "field-id": 1002,
            "name": "id_truncate",
            "transform": "truncate[4]"
            } ]
        }
        "#;

        let partition_spec: PartitionSpec = serde_json::from_str(spec).unwrap();
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
                NestedField::required(
                    2,
                    "name",
                    Type::Primitive(crate::spec::PrimitiveType::String),
                )
                .into(),
            ])
            .build()
            .unwrap();

        assert!(partition_spec.partition_type(&schema).is_err());
    }

    #[test]
    fn test_builder_disallow_duplicate_names() {
        UnboundPartitionSpec::builder()
            .add_partition_field(1, "ts_day".to_string(), Transform::Day)
            .unwrap()
            .add_partition_field(2, "ts_day".to_string(), Transform::Day)
            .unwrap_err();
    }

    /// A two-column schema whose second column `v` (id 2) is variant — input for the variant
    /// partition-source rejection pins.
    fn schema_with_variant_column() -> Schema {
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
                NestedField::optional(2, "v", Type::Variant).into(),
            ])
            .build()
            .unwrap()
    }

    // RISK: a variant column must NOT be a partition source for any value-producing transform. Java
    // `PartitionSpec.checkCompatibility` rejects it at the non-primitive door, before canTransform,
    // and `Identity.UNSUPPORTED_TYPES` lists VARIANT. Partitioning by variant writes tuples with no
    // single-value representation, which is silent layout corruption.
    #[test]
    fn test_variant_rejected_as_partition_source_for_identity_and_bucket() {
        for transform in [
            Transform::Identity,
            Transform::Bucket(16),
            Transform::Truncate(4),
            Transform::Year,
            Transform::Month,
            Transform::Day,
            Transform::Hour,
        ] {
            let error = PartitionSpec::builder(schema_with_variant_column())
                .add_unbound_field(UnboundPartitionField {
                    source_id: 2,
                    field_id: None,
                    name: "v_part".to_string(),
                    transform,
                })
                .expect_err("a variant partition source must be rejected");
            assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
            assert!(
                error
                    .message()
                    .contains("Cannot partition by non-primitive source field: 'variant'"),
                "{transform} must reject variant at the non-primitive door (Java fires it before \
                 canTransform), got: {}",
                error.message()
            );
        }
    }

    // RISK: the VOID transform must still ACCEPT a variant source. Java's checkCompatibility skips
    // `alwaysNull()` fields, and over-firing breaks partition-field removal on a variant schema.
    #[test]
    fn test_variant_accepted_as_void_partition_source() {
        let spec = PartitionSpec::builder(schema_with_variant_column())
            .add_unbound_field(UnboundPartitionField {
                source_id: 2,
                field_id: None,
                name: "v_void".to_string(),
                transform: Transform::Void,
            })
            .expect("void on a variant source is legal (Java skips alwaysNull)")
            .build()
            .expect("build the spec");
        assert_eq!(spec.fields().len(), 1);
        assert_eq!(spec.fields()[0].transform, Transform::Void);
    }

    #[test]
    fn test_builder_disallow_duplicate_field_ids() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
                NestedField::required(
                    2,
                    "name",
                    Type::Primitive(crate::spec::PrimitiveType::String),
                )
                .into(),
            ])
            .build()
            .unwrap();
        PartitionSpec::builder(schema.clone())
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: Some(1000),
                name: "id".to_string(),
                transform: Transform::Identity,
            })
            .unwrap()
            .add_unbound_field(UnboundPartitionField {
                source_id: 2,
                field_id: Some(1000),
                name: "id_bucket".to_string(),
                transform: Transform::Bucket(16),
            })
            .unwrap_err();
    }

    #[test]
    fn test_builder_auto_assign_field_ids() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
                NestedField::required(
                    2,
                    "name",
                    Type::Primitive(crate::spec::PrimitiveType::String),
                )
                .into(),
                NestedField::required(
                    3,
                    "ts",
                    Type::Primitive(crate::spec::PrimitiveType::Timestamp),
                )
                .into(),
            ])
            .build()
            .unwrap();
        let spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(1)
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                name: "id".to_string(),
                transform: Transform::Identity,
                field_id: Some(1012),
            })
            .unwrap()
            .add_unbound_field(UnboundPartitionField {
                source_id: 2,
                name: "name_void".to_string(),
                transform: Transform::Void,
                field_id: None,
            })
            .unwrap()
            // Should keep its ID even if its lower
            .add_unbound_field(UnboundPartitionField {
                source_id: 3,
                name: "year".to_string(),
                transform: Transform::Year,
                field_id: Some(1),
            })
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(1012, spec.fields[0].field_id);
        assert_eq!(1013, spec.fields[1].field_id);
        assert_eq!(1, spec.fields[2].field_id);
    }

    #[test]
    fn test_builder_valid_schema() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
                NestedField::required(
                    2,
                    "name",
                    Type::Primitive(crate::spec::PrimitiveType::String),
                )
                .into(),
            ])
            .build()
            .unwrap();

        PartitionSpec::builder(schema.clone())
            .with_spec_id(1)
            .build()
            .unwrap();

        let spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(1)
            .add_partition_field("id", "id_bucket[16]", Transform::Bucket(16))
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(spec, PartitionSpec {
            spec_id: 1,
            fields: vec![PartitionField {
                source_id: 1,
                field_id: 1000,
                name: "id_bucket[16]".to_string(),
                transform: Transform::Bucket(16),
            }],
        });
        assert_eq!(
            spec.partition_type(&schema).unwrap(),
            StructType::new(vec![
                NestedField::optional(1000, "id_bucket[16]", Type::Primitive(PrimitiveType::Int))
                    .into()
            ])
        )
    }

    #[test]
    fn test_collision_with_schema_name() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
            ])
            .build()
            .unwrap();

        PartitionSpec::builder(schema.clone())
            .with_spec_id(1)
            .build()
            .unwrap();

        let err = PartitionSpec::builder(schema)
            .with_spec_id(1)
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: None,
                name: "id".to_string(),
                transform: Transform::Bucket(16),
            })
            .unwrap_err();
        assert!(err.message().contains("conflicts with schema"))
    }

    #[test]
    fn test_builder_collision_is_ok_for_identity_transforms() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
                NestedField::required(
                    2,
                    "number",
                    Type::Primitive(crate::spec::PrimitiveType::Int),
                )
                .into(),
            ])
            .build()
            .unwrap();

        PartitionSpec::builder(schema.clone())
            .with_spec_id(1)
            .build()
            .unwrap();

        PartitionSpec::builder(schema.clone())
            .with_spec_id(1)
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: None,
                name: "id".to_string(),
                transform: Transform::Identity,
            })
            .unwrap()
            .build()
            .unwrap();

        // Not OK for different source id
        PartitionSpec::builder(schema)
            .with_spec_id(1)
            .add_unbound_field(UnboundPartitionField {
                source_id: 2,
                field_id: None,
                name: "id".to_string(),
                transform: Transform::Identity,
            })
            .unwrap_err();
    }

    // RISK (Java parity): a `void` partition named after its OWN source column must be accepted (the
    // V1 removed-field replacement), while a `void` named after a DIFFERENT column stays rejected.
    // Java's `checkAndAddPartitionName(name, sourceId)` rules on the name to source-id
    // correspondence, not on the transform.
    #[test]
    fn test_builder_collision_is_ok_for_void_named_after_its_own_source() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
                NestedField::required(
                    2,
                    "number",
                    Type::Primitive(crate::spec::PrimitiveType::Int),
                )
                .into(),
            ])
            .build()
            .unwrap();

        // OK: void("id") sourced from id 1 (== the colliding schema field's id).
        PartitionSpec::builder(schema.clone())
            .with_spec_id(1)
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: Some(1000),
                name: "id".to_string(),
                transform: Transform::Void,
            })
            .unwrap()
            .build()
            .unwrap();

        // Not OK: void("id") sourced from a DIFFERENT column (id 2).
        PartitionSpec::builder(schema)
            .with_spec_id(1)
            .add_unbound_field(UnboundPartitionField {
                source_id: 2,
                field_id: Some(1000),
                name: "id".to_string(),
                transform: Transform::Void,
            })
            .unwrap_err();
    }

    #[test]
    fn test_builder_all_source_ids_must_exist() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
                NestedField::required(
                    2,
                    "name",
                    Type::Primitive(crate::spec::PrimitiveType::String),
                )
                .into(),
                NestedField::required(
                    3,
                    "ts",
                    Type::Primitive(crate::spec::PrimitiveType::Timestamp),
                )
                .into(),
            ])
            .build()
            .unwrap();

        // Valid
        PartitionSpec::builder(schema.clone())
            .with_spec_id(1)
            .add_unbound_fields(vec![
                UnboundPartitionField {
                    source_id: 1,
                    field_id: None,
                    name: "id_bucket".to_string(),
                    transform: Transform::Bucket(16),
                },
                UnboundPartitionField {
                    source_id: 2,
                    field_id: None,
                    name: "name".to_string(),
                    transform: Transform::Identity,
                },
            ])
            .unwrap()
            .build()
            .unwrap();

        // Invalid
        PartitionSpec::builder(schema)
            .with_spec_id(1)
            .add_unbound_fields(vec![
                UnboundPartitionField {
                    source_id: 1,
                    field_id: None,
                    name: "id_bucket".to_string(),
                    transform: Transform::Bucket(16),
                },
                UnboundPartitionField {
                    source_id: 4,
                    field_id: None,
                    name: "name".to_string(),
                    transform: Transform::Identity,
                },
            ])
            .unwrap_err();
    }

    #[test]
    fn test_builder_disallows_redundant() {
        let err = UnboundPartitionSpec::builder()
            .with_spec_id(1)
            .add_partition_field(1, "id_bucket[16]".to_string(), Transform::Bucket(16))
            .unwrap()
            .add_partition_field(
                1,
                "id_bucket_with_other_name".to_string(),
                Transform::Bucket(16),
            )
            .unwrap_err();
        assert!(err.message().contains("redundant partition"));
    }

    #[test]
    fn test_builder_incompatible_transforms_disallowed() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
            ])
            .build()
            .unwrap();

        PartitionSpec::builder(schema)
            .with_spec_id(1)
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: None,
                name: "id_year".to_string(),
                transform: Transform::Year,
            })
            .unwrap_err();
    }

    #[test]
    fn test_build_unbound_specs_without_partition_id() {
        let spec = UnboundPartitionSpec::builder()
            .with_spec_id(1)
            .add_partition_fields(vec![UnboundPartitionField {
                source_id: 1,
                field_id: None,
                name: "id_bucket[16]".to_string(),
                transform: Transform::Bucket(16),
            }])
            .unwrap()
            .build();

        assert_eq!(spec, UnboundPartitionSpec {
            spec_id: Some(1),
            fields: vec![UnboundPartitionField {
                source_id: 1,
                field_id: None,
                name: "id_bucket[16]".to_string(),
                transform: Transform::Bucket(16),
            }]
        });
    }

    #[test]
    fn test_is_compatible_with() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
                NestedField::required(
                    2,
                    "name",
                    Type::Primitive(crate::spec::PrimitiveType::String),
                )
                .into(),
            ])
            .build()
            .unwrap();

        let partition_spec_1 = PartitionSpec::builder(schema.clone())
            .with_spec_id(1)
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: None,
                name: "id_bucket".to_string(),
                transform: Transform::Bucket(16),
            })
            .unwrap()
            .build()
            .unwrap();

        let partition_spec_2 = PartitionSpec::builder(schema)
            .with_spec_id(1)
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: None,
                name: "id_bucket".to_string(),
                transform: Transform::Bucket(16),
            })
            .unwrap()
            .build()
            .unwrap();

        assert!(partition_spec_1.is_compatible_with(&partition_spec_2));
    }

    #[test]
    fn test_not_compatible_with_transform_different() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
            ])
            .build()
            .unwrap();

        let partition_spec_1 = PartitionSpec::builder(schema.clone())
            .with_spec_id(1)
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: None,
                name: "id_bucket".to_string(),
                transform: Transform::Bucket(16),
            })
            .unwrap()
            .build()
            .unwrap();

        let partition_spec_2 = PartitionSpec::builder(schema)
            .with_spec_id(1)
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: None,
                name: "id_bucket".to_string(),
                transform: Transform::Bucket(32),
            })
            .unwrap()
            .build()
            .unwrap();

        assert!(!partition_spec_1.is_compatible_with(&partition_spec_2));
    }

    #[test]
    fn test_not_compatible_with_source_id_different() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
                NestedField::required(
                    2,
                    "name",
                    Type::Primitive(crate::spec::PrimitiveType::String),
                )
                .into(),
            ])
            .build()
            .unwrap();

        let partition_spec_1 = PartitionSpec::builder(schema.clone())
            .with_spec_id(1)
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: None,
                name: "id_bucket".to_string(),
                transform: Transform::Bucket(16),
            })
            .unwrap()
            .build()
            .unwrap();

        let partition_spec_2 = PartitionSpec::builder(schema)
            .with_spec_id(1)
            .add_unbound_field(UnboundPartitionField {
                source_id: 2,
                field_id: None,
                name: "id_bucket".to_string(),
                transform: Transform::Bucket(16),
            })
            .unwrap()
            .build()
            .unwrap();

        assert!(!partition_spec_1.is_compatible_with(&partition_spec_2));
    }

    #[test]
    fn test_not_compatible_with_order_different() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
                NestedField::required(
                    2,
                    "name",
                    Type::Primitive(crate::spec::PrimitiveType::String),
                )
                .into(),
            ])
            .build()
            .unwrap();

        let partition_spec_1 = PartitionSpec::builder(schema.clone())
            .with_spec_id(1)
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: None,
                name: "id_bucket".to_string(),
                transform: Transform::Bucket(16),
            })
            .unwrap()
            .add_unbound_field(UnboundPartitionField {
                source_id: 2,
                field_id: None,
                name: "name".to_string(),
                transform: Transform::Identity,
            })
            .unwrap()
            .build()
            .unwrap();

        let partition_spec_2 = PartitionSpec::builder(schema)
            .with_spec_id(1)
            .add_unbound_field(UnboundPartitionField {
                source_id: 2,
                field_id: None,
                name: "name".to_string(),
                transform: Transform::Identity,
            })
            .unwrap()
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: None,
                name: "id_bucket".to_string(),
                transform: Transform::Bucket(16),
            })
            .unwrap()
            .build()
            .unwrap();

        assert!(!partition_spec_1.is_compatible_with(&partition_spec_2));
    }

    #[test]
    fn test_highest_field_id_unpartitioned() {
        let spec = PartitionSpec::builder(Schema::builder().with_fields(vec![]).build().unwrap())
            .with_spec_id(1)
            .build()
            .unwrap();

        assert!(spec.highest_field_id().is_none());
    }

    #[test]
    fn test_highest_field_id() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
                NestedField::required(
                    2,
                    "name",
                    Type::Primitive(crate::spec::PrimitiveType::String),
                )
                .into(),
            ])
            .build()
            .unwrap();

        let spec = PartitionSpec::builder(schema)
            .with_spec_id(1)
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: Some(1001),
                name: "id".to_string(),
                transform: Transform::Identity,
            })
            .unwrap()
            .add_unbound_field(UnboundPartitionField {
                source_id: 2,
                field_id: Some(1000),
                name: "name".to_string(),
                transform: Transform::Identity,
            })
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(Some(1001), spec.highest_field_id());
    }

    #[test]
    fn test_has_sequential_ids() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
                NestedField::required(
                    2,
                    "name",
                    Type::Primitive(crate::spec::PrimitiveType::String),
                )
                .into(),
            ])
            .build()
            .unwrap();

        let spec = PartitionSpec::builder(schema)
            .with_spec_id(1)
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: Some(1000),
                name: "id".to_string(),
                transform: Transform::Identity,
            })
            .unwrap()
            .add_unbound_field(UnboundPartitionField {
                source_id: 2,
                field_id: Some(1001),
                name: "name".to_string(),
                transform: Transform::Identity,
            })
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(1000, spec.fields[0].field_id);
        assert_eq!(1001, spec.fields[1].field_id);
        assert!(spec.has_sequential_ids());
    }

    #[test]
    fn test_sequential_ids_must_start_at_1000() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
                NestedField::required(
                    2,
                    "name",
                    Type::Primitive(crate::spec::PrimitiveType::String),
                )
                .into(),
            ])
            .build()
            .unwrap();

        let spec = PartitionSpec::builder(schema)
            .with_spec_id(1)
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: Some(999),
                name: "id".to_string(),
                transform: Transform::Identity,
            })
            .unwrap()
            .add_unbound_field(UnboundPartitionField {
                source_id: 2,
                field_id: Some(1000),
                name: "name".to_string(),
                transform: Transform::Identity,
            })
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(999, spec.fields[0].field_id);
        assert_eq!(1000, spec.fields[1].field_id);
        assert!(!spec.has_sequential_ids());
    }

    #[test]
    fn test_sequential_ids_must_have_no_gaps() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(crate::spec::PrimitiveType::Int))
                    .into(),
                NestedField::required(
                    2,
                    "name",
                    Type::Primitive(crate::spec::PrimitiveType::String),
                )
                .into(),
            ])
            .build()
            .unwrap();

        let spec = PartitionSpec::builder(schema)
            .with_spec_id(1)
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: Some(1000),
                name: "id".to_string(),
                transform: Transform::Identity,
            })
            .unwrap()
            .add_unbound_field(UnboundPartitionField {
                source_id: 2,
                field_id: Some(1002),
                name: "name".to_string(),
                transform: Transform::Identity,
            })
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(1000, spec.fields[0].field_id);
        assert_eq!(1002, spec.fields[1].field_id);
        assert!(!spec.has_sequential_ids());
    }

    #[test]
    fn test_partition_to_path() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(3, "timestamp", Type::Primitive(PrimitiveType::Timestamp))
                    .into(),
                NestedField::required(4, "empty", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();

        let spec = PartitionSpec::builder(schema.clone())
            .add_partition_field("id", "id", Transform::Identity)
            .unwrap()
            .add_partition_field("name", "name", Transform::Identity)
            .unwrap()
            .add_partition_field("timestamp", "ts_hour", Transform::Hour)
            .unwrap()
            .add_partition_field("empty", "empty_void", Transform::Void)
            .unwrap()
            .build()
            .unwrap();

        let data = Struct::from_iter([
            Some(Literal::int(42)),
            Some(Literal::string("alice")),
            Some(Literal::int(1000)),
            Some(Literal::string("empty")),
        ]);

        assert_eq!(
            spec.partition_to_path(&data, schema.into()),
            "id=42/name=alice/ts_hour=1000/empty_void=null"
        );
    }
}

#[cfg(test)]
mod partition_path_totalisation_tests {
    //! WG3-L2 pins: [`PartitionSpec::partition_to_path`] is TOTAL. An inconsistent
    //! `(spec, schema, tuple)` triple renders `name=null` for the offending field and warns, and it
    //! never aborts. Four abort vectors were reachable before this change:
    //!
    //! | # | input | pre-change abort |
    //! |---|---|---|
    //! | V1 | tuple shorter than the spec | `data[i]` index out of bounds |
    //! | V2 | source column absent from the schema | `partition_type(..).unwrap()` |
    //! | V3 | non-primitive partition-field type + a primitive value | `as_primitive_type().unwrap()` |
    //! | V4 | value literal kind incompatible with the field type | `Display for Datum`'s `unreachable!()` |
    //!
    //! Java is LENIENT for V1: `PartitionData.get(pos)` returns `null` past the end of the tuple and
    //! `Transform.toHumanString(type, null)` renders `"null"`. Java throws for the other three.
    //! [`PartitionSpec::try_partition_to_path`] surfaces all four as typed errors.

    use std::sync::Arc;

    use super::*;
    use crate::spec::{Datum, Literal, PrimitiveLiteral, PrimitiveType, Type};

    /// `identity(x: long)` + `identity(y: long)` over a two-column schema.
    fn two_field_spec() -> (SchemaRef, PartitionSpec) {
        let schema: SchemaRef = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "x", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .expect("two-column schema must build"),
        );
        let spec = PartitionSpec::builder(schema.clone())
            .add_partition_field("x", "x", Transform::Identity)
            .expect("identity(x) is a legal partition field")
            .add_partition_field("y", "y", Transform::Identity)
            .expect("identity(y) is a legal partition field")
            .build()
            .expect("the two-field spec must build");
        (schema, spec)
    }

    // ============================================================================================
    // NULL partition values stay legal: a NULL tuple slot is a first-class Iceberg value.
    // ============================================================================================

    /// A `PartitionKey` carrying a NULL value renders `name=null` on both paths. Java renders a
    /// null partition value as the literal `"null"` (`Transform.toHumanString(type, null)`).
    #[test]
    fn partition_key_new_accepts_null_value() {
        let (schema, spec) = two_field_spec();
        let data = Struct::from_iter([Some(Literal::long(5)), None]);
        let key = PartitionKey::new(spec.clone(), schema.clone(), data.clone())
            .expect("PartitionKey::new: valid partition tuple");

        assert_eq!(key.to_path(), "x=5/y=null");
        assert_eq!(
            spec.try_partition_to_path(&data, schema)
                .expect("a NULL partition value is legal, not an anomaly"),
            "x=5/y=null"
        );
    }

    /// `PartitionKey::new` rejects an invalid triple (a short non-void tuple) with a typed error.
    #[test]
    fn partition_key_new_rejects_short_non_void_tuple() {
        let (schema, spec) = two_field_spec();
        let data = Struct::from_iter([Some(Literal::long(5))]); // missing y
        let err = PartitionKey::new(spec, schema, data)
            .expect_err("a short non-void tuple must not construct a PartitionKey");
        assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
    }

    /// Unit 3: `PartitionKey::new` rejects an incompatible literal kind.
    #[test]
    fn partition_key_new_rejects_incompatible_literal() {
        let (schema, spec) = two_field_spec();
        let data = Struct::from_iter([Some(Literal::long(5)), Some(Literal::int(7))]);
        let err = PartitionKey::new(spec, schema, data)
            .expect_err("an Int in a Long partition slot must not construct a PartitionKey");
        assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
    }

    /// Unit 3: all-void + empty tuple remains constructible (the void trap).
    #[test]
    fn partition_key_new_accepts_all_void_empty_tuple() {
        let schema: SchemaRef = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "x", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .expect("one-column schema must build"),
        );
        let spec = PartitionSpec::builder(schema.clone())
            .add_partition_field("x", "x_void", Transform::Void)
            .expect("void(x) is a legal partition field")
            .build()
            .expect("the all-void spec must build");
        let key = PartitionKey::new(spec, schema, Struct::empty())
            .expect("all-void + empty tuple is a legitimate PartitionKey");
        assert_eq!(key.to_path(), "x_void=null");
    }

    // ============================================================================================
    // V1 — tuple shorter than the spec.
    // ============================================================================================

    /// V1: a tuple shorter than the spec renders the missing fields as `null` (Java's past-end
    /// `PartitionData.get` leniency) instead of indexing out of bounds.
    #[test]
    fn test_partition_to_path_short_tuple_renders_null_instead_of_aborting() {
        let (schema, spec) = two_field_spec();
        let data = Struct::from_iter([Some(Literal::long(5))]);

        assert_eq!(spec.partition_to_path(&data, schema), "x=5/y=null");
    }

    /// The same short tuple is a typed `DataInvalid` on the fallible path.
    #[test]
    fn test_try_partition_to_path_short_tuple_errors() {
        let (schema, spec) = two_field_spec();
        let data = Struct::from_iter([Some(Literal::long(5))]);

        let err = spec
            .try_partition_to_path(&data, schema)
            .expect_err("a tuple shorter than the spec must be a typed error");
        assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
        assert!(
            err.message().contains("has 1 value(s)"),
            "unexpected message: {}",
            err.message()
        );
    }

    // ============================================================================================
    // V2 — source column absent from the schema (the spec-evolved commit-path shape).
    // ============================================================================================

    /// Rendering a spec against a schema that dropped one of its source columns renders THAT field
    /// as `null` and still renders the others. Java's `partitionType()` substitutes `UnknownType`.
    #[test]
    fn test_partition_to_path_missing_source_column_renders_null_per_field() {
        let (_schema, spec) = two_field_spec();
        // The evolved schema dropped `x` (source id 1) and kept `y` (source id 2).
        let evolved: SchemaRef = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .expect("evolved schema must build"),
        );
        let data = Struct::from_iter([Some(Literal::long(5)), Some(Literal::long(7))]);

        assert_eq!(
            spec.partition_to_path(&data, evolved.clone()),
            "x=null/y=7",
            "the field whose source survived must still render its value"
        );
        let err = spec
            .try_partition_to_path(&data, evolved)
            .expect_err("a dropped source column must be a typed error on the fallible path");
        assert_eq!(err.kind(), crate::ErrorKind::Unexpected);
    }

    // ============================================================================================
    // V3 — non-primitive partition-field type (a legal `void` over a non-primitive source).
    // ============================================================================================

    /// `void` over a STRUCT source is a legal partition field, so the partition type can be
    /// non-primitive. A primitive value in that slot renders `null` instead of aborting.
    #[test]
    fn test_partition_to_path_non_primitive_field_type_renders_null() {
        let schema: SchemaRef = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(
                        1,
                        "s",
                        Type::Struct(StructType::new(vec![
                            NestedField::required(2, "inner", Type::Primitive(PrimitiveType::Long))
                                .into(),
                        ])),
                    )
                    .into(),
                ])
                .build()
                .expect("struct-column schema must build"),
        );
        let spec = PartitionSpec::builder(schema.clone())
            .add_partition_field("s", "s_void", Transform::Void)
            .expect("void over a non-primitive source is legal")
            .build()
            .expect("the void spec must build");
        let data = Struct::from_iter([Some(Literal::long(5))]);

        assert_eq!(spec.partition_to_path(&data, schema.clone()), "s_void=null");
        let err = spec
            .try_partition_to_path(&data, schema)
            .expect_err("a primitive value under a non-primitive field type must be a typed error");
        assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
    }

    // ============================================================================================
    // V4 — value literal kind incompatible with the partition-field type.
    // ============================================================================================

    /// An `Int` literal in a `Long`-typed partition slot renders `null`. `PrimitiveType::compatible`
    /// decides, the same predicate the commit-path `validate_partition_value` uses.
    #[test]
    fn test_partition_to_path_incompatible_literal_renders_null() {
        let (schema, spec) = two_field_spec();
        let data = Struct::from_iter([Some(Literal::long(5)), Some(Literal::int(7))]);

        assert_eq!(spec.partition_to_path(&data, schema.clone()), "x=5/y=null");
        let err = spec
            .try_partition_to_path(&data, schema)
            .expect_err("an incompatible literal kind must be a typed error");
        assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
        assert!(
            err.message().contains("not compatible"),
            "unexpected message: {}",
            err.message()
        );
    }

    /// A NON-primitive literal in a primitive slot already rendered `null` (never aborted); the
    /// fallible path surfaces it, matching `SnapshotProducer::validate_partition_value`'s posture.
    #[test]
    fn test_partition_to_path_non_primitive_literal_renders_null() {
        let (schema, spec) = two_field_spec();
        let nested = Struct::from_iter([Some(Literal::long(1))]);
        let data = Struct::from_iter([Some(Literal::long(5)), Some(Literal::Struct(nested))]);

        assert_eq!(spec.partition_to_path(&data, schema.clone()), "x=5/y=null");
        let err = spec
            .try_partition_to_path(&data, schema)
            .expect_err("a non-primitive partition literal must be a typed error");
        assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
        assert!(
            err.message().contains("primitive literal"),
            "unexpected message: {}",
            err.message()
        );
    }

    // ============================================================================================
    // The void trap: an all-`void` spec is `is_unpartitioned() == true`, so `(void_spec,
    // Struct::empty())` is a LEGITIMATE pair that a naive arity rule would reject.
    // ============================================================================================

    /// TRAP: an all-`void` spec reports `is_unpartitioned() == true`, and callers legitimately hand
    /// it an EMPTY tuple. A missing value for a `void` field carries no information, so neither path
    /// treats it as an anomaly.
    ///
    /// MUTATION (drop the `void` carve-out): this test and the mixed-void test go RED while
    /// `test_try_partition_to_path_short_tuple_errors` stays GREEN.
    #[test]
    fn test_all_void_spec_with_empty_tuple_is_not_an_anomaly() {
        let schema: SchemaRef = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "x", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .expect("one-column schema must build"),
        );
        let spec = PartitionSpec::builder(schema.clone())
            .add_partition_field("x", "x_void", Transform::Void)
            .expect("void(x) is a legal partition field")
            .build()
            .expect("the all-void spec must build");
        assert!(
            spec.is_unpartitioned(),
            "fixture sanity: an all-void spec reports unpartitioned"
        );

        let data = Struct::empty();
        assert_eq!(spec.partition_to_path(&data, schema.clone()), "x_void=null");
        assert_eq!(
            spec.try_partition_to_path(&data, schema)
                .expect("an all-void spec paired with an empty tuple is legitimate"),
            "x_void=null"
        );
    }

    /// The MIXED shape (`identity(x)` + `void(y)`) with a tuple covering only `x`: the identity
    /// field renders its value, the past-the-end `void` field renders `null`, and neither path errors.
    #[test]
    fn test_partition_to_path_mixed_void_short_tuple_is_not_an_anomaly() {
        let schema: SchemaRef = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "x", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .expect("two-column schema must build"),
        );
        let spec = PartitionSpec::builder(schema.clone())
            .add_partition_field("x", "x", Transform::Identity)
            .expect("identity(x) is a legal partition field")
            .add_partition_field("y", "y_void", Transform::Void)
            .expect("void(y) is a legal partition field")
            .build()
            .expect("the mixed spec must build");
        assert!(
            !spec.is_unpartitioned(),
            "fixture sanity: a spec with a non-void field is partitioned"
        );

        let data = Struct::from_iter([Some(Literal::long(5))]);
        assert_eq!(
            spec.partition_to_path(&data, schema.clone()),
            "x=5/y_void=null"
        );
        assert_eq!(
            spec.try_partition_to_path(&data, schema)
                .expect("a missing value for a void field is not an anomaly"),
            "x=5/y_void=null"
        );
    }

    // ============================================================================================
    // The two paths agree on well-formed input.
    // ============================================================================================

    /// On a self-consistent triple the fallible path returns EXACTLY the string the total path
    /// renders — the total path's leniency is confined to the anomaly branches.
    #[test]
    fn test_try_partition_to_path_matches_partition_to_path_when_well_formed() {
        let (schema, spec) = two_field_spec();
        let data = Struct::from_iter([Some(Literal::long(5)), Some(Literal::long(7))]);

        let total = spec.partition_to_path(&data, schema.clone());
        let fallible = spec
            .try_partition_to_path(&data, schema)
            .expect("a well-formed triple must not error");
        assert_eq!(total, fallible);
        assert_eq!(total, "x=5/y=7");
    }

    // ============================================================================================
    // Drift alarm: every pair `PrimitiveType::compatible` accepts must RENDER.
    // ============================================================================================

    /// The anomaly guard admits exactly the pairs `PrimitiveType::compatible` accepts, and every
    /// admitted pair must survive `Datum`'s `Display`, whose `(_, _)` arm is `unreachable!()`. This
    /// test runs the whole accepted matrix, so a dropped `Display` arm PANICS here. The converse is
    /// safe: `compatible` is narrower, so a rejected pair renders `null`.
    #[test]
    fn test_every_compatible_type_literal_pair_renders() {
        let types = [
            PrimitiveType::Boolean,
            PrimitiveType::Int,
            PrimitiveType::Long,
            PrimitiveType::Float,
            PrimitiveType::Double,
            PrimitiveType::Decimal {
                precision: 10,
                scale: 2,
            },
            PrimitiveType::Date,
            PrimitiveType::Time,
            PrimitiveType::Timestamp,
            PrimitiveType::Timestamptz,
            PrimitiveType::TimestampNs,
            PrimitiveType::TimestamptzNs,
            PrimitiveType::String,
            PrimitiveType::Uuid,
            PrimitiveType::Fixed(4),
            PrimitiveType::Binary,
        ];
        let literals = [
            PrimitiveLiteral::Boolean(true),
            PrimitiveLiteral::Int(1),
            PrimitiveLiteral::Long(1),
            PrimitiveLiteral::Float(1.0.into()),
            PrimitiveLiteral::Double(1.0.into()),
            PrimitiveLiteral::String("s".to_string()),
            PrimitiveLiteral::Binary(vec![1, 2, 3, 4]),
            PrimitiveLiteral::Int128(1),
            PrimitiveLiteral::UInt128(1),
            PrimitiveLiteral::AboveMax,
            PrimitiveLiteral::BelowMin,
        ];

        let mut rendered = 0usize;
        for ty in &types {
            for literal in &literals {
                if ty.compatible(literal) {
                    // Panics loudly (and names the pair) if `Display for Datum` cannot render it.
                    let _ = Datum::new(ty.clone(), literal.clone()).to_human_string();
                    rendered += 1;
                }
            }
        }
        assert_eq!(
            rendered, 16,
            "the accepted matrix changed shape — re-check the guard against `Display for Datum`"
        );
    }
}

#[cfg(test)]
mod partition_path_escaping_tests {
    //! R161 pins: BOTH sides of every `name=value` pair are escaped, exactly as Java does.
    //!
    //! Java `partitionToPath` appends `escape(name)`, `"="`, `escape(humanString)` per field and
    //! joins the pairs with a raw `"/"`; those two separators are STRUCTURE and stay raw. `escape`
    //! is `java.net.URLEncoder.encode(s, "UTF-8")`. Every expectation below is a verbatim jar-oracle
    //! result against `iceberg-api-1.10.0` (`dev/java-interop/run-interop-partition-path.sh`).

    use std::sync::Arc;

    use super::*;
    use crate::spec::{Literal, PrimitiveType, Type};

    /// A one-column `s: string` schema — the binding target for every one-field spec below.
    fn string_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::optional(1, "s", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .expect("one-column schema must build"),
        )
    }

    /// `identity(s)` exposed under `field_name`.
    fn string_spec(field_name: &str) -> PartitionSpec {
        PartitionSpec::builder(string_schema())
            .add_partition_field("s", field_name, Transform::Identity)
            .expect("identity(s) under an arbitrary partition-field name is legal")
            .build()
            .expect("the one-field spec must build")
    }

    /// Render `field_name=value` through EVERY public entry point and assert they agree — the
    /// total path, the fallible path, and `PartitionKey::to_path`.
    fn render(field_name: &str, value: Option<&str>) -> String {
        let schema = string_schema();
        let spec = string_spec(field_name);
        let data = Struct::from_iter([value.map(Literal::string)]);

        let total = spec.partition_to_path(&data, schema.clone());
        let fallible = spec
            .try_partition_to_path(&data, schema.clone())
            .expect("a well-formed (spec, schema, tuple) triple must not error");
        let via_key = PartitionKey::new(spec, schema, data)
            .expect("PartitionKey::new: valid partition tuple")
            .to_path();

        assert_eq!(
            total, fallible,
            "the total and fallible paths must render identically"
        );
        assert_eq!(
            total, via_key,
            "`PartitionKey::to_path` must render identically"
        );
        total
    }

    // ============================================================================================
    // The escaper itself — a full printable-ASCII sweep against Java's `URLEncoder`.
    // ============================================================================================

    /// The printable-ASCII characters `URLEncoder.encode(s, "UTF-8")` leaves untouched, verbatim
    /// from the jar sweep over `0x20..=0x7E` (note: a space is NOT here — it maps to `+`).
    const JAVA_SAFE_ASCII: &str =
        "*-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";

    /// Every printable-ASCII partition value renders exactly as Java's `URLEncoder` renders it: the
    /// 66 safe characters pass through, a space becomes `+`, and the remaining 28 become `%XX`.
    #[test]
    fn printable_ascii_sweep_matches_java_url_encoder() {
        let mut passed_through = 0usize;
        let mut percent_encoded = 0usize;
        for byte in 0x20u8..=0x7Eu8 {
            let ch = char::from(byte);
            let expected = if JAVA_SAFE_ASCII.contains(ch) {
                passed_through += 1;
                ch.to_string()
            } else if ch == ' ' {
                "+".to_string()
            } else {
                percent_encoded += 1;
                format!("%{byte:02X}")
            };
            assert_eq!(
                render("s", Some(&ch.to_string())),
                format!("s={expected}"),
                "ASCII 0x{byte:02X} ({ch:?}) must render as Java's URLEncoder renders it"
            );
        }
        assert_eq!(
            passed_through, 66,
            "the URLEncoder safe set is `A-Z a-z 0-9 - _ . *` — 66 printable-ASCII characters"
        );
        assert_eq!(
            percent_encoded, 28,
            "95 printable ASCII = 66 safe + 1 space + 28 percent-encoded"
        );
    }

    // ============================================================================================
    // The VALUE side — jar-oracle table.
    // ============================================================================================

    /// `identity(s: string)` named `s`: (partition value, Java `partitionToPath`).
    const JAVA_IDENTITY_STRING_PATHS: &[(&str, &str)] = &[
        ("plain", "s=plain"),
        ("AZaz09", "s=AZaz09"),
        ("-_.*", "s=-_.*"),
        ("a/b", "s=a%2Fb"),
        ("a b", "s=a+b"),
        ("a+b", "s=a%2Bb"),
        ("a%b", "s=a%25b"),
        ("a=b", "s=a%3Db"),
        ("a&b", "s=a%26b"),
        ("a?b", "s=a%3Fb"),
        ("a#b", "s=a%23b"),
        ("a:b", "s=a%3Ab"),
        ("a~b", "s=a%7Eb"),
        ("a!b", "s=a%21b"),
        ("a'b", "s=a%27b"),
        ("a(b)c", "s=a%28b%29c"),
        ("a,b", "s=a%2Cb"),
        ("a;b", "s=a%3Bb"),
        ("a@b", "s=a%40b"),
        ("a$b", "s=a%24b"),
        ("", "s="),
        ("  ", "s=++"),
        ("\u{e9}", "s=%C3%A9"),
        ("\u{4e2d}\u{6587}", "s=%E4%B8%AD%E6%96%87"),
        ("\u{1f600}", "s=%F0%9F%98%80"),
        ("x\u{e9} / y", "s=x%C3%A9+%2F+y"),
        ("%2F", "s=%252F"),
        ("a\nb", "s=a%0Ab"),
        ("..", "s=.."),
        (".", "s=."),
        ("null", "s=null"),
    ];

    /// Every value in the jar-oracle table renders byte-identically to Java, including multi-byte
    /// UTF-8: one `%XX` group per UTF-8 byte, never per `char`.
    #[test]
    fn identity_string_values_match_java() {
        for (value, expected) in JAVA_IDENTITY_STRING_PATHS {
            assert_eq!(
                &render("s", Some(value)),
                expected,
                "partition value {value:?} must render exactly as Java does"
            );
        }
        assert_eq!(
            JAVA_IDENTITY_STRING_PATHS.len(),
            31,
            "the jar-oracle value table lost rows"
        );
    }

    // ============================================================================================
    // The NAME side — Java escapes it too.
    // ============================================================================================

    /// `identity(s)` under a tricky partition-field NAME, value `"v"`: (field name, Java path).
    const JAVA_FIELD_NAME_PATHS: &[(&str, &str)] = &[
        ("weird name", "weird+name=v"),
        ("a/b", "a%2Fb=v"),
        ("a=b", "a%3Db=v"),
        ("a%b", "a%25b=v"),
        ("s_bucket", "s_bucket=v"),
        ("x\u{e9}", "x%C3%A9=v"),
        ("a+b", "a%2Bb=v"),
        ("*star*", "*star*=v"),
    ];

    /// The partition-field NAME goes through the same escaper as the value (Java escapes both
    /// sides; escaping only the value would still let a `/` in a field name forge a directory).
    #[test]
    fn field_names_match_java() {
        for (field_name, expected) in JAVA_FIELD_NAME_PATHS {
            assert_eq!(
                &render(field_name, Some("v")),
                expected,
                "partition-field name {field_name:?} must render exactly as Java does"
            );
        }
        assert_eq!(
            JAVA_FIELD_NAME_PATHS.len(),
            8,
            "the jar-oracle field-name table lost rows"
        );
    }

    /// A NULL partition value stays the literal `null`, and the NAME is still escaped on that
    /// branch. The `name=null` fallbacks are a separate code path and need their own pin.
    #[test]
    fn null_values_keep_rendering_null_with_an_escaped_name() {
        const JAVA_FIELD_NAME_NULL_PATHS: &[(&str, &str)] = &[
            ("a/b", "a%2Fb=null"),
            ("weird name", "weird+name=null"),
            ("a%b", "a%25b=null"),
        ];
        for (field_name, expected) in JAVA_FIELD_NAME_NULL_PATHS {
            assert_eq!(
                &render(field_name, None),
                expected,
                "a NULL value under field name {field_name:?} must render exactly as Java does"
            );
        }
        // A string value that literally reads "null" is indistinguishable from a NULL value — the
        // same ambiguity Java has, pinned so nobody "fixes" it into a divergence.
        assert_eq!(render("s", Some("null")), render("s", None));
    }

    /// `name=null` is emitted from THREE sites, and each needs its own pin: a mutation of one is
    /// invisible to the others. Java `Transform.toHumanString` returns the literal `"null"` before
    /// it switches on the type.
    ///
    /// Site 1 is the lenient fallback in `partition_to_path`. The commit path pairs a file's older
    /// spec with the current schema, so an unescaped name puts a raw `/` into a `partitions.` key.
    #[test]
    fn the_lenient_fallback_null_still_escapes_the_field_name() {
        let spec = string_spec("a/b");
        // A schema without source id 1: the field's partition type is not derivable, so the total
        // path falls back to `null` for it.
        let evolved: SchemaRef = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::optional(2, "other", Type::Primitive(PrimitiveType::String))
                        .into(),
                ])
                .build()
                .expect("the evolved schema must build"),
        );
        // A value that WOULD have rendered, to prove the fallback is what emits the pair.
        let data = Struct::from_iter([Some(Literal::string("x/y"))]);

        let path = spec.partition_to_path(&data, evolved.clone());
        assert_eq!(
            path, "a%2Fb=null",
            "the lenient fallback must escape the field name exactly as Java does"
        );
        assert_eq!(
            path.matches('/').count(),
            0,
            "a `/` in the field name must not forge a directory level on the fallback branch"
        );
        // Fixture sanity: this branch is reached only because the triple is inconsistent.
        let err = spec
            .try_partition_to_path(&data, evolved)
            .expect_err("a dropped source column must be a typed error on the fallible path");
        assert_eq!(err.kind(), crate::ErrorKind::Unexpected);
    }

    /// Site 2 of three: the `void`-past-the-end-of-tuple branch in `render_partition_field`. An
    /// all-`void` spec reports `is_unpartitioned()`, so an empty tuple reaches `name=null` here.
    #[test]
    fn the_void_past_end_null_still_escapes_the_field_name() {
        let schema: SchemaRef = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::optional(1, "s", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .expect("one-column schema must build"),
        );
        let spec = PartitionSpec::builder(schema.clone())
            .add_partition_field("s", "c/d", Transform::Void)
            .expect("void(s) under an arbitrary partition-field name is legal")
            .build()
            .expect("the one-field void spec must build");
        assert!(
            spec.is_unpartitioned(),
            "fixture sanity: an all-void spec reports unpartitioned, which is why callers pair it \
             with an empty tuple"
        );
        let data = Struct::empty();

        let path = spec.partition_to_path(&data, schema.clone());
        assert_eq!(
            path, "c%2Fd=null",
            "the void-past-end branch must escape the field name exactly as Java does"
        );
        assert_eq!(
            path.matches('/').count(),
            0,
            "a `/` in the field name must not forge a directory level on the void branch"
        );
        assert_eq!(
            spec.try_partition_to_path(&data, schema)
                .expect("an all-void spec paired with an empty tuple is legitimate"),
            path,
            "the fallible path must render the same escaped pair"
        );
    }

    // ============================================================================================
    // Structure vs. content.
    // ============================================================================================

    /// The `/` between pairs and the `=` inside a pair are STRUCTURE — they stay raw — while a `/`
    /// or `=` inside a name or a value is CONTENT and is escaped.
    #[test]
    fn pair_and_field_separators_stay_raw() {
        let schema: SchemaRef = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::optional(1, "s", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(2, "i", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .expect("two-column schema must build"),
        );
        let spec = PartitionSpec::builder(schema.clone())
            .add_partition_field("s", "a b", Transform::Identity)
            .expect("identity(s) as `a b` is legal")
            .add_partition_field("i", "c/d", Transform::Identity)
            .expect("identity(i) as `c/d` is legal")
            .build()
            .expect("the two-field spec must build");
        let data = Struct::from_iter([Some(Literal::string("x/y")), Some(Literal::int(5))]);

        // Jar oracle: `a+b=x%2Fy/c%2Fd=5`.
        let path = spec.partition_to_path(&data, schema);
        assert_eq!(path, "a+b=x%2Fy/c%2Fd=5");
        assert_eq!(
            path.matches('/').count(),
            1,
            "exactly ONE raw `/` — the separator between the two pairs"
        );
        assert_eq!(
            path.matches('=').count(),
            2,
            "exactly TWO raw `=` — one per pair"
        );
    }

    /// The headline safety property: a `/` inside a partition VALUE can no longer forge an extra
    /// directory level in a data file's location (nor a bogus `partitions.` summary key).
    #[test]
    fn a_slash_in_a_value_cannot_forge_a_directory_level() {
        let path = render("s", Some("a/b/c"));
        assert_eq!(path, "s=a%2Fb%2Fc");
        assert_eq!(
            path.matches('/').count(),
            0,
            "a single-field path must contain no raw `/` whatever the value holds"
        );
    }

    /// A space and a `+` must not collide: Java maps space to `+` and `+` to `%2B`, so the two
    /// values keep distinct paths (a naive "escape `/` only" fix would collapse them).
    #[test]
    fn space_and_plus_stay_distinct() {
        assert_eq!(render("s", Some("a b")), "s=a+b");
        assert_eq!(render("s", Some("a+b")), "s=a%2Bb");
        assert_ne!(render("s", Some("a b")), render("s", Some("a+b")));
    }

    // ============================================================================================
    // The no-churn invariant — the overwhelmingly common case must be BYTE-IDENTICAL to pre-R161.
    // ============================================================================================

    /// Every partition value inside the URLEncoder safe set renders EXACTLY as it did before R161:
    /// no `%XX`, no `+`. This keeps ordinary table layouts unchanged, and it fails loudly under an
    /// over-eager escaper such as RFC-3986 `NON_ALPHANUMERIC`, which mangles `-`, `_`, `.` and `*`.
    #[test]
    fn safe_partition_values_are_byte_identical_to_the_unescaped_rendering() {
        const COMMON: &[(&str, &str)] = &[
            ("dt", "2024-01-31"),
            ("category", "electronics"),
            ("id", "42"),
            ("region", "us-east-1"),
            ("s_bucket", "7"),
            ("amount", "-12.34"),
            ("uu", "f79c3e09-677c-4bbd-a479-3f349cb785e7"),
            ("star.name_1", "*star.name_1*"),
            ("empty_void", "null"),
        ];
        for (field_name, value) in COMMON {
            assert_eq!(
                &render(field_name, Some(value)),
                &format!("{field_name}={value}"),
                "a safe-set partition value must render byte-identically to pre-R161"
            );
        }
    }

    /// The pre-R161 fixture from `tests::test_partition_to_path` is byte-stable: a realistic
    /// four-field path over plain values is untouched by the escaper.
    #[test]
    fn the_pre_r161_multi_field_fixture_is_byte_stable() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(3, "timestamp", Type::Primitive(PrimitiveType::Timestamp))
                    .into(),
                NestedField::required(4, "empty", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("the four-column schema must build");
        let spec = PartitionSpec::builder(schema.clone())
            .add_partition_field("id", "id", Transform::Identity)
            .expect("identity(id) is legal")
            .add_partition_field("name", "name", Transform::Identity)
            .expect("identity(name) is legal")
            .add_partition_field("timestamp", "ts_hour", Transform::Hour)
            .expect("hour(timestamp) is legal")
            .add_partition_field("empty", "empty_void", Transform::Void)
            .expect("void(empty) is legal")
            .build()
            .expect("the four-field spec must build");
        let data = Struct::from_iter([
            Some(Literal::int(42)),
            Some(Literal::string("alice")),
            Some(Literal::int(1000)),
            Some(Literal::string("empty")),
        ]);

        assert_eq!(
            spec.partition_to_path(&data, schema.into()),
            "id=42/name=alice/ts_hour=1000/empty_void=null"
        );
    }

    /// Which value CLASSES move under R161. FIVE fork-supported column types render a human string
    /// containing `:`, four of them a space too, so their path changes for EVERY value, the V3
    /// nanosecond pair included. `date` is the byte-stable control in the same tuple.
    ///
    /// The four `assert_ne!`s ALARM on the named human-string residue: when one becomes equal, that
    /// residue is closed and row R161 must change in the same commit.
    #[test]
    fn the_five_always_moving_temporal_types_move_for_every_value() {
        let schema: SchemaRef = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::optional(1, "ts", Type::Primitive(PrimitiveType::Timestamp))
                        .into(),
                    NestedField::optional(2, "tz", Type::Primitive(PrimitiveType::Timestamptz))
                        .into(),
                    NestedField::optional(3, "tm", Type::Primitive(PrimitiveType::Time)).into(),
                    NestedField::optional(4, "tsn", Type::Primitive(PrimitiveType::TimestampNs))
                        .into(),
                    NestedField::optional(5, "tzn", Type::Primitive(PrimitiveType::TimestamptzNs))
                        .into(),
                    NestedField::optional(6, "dt", Type::Primitive(PrimitiveType::Date)).into(),
                ])
                .build()
                .expect("the six-column temporal schema must build"),
        );
        let spec = PartitionSpec::builder(schema.clone())
            .add_partition_field("ts", "ts", Transform::Identity)
            .expect("identity(ts) is legal")
            .add_partition_field("tz", "tz", Transform::Identity)
            .expect("identity(tz) is legal")
            .add_partition_field("tm", "tm", Transform::Identity)
            .expect("identity(tm) is legal")
            .add_partition_field("tsn", "tsn", Transform::Identity)
            .expect("identity(tsn) is legal")
            .add_partition_field("tzn", "tzn", Transform::Identity)
            .expect("identity(tzn) is legal")
            .add_partition_field("dt", "dt", Transform::Identity)
            .expect("identity(dt) is legal")
            .build()
            .expect("the six-field temporal spec must build");
        // 2017-11-16T22:31:08 in micros (and the same instant in nanos); 22:31:08 in micros;
        // 2022-01-08 in days.
        let data = Struct::from_iter([
            Some(Literal::timestamp(1_510_871_468_000_000)),
            Some(Literal::timestamptz(1_510_871_468_000_000)),
            Some(Literal::time(81_068_000_000)),
            Some(Literal::timestamp_nano(1_510_871_468_000_000_000)),
            Some(Literal::timestamptz_nano(1_510_871_468_000_000_000)),
            Some(Literal::date(19_000)),
        ]);

        let path = spec.partition_to_path(&data, schema);
        let pairs: Vec<&str> = path.split('/').collect();
        assert_eq!(
            pairs,
            vec![
                "ts=2017-11-16+22%3A31%3A08",
                "tz=2017-11-16+22%3A31%3A08+UTC",
                "tm=22%3A31%3A08",
                "tsn=2017-11-16+22%3A31%3A08",
                "tzn=2017-11-16+22%3A31%3A08+UTC",
                "dt=2022-01-08",
            ],
            "the five temporal types whose human string holds a `:` move under the escaper; \
             `date` does not"
        );

        // `time` MATCHES Java post-R161: this change closes that divergence.
        assert_eq!(
            pairs[2], "tm=22%3A31%3A08",
            "Java: `22:31:08` escapes to `22%3A31%3A08`"
        );
        // `date` is byte-stable: its human string holds no character outside the safe set.
        assert_eq!(
            pairs[5], "dt=2022-01-08",
            "Java: `2022-01-08`, untouched by the escaper"
        );
        // The four remaining divergences, pinned as an alarm. Each expected form is Java's own,
        // measured on the JVM, so none is a dead comparison.
        assert_ne!(
            pairs[0], "ts=2017-11-16T22%3A31%3A08",
            "residue R161: Java renders ISO `T`, the fork renders a space (escaped `+`)"
        );
        assert_ne!(
            pairs[1], "tz=2017-11-16T22%3A31%3A08%2B00%3A00",
            "residue R161: Java renders ISO `T` and `+00:00`, the fork renders a space and ` UTC`"
        );
        assert_ne!(
            pairs[3], "tsn=2017-11-16T22%3A31%3A08",
            "residue R161: the nanosecond pair diverges exactly like the microsecond pair"
        );
        assert_ne!(
            pairs[4], "tzn=2017-11-16T22%3A31%3A08%2B00%3A00",
            "residue R161: the nanosecond pair diverges exactly like the microsecond pair"
        );
    }

    /// Render a one-field `transform(column)` spec over `column: ty` holding `value`.
    fn render_one(
        column: &str,
        field_name: &str,
        ty: PrimitiveType,
        transform: Transform,
        value: Literal,
    ) -> String {
        let schema: SchemaRef = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::optional(1, column, Type::Primitive(ty)).into(),
                ])
                .build()
                .expect("the one-column schema must build"),
        );
        let spec = PartitionSpec::builder(schema.clone())
            .add_partition_field(column, field_name, transform)
            .expect("the transform must be legal for this column type")
            .build()
            .expect("the one-field spec must build");
        spec.partition_to_path(&Struct::from_iter([Some(value)]), schema)
    }

    /// Byte-stability is a property of the OUTPUT type, not of the transform name.
    /// `Transform::result_type` returns the input type for `Truncate`, so `truncate(string, N)`
    /// renders a STRING and is as escaper-sensitive as `identity(string)`. Over int, long, decimal
    /// and binary it stays in the safe set: binary and fixed render as Java's standard Base64
    /// (`TransformUtil.base64encode`, no `Truncate` override), so a source byte such as `0x2F` never
    /// reaches the path. Base64's `+`, `/` and `=` are then escaped like any other character.
    #[test]
    fn truncate_is_byte_stable_except_over_string() {
        let moving = [
            (
                render_one(
                    "s",
                    "s_trunc",
                    PrimitiveType::String,
                    Transform::Truncate(4),
                    Literal::string("a/b c"),
                ),
                "s_trunc=a%2Fb+c",
            ),
            (
                render_one(
                    "s",
                    "t5",
                    PrimitiveType::String,
                    Transform::Truncate(5),
                    Literal::string("east 1x"),
                ),
                "t5=east+1x",
            ),
        ];
        for (rendered, expected) in &moving {
            assert_eq!(
                rendered, expected,
                "truncate over `string` renders a string and MUST be escaped"
            );
        }

        // CLOSED 2026-07-31 (QC / R161): Java base64 for binary partition values.
        let truncated_binary = render_one(
            "bn",
            "tb",
            PrimitiveType::Binary,
            Transform::Truncate(2),
            Literal::binary(vec![0x61, 0x2F, 0x62]),
        );
        assert_eq!(
            truncated_binary, "tb=YS9i",
            "Java `partitionToPath` emits `tb=YS9i` for truncate(binary,2) over bytes 61 2F 62 \
             (TransformUtil.base64encode / java.util.Base64.getEncoder)"
        );

        let stable = [
            (
                render_one(
                    "s",
                    "tsafe",
                    PrimitiveType::String,
                    Transform::Truncate(16),
                    Literal::string("us-east-1"),
                ),
                "tsafe=us-east-1",
            ),
            (
                render_one(
                    "i",
                    "ti",
                    PrimitiveType::Int,
                    Transform::Truncate(10),
                    Literal::int(25),
                ),
                "ti=25",
            ),
            (
                render_one(
                    "l",
                    "tl",
                    PrimitiveType::Long,
                    Transform::Truncate(10),
                    Literal::long(-25),
                ),
                "tl=-25",
            ),
            (
                render_one(
                    "d",
                    "td",
                    PrimitiveType::Decimal {
                        precision: 9,
                        scale: 2,
                    },
                    Transform::Truncate(50),
                    Literal::decimal(12345),
                ),
                "td=123.45",
            ),
            // Base64 `YS9i` is inside the URLEncoder safe set — no further escaping.
            (truncated_binary.clone(), "tb=YS9i"),
        ];
        for (rendered, expected) in &stable {
            assert_eq!(
                rendered, expected,
                "this truncate output holds no character outside the safe set after the human \
                 string is formed"
            );
        }
    }

    /// `identity(binary)` and `identity(fixed[N])` use the same Base64 human string as Java, not
    /// UPPERCASE hex. `Display for Datum` still renders hex; only the partition-path seam is base64.
    #[test]
    fn identity_binary_and_fixed_render_java_base64() {
        let bytes = vec![0x61, 0x2F, 0x62]; // ASCII "a/b" → base64 "YS9i"
        assert_eq!(
            render_one(
                "bn",
                "b",
                PrimitiveType::Binary,
                Transform::Identity,
                Literal::binary(bytes.clone()),
            ),
            "b=YS9i",
            "identity(binary) must match Java TransformUtil.base64encode"
        );
        assert_eq!(
            render_one(
                "fx",
                "f",
                PrimitiveType::Fixed(3),
                Transform::Identity,
                Literal::fixed(bytes.clone()),
            ),
            "f=YS9i",
            "identity(fixed[3]) must match Java TransformUtil.base64encode"
        );

        // Standard Base64, NOT URL-safe: bytes that produce `+`, `/` or `=` are then URL-escaped.
        // 0xFB 0xFF -> base64 "+/8=" -> escaped "%2B%2F8%3D".
        assert_eq!(
            render_one(
                "bn",
                "b",
                PrimitiveType::Binary,
                Transform::Identity,
                Literal::binary(vec![0xFB, 0xFF]),
            ),
            "b=%2B%2F8%3D",
            "base64 alphabet chars outside the URLEncoder safe set must be escaped"
        );

        // Display stays hex (orthogonal surface).
        let datum = crate::spec::Datum::new(
            PrimitiveType::Binary,
            crate::spec::PrimitiveLiteral::Binary(vec![0x61, 0x2F, 0x62]),
        );
        assert_eq!(
            datum.to_string(),
            "612F62",
            "Display for Binary stays UPPERCASE hex; only to_human_string is base64"
        );
        assert_eq!(datum.to_human_string(), "YS9i");

        // Empty binary → empty human string (Java: `toHumanString(Binary, empty ByteBuffer)` → "").
        assert_eq!(
            render_one(
                "bn",
                "b",
                PrimitiveType::Binary,
                Transform::Identity,
                Literal::binary(Vec::<u8>::new()),
            ),
            "b=",
            "empty binary human string is empty (Java jar-oracle)"
        );

        // UUID is NOT base64 — Java uses UUID.toString(); fork uses Display for UInt128.
        assert_eq!(
            render_one(
                "u",
                "u",
                PrimitiveType::Uuid,
                Transform::Identity,
                Literal::uuid(uuid::Uuid::from_u128(1)),
            ),
            "u=00000000-0000-0000-0000-000000000001",
            "identity(uuid) must not be routed through base64"
        );
    }

    /// R161 restores INJECTIVITY of partition tuple to directory, the data-trust half of the defect.
    ///
    /// Before R161 the pair was `format!("{name}={value}")` with both sides raw, so a `/` or an `=`
    /// inside a VALUE made two DISTINCT tuples render the SAME path. Colliding paths put two
    /// partitions' data files in one directory and merge their `partitions.<path>` summary entries,
    /// so the per-partition record counts are silently summed.
    #[test]
    fn two_distinct_tuples_can_no_longer_collide_on_one_directory() {
        let schema: SchemaRef = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::optional(1, "a", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(2, "b", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .expect("the two-column schema must build"),
        );
        let two_field = PartitionSpec::builder(schema.clone())
            .add_partition_field("a", "a", Transform::Identity)
            .expect("identity(a) is legal")
            .add_partition_field("b", "b", Transform::Identity)
            .expect("identity(b) is legal")
            .build()
            .expect("the two-field spec must build");

        // Same spec, two distinct tuples — both rendered `a=1/b=2/b=3` before R161.
        let x = two_field.partition_to_path(
            &Struct::from_iter([Some(Literal::string("1/b=2")), Some(Literal::string("3"))]),
            schema.clone(),
        );
        let y = two_field.partition_to_path(
            &Struct::from_iter([Some(Literal::string("1")), Some(Literal::string("2/b=3"))]),
            schema.clone(),
        );
        // Injectivity FIRST, so a regression's failure message displays the collision itself
        // rather than a byte mismatch on one side of it.
        assert_ne!(
            x, y,
            "two distinct partition tuples of ONE spec must never share a directory"
        );
        assert_eq!(x, "a=1%2Fb%3D2/b=3");
        assert_eq!(y, "a=1/b=2%2Fb%3D3");

        // Cross-arity: a 1-field spec and a 2-field spec of the same evolving table — both
        // rendered `a=1/b=2` before R161.
        let one_field = PartitionSpec::builder(schema.clone())
            .add_partition_field("a", "a", Transform::Identity)
            .expect("identity(a) is legal")
            .build()
            .expect("the one-field spec must build");
        let narrow = one_field.partition_to_path(
            &Struct::from_iter([Some(Literal::string("1/b=2"))]),
            schema.clone(),
        );
        let wide = two_field.partition_to_path(
            &Struct::from_iter([Some(Literal::string("1")), Some(Literal::string("2"))]),
            schema,
        );
        assert_ne!(
            narrow, wide,
            "tuples under two specs of ONE table must never share a directory"
        );
        assert_eq!(narrow, "a=1%2Fb%3D2");
        assert_eq!(wide, "a=1/b=2");
    }
}
