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

//! The shared `data_file` projection used by the `files` family AND the `entries` metadata table.
//!
//! Both expose the same column set of an Iceberg [`crate::spec::DataFile`] — content, file path/format,
//! partition, record/size counts, the metrics maps, the list columns, and the V3 deletion-vector fields —
//! mirroring Java `DataFile.getType(partitionType).fields()` (field ids from `api/DataFile.java`). They
//! differ ONLY in shape:
//!
//! - [`crate::inspect::FilesTable`] FLATTENS the projection to top-level columns (Java `BaseFilesTable`,
//!   whose schema IS `DataFile.getType(...)`).
//! - [`crate::inspect::EntriesTable`] NESTS the projection under a single `data_file` STRUCT column (Java
//!   `BaseEntriesTable` / `ManifestEntry.wrapFileSchema`).
//!
//! This module is the single source of truth for the field list ([`data_file_fields`]) and the row
//! builder ([`DataFileStructBuilder`]) so the two tables cannot drift — the Rule of Three (2nd use).
//!
//! The raw metrics maps (`column_sizes`/`value_counts`/…/`lower_bounds`/`upper_bounds`) are part of this
//! projection. The separate `readable_metrics` virtual STRUCT column (Java
//! `MetricsUtil.readableMetricsStruct` — the per-column typed/human-readable view of those metrics) is
//! built by [`crate::inspect::readable_metrics`] and appended alongside this projection by both tables.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::StructArray;
use arrow_array::builder::{
    ArrayBuilder, BinaryBuilder, BooleanBuilder, Date32Builder, Decimal128Builder, Float32Builder,
    Float64Builder, Int32Builder, Int64Builder, LargeBinaryBuilder, ListBuilder, MapBuilder,
    StringBuilder, StructBuilder, Time64MicrosecondBuilder, TimestampMicrosecondBuilder,
    TimestampNanosecondBuilder,
};
use arrow_schema::Fields;

use crate::spec::{
    Datum, ListType, Literal, MapType, NestedField, NestedFieldRef, PrimitiveLiteral,
    PrimitiveType, Schema, StructType, TableMetadata, Type, select_not,
};
use crate::{Error, ErrorKind, Result};

/// Java `DataFile.PARTITION_ID` (`api/DataFile.java`). `BaseFilesTable.schema` and
/// `BaseEntriesTable.schema` drop this field via `TypeUtil.selectNot` when the table's
/// partition type is empty, so consumers never see an empty struct.
pub(super) const DATA_FILE_PARTITION_ID: i32 = 102;

/// Java `BaseFilesTable` / `BaseEntriesTable`: if `partitionType.fields()` is empty,
/// `TypeUtil.selectNot(schema, DataFile.PARTITION_ID)` so an empty struct is never advertised.
pub(super) fn omit_empty_partition_field(schema: Schema, partition_type: &StructType) -> Schema {
    if partition_type.fields().is_empty() {
        select_not(&schema, &HashSet::from([DATA_FILE_PARTITION_ID])).expect(
            "TypeUtil.selectNot of DataFile.PARTITION_ID from a static metadata schema is valid",
        )
    } else {
        schema
    }
}

/// Every spec's partition-field ids, in that spec's OWN field order, keyed by spec id.
///
/// A [`crate::spec::DataFile`]'s partition `Struct` is a bare `Vec<Option<Literal>>` whose values are
/// positionally aligned with the fields of the spec the file was WRITTEN under
/// (`DataFile::partition_spec_id`) — not with the partition type the metadata table projects. Java
/// recovers the same information inside `PartitionUtil.coercePartition`, which builds a
/// `StructProjection.createAllowMissing(spec.partitionType(), partitionType)` — i.e. a FIELD-ID match
/// between the file's own spec and the projected type. This map is the fork's equivalent lookup, and it
/// is what lets [`append_partition`] match by field id instead of by position.
pub(super) fn partition_field_ids_by_spec(metadata: &TableMetadata) -> HashMap<i32, Vec<i32>> {
    metadata
        .partition_specs_iter()
        .map(|spec| {
            (
                spec.spec_id(),
                spec.fields().iter().map(|field| field.field_id).collect(),
            )
        })
        .collect()
}

/// The boxed `MapBuilder` shape `StructBuilder::from_fields` produces for a `DataType::Map` child (its
/// key/value field metadata is preserved by `make_builder`, so we only supply the values).
type DynMapBuilder = MapBuilder<Box<dyn ArrayBuilder>, Box<dyn ArrayBuilder>>;
/// The boxed `ListBuilder` shape `StructBuilder::from_fields` produces for a `DataType::List` child.
type DynListBuilder = ListBuilder<Box<dyn ArrayBuilder>>;

/// The 21 `data_file` columns, mirroring Java `DataFile.getType(partitionType).fields()` — the
/// canonical `DataFile` field ids from `api/DataFile.java`. The `partition` column carries the table's
/// DEFAULT partition type (always included here; [`omit_empty_partition_field`] drops field 102
/// afterwards, matching Java `TypeUtil.selectNot`). `readable_metrics` is deferred.
///
/// `files` uses these directly as its top-level columns; `entries` nests them under a `data_file` struct.
pub(super) fn data_file_fields(partition_type: &StructType) -> Vec<NestedFieldRef> {
    vec![
        Arc::new(NestedField::optional(
            134,
            "content",
            Type::Primitive(PrimitiveType::Int),
        )),
        Arc::new(NestedField::required(
            100,
            "file_path",
            Type::Primitive(PrimitiveType::String),
        )),
        Arc::new(NestedField::required(
            101,
            "file_format",
            Type::Primitive(PrimitiveType::String),
        )),
        Arc::new(NestedField::optional(
            141,
            "spec_id",
            Type::Primitive(PrimitiveType::Int),
        )),
        Arc::new(NestedField::required(
            102,
            "partition",
            Type::Struct(partition_type.clone()),
        )),
        Arc::new(NestedField::required(
            103,
            "record_count",
            Type::Primitive(PrimitiveType::Long),
        )),
        Arc::new(NestedField::required(
            104,
            "file_size_in_bytes",
            Type::Primitive(PrimitiveType::Long),
        )),
        Arc::new(NestedField::optional(
            108,
            "column_sizes",
            int_long_map(117, 118),
        )),
        Arc::new(NestedField::optional(
            109,
            "value_counts",
            int_long_map(119, 120),
        )),
        Arc::new(NestedField::optional(
            110,
            "null_value_counts",
            int_long_map(121, 122),
        )),
        Arc::new(NestedField::optional(
            137,
            "nan_value_counts",
            int_long_map(138, 139),
        )),
        Arc::new(NestedField::optional(
            125,
            "lower_bounds",
            int_binary_map(126, 127),
        )),
        Arc::new(NestedField::optional(
            128,
            "upper_bounds",
            int_binary_map(129, 130),
        )),
        Arc::new(NestedField::optional(
            131,
            "key_metadata",
            Type::Primitive(PrimitiveType::Binary),
        )),
        Arc::new(NestedField::optional(132, "split_offsets", long_list(133))),
        Arc::new(NestedField::optional(135, "equality_ids", int_list(136))),
        Arc::new(NestedField::optional(
            140,
            "sort_order_id",
            Type::Primitive(PrimitiveType::Int),
        )),
        Arc::new(NestedField::optional(
            142,
            "first_row_id",
            Type::Primitive(PrimitiveType::Long),
        )),
        Arc::new(NestedField::optional(
            143,
            "referenced_data_file",
            Type::Primitive(PrimitiveType::String),
        )),
        Arc::new(NestedField::optional(
            144,
            "content_offset",
            Type::Primitive(PrimitiveType::Long),
        )),
        Arc::new(NestedField::optional(
            145,
            "content_size_in_bytes",
            Type::Primitive(PrimitiveType::Long),
        )),
    ]
}

/// Iceberg `map<int, long>` with the given key/value field ids (the metrics-count maps).
fn int_long_map(key_id: i32, value_id: i32) -> Type {
    Type::Map(MapType {
        key_field: Arc::new(NestedField::map_key_element(
            key_id,
            Type::Primitive(PrimitiveType::Int),
        )),
        value_field: Arc::new(NestedField::map_value_element(
            value_id,
            Type::Primitive(PrimitiveType::Long),
            true,
        )),
    })
}

/// Iceberg `map<int, binary>` with the given key/value field ids (the lower/upper-bound maps).
fn int_binary_map(key_id: i32, value_id: i32) -> Type {
    Type::Map(MapType {
        key_field: Arc::new(NestedField::map_key_element(
            key_id,
            Type::Primitive(PrimitiveType::Int),
        )),
        value_field: Arc::new(NestedField::map_value_element(
            value_id,
            Type::Primitive(PrimitiveType::Binary),
            true,
        )),
    })
}

/// Iceberg `list<long>` (required element) with the given element field id (split offsets).
fn long_list(element_id: i32) -> Type {
    Type::List(ListType {
        element_field: Arc::new(NestedField::list_element(
            element_id,
            Type::Primitive(PrimitiveType::Long),
            true,
        )),
    })
}

/// Iceberg `list<int>` (required element) with the given element field id (equality ids).
fn int_list(element_id: i32) -> Type {
    Type::List(ListType {
        element_field: Arc::new(NestedField::list_element(
            element_id,
            Type::Primitive(PrimitiveType::Int),
            true,
        )),
    })
}

/// Accumulates `data_file` rows into a single Arrow [`StructBuilder`] (one child per `DataFile` column).
///
/// The struct's child fields are the Arrow conversion of [`data_file_fields`] — so the produced
/// [`StructArray`] is exactly the `data_file` STRUCT the `entries` table nests, and its `.columns()` are
/// exactly the top-level columns the `files` table flattens. One builder, both shapes.
pub(super) struct DataFileStructBuilder<'a> {
    partition_type: &'a StructType,
    /// Spec id → that spec's partition-field ids in its own tuple order, from
    /// [`partition_field_ids_by_spec`]. Used to read each file's partition tuple by FIELD ID.
    partition_field_ids_by_spec: &'a HashMap<i32, Vec<i32>>,
    /// Child name → builder index. Name-keyed so dropping `partition` (Java empty-struct
    /// `selectNot`) does not shift the remaining columns.
    indices: HashMap<String, usize>,
    builder: StructBuilder,
}

impl<'a> DataFileStructBuilder<'a> {
    /// Creates a builder over the given Arrow `data_file` struct fields (the converted [`data_file_fields`],
    /// possibly without `partition`) and the table's DEFAULT partition type (used to dispatch the
    /// partition tuple's per-field types when that column is present).
    ///
    /// `partition_field_ids_by_spec` ([`partition_field_ids_by_spec`]) resolves each appended file's OWN
    /// spec so its partition tuple is read by field id rather than by position.
    pub(super) fn new(
        data_file_arrow_fields: &Fields,
        partition_type: &'a StructType,
        partition_field_ids_by_spec: &'a HashMap<i32, Vec<i32>>,
    ) -> Self {
        let indices = data_file_arrow_fields
            .iter()
            .enumerate()
            .map(|(index, field)| (field.name().clone(), index))
            .collect();
        Self {
            partition_type,
            partition_field_ids_by_spec,
            indices,
            builder: StructBuilder::from_fields(data_file_arrow_fields.clone(), 0),
        }
    }

    /// Appends one row built from a [`crate::spec::DataFile`].
    pub(super) fn append(&mut self, data_file: &crate::spec::DataFile) -> Result<()> {
        let i_content = field_index(&self.indices, "content")?;
        let i_file_path = field_index(&self.indices, "file_path")?;
        let i_file_format = field_index(&self.indices, "file_format")?;
        let i_spec_id = field_index(&self.indices, "spec_id")?;
        let i_partition = self.indices.get("partition").copied();
        let i_record_count = field_index(&self.indices, "record_count")?;
        let i_file_size = field_index(&self.indices, "file_size_in_bytes")?;
        let i_column_sizes = field_index(&self.indices, "column_sizes")?;
        let i_value_counts = field_index(&self.indices, "value_counts")?;
        let i_null_value_counts = field_index(&self.indices, "null_value_counts")?;
        let i_nan_value_counts = field_index(&self.indices, "nan_value_counts")?;
        let i_lower_bounds = field_index(&self.indices, "lower_bounds")?;
        let i_upper_bounds = field_index(&self.indices, "upper_bounds")?;
        let i_key_metadata = field_index(&self.indices, "key_metadata")?;
        let i_split_offsets = field_index(&self.indices, "split_offsets")?;
        let i_equality_ids = field_index(&self.indices, "equality_ids")?;
        let i_sort_order_id = field_index(&self.indices, "sort_order_id")?;
        let i_first_row_id = field_index(&self.indices, "first_row_id")?;
        let i_referenced_data_file = field_index(&self.indices, "referenced_data_file")?;
        let i_content_offset = field_index(&self.indices, "content_offset")?;
        let i_content_size = field_index(&self.indices, "content_size_in_bytes")?;

        let b = &mut self.builder;

        struct_child::<Int32Builder>(b, i_content)?.append_value(data_file.content_type() as i32);
        struct_child::<StringBuilder>(b, i_file_path)?.append_value(data_file.file_path());
        // Java's `FilesTable`/`ManifestEntriesTable` render `file_format` as the UPPERCASE `FileFormat`
        // enum NAME (`PARQUET`/`AVRO`/`ORC`) via `format.toString()`. `DataFileFormat`'s `Display` is
        // lowercase (the on-disk manifest string), so upper-case ONLY here in the inspection projection to
        // match Java exactly — the on-disk write path (Display/serde) is unchanged.
        struct_child::<StringBuilder>(b, i_file_format)?
            .append_value(data_file.file_format().to_string().to_uppercase());
        struct_child::<Int32Builder>(b, i_spec_id)?.append_value(data_file.partition_spec_id);

        if let Some(i_partition) = i_partition {
            // The tuple is aligned to the file's OWN spec (Java `table.specs().get(file.specId())`
            // inside `PartitionUtil.coercePartition`), so resolve that spec's field ids before reading.
            let source_field_ids = self
                .partition_field_ids_by_spec
                .get(&data_file.partition_spec_id)
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "data file {} was written under partition spec id {}, which is not in the table metadata",
                            data_file.file_path(),
                            data_file.partition_spec_id
                        ),
                    )
                })?;
            let partition_builder = struct_child::<StructBuilder>(b, i_partition)?;
            append_partition(
                partition_builder,
                self.partition_type,
                source_field_ids,
                data_file.partition(),
            )?;
        }

        struct_child::<Int64Builder>(b, i_record_count)?
            .append_value(data_file.record_count() as i64);
        struct_child::<Int64Builder>(b, i_file_size)?
            .append_value(data_file.file_size_in_bytes() as i64);

        append_count_map(
            struct_child::<DynMapBuilder>(b, i_column_sizes)?,
            data_file.column_sizes(),
        )?;
        append_count_map(
            struct_child::<DynMapBuilder>(b, i_value_counts)?,
            data_file.value_counts(),
        )?;
        append_count_map(
            struct_child::<DynMapBuilder>(b, i_null_value_counts)?,
            data_file.null_value_counts(),
        )?;
        append_count_map(
            struct_child::<DynMapBuilder>(b, i_nan_value_counts)?,
            data_file.nan_value_counts(),
        )?;
        append_bound_map(
            struct_child::<DynMapBuilder>(b, i_lower_bounds)?,
            data_file.lower_bounds(),
        )?;
        append_bound_map(
            struct_child::<DynMapBuilder>(b, i_upper_bounds)?,
            data_file.upper_bounds(),
        )?;

        struct_child::<LargeBinaryBuilder>(b, i_key_metadata)?
            .append_option(data_file.key_metadata());

        append_i64_list(
            struct_child::<DynListBuilder>(b, i_split_offsets)?,
            data_file.split_offsets(),
        )?;
        append_i32_list(
            struct_child::<DynListBuilder>(b, i_equality_ids)?,
            data_file.equality_ids().as_deref(),
        )?;

        struct_child::<Int32Builder>(b, i_sort_order_id)?.append_option(data_file.sort_order_id());
        struct_child::<Int64Builder>(b, i_first_row_id)?.append_option(data_file.first_row_id());
        struct_child::<StringBuilder>(b, i_referenced_data_file)?
            .append_option(data_file.referenced_data_file());
        struct_child::<Int64Builder>(b, i_content_offset)?
            .append_option(data_file.content_offset());
        struct_child::<Int64Builder>(b, i_content_size)?
            .append_option(data_file.content_size_in_bytes());

        // The struct itself is always present (a row is never a null data_file).
        b.append(true);
        Ok(())
    }

    /// Appends a NULL `data_file` struct row (every child gets a null, the struct slot is null). Unused by
    /// `files`/`entries` today (a manifest entry always carries a data_file) but kept for completeness so a
    /// future optional-struct caller cannot misuse the builder.
    #[allow(dead_code)]
    pub(super) fn append_null(&mut self) {
        self.builder.append_null();
    }

    /// Finishes into a single [`StructArray`] — the `data_file` column for `entries`, or (via
    /// [`StructArray::columns`]) the flattened top-level columns for `files`.
    pub(super) fn finish(mut self) -> StructArray {
        self.builder.finish()
    }
}

/// Looks up a `data_file` child by column name. Missing names are a programming error (the
/// builder was constructed from a schema that omitted a required DataFile column).
fn field_index(indices: &HashMap<String, usize>, name: &str) -> Result<usize> {
    indices.get(name).copied().ok_or_else(|| {
        Error::new(
            ErrorKind::Unexpected,
            format!("data_file struct is missing field `{name}`"),
        )
    })
}

/// Looks up a typed child builder of a [`StructBuilder`] by index, erroring (never panicking) if the
/// child at that index is not the expected type — a programming-error guard, since the struct fields are
/// constructed from [`data_file_fields`] in this same module.
fn struct_child<T: arrow_array::builder::ArrayBuilder>(
    builder: &mut StructBuilder,
    index: usize,
) -> Result<&mut T> {
    builder.field_builder::<T>(index).ok_or_else(|| {
        Error::new(
            ErrorKind::Unexpected,
            format!("data_file struct child builder at index {index} has an unexpected type"),
        )
    })
}

/// Downcasts a boxed (`Box<dyn ArrayBuilder>`) inner builder to a concrete type, erroring rather than
/// panicking — a programming-error guard, since the builder shapes come from this module's field list.
fn dyn_child<'a, T: ArrayBuilder>(
    builder: &'a mut Box<dyn ArrayBuilder>,
    what: &str,
) -> Result<&'a mut T> {
    builder.as_any_mut().downcast_mut::<T>().ok_or_else(|| {
        Error::new(
            ErrorKind::Unexpected,
            format!("data_file {what} builder has an unexpected inner type"),
        )
    })
}

/// Appends a `map<int, long>` value (one of the metrics-count maps), keys sorted for determinism. The map
/// builder is the boxed shape `make_builder` produces, so we downcast its key/value inner builders.
fn append_count_map(builder: &mut DynMapBuilder, map: &HashMap<i32, u64>) -> Result<()> {
    let mut keys: Vec<&i32> = map.keys().collect();
    keys.sort_unstable();
    for key in keys {
        dyn_child::<Int32Builder>(builder.keys(), "count map key")?.append_value(*key);
        dyn_child::<Int64Builder>(builder.values(), "count map value")?
            .append_value(map[key] as i64);
    }
    builder.append(true)?;
    Ok(())
}

/// Appends a `map<int, binary>` value (lower/upper bounds), keys sorted; values are the raw serialized
/// single-value bytes (Java map<int, binary>).
fn append_bound_map(builder: &mut DynMapBuilder, map: &HashMap<i32, Datum>) -> Result<()> {
    let mut keys: Vec<&i32> = map.keys().collect();
    keys.sort_unstable();
    for key in keys {
        dyn_child::<Int32Builder>(builder.keys(), "bound map key")?.append_value(*key);
        dyn_child::<LargeBinaryBuilder>(builder.values(), "bound map value")?
            .append_value(map[key].to_bytes()?);
    }
    builder.append(true)?;
    Ok(())
}

/// Appends an optional `list<long>` value (split offsets).
fn append_i64_list(builder: &mut DynListBuilder, values: Option<&[i64]>) -> Result<()> {
    match values {
        Some(values) => {
            let inner = dyn_child::<Int64Builder>(builder.values(), "split_offsets element")?;
            for value in values {
                inner.append_value(*value);
            }
            builder.append(true);
        }
        None => builder.append(false),
    }
    Ok(())
}

/// Appends an optional `list<int>` value (equality ids).
fn append_i32_list(builder: &mut DynListBuilder, values: Option<&[i32]>) -> Result<()> {
    match values {
        Some(values) => {
            let inner = dyn_child::<Int32Builder>(builder.values(), "equality_ids element")?;
            for value in values {
                inner.append_value(*value);
            }
            builder.append(true);
        }
        None => builder.append(false),
    }
    Ok(())
}

/// Appends one partition tuple to the partition [`StructBuilder`], dispatching each field on its
/// primitive type.
///
/// `partition` is a bare tuple: its values are positionally aligned with `source_field_ids` — the
/// partition-field ids of the spec the tuple was written under, in that spec's own field order (see
/// [`partition_field_ids_by_spec`]). `partition_type` is the type the metadata table PROJECTS, which
/// under partition evolution is a different spec's shape. Each projected field is therefore matched to
/// the tuple BY FIELD ID; a projected field the source spec does not carry is null-filled.
///
/// This is the field-id half of Java `PartitionUtil.coercePartition`
/// (`core/src/main/java/org/apache/iceberg/util/PartitionUtil.java`), which wraps the tuple in
/// `StructProjection.createAllowMissing(spec.partitionType(), partitionType)` — a projection whose
/// `positionMap` is built by matching field ids, and whose not-found entries read as null. Matching by
/// POSITION instead silently writes one partition field's value into another field's column whenever
/// the two specs agree on type but not on field id.
///
/// The other half of Java's coercion — projecting into the cross-spec UNIFIED partition type rather
/// than the table's default one — is still outstanding: `partition_type` here is
/// `TableMetadata::default_partition_type`, so a partition field that exists only in an older spec has
/// no column to land in and is dropped rather than surfaced. That is the
/// `Partitioning.partitionType` unification tracked in GAP_MATRIX R142.
///
/// Shared in-module helper: `files`/`entries` reach it through [`DataFileStructBuilder::append`], and
/// the `partitions` aggregating table reuses it directly for its `partition` column (Rule of Three).
pub(super) fn append_partition(
    builder: &mut StructBuilder,
    partition_type: &StructType,
    source_field_ids: &[i32],
    partition: &crate::spec::Struct,
) -> Result<()> {
    for (index, field) in partition_type.fields().iter().enumerate() {
        let primitive_type = field.field_type.as_primitive_type().ok_or_else(|| {
            Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "partition field '{}' has non-primitive type {:?}; not supported in the data_file metadata projection",
                    field.name, field.field_type
                ),
            )
        })?;
        let value = source_field_ids
            .iter()
            .position(|source_field_id| *source_field_id == field.id)
            .and_then(|source_index| partition.fields().get(source_index))
            .and_then(|value| value.as_ref());
        append_partition_field(builder, index, primitive_type, value)?;
    }
    builder.append(true);
    Ok(())
}

/// Appends a single partition-field value (or null) to the struct child builder at `index`, dispatching
/// on the field's primitive type. Mirrors the Arrow types produced by `type_to_arrow_type`.
fn append_partition_field(
    builder: &mut StructBuilder,
    index: usize,
    primitive_type: &PrimitiveType,
    value: Option<&Literal>,
) -> Result<()> {
    let primitive = match value {
        Some(Literal::Primitive(primitive)) => Some(primitive),
        Some(other) => {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!("non-primitive partition literal {other:?} is not supported"),
            ));
        }
        None => None,
    };

    macro_rules! append_typed {
        ($builder_ty:ty, $extract:expr) => {{
            let child = builder.field_builder::<$builder_ty>(index).ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("partition child builder at index {index} has an unexpected type"),
                )
            })?;
            match primitive {
                Some(primitive) => child.append_value($extract(primitive)?),
                None => child.append_null(),
            }
        }};
    }

    match primitive_type {
        PrimitiveType::Boolean => append_typed!(BooleanBuilder, extract_bool),
        PrimitiveType::Int => append_typed!(Int32Builder, extract_i32),
        PrimitiveType::Long => append_typed!(Int64Builder, extract_i64),
        PrimitiveType::Float => append_typed!(Float32Builder, extract_f32),
        PrimitiveType::Double => append_typed!(Float64Builder, extract_f64),
        PrimitiveType::Date => append_typed!(Date32Builder, extract_i32),
        PrimitiveType::Time => append_typed!(Time64MicrosecondBuilder, extract_i64),
        PrimitiveType::Timestamp => append_typed!(TimestampMicrosecondBuilder, extract_i64),
        PrimitiveType::Timestamptz => append_typed!(TimestampMicrosecondBuilder, extract_i64),
        PrimitiveType::TimestampNs => append_typed!(TimestampNanosecondBuilder, extract_i64),
        PrimitiveType::TimestamptzNs => append_typed!(TimestampNanosecondBuilder, extract_i64),
        PrimitiveType::String => append_typed!(StringBuilder, extract_string),
        PrimitiveType::Binary => append_typed!(BinaryBuilder, extract_binary),
        PrimitiveType::Decimal { .. } => append_typed!(Decimal128Builder, extract_i128),
        other => {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "partition field type {other:?} is not supported in the data_file metadata projection"
                ),
            ));
        }
    }
    Ok(())
}

fn type_mismatch(primitive: &PrimitiveLiteral) -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        format!("partition literal {primitive:?} does not match its partition field type"),
    )
}

fn extract_bool(primitive: &PrimitiveLiteral) -> Result<bool> {
    match primitive {
        PrimitiveLiteral::Boolean(value) => Ok(*value),
        other => Err(type_mismatch(other)),
    }
}

fn extract_i32(primitive: &PrimitiveLiteral) -> Result<i32> {
    match primitive {
        PrimitiveLiteral::Int(value) => Ok(*value),
        other => Err(type_mismatch(other)),
    }
}

fn extract_i64(primitive: &PrimitiveLiteral) -> Result<i64> {
    match primitive {
        PrimitiveLiteral::Long(value) => Ok(*value),
        other => Err(type_mismatch(other)),
    }
}

fn extract_f32(primitive: &PrimitiveLiteral) -> Result<f32> {
    match primitive {
        PrimitiveLiteral::Float(value) => Ok(value.into_inner()),
        other => Err(type_mismatch(other)),
    }
}

fn extract_f64(primitive: &PrimitiveLiteral) -> Result<f64> {
    match primitive {
        PrimitiveLiteral::Double(value) => Ok(value.into_inner()),
        other => Err(type_mismatch(other)),
    }
}

fn extract_string(primitive: &PrimitiveLiteral) -> Result<&str> {
    match primitive {
        PrimitiveLiteral::String(value) => Ok(value.as_str()),
        other => Err(type_mismatch(other)),
    }
}

fn extract_binary(primitive: &PrimitiveLiteral) -> Result<&[u8]> {
    match primitive {
        PrimitiveLiteral::Binary(value) => Ok(value.as_slice()),
        other => Err(type_mismatch(other)),
    }
}

fn extract_i128(primitive: &PrimitiveLiteral) -> Result<i128> {
    match primitive {
        PrimitiveLiteral::Int128(value) => Ok(*value),
        other => Err(type_mismatch(other)),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::{
        Int32Type, Int64Type, TimestampMicrosecondType, TimestampNanosecondType,
    };
    use arrow_array::{Array, StructArray};
    use arrow_schema::{DataType, TimeUnit};
    use futures::TryStreamExt;

    use super::{append_partition, data_file_fields};
    use crate::arrow::{UTC_TIME_ZONE, schema_to_arrow_schema};
    use crate::scan::tests::TableTestFixture;
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Literal, ManifestEntry,
        ManifestListWriter, ManifestStatus, ManifestWriterBuilder, NestedField, PartitionSpec,
        PrimitiveType, Schema, Struct, StructType, Transform, Type, UnboundPartitionField,
    };
    use crate::{ErrorKind, Result};

    /// Micros for `2024-01-01T04:30:00.000000Z` — the Java Spark example from F-V4-1.
    const TIMESTAMPTZ_MICROS: i64 = 1_704_083_400_000_000;
    /// Same instant as [`TIMESTAMPTZ_MICROS`], in nanoseconds.
    const TIMESTAMPTZ_NANOS: i64 = 1_704_083_400_000_000_000;
    const FILE_SIZE: u64 = 1024;

    /// Projects one partition field through [`append_partition`] (the refuse site), using the same
    /// Arrow child types `type_to_arrow_type` produces for `data_file_fields`.
    fn project_one_partition_field(
        primitive: PrimitiveType,
        value: Option<Literal>,
    ) -> Result<StructArray> {
        let partition_type = StructType::new(vec![Arc::new(NestedField::optional(
            1000,
            "ts",
            Type::Primitive(primitive),
        ))]);
        let iceberg_schema = Schema::builder()
            .with_fields(data_file_fields(&partition_type))
            .build()
            .expect("data_file_fields schema is statically valid");
        let arrow_schema =
            schema_to_arrow_schema(&iceberg_schema).expect("data_file_fields convert to Arrow");
        let partition_arrow = match arrow_schema
            .field_with_name("partition")
            .expect("data_file projection has a partition column")
            .data_type()
        {
            DataType::Struct(fields) => fields.clone(),
            other => panic!("partition column must be a struct, got {other:?}"),
        };
        let mut builder = arrow_array::builder::StructBuilder::from_fields(partition_arrow, 0);
        append_partition(
            &mut builder,
            &partition_type,
            &[1000],
            &Struct::from_iter([value]),
        )?;
        Ok(builder.finish())
    }

    /// Rewrites the example-table fixture so identity(`x`) is sourced from `primitive` and the
    /// default partition type is recomputed. Writer/commit path is not used.
    fn with_identity_partition_source_type(
        fixture: TableTestFixture,
        primitive: PrimitiveType,
    ) -> TableTestFixture {
        let mut metadata = fixture.table.metadata().clone();
        let schema_ids: Vec<i32> = metadata.schemas.keys().copied().collect();
        for schema_id in schema_ids {
            let schema = metadata
                .schemas
                .get(&schema_id)
                .expect("schema id taken from keys")
                .clone();
            let fields: Vec<_> = schema
                .as_struct()
                .fields()
                .iter()
                .map(|field| {
                    if field.id == 1 {
                        Arc::new(NestedField::required(
                            1,
                            field.name.clone(),
                            Type::Primitive(primitive.clone()),
                        ))
                    } else {
                        field.clone()
                    }
                })
                .collect();
            let rebuilt = Schema::builder()
                .with_schema_id(schema.schema_id())
                .with_identifier_field_ids(schema.identifier_field_ids())
                .with_fields(fields)
                .build()
                .expect("rebuild schema with swapped identity source type");
            metadata.schemas.insert(schema_id, Arc::new(rebuilt));
        }
        let current_schema = metadata.current_schema().clone();
        metadata.default_partition_type = metadata
            .default_spec
            .partition_type(current_schema.as_ref())
            .expect("recompute default partition type after source-type swap");
        let mut fixture = fixture;
        fixture.table = fixture.table.clone().with_metadata(Arc::new(metadata));
        fixture
    }

    /// Partition-field id of `identity(x)` (source column `x`, schema field 1) — carried by BOTH specs
    /// of [`with_reordered_evolved_spec`].
    const PARTITION_FIELD_ID_X: i32 = 1000;
    /// Partition-field id of `identity(y)` (source column `y`, schema field 2) — added by spec 1.
    const PARTITION_FIELD_ID_Y: i32 = 1001;

    /// Spec-EVOLUTION fixture. The template's spec 0 is `identity(x)` @ field id 1000. This adds spec 1
    /// as the new DEFAULT: `identity(y)` @ 1001 FOLLOWED BY `identity(x)` @ 1000 — monotonic, non-reused
    /// field ids (a Java-LEGAL evolution, unlike `TableTestFixture::new_with_two_identity_specs`, which
    /// reuses field id 1000 across two source columns and is a `"Conflicting partition fields"` refusal
    /// in Java `Partitioning.buildPartitionProjectionType`).
    ///
    /// Two properties make it the PT-0 corruption pin:
    ///  * spec 1's TUPLE order (`y`, `x`) differs from its field-id order, so a spec-0 file's 1-tuple
    ///    `(x)` lines up positionally with the projected `y` column;
    ///  * both partition fields are `long`, so a positional read produces no type error — the value is
    ///    silently written into the wrong column.
    fn with_reordered_evolved_spec(fixture: TableTestFixture) -> TableTestFixture {
        let mut metadata = fixture.table.metadata().clone();
        let current_schema = metadata.current_schema().clone();
        let evolved_spec = Arc::new(
            PartitionSpec::builder(current_schema.clone())
                .with_spec_id(1)
                .add_unbound_field(
                    UnboundPartitionField::builder()
                        .source_id(2)
                        .name("y".to_string())
                        .field_id(PARTITION_FIELD_ID_Y)
                        .transform(Transform::Identity)
                        .build(),
                )
                .expect("identity(y) is a valid partition field")
                .add_unbound_field(
                    UnboundPartitionField::builder()
                        .source_id(1)
                        .name("x".to_string())
                        .field_id(PARTITION_FIELD_ID_X)
                        .transform(Transform::Identity)
                        .build(),
                )
                .expect("identity(x) is a valid partition field")
                .build()
                .expect("build the evolved spec"),
        );
        metadata.default_partition_type = evolved_spec
            .partition_type(current_schema.as_ref())
            .expect("evolved spec resolves against the current schema");
        metadata.default_spec = evolved_spec.clone();
        metadata.partition_specs.insert(1, evolved_spec);

        let mut fixture = fixture;
        fixture.table = fixture.table.clone().with_metadata(Arc::new(metadata));
        fixture
    }

    /// Projects one partition tuple through [`append_partition`] against the two-field `long` projection
    /// `{1001 y, 1000 x}` that [`with_reordered_evolved_spec`]'s default spec produces.
    fn project_reordered_pair(
        source_field_ids: &[i32],
        values: Vec<Option<Literal>>,
    ) -> StructArray {
        let partition_type = StructType::new(vec![
            Arc::new(NestedField::optional(
                PARTITION_FIELD_ID_Y,
                "y",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                PARTITION_FIELD_ID_X,
                "x",
                Type::Primitive(PrimitiveType::Long),
            )),
        ]);
        let iceberg_schema = Schema::builder()
            .with_fields(data_file_fields(&partition_type))
            .build()
            .expect("data_file_fields schema is statically valid");
        let arrow_schema =
            schema_to_arrow_schema(&iceberg_schema).expect("data_file_fields convert to Arrow");
        let partition_arrow = match arrow_schema
            .field_with_name("partition")
            .expect("data_file projection has a partition column")
            .data_type()
        {
            DataType::Struct(fields) => fields.clone(),
            other => panic!("partition column must be a struct, got {other:?}"),
        };
        let mut builder = arrow_array::builder::StructBuilder::from_fields(partition_arrow, 0);
        append_partition(
            &mut builder,
            &partition_type,
            source_field_ids,
            &Struct::from_iter(values),
        )
        .expect("same-typed long partition fields must project");
        builder.finish()
    }

    /// Writes one Added DATA file with `partition` under the table's DEFAULT spec and stitches it into
    /// the current snapshot's manifest list (inspect-test mold: no real parquet).
    async fn write_one_partitioned_data_file(fixture: &TableTestFixture, partition: Struct) {
        let default_spec_id = fixture.table.metadata().default_partition_spec_id();
        write_data_files_under_specs(fixture, &[(default_spec_id, partition)]).await;
    }

    /// Writes one Added DATA file per `(spec_id, partition)` pair, EACH IN ITS OWN MANIFEST written
    /// under that spec — the shape partition evolution actually produces — then rewrites the current
    /// snapshot's manifest list to reference them all (inspect-test mold: no real parquet).
    ///
    /// Each `partition` tuple must be ordered by ITS OWN spec's field order, exactly as a writer under
    /// that spec would have serialized it.
    async fn write_data_files_under_specs(fixture: &TableTestFixture, files: &[(i32, Struct)]) {
        let metadata = fixture.table.metadata().clone();
        let current_snapshot = metadata
            .current_snapshot()
            .expect("example fixture has a current snapshot");
        let current_schema = current_snapshot
            .schema(&metadata)
            .expect("current snapshot schema");

        let mut manifests = Vec::with_capacity(files.len());
        for (index, (spec_id, partition)) in files.iter().enumerate() {
            let spec = metadata
                .partition_spec_by_id(*spec_id)
                .expect("inspect-test data file references a spec in the fixture metadata")
                .clone();
            let output = fixture
                .table
                .file_io()
                .new_output(format!(
                    "{}/metadata/manifest_proj_{}.avro",
                    fixture.table_location,
                    uuid::Uuid::new_v4()
                ))
                .expect("create inspect-test manifest output");
            let mut writer = ManifestWriterBuilder::new(
                output,
                Some(current_snapshot.snapshot_id()),
                None,
                current_schema.clone(),
                spec.as_ref().clone(),
            )
            .build_v2_data();
            writer
                .add_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Added)
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(*spec_id)
                                .content(DataContentType::Data)
                                .file_path(format!(
                                    "{}/proj_{index}.parquet",
                                    &fixture.table_location
                                ))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(FILE_SIZE)
                                .record_count(1)
                                .partition(partition.clone())
                                .build()
                                .expect("build inspect-test data file"),
                        )
                        .build(),
                )
                .expect("add inspect-test manifest entry");
            manifests.push(
                writer
                    .write_manifest_file()
                    .await
                    .expect("write inspect-test data manifest"),
            );
        }

        let mut manifest_list = ManifestListWriter::v2(
            fixture
                .table
                .file_io()
                .new_output(current_snapshot.manifest_list())
                .expect("open current snapshot manifest list"),
            current_snapshot.snapshot_id(),
            current_snapshot.parent_snapshot_id(),
            current_snapshot.sequence_number(),
        );
        manifest_list
            .add_manifests(manifests.into_iter())
            .expect("add inspect-test manifest to list");
        manifest_list
            .close()
            .await
            .expect("close inspect-test manifest list");
    }

    async fn scan_files_single_batch(fixture: &TableTestFixture) -> arrow_array::RecordBatch {
        let batches: Vec<_> = fixture
            .table
            .inspect()
            .files()
            .scan()
            .await
            .expect("files metadata-table scan")
            .try_collect()
            .await
            .expect("collect files metadata-table batches");
        arrow_select::concat::concat_batches(&batches[0].schema(), &batches)
            .expect("concat files metadata-table batches")
    }

    #[test]
    fn append_partition_projects_timestamptz_micros() {
        // RISK: Timestamptz falls through `other` and FeatureUnsupported's the `.files` / `.partitions`
        // projection. The arm must emit the same i64 micros the sibling `readable_metrics` path uses.
        let projected = project_one_partition_field(
            PrimitiveType::Timestamptz,
            Some(Literal::timestamptz(TIMESTAMPTZ_MICROS)),
        )
        .expect("Timestamptz identity partition must project");
        assert_eq!(projected.len(), 1);
        assert_eq!(
            projected.column(0).data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, Some(UTC_TIME_ZONE.into()))
        );
        let values = projected
            .column(0)
            .as_primitive::<TimestampMicrosecondType>();
        assert_eq!(values.value(0), TIMESTAMPTZ_MICROS);
    }

    #[test]
    fn append_partition_projects_timestamptz_ns() {
        // RISK: TimestamptzNs is the same-class twin of Timestamptz; leaving it in `other` reopens
        // the F-V4-1 refuse on V3 tables. Pin the ns builder + extract_i64 arm.
        let projected = project_one_partition_field(
            PrimitiveType::TimestamptzNs,
            Some(Literal::timestamptz_nano(TIMESTAMPTZ_NANOS)),
        )
        .expect("TimestamptzNs identity partition must project");
        assert_eq!(projected.len(), 1);
        assert_eq!(
            projected.column(0).data_type(),
            &DataType::Timestamp(TimeUnit::Nanosecond, Some(UTC_TIME_ZONE.into()))
        );
        let values = projected
            .column(0)
            .as_primitive::<TimestampNanosecondType>();
        assert_eq!(values.value(0), TIMESTAMPTZ_NANOS);
    }

    #[test]
    fn append_partition_still_refuses_uuid() {
        // RISK: A7 — Uuid stays refused. A drive-by "support every leftover primitive" would flip
        // this silently; the existing needle must keep firing.
        let error = project_one_partition_field(
            PrimitiveType::Uuid,
            Some(Literal::uuid(uuid::Uuid::nil())),
        )
        .expect_err("Uuid identity partition must stay unsupported");
        assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
        assert!(
            error.message().contains(
                "partition field type Uuid is not supported in the data_file metadata projection"
            ),
            "expected existing refuse needle, got: {}",
            error.message()
        );
    }

    #[test]
    fn append_partition_still_refuses_fixed() {
        // RISK: A7 — Fixed stays refused, independently of Uuid (one leftover-arm mutation must
        // not be able to cover both negatives).
        let error =
            project_one_partition_field(PrimitiveType::Fixed(16), Some(Literal::fixed([0u8; 16])))
                .expect_err("Fixed identity partition must stay unsupported");
        assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
        assert!(
            error.message().contains(
                "partition field type Fixed(16) is not supported in the data_file metadata projection"
            ),
            "expected existing refuse needle, got: {}",
            error.message()
        );
    }

    #[tokio::test]
    async fn files_and_partitions_project_timestamptz_identity_value() {
        // RISK: schema conversion can already emit timestamptz children while the append match
        // still refuses. This is the metadata-table read over a timestamptz-identity-partitioned
        // table (F-V4-1). Writer/commit path is not exercised.
        let fixture = with_identity_partition_source_type(
            TableTestFixture::new(),
            PrimitiveType::Timestamptz,
        );
        let files_schema = schema_to_arrow_schema(&fixture.table.inspect().files().schema())
            .expect("files metadata-table schema converts");
        let partition_type = match files_schema
            .field_with_name("partition")
            .expect("files table has partition")
            .data_type()
        {
            DataType::Struct(fields) => fields[0].data_type().clone(),
            other => panic!("expected partition struct, got {other:?}"),
        };
        assert_eq!(
            partition_type,
            DataType::Timestamp(TimeUnit::Microsecond, Some(UTC_TIME_ZONE.into())),
            "type_to_arrow_type must already produce a timestamptz partition child"
        );

        write_one_partitioned_data_file(
            &fixture,
            Struct::from_iter([Some(Literal::timestamptz(TIMESTAMPTZ_MICROS))]),
        )
        .await;

        let files_batch = scan_files_single_batch(&fixture).await;
        assert_eq!(files_batch.num_rows(), 1);
        let files_partition = files_batch
            .column_by_name("partition")
            .expect("files.partition")
            .as_struct();
        let files_ts = files_partition
            .column(0)
            .as_primitive::<TimestampMicrosecondType>();
        assert_eq!(files_ts.value(0), TIMESTAMPTZ_MICROS);

        let partitions_batches: Vec<_> = fixture
            .table
            .inspect()
            .partitions()
            .scan()
            .await
            .expect("partitions metadata-table scan")
            .try_collect()
            .await
            .expect("collect partitions batches");
        let partitions_batch = arrow_select::concat::concat_batches(
            &partitions_batches[0].schema(),
            &partitions_batches,
        )
        .expect("concat partitions batches");
        assert_eq!(partitions_batch.num_rows(), 1);
        let partitions_ts = partitions_batch
            .column_by_name("partition")
            .expect("partitions.partition")
            .as_struct()
            .column(0)
            .as_primitive::<TimestampMicrosecondType>();
        assert_eq!(partitions_ts.value(0), TIMESTAMPTZ_MICROS);
    }

    #[tokio::test]
    async fn files_table_projects_timestamptz_ns_identity_value() {
        // RISK: same-class twin of Timestamptz. A V3 identity(timestamptz_ns) table must project
        // the ns value through `.files`, not FeatureUnsupported.
        let fixture = with_identity_partition_source_type(
            TableTestFixture::new(),
            PrimitiveType::TimestamptzNs,
        );
        write_one_partitioned_data_file(
            &fixture,
            Struct::from_iter([Some(Literal::timestamptz_nano(TIMESTAMPTZ_NANOS))]),
        )
        .await;

        let batch = scan_files_single_batch(&fixture).await;
        assert_eq!(batch.num_rows(), 1);
        let ts = batch
            .column_by_name("partition")
            .expect("files.partition")
            .as_struct()
            .column(0)
            .as_primitive::<TimestampNanosecondType>();
        assert_eq!(ts.value(0), TIMESTAMPTZ_NANOS);
        assert_eq!(
            batch
                .column_by_name("partition")
                .expect("files.partition")
                .as_struct()
                .column(0)
                .data_type(),
            &DataType::Timestamp(TimeUnit::Nanosecond, Some(UTC_TIME_ZONE.into()))
        );
    }

    #[tokio::test]
    async fn files_table_still_refuses_uuid_identity_partition() {
        // RISK: A7 negative pin — Uuid identity partition must keep the existing needle on the
        // metadata-table path, not only on the isolated append helper.
        let fixture =
            with_identity_partition_source_type(TableTestFixture::new(), PrimitiveType::Uuid);
        write_one_partitioned_data_file(
            &fixture,
            Struct::from_iter([Some(Literal::uuid(uuid::Uuid::nil()))]),
        )
        .await;
        let error = match fixture.table.inspect().files().scan().await {
            Ok(_) => panic!("Uuid identity partition must stay unsupported on .files"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
        assert!(
            error.message().contains(
                "partition field type Uuid is not supported in the data_file metadata projection"
            ),
            "expected existing refuse needle, got: {}",
            error.message()
        );
    }

    #[tokio::test]
    async fn files_table_still_refuses_fixed_identity_partition() {
        // RISK: A7 negative pin — Fixed identity partition must keep the existing needle on the
        // metadata-table path.
        let fixture =
            with_identity_partition_source_type(TableTestFixture::new(), PrimitiveType::Fixed(16));
        write_one_partitioned_data_file(
            &fixture,
            Struct::from_iter([Some(Literal::fixed([0u8; 16]))]),
        )
        .await;
        let error = match fixture.table.inspect().files().scan().await {
            Ok(_) => panic!("Fixed identity partition must stay unsupported on .files"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
        assert!(
            error.message().contains(
                "partition field type Fixed(16) is not supported in the data_file metadata projection"
            ),
            "expected existing refuse needle, got: {}",
            error.message()
        );
    }
    #[test]
    fn append_partition_null_fills_a_projected_field_absent_from_the_source_spec() {
        // RISK (PT-0): the tuple `(x=777)` was written under spec 0 (`[1000]`), and the projection is
        // spec 1's `{1001 y, 1000 x}`. A POSITIONAL read takes tuple index 0 for projected index 0 and
        // writes 777 — an `x` value — into the `y` column. Both fields are `long`, so nothing errors.
        let projected =
            project_reordered_pair(&[PARTITION_FIELD_ID_X], vec![Some(Literal::long(777))]);
        assert_eq!(projected.len(), 1);
        let y = projected.column(0).as_primitive::<Int64Type>();
        let x = projected.column(1).as_primitive::<Int64Type>();
        assert!(
            y.is_null(0),
            "projected field 1001 (`y`) is absent from spec 0, so it must null-fill (Java \
             `StructProjection.createAllowMissing`), got {}",
            y.value(0)
        );
        assert_eq!(
            x.value(0),
            777,
            "projected field 1000 (`x`) must take the value the source spec holds at ITS position"
        );
    }

    #[test]
    fn append_partition_reorders_a_full_tuple_by_field_id() {
        // RISK (PT-0): a source spec whose tuple order differs from the projection's. Here the source is
        // `[1000, 1001]` = `(x=777, y=888)` and the projection is `{1001 y, 1000 x}` — a positional walk
        // swaps the two values with no error at all.
        let projected =
            project_reordered_pair(&[PARTITION_FIELD_ID_X, PARTITION_FIELD_ID_Y], vec![
                Some(Literal::long(777)),
                Some(Literal::long(888)),
            ]);
        assert_eq!(projected.len(), 1);
        assert_eq!(
            projected.column(0).as_primitive::<Int64Type>().value(0),
            888,
            "`y` (field 1001) must come from the source tuple's field-1001 slot"
        );
        assert_eq!(
            projected.column(1).as_primitive::<Int64Type>().value(0),
            777,
            "`x` (field 1000) must come from the source tuple's field-1000 slot"
        );
    }

    #[tokio::test]
    async fn files_table_projects_partition_values_by_field_id_under_spec_evolution() {
        // RISK (PT-0): the end-to-end corruption. Spec 0's live file carries the 1-tuple `(x=777)`;
        // the projected (default = spec 1) partition type is `{1001 y, 1000 x}`. Reading the tuple
        // POSITIONALLY surfaces `y = 777` / `x = null` on the `files` table — an `x` value reported
        // under `y`, silently, because both fields are `long`. Java coerces each file's partition into
        // the projected type BY FIELD ID (`PartitionUtil.coercePartition` →
        // `StructProjection.createAllowMissing`), which yields `y = null` / `x = 777`.
        let fixture = with_reordered_evolved_spec(TableTestFixture::new());
        write_data_files_under_specs(&fixture, &[
            // Written under spec 0 = `[identity(x)]`.
            (0, Struct::from_iter([Some(Literal::long(777))])),
            // Written under spec 1 = `[identity(y), identity(x)]`, in THAT tuple order.
            (
                1,
                Struct::from_iter([Some(Literal::long(888)), Some(Literal::long(999))]),
            ),
        ])
        .await;

        let batch = scan_files_single_batch(&fixture).await;
        assert_eq!(batch.num_rows(), 2);

        let spec_ids = batch
            .column_by_name("spec_id")
            .expect("files.spec_id")
            .as_primitive::<Int32Type>();
        assert_eq!(
            (spec_ids.value(0), spec_ids.value(1)),
            (0, 1),
            "rows follow manifest-list order: the spec-0 file first, then the spec-1 file"
        );

        let partition = batch
            .column_by_name("partition")
            .expect("files.partition")
            .as_struct();
        let y = partition.column(0).as_primitive::<Int64Type>();
        let x = partition.column(1).as_primitive::<Int64Type>();

        assert!(
            y.is_null(0),
            "the spec-0 file has no field-1001 (`y`) partition value; positional reading reports {} \
             there, which is the spec-0 file's `x` value",
            y.value(0)
        );
        assert_eq!(
            x.value(0),
            777,
            "the spec-0 file's field-1000 (`x`) value must land in the `x` column"
        );

        assert_eq!(y.value(1), 888, "the spec-1 file's `y` value");
        assert_eq!(x.value(1), 999, "the spec-1 file's `x` value");
    }
}
