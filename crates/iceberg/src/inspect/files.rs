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

//! The `files` family of metadata tables: `files`, `data_files`, `delete_files`.
//!
//! Each exposes the data/delete files referenced by the table's **current snapshot** as rows, with the
//! data-file column set (content, file path/format, partition, record/size counts, the metrics maps,
//! and the V3 deletion-vector fields). The three tables share one schema, one read, and one row builder
//! and differ ONLY by which manifests they read — mirroring Java `BaseFilesTable`:
//!
//! - [`FilesTable`]       → all manifests          (Java `FilesTable` / `snapshot().allManifests()`)
//! - [`DataFilesTable`]   → DATA-content manifests  (Java `DataFilesTable` / `snapshot().dataManifests()`)
//! - [`DeleteFilesTable`] → DELETE-content manifests (Java `DeleteFilesTable` / `snapshot().deleteManifests()`)
//!
//! Within a selected manifest only LIVE entries (Added/Existing, [`ManifestEntry::is_alive`]) are rows.
//!
//! References:
//! - <https://github.com/apache/iceberg/blob/main/core/src/main/java/org/apache/iceberg/BaseFilesTable.java>
//! - <https://github.com/apache/iceberg/blob/main/api/src/main/java/org/apache/iceberg/DataFile.java>
//!
//! Deferred column: `readable_metrics` (Java `MetricsUtil.readableMetricsStruct` — a virtual per-data-column
//! struct of human-readable min/max/counts). All raw columns, including the metrics maps, are present.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{
    BinaryBuilder, BooleanBuilder, Date32Builder, Decimal128Builder, Float32Builder,
    Float64Builder, Int32Builder, Int64Builder, LargeBinaryBuilder, ListBuilder, MapBuilder,
    MapFieldNames, StringBuilder, StructBuilder, Time64MicrosecondBuilder,
    TimestampMicrosecondBuilder, TimestampNanosecondBuilder,
};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Fields};
use futures::{StreamExt, stream};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::{DEFAULT_MAP_FIELD_NAME, schema_to_arrow_schema};
use crate::scan::ArrowRecordBatchStream;
use crate::spec::{
    Datum, ListType, Literal, ManifestContentType, MapType, NestedField, PrimitiveLiteral,
    PrimitiveType, Schema, StructType, Type,
};
use crate::table::Table;
use crate::{Error, ErrorKind, Result};

/// Which files a [`FilesTable`] exposes — the only thing that differs across the three tables.
///
/// Mirrors the Java `BaseFilesTableScan.manifests()` override on each concrete table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilesTableKind {
    /// All manifests (Java `FilesTable`).
    All,
    /// DATA-content manifests only (Java `DataFilesTable`).
    Data,
    /// DELETE-content manifests only (Java `DeleteFilesTable`).
    Deletes,
}

impl FilesTableKind {
    /// Returns whether a manifest of the given content type should be read for this table.
    fn includes_manifest(&self, content: ManifestContentType) -> bool {
        match self {
            FilesTableKind::All => true,
            FilesTableKind::Data => content == ManifestContentType::Data,
            FilesTableKind::Deletes => content == ManifestContentType::Deletes,
        }
    }
}

/// The shared base for the `files` / `data_files` / `delete_files` metadata tables (Java
/// `BaseFilesTable`). The three concrete tables wrap this with a fixed [`FilesTableKind`].
pub struct FilesTable<'a> {
    table: &'a Table,
    kind: FilesTableKind,
}

impl<'a> FilesTable<'a> {
    fn new(table: &'a Table, kind: FilesTableKind) -> Self {
        Self { table, kind }
    }

    /// Create a `files` table (all data + delete files in the current snapshot).
    pub fn all(table: &'a Table) -> Self {
        Self::new(table, FilesTableKind::All)
    }

    /// Create a `data_files` table (only DATA-content files in the current snapshot).
    pub fn data(table: &'a Table) -> Self {
        Self::new(table, FilesTableKind::Data)
    }

    /// Create a `delete_files` table (only position/equality delete files in the current snapshot).
    pub fn deletes(table: &'a Table) -> Self {
        Self::new(table, FilesTableKind::Deletes)
    }

    /// Returns the iceberg schema of the files metadata table.
    ///
    /// Mirrors Java `DataFile.getType(partitionType).fields()` — the field ids are the canonical
    /// `DataFile` ids from `api/DataFile.java`. The partition column carries the table's DEFAULT
    /// partition type. `readable_metrics` is deferred.
    pub fn schema(&self) -> Schema {
        let partition_type = self.table.metadata().default_partition_type().clone();
        let fields = vec![
            NestedField::optional(134, "content", Type::Primitive(PrimitiveType::Int)),
            NestedField::required(100, "file_path", Type::Primitive(PrimitiveType::String)),
            NestedField::required(101, "file_format", Type::Primitive(PrimitiveType::String)),
            NestedField::optional(141, "spec_id", Type::Primitive(PrimitiveType::Int)),
            NestedField::required(102, "partition", Type::Struct(partition_type)),
            NestedField::required(103, "record_count", Type::Primitive(PrimitiveType::Long)),
            NestedField::required(
                104,
                "file_size_in_bytes",
                Type::Primitive(PrimitiveType::Long),
            ),
            NestedField::optional(108, "column_sizes", int_long_map(117, 118)),
            NestedField::optional(109, "value_counts", int_long_map(119, 120)),
            NestedField::optional(110, "null_value_counts", int_long_map(121, 122)),
            NestedField::optional(137, "nan_value_counts", int_long_map(138, 139)),
            NestedField::optional(125, "lower_bounds", int_binary_map(126, 127)),
            NestedField::optional(128, "upper_bounds", int_binary_map(129, 130)),
            NestedField::optional(131, "key_metadata", Type::Primitive(PrimitiveType::Binary)),
            NestedField::optional(132, "split_offsets", long_list(133)),
            NestedField::optional(135, "equality_ids", int_list(136)),
            NestedField::optional(140, "sort_order_id", Type::Primitive(PrimitiveType::Int)),
            NestedField::optional(142, "first_row_id", Type::Primitive(PrimitiveType::Long)),
            NestedField::optional(
                143,
                "referenced_data_file",
                Type::Primitive(PrimitiveType::String),
            ),
            NestedField::optional(144, "content_offset", Type::Primitive(PrimitiveType::Long)),
            NestedField::optional(
                145,
                "content_size_in_bytes",
                Type::Primitive(PrimitiveType::Long),
            ),
        ];
        Schema::builder()
            .with_fields(fields.into_iter().map(Arc::new))
            .build()
            .expect("files metadata table schema is statically valid")
    }

    /// Scans the files metadata table.
    ///
    /// Reads the current snapshot's manifest list, selects the manifests whose content passes this
    /// table's [`FilesTableKind`] filter, and emits one row per LIVE manifest entry built from its
    /// [`crate::spec::DataFile`]. An empty table (no current snapshot) yields a single empty batch.
    pub async fn scan(&self) -> Result<ArrowRecordBatchStream> {
        let arrow_schema = Arc::new(schema_to_arrow_schema(&self.schema())?);
        let partition_type = self.table.metadata().default_partition_type().clone();
        let partition_arrow_fields = partition_struct_fields(&arrow_schema)?;

        let mut builder = FilesRowBuilder::new(&partition_arrow_fields, &partition_type)?;

        if let Some(snapshot) = self.table.metadata().current_snapshot() {
            let manifest_list = snapshot
                .load_manifest_list(self.table.file_io(), self.table.metadata())
                .await?;
            for manifest_file in manifest_list.entries() {
                if !self.kind.includes_manifest(manifest_file.content) {
                    continue;
                }
                let manifest = manifest_file.load_manifest(self.table.file_io()).await?;
                for entry in manifest.entries() {
                    if entry.is_alive() {
                        builder.append(entry.data_file())?;
                    }
                }
            }
        }

        let batch = builder.finish(arrow_schema)?;
        Ok(stream::iter(vec![Ok(batch)]).boxed())
    }
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

/// Extracts the Arrow `partition` struct field list from the converted files-table Arrow schema.
fn partition_struct_fields(arrow_schema: &arrow_schema::Schema) -> Result<Fields> {
    match arrow_schema.field_with_name("partition")?.data_type() {
        DataType::Struct(fields) => Ok(fields.clone()),
        other => Err(Error::new(
            ErrorKind::Unexpected,
            format!("files metadata table partition column must be a struct, got {other:?}"),
        )),
    }
}

/// Builds the metrics map field (`key_value` struct of `key: int`, `value`) for a `MapBuilder`,
/// carrying the canonical Iceberg field ids so the produced Arrow schema matches `schema_to_arrow_schema`.
fn metrics_map_fields(
    key_id: &str,
    value_id: &str,
    value_type: DataType,
) -> (Arc<Field>, Arc<Field>) {
    let keys_field = Arc::new(Field::new("key", DataType::Int32, false).with_metadata(
        HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), key_id.to_string())]),
    ));
    // The `DataFile` metric maps use `MapType.ofRequired`, so the value is non-null.
    let values_field = Arc::new(Field::new("value", value_type, false).with_metadata(
        HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), value_id.to_string())]),
    ));
    (keys_field, values_field)
}

fn map_field_names() -> MapFieldNames {
    MapFieldNames {
        entry: DEFAULT_MAP_FIELD_NAME.to_string(),
        key: "key".to_string(),
        value: "value".to_string(),
    }
}

/// Accumulates files-table rows column-by-column and finishes them into a single [`RecordBatch`].
///
/// One builder per output column, mirroring the manifests/snapshots tables' style. The partition
/// column is a [`StructBuilder`] whose children match the table's DEFAULT partition type.
struct FilesRowBuilder<'a> {
    partition_type: &'a StructType,

    content: Int32Builder,
    file_path: StringBuilder,
    file_format: StringBuilder,
    spec_id: Int32Builder,
    partition: StructBuilder,
    record_count: Int64Builder,
    file_size_in_bytes: Int64Builder,
    column_sizes: MapBuilder<Int32Builder, Int64Builder>,
    value_counts: MapBuilder<Int32Builder, Int64Builder>,
    null_value_counts: MapBuilder<Int32Builder, Int64Builder>,
    nan_value_counts: MapBuilder<Int32Builder, Int64Builder>,
    lower_bounds: MapBuilder<Int32Builder, LargeBinaryBuilder>,
    upper_bounds: MapBuilder<Int32Builder, LargeBinaryBuilder>,
    key_metadata: LargeBinaryBuilder,
    split_offsets: ListBuilder<Int64Builder>,
    equality_ids: ListBuilder<Int32Builder>,
    sort_order_id: Int32Builder,
    first_row_id: Int64Builder,
    referenced_data_file: StringBuilder,
    content_offset: Int64Builder,
    content_size_in_bytes: Int64Builder,
}

impl<'a> FilesRowBuilder<'a> {
    fn new(partition_arrow_fields: &Fields, partition_type: &'a StructType) -> Result<Self> {
        let count_map = |key_id: &str, value_id: &str| {
            let (k, v) = metrics_map_fields(key_id, value_id, DataType::Int64);
            MapBuilder::new(
                Some(map_field_names()),
                Int32Builder::new(),
                Int64Builder::new(),
            )
            .with_keys_field(k)
            .with_values_field(v)
        };
        let binary_map = |key_id: &str, value_id: &str| {
            let (k, v) = metrics_map_fields(key_id, value_id, DataType::LargeBinary);
            MapBuilder::new(
                Some(map_field_names()),
                Int32Builder::new(),
                LargeBinaryBuilder::new(),
            )
            .with_keys_field(k)
            .with_values_field(v)
        };

        Ok(Self {
            partition_type,
            content: Int32Builder::new(),
            file_path: StringBuilder::new(),
            file_format: StringBuilder::new(),
            spec_id: Int32Builder::new(),
            partition: StructBuilder::from_fields(partition_arrow_fields.clone(), 0),
            record_count: Int64Builder::new(),
            file_size_in_bytes: Int64Builder::new(),
            column_sizes: count_map("117", "118"),
            value_counts: count_map("119", "120"),
            null_value_counts: count_map("121", "122"),
            nan_value_counts: count_map("138", "139"),
            lower_bounds: binary_map("126", "127"),
            upper_bounds: binary_map("129", "130"),
            key_metadata: LargeBinaryBuilder::new(),
            split_offsets: list_builder_i64(133),
            equality_ids: list_builder_i32(136),
            sort_order_id: Int32Builder::new(),
            first_row_id: Int64Builder::new(),
            referenced_data_file: StringBuilder::new(),
            content_offset: Int64Builder::new(),
            content_size_in_bytes: Int64Builder::new(),
        })
    }

    /// Appends one row from a [`crate::spec::DataFile`].
    fn append(&mut self, data_file: &crate::spec::DataFile) -> Result<()> {
        self.content.append_value(data_file.content_type() as i32);
        self.file_path.append_value(data_file.file_path());
        self.file_format
            .append_value(data_file.file_format().to_string());
        self.spec_id.append_value(data_file.partition_spec_id);

        append_partition(
            &mut self.partition,
            self.partition_type,
            data_file.partition(),
        )?;

        self.record_count
            .append_value(data_file.record_count() as i64);
        self.file_size_in_bytes
            .append_value(data_file.file_size_in_bytes() as i64);

        append_count_map(&mut self.column_sizes, data_file.column_sizes())?;
        append_count_map(&mut self.value_counts, data_file.value_counts())?;
        append_count_map(&mut self.null_value_counts, data_file.null_value_counts())?;
        append_count_map(&mut self.nan_value_counts, data_file.nan_value_counts())?;
        append_bound_map(&mut self.lower_bounds, data_file.lower_bounds())?;
        append_bound_map(&mut self.upper_bounds, data_file.upper_bounds())?;

        self.key_metadata.append_option(data_file.key_metadata());

        append_i64_list(&mut self.split_offsets, data_file.split_offsets());
        append_i32_list(&mut self.equality_ids, data_file.equality_ids().as_deref());

        self.sort_order_id.append_option(data_file.sort_order_id());
        self.first_row_id.append_option(data_file.first_row_id());
        self.referenced_data_file
            .append_option(data_file.referenced_data_file());
        self.content_offset
            .append_option(data_file.content_offset());
        self.content_size_in_bytes
            .append_option(data_file.content_size_in_bytes());
        Ok(())
    }

    /// Finishes all column builders into a single [`RecordBatch`].
    fn finish(mut self, arrow_schema: Arc<arrow_schema::Schema>) -> Result<RecordBatch> {
        let columns: Vec<ArrayRef> = vec![
            Arc::new(self.content.finish()),
            Arc::new(self.file_path.finish()),
            Arc::new(self.file_format.finish()),
            Arc::new(self.spec_id.finish()),
            Arc::new(self.partition.finish()),
            Arc::new(self.record_count.finish()),
            Arc::new(self.file_size_in_bytes.finish()),
            Arc::new(self.column_sizes.finish()),
            Arc::new(self.value_counts.finish()),
            Arc::new(self.null_value_counts.finish()),
            Arc::new(self.nan_value_counts.finish()),
            Arc::new(self.lower_bounds.finish()),
            Arc::new(self.upper_bounds.finish()),
            Arc::new(self.key_metadata.finish()),
            Arc::new(self.split_offsets.finish()),
            Arc::new(self.equality_ids.finish()),
            Arc::new(self.sort_order_id.finish()),
            Arc::new(self.first_row_id.finish()),
            Arc::new(self.referenced_data_file.finish()),
            Arc::new(self.content_offset.finish()),
            Arc::new(self.content_size_in_bytes.finish()),
        ];
        Ok(RecordBatch::try_new(arrow_schema, columns)?)
    }
}

/// Builds a `list<long>` builder whose element field carries the given Iceberg field id.
fn list_builder_i64(element_id: i32) -> ListBuilder<Int64Builder> {
    ListBuilder::new(Int64Builder::new())
        .with_field(list_element_field(element_id, DataType::Int64))
}

/// Builds a `list<int>` builder whose element field carries the given Iceberg field id.
fn list_builder_i32(element_id: i32) -> ListBuilder<Int32Builder> {
    ListBuilder::new(Int32Builder::new())
        .with_field(list_element_field(element_id, DataType::Int32))
}

/// The Arrow element field (`element`, required) for a list column, carrying the Iceberg field id so the
/// produced schema matches `schema_to_arrow_schema` (list elements are required in the `DataFile` schema).
fn list_element_field(element_id: i32, data_type: DataType) -> Arc<Field> {
    Arc::new(
        Field::new("element", data_type, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            element_id.to_string(),
        )])),
    )
}

/// Appends a `map<int, long>` value (one of the metrics-count maps), keys sorted for determinism.
fn append_count_map(
    builder: &mut MapBuilder<Int32Builder, Int64Builder>,
    map: &HashMap<i32, u64>,
) -> Result<()> {
    let mut keys: Vec<&i32> = map.keys().collect();
    keys.sort_unstable();
    for key in keys {
        builder.keys().append_value(*key);
        builder.values().append_value(map[key] as i64);
    }
    builder.append(true)?;
    Ok(())
}

/// Appends a `map<int, binary>` value (lower/upper bounds), keys sorted; values are the raw serialized
/// single-value bytes (Java map<int, binary>).
fn append_bound_map(
    builder: &mut MapBuilder<Int32Builder, LargeBinaryBuilder>,
    map: &HashMap<i32, Datum>,
) -> Result<()> {
    let mut keys: Vec<&i32> = map.keys().collect();
    keys.sort_unstable();
    for key in keys {
        builder.keys().append_value(*key);
        builder.values().append_value(map[key].to_bytes()?);
    }
    builder.append(true)?;
    Ok(())
}

/// Appends an optional `list<long>` value (split offsets).
fn append_i64_list(builder: &mut ListBuilder<Int64Builder>, values: Option<&[i64]>) {
    match values {
        Some(values) => {
            for value in values {
                builder.values().append_value(*value);
            }
            builder.append(true);
        }
        None => builder.append(false),
    }
}

/// Appends an optional `list<int>` value (equality ids).
fn append_i32_list(builder: &mut ListBuilder<Int32Builder>, values: Option<&[i32]>) {
    match values {
        Some(values) => {
            for value in values {
                builder.values().append_value(*value);
            }
            builder.append(true);
        }
        None => builder.append(false),
    }
}

/// Appends one partition tuple to the partition [`StructBuilder`], dispatching each field on its
/// primitive type. The partition `Struct`'s values are aligned with `partition_type`'s fields.
fn append_partition(
    builder: &mut StructBuilder,
    partition_type: &StructType,
    partition: &crate::spec::Struct,
) -> Result<()> {
    for (index, field) in partition_type.fields().iter().enumerate() {
        let primitive_type = field.field_type.as_primitive_type().ok_or_else(|| {
            Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "partition field '{}' has non-primitive type {:?}; not supported in the files metadata table",
                    field.name, field.field_type
                ),
            )
        })?;
        let value = partition
            .fields()
            .get(index)
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
        PrimitiveType::TimestampNs => {
            append_typed!(TimestampNanosecondBuilder, extract_i64)
        }
        PrimitiveType::String => append_typed!(StringBuilder, extract_string),
        PrimitiveType::Binary => append_typed!(BinaryBuilder, extract_binary),
        PrimitiveType::Decimal { .. } => append_typed!(Decimal128Builder, extract_i128),
        other => {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "partition field type {other:?} is not supported in the files metadata table"
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
    use std::collections::HashMap;

    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use futures::TryStreamExt;

    use crate::scan::tests::TableTestFixture;
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Datum, Literal, ManifestContentType,
        ManifestEntry, ManifestListWriter, ManifestStatus, ManifestWriterBuilder, Struct,
    };

    /// A known, fixed file size used for every file in the fixtures (the metadata table reads only the
    /// manifest metadata, so no real parquet data file is needed).
    const FILE_SIZE: u64 = 1024;

    /// Builds the current snapshot's manifest list with one DATA manifest (3 data files:
    /// Added/Deleted/Existing across partitions 100/200/300) AND one DELETE manifest (1 Added
    /// position-delete file in partition 100). Returns nothing — the fixture's current snapshot is wired.
    ///
    /// This drives only public crate APIs (`ManifestWriterBuilder`, `ManifestListWriter`, the fixture's
    /// public `table`/`table_location`), so it does not depend on the scan fixture's private helpers.
    async fn setup_data_and_delete_manifests(fixture: &TableTestFixture) {
        let metadata = fixture.table.metadata().clone();
        let current_snapshot = metadata.current_snapshot().unwrap();
        let parent_snapshot = current_snapshot.parent_snapshot(&metadata).unwrap();
        let current_schema = current_snapshot.schema(&metadata).unwrap();
        let current_partition_spec = metadata.default_partition_spec();

        let manifest_output = |fixture: &TableTestFixture| {
            fixture
                .table
                .file_io()
                .new_output(format!(
                    "{}/metadata/manifest_{}.avro",
                    fixture.table_location,
                    uuid::Uuid::new_v4()
                ))
                .unwrap()
        };

        // DATA manifest.
        let mut data_writer = ManifestWriterBuilder::new(
            manifest_output(fixture),
            Some(current_snapshot.snapshot_id()),
            None,
            current_schema.clone(),
            current_partition_spec.as_ref().clone(),
        )
        .build_v2_data();
        data_writer
            .add_entry(
                ManifestEntry::builder()
                    .status(ManifestStatus::Added)
                    .data_file(
                        DataFileBuilder::default()
                            .partition_spec_id(0)
                            .content(DataContentType::Data)
                            .file_path(format!("{}/1.parquet", &fixture.table_location))
                            .file_format(DataFileFormat::Parquet)
                            .file_size_in_bytes(FILE_SIZE)
                            .record_count(1)
                            .partition(Struct::from_iter([Some(Literal::long(100))]))
                            .column_sizes(HashMap::from([(1, 42u64)]))
                            .lower_bounds(HashMap::from([(1, Datum::long(1))]))
                            .key_metadata(None)
                            .build()
                            .unwrap(),
                    )
                    .build(),
            )
            .unwrap();
        data_writer
            .add_delete_entry(
                ManifestEntry::builder()
                    .status(ManifestStatus::Deleted)
                    .snapshot_id(parent_snapshot.snapshot_id())
                    .sequence_number(parent_snapshot.sequence_number())
                    .file_sequence_number(parent_snapshot.sequence_number())
                    .data_file(
                        DataFileBuilder::default()
                            .partition_spec_id(0)
                            .content(DataContentType::Data)
                            .file_path(format!("{}/2.parquet", &fixture.table_location))
                            .file_format(DataFileFormat::Parquet)
                            .file_size_in_bytes(FILE_SIZE)
                            .record_count(1)
                            .partition(Struct::from_iter([Some(Literal::long(200))]))
                            .build()
                            .unwrap(),
                    )
                    .build(),
            )
            .unwrap();
        data_writer
            .add_existing_entry(
                ManifestEntry::builder()
                    .status(ManifestStatus::Existing)
                    .snapshot_id(parent_snapshot.snapshot_id())
                    .sequence_number(parent_snapshot.sequence_number())
                    .file_sequence_number(parent_snapshot.sequence_number())
                    .data_file(
                        DataFileBuilder::default()
                            .partition_spec_id(0)
                            .content(DataContentType::Data)
                            .file_path(format!("{}/3.parquet", &fixture.table_location))
                            .file_format(DataFileFormat::Parquet)
                            .file_size_in_bytes(FILE_SIZE)
                            .record_count(1)
                            .partition(Struct::from_iter([Some(Literal::long(300))]))
                            .build()
                            .unwrap(),
                    )
                    .build(),
            )
            .unwrap();
        let data_manifest = data_writer.write_manifest_file().await.unwrap();

        // DELETE manifest: one Added position-delete file in partition 100.
        let mut delete_writer = ManifestWriterBuilder::new(
            manifest_output(fixture),
            Some(current_snapshot.snapshot_id()),
            None,
            current_schema.clone(),
            current_partition_spec.as_ref().clone(),
        )
        .build_v2_deletes();
        delete_writer
            .add_entry(
                ManifestEntry::builder()
                    .status(ManifestStatus::Added)
                    .data_file(
                        DataFileBuilder::default()
                            .partition_spec_id(0)
                            .content(DataContentType::PositionDeletes)
                            .file_path(format!("{}/delete-1.parquet", &fixture.table_location))
                            .file_format(DataFileFormat::Parquet)
                            .file_size_in_bytes(FILE_SIZE)
                            .record_count(1)
                            .partition(Struct::from_iter([Some(Literal::long(100))]))
                            .build()
                            .unwrap(),
                    )
                    .build(),
            )
            .unwrap();
        let delete_manifest = delete_writer.write_manifest_file().await.unwrap();

        let mut manifest_list_write = ManifestListWriter::v2(
            fixture
                .table
                .file_io()
                .new_output(current_snapshot.manifest_list())
                .unwrap(),
            current_snapshot.snapshot_id(),
            current_snapshot.parent_snapshot_id(),
            current_snapshot.sequence_number(),
        );
        manifest_list_write
            .add_manifests(vec![data_manifest, delete_manifest].into_iter())
            .unwrap();
        manifest_list_write.close().await.unwrap();

        // Sanity: the manifest list now carries exactly one DATA and one DELETE manifest.
        let manifest_list = current_snapshot
            .load_manifest_list(fixture.table.file_io(), &metadata)
            .await
            .unwrap();
        let contents: Vec<ManifestContentType> =
            manifest_list.entries().iter().map(|m| m.content).collect();
        assert!(contents.contains(&ManifestContentType::Data));
        assert!(contents.contains(&ManifestContentType::Deletes));
    }

    /// Collects the sorted `file_path` set of a files-table scan.
    async fn scan_paths(stream: crate::scan::ArrowRecordBatchStream) -> Vec<String> {
        let batches: Vec<_> = stream.try_collect().await.unwrap();
        let mut paths = Vec::new();
        for batch in &batches {
            let column = batch
                .column_by_name("file_path")
                .unwrap()
                .as_string::<i32>();
            for index in 0..column.len() {
                paths.push(column.value(index).to_string());
            }
        }
        paths.sort();
        paths
    }

    /// Concatenates a files-table scan into a single batch.
    async fn scan_single_batch(
        stream: crate::scan::ArrowRecordBatchStream,
    ) -> arrow_array::RecordBatch {
        let batches: Vec<_> = stream.try_collect().await.unwrap();
        arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap()
    }

    #[tokio::test]
    async fn test_files_table_lists_live_data_and_delete_files() {
        // RISK: wrong file set — `files` must list every LIVE data + delete file (Added/Existing),
        // never the Deleted tombstone (2.parquet).
        let fixture = TableTestFixture::new();
        setup_data_and_delete_manifests(&fixture).await;

        let stream = fixture.table.inspect().files().scan().await.unwrap();
        let paths = scan_paths(stream).await;

        assert_eq!(paths, vec![
            format!("{}/1.parquet", fixture.table_location),
            format!("{}/3.parquet", fixture.table_location),
            format!("{}/delete-1.parquet", fixture.table_location),
        ]);
    }

    #[tokio::test]
    async fn test_data_files_table_excludes_delete_files() {
        // RISK: wrong content filter — `data_files` reads DATA manifests only, so the position-delete
        // file must NOT appear, and the Deleted 2.parquet stays excluded as a tombstone.
        let fixture = TableTestFixture::new();
        setup_data_and_delete_manifests(&fixture).await;

        let stream = fixture.table.inspect().data_files().scan().await.unwrap();
        let paths = scan_paths(stream).await;

        assert_eq!(paths, vec![
            format!("{}/1.parquet", fixture.table_location),
            format!("{}/3.parquet", fixture.table_location),
        ]);
    }

    #[tokio::test]
    async fn test_delete_files_table_lists_only_delete_files() {
        // RISK: wrong content filter — `delete_files` reads DELETE manifests only; exactly the one
        // position-delete file, none of the data files.
        let fixture = TableTestFixture::new();
        setup_data_and_delete_manifests(&fixture).await;

        let stream = fixture.table.inspect().delete_files().scan().await.unwrap();
        let paths = scan_paths(stream).await;

        assert_eq!(paths, vec![format!(
            "{}/delete-1.parquet",
            fixture.table_location
        )]);
    }

    #[tokio::test]
    async fn test_files_table_content_column_distinguishes_data_and_deletes() {
        // RISK: wrong `content` value — DATA files must report content 0, the position-delete file 1.
        let fixture = TableTestFixture::new();
        setup_data_and_delete_manifests(&fixture).await;

        let batch = scan_single_batch(fixture.table.inspect().files().scan().await.unwrap()).await;

        let paths = batch
            .column_by_name("file_path")
            .unwrap()
            .as_string::<i32>();
        let content = batch
            .column_by_name("content")
            .unwrap()
            .as_primitive::<arrow_array::types::Int32Type>();
        let mut content_by_suffix = HashMap::new();
        for index in 0..paths.len() {
            let suffix = paths.value(index).rsplit('/').next().unwrap().to_string();
            content_by_suffix.insert(suffix, content.value(index));
        }
        assert_eq!(content_by_suffix["1.parquet"], 0);
        assert_eq!(content_by_suffix["3.parquet"], 0);
        assert_eq!(content_by_suffix["delete-1.parquet"], 1);
    }

    #[tokio::test]
    async fn test_files_table_record_count_and_size_match_committed_metadata() {
        // RISK: wrong column mapping — record_count / file_size_in_bytes must reflect the committed
        // DataFile values (record_count == 1; file_size == FILE_SIZE).
        let fixture = TableTestFixture::new();
        setup_data_and_delete_manifests(&fixture).await;

        let batch = scan_single_batch(fixture.table.inspect().files().scan().await.unwrap()).await;

        let record_count = batch
            .column_by_name("record_count")
            .unwrap()
            .as_primitive::<arrow_array::types::Int64Type>();
        let file_size = batch
            .column_by_name("file_size_in_bytes")
            .unwrap()
            .as_primitive::<arrow_array::types::Int64Type>();
        assert_eq!(record_count.len(), 3);
        for index in 0..record_count.len() {
            assert_eq!(record_count.value(index), 1);
            assert_eq!(file_size.value(index), FILE_SIZE as i64);
        }
    }

    #[tokio::test]
    async fn test_files_table_partition_struct_and_metrics_map_present() {
        // RISK: wrong column — the partition column must be the partition struct (long `x`), and the
        // metrics maps must be populated for the Added file (column_sizes {1: 42}).
        let fixture = TableTestFixture::new();
        setup_data_and_delete_manifests(&fixture).await;

        let batch = scan_single_batch(fixture.table.inspect().files().scan().await.unwrap()).await;

        let partition = batch.column_by_name("partition").unwrap().as_struct();
        assert_eq!(partition.num_columns(), 1);
        let partition_values = partition
            .column(0)
            .as_primitive::<arrow_array::types::Int64Type>();
        let mut partitions: Vec<i64> = (0..partition_values.len())
            .map(|index| partition_values.value(index))
            .collect();
        partitions.sort();
        assert_eq!(partitions, vec![100, 100, 300]);

        let column_sizes = batch.column_by_name("column_sizes").unwrap().as_map();
        let mut found_added_metrics = false;
        for index in 0..column_sizes.len() {
            let entries = column_sizes.value(index);
            let keys = entries
                .column(0)
                .as_primitive::<arrow_array::types::Int32Type>();
            let values = entries
                .column(1)
                .as_primitive::<arrow_array::types::Int64Type>();
            if keys.len() == 1 && keys.value(0) == 1 && values.value(0) == 42 {
                found_added_metrics = true;
            }
        }
        assert!(
            found_added_metrics,
            "expected column_sizes {{1: 42}} on the Added file"
        );
    }

    #[tokio::test]
    async fn test_files_table_arrow_schema_columns_and_types() {
        // RISK: wrong column set / type — assert the Arrow schema is the DataFile column set with the
        // expected leading types (content Int32, file_path Utf8, partition Struct, the metrics Maps).
        let fixture = TableTestFixture::new();
        let schema = fixture.table.inspect().files().schema();
        let arrow = crate::arrow::schema_to_arrow_schema(&schema).unwrap();

        let names: Vec<&str> = arrow.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec![
            "content",
            "file_path",
            "file_format",
            "spec_id",
            "partition",
            "record_count",
            "file_size_in_bytes",
            "column_sizes",
            "value_counts",
            "null_value_counts",
            "nan_value_counts",
            "lower_bounds",
            "upper_bounds",
            "key_metadata",
            "split_offsets",
            "equality_ids",
            "sort_order_id",
            "first_row_id",
            "referenced_data_file",
            "content_offset",
            "content_size_in_bytes",
        ]);

        use arrow_schema::DataType;
        assert_eq!(
            arrow.field_with_name("content").unwrap().data_type(),
            &DataType::Int32
        );
        assert_eq!(
            arrow.field_with_name("file_path").unwrap().data_type(),
            &DataType::Utf8
        );
        assert_eq!(
            arrow.field_with_name("record_count").unwrap().data_type(),
            &DataType::Int64
        );
        assert!(matches!(
            arrow.field_with_name("partition").unwrap().data_type(),
            DataType::Struct(_)
        ));
        assert!(matches!(
            arrow.field_with_name("column_sizes").unwrap().data_type(),
            DataType::Map(_, _)
        ));
        assert!(matches!(
            arrow.field_with_name("lower_bounds").unwrap().data_type(),
            DataType::Map(_, _)
        ));
    }

    #[tokio::test]
    async fn test_files_table_empty_table_yields_empty_batch() {
        // RISK: panic / non-empty on an empty table — no current snapshot must yield zero rows.
        let fixture = TableTestFixture::new_empty();
        let batches: Vec<_> = fixture
            .table
            .inspect()
            .files()
            .scan()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0);
    }

    #[tokio::test]
    async fn test_files_table_unpartitioned_keeps_empty_partition_struct_known_divergence() {
        // RISK / KNOWN DIVERGENCE from Java: for an UNPARTITIONED table Java `BaseFilesTable.schema()`
        // DROPS the `partition` field entirely ("avoid returning an empty struct, which is not always
        // supported. instead, drop the partition field" — `TypeUtil.selectNot(schema, PARTITION_ID)`).
        // The Rust port currently KEEPS a `partition` column typed as an empty struct (`Struct([])`).
        // This is non-corrupting (the file rows + every other column are correct, the row count is
        // right) but is a schema-shape divergence that matters for eventual Java interop — tracked in
        // GAP_MATRIX/todo as a deferral, NOT silently wrong. This test PINS the current behavior so the
        // divergence cannot change unnoticed; when the Java drop-empty-partition rule is implemented,
        // this test flips to assert the `partition` column is ABSENT.
        let fixture = TableTestFixture::new_unpartitioned();
        let metadata = fixture.table.metadata().clone();
        let current_snapshot = metadata.current_snapshot().unwrap();
        let current_schema = current_snapshot.schema(&metadata).unwrap();
        let current_partition_spec = metadata.default_partition_spec();

        let output = fixture
            .table
            .file_io()
            .new_output(format!(
                "{}/metadata/manifest_unp_{}.avro",
                fixture.table_location,
                uuid::Uuid::new_v4()
            ))
            .unwrap();
        let mut data_writer = ManifestWriterBuilder::new(
            output,
            Some(current_snapshot.snapshot_id()),
            None,
            current_schema.clone(),
            current_partition_spec.as_ref().clone(),
        )
        .build_v2_data();
        data_writer
            .add_entry(
                ManifestEntry::builder()
                    .status(ManifestStatus::Added)
                    .data_file(
                        DataFileBuilder::default()
                            .partition_spec_id(0)
                            .content(DataContentType::Data)
                            .file_path(format!("{}/u1.parquet", &fixture.table_location))
                            .file_format(DataFileFormat::Parquet)
                            .file_size_in_bytes(FILE_SIZE)
                            .record_count(1)
                            .partition(Struct::empty())
                            .build()
                            .unwrap(),
                    )
                    .build(),
            )
            .unwrap();
        let data_manifest = data_writer.write_manifest_file().await.unwrap();

        let mut manifest_list_write = ManifestListWriter::v2(
            fixture
                .table
                .file_io()
                .new_output(current_snapshot.manifest_list())
                .unwrap(),
            current_snapshot.snapshot_id(),
            current_snapshot.parent_snapshot_id(),
            current_snapshot.sequence_number(),
        );
        manifest_list_write
            .add_manifests(vec![data_manifest].into_iter())
            .unwrap();
        manifest_list_write.close().await.unwrap();

        let batch = scan_single_batch(fixture.table.inspect().files().scan().await.unwrap()).await;

        // Does not panic; the single data file is listed.
        assert_eq!(batch.num_rows(), 1);
        // CURRENT (divergent) behavior: the partition column is present as an empty struct.
        let partition = batch.column_by_name("partition").unwrap().as_struct();
        assert_eq!(
            partition.num_columns(),
            0,
            "unpartitioned files table currently keeps an empty-struct partition column \
             (Java drops it) — see the GAP_MATRIX deferral"
        );
    }
}
