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

//! Partition value calculation for Iceberg tables.
//!
//! This module provides utilities for calculating partition values from record batches
//! based on a partition specification.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, LargeListArray, ListArray,
    MapArray, RecordBatch, RecordBatchOptions, StructArray,
};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, SchemaRef};

use super::record_batch_projector::RecordBatchProjector;
use super::type_to_arrow_type;
use crate::arrow::schema::schema_to_arrow_schema;
use crate::spec::{PartitionSpec, Schema, StructType, Type};
use crate::transform::{BoxedTransformFunction, create_transform_function};
use crate::{Error, ErrorKind, Result};

const CANONICAL_NAN_F64: u64 = 0x7ff8_0000_0000_0000;
const CANONICAL_NAN_F32: u32 = 0x7fc0_0000;
const NAN_CHUNK: usize = 1024;

fn holds_other_nan_f64(values: &[f64]) -> bool {
    values.chunks(NAN_CHUNK).any(|chunk| {
        chunk.iter().fold(false, |seen, value| {
            let bits = value.to_bits();
            seen | ((bits & 0x7fff_ffff_ffff_ffff) > 0x7ff0_0000_0000_0000
                && bits != CANONICAL_NAN_F64)
        })
    })
}

fn holds_other_nan_f32(values: &[f32]) -> bool {
    values.chunks(NAN_CHUNK).any(|chunk| {
        chunk.iter().fold(false, |seen, value| {
            let bits = value.to_bits();
            seen | ((bits & 0x7fff_ffff) > 0x7f80_0000 && bits != CANONICAL_NAN_F32)
        })
    })
}

fn canonical_f64(array: &Float64Array) -> Option<ArrayRef> {
    if !holds_other_nan_f64(array.values()) {
        return None;
    }
    let canonical = f64::from_bits(CANONICAL_NAN_F64);
    let values: ScalarBuffer<f64> = array
        .values()
        .iter()
        .map(|value| if value.is_nan() { canonical } else { *value })
        .collect();
    Some(Arc::new(Float64Array::new(values, array.nulls().cloned())))
}

fn canonical_f32(array: &Float32Array) -> Option<ArrayRef> {
    if !holds_other_nan_f32(array.values()) {
        return None;
    }
    let canonical = f32::from_bits(CANONICAL_NAN_F32);
    let values: ScalarBuffer<f32> = array
        .values()
        .iter()
        .map(|value| if value.is_nan() { canonical } else { *value })
        .collect();
    Some(Arc::new(Float32Array::new(values, array.nulls().cloned())))
}

fn canonical_struct(array: &StructArray) -> Result<Option<StructArray>> {
    let mut changed = false;
    let mut columns = Vec::with_capacity(array.num_columns());
    for column in array.columns() {
        match canonical_nan_change(column)? {
            Some(rebuilt) => {
                changed = true;
                columns.push(rebuilt);
            }
            None => columns.push(Arc::clone(column)),
        }
    }
    if !changed {
        return Ok(None);
    }
    StructArray::try_new_with_length(
        array.fields().clone(),
        columns,
        array.nulls().cloned(),
        array.len(),
    )
    .map(Some)
    .map_err(canonical_error)
}

fn canonical_nan_change(array: &ArrayRef) -> Result<Option<ArrayRef>> {
    match array.data_type() {
        DataType::Float64 => Ok(canonical_f64(array.as_primitive::<Float64Type>())),
        DataType::Float32 => Ok(canonical_f32(array.as_primitive::<Float32Type>())),
        DataType::Struct(_) => {
            Ok(canonical_struct(array.as_struct())?.map(|rebuilt| Arc::new(rebuilt) as ArrayRef))
        }
        DataType::List(field) => {
            let list = array.as_list::<i32>();
            let Some(values) = canonical_nan_change(list.values())? else {
                return Ok(None);
            };
            ListArray::try_new(
                Arc::clone(field),
                list.offsets().clone(),
                values,
                list.nulls().cloned(),
            )
            .map(|rebuilt| Some(Arc::new(rebuilt) as ArrayRef))
            .map_err(canonical_error)
        }
        DataType::LargeList(field) => {
            let list = array.as_list::<i64>();
            let Some(values) = canonical_nan_change(list.values())? else {
                return Ok(None);
            };
            LargeListArray::try_new(
                Arc::clone(field),
                list.offsets().clone(),
                values,
                list.nulls().cloned(),
            )
            .map(|rebuilt| Some(Arc::new(rebuilt) as ArrayRef))
            .map_err(canonical_error)
        }
        DataType::Map(field, ordered) => {
            let map = array.as_map();
            let entries = map.entries();
            let Some(values) = canonical_nan_change(entries.column(1))? else {
                return Ok(None);
            };
            let entries = StructArray::try_new_with_length(
                entries.fields().clone(),
                vec![Arc::clone(entries.column(0)), values],
                entries.nulls().cloned(),
                entries.len(),
            )
            .map_err(canonical_error)?;
            MapArray::try_new(
                Arc::clone(field),
                map.offsets().clone(),
                entries,
                map.nulls().cloned(),
                *ordered,
            )
            .map(|rebuilt| Some(Arc::new(rebuilt) as ArrayRef))
            .map_err(canonical_error)
        }
        DataType::FixedSizeList(field, size) => {
            let list = array.as_fixed_size_list();
            let Some(values) = canonical_nan_change(list.values())? else {
                return Ok(None);
            };
            FixedSizeListArray::try_new(Arc::clone(field), *size, values, list.nulls().cloned())
                .map(|rebuilt| Some(Arc::new(rebuilt) as ArrayRef))
                .map_err(canonical_error)
        }
        DataType::Dictionary(_, _) => {
            let dictionary = array.as_any_dictionary();
            let Some(values) = canonical_nan_change(dictionary.values())? else {
                return Ok(None);
            };
            Ok(Some(dictionary.with_values(values)))
        }
        _ => Ok(None),
    }
}

fn canonical_error(error: arrow_schema::ArrowError) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        format!("Failed to rebuild an array with canonical NaN values: {error}"),
    )
}

pub(crate) fn canonical_nan_array(array: &ArrayRef) -> Result<ArrayRef> {
    Ok(canonical_nan_change(array)?.unwrap_or_else(|| Arc::clone(array)))
}

pub(crate) fn canonical_nan_batch(batch: &RecordBatch) -> Result<Option<RecordBatch>> {
    let mut changed = false;
    let mut columns = Vec::with_capacity(batch.num_columns());
    for column in batch.columns() {
        match canonical_nan_change(column)? {
            Some(rebuilt) => {
                changed = true;
                columns.push(rebuilt);
            }
            None => columns.push(Arc::clone(column)),
        }
    }
    if !changed {
        return Ok(None);
    }
    RecordBatch::try_new_with_options(
        batch.schema(),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )
    .map(Some)
    .map_err(canonical_error)
}

/// Calculator for partition values in Iceberg tables.
///
/// This struct handles the projection of source columns and application of
/// partition transforms to compute partition values for a given record batch.
#[derive(Debug)]
pub struct PartitionValueCalculator {
    projector: RecordBatchProjector,
    transform_functions: Vec<BoxedTransformFunction>,
    partition_type: StructType,
    partition_arrow_type: DataType,
    /// The Arrow rendering of the table schema the projector's index paths were
    /// derived from. [`Self::calculate`] validates incoming batches against it:
    /// the projector reads columns POSITIONALLY, so a batch whose top-level
    /// layout differs from this schema would silently yield wrong partition
    /// values.
    source_arrow_schema: SchemaRef,
}

impl PartitionValueCalculator {
    /// Create a new PartitionValueCalculator.
    ///
    /// # Arguments
    ///
    /// * `partition_spec` - The partition specification
    /// * `table_schema` - The Iceberg table schema
    ///
    /// # Returns
    ///
    /// Returns a new `PartitionValueCalculator` instance or an error if initialization fails.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The partition spec is unpartitioned
    /// - Transform function creation fails
    /// - Projector initialization fails
    pub fn try_new(partition_spec: &PartitionSpec, table_schema: &Schema) -> Result<Self> {
        if partition_spec.is_unpartitioned() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Cannot create partition calculator for unpartitioned table",
            ));
        }

        // Create transform functions for each partition field
        let transform_functions: Vec<BoxedTransformFunction> = partition_spec
            .fields()
            .iter()
            .map(|pf| create_transform_function(&pf.transform))
            .collect::<Result<Vec<_>>>()?;

        // Extract source field IDs for projection
        let source_field_ids: Vec<i32> = partition_spec
            .fields()
            .iter()
            .map(|pf| pf.source_id)
            .collect();

        // Create projector for extracting source columns
        let projector = RecordBatchProjector::from_iceberg_schema(
            Arc::new(table_schema.clone()),
            &source_field_ids,
        )?;

        // Get partition type information
        let partition_type = partition_spec.partition_type(table_schema)?;
        let partition_arrow_type = type_to_arrow_type(&Type::Struct(partition_type.clone()))?;

        // The schema the projector's positional index paths are valid against.
        let source_arrow_schema = Arc::new(schema_to_arrow_schema(table_schema)?);

        Ok(Self {
            projector,
            transform_functions,
            partition_type,
            partition_arrow_type,
            source_arrow_schema,
        })
    }

    /// Get the partition type as an Iceberg StructType.
    pub fn partition_type(&self) -> &StructType {
        &self.partition_type
    }

    /// Get the partition type as an Arrow DataType.
    pub fn partition_arrow_type(&self) -> &DataType {
        &self.partition_arrow_type
    }

    /// Calculate partition values for a record batch.
    ///
    /// This method:
    /// 1. Validates the batch's top-level layout against the table schema the
    ///    calculator was built from (the projector reads columns positionally)
    /// 2. Projects the source columns from the batch
    /// 3. Applies partition transforms to each source column
    /// 4. Constructs a StructArray containing the partition values
    ///
    /// # Arguments
    ///
    /// * `batch` - The record batch to calculate partition values for. Its
    ///   top-level columns must be the table's columns, in table-schema order.
    ///
    /// # Returns
    ///
    /// Returns an ArrayRef containing a StructArray of partition values, or an error if calculation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The batch's width or top-level column names do not match the table
    ///   schema the calculator was built from
    /// - Column projection fails
    /// - Transform application fails
    /// - StructArray construction fails
    pub fn calculate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        // The projector's index paths are positional: a batch with the right
        // arity but the wrong columns would not fail below — it would produce
        // wrong-but-well-formed partition values. Validate NAME + WIDTH here
        // (types are deliberately not compared: legitimate callers may feed
        // representationally different but logically identical arrays, e.g.
        // dictionary-encoded columns; value types are enforced downstream by
        // the transforms and the struct construction).
        let batch_schema = batch.schema();
        let expected_fields = self.source_arrow_schema.fields();
        if batch_schema.fields().len() != expected_fields.len() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Partition source batch width does not match the table schema \
                 the partition calculator was built from",
            )
            .with_context("batch_columns", batch_schema.fields().len().to_string())
            .with_context("expected_columns", expected_fields.len().to_string()));
        }
        for (position, (batch_field, expected_field)) in batch_schema
            .fields()
            .iter()
            .zip(expected_fields.iter())
            .enumerate()
        {
            if batch_field.name() != expected_field.name() {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Partition source batch column name does not match the table \
                     schema the partition calculator was built from",
                )
                .with_context("position", position.to_string())
                .with_context("batch_column", batch_field.name().clone())
                .with_context("expected_column", expected_field.name().clone()));
            }
        }

        self.calculate_from_columns(batch.columns(), batch.num_rows())
    }

    /// Calculate partition values from raw top-level columns.
    ///
    /// The columns must be the table's top-level columns in table-schema order
    /// (the same layout [`Self::calculate`] expects as a `RecordBatch`). This
    /// entry point exists for callers that already hold per-column arrays —
    /// e.g. an engine evaluating one expression per table column — and cannot
    /// or should not reassemble a `RecordBatch` first.
    ///
    /// # Arguments
    ///
    /// * `columns` - The table's top-level columns, in table-schema order
    /// * `num_rows` - The row count every column must have
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `columns` does not have exactly one entry per table column
    /// - Any column's length differs from `num_rows`
    /// - Column projection fails
    /// - Transform application fails
    /// - StructArray construction fails
    pub fn calculate_from_columns(
        &self,
        columns: &[ArrayRef],
        num_rows: usize,
    ) -> Result<ArrayRef> {
        let expected_width = self.source_arrow_schema.fields().len();
        if columns.len() != expected_width {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Partition source column count does not match the table schema \
                 the partition calculator was built from",
            )
            .with_context("columns", columns.len().to_string())
            .with_context("expected_columns", expected_width.to_string()));
        }
        for (position, column) in columns.iter().enumerate() {
            if column.len() != num_rows {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Partition source column length does not match the row count",
                )
                .with_context("position", position.to_string())
                .with_context("column_length", column.len().to_string())
                .with_context("num_rows", num_rows.to_string()));
            }
        }

        // Project source columns from the batch
        let source_columns = self.projector.project_column(columns)?;

        // Get expected struct fields for the result
        let expected_struct_fields = match &self.partition_arrow_type {
            DataType::Struct(fields) => fields.clone(),
            _ => {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Expected partition type must be a struct",
                ));
            }
        };

        // Apply transforms to each source column
        let mut partition_values = Vec::with_capacity(self.transform_functions.len());
        for ((source_column, transform_fn), expected_field) in source_columns
            .iter()
            .zip(&self.transform_functions)
            .zip(expected_struct_fields.iter())
        {
            let partition_value = match transform_fn
                .transform_to_type(source_column, expected_field.data_type())
            {
                Some(result) => result?,
                None => {
                    let partition_value = transform_fn.transform(source_column.clone())?;
                    if partition_value.data_type() == expected_field.data_type() {
                        partition_value
                    } else {
                        arrow_cast::cast(partition_value.as_ref(), expected_field.data_type())
                            .map_err(|e| {
                                Error::new(
                                    ErrorKind::DataInvalid,
                                    format!(
                                        "Failed to cast partition value to the partition field type: {e}"
                                    ),
                                )
                            })?
                    }
                }
            };
            partition_values.push(canonical_nan_array(&partition_value)?);
        }

        // Construct the StructArray
        let struct_array = StructArray::try_new(expected_struct_fields, partition_values, None)
            .map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Failed to create partition struct array: {e}"),
                )
            })?;

        Ok(Arc::new(struct_array))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::types::Int32Type;
    use arrow_array::{DictionaryArray, Int32Array, RecordBatch, StringArray, UInt32Array};
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::{Field, Fields, Schema as ArrowSchema};

    use super::*;
    use crate::arrow::FieldMatchMode;
    use crate::spec::{
        DataFileFormat, NestedField, PartitionSpecBuilder, PrimitiveType, Transform,
    };

    const NAN_F64: [u64; 8] = [
        0x7ff8_0000_0000_0000,
        0xfff8_0000_0000_0000,
        0x7ff8_dead_beef_0001,
        0xfff8_dead_beef_0001,
        0x7ff0_0000_0000_0001,
        0xfff0_0000_0000_0001,
        0x7fff_ffff_ffff_ffff,
        0xffff_ffff_ffff_ffff,
    ];
    const NAN_F32: [u32; 8] = [
        0x7fc0_0000,
        0xffc0_0000,
        0x7fc0_0123,
        0xffc0_0123,
        0x7f80_0001,
        0xff80_0001,
        0x7fff_ffff,
        0xffff_ffff,
    ];
    const OTHER_F64: [u64; 7] = [
        0x8000_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x3ff8_0000_0000_0000,
        0x8000_0000_0000_0001,
        0xffef_ffff_ffff_ffff,
    ];
    const OTHER_F32: [u32; 7] = [
        0x8000_0000,
        0x0000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x3fc0_0000,
        0x8000_0001,
        0xff7f_ffff,
    ];

    fn nan_doubles() -> Float64Array {
        NAN_F64
            .iter()
            .chain(OTHER_F64.iter())
            .map(|bits| Some(f64::from_bits(*bits)))
            .chain(std::iter::once(None))
            .collect()
    }

    fn nan_floats() -> Float32Array {
        NAN_F32
            .iter()
            .chain(OTHER_F32.iter())
            .map(|bits| Some(f32::from_bits(*bits)))
            .chain(std::iter::once(None))
            .collect()
    }

    fn canonical_f64_bits() -> Vec<Option<u64>> {
        NAN_F64
            .iter()
            .map(|_| Some(CANONICAL_NAN_F64))
            .chain(OTHER_F64.iter().map(|bits| Some(*bits)))
            .chain(std::iter::once(None))
            .collect()
    }

    fn canonical_f32_bits() -> Vec<Option<u32>> {
        NAN_F32
            .iter()
            .map(|_| Some(CANONICAL_NAN_F32))
            .chain(OTHER_F32.iter().map(|bits| Some(*bits)))
            .chain(std::iter::once(None))
            .collect()
    }

    fn f64_bits(array: &dyn Array) -> Vec<Option<u64>> {
        array
            .as_primitive::<Float64Type>()
            .iter()
            .map(|value| value.map(f64::to_bits))
            .collect()
    }

    fn f32_bits(array: &dyn Array) -> Vec<Option<u32>> {
        array
            .as_primitive::<Float32Type>()
            .iter()
            .map(|value| value.map(f32::to_bits))
            .collect()
    }

    fn nan_table_schema() -> Arc<Schema> {
        Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::optional(1, "d", Type::Primitive(PrimitiveType::Double)).into(),
                    NestedField::optional(2, "f", Type::Primitive(PrimitiveType::Float)).into(),
                ])
                .build()
                .unwrap(),
        )
    }

    fn nan_batch() -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(schema_to_arrow_schema(&nan_table_schema()).unwrap()),
            vec![Arc::new(nan_doubles()), Arc::new(nan_floats())],
        )
        .unwrap()
    }

    #[test]
    fn canonical_nan_batch_rewrites_every_nan_and_keeps_every_other_bit() {
        let batch = nan_batch();
        let canonical = canonical_nan_batch(&batch)
            .unwrap()
            .expect("a NaN to rewrite");
        assert_eq!(canonical.schema(), batch.schema());
        assert_eq!(f64_bits(canonical.column(0).as_ref()), canonical_f64_bits());
        assert_eq!(f32_bits(canonical.column(1).as_ref()), canonical_f32_bits());
        assert!(canonical_nan_batch(&canonical).unwrap().is_none());
        let rows: Vec<u32> = std::iter::once(0)
            .chain(NAN_F64.len() as u32..batch.num_rows() as u32)
            .collect();
        let others =
            arrow_select::take::take_record_batch(&batch, &UInt32Array::from(rows)).unwrap();
        assert!(canonical_nan_batch(&others).unwrap().is_none());
    }

    #[test]
    fn canonical_nan_array_reaches_nested_leaves_and_keeps_map_keys() {
        let doubles: ArrayRef = Arc::new(nan_doubles());
        let rows = doubles.len();
        let expected = canonical_f64_bits();
        let leaf = Arc::new(Field::new("a", DataType::Float64, true));
        let structs: ArrayRef = Arc::new(StructArray::from(vec![(leaf, Arc::clone(&doubles))]));
        let item = Arc::new(Field::new("item", DataType::Float64, true));
        let offsets = OffsetBuffer::from_lengths(std::iter::repeat_n(1, rows));
        let lists: ArrayRef = Arc::new(
            ListArray::try_new(
                Arc::clone(&item),
                offsets.clone(),
                Arc::clone(&doubles),
                None,
            )
            .unwrap(),
        );
        let fixed: ArrayRef = Arc::new(
            FixedSizeListArray::try_new(Arc::clone(&item), 1, Arc::clone(&doubles), None).unwrap(),
        );
        let keys_dictionary: ArrayRef = Arc::new(
            DictionaryArray::<Int32Type>::try_new(
                Int32Array::from_iter_values(0..rows as i32),
                Arc::clone(&doubles),
            )
            .unwrap(),
        );
        let entries_fields = Fields::from(vec![
            Field::new("key", DataType::Float64, false),
            Field::new("value", DataType::Float64, true),
        ]);
        let key_values: ArrayRef = Arc::new(Float64Array::from_iter_values(
            NAN_F64
                .iter()
                .chain(OTHER_F64.iter())
                .chain(std::iter::once(&0))
                .map(|bits| f64::from_bits(*bits)),
        ));
        let entries = StructArray::try_new(
            entries_fields.clone(),
            vec![Arc::clone(&key_values), Arc::clone(&doubles)],
            None,
        )
        .unwrap();
        let entries_field = Arc::new(Field::new(
            "entries",
            DataType::Struct(entries_fields),
            false,
        ));
        let maps: ArrayRef =
            Arc::new(MapArray::try_new(entries_field, offsets, entries, None, false).unwrap());

        let canonical = canonical_nan_array(&structs).unwrap();
        assert_eq!(f64_bits(canonical.as_struct().column(0).as_ref()), expected);
        let canonical = canonical_nan_array(&lists).unwrap();
        assert_eq!(
            f64_bits(canonical.as_list::<i32>().values().as_ref()),
            expected
        );
        let canonical = canonical_nan_array(&fixed).unwrap();
        assert_eq!(
            f64_bits(canonical.as_fixed_size_list().values().as_ref()),
            expected
        );
        let canonical = canonical_nan_array(&keys_dictionary).unwrap();
        assert_eq!(
            f64_bits(canonical.as_any_dictionary().values().as_ref()),
            expected
        );
        let canonical = canonical_nan_array(&maps).unwrap();
        assert_eq!(f64_bits(canonical.as_map().values().as_ref()), expected);
        assert_eq!(
            f64_bits(canonical.as_map().keys().as_ref()),
            f64_bits(key_values.as_ref())
        );
        assert_eq!(f64_bits(doubles.as_ref())[1], Some(0xfff8_0000_0000_0000));
    }

    async fn written_bits(format: DataFileFormat) -> (Vec<Option<u64>>, Vec<Option<u32>>, u64) {
        use crate::io::FileIO;
        use crate::spec::MetricsConfig;
        use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
        use crate::writer::file_writer::AnyFileWriterBuilder;
        use crate::writer::file_writer::location_generator::{
            DefaultFileNameGenerator, DefaultLocationGenerator,
        };
        use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
        use crate::writer::{IcebergWriter, IcebergWriterBuilder};

        let temp_dir = tempfile::TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let location = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let names = DefaultFileNameGenerator::new("nan".to_string(), None, format);
        let any = AnyFileWriterBuilder::for_format(
            format,
            nan_table_schema(),
            &std::collections::HashMap::new(),
            MetricsConfig::default(),
            FieldMatchMode::Id,
        )
        .unwrap();
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            any,
            file_io.clone(),
            location,
            names,
        );
        let mut writer = DataFileWriterBuilder::new(rolling)
            .unpartitioned()
            .build(None)
            .await
            .unwrap();
        writer.write(nan_batch()).await.unwrap();
        let files = writer.close().await.unwrap();
        assert_eq!(files.len(), 1);
        let nan_count = files[0].nan_value_counts.get(&1).copied().unwrap_or(0);
        let bytes = file_io
            .new_input(files[0].file_path.clone())
            .unwrap()
            .read()
            .await
            .unwrap();
        let decoded: Vec<RecordBatch> = match format {
            DataFileFormat::Parquet => {
                parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(bytes)
                    .unwrap()
                    .build()
                    .unwrap()
                    .map(|batch| batch.unwrap())
                    .collect()
            }
            DataFileFormat::Orc => orc_rust::ArrowReaderBuilder::try_new(bytes)
                .unwrap()
                .build()
                .map(|batch| batch.unwrap())
                .collect(),
            _ => {
                let mut doubles = Vec::new();
                let mut floats = Vec::new();
                for row in apache_avro::Reader::new(&bytes[..]).unwrap() {
                    let apache_avro::types::Value::Record(fields) = row.unwrap() else {
                        panic!("an Avro record");
                    };
                    for (name, value) in fields {
                        let value = match value {
                            apache_avro::types::Value::Union(_, inner) => *inner,
                            other => other,
                        };
                        match (name.as_str(), value) {
                            ("d", apache_avro::types::Value::Double(v)) => {
                                doubles.push(Some(v.to_bits()))
                            }
                            ("f", apache_avro::types::Value::Float(v)) => {
                                floats.push(Some(v.to_bits()))
                            }
                            ("d", _) => doubles.push(None),
                            _ => floats.push(None),
                        }
                    }
                }
                return (doubles, floats, nan_count);
            }
        };
        let batch = arrow_select::concat::concat_batches(&decoded[0].schema(), &decoded).unwrap();
        (
            f64_bits(batch.column(0).as_ref()),
            f32_bits(batch.column(1).as_ref()),
            nan_count,
        )
    }

    #[tokio::test]
    async fn parquet_and_orc_store_the_canonical_nan_and_avro_keeps_the_bits() {
        for format in [DataFileFormat::Parquet, DataFileFormat::Orc] {
            let (doubles, floats, nan_count) = written_bits(format).await;
            assert_eq!(nan_count, NAN_F64.len() as u64, "{format:?}");
            assert_eq!(doubles, canonical_f64_bits(), "{format:?}");
            assert_eq!(floats, canonical_f32_bits(), "{format:?}");
        }
        let (doubles, floats, _) = written_bits(DataFileFormat::Avro).await;
        assert_eq!(doubles, f64_bits(&nan_doubles()));
        assert_eq!(floats, f32_bits(&nan_floats()));
    }

    #[test]
    fn every_nan_pattern_is_one_canonical_partition_value() {
        let table_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::optional(1, "d", Type::Primitive(PrimitiveType::Double)).into(),
                    NestedField::optional(2, "f", Type::Primitive(PrimitiveType::Float)).into(),
                ])
                .build()
                .unwrap(),
        );
        let spec = Arc::new(
            PartitionSpecBuilder::new(Arc::clone(&table_schema))
                .add_partition_field("d", "d", Transform::Identity)
                .unwrap()
                .add_partition_field("f", "f", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );
        let doubles: [u64; 6] = [
            0x7ff8_0000_0000_0000,
            0xfff8_0000_0000_0000,
            0x7ff8_dead_beef_0001,
            0xfff0_0000_0000_0001,
            0x8000_0000_0000_0000,
            0x3ff8_0000_0000_0000,
        ];
        let floats: [u32; 6] = [
            0x7fc0_0000,
            0xffc0_0000,
            0x7fc0_0123,
            0xff80_0001,
            0x8000_0000,
            0x3fc0_0000,
        ];
        let batch = RecordBatch::try_new(
            Arc::new(schema_to_arrow_schema(&table_schema).unwrap()),
            vec![
                Arc::new(Float64Array::from_iter_values(
                    doubles.iter().map(|bits| f64::from_bits(*bits)),
                )),
                Arc::new(Float32Array::from_iter_values(
                    floats.iter().map(|bits| f32::from_bits(*bits)),
                )),
            ],
        )
        .unwrap();
        let calculator = PartitionValueCalculator::try_new(&spec, &table_schema).unwrap();
        let values = calculator.calculate(&batch).unwrap();
        let d: Vec<u64> = values
            .as_struct()
            .column(0)
            .as_primitive::<Float64Type>()
            .values()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let f: Vec<u32> = values
            .as_struct()
            .column(1)
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(d, vec![
            CANONICAL_NAN_F64,
            CANONICAL_NAN_F64,
            CANONICAL_NAN_F64,
            CANONICAL_NAN_F64,
            doubles[4],
            doubles[5]
        ]);
        assert_eq!(f, vec![
            CANONICAL_NAN_F32,
            CANONICAL_NAN_F32,
            CANONICAL_NAN_F32,
            CANONICAL_NAN_F32,
            floats[4],
            floats[5]
        ]);
        let splitter = crate::arrow::RecordBatchPartitionSplitter::try_new_with_computed_values(
            table_schema,
            spec,
        )
        .unwrap();
        let parts = splitter.split(&batch).unwrap();
        let nan_parts: Vec<_> = parts
            .iter()
            .filter_map(|(key, rows)| match key.data().fields().first() {
                Some(Some(crate::spec::Literal::Primitive(
                    crate::spec::PrimitiveLiteral::Double(value),
                ))) if value.is_nan() => Some((value.0.to_bits(), rows.num_rows())),
                _ => None,
            })
            .collect();
        assert_eq!(nan_parts, vec![(CANONICAL_NAN_F64, 4)]);
    }

    #[test]
    fn test_partition_calculator_identity_transform() {
        let table_schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();

        let partition_spec = PartitionSpecBuilder::new(Arc::new(table_schema.clone()))
            .add_partition_field("id", "id_partition", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();

        let calculator = PartitionValueCalculator::try_new(&partition_spec, &table_schema).unwrap();

        // Verify partition type
        assert_eq!(calculator.partition_type().fields().len(), 1);
        assert_eq!(calculator.partition_type().fields()[0].name, "id_partition");

        // Create test batch
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int32Array::from(vec![10, 20, 30])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ])
        .unwrap();

        // Calculate partition values
        let result = calculator.calculate(&batch).unwrap();
        let struct_array = result.as_any().downcast_ref::<StructArray>().unwrap();

        let id_partition = struct_array
            .column_by_name("id_partition")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

        assert_eq!(id_partition.value(0), 10);
        assert_eq!(id_partition.value(1), 20);
        assert_eq!(id_partition.value(2), 30);
    }

    #[test]
    fn test_partition_calculator_unpartitioned_error() {
        let table_schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();

        let partition_spec = PartitionSpecBuilder::new(Arc::new(table_schema.clone()))
            .build()
            .unwrap();

        let result = PartitionValueCalculator::try_new(&partition_spec, &table_schema);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("unpartitioned table")
        );
    }
}
