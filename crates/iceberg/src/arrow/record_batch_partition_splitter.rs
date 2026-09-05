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

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch, StructArray, UInt32Array};
use arrow_ord::partition::partition;
use arrow_ord::sort::{SortColumn, lexsort_to_indices};
use arrow_schema::ArrowError;
use arrow_select::filter::filter_record_batch;
use arrow_select::take::{take, take_record_batch};

use super::arrow_struct_to_literal;
use super::partition_value_calculator::PartitionValueCalculator;
use crate::spec::{
    Literal, PartitionKey, PartitionSpecRef, PrimitiveType, SchemaRef, Struct, StructType, Type,
};
use crate::{Error, ErrorKind, Result};

/// Column name for the projected partition values struct
pub const PROJECTED_PARTITION_VALUE_COLUMN: &str = "_partition";

/// The splitter used to split the record batch into multiple record batches by the partition spec.
/// 1. It will project and transform the input record batch based on the partition spec, get the partitioned record batch.
/// 2. Split the input record batch into multiple record batches based on the partitioned record batch.
///
/// # Partition Value Modes
///
/// The splitter supports two modes for obtaining partition values:
/// - **Computed mode** (`calculator` is `Some`): Computes partition values from source columns using transforms
/// - **Pre-computed mode** (`calculator` is `None`): Expects a `_partition` column in the input batch
pub struct RecordBatchPartitionSplitter {
    schema: SchemaRef,
    partition_spec: PartitionSpecRef,
    calculator: Option<PartitionValueCalculator>,
    partition_type: StructType,
    arrow_grouping: bool,
}

impl RecordBatchPartitionSplitter {
    /// Create a new RecordBatchPartitionSplitter.
    ///
    /// # Arguments
    ///
    /// * `iceberg_schema` - The Iceberg schema reference
    /// * `partition_spec` - The partition specification reference
    /// * `calculator` - Optional calculator for computing partition values from source columns.
    ///   - `Some(calculator)`: Compute partition values from source columns using transforms
    ///   - `None`: Expect a pre-computed `_partition` column in the input batch
    ///
    /// # Returns
    ///
    /// Returns a new `RecordBatchPartitionSplitter` instance or an error if initialization fails.
    pub fn try_new(
        iceberg_schema: SchemaRef,
        partition_spec: PartitionSpecRef,
        calculator: Option<PartitionValueCalculator>,
    ) -> Result<Self> {
        let partition_type = partition_spec.partition_type(&iceberg_schema)?;
        let arrow_grouping = arrow_order_matches_struct_equality(&partition_type);

        Ok(Self {
            schema: iceberg_schema,
            partition_spec,
            calculator,
            partition_type,
            arrow_grouping,
        })
    }

    /// Create a new RecordBatchPartitionSplitter with computed partition values.
    ///
    /// This is a convenience method that creates a calculator and initializes the splitter
    /// to compute partition values from source columns.
    ///
    /// # Arguments
    ///
    /// * `iceberg_schema` - The Iceberg schema reference
    /// * `partition_spec` - The partition specification reference
    ///
    /// # Returns
    ///
    /// Returns a new `RecordBatchPartitionSplitter` instance or an error if initialization fails.
    pub fn try_new_with_computed_values(
        iceberg_schema: SchemaRef,
        partition_spec: PartitionSpecRef,
    ) -> Result<Self> {
        let calculator = PartitionValueCalculator::try_new(&partition_spec, &iceberg_schema)?;
        Self::try_new(iceberg_schema, partition_spec, Some(calculator))
    }

    /// Create a new RecordBatchPartitionSplitter expecting pre-computed partition values.
    ///
    /// This is a convenience method that initializes the splitter to expect a `_partition`
    /// column in the input batches.
    ///
    /// # Arguments
    ///
    /// * `iceberg_schema` - The Iceberg schema reference
    /// * `partition_spec` - The partition specification reference
    ///
    /// # Returns
    ///
    /// Returns a new `RecordBatchPartitionSplitter` instance or an error if initialization fails.
    pub fn try_new_with_precomputed_values(
        iceberg_schema: SchemaRef,
        partition_spec: PartitionSpecRef,
    ) -> Result<Self> {
        Self::try_new(iceberg_schema, partition_spec, None)
    }

    /// Split the record batch into multiple record batches based on the partition spec.
    pub fn split(&self, batch: &RecordBatch) -> Result<Vec<(PartitionKey, RecordBatch)>> {
        let partition_array = self.partition_value_array(batch)?;
        if self.arrow_grouping {
            self.split_by_arrow_order(batch, &partition_array)
        } else {
            self.split_row_wise(batch, &partition_array)
        }
    }

    fn partition_value_array(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        if let Some(calculator) = &self.calculator {
            return calculator.calculate(batch);
        }
        let partition_column = batch
            .column_by_name(PROJECTED_PARTITION_VALUE_COLUMN)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Partition column '{PROJECTED_PARTITION_VALUE_COLUMN}' not found in batch"
                    ),
                )
            })?;
        let partition_struct_array = partition_column
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Partition column is not a StructArray",
                )
            })?;
        Ok(Arc::new(partition_struct_array.clone()) as ArrayRef)
    }

    fn split_by_arrow_order(
        &self,
        batch: &RecordBatch,
        partition_array: &ArrayRef,
    ) -> Result<Vec<(PartitionKey, RecordBatch)>> {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(Vec::new());
        }
        let key_columns = self.partition_key_columns(partition_array)?;
        let sort_columns = key_columns
            .iter()
            .map(|values| SortColumn {
                values: values.clone(),
                options: None,
            })
            .collect::<Vec<_>>();
        let sorted_positions = lexsort_to_indices(&sort_columns, None).map_err(arrow_err)?;
        let sorted_keys = key_columns
            .iter()
            .map(|values| take(values.as_ref(), &sorted_positions, None))
            .collect::<std::result::Result<Vec<_>, ArrowError>>()
            .map_err(arrow_err)?;
        let ranges = partition(&sorted_keys).map_err(arrow_err)?.ranges();

        let mut group_of_row = vec![0usize; num_rows];
        for (group, range) in ranges.iter().enumerate() {
            for position in range.clone() {
                group_of_row[sorted_positions.value(position) as usize] = group;
            }
        }
        let mut row_ids = ranges
            .iter()
            .map(|range| Vec::with_capacity(range.len()))
            .collect::<Vec<Vec<u32>>>();
        for (row, group) in group_of_row.into_iter().enumerate() {
            row_ids[group].push(row as u32);
        }

        let representatives = UInt32Array::from(
            row_ids
                .iter()
                .filter_map(|ids| ids.first().copied())
                .collect::<Vec<u32>>(),
        );
        let representative_values =
            take(partition_array.as_ref(), &representatives, None).map_err(arrow_err)?;
        let literals = arrow_struct_to_literal(&representative_values, &self.partition_type)?;

        let mut partition_batches = Vec::with_capacity(row_ids.len());
        for (literal, ids) in literals.into_iter().zip(row_ids) {
            let Some(Literal::Struct(row)) = literal else {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Partition value is not a struct literal or is null",
                ));
            };
            let partition_key = PartitionKey::new(
                self.partition_spec.as_ref().clone(),
                self.schema.clone(),
                row,
            )?;
            let indices = UInt32Array::from(ids);
            let partition_batch = take_record_batch(batch, &indices).map_err(arrow_err)?;
            partition_batches.push((partition_key, partition_batch));
        }
        Ok(partition_batches)
    }

    fn partition_key_columns(&self, partition_array: &ArrayRef) -> Result<Vec<ArrayRef>> {
        let struct_array = partition_array
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Partition column is not a StructArray",
                )
            })?;
        if struct_array.null_count() > 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Partition value is not a struct literal or is null",
            ));
        }
        Ok(struct_array.columns().to_vec())
    }

    fn split_row_wise(
        &self,
        batch: &RecordBatch,
        partition_array: &ArrayRef,
    ) -> Result<Vec<(PartitionKey, RecordBatch)>> {
        let partition_structs = arrow_struct_to_literal(partition_array, &self.partition_type)?
            .into_iter()
            .map(|s| {
                if let Some(Literal::Struct(s)) = s {
                    Ok(s)
                } else {
                    Err(Error::new(
                        ErrorKind::DataInvalid,
                        "Partition value is not a struct literal or is null",
                    ))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // Group the batch by row value. Key the group map by a BORROW of each partition struct (and
        // remember the first row at which the group appears) so we clone a `Struct` only ONCE per
        // distinct partition value — used as the `PartitionKey` below — instead of once per row. The
        // grouping is identical to cloning every key: the same `Struct` `Hash`/`Eq` decides
        // membership, only the owned-clone count changes (R → G distinct partitions).
        let mut group_ids: HashMap<&Struct, (usize, Vec<usize>)> = HashMap::new();
        partition_structs
            .iter()
            .enumerate()
            .for_each(|(row_id, row)| {
                group_ids
                    .entry(row)
                    .or_insert_with(|| (row_id, vec![]))
                    .1
                    .push(row_id);
            });

        // Partition the batch with same partition partition_values
        let mut partition_batches = Vec::with_capacity(group_ids.len());
        for (representative_row, row_ids) in group_ids.into_values() {
            // Clone the partition struct ONCE per group (the representative first-occurrence row).
            let row = partition_structs[representative_row].clone();
            // generate the bool filter array from column_ids
            let filter_array: BooleanArray = {
                let mut filter = vec![false; batch.num_rows()];
                row_ids.into_iter().for_each(|row_id| {
                    filter[row_id] = true;
                });
                filter.into()
            };

            // Create PartitionKey from the partition struct
            let partition_key = PartitionKey::new(
                self.partition_spec.as_ref().clone(),
                self.schema.clone(),
                row,
            )?;

            // filter the RecordBatch
            partition_batches.push((partition_key, filter_record_batch(batch, &filter_array)?));
        }

        Ok(partition_batches)
    }
}

fn arrow_order_matches_struct_equality(partition_type: &StructType) -> bool {
    !partition_type.fields().is_empty()
        && partition_type.fields().iter().all(|field| {
            matches!(
                field.field_type.as_ref(),
                Type::Primitive(
                    PrimitiveType::Boolean
                        | PrimitiveType::Int
                        | PrimitiveType::Long
                        | PrimitiveType::Decimal { .. }
                        | PrimitiveType::Date
                        | PrimitiveType::Time
                        | PrimitiveType::Timestamp
                        | PrimitiveType::Timestamptz
                        | PrimitiveType::TimestampNs
                        | PrimitiveType::TimestamptzNs
                        | PrimitiveType::String
                        | PrimitiveType::Uuid
                        | PrimitiveType::Fixed(_)
                        | PrimitiveType::Binary
                )
            )
        })
}

fn arrow_err(error: ArrowError) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        "Failed to group a record batch by its partition values",
    )
    .with_source(error)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        BooleanArray, Date32Array, Decimal128Array, FixedSizeBinaryArray, Float64Array, Int32Array,
        Int64Array, LargeBinaryArray, RecordBatch, StringArray, Time64MicrosecondArray,
        TimestampMicrosecondArray, TimestampNanosecondArray,
    };
    use arrow_schema::DataType;
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::arrow::schema_to_arrow_schema;
    use crate::spec::{
        NestedField, PartitionSpecBuilder, PrimitiveLiteral, Schema, Struct, Transform, Type,
        UnboundPartitionField,
    };

    #[test]
    fn test_record_batch_partition_split() {
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(
                        1,
                        "id",
                        Type::Primitive(crate::spec::PrimitiveType::Int),
                    )
                    .into(),
                    NestedField::required(
                        2,
                        "name",
                        Type::Primitive(crate::spec::PrimitiveType::String),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        let partition_spec = Arc::new(
            PartitionSpecBuilder::new(schema.clone())
                .with_spec_id(1)
                .add_unbound_field(UnboundPartitionField {
                    source_id: 1,
                    field_id: None,
                    name: "id_bucket".to_string(),
                    transform: Transform::Identity,
                })
                .unwrap()
                .build()
                .unwrap(),
        );
        let partition_splitter = RecordBatchPartitionSplitter::try_new_with_computed_values(
            schema.clone(),
            partition_spec,
        )
        .expect("Failed to create splitter");

        let arrow_schema = Arc::new(schema_to_arrow_schema(&schema).unwrap());
        let id_array = Int32Array::from(vec![1, 2, 1, 3, 2, 3, 1]);
        let data_array = StringArray::from(vec!["a", "b", "c", "d", "e", "f", "g"]);
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(id_array),
            Arc::new(data_array),
        ])
        .expect("Failed to create RecordBatch");

        let mut partitioned_batches = partition_splitter
            .split(&batch)
            .expect("Failed to split RecordBatch");
        partitioned_batches.sort_by_key(|(partition_key, _)| {
            if let PrimitiveLiteral::Int(i) = partition_key.data().fields()[0]
                .as_ref()
                .unwrap()
                .as_primitive_literal()
                .unwrap()
            {
                i
            } else {
                panic!("The partition value is not a int");
            }
        });
        assert_eq!(partitioned_batches.len(), 3);
        {
            // check the first partition
            let expected_id_array = Int32Array::from(vec![1, 1, 1]);
            let expected_data_array = StringArray::from(vec!["a", "c", "g"]);
            let expected_batch = RecordBatch::try_new(arrow_schema.clone(), vec![
                Arc::new(expected_id_array),
                Arc::new(expected_data_array),
            ])
            .expect("Failed to create expected RecordBatch");
            assert_eq!(partitioned_batches[0].1, expected_batch);
        }
        {
            // check the second partition
            let expected_id_array = Int32Array::from(vec![2, 2]);
            let expected_data_array = StringArray::from(vec!["b", "e"]);
            let expected_batch = RecordBatch::try_new(arrow_schema.clone(), vec![
                Arc::new(expected_id_array),
                Arc::new(expected_data_array),
            ])
            .expect("Failed to create expected RecordBatch");
            assert_eq!(partitioned_batches[1].1, expected_batch);
        }
        {
            // check the third partition
            let expected_id_array = Int32Array::from(vec![3, 3]);
            let expected_data_array = StringArray::from(vec!["d", "f"]);
            let expected_batch = RecordBatch::try_new(arrow_schema.clone(), vec![
                Arc::new(expected_id_array),
                Arc::new(expected_data_array),
            ])
            .expect("Failed to create expected RecordBatch");
            assert_eq!(partitioned_batches[2].1, expected_batch);
        }

        let partition_values = partitioned_batches
            .iter()
            .map(|(partition_key, _)| partition_key.data().clone())
            .collect::<Vec<_>>();
        // check partition value is struct(1), struct(2), struct(3)
        assert_eq!(partition_values, vec![
            Struct::from_iter(vec![Some(Literal::int(1))]),
            Struct::from_iter(vec![Some(Literal::int(2))]),
            Struct::from_iter(vec![Some(Literal::int(3))]),
        ]);
    }

    #[test]
    fn test_record_batch_partition_split_with_partition_column() {
        use arrow_array::StructArray;
        use arrow_schema::{Field, Schema as ArrowSchema};

        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(
                        1,
                        "id",
                        Type::Primitive(crate::spec::PrimitiveType::Int),
                    )
                    .into(),
                    NestedField::required(
                        2,
                        "name",
                        Type::Primitive(crate::spec::PrimitiveType::String),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        let partition_spec = Arc::new(
            PartitionSpecBuilder::new(schema.clone())
                .with_spec_id(1)
                .add_unbound_field(UnboundPartitionField {
                    source_id: 1,
                    field_id: None,
                    name: "id_bucket".to_string(),
                    transform: Transform::Identity,
                })
                .unwrap()
                .build()
                .unwrap(),
        );

        // Create input schema with _partition column
        // Note: partition field IDs start from 1000 by default
        let partition_field = Field::new("id_bucket", DataType::Int32, false).with_metadata(
            HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1000".to_string())]),
        );
        let partition_struct_field = Field::new(
            PROJECTED_PARTITION_VALUE_COLUMN,
            DataType::Struct(vec![partition_field.clone()].into()),
            false,
        );

        let input_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            partition_struct_field,
        ]));

        // Create splitter expecting pre-computed partition column
        let partition_splitter = RecordBatchPartitionSplitter::try_new_with_precomputed_values(
            schema.clone(),
            partition_spec,
        )
        .expect("Failed to create splitter");

        // Create test data with pre-computed partition column
        let id_array = Int32Array::from(vec![1, 2, 1, 3, 2, 3, 1]);
        let data_array = StringArray::from(vec!["a", "b", "c", "d", "e", "f", "g"]);

        // Create partition column (same values as id for Identity transform)
        let partition_values = Int32Array::from(vec![1, 2, 1, 3, 2, 3, 1]);
        let partition_struct = StructArray::from(vec![(
            Arc::new(partition_field),
            Arc::new(partition_values) as ArrayRef,
        )]);

        let batch = RecordBatch::try_new(input_schema.clone(), vec![
            Arc::new(id_array),
            Arc::new(data_array),
            Arc::new(partition_struct),
        ])
        .expect("Failed to create RecordBatch");

        // Split using the pre-computed partition column
        let mut partitioned_batches = partition_splitter
            .split(&batch)
            .expect("Failed to split RecordBatch");

        partitioned_batches.sort_by_key(|(partition_key, _)| {
            if let PrimitiveLiteral::Int(i) = partition_key.data().fields()[0]
                .as_ref()
                .unwrap()
                .as_primitive_literal()
                .unwrap()
            {
                i
            } else {
                panic!("The partition value is not a int");
            }
        });

        assert_eq!(partitioned_batches.len(), 3);

        // Helper to extract id and name values from a batch
        let extract_values = |batch: &RecordBatch| -> (Vec<i32>, Vec<String>) {
            let id_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let name_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            (
                id_col.values().to_vec(),
                name_col.iter().map(|s| s.unwrap().to_string()).collect(),
            )
        };

        // Verify partition 1: id=1, names=["a", "c", "g"]
        let (key, batch) = &partitioned_batches[0];
        assert_eq!(key.data(), &Struct::from_iter(vec![Some(Literal::int(1))]));
        let (ids, names) = extract_values(batch);
        assert_eq!(ids, vec![1, 1, 1]);
        assert_eq!(names, vec!["a", "c", "g"]);

        // Verify partition 2: id=2, names=["b", "e"]
        let (key, batch) = &partitioned_batches[1];
        assert_eq!(key.data(), &Struct::from_iter(vec![Some(Literal::int(2))]));
        let (ids, names) = extract_values(batch);
        assert_eq!(ids, vec![2, 2]);
        assert_eq!(names, vec!["b", "e"]);

        // Verify partition 3: id=3, names=["d", "f"]
        let (key, batch) = &partitioned_batches[2];
        assert_eq!(key.data(), &Struct::from_iter(vec![Some(Literal::int(3))]));
        let (ids, names) = extract_values(batch);
        assert_eq!(ids, vec![3, 3]);
        assert_eq!(names, vec!["d", "f"]);
    }

    const BINARY_WORDS: [&[u8]; 5] = [b"", b"a", b"ab", b"abc", b"\xff\x00"];

    fn property_schema() -> Arc<Schema> {
        Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "i", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "l", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(3, "s", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(4, "d", Type::Primitive(PrimitiveType::Date)).into(),
                    NestedField::optional(5, "t", Type::Primitive(PrimitiveType::Timestamp)).into(),
                    NestedField::optional(6, "b", Type::Primitive(PrimitiveType::Boolean)).into(),
                    NestedField::optional(
                        7,
                        "n",
                        Type::Primitive(PrimitiveType::Decimal {
                            precision: 9,
                            scale: 2,
                        }),
                    )
                    .into(),
                    NestedField::optional(8, "y", Type::Primitive(PrimitiveType::Binary)).into(),
                    NestedField::optional(9, "f", Type::Primitive(PrimitiveType::Fixed(4))).into(),
                    NestedField::optional(10, "u", Type::Primitive(PrimitiveType::Uuid)).into(),
                    NestedField::optional(11, "z", Type::Primitive(PrimitiveType::Timestamptz))
                        .into(),
                    NestedField::optional(12, "m", Type::Primitive(PrimitiveType::Time)).into(),
                    NestedField::optional(13, "g", Type::Primitive(PrimitiveType::TimestampNs))
                        .into(),
                    NestedField::optional(14, "h", Type::Primitive(PrimitiveType::TimestamptzNs))
                        .into(),
                ])
                .build()
                .unwrap(),
        )
    }

    fn random_batch(schema: &Schema, rows: usize, rng: &mut StdRng) -> RecordBatch {
        let words = ["", "a", "ab", "abc", "abcd", "zzzz"];
        let ints = Int32Array::from(
            (0..rows)
                .map(|_| rng.random_range(-3i32..3))
                .collect::<Vec<_>>(),
        );
        let longs = Int64Array::from(
            (0..rows)
                .map(|_| (!rng.random_bool(0.2)).then(|| rng.random_range(-4i64..4)))
                .collect::<Vec<_>>(),
        );
        let strings = StringArray::from(
            (0..rows)
                .map(|_| {
                    (!rng.random_bool(0.2))
                        .then(|| words[rng.random_range(0..words.len())].to_string())
                })
                .collect::<Vec<_>>(),
        );
        let dates = Date32Array::from(
            (0..rows)
                .map(|_| (!rng.random_bool(0.2)).then(|| rng.random_range(-800i32..800)))
                .collect::<Vec<_>>(),
        );
        let stamps = TimestampMicrosecondArray::from(
            (0..rows)
                .map(|_| {
                    (!rng.random_bool(0.2))
                        .then(|| rng.random_range(-5_000_000_000i64..5_000_000_000))
                })
                .collect::<Vec<_>>(),
        );
        let bools = BooleanArray::from(
            (0..rows)
                .map(|_| (!rng.random_bool(0.2)).then(|| rng.random_bool(0.5)))
                .collect::<Vec<_>>(),
        );
        let decimals = Decimal128Array::from(
            (0..rows)
                .map(|_| (!rng.random_bool(0.2)).then(|| rng.random_range(-500i128..500)))
                .collect::<Vec<_>>(),
        )
        .with_precision_and_scale(9, 2)
        .unwrap();
        let binaries = LargeBinaryArray::from_opt_vec(
            (0..rows)
                .map(|_| {
                    (!rng.random_bool(0.2))
                        .then(|| BINARY_WORDS[rng.random_range(0..BINARY_WORDS.len())])
                })
                .collect::<Vec<_>>(),
        );
        let fixed = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
            (0..rows).map(|_| {
                (!rng.random_bool(0.2)).then(|| {
                    let value = rng.random_range(0u32..6).to_be_bytes();
                    value.to_vec()
                })
            }),
            4,
        )
        .unwrap();
        let uuids = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
            (0..rows).map(|_| {
                (!rng.random_bool(0.2)).then(|| {
                    let value = u128::from(rng.random_range(0u32..6));
                    value.to_be_bytes().to_vec()
                })
            }),
            16,
        )
        .unwrap();
        let stamps_tz = TimestampMicrosecondArray::from(
            (0..rows)
                .map(|_| {
                    (!rng.random_bool(0.2))
                        .then(|| rng.random_range(-5_000_000_000i64..5_000_000_000))
                })
                .collect::<Vec<_>>(),
        )
        .with_timezone("UTC");
        let times = Time64MicrosecondArray::from(
            (0..rows)
                .map(|_| (!rng.random_bool(0.2)).then(|| rng.random_range(0i64..86_400_000_000)))
                .collect::<Vec<_>>(),
        );
        let nanos = TimestampNanosecondArray::from(
            (0..rows)
                .map(|_| {
                    (!rng.random_bool(0.2))
                        .then(|| rng.random_range(-5_000_000_000i64..5_000_000_000))
                })
                .collect::<Vec<_>>(),
        );
        let nanos_tz = TimestampNanosecondArray::from(
            (0..rows)
                .map(|_| {
                    (!rng.random_bool(0.2))
                        .then(|| rng.random_range(-5_000_000_000i64..5_000_000_000))
                })
                .collect::<Vec<_>>(),
        )
        .with_timezone("UTC");
        RecordBatch::try_new(Arc::new(schema_to_arrow_schema(schema).unwrap()), vec![
            Arc::new(ints),
            Arc::new(longs),
            Arc::new(strings),
            Arc::new(dates),
            Arc::new(stamps),
            Arc::new(bools),
            Arc::new(decimals),
            Arc::new(binaries),
            Arc::new(fixed),
            Arc::new(uuids),
            Arc::new(stamps_tz),
            Arc::new(times),
            Arc::new(nanos),
            Arc::new(nanos_tz),
        ])
        .unwrap()
    }

    fn canonical(parts: Vec<(PartitionKey, RecordBatch)>) -> Vec<(String, RecordBatch)> {
        let mut rows = parts
            .into_iter()
            .map(|(key, batch)| (format!("{:?}", key.data()), batch))
            .collect::<Vec<_>>();
        rows.sort_by(|left, right| left.0.cmp(&right.0));
        rows
    }

    fn built_spec(schema: &Arc<Schema>, fields: &[(i32, &str, Transform)]) -> PartitionSpecRef {
        let mut builder = PartitionSpecBuilder::new(schema.clone()).with_spec_id(7);
        for (source_id, name, transform) in fields {
            builder = builder
                .add_unbound_field(UnboundPartitionField {
                    source_id: *source_id,
                    field_id: None,
                    name: (*name).to_string(),
                    transform: *transform,
                })
                .unwrap();
        }
        Arc::new(builder.build().unwrap())
    }

    #[test]
    fn test_arrow_order_grouping_equals_row_wise_grouping() {
        let schema = property_schema();
        let specs: Vec<Vec<(i32, &str, Transform)>> = vec![
            vec![(1, "i_identity", Transform::Identity)],
            vec![(2, "l_identity", Transform::Identity)],
            vec![(3, "s_identity", Transform::Identity)],
            vec![(3, "s_truncate", Transform::Truncate(3))],
            vec![(2, "l_truncate", Transform::Truncate(2))],
            vec![(1, "i_bucket", Transform::Bucket(4))],
            vec![(3, "s_bucket", Transform::Bucket(8))],
            vec![(4, "d_year", Transform::Year)],
            vec![(4, "d_month", Transform::Month)],
            vec![(4, "d_day", Transform::Day)],
            vec![(5, "t_hour", Transform::Hour)],
            vec![(6, "b_identity", Transform::Identity)],
            vec![(7, "n_identity", Transform::Identity)],
            vec![(7, "n_bucket", Transform::Bucket(5))],
            vec![(8, "y_identity", Transform::Identity)],
            vec![(9, "f_identity", Transform::Identity)],
            vec![(9, "f_bucket", Transform::Bucket(3))],
            vec![(10, "u_identity", Transform::Identity)],
            vec![(10, "u_bucket", Transform::Bucket(4))],
            vec![(11, "z_identity", Transform::Identity)],
            vec![(11, "z_hour", Transform::Hour)],
            vec![(12, "m_identity", Transform::Identity)],
            vec![(12, "m_bucket", Transform::Bucket(4))],
            vec![(13, "g_identity", Transform::Identity)],
            vec![(14, "h_identity", Transform::Identity)],
            vec![
                (6, "b_identity", Transform::Identity),
                (7, "n_identity", Transform::Identity),
                (9, "f_identity", Transform::Identity),
            ],
            vec![
                (11, "z_day", Transform::Day),
                (8, "y_identity", Transform::Identity),
            ],
            vec![
                (1, "i_identity", Transform::Identity),
                (2, "l_void", Transform::Void),
            ],
            vec![(3, "s_void", Transform::Void), (4, "d_day", Transform::Day)],
            vec![
                (1, "i_identity", Transform::Identity),
                (3, "s_truncate", Transform::Truncate(2)),
            ],
            vec![
                (2, "l_bucket", Transform::Bucket(3)),
                (4, "d_month", Transform::Month),
                (3, "s_identity", Transform::Identity),
            ],
        ];

        let mut rng = StdRng::seed_from_u64(0x5150_2026);
        for fields in &specs {
            let splitter = RecordBatchPartitionSplitter::try_new_with_computed_values(
                schema.clone(),
                built_spec(&schema, fields),
            )
            .unwrap();
            assert!(
                splitter.arrow_grouping,
                "{fields:?} should group by arrow order"
            );
            for rows in [0usize, 1, 2, 7, 64, 257, 1024] {
                let batch = random_batch(&schema, rows, &mut rng);
                let values = splitter.partition_value_array(&batch).unwrap();
                let expected = canonical(splitter.split_row_wise(&batch, &values).unwrap());
                let actual = canonical(splitter.split_by_arrow_order(&batch, &values).unwrap());
                assert_eq!(actual, expected, "{fields:?} at {rows} rows");
            }
        }
    }

    #[test]
    fn test_float_partition_values_stay_on_the_row_wise_split() {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "v", Type::Primitive(PrimitiveType::Double)).into(),
                ])
                .build()
                .unwrap(),
        );
        let splitter = RecordBatchPartitionSplitter::try_new_with_computed_values(
            schema.clone(),
            built_spec(&schema, &[(1, "v_identity", Transform::Identity)]),
        )
        .unwrap();
        assert!(!splitter.arrow_grouping);

        let batch = RecordBatch::try_new(Arc::new(schema_to_arrow_schema(&schema).unwrap()), vec![
            Arc::new(Float64Array::from(vec![0.0f64, -0.0, 1.0, 0.0])),
        ])
        .unwrap();
        let parts = splitter.split(&batch).unwrap();
        assert_eq!(parts.len(), 2);
        let rows: usize = parts.iter().map(|(_, batch)| batch.num_rows()).sum();
        assert_eq!(rows, 4);
    }
}
