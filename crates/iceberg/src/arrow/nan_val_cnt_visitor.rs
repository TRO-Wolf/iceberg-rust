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

//! The module contains the visitor for calculating NaN values in give arrow record batch.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Float32Array, Float64Array, RecordBatch, StructArray};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;

use crate::Result;
use crate::arrow::{ArrowArrayAccessor, FieldMatchMode};
use crate::spec::{
    ListType, MapType, NestedFieldRef, PrimitiveType, Schema, SchemaRef, SchemaWithPartnerVisitor,
    StructType, Type, visit_struct_with_partner,
};

/// Count NaN values in a float/double column, respecting the null bitmap (null is not NaN).
///
/// Walks the values buffer + validity instead of the slower `Option` iterator from
/// `PrimitiveArray::iter()`, which is the hot path on every Parquet write batch when the schema
/// carries float leaves under metrics collection.
fn count_nans_in_float_array(col: &ArrayRef) -> Option<u64> {
    match col.data_type() {
        DataType::Float32 => {
            let arr = col.as_any().downcast_ref::<Float32Array>()?;
            Some(count_nans_f32(arr.values().as_ref(), arr.nulls()))
        }
        DataType::Float64 => {
            let arr = col.as_any().downcast_ref::<Float64Array>()?;
            Some(count_nans_f64(arr.values().as_ref(), arr.nulls()))
        }
        _ => None,
    }
}

fn count_nans_f32(values: &[f32], nulls: Option<&NullBuffer>) -> u64 {
    match nulls {
        None => values.iter().filter(|v| v.is_nan()).count() as u64,
        Some(n) => values
            .iter()
            .enumerate()
            .filter(|(i, v)| n.is_valid(*i) && v.is_nan())
            .count() as u64,
    }
}

fn count_nans_f64(values: &[f64], nulls: Option<&NullBuffer>) -> u64 {
    match nulls {
        None => values.iter().filter(|v| v.is_nan()).count() as u64,
        Some(n) => values
            .iter()
            .enumerate()
            .filter(|(i, v)| n.is_valid(*i) && v.is_nan())
            .count() as u64,
    }
}

/// Visitor which counts and keeps track of NaN value counts in given record batch(s)
pub struct NanValueCountVisitor {
    /// Stores field ID to NaN value count mapping
    pub nan_value_counts: HashMap<i32, u64>,
    match_mode: FieldMatchMode,
}

impl SchemaWithPartnerVisitor<ArrayRef> for NanValueCountVisitor {
    type T = ();

    fn schema(
        &mut self,
        _schema: &Schema,
        _partner: &ArrayRef,
        _value: Self::T,
    ) -> Result<Self::T> {
        Ok(())
    }

    fn field(
        &mut self,
        _field: &NestedFieldRef,
        _partner: &ArrayRef,
        _value: Self::T,
    ) -> Result<Self::T> {
        Ok(())
    }

    fn r#struct(
        &mut self,
        _struct: &StructType,
        _partner: &ArrayRef,
        _results: Vec<Self::T>,
    ) -> Result<Self::T> {
        Ok(())
    }

    fn list(&mut self, _list: &ListType, _list_arr: &ArrayRef, _value: Self::T) -> Result<Self::T> {
        Ok(())
    }

    fn map(
        &mut self,
        _map: &MapType,
        _partner: &ArrayRef,
        _key_value: Self::T,
        _value: Self::T,
    ) -> Result<Self::T> {
        Ok(())
    }

    fn primitive(&mut self, _p: &PrimitiveType, _col: &ArrayRef) -> Result<Self::T> {
        Ok(())
    }

    fn after_struct_field(&mut self, field: &NestedFieldRef, partner: &ArrayRef) -> Result<()> {
        self.accumulate_nan_count(field.id, partner);
        Ok(())
    }

    fn after_list_element(&mut self, field: &NestedFieldRef, partner: &ArrayRef) -> Result<()> {
        self.accumulate_nan_count(field.id, partner);
        Ok(())
    }

    fn after_map_key(&mut self, field: &NestedFieldRef, partner: &ArrayRef) -> Result<()> {
        self.accumulate_nan_count(field.id, partner);
        Ok(())
    }

    fn after_map_value(&mut self, field: &NestedFieldRef, partner: &ArrayRef) -> Result<()> {
        self.accumulate_nan_count(field.id, partner);
        Ok(())
    }
}

impl NanValueCountVisitor {
    /// Creates new instance of NanValueCountVisitor
    pub fn new() -> Self {
        Self::new_with_match_mode(FieldMatchMode::Id)
    }

    /// Creates new instance of NanValueCountVisitor with explicit match mode
    pub fn new_with_match_mode(match_mode: FieldMatchMode) -> Self {
        Self {
            nan_value_counts: HashMap::new(),
            match_mode,
        }
    }

    /// Compute nan value counts in given schema and record batch.
    ///
    /// Takes the batch by shared reference so the Parquet write path does not need to
    /// `batch.clone()` before visiting. The visitor builds a view over the batch columns
    /// (Arc clones of the column arrays only).
    pub fn compute(&mut self, schema: SchemaRef, batch: &RecordBatch) -> Result<()> {
        let arrow_arr_partner_accessor = ArrowArrayAccessor::new_with_match_mode(self.match_mode);

        // Build a StructArray view without taking ownership of `batch` (Arc-clone columns only).
        let struct_arr = Arc::new(StructArray::new(
            batch.schema().fields().clone(),
            batch.columns().to_vec(),
            None,
        )) as ArrayRef;
        visit_struct_with_partner(
            schema.as_struct(),
            &struct_arr,
            self,
            &arrow_arr_partner_accessor,
        )?;

        Ok(())
    }

    fn accumulate_nan_count(&mut self, field_id: i32, col: &ArrayRef) {
        let Some(nan_val_cnt) = count_nans_in_float_array(col) else {
            // Non-float partner — nothing to track.
            return;
        };
        match self.nan_value_counts.entry(field_id) {
            Entry::Occupied(mut ele) => {
                let total_nan_val_cnt = ele.get() + nan_val_cnt;
                ele.insert(total_nan_val_cnt);
            }
            Entry::Vacant(v) => {
                v.insert(nan_val_cnt);
            }
        }
    }
}

impl Default for NanValueCountVisitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Whether any float/double leaf in `schema` will persist `nan_value_counts` under `metrics`.
///
/// Used by [`crate::writer::file_writer::parquet_writer::ParquetWriter`] to skip the NaN visitor
/// entirely when the write schema has no float leaves under a counts-collecting metrics mode —
/// the common case for int/string/timestamp-only tables.
pub(crate) fn schema_needs_nan_value_counts(
    schema: &Schema,
    metrics: &crate::spec::MetricsConfig,
) -> bool {
    for (field_id, field) in schema.field_id_to_fields() {
        let is_float = matches!(
            field.field_type.as_ref(),
            Type::Primitive(PrimitiveType::Float | PrimitiveType::Double)
        );
        if !is_float {
            continue;
        }
        let mode = match schema.name_by_field_id(*field_id) {
            Some(name) => metrics.column_mode(name),
            None => metrics.default_mode_of(),
        };
        if mode.collects_counts() {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::spec::{
        MetricsConfig, MetricsMode, NestedField, PrimitiveType, Schema, StructType, Type,
    };

    fn schema_with_fields(fields: Vec<NestedField>) -> Schema {
        Schema::builder()
            .with_fields(fields.into_iter().map(Arc::new).collect::<Vec<_>>())
            .build()
            .expect("schema")
    }

    #[test]
    fn schema_without_floats_does_not_need_nan_counts() {
        let schema = schema_with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)),
            NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)),
        ]);
        assert!(
            !schema_needs_nan_value_counts(&schema, &MetricsConfig::default()),
            "int/string schema must skip the NaN visitor"
        );
    }

    #[test]
    fn schema_with_float_leaf_needs_nan_counts() {
        let schema = schema_with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)),
            NestedField::optional(2, "score", Type::Primitive(PrimitiveType::Float)),
        ]);
        assert!(
            schema_needs_nan_value_counts(&schema, &MetricsConfig::default()),
            "float leaf under default metrics must enable the NaN visitor"
        );
    }

    #[test]
    fn schema_with_double_leaf_needs_nan_counts() {
        let schema = schema_with_fields(vec![NestedField::optional(
            1,
            "x",
            Type::Primitive(PrimitiveType::Double),
        )]);
        assert!(schema_needs_nan_value_counts(
            &schema,
            &MetricsConfig::default()
        ));
    }

    #[test]
    fn metrics_none_on_float_skips_nan_visitor() {
        let schema = schema_with_fields(vec![NestedField::optional(
            1,
            "score",
            Type::Primitive(PrimitiveType::Float),
        )]);
        let metrics = MetricsConfig::from_properties(&std::collections::HashMap::from([(
            "write.metadata.metrics.default".to_string(),
            "none".to_string(),
        )]));
        assert_eq!(metrics.default_mode_of(), MetricsMode::None);
        assert!(
            !schema_needs_nan_value_counts(&schema, &metrics),
            "MetricsMode::None must skip NaN counting even when float leaves exist"
        );
    }

    #[test]
    fn nested_float_leaf_is_detected() {
        let inner = Type::Struct(StructType::new(vec![Arc::new(NestedField::optional(
            2,
            "lat",
            Type::Primitive(PrimitiveType::Float),
        ))]));
        let schema = schema_with_fields(vec![NestedField::optional(1, "loc", inner)]);
        assert!(
            schema_needs_nan_value_counts(&schema, &MetricsConfig::default()),
            "nested float leaf must enable the NaN visitor"
        );
    }

    #[test]
    fn list_element_float_enables_gate() {
        use crate::spec::ListType;
        let element = Arc::new(NestedField::list_element(
            2,
            Type::Primitive(PrimitiveType::Float),
            false,
        ));
        let list = Type::List(ListType::new(element));
        let schema = schema_with_fields(vec![NestedField::optional(1, "scores", list)]);
        assert!(
            schema_needs_nan_value_counts(&schema, &MetricsConfig::default()),
            "list<float> element leaf must enable the NaN visitor"
        );
    }

    #[test]
    fn map_value_float_enables_gate() {
        use crate::spec::MapType;
        let key = Arc::new(NestedField::map_key_element(
            2,
            Type::Primitive(PrimitiveType::String),
        ));
        let value = Arc::new(NestedField::map_value_element(
            3,
            Type::Primitive(PrimitiveType::Double),
            false,
        ));
        let map = Type::Map(MapType::new(key, value));
        let schema = schema_with_fields(vec![NestedField::optional(1, "m", map)]);
        assert!(
            schema_needs_nan_value_counts(&schema, &MetricsConfig::default()),
            "map value double leaf must enable the NaN visitor"
        );
    }

    #[test]
    fn map_key_float_enables_gate() {
        use crate::spec::MapType;
        let key = Arc::new(NestedField::map_key_element(
            2,
            Type::Primitive(PrimitiveType::Float),
        ));
        let value = Arc::new(NestedField::map_value_element(
            3,
            Type::Primitive(PrimitiveType::String),
            false,
        ));
        let map = Type::Map(MapType::new(key, value));
        let schema = schema_with_fields(vec![NestedField::optional(1, "m", map)]);
        assert!(
            schema_needs_nan_value_counts(&schema, &MetricsConfig::default()),
            "map key float leaf must enable the NaN visitor"
        );
    }

    #[test]
    fn metrics_counts_mode_enables_gate_for_float() {
        let schema = schema_with_fields(vec![NestedField::optional(
            1,
            "score",
            Type::Primitive(PrimitiveType::Float),
        )]);
        let metrics = MetricsConfig::from_properties(&std::collections::HashMap::from([(
            "write.metadata.metrics.default".to_string(),
            "counts".to_string(),
        )]));
        assert_eq!(metrics.default_mode_of(), MetricsMode::Counts);
        assert!(
            schema_needs_nan_value_counts(&schema, &metrics),
            "MetricsMode::Counts must collect nan_value_counts"
        );
    }

    #[test]
    fn column_override_full_enables_gate_when_default_none() {
        let schema = schema_with_fields(vec![NestedField::optional(
            1,
            "score",
            Type::Primitive(PrimitiveType::Float),
        )]);
        let metrics = MetricsConfig::from_properties(&std::collections::HashMap::from([
            (
                "write.metadata.metrics.default".to_string(),
                "none".to_string(),
            ),
            (
                "write.metadata.metrics.column.score".to_string(),
                "full".to_string(),
            ),
        ]));
        assert!(
            schema_needs_nan_value_counts(&schema, &metrics),
            "per-column full override must enable the NaN visitor when default is none"
        );
    }

    #[test]
    fn column_override_none_disables_gate_for_only_float() {
        let schema = schema_with_fields(vec![NestedField::optional(
            1,
            "score",
            Type::Primitive(PrimitiveType::Float),
        )]);
        let metrics = MetricsConfig::from_properties(&std::collections::HashMap::from([(
            "write.metadata.metrics.column.score".to_string(),
            "none".to_string(),
        )]));
        assert!(
            !schema_needs_nan_value_counts(&schema, &metrics),
            "per-column none on the only float must skip the NaN visitor"
        );
    }

    /// MUTATION: drop `n.is_valid(*i) &&` in `count_nans_f32`/`count_nans_f64` ⇒ null slot with a
    /// NaN bit-pattern is counted and this pin fails (Iceberg: null is not NaN).
    #[test]
    fn null_slot_with_nan_bits_is_not_counted() {
        use arrow_array::{Float32Array, Float64Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema as ArrowSchema};
        use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

        let iceberg = Arc::new(schema_with_fields(vec![
            NestedField::optional(1, "f", Type::Primitive(PrimitiveType::Float)),
            NestedField::optional(2, "d", Type::Primitive(PrimitiveType::Double)),
        ]));

        // Index 1 is null but the values buffer holds NaN — must NOT count as a NaN value.
        let f32 = Float32Array::from(vec![Some(1.0_f32), None, Some(f32::NAN)]);
        // Force NaN bits into the null slot (Arrow may leave arbitrary payload there).
        let mut f32_builder_vals = f32.values().to_vec();
        f32_builder_vals[1] = f32::NAN;
        let f32 = Float32Array::new(
            arrow_buffer::ScalarBuffer::from(f32_builder_vals),
            f32.nulls().cloned(),
        );

        let f64 = Float64Array::from(vec![Some(1.0_f64), None, Some(f64::NAN)]);
        let mut f64_vals = f64.values().to_vec();
        f64_vals[1] = f64::NAN;
        let f64 = Float64Array::new(
            arrow_buffer::ScalarBuffer::from(f64_vals),
            f64.nulls().cloned(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("f", DataType::Float32, true).with_metadata(
                std::collections::HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    "1".to_string(),
                )]),
            ),
            Field::new("d", DataType::Float64, true).with_metadata(
                std::collections::HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    "2".to_string(),
                )]),
            ),
        ]));
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(f32) as ArrayRef,
            Arc::new(f64) as ArrayRef,
        ])
        .expect("batch");

        let mut visitor = NanValueCountVisitor::new();
        visitor
            .compute(iceberg, &batch)
            .expect("compute nan counts");
        assert_eq!(
            visitor.nan_value_counts.get(&1).copied().unwrap_or(0),
            1,
            "exactly one valid f32 NaN (null slot must not count)"
        );
        assert_eq!(
            visitor.nan_value_counts.get(&2).copied().unwrap_or(0),
            1,
            "exactly one valid f64 NaN (null slot must not count)"
        );
    }

    #[test]
    fn all_valid_nans_are_counted() {
        use arrow_array::{Float32Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema as ArrowSchema};
        use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

        let iceberg = Arc::new(schema_with_fields(vec![NestedField::optional(
            1,
            "f",
            Type::Primitive(PrimitiveType::Float),
        )]));
        let f32 = Float32Array::from(vec![f32::NAN, 1.0, f32::NAN]);
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("f", DataType::Float32, true).with_metadata(
                std::collections::HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    "1".to_string(),
                )]),
            ),
        ]));
        let batch =
            RecordBatch::try_new(arrow_schema, vec![Arc::new(f32) as ArrayRef]).expect("batch");
        let mut visitor = NanValueCountVisitor::new();
        visitor
            .compute(iceberg, &batch)
            .expect("compute nan counts");
        assert_eq!(visitor.nan_value_counts.get(&1).copied(), Some(2));
    }

    /// Inf/-Inf are not NaN; the buffer walk must not treat them as NaN.
    #[test]
    fn infinity_is_not_counted_as_nan() {
        use arrow_array::{Float32Array, Float64Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema as ArrowSchema};
        use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

        let iceberg = Arc::new(schema_with_fields(vec![
            NestedField::optional(1, "f", Type::Primitive(PrimitiveType::Float)),
            NestedField::optional(2, "d", Type::Primitive(PrimitiveType::Double)),
        ]));
        let f32 = Float32Array::from(vec![f32::INFINITY, f32::NEG_INFINITY, 0.0, f32::NAN]);
        let f64 = Float64Array::from(vec![f64::INFINITY, f64::NEG_INFINITY, 0.0, f64::NAN]);
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("f", DataType::Float32, true).with_metadata(
                std::collections::HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    "1".to_string(),
                )]),
            ),
            Field::new("d", DataType::Float64, true).with_metadata(
                std::collections::HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    "2".to_string(),
                )]),
            ),
        ]));
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(f32) as ArrayRef,
            Arc::new(f64) as ArrayRef,
        ])
        .expect("batch");
        let mut visitor = NanValueCountVisitor::new();
        visitor
            .compute(iceberg, &batch)
            .expect("compute nan counts");
        assert_eq!(visitor.nan_value_counts.get(&1).copied(), Some(1));
        assert_eq!(visitor.nan_value_counts.get(&2).copied(), Some(1));
    }
}
