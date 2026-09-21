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

use super::orc_type::OrcSchema;
use crate::spec::{
    Datum, Literal, MetricsByFieldId, MetricsConfig, NestedFieldRef, PrimitiveType, Schema, Struct,
    StructType, Type,
};

#[derive(Debug, Default)]
pub(crate) struct OrcColumnMetrics {
    pub(crate) column_sizes: HashMap<i32, u64>,
    pub(crate) value_counts: HashMap<i32, u64>,
    pub(crate) null_value_counts: HashMap<i32, u64>,
    pub(crate) nan_value_counts: HashMap<i32, u64>,
    pub(crate) lower_bounds: HashMap<i32, Datum>,
    pub(crate) upper_bounds: HashMap<i32, Datum>,
}

pub(crate) struct OrcMetricsCollector {
    metrics: MetricsByFieldId,
    bound_types: HashMap<i32, PrimitiveType>,
    nan_tracked: Vec<i32>,
    row_count: u64,
    null_counts: HashMap<i32, u64>,
    nan_counts: HashMap<i32, u64>,
    lower: HashMap<i32, Datum>,
    upper: HashMap<i32, Datum>,
    column_sizes: HashMap<i32, u64>,
}

impl OrcMetricsCollector {
    pub(crate) fn new(schema: &Schema, config: &MetricsConfig) -> Self {
        let mut collector = OrcMetricsCollector {
            metrics: MetricsByFieldId::new(schema, config),
            bound_types: HashMap::new(),
            nan_tracked: Vec::new(),
            row_count: 0,
            null_counts: HashMap::new(),
            nan_counts: HashMap::new(),
            lower: HashMap::new(),
            upper: HashMap::new(),
            column_sizes: HashMap::new(),
        };
        collector.register_struct(schema.as_struct());
        collector
    }

    fn register_struct(&mut self, struct_type: &StructType) {
        for field in struct_type.fields() {
            match field.field_type.as_ref() {
                Type::Primitive(primitive) => {
                    if bounds_are_kept_for(primitive) {
                        self.bound_types.insert(field.id, primitive.clone());
                    }
                    if matches!(primitive, PrimitiveType::Float | PrimitiveType::Double) {
                        self.nan_tracked.push(field.id);
                    }
                }
                Type::Struct(inner) => self.register_struct(inner),
                _ => {}
            }
        }
    }

    pub(crate) fn observe_row(&mut self, schema: &Schema, row: Option<&Literal>) {
        self.row_count += 1;
        let values = match row {
            Some(Literal::Struct(fields)) => Some(fields),
            _ => None,
        };
        self.observe_struct(schema.as_struct(), values);
    }

    fn observe_struct(&mut self, struct_type: &StructType, values: Option<&Struct>) {
        for (index, field) in struct_type.fields().iter().enumerate() {
            let value = values.and_then(|struct_value| struct_value.iter().nth(index).flatten());
            self.observe_field(field, value);
        }
    }

    fn observe_field(&mut self, field: &NestedFieldRef, value: Option<&Literal>) {
        if !self.is_tracked(field.id) {
            return;
        }
        match value {
            None => {
                *self.null_counts.entry(field.id).or_insert(0) += 1;
            }
            Some(Literal::Primitive(primitive)) => {
                if let Some(primitive_type) = self.bound_types.get(&field.id).cloned() {
                    let datum = Datum::new(primitive_type, primitive.clone());
                    if datum.is_nan() {
                        *self.nan_counts.entry(field.id).or_insert(0) += 1;
                    } else {
                        self.update_bounds(field.id, datum);
                    }
                }
            }
            Some(_) => {}
        }
        if let Type::Struct(inner) = field.field_type.as_ref() {
            let nested = match value {
                Some(Literal::Struct(fields)) => Some(fields),
                _ => None,
            };
            self.observe_struct(inner, nested);
        }
    }

    fn is_tracked(&self, field_id: i32) -> bool {
        self.metrics.stats_eligible.contains(&field_id)
            && self.metrics.mode_for(field_id).collects_counts()
    }

    fn update_bounds(&mut self, field_id: i32, datum: Datum) {
        self.lower
            .entry(field_id)
            .and_modify(|current| {
                if *current > datum {
                    *current = datum.clone();
                }
            })
            .or_insert_with(|| datum.clone());
        self.upper
            .entry(field_id)
            .and_modify(|current| {
                if *current < datum {
                    *current = datum.clone();
                }
            })
            .or_insert(datum);
    }

    pub(crate) fn observe_stripe_column_sizes(
        &mut self,
        orc_schema: &OrcSchema,
        sizes_by_orc_index: &HashMap<usize, u64>,
    ) {
        for (orc_index, size) in sizes_by_orc_index {
            let Some(field_id) = orc_schema.columns[*orc_index].field_id else {
                continue;
            };
            if !self.is_tracked(field_id) {
                continue;
            }
            *self.column_sizes.entry(field_id).or_insert(0) += size;
        }
    }

    pub(crate) fn build(self) -> OrcColumnMetrics {
        let mut value_counts = HashMap::with_capacity(self.metrics.stats_eligible.len());
        for field_id in &self.metrics.stats_eligible {
            if self.metrics.mode_for(*field_id).collects_counts() {
                value_counts.insert(*field_id, self.row_count);
            }
        }

        let mut null_value_counts = self.null_counts;
        for field_id in value_counts.keys() {
            null_value_counts.entry(*field_id).or_insert(0);
        }

        let mut nan_value_counts = self.nan_counts;
        for field_id in &self.nan_tracked {
            if value_counts.contains_key(field_id) {
                nan_value_counts.entry(*field_id).or_insert(0);
            }
        }
        nan_value_counts.retain(|field_id, _| value_counts.contains_key(field_id));

        let mut lower_bounds = HashMap::with_capacity(self.lower.len());
        for (field_id, datum) in &self.lower {
            if let Some(bound) = self.metrics.mode_for(*field_id).truncate_lower_bound(datum) {
                lower_bounds.insert(*field_id, bound);
            }
        }
        let mut upper_bounds = HashMap::with_capacity(self.upper.len());
        for (field_id, datum) in &self.upper {
            if let Some(bound) = self.metrics.mode_for(*field_id).truncate_upper_bound(datum) {
                upper_bounds.insert(*field_id, bound);
            }
        }

        OrcColumnMetrics {
            column_sizes: self.column_sizes,
            value_counts,
            null_value_counts,
            nan_value_counts,
            lower_bounds,
            upper_bounds,
        }
    }
}

fn bounds_are_kept_for(primitive: &PrimitiveType) -> bool {
    !matches!(
        primitive,
        PrimitiveType::Binary
            | PrimitiveType::Fixed(_)
            | PrimitiveType::Uuid
            | PrimitiveType::Unknown
    )
}

#[cfg(test)]
mod tests {
    include!("metrics_tests.rs");
}
