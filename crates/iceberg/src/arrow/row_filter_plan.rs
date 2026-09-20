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

use std::collections::{HashMap, HashSet};
use std::str::FromStr;

use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, DataType, SchemaRef as ArrowSchemaRef};
use parquet::arrow::arrow_reader::{ArrowPredicateFn, RowFilter};
use parquet::arrow::{PARQUET_FIELD_ID_META_KEY, ProjectionMask};
use parquet::schema::types::{SchemaDescriptor, Type as ParquetType};

use crate::arrow::record_batch_predicate::evaluate_predicate_to_mask;
use crate::error::Result;
use crate::expr::BoundPredicate;
use crate::spec::{NestedField, Schema, Type};

pub(crate) enum RowFilterPlan {
    Push(RowFilter),
    Residual,
}

pub(crate) fn plan_row_filter(
    predicates: &BoundPredicate,
    table_schema: &Schema,
    parquet_schema: &SchemaDescriptor,
    stamped_arrow_schema: &ArrowSchemaRef,
    iceberg_field_ids: &HashSet<i32>,
) -> Result<RowFilterPlan> {
    let leaf_lists = match build_field_id_leaf_lists(parquet_schema)? {
        Some(lists) => lists,
        None => {
            let stamped = stamped_top_level_leaf_lists(parquet_schema, stamped_arrow_schema);
            let present_but_unmapped = iceberg_field_ids.iter().any(|field_id| {
                !stamped.contains_key(field_id)
                    && top_level_ancestor_id(table_schema, *field_id)
                        .is_some_and(|top_id| stamped.contains_key(&top_id))
            });
            if present_but_unmapped {
                return Ok(RowFilterPlan::Residual);
            }
            stamped
        }
    };

    let mut column_indices = iceberg_field_ids
        .iter()
        .flat_map(|field_id| leaf_lists.get(field_id).into_iter().flatten().copied())
        .collect::<Vec<_>>();
    column_indices.sort_unstable();
    column_indices.dedup();

    let projection_mask = ProjectionMask::leaves(parquet_schema, column_indices);
    let predicate = predicates.clone();
    let predicate_func = move |batch: RecordBatch| {
        evaluate_predicate_to_mask(&predicate, &batch)
            .map_err(|e| ArrowError::ExternalError(Box::new(e)))
    };
    let arrow_predicate = ArrowPredicateFn::new(projection_mask, predicate_func);
    Ok(RowFilterPlan::Push(RowFilter::new(vec![Box::new(
        arrow_predicate,
    )])))
}

pub(crate) fn top_level_ancestor_id(schema: &Schema, field_id: i32) -> Option<i32> {
    fn contains(field: &NestedField, target: i32) -> bool {
        if field.id == target {
            return true;
        }
        match field.field_type.as_ref() {
            Type::Struct(struct_type) => struct_type.fields().iter().any(|f| contains(f, target)),
            Type::List(list_type) => contains(&list_type.element_field, target),
            Type::Map(map_type) => {
                contains(&map_type.key_field, target) || contains(&map_type.value_field, target)
            }
            Type::Primitive(_) | Type::Variant => false,
        }
    }
    schema
        .as_struct()
        .fields()
        .iter()
        .find(|field| contains(field, field_id))
        .map(|field| field.id)
}

pub(crate) fn unmapped_group_leaf_indices(
    leaf_field_ids: &[i32],
    iceberg_schema_of_task: &Schema,
    parquet_schema: &SchemaDescriptor,
    arrow_schema: &ArrowSchemaRef,
) -> Vec<usize> {
    let mut top_level_ids = HashSet::new();
    for leaf_id in leaf_field_ids {
        if let Some(top_id) = top_level_ancestor_id(iceberg_schema_of_task, *leaf_id) {
            top_level_ids.insert(top_id);
        }
    }
    let root_fields = parquet_schema.root_schema().get_fields();
    let mut indices = vec![];
    let mut leaf_idx = 0;
    for (pos, arrow_field) in arrow_schema.fields().iter().enumerate() {
        let count = root_fields
            .get(pos)
            .map(|field| leaf_count(field))
            .unwrap_or(0);
        let is_group = matches!(
            arrow_field.data_type(),
            DataType::Struct(_)
                | DataType::List(_)
                | DataType::LargeList(_)
                | DataType::FixedSizeList(_, _)
                | DataType::Map(_, _)
        );
        if is_group {
            let stamped = arrow_field
                .metadata()
                .get(PARQUET_FIELD_ID_META_KEY)
                .and_then(|id| i32::from_str(id).ok());
            if stamped.is_some_and(|id| top_level_ids.contains(&id)) {
                indices.extend(leaf_idx..leaf_idx + count);
            }
        }
        leaf_idx += count;
    }
    indices
}

fn build_field_id_leaf_lists(
    parquet_schema: &SchemaDescriptor,
) -> Result<Option<HashMap<i32, Vec<usize>>>> {
    fn walk(
        ty: &ParquetType,
        leaf_idx: &mut usize,
        map: &mut HashMap<i32, Vec<usize>>,
    ) -> Result<bool> {
        let start = *leaf_idx;
        match ty {
            ParquetType::PrimitiveType { basic_info, .. } => {
                if !basic_info.has_id() {
                    return Ok(false);
                }
                map.insert(basic_info.id(), vec![*leaf_idx]);
                *leaf_idx += 1;
            }
            ParquetType::GroupType { basic_info, .. } => {
                for field in ty.get_fields() {
                    if !walk(field, leaf_idx, map)? {
                        return Ok(false);
                    }
                }
                if basic_info.has_id() {
                    map.insert(basic_info.id(), (start..*leaf_idx).collect());
                }
            }
        }
        Ok(true)
    }

    let mut map = HashMap::new();
    let mut leaf_idx = 0;
    for field in parquet_schema.root_schema().get_fields() {
        if !walk(field, &mut leaf_idx, &mut map)? {
            return Ok(None);
        }
    }
    Ok(Some(map))
}

fn stamped_top_level_leaf_lists(
    parquet_schema: &SchemaDescriptor,
    stamped_arrow_schema: &ArrowSchemaRef,
) -> HashMap<i32, Vec<usize>> {
    let mut map = HashMap::new();
    let root_fields = parquet_schema.root_schema().get_fields();
    let mut leaf_idx = 0;
    for (pos, arrow_field) in stamped_arrow_schema.fields().iter().enumerate() {
        let count = root_fields
            .get(pos)
            .map(|field| leaf_count(field))
            .unwrap_or(0);
        if let Some(field_id) = arrow_field
            .metadata()
            .get(PARQUET_FIELD_ID_META_KEY)
            .and_then(|id| i32::from_str(id).ok())
        {
            map.insert(field_id, (leaf_idx..leaf_idx + count).collect());
        }
        leaf_idx += count;
    }
    map
}

pub(crate) fn leaf_count(ty: &ParquetType) -> usize {
    if ty.is_primitive() {
        1
    } else {
        ty.get_fields().iter().map(|f| leaf_count(f)).sum()
    }
}
