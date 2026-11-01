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
use std::ops::Range;
use std::str::FromStr;

use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, DataType, Field, SchemaRef as ArrowSchemaRef};
use parquet::arrow::arrow_reader::{ArrowPredicateFn, RowFilter};
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::{PARQUET_FIELD_ID_META_KEY, ParquetRecordBatchStreamBuilder, ProjectionMask};
use parquet::basic::{ConvertedType, LogicalType};
use parquet::schema::types::{SchemaDescriptor, Type as ParquetType};

use crate::arrow::nested_projection::MAX_NESTED_PROJECTION_DEPTH;
use crate::arrow::record_batch_predicate::evaluate_predicate_to_mask;
use crate::expr::BoundPredicate;
use crate::metadata_columns::is_metadata_field;
use crate::scan::FileScanTask;
use crate::spec::{NestedField, Schema, Type};
use crate::{Error, ErrorKind, Result};

pub(crate) struct PushedRowFilter {
    pub(crate) filter: RowFilter,
    pub(crate) disable_predicate_cache: bool,
}

impl PushedRowFilter {
    pub(crate) fn apply_to_stream<T: AsyncFileReader + Send + 'static>(
        self,
        builder: ParquetRecordBatchStreamBuilder<T>,
    ) -> ParquetRecordBatchStreamBuilder<T> {
        let builder = builder.with_row_filter(self.filter);
        if self.disable_predicate_cache {
            builder.with_max_predicate_cache_size(0)
        } else {
            builder
        }
    }
}

pub(crate) enum RowFilterPlan {
    Push(PushedRowFilter),
    Residual,
}

pub(crate) fn plan_row_filter(
    predicates: &BoundPredicate,
    table_schema: &Schema,
    parquet_schema: &SchemaDescriptor,
    stamped_arrow_schema: &ArrowSchemaRef,
    iceberg_field_ids: &HashSet<i32>,
    force_residual: bool,
) -> Result<RowFilterPlan> {
    let absent: Vec<i32> = {
        let mut ids: Vec<i32> = iceberg_field_ids
            .iter()
            .copied()
            .filter(|id| !is_metadata_field(*id) && table_schema.field_by_id(*id).is_none())
            .collect();
        ids.sort_unstable();
        ids
    };
    if force_residual {
        if !absent.is_empty() {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "cannot apply equality deletes on dropped column ids {absent:?} through \
                     the residual read path, which resolves columns through the current table \
                     schema"
                ),
            ));
        }
        return Ok(RowFilterPlan::Residual);
    }
    let leaf_lists = match build_field_id_leaf_lists(parquet_schema)? {
        Some(lists) => lists,
        None => {
            if !absent.is_empty() {
                return Err(Error::new(
                    ErrorKind::FeatureUnsupported,
                    format!(
                        "cannot apply equality deletes on dropped column ids {absent:?}: the \
                         data file carries no embedded Parquet field ids to map them by"
                    ),
                ));
            }
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

    let disable_predicate_cache =
        pushed_mask_disables_predicate_cache(parquet_schema, &column_indices);
    let projection_mask = ProjectionMask::leaves(parquet_schema, column_indices);
    let predicate = predicates.clone();
    let predicate_func = move |batch: RecordBatch| {
        evaluate_predicate_to_mask(&predicate, &batch)
            .map_err(|e| ArrowError::ExternalError(Box::new(e)))
    };
    let arrow_predicate = ArrowPredicateFn::new(projection_mask, predicate_func);
    Ok(RowFilterPlan::Push(PushedRowFilter {
        filter: RowFilter::new(vec![Box::new(arrow_predicate)]),
        disable_predicate_cache,
    }))
}

pub(crate) fn dropped_equality_key_mask(
    task: &FileScanTask,
    parquet_schema: &SchemaDescriptor,
) -> Result<Option<ProjectionMask>> {
    let mut absent: Vec<i32> = task
        .deletes
        .iter()
        .filter_map(|delete| delete.equality_ids.as_ref())
        .flatten()
        .copied()
        .filter(|id| !is_metadata_field(*id) && task.schema.field_by_id(*id).is_none())
        .collect();
    absent.sort_unstable();
    absent.dedup();
    if absent.is_empty() {
        return Ok(None);
    }
    let Some(leaf_lists) = build_field_id_leaf_lists(parquet_schema)? else {
        return Err(Error::new(
            ErrorKind::FeatureUnsupported,
            format!(
                "cannot apply equality deletes on dropped column ids {absent:?} to data file \
                 '{}': the data file carries no embedded Parquet field ids to map them by",
                task.data_file_path
            ),
        ));
    };
    let leaves: Vec<usize> = absent
        .iter()
        .flat_map(|id| leaf_lists.get(id).into_iter().flatten().copied())
        .collect();
    Ok(Some(ProjectionMask::leaves(parquet_schema, leaves)))
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

pub(crate) fn validity_leaves(
    field_ids: &[i32],
    schema: &Schema,
    arrow_schema: &ArrowSchemaRef,
) -> Result<Vec<usize>> {
    let fields: Vec<&NestedField> = field_ids
        .iter()
        .filter_map(|field_id| schema.field_by_id(*field_id).map(|field| field.as_ref()))
        .collect();
    let mut borrowed = Vec::new();
    let mut holds_struct = false;
    for field in &fields {
        holds_struct |= type_holds_struct(&field.field_type, 0)?;
    }
    if !holds_struct {
        return Ok(borrowed);
    }
    let groups = index_file_groups(arrow_schema)?;
    let mut leafless_structs = Vec::new();
    for field in fields {
        collect_leafless_structs(field, 0, &groups, None, &mut leafless_structs)?;
    }
    for struct_id in leafless_structs {
        let Some(group) = groups.get(&struct_id) else {
            continue;
        };
        if !group.is_struct || group.leaves.is_empty() {
            continue;
        }
        if !group.leaves.clone().any(|leaf| borrowed.contains(&leaf)) {
            borrowed.push(group.leaves.start);
        }
    }
    Ok(borrowed)
}

fn type_holds_struct(field_type: &Type, depth: usize) -> Result<bool> {
    check_depth(depth)?;
    Ok(match field_type {
        Type::Primitive(_) | Type::Variant => false,
        Type::Struct(_) => true,
        Type::List(list_type) => type_holds_struct(&list_type.element_field.field_type, depth + 1)?,
        Type::Map(map_type) => {
            type_holds_struct(&map_type.key_field.field_type, depth + 1)?
                || type_holds_struct(&map_type.value_field.field_type, depth + 1)?
        }
    })
}

fn collect_leafless_structs(
    field: &NestedField,
    depth: usize,
    groups: &HashMap<i32, FileGroup>,
    within: Option<&Range<usize>>,
    leafless_structs: &mut Vec<i32>,
) -> Result<bool> {
    check_depth(depth)?;
    let Some(group) = groups.get(&field.id).filter(|group| {
        within
            .is_none_or(|range| range.start <= group.leaves.start && group.leaves.end <= range.end)
    }) else {
        return Ok(false);
    };
    let leaves = Some(&group.leaves);
    Ok(match field.field_type.as_ref() {
        Type::Primitive(_) | Type::Variant => true,
        Type::Struct(struct_type) => {
            let mut has_leaf = false;
            for child in struct_type.fields() {
                has_leaf |=
                    collect_leafless_structs(child, depth + 1, groups, leaves, leafless_structs)?;
            }
            if !has_leaf {
                leafless_structs.push(field.id);
            }
            has_leaf
        }
        Type::List(list_type) => collect_leafless_structs(
            &list_type.element_field,
            depth + 1,
            groups,
            leaves,
            leafless_structs,
        )?,
        Type::Map(map_type) => {
            let key = collect_leafless_structs(
                &map_type.key_field,
                depth + 1,
                groups,
                leaves,
                leafless_structs,
            )?;
            let value = collect_leafless_structs(
                &map_type.value_field,
                depth + 1,
                groups,
                leaves,
                leafless_structs,
            )?;
            key || value
        }
    })
}

struct FileGroup {
    is_struct: bool,
    leaves: Range<usize>,
}

fn index_file_groups(arrow_schema: &ArrowSchemaRef) -> Result<HashMap<i32, FileGroup>> {
    let mut groups = HashMap::new();
    let mut next_leaf = 0;
    for field in arrow_schema.fields() {
        index_field(field, 0, &mut next_leaf, &mut groups)?;
    }
    Ok(groups)
}

fn index_field(
    field: &Field,
    depth: usize,
    next_leaf: &mut usize,
    groups: &mut HashMap<i32, FileGroup>,
) -> Result<()> {
    check_depth(depth)?;
    let start = *next_leaf;
    let data_type = match field.data_type() {
        DataType::Dictionary(_, value) => value.as_ref(),
        DataType::RunEndEncoded(_, value) => value.data_type(),
        other => other,
    };
    match data_type {
        DataType::Struct(children) => {
            for child in children {
                index_field(child, depth + 1, next_leaf, groups)?;
            }
        }
        DataType::Union(children, _) => {
            for (_, child) in children.iter() {
                index_field(child, depth + 1, next_leaf, groups)?;
            }
        }
        DataType::List(child)
        | DataType::LargeList(child)
        | DataType::FixedSizeList(child, _)
        | DataType::Map(child, _) => index_field(child, depth + 1, next_leaf, groups)?,
        _ => *next_leaf += 1,
    }
    let field_id = field
        .metadata()
        .get(PARQUET_FIELD_ID_META_KEY)
        .and_then(|id| i32::from_str(id).ok());
    if let Some(field_id) = field_id {
        groups.insert(field_id, FileGroup {
            is_struct: matches!(data_type, DataType::Struct(_)),
            leaves: start..*next_leaf,
        });
    }
    Ok(())
}

fn check_depth(depth: usize) -> Result<()> {
    if depth > MAX_NESTED_PROJECTION_DEPTH {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("nested schema exceeds depth {MAX_NESTED_PROJECTION_DEPTH}"),
        ));
    }
    Ok(())
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

pub(crate) fn pushed_mask_disables_predicate_cache(
    parquet_schema: &SchemaDescriptor,
    column_indices: &[usize],
) -> bool {
    let wanted: HashSet<usize> = column_indices.iter().copied().collect();
    let mut leaf_idx = 0;
    parquet_schema
        .root_schema()
        .get_fields()
        .iter()
        .any(|root| {
            let start = leaf_idx;
            let count = leaf_count(root);
            leaf_idx += count;
            root.is_group() && count == 1 && !group_is_list(root) && wanted.contains(&start)
        })
}

fn group_is_list(ty: &ParquetType) -> bool {
    if !ty.is_group() {
        return false;
    }
    let basic_info = ty.get_basic_info();
    if let Some(logical) = basic_info.logical_type_ref() {
        return *logical == LogicalType::List;
    }
    basic_info.converted_type() == ConvertedType::LIST
}

pub(crate) fn leaf_count(ty: &ParquetType) -> usize {
    if ty.is_primitive() {
        1
    } else {
        ty.get_fields().iter().map(|f| leaf_count(f)).sum()
    }
}
