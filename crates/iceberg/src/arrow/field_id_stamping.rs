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
use std::sync::Arc;

use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::arrow_type_to_type;
use crate::arrow::nested_projection::{LEFT_UNSTAMPED_META_KEY, MAX_NESTED_PROJECTION_DEPTH};
use crate::arrow::reader::CollectFieldIdVisitor;
use crate::arrow::row_filter_plan::top_level_ancestor_id;
use crate::expr::visitors::bound_predicate_visitor::visit;
use crate::scan::FileScanTask;
use crate::spec::{
    MappedField, NameMapping, NestedField, PrimitiveType, Schema, Type, is_promotion_allowed,
};
use crate::{Error, ErrorKind, Result};

enum Slot {
    Member,
    Element,
    Key,
    Value,
}

impl Slot {
    fn mapping_name<'a>(&self, field: &'a Field) -> &'a str {
        match self {
            Slot::Member => field.name(),
            Slot::Element => "element",
            Slot::Key => "key",
            Slot::Value => "value",
        }
    }
}

fn check_depth(depth: usize) -> Result<()> {
    if depth > MAX_NESTED_PROJECTION_DEPTH {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("nested file schema exceeds depth {MAX_NESTED_PROJECTION_DEPTH}"),
        ));
    }
    Ok(())
}

fn child_slots(data_type: &DataType) -> Vec<(&Field, Slot)> {
    match data_type {
        DataType::Struct(children) => children
            .iter()
            .map(|child| (child.as_ref(), Slot::Member))
            .collect(),
        DataType::List(element)
        | DataType::LargeList(element)
        | DataType::FixedSizeList(element, _) => vec![(element.as_ref(), Slot::Element)],
        DataType::Map(entries, _) => match entries.data_type() {
            DataType::Struct(key_value) => {
                let mut iter = key_value.iter();
                match (iter.next(), iter.next(), iter.next()) {
                    (Some(key), Some(value), None) => {
                        vec![(key.as_ref(), Slot::Key), (value.as_ref(), Slot::Value)]
                    }
                    _ => Vec::new(),
                }
            }
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

fn with_children(data_type: &DataType, children: Vec<Field>) -> DataType {
    let mut iter = children.into_iter();
    match data_type {
        DataType::Struct(_) => DataType::Struct(Fields::from(iter.collect::<Vec<_>>())),
        DataType::List(_) => match iter.next() {
            Some(element) => DataType::List(Arc::new(element)),
            None => data_type.clone(),
        },
        DataType::LargeList(_) => match iter.next() {
            Some(element) => DataType::LargeList(Arc::new(element)),
            None => data_type.clone(),
        },
        DataType::FixedSizeList(_, size) => match iter.next() {
            Some(element) => DataType::FixedSizeList(Arc::new(element), *size),
            None => data_type.clone(),
        },
        DataType::Map(entries, sorted) => match (iter.next(), iter.next()) {
            (Some(key), Some(value)) => DataType::Map(
                Arc::new(
                    entries
                        .as_ref()
                        .clone()
                        .with_data_type(DataType::Struct(Fields::from(vec![key, value]))),
                ),
                *sorted,
            ),
            _ => data_type.clone(),
        },
        _ => data_type.clone(),
    }
}

fn with_field_id(field: &Field, field_id: i32, data_type: DataType) -> Field {
    let mut metadata = field.metadata().clone();
    metadata
        .entry(PARQUET_FIELD_ID_META_KEY.to_string())
        .or_insert_with(|| field_id.to_string());
    field
        .clone()
        .with_data_type(data_type)
        .with_metadata(metadata)
}

fn left_unstamped(field: &Field) -> Field {
    if field.metadata().contains_key(PARQUET_FIELD_ID_META_KEY) {
        return field.clone();
    }
    let mut metadata = field.metadata().clone();
    metadata.insert(LEFT_UNSTAMPED_META_KEY.to_string(), "true".to_string());
    field.clone().with_metadata(metadata)
}

pub(crate) fn file_has_field_ids(arrow_schema: &ArrowSchema) -> bool {
    let mut pending: Vec<&Field> = arrow_schema.fields().iter().map(|f| f.as_ref()).collect();
    while let Some(field) = pending.pop() {
        if field.metadata().contains_key(PARQUET_FIELD_ID_META_KEY) {
            return true;
        }
        pending.extend(
            child_slots(field.data_type())
                .into_iter()
                .map(|(child, _)| child),
        );
    }
    false
}

pub(crate) fn apply_name_mapping_to_arrow_schema(
    arrow_schema: ArrowSchemaRef,
    name_mapping: &NameMapping,
    projected_field_ids: &[i32],
) -> Result<ArrowSchemaRef> {
    let matches: Vec<Option<&MappedField>> = arrow_schema
        .fields()
        .iter()
        .map(|field| {
            name_mapping
                .fields()
                .iter()
                .find(|mapped| mapped.names().iter().any(|name| name == field.name()))
        })
        .collect();
    let fields = arrow_schema
        .fields()
        .iter()
        .zip(last_match_per_id(&matches))
        .map(|(field, mapped)| {
            stamp_mapped(field, mapped, field.name(), 0, projected_field_ids, false)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Arc::new(ArrowSchema::new_with_metadata(
        fields,
        arrow_schema.metadata().clone(),
    )))
}

fn last_match_per_id<'a>(matches: &[Option<&'a MappedField>]) -> Vec<Option<&'a MappedField>> {
    let mut seen = HashSet::new();
    let mut kept: Vec<Option<&MappedField>> = matches
        .iter()
        .rev()
        .map(|mapped| match mapped.and_then(MappedField::field_id) {
            Some(field_id) if !seen.insert(field_id) => None,
            _ => *mapped,
        })
        .collect();
    kept.reverse();
    kept
}

fn stamp_mapped(
    field: &Field,
    mapped: Option<&MappedField>,
    path: &str,
    depth: usize,
    projected_field_ids: &[i32],
    projected: bool,
) -> Result<Field> {
    check_depth(depth)?;
    let Some((mapped, field_id)) = mapped.and_then(|m| m.field_id().map(|id| (m, id))) else {
        return Ok(match depth {
            0 => field.clone(),
            _ => left_unstamped(field),
        });
    };
    let projected = projected || projected_field_ids.contains(&field_id);
    let slots = child_slots(field.data_type());
    let matches: Vec<Option<&MappedField>> = slots
        .iter()
        .map(|(child, slot)| {
            let name = slot.mapping_name(child);
            mapped
                .fields()
                .iter()
                .map(|f| f.as_ref())
                .find(|f| f.names().iter().any(|n| n == name))
        })
        .collect();
    let children = slots
        .into_iter()
        .zip(matches.iter().zip(last_match_per_id(&matches)))
        .map(|((child, slot), (child_match, child_mapping))| {
            let name = slot.mapping_name(child);
            let kind = match slot {
                Slot::Member => None,
                Slot::Element => Some("list"),
                Slot::Key | Slot::Value => Some("map"),
            };
            if let Some(kind) = kind
                && projected
                && child_match.and_then(MappedField::field_id).is_none()
            {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "the name mapping names {kind} '{path}' but has no '{name}' entry with a field id for it"
                    ),
                ));
            }
            stamp_mapped(
                child,
                child_mapping,
                &format!("{path}.{name}"),
                depth + 1,
                projected_field_ids,
                projected,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let data_type = with_children(field.data_type(), children);
    Ok(with_field_id(field, field_id, data_type))
}

fn table_child<'a>(
    parent: Option<&'a NestedField>,
    field: &Field,
    slot: &Slot,
) -> Option<&'a NestedField> {
    match (parent?.field_type.as_ref(), slot) {
        (Type::Struct(table_struct), Slot::Member) => table_struct
            .field_by_name(field.name())
            .map(|table_field| table_field.as_ref()),
        (Type::List(list), Slot::Element) => Some(list.element_field.as_ref()),
        (Type::Map(map), Slot::Key) => Some(map.key_field.as_ref()),
        (Type::Map(map), Slot::Value) => Some(map.value_field.as_ref()),
        _ => None,
    }
}

fn claim_matches(
    field: &Field,
    table_field: &NestedField,
    depth: usize,
    claimed: &mut HashSet<i32>,
) -> Result<()> {
    check_depth(depth)?;
    claimed.insert(table_field.id);
    for (child, slot) in child_slots(field.data_type()) {
        if let Some(table_child) = table_child(Some(table_field), child, &slot) {
            claim_matches(child, table_child, depth + 1, claimed)?;
        }
    }
    Ok(())
}

struct FallbackIds<'a> {
    table_schema: &'a Schema,
    claimed: HashSet<i32>,
}

impl FallbackIds<'_> {
    fn claim(&mut self, field_id: i32) -> Option<i32> {
        self.claimed.insert(field_id).then_some(field_id)
    }

    fn stamp_children(
        &mut self,
        field: &Field,
        field_id: i32,
        table_field: Option<&NestedField>,
        preclaimed: bool,
        depth: usize,
    ) -> Result<Field> {
        check_depth(depth)?;
        let table_field = table_field.or_else(|| {
            self.table_schema
                .field_by_id(field_id)
                .map(|table_field| table_field.as_ref())
        });
        let mut children = Vec::new();
        for (child, slot) in child_slots(field.data_type()) {
            let stamped = match table_child(table_field, child, &slot) {
                Some(table_child) => {
                    let child_id = if preclaimed {
                        Some(table_child.id)
                    } else {
                        self.claim(table_child.id)
                    };
                    match child_id {
                        Some(child_id) => self.stamp_children(
                            child,
                            child_id,
                            Some(table_child),
                            preclaimed,
                            depth + 1,
                        )?,
                        None => left_unstamped(child),
                    }
                }
                None => left_unstamped(child),
            };
            children.push(stamped);
        }
        let data_type = with_children(field.data_type(), children);
        Ok(with_field_id(field, field_id, data_type))
    }
}

fn nested_kind(data_type: &DataType) -> Option<&'static str> {
    match data_type {
        DataType::Struct(_) => Some("struct"),
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _) => Some("list"),
        DataType::Map(_, _) => Some("map"),
        _ => None,
    }
}

fn is_timestamp(primitive: &PrimitiveType) -> bool {
    matches!(
        primitive,
        PrimitiveType::Timestamp
            | PrimitiveType::Timestamptz
            | PrimitiveType::TimestampNs
            | PrimitiveType::TimestamptzNs
    )
}

fn positional_type_change(file_type: &DataType, table_type: &Type) -> Option<String> {
    match (nested_kind(file_type), table_type) {
        (Some(kind), Type::Primitive(_)) => Some(kind.to_string()),
        (Some(_), _) | (None, Type::Variant) => None,
        (None, Type::Primitive(table_primitive)) => {
            let Ok(Type::Primitive(file_primitive)) = arrow_type_to_type(file_type) else {
                return None;
            };
            let readable =
                is_promotion_allowed(&Type::Primitive(file_primitive.clone()), table_primitive)
                    || matches!(
                        (&file_primitive, table_primitive),
                        (PrimitiveType::Fixed(16), PrimitiveType::Uuid)
                    )
                    || (is_timestamp(&file_primitive) && is_timestamp(table_primitive));
            (!readable).then(|| file_primitive.to_string())
        }
        (None, _) => Some(
            arrow_type_to_type(file_type)
                .map(|file_type| file_type.to_string())
                .unwrap_or_else(|_| file_type.to_string()),
        ),
    }
}

pub(crate) fn guarded_field_ids(task: &FileScanTask) -> Result<Vec<i32>> {
    let mut field_ids = task.project_field_ids().to_vec();
    if let Some(predicate) = task.predicate.as_deref() {
        let mut collector = CollectFieldIdVisitor::default();
        visit(&mut collector, predicate)?;
        field_ids.extend(collector.field_ids());
    }
    Ok(field_ids)
}

fn refuse_positional_type_change(
    position: usize,
    field: &Field,
    table_field: &NestedField,
    table_schema: &Schema,
    guarded_field_ids: &[i32],
) -> Result<()> {
    let guarded = guarded_field_ids
        .iter()
        .any(|id| top_level_ancestor_id(table_schema, *id) == Some(table_field.id));
    if !guarded {
        return Ok(());
    }
    let Some(file_type) = positional_type_change(field.data_type(), &table_field.field_type) else {
        return Ok(());
    };
    Err(Error::new(
        ErrorKind::DataInvalid,
        format!(
            "data file column '{}' at position {position} has type {file_type} and cannot be read as table field '{}' (id {}) of type {}: the data file has no field ids and the table has no name mapping, so its columns are read by position",
            field.name(),
            table_field.name,
            table_field.id,
            table_field.field_type,
        ),
    ))
}

pub(crate) fn add_fallback_field_ids_to_arrow_schema(
    arrow_schema: &ArrowSchemaRef,
    table_schema: &Schema,
    guarded_field_ids: &[i32],
) -> Result<ArrowSchemaRef> {
    let too_many = || {
        Error::new(
            ErrorKind::DataInvalid,
            "file schema has too many top-level fields for fallback field ids",
        )
    };
    let positional_ids = (1..=arrow_schema.fields().len())
        .map(|id| i32::try_from(id).map_err(|_| too_many()))
        .collect::<Result<Vec<_>>>()?;
    let matches: Vec<Option<&NestedField>> = positional_ids
        .iter()
        .map(|id| {
            table_schema
                .as_struct()
                .field_by_id(*id)
                .map(|table_field| table_field.as_ref())
        })
        .collect();
    for (pos, (field, table_field)) in arrow_schema.fields().iter().zip(&matches).enumerate() {
        if let Some(table_field) = table_field {
            refuse_positional_type_change(
                pos,
                field,
                table_field,
                table_schema,
                guarded_field_ids,
            )?;
        }
    }
    let mut claimed = HashSet::new();
    for (field, table_field) in arrow_schema.fields().iter().zip(&matches) {
        if let Some(table_field) = table_field {
            claim_matches(field, table_field, 0, &mut claimed)?;
        }
    }
    let mut ids = FallbackIds {
        table_schema,
        claimed,
    };
    let mut fields = Vec::with_capacity(arrow_schema.fields().len());
    for ((field, table_field), field_id) in arrow_schema
        .fields()
        .iter()
        .zip(&matches)
        .zip(positional_ids)
    {
        let stamped = match table_field {
            Some(table_field) => {
                ids.stamp_children(field, table_field.id, Some(table_field), true, 0)?
            }
            None if table_schema.field_by_id(field_id).is_some() => field.as_ref().clone(),
            None => ids.stamp_children(field, field_id, None, false, 0)?,
        };
        fields.push(stamped);
    }
    Ok(Arc::new(ArrowSchema::new_with_metadata(
        fields,
        arrow_schema.metadata().clone(),
    )))
}

pub(crate) fn stamped_field_id_map(arrow_schema: &ArrowSchemaRef) -> HashMap<i32, usize> {
    let mut column_map = HashMap::new();
    arrow_schema.fields().filter_leaves(|idx, field| {
        if let Some(field_id) = field
            .metadata()
            .get(PARQUET_FIELD_ID_META_KEY)
            .and_then(|id| id.parse::<i32>().ok())
        {
            column_map.insert(field_id, idx);
        }
        false
    });
    column_map
}
