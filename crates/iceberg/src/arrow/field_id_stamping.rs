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

use crate::arrow::nested_projection::{LEFT_UNSTAMPED_META_KEY, MAX_NESTED_PROJECTION_DEPTH};
use crate::spec::{MappedField, NameMapping, NestedField, Schema, Type};
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
) -> Result<ArrowSchemaRef> {
    let fields = arrow_schema
        .fields()
        .iter()
        .map(|field| {
            let mapped = name_mapping
                .fields()
                .iter()
                .find(|mapped| mapped.names().iter().any(|name| name == field.name()));
            stamp_mapped(field, mapped, field.name(), 0)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Arc::new(ArrowSchema::new_with_metadata(
        fields,
        arrow_schema.metadata().clone(),
    )))
}

fn stamp_mapped(
    field: &Field,
    mapped: Option<&MappedField>,
    path: &str,
    depth: usize,
) -> Result<Field> {
    check_depth(depth)?;
    let Some((mapped, field_id)) = mapped.and_then(|m| m.field_id().map(|id| (m, id))) else {
        return Ok(match depth {
            0 => field.clone(),
            _ => left_unstamped(field),
        });
    };
    let children = child_slots(field.data_type())
        .into_iter()
        .map(|(child, slot)| {
            let name = slot.mapping_name(child);
            let child_mapping = mapped
                .fields()
                .iter()
                .map(|f| f.as_ref())
                .find(|f| f.names().iter().any(|n| n == name));
            let kind = match slot {
                Slot::Member => None,
                Slot::Element => Some("list"),
                Slot::Key | Slot::Value => Some("map"),
            };
            if let Some(kind) = kind
                && child_mapping.and_then(MappedField::field_id).is_none()
            {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "the name mapping names {kind} '{path}' but has no '{name}' entry with a field id for it"
                    ),
                ));
            }
            stamp_mapped(child, child_mapping, &format!("{path}.{name}"), depth + 1)
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

pub(crate) fn add_fallback_field_ids_to_arrow_schema(
    arrow_schema: &ArrowSchemaRef,
    table_schema: &Schema,
) -> Result<ArrowSchemaRef> {
    let too_many = || {
        Error::new(
            ErrorKind::DataInvalid,
            "file schema has too many top-level fields for fallback field ids",
        )
    };
    let table_top_level_fields = table_schema.as_struct().fields();
    let matches: Vec<Option<&NestedField>> = arrow_schema
        .fields()
        .iter()
        .enumerate()
        .map(|(pos, field)| {
            table_top_level_fields
                .get(pos)
                .filter(|table_field| field.name() == table_field.name.as_str())
                .map(|table_field| table_field.as_ref())
        })
        .collect();
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
    for (pos, (field, table_field)) in arrow_schema.fields().iter().zip(&matches).enumerate() {
        let stamped = match table_field {
            Some(table_field) => {
                ids.stamp_children(field, table_field.id, Some(table_field), true, 0)?
            }
            None => {
                let counter = i32::try_from(pos + 1).map_err(|_| too_many())?;
                match ids.claim(counter) {
                    Some(field_id) => ids.stamp_children(field, field_id, None, false, 0)?,
                    None => field.as_ref().clone(),
                }
            }
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
