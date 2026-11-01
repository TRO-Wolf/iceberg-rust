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

use std::collections::HashSet;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::nested_projection::MAX_NESTED_PROJECTION_DEPTH;
use crate::spec::{MappedField, NameMapping, NestedField, Schema, Type};
use crate::{Error, ErrorKind, Result};

enum Slot {
    Member(usize),
    Element,
    Key,
    Value,
}

impl Slot {
    fn mapping_name<'a>(&self, field: &'a Field) -> &'a str {
        match self {
            Slot::Member(_) => field.name(),
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
            .enumerate()
            .map(|(pos, child)| (child.as_ref(), Slot::Member(pos)))
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
    metadata.insert(PARQUET_FIELD_ID_META_KEY.to_string(), field_id.to_string());
    field
        .clone()
        .with_data_type(data_type)
        .with_metadata(metadata)
}

fn has_no_field_ids(arrow_schema: &ArrowSchema) -> bool {
    arrow_schema
        .fields()
        .iter()
        .next()
        .is_none_or(|f| f.metadata().get(PARQUET_FIELD_ID_META_KEY).is_none())
}

pub(crate) fn apply_name_mapping_to_arrow_schema(
    arrow_schema: ArrowSchemaRef,
    name_mapping: &NameMapping,
) -> Result<ArrowSchemaRef> {
    debug_assert!(
        has_no_field_ids(&arrow_schema),
        "Schema already has field IDs - name mapping should not be applied"
    );
    let fields = arrow_schema
        .fields()
        .iter()
        .map(|field| {
            let mapped = name_mapping
                .fields()
                .iter()
                .find(|mapped| mapped.names().iter().any(|name| name == field.name()));
            stamp_mapped(field, mapped, 0)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Arc::new(ArrowSchema::new_with_metadata(
        fields,
        arrow_schema.metadata().clone(),
    )))
}

fn stamp_mapped(field: &Field, mapped: Option<&MappedField>, depth: usize) -> Result<Field> {
    check_depth(depth)?;
    let Some((mapped, field_id)) = mapped.and_then(|m| m.field_id().map(|id| (m, id))) else {
        return Ok(field.clone());
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
            stamp_mapped(child, child_mapping, depth + 1)
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
        (Type::Struct(table_struct), Slot::Member(pos)) => table_struct
            .fields()
            .get(*pos)
            .filter(|table_field| table_field.name == *field.name())
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
    next_counter: i32,
}

impl FallbackIds<'_> {
    fn take_counter(&mut self) -> Result<i32> {
        let counter = self.next_counter;
        self.next_counter = counter.checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "file schema has too many fields for fallback field ids",
            )
        })?;
        Ok(counter)
    }

    fn claim(&mut self, field_id: i32) -> Option<i32> {
        self.claimed.insert(field_id).then_some(field_id)
    }

    fn skip_subtree(&mut self, field: &Field, depth: usize) -> Result<()> {
        check_depth(depth)?;
        for (child, _) in child_slots(field.data_type()) {
            self.take_counter()?;
            self.skip_subtree(child, depth + 1)?;
        }
        Ok(())
    }

    fn stamp_children(
        &mut self,
        field: &Field,
        field_id: i32,
        table_field: Option<&NestedField>,
        matched: bool,
        depth: usize,
    ) -> Result<Field> {
        let table_field = table_field.or_else(|| {
            self.table_schema
                .field_by_id(field_id)
                .map(|table_field| table_field.as_ref())
        });
        let mut children = Vec::new();
        for (child, slot) in child_slots(field.data_type()) {
            let counter = self.take_counter()?;
            let candidate = table_child(table_field, child, &slot);
            children.push(self.stamp_nested(child, candidate, matched, counter, depth + 1)?);
        }
        let data_type = with_children(field.data_type(), children);
        Ok(with_field_id(field, field_id, data_type))
    }

    fn stamp_nested(
        &mut self,
        field: &Field,
        candidate: Option<&NestedField>,
        parent_matched: bool,
        counter: i32,
        depth: usize,
    ) -> Result<Field> {
        check_depth(depth)?;
        let (field_id, matched) = match candidate {
            Some(table_field) if parent_matched => (Some(table_field.id), true),
            Some(table_field) => (self.claim(table_field.id), false),
            None => (self.claim(counter), false),
        };
        let Some(field_id) = field_id else {
            self.skip_subtree(field, depth)?;
            return Ok(field.clone());
        };
        let table_field = candidate.filter(|table_field| table_field.id == field_id);
        self.stamp_children(field, field_id, table_field, matched, depth)
    }
}

pub(crate) fn add_fallback_field_ids_to_arrow_schema(
    arrow_schema: &ArrowSchemaRef,
    table_schema: &Schema,
) -> Result<ArrowSchemaRef> {
    debug_assert!(
        has_no_field_ids(arrow_schema),
        "Schema already has field IDs"
    );
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
    let top_level_count = i32::try_from(arrow_schema.fields().len()).map_err(|_| too_many())?;
    let mut ids = FallbackIds {
        table_schema,
        claimed,
        next_counter: top_level_count.checked_add(1).ok_or_else(too_many)?,
    };
    let mut fields = Vec::with_capacity(arrow_schema.fields().len());
    for (pos, (field, table_field)) in arrow_schema.fields().iter().zip(&matches).enumerate() {
        let counter = i32::try_from(pos + 1).map_err(|_| too_many())?;
        let field_id = match table_field {
            Some(table_field) => Some(table_field.id),
            None => ids.claim(counter),
        };
        let stamped = match field_id {
            Some(field_id) => {
                ids.stamp_children(field, field_id, *table_field, table_field.is_some(), 0)?
            }
            None => {
                ids.skip_subtree(field, 0)?;
                field.as_ref().clone()
            }
        };
        fields.push(stamped);
    }
    Ok(Arc::new(ArrowSchema::new_with_metadata(
        fields,
        arrow_schema.metadata().clone(),
    )))
}
