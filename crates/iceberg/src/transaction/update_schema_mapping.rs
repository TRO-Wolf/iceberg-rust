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

use crate::spec::{MappedField, NameMapping, NestedField, TableProperties, nested_mapping};
use crate::{Error, ErrorKind, Result, TableUpdate};

const TABLE_ROOT_ID: i32 = -1;

const MAX_MAPPING_DEPTH: usize = 100;

struct MappingChanges<'a> {
    renames: HashMap<i32, &'a str>,
    adds: HashMap<i32, Vec<&'a NestedField>>,
}

pub(crate) fn name_mapping_update(
    properties: &HashMap<String, String>,
    updates: &HashMap<i32, NestedField>,
    parent_to_added_ids: &HashMap<i32, Vec<i32>>,
    added_fields: &HashMap<i32, NestedField>,
) -> Option<TableUpdate> {
    let key = TableProperties::PROPERTY_DEFAULT_NAME_MAPPING;
    let json = properties.get(key)?;
    match updated_mapping_json(json, updates, parent_to_added_ids, added_fields) {
        Ok(Some(updated)) => Some(TableUpdate::SetProperties {
            updates: HashMap::from([(key.to_string(), updated)]),
        }),
        Ok(None) => None,
        Err(error) => {
            tracing::warn!(?error, property = key, "failed to update the name mapping");
            None
        }
    }
}

fn updated_mapping_json(
    json: &str,
    updates: &HashMap<i32, NestedField>,
    parent_to_added_ids: &HashMap<i32, Vec<i32>>,
    added_fields: &HashMap<i32, NestedField>,
) -> Result<Option<String>> {
    let mapping: NameMapping = serde_json::from_str(json).map_err(|error| {
        Error::new(ErrorKind::DataInvalid, "cannot parse the name mapping").with_source(error)
    })?;
    let mut adds: HashMap<i32, Vec<&NestedField>> = HashMap::new();
    for (parent_id, added_ids) in parent_to_added_ids {
        let fields = added_ids
            .iter()
            .map(|id| {
                added_fields.get(id).ok_or_else(|| {
                    Error::new(
                        ErrorKind::Unexpected,
                        format!("added field {id} is missing"),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        adds.insert(*parent_id, fields);
    }
    let changes = MappingChanges {
        renames: updates
            .iter()
            .map(|(id, field)| (*id, field.name.as_str()))
            .collect(),
        adds,
    };
    let updated = update_name_mapping(&mapping, &changes)?;
    if updated == mapping {
        return Ok(None);
    }
    serde_json::to_string(&updated).map(Some).map_err(|error| {
        Error::new(ErrorKind::Unexpected, "cannot serialize the name mapping").with_source(error)
    })
}

fn update_name_mapping(mapping: &NameMapping, changes: &MappingChanges<'_>) -> Result<NameMapping> {
    let fields = update_fields(mapping.fields().iter(), Some(TABLE_ROOT_ID), changes, 0)?;
    Ok(NameMapping::new(fields))
}

fn update_fields<'a>(
    fields: impl Iterator<Item = &'a MappedField>,
    parent_id: Option<i32>,
    changes: &MappingChanges<'_>,
    depth: usize,
) -> Result<Vec<MappedField>> {
    check_depth(depth)?;
    let mut updated = Vec::new();
    for field in fields {
        updated.push(update_field(field, changes, depth)?);
    }
    let updated = without_reassigned_names(updated, changes);
    add_new_fields(updated, parent_id, changes, depth)
}

fn without_reassigned_names(
    fields: Vec<MappedField>,
    changes: &MappingChanges<'_>,
) -> Vec<MappedField> {
    let assignments: HashMap<&str, i32> = fields
        .iter()
        .filter_map(MappedField::field_id)
        .filter_map(|id| changes.renames.get(&id).map(|name| (*name, id)))
        .collect();
    if assignments.is_empty() {
        return fields;
    }
    fields
        .into_iter()
        .map(|field| {
            let reassigned: HashSet<&str> = assignments
                .iter()
                .filter(|(_, id)| Some(**id) != field.field_id())
                .map(|(name, _)| *name)
                .collect();
            without_names(field, &reassigned)
        })
        .collect()
}

fn update_field(
    field: &MappedField,
    changes: &MappingChanges<'_>,
    depth: usize,
) -> Result<MappedField> {
    let mut names = field.names().to_vec();
    if let Some(new_name) = field.field_id().and_then(|id| changes.renames.get(&id))
        && !names.iter().any(|name| name == new_name)
    {
        names.push((*new_name).to_string());
    }
    let nested = update_fields(
        field.fields().iter().map(AsRef::as_ref),
        field.field_id(),
        changes,
        depth + 1,
    )?;
    Ok(MappedField::new(field.field_id(), names, nested))
}

fn add_new_fields(
    fields: Vec<MappedField>,
    parent_id: Option<i32>,
    changes: &MappingChanges<'_>,
    depth: usize,
) -> Result<Vec<MappedField>> {
    let Some(added) = parent_id.and_then(|id| changes.adds.get(&id)) else {
        return Ok(fields);
    };
    let added_names: HashSet<&str> = added.iter().map(|field| field.name.as_str()).collect();
    let mut updated: Vec<MappedField> = fields
        .into_iter()
        .map(|field| without_names(field, &added_names))
        .collect();
    for field in added {
        updated.push(MappedField::new(
            Some(field.id),
            vec![field.name.clone()],
            nested_mapping(&field.field_type, depth + 1)?,
        ));
    }
    Ok(updated)
}

fn without_names(field: MappedField, removed: &HashSet<&str>) -> MappedField {
    if !field
        .names()
        .iter()
        .any(|name| removed.contains(name.as_str()))
    {
        return field;
    }
    let names = field
        .names()
        .iter()
        .filter(|name| !removed.contains(name.as_str()))
        .cloned()
        .collect();
    let nested = field
        .fields()
        .iter()
        .map(|child| (**child).clone())
        .collect();
    MappedField::new(field.field_id(), names, nested)
}

fn check_depth(depth: usize) -> Result<()> {
    if depth > MAX_MAPPING_DEPTH {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("name mapping nesting exceeds the {MAX_MAPPING_DEPTH}-level limit"),
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "update_schema_mapping_tests.rs"]
mod tests;
