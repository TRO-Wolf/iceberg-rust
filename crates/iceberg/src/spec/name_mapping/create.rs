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

use super::{MappedField, NameMapping};
use crate::spec::{NestedField, Schema, Type};
use crate::{Error, ErrorKind, Result};

const MAX_SCHEMA_DEPTH: usize = 100;

#[allow(missing_docs)]
pub fn create_name_mapping(schema: &Schema) -> Result<NameMapping> {
    Ok(NameMapping::new(mapped_struct_fields(
        schema.as_struct().fields(),
        0,
    )?))
}

fn mapped_struct_fields(
    fields: &[std::sync::Arc<NestedField>],
    depth: usize,
) -> Result<Vec<MappedField>> {
    check_depth(depth)?;
    let mut mapped = Vec::with_capacity(fields.len());
    for field in fields {
        mapped.push(MappedField::new(
            Some(field.id),
            vec![field.name.clone()],
            nested_mapping(&field.field_type, depth + 1)?,
        ));
    }
    Ok(mapped)
}

fn nested_mapping(field_type: &Type, depth: usize) -> Result<Vec<MappedField>> {
    check_depth(depth)?;
    match field_type {
        Type::Struct(struct_type) => mapped_struct_fields(struct_type.fields(), depth),
        Type::List(list_type) => Ok(vec![MappedField::new(
            Some(list_type.element_field.id),
            vec!["element".to_string()],
            nested_mapping(&list_type.element_field.field_type, depth + 1)?,
        )]),
        Type::Map(map_type) => Ok(vec![
            MappedField::new(
                Some(map_type.key_field.id),
                vec!["key".to_string()],
                nested_mapping(&map_type.key_field.field_type, depth + 1)?,
            ),
            MappedField::new(
                Some(map_type.value_field.id),
                vec!["value".to_string()],
                nested_mapping(&map_type.value_field.field_type, depth + 1)?,
            ),
        ]),
        Type::Primitive(_) | Type::Variant => Ok(Vec::new()),
    }
}

fn check_depth(depth: usize) -> Result<()> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Schema nesting exceeds the {MAX_SCHEMA_DEPTH}-level limit for name mapping"),
        ));
    }
    Ok(())
}
