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

use arrow_array::{Array, ArrayRef, StructArray};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::spec::{Literal, Struct, StructType, Type};

pub(crate) fn repair_container_nulls(
    struct_type: &StructType,
    array: &StructArray,
    row: usize,
    value: Option<Literal>,
) -> Option<Literal> {
    let Some(Literal::Struct(fields)) = value else {
        return value;
    };
    let mut repaired: Vec<Option<Literal>> = fields.iter().map(|field| field.cloned()).collect();
    for (index, field) in struct_type.fields().iter().enumerate() {
        let Some(column) = column_for(array, field.id, index) else {
            continue;
        };
        let Some(slot) = repaired.get_mut(index) else {
            continue;
        };
        match field.field_type.as_ref() {
            Type::List(_) | Type::Map(_) if row < column.len() && column.is_null(row) => {
                *slot = None;
            }
            Type::Struct(inner) => {
                if let Some(child) = column.as_any().downcast_ref::<StructArray>() {
                    *slot = repair_container_nulls(inner, child, row, slot.take());
                }
            }
            _ => {}
        }
    }
    Some(Literal::Struct(Struct::from_iter(repaired)))
}

fn column_for(array: &StructArray, field_id: i32, index: usize) -> Option<&ArrayRef> {
    for (position, field) in array.fields().iter().enumerate() {
        let matches = field
            .metadata()
            .get(PARQUET_FIELD_ID_META_KEY)
            .and_then(|raw| raw.parse::<i32>().ok())
            == Some(field_id);
        if matches {
            return array.columns().get(position);
        }
    }
    array.columns().get(index)
}

pub(crate) fn schema_has_container(struct_type: &StructType) -> bool {
    struct_type
        .fields()
        .iter()
        .any(|field| match field.field_type.as_ref() {
            Type::List(_) | Type::Map(_) => true,
            Type::Struct(inner) => schema_has_container(inner),
            _ => false,
        })
}
