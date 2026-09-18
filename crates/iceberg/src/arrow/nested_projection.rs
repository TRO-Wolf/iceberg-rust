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
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Int32Array, LargeListArray, ListArray, MapArray, RunArray,
    StructArray, new_null_array,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_cast::cast;
use arrow_schema::{DataType, Field, FieldRef, Fields};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::value::{create_primitive_array_repeated, create_primitive_array_single_element};
use crate::spec::{Literal, PrimitiveLiteral, Schema as IcebergSchema};
use crate::{Error, ErrorKind, Result};

pub(crate) const MAX_NESTED_PROJECTION_DEPTH: usize = 128;

pub(crate) fn nested_projection_applies(source: &DataType, target: &DataType) -> bool {
    matches!(
        (source, target),
        (DataType::Struct(_), DataType::Struct(_))
            | (
                DataType::List(_) | DataType::LargeList(_),
                DataType::List(_) | DataType::LargeList(_)
            )
            | (DataType::FixedSizeList(_, _), DataType::FixedSizeList(_, _))
            | (DataType::Map(_, _), DataType::Map(_, _))
    )
}

#[derive(Debug)]
pub(crate) struct NestedProjectionPlan {
    root: PlanNode,
}

impl NestedProjectionPlan {
    pub(crate) fn build(
        source_type: &DataType,
        target_type: &DataType,
        schema: &IcebergSchema,
        name: &str,
    ) -> Result<Self> {
        Ok(Self {
            root: PlanNode::build(source_type, target_type, schema, 0, name)?,
        })
    }

    pub(crate) fn apply(&mut self, source: ArrayRef) -> Result<ArrayRef> {
        let len = source.len();
        self.root.apply(Some(source), len)
    }
}

#[derive(Debug)]
struct StructChildPlan {
    source_index: Option<usize>,
    node: PlanNode,
}

#[derive(Debug)]
enum FillKind {
    Null,
    Constant(PrimitiveLiteral),
}

#[derive(Debug)]
enum PlanNode {
    Passthrough,
    Cast {
        target_type: DataType,
    },
    Fill {
        target_field: FieldRef,
        fill: FillKind,
        cached: Option<(usize, ArrayRef)>,
    },
    Struct {
        target_fields: Fields,
        children: Vec<StructChildPlan>,
    },
    List {
        target_element: FieldRef,
        element: Box<PlanNode>,
        from_large: bool,
    },
    LargeList {
        target_element: FieldRef,
        element: Box<PlanNode>,
        from_small: bool,
    },
    FixedSizeList {
        target_element: FieldRef,
        size: i32,
        element: Box<PlanNode>,
    },
    Map {
        target_entries: FieldRef,
        entries_children: Fields,
        ordered: bool,
        key: Box<PlanNode>,
        value: Box<PlanNode>,
    },
}

impl PlanNode {
    fn build(
        source_type: &DataType,
        target_type: &DataType,
        schema: &IcebergSchema,
        depth: usize,
        path: &str,
    ) -> Result<Self> {
        if depth > MAX_NESTED_PROJECTION_DEPTH {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("nested schema projection exceeds depth {MAX_NESTED_PROJECTION_DEPTH}"),
            ));
        }
        if source_type == target_type {
            return Ok(PlanNode::Passthrough);
        }
        match (source_type, target_type) {
            (DataType::Struct(source_fields), DataType::Struct(target_fields)) => {
                Self::build_struct(source_fields, target_fields, schema, depth, path)
            }
            (DataType::List(source_element), DataType::List(target_element)) => {
                Ok(PlanNode::List {
                    target_element: target_element.clone(),
                    element: Box::new(Self::build(
                        source_element.data_type(),
                        target_element.data_type(),
                        schema,
                        depth + 1,
                        &format!("{path}.{}", target_element.name()),
                    )?),
                    from_large: false,
                })
            }
            (DataType::LargeList(source_element), DataType::LargeList(target_element)) => {
                Ok(PlanNode::LargeList {
                    target_element: target_element.clone(),
                    element: Box::new(Self::build(
                        source_element.data_type(),
                        target_element.data_type(),
                        schema,
                        depth + 1,
                        &format!("{path}.{}", target_element.name()),
                    )?),
                    from_small: false,
                })
            }
            (DataType::List(source_element), DataType::LargeList(target_element)) => {
                Ok(PlanNode::LargeList {
                    target_element: target_element.clone(),
                    element: Box::new(Self::build(
                        source_element.data_type(),
                        target_element.data_type(),
                        schema,
                        depth + 1,
                        &format!("{path}.{}", target_element.name()),
                    )?),
                    from_small: true,
                })
            }
            (DataType::LargeList(source_element), DataType::List(target_element)) => {
                Ok(PlanNode::List {
                    target_element: target_element.clone(),
                    element: Box::new(Self::build(
                        source_element.data_type(),
                        target_element.data_type(),
                        schema,
                        depth + 1,
                        &format!("{path}.{}", target_element.name()),
                    )?),
                    from_large: true,
                })
            }
            (
                DataType::FixedSizeList(source_element, _),
                DataType::FixedSizeList(target_element, size),
            ) => Ok(PlanNode::FixedSizeList {
                target_element: target_element.clone(),
                size: *size,
                element: Box::new(Self::build(
                    source_element.data_type(),
                    target_element.data_type(),
                    schema,
                    depth + 1,
                    &format!("{path}.{}", target_element.name()),
                )?),
            }),
            (DataType::Map(source_entries, _), DataType::Map(target_entries, ordered)) => {
                let (source_key, source_value) = map_key_value_fields(source_entries)?;
                let (target_key, target_value) = map_key_value_fields(target_entries)?;
                let entries_children = match target_entries.data_type() {
                    DataType::Struct(fields) => fields.clone(),
                    _ => {
                        return Err(Error::new(
                            ErrorKind::DataInvalid,
                            "map entries field is not a struct".to_string(),
                        ));
                    }
                };
                Ok(PlanNode::Map {
                    target_entries: target_entries.clone(),
                    entries_children,
                    ordered: *ordered,
                    key: Box::new(Self::build(
                        source_key.data_type(),
                        target_key.data_type(),
                        schema,
                        depth + 1,
                        &format!("{path}.{}", target_key.name()),
                    )?),
                    value: Box::new(Self::build(
                        source_value.data_type(),
                        target_value.data_type(),
                        schema,
                        depth + 1,
                        &format!("{path}.{}", target_value.name()),
                    )?),
                })
            }
            _ => Ok(PlanNode::Cast {
                target_type: target_type.clone(),
            }),
        }
    }

    fn build_struct(
        source_fields: &Fields,
        target_fields: &Fields,
        schema: &IcebergSchema,
        depth: usize,
        path: &str,
    ) -> Result<Self> {
        if !source_fields.is_empty() && source_fields.iter().all(|f| field_id_of(f).is_none()) {
            let source_type = DataType::Struct(source_fields.clone());
            let target_type = DataType::Struct(target_fields.clone());
            return Ok(if source_type.equals_datatype(&target_type) {
                PlanNode::Passthrough
            } else {
                PlanNode::Cast { target_type }
            });
        }
        let mut source_by_id: HashMap<i32, usize> = HashMap::with_capacity(source_fields.len());
        let mut source_by_name: HashMap<&str, usize> = HashMap::new();
        for (pos, field) in source_fields.iter().enumerate() {
            if let Some(id) = field_id_of(field) {
                source_by_id.insert(id, pos);
            } else {
                source_by_name.entry(field.name().as_str()).or_insert(pos);
            }
        }
        let mut children = Vec::with_capacity(target_fields.len());
        for target_child in target_fields.iter() {
            let source_index =
                field_id_of(target_child).and_then(|id| source_by_id.get(&id).copied());
            let source_index = match source_index {
                Some(index) => Some(index),
                None => {
                    if source_by_name.contains_key(target_child.name().as_str()) {
                        return Err(Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "file mixes stamped and unstamped nested field ids in struct '{path}' at child '{}'; set a name mapping (schema.name-mapping.default) to resolve it",
                                target_child.name()
                            ),
                        ));
                    }
                    None
                }
            };
            let node = match source_index {
                Some(index) => Self::build(
                    source_fields[index].data_type(),
                    target_child.data_type(),
                    schema,
                    depth + 1,
                    &format!("{path}.{}", target_child.name()),
                )?,
                None => Self::build_fill(target_child, schema)?,
            };
            children.push(StructChildPlan { source_index, node });
        }
        Ok(PlanNode::Struct {
            target_fields: target_fields.clone(),
            children,
        })
    }

    fn build_fill(target_child: &FieldRef, schema: &IcebergSchema) -> Result<Self> {
        let iceberg_field = field_id_of(target_child).and_then(|id| schema.field_by_id(id));
        if let Some(default) = iceberg_field.and_then(|field| field.initial_default.as_ref()) {
            return match default {
                Literal::Primitive(primitive) => Ok(PlanNode::Fill {
                    target_field: target_child.clone(),
                    fill: FillKind::Constant(primitive.clone()),
                    cached: None,
                }),
                other => Err(Error::new(
                    ErrorKind::FeatureUnsupported,
                    format!(
                        "constant default for nested type is not supported in nested projection (got {other:?})"
                    ),
                )),
            };
        }
        if target_child.is_nullable() {
            return Ok(PlanNode::Fill {
                target_field: target_child.clone(),
                fill: FillKind::Null,
                cached: None,
            });
        }
        Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Missing required field: {}", target_child.name()),
        ))
    }

    fn apply(&mut self, source: Option<ArrayRef>, len: usize) -> Result<ArrayRef> {
        match self {
            PlanNode::Passthrough => source.ok_or_else(missing_source),
            PlanNode::Cast { target_type } => {
                let source = source.ok_or_else(missing_source)?;
                cast(source.as_ref(), target_type).map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("failed to cast nested column to {target_type}"),
                    )
                    .with_source(e)
                })
            }
            PlanNode::Fill {
                target_field,
                fill,
                cached,
            } => {
                if let Some((cached_len, array)) = cached
                    && *cached_len == len
                {
                    return Ok(array.clone());
                }
                let array = match fill {
                    FillKind::Null => new_null_array(target_field.data_type(), len),
                    FillKind::Constant(literal) => create_primitive_array_repeated(
                        target_field.data_type(),
                        &Some(literal.clone()),
                        len,
                    )?,
                };
                *cached = Some((len, array.clone()));
                Ok(array)
            }
            PlanNode::Struct {
                target_fields,
                children,
            } => {
                let source = source.ok_or_else(missing_source)?;
                let source_struct =
                    source
                        .as_any()
                        .downcast_ref::<StructArray>()
                        .ok_or_else(|| {
                            Error::new(
                                ErrorKind::DataInvalid,
                                "struct column is not a StructArray".to_string(),
                            )
                        })?;
                let len = source_struct.len();
                let mut columns: Vec<ArrayRef> = Vec::with_capacity(children.len());
                for child in children.iter_mut() {
                    let child_source = match child.source_index {
                        Some(index) => {
                            Some(source_struct.columns().get(index).cloned().ok_or_else(|| {
                                Error::new(
                                    ErrorKind::DataInvalid,
                                    "nested column index out of bounds".to_string(),
                                )
                            })?)
                        }
                        None => None,
                    };
                    columns.push(child.node.apply(child_source, len)?);
                }
                StructArray::try_new(
                    target_fields.clone(),
                    columns,
                    source_struct.nulls().cloned(),
                )
                .map(|array| Arc::new(array) as ArrayRef)
                .map_err(|e| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "failed to rebuild projected struct".to_string(),
                    )
                    .with_source(e)
                })
            }
            PlanNode::List {
                target_element,
                element,
                from_large,
            } => {
                let source = source.ok_or_else(missing_source)?;
                if *from_large {
                    let source_list = source
                        .as_any()
                        .downcast_ref::<LargeListArray>()
                        .ok_or_else(|| {
                            Error::new(
                                ErrorKind::DataInvalid,
                                "list column is not a LargeListArray".to_string(),
                            )
                        })?;
                    let values = element.apply(
                        Some(source_list.values().clone()),
                        source_list.values().len(),
                    )?;
                    ListArray::try_new(
                        target_element.clone(),
                        narrow_list_offsets(source_list.offsets())?,
                        values,
                        source_list.nulls().cloned(),
                    )
                    .map(|array| Arc::new(array) as ArrayRef)
                    .map_err(|e| {
                        Error::new(
                            ErrorKind::Unexpected,
                            "failed to rebuild projected list".to_string(),
                        )
                        .with_source(e)
                    })
                } else {
                    let source_list =
                        source.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                            Error::new(
                                ErrorKind::DataInvalid,
                                "list column is not a ListArray".to_string(),
                            )
                        })?;
                    let values = element.apply(
                        Some(source_list.values().clone()),
                        source_list.values().len(),
                    )?;
                    ListArray::try_new(
                        target_element.clone(),
                        source_list.offsets().clone(),
                        values,
                        source_list.nulls().cloned(),
                    )
                    .map(|array| Arc::new(array) as ArrayRef)
                    .map_err(|e| {
                        Error::new(
                            ErrorKind::Unexpected,
                            "failed to rebuild projected list".to_string(),
                        )
                        .with_source(e)
                    })
                }
            }
            PlanNode::LargeList {
                target_element,
                element,
                from_small,
            } => {
                let source = source.ok_or_else(missing_source)?;
                if *from_small {
                    let source_list =
                        source.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                            Error::new(
                                ErrorKind::DataInvalid,
                                "large list column is not a ListArray".to_string(),
                            )
                        })?;
                    let values = element.apply(
                        Some(source_list.values().clone()),
                        source_list.values().len(),
                    )?;
                    LargeListArray::try_new(
                        target_element.clone(),
                        widen_list_offsets(source_list.offsets()),
                        values,
                        source_list.nulls().cloned(),
                    )
                    .map(|array| Arc::new(array) as ArrayRef)
                    .map_err(|e| {
                        Error::new(
                            ErrorKind::Unexpected,
                            "failed to rebuild projected large list".to_string(),
                        )
                        .with_source(e)
                    })
                } else {
                    let source_list = source
                        .as_any()
                        .downcast_ref::<LargeListArray>()
                        .ok_or_else(|| {
                            Error::new(
                                ErrorKind::DataInvalid,
                                "large list column is not a LargeListArray".to_string(),
                            )
                        })?;
                    let values = element.apply(
                        Some(source_list.values().clone()),
                        source_list.values().len(),
                    )?;
                    LargeListArray::try_new(
                        target_element.clone(),
                        source_list.offsets().clone(),
                        values,
                        source_list.nulls().cloned(),
                    )
                    .map(|array| Arc::new(array) as ArrayRef)
                    .map_err(|e| {
                        Error::new(
                            ErrorKind::Unexpected,
                            "failed to rebuild projected large list".to_string(),
                        )
                        .with_source(e)
                    })
                }
            }
            PlanNode::FixedSizeList {
                target_element,
                size,
                element,
            } => {
                let source = source.ok_or_else(missing_source)?;
                let source_list = source
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "fixed size list column is not a FixedSizeListArray".to_string(),
                        )
                    })?;
                let values = element.apply(
                    Some(source_list.values().clone()),
                    source_list.values().len(),
                )?;
                FixedSizeListArray::try_new(
                    target_element.clone(),
                    *size,
                    values,
                    source_list.nulls().cloned(),
                )
                .map(|array| Arc::new(array) as ArrayRef)
                .map_err(|e| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "failed to rebuild projected fixed-size list".to_string(),
                    )
                    .with_source(e)
                })
            }
            PlanNode::Map {
                target_entries,
                entries_children,
                ordered,
                key,
                value,
            } => {
                let source = source.ok_or_else(missing_source)?;
                let source_map = source.as_any().downcast_ref::<MapArray>().ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        "map column is not a MapArray".to_string(),
                    )
                })?;
                let keys = key.apply(Some(source_map.keys().clone()), source_map.keys().len())?;
                let values =
                    value.apply(Some(source_map.values().clone()), source_map.values().len())?;
                let entries = StructArray::try_new(
                    entries_children.clone(),
                    vec![keys, values],
                    source_map.entries().nulls().cloned(),
                )
                .map_err(|e| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "failed to rebuild projected map entries".to_string(),
                    )
                    .with_source(e)
                })?;
                MapArray::try_new(
                    target_entries.clone(),
                    source_map.offsets().clone(),
                    entries,
                    source_map.nulls().cloned(),
                    *ordered,
                )
                .map(|array| Arc::new(array) as ArrayRef)
                .map_err(|e| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "failed to rebuild projected map".to_string(),
                    )
                    .with_source(e)
                })
            }
        }
    }
}

fn missing_source() -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        "nested projection expected a source array".to_string(),
    )
}

fn narrow_list_offsets(offsets: &OffsetBuffer<i64>) -> Result<OffsetBuffer<i32>> {
    let narrowed = offsets
        .iter()
        .map(|offset| {
            i32::try_from(*offset).map_err(|_| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "large list offsets exceed the i32 range".to_string(),
                )
            })
        })
        .collect::<Result<Vec<i32>>>()?;
    Ok(OffsetBuffer::new(ScalarBuffer::from(narrowed)))
}

fn widen_list_offsets(offsets: &OffsetBuffer<i32>) -> OffsetBuffer<i64> {
    let widened: ScalarBuffer<i64> = offsets.iter().map(|offset| i64::from(*offset)).collect();
    OffsetBuffer::new(widened)
}

fn map_key_value_fields(entries_field: &Field) -> Result<(FieldRef, FieldRef)> {
    let DataType::Struct(children) = entries_field.data_type() else {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            "map entries field is not a struct".to_string(),
        ));
    };
    let key = children.first().ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            "map entries field has no key child".to_string(),
        )
    })?;
    let value = children.get(1).ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            "map entries field has no value child".to_string(),
        )
    })?;
    Ok((key.clone(), value.clone()))
}

fn field_id_of(field: &Field) -> Option<i32> {
    field
        .metadata()
        .get(PARQUET_FIELD_ID_META_KEY)
        .and_then(|id| id.parse().ok())
}

pub(crate) fn create_constant_column(
    target_type: &DataType,
    prim_lit: &Option<PrimitiveLiteral>,
    num_rows: usize,
) -> Result<ArrayRef> {
    if let DataType::RunEndEncoded(_, values_field) = target_type {
        let create_ree_array = |values_array: ArrayRef| -> Result<ArrayRef> {
            let run_ends = if num_rows == 0 {
                Int32Array::from(Vec::<i32>::new())
            } else {
                Int32Array::from(vec![num_rows as i32])
            };
            Ok(Arc::new(
                RunArray::try_new(&run_ends, &values_array).map_err(|e| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "Failed to create RunArray for constant value",
                    )
                    .with_source(e)
                })?,
            ))
        };

        let values_array =
            create_primitive_array_single_element(values_field.data_type(), prim_lit)?;

        create_ree_array(values_array)
    } else {
        create_primitive_array_repeated(target_type, prim_lit, num_rows)
    }
}

#[cfg(test)]
mod test {
    include!("nested_projection_tests.rs");
    include!("nested_projection_evo_tests.rs");
    include!("nested_projection_evo_pin_tests.rs");
}
