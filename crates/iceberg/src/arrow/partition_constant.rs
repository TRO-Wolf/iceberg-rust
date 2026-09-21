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

use std::sync::Arc;

use arrow_array::builder::StructBuilder;
use arrow_array::{ArrayRef, StructArray};
use arrow_schema::DataType;

use crate::arrow::record_batch_transformer::ColumnSource;
use crate::arrow::type_to_arrow_type;
use crate::inspect::partition_values::append_partition;
use crate::metadata_columns::RESERVED_FIELD_ID_PARTITION;
use crate::spec::{
    NestedFieldRef, PartitionSpec, PartitionSpecRef, Schema, Struct, StructType, Type,
};
use crate::{Error, ErrorKind, Result};

pub(crate) fn unified_partition_type(
    specs: &[PartitionSpecRef],
    schema: &Schema,
) -> Result<StructType> {
    let mut ordered: Vec<&PartitionSpec> = specs.iter().map(Arc::as_ref).collect();
    ordered.sort_by_key(|spec| spec.spec_id());
    let mut fields: Vec<NestedFieldRef> = Vec::new();
    for spec in ordered {
        let partition_type = spec.partition_type(schema).map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Failed to compute the partition type of spec {} for the _partition column",
                    spec.spec_id()
                ),
            )
            .with_source(e)
        })?;
        for field in partition_type.fields() {
            if !fields.iter().any(|seen| seen.name == field.name) {
                fields.push(field.clone());
            }
        }
    }
    Ok(StructType::new(fields))
}

pub(crate) fn coerce_partition_value(
    partition_union: &StructType,
    spec: Option<&PartitionSpec>,
    tuple: Option<&Struct>,
) -> Option<Struct> {
    if partition_union.fields().is_empty() {
        return None;
    }
    let (spec, tuple) = match (spec, tuple) {
        (Some(spec), Some(tuple)) => (spec, tuple),
        _ => return None,
    };
    Some(
        partition_union
            .fields()
            .iter()
            .map(|slot| {
                let pos = spec
                    .fields()
                    .iter()
                    .position(|field| field.name == slot.name)?;
                match tuple.fields().get(pos) {
                    Some(literal) => literal.clone(),
                    None => {
                        tracing::warn!(
                            spec_id = spec.spec_id(),
                            position = pos,
                            tuple_len = tuple.fields().len(),
                            slot = slot.name.as_str(),
                            "partition tuple is shorter than its partition spec; resolving the \
                             _partition slot as null"
                        );
                        None
                    }
                }
            })
            .collect(),
    )
}

pub(crate) fn partition_value_for_task(
    task_schema: &Schema,
    spec: Option<&PartitionSpec>,
    tuple: Option<&Struct>,
) -> Result<Option<Struct>> {
    let partition_union = partition_union_from_schema(task_schema)?;
    Ok(coerce_partition_value(partition_union, spec, tuple))
}

pub(crate) fn partition_constant_source(
    snapshot_schema: &Schema,
    value: Option<Option<Struct>>,
) -> Result<ColumnSource> {
    let Some(value) = value else {
        return Err(Error::new(
            ErrorKind::Unexpected,
            "Record batch transformer projects _partition without a partition value",
        ));
    };
    let field = snapshot_schema
        .field_by_id(RESERVED_FIELD_ID_PARTITION)
        .ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "Task schema lacks the _partition field",
            )
        })?;
    if value.is_none() && field.required {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            "Cannot serve a null _partition column for a required _partition field",
        ));
    }
    match field.field_type.as_ref() {
        Type::Struct(partition_type) => Ok(ColumnSource::PartitionConstant {
            partition_type: partition_type.clone(),
            value,
        }),
        other => Err(Error::new(
            ErrorKind::Unexpected,
            format!("_partition field has non-struct type {other:?}"),
        )),
    }
}

pub(crate) fn partition_constant_array(
    partition_type: &StructType,
    value: Option<&Struct>,
    num_rows: usize,
) -> Result<ArrayRef> {
    let arrow_type = type_to_arrow_type(&Type::Struct(partition_type.clone()))?;
    let arrow_fields = match &arrow_type {
        DataType::Struct(fields) => fields.clone(),
        other => {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!("Union partition type converted to non-struct Arrow type {other:?}"),
            ));
        }
    };
    match value {
        None => Ok(Arc::new(StructArray::new_null(arrow_fields, num_rows))),
        Some(tuple) => {
            let mut builder = StructBuilder::from_fields(arrow_fields, num_rows);
            let source_ids: Vec<i32> = partition_type
                .fields()
                .iter()
                .map(|field| field.id)
                .collect();
            for _ in 0..num_rows {
                append_partition(&mut builder, partition_type, &source_ids, tuple)?;
            }
            Ok(Arc::new(builder.finish()))
        }
    }
}

fn partition_union_from_schema(schema: &Schema) -> Result<&StructType> {
    let field = schema
        .field_by_id(RESERVED_FIELD_ID_PARTITION)
        .ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "Task schema lacks the _partition field",
            )
        })?;
    match field.field_type.as_ref() {
        Type::Struct(partition_type) => Ok(partition_type),
        other => Err(Error::new(
            ErrorKind::Unexpected,
            format!("_partition field has non-struct type {other:?}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;

    use super::{
        coerce_partition_value, partition_constant_array, partition_constant_source,
        partition_value_for_task, unified_partition_type,
    };
    use crate::arrow::record_batch_transformer::ColumnSource;
    use crate::metadata_columns::{
        RESERVED_COL_NAME_PARTITION, RESERVED_FIELD_ID_PARTITION, schema_with_partition,
    };
    use crate::spec::{
        Literal, NestedField, PartitionField, PartitionSpec, PrimitiveType, Schema, Struct,
        StructType, Transform, Type,
    };

    fn test_schema() -> Schema {
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                Arc::new(NestedField::required(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Long),
                )),
                Arc::new(NestedField::optional(
                    2,
                    "x",
                    Type::Primitive(PrimitiveType::Long),
                )),
                Arc::new(NestedField::optional(
                    3,
                    "y",
                    Type::Primitive(PrimitiveType::String),
                )),
            ])
            .build()
            .expect("test schema builds")
    }

    fn spec_field(source_id: i32, field_id: i32, name: &str) -> PartitionField {
        PartitionField {
            source_id,
            field_id,
            name: name.to_string(),
            transform: Transform::Identity,
        }
    }

    fn spec_with_fields(spec_id: i32, fields: Vec<PartitionField>) -> Arc<PartitionSpec> {
        Arc::new(PartitionSpec::from_fields_unchecked(spec_id, fields))
    }

    fn union_names(partition_union: &StructType) -> Vec<String> {
        partition_union
            .fields()
            .iter()
            .map(|field| field.name.clone())
            .collect()
    }

    #[test]
    fn union_orders_first_seen_fields_by_ascending_spec_id() {
        let schema = test_schema();
        let old = spec_with_fields(0, vec![spec_field(2, 1000, "x")]);
        let new = spec_with_fields(1, vec![spec_field(2, 1000, "x"), spec_field(3, 1001, "y")]);
        let union = unified_partition_type(&[new, old], &schema).expect("union builds");
        assert_eq!(union_names(&union), vec!["x".to_string(), "y".to_string()]);
        assert_eq!(union.fields()[0].id, 1000);
        assert_eq!(union.fields()[1].id, 1001);
    }

    #[test]
    fn union_keeps_first_seen_slot_when_name_repeats_across_specs() {
        let schema = test_schema();
        let old = spec_with_fields(0, vec![spec_field(2, 1000, "x")]);
        let renamed = spec_with_fields(1, vec![spec_field(2, 1005, "x")]);
        let union = unified_partition_type(&[old, renamed], &schema).expect("union builds");
        assert_eq!(union.fields().len(), 1);
        assert_eq!(union.fields()[0].id, 1000);
    }

    #[test]
    fn union_is_empty_when_every_spec_is_unpartitioned() {
        let schema = test_schema();
        let empty = spec_with_fields(0, vec![]);
        let union = unified_partition_type(&[empty], &schema).expect("union builds");
        assert!(union.fields().is_empty());
    }

    #[test]
    fn coerce_aligns_tuple_to_union_by_name() {
        let schema = test_schema();
        let old = spec_with_fields(0, vec![spec_field(2, 1000, "x"), spec_field(3, 1001, "y")]);
        let evolved = spec_with_fields(1, vec![spec_field(3, 1001, "y")]);
        let union = unified_partition_type(&[old, evolved.clone()], &schema).expect("union builds");
        let tuple = Struct::from_iter([Some(Literal::string("s"))]);
        let aligned = coerce_partition_value(&union, Some(&evolved), Some(&tuple))
            .expect("evolved file yields a struct");
        assert_eq!(aligned.fields().len(), 2);
        assert_eq!(aligned.fields()[0], None);
        assert_eq!(aligned.fields()[1], Some(Literal::string("s")),);
    }

    #[test]
    fn coerce_yields_none_when_spec_tuple_or_union_is_missing() {
        let schema = test_schema();
        let spec = spec_with_fields(0, vec![spec_field(2, 1000, "x")]);
        let union =
            unified_partition_type(std::slice::from_ref(&spec), &schema).expect("union builds");
        let empty_union = StructType::new(vec![]);
        let tuple = Struct::from_iter([Some(Literal::long(7))]);
        assert_eq!(coerce_partition_value(&union, None, Some(&tuple)), None);
        assert_eq!(coerce_partition_value(&union, Some(&spec), None), None);
        assert_eq!(
            coerce_partition_value(&empty_union, Some(&spec), Some(&tuple)),
            None
        );
    }

    #[test]
    fn coerce_tolerates_tuple_shorter_than_spec() {
        let schema = test_schema();
        let spec = spec_with_fields(0, vec![spec_field(2, 1000, "x"), spec_field(3, 1001, "y")]);
        let union =
            unified_partition_type(std::slice::from_ref(&spec), &schema).expect("union builds");
        let short = Struct::from_iter([Some(Literal::long(7))]);
        let aligned = coerce_partition_value(&union, Some(&spec), Some(&short))
            .expect("short tuple still yields a struct");
        assert_eq!(aligned.fields()[0], Some(Literal::long(7)));
        assert_eq!(aligned.fields()[1], None);
    }

    #[test]
    fn value_for_task_reads_union_from_task_schema() {
        let schema = test_schema();
        let spec = spec_with_fields(0, vec![spec_field(2, 1000, "x")]);
        let union =
            unified_partition_type(std::slice::from_ref(&spec), &schema).expect("union builds");
        let task_schema = schema_with_partition(&schema, &union).expect("augmented schema");
        let tuple = Struct::from_iter([Some(Literal::long(7))]);
        let value = partition_value_for_task(&task_schema, Some(&spec), Some(&tuple))
            .expect("task value builds");
        assert_eq!(value, Some(Struct::from_iter([Some(Literal::long(7))])));
    }

    #[test]
    fn value_for_task_errors_when_schema_lacks_partition_field() {
        let schema = test_schema();
        let spec = spec_with_fields(0, vec![spec_field(2, 1000, "x")]);
        let tuple = Struct::from_iter([Some(Literal::long(7))]);
        let error = partition_value_for_task(&schema, Some(&spec), Some(&tuple))
            .expect_err("unaugmented schema must fail");
        assert!(error.to_string().contains("lacks the _partition field"));
    }

    #[test]
    fn source_errors_when_value_unset() {
        let schema = test_schema();
        let spec = spec_with_fields(0, vec![spec_field(2, 1000, "x")]);
        let union = unified_partition_type(&[spec], &schema).expect("union builds");
        let task_schema = schema_with_partition(&schema, &union).expect("augmented schema");
        let error =
            partition_constant_source(&task_schema, None).expect_err("unset value must fail");
        assert!(error.to_string().contains("without a partition value"));
    }

    #[test]
    fn source_errors_for_null_struct_under_required_field() {
        let schema = test_schema();
        let required = Schema::builder()
            .with_schema_id(schema.schema_id())
            .with_fields(
                schema
                    .as_struct()
                    .fields()
                    .iter()
                    .cloned()
                    .chain(std::iter::once(Arc::new(NestedField::required(
                        RESERVED_FIELD_ID_PARTITION,
                        RESERVED_COL_NAME_PARTITION,
                        Type::Struct(StructType::new(vec![])),
                    )))),
            )
            .build()
            .expect("required-partition schema builds");
        let error = partition_constant_source(&required, Some(None))
            .expect_err("null struct under a required field must fail");
        assert!(error.to_string().contains("required _partition field"));
    }

    #[test]
    fn source_accepts_null_struct_under_optional_field() {
        let schema = test_schema();
        let spec = spec_with_fields(0, vec![spec_field(2, 1000, "x")]);
        let union = unified_partition_type(&[spec], &schema).expect("union builds");
        let task_schema = schema_with_partition(&schema, &union).expect("augmented schema");
        let source = partition_constant_source(&task_schema, Some(None))
            .expect("null struct under an optional field builds");
        assert!(matches!(source, ColumnSource::PartitionConstant {
            value: None,
            ..
        }));
    }

    #[test]
    fn array_builds_valid_struct_with_values() {
        let schema = test_schema();
        let spec = spec_with_fields(0, vec![spec_field(2, 1000, "x")]);
        let union = unified_partition_type(&[spec], &schema).expect("union builds");
        let tuple = Struct::from_iter([Some(Literal::long(7))]);
        let array = partition_constant_array(&union, Some(&tuple), 3).expect("array builds");
        assert_eq!(array.len(), 3);
        let structs = array
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .expect("struct array");
        assert_eq!(structs.null_count(), 0);
        for row in 0..3 {
            assert!(!structs.is_null(row));
        }
        let values = structs.column(0).as_primitive::<Int64Type>();
        for row in 0..3 {
            assert_eq!(values.value(row), 7);
        }
    }

    #[test]
    fn array_builds_null_struct_when_value_missing() {
        let schema = test_schema();
        let spec = spec_with_fields(0, vec![spec_field(2, 1000, "x")]);
        let union = unified_partition_type(&[spec], &schema).expect("union builds");
        let array = partition_constant_array(&union, None, 2).expect("array builds");
        assert_eq!(array.len(), 2);
        let structs = array
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .expect("struct array");
        assert_eq!(structs.null_count(), 2);
        for row in 0..2 {
            assert!(structs.is_null(row));
        }
    }

    #[test]
    fn array_builds_null_empty_struct_for_unpartitioned_union() {
        let empty_union = StructType::new(vec![]);
        let array = partition_constant_array(&empty_union, None, 2).expect("array builds");
        assert_eq!(array.len(), 2);
        let structs = array
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .expect("struct array");
        assert_eq!(structs.num_columns(), 0);
        assert_eq!(structs.null_count(), 2);
        for row in 0..2 {
            assert!(structs.is_null(row));
        }
    }
}
