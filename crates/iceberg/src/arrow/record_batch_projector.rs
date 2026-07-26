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

use arrow_array::{Array, ArrayRef, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema, SchemaRef};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::null_propagation::array_with_parent_validity;
use crate::arrow::schema::schema_to_arrow_schema;
use crate::error::Result;
use crate::spec::Schema as IcebergSchema;
use crate::{Error, ErrorKind};

/// Help to project specific field from `RecordBatch`` according to the fields id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordBatchProjector {
    // A vector of vectors, where each inner vector represents the index path to access a specific field in a nested structure.
    // E.g. [[0], [1, 2]] means the first field is accessed directly from the first column,
    // while the second field is accessed from the second column and then from its third subcolumn (second column must be a struct column).
    field_indices: Vec<Vec<usize>>,
    // The schema reference after projection. This schema is derived from the original schema based on the given field IDs.
    projected_schema: SchemaRef,
}

impl RecordBatchProjector {
    /// Init ArrowFieldProjector
    ///
    /// This function will iterate through the field and fetch the field from the original schema according to the field ids.
    /// The function to fetch the field id from the field is provided by `field_id_fetch_func`, return None if the field need to be skipped.
    /// This function will iterate through the nested fields if the field is a struct, `searchable_field_func` can be used to control whether
    /// iterate into the nested fields.
    pub(crate) fn new<F1, F2>(
        original_schema: SchemaRef,
        field_ids: &[i32],
        field_id_fetch_func: F1,
        searchable_field_func: F2,
    ) -> Result<Self>
    where
        F1: Fn(&Field) -> Result<Option<i64>>,
        F2: Fn(&Field) -> bool,
    {
        let mut field_indices = Vec::with_capacity(field_ids.len());
        let mut fields = Vec::with_capacity(field_ids.len());
        for &id in field_ids {
            let mut field_index = vec![];
            let field = Self::fetch_field_index(
                original_schema.fields(),
                &mut field_index,
                id as i64,
                &field_id_fetch_func,
                &searchable_field_func,
            )?
            .ok_or_else(|| {
                Error::new(ErrorKind::Unexpected, "Field not found")
                    .with_context("field_id", id.to_string())
            })?;
            fields.push(field.clone());
            field_indices.push(field_index);
        }
        let delete_arrow_schema = Arc::new(Schema::new(fields));
        Ok(Self {
            field_indices,
            projected_schema: delete_arrow_schema,
        })
    }

    /// Create RecordBatchProjector using Iceberg schema.
    ///
    /// This constructor converts the Iceberg schema to Arrow schema with field ID metadata,
    /// then uses the standard field ID lookup for projection.
    ///
    /// # Arguments
    /// * `iceberg_schema` - The Iceberg schema for field ID mapping  
    /// * `target_field_ids` - The field IDs to project
    pub fn from_iceberg_schema(
        iceberg_schema: Arc<IcebergSchema>,
        target_field_ids: &[i32],
    ) -> Result<Self> {
        let arrow_schema_with_ids = Arc::new(schema_to_arrow_schema(&iceberg_schema)?);

        let field_id_fetch_func = |field: &Field| -> Result<Option<i64>> {
            if let Some(value) = field.metadata().get(PARQUET_FIELD_ID_META_KEY) {
                let field_id = value.parse::<i32>().map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        "Failed to parse field id".to_string(),
                    )
                    .with_context("value", value)
                    .with_source(e)
                })?;
                Ok(Some(field_id as i64))
            } else {
                Ok(None)
            }
        };

        let searchable_field_func = |_field: &Field| -> bool { true };

        Self::new(
            arrow_schema_with_ids,
            target_field_ids,
            field_id_fetch_func,
            searchable_field_func,
        )
    }

    fn fetch_field_index<F1, F2>(
        fields: &Fields,
        index_vec: &mut Vec<usize>,
        target_field_id: i64,
        field_id_fetch_func: &F1,
        searchable_field_func: &F2,
    ) -> Result<Option<FieldRef>>
    where
        F1: Fn(&Field) -> Result<Option<i64>>,
        F2: Fn(&Field) -> bool,
    {
        for (pos, field) in fields.iter().enumerate() {
            let id = field_id_fetch_func(field)?;
            if let Some(id) = id
                && target_field_id == id
            {
                index_vec.push(pos);
                return Ok(Some(field.clone()));
            }
            if let DataType::Struct(inner) = field.data_type()
                && searchable_field_func(field)
                && let Some(res) = Self::fetch_field_index(
                    inner,
                    index_vec,
                    target_field_id,
                    field_id_fetch_func,
                    searchable_field_func,
                )?
            {
                index_vec.push(pos);
                return Ok(Some(res));
            }
        }
        Ok(None)
    }

    /// Return the reference of projected schema
    pub(crate) fn projected_schema_ref(&self) -> &SchemaRef {
        &self.projected_schema
    }

    /// Do projection with record batch
    pub(crate) fn project_batch(&self, batch: RecordBatch) -> Result<RecordBatch> {
        RecordBatch::try_new(
            self.projected_schema.clone(),
            self.project_column(batch.columns())?,
        )
        .map_err(|err| Error::new(ErrorKind::DataInvalid, format!("{err}")))
    }

    /// Do projection with columns
    pub fn project_column(&self, batch: &[ArrayRef]) -> Result<Vec<ArrayRef>> {
        self.field_indices
            .iter()
            .map(|index_vec| Self::get_column_by_field_index(batch, index_vec))
            .collect::<Result<Vec<_>>>()
    }

    fn get_column_by_field_index(batch: &[ArrayRef], field_index: &[usize]) -> Result<ArrayRef> {
        let mut rev_iterator = field_index.iter().rev();
        let top_index = *rev_iterator.next().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "Field index path is empty in RecordBatchProjector",
            )
        })?;
        // Bounds-checked: the index paths were derived from the schema the
        // projector was built with — a batch with FEWER top-level columns
        // (e.g. one produced by a different plan node than the projector was
        // planned against) must yield a typed error, not a slice panic.
        let mut array = batch
            .get(top_index)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Column index out of bounds for batch in RecordBatchProjector",
                )
                .with_context("column_index", top_index.to_string())
                .with_context("batch_columns", batch.len().to_string())
            })?
            .clone();
        for idx in rev_iterator {
            let struct_array = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or(Error::new(
                    ErrorKind::Unexpected,
                    "Cannot convert Array to StructArray",
                ))?;
            let child = struct_array
                .columns()
                .get(*idx)
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        "Struct child index out of bounds in RecordBatchProjector",
                    )
                    .with_context("child_index", idx.to_string())
                    .with_context("struct_children", struct_array.num_columns().to_string())
                })?
                .clone();
            // Detaching a child from its parent loses the parent's NULLs unless they are unioned
            // in — the shared walk in `crate::arrow::null_propagation` owns that step.
            array = array_with_parent_validity(&child, array.logical_nulls().as_ref())?;
        }
        Ok(array)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch, StringArray, StructArray};
    use arrow_schema::{DataType, Field, Fields, Schema};

    use crate::arrow::record_batch_projector::RecordBatchProjector;
    use crate::spec::{NestedField, PrimitiveType, Schema as IcebergSchema, Type};
    use crate::{Error, ErrorKind};

    #[test]
    fn test_record_batch_projector_nested_level() {
        let inner_fields = vec![
            Field::new("inner_field1", DataType::Int32, false),
            Field::new("inner_field2", DataType::Utf8, false),
        ];
        let fields = vec![
            Field::new("field1", DataType::Int32, false),
            Field::new(
                "field2",
                DataType::Struct(Fields::from(inner_fields.clone())),
                false,
            ),
        ];
        let schema = Arc::new(Schema::new(fields));

        let field_id_fetch_func = |field: &Field| match field.name().as_str() {
            "field1" => Ok(Some(1)),
            "field2" => Ok(Some(2)),
            "inner_field1" => Ok(Some(3)),
            "inner_field2" => Ok(Some(4)),
            _ => Err(Error::new(ErrorKind::Unexpected, "Field id not found")),
        };
        let projector =
            RecordBatchProjector::new(schema.clone(), &[1, 3], field_id_fetch_func, |_| true)
                .unwrap();

        assert_eq!(projector.field_indices.len(), 2);
        assert_eq!(projector.field_indices[0], vec![0]);
        assert_eq!(projector.field_indices[1], vec![0, 1]);

        let int_array = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let inner_int_array = Arc::new(Int32Array::from(vec![4, 5, 6])) as ArrayRef;
        let inner_string_array = Arc::new(StringArray::from(vec!["x", "y", "z"])) as ArrayRef;
        let struct_array = Arc::new(StructArray::from(vec![
            (
                Arc::new(inner_fields[0].clone()),
                inner_int_array as ArrayRef,
            ),
            (
                Arc::new(inner_fields[1].clone()),
                inner_string_array as ArrayRef,
            ),
        ])) as ArrayRef;
        let batch = RecordBatch::try_new(schema, vec![int_array, struct_array]).unwrap();

        let projected_batch = projector.project_batch(batch).unwrap();
        assert_eq!(projected_batch.num_columns(), 2);
        let projected_int_array = projected_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let projected_inner_int_array = projected_batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

        assert_eq!(projected_int_array.values(), &[1, 2, 3]);
        assert_eq!(projected_inner_int_array.values(), &[4, 5, 6]);
    }

    /// Projecting a field OUT of a nullable struct detaches it from the parent that made it
    /// unreachable, so the parent's validity must be unioned in. Arrow does not require a null
    /// struct slot to mask its children — here every inner value is physically live while the
    /// parent is NULL at row 1, so without the union the projection would hand back a live `5`
    /// for a logically-NULL row. (Pre-existing behavior; pinned here because the walk moved to
    /// the shared `null_propagation` helper and the other nested test has no nulls at all.)
    #[test]
    fn test_record_batch_projector_nested_null_parent_masks_child() {
        let inner_fields = vec![
            Field::new("inner_field1", DataType::Int32, true),
            Field::new("inner_field2", DataType::Utf8, true),
        ];
        let fields = vec![
            Field::new("field1", DataType::Int32, false),
            Field::new(
                "field2",
                DataType::Struct(Fields::from(inner_fields.clone())),
                true,
            ),
        ];
        let schema = Arc::new(Schema::new(fields));

        let field_id_fetch_func = |field: &Field| match field.name().as_str() {
            "field1" => Ok(Some(1)),
            "field2" => Ok(Some(2)),
            "inner_field1" => Ok(Some(3)),
            "inner_field2" => Ok(Some(4)),
            _ => Err(Error::new(ErrorKind::Unexpected, "Field id not found")),
        };
        let projector =
            RecordBatchProjector::new(schema.clone(), &[3], field_id_fetch_func, |_| true)
                .expect("projector");

        let int_array = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        // Every inner value is live; only the PARENT is null at row 1.
        let inner_int_array = Arc::new(Int32Array::from(vec![4, 5, 6])) as ArrayRef;
        let inner_string_array = Arc::new(StringArray::from(vec!["x", "y", "z"])) as ArrayRef;
        let struct_array = Arc::new(
            StructArray::try_new(
                Fields::from(inner_fields),
                vec![inner_int_array, inner_string_array],
                Some(arrow_buffer::NullBuffer::from(vec![true, false, true])),
            )
            .expect("struct array"),
        ) as ArrayRef;
        let batch = RecordBatch::try_new(schema, vec![int_array, struct_array]).expect("batch");

        let projected_batch = projector.project_batch(batch).expect("project");
        let projected = projected_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("projected inner int array");

        assert_eq!(projected.len(), 3);
        assert!(projected.is_valid(0), "row 0 parent is live");
        assert!(
            projected.is_null(1),
            "row 1 parent is NULL, so the projected child must be NULL — not a live 5"
        );
        assert!(projected.is_valid(2), "row 2 parent is live");
        assert_eq!(projected.value(0), 4);
        assert_eq!(projected.value(2), 6);
    }

    #[test]
    fn test_field_not_found() {
        let inner_fields = vec![
            Field::new("inner_field1", DataType::Int32, false),
            Field::new("inner_field2", DataType::Utf8, false),
        ];

        let fields = vec![
            Field::new("field1", DataType::Int32, false),
            Field::new(
                "field2",
                DataType::Struct(Fields::from(inner_fields.clone())),
                false,
            ),
        ];
        let schema = Arc::new(Schema::new(fields));

        let field_id_fetch_func = |field: &Field| match field.name().as_str() {
            "field1" => Ok(Some(1)),
            "field2" => Ok(Some(2)),
            "inner_field1" => Ok(Some(3)),
            "inner_field2" => Ok(Some(4)),
            _ => Err(Error::new(ErrorKind::Unexpected, "Field id not found")),
        };
        let projector =
            RecordBatchProjector::new(schema.clone(), &[1, 5], field_id_fetch_func, |_| true);

        assert!(projector.is_err());
    }

    #[test]
    fn test_field_not_reachable() {
        let inner_fields = vec![
            Field::new("inner_field1", DataType::Int32, false),
            Field::new("inner_field2", DataType::Utf8, false),
        ];

        let fields = vec![
            Field::new("field1", DataType::Int32, false),
            Field::new(
                "field2",
                DataType::Struct(Fields::from(inner_fields.clone())),
                false,
            ),
        ];
        let schema = Arc::new(Schema::new(fields));

        let field_id_fetch_func = |field: &Field| match field.name().as_str() {
            "field1" => Ok(Some(1)),
            "field2" => Ok(Some(2)),
            "inner_field1" => Ok(Some(3)),
            "inner_field2" => Ok(Some(4)),
            _ => Err(Error::new(ErrorKind::Unexpected, "Field id not found")),
        };
        let projector =
            RecordBatchProjector::new(schema.clone(), &[3], field_id_fetch_func, |_| false);
        assert!(projector.is_err());

        let projector =
            RecordBatchProjector::new(schema.clone(), &[3], field_id_fetch_func, |_| true);
        assert!(projector.is_ok());
    }

    #[test]
    fn test_from_iceberg_schema() {
        let iceberg_schema = IcebergSchema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(3, "age", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();

        let projector =
            RecordBatchProjector::from_iceberg_schema(Arc::new(iceberg_schema), &[1, 3]).unwrap();

        assert_eq!(projector.field_indices.len(), 2);
        assert_eq!(projector.projected_schema_ref().fields().len(), 2);
        assert_eq!(projector.projected_schema_ref().field(0).name(), "id");
        assert_eq!(projector.projected_schema_ref().field(1).name(), "age");
    }
}
