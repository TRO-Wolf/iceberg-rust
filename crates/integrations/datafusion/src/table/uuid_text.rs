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
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, FixedSizeBinaryArray, FixedSizeBinaryBuilder, LargeStringArray, MapArray,
    RecordBatch, RecordBatchOptions, StringArray, StringViewArray, StructArray, new_null_array,
};
use datafusion::arrow::datatypes::{
    DataType, Field, Fields, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef,
};
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use datafusion::scalar::ScalarValue;
use futures::StreamExt;
use iceberg::spec::{
    NestedFieldRef, PrimitiveType, Schema as IcebergSchema, StructType, Type as IcebergType,
};
use iceberg::{Error, ErrorKind};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::to_datafusion_error;

pub(crate) fn collect_uuid_field_ids(schema: &IcebergSchema) -> HashSet<i32> {
    let mut ids = HashSet::new();
    collect_struct_uuid_ids(schema.as_struct(), &mut ids);
    ids
}

fn collect_struct_uuid_ids(struct_type: &StructType, ids: &mut HashSet<i32>) {
    for field in struct_type.fields() {
        collect_nested_uuid_ids(field, ids);
    }
}

fn collect_nested_uuid_ids(field: &NestedFieldRef, ids: &mut HashSet<i32>) {
    match field.field_type.as_ref() {
        IcebergType::Primitive(PrimitiveType::Uuid) => {
            ids.insert(field.id);
        }
        IcebergType::Struct(struct_type) => collect_struct_uuid_ids(struct_type, ids),
        IcebergType::List(list_type) => {
            collect_nested_uuid_ids(&list_type.element_field, ids);
        }
        IcebergType::Map(map_type) => {
            collect_nested_uuid_ids(&map_type.key_field, ids);
            collect_nested_uuid_ids(&map_type.value_field, ids);
        }
        _ => {}
    }
}

pub(crate) fn arrow_schema_with_uuid_as_text(
    schema: &ArrowSchema,
    uuid_ids: &HashSet<i32>,
) -> ArrowSchema {
    ArrowSchema::new_with_metadata(
        schema
            .fields()
            .iter()
            .map(|field| Arc::new(uuid_text_field(field, uuid_ids)))
            .collect::<Fields>(),
        schema.metadata().clone(),
    )
}

fn uuid_text_field(field: &Field, uuid_ids: &HashSet<i32>) -> Field {
    let data_type = match field.data_type() {
        DataType::FixedSizeBinary(16)
            if arrow_field_id(field).is_some_and(|id| uuid_ids.contains(&id)) =>
        {
            DataType::Utf8
        }
        DataType::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|child| Arc::new(uuid_text_field(child, uuid_ids)))
                .collect(),
        ),
        DataType::List(element) => DataType::List(Arc::new(uuid_text_field(element, uuid_ids))),
        DataType::LargeList(element) => {
            DataType::LargeList(Arc::new(uuid_text_field(element, uuid_ids)))
        }
        DataType::FixedSizeList(element, width) => {
            DataType::FixedSizeList(Arc::new(uuid_text_field(element, uuid_ids)), *width)
        }
        DataType::Map(entries, ordered) => {
            DataType::Map(Arc::new(uuid_text_field(entries, uuid_ids)), *ordered)
        }
        _ => field.data_type().clone(),
    };
    Field::new(field.name().clone(), data_type, field.is_nullable())
        .with_metadata(field.metadata().clone())
}

fn arrow_field_id(field: &Field) -> Option<i32> {
    field
        .metadata()
        .get(PARQUET_FIELD_ID_META_KEY)?
        .parse::<i32>()
        .ok()
}

pub(crate) fn parse_uuid_text(value: &str) -> Result<[u8; 16], ()> {
    let bytes = value.as_bytes();
    if bytes.len() != 36 {
        return Err(());
    }
    for (index, byte) in bytes.iter().enumerate() {
        let hyphen = index == 8 || index == 13 || index == 18 || index == 23;
        if hyphen {
            if *byte != b'-' {
                return Err(());
            }
        } else if !byte.is_ascii_hexdigit() {
            return Err(());
        }
    }
    let mut out = [0u8; 16];
    let mut nibbles = 0usize;
    for byte in bytes.iter() {
        if *byte == b'-' {
            continue;
        }
        let digit = (*byte as char).to_digit(16).ok_or(())? as u8;
        if nibbles.is_multiple_of(2) {
            out[nibbles / 2] = digit << 4;
        } else {
            out[nibbles / 2] |= digit;
        }
        nibbles += 1;
    }
    Ok(out)
}

pub(crate) fn is_canonical_uuid_text(value: &str) -> bool {
    parse_uuid_text(value).is_ok() && value.bytes().all(|b| !b.is_ascii_uppercase())
}

const HEX_LOWER: &[u8; 16] = b"0123456789abcdef";

pub(crate) fn render_uuid_text(bytes: &[u8; 16]) -> String {
    let mut text = String::with_capacity(36);
    for (index, byte) in bytes.iter().enumerate() {
        if index == 4 || index == 6 || index == 8 || index == 10 {
            text.push('-');
        }
        text.push(HEX_LOWER[(byte >> 4) as usize] as char);
        text.push(HEX_LOWER[(byte & 0x0f) as usize] as char);
    }
    text
}

pub(crate) fn render_batch_uuid_as_text(
    batch: RecordBatch,
    text_schema: &ArrowSchemaRef,
) -> DFResult<RecordBatch> {
    if batch.schema_ref() == text_schema {
        return Ok(batch);
    }
    let num_rows = batch.num_rows();
    if batch.num_columns() != text_schema.fields().len() {
        return Err(DataFusionError::Internal(format!(
            "cannot render {} scanned columns as {} advertised text columns",
            batch.num_columns(),
            text_schema.fields().len()
        )));
    }
    let mut columns = Vec::with_capacity(text_schema.fields().len());
    for (column, field) in batch.columns().iter().zip(text_schema.fields()) {
        columns.push(render_column_uuid_as_text(column, field.data_type())?);
    }
    RecordBatch::try_new_with_options(
        text_schema.clone(),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(num_rows)),
    )
    .map_err(|e| {
        DataFusionError::ArrowError(
            Box::new(e),
            Some("failed to render uuid columns as text".to_string()),
        )
    })
}

fn render_column_uuid_as_text(column: &ArrayRef, target: &DataType) -> DFResult<ArrayRef> {
    if column.data_type() == target {
        return Ok(column.clone());
    }
    match (column.data_type(), target) {
        (DataType::FixedSizeBinary(16), DataType::Utf8) => {
            let source = column
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal(
                        "a FixedSizeBinary(16) column is not backed by FixedSizeBinaryArray"
                            .to_string(),
                    )
                })?;
            let mut rendered = Vec::with_capacity(source.len());
            for row in 0..source.len() {
                if source.is_null(row) {
                    rendered.push(None);
                } else {
                    let bytes: [u8; 16] = source.value(row).try_into().map_err(|_| {
                        DataFusionError::Internal(
                            "a uuid value holds back a short byte slice".to_string(),
                        )
                    })?;
                    rendered.push(Some(render_uuid_text(&bytes)));
                }
            }
            Ok(Arc::new(StringArray::from(rendered)))
        }
        (DataType::Null, _) => Ok(new_null_array(target, column.len())),
        (DataType::Struct(_), DataType::Struct(target_fields)) => {
            let source = column
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal(
                        "a struct column is not backed by StructArray".to_string(),
                    )
                })?;
            if source.num_columns() != target_fields.len() {
                return Err(DataFusionError::Internal(
                    "a scanned struct does not carry the child arrays its text type implies"
                        .to_string(),
                ));
            }
            let mut children = Vec::with_capacity(target_fields.len());
            for (child, target_child) in source.columns().iter().zip(target_fields.iter()) {
                children.push(render_column_uuid_as_text(child, target_child.data_type())?);
            }
            Ok(Arc::new(
                StructArray::try_new_with_length(
                    target_fields.clone(),
                    children,
                    source.nulls().cloned(),
                    source.len(),
                )
                .map_err(|e| {
                    DataFusionError::ArrowError(
                        Box::new(e),
                        Some("failed to render a struct column as text".to_string()),
                    )
                })?,
            ))
        }
        (DataType::List(_), DataType::List(target_element)) => {
            render_list_uuid_as_text(column, target_element, false)
        }
        (DataType::LargeList(_), DataType::LargeList(target_element)) => {
            render_list_uuid_as_text(column, target_element, true)
        }
        (DataType::Map(_, _), DataType::Map(target_entries, ordered)) => {
            let source = column.as_any().downcast_ref::<MapArray>().ok_or_else(|| {
                DataFusionError::Internal("a Map column is not backed by MapArray".to_string())
            })?;
            let entries: ArrayRef = Arc::new(source.entries().clone());
            let rendered = render_column_uuid_as_text(&entries, target_entries.data_type())?;
            let rendered = rendered
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal(
                        "rendered map entries are not backed by StructArray".to_string(),
                    )
                })?
                .clone();
            Ok(Arc::new(
                MapArray::try_new(
                    target_entries.clone(),
                    source.offsets().clone(),
                    rendered,
                    source.nulls().cloned(),
                    *ordered,
                )
                .map_err(|e| {
                    DataFusionError::ArrowError(
                        Box::new(e),
                        Some("failed to render a map column as text".to_string()),
                    )
                })?,
            ))
        }
        _ => Err(DataFusionError::Internal(format!(
            "cannot render scanned {} as advertised {target}",
            column.data_type()
        ))),
    }
}

fn render_list_uuid_as_text(
    column: &ArrayRef,
    target_element: &Field,
    large: bool,
) -> DFResult<ArrayRef> {
    use datafusion::arrow::array::{LargeListArray, ListArray, OffsetSizeTrait};
    fn render_values<T: OffsetSizeTrait>(
        list: &datafusion::arrow::array::GenericListArray<T>,
        target_element: &Field,
    ) -> DFResult<ArrayRef> {
        let values = render_column_uuid_as_text(list.values(), target_element.data_type())?;
        Ok(Arc::new(
            datafusion::arrow::array::GenericListArray::<T>::try_new(
                Arc::new(target_element.clone()),
                list.offsets().clone(),
                values,
                list.nulls().cloned(),
            )
            .map_err(|e| {
                DataFusionError::ArrowError(
                    Box::new(e),
                    Some("failed to render a list column as text".to_string()),
                )
            })?,
        ))
    }
    if large {
        let list = column
            .as_any()
            .downcast_ref::<LargeListArray>()
            .ok_or_else(|| {
                DataFusionError::Internal(
                    "a LargeList column is not backed by LargeListArray".to_string(),
                )
            })?;
        render_values(list, target_element)
    } else {
        let list = column.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
            DataFusionError::Internal("a List column is not backed by ListArray".to_string())
        })?;
        render_values(list, target_element)
    }
}

pub(crate) fn invalid_uuid_string(value: &str) -> DataFusionError {
    to_datafusion_error(Error::new(
        ErrorKind::DataInvalid,
        format!("Invalid UUID string: {value}"),
    ))
}

pub(crate) fn convert_batch_uuid_text_to_bytes(
    batch: RecordBatch,
    byte_schema: &ArrowSchemaRef,
) -> DFResult<RecordBatch> {
    if batch.schema_ref() == byte_schema {
        return Ok(batch);
    }
    let num_rows = batch.num_rows();
    if batch.num_columns() != byte_schema.fields().len() {
        return Err(DataFusionError::Internal(format!(
            "cannot convert {} text columns into {} byte columns",
            batch.num_columns(),
            byte_schema.fields().len()
        )));
    }
    let mut columns = Vec::with_capacity(byte_schema.fields().len());
    for (column, field) in batch.columns().iter().zip(byte_schema.fields()) {
        columns.push(convert_column_uuid_text_to_bytes(
            column,
            field.data_type(),
        )?);
    }
    RecordBatch::try_new_with_options(
        byte_schema.clone(),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(num_rows)),
    )
    .map_err(|e| {
        DataFusionError::ArrowError(
            Box::new(e),
            Some("failed to convert uuid text columns into bytes".to_string()),
        )
    })
}

fn convert_column_uuid_text_to_bytes(column: &ArrayRef, target: &DataType) -> DFResult<ArrayRef> {
    if column.data_type() == target {
        return Ok(column.clone());
    }
    match (column.data_type(), target) {
        (DataType::Utf8, DataType::FixedSizeBinary(16)) => {
            let source = column
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal(
                        "a Utf8 column is not backed by StringArray".to_string(),
                    )
                })?;
            parse_text_values(source.len(), |row| {
                if source.is_null(row) {
                    Ok(None)
                } else {
                    parse_uuid_text(source.value(row))
                        .map(Some)
                        .map_err(|_| source.value(row).to_string())
                }
            })
        }
        (DataType::LargeUtf8, DataType::FixedSizeBinary(16)) => {
            let source = column
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal(
                        "a LargeUtf8 column is not backed by LargeStringArray".to_string(),
                    )
                })?;
            parse_text_values(source.len(), |row| {
                if source.is_null(row) {
                    Ok(None)
                } else {
                    parse_uuid_text(source.value(row))
                        .map(Some)
                        .map_err(|_| source.value(row).to_string())
                }
            })
        }
        (DataType::Utf8View, DataType::FixedSizeBinary(16)) => {
            let source = column
                .as_any()
                .downcast_ref::<StringViewArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal(
                        "a Utf8View column is not backed by StringViewArray".to_string(),
                    )
                })?;
            parse_text_values(source.len(), |row| {
                if source.is_null(row) {
                    Ok(None)
                } else {
                    parse_uuid_text(source.value(row))
                        .map(Some)
                        .map_err(|_| source.value(row).to_string())
                }
            })
        }
        (DataType::Null, _) => Ok(new_null_array(target, column.len())),
        (DataType::Struct(_), DataType::Struct(target_fields)) => {
            let source = column
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal(
                        "a struct column is not backed by StructArray".to_string(),
                    )
                })?;
            if source.num_columns() != target_fields.len() {
                return Err(DataFusionError::Internal(
                    "a text struct does not carry the child arrays its byte type implies"
                        .to_string(),
                ));
            }
            let mut children = Vec::with_capacity(target_fields.len());
            for (child, target_child) in source.columns().iter().zip(target_fields.iter()) {
                children.push(convert_column_uuid_text_to_bytes(
                    child,
                    target_child.data_type(),
                )?);
            }
            Ok(Arc::new(
                StructArray::try_new_with_length(
                    target_fields.clone(),
                    children,
                    source.nulls().cloned(),
                    source.len(),
                )
                .map_err(|e| {
                    DataFusionError::ArrowError(
                        Box::new(e),
                        Some("failed to convert a struct column into bytes".to_string()),
                    )
                })?,
            ))
        }
        (DataType::List(_), DataType::List(target_element)) => {
            convert_list_uuid_text_to_bytes(column, target_element, false)
        }
        (DataType::LargeList(_), DataType::LargeList(target_element)) => {
            convert_list_uuid_text_to_bytes(column, target_element, true)
        }
        (DataType::Map(_, _), DataType::Map(target_entries, ordered)) => {
            let source = column.as_any().downcast_ref::<MapArray>().ok_or_else(|| {
                DataFusionError::Internal("a Map column is not backed by MapArray".to_string())
            })?;
            let entries: ArrayRef = Arc::new(source.entries().clone());
            let converted =
                convert_column_uuid_text_to_bytes(&entries, target_entries.data_type())?;
            let converted = converted
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal(
                        "converted map entries are not backed by StructArray".to_string(),
                    )
                })?
                .clone();
            Ok(Arc::new(
                MapArray::try_new(
                    target_entries.clone(),
                    source.offsets().clone(),
                    converted,
                    source.nulls().cloned(),
                    *ordered,
                )
                .map_err(|e| {
                    DataFusionError::ArrowError(
                        Box::new(e),
                        Some("failed to convert a map column into bytes".to_string()),
                    )
                })?,
            ))
        }
        _ => Err(DataFusionError::Internal(format!(
            "cannot convert text {} into byte {target}",
            column.data_type()
        ))),
    }
}

fn parse_text_values(
    len: usize,
    mut value_at: impl FnMut(usize) -> Result<Option<[u8; 16]>, String>,
) -> DFResult<ArrayRef> {
    let mut values = Vec::with_capacity(len);
    let mut valid = Vec::with_capacity(len);
    for row in 0..len {
        match value_at(row) {
            Ok(None) => {
                values.extend_from_slice(&[0u8; 16]);
                valid.push(false);
            }
            Ok(Some(bytes)) => {
                values.extend_from_slice(&bytes);
                valid.push(true);
            }
            Err(text) => return Err(invalid_uuid_string(&text)),
        }
    }
    let mut builder = FixedSizeBinaryBuilder::new(16);
    let mut rows = values.into_iter();
    for valid_row in valid {
        if valid_row {
            let mut bytes = [0u8; 16];
            for slot in bytes.iter_mut() {
                *slot = rows.next().unwrap_or(0);
            }
            builder.append_value(bytes).map_err(|e| {
                DataFusionError::ArrowError(
                    Box::new(e),
                    Some("failed to build a uuid byte column".to_string()),
                )
            })?;
        } else {
            for _ in 0..16 {
                rows.next();
            }
            builder.append_null();
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn convert_list_uuid_text_to_bytes(
    column: &ArrayRef,
    target_element: &Field,
    large: bool,
) -> DFResult<ArrayRef> {
    use datafusion::arrow::array::{LargeListArray, ListArray, OffsetSizeTrait};
    fn convert_values<T: OffsetSizeTrait>(
        list: &datafusion::arrow::array::GenericListArray<T>,
        target_element: &Field,
    ) -> DFResult<ArrayRef> {
        let values = convert_column_uuid_text_to_bytes(list.values(), target_element.data_type())?;
        Ok(Arc::new(
            datafusion::arrow::array::GenericListArray::<T>::try_new(
                Arc::new(target_element.clone()),
                list.offsets().clone(),
                values,
                list.nulls().cloned(),
            )
            .map_err(|e| {
                DataFusionError::ArrowError(
                    Box::new(e),
                    Some("failed to convert a list column into bytes".to_string()),
                )
            })?,
        ))
    }
    if large {
        let list = column
            .as_any()
            .downcast_ref::<LargeListArray>()
            .ok_or_else(|| {
                DataFusionError::Internal(
                    "a LargeList column is not backed by LargeListArray".to_string(),
                )
            })?;
        convert_values(list, target_element)
    } else {
        let list = column.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
            DataFusionError::Internal("a List column is not backed by ListArray".to_string())
        })?;
        convert_values(list, target_element)
    }
}

pub(crate) struct UuidTextToBytesExec {
    input: Arc<dyn ExecutionPlan>,
    byte_schema: ArrowSchemaRef,
    properties: Arc<PlanProperties>,
}

impl UuidTextToBytesExec {
    pub(crate) fn new(input: Arc<dyn ExecutionPlan>, byte_schema: ArrowSchemaRef) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(byte_schema.clone()),
            input.output_partitioning().clone(),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            input,
            byte_schema,
            properties,
        }
    }
}

impl Debug for UuidTextToBytesExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "UuidTextToBytesExec")
    }
}

impl DisplayAs for UuidTextToBytesExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(f, "UuidTextToBytesExec")
            }
        }
    }
}

impl ExecutionPlan for UuidTextToBytesExec {
    fn name(&self) -> &str {
        "UuidTextToBytesExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let mut children = children;
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "UuidTextToBytesExec expects exactly one child, but provided {}",
                children.len()
            )));
        }
        let Some(child) = children.pop() else {
            return Err(DataFusionError::Internal(
                "UuidTextToBytesExec lost its child".to_string(),
            ));
        };
        Ok(Arc::new(Self::new(child, self.byte_schema.clone())))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let byte_schema = self.byte_schema.clone();
        let stream = self.input.execute(partition, context)?;
        let converted = stream.map(move |batch| {
            batch.and_then(|batch| convert_batch_uuid_text_to_bytes(batch, &byte_schema))
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.byte_schema.clone(),
            Box::pin(converted),
        )))
    }
}

pub(crate) fn uuid_column_names(schema: &IcebergSchema) -> HashSet<String> {
    schema
        .as_struct()
        .fields()
        .iter()
        .filter(|field| {
            matches!(
                field.field_type.as_ref(),
                IcebergType::Primitive(PrimitiveType::Uuid)
            )
        })
        .map(|field| field.name.clone())
        .collect()
}

pub(crate) fn rewrite_uuid_text_filters(
    filters: &[Expr],
    schema: &IcebergSchema,
    byte_residual: bool,
) -> Vec<Expr> {
    let uuid_columns = uuid_column_names(schema);
    filters
        .iter()
        .map(|filter| rewrite_uuid_text_expr(filter, &uuid_columns, byte_residual, false))
        .collect()
}

fn rewrite_uuid_text_expr(
    expr: &Expr,
    uuid_columns: &HashSet<String>,
    byte_residual: bool,
    negated: bool,
) -> Expr {
    match expr {
        Expr::Not(inner) => Expr::Not(Box::new(rewrite_uuid_text_expr(
            inner,
            uuid_columns,
            byte_residual,
            !negated,
        ))),
        Expr::BinaryExpr(binary) => match binary.op {
            Operator::And | Operator::Or => Expr::BinaryExpr(BinaryExpr::new(
                Box::new(rewrite_uuid_text_expr(
                    &binary.left,
                    uuid_columns,
                    byte_residual,
                    negated,
                )),
                binary.op,
                Box::new(rewrite_uuid_text_expr(
                    &binary.right,
                    uuid_columns,
                    byte_residual,
                    negated,
                )),
            )),
            Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq => rewrite_uuid_comparison(
                &binary.left,
                binary.op,
                &binary.right,
                uuid_columns,
                byte_residual,
                negated,
            )
            .unwrap_or_else(|| expr.clone()),
            _ => expr.clone(),
        },
        Expr::InList(inlist) => rewrite_uuid_in_list(inlist, uuid_columns, byte_residual, negated)
            .unwrap_or_else(|| expr.clone()),
        _ => expr.clone(),
    }
}

fn string_literal_value(value: &ScalarValue) -> Option<String> {
    match value {
        ScalarValue::Utf8(Some(text))
        | ScalarValue::LargeUtf8(Some(text))
        | ScalarValue::Utf8View(Some(text)) => Some(text.clone()),
        _ => None,
    }
}

fn uuid_byte_literal(bytes: [u8; 16]) -> Expr {
    Expr::Literal(ScalarValue::FixedSizeBinary(16, Some(bytes.to_vec())), None)
}

fn rewrite_uuid_comparison(
    left: &Expr,
    op: Operator,
    right: &Expr,
    uuid_columns: &HashSet<String>,
    byte_residual: bool,
    negated: bool,
) -> Option<Expr> {
    let (name, text, swapped) = match (left, right) {
        (Expr::Column(column), Expr::Literal(value, _)) => {
            (column.name.clone(), string_literal_value(value)?, false)
        }
        (Expr::Literal(value, _), Expr::Column(column)) => {
            (column.name.clone(), string_literal_value(value)?, true)
        }
        _ => return None,
    };
    if !uuid_columns.contains(&name) {
        return None;
    }
    let bytes = parse_uuid_text(&text).ok()?;
    let effective_not_eq = (op == Operator::NotEq) != negated;
    let is_range = matches!(
        op,
        Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq
    );
    if !byte_residual && (is_range || effective_not_eq) && !is_canonical_uuid_text(&text) {
        return None;
    }
    let literal = uuid_byte_literal(bytes);
    if swapped {
        Some(Expr::BinaryExpr(BinaryExpr::new(
            Box::new(literal),
            op,
            Box::new(left.clone()),
        )))
    } else {
        Some(Expr::BinaryExpr(BinaryExpr::new(
            Box::new(left.clone()),
            op,
            Box::new(literal),
        )))
    }
}

fn rewrite_uuid_in_list(
    inlist: &InList,
    uuid_columns: &HashSet<String>,
    byte_residual: bool,
    negated: bool,
) -> Option<Expr> {
    let Expr::Column(column) = inlist.expr.as_ref() else {
        return None;
    };
    if !uuid_columns.contains(&column.name) {
        return None;
    }
    let effective_negated = inlist.negated != negated;
    let mut literals = Vec::with_capacity(inlist.list.len());
    for item in &inlist.list {
        let Expr::Literal(value, _) = item else {
            return None;
        };
        let text = string_literal_value(value)?;
        if !byte_residual && effective_negated && !is_canonical_uuid_text(&text) {
            return None;
        }
        literals.push(uuid_byte_literal(parse_uuid_text(&text).ok()?));
    }
    Some(Expr::InList(InList::new(
        inlist.expr.clone(),
        literals,
        inlist.negated,
    )))
}

pub(crate) fn rewrite_uuid_text_assignment(
    value: Expr,
    field_type: &IcebergType,
) -> DFResult<Expr> {
    let IcebergType::Primitive(PrimitiveType::Uuid) = field_type else {
        return Ok(value);
    };
    let Expr::Literal(literal, meta) = &value else {
        return Ok(value);
    };
    let Some(text) = string_literal_value(literal) else {
        return Ok(value);
    };
    let bytes = parse_uuid_text(&text).map_err(|_| invalid_uuid_string(&text))?;
    Ok(Expr::Literal(
        ScalarValue::FixedSizeBinary(16, Some(bytes.to_vec())),
        meta.clone(),
    ))
}
