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

use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Decimal128Array,
    FixedSizeBinaryArray, FixedSizeListArray, Float32Array, Float64Array, Int32Array, Int64Array,
    LargeBinaryArray, LargeListArray, LargeStringArray, ListArray, MapArray, StringArray,
    StringViewArray, StructArray, Time64MicrosecondArray, TimestampMicrosecondArray,
    TimestampNanosecondArray, new_null_array,
};
use arrow_buffer::{Buffer, NullBuffer};
use arrow_schema::{DataType, FieldRef, TimeUnit};
use uuid::Uuid;

use super::get_field_id_from_metadata;
use crate::spec::{
    ListType, Literal, Map, MapType, NestedField, PartnerAccessor, PrimitiveLiteral, PrimitiveType,
    SchemaWithPartnerVisitor, Struct, StructType, Type, visit_struct_with_partner,
    visit_type_with_partner,
};
use crate::{Error, ErrorKind, Result};

/// Reassemble per-row list literals from a flat `elements` buffer and an Arrow offset buffer,
/// with every offset bounds-checked.
///
/// arrow-rs guarantees its own offsets are monotonic and in-range, but a custom (non-arrow-rs)
/// FileIO producer could feed a degenerate buffer: empty offsets (`len() - 1` would underflow to
/// `usize::MAX` and abort `with_capacity`), `end < start`, or `end > elements.len()` (a slice
/// panic). Each of those is turned into a typed `DataInvalid` error here.
fn slice_list_by_offsets<O: arrow_array::OffsetSizeTrait>(
    offsets: &[O],
    elements: &[Option<Literal>],
) -> Result<Vec<Option<Literal>>> {
    // `saturating_sub` so an empty offset buffer yields capacity 0 instead of underflowing.
    let row_count = offsets.len().saturating_sub(1);
    let mut result = Vec::with_capacity(row_count);
    for i in 0..row_count {
        let start = offsets[i].as_usize();
        let end = offsets[i + 1].as_usize();
        if start > end {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "List offsets are not monotonically increasing",
            ));
        }
        let slice = elements.get(start..end).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "List offset slice is out of bounds for the element buffer",
            )
        })?;
        result.push(Some(Literal::List(slice.to_vec())));
    }
    Ok(result)
}

struct ArrowArrayToIcebergStructConverter;

impl SchemaWithPartnerVisitor<ArrayRef> for ArrowArrayToIcebergStructConverter {
    type T = Vec<Option<Literal>>;

    fn schema(
        &mut self,
        _schema: &crate::spec::Schema,
        _partner: &ArrayRef,
        value: Vec<Option<Literal>>,
    ) -> Result<Vec<Option<Literal>>> {
        Ok(value)
    }

    fn field(
        &mut self,
        _field: &crate::spec::NestedFieldRef,
        _partner: &ArrayRef,
        value: Vec<Option<Literal>>,
    ) -> Result<Vec<Option<Literal>>> {
        // NOTE: the `required` check deliberately does NOT live here. A field's null-ness is only
        // a violation relative to its ENCLOSING struct's validity, and this callback cannot see
        // the parent — see [`Self::struct`], which owns the check.
        Ok(value)
    }

    fn r#struct(
        &mut self,
        r#struct: &StructType,
        array: &ArrayRef,
        results: Vec<Vec<Option<Literal>>>,
    ) -> Result<Vec<Option<Literal>>> {
        let row_len = results.first().map(|column| column.len()).unwrap_or(0);
        if let Some(col) = results.iter().find(|col| col.len() != row_len) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "The struct columns have different row length",
            )
            .with_context("first col length", row_len.to_string())
            .with_context("actual col length", col.len().to_string()));
        }

        let mut struct_literals = Vec::with_capacity(row_len);
        let fields = r#struct.fields();
        let mut columns_iters = results
            .into_iter()
            .map(|column| column.into_iter())
            .collect::<Vec<_>>();

        for i in 0..row_len {
            let mut literals = Vec::with_capacity(columns_iters.len());
            for column_iter in columns_iters.iter_mut() {
                // `flatten`, not `unwrap`: the equal-length check above already guarantees every
                // column yields `row_len` items, so an exhausted iterator is unreachable — and if
                // it ever happened it degrades to a NULL cell (caught below for a `required`
                // field) instead of panicking a write.
                literals.push(column_iter.next().flatten());
            }
            if array.is_null(i) {
                // The row's struct is NULL: its fields are unreachable, so a `required` field
                // carrying no value here is NOT a violation. Java agrees by construction — the
                // Avro writer's `ValueWriters$OptionWriter.write` (iceberg-core 1.10.0) emits the
                // null union branch and never invokes the value writer for the struct's fields.
                struct_literals.push(None);
            } else {
                // The row's struct is live, so every `required` field MUST carry a value.
                for (field, literal) in fields.iter().zip(literals.iter()) {
                    if field.required && literal.is_none() {
                        return Err(Error::new(
                            ErrorKind::DataInvalid,
                            "The field is required but has null value",
                        )
                        .with_context("field_id", field.id.to_string())
                        .with_context("field_name", &field.name)
                        .with_context("row", i.to_string()));
                    }
                }
                struct_literals.push(Some(Literal::Struct(Struct::from_iter(literals))));
            }
        }

        Ok(struct_literals)
    }

    fn list(
        &mut self,
        list: &ListType,
        array: &ArrayRef,
        elements: Vec<Option<Literal>>,
    ) -> Result<Vec<Option<Literal>>> {
        if list.element_field.required && elements.iter().any(Option::is_none) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "The list should not have null value",
            ));
        }
        match array.data_type() {
            DataType::List(_) => {
                let offset = array
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .ok_or_else(|| {
                        Error::new(ErrorKind::DataInvalid, "The partner is not a list array")
                    })?
                    .offsets();
                slice_list_by_offsets(offset, &elements)
            }
            DataType::LargeList(_) => {
                let offset = array
                    .as_any()
                    .downcast_ref::<LargeListArray>()
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "The partner is not a large list array",
                        )
                    })?
                    .offsets();
                slice_list_by_offsets(offset, &elements)
            }
            DataType::FixedSizeList(_, len) => {
                // A zero-width FixedSizeList would divide-by-zero below; arrow-rs forbids it but a
                // custom (non-arrow-rs) FileIO producer could feed one in.
                let width = usize::try_from(*len).map_err(|_| {
                    Error::new(ErrorKind::DataInvalid, "FixedSizeList width is negative")
                })?;
                if width == 0 {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        "FixedSizeList width must be greater than zero",
                    ));
                }
                let count = elements.len() / width;
                let mut result = Vec::with_capacity(count);
                for i in 0..count {
                    let start = i * width;
                    let end = (i + 1) * width;
                    let slice = elements.get(start..end).ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "FixedSizeList slice is out of bounds for the element buffer",
                        )
                    })?;
                    result.push(Some(Literal::List(slice.to_vec())));
                }
                Ok(result)
            }
            _ => Err(Error::new(
                ErrorKind::DataInvalid,
                "The partner is not a list type",
            )),
        }
    }

    fn map(
        &mut self,
        _map: &MapType,
        partner: &ArrayRef,
        key_values: Vec<Option<Literal>>,
        values: Vec<Option<Literal>>,
    ) -> Result<Vec<Option<Literal>>> {
        // Make sure key_value and value have the same row length
        if key_values.len() != values.len() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "The key value and value of map should have the same row length",
            ));
        }

        let offsets = partner
            .as_any()
            .downcast_ref::<MapArray>()
            .ok_or_else(|| Error::new(ErrorKind::DataInvalid, "The partner is not a map array"))?
            .offsets();
        // combine the result according to the offset; bounds-check every access so a degenerate
        // offset buffer from a custom FileIO producer yields a typed error, not a panic/abort.
        let row_count = offsets.len().saturating_sub(1);
        let mut result = Vec::with_capacity(row_count);
        for i in 0..row_count {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            if start > end {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Map offsets are not monotonically increasing",
                ));
            }
            let key_slice = key_values.get(start..end).ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Map key offset slice is out of bounds",
                )
            })?;
            let value_slice = values.get(start..end).ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Map value offset slice is out of bounds",
                )
            })?;
            let mut map = Map::new();
            for (key, value) in key_slice.iter().zip(value_slice.iter()) {
                // A map key is non-nullable in Iceberg; reject a null key instead of unwrapping.
                let key = key.clone().ok_or_else(|| {
                    Error::new(ErrorKind::DataInvalid, "Map key must not be null")
                })?;
                map.insert(key, value.clone());
            }
            result.push(Some(Literal::Map(map)));
        }
        Ok(result)
    }

    fn primitive(&mut self, p: &PrimitiveType, partner: &ArrayRef) -> Result<Vec<Option<Literal>>> {
        match p {
            PrimitiveType::Boolean => {
                let array = partner
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| {
                        Error::new(ErrorKind::DataInvalid, "The partner is not a boolean array")
                    })?;
                Ok(array.iter().map(|v| v.map(Literal::bool)).collect())
            }
            PrimitiveType::Int => {
                let array = partner
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| {
                        Error::new(ErrorKind::DataInvalid, "The partner is not a int32 array")
                    })?;
                Ok(array.iter().map(|v| v.map(Literal::int)).collect())
            }
            PrimitiveType::Long => {
                let array = partner
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        Error::new(ErrorKind::DataInvalid, "The partner is not a int64 array")
                    })?;
                Ok(array.iter().map(|v| v.map(Literal::long)).collect())
            }
            PrimitiveType::Float => {
                let array = partner
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| {
                        Error::new(ErrorKind::DataInvalid, "The partner is not a float32 array")
                    })?;
                Ok(array.iter().map(|v| v.map(Literal::float)).collect())
            }
            PrimitiveType::Double => {
                let array = partner
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        Error::new(ErrorKind::DataInvalid, "The partner is not a float64 array")
                    })?;
                Ok(array.iter().map(|v| v.map(Literal::double)).collect())
            }
            PrimitiveType::Decimal { precision, scale } => {
                let array = partner
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "The partner is not a decimal128 array",
                        )
                    })?;
                if let DataType::Decimal128(arrow_precision, arrow_scale) = array.data_type()
                    && (*arrow_precision as u32 != *precision || *arrow_scale as u32 != *scale)
                {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "The precision or scale ({arrow_precision},{arrow_scale}) of arrow decimal128 array is not compatible with iceberg decimal type ({precision},{scale})"
                        ),
                    ));
                }
                Ok(array.iter().map(|v| v.map(Literal::decimal)).collect())
            }
            PrimitiveType::Date => {
                let array = partner
                    .as_any()
                    .downcast_ref::<Date32Array>()
                    .ok_or_else(|| {
                        Error::new(ErrorKind::DataInvalid, "The partner is not a date32 array")
                    })?;
                Ok(array.iter().map(|v| v.map(Literal::date)).collect())
            }
            PrimitiveType::Time => {
                let array = partner
                    .as_any()
                    .downcast_ref::<Time64MicrosecondArray>()
                    .ok_or_else(|| {
                        Error::new(ErrorKind::DataInvalid, "The partner is not a time64 array")
                    })?;
                Ok(array.iter().map(|v| v.map(Literal::time)).collect())
            }
            PrimitiveType::Timestamp => {
                let array = partner
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "The partner is not a timestamp array",
                        )
                    })?;
                Ok(array.iter().map(|v| v.map(Literal::timestamp)).collect())
            }
            PrimitiveType::Timestamptz => {
                let array = partner
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "The partner is not a timestamptz array",
                        )
                    })?;
                Ok(array.iter().map(|v| v.map(Literal::timestamptz)).collect())
            }
            PrimitiveType::TimestampNs => {
                let array = partner
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "The partner is not a timestamp_ns array",
                        )
                    })?;
                Ok(array
                    .iter()
                    .map(|v| v.map(Literal::timestamp_nano))
                    .collect())
            }
            PrimitiveType::TimestamptzNs => {
                let array = partner
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "The partner is not a timestamptz_ns array",
                        )
                    })?;
                Ok(array
                    .iter()
                    .map(|v| v.map(Literal::timestamptz_nano))
                    .collect())
            }
            PrimitiveType::String => {
                if let Some(array) = partner.as_any().downcast_ref::<LargeStringArray>() {
                    Ok(array.iter().map(|v| v.map(Literal::string)).collect())
                } else if let Some(array) = partner.as_any().downcast_ref::<StringArray>() {
                    Ok(array.iter().map(|v| v.map(Literal::string)).collect())
                } else if let Some(array) = partner.as_any().downcast_ref::<StringViewArray>() {
                    Ok(array.iter().map(|v| v.map(Literal::string)).collect())
                } else {
                    Err(Error::new(
                        ErrorKind::DataInvalid,
                        "The partner is not a string array",
                    ))
                }
            }
            PrimitiveType::Uuid => {
                if let Some(array) = partner.as_any().downcast_ref::<FixedSizeBinaryArray>() {
                    if array.value_length() != 16 {
                        return Err(Error::new(
                            ErrorKind::DataInvalid,
                            "The partner is not a uuid array",
                        ));
                    }
                    Ok(array
                        .iter()
                        .map(|v| {
                            v.map(|v| {
                                Ok(Literal::uuid(Uuid::from_bytes(v.try_into().map_err(
                                    |_| {
                                        Error::new(
                                            ErrorKind::DataInvalid,
                                            "Failed to convert binary to uuid",
                                        )
                                    },
                                )?)))
                            })
                            .transpose()
                        })
                        .collect::<Result<Vec<_>>>()?)
                } else {
                    Err(Error::new(
                        ErrorKind::DataInvalid,
                        "The partner is not a uuid array",
                    ))
                }
            }
            PrimitiveType::Fixed(len) => {
                let array = partner
                    .as_any()
                    .downcast_ref::<FixedSizeBinaryArray>()
                    .ok_or_else(|| {
                        Error::new(ErrorKind::DataInvalid, "The partner is not a fixed array")
                    })?;
                if array.value_length() != *len as i32 {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        "The length of fixed size binary array is not compatible with iceberg fixed type",
                    ));
                }
                Ok(array
                    .iter()
                    .map(|v| v.map(|v| Literal::fixed(v.iter().cloned())))
                    .collect())
            }
            PrimitiveType::Binary => {
                if let Some(array) = partner.as_any().downcast_ref::<LargeBinaryArray>() {
                    Ok(array
                        .iter()
                        .map(|v| v.map(|v| Literal::binary(v.to_vec())))
                        .collect())
                } else if let Some(array) = partner.as_any().downcast_ref::<BinaryArray>() {
                    Ok(array
                        .iter()
                        .map(|v| v.map(|v| Literal::binary(v.to_vec())))
                        .collect())
                } else if let Some(array) = partner.as_any().downcast_ref::<BinaryViewArray>() {
                    Ok(array
                        .iter()
                        .map(|v| v.map(|v| Literal::binary(v.to_vec())))
                        .collect())
                } else {
                    Err(Error::new(
                        ErrorKind::DataInvalid,
                        "The partner is not a binary array",
                    ))
                }
            }
            // DEFERRED: data-file read for the `unknown` type. `unknown` is an always-null column
            // with no physical storage (Java `TypeToMessageType` returns null — no parquet
            // column), so a full reader would synthesize a column of nulls without touching the
            // file. That data-path is a deeper change; defer it loudly here (mirroring the variant
            // Arrow loud-error) so a scan of an `unknown` column fails visibly rather than reading
            // arbitrary bytes. Metadata-only schema round-trips do not reach this path.
            PrimitiveType::Unknown => Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "Reading the unknown type from a data file is not supported yet: unknown is always null and has no physical column; this always-null read path is deferred",
            )),
        }
    }
}

/// Defines how Arrow fields are matched with Iceberg fields when converting data.
///
/// This enum provides two strategies for matching fields:
/// - `Id`: Match fields by their ID, which is stored in Arrow field metadata.
/// - `Name`: Match fields by their name, ignoring the field ID.
///
/// The ID matching mode is the default and preferred approach as it's more robust
/// against schema evolution where field names might change but IDs remain stable.
/// The name matching mode can be useful in scenarios where field IDs are not available
/// or when working with systems that don't preserve field IDs.
#[derive(Clone, Copy, Debug)]
pub enum FieldMatchMode {
    /// Match fields by their ID stored in Arrow field metadata
    Id,
    /// Match fields by their name, ignoring field IDs
    Name,
}

impl FieldMatchMode {
    /// Determines if an Arrow field matches an Iceberg field based on the matching mode.
    pub fn match_field(&self, arrow_field: &FieldRef, iceberg_field: &NestedField) -> bool {
        match self {
            FieldMatchMode::Id => get_field_id_from_metadata(arrow_field)
                .map(|id| id == iceberg_field.id)
                .unwrap_or(false),
            FieldMatchMode::Name => arrow_field.name() == &iceberg_field.name,
        }
    }
}

/// Partner type representing accessing and walking arrow arrays alongside iceberg schema
pub struct ArrowArrayAccessor {
    match_mode: FieldMatchMode,
}

impl ArrowArrayAccessor {
    /// Creates a new instance of ArrowArrayAccessor with the default ID matching mode
    pub fn new() -> Self {
        Self {
            match_mode: FieldMatchMode::Id,
        }
    }

    /// Creates a new instance of ArrowArrayAccessor with the specified matching mode
    pub fn new_with_match_mode(match_mode: FieldMatchMode) -> Self {
        Self { match_mode }
    }
}

impl Default for ArrowArrayAccessor {
    fn default() -> Self {
        Self::new()
    }
}

impl PartnerAccessor<ArrayRef> for ArrowArrayAccessor {
    fn struct_partner<'a>(&self, schema_partner: &'a ArrayRef) -> Result<&'a ArrayRef> {
        if !matches!(schema_partner.data_type(), DataType::Struct(_)) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "The schema partner is not a struct type",
            ));
        }

        Ok(schema_partner)
    }

    fn field_partner<'a>(
        &self,
        struct_partner: &'a ArrayRef,
        field: &NestedField,
    ) -> Result<&'a ArrayRef> {
        let struct_array = struct_partner
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "The struct partner is not a struct array, partner: {struct_partner:?}"
                    ),
                )
            })?;

        let field_pos = struct_array
            .fields()
            .iter()
            .position(|arrow_field| self.match_mode.match_field(arrow_field, field))
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Field id {} not found in struct array", field.id),
                )
            })?;

        Ok(struct_array.column(field_pos))
    }

    fn list_element_partner<'a>(&self, list_partner: &'a ArrayRef) -> Result<&'a ArrayRef> {
        match list_partner.data_type() {
            DataType::List(_) => {
                let list_array = list_partner
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "The list partner is not a list array",
                        )
                    })?;
                Ok(list_array.values())
            }
            DataType::LargeList(_) => {
                let list_array = list_partner
                    .as_any()
                    .downcast_ref::<LargeListArray>()
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "The list partner is not a large list array",
                        )
                    })?;
                Ok(list_array.values())
            }
            DataType::FixedSizeList(_, _) => {
                let list_array = list_partner
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "The list partner is not a fixed size list array",
                        )
                    })?;
                Ok(list_array.values())
            }
            _ => Err(Error::new(
                ErrorKind::DataInvalid,
                "The list partner is not a list type",
            )),
        }
    }

    fn map_key_partner<'a>(&self, map_partner: &'a ArrayRef) -> Result<&'a ArrayRef> {
        let map_array = map_partner
            .as_any()
            .downcast_ref::<MapArray>()
            .ok_or_else(|| {
                Error::new(ErrorKind::DataInvalid, "The map partner is not a map array")
            })?;
        Ok(map_array.keys())
    }

    fn map_value_partner<'a>(&self, map_partner: &'a ArrayRef) -> Result<&'a ArrayRef> {
        let map_array = map_partner
            .as_any()
            .downcast_ref::<MapArray>()
            .ok_or_else(|| {
                Error::new(ErrorKind::DataInvalid, "The map partner is not a map array")
            })?;
        Ok(map_array.values())
    }
}

/// Convert arrow struct array to iceberg struct value array.
/// This function will assume the schema of arrow struct array is the same as iceberg struct type.
pub fn arrow_struct_to_literal(
    struct_array: &ArrayRef,
    ty: &StructType,
) -> Result<Vec<Option<Literal>>> {
    visit_struct_with_partner(
        ty,
        struct_array,
        &mut ArrowArrayToIcebergStructConverter,
        &ArrowArrayAccessor::new(),
    )
}

/// Convert arrow primitive array to iceberg primitive value array.
/// This function will assume the schema of arrow struct array is the same as iceberg struct type.
pub fn arrow_primitive_to_literal(
    primitive_array: &ArrayRef,
    ty: &Type,
) -> Result<Vec<Option<Literal>>> {
    visit_type_with_partner(
        ty,
        primitive_array,
        &mut ArrowArrayToIcebergStructConverter,
        &ArrowArrayAccessor::new(),
    )
}

/// Create a single-element array from a primitive literal.
///
/// This is used for creating constant arrays (Run-End Encoded arrays) where we need
/// a single value that represents all rows.
pub(crate) fn create_primitive_array_single_element(
    data_type: &DataType,
    prim_lit: &Option<PrimitiveLiteral>,
) -> Result<ArrayRef> {
    match (data_type, prim_lit) {
        (DataType::Boolean, Some(PrimitiveLiteral::Boolean(v))) => {
            Ok(Arc::new(BooleanArray::from(vec![*v])))
        }
        (DataType::Boolean, None) => Ok(Arc::new(BooleanArray::from(vec![Option::<bool>::None]))),
        (DataType::Int32, Some(PrimitiveLiteral::Int(v))) => {
            Ok(Arc::new(Int32Array::from(vec![*v])))
        }
        (DataType::Int32, None) => Ok(Arc::new(Int32Array::from(vec![Option::<i32>::None]))),
        (DataType::Date32, Some(PrimitiveLiteral::Int(v))) => {
            Ok(Arc::new(Date32Array::from(vec![*v])))
        }
        (DataType::Date32, None) => Ok(Arc::new(Date32Array::from(vec![Option::<i32>::None]))),
        (DataType::Int64, Some(PrimitiveLiteral::Long(v))) => {
            Ok(Arc::new(Int64Array::from(vec![*v])))
        }
        (DataType::Int64, None) => Ok(Arc::new(Int64Array::from(vec![Option::<i64>::None]))),
        (DataType::Timestamp(TimeUnit::Microsecond, timezone), Some(PrimitiveLiteral::Long(v))) => {
            let array = TimestampMicrosecondArray::from(vec![*v]);
            if let Some(timezone) = timezone {
                Ok(Arc::new(array.with_timezone(timezone.clone())))
            } else {
                Ok(Arc::new(array))
            }
        }
        (DataType::Timestamp(TimeUnit::Microsecond, timezone), None) => {
            let array = TimestampMicrosecondArray::from(vec![Option::<i64>::None]);
            if let Some(timezone) = timezone {
                Ok(Arc::new(array.with_timezone(timezone.clone())))
            } else {
                Ok(Arc::new(array))
            }
        }
        (DataType::Timestamp(TimeUnit::Nanosecond, timezone), Some(PrimitiveLiteral::Long(v))) => {
            let array = TimestampNanosecondArray::from(vec![*v]);
            if let Some(timezone) = timezone {
                Ok(Arc::new(array.with_timezone(timezone.clone())))
            } else {
                Ok(Arc::new(array))
            }
        }
        (DataType::Timestamp(TimeUnit::Nanosecond, timezone), None) => {
            let array = TimestampNanosecondArray::from(vec![Option::<i64>::None]);
            if let Some(timezone) = timezone {
                Ok(Arc::new(array.with_timezone(timezone.clone())))
            } else {
                Ok(Arc::new(array))
            }
        }
        (DataType::Float32, Some(PrimitiveLiteral::Float(v))) => {
            Ok(Arc::new(Float32Array::from(vec![v.0])))
        }
        (DataType::Float32, None) => Ok(Arc::new(Float32Array::from(vec![Option::<f32>::None]))),
        (DataType::Float64, Some(PrimitiveLiteral::Double(v))) => {
            Ok(Arc::new(Float64Array::from(vec![v.0])))
        }
        (DataType::Float64, None) => Ok(Arc::new(Float64Array::from(vec![Option::<f64>::None]))),
        (DataType::Utf8, Some(PrimitiveLiteral::String(v))) => {
            Ok(Arc::new(StringArray::from(vec![v.as_str()])))
        }
        (DataType::Utf8, None) => Ok(Arc::new(StringArray::from(vec![Option::<&str>::None]))),
        (DataType::Binary, Some(PrimitiveLiteral::Binary(v))) => {
            Ok(Arc::new(BinaryArray::from_vec(vec![v.as_slice()])))
        }
        (DataType::Binary, None) => Ok(Arc::new(BinaryArray::from_opt_vec(vec![
            Option::<&[u8]>::None,
        ]))),
        (DataType::Decimal128(precision, scale), Some(PrimitiveLiteral::Int128(v))) => {
            let array = Decimal128Array::from(vec![{ *v }])
                .with_precision_and_scale(*precision, *scale)
                .map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Failed to create Decimal128Array with precision {precision} and scale {scale}: {e}"
                        ),
                    )
                })?;
            Ok(Arc::new(array))
        }
        (DataType::Decimal128(precision, scale), Some(PrimitiveLiteral::UInt128(v))) => {
            let array = Decimal128Array::from(vec![*v as i128])
                .with_precision_and_scale(*precision, *scale)
                .map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Failed to create Decimal128Array with precision {precision} and scale {scale}: {e}"
                        ),
                    )
                })?;
            Ok(Arc::new(array))
        }
        (DataType::Decimal128(precision, scale), None) => {
            let array = Decimal128Array::from(vec![Option::<i128>::None])
                .with_precision_and_scale(*precision, *scale)
                .map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Failed to create Decimal128Array with precision {precision} and scale {scale}: {e}"
                        ),
                    )
                })?;
            Ok(Arc::new(array))
        }
        (DataType::Struct(fields), None) => {
            // Create a single-element StructArray with nulls
            let null_arrays: Vec<ArrayRef> = fields
                .iter()
                .map(|f| {
                    // Recursively create null arrays for struct fields
                    // For primitive fields in structs, use simple null arrays (not REE within struct)
                    match f.data_type() {
                        DataType::Boolean => {
                            Ok(Arc::new(BooleanArray::from(vec![Option::<bool>::None]))
                                as ArrayRef)
                        }
                        DataType::Int32 | DataType::Date32 => {
                            Ok(Arc::new(Int32Array::from(vec![Option::<i32>::None])) as ArrayRef)
                        }
                        DataType::Int64 => {
                            Ok(Arc::new(Int64Array::from(vec![Option::<i64>::None])) as ArrayRef)
                        }
                        DataType::Timestamp(TimeUnit::Microsecond, timezone) => {
                            let array = TimestampMicrosecondArray::from(vec![Option::<i64>::None]);
                            if let Some(timezone) = timezone {
                                Ok(Arc::new(array.with_timezone(timezone.clone())) as ArrayRef)
                            } else {
                                Ok(Arc::new(array) as ArrayRef)
                            }
                        }
                        DataType::Timestamp(TimeUnit::Nanosecond, timezone) => {
                            let array = TimestampNanosecondArray::from(vec![Option::<i64>::None]);
                            if let Some(timezone) = timezone {
                                Ok(Arc::new(array.with_timezone(timezone.clone())) as ArrayRef)
                            } else {
                                Ok(Arc::new(array) as ArrayRef)
                            }
                        }
                        DataType::Float32 => {
                            Ok(Arc::new(Float32Array::from(vec![Option::<f32>::None])) as ArrayRef)
                        }
                        DataType::Float64 => {
                            Ok(Arc::new(Float64Array::from(vec![Option::<f64>::None])) as ArrayRef)
                        }
                        DataType::Utf8 => {
                            Ok(Arc::new(StringArray::from(vec![Option::<&str>::None])) as ArrayRef)
                        }
                        DataType::Binary => {
                            Ok(
                                Arc::new(BinaryArray::from_opt_vec(vec![Option::<&[u8]>::None]))
                                    as ArrayRef,
                            )
                        }
                        _ => Err(Error::new(
                            ErrorKind::Unexpected,
                            format!("Unsupported struct field type: {:?}", f.data_type()),
                        )),
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Arc::new(arrow_array::StructArray::new(
                fields.clone(),
                null_arrays,
                Some(arrow_buffer::NullBuffer::new_null(1)),
            )))
        }
        _ => Err(Error::new(
            ErrorKind::Unexpected,
            format!("Unsupported constant type combination: {data_type:?} with {prim_lit:?}"),
        )),
    }
}

/// Create a repeated array from a primitive literal for a given number of rows.
///
/// This is used for creating non-constant arrays where we need the same value
/// repeated for each row.
pub(crate) fn create_primitive_array_repeated(
    data_type: &DataType,
    prim_lit: &Option<PrimitiveLiteral>,
    num_rows: usize,
) -> Result<ArrayRef> {
    Ok(match (data_type, prim_lit) {
        (DataType::Boolean, Some(PrimitiveLiteral::Boolean(value))) => {
            Arc::new(BooleanArray::from(vec![*value; num_rows]))
        }
        (DataType::Boolean, None) => {
            let vals: Vec<Option<bool>> = vec![None; num_rows];
            Arc::new(BooleanArray::from(vals))
        }
        (DataType::Int32, Some(PrimitiveLiteral::Int(value))) => {
            Arc::new(Int32Array::from(vec![*value; num_rows]))
        }
        (DataType::Int32, None) => {
            let vals: Vec<Option<i32>> = vec![None; num_rows];
            Arc::new(Int32Array::from(vals))
        }
        (DataType::Date32, Some(PrimitiveLiteral::Int(value))) => {
            Arc::new(Date32Array::from(vec![*value; num_rows]))
        }
        (DataType::Date32, None) => {
            let vals: Vec<Option<i32>> = vec![None; num_rows];
            Arc::new(Date32Array::from(vals))
        }
        (DataType::Int64, Some(PrimitiveLiteral::Long(value))) => {
            Arc::new(Int64Array::from(vec![*value; num_rows]))
        }
        (DataType::Int64, None) => {
            let vals: Vec<Option<i64>> = vec![None; num_rows];
            Arc::new(Int64Array::from(vals))
        }
        (
            DataType::Timestamp(TimeUnit::Microsecond, timezone),
            Some(PrimitiveLiteral::Long(value)),
        ) => {
            let array = TimestampMicrosecondArray::from(vec![*value; num_rows]);
            if let Some(timezone) = timezone {
                Arc::new(array.with_timezone(timezone.clone()))
            } else {
                Arc::new(array)
            }
        }
        (DataType::Timestamp(TimeUnit::Microsecond, timezone), None) => {
            let vals: Vec<Option<i64>> = vec![None; num_rows];
            let array = TimestampMicrosecondArray::from(vals);
            if let Some(timezone) = timezone {
                Arc::new(array.with_timezone(timezone.clone()))
            } else {
                Arc::new(array)
            }
        }
        (
            DataType::Timestamp(TimeUnit::Nanosecond, timezone),
            Some(PrimitiveLiteral::Long(value)),
        ) => {
            let array = TimestampNanosecondArray::from(vec![*value; num_rows]);
            if let Some(timezone) = timezone {
                Arc::new(array.with_timezone(timezone.clone()))
            } else {
                Arc::new(array)
            }
        }
        (DataType::Timestamp(TimeUnit::Nanosecond, timezone), None) => {
            let vals: Vec<Option<i64>> = vec![None; num_rows];
            let array = TimestampNanosecondArray::from(vals);
            if let Some(timezone) = timezone {
                Arc::new(array.with_timezone(timezone.clone()))
            } else {
                Arc::new(array)
            }
        }
        (DataType::Float32, Some(PrimitiveLiteral::Float(value))) => {
            Arc::new(Float32Array::from(vec![value.0; num_rows]))
        }
        (DataType::Float32, None) => {
            let vals: Vec<Option<f32>> = vec![None; num_rows];
            Arc::new(Float32Array::from(vals))
        }
        (DataType::Float64, Some(PrimitiveLiteral::Double(value))) => {
            Arc::new(Float64Array::from(vec![value.0; num_rows]))
        }
        (DataType::Float64, None) => {
            let vals: Vec<Option<f64>> = vec![None; num_rows];
            Arc::new(Float64Array::from(vals))
        }
        (DataType::Utf8, Some(PrimitiveLiteral::String(value))) => {
            Arc::new(StringArray::from(vec![value.clone(); num_rows]))
        }
        (DataType::Utf8, None) => {
            let vals: Vec<Option<String>> = vec![None; num_rows];
            Arc::new(StringArray::from(vals))
        }
        (DataType::Binary, Some(PrimitiveLiteral::Binary(value))) => {
            Arc::new(BinaryArray::from_vec(vec![value; num_rows]))
        }
        (DataType::Binary, None) => {
            let vals: Vec<Option<&[u8]>> = vec![None; num_rows];
            Arc::new(BinaryArray::from_opt_vec(vals))
        }
        (DataType::Decimal128(precision, scale), Some(PrimitiveLiteral::Int128(value))) => {
            Arc::new(
                Decimal128Array::from(vec![*value; num_rows])
                    .with_precision_and_scale(*precision, *scale)
                    .map_err(|e| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "Failed to create Decimal128Array with precision {precision} and scale {scale}: {e}"
                            ),
                        )
                    })?,
            )
        }
        (DataType::Decimal128(precision, scale), Some(PrimitiveLiteral::UInt128(value))) => {
            Arc::new(
                Decimal128Array::from(vec![*value as i128; num_rows])
                    .with_precision_and_scale(*precision, *scale)
                    .map_err(|e| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "Failed to create Decimal128Array with precision {precision} and scale {scale}: {e}"
                            ),
                        )
                    })?,
            )
        }
        (DataType::Decimal128(precision, scale), None) => {
            let vals: Vec<Option<i128>> = vec![None; num_rows];
            Arc::new(
                Decimal128Array::from(vals)
                    .with_precision_and_scale(*precision, *scale)
                    .map_err(|e| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "Failed to create Decimal128Array with precision {precision} and scale {scale}: {e}"
                            ),
                        )
                    })?,
            )
        }
        (DataType::Struct(fields), None) => {
            let null_arrays: Vec<ArrayRef> = fields
                .iter()
                .map(|field| create_primitive_array_repeated(field.data_type(), &None, num_rows))
                .collect::<Result<Vec<_>>>()?;

            Arc::new(StructArray::new(
                fields.clone(),
                null_arrays,
                Some(NullBuffer::new_null(num_rows)),
            ))
        }
        (DataType::Time64(TimeUnit::Microsecond), Some(PrimitiveLiteral::Long(value))) => {
            Arc::new(Time64MicrosecondArray::from(vec![*value; num_rows]))
        }
        (DataType::Time64(TimeUnit::Microsecond), None) => {
            let vals: Vec<Option<i64>> = vec![None; num_rows];
            Arc::new(Time64MicrosecondArray::from(vals))
        }
        (DataType::LargeBinary, Some(PrimitiveLiteral::Binary(value))) => {
            Arc::new(LargeBinaryArray::from_vec(vec![value; num_rows]))
        }
        (DataType::LargeBinary, None) => {
            let vals: Vec<Option<&[u8]>> = vec![None; num_rows];
            Arc::new(LargeBinaryArray::from_opt_vec(vals))
        }
        (DataType::FixedSizeBinary(size), Some(PrimitiveLiteral::Binary(value))) => {
            Arc::new(fixed_size_binary_column(*size, value, num_rows)?)
        }
        (DataType::FixedSizeBinary(size), Some(PrimitiveLiteral::UInt128(value))) => {
            Arc::new(fixed_size_binary_column(*size, &value.to_be_bytes(), num_rows)?)
        }
        (DataType::FixedSizeBinary(_) | DataType::List(_) | DataType::Map(..), None) => {
            new_null_array(data_type, num_rows)
        }
        (DataType::Null, _) => Arc::new(arrow_array::NullArray::new(num_rows)),
        (dt, _) => {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!("unexpected target column type {dt}"),
            ));
        }
    })
}

fn fixed_size_binary_column(
    size: i32,
    value: &[u8],
    num_rows: usize,
) -> Result<FixedSizeBinaryArray> {
    if value.len() as i32 != size {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "FixedSizeBinary({size}) cannot hold a {}-byte value",
                value.len()
            ),
        ));
    }
    Ok(FixedSizeBinaryArray::new(
        size,
        Buffer::from(value.repeat(num_rows)),
        None,
    ))
}

#[cfg(test)]
#[path = "value_tests.rs"]
mod test;
