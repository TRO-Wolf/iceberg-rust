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

//! Conversion between Arrow schema and Iceberg schema.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::types::{Decimal128Type, validate_decimal_precision_and_scale};
use arrow_array::{
    BinaryArray, BooleanArray, Date32Array, Datum as ArrowDatum, Decimal128Array,
    FixedSizeBinaryArray, Float32Array, Float64Array, Int32Array, Int64Array, Scalar, StringArray,
    Time64MicrosecondArray, TimestampMicrosecondArray, TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema, TimeUnit};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::file::statistics::Statistics;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::decimal_utils::i128_from_be_bytes;
use crate::spec::{
    Datum, FIRST_FIELD_ID, ListType, MapType, NestedField, NestedFieldRef, PrimitiveLiteral,
    PrimitiveType, Schema, SchemaVisitor, StructType, Type,
};
use crate::{Error, ErrorKind};

/// The canonical Arrow extension-type name for a Parquet/Iceberg variant, and the two field names
/// its struct carries. Taken from `parquet_variant_compute::VariantType` (`NAME`) and the Arrow
/// canonical-extension-type spec rather than invented here; re-stated as constants so this module
/// does not depend on the experimental crate for a string.
pub(crate) const VARIANT_EXTENSION_NAME: &str = "arrow.parquet.variant";
/// The extension metadata for a variant is the EMPTY STRING, not an absent key — a reader that
/// checks only for the name is lenient, one that requires absence is wrong.
pub(crate) const VARIANT_EXTENSION_METADATA: &str = "";
pub(crate) const VARIANT_METADATA_FIELD: &str = "metadata";
pub(crate) const VARIANT_VALUE_FIELD: &str = "value";
const ARROW_EXTENSION_NAME_KEY: &str = "ARROW:extension:name";
const ARROW_EXTENSION_METADATA_KEY: &str = "ARROW:extension:metadata";

/// The Arrow `DataType` of a variant: `Struct<metadata: Binary, value: Binary>`.
///
/// Both children are NON-NULLABLE. A variant that is absent is a null at the struct level; a
/// struct whose `metadata` child is null is malformed, because variant metadata is what makes the
/// value decodable at all.
pub(crate) fn variant_arrow_data_type() -> DataType {
    DataType::Struct(Fields::from(vec![
        Field::new(VARIANT_METADATA_FIELD, DataType::Binary, false),
        Field::new(VARIANT_VALUE_FIELD, DataType::Binary, false),
    ]))
}

/// Depth bound for [`variant_path_within`], mirroring `spec::schema::visitor`'s
/// `MAX_SCHEMA_NESTING_DEPTH`: an unbounded recursive walk over a partner-supplied metadata file
/// overflows the stack. No constructible schema reaches it — the builder refuses first.
pub(crate) const MAX_VARIANT_NESTING_DEPTH: usize = 128;

/// The dotted path to the first `variant` at or beneath `ty`, or `None`.
///
/// ONE walk, shared by the read guard (`ArrowReader::reject_variant_projection`) and the write
/// guard (`ParquetWriterBuilder::build`); as two copies they drifted in coverage.
///
/// All four container positions are descended — struct field, list element, map KEY and map value.
/// The key is not skippable: Java constrains only a map's VALUE type, so `map<variant, _>` is
/// legal. Bounded by [`MAX_VARIANT_NESTING_DEPTH`], past which it reports "no variant" rather than
/// erroring — a type nested that deep is rejected by the schema builder that owns that rule.
pub(crate) fn variant_path_within(name: &str, ty: &Type) -> Option<String> {
    fn walk(name: &str, ty: &Type, depth: usize) -> Option<String> {
        if depth > MAX_VARIANT_NESTING_DEPTH {
            return None;
        }
        let next = depth + 1;
        match ty {
            Type::Variant => Some(name.to_string()),
            Type::Struct(struct_type) => struct_type.fields().iter().find_map(|nested| {
                walk(
                    &format!("{name}.{}", nested.name),
                    nested.field_type.as_ref(),
                    next,
                )
            }),
            Type::List(list) => walk(
                &format!("{name}.element"),
                list.element_field.field_type.as_ref(),
                next,
            ),
            Type::Map(map) => walk(
                &format!("{name}.key"),
                map.key_field.field_type.as_ref(),
                next,
            )
            .or_else(|| {
                walk(
                    &format!("{name}.value"),
                    map.value_field.field_type.as_ref(),
                    next,
                )
            }),
            Type::Primitive(_) => None,
        }
    }
    walk(name, ty, 0)
}

/// Whether an Arrow field is a variant — i.e. carries the canonical extension name.
///
/// Keyed on the extension NAME alone. The metadata value is not part of the identity check: the
/// spec fixes it to the empty string, and a writer that omitted the key entirely still produced a
/// variant. Requiring the metadata key would reject those files.
pub(crate) fn is_variant_arrow_field(field: &Field) -> bool {
    field
        .metadata()
        .get(ARROW_EXTENSION_NAME_KEY)
        .map(String::as_str)
        == Some(VARIANT_EXTENSION_NAME)
}

/// Stamp the canonical variant extension metadata onto a field's metadata map.
pub(crate) fn with_variant_extension_metadata(
    mut meta: HashMap<String, String>,
) -> HashMap<String, String> {
    meta.insert(
        ARROW_EXTENSION_NAME_KEY.to_string(),
        VARIANT_EXTENSION_NAME.to_string(),
    );
    meta.insert(
        ARROW_EXTENSION_METADATA_KEY.to_string(),
        VARIANT_EXTENSION_METADATA.to_string(),
    );
    meta
}

/// When iceberg map type convert to Arrow map type, the default map field name is "key_value".
pub const DEFAULT_MAP_FIELD_NAME: &str = "key_value";
/// UTC timezone annotation produced for Iceberg `timestamptz` / `timestamptz_ns`
/// Arrow fields. Matches Spark `toArrow` (`timestamp[us, tz=UTC]`).
///
/// The historical offset spelling `"+00:00"` is still **accepted** on the
/// Arrow→Iceberg inverse (see [`is_utc_time_zone`]); it is never produced.
pub const UTC_TIME_ZONE: &str = "UTC";

/// Historical offset-form UTC alias this crate used to emit on Iceberg→Arrow.
///
/// Still accepted as Iceberg `timestamptz` so files and batches tagged under
/// the old annotation continue to resolve. Never produced.
pub const UTC_OFFSET_TIME_ZONE: &str = "+00:00";

/// True if `zone` is an accepted UTC alias for Iceberg `timestamptz`.
///
/// The produced annotation is [`UTC_TIME_ZONE`]. [`UTC_OFFSET_TIME_ZONE`] remains
/// accepted so the inverse mapping is never narrowed.
#[inline]
pub fn is_utc_time_zone(zone: &str) -> bool {
    zone == UTC_TIME_ZONE || zone == UTC_OFFSET_TIME_ZONE
}

/// Maximum Arrow schema-type nesting depth the visitor will descend.
///
/// Arrow schemas can be constructed directly by callers and may therefore contain attacker-
/// influenced nesting. Keep this aligned with the Iceberg schema visitor's 128-level policy: a
/// type root is at depth `0`, while fields of a schema's implicit root struct are at depth `1`.
const MAX_ARROW_SCHEMA_NESTING_DEPTH: usize = 128;

fn decimal128_precision_and_scale(precision: u32, scale: u32, context: &str) -> Result<(u8, i8)> {
    let precision = u8::try_from(precision).map_err(|err| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("{context}: decimal precision is out of Arrow Decimal128 range"),
        )
        .with_source(err)
    })?;
    let scale = i8::try_from(scale).map_err(|err| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("{context}: decimal scale is out of Arrow Decimal128 range"),
        )
        .with_source(err)
    })?;

    validate_decimal_precision_and_scale::<Decimal128Type>(precision, scale).map_err(|err| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("{context}: decimal precision/scale is not valid for Arrow Decimal128"),
        )
        .with_source(err)
    })?;

    Ok((precision, scale))
}

fn decimal128_arrow_type(precision: u32, scale: u32, context: &str) -> Result<DataType> {
    let (precision, scale) = decimal128_precision_and_scale(precision, scale, context)?;

    Ok(DataType::Decimal128(precision, scale))
}

/// A post order arrow schema visitor.
///
/// For order of methods called, please refer to [`visit_schema`].
pub trait ArrowSchemaVisitor {
    /// Return type of this visitor on arrow field.
    type T;

    /// Return type of this visitor on arrow schema.
    type U;

    /// Called for every field BEFORE its data type is descended into; returning `Some` short-
    /// circuits the walk for that field.
    ///
    /// This exists for the canonical Arrow variant extension type, a `Struct` carrying the
    /// `arrow.parquet.variant` name on the FIELD. Its `metadata` / `value` children are components
    /// of one Iceberg field, not Iceberg fields, so descending would fail on their missing ids.
    fn variant_field(&mut self, _field: &Field) -> Result<Option<Self::T>> {
        Ok(None)
    }

    /// Called before struct/list/map field.
    fn before_field(&mut self, _field: &Field) -> Result<()> {
        Ok(())
    }

    /// Called after struct/list/map field.
    fn after_field(&mut self, _field: &Field) -> Result<()> {
        Ok(())
    }

    /// Called before list element.
    fn before_list_element(&mut self, _field: &Field) -> Result<()> {
        Ok(())
    }

    /// Called after list element.
    fn after_list_element(&mut self, _field: &Field) -> Result<()> {
        Ok(())
    }

    /// Called before map key.
    fn before_map_key(&mut self, _field: &Field) -> Result<()> {
        Ok(())
    }

    /// Called after map key.
    fn after_map_key(&mut self, _field: &Field) -> Result<()> {
        Ok(())
    }

    /// Called before map value.
    fn before_map_value(&mut self, _field: &Field) -> Result<()> {
        Ok(())
    }

    /// Called after map value.
    fn after_map_value(&mut self, _field: &Field) -> Result<()> {
        Ok(())
    }

    /// Called after schema's type visited.
    fn schema(&mut self, schema: &ArrowSchema, values: Vec<Self::T>) -> Result<Self::U>;

    /// Called after struct's fields visited.
    fn r#struct(&mut self, fields: &Fields, results: Vec<Self::T>) -> Result<Self::T>;

    /// Called after list fields visited.
    fn list(&mut self, list: &DataType, value: Self::T) -> Result<Self::T>;

    /// Called after map's key and value fields visited.
    fn map(&mut self, map: &DataType, key_value: Self::T, value: Self::T) -> Result<Self::T>;

    /// Called when see a primitive type.
    fn primitive(&mut self, p: &DataType) -> Result<Self::T>;
}

/// Visiting a type in post order.
fn visit_type<V: ArrowSchemaVisitor>(r#type: &DataType, visitor: &mut V) -> Result<V::T> {
    visit_type_at_depth(r#type, visitor, 0)
}

/// Depth-bounded body of [`visit_type`]. Each struct field, list element, map key/value, and
/// dictionary value advances the depth by one.
fn visit_type_at_depth<V: ArrowSchemaVisitor>(
    r#type: &DataType,
    visitor: &mut V,
    depth: usize,
) -> Result<V::T> {
    if depth > MAX_ARROW_SCHEMA_NESTING_DEPTH {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Arrow schema type nesting exceeds maximum depth {MAX_ARROW_SCHEMA_NESTING_DEPTH}"
            ),
        ));
    }

    match r#type {
        p @ (DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float16
        | DataType::Float32
        | DataType::Float64
        | DataType::Timestamp(_, _)
        | DataType::Date32
        | DataType::Date64
        | DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::Duration(_)
        | DataType::Interval(_)
        | DataType::Binary
        | DataType::FixedSizeBinary(_)
        | DataType::LargeBinary
        | DataType::BinaryView
        | DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Decimal32(_, _)
        | DataType::Decimal64(_, _)
        | DataType::Decimal128(_, _)
        | DataType::Decimal256(_, _)) => visitor.primitive(p),
        DataType::List(element_field) => visit_list(r#type, element_field, visitor, depth),
        DataType::LargeList(element_field) => visit_list(r#type, element_field, visitor, depth),
        DataType::FixedSizeList(element_field, _) => {
            visit_list(r#type, element_field, visitor, depth)
        }
        DataType::Map(field, _) => match field.data_type() {
            DataType::Struct(fields) => {
                if fields.len() != 2 {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        "Map field must have exactly 2 fields",
                    ));
                }

                let key_field = &fields[0];
                let value_field = &fields[1];

                let key_result = {
                    visitor.before_map_key(key_field)?;
                    // A variant map KEY is short-circuited like every other position. Java places
                    // no type constraint on a map key (`Types$MapType.ofOptional` null-checks only
                    // the VALUE), so `map<variant, _>` is constructible and must round-trip.
                    let ret = match visitor.variant_field(key_field)? {
                        Some(ret) => ret,
                        None => visit_type_at_depth(key_field.data_type(), visitor, depth + 1)?,
                    };
                    visitor.after_map_key(key_field)?;
                    ret
                };

                let value_result = {
                    visitor.before_map_value(value_field)?;
                    // A variant map VALUE is short-circuited like a variant field, list element
                    // or map key.
                    let ret = match visitor.variant_field(value_field)? {
                        Some(ret) => ret,
                        None => visit_type_at_depth(value_field.data_type(), visitor, depth + 1)?,
                    };
                    visitor.after_map_value(value_field)?;
                    ret
                };

                visitor.map(r#type, key_result, value_result)
            }
            _ => Err(Error::new(
                ErrorKind::DataInvalid,
                "Map field must have struct type",
            )),
        },
        DataType::Struct(fields) => visit_struct(fields, visitor, depth),
        DataType::Dictionary(_key_type, value_type) => {
            visit_type_at_depth(value_type, visitor, depth + 1)
        }
        // These Arrow types contain recursively formatted child types in their Display
        // implementations. Keep their diagnostics static: formatting an attacker-controlled,
        // deeply nested child here can overflow the stack before the typed error is returned.
        DataType::ListView(_) => Err(Error::new(
            ErrorKind::DataInvalid,
            "Cannot visit Arrow data type: ListView",
        )),
        DataType::LargeListView(_) => Err(Error::new(
            ErrorKind::DataInvalid,
            "Cannot visit Arrow data type: LargeListView",
        )),
        DataType::Union(_, _) => Err(Error::new(
            ErrorKind::DataInvalid,
            "Cannot visit Arrow data type: Union",
        )),
        DataType::RunEndEncoded(_, _) => Err(Error::new(
            ErrorKind::DataInvalid,
            "Cannot visit Arrow data type: RunEndEncoded",
        )),
        DataType::Null => Err(Error::new(
            ErrorKind::DataInvalid,
            "Cannot visit Arrow data type: Null",
        )),
    }
}

/// Visit list types in post order.
fn visit_list<V: ArrowSchemaVisitor>(
    data_type: &DataType,
    element_field: &Field,
    visitor: &mut V,
    depth: usize,
) -> Result<V::T> {
    visitor.before_list_element(element_field)?;
    // A variant ELEMENT is short-circuited exactly like a variant field: descending would fail on
    // its id-less `metadata` / `value` children and would yield a struct even if it succeeded.
    let value = match visitor.variant_field(element_field)? {
        Some(value) => value,
        None => visit_type_at_depth(element_field.data_type(), visitor, depth + 1)?,
    };
    visitor.after_list_element(element_field)?;
    visitor.list(data_type, value)
}

/// Visit struct type in post order.
fn visit_struct<V: ArrowSchemaVisitor>(
    fields: &Fields,
    visitor: &mut V,
    depth: usize,
) -> Result<V::T> {
    let mut results = Vec::with_capacity(fields.len());
    for field in fields {
        visitor.before_field(field)?;
        let result = match visitor.variant_field(field)? {
            Some(result) => result,
            None => visit_type_at_depth(field.data_type(), visitor, depth + 1)?,
        };
        visitor.after_field(field)?;
        results.push(result);
    }

    visitor.r#struct(fields, results)
}

/// Visit schema in post order.
pub(crate) fn visit_schema<V: ArrowSchemaVisitor>(
    schema: &ArrowSchema,
    visitor: &mut V,
) -> Result<V::U> {
    let mut results = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        visitor.before_field(field)?;
        // An Arrow schema is an implicit root struct at depth 0, matching Iceberg's schema visitor.
        let result = match visitor.variant_field(field)? {
            Some(result) => result,
            None => visit_type_at_depth(field.data_type(), visitor, 1)?,
        };
        visitor.after_field(field)?;
        results.push(result);
    }
    visitor.schema(schema, results)
}

/// Convert Arrow schema to Iceberg schema.
///
/// Iceberg schema fields require a unique field id, and this function assumes that each field
/// in the provided Arrow schema contains a field id in its metadata. If the metadata is missing
/// or the field id is not set, the conversion will fail
pub fn arrow_schema_to_schema(schema: &ArrowSchema) -> Result<Schema> {
    let mut visitor = ArrowSchemaConverter::new();
    visit_schema(schema, &mut visitor)
}

/// Convert Arrow schema to Iceberg schema with automatically assigned field IDs.
///
/// Unlike [`arrow_schema_to_schema`], this function does not require field IDs in the Arrow
/// schema metadata. Instead, it automatically assigns unique field IDs starting from 1,
/// following Iceberg's field ID assignment rules.
///
/// This is useful when converting Arrow schemas that don't originate from Iceberg tables,
/// such as schemas from DataFusion or other Arrow-based systems.
pub fn arrow_schema_to_schema_auto_assign_ids(schema: &ArrowSchema) -> Result<Schema> {
    let mut visitor = ArrowSchemaConverter::new_with_field_ids_from(FIRST_FIELD_ID);
    visit_schema(schema, &mut visitor)
}

/// Convert Arrow type to iceberg type.
pub fn arrow_type_to_type(ty: &DataType) -> Result<Type> {
    let mut visitor = ArrowSchemaConverter::new();
    visit_type(ty, &mut visitor)
}

const ARROW_FIELD_DOC_KEY: &str = "doc";

pub(super) fn get_field_id_from_metadata(field: &Field) -> Result<i32> {
    if let Some(value) = field.metadata().get(PARQUET_FIELD_ID_META_KEY) {
        return value.parse::<i32>().map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                "Failed to parse field id".to_string(),
            )
            .with_context("value", value)
            .with_source(e)
        });
    }
    Err(Error::new(
        ErrorKind::DataInvalid,
        "Field id not found in metadata",
    ))
}

fn get_field_doc(field: &Field) -> Option<String> {
    if let Some(value) = field.metadata().get(ARROW_FIELD_DOC_KEY) {
        return Some(value.clone());
    }
    None
}

struct ArrowSchemaConverter {
    /// When set, the schema builder will reassign field IDs starting from this value
    /// using level-order traversal (breadth-first).
    reassign_field_ids_from: Option<i32>,
    /// Generates unique placeholder IDs for fields before reassignment.
    /// Required because `ReassignFieldIds` builds an old-to-new ID mapping
    /// that expects unique input IDs.
    next_field_id: i32,
}

impl ArrowSchemaConverter {
    fn new() -> Self {
        Self {
            reassign_field_ids_from: None,
            next_field_id: 0,
        }
    }

    fn new_with_field_ids_from(start_from: i32) -> Self {
        Self {
            reassign_field_ids_from: Some(start_from),
            next_field_id: 0,
        }
    }

    fn get_field_id(&mut self, field: &Field) -> Result<i32> {
        if self.reassign_field_ids_from.is_some() {
            // Field IDs will be reassigned by the schema builder.
            // We need unique temporary IDs because ReassignFieldIds builds an
            // old->new ID mapping that requires unique input IDs.
            let temp_id = self.next_field_id;
            self.next_field_id += 1;
            Ok(temp_id)
        } else {
            // Get field ID from arrow field metadata
            get_field_id_from_metadata(field)
        }
    }

    fn convert_fields(
        &mut self,
        fields: &Fields,
        field_results: &[Type],
    ) -> Result<Vec<NestedFieldRef>> {
        let mut results = Vec::with_capacity(fields.len());
        for i in 0..fields.len() {
            let field = &fields[i];
            let field_type = &field_results[i];
            let id = self.get_field_id(field)?;
            let doc = get_field_doc(field);
            let nested_field = NestedField {
                id,
                doc,
                name: field.name().clone(),
                required: !field.is_nullable(),
                field_type: Box::new(field_type.clone()),
                initial_default: None,
                write_default: None,
            };
            results.push(Arc::new(nested_field));
        }
        Ok(results)
    }
}

impl ArrowSchemaVisitor for ArrowSchemaConverter {
    type T = Type;
    type U = Schema;

    fn schema(&mut self, schema: &ArrowSchema, values: Vec<Self::T>) -> Result<Self::U> {
        let fields = self.convert_fields(schema.fields(), &values)?;
        let mut builder = Schema::builder().with_fields(fields);
        if let Some(start_from) = self.reassign_field_ids_from {
            builder = builder.with_reassigned_field_ids(start_from)
        }
        builder.build()
    }

    /// A field carrying the canonical variant extension name IS the Iceberg variant type — the
    /// mirror of `SchemaToType.variant` returning `Types.VariantType.get()` on the Avro side
    /// (a bare `VariantType.get()` return).
    fn variant_field(&mut self, field: &Field) -> Result<Option<Self::T>> {
        Ok(is_variant_arrow_field(field).then_some(Type::Variant))
    }

    fn r#struct(&mut self, fields: &Fields, results: Vec<Self::T>) -> Result<Self::T> {
        let fields = self.convert_fields(fields, &results)?;
        Ok(Type::Struct(StructType::new(fields)))
    }

    fn list(&mut self, list: &DataType, value: Self::T) -> Result<Self::T> {
        let element_field = match list {
            DataType::List(element_field) => element_field,
            DataType::LargeList(element_field) => element_field,
            DataType::FixedSizeList(element_field, _) => element_field,
            _ => {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "List type must have list data type",
                ));
            }
        };

        let id = self.get_field_id(element_field)?;
        let doc = get_field_doc(element_field);
        let mut element_field =
            NestedField::list_element(id, value.clone(), !element_field.is_nullable());
        if let Some(doc) = doc {
            element_field = element_field.with_doc(doc);
        }
        let element_field = Arc::new(element_field);
        Ok(Type::List(ListType { element_field }))
    }

    fn map(&mut self, map: &DataType, key_value: Self::T, value: Self::T) -> Result<Self::T> {
        match map {
            DataType::Map(field, _) => match field.data_type() {
                DataType::Struct(fields) => {
                    if fields.len() != 2 {
                        return Err(Error::new(
                            ErrorKind::DataInvalid,
                            "Map field must have exactly 2 fields",
                        ));
                    }

                    let key_field = &fields[0];
                    let value_field = &fields[1];

                    let key_id = self.get_field_id(key_field)?;
                    let key_doc = get_field_doc(key_field);
                    let mut key_field = NestedField::map_key_element(key_id, key_value.clone());
                    if let Some(doc) = key_doc {
                        key_field = key_field.with_doc(doc);
                    }
                    let key_field = Arc::new(key_field);

                    let value_id = self.get_field_id(value_field)?;
                    let value_doc = get_field_doc(value_field);
                    let mut value_field = NestedField::map_value_element(
                        value_id,
                        value.clone(),
                        !value_field.is_nullable(),
                    );
                    if let Some(doc) = value_doc {
                        value_field = value_field.with_doc(doc);
                    }
                    let value_field = Arc::new(value_field);

                    Ok(Type::Map(MapType {
                        key_field,
                        value_field,
                    }))
                }
                _ => Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Map field must have struct type",
                )),
            },
            _ => Err(Error::new(
                ErrorKind::DataInvalid,
                "Map type must have map data type",
            )),
        }
    }

    fn primitive(&mut self, p: &DataType) -> Result<Self::T> {
        match p {
            DataType::Boolean => Ok(Type::Primitive(PrimitiveType::Boolean)),
            DataType::Int8 | DataType::Int16 | DataType::Int32 => {
                Ok(Type::Primitive(PrimitiveType::Int))
            }
            DataType::UInt8 | DataType::UInt16 => Ok(Type::Primitive(PrimitiveType::Int)),
            DataType::UInt32 => Ok(Type::Primitive(PrimitiveType::Long)),
            DataType::Int64 => Ok(Type::Primitive(PrimitiveType::Long)),
            DataType::UInt64 => {
                // Block uint64 - no safe casting option
                Err(Error::new(
                    ErrorKind::DataInvalid,
                    "UInt64 is not supported. Use Int64 for values ≤ 9,223,372,036,854,775,807 or Decimal(20,0) for full uint64 range.",
                ))
            }
            DataType::Float32 => Ok(Type::Primitive(PrimitiveType::Float)),
            DataType::Float64 => Ok(Type::Primitive(PrimitiveType::Double)),
            DataType::Decimal128(p, s) => {
                let scale = u32::try_from(*s).map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Arrow decimal scale must be non-negative: {s}"),
                    )
                    .with_source(e)
                })?;
                Type::decimal(u32::from(*p), scale).map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        "Failed to create decimal type".to_string(),
                    )
                    .with_source(e)
                })
            }
            DataType::Date32 => Ok(Type::Primitive(PrimitiveType::Date)),
            DataType::Time64(unit) if unit == &TimeUnit::Microsecond => {
                Ok(Type::Primitive(PrimitiveType::Time))
            }
            DataType::Timestamp(unit, None) if unit == &TimeUnit::Microsecond => {
                Ok(Type::Primitive(PrimitiveType::Timestamp))
            }
            DataType::Timestamp(unit, None) if unit == &TimeUnit::Nanosecond => {
                Ok(Type::Primitive(PrimitiveType::TimestampNs))
            }
            DataType::Timestamp(unit, Some(zone))
                if unit == &TimeUnit::Microsecond && is_utc_time_zone(zone.as_ref()) =>
            {
                Ok(Type::Primitive(PrimitiveType::Timestamptz))
            }
            DataType::Timestamp(unit, Some(zone))
                if unit == &TimeUnit::Nanosecond && is_utc_time_zone(zone.as_ref()) =>
            {
                Ok(Type::Primitive(PrimitiveType::TimestamptzNs))
            }
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView => {
                Ok(Type::Primitive(PrimitiveType::Binary))
            }
            DataType::FixedSizeBinary(width) => {
                Ok(Type::Primitive(PrimitiveType::Fixed(*width as u64)))
            }
            DataType::Utf8View | DataType::Utf8 | DataType::LargeUtf8 => {
                Ok(Type::Primitive(PrimitiveType::String))
            }
            _ => Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Unsupported Arrow data type: {p}"),
            )),
        }
    }
}

struct ToArrowSchemaConverter;

enum ArrowSchemaOrFieldOrType {
    Schema(ArrowSchema),
    Field(Field),
    Type(DataType),
}

impl SchemaVisitor for ToArrowSchemaConverter {
    type T = ArrowSchemaOrFieldOrType;

    fn schema(
        &mut self,
        _schema: &crate::spec::Schema,
        value: ArrowSchemaOrFieldOrType,
    ) -> crate::Result<ArrowSchemaOrFieldOrType> {
        let struct_type = match value {
            ArrowSchemaOrFieldOrType::Type(DataType::Struct(fields)) => fields,
            _ => unreachable!(),
        };
        Ok(ArrowSchemaOrFieldOrType::Schema(ArrowSchema::new(
            struct_type,
        )))
    }

    fn field(
        &mut self,
        field: &crate::spec::NestedFieldRef,
        value: ArrowSchemaOrFieldOrType,
    ) -> crate::Result<ArrowSchemaOrFieldOrType> {
        let ty = match value {
            ArrowSchemaOrFieldOrType::Type(ty) => ty,
            _ => unreachable!(),
        };
        let metadata = if let Some(doc) = &field.doc {
            HashMap::from([
                (PARQUET_FIELD_ID_META_KEY.to_string(), field.id.to_string()),
                (ARROW_FIELD_DOC_KEY.to_string(), doc.clone()),
            ])
        } else {
            HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), field.id.to_string())])
        };
        // A variant field carries the canonical Arrow extension metadata ALONGSIDE its field id.
        // Stamped here rather than in `variant()` because that method returns a `DataType`, and
        // the extension name lives on the FIELD — a variant is identified by its field metadata,
        // never by its struct shape (a plain `{metadata, value}` struct is not a variant).
        let metadata = if matches!(field.field_type.as_ref(), crate::spec::Type::Variant) {
            with_variant_extension_metadata(metadata)
        } else {
            metadata
        };
        Ok(ArrowSchemaOrFieldOrType::Field(
            Field::new(field.name.clone(), ty, !field.required).with_metadata(metadata),
        ))
    }

    fn r#struct(
        &mut self,
        _: &crate::spec::StructType,
        results: Vec<ArrowSchemaOrFieldOrType>,
    ) -> crate::Result<ArrowSchemaOrFieldOrType> {
        let fields = results
            .into_iter()
            .map(|result| match result {
                ArrowSchemaOrFieldOrType::Field(field) => field,
                _ => unreachable!(),
            })
            .collect();
        Ok(ArrowSchemaOrFieldOrType::Type(DataType::Struct(fields)))
    }

    fn list(
        &mut self,
        list: &crate::spec::ListType,
        value: ArrowSchemaOrFieldOrType,
    ) -> crate::Result<Self::T> {
        let field = match self.field(&list.element_field, value)? {
            ArrowSchemaOrFieldOrType::Field(field) => field,
            _ => unreachable!(),
        };
        let meta = if let Some(doc) = &list.element_field.doc {
            HashMap::from([
                (
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    list.element_field.id.to_string(),
                ),
                (ARROW_FIELD_DOC_KEY.to_string(), doc.clone()),
            ])
        } else {
            HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                list.element_field.id.to_string(),
            )])
        };
        // `self.field` above already stamped the canonical variant extension metadata when the
        // element is a variant; this rebuild REPLACES the whole metadata map, so re-stamp it or
        // the element silently degrades to a plain `{metadata, value}` struct — a variant column
        // treated as an ordinary struct, which is the exact hazard the type's identity rule
        // exists to prevent.
        let meta = if matches!(
            list.element_field.field_type.as_ref(),
            crate::spec::Type::Variant
        ) {
            with_variant_extension_metadata(meta)
        } else {
            meta
        };
        let field = field.with_metadata(meta);
        Ok(ArrowSchemaOrFieldOrType::Type(DataType::List(Arc::new(
            field,
        ))))
    }

    fn map(
        &mut self,
        map: &crate::spec::MapType,
        key_value: ArrowSchemaOrFieldOrType,
        value: ArrowSchemaOrFieldOrType,
    ) -> crate::Result<ArrowSchemaOrFieldOrType> {
        let key_field = match self.field(&map.key_field, key_value)? {
            ArrowSchemaOrFieldOrType::Field(field) => field,
            _ => unreachable!(),
        };
        let value_field = match self.field(&map.value_field, value)? {
            ArrowSchemaOrFieldOrType::Field(field) => field,
            _ => unreachable!(),
        };
        let field = Field::new(
            DEFAULT_MAP_FIELD_NAME,
            DataType::Struct(vec![key_field, value_field].into()),
            // Map field is always not nullable
            false,
        );

        Ok(ArrowSchemaOrFieldOrType::Type(DataType::Map(
            field.into(),
            false,
        )))
    }

    /// Iceberg `variant` becomes the canonical Arrow variant extension type: a two-field struct
    /// `{metadata: Binary, value: Binary}` carrying the `arrow.parquet.variant` extension name.
    ///
    /// A DELIBERATE divergence from Java, which throws here because its Arrow bridge predates the
    /// canonical type. Emitting it is the same type `parquet`'s own variant support reads and
    /// writes, so the on-disk contract is preserved rather than bypassed.
    ///
    /// The shape is fixed by the Arrow spec: the extension metadata is the empty string (not
    /// absent), and both children are non-nullable — a variant with no value is a null at the
    /// STRUCT level, never a null child.
    ///
    /// # Notes
    ///
    /// Field ids are stamped by the caller (`Self::field`), as for every other type. The
    /// `metadata` / `value` children carry NO field ids: they are components of one Iceberg
    /// field, not Iceberg fields themselves.
    fn variant(&mut self) -> crate::Result<ArrowSchemaOrFieldOrType> {
        Ok(ArrowSchemaOrFieldOrType::Type(variant_arrow_data_type()))
    }

    fn primitive(
        &mut self,
        p: &crate::spec::PrimitiveType,
    ) -> crate::Result<ArrowSchemaOrFieldOrType> {
        match p {
            crate::spec::PrimitiveType::Boolean => {
                Ok(ArrowSchemaOrFieldOrType::Type(DataType::Boolean))
            }
            crate::spec::PrimitiveType::Int => Ok(ArrowSchemaOrFieldOrType::Type(DataType::Int32)),
            crate::spec::PrimitiveType::Long => Ok(ArrowSchemaOrFieldOrType::Type(DataType::Int64)),
            crate::spec::PrimitiveType::Float => {
                Ok(ArrowSchemaOrFieldOrType::Type(DataType::Float32))
            }
            crate::spec::PrimitiveType::Double => {
                Ok(ArrowSchemaOrFieldOrType::Type(DataType::Float64))
            }
            crate::spec::PrimitiveType::Decimal { precision, scale } => {
                Ok(ArrowSchemaOrFieldOrType::Type(decimal128_arrow_type(
                    *precision,
                    *scale,
                    "Iceberg-to-Arrow decimal type convert",
                )?))
            }
            crate::spec::PrimitiveType::Date => {
                Ok(ArrowSchemaOrFieldOrType::Type(DataType::Date32))
            }
            crate::spec::PrimitiveType::Time => Ok(ArrowSchemaOrFieldOrType::Type(
                DataType::Time64(TimeUnit::Microsecond),
            )),
            crate::spec::PrimitiveType::Timestamp => Ok(ArrowSchemaOrFieldOrType::Type(
                DataType::Timestamp(TimeUnit::Microsecond, None),
            )),
            crate::spec::PrimitiveType::Timestamptz => Ok(ArrowSchemaOrFieldOrType::Type(
                // Timestampz always stored as UTC
                DataType::Timestamp(TimeUnit::Microsecond, Some(UTC_TIME_ZONE.into())),
            )),
            crate::spec::PrimitiveType::TimestampNs => Ok(ArrowSchemaOrFieldOrType::Type(
                DataType::Timestamp(TimeUnit::Nanosecond, None),
            )),
            crate::spec::PrimitiveType::TimestamptzNs => Ok(ArrowSchemaOrFieldOrType::Type(
                // Store timestamptz_ns as UTC
                DataType::Timestamp(TimeUnit::Nanosecond, Some(UTC_TIME_ZONE.into())),
            )),
            crate::spec::PrimitiveType::String => {
                Ok(ArrowSchemaOrFieldOrType::Type(DataType::Utf8))
            }
            crate::spec::PrimitiveType::Uuid => Ok(ArrowSchemaOrFieldOrType::Type(
                DataType::FixedSizeBinary(16),
            )),
            crate::spec::PrimitiveType::Fixed(len) => Ok(ArrowSchemaOrFieldOrType::Type(
                i32::try_from(*len)
                    .ok()
                    .map(DataType::FixedSizeBinary)
                    .unwrap_or(DataType::LargeBinary),
            )),
            crate::spec::PrimitiveType::Binary => {
                Ok(ArrowSchemaOrFieldOrType::Type(DataType::LargeBinary))
            }
            // `unknown` is an always-null column with no physical storage; Arrow's `Null` type is
            // its natural in-memory shape (Java `TypeToMessageType` returns null — no parquet
            // column). This lets a metadata schema carrying `unknown` participate in Arrow schema
            // conversion; the file-level always-null write/read I/O is deferred (the parquet
            // writer and Arrow value path fail loudly on `unknown`).
            crate::spec::PrimitiveType::Unknown => {
                Ok(ArrowSchemaOrFieldOrType::Type(DataType::Null))
            }
        }
    }
}

/// Convert iceberg schema to an arrow schema.
pub fn schema_to_arrow_schema(schema: &crate::spec::Schema) -> crate::Result<ArrowSchema> {
    let mut converter = ToArrowSchemaConverter;
    match crate::spec::visit_schema(schema, &mut converter)? {
        ArrowSchemaOrFieldOrType::Schema(schema) => Ok(schema),
        _ => unreachable!(),
    }
}

/// Convert iceberg type to an arrow type.
pub fn type_to_arrow_type(ty: &crate::spec::Type) -> crate::Result<DataType> {
    let mut converter = ToArrowSchemaConverter;
    match crate::spec::visit_type(ty, &mut converter)? {
        ArrowSchemaOrFieldOrType::Type(ty) => Ok(ty),
        _ => unreachable!(),
    }
}

/// Convert Iceberg Datum to Arrow Datum.
pub(crate) fn get_arrow_datum(datum: &Datum) -> Result<Arc<dyn ArrowDatum + Send + Sync>> {
    match (datum.data_type(), datum.literal()) {
        (PrimitiveType::Boolean, PrimitiveLiteral::Boolean(value)) => {
            Ok(Arc::new(BooleanArray::new_scalar(*value)))
        }
        (PrimitiveType::Int, PrimitiveLiteral::Int(value)) => {
            Ok(Arc::new(Int32Array::new_scalar(*value)))
        }
        (PrimitiveType::Long, PrimitiveLiteral::Long(value)) => {
            Ok(Arc::new(Int64Array::new_scalar(*value)))
        }
        (PrimitiveType::Float, PrimitiveLiteral::Float(value)) => {
            Ok(Arc::new(Float32Array::new_scalar(value.into_inner())))
        }
        (PrimitiveType::Double, PrimitiveLiteral::Double(value)) => {
            Ok(Arc::new(Float64Array::new_scalar(value.into_inner())))
        }
        (PrimitiveType::String, PrimitiveLiteral::String(value)) => {
            Ok(Arc::new(StringArray::new_scalar(value.as_str())))
        }
        (PrimitiveType::Binary, PrimitiveLiteral::Binary(value)) => {
            Ok(Arc::new(BinaryArray::new_scalar(value.as_slice())))
        }
        (PrimitiveType::Date, PrimitiveLiteral::Int(value)) => {
            Ok(Arc::new(Date32Array::new_scalar(*value)))
        }
        (PrimitiveType::Timestamp, PrimitiveLiteral::Long(value)) => {
            Ok(Arc::new(TimestampMicrosecondArray::new_scalar(*value)))
        }
        (PrimitiveType::Timestamptz, PrimitiveLiteral::Long(value)) => Ok(Arc::new(Scalar::new(
            TimestampMicrosecondArray::new(vec![*value; 1].into(), None)
                .with_timezone(UTC_TIME_ZONE),
        ))),
        (PrimitiveType::TimestampNs, PrimitiveLiteral::Long(value)) => {
            Ok(Arc::new(TimestampNanosecondArray::new_scalar(*value)))
        }
        (PrimitiveType::TimestamptzNs, PrimitiveLiteral::Long(value)) => Ok(Arc::new(Scalar::new(
            TimestampNanosecondArray::new(vec![*value; 1].into(), None)
                .with_timezone(UTC_TIME_ZONE),
        ))),
        (PrimitiveType::Decimal { precision, scale }, PrimitiveLiteral::Int128(value)) => {
            // `precision`/`scale` can arrive here through bypass paths such as `Datum::new` or
            // `Datum::try_from_bytes`, so a corrupt/hostile catalog or manifest can carry a
            // precision/scale far outside Arrow's Decimal128 range. Reject in three stages,
            // each as a typed error (AGENTS.md: no bare unwrap AND no truncating `as` in production
            // paths):
            //   1. `u8::try_from` / `i8::try_from` — Arrow takes a `u8` precision + `i8` scale, so a
            //      plain `as` cast would WRAP (e.g. precision 294 → 38, scale 256 → 0) and SILENTLY
            //      ACCEPT an invalid value; `try_from` rejects anything outside the numeric range.
            //   2. `validate_decimal_precision_and_scale` — enforces Arrow's own rules
            //      (precision ≤ 38, and scale ≤ precision) on the now in-range values.
            //   3. `validate_decimal_literal` — rejects a scalar whose unscaled value needs more
            //      digits than the declared precision, which Arrow does not check for us here.
            datum.validate_decimal()?;
            let (arrow_precision, arrow_scale) =
                decimal128_precision_and_scale(*precision, *scale, "Decimal literal type convert")?;
            let array = Decimal128Array::from_value(*value, 1)
                .with_precision_and_scale(arrow_precision, arrow_scale)
                .map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Decimal literal precision/scale ({precision},{scale}) is not a valid \
                             Arrow Decimal128"
                        ),
                    )
                    .with_source(e)
                })?;
            Ok(Arc::new(Scalar::new(array)))
        }
        (PrimitiveType::Uuid, PrimitiveLiteral::UInt128(value)) => {
            let bytes = Uuid::from_u128(*value).into_bytes();
            let array = FixedSizeBinaryArray::try_from_iter(vec![bytes].into_iter()).unwrap();
            Ok(Arc::new(Scalar::new(array)))
        }
        (PrimitiveType::Time, PrimitiveLiteral::Long(value)) => {
            Ok(Arc::new(Time64MicrosecondArray::new_scalar(*value)))
        }
        (PrimitiveType::Fixed(_), PrimitiveLiteral::Binary(value)) => {
            // A 1-element `FixedSizeBinaryArray` whose width is `value.len()` — the data column for a
            // `Fixed(n)` field is `FixedSizeBinary(n)`, and Arrow's `eq` kernel compares the scalar's
            // byte buffer against each row's fixed-width bytes. `try_from_iter` derives the width from
            // the single element, so a width mismatch with the column is surfaced by the kernel, not
            // here. Unlike the `Uuid` arm (whose `[u8; 16]` width is statically known and cannot fail),
            // this width is data-derived, so the `Result` is mapped to a typed error rather than
            // `.unwrap()`ed (AGENTS.md: no bare unwrap in production paths).
            let array =
                FixedSizeBinaryArray::try_from_iter([value.clone()].into_iter()).map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        "Failed to build a FixedSizeBinary scalar from a Fixed literal",
                    )
                    .with_source(e)
                })?;
            Ok(Arc::new(Scalar::new(array)))
        }

        (primitive_type, _) => Err(Error::new(
            ErrorKind::FeatureUnsupported,
            format!("Converting datum from type {primitive_type:?} to arrow not supported yet."),
        )),
    }
}

pub(crate) fn get_parquet_stat_min_as_datum(
    primitive_type: &PrimitiveType,
    stats: &Statistics,
) -> Result<Option<Datum>> {
    Ok(match (primitive_type, stats) {
        (PrimitiveType::Boolean, Statistics::Boolean(stats)) => {
            stats.min_opt().map(|val| Datum::bool(*val))
        }
        (PrimitiveType::Int, Statistics::Int32(stats)) => {
            stats.min_opt().map(|val| Datum::int(*val))
        }
        (PrimitiveType::Date, Statistics::Int32(stats)) => {
            stats.min_opt().map(|val| Datum::date(*val))
        }
        (PrimitiveType::Long, Statistics::Int64(stats)) => {
            stats.min_opt().map(|val| Datum::long(*val))
        }
        (PrimitiveType::Time, Statistics::Int64(stats)) => {
            let Some(val) = stats.min_opt() else {
                return Ok(None);
            };

            Some(Datum::time_micros(*val)?)
        }
        (PrimitiveType::Timestamp, Statistics::Int64(stats)) => {
            stats.min_opt().map(|val| Datum::timestamp_micros(*val))
        }
        (PrimitiveType::Timestamptz, Statistics::Int64(stats)) => {
            stats.min_opt().map(|val| Datum::timestamptz_micros(*val))
        }
        (PrimitiveType::TimestampNs, Statistics::Int64(stats)) => {
            stats.min_opt().map(|val| Datum::timestamp_nanos(*val))
        }
        (PrimitiveType::TimestamptzNs, Statistics::Int64(stats)) => {
            stats.min_opt().map(|val| Datum::timestamptz_nanos(*val))
        }
        (PrimitiveType::Float, Statistics::Float(stats)) => {
            stats.min_opt().map(|val| Datum::float(*val))
        }
        (PrimitiveType::Double, Statistics::Double(stats)) => {
            stats.min_opt().map(|val| Datum::double(*val))
        }
        (PrimitiveType::String, Statistics::ByteArray(stats)) => {
            let Some(val) = stats.min_opt() else {
                return Ok(None);
            };

            Some(Datum::string(val.as_utf8()?))
        }
        (
            PrimitiveType::Decimal {
                precision: _,
                scale: _,
            },
            Statistics::ByteArray(stats),
        ) => {
            let Some(bytes) = stats.min_bytes_opt() else {
                return Ok(None);
            };
            Some(Datum::new(
                primitive_type.clone(),
                PrimitiveLiteral::Int128(i128::from_be_bytes(bytes.try_into()?)),
            ))
        }
        (
            PrimitiveType::Decimal {
                precision: _,
                scale: _,
            },
            Statistics::FixedLenByteArray(stats),
        ) => {
            let Some(bytes) = stats.min_bytes_opt() else {
                return Ok(None);
            };
            Some(Datum::new(
                primitive_type.clone(),
                PrimitiveLiteral::Int128(i128_from_be_bytes(bytes).ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Can't convert bytes to i128: {bytes:?}"),
                    )
                })?),
            ))
        }
        (
            PrimitiveType::Decimal {
                precision: _,
                scale: _,
            },
            Statistics::Int32(stats),
        ) => stats.min_opt().map(|val| {
            Datum::new(
                primitive_type.clone(),
                PrimitiveLiteral::Int128(i128::from(*val)),
            )
        }),

        (
            PrimitiveType::Decimal {
                precision: _,
                scale: _,
            },
            Statistics::Int64(stats),
        ) => stats.min_opt().map(|val| {
            Datum::new(
                primitive_type.clone(),
                PrimitiveLiteral::Int128(i128::from(*val)),
            )
        }),
        (PrimitiveType::Uuid, Statistics::FixedLenByteArray(stats)) => {
            let Some(bytes) = stats.min_bytes_opt() else {
                return Ok(None);
            };
            if bytes.len() != 16 {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    "Invalid length of uuid bytes.",
                ));
            }
            Some(Datum::uuid(Uuid::from_bytes(
                bytes[..16].try_into().unwrap(),
            )))
        }
        (PrimitiveType::Fixed(len), Statistics::FixedLenByteArray(stat)) => {
            let Some(bytes) = stat.min_bytes_opt() else {
                return Ok(None);
            };
            if bytes.len() != *len as usize {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    "Invalid length of fixed bytes.",
                ));
            }
            Some(Datum::fixed(bytes.to_vec()))
        }
        (PrimitiveType::Binary, Statistics::ByteArray(stat)) => {
            return Ok(stat
                .min_bytes_opt()
                .map(|bytes| Datum::binary(bytes.to_vec())));
        }
        _ => {
            return Ok(None);
        }
    })
}

pub(crate) fn get_parquet_stat_max_as_datum(
    primitive_type: &PrimitiveType,
    stats: &Statistics,
) -> Result<Option<Datum>> {
    Ok(match (primitive_type, stats) {
        (PrimitiveType::Boolean, Statistics::Boolean(stats)) => {
            stats.max_opt().map(|val| Datum::bool(*val))
        }
        (PrimitiveType::Int, Statistics::Int32(stats)) => {
            stats.max_opt().map(|val| Datum::int(*val))
        }
        (PrimitiveType::Date, Statistics::Int32(stats)) => {
            stats.max_opt().map(|val| Datum::date(*val))
        }
        (PrimitiveType::Long, Statistics::Int64(stats)) => {
            stats.max_opt().map(|val| Datum::long(*val))
        }
        (PrimitiveType::Time, Statistics::Int64(stats)) => {
            let Some(val) = stats.max_opt() else {
                return Ok(None);
            };

            Some(Datum::time_micros(*val)?)
        }
        (PrimitiveType::Timestamp, Statistics::Int64(stats)) => {
            stats.max_opt().map(|val| Datum::timestamp_micros(*val))
        }
        (PrimitiveType::Timestamptz, Statistics::Int64(stats)) => {
            stats.max_opt().map(|val| Datum::timestamptz_micros(*val))
        }
        (PrimitiveType::TimestampNs, Statistics::Int64(stats)) => {
            stats.max_opt().map(|val| Datum::timestamp_nanos(*val))
        }
        (PrimitiveType::TimestamptzNs, Statistics::Int64(stats)) => {
            stats.max_opt().map(|val| Datum::timestamptz_nanos(*val))
        }
        (PrimitiveType::Float, Statistics::Float(stats)) => {
            stats.max_opt().map(|val| Datum::float(*val))
        }
        (PrimitiveType::Double, Statistics::Double(stats)) => {
            stats.max_opt().map(|val| Datum::double(*val))
        }
        (PrimitiveType::String, Statistics::ByteArray(stats)) => {
            let Some(val) = stats.max_opt() else {
                return Ok(None);
            };

            Some(Datum::string(val.as_utf8()?))
        }
        (
            PrimitiveType::Decimal {
                precision: _,
                scale: _,
            },
            Statistics::ByteArray(stats),
        ) => {
            let Some(bytes) = stats.max_bytes_opt() else {
                return Ok(None);
            };
            Some(Datum::new(
                primitive_type.clone(),
                PrimitiveLiteral::Int128(i128::from_be_bytes(bytes.try_into()?)),
            ))
        }
        (
            PrimitiveType::Decimal {
                precision: _,
                scale: _,
            },
            Statistics::FixedLenByteArray(stats),
        ) => {
            let Some(bytes) = stats.max_bytes_opt() else {
                return Ok(None);
            };
            Some(Datum::new(
                primitive_type.clone(),
                PrimitiveLiteral::Int128(i128_from_be_bytes(bytes).ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Can't convert bytes to i128: {bytes:?}"),
                    )
                })?),
            ))
        }
        (
            PrimitiveType::Decimal {
                precision: _,
                scale: _,
            },
            Statistics::Int32(stats),
        ) => stats.max_opt().map(|val| {
            Datum::new(
                primitive_type.clone(),
                PrimitiveLiteral::Int128(i128::from(*val)),
            )
        }),

        (
            PrimitiveType::Decimal {
                precision: _,
                scale: _,
            },
            Statistics::Int64(stats),
        ) => stats.max_opt().map(|val| {
            Datum::new(
                primitive_type.clone(),
                PrimitiveLiteral::Int128(i128::from(*val)),
            )
        }),
        (PrimitiveType::Uuid, Statistics::FixedLenByteArray(stats)) => {
            let Some(bytes) = stats.max_bytes_opt() else {
                return Ok(None);
            };
            if bytes.len() != 16 {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    "Invalid length of uuid bytes.",
                ));
            }
            Some(Datum::uuid(Uuid::from_bytes(
                bytes[..16].try_into().unwrap(),
            )))
        }
        (PrimitiveType::Fixed(len), Statistics::FixedLenByteArray(stat)) => {
            let Some(bytes) = stat.max_bytes_opt() else {
                return Ok(None);
            };
            if bytes.len() != *len as usize {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    "Invalid length of fixed bytes.",
                ));
            }
            Some(Datum::fixed(bytes.to_vec()))
        }
        (PrimitiveType::Binary, Statistics::ByteArray(stat)) => {
            return Ok(stat
                .max_bytes_opt()
                .map(|bytes| Datum::binary(bytes.to_vec())));
        }
        _ => {
            return Ok(None);
        }
    })
}

impl TryFrom<&ArrowSchema> for crate::spec::Schema {
    type Error = Error;

    fn try_from(schema: &ArrowSchema) -> crate::Result<Self> {
        arrow_schema_to_schema(schema)
    }
}

impl TryFrom<&crate::spec::Schema> for ArrowSchema {
    type Error = Error;

    fn try_from(schema: &crate::spec::Schema) -> crate::Result<Self> {
        schema_to_arrow_schema(schema)
    }
}

/// Converts a Datum (Iceberg type + primitive literal) to its corresponding Arrow DataType
/// with Run-End Encoding (REE).
///
/// This function is used for constant fields in record batches, where all values are the same.
/// Run-End Encoding provides efficient storage for such constant columns.
///
/// # Arguments
/// * `datum` - The Datum to convert, which contains both type and value information
///
/// # Returns
/// Arrow DataType with Run-End Encoding applied
///
/// # Example
/// ```
/// use iceberg::arrow::datum_to_arrow_type_with_ree;
/// use iceberg::spec::Datum;
///
/// let datum = Datum::string("test_file.parquet");
/// let ree_type = datum_to_arrow_type_with_ree(&datum).unwrap();
/// // Returns: RunEndEncoded(Int32, Utf8)
/// ```
pub fn datum_to_arrow_type_with_ree(datum: &Datum) -> Result<DataType> {
    datum.validate_decimal()?;

    // Helper to create REE type with the given values type.
    // Note: values field is nullable as Arrow expects this when building the
    // final Arrow schema with `RunArray::try_new`.
    let make_ree = |values_type: DataType| -> DataType {
        let run_ends_field = Arc::new(Field::new("run_ends", DataType::Int32, false));
        let values_field = Arc::new(Field::new("values", values_type, true));
        DataType::RunEndEncoded(run_ends_field, values_field)
    };

    // Match on the PrimitiveType from the Datum to determine the Arrow type
    match datum.data_type() {
        PrimitiveType::Boolean => Ok(make_ree(DataType::Boolean)),
        PrimitiveType::Int => Ok(make_ree(DataType::Int32)),
        PrimitiveType::Long => Ok(make_ree(DataType::Int64)),
        PrimitiveType::Float => Ok(make_ree(DataType::Float32)),
        PrimitiveType::Double => Ok(make_ree(DataType::Float64)),
        PrimitiveType::Date => Ok(make_ree(DataType::Date32)),
        PrimitiveType::Time => Ok(make_ree(DataType::Int64)),
        PrimitiveType::Timestamp => Ok(make_ree(DataType::Int64)),
        PrimitiveType::Timestamptz => Ok(make_ree(DataType::Int64)),
        PrimitiveType::TimestampNs => Ok(make_ree(DataType::Int64)),
        PrimitiveType::TimestamptzNs => Ok(make_ree(DataType::Int64)),
        PrimitiveType::String => Ok(make_ree(DataType::Utf8)),
        PrimitiveType::Uuid => Ok(make_ree(DataType::Binary)),
        PrimitiveType::Fixed(_) => Ok(make_ree(DataType::Binary)),
        PrimitiveType::Binary => Ok(make_ree(DataType::Binary)),
        PrimitiveType::Decimal { precision, scale } => Ok(make_ree(decimal128_arrow_type(
            *precision,
            *scale,
            "Run-end-encoded decimal datum type convert",
        )?)),
        // `unknown` carries no `PrimitiveLiteral`, so a `Datum` of this type is unconstructable —
        // this arm is unreachable in practice. Keep it consistent with `type_to_arrow_type`
        // (`unknown` -> Arrow `Null`) rather than panicking.
        PrimitiveType::Unknown => Ok(make_ree(DataType::Null)),
    }
}

/// A visitor that strips metadata from an Arrow schema.
///
/// This visitor recursively removes all metadata from fields at every level of the schema,
/// including nested struct, list, and map fields. This is useful for schema comparison
/// where metadata differences should be ignored.
struct MetadataStripVisitor {
    /// Stack to track field information during traversal
    field_stack: Vec<Field>,
}

impl MetadataStripVisitor {
    fn new() -> Self {
        Self {
            field_stack: Vec::new(),
        }
    }

    fn push_field_info(&mut self, field: &Field) {
        self.field_stack.push(Field::new(
            field.name(),
            DataType::Null, // Placeholder, will be replaced
            field.is_nullable(),
        ));
    }
}

impl ArrowSchemaVisitor for MetadataStripVisitor {
    type T = Field;
    type U = ArrowSchema;

    fn before_field(&mut self, field: &Field) -> Result<()> {
        // Store field name and nullability for later reconstruction
        self.push_field_info(field);
        Ok(())
    }

    fn after_field(&mut self, _field: &Field) -> Result<()> {
        Ok(())
    }

    fn before_list_element(&mut self, field: &Field) -> Result<()> {
        self.push_field_info(field);
        Ok(())
    }

    fn before_map_key(&mut self, field: &Field) -> Result<()> {
        self.push_field_info(field);
        Ok(())
    }

    fn before_map_value(&mut self, field: &Field) -> Result<()> {
        self.push_field_info(field);
        Ok(())
    }

    fn schema(&mut self, _schema: &ArrowSchema, values: Vec<Self::T>) -> Result<Self::U> {
        Ok(ArrowSchema::new(values))
    }

    fn r#struct(&mut self, _fields: &Fields, results: Vec<Self::T>) -> Result<Self::T> {
        // Pop the struct field from the stack
        let field_info = self
            .field_stack
            .pop()
            .ok_or_else(|| Error::new(ErrorKind::Unexpected, "Field stack underflow in struct"))?;

        // Reconstruct struct field without metadata
        Ok(Field::new(
            field_info.name(),
            DataType::Struct(Fields::from(results)),
            field_info.is_nullable(),
        ))
    }

    fn list(&mut self, list: &DataType, value: Self::T) -> Result<Self::T> {
        // Pop the list field from the stack
        let field_info = self
            .field_stack
            .pop()
            .ok_or_else(|| Error::new(ErrorKind::Unexpected, "Field stack underflow in list"))?;

        // Reconstruct list field without metadata
        let list_type = match list {
            DataType::List(_) => DataType::List(Arc::new(value)),
            DataType::LargeList(_) => DataType::LargeList(Arc::new(value)),
            DataType::FixedSizeList(_, size) => DataType::FixedSizeList(Arc::new(value), *size),
            _ => {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    format!("Expected list type, got {list}"),
                ));
            }
        };

        Ok(Field::new(
            field_info.name(),
            list_type,
            field_info.is_nullable(),
        ))
    }

    fn map(&mut self, map: &DataType, key_value: Self::T, value: Self::T) -> Result<Self::T> {
        // Pop the map field from the stack
        let field_info = self
            .field_stack
            .pop()
            .ok_or_else(|| Error::new(ErrorKind::Unexpected, "Field stack underflow in map"))?;

        // Reconstruct the map's struct field (contains key and value)
        let struct_field = Field::new(
            DEFAULT_MAP_FIELD_NAME,
            DataType::Struct(Fields::from(vec![key_value, value])),
            false,
        );

        // Get the sorted flag from the original map type
        let sorted = match map {
            DataType::Map(_, sorted) => *sorted,
            _ => {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    format!("Expected map type, got {map}"),
                ));
            }
        };

        // Reconstruct map field without metadata
        Ok(Field::new(
            field_info.name(),
            DataType::Map(Arc::new(struct_field), sorted),
            field_info.is_nullable(),
        ))
    }

    fn primitive(&mut self, p: &DataType) -> Result<Self::T> {
        // Pop the primitive field from the stack
        let field_info = self.field_stack.pop().ok_or_else(|| {
            Error::new(ErrorKind::Unexpected, "Field stack underflow in primitive")
        })?;

        // Return field without metadata
        Ok(Field::new(
            field_info.name(),
            p.clone(),
            field_info.is_nullable(),
        ))
    }
}

/// Strips all metadata from an Arrow schema and its nested fields.
///
/// This function recursively removes metadata from all fields at every level of the schema,
/// including nested struct, list, and map fields. This is useful for schema comparison
/// where metadata differences should be ignored.
///
/// # Arguments
/// * `schema` - The Arrow schema to strip metadata from
///
/// # Returns
/// A new Arrow schema with all metadata removed, or an error if the schema structure
/// is invalid.
///
/// # Example
/// ```
/// use std::collections::HashMap;
///
/// use arrow_schema::{DataType, Field, Schema as ArrowSchema};
/// use iceberg::arrow::strip_metadata_from_schema;
///
/// let mut metadata = HashMap::new();
/// metadata.insert("key".to_string(), "value".to_string());
///
/// let field = Field::new("col1", DataType::Int32, false).with_metadata(metadata);
/// let schema = ArrowSchema::new(vec![field]);
///
/// let stripped = strip_metadata_from_schema(&schema).unwrap();
/// assert!(stripped.field(0).metadata().is_empty());
/// ```
pub fn strip_metadata_from_schema(schema: &ArrowSchema) -> Result<ArrowSchema> {
    let mut visitor = MetadataStripVisitor::new();
    visit_schema(schema, &mut visitor)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Schema as ArrowSchema, TimeUnit, UnionFields, UnionMode};

    use super::*;
    use crate::spec::decimal_utils::decimal_new;
    use crate::spec::{Literal, Schema};

    /// Create a simple field with metadata.
    fn simple_field(name: &str, ty: DataType, nullable: bool, value: &str) -> Field {
        Field::new(name, ty, nullable).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            value.to_string(),
        )]))
    }

    fn field_with_next_id(
        name: &str,
        ty: DataType,
        nullable: bool,
        next_field_id: &mut i32,
    ) -> Field {
        let field = simple_field(name, ty, nullable, &next_field_id.to_string());
        *next_field_id += 1;
        field
    }

    /// Wrap a primitive in a valid struct/list/map-value cycle. Each wrapper adds exactly one
    /// visitor edge, while unique field IDs keep the converted Iceberg schema valid.
    fn nested_composite_type(nesting: usize) -> DataType {
        let mut data_type = DataType::Int32;
        let mut next_field_id = 1;

        for level in 0..nesting {
            data_type = match level % 3 {
                0 => DataType::List(Arc::new(field_with_next_id(
                    "element",
                    data_type,
                    true,
                    &mut next_field_id,
                ))),
                1 => DataType::Struct(Fields::from(vec![field_with_next_id(
                    "nested",
                    data_type,
                    true,
                    &mut next_field_id,
                )])),
                2 => {
                    let key = field_with_next_id("key", DataType::Utf8, false, &mut next_field_id);
                    let value = field_with_next_id("value", data_type, true, &mut next_field_id);
                    DataType::Map(
                        Arc::new(Field::new(
                            DEFAULT_MAP_FIELD_NAME,
                            DataType::Struct(Fields::from(vec![key, value])),
                            false,
                        )),
                        false,
                    )
                }
                _ => unreachable!(),
            };
        }

        data_type
    }

    /// Build a malicious Arrow map chain through the key slot. Iceberg map keys cannot be nested,
    /// but a caller can manually construct this Arrow type, so that recursion edge still needs the
    /// same pre-conversion bound as the valid map-value path.
    fn nested_map_key_type(nesting: usize) -> DataType {
        let mut data_type = DataType::Int32;
        let mut next_field_id = 1;

        for _ in 0..nesting {
            let key = field_with_next_id("key", data_type, false, &mut next_field_id);
            let value = field_with_next_id("value", DataType::Int32, true, &mut next_field_id);
            data_type = DataType::Map(
                Arc::new(Field::new(
                    DEFAULT_MAP_FIELD_NAME,
                    DataType::Struct(Fields::from(vec![key, value])),
                    false,
                )),
                false,
            );
        }

        data_type
    }

    fn nested_list_type(nesting: usize, large: bool, fixed_size: bool) -> DataType {
        let mut data_type = DataType::Int32;
        let mut next_field_id = 1;

        for _ in 0..nesting {
            let element = Arc::new(field_with_next_id(
                "element",
                data_type,
                true,
                &mut next_field_id,
            ));
            data_type = if large {
                DataType::LargeList(element)
            } else if fixed_size {
                DataType::FixedSizeList(element, 1)
            } else {
                DataType::List(element)
            };
        }

        data_type
    }

    fn nested_dictionary_type(nesting: usize) -> DataType {
        let mut data_type = DataType::Int32;
        for _ in 0..nesting {
            data_type = DataType::Dictionary(Box::new(DataType::Int32), Box::new(data_type));
        }
        data_type
    }

    fn drop_dictionary_type_iteratively(data_type: DataType) {
        let mut current = Some(data_type);
        while let Some(data_type) = current.take() {
            match data_type {
                DataType::Dictionary(key_type, value_type) => {
                    drop(key_type);
                    current = Some(*value_type);
                }
                other => drop(other),
            }
        }
    }

    #[derive(Clone, Copy)]
    enum UnsupportedRecursiveArrowType {
        ListView,
        LargeListView,
        Union,
        RunEndEncoded,
    }

    impl UnsupportedRecursiveArrowType {
        fn diagnostic_name(self) -> &'static str {
            match self {
                Self::ListView => "ListView",
                Self::LargeListView => "LargeListView",
                Self::Union => "Union",
                Self::RunEndEncoded => "RunEndEncoded",
            }
        }
    }

    struct RetainedDeepArrowType {
        data_type: DataType,
        /// One extra reference to each nested field, ordered innermost to outermost.
        /// Popping reverses that order so dropping one field never recursively drops its child.
        retained_fields: Vec<Arc<Field>>,
    }

    impl RetainedDeepArrowType {
        fn into_schema(self) -> RetainedDeepArrowSchema {
            let Self {
                data_type,
                mut retained_fields,
            } = self;
            let root_field = Arc::new(simple_field("root", data_type, true, "1"));
            retained_fields.push(Arc::clone(&root_field));

            RetainedDeepArrowSchema {
                schema: ArrowSchema::new(vec![root_field]),
                retained_fields,
            }
        }

        fn drop_iteratively(self) {
            let Self {
                data_type,
                mut retained_fields,
            } = self;
            drop(data_type);
            while let Some(field) = retained_fields.pop() {
                drop(field);
            }
        }
    }

    struct RetainedDeepArrowSchema {
        schema: ArrowSchema,
        retained_fields: Vec<Arc<Field>>,
    }

    impl RetainedDeepArrowSchema {
        fn drop_iteratively(self) {
            let Self {
                schema,
                mut retained_fields,
            } = self;
            drop(schema);
            while let Some(field) = retained_fields.pop() {
                drop(field);
            }
        }
    }

    fn nested_list_view_type(nesting: usize, is_large: bool) -> RetainedDeepArrowType {
        let mut data_type = DataType::Int32;
        let mut retained_fields = Vec::with_capacity(nesting);

        for _ in 0..nesting {
            let element = Arc::new(Field::new("item", data_type, true));
            retained_fields.push(Arc::clone(&element));
            data_type = if is_large {
                DataType::LargeListView(element)
            } else {
                DataType::ListView(element)
            };
        }

        RetainedDeepArrowType {
            data_type,
            retained_fields,
        }
    }

    fn hostile_unsupported_type(
        unsupported_type: UnsupportedRecursiveArrowType,
        nesting: usize,
    ) -> RetainedDeepArrowType {
        match unsupported_type {
            UnsupportedRecursiveArrowType::ListView => nested_list_view_type(nesting, false),
            UnsupportedRecursiveArrowType::LargeListView => nested_list_view_type(nesting, true),
            UnsupportedRecursiveArrowType::Union => {
                let mut hostile = nested_list_view_type(nesting, false);
                let union_field = Arc::new(Field::new("member", hostile.data_type, true));
                let union_fields = UnionFields::try_new([0], [Arc::clone(&union_field)])
                    .expect("one-field hostile union fixture must be valid");
                hostile.retained_fields.push(union_field);
                hostile.data_type = DataType::Union(union_fields, UnionMode::Dense);
                hostile
            }
            UnsupportedRecursiveArrowType::RunEndEncoded => {
                let mut hostile = nested_list_view_type(nesting, true);
                let run_ends = Arc::new(Field::new("run_ends", DataType::Int32, false));
                let values = Arc::new(Field::new("values", hostile.data_type, true));
                hostile.retained_fields.push(Arc::clone(&values));
                hostile.data_type = DataType::RunEndEncoded(run_ends, values);
                hostile
            }
        }
    }

    fn assert_unsupported_recursive_error<T: std::fmt::Debug>(
        result: Result<T>,
        unsupported_type: UnsupportedRecursiveArrowType,
    ) {
        let error = result.expect_err("unsupported recursive Arrow type must return a typed error");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(error.to_string().contains(&format!(
            "Cannot visit Arrow data type: {}",
            unsupported_type.diagnostic_name()
        )));
    }

    #[test]
    fn deep_list_view_types_fail_safely_at_every_public_arrow_schema_entry() {
        let mut checked_entries = 0;

        for unsupported_type in [
            UnsupportedRecursiveArrowType::ListView,
            UnsupportedRecursiveArrowType::LargeListView,
        ] {
            let hostile = hostile_unsupported_type(unsupported_type, 10_000);
            assert_unsupported_recursive_error(
                arrow_type_to_type(&hostile.data_type),
                unsupported_type,
            );
            hostile.drop_iteratively();
            checked_entries += 1;

            let hostile = hostile_unsupported_type(unsupported_type, 10_000).into_schema();
            assert_unsupported_recursive_error(
                arrow_schema_to_schema(&hostile.schema),
                unsupported_type,
            );
            assert_unsupported_recursive_error(
                arrow_schema_to_schema_auto_assign_ids(&hostile.schema),
                unsupported_type,
            );
            assert_unsupported_recursive_error(
                strip_metadata_from_schema(&hostile.schema),
                unsupported_type,
            );
            hostile.drop_iteratively();
            checked_entries += 3;
        }

        assert_eq!(checked_entries, 8);
    }

    #[test]
    fn recursive_union_and_run_end_encoded_diagnostics_do_not_format_deep_children() {
        let mut checked_entries = 0;

        for unsupported_type in [
            UnsupportedRecursiveArrowType::Union,
            UnsupportedRecursiveArrowType::RunEndEncoded,
        ] {
            let hostile = hostile_unsupported_type(unsupported_type, 10_000);
            assert_unsupported_recursive_error(
                arrow_type_to_type(&hostile.data_type),
                unsupported_type,
            );
            hostile.drop_iteratively();
            checked_entries += 1;

            let hostile = hostile_unsupported_type(unsupported_type, 10_000).into_schema();
            assert_unsupported_recursive_error(
                arrow_schema_to_schema(&hostile.schema),
                unsupported_type,
            );
            assert_unsupported_recursive_error(
                arrow_schema_to_schema_auto_assign_ids(&hostile.schema),
                unsupported_type,
            );
            assert_unsupported_recursive_error(
                strip_metadata_from_schema(&hostile.schema),
                unsupported_type,
            );
            hostile.drop_iteratively();
            checked_entries += 3;
        }

        assert_eq!(checked_entries, 8);
    }

    #[test]
    fn shallow_unsupported_nested_types_keep_useful_variant_diagnostics() {
        let mut checked_variants = 0;

        for unsupported_type in [
            UnsupportedRecursiveArrowType::ListView,
            UnsupportedRecursiveArrowType::LargeListView,
            UnsupportedRecursiveArrowType::Union,
            UnsupportedRecursiveArrowType::RunEndEncoded,
        ] {
            let hostile = hostile_unsupported_type(unsupported_type, 1);
            assert_unsupported_recursive_error(
                arrow_type_to_type(&hostile.data_type),
                unsupported_type,
            );
            hostile.drop_iteratively();
            checked_variants += 1;
        }

        assert_eq!(checked_variants, 4);
    }

    #[test]
    fn arrow_schema_visitors_accept_the_exact_nesting_boundary() {
        // A standalone type starts at depth 0, so 128 composite edges put the primitive exactly at
        // the accepted depth 128.
        let boundary_type = nested_composite_type(MAX_ARROW_SCHEMA_NESTING_DEPTH);
        let converted_type = arrow_type_to_type(&boundary_type)
            .expect("standalone Arrow type at the exact nesting boundary must convert");
        assert!(matches!(converted_type, Type::Struct(_)));

        // A schema is an implicit root struct at depth 0. Its field starts at depth 1, so one fewer
        // composite edge reaches the same exact boundary. Exercise every schema visitor consumer.
        let boundary_schema = ArrowSchema::new(vec![simple_field(
            "root",
            nested_composite_type(MAX_ARROW_SCHEMA_NESTING_DEPTH - 1),
            true,
            "10000",
        )]);
        let converted = arrow_schema_to_schema(&boundary_schema)
            .expect("explicit-ID schema at the exact nesting boundary must convert");
        let auto_assigned = arrow_schema_to_schema_auto_assign_ids(&boundary_schema)
            .expect("auto-ID schema at the exact nesting boundary must convert");
        let stripped = strip_metadata_from_schema(&boundary_schema)
            .expect("metadata stripping at the exact nesting boundary must succeed");

        assert_eq!(converted.as_struct().fields().len(), 1);
        assert_eq!(auto_assigned.as_struct().fields().len(), 1);
        assert!(stripped.field(0).metadata().is_empty());
    }

    #[test]
    fn arrow_schema_visitors_reject_one_level_beyond_every_public_entry() {
        let expected_message = format!(
            "Arrow schema type nesting exceeds maximum depth {MAX_ARROW_SCHEMA_NESTING_DEPTH}"
        );

        let overdeep_type = nested_composite_type(MAX_ARROW_SCHEMA_NESTING_DEPTH + 1);
        let type_error = arrow_type_to_type(&overdeep_type)
            .expect_err("standalone Arrow type one level beyond the limit must fail");
        assert_eq!(type_error.kind(), ErrorKind::DataInvalid);
        assert!(type_error.to_string().contains(&expected_message));

        // A schema field starts one level below its implicit root, so 128 composite edges are one
        // beyond the schema boundary. The Arrow-specific diagnostic proves these public converters
        // fail at this visitor, rather than only at the downstream Iceberg schema builder.
        let overdeep_schema = ArrowSchema::new(vec![simple_field(
            "root",
            nested_composite_type(MAX_ARROW_SCHEMA_NESTING_DEPTH),
            true,
            "10000",
        )]);
        let explicit_error = arrow_schema_to_schema(&overdeep_schema)
            .expect_err("explicit-ID schema one level beyond the limit must fail");
        assert_eq!(explicit_error.kind(), ErrorKind::DataInvalid);
        assert!(explicit_error.to_string().contains(&expected_message));

        let auto_error = arrow_schema_to_schema_auto_assign_ids(&overdeep_schema)
            .expect_err("auto-ID schema one level beyond the limit must fail");
        assert_eq!(auto_error.kind(), ErrorKind::DataInvalid);
        assert!(auto_error.to_string().contains(&expected_message));

        let strip_error = strip_metadata_from_schema(&overdeep_schema)
            .expect_err("metadata stripping one level beyond the limit must fail");
        assert_eq!(strip_error.kind(), ErrorKind::DataInvalid);
        assert!(strip_error.to_string().contains(&expected_message));
    }

    #[test]
    fn arrow_schema_visitor_bounds_every_recursive_arrow_edge() {
        let expected_message = format!(
            "Arrow schema type nesting exceeds maximum depth {MAX_ARROW_SCHEMA_NESTING_DEPTH}"
        );
        let overdeep = MAX_ARROW_SCHEMA_NESTING_DEPTH + 1;

        // The mixed boundary tests cover struct fields, ordinary lists, and map values. Exercise
        // manually constructible map-key, LargeList, FixedSizeList, and dictionary chains too, so
        // no recursive Arrow edge can bypass the shared depth check.
        for (name, data_type) in [
            ("map key", nested_map_key_type(overdeep)),
            (
                "large-list element",
                nested_list_type(overdeep, true, false),
            ),
            (
                "fixed-size-list element",
                nested_list_type(overdeep, false, true),
            ),
            ("dictionary value", nested_dictionary_type(overdeep)),
        ] {
            let error = arrow_type_to_type(&data_type)
                .expect_err("every overdeep recursive Arrow edge must be rejected");
            assert_eq!(error.kind(), ErrorKind::DataInvalid, "{name}");
            assert!(error.to_string().contains(&expected_message), "{name}");
        }
    }

    #[test]
    fn malicious_arrow_dictionary_depth_errors_and_drops_iteratively() {
        // Arrow permits callers to manually build far deeper trees than its normal producers emit.
        // The visitor must stop after 128 edges, independent of the input's total depth. Tear down
        // the synthetic 10,000-node Box chain iteratively after the call: recursively dropping the
        // hostile fixture could itself overflow the test thread's stack and would test Arrow's Drop
        // behavior rather than this borrowed visitor.
        let hostile = nested_dictionary_type(10_000);
        let result = arrow_type_to_type(&hostile);
        drop_dictionary_type_iteratively(hostile);

        let error = result.expect_err("hostile dictionary nesting must return a typed error");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(error.to_string().contains(&format!(
            "Arrow schema type nesting exceeds maximum depth {MAX_ARROW_SCHEMA_NESTING_DEPTH}"
        )));
    }

    fn arrow_schema_for_arrow_schema_to_schema_test() -> ArrowSchema {
        let fields = Fields::from(vec![
            simple_field("key", DataType::Int32, false, "28"),
            simple_field("value", DataType::Utf8, true, "29"),
        ]);

        let r#struct = DataType::Struct(fields);
        let map = DataType::Map(
            Arc::new(simple_field(DEFAULT_MAP_FIELD_NAME, r#struct, false, "17")),
            false,
        );
        let dictionary = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));

        let fields = Fields::from(vec![
            simple_field("aa", DataType::Int32, false, "18"),
            simple_field("bb", DataType::Utf8, true, "19"),
            simple_field(
                "cc",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
                "20",
            ),
        ]);

        let r#struct = DataType::Struct(fields);

        ArrowSchema::new(vec![
            simple_field("a", DataType::Int32, false, "2"),
            simple_field("b", DataType::Int64, false, "1"),
            simple_field("c", DataType::Utf8, false, "3"),
            simple_field("n", DataType::Utf8, false, "21"),
            simple_field(
                "d",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                true,
                "4",
            ),
            simple_field("e", DataType::Boolean, true, "6"),
            simple_field("f", DataType::Float32, false, "5"),
            simple_field("g", DataType::Float64, false, "7"),
            simple_field("p", DataType::Decimal128(10, 2), false, "27"),
            simple_field("h", DataType::Date32, false, "8"),
            simple_field("i", DataType::Time64(TimeUnit::Microsecond), false, "9"),
            simple_field(
                "j",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
                "10",
            ),
            simple_field(
                "k",
                DataType::Timestamp(TimeUnit::Microsecond, Some("+00:00".into())),
                false,
                "12",
            ),
            simple_field("l", DataType::Binary, false, "13"),
            simple_field("o", DataType::LargeBinary, false, "22"),
            simple_field("m", DataType::FixedSizeBinary(10), false, "11"),
            simple_field(
                "list",
                DataType::List(Arc::new(simple_field(
                    "element",
                    DataType::Int32,
                    false,
                    "15",
                ))),
                true,
                "14",
            ),
            simple_field(
                "large_list",
                DataType::LargeList(Arc::new(simple_field(
                    "element",
                    DataType::Utf8,
                    false,
                    "23",
                ))),
                true,
                "24",
            ),
            simple_field(
                "fixed_list",
                DataType::FixedSizeList(
                    Arc::new(simple_field("element", DataType::Binary, false, "26")),
                    10,
                ),
                true,
                "25",
            ),
            simple_field("map", map, false, "16"),
            simple_field("struct", r#struct, false, "17"),
            simple_field("dictionary", dictionary, false, "30"),
        ])
    }

    fn iceberg_schema_for_arrow_schema_to_schema_test() -> Schema {
        let schema_json = r#"{
            "type":"struct",
            "schema-id":0,
            "fields":[
                {
                    "id":2,
                    "name":"a",
                    "required":true,
                    "type":"int"
                },
                {
                    "id":1,
                    "name":"b",
                    "required":true,
                    "type":"long"
                },
                {
                    "id":3,
                    "name":"c",
                    "required":true,
                    "type":"string"
                },
                {
                    "id":21,
                    "name":"n",
                    "required":true,
                    "type":"string"
                },
                {
                    "id":4,
                    "name":"d",
                    "required":false,
                    "type":"timestamp"
                },
                {
                    "id":6,
                    "name":"e",
                    "required":false,
                    "type":"boolean"
                },
                {
                    "id":5,
                    "name":"f",
                    "required":true,
                    "type":"float"
                },
                {
                    "id":7,
                    "name":"g",
                    "required":true,
                    "type":"double"
                },
                {
                    "id":27,
                    "name":"p",
                    "required":true,
                    "type":"decimal(10,2)"
                },
                {
                    "id":8,
                    "name":"h",
                    "required":true,
                    "type":"date"
                },
                {
                    "id":9,
                    "name":"i",
                    "required":true,
                    "type":"time"
                },
                {
                    "id":10,
                    "name":"j",
                    "required":true,
                    "type":"timestamptz"
                },
                {
                    "id":12,
                    "name":"k",
                    "required":true,
                    "type":"timestamptz"
                },
                {
                    "id":13,
                    "name":"l",
                    "required":true,
                    "type":"binary"
                },
                {
                    "id":22,
                    "name":"o",
                    "required":true,
                    "type":"binary"
                },
                {
                    "id":11,
                    "name":"m",
                    "required":true,
                    "type":"fixed[10]"
                },
                {
                    "id":14,
                    "name":"list",
                    "required": false,
                    "type": {
                        "type": "list",
                        "element-id": 15,
                        "element-required": true,
                        "element": "int"
                    }
                },
                {
                    "id":24,
                    "name":"large_list",
                    "required": false,
                    "type": {
                        "type": "list",
                        "element-id": 23,
                        "element-required": true,
                        "element": "string"
                    }
                },
                {
                    "id":25,
                    "name":"fixed_list",
                    "required": false,
                    "type": {
                        "type": "list",
                        "element-id": 26,
                        "element-required": true,
                        "element": "binary"
                    }
                },
                {
                    "id":16,
                    "name":"map",
                    "required": true,
                    "type": {
                        "type": "map",
                        "key-id": 28,
                        "key": "int",
                        "value-id": 29,
                        "value-required": false,
                        "value": "string"
                    }
                },
                {
                    "id":17,
                    "name":"struct",
                    "required": true,
                    "type": {
                        "type": "struct",
                        "fields": [
                            {
                                "id":18,
                                "name":"aa",
                                "required":true,
                                "type":"int"
                            },
                            {
                                "id":19,
                                "name":"bb",
                                "required":false,
                                "type":"string"
                            },
                            {
                                "id":20,
                                "name":"cc",
                                "required":true,
                                "type":"timestamp"
                            }
                        ]
                    }
                },
                {
                    "id":30,
                    "name":"dictionary",
                    "required":true,
                    "type":"string"
                }
            ],
            "identifier-field-ids":[]
        }"#;

        let schema: Schema = serde_json::from_str(schema_json).unwrap();
        schema
    }

    #[test]
    fn test_arrow_schema_to_schema() {
        let arrow_schema = arrow_schema_for_arrow_schema_to_schema_test();
        let schema = iceberg_schema_for_arrow_schema_to_schema_test();
        let converted_schema = arrow_schema_to_schema(&arrow_schema).unwrap();
        pretty_assertions::assert_eq!(converted_schema, schema);
    }

    fn arrow_schema_for_schema_to_arrow_schema_test() -> ArrowSchema {
        let fields = Fields::from(vec![
            simple_field("key", DataType::Int32, false, "28"),
            simple_field("value", DataType::Utf8, true, "29"),
        ]);

        let r#struct = DataType::Struct(fields);
        let map = DataType::Map(
            Arc::new(Field::new(DEFAULT_MAP_FIELD_NAME, r#struct, false)),
            false,
        );

        let fields = Fields::from(vec![
            simple_field("aa", DataType::Int32, false, "18"),
            simple_field("bb", DataType::Utf8, true, "19"),
            simple_field(
                "cc",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
                "20",
            ),
        ]);

        let r#struct = DataType::Struct(fields);

        ArrowSchema::new(vec![
            simple_field("a", DataType::Int32, false, "2"),
            simple_field("b", DataType::Int64, false, "1"),
            simple_field("c", DataType::Utf8, false, "3"),
            simple_field("n", DataType::Utf8, false, "21"),
            simple_field(
                "d",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                true,
                "4",
            ),
            simple_field("e", DataType::Boolean, true, "6"),
            simple_field("f", DataType::Float32, false, "5"),
            simple_field("g", DataType::Float64, false, "7"),
            simple_field("p", DataType::Decimal128(10, 2), false, "27"),
            simple_field("h", DataType::Date32, false, "8"),
            simple_field("i", DataType::Time64(TimeUnit::Microsecond), false, "9"),
            simple_field(
                "j",
                DataType::Timestamp(TimeUnit::Microsecond, Some(UTC_TIME_ZONE.into())),
                false,
                "10",
            ),
            simple_field(
                "k",
                DataType::Timestamp(TimeUnit::Microsecond, Some(UTC_TIME_ZONE.into())),
                false,
                "12",
            ),
            simple_field("l", DataType::LargeBinary, false, "13"),
            simple_field("o", DataType::LargeBinary, false, "22"),
            simple_field("m", DataType::FixedSizeBinary(10), false, "11"),
            simple_field(
                "list",
                DataType::List(Arc::new(simple_field(
                    "element",
                    DataType::Int32,
                    false,
                    "15",
                ))),
                true,
                "14",
            ),
            simple_field(
                "large_list",
                DataType::List(Arc::new(simple_field(
                    "element",
                    DataType::Utf8,
                    false,
                    "23",
                ))),
                true,
                "24",
            ),
            simple_field(
                "fixed_list",
                DataType::List(Arc::new(simple_field(
                    "element",
                    DataType::LargeBinary,
                    false,
                    "26",
                ))),
                true,
                "25",
            ),
            simple_field("map", map, false, "16"),
            simple_field("struct", r#struct, false, "17"),
            simple_field("uuid", DataType::FixedSizeBinary(16), false, "30"),
        ])
    }

    fn iceberg_schema_for_schema_to_arrow_schema() -> Schema {
        let schema_json = r#"{
            "type":"struct",
            "schema-id":0,
            "fields":[
                {
                    "id":2,
                    "name":"a",
                    "required":true,
                    "type":"int"
                },
                {
                    "id":1,
                    "name":"b",
                    "required":true,
                    "type":"long"
                },
                {
                    "id":3,
                    "name":"c",
                    "required":true,
                    "type":"string"
                },
                {
                    "id":21,
                    "name":"n",
                    "required":true,
                    "type":"string"
                },
                {
                    "id":4,
                    "name":"d",
                    "required":false,
                    "type":"timestamp"
                },
                {
                    "id":6,
                    "name":"e",
                    "required":false,
                    "type":"boolean"
                },
                {
                    "id":5,
                    "name":"f",
                    "required":true,
                    "type":"float"
                },
                {
                    "id":7,
                    "name":"g",
                    "required":true,
                    "type":"double"
                },
                {
                    "id":27,
                    "name":"p",
                    "required":true,
                    "type":"decimal(10,2)"
                },
                {
                    "id":8,
                    "name":"h",
                    "required":true,
                    "type":"date"
                },
                {
                    "id":9,
                    "name":"i",
                    "required":true,
                    "type":"time"
                },
                {
                    "id":10,
                    "name":"j",
                    "required":true,
                    "type":"timestamptz"
                },
                {
                    "id":12,
                    "name":"k",
                    "required":true,
                    "type":"timestamptz"
                },
                {
                    "id":13,
                    "name":"l",
                    "required":true,
                    "type":"binary"
                },
                {
                    "id":22,
                    "name":"o",
                    "required":true,
                    "type":"binary"
                },
                {
                    "id":11,
                    "name":"m",
                    "required":true,
                    "type":"fixed[10]"
                },
                {
                    "id":14,
                    "name":"list",
                    "required": false,
                    "type": {
                        "type": "list",
                        "element-id": 15,
                        "element-required": true,
                        "element": "int"
                    }
                },
                {
                    "id":24,
                    "name":"large_list",
                    "required": false,
                    "type": {
                        "type": "list",
                        "element-id": 23,
                        "element-required": true,
                        "element": "string"
                    }
                },
                {
                    "id":25,
                    "name":"fixed_list",
                    "required": false,
                    "type": {
                        "type": "list",
                        "element-id": 26,
                        "element-required": true,
                        "element": "binary"
                    }
                },
                {
                    "id":16,
                    "name":"map",
                    "required": true,
                    "type": {
                        "type": "map",
                        "key-id": 28,
                        "key": "int",
                        "value-id": 29,
                        "value-required": false,
                        "value": "string"
                    }
                },
                {
                    "id":17,
                    "name":"struct",
                    "required": true,
                    "type": {
                        "type": "struct",
                        "fields": [
                            {
                                "id":18,
                                "name":"aa",
                                "required":true,
                                "type":"int"
                            },
                            {
                                "id":19,
                                "name":"bb",
                                "required":false,
                                "type":"string"
                            },
                            {
                                "id":20,
                                "name":"cc",
                                "required":true,
                                "type":"timestamp"
                            }
                        ]
                    }
                },
                {
                    "id":30,
                    "name":"uuid",
                    "required":true,
                    "type":"uuid"
                }
            ],
            "identifier-field-ids":[]
        }"#;

        let schema: Schema = serde_json::from_str(schema_json).unwrap();
        schema
    }

    #[test]
    fn test_schema_to_arrow_schema() {
        let arrow_schema = arrow_schema_for_schema_to_arrow_schema_test();
        let schema = iceberg_schema_for_schema_to_arrow_schema();
        let converted_arrow_schema = schema_to_arrow_schema(&schema).unwrap();
        assert_eq!(converted_arrow_schema, arrow_schema);
    }

    /// RISK: Iceberg `timestamptz` / `timestamptz_ns` must emit Spark's `tz=UTC`
    /// annotation, not the historical `+00:00`. Values are unchanged — this pins
    /// the schema metadata only.
    #[test]
    fn schema_to_arrow_schema_annotates_timestamptz_as_utc() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "ts", Type::Primitive(PrimitiveType::Timestamptz)).into(),
                NestedField::required(2, "ts_ns", Type::Primitive(PrimitiveType::TimestamptzNs))
                    .into(),
            ])
            .build()
            .expect("static timestamptz schema");
        let arrow = schema_to_arrow_schema(&schema).expect("Iceberg→Arrow");
        assert_eq!(
            arrow.field(0).data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, Some(UTC_TIME_ZONE.into()))
        );
        assert_eq!(
            arrow.field(1).data_type(),
            &DataType::Timestamp(TimeUnit::Nanosecond, Some(UTC_TIME_ZONE.into()))
        );
        assert_eq!(UTC_TIME_ZONE, "UTC");
    }

    /// RISK: narrowing the inverse to only `"UTC"` would refuse parquet/Arrow
    /// written under the old `+00:00` annotation. Both aliases, both units,
    /// both public converters must still resolve to Iceberg `timestamptz`.
    #[test]
    fn arrow_schema_to_schema_accepts_utc_and_offset_aliases() {
        for (zone, unit, expected) in [
            (
                UTC_TIME_ZONE,
                TimeUnit::Microsecond,
                PrimitiveType::Timestamptz,
            ),
            (
                UTC_OFFSET_TIME_ZONE,
                TimeUnit::Microsecond,
                PrimitiveType::Timestamptz,
            ),
            (
                UTC_TIME_ZONE,
                TimeUnit::Nanosecond,
                PrimitiveType::TimestamptzNs,
            ),
            (
                UTC_OFFSET_TIME_ZONE,
                TimeUnit::Nanosecond,
                PrimitiveType::TimestamptzNs,
            ),
        ] {
            let arrow = ArrowSchema::new(vec![
                Field::new("ts", DataType::Timestamp(unit, Some(zone.into())), true).with_metadata(
                    HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())]),
                ),
            ]);
            let with_ids = arrow_schema_to_schema(&arrow)
                .unwrap_or_else(|e| panic!("{zone:?}/{unit:?} with ids must resolve: {e}"));
            let auto = arrow_schema_to_schema_auto_assign_ids(&arrow)
                .unwrap_or_else(|e| panic!("{zone:?}/{unit:?} auto-assign must resolve: {e}"));
            match with_ids.as_struct().fields()[0].field_type.as_ref() {
                Type::Primitive(got) => assert_eq!(got, &expected, "with-ids {zone:?}/{unit:?}"),
                other => panic!("with-ids {zone:?}/{unit:?} produced {other:?}"),
            }
            match auto.as_struct().fields()[0].field_type.as_ref() {
                Type::Primitive(got) => assert_eq!(got, &expected, "auto {zone:?}/{unit:?}"),
                other => panic!("auto {zone:?}/{unit:?} produced {other:?}"),
            }
        }
        assert!(is_utc_time_zone(UTC_OFFSET_TIME_ZONE));
        assert!(is_utc_time_zone("UTC"));
        assert!(!is_utc_time_zone("+05:00"));
    }

    /// RISK: a genuinely different timezone must stay a loud type error, not
    /// silently become `timestamptz` (the UTC-alias set is closed).
    #[test]
    fn arrow_schema_to_schema_rejects_non_utc_timezone() {
        let arrow = ArrowSchema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some("+05:00".into())),
                true,
            )
            .with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]);
        let error = arrow_schema_to_schema(&arrow)
            .expect_err("a non-UTC timezone must not map to Iceberg timestamptz");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.to_string().contains("Unsupported Arrow data type"),
            "expected a type-rejection, got: {error}"
        );
    }

    /// RISK: files written under the old `+00:00` annotation reach the scan
    /// transformer as `Timestamp(_, "+00:00")` against a `UTC` target. The
    /// Promote path (`arrow_cast::cast`) must succeed and keep the i64 instants
    /// bit-identical — otherwise every previously-written timestamptz file
    /// becomes unreadable.
    #[test]
    fn arrow_cast_from_offset_alias_to_utc_is_bit_identical() {
        use arrow_array::TimestampMicrosecondArray;
        use arrow_cast::cast;

        let values = vec![Some(42_i64), None, Some(-1)];
        let src =
            TimestampMicrosecondArray::from(values.clone()).with_timezone(UTC_OFFSET_TIME_ZONE);
        let target = DataType::Timestamp(TimeUnit::Microsecond, Some(UTC_TIME_ZONE.into()));
        let out = cast(&src, &target).expect("UTC-alias cast must succeed");
        let out = out
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .expect("cast must stay TimestampMicrosecondArray");
        assert_eq!(out.timezone(), Some(UTC_TIME_ZONE));
        assert_eq!(out.iter().collect::<Vec<_>>(), values);
    }

    // Variant converts to the canonical Arrow extension type rather than erroring, replacing
    // `test_variant_to_arrow_errors_loudly`. What must NOT happen is a silent fallback: the type
    // is identified by the FIELD's extension metadata, never by its struct shape, so a plain
    // `{metadata, value}` struct with no extension name still converts back to a STRUCT. That is
    // the discriminating cell below.

    #[test]
    fn test_variant_converts_to_the_canonical_arrow_extension_type() {
        let arrow_type =
            type_to_arrow_type(&Type::Variant).expect("variant now has an Arrow representation");
        assert_eq!(
            arrow_type,
            DataType::Struct(Fields::from(vec![
                Field::new("metadata", DataType::Binary, false),
                Field::new("value", DataType::Binary, false),
            ])),
            "both children are NON-nullable — an absent variant is a null at the struct level"
        );
    }

    #[test]
    fn test_variant_field_carries_the_extension_name_and_its_field_id() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "v", Type::Variant).into(),
            ])
            .build()
            .unwrap();
        let arrow_schema = schema_to_arrow_schema(&schema)
            .expect("a schema containing variant now converts to Arrow");

        let variant_field = arrow_schema.field_with_name("v").expect("variant field");
        assert_eq!(
            variant_field.metadata().get("ARROW:extension:name"),
            Some(&"arrow.parquet.variant".to_string())
        );
        assert_eq!(
            variant_field.metadata().get("ARROW:extension:metadata"),
            Some(&String::new()),
            "the extension metadata is the EMPTY STRING, not an absent key"
        );
        assert_eq!(
            variant_field.metadata().get(PARQUET_FIELD_ID_META_KEY),
            Some(&"2".to_string()),
            "the Iceberg field id survives alongside the extension metadata"
        );
        assert!(is_variant_arrow_field(variant_field));
        assert!(
            !is_variant_arrow_field(arrow_schema.field_with_name("id").expect("id field")),
            "a non-variant field must not be mistaken for one"
        );
    }

    #[test]
    fn test_variant_round_trips_back_to_the_iceberg_variant_type() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "v", Type::Variant).into(),
            ])
            .build()
            .unwrap();
        let arrow_schema = schema_to_arrow_schema(&schema).expect("to arrow");
        let back = arrow_schema_to_schema(&arrow_schema).expect("back to iceberg");
        assert_eq!(
            back.as_struct().fields()[1].field_type.as_ref(),
            &Type::Variant,
            "the extension name must be recognised on the way back, not flattened to a struct"
        );
        assert_eq!(back.as_struct().fields()[1].id, 2);
    }

    /// NESTED variant — inside a `list` and inside a `map` value. `list` rebuilds its element's
    /// metadata map, silently erasing the extension name and degrading `list<variant>` to a list
    /// of plain structs; neither descent consulted the `variant_field` hook on the way back.
    #[test]
    fn test_variant_nested_in_a_list_and_a_map_round_trips() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::optional(
                    1,
                    "vs",
                    Type::List(crate::spec::ListType {
                        element_field: NestedField::list_element(2, Type::Variant, true).into(),
                    }),
                )
                .into(),
                NestedField::optional(
                    3,
                    "vm",
                    Type::Map(crate::spec::MapType {
                        key_field: NestedField::map_key_element(
                            4,
                            Type::Primitive(PrimitiveType::String),
                        )
                        .into(),
                        value_field: NestedField::map_value_element(5, Type::Variant, true).into(),
                    }),
                )
                .into(),
                // A variant map KEY. Java constrains only the VALUE type
                // (`Types$MapType.ofOptional` null-checks it and nothing else), so this is a legal
                // Iceberg type and must round-trip like every other position.
                NestedField::optional(
                    6,
                    "vk",
                    Type::Map(crate::spec::MapType {
                        key_field: NestedField::map_key_element(7, Type::Variant).into(),
                        value_field: NestedField::map_value_element(
                            8,
                            Type::Primitive(PrimitiveType::String),
                            true,
                        )
                        .into(),
                    }),
                )
                .into(),
            ])
            .build()
            .unwrap();

        let arrow_schema = schema_to_arrow_schema(&schema).expect("to arrow");

        // The list element keeps the extension name despite `list()` rebuilding its metadata.
        let DataType::List(element) = arrow_schema.field_with_name("vs").unwrap().data_type()
        else {
            panic!("expected a list");
        };
        assert!(
            is_variant_arrow_field(element),
            "the list ELEMENT must still be identifiable as a variant, got metadata {:?}",
            element.metadata()
        );

        let back = arrow_schema_to_schema(&arrow_schema).expect("back to iceberg");
        let Type::List(list) = back.as_struct().fields()[0].field_type.as_ref() else {
            panic!("expected a list");
        };
        assert_eq!(
            list.element_field.field_type.as_ref(),
            &Type::Variant,
            "a list element must not flatten to a struct on the way back"
        );
        let Type::Map(map) = back.as_struct().fields()[1].field_type.as_ref() else {
            panic!("expected a map");
        };
        assert_eq!(
            map.value_field.field_type.as_ref(),
            &Type::Variant,
            "a map value must not flatten to a struct on the way back"
        );
        let Type::Map(key_map) = back.as_struct().fields()[2].field_type.as_ref() else {
            panic!("expected a map");
        };
        assert_eq!(
            key_map.key_field.field_type.as_ref(),
            &Type::Variant,
            "a map KEY must round-trip too — the position skipped on a false premise"
        );
    }

    /// The fifth descent position: a variant inside a STRUCT, on the Arrow→Iceberg side. Removing
    /// `visit_struct`'s `variant_field` hook survived the suite. It fails loudly rather than
    /// corrupting, but a loud failure on a legal type is still a defect.
    #[test]
    fn test_variant_nested_in_a_struct_round_trips() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::optional(
                    1,
                    "s",
                    Type::Struct(crate::spec::StructType::new(vec![
                        NestedField::optional(2, "v", Type::Variant).into(),
                        NestedField::required(3, "n", Type::Primitive(PrimitiveType::Long)).into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap();

        let arrow_schema = schema_to_arrow_schema(&schema).expect("to arrow");
        let DataType::Struct(children) = arrow_schema.field_with_name("s").unwrap().data_type()
        else {
            panic!("expected a struct");
        };
        assert!(
            is_variant_arrow_field(&children[0]),
            "the struct's variant CHILD must carry the extension name, got {:?}",
            children[0].metadata()
        );

        let back = arrow_schema_to_schema(&arrow_schema).expect("back to iceberg");
        let Type::Struct(struct_type) = back.as_struct().fields()[0].field_type.as_ref() else {
            panic!("expected a struct");
        };
        assert_eq!(
            struct_type.fields()[0].field_type.as_ref(),
            &Type::Variant,
            "a variant inside a struct must not flatten on the way back"
        );
        assert_eq!(
            struct_type.fields()[1].field_type.as_ref(),
            &Type::Primitive(PrimitiveType::Long),
            "its non-variant sibling is unaffected"
        );
    }

    /// The LENIENCY clause of `is_variant_arrow_field`, which its own doc states: identity is the
    /// extension NAME alone, because a writer that omitted the metadata key still produced a
    /// variant. Every other fixture stamps both keys, so requiring the metadata key survived the
    /// suite — an unpinned documented clause (symmetry sweep F-9).
    #[test]
    fn test_a_variant_field_without_the_extension_metadata_key_is_still_a_variant() {
        let name_only =
            Field::new("v", variant_arrow_data_type(), true).with_metadata(HashMap::from([
                (
                    "ARROW:extension:name".to_string(),
                    VARIANT_EXTENSION_NAME.to_string(),
                ),
                (PARQUET_FIELD_ID_META_KEY.to_string(), "2".to_string()),
            ]));
        assert!(
            name_only
                .metadata()
                .get("ARROW:extension:metadata")
                .is_none(),
            "fixture precondition: the metadata key is ABSENT"
        );
        assert!(
            is_variant_arrow_field(&name_only),
            "the extension NAME alone identifies a variant — requiring the metadata key would \
             reject files written without it"
        );

        let converted =
            arrow_schema_to_schema(&ArrowSchema::new(vec![name_only])).expect("converts");
        assert_eq!(
            converted.as_struct().fields()[0].field_type.as_ref(),
            &Type::Variant
        );
    }

    /// The DISCRIMINATING cell. A struct of the same SHAPE but WITHOUT the extension name is a
    /// plain struct. If identity were keyed on shape, any `{metadata: Binary, value: Binary}`
    /// column in a user's table would silently become a variant — the exact
    /// treat-raw-bytes-as-something-else hazard the old refusal existed to prevent.
    #[test]
    fn test_a_shape_alike_struct_without_the_extension_name_is_not_a_variant() {
        let arrow_schema = ArrowSchema::new(vec![
            Field::new(
                "not_a_variant",
                DataType::Struct(Fields::from(vec![
                    Field::new("metadata", DataType::Binary, false).with_metadata(HashMap::from([
                        (PARQUET_FIELD_ID_META_KEY.to_string(), "3".to_string()),
                    ])),
                    Field::new("value", DataType::Binary, false).with_metadata(HashMap::from([(
                        PARQUET_FIELD_ID_META_KEY.to_string(),
                        "4".to_string(),
                    )])),
                ])),
                true,
            )
            .with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]);

        let converted = arrow_schema_to_schema(&arrow_schema).expect("plain struct converts");
        assert!(
            matches!(
                converted.as_struct().fields()[0].field_type.as_ref(),
                Type::Struct(_)
            ),
            "shape alone must not make a variant, got {:?}",
            converted.as_struct().fields()[0].field_type
        );
    }

    // RISK: `unknown` is an always-null column with no physical storage (Java `TypeToMessageType`
    // returns null — no parquet column). Its natural Arrow shape is `DataType::Null`, which lets a
    // metadata schema carrying `unknown` participate in Arrow schema conversion (the metadata-only
    // round-trip contract). A wrong mapping (e.g. a real physical type) would invent storage Java
    // never emits.
    #[test]
    fn test_unknown_to_arrow_is_null_type() {
        let arrow_type = type_to_arrow_type(&Type::Primitive(PrimitiveType::Unknown))
            .expect("unknown converts to the Arrow Null type");
        assert_eq!(arrow_type, DataType::Null);

        // Whole-schema conversion succeeds with an unknown column present.
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "u", Type::Primitive(PrimitiveType::Unknown)).into(),
            ])
            .build()
            .unwrap();
        let arrow =
            schema_to_arrow_schema(&schema).expect("a schema containing unknown converts to Arrow");
        assert_eq!(arrow.field(1).data_type(), &DataType::Null);
    }

    #[test]
    fn arrow_to_iceberg_decimal_rejects_negative_scale_without_wrapping() {
        let arrow_schema = ArrowSchema::new(vec![Field::new(
            "bad_decimal",
            DataType::Decimal128(10, -1),
            false,
        )]);

        let error = arrow_schema_to_schema_auto_assign_ids(&arrow_schema)
            .expect_err("negative Arrow decimal scale must not wrap into a huge Iceberg scale");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains("non-negative"),
            "negative scale should be rejected at the Arrow boundary, got: {error}"
        );
    }

    #[test]
    fn arrow_to_iceberg_decimal_rejects_scale_greater_than_precision() {
        let arrow_schema = ArrowSchema::new(vec![Field::new(
            "bad_decimal",
            DataType::Decimal128(10, 11),
            false,
        )]);

        let error = arrow_schema_to_schema_auto_assign_ids(&arrow_schema).expect_err(
            "Arrow decimal scale greater than precision must not become Iceberg decimal",
        );
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
    }

    #[test]
    fn iceberg_to_arrow_decimal_rejects_unrepresentable_precision_scale() {
        for (iceberg_type, context) in [
            (
                Type::Primitive(PrimitiveType::Decimal {
                    precision: 39,
                    scale: 0,
                }),
                "precision above Arrow Decimal128 max",
            ),
            (
                Type::Primitive(PrimitiveType::Decimal {
                    precision: 10,
                    scale: 11,
                }),
                "scale greater than precision",
            ),
            (
                Type::Primitive(PrimitiveType::Decimal {
                    precision: 10,
                    scale: 256,
                }),
                "scale that would wrap to zero under `as i8`",
            ),
        ] {
            let error = match type_to_arrow_type(&iceberg_type) {
                Ok(_) => panic!("{context}: {iceberg_type:?} must be rejected"),
                Err(error) => error,
            };
            assert_eq!(error.kind(), ErrorKind::DataInvalid, "{context}");
        }
    }

    #[test]
    fn run_end_encoded_decimal_preserves_valid_precision_scale() {
        let datum = Datum::decimal_with_precision(decimal_new(123, 38), 38)
            .expect("decimal(38,38) datum should be constructible");

        let arrow_type = datum_to_arrow_type_with_ree(&datum)
            .expect("valid decimal datum should produce a REE Arrow type");
        let DataType::RunEndEncoded(_, values_field) = arrow_type else {
            panic!("decimal datum must be wrapped in RunEndEncoded");
        };
        assert_eq!(values_field.data_type(), &DataType::Decimal128(38, 38));
    }

    #[test]
    fn run_end_encoded_decimal_rejects_wrapping_precision_scale() {
        for (datum, context) in [
            (
                Datum::new(
                    PrimitiveType::Decimal {
                        precision: 294,
                        scale: 0,
                    },
                    PrimitiveLiteral::Int128(1234),
                ),
                "precision 294 wraps to valid precision 38 under `as u8`",
            ),
            (
                Datum::new(
                    PrimitiveType::Decimal {
                        precision: 10,
                        scale: 256,
                    },
                    PrimitiveLiteral::Int128(1234),
                ),
                "scale 256 wraps to valid scale 0 under `as i8`",
            ),
        ] {
            let error = match datum_to_arrow_type_with_ree(&datum) {
                Ok(_) => panic!("{context}: {datum:?} must be rejected"),
                Err(error) => error,
            };
            assert_eq!(error.kind(), ErrorKind::DataInvalid, "{context}");
        }
    }

    #[test]
    fn test_type_conversion() {
        // test primitive type
        {
            let arrow_type = DataType::Int32;
            let iceberg_type = Type::Primitive(PrimitiveType::Int);
            assert_eq!(arrow_type, type_to_arrow_type(&iceberg_type).unwrap());
            assert_eq!(iceberg_type, arrow_type_to_type(&arrow_type).unwrap());
        }

        // test struct type
        {
            // no metadata will cause error
            let arrow_type = DataType::Struct(Fields::from(vec![
                Field::new("a", DataType::Int64, false),
                Field::new("b", DataType::Utf8, true),
            ]));
            assert_eq!(
                &arrow_type_to_type(&arrow_type).unwrap_err().to_string(),
                "DataInvalid => Field id not found in metadata"
            );

            let arrow_type = DataType::Struct(Fields::from(vec![
                Field::new("a", DataType::Int64, false).with_metadata(HashMap::from_iter([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    1.to_string(),
                )])),
                Field::new("b", DataType::Utf8, true).with_metadata(HashMap::from_iter([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    2.to_string(),
                )])),
            ]));
            let iceberg_type = Type::Struct(StructType::new(vec![
                NestedField {
                    id: 1,
                    doc: None,
                    name: "a".to_string(),
                    required: true,
                    field_type: Box::new(Type::Primitive(PrimitiveType::Long)),
                    initial_default: None,
                    write_default: None,
                }
                .into(),
                NestedField {
                    id: 2,
                    doc: None,
                    name: "b".to_string(),
                    required: false,
                    field_type: Box::new(Type::Primitive(PrimitiveType::String)),
                    initial_default: None,
                    write_default: None,
                }
                .into(),
            ]));
            assert_eq!(iceberg_type, arrow_type_to_type(&arrow_type).unwrap());
            assert_eq!(arrow_type, type_to_arrow_type(&iceberg_type).unwrap());

            // initial_default and write_default is ignored
            let iceberg_type = Type::Struct(StructType::new(vec![
                NestedField {
                    id: 1,
                    doc: None,
                    name: "a".to_string(),
                    required: true,
                    field_type: Box::new(Type::Primitive(PrimitiveType::Long)),
                    initial_default: Some(Literal::Primitive(PrimitiveLiteral::Int(114514))),
                    write_default: None,
                }
                .into(),
                NestedField {
                    id: 2,
                    doc: None,
                    name: "b".to_string(),
                    required: false,
                    field_type: Box::new(Type::Primitive(PrimitiveType::String)),
                    initial_default: None,
                    write_default: Some(Literal::Primitive(PrimitiveLiteral::String(
                        "514".to_string(),
                    ))),
                }
                .into(),
            ]));
            assert_eq!(arrow_type, type_to_arrow_type(&iceberg_type).unwrap());
        }

        // test dictionary type
        {
            let arrow_type =
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Int8));
            let iceberg_type = Type::Primitive(PrimitiveType::Int);
            assert_eq!(
                iceberg_type,
                arrow_type_to_type(&arrow_type).unwrap(),
                "Expected dictionary conversion to use the contained value"
            );

            let arrow_type =
                DataType::Dictionary(Box::new(DataType::Utf8), Box::new(DataType::Boolean));
            let iceberg_type = Type::Primitive(PrimitiveType::Boolean);
            assert_eq!(iceberg_type, arrow_type_to_type(&arrow_type).unwrap());
        }
    }

    #[test]
    fn test_unsigned_integer_type_conversion() {
        let test_cases = vec![
            (DataType::UInt8, PrimitiveType::Int),
            (DataType::UInt16, PrimitiveType::Int),
            (DataType::UInt32, PrimitiveType::Long),
        ];

        for (arrow_type, expected_iceberg_type) in test_cases {
            let arrow_field = Field::new("test", arrow_type.clone(), false).with_metadata(
                HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())]),
            );
            let arrow_schema = ArrowSchema::new(vec![arrow_field]);

            let iceberg_schema = arrow_schema_to_schema(&arrow_schema).unwrap();
            let iceberg_field = iceberg_schema.as_struct().fields().first().unwrap();

            assert!(
                matches!(iceberg_field.field_type.as_ref(), Type::Primitive(t) if *t == expected_iceberg_type),
                "Expected {arrow_type:?} to map to {expected_iceberg_type:?}"
            );
        }

        // Test UInt64 blocking
        {
            let arrow_field = Field::new("test", DataType::UInt64, false).with_metadata(
                HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())]),
            );
            let arrow_schema = ArrowSchema::new(vec![arrow_field]);

            let result = arrow_schema_to_schema(&arrow_schema);
            assert!(result.is_err());
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("UInt64 is not supported")
            );
        }
    }

    #[test]
    fn test_datum_conversion() {
        {
            let datum = Datum::bool(true);
            let arrow_datum = get_arrow_datum(&datum).unwrap();
            let (array, is_scalar) = arrow_datum.get();
            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            assert!(is_scalar);
            assert!(array.value(0));
        }
        {
            let datum = Datum::int(42);
            let arrow_datum = get_arrow_datum(&datum).unwrap();
            let (array, is_scalar) = arrow_datum.get();
            let array = array.as_any().downcast_ref::<Int32Array>().unwrap();
            assert!(is_scalar);
            assert_eq!(array.value(0), 42);
        }
        {
            let datum = Datum::long(42);
            let arrow_datum = get_arrow_datum(&datum).unwrap();
            let (array, is_scalar) = arrow_datum.get();
            let array = array.as_any().downcast_ref::<Int64Array>().unwrap();
            assert!(is_scalar);
            assert_eq!(array.value(0), 42);
        }
        {
            let datum = Datum::float(42.42);
            let arrow_datum = get_arrow_datum(&datum).unwrap();
            let (array, is_scalar) = arrow_datum.get();
            let array = array.as_any().downcast_ref::<Float32Array>().unwrap();
            assert!(is_scalar);
            assert_eq!(array.value(0), 42.42);
        }
        {
            let datum = Datum::double(42.42);
            let arrow_datum = get_arrow_datum(&datum).unwrap();
            let (array, is_scalar) = arrow_datum.get();
            let array = array.as_any().downcast_ref::<Float64Array>().unwrap();
            assert!(is_scalar);
            assert_eq!(array.value(0), 42.42);
        }
        {
            let datum = Datum::string("abc");
            let arrow_datum = get_arrow_datum(&datum).unwrap();
            let (array, is_scalar) = arrow_datum.get();
            let array = array.as_any().downcast_ref::<StringArray>().unwrap();
            assert!(is_scalar);
            assert_eq!(array.value(0), "abc");
        }
        {
            let datum = Datum::binary(vec![1, 2, 3, 4]);
            let arrow_datum = get_arrow_datum(&datum).unwrap();
            let (array, is_scalar) = arrow_datum.get();
            let array = array.as_any().downcast_ref::<BinaryArray>().unwrap();
            assert!(is_scalar);
            assert_eq!(array.value(0), &[1, 2, 3, 4]);
        }
        {
            let datum = Datum::date(42);
            let arrow_datum = get_arrow_datum(&datum).unwrap();
            let (array, is_scalar) = arrow_datum.get();
            let array = array.as_any().downcast_ref::<Date32Array>().unwrap();
            assert!(is_scalar);
            assert_eq!(array.value(0), 42);
        }
        {
            let datum = Datum::timestamp_micros(42);
            let arrow_datum = get_arrow_datum(&datum).unwrap();
            let (array, is_scalar) = arrow_datum.get();
            let array = array
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            assert!(is_scalar);
            assert_eq!(array.value(0), 42);
        }
        {
            let datum = Datum::timestamptz_micros(42);
            let arrow_datum = get_arrow_datum(&datum).unwrap();
            let (array, is_scalar) = arrow_datum.get();
            let array = array
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            assert!(is_scalar);
            assert_eq!(array.timezone(), Some(UTC_TIME_ZONE));
            assert_eq!(array.value(0), 42);
        }
        {
            let datum = Datum::decimal_with_precision(decimal_new(123, 2), 30).unwrap();
            let arrow_datum = get_arrow_datum(&datum).unwrap();
            let (array, is_scalar) = arrow_datum.get();
            let array = array.as_any().downcast_ref::<Decimal128Array>().unwrap();
            assert!(is_scalar);
            assert_eq!(array.precision(), 30);
            assert_eq!(array.scale(), 2);
            assert_eq!(array.value(0), 123);
        }
        {
            let datum = Datum::uuid_from_str("42424242-4242-4242-4242-424242424242").unwrap();
            let arrow_datum = get_arrow_datum(&datum).unwrap();
            let (array, is_scalar) = arrow_datum.get();
            let array = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap();
            assert!(is_scalar);
            assert_eq!(array.value(0), [66u8; 16]);
        }
        {
            // Time → Time64(Microsecond) scalar carrying the i64 micros-from-midnight backing.
            let datum = Datum::time_micros(3_661_000_000).expect("valid time-of-day micros");
            let arrow_datum = get_arrow_datum(&datum).expect("Time datum must convert");
            let (array, is_scalar) = arrow_datum.get();
            let array = array
                .as_any()
                .downcast_ref::<Time64MicrosecondArray>()
                .expect("Time scalar must be a Time64MicrosecondArray");
            assert!(is_scalar);
            assert_eq!(array.value(0), 3_661_000_000);
        }
        {
            // Fixed(n) → FixedSizeBinary(n) scalar carrying the exact byte buffer.
            let datum = Datum::fixed(vec![0xDEu8, 0xAD, 0xBE, 0xEF]);
            let arrow_datum = get_arrow_datum(&datum).expect("Fixed datum must convert");
            let (array, is_scalar) = arrow_datum.get();
            let array = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .expect("Fixed scalar must be a FixedSizeBinaryArray");
            assert!(is_scalar);
            assert_eq!(array.value_length(), 4);
            assert_eq!(array.value(0), &[0xDEu8, 0xAD, 0xBE, 0xEF]);
        }
    }

    /// A `Datum` can carry a decimal `precision > 38` through bypass paths such as
    /// [`Datum::new`] or [`Datum::try_from_bytes`], so a corrupt or hostile catalog/manifest can
    /// hand the predicate path such a datum. Arrow's Decimal128 tops out at precision 38, so
    /// `with_precision_and_scale` rejects it. `get_arrow_datum` must surface that as a typed
    /// [`ErrorKind::DataInvalid`], never a panic (a predicate pushdown that panics takes down the
    /// scan/worker instead of failing the one bad query).
    #[test]
    fn get_arrow_datum_rejects_over_max_decimal_precision_without_panicking() {
        // precision 50 > Arrow's Decimal128 max of 38; built via the pub(crate) constructor to
        // mirror what bypass paths can still produce from corrupt metadata.
        let datum = Datum::new(
            PrimitiveType::Decimal {
                precision: 50,
                scale: 0,
            },
            PrimitiveLiteral::Int128(1234),
        );

        match get_arrow_datum(&datum) {
            Ok(_) => {
                panic!("decimal precision 50 exceeds Arrow Decimal128 max and must be an error")
            }
            Err(err) => assert_eq!(err.kind(), ErrorKind::DataInvalid),
        }

        // A second, independent rejection class Arrow's `with_precision_and_scale` enforces:
        // `scale > precision`. Also unvalidated at the type/datum layer, also must be a typed error.
        let bad_scale = Datum::new(
            PrimitiveType::Decimal {
                precision: 10,
                scale: 20,
            },
            PrimitiveLiteral::Int128(1234),
        );
        match get_arrow_datum(&bad_scale) {
            Ok(_) => panic!("decimal scale 20 > precision 10 is invalid and must be an error"),
            Err(err) => assert_eq!(err.kind(), ErrorKind::DataInvalid),
        }
    }

    /// The precision/scale rejection must be COMPLETE, not just "large values". Arrow takes a `u8`
    /// precision + `i8` scale, so casting the `u32` fields with `as` would WRAP an out-of-range value
    /// INTO Arrow's valid range and SILENTLY accept it — `decimal(294,0)` wraps to precision 38,
    /// `scale=256` wraps to i8 0, both of which Arrow's `with_precision_and_scale` then ACCEPTS.
    /// `get_arrow_datum` uses `try_from` (not `as`) so these are rejected as typed
    /// [`ErrorKind::DataInvalid`]. (Without the `try_from`, this test FAILS — the wrapped values pass.)
    #[test]
    fn get_arrow_datum_rejects_wrapping_decimal_precision_scale() {
        // precision 294 wraps to 38 under `as u8` (294 - 256) — a VALID Arrow precision.
        let wrapping_precision = Datum::new(
            PrimitiveType::Decimal {
                precision: 294,
                scale: 0,
            },
            PrimitiveLiteral::Int128(1234),
        );
        match get_arrow_datum(&wrapping_precision) {
            Ok(_) => panic!("decimal precision 294 wraps to 38 under `as u8` and must be rejected"),
            Err(err) => assert_eq!(err.kind(), ErrorKind::DataInvalid),
        }

        // scale 256 wraps to i8 0 under `as i8`; precision 10 is valid, so the wrapped (10,0) is a
        // decimal Arrow accepts — the truncating cast would silently change scale 256 into 0.
        let wrapping_scale = Datum::new(
            PrimitiveType::Decimal {
                precision: 10,
                scale: 256,
            },
            PrimitiveLiteral::Int128(1234),
        );
        match get_arrow_datum(&wrapping_scale) {
            Ok(_) => panic!("decimal scale 256 wraps to i8 0 under `as i8` and must be rejected"),
            Err(err) => assert_eq!(err.kind(), ErrorKind::DataInvalid),
        }
    }

    /// A decimal value whose unscaled magnitude needs more digits than the declared precision is
    /// not representable by Arrow Decimal128 at that precision. `get_arrow_datum` must reject it
    /// instead of accepting a value whose type metadata lies about its precision.
    #[test]
    fn get_arrow_datum_rejects_decimal_values_outside_declared_precision_and_accepts_boundaries() {
        for (precision, value, context) in [
            (2, 123, "positive value with too many digits"),
            (2, -100, "negative value one past the precision boundary"),
            (
                38,
                i128::MIN,
                "i128::MIN cannot fit Arrow Decimal128's maximum precision",
            ),
        ] {
            let datum = Datum::new(
                PrimitiveType::Decimal {
                    precision,
                    scale: 0,
                },
                PrimitiveLiteral::Int128(value),
            );
            match get_arrow_datum(&datum) {
                Ok(_) => panic!(
                    "{context}: decimal({precision},0) cannot represent unscaled value {value}"
                ),
                Err(err) => assert_eq!(err.kind(), ErrorKind::DataInvalid, "{context}"),
            }
        }

        for value in [99, -99] {
            let boundary = Datum::new(
                PrimitiveType::Decimal {
                    precision: 2,
                    scale: 0,
                },
                PrimitiveLiteral::Int128(value),
            );
            let arrow_datum =
                get_arrow_datum(&boundary).expect("decimal(2,0) boundary must convert exactly");
            let (array, is_scalar) = arrow_datum.get();
            let array = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("decimal datum must produce a Decimal128Array");
            assert!(is_scalar);
            assert_eq!(array.precision(), 2);
            assert_eq!(array.scale(), 0);
            assert_eq!(array.value(0), value);
        }
    }

    #[test]
    fn test_arrow_schema_to_schema_with_field_id() {
        // Create a complex Arrow schema without field ID metadata
        // Including: primitives, list, nested struct, map, and nested list of structs
        let arrow_schema = ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("price", DataType::Decimal128(10, 2), false),
            Field::new(
                "created_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("+00:00".into())),
                true,
            ),
            Field::new(
                "tags",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
            Field::new(
                "address",
                DataType::Struct(Fields::from(vec![
                    Field::new("street", DataType::Utf8, true),
                    Field::new("city", DataType::Utf8, false),
                    Field::new("zip", DataType::Int32, true),
                ])),
                true,
            ),
            Field::new(
                "attributes",
                DataType::Map(
                    Arc::new(Field::new(
                        DEFAULT_MAP_FIELD_NAME,
                        DataType::Struct(Fields::from(vec![
                            Field::new("key", DataType::Utf8, false),
                            Field::new("value", DataType::Utf8, true),
                        ])),
                        false,
                    )),
                    false,
                ),
                true,
            ),
            Field::new(
                "orders",
                DataType::List(Arc::new(Field::new(
                    "element",
                    DataType::Struct(Fields::from(vec![
                        Field::new("order_id", DataType::Int64, false),
                        Field::new("amount", DataType::Float64, false),
                    ])),
                    true,
                ))),
                true,
            ),
        ]);

        let schema = arrow_schema_to_schema_auto_assign_ids(&arrow_schema).unwrap();

        // Build expected schema with exact field IDs following level-order assignment:
        // Level 0: id=1, name=2, price=3, created_at=4, tags=5, address=6, attributes=7, orders=8
        // Level 1: tags.element=9, address.{street=10,city=11,zip=12}, attributes.{key=13,value=14}, orders.element=15
        // Level 2: orders.element.{order_id=16,amount=17}
        let expected = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(
                    3,
                    "price",
                    Type::Primitive(PrimitiveType::Decimal {
                        precision: 10,
                        scale: 2,
                    }),
                )
                .into(),
                NestedField::optional(4, "created_at", Type::Primitive(PrimitiveType::Timestamptz))
                    .into(),
                NestedField::optional(
                    5,
                    "tags",
                    Type::List(ListType {
                        element_field: NestedField::list_element(
                            9,
                            Type::Primitive(PrimitiveType::String),
                            false,
                        )
                        .into(),
                    }),
                )
                .into(),
                NestedField::optional(
                    6,
                    "address",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(10, "street", Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::required(11, "city", Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::optional(12, "zip", Type::Primitive(PrimitiveType::Int))
                            .into(),
                    ])),
                )
                .into(),
                NestedField::optional(
                    7,
                    "attributes",
                    Type::Map(MapType {
                        key_field: NestedField::map_key_element(
                            13,
                            Type::Primitive(PrimitiveType::String),
                        )
                        .into(),
                        value_field: NestedField::map_value_element(
                            14,
                            Type::Primitive(PrimitiveType::String),
                            false,
                        )
                        .into(),
                    }),
                )
                .into(),
                NestedField::optional(
                    8,
                    "orders",
                    Type::List(ListType {
                        element_field: NestedField::list_element(
                            15,
                            Type::Struct(StructType::new(vec![
                                NestedField::required(
                                    16,
                                    "order_id",
                                    Type::Primitive(PrimitiveType::Long),
                                )
                                .into(),
                                NestedField::required(
                                    17,
                                    "amount",
                                    Type::Primitive(PrimitiveType::Double),
                                )
                                .into(),
                            ])),
                            false,
                        )
                        .into(),
                    }),
                )
                .into(),
            ])
            .build()
            .unwrap();

        pretty_assertions::assert_eq!(schema, expected);
        assert_eq!(schema.highest_field_id(), 17);
    }
}
