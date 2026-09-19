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

use parquet::arrow::arrow_writer::ArrowWriterOptions;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use serde::Serialize;

use super::parquet_compression_from_properties;
use crate::spec::{NestedFieldRef, Schema};
use crate::{Error, ErrorKind, Result};

pub(crate) const ICEBERG_SCHEMA_META_KEY: &str = "iceberg.schema";

pub(crate) const DELETE_TYPE_META_KEY: &str = "delete-type";

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
struct JavaOrderedSchema<'a> {
    r#type: &'static str,
    schema_id: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    identifier_field_ids: Option<Vec<i32>>,
    fields: &'a [NestedFieldRef],
}

pub(crate) fn java_ordered_schema_json(schema: &Schema) -> Result<String> {
    let mut identifier_field_ids: Vec<i32> = schema.identifier_field_ids().collect();
    identifier_field_ids.sort_unstable();
    serde_json::to_string(&JavaOrderedSchema {
        r#type: "struct",
        schema_id: schema.schema_id(),
        identifier_field_ids: (!identifier_field_ids.is_empty()).then_some(identifier_field_ids),
        fields: schema.as_struct().fields(),
    })
    .map_err(|err| {
        Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Failed to serialize the Iceberg schema (schema_id {}) for the parquet footer: {err}",
                schema.schema_id()
            ),
        )
    })
}

pub(super) fn writer_options(
    props: &WriterProperties,
    schema: &Schema,
) -> Result<ArrowWriterOptions> {
    let schema_json = java_ordered_schema_json(schema)?;
    let mut key_values = props.key_value_metadata().cloned().unwrap_or_default();
    key_values.retain(|entry| entry.key != ICEBERG_SCHEMA_META_KEY);
    key_values.push(KeyValue::new(
        ICEBERG_SCHEMA_META_KEY.to_string(),
        schema_json,
    ));
    let props = props
        .clone()
        .into_builder()
        .set_key_value_metadata(Some(key_values))
        .build();
    Ok(ArrowWriterOptions::new()
        .with_properties(props)
        .with_schema_root("table".to_string())
        .with_skip_arrow_metadata(true))
}

#[allow(missing_docs)]
pub fn equality_delete_writer_properties() -> WriterProperties {
    WriterProperties::builder()
        .set_key_value_metadata(Some(vec![KeyValue::new(
            DELETE_TYPE_META_KEY.to_string(),
            "equality".to_string(),
        )]))
        .build()
}

#[allow(missing_docs)]
pub fn equality_delete_writer_properties_for(
    properties: &HashMap<String, String>,
) -> Result<WriterProperties> {
    Ok(WriterProperties::builder()
        .set_key_value_metadata(Some(vec![KeyValue::new(
            DELETE_TYPE_META_KEY.to_string(),
            "equality".to_string(),
        )]))
        .set_compression(parquet_compression_from_properties(properties)?)
        .build())
}
