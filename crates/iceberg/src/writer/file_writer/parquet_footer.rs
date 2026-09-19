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

use super::parquet_compression_from_properties;
use crate::spec::Schema;
use crate::{Error, ErrorKind, Result};

pub(crate) const ICEBERG_SCHEMA_META_KEY: &str = "iceberg.schema";

pub(crate) const DELETE_TYPE_META_KEY: &str = "delete-type";

pub(super) fn writer_options(
    props: &WriterProperties,
    schema: &Schema,
) -> Result<ArrowWriterOptions> {
    let schema_json = serde_json::to_string(schema).map_err(|err| {
        Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Failed to serialize the Iceberg schema (schema_id {}) for the parquet footer: {err}",
                schema.schema_id()
            ),
        )
    })?;
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
