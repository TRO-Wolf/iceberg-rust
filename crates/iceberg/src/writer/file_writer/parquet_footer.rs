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

use parquet::arrow::arrow_writer::ArrowWriterOptions;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;

use crate::spec::Schema;

pub(crate) const ICEBERG_SCHEMA_META_KEY: &str = "iceberg.schema";

pub(super) fn writer_options(props: &WriterProperties, schema: &Schema) -> ArrowWriterOptions {
    let schema_json = serde_json::to_string(schema).expect("the Iceberg schema serializes to JSON");
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
    ArrowWriterOptions::new()
        .with_properties(props)
        .with_skip_arrow_metadata(true)
}
