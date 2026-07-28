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

use typed_builder::TypedBuilder;

use super::{FormatVersion, ManifestContentType, PartitionSpec, Schema};
use crate::error::Result;
use crate::spec::{PartitionField, SchemaId, SchemaRef};
use crate::{Error, ErrorKind};

/// Meta data of a manifest that is stored in the key-value metadata of the Avro file
#[derive(Debug, PartialEq, Clone, Eq, TypedBuilder)]
pub struct ManifestMetadata {
    /// The table schema at the time the manifest
    /// was written
    pub schema: SchemaRef,
    /// ID of the schema used to write the manifest as a string
    pub schema_id: SchemaId,
    /// The partition spec used to write the manifest
    pub partition_spec: PartitionSpec,
    /// Table format version number of the manifest as a string
    pub format_version: FormatVersion,
    /// Type of content files tracked by the manifest: “data” or “deletes”
    pub content: ManifestContentType,
}

impl ManifestMetadata {
    /// Parse from metadata in avro file (strict — no table-schema fallback).
    pub fn parse(meta: &HashMap<String, Vec<u8>>) -> Result<Self> {
        Self::parse_with_schema_fallback(meta, None)
    }

    /// Parse from Avro file-metadata, with optional **table-schema fallback**.
    ///
    /// The Iceberg `"schema"` key must deserialize as a real table [`Schema`]. Some third-party
    /// writers (notably DuckDB 1.5.x) put the **manifest-entry** Avro record shape there instead,
    /// which fails untagged [`SchemaEnum`] deserialization.
    ///
    /// Contract (QD / RePark interop — deliberate fork tolerance):
    /// - **Strict parse first.** Never salvage or partially decode the malformed JSON.
    /// - On parse **failure** (or missing key), if `schema_fallback` is `Some`, use that schema
    ///   (the table/snapshot schema) and emit a `tracing` warning.
    /// - On parse **failure** with no fallback, return the same hard error as historically.
    /// - On parse **success**, always use the embedded schema — even if a fallback was provided.
    pub fn parse_with_schema_fallback(
        meta: &HashMap<String, Vec<u8>>,
        schema_fallback: Option<SchemaRef>,
    ) -> Result<Self> {
        let (schema, used_fallback) = match resolve_manifest_schema(meta, schema_fallback)? {
            ResolvedManifestSchema::Embedded(s) => (s, false),
            ResolvedManifestSchema::Fallback(s) => (s, true),
        };
        // When we discarded the embedded schema body, the free-standing `schema-id`
        // key (often 0 for DuckDB poison) must not disagree with the body we installed
        // (C1-Q-005 / C1-L-002). Prefer the fallback schema's id.
        let schema_id: i32 = if used_fallback {
            schema.schema_id()
        } else {
            meta.get("schema-id")
                .map(|bs| {
                    String::from_utf8_lossy(bs).parse().map_err(|err| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "Fail to parse schema id in manifest metadata",
                        )
                        .with_source(err)
                    })
                })
                .transpose()?
                .unwrap_or_else(|| schema.schema_id())
        };
        let partition_spec = {
            let fields = {
                let bs = meta.get("partition-spec").ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        "partition-spec is required in manifest metadata but not found",
                    )
                })?;
                serde_json::from_slice::<Vec<PartitionField>>(bs).map_err(|err| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        "Fail to parse partition spec in manifest metadata",
                    )
                    .with_source(err)
                })?
            };
            let spec_id = meta
                .get("partition-spec-id")
                .map(|bs| {
                    String::from_utf8_lossy(bs).parse().map_err(|err| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "Fail to parse partition spec id in manifest metadata",
                        )
                        .with_source(err)
                    })
                })
                .transpose()?
                .unwrap_or(0);
            PartitionSpec::builder(schema.clone())
                .with_spec_id(spec_id)
                .add_unbound_fields(fields.into_iter().map(|f| f.into_unbound()))?
                .build()?
        };
        let format_version = if let Some(bs) = meta.get("format-version") {
            serde_json::from_slice::<FormatVersion>(bs).map_err(|err| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Fail to parse format version in manifest metadata",
                )
                .with_source(err)
            })?
        } else {
            FormatVersion::V1
        };
        let content = if let Some(v) = meta.get("content") {
            let v = String::from_utf8_lossy(v);
            v.parse()?
        } else {
            ManifestContentType::Data
        };
        Ok(ManifestMetadata {
            schema,
            schema_id,
            partition_spec,
            format_version,
            content,
        })
    }

    /// Get the schema of table at the time manifest was written
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Get the ID of schema used to write the manifest
    pub fn schema_id(&self) -> SchemaId {
        self.schema_id
    }

    /// Get the partition spec used to write manifest
    pub fn partition_spec(&self) -> &PartitionSpec {
        &self.partition_spec
    }

    /// Get the table format version
    pub fn format_version(&self) -> &FormatVersion {
        &self.format_version
    }

    /// Get the type of content files tracked by manifest
    pub fn content(&self) -> &ManifestContentType {
        &self.content
    }
}

/// Outcome of resolving the Iceberg `"schema"` key in manifest Avro file-metadata.
enum ResolvedManifestSchema {
    Embedded(SchemaRef),
    Fallback(SchemaRef),
}

/// Strictly parse the embedded `"schema"` key, or fall back to `schema_fallback` on failure.
///
/// Never attempts a lenient / partial decode of a non-conforming payload.
fn resolve_manifest_schema(
    meta: &HashMap<String, Vec<u8>>,
    schema_fallback: Option<SchemaRef>,
) -> Result<ResolvedManifestSchema> {
    let Some(bs) = meta.get("schema") else {
        return match schema_fallback {
            Some(fallback) => {
                tracing::warn!(
                    "manifest Avro file-metadata missing required key `schema`; \
                     falling back to table/snapshot schema id={}",
                    fallback.schema_id()
                );
                Ok(ResolvedManifestSchema::Fallback(fallback))
            }
            None => Err(Error::new(
                ErrorKind::DataInvalid,
                "schema is required in manifest metadata but not found",
            )),
        };
    };

    match serde_json::from_slice::<Schema>(bs) {
        Ok(schema) => Ok(ResolvedManifestSchema::Embedded(Arc::new(schema))),
        Err(err) => match schema_fallback {
            Some(fallback) => {
                // Do not include the poison JSON body (can be large / noisy); source chain
                // keeps the serde detail for debugging.
                tracing::warn!(
                    error = %err,
                    fallback_schema_id = fallback.schema_id(),
                    "manifest Avro file-metadata `schema` failed strict Schema parse \
                     (untagged SchemaEnum / type JSON); discarding embedded payload and \
                     falling back to table/snapshot schema — third-party writer malformation \
                     (e.g. DuckDB putting the manifest-entry record in this key)"
                );
                Ok(ResolvedManifestSchema::Fallback(fallback))
            }
            None => Err(Error::new(
                ErrorKind::DataInvalid,
                "Fail to parse schema in manifest metadata",
            )
            .with_source(err)),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{NestedField, PrimitiveType, Type};

    fn toy_table_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(7)
                .with_fields([
                    NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .expect("toy schema"),
        )
    }

    fn base_meta(schema_json: &str) -> HashMap<String, Vec<u8>> {
        let mut m = HashMap::new();
        m.insert("schema".to_string(), schema_json.as_bytes().to_vec());
        m.insert("schema-id".to_string(), b"0".to_vec());
        m.insert("partition-spec".to_string(), b"[]".to_vec());
        m.insert("partition-spec-id".to_string(), b"0".to_vec());
        m.insert("format-version".to_string(), b"2".to_vec());
        m.insert("content".to_string(), b"data".to_vec());
        m
    }

    /// DuckDB-shaped poison: manifest-entry field names + raw Avro map types in the Iceberg slot.
    fn duckdb_poison_schema_json() -> String {
        r#"{
          "type": "struct",
          "schema-id": 0,
          "fields": [
            {"id": 0, "name": "status", "required": true, "type": "int"},
            {"id": 1, "name": "snapshot_id", "required": false, "type": "long"},
            {
              "id": 2,
              "name": "data_file",
              "required": true,
              "type": {
                "type": "struct",
                "fields": [
                  {
                    "id": 125,
                    "name": "lower_bounds",
                    "required": false,
                    "type": {
                      "type": "array",
                      "items": {
                        "type": "record",
                        "name": "k126_k127",
                        "fields": [
                          {"name": "key", "type": "int", "id": 126},
                          {"name": "value", "type": "binary", "id": 127}
                        ]
                      }
                    }
                  }
                ]
              }
            }
          ]
        }"#
        .to_string()
    }

    #[test]
    fn parse_valid_embedded_schema_ignores_fallback() {
        let good = r#"{
          "type": "struct",
          "schema-id": 3,
          "fields": [
            {"id": 1, "name": "id", "required": false, "type": "int"}
          ]
        }"#;
        let meta = base_meta(good);
        let fallback = toy_table_schema();
        let parsed = ManifestMetadata::parse_with_schema_fallback(&meta, Some(fallback.clone()))
            .expect("valid schema must parse");
        assert_eq!(parsed.schema.schema_id(), 3);
        assert_eq!(parsed.schema.as_ref().field_by_id(1).unwrap().name, "id");
        // Must NOT have substituted the fallback (id 7).
        assert_ne!(parsed.schema.schema_id(), fallback.schema_id());
    }

    #[test]
    fn parse_poison_without_fallback_hard_fails() {
        let meta = base_meta(&duckdb_poison_schema_json());
        let err = ManifestMetadata::parse(&meta).expect_err("poison must fail without fallback");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Fail to parse schema in manifest metadata")
                || msg.contains("SchemaEnum")
                || msg.contains("did not match"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn parse_poison_with_fallback_uses_table_schema() {
        let meta = base_meta(&duckdb_poison_schema_json());
        let fallback = toy_table_schema();
        let parsed = ManifestMetadata::parse_with_schema_fallback(&meta, Some(fallback.clone()))
            .expect("poison + fallback must succeed");
        assert_eq!(parsed.schema.schema_id(), fallback.schema_id());
        // Free-standing schema_id must agree with body after fallback (C1-Q-005).
        assert_eq!(parsed.schema_id(), fallback.schema_id());
        assert!(parsed.schema.as_ref().field_by_id(1).is_some());
        assert!(
            parsed.schema.as_ref().field_by_name("status").is_none(),
            "must not keep manifest-entry field names from the poison payload"
        );
    }

    #[test]
    fn parse_missing_schema_with_fallback() {
        let mut meta = base_meta(r#"{"type":"struct","schema-id":0,"fields":[]}"#);
        meta.remove("schema");
        let fallback = toy_table_schema();
        let parsed = ManifestMetadata::parse_with_schema_fallback(&meta, Some(fallback.clone()))
            .expect("missing schema + fallback");
        assert_eq!(parsed.schema.schema_id(), fallback.schema_id());
    }
}
