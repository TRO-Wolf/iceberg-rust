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
use std::fmt::Display;
use std::str::FromStr;

use uuid::Uuid;

use crate::spec::{TableMetadata, TableProperties};
use crate::utils::strip_trailing_slash;
use crate::{Error, ErrorKind, Result};

/// Helper for parsing a metadata JSON location under `<table>/metadata/`.
///
/// Hive/REST names are `<version>-<uuid>.metadata.json`. Hadoop names are
/// `v<version>.metadata.json` (Java `HadoopTableOperations`, row R167).
#[derive(Clone, Debug, PartialEq)]
pub struct MetadataLocation {
    metadata_dir: String,
    version: i32,
    /// `None` is the Hadoop convention. A uuid is the Hive/REST convention.
    id: Option<Uuid>,
}

pub(crate) fn write_metadata_dir(
    table_location: &str,
    properties: &HashMap<String, String>,
) -> Result<String> {
    match properties.get(TableProperties::PROPERTY_WRITE_METADATA_LOCATION) {
        Some(dir) => strip_trailing_slash(dir).map(str::to_string),
        None => Ok(format!("{table_location}/metadata")),
    }
}

impl MetadataLocation {
    /// Creates a completely new metadata location starting at version 0.
    /// Only used for creating a new table. For updates, see `with_next_version`.
    pub fn new_with_table_location(table_location: impl ToString) -> Self {
        Self {
            metadata_dir: format!("{}/metadata", table_location.to_string()),
            version: 0,
            id: Some(Uuid::new_v4()),
        }
    }

    #[allow(missing_docs)]
    pub fn for_metadata(metadata: &TableMetadata) -> Result<Self> {
        Ok(Self {
            metadata_dir: write_metadata_dir(metadata.location(), metadata.properties())?,
            version: 0,
            id: Some(Uuid::new_v4()),
        })
    }

    pub(crate) fn for_hadoop_metadata(metadata: &TableMetadata) -> Result<Self> {
        Self {
            metadata_dir: format!("{}/metadata", metadata.location()),
            version: 1,
            id: None,
        }
        .rebased(metadata)
    }

    /// Creates a new metadata location for an updated metadata file.
    ///
    /// A Hadoop pointer stays Hadoop: `vN` becomes `v(N+1)`. Hive/REST gets a new uuid.
    /// The next Hadoop file is uncompressed `.metadata.json` even if the current file was gzip.
    pub fn with_next_version(&self) -> Self {
        Self {
            metadata_dir: self.metadata_dir.clone(),
            version: self.version.wrapping_add(1),
            id: self.id.map(|_| Uuid::new_v4()),
        }
    }

    /// Reports whether this location uses the Hadoop `vN.metadata.json` convention.
    pub fn is_hadoop_convention(&self) -> bool {
        self.id.is_none()
    }

    pub(crate) fn rebased(&self, metadata: &TableMetadata) -> Result<Self> {
        if self.is_hadoop_convention()
            && metadata
                .properties()
                .contains_key(TableProperties::PROPERTY_WRITE_METADATA_LOCATION)
        {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Hadoop path-based tables cannot relocate metadata",
            ));
        }
        let metadata_dir = if self.is_hadoop_convention() {
            self.metadata_dir.clone()
        } else {
            write_metadata_dir(metadata.location(), metadata.properties())?
        };
        Ok(Self {
            metadata_dir,
            version: self.version,
            id: self.id,
        })
    }

    pub(crate) fn hadoop_version_hint(&self) -> Option<(String, String)> {
        self.is_hadoop_convention().then(|| {
            (
                format!("{}/version-hint.text", self.metadata_dir),
                self.version.to_string(),
            )
        })
    }

    pub(crate) fn hadoop_version_siblings(&self) -> Option<[String; 2]> {
        if !self.is_hadoop_convention() {
            return None;
        }
        Some([
            format!("{}/v{}.gz.metadata.json", self.metadata_dir, self.version),
            format!("{}/v{}.metadata.json.gz", self.metadata_dir, self.version),
        ])
    }

    pub(crate) fn from_file_path(s: &str) -> Result<Self> {
        let (dir, file_name) = s.rsplit_once('/').ok_or(Error::new(
            ErrorKind::Unexpected,
            format!("Invalid metadata location: {s}"),
        ))?;
        let (version, id) = Self::parse_file_name(file_name)?;
        Ok(MetadataLocation {
            metadata_dir: dir.to_string(),
            version,
            id,
        })
    }

    fn parse_metadata_path_prefix(path: &str) -> Result<String> {
        let prefix = path.strip_suffix("/metadata").ok_or(Error::new(
            ErrorKind::Unexpected,
            format!("Metadata location not under \"/metadata\" subdirectory: {path}"),
        ))?;

        Ok(prefix.to_string())
    }

    /// Parses Hive `<version>-<uuid>` or Hadoop `v<version>`, including gzip suffixes.
    fn parse_file_name(file_name: &str) -> Result<(i32, Option<Uuid>)> {
        let stem = file_name
            .strip_suffix(".metadata.json.gz")
            .or_else(|| file_name.strip_suffix(".gz.metadata.json"))
            .or_else(|| file_name.strip_suffix(".metadata.json"))
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Invalid metadata file ending: {file_name}"),
                )
            })?;

        if let Some(rest) = stem.strip_prefix('v')
            && let Ok(version) = rest.parse::<i32>()
        {
            return Ok((version, None));
        }

        let (version, id) = stem.split_once('-').ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                format!("Invalid metadata file name format: {file_name}"),
            )
        })?;

        Ok((version.parse::<i32>()?, Some(Uuid::parse_str(id)?)))
    }
}

impl Display for MetadataLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.id {
            Some(id) => write!(
                f,
                "{}/{:0>5}-{}.metadata.json",
                self.metadata_dir, self.version, id
            ),
            None => write!(f, "{}/v{}.metadata.json", self.metadata_dir, self.version),
        }
    }
}

impl FromStr for MetadataLocation {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let (path, file_name) = s.rsplit_once('/').ok_or(Error::new(
            ErrorKind::Unexpected,
            format!("Invalid metadata location: {s}"),
        ))?;

        Self::parse_metadata_path_prefix(path)?;
        let (version, id) = Self::parse_file_name(file_name)?;

        Ok(MetadataLocation {
            metadata_dir: path.to_string(),
            version,
            id,
        })
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use std::str::FromStr;

    use uuid::Uuid;

    use crate::spec::{FormatVersion, PartitionSpec, StructType, TableMetadata};
    use crate::{ErrorKind, MetadataLocation};

    fn table_metadata(location: &str, properties: HashMap<String, String>) -> TableMetadata {
        TableMetadata {
            format_version: FormatVersion::V2,
            table_uuid: Uuid::new_v4(),
            location: location.to_string(),
            last_updated_ms: 0,
            last_column_id: 1,
            schemas: HashMap::new(),
            current_schema_id: 1,
            partition_specs: HashMap::new(),
            default_spec: PartitionSpec::unpartition_spec().into(),
            default_partition_type: StructType::new(vec![]),
            last_partition_id: 1000,
            default_sort_order_id: 0,
            sort_orders: HashMap::new(),
            snapshots: HashMap::new(),
            current_snapshot_id: None,
            last_sequence_number: 1,
            properties,
            snapshot_log: Vec::new(),
            metadata_log: vec![],
            refs: HashMap::new(),
            statistics: HashMap::new(),
            partition_statistics: HashMap::new(),
            encryption_keys: HashMap::new(),
            next_row_id: 0,
        }
    }

    #[test]
    fn test_metadata_location_from_string() {
        let test_cases = vec![
            // No prefix
            (
                "/metadata/1234567-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json",
                Ok(MetadataLocation {
                    metadata_dir: "/metadata".to_string(),
                    version: 1234567,
                    id: Some(Uuid::from_str("2cd22b57-5127-4198-92ba-e4e67c79821b").unwrap()),
                }),
            ),
            // Some prefix
            (
                "/abc/metadata/1234567-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json",
                Ok(MetadataLocation {
                    metadata_dir: "/abc/metadata".to_string(),
                    version: 1234567,
                    id: Some(Uuid::from_str("2cd22b57-5127-4198-92ba-e4e67c79821b").unwrap()),
                }),
            ),
            // Longer prefix
            (
                "/abc/def/metadata/1234567-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json",
                Ok(MetadataLocation {
                    metadata_dir: "/abc/def/metadata".to_string(),
                    version: 1234567,
                    id: Some(Uuid::from_str("2cd22b57-5127-4198-92ba-e4e67c79821b").unwrap()),
                }),
            ),
            // Prefix with special characters
            (
                "https://127.0.0.1/metadata/1234567-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json",
                Ok(MetadataLocation {
                    metadata_dir: "https://127.0.0.1/metadata".to_string(),
                    version: 1234567,
                    id: Some(Uuid::from_str("2cd22b57-5127-4198-92ba-e4e67c79821b").unwrap()),
                }),
            ),
            // Another id
            (
                "/abc/metadata/1234567-81056704-ce5b-41c4-bb83-eb6408081af6.metadata.json",
                Ok(MetadataLocation {
                    metadata_dir: "/abc/metadata".to_string(),
                    version: 1234567,
                    id: Some(Uuid::from_str("81056704-ce5b-41c4-bb83-eb6408081af6").unwrap()),
                }),
            ),
            // Version 0
            (
                "/abc/metadata/00000-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json",
                Ok(MetadataLocation {
                    metadata_dir: "/abc/metadata".to_string(),
                    version: 0,
                    id: Some(Uuid::from_str("2cd22b57-5127-4198-92ba-e4e67c79821b").unwrap()),
                }),
            ),
            // Negative version
            (
                "/metadata/-123-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json",
                Err("".to_string()),
            ),
            // Invalid uuid
            (
                "/metadata/1234567-no-valid-id.metadata.json",
                Err("".to_string()),
            ),
            // Non-numeric version
            (
                "/metadata/noversion-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json",
                Err("".to_string()),
            ),
            // No /metadata subdirectory
            (
                "/wrongsubdir/1234567-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json",
                Err("".to_string()),
            ),
            // No .metadata.json suffix
            (
                "/metadata/1234567-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata",
                Err("".to_string()),
            ),
            (
                "/metadata/1234567-2cd22b57-5127-4198-92ba-e4e67c79821b.wrong.file",
                Err("".to_string()),
            ),
            (
                "/abc/metadata/v3.metadata.json",
                Ok(MetadataLocation {
                    metadata_dir: "/abc/metadata".to_string(),
                    version: 3,
                    id: None,
                }),
            ),
            (
                "/abc/metadata/v0.metadata.json",
                Ok(MetadataLocation {
                    metadata_dir: "/abc/metadata".to_string(),
                    version: 0,
                    id: None,
                }),
            ),
            (
                "/abc/metadata/v12.metadata.json",
                Ok(MetadataLocation {
                    metadata_dir: "/abc/metadata".to_string(),
                    version: 12,
                    id: None,
                }),
            ),
            (
                "/abc/metadata/v00003.metadata.json",
                Ok(MetadataLocation {
                    metadata_dir: "/abc/metadata".to_string(),
                    version: 3,
                    id: None,
                }),
            ),
            (
                "/abc/metadata/00003-2cd22b57-5127-4198-92ba-e4e67c79821b.gz.metadata.json",
                Ok(MetadataLocation {
                    metadata_dir: "/abc/metadata".to_string(),
                    version: 3,
                    id: Some(Uuid::from_str("2cd22b57-5127-4198-92ba-e4e67c79821b").unwrap()),
                }),
            ),
            (
                "/abc/metadata/00003-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json.gz",
                Ok(MetadataLocation {
                    metadata_dir: "/abc/metadata".to_string(),
                    version: 3,
                    id: Some(Uuid::from_str("2cd22b57-5127-4198-92ba-e4e67c79821b").unwrap()),
                }),
            ),
            (
                "/abc/metadata/v3.gz.metadata.json",
                Ok(MetadataLocation {
                    metadata_dir: "/abc/metadata".to_string(),
                    version: 3,
                    id: None,
                }),
            ),
            (
                "/abc/metadata/v3.metadata.json.gz",
                Ok(MetadataLocation {
                    metadata_dir: "/abc/metadata".to_string(),
                    version: 3,
                    id: None,
                }),
            ),
            ("/metadata/v.metadata.json", Err("".to_string())),
            // Entire rest after `v` must be i32. A digit prefix plus junk is not Hadoop.
            (
                "/metadata/v3-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json",
                Err("".to_string()),
            ),
            (
                "/metadata/version-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json",
                Err("".to_string()),
            ),
            ("/metadata/v3.0.metadata.json", Err("".to_string())),
            ("/metadata/v3.foo.metadata.json", Err("".to_string())),
            ("/metadata/v3x.metadata.json", Err("".to_string())),
        ];

        for (input, expected) in test_cases {
            match MetadataLocation::from_str(input) {
                Ok(metadata_location) => {
                    assert!(expected.is_ok());
                    assert_eq!(metadata_location, expected.unwrap());
                }
                Err(_) => assert!(expected.is_err()),
            }
        }
    }

    #[test]
    fn test_metadata_location_with_next_version() {
        let test_cases = vec![
            MetadataLocation::new_with_table_location("/abc"),
            MetadataLocation::from_str(
                "/abc/def/metadata/1234567-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json",
            )
            .unwrap(),
        ];

        for input in test_cases {
            let next = MetadataLocation::from_str(&input.to_string())
                .unwrap()
                .with_next_version();
            assert_eq!(next.metadata_dir, input.metadata_dir);
            assert_eq!(next.version, input.version + 1);
            assert_ne!(next.id, input.id);
            assert!(next.id.is_some());
        }
    }

    #[test]
    fn hadoop_next_version_is_v_n_plus_one_without_uuid() {
        let current =
            MetadataLocation::from_str("/wh/t/metadata/v3.metadata.json").expect("parse hadoop v3");
        let next = current.with_next_version();
        assert_eq!(next.metadata_dir, "/wh/t/metadata");
        assert_eq!(next.version, 4);
        assert_eq!(next.id, None);
        assert_eq!(next.to_string(), "/wh/t/metadata/v4.metadata.json");
    }

    #[test]
    fn gzip_hadoop_next_version_is_uncompressed() {
        let current = MetadataLocation::from_str("/wh/t/metadata/v3.gz.metadata.json")
            .expect("parse gzip hadoop");
        assert_eq!(
            current.with_next_version().to_string(),
            "/wh/t/metadata/v4.metadata.json"
        );
    }

    #[test]
    fn hive_next_version_stays_uuid_convention() {
        let current = MetadataLocation::from_str(
            "/abc/metadata/00003-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json",
        )
        .expect("parse hive");
        let next = current.with_next_version();
        assert_eq!(next.version, 4);
        assert!(next.id.is_some());
        assert_ne!(next.id, current.id);
        let rendered = next.to_string();
        assert!(
            rendered.starts_with("/abc/metadata/00004-"),
            "hive next must stay padded uuid form, got {rendered}"
        );
        assert!(rendered.ends_with(".metadata.json"));
    }

    #[test]
    fn hive_gzip_next_version_stays_uncompressed_uuid_convention() {
        let current = MetadataLocation::from_str(
            "/abc/metadata/00003-2cd22b57-5127-4198-92ba-e4e67c79821b.gz.metadata.json",
        )
        .expect("parse hive gzip");
        let next = current.with_next_version();
        assert_eq!(next.version, 4);
        assert!(next.id.is_some());
        let rendered = next.to_string();
        assert!(
            rendered.starts_with("/abc/metadata/00004-"),
            "hive gzip next must stay padded uuid form, got {rendered}"
        );
        assert!(
            rendered.ends_with(".metadata.json"),
            "next file is uncompressed, got {rendered}"
        );
    }

    #[test]
    fn hadoop_padded_v00003_next_is_unpadded_v4() {
        let current = MetadataLocation::from_str("/wh/t/metadata/v00003.metadata.json")
            .expect("parse padded hadoop");
        assert_eq!(current.version, 3);
        assert_eq!(
            current.with_next_version().to_string(),
            "/wh/t/metadata/v4.metadata.json"
        );
    }

    #[tokio::test]
    async fn register_hadoop_named_metadata_then_commit_writes_v_n_plus_one() {
        use std::collections::HashMap;

        use crate::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
        use crate::spec::{NestedField, PrimitiveType, Schema, Type};
        use crate::transaction::{ApplyTransactionAction, Transaction};
        use crate::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};

        let catalog = MemoryCatalogBuilder::default()
            .load(
                "mem",
                HashMap::from([(
                    MEMORY_CATALOG_WAREHOUSE.to_string(),
                    "/f14-hadoop-wh".to_string(),
                )]),
            )
            .await
            .expect("load catalog");

        let ns = NamespaceIdent::new("ns".into());
        catalog
            .create_namespace(&ns, HashMap::new())
            .await
            .expect("namespace");
        let source = catalog
            .create_table(
                &ns,
                TableCreation::builder()
                    .name("src".into())
                    .schema(
                        Schema::builder()
                            .with_fields(vec![
                                NestedField::required(
                                    1,
                                    "id",
                                    Type::Primitive(PrimitiveType::Long),
                                )
                                .into(),
                            ])
                            .build()
                            .expect("schema"),
                    )
                    .build(),
            )
            .await
            .expect("create source");

        let v3 = format!("{}/metadata/v3.metadata.json", source.metadata().location());
        source
            .metadata()
            .write_to(source.file_io(), &v3)
            .await
            .expect("write v3");

        let ident = TableIdent::new(ns, "hadoop".into());
        let registered = catalog
            .register_table(&ident, v3)
            .await
            .expect("register v3");
        assert_eq!(
            registered.metadata_location().expect("location"),
            format!("{}/metadata/v3.metadata.json", source.metadata().location())
        );

        let tx = Transaction::new(&registered);
        let committed = tx
            .update_table_properties()
            .set("k".to_string(), "v".to_string())
            .apply(tx)
            .expect("apply")
            .commit(&catalog)
            .await
            .expect("commit after hadoop register");

        assert_eq!(
            committed.metadata_location().expect("next location"),
            format!("{}/metadata/v4.metadata.json", source.metadata().location())
        );
        assert_eq!(
            committed
                .metadata()
                .properties()
                .get("k")
                .map(String::as_str),
            Some("v")
        );
    }

    #[tokio::test]
    async fn hadoop_pointer_commit_refuses_write_metadata_path() {
        use std::collections::HashMap;

        use crate::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
        use crate::spec::{NestedField, PrimitiveType, Schema, Type};
        use crate::transaction::{ApplyTransactionAction, Transaction};
        use crate::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};

        let catalog = MemoryCatalogBuilder::default()
            .load(
                "mem",
                HashMap::from([(
                    MEMORY_CATALOG_WAREHOUSE.to_string(),
                    "/f14-hadoop-wh".to_string(),
                )]),
            )
            .await
            .expect("load catalog");

        let ns = NamespaceIdent::new("ns".into());
        catalog
            .create_namespace(&ns, HashMap::new())
            .await
            .expect("namespace");
        let source = catalog
            .create_table(
                &ns,
                TableCreation::builder()
                    .name("src".into())
                    .schema(
                        Schema::builder()
                            .with_fields(vec![
                                NestedField::required(
                                    1,
                                    "id",
                                    Type::Primitive(PrimitiveType::Long),
                                )
                                .into(),
                            ])
                            .build()
                            .expect("schema"),
                    )
                    .build(),
            )
            .await
            .expect("create source");

        let v3 = format!("{}/metadata/v3.metadata.json", source.metadata().location());
        source
            .metadata()
            .write_to(source.file_io(), &v3)
            .await
            .expect("write v3");

        let ident = TableIdent::new(ns, "hadoop".into());
        let registered = catalog
            .register_table(&ident, v3.clone())
            .await
            .expect("register v3");

        let tx = Transaction::new(&registered);
        let err = match tx
            .update_table_properties()
            .set("write.metadata.path".to_string(), "/alt-meta".to_string())
            .apply(tx)
            .expect("apply")
            .commit(&catalog)
            .await
        {
            Ok(_) => {
                panic!("a hadoop-convention commit carrying write.metadata.path must fail")
            }
            Err(e) => e,
        };
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message()
                .contains("Hadoop path-based tables cannot relocate metadata"),
            "the refusal must carry Java's message, got: {err}"
        );

        let loaded = catalog.load_table(&ident).await.expect("load back");
        assert_eq!(
            loaded.metadata_location().expect("location"),
            v3,
            "a refused commit must not move the catalog pointer"
        );
        assert!(
            registered
                .file_io()
                .list("/alt-meta")
                .await
                .expect("list /alt-meta")
                .is_empty(),
            "no metadata file may land under write.metadata.path"
        );
        assert!(
            !registered
                .file_io()
                .exists(format!(
                    "{}/metadata/v4.metadata.json",
                    source.metadata().location()
                ))
                .await
                .expect("v4 exists check"),
            "no next-version file may land under the pointer directory either"
        );
    }

    #[test]
    fn from_file_path_accepts_relocated_metadata_dir() {
        let parsed = MetadataLocation::from_file_path(
            "/alt-meta/00000-a0c2e704-85a1-4368-8ed6-0b0b2a337ec9.metadata.json",
        )
        .expect("a write.metadata.path directory must parse");
        assert_eq!(parsed.version, 0);
        assert!(!parsed.is_hadoop_convention());
        assert_eq!(
            parsed.to_string(),
            "/alt-meta/00000-a0c2e704-85a1-4368-8ed6-0b0b2a337ec9.metadata.json"
        );

        let hadoop = MetadataLocation::from_file_path("/alt-meta/v3.metadata.json")
            .expect("a hadoop name under a relocated dir must parse");
        assert_eq!(hadoop.version, 3);
        assert!(hadoop.is_hadoop_convention());
        assert_eq!(
            hadoop.with_next_version().to_string(),
            "/alt-meta/v4.metadata.json"
        );
    }

    #[test]
    fn for_metadata_honors_write_metadata_path() {
        let relocated = MetadataLocation::for_metadata(&table_metadata(
            "/wh/ns/t",
            HashMap::from([("write.metadata.path".to_string(), "/alt-meta/".to_string())]),
        ))
        .expect("relocated create location");
        let rendered = relocated.to_string();
        assert!(
            rendered.starts_with("/alt-meta/00000-") && rendered.ends_with(".metadata.json"),
            "write.metadata.path is the complete directory with trailing slash stripped, got {rendered}"
        );

        let plain = MetadataLocation::for_metadata(&table_metadata("/wh/ns/t", HashMap::new()))
            .expect("default create location");
        let rendered = plain.to_string();
        assert!(
            rendered.starts_with("/wh/ns/t/metadata/00000-"),
            "absent property keeps the metadata subdirectory, got {rendered}"
        );
    }

    #[test]
    fn rebased_moves_dir_with_write_metadata_path() {
        let base = MetadataLocation::from_str(
            "/wh/ns/t/metadata/00000-a0c2e704-85a1-4368-8ed6-0b0b2a337ec9.metadata.json",
        )
        .expect("parse base");
        let metadata = table_metadata(
            "/wh/ns/t",
            HashMap::from([("write.metadata.path".to_string(), "/alt-meta".to_string())]),
        );
        let moved = base
            .with_next_version()
            .rebased(&metadata)
            .expect("rebased");
        let rendered = moved.to_string();
        assert!(
            rendered.starts_with("/alt-meta/00001-") && rendered.ends_with(".metadata.json"),
            "the next version lands under write.metadata.path, got {rendered}"
        );

        let mut unrelocated = metadata.clone();
        unrelocated.properties.clear();
        unrelocated.location = "/wh/moved".to_string();
        let next = base
            .with_next_version()
            .rebased(&unrelocated)
            .expect("rebased to new table location");
        let rendered = next.to_string();
        assert!(
            rendered.starts_with("/wh/moved/metadata/00001-"),
            "absent the property the dir follows the new metadata location, got {rendered}"
        );
    }

    #[test]
    fn rebased_keeps_pointer_dir_for_hadoop_convention() {
        let base = MetadataLocation::from_file_path("/wh/sales/orders/metadata/v2.metadata.json")
            .expect("parse hadoop base");

        let plain = table_metadata("/wh/sales/seed", HashMap::new());
        let next = base
            .with_next_version()
            .rebased(&plain)
            .expect("rebased hadoop");
        assert_eq!(
            next.to_string(),
            "/wh/sales/orders/metadata/v3.metadata.json",
            "a hadoop-convention pointer keeps its own directory, not the metadata location"
        );
    }

    #[test]
    fn rebased_refuses_write_metadata_path_for_hadoop_convention() {
        let base = MetadataLocation::from_file_path("/wh/sales/orders/metadata/v2.metadata.json")
            .expect("parse hadoop base");
        let relocated = table_metadata(
            "/wh/sales/seed",
            HashMap::from([("write.metadata.path".to_string(), "/alt-meta/".to_string())]),
        );
        let err = base
            .with_next_version()
            .rebased(&relocated)
            .expect_err("a hadoop-convention pointer carrying write.metadata.path must refuse");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert_eq!(
            err.message(),
            "Hadoop path-based tables cannot relocate metadata"
        );
    }

    #[tokio::test]
    async fn write_metadata_path_relocates_metadata_files_on_memory_catalog() {
        use std::collections::HashMap;

        use crate::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
        use crate::spec::{
            DataContentType, DataFileBuilder, DataFileFormat, Literal, NestedField, PartitionSpec,
            PrimitiveType, Schema, Struct, Transform, Type,
        };
        use crate::transaction::{ApplyTransactionAction, Transaction};
        use crate::{
            Catalog, CatalogBuilder, MetadataLocation, NamespaceIdent, TableCreation, TableIdent,
        };

        let catalog = MemoryCatalogBuilder::default()
            .load(
                "mem",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), "/wh".to_string())]),
            )
            .await
            .expect("load catalog");

        let ns = NamespaceIdent::new("ns".into());
        catalog
            .create_namespace(&ns, HashMap::new())
            .await
            .expect("namespace");

        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "cat", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("schema");
        let spec = PartitionSpec::builder(schema.clone())
            .add_partition_field("cat", "cat", Transform::Identity)
            .expect("spec field")
            .build()
            .expect("spec");

        let ident = TableIdent::new(ns, "metadata_path".into());
        let table = catalog
            .create_table(
                ident.namespace(),
                TableCreation::builder()
                    .name("metadata_path".into())
                    .schema(schema)
                    .partition_spec(spec)
                    .properties(HashMap::from([(
                        "write.metadata.path".to_string(),
                        "/alt-meta".to_string(),
                    )]))
                    .build(),
            )
            .await
            .expect("create with write.metadata.path");

        let create_location = table.metadata_location().expect("create location");
        assert!(
            create_location.starts_with("/alt-meta/00000-")
                && create_location.ends_with(".metadata.json"),
            "create metadata must land under write.metadata.path, got {create_location}"
        );
        let parsed = MetadataLocation::from_file_path(create_location)
            .expect("the catalog pointer must round-trip through MetadataLocation parsing");
        assert_eq!(parsed.version, 0);

        let mut data_file = DataFileBuilder::default();
        data_file
            .content(DataContentType::Data)
            .file_path("/wh/ns/metadata_path/data/cat=x/f1.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::string("x"))]));
        let tx = Transaction::new(&table);
        let table = tx
            .fast_append()
            .add_data_files(vec![data_file.build().expect("data file")])
            .apply(tx)
            .expect("apply")
            .commit(&catalog)
            .await
            .expect("append commit");

        let append_location = table.metadata_location().expect("append location");
        assert!(
            append_location.starts_with("/alt-meta/00001-"),
            "commit metadata must land under write.metadata.path, got {append_location}"
        );

        let snapshot = table.metadata().current_snapshot().expect("snapshot");
        assert!(
            snapshot.manifest_list().starts_with("/alt-meta/snap-"),
            "manifest list must land under write.metadata.path, got {}",
            snapshot.manifest_list()
        );
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .expect("manifest list");
        let manifest_paths: Vec<&str> = manifest_list
            .entries()
            .iter()
            .map(|entry| entry.manifest_path.as_str())
            .collect();
        assert!(!manifest_paths.is_empty());
        for path in &manifest_paths {
            assert!(
                path.starts_with("/alt-meta/") && path.ends_with("-m0.avro"),
                "manifest must land under write.metadata.path, got {path}"
            );
        }

        let tx = Transaction::new(&table);
        let committed = tx
            .update_table_properties()
            .set("k".to_string(), "v".to_string())
            .apply(tx)
            .expect("apply")
            .commit(&catalog)
            .await
            .expect("update commit");
        assert!(
            committed
                .metadata_location()
                .expect("update location")
                .starts_with("/alt-meta/00002-"),
            "the next catalog update keeps writing under write.metadata.path"
        );

        let loaded = catalog.load_table(&ident).await.expect("load back");
        let loaded_location = loaded.metadata_location().expect("loaded location");
        let parsed = MetadataLocation::from_file_path(loaded_location)
            .expect("the relocated pointer round-trips through MetadataLocation parsing");
        assert_eq!(parsed.version, 2);
        assert_eq!(
            loaded.metadata().location(),
            "/wh/ns/metadata_path",
            "the table location itself never moves"
        );
    }
}
