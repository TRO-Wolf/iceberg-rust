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

use std::fmt::Display;
use std::str::FromStr;

use uuid::Uuid;

use crate::{Error, ErrorKind, Result};

/// Helper for parsing a metadata JSON location under `<table>/metadata/`.
///
/// Hive/REST names are `<version>-<uuid>.metadata.json`. Hadoop names are
/// `v<version>.metadata.json` (Java `HadoopTableOperations`, row R167).
#[derive(Clone, Debug, PartialEq)]
pub struct MetadataLocation {
    table_location: String,
    version: i32,
    /// `None` is the Hadoop convention. A uuid is the Hive/REST convention.
    id: Option<Uuid>,
}

impl MetadataLocation {
    /// Creates a completely new metadata location starting at version 0.
    /// Only used for creating a new table. For updates, see `with_next_version`.
    pub fn new_with_table_location(table_location: impl ToString) -> Self {
        Self {
            table_location: table_location.to_string(),
            version: 0,
            id: Some(Uuid::new_v4()),
        }
    }

    /// Creates a new metadata location for an updated metadata file.
    ///
    /// A Hadoop pointer stays Hadoop: `vN` becomes `v(N+1)`. Hive/REST gets a new uuid.
    /// The next Hadoop file is uncompressed `.metadata.json` even if the current file was gzip.
    pub fn with_next_version(&self) -> Self {
        Self {
            table_location: self.table_location.clone(),
            version: self.version.wrapping_add(1),
            id: self.id.map(|_| Uuid::new_v4()),
        }
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
                "{}/metadata/{:0>5}-{}.metadata.json",
                self.table_location, self.version, id
            ),
            None => write!(
                f,
                "{}/metadata/v{}.metadata.json",
                self.table_location, self.version
            ),
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

        let prefix = Self::parse_metadata_path_prefix(path)?;
        let (version, id) = Self::parse_file_name(file_name)?;

        Ok(MetadataLocation {
            table_location: prefix,
            version,
            id,
        })
    }
}

#[cfg(test)]
mod test {
    use std::str::FromStr;

    use uuid::Uuid;

    use crate::MetadataLocation;

    #[test]
    fn test_metadata_location_from_string() {
        let test_cases = vec![
            // No prefix
            (
                "/metadata/1234567-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json",
                Ok(MetadataLocation {
                    table_location: "".to_string(),
                    version: 1234567,
                    id: Some(Uuid::from_str("2cd22b57-5127-4198-92ba-e4e67c79821b").unwrap()),
                }),
            ),
            // Some prefix
            (
                "/abc/metadata/1234567-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json",
                Ok(MetadataLocation {
                    table_location: "/abc".to_string(),
                    version: 1234567,
                    id: Some(Uuid::from_str("2cd22b57-5127-4198-92ba-e4e67c79821b").unwrap()),
                }),
            ),
            // Longer prefix
            (
                "/abc/def/metadata/1234567-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json",
                Ok(MetadataLocation {
                    table_location: "/abc/def".to_string(),
                    version: 1234567,
                    id: Some(Uuid::from_str("2cd22b57-5127-4198-92ba-e4e67c79821b").unwrap()),
                }),
            ),
            // Prefix with special characters
            (
                "https://127.0.0.1/metadata/1234567-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json",
                Ok(MetadataLocation {
                    table_location: "https://127.0.0.1".to_string(),
                    version: 1234567,
                    id: Some(Uuid::from_str("2cd22b57-5127-4198-92ba-e4e67c79821b").unwrap()),
                }),
            ),
            // Another id
            (
                "/abc/metadata/1234567-81056704-ce5b-41c4-bb83-eb6408081af6.metadata.json",
                Ok(MetadataLocation {
                    table_location: "/abc".to_string(),
                    version: 1234567,
                    id: Some(Uuid::from_str("81056704-ce5b-41c4-bb83-eb6408081af6").unwrap()),
                }),
            ),
            // Version 0
            (
                "/abc/metadata/00000-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json",
                Ok(MetadataLocation {
                    table_location: "/abc".to_string(),
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
                    table_location: "/abc".to_string(),
                    version: 3,
                    id: None,
                }),
            ),
            (
                "/abc/metadata/v0.metadata.json",
                Ok(MetadataLocation {
                    table_location: "/abc".to_string(),
                    version: 0,
                    id: None,
                }),
            ),
            (
                "/abc/metadata/v12.metadata.json",
                Ok(MetadataLocation {
                    table_location: "/abc".to_string(),
                    version: 12,
                    id: None,
                }),
            ),
            (
                "/abc/metadata/v00003.metadata.json",
                Ok(MetadataLocation {
                    table_location: "/abc".to_string(),
                    version: 3,
                    id: None,
                }),
            ),
            (
                "/abc/metadata/00003-2cd22b57-5127-4198-92ba-e4e67c79821b.gz.metadata.json",
                Ok(MetadataLocation {
                    table_location: "/abc".to_string(),
                    version: 3,
                    id: Some(Uuid::from_str("2cd22b57-5127-4198-92ba-e4e67c79821b").unwrap()),
                }),
            ),
            (
                "/abc/metadata/00003-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json.gz",
                Ok(MetadataLocation {
                    table_location: "/abc".to_string(),
                    version: 3,
                    id: Some(Uuid::from_str("2cd22b57-5127-4198-92ba-e4e67c79821b").unwrap()),
                }),
            ),
            (
                "/abc/metadata/v3.gz.metadata.json",
                Ok(MetadataLocation {
                    table_location: "/abc".to_string(),
                    version: 3,
                    id: None,
                }),
            ),
            (
                "/abc/metadata/v3.metadata.json.gz",
                Ok(MetadataLocation {
                    table_location: "/abc".to_string(),
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
            assert_eq!(next.table_location, input.table_location);
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
        assert_eq!(next.table_location, "/wh/t");
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
}
