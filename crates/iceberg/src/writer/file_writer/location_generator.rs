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

//! This module contains the location generator and file name generator for generating path of data file.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use crate::spec::{DataFileFormat, PartitionKey, TableMetadata, TableProperties};
use crate::utils::strip_trailing_slash;
use crate::{Error, ErrorKind, Result};

/// `LocationGenerator` used to generate the location of data file.
pub trait LocationGenerator: Clone + Send + Sync + 'static {
    /// Generate an absolute path for the given file name that includes the partition path.
    ///
    /// # Arguments
    ///
    /// * `partition_key` - The partition key of the file. If None, generate a non-partitioned path.
    /// * `file_name` - The name of the file
    ///
    /// # Returns
    ///
    /// An absolute path that includes the partition path, e.g.,
    /// "/table/data/id=1/name=alice/part-00000.parquet"
    /// or non-partitioned path:
    /// "/table/data/part-00000.parquet"
    fn generate_location(&self, partition_key: Option<&PartitionKey>, file_name: &str) -> String;
}

const DEFAULT_DATA_DIR: &str = "/data";

#[derive(Clone, Debug)]
/// `DefaultLocationGenerator` used to generate the data dir location of data file.
/// The location is generated based on the table location and the data location in table properties.
pub struct DefaultLocationGenerator {
    data_location: String,
}

impl DefaultLocationGenerator {
    /// Create a new `DefaultLocationGenerator`.
    pub fn new(table_metadata: &TableMetadata) -> Result<Self> {
        let table_location = table_metadata.location();
        let prop = table_metadata.properties();
        let configured_data_location = prop
            .get(TableProperties::PROPERTY_WRITE_DATA_LOCATION)
            .or(prop.get(TableProperties::PROPERTY_WRITE_FOLDER_STORAGE_LOCATION));
        let data_location = if let Some(data_location) = configured_data_location {
            data_location.clone()
        } else {
            format!("{table_location}{DEFAULT_DATA_DIR}")
        };
        Ok(Self { data_location })
    }

    /// Create a new `DefaultLocationGenerator` with a specified data location.
    ///
    /// # Arguments
    ///
    /// * `data_location` - The data location to use for generating file locations.
    pub fn with_data_location(data_location: String) -> Self {
        Self { data_location }
    }
}

impl LocationGenerator for DefaultLocationGenerator {
    fn generate_location(&self, partition_key: Option<&PartitionKey>, file_name: &str) -> String {
        if PartitionKey::is_effectively_none(partition_key) {
            format!("{}/{}", self.data_location, file_name)
        } else {
            format!(
                "{}/{}/{}",
                self.data_location,
                partition_key.unwrap().to_path(),
                file_name
            )
        }
    }
}

fn property_as_boolean(properties: &HashMap<String, String>, key: &str, default: bool) -> bool {
    properties
        .get(key)
        .map_or(default, |value| value.eq_ignore_ascii_case("true"))
}

fn deprecated_property_error(name: &str) -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        format!(
            "Property '{name}' has been deprecated and will be removed in 2.0.0, use 'write.data.path' instead."
        ),
    )
}

fn default_data_location(
    properties: &HashMap<String, String>,
    table_location: &str,
) -> Result<String> {
    let raw = if let Some(value) = properties.get(TableProperties::PROPERTY_WRITE_DATA_LOCATION) {
        value.clone()
    } else {
        if properties.contains_key(TableProperties::PROPERTY_WRITE_FOLDER_STORAGE_LOCATION) {
            return Err(deprecated_property_error(
                TableProperties::PROPERTY_WRITE_FOLDER_STORAGE_LOCATION,
            ));
        }
        format!("{table_location}{DEFAULT_DATA_DIR}")
    };
    strip_trailing_slash(&raw).map(str::to_string)
}

fn object_store_data_location(
    properties: &HashMap<String, String>,
    table_location: &str,
) -> Result<String> {
    let raw = if let Some(value) = properties.get(TableProperties::PROPERTY_WRITE_DATA_LOCATION) {
        value.clone()
    } else {
        if properties.contains_key(TableProperties::PROPERTY_WRITE_OBJECT_STORAGE_PATH) {
            return Err(deprecated_property_error(
                TableProperties::PROPERTY_WRITE_OBJECT_STORAGE_PATH,
            ));
        }
        if properties.contains_key(TableProperties::PROPERTY_WRITE_FOLDER_STORAGE_LOCATION) {
            return Err(deprecated_property_error(
                TableProperties::PROPERTY_WRITE_FOLDER_STORAGE_LOCATION,
            ));
        }
        format!("{table_location}{DEFAULT_DATA_DIR}")
    };
    strip_trailing_slash(&raw).map(str::to_string)
}

fn path_context(table_location: &str) -> String {
    let path = match table_location.split_once("://") {
        Some((_, rest)) => rest.split_once('/').map(|(_, path)| path).unwrap_or(""),
        None => match table_location.split_once(':') {
            Some((_, rest)) => rest,
            None => table_location,
        },
    };
    let segments: Vec<&str> = path
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect();
    match segments.len() {
        0 => String::new(),
        1 => format!("/{}", segments[0]),
        _ => format!(
            "{}/{}",
            segments[segments.len() - 2],
            segments[segments.len() - 1]
        ),
    }
}

fn hash_dirs(file_name: &str) -> u32 {
    murmur3::murmur3_32(&mut file_name.as_bytes(), 0)
        .expect("murmur3_32 over a byte slice cannot fail")
        | 0x8000_0000
}

#[allow(missing_docs)]
#[derive(Clone, Debug)]
pub struct ObjectStoreLocationGenerator {
    storage_location: String,
    context: Option<String>,
    include_partition_paths: bool,
}

impl ObjectStoreLocationGenerator {
    #[allow(missing_docs)]
    pub fn new(table_location: &str, properties: &HashMap<String, String>) -> Result<Self> {
        let storage_location = object_store_data_location(properties, table_location)?;
        let context = if storage_location.starts_with(table_location) {
            None
        } else {
            Some(path_context(table_location))
        };
        let include_partition_paths = property_as_boolean(
            properties,
            TableProperties::PROPERTY_WRITE_OBJECT_STORAGE_PARTITIONED_PATHS,
            true,
        );
        Ok(Self {
            storage_location,
            context,
            include_partition_paths,
        })
    }
}

impl LocationGenerator for ObjectStoreLocationGenerator {
    fn generate_location(&self, partition_key: Option<&PartitionKey>, file_name: &str) -> String {
        let partitioned_name;
        let file_name = match partition_key {
            Some(key)
                if self.include_partition_paths
                    && !PartitionKey::is_effectively_none(Some(key)) =>
            {
                partitioned_name = format!("{}/{}", key.to_path(), file_name);
                partitioned_name.as_str()
            }
            _ => file_name,
        };
        let hash = hash_dirs(file_name);
        let mut location =
            String::with_capacity(self.storage_location.len() + file_name.len() + 24);
        location.push_str(&self.storage_location);
        write!(
            location,
            "/{:04b}/{:04b}/{:04b}/{:08b}",
            (hash >> 16) & 0xF,
            (hash >> 12) & 0xF,
            (hash >> 8) & 0xF,
            hash & 0xFF
        )
        .expect("writing to a String cannot fail");
        match &self.context {
            Some(context) => {
                location.push('/');
                location.push_str(context);
                location.push('/');
                location.push_str(file_name);
            }
            None if self.include_partition_paths => {
                location.push('/');
                location.push_str(file_name);
            }
            None => {
                location.push('-');
                location.push_str(file_name);
            }
        }
        location
    }
}

#[allow(missing_docs)]
#[derive(Clone, Debug)]
pub enum TableLocationGenerator {
    #[allow(missing_docs)]
    Default(DefaultLocationGenerator),
    #[allow(missing_docs)]
    ObjectStore(ObjectStoreLocationGenerator),
}

impl TableLocationGenerator {
    #[allow(missing_docs)]
    pub fn new(table_metadata: &TableMetadata) -> Result<Self> {
        let table_location = strip_trailing_slash(table_metadata.location())?;
        let properties = table_metadata.properties();
        if let Some(implementation) =
            properties.get(TableProperties::PROPERTY_WRITE_LOCATION_PROVIDER_IMPL)
        {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "write.location-provider.impl names the Java class '{implementation}', which cannot be instantiated here; remove the property to use the default or object-storage location provider"
                ),
            ));
        }
        if property_as_boolean(
            properties,
            TableProperties::PROPERTY_WRITE_OBJECT_STORAGE_ENABLED,
            false,
        ) {
            Ok(Self::ObjectStore(ObjectStoreLocationGenerator::new(
                table_location,
                properties,
            )?))
        } else {
            Ok(Self::Default(DefaultLocationGenerator::with_data_location(
                default_data_location(properties, table_location)?,
            )))
        }
    }
}

impl LocationGenerator for TableLocationGenerator {
    fn generate_location(&self, partition_key: Option<&PartitionKey>, file_name: &str) -> String {
        match self {
            Self::Default(generator) => generator.generate_location(partition_key, file_name),
            Self::ObjectStore(generator) => generator.generate_location(partition_key, file_name),
        }
    }
}

/// `FileNameGeneratorTrait` used to generate file name for data file. The file name can be passed to `LocationGenerator` to generate the location of the file.
pub trait FileNameGenerator: Clone + Send + Sync + 'static {
    /// Generate a file name.
    fn generate_file_name(&self) -> String;
}

/// `DefaultFileNameGenerator` used to generate file name for data file. The file name can be
/// passed to `LocationGenerator` to generate the location of the file.
/// The file name format is "{prefix}-{file_count}[-{suffix}].{file_format}".
#[derive(Clone, Debug)]
pub struct DefaultFileNameGenerator {
    prefix: String,
    suffix: String,
    format: String,
    file_count: Arc<AtomicU64>,
}

impl DefaultFileNameGenerator {
    /// Create a new `FileNameGenerator`.
    pub fn new(prefix: String, suffix: Option<String>, format: DataFileFormat) -> Self {
        let suffix = if let Some(suffix) = suffix {
            format!("-{suffix}")
        } else {
            "".to_string()
        };

        Self {
            prefix,
            suffix,
            format: format.to_string(),
            file_count: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl FileNameGenerator for DefaultFileNameGenerator {
    fn generate_file_name(&self) -> String {
        let file_id = self
            .file_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        format!(
            "{}-{:05}{}.{}",
            self.prefix, file_id, self.suffix, self.format
        )
    }
}

#[cfg(test)]
pub(crate) mod test {
    use std::collections::HashMap;
    use std::sync::Arc;

    use uuid::Uuid;

    use super::LocationGenerator;
    use crate::ErrorKind;
    use crate::spec::{
        FormatVersion, Literal, NestedField, PartitionKey, PartitionSpec, PrimitiveType, Schema,
        Struct, StructType, TableMetadata, TableProperties, Transform, Type,
    };
    use crate::writer::file_writer::location_generator::{
        DefaultLocationGenerator, FileNameGenerator, TableLocationGenerator,
    };

    #[test]
    fn test_default_location_generate() {
        let mut table_metadata = TableMetadata {
            format_version: FormatVersion::V2,
            table_uuid: Uuid::parse_str("fb072c92-a02b-11e9-ae9c-1bb7bc9eca94").unwrap(),
            location: "s3://data.db/table".to_string(),
            last_updated_ms: 1515100955770,
            last_column_id: 1,
            schemas: HashMap::new(),
            current_schema_id: 1,
            partition_specs: HashMap::new(),
            default_spec: PartitionSpec::unpartition_spec().into(),
            default_partition_type: StructType::new(vec![]),
            last_partition_id: 1000,
            default_sort_order_id: 0,
            sort_orders: HashMap::from_iter(vec![]),
            snapshots: HashMap::default(),
            current_snapshot_id: None,
            last_sequence_number: 1,
            properties: HashMap::new(),
            snapshot_log: Vec::new(),
            metadata_log: vec![],
            refs: HashMap::new(),
            statistics: HashMap::new(),
            partition_statistics: HashMap::new(),
            encryption_keys: HashMap::new(),
            next_row_id: 0,
        };

        let file_name_generator = super::DefaultFileNameGenerator::new(
            "part".to_string(),
            Some("test".to_string()),
            crate::spec::DataFileFormat::Parquet,
        );

        // test default data location
        let location_generator = super::DefaultLocationGenerator::new(&table_metadata).unwrap();
        let location =
            location_generator.generate_location(None, &file_name_generator.generate_file_name());
        assert_eq!(location, "s3://data.db/table/data/part-00000-test.parquet");

        // test custom data location
        table_metadata.properties.insert(
            TableProperties::PROPERTY_WRITE_FOLDER_STORAGE_LOCATION.to_string(),
            "s3://data.db/table/data_1".to_string(),
        );
        let location_generator = super::DefaultLocationGenerator::new(&table_metadata).unwrap();
        let location =
            location_generator.generate_location(None, &file_name_generator.generate_file_name());
        assert_eq!(
            location,
            "s3://data.db/table/data_1/part-00001-test.parquet"
        );

        table_metadata.properties.insert(
            TableProperties::PROPERTY_WRITE_DATA_LOCATION.to_string(),
            "s3://data.db/table/data_2".to_string(),
        );
        let location_generator = super::DefaultLocationGenerator::new(&table_metadata).unwrap();
        let location =
            location_generator.generate_location(None, &file_name_generator.generate_file_name());
        assert_eq!(
            location,
            "s3://data.db/table/data_2/part-00002-test.parquet"
        );

        table_metadata.properties.insert(
            TableProperties::PROPERTY_WRITE_DATA_LOCATION.to_string(),
            // invalid table location
            "s3://data.db/data_3".to_string(),
        );
        let location_generator = super::DefaultLocationGenerator::new(&table_metadata).unwrap();
        let location =
            location_generator.generate_location(None, &file_name_generator.generate_file_name());
        assert_eq!(location, "s3://data.db/data_3/part-00003-test.parquet");
    }

    #[test]
    fn test_location_generate_with_partition() {
        // Create a schema with two fields: id (int) and name (string)
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        // Create a partition spec with both fields
        let partition_spec = PartitionSpec::builder(schema.clone())
            .add_partition_field("id", "id", Transform::Identity)
            .unwrap()
            .add_partition_field("name", "name", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();

        // Create partition data with values
        let partition_data =
            Struct::from_iter([Some(Literal::int(42)), Some(Literal::string("alice"))]);

        // Create a partition key
        let partition_key = PartitionKey::new(partition_spec, schema, partition_data)
            .expect("PartitionKey::new: valid partition tuple");

        let location_gen = DefaultLocationGenerator::with_data_location("/base/path".to_string());
        let file_name = "data-00000.parquet";
        let location = location_gen.generate_location(Some(&partition_key), file_name);
        assert_eq!(location, "/base/path/id=42/name=alice/data-00000.parquet");

        // Create a table metadata for DefaultLocationGenerator
        let table_metadata = TableMetadata {
            format_version: FormatVersion::V2,
            table_uuid: Uuid::parse_str("fb072c92-a02b-11e9-ae9c-1bb7bc9eca94").unwrap(),
            location: "s3://data.db/table".to_string(),
            last_updated_ms: 1515100955770,
            last_column_id: 2,
            schemas: HashMap::new(),
            current_schema_id: 1,
            partition_specs: HashMap::new(),
            default_spec: PartitionSpec::unpartition_spec().into(),
            default_partition_type: StructType::new(vec![]),
            last_partition_id: 1000,
            default_sort_order_id: 0,
            sort_orders: HashMap::from_iter(vec![]),
            snapshots: HashMap::default(),
            current_snapshot_id: None,
            last_sequence_number: 1,
            properties: HashMap::new(),
            snapshot_log: Vec::new(),
            metadata_log: vec![],
            refs: HashMap::new(),
            statistics: HashMap::new(),
            partition_statistics: HashMap::new(),
            encryption_keys: HashMap::new(),
            next_row_id: 0,
        };

        // Test with DefaultLocationGenerator
        let default_location_gen = super::DefaultLocationGenerator::new(&table_metadata).unwrap();
        let location = default_location_gen.generate_location(Some(&partition_key), file_name);
        assert_eq!(
            location,
            "s3://data.db/table/data/id=42/name=alice/data-00000.parquet"
        );
    }

    fn props(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn table_metadata(location: &str, properties: HashMap<String, String>) -> TableMetadata {
        TableMetadata {
            format_version: FormatVersion::V2,
            table_uuid: Uuid::parse_str("fb072c92-a02b-11e9-ae9c-1bb7bc9eca94").unwrap(),
            location: location.to_string(),
            last_updated_ms: 1515100955770,
            last_column_id: 1,
            schemas: HashMap::new(),
            current_schema_id: 1,
            partition_specs: HashMap::new(),
            default_spec: PartitionSpec::unpartition_spec().into(),
            default_partition_type: StructType::new(vec![]),
            last_partition_id: 1000,
            default_sort_order_id: 0,
            sort_orders: HashMap::from_iter(vec![]),
            snapshots: HashMap::default(),
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

    fn cat_partition_key(value: &str) -> PartitionKey {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "cat", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );
        let partition_spec = PartitionSpec::builder(schema.clone())
            .add_partition_field("cat", "cat", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();
        PartitionKey::new(
            partition_spec,
            schema,
            Struct::from_iter([Some(Literal::string(value))]),
        )
        .expect("PartitionKey::new: valid cat value")
    }

    #[test]
    fn table_location_generator_default_unpartitioned() {
        let metadata = table_metadata("s3://wh/ns/t", HashMap::new());
        let generator = TableLocationGenerator::new(&metadata).unwrap();
        assert_eq!(
            generator.generate_location(None, "f.parquet"),
            "s3://wh/ns/t/data/f.parquet"
        );
        assert_eq!(
            generator.generate_location(Some(&cat_partition_key("x")), "f.parquet"),
            "s3://wh/ns/t/data/cat=x/f.parquet"
        );
    }

    #[test]
    fn table_location_generator_object_storage_unpartitioned() {
        let metadata = table_metadata(
            "s3://wh/ns/l_object_storage",
            props(&[("write.object-storage.enabled", "true")]),
        );
        let generator = TableLocationGenerator::new(&metadata).unwrap();
        assert_eq!(
            generator.generate_location(
                None,
                "00000-28-3812449f-1cb2-488c-9cc3-8f96668c44cd-0-00001.parquet"
            ),
            "s3://wh/ns/l_object_storage/data/1110/1011/1110/01000111/00000-28-3812449f-1cb2-488c-9cc3-8f96668c44cd-0-00001.parquet"
        );
    }

    #[test]
    fn table_location_generator_object_storage_partitioned() {
        let metadata = table_metadata(
            "s3://wh/ns/l_object_storage_p",
            props(&[("write.object-storage.enabled", "true")]),
        );
        let generator = TableLocationGenerator::new(&metadata).unwrap();
        assert_eq!(
            generator.generate_location(
                Some(&cat_partition_key("x")),
                "00000-34-96eea29b-37d5-44d2-8f8e-790fbe133349-0-00001.parquet"
            ),
            "s3://wh/ns/l_object_storage_p/data/0001/1111/1111/10111111/cat=x/00000-34-96eea29b-37d5-44d2-8f8e-790fbe133349-0-00001.parquet"
        );
        assert_eq!(
            generator.generate_location(
                Some(&cat_partition_key("y")),
                "00000-34-96eea29b-37d5-44d2-8f8e-790fbe133349-0-00002.parquet"
            ),
            "s3://wh/ns/l_object_storage_p/data/1001/1011/1111/11111010/cat=y/00000-34-96eea29b-37d5-44d2-8f8e-790fbe133349-0-00002.parquet"
        );
    }

    #[test]
    fn table_location_generator_object_storage_unpartitioned_paths() {
        let metadata = table_metadata(
            "s3://wh/ns/l_object_storage_unpartitioned_paths",
            props(&[
                ("write.object-storage.enabled", "true"),
                ("write.object-storage.partitioned-paths", "false"),
            ]),
        );
        let generator = TableLocationGenerator::new(&metadata).unwrap();
        assert_eq!(
            generator.generate_location(
                None,
                "00000-39-afc298d7-6274-47c4-a229-c4bd98b470f7-0-00001.parquet"
            ),
            "s3://wh/ns/l_object_storage_unpartitioned_paths/data/0001/1001/0110/10000001-00000-39-afc298d7-6274-47c4-a229-c4bd98b470f7-0-00001.parquet"
        );
    }

    #[test]
    fn table_location_generator_object_storage_unpartitioned_paths_partitioned() {
        let metadata = table_metadata(
            "s3://wh/ns/l_object_storage_unpartitioned_paths_p",
            props(&[
                ("write.object-storage.enabled", "true"),
                ("write.object-storage.partitioned-paths", "false"),
            ]),
        );
        let generator = TableLocationGenerator::new(&metadata).unwrap();
        assert_eq!(
            generator.generate_location(
                Some(&cat_partition_key("x")),
                "00000-45-9205b7af-4125-4711-912d-a933510ba235-0-00001.parquet"
            ),
            "s3://wh/ns/l_object_storage_unpartitioned_paths_p/data/0110/1101/0111/11111000-00000-45-9205b7af-4125-4711-912d-a933510ba235-0-00001.parquet"
        );
        assert_eq!(
            generator.generate_location(
                Some(&cat_partition_key("y")),
                "00000-45-9205b7af-4125-4711-912d-a933510ba235-0-00002.parquet"
            ),
            "s3://wh/ns/l_object_storage_unpartitioned_paths_p/data/1010/0011/1100/11110001-00000-45-9205b7af-4125-4711-912d-a933510ba235-0-00002.parquet"
        );
    }

    #[test]
    fn table_location_generator_object_storage_data_path_unpartitioned() {
        let metadata = table_metadata(
            "s3://wh/ns/l_object_storage_data_path",
            props(&[
                ("write.object-storage.enabled", "true"),
                ("write.data.path", "s3://wh/alt-data"),
            ]),
        );
        let generator = TableLocationGenerator::new(&metadata).unwrap();
        assert_eq!(
            generator.generate_location(
                None,
                "00000-50-661c5ab6-de30-4594-858b-7f8d5fbc98e1-0-00001.parquet"
            ),
            "s3://wh/alt-data/1111/0011/1100/11110011/ns/l_object_storage_data_path/00000-50-661c5ab6-de30-4594-858b-7f8d5fbc98e1-0-00001.parquet"
        );
    }

    #[test]
    fn table_location_generator_object_storage_data_path_partitioned() {
        let metadata = table_metadata(
            "s3://wh/ns/l_object_storage_data_path_p",
            props(&[
                ("write.object-storage.enabled", "true"),
                ("write.data.path", "s3://wh/alt-data"),
            ]),
        );
        let generator = TableLocationGenerator::new(&metadata).unwrap();
        assert_eq!(
            generator.generate_location(
                Some(&cat_partition_key("x")),
                "00000-56-c7fb03ac-63e3-46aa-8da7-1608d002a1a4-0-00001.parquet"
            ),
            "s3://wh/alt-data/0101/1111/0100/00101010/ns/l_object_storage_data_path_p/cat=x/00000-56-c7fb03ac-63e3-46aa-8da7-1608d002a1a4-0-00001.parquet"
        );
        assert_eq!(
            generator.generate_location(
                Some(&cat_partition_key("y")),
                "00000-56-c7fb03ac-63e3-46aa-8da7-1608d002a1a4-0-00002.parquet"
            ),
            "s3://wh/alt-data/0110/0010/1100/10100111/ns/l_object_storage_data_path_p/cat=y/00000-56-c7fb03ac-63e3-46aa-8da7-1608d002a1a4-0-00002.parquet"
        );
    }

    #[test]
    fn table_location_generator_data_path_alone() {
        let metadata = table_metadata(
            "s3://wh/ns/l_data_path",
            props(&[("write.data.path", "s3://wh/alt-data2")]),
        );
        let generator = TableLocationGenerator::new(&metadata).unwrap();
        assert_eq!(
            generator.generate_location(
                None,
                "00000-61-d9a9dc5f-d7f2-4a5f-8240-2c8ab7e9f00c-0-00001.parquet"
            ),
            "s3://wh/alt-data2/00000-61-d9a9dc5f-d7f2-4a5f-8240-2c8ab7e9f00c-0-00001.parquet"
        );
        assert_eq!(
            generator.generate_location(
                Some(&cat_partition_key("x")),
                "00000-61-d9a9dc5f-d7f2-4a5f-8240-2c8ab7e9f00c-0-00001.parquet"
            ),
            "s3://wh/alt-data2/cat=x/00000-61-d9a9dc5f-d7f2-4a5f-8240-2c8ab7e9f00c-0-00001.parquet"
        );
    }

    #[test]
    fn table_location_generator_refuses_java_provider_impl() {
        for enabled in ["true", "false"] {
            let metadata = table_metadata(
                "s3://wh/ns/t",
                props(&[
                    ("write.location-provider.impl", "com.example.CustomProvider"),
                    ("write.object-storage.enabled", enabled),
                ]),
            );
            let error = TableLocationGenerator::new(&metadata).unwrap_err();
            assert_eq!(
                error.kind(),
                ErrorKind::FeatureUnsupported,
                "write.location-provider.impl must fail loudly (enabled={enabled})"
            );
        }
    }

    #[test]
    fn table_location_generator_rejects_deprecated_folder_storage() {
        let metadata = table_metadata(
            "s3://wh/ns/t",
            props(&[("write.folder-storage.path", "s3://wh/old")]),
        );
        assert!(
            TableLocationGenerator::new(&metadata).is_err(),
            "write.folder-storage.path alone must be rejected"
        );

        let metadata = table_metadata(
            "s3://wh/ns/t",
            props(&[
                ("write.folder-storage.path", "s3://wh/old"),
                ("write.object-storage.enabled", "true"),
            ]),
        );
        assert!(
            TableLocationGenerator::new(&metadata).is_err(),
            "write.folder-storage.path must be rejected on the object-store path"
        );

        let metadata = table_metadata(
            "s3://wh/ns/t",
            props(&[
                ("write.data.path", "s3://wh/new"),
                ("write.folder-storage.path", "s3://wh/old"),
            ]),
        );
        let generator = TableLocationGenerator::new(&metadata).unwrap();
        assert_eq!(
            generator.generate_location(None, "f.parquet"),
            "s3://wh/new/f.parquet",
            "write.data.path short-circuits the deprecated check, as Java does"
        );
    }

    #[test]
    fn table_location_generator_rejects_deprecated_object_storage_path() {
        let metadata = table_metadata(
            "s3://wh/ns/t",
            props(&[
                ("write.object-storage.enabled", "true"),
                ("write.object-storage.path", "s3://wh/old"),
            ]),
        );
        assert!(
            TableLocationGenerator::new(&metadata).is_err(),
            "write.object-storage.path must be rejected when object storage is selected"
        );

        let metadata = table_metadata(
            "s3://wh/ns/t",
            props(&[("write.object-storage.path", "s3://wh/old")]),
        );
        let generator = TableLocationGenerator::new(&metadata).unwrap();
        assert_eq!(
            generator.generate_location(None, "f.parquet"),
            "s3://wh/ns/t/data/f.parquet",
            "the default provider never checks write.object-storage.path, as Java does"
        );

        let metadata = table_metadata(
            "s3://wh/ns/t",
            props(&[
                ("write.object-storage.enabled", "true"),
                ("write.data.path", "s3://wh/new"),
                ("write.object-storage.path", "s3://wh/old"),
            ]),
        );
        let generator = TableLocationGenerator::new(&metadata).unwrap();
        assert_eq!(
            generator.generate_location(None, "f.parquet"),
            "s3://wh/new/0111/1111/1110/11001100/ns/t/f.parquet",
            "write.data.path short-circuits the deprecated check, as Java does"
        );
    }

    #[test]
    fn table_location_generator_object_storage_context_bucket_root_parent() {
        for location in ["s3://bucket/mytable", "s3://bucket/mytable/"] {
            let metadata = table_metadata(
                location,
                props(&[
                    ("write.object-storage.enabled", "true"),
                    ("write.data.path", "s3://alt-data"),
                ]),
            );
            let generator = TableLocationGenerator::new(&metadata).unwrap();
            assert_eq!(
                generator.generate_location(None, "f.parquet"),
                "s3://alt-data/0111/1111/1110/11001100//mytable/f.parquet",
                "Hadoop Path({location}) has an empty-named parent, so the context keeps a leading slash"
            );
        }
    }

    #[test]
    fn table_location_generator_object_storage_context_single_segment_relative() {
        let metadata = table_metadata(
            "mytable",
            props(&[
                ("write.object-storage.enabled", "true"),
                ("write.data.path", "s3://alt-data"),
            ]),
        );
        let generator = TableLocationGenerator::new(&metadata).unwrap();
        assert_eq!(
            generator.generate_location(None, "f.parquet"),
            "s3://alt-data/0111/1111/1110/11001100//mytable/f.parquet",
            "Hadoop Path(mytable) has the empty path as parent, so the context keeps a leading slash"
        );
    }

    #[test]
    fn table_location_generator_object_storage_context_bucket_only() {
        let metadata = table_metadata(
            "s3://bucket",
            props(&[
                ("write.object-storage.enabled", "true"),
                ("write.data.path", "s3://alt-data"),
            ]),
        );
        let generator = TableLocationGenerator::new(&metadata).unwrap();
        assert_eq!(
            generator.generate_location(None, "f.parquet"),
            "s3://alt-data/0111/1111/1110/11001100//f.parquet",
            "Hadoop Path(s3://bucket) has a null parent and an empty name, so the context is empty"
        );
    }

    #[test]
    fn table_location_generator_object_storage_context_suppressed() {
        let metadata = table_metadata(
            "s3://wh/ns/t",
            props(&[
                ("write.object-storage.enabled", "true"),
                ("write.data.path", "s3://wh/ns/t/data2/"),
            ]),
        );
        let generator = TableLocationGenerator::new(&metadata).unwrap();
        assert_eq!(
            generator.generate_location(None, "f.parquet"),
            "s3://wh/ns/t/data2/0111/1111/1110/11001100/f.parquet",
            "a storage location under the table location carries no path context"
        );
    }
}
