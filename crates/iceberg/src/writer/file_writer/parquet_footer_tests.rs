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

use arrow_array::builder::{MapBuilder, PrimitiveBuilder, StringBuilder};
use arrow_array::types::Int64Type;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Decimal128Array, FixedSizeBinaryArray,
    Float32Array, Float64Array, Int32Array, Int64Array, LargeBinaryArray, ListArray, MapArray,
    RecordBatch, StringArray, StructArray, Time64MicrosecondArray, TimestampMicrosecondArray,
    TimestampNanosecondArray,
};
use arrow_schema::{DataType, SchemaRef as ArrowSchemaRef};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use tempfile::TempDir;
use uuid::Uuid;

use super::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator, FileNameGenerator, LocationGenerator,
};
use super::rolling_writer::RollingFileWriterBuilder;
use super::{
    FileWriter, FileWriterBuilder, ParquetWriterBuilder, parquet_compression_from_properties,
};
use crate::Result;
use crate::arrow::{UTC_TIME_ZONE, schema_to_arrow_schema};
use crate::io::FileIO;
use crate::spec::{
    DataFileFormat, ListType, Literal, MapType, NestedField, NestedFieldRef, PartitionKey,
    PartitionSpec, PrimitiveType, Schema, Struct, Transform, Type, UnboundPartitionField,
};
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};

fn repark_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "p", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(3, "v", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .unwrap()
}

fn all_types_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::optional(1, "boolean", Type::Primitive(PrimitiveType::Boolean)).into(),
            NestedField::optional(2, "int", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(3, "long", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(4, "float", Type::Primitive(PrimitiveType::Float)).into(),
            NestedField::optional(5, "double", Type::Primitive(PrimitiveType::Double)).into(),
            NestedField::optional(
                6,
                "decimal",
                Type::Primitive(PrimitiveType::Decimal {
                    precision: 10,
                    scale: 5,
                }),
            )
            .into(),
            NestedField::optional(7, "date", Type::Primitive(PrimitiveType::Date)).into(),
            NestedField::optional(8, "time", Type::Primitive(PrimitiveType::Time)).into(),
            NestedField::optional(9, "timestamp", Type::Primitive(PrimitiveType::Timestamp)).into(),
            NestedField::optional(
                10,
                "timestamptz",
                Type::Primitive(PrimitiveType::Timestamptz),
            )
            .into(),
            NestedField::optional(
                11,
                "timestamp_ns",
                Type::Primitive(PrimitiveType::TimestampNs),
            )
            .into(),
            NestedField::optional(
                12,
                "timestamptz_ns",
                Type::Primitive(PrimitiveType::TimestamptzNs),
            )
            .into(),
            NestedField::optional(13, "string", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(14, "uuid", Type::Primitive(PrimitiveType::Uuid)).into(),
            NestedField::optional(15, "fixed", Type::Primitive(PrimitiveType::Fixed(10))).into(),
            NestedField::optional(16, "binary", Type::Primitive(PrimitiveType::Binary)).into(),
            NestedField::optional(
                17,
                "list",
                Type::List(ListType::new(NestedFieldRef::new(NestedField::optional(
                    18,
                    "element",
                    Type::Primitive(PrimitiveType::Long),
                )))),
            )
            .into(),
            NestedField::optional(
                19,
                "map",
                Type::Map(MapType::new(
                    NestedFieldRef::new(NestedField::required(
                        20,
                        "key",
                        Type::Primitive(PrimitiveType::String),
                    )),
                    NestedFieldRef::new(NestedField::optional(
                        21,
                        "value",
                        Type::Primitive(PrimitiveType::Long),
                    )),
                )),
            )
            .into(),
            NestedField::optional(
                22,
                "struct",
                Type::Struct(crate::spec::StructType::new(vec![
                    NestedFieldRef::new(NestedField::optional(
                        23,
                        "a",
                        Type::Primitive(PrimitiveType::Int),
                    )),
                    NestedFieldRef::new(NestedField::optional(
                        24,
                        "b",
                        Type::Primitive(PrimitiveType::String),
                    )),
                ])),
            )
            .into(),
        ])
        .build()
        .unwrap()
}

fn all_types_batch(arrow_schema: &ArrowSchemaRef) -> RecordBatch {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(BooleanArray::from(vec![Some(true), None])) as ArrayRef,
        Arc::new(Int32Array::from(vec![Some(1), None])) as ArrayRef,
        Arc::new(Int64Array::from(vec![Some(1), None])) as ArrayRef,
        Arc::new(Float32Array::from(vec![Some(0.5), None])) as ArrayRef,
        Arc::new(Float64Array::from(vec![Some(0.5), None])) as ArrayRef,
        Arc::new(
            Decimal128Array::from(vec![Some(1), None])
                .with_precision_and_scale(10, 5)
                .unwrap(),
        ) as ArrayRef,
        Arc::new(Date32Array::from(vec![Some(0), None])) as ArrayRef,
        Arc::new(Time64MicrosecondArray::from(vec![Some(0), None])) as ArrayRef,
        Arc::new(TimestampMicrosecondArray::from(vec![Some(0), None])) as ArrayRef,
        Arc::new(TimestampMicrosecondArray::from(vec![Some(0), None]).with_timezone(UTC_TIME_ZONE))
            as ArrayRef,
        Arc::new(TimestampNanosecondArray::from(vec![Some(0), None])) as ArrayRef,
        Arc::new(TimestampNanosecondArray::from(vec![Some(0), None]).with_timezone(UTC_TIME_ZONE))
            as ArrayRef,
        Arc::new(StringArray::from(vec![Some("a"), None])) as ArrayRef,
        Arc::new(
            FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                vec![Some(Uuid::from_u128(0).as_bytes().to_vec()), None].into_iter(),
                16,
            )
            .unwrap(),
        ) as ArrayRef,
        Arc::new(
            FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                vec![Some(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), None].into_iter(),
                10,
            )
            .unwrap(),
        ) as ArrayRef,
        Arc::new(LargeBinaryArray::from_opt_vec(vec![Some(b"one"), None])) as ArrayRef,
        Arc::new({
            let parts = ListArray::from_iter_primitive::<Int64Type, _, _>(
                (0..2).map(|_| Some(vec![Some(7)])),
            )
            .into_parts();
            let field = match arrow_schema.field(16).data_type() {
                DataType::List(field) => field.clone(),
                _ => unreachable!(),
            };
            ListArray::new(field, parts.1, parts.2, parts.3)
        }) as ArrayRef,
        Arc::new({
            let mut builder = MapBuilder::new(
                None,
                StringBuilder::new(),
                PrimitiveBuilder::<Int64Type>::new(),
            );
            for _ in 0..2 {
                builder.keys().append_value("k");
                builder.values().append_value(9);
                builder.append(true).unwrap();
            }
            let (_, offsets, entries, nulls, ordered) = builder.finish().into_parts();
            let map_field = match arrow_schema.field(17).data_type() {
                DataType::Map(field, _) => field.clone(),
                _ => unreachable!(),
            };
            let entry_fields = match map_field.data_type() {
                DataType::Struct(fields) => fields.clone(),
                _ => unreachable!(),
            };
            let entries = StructArray::new(
                entry_fields,
                entries.columns().to_vec(),
                entries.nulls().cloned(),
            );
            MapArray::new(map_field, offsets, entries, nulls, ordered)
        }) as ArrayRef,
        Arc::new({
            let struct_fields = match arrow_schema.field(18).data_type() {
                DataType::Struct(fields) => fields.clone(),
                _ => unreachable!(),
            };
            StructArray::new(
                struct_fields,
                vec![
                    Arc::new(Int32Array::from(vec![Some(5), None])) as ArrayRef,
                    Arc::new(StringArray::from(vec![Some("s"), None])) as ArrayRef,
                ],
                None,
            )
        }) as ArrayRef,
    ];
    RecordBatch::try_new(arrow_schema.clone(), columns).unwrap()
}

fn footer_key_values(path: &str) -> Vec<KeyValue> {
    let file = std::fs::File::open(path).unwrap();
    let reader = SerializedFileReader::new(file).unwrap();
    reader
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .cloned()
        .unwrap_or_default()
}

async fn write_repark_file(dir: &TempDir, id_start: i64) -> (String, u64) {
    let file_io = FileIO::new_with_fs();
    let location_gen =
        DefaultLocationGenerator::with_data_location(dir.path().to_str().unwrap().to_string());
    let file_name_gen =
        DefaultFileNameGenerator::new("probe".to_string(), None, DataFileFormat::Parquet);
    let schema = Arc::new(repark_schema());
    let compression = parquet_compression_from_properties(&HashMap::new()).unwrap();
    let props = WriterProperties::builder()
        .set_compression(compression)
        .build();
    let parquet_builder = ParquetWriterBuilder::new(props, schema.clone());
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        file_io,
        location_gen,
        file_name_gen,
    );
    let spec = PartitionSpec::builder(repark_schema())
        .with_spec_id(0)
        .add_unbound_field(
            UnboundPartitionField::builder()
                .source_id(2)
                .name("p".to_string())
                .transform(Transform::Identity)
                .build(),
        )
        .unwrap()
        .build()
        .unwrap();
    let key = PartitionKey::new(
        spec,
        schema.clone(),
        Struct::from_iter([Some(Literal::int(1))]),
    )
    .unwrap();
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(Some(key))
        .await
        .unwrap();
    let arrow_schema: ArrowSchemaRef = Arc::new(schema_to_arrow_schema(&schema).unwrap());
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from_iter_values(id_start..id_start + 50)) as ArrayRef,
        Arc::new(Int32Array::from_value(1, 50)) as ArrayRef,
        Arc::new(StringArray::from(vec!["xxxxxxxxxxxxxxxxxxxx"; 50])) as ArrayRef,
    ])
    .unwrap();
    writer.write(batch).await.unwrap();
    let files = writer.close().await.unwrap();
    assert_eq!(files.len(), 1);
    (
        files[0].file_path().to_string(),
        files[0].file_size_in_bytes(),
    )
}

#[tokio::test]
async fn footer_key_values_match_java() -> Result<()> {
    let dir = TempDir::new().unwrap();
    let (path, _) = write_repark_file(&dir, 200).await;
    let key_values = footer_key_values(&path);
    let keys: Vec<&str> = key_values.iter().map(|kv| kv.key.as_str()).collect();
    assert_eq!(
        keys,
        vec![super::parquet_footer::ICEBERG_SCHEMA_META_KEY],
        "the parquet footer must carry exactly the key-values Java's writer emits"
    );
    let parsed: Schema = serde_json::from_str(key_values[0].value.as_deref().unwrap()).unwrap();
    assert_eq!(
        parsed.as_struct().fields(),
        repark_schema().as_struct().fields()
    );
    Ok(())
}

#[tokio::test]
async fn footer_iceberg_schema_round_trips_all_types() -> Result<()> {
    let dir = TempDir::new().unwrap();
    let file_io = FileIO::new_with_fs();
    let location_gen =
        DefaultLocationGenerator::with_data_location(dir.path().to_str().unwrap().to_string());
    let file_name_gen =
        DefaultFileNameGenerator::new("all".to_string(), None, DataFileFormat::Parquet);
    let schema = Arc::new(all_types_schema());
    let output_file = file_io
        .new_output(location_gen.generate_location(None, &file_name_gen.generate_file_name()))?;
    let path = output_file.location().to_string();
    let mut writer = ParquetWriterBuilder::new(WriterProperties::builder().build(), schema.clone())
        .build(output_file)
        .await?;
    let arrow_schema: ArrowSchemaRef = Arc::new(schema.as_ref().try_into().unwrap());
    let batch = all_types_batch(&arrow_schema);
    writer.write(&batch).await?;
    let files = writer.close().await?;
    assert_eq!(files.len(), 1);
    let key_values = footer_key_values(&path);
    assert_eq!(key_values.len(), 1);
    assert_eq!(
        key_values[0].key,
        super::parquet_footer::ICEBERG_SCHEMA_META_KEY
    );
    let parsed: Schema = serde_json::from_str(key_values[0].value.as_deref().unwrap()).unwrap();
    assert_eq!(
        parsed.as_struct().fields(),
        all_types_schema().as_struct().fields()
    );
    Ok(())
}

#[tokio::test]
async fn all_types_values_round_trip_without_arrow_schema() -> Result<()> {
    let dir = TempDir::new().unwrap();
    let file_io = FileIO::new_with_fs();
    let location_gen =
        DefaultLocationGenerator::with_data_location(dir.path().to_str().unwrap().to_string());
    let file_name_gen =
        DefaultFileNameGenerator::new("all".to_string(), None, DataFileFormat::Parquet);
    let schema = Arc::new(all_types_schema());
    let output_file = file_io
        .new_output(location_gen.generate_location(None, &file_name_gen.generate_file_name()))?;
    let mut writer = ParquetWriterBuilder::new(WriterProperties::builder().build(), schema)
        .build(output_file)
        .await?;
    let arrow_schema: ArrowSchemaRef = Arc::new((&all_types_schema()).try_into().unwrap());
    let batch = all_types_batch(&arrow_schema);
    writer.write(&batch).await?;
    let files = writer.close().await?;
    assert_eq!(files.len(), 1);
    let data_file = files
        .into_iter()
        .next()
        .unwrap()
        .content(crate::spec::DataContentType::Data)
        .partition(Struct::empty())
        .partition_spec_id(0)
        .build()
        .unwrap();
    crate::writer::tests::check_parquet_data_file(&file_io, &data_file, &batch).await;
    Ok(())
}

#[tokio::test]
async fn repark_shape_file_size_matches_java_scale() -> Result<()> {
    let dir = TempDir::new().unwrap();
    let (_, size) = write_repark_file(&dir, 200).await;
    assert!(
        size < 1300,
        "the 50-row RePark shape must land near the Java writer's ~1150 B, not {size} B"
    );
    Ok(())
}

#[tokio::test]
async fn rolled_files_from_one_builder_have_identical_bytes() -> Result<()> {
    let dir = TempDir::new().unwrap();
    let file_io = FileIO::new_with_fs();
    let location_gen =
        DefaultLocationGenerator::with_data_location(dir.path().to_str().unwrap().to_string());
    let file_name_gen =
        DefaultFileNameGenerator::new("rolled".to_string(), None, DataFileFormat::Parquet);
    let schema = Arc::new(repark_schema());
    let builder = ParquetWriterBuilder::new(WriterProperties::builder().build(), schema.clone());
    let arrow_schema: ArrowSchemaRef = Arc::new(schema_to_arrow_schema(&schema).unwrap());
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from_iter_values(200..250)) as ArrayRef,
        Arc::new(Int32Array::from_value(1, 50)) as ArrayRef,
        Arc::new(StringArray::from(vec!["xxxxxxxxxxxxxxxxxxxx"; 50])) as ArrayRef,
    ])
    .unwrap();
    let mut paths = vec![];
    for _ in 0..2 {
        let output_file = file_io.new_output(
            location_gen.generate_location(None, &file_name_gen.generate_file_name()),
        )?;
        let path = output_file.location().to_string();
        let mut writer = builder.build(output_file).await?;
        writer.write(&batch).await?;
        writer.close().await?;
        paths.push(path);
    }
    assert_eq!(
        std::fs::read(&paths[0]).unwrap(),
        std::fs::read(&paths[1]).unwrap(),
        "files rolled from one writer builder must be byte-identical"
    );
    Ok(())
}
