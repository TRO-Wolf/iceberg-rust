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

use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::sync::Arc;

use apache_avro::Reader as AvroReader;
use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use serde_json::Value as JsonValue;
use tempfile::TempDir;

use super::schema_to_avro_schema;
use crate::arrow::avro_reader::read_avro_data_bytes;
use crate::arrow::schema_to_arrow_schema;
use crate::avro::ocf::{ocf_json_parse_count, repair_avro_container};
use crate::expr::Reference;
use crate::io::{FileIO, LocalFsStorageFactory};
use crate::memory::MemoryCatalogBuilder;
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, FormatVersion, Literal,
    Manifest, ManifestEntry, ManifestStatus, ManifestWriterBuilder, NestedField, PartitionSpec,
    PrimitiveType, Schema, Struct, StructType, Transform, Type, read_data_files_from_avro,
    write_data_files_to_avro,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::writer::file_writer::{
    AvroWriterBuilder, FileWriter, FileWriterBuilder, ParquetWriterBuilder,
};
use crate::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};

const CASES: &[(&str, &str, Option<&str>)] = &[
    ("my col", "my_x20col", Some("my col")),
    ("my col_bucket", "my_x20col_bucket", Some("my col_bucket")),
    ("1st", "_1st", Some("1st")),
    ("a-b", "a_x2Db", Some("a-b")),
    ("a-b_trunc", "a_x2Db_trunc", Some("a-b_trunc")),
    ("a.b", "a_x2Eb", Some("a.b")),
    ("c\u{1F600}", "c_xD83D_xDE00", Some("c\u{1F600}")),
    ("é", "é", None),
    ("列", "列", None),
    ("ok_col", "ok_col", None),
];

enum Pv {
    S(&'static str),
    I(i32),
}

impl Pv {
    fn literal(&self) -> Literal {
        match self {
            Pv::S(s) => Literal::string(*s),
            Pv::I(i) => Literal::int(*i),
        }
    }
}

const FIXTURES: &[(&str, &str, Pv)] = &[
    ("space_v2-m0.avro", "my col", Pv::S("x")),
    ("space_v3-m0.avro", "my col", Pv::S("x")),
    ("leading_digit_v2-m0.avro", "1st", Pv::S("x")),
    ("leading_digit_v3-m0.avro", "1st", Pv::S("x")),
    ("dash_v2-m0.avro", "a-b", Pv::S("x")),
    ("dash_v3-m0.avro", "a-b", Pv::S("x")),
    ("dot_v2-m0.avro", "a.b", Pv::S("x")),
    ("dot_v3-m0.avro", "a.b", Pv::S("x")),
    ("non_ascii_letter_v2-m0.avro", "é", Pv::S("x")),
    ("non_ascii_letter_v3-m0.avro", "é", Pv::S("x")),
    ("cjk_v2-m0.avro", "列", Pv::S("x")),
    ("cjk_v3-m0.avro", "列", Pv::S("x")),
    ("emoji_v2-m0.avro", "c\u{1F600}", Pv::S("x")),
    ("emoji_v3-m0.avro", "c\u{1F600}", Pv::S("x")),
    ("bucket_space_v2-m0.avro", "my col_bucket", Pv::I(3)),
    ("bucket_space_v3-m0.avro", "my col_bucket", Pv::I(3)),
    ("truncate_dash_v2-m0.avro", "a-b_trunc", Pv::S("xy")),
    ("truncate_dash_v3-m0.avro", "a-b_trunc", Pv::S("xy")),
    ("valid_v2-m0.avro", "ok_col", Pv::S("x")),
    ("valid_v3-m0.avro", "ok_col", Pv::S("x")),
    ("repark_broken_space_m0.avro", "my col", Pv::S("x")),
];

fn two_column_schema() -> Arc<Schema> {
    Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap(),
    )
}

fn spec_for(schema: Arc<Schema>, name: &str) -> PartitionSpec {
    PartitionSpec::builder(schema)
        .with_spec_id(0)
        .add_partition_field("data", name, Transform::Identity)
        .unwrap()
        .build()
        .unwrap()
}

fn one_entry(partition_value: Literal) -> ManifestEntry {
    let data_file = DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path("test/m.parquet".to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(partition_value)]))
        .build()
        .unwrap();
    ManifestEntry {
        status: ManifestStatus::Added,
        snapshot_id: None,
        sequence_number: None,
        file_sequence_number: None,
        data_file,
    }
}

fn read_long(bs: &[u8], pos: &mut usize) -> i64 {
    let mut shift = 0;
    let mut val: u64 = 0;
    loop {
        let b = bs[*pos];
        *pos += 1;
        val |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    ((val >> 1) as i64) ^ -((val & 1) as i64)
}

fn ocf_meta(bs: &[u8]) -> HashMap<String, Vec<u8>> {
    assert_eq!(&bs[..4], b"Obj\x01");
    let mut pos = 4;
    let mut meta = HashMap::new();
    loop {
        let mut n = read_long(bs, &mut pos);
        if n == 0 {
            break;
        }
        if n < 0 {
            n = -n;
            read_long(bs, &mut pos);
        }
        for _ in 0..n {
            let kl = read_long(bs, &mut pos) as usize;
            let key = String::from_utf8(bs[pos..pos + kl].to_vec()).unwrap();
            pos += kl;
            let vl = read_long(bs, &mut pos) as usize;
            let val = bs[pos..pos + vl].to_vec();
            pos += vl;
            meta.insert(key, val);
        }
    }
    meta
}

fn find_record<'v>(v: &'v serde_json::Value, record_name: &str) -> Option<&'v serde_json::Value> {
    if let serde_json::Value::Object(map) = v {
        if map.get("type").and_then(|t| t.as_str()) == Some("record")
            && map.get("name").and_then(|n| n.as_str()) == Some(record_name)
        {
            return Some(v);
        }
        for sub in map.values() {
            if let Some(found) = find_record(sub, record_name) {
                return Some(found);
            }
        }
    }
    if let serde_json::Value::Array(arr) = v {
        for sub in arr {
            if let Some(found) = find_record(sub, record_name) {
                return Some(found);
            }
        }
    }
    None
}

async fn write_manifest_bytes(name: &str) -> Vec<u8> {
    let schema = two_column_schema();
    let spec = spec_for(schema.clone(), name);
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("m.avro");
    let io = FileIO::new_with_fs();
    let out = io.new_output(path.to_str().unwrap()).unwrap();
    let mut writer = ManifestWriterBuilder::new(out, Some(1), None, schema, spec).build_v2_data();
    writer.add_entry(one_entry(Literal::string("x"))).unwrap();
    writer.write_manifest_file().await.unwrap();
    std::fs::read(path).unwrap()
}

#[tokio::test]
async fn writes_java_avro_field_names() {
    for (iceberg_name, avro_name, attr) in CASES {
        let bs = write_manifest_bytes(iceberg_name).await;
        let meta = ocf_meta(&bs);
        let schema_json: serde_json::Value = serde_json::from_slice(&meta["avro.schema"]).unwrap();
        let r102 = find_record(&schema_json, "r102")
            .unwrap_or_else(|| panic!("no r102 record for {iceberg_name}"));
        let fields = r102["fields"].as_array().unwrap();
        assert_eq!(fields.len(), 1, "{iceberg_name}");
        let field = &fields[0];
        assert_eq!(
            field["name"].as_str().unwrap(),
            *avro_name,
            "avro field name for {iceberg_name}"
        );
        assert_eq!(
            field["field-id"].as_i64().unwrap(),
            1000,
            "field-id for {iceberg_name}"
        );
        match attr {
            Some(original) => assert_eq!(
                field["iceberg-field-name"].as_str().unwrap(),
                *original,
                "iceberg-field-name for {iceberg_name}"
            ),
            None => assert!(
                field.get("iceberg-field-name").is_none(),
                "iceberg-field-name must be absent for {iceberg_name}"
            ),
        }
    }
}

#[tokio::test]
async fn sanitizes_every_record_from_schema_to_avro_schema() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::optional(1, "my col", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(
                2,
                "outer",
                Type::Struct(crate::spec::StructType::new(vec![
                    NestedField::optional(3, "a-b", Type::Primitive(PrimitiveType::Int)).into(),
                ])),
            )
            .into(),
        ])
        .build()
        .unwrap();
    let avro = schema_to_avro_schema("data", &schema).unwrap();
    let json = serde_json::to_value(&avro).unwrap();
    let top = &json["fields"];
    assert_eq!(top[0]["name"].as_str().unwrap(), "my_x20col");
    assert_eq!(top[0]["iceberg-field-name"].as_str().unwrap(), "my col");
    let inner = find_record(&json, "r2").expect("nested record r2");
    assert_eq!(inner["fields"][0]["name"].as_str().unwrap(), "a_x2Db");
    assert_eq!(
        inner["fields"][0]["iceberg-field-name"].as_str().unwrap(),
        "a-b"
    );
}

#[test]
fn reads_spark_manifest_partition_values() {
    let dir = format!("{}/testdata/avro_names", env!("CARGO_MANIFEST_DIR"));
    for (file, iceberg_name, expected) in FIXTURES {
        let bs = std::fs::read(format!("{dir}/{file}")).unwrap();
        let manifest =
            Manifest::parse_avro(&bs).unwrap_or_else(|e| panic!("{file} must parse: {e:?}"));
        assert!(!manifest.entries().is_empty(), "{file} must hold entries");
        assert_eq!(
            manifest.metadata().partition_spec.fields()[0].name.as_str(),
            *iceberg_name,
            "spec field name for {file}"
        );
        for entry in manifest.entries() {
            assert_eq!(
                *entry.data_file().partition(),
                Struct::from_iter([Some(expected.literal())]),
                "partition value for {file}"
            );
        }
    }
}

#[tokio::test]
async fn write_then_read_round_trip() {
    for (iceberg_name, _, _) in CASES {
        let bs = write_manifest_bytes(iceberg_name).await;
        let manifest = Manifest::parse_avro(&bs)
            .unwrap_or_else(|e| panic!("{iceberg_name} round trip must parse: {e:?}"));
        assert_eq!(manifest.entries().len(), 1, "{iceberg_name}");
        assert_eq!(
            *manifest.entries()[0].data_file().partition(),
            Struct::from_iter([Some(Literal::string("x"))]),
            "{iceberg_name}"
        );
    }
}

async fn local_catalog() -> (impl Catalog, TempDir) {
    let temp_dir = TempDir::new().expect("temp dir");
    let warehouse = temp_dir.path().to_str().expect("utf8 path").to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([("warehouse".to_string(), warehouse)]),
        )
        .await
        .expect("local-fs memory catalog");
    (catalog, temp_dir)
}

async fn write_partitioned_file(
    table: &Table,
    name: &str,
    id: i32,
    col_value: &str,
) -> crate::spec::DataFile {
    let schema = table.metadata().current_schema().clone();
    let arrow_schema = Arc::new(schema_to_arrow_schema(&schema).expect("arrow schema"));
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int32Array::from(vec![id])) as ArrayRef,
        Arc::new(StringArray::from(vec![col_value])) as ArrayRef,
    ])
    .expect("batch");
    let location = format!("{}/data/{name}", table.metadata().location());
    let output = table.file_io().new_output(location).expect("output");
    let mut writer = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema,
    )
    .build(output)
    .await
    .expect("parquet writer");
    writer.write(&batch).await.expect("write");
    writer
        .close()
        .await
        .expect("close")
        .into_iter()
        .next()
        .expect("one data file")
        .content(DataContentType::Data)
        .partition_spec_id(table.metadata().default_partition_spec_id())
        .partition(Struct::from_iter([Some(Literal::string(col_value))]))
        .build()
        .expect("data file")
}

#[tokio::test]
async fn table_scan_filters_on_spaced_partition_column() {
    let (catalog, _guard) = local_catalog().await;
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "my col", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .unwrap();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("my col", "my col", Transform::Identity)
        .unwrap()
        .build()
        .unwrap();
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("t".to_string())
        .schema(schema)
        .partition_spec(spec)
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table");
    let file_x = write_partitioned_file(&table, "f-x.parquet", 1, "x").await;
    let file_y = write_partitioned_file(&table, "f-y.parquet", 2, "y").await;
    let tx = Transaction::new(&table);
    let action = tx.fast_append().add_data_files(vec![file_x, file_y]);
    let table = action
        .apply(tx)
        .expect("apply append")
        .commit(&catalog)
        .await
        .expect("commit");

    let batches: Vec<RecordBatch> = table
        .scan()
        .with_filter(Reference::new("my col").equal_to(Datum::string("x")))
        .build()
        .expect("scan")
        .to_arrow()
        .await
        .expect("to_arrow")
        .try_collect()
        .await
        .expect("collect");
    let rows: i64 = batches.iter().map(|b| b.num_rows() as i64).sum();
    assert_eq!(rows, 1, "exactly the `my col` = 'x' row survives");
    let batch = &batches[0];
    let id_col = batch
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let col = batch
        .column_by_name("my col")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(id_col.value(0), 1);
    assert_eq!(col.value(0), "x");
}

async fn write_avro_data_file(schema: Arc<Schema>, batch: &RecordBatch) -> Vec<u8> {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("d.avro");
    let io = FileIO::new_with_fs();
    let out = io.new_output(path.to_str().unwrap()).unwrap();
    let mut writer = AvroWriterBuilder::new(schema)
        .build(out)
        .await
        .expect("avro writer");
    writer.write(batch).await.expect("write batch");
    writer.close().await.expect("close");
    std::fs::read(&path).unwrap()
}

fn string_col<'b>(batch: &'b RecordBatch, name: &str) -> &'b StringArray {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("column {name}"))
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
}

fn int_col<'b>(batch: &'b RecordBatch, name: &str) -> &'b Int32Array {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("column {name}"))
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
}

#[tokio::test]
async fn avro_data_file_round_trip_sanitizes_value_keys() {
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "my col", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(3, "a-b", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(4, "1st", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(5, "c\u{1F600}", Type::Primitive(PrimitiveType::String))
                    .into(),
            ])
            .build()
            .unwrap(),
    );
    let arrow_schema = Arc::new(schema_to_arrow_schema(&schema).unwrap());
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        Arc::new(StringArray::from(vec![Some("x"), None])) as ArrayRef,
        Arc::new(Int32Array::from(vec![Some(7), None])) as ArrayRef,
        Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef,
        Arc::new(StringArray::from(vec![Some("e"), None])) as ArrayRef,
    ])
    .unwrap();

    let bs = write_avro_data_file(schema.clone(), &batch).await;
    let batches = read_avro_data_bytes(&bs, &schema, 1024).expect("read avro data file");
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 2);
    let batch = &batches[0];
    assert_eq!(int_col(batch, "id").value(0), 1);
    let my_col = string_col(batch, "my col");
    assert_eq!(my_col.value(0), "x");
    assert!(my_col.is_null(1));
    let dash = int_col(batch, "a-b");
    assert_eq!(dash.value(0), 7);
    assert!(dash.is_null(1));
    let leading = string_col(batch, "1st");
    assert_eq!(leading.value(0), "a");
    assert_eq!(leading.value(1), "b");
    let emoji = string_col(batch, "c\u{1F600}");
    assert_eq!(emoji.value(0), "e");
    assert!(emoji.is_null(1));
}

struct NoReadToEnd<R>(R);

impl<R: Read> Read for NoReadToEnd<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }

    fn read_to_end(&mut self, _buf: &mut Vec<u8>) -> std::io::Result<usize> {
        Err(std::io::Error::other(
            "read_to_end is forbidden: the container must be streamed",
        ))
    }
}

fn avro_long_out(out: &mut Vec<u8>, value: i64) {
    let mut n = ((value << 1) ^ (value >> 63)) as u64;
    loop {
        if n & !0x7f == 0 {
            out.push(n as u8);
            return;
        }
        out.push((n as u8 & 0x7f) | 0x80);
        n >>= 7;
    }
}

fn test_long(bs: &[u8], i: &mut usize) -> i64 {
    let mut v: u64 = 0;
    let mut s = 0u32;
    loop {
        let c = bs[*i];
        *i += 1;
        v |= u64::from(c & 0x7f) << s;
        if c & 0x80 == 0 {
            return ((v >> 1) as i64) ^ -((v & 1) as i64);
        }
        s += 7;
    }
}

fn test_bytes<'a>(bs: &'a [u8], i: &mut usize) -> &'a [u8] {
    let n = test_long(bs, i) as usize;
    let s = &bs[*i..*i + n];
    *i += n;
    s
}

type OcfMeta = Vec<(Vec<u8>, Vec<u8>)>;

fn ocf_parts(bs: &[u8]) -> (OcfMeta, [u8; 16], usize) {
    let mut i = 4usize;
    let mut entries = Vec::new();
    loop {
        let mut count = test_long(bs, &mut i);
        if count == 0 {
            break;
        }
        if count < 0 {
            count = -count;
            test_long(bs, &mut i);
        }
        for _ in 0..count {
            let k = test_bytes(bs, &mut i).to_vec();
            let v = test_bytes(bs, &mut i).to_vec();
            entries.push((k, v));
        }
    }
    let sync: [u8; 16] = bs[i..i + 16].try_into().unwrap();
    (entries, sync, i + 16)
}

fn ocf_container(entries: &OcfMeta, sync: [u8; 16], body: &[u8]) -> Vec<u8> {
    let mut out = b"Obj\x01".to_vec();
    avro_long_out(&mut out, entries.len() as i64);
    for (k, v) in entries {
        avro_long_out(&mut out, k.len() as i64);
        out.extend_from_slice(k);
        avro_long_out(&mut out, v.len() as i64);
        out.extend_from_slice(v);
    }
    avro_long_out(&mut out, 0);
    out.extend_from_slice(&sync);
    out.extend_from_slice(body);
    out
}

fn unfix_schema_name(node: &mut JsonValue, good: &str, broken: &str) {
    unfix_schema_names(node, &[(good.to_string(), broken.to_string())]);
}

fn unfix_schema_names(node: &mut JsonValue, renames: &[(String, String)]) {
    match node {
        JsonValue::Object(map) => {
            if let Some(name) = map.get("name").and_then(JsonValue::as_str)
                && let Some((_, broken)) = renames.iter().find(|(good, _)| good == name)
            {
                map.insert("name".to_string(), JsonValue::String(broken.clone()));
                map.remove("iceberg-field-name");
            }
            for value in map.values_mut() {
                unfix_schema_names(value, renames);
            }
        }
        JsonValue::Array(items) => items
            .iter_mut()
            .for_each(|v| unfix_schema_names(v, renames)),
        _ => {}
    }
}

fn struct_of(names: &[&str]) -> StructType {
    StructType::new(
        names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                NestedField::optional(
                    1000 + i as i32,
                    *name,
                    Type::Primitive(PrimitiveType::String),
                )
                .into()
            })
            .collect(),
    )
}

fn broken_container(
    write_names: &[&str],
    iceberg_names: &[&str],
    partition: Struct,
) -> (Vec<u8>, StructType) {
    let write_type = struct_of(write_names);
    let mut good = Vec::new();
    write_data_files_to_avro(
        &mut good,
        vec![one_data_file(partition)],
        &write_type,
        FormatVersion::V2,
    )
    .unwrap();
    let (mut entries, sync, body_off) = ocf_parts(&good);
    let schema_entry = entries
        .iter_mut()
        .find(|(k, _)| k == b"avro.schema")
        .unwrap();
    let mut json: JsonValue = serde_json::from_slice(&schema_entry.1).unwrap();
    let renames: Vec<(String, String)> = write_names
        .iter()
        .zip(iceberg_names.iter())
        .map(|(w, r)| {
            (
                crate::avro::name::java_avro_name(w).into_owned(),
                (*r).to_string(),
            )
        })
        .collect();
    unfix_schema_names(&mut json, &renames);
    schema_entry.1 = serde_json::to_vec(&json).unwrap();
    (
        ocf_container(&entries, sync, &good[body_off..]),
        struct_of(iceberg_names),
    )
}

fn one_data_file(partition: Struct) -> DataFile {
    DataFile {
        content: DataContentType::Data,
        file_path: "s3://b/f.parquet".to_string(),
        file_format: DataFileFormat::Parquet,
        partition,
        record_count: 1,
        file_size_in_bytes: 10,
        column_sizes: HashMap::new(),
        value_counts: HashMap::new(),
        null_value_counts: HashMap::new(),
        nan_value_counts: HashMap::new(),
        lower_bounds: HashMap::new(),
        upper_bounds: HashMap::new(),
        key_metadata: None,
        split_offsets: None,
        equality_ids: None,
        sort_order_id: None,
        partition_spec_id: 0,
        first_row_id: None,
        referenced_data_file: None,
        content_offset: None,
        content_size_in_bytes: None,
    }
}

fn tiny_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()
        .unwrap()
}

#[test]
fn ocf_repair_json_parses_only_when_schema_names_need_it() {
    let dir = format!("{}/testdata/avro_names", env!("CARGO_MANIFEST_DIR"));
    let valid = std::fs::read(format!("{dir}/valid_v2-m0.avro")).unwrap();
    let renamed = std::fs::read(format!("{dir}/space_v2-m0.avro")).unwrap();
    let before = ocf_json_parse_count();
    for bs in [&valid, &renamed] {
        let repaired = repair_avro_container(bs).unwrap();
        assert!(matches!(repaired, Cow::Borrowed(_)));
        assert_eq!(repaired.as_ref(), bs.as_slice());
    }
    assert_eq!(ocf_json_parse_count(), before);

    let broken = std::fs::read(format!("{dir}/repark_broken_space_m0.avro")).unwrap();
    let repaired = repair_avro_container(&broken).unwrap();
    assert!(matches!(repaired, Cow::Owned(_)));
    assert_eq!(ocf_json_parse_count(), before + 1);
}

#[test]
fn data_files_avro_reader_streams_without_read_to_end() {
    let schema = tiny_schema();
    let partition_type = StructType::new(vec![]);
    let mut bytes = Vec::new();
    write_data_files_to_avro(
        &mut bytes,
        vec![one_data_file(Struct::empty())],
        &partition_type,
        FormatVersion::V2,
    )
    .unwrap();
    let mut reader = NoReadToEnd(Cursor::new(bytes));
    let files =
        read_data_files_from_avro(&mut reader, &schema, 0, &partition_type, FormatVersion::V2)
            .unwrap();
    assert_eq!(files.len(), 1);
}

#[test]
fn data_files_avro_reader_repairs_schema_names_while_streaming() {
    let schema = tiny_schema();
    let partition_type = StructType::new(vec![
        NestedField::optional(1000, "my col", Type::Primitive(PrimitiveType::String)).into(),
    ]);
    let mut good = Vec::new();
    write_data_files_to_avro(
        &mut good,
        vec![one_data_file(Struct::from_iter([Some(Literal::string(
            "x",
        ))]))],
        &partition_type,
        FormatVersion::V2,
    )
    .unwrap();
    let (mut entries, sync, body_off) = ocf_parts(&good);
    let schema_entry = entries
        .iter_mut()
        .find(|(k, _)| k == b"avro.schema")
        .unwrap();
    let mut json: JsonValue = serde_json::from_slice(&schema_entry.1).unwrap();
    unfix_schema_name(&mut json, "my_x20col", "my col");
    schema_entry.1 = serde_json::to_vec(&json).unwrap();
    let broken = ocf_container(&entries, sync, &good[body_off..]);
    assert!(AvroReader::new(Cursor::new(&broken)).is_err());

    let mut reader = NoReadToEnd(Cursor::new(broken));
    let files =
        read_data_files_from_avro(&mut reader, &schema, 0, &partition_type, FormatVersion::V2)
            .unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].partition[0], Some(Literal::string("x")));
}

fn record_fields(node: &JsonValue, record_name: &str) -> Option<Vec<(String, Option<String>)>> {
    match node {
        JsonValue::Object(map) => {
            if map.get("name").and_then(JsonValue::as_str) == Some(record_name)
                && let Some(JsonValue::Array(fields)) = map.get("fields")
            {
                return Some(
                    fields
                        .iter()
                        .map(|f| {
                            (
                                f.get("name")
                                    .and_then(JsonValue::as_str)
                                    .unwrap_or_default()
                                    .to_string(),
                                f.get("iceberg-field-name")
                                    .and_then(JsonValue::as_str)
                                    .map(str::to_string),
                            )
                        })
                        .collect(),
                );
            }
            map.values().find_map(|v| record_fields(v, record_name))
        }
        JsonValue::Array(items) => items.iter().find_map(|v| record_fields(v, record_name)),
        _ => None,
    }
}

#[test]
fn colliding_avro_field_names_fail_at_schema_build() {
    for (a, b, avro_name) in [("a b", "a_x20b", "a_x20b"), ("1a", "_1a", "_1a")] {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, a, Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(2, b, Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();
        let err = schema_to_avro_schema("data", &schema).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
        let msg = err.to_string();
        assert!(
            msg.contains(a) && msg.contains(b) && msg.contains(avro_name),
            "error must name both Iceberg fields and the Avro name: {msg}"
        );
    }
}

#[test]
fn repaired_colliding_names_bind_distinctly() {
    for (ice_a, ice_b, avro_a, avro_b) in [
        ("a b", "a_x20b", "a_x20b_1", "a_x20b"),
        ("1a", "_1a", "_1a_1", "_1a"),
        ("é", "_xE9", "_xE9_1", "_xE9"),
        ("列", "_x5217", "_x5217_1", "_x5217"),
    ] {
        let (broken, read_type) = broken_container(
            &["f1", "f2"],
            &[ice_a, ice_b],
            Struct::from_iter([Some(Literal::string("v1")), Some(Literal::string("v2"))]),
        );
        assert!(
            AvroReader::new(Cursor::new(&broken)).is_err(),
            "container must be unparseable before repair"
        );

        let repaired = repair_avro_container(&broken).unwrap();
        assert!(matches!(repaired, Cow::Owned(_)));
        let (entries, _, _) = ocf_parts(&repaired);
        let schema_json = &entries.iter().find(|(k, _)| k == b"avro.schema").unwrap().1;
        let json: JsonValue = serde_json::from_slice(schema_json).unwrap();
        let fields = record_fields(&json, "r102").expect("partition record r102");
        assert_eq!(
            fields,
            vec![
                (avro_a.to_string(), Some(ice_a.to_string())),
                (avro_b.to_string(), None),
            ],
            "repaired names for {ice_a}/{ice_b}"
        );

        let mut reader = Cursor::new(repaired.into_owned());
        let files = read_data_files_from_avro(
            &mut reader,
            &tiny_schema(),
            0,
            &read_type,
            FormatVersion::V2,
        )
        .unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].partition[0], Some(Literal::string("v1")));
        assert_eq!(files[0].partition[1], Some(Literal::string("v2")));
    }
}

#[test]
fn unique_avro_names_bind_before_literal_names() {
    let ty = Type::Struct(struct_of(&["a b", "a_x20b"]));
    let value = apache_avro::types::Value::Record(vec![
        (
            "a_x20b_1".to_string(),
            apache_avro::types::Value::String("v_ab".to_string()),
        ),
        (
            "a_x20b".to_string(),
            apache_avro::types::Value::String("v_lit".to_string()),
        ),
    ]);
    let raw: crate::spec::RawLiteral = apache_avro::from_value(&value).unwrap();
    let lit = raw.try_into(&ty).unwrap();
    assert_eq!(
        lit,
        Some(Literal::Struct(Struct::from_iter([
            Some(Literal::string("v_ab")),
            Some(Literal::string("v_lit")),
        ])))
    );
}
