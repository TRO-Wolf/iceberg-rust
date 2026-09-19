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

use apache_avro::Reader as AvroReader;
use serde_json::Value as JsonValue;

use super::schema_to_avro_schema;
use crate::avro::ocf::{ocf_json_parse_count, repair_avro_container};
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, Literal, NestedField, PrimitiveType,
    Schema, Struct, StructType, Type, read_data_files_from_avro, write_data_files_to_avro,
};

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
            "container must be unparsable before repair"
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

fn varint_len(n: usize) -> usize {
    let mut v = (n as u64) << 1;
    let mut len = 1;
    while v >= 0x80 {
        v >>= 7;
        len += 1;
    }
    len
}

fn ocf_container_blocked(
    entries: &OcfMeta,
    blocks: &[i64],
    sync: [u8; 16],
    body: &[u8],
) -> Vec<u8> {
    let mut out = b"Obj\x01".to_vec();
    let mut i = 0;
    for &n in blocks {
        let count = n.unsigned_abs() as usize;
        let chunk = &entries[i..i + count];
        avro_long_out(&mut out, n);
        if n < 0 {
            let byte_size: usize = chunk
                .iter()
                .map(|(k, v)| varint_len(k.len()) + k.len() + varint_len(v.len()) + v.len())
                .sum();
            avro_long_out(&mut out, byte_size as i64);
        }
        for (k, v) in chunk {
            avro_long_out(&mut out, k.len() as i64);
            out.extend_from_slice(k);
            avro_long_out(&mut out, v.len() as i64);
            out.extend_from_slice(v);
        }
        i += count;
    }
    avro_long_out(&mut out, 0);
    out.extend_from_slice(&sync);
    out.extend_from_slice(body);
    out
}

fn codec_container(codec: apache_avro::Codec, name: &str) -> Vec<u8> {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::optional(1, name, Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .unwrap();
    let avro = schema_to_avro_schema("data", &schema).unwrap();
    let mut out = Vec::new();
    let mut writer = apache_avro::Writer::with_codec(&avro, &mut out, codec);
    let value = apache_avro::types::Value::Record(vec![(
        crate::avro::name::java_avro_name(name).into_owned(),
        apache_avro::types::Value::Union(
            1,
            Box::new(apache_avro::types::Value::String("v".to_string())),
        ),
    )]);
    writer.append(value.resolve(&avro).unwrap()).unwrap();
    writer.into_inner().unwrap();
    out
}

#[test]
fn ocf_metadata_multi_block_and_negative_blocks_decode() {
    let partition_type = struct_of(&["my_col"]);
    let partition = Struct::from_iter([Some(Literal::string("x"))]);
    let mut good = Vec::new();
    write_data_files_to_avro(
        &mut good,
        vec![one_data_file(partition)],
        &partition_type,
        FormatVersion::V2,
    )
    .unwrap();
    let (entries, sync, body_off) = ocf_parts(&good);
    let body = &good[body_off..];
    let last = entries.len() as i64 - 1;
    for (label, blocks) in [
        ("two positive", &[1i64, last][..]),
        ("negative", &[-(entries.len() as i64)][..]),
        ("mixed", &[1i64, -last][..]),
    ] {
        let container = ocf_container_blocked(&entries, blocks, sync, body);
        assert_eq!(ocf_parts(&container).0, entries, "{label}");
        assert!(AvroReader::new(Cursor::new(&container)).is_ok(), "{label}");
        assert!(
            matches!(repair_avro_container(&container).unwrap(), Cow::Borrowed(_)),
            "{label}"
        );
        let mut reader = Cursor::new(container);
        let files = read_data_files_from_avro(
            &mut reader,
            &tiny_schema(),
            0,
            &partition_type,
            FormatVersion::V2,
        )
        .unwrap();
        assert_eq!(files[0].partition[0], Some(Literal::string("x")), "{label}");
    }

    let mut broken_entries = entries.clone();
    let schema_entry = broken_entries
        .iter_mut()
        .find(|(k, _)| k == b"avro.schema")
        .unwrap();
    let mut json: JsonValue = serde_json::from_slice(&schema_entry.1).unwrap();
    unfix_schema_name(&mut json, "my_col", "my col");
    schema_entry.1 = serde_json::to_vec(&json).unwrap();
    let broken = ocf_container_blocked(
        &broken_entries,
        &[1, -(entries.len() as i64 - 1)],
        sync,
        body,
    );
    let repaired = repair_avro_container(&broken).unwrap().into_owned();
    let mut reader = Cursor::new(repaired);
    let files = read_data_files_from_avro(
        &mut reader,
        &tiny_schema(),
        0,
        &struct_of(&["my col"]),
        FormatVersion::V2,
    )
    .unwrap();
    assert_eq!(files[0].partition[0], Some(Literal::string("x")));
}

#[test]
fn ocf_repair_passes_through_valid_containers_byte_identical() {
    for codec in [
        apache_avro::Codec::Null,
        apache_avro::Codec::Deflate(Default::default()),
        apache_avro::Codec::Zstandard(Default::default()),
    ] {
        let container = codec_container(codec, "ok_col");
        assert!(AvroReader::new(Cursor::new(&container)).is_ok());
        let repaired = repair_avro_container(&container).unwrap();
        assert!(matches!(repaired, Cow::Borrowed(_)));
        assert_eq!(repaired.as_ref(), container.as_slice());
    }
}

#[test]
fn ocf_repair_fixes_schema_names_inside_snappy_and_zstd_containers() {
    for codec in [
        apache_avro::Codec::Null,
        apache_avro::Codec::Zstandard(Default::default()),
    ] {
        let good = codec_container(codec, "my col");
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

        let repaired = repair_avro_container(&broken).unwrap().into_owned();
        let mut reader = AvroReader::new(Cursor::new(&repaired)).unwrap();
        let value = reader.next().unwrap().unwrap();
        assert_eq!(
            value,
            apache_avro::types::Value::Record(vec![(
                "my_x20col".to_string(),
                apache_avro::types::Value::Union(
                    1,
                    Box::new(apache_avro::types::Value::String("v".to_string()))
                )
            )])
        );
    }

    let good = codec_container(apache_avro::Codec::Null, "my col");
    let (mut entries, sync, body_off) = ocf_parts(&good);
    let codec_entry = entries
        .iter_mut()
        .find(|(k, _)| k == b"avro.codec")
        .unwrap();
    codec_entry.1 = b"snappy".to_vec();
    let schema_entry = entries
        .iter_mut()
        .find(|(k, _)| k == b"avro.schema")
        .unwrap();
    let mut json: JsonValue = serde_json::from_slice(&schema_entry.1).unwrap();
    unfix_schema_name(&mut json, "my_x20col", "my col");
    schema_entry.1 = serde_json::to_vec(&json).unwrap();
    let broken = ocf_container(&entries, sync, &good[body_off..]);
    let repaired = repair_avro_container(&broken).unwrap().into_owned();
    let (rep_entries, _, _) = ocf_parts(&repaired);
    let codec_entry = rep_entries
        .iter()
        .find(|(k, _)| k == b"avro.codec")
        .unwrap();
    assert_eq!(codec_entry.1, b"snappy");
    let schema_entry = rep_entries
        .iter()
        .find(|(k, _)| k == b"avro.schema")
        .unwrap();
    let json: JsonValue = serde_json::from_slice(&schema_entry.1).unwrap();
    let fields = record_fields(&json, "data").expect("root record");
    assert_eq!(fields, vec![(
        "my_x20col".to_string(),
        Some("my col".to_string())
    )]);
}
