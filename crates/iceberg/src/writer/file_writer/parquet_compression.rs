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

use parquet::basic::{Compression, GzipLevel, ZstdLevel};

use crate::spec::TableProperties;
use crate::{Error, ErrorKind, Result};

fn invalid_property(key: &str, value: &str) -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        format!("Invalid value for {key}: {value}"),
    )
}

fn parse_zstd_level(value: &str) -> Result<ZstdLevel> {
    let parsed = value.trim().parse::<i32>().map_err(|_| {
        invalid_property(TableProperties::PROPERTY_PARQUET_COMPRESSION_LEVEL, value)
    })?;
    ZstdLevel::try_new(parsed).map_err(|error| {
        invalid_property(TableProperties::PROPERTY_PARQUET_COMPRESSION_LEVEL, value)
            .with_source(error)
    })
}

fn parse_gzip_level(value: &str) -> Result<GzipLevel> {
    let parsed = value.trim().parse::<u32>().map_err(|_| {
        invalid_property(TableProperties::PROPERTY_PARQUET_COMPRESSION_LEVEL, value)
    })?;
    GzipLevel::try_new(parsed).map_err(|error| {
        invalid_property(TableProperties::PROPERTY_PARQUET_COMPRESSION_LEVEL, value)
            .with_source(error)
    })
}

/// Parse Iceberg parquet compression table properties into a parquet codec.
pub fn parquet_compression_from_properties(
    properties: &HashMap<String, String>,
) -> Result<Compression> {
    let codec_raw = properties
        .get(TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC)
        .map(String::as_str)
        .unwrap_or(TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC_DEFAULT);
    let codec = codec_raw.trim();
    let level_raw = properties.get(TableProperties::PROPERTY_PARQUET_COMPRESSION_LEVEL);

    match codec.to_ascii_lowercase().as_str() {
        "zstd" => {
            let level = match level_raw {
                None => ZstdLevel::try_new(3)?,
                Some(value) => parse_zstd_level(value)?,
            };
            Ok(Compression::ZSTD(level))
        }
        "gzip" => {
            let level = match level_raw {
                None => GzipLevel::default(),
                Some(value) => parse_gzip_level(value)?,
            };
            Ok(Compression::GZIP(level))
        }
        "snappy" => Ok(Compression::SNAPPY),
        "lz4" | "lz4_raw" => Ok(Compression::LZ4_RAW),
        "uncompressed" => Ok(Compression::UNCOMPRESSED),
        _ => Err(invalid_property(
            TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC,
            codec_raw,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect()
    }

    fn parse(pairs: &[(&str, &str)]) -> Compression {
        parquet_compression_from_properties(&props(pairs))
            .expect("parse parquet compression properties")
    }

    fn parse_err(pairs: &[(&str, &str)]) -> Error {
        parquet_compression_from_properties(&props(pairs)).expect_err("parse must fail")
    }

    #[test]
    fn parquet_compression_default_is_zstd() {
        let compression = parse(&[]);
        match compression {
            Compression::ZSTD(level) => {
                assert_eq!(level.compression_level(), 3);
            }
            other => panic!("expected default ZSTD, got {other:?}"),
        }
    }

    #[test]
    fn parquet_compression_parses_each_codec() {
        assert_eq!(
            parse(&[(
                TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC,
                "snappy"
            )]),
            Compression::SNAPPY
        );
        assert!(matches!(
            parse(&[(TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC, "gzip")]),
            Compression::GZIP(_)
        ));
        assert_eq!(
            parse(&[(TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC, "lz4")]),
            Compression::LZ4_RAW
        );
        assert_eq!(
            parse(&[(
                TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC,
                "lz4_raw"
            )]),
            Compression::LZ4_RAW
        );
        assert_eq!(
            parse(&[(
                TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC,
                "uncompressed"
            )]),
            Compression::UNCOMPRESSED
        );
        assert!(matches!(
            parse(&[(TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC, "zstd")]),
            Compression::ZSTD(_)
        ));
    }

    #[test]
    fn parquet_compression_codec_is_case_insensitive() {
        assert_eq!(
            parse(&[(
                TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC,
                "SNAPPY"
            )]),
            Compression::SNAPPY
        );
        assert_eq!(
            parse(&[(
                TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC,
                "Lz4_Raw"
            )]),
            Compression::LZ4_RAW
        );
        assert!(matches!(
            parse(&[(TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC, "GzIp")]),
            Compression::GZIP(_)
        ));
        assert!(matches!(
            parse(&[(TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC, "ZSTD")]),
            Compression::ZSTD(_)
        ));
    }

    #[test]
    fn parquet_compression_honours_zstd_and_gzip_level() {
        match parse(&[
            (TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC, "zstd"),
            (TableProperties::PROPERTY_PARQUET_COMPRESSION_LEVEL, "3"),
        ]) {
            Compression::ZSTD(level) => assert_eq!(level.compression_level(), 3),
            other => panic!("expected ZSTD(3), got {other:?}"),
        }
        match parse(&[
            (TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC, "gzip"),
            (TableProperties::PROPERTY_PARQUET_COMPRESSION_LEVEL, "3"),
        ]) {
            Compression::GZIP(level) => assert_eq!(level.compression_level(), 3),
            other => panic!("expected GZIP(3), got {other:?}"),
        }
    }

    #[test]
    fn parquet_compression_bad_level_fails_loud() {
        for (codec, value) in [
            ("zstd", "abc"),
            ("zstd", "0"),
            ("zstd", "23"),
            ("gzip", "nope"),
            ("gzip", "10"),
        ] {
            let error = parse_err(&[
                (TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC, codec),
                (TableProperties::PROPERTY_PARQUET_COMPRESSION_LEVEL, value),
            ]);
            assert_eq!(error.kind(), ErrorKind::DataInvalid);
            let message = error.to_string();
            assert!(
                message.contains(TableProperties::PROPERTY_PARQUET_COMPRESSION_LEVEL),
                "error must name the level key, got {message}"
            );
            assert!(
                message.contains(value),
                "error must name the bad level {value}, got {message}"
            );
        }
    }

    #[test]
    fn parquet_compression_unknown_codec_fails_loud() {
        let error = parse_err(&[(
            TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC,
            "brotli",
        )]);
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        let message = error.to_string();
        assert!(
            message.contains(TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC),
            "error must name the codec key, got {message}"
        );
        assert!(
            message.contains("brotli"),
            "error must name the bad value, got {message}"
        );
    }
}
