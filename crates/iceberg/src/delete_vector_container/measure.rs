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

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use super::{LegacyPositionDelete, load_legacy_positions_by_path};
use crate::io::FileIO;
use crate::metadata_columns::{
    RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
};
use crate::spec::{DataContentType, DataFileBuilder, DataFileFormat, Struct};

fn write_pos_parquet(path: &str, rows: Vec<(String, i64)>, with_row_column: bool) -> u64 {
    let mut fields = vec![
        Field::new("file_path", DataType::Utf8, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            RESERVED_FIELD_ID_DELETE_FILE_PATH.to_string(),
        )])),
        Field::new("pos", DataType::Int64, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            RESERVED_FIELD_ID_DELETE_FILE_POS.to_string(),
        )])),
    ];
    if with_row_column {
        fields.push(Field::new("row", DataType::Utf8, true));
    }
    let schema = Arc::new(ArrowSchema::new(fields));
    let paths: Vec<String> = rows.iter().map(|(p, _)| p.clone()).collect();
    let positions: Vec<i64> = rows.iter().map(|(_, pos)| *pos).collect();
    let mut cols: Vec<Arc<dyn arrow_array::Array>> = vec![
        Arc::new(StringArray::from(paths)),
        Arc::new(Int64Array::from(positions)),
    ];
    if with_row_column {
        let pad: Vec<String> = (0..rows.len())
            .map(|index| format!("row-{index:0200}"))
            .collect();
        cols.push(Arc::new(StringArray::from(pad)));
    }
    let batch = RecordBatch::try_new(schema.clone(), cols).expect("batch");
    let file = std::fs::File::create(path).expect("create");
    let mut writer = ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build()))
        .expect("writer");
    writer.write(&batch).expect("write");
    writer.close().expect("close");
    std::fs::metadata(path).expect("metadata").len()
}

#[tokio::test]
#[ignore = "measurement, not a CI pin"]
async fn measure_load_legacy_64_paths() {
    let warehouse = TempDir::new().expect("warehouse");
    let del_path = format!("{}/d.parquet", warehouse.path().to_str().expect("utf8"));
    let mut rows = Vec::new();
    for file in 0..64 {
        let path = format!("s3://b/f{file}.parquet");
        for pos in 0..100 {
            rows.push((path.clone(), pos));
        }
    }
    let file_size = write_pos_parquet(&del_path, rows, false);
    let file = DataFileBuilder::default()
        .content(DataContentType::PositionDeletes)
        .file_path(del_path)
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(file_size)
        .record_count(64 * 100)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .expect("delete file");
    let touched: Vec<String> = (0..64)
        .map(|file| format!("s3://b/f{file}.parquet"))
        .collect();
    let delete = LegacyPositionDelete {
        file,
        touched,
        file_scoped: false,
        data_sequence_number: Some(1),
    };
    let file_io = FileIO::new_with_fs();
    let start = std::time::Instant::now();
    let index = load_legacy_positions_by_path(&file_io, &delete)
        .await
        .expect("load");
    let elapsed = start.elapsed();
    println!(
        "F-22 P1-a 64-path load_legacy_positions_by_path elapsed={elapsed:?} paths={}",
        index.len()
    );
    assert_eq!(index.len(), 64);
}

#[tokio::test]
#[ignore = "measurement, not a CI pin"]
async fn measure_load_legacy_512k_file_scoped() {
    let warehouse = TempDir::new().expect("warehouse");
    let del_path = format!("{}/d.parquet", warehouse.path().to_str().expect("utf8"));
    let data_path = "s3://b/a.parquet";
    let n = 512_000i64;
    let rows: Vec<(String, i64)> = (0..n).map(|pos| (data_path.to_string(), pos)).collect();
    let file_size = write_pos_parquet(&del_path, rows, false);
    let file = DataFileBuilder::default()
        .content(DataContentType::PositionDeletes)
        .file_path(del_path)
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(file_size)
        .record_count(u64::try_from(n).expect("n"))
        .partition_spec_id(0)
        .partition(Struct::empty())
        .referenced_data_file(Some(data_path.to_string()))
        .build()
        .expect("delete file");
    let delete = LegacyPositionDelete {
        file,
        touched: vec![data_path.to_string()],
        file_scoped: true,
        data_sequence_number: Some(1),
    };
    let file_io = FileIO::new_with_fs();
    let start = std::time::Instant::now();
    let index = load_legacy_positions_by_path(&file_io, &delete)
        .await
        .expect("load");
    let elapsed = start.elapsed();
    println!(
        "F-22 P1-c 512k file-scoped load_legacy_positions_by_path elapsed={elapsed:?} n={}",
        index.get(data_path).map(Vec::len).unwrap_or(0)
    );
    assert_eq!(
        index.get(data_path).map(Vec::len),
        Some(usize::try_from(n).expect("n"))
    );
}

#[tokio::test]
#[ignore = "measurement, not a CI pin"]
async fn measure_load_legacy_touched_128_partition_scoped() {
    let warehouse = TempDir::new().expect("warehouse");
    let del_path = format!("{}/d.parquet", warehouse.path().to_str().expect("utf8"));
    let n = 512_000i64;
    let mut rows = Vec::new();
    for pos in 0..n {
        let file = pos % 128;
        rows.push((format!("s3://b/f{file}.parquet"), pos));
    }
    let file_size = write_pos_parquet(&del_path, rows, false);
    let file = DataFileBuilder::default()
        .content(DataContentType::PositionDeletes)
        .file_path(del_path)
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(file_size)
        .record_count(u64::try_from(n).expect("n"))
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .expect("delete file");
    let touched: Vec<String> = (0..128)
        .map(|file| format!("s3://b/f{file}.parquet"))
        .collect();
    let delete = LegacyPositionDelete {
        file,
        touched,
        file_scoped: false,
        data_sequence_number: Some(1),
    };
    let file_io = FileIO::new_with_fs();
    let start = std::time::Instant::now();
    let index = load_legacy_positions_by_path(&file_io, &delete)
        .await
        .expect("load");
    let elapsed = start.elapsed();
    println!(
        "F-22 P1-d touched=128 partition-scoped load_legacy_positions_by_path elapsed={elapsed:?} paths={}",
        index.len()
    );
    assert_eq!(index.len(), 128);
}
