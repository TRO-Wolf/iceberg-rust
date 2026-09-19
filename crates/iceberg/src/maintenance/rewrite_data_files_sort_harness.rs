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

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type, TimestampMicrosecondType};
use arrow_array::{
    ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray,
};
use futures::TryStreamExt;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;

use crate::arrow::{ArrowFileReader, schema_to_arrow_schema};
use crate::io::FileMetadata;
use crate::maintenance::rewrite_data_files::tests::append_files;
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, Literal, NestedField, PartitionSpec,
    PrimitiveType, Schema, SortOrder, Struct, Transform, Type,
};
use crate::table::Table;
use crate::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
use crate::{Catalog, NamespaceIdent, TableCreation};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct OracleRow {
    pub(super) index: i64,
    pub(super) id: Option<i64>,
    pub(super) cat: Option<String>,
    pub(super) ts: i64,
    pub(super) v: Option<f64>,
    pub(super) s: Option<String>,
}

pub(super) fn oracle_row(index: i64) -> OracleRow {
    let cat = match (index * 7) % 4 {
        0 => Some("a".to_string()),
        1 => Some("b".to_string()),
        2 => Some("c".to_string()),
        _ => None,
    };
    let s = if index % 9 == 0 {
        None
    } else {
        Some(format!("s{:02}", (index * 13) % 50))
    };
    let v = if index == 42 {
        Some(f64::NAN)
    } else if index % 17 == 0 {
        None
    } else {
        Some((index as f64 * 1.5) % 23.0)
    };
    OracleRow {
        index,
        id: if index == 77 { None } else { Some(index) },
        cat,
        ts: oracle_micros(1 + (index % 3), 1 + (index * 3) % 28, (index * 5) % 24),
        v,
        s,
    }
}

fn oracle_micros(month: i64, day: i64, hour: i64) -> i64 {
    let date = chrono::NaiveDate::from_ymd_opt(2024, month as u32, day as u32)
        .expect("oracle date")
        .and_hms_opt(hour as u32, 0, 0)
        .expect("oracle time");
    date.and_utc().timestamp_micros()
}

pub(super) fn oracle_batches() -> Vec<Vec<OracleRow>> {
    (0..4)
        .map(|batch| {
            (0..25)
                .map(|row| oracle_row((row * 37 + batch * 11) % 100))
                .collect()
        })
        .collect()
}

pub(super) fn oracle_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            Arc::new(NestedField::optional(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                2,
                "cat",
                Type::Primitive(PrimitiveType::String),
            )),
            Arc::new(NestedField::optional(
                3,
                "ts",
                Type::Primitive(PrimitiveType::Timestamptz),
            )),
            Arc::new(NestedField::optional(
                4,
                "v",
                Type::Primitive(PrimitiveType::Double),
            )),
            Arc::new(NestedField::optional(
                5,
                "s",
                Type::Primitive(PrimitiveType::String),
            )),
        ])
        .build()
        .expect("build the oracle schema")
}

pub(super) async fn oracle_table(
    catalog: &impl Catalog,
    format_version: FormatVersion,
    partitioned: bool,
    sort_order: Option<SortOrder>,
) -> Table {
    let schema = oracle_schema();
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let mut spec = None;
    if partitioned {
        spec = Some(
            PartitionSpec::builder(schema.clone())
                .with_spec_id(0)
                .add_partition_field("cat", "cat", Transform::Identity)
                .expect("add partition field")
                .build()
                .expect("build spec")
                .into_unbound(),
        );
    }
    let creation = TableCreation {
        name: "t".to_string(),
        location: None,
        schema: schema.clone(),
        partition_spec: spec,
        sort_order,
        properties: HashMap::new(),
        format_version,
    };
    let mut table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create the oracle table");

    for (batch_index, rows) in oracle_batches().into_iter().enumerate() {
        let files = if partitioned {
            let mut by_partition: Vec<(Option<String>, Vec<OracleRow>)> = Vec::new();
            for row in rows {
                match by_partition.iter_mut().find(|(cat, _)| *cat == row.cat) {
                    Some((_, bucket)) => bucket.push(row),
                    None => by_partition.push((row.cat.clone(), vec![row])),
                }
            }
            let mut files = Vec::new();
            for (partition, bucket) in by_partition {
                files.push(
                    write_oracle_file(
                        &table,
                        &format!("in-{batch_index}-{}.parquet", files.len()),
                        &bucket,
                        Some(partition),
                    )
                    .await,
                );
            }
            files
        } else {
            vec![write_oracle_file(&table, &format!("in-{batch_index}.parquet"), &rows, None).await]
        };
        table = append_files(catalog, &table, files).await;
    }
    table
}

pub(super) async fn write_oracle_file(
    table: &Table,
    file_name: &str,
    rows: &[OracleRow],
    partition: Option<Option<String>>,
) -> DataFile {
    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("arrow schema"));
    let batch = oracle_batch(rows, &arrow_schema);

    let file_path = format!("{}/data/{file_name}", table.metadata().location());
    let output = table.file_io().new_output(file_path).expect("output file");
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let mut writer = parquet_builder.build(output).await.expect("parquet writer");
    writer.write(&batch).await.expect("write oracle rows");
    let mut builder = writer
        .close()
        .await
        .expect("close oracle writer")
        .into_iter()
        .next()
        .expect("one data file builder");
    builder.content(DataContentType::Data);
    match partition {
        Some(cat) => {
            builder
                .partition_spec_id(0)
                .partition(Struct::from_iter([cat.map(|value| Literal::string(value))]));
        }
        None => {
            builder.partition_spec_id(0).partition(Struct::empty());
        }
    }
    builder.build().expect("build the oracle data file")
}

fn oracle_batch(rows: &[OracleRow], arrow_schema: &Arc<arrow_schema::Schema>) -> RecordBatch {
    let ids = Int64Array::from(rows.iter().map(|row| row.id).collect::<Vec<_>>());
    let cats = StringArray::from(rows.iter().map(|row| row.cat.clone()).collect::<Vec<_>>());
    let timestamps =
        TimestampMicrosecondArray::from(rows.iter().map(|row| row.ts).collect::<Vec<_>>())
            .with_timezone("UTC");
    let values = Float64Array::from(rows.iter().map(|row| row.v).collect::<Vec<_>>());
    let strings = StringArray::from(rows.iter().map(|row| row.s.clone()).collect::<Vec<_>>());
    RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(ids) as ArrayRef,
        Arc::new(cats) as ArrayRef,
        Arc::new(timestamps) as ArrayRef,
        Arc::new(values) as ArrayRef,
        Arc::new(strings) as ArrayRef,
    ])
    .expect("build the oracle batch")
}

pub(super) struct OutputFile {
    pub(super) name: String,
    pub(super) path: String,
    pub(super) sort_order_id: Option<i32>,
    pub(super) record_count: u64,
    pub(super) partition: Struct,
    pub(super) rows: Vec<OracleRow>,
}

pub(super) async fn output_files(table: &Table) -> Vec<OutputFile> {
    let mut files = Vec::new();
    let Some(snapshot) = table.metadata().current_snapshot() else {
        return files;
    };
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load manifest list");
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("load manifest");
        for entry in manifest.entries() {
            if !entry.is_alive() || entry.content_type() != DataContentType::Data {
                continue;
            }
            let data_file = entry.data_file();
            let path = data_file.file_path().to_string();
            let name = path.rsplit('/').next().unwrap_or(&path).to_string();
            files.push(OutputFile {
                name,
                rows: read_rows_in_file_order(table, &path).await,
                sort_order_id: data_file.sort_order_id(),
                record_count: data_file.record_count(),
                partition: data_file.partition().clone(),
                path,
            });
        }
    }
    files.sort_by(|left, right| left.name.cmp(&right.name));
    files
}

pub(super) async fn live_output_paths(table: &Table) -> Vec<(String, Option<i32>)> {
    let mut files = Vec::new();
    let Some(snapshot) = table.metadata().current_snapshot() else {
        return files;
    };
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load manifest list");
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("load manifest");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.content_type() == DataContentType::Data {
                files.push((
                    entry.data_file().file_path().to_string(),
                    entry.data_file().sort_order_id(),
                ));
            }
        }
    }
    files.sort_by(|left, right| {
        let name = |path: &String| path.rsplit('/').next().unwrap_or(path).to_string();
        name(&left.0).cmp(&name(&right.0))
    });
    files
}

pub(super) async fn read_rows_in_file_order(table: &Table, path: &str) -> Vec<OracleRow> {
    let input = table.file_io().new_input(path).expect("input file");
    let size = input.metadata().await.expect("file metadata").size;
    let reader = ArrowFileReader::new(
        FileMetadata { size },
        input.reader().await.expect("file reader"),
    );
    let stream = ParquetRecordBatchStreamBuilder::new(reader)
        .await
        .expect("parquet stream builder")
        .build()
        .expect("parquet stream");
    let batches: Vec<RecordBatch> = stream.try_collect().await.expect("read output batches");

    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch.column_by_name("id").expect("id column").clone();
        let cats = batch.column_by_name("cat").expect("cat column").clone();
        let timestamps = batch.column_by_name("ts").expect("ts column").clone();
        let values = batch.column_by_name("v").expect("v column").clone();
        let strings = batch.column_by_name("s").expect("s column").clone();
        for row in 0..batch.num_rows() {
            let id = ids
                .is_null(row)
                .then_some(None)
                .unwrap_or_else(|| Some(ids.as_primitive::<Int64Type>().value(row)));
            let index = id.unwrap_or(77);
            rows.push(OracleRow {
                index,
                id,
                cat: (!cats.is_null(row)).then(|| cats.as_string::<i32>().value(row).to_string()),
                ts: timestamps
                    .as_primitive::<TimestampMicrosecondType>()
                    .value(row),
                v: (!values.is_null(row)).then(|| values.as_primitive::<Float64Type>().value(row)),
                s: (!strings.is_null(row))
                    .then(|| strings.as_string::<i32>().value(row).to_string()),
            });
        }
    }
    rows
}

pub(super) fn concatenated_rows(files: &[OutputFile]) -> Vec<OracleRow> {
    files.iter().flat_map(|file| file.rows.clone()).collect()
}

pub(super) fn indexes(rows: &[OracleRow]) -> Vec<i64> {
    rows.iter().map(|row| row.index).collect()
}

pub(super) fn spill_files(table: &Table) -> Vec<String> {
    let location = table
        .metadata()
        .location()
        .trim_start_matches("file:/")
        .to_string();
    let mut found = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(format!("/{location}"))];
    while let Some(directory) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.to_string_lossy().contains("rewrite-sort-spill-") {
                found.push(path.to_string_lossy().to_string());
            }
        }
    }
    found
}

pub(super) const OUTPUT_FORMAT: DataFileFormat = DataFileFormat::Parquet;
