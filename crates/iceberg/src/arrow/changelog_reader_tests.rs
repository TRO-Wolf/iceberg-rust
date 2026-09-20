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
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use futures::TryStreamExt;
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::arrow::{ArrowReaderBuilder, ChangelogReader, changelog_arrow_fields};
use crate::memory::tests::new_memory_catalog;
use crate::metadata_columns::{
    RESERVED_FIELD_ID_CHANGE_ORDINAL, RESERVED_FIELD_ID_CHANGE_TYPE,
    RESERVED_FIELD_ID_COMMIT_SNAPSHOT_ID,
};
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, Literal, Struct,
    TableMetadata,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{Catalog, TableCreation, TableIdent};

async fn minimal_table(catalog: &impl Catalog) -> Table {
    let table_ident =
        TableIdent::from_strs([format!("ns-{}", uuid::Uuid::new_v4()), "t".to_string()]).unwrap();
    catalog
        .create_namespace(table_ident.namespace(), HashMap::new())
        .await
        .unwrap();
    let file = File::open(format!(
        "{}/testdata/table_metadata/TableMetadataV3ValidMinimal.json",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let base_metadata = serde_json::from_reader::<_, TableMetadata>(BufReader::new(file)).unwrap();
    let table_creation = TableCreation::builder()
        .schema((**base_metadata.current_schema()).clone())
        .partition_spec((**base_metadata.default_partition_spec()).clone())
        .sort_order((**base_metadata.default_sort_order()).clone())
        .name(table_ident.name().to_string())
        .format_version(FormatVersion::V3)
        .build();
    catalog
        .create_table(table_ident.namespace(), table_creation)
        .await
        .unwrap()
}

async fn write_rows(table: &Table, name: &str, ys: &[i64]) -> DataFile {
    let path = format!("{}/{name}", table.metadata().location());
    let schema = Arc::new(ArrowSchema::new(vec![
        field("x", 1),
        field("y", 2),
        field("z", 3),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from_iter_values(vec![1_i64; ys.len()])),
        Arc::new(Int64Array::from_iter_values(ys.to_vec())),
        Arc::new(Int64Array::from_iter_values(
            ys.iter().map(|value| value * 10).collect::<Vec<i64>>(),
        )),
    ];
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
    let mut buffer: Vec<u8> = Vec::new();
    let mut writer = ArrowWriter::try_new(
        &mut buffer,
        Arc::clone(&schema),
        Some(
            WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .build(),
        ),
    )
    .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let size = buffer.len() as u64;
    table
        .file_io()
        .new_output(&path)
        .unwrap()
        .write(Bytes::from(buffer))
        .await
        .unwrap();
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path)
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(size)
        .record_count(ys.len() as u64)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::long(1))]))
        .build()
        .unwrap()
}

fn field(name: &str, field_id: i32) -> Field {
    Field::new(name, DataType::Int64, false).with_metadata(HashMap::from([(
        PARQUET_FIELD_ID_META_KEY.to_string(),
        field_id.to_string(),
    )]))
}

async fn append(catalog: &impl Catalog, table: &Table, file: DataFile) -> Table {
    let tx = Transaction::new(table);
    let action = tx.fast_append().add_data_files(vec![file]);
    let tx = action.apply(tx).unwrap();
    tx.commit(catalog).await.unwrap()
}

fn column_i64(batch: &RecordBatch, name: &str) -> Vec<i64> {
    batch
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec()
}

fn column_i32(batch: &RecordBatch, name: &str) -> Vec<i32> {
    batch
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .values()
        .to_vec()
}

fn column_str(batch: &RecordBatch, name: &str) -> Vec<String> {
    let values = batch
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    (0..values.len())
        .map(|row| values.value(row).to_string())
        .collect()
}

#[tokio::test]
async fn changelog_rows_carry_the_window_snapshot_of_their_own_ordinal() {
    let catalog = new_memory_catalog().await;
    let table = minimal_table(&catalog).await;
    let base = write_rows(&table, "base.parquet", &[2]).await;
    let table = append(&catalog, &table, base).await;
    let s0 = table.metadata().current_snapshot_id().unwrap();
    let first = write_rows(&table, "a.parquet", &[3, 4]).await;
    let table = append(&catalog, &table, first).await;
    let s1 = table.metadata().current_snapshot_id().unwrap();
    let second = write_rows(&table, "b.parquet", &[5]).await;
    let table = append(&catalog, &table, second).await;
    let s2 = table.metadata().current_snapshot_id().unwrap();

    let scan = table
        .incremental_changelog_scan()
        .from_snapshot_id_exclusive(s0)
        .to_snapshot_id(s2)
        .build()
        .unwrap();
    let batches: Vec<RecordBatch> =
        ChangelogReader::new(ArrowReaderBuilder::new(table.file_io().clone()).build())
            .read(scan.plan_files().await.unwrap())
            .unwrap()
            .try_collect()
            .await
            .unwrap();

    let mut seen: Vec<(i64, String, i32, i64)> = Vec::new();
    for batch in &batches {
        let ys = column_i64(batch, "y");
        let types = column_str(batch, "_change_type");
        let ordinals = column_i32(batch, "_change_ordinal");
        let commits = column_i64(batch, "_commit_snapshot_id");
        for row in 0..batch.num_rows() {
            seen.push((ys[row], types[row].clone(), ordinals[row], commits[row]));
        }
    }
    seen.sort();
    assert_eq!(seen, vec![
        (3, "INSERT".to_string(), 0, s1),
        (4, "INSERT".to_string(), 0, s1),
        (5, "INSERT".to_string(), 1, s2),
    ]);
    assert_ne!(s1, s2, "the two ordinals must be different snapshots");
}

#[tokio::test]
async fn a_deleted_data_file_reads_as_delete_rows_of_its_commit_snapshot() {
    let catalog = new_memory_catalog().await;
    let table = minimal_table(&catalog).await;
    let base = write_rows(&table, "base.parquet", &[2]).await;
    let table = append(&catalog, &table, base).await;
    let s0 = table.metadata().current_snapshot_id().unwrap();
    let kept = write_rows(&table, "a.parquet", &[3]).await;
    let a_path = kept.file_path().to_string();
    let table = append(&catalog, &table, kept).await;
    let s1 = table.metadata().current_snapshot_id().unwrap();

    let replacement = write_rows(&table, "c.parquet", &[9]).await;
    let tx = Transaction::new(&table);
    let action = tx
        .overwrite_files()
        .delete_file(a_path)
        .add_file(replacement);
    let tx = action.apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();
    let s2 = table.metadata().current_snapshot_id().unwrap();

    let scan = table
        .incremental_changelog_scan()
        .from_snapshot_id_exclusive(s0)
        .to_snapshot_id(s2)
        .build()
        .unwrap();
    let batches: Vec<RecordBatch> =
        ChangelogReader::new(ArrowReaderBuilder::new(table.file_io().clone()).build())
            .read(scan.plan_files().await.unwrap())
            .unwrap()
            .try_collect()
            .await
            .unwrap();

    let mut seen: Vec<(i64, String, i32, i64)> = Vec::new();
    for batch in &batches {
        let ys = column_i64(batch, "y");
        let types = column_str(batch, "_change_type");
        let ordinals = column_i32(batch, "_change_ordinal");
        let commits = column_i64(batch, "_commit_snapshot_id");
        for row in 0..batch.num_rows() {
            seen.push((ys[row], types[row].clone(), ordinals[row], commits[row]));
        }
    }
    seen.sort();
    assert_eq!(seen, vec![
        (3, "DELETE".to_string(), 1, s2),
        (3, "INSERT".to_string(), 0, s1),
        (9, "INSERT".to_string(), 1, s2),
    ]);
}

#[test]
fn the_three_reserved_columns_keep_their_reserved_field_ids() {
    let ids: Vec<String> = changelog_arrow_fields()
        .iter()
        .map(|field| field.metadata()[PARQUET_FIELD_ID_META_KEY].clone())
        .collect();
    assert_eq!(ids, vec![
        RESERVED_FIELD_ID_CHANGE_TYPE.to_string(),
        RESERVED_FIELD_ID_CHANGE_ORDINAL.to_string(),
        RESERVED_FIELD_ID_COMMIT_SNAPSHOT_ID.to_string(),
    ]);
}
