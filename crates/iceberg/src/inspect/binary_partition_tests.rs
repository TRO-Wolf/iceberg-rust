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
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{RecordBatch, StructArray};
use arrow_schema::DataType;
use futures::TryStreamExt;
use tempfile::TempDir;

use super::{EntriesTable, FilesTable, PartitionsTable};
use crate::io::LocalFsStorageFactory;
use crate::memory::MemoryCatalogBuilder;
use crate::scan::ArrowRecordBatchStream;
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, Literal,
    NestedField, PartitionSpec, PrimitiveType, Schema, Struct, Transform, Type,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};

fn truncated_binary_table_creation() -> TableCreation {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "b", Type::Primitive(PrimitiveType::Binary)).into(),
        ])
        .build()
        .expect("binary partition source schema");
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("b", "b_trunc", Transform::Truncate(1))
        .expect("truncate[1](b)")
        .build()
        .expect("spec");
    TableCreation::builder()
        .name("t".to_string())
        .schema(schema)
        .partition_spec(spec)
        .format_version(FormatVersion::V2)
        .build()
}

fn partition_file(location: &str, name: &str, partition: &[u8], records: u64) -> DataFile {
    DataFileBuilder::default()
        .partition_spec_id(0)
        .content(DataContentType::Data)
        .file_path(format!("{location}/data/{name}"))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(1024)
        .record_count(records)
        .partition(Struct::from_iter([Some(Literal::binary(
            partition.to_vec(),
        ))]))
        .build()
        .expect("data file")
}

async fn truncated_binary_table() -> (Table, TempDir) {
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
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let table = catalog
        .create_table(&namespace, truncated_binary_table_creation())
        .await
        .expect("create table");
    let location = table.metadata().location().to_string();
    let files = vec![
        partition_file(&location, "a.parquet", b"a", 2),
        partition_file(&location, "z.parquet", b"z", 3),
    ];
    let transaction = Transaction::new(&table);
    let action = transaction.fast_append().add_data_files(files);
    let table = action
        .apply(transaction)
        .expect("apply append")
        .commit(&catalog)
        .await
        .expect("commit");
    (table, temp_dir)
}

async fn scan_single_batch(stream: ArrowRecordBatchStream) -> RecordBatch {
    let batches: Vec<_> = stream.try_collect().await.expect("collect scan");
    arrow_select::concat::concat_batches(&batches[0].schema(), &batches).expect("concat")
}

fn binary_partition_values(partition: &StructArray) -> Vec<Vec<u8>> {
    assert_eq!(partition.column(0).data_type(), &DataType::LargeBinary);
    let values = partition.column(0).as_binary::<i64>();
    let mut out: Vec<Vec<u8>> = values
        .iter()
        .map(|value| value.expect("non-null partition value").to_vec())
        .collect();
    out.sort_unstable();
    out
}

#[tokio::test]
async fn inspect_partitions_answers_truncated_binary_partition_rows() {
    let (table, _guard) = truncated_binary_table().await;
    let batch = scan_single_batch(
        PartitionsTable::new(&table)
            .scan()
            .await
            .expect("partitions scan"),
    )
    .await;
    assert_eq!(batch.num_rows(), 2);
    let partition = batch
        .column_by_name("partition")
        .expect("partition")
        .as_struct();
    assert_eq!(binary_partition_values(partition), vec![
        b"a".to_vec(),
        b"z".to_vec()
    ]);
    let values = partition.column(0).as_binary::<i64>();
    let record_count = batch
        .column_by_name("record_count")
        .expect("record_count")
        .as_primitive::<Int64Type>();
    let file_count = batch
        .column_by_name("file_count")
        .expect("file_count")
        .as_primitive::<Int32Type>();
    let mut rows: Vec<(Vec<u8>, i64, i32)> = (0..batch.num_rows())
        .map(|index| {
            (
                values.value(index).to_vec(),
                record_count.value(index),
                file_count.value(index),
            )
        })
        .collect();
    rows.sort_unstable();
    assert_eq!(rows, vec![(b"a".to_vec(), 2, 1), (b"z".to_vec(), 3, 1)]);
}

#[tokio::test]
async fn inspect_files_answers_truncated_binary_partition_values() {
    let (table, _guard) = truncated_binary_table().await;
    let batch = scan_single_batch(FilesTable::all(&table).scan().await.expect("files scan")).await;
    assert_eq!(batch.num_rows(), 2);
    let partition = batch
        .column_by_name("partition")
        .expect("partition")
        .as_struct();
    assert_eq!(binary_partition_values(partition), vec![
        b"a".to_vec(),
        b"z".to_vec()
    ]);
}

#[tokio::test]
async fn inspect_entries_answers_truncated_binary_partition_values() {
    let (table, _guard) = truncated_binary_table().await;
    let batch = scan_single_batch(
        EntriesTable::new(&table)
            .scan()
            .await
            .expect("entries scan"),
    )
    .await;
    assert_eq!(batch.num_rows(), 2);
    let partition = batch
        .column_by_name("data_file")
        .expect("data_file")
        .as_struct()
        .column_by_name("partition")
        .expect("partition")
        .as_struct();
    assert_eq!(binary_partition_values(partition), vec![
        b"a".to_vec(),
        b"z".to_vec()
    ]);
}
