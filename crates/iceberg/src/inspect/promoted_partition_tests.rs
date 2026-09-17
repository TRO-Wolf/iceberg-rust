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

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
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

async fn commit(catalog: &impl Catalog, transaction: Transaction) -> Table {
    transaction.commit(catalog).await.expect("commit")
}

fn identity_partitioned_table_creation() -> TableCreation {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "p", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "s", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("int partition source schema");
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("p", "p", Transform::Identity)
        .expect("identity(p)")
        .build()
        .expect("spec");
    TableCreation::builder()
        .name("t".to_string())
        .schema(schema)
        .partition_spec(spec)
        .format_version(FormatVersion::V2)
        .build()
}

fn partition_file(name: &str, location: &str, partition: Struct, records: u64) -> DataFile {
    DataFileBuilder::default()
        .partition_spec_id(0)
        .content(DataContentType::Data)
        .file_path(format!("{location}/data/{name}"))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(1024)
        .record_count(records)
        .partition(partition)
        .build()
        .expect("data file")
}

async fn mixed_era_identity_table() -> (Table, TempDir) {
    let (catalog, guard) = local_catalog().await;
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let table = catalog
        .create_table(&namespace, identity_partitioned_table_creation())
        .await
        .expect("create table");
    let location = table.metadata().location().to_string();
    let old_first = partition_file(
        "old-1.parquet",
        &location,
        Struct::from_iter([Some(Literal::int(1))]),
        1,
    );
    let old_second = partition_file(
        "old-22.parquet",
        &location,
        Struct::from_iter([Some(Literal::int(22))]),
        2,
    );
    let transaction = Transaction::new(&table);
    let action = transaction
        .fast_append()
        .add_data_files(vec![old_first, old_second]);
    let table = commit(&catalog, action.apply(transaction).expect("apply append")).await;
    let transaction = Transaction::new(&table);
    let action = transaction
        .update_schema()
        .update_column("p", PrimitiveType::Long);
    let table = commit(
        &catalog,
        action.apply(transaction).expect("apply promotion"),
    )
    .await;
    let location = table.metadata().location().to_string();
    let new = partition_file(
        "new-3e9.parquet",
        &location,
        Struct::from_iter([Some(Literal::long(3_000_000_000_i64))]),
        3,
    );
    let transaction = Transaction::new(&table);
    let action = transaction.fast_append().add_data_files(vec![new]);
    (
        commit(&catalog, action.apply(transaction).expect("apply append")).await,
        guard,
    )
}

async fn scan_single_batch(stream: ArrowRecordBatchStream) -> RecordBatch {
    let batches: Vec<_> = stream.try_collect().await.expect("collect scan");
    arrow_select::concat::concat_batches(&batches[0].schema(), &batches).expect("concat")
}

fn long_partition_values(partition: &arrow_array::StructArray) -> Vec<i64> {
    assert_eq!(
        partition.column(0).data_type(),
        &DataType::Int64,
        "partition column answers the promoted long type"
    );
    let values = partition.column(0).as_primitive::<Int64Type>();
    let mut out: Vec<i64> = (0..values.len()).map(|index| values.value(index)).collect();
    out.sort_unstable();
    out
}

#[tokio::test]
async fn inspect_files_answers_long_partitions_after_identity_source_promotion() {
    let (table, _guard) = mixed_era_identity_table().await;
    let batch = scan_single_batch(FilesTable::all(&table).scan().await.expect("files scan")).await;
    assert_eq!(batch.num_rows(), 3);
    let partition = batch
        .column_by_name("partition")
        .expect("partition")
        .as_struct();
    assert_eq!(long_partition_values(partition), vec![
        1,
        22,
        3_000_000_000_i64
    ]);
}

#[tokio::test]
async fn inspect_entries_answers_long_partitions_after_identity_source_promotion() {
    let (table, _guard) = mixed_era_identity_table().await;
    let batch = scan_single_batch(
        EntriesTable::new(&table)
            .scan()
            .await
            .expect("entries scan"),
    )
    .await;
    assert_eq!(batch.num_rows(), 3);
    let data_file = batch
        .column_by_name("data_file")
        .expect("data_file")
        .as_struct();
    let partition = data_file
        .column_by_name("partition")
        .expect("partition")
        .as_struct();
    assert_eq!(long_partition_values(partition), vec![
        1,
        22,
        3_000_000_000_i64
    ]);
}

#[tokio::test]
async fn inspect_partitions_groups_promoted_identity_partitions_into_typed_rows() {
    let (table, _guard) = mixed_era_identity_table().await;
    let batch = scan_single_batch(
        PartitionsTable::new(&table)
            .scan()
            .await
            .expect("partitions scan"),
    )
    .await;
    assert_eq!(batch.num_rows(), 3);
    let partition = batch
        .column_by_name("partition")
        .expect("partition")
        .as_struct();
    assert_eq!(long_partition_values(partition), vec![
        1,
        22,
        3_000_000_000_i64
    ]);
    let record_count = batch
        .column_by_name("record_count")
        .expect("record_count")
        .as_primitive::<Int64Type>();
    let file_count = batch
        .column_by_name("file_count")
        .expect("file_count")
        .as_primitive::<Int32Type>();
    let partition_values = partition.column(0).as_primitive::<Int64Type>();
    let mut rows: Vec<(i64, i64, i32)> = (0..batch.num_rows())
        .map(|index| {
            (
                partition_values.value(index),
                record_count.value(index),
                file_count.value(index),
            )
        })
        .collect();
    rows.sort_unstable();
    assert_eq!(rows, vec![(1, 1, 1), (22, 2, 1), (3_000_000_000_i64, 3, 1)]);
}
