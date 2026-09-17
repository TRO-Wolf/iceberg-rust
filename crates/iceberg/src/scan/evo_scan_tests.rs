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

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use tempfile::TempDir;

use crate::arrow::schema_to_arrow_schema;
use crate::io::LocalFsStorageFactory;
use crate::memory::MemoryCatalogBuilder;
use crate::spec::{
    DataContentType, DataFile, FormatVersion, NestedField, PrimitiveType, Schema, Struct, Type,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
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

async fn commit(catalog: &impl Catalog, tx: Transaction) -> Table {
    tx.commit(catalog).await.expect("commit")
}

async fn create_table(catalog: &impl Catalog, fields: Vec<Arc<NestedField>>) -> Table {
    let schema = Schema::builder()
        .with_fields(fields)
        .build()
        .expect("evo schema");
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("t".to_string())
        .schema(schema)
        .format_version(FormatVersion::V2)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

async fn write_string_file(
    table: &Table,
    name: &str,
    ids: &[i64],
    columns: &[&[&str]],
) -> DataFile {
    let schema = table.metadata().current_schema().clone();
    let arrow_schema = Arc::new(schema_to_arrow_schema(&schema).expect("arrow schema"));
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(ids.to_vec()))];
    for column in columns {
        arrays.push(Arc::new(StringArray::from(
            column
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>(),
        )));
    }
    let batch = RecordBatch::try_new(arrow_schema, arrays).expect("batch");
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
    let mut builder = writer
        .close()
        .await
        .expect("close")
        .into_iter()
        .next()
        .expect("one data file");
    builder
        .content(DataContentType::Data)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .expect("data file")
}

async fn append(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let action = tx.fast_append().add_data_files(files);
    commit(catalog, action.apply(tx).expect("apply append")).await
}

fn string_field(id: i32, name: &str) -> Arc<NestedField> {
    NestedField::optional(id, name, Type::Primitive(PrimitiveType::String)).into()
}

fn long_values(batch: &RecordBatch, name: &str) -> Vec<i64> {
    let values = batch
        .column_by_name(name)
        .expect("long column")
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("long type");
    (0..values.len()).map(|index| values.value(index)).collect()
}

fn string_values(batch: &RecordBatch, name: &str) -> Vec<Option<String>> {
    let values = batch
        .column_by_name(name)
        .expect("string column")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("string type");
    (0..values.len())
        .map(|index| {
            if values.is_null(index) {
                None
            } else {
                Some(values.value(index).to_string())
            }
        })
        .collect()
}

async fn scan_batches(table: &Table, columns: &[&str]) -> Vec<RecordBatch> {
    table
        .scan()
        .select(columns.iter().map(|name| name.to_string()))
        .build()
        .expect("scan")
        .to_arrow()
        .await
        .expect("to_arrow")
        .try_collect()
        .await
        .expect("collect")
}

async fn added_column_table() -> (Table, TempDir) {
    let (catalog, guard) = local_catalog().await;
    let table = create_table(&catalog, vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
        string_field(2, "v"),
    ])
    .await;
    let first = write_string_file(&table, "f1.parquet", &[1], &[&["a"]]).await;
    let second = write_string_file(&table, "f2.parquet", &[2], &[&["b"]]).await;
    let table = append(&catalog, &table, vec![first, second]).await;
    let tx = Transaction::new(&table);
    let action = tx
        .update_schema()
        .add_column("extra", Type::Primitive(PrimitiveType::String));
    (
        commit(&catalog, action.apply(tx).expect("apply add column")).await,
        guard,
    )
}

async fn renamed_column_table() -> (Table, TempDir) {
    let (catalog, guard) = local_catalog().await;
    let table = create_table(&catalog, vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
        string_field(2, "w"),
    ])
    .await;
    let first = write_string_file(&table, "f1.parquet", &[1], &[&["a"]]).await;
    let second = write_string_file(&table, "f2.parquet", &[2], &[&["b"]]).await;
    let table = append(&catalog, &table, vec![first, second]).await;
    let tx = Transaction::new(&table);
    let action = tx.update_schema().rename_column("w", "v");
    (
        commit(&catalog, action.apply(tx).expect("apply rename")).await,
        guard,
    )
}

async fn swapped_names_table() -> (Table, TempDir) {
    let (catalog, guard) = local_catalog().await;
    let table = create_table(&catalog, vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
        string_field(2, "v"),
        string_field(3, "extra"),
    ])
    .await;
    let first = write_string_file(&table, "f1.parquet", &[1], &[&["a"], &["e1"]]).await;
    let second = write_string_file(&table, "f2.parquet", &[2], &[&["b"], &["e2"]]).await;
    let table = append(&catalog, &table, vec![first, second]).await;
    let tx = Transaction::new(&table);
    let action = tx.update_schema().rename_column("v", "tmp");
    let table = commit(&catalog, action.apply(tx).expect("apply first rename")).await;
    let tx = Transaction::new(&table);
    let action = tx.update_schema().rename_column("extra", "v");
    let table = commit(&catalog, action.apply(tx).expect("apply second rename")).await;
    let tx = Transaction::new(&table);
    let action = tx.update_schema().rename_column("tmp", "extra");
    (
        commit(&catalog, action.apply(tx).expect("apply swap")).await,
        guard,
    )
}

#[tokio::test]
async fn unpinned_scan_after_add_column_null_fills_the_added_column() {
    let (table, _guard) = added_column_table().await;
    let batches = scan_batches(&table, &["id", "v", "extra"]).await;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    let mut rows = vec![];
    for batch in &batches {
        let ids = long_values(batch, "id");
        let names = string_values(batch, "v");
        let extras = string_values(batch, "extra");
        for index in 0..ids.len() {
            rows.push((ids[index], names[index].clone(), extras[index].clone()));
        }
    }
    rows.sort();
    assert_eq!(rows, vec![
        (1, Some("a".to_string()), None),
        (2, Some("b".to_string()), None),
    ]);
}

#[tokio::test]
async fn unpinned_scan_after_rename_reads_the_renamed_column_by_field_id() {
    let (table, _guard) = renamed_column_table().await;
    let batches = scan_batches(&table, &["id", "v"]).await;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    let mut rows = vec![];
    for batch in &batches {
        let ids = long_values(batch, "id");
        let names = string_values(batch, "v");
        for index in 0..ids.len() {
            rows.push((ids[index], names[index].clone()));
        }
    }
    rows.sort();
    assert_eq!(rows, vec![
        (1, Some("a".to_string())),
        (2, Some("b".to_string())),
    ]);
}

#[tokio::test]
async fn unpinned_scan_after_swapping_two_names_reads_each_field_by_id() {
    let (table, _guard) = swapped_names_table().await;
    let batches = scan_batches(&table, &["id", "v", "extra"]).await;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    let mut rows = vec![];
    for batch in &batches {
        let ids = long_values(batch, "id");
        let renamed = string_values(batch, "v");
        let swapped = string_values(batch, "extra");
        for index in 0..ids.len() {
            rows.push((ids[index], renamed[index].clone(), swapped[index].clone()));
        }
    }
    rows.sort();
    assert_eq!(rows, vec![
        (1, Some("e1".to_string()), Some("a".to_string())),
        (2, Some("e2".to_string()), Some("b".to_string())),
    ]);
}

#[tokio::test]
async fn snapshot_pinned_select_of_an_added_column_fails() {
    let (table, _guard) = added_column_table().await;
    let snapshot_id = table.metadata().current_snapshot_id().expect("snapshot");
    let error = table
        .scan()
        .snapshot_id(snapshot_id)
        .select(["extra"])
        .build()
        .expect_err("added column is absent from the snapshot schema");
    assert!(
        error.to_string().contains("Column extra not found"),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn snapshot_pinned_select_of_a_pre_rename_name_reads_the_field() {
    let (table, _guard) = renamed_column_table().await;
    let snapshot_id = table.metadata().current_snapshot_id().expect("snapshot");
    let batches: Vec<RecordBatch> = table
        .scan()
        .snapshot_id(snapshot_id)
        .select(["id", "w"])
        .build()
        .expect("pinned scan")
        .to_arrow()
        .await
        .expect("to_arrow")
        .try_collect()
        .await
        .expect("collect");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    let mut rows = vec![];
    for batch in &batches {
        let ids = long_values(batch, "id");
        let names = string_values(batch, "w");
        for index in 0..ids.len() {
            rows.push((ids[index], names[index].clone()));
        }
    }
    rows.sort();
    assert_eq!(rows, vec![
        (1, Some("a".to_string())),
        (2, Some("b".to_string())),
    ]);
}

#[tokio::test]
async fn snapshot_pinned_scan_after_a_name_swap_reads_snapshot_names() {
    let (table, _guard) = swapped_names_table().await;
    let snapshot_id = table.metadata().current_snapshot_id().expect("snapshot");
    let batches: Vec<RecordBatch> = table
        .scan()
        .snapshot_id(snapshot_id)
        .select(["id", "v", "extra"])
        .build()
        .expect("pinned scan")
        .to_arrow()
        .await
        .expect("to_arrow")
        .try_collect()
        .await
        .expect("collect");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    let mut rows = vec![];
    for batch in &batches {
        let ids = long_values(batch, "id");
        let names = string_values(batch, "v");
        let extras = string_values(batch, "extra");
        for index in 0..ids.len() {
            rows.push((ids[index], names[index].clone(), extras[index].clone()));
        }
    }
    rows.sort();
    assert_eq!(rows, vec![
        (1, Some("a".to_string()), Some("e1".to_string())),
        (2, Some("b".to_string()), Some("e2".to_string())),
    ]);
}

async fn tagged_pre_ddl_table() -> (Table, TempDir) {
    let (catalog, guard) = local_catalog().await;
    let table = create_table(&catalog, vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
        string_field(2, "v"),
    ])
    .await;
    let first = write_string_file(&table, "f1.parquet", &[1], &[&["a"]]).await;
    let second = write_string_file(&table, "f2.parquet", &[2], &[&["b"]]).await;
    let table = append(&catalog, &table, vec![first, second]).await;
    let pre_ddl = table.metadata().current_snapshot_id().expect("snapshot");
    let tx = Transaction::new(&table);
    let table = commit(
        &catalog,
        tx.manage_snapshots()
            .create_tag("pre-ddl", pre_ddl)
            .apply(tx)
            .expect("apply create tag"),
    )
    .await;
    let tx = Transaction::new(&table);
    let action = tx
        .update_schema()
        .add_column("extra", Type::Primitive(PrimitiveType::String));
    (
        commit(&catalog, action.apply(tx).expect("apply add column")).await,
        guard,
    )
}

#[tokio::test]
async fn main_ref_scan_after_add_column_null_fills_the_added_column() {
    let (table, _guard) = added_column_table().await;
    let batches: Vec<RecordBatch> = table
        .scan()
        .use_ref("main")
        .select(["id", "v", "extra"])
        .build()
        .expect("main ref scan")
        .to_arrow()
        .await
        .expect("to_arrow")
        .try_collect()
        .await
        .expect("collect");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    let mut rows = vec![];
    for batch in &batches {
        let ids = long_values(batch, "id");
        let names = string_values(batch, "v");
        let extras = string_values(batch, "extra");
        for index in 0..ids.len() {
            rows.push((ids[index], names[index].clone(), extras[index].clone()));
        }
    }
    rows.sort();
    assert_eq!(rows, vec![
        (1, Some("a".to_string()), None),
        (2, Some("b".to_string()), None),
    ]);
}

#[tokio::test]
async fn main_ref_select_all_after_add_column_includes_the_added_column() {
    let (table, _guard) = added_column_table().await;
    let batches: Vec<RecordBatch> = table
        .scan()
        .use_ref("main")
        .select_all()
        .build()
        .expect("main ref scan")
        .to_arrow()
        .await
        .expect("to_arrow")
        .try_collect()
        .await
        .expect("collect");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    for batch in &batches {
        let extras = string_values(batch, "extra");
        assert_eq!(extras, vec![None; extras.len()]);
    }
}

#[tokio::test]
async fn main_ref_scan_after_swapping_two_names_reads_each_field_by_id() {
    let (table, _guard) = swapped_names_table().await;
    let batches: Vec<RecordBatch> = table
        .scan()
        .use_ref("main")
        .select(["id", "v", "extra"])
        .build()
        .expect("main ref scan")
        .to_arrow()
        .await
        .expect("to_arrow")
        .try_collect()
        .await
        .expect("collect");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    let mut rows = vec![];
    for batch in &batches {
        let ids = long_values(batch, "id");
        let renamed = string_values(batch, "v");
        let swapped = string_values(batch, "extra");
        for index in 0..ids.len() {
            rows.push((ids[index], renamed[index].clone(), swapped[index].clone()));
        }
    }
    rows.sort();
    assert_eq!(rows, vec![
        (1, Some("e1".to_string()), Some("a".to_string())),
        (2, Some("e2".to_string()), Some("b".to_string())),
    ]);
}

#[tokio::test]
async fn tag_ref_on_the_pre_ddl_snapshot_binds_the_snapshot_schema() {
    let (table, _guard) = tagged_pre_ddl_table().await;
    let batches: Vec<RecordBatch> = table
        .scan()
        .use_ref("pre-ddl")
        .select(["id", "v"])
        .build()
        .expect("tag ref scan")
        .to_arrow()
        .await
        .expect("to_arrow")
        .try_collect()
        .await
        .expect("collect");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    let mut rows = vec![];
    for batch in &batches {
        let ids = long_values(batch, "id");
        let names = string_values(batch, "v");
        for index in 0..ids.len() {
            rows.push((ids[index], names[index].clone()));
        }
    }
    rows.sort();
    assert_eq!(rows, vec![
        (1, Some("a".to_string())),
        (2, Some("b".to_string())),
    ]);
}
