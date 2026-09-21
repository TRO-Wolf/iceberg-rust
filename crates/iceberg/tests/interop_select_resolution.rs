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
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray, StructArray};
use arrow_schema::DataType;
use futures::TryStreamExt;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::metadata_columns::{
    RESERVED_COL_NAME_CHANGE_TYPE, RESERVED_COL_NAME_DELETED, RESERVED_COL_NAME_FILE,
    RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER, RESERVED_COL_NAME_PARTITION,
    RESERVED_COL_NAME_POS, RESERVED_COL_NAME_ROW_ID, RESERVED_COL_NAME_SPEC_ID,
};
use iceberg::spec::{
    DataContentType, DataFile, FormatVersion, Literal, NestedField, PrimitiveType, Schema,
    SortOrder, Struct, Transform, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation};
use tempfile::TempDir;

type FullDataRow = (i64, Option<i64>, Option<String>, Option<String>);

fn gen_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "pos", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(3, "file_path", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(4, "cat", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build the {id long, pos long, file_path string, cat string} schema")
}

fn cat_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .add_partition_field(4, "cat", Transform::Identity)
        .expect("bind identity(cat)")
        .build()
}

async fn empty_table(name: &str, version: FormatVersion) -> (TempDir, MemoryCatalog, Table) {
    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            name,
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let namespace = NamespaceIdent::new("select_resolution".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location)
        .schema(gen_schema())
        .partition_spec(cat_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(version)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create rust_table");
    (tmp, catalog, table)
}

async fn write_data_file(
    table: &Table,
    file_name: &str,
    ids: &[i64],
    poss: &[Option<i64>],
    file_paths: &[Option<&str>],
    cats: &[Option<&str>],
    partition: Struct,
) -> DataFile {
    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema to arrow"));
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
        Arc::new(Int64Array::from_iter(poss.iter().copied())) as ArrayRef,
        Arc::new(StringArray::from_iter(file_paths.iter().copied())) as ArrayRef,
        Arc::new(StringArray::from_iter(cats.iter().copied())) as ArrayRef,
    ])
    .expect("build the data batch");
    let file_path = format!("{}/data/{file_name}", table.metadata().location());
    let output = table.file_io().new_output(file_path).expect("new output");
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let mut writer = parquet_builder
        .build(output)
        .await
        .expect("build parquet writer");
    writer.write(&batch).await.expect("write data batch");
    let builders = writer.close().await.expect("close parquet writer");
    let mut builder = builders.into_iter().next().expect("one data file builder");
    builder
        .content(DataContentType::Data)
        .partition_spec_id(0)
        .partition(partition)
        .build()
        .expect("build data file")
}

async fn append_data(catalog: &MemoryCatalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(files)
        .apply(tx)
        .expect("apply fast append");
    tx.commit(catalog).await.expect("commit fast append")
}

async fn fixture_table(name: &str, version: FormatVersion) -> (TempDir, MemoryCatalog, Table) {
    let (tmp, catalog, table) = empty_table(name, version).await;
    let first = write_data_file(
        &table,
        "00000-select-a.parquet",
        &[1, 2],
        &[Some(101), Some(102)],
        &[Some("row-a1"), Some("row-a2")],
        &[Some("a"), Some("a")],
        Struct::from_iter([Some(Literal::string("a"))]),
    )
    .await;
    let second = write_data_file(
        &table,
        "00000-select-b.parquet",
        &[3],
        &[Some(203)],
        &[Some("row-b3")],
        &[Some("b")],
        Struct::from_iter([Some(Literal::string("b"))]),
    )
    .await;
    let table = append_data(&catalog, &table, vec![first, second]).await;
    (tmp, catalog, table)
}

async fn scan_batches(table: &Table, columns: &[&str]) -> Vec<RecordBatch> {
    table
        .scan()
        .select(columns.iter().copied())
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect batches")
}

async fn scan_default_projection(table: &Table) -> Vec<RecordBatch> {
    table
        .scan()
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect batches")
}

fn id_pos_data_pairs(batches: &[RecordBatch]) -> Vec<(i64, Option<i64>)> {
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .expect("id column")
            .as_primitive::<Int64Type>();
        let poss = batch
            .column_by_name("pos")
            .expect("pos column")
            .as_primitive::<Int64Type>();
        for i in 0..batch.num_rows() {
            let pos = if poss.is_null(i) {
                None
            } else {
                Some(poss.value(i))
            };
            rows.push((ids.value(i), pos));
        }
    }
    rows.sort();
    rows
}

fn id_file_path_data_pairs(batches: &[RecordBatch]) -> Vec<(i64, Option<String>)> {
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .expect("id column")
            .as_primitive::<Int64Type>();
        let paths = batch
            .column_by_name("file_path")
            .expect("file_path column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("file_path is Utf8");
        for i in 0..batch.num_rows() {
            let path = if paths.is_null(i) {
                None
            } else {
                Some(paths.value(i).to_string())
            };
            rows.push((ids.value(i), path));
        }
    }
    rows.sort();
    rows
}

fn id_full_data_rows(batches: &[RecordBatch]) -> Vec<FullDataRow> {
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .expect("id column")
            .as_primitive::<Int64Type>();
        let poss = batch
            .column_by_name("pos")
            .expect("pos column")
            .as_primitive::<Int64Type>();
        let paths = batch
            .column_by_name("file_path")
            .expect("file_path column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("file_path is Utf8");
        let cats = batch
            .column_by_name("cat")
            .expect("cat column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("cat is Utf8");
        for i in 0..batch.num_rows() {
            let pos = if poss.is_null(i) {
                None
            } else {
                Some(poss.value(i))
            };
            let path = if paths.is_null(i) {
                None
            } else {
                Some(paths.value(i).to_string())
            };
            let cat = if cats.is_null(i) {
                None
            } else {
                Some(cats.value(i).to_string())
            };
            rows.push((ids.value(i), pos, path, cat));
        }
    }
    rows.sort();
    rows
}

fn decode_file_path(col: &ArrayRef, i: usize) -> String {
    use arrow_array::RunArray;

    if let Some(plain) = col.as_any().downcast_ref::<StringArray>() {
        return plain.value(i).to_string();
    }
    if let Some(run) = col.as_any().downcast_ref::<RunArray<Int32Type>>() {
        let physical = run.get_physical_index(i);
        return run
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("_file REE values are Utf8")
            .value(physical)
            .to_string();
    }
    panic!("unexpected _file column type: {:?}", col.data_type());
}

fn decode_spec_id(col: &ArrayRef, i: usize) -> i32 {
    use arrow_array::RunArray;

    if let Some(plain) = col.as_any().downcast_ref::<arrow_array::Int32Array>() {
        return plain.value(i);
    }
    if let Some(run) = col.as_any().downcast_ref::<RunArray<Int32Type>>() {
        let physical = run.get_physical_index(i);
        return run.values().as_primitive::<Int32Type>().value(physical);
    }
    panic!("unexpected _spec_id column type: {:?}", col.data_type());
}

#[tokio::test]
async fn pos_select_serves_data_values() {
    let (_tmp, _catalog, table) = fixture_table("pos_data", FormatVersion::V2).await;
    let batches = scan_batches(&table, &["id", "pos"]).await;
    assert_eq!(id_pos_data_pairs(&batches), vec![
        (1, Some(101)),
        (2, Some(102)),
        (3, Some(203)),
    ]);
}

#[tokio::test]
async fn file_path_select_serves_data_values() {
    let (_tmp, _catalog, table) = fixture_table("file_path_data", FormatVersion::V2).await;
    let batches = scan_batches(&table, &["id", "file_path"]).await;
    assert_eq!(id_file_path_data_pairs(&batches), vec![
        (1, Some("row-a1".to_string())),
        (2, Some("row-a2".to_string())),
        (3, Some("row-b3".to_string())),
    ]);
}

#[tokio::test]
async fn default_projection_serves_every_column_as_ordinary() {
    let (_tmp, _catalog, table) = fixture_table("default_projection", FormatVersion::V2).await;
    let batches = scan_default_projection(&table).await;
    assert!(
        !batches.is_empty(),
        "expected at least one batch from the default projection"
    );
    for batch in &batches {
        let schema = batch.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["id", "pos", "file_path", "cat"]);
    }
    assert_eq!(id_full_data_rows(&batches), vec![
        (
            1,
            Some(101),
            Some("row-a1".to_string()),
            Some("a".to_string())
        ),
        (
            2,
            Some(102),
            Some("row-a2".to_string()),
            Some("a".to_string())
        ),
        (
            3,
            Some(203),
            Some("row-b3".to_string()),
            Some("b".to_string())
        ),
    ]);
}

#[tokio::test]
async fn change_type_absent_column_behavior_unchanged() {
    let (_tmp, _catalog, table) = fixture_table("change_type_pin", FormatVersion::V2).await;
    let scan = table
        .scan()
        .select([RESERVED_COL_NAME_CHANGE_TYPE])
        .build()
        .expect("selecting _change_type without such a column still plans");
    let stream = scan.to_arrow().await.expect("to_arrow still opens");
    let read: Result<Vec<RecordBatch>, _> = stream.try_collect().await;
    let err = read.expect_err("reading _change_type without such a column still fails");
    assert_eq!(err.kind(), ErrorKind::Unexpected);
    assert_eq!(err.message(), "field not found");
}

#[tokio::test]
async fn row_id_on_v3_without_column_serves_lineage_values() {
    let (_tmp, _catalog, table) = fixture_table("row_id_lineage", FormatVersion::V3).await;
    let batches = scan_batches(&table, &[
        "id",
        RESERVED_COL_NAME_ROW_ID,
        RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER,
    ])
    .await;
    let mut row_ids = Vec::new();
    let mut seqs = Vec::new();
    for batch in &batches {
        let ids = batch
            .column_by_name(RESERVED_COL_NAME_ROW_ID)
            .expect("_row_id column")
            .as_primitive::<Int64Type>();
        let updates = batch
            .column_by_name(RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER)
            .expect("_last_updated_sequence_number column")
            .as_primitive::<Int64Type>();
        for i in 0..batch.num_rows() {
            assert!(ids.is_valid(i), "v3 row must have a _row_id");
            assert!(updates.is_valid(i), "v3 row must have a last_updated_seq");
            row_ids.push(ids.value(i));
            seqs.push(updates.value(i));
        }
    }
    row_ids.sort();
    assert_eq!(row_ids, vec![0, 1, 2]);
    assert_eq!(seqs.len(), 3);
    assert!(
        seqs.iter().all(|seq| *seq == seqs[0]),
        "one append leaves one last_updated_seq"
    );
}

#[tokio::test]
async fn five_metadata_names_still_resolve() {
    let (_tmp, _catalog, table) = fixture_table("five_names", FormatVersion::V2).await;
    let batches = scan_batches(&table, &[
        "id",
        RESERVED_COL_NAME_FILE,
        RESERVED_COL_NAME_POS,
        RESERVED_COL_NAME_SPEC_ID,
        RESERVED_COL_NAME_PARTITION,
        RESERVED_COL_NAME_DELETED,
    ])
    .await;
    assert!(
        !batches.is_empty(),
        "expected at least one batch to pin the five metadata columns"
    );
    let mut id_file_pos = Vec::new();
    let mut id_spec = Vec::new();
    let mut id_partition = Vec::new();
    let mut id_deleted = Vec::new();
    for batch in &batches {
        let ids = batch
            .column_by_name("id")
            .expect("id column")
            .as_primitive::<Int64Type>();
        let files = batch
            .column_by_name(RESERVED_COL_NAME_FILE)
            .expect("_file column");
        let poss = batch
            .column_by_name(RESERVED_COL_NAME_POS)
            .expect("_pos column")
            .as_primitive::<Int64Type>();
        let specs = batch
            .column_by_name(RESERVED_COL_NAME_SPEC_ID)
            .expect("_spec_id column");
        let partitions = batch
            .column_by_name(RESERVED_COL_NAME_PARTITION)
            .expect("_partition column")
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("_partition is a struct");
        let cats = partitions
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("cat child is Utf8");
        let flags = batch
            .column_by_name(RESERVED_COL_NAME_DELETED)
            .expect("_deleted column")
            .as_boolean();
        for i in 0..batch.num_rows() {
            let id = ids.value(i);
            let file = decode_file_path(files, i);
            let basename = file.rsplit('/').next().expect("basename").to_string();
            id_file_pos.push((id, basename, poss.value(i)));
            id_spec.push((id, decode_spec_id(specs, i)));
            let cat = if cats.is_null(i) {
                None
            } else {
                Some(cats.value(i).to_string())
            };
            id_partition.push((id, partitions.is_null(i), cat));
            id_deleted.push((id, flags.value(i)));
        }
    }
    id_file_pos.sort();
    id_spec.sort();
    id_partition.sort();
    id_deleted.sort();
    assert_eq!(id_file_pos, vec![
        (1, "00000-select-a.parquet".to_string(), 0),
        (2, "00000-select-a.parquet".to_string(), 1),
        (3, "00000-select-b.parquet".to_string(), 0),
    ]);
    assert_eq!(id_spec, vec![(1, 0), (2, 0), (3, 0)]);
    assert_eq!(id_partition, vec![
        (1, false, Some("a".to_string())),
        (2, false, Some("a".to_string())),
        (3, false, Some("b".to_string())),
    ]);
    assert_eq!(id_deleted, vec![(1, false), (2, false), (3, false)]);
    for batch in &batches {
        let batch_schema = batch.schema();
        let field = batch_schema
            .field_with_name(RESERVED_COL_NAME_PARTITION)
            .expect("_partition field");
        assert!(
            matches!(field.data_type(), DataType::Struct(_)),
            "_partition must stay a struct"
        );
    }
}
