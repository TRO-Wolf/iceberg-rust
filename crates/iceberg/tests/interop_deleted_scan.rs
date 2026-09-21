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
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use iceberg::expr::Reference;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::metadata_columns::{
    RESERVED_COL_NAME_DELETED, RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_POS,
    RESERVED_FIELD_ID_DELETED,
};
use iceberg::scan::FileScanTask;
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, Datum, FormatVersion, NestedField, PrimitiveType,
    Schema, SortOrder, Struct, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::equality_delete_writer::{
    EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
};
use iceberg::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
};
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use tempfile::TempDir;

fn gen_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build the {id long, data string} schema")
}

async fn empty_table(name: &str) -> (TempDir, MemoryCatalog, Table) {
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
    let namespace = NamespaceIdent::new("deleted_scan".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location)
        .schema(gen_schema())
        .partition_spec(UnboundPartitionSpec::builder().build())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create rust_table");
    (tmp, catalog, table)
}

async fn write_data_file(table: &Table, ids: &[i64]) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema to arrow"));
    let data: Vec<String> = ids.iter().map(|id| format!("d{id}")).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
        Arc::new(StringArray::from(data)) as ArrayRef,
    ])
    .expect("build the data batch");
    let file_path = format!(
        "{}/data/00000-deleted-scan-data.parquet",
        table.metadata().location()
    );
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
        .partition(Struct::empty())
        .build()
        .expect("build unpartitioned data file")
}

async fn append_data(catalog: &MemoryCatalog, table: &Table, ids: &[i64]) -> Table {
    let data_file = write_data_file(table, ids).await;
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![data_file])
        .apply(tx)
        .expect("apply fast append");
    tx.commit(catalog).await.expect("commit fast append")
}

async fn add_deletes(catalog: &MemoryCatalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .row_delta()
        .add_deletes(files)
        .apply(tx)
        .expect("apply row delta");
    tx.commit(catalog).await.expect("commit row delta")
}

fn decode_file_path(col: &ArrayRef, i: usize) -> String {
    use arrow_array::RunArray;
    use arrow_array::types::Int32Type;

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

async fn discover_positions(table: &Table, target_ids: &[i64]) -> Vec<(String, i64)> {
    let batches: Vec<RecordBatch> = table
        .scan()
        .select(["id", RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_POS])
        .build()
        .expect("build identity scan")
        .to_arrow()
        .await
        .expect("identity scan to_arrow")
        .try_collect()
        .await
        .expect("collect identity batches");
    let mut pairs = Vec::new();
    for batch in &batches {
        let id_col = batch
            .column_by_name("id")
            .expect("id column")
            .as_primitive::<Int64Type>();
        let file_col = batch
            .column_by_name(RESERVED_COL_NAME_FILE)
            .expect("_file column");
        let pos_col = batch
            .column_by_name(RESERVED_COL_NAME_POS)
            .expect("_pos column")
            .as_primitive::<Int64Type>();
        for i in 0..batch.num_rows() {
            if target_ids.contains(&id_col.value(i)) {
                pairs.push((decode_file_path(file_col, i), pos_col.value(i)));
            }
        }
    }
    pairs
}

async fn write_pos_deletes(table: &Table, pairs: &[(String, i64)]) -> DataFile {
    let config = PositionDeleteWriterConfig::new().expect("position-delete writer config");
    let location_gen = DefaultLocationGenerator::new(table.metadata()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "pos-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        config.schema().clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .unpartitioned()
        .build(None)
        .await
        .expect("build position-delete writer");
    let paths: Vec<&str> = pairs.iter().map(|(path, _)| path.as_str()).collect();
    let positions: Vec<i64> = pairs.iter().map(|(_, pos)| *pos).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions)) as ArrayRef,
    ])
    .expect("build position-delete batch");
    writer
        .write(batch)
        .await
        .expect("write position-delete batch");
    writer
        .close()
        .await
        .expect("close position-delete writer")
        .into_iter()
        .next()
        .expect("one position-delete file")
}

async fn write_eq_deletes(table: &Table, ids: &[i64]) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let config = EqualityDeleteWriterConfig::new(vec![1], schema.clone())
        .expect("equality-delete writer config");
    let location_gen = DefaultLocationGenerator::new(table.metadata()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "eq-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let projected_iceberg_schema = Arc::new(
        iceberg::arrow::arrow_schema_to_schema(config.projected_arrow_schema_ref())
            .expect("projected arrow schema to iceberg schema"),
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        projected_iceberg_schema,
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .unpartitioned()
        .build(None)
        .await
        .expect("build equality-delete writer");
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema to arrow"));
    let data: Vec<&str> = std::iter::repeat_n("x", ids.len()).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
        Arc::new(StringArray::from(data)) as ArrayRef,
    ])
    .expect("build the equality-delete key batch");
    writer
        .write(batch)
        .await
        .expect("write equality-delete batch");
    writer
        .close()
        .await
        .expect("close equality-delete writer")
        .into_iter()
        .next()
        .expect("one equality-delete file")
}

async fn scan_batches(table: &Table, columns: Option<&[&str]>) -> Vec<RecordBatch> {
    let builder = table.scan();
    let builder = match columns {
        Some(cols) => builder.select(cols.iter().copied()),
        None => builder.select_all(),
    };
    builder
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect batches")
}

fn id_deleted_pairs(batches: &[RecordBatch]) -> Vec<(i64, bool)> {
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .expect("id column")
            .as_primitive::<Int64Type>();
        let flags = batch
            .column_by_name(RESERVED_COL_NAME_DELETED)
            .expect("_deleted column")
            .as_boolean();
        assert_eq!(
            flags.null_count(),
            0,
            "_deleted is a required boolean and must never be null"
        );
        for i in 0..batch.num_rows() {
            rows.push((ids.value(i), flags.value(i)));
        }
    }
    rows.sort();
    rows
}

fn data_deleted_pairs(batches: &[RecordBatch]) -> Vec<(String, bool)> {
    let mut rows = Vec::new();
    for batch in batches {
        let data = batch
            .column_by_name("data")
            .expect("data column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("data is Utf8");
        let flags = batch
            .column_by_name(RESERVED_COL_NAME_DELETED)
            .expect("_deleted column")
            .as_boolean();
        for i in 0..batch.num_rows() {
            rows.push((data.value(i).to_string(), flags.value(i)));
        }
    }
    rows.sort();
    rows
}

fn id_list(batches: &[RecordBatch]) -> Vec<i64> {
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .expect("id column")
            .as_primitive::<Int64Type>();
        for i in 0..batch.num_rows() {
            rows.push(ids.value(i));
        }
    }
    rows.sort();
    rows
}

fn id_pos_pairs(batches: &[RecordBatch]) -> Vec<(i64, i64)> {
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .expect("id column")
            .as_primitive::<Int64Type>();
        let pos = batch
            .column_by_name(RESERVED_COL_NAME_POS)
            .expect("_pos column")
            .as_primitive::<Int64Type>();
        for i in 0..batch.num_rows() {
            rows.push((ids.value(i), pos.value(i)));
        }
    }
    rows.sort();
    rows
}

fn assert_one_deleted_column(batches: &[RecordBatch]) {
    for batch in batches {
        let count = batch
            .schema()
            .fields()
            .iter()
            .filter(|f| f.name() == RESERVED_COL_NAME_DELETED)
            .count();
        assert_eq!(count, 1, "exactly one _deleted column per batch");
    }
}

fn assert_no_deleted_column(batches: &[RecordBatch]) {
    for batch in batches {
        assert!(
            batch.column_by_name(RESERVED_COL_NAME_DELETED).is_none(),
            "no _deleted column without an explicit projection"
        );
    }
}

async fn mor_table_1_deleted() -> (TempDir, MemoryCatalog, Table) {
    let (tmp, catalog, table) = empty_table("deleted_mor").await;
    let table = append_data(&catalog, &table, &[1, 2, 3, 4]).await;
    let mut pairs = discover_positions(&table, &[1]).await;
    pairs.sort();
    assert_eq!(pairs.len(), 1, "id 1 has exactly one row identity");
    assert_eq!(pairs[0].1, 0, "id 1 sits at file position 0");
    let delete_file = write_pos_deletes(&table, &pairs).await;
    assert_eq!(delete_file.content_type(), DataContentType::PositionDeletes);
    let table = add_deletes(&catalog, &table, vec![delete_file]).await;
    (tmp, catalog, table)
}

#[tokio::test]
async fn mor_projecting_deleted_returns_deleted_rows_marked_true() {
    let (_tmp, _catalog, table) = mor_table_1_deleted().await;
    let batches = scan_batches(&table, Some(&["id", RESERVED_COL_NAME_DELETED])).await;
    assert_one_deleted_column(&batches);
    assert_eq!(id_deleted_pairs(&batches), vec![
        (1, true),
        (2, false),
        (3, false),
        (4, false)
    ],);
}

#[tokio::test]
async fn test_not_projecting_deleted_still_filters_deletes() {
    let (_tmp, _catalog, table) = mor_table_1_deleted().await;
    let batches = scan_batches(&table, Some(&["id"])).await;
    assert_no_deleted_column(&batches);
    assert_eq!(id_list(&batches), vec![2, 3, 4]);
}

#[tokio::test]
async fn cow_projecting_deleted_returns_live_rows_all_false() {
    let (_tmp, catalog, table) = empty_table("deleted_cow").await;
    let table = append_data(&catalog, &table, &[1, 2, 3, 4]).await;
    let batches = scan_batches(&table, Some(&["id", RESERVED_COL_NAME_DELETED])).await;
    assert_one_deleted_column(&batches);
    assert_eq!(id_deleted_pairs(&batches), vec![
        (1, false),
        (2, false),
        (3, false),
        (4, false)
    ],);
}

#[tokio::test]
async fn pos_without_deleted_keeps_filter_semantics() {
    let (_tmp, _catalog, table) = mor_table_1_deleted().await;
    let batches = scan_batches(&table, Some(&["id", RESERVED_COL_NAME_POS])).await;
    assert_no_deleted_column(&batches);
    assert_eq!(id_pos_pairs(&batches), vec![(2, 1), (3, 2), (4, 3)]);
}

#[tokio::test]
async fn select_all_on_mor_table_returns_live_rows_without_deleted_column() {
    let (_tmp, _catalog, table) = mor_table_1_deleted().await;
    let batches = scan_batches(&table, None).await;
    assert_no_deleted_column(&batches);
    assert_eq!(id_list(&batches), vec![2, 3, 4]);
    for batch in &batches {
        let schema = batch.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["id", "data"]);
    }
}

#[tokio::test]
async fn table_without_delete_files_unchanged_for_every_projection_shape() {
    let (_tmp, catalog, table) = empty_table("deleted_nodeletes").await;
    let table = append_data(&catalog, &table, &[1, 2, 3, 4]).await;
    let batches = scan_batches(&table, Some(&["id"])).await;
    assert_eq!(id_list(&batches), vec![1, 2, 3, 4]);
    let batches = scan_batches(&table, Some(&["id", RESERVED_COL_NAME_DELETED])).await;
    assert_eq!(id_deleted_pairs(&batches), vec![
        (1, false),
        (2, false),
        (3, false),
        (4, false)
    ],);
    let batches = scan_batches(&table, Some(&["id", RESERVED_COL_NAME_POS])).await;
    assert_eq!(id_pos_pairs(&batches), vec![(1, 0), (2, 1), (3, 2), (4, 3)],);
    let batches = scan_batches(&table, None).await;
    assert_eq!(id_list(&batches), vec![1, 2, 3, 4]);
}

#[test]
fn deleted_projecting_task_still_splits() {
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .expect("schema builds"),
    );
    let mut task = FileScanTask {
        file_size_in_bytes: 1000,
        start: 0,
        length: 1000,
        record_count: Some(1000),
        file_record_count: Some(1000),
        data_file_path: Arc::from("memory://t/data/1.parquet"),
        data_file_format: DataFileFormat::Parquet,
        schema,
        project_field_ids: Arc::from(vec![1]),
        predicate: None,
        deletes: Arc::from(vec![]),
        partition: None,
        partition_spec: None,
        name_mapping: None,
        case_sensitive: true,
        split_offsets: None,
        first_row_id: None,
        file_sequence_number: None,
    };
    task.project_field_ids = Arc::from(vec![1, RESERVED_FIELD_ID_DELETED]);
    assert!(
        task.split(200).expect("split ok").len() > 1,
        "`_deleted` is served over a ranged window exactly as the whole file serves it, so a \
         `_deleted`-projecting task must still split"
    );
}

#[tokio::test]
async fn equality_delete_projecting_deleted_marks_deleted_row_true() {
    let (_tmp, catalog, table) = empty_table("deleted_eq").await;
    let table = append_data(&catalog, &table, &[1, 2, 3, 4]).await;
    let delete_file = write_eq_deletes(&table, &[3]).await;
    assert_eq!(delete_file.content_type(), DataContentType::EqualityDeletes);
    let table = add_deletes(&catalog, &table, vec![delete_file]).await;
    let batches = scan_batches(&table, Some(&["id", RESERVED_COL_NAME_DELETED])).await;
    assert_eq!(id_deleted_pairs(&batches), vec![
        (1, false),
        (2, false),
        (3, true),
        (4, false)
    ],);
    let batches = scan_batches(&table, Some(&["data", RESERVED_COL_NAME_DELETED])).await;
    assert_eq!(data_deleted_pairs(&batches), vec![
        ("d1".to_string(), false),
        ("d2".to_string(), false),
        ("d3".to_string(), true),
        ("d4".to_string(), false),
    ],);
}

#[tokio::test]
async fn pos_and_deleted_together_mark_victims_with_true_ordinals() {
    let (_tmp, _catalog, table) = mor_table_1_deleted().await;
    let batches = scan_batches(
        &table,
        Some(&["id", RESERVED_COL_NAME_POS, RESERVED_COL_NAME_DELETED]),
    )
    .await;
    assert_one_deleted_column(&batches);
    let mut rows = Vec::new();
    for batch in &batches {
        let ids = batch
            .column_by_name("id")
            .expect("id column")
            .as_primitive::<Int64Type>();
        let pos = batch
            .column_by_name(RESERVED_COL_NAME_POS)
            .expect("_pos column")
            .as_primitive::<Int64Type>();
        let flags = batch
            .column_by_name(RESERVED_COL_NAME_DELETED)
            .expect("_deleted column")
            .as_boolean();
        for i in 0..batch.num_rows() {
            rows.push((ids.value(i), pos.value(i), flags.value(i)));
        }
    }
    rows.sort();
    assert_eq!(rows, vec![
        (1, 0, true),
        (2, 1, false),
        (3, 2, false),
        (4, 3, false)
    ],);
}

#[tokio::test]
async fn predicate_with_deleted_still_filters() {
    let (_tmp, _catalog, table) = mor_table_1_deleted().await;
    let batches: Vec<RecordBatch> = table
        .scan()
        .select(["id", RESERVED_COL_NAME_DELETED])
        .with_filter(Reference::new("id").greater_than_or_equal_to(Datum::long(2)))
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect batches");
    assert_eq!(id_deleted_pairs(&batches), vec![
        (2, false),
        (3, false),
        (4, false)
    ],);
    let batches: Vec<RecordBatch> = table
        .scan()
        .select(["id", RESERVED_COL_NAME_DELETED])
        .with_filter(Reference::new("id").equal_to(Datum::long(1)))
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect batches");
    assert_eq!(id_deleted_pairs(&batches), vec![(1, true)]);
}
