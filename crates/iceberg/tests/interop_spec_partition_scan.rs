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
use iceberg::arrow::{ArrowReaderBuilder, schema_to_arrow_schema};
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::metadata_columns::{
    RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_PARTITION, RESERVED_COL_NAME_POS,
    RESERVED_COL_NAME_SPEC_ID, RESERVED_FIELD_ID_PARTITION, partition_field,
};
use iceberg::scan::FileScanTask;
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, Literal, NestedField, PrimitiveType,
    Schema, SortOrder, Struct, StructType, Transform, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use tempfile::TempDir;

fn gen_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "cat", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "y", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("build the {id long, cat string, y long} schema")
}

fn unpartitioned_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder().build()
}

fn cat_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .add_partition_field(2, "cat", Transform::Identity)
        .expect("bind identity(cat)")
        .build()
}

fn cat_y_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .add_partition_field(2, "cat", Transform::Identity)
        .expect("bind identity(cat)")
        .add_partition_field(3, "y", Transform::Identity)
        .expect("bind identity(y)")
        .build()
}

async fn empty_table(name: &str, spec: UnboundPartitionSpec) -> (TempDir, MemoryCatalog, Table) {
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
    let namespace = NamespaceIdent::new("spec_partition_scan".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location)
        .schema(gen_schema())
        .partition_spec(spec)
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
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
    cats: &[Option<&str>],
    ys: &[Option<i64>],
    spec_id: i32,
    partition: Struct,
) -> DataFile {
    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema to arrow"));
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
        Arc::new(StringArray::from_iter(cats.iter().copied())) as ArrayRef,
        Arc::new(Int64Array::from_iter(ys.iter().copied())) as ArrayRef,
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
        .partition_spec_id(spec_id)
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

async fn evolve_add_field(catalog: &MemoryCatalog, table: &Table, source: &str) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .update_partition_spec()
        .add_field(source)
        .apply(tx)
        .expect("apply spec evolution");
    tx.commit(catalog).await.expect("commit spec evolution")
}

async fn evolve_remove_field(catalog: &MemoryCatalog, table: &Table, name: &str) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .update_partition_spec()
        .remove_field(name)
        .apply(tx)
        .expect("apply spec evolution");
    tx.commit(catalog).await.expect("commit spec evolution")
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

fn id_spec_pairs(batches: &[RecordBatch]) -> Vec<(i64, i32)> {
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .expect("id column")
            .as_primitive::<Int64Type>();
        let specs = batch
            .column_by_name(RESERVED_COL_NAME_SPEC_ID)
            .expect("_spec_id column");
        for i in 0..batch.num_rows() {
            rows.push((ids.value(i), decode_spec_id(specs, i)));
        }
    }
    rows.sort();
    rows
}

fn id_single_partition_rows(batches: &[RecordBatch]) -> Vec<(i64, bool, Option<String>)> {
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .expect("id column")
            .as_primitive::<Int64Type>();
        let structs = batch
            .column_by_name(RESERVED_COL_NAME_PARTITION)
            .expect("_partition column")
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("_partition is a struct");
        let cats = structs
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("cat child is Utf8");
        for i in 0..batch.num_rows() {
            let child = if cats.is_null(i) {
                None
            } else {
                Some(cats.value(i).to_string())
            };
            rows.push((ids.value(i), structs.is_null(i), child));
        }
    }
    rows.sort();
    rows
}

fn id_double_partition_rows(
    batches: &[RecordBatch],
) -> Vec<(i64, bool, Option<String>, Option<i64>)> {
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .expect("id column")
            .as_primitive::<Int64Type>();
        let structs = batch
            .column_by_name(RESERVED_COL_NAME_PARTITION)
            .expect("_partition column")
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("_partition is a struct");
        let cats = structs
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("cat child is Utf8");
        let ys = structs.column(1).as_primitive::<Int64Type>();
        for i in 0..batch.num_rows() {
            let cat = if cats.is_null(i) {
                None
            } else {
                Some(cats.value(i).to_string())
            };
            let y = if ys.is_null(i) {
                None
            } else {
                Some(ys.value(i))
            };
            rows.push((ids.value(i), structs.is_null(i), cat, y));
        }
    }
    rows.sort();
    rows
}

fn assert_partition_shape(batches: &[RecordBatch], names: &[&str]) {
    assert!(
        !batches.is_empty(),
        "expected at least one batch to pin the _partition shape"
    );
    let mut shapes = Vec::new();
    for batch in batches {
        let batch_schema = batch.schema();
        let field = batch_schema
            .field_with_name(RESERVED_COL_NAME_PARTITION)
            .expect("_partition field");
        assert!(
            field.is_nullable(),
            "_partition is optional and must stay nullable"
        );
        match field.data_type() {
            DataType::Struct(children) => {
                let got: Vec<&str> = children.iter().map(|child| child.name().as_str()).collect();
                assert_eq!(got, names, "union struct shape must match on every batch");
                shapes.push(field.data_type().clone());
            }
            other => panic!("_partition must be a struct, got {other:?}"),
        }
    }
    for shape in &shapes[1..] {
        assert_eq!(
            shape, &shapes[0],
            "every file in one scan must serve the same union struct"
        );
    }
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

async fn first_data_file_path(table: &Table) -> String {
    let batches = scan_batches(table, &["id", RESERVED_COL_NAME_FILE]).await;
    assert!(
        !batches.is_empty(),
        "expected at least one batch to find a data file"
    );
    assert!(
        batches[0].num_rows() > 0,
        "expected at least one row to find a data file"
    );
    let files = batches[0]
        .column_by_name(RESERVED_COL_NAME_FILE)
        .expect("_file column");
    decode_file_path(files, 0)
}

#[tokio::test]
async fn spec_id_served_per_file_on_partitioned_table() {
    let (_tmp, catalog, table) = empty_table("spec_id", cat_spec()).await;
    let first = write_data_file(
        &table,
        "00000-spec-id-a.parquet",
        &[2],
        &[Some("x")],
        &[None],
        0,
        Struct::from_iter([Some(Literal::string("x"))]),
    )
    .await;
    let second = write_data_file(
        &table,
        "00000-spec-id-b.parquet",
        &[3, 4],
        &[Some("y"), Some("y")],
        &[None, None],
        0,
        Struct::from_iter([Some(Literal::string("y"))]),
    )
    .await;
    let table = append_data(&catalog, &table, vec![first, second]).await;
    let batches = scan_batches(&table, &["id", RESERVED_COL_NAME_SPEC_ID]).await;
    assert_eq!(id_spec_pairs(&batches), vec![(2, 0), (3, 0), (4, 0)]);
}

#[tokio::test]
async fn partition_struct_values_match_file_tuples() {
    let (_tmp, catalog, table) = empty_table("partition_values", cat_spec()).await;
    let first = write_data_file(
        &table,
        "00000-partition-a.parquet",
        &[2],
        &[Some("x")],
        &[None],
        0,
        Struct::from_iter([Some(Literal::string("x"))]),
    )
    .await;
    let second = write_data_file(
        &table,
        "00000-partition-b.parquet",
        &[3, 4],
        &[Some("y"), Some("y")],
        &[None, None],
        0,
        Struct::from_iter([Some(Literal::string("y"))]),
    )
    .await;
    let table = append_data(&catalog, &table, vec![first, second]).await;
    let batches = scan_batches(&table, &["id", RESERVED_COL_NAME_PARTITION]).await;
    assert_eq!(id_single_partition_rows(&batches), vec![
        (2, false, Some("x".to_string())),
        (3, false, Some("y".to_string())),
        (4, false, Some("y".to_string())),
    ]);
    assert_partition_shape(&batches, &["cat"]);
}

#[tokio::test]
async fn evolved_add_field_yields_union_struct_with_null_where_absent() {
    let (_tmp, catalog, table) = empty_table("evo_add", unpartitioned_spec()).await;
    let first = write_data_file(
        &table,
        "00000-evo-add-a.parquet",
        &[1],
        &[None],
        &[None],
        0,
        Struct::empty(),
    )
    .await;
    let table = append_data(&catalog, &table, vec![first]).await;
    let table = evolve_add_field(&catalog, &table, "cat").await;
    let second = write_data_file(
        &table,
        "00000-evo-add-b.parquet",
        &[2],
        &[Some("y")],
        &[None],
        1,
        Struct::from_iter([Some(Literal::string("y"))]),
    )
    .await;
    let table = append_data(&catalog, &table, vec![second]).await;
    let batches = scan_batches(&table, &[
        "id",
        RESERVED_COL_NAME_SPEC_ID,
        RESERVED_COL_NAME_PARTITION,
    ])
    .await;
    assert_eq!(id_spec_pairs(&batches), vec![(1, 0), (2, 1)]);
    assert_eq!(id_single_partition_rows(&batches), vec![
        (1, false, None),
        (2, false, Some("y".to_string())),
    ]);
    assert_partition_shape(&batches, &["cat"]);
}

#[tokio::test]
async fn evolved_remove_field_reads_null_where_absent() {
    let (_tmp, catalog, table) = empty_table("evo_remove", cat_y_spec()).await;
    let first = write_data_file(
        &table,
        "00000-evo-remove-a.parquet",
        &[10],
        &[Some("c0")],
        &[Some(70)],
        0,
        Struct::from_iter([Some(Literal::string("c0")), Some(Literal::long(70))]),
    )
    .await;
    let table = append_data(&catalog, &table, vec![first]).await;
    let table = evolve_remove_field(&catalog, &table, "cat").await;
    let second = write_data_file(
        &table,
        "00000-evo-remove-b.parquet",
        &[11],
        &[Some("c1")],
        &[Some(71)],
        1,
        Struct::from_iter([Some(Literal::long(71))]),
    )
    .await;
    let table = append_data(&catalog, &table, vec![second]).await;
    let batches = scan_batches(&table, &[
        "id",
        RESERVED_COL_NAME_SPEC_ID,
        RESERVED_COL_NAME_PARTITION,
    ])
    .await;
    assert_eq!(id_spec_pairs(&batches), vec![(10, 0), (11, 1)]);
    assert_eq!(id_double_partition_rows(&batches), vec![
        (10, false, Some("c0".to_string()), Some(70)),
        (11, false, None, Some(71)),
    ]);
    assert_partition_shape(&batches, &["cat", "y"]);
}

#[tokio::test]
async fn unpartitioned_table_yields_null_partition_struct() {
    let (_tmp, catalog, table) = empty_table("unpartitioned", unpartitioned_spec()).await;
    let file = write_data_file(
        &table,
        "00000-unpartitioned.parquet",
        &[2, 3],
        &[None, None],
        &[None, None],
        0,
        Struct::empty(),
    )
    .await;
    let table = append_data(&catalog, &table, vec![file]).await;
    let batches = scan_batches(&table, &["id", RESERVED_COL_NAME_PARTITION]).await;
    assert!(
        !batches.is_empty(),
        "expected at least one batch to pin the null struct"
    );
    let mut row_count = 0;
    for batch in &batches {
        let structs = batch
            .column_by_name(RESERVED_COL_NAME_PARTITION)
            .expect("_partition column")
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("_partition is a struct");
        assert_eq!(structs.num_columns(), 0);
        for i in 0..batch.num_rows() {
            assert!(
                structs.is_null(i),
                "unpartitioned tables serve a null struct, never an empty one"
            );
            row_count += 1;
        }
    }
    assert_eq!(row_count, 2);
    assert_partition_shape(&batches, &[]);
}

#[tokio::test]
async fn missing_spec_file_yields_null_partition_struct() {
    let (_tmp, catalog, table) = empty_table("missing_spec", cat_spec()).await;
    let file = write_data_file(
        &table,
        "00000-missing-spec.parquet",
        &[2, 3],
        &[Some("x"), Some("x")],
        &[None, None],
        0,
        Struct::from_iter([Some(Literal::string("x"))]),
    )
    .await;
    let table = append_data(&catalog, &table, vec![file]).await;
    let path = first_data_file_path(&table).await;
    let schema = table.metadata().current_schema();
    let union = StructType::new(vec![Arc::new(NestedField::optional(
        1000,
        "cat",
        Type::Primitive(PrimitiveType::String),
    ))]);
    let mut fields = schema.as_struct().fields().to_vec();
    fields.push(partition_field(union.fields().to_vec()));
    let task_schema = Arc::new(
        Schema::builder()
            .with_schema_id(schema.schema_id())
            .with_fields(fields)
            .build()
            .expect("augmented task schema builds"),
    );
    let file_size = std::fs::metadata(&path).expect("stat data file").len();
    let task = FileScanTask {
        file_size_in_bytes: file_size,
        start: 0,
        length: file_size,
        record_count: None,
        file_record_count: None,
        data_file_path: Arc::from(path),
        data_file_format: DataFileFormat::Parquet,
        schema: task_schema,
        project_field_ids: Arc::from(vec![1, RESERVED_FIELD_ID_PARTITION]),
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
    let reader = ArrowReaderBuilder::new(table.file_io().clone()).build();
    let batches: Vec<RecordBatch> = reader
        .read(Box::pin(futures::stream::iter(vec![Ok(task)])))
        .expect("read hand-built task")
        .try_collect()
        .await
        .expect("collect batches");
    assert!(
        !batches.is_empty(),
        "expected at least one batch to pin the null struct"
    );
    let mut row_count = 0;
    for batch in &batches {
        let structs = batch
            .column_by_name(RESERVED_COL_NAME_PARTITION)
            .expect("_partition column")
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("_partition is a struct");
        assert_eq!(structs.num_columns(), 1);
        for i in 0..batch.num_rows() {
            assert!(
                structs.is_null(i),
                "a file with a missing spec serves a null struct"
            );
            row_count += 1;
        }
    }
    assert_eq!(row_count, 2);
    assert_partition_shape(&batches, &["cat"]);
}

#[tokio::test]
async fn pos_projection_path_serves_spec_id_and_partition() {
    let (_tmp, catalog, table) = empty_table("pos_path", cat_spec()).await;
    let file = write_data_file(
        &table,
        "00000-pos-path.parquet",
        &[5],
        &[Some("z")],
        &[None],
        0,
        Struct::from_iter([Some(Literal::string("z"))]),
    )
    .await;
    let table = append_data(&catalog, &table, vec![file]).await;
    let batches = scan_batches(&table, &[
        "id",
        RESERVED_COL_NAME_POS,
        RESERVED_COL_NAME_SPEC_ID,
        RESERVED_COL_NAME_PARTITION,
    ])
    .await;
    assert_eq!(id_spec_pairs(&batches), vec![(5, 0)]);
    assert_eq!(id_single_partition_rows(&batches), vec![(
        5,
        false,
        Some("z".to_string())
    )]);
    let mut positions = Vec::new();
    for batch in &batches {
        let pos = batch
            .column_by_name(RESERVED_COL_NAME_POS)
            .expect("_pos column")
            .as_primitive::<Int64Type>();
        for i in 0..batch.num_rows() {
            positions.push(pos.value(i));
        }
    }
    assert_eq!(positions, vec![0]);
}
