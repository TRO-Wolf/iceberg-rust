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

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Float32Array, Int64Array, RecordBatch};
use futures::TryStreamExt;

use crate::arrow::{ArrowReaderBuilder, schema_to_arrow_schema};
use crate::maintenance::rewrite_data_files::RewriteDataFiles;
use crate::maintenance::rewrite_data_files::tests::{
    append_files, create_partitioned_table, live_data_file_paths, local_fs_catalog, scan_rows,
    write_data_file,
};
use crate::metadata_columns::{
    RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER, RESERVED_COL_NAME_ROW_ID,
};
use crate::scan::{FileScanTask, FileScanTaskStream};
use crate::spec::{
    DataContentType, DataFile, FormatVersion, Literal, NestedField, NullOrder, PartitionSpec,
    PrimitiveType, Schema, SortDirection, SortField, SortOrder, Struct, Transform, Type,
};
use crate::table::Table;
use crate::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
use crate::{Catalog, NamespaceIdent, TableCreation};

async fn scan_lineage(table: &Table) -> Vec<(i64, i64, i64)> {
    let stream = table
        .scan()
        .select([
            "y",
            RESERVED_COL_NAME_ROW_ID,
            RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER,
        ])
        .build()
        .expect("scan")
        .to_arrow()
        .await
        .expect("to_arrow");
    let batches: Vec<_> = stream.try_collect().await.expect("collect");
    let mut rows = Vec::new();
    for batch in batches {
        let ys = batch
            .column_by_name("y")
            .expect("y")
            .as_primitive::<arrow_array::types::Int64Type>();
        let row_ids = batch
            .column_by_name(RESERVED_COL_NAME_ROW_ID)
            .expect("_row_id")
            .as_primitive::<arrow_array::types::Int64Type>();
        let seqs = batch
            .column_by_name(RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER)
            .expect("_last_updated_sequence_number")
            .as_primitive::<arrow_array::types::Int64Type>();
        for index in 0..batch.num_rows() {
            assert!(
                row_ids.is_valid(index),
                "compacted v3 row must have a _row_id"
            );
            assert!(
                seqs.is_valid(index),
                "compacted v3 row must have a last_updated_seq"
            );
            rows.push((ys.value(index), row_ids.value(index), seqs.value(index)));
        }
    }
    rows.sort_unstable();
    rows
}

#[tokio::test]
async fn v3_compaction_keeps_row_id_and_last_updated_seq() {
    let (catalog, _temp) = local_fs_catalog().await;
    let mut table = create_partitioned_table(&catalog, FormatVersion::V3).await;

    for index in 0..6i64 {
        let file = write_data_file(&table, &format!("small-{index}.parquet"), 0, &[(
            0,
            100 + index,
            1000 + index,
        )])
        .await;
        table = append_files(&catalog, &table, vec![file]).await;
    }

    let before = scan_lineage(&table).await;
    assert_eq!(before.len(), 6, "fixture: six rows");
    for (index, row) in before.iter().enumerate() {
        assert_eq!(
            row.1, index as i64,
            "pre-compaction _row_id is first_row_id + pos"
        );
        assert_eq!(
            row.2,
            (index as i64) + 1,
            "pre-compaction last_updated_seq is the append snapshot sequence"
        );
    }

    let files_before = live_data_file_paths(&table).await.len();
    let result = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(1_000_000)
        .execute(&catalog)
        .await
        .expect("compaction");
    assert_eq!(result.rewritten_data_files_count, 6);
    assert!(result.added_data_files_count >= 1);

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload");
    let files_after = live_data_file_paths(&table).await.len();
    assert!(
        files_after < files_before,
        "compaction must reduce the file count"
    );

    let after = scan_lineage(&table).await;
    assert_eq!(
        after, before,
        "compaction must keep _row_id and last_updated_seq for every live row"
    );
}

#[tokio::test]
async fn v2_compaction_does_not_persist_row_lineage_columns() {
    let (catalog, _temp) = local_fs_catalog().await;
    let mut table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    for index in 0..6i64 {
        let file = write_data_file(&table, &format!("small-{index}.parquet"), 0, &[(
            0,
            100 + index,
            1000 + index,
        )])
        .await;
        table = append_files(&catalog, &table, vec![file]).await;
    }

    RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(1_000_000)
        .execute(&catalog)
        .await
        .expect("v2 compaction");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload");

    let stream = table
        .scan()
        .select(["y", RESERVED_COL_NAME_ROW_ID])
        .build()
        .expect("scan")
        .to_arrow()
        .await
        .expect("to_arrow");
    let batches: Vec<_> = stream.try_collect().await.expect("collect");
    for batch in batches {
        let row_ids = batch
            .column_by_name(RESERVED_COL_NAME_ROW_ID)
            .expect("_row_id");
        assert_eq!(
            row_ids.null_count(),
            row_ids.len(),
            "v2 files must not grow a stored _row_id column; a v2 scan reports all-null lineage"
        );
    }
}

fn sort_field(source_id: i32, direction: SortDirection, null_order: NullOrder) -> SortField {
    SortField::builder()
        .source_id(source_id)
        .transform(Transform::Identity)
        .direction(direction)
        .null_order(null_order)
        .build()
}

fn nullable_yz_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "x",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                2,
                "y",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                3,
                "z",
                Type::Primitive(PrimitiveType::Long),
            )),
        ])
        .build()
        .expect("build (x, y, z) schema")
}

fn float_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                2,
                "f",
                Type::Primitive(PrimitiveType::Float),
            )),
        ])
        .build()
        .expect("build (id, f) schema")
}

async fn create_sort_table(catalog: &impl Catalog, schema: Schema, order: SortOrder) -> Table {
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("x", "x", Transform::Identity)
        .expect("add partition field")
        .build()
        .expect("build spec");
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("t".to_string())
        .schema(schema)
        .partition_spec(spec)
        .sort_order(order)
        .format_version(FormatVersion::V2)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

async fn write_batch_file(
    table: &Table,
    file_name: &str,
    batch: RecordBatch,
    partition: Struct,
) -> DataFile {
    let schema = table.metadata().current_schema();
    let file_path = format!("{}/data/{}", table.metadata().location(), file_name);
    let output = table.file_io().new_output(file_path).expect("output");
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let mut writer = parquet_builder.build(output).await.expect("writer");
    writer.write(&batch).await.expect("write batch");
    let mut builder = writer
        .close()
        .await
        .expect("close writer")
        .into_iter()
        .next()
        .expect("one data file");
    builder
        .content(DataContentType::Data)
        .partition_spec_id(0)
        .partition(partition)
        .build()
        .expect("build data file")
}

async fn write_nullable_y_file(
    table: &Table,
    file_name: &str,
    part: i64,
    ys: &[Option<i64>],
) -> DataFile {
    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("arrow schema"));
    let n = ys.len();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(vec![part; n])) as ArrayRef,
        Arc::new(Int64Array::from(ys.to_vec())) as ArrayRef,
        Arc::new(Int64Array::from_iter_values(0..n as i64)) as ArrayRef,
    ])
    .expect("build batch");
    write_batch_file(
        table,
        file_name,
        batch,
        Struct::from_iter([Some(Literal::long(part))]),
    )
    .await
}

async fn write_float_file(table: &Table, file_name: &str, fs: &[Option<f32>]) -> DataFile {
    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("arrow schema"));
    let n = fs.len();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from_iter_values(0..n as i64)) as ArrayRef,
        Arc::new(Float32Array::from(fs.to_vec())) as ArrayRef,
    ])
    .expect("build batch");
    write_batch_file(table, file_name, batch, Struct::empty()).await
}

async fn live_data_files(table: &Table) -> Vec<DataFile> {
    let snapshot = table.metadata().current_snapshot().unwrap();
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .unwrap();
    let mut files = Vec::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
        for entry in manifest.entries() {
            if entry.is_alive() && entry.content_type() == DataContentType::Data {
                files.push(entry.data_file().clone());
            }
        }
    }
    files
}

async fn read_file_batches(table: &Table, file: &DataFile) -> Vec<RecordBatch> {
    let schema = table.metadata().current_schema().clone();
    let field_ids: Vec<i32> = schema
        .as_struct()
        .fields()
        .iter()
        .map(|field| field.id)
        .collect();
    let task = FileScanTask {
        file_size_in_bytes: file.file_size_in_bytes(),
        start: 0,
        length: file.file_size_in_bytes(),
        record_count: Some(file.record_count()),
        file_record_count: Some(file.record_count()),
        data_file_path: Arc::from(file.file_path()),
        data_file_format: file.file_format(),
        schema: schema.clone(),
        project_field_ids: Arc::from(field_ids.as_slice()),
        predicate: None,
        deletes: Arc::from(vec![]),
        partition: Some(file.partition().clone()),
        partition_spec: table.metadata().partition_spec_by_id(0).cloned(),
        name_mapping: None,
        case_sensitive: false,
        split_offsets: None,
        first_row_id: None,
        file_sequence_number: None,
    };
    let tasks = Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream;
    let stream = ArrowReaderBuilder::new(table.file_io().clone())
        .build()
        .read(tasks)
        .expect("read output file");
    stream.try_collect().await.expect("collect output rows")
}

async fn file_i64_column(table: &Table, file: &DataFile, name: &str) -> Vec<Option<i64>> {
    let mut values = Vec::new();
    for batch in read_file_batches(table, file).await {
        let column = batch
            .column_by_name(name)
            .expect("column")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("long column");
        values.extend(column.iter());
    }
    values
}

async fn file_f32_column(table: &Table, file: &DataFile, name: &str) -> Vec<Option<f32>> {
    let mut values = Vec::new();
    for batch in read_file_batches(table, file).await {
        let column = batch
            .column_by_name(name)
            .expect("column")
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("float column");
        values.extend(column.iter());
    }
    values
}

fn desc_nulls_last(values: &[Option<i64>]) -> Vec<Option<i64>> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(x), Some(y)) => y.cmp(x),
    });
    sorted
}

fn float_asc_nulls_first_nan_last(a: &Option<f32>, b: &Option<f32>) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(x), Some(y)) => match (x.is_nan(), y.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => x.partial_cmp(y).expect("no NaN"),
        },
    }
}

#[tokio::test]
async fn binpack_output_files_sorted_by_default_order_and_stamped() {
    let (catalog, _temp) = local_fs_catalog().await;
    let order = SortOrder::builder()
        .with_sort_field(sort_field(2, SortDirection::Ascending, NullOrder::First))
        .build(&nullable_yz_schema())
        .expect("sort order");
    let table = create_sort_table(&catalog, nullable_yz_schema(), order).await;
    let order_id = i32::try_from(table.metadata().default_sort_order_id()).unwrap();

    let file_a = write_nullable_y_file(
        &table,
        "a.parquet",
        0,
        &(1400..2100i64).rev().map(Some).collect::<Vec<_>>(),
    )
    .await;
    let file_b = write_nullable_y_file(
        &table,
        "b.parquet",
        0,
        &(700..1400i64).rev().map(Some).collect::<Vec<_>>(),
    )
    .await;
    let file_c = write_nullable_y_file(
        &table,
        "c.parquet",
        0,
        &(0..700i64).rev().map(Some).collect::<Vec<_>>(),
    )
    .await;
    let table = append_files(&catalog, &table, vec![file_a, file_b, file_c]).await;

    let result = RewriteDataFiles::new(table.clone())
        .rewrite_all(true)
        .target_file_size_bytes(1)
        .max_file_size_bytes(1_000_000)
        .execute(&catalog)
        .await
        .expect("binpack rewrite");
    assert_eq!(result.rewritten_data_files_count, 3);
    assert!(
        result.added_data_files_count >= 2,
        "target size 1 must roll output into several files"
    );

    let table = catalog.load_table(table.identifier()).await.unwrap();
    let files = live_data_files(&table).await;
    let mut all = Vec::new();
    for file in &files {
        assert_eq!(
            file.sort_order_id(),
            Some(order_id),
            "output file stamps the default sort order id"
        );
        let ys = file_i64_column(&table, file, "y").await;
        assert!(
            ys.windows(2).all(|pair| pair[0] <= pair[1]),
            "output file rows ascending by y: {ys:?}"
        );
        all.extend(ys);
    }
    all.sort();
    assert_eq!(all, (0..2100).map(Some).collect::<Vec<_>>());
}

#[tokio::test]
async fn binpack_desc_nulls_last_orders_nulls_last() {
    let (catalog, _temp) = local_fs_catalog().await;
    let order = SortOrder::builder()
        .with_sort_field(sort_field(2, SortDirection::Descending, NullOrder::Last))
        .build(&nullable_yz_schema())
        .expect("sort order");
    let table = create_sort_table(&catalog, nullable_yz_schema(), order).await;
    let order_id = i32::try_from(table.metadata().default_sort_order_id()).unwrap();

    let file_a =
        write_nullable_y_file(&table, "a.parquet", 0, &[Some(3), None, Some(9), Some(1)]).await;
    let file_b =
        write_nullable_y_file(&table, "b.parquet", 0, &[Some(7), Some(2), None, Some(5)]).await;
    let table = append_files(&catalog, &table, vec![file_a, file_b]).await;

    RewriteDataFiles::new(table.clone())
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("binpack rewrite");

    let table = catalog.load_table(table.identifier()).await.unwrap();
    let files = live_data_files(&table).await;
    let mut all = Vec::new();
    for file in &files {
        assert_eq!(file.sort_order_id(), Some(order_id));
        let ys = file_i64_column(&table, file, "y").await;
        assert_eq!(ys, desc_nulls_last(&ys), "desc nulls-last per file");
        all.extend(ys);
    }
    assert_eq!(desc_nulls_last(&all), vec![
        Some(9),
        Some(7),
        Some(5),
        Some(3),
        Some(2),
        Some(1),
        None,
        None
    ]);
}

#[tokio::test]
async fn binpack_float_sort_places_nan_last_and_honors_nulls_first() {
    let (catalog, _temp) = local_fs_catalog().await;
    let order = SortOrder::builder()
        .with_sort_field(sort_field(2, SortDirection::Ascending, NullOrder::First))
        .build(&float_schema())
        .expect("sort order");
    let spec = PartitionSpec::unpartition_spec();
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("t".to_string())
        .schema(float_schema())
        .partition_spec(spec)
        .sort_order(order)
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table");
    let order_id = i32::try_from(table.metadata().default_sort_order_id()).unwrap();

    let file_a = write_float_file(&table, "a.parquet", &[Some(3.0), Some(f32::NAN), None]).await;
    let file_b = write_float_file(&table, "b.parquet", &[Some(1.0), None, Some(2.5)]).await;
    let file_c = write_float_file(&table, "c.parquet", &[Some(f32::NAN), Some(0.5)]).await;
    let table = append_files(&catalog, &table, vec![file_a, file_b, file_c]).await;

    RewriteDataFiles::new(table.clone())
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("binpack rewrite");

    let table = catalog.load_table(table.identifier()).await.unwrap();
    let files = live_data_files(&table).await;
    let mut all = Vec::new();
    for file in &files {
        assert_eq!(file.sort_order_id(), Some(order_id));
        let fs = file_f32_column(&table, file, "f").await;
        assert!(
            fs.windows(2)
                .all(|pair| float_asc_nulls_first_nan_last(&pair[0], &pair[1])
                    != Ordering::Greater),
            "nulls first, values ascending, NaN last: {fs:?}"
        );
        all.extend(fs);
    }
    assert_eq!(all.len(), 8);
    assert_eq!(all.iter().filter(|value| value.is_none()).count(), 2);
    assert_eq!(
        all.iter()
            .filter(|value| value.is_some_and(|float| float.is_nan()))
            .count(),
        2
    );
}

#[tokio::test]
async fn binpack_unsorted_table_stamps_zero_and_keeps_row_union() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let file_a = write_data_file(&table, "u-a.parquet", 0, &[(0, 5, 0), (0, 1, 1)]).await;
    let file_b = write_data_file(&table, "u-b.parquet", 0, &[(0, 3, 2), (0, 9, 3)]).await;
    let table = append_files(&catalog, &table, vec![file_a, file_b]).await;
    let rows_before = scan_rows(&table).await;

    RewriteDataFiles::new(table.clone())
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("binpack rewrite");

    let table = catalog.load_table(table.identifier()).await.unwrap();
    let files = live_data_files(&table).await;
    for file in &files {
        assert_eq!(
            file.sort_order_id(),
            Some(0),
            "unsorted table stamps sort order id 0"
        );
    }
    assert_eq!(scan_rows(&table).await, rows_before);
}
