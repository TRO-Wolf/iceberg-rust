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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use futures::TryStreamExt;

use crate::arrow::RecordBatchPartitionSplitter;
use crate::error::ErrorKind;
use crate::expr::Reference;
use crate::maintenance::rewrite_data_files::RewriteDataFiles;
use crate::maintenance::rewrite_data_files::tests::{
    append_files, create_partitioned_table, live_data_file_paths, local_fs_catalog, scan_rows,
    write_data_file,
};
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, Datum, FormatVersion, Literal, NestedField,
    PrimitiveLiteral, PrimitiveType, Schema, Struct, Transform, Type,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::transform::create_transform_function;
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Catalog, NamespaceIdent, TableCreation};

pub(crate) async fn create_unpartitioned_table(
    catalog: &impl Catalog,
    format_version: FormatVersion,
) -> Table {
    let schema = Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "x",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
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
        .expect("schema");
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let creation = TableCreation::builder()
        .name("t".to_string())
        .schema(schema)
        .format_version(format_version)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

pub(crate) async fn evolve_spec(
    catalog: &impl Catalog,
    table: &Table,
    action: impl ApplyTransactionAction,
) -> Table {
    let tx = Transaction::new(table);
    action
        .apply(tx)
        .expect("apply spec update")
        .commit(catalog)
        .await
        .expect("commit spec update")
}

pub(crate) fn compact_action(table: Table) -> RewriteDataFiles {
    RewriteDataFiles::new(table)
        .target_file_size_bytes(1_000_000)
        .min_input_files(2)
}

pub(crate) async fn compact(
    catalog: &impl Catalog,
    table: Table,
) -> (Table, crate::maintenance::RewriteDataFilesResult) {
    let ident = table.identifier().clone();
    let result = compact_action(table)
        .execute(catalog)
        .await
        .expect("compact");
    let table = catalog.load_table(&ident).await.expect("reload");
    (table, result)
}

pub(crate) async fn live_data_files(table: &Table) -> Vec<DataFile> {
    let snapshot = table.metadata().current_snapshot().expect("snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    let mut files = Vec::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.content_type() == DataContentType::Data {
                files.push(entry.data_file().clone());
            }
        }
    }
    files
}

pub(crate) async fn scan_pruned_rows(
    table: &Table,
    column: &str,
    value: i64,
) -> Vec<(i64, i64, i64)> {
    let stream = table
        .scan()
        .with_filter(Reference::new(column).equal_to(Datum::long(value)))
        .select(["x", "y", "z"])
        .build()
        .expect("scan")
        .to_arrow()
        .await
        .expect("to_arrow");
    let batches: Vec<RecordBatch> = stream.try_collect().await.expect("collect");
    let mut rows = Vec::new();
    for batch in batches {
        let xs = batch
            .column_by_name("x")
            .expect("x")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("x i64");
        let ys = batch
            .column_by_name("y")
            .expect("y")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("y i64");
        let zs = batch
            .column_by_name("z")
            .expect("z")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("z i64");
        for index in 0..xs.len() {
            rows.push((xs.value(index), ys.value(index), zs.value(index)));
        }
    }
    rows.sort_unstable();
    rows
}

pub(crate) fn literal_from_long_transform(transform: Transform, value: i64) -> Literal {
    let func = create_transform_function(&transform).expect("transform fn");
    let datum = func
        .transform_literal_result(&Datum::long(value))
        .expect("apply transform");
    Literal::from(datum)
}

fn assert_output_matches_current_spec(files: &[DataFile], table: &Table, expected: &[Struct]) {
    let spec_id = table.metadata().default_partition_spec().spec_id();
    assert!(!files.is_empty(), "rewrite must emit output data files");
    let actual: HashSet<Struct> = files
        .iter()
        .map(|file| {
            assert_eq!(
                file.partition_spec_id(),
                spec_id,
                "output file {} must claim the current spec",
                file.file_path()
            );
            file.partition().clone()
        })
        .collect();
    let expected: HashSet<Struct> = expected.iter().cloned().collect();
    assert_eq!(actual, expected, "output tuples must equal recomputed keys");
}

pub(crate) async fn write_current_spec_file(
    table: &Table,
    file_name: &str,
    rows: &[(i64, i64, i64)],
) -> DataFile {
    use crate::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let spec = table.metadata().default_partition_spec().as_ref().clone();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("arrow schema"));
    let xs: Vec<i64> = rows.iter().map(|(x, _, _)| *x).collect();
    let ys: Vec<i64> = rows.iter().map(|(_, y, _)| *y).collect();
    let zs: Vec<i64> = rows.iter().map(|(_, _, z)| *z).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(xs)) as ArrayRef,
        Arc::new(Int64Array::from(ys)) as ArrayRef,
        Arc::new(Int64Array::from(zs)) as ArrayRef,
    ])
    .expect("batch");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location gen");
    let file_name_gen = DefaultFileNameGenerator::new(
        file_name.to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let builder = DataFileWriterBuilder::new(rolling).with_partition_spec(spec.clone());
    if spec.fields().is_empty() {
        let mut writer = builder.build(None).await.expect("unpartitioned writer");
        writer.write(batch).await.expect("write");
        return writer
            .close()
            .await
            .expect("close")
            .into_iter()
            .next()
            .expect("file");
    }
    let splitter = RecordBatchPartitionSplitter::try_new_with_computed_values(
        schema.clone(),
        table.metadata().default_partition_spec().clone(),
    )
    .expect("splitter");
    let splits = splitter.split(&batch).expect("split");
    assert_eq!(splits.len(), 1, "helper writes one current partition");
    let (partition_key, partition_batch) = splits.into_iter().next().expect("one split");
    let mut writer = builder
        .build(Some(partition_key))
        .await
        .expect("partitioned writer");
    writer.write(partition_batch).await.expect("write");
    writer
        .close()
        .await
        .expect("close")
        .into_iter()
        .next()
        .expect("file")
}

#[tokio::test]
async fn source_field_identity_x_to_identity_y_rewrites_two_old_partitions() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let a = write_data_file(&table, "a.parquet", 1, &[(1, 10, 100)]).await;
    let b = write_data_file(&table, "b.parquet", 2, &[(2, 20, 200)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;
    let before = scan_rows(&table).await;
    assert_eq!(before, vec![(1, 10, 100), (2, 20, 200)]);

    let table = evolve_spec(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_partition_spec()
            .remove_field("x")
            .add_field("y"),
    )
    .await;
    let (table, result) = compact(&catalog, table).await;
    assert!(
        result.rewritten_data_files_count >= 2,
        "two old partitions must co-enter the rewrite, rewritten={}",
        result.rewritten_data_files_count
    );

    let after = scan_rows(&table).await;
    assert_eq!(after, before, "full scan must keep the live rows");
    assert_eq!(
        scan_pruned_rows(&table, "y", 10).await,
        vec![(1, 10, 100)],
        "pruned y=10 must return only that row"
    );
    assert_eq!(
        scan_pruned_rows(&table, "y", 20).await,
        vec![(2, 20, 200)],
        "pruned y=20 must return only that row"
    );

    let files = live_data_files(&table).await;
    assert_output_matches_current_spec(&files, &table, &[
        Struct::from_iter([Some(Literal::long(10))]),
        Struct::from_iter([Some(Literal::long(20))]),
    ]);
}

#[tokio::test]
async fn transform_identity_x_to_bucket_x_stamps_bucket_not_identity() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let a = write_data_file(&table, "a.parquet", 1, &[(1, 10, 100)]).await;
    let b = write_data_file(&table, "b.parquet", 2, &[(2, 20, 200)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;
    let before = scan_rows(&table).await;

    let table = evolve_spec(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_partition_spec()
            .remove_field("x")
            .add_field_with_transform(None, "x", Transform::Bucket(8)),
    )
    .await;
    let (table, result) = compact(&catalog, table).await;
    assert!(result.rewritten_data_files_count >= 2);

    assert_eq!(scan_rows(&table).await, before);
    assert_eq!(
        scan_pruned_rows(&table, "x", 1).await,
        vec![(1, 10, 100)],
        "pruned x=1 must return only that row"
    );
    assert_eq!(
        scan_pruned_rows(&table, "x", 2).await,
        vec![(2, 20, 200)],
        "pruned x=2 must return only that row"
    );
    let files = live_data_files(&table).await;
    let bucket_1 = literal_from_long_transform(Transform::Bucket(8), 1);
    let bucket_2 = literal_from_long_transform(Transform::Bucket(8), 2);
    assert_ne!(
        bucket_1,
        Literal::long(1),
        "bucket(1) must not equal the identity value"
    );
    assert_output_matches_current_spec(&files, &table, &[
        Struct::from_iter([Some(bucket_1)]),
        Struct::from_iter([Some(bucket_2)]),
    ]);
    for file in &files {
        match file.partition().fields().first().and_then(|f| f.as_ref()) {
            Some(Literal::Primitive(PrimitiveLiteral::Int(_))) => {}
            other => panic!("bucket output must be int, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn transform_bucket8_to_truncate10_stamps_current_transform() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let table = evolve_spec(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_partition_spec()
            .remove_field("x")
            .add_field_with_transform(None, "x", Transform::Bucket(8)),
    )
    .await;
    let a = write_current_spec_file(&table, "a", &[(11, 10, 100)]).await;
    let b = write_current_spec_file(&table, "b", &[(22, 20, 200)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;
    let before = scan_rows(&table).await;

    let table = evolve_spec(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_partition_spec()
            .remove_field_by_transform("x", Transform::Bucket(8))
            .add_field_with_transform(None, "x", Transform::Truncate(10)),
    )
    .await;
    let (table, result) = compact(&catalog, table).await;
    assert!(result.rewritten_data_files_count >= 2);
    assert_eq!(scan_rows(&table).await, before);
    assert_eq!(
        scan_pruned_rows(&table, "x", 11).await,
        vec![(11, 10, 100)],
        "pruned x=11 must return only that row"
    );
    assert_eq!(
        scan_pruned_rows(&table, "x", 22).await,
        vec![(22, 20, 200)],
        "pruned x=22 must return only that row"
    );

    let files = live_data_files(&table).await;
    let t11 = literal_from_long_transform(Transform::Truncate(10), 11);
    let t22 = literal_from_long_transform(Transform::Truncate(10), 22);
    assert_output_matches_current_spec(&files, &table, &[
        Struct::from_iter([Some(t11)]),
        Struct::from_iter([Some(t22)]),
    ]);
}

#[tokio::test]
async fn partitioned_to_unpartitioned_uses_empty_tuples() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let a = write_data_file(&table, "a.parquet", 1, &[(1, 10, 100)]).await;
    let b = write_data_file(&table, "b.parquet", 2, &[(2, 20, 200)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;
    let before = scan_rows(&table).await;

    let table = evolve_spec(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_partition_spec()
            .remove_field("x"),
    )
    .await;
    assert!(
        table
            .metadata()
            .default_partition_spec()
            .fields()
            .is_empty(),
        "current spec must have no partition fields"
    );
    let (table, result) = compact(&catalog, table).await;
    assert!(result.rewritten_data_files_count >= 2);
    assert_eq!(scan_rows(&table).await, before);
    assert_eq!(
        scan_pruned_rows(&table, "x", 1).await,
        vec![(1, 10, 100)],
        "residual x=1 must return only that row after the spec is empty"
    );
    assert_eq!(
        scan_pruned_rows(&table, "x", 2).await,
        vec![(2, 20, 200)],
        "residual x=2 must return only that row after the spec is empty"
    );

    let files = live_data_files(&table).await;
    let spec_id = table.metadata().default_partition_spec().spec_id();
    assert!(!files.is_empty());
    for file in &files {
        assert_eq!(file.partition_spec_id(), spec_id);
        assert_eq!(file.partition(), &Struct::empty());
    }
}

#[tokio::test]
async fn unpartitioned_to_partitioned_fans_out_to_recomputed_keys() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_unpartitioned_table(&catalog, FormatVersion::V2).await;
    let mixed = write_current_spec_file(&table, "mixed-a", &[(1, 10, 100), (2, 20, 200)]).await;
    let extra = write_current_spec_file(&table, "mixed-b", &[(3, 30, 300)]).await;
    let table = append_files(&catalog, &table, vec![mixed, extra]).await;
    let before = scan_rows(&table).await;

    let table = evolve_spec(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_partition_spec()
            .add_field("x"),
    )
    .await;
    let (table, result) = compact(&catalog, table).await;
    assert!(result.rewritten_data_files_count >= 2);
    assert_eq!(scan_rows(&table).await, before);
    assert_eq!(scan_pruned_rows(&table, "x", 1).await, vec![(1, 10, 100)]);
    assert_eq!(scan_pruned_rows(&table, "x", 2).await, vec![(2, 20, 200)]);
    assert_eq!(scan_pruned_rows(&table, "x", 3).await, vec![(3, 30, 300)]);

    let files = live_data_files(&table).await;
    assert_output_matches_current_spec(&files, &table, &[
        Struct::from_iter([Some(Literal::long(1))]),
        Struct::from_iter([Some(Literal::long(2))]),
        Struct::from_iter([Some(Literal::long(3))]),
    ]);
}

#[tokio::test]
async fn mixed_current_and_old_files_all_use_recomputed_current_spec() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let old_a = write_data_file(&table, "old-a.parquet", 1, &[(1, 10, 100)]).await;
    let old_b = write_data_file(&table, "old-b.parquet", 2, &[(2, 20, 200)]).await;
    let table = append_files(&catalog, &table, vec![old_a, old_b]).await;

    let table = evolve_spec(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_partition_spec()
            .remove_field("x")
            .add_field("y"),
    )
    .await;
    let current = write_current_spec_file(&table, "cur", &[(9, 30, 300)]).await;
    let current2 = write_current_spec_file(&table, "cur2", &[(8, 30, 301)]).await;
    let table = append_files(&catalog, &table, vec![current, current2]).await;
    let before = scan_rows(&table).await;
    assert_eq!(before.len(), 4);

    let files_before = live_data_file_paths(&table).await.len();
    let (table, result) = compact(&catalog, table).await;
    assert!(result.rewritten_data_files_count >= 2);
    assert_eq!(scan_rows(&table).await, before);
    assert_eq!(scan_pruned_rows(&table, "y", 10).await, vec![(1, 10, 100)]);
    assert_eq!(scan_pruned_rows(&table, "y", 20).await, vec![(2, 20, 200)]);
    let pruned_30 = scan_pruned_rows(&table, "y", 30).await;
    assert_eq!(pruned_30, vec![(8, 30, 301), (9, 30, 300)]);

    let files = live_data_files(&table).await;
    let spec_id = table.metadata().default_partition_spec().spec_id();
    for file in &files {
        assert_eq!(file.partition_spec_id(), spec_id);
    }
    assert_ne!(
        live_data_file_paths(&table).await.len(),
        files_before,
        "mixed rewrite must change the live file set"
    );
}

#[tokio::test]
async fn all_void_current_spec_is_refused() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_unpartitioned_table(&catalog, FormatVersion::V2).await;
    let a = write_current_spec_file(&table, "a", &[(1, 10, 100)]).await;
    let b = write_current_spec_file(&table, "b", &[(2, 20, 200)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;
    let table = evolve_spec(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_partition_spec()
            .add_field_with_transform(None, "x", Transform::Void),
    )
    .await;
    let spec = table.metadata().default_partition_spec();
    assert_eq!(spec.fields().len(), 1);
    assert!(spec.is_unpartitioned());
    assert_eq!(spec.fields()[0].transform, Transform::Void);

    let err = compact_action(table)
        .execute(&catalog)
        .await
        .expect_err("all-void current spec must fail");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(
        err.message()
            .contains("Cannot create partition calculator for unpartitioned table"),
        "unexpected message: {}",
        err.message()
    );
}
