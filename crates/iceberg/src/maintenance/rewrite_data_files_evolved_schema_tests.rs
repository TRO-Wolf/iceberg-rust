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

use arrow_array::{Array, ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;

use crate::expr::Reference;
use crate::maintenance::rewrite_data_files::tests::{
    add_deletes, append_files, create_partitioned_table, live_data_file_paths, local_fs_catalog,
    scan_rows, write_data_file,
};
use crate::maintenance::rewrite_data_files_evolved_spec_tests::{
    assert_output_matches_current_spec, compact, create_unpartitioned_table, evolve_spec,
    literal_from_long_transform, live_data_files, scan_pruned_rows, write_current_spec_file,
};
use crate::maintenance::rewrite_data_files_router_bound_tests::write_dv;
use crate::spec::{
    DataContentType, DataFile, Datum, FormatVersion, Literal, NestedField, PartitionSpec,
    PrimitiveType, Schema, Struct, Transform, Type,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{Catalog, NamespaceIdent, TableCreation, TableIdent};

pub(crate) async fn evolve_schema(
    catalog: &impl Catalog,
    table: &Table,
    action: impl ApplyTransactionAction,
) -> Table {
    let tx = Transaction::new(table);
    action
        .apply(tx)
        .expect("apply schema update")
        .commit(catalog)
        .await
        .expect("commit schema update")
}

pub(crate) async fn create_int_id_table(catalog: &impl Catalog, partition_column: &str) -> Table {
    let schema = Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Int),
            )),
            Arc::new(NestedField::required(
                2,
                "v",
                Type::Primitive(PrimitiveType::Long),
            )),
        ])
        .build()
        .expect("build int schema");
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field(partition_column, partition_column, Transform::Identity)
        .expect("add partition field")
        .build()
        .expect("build spec");
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let table_ident = TableIdent::new(namespace.clone(), "t".to_string());
    let creation = TableCreation::builder()
        .name(table_ident.name().to_string())
        .schema(schema)
        .partition_spec(spec)
        .format_version(FormatVersion::V2)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

pub(crate) async fn write_int_id_data_file(
    table: &Table,
    file_name: &str,
    partition: Struct,
    rows: &[(i32, i64)],
) -> DataFile {
    use crate::arrow::schema_to_arrow_schema;
    use crate::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("arrow schema"));
    let ids: Vec<i32> = rows.iter().map(|(id, _)| *id).collect();
    let vs: Vec<i64> = rows.iter().map(|(_, v)| *v).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int32Array::from(ids)) as ArrayRef,
        Arc::new(Int64Array::from(vs)) as ArrayRef,
    ])
    .expect("batch");
    let file_path = format!("{}/data/{}", table.metadata().location(), file_name);
    let output = table.file_io().new_output(file_path).expect("output");
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let mut writer = parquet_builder.build(output).await.expect("writer");
    writer.write(&batch).await.expect("write");
    let builders = writer.close().await.expect("close");
    let mut builder = builders.into_iter().next().expect("builder");
    builder
        .content(DataContentType::Data)
        .partition_spec_id(0)
        .partition(partition)
        .build()
        .expect("data file")
}

pub(crate) async fn scan_note_rows(table: &Table) -> Vec<(i64, i64, i64, Option<String>)> {
    let stream = table
        .scan()
        .select(["x", "y", "z", "note"])
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
        let notes = batch
            .column_by_name("note")
            .expect("note")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("note string");
        for index in 0..xs.len() {
            let note = if notes.is_null(index) {
                None
            } else {
                Some(notes.value(index).to_string())
            };
            rows.push((xs.value(index), ys.value(index), zs.value(index), note));
        }
    }
    rows.sort_unstable();
    rows
}

pub(crate) async fn scan_w_rows(table: &Table) -> Vec<(i64, i64, i64, Option<i64>)> {
    let stream = table
        .scan()
        .select(["x", "y", "z", "w"])
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
        let ws = batch
            .column_by_name("w")
            .expect("w")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("w i64");
        for index in 0..xs.len() {
            let w = if ws.is_null(index) {
                None
            } else {
                Some(ws.value(index))
            };
            rows.push((xs.value(index), ys.value(index), zs.value(index), w));
        }
    }
    rows.sort_unstable();
    rows
}

pub(crate) async fn scan_xy_rows(table: &Table) -> Vec<(i64, i64)> {
    let stream = table
        .scan()
        .select(["x", "y"])
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
        for index in 0..xs.len() {
            rows.push((xs.value(index), ys.value(index)));
        }
    }
    rows.sort_unstable();
    rows
}

pub(crate) async fn scan_pruned_xy_rows(
    table: &Table,
    column: &str,
    value: i64,
) -> Vec<(i64, i64)> {
    let stream = table
        .scan()
        .with_filter(Reference::new(column).equal_to(Datum::long(value)))
        .select(["x", "y"])
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
        for index in 0..xs.len() {
            rows.push((xs.value(index), ys.value(index)));
        }
    }
    rows.sort_unstable();
    rows
}

pub(crate) async fn scan_renamed_rows(table: &Table) -> Vec<(i64, i64, i64)> {
    let stream = table
        .scan()
        .select(["x", "y", "zz"])
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
        let zzs = batch
            .column_by_name("zz")
            .expect("zz")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("zz i64");
        for index in 0..xs.len() {
            rows.push((xs.value(index), ys.value(index), zzs.value(index)));
        }
    }
    rows.sort_unstable();
    rows
}

pub(crate) async fn scan_pruned_renamed_rows(
    table: &Table,
    column: &str,
    value: i64,
) -> Vec<(i64, i64, i64)> {
    let stream = table
        .scan()
        .with_filter(Reference::new(column).equal_to(Datum::long(value)))
        .select(["x", "y", "zz"])
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
        let zzs = batch
            .column_by_name("zz")
            .expect("zz")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("zz i64");
        for index in 0..xs.len() {
            rows.push((xs.value(index), ys.value(index), zzs.value(index)));
        }
    }
    rows.sort_unstable();
    rows
}

pub(crate) async fn scan_promoted_rows(table: &Table) -> Vec<(i64, i64)> {
    let stream = table
        .scan()
        .select(["id", "v"])
        .build()
        .expect("scan")
        .to_arrow()
        .await
        .expect("to_arrow");
    let batches: Vec<RecordBatch> = stream.try_collect().await.expect("collect");
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .expect("id")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("id i64");
        let vs = batch
            .column_by_name("v")
            .expect("v")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("v i64");
        for index in 0..ids.len() {
            rows.push((ids.value(index), vs.value(index)));
        }
    }
    rows.sort_unstable();
    rows
}

#[tokio::test]
async fn add_column_then_partition_field_rewrites_old_files_through_current_schema() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_unpartitioned_table(&catalog, FormatVersion::V2).await;
    let a = write_current_spec_file(&table, "a", &[(1, 10, 100)]).await;
    let b = write_current_spec_file(&table, "b", &[(2, 20, 200)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;
    assert_eq!(scan_rows(&table).await, vec![(1, 10, 100), (2, 20, 200)]);

    let table = evolve_schema(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_schema()
            .add_column("note", Type::Primitive(PrimitiveType::String)),
    )
    .await;
    let table = evolve_spec(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_partition_spec()
            .add_field_with_transform(None, "x", Transform::Bucket(4)),
    )
    .await;
    let files_before = live_data_file_paths(&table).await.len();
    let (table, result) = compact(&catalog, table).await;
    assert!(
        result.rewritten_data_files_count >= 2,
        "two old files must co-enter the rewrite, rewritten={}",
        result.rewritten_data_files_count
    );

    assert_eq!(
        scan_note_rows(&table).await,
        vec![(1, 10, 100, None), (2, 20, 200, None)],
        "old rows must survive with a null added column"
    );
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
    let bucket_1 = literal_from_long_transform(Transform::Bucket(4), 1);
    let bucket_2 = literal_from_long_transform(Transform::Bucket(4), 2);
    assert_output_matches_current_spec(&files, &table, &[
        Struct::from_iter([Some(bucket_1)]),
        Struct::from_iter([Some(bucket_2)]),
    ]);
    assert_ne!(
        live_data_file_paths(&table).await.len(),
        files_before,
        "rewrite must change the live file set"
    );
}

#[tokio::test]
async fn add_column_without_spec_change_rewrites_old_files() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let a = write_data_file(&table, "a.parquet", 1, &[(1, 10, 100)]).await;
    let b = write_data_file(&table, "b.parquet", 1, &[(1, 11, 110)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;

    let table = evolve_schema(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_schema()
            .add_column("w", Type::Primitive(PrimitiveType::Long)),
    )
    .await;
    let (table, result) = compact(&catalog, table).await;
    assert!(
        result.rewritten_data_files_count >= 2,
        "two old files must co-enter the rewrite, rewritten={}",
        result.rewritten_data_files_count
    );
    assert_eq!(
        scan_w_rows(&table).await,
        vec![(1, 10, 100, None), (1, 11, 110, None)],
        "old rows must survive with a null added column"
    );
    let files = live_data_files(&table).await;
    assert_output_matches_current_spec(&files, &table, &[Struct::from_iter([Some(
        Literal::long(1),
    )])]);
}

#[tokio::test]
async fn add_column_on_unpartitioned_table_rewrites_old_files() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_unpartitioned_table(&catalog, FormatVersion::V2).await;
    let a = write_current_spec_file(&table, "a", &[(1, 10, 100)]).await;
    let b = write_current_spec_file(&table, "b", &[(2, 20, 200)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;

    let table = evolve_schema(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_schema()
            .add_column("w", Type::Primitive(PrimitiveType::Long)),
    )
    .await;
    let (table, result) = compact(&catalog, table).await;
    assert!(
        result.rewritten_data_files_count >= 2,
        "two old files must co-enter the rewrite, rewritten={}",
        result.rewritten_data_files_count
    );
    assert_eq!(
        scan_w_rows(&table).await,
        vec![(1, 10, 100, None), (2, 20, 200, None)],
        "old rows must survive with a null added column"
    );
}

#[tokio::test]
async fn drop_column_on_unpartitioned_table_rewrites_old_files() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_unpartitioned_table(&catalog, FormatVersion::V2).await;
    let a = write_current_spec_file(&table, "a", &[(1, 10, 100)]).await;
    let b = write_current_spec_file(&table, "b", &[(2, 20, 200)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;

    let table = evolve_schema(
        &catalog,
        &table,
        Transaction::new(&table).update_schema().delete_column("z"),
    )
    .await;
    let (table, result) = compact(&catalog, table).await;
    assert!(
        result.rewritten_data_files_count >= 2,
        "two old files must co-enter the rewrite, rewritten={}",
        result.rewritten_data_files_count
    );
    assert_eq!(
        scan_xy_rows(&table).await,
        vec![(1, 10), (2, 20)],
        "kept columns must survive the drop"
    );
}

#[tokio::test]
async fn rename_column_on_unpartitioned_table_rewrites_old_files() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_unpartitioned_table(&catalog, FormatVersion::V2).await;
    let a = write_current_spec_file(&table, "a", &[(1, 10, 100)]).await;
    let b = write_current_spec_file(&table, "b", &[(2, 20, 200)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;

    let table = evolve_schema(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_schema()
            .rename_column("z", "zz"),
    )
    .await;
    let (table, result) = compact(&catalog, table).await;
    assert!(
        result.rewritten_data_files_count >= 2,
        "two old files must co-enter the rewrite, rewritten={}",
        result.rewritten_data_files_count
    );
    assert_eq!(
        scan_renamed_rows(&table).await,
        vec![(1, 10, 100), (2, 20, 200)],
        "rows must survive under the new name"
    );
}

#[tokio::test]
async fn drop_column_rewrites_old_files_without_dropped_data() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let a = write_data_file(&table, "a.parquet", 1, &[(1, 10, 100)]).await;
    let b = write_data_file(&table, "b.parquet", 1, &[(1, 11, 110)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;
    assert_eq!(scan_rows(&table).await, vec![(1, 10, 100), (1, 11, 110)]);

    let table = evolve_schema(
        &catalog,
        &table,
        Transaction::new(&table).update_schema().delete_column("z"),
    )
    .await;
    let (table, result) = compact(&catalog, table).await;
    assert!(
        result.rewritten_data_files_count >= 2,
        "two old files must co-enter the rewrite, rewritten={}",
        result.rewritten_data_files_count
    );
    assert_eq!(
        scan_xy_rows(&table).await,
        vec![(1, 10), (1, 11)],
        "kept columns must survive the drop"
    );
    assert_eq!(
        scan_pruned_xy_rows(&table, "y", 10).await,
        vec![(1, 10)],
        "pruned y=10 must return only that row"
    );
    let files = live_data_files(&table).await;
    assert_output_matches_current_spec(&files, &table, &[Struct::from_iter([Some(
        Literal::long(1),
    )])]);
}

#[tokio::test]
async fn rename_column_rewrites_old_files_under_new_name() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let a = write_data_file(&table, "a.parquet", 1, &[(1, 10, 100)]).await;
    let b = write_data_file(&table, "b.parquet", 1, &[(1, 11, 110)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;

    let table = evolve_schema(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_schema()
            .rename_column("z", "zz"),
    )
    .await;
    let (table, result) = compact(&catalog, table).await;
    assert!(
        result.rewritten_data_files_count >= 2,
        "two old files must co-enter the rewrite, rewritten={}",
        result.rewritten_data_files_count
    );
    assert_eq!(
        scan_renamed_rows(&table).await,
        vec![(1, 10, 100), (1, 11, 110)],
        "rows must survive under the new name"
    );
    assert_eq!(
        scan_pruned_renamed_rows(&table, "y", 10).await,
        vec![(1, 10, 100)],
        "pruned y=10 must return only that row"
    );
    let files = live_data_files(&table).await;
    assert_output_matches_current_spec(&files, &table, &[Struct::from_iter([Some(
        Literal::long(1),
    )])]);
}

#[tokio::test]
async fn promote_int_to_long_rewrites_old_files() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_int_id_table(&catalog, "v").await;
    let partition = Struct::from_iter([Some(Literal::long(100))]);
    let a = write_int_id_data_file(&table, "a.parquet", partition.clone(), &[(1, 100)]).await;
    let b = write_int_id_data_file(&table, "b.parquet", partition, &[(2, 100)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;

    let table = evolve_schema(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_schema()
            .update_column("id", PrimitiveType::Long),
    )
    .await;
    let (table, result) = compact(&catalog, table).await;
    assert!(
        result.rewritten_data_files_count >= 2,
        "two old files must co-enter the rewrite, rewritten={}",
        result.rewritten_data_files_count
    );
    assert_eq!(
        scan_promoted_rows(&table).await,
        vec![(1, 100), (2, 100)],
        "promoted values must survive widening"
    );
    let files = live_data_files(&table).await;
    assert_output_matches_current_spec(&files, &table, &[Struct::from_iter([Some(
        Literal::long(100),
    )])]);
}

#[tokio::test]
async fn promote_partition_source_int_to_long_rewrites_old_files() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_int_id_table(&catalog, "id").await;
    let partition = Struct::from_iter([Some(Literal::int(7))]);
    let a = write_int_id_data_file(&table, "a.parquet", partition.clone(), &[(7, 100)]).await;
    let b = write_int_id_data_file(&table, "b.parquet", partition, &[(7, 101)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;

    let table = evolve_schema(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_schema()
            .update_column("id", PrimitiveType::Long),
    )
    .await;
    let (table, result) = compact(&catalog, table).await;
    assert!(
        result.rewritten_data_files_count >= 2,
        "two old files must co-enter the rewrite, rewritten={}",
        result.rewritten_data_files_count
    );
    assert_eq!(
        scan_promoted_rows(&table).await,
        vec![(7, 100), (7, 101)],
        "promoted partition values must survive widening"
    );
    let files = live_data_files(&table).await;
    assert_output_matches_current_spec(&files, &table, &[Struct::from_iter([Some(
        Literal::long(7),
    )])]);
}

#[tokio::test]
async fn v3_deletion_vectors_survive_schema_and_spec_evolution() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V3).await;
    let a = write_data_file(&table, "a.parquet", 1, &[(1, 10, 100), (1, 11, 110)]).await;
    let a_path = a.file_path().to_string();
    let b = write_data_file(&table, "b.parquet", 2, &[(2, 20, 200)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;
    let dvs = write_dv(&table, 1, &[(a_path.as_str(), &[1u64][..])]).await;
    let table = add_deletes(&catalog, &table, dvs).await;
    assert_eq!(scan_rows(&table).await, vec![(1, 10, 100), (2, 20, 200)]);

    let table = evolve_schema(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_schema()
            .add_column("note", Type::Primitive(PrimitiveType::String)),
    )
    .await;
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
        "two old files must co-enter the rewrite, rewritten={}",
        result.rewritten_data_files_count
    );
    assert_eq!(
        scan_note_rows(&table).await,
        vec![(1, 10, 100, None), (2, 20, 200, None)],
        "deleted row must stay gone and old rows must carry a null note"
    );
    assert_eq!(
        scan_pruned_rows(&table, "y", 10).await,
        vec![(1, 10, 100)],
        "pruned y=10 must return only that row"
    );
    assert_eq!(
        result.removed_delete_files_count, 1,
        "the DV of the rewritten file must leave with it"
    );
    let files = live_data_files(&table).await;
    assert_output_matches_current_spec(&files, &table, &[
        Struct::from_iter([Some(Literal::long(10))]),
        Struct::from_iter([Some(Literal::long(20))]),
    ]);
}
