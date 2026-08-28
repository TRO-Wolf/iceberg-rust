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

//! End-to-end partition-spec-id stamping. Contract: `docs/ENGINE_CONTRACT.md` §7a.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use parquet::file::properties::WriterProperties;

use super::{PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig};
use crate::arrow::{arrow_schema_to_schema, schema_to_arrow_schema};
use crate::memory::tests::new_memory_catalog;
use crate::spec::{
    DataFile, DataFileFormat, FormatVersion, Literal, NestedField, PartitionKey, PartitionSpec,
    PrimitiveType, Schema, SchemaRef, Struct, Transform, Type, UnboundPartitionField,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::base_writer::equality_delete_writer::{
    EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
};
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Catalog, ErrorKind, TableCreation, TableIdent};

// Fixtures.

/// `1: id long`, `2: dept string`, both required.
fn test_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "dept", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build test schema")
}

/// An `identity(dept)` spec under `spec_id`.
fn identity_dept_spec(spec_id: i32) -> PartitionSpec {
    PartitionSpec::builder(test_schema())
        .with_spec_id(spec_id)
        .add_unbound_field(
            UnboundPartitionField::builder()
                .source_id(2)
                .name("dept".to_string())
                .transform(Transform::Identity)
                .build(),
        )
        .expect("add identity(dept)")
        .build()
        .expect("build identity(dept) spec")
}

/// A `truncate[5](dept)` spec under `spec_id`, partition field named `dept_trunc`.
///
/// Its partition type has the same shape as [`identity_dept_spec`]'s. For a `dept` of five
/// characters or fewer both transforms produce the same tuple. A fixture can then vary the
/// spec id and hold the tuple constant.
fn truncate5_dept_spec(spec_id: i32) -> PartitionSpec {
    PartitionSpec::builder(test_schema())
        .with_spec_id(spec_id)
        .add_unbound_field(
            UnboundPartitionField::builder()
                .source_id(2)
                .name("dept_trunc".to_string())
                .transform(Transform::Truncate(5))
                .build(),
        )
        .expect("add truncate[5](dept)")
        .build()
        .expect("build truncate[5](dept) spec")
}

/// A fresh V2 table in `catalog` under `spec`.
async fn make_table(catalog: &impl Catalog, spec: PartitionSpec) -> Table {
    let table_ident =
        TableIdent::from_strs([format!("ns-{}", uuid::Uuid::new_v4()), "t".to_string()])
            .expect("table ident");
    catalog
        .create_namespace(table_ident.namespace(), HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .schema(test_schema())
        .partition_spec(spec)
        .name(table_ident.name().to_string())
        .format_version(FormatVersion::V2)
        .build();
    catalog
        .create_table(table_ident.namespace(), creation)
        .await
        .expect("create table")
}

fn rows_batch(schema: &SchemaRef, rows: &[(i64, &str)]) -> RecordBatch {
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("arrow schema"));
    let ids: Vec<i64> = rows.iter().map(|(id, _)| *id).collect();
    let depts: Vec<&str> = rows.iter().map(|(_, dept)| *dept).collect();
    RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(depts)) as ArrayRef,
    ])
    .expect("rows batch")
}

fn rolling_builder(
    table: &Table,
    prefix: &str,
    schema: SchemaRef,
) -> RollingFileWriterBuilder<
    ParquetWriterBuilder,
    DefaultLocationGenerator,
    DefaultFileNameGenerator,
> {
    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen =
        DefaultFileNameGenerator::new(prefix.to_string(), None, DataFileFormat::Parquet);
    RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(WriterProperties::builder().build(), schema),
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    )
}

/// Write one data file, optionally under a configured spec and/or a partition key.
async fn write_data_file(
    table: &Table,
    configured_spec: Option<PartitionSpec>,
    partition_key: Option<PartitionKey>,
    rows: &[(i64, &str)],
) -> DataFile {
    let schema = table.metadata().current_schema();
    let mut builder = DataFileWriterBuilder::new(rolling_builder(table, "data", schema.clone()));
    if let Some(spec) = configured_spec {
        builder = builder.with_partition_spec(spec);
    }
    let mut writer = builder
        .build(partition_key)
        .await
        .expect("build data writer");
    writer
        .write(rows_batch(schema, rows))
        .await
        .expect("write rows");
    writer
        .close()
        .await
        .expect("close data writer")
        .into_iter()
        .next()
        .expect("one data file")
}

/// Write one position-delete file, optionally under a configured spec and/or a partition key.
async fn write_pos_delete(
    table: &Table,
    configured_spec: Option<PartitionSpec>,
    partition_key: Option<PartitionKey>,
    pairs: &[(&str, i64)],
) -> DataFile {
    let config = PositionDeleteWriterConfig::new().expect("pos-delete config");
    let mut builder = PositionDeleteFileWriterBuilder::new(
        rolling_builder(table, "pos-del", config.schema().clone()),
        config.clone(),
    );
    if let Some(spec) = configured_spec {
        builder = builder.with_partition_spec(spec);
    }
    let mut writer = builder
        .build(partition_key)
        .await
        .expect("build pos-delete writer");
    let paths: Vec<&str> = pairs.iter().map(|(path, _)| *path).collect();
    let positions: Vec<i64> = pairs.iter().map(|(_, pos)| *pos).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions)) as ArrayRef,
    ])
    .expect("pos-delete batch");
    writer.write(batch).await.expect("write pos deletes");
    writer
        .close()
        .await
        .expect("close pos-delete writer")
        .into_iter()
        .next()
        .expect("one pos-delete file")
}

/// Write one equality-delete file on `id`, optionally under a configured spec and/or a key.
///
/// `rows` are full table rows; the writer projects them down to the equality columns.
async fn write_eq_delete_on_id(
    table: &Table,
    configured_spec: Option<PartitionSpec>,
    partition_key: Option<PartitionKey>,
    rows: &[(i64, &str)],
) -> DataFile {
    let schema = table.metadata().current_schema();
    let config = EqualityDeleteWriterConfig::new(vec![1], schema.clone())
        .expect("equality-delete config on id");
    let projected = Arc::new(
        arrow_schema_to_schema(config.projected_arrow_schema_ref())
            .expect("projected iceberg schema"),
    );
    let mut builder =
        EqualityDeleteFileWriterBuilder::new(rolling_builder(table, "eq-del", projected), config);
    if let Some(spec) = configured_spec {
        builder = builder.with_partition_spec(spec);
    }
    let mut writer = builder
        .build(partition_key)
        .await
        .expect("build eq-delete writer");
    writer
        .write(rows_batch(schema, rows))
        .await
        .expect("write eq deletes");
    writer
        .close()
        .await
        .expect("close eq-delete writer")
        .into_iter()
        .next()
        .expect("one eq-delete file")
}

async fn fast_append(
    catalog: &impl Catalog,
    table: &Table,
    files: Vec<DataFile>,
) -> crate::Result<Table> {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(files)
        .apply(tx)
        .expect("apply fast_append");
    tx.commit(catalog).await
}

async fn add_deletes(catalog: &impl Catalog, table: &Table, deletes: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .row_delta()
        .add_deletes(deletes)
        .apply(tx)
        .expect("apply row_delta");
    tx.commit(catalog).await.expect("commit row_delta")
}

/// The merge-on-read live `id` set, ascending.
async fn scan_ids(table: &Table) -> Vec<i64> {
    let stream = table
        .scan()
        .select(["id"])
        .build()
        .expect("scan build")
        .to_arrow()
        .await
        .expect("to_arrow");
    let batches: Vec<RecordBatch> = stream.try_collect().await.expect("collect");
    let mut ids = Vec::new();
    for batch in batches {
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("id column is int64");
        for i in 0..col.len() {
            ids.push(col.value(i));
        }
    }
    ids.sort_unstable();
    ids
}

/// Evolve the spec by ADDING `identity(field)`; returns the table and its new default spec id.
async fn evolve_add_field(catalog: &impl Catalog, table: &Table, field: &str) -> (Table, i32) {
    let tx = Transaction::new(table);
    let tx = tx
        .update_partition_spec()
        .add_field(field)
        .apply(tx)
        .expect("apply update_partition_spec");
    let table = tx.commit(catalog).await.expect("commit spec evolution");
    let spec_id = table.metadata().default_partition_spec_id();
    (table, spec_id)
}

/// Evolve the spec by REMOVING `field`; returns the table and its new default spec id.
async fn evolve_remove_field(catalog: &impl Catalog, table: &Table, field: &str) -> (Table, i32) {
    let tx = Transaction::new(table);
    let tx = tx
        .update_partition_spec()
        .remove_field(field)
        .apply(tx)
        .expect("apply update_partition_spec");
    let table = tx.commit(catalog).await.expect("commit spec evolution");
    let spec_id = table.metadata().default_partition_spec_id();
    (table, spec_id)
}

// The two consequences.

/// Silent under-delete, in the shape an engine reaches. Spec 0 is unpartitioned and the table
/// evolved to a partitioned spec 1. Data lands under spec 1.
#[tokio::test]
async fn test_e2e_unstamped_delete_under_evolved_spec_commits_and_never_applies() {
    let catalog = new_memory_catalog().await;
    let table = make_table(
        &catalog,
        PartitionSpec::builder(test_schema())
            .with_spec_id(0)
            .build()
            .expect("unpartitioned spec 0"),
    )
    .await;
    assert!(
        table.metadata().default_partition_spec().is_unpartitioned(),
        "fixture: spec 0 is unpartitioned"
    );

    let (table, cur_spec_id) = evolve_add_field(&catalog, &table, "dept").await;
    assert_ne!(cur_spec_id, 0, "fixture: the spec evolved away from 0");

    // Data under the CURRENT (partitioned) spec, via its PartitionKey — the correct path.
    let partition = Struct::from_iter([Some(Literal::string("eng"))]);
    let partition_key = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        partition,
    )
    .expect("PartitionKey::new: valid partition tuple");
    let data = write_data_file(&table, None, Some(partition_key), &[
        (1, "eng"),
        (2, "eng"),
        (3, "eng"),
    ])
    .await;
    assert_eq!(data.partition_spec_id(), cur_spec_id);
    let data_path = data.file_path().to_string();
    let table = fast_append(&catalog, &table, vec![data])
        .await
        .expect("commit data");
    assert_eq!(scan_ids(&table).await, vec![1, 2, 3]);

    // A delete built with NEITHER a key NOR a configured spec now errors at build.
    let config = PositionDeleteWriterConfig::new().expect("pos-delete config");
    let err = PositionDeleteFileWriterBuilder::new(
        rolling_builder(&table, "pos-del", config.schema().clone()),
        config,
    )
    .build(None)
    .await
    .expect_err("build(None) with no spec must error");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(
        err.to_string().contains("unpartitioned()"),
        "unexpected error: {err}"
    );

    // Misusing unpartitioned() on this partitioned table still silent-never-applies.
    let delete = write_pos_delete(&table, Some(PartitionSpec::unpartition_spec()), None, &[
        (data_path.as_str(), 0),
        (data_path.as_str(), 1),
        (data_path.as_str(), 2),
    ])
    .await;
    assert_eq!(delete.partition_spec_id(), 0);
    let table = add_deletes(&catalog, &table, vec![delete]).await;
    assert_eq!(
        scan_ids(&table).await,
        vec![1, 2, 3],
        "an unpartitioned() delete on partitioned data never applies"
    );

    // Positive control on the same table, file, and positions. A delete that carries the data
    // file's own PartitionKey removes the rows. Without it, the survival above could mean
    // "deletes never work in this fixture".
    let correct_key = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        Struct::from_iter([Some(Literal::string("eng"))]),
    )
    .expect("PartitionKey::new: valid partition tuple");
    let correct_delete = write_pos_delete(&table, None, Some(correct_key), &[
        (data_path.as_str(), 0),
        (data_path.as_str(), 1),
        (data_path.as_str(), 2),
    ])
    .await;
    assert_eq!(correct_delete.partition_spec_id(), cur_spec_id);
    let table = add_deletes(&catalog, &table, vec![correct_delete]).await;
    assert!(
        scan_ids(&table).await.is_empty(),
        "the correctly-stamped delete DOES apply — the stamp is the only difference"
    );

    // THE FIX: configure the spec and the same unstamped call cannot produce that artifact.
    let config = PositionDeleteWriterConfig::new().expect("pos-delete config");
    let err = PositionDeleteFileWriterBuilder::new(
        rolling_builder(&table, "pos-del", config.schema().clone()),
        config,
    )
    .with_partition_spec(table.metadata().default_partition_spec().as_ref().clone())
    .build(None)
    .await
    .expect_err("a partitioned spec with no PartitionKey must be rejected");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(
        err.to_string().contains("must carry its partition tuple"),
        "unexpected error: {err}"
    );
}

/// Unwritable table, the other half of the same defect.
///
/// Spec 0 is `identity(dept)`. Removing its only field leaves an unpartitioned current spec
/// with a non-zero id. `build(None)` with no spec now errors. With the spec configured the
/// round trip works and the read side applies the delete.
#[tokio::test]
async fn test_e2e_unpartitioned_nonzero_spec_round_trips_with_configured_spec() {
    let catalog = new_memory_catalog().await;
    let table = make_table(&catalog, identity_dept_spec(0)).await;
    let (table, cur_spec_id) = evolve_remove_field(&catalog, &table, "dept").await;
    let cur_spec = table.metadata().default_partition_spec().as_ref().clone();
    assert!(
        cur_spec.is_unpartitioned(),
        "fixture: the current spec is unpartitioned"
    );
    assert_ne!(cur_spec_id, 0, "fixture: its id is NOT 0");

    // CONTROL — neither spec nor key now errors at build, before any stamp.
    let err = DataFileWriterBuilder::new(rolling_builder(
        &table,
        "data",
        table.metadata().current_schema().clone(),
    ))
    .build(None)
    .await
    .expect_err("build(None) with no spec must error");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);

    // FIXED — configure the current spec; the file claims it and commits.
    let data = write_data_file(&table, Some(cur_spec.clone()), None, &[
        (1, "eng"),
        (2, "eng"),
        (3, "eng"),
    ])
    .await;
    assert_eq!(
        data.partition_spec_id(),
        cur_spec_id,
        "the data file must claim the CURRENT unpartitioned spec"
    );
    let data_path = data.file_path().to_string();
    let table = fast_append(&catalog, &table, vec![data])
        .await
        .expect("commit data under the configured spec");
    assert_eq!(scan_ids(&table).await, vec![1, 2, 3]);

    // Assert the ROW-LEVEL outcome first. It discriminates this test from the wrong-spec twin.
    // Asserting the stamp first makes every wrong-stamp mutation red on the stamp instead.
    let delete = write_pos_delete(&table, Some(cur_spec), None, &[
        (data_path.as_str(), 0),
        (data_path.as_str(), 2),
    ])
    .await;
    let delete_spec_id = delete.partition_spec_id();
    let table = add_deletes(&catalog, &table, vec![delete]).await;
    assert_eq!(
        scan_ids(&table).await,
        vec![2],
        "positions 0 and 2 must be deleted — the delete and the data agree on the spec id"
    );
    // Corroborating guard: the delete claimed the current spec, not some other route.
    assert_eq!(delete_spec_id, cur_spec_id);
}

/// Silent under-delete, isolated on the spec id alone. The twin above cannot attribute the miss
/// to the spec id, because its delete also differs in the tuple. Here the tuple is constant:
/// `truncate[5](dept)` and `identity(dept)` both yield `{"eng"}`.
#[tokio::test]
async fn test_e2e_same_tuple_wrong_spec_id_alone_silently_under_deletes() {
    let catalog = new_memory_catalog().await;
    let old_spec = truncate5_dept_spec(0);
    let table = make_table(&catalog, old_spec.clone()).await;

    // spec 0 `truncate[5](dept)` → (remove) unpartitioned → (add) `identity(dept)`.
    let (table, _) = evolve_remove_field(&catalog, &table, "dept_trunc").await;
    let (table, cur_spec_id) = evolve_add_field(&catalog, &table, "dept").await;
    let cur_spec = table.metadata().default_partition_spec().as_ref().clone();
    assert_ne!(cur_spec_id, 0, "fixture: the current spec is not spec 0");
    assert_eq!(
        cur_spec.fields().len(),
        old_spec.fields().len(),
        "fixture: both specs have the same partition arity"
    );

    // The one tuple both specs produce for dept = "eng".
    let tuple = Struct::from_iter([Some(Literal::string("eng"))]);

    let data = write_data_file(
        &table,
        None,
        Some(
            PartitionKey::new(
                cur_spec.clone(),
                table.metadata().current_schema().clone(),
                tuple.clone(),
            )
            .expect("PartitionKey::new: valid partition tuple"),
        ),
        &[(1, "eng"), (2, "eng")],
    )
    .await;
    assert_eq!(data.partition_spec_id(), cur_spec_id);
    assert_eq!(
        data.partition, tuple,
        "fixture: the data carries {{\"eng\"}}"
    );
    let data_path = data.file_path().to_string();
    let table = fast_append(&catalog, &table, vec![data])
        .await
        .expect("commit data");
    assert_eq!(scan_ids(&table).await, vec![1, 2]);

    // The delete: SAME tuple, OLD spec id.
    let wrong_key = PartitionKey::new(
        old_spec,
        table.metadata().current_schema().clone(),
        tuple.clone(),
    )
    .expect("PartitionKey::new: valid partition tuple");
    let delete = write_pos_delete(&table, None, Some(wrong_key), &[
        (data_path.as_str(), 0),
        (data_path.as_str(), 1),
    ])
    .await;
    assert_eq!(
        delete.partition, tuple,
        "ISOLATION: the delete's tuple equals the data's, byte for byte"
    );
    assert_eq!(
        delete.partition_spec_id(),
        0,
        "ISOLATION: the spec id is the ONLY difference"
    );

    let table = add_deletes(&catalog, &table, vec![delete]).await;
    assert_eq!(
        scan_ids(&table).await,
        vec![1, 2],
        "a delete differing ONLY in spec id commits and silently never applies"
    );

    // POSITIVE CONTROL: the same positions, the same tuple, the CURRENT spec id ⇒ applied.
    let correct_key = PartitionKey::new(cur_spec, table.metadata().current_schema().clone(), tuple)
        .expect("PartitionKey::new: valid partition tuple");
    let correct_delete = write_pos_delete(&table, None, Some(correct_key), &[
        (data_path.as_str(), 0),
        (data_path.as_str(), 1),
    ])
    .await;
    let correct_spec_id = correct_delete.partition_spec_id();
    let table = add_deletes(&catalog, &table, vec![correct_delete]).await;
    assert!(
        scan_ids(&table).await.is_empty(),
        "the same delete under the matching spec id DOES apply"
    );
    assert_eq!(correct_spec_id, cur_spec_id);
}

/// A keyless equality delete is GLOBAL, not inert. The spec applies an equality delete stored
/// with an unpartitioned spec as a global delete. Rust routes on the file's empty tuple, Java
/// on the spec being unpartitioned.
#[tokio::test]
async fn test_e2e_keyless_equality_delete_is_global_not_inert() {
    let catalog = new_memory_catalog().await;
    let table = make_table(
        &catalog,
        PartitionSpec::builder(test_schema())
            .with_spec_id(0)
            .build()
            .expect("unpartitioned spec 0"),
    )
    .await;
    let (table, cur_spec_id) = evolve_add_field(&catalog, &table, "dept").await;
    let cur_spec = table.metadata().default_partition_spec().as_ref().clone();
    let schema = table.metadata().current_schema().clone();

    let eng = write_data_file(
        &table,
        None,
        Some(
            PartitionKey::new(
                cur_spec.clone(),
                schema.clone(),
                Struct::from_iter([Some(Literal::string("eng"))]),
            )
            .expect("PartitionKey::new: valid partition tuple"),
        ),
        &[(1, "eng"), (2, "eng")],
    )
    .await;
    let ops = write_data_file(
        &table,
        None,
        Some(
            PartitionKey::new(
                cur_spec,
                schema,
                Struct::from_iter([Some(Literal::string("ops"))]),
            )
            .expect("PartitionKey::new: valid partition tuple"),
        ),
        &[(1, "ops"), (3, "ops")],
    )
    .await;
    assert_eq!(eng.partition_spec_id(), cur_spec_id);
    let table = fast_append(&catalog, &table, vec![eng, ops])
        .await
        .expect("commit data");
    assert_eq!(scan_ids(&table).await, vec![1, 1, 2, 3]);

    // Explicit unpartitioned spec + empty tuple: a GLOBAL equality delete.
    let delete = write_eq_delete_on_id(&table, Some(PartitionSpec::unpartition_spec()), None, &[(
        1, "eng",
    )])
    .await;
    assert_eq!(delete.partition_spec_id(), 0);
    assert!(
        delete.partition.fields().is_empty(),
        "the keyless equality delete carries an empty tuple"
    );

    let table = add_deletes(&catalog, &table, vec![delete]).await;
    assert_eq!(
        scan_ids(&table).await,
        vec![2, 3],
        "id = 1 is gone from BOTH partitions: the keyless equality delete is GLOBAL, and it \
         ignored the spec id it claimed"
    );
}

/// Contrast leg: an equality delete with a non-empty tuple is partition-scoped.
///
/// Same fixture as the global twin, but the delete carries the `eng` `PartitionKey`. It removes
/// `id = 1` from `eng` only, so `ops` keeps its `id = 1`. That is `[1, 2, 3]` against `[2, 3]`.
#[tokio::test]
async fn test_e2e_keyed_equality_delete_is_partition_scoped() {
    let catalog = new_memory_catalog().await;
    let table = make_table(
        &catalog,
        PartitionSpec::builder(test_schema())
            .with_spec_id(0)
            .build()
            .expect("unpartitioned spec 0"),
    )
    .await;
    let (table, cur_spec_id) = evolve_add_field(&catalog, &table, "dept").await;
    let cur_spec = table.metadata().default_partition_spec().as_ref().clone();
    let schema = table.metadata().current_schema().clone();
    let eng_tuple = Struct::from_iter([Some(Literal::string("eng"))]);

    let eng = write_data_file(
        &table,
        None,
        Some(
            PartitionKey::new(cur_spec.clone(), schema.clone(), eng_tuple.clone())
                .expect("PartitionKey::new: valid partition tuple"),
        ),
        &[(1, "eng"), (2, "eng")],
    )
    .await;
    let ops = write_data_file(
        &table,
        None,
        Some(
            PartitionKey::new(
                cur_spec.clone(),
                schema.clone(),
                Struct::from_iter([Some(Literal::string("ops"))]),
            )
            .expect("PartitionKey::new: valid partition tuple"),
        ),
        &[(1, "ops"), (3, "ops")],
    )
    .await;
    let table = fast_append(&catalog, &table, vec![eng, ops])
        .await
        .expect("commit data");
    assert_eq!(scan_ids(&table).await, vec![1, 1, 2, 3]);

    let delete = write_eq_delete_on_id(
        &table,
        None,
        Some(
            PartitionKey::new(cur_spec, schema, eng_tuple.clone())
                .expect("PartitionKey::new: valid partition tuple"),
        ),
        &[(1, "eng")],
    )
    .await;
    let delete_spec_id = delete.partition_spec_id();
    let delete_partition = delete.partition.clone();

    let table = add_deletes(&catalog, &table, vec![delete]).await;
    assert_eq!(
        scan_ids(&table).await,
        vec![1, 2, 3],
        "only eng's id = 1 is deleted — ops's survives, so this delete is partition-scoped"
    );
    // Corroborating guards, after the row-level outcome.
    assert_eq!(delete_spec_id, cur_spec_id);
    assert_eq!(delete_partition, eng_tuple);
}
