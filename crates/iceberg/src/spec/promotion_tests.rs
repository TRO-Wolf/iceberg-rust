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

use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
};
use futures::TryStreamExt;
use tempfile::TempDir;

use crate::expr::accessor::StructAccessor;
use crate::expr::visitors::inclusive_metrics_evaluator::InclusiveMetricsEvaluator;
use crate::expr::visitors::strict_metrics_evaluator::StrictMetricsEvaluator;
use crate::expr::{Bind, BoundPredicate, Predicate, Reference};
use crate::io::LocalFsStorageFactory;
use crate::memory::MemoryCatalogBuilder;
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, Datum, FormatVersion, Literal, NestedField,
    PartitionKey, PartitionSpec, PrimitiveType, Schema, SchemaRef, Struct, Transform, Type,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::writer::base_writer::equality_delete_writer::{
    EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
};
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};

fn promoted_schema() -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "f", Type::Primitive(PrimitiveType::Double)).into(),
            ])
            .build()
            .expect("promoted schema must build"),
    )
}

fn pre_promotion_file() -> DataFile {
    DataFile {
        content: DataContentType::Data,
        file_path: "/promoted/old.parquet".to_string(),
        file_format: DataFileFormat::Parquet,
        partition: Struct::empty(),
        record_count: 2,
        file_size_in_bytes: 10,
        column_sizes: HashMap::new(),
        value_counts: HashMap::from([(1, 2), (2, 2)]),
        null_value_counts: HashMap::from([(1, 0), (2, 0)]),
        nan_value_counts: HashMap::from([(2, 0)]),
        lower_bounds: HashMap::from([(1, Datum::int(1)), (2, Datum::float(1.5))]),
        upper_bounds: HashMap::from([(1, Datum::int(2)), (2, Datum::float(2.5))]),
        key_metadata: None,
        split_offsets: None,
        equality_ids: None,
        sort_order_id: None,
        partition_spec_id: 0,
        first_row_id: None,
        referenced_data_file: None,
        content_offset: None,
        content_size_in_bytes: None,
    }
}

fn bound(predicate: Predicate) -> BoundPredicate {
    predicate
        .bind(promoted_schema(), true)
        .expect("predicate must bind against the promoted schema")
}

fn might_match(predicate: Predicate) -> bool {
    InclusiveMetricsEvaluator::eval(&bound(predicate), &pre_promotion_file(), false)
        .expect("inclusive metrics evaluation")
}

fn must_match(predicate: Predicate) -> bool {
    StrictMetricsEvaluator::eval(&bound(predicate), &pre_promotion_file())
        .expect("strict metrics evaluation")
}

#[test]
fn inclusive_metrics_keep_an_int_bounded_file_for_long_predicates() {
    assert!(might_match(Reference::new("id").less_than(Datum::long(2))));
    assert!(might_match(
        Reference::new("id").less_than_or_equal_to(Datum::long(1))
    ));
    assert!(might_match(
        Reference::new("id").greater_than(Datum::long(1))
    ));
    assert!(might_match(
        Reference::new("id").greater_than_or_equal_to(Datum::long(2))
    ));
    assert!(might_match(
        Reference::new("id").is_in((4..=25).chain([1, 2]).map(Datum::long))
    ));
    assert!(!might_match(Reference::new("id").less_than(Datum::long(1))));
    assert!(!might_match(
        Reference::new("id").greater_than(Datum::long(2))
    ));
    assert!(!might_match(
        Reference::new("id").is_in([Datum::long(5), Datum::long(3_000_000_000_i64)])
    ));
}

#[test]
fn inclusive_metrics_keep_a_float_bounded_file_for_double_predicates() {
    assert!(might_match(
        Reference::new("f").less_than(Datum::double(2.0))
    ));
    assert!(might_match(
        Reference::new("f").greater_than(Datum::double(2.0))
    ));
    assert!(!might_match(
        Reference::new("f").less_than(Datum::double(1.5))
    ));
    assert!(!might_match(
        Reference::new("f").greater_than(Datum::double(2.5))
    ));
}

#[test]
fn strict_metrics_decide_an_int_bounded_file_under_long_predicates() {
    assert!(!must_match(
        Reference::new("id").not_equal_to(Datum::long(1))
    ));
    assert!(!must_match(
        Reference::new("id").is_not_in([Datum::long(1), Datum::long(5)])
    ));
    assert!(must_match(Reference::new("id").less_than(Datum::long(5))));
    assert!(must_match(
        Reference::new("id").greater_than(Datum::long(0))
    ));
    assert!(must_match(
        Reference::new("id").not_equal_to(Datum::long(3_000_000_000_i64))
    ));
    assert!(must_match(
        Reference::new("f").less_than(Datum::double(3.0))
    ));
}

#[test]
fn partition_accessor_reads_pre_promotion_literals_under_the_promoted_type() {
    let tuple = Struct::from_iter([Some(Literal::int(7)), Some(Literal::float(1.5))]);
    assert_eq!(
        StructAccessor::new(0, PrimitiveType::Long)
            .get(&tuple)
            .expect("an int literal reads under long"),
        Some(Datum::long(7))
    );
    assert_eq!(
        StructAccessor::new(1, PrimitiveType::Double)
            .get(&tuple)
            .expect("a float literal reads under double"),
        Some(Datum::double(1.5))
    );
    assert!(
        StructAccessor::new(0, PrimitiveType::String)
            .get(&tuple)
            .is_err()
    );
}

#[test]
fn partition_key_new_promotes_a_pre_promotion_tuple() {
    let int_schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()
        .expect("int schema must build");
    let spec = PartitionSpec::builder(int_schema)
        .with_spec_id(0)
        .add_partition_field("id", "id", Transform::Identity)
        .expect("identity(id)")
        .build()
        .expect("spec must build");
    let long_schema: SchemaRef = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .expect("long schema must build"),
    );
    let key = PartitionKey::new(
        spec,
        long_schema,
        Struct::from_iter([Some(Literal::int(7))]),
    )
    .expect("a tuple written before a legal promotion constructs a key");
    assert_eq!(key.data(), &Struct::from_iter([Some(Literal::long(7))]));
}

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

async fn create_table(catalog: &impl Catalog, partition_column: Option<&str>) -> Table {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "v", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("int id schema");
    let mut spec = PartitionSpec::builder(schema.clone()).with_spec_id(0);
    if let Some(column) = partition_column {
        spec = spec
            .add_partition_field(column, column, Transform::Identity)
            .expect("identity partition field");
    }
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("t".to_string())
        .schema(schema)
        .partition_spec(spec.build().expect("spec"))
        .format_version(FormatVersion::V2)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

async fn commit(catalog: &impl Catalog, tx: Transaction) -> Table {
    tx.commit(catalog).await.expect("commit")
}

async fn promote_id(catalog: &impl Catalog, table: &Table) -> Table {
    let tx = Transaction::new(table);
    let action = tx.update_schema().update_column("id", PrimitiveType::Long);
    commit(catalog, action.apply(tx).expect("apply promotion")).await
}

async fn append(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let action = tx.fast_append().add_data_files(files);
    commit(catalog, action.apply(tx).expect("apply append")).await
}

async fn write_file(table: &Table, name: &str, partition: Struct, rows: &[(i64, i64)]) -> DataFile {
    let schema = table.metadata().current_schema().clone();
    let arrow_schema = Arc::new(crate::arrow::schema_to_arrow_schema(&schema).expect("arrow"));
    let ids: ArrayRef = match schema.field_by_id(1).map(|field| field.field_type.as_ref()) {
        Some(Type::Primitive(PrimitiveType::Int)) => Arc::new(Int32Array::from(
            rows.iter()
                .map(|(id, _)| i32::try_from(*id).expect("int id"))
                .collect::<Vec<_>>(),
        )),
        _ => Arc::new(Int64Array::from(
            rows.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        )),
    };
    let values: ArrayRef = Arc::new(Int64Array::from(
        rows.iter().map(|(_, value)| *value).collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::try_new(arrow_schema, vec![ids, values]).expect("batch");
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
        .partition(partition)
        .build()
        .expect("data file")
}

async fn rows(table: &Table, filter: Option<Predicate>, row_selection: bool) -> Vec<(i64, i64)> {
    let mut scan = table
        .scan()
        .select(["id", "v"])
        .with_row_selection_enabled(row_selection);
    if let Some(predicate) = filter {
        scan = scan.with_filter(predicate);
    }
    let batches: Vec<RecordBatch> = scan
        .build()
        .expect("scan")
        .to_arrow()
        .await
        .expect("to_arrow")
        .try_collect()
        .await
        .expect("collect");
    let mut out = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("id is long");
        let values = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("v is long");
        for index in 0..ids.len() {
            out.push((ids.value(index), values.value(index)));
        }
    }
    out.sort_unstable();
    out
}

async fn mixed_era_table(catalog: &impl Catalog, partition_column: Option<&str>) -> Table {
    let table = create_table(catalog, partition_column).await;
    let old_partition = |id: i32| match partition_column {
        Some(_) => Struct::from_iter([Some(Literal::int(id))]),
        None => Struct::empty(),
    };
    let first = write_file(&table, "old-1.parquet", old_partition(1), &[(1, 10)]).await;
    let second = write_file(&table, "old-2.parquet", old_partition(2), &[(2, 20)]).await;
    let table = append(catalog, &table, vec![first, second]).await;
    let table = promote_id(catalog, &table).await;
    let new_partition = match partition_column {
        Some(_) => Struct::from_iter([Some(Literal::long(3_000_000_000_i64))]),
        None => Struct::empty(),
    };
    let new = write_file(&table, "new-3.parquet", new_partition, &[(
        3_000_000_000,
        30,
    )])
    .await;
    append(catalog, &table, vec![new]).await
}

#[tokio::test]
async fn mixed_era_range_and_in_filters_return_pre_promotion_rows() {
    let (catalog, _guard) = local_catalog().await;
    let table = mixed_era_table(&catalog, None).await;
    let id = || Reference::new("id");
    assert_eq!(
        rows(&table, Some(id().less_than(Datum::long(2))), false).await,
        vec![(1, 10)]
    );
    assert_eq!(
        rows(&table, Some(id().greater_than(Datum::long(1))), false).await,
        vec![(2, 20), (3_000_000_000, 30)]
    );
    assert_eq!(
        rows(
            &table,
            Some(id().is_in((4..=25).chain([1, 2]).map(Datum::long))),
            false
        )
        .await,
        vec![(1, 10), (2, 20)]
    );
    assert_eq!(
        rows(
            &table,
            Some(id().less_than_or_equal_to(Datum::long(2))),
            true
        )
        .await,
        vec![(1, 10), (2, 20)]
    );
}

#[tokio::test]
async fn promoted_identity_partition_source_filters_and_plans_long_partitions() {
    let (catalog, _guard) = local_catalog().await;
    let table = mixed_era_table(&catalog, Some("id")).await;
    let id = || Reference::new("id");
    assert_eq!(
        rows(&table, Some(id().equal_to(Datum::long(1))), false).await,
        vec![(1, 10)]
    );
    assert_eq!(
        rows(&table, Some(id().less_than(Datum::long(2))), false).await,
        vec![(1, 10)]
    );
    let tasks: Vec<_> = table
        .scan()
        .build()
        .expect("scan")
        .plan_files()
        .await
        .expect("plan")
        .try_collect()
        .await
        .expect("tasks");
    assert_eq!(tasks.len(), 3);
    for task in tasks {
        let partition = task.partition.expect("task partition");
        let literal = partition.fields()[0].clone().expect("non-null partition");
        assert!(
            matches!(
                literal,
                Literal::Primitive(crate::spec::PrimitiveLiteral::Long(_))
            ),
            "partition {literal:?} must read under the promoted long type"
        );
    }
}

#[tokio::test]
async fn equality_delete_written_after_promotion_applies_to_a_pre_promotion_partition() {
    let (catalog, _guard) = local_catalog().await;
    let table = create_table(&catalog, Some("id")).await;
    let old = write_file(
        &table,
        "old-7.parquet",
        Struct::from_iter([Some(Literal::int(7))]),
        &[(7, 100), (7, 101)],
    )
    .await;
    let table = append(&catalog, &table, vec![old]).await;
    let table = promote_id(&catalog, &table).await;
    let schema = table.metadata().current_schema().clone();
    let config = EqualityDeleteWriterConfig::new(vec![2], schema.clone()).expect("eq config");
    let delete_schema = Arc::new(
        crate::arrow::arrow_schema_to_schema(config.projected_arrow_schema_ref())
            .expect("delete schema"),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            delete_schema,
        ),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).expect("locations"),
        DefaultFileNameGenerator::new("eq-del".to_string(), None, DataFileFormat::Parquet),
    );
    let key = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        schema.clone(),
        Struct::from_iter([Some(Literal::long(7))]),
    )
    .expect("post-promotion partition key");
    let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .build(Some(key))
        .await
        .expect("eq delete writer");
    let batch = RecordBatch::try_new(
        Arc::new(crate::arrow::schema_to_arrow_schema(&schema).expect("arrow")),
        vec![
            Arc::new(Int64Array::from(vec![7_i64])) as ArrayRef,
            Arc::new(Int64Array::from(vec![100_i64])) as ArrayRef,
        ],
    )
    .expect("delete batch");
    writer.write(batch).await.expect("write delete");
    let deletes = writer.close().await.expect("close delete");
    let tx = Transaction::new(&table);
    let action = tx.row_delta().add_deletes(deletes);
    let table = commit(&catalog, action.apply(tx).expect("apply row delta")).await;
    assert_eq!(rows(&table, None, false).await, vec![(7, 101)]);
}

#[tokio::test]
async fn replace_partitions_after_promotion_drops_the_pre_promotion_partition() {
    let (catalog, _guard) = local_catalog().await;
    let table = create_table(&catalog, Some("id")).await;
    let old = write_file(
        &table,
        "old-7.parquet",
        Struct::from_iter([Some(Literal::int(7))]),
        &[(7, 100)],
    )
    .await;
    let table = append(&catalog, &table, vec![old]).await;
    let table = promote_id(&catalog, &table).await;
    let new = write_file(
        &table,
        "new-7.parquet",
        Struct::from_iter([Some(Literal::long(7))]),
        &[(7, 999)],
    )
    .await;
    let tx = Transaction::new(&table);
    let action = tx.replace_partitions().add_file(new);
    let table = commit(
        &catalog,
        action.apply(tx).expect("apply replace partitions"),
    )
    .await;
    assert_eq!(rows(&table, None, false).await, vec![(7, 999)]);
}

#[tokio::test]
async fn overwrite_by_row_filter_on_a_promoted_identity_partition_replaces_it() {
    let (catalog, _guard) = local_catalog().await;
    let table = create_table(&catalog, Some("id")).await;
    let old = write_file(
        &table,
        "old-7.parquet",
        Struct::from_iter([Some(Literal::int(7))]),
        &[(7, 100)],
    )
    .await;
    let other = write_file(
        &table,
        "old-8.parquet",
        Struct::from_iter([Some(Literal::int(8))]),
        &[(8, 200)],
    )
    .await;
    let table = append(&catalog, &table, vec![old, other]).await;
    let table = promote_id(&catalog, &table).await;
    let new = write_file(
        &table,
        "new-7.parquet",
        Struct::from_iter([Some(Literal::long(7))]),
        &[(7, 999)],
    )
    .await;
    let tx = Transaction::new(&table);
    let action = tx
        .overwrite_files()
        .overwrite_by_row_filter(Reference::new("id").equal_to(Datum::long(7)))
        .add_file(new);
    let table = commit(&catalog, action.apply(tx).expect("apply overwrite")).await;
    assert_eq!(rows(&table, None, false).await, vec![(7, 999), (8, 200)]);
}

async fn write_float_file(table: &Table, name: &str, values: &[(f64, i64)]) -> DataFile {
    let schema = table.metadata().current_schema().clone();
    let arrow_schema = Arc::new(crate::arrow::schema_to_arrow_schema(&schema).expect("arrow"));
    let floats: ArrayRef = match schema.field_by_id(1).map(|field| field.field_type.as_ref()) {
        Some(Type::Primitive(PrimitiveType::Float)) => Arc::new(Float32Array::from(
            values
                .iter()
                .map(|(value, _)| format!("{value}").parse::<f32>().expect("f32 value"))
                .collect::<Vec<_>>(),
        )),
        _ => Arc::new(Float64Array::from(
            values.iter().map(|(value, _)| *value).collect::<Vec<_>>(),
        )),
    };
    let longs: ArrayRef = Arc::new(Int64Array::from(
        values.iter().map(|(_, long)| *long).collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::try_new(arrow_schema, vec![floats, longs]).expect("float batch");
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
        .expect("float data file")
}

#[tokio::test]
async fn row_selection_keeps_float_pages_under_a_promoted_double() {
    let (catalog, _guard) = local_catalog().await;
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "f", Type::Primitive(PrimitiveType::Float)).into(),
            NestedField::required(2, "v", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("float schema");
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("floats".to_string())
        .schema(schema)
        .partition_spec(
            PartitionSpec::builder(
                Schema::builder()
                    .with_fields(vec![
                        NestedField::required(1, "f", Type::Primitive(PrimitiveType::Float)).into(),
                    ])
                    .build()
                    .expect("spec schema"),
            )
            .with_spec_id(0)
            .build()
            .expect("unpartitioned spec"),
        )
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create float table");
    let old = write_float_file(&table, "old-f.parquet", &[(1.5, 10), (2.5, 20)]).await;
    let table = append(&catalog, &table, vec![old]).await;
    let tx = Transaction::new(&table);
    let action = tx.update_schema().update_column("f", PrimitiveType::Double);
    let table = commit(&catalog, action.apply(tx).expect("apply float promotion")).await;
    let new = write_float_file(&table, "new-f.parquet", &[(3.5, 30)]).await;
    let table = append(&catalog, &table, vec![new]).await;
    let batches: Vec<RecordBatch> = table
        .scan()
        .select(["f", "v"])
        .with_row_selection_enabled(true)
        .with_filter(Reference::new("f").less_than(Datum::double(2.0)))
        .build()
        .expect("scan")
        .to_arrow()
        .await
        .expect("to_arrow")
        .try_collect()
        .await
        .expect("collect");
    let values: Vec<i64> = batches
        .iter()
        .flat_map(|batch| {
            let column = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("v long");
            (0..column.len())
                .map(|row| column.value(row))
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(values, vec![10]);
}
