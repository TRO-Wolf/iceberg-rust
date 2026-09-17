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

use crate::expr::Reference;
use crate::memory::tests::new_memory_catalog;
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, FormatVersion, Literal,
    NestedField, PartitionSpec, PrimitiveType, Schema, Struct, Transform, Type,
    UnboundPartitionField,
};
use crate::table::Table;
use crate::transaction::action::occ_scoped_tests::{
    append_files, data_file, live_file_paths, x_equals,
};
use crate::transaction::tests::make_v2_minimal_table_in_catalog;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{Catalog, ErrorKind, TableCreation, TableIdent};

fn int_data_file(path: &str, part_value: i32) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::int(part_value))]))
        .build()
        .unwrap()
}

fn long_schema(int_x: bool) -> Schema {
    let field_x = if int_x {
        NestedField::required(1, "x", Type::Primitive(PrimitiveType::Int)).into()
    } else {
        NestedField::required(1, "x", Type::Primitive(PrimitiveType::Long)).into()
    };
    Schema::builder()
        .with_fields(vec![
            field_x,
            NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(3, "z", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .unwrap()
}

async fn create_table_with_spec(
    catalog: &impl Catalog,
    schema: Schema,
    spec: PartitionSpec,
) -> Table {
    let table_ident =
        TableIdent::from_strs([format!("ns1-{}", uuid::Uuid::new_v4()), "test1".to_string()])
            .unwrap();
    catalog
        .create_namespace(table_ident.namespace(), HashMap::new())
        .await
        .unwrap();
    let table_creation = TableCreation::builder()
        .schema(schema)
        .partition_spec(spec)
        .name(table_ident.name().to_string())
        .format_version(FormatVersion::V2)
        .build();
    catalog
        .create_table(table_ident.namespace(), table_creation)
        .await
        .unwrap()
}

fn identity_spec(schema: &Schema) -> PartitionSpec {
    PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_unbound_field(
            UnboundPartitionField::builder()
                .source_id(1)
                .name("x".to_string())
                .transform(Transform::Identity)
                .build(),
        )
        .unwrap()
        .build()
        .unwrap()
}

fn truncate_spec(schema: &Schema) -> PartitionSpec {
    PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_unbound_field(
            UnboundPartitionField::builder()
                .source_id(1)
                .name("x_trunc_10".to_string())
                .transform(Transform::Truncate(10))
                .build(),
        )
        .unwrap()
        .build()
        .unwrap()
}

#[tokio::test]
async fn row_delta_truncate_range_filter_conflicts_on_boundary_partition() {
    let catalog = new_memory_catalog().await;
    let schema = long_schema(false);
    let table = create_table_with_spec(&catalog, schema, truncate_spec(&long_schema(false))).await;
    let table = append_files(&catalog, &table, vec![data_file("test/base.parquet", 10)]).await;

    let tx = Transaction::new(&table);
    let action = tx
        .row_delta()
        .add_data_files(vec![data_file("test/op.parquet", 10)])
        .conflict_detection_filter(Reference::new("x").greater_than(Datum::long(15)))
        .validate_no_conflicting_data_files();
    let tx = action.apply(tx).unwrap();

    let _concurrent = append_files(&catalog, &table, vec![data_file(
        "test/trunc-10.parquet",
        10,
    )])
    .await;

    let err = tx.commit(&catalog).await.expect_err(
        "truncate[10] partition 10 holds rows above 15, so it must conflict under x > 15",
    );
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(!err.retryable());
    assert!(err.message().contains("test/trunc-10.parquet"));
}

#[tokio::test]
async fn row_delta_older_spec_file_without_filter_source_stays_conflicting() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![data_file("test/base.parquet", 5)]).await;

    let evolve_tx = Transaction::new(&table);
    let evolve_action = evolve_tx.update_partition_spec().add_field("y");
    let evolve_tx = evolve_action.apply(evolve_tx).unwrap();
    let table = evolve_tx.commit(&catalog).await.unwrap();
    assert_eq!(
        table.metadata().default_partition_spec_id(),
        1,
        "the evolution must mint spec 1 for the older-spec pin"
    );

    let tx = Transaction::new(&table);
    let action = tx
        .row_delta()
        .add_data_files(vec![data_file("test/op.parquet", 5)])
        .conflict_detection_filter(Reference::new("y").equal_to(Datum::long(1)))
        .validate_no_conflicting_data_files();
    let tx = action.apply(tx).unwrap();

    let _concurrent = append_files(&catalog, &table, vec![data_file(
        "test/old-spec.parquet",
        5,
    )])
    .await;

    let err = tx
        .commit(&catalog)
        .await
        .expect_err("a spec-0 file carries no y partition, so the y filter cannot prune it");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(!err.retryable());
    assert!(err.message().contains("test/old-spec.parquet"));
}

#[tokio::test]
async fn row_delta_promoted_identity_source_conflicts_on_match() {
    let catalog = new_memory_catalog().await;
    let schema = long_schema(true);
    let spec = identity_spec(&schema);
    let table = create_table_with_spec(&catalog, schema, spec).await;
    let table = append_files(&catalog, &table, vec![int_data_file(
        "test/base.parquet",
        0,
    )])
    .await;

    let tx = Transaction::new(&table);
    let action = tx
        .row_delta()
        .add_data_files(vec![data_file("test/op.parquet", 0)])
        .conflict_detection_filter(x_equals(1))
        .validate_no_conflicting_data_files();
    let tx = action.apply(tx).unwrap();

    let _concurrent = append_files(&catalog, &table, vec![int_data_file(
        "test/concurrent.parquet",
        1,
    )])
    .await;

    let promote_tx = Transaction::new(&catalog.load_table(table.identifier()).await.unwrap());
    let promote_action = promote_tx
        .update_schema()
        .update_column("x", PrimitiveType::Long);
    let promote_tx = promote_action.apply(promote_tx).unwrap();
    let _promoted = promote_tx.commit(&catalog).await.unwrap();

    let err = tx.commit(&catalog).await.expect_err(
        "an Int-tuple file in partition 1 must conflict a long x = 1 filter after promotion",
    );
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(!err.retryable());
    assert!(err.message().contains("test/concurrent.parquet"));
}

#[tokio::test]
async fn row_delta_promoted_identity_source_commits_on_mismatch() {
    let catalog = new_memory_catalog().await;
    let schema = long_schema(true);
    let spec = identity_spec(&schema);
    let table = create_table_with_spec(&catalog, schema, spec).await;
    let table = append_files(&catalog, &table, vec![int_data_file(
        "test/base.parquet",
        0,
    )])
    .await;

    let tx = Transaction::new(&table);
    let action = tx
        .row_delta()
        .add_data_files(vec![data_file("test/op.parquet", 0)])
        .conflict_detection_filter(x_equals(1))
        .validate_no_conflicting_data_files();
    let tx = action.apply(tx).unwrap();

    let _concurrent = append_files(&catalog, &table, vec![int_data_file(
        "test/concurrent.parquet",
        5,
    )])
    .await;

    let promote_tx = Transaction::new(&catalog.load_table(table.identifier()).await.unwrap());
    let promote_action = promote_tx
        .update_schema()
        .update_column("x", PrimitiveType::Long);
    let promote_tx = promote_action.apply(promote_tx).unwrap();
    let _promoted = promote_tx.commit(&catalog).await.unwrap();

    let table = tx.commit(&catalog).await.expect(
        "an Int-tuple file in partition 5 must not conflict a long x = 1 filter after promotion",
    );
    let live = live_file_paths(&table).await;
    assert!(live.contains("test/op.parquet"));
    assert!(live.contains("test/concurrent.parquet"));
}
