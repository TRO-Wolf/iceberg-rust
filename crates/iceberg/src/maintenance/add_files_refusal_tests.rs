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

use super::add_files::{AddFiles, AddFilesSource};
use super::add_files_tests::{
    flat_source, hive_source, id_v_cat_schema, id_v_schema, local_fs_catalog, long_column,
    source_root, write_source_file,
};
use crate::spec::{
    FormatVersion, NestedField, PartitionSpec, PrimitiveType, Schema, Transform, Type,
};
use crate::table::Table;
use crate::{Catalog, ErrorKind, NamespaceIdent, TableCreation, TableIdent};

async fn create_table_with_spec(
    catalog: &impl Catalog,
    schema: Schema,
    spec: PartitionSpec,
) -> Table {
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let ident = TableIdent::new(namespace.clone(), "t".to_string());
    let creation = TableCreation::builder()
        .name(ident.name().to_string())
        .schema(schema)
        .partition_spec(spec)
        .format_version(FormatVersion::V2)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

fn identity_spec(schema: &Schema, columns: &[&str]) -> PartitionSpec {
    let mut builder = PartitionSpec::builder(schema.clone()).with_spec_id(0);
    for column in columns {
        builder = builder
            .add_partition_field(*column, *column, Transform::Identity)
            .expect("add partition field");
    }
    builder.build().expect("build spec")
}

fn id_typed_part_schema(part_type: PrimitiveType) -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::optional(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(2, "part", Type::Primitive(part_type))),
        ])
        .build()
        .expect("build id/part schema")
}

fn id_cat_dept_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::optional(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                2,
                "cat",
                Type::Primitive(PrimitiveType::String),
            )),
            Arc::new(NestedField::optional(
                3,
                "dept",
                Type::Primitive(PrimitiveType::String),
            )),
        ])
        .build()
        .expect("build id/cat/dept schema")
}

#[tokio::test]
async fn a_partition_filter_wider_than_the_spec_is_refused() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let schema = id_v_cat_schema();
    let spec = identity_spec(&schema, &["cat"]);
    let table = create_table_with_spec(&catalog, schema, spec).await;
    let root = source_root(&temp_dir, "wide-filter");
    hive_source(&table, &root).await;

    let error = AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .partition_filter(HashMap::from([
            ("cat".to_string(), "x".to_string()),
            ("dept".to_string(), "hr".to_string()),
        ]))
        .execute(&catalog)
        .await
        .expect_err("a filter wider than the spec is refused");

    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error.to_string().contains(&format!(
            "Cannot add data files to target table {} because that table is partitioned, but the number of columns in the provided partition filter (2) is greater than the number of partitioned columns in table (1)",
            table.identifier()
        )),
        "Java SparkTableUtil.validatePartitionFilter clause 1: {error}"
    );
}

#[tokio::test]
async fn a_partition_filter_naming_a_non_partition_column_is_refused() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let schema = id_cat_dept_schema();
    let spec = identity_spec(&schema, &["cat", "dept"]);
    let table = create_table_with_spec(&catalog, schema, spec).await;
    let root = source_root(&temp_dir, "unknown-filter");
    write_source_file(
        &table,
        &format!("{root}/cat=x/dept=hr/part-00000.parquet"),
        &[("id", long_column(&[1]))],
    )
    .await;

    let error = AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .partition_filter(HashMap::from([("zzz".to_string(), "1".to_string())]))
        .execute(&catalog)
        .await
        .expect_err("a filter naming a non-partition column is refused");

    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    let name = table.identifier().to_string();
    assert!(
        error.to_string().contains(&format!(
            "Cannot add files to target table {name}. {name} is partitioned but the specified partition filter refers to columns that are not partitioned: zzz . Valid partition columns: [cat,dept]"
        )),
        "Java SparkTableUtil.validatePartitionFilter clause 2: {error}"
    );
}

#[tokio::test]
async fn a_partition_filter_matching_no_partition_is_refused() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let schema = id_v_cat_schema();
    let spec = identity_spec(&schema, &["cat"]);
    let table = create_table_with_spec(&catalog, schema, spec).await;
    let root = source_root(&temp_dir, "empty-filter");
    hive_source(&table, &root).await;

    let error = AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .partition_filter(HashMap::from([("cat".to_string(), "absent".to_string())]))
        .execute(&catalog)
        .await
        .expect_err("a filter that matches no partition is refused");

    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error.to_string().contains(&format!(
            "Cannot find any matching partitions in table {}",
            table.identifier()
        )),
        "Java AddFilesProcedure.importFileTable: {error}"
    );
}

#[tokio::test]
async fn the_duplicate_refusal_is_a_typed_error_naming_every_duplicate() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let schema = id_v_cat_schema();
    let spec = identity_spec(&schema, &["cat"]);
    let table = create_table_with_spec(&catalog, schema, spec).await;
    let root = source_root(&temp_dir, "dup-typed");
    hive_source(&table, &root).await;

    AddFiles::new(table.clone(), AddFilesSource::Directory(root.clone()))
        .execute(&catalog)
        .await
        .expect("first import");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");

    let error = AddFiles::new(table, AddFilesSource::Directory(root.clone()))
        .execute(&catalog)
        .await
        .expect_err("the duplicate import is refused");

    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    for partition in ["x", "y", "z"] {
        assert!(
            error
                .to_string()
                .contains(&format!("{root}/cat={partition}/part-00000.parquet")),
            "every duplicate is named: {error}"
        );
    }
}

#[tokio::test]
async fn a_source_directory_that_is_not_a_partition_directory_is_refused() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let schema = id_v_schema();
    let spec = identity_spec(&schema, &[]);
    let table = create_table_with_spec(&catalog, schema, spec).await;
    let root = source_root(&temp_dir, "plain-subdir");
    write_source_file(&table, &format!("{root}/sub/part-00000.parquet"), &[(
        "id",
        long_column(&[1]),
    )])
    .await;

    let error = AddFiles::new(table, AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect_err("a plain subdirectory is not a partition directory");

    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error
            .to_string()
            .contains("the directory 'sub' is not a 'name=value' partition directory"),
        "typed refusal: {error}"
    );
}

#[tokio::test]
async fn conflicting_source_directory_structures_are_refused() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let schema = id_v_cat_schema();
    let spec = identity_spec(&schema, &["cat"]);
    let table = create_table_with_spec(&catalog, schema, spec).await;
    let root = source_root(&temp_dir, "conflicting");
    write_source_file(&table, &format!("{root}/cat=x/part-00000.parquet"), &[(
        "id",
        long_column(&[1]),
    )])
    .await;
    write_source_file(&table, &format!("{root}/dept=hr/part-00000.parquet"), &[(
        "id",
        long_column(&[2]),
    )])
    .await;

    let error = AddFiles::new(table, AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect_err("two partition layouts under one root are refused");

    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error
            .to_string()
            .contains("Conflicting directory structures in the source to import"),
        "typed refusal: {error}"
    );
}

#[tokio::test]
async fn a_partition_value_that_does_not_parse_for_its_type_is_refused() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let schema = id_typed_part_schema(PrimitiveType::Long);
    let spec = identity_spec(&schema, &["part"]);
    let table = create_table_with_spec(&catalog, schema, spec).await;
    let root = source_root(&temp_dir, "bad-value");
    write_source_file(
        &table,
        &format!("{root}/part=notanumber/part-00000.parquet"),
        &[("id", long_column(&[1]))],
    )
    .await;

    let error = AddFiles::new(table, AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect_err("a partition value that does not parse is refused");

    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error
            .to_string()
            .contains("Cannot parse the partition value 'notanumber' of column part"),
        "Java Long.valueOf throws NumberFormatException: {error}"
    );
}

#[tokio::test]
async fn a_partition_type_java_cannot_parse_from_a_string_is_refused() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let schema = id_typed_part_schema(PrimitiveType::Timestamp);
    let spec = identity_spec(&schema, &["part"]);
    let table = create_table_with_spec(&catalog, schema, spec).await;
    let root = source_root(&temp_dir, "unsupported-type");
    write_source_file(
        &table,
        &format!("{root}/part=2020-01-01 00:00:00/part-00000.parquet"),
        &[("id", long_column(&[1]))],
    )
    .await;

    let error = AddFiles::new(table, AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect_err("a timestamp partition column has no hive-string parse");

    assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
    assert!(
        error
            .to_string()
            .contains("Unsupported type for fromPartitionString"),
        "Java Conversions.fromPartitionString default branch: {error}"
    );
}

#[tokio::test]
async fn a_missing_source_file_is_a_typed_error_naming_the_path() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let schema = id_v_schema();
    let spec = identity_spec(&schema, &[]);
    let table = create_table_with_spec(&catalog, schema, spec).await;
    let root = source_root(&temp_dir, "missing-dir");
    flat_source(&table, &root).await;

    let error = AddFiles::new(
        table,
        AddFilesSource::Directory(format!("{root}-does-not-exist")),
    )
    .execute(&catalog)
    .await
    .expect_err("a source directory that does not exist is refused");

    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error.to_string().contains(&format!(
            "Cannot find any file to import under {root}-does-not-exist"
        )),
        "the refusal names the source: {error}"
    );
}

#[tokio::test]
async fn a_non_parquet_file_refusal_names_the_file() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let schema = id_v_schema();
    let spec = identity_spec(&schema, &[]);
    let table = create_table_with_spec(&catalog, schema, spec).await;
    let root = source_root(&temp_dir, "garbage");
    let path = format!("{root}/part-00000.parquet");
    table
        .file_io()
        .new_output(&path)
        .expect("output")
        .write(bytes::Bytes::from_static(b"PAR0 not really parquet"))
        .await
        .expect("write garbage");

    let error = AddFiles::new(table, AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect_err("a file that is not parquet is refused");

    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error.to_string().contains(&path),
        "the refusal names the file: {error}"
    );
}
