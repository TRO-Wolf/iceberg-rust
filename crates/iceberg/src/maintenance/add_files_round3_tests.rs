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

use super::add_files::{AddFiles, AddFilesEntry, AddFilesSource, find_compatible_spec};
use super::add_files_datafile::{AdoptionContext, partition_struct};
use super::add_files_field_id_tests::field_id_parquet_bytes;
use super::add_files_tests::{
    create_table, flat_source, id_v_schema, live_data_files, local_fs_catalog, long_column,
    scan_id_v, source_root, string_column, write_source_file,
};
use crate::scan::context::parse_name_mapping;
use crate::spec::{
    Datum, FormatVersion, Literal, MetricsConfig, NestedField, PartitionSpec, PrimitiveType, Schema,
    Transform, Type,
};
use crate::table::Table;
use crate::{Catalog, ErrorKind, NamespaceIdent, TableCreation};

#[tokio::test]
async fn a_source_sharing_only_its_basename_with_a_live_file_is_not_a_duplicate() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_schema(), None, FormatVersion::V2).await;
    let first_root = source_root(&temp_dir, "basename-first");
    flat_source(&table, &first_root).await;
    AddFiles::new(table.clone(), AddFilesSource::Directory(first_root.clone()))
        .execute(&catalog)
        .await
        .expect("adopt the first directory");

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    let second_root = source_root(&temp_dir, "basename-second");
    flat_source(&table, &second_root).await;

    let result = AddFiles::new(table.clone(), AddFilesSource::Directory(second_root.clone()))
        .execute(&catalog)
        .await
        .expect("Java joins on data_file.file_path, which is the WHOLE path");

    assert_eq!(result.added_files_count, 1);
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    let paths: Vec<String> = live_data_files(&table)
        .await
        .iter()
        .map(|file| file.file_path().to_string())
        .collect();
    assert_eq!(paths, vec![
        format!("{first_root}/part-00000.parquet"),
        format!("{second_root}/part-00000.parquet"),
    ]);
    let basenames: Vec<&str> = paths
        .iter()
        .map(|path| path.rsplit('/').next().expect("a basename"))
        .collect();
    assert_eq!(basenames, vec!["part-00000.parquet", "part-00000.parquet"]);
}

async fn cat_dept_table(catalog: &impl Catalog) -> Table {
    let schema = Schema::builder()
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
        .expect("build id/cat/dept schema");
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("cat", "cat", Transform::Identity)
        .expect("add cat")
        .add_partition_field("dept", "dept", Transform::Identity)
        .expect("add dept")
        .build()
        .expect("build the (cat, dept) spec");
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("t".to_string())
        .schema(schema)
        .partition_spec(spec)
        .format_version(FormatVersion::V2)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

#[tokio::test]
async fn a_partition_value_map_out_of_spec_order_still_builds_the_spec_s_tuple() {
    let (catalog, _temp_dir) = local_fs_catalog().await;
    let table = cat_dept_table(&catalog).await;
    let metadata = table.metadata();
    let spec = find_compatible_spec(&["cat".to_string(), "dept".to_string()], &table)
        .expect("the (cat, dept) spec");
    let context = AdoptionContext {
        schema: metadata.current_schema().clone(),
        metrics_config: MetricsConfig::for_table(metadata).expect("metrics config"),
        name_mapping: parse_name_mapping(metadata).expect("name mapping"),
        partition_type: spec
            .partition_type(metadata.current_schema())
            .expect("partition type"),
        spec,
    };

    let tuple = partition_struct(&context, &[
        ("dept".to_string(), "hr".to_string()),
        ("cat".to_string(), "x".to_string()),
    ])
    .expect("build the partition tuple");

    let values: Vec<Option<&Literal>> = tuple.iter().collect();
    assert_eq!(values, vec![
        Some(&Literal::string("x")),
        Some(&Literal::string("hr")),
    ]);
}

#[tokio::test]
async fn a_file_list_whose_values_are_out_of_spec_order_is_refused_by_the_spec_match() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = cat_dept_table(&catalog).await;
    let root = source_root(&temp_dir, "reordered-values");
    let path = format!("{root}/part-00000.parquet");
    write_source_file(&table, &path, &[("id", long_column(&[1]))]).await;

    let error = AddFiles::new(
        table.clone(),
        AddFilesSource::Files(vec![AddFilesEntry::new(path).with_partition(vec![
            ("dept".to_string(), "hr".to_string()),
            ("cat".to_string(), "x".to_string()),
        ])]),
    )
    .execute(&catalog)
    .await
    .expect_err("an entry's value order IS the source's declared partition column order");

    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error.message().contains(
            "that matches the partition columns ([dept, cat]) in input table"
        ),
        "{error}"
    );
}

#[tokio::test]
async fn embedded_field_ids_bind_by_id_even_when_they_run_against_position() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_schema(), None, FormatVersion::V2).await;
    let root = source_root(&temp_dir, "ids-against-position");
    table
        .file_io()
        .new_output(format!("{root}/part-00000.parquet"))
        .expect("source output file")
        .write(field_id_parquet_bytes(&[
            ("x", 2, string_column(&["p", "q"])),
            ("y", 1, long_column(&[10, 11])),
        ]))
        .await
        .expect("write a source whose ids run against its column order");

    AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect("a source whose embedded ids are not its positional ids");

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    let files = live_data_files(&table).await;
    assert_eq!(
        files[0].lower_bounds().get(&1),
        Some(&Datum::long(10)),
        "the SECOND column carries id 1, so field 1 takes its bounds"
    );
    assert_eq!(
        files[0].lower_bounds().get(&2),
        Some(&Datum::string("p")),
        "the FIRST column carries id 2"
    );
    assert_eq!(scan_id_v(&table).await, vec![
        (Some(10), Some("p".to_string())),
        (Some(11), Some("q".to_string())),
    ]);
}
