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

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{Field, Schema as ArrowSchema};
use bytes::Bytes;
use futures::TryStreamExt;
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use crate::maintenance::add_files::{AddFiles, AddFilesEntry, AddFilesSource};
pub(super) use crate::maintenance::rewrite_data_files::tests::local_fs_catalog;
use crate::spec::{
    DEFAULT_SCHEMA_NAME_MAPPING, DataFile, FormatVersion, ManifestContentType, NestedField,
    PartitionSpec, PrimitiveType, Schema, TableProperties, Transform, Type,
};
use crate::table::Table;
use crate::{Catalog, NamespaceIdent, TableCreation, TableIdent};

pub(super) fn id_v_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::optional(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                2,
                "v",
                Type::Primitive(PrimitiveType::String),
            )),
        ])
        .build()
        .expect("build id/v schema")
}

pub(super) fn id_v_cat_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::optional(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                2,
                "v",
                Type::Primitive(PrimitiveType::String),
            )),
            Arc::new(NestedField::optional(
                3,
                "cat",
                Type::Primitive(PrimitiveType::String),
            )),
        ])
        .build()
        .expect("build id/v/cat schema")
}

fn id_other_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::optional(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                2,
                "other",
                Type::Primitive(PrimitiveType::String),
            )),
        ])
        .build()
        .expect("build id/other schema")
}

pub(super) async fn create_table(
    catalog: &impl Catalog,
    schema: Schema,
    partition_by: Option<&str>,
    format_version: FormatVersion,
) -> Table {
    create_table_with_properties(
        catalog,
        schema,
        partition_by,
        format_version,
        HashMap::new(),
    )
    .await
}

pub(super) async fn create_table_with_properties(
    catalog: &impl Catalog,
    schema: Schema,
    partition_by: Option<&str>,
    format_version: FormatVersion,
    properties: HashMap<String, String>,
) -> Table {
    let spec = match partition_by {
        Some(column) => PartitionSpec::builder(schema.clone())
            .with_spec_id(0)
            .add_partition_field(column, column, Transform::Identity)
            .expect("add partition field")
            .build()
            .expect("build spec"),
        None => PartitionSpec::builder(schema.clone())
            .with_spec_id(0)
            .build()
            .expect("build unpartitioned spec"),
    };
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
        .properties(properties)
        .format_version(format_version)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

fn id_less_parquet_bytes(columns: &[(&str, ArrayRef)]) -> Bytes {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
        .collect();
    let arrow_schema = Arc::new(ArrowSchema::new(fields));
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        columns.iter().map(|(_, array)| array.clone()).collect(),
    )
    .expect("source record batch");
    let mut buffer: Vec<u8> = Vec::new();
    let mut writer =
        ArrowWriter::try_new(&mut buffer, arrow_schema, None).expect("source parquet writer");
    writer.write(&batch).expect("write source batch");
    writer.close().expect("close source parquet writer");
    Bytes::from(buffer)
}

pub(super) async fn write_source_file(table: &Table, path: &str, columns: &[(&str, ArrayRef)]) {
    table
        .file_io()
        .new_output(path)
        .expect("source output file")
        .write(id_less_parquet_bytes(columns))
        .await
        .expect("write source file");
}

pub(super) fn long_column(values: &[i64]) -> ArrayRef {
    Arc::new(Int64Array::from(values.to_vec())) as ArrayRef
}

pub(super) fn string_column(values: &[&str]) -> ArrayRef {
    Arc::new(StringArray::from(values.to_vec())) as ArrayRef
}

pub(super) async fn flat_source(table: &Table, root: &str) {
    write_source_file(table, &format!("{root}/part-00000.parquet"), &[
        ("id", long_column(&[10, 11])),
        ("v", string_column(&["p", "q"])),
    ])
    .await;
}

pub(super) async fn hive_source(table: &Table, root: &str) {
    write_source_file(table, &format!("{root}/cat=x/part-00000.parquet"), &[
        ("id", long_column(&[1, 2])),
        ("v", string_column(&["a", "b"])),
    ])
    .await;
    write_source_file(table, &format!("{root}/cat=y/part-00000.parquet"), &[
        ("id", long_column(&[3])),
        ("v", string_column(&["c"])),
    ])
    .await;
    write_source_file(table, &format!("{root}/cat=z/part-00000.parquet"), &[
        ("id", long_column(&[4])),
        ("v", string_column(&["d"])),
    ])
    .await;
}

pub(super) fn source_root(temp_dir: &TempDir, name: &str) -> String {
    format!(
        "{}/source-{name}",
        temp_dir.path().to_str().expect("utf8 temp path")
    )
}

pub(super) async fn live_data_files(table: &Table) -> Vec<DataFile> {
    let metadata = table.metadata();
    let Some(snapshot) = metadata.current_snapshot() else {
        return Vec::new();
    };
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), metadata)
        .await
        .expect("load manifest list");
    let mut files = Vec::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("load manifest");
        for entry in manifest.entries() {
            if entry.is_alive() {
                files.push(entry.data_file().clone());
            }
        }
    }
    files.sort_by(|left, right| left.file_path().cmp(right.file_path()));
    files
}

pub(super) async fn scan_id_v(table: &Table) -> Vec<(Option<i64>, Option<String>)> {
    scan_two(table, "id", "v").await
}

async fn scan_two(table: &Table, first: &str, second: &str) -> Vec<(Option<i64>, Option<String>)> {
    let stream = table
        .scan()
        .select([first, second])
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to arrow");
    let batches: Vec<RecordBatch> = stream.try_collect().await.expect("collect batches");
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name(first)
            .expect("id column")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("id is int64")
            .clone();
        let values = batch
            .column_by_name(second)
            .expect("value column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("value is utf8")
            .clone();
        for index in 0..batch.num_rows() {
            rows.push((
                (!ids.is_null(index)).then(|| ids.value(index)),
                (!values.is_null(index)).then(|| values.value(index).to_string()),
            ));
        }
    }
    rows.sort();
    rows
}

#[tokio::test]
async fn unpartitioned_source_is_adopted_in_place_in_one_append_snapshot() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_schema(), None, FormatVersion::V2).await;
    let root = source_root(&temp_dir, "flat");
    flat_source(&table, &root).await;

    let result = AddFiles::new(table.clone(), AddFilesSource::Directory(root.clone()))
        .execute(&catalog)
        .await
        .expect("add_files on an unpartitioned source");

    assert_eq!(result.added_files_count, 1, "oracle cell UNPARTITIONED");
    assert_eq!(result.changed_partition_count, None);

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    let files = live_data_files(&table).await;
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].file_path(), format!("{root}/part-00000.parquet"));
    assert_eq!(files[0].record_count(), 2);
    assert_eq!(
        table
            .metadata()
            .current_snapshot()
            .expect("snapshot")
            .summary()
            .operation
            .as_str(),
        "append"
    );
    assert_eq!(scan_id_v(&table).await, vec![
        (Some(10), Some("p".to_string())),
        (Some(11), Some("q".to_string())),
    ]);
}

#[tokio::test]
async fn partitioned_source_adopts_one_file_per_hive_directory() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_cat_schema(), Some("cat"), FormatVersion::V2).await;
    let root = source_root(&temp_dir, "hive");
    hive_source(&table, &root).await;

    let result = AddFiles::new(table.clone(), AddFilesSource::Directory(root.clone()))
        .execute(&catalog)
        .await
        .expect("add_files on a hive-layout source");

    assert_eq!(result.added_files_count, 3, "oracle cell PARTITIONED");

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    let files = live_data_files(&table).await;
    assert_eq!(files.len(), 3);
    let by_partition: Vec<(String, u64)> = files
        .iter()
        .map(|file| (partition_string(file), file.record_count()))
        .collect();
    assert_eq!(by_partition, vec![
        ("x".to_string(), 2),
        ("y".to_string(), 1),
        ("z".to_string(), 1),
    ]);
    for file in &files {
        assert!(
            file.file_path().starts_with(&root),
            "adoption is in place: {}",
            file.file_path()
        );
    }
    assert_eq!(scan_two(&table, "id", "cat").await, vec![
        (Some(1), Some("x".to_string())),
        (Some(2), Some("x".to_string())),
        (Some(3), Some("y".to_string())),
        (Some(4), Some("z".to_string())),
    ]);
}

fn partition_string(file: &DataFile) -> String {
    file.partition()
        .iter()
        .map(|value| match value {
            Some(literal) => format!("{literal:?}"),
            None => "null".to_string(),
        })
        .collect::<Vec<String>>()
        .join(",")
        .replace("Primitive(String(\"", "")
        .replace("\"))", "")
}

#[tokio::test]
async fn partition_filter_adopts_only_the_named_partition() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_cat_schema(), Some("cat"), FormatVersion::V2).await;
    let root = source_root(&temp_dir, "filter");
    hive_source(&table, &root).await;

    let result = AddFiles::new(table.clone(), AddFilesSource::Directory(root.clone()))
        .partition_filter(HashMap::from([("cat".to_string(), "x".to_string())]))
        .execute(&catalog)
        .await
        .expect("add_files with a partition filter");

    assert_eq!(result.added_files_count, 1, "oracle cell PARTITION-FILTER");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    assert_eq!(scan_two(&table, "id", "cat").await, vec![
        (Some(1), Some("x".to_string())),
        (Some(2), Some("x".to_string())),
    ]);
}

#[tokio::test]
async fn partition_filter_on_an_unpartitioned_table_is_refused() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_schema(), None, FormatVersion::V2).await;
    let root = source_root(&temp_dir, "unpart-filter");
    flat_source(&table, &root).await;

    let error = AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .partition_filter(HashMap::from([("cat".to_string(), "x".to_string())]))
        .execute(&catalog)
        .await
        .expect_err("a partition filter on an unpartitioned table is refused");

    assert!(
        error.to_string().contains(&format!(
            "Cannot use partition filter with an unpartitioned table {}",
            table.identifier()
        )),
        "oracle cell PARTITION-FILTER-ON-UNPARTITIONED: {error}"
    );
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    assert!(
        table.metadata().current_snapshot().is_none(),
        "a refusal commits no snapshot"
    );
}

#[tokio::test]
async fn check_duplicate_files_refuses_the_second_import() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_cat_schema(), Some("cat"), FormatVersion::V2).await;
    let root = source_root(&temp_dir, "dup-true");
    hive_source(&table, &root).await;

    AddFiles::new(table.clone(), AddFilesSource::Directory(root.clone()))
        .execute(&catalog)
        .await
        .expect("first import");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");

    let error = AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect_err("the second import of the same files is refused");

    assert!(
        error.to_string().contains(
            "Cannot complete import because data files to be imported already exist within the target table:"
        ),
        "oracle cell CHECK-DUP-TRUE-TWICE: {error}"
    );
    assert!(
        error
            .to_string()
            .contains("you may set 'check_duplicate_files' to false to force the import."),
        "the Java message tail: {error}"
    );
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    assert_eq!(live_data_files(&table).await.len(), 3);
}

#[tokio::test]
async fn check_duplicate_files_false_adopts_the_same_files_twice() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_cat_schema(), Some("cat"), FormatVersion::V2).await;
    let root = source_root(&temp_dir, "dup-false");
    hive_source(&table, &root).await;

    for _ in 0..2 {
        let table = catalog
            .load_table(table.identifier())
            .await
            .expect("reload table");
        let result = AddFiles::new(table, AddFilesSource::Directory(root.clone()))
            .check_duplicate_files(false)
            .execute(&catalog)
            .await
            .expect("import with the duplicate check off");
        assert_eq!(result.added_files_count, 3);
    }

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    assert_eq!(
        live_data_files(&table).await.len(),
        6,
        "oracle cell CHECK-DUP-FALSE-TWICE"
    );
    assert_eq!(scan_two(&table, "id", "cat").await.len(), 8);
}

#[tokio::test]
async fn a_missing_source_is_refused() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_schema(), None, FormatVersion::V2).await;
    let root = source_root(&temp_dir, "nosuch");

    let error = AddFiles::new(
        table.clone(),
        AddFilesSource::Files(vec![AddFilesEntry::new(format!("{root}/nosuch.parquet"))]),
    )
    .execute(&catalog)
    .await
    .expect_err("a missing source file is refused");

    assert!(
        error.to_string().contains("nosuch.parquet"),
        "oracle cell MISSING-SOURCE names the source that does not exist: {error}"
    );
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    assert!(table.metadata().current_snapshot().is_none());
}

#[tokio::test]
async fn a_source_column_the_target_lacks_is_dropped_and_reads_back_null() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_other_schema(), None, FormatVersion::V2).await;
    let root = source_root(&temp_dir, "mismatch");
    flat_source(&table, &root).await;

    let result = AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect("add_files with a source column the target lacks");

    assert_eq!(result.added_files_count, 1, "oracle cell SCHEMA-MISMATCH");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    assert_eq!(
        scan_two(&table, "id", "other").await,
        vec![(Some(10), None), (Some(11), None)],
        "Java maps by name and silently drops the source column v"
    );
}

#[tokio::test]
async fn the_default_name_mapping_is_created_when_absent() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_schema(), None, FormatVersion::V2).await;
    assert!(
        !table
            .metadata()
            .properties()
            .contains_key(DEFAULT_SCHEMA_NAME_MAPPING),
        "a fresh table carries no name mapping"
    );
    let root = source_root(&temp_dir, "mapping");
    flat_source(&table, &root).await;

    AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect("add_files creates the name mapping");

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    let mapping = table
        .metadata()
        .properties()
        .get(DEFAULT_SCHEMA_NAME_MAPPING)
        .expect("Java ensureNameMappingPresent sets the property");
    assert_eq!(
        mapping, r#"[{"field-id":1,"names":["id"]},{"field-id":2,"names":["v"]}]"#,
        "MappingUtil.create over the table schema"
    );
}

#[tokio::test]
async fn parallelism_two_adopts_the_same_files() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_cat_schema(), Some("cat"), FormatVersion::V2).await;
    let root = source_root(&temp_dir, "parallel");
    hive_source(&table, &root).await;

    let result = AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .parallelism(2)
        .execute(&catalog)
        .await
        .expect("add_files with parallelism 2");

    assert_eq!(result.added_files_count, 3, "oracle cell PARALLELISM");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    assert_eq!(scan_two(&table, "id", "cat").await.len(), 4);
}

#[tokio::test]
async fn parallelism_zero_is_refused() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_schema(), None, FormatVersion::V2).await;
    let root = source_root(&temp_dir, "parallel-zero");
    flat_source(&table, &root).await;

    let error = AddFiles::new(table, AddFilesSource::Directory(root))
        .parallelism(0)
        .execute(&catalog)
        .await
        .expect_err("parallelism 0 is refused");

    assert!(
        error
            .to_string()
            .contains("Parallelism should be larger than 0"),
        "Java AddFilesProcedure.call precondition: {error}"
    );
}

#[tokio::test]
async fn a_v3_target_adopts_the_file() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_schema(), None, FormatVersion::V3).await;
    let root = source_root(&temp_dir, "v3");
    flat_source(&table, &root).await;

    let result = AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect("add_files onto a v3 table");

    assert_eq!(result.added_files_count, 1, "oracle cell V3-TARGET");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    assert_eq!(scan_id_v(&table).await, vec![
        (Some(10), Some("p".to_string())),
        (Some(11), Some("q".to_string())),
    ]);
}

#[tokio::test]
async fn an_adopted_file_carries_sort_order_id_zero_and_no_split_offsets() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_schema(), None, FormatVersion::V2).await;
    let root = source_root(&temp_dir, "stamps");
    flat_source(&table, &root).await;

    AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect("add_files");

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    let files = live_data_files(&table).await;
    assert_eq!(
        files[0].sort_order_id(),
        Some(0),
        "Java DataFiles.Builder defaults sortOrderId to SortOrder.unsorted().orderId()"
    );
    assert!(
        files[0]
            .split_offsets()
            .is_none_or(|offsets| offsets.is_empty()),
        "Java TableMigrationUtil.buildDataFile never calls withSplitOffsets"
    );
    let on_disk = table
        .file_io()
        .new_input(files[0].file_path())
        .expect("input file")
        .metadata()
        .await
        .expect("source file metadata")
        .size;
    assert_eq!(
        files[0].file_size_in_bytes(),
        on_disk,
        "Java takes withFileSizeInBytes from FileStatus.getLen(), not from the footer"
    );
    assert!(on_disk > 0);
}

#[tokio::test]
async fn the_hive_default_partition_directory_becomes_a_null_partition_value() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_cat_schema(), Some("cat"), FormatVersion::V2).await;
    let root = source_root(&temp_dir, "hive-null");
    write_source_file(
        &table,
        &format!("{root}/cat=__HIVE_DEFAULT_PARTITION__/part-00000.parquet"),
        &[("id", long_column(&[7])), ("v", string_column(&["n"]))],
    )
    .await;

    AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect("add_files over a hive default partition directory");

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    let files = live_data_files(&table).await;
    assert_eq!(files.len(), 1);
    let partition: Vec<Option<&crate::spec::Literal>> = files[0].partition().iter().collect();
    assert_eq!(partition.len(), 1);
    assert!(
        partition[0].is_none(),
        "Java Conversions.fromPartitionString maps __HIVE_DEFAULT_PARTITION__ to null"
    );
}

#[tokio::test]
async fn a_file_that_is_not_parquet_is_refused() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_schema(), None, FormatVersion::V2).await;
    let root = source_root(&temp_dir, "not-parquet");
    table
        .file_io()
        .new_output(format!("{root}/part-00000.parquet"))
        .expect("output")
        .write(Bytes::from_static(b"this is not a parquet file"))
        .await
        .expect("write a non-parquet file");

    let error = AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect_err("a file that is not parquet is refused");

    assert!(
        error
            .to_string()
            .contains("Cannot read the parquet footer of the file to import"),
        "typed refusal: {error}"
    );
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    assert!(table.metadata().current_snapshot().is_none());
}

#[tokio::test]
async fn an_explicit_file_list_carries_its_own_partition_values() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_cat_schema(), Some("cat"), FormatVersion::V2).await;
    let root = source_root(&temp_dir, "explicit");
    hive_source(&table, &root).await;

    let result = AddFiles::new(
        table.clone(),
        AddFilesSource::Files(vec![
            AddFilesEntry::new(format!("{root}/cat=y/part-00000.parquet"))
                .with_partition(vec![("cat".to_string(), "y".to_string())]),
        ]),
    )
    .execute(&catalog)
    .await
    .expect("add_files over an explicit file list");

    assert_eq!(result.added_files_count, 1);
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    assert_eq!(scan_two(&table, "id", "cat").await, vec![(
        Some(3),
        Some("y".to_string())
    )]);
}

#[tokio::test]
async fn a_hidden_directory_or_file_is_skipped() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_schema(), None, FormatVersion::V2).await;
    let root = source_root(&temp_dir, "hidden");
    flat_source(&table, &root).await;
    table
        .file_io()
        .new_output(format!("{root}/_SUCCESS"))
        .expect("output")
        .write(Bytes::from_static(b""))
        .await
        .expect("write _SUCCESS");
    table
        .file_io()
        .new_output(format!("{root}/_temporary/0/part-00001.parquet"))
        .expect("output")
        .write(Bytes::from_static(b"not parquet"))
        .await
        .expect("write a hidden directory file");

    let result = AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect("hidden paths are skipped");

    assert_eq!(
        result.added_files_count, 1,
        "Java HIDDEN_PATH_FILTER skips names starting with _ or ."
    );
}

#[tokio::test]
async fn a_source_whose_partition_columns_match_no_spec_is_refused() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_cat_schema(), Some("cat"), FormatVersion::V2).await;
    let root = source_root(&temp_dir, "wrong-columns");
    write_source_file(&table, &format!("{root}/dept=hr/part-00000.parquet"), &[
        ("id", long_column(&[1])),
        ("v", string_column(&["a"])),
    ])
    .await;

    let error = AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect_err("a source whose partition columns match no spec is refused");

    assert!(
        error
            .to_string()
            .contains("that matches the partition columns ([dept]) in input table"),
        "Java SparkTableUtil.findCompatibleSpec: {error}"
    );
}

async fn assert_add_files_summary(table: &Table, added_files: u64, added_records: u64) {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("add_files commits one snapshot");
    assert_eq!(snapshot.summary().operation.as_str(), "append");
    let files = live_data_files(table).await;
    let size: u64 = files.iter().map(|file| file.file_size_in_bytes()).sum();
    assert!(size > 0);
    let expected = HashMap::from([
        ("added-data-files".to_string(), added_files.to_string()),
        ("added-records".to_string(), added_records.to_string()),
        ("added-files-size".to_string(), size.to_string()),
        ("total-data-files".to_string(), added_files.to_string()),
        ("total-delete-files".to_string(), "0".to_string()),
        ("total-records".to_string(), added_records.to_string()),
        ("total-files-size".to_string(), size.to_string()),
        ("total-position-deletes".to_string(), "0".to_string()),
        ("total-equality-deletes".to_string(), "0".to_string()),
        ("manifests-created".to_string(), "1".to_string()),
        ("manifests-kept".to_string(), "0".to_string()),
        ("manifests-replaced".to_string(), "0".to_string()),
    ]);
    assert_eq!(
        snapshot.summary().additional_properties,
        expected,
        "Java importSparkPartitions commits appendManifest: no changed-partition-count"
    );
}

#[tokio::test]
async fn partitioned_import_summary_carries_no_changed_partition_count() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_cat_schema(), Some("cat"), FormatVersion::V2).await;
    let root = source_root(&temp_dir, "summary-partitioned");
    hive_source(&table, &root).await;

    let result = AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect("add_files on a hive-layout source");

    assert_eq!(result.added_files_count, 3);
    assert_eq!(result.changed_partition_count, None);
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    assert_add_files_summary(&table, 3, 4).await;
}

#[tokio::test]
async fn unpartitioned_import_summary_carries_no_changed_partition_count() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_schema(), None, FormatVersion::V2).await;
    let root = source_root(&temp_dir, "summary-unpartitioned");
    flat_source(&table, &root).await;

    let result = AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect("add_files on an unpartitioned source");

    assert_eq!(result.added_files_count, 1);
    assert_eq!(result.changed_partition_count, None);
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    assert_add_files_summary(&table, 1, 2).await;
}

#[tokio::test]
async fn import_summary_carries_no_partition_keys_under_a_high_summary_limit() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table_with_properties(
        &catalog,
        id_v_cat_schema(),
        Some("cat"),
        FormatVersion::V2,
        HashMap::from([(
            TableProperties::PROPERTY_WRITE_PARTITION_SUMMARY_LIMIT.to_string(),
            "100".to_string(),
        )]),
    )
    .await;
    let root = source_root(&temp_dir, "summary-limit");
    hive_source(&table, &root).await;

    AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect("add_files under a high partition summary limit");

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    assert_add_files_summary(&table, 3, 4).await;
}
