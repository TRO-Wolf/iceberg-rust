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

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use tempfile::TempDir;

use super::add_files::{
    AddFiles, AddFilesEntry, AddFilesResult, AddFilesSource, find_compatible_spec,
    validate_partition_filter,
};
use super::add_files_datafile::unescape_hive_path_name;
use super::add_files_tests::{
    create_table, flat_source, id_v_cat_schema, id_v_schema, live_data_files, local_fs_catalog,
    long_column, scan_id_v, source_root, string_column, write_source_file,
};
use crate::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, Datum, FormatVersion, Literal, NestedField,
    PartitionSpec, PartitionSpecRef, PrimitiveLiteral, PrimitiveType, Schema, Struct,
    TableMetadataBuilder, Transform, Type, UnboundPartitionSpec,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{Catalog, ErrorKind, Result};

#[test]
fn the_hive_path_unescape_is_spark_s_unescape_path_name() {
    for (raw, expected) in [
        ("a b", "a b"),
        ("a+b", "a+b"),
        ("a%20b", "a b"),
        ("a%2Fb", "a/b"),
        ("a%25b", "a%b"),
        ("caf%C3%A9", "caf\u{c3}\u{a9}"),
        ("caf\u{e9}", "caf\u{e9}"),
        ("a%zzb", "a%zzb"),
        ("a%2", "a%2"),
        ("a%", "a%"),
        ("%41", "A"),
        ("a%2Gb", "a%2Gb"),
        ("100%", "100%"),
        ("%%20", "% "),
        ("a%2520b", "a%20b"),
        ("%2f", "/"),
        ("%00", "\u{0}"),
    ] {
        assert_eq!(
            unescape_hive_path_name(raw),
            expected,
            "ExternalCatalogUtils.unescapePathName({raw:?})"
        );
    }
}

async fn adopt_escaped_dirs(name: &str, dirs: &[&str]) -> Vec<(String, Option<Literal>)> {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_cat_schema(), Some("cat"), FormatVersion::V2).await;
    let root = source_root(&temp_dir, name);
    for (index, dir) in dirs.iter().enumerate() {
        write_source_file(&table, &format!("{root}/{dir}/part-00000.parquet"), &[
            ("id", long_column(&[index as i64])),
            ("v", string_column(&["a"])),
        ])
        .await;
    }
    AddFiles::new(table.clone(), AddFilesSource::Directory(root.clone()))
        .execute(&catalog)
        .await
        .expect("adopt the escaped hive directories");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    live_data_files(&table)
        .await
        .into_iter()
        .map(|file| {
            let directory = file
                .file_path()
                .strip_prefix(&format!("{root}/"))
                .and_then(|rest| rest.split('/').next())
                .expect("the adopted path stays under the source root")
                .to_string();
            let value = file.partition().iter().next().flatten().cloned();
            (directory, value)
        })
        .collect()
}

#[tokio::test]
async fn a_percent_escaped_hive_directory_adopts_spark_s_unescaped_value() {
    let adopted = adopt_escaped_dirs("escaped", &[
        "cat=a%20b",
        "cat=caf%C3%A9",
        "cat=a%2Fb",
        "cat=a+b",
        "cat=a%25b",
        "cat=a%zzb",
        "cat=a%2",
    ])
    .await;
    let by_directory: HashMap<String, Option<Literal>> = adopted.into_iter().collect();
    for (directory, expected) in [
        ("cat=a%20b", "a b"),
        ("cat=caf%C3%A9", "caf\u{c3}\u{a9}"),
        ("cat=a%2Fb", "a/b"),
        ("cat=a+b", "a+b"),
        ("cat=a%25b", "a%b"),
        ("cat=a%zzb", "a%zzb"),
        ("cat=a%2", "a%2"),
    ] {
        assert_eq!(
            by_directory.get(directory),
            Some(&Some(Literal::string(expected))),
            "oracle cell G: {directory} adopts {expected:?}"
        );
    }
}

async fn adopt_with_filter(name: &str, dirs: &[&str], filter: &str) -> Vec<String> {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_cat_schema(), Some("cat"), FormatVersion::V2).await;
    let root = source_root(&temp_dir, name);
    for (index, dir) in dirs.iter().enumerate() {
        write_source_file(&table, &format!("{root}/{dir}/part-00000.parquet"), &[
            ("id", long_column(&[index as i64])),
            ("v", string_column(&["a"])),
        ])
        .await;
    }
    AddFiles::new(table.clone(), AddFilesSource::Directory(root.clone()))
        .partition_filter(HashMap::from([("cat".to_string(), filter.to_string())]))
        .execute(&catalog)
        .await
        .expect("adopt under the partition filter");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    live_data_files(&table)
        .await
        .into_iter()
        .map(|file| {
            file.file_path()
                .strip_prefix(&format!("{root}/"))
                .and_then(|rest| rest.split('/').next())
                .expect("the adopted path stays under the source root")
                .to_string()
        })
        .collect()
}

#[tokio::test]
async fn a_partition_filter_matches_the_unescaped_hive_value() {
    let dirs = ["cat=a%20b", "cat=a%2520b", "cat=a%2Fb"];
    assert_eq!(
        adopt_with_filter("filter-unescaped", &dirs, "a b").await,
        vec!["cat=a%20b".to_string()],
        "oracle cell B: map('cat','a b') selects the directory whose UNESCAPED value is 'a b'"
    );
    assert_eq!(
        adopt_with_filter("filter-raw", &dirs, "a%20b").await,
        vec!["cat=a%2520b".to_string()],
        "oracle cell B: map('cat','a%20b') selects cat=a%2520b, not the raw directory text"
    );
    assert_eq!(
        adopt_with_filter("filter-slash", &dirs, "a/b").await,
        vec!["cat=a%2Fb".to_string()],
        "oracle cell B: a value Spark had to escape is matched unescaped"
    );
}

fn optional_id_parquet_bytes(columns: &[(&str, Option<i32>, ArrayRef)]) -> Bytes {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, field_id, array)| {
            let field = Field::new(*name, array.data_type().clone(), true);
            match field_id {
                Some(field_id) => field.with_metadata(HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    field_id.to_string(),
                )])),
                None => field,
            }
        })
        .collect();
    let arrow_schema = Arc::new(ArrowSchema::new(fields));
    let batch = RecordBatch::try_new(
        Arc::clone(&arrow_schema),
        columns.iter().map(|(_, _, array)| array.clone()).collect(),
    )
    .expect("source record batch");
    let mut buffer: Vec<u8> = Vec::new();
    let mut writer =
        ArrowWriter::try_new(&mut buffer, arrow_schema, None).expect("source parquet writer");
    writer.write(&batch).expect("write source batch");
    writer.close().expect("close source parquet writer");
    Bytes::from(buffer)
}

async fn adopt_bytes(name: &str, bytes: Bytes) -> (TempDir, Table, Result<AddFilesResult>) {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_schema(), None, FormatVersion::V2).await;
    let root = source_root(&temp_dir, name);
    table
        .file_io()
        .new_output(format!("{root}/part-00000.parquet"))
        .expect("source output file")
        .write(bytes)
        .await
        .expect("write the source file");
    let outcome = AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .execute(&catalog)
        .await;
    let reloaded = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    (temp_dir, reloaded, outcome)
}

#[tokio::test]
async fn a_source_carrying_field_ids_on_only_some_columns_is_refused() {
    let (_temp_dir, table, outcome) = adopt_bytes(
        "mixed-ids",
        optional_id_parquet_bytes(&[
            ("v", None, string_column(&["p", "q"])),
            ("w", Some(2), string_column(&["z", "z"])),
        ]),
    )
    .await;
    let error = outcome.expect_err("a file whose columns carry ids unevenly is refused");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error
            .message()
            .contains("only some of its columns carry Iceberg field ids"),
        "{error}"
    );
    assert!(
        error.message().contains("These columns carry none: v"),
        "the refusal names the columns that carry no id: {error}"
    );
    assert!(
        table.metadata().current_snapshot().is_none(),
        "the refusal leaves no append snapshot behind"
    );
}

#[tokio::test]
async fn a_source_whose_field_ids_are_only_nested_is_refused() {
    let child = Arc::new(
        Field::new("inner", DataType::Int64, true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
    );
    let nested: ArrayRef = Arc::new(StructArray::from(vec![(
        Arc::clone(&child),
        Arc::new(Int64Array::from(vec![10i64, 11])) as ArrayRef,
    )]));
    let bytes = optional_id_parquet_bytes(&[("outer", None, nested)]);
    let (_temp_dir, _, outcome) = adopt_bytes("nested-ids", bytes).await;
    let error = outcome.expect_err("a file whose ids live only below the top level is refused");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error.message().contains("These columns carry none: outer"),
        "{error}"
    );
}

#[tokio::test]
async fn an_all_ids_source_adopts_and_scans_back_its_rows() {
    let (_temp_dir, table, outcome) = adopt_bytes(
        "all-ids",
        optional_id_parquet_bytes(&[
            ("x", Some(1), long_column(&[10, 11])),
            ("y", Some(2), string_column(&["p", "q"])),
        ]),
    )
    .await;
    outcome.expect("a file with a field id on every column adopts");
    assert_eq!(scan_id_v(&table).await, vec![
        (Some(10), Some("p".to_string())),
        (Some(11), Some("q".to_string())),
    ]);
    let files = live_data_files(&table).await;
    assert_eq!(files[0].lower_bounds().get(&1), Some(&Datum::long(10)));
    assert_eq!(files[0].lower_bounds().get(&2), Some(&Datum::string("p")));
}

#[tokio::test]
async fn an_id_less_source_adopts_and_scans_back_its_rows() {
    let (_temp_dir, table, outcome) = adopt_bytes(
        "no-ids",
        optional_id_parquet_bytes(&[
            ("id", None, long_column(&[10, 11])),
            ("v", None, string_column(&["p", "q"])),
        ]),
    )
    .await;
    outcome.expect("a file with no field id anywhere adopts");
    assert_eq!(scan_id_v(&table).await, vec![
        (Some(10), Some("p".to_string())),
        (Some(11), Some("q".to_string())),
    ]);
    let files = live_data_files(&table).await;
    assert_eq!(files[0].lower_bounds().get(&1), Some(&Datum::long(10)));
    assert_eq!(files[0].lower_bounds().get(&2), Some(&Datum::string("p")));
}

fn id_typed_schema(column: &str, column_type: PrimitiveType) -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::optional(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                2,
                column,
                Type::Primitive(column_type),
            )),
        ])
        .build()
        .expect("build a typed partition schema")
}

async fn adopt_typed_dirs(
    name: &str,
    column: &str,
    column_type: PrimitiveType,
    dirs: &[&str],
) -> (TempDir, Result<Vec<(String, Option<Literal>)>>) {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(
        &catalog,
        id_typed_schema(column, column_type),
        Some(column),
        FormatVersion::V2,
    )
    .await;
    let root = source_root(&temp_dir, name);
    for (index, dir) in dirs.iter().enumerate() {
        write_source_file(&table, &format!("{root}/{dir}/part-00000.parquet"), &[(
            "id",
            long_column(&[index as i64]),
        )])
        .await;
    }
    let outcome = AddFiles::new(table.clone(), AddFilesSource::Directory(root.clone()))
        .execute(&catalog)
        .await;
    let adopted = match outcome {
        Err(error) => return (temp_dir, Err(error)),
        Ok(_) => {
            let table = catalog
                .load_table(table.identifier())
                .await
                .expect("reload table");
            live_data_files(&table)
                .await
                .into_iter()
                .map(|file| {
                    let directory = file
                        .file_path()
                        .strip_prefix(&format!("{root}/"))
                        .and_then(|rest| rest.split('/').next())
                        .expect("the adopted path stays under the source root")
                        .to_string();
                    (directory, file.partition().iter().next().flatten().cloned())
                })
                .collect()
        }
    };
    (temp_dir, Ok(adopted))
}

#[tokio::test]
async fn a_boolean_hive_value_follows_java_s_boolean_value_of() {
    let (_temp_dir, adopted) =
        adopt_typed_dirs("boolean-values", "flag", PrimitiveType::Boolean, &[
            "flag=true",
            "flag=TRUE",
            "flag=True",
            "flag=tRuE",
            "flag=false",
            "flag=FALSE",
            "flag=yes",
            "flag=1",
            "flag= true",
        ])
        .await;
    let adopted: HashMap<String, Option<Literal>> = adopted
        .expect("Boolean.valueOf never refuses a string")
        .into_iter()
        .collect();
    for (directory, expected) in [
        ("flag=true", true),
        ("flag=TRUE", true),
        ("flag=True", true),
        ("flag=tRuE", true),
        ("flag=false", false),
        ("flag=FALSE", false),
        ("flag=yes", false),
        ("flag=1", false),
        ("flag= true", false),
    ] {
        assert_eq!(
            adopted.get(directory),
            Some(&Some(Literal::bool(expected))),
            "Conversions.fromPartitionString(BooleanType, ...) is Boolean.valueOf: {directory}"
        );
    }
}

#[tokio::test]
async fn a_float_hive_value_follows_java_s_float_value_of() {
    let (_temp_dir, adopted) = adopt_typed_dirs("float-values", "f", PrimitiveType::Float, &[
        "f=1.5", "f=1.5f", "f=1.5D", "f=+1.5", "f= 1.5", "f=1.", "f=.5", "f=1e5",
    ])
    .await;
    let adopted: HashMap<String, Option<Literal>> = adopted
        .expect("every form Java accepts")
        .into_iter()
        .collect();
    for (directory, expected) in [
        ("f=1.5", 1.5f32),
        ("f=1.5f", 1.5),
        ("f=1.5D", 1.5),
        ("f=+1.5", 1.5),
        ("f= 1.5", 1.5),
        ("f=1.", 1.0),
        ("f=.5", 0.5),
        ("f=1e5", 100000.0),
    ] {
        assert_eq!(
            adopted.get(directory),
            Some(&Some(Literal::float(expected))),
            "Float.valueOf trims, takes an f/F/d/D suffix, and parses the rest: {directory}"
        );
    }

    let (_temp_dir, adopted) = adopt_typed_dirs("float-nan", "f", PrimitiveType::Float, &[
        "f=NaN",
        "f=Infinity",
        "f=-Infinity",
    ])
    .await;
    let adopted: HashMap<String, Option<Literal>> = adopted
        .expect("Java's exact NaN/Infinity spellings")
        .into_iter()
        .collect();
    assert_eq!(
        adopted.get("f=Infinity"),
        Some(&Some(Literal::float(f32::INFINITY)))
    );
    assert_eq!(
        adopted.get("f=-Infinity"),
        Some(&Some(Literal::float(f32::NEG_INFINITY)))
    );
    assert!(
        matches!(
            adopted.get("f=NaN"),
            Some(Some(Literal::Primitive(PrimitiveLiteral::Float(value)))) if value.is_nan()
        ),
        "NaN is the only spelling Float.valueOf accepts"
    );

    for spelling in [
        "f=nan",
        "f=NAN",
        "f=inf",
        "f=infinity",
        "f=INFINITY",
        "f=0x1p3",
    ] {
        let (_temp_dir, adopted) =
            adopt_typed_dirs("float-refused", "f", PrimitiveType::Float, &[spelling]).await;
        let error = adopted.expect_err("a spelling Float.valueOf rejects");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains("Cannot parse the partition value"),
            "{spelling}: {error}"
        );
    }
}

#[tokio::test]
async fn a_path_already_referenced_by_a_delete_file_is_a_duplicate() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_schema(), None, FormatVersion::V2).await;
    let root = source_root(&temp_dir, "delete-duplicate");
    flat_source(&table, &root).await;
    AddFiles::new(table.clone(), AddFilesSource::Directory(root.clone()))
        .execute(&catalog)
        .await
        .expect("seed the target with one adopted file");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");

    let delete_path = format!("{root}/deletes-00000.parquet");
    write_source_file(&table, &delete_path, &[("id", long_column(&[0]))]).await;
    let mut builder = DataFileBuilder::default();
    builder
        .content(DataContentType::PositionDeletes)
        .file_path(delete_path.clone())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(64)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::empty());
    let delete_file = builder.build().expect("build a position delete entry");
    let transaction = Transaction::new(&table);
    let action = transaction.row_delta().add_deletes(vec![delete_file]);
    let transaction = action.apply(transaction).expect("apply the row delta");
    let table = transaction
        .commit(&catalog)
        .await
        .expect("commit the delete file");

    let error = AddFiles::new(
        table,
        AddFilesSource::Files(vec![AddFilesEntry::new(delete_path.clone())]),
    )
    .execute(&catalog)
    .await
    .expect_err("Java joins against ENTRIES, which spans the DELETE manifests too");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error
            .message()
            .contains("Cannot complete import because data files to be imported already exist"),
        "{error}"
    );
    assert!(error.message().contains(&delete_path), "{error}");
}

fn all_identity_spec(spec_id: i32, field_name: &str) -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .with_spec_id(spec_id)
        .add_partition_field(3, field_name, Transform::Identity)
        .expect("add an identity partition field")
        .build()
}

fn with_extra_specs(table: &Table, specs: Vec<UnboundPartitionSpec>) -> Table {
    let mut builder = TableMetadataBuilder::new_from_metadata(
        table.metadata().clone(),
        table.metadata_location().map(str::to_string),
    );
    for spec in specs {
        builder = builder
            .add_partition_spec(spec)
            .expect("add a partition spec");
    }
    let metadata = builder
        .build()
        .expect("rebuild the table metadata")
        .metadata;
    Table::builder()
        .metadata(metadata)
        .identifier(table.identifier().clone())
        .file_io(table.file_io().clone())
        .build()
        .expect("rebuild the table")
}

#[tokio::test]
async fn the_lowest_spec_id_wins_when_several_specs_match() {
    let (catalog, _temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_cat_schema(), Some("cat"), FormatVersion::V2).await;
    let equivalents: Vec<UnboundPartitionSpec> = ["cAt", "caT", "cAT", "Cat", "CAt", "CaT", "CAT"]
        .into_iter()
        .enumerate()
        .map(|(index, name)| all_identity_spec(7 + index as i32, name))
        .collect();
    let table = with_extra_specs(&table, equivalents);
    assert_eq!(
        table.metadata().partition_specs_iter().len(),
        8,
        "eight distinct specs whose field names all lowercase to 'cat'"
    );
    let chosen = find_compatible_spec(&["cat".to_string()], &table).expect("a matching spec");
    let lowest = table
        .metadata()
        .partition_specs_iter()
        .filter(|spec| {
            spec.fields()
                .iter()
                .all(|field| field.transform == Transform::Identity)
                && spec
                    .fields()
                    .iter()
                    .map(|field| field.name.to_lowercase())
                    .eq(["cat".to_string()])
        })
        .map(|spec| spec.spec_id())
        .min()
        .expect("at least one matching spec");
    assert_eq!(
        chosen.spec_id(),
        lowest,
        "Java walks table.specs() in metadata list order; the fork walks it by ascending spec id"
    );
}

#[tokio::test]
async fn a_table_whose_only_matching_spec_is_void_is_refused() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_cat_schema(), Some("cat"), FormatVersion::V2).await;
    let void = UnboundPartitionSpec::builder()
        .with_spec_id(7)
        .add_partition_field(3, "cat_void", Transform::Void)
        .expect("add a void partition field")
        .build();
    let table = with_extra_specs(&table, vec![void]);
    let root = source_root(&temp_dir, "void-spec");
    write_source_file(&table, &format!("{root}/part-00000.parquet"), &[
        ("id", long_column(&[1])),
        ("v", string_column(&["a"])),
    ])
    .await;

    let error = AddFiles::new(table, AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect_err("oracle cell F: a void spec is not an identity spec");
    assert!(
        error
            .message()
            .contains("that matches the partition columns ([]) in input table"),
        "Java's message for an unpartitioned source over a void-spec table: {error}"
    );
}

#[test]
fn a_partition_filter_over_an_all_void_spec_is_refused_as_unpartitioned() {
    let schema = id_v_cat_schema();
    let spec: PartitionSpecRef = Arc::new(
        PartitionSpec::builder(schema)
            .with_spec_id(7)
            .add_partition_field("cat", "cat_void", Transform::Void)
            .expect("add a void partition field")
            .build()
            .expect("build the void spec"),
    );
    assert!(
        spec.is_unpartitioned(),
        "Java PartitionSpec.isUnpartitioned(): every field is void"
    );
    let error = validate_partition_filter(
        &spec,
        &HashMap::from([("cat".to_string(), "x".to_string())]),
        "ns.t",
    )
    .expect_err("oracle cell E: a void spec takes the unpartitioned refusal");
    assert_eq!(
        error.message(),
        "Cannot use partition filter with an unpartitioned table ns.t"
    );
    validate_partition_filter(&spec, &HashMap::new(), "ns.t")
        .expect("an empty filter over a void spec is accepted");
}
