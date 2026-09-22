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
use std::fs::File;
use std::sync::Arc;

use datafusion::arrow::array::{Int32Array, StringArray};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::DataFusionError;
use datafusion::execution::context::SessionContext;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH;
use iceberg::spec::{
    DataContentType, DataFile, Datum, FormatVersion, NestedField, PrimitiveType, Schema, Transform,
    Type, UnboundPartitionSpec,
};
use iceberg::{
    Catalog, CatalogBuilder, Error, ErrorKind, NamespaceIdent, TableCreation, TableIdent,
};
use iceberg_datafusion::IcebergCatalogProvider;
use parquet::basic::Encoding;
use parquet::file::metadata::ColumnChunkMetaData;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::statistics::Statistics;
use tempfile::TempDir;

const DELETE_MODE_MOR: (&str, &str) = ("write.delete.mode", "merge-on-read");

fn mor_properties(extra: &[(&str, &str)]) -> HashMap<String, String> {
    let mut properties =
        HashMap::from([(DELETE_MODE_MOR.0.to_string(), DELETE_MODE_MOR.1.to_string())]);
    for (key, value) in extra {
        properties.insert((*key).to_string(), (*value).to_string());
    }
    properties
}

async fn create_fixture(
    namespace: &str,
    table_name: &str,
    properties: HashMap<String, String>,
    format_version: FormatVersion,
    partitioned: bool,
    deep_location: bool,
) -> (SessionContext, Arc<MemoryCatalog>, TableIdent, TempDir) {
    let warehouse = TempDir::new().expect("create warehouse");
    let warehouse_path = warehouse
        .path()
        .to_str()
        .expect("warehouse path is UTF-8")
        .to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
        )
        .await
        .expect("build memory catalog");
    let namespace = NamespaceIdent::new(namespace.to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "grp", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build schema");
    let location = if deep_location {
        let deep = format!("{warehouse_path}/{}/{table_name}", "l".repeat(110));
        std::fs::create_dir_all(&deep).expect("create deep table location");
        deep
    } else {
        format!("{warehouse_path}/{table_name}")
    };
    let table_ident = TableIdent::new(namespace.clone(), table_name.to_string());
    if partitioned {
        let partition_spec = UnboundPartitionSpec::builder()
            .with_spec_id(0)
            .add_partition_field(2, "grp", Transform::Identity)
            .expect("identity(grp)")
            .build();
        catalog
            .create_table(
                &namespace,
                TableCreation::builder()
                    .name(table_name.to_string())
                    .location(location)
                    .schema(schema)
                    .properties(properties)
                    .format_version(format_version)
                    .partition_spec(partition_spec)
                    .build(),
            )
            .await
            .expect("create table");
    } else {
        catalog
            .create_table(
                &namespace,
                TableCreation::builder()
                    .name(table_name.to_string())
                    .location(location)
                    .schema(schema)
                    .properties(properties)
                    .format_version(format_version)
                    .build(),
            )
            .await
            .expect("create table");
    }
    let catalog = Arc::new(catalog);
    let provider = Arc::new(
        IcebergCatalogProvider::try_new(catalog.clone())
            .await
            .expect("catalog provider"),
    );
    let context = SessionContext::new();
    context.register_catalog("catalog", provider);
    (context, catalog, table_ident, warehouse)
}

fn table_sql(table_ident: &TableIdent) -> String {
    format!(
        "catalog.{}.{}",
        table_ident.namespace()[0],
        table_ident.name()
    )
}

async fn run_sql(context: &SessionContext, sql: &str) {
    context
        .sql(sql)
        .await
        .unwrap_or_else(|error| panic!("plan `{sql}`: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("execute `{sql}`: {error}"));
}

async fn query_batches(context: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    context
        .sql(sql)
        .await
        .unwrap_or_else(|error| panic!("plan `{sql}`: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("execute `{sql}`: {error}"))
}

async fn insert_values(context: &SessionContext, table_ident: &TableIdent, values: &str) {
    run_sql(
        context,
        &format!("INSERT INTO {} VALUES {values}", table_sql(table_ident)),
    )
    .await;
}

async fn delete_where(context: &SessionContext, table_ident: &TableIdent, predicate: &str) {
    run_sql(
        context,
        &format!("DELETE FROM {} WHERE {predicate}", table_sql(table_ident)),
    )
    .await;
}

async fn files_content_format(
    context: &SessionContext,
    table_ident: &TableIdent,
) -> Vec<(i32, String, String)> {
    let sql = format!(
        "SELECT content, file_format, file_path FROM {}$files",
        table_sql(table_ident)
    );
    let batches = query_batches(context, &sql).await;
    let mut rows = Vec::new();
    for batch in batches {
        let content = batch
            .column_by_name("content")
            .expect("content column")
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("content is Int32");
        let format = batch
            .column_by_name("file_format")
            .expect("file_format column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("file_format is Utf8");
        let path = batch
            .column_by_name("file_path")
            .expect("file_path column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("file_path is Utf8");
        for row in 0..batch.num_rows() {
            rows.push((
                content.value(row),
                format.value(row).to_string(),
                path.value(row).to_string(),
            ));
        }
    }
    rows.sort();
    rows
}

async fn surviving_ids(context: &SessionContext, table_ident: &TableIdent) -> Vec<i32> {
    let sql = format!("SELECT id FROM {} ORDER BY id", table_sql(table_ident));
    let batches = query_batches(context, &sql).await;
    let mut ids = Vec::new();
    for batch in batches {
        let column = batch
            .column_by_name("id")
            .expect("id column")
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("id is Int32");
        for row in 0..column.len() {
            ids.push(column.value(row));
        }
    }
    ids.sort_unstable();
    ids
}

fn local_fs_path(file_path: &str) -> &str {
    file_path.strip_prefix("file://").unwrap_or(file_path)
}

fn chunk_uses_dictionary(column: &ColumnChunkMetaData) -> bool {
    column.encodings().any(|encoding| {
        matches!(
            encoding,
            Encoding::PLAIN_DICTIONARY | Encoding::RLE_DICTIONARY
        )
    })
}

async fn committed_position_deletes(
    catalog: &Arc<MemoryCatalog>,
    table_ident: &TableIdent,
) -> Vec<DataFile> {
    let table = catalog
        .load_table(table_ident)
        .await
        .expect("load the table");
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load the manifest list");
    let mut deletes = Vec::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("load the manifest");
        for entry in manifest.entries() {
            if entry.is_alive()
                && entry.data_file().content_type() == DataContentType::PositionDeletes
            {
                deletes.push(entry.data_file().clone());
            }
        }
    }
    deletes.sort_by(|left, right| left.file_path().cmp(right.file_path()));
    deletes
}

fn read_uvarint(rest: &[u8]) -> (u64, &[u8]) {
    let mut value = 0u64;
    let mut shift = 0u32;
    let mut consumed = 0usize;
    loop {
        assert!(
            shift < 70,
            "the postscript holds a malformed varint past ten bytes"
        );
        let byte = *rest
            .get(consumed)
            .expect("the postscript ends inside a varint");
        consumed += 1;
        value |= u64::from(byte & 0x7f) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            return (value, &rest[consumed..]);
        }
    }
}

fn orc_postscript_compression_kind(bytes: &[u8]) -> u64 {
    assert!(
        bytes.len() > 8,
        "an orc file must hold a postscript, got {} bytes",
        bytes.len()
    );
    assert_eq!(
        &bytes[bytes.len() - 4..bytes.len() - 1],
        b"ORC",
        "the file must close with the ORC magic"
    );
    let ps_len = usize::from(bytes[bytes.len() - 1]);
    assert!(
        ps_len + 1 < bytes.len(),
        "the postscript length must fit the file"
    );
    let mut rest = &bytes[bytes.len() - 1 - ps_len..bytes.len() - 1];
    loop {
        assert!(!rest.is_empty(), "the postscript must carry field 2");
        let (tag, tail) = read_uvarint(rest);
        rest = tail;
        match tag & 7 {
            0 => {
                let (value, tail) = read_uvarint(rest);
                rest = tail;
                if tag >> 3 == 2 {
                    return value;
                }
            }
            2 => {
                let (length, tail) = read_uvarint(rest);
                let length =
                    usize::try_from(length).expect("the postscript holds a length past usize");
                assert!(
                    tail.len() >= length,
                    "the postscript ends inside a length-prefixed field"
                );
                rest = &tail[length..];
            }
            5 => {
                assert!(rest.len() >= 4, "the postscript ends inside a fixed32");
                rest = &rest[4..];
            }
            wire => panic!("the postscript carries unexpected wire type {wire}"),
        }
    }
}

#[tokio::test]
async fn test_mor_delete_on_orc_writes_orc_delete_file() {
    let (context, _catalog, table_ident, _warehouse) = create_fixture(
        "mor_delete_format_orc",
        "target",
        mor_properties(&[("write.format.default", "orc")]),
        FormatVersion::V2,
        false,
        false,
    )
    .await;
    insert_values(&context, &table_ident, "(1, 'a'), (2, 'b'), (3, 'c')").await;
    delete_where(&context, &table_ident, "id = 2").await;

    let rows = files_content_format(&context, &table_ident).await;
    let content_format: Vec<(i32, String)> = rows
        .iter()
        .map(|(content, format, _)| (*content, format.clone()))
        .collect();
    assert_eq!(
        content_format,
        vec![(0, "ORC".to_string()), (1, "ORC".to_string())],
        "an ORC table takes ORC data and ORC position deletes, got {rows:?}"
    );
    let delete_path = &rows[1].2;
    assert!(
        delete_path.ends_with(".orc"),
        "delete file {delete_path} keeps the ORC suffix"
    );
    assert!(
        !delete_path.ends_with(".parquet"),
        "delete file {delete_path} must not keep a parquet suffix"
    );
    let bytes = std::fs::read(local_fs_path(delete_path)).expect("read delete file bytes");
    assert!(
        bytes.len() > 4 && &bytes[bytes.len() - 4..bytes.len() - 1] == b"ORC",
        "delete file {delete_path} carries the ORC tail magic"
    );
}

#[tokio::test]
async fn test_mor_delete_on_avro_writes_avro_delete_file() {
    let (context, _catalog, table_ident, _warehouse) = create_fixture(
        "mor_delete_format_avro",
        "target",
        mor_properties(&[("write.format.default", "avro")]),
        FormatVersion::V2,
        false,
        false,
    )
    .await;
    insert_values(&context, &table_ident, "(1, 'a'), (2, 'b'), (3, 'c')").await;
    delete_where(&context, &table_ident, "id = 2").await;

    let rows = files_content_format(&context, &table_ident).await;
    let content_format: Vec<(i32, String)> = rows
        .iter()
        .map(|(content, format, _)| (*content, format.clone()))
        .collect();
    assert_eq!(
        content_format,
        vec![(0, "AVRO".to_string()), (1, "AVRO".to_string())],
        "an Avro table takes Avro data and Avro position deletes, got {rows:?}"
    );
    let delete_path = &rows[1].2;
    assert!(
        delete_path.ends_with(".avro"),
        "delete file {delete_path} keeps the Avro suffix"
    );
    assert!(
        !delete_path.ends_with(".parquet"),
        "delete file {delete_path} must not keep a parquet suffix"
    );
    let bytes = std::fs::read(local_fs_path(delete_path)).expect("read delete file bytes");
    assert!(
        bytes.starts_with(b"Obj\x01"),
        "delete file {delete_path} carries the Avro OCF header"
    );
}

#[tokio::test]
async fn test_delete_format_default_overrides() {
    let (context, _catalog, table_ident, _warehouse) = create_fixture(
        "mor_delete_format_override",
        "target",
        mor_properties(&[
            ("write.format.default", "orc"),
            ("write.delete.format.default", "parquet"),
        ]),
        FormatVersion::V2,
        false,
        false,
    )
    .await;
    insert_values(&context, &table_ident, "(1, 'a'), (2, 'b'), (3, 'c')").await;
    delete_where(&context, &table_ident, "id = 2").await;

    let rows = files_content_format(&context, &table_ident).await;
    let content_format: Vec<(i32, String)> = rows
        .iter()
        .map(|(content, format, _)| (*content, format.clone()))
        .collect();
    assert_eq!(
        content_format,
        vec![(0, "ORC".to_string()), (1, "PARQUET".to_string())],
        "the delete-format property overrides the ORC data format, got {rows:?}"
    );
    let delete_path = &rows[1].2;
    assert!(
        delete_path.ends_with(".parquet"),
        "delete file {delete_path} keeps the parquet suffix"
    );
    let bytes = std::fs::read(local_fs_path(delete_path)).expect("read delete file bytes");
    assert!(
        bytes.starts_with(b"PAR1"),
        "delete file {delete_path} carries the parquet magic"
    );
    assert_eq!(
        surviving_ids(&context, &table_ident).await,
        vec![1, 3],
        "the parquet delete applies and no row resurrects"
    );
}

#[tokio::test]
async fn test_v3_delete_side_is_puffin_for_every_data_format() {
    for data_format in ["parquet", "orc", "avro"] {
        let (context, _catalog, table_ident, _warehouse) = create_fixture(
            &format!("mor_delete_format_v3_{data_format}"),
            "target",
            mor_properties(&[("write.format.default", data_format)]),
            FormatVersion::V3,
            false,
            false,
        )
        .await;
        insert_values(&context, &table_ident, "(1, 'a'), (2, 'b'), (3, 'c')").await;
        delete_where(&context, &table_ident, "id = 2").await;

        let rows = files_content_format(&context, &table_ident).await;
        let content_format: Vec<(i32, String)> = rows
            .iter()
            .map(|(content, format, _)| (*content, format.clone()))
            .collect();
        assert_eq!(
            content_format,
            vec![(0, data_format.to_uppercase()), (1, "PUFFIN".to_string()),],
            "a v3 {data_format} table takes a Puffin delete side, got {rows:?}"
        );
        assert_eq!(
            surviving_ids(&context, &table_ident).await,
            vec![1, 3],
            "the v3 {data_format} delete applies and no row resurrects"
        );
    }
}

async fn delete_refusal(delete_format: &str) -> DataFusionError {
    let (context, _catalog, table_ident, _warehouse) = create_fixture(
        &format!("mor_delete_format_refuse_{delete_format}"),
        "target",
        mor_properties(&[("write.delete.format.default", delete_format)]),
        FormatVersion::V2,
        false,
        false,
    )
    .await;
    insert_values(&context, &table_ident, "(1, 'a'), (2, 'b'), (3, 'c')").await;
    let sql = format!("DELETE FROM {} WHERE id = 2", table_sql(&table_ident));
    let Err(error) = context
        .sql(&sql)
        .await
        .expect("plan delete")
        .collect()
        .await
    else {
        panic!("delete with write.delete.format.default={delete_format} must refuse");
    };
    error
}

fn refused_iceberg_message(error: &DataFusionError, expected: &str) {
    let DataFusionError::External(inner) = error else {
        panic!("a garbage delete format must surface the iceberg refusal, got {error}");
    };
    let iceberg_error = inner
        .downcast_ref::<Error>()
        .expect("external wraps iceberg Error");
    assert_eq!(iceberg_error.kind(), ErrorKind::DataInvalid);
    assert_eq!(iceberg_error.message(), expected);
}

#[tokio::test]
async fn test_mor_delete_refuses_garbage_delete_format() {
    let error = delete_refusal("csv").await;
    refused_iceberg_message(&error, "Unsupported data file format: csv");
}

#[tokio::test]
async fn test_mor_delete_refuses_puffin_delete_format() {
    let error = delete_refusal("puffin").await;
    refused_iceberg_message(
        &error,
        "Cannot build a data-file writer for format puffin: a sidecar is never a data file",
    );
}

#[tokio::test]
async fn test_mor_delete_returns_every_partition_delete_file() {
    let (context, _catalog, table_ident, _warehouse) = create_fixture(
        "mor_delete_format_every_file",
        "target",
        mor_properties(&[]),
        FormatVersion::V2,
        true,
        false,
    )
    .await;
    insert_values(
        &context,
        &table_ident,
        "(1, 'a'), (2, 'a'), (3, 'b'), (4, 'b')",
    )
    .await;

    let before = files_content_format(&context, &table_ident).await;
    assert_eq!(
        before
            .iter()
            .filter(|(content, _, _)| *content == 0)
            .count(),
        2,
        "the fixture must hold one data file per partition, got {before:?}"
    );

    delete_where(&context, &table_ident, "id IN (2, 3)").await;

    let rows = files_content_format(&context, &table_ident).await;
    let deletes: Vec<&(i32, String, String)> = rows
        .iter()
        .filter(|(content, _, _)| *content == 1)
        .collect();
    assert_eq!(
        deletes.len(),
        2,
        "a delete spanning two partitions writes two delete files, got {rows:?}"
    );
    for (_, format, path) in &deletes {
        assert_eq!(
            format, "PARQUET",
            "delete file {path} stays parquet on a parquet table"
        );
    }
    assert_eq!(
        surviving_ids(&context, &table_ident).await,
        vec![1, 4],
        "both deletes apply and no row resurrects"
    );
}

#[tokio::test]
async fn test_mor_delete_parquet_keeps_full_exact_path_bounds() {
    let (context, _catalog, table_ident, _warehouse) = create_fixture(
        "mor_delete_format_bounds",
        "target",
        mor_properties(&[]),
        FormatVersion::V2,
        false,
        true,
    )
    .await;
    insert_values(&context, &table_ident, "(1, 'a'), (2, 'b'), (3, 'c')").await;
    delete_where(&context, &table_ident, "id = 2").await;

    let rows = files_content_format(&context, &table_ident).await;
    assert_eq!(
        rows.len(),
        2,
        "one data file plus one delete file, got {rows:?}"
    );
    let data_path = rows[0].2.clone();
    let delete_path = rows[1].2.clone();
    assert!(
        data_path.len() > 64,
        "the data path must exceed the 64-byte truncation window, got {data_path}"
    );
    assert!(
        delete_path.ends_with(".parquet"),
        "delete file {delete_path} keeps the parquet suffix"
    );

    let local = local_fs_path(&delete_path);
    let file = File::open(local).expect("open delete file");
    let reader = SerializedFileReader::new(file).expect("read delete file footer");
    let metadata = reader.metadata();
    assert_eq!(
        metadata.num_row_groups(),
        1,
        "one matched row writes one row group"
    );
    let row_group = metadata.row_group(0);
    assert_eq!(row_group.num_rows(), 1, "one matched row writes one row");
    assert_eq!(
        row_group.num_columns(),
        2,
        "a position delete holds file_path and pos"
    );
    for column in row_group.columns() {
        let name = column.column_descr().name();
        let dictionary = chunk_uses_dictionary(column);
        if name == "file_path" {
            assert!(dictionary, "the file_path chunk keeps dictionary encoding");
            let Some(Statistics::ByteArray(stats)) = column.statistics() else {
                panic!("the file_path chunk must carry string statistics");
            };
            assert!(
                stats.min_is_exact() && stats.max_is_exact(),
                "the file_path bounds stay exact"
            );
            assert_eq!(
                stats.min_opt().map(|bound| bound.data().to_vec()),
                Some(data_path.as_bytes().to_vec()),
                "the file_path lower bound is the full data path"
            );
            assert_eq!(
                stats.max_opt().map(|bound| bound.data().to_vec()),
                Some(data_path.as_bytes().to_vec()),
                "the file_path upper bound is the full data path"
            );
        } else if name == "pos" {
            assert!(!dictionary, "the pos chunk leaves dictionary encoding off");
            let Some(Statistics::Int64(stats)) = column.statistics() else {
                panic!("the pos chunk must carry long statistics");
            };
            assert_eq!(
                (stats.min_opt(), stats.max_opt()),
                (Some(&1), Some(&1)),
                "the single deleted row sits at pos 1"
            );
        } else {
            panic!("unexpected position-delete column {name}");
        }
    }

    let key_values = metadata
        .file_metadata()
        .key_value_metadata()
        .cloned()
        .unwrap_or_default();
    assert!(
        key_values
            .iter()
            .any(|entry| entry.key == "delete-type" && entry.value.as_deref() == Some("position")),
        "the delete file stamps delete-type=position, got {key_values:?}"
    );
    assert_eq!(
        surviving_ids(&context, &table_ident).await,
        vec![1, 3],
        "the delete applies and no row resurrects"
    );
}

#[tokio::test]
async fn test_mor_delete_committed_bounds_hold_the_full_data_path() {
    let (context, catalog, table_ident, _warehouse) = create_fixture(
        "mor_delete_format_committed_bounds",
        "target",
        mor_properties(&[]),
        FormatVersion::V2,
        false,
        true,
    )
    .await;
    insert_values(&context, &table_ident, "(1, 'a'), (2, 'b'), (3, 'c')").await;
    delete_where(&context, &table_ident, "id = 2").await;

    let rows = files_content_format(&context, &table_ident).await;
    assert_eq!(
        rows.len(),
        2,
        "one data file plus one delete file, got {rows:?}"
    );
    let data_path = rows[0].2.clone();
    assert!(
        data_path.len() > 16,
        "the data path must exceed the 16-byte truncation window, got {data_path}"
    );

    let deletes = committed_position_deletes(&catalog, &table_ident).await;
    assert_eq!(
        deletes.len(),
        1,
        "one committed position delete, got {deletes:?}"
    );
    let expected = Datum::string(data_path.clone());
    assert_eq!(
        deletes[0]
            .lower_bounds()
            .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH),
        Some(&expected),
        "the committed lower bound is the full data path"
    );
    assert_eq!(
        deletes[0]
            .upper_bounds()
            .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH),
        Some(&expected),
        "the committed upper bound is the full data path"
    );
    assert_eq!(
        surviving_ids(&context, &table_ident).await,
        vec![1, 3],
        "the delete applies and no row resurrects"
    );
}

#[tokio::test]
async fn test_mor_delete_on_orc_keeps_full_committed_path_bounds() {
    let (context, catalog, table_ident, _warehouse) = create_fixture(
        "mor_delete_format_orc_bounds",
        "target",
        mor_properties(&[("write.format.default", "orc")]),
        FormatVersion::V2,
        false,
        true,
    )
    .await;
    insert_values(&context, &table_ident, "(1, 'a'), (2, 'b'), (3, 'c')").await;
    delete_where(&context, &table_ident, "id = 2").await;

    let rows = files_content_format(&context, &table_ident).await;
    let content_format: Vec<(i32, String)> = rows
        .iter()
        .map(|(content, format, _)| (*content, format.clone()))
        .collect();
    assert_eq!(
        content_format,
        vec![(0, "ORC".to_string()), (1, "ORC".to_string())],
        "an ORC table takes ORC data and ORC position deletes, got {rows:?}"
    );
    let data_path = rows[0].2.clone();
    assert!(
        data_path.len() > 16,
        "the data path must exceed the 16-byte truncation window, got {data_path}"
    );

    let deletes = committed_position_deletes(&catalog, &table_ident).await;
    assert_eq!(
        deletes.len(),
        1,
        "one committed position delete, got {deletes:?}"
    );
    let expected = Datum::string(data_path.clone());
    assert_eq!(
        deletes[0]
            .lower_bounds()
            .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH),
        Some(&expected),
        "the committed lower bound is the full data path"
    );
    assert_eq!(
        deletes[0]
            .upper_bounds()
            .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH),
        Some(&expected),
        "the committed upper bound is the full data path"
    );
    assert_eq!(
        files_content_format(&context, &table_ident).await.len(),
        2,
        "one data file plus one delete file after the commit"
    );
}

#[tokio::test]
async fn test_mor_delete_on_orc_honours_the_none_compression_codec() {
    let (context, _catalog, table_ident, _warehouse) = create_fixture(
        "mor_delete_format_orc_codec",
        "target",
        mor_properties(&[
            ("write.format.default", "orc"),
            ("write.orc.compression-codec", "none"),
        ]),
        FormatVersion::V2,
        false,
        false,
    )
    .await;
    insert_values(&context, &table_ident, "(1, 'a'), (2, 'b'), (3, 'c')").await;
    delete_where(&context, &table_ident, "id = 2").await;

    let rows = files_content_format(&context, &table_ident).await;
    let content_format: Vec<(i32, String)> = rows
        .iter()
        .map(|(content, format, _)| (*content, format.clone()))
        .collect();
    assert_eq!(
        content_format,
        vec![(0, "ORC".to_string()), (1, "ORC".to_string())],
        "an ORC table takes ORC data and ORC position deletes, got {rows:?}"
    );
    let delete_path = &rows[1].2;
    assert!(
        delete_path.ends_with(".orc"),
        "delete file {delete_path} keeps the ORC suffix"
    );
    let bytes = std::fs::read(local_fs_path(delete_path)).expect("read delete file bytes");
    assert_eq!(
        orc_postscript_compression_kind(&bytes),
        0,
        "the codec property must reach the delete postscript"
    );
    assert_eq!(
        files_content_format(&context, &table_ident).await.len(),
        2,
        "one data file plus one delete file after the commit"
    );
}
