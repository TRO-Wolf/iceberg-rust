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
    Array, ArrayRef, Decimal128Array, Int64Array, RecordBatch, StringArray, StructArray,
};
use arrow_schema::{DataType, Fields};
use futures::TryStreamExt;
use iceberg::arrow::{arrow_schema_to_schema, schema_to_arrow_schema};
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, NestedField, PrimitiveType, Schema,
    SchemaRef, SortOrder, Struct, StructType, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::equality_delete_writer::{
    EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
};
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::file_writer::{
    AvroWriterBuilder, FileWriter, FileWriterBuilder, OrcWriterBuilder, ParquetWriterBuilder,
};
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation};
use tempfile::TempDir;

async fn setup(schema: Schema, version: FormatVersion) -> (MemoryCatalog, Table, TempDir) {
    let warehouse = TempDir::new().expect("create warehouse");
    let warehouse_path = warehouse.path().to_string_lossy().to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
        )
        .await
        .expect("build memory catalog");
    let namespace = NamespaceIdent::new("eqdel_key_paths".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("target".to_string())
        .location(format!("{warehouse_path}/target"))
        .schema(schema)
        .partition_spec(UnboundPartitionSpec::builder().build())
        .sort_order(SortOrder::unsorted_order())
        .format_version(version)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table");
    (catalog, table, warehouse)
}

fn arrow_batch(schema: &Schema, columns: Vec<ArrayRef>) -> RecordBatch {
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("schema to arrow"));
    RecordBatch::try_new(arrow_schema, columns).expect("build batch")
}

fn struct_column(schema: &Schema, name: &str, children: Vec<ArrayRef>, valid: &[bool]) -> ArrayRef {
    let arrow_schema = schema_to_arrow_schema(schema).expect("schema to arrow");
    let DataType::Struct(fields) = arrow_schema
        .field_with_name(name)
        .expect("struct field present")
        .data_type()
        .clone()
    else {
        panic!("{name} is a struct");
    };
    let fields: Fields = fields;
    Arc::new(StructArray::new(
        fields,
        children,
        Some(valid.iter().copied().collect()),
    ))
}

async fn write_file<B: FileWriterBuilder>(
    table: &Table,
    builder: B,
    extension: &str,
    batch: &RecordBatch,
    content: DataContentType,
    equality_ids: Option<Vec<i32>>,
) -> DataFile {
    let path = format!(
        "{}/data/{}-fixture.{extension}",
        table.metadata().location(),
        uuid::Uuid::now_v7()
    );
    let output = table.file_io().new_output(path).expect("new output");
    let mut writer = builder.build(output).await.expect("build file writer");
    writer.write(batch).await.expect("write batch");
    let mut builder = writer
        .close()
        .await
        .expect("close file writer")
        .into_iter()
        .next()
        .expect("one file builder");
    builder
        .content(content)
        .equality_ids(equality_ids)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .expect("build data file")
}

async fn write_parquet(
    table: &Table,
    schema: SchemaRef,
    batch: &RecordBatch,
    content: DataContentType,
    equality_ids: Option<Vec<i32>>,
) -> DataFile {
    let builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema,
    );
    write_file(table, builder, "parquet", batch, content, equality_ids).await
}

async fn write_rust_eq_delete(
    table: &Table,
    equality_ids: Vec<i32>,
    batch: RecordBatch,
) -> DataFile {
    let schema = table.metadata().current_schema().clone();
    let config =
        EqualityDeleteWriterConfig::new(equality_ids, schema).expect("equality-delete config");
    let projected = Arc::new(
        arrow_schema_to_schema(config.projected_arrow_schema_ref()).expect("projected schema"),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            projected,
        ),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).expect("location generator"),
        DefaultFileNameGenerator::new(
            "eqdel".to_string(),
            Some(uuid::Uuid::now_v7().to_string()),
            DataFileFormat::Parquet,
        ),
    );
    let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .unpartitioned()
        .build(None)
        .await
        .expect("build equality-delete writer");
    writer.write(batch).await.expect("write deletes");
    writer
        .close()
        .await
        .expect("close equality-delete writer")
        .into_iter()
        .next()
        .expect("one equality-delete file")
}

async fn append(catalog: &MemoryCatalog, table: &Table, file: DataFile) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file])
        .apply(tx)
        .expect("apply fast append");
    tx.commit(catalog).await.expect("commit fast append")
}

async fn commit_delete(catalog: &MemoryCatalog, table: &Table, file: DataFile) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![file])
        .apply(tx)
        .expect("apply row delta");
    tx.commit(catalog).await.expect("commit row delta")
}

async fn drop_column(catalog: &MemoryCatalog, table: &Table, name: &str) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .update_schema()
        .delete_column(name)
        .apply(tx)
        .expect("apply drop column");
    tx.commit(catalog).await.expect("commit drop column")
}

async fn scan(table: &Table, columns: &[&str]) -> iceberg::Result<Vec<RecordBatch>> {
    let stream = table
        .scan()
        .select(columns.iter().copied())
        .build()?
        .to_arrow()
        .await?;
    stream.try_collect().await
}

fn strings(batches: &[RecordBatch], column: &str) -> Vec<String> {
    let mut values = Vec::new();
    for batch in batches {
        let array = batch
            .column_by_name(column)
            .expect("column present")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("string column");
        for row in 0..array.len() {
            values.push(array.value(row).to_string());
        }
    }
    values.sort();
    values
}

fn longs(batches: &[RecordBatch], column: &str) -> Vec<Option<i64>> {
    let mut values = Vec::new();
    for batch in batches {
        let array = batch
            .column_by_name(column)
            .expect("column present")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("long column");
        for row in 0..array.len() {
            values.push(array.is_valid(row).then(|| array.value(row)));
        }
    }
    values
}

fn decimal_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Decimal {
                    precision: 10,
                    scale: 2,
                }),
            )
            .into(),
            NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("decimal schema")
}

fn decimals(values: &[i128]) -> ArrayRef {
    Arc::new(
        Decimal128Array::from(values.to_vec())
            .with_precision_and_scale(10, 2)
            .expect("decimal(10, 2)"),
    )
}

async fn decimal_table_with_dropped_key(version: FormatVersion) -> (MemoryCatalog, Table, TempDir) {
    let (catalog, table, warehouse) = setup(decimal_schema(), version).await;
    let schema = table.metadata().current_schema().clone();
    let data = arrow_batch(&schema, vec![
        decimals(&[1000, 2000, 3000, 4000, 5000]),
        Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"])),
    ]);
    let file = write_parquet(&table, schema.clone(), &data, DataContentType::Data, None).await;
    let table = append(&catalog, &table, file).await;
    let deletes = arrow_batch(&schema, vec![
        decimals(&[2000, 4000]),
        Arc::new(StringArray::from(vec!["x", "x"])),
    ]);
    let delete = write_rust_eq_delete(&table, vec![1], deletes).await;
    let table = commit_delete(&catalog, &table, delete).await;
    let before = scan(&table, &["data", "_pos"])
        .await
        .expect("scan before drop");
    assert_eq!(strings(&before, "data"), vec!["a", "c", "e"]);
    let table = drop_column(&catalog, &table, "id").await;
    (catalog, table, warehouse)
}

#[tokio::test]
async fn decimal_key_dropped_pos_projection_answers_survivors() {
    let (_catalog, table, _warehouse) = decimal_table_with_dropped_key(FormatVersion::V2).await;
    let batches = scan(&table, &["data", "_pos"])
        .await
        .expect("a _pos scan after the drop");
    assert_eq!(strings(&batches, "data"), vec!["a", "c", "e"]);
    assert_eq!(longs(&batches, "_pos"), vec![Some(0), Some(2), Some(4)]);
}

#[tokio::test]
async fn decimal_key_dropped_row_id_projection_answers_survivors() {
    let (_catalog, table, _warehouse) = decimal_table_with_dropped_key(FormatVersion::V3).await;
    let batches = scan(&table, &["data", "_row_id"])
        .await
        .expect("a _row_id scan after the drop");
    assert_eq!(strings(&batches, "data"), vec!["a", "c", "e"]);
    assert_eq!(longs(&batches, "_row_id").len(), 3);
}

#[tokio::test]
async fn long_key_dropped_pos_projection_keeps_post_drop_rows() {
    let (catalog, table, _warehouse) = setup(optional_key_schema(), FormatVersion::V2).await;
    let schema = table.metadata().current_schema().clone();
    let data = arrow_batch(&schema, vec![
        Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50])),
        Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"])),
    ]);
    let file = write_parquet(&table, schema.clone(), &data, DataContentType::Data, None).await;
    let table = append(&catalog, &table, file).await;
    let deletes = arrow_batch(&schema, vec![
        Arc::new(Int64Array::from(vec![20, 40])),
        Arc::new(StringArray::from(vec!["x", "x"])),
    ]);
    let delete = write_rust_eq_delete(&table, vec![1], deletes).await;
    let table = commit_delete(&catalog, &table, delete).await;
    let table = drop_column(&catalog, &table, "id").await;
    let schema = table.metadata().current_schema().clone();
    let later = arrow_batch(&schema, vec![Arc::new(StringArray::from(vec!["f", "g"]))]);
    let file = write_parquet(&table, schema, &later, DataContentType::Data, None).await;
    let table = append(&catalog, &table, file).await;
    for columns in [&["data"][..], &["data", "_pos"][..]] {
        let batches = scan(&table, columns).await.expect("scan after the drop");
        assert_eq!(
            strings(&batches, "data"),
            vec!["a", "c", "e", "f", "g"],
            "{columns:?}"
        );
    }
}

#[tokio::test]
async fn composite_key_with_null_first_column_dropped_pos_projection_answers_survivors() {
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::optional(1, "x", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(3, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("composite schema");
    let (catalog, table, _warehouse) = setup(schema, FormatVersion::V2).await;
    let schema = table.metadata().current_schema().clone();
    let data = arrow_batch(&schema, vec![
        Arc::new(Int64Array::from(vec![
            None,
            Some(1),
            Some(2),
            None,
            Some(3),
        ])),
        Arc::new(Int64Array::from(vec![20, 20, 30, 40, 50])),
        Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"])),
    ]);
    let file = write_parquet(&table, schema.clone(), &data, DataContentType::Data, None).await;
    let table = append(&catalog, &table, file).await;
    let deletes = arrow_batch(&schema, vec![
        Arc::new(Int64Array::from(vec![None, Some(2)])),
        Arc::new(Int64Array::from(vec![20, 30])),
        Arc::new(StringArray::from(vec!["x", "x"])),
    ]);
    let delete = write_rust_eq_delete(&table, vec![1, 2], deletes).await;
    let table = commit_delete(&catalog, &table, delete).await;
    let before = scan(&table, &["data", "_pos"])
        .await
        .expect("scan before drop");
    assert_eq!(strings(&before, "data"), vec!["b", "d", "e"]);
    let table = drop_column(&catalog, &table, "y").await;
    let batches = scan(&table, &["data", "_pos"])
        .await
        .expect("a _pos scan after the drop");
    assert_eq!(strings(&batches, "data"), vec!["b", "d", "e"]);
    assert_eq!(longs(&batches, "_pos"), vec![Some(1), Some(3), Some(4)]);
}

fn nested_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(
                2,
                "s",
                Type::Struct(StructType::new(vec![
                    NestedField::optional(3, "k", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(4, "z", Type::Primitive(PrimitiveType::String)).into(),
                ])),
            )
            .into(),
            NestedField::optional(5, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("nested schema")
}

fn spec_shaped_delete_schema(struct_id: i32, key_id: i32) -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::optional(
                struct_id,
                "s",
                Type::Struct(StructType::new(vec![
                    NestedField::optional(key_id, "k", Type::Primitive(PrimitiveType::Long)).into(),
                ])),
            )
            .into(),
        ])
        .build()
        .expect("spec-shaped delete schema")
}

async fn nested_table(spec_shaped: bool) -> (MemoryCatalog, Table, TempDir) {
    let (catalog, table, warehouse) = setup(nested_schema(), FormatVersion::V2).await;
    let schema = table.metadata().current_schema().clone();
    let struct_id = schema.field_by_name("s").expect("s").id;
    let key_id = schema.field_by_name("s.k").expect("s.k").id;
    let keys: ArrayRef = Arc::new(Int64Array::from(vec![
        Some(10),
        Some(20),
        Some(30),
        Some(40),
        Some(50),
        None,
    ]));
    let zs: ArrayRef = Arc::new(StringArray::from(vec!["p", "q", "r", "s", "t", "u"]));
    let data = arrow_batch(&schema, vec![
        Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6])),
        struct_column(&schema, "s", vec![keys, zs], &[true; 6]),
        Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e", "f"])),
    ]);
    let file = write_parquet(&table, schema.clone(), &data, DataContentType::Data, None).await;
    let table = append(&catalog, &table, file).await;
    let delete = if spec_shaped {
        let delete_schema = Arc::new(spec_shaped_delete_schema(struct_id, key_id));
        let keys: ArrayRef = Arc::new(Int64Array::from(vec![20, 40]));
        let batch = arrow_batch(&delete_schema, vec![struct_column(
            &delete_schema,
            "s",
            vec![keys],
            &[true, true],
        )]);
        write_parquet(
            &table,
            delete_schema,
            &batch,
            DataContentType::EqualityDeletes,
            Some(vec![key_id]),
        )
        .await
    } else {
        let keys: ArrayRef = Arc::new(Int64Array::from(vec![20, 40]));
        let zs: ArrayRef = Arc::new(StringArray::from(vec!["x", "x"]));
        let batch = arrow_batch(&schema, vec![
            Arc::new(Int64Array::from(vec![0, 0])),
            struct_column(&schema, "s", vec![keys, zs], &[true, true]),
            Arc::new(StringArray::from(vec!["x", "x"])),
        ]);
        write_rust_eq_delete(&table, vec![key_id], batch).await
    };
    let table = commit_delete(&catalog, &table, delete).await;
    (catalog, table, warehouse)
}

const NESTED_SURVIVORS: [&str; 4] = ["a", "c", "e", "f"];

#[tokio::test]
async fn nested_key_spec_shaped_delete_answers_survivors() {
    let (catalog, table, _warehouse) = nested_table(true).await;
    let without_struct = scan(&table, &["data"]).await.expect("scan without s");
    assert_eq!(strings(&without_struct, "data"), NESTED_SURVIVORS);
    let with_struct = scan(&table, &["data", "s"]).await.expect("scan with s");
    assert_eq!(strings(&with_struct, "data"), NESTED_SURVIVORS);
    let with_pos = scan(&table, &["data", "_pos"])
        .await
        .expect("scan with _pos");
    assert_eq!(strings(&with_pos, "data"), NESTED_SURVIVORS);
    let table = drop_column(&catalog, &table, "id").await;
    let after_drop = scan(&table, &["data"])
        .await
        .expect("scan after unrelated drop");
    assert_eq!(strings(&after_drop, "data"), NESTED_SURVIVORS);
    let after_drop = scan(&table, &["data", "s"])
        .await
        .expect("scan with s after unrelated drop");
    assert_eq!(strings(&after_drop, "data"), NESTED_SURVIVORS);
    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .rename_column("s.k", "key")
        .apply(tx)
        .expect("apply nested rename");
    let table = tx.commit(&catalog).await.expect("commit nested rename");
    let after_rename = scan(&table, &["data", "s"])
        .await
        .expect("scan after nested rename");
    assert_eq!(strings(&after_rename, "data"), NESTED_SURVIVORS);
}

#[tokio::test]
async fn nested_key_rust_writer_delete_answers_survivors() {
    let (catalog, table, _warehouse) = nested_table(false).await;
    let without_struct = scan(&table, &["data"]).await.expect("scan without s");
    assert_eq!(strings(&without_struct, "data"), NESTED_SURVIVORS);
    let with_struct = scan(&table, &["data", "s"]).await.expect("scan with s");
    assert_eq!(strings(&with_struct, "data"), NESTED_SURVIVORS);
    let table = drop_column(&catalog, &table, "id").await;
    let after_drop = scan(&table, &["data"])
        .await
        .expect("scan after unrelated drop");
    assert_eq!(strings(&after_drop, "data"), NESTED_SURVIVORS);
}

#[tokio::test]
async fn nested_key_with_struct_dropped_is_loud_or_correct() {
    for spec_shaped in [true, false] {
        let (catalog, table, _warehouse) = nested_table(spec_shaped).await;
        let table = drop_column(&catalog, &table, "s").await;
        for columns in [&["data"][..], &["data", "_pos"][..]] {
            match scan(&table, columns).await {
                Ok(batches) => assert_eq!(
                    strings(&batches, "data"),
                    NESTED_SURVIVORS,
                    "spec_shaped={spec_shaped} columns={columns:?}"
                ),
                Err(error) => assert!(
                    matches!(
                        error.kind(),
                        ErrorKind::FeatureUnsupported | ErrorKind::DataInvalid
                    ),
                    "spec_shaped={spec_shaped} columns={columns:?}: {error}"
                ),
            }
        }
    }
}

fn optional_key_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("optional-key schema")
}

fn data_only_schema() -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("data-only schema"),
    )
}

async fn assert_missing_key_column_refuses(format: DataFileFormat) {
    let (catalog, table, _warehouse) = setup(optional_key_schema(), FormatVersion::V2).await;
    let schema = table.metadata().current_schema().clone();
    let data = arrow_batch(&schema, vec![
        Arc::new(Int64Array::from(vec![Some(10), None, Some(30)])),
        Arc::new(StringArray::from(vec!["a", "b", "c"])),
    ]);
    let file = write_parquet(&table, schema.clone(), &data, DataContentType::Data, None).await;
    let table = append(&catalog, &table, file).await;
    let delete_schema = data_only_schema();
    let batch = arrow_batch(&delete_schema, vec![Arc::new(StringArray::from(vec![
        "zzz",
    ]))]);
    let content = DataContentType::EqualityDeletes;
    let delete = match format {
        DataFileFormat::Parquet => {
            write_parquet(&table, delete_schema, &batch, content, Some(vec![1])).await
        }
        DataFileFormat::Avro => {
            let builder = AvroWriterBuilder::new(delete_schema);
            write_file(&table, builder, "avro", &batch, content, Some(vec![1])).await
        }
        DataFileFormat::Orc => {
            let builder = OrcWriterBuilder::new(delete_schema);
            write_file(&table, builder, "orc", &batch, content, Some(vec![1])).await
        }
        DataFileFormat::Puffin => panic!("no Puffin equality deletes"),
    };
    assert_eq!(delete.file_format(), format);
    let delete_path = delete.file_path().to_string();
    let table = commit_delete(&catalog, &table, delete).await;
    for columns in [&["data"][..], &["id", "data"][..], &["data", "_pos"][..]] {
        match scan(&table, columns).await {
            Ok(batches) => panic!(
                "{format:?} {columns:?}: a delete file missing its key column must refuse, got {:?}",
                strings(&batches, "data")
            ),
            Err(error) => {
                assert_eq!(error.kind(), ErrorKind::DataInvalid, "{format:?}: {error}");
                assert!(
                    error.message().contains(&delete_path),
                    "{format:?}: the refusal names the delete file: {error}"
                );
                assert!(
                    error.message().contains("field id 1"),
                    "{format:?}: the refusal names the field id: {error}"
                );
            }
        }
    }
}

#[tokio::test]
async fn parquet_delete_file_missing_key_column_refuses() {
    assert_missing_key_column_refuses(DataFileFormat::Parquet).await;
}

#[tokio::test]
async fn avro_delete_file_missing_key_column_refuses() {
    assert_missing_key_column_refuses(DataFileFormat::Avro).await;
}

#[tokio::test]
async fn orc_delete_file_missing_key_column_refuses() {
    assert_missing_key_column_refuses(DataFileFormat::Orc).await;
}
