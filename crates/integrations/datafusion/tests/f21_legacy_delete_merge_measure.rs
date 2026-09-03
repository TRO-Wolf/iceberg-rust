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

use datafusion::arrow::array::{Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::execution::context::SessionContext;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, NestedField, PrimitiveType, Schema,
    Struct, Type,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;
use tempfile::TempDir;

fn v2_mor_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .unwrap()
}

async fn write_data_file(table: &Table, file_name: &str, rows: &[(i32, &str)]) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;
    use iceberg::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).unwrap());
    let ids: Vec<i32> = rows.iter().map(|(id, _)| *id).collect();
    let data_vals: Vec<String> = rows.iter().map(|(_, d)| d.to_string()).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(datafusion::arrow::array::Int32Array::from(ids)) as _,
        Arc::new(StringArray::from(data_vals)) as _,
    ])
    .unwrap();
    let file_path = format!("{}/data/{}", table.metadata().location(), file_name);
    let output = table.file_io().new_output(file_path.clone()).unwrap();
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let mut writer = parquet_builder.build(output).await.unwrap();
    writer.write(&batch).await.unwrap();
    let builders = writer.close().await.unwrap();
    let mut b = builders.into_iter().next().unwrap();
    b.content(DataContentType::Data)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .unwrap()
}

async fn write_pos_delete(table: &Table, deletes: &[(String, i64)]) -> DataFile {
    use iceberg::spec::PartitionKey;
    use iceberg::writer::base_writer::position_delete_writer::{
        PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
    };
    use iceberg::writer::file_writer::ParquetWriterBuilder;
    use iceberg::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
    let config = PositionDeleteWriterConfig::new().unwrap();
    let location_gen = DefaultLocationGenerator::new(table.metadata().clone()).unwrap();
    let file_name_gen = DefaultFileNameGenerator::new(
        "pos-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        iceberg::writer::base_writer::position_delete_writer::position_delete_writer_properties(),
        config.schema().clone(),
    )
    .with_metrics_config(iceberg::spec::MetricsConfig::for_position_delete());
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let partition_key = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        Struct::empty(),
    )
    .unwrap();
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(Some(partition_key))
        .await
        .unwrap();
    let paths: Vec<&str> = deletes.iter().map(|(p, _)| p.as_str()).collect();
    let positions: Vec<i64> = deletes.iter().map(|(_, pos)| *pos).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as _,
        Arc::new(Int64Array::from(positions)) as _,
    ])
    .unwrap();
    writer.write(batch).await.unwrap();
    writer.close().await.unwrap().into_iter().next().unwrap()
}

#[tokio::test]
#[ignore = "measurement, not a CI pin"]
async fn test_f21_measure_k8_partition_scoped_100k() {
    let warehouse = TempDir::new().unwrap();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                warehouse.path().to_str().unwrap().to_string(),
            )]),
        )
        .await
        .unwrap();
    let catalog = Arc::new(catalog);
    let namespace = NamespaceIdent::new("ns_k8".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .unwrap();
    let loc = format!("{}/tk8", warehouse.path().to_str().unwrap());
    let creation = TableCreation::builder()
        .location(loc)
        .name("tk8".to_string())
        .properties(HashMap::from([
            ("write.delete.mode".to_string(), "merge-on-read".to_string()),
            ("write.update.mode".to_string(), "merge-on-read".to_string()),
        ]))
        .schema(v2_mor_schema())
        .build();
    let table_ident = TableIdent::new(namespace.clone(), "tk8".to_string());
    catalog.create_table(&namespace, creation).await.unwrap();
    let mut table = catalog.load_table(&table_ident).await.unwrap();
    let mut files = Vec::new();
    let mut paths = Vec::new();
    for i in 0..8 {
        let f = write_data_file(&table, &format!("k8-{i}.parquet"), &[
            (i * 2, "a"),
            (i * 2 + 1, "b"),
        ])
        .await;
        paths.push(f.file_path().to_string());
        files.push(f);
    }
    let tx = Transaction::new(&table);
    let tx = tx.fast_append().add_data_files(files).apply(tx).unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    let mut deletes = Vec::with_capacity(100_000);
    for (i, path) in paths.iter().enumerate() {
        for pos in 0..12_500 {
            deletes.push((path.clone(), i as i64 * 12_500 + pos));
        }
    }
    let pos_delete = write_pos_delete(&table, &deletes).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![pos_delete])
        .apply(tx)
        .unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    let tx = Transaction::new(&table);
    let tx = tx
        .upgrade_table_version()
        .set_format_version(FormatVersion::V3)
        .apply(tx)
        .unwrap();
    tx.commit(catalog.as_ref()).await.unwrap();
    let client = catalog.clone() as Arc<dyn Catalog>;
    let provider = IcebergCatalogProvider::try_new(client.clone())
        .await
        .unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));
    let start = std::time::Instant::now();
    ctx.sql("DELETE FROM catalog.ns_k8.tk8 WHERE id % 2 = 1")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let elapsed = start.elapsed();
    println!("F21 K=8 100k DELETE elapsed={elapsed:?}");
    assert!(elapsed.as_secs_f64() > 0.0);
}

#[tokio::test]
#[ignore = "measurement, not a CI pin"]
async fn test_f21_measure_row_column_100k() {
    use iceberg::metadata_columns::{
        RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
    };
    use iceberg::spec::DataFileBuilder;
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
    use parquet::file::properties::WriterProperties;
    let warehouse = TempDir::new().unwrap();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                warehouse.path().to_str().unwrap().to_string(),
            )]),
        )
        .await
        .unwrap();
    let catalog = Arc::new(catalog);
    let namespace = NamespaceIdent::new("ns_row".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .unwrap();
    let loc = format!("{}/trow", warehouse.path().to_str().unwrap());
    let creation = TableCreation::builder()
        .location(loc)
        .name("trow".to_string())
        .properties(HashMap::from([
            ("write.delete.mode".to_string(), "merge-on-read".to_string()),
            ("write.update.mode".to_string(), "merge-on-read".to_string()),
        ]))
        .schema(v2_mor_schema())
        .build();
    let table_ident = TableIdent::new(namespace.clone(), "trow".to_string());
    catalog.create_table(&namespace, creation).await.unwrap();
    let mut table = catalog.load_table(&table_ident).await.unwrap();
    let data_file = write_data_file(&table, "data.parquet", &[(1, "a"), (2, "b")]).await;
    let data_path = data_file.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![data_file])
        .apply(tx)
        .unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    let del_path = format!("{}/data/row-del.parquet", table.metadata().location());
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("file_path", DataType::Utf8, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            RESERVED_FIELD_ID_DELETE_FILE_PATH.to_string(),
        )])),
        Field::new("pos", DataType::Int64, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            RESERVED_FIELD_ID_DELETE_FILE_POS.to_string(),
        )])),
        Field::new("row", DataType::Utf8, true),
    ]));
    let pad = "x".repeat(200);
    let paths: Vec<String> = vec![data_path.clone(); 100_000];
    let positions: Vec<i64> = (0..100_000).map(|i| i as i64).collect();
    let rows: Vec<String> = vec![pad; 100_000];
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(StringArray::from(paths)) as _,
        Arc::new(Int64Array::from(positions)) as _,
        Arc::new(StringArray::from(rows)) as _,
    ])
    .unwrap();
    {
        let file = std::fs::File::create(&del_path).unwrap();
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, arrow_schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    let file_size = std::fs::metadata(&del_path).unwrap().len();
    let del = DataFileBuilder::default()
        .content(DataContentType::PositionDeletes)
        .file_path(del_path)
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(file_size)
        .record_count(100_000)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .referenced_data_file(Some(data_path))
        .build()
        .unwrap();
    let tx = Transaction::new(&table);
    let tx = tx.row_delta().add_deletes(vec![del]).apply(tx).unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    let tx = Transaction::new(&table);
    let tx = tx
        .upgrade_table_version()
        .set_format_version(FormatVersion::V3)
        .apply(tx)
        .unwrap();
    tx.commit(catalog.as_ref()).await.unwrap();
    let client = catalog.clone() as Arc<dyn Catalog>;
    let provider = IcebergCatalogProvider::try_new(client.clone())
        .await
        .unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));
    let start = std::time::Instant::now();
    ctx.sql("DELETE FROM catalog.ns_row.trow WHERE id = 2")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let elapsed = start.elapsed();
    println!("F21 row-column 100k DELETE elapsed={elapsed:?}");
    assert!(elapsed.as_secs_f64() > 0.0);
}
