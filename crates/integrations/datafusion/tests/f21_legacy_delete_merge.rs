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

#![allow(unused_assignments)]

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, RecordBatch, StringArray, UInt64Array};
use datafusion::execution::context::SessionContext;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, Literal, NestedField, PrimitiveType,
    Schema, Struct, Transform, Type, UnboundPartitionSpec,
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

async fn live_delete_files(table: &Table) -> Vec<DataFile> {
    let snapshot = table.metadata().current_snapshot().unwrap();
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .unwrap();
    let mut files = Vec::new();
    for mf in manifest_list.entries() {
        if mf.content != iceberg::spec::ManifestContentType::Deletes {
            continue;
        }
        let manifest = mf.load_manifest(table.file_io()).await.unwrap();
        for entry in manifest.entries() {
            if entry.is_alive() {
                files.push(entry.data_file().clone());
            }
        }
    }
    files
}

#[tokio::test]
async fn test_f21_base_cell_delete_merges_parquet_into_dv() {
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
    let namespace = NamespaceIdent::new("ns".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .unwrap();
    let loc = format!("{}/t1", warehouse.path().to_str().unwrap());
    let creation = TableCreation::builder()
        .location(loc)
        .name("t1".to_string())
        .properties(HashMap::from([
            ("write.delete.mode".to_string(), "merge-on-read".to_string()),
            ("write.update.mode".to_string(), "merge-on-read".to_string()),
        ]))
        .schema(v2_mor_schema())
        .build();
    let table_ident = TableIdent::new(namespace.clone(), "t1".to_string());
    catalog.create_table(&namespace, creation).await.unwrap();
    let mut table = catalog.load_table(&table_ident).await.unwrap();
    let data_file = write_data_file(&table, "data.parquet", &[
        (1, "a"),
        (2, "b"),
        (3, "c"),
        (4, "d"),
    ])
    .await;
    let data_path = data_file.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![data_file])
        .apply(tx)
        .unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    let delete_file = write_pos_delete(&table, &[(data_path.clone(), 1)]).await;
    assert_eq!(delete_file.record_count(), 1);
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    let deletes = live_delete_files(&table).await;
    assert_eq!(deletes.len(), 1);
    assert_eq!(deletes[0].file_format(), DataFileFormat::Parquet);
    let tx = Transaction::new(&table);
    let tx = tx
        .upgrade_table_version()
        .set_format_version(FormatVersion::V3)
        .apply(tx)
        .unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    assert_eq!(table.metadata().format_version(), FormatVersion::V3);
    let client = catalog.clone() as Arc<dyn Catalog>;
    let provider = IcebergCatalogProvider::try_new(client.clone())
        .await
        .unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));
    let result = ctx
        .sql("DELETE FROM catalog.ns.t1 WHERE id = 3")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let deleted = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0);
    assert_eq!(deleted, 1);
    let table = catalog.load_table(&table_ident).await.unwrap();
    let deletes = live_delete_files(&table).await;
    assert_eq!(deletes.len(), 1, "one DV should remain");
    assert_eq!(deletes[0].file_format(), DataFileFormat::Puffin);
    assert_eq!(
        deletes[0].record_count(),
        2,
        "DV must have merged 2 positions"
    );
    let batches = ctx
        .sql("SELECT id FROM catalog.ns.t1 ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let ids: Vec<i32> = batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(ids, vec![1, 4]);
}

#[tokio::test]
async fn test_f21_partition_scoped_merge_keeps_parquet() {
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
    let namespace = NamespaceIdent::new("ns2".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .unwrap();
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "part", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .unwrap();
    let spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "part", Transform::Identity)
        .unwrap()
        .build();
    let loc = format!("{}/t2", warehouse.path().to_str().unwrap());
    let creation = TableCreation::builder()
        .location(loc)
        .name("t2".to_string())
        .properties(HashMap::from([
            ("write.delete.mode".to_string(), "merge-on-read".to_string()),
            ("write.update.mode".to_string(), "merge-on-read".to_string()),
        ]))
        .schema(schema)
        .partition_spec(spec)
        .build();
    let table_ident = TableIdent::new(namespace.clone(), "t2".to_string());
    catalog.create_table(&namespace, creation).await.unwrap();
    let mut table = catalog.load_table(&table_ident).await.unwrap();
    let data_file1 = {
        use iceberg::arrow::schema_to_arrow_schema;
        use iceberg::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
        let schema = table.metadata().current_schema();
        let arrow_schema = Arc::new(schema_to_arrow_schema(schema).unwrap());
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(datafusion::arrow::array::Int32Array::from(vec![1, 2])) as _,
            Arc::new(StringArray::from(vec!["a".to_string(), "a".to_string()])) as _,
        ])
        .unwrap();
        let path = format!("{}/data/f1.parquet", table.metadata().location());
        let out = table.file_io().new_output(path.clone()).unwrap();
        let parquet_builder = ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            schema.clone(),
        );
        let mut writer = parquet_builder.build(out).await.unwrap();
        writer.write(&batch).await.unwrap();
        let mut b = writer.close().await.unwrap().into_iter().next().unwrap();
        b.content(DataContentType::Data)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::string("a"))]))
            .build()
            .unwrap()
    };
    let data_file2 = {
        use iceberg::arrow::schema_to_arrow_schema;
        use iceberg::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
        let schema = table.metadata().current_schema();
        let arrow_schema = Arc::new(schema_to_arrow_schema(schema).unwrap());
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(datafusion::arrow::array::Int32Array::from(vec![3, 4])) as _,
            Arc::new(StringArray::from(vec!["a".to_string(), "a".to_string()])) as _,
        ])
        .unwrap();
        let path = format!("{}/data/f2.parquet", table.metadata().location());
        let out = table.file_io().new_output(path.clone()).unwrap();
        let parquet_builder = ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            schema.clone(),
        );
        let mut writer = parquet_builder.build(out).await.unwrap();
        writer.write(&batch).await.unwrap();
        let mut b = writer.close().await.unwrap().into_iter().next().unwrap();
        b.content(DataContentType::Data)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::string("a"))]))
            .build()
            .unwrap()
    };
    let p1 = data_file1.file_path().to_string();
    let p2 = data_file2.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![data_file1, data_file2])
        .apply(tx)
        .unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    let pos_delete = {
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
            iceberg::writer::base_writer::position_delete_writer::position_delete_writer_properties(
            ),
            config.schema().clone(),
        )
        .with_metrics_config(iceberg::spec::MetricsConfig::for_position_delete());
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            table.file_io().clone(),
            location_gen,
            file_name_gen,
        );
        let partition_key = iceberg::spec::PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            table.metadata().current_schema().clone(),
            Struct::from_iter([Some(Literal::string("a"))]),
        )
        .unwrap();
        let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
            .build(Some(partition_key))
            .await
            .unwrap();
        let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
            Arc::new(StringArray::from(vec![p1.clone(), p2.clone()])) as _,
            Arc::new(Int64Array::from(vec![0, 1])) as _,
        ])
        .unwrap();
        writer.write(batch).await.unwrap();
        writer.close().await.unwrap().into_iter().next().unwrap()
    };
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![pos_delete.clone()])
        .apply(tx)
        .unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    let tx = Transaction::new(&table);
    let tx = tx
        .upgrade_table_version()
        .set_format_version(FormatVersion::V3)
        .apply(tx)
        .unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    let client = catalog.clone() as Arc<dyn Catalog>;
    let provider = IcebergCatalogProvider::try_new(client.clone())
        .await
        .unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));
    ctx.sql("DELETE FROM catalog.ns2.t2 WHERE id = 2")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let table = catalog.load_table(&table_ident).await.unwrap();
    let deletes = live_delete_files(&table).await;
    assert_eq!(deletes.len(), 2, "parquet plus DV");
    let puffin_count = deletes
        .iter()
        .filter(|f| f.file_format() == DataFileFormat::Puffin)
        .count();
    let parquet_count = deletes
        .iter()
        .filter(|f| f.file_format() == DataFileFormat::Parquet)
        .count();
    assert_eq!(puffin_count, 1);
    assert_eq!(parquet_count, 1);
    let dv = deletes
        .iter()
        .find(|f| f.file_format() == DataFileFormat::Puffin)
        .unwrap();
    assert_eq!(dv.record_count(), 2);
}

#[tokio::test]
async fn test_f21_update_merges_parquet_into_dv() {
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
    let namespace = NamespaceIdent::new("ns3".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .unwrap();
    let loc = format!("{}/t3", warehouse.path().to_str().unwrap());
    let creation = TableCreation::builder()
        .location(loc)
        .name("t3".to_string())
        .properties(HashMap::from([
            ("write.delete.mode".to_string(), "merge-on-read".to_string()),
            ("write.update.mode".to_string(), "merge-on-read".to_string()),
        ]))
        .schema(v2_mor_schema())
        .build();
    let table_ident = TableIdent::new(namespace.clone(), "t3".to_string());
    catalog.create_table(&namespace, creation).await.unwrap();
    let mut table = catalog.load_table(&table_ident).await.unwrap();
    let data_file = write_data_file(&table, "data.parquet", &[
        (1, "a"),
        (2, "b"),
        (3, "c"),
        (4, "d"),
    ])
    .await;
    let data_path = data_file.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![data_file])
        .apply(tx)
        .unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    let delete_file = write_pos_delete(&table, &[(data_path.clone(), 1)]).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    let tx = Transaction::new(&table);
    let tx = tx
        .upgrade_table_version()
        .set_format_version(FormatVersion::V3)
        .apply(tx)
        .unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    let client = catalog.clone() as Arc<dyn Catalog>;
    let provider = IcebergCatalogProvider::try_new(client.clone())
        .await
        .unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));
    ctx.sql("UPDATE catalog.ns3.t3 SET data = 'z' WHERE id = 3")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let table = catalog.load_table(&table_ident).await.unwrap();
    let deletes = live_delete_files(&table).await;
    assert_eq!(deletes.len(), 1);
    assert_eq!(deletes[0].file_format(), DataFileFormat::Puffin);
    assert_eq!(deletes[0].record_count(), 2);
}

#[tokio::test]
async fn test_f21_untouched_file_stays_live() {
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
    let namespace = NamespaceIdent::new("ns4".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .unwrap();
    let loc = format!("{}/t4", warehouse.path().to_str().unwrap());
    let creation = TableCreation::builder()
        .location(loc)
        .name("t4".to_string())
        .properties(HashMap::from([
            ("write.delete.mode".to_string(), "merge-on-read".to_string()),
            ("write.update.mode".to_string(), "merge-on-read".to_string()),
        ]))
        .schema(v2_mor_schema())
        .build();
    let table_ident = TableIdent::new(namespace.clone(), "t4".to_string());
    catalog.create_table(&namespace, creation).await.unwrap();
    let mut table = catalog.load_table(&table_ident).await.unwrap();
    let f1 = write_data_file(&table, "f1.parquet", &[(1, "a"), (2, "b")]).await;
    let f2 = write_data_file(&table, "f2.parquet", &[(3, "c"), (4, "d")]).await;
    let p1 = f1.file_path().to_string();
    let p2 = f2.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![f1, f2])
        .apply(tx)
        .unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    let del = write_pos_delete(&table, &[(p1.clone(), 0)]).await;
    let tx = Transaction::new(&table);
    let tx = tx.row_delta().add_deletes(vec![del]).apply(tx).unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    let tx = Transaction::new(&table);
    let tx = tx
        .upgrade_table_version()
        .set_format_version(FormatVersion::V3)
        .apply(tx)
        .unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    let client = catalog.clone() as Arc<dyn Catalog>;
    let provider = IcebergCatalogProvider::try_new(client.clone())
        .await
        .unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));
    ctx.sql("DELETE FROM catalog.ns4.t4 WHERE id = 4")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let table = catalog.load_table(&table_ident).await.unwrap();
    let deletes = live_delete_files(&table).await;
    assert_eq!(deletes.len(), 2);
    let puffin = deletes
        .iter()
        .find(|f| f.file_format() == DataFileFormat::Puffin)
        .unwrap();
    assert_eq!(puffin.record_count(), 1);
    assert_eq!(puffin.referenced_data_file().as_deref(), Some(p2.as_str()));
}

#[tokio::test]
async fn test_f21_sequence_number_not_apply() {
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
    let namespace = NamespaceIdent::new("ns5".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .unwrap();
    let loc = format!("{}/t5", warehouse.path().to_str().unwrap());
    let creation = TableCreation::builder()
        .location(loc)
        .name("t5".to_string())
        .properties(HashMap::from([
            ("write.delete.mode".to_string(), "merge-on-read".to_string()),
            ("write.update.mode".to_string(), "merge-on-read".to_string()),
        ]))
        .schema(v2_mor_schema())
        .build();
    let table_ident = TableIdent::new(namespace.clone(), "t5".to_string());
    catalog.create_table(&namespace, creation).await.unwrap();
    let mut table = catalog.load_table(&table_ident).await.unwrap();
    let f1 = write_data_file(&table, "f1.parquet", &[(1, "a"), (2, "b")]).await;
    let tx = Transaction::new(&table);
    let tx = tx.fast_append().add_data_files(vec![f1]).apply(tx).unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    let del = {
        use iceberg::spec::DataFileBuilder;
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("/tmp/pos.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::empty())
            .build()
            .unwrap()
    };
    let tx = Transaction::new(&table);
    let tx = tx.row_delta().add_deletes(vec![del]).apply(tx).unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    let f2 = write_data_file(&table, "f2.parquet", &[(3, "c"), (4, "d")]).await;
    let p2 = f2.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx.fast_append().add_data_files(vec![f2]).apply(tx).unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    let tx = Transaction::new(&table);
    let tx = tx
        .upgrade_table_version()
        .set_format_version(FormatVersion::V3)
        .apply(tx)
        .unwrap();
    table = tx.commit(catalog.as_ref()).await.unwrap();
    let client = catalog.clone() as Arc<dyn Catalog>;
    let provider = IcebergCatalogProvider::try_new(client.clone())
        .await
        .unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));
    ctx.sql("DELETE FROM catalog.ns5.t5 WHERE id = 4")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let table = catalog.load_table(&table_ident).await.unwrap();
    let deletes = live_delete_files(&table).await;
    let dv = deletes
        .iter()
        .find(|f| f.file_format() == DataFileFormat::Puffin)
        .unwrap();
    assert_eq!(
        dv.record_count(),
        1,
        "only new delete, old parquet not merged for new file"
    );
    assert_eq!(dv.referenced_data_file().as_deref(), Some(p2.as_str()));
    let parquet_still = deletes
        .iter()
        .any(|f| f.file_format() == DataFileFormat::Parquet);
    assert!(parquet_still, "old parquet for f1 still live");
}
