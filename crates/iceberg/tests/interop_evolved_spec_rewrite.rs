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
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch};
use futures::TryStreamExt;
use iceberg::expr::Reference;
use iceberg::io::{FileIO, LocalFsStorageFactory};
use iceberg::maintenance::RewriteDataFiles;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::metadata_columns::RESERVED_COL_NAME_ROW_ID;
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, Datum, FormatVersion, Literal, NestedField,
    PartitionSpec, PrimitiveType, Schema, SortOrder, Struct, TableMetadata, Transform, Type,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};

fn interop_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_EVOLVED_SPEC_REWRITE_DIR").map(PathBuf::from)
}

async fn build_catalog(name: &str, warehouse: &str) -> impl Catalog + use<> {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            name,
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_string())]),
        )
        .await
        .expect("catalog")
}

fn xyz_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "x",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                2,
                "y",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                3,
                "z",
                Type::Primitive(PrimitiveType::Long),
            )),
        ])
        .build()
        .expect("schema")
}

async fn create_xyz_table(
    catalog: &impl Catalog,
    name: &str,
    location: &str,
    format_version: FormatVersion,
) -> Table {
    let schema = xyz_schema();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("x", "x", Transform::Identity)
        .expect("identity(x)")
        .build()
        .expect("spec");
    let namespace = NamespaceIdent::new("interop".to_string());
    let _ = catalog.create_namespace(&namespace, HashMap::new()).await;
    let creation = TableCreation::builder()
        .name(name.to_string())
        .location(location.to_string())
        .schema(schema)
        .partition_spec(spec)
        .sort_order(SortOrder::unsorted_order())
        .format_version(format_version)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

async fn write_xyz(table: &Table, part_x: i64, rows: &[(i64, i64, i64)]) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;
    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("arrow"));
    let xs: Vec<i64> = rows.iter().map(|(x, _, _)| *x).collect();
    let ys: Vec<i64> = rows.iter().map(|(_, y, _)| *y).collect();
    let zs: Vec<i64> = rows.iter().map(|(_, _, z)| *z).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(xs)) as ArrayRef,
        Arc::new(Int64Array::from(ys)) as ArrayRef,
        Arc::new(Int64Array::from(zs)) as ArrayRef,
    ])
    .expect("batch");
    let location_gen = DefaultLocationGenerator::new(table.metadata().clone()).expect("loc");
    let file_name_gen = DefaultFileNameGenerator::new(
        format!("x{part_x}"),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let key = iceberg::spec::PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        schema.clone(),
        Struct::from_iter([Some(Literal::long(part_x))]),
    )
    .expect("key");
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(Some(key))
        .await
        .expect("writer");
    writer.write(batch).await.expect("write");
    writer
        .close()
        .await
        .expect("close")
        .into_iter()
        .next()
        .expect("file")
}

async fn append(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    tx.fast_append()
        .add_data_files(files)
        .apply(tx)
        .expect("apply")
        .commit(catalog)
        .await
        .expect("commit")
}

async fn scan_xyz(table: &Table) -> Vec<(i64, i64, i64)> {
    collect_xyz(table, None).await
}

async fn scan_xyz_eq(table: &Table, column: &str, value: i64) -> Vec<(i64, i64, i64)> {
    collect_xyz(table, Some((column, value))).await
}

async fn collect_xyz(table: &Table, filter: Option<(&str, i64)>) -> Vec<(i64, i64, i64)> {
    let mut scan = table.scan().select(["x", "y", "z"]);
    if let Some((column, value)) = filter {
        scan = scan.with_filter(Reference::new(column).equal_to(Datum::long(value)));
    }
    let stream = scan.build().expect("scan").to_arrow().await.expect("arrow");
    let batches: Vec<RecordBatch> = stream.try_collect().await.expect("collect");
    let mut rows = Vec::new();
    for batch in batches {
        let xs = batch
            .column_by_name("x")
            .expect("x")
            .as_primitive::<Int64Type>();
        let ys = batch
            .column_by_name("y")
            .expect("y")
            .as_primitive::<Int64Type>();
        let zs = batch
            .column_by_name("z")
            .expect("z")
            .as_primitive::<Int64Type>();
        for i in 0..batch.num_rows() {
            rows.push((xs.value(i), ys.value(i), zs.value(i)));
        }
    }
    rows.sort_unstable();
    rows
}

async fn scan_row_ids(table: &Table) -> Vec<i64> {
    let stream = table
        .scan()
        .select(["y", RESERVED_COL_NAME_ROW_ID])
        .build()
        .expect("scan")
        .to_arrow()
        .await
        .expect("arrow");
    let batches: Vec<RecordBatch> = stream.try_collect().await.expect("collect");
    let mut ids = Vec::new();
    for batch in batches {
        let col = batch
            .column_by_name(RESERVED_COL_NAME_ROW_ID)
            .expect("_row_id")
            .as_primitive::<Int64Type>();
        for i in 0..batch.num_rows() {
            assert!(col.is_valid(i));
            ids.push(col.value(i));
        }
    }
    ids.sort_unstable();
    ids
}

async fn live_data_files(table: &Table) -> Vec<DataFile> {
    let snapshot = table.metadata().current_snapshot().expect("snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("list");
    let mut files = Vec::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.content_type() == DataContentType::Data {
                files.push(entry.data_file().clone());
            }
        }
    }
    files
}

fn assert_current_spec_tuples(table: &Table, files: &[DataFile], expected_y: &[i64]) {
    let spec_id = table.metadata().default_partition_spec().spec_id();
    let mut got: Vec<i64> = files
        .iter()
        .map(|file| {
            assert_eq!(file.partition_spec_id(), spec_id);
            match file.partition().fields().first().and_then(|f| f.as_ref()) {
                Some(Literal::Primitive(iceberg::spec::PrimitiveLiteral::Long(v))) => *v,
                other => panic!("expected long tuple, got {other:?}"),
            }
        })
        .collect();
    got.sort_unstable();
    let mut want = expected_y.to_vec();
    want.sort_unstable();
    assert_eq!(got, want);
}

async fn compact_evolved(catalog: &impl Catalog, table: Table) -> Table {
    let ident = table.identifier().clone();
    let tx = Transaction::new(&table);
    let table = tx
        .update_partition_spec()
        .remove_field("x")
        .add_field("y")
        .apply(tx)
        .expect("apply spec")
        .commit(catalog)
        .await
        .expect("commit spec");
    let result = RewriteDataFiles::new(table)
        .target_file_size_bytes(1_000_000)
        .min_input_files(2)
        .execute(catalog)
        .await
        .expect("compact");
    assert!(
        result.rewritten_data_files_count >= 2,
        "rewrite must replace the two old-spec files"
    );
    catalog.load_table(&ident).await.expect("reload")
}

async fn write_final_async(table: &Table, path: &str) {
    table
        .metadata()
        .clone()
        .write_to(table.file_io(), path)
        .await
        .expect("write final metadata");
}

fn load_metadata_table(meta: &Path, ident: &str) -> Table {
    let json =
        std::fs::read_to_string(meta).unwrap_or_else(|e| panic!("read {}: {e}", meta.display()));
    let metadata: TableMetadata =
        serde_json::from_str(&json).unwrap_or_else(|e| panic!("parse {}: {e}", meta.display()));
    Table::builder()
        .metadata(metadata)
        .metadata_location(meta.to_string_lossy().to_string())
        .identifier(TableIdent::from_strs(["interop", ident]).expect("ident"))
        .file_io(FileIO::new_with_fs())
        .build()
        .expect("build table")
}

#[tokio::test]
async fn rust_compacts_java_d1_and_writes_d2_and_v3() {
    let Some(dir) = interop_dir() else {
        println!(
            "skipping interop_evolved_spec_rewrite GEN — set ICEBERG_INTEROP_EVOLVED_SPEC_REWRITE_DIR"
        );
        return;
    };

    let warehouse = dir.to_string_lossy().to_string();
    let catalog = build_catalog("evolved_spec_rewrite", &warehouse).await;
    let namespace = NamespaceIdent::new("interop".to_string());
    let _ = catalog.create_namespace(&namespace, HashMap::new()).await;

    let d1_meta = dir.join("d1/table/metadata/final.metadata.json");
    assert!(
        d1_meta.exists(),
        "missing Java D1 table at {} — Java generate must run first",
        d1_meta.display()
    );
    let d1_reg = dir.join(format!(
        "d1/table/metadata/99999-{}.metadata.json",
        uuid::Uuid::now_v7()
    ));
    std::fs::copy(&d1_meta, &d1_reg).expect("copy d1 metadata");
    let d1 = catalog
        .register_table(
            &TableIdent::new(namespace.clone(), "d1".to_string()),
            d1_reg.to_string_lossy().to_string(),
        )
        .await
        .expect("register d1");
    assert_eq!(scan_xyz(&d1).await, vec![(1, 10, 100), (2, 20, 200)]);
    let d1 = compact_evolved(&catalog, d1).await;
    assert_eq!(scan_xyz(&d1).await, vec![(1, 10, 100), (2, 20, 200)]);
    assert_eq!(scan_xyz_eq(&d1, "y", 10).await, vec![(1, 10, 100)]);
    assert_eq!(scan_xyz_eq(&d1, "y", 20).await, vec![(2, 20, 200)]);
    let d1_files = live_data_files(&d1).await;
    assert_current_spec_tuples(&d1, &d1_files, &[10, 20]);
    let d1_out = dir.join("d1/compacted/metadata");
    std::fs::create_dir_all(&d1_out).expect("d1 out");
    write_final_async(&d1, &d1_out.join("final.metadata.json").to_string_lossy()).await;

    let d2_loc = format!("{warehouse}/d2/rust_table");
    let d2 = create_xyz_table(&catalog, "d2", &d2_loc, FormatVersion::V2).await;
    let a = write_xyz(&d2, 1, &[(1, 10, 100)]).await;
    let b = write_xyz(&d2, 2, &[(2, 20, 200)]).await;
    let d2 = append(&catalog, &d2, vec![a, b]).await;
    write_final_async(&d2, &format!("{d2_loc}/metadata/final.metadata.json")).await;

    let v3_loc = format!("{warehouse}/v3/rust_table");
    let v3 = create_xyz_table(&catalog, "v3", &v3_loc, FormatVersion::V3).await;
    let a = write_xyz(&v3, 1, &[(1, 10, 100)]).await;
    let v3 = append(&catalog, &v3, vec![a]).await;
    let b = write_xyz(&v3, 2, &[(2, 20, 200)]).await;
    let v3 = append(&catalog, &v3, vec![b]).await;
    let before_ids = scan_row_ids(&v3).await;
    assert_eq!(before_ids.len(), 2);
    let v3 = compact_evolved(&catalog, v3).await;
    let after_ids = scan_row_ids(&v3).await;
    assert_eq!(after_ids, before_ids);
    let v3_out = dir.join("v3/compacted/metadata");
    std::fs::create_dir_all(&v3_out).expect("v3 out");
    write_final_async(&v3, &v3_out.join("final.metadata.json").to_string_lossy()).await;
    std::fs::write(
        dir.join("v3/compacted/expected_row_ids.json"),
        serde_json::to_string(&before_ids).expect("json"),
    )
    .expect("write expected row ids");
}

#[tokio::test]
async fn rust_reads_java_d2_rewritten_table() {
    let Some(dir) = interop_dir() else {
        println!(
            "skipping interop_evolved_spec_rewrite D2 — set ICEBERG_INTEROP_EVOLVED_SPEC_REWRITE_DIR"
        );
        return;
    };
    let meta = dir.join("d2/rewritten/metadata/final.metadata.json");
    if !meta.exists() {
        println!(
            "skipping D2 verify — missing {} (Java rewrite step must run first)",
            meta.display()
        );
        return;
    }
    let table = load_metadata_table(&meta, "d2");
    assert_eq!(scan_xyz(&table).await, vec![(1, 10, 100), (2, 20, 200)]);
    assert_eq!(scan_xyz_eq(&table, "y", 10).await, vec![(1, 10, 100)]);
    assert_eq!(scan_xyz_eq(&table, "y", 20).await, vec![(2, 20, 200)]);
    let files = live_data_files(&table).await;
    assert_current_spec_tuples(&table, &files, &[10, 20]);
}
