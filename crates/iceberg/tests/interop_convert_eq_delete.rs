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

//! MAINTENANCE `ConvertEqualityDeleteFiles` interop — the eq→pos delete-conversion action proven against
//! Java's OWN merge-on-read read WITHOUT Spark (the real Java action is a Spark-surface class NOT on the
//! iceberg-core oracle classpath, and Java cannot DRIVE the conversion). The proof is therefore the
//! corruption-class READ-IDENTITY claim, in the GEN direction only:
//!
//! - **Rust GEN (the only direction):** Rust writes a PRE table (data + a real EQUALITY-delete masking a
//!   known subset) to `<gen_dir>/rust_table`, runs `ConvertEqualityDeleteFiles`, reloads, and writes the
//!   POST table (the same rows now masked by a POSITION-delete) to `<gen_dir>/rust_table_converted`.
//!   Java's `verify-interop-convert-eq-delete` then loads BOTH tables, reads each via `IcebergGenerics`
//!   (which applies whichever delete the table carries), and asserts the live row sets are IDENTICAL —
//!   AND that the PRE table carried an EQUALITY delete while the POST table carries a POSITION delete (so
//!   the conversion genuinely converted, not merely no-op'd). This is the no-Spark corroboration that the
//!   converted position delete masks EXACTLY the rows the equality delete masked.
//!
//! ANTI-CIRCULAR: the masked subset + the expected live set are hand-declared HERE and INDEPENDENTLY in
//! the Java oracle from the fixture definition, never from the other engine's output.
//!
//! GATED on `ICEBERG_INTEROP_CONVERT_EQ_DELETE_GEN_DIR` (unset ⇒ a clean no-op; the offline `cargo test`
//! gate stays green). `dev/java-interop/run-interop-convert-eq-delete.sh` is the driver.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use iceberg::io::LocalFsStorageFactory;
use iceberg::maintenance::ConvertEqualityDeleteFiles;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, Literal, ManifestContentType,
    NestedField, PartitionKey, PartitionSpec, PrimitiveType, Schema, SortOrder, Struct, Transform,
    Type,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::equality_delete_writer::{
    EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
};
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};

/// Schema `{1 id long, 2 cat string, 3 y long}`, spec 0 `identity(cat)` — mirrors the Java oracle.
fn convert_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                2,
                "cat",
                Type::Primitive(PrimitiveType::String),
            )),
            Arc::new(NestedField::required(
                3,
                "y",
                Type::Primitive(PrimitiveType::Long),
            )),
        ])
        .build()
        .expect("build schema")
}

/// The hand-declared live `id` set before AND after conversion: the eq-delete (y=20) masks id=120 and
/// id=220; everything else lives.
fn expected_live_ids() -> HashSet<i64> {
    HashSet::from([100, 130, 200, 230])
}

fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_CONVERT_EQ_DELETE_GEN_DIR").map(PathBuf::from)
}

async fn build_catalog(name: &str, warehouse: &str) -> impl Catalog + use<> {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            name,
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_string())]),
        )
        .await
        .expect("build MemoryCatalog over local FS")
}

async fn create_table(catalog: &impl Catalog, table_name: &str, table_location: &str) -> Table {
    let schema = convert_schema();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("cat", "cat", Transform::Identity)
        .expect("add identity(cat) partition field")
        .build()
        .expect("build identity(cat) spec");
    let namespace = NamespaceIdent::new("interop".to_string());
    let _ = catalog.create_namespace(&namespace, HashMap::new()).await;
    let creation = TableCreation::builder()
        .name(table_name.to_string())
        .location(table_location.to_string())
        .schema(schema)
        .partition_spec(spec)
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

fn partition_key_for(table: &Table, cat: &str) -> PartitionKey {
    PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        Struct::from_iter([Some(Literal::string(cat))]),
    )
}

async fn write_data_file(table: &Table, cat: &str, rows: &[(i64, i64)]) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));
    let ids: Vec<i64> = rows.iter().map(|(id, _)| *id).collect();
    let cats: Vec<String> = rows.iter().map(|_| cat.to_string()).collect();
    let ys: Vec<i64> = rows.iter().map(|(_, y)| *y).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(cats)) as ArrayRef,
        Arc::new(Int64Array::from(ys)) as ArrayRef,
    ])
    .expect("build {id, cat, y} batch");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        format!("data-{cat}"),
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
    let mut writer =
        iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder::new(rolling)
            .build(Some(partition_key_for(table, cat)))
            .await
            .expect("build data file writer");
    writer.write(batch).await.expect("write data batch");
    writer
        .close()
        .await
        .expect("close data writer")
        .into_iter()
        .next()
        .expect("one data file")
}

async fn write_equality_delete(table: &Table, cat: &str, delete_ys: &[i64]) -> DataFile {
    use iceberg::arrow::{arrow_schema_to_schema, schema_to_arrow_schema};

    let schema = table.metadata().current_schema().clone();
    let config =
        EqualityDeleteWriterConfig::new(vec![3], schema.clone()).expect("eq-delete config");
    let delete_schema =
        Arc::new(arrow_schema_to_schema(config.projected_arrow_schema_ref()).expect("eq schema"));

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        format!("eqdel-{cat}"),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        delete_schema,
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .build(Some(partition_key_for(table, cat)))
        .await
        .expect("build eq-delete writer");

    let arrow_schema = Arc::new(schema_to_arrow_schema(&schema).expect("schema → arrow"));
    let ids: Vec<i64> = delete_ys.iter().map(|_| 0).collect();
    let cats: Vec<String> = delete_ys.iter().map(|_| cat.to_string()).collect();
    let ys: Vec<i64> = delete_ys.to_vec();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(cats)) as ArrayRef,
        Arc::new(Int64Array::from(ys)) as ArrayRef,
    ])
    .expect("build eq-delete batch");
    writer.write(batch).await.expect("write eq-delete batch");
    writer
        .close()
        .await
        .expect("close eq-delete writer")
        .into_iter()
        .next()
        .expect("one eq-delete file")
}

async fn fast_append(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(files)
        .apply(tx)
        .expect("apply fast_append");
    tx.commit(catalog).await.expect("commit fast_append")
}

async fn add_deletes(catalog: &impl Catalog, table: &Table, deletes: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .row_delta()
        .add_deletes(deletes)
        .apply(tx)
        .expect("apply row_delta");
    tx.commit(catalog).await.expect("commit row_delta")
}

/// Build the PRE table: two partitions (cat=A id 100/120/130, cat=B id 200/220/230) at seq 1; one
/// equality delete per partition deleting y=20 (so id=120 and id=220 are masked) at seq 2.
async fn build_pre_world(catalog: &impl Catalog, table: Table) -> Table {
    let a = write_data_file(&table, "A", &[(100, 10), (120, 20), (130, 30)]).await;
    let b = write_data_file(&table, "B", &[(200, 10), (220, 20), (230, 30)]).await;
    let table = fast_append(catalog, &table, vec![a, b]).await;

    let eq_a = write_equality_delete(&table, "A", &[20]).await;
    let eq_b = write_equality_delete(&table, "B", &[20]).await;
    add_deletes(catalog, &table, vec![eq_a, eq_b]).await
}

/// The merge-on-read live `id` set.
async fn scan_ids(table: &Table) -> HashSet<i64> {
    let stream = table
        .scan()
        .select(["id"])
        .build()
        .expect("scan build")
        .to_arrow()
        .await
        .expect("to_arrow");
    let batches: Vec<RecordBatch> = stream.try_collect().await.expect("collect");
    let mut ids = HashSet::new();
    for batch in batches {
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("id is int64");
        for i in 0..col.len() {
            ids.insert(col.value(i));
        }
    }
    ids
}

/// The count of live delete files of `content` in the current snapshot.
async fn live_delete_count(table: &Table, content: DataContentType) -> usize {
    let snapshot = table.metadata().current_snapshot().expect("snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    let mut count = 0;
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.content_type() == content {
                count += 1;
            }
        }
    }
    count
}

// =================================================================================================
// Rust GEN — write PRE (eq-delete) + POST (converted pos-delete) tables for Java to verify.
// =================================================================================================

#[tokio::test]
async fn test_convert_eq_delete_gen() {
    let Some(gen_dir) = gen_dir() else {
        println!(
            "skipping interop_convert_eq_delete GEN — set ICEBERG_INTEROP_CONVERT_EQ_DELETE_GEN_DIR \
             (run dev/java-interop/run-interop-convert-eq-delete.sh)"
        );
        return;
    };

    let warehouse = gen_dir.to_string_lossy().to_string();
    let pre_location = format!("{warehouse}/rust_table");
    let converted_location = format!("{warehouse}/rust_table_converted");
    let catalog = build_catalog("interop_convert_eq_delete_gen", &warehouse).await;

    // PRE table (eq-delete). Land its final metadata BEFORE the action.
    let pre = create_table(&catalog, "rust_table", &pre_location).await;
    let pre = build_pre_world(&catalog, pre).await;
    let pre_final = format!("{pre_location}/metadata/final.metadata.json");
    pre.metadata()
        .clone()
        .write_to(pre.file_io(), &pre_final)
        .await
        .expect("write pre final.metadata.json");

    let pre_ids = scan_ids(&pre).await;
    assert_eq!(
        pre_ids,
        expected_live_ids(),
        "GEN sanity: PRE live ids must equal the hand-declared set (eq-delete masks id 120/220)"
    );
    assert_eq!(
        live_delete_count(&pre, DataContentType::EqualityDeletes).await,
        2,
        "GEN sanity: PRE table must carry the two equality deletes"
    );

    // Build a SEPARATE converted table so Java can read BOTH the PRE (eq) and POST (pos) tables.
    let converted = create_table(&catalog, "rust_table_converted", &converted_location).await;
    let converted = build_pre_world(&catalog, converted).await;
    let result = ConvertEqualityDeleteFiles::new(converted.clone())
        .execute(&catalog)
        .await
        .expect("run ConvertEqualityDeleteFiles");
    assert_eq!(
        result.converted_equality_delete_files_count, 2,
        "GEN sanity: both equality deletes converted"
    );
    assert_eq!(
        result.added_position_delete_files_count, 2,
        "GEN sanity: two position deletes added"
    );

    let converted = catalog
        .load_table(converted.identifier())
        .await
        .expect("reload converted table");
    let converted_final = format!("{converted_location}/metadata/final.metadata.json");
    converted
        .metadata()
        .clone()
        .write_to(converted.file_io(), &converted_final)
        .await
        .expect("write converted final.metadata.json");

    // Read-identity sanity (Rust side): converted live ids == pre live ids; eq gone, pos present.
    assert_eq!(
        scan_ids(&converted).await,
        pre_ids,
        "GEN sanity: read-identity — converted live ids must equal pre live ids"
    );
    assert_eq!(
        live_delete_count(&converted, DataContentType::EqualityDeletes).await,
        0,
        "GEN sanity: POST table must carry NO equality deletes"
    );
    assert_eq!(
        live_delete_count(&converted, DataContentType::PositionDeletes).await,
        2,
        "GEN sanity: POST table must carry the two converted position deletes"
    );

    println!(
        "interop_convert_eq_delete GEN OK — wrote {pre_location} (eq) + {converted_location} \
         (pos); converted 2 eq → 2 pos; read-identity holds (live ids {pre_ids:?})"
    );
}
