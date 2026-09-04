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

//! GEN-direction interop for [`RewritePositionDeleteFiles`]. Java has no core runner; the claim
//! is read identity. PRE and POST live ids are declared here and in the Java oracle, never taken
//! from the other engine. The V3 leg repeats the pair after conversion to Puffin DVs.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use iceberg::io::LocalFsStorageFactory;
use iceberg::maintenance::RewritePositionDeleteFiles;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, Literal, ManifestContentType,
    NestedField, PartitionKey, PartitionSpec, PrimitiveType, Schema, SortOrder, Struct, Transform,
    Type,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
};
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};

/// Schema `{1 id long, 2 cat string, 3 y long}`, spec 0 `identity(cat)` — mirrors the Java oracle.
fn rewrite_schema() -> Schema {
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

/// Live `id` set before and after: the two pos-deletes mask 120 and 220.
fn expected_live_ids() -> HashSet<i64> {
    HashSet::from([100, 130, 200, 230])
}

fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_REWRITE_POS_DELETES_GEN_DIR").map(PathBuf::from)
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
    let schema = rewrite_schema();
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
    .expect("PartitionKey::new: valid partition tuple")
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

/// Write a parquet position-delete file in partition `cat`.
async fn write_position_delete(table: &Table, cat: &str, pairs: &[(&str, i64)]) -> DataFile {
    let config = PositionDeleteWriterConfig::new().expect("pos-delete config");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        format!("posdel-{cat}"),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        config.schema().clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(Some(partition_key_for(table, cat)))
        .await
        .expect("build pos-delete writer");

    let paths: Vec<&str> = pairs.iter().map(|(p, _)| *p).collect();
    let positions: Vec<i64> = pairs.iter().map(|(_, p)| *p).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions)) as ArrayRef,
    ])
    .expect("build pos-delete batch");
    writer.write(batch).await.expect("write pos-delete batch");
    writer
        .close()
        .await
        .expect("close pos-delete writer")
        .into_iter()
        .next()
        .expect("one pos-delete file")
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

/// PRE world: two partitions, then two position deletes per partition in separate commits.
async fn build_pre_world(catalog: &impl Catalog, table: Table) -> (Table, String, String) {
    let a = write_data_file(&table, "A", &[(100, 10), (120, 20), (130, 30)]).await;
    let b = write_data_file(&table, "B", &[(200, 10), (220, 20), (230, 30)]).await;
    let a_path = a.file_path().to_string();
    let b_path = b.file_path().to_string();
    let table = fast_append(catalog, &table, vec![a, b]).await;

    // First pos-delete per partition (seq 2): mask id=120 (cat=A pos 1) and id=220 (cat=B pos 1).
    let pd_a1 = write_position_delete(&table, "A", &[(&a_path, 1)]).await;
    let pd_b1 = write_position_delete(&table, "B", &[(&b_path, 1)]).await;
    let table = add_deletes(catalog, &table, vec![pd_a1, pd_b1]).await;

    // Duplicate the mask. Java does not dedup; the reader bitmap does.
    let pd_a2 = write_position_delete(&table, "A", &[(&a_path, 1)]).await;
    let pd_b2 = write_position_delete(&table, "B", &[(&b_path, 1)]).await;
    let table = add_deletes(catalog, &table, vec![pd_a2, pd_b2]).await;
    (table, a_path, b_path)
}

async fn top_up_to_floor(
    catalog: &impl Catalog,
    table: Table,
    a_path: &str,
    b_path: &str,
) -> Table {
    let mut extra = Vec::new();
    for _ in 0..3 {
        extra.push(write_position_delete(&table, "A", &[(a_path, 1)]).await);
        extra.push(write_position_delete(&table, "B", &[(b_path, 1)]).await);
    }
    add_deletes(catalog, &table, extra).await
}

/// Upgrade to V3, leaving parquet position deletes the table can no longer write.
async fn upgrade_to_v3(catalog: &impl Catalog, table: &Table) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .upgrade_table_version()
        .set_format_version(FormatVersion::V3)
        .apply(tx)
        .expect("apply upgrade_table_version");
    tx.commit(catalog)
        .await
        .expect("commit upgrade_table_version")
}

/// The count of live delete files of `content` in `format`.
async fn live_delete_count_of_format(
    table: &Table,
    content: DataContentType,
    format: DataFileFormat,
) -> usize {
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
            if entry.is_alive()
                && entry.content_type() == content
                && entry.data_file().file_format() == format
            {
                count += 1;
            }
        }
    }
    count
}

/// Write V3 PRE (legacy parquet deletes) and POST (converted Puffin DVs).
async fn write_v3_pair(catalog: &impl Catalog, warehouse: &str, expected_ids: &HashSet<i64>) {
    let pre_location = format!("{warehouse}/rust_table_v3");
    let pre = create_table(catalog, "rust_table_v3", &pre_location).await;
    let (pre, a_path, b_path) = build_pre_world(catalog, pre).await;
    let pre = top_up_to_floor(catalog, pre, &a_path, &b_path).await;
    let pre = upgrade_to_v3(catalog, &pre).await;
    pre.metadata()
        .clone()
        .write_to(
            pre.file_io(),
            &format!("{pre_location}/metadata/final.metadata.json"),
        )
        .await
        .expect("write v3 pre final.metadata.json");
    assert_eq!(
        &scan_ids(&pre).await,
        expected_ids,
        "GEN sanity: the upgraded V3 table still masks id 120/220"
    );
    assert_eq!(
        live_delete_count_of_format(
            &pre,
            DataContentType::PositionDeletes,
            DataFileFormat::Parquet
        )
        .await,
        10,
        "GEN sanity: the V3 PRE table carries TEN legacy parquet position deletes"
    );

    let post_location = format!("{warehouse}/rust_table_v3_dv");
    let post = create_table(catalog, "rust_table_v3_dv", &post_location).await;
    let (post, a_path, b_path) = build_pre_world(catalog, post).await;
    let post = top_up_to_floor(catalog, post, &a_path, &b_path).await;
    let post = upgrade_to_v3(catalog, &post).await;
    // Five deletes per partition clear the default min-input-files floor of five.
    let result = RewritePositionDeleteFiles::new(post.clone())
        .execute(catalog)
        .await
        .expect("run RewritePositionDeleteFiles on the V3 table");
    assert_eq!(
        result.rewritten_delete_files_count, 10,
        "GEN sanity: all ten legacy parquet position deletes consumed"
    );
    assert_eq!(
        result.added_delete_files_count, 2,
        "GEN sanity: one deletion vector per referenced data file"
    );

    let post = catalog
        .load_table(post.identifier())
        .await
        .expect("reload v3 post table");
    post.metadata()
        .clone()
        .write_to(
            post.file_io(),
            &format!("{post_location}/metadata/final.metadata.json"),
        )
        .await
        .expect("write v3 post final.metadata.json");
    assert_eq!(
        &scan_ids(&post).await,
        expected_ids,
        "GEN sanity: read identity — the deletion vectors mask exactly the same rows"
    );
    assert_eq!(
        live_delete_count_of_format(
            &post,
            DataContentType::PositionDeletes,
            DataFileFormat::Parquet
        )
        .await,
        0,
        "GEN sanity: no legacy parquet position delete survives on the V3 table"
    );
    assert_eq!(
        live_delete_count_of_format(
            &post,
            DataContentType::PositionDeletes,
            DataFileFormat::Puffin
        )
        .await,
        2,
        "GEN sanity: TWO Puffin deletion vectors, one per referenced data file"
    );
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
// Rust GEN — write PRE (many pos-deletes) + POST (compacted pos-deletes) tables for Java to verify.
// =================================================================================================

#[tokio::test]
async fn test_rewrite_pos_deletes_gen() {
    let Some(gen_dir) = gen_dir() else {
        println!(
            "skipping interop_rewrite_pos_deletes GEN — set ICEBERG_INTEROP_REWRITE_POS_DELETES_GEN_DIR \
             (run dev/java-interop/run-interop-rewrite-pos-deletes.sh)"
        );
        return;
    };

    let warehouse = gen_dir.to_string_lossy().to_string();
    let pre_location = format!("{warehouse}/rust_table");
    let compacted_location = format!("{warehouse}/rust_table_compacted");
    let catalog = build_catalog("interop_rewrite_pos_deletes_gen", &warehouse).await;

    // PRE table (many pos-deletes). Land its final metadata BEFORE the action.
    let pre = create_table(&catalog, "rust_table", &pre_location).await;
    let (pre, _, _) = build_pre_world(&catalog, pre).await;
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
        "GEN sanity: PRE live ids must equal the hand-declared set (pos-deletes mask id 120/220)"
    );
    let pre_pos = live_delete_count(&pre, DataContentType::PositionDeletes).await;
    assert_eq!(
        pre_pos, 4,
        "GEN sanity: PRE table must carry FOUR position-delete files (2 per partition)"
    );

    // Build a SEPARATE compacted table so Java can read BOTH the PRE and POST tables.
    let compacted = create_table(&catalog, "rust_table_compacted", &compacted_location).await;
    let (compacted, _, _) = build_pre_world(&catalog, compacted).await;
    // Floor 2: Java's default 5 declines this two-file bin. 1 would disable the size>=N clause.
    let result = RewritePositionDeleteFiles::new(compacted.clone())
        .min_input_files(2)
        .execute(&catalog)
        .await
        .expect("run RewritePositionDeleteFiles");
    assert_eq!(
        result.rewritten_delete_files_count, 4,
        "GEN sanity: all four pos-delete files rewritten"
    );
    assert_eq!(
        result.added_delete_files_count, 2,
        "GEN sanity: one compacted pos-delete added per partition group"
    );

    let compacted = catalog
        .load_table(compacted.identifier())
        .await
        .expect("reload compacted table");
    let compacted_final = format!("{compacted_location}/metadata/final.metadata.json");
    compacted
        .metadata()
        .clone()
        .write_to(compacted.file_io(), &compacted_final)
        .await
        .expect("write compacted final.metadata.json");

    // Read-identity sanity (Rust side): compacted live ids == pre live ids; fewer pos-delete files.
    assert_eq!(
        scan_ids(&compacted).await,
        pre_ids,
        "GEN sanity: read-identity — compacted live ids must equal pre live ids"
    );
    let post_pos = live_delete_count(&compacted, DataContentType::PositionDeletes).await;
    assert_eq!(
        post_pos, 2,
        "GEN sanity: POST table must carry exactly TWO compacted position-delete files"
    );
    assert!(
        post_pos < pre_pos,
        "GEN sanity: compaction must FUSE files (POST {post_pos} < PRE {pre_pos})"
    );

    // A third table with the same data and no deletes. The shell sabotage battery swaps it for the
    // POST metadata, so the read-identity leg must fail closed. The verify path never reads it.
    let nodeletes_location = format!("{warehouse}/rust_table_nodeletes");
    let nodeletes = create_table(&catalog, "rust_table_nodeletes", &nodeletes_location).await;
    let a = write_data_file(&nodeletes, "A", &[(100, 10), (120, 20), (130, 30)]).await;
    let b = write_data_file(&nodeletes, "B", &[(200, 10), (220, 20), (230, 30)]).await;
    let nodeletes = fast_append(&catalog, &nodeletes, vec![a, b]).await;
    let nodeletes_final = format!("{nodeletes_location}/metadata/final.metadata.json");
    nodeletes
        .metadata()
        .clone()
        .write_to(nodeletes.file_io(), &nodeletes_final)
        .await
        .expect("write nodeletes final.metadata.json");
    let nodeletes_ids = scan_ids(&nodeletes).await;
    assert_eq!(
        nodeletes_ids,
        HashSet::from([100, 120, 130, 200, 220, 230]),
        "GEN sanity: the no-deletes table reads the FULL id set (the sabotage read-identity breaker)"
    );

    // The V3 PAIR — legacy parquet position deletes converted into Puffin deletion vectors.
    write_v3_pair(&catalog, &warehouse, &pre_ids).await;

    println!(
        "interop_rewrite_pos_deletes GEN OK — wrote {pre_location} ({pre_pos} pos-deletes) + \
         {compacted_location} ({post_pos} compacted) + {nodeletes_location} (sabotage breaker); \
         rewrote 4 → added 2; read-identity holds (live ids {pre_ids:?})"
    );
}
