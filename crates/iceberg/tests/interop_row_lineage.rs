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

//! V3 row-lineage interop for GAP_MATRIX row R166: `first_row_id` assignment and inheritance,
//! compared with Java in both directions over one canonical view. D2 renders a single table
//! twice, so only the cross-check against Java's INDEPENDENT assignment can fail on a
//! wrong-but-consistent writer. Fixture, mutations and what stays unpinned:
//! `dev/java-interop/run-interop-row-lineage.sh`. Each test no-ops with its env var unset.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::io::{FileIO, LocalFsStorageFactory};
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::metadata_columns::{
    RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER, RESERVED_COL_NAME_ROW_ID,
};
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, ManifestStatus, NestedField,
    PrimitiveType, Schema, SortOrder, TableMetadata, Type,
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

/// The env var naming D1's fixture directory, written by the Java oracle.
const D1_ENV: &str = "ICEBERG_INTEROP_ROW_LINEAGE_DIR";

/// Read an env var, treating an empty value as unset (the repo's interop convention).
fn fixture_dir(var: &str) -> Option<PathBuf> {
    match std::env::var(var) {
        Ok(value) if !value.trim().is_empty() => Some(PathBuf::from(value)),
        _ => None,
    }
}

/// Load a table from a `final.metadata.json` written by either side.
fn load_table(metadata_path: &Path, name: &str) -> Table {
    let json = fs::read_to_string(metadata_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", metadata_path.display()));
    let metadata: TableMetadata = serde_json::from_str(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", metadata_path.display()));
    Table::builder()
        .metadata(metadata)
        .metadata_location(metadata_path.to_string_lossy().to_string())
        .identifier(TableIdent::from_strs(["interop", name]).expect("valid identifier"))
        .file_io(FileIO::new_with_fs())
        .build()
        .expect("build table from final.metadata.json")
}

/// Render `Option<u64>` the way the oracle does, so the two views are byte-comparable.
fn opt(value: Option<u64>) -> String {
    value.map_or_else(|| "null".to_string(), |v| v.to_string())
}

/// The canonical lineage view. Field order and spacing mirror `RowLineageOracle.lineageJson`; the
/// two lists are sorted as RENDERED STRINGS on both sides, so identical content sorts identically
/// without pinning either side's manifest traversal order.
pub async fn lineage_view(table: &Table) -> String {
    let metadata = table.metadata();
    let file_io = table.file_io();
    let snapshot = metadata
        .current_snapshot()
        .expect("a committed table has a current snapshot");

    let mut manifests: Vec<String> = Vec::new();
    let mut files: Vec<String> = Vec::new();
    let manifest_list = snapshot
        .load_manifest_list(file_io, metadata)
        .await
        .expect("load manifest list");
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != iceberg::spec::ManifestContentType::Data {
            continue;
        }
        manifests.push(format!(
            "{{\"first_row_id\":{},\"added_files\":{}}}",
            opt(manifest_file.first_row_id),
            opt(manifest_file.added_files_count.map(u64::from))
        ));
        let manifest = manifest_file
            .load_manifest(file_io)
            .await
            .expect("load manifest");
        for entry in manifest.entries() {
            if entry.status() == ManifestStatus::Deleted
                || entry.data_file().content_type() != DataContentType::Data
            {
                continue;
            }
            let data_file = entry.data_file();
            let name = Path::new(data_file.file_path())
                .file_name()
                .expect("data file has a name")
                .to_string_lossy()
                .to_string();
            files.push(format!(
                "{{\"file\":\"{}\",\"first_row_id\":{},\"record_count\":{}}}",
                name,
                data_file
                    .first_row_id()
                    .map_or_else(|| "null".to_string(), |v| v.to_string()),
                data_file.record_count()
            ));
        }
    }
    // Java sorts the same rendered strings with `Collections.sort`. UTF-16 code-unit order and
    // UTF-8 byte order agree on ASCII, which is all these strings carry.
    manifests.sort();
    files.sort();

    format!(
        "{{\"format_version\":{},\"next_row_id\":{},\"snapshot_first_row_id\":{},\
         \"snapshot_added_rows\":{},\"manifests\":[{}],\"files\":[{}]}}",
        metadata.format_version() as u8,
        metadata.next_row_id(),
        opt(snapshot.first_row_id()),
        opt(snapshot.added_rows_count()),
        manifests.join(","),
        files.join(",")
    )
}

/// D1 — the fork's view of a JAVA-written V3 table must equal Java's own.
#[tokio::test]
async fn rust_reads_java_assigned_row_lineage() {
    let Some(dir) = fixture_dir(D1_ENV) else {
        eprintln!("{D1_ENV} unset — skipping (run dev/java-interop/run-interop-row-lineage.sh)");
        return;
    };

    let table = load_table(
        &dir.join("table/metadata/final.metadata.json"),
        "row_lineage",
    );
    let rust_view = lineage_view(&table).await;
    let java_view = fs::read_to_string(dir.join("java_row_lineage.json"))
        .expect("read java_row_lineage.json")
        .trim()
        .to_string();

    assert_eq!(
        rust_view, java_view,
        "the fork's row-lineage view of a JAVA-written V3 table differs from Java's own.\n \
         rust: {rust_view}\n java: {java_view}"
    );
}

// ---- D2: "JAVA reads what RUST writes" ---------------------------------------------------------

/// The env var naming D2's output directory, consumed by `verify-interop-row-lineage`.
const D2_ENV: &str = "ICEBERG_INTEROP_ROW_LINEAGE_WRITE_DIR";

/// The Java fixture's schema, so both sides commit the same logical table.
fn row_lineage_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("row-lineage schema")
}

/// Write one real parquet data file through the production writer.
///
/// `with_partition_spec` is called even though the table is unpartitioned: it is the call pattern
/// R114 bound (c) tells an external engine to follow, and building with neither a spec nor a key
/// reaches the arm that stamps a spec id by default.
async fn write_data_file(table: &Table, ids: Vec<i64>, values: Vec<&str>) -> DataFile {
    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(values)) as ArrayRef,
    ])
    .expect("build the data batch");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "rust-data".to_string(),
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

    let mut writer = DataFileWriterBuilder::new(rolling)
        .with_partition_spec(table.metadata().default_partition_spec().as_ref().clone())
        .build(None)
        .await
        .expect("build data file writer");
    writer.write(batch).await.expect("write batch");
    writer
        .close()
        .await
        .expect("close data file writer")
        .into_iter()
        .next()
        .expect("one data file per close")
}

/// D2 GEN — commit a V3 table whose row lineage the fork assigned, for Java to read back.
///
/// The commit shape mirrors the Java fixture exactly: two files in ONE `fast_append` (the counter
/// must advance within the manifest), then a third in a SECOND commit (it must continue across
/// snapshots).
#[tokio::test]
async fn row_lineage_write_gen() {
    let Some(dir) = fixture_dir(D2_ENV) else {
        eprintln!("{D2_ENV} unset — skipping (run dev/java-interop/run-interop-row-lineage.sh)");
        return;
    };
    fs::create_dir_all(&dir).expect("create the gen dir");

    let table_location = format!("{}/rust_table", dir.to_string_lossy());
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                dir.to_string_lossy().to_string(),
            )]),
        )
        .await
        .expect("build MemoryCatalog over local FS");

    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location.clone())
        .schema(row_lineage_schema())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V3)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create V3 rust_table");

    let file_a = write_data_file(&table, vec![10, 20, 30, 40, 50], vec![
        "a", "b", "c", "d", "e",
    ])
    .await;
    let file_b = write_data_file(&table, vec![60, 70, 80], vec!["f", "g", "h"]).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file_a, file_b])
        .apply(tx)
        .expect("apply the first fast append");
    let table = tx.commit(&catalog).await.expect("commit the first append");

    let file_c = write_data_file(&table, vec![90, 100, 110, 120], vec!["i", "j", "k", "l"]).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file_c])
        .apply(tx)
        .expect("apply the second fast append");
    let table = tx.commit(&catalog).await.expect("commit the second append");

    let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), &final_metadata_path)
        .await
        .expect("write final.metadata.json");

    let view = lineage_view(&table).await;
    fs::write(dir.join("rust_row_lineage_expected.json"), &view)
        .expect("write rust_row_lineage_expected.json");
    println!("interop_row_lineage GEN OK — {table_location}\n  rust view: {view}");
}

/// The cross-check that closes D2's circularity.
///
/// D2 proves Java can READ what the fork wrote, because both sides render the same table. It
/// cannot prove the fork ASSIGNS what Java assigns — a wrong-but-consistent writer passes it. This
/// compares the two INDEPENDENTLY produced views of the SAME logical chain (two files in one
/// append, then a third) with the file names stripped, so only the lineage numbers remain: the
/// metadata counter, the snapshot range, each manifest's range, and each file's inherited id.
#[tokio::test]
async fn rust_assigns_the_same_row_ids_java_does() {
    let (Some(d1), Some(d2)) = (fixture_dir(D1_ENV), fixture_dir(D2_ENV)) else {
        eprintln!("{D1_ENV}/{D2_ENV} unset — skipping");
        return;
    };

    // The file NAME is the only field that legitimately differs: Java names by ordinal, the fork
    // by uuid. Everything else is the contract under test.
    let strip_names = |view: &str| -> String {
        let mut out = String::with_capacity(view.len());
        let mut rest = view;
        while let Some(start) = rest.find("\"file\":\"") {
            out.push_str(&rest[..start + "\"file\":\"".len()]);
            rest = &rest[start + "\"file\":\"".len()..];
            let end = rest.find('"').expect("a closing quote on the file name");
            rest = &rest[end..];
        }
        out.push_str(rest);
        out
    };

    let java = fs::read_to_string(d1.join("java_row_lineage.json")).expect("read the Java view");
    let rust =
        fs::read_to_string(d2.join("rust_row_lineage_expected.json")).expect("read the Rust view");
    // Strip BEFORE any ordering matters: the lists arrive sorted by the file name, which this
    // erases, so comparing them as-is would rest on the fork's uuid names happening to sort in
    // creation order.
    let java_numbers = strip_names(java.trim());
    let rust_numbers = strip_names(rust.trim());

    assert_eq!(
        rust_numbers, java_numbers,
        "the fork ASSIGNED different row ids than Java did for the same logical chain.\n \
         rust: {rust_numbers}\n java: {java_numbers}"
    );
}

// ---- MATERIALIZATION: the per-row half of R166 ------------------------------------------------

/// Per-row `(id, _row_id, _last_updated_sequence_number)` from a REAL scan, rendered like the
/// oracle's `rowIdsJson` and sorted the same way.
///
/// The manifest view cannot reach this: inheritance is metadata, but `_row_id` is resolved PER ROW
/// by the reader — the stored value, else `first_row_id + pos`.
async fn row_ids_view(table: &Table) -> String {
    use futures::TryStreamExt;

    let batches: Vec<RecordBatch> = table
        .scan()
        .select([
            "id",
            RESERVED_COL_NAME_ROW_ID,
            RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER,
        ])
        .build()
        .expect("build the row-lineage scan")
        .to_arrow()
        .await
        .expect("scan to arrow")
        .try_collect()
        .await
        .expect("collect batches");

    let mut rows: Vec<String> = Vec::new();
    for batch in &batches {
        let ids = batch.column(0).as_primitive::<Int64Type>();
        let row_ids = batch.column(1).as_primitive::<Int64Type>();
        let seqs = batch.column(2).as_primitive::<Int64Type>();
        for row in 0..batch.num_rows() {
            let render = |array: &arrow_array::PrimitiveArray<Int64Type>| {
                if array.is_null(row) {
                    "null".to_string()
                } else {
                    array.value(row).to_string()
                }
            };
            rows.push(format!(
                "{{\"id\":{},\"row_id\":{},\"last_updated_seq\":{}}}",
                ids.value(row),
                render(row_ids),
                render(seqs)
            ));
        }
    }
    rows.sort();
    format!("[{}]", rows.join(","))
}

/// D1 materialization — the fork's per-row `_row_id` over a JAVA-written table must equal Java's.
#[tokio::test]
async fn rust_materializes_java_row_ids() {
    let Some(dir) = fixture_dir(D1_ENV) else {
        eprintln!("{D1_ENV} unset — skipping");
        return;
    };
    let table = load_table(
        &dir.join("table/metadata/final.metadata.json"),
        "row_lineage",
    );
    let rust_rows = row_ids_view(&table).await;
    let java_rows = fs::read_to_string(dir.join("java_row_ids.json"))
        .expect("read java_row_ids.json")
        .trim()
        .to_string();
    assert_eq!(
        rust_rows, java_rows,
        "the fork's per-row _row_id over a JAVA table differs from Java's own read.\n \
         rust: {rust_rows}\n java: {java_rows}"
    );
}

/// D2 materialization — Java's per-row read of the RUST table must equal the fork's own.
#[tokio::test]
async fn java_materializes_rust_row_ids() {
    let Some(dir) = fixture_dir(D2_ENV) else {
        eprintln!("{D2_ENV} unset — skipping");
        return;
    };
    let java_path = dir.join("java_row_ids_of_rust_table.json");
    let Ok(java_rows) = fs::read_to_string(&java_path) else {
        eprintln!(
            "{} absent — run the oracle's verify first",
            java_path.display()
        );
        return;
    };
    let table = load_table(
        &dir.join("rust_table/metadata/final.metadata.json"),
        "rust_row_lineage",
    );
    let rust_rows = row_ids_view(&table).await;
    assert_eq!(
        rust_rows,
        java_rows.trim(),
        "Java's per-row read of the RUST-written table differs from the fork's own.\n \
         rust: {rust_rows}\n java: {}",
        java_rows.trim()
    );
}
