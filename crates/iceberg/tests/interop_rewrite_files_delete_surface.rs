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

//! Bidirectional Java interop for `RewriteFiles` DELETE-file ADD surface (GAP_MATRIX row R152).
//!
//! Proves the two Java seq overloads for added delete files in a `RewriteFiles` commit:
//!
//! - `addFile(DeleteFile)` (Rust `add_delete_file`) → the added delete file **INHERITS** the new
//!   (rewrite) snapshot's data sequence number.
//! - `addFile(DeleteFile, long)` (Rust `add_delete_file_with_sequence_number`) → the added delete
//!   file is **STAMPED** with that explicit data sequence number.
//!
//! **LOCKED FIXTURE** (V2 unpartitioned, deterministic seqs):
//! - S1 `fast_append(D)`        → data file D,   seq 1.
//! - S2 `row_delta(DF0)`        → seed pos-del,  seq 2.
//! - S3 `rewrite_files`: `delete_delete_file(DF0)` + `add_delete_file(DF_inherited)` +
//!   `add_delete_file_with_sequence_number(DF_exp, 2)` → seq 3.
//!   - `DF_inherited`: expected data seq = 3 (== S3 snapshot seq, inherited).
//!   - `DF_exp`: expected data seq = 2 (== explicit stamp, preserved).
//!
//! **DIRECTION 1** (Rust reads Java-written table): loads `<dir>/table/metadata/final.metadata.json`
//! (the Java-generated fixture); enumerates the S3 delete-manifest entries; for `DF_inherited` and
//! `DF_exp` (matched by path suffix from `java_delete_seqs.json`), asserts their
//! `sequence_number` equals both the value in `java_delete_seqs.json` AND the hand-declared
//! expected (inherited == S3 snapshot seq, explicit == 2).
//!
//! **DIRECTION 2** (Rust writes, Java reads): independently builds the SAME locked fixture in Rust
//! (append D; `row_delta` DF0; `rewrite_files` delete DF0 + `add_delete_file(DF_inherited)` +
//! `add_delete_file_with_sequence_number(DF_exp, 2)`); asserts the two added-delete seqs
//! (inherited == S3 snapshot seq, explicit == 2); writes
//! `<dir>/rust_rewritten/metadata/final.metadata.json` for the Java `verify-interop-rfds` step.
//!
//! **NON-VACUITY** is proven by the shell driver, which corrupts `java_delete_seqs.json` (changes
//! inherited seq 3 → 99), re-runs Direction-1 (MUST FAIL), then restores via a fresh Java
//! generate pass.
//!
//! Gated on `ICEBERG_INTEROP_RFDS_DIR` (unset ⇒ clean skip; offline `cargo test` stays green).
//! `dev/java-interop/run-interop-rewrite-files-delete-surface.sh` is the driver.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use iceberg::io::{FileIO, LocalFsStorageFactory};
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, ManifestContentType,
    ManifestStatus, NestedField, PrimitiveType, Schema, SortOrder, Struct, TableMetadata, Type,
    UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use serde::Deserialize;

// ===========================================================================================
// File-name constants (must match the Java oracle's constants byte-for-byte).
// ===========================================================================================

const DATA_FILE_NAME: &str = "00000-rfds-data.parquet";
const SEED_DEL_NAME: &str = "00000-rfds-seed-del.parquet";
const INHERITED_DEL_NAME: &str = "00000-rfds-inherited-del.parquet";
const EXPLICIT_DEL_NAME: &str = "00000-rfds-exp-del.parquet";

// Hand-declared expected seqs (the "spec" this test proves — must match the Java oracle).
const EXPECTED_EXPLICIT_SEQ: i64 = 2;

// ===========================================================================================
// Env gate.
// ===========================================================================================

fn rfds_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_RFDS_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

// ===========================================================================================
// JSON model for java_delete_seqs.json.
// ===========================================================================================

/// `java_delete_seqs.json` shape:
/// `{"inherited":{"path":"...","seq":3},"explicit":{"path":"...","seq":2}}`
#[derive(Debug, Deserialize)]
struct DeleteSeqEntry {
    path: String,
    seq: i64,
}

#[derive(Debug, Deserialize)]
struct DeleteSeqsJson {
    inherited: DeleteSeqEntry,
    explicit: DeleteSeqEntry,
}

fn read_delete_seqs(dir: &Path) -> DeleteSeqsJson {
    let path = dir.join("java_delete_seqs.json");
    let json = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&json).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

// ===========================================================================================
// Table-build helpers.
// ===========================================================================================

/// Schema `{1 id long required, 2 data string optional}` — matches the Java oracle's schema.
fn rfds_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build {id long, data string} schema")
}

/// Build a metadata-only DATA file at `path` (the file need not physically exist — this is a
/// METADATA-LEVEL seq proof; delete content is never read).
fn metadata_data_file(path: &str) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(512)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .expect("build metadata-only data file")
}

/// Build a metadata-only POSITION-DELETE file at `path`.
fn metadata_pos_delete_file(path: &str) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::PositionDeletes)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(128)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .expect("build metadata-only pos-delete file")
}

/// Build a fresh `MemoryCatalog` over a local-FS `FileIO` rooted at `warehouse`.
async fn build_catalog(name: &str, warehouse: &str) -> impl Catalog {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            name,
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_string())]),
        )
        .await
        .expect("build MemoryCatalog over local FS")
}

/// Create an unpartitioned V2 table at `location` inside `catalog`.
async fn create_rfds_table(catalog: &impl Catalog, location: &str) -> Table {
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");

    let creation = TableCreation::builder()
        .name("rfds_table".to_string())
        .location(location.to_string())
        .schema(rfds_schema())
        .partition_spec(UnboundPartitionSpec::builder().build())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();

    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create rfds table in catalog")
}

/// Fast-append `files` and return the updated table.
async fn fast_append(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let action = tx.fast_append().add_data_files(files);
    let tx = action.apply(tx).expect("apply fast_append");
    tx.commit(catalog).await.expect("commit fast_append")
}

/// Row-delta: add delete files and return the updated table.
async fn row_delta_add_deletes(
    catalog: &impl Catalog,
    table: &Table,
    deletes: Vec<DataFile>,
) -> Table {
    let tx = Transaction::new(table);
    let action = tx.row_delta().add_deletes(deletes);
    let tx = action.apply(tx).expect("apply row_delta");
    tx.commit(catalog).await.expect("commit row_delta")
}

/// Find the INHERITED data sequence number of an ADDED delete entry matched by path suffix in the
/// table's current snapshot's delete manifests. After `load_manifest`, null on-disk seq entries
/// have been resolved to the snapshot seq (inheritance), so `entry.sequence_number()` returns the
/// effective seq used at read time.
async fn added_delete_seq_by_name_suffix(table: &Table, name_suffix: &str) -> Option<i64> {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("table has a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load manifest list");

    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("load delete manifest");
        for entry in manifest.entries() {
            if entry.status() == ManifestStatus::Added && entry.file_path().ends_with(name_suffix) {
                return entry.sequence_number();
            }
        }
    }
    None
}

/// Write the table's current metadata to `<dir>/rust_rewritten/metadata/final.metadata.json`.
async fn write_rust_rewritten(table: &Table, dir: &Path) {
    let out_dir = dir.join("rust_rewritten").join("metadata");
    fs::create_dir_all(&out_dir).expect("create rust_rewritten/metadata");
    let final_path = out_dir
        .join("final.metadata.json")
        .to_string_lossy()
        .to_string();
    table
        .metadata()
        .write_to(table.file_io(), &final_path)
        .await
        .expect("write rust_rewritten/metadata/final.metadata.json");
    println!("interop_rfds: wrote Rust rewritten metadata → {final_path}");
}

// ===========================================================================================
// Direction 1 helper: load a Table from Java's final.metadata.json via Table::builder.
// ===========================================================================================

fn load_java_table(dir: &Path) -> Table {
    let meta_path = dir
        .join("table")
        .join("metadata")
        .join("final.metadata.json");
    let json = fs::read_to_string(&meta_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", meta_path.display()));
    let metadata: TableMetadata = serde_json::from_str(&json)
        .unwrap_or_else(|e| panic!("parse {}: {e}", meta_path.display()));
    Table::builder()
        .metadata(metadata)
        .metadata_location(meta_path.to_string_lossy().to_string())
        .identifier(TableIdent::from_strs(["interop", "rfds_table"]).expect("valid TableIdent"))
        .file_io(FileIO::new_with_fs())
        .build()
        .expect("build Table from Java final.metadata.json")
}

// ===========================================================================================
// The env-gated bidirectional interop test.
// ===========================================================================================

#[tokio::test]
async fn test_rewrite_files_delete_surface_interop() {
    let Some(dir) = rfds_dir() else {
        println!(
            "skipping interop_rewrite_files_delete_surface — set ICEBERG_INTEROP_RFDS_DIR \
             (run dev/java-interop/run-interop-rewrite-files-delete-surface.sh)"
        );
        return;
    };

    // ---- Read java_delete_seqs.json (ground truth from the Java generate step). ----
    let java_seqs = read_delete_seqs(&dir);
    println!(
        "interop_rfds: java_delete_seqs.json: \
         inherited path={} seq={}, exp path={} seq={}",
        java_seqs.inherited.path,
        java_seqs.inherited.seq,
        java_seqs.explicit.path,
        java_seqs.explicit.seq
    );

    // ==========================================================================================
    // DIRECTION 1: Rust reads the Java-written table and asserts the added-delete seqs.
    // ==========================================================================================
    //
    // Java wrote a V2 table under <dir>/table with 3 snapshots:
    //   S1 newAppend(D)       → seq 1
    //   S2 newRowDelta(DF0)   → seq 2
    //   S3 newRewrite: deleteFile(DF0) + addFile(DF_inherited) + addFile(DF_exp, 2L) → seq 3
    //
    // We load final.metadata.json via Table::builder (local-FS FileIO) and read the sequence
    // numbers of the two ADDED delete entries from the S3 delete manifests.

    let java_table = load_java_table(&dir);
    let java_s3_seq = java_table
        .metadata()
        .current_snapshot()
        .expect("Java table has a current snapshot (S3)")
        .sequence_number();
    println!("interop_rfds D1: Java S3 snapshot seq = {java_s3_seq}");

    let d1_inherited_seq = added_delete_seq_by_name_suffix(&java_table, INHERITED_DEL_NAME)
        .await
        .unwrap_or_else(|| {
            panic!(
                "DIRECTION 1: ADDED entry for {INHERITED_DEL_NAME} \
                 not found in Java S3 delete manifests"
            )
        });
    let d1_exp_seq = added_delete_seq_by_name_suffix(&java_table, EXPLICIT_DEL_NAME)
        .await
        .unwrap_or_else(|| {
            panic!(
                "DIRECTION 1: ADDED entry for {EXPLICIT_DEL_NAME} \
                 not found in Java S3 delete manifests"
            )
        });

    assert_eq!(
        d1_inherited_seq, java_seqs.inherited.seq,
        "DIRECTION 1: inherited-del seq from manifest ({d1_inherited_seq}) != java_delete_seqs.json ({})",
        java_seqs.inherited.seq
    );
    assert_eq!(
        d1_inherited_seq, java_s3_seq,
        "DIRECTION 1: inherited-del seq ({d1_inherited_seq}) must equal Java S3 snapshot seq \
         ({java_s3_seq}); add_delete_file must inherit the snapshot's data seq"
    );
    assert_eq!(
        d1_exp_seq, java_seqs.explicit.seq,
        "DIRECTION 1: explicit-del seq from manifest ({d1_exp_seq}) != java_delete_seqs.json ({})",
        java_seqs.explicit.seq
    );
    assert_eq!(
        d1_exp_seq, EXPECTED_EXPLICIT_SEQ,
        "DIRECTION 1: explicit-del seq ({d1_exp_seq}) must equal the hand-declared expected \
         ({EXPECTED_EXPLICIT_SEQ})"
    );
    assert_ne!(
        d1_inherited_seq, d1_exp_seq,
        "DIRECTION 1: inherited and explicit seqs must differ (both equal {d1_inherited_seq}); \
         the test proves nothing if seqs are equal"
    );

    println!(
        "interop_rfds D1 PASS: inherited-del seq={d1_inherited_seq} (==Java S3 seq {java_s3_seq}); \
         exp-del seq={d1_exp_seq} (=={EXPECTED_EXPLICIT_SEQ})"
    );

    // ==========================================================================================
    // DIRECTION 2: Rust independently builds the SAME locked fixture; asserts seqs; writes
    //              rust_rewritten/metadata/final.metadata.json for Java verify.
    // ==========================================================================================
    //
    // We build the SAME table structure in Rust independently (no reference to Java data):
    //   S1 fast_append(D)        → seq 1
    //   S2 row_delta(DF0)        → seq 2
    //   S3 rewrite_files:
    //      delete_delete_file(DF0)
    //      add_delete_file(DF_inherited)              [inherited seq → will be 3]
    //      add_delete_file_with_sequence_number(DF_exp, 2)  [explicit seq → stays 2]
    //      → seq 3
    //
    // Place the Rust table at <dir>/rust_rewritten/table (distinct from Java's <dir>/table).

    let rust_table_location = dir
        .join("rust_rewritten")
        .join("table")
        .to_string_lossy()
        .to_string();
    let rust_table_data_dir = format!("{rust_table_location}/data");
    fs::create_dir_all(&rust_table_data_dir).expect("create rust_rewritten/table/data");

    let warehouse_d2 = dir.join("rust_rewritten").to_string_lossy().to_string();
    let catalog_d2 = build_catalog("interop_rfds_d2", &warehouse_d2).await;
    let table = create_rfds_table(&catalog_d2, &rust_table_location).await;

    // S1: fast_append(D).
    let data_path = format!("{rust_table_data_dir}/{DATA_FILE_NAME}");
    let data_file = metadata_data_file(&data_path);
    let table = fast_append(&catalog_d2, &table, vec![data_file]).await;
    assert_eq!(
        table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number(),
        1,
        "S1 must have sequence number 1"
    );

    // S2: row_delta(DF0) — seed position-delete (satisfies the precondition that a delete must
    // be deleted when adds include delete files).
    let seed_del_path = format!("{rust_table_data_dir}/{SEED_DEL_NAME}");
    let seed_del = metadata_pos_delete_file(&seed_del_path);
    let table = row_delta_add_deletes(&catalog_d2, &table, vec![seed_del.clone()]).await;
    assert_eq!(
        table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number(),
        2,
        "S2 must have sequence number 2"
    );

    // S3: rewrite_files — delete_delete_file(DF0) + add_delete_file(DF_inherited) +
    //     add_delete_file_with_sequence_number(DF_exp, 2).
    let inherited_del_path = format!("{rust_table_data_dir}/{INHERITED_DEL_NAME}");
    let exp_del_path = format!("{rust_table_data_dir}/{EXPLICIT_DEL_NAME}");
    let inherited_del = metadata_pos_delete_file(&inherited_del_path);
    let exp_del = metadata_pos_delete_file(&exp_del_path);

    let tx = Transaction::new(&table);
    let action = tx
        .rewrite_files(vec![], vec![])
        .delete_delete_file(seed_del)
        .add_delete_file(inherited_del)
        .add_delete_file_with_sequence_number(exp_del, EXPECTED_EXPLICIT_SEQ);
    let tx = action.apply(tx).expect("apply rewrite_files (D2)");
    let table = tx
        .commit(&catalog_d2)
        .await
        .expect("commit rewrite_files (D2)");

    let s3_seq = table
        .metadata()
        .current_snapshot()
        .unwrap()
        .sequence_number();
    assert_eq!(s3_seq, 3, "Rust S3 must have sequence number 3");

    // Assert the two added-delete seqs from the Rust-written delete manifests.
    let d2_inherited_seq = added_delete_seq_by_name_suffix(&table, INHERITED_DEL_NAME)
        .await
        .unwrap_or_else(|| {
            panic!(
                "DIRECTION 2: ADDED entry for {INHERITED_DEL_NAME} \
                 not found in Rust S3 delete manifests"
            )
        });
    let d2_exp_seq = added_delete_seq_by_name_suffix(&table, EXPLICIT_DEL_NAME)
        .await
        .unwrap_or_else(|| {
            panic!(
                "DIRECTION 2: ADDED entry for {EXPLICIT_DEL_NAME} \
                 not found in Rust S3 delete manifests"
            )
        });

    assert_eq!(
        d2_inherited_seq, s3_seq,
        "DIRECTION 2: inherited-del seq ({d2_inherited_seq}) must equal Rust S3 snapshot seq \
         ({s3_seq}); add_delete_file must inherit the snapshot seq"
    );
    assert_eq!(
        d2_exp_seq, EXPECTED_EXPLICIT_SEQ,
        "DIRECTION 2: explicit-del seq ({d2_exp_seq}) must equal the hand-declared expected \
         ({EXPECTED_EXPLICIT_SEQ}); add_delete_file_with_sequence_number must stamp the given seq"
    );
    assert_ne!(
        d2_inherited_seq, d2_exp_seq,
        "DIRECTION 2: inherited and explicit seqs must differ (both equal {d2_inherited_seq}); \
         seq semantics not distinguished"
    );

    println!(
        "interop_rfds D2 PASS: Rust S3 seq={s3_seq}; \
         inherited-del seq={d2_inherited_seq} (==S3); exp-del seq={d2_exp_seq} (=={EXPECTED_EXPLICIT_SEQ})"
    );

    // Write the Rust-built table metadata for the Java verify step.
    write_rust_rewritten(&table, &dir).await;

    println!(
        "interop_rfds BIDIRECTIONAL PASS — \
         D1: Java wrote, Rust verified seqs; \
         D2: Rust wrote, Java verify pending (final.metadata.json emitted)"
    );
}
