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

//! METADATA-LEVEL multi-spec MERGING-ACTION interop fixture (AC·OO #5, Wave 2) — proving that a
//! Rust-written `row_delta` adding DELETE files under MORE THAN ONE partition spec in ONE commit
//! produces PER-SPEC delete manifests that match Java 1.10.0 byte-for-byte in the canonical
//! snapshot-metadata view.
//!
//! # The residue this closes
//!
//! The Z2 fixture ([`interop_multi_spec.rs`]) proved a single multi-spec `fast_append` produces TWO
//! DATA manifests with different `partition_spec_id` — exercising the producer's per-spec DATA
//! grouping (`SnapshotProducer::write_added_manifests`). Z2 carried NO delete files, so the
//! SYMMETRIC per-spec DELETE-manifest grouping (`SnapshotProducer::write_added_delete_manifests`,
//! Java `MergingSnapshotProducer.newDeleteFilesAsManifests`) was never reached cross-language. This
//! fixture closes that gap by adding a multi-spec DELETE commit driven by the `RowDelta` merging
//! action (GAP_MATRIX row 94's named residue: "multi-spec delete commits").
//!
//! # The chain (both sides, identical logical constants; V2 table; NO parquet)
//!
//! - ms1: `fast_append` F0(a="q", rc=10) under spec 0 [identity(a)]          seq 1, op append
//! - ms2: `update_partition_spec` add identity(b) → spec 1 becomes default    NO SNAPSHOT
//! - ms3: `fast_append` F3(a="r", b="s", rc=10) under spec 1                 seq 2, op append
//! - ms4: ONE multi-spec `row_delta`: D0(posDelete, spec0, a="q", rc=1) AND
//!   D1(posDelete, spec1, a="r", b="s", rc=1)                                seq 3, op delete
//!   (THE HEADLINE: TWO DELETE manifests, `partition_spec_id` 0 AND 1)
//!
//! # Exactly one data manifest per spec before ms4 (deliberate)
//!
//! ms1/ms3 seed spec 0 and spec 1 with EXACTLY ONE data manifest each. On the ms4 row_delta, the
//! merging producer still scans the existing DATA manifests, but a spec group of a single manifest
//! is a size-1 bin that the manifest merge manager returns AS-IS (Java's `bin.size() == 1` early
//! return) — so the DATA manifests pass through unchanged on both sides. A second data manifest in
//! one spec would trip Java's `first`-relative force-merge of the OTHER spec group (an
//! order-dependent DATA-manifest rewrite the Rust merging producer does not mirror); the chain is
//! shaped to keep that DATA-merge path dormant so this unit isolates the DELETE-manifest grouping.
//!
//! # Tie-shaping — the spec-id tiebreaker is the ONLY disambiguator (on the DELETE side this time)
//!
//! D0 and D1 have IDENTICAL record_count=1, so the two ms4 DELETE manifests tie on ALL nine prior
//! sort-tuple keys (content=deletes, seq=3, min_seq=3, added_files_count=1, existing_files_count=0,
//! deleted_files_count=0, added_rows=1, existing_rows=0, deleted_rows=0) and differ ONLY on
//! `partition_spec_id` (0 vs 1). The W3 tiebreaker at position 10 is the sole disambiguator —
//! identical to Z2's ms4 but exercising the DELETE-manifest grouping path.
//!
//! # The two comparison directions
//!
//! **Direction 1 (GEN — Rust writes, Java judges).**
//! [`test_multispec_merge_gen_rust_performs_the_chain`] performs the four-step chain on a local-FS
//! `MemoryCatalog`, landing `final.metadata.json` at `<gen_dir>/rust_table/metadata/`. The run
//! script has Java emit its canonical view of that table and byte-diffs it against `java_meta.json`.
//!
//! **Direction 2 (READ parity — Java writes, Rust verifies).**
//! [`test_rust_view_of_java_multispec_merge_chain_matches_java_view`] asserts Rust's canonical view
//! of the Java-written chain equals Java's own `java_meta.json`.
//!
//! **Direction 2b (WRITE parity — Rust writes, Rust verifies).**
//! [`test_rust_multispec_merge_chain_matches_java_semantics`] asserts Rust's canonical view of the
//! Rust-written chain (from the GEN step) equals `java_meta.json`.
//!
//! # The env gate
//!
//! Tests are NO-OPS unless their env var is set non-empty (offline `cargo test` passes cleanly).
//! `ICEBERG_INTEROP_MULTISPEC_MERGE_GEN_DIR` — write the Rust chain here (Direction 1).
//! `ICEBERG_INTEROP_MULTISPEC_MERGE_DIR` — compare tests read Java's fixtures from here (Direction 2).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, Literal,
    ManifestContentType, NestedField, PrimitiveType, Schema, SortOrder, Struct, Transform, Type,
    UnboundPartitionSpec,
};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use serde_json::Value as JsonValue;

mod common;
use common::snapshot_meta_view::snapshot_meta_view;

fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_MULTISPEC_MERGE_GEN_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn compare_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_MULTISPEC_MERGE_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

// ===========================================================================================
// Schema and partition spec builders — identical to the Java MultiSpecMergeOracle (and the Z2
// MultiSpecOracle): the fixtures share the table shape, diverging only at the merging commit.
// ===========================================================================================

/// Schema: `{1 a string required, 2 b string optional}`.
/// `b` is optional so a spec-0 partition tuple (only "a") round-trips without a null-fill error.
fn multi_spec_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "a", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(2, "b", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build the {a, b} schema")
}

/// Spec 0: `identity(a)` only — the initial partition spec (partition field id 1000, name "a").
fn spec_zero() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(1, "a".to_string(), Transform::Identity)
        .expect("add identity(a) partition field")
        .build()
}

/// A metadata-only DATA `DataFile` under SPEC 0 (`identity(a)`, one-field partition tuple).
fn fake_data_file_spec0(
    table_location: &str,
    name: &str,
    a_val: &str,
    record_count: u64,
) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(format!("{table_location}/data/{name}"))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(record_count * 100)
        .record_count(record_count)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::string(a_val))]))
        .build()
        .expect("build spec-0 metadata-only data file")
}

/// A metadata-only DATA `DataFile` under SPEC 1 (`identity(a), identity(b)`, two-field tuple).
fn fake_data_file_spec1(
    table_location: &str,
    name: &str,
    a_val: &str,
    b_val: &str,
    record_count: u64,
) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(format!("{table_location}/data/{name}"))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(record_count * 100)
        .record_count(record_count)
        .partition_spec_id(1)
        .partition(Struct::from_iter([
            Some(Literal::string(a_val)),
            Some(Literal::string(b_val)),
        ]))
        .build()
        .expect("build spec-1 metadata-only data file")
}

/// A metadata-only POSITION-delete file under SPEC 0 (one-field partition tuple `a`), referencing a
/// data file. The mirror of the Java `FileMetadata.deleteFileBuilder(spec0).ofPositionDeletes()`
/// idiom — no parquet, the fixture only reads manifests. Carries `partition_spec_id=0`.
fn fake_pos_delete_spec0(
    table_location: &str,
    name: &str,
    a_val: &str,
    referenced_data_file: &str,
    record_count: u64,
) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::PositionDeletes)
        .file_path(format!("{table_location}/data/{name}"))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(record_count * 100)
        .record_count(record_count)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::string(a_val))]))
        .referenced_data_file(Some(referenced_data_file.to_string()))
        .build()
        .expect("build spec-0 metadata-only position-delete file")
}

/// A metadata-only POSITION-delete file under SPEC 1 (two-field partition tuple `a`, `b`),
/// referencing a data file. Carries `partition_spec_id=1`.
fn fake_pos_delete_spec1(
    table_location: &str,
    name: &str,
    a_val: &str,
    b_val: &str,
    referenced_data_file: &str,
    record_count: u64,
) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::PositionDeletes)
        .file_path(format!("{table_location}/data/{name}"))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(record_count * 100)
        .record_count(record_count)
        .partition_spec_id(1)
        .partition(Struct::from_iter([
            Some(Literal::string(a_val)),
            Some(Literal::string(b_val)),
        ]))
        .referenced_data_file(Some(referenced_data_file.to_string()))
        .build()
        .expect("build spec-1 metadata-only position-delete file")
}

// ===========================================================================================
// The Rust GEN path — Direction 1: Rust writes the chain, Java judges it.
// ===========================================================================================

/// Rust performs the SAME four-step multi-spec merging-action chain as Java's
/// `MultiSpecMergeOracle.generate`, writing the table to `<gen_dir>/rust_table` through the
/// PRODUCTION write paths, and lands `final.metadata.json` for the Java emitter + the comparison
/// tests.
///
/// ms1-ms3 seed one data manifest per spec; ms4 is THE HEADLINE: a single `row_delta` carrying ONE
/// position-delete under spec 0 AND ONE under spec 1. `group_delete_files_by_spec` routes D0 into
/// the spec-0 DELETE manifest writer and D1 into the spec-1 DELETE manifest writer, producing TWO
/// DELETE manifests with `partition_spec_id=0` and `partition_spec_id=1` respectively — exercising
/// the W3 spec-id tiebreaker as the ONLY disambiguator (both deletes have record_count=1).
#[tokio::test]
async fn test_multispec_merge_gen_rust_performs_the_chain() {
    let Some(gen_dir) = gen_dir() else {
        println!(
            "skipping interop_multispec_merge GEN — set ICEBERG_INTEROP_MULTISPEC_MERGE_GEN_DIR \
             (run dev/java-interop/run-interop-multispec-merge.sh)"
        );
        return;
    };

    let warehouse = gen_dir.to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_multispec_merge_gen",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");

    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");

    // Create the table with spec 0 (identity(a) only).
    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location.clone())
        .schema(multi_spec_schema())
        .partition_spec(spec_zero())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create multispec-merge rust_table");

    // ms1: fast_append F0 under spec 0 (seq 1) — the SINGLE spec-0 data manifest.
    let file_f0 = fake_data_file_spec0(&table_location, "s0_f0.parquet", "q", 10);
    let f0_path = file_f0.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file_f0])
        .apply(tx)
        .expect("apply ms1 fast_append (spec 0, F0)");
    let table = tx.commit(&catalog).await.expect("commit ms1 fast_append");

    // ms2: evolve the partition spec — add identity(b) to produce spec 1. NO snapshot.
    let tx = Transaction::new(&table);
    let tx = tx
        .update_partition_spec()
        .add_field("b")
        .apply(tx)
        .expect("apply ms2 update_partition_spec (add identity(b))");
    let table = tx
        .commit(&catalog)
        .await
        .expect("commit ms2 update_partition_spec");

    let default_spec_id = table.metadata().default_partition_spec_id();
    assert_eq!(
        default_spec_id, 1,
        "after adding identity(b) the default spec must be spec 1; got {default_spec_id}"
    );

    // ms3: fast_append F3 under spec 1 (seq 2) — the SINGLE spec-1 data manifest.
    let file_f3 = fake_data_file_spec1(&table_location, "s1_f3.parquet", "r", "s", 10);
    let f3_path = file_f3.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file_f3])
        .apply(tx)
        .expect("apply ms3 fast_append (spec 1, F3)");
    let table = tx.commit(&catalog).await.expect("commit ms3 fast_append");

    // ms4: THE MULTI-SPEC DELETE commit — ONE row_delta carrying:
    //   D0: spec-0-stamped position-delete (partition a="q", referencing F0, record_count=1)
    //   D1: spec-1-stamped position-delete (partition a="r"/b="s", referencing F3, record_count=1)
    //
    // TIE-SHAPING: both D0 and D1 have record_count=1 so the two ms4 DELETE manifests tie on ALL
    // nine prior sort-tuple keys and differ ONLY on partition_spec_id (0 vs 1) — the W3 tiebreaker
    // at position 10 is the ONLY disambiguator. `group_delete_files_by_spec` routes D0 into the
    // spec-0 DELETE manifest writer and D1 into the spec-1 one, producing TWO DELETE manifests with
    // different partition_spec_id within ONE snapshot — the symmetric DELETE-side sibling of Z2's
    // ms4 DATA grouping (write_added_delete_manifests vs write_added_manifests).
    let delete_d0 =
        fake_pos_delete_spec0(&table_location, "s0_d0-deletes.parquet", "q", &f0_path, 1);
    let delete_d1 = fake_pos_delete_spec1(
        &table_location,
        "s1_d1-deletes.parquet",
        "r",
        "s",
        &f3_path,
        1,
    );

    // Assert tie-shaping property on the DELETE side: D0 and D1 have the same record_count so the
    // two ms4 DELETE manifests tie on the prior 9 sort-tuple keys; the spec-id field is the only
    // difference (spec-0 tuple has 1 field, spec-1 tuple has 2). The spec-id tiebreaker (W3,
    // position 10) is the ONLY disambiguator.
    assert_eq!(
        delete_d0.record_count(),
        delete_d1.record_count(),
        "tie-shaping violated: D0 and D1 must have the same record_count so the two ms4 DELETE \
         manifests tie on all 9 prior sort keys (W3 ruling: spec_id tiebreaker at position 10 is \
         the ONLY disambiguator)"
    );
    assert_ne!(
        delete_d0.partition().fields().len(),
        delete_d1.partition().fields().len(),
        "D0 (spec 0, 1-field partition) and D1 (spec 1, 2-field partition) must have different \
         partition tuple arities — they carry different spec ids"
    );

    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_d0, delete_d1])
        .apply(tx)
        .expect("apply ms4 multi-spec row_delta (spec 0 D0 + spec 1 D1)");
    let table = tx
        .commit(&catalog)
        .await
        .expect("commit ms4 multi-spec row_delta");

    // Assert the multi-spec DELETE commit produced TWO DELETE manifests in the ms4 snapshot.
    let ms4_snapshot = table
        .metadata()
        .current_snapshot()
        .expect("ms4 snapshot must be current after commit");
    assert_eq!(
        ms4_snapshot.summary().operation.as_str(),
        "delete",
        "ms4 must classify as `delete` (addsDeleteFiles && !addsDataFiles, the 1.10.0 RowDelta \
         two-branch rule); got {}",
        ms4_snapshot.summary().operation.as_str()
    );
    let manifest_list = ms4_snapshot
        .load_manifest_list(table.file_io(), &table.metadata_ref())
        .await
        .expect("load ms4 manifest list");
    let ms4_seq = ms4_snapshot.sequence_number();
    let ms4_delete_manifests: Vec<_> = manifest_list
        .entries()
        .iter()
        .filter(|manifest| {
            manifest.sequence_number == ms4_seq && manifest.content == ManifestContentType::Deletes
        })
        .collect();
    assert_eq!(
        ms4_delete_manifests.len(),
        2,
        "the ms4 multi-spec row_delta must produce exactly 2 NEW DELETE manifests (one per spec \
         group); got {} at seq {ms4_seq}",
        ms4_delete_manifests.len()
    );
    let delete_spec_ids_in_ms4: Vec<i32> = {
        let mut ids: Vec<i32> = ms4_delete_manifests
            .iter()
            .map(|m| m.partition_spec_id)
            .collect();
        ids.sort_unstable();
        ids
    };
    assert_eq!(
        delete_spec_ids_in_ms4,
        vec![0, 1],
        "the two ms4 DELETE manifests must carry partition_spec_id 0 and 1 respectively; got {:?}",
        delete_spec_ids_in_ms4
    );

    // Land the final metadata at the known path for the Java emitter + the comparison tests.
    let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
    table
        .metadata()
        .clone()
        .write_to(table.file_io(), &final_metadata_path)
        .await
        .expect("write final.metadata.json");

    println!(
        "interop_multispec_merge GEN OK — Rust performed the four-step multi-spec merging-action \
         chain at {table_location} (ms1:spec0 F0 + ms2:evolve + ms3:spec1 F3 + ms4:multi-spec \
         DELETE row_delta → two per-spec DELETE manifests). The Java emitter + diff judge it next."
    );
}

// ===========================================================================================
// The comparison tests — Direction 2: Java acts, Rust verifies.
// ===========================================================================================

fn load_json(path: &std::path::Path) -> JsonValue {
    let raw =
        fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// READ parity: Rust's canonical view of the JAVA multi-spec merging-action chain equals Java's own
/// view. Load-bearing assertions riding inside the canonical view:
/// - ms4 has two DELETE manifests with DIFFERENT partition_spec_id values (0 and 1).
/// - The spec-id tiebreaker (position 10) resolves the same-arity tie — both deletes have rc=1.
/// - DELETE-entry partition tuples rendered under each manifest's OWN spec (the file's-own-spec
///   rule): spec-0 DELETE entries render under identity(a) only; spec-1 under identity(a)+identity(b).
#[tokio::test]
async fn test_rust_view_of_java_multispec_merge_chain_matches_java_view() {
    let Some(dir) = compare_dir() else {
        println!(
            "skipping interop_multispec_merge compare (D2 read parity) — set \
             ICEBERG_INTEROP_MULTISPEC_MERGE_DIR (run \
             dev/java-interop/run-interop-multispec-merge.sh)"
        );
        return;
    };

    let java_view = load_json(&dir.join("java_meta.json"));
    let rust_view = snapshot_meta_view(&dir.join("table/metadata/final.metadata.json")).await;
    assert_eq!(
        rust_view, java_view,
        "Rust's view of the JAVA multi-spec merging-action chain diverges from Java's own view"
    );
    println!("multispec_merge: Rust view of Java chain == Java view OK");
}

/// WRITE parity — the crown jewel: the RUST-written four-step multi-spec merging-action chain
/// produces canonical metadata indistinguishable from Java's across spec evolution, a multi-spec
/// DELETE row_delta, PER-SPEC delete-manifest grouping, and the spec-id tiebreaker. The tie-shaping
/// property (both ms4 deletes have the same record count so the two DELETE manifests are pure
/// spec-id apart) was asserted in the GEN step above; here we confirm the byte-level view matches
/// Java.
#[tokio::test]
async fn test_rust_multispec_merge_chain_matches_java_semantics() {
    let Some(dir) = compare_dir() else {
        println!(
            "skipping interop_multispec_merge compare (D2 write parity) — set \
             ICEBERG_INTEROP_MULTISPEC_MERGE_DIR (run \
             dev/java-interop/run-interop-multispec-merge.sh)"
        );
        return;
    };

    let rust_metadata = dir.join("rust_table/metadata/final.metadata.json");
    assert!(
        rust_metadata.exists(),
        "missing {} — run the GEN step of run-interop-multispec-merge.sh first",
        rust_metadata.display()
    );
    let java_view = load_json(&dir.join("java_meta.json"));
    let rust_view = snapshot_meta_view(&rust_metadata).await;
    assert_eq!(
        rust_view, java_view,
        "the RUST multi-spec merging-action chain's canonical metadata diverges from Java's \
         semantics"
    );
    println!("multispec_merge: Rust chain metadata == Java semantics OK");
}
