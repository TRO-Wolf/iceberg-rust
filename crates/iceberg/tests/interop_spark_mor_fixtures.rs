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

//! SPARK-WRITTEN merge-on-read fixtures — the real-writer leg of file-scoped delete routing.
//!
//! Every other delete-routing pin in this crate reads metadata that **this repo or the Java
//! interop oracle built on purpose**: the oracle's file-scoped fixture
//! (`dev/java-interop/run-interop-file-scoped-deletes.sh`) has to set
//! `write.metadata.metrics.column.file_path=full` by hand to get equal `file_path` bounds, and its
//! FIELD leg is a hand-assembled `FileMetadata.deleteFileBuilder(…).withReferencedDataFile(…)`.
//! THIS suite reads four tables written by **Spark's own connector at default properties**, so it
//! is the first evidence in the repo that the bounds leg is what a production writer actually
//! emits.
//!
//! # Why the bounds leg is the load-bearing one
//!
//! Bytecode-verified independently at Iceberg 1.10.0 and 1.11.0 (RePark consumer, 2026-07-25): the
//! ONLY writer that calls `FileMetadata$Builder.withReferencedDataFile` is `deletes.BaseDVFileWriter`
//! — the format-V3 deletion-vector writer (1.11.0 adds only `RewriteTablePathUtil`, a
//! metadata-rewrite utility). `PositionDeleteWriter`, `SortingPositionOnlyDeleteWriter` and
//! `FileScopedPositionDeleteWriter` never set it, and 1.4.3 does not contain the string at all.
//! For a Spark-written **V2** table, equal `file_path` bounds are therefore the ONLY file-scoping
//! signal that exists. The inverse trap is covered here too: a PARTITION-granularity delete carries
//! NO `file_path` bounds at all (Iceberg omits them rather than widening them), so **absent** bounds
//! — never *unequal* bounds — is the partition-scoped signal.
//!
//! # The measured differential (probe, 2026-07-25)
//!
//! Driving these same four fixtures at the pre-routing base (`a6199ca5`) and at this branch's tip:
//!
//! | fixture | base attachments | tip attachments |
//! |---|---|---|
//! | `mor_file_gran` | 4 — both deletes broadcast to both data files | 2 — each delete bound to the one file its equal bounds name |
//! | `mor_legacy_no_field` | 4 — both deletes broadcast to both data files | 2 — likewise |
//! | `mor_partition_gran` | 2 — one delete to both files | 2 — IDENTICAL (absent bounds must NOT narrow) |
//! | `mor_dv_v3` | 2 — one DV per data file | 2 — IDENTICAL (DVs already routed by `referenced_data_file`) |
//!
//! MUTATION-PINNED at the tip (2026-07-26): disabling leg 3 of
//! `delete_file_index::referenced_data_file_location` — an early `return None` before the
//! `file_path`-bounds lookup, which is exactly the pre-routing behaviour — turns
//! [`test_spark_mor_fixture_delete_routing`] RED on `db.mor_file_gran` with the base's two
//! attachments per data file. [`test_spark_mor_fixture_live_rows`] stays GREEN under that same
//! mutation, which is the honest measure of what the row assertion can and cannot see (below).
//!
//! # What this suite does NOT pin — read before extending it
//!
//! **Row resurrection.** All four tables are UNPARTITIONED and single-spec, so every data file and
//! every delete file carries `partition_spec_id = 0` and an EMPTY partition tuple. The pre-routing
//! `(spec_id, partition)`-keyed index therefore reached a strict SUPERSET of the correct data files
//! — it over-attached, it never missed — and the over-attachment is row-invisible because the read
//! side keys parsed positional deletes by the data-file path each RECORD names. The live row set is
//! `{1,3,4,5,7,8}` at BOTH revisions on all four fixtures. The row-level corruption class (a delete
//! whose `(spec_id, partition)` stamp differs from the file it names, which makes the pre-routing
//! index DROP it and resurrect rows) is pinned only by the Java oracle fixture in
//! `interop_scan_exec.rs::test_file_scoped_delete_scan_matches_java_read`. Do not describe this
//! suite as a resurrection pin.
//!
//! The row assertion here is still load-bearing, but for a different reason: combined with
//! "exactly one delete per data file", it is what forces the pairing to be the CORRECT one. A
//! swapped pairing would attach to each data file a delete whose records name the *other* file, so
//! nothing would be deleted and ids 2 and 6 would come back.
//!
//! # The fixture (NOT in this repo)
//!
//! Generated 2026-07-25 by the RePark consumer with real Spark, and deliberately **not committed
//! here**: the four warehouses are ~216 KB of binary parquet/Avro/Puffin whose manifests are
//! DEFLATE-compressed with the generating machine's ABSOLUTE paths baked in (metadata JSON,
//! manifest lists, manifests, and the position-delete `file_path` column alike), so they are not
//! relocatable without either a prefix-remapping storage shim or regeneration. See
//! `dev/java-interop/map.md` for the pointer and the relocation options.
//!
//! | table | writer | fmt | shape |
//! |---|---|---|---|
//! | `db.mor_file_gran` | Spark 4.0.0 + Iceberg 1.10.0 | v2 | FILE granularity: 2 parquet position deletes, `referenced_data_file` DECLARED but NULL, `file_path` lower == upper |
//! | `db.mor_partition_gran` | Spark 4.0.0 + Iceberg 1.10.0 | v2 | PARTITION granularity (`write.delete.granularity=partition`): ONE parquet position delete spanning both data files, field NULL, `file_path` bounds ABSENT |
//! | `db.mor_dv_v3` | Spark 4.0.0 + Iceberg 1.10.0 | v3 | two Puffin deletion vectors in ONE `.puffin` file at distinct content offsets, `referenced_data_file` SET on both |
//! | `db.mor_legacy_no_field` | Spark 3.5.6 + Iceberg 1.4.3 | v2 | FILE granularity, and `referenced_data_file` is ABSENT FROM THE MANIFEST AVRO SCHEMA — not merely null |
//!
//! `mor_legacy_no_field` is the one shape with no other coverage anywhere: `DataFileSerde`'s
//! `referenced_data_file` is an `Option<String>` with NO `#[serde(default)]`, and decoding it out of
//! a manifest whose Avro schema never declared the field works only via serde's missing-field-to-
//! `None` path over `apache_avro::from_value`. Until this suite that was an inference, not a test —
//! every Iceberg table written before ~1.7 has that shape.
//!
//! # The env gate
//!
//! Gated on `ICEBERG_SPARK_MOR_FIXTURE_DIR` = the directory holding `warehouse-iceberg-1.10.0/` and
//! `warehouse-iceberg-1.4.3/`. When the var is UNSET every test is a clean runtime NO-OP (an early
//! return, not `#[ignore]`), so the offline `cargo test` gate stays green. Unlike the `interop_*`
//! suites this one needs NO Java and NO Maven when the fixture is present — the tables are already
//! written.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use futures::TryStreamExt;
use iceberg::TableIdent;
use iceberg::io::FileIO;
use iceberg::scan::FileScanTask;
use iceberg::spec::{DataContentType, DataFileFormat, TableMetadata};
use iceberg::table::Table;

/// The env var naming the fixture root. Unset ⇒ every test in this file is a clean no-op.
const FIXTURE_DIR_ENV: &str = "ICEBERG_SPARK_MOR_FIXTURE_DIR";

/// The live ids of every fixture table after merge-on-read: 8 rows written, ids 2 and 6 deleted.
/// Identical to `gen/verify_readback.py`'s `EXPECTED_IDS`, which Spark itself asserts.
const EXPECTED_LIVE_IDS: [i64; 6] = [1, 3, 4, 5, 7, 8];

/// How the read side is expected to bind this fixture's delete files to its data files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeleteRouting {
    /// FILE granularity via EQUAL `file_path` bounds (leg 3 of `referenced_data_file_location`):
    /// one parquet position delete per data file, no two data files sharing one, and the explicit
    /// `referenced_data_file` field NOT set — which is what makes this the bounds leg and not the
    /// field leg.
    FileScopedByBounds,
    /// PARTITION granularity: ONE parquet position delete with ABSENT bounds, which must stay
    /// partition-scoped and therefore reach EVERY data file in the partition.
    PartitionScoped,
    /// V3 deletion vectors: one Puffin blob per data file, bound by `referenced_data_file`.
    DeletionVector,
}

/// One fixture table: where it lives under the fixture root, and what routing it must produce.
struct Fixture {
    /// Path of the table directory relative to the fixture root.
    relative_path: &'static str,
    /// `namespace.table`, for identifiers and assertion messages.
    name: &'static str,
    routing: DeleteRouting,
}

const FIXTURES: [Fixture; 4] = [
    Fixture {
        relative_path: "warehouse-iceberg-1.10.0/db/mor_file_gran",
        name: "db.mor_file_gran",
        routing: DeleteRouting::FileScopedByBounds,
    },
    Fixture {
        relative_path: "warehouse-iceberg-1.10.0/db/mor_partition_gran",
        name: "db.mor_partition_gran",
        routing: DeleteRouting::PartitionScoped,
    },
    Fixture {
        relative_path: "warehouse-iceberg-1.10.0/db/mor_dv_v3",
        name: "db.mor_dv_v3",
        routing: DeleteRouting::DeletionVector,
    },
    Fixture {
        relative_path: "warehouse-iceberg-1.4.3/db/mor_legacy_no_field",
        name: "db.mor_legacy_no_field",
        routing: DeleteRouting::FileScopedByBounds,
    },
];

/// The fixture root, or `None` when the gate var is unset (⇒ clean no-op).
fn fixture_root() -> Option<PathBuf> {
    let raw = std::env::var(FIXTURE_DIR_ENV).ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    let path = PathBuf::from(raw);
    assert!(
        path.is_dir(),
        "{FIXTURE_DIR_ENV} points at {}, which is not a directory",
        path.display()
    );
    Some(path)
}

/// Load a Spark-written table from its on-disk `metadata/` directory.
///
/// Spark maintains `version-hint.text` (the current metadata version) next to `v<N>.metadata.json`,
/// exactly as a Hadoop-catalog table does; the absolute manifest / data paths baked into that JSON
/// resolve through a local-filesystem `FileIO`.
fn load_table(table_dir: &Path, name: &str) -> Table {
    let hint_path = table_dir.join("metadata/version-hint.text");
    let hint = fs::read_to_string(&hint_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", hint_path.display()));
    let version: u32 = hint
        .trim()
        .parse()
        .unwrap_or_else(|error| panic!("parse version hint {:?} of {name}: {error}", hint.trim()));

    let metadata_path = table_dir.join(format!("metadata/v{version}.metadata.json"));
    let json = fs::read_to_string(&metadata_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", metadata_path.display()));
    let metadata: TableMetadata = serde_json::from_str(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", metadata_path.display()));

    let (namespace, table) = name
        .split_once('.')
        .unwrap_or_else(|| panic!("fixture name {name} is not `namespace.table`"));

    Table::builder()
        .metadata(metadata)
        .metadata_location(metadata_path.to_string_lossy().to_string())
        .identifier(TableIdent::from_strs([namespace, table]).expect("valid identifier"))
        .file_io(FileIO::new_with_fs())
        .build()
        .unwrap_or_else(|error| {
            panic!(
                "build table {name} from {}: {error}",
                metadata_path.display()
            )
        })
}

/// Plan the whole table, returning the tasks sorted by data-file path so assertions are stable.
async fn plan(table: &Table, name: &str) -> Vec<FileScanTask> {
    let mut tasks: Vec<FileScanTask> = table
        .scan()
        .build()
        .unwrap_or_else(|error| panic!("build scan for {name}: {error}"))
        .plan_files()
        .await
        .unwrap_or_else(|error| panic!("plan_files for {name}: {error}"))
        .try_collect()
        .await
        .unwrap_or_else(|error| panic!("collect plan for {name}: {error}"));
    tasks.sort_by(|left, right| left.data_file_path.cmp(&right.data_file_path));
    tasks
}

/// The LIVE `id` values a merge-on-read scan of the table serves, sorted ascending.
async fn live_ids(table: &Table, name: &str) -> Vec<i64> {
    let batches: Vec<arrow_array::RecordBatch> = table
        .scan()
        .build()
        .unwrap_or_else(|error| panic!("build scan for {name}: {error}"))
        .to_arrow()
        .await
        .unwrap_or_else(|error| panic!("to_arrow for {name}: {error}"))
        .try_collect()
        .await
        .unwrap_or_else(|error| panic!("collect batches for {name}: {error}"));

    let mut ids = Vec::new();
    for batch in &batches {
        let column = batch
            .column_by_name("id")
            .unwrap_or_else(|| panic!("{name}: scan batch has no `id` column"));
        let ids_array = column.as_primitive::<Int64Type>();
        for row in 0..batch.num_rows() {
            assert!(
                !ids_array.is_null(row),
                "{name}: fixture rows all carry a non-null id"
            );
            ids.push(ids_array.value(row));
        }
    }
    ids.sort_unstable();
    ids
}

/// Assert the delete-file attachments of a planned table match the fixture's expected routing.
fn assert_routing(tasks: &[FileScanTask], fixture: &Fixture) {
    let name = fixture.name;
    assert_eq!(
        tasks.len(),
        2,
        "{name}: every fixture holds exactly two data files (two appends)"
    );

    for task in tasks {
        assert_eq!(
            task.deletes.len(),
            1,
            "{name}: data file {} must carry EXACTLY ONE delete file, got {:?}",
            task.data_file_path,
            task.deletes
                .iter()
                .map(|delete| delete.file_path.as_str())
                .collect::<Vec<_>>()
        );
        let delete = &task.deletes[0];
        assert_eq!(
            delete.file_type,
            DataContentType::PositionDeletes,
            "{name}: {} is a position delete",
            delete.file_path
        );
        let task_spec_id = task
            .partition_spec
            .as_ref()
            .unwrap_or_else(|| panic!("{name}: a planned task carries its partition spec"))
            .spec_id();
        assert_eq!(
            delete.partition_spec_id, task_spec_id,
            "{name}: fixture is single-spec, so the delete shares the data file's spec id"
        );
    }

    let attached: BTreeSet<&str> = tasks
        .iter()
        .flat_map(|task| task.deletes.iter().map(|delete| delete.file_path.as_str()))
        .collect();

    match fixture.routing {
        DeleteRouting::FileScopedByBounds => {
            assert_eq!(
                attached.len(),
                tasks.len(),
                "{name}: file-granularity deletes must be a BIJECTION onto the data files \
                 (the pre-routing base broadcast both deletes to both files, giving 1 here); \
                 attached = {attached:?}"
            );
            for task in tasks {
                let delete = &task.deletes[0];
                assert_eq!(
                    delete.file_format,
                    DataFileFormat::Parquet,
                    "{name}: a Spark v2 position delete is parquet, not a DV"
                );
                assert!(
                    delete.referenced_data_file.is_none(),
                    "{name}: {} must be routed by its EQUAL `file_path` BOUNDS — no Java writer \
                     through 1.11.0 sets `referenced_data_file` on a parquet position delete, so a \
                     value here would mean the fixture no longer exercises the bounds leg (got {:?})",
                    delete.file_path,
                    delete.referenced_data_file
                );
            }
        }
        DeleteRouting::PartitionScoped => {
            assert_eq!(
                attached.len(),
                1,
                "{name}: a delete with ABSENT `file_path` bounds stays PARTITION-scoped and must \
                 reach EVERY data file in the partition — narrowing it to one file resurrects rows; \
                 attached = {attached:?}"
            );
            for task in tasks {
                let delete = &task.deletes[0];
                assert_eq!(
                    delete.file_format,
                    DataFileFormat::Parquet,
                    "{name}: parquet"
                );
                assert!(
                    delete.referenced_data_file.is_none(),
                    "{name}: the partition-granularity delete carries no back-reference"
                );
            }
        }
        DeleteRouting::DeletionVector => {
            assert_eq!(
                attached.len(),
                1,
                "{name}: both deletion vectors live in ONE Puffin file; attached = {attached:?}"
            );
            let mut offsets = BTreeSet::new();
            for task in tasks {
                let delete = &task.deletes[0];
                assert_eq!(
                    delete.file_format,
                    DataFileFormat::Puffin,
                    "{name}: a V3 deletion vector is a Puffin blob"
                );
                assert_eq!(
                    delete.referenced_data_file.as_deref(),
                    Some(task.data_file_path.as_str()),
                    "{name}: a DV is routed by its explicit back-reference (the ONLY shape any \
                     Java writer sets `referenced_data_file` on)"
                );
                let offset = delete
                    .content_offset
                    .unwrap_or_else(|| panic!("{name}: a DV carries a Puffin content offset"));
                assert!(
                    offsets.insert(offset),
                    "{name}: the two DVs occupy DISTINCT offsets in the shared Puffin file"
                );
            }
        }
    }
}

/// Every fixture: the planned delete-file attachments match the writer's own granularity.
///
/// This is the assertion the pre-routing base fails on the two equal-bounds fixtures (4 attachments
/// instead of 2) while passing on the partition-granularity and DV fixtures — the differential is
/// entirely in the PLAN, see this module's header.
#[tokio::test]
async fn test_spark_mor_fixture_delete_routing() {
    let Some(root) = fixture_root() else {
        return;
    };

    for fixture in &FIXTURES {
        let table = load_table(&root.join(fixture.relative_path), fixture.name);
        let tasks = plan(&table, fixture.name).await;
        assert_routing(&tasks, fixture);
    }
}

/// Every fixture: the merge-on-read read serves exactly the live rows Spark itself read back.
///
/// Combined with the one-delete-per-data-file assertion above, this is what pins the pairing as the
/// CORRECT one rather than merely a narrowed one: swap the two file-scoped deletes and each data
/// file receives a delete whose records name its sibling, so ids 2 and 6 resurrect.
#[tokio::test]
async fn test_spark_mor_fixture_live_rows() {
    let Some(root) = fixture_root() else {
        return;
    };

    for fixture in &FIXTURES {
        let table = load_table(&root.join(fixture.relative_path), fixture.name);
        let ids = live_ids(&table, fixture.name).await;
        assert_eq!(
            ids,
            EXPECTED_LIVE_IDS.to_vec(),
            "{}: 8 rows written, ids 2 and 6 deleted by a Spark merge-on-read DELETE",
            fixture.name
        );
    }
}

/// `db.mor_legacy_no_field` decodes even though its manifest Avro schema NEVER DECLARED
/// `referenced_data_file`.
///
/// The field was added to the manifest-entry schema around Iceberg 1.7; a 1.4.3 manifest has no
/// such column at all — `grep -a` over the uncompressed Avro header finds zero occurrences of the
/// name. `DataFileSerde::referenced_data_file` is an `Option<String>` with NO `#[serde(default)]`,
/// so this table decodes only via serde's missing-field-to-`None` path over
/// `apache_avro::from_value`. Planning it at all is the proof; asserting `None` on the attachment
/// is the proof that the absence became `None` rather than something else.
#[tokio::test]
async fn test_spark_mor_legacy_manifest_without_referenced_data_file_field() {
    let Some(root) = fixture_root() else {
        return;
    };

    let fixture = &FIXTURES[3];
    assert_eq!(
        fixture.name, "db.mor_legacy_no_field",
        "fixture table order is load-bearing here"
    );

    let table = load_table(&root.join(fixture.relative_path), fixture.name);
    let tasks = plan(&table, fixture.name).await;

    assert_eq!(tasks.len(), 2, "the 1.4.3 fixture plans its two data files");
    for task in &tasks {
        for delete in &task.deletes {
            assert!(
                delete.referenced_data_file.is_none(),
                "a field absent from the manifest AVRO SCHEMA must decode to None, got {:?}",
                delete.referenced_data_file
            );
        }
    }
}
