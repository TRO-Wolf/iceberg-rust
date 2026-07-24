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

//! Java interop for the SCAN-PLAN layer (`plan_tasks`) — GAP_MATRIX row R148.
//!
//! Proves Rust `TableScan::plan_tasks` produces the SAME bin-packed `CombinedScanTask` GROUPS as Java's
//! REAL `table.newScan().option(SPLIT_SIZE/LOOKBACK/OPEN_FILE_COST, ...).planTasks()`, in BOTH directions,
//! with the target/lookback/open-file-cost HAND-DECLARED IDENTICALLY on both sides (anti-circular — the
//! constants below mirror `InteropOracle.ScanPlanOracle` EXACTLY; neither side derives its knobs from the
//! other).
//!
//! THE FIXTURE (V2, UNPARTITIONED, schema `{1 id long required, 2 data string optional}`), built identically
//! by BOTH the Java oracle and the Rust GEN path: several REAL parquet data files of VARYING size + a MoR
//! position delete, so split + bin-pack are non-trivial:
//!
//! * `big.parquet` — many rows written with a TINY parquet row-group size, so it has MULTIPLE row groups
//!   ⇒ non-null strictly-ascending split offsets ⇒ the OFFSETS-AWARE split fires.
//! * `mid.parquet` — a medium single-row-group file ⇒ FIXED-SIZE split under the small target.
//! * `small1/small2` — two small files that PACK together.
//! * `big-deletes` — a position delete over `big.parquet` so big's sub-tasks carry deletes and the
//!   bin-pack WEIGHT includes the delete bytes.
//! * `merge.parquet` — the ADJACENT-SPLIT MERGE fixture (2 row groups, whole file comfortably UNDER
//!   the split target) — see below.
//! * `gap.parquet` — the ADJACENCY-IS-RESPECTED fixture (3 row groups, the MIDDLE one alone heavier
//!   than the target) — see below.
//!
//! THE MERGE FIXTURES + THEIR ISOLATING FILTERS. `CombinedScanTask::new` ports Java's
//! `BaseCombinedScanTask(List)` constructor, whose bytecode calls `TableScanUtil.mergeTasks`: within ONE
//! bin, a run of LIST-ADJACENT splits of the SAME file that are exactly CONTIGUOUS collapses into one
//! spanning member. The `big/mid/small` fixture above never deterministically puts two adjacent splits of
//! one file in one bin, so it exercises the merge only by accident (that accident — a delete-file path
//! length nudging the pack weights across the 4096 knife edge — is exactly what made the original failure
//! runner-only). Two dedicated files close that coverage hole, each planned under a HAND-DECLARED,
//! metrics-prunable row filter so its splits meet an EMPTY bin-packer and the outcome cannot depend on
//! how the other files happened to pack (the fixture files occupy DISJOINT `id` ranges), and over the
//! delete-free APPEND snapshot (see [`append_snapshot_id`] — the position delete attaches to every file
//! of an unpartitioned table and its size tracks the checkout path length, which is precisely the
//! environment sensitivity these fixtures exist to remove):
//!
//! * `merge.parquet` (ids from [`MERGE_ID_BASE`], filter [`merge_filter`]) — 2 row groups and a TOTAL
//!   LENGTH comfortably below [`TARGET`], so BOTH its splits necessarily share the first bin on any
//!   parquet build and merge into the single spanning member
//!   `(merge.parquet, firstOffset, fileLength - firstOffset)` (`firstOffset` is 4 — parquet's "PAR1"
//!   magic). Note the merged key is INDEPENDENT of the internal row-group grid — merging removes the
//!   environment sensitivity rather than adding it. The sizing is ASSERTED
//!   (`fileLength < TARGET`), so a future fixture edit that pushes it over fails loudly instead of
//!   silently reverting the pin to vacuity. What proves the merge FIRED is the exact
//!   [`assert_eq`](assert_merge_fixture_pins) on the plan SHAPE — ONE group holding ONE member spanning
//!   the whole file — read against the offsets-aware-split invariant: the splitter emits exactly one
//!   sub-task PER SPLIT OFFSET, target-independent (Java `OffsetsAwareSplitScanTaskIterator`, Rust
//!   `FileScanTask::split_at_offsets`), so `>= 2` offsets means `>= 2` splits and a single spanning
//!   member is only producible by the merge.
//! * `gap.parquet` (ids from [`GAP_ID_BASE`], filter [`gap_filter`]) — 3 row groups whose MIDDLE one
//!   carries [`WIDE_ROWS`] high-entropy [`WIDE_CHARS`]-char strings so its span ALONE exceeds
//!   [`TARGET`] (asserted), while the outer two are null-`data` row groups whose spans SUM to under
//!   [`TARGET`] (asserted). First-fit packing therefore puts split 0 in bin 0, split 1 in its own bin
//!   (it fits nowhere), and split 2 back in bin 0 — a CO-BINNED, SAME-FILE, NON-CONTIGUOUS pair that
//!   must NOT merge. Java's `mergeTasks` is a single-pass adjacent-run collapse, not a group-by-file;
//!   this pins that cross-engine instead of only in the offline unit test.
//!
//! THE COMPARISON. Each emitted group is a SORTED set of member keys `(basename,start,length)` (basename =
//! the file's tail — the cross-engine key, since the two engines write at different roots). The plan is the
//! MULTISET of per-group member-key sets + the group count. Rust and Java BOTH plan the SAME on-disk table
//! within a direction, so split offsets (hence start/length) are byte-identical; the set-of-sets + count
//! must match exactly. Group emission ORDER is NOT compared (an internal bin-packer detail).
//!
//! THE TWO DIRECTIONS (driven by `dev/java-interop/run-interop-scan-plan.sh`):
//!
//! * D1 (`ICEBERG_INTEROP_SCAN_PLAN_DIR`): Java writes the table + emits `java_scan_plan.json`; Rust loads
//!   the SAME table, runs `plan_tasks` with the hand-declared knobs, asserts its plan == Java's.
//! * GEN/D2 (`ICEBERG_INTEROP_SCAN_PLAN_GEN_DIR`): Rust WRITES the same logical table to `<dir>/rust_table`
//!   and emits `rust_scan_plan.json`; the Java oracle's `verify-interop-scan-plan` runs the REAL Java
//!   planTasks over the RUST-written table and asserts the SAME plan.
//!
//! THE ENV GATE. Both tests are clean NO-OPs when their env var is unset (a runtime early-return, NOT
//! `#[ignore]`), so the offline `cargo test` gate stays green with no Java/Maven.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use iceberg::expr::{Predicate, Reference};
use iceberg::io::{FileIO, LocalFsStorageFactory};
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::scan::CombinedScanTask;
use iceberg::spec::{
    DataContentType, DataFile, Datum, FormatVersion, NestedField, PrimitiveType, Schema, SortOrder,
    Struct, TableMetadata, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
};
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use serde::Deserialize;

// ===========================================================================================
// HAND-DECLARED knobs (anti-circular — mirror InteropOracle.ScanPlanOracle EXACTLY).
// ===========================================================================================

/// The bin-pack target in bytes (Java `ScanPlanOracle.TARGET`).
const TARGET: u64 = 4096;
/// The planning lookback (Java `ScanPlanOracle.LOOKBACK`).
const LOOKBACK: usize = 5;
/// The per-open file cost in bytes (Java `ScanPlanOracle.OPEN_FILE_COST`).
const OPEN_FILE_COST: u64 = 0;

// ===========================================================================================
// HAND-DECLARED merge-fixture shape (anti-circular — mirrors InteropOracle.ScanPlanOracle EXACTLY).
//
// Both engines flush a row group every ROW_GROUP_ROWS rows: Rust sets the parquet writer's
// max-row-group ROW count directly; parquet-mr flushes at its 100-row memory-check FLOOR because the
// table's `write.parquet.row-group-size-bytes` is a deliberately tiny 64 bytes (the same trick the
// `big.parquet` grid already relies on — see InteropOracle.buildTableWithFiles). So a 120-row file is
// 100 + 20 and a 210-row file is 100 + 100 + 10 on BOTH sides.
// ===========================================================================================

/// Rows per parquet row group for the merge fixtures — parquet-mr's memory-check floor, mirrored as
/// the Rust writer's max-row-group row count so both engines produce the SAME row-group grid.
const ROW_GROUP_ROWS: usize = 100;
/// `merge.parquet` row count ⇒ 2 row groups (100 + 20) with a total length under [`TARGET`].
const MERGE_ROWS: usize = 120;
/// The `id` value of `merge.parquet`'s first row; its whole range is `[MERGE_ID_BASE, GAP_ID_BASE)`.
const MERGE_ID_BASE: i64 = 1_000_000;
/// `gap.parquet` row count ⇒ 3 row groups (100 + 100 + 10).
const GAP_ROWS: usize = 210;
/// The `id` value of `gap.parquet`'s first row; its range is `[GAP_ID_BASE, ..)`.
const GAP_ID_BASE: i64 = 2_000_000;
/// `gap.parquet`'s WIDE row window `[WIDE_FROM, WIDE_FROM + WIDE_ROWS)` — exactly its MIDDLE row
/// group, so that group's span alone exceeds [`TARGET`].
const WIDE_FROM: usize = 100;
/// How many rows carry the wide `data` value (one full row group).
const WIDE_ROWS: usize = 100;
/// Characters per wide `data` value. `WIDE_ROWS * WIDE_CHARS` = 25,600 bytes of HIGH-ENTROPY text,
/// so the middle row group clears the 4,096-byte target by ~6x even after compression, while the
/// per-row-group min/max bounds it puts in the parquet FOOTER (which land in the LAST row group's
/// span) stay small enough that the two outer spans still sum to well under the target.
const WIDE_CHARS: usize = 256;

/// The `data` column shape of a fixture file.
#[derive(Clone, Copy)]
enum DataShape {
    /// Every row carries the narrow `row-{i:06}` string (the original big/mid/small fixture files).
    Narrow,
    /// Every row carries NULL — the smallest possible file for a given row count, which is how
    /// `merge.parquet` stays under the split target on both engines.
    NullData,
    /// Rows in `[WIDE_FROM, WIDE_FROM + WIDE_ROWS)` carry a wide high-entropy string; every other
    /// row carries NULL (keeping `gap.parquet`'s outer row groups small).
    SparseWide,
}

/// The row-group + content shape of one fixture data file.
#[derive(Clone, Copy)]
struct FixtureShape {
    /// How many rows to write.
    rows: usize,
    /// Max rows per parquet row group; `None` ⇒ the writer default (⇒ ONE row group here).
    max_row_group_rows: Option<usize>,
    /// The `id` of the first row — fixture files occupy DISJOINT id ranges so a metrics-pruned
    /// filtered scan can isolate exactly one of them.
    id_base: i64,
    /// The `data` column shape.
    data: DataShape,
}

/// A deterministic HIGH-ENTROPY [`WIDE_CHARS`]-character string for `row` (mirrored EXACTLY by
/// `InteropOracle.ScanPlanOracle.wideValue` — a 32-bit LCG over a 62-character alphabet).
///
/// Entropy is LOAD-BEARING, not decoration: the two engines write parquet with different default
/// codecs (the Rust fixture writer uncompressed, Iceberg's Java default zstd), so a low-entropy filler
/// (zero-padded digits, a repeated character) would compress away on the Java side and collapse
/// `gap.parquet`'s middle row group BELOW the split target — silently reverting the M-4 adjacency pin
/// to vacuity. The span is asserted anyway; the entropy keeps that assertion from being the thing that
/// fails.
fn wide_value(row: usize) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    // `row` is a small fixture index; the cast is on a bounded domain (< GAP_ROWS).
    let mut seed = u32::try_from(row).unwrap_or(0).wrapping_mul(0x9E37_79B1);
    let mut out = String::with_capacity(WIDE_CHARS);
    for _ in 0..WIDE_CHARS {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let index = ((seed >> 16) % 62) as usize;
        out.push(char::from(ALPHABET[index]));
    }
    out
}

/// The `data` value of fixture row `row` under `shape` (`None` ⇒ a NULL `data` cell).
fn fixture_data_value(shape: DataShape, row: usize) -> Option<String> {
    match shape {
        DataShape::Narrow => Some(format!("row-{row:06}")),
        DataShape::NullData => None,
        DataShape::SparseWide if (WIDE_FROM..WIDE_FROM + WIDE_ROWS).contains(&row) => {
            Some(wide_value(row))
        }
        DataShape::SparseWide => None,
    }
}

/// The metrics-prunable row filter that isolates `merge.parquet` (hand-declared; mirrors
/// `InteropOracle.ScanPlanOracle.mergeFilter`). Both bounds are needed so `gap.parquet`'s higher ids
/// are excluded too.
fn merge_filter() -> Predicate {
    Reference::new("id")
        .greater_than_or_equal_to(Datum::long(MERGE_ID_BASE))
        .and(Reference::new("id").less_than(Datum::long(GAP_ID_BASE)))
}

/// The metrics-prunable row filter that isolates `gap.parquet` (hand-declared; mirrors
/// `InteropOracle.ScanPlanOracle.gapFilter`).
fn gap_filter() -> Predicate {
    Reference::new("id").greater_than_or_equal_to(Datum::long(GAP_ID_BASE))
}

// ===========================================================================================
// Env gates + the Java plan model.
// ===========================================================================================

/// The dir the Java oracle wrote its table + `java_scan_plan.json` into (Direction 1).
fn d1_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_SCAN_PLAN_DIR").map(PathBuf::from)
}

/// The dir into which the Direction-2 GEN path writes the Rust-authored table for Java to judge.
fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_SCAN_PLAN_GEN_DIR").map(PathBuf::from)
}

/// Java's emitted plan: `{ groupCount, groups: [[memberKey, ...], ...] }`.
#[derive(Debug, Deserialize)]
struct JavaScanPlan {
    #[serde(rename = "groupCount")]
    group_count: usize,
    groups: Vec<Vec<String>>,
}

/// Strip a path to its basename (the cross-engine comparison key).
fn basename(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_string()
}

/// The MULTISET of per-group member-key sets, as a sorted `Vec` of sorted member `Vec`s. Using a sorted
/// `Vec`-of-`Vec`s (NOT a `Set`-of-`Set`s) preserves DUPLICATE groups — two distinct bins that happen to
/// hold the same member set must both count — which is the faithful multiset contract.
type PlanMultiset = Vec<Vec<String>>;

/// Normalize a set of groups (each a member-key set) into the canonical comparison form: each group sorted,
/// then the list of groups sorted. Order-insensitive across groups, duplicate-preserving.
fn normalize(groups: Vec<BTreeSet<String>>) -> PlanMultiset {
    let mut out: PlanMultiset = groups
        .into_iter()
        .map(|group| group.into_iter().collect::<Vec<_>>())
        .collect();
    out.sort();
    out
}

/// The member key for one file-scan task: `(basename,start,length)` — identical to Java's `memberKey`.
fn member_key(task: &iceberg::scan::FileScanTask) -> String {
    format!(
        "({},{},{})",
        basename(task.data_file_path()),
        task.start(),
        task.length()
    )
}

/// Reduce a list of [`CombinedScanTask`] groups to the canonical comparison multiset.
fn groups_to_multiset(groups: &[CombinedScanTask]) -> PlanMultiset {
    let group_sets: Vec<BTreeSet<String>> = groups
        .iter()
        .map(|group| group.tasks().iter().map(member_key).collect())
        .collect();
    normalize(group_sets)
}

/// Run `plan_tasks` and collect the canonical plan multiset for the table scan built with the hand-declared
/// knobs (target / lookback / open-file-cost set via the builder).
async fn rust_plan_multiset(
    table: &Table,
    target: u64,
    lookback: usize,
    cost: u64,
) -> PlanMultiset {
    let scan = table
        .scan()
        .with_split_size(target)
        .with_split_lookback(lookback)
        .with_split_open_file_cost(cost)
        .build()
        .expect("build scan");
    let groups: Vec<CombinedScanTask> = scan
        .plan_tasks()
        .await
        .expect("plan_tasks")
        .try_collect()
        .await
        .expect("collect groups");
    groups_to_multiset(&groups)
}

/// The APPEND snapshot — the parent of the current one, i.e. the fixture state BEFORE the MoR position
/// delete was committed. The merge legs plan THIS snapshot, and that choice is LOAD-BEARING.
///
/// The table is UNPARTITIONED, so a position delete attaches to EVERY data file in the (single, empty)
/// partition — measured on the Java fixture: `merge.parquet`'s whole-file task comes back with one
/// delete of 1,545 bytes. Those delete bytes enter the bin-pack weight of EVERY sub-task
/// ([`FileScanTask::weight`] / Java `ScanTaskUtil.contentSizeInBytes`), and the delete file's size is
/// dominated by the ABSOLUTE data-file path it embeds — so it varies with the checkout directory.
/// Charging it to each split would put the merge fixture's co-binning back on exactly the
/// environment-sensitive knife edge that made the original failure runner-only (2 x 1,545 is already
/// 75% of the 4,096 target). Planning the delete-free append snapshot makes each split's weight equal
/// its LENGTH — a property of the fixture, not of the filesystem it was written on.
fn append_snapshot_id(table: &Table) -> i64 {
    table
        .metadata()
        .current_snapshot()
        .expect("the scan-plan fixture must have a current snapshot")
        .parent_snapshot_id()
        .expect("the scan-plan fixture must have an APPEND snapshot before the row-delta")
}

/// Run `plan_tasks` under a ROW FILTER and collect the canonical plan multiset, with the same
/// hand-declared knobs. Used by the merge fixtures: the filter is metrics-prunable (each fixture file
/// owns a disjoint `id` range), so the filtered scan plans exactly ONE data file and its splits meet an
/// EMPTY bin-packer — making the co-binning deterministic instead of a function of how the rest of the
/// fixture happened to pack.
async fn rust_filtered_plan_multiset(table: &Table, filter: Predicate) -> PlanMultiset {
    let scan = table
        .scan()
        .snapshot_id(append_snapshot_id(table))
        .with_filter(filter)
        .with_split_size(TARGET)
        .with_split_lookback(LOOKBACK)
        .with_split_open_file_cost(OPEN_FILE_COST)
        .build()
        .expect("build filtered scan");
    let groups: Vec<CombinedScanTask> = scan
        .plan_tasks()
        .await
        .expect("filtered plan_tasks")
        .try_collect()
        .await
        .expect("collect filtered groups");
    groups_to_multiset(&groups)
}

/// Run the typed [`BatchScan`] `plan_tasks` (row R124) and collect the canonical plan multiset, with the
/// SAME hand-declared knobs set via the BatchScan builder. The BatchScan adapter delegates to the same
/// pipeline as [`rust_plan_multiset`], so the two multisets MUST be equal — the tests assert exactly that.
async fn rust_batch_plan_multiset(
    table: &Table,
    target: u64,
    lookback: usize,
    cost: u64,
) -> PlanMultiset {
    let groups: Vec<CombinedScanTask> = table
        .batch_scan()
        .with_split_size(target)
        .with_split_lookback(lookback)
        .with_split_open_file_cost(cost)
        .plan_tasks()
        .await
        .expect("batch_scan plan_tasks")
        .try_collect()
        .await
        .expect("collect batch groups");
    groups_to_multiset(&groups)
}

/// Convert Java's emitted plan into the same canonical multiset form for comparison.
fn java_plan_multiset(plan: &JavaScanPlan) -> PlanMultiset {
    let group_sets: Vec<BTreeSet<String>> = plan
        .groups
        .iter()
        .map(|group| group.iter().cloned().collect())
        .collect();
    let normalized = normalize(group_sets);
    assert_eq!(
        normalized.len(),
        plan.group_count,
        "Java groupCount must equal the number of emitted groups"
    );
    normalized
}

fn read_java_plan(dir: &Path) -> JavaScanPlan {
    read_java_plan_file(dir, "java_scan_plan.json")
}

/// Read a named plan JSON file emitted by the Java oracle into a [`JavaScanPlan`].
fn read_java_plan_file(dir: &Path, file_name: &str) -> JavaScanPlan {
    let path = dir.join(file_name);
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str(&json).unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

// ===========================================================================================
// D1 mismatch diagnostics.
//
// The scan-plan D1 leg is the parity net's most environment-sensitive fixture: the split member keys
// `(basename,start,length)` ARE `big.parquet`'s row-group offsets, and those offsets come from parquet-mr's
// (version-sensitive) row-group flush. When D1 fails in CI, the raw failure carries only the two plans;
// this block dumps the UPSTREAM facts — the manifest field-132 `split_offsets` Rust actually plans from, plus
// the PHYSICAL parquet-footer row-group offsets and `created_by` (the parquet-mr build that wrote the file) —
// so a single CI `tail -40` localizes the fault instead of requiring another round trip. Kept best-effort:
// this code runs ONLY on the failure path, so it must never itself panic (it degrades to `<unavailable: …>`).
// ===========================================================================================

/// The manifest field-132 `split_offsets` Rust reads for every data file in the current snapshot — the exact
/// input to the offsets-aware split. Returned as `(basename, offsets)` rows (empty vec ⇒ field absent).
async fn manifest_split_offsets(table: &Table) -> Result<Vec<(String, Vec<i64>)>, String> {
    Ok(manifest_data_files(table)
        .await?
        .into_iter()
        .map(|(name, offsets, _)| (name, offsets))
        .collect())
}

/// Every file in the current snapshot's manifests as `(basename, split_offsets, file_size_in_bytes)`
/// — the manifest facts the split layer plans from. Backs both the D1 diagnostics above and the
/// merge-fixture non-vacuity pins ([`assert_merge_fixture_pins`]).
async fn manifest_data_files(table: &Table) -> Result<Vec<(String, Vec<i64>, u64)>, String> {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .ok_or_else(|| "no current snapshot".to_string())?;
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), &table.metadata_ref())
        .await
        .map_err(|error| error.to_string())?;
    let mut rows = Vec::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .map_err(|error| error.to_string())?;
        for entry in manifest.entries() {
            let data_file = entry.data_file();
            rows.push((
                basename(data_file.file_path()),
                data_file
                    .split_offsets()
                    .map(<[i64]>::to_vec)
                    .unwrap_or_default(),
                data_file.file_size_in_bytes(),
            ));
        }
    }
    Ok(rows)
}

/// `big.parquet`'s PHYSICAL row-group layout, read straight from the parquet footer (bypassing the manifest):
/// `(created_by, row_group_start_offsets, file_len)`. A row group's start is its first column chunk's
/// dictionary-page offset when present, else its first data-page offset — exactly Java `BlockMetaData
/// .getStartingPos()`, i.e. what iceberg records as field-132. Comparing this to [`manifest_split_offsets`]
/// reveals a write that recorded offsets inconsistent with the physical grid.
fn big_parquet_footer(path: &Path) -> Result<(String, Vec<i64>, u64), String> {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    let file = fs::File::open(path).map_err(|error| error.to_string())?;
    let reader = SerializedFileReader::new(file).map_err(|error| error.to_string())?;
    let metadata = reader.metadata();
    let created_by = metadata
        .file_metadata()
        .created_by()
        .unwrap_or("<none>")
        .to_string();
    let mut offsets = Vec::with_capacity(metadata.num_row_groups());
    for i in 0..metadata.num_row_groups() {
        let column = metadata.row_group(i).column(0);
        offsets.push(
            column
                .dictionary_page_offset()
                .unwrap_or_else(|| column.data_page_offset()),
        );
    }
    let file_len = fs::metadata(path).map_err(|error| error.to_string())?.len();
    Ok((created_by, offsets, file_len))
}

/// Build the human-readable D1 mismatch report (see the section banner above). Never panics.
async fn d1_mismatch_report(table: &Table, dir: &Path) -> String {
    let mut out =
        String::from("\n--- D1 DIAGNOSTICS (upstream facts behind the plan mismatch) ---\n");

    out.push_str("manifest field-132 split_offsets Rust plans from (per data file):\n");
    match manifest_split_offsets(table).await {
        Ok(rows) => {
            for (name, offsets) in rows {
                out.push_str(&format!("    {name}: {offsets:?}\n"));
            }
        }
        Err(error) => out.push_str(&format!("    <unavailable: {error}>\n")),
    }

    let big = dir.join("table/data/big.parquet");
    out.push_str(&format!(
        "big.parquet physical parquet footer ({}):\n",
        big.display()
    ));
    match big_parquet_footer(&big) {
        Ok((created_by, offsets, file_len)) => {
            out.push_str(&format!("    created_by:        {created_by}\n"));
            out.push_str(&format!("    file_len:          {file_len}\n"));
            out.push_str(&format!(
                "    row_group_offsets: {offsets:?} ({} row groups)\n",
                offsets.len()
            ));
        }
        Err(error) => out.push_str(&format!("    <unavailable: {error}>\n")),
    }

    out.push_str(
        "READ THIS AS: both engines split ONE sub-task per field-132 offset, so if field-132 == the footer \
         offsets == Java's emitted plan, the plans MUST match. field-132 != footer ⇒ the write recorded \
         offsets inconsistent with the physical grid; field-132 == footer but != Java's plan ⇒ the emitted \
         java_scan_plan.json does not reflect THIS manifest. `created_by` names the parquet-mr build that \
         wrote big.parquet (differences across environments produce different row-group grids).\n\
         --- end D1 DIAGNOSTICS ---",
    );
    out
}

// ===========================================================================================
// The adjacent-split MERGE pins (M-1 / M-3 / M-4).
//
// These assertions are what keep the merge legs from being vacuous. The cross-engine equality of the
// filtered plans (asserted by the callers) proves the two engines AGREE; the assertions here prove the
// thing they agree on is actually the merge.
//
// WHAT DOES THE PROVING — read this before trusting any single assert below. The load-bearing
// assertions are the two exact `assert_eq`s on the plan SHAPE, interpreted through the
// OFFSETS-AWARE-SPLIT INVARIANT: the splitter emits exactly ONE sub-task PER SPLIT OFFSET and IGNORES
// the target in that branch (Java `OffsetsAwareSplitScanTaskIterator`; Rust
// `FileScanTask::split_at_offsets`). Therefore:
//
//   * `merge.parquet` has `>= 2` split offsets (asserted) ⇒ the splitter MUST emit `>= 2` sub-tasks ⇒
//     a plan of ONE group holding ONE member that spans the whole file is producible ONLY by the merge
//     collapsing them. That equality is the proof.
//   * `gap.parquet` is asserted to plan as EXACTLY two groups with the outer pair intact — two separate
//     same-file members in ONE group. A group-by-file coalesce would emit one member there, so the
//     equality pins the adjacent-run semantics.
//
// The surrounding numeric asserts are FIXTURE guards, not the proof: the SIZING trio (merge total <
// target; gap middle span > target; gap outer spans sum <= target) pins the packing preconditions, and
// the "member length exceeds the largest single row-group span" / "the co-binned pair is
// non-contiguous" checks are degenerate-fixture guards — they catch a file that collapsed to a single
// row group or offsets that stopped ascending. Both are arithmetically implied once the offset counts
// and ascending-offset invariant hold, so neither is doing the discriminating work; do not read them
// as the non-vacuity mechanism. The EXECUTABLE proof that all of this is load-bearing is stage [7] of
// `dev/java-interop/run-interop-scan-plan.sh`, which removes the merge (and, separately, its
// contiguity clause) from production source and requires these assertions to go RED.
// ===========================================================================================

/// The `length` field of an EMITTED member key `(basename,start,length)` — so a guard can read the
/// plan's own observable instead of recomputing it from the manifest facts.
fn member_length(member: &str) -> u64 {
    member
        .trim_end_matches(')')
        .rsplit(',')
        .next()
        .and_then(|length| length.parse::<u64>().ok())
        .unwrap_or_else(|| panic!("member key {member} must end in a numeric length"))
}

/// The per-row-group byte spans of a data file: `offsets[i+1] - offsets[i]`, the last running to the
/// file's end — exactly the sub-task lengths the offsets-aware split emits.
fn row_group_spans(offsets: &[u64], file_size: u64) -> Vec<u64> {
    offsets
        .iter()
        .enumerate()
        .map(|(i, &start)| {
            let end = offsets.get(i + 1).copied().unwrap_or(file_size);
            end.saturating_sub(start)
        })
        .collect()
}

/// The `(split_offsets, file_size_in_bytes)` of the fixture file `name`, as `u64`s.
async fn fixture_file_facts(table: &Table, name: &str) -> (Vec<u64>, u64) {
    let files = manifest_data_files(table)
        .await
        .unwrap_or_else(|error| panic!("read manifest data files: {error}"));
    let (_, offsets, file_size) = files
        .into_iter()
        .find(|(basename, _, _)| basename == name)
        .unwrap_or_else(|| {
            panic!("fixture file {name} must be in the current snapshot's manifests")
        });
    let offsets = offsets
        .into_iter()
        .map(|offset| {
            u64::try_from(offset).unwrap_or_else(|_| {
                panic!("{name}: split offset must be non-negative, got {offset}")
            })
        })
        .collect();
    (offsets, file_size)
}

/// Assert the merge pins over `table`'s two dedicated fixture files, given the plans produced under
/// [`merge_filter`] / [`gap_filter`]. `label` names the direction for the failure message.
async fn assert_merge_fixture_pins(
    table: &Table,
    merge_plan: &PlanMultiset,
    gap_plan: &PlanMultiset,
    label: &str,
) {
    // -- merge.parquet: M-1/M-3, the single spanning member + the fixture guards. --
    let (merge_offsets, merge_len) = fixture_file_facts(table, "merge.parquet").await;
    // The PREMISE of the M-1 proof: with >= 2 split offsets the offsets-aware splitter necessarily
    // emits >= 2 sub-tasks (one per offset, target-ignored), so the single-member plan asserted below
    // can only be the merge's doing.
    assert!(
        merge_offsets.len() >= 2,
        "{label}: merge.parquet must have >= 2 row groups to have anything to merge, got offsets \
         {merge_offsets:?}"
    );
    assert!(
        merge_len < TARGET,
        "{label}: merge.parquet must fit ONE bin so all its splits co-bin on any parquet build — \
         fileLength {merge_len} must be < TARGET {TARGET} (no delete file is attached, so its total \
         bin-pack weight IS its length). Shrink the fixture rather than raising the target."
    );
    let merge_spans = row_group_spans(&merge_offsets, merge_len);
    let largest_merge_span = merge_spans.iter().copied().max().unwrap_or(0);
    // The merged span starts at the FIRST split offset (4 for parquet — after the "PAR1" magic), not
    // at 0, and runs to the end of the file. It is INDEPENDENT of the internal row-group grid.
    let merge_start = merge_offsets[0];
    let merge_span_length = merge_len - merge_start;

    let expected_merge = normalize(vec![BTreeSet::from([format!(
        "(merge.parquet,{merge_start},{merge_span_length})"
    )])]);
    assert_eq!(
        merge_plan,
        &expected_merge,
        "{label}: the merge-filtered plan must be ONE group holding ONE member — the whole file as a \
         single spanning split, i.e. Java's `BaseCombinedScanTask(List)` → `TableScanUtil.mergeTasks` \
         collapse of merge.parquet's {} contiguous splits (spans {merge_spans:?})",
        merge_offsets.len()
    );

    // DEGENERATE-FIXTURE guard (NOT the non-vacuity proof — see the section banner): the length of the
    // member the plan ACTUALLY emitted must exceed the largest single row-group span, i.e. the file did
    // not collapse to one row group. Read out of the emitted plan rather than recomputed, so it
    // describes the observable; given the `>= 2` ascending offsets above it is arithmetically implied,
    // which is exactly why the SHAPE equality above — not this — is what discriminates.
    let emitted_merge_length = merge_plan
        .iter()
        .flat_map(|group| group.iter())
        .map(|member| member_length(member))
        .max()
        .unwrap_or_else(|| panic!("{label}: the merge-filtered plan emitted no members at all"));
    assert!(
        emitted_merge_length > largest_merge_span,
        "{label}: DEGENERATE FIXTURE — the emitted merge.parquet member length \
         {emitted_merge_length} does not exceed its largest single row-group span \
         {largest_merge_span} (spans {merge_spans:?}); the file no longer has two distinct row groups \
         to merge"
    );

    // -- gap.parquet: M-4 (adjacency, not group-by-file) + its sizing guards. --
    let (gap_offsets, gap_len) = fixture_file_facts(table, "gap.parquet").await;
    assert_eq!(
        gap_offsets.len(),
        3,
        "{label}: gap.parquet must have exactly 3 row groups (small / oversized / small), got offsets \
         {gap_offsets:?}"
    );
    let gap_spans = row_group_spans(&gap_offsets, gap_len);
    assert!(
        gap_spans[1] > TARGET,
        "{label}: gap.parquet's MIDDLE row-group span {} must exceed TARGET {TARGET} so its split can \
         never join the bin holding the outer two (spans {gap_spans:?})",
        gap_spans[1]
    );
    assert!(
        gap_spans[0] + gap_spans[2] <= TARGET,
        "{label}: gap.parquet's OUTER row-group spans must SUM to <= TARGET {TARGET} so first-fit \
         co-bins them (spans {gap_spans:?})"
    );

    let outer = BTreeSet::from([
        format!("(gap.parquet,{},{})", gap_offsets[0], gap_spans[0]),
        format!("(gap.parquet,{},{})", gap_offsets[2], gap_spans[2]),
    ]);
    let middle = BTreeSet::from([format!("(gap.parquet,{},{})", gap_offsets[1], gap_spans[1])]);
    let expected_gap = normalize(vec![outer, middle]);
    assert_eq!(
        gap_plan, &expected_gap,
        "{label}: the gap-filtered plan must be TWO groups — the oversized middle split alone, and the \
         two outer splits co-binned but UNMERGED (they are separated by the middle span, so \
         `canMerge`'s `offset + len == next.start` is false). Getting ONE member for the outer pair \
         would mean the merge coalesced by FILE instead of by adjacency (offsets {gap_offsets:?}, \
         spans {gap_spans:?})"
    );

    // DEGENERATE-FIXTURE guard, not the M-4 proof (see the section banner): the co-binned pair must be
    // non-contiguous. Given ascending offsets and a middle span that is asserted `> TARGET` this is
    // arithmetically implied — the discriminating assertion is the two-group SHAPE equality above,
    // whose RED under a group-by-file mutation is proven by stage [7] of the run script.
    assert_ne!(
        gap_offsets[0] + gap_spans[0],
        gap_offsets[2],
        "{label}: the co-binned gap.parquet pair must be NON-CONTIGUOUS for this pin to mean anything"
    );

    println!(
        "    {label} merge pins OK — merge.parquet: {} splits (spans {merge_spans:?}) ⇒ ONE member \
         (merge.parquet,{merge_start},{merge_span_length}) where the splitter must emit one per \
         offset; gap.parquet: spans {gap_spans:?} ⇒ co-binned NON-contiguous outer pair survives \
         unmerged as TWO members",
        merge_offsets.len()
    );
}

// ===========================================================================================
// Table construction (shared by GEN + the load path).
// ===========================================================================================

fn gen_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build the {id long, data string} schema")
}

/// Build a `Table` over the Java-written `final.metadata.json` (local-filesystem `FileIO`).
fn load_table(dir: &Path) -> Table {
    let metadata_path = dir.join("table/metadata/final.metadata.json");
    let json = fs::read_to_string(&metadata_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", metadata_path.display()));
    let metadata: TableMetadata = serde_json::from_str(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", metadata_path.display()));

    Table::builder()
        .metadata(metadata)
        .metadata_location(metadata_path.to_string_lossy().to_string())
        .identifier(TableIdent::from_strs(["interop", "scan_plan"]).expect("valid identifier"))
        .file_io(FileIO::new_with_fs())
        .build()
        .expect("build table from Java-written final.metadata.json")
}

/// Write a REAL parquet data file at `<table>/data/<basename>` with the row-group + content shape
/// `shape`. `max_row_group_rows` drives the parquet writer's max row-group size so a many-row file gets
/// MULTIPLE row groups (hence non-null strictly-ascending split offsets, exercising the offsets-aware
/// split); `id_base` + `data` place the file in its own id range with its own column width.
async fn write_data_file(table: &Table, basename: &str, shape: FixtureShape) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));

    let ids: Vec<i64> = (0..shape.rows)
        .map(|i| shape.id_base + i64::try_from(i).expect("fixture row index fits an i64"))
        .collect();
    let values: Vec<Option<String>> = (0..shape.rows)
        .map(|i| fixture_data_value(shape.data, i))
        .collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(values)) as ArrayRef,
    ])
    .expect("build the data batch");

    let mut props = parquet::file::properties::WriterProperties::builder();
    if let Some(rows_per_group) = shape.max_row_group_rows {
        props = props.set_max_row_group_size(rows_per_group);
    }
    let file_path = format!("{}/data/{}", table.metadata().location(), basename);
    let output = table
        .file_io()
        .new_output(file_path)
        .expect("new parquet output");
    let parquet_builder = ParquetWriterBuilder::new(props.build(), schema.clone());
    let mut writer = parquet_builder
        .build(output)
        .await
        .expect("build parquet writer");
    writer.write(&batch).await.expect("write data batch");
    let data_file_builders = writer.close().await.expect("close parquet writer");

    data_file_builders
        .into_iter()
        .next()
        .expect("one data file builder")
        .content(DataContentType::Data)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .expect("build unpartitioned data file")
}

/// Write a REAL parquet position-delete deleting position 0 of `data_file_path` (unpartitioned).
async fn write_position_delete(table: &Table, data_file_path: &str) -> DataFile {
    let config = PositionDeleteWriterConfig::new().expect("position-delete writer config");
    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "big-deletes".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        iceberg::spec::DataFileFormat::Parquet,
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
        .build(None)
        .await
        .expect("build position-delete writer");

    let paths = StringArray::from(vec![data_file_path]);
    let positions = Int64Array::from(vec![0_i64]);
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(paths) as ArrayRef,
        Arc::new(positions) as ArrayRef,
    ])
    .expect("build the position-delete batch");
    writer
        .write(batch)
        .await
        .expect("write position-delete batch");
    writer
        .close()
        .await
        .expect("close position-delete writer")
        .into_iter()
        .next()
        .expect("one position-delete file")
}

/// Create the unpartitioned V2 table at exactly `table_location`.
async fn create_rust_table(catalog: &impl Catalog, table_location: &str) -> Table {
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, std::collections::HashMap::new())
        .await
        .expect("create namespace");

    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location.to_string())
        .schema(gen_schema())
        .partition_spec(UnboundPartitionSpec::builder().build())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();

    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create rust_table")
}

// ===========================================================================================
// Direction 1 — Rust plans the JAVA-written table.
// ===========================================================================================

#[tokio::test]
async fn test_scan_plan_d1_rust_plans_java_table() {
    let Some(dir) = d1_dir() else {
        println!(
            "skipping interop_scan_plan D1 — set ICEBERG_INTEROP_SCAN_PLAN_DIR \
             (run dev/java-interop/run-interop-scan-plan.sh)"
        );
        return;
    };

    let table = load_table(&dir);
    let rust = rust_plan_multiset(&table, TARGET, LOOKBACK, OPEN_FILE_COST).await;
    let java = java_plan_multiset(&read_java_plan(&dir));

    // A plain `assert_eq!` here would report only the two plans; on the D1 leg the interesting evidence is
    // UPSTREAM (the manifest offsets + the physical row-group grid), so on mismatch we panic with that dumped
    // (see `d1_mismatch_report`). Same failure semantics, far richer CI `tail -40`.
    if rust != java {
        let report = d1_mismatch_report(&table, &dir).await;
        panic!(
            "Rust plan_tasks over the Java table must equal Java's planTasks plan (multiset of per-group \
             member-key sets + group count)\n  left  (rust) = {rust:?}\n  right (java) = {java:?}{report}"
        );
    }

    // BatchScan leg (row R124): Rust `BatchScan::plan_tasks` over the SAME Java table must equal
    //   (a) Java's `newBatchScan().planTasks()` plan (java_batch_scan_plan.json), AND
    //   (b) the plain scan plan (proving the Rust adapter delegates to the same pipeline).
    let rust_batch = rust_batch_plan_multiset(&table, TARGET, LOOKBACK, OPEN_FILE_COST).await;
    let java_batch = java_plan_multiset(&read_java_plan_file(&dir, "java_batch_scan_plan.json"));
    if rust_batch != java_batch {
        let report = d1_mismatch_report(&table, &dir).await;
        panic!(
            "Rust BatchScan::plan_tasks over the Java table must equal Java's newBatchScan().planTasks() \
             plan\n  left  (rust_batch) = {rust_batch:?}\n  right (java_batch) = {java_batch:?}{report}"
        );
    }
    assert_eq!(
        rust_batch, rust,
        "Rust BatchScan::plan_tasks must equal Rust TableScan::plan_tasks (adapter delegation)"
    );

    // MERGE legs (M-1 / M-3 / M-4): the two isolating filtered plans must ALSO equal Java's, and the
    // merge pins must hold over the Java-written table.
    let rust_merge = rust_filtered_plan_multiset(&table, merge_filter()).await;
    let java_merge = java_plan_multiset(&read_java_plan_file(&dir, "java_merge_scan_plan.json"));
    assert_eq!(
        rust_merge, java_merge,
        "Rust plan_tasks under the merge filter must equal Java's planTasks plan — this is the \
         adjacent-split merge (Java `BaseCombinedScanTask(List)` → `TableScanUtil.mergeTasks`)"
    );
    let rust_gap = rust_filtered_plan_multiset(&table, gap_filter()).await;
    let java_gap = java_plan_multiset(&read_java_plan_file(&dir, "java_gap_scan_plan.json"));
    assert_eq!(
        rust_gap, java_gap,
        "Rust plan_tasks under the gap filter must equal Java's planTasks plan — this is the \
         co-binned-but-NON-CONTIGUOUS same-file pair that must NOT merge"
    );
    assert_merge_fixture_pins(&table, &rust_merge, &rust_gap, "D1").await;

    println!(
        "interop_scan_plan D1 OK — Rust plan_tasks AND BatchScan::plan_tasks over the Java table match \
         Java ({} groups), and so do the merge-filtered ({} groups) + gap-filtered ({} groups) plans",
        rust.len(),
        rust_merge.len(),
        rust_gap.len()
    );
}

// ===========================================================================================
// Direction 2 GEN — Rust writes a Java-judgeable table + emits its own plan.
// ===========================================================================================

#[tokio::test]
async fn test_scan_plan_gen_rust_writes_java_judgeable_table() {
    let Some(gen_dir) = gen_dir() else {
        println!(
            "skipping interop_scan_plan GEN — set ICEBERG_INTEROP_SCAN_PLAN_GEN_DIR \
             (run dev/java-interop/run-interop-scan-plan.sh)"
        );
        return;
    };

    let warehouse = gen_dir.to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_scan_plan_gen",
            std::collections::HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                warehouse.clone(),
            )]),
        )
        .await
        .expect("build MemoryCatalog over local FS");

    let table = create_rust_table(&catalog, &table_location).await;

    // The same varying-size fixture the Java oracle builds: big (multi-row-group), mid, small1, small2,
    // plus the two dedicated merge fixtures (merge = contiguous co-binned pair, gap = non-contiguous
    // co-binned pair). See the module header for their shapes.
    let narrow = |rows: usize, max_row_group_rows: Option<usize>| FixtureShape {
        rows,
        max_row_group_rows,
        id_base: 0,
        data: DataShape::Narrow,
    };
    let big = write_data_file(&table, "big.parquet", narrow(800, Some(64))).await;
    let big_path = big.file_path().to_string();
    let mid = write_data_file(&table, "mid.parquet", narrow(40, None)).await;
    let small1 = write_data_file(&table, "small1.parquet", narrow(5, None)).await;
    let small2 = write_data_file(&table, "small2.parquet", narrow(5, None)).await;
    let merge = write_data_file(&table, "merge.parquet", FixtureShape {
        rows: MERGE_ROWS,
        max_row_group_rows: Some(ROW_GROUP_ROWS),
        id_base: MERGE_ID_BASE,
        data: DataShape::NullData,
    })
    .await;
    let gap = write_data_file(&table, "gap.parquet", FixtureShape {
        rows: GAP_ROWS,
        max_row_group_rows: Some(ROW_GROUP_ROWS),
        id_base: GAP_ID_BASE,
        data: DataShape::SparseWide,
    })
    .await;

    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![big, mid, small1, small2, merge, gap])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");

    // A MoR position delete over big.parquet (position 0).
    let delete_file = write_position_delete(&table, &big_path).await;
    assert_eq!(delete_file.content_type(), DataContentType::PositionDeletes);
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // Compute OUR OWN plan and emit it for Java to verify.
    let rust = rust_plan_multiset(&table, TARGET, LOOKBACK, OPEN_FILE_COST).await;
    // The typed BatchScan plan over the SAME Rust table must equal the scan plan (adapter delegation),
    // and is emitted for the Java verify's BatchScan leg.
    let rust_batch = rust_batch_plan_multiset(&table, TARGET, LOOKBACK, OPEN_FILE_COST).await;
    assert_eq!(
        rust_batch, rust,
        "GEN: Rust BatchScan::plan_tasks must equal Rust TableScan::plan_tasks (adapter delegation)"
    );

    // SANITY: the big file MUST have split into more than one sub-task (multi-row-group ⇒ offsets-aware
    // split), otherwise the offsets-aware branch is not actually exercised by the GEN fixture.
    let big_sub_tasks: usize = rust
        .iter()
        .flat_map(|group| group.iter())
        .filter(|member| member.starts_with("(big.parquet,"))
        .count();
    assert!(
        big_sub_tasks > 1,
        "GEN sanity: big.parquet must split into >1 sub-task (got {big_sub_tasks}); the tiny row-group \
         size should produce multiple row groups + split offsets"
    );

    // Write the FINAL metadata for Java to load, and the Rust plan for Java to compare.
    let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), &final_metadata_path)
        .await
        .expect("write final.metadata.json");

    let plan_json = rust_plan_to_json(&rust);
    let plan_path = format!("{table_location}/rust_scan_plan.json");
    fs::write(&plan_path, plan_json).expect("write rust_scan_plan.json");

    // Emit the BatchScan plan for the Java verify's BatchScan leg (row R124).
    let batch_plan_json = rust_plan_to_json(&rust_batch);
    let batch_plan_path = format!("{table_location}/rust_batch_scan_plan.json");
    fs::write(&batch_plan_path, batch_plan_json).expect("write rust_batch_scan_plan.json");

    // MERGE legs (M-2 / M-3 / M-4): plan the two isolating filtered scans, assert the merge pins hold
    // over the RUST-written table, and emit both plans for the Java verify to judge.
    let rust_merge = rust_filtered_plan_multiset(&table, merge_filter()).await;
    let rust_gap = rust_filtered_plan_multiset(&table, gap_filter()).await;
    assert_merge_fixture_pins(&table, &rust_merge, &rust_gap, "GEN").await;
    fs::write(
        format!("{table_location}/rust_merge_scan_plan.json"),
        rust_plan_to_json(&rust_merge),
    )
    .expect("write rust_merge_scan_plan.json");
    fs::write(
        format!("{table_location}/rust_gap_scan_plan.json"),
        rust_plan_to_json(&rust_gap),
    )
    .expect("write rust_gap_scan_plan.json");

    println!(
        "interop_scan_plan GEN OK — Rust wrote {table_location} (big/mid/small1/small2 + a position \
         delete + merge/gap) and emitted rust_scan_plan.json + rust_batch_scan_plan.json + \
         rust_merge_scan_plan.json + rust_gap_scan_plan.json ({} groups, big split into \
         {big_sub_tasks} sub-tasks). Java verify-interop-scan-plan runs the REAL planTasks (scan + \
         batchScan + the two filtered merge plans) over it next.",
        rust.len()
    );
}

/// Serialize the Rust plan multiset as `{groupCount, groups:[[memberKey,...],...]}` for the Java verify
/// (the SAME JSON shape Java's `planToJson` emits — each group sorted, the group list sorted).
fn rust_plan_to_json(plan: &PlanMultiset) -> String {
    // `plan` is already normalized (each group sorted, list sorted).
    let groups: Vec<serde_json::Value> = plan
        .iter()
        .map(|group| serde_json::Value::Array(group.iter().map(|m| m.clone().into()).collect()))
        .collect();
    serde_json::json!({
        "groupCount": plan.len(),
        "groups": groups,
    })
    .to_string()
}

// ===========================================================================================
// Offline self-test — the env-gated tests are no-ops without Java; this one runs ALWAYS and exercises
// the comparison plumbing (normalize / multiset equality) so the gate has live coverage of the oracle's
// own model independent of Java.
// ===========================================================================================

#[test]
fn normalize_is_order_insensitive_across_groups_and_duplicate_preserving() {
    // Two groups in either order normalize to the same canonical form.
    let a = normalize(vec![
        BTreeSet::from(["(b.parquet,0,10)".to_string()]),
        BTreeSet::from(["(a.parquet,0,10)".to_string()]),
    ]);
    let b = normalize(vec![
        BTreeSet::from(["(a.parquet,0,10)".to_string()]),
        BTreeSet::from(["(b.parquet,0,10)".to_string()]),
    ]);
    assert_eq!(a, b, "group order must not matter");

    // A duplicate group (same member set) is PRESERVED — the multiset, not the set, is the contract.
    let with_dup = normalize(vec![
        BTreeSet::from(["(a.parquet,0,10)".to_string()]),
        BTreeSet::from(["(a.parquet,0,10)".to_string()]),
    ]);
    assert_eq!(with_dup.len(), 2, "duplicate groups must both count");

    // A genuinely different plan does NOT compare equal (the off-by-one start that the sabotage exploits).
    let shifted = normalize(vec![BTreeSet::from(["(a.parquet,1,10)".to_string()])]);
    let original = normalize(vec![BTreeSet::from(["(a.parquet,0,10)".to_string()])]);
    assert_ne!(shifted, original, "a shifted start must diverge");

    // Round-trip through the JSON the Java verify reads, then parse it back as a Java plan.
    let plan = normalize(vec![
        BTreeSet::from([
            "(a.parquet,0,10)".to_string(),
            "(b.parquet,0,20)".to_string(),
        ]),
        BTreeSet::from(["(c.parquet,0,30)".to_string()]),
    ]);
    let json = rust_plan_to_json(&plan);
    let parsed: JavaScanPlan = serde_json::from_str(&json).expect("round-trip JSON");
    assert_eq!(
        java_plan_multiset(&parsed),
        plan,
        "JSON round-trip must be lossless"
    );
}
