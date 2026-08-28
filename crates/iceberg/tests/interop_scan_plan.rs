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

//!
//! It proves Rust `TableScan::plan_tasks` produces the same bin-packed `CombinedScanTask` groups as
//! Java `planTasks()`, in both directions. The knobs are hand-declared on both sides, mirroring
//! `InteropOracle.ScanPlanOracle`, so neither side derives them from the other.
//!
//! The fixture is a V2 unpartitioned table, schema `{1 id long required, 2 data string optional}`,
//! built identically by the Java oracle and by the Rust GEN path:
//!
//! | File | Shape | Exercises |
//! |---|---|---|
//! | `big.parquet` | many rows, tiny row groups | the offsets-aware split |
//! | `mid.parquet` | one medium row group | the fixed-size split |
//! | `small1` / `small2` | small | packing two files into one bin |
//! | `big-deletes` | a position delete over `big.parquet` | delete bytes in the pack weight |
//! | `merge.parquet` | 2 row groups, whole file under [`TARGET`] | the adjacent-split merge |
//! | `gap.parquet` | 3 row groups, the middle one over [`TARGET`] | that adjacency is respected |
//!
//! `CombinedScanTask::new` ports Java `BaseCombinedScanTask(List)`, whose `TableScanUtil.mergeTasks`
//! collapses a run of list-adjacent, contiguous splits of one file into one spanning member. The
//! big/mid/small files hit that path by accident only, which made the original failure runner-only.
//! So `merge.parquet` and `gap.parquet` each plan under a metrics-prunable filter over a disjoint
//! `id` range, and over the delete-free APPEND snapshot; [`append_snapshot_id`] says why.
//!
//! The compared plan is the MULTISET of per-group member-key sets plus the group count. A key is
//! `(basename,start,length)`, keyed on the basename because the engines write at different roots.
//! Group emission order is a bin-packer detail and is not compared.
//!
//! `dev/java-interop/run-interop-scan-plan.sh` drives two directions: in D1 Java writes the table and
//! its plan and Rust matches it, and in GEN/D2 Rust writes both and Java judges them. Each test
//! returns early when its env var is unset, so the offline gate stays green without Java.

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

// ===== HAND-DECLARED knobs, mirroring InteropOracle.ScanPlanOracle =====

/// The bin-pack target in bytes (Java `ScanPlanOracle.TARGET`).
const TARGET: u64 = 4096;
/// The planning lookback (Java `ScanPlanOracle.LOOKBACK`).
const LOOKBACK: usize = 5;
/// The per-open file cost in bytes (Java `ScanPlanOracle.OPEN_FILE_COST`).
const OPEN_FILE_COST: u64 = 0;

// ===== HAND-DECLARED merge-fixture shape, mirroring InteropOracle.ScanPlanOracle =====
//
// Both flush every ROW_GROUP_ROWS rows: Rust by row count, parquet-mr at its 100-row floor, which
// the table's 64-byte row-group size reaches at once.

/// Rows per row group for the merge fixtures, so both engines produce the SAME grid.
const ROW_GROUP_ROWS: usize = 100;
/// `merge.parquet` row count ⇒ 2 row groups (100 + 20) with a total length under [`TARGET`].
const MERGE_ROWS: usize = 120;
/// `merge.parquet`'s first row id; its range is `[MERGE_ID_BASE, GAP_ID_BASE)`.
const MERGE_ID_BASE: i64 = 1_000_000;
/// `gap.parquet` row count ⇒ 3 row groups (100 + 100 + 10).
const GAP_ROWS: usize = 210;
/// `gap.parquet`'s first row id; its range is `[GAP_ID_BASE, ..)`.
const GAP_ID_BASE: i64 = 2_000_000;
/// The start of `gap.parquet`'s wide window: its MIDDLE row group, whose span alone exceeds [`TARGET`].
const WIDE_FROM: usize = 100;
/// How many rows carry the wide `data` value (one full row group).
const WIDE_ROWS: usize = 100;
/// Characters per wide value. The middle row group then clears the target by about 6x.
const WIDE_CHARS: usize = 256;

/// The `data` column shape of a fixture file.
#[derive(Clone, Copy)]
enum DataShape {
    /// Every row carries the narrow `row-{i:06}` string.
    Narrow,
    /// Every row carries NULL, so `merge.parquet` stays under the split target.
    NullData,
    /// The wide window carries a high-entropy string, and every other row NULL.
    SparseWide,
}

/// The row-group + content shape of one fixture data file.
#[derive(Clone, Copy)]
struct FixtureShape {
    /// How many rows to write.
    rows: usize,
    /// `None` takes the writer default, which gives one row group here.
    max_row_group_rows: Option<usize>,
    /// Fixture files occupy DISJOINT id ranges, so a metrics-pruned filter isolates one file.
    id_base: i64,
    /// The `data` column shape.
    data: DataShape,
}

/// A deterministic high-entropy [`WIDE_CHARS`]-character string for `row`, mirroring
/// `InteropOracle.ScanPlanOracle.wideValue`.
///
/// The entropy is LOAD-BEARING. The engines use different default codecs, so a low-entropy filler
/// compresses away on the Java side and the middle row group falls below the split target.
fn wide_value(row: usize) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    // `row` is a fixture index below GAP_ROWS, so the cast is on a bounded domain.
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

/// The row filter isolating `merge.parquet` (`ScanPlanOracle.mergeFilter`); its upper bound
/// excludes `gap.parquet`.
fn merge_filter() -> Predicate {
    Reference::new("id")
        .greater_than_or_equal_to(Datum::long(MERGE_ID_BASE))
        .and(Reference::new("id").less_than(Datum::long(GAP_ID_BASE)))
}

/// The row filter that isolates `gap.parquet`, mirroring `ScanPlanOracle.gapFilter`.
fn gap_filter() -> Predicate {
    Reference::new("id").greater_than_or_equal_to(Datum::long(GAP_ID_BASE))
}

// ===== Env gates + the Java plan model =====

/// The dir holding the Java oracle's table and `java_scan_plan.json` (Direction 1).
fn d1_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_SCAN_PLAN_DIR").map(PathBuf::from)
}

/// The dir the GEN path writes the Rust-authored table into, for Java to judge.
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

/// The MULTISET of per-group member-key sets. A `Set` would drop two bins with equal members.
type PlanMultiset = Vec<Vec<String>>;

/// Normalize groups into the comparison form: each group sorted, then the list of groups sorted.
fn normalize(groups: Vec<BTreeSet<String>>) -> PlanMultiset {
    let mut out: PlanMultiset = groups
        .into_iter()
        .map(|group| group.into_iter().collect::<Vec<_>>())
        .collect();
    out.sort();
    out
}

/// The member key for one file-scan task, identical to Java's `memberKey`.
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

/// Run `plan_tasks` with the hand-declared knobs and collect the canonical plan multiset.
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

/// The APPEND snapshot: the fixture state before the MoR position delete. The merge legs plan it,
/// and that choice is LOAD-BEARING. The table is unpartitioned, so a position delete attaches to
/// EVERY data file and its bytes enter every sub-task's weight. Its size tracks the absolute path
/// it embeds, so the co-binning would follow the checkout directory.
fn append_snapshot_id(table: &Table) -> i64 {
    table
        .metadata()
        .current_snapshot()
        .expect("the scan-plan fixture must have a current snapshot")
        .parent_snapshot_id()
        .expect("the scan-plan fixture must have an APPEND snapshot before the row-delta")
}

/// Run `plan_tasks` under a ROW FILTER. It prunes to ONE data file, so the splits meet an EMPTY
/// bin-packer and the co-binning no longer depends on the other files.
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

/// Run the typed [`BatchScan`] `plan_tasks`. It delegates to the same pipeline as
/// [`rust_plan_multiset`], so the tests assert the two multisets are equal.
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

// ===== D1 mismatch diagnostics =====
//
// The D1 member keys ARE `big.parquet`'s row-group offsets, from a version-sensitive parquet-mr
// flush, so this block dumps the upstream facts. It runs on the failure path only and must never
// panic itself.

/// The field-132 `split_offsets` of every data file; an empty vec means absent.
async fn manifest_split_offsets(table: &Table) -> Result<Vec<(String, Vec<i64>)>, String> {
    Ok(manifest_data_files(table)
        .await?
        .into_iter()
        .map(|(name, offsets, _)| (name, offsets))
        .collect())
}

/// The manifest facts the split layer plans from, as `(basename, split_offsets, file_size_in_bytes)`.
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

/// `big.parquet`'s physical row-group layout, read from the parquet footer, not the manifest. A row
/// group starts at its first dictionary page, else its first data page, as Java
/// `BlockMetaData.getStartingPos()` does. A diff against [`manifest_split_offsets`] shows a write
/// whose offsets the physical grid does not support.
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

/// Build the human-readable D1 mismatch report. It never panics.
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

// ===== The adjacent-split MERGE pins =====
//
// The callers assert the two engines agree. These assertions prove they agree on the merge. The
// splitter emits one sub-task per split offset and ignores the target, so `merge.parquet`, with
// `>= 2` offsets, reaches one whole-file member only through the merge. `gap.parquet` must plan two
// groups with the outer pair intact, where a group-by-file coalesce emits one member. The other
// asserts are fixture guards. Stage [7] of `dev/java-interop/run-interop-scan-plan.sh` deletes the
// merge, then its contiguity clause, and requires these assertions to go RED.

/// The `length` of an emitted member key, so a guard reads the plan's own observable.
fn member_length(member: &str) -> u64 {
    member
        .trim_end_matches(')')
        .rsplit(',')
        .next()
        .and_then(|length| length.parse::<u64>().ok())
        .unwrap_or_else(|| panic!("member key {member} must end in a numeric length"))
}

/// The per-row-group byte spans of a data file: the sub-task lengths the offsets-aware split emits.
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

/// Assert the merge pins over the two dedicated fixture files. `label` names the direction.
async fn assert_merge_fixture_pins(
    table: &Table,
    merge_plan: &PlanMultiset,
    gap_plan: &PlanMultiset,
    label: &str,
) {
    // -- merge.parquet: the single spanning member, plus the fixture guards. --
    let (merge_offsets, merge_len) = fixture_file_facts(table, "merge.parquet").await;
    // With >= 2 offsets the splitter emits >= 2 sub-tasks, so only the merge reaches one member.
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
    // The merged span runs from the FIRST split offset, 4 for parquet, to the end of the file.
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

    // Degenerate-fixture guard: the file did not collapse to one row group.
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

    // -- gap.parquet: adjacency, not group-by-file, plus its sizing guards. --
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

    // Degenerate-fixture guard, not the proof: the co-binned pair must be non-contiguous.
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

// ===== Table construction, shared by GEN and the load path =====

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

/// Write a REAL parquet data file at `<table>/data/<basename>`. A small `max_row_group_rows` gives
/// multiple row groups, which exercises the offsets-aware split.
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
        props = props.set_max_row_group_row_count(Some(rows_per_group));
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
        .unpartitioned()
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

// ===== Direction 1: Rust plans the JAVA-written table =====

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

    // A plain `assert_eq!` reports only the two plans, and the D1 evidence is upstream.
    if rust != java {
        let report = d1_mismatch_report(&table, &dir).await;
        panic!(
            "Rust plan_tasks over the Java table must equal Java's planTasks plan (multiset of per-group \
             member-key sets + group count)\n  left  (rust) = {rust:?}\n  right (java) = {java:?}{report}"
        );
    }

    // It must equal Java's `newBatchScan().planTasks()` plan and the plain scan plan, so the
    // adapter delegates to the same pipeline.
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

    // The two isolating filtered plans must also equal Java's, with the merge pins holding.
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

// ===== Direction 2 GEN: Rust writes a Java-judgeable table and emits its own plan =====

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

    // The varying-size fixture the Java oracle builds; the module header gives the shapes.
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
    // The typed BatchScan plan must equal the scan plan, proving the adapter delegates.
    let rust_batch = rust_batch_plan_multiset(&table, TARGET, LOOKBACK, OPEN_FILE_COST).await;
    assert_eq!(
        rust_batch, rust,
        "GEN: Rust BatchScan::plan_tasks must equal Rust TableScan::plan_tasks (adapter delegation)"
    );

    // With one sub-task for the big file, the GEN fixture never exercises the split.
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

    // Assert the merge pins over the RUST-written table, then emit both plans for Java to judge.
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

/// Serialize the Rust plan multiset in the JSON shape Java's `planToJson` emits.
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

// ===== Offline self-test =====
//
// The env-gated tests are no-ops without Java. This one always runs and exercises the comparison
// plumbing, so the gate covers the oracle's own model.

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

    // A duplicate group is PRESERVED: the contract is the multiset, not the set.
    let with_dup = normalize(vec![
        BTreeSet::from(["(a.parquet,0,10)".to_string()]),
        BTreeSet::from(["(a.parquet,0,10)".to_string()]),
    ]);
    assert_eq!(with_dup.len(), 2, "duplicate groups must both count");

    // A different plan must not compare equal: the off-by-one start the sabotage exploits.
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
