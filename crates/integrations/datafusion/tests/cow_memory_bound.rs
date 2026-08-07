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

//! H7-S2 — evidence that the **copy-on-write** DELETE/UPDATE paths do NOT hold the live row set.
//!
//! # Why this binary exists
//!
//! The existing COW suites assert *which rows survive*. Those assertions are byte-identical before
//! and after the streaming refactor, so they pass either way — a green run there is not evidence of
//! the memory claim. This test is designed to be **RED against the buffered code and GREEN against
//! the streaming code**; the mutation numbers are recorded in
//! `task/v1-h7-s2-cow-streaming-ledger.md`.
//!
//! # The measurement
//!
//! A counting `#[global_allocator]`. A `tests/*.rs` file compiles to its own binary and runs in its
//! own process, so the allocator has **zero blast radius** on the library or the other test binaries.
//!
//! The assertion is **marginal, never absolute** — the same DML runs at `N` and `4N` rows and the
//! *difference* in peak is what is asserted, so every constant overhead (tokio, the planner, the
//! catalog) cancels and there is no magic byte threshold:
//!
//! ```text
//! [peak(4N) − peak(N)]  −  baseline_delta   <   ¼ × (added live bytes)
//! ```
//!
//! * **Buffered (pre-H7-S2):** the whole live row set is held in `batches`, and the survivor/rewrite
//!   vector is live at the same time, so the excess tracks the added volume and fails by a wide
//!   margin.
//! * **Streaming (now):** the excess is a `HashSet<String>` of affected file paths plus one batch.
//!   Measured at a few KB against a ~10 MB threshold — three orders of magnitude of headroom, which
//!   is what makes the verdict robust to the counters being `Relaxed` and the runtime interleaving.
//!
//! ## Why `baseline_delta` is subtracted — measured, not assumed
//!
//! The naive form (no subtraction) does **not** work here, and the reason is worth recording because
//! it is not obvious: the **Iceberg scan itself** has memory that grows with table size. `to_arrow`
//! reads up to `concurrency_limit_data_files` files at once — defaulting to `num_cpus`, which is 64
//! on the development machine — and materializes a Parquet row group for each, on top of per-file
//! scan-task overhead. Measured on the streaming code, that term alone was 4–58 MB depending on the
//! fixture's file layout, i.e. large enough to blow a ¼ × threshold on its own and to vary with the
//! host's core count. It is present identically whether copy-on-write buffers or streams.
//!
//! `baseline_delta` is that term, measured directly: the same marginal quantity for a zero-match
//! **merge-on-read** DELETE over an identical fixture. Merge-on-read already streams (H7-S1) and is
//! untouched by this unit; matching nothing, it writes no file and commits nothing, so what it
//! measures is the scan. Subtracting it is sound against the mutation this test exists to catch —
//! reinstating the `try_collect` buffer changes ONLY the copy-on-write path, moving the measured
//! delta by a whole live row set while leaving the baseline exactly where it was.
//!
//! # Conditions on the claim (deliberate, and NOT incidental)
//!
//! 1. **The fixture is UNPARTITIONED.** `StreamingDataFileWriter` wraps a `TaskWriter` with
//!    `fanout_enabled = true`, which holds one open file writer per partition until close — so
//!    partition cardinality is a **second, independent** memory axis that this unit does not address.
//!    Holding it at zero is what isolates the row-count claim. High-cardinality partitioned writes
//!    remain a genuine unbounded-writer question (follow-up: the QB writer-bounds unit).
//! 2. **File SIZE is constant across scales and exactly ONE file is affected.** A *third* axis, also
//!    out of scope: the Parquet writer accumulates a row group before flushing, so peak scales with
//!    rows **WRITTEN** too. Measured, not assumed — with 90% of rows rewritten the marginal delta was
//!    6.7 MB, and COW UPDATE's delta was byte-identical under two different predicates precisely
//!    because COW UPDATE rewrites *every* row of an affected file either way. Holding
//!    [`ROWS_PER_FILE`] constant and confining the match to file 0 makes the written volume identical
//!    at both scales, so it cancels. Writer-side row-group buffering is real and still unbounded —
//!    follow-up, same QB unit.
//! 3. **One test function.** The counters are process-global, so anything running concurrently in
//!    this binary would pollute them. The seven measured runs are sequential inside one test.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use datafusion::arrow::array::{Int32Array, RecordBatch, StringArray, UInt64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;
use tempfile::TempDir;

// ===========================================================================
// The counting global allocator
// ===========================================================================

/// Bytes currently live (allocated minus freed). Approximate under concurrency — `Relaxed` ordering
/// is deliberate; see the module note on the order-of-magnitude margin.
static LIVE: AtomicUsize = AtomicUsize::new(0);
/// High-water mark of `LIVE`.
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: every method forwards to `System`, the platform allocator, with identical pointers and
// layouts; the atomics are pure bookkeeping and do not affect the returned allocations.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    /// `realloc` is accounted **explicitly** rather than being left to the `GlobalAlloc` default.
    ///
    /// The default provided impl would route through our `alloc` + `dealloc` (so it would also be
    /// correct), but it always allocates a fresh block and memcpys — for the many growing `Vec`s on
    /// this path that is materially slower than `System.realloc`, which can often extend in place.
    /// We therefore delegate to `System.realloc` and adjust the counters by the size delta. Growth
    /// updates the peak; shrink only lowers `LIVE`. This under-counts the transient moment when both
    /// blocks are briefly live inside a copying `System.realloc` — an under-count is the SAFE
    /// direction here: it can only make the measured peak smaller, i.e. make the buffered form look
    /// better, never make the streaming form look better than it is.
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            let old = layout.size();
            if new_size >= old {
                let live = LIVE.fetch_add(new_size - old, Ordering::Relaxed) + (new_size - old);
                PEAK.fetch_max(live, Ordering::Relaxed);
            } else {
                LIVE.fetch_sub(old - new_size, Ordering::Relaxed);
            }
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Snapshot the current live total and re-arm the high-water mark from it. Returns the baseline to
/// hand to [`end_measure`].
fn begin_measure() -> usize {
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    base
}

/// The measured region's own marginal high-water: how much MORE was live at the peak than at the
/// moment measurement started. Everything allocated before `begin_measure` (the loaded fixture, the
/// catalog, the runtime) is in the baseline and cancels out.
fn end_measure(base: usize) -> usize {
    PEAK.load(Ordering::Relaxed).saturating_sub(base)
}

// ===========================================================================
// Fixture — an UNPARTITIONED table with `rows` rows, loaded via INSERT ... SELECT
// ===========================================================================

/// Length of the `payload` string column. With the `id` int this fixes the per-row volume.
const PAYLOAD_LEN: usize = 100;

/// A deliberately CONSERVATIVE lower bound on the in-memory Arrow footprint of one scanned row:
/// `id` (i32, 4 B) + the `payload` bytes + one 4-byte offset. The real footprint is larger (validity
/// bitmaps, the `_file` column, batch/array overhead), and under-stating it makes the threshold
/// SMALLER, i.e. the assertion STRICTER. Under-stating can therefore only cost a false RED on the
/// streaming path (which measures ~3 orders of magnitude of headroom), never a false GREEN on the
/// buffered path.
const ROW_BYTES_FLOOR: usize = 4 + PAYLOAD_LEN + 4;

/// A temp directory and its path, with the GUARD RETURNED so the caller can keep it alive and let it
/// clean up.
///
/// Deliberately NOT the `temp_path()` idiom used elsewhere in this crate's tests (which drops the
/// guard immediately and leaks the recreated directory): this binary writes ~490 MB of Parquet per
/// run across seven fixtures, so leaking would be a per-invocation, permanent cost. The guards travel
/// out of [`setup`] and are dropped AFTER the measured region closes, so the recursive delete's own
/// allocations cannot land in the measurement.
fn temp_dir() -> (String, TempDir) {
    let dir = TempDir::new().expect("create temp dir");
    let path = dir
        .path()
        .to_str()
        .expect("temp dir path is valid UTF-8")
        .to_string();
    (path, dir)
}

/// Rows per data file — **held CONSTANT across both scales**, so the two runs differ only in how
/// many files there are, never in how big one file is. See condition 2 in the module note: a Parquet
/// reader materializes a whole row group per open file, and the writer accumulates a row group
/// before flushing, so per-FILE size is its own memory axis. Pinning it is what isolates the axis
/// this unit changed — memory proportional to the number of rows SCANNED.
const ROWS_PER_FILE: usize = 2_000;

/// Data files in a `rows`-row fixture — one `INSERT` per file.
fn files(rows: usize) -> usize {
    rows / ROWS_PER_FILE
}

/// Create `catalog.<ns>.t` = `{id int required, payload string required}`, **unpartitioned**, with
/// EMPTY table properties — an absent `write.delete.mode` / `write.update.mode` resolves to
/// copy-on-write, which is the path under test.
///
/// Loads `rows` rows as `(i, "<zero-padded i>")` into `rows / ROWS_PER_FILE` data files holding
/// **disjoint, contiguous** `id` ranges: file `k` holds `id ∈ [k·ROWS_PER_FILE, (k+1)·ROWS_PER_FILE)`.
/// That disjointness is what lets a predicate over `id` select a known set of FILES, not just a known
/// set of rows.
///
/// `merge_on_read` selects the table's row-level write mode: `false` (empty properties) is
/// copy-on-write, the path under test; `true` builds the merge-on-read baseline fixture.
///
/// Returns the context, the number of live data files (which the caller pins — the memory margin
/// depends on the fixture really being multi-file, so it must be verified, not assumed), and the
/// `TempDir` guards, which the caller keeps alive for the DML and drops afterwards.
async fn setup(
    ns: &str,
    rows: usize,
    merge_on_read: bool,
) -> (SessionContext, usize, Vec<TempDir>) {
    let (warehouse_path, warehouse_dir) = temp_dir();
    let (table_path, table_dir) = temp_dir();

    let iceberg_catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path)]),
        )
        .await
        .expect("build memory catalog");

    let namespace = NamespaceIdent::new(ns.to_string());
    iceberg_catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "payload", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build schema");

    let creation = TableCreation::builder()
        .name("t".to_string())
        .location(table_path)
        .schema(schema)
        // Empty properties ⇒ copy-on-write for both DELETE and UPDATE. The CONTROL table instead
        // asks for merge-on-read, which streams (H7-S1) and is untouched by this unit.
        .properties(if merge_on_read {
            HashMap::from([
                ("write.delete.mode".to_string(), "merge-on-read".to_string()),
                ("write.update.mode".to_string(), "merge-on-read".to_string()),
            ])
        } else {
            HashMap::new()
        })
        .build();
    iceberg_catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table");

    let client = Arc::new(iceberg_catalog);
    let provider = Arc::new(
        IcebergCatalogProvider::try_new(client.clone())
            .await
            .expect("build catalog provider"),
    );
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", provider);

    // The source is registered with ONE DataFusion partition per intended data file, and loaded in a
    // SINGLE `INSERT` — the write plan emits one data file per input partition. (Looping one INSERT
    // per file also works but costs one Iceberg commit each, which dominates the runtime at the file
    // counts this test needs.) The `files_before` count returned below verifies the result.
    //
    // Payload values are DISTINCT per row, on purpose: a constant filler would dictionary-encode to
    // almost nothing in Parquet AND could come back from the scan as a dictionary/REE array, so the
    // buffered form would not actually hold PAYLOAD_LEN bytes per row and ROW_BYTES_FLOOR would be a
    // wild OVER-estimate — a loose threshold, the dangerous direction. Unique values force real bytes
    // end to end.
    let pad = PAYLOAD_LEN - 1;
    let src_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let partitions: Vec<Vec<RecordBatch>> = (0..files(rows))
        .map(|file| {
            let lo = file * ROWS_PER_FILE;
            let ids: Vec<i32> = (lo..lo + ROWS_PER_FILE)
                .map(|i| i32::try_from(i).expect("row index fits in i32"))
                .collect();
            let payloads: Vec<String> = (lo..lo + ROWS_PER_FILE)
                .map(|i| format!("{i:0>pad$}#"))
                .collect();
            let batch = RecordBatch::try_new(src_schema.clone(), vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(payloads)),
            ])
            .expect("build source batch");
            vec![batch]
        })
        .collect();
    let src = MemTable::try_new(src_schema.clone(), partitions).expect("build MemTable");
    ctx.register_table("src", Arc::new(src))
        .expect("register source table");

    ctx.sql(&format!(
        "INSERT INTO catalog.{ns}.t SELECT id, payload FROM src"
    ))
    .await
    .expect("plan insert")
    .collect()
    .await
    .expect("run insert");

    // Drop the in-memory source so its rows are NOT live during the measured DML.
    ctx.deregister_table("src")
        .expect("deregister source table");

    // Count the live data files, so the multi-file precondition of the margin is VERIFIED.
    let table = client
        .load_table(&TableIdent::new(
            NamespaceIdent::new(ns.to_string()),
            "t".to_string(),
        ))
        .await
        .expect("load table");
    let metadata = table.metadata();
    let snapshot = metadata.current_snapshot().expect("table has a snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), metadata)
        .await
        .expect("load manifest list");
    let mut data_files = 0usize;
    for manifest_entry in manifest_list.entries() {
        if manifest_entry.content != iceberg::spec::ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_entry
            .load_manifest(table.file_io())
            .await
            .expect("load manifest");
        data_files += manifest.entries().iter().filter(|e| e.is_alive()).count();
    }

    (ctx, data_files, vec![warehouse_dir, table_dir])
}

/// One measured run.
struct Measured {
    /// Marginal high-water bytes over the run's own baseline.
    peak: usize,
    /// The DML's reported row count.
    affected: u64,
    /// Live data files in the table immediately BEFORE the DML.
    files_before: usize,
}

/// Run one DML statement against a freshly loaded `rows`-row table.
async fn measure_dml(ns: &str, rows: usize, sql: &str) -> Measured {
    measure_dml_mode(ns, rows, sql, false).await
}

/// As [`measure_dml`], but selects the table's row-level write mode.
async fn measure_dml_mode(ns: &str, rows: usize, sql: &str, merge_on_read: bool) -> Measured {
    let (ctx, files_before, temp_dirs) = setup(ns, rows, merge_on_read).await;

    let base = begin_measure();
    let batches = ctx.sql(sql).await.expect("plan dml").collect().await;
    let peak = end_measure(base);

    let batches = batches.expect("run dml");
    let affected = batches
        .first()
        .expect("dml returns a count batch")
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("dml count column is UInt64")
        .value(0);

    // Fixture teardown, OUTSIDE the measured region (`end_measure` has already read the high-water
    // mark, so the recursive delete's allocations cannot affect the verdict). Dropping the context
    // first releases the catalog's handles on the files before the directories go.
    drop(ctx);
    drop(temp_dirs);

    Measured {
        peak,
        affected,
        files_before,
    }
}

// ===========================================================================
// The test
// ===========================================================================

/// Rows in the small run. The large run is `4 ×` this.
const N: usize = 128_000;

/// The `WHERE` clause used by every measured run. It does NOT depend on the scale.
///
/// `id < ROWS_PER_FILE` confines the match to **file 0** (ids are disjoint and contiguous per file);
/// `id % 2 = 0` then matches half of that file's rows, so file 0 has both matching and surviving
/// rows and the writer is genuinely exercised. Because file size is constant across scales, the
/// matched-row count and the rewritten volume are *identical* at both scales — the only thing that
/// grows is the number of rows SCANNED, which is exactly the quantity under test.
fn where_clause() -> String {
    format!("id < {ROWS_PER_FILE} AND id % 2 = 0")
}

/// Rows [`where_clause`] matches — constant across scales, by construction.
const MATCHED_ROWS: u64 = (ROWS_PER_FILE / 2) as u64;

/// Assert the marginal-peak bound for one DML shape.
///
/// `baseline_delta` is the SAME marginal quantity measured for a **merge-on-read** zero-match DELETE
/// over an identical fixture — a path that streams (H7-S1) and that this unit does not touch. It is
/// subtracted because the Iceberg scan has its own memory that grows with table size and is nobody's
/// bug here: `to_arrow` reads up to `concurrency_limit_data_files` (defaulting to `num_cpus`) data
/// files at once and materializes a Parquet row group for each, plus per-file scan-task overhead.
/// That term is present identically in both paths, so subtracting it leaves the quantity actually
/// under test: **the memory copy-on-write adds on top of streaming the same rows.**
///
/// Subtracting is sound against the mutation this test exists to catch: reinstating the `try_collect`
/// buffer changes ONLY the copy-on-write path, so it moves `delta` by the whole live row set while
/// leaving `baseline_delta` untouched.
fn assert_marginal_bound(
    label: &str,
    peak_small: usize,
    peak_large: usize,
    baseline_delta: usize,
    added_bytes: usize,
) {
    let threshold = added_bytes / 4;
    let delta = peak_large.saturating_sub(peak_small);
    let excess = delta.saturating_sub(baseline_delta);
    assert!(
        excess < threshold,
        "{label}: copy-on-write is holding the live row set.\n  \
         peak(N={N}) = {peak_small} B\n  \
         peak(4N={large}) = {peak_large} B\n  \
         delta = {delta} B\n  \
         merge-on-read baseline delta = {baseline_delta} B (the scan's own growth)\n  \
         excess over baseline = {excess} B\n  \
         added live bytes = {added_bytes} B (floor estimate)\n  \
         threshold (¼ × added) = {threshold} B\n  \
         A streaming copy-on-write adds a file-path set plus one batch over the baseline, which is \
         orders of magnitude under the threshold; an excess at or above the added volume means the \
         rows are being buffered.",
        large = 4 * N,
    );
}

/// Pin that a measured run did the work it was supposed to: the right number of rows matched, and
/// the fixture really was multi-file. Without this, a peak comparison could be comparing two no-ops.
fn assert_fixture(label: &str, run: &Measured, rows: usize) {
    assert_eq!(
        run.affected, MATCHED_ROWS,
        "{label}: wrong number of rows affected — the fixture or predicate is not what the memory \
         comparison assumes"
    );
    assert_eq!(
        run.files_before,
        files(rows),
        "{label}: the fixture must be one data file per INSERT of {ROWS_PER_FILE} rows; the margin \
         depends on file SIZE being constant across scales and only ONE file being affected"
    );
}

/// Both copy-on-write paths must hold memory that does NOT scale with the live row count.
///
/// One test function on purpose: the allocator counters are process-global, so the runs must be
/// sequential and nothing else may be in flight. NOT `#[ignore]`d — an ignored memory test is a
/// false-green in CI, which is the exact failure mode this unit exists to close.
#[tokio::test]
async fn copy_on_write_peak_memory_does_not_scale_with_row_count() {
    // The added live volume between the two scales.
    let added_bytes = 3 * N * ROW_BYTES_FLOOR;

    // ---- Warm-up, DISCARDED. Pays the one-time init (tokio, the DataFusion planner, parquet's
    // ---- statics, the object-store client) before either measurement, so it cannot land in one
    // ---- scale's peak and not the other's.
    let warm_rows = 2 * ROWS_PER_FILE;
    let warm = measure_dml(
        "cow_mem_warmup",
        warm_rows,
        &format!(
            "DELETE FROM catalog.cow_mem_warmup.t WHERE {}",
            where_clause()
        ),
    )
    .await;
    assert_fixture("warm-up", &warm, warm_rows);

    // ---- BASELINE: how much the SCAN's own peak grows between the two scales, measured on a path
    // ---- that already streams and that this unit does not touch — a zero-match MERGE-ON-READ
    // ---- DELETE. It drains the identical scan over the identical fixture, matches nothing, and so
    // ---- writes no file and commits nothing: what remains is the scan. See `assert_marginal_bound`
    // ---- for why subtracting it is sound against the mutation this test must catch.
    let base_small = measure_dml_mode(
        "cow_mem_base_small",
        N,
        "DELETE FROM catalog.cow_mem_base_small.t WHERE id < 0",
        true,
    )
    .await;
    let base_large = measure_dml_mode(
        "cow_mem_base_large",
        4 * N,
        "DELETE FROM catalog.cow_mem_base_large.t WHERE id < 0",
        true,
    )
    .await;
    assert_eq!(
        (base_small.affected, base_large.affected),
        (0, 0),
        "the baseline DELETE must match no rows, or it is not measuring a bare scan"
    );
    assert_eq!(
        (base_small.files_before, base_large.files_before),
        (files(N), files(4 * N)),
        "the baseline fixture must have the same file layout as the measured runs"
    );
    let baseline_delta = base_large.peak.saturating_sub(base_small.peak);

    // ---- DELETE ----
    let del_small = measure_dml(
        "cow_mem_del_small",
        N,
        &format!(
            "DELETE FROM catalog.cow_mem_del_small.t WHERE {}",
            where_clause()
        ),
    )
    .await;
    let del_large = measure_dml(
        "cow_mem_del_large",
        4 * N,
        &format!(
            "DELETE FROM catalog.cow_mem_del_large.t WHERE {}",
            where_clause()
        ),
    )
    .await;
    assert_fixture("small DELETE", &del_small, N);
    assert_fixture("large DELETE", &del_large, 4 * N);

    // ---- UPDATE ----
    let upd_small = measure_dml(
        "cow_mem_upd_small",
        N,
        &format!(
            "UPDATE catalog.cow_mem_upd_small.t SET payload = 'updated' WHERE {}",
            where_clause()
        ),
    )
    .await;
    let upd_large = measure_dml(
        "cow_mem_upd_large",
        4 * N,
        &format!(
            "UPDATE catalog.cow_mem_upd_large.t SET payload = 'updated' WHERE {}",
            where_clause()
        ),
    )
    .await;
    assert_fixture("small UPDATE", &upd_small, N);
    assert_fixture("large UPDATE", &upd_large, 4 * N);

    // Printed so a `--nocapture` run (and the mutation proof in the ledger) has the real numbers.
    let del_delta = del_large.peak.saturating_sub(del_small.peak);
    let upd_delta = upd_large.peak.saturating_sub(upd_small.peak);
    println!(
        "H7-S2 marginal peak — added={added_bytes} B, threshold={} B\n  \
         BASELINE (merge-on-read, zero match): peak(N)={} peak(4N)={} delta={baseline_delta}\n  \
         DELETE: peak(N)={} peak(4N)={} delta={del_delta} excess={}\n  \
         UPDATE: peak(N)={} peak(4N)={} delta={upd_delta} excess={}",
        added_bytes / 4,
        base_small.peak,
        base_large.peak,
        del_small.peak,
        del_large.peak,
        del_delta.saturating_sub(baseline_delta),
        upd_small.peak,
        upd_large.peak,
        upd_delta.saturating_sub(baseline_delta),
    );

    assert_marginal_bound(
        "copy-on-write DELETE",
        del_small.peak,
        del_large.peak,
        baseline_delta,
        added_bytes,
    );
    assert_marginal_bound(
        "copy-on-write UPDATE",
        upd_small.peak,
        upd_large.peak,
        baseline_delta,
        added_bytes,
    );
}
