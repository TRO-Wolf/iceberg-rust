<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# Recipe — auditing and repairing wrong partition keys

An operator recipe for the **wrong-partition-tuple** corruption class: a data file whose manifest
entry records a partition tuple that its own rows do not produce. Two engine-agnostic actions in
the core crate implement it — `iceberg::maintenance::AuditPartitionKeys` (read-only detection) and
`iceberg::maintenance::RepairPartitionKeys` (remediation).

Both are **fork-original**: Java `org.apache.iceberg.actions` has no counterpart, so they are
deliberately NOT part of the `ActionsProvider` surface (that mirrors Java's fixed 12-method set).

---

## 1. The exposure window

Verbatim from the back-to-goal brief (`task/back-to-goal-2026-07-25-brief.md`, Unit 2 · G1):

> commits via the DataFusion provider's `insert_into` into a partitioned table where the SELECT list
> contained a computed or reordered partition-source column; `VALUES` and plain passthrough are
> clean.

Concretely, a table is a candidate if **all** of the following held for some commit:

1. the table is **partitioned**;
2. the rows were written through the **DataFusion provider's `insert_into`** (`INSERT INTO … SELECT`
   / `CREATE TABLE AS SELECT`), not through the core writers directly;
3. the SELECT list contained a **computed** partition-source column (`CASE`, `CAST`, `coalesce`,
   arithmetic, a UDF, …) **or reordered / permuted same-typed columns**;
4. the commit predates the `PartitionExpr` honest-children fix (fork PR #172).

`INSERT INTO … VALUES` and a plain column passthrough (`SELECT a, b FROM t`) are **clean** — those
plans give DataFusion's `ProjectionPushdown` nothing to fuse.

### Why it is silent

The mechanism was an expression that computed the partition tuple while declaring no children, so
`ProjectionPushdown` re-parented it onto a different input and the tuple was computed **from the
wrong batch**. The result is a *real-but-wrong* tuple, and nothing rejects it:

- the commit path validates a partition tuple's **arity and types** against the spec, never its
  **values** (same in Java — `MergingSnapshotProducer.add(DataFile)`);
- a partition-**pruned** read then drops the file, so its rows silently vanish from query results —
  in Rust **and** in Java/Spark, because the damage is on-disk metadata, not an engine artifact;
- an **unpruned** read hands back the RECORDED value for every identity-partitioned column, because
  partition metadata is authoritative over the file's own column (Iceberg spec "Column Projection"
  rule 1 / Java `PartitionUtil.constantsMap`). The rows come back with the wrong values.

The fix (#172) stops NEW corruption. It does not repair tuples already written; that is what this
recipe is for.

---

## 2. Detect

```rust
use iceberg::maintenance::AuditPartitionKeys;

let report = AuditPartitionKeys::new(table.clone()).execute().await?;

println!(
    "{} files examined ({} skipped, unpartitioned specs), {} rows read",
    report.data_files_examined, report.data_files_skipped, report.rows_examined
);
for finding in &report.findings {
    println!(
        "MISKEYED {}\n  spec id           {}\n  recorded tuple    {:?}\n  rows say          {:?}\n  rows {} of {} disagree",
        finding.data_file_path,
        finding.partition_spec_id,
        finding.recorded_partition,
        finding.computed_partitions,
        finding.mismatched_rows,
        finding.rows_examined,
    );
}
assert!(report.is_clean(), "table has miskeyed data files");
```

For every LIVE data file of the current snapshot the audit re-reads the partition **source columns
from the file**, recomputes the transform chain under the file's own partition spec, and compares
against the manifest tuple. Every transform family is a pure function of its source columns and is
therefore recomputable: **identity, bucket, truncate, temporal (`year`/`month`/`day`/`hour`), void**.

`computed_partitions` lists every distinct tuple the file's rows produce, in first-seen order — a
miskeyed file may legitimately hold rows of several true partitions (that is what a reordered
source column produces), and one of them may even equal the recorded tuple (a partially-wrong file).

### What the audit does NOT do

- It does not write, commit, or delete anything.
- It reads **every live row of the table**. Treat it as a diagnostic sweep, not a hot path; scope it
  by running against a table clone / a specific snapshot if the table is large.
- It is not a substitute for a snapshot-level review: a corrupted commit may also have written
  delete files carrying the same wrong tuple (harmless for the audit — they are not data files, and
  the repair reads through them).

### Scope limits — read before acting on a finding

- **The partition source columns must be stored in the data file.** Everything this fork writes
  (including the DataFusion provider) stores every table column. A file registered by a *Hive-style
  migration* that OMITS its identity partition columns from the data would recompute to NULL and be
  reported as miskeyed — a **false finding**. Such tables are out of scope; the tell is a
  `computed_partitions` entry that is all-NULL where the recorded tuple is not.
- Files under an **unpartitioned** spec are skipped and counted in `data_files_skipped`: a manifest
  decodes each tuple against its own spec's partition type, so an unpartitioned spec's tuple is
  empty and cannot be falsified.
- The audit compares the CURRENT snapshot. Older snapshots retained for time travel keep their
  miskeyed entries; repairing the current snapshot does not rewrite history (by design — Iceberg
  snapshots are immutable). Expire the affected snapshots if the historical view must be clean.

---

## 3. Repair

```rust
use iceberg::maintenance::RepairPartitionKeys;

let result = RepairPartitionKeys::new(table.clone())
    .execute(&catalog)
    .await?;

println!(
    "repaired {} file(s) into {} correctly-keyed file(s), {} rows rewritten",
    result.repaired_data_files_count, result.added_data_files_count, result.repaired_rows_count
);
```

The repair runs the audit itself, and for each flagged file:

1. re-reads the file's **LIVE** rows (merge-on-read deletes applied);
2. splits them by their **recomputed** partition value;
3. writes one data file per distinct correct key through the standard rolling data-file writer
   behind a fanout writer (rolling at `write.target-file-size-bytes`, default 512 MiB);
4. commits **one `RewriteFiles` `Replace` snapshot per repaired file** that replaces the miskeyed
   file with the new ones.

There is **no in-place metadata surgery**: every change is an ordinary atomic commit through the
catalog, so a concurrent writer either serializes behind it or the repair fails loudly.

Properties worth knowing before you run it:

- **Sequence numbers are preserved.** The rewritten files keep the starting snapshot's data sequence
  number (Java `RewriteDataFiles.USE_STARTING_SEQUENCE_NUMBER_DEFAULT = true`), so outstanding
  equality deletes still apply and deleted rows are not resurrected.
  `use_starting_sequence_number(false)` exposes the Java-identical opposite behaviour; do not use it
  on a table with merge-on-read deletes.
- **Each file keeps its own partition spec.** A file written under an older spec is repaired under
  *that* spec — the repair fixes the tuple and changes nothing else. It is not a re-partitioning
  tool; use `RewriteDataFiles` for that.
- **Position deletes that referenced a repaired file dangle afterwards** (their rows were already
  excluded from what was rewritten). This is the same posture `RewriteDataFiles` takes; run
  `RemoveDanglingDeleteFiles` afterwards to clean them up.
- **A concurrent row-level delete on a file being repaired aborts the commit** (the shared
  `RewriteFiles` conflict validation), rather than silently resurrecting rows.
- **A clean table is a no-op**: nothing is committed and the result counts are zero.
- Re-run the audit afterwards. A clean re-audit is the acceptance signal.

Ordering with other maintenance: repair **before** compaction. `RewriteDataFiles` groups by
partition tuple and would carry a wrong tuple forward into the compacted output.

---

## 4. Reproducing a corrupted table (for drills / regression fixtures)

After #172 the engine can no longer produce this corruption, so a fixture must be built at the
**manifest level**: write a real data file, then commit a manifest entry stamped with a deliberately
wrong tuple. The commit path accepts it because it only checks arity and types.

```rust
// … write `batch` to a real parquet file with the ParquetWriterBuilder, then:
let mut builder = writer.close().await?.into_iter().next().expect("a data-file builder");
let data_file = builder
    .content(DataContentType::Data)
    .partition_spec_id(table.metadata().default_partition_spec_id())
    // the rows say dept = "eng"; record "sales"
    .partition(Struct::from_iter([Some(Literal::string("sales"))]))
    .build()?;

let tx = Transaction::new(&table);
let tx = tx.fast_append().add_data_files(vec![data_file]).apply(tx)?;
let table = tx.commit(&catalog).await?;
```

The worked fixtures (including a file miskeyed only in its identity component, one miskeyed only
in its truncate component, and one whose rows span two true partitions) live in
`crates/iceberg/src/maintenance/partition_key_audit_tests.rs`.

---

## 5. Pointers

- Implementation + design notes: `crates/iceberg/src/maintenance/partition_key_audit.rs`.
- Engine-side write contract: [`ENGINE_CONTRACT.md`](ENGINE_CONTRACT.md) §4 (write surface) and §7
  (distribution & ordering).
- Related maintenance actions: `RewriteDataFiles` (compaction), `RemoveDanglingDeleteFiles`
  (post-repair cleanup), `DeleteOrphanFiles` (physical leftovers).
