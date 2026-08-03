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

# iceberg-rust fork slate — MoR read/plan perf campaign (FK1–FK5)
# Source: RePark 10-agent scout 2026-08-02 (scanned POST perf-waves A–E #181 — all ideas
# are residuals NET of that campaign). Routed via PrimarySync hub handoff table.

Fork standing rules apply (fork AGENTS.md / CLAUDE.md; SEPMO octo per unit; map.md
lockstep; `[fork]` commit/PR tags (no legacy workstream names)). Import this brief to the fork's `task/` on first commit
(`task/fk-mor-perf-campaign-brief.md`). Local tips only; user merges.

**Doctrine: MEASURE FIRST, per idea** (same as RePark P-tracks): hour-0 = the scout's
named validation bench at base; a disconfirming measurement closes the idea as a WIN
(ledger the number). Perf must move ZERO behavior: the delete_filter equivalence
harness, interop suites (`dev/java-interop` oracles), and the full crate test set are
invariant bars at every tip. Java 1.10.0 bytecode is the semantics oracle wherever a
fix touches an ordering/precedence question — cite class+method in the ledger like the
scan-plan units did.

**Bases:** fork `origin/main` post-#182 (record SHA). No `[patch]`/RePark coupling —
RePark repins on its own cadence afterward (single-writer-per-pin, hub R5).

---

## FK1 — eq-delete apply: columnar keyset, no per-cell Datum storm (scout #3) — MUST-LAND

Files: `crates/iceberg/src/arrow/equality_delete_set.rs` (~174–251),
`caching_delete_file_loader.rs` (~724–837), `delete_filter.rs` (~671–738).

1. Hour-0 criterion bench: eq-delete apply ns/row + allocations, 1M data rows ×
   {100k, 1M} eq-deletes, single + multi-column keys, with/without nulls + floats.
2. Columnar key hashing: hash key columns directly from Arrow arrays (primitive fast
   paths; multi-column via row-wise combined hash over columnar accessors) — kill the
   `Vec<Option<Datum>>` decode + `HashSet<Vec<Datum>>` clone per cell. Decode the probe
   side once per batch.
3. Defer/lazy-build the Θ(E) survival predicate trees: only construct when the keyset
   path can't serve (the fallback), not unconditionally at parse.
4. **P0 soundness bar:** Java NULL semantics (null key cells never match) and float
   semantics (NaN/-0.0 per Java's Comparators) — bytecode-cite the Java behavior; the
   delete_filter equivalence harness + a dedicated null/float gate battery must be
   green; mutation-proof both gates (break the null rule → harness RED).

## FK2 — plan-path ownership + overlap (scout #5 + #14 + #15) — one unit, three cuts

1. **#5 Arc-share FileScanTask innards** (`scan/task.rs` split/sub_task,
   `context.rs into_file_scan_task`): residual memo currently deep-clones into every
   task; splits re-clone path/deletes/projection/partition. `Option<Arc<BoundPredicate>>`
   residual, `Arc<str>` paths, Arc'd project-field-ids/delete lists; splits share a
   parent + offset window. Watch the public-type/serde surface — if `FileScanTask`
   serialized shape must change, STOP-disclose (engine consumers).
   Hour-0: plan-only heaptrack, many-files × many-deletes fixture.
2. **#14 overlap delete/data manifest planning** (`scan/mod.rs` ~718–771): the barrier
   is not correctness-required — populate concurrently; data-entry processing parks on
   the existing `Notify` until the delete index is ready. Lost-wakeup + failed-populate
   hang tests (the two named risk classes).
3. **#15 delete-index keys** (`delete_file_index.rs`): key partition maps by
   `(spec_id, partition)` (post-filter linear scan today); sort global lists by
   sequence once, `partition_point` the applicable tail. Wrong-key = delete
   resurrection — identical-result-set pins across multi-spec fixtures are the bar.

## FK3 — lock hygiene pair (scout #12 + #13) — small unit

1. **#12** `delete_filter.rs`: `Arc<Mutex<DeleteVector>>` guards data the comments
   declare immutable post-install. Freeze to `Arc<DeleteVector>` at publish; merge ORs
   by reference (today clones full roaring bitmaps). FIRST: audit the load path for any
   post-publish mutation — if one exists, that's the finding (fix the mutation or keep
   the lock, ledger which).
2. **#13** memory-catalog: global mutex held across FileIO awaits — short critical
   sections + optimistic pointer CAS; I/O outside the lock. Atomicity pins
   (half-create refused; parallel load/update latency histogram before/after).

## FK4 — I/O pair (scout #7 + #30)

1. **#7 metadata-pointer cache**: session-scoped
   `(metadata_location → Arc<TableMetadata>)`; on load, HEAD/GetTable the pointer and
   skip body GET + re-parse when unchanged. **Fail CLOSED on any mismatch** — no soft
   reuse in v1 (the scout's rebase idea is explicitly OUT). Bench: two loads unchanged
   pointer → zero body GET (op-count injector); commit-retry op counts before/after.
2. **#30 OpenDAL list**: concurrent bounded stat for incomplete list entries
   (sequential HEAD today); prefer contiguous zero-copy Bytes over consolidating
   `to_bytes()` copies. 10k-key list HEAD-count bench; rate-limit disclosure.

## FK5 — `_pos` projection pushdown (scout #16) — OWN unit, STOP-gated,
## CORRECTNESS-ADJACENT (schedule LAST or defer to next slate)

`arrow/reader.rs` (~514–534, ~873–925): projecting `_pos` disables
RowFilter/RowSelection/row-group pruning and whole-file `try_collect`s. Wrong `_pos`
corrupts WRITTEN position deletes (RePark MERGE identity rides on it — hub note).

1. Track absolute row ordinals under RowSelection (offset accounting per selected
   range); minimum viable = stream batches instead of whole-file collect even without
   restored pruning (memory win alone justifies).
2. Bar: `_pos` oracle — dense + sparse pos-deletes + residual filters, row sets AND
   `_pos` values vs the unpruned baseline, mutation-proven (break the ordinal math →
   RED); the RePark-side MERGE battery re-run by the RePark workstream post-repin is
   the second net (hub coordination note).
3. STOP with the exact accounting gap named if selection-aware ordinals can't be
   guaranteed; the streaming-only half still ships.

---

## Execution: STRICT SERIAL — FK1 → FK2.1 → FK3 → FK4.1 → FK2.2/2.3 → FK4.2 → FK5.
## NO parallel units. Disk protocol between EVERY unit: (1) teardown the finished
## unit's throwaway worktree (`cargo clean` then `git worktree remove`), delete bench
## scratch artifacts; (2) `df -h` check — HARD STOP below 100 GiB free before the next
## unit's first build; (3) record freed/free GiB in the ledger. FK1 runs in the
## persistent `iceberg-rust-ws`; every later unit gets a throwaway tree.
## Morning: combine per fork convention, ONE gate (fork full battery + clippy
## --all-features -D warnings per fork rules + interop suites), per-idea before/after
## table in the PR body, GAP_MATRIX untouched (no parity claims — perf only),
## hub claim rows updated. RePark repin is a SEPARATE later step (R5).
