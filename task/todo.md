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

# Plan / Todo

The current plan for in-flight work. The operating manuals
([SEPMO](../.agents/skills/sepmo/SKILL.md) and the
[engineering method](../.agents/skills/engineering-method/SKILL.md)) require this file to be written
**before** any non-trivial change and kept current as work proceeds.

## ACTIVE (2026-09-01): F-6c branch-following reads (row R168)

Ledger: [`f6c-branch-following-reads-ledger.md`](f6c-branch-following-reads-ledger.md). Completes F-6b: `with_commit_branch` now scopes scans as well as commits.

- [x] Resolve the named-ref snapshot at every DataFusion scan (provider `scan`, CoW/MoR DELETE/UPDATE). Missing ref errors (Java `useRef`); INSERT without a target scan still creates the branch (F-6b).
- [x] Arm `validate_from_snapshot` with the same scan snapshot id (drop the F-6b skip-when-branch-set). Re-pin diverged serializable INSERT OVERWRITE.
- [x] CoW/MoR live-file walks follow the scanned snapshot, not `current_snapshot()` (main).
- [x] Pins: diverged SELECT/DELETE/INSERT-SELECT; default path; missing-ref scan vs DML vs INSERT; schema field-id bind; OCC scan==validate-from.
- [x] GAP_MATRIX row R168, ledger, map.md, `make check` + `cargo test -p iceberg -p iceberg-datafusion --locked`.

Outcome: `with_commit_branch` scans the named-ref head as well as committing onto it. Missing-ref SELECT/DELETE error; INSERT VALUES still creates. Schema follows IcebergTableScan field-id bind. `make check` and `cargo test -p iceberg -p iceberg-datafusion --locked` green. Docker `make test` legs excused.

## ACTIVE (2026-09-01): R91 parquet write refuses `unknown` loud

Ledger: [`r91-unknown-parquet-loud-ledger.md`](r91-unknown-parquet-loud-ledger.md). GAP_MATRIX row R91.

- [x] Reproduce silent `DataFileWriter::write` of a Null/`unknown` column (red leg).
- [x] Refuse at `ParquetWriterBuilder::build` with `FeatureUnsupported` naming the type and column.
- [x] Pin refusal + a neighbouring write/read of a legitimate batch in the same test module.
- [x] Enumerate parquet write doors (data / eq-delete / pos-delete / rolling) in the ledger.
- [x] Restate GAP_MATRIX row R91. Run `make check` and `cargo test -p iceberg --locked`.

Outcome: Parquet write of Iceberg `unknown` is `FeatureUnsupported` at
`ParquetWriterBuilder::build`. Red-leg at `33be9a0f4` was silent commit of an
unreadable Null column. Pins in `data_file_writer.rs` and
`parquet_writer_unsupported_tests.rs`. Row R91 is 🟡. `make check` and
`cargo test -p iceberg --locked` green. Docker `make test` legs excused.

How to use it (see the manuals' §1):

- Write a 3–7 bullet plan here before writing code.
- Flip `[ ]` → `[x]` as items complete; add a one-sentence "what changed and why" per step.
- Add indented sub-bullets when a step reveals unexpected complexity.
- Leave an `Outcome:` / `Done:` note when the work lands.

---

## ACTIVE (2026-09-01): F-6b DataFusion commit target (row R168)

- [x] Enumerate every snapshot-producing DataFusion commit site; store a provider-level branch.
- [x] Wire `to_branch` at INSERT INTO, INSERT OVERWRITE, CoW/MoR DELETE, CoW/MoR UPDATE.
- [x] Pins: default=`main`, named branch, missing ref, tag, each DML site.
- [x] Extend GAP_MATRIX row R168; unit gate + commit.

Outcome: `IcebergTableProvider::with_commit_branch` reaches all six DataFusion snapshot-producing DML sites. Default remains `main`. Pins in `tests/commit_branch.rs`.

## ACTIVE (2026-08-31): F-6 Critic remediation (row R168)

- [x] F-1: `starting_snapshot_for` falls back to txn-start main when the named ref is missing.
- [x] F-2: `validate_fresh_dvs_only` walks `latest_snapshot(..., target_branch)`.
- [x] F-3: diverged-branch validate-start pins (main-only ignored; branch-only rejected).
- [x] Q-002: re-fixture existing-branch as diverged; retry pin on OverwriteFiles.
- [x] Q-003: `TransactionAction::validate` docs name `starting_snapshot_for(target_ref())`.
- [x] Gates (`make check`, `make unit-test`, typos, file-size) + commit. Do not rewrite ce3719a6.

Outcome: Critic F-1/F-2/F-3, Q-002, Q-003 remediations landed on `parity/f6-branch-commit-target` as a second commit after ce3719a6. Missing-ref validate start falls back to txn-start main; fresh-DV walks the named branch; mutation C is red on the diverged-branch pins.

## ACTIVE (2026-08-28): relocate SEPMO into `.agents/skills/`

Scope ledger: C-001 `PROVEN` — `.agents/skills/` is the canonical agent-skill home and Claude
discovers it through `.claude/skills`. C-002 `PROVEN` — the complete SEPMO package moves as one
unit. C-003 `PROVEN` — every legacy SEPMO pointer retargets. C-004 `PROVEN` — package-relative
links account for the extra directory depth. C-005 `PROVEN` — canon and binding content stay
unchanged except for paths. C-006 `PROVEN` — validation includes skill, link, stale-pointer, license,
and repository gates. The user's request is the approval gate for this charter.

- [x] Move the complete SEPMO package to `.agents/skills/sepmo/` and retire the empty `skills/` index.
  - All 12 package files moved, and the former root `skills/` directory no longer exists.
- [x] Retarget repository-contract, adapter, license, map, task, and archive pointers.
  - Repository-wide scans find no pointer to the former root location.
- [x] Repair depth-sensitive links inside the moved package without changing its governance rules.
  - The spine and binding template remain byte-identical; only routing paths changed elsewhere.
- [x] Validate skill structure, every Markdown link, and the absence of stale legacy paths.
  - All 107 package links and all 125 links added by this change resolve.
- [x] Run the repository gates and an independent Critic.
  - The Critic converged with no blocking findings. The full retry passed all 4,509 tests.

Outcome: SEPMO now has one canonical home under `.agents/skills/`. Its governance spine and binding
template are byte-identical, all changed links resolve, and the former root `skills/` tree is retired.

## QUEUED (2026-08-28): F-17 shared-Puffin deletion-vector closure

GAP_MATRIX row R114 owns capability status. The detailed scope and evidence gates live in
[`f17-shared-puffin-dv-closure-ledger.md`](f17-shared-puffin-dv-closure-ledger.md).

- [x] Freeze the current source base (`e445b56ae`). Prove C-001 on that SHA.
- [x] Reproduce the two-file shared-Puffin live-row failure through the production reader (`[3,4,5,6]` vs `[3,4,6]`).
- [x] C-014 seam: `close_touched_dv_containers` + `add_delete_file_with_sequence_number`. Owner approval is the F-17 goal.
- [x] Extract one core-owned container-closure primitive with per-entry sequence semantics for maintenance and DataFusion DML.
- [x] Pin `DELETE` and `UPDATE` on the real SQL path (`shared_puffin_dv`). T1–T5 and T8–T23. Java verifies Rust DELETE, Rust UPDATE, and a Java-written shared Puffin after Rust DELETE.
- [x] Independent Critic + unit gate + PR. Update row R114 from measured evidence.

## ACTIVE (2026-08-28): F-14 Hadoop metadata pointer math

Parse and bump Hadoop `vN.metadata.json` on `MetadataLocation`. New GAP_MATRIX row R167.

- [x] Parse `vN.metadata.json` (and gzip names). Next pointer is `v(N+1).metadata.json`.
- [x] Hive/REST `<version>-<uuid>` is unchanged.
- [x] `register_table` of a v3 pointer then a commit writes v4.
- [x] Independent Critic (C1 CONVERGED; C2 F-F14-2 pins + F-F14-1 residue on row R167). Gates + PR.

---

## ACTIVE (2026-08-27): F-9 S3 Tables register_table + F-15 write_default

One PR. F-9 is a dated service-gap ruling (S3 Tables has no register-by-metadata-location API).
F-15 consumes `write_default` on the data-file write path (row R92).

- [x] F-9: keep `FeatureUnsupported`, name the service gap, pin it, record on row R126.
- [x] F-15: fill missing top-level primitive columns from `write_default` in DataFileWriter.
- [x] Tests on the real `register_table` and `DataFileWriter` entry points.
- [x] Independent Critic + gates + PR.

---

## ACTIVE (2026-08-27): F-16 delete-ratio + F-7 V3 DV accounting + comment part 2

One PR. Java `BinPackRewriteFilePlanner.tooHighDeleteRatio` (default 0.3) and
`MergingSnapshotProducer.removeDanglingDeletesFor` DV drops on `RewriteDataFiles`.
Status lives on GAP_MATRIX row R135 (and the limit-(k) wording on row R136).

- [x] Port `tooHighDeleteRatio` / `DELETE_RATIO_THRESHOLD_DEFAULT = 0.3`. Only file-scoped deletes count.
- [x] Drop DVs that reference rewritten data files in the same `RewriteFiles` commit. Rewrite Puffin siblings.
- [x] Comment part 2: apply `origin/docs/comment-compaction-part2` onto the 13 code-identical stack files.
- [x] Gates + independent Critic + PR. Merged #232 (2026-08-27).

---

## CLOSED (2026-08-26): 1,000-line Rust source-file gate

Port RePark's fail-closed Rust file-size guard to this workspace. The default ceiling is 1,000
lines. Existing files above the ceiling receive exact-count legacy ratchets because 104 files
already exceed the limit; no legacy file may grow and no new over-limit file may land.

- [x] Add `scripts/check_rust_file_size.py` with the 1,000-line default, exact legacy ratchets,
      actionable failures, and fail-closed handling for empty scans, unreadable files, and stale
      ratchet paths. The live scan passes over all 363 Rust files.
- [x] Add `scripts/check_rust_file_size_test.py` and the thin shell wrapper. Pin the boundary,
      over-limit rejection, ratchet behavior, stale paths, unreadable files, and empty scans.
      Eleven tests pass, including a real 1,001-line rejection.
- [x] Wire `check-rust-file-size` into `Makefile` `check` and add a direct CI check step in
      `.github/workflows/ci.yml`.
- [x] Update `AGENTS.md` so the build-gate roster points to the checker without duplicating its
      ceiling table.
- [x] Run the checker tests, clean-tree gate, explicit over-limit provocation, `make check`, and an
      independent fresh-context Critic review. The Critic's findings were remediated before it declared
      convergence with no open S0–S2 findings. All unrelated working-tree changes remain intact.

Outcome: the gate scans 363 Rust files. It freezes 104 inherited overages at their exact current
counts, rejects new files above 1,000 lines, and fails if legacy debt grows or leaves stale headroom.

## ACTIVE (2026-08-25): F-7 U3 — `RewritePositionDeleteFiles` extends to format v3

Branch `parity/f7-u3-rewrite-pos-deletes-v3` off `2b34ec414`. ENGINE-FIRST. **Correction 2026-08-26:
a Java counterpart DOES exist and the first pass wrongly said it did not.** It is Spark-only —
`RewritePositionDeleteFilesSparkAction`, decoded from
`iceberg-spark-runtime-4.0_2.13-1.10.0.jar`: `execute()` returns `EMPTY_RESULT` only when
`TableUtil.formatVersion >= 3` (offset 66-70) AND `requiresRewriteToDVs()` is false (74-77), so on a
v3 table still holding legacy parquet position deletes it falls through to the planner and
`doExecute` and CONVERTS them to DVs. Not mirroring the Spark surface is a scope decision;
`iceberg-spark` is outside this fork's core/api parity envelope. Two divergences from it are
deliberate and argued on row R136: no size gate (V-1) and one commit per run (V-2).

- [x] Version dispatch in `execute`: V1/V2 keep the bin-pack arm; V3 takes the DV arm.
- [x] Case 2: one Puffin DV per referenced data file, merged with its existing DV.
- [x] THE PUFFIN CLOSURE — path-keyed removal drags every sibling blob out of a superseded Puffin,
      so each sibling is rewritten too (including in partitions the filter excluded).
- [x] THE SHADOW CLOSURE — a DV shadows every position delete for its data file, so a live delete
      the filter EXCLUDED goes inert. The run now fails closed. Repro before the fix:
      two file-scoped deletes stamped `x=0` / `x=1` for one data file, `filter(x = 0)` →
      `Ok {rewritten: 1, added: 1}`, live `{12}` → `{12, 11}`.
- [x] Case 1: `Ok(zeros)` = "looked, found nothing", total WITHIN the filter's scope.
- [x] Hazard R114 bound (c): a `PartitionKey` on every `DVFileWriter::delete` call.
- [x] Stale positions are DROPPED, not refused — refusing dead-ended the table, since R137 keys on
      `(spec_id, partition)` and cannot clear a delete file that still names one live data file.
- [x] Per-DV sequence stamp pinned (the V3 twin of the bin-max pin).
- [x] Tests + the interop V3 leg; GAP_MATRIX R136 cell.

- [x] **S2 (2026-08-26) — the closure completed in BOTH directions.** The shadow guard's PARTITION
      leg is now pinned (it was 0-red before: the fork's `truncate(16)` bounds make partition-scoped
      the DEFAULT delete shape, so the unpinned leg was the common one). And the OPPOSITE loss is
      closed: an admitted legacy delete whose data file already holds a NON-SUPERSET DV is refused,
      because merging it would DELETE rows the table returns today. Java 1.10.0's own rewrite writes
      that shape — its `loadPreviousDeletes` is `path -> null` — so divergence (g)'s old "no real
      writer produces this" envelope was wrong and is corrected on row R136.
- [x] **Limit (k) named as a residue — and CORRECTED 2026-08-27: it is not a dead end.** The (g) fix
      makes a Java-rewrite-shaped table unconvertible BY THIS ARM at any filter width, but
      `RewriteDataFiles` clears it — but with TWO knobs, corrected again 2026-08-28: the one-knob
      wording shipped twice and is a NO-OP (0/0/0, both deletes still live, same refusal). Executed:
      `remove_dangling_deletes(true).delete_file_threshold(1)` gives 1 rewritten / 1 added / 2
      removed, live rows preserved exactly, second V3 run honest zeros. The threshold's default is
      `usize::MAX`, so the delete-count clause is off; `min_input_files` is NOT needed because
      `any_too_many_deletes` has no `size > 1` guard. Not universally unavailable at defaults —
      a partition of five small files admits the group anyway. R137 and the V3 DML arm still do NOT
      work and stay cited. The test runs the ADVERTISED invocation, keeps the one-knob no-op as a
      negative control, and asserts the refusal names every knob it passes.
- [x] (j) gains a TRIGGER: the block is an incidental parquet parse failure, not a guard, so (j)
      dissolves silently the day an ORC/Avro delete reader lands. Re-audit it in that change.
- [x] Capability limit (j) stated: a v3 table whose ORC/Avro position deletes OVERLAP convertible
      ones cannot be converted at any filter width, and the refusal says so.
- [x] V-1's third reason re-argued (gating can leave TWO live DVs for one data file, which
      `DeleteFileIndex` rejects); V-2's cost stated (no partial progress under `validate_from_snapshot`).

Refusal placement, corrected 2026-08-27. Mutation E witnessed ONE of the two facts, not both: it
proves closure siblings reach the check with an EMPTY position set (refusing an empty set reds the
sibling test, 1 of 3490). It does NOT witness the ordering claim — moving the check after the merge
is 0 red, because the refusal aborts before any IO wherever it sits. The real guarantee is that
`write_deletion_vectors` opens the first Puffin and is strictly later than every refusal; the
comment now says that instead.

Outcome: 15 offline V3 tests + the interop V3 leg green. Mutations applied one at a time, each RED.
Residue lettering on R136 runs (e), (f), (g), (i), (j); (h) was retired with the dangling refusal.

## QUEUED (2026-08-25): engine-agreed order — R166 interop, then F-13-or-F-7

Order set with the engine side 2026-08-25. F-14 and F-15 are explicitly NOT next.

- [x] **R166 interop leg — DONE 2026-08-25, and it was a BUILD, not a citation.** The engine's
      V3-0 lineage evidence turned out to be an UN-PINNED measurement — its own §5 says the CI pins
      deliberately avoid the Spark fixture, and the lineage numbers "stay a measurement in this
      ledger until V3-1 lands a fixture CI can read" — so citing it could not carry the row under
      the fork's definition of done. Built instead on the fork's OWN Java oracle, which already
      writes and reads V3 tables: `run-interop-row-lineage.sh`, both directions plus a cross-check
      that closes D2's circularity. R166 🟡→✅; residues (1) and (2) stay open and named on the row.
      The original brief follows, kept for its reasoning.
- [ ] ~~**R166 interop leg — a CITATION-AND-PIN job, not a build.**~~ The leg already exists on the
      engine side: unit V3-0 (RePark #199) appended rows with lineage through the fork and
      round-tripped through Spark→Java with the read verified Spark-exact. **Read
      `/home/john/CodeRepos/LocalRepark/repark/task/ledgers/staging/v3-0-charter-ledger.md` and
      `/home/john/CodeRepos/LocalRepark/repark/task/ledgers/archive/2026-08/2026-08-24-v3e-1-2-cow-oracle-ledger.md`
      BEFORE writing anything.** The named v3 oracle — PySpark 4.1.2 +
      `iceberg-spark-runtime-4.1_2.13:1.11.0` — is live on this machine with the V3E-3 fixtures
      (partitioned DV + equality-delete). A fork-local pin that REUSES that oracle is fine; a
      separate harness is not. Closes residue (3). It does NOT by itself close the row: residue
      (1), the untested ORC stored-`_row_id` arm, has no oracle either, so the flip needs the
      legend's named-unproven-slice allowance applied deliberately, not silently.
- [x] **F-7 slice 1** (2026-08-30, `parity/f7-row-lineage-carry`): V3-LINEAGE-1 + V3-COW-1 —
      stored `_row_id` / `_last_updated_sequence_number` through `RewriteDataFiles` and
      DataFusion COW `OverwriteFiles`. Ledger:
      [`f7-row-lineage-carry-ledger.md`](f7-row-lineage-carry-ledger.md). Unlocks engine
      units V3-4 + V3-5 at repin.
- [ ] **F-7 remaining — V3-DANGLE-1 / row R137** dangling-DV drop on compaction.
- [ ] **F-7 remaining — row R136** RewritePositionDeleteFiles DV-aware remainder.
- [ ] **F-7 remaining — MoR RowDelta** `_row_id` carry on added data files. MoR UPDATE currently
      writes all-null lineage columns through the shared v3 writer schema; see the F-7 ledger.
- [ ] R166's other two residues stay open and named: the ORC stored-column arm has no oracle
      (Java's ORC reader is outside `iceberg-core`), and the ranged-split refusal is unreachable
      through `plan_tasks` but reachable through the public `PartitionWork` seam.
- [x] **Fork units (a)/(a2)/(c) on row R114.** Landed on `parity/h7-p1-r114-dml-prune`
      (2026-08-28) with H7-P1. (a) `spec::is_deletion_vector` + `referenced_data_file_location`
      are public. (a2) `live_deletion_vectors_by_data_file` is public; missing referenced path
      and duplicate path error. (c) `(None, None)` is `DataInvalid`; call `unpartitioned()`.
      `rewrite_table_path` stamps the source spec. R114 stays 🟡.

---

## ACTIVE (2026-08-16): PT increment C — files/entries project the unified type

Branch: `grok/c16-files-entries-unified` off `16abae8b` (#204). Projected-type
swap only (A1): `append_partition` stays the PT-0 field-id walk.

- [x] `FilesTable` / `EntriesTable` `try_*` store `unified_partition_type`;
      schema+scan use it. DataFusion `try_new` is the G1/G2 seam.
- [x] Fixtures 2/3/6 in `scan/partitioning_fixtures.rs`.
- [x] Pins: dropped-v2 field kept; void-repair name+type; evolved-to-unpartitioned
      keeps `partition`; swapped same-typed specs match by field id; G2 refuse.
- [x] GAP_MATRIX R142 + `inspect/map.md` + `scan/map.md` lockstep.

## ACTIVE (2026-08-16): PT increment D — `position_deletes` schema adopts the unified type

Branch: `grok/c16-position-deletes-unified` off `d09d4831` (#203). Schema-only (FB-2);
do **not** un-refuse the scan.

- [x] `PositionDeletesTable::try_new` stores `unified_partition_type`; schema uses it
      for remapped children + empty-partition drop. `scan()` stays `FeatureUnsupported`.
- [x] DataFusion `IcebergMetadataTableProvider::try_new` uses `try_new` for this table.
- [x] Pins: widening two-child remap; evolved-to-unpartitioned keeps `partition`;
      scan still refused; `try_new` G2 refuse on `new_with_two_identity_specs`.
- [x] GAP_MATRIX R142 + `inspect/map.md` lockstep.

## ACTIVE (2026-08-16): PT increment B — `partitions` adopts the unified type

Branch: `grok/c16-partitions-unified` off the #202 squash-merge. Java `PartitionsTable` +
`PartitionUtil.coercePartition` rollup.

- [x] `PartitionsTable::try_new` stores `unified_partition_type`; `schema()` stays
      infallible; `new()` falls back on G1/G2 so the inspect API cannot panic.
      `IcebergMetadataTableProvider::try_new` is the fallible seam.
- [x] `is_unpartitioned` / schema / scan use the unified type; rollup keyed on
      `coerce_partition`. Module-doc scoping note replaced with a parity statement.
- [x] Fixtures 1 + 5 in `scan/partitioning_fixtures.rs` (A8). `new_with_two_identity_specs`
      used only as a `try_new` G2 refusal pin.
- [x] GAP_MATRIX R142 + `inspect/map.md` + `scan/map.md` lockstep.

## ACTIVE (2026-08-16): PT increment A — hoist `Partitioning` into `spec/partitioning.rs`

Branch: `grok/c16-partitioning-analogue` off `250ea37f` (#201). First landable increment of
the PT campaign (`PARTITIONING-UNIFICATION-DESIGN.md` §3.3 A). Inspect adoption is B/C/D.

- [x] New `crates/iceberg/src/spec/partitioning.rs`: `partition_type`, `grouping_key_type`,
      `union_partition_types` (`pub(crate)`), `is_partitioned`, `coerce_partition`, plus
      `TableMetadata::unified_partition_type`. G1 unknown-transform refusal, G2 conflict
      `DataInvalid`, G3 void-transform type repair, G4 per-field type resolution after the
      id filter (dropped source is ignored, not `Err`).
- [x] `maintenance/partition_stats.rs::unified_partition_type` is a thin delegate; private
      `coerce_partition` deleted; stats collect path uses `spec::coerce_partition`.
- [x] Shape pins + coerce units + mutation baits in `spec/partitioning.rs`; existing
      partition_stats coerce tests retargeted; delegation-equivalence pin.
- [x] `TableTestFixture::new_with_two_identity_specs` is a Java-invalid unifier input
      (duplicate field id 1000, different sources) — not used here; stays scan-only.
- [x] GAP_MATRIX R142 + `inspect/map.md` lockstep. ENGINE_CONTRACT untouched.

## ACTIVE (2026-08-15): PT-0 field-id partition-tuple projection in `data_file`

Branch: `repark/pt0-positional-partition-walk` off `d3c30181` (#199). Interim correctness fix ahead of
the PT-1..4 `Partitioning.partitionType` unification campaign
(`planning/hardening/PARTITIONING-UNIFICATION-DESIGN.md` §2.4, priority-1 break).

- [x] REPRO PIN. Java-legal spec-evolution fixture in `inspect/data_file.rs` tests
      (`with_reordered_evolved_spec`): spec 0 = `identity(x)`@1000; spec 1 (default) =
      `identity(y)`@1001 THEN `identity(x)`@1000 — monotonic field ids (NOT built on
      `TableTestFixture::new_with_two_identity_specs`, which reuses id 1000 across two source columns
      and is a Java `"Conflicting partition fields"` refusal), tuple order ≠ field-id order, both
      fields `long`. Verdict: the `files` scan reported the spec-0 file's `x` value (777) under the
      `y` column and `null` under `x`. Silent — no error, because the types agree.
- [x] INTERIM FIX. `append_partition` takes the source spec's field ids and matches each PROJECTED
      field by id, null-filling what the source spec lacks — the field-id half of Java
      `PartitionUtil.coercePartition` / `StructProjection.createAllowMissing`. New
      `partition_field_ids_by_spec(&TableMetadata)`; `DataFileStructBuilder` resolves each file's own
      spec via `DataFile::partition_spec_id` (unknown spec id → loud `DataInvalid`, Java NPEs there).
      `files`/`entries`/`partitions` call sites updated. NOT the unification: the projected type is
      still `default_partition_type`, so an older-spec-only partition field is dropped rather than
      surfaced — named residue, PT-1..4.
- [x] Tests same commit: two `append_partition` units (null-fill of an absent projected field;
      full-tuple reorder) + the end-to-end `files` pin. Mutation bait: restoring the positional walk
      reds all three.
- [x] GAP_MATRIX R142 in place (finding + fix note; no new row). `inspect/map.md` lockstep
      (`data_file.rs`/`partitions.rs` rows, divergence row, two new failure modes).
- [x] Gates: `make check` EXIT=0; `make unit-test` EXIT=0; `make test` EXIT=0 on retry after one
      `make docker-down` cycle (4211 passed / 3 skipped). One `[repark]` PR, no merge.
## ACTIVE (2026-08-15): F-1 Java-cited doc strings on `readable_metrics` leaves

Branch: `repark/f1-readable-metrics-leaf-docs` off `d3c30181` (#199). Closes the finding #199
reported and deliberately did not ship (omitted, never xfailed).

- [x] `readable_metrics.rs::readable_metrics_field` passes each leaf's Java doc into
      `NestedField::with_doc`, from six new `*_DOC` consts transcribed verbatim from
      `MetricsUtil.READABLE_METRIC_COLS` — Java builds every leaf with the FOUR-arg
      `optional(nextId.incrementAndGet(), m.name(), m.colType(field), m.doc())`. Module-doc table
      gains the `doc()` column. Nothing else changes: ids, order, types, values untouched.
- [x] Re-enable the ported test O-8 removed: `java_schema_shape.rs`'s `READABLE_METRIC_COLS` becomes
      `(name, is_long, doc)` (mirroring `ReadableMetricColDefinition` fully) and
      `readable_metrics_metric_cols_carry_their_java_doc_strings` asserts all six docs on every leaf
      column of both `entries` and `all_entries`. Mutation bait: dropping any single `.with_doc`
      reds it with the exact failure #199 quoted (`doc for column_size` / left `None` / right
      `Some("Total size on disk")`).
- [x] GAP_MATRIX R142 in place (F-1 landed note; anchors untouched, no new row — the drafted row in
      #199 would have opened a gap that is closed in the same breath). `inspect/map.md` lockstep.
- [x] Gates: `make check` EXIT=0; `make unit-test` EXIT=0; `make test` EXIT=0 on retry after one
      `make docker-down` cycle (4209 passed / 3 skipped; first attempt was the known SQLITE_BUSY
      REST-catalog flake on `test_register_table`/`test_update_table` — never `crates/iceberg`).
      One `[repark]` PR, no merge.

## ACTIVE (2026-08-15): FB-1 Java inspect schema-shape battery (increment 1)

Branch: `repark/fb1-java-schema-shape` off freeze `0c5fd58d` (#195). Additive tests only
(conductor-13F A4/A5). No GAP_MATRIX cell edit.

- [x] NEW `inspect/java_schema_shape.rs` citing `MetadataTableUtils.createMetadataTableInstance`,
      `BaseFilesTable.schema` / `DataFile.getType`, and `PartitionsTable.schema`.
- [x] Cover all six files-family analogues + partitioned/unpartitioned PartitionsTable field
      id/name/required. OUT: `position_deletes`, entries shape, cross-spec unification,
      `readable_metrics` interior field-id order.
- [x] `#[cfg(test)] mod java_schema_shape;` hook + inspect/map.md row + doc-truth correction of
      the false "full Java table set is implemented" sentence (row R142 residual).
- [x] A6: `make check` EXIT=0; `make unit-test` EXIT=0; `make test` EXIT=0 on retry
      (first attempt SQLITE_BUSY on REST `test_get_namespace`/`test_update_table`; retry
      4193 passed / 3 skipped). One `[repark]` PR. Report after `gh pr create`.
## ACTIVE (2026-08-15): FB-4 Java inspect schema-shape battery (increment 2)

Branch: `repark/fb4-java-battery-2` off `f40d3faa` (#198). Additive tests only
(conductor-13F A4/A5). No GAP_MATRIX cell edit. Covers exactly what #196 deferred.

- [x] `BaseEntriesTable.schema()` shape: `ManifestEntry.wrapFileSchema` rows (0/1/3/4 then
      `data_file`/2 LAST), `data_file` == `DataFile.getType(partitionType)`, nested
      `selectNot(102)` on the unpartitioned branch (FW-3 drop, #194), `entries` ==
      `all_entries`, and `entries.data_file` == the flat `files` projection.
- [x] `MetricsUtil.readableMetricsSchema` STRUCTURE: one optional struct per PRIMITIVE
      leaf column with Java's docs, the six `READABLE_METRIC_COLS` in list order (four
      `long` + typed `lower_bound`/`upper_bound`), the by-name emit sort, and the
      pre-increment id counter seeded at the host `highestFieldId()` (1000 partitioned /
      145 unpartitioned). Interior field-id ORDER pinned as the fork's DOCUMENTED
      divergence — Java's `idToName()` HashMap order is not portable, so it is not chased.
- [x] FINDING F-1 (reported, NOT shipped here): the six `readable_metrics` sub-fields carry no
      doc string; Java passes `m.doc()` ("Total size on disk", "Total count, including null
      and NaN", "Null value count", "NaN value count", "Lower bound", "Upper bound") into
      `optional(id, name, type, doc)`. The column struct + top-level docs ARE present.
      Cosmetic (docs are not in the Avro/Arrow read path). **Shipped separately 2026-08-15 —
      see the F-1 section below.**
- [x] A6: `make check` / `make unit-test` / `make test`. One `[repark]` PR, no merge.

## ACTIVE (2026-08-15): FB-3 lazy per-namespace `list_tables`

Branch: `repark/fb3-lazy-list-tables` off freeze `0c5fd58d` (#195). Independent of FB-1 (#196).
FB-2 skipped (conductor-13F A6).

- [x] Remove eager `list_tables` from `IcebergSchemaProvider::try_new`; populate a once-success
      directory on first access. Sync paths use existing `block_on_off_caller_runtime` only.
- [x] Failed listing not cached; `table()` / register / deregister surface it; `table_names` /
      `table_exist` return empty/false. `list_namespaces` failures still fail construction.
- [x] Update `fail_tables_of` pins + ENGINE_CONTRACT §1 + GAP_MATRIX R164 in place (lazy residual
      closed; 🟡→✅ — no other residual).
- [x] A6: `make check` EXIT=0; `make unit-test` EXIT=0; `make test` EXIT=0 on retry
      (first attempt SQLITE_BUSY on REST `test_update_table`/`test_register_table`; retry
      4188 passed / 3 skipped). One `[repark]` PR.

## ACTIVE (2026-08-14): FW-1 timestamptz `data_file` metadata projection

Branch: `repark/fw1-timestamptz-datafile-projection`. Charter: F-V4-1 / A7 — projection/read only.

- [x] Add `Timestamptz` + `TimestamptzNs` arms in `inspect/data_file.rs` `append_partition_field` (mirror `readable_metrics.rs`). `Uuid`/`Fixed` stay in `other`.
- [x] Confirm `type_to_arrow_type` already produces timestamptz partition children; fix only if it also refuses. (already produced `Timestamp(µs/ns, +00:00)`; Avro `Long` decode of `timestamp_ns`/`timestamptz_ns` was the inspect-read blocker and was added in `serde.rs`.)
- [x] Tests (same commit): `.files` / `.partitions` over timestamptz-identity and `.files` over timestamptz_ns-identity asserting the projected value; Uuid + Fixed identity partitions still error on the existing needle (isolated + metadata-table).
- [x] New GAP_MATRIX row R162 (do not overload R142). Update `crates/iceberg/src/inspect/map.md`. ENGINE_CONTRACT.md and repark pins untouched.
- [x] A6 gates: `make check` EXIT=0; `make unit-test` EXIT=0; `make test` EXIT=2 — docker-up bind failed on host `:9000` (pre-existing host process, not this compose). A6 HALT; no push / no PR.
## ACTIVE (2026-08-14): FW-2 Arrow `UTC` annotation on timestamptz read

Branch: `repark/fw2-arrow-utc-annotation` off freeze `1dae9b66`. Independent of FW-1 (#192).
Charter: F-V4-2 — schema annotation only; value/kernel CLOSED; repark harness NOT edited.

- [x] Flip `UTC_TIME_ZONE` `"+00:00"` → `"UTC"` and route every Arrow timestamptz *producer* through it (schema visitor, inspect builders, `get_arrow_datum`, `with_timezone_utc` fixtures that `try_new` against the Iceberg Arrow schema).
- [x] Keep Arrow→Iceberg acceptance of BOTH `"UTC"` and `"+00:00"` (`is_utc_time_zone`); do not narrow. Writer `is_utc_alias` must not collapse when the constant flips.
- [x] Tests same commit: Iceberg→Arrow emits `UTC` (µs + ns); `+00:00` input still maps; non-UTC tz still rejected; flip output pins (schema expect, inspect Debug, parquet writer, datafusion snapshots).
- [x] GAP_MATRIX new row R163 (R162 reserved by independent FW-1). `writer/map.md` lockstep. ENGINE_CONTRACT §F-A2-3 sentence updated so the contract is not stale.
- [x] acc + C4, then A6 `make check && make unit-test && make test`. One `[repark]` PR. Report after `gh pr create`.
      **Done 2026-08-14:** `UTC_TIME_ZONE` is `"UTC"`; inverse still accepts `+00:00`; GAP_MATRIX R163; A6 green (first `make test` SQLITE_BUSY flake, retry 4162/4162).

## ACTIVE (2026-08-14): FW-3 metadata-provider parity (F-2 + F-3)

Branch: `repark/fw3-metadata-provider-parity` off freeze `1dae9b66`. Provider/read only.

- [x] **F-3 `$`-name split.** Last `$` + `MetadataTableType` vocabulary. Tests: enumerate / exist / metadata twins / empty read of `a$b`.
- [x] **F-2 unpartitioned `partition` column.** Java-cited drop on files+entries (id 102) and partitions (ids 1 and 4). `PositionDeletesTable` named residual (not ported).
- [x] **R142 in place.** Empty-partition residual flipped; remaining: cross-spec unification, readable_metrics order, unported `position_deletes`. repark pin not flipped.
- [x] **A6.** `make check` + `make unit-test` green; `make test` 4165 passed / 3 skipped after one SQLITE_BUSY retry.

---

## CLOSED (2026-08-08): 2026-08 audit hardening — SEPMO bundled branch

Branch: `fix/2026-08-audit-hardening`. User approved the 8/8 proposition ledger on 2026-08-08.
Delivery mode: one bundled branch/PR with clause-separated groups, an independent fresh-context
Critic after every group, and an independent bundle-scope closing Critic. Every group is STANDARD
(public API, data-integrity, or security surface); the repository severity floor is S2.

Frozen charter clauses: C-001 decimal construction/conversion invariants; C-002 typed failures for
short partition structs and manifest summaries; C-003 bounded predicate/Arrow/schema-evolution
recursion; C-004 qualified DataFusion namespaces; C-005 byte-weighted cache-moka capacity; C-006
REST Error/source and table/view Debug secret redaction; C-007 preserve the existing trusted-catalog,
Java-parity, on-disk-format, and dependency contracts; C-008 deliver the bound SEPMO evidence and
green gates. Explicit exclusions: REST endpoint/header trust redesign, LocalFS jail, HMS TLS/SASL,
V3 MoR, delete-sequence/partial-resolver redesign, architecture reorganization, dependency changes.

> **Evidence lives in [2026-08-audit-hardening-ledger.md](2026-08-audit-hardening-ledger.md)** — gate
> records, Critic dispositions, the Invariant V amendments, the scope-delta adjudications and all 31
> named S3 residues. Per the de-triplication rule this tracker holds one-line statuses only; nothing
> here restates capability STATUS (that stays in [GAP_MATRIX.md](../docs/parity/GAP_MATRIX.md)).

- [x] **G1 — decimal invariants (C-001, C-007).** In scope:
      `crates/iceberg/src/spec/datatypes.rs`, `crates/iceberg/src/arrow/schema.rs`,
      `crates/iceberg/src/arrow/record_batch_transformer.rs`, affected decimal fixture tests in
      `crates/iceberg/src/spec/values/tests.rs`, `crates/iceberg/src/spec/values/datum.rs`,
      `crates/iceberg/src/spec/values/serde.rs`, `crates/iceberg/src/spec/values/literal.rs`, and
      adjacent unit tests. Reject negative/unrepresentable/out-of-domain scale and precision without
      numeric truncation; preserve valid encodings. The production caller, affected Avro-decimal
      fixtures, and public Datum boundary additions were explicitly approved by the user on
      2026-08-08; the RawLiteral and Literal JSON boundary expansion was explicitly approved on
      2026-08-09 after the second independent Critic filed five S2 findings. Unit gate →
      independent Critic → recorded disposition.
      **Done 2026-08-09:** decimal type/value validation covers the constructor/encode paths
      (`Datum` constructors, `Datum::to_bytes`, Arrow conversion) without truncating casts, and the
      legacy diagnostics are pinned. Four independent remediation Critic cycles closed all filed
      S1/S2 findings; verdict at that point was CONVERGED (zero S1/S2).
      **R1 correction 2026-08-09 (commit `ff53c252`), supersedes the two clauses struck above:** a
      later independent review filed S2s showing G1 had imposed two invariants Java 1.10.0 does not
      have, on paths that READ existing on-disk metadata. G1's read-path gates were therefore
      DELIBERATELY REVERTED to Java permissiveness, so the earlier claims that "canonical encodings
      are pinned" and that validation "spans Datum/RawLiteral bytes, Serde/JSON, and nested
      literals" are no longer true of the shipped code and must not be relied on:
      (a) `Datum::try_from_bytes` no longer requires the canonical minimal two's-complement
      encoding — Java `Conversions.internalFromByteBuffer` is a bare `new BigInteger(bytes)` — and
      the replacement test asserts padded encodings decode;
      (b) `deserialize_decimal` is no longer routed through `Type::decimal`, and
      `validate_decimal_type` no longer requires `scale <= precision` — Java
      `Types$DecimalType.<init>` checks only `precision <= 38`;
      (c) `validate_decimal_value` / `validate_decimal_literal` were removed from five read/JSON
      doors: the `RawLiteral` bytes arm, the 16-byte list arm, the `RawLiteral` `Int128` write arm,
      `Literal::try_from_json` / `Literal::try_into_json`, and both `Datum` serde impls.
      The encode-side anti-truncation gate on `Datum::to_bytes` is RETAINED and mutation-proved.
      Read/encode split and every Java citation are documented at the call sites. Capability status
      stays in GAP_MATRIX R87 (already updated); this entry records only that R1 happened, so the
      G6 bundle-close Critic adjudicates the corrected state rather than the struck clauses.
      **R1 remediation cycle 1 (this branch):** added a `DatumVisitor::visit_seq` pin (the compact,
      non-self-describing serde route) to
      `datum_decimal_serde_round_trip_preserves_java_readable_values`; the previous test reached
      `visit_map` only, so a re-added gate on the seq arm survived mutation.
- [x] **G2 — malformed metadata + recursion safety (C-002, C-003, C-007).** In scope:
      `crates/iceberg/src/expr/accessor.rs`, `crates/iceberg/src/expr/visitors/{predicate_visitor.rs,
      bound_predicate_visitor.rs,manifest_evaluator.rs}`, `crates/iceberg/src/arrow/schema.rs`,
      `crates/iceberg/src/transaction/update_schema.rs`, and affected `map.md`/adjacent tests. Return
      typed errors for short metadata and impose tested traversal limits. Unit gate → independent
      Critic → recorded disposition.
      **R2 correction 2026-08-09, supersedes the predicate half of the G2 first pass (`de3961da`):**
      an independent review filed S2s showing that pass had made the predicate recursion *worse*
      than main. Remediated in this commit:
      (a) making `predicate_visitor::visit` / `bound_predicate_visitor::visit` fallible turned the
      two pre-existing `.expect("RewriteNotVisitor guarantees always success")` calls into a LIVE
      PANIC reachable from `TableScanBuilder::with_filter`. `Predicate::rewrite_not`,
      `BoundPredicate::rewrite_not`, both `negate`s and both `Display` impls are now
      explicit-stack walks that neither recurse nor panic; `rewrite_not` no longer routes through
      the depth-limited visitor at all, so the typed depth error surfaces at `bind()` where a
      caller can handle it. The unbound `PredicateVisitor` and `RewriteNotVisitor` lost their last
      production callers and are `#[cfg(test)]`-retained as the differential oracle.
      (b) `MAX_PREDICATE_DEPTH` was 100 — copied from the JSON parser and BELOW what this crate's
      own read path builds, so ~102 equality-delete files or 102 pushed-down conjuncts turned a
      Java-readable table into a scan failure. It is now 1000, re-derived from a MEASURED
      per-level stack cost (bisect against a known `stack_size`) sized for a 2 MiB tokio worker;
      the measurement, the arithmetic and the dev-profile caveat are in the constant's doc
      comment, and splitting the leaf arms out of the recursive `bind`/`visit` bodies halved the
      per-level cost that number rests on.
      **R3 correction 2026-08-09 (`006dc721` + `340fa4ea`) — closes the line above:**
      `assign_fresh_ids` is no longer unbounded. G2's first pass had hardened `index_parents`, which
      only ever receives a `Schema` already validated by `SchemaBuilder` against
      `MAX_SCHEMA_NESTING_DEPTH`, while the walk reachable from the public
      `UpdateSchemaAction::add_column` took a caller-supplied `Type` with no bound at all. That walk
      is now bounded at 128 (consistent with the crate's existing constant family) and proved on a
      small-stack thread, with one independent chain per recursion arm. The false `index_parents`
      rationale was corrected in the same pass: it is defence in depth, not the fix.
      REMAINING residue: the derived `Drop`/`Clone`/`PartialEq` glue on `Predicate`/`BoundPredicate`
      still recurses — nothing in this bundle can protect it.
      **Done 2026-08-09.** Critic CONVERGED zero S1/S2 on R2 (1 cycle) and R3 (2 cycles); 11 S3
      ledgered. Invariant V amendments for `expr/predicate.rs` and `spec/schema/id_reassigner.rs`
      recorded in the ledger §2.
- [x] **G3 — DataFusion namespaces (C-004, C-007).** In scope:
      `crates/integrations/datafusion/src/catalog.rs` and adjacent tests. Preserve full namespace
      identity, cover nesting/collisions/failures, and do not broaden DataFusion API scope. Unit gate
      → independent Critic → recorded disposition.
      **Done 2026-08-09** (`d99e56d6` + `2556e109`, Critic CONVERGED cycle 2, 5 S3): nested
      namespaces discovered by explicit-queue BFS with a depth cap and bounded concurrency; identity
      preserved by joining levels on U+001F — the convention `NamespaceIdent::to_url_string()`
      already uses and REST/S3 Tables already rely on — so `split('\u{1f}')` is a total inverse.
      Collisions rejected rather than shadowed; a dot alias keeps nested namespaces SQL-typeable.
      No public API added.
- [x] **G4 — cache-moka byte capacity (C-005, C-007).** In scope:
      `crates/integrations/cache-moka/src/lib.rs` and adjacent tests. Replace entry-count semantics
      with deterministic byte weighting without dependency edits. Unit gate → independent Critic →
      recorded disposition.
      **Done 2026-08-09** (`cb489615` + `73ff5157`, Critic CONVERGED cycle 2, 6 S3): `weigher` +
      `max_capacity` mirroring `io/object_cache.rs`, so 32 MiB means bytes rather than ~33.5M
      entries. Operator-visible consequence recorded: the default aggregate ceiling is now
      2 × 32 MiB = 64 MiB, twice the core cache's single budget for the same two object kinds
      (merging them is ARCH-004, excluded).
- [x] **G5 — secret rendering (C-006, C-007).** In scope:
      `crates/catalog/rest/src/{client.rs,types.rs}`, core table/view metadata and facade types,
      `crates/iceberg/src/error.rs`, a narrowly scoped shared redaction helper if required, affected
      `map.md`, and adjacent tests. Preserve error chaining and non-sensitive diagnostics; eliminate
      the enumerated credential render paths without changing catalog trust policy. Unit gate →
      independent Critic → recorded disposition.
      **Done 2026-08-09** (`879bf55e`, Critic CONVERGED cycle 3, 3 S3): SEC-001 closed by replacing
      the raw `serde_json::Error` source with a `SanitizedJsonError` carrying failure category and
      position but nothing derived from the body — the chain obligation survives because `source()`
      still EXISTS, which was the actual requirement. The former residue pin in
      `crates/catalog/rest/src/catalog.rs` is INVERTED, not deleted. SEC-002 / SEC-009 closed via
      per-key `is_secret_prop_key` redaction. NAMED residue: `Table`'s `{:?}` is still not wholesale
      credential-safe (snapshot summaries, encryption keys, statistics blob properties render in
      clear) and the rustdoc now says so instead of asserting a blanket invariant.
- [x] **G6 — bundle close (C-008).** Run the independent bundle Critic, disposition any remands,
      run `typos . && make check && make check-msrv && cargo build -p iceberg
      --no-default-features && cargo deny check advisories && make test`, execute targeted interop if
      a group changes an interop-bearing contract, file PR-readiness evidence, update this tracker,
      and file the SEPMO retrospective/metrics. No push or PR creation without separate user request.
      **Done 2026-08-09.** Bundle Critic ran three lenses (cross-group interaction, ledger truth,
      adversary) + two verification rounds, all serialized in the one worktree. It filed **six S2s
      against the ARTIFACT and zero against the code** — including a breaking public API change
      (`datum_to_arrow_type_with_ree -> Result<DataType>`, G1) that nothing had called out while a
      downstream consumer is mid-repin, and two false statements in the gate evidence itself. All
      dispositioned; terminal round CONVERGED with zero blocking findings and six ledgered.
      Evidence: [2026-08-audit-hardening-ledger.md](2026-08-audit-hardening-ledger.md) ·
      [critic verdicts / S3 register](2026-08-audit-hardening-critic-verdicts.md) ·
      [sepmo-metrics.md](sepmo-metrics.md).
      **Gate gap RESOLVED 2026-08-10 — it was local only.** The Docker daemon was down so `make
      test` did not run here, but CI's `Tests (default)` job does `make docker-up` + full-workspace
      nextest on every PR: **4158/4158 passed on #191**, including all eight binaries that failed
      locally. Local `make test` is a pre-flight convenience, not a coverage requirement.
      **PUSHED 2026-08-09 on user instruction; PR #191 open, CI 14/14 green.** Merging is the user's.

Contingencies: a group that cannot converge is either REMOVED with an additive revert commit or
REMANDED with enumerated findings to the closing Critic; no destructive reset/checkout is authorized.
Any dependency-file need, on-disk-format change, trust-model change, or unexpected-file requirement
raises Invariant V and returns the affected scope to audit.

---

## DONE (2026-08-08) — U3 / hazard-1: midpoint row-group selection

**MERGED as #190** (`e4f7f010` = main tip). Retained for its cycle record; the named residues live in the ledger.

Spec: [reconciliation-qb-bug001-work-order.md](reconciliation-qb-bug001-work-order.md) §6. Ledger:
[u3-midpoint-rowgroup-ledger.md](u3-midpoint-rowgroup-ledger.md). Zero-dependency-change unit.

- [x] **P1 — Replace the selection rule.** `ArrowReader::filter_row_groups_by_byte_range`: keep a
      row group iff `rg_start + compressed_size/2 ∈ [start, start+length)`, with `rg_start` =
      Java `getOffset(columns[0])` = `min(data_page_offset, dictionary_page_offset)` read from the
      REAL footer. Delete the `4 + Σ compressed_size` accumulator entirely; no fallback branch.
      Typed `DataInvalid` on zero-column row groups, negative offsets, and midpoint overflow.
- [x] **P2 — New discriminating pins** (T1 straddling exactly-once, T2 bloom-padded offset drift,
      T3 exactly-once partition property over a stride sweep, T4 `getOffset`/boundary/error unit
      matrix). Expectations derived from real footer metadata, never from the synthetic model.
- [x] **P3 — Repair the three self-blind tests** that build their windows with the same
      `4 + Σ compressed_size` model the production code used (`reader.rs` ~3459 / ~4949 / ~5180).
- [x] **P4 — Mutation proof** M1 (OVERLAP rule) · M2 (synthetic offsets, midpoint rule kept) ·
      M3/M4 (boundary flips) · M5 (`getOffset` → dict-wins) · M6 control (must stay GREEN).
- [x] **P5 — Amplifier 4 measured and reported**: annotate the `scan/partition_work.rs`
      split-size-1024 fixture as NON-discriminating (single row group; no live duplication pin).
- [x] **P6 — RIDER (h), reported separately**: make
      `fk5_pos_oracle_sparse_pos_deletes_multi_rg` discriminating (it is green today with
      `max_row_group_row_count = None`) and mutation-prove it RED.
- [x] **P7 — Gate + ledger + `[fork]` commit** in one `&&` chain. Interop leg: see ledger
      §Residue.
- [x] **P8 — Remediation cycle 2** (Critic S3s + Falsifier counterexamples; ledger §11): make the
      fabricated footer fixture multi-column and give it a distinct `total_byte_size` so
      `columns().first()→last()` and `compressed_size()→total_byte_size()` are RED offline; add an
      ODD-size case so `/2 → div_ceil(2)` is RED; add a REAL multi-column pin; fix the stale
      "duplicate rows" rationale in `scan/mod.rs`; re-point the amplifier-4 annotation at the
      load-bearing quantity; name the silent-row-loss and fail-closed-divergence residue.
- [x] **P9 — Remediation cycle 3** (Falsifier counterexamples; ledger §12). Plan:
      - [x] **P9a — AVRO/ORC ranged splits duplicate every row (HIGH, live, same hazard class).**
            `is_splittable` calls AVRO/ORC splittable (the Java `FileFormat` port) and
            `plan_tasks` splits unconditionally, but `process_avro_file_scan_task` /
            `process_orc_file_scan_task` never read `task.start`/`task.length` — every sub-task
            re-reads the whole file. Stop the planner emitting ranged AVRO/ORC tasks, add a
            fail-closed read guard (the `_pos` guard's shape) as defence in depth, and split the
            `scan/map.md` Debug row that currently attributes the symptom solely to parquet.
      - [x] **P9b — Pin the negative-`compressed_size` guard (MZ1 survived).** Add the case to
            the `(a)`–`(g)` semantics matrix and delete the in-tree comment claiming the public
            builder cannot construct a negative size — it can, and the Falsifier did.
      - [x] **P9c — `compressed_size()` overflows before any guard runs.** parquet-rs sums the
            column `total_compressed_size` values with an unchecked `i64` `sum()`; a corrupt
            footer panics (debug) or wraps (release). Sum it here with `checked_add` → typed
            `DataInvalid`.
      - [x] **P9d — Narrow the over-stated residue sentence.** An understated manifest
            `file_size_in_bytes` fails LOUDLY at footer decode, not silently; the silent-row-loss
            residue is real only for under-covering windows / non-tiling `split_offsets`.
      - [x] **P9e — Re-run the full gate and the whole mutation sweep in an ISOLATED tree**
            (`git archive | tar -x`, own `CARGO_TARGET_DIR`, `touch` after extraction) — the
            shared worktree was carrying a sibling agent's uncommitted mutation during cycle 2.
- [x] **P10 — Remediation cycle 4** (Critic C-S2/C-S3 + Falsifier F-1..F-4; ledger §13). Plan:
      - [x] **P10a — F-1: `split` evaporated a whole-file `length == 0` task (HIGH, silent total
            row loss).** `split_fixed_size` starts `remaining = self.length` and loops
            `while remaining > 0`, so the legacy sentinel returned ZERO sub-tasks and
            `mod.rs`'s `split_tasks.extend(...)` dropped the file — `plan_files` returned it,
            `plan_tasks` read 0 rows, no error. Cycle 3 made it an ASYMMETRY (AVRO passes such a
            task through; both reader guards bless the spelling). Fixed by returning `[self]`,
            pinned at the unit AND read level, mutation-proven RED.
      - [x] **P10b — F-2: pin the byte-range ENTRY gate** (`task.start != 0 || task.length != 0`).
            Weakening it to `task.length != 0` was GREEN across 3,146 tests: it turns the empty
            window `[start, start)` into a whole-file read.
      - [x] **P10c — F-3: pin the START half of both whole-file guards.** `start = 1,
            length = file_size_in_bytes` is a genuine window that
            `reject_ranged_whole_file_task` (and the copy-pasted `_pos` guard, which had the same
            gap) would ACCEPT without the `task.start == 0 &&` clause.
      - [x] **P10d — F-4: distinguish the two parquet-mr offset helpers.** `dict Some(0)` must
            still win (`ParquetMetadataConverter.getOffset`, no `> 0`), unlike
            `ColumnChunkMetaData.getStartingPos` which the split-offset WRITER uses.
      - [x] **P10e — C-S2: GAP_MATRIX row R148** — corrected the `FileScanTask::split`
            parenthetical and added the NAMED divergence for the AVRO/ORC decline + the sentinel
            passthrough; anchors green at 75 rows.
      - [x] **P10f — C-S3: lessons entry** — never pipe a gate command into `tail`/`grep`/`head`
            in a verification `&&` chain (the pipeline's status is the LAST command's).

- [x] **P11 — Remediation cycle 5** (Critic ORC-S2 + Falsifier F5..F9; ledger §14). Plan:
      - [x] **P11a — F5: `split` RELOCATED an already-ranged task's window (HIGH, wrong bytes).**
            Both real branches anchor at 0, so re-splitting a `start != 0` parent read bytes it
            never owned and dropped its tail (measured: parent ids 20..59, products ids 0..59).
            Fixed by returning `[self]` for `start != 0` — Java forecloses the shape structurally
            (`SplitScanTask` is not `SplittableScanTask`); pinned at the unit AND read level.
      - [x] **P11b — F7: `plan_tasks` split unconditionally under a `_pos` projection**, which the
            reader then rejects — a total outage of `_pos` on the `plan_tasks` / `PartitionWork`
            seam while `to_arrow()` worked. Suppression hoisted to `plan_tasks`.
      - [x] **P11c — Critic S2 / F9: pin the ORC call site** of
            `reject_ranged_whole_file_task` (deleting it was GREEN; only AVRO was pinned).
      - [x] **P11d — F6: second interop sabotage leg** — mutate the OFFSET SOURCE to the synthetic
            `4 + Σ compressed_size` model; the D2 JAVA verify must go RED with a per-window
            comparison signal. Both Rust legs are blind to it. Plus `JAVA_ROWS` declared instead
            of derived in `assert_exactly_once`.
      - [x] **P11e — notes promoted**: direct four-arm tests for `is_splittable` /
            `reader_honors_byte_range`; the negative-split-offset typed error.
      - [x] **P11f — F8 and the `can_expand` pin DECLINED with executed equivalence proofs**
            (mutants E1/E2/E3 GREEN by construction — ledger §14.6).

- [x] **P12 — Residue cleanup cycle 6** (Critic CONVERGED on `49ee3c5a`; Falsifier 17/17 RED,
      2,340 split/read pairs exactly-once, interop rc=0 both sabotage legs RED — the core is
      sound). SIX S3 items, no redesign, selection predicate untouched. Ledger §15. Plan:
      - [x] **P12a — R1: hoist the `_pos` rule to the split PRIMITIVE** (new branch 1c). The
            public `FileScanTask::split` still manufactured the shape the reader refuses: a task
            projecting `[1, RESERVED_FIELD_ID_POS]` reads 60 rows whole-file but `split(391)`
            returns 3 sub-tasks that ALL fail typed (even `start == 0`, whose length is no longer
            the file size). Both call-site guards retained + re-commented as defensive. M1 RED.
      - [x] **P12b — R2: widen branch (1a) to the PARTIAL parent** (`start != 0` OR
            `length != file_size_in_bytes`). TAKEN, not deferred. The offsets-aware branch took
            manifest offsets verbatim, so a parent owning `[0,500)` of a 1000-byte file with
            offsets `[0,300,700]` covered `[0,700)` plus a degenerate empty window. Chose the
            passthrough over the last-window clip: Java forecloses re-splitting structurally
            (`BaseFileScanTask.length()` is always the file size), the failure direction stays
            bounded to lost parallelism, and the clip would leave the invariant branch-dependent.
            No planner path changes; the `length == 0` sentinel pin stays green. M2 RED.
      - [x] **P12c — R3/R4: `map.md` lockstep** (the repo-contract violations). `scan/map.md`'s
            `task.rs` cell now enumerates all six branches in order (it was short two and called
            branch 1 merely "non-splittable", which means PUFFIN ONLY) + two new Debug rows;
            `tests/map.md`'s `interop_ranged_read.rs` row now says 6 steps / TWO mutations and
            which half of the claim each leg proves (it still said "a source mutation", singular).
      - [x] **P12d — R5: pin BOTH surviving single-`split_offsets` mutants** (MX21 in `split`,
            MX19 in `can_expand`). The two sites deliberately DISAGREE — `split` ports Java's
            `!offsets.is_empty()` gate (Java loses the same rows on a hostile `[585]`, so it is
            parity), `can_expand` requires `> 1` so `to_arrow()` never disagrees with a whole-file
            read. Behaviour UNCHANGED at both; the disagreement is now stated in both tests, both
            comments, and a `map.md` Debug row. M3 + M4 RED.
      - [x] **P12e — R6: mark `can_expand`'s `== Parquet` conjunct defensive**, like the
            `start == 0` one. After cycles 3-6 only the `split_offsets` conjunct is load-bearing.
            E1 (drop both) GREEN at 3159/0 — an executed equivalence proof, error path included.

---


> **Archival log.** Last pass: 2026-07-26 (pass 6 — size trigger, 2,012 lines; run by the RePark
> workstream under the hub concurrency-protocol claim of the same date) →
> [todo-archive/2026-07_audit-hardening-engine-trust.md](todo-archive/2026-07_audit-hardening-engine-trust.md)
> (17 narratives, 2026-07-01 → 07-26) +
> [todo-archive/2026-06_charter-8hour-blocks.md](todo-archive/2026-06_charter-8hour-blocks.md)
> (18 narratives — the June charter / 8-hour blocks / superseded queue). The 2026-07-01 open queue
> kept live and reconciled in place; 7 buried open items lifted to Carried-forward. Prior passes:
> 2026-06-13 (pass 5 — Wave-6/7 → the wave6-wave7 file), 2026-06-12 (pass 4 → wave5), 2026-06-12
> (pass 3 → wave3-wave4), 2026-06-11 (pass 2), 2026-06-09 (pass 1). Procedure:
> [the compaction skill](../.agents/skills/compaction/SKILL.md) §Todo Archival.

## POST-BUNDLE QUEUE (2026-07-26, signed off) — D7 order + D8 toHumanString approval

Spec: [post-bundle-queue-2026-07-26-brief.md](post-bundle-queue-2026-07-26-brief.md). Signed
order: QA (R117 cross-task over-delete) → Unit 3 breaking → QB (writer path bounds) → H7-S2 →
H7-P1, with QC (R161-completion toHumanString parity, format-visible, D8-APPROVED) alongside
Unit 3. Mode A per-unit PRs; SEPMO v2.3 duties. Context at signing: nightly interop FULLY GREEN
(scan-plan arc live-proven); all review branches pruned; main `a08a0957`.

- [x] **QA — per-task DeleteFilter scope** (S1 read correctness; branch
      `fix/delete-filter-per-task-scope`): scope delete APPLICATION to `task.deletes()`
      Java-exact while keeping load caching; RED-first the recorded category=b repro (id 30
      resurrects correctly); eq-delete path checked same-class; R117 🟡→✅ IF this was the sole
      blocker (Actor adjudicates; anchors gate).
      **MERGED #176 (`14921e78`) 2026-07-26**, content-verified R4 (per-task resolve +
      contribution install on main; R117 cell reads ✅). AC converged cycle 1, zero S1/S2 (2 S3
      residues: eq KEYSET fast-path pin coverage; theoretical claim-key namespace collision).
      Java-exact shape: per-SOURCE contribution maps (load-once preserved; G8 claim/notify/
      `Failed` machinery byte-untouched) + per-task application over `task.deletes()` with a
      per-(file, claims) memo; the defect's unattributed path-keyed API REMOVED, `reader.rs`
      unchanged. Eq-deletes PROVEN not-same-class (pinned + mutation-proven); DVs covered.
      Interop crosstask leg added (id 30 survives == Java `{10,30,40,60}`). Lib 2977.
- [x] **Unit 3 — breaking** (`PartitionKey::new -> Result` + `CurrentFileStatus`): live
      main already returns `Result` and uses empty/zero status sentinels. This unit pins
      never-started + post-close `CurrentFileStatus` and names the breaking surface for
      RoadMapSync.
- [ ] **QC — toHumanString parity** (D8-approved, format-visible; FIXED/BINARY hex→base64 +
      Unknown taxonomy + identity(float/double)): alongside Unit 3; format-stability attestation.
- [x] **QB — delete-writer file_path bounds** (fork file-scoped deletes must self-identify;
      parquet-rs 64-byte stat truncation; investigate-first, STOP on any Cargo.toml need).
      **MERGED #184 (`7e26c2a0`) 2026-08-03**, content-verified R4. Landed with BUG-001 (the
      evolved-spec partition stamp) as one PR — both legs of position-delete attachment were
      broken simultaneously and each masked the other. No Cargo.toml need materialised
      (`set_statistics_truncate_length` was already on the pinned parquet). R113 stays 🟡 (owes
      the Java-read interop leg on the evolved-DROP shape); R117 note added.
- [x] **H7-S2** COW streaming. Merged #189 (2026-08-07). `copy_on_write_*` no longer `try_collect`.
- [x] **H7-P1** DML pushdown (prune only; `NOT`-over-dropped-conjunct footgun is a precondition).
      Iceberg scan uses `with_file_prune_only`; exact `WHERE` stays a DataFusion `PhysicalExpr`.
      Nested partial AND no longer converts (so `NOT` cannot invert a dropped conjunct).

**Queue state 2026-08-05.** Since signing, the line has also absorbed: QD/QE (#178/#179, the two
RePark filings — manifest schema tolerance + s3tables replace), the ledger archive (#177), interop
weekly cadence (#180), perf waves A–E (#181), the 07-31 slate (#182), the FK1–FK5 MoR perf campaign
(#183), the V0 DF 52→54 churn map (#185), and the **DF 54.1 / arrow 58.4 family bump re-cut
(#187)** — which moved MSRV 1.92 → 1.94 and toolchain to nightly-2026-03-05.

**Remaining in signed order: QC alongside (Unit 3 pins land with this unit).**

Two things now owed that were not at signing:

1. **RoadMapSync comms for MSRV 1.92 → 1.94 and the DF/arrow/parquet floors** (#187). Downstream-
   visible; RePark must see this before its next repin. Unit 3's own RoadMapSync warning
   (`PartitionKey::new -> Result` + `CurrentFileStatus`) should go out in the same message.
2. **RePark is still pinned at `b009ac15`** — the tip of the superseded pre-recut df54 branch,
   which predates BUG-001 (#184) and the FK campaign (#183). That branch was deleted 2026-08-05;
   the pinned commit is preserved by tag `archive/df54-family-bump-b009ac15` so the pin stays
   reachable. **Retire that tag once RePark repins to main.** Repin target: `3f63a6c7` (#187).

## ACTIVE (2026-07-01): Engine-first closeout — re-ranked open queue

Supersedes the 2026-06-13 queue below. **One home for PRIORITY: this list** (the Roadmap's
re-anchor carries a "Priority home" pointer here; do not grow ranked lists elsewhere). Re-ranked
after the 2026-07-01 review pass, which reconciled the old queue (most items had landed) and
surfaced two new items. Statuses live ONLY in
[docs/parity/GAP_MATRIX.md](../docs/parity/GAP_MATRIX.md).

- [x] **1. Commit-outcome taxonomy (`CommitStateUnknown`)** — DONE, merged #144 (2026-07-08);
      narrative archived in
      [todo-archive/2026-07_audit-hardening-engine-trust.md](todo-archive/2026-07_audit-hardening-engine-trust.md).
      *(Reconciled 2026-07-26, archival pass 6 — the box was never flipped.)* Was: NEW, GAP_MATRIX row R157. An
      unknown-outcome `ErrorKind` (or flag) honored by the retry gate + sent-vs-unsent
      transport-error classification in the Glue / S3 Tables / REST / SQL catalogs +
      surfaced-no-retry-no-cleanup semantics matching Java + mock-catalog tests. Buildable
      WITHOUT AWS creds. Slots ahead of CDC: the named consumer commits continuously against
      S3 Tables, whose service-side maintenance ALSO commits concurrently — an ambiguous outcome
      today risks a duplicate commit (see the row cell). The credentialed conformance slice
      stays with item 6.
- [ ] **2. CDC row-level changelog** (re-anchor item 2) — **RE-CHARACTERIZED 2026-07-31 (G3
      ledger):** mostly **parity-correct as-is**. `ChangelogOperation::UpdateBefore` /
      `UpdateAfter` are declared for API parity (`scan/task.rs`) but are **never emitted by the
      core planner** — Java 1.10.0 `BaseIncrementalChangelogScan` only produces INSERT/DELETE
      task kinds; collapsing delete+insert into update pairs is an **engine-side** step (Spark
      `ChangelogIterator`), not owed by `iceberg-core`/`iceberg-api`. Residual (if any engine
      pull) is accepting ranges that carry row-level DELETE manifests
      (`IncrementalChangelogScan` is whole-data-file-level today) — engine-gated, not a
      library correctness hole.
- [ ] **3. ORC/Avro DATA-read residue** (re-anchor item 3) — footer codec / nested + V3 types /
      the Avro `timestamptz` mapping — pull only if the engine queries non-parquet tables.
- [ ] **4. ENGINE_CONTRACT.md recipes → NORMATIVE** — bytecode/oracle-verify the
      isolation-level → validation table (DRAFT landed 2026-07-01,
      [docs/ENGINE_CONTRACT.md](../docs/ENGINE_CONTRACT.md)) against Java 1.10.0
      `SparkWrite` / `SparkCopyOnWriteOperation` / `SparkPositionDeltaWrite`, one interop
      conflict scenario per cell.
- [x] **5. Nightly interop CI** — DONE: the Nightly Interop workflow runs the suites on `main`
      on a schedule and is fully green as of 2026-07-26 (scan-plan arc live proof landed).
      *(Reconciled 2026-07-26, archival pass 6 — the box was never flipped.)* Was: run the
      `dev/java-interop/` suites on a schedule. The oracle is
      the model-tier equalizer only if it runs unprompted; this is the cheap 80% of Phase 7.
- [ ] **6. Real-catalog hardening (credentialed)** — Glue + S3 Tables conflict/retry conformance
      + item 1's real-catalog classification slice; scheduled with the user (needs AWS creds).

**In-flight (off-matrix, user-gated — staged work, not ranked above): H7 DML
streaming/pushdown** on the DataFusion reference impl (scope converged 2026-06-30; engine-first
hardening of the #124 DML loop, flips no matrix row). **H7-S1** (MoR DELETE/UPDATE streaming) landed #140.
**H7-S2** (COW streaming) landed #189.
**H7-P1** (pushdown pruning) landed: Iceberg `with_file_prune_only` on DELETE/UPDATE;
exact `WHERE` stays the DataFusion filter. Nested partial AND is not converted.

PULL-BASED / DEMOTED: unchanged from the Roadmap re-anchor — link, do not restate.

## Carried-forward open items (detail in todo-archive/)

Lifted verbatim from archived narratives by archival pass 6 (2026-07-26). Status caveat for the
three 07-17 audit units below: **no merged PR names them** (sibling unit D from the same list
shipped as #159), but they were never explicitly closed either — verify against the code /
GAP_MATRIX before starting one. Full context:
[todo-archive/2026-07_audit-hardening-engine-trust.md](todo-archive/2026-07_audit-hardening-engine-trust.md)
(the "OVERNIGHT BLOCK (2026-07-17)" section) and
[todo-archive/2026-06_charter-8hour-blocks.md](todo-archive/2026-06_charter-8hour-blocks.md)
(the "SUPERSEDED 2026-07-01" queue + "BLOCK 10").

- [x] **B (OO max): MoR eq-delete panic/hang** — **LANDED** (reconciled 2026-07-31 G3). Cite
      `caching_delete_file_loader.rs` (`equality_ids: None` → `DataInvalid`, not unwrap) and
      `delete_filter.rs` (oneshot sender-drop → terminal `Failed` + `notify_waiters`). Merged
      with the 07-18 audit bundle (#160 / follow-ons); content-verified on main.
- [x] **C (OO max): predicate serde arity validation (SAF-004)** — **LANDED** (reconciled
      2026-07-31 G3). Custom `Deserialize` on Unary/Binary/Set validates op/arity at the wire
      boundary; visitor dispatch returns typed `Err` instead of `panic!`. Pins in
      `expr/predicate.rs` `serde_arity_pins`.
- [x] **E (OO max): typed error kinds** — **LANDED** (reconciled 2026-07-31 G3). SQL helpers
      (`no_such_*` / `*_already_exists_*`) emit typed kinds; HMS thrift mappers in
      `crates/catalog/hms/src/error.rs` + call-site wiring in `catalog.rs` map
      `NoSuchObjectException` / `AlreadyExistsException` (and drop-namespace not-empty) to
      `NamespaceNotFound` / `TableNotFound` / `NamespaceAlreadyExists` / `TableAlreadyExists` /
      `NamespaceNotEmpty`. Config: empty required fields → `DataInvalid`; malformed/
      unresolvable address and missing StorageFactory → `Unexpected`. Unit G3
      (`fix/hms-typed-error-kinds`) closed the ledger + residual config pins; mapper unit
      tests offline.
- [ ] **2. Multi-spec write interop** — STILL OPEN (reconciled 2026-07-01; citations corrected
      same day). TWO distinct residues: (a) the manifest-merge LAYOUT gap —
      `MergeManifestProcess` is not routed into the non-append merging actions (the `RowDelta`
      row, currently row R106 — the old "row 94" pointer was dead); (b) the writer-layer spec
      threading — `DataFileWriter`/`DeletionVectorWriter` stamp the table default spec (row R110)
      — plus the multi-spec-DATA interop slices on the merging actions (one slice landed: #69,
      multi-spec RowDelta DELETE-commit); `fast_append` multi-spec is ✅ (Z2 — the template).
- [ ] **Multi-spec MERGING-path wiring gap** — the companion detail of item 2(a) above: route
      `MergeManifestProcess` into the non-append merging actions. The RE-CHARACTERIZED
      2026-06-16 narrative (what is already ported vs the real gap) is in the "BLOCK 10"
      section of the 2026-06 archive — read it before scoping; the earlier framing was a
      phantom bug.
- [ ] **4. geometry / geography types** — HALF DONE (reconciled 2026-07-01): `unknown` landed ✅
      2026-06-17 (interop-proven); geometry/geography remain ❌ and are DEMOTED to opportunistic
      by the 2026-06-21 engine-first re-anchor (a query engine does not pull them).
- [ ] **7. [PARKED] encryption** — reconciled 2026-07-01: the Glue / S3Tables VIEWS half is
      RESOLVED as parity-correct-unsupported (rows R126/R127, verified 2026-06-17 — NOT owed);
      encryption remains ❌ and is DEMOTED to opportunistic by the engine-first re-anchor. The
      credentialed real-catalog hardening piece moved to the 2026-07-01 queue (item 6).

## Archived increment narratives

Completed-increment narratives moved verbatim out of this file (see [the compaction skill](../.agents/skills/compaction/SKILL.md)
§Todo Archival). Not session-start reading — grep/open on demand.

- [todo-archive/phase1.md](todo-archive/phase1.md) — Phase 1 spec & metadata completeness (schema /
  partition / snapshot evolution + spec-read robustness).
- [todo-archive/phase2.md](todo-archive/phase2.md) — Phase 2 write engine (write actions + the
  concurrent-commit conflict-validation cluster, incl. the merged write-validation PR #9).
- [todo-archive/phase3.md](todo-archive/phase3.md) — Phase 3 scan parity (residual evaluation,
  inspection tables, scan-metrics emission, and inspection / scan-execution interop).
- [todo-archive/2026-06_ops-hardening.md](todo-archive/2026-06_ops-hardening.md) — the doc-infrastructure / hardening meta-sprints (not phase work).
- [todo-archive/2026-06_wave3-wave4-overnight.md](todo-archive/2026-06_wave3-wave4-overnight.md) — Waves 3–4 + the overnight session (PRs #25–#37; pass-scoped).
- [todo-archive/2026-06_wave5.md](todo-archive/2026-06_wave5.md) — Wave 5 (PRs #39–#41; pass-scoped).
- [todo-archive/2026-06_wave6-wave7.md](todo-archive/2026-06_wave6-wave7.md) — Waves 6–7 (PRs #43–#47; pass-scoped): the I1/I2/I3 interop increments + O1/O2/O3 + R1/R2/R3.
- [todo-archive/2026-07_audit-hardening-engine-trust.md](todo-archive/2026-07_audit-hardening-engine-trust.md)
  — the 2026-07-01 → 07-26 audit / hardening / engine-trust era (pass 6).
- [todo-archive/2026-06_charter-8hour-blocks.md](todo-archive/2026-06_charter-8hour-blocks.md)
  — the 2026-06-13 → 06-19 charter / 8-hour blocks / superseded queue (pass 6).
- Index: [todo-archive/map.md](todo-archive/map.md).

## ACTIVE (2026-08-15): FB-2 `position_deletes` schema-only stub

Owner-authorized bounded increment (conductor-13F addendum, option (b)): the
`MetadataTableType::PositionDeletes` variant + `inspect/position_deletes.rs` `schema()`
transcribed from Java `PositionDeletesTable.calculateSchema` (fixed metadata-column ids,
partition child-id reassignment, empty-partition drop; v3 DV columns behind the format
gate). Scan refused loud (`FeatureUnsupported`). `$`-name resolution and the provider
`table_names` enumeration extend automatically; the enumeration pin flip is declared.
Partition type: cross-spec unified as of conductor-16 increment D (#204).
Residual: `PositionDeletesBatchScan` port (R142).

## CLOSED (2026-08-16): conductor-16 — partitioning unification (increments A-E)

The `Partitioning.partitionType` / `PartitionUtil.coercePartition` campaign is complete:
A `spec/partitioning.rs` module (#202), B `partitions` adoption (#203), D
`position_deletes` schema (#204), C `files`/`entries`/`all_*` projection (#205), E docs
sweep (this change): new row R165, R138/R140/R145/R93/R95 notes, and the
`rewrite_position_delete_files` accepted-finer grouping residue recorded on R165.
Remaining to flip R165 green: a bidirectional interop round-trip (out of this campaign's
scope per the charter — `make interop` excluded).

## CLOSED (2026-08-22): `RewritePositionDeleteFiles` size-based admission gate

**Delivered.** The charter and its five binding addenda are in
[rpdf-size-gate-2026-08-21-brief.md](rpdf-size-gate-2026-08-21-brief.md); the 45-clause proof
ledger, the build carving and the fourteen residues are in
[rpdf-size-gate-2026-08-21-ledger.md](rpdf-size-gate-2026-08-21-ledger.md), whose dated
**CLOSE-OUT (2026-08-22)** section carries the per-clause dispositions, the eighteen charter
defects and the five residues this unit opened (R-G7-1..5). The PR body is
[rpdf-size-gate-pr-body.md](rpdf-size-gate-pr-body.md).

**The defect.** `rewrite_position_delete_files.rs:222` admits any `(spec, partition)` group with
two or more live position-delete files. Java's `BinPackRewritePositionDeletePlanner` admits a group
only through `enoughInputFiles || enoughContent || tooMuchContent`, whose file-count floor is
**five**. Reported by the RePark engine side (MW-2) with live Spark 4.0.1 / Iceberg 1.10.0
measurements; independently re-verified against `9f85a086`. No wrong answers — the fork compacts
*more* than Java, so this is a parity fix, not a bug fix.

**Base ref** `9f85a086`. **The plan said one PR; it shipped as FOUR** — the seven build groups were
carved across four merges, and the two unrelated PRs that interleaved (#210 the AGENTS.md spine
move, #211 an h2 bump that edited `Cargo.lock`) belong to other units:

| PR | Squash commit | Carried |
|---|---|---|
| #207 | `51edcc2c` | scope charter (brief + ledger + this plan block) |
| #208 | `972b932c` | G1 CONFIG, G2 PLANNER, G3 WRITER |
| #209 | `77ddf5d4` | G4 COMMIT LOOP, G5 TESTS |
| #212 | `a4cdc419` | G6 DOCS + STATUS (re-cut over #210/#211) |

**The BREAKING flip.** The default admission floor moves 2 → 5, so existing two-file callers become
no-ops unless they set `.min_input_files(2)`. Three further behaviour flips ship with it (an
unbindable `filter` errors earlier; the output and snapshot shape is per BIN and can be more than
one file; a pre-existing `write.delete.target-file-size-bytes` in
`{unparsable, > i64::MAX, <= 1, == i64::MAX}` now makes `execute` return `Err`). All four are named
in the PR body. Public surface is purely additive — five builder methods plus two `TableProperties`
consts. Status is on matrix rows R136 (corrected, keeps `✅`) and R135 (sibling roll bound).

Lib suite at `a4cdc419`: 3378 passed / 0 failed / 1 ignored (`cargo test -p iceberg --lib`).

**Seven ordered build groups** (45 clauses, clause-complete, each clause in exactly one group):
G1 CONFIG (7) · G2 PLANNER (8) · G3 WRITER (6) · G4 COMMIT LOOP (3) · G5 TESTS (7) ·
G6 DOCS + STATUS (7) · G7 GOVERNANCE (7). G4 depended on G3; G5 on G1-G4. Every group ran a
single Actor with an independent fresh-context Critic. **G1–G6 converged**; G7's convergence is
G7's Critic's call to record, not the Actor's ([AGENTS.md](../AGENTS.md) `<subagent_policy>`:
"convergence is the Critic's call"), and G7's first review REMANDED.

**Doc obligations carried in the same change:** R136 currently claims a "1:1 port" while the gate is
absent — the sentence is corrected, not dropped, and the row keeps `✅` with named residues (R-8).
A sibling roll-bound divergence is logged against R135 (RES-1). The `rewrite_position_delete_files`
grouping residue already recorded on R165 by the C16 docs sweep is RES-2 here — one home, cross-link
rather than restate.

**Known residues that stay open** (14, each homed): the sibling roll bound (RES-1), the
`(spec_id, partition)` grouping key (RES-2), the fork-only non-Parquet skip (RES-3), per-bin commit
granularity (RES-4), and ten others enumerated in the ledger.

**Follow-ups FILED by this unit, deliberately NOT fixed in it** (each a pre-existing defect that this
change does not falsify, so R-10's in-scope test does not reach it):

- **Sequence-direction inversion in `crates/iceberg/src/transaction/rewrite_files.rs`** (R-13, named
  residue). SEVEN statements about the seq stamped on a rewritten DELETE file, verified at source
  2026-08-22 — FIVE inverted, TWO direction-less. INVERTED: the module rustdoc's "Added-delete
  SEQUENCE NUMBER" paragraph, `add_delete_file`'s rustdoc, `add_delete_file_with_sequence_number`'s
  rustdoc, the in-code comment on the ADDED-DELETE negative-seq guard, and the user-facing
  `DataInvalid` message that same guard emits — all say a HIGHER (inherited) seq makes the delete
  stop applying and RESURRECT rows, with `add_delete_file_with_sequence_number` also calling a LOWER
  seq an over-apply. DIRECTION-LESS: the `added_delete_files` field doc ("stops applying (rows
  resurrect) or over-applies") and `test_rewrite_add_delete_file_negative_sequence_number_rejected`'s
  rustdoc ("resurrection/over-deletion"). Seven is the MEASURED set, not a ceiling — re-derive it in
  the unit rather than trusting this count. Backwards: `delete_file_index.rs`'s
  `applicable_pos_deletes` keeps a delete whose `delete_seq >= data_seq`, so a HIGHER stamp reaches
  data it never masked and OVER-APPLIES while a LOWER stamp RESURRECTS. The same paragraphs also give
  the applicability rule as the STRICT `data_seq < delete_seq`, which is `applicable_eq_deletes`'
  equality-delete rule, not the position-delete one. `add_delete_file_with_sequence_number` is the
  very call `RewritePositionDeleteFiles` now fans out from one to N. Wrong before this PR, so outside
  its manifest; open a bounded doc unit. (The DATA-file paragraphs in the same file — a fresh higher
  seq on an added DATA file making outstanding equality deletes stop applying — are CORRECT and must
  not be "fixed" alongside them.)
- **Stale bare-number matrix citation "GAP_MATRIX row 134"** in `crates/iceberg/tests/map.md` and
  `dev/java-interop/map.md` (C-038 residue). R134 is `DeleteOrphanFiles`; the intended row is R136.
  Both files are OUT of this unit's manifest (R-10) and both stay accurate otherwise, so the
  citation is recorded here rather than drive-by fixed. Bare-number citations are exactly the class
  `make check-matrix-anchors` was built to retire — fix with the `row R136` anchor form.

**Residues opened at close-out (G7, 2026-08-22)** — FIVE were opened (R-G7-1..5); the THREE that
are actionable in this repo are listed here. R-G7-1 (C-017's pin unmet) has nothing to fix
retroactively and R-G7-5 (R2's six-field mandate vs the claim board's table shape) is homed on
`PrimarySync/Concurrency-Protocol.md` and its owner. Full text in the ledger's CLOSE-OUT section:

- **The R7 pre-merge gate never ran in full for this unit** (R-G7-2). `make test` begins with
  `docker-up` and the Docker daemon is unavailable on this host, so the Docker-backed leg was never
  executed; `make check-msrv` and `cargo deny check advisories` are likewise unevidenced anywhere in
  this unit's artefacts. Clause C-018 asserted the whole gate runs green before merge, and that
  assertion was never true here. This is a named instance of the standing Docker-`make test` gap,
  not a new one — but do not read the PR body's Verification block as an R7 transcript.
- **`crates/iceberg/src/maintenance/mod.rs` names only Puffin V3 DVs** — the sentence "Puffin V3
  DELETION VECTORS are SKIPPED (file-scoped, never bin-packed) — V2 PARQUET only", at `:82-83`
  measured 2026-08-22 (G6-5 cited `:80-81`; already stale when written — find it by its text) —
  where the action file, the test battery and matrix row R136 also name V2 ORC/Avro. **DEFERRED
  deliberately:** the existing sentence ("V2 PARQUET only") is TRUE, so this is consistency and not
  accuracy, and re-editing a true sentence is the over-correction failure mode this unit hit twice.
  A future unit may widen it; it must verify the sentence is false first.
- **R3 was violated by #208 and #209** (R-G7-3) — both carry GitHub's auto-generated branch-name
  titles with no `[repark]` tag, and no merged PR title names the BREAKING flip. Nothing to fix
  retroactively; recorded so the next unit tags its PRs at open time rather than at merge time.

## F-7 U1 — `first_row_id` suppression at the merging-producer add seam (2026-08-25)

Branch `parity/f7-u1-suppress-first-row-id`, base `249a9556b`. Two stages, one branch.

- [x] Stage 1 — port Java `MergingSnapshotProducer.add(DataFile)` → `Delegates.suppressFirstRowId`.
      Seam: a REQUIRED `FirstRowIdPolicy` argument on `SnapshotProducer::new`, so every call site
      states its policy and a new producer cannot inherit one by omission. `FastAppend` +
      `RewriteManifests` pass `Preserve` (both extend Java `SnapshotProducer`); the six merging
      producers plus `CherryPickOperation` pass `Suppress`.
- [x] Stage 1 evidence — domain table over the seven producers of the charter's partition, plus a
      per-rule mutation run with its arithmetic.
- [x] Stage 2 — extend the row-lineage interop fixture with a `RewriteFiles` and an
      `OverwriteFiles` commit so `Existing` entries exist, and pin that a survivor keeps its
      `first_row_id` and its per-row `_row_id`, both directions.
- [x] Stage 2 evidence — measure the `== Added` vs `!= Deleted` mutation named in row R166.
- [x] Matrix cells + full done gate.

Outcome: Stage 1 landed as a required `FirstRowIdPolicy` argument on `SnapshotProducer::new`.
Stage 2 found a REAL divergence and it is fixed in the same change: the fork emitted the
carried-forward manifests before the newly written ones, where Java emits new first, and the V3
manifest-list writer assigns row-id ranges in list order — so a newly added file took a row id
Java does not give it (15 against Java's 12 on the rewrite fixture). Both facts are recorded in
row R166. The order has three conjuncts and each is now mutation-pinned separately; the
data-before-deletes one is `MergingSnapshotProducer.apply`'s shape alone, and is unreachable
because no producer on either side emits a delete manifest ahead of a data manifest.
Stage 2's named mutation now goes RED, but only through a V2-to-V3 upgrade fixture: a
rewrite reads its source through the assigning reader on both sides, so an ordinary rewrite's
survivor already carries a stored id. NOT built and escalated: Java's `add(ManifestFile)`
`first_row_id` precondition has no fork surface to land on (see the R166 residue).
