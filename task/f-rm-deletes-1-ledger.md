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

# F-RM-DELETES-1 — `rewrite_manifests` rewrites DELETE manifests too (ledger)

**Branch:** `fix/f-rm-deletes-1`, cut from fork main. RePark slate `IPI-11`, fork half.
**Scope:** an opt-in on `RewriteManifestsAction` that re-clusters DELETE manifests with
`RewriteManifestsSparkAction` semantics. RePark's `CALL rewrite_manifests` switch (and its
`spec_id` argument) is run 24c's half — untouched here.
**Oracle:** the run-24d rewrite-manifests oracle — Spark 4.1.2 + Iceberg 1.11.0
(`rewrite_manifests_truth.json`, `rewrite_manifests_truth2.json`, recorders beside them).

## 1. The Java rule — verified from `iceberg-spark-runtime-4.1_2.13-1.11.0.jar` bytecode

### `org.apache.iceberg.spark.actions.RewriteManifestsSparkAction`

`javap -c -p` on the 1.11.0 jar.

- **Ctor**: `spec = table.spec()` (default spec; `specId(int)` re-points it after
  `Preconditions.checkArgument(table.specs().containsKey(id), "Invalid spec id %s")`);
  `predicate = manifest -> true` (`lambda$new$0` returns constant 1);
  `targetManifestSizeBytes = PropertyUtil.propertyAsLong(props, "commit.manifest.target-size-bytes", 8388608)`;
  `shouldStageManifests = formatVersion == 1 && !compatibility.snapshot-id-inheritance.enabled`.
- **`doExecute()`** runs TWO independent legs and unions their results:
  `rewriteManifests(ManifestContent.DATA)` then `rewriteManifests(ManifestContent.DELETES)`.
  Both rewritten lists empty ⇒ `EMPTY_RESULT` (no commit at all). Else
  `replaceManifests(rewritten, added)` then `Result(rewritten, added)`.
- **`findMatchingManifests(content)`**: `table.currentSnapshot()` null ⇒ empty list;
  `loadManifests(content, snapshot)` = `snapshot.dataManifests(io)` or
  `snapshot.deleteManifests(io)`; filtered by
  `manifest.partitionSpecId() == spec.specId() && predicate.test(manifest)`
  (`lambda$findMatchingManifests$5`). The spec filter AND the user predicate apply to BOTH
  content legs.
- **`rewriteManifests(content)`**: matching empty ⇒ `EMPTY_RESULT`;
  `numManifests = ceil(totalSizeBytes(matching) / targetManifestSizeBytes)`
  (`targetNumManifests`: `(total + target - 1) / target`; `totalSizeBytes` sums
  `manifest.length()` and `ValidationException.check(hasFileCounts(m), "No file counts in manifest: %s")`);
  **`numManifests == 1 && matching.size() == 1` ⇒ `EMPTY_RESULT`** (a single manifest that
  already fits is never rewritten); then entries are written through Spark
  (`buildManifestEntryDF` reads live entries; unpartitioned spec ⇒
  `writeUnpartitionedManifests`, else `writePartitionedManifests` which repartitions+sorts on
  the partition columns before writing).
- **`WriteManifests.call`**: rows `[snapshotId, sequenceNumber, fileSequenceNumber(nullable),
  file struct]` → `RollingManifestWriter.existing(file, snapshotId, seqNum, fileSeqNum)` —
  provenance is materialized explicitly, never re-inherited.
- **`ManifestWriterFactory`**: data leg ⇒ `ManifestFiles.write(formatVersion, spec, output,
  snapshotId)`; delete leg ⇒ `ManifestFiles.writeDeleteManifest(formatVersion, spec, output,
  snapshotId)`. Rolling at `maxManifestSizeBytes`.
- **`replaceManifests(rewritten, added)`**: core `table.rewriteManifests()` →
  `deleteManifest` × rewritten, `addManifest` × added, `commit(SnapshotUpdate)` — ONE core
  commit for both legs. The `rewritten`/`added` counts RePark's CALL returns are the two
  leg-unioned lists; the snapshot summary keys (`manifests-created`/`kept`/`replaced`) come
  from core `BaseRewriteManifests.summary()` (created = addedManifests + written,
  replaced = deletedManifests, kept = the untouched rest).
- **`use_caching`** is an execution detail only (`withReusableDS` cache of the manifest-entry
  DataFrame); `part_mor_nocache` produces byte-identical outcomes in the oracle.

### `org.apache.iceberg.BaseRewriteManifests` (core — the fork's current analogue)

- `performRewrite` per manifest: `containsDeletes(m) || !matchesPredicate(m)` ⇒ KEPT;
  `containsDeletes` = `manifest.content() == ManifestContent.DELETES`. **Core keeps delete
  manifests byte-identical — the Spark action, not core, owns the delete leg.**
- Rewritten path: `ManifestFiles.read(...).select("*").liveEntries()` (DELETED-status entries
  skipped) → `appendEntry(entry, clusterByFunc.apply(entry.file()), manifest.partitionSpecId())`.
- `WriterWrapper.addEntry`: `writer == null` ⇒ open one; `writer.length() >= manifestTargetSizeBytes`
  ⇒ close + reopen; `writer.existing(entry)`. Writers keyed `Pair.of(key, specId)` via
  `getWriter` → `newManifestWriter(spec)` (a DATA writer — core never builds a delete writer).
- `apply`: `validateDeletedManifests` (deleted set ⊆ current, path equality), `performRewrite`,
  `validateFilesCounts` (`activeFilesCount(created) == activeFilesCount(replaced)`, added +
  existing counts, "Missing file counts in %s", "Replaced and created manifests must have the
  same number of active files: %d (new), %d (old)"), `withSnapshotId` on added manifests,
  new manifests first then added then kept.
- `summary()`: `manifests-created` / `manifests-kept` / `manifests-replaced` /
  `entries-processed`, computed values overwriting user `set()` on the same keys.

## 2. The oracle shapes (run-24d rewrite-manifests oracle, Spark 4.1.2 / Iceberg 1.11.0)

Manifest rows `[content, spec_id, data_files, delete_files, added_data, existing_data,
added_delete, existing_delete]`; `result` = `[rewritten, added]` as the CALL returns them.

| shape | before | result | after | summary | rows |
|---|---|---|---|---|---|
| `unpart_mor_v2/v3` | 3 data (1 added each) + 2 delete (1 added each), all spec 0 | `[5,2]` | 1 data (3 existing) + 1 delete (2 existing) | created 2 / kept 0 / replaced 5, op `replace` | 2,3,5,6 |
| `part_mor_real_v2/v3` | 4 data (2 added each) + 3 delete (two with 1 added, one EMPTY — the third delete metadata-deleted a whole file) | `[7,2]` | 1 data (8 existing) + 1 delete (2 existing) | created 2 / kept 0 / replaced 7, op `replace` | 1-6,10 |
| `part_mor`, `part_mor_spec`, `part_mor_nocache`, `no_deletes` | 3 data only | `[3,1]` | 1 data | created 1 / kept 0 / replaced 3 | — |
| `evolved_spec` | 1 data spec 0 (5 existing) + 1 EMPTY data spec 1 | `[0,0]` | unchanged | — | — |

`evolved_spec` mechanics: the spec-0 manifest fails the `partitionSpecId == spec.specId()`
filter (default spec is now 1); the empty spec-1 manifest matches but hits the
`numManifests == 1 && matching.size() == 1` already-optimal skip. Both legs empty ⇒ no commit.
The recorded `op`/`summary` for that cell are the previous snapshot's (a `delete`), because
no new snapshot exists.

## 3. The fork's measured behavior — the defect

`crates/iceberg/src/transaction/rewrite_manifests.rs` is a faithful port of Java CORE
`BaseRewriteManifests`: `perform_rewrite` rewrites only `ManifestContentType::Data` manifests
(≈L425-431), and `ClusterWriters` builds only `build_v*_data` writers via
`new_cluster_manifest_writer(spec_id, ManifestContentType::Data)`. Under RePark's CALL —
`cluster_by(|_| "")` + `rewrite_if(spec == current)` — on the `unpart_mor` shape the fork
answers `rewritten 3, added 1` and carries both delete manifests byte-identical, where the
Spark action answers `5, 2` and ends with ONE delete manifest.

Measured RED (step-2 pin run, pre-implementation): recorded in §5 after the pin run —
each pin's failure output is the fork's verbatim answer.

## 4. Design — the opt-in

`RewriteManifestsAction.rewrite_delete_manifests(bool)` (default `false` ⇒ Java core
semantics preserved). When `true` and `cluster_by` is set, a `Deletes` manifest that passes
`rewrite_if` is re-clustered like a data manifest BUT into its own DELETE manifests:
`ClusterWriters` keys become `(cluster_key, spec_id, content)` and the writer is built via
`new_cluster_manifest_writer(spec_id, content)` (`build_v2_deletes` / `build_v3_deletes` —
the same machinery row-delta uses). Entries go through `add_existing_entry` — the fork
analogue of Java's `writer.existing(entry)` / Spark's `RollingManifestWriter.existing` —
so snapshot id, data sequence number, and file sequence number are written explicitly.

Semantics the opt-in inherits for free, matched to the oracle:

- A rewritten delete manifest with NO live entries produces no output but counts toward
  `manifests-replaced` (the `part_mor_real` empty delete manifest: replaced side 7 includes
  it; created side unaffected — `validate_files_counts` balances at +0).
- `rewrite_if` gates delete manifests exactly as Spark's `findMatchingManifests` predicate
  gates both legs.
- `manifests-created`/`kept`/`replaced` count both content kinds (oracle: created 2 /
  kept 0 / replaced 5 on `unpart_mor`).
- `manifest_file`'s `(data, deletes)` partition in `snapshot.rs` reorders the final list
  data-then-deletes anyway, so writer-finish order inside `ClusterWriters` is not load-bearing.
- V3 row-id ranges: `assign_first_row_id` forces `first_row_id = None` on Deletes manifests
  and only advances `next_row_id` over Data manifests — a delete manifest interleaved in the
  new set cannot disturb data-manifest ranges.

Named residue (NOT this unit): Spark's `numManifests == 1 && matching.size() == 1`
already-optimal skip is a Spark-action rule with no core analogue — the fork's action
rewrites any matching manifest unconditionally, so `evolved_spec`-style single-manifest
inputs still produce a commit where Spark produces none. That is a pre-existing DATA-leg
divergence, unchanged by this flag (see §5 OPEN).

## 5. Pin map / verdicts

| pin | risk pinned | status |
|---|---|---|
| `test_rewrite_delete_manifests_clusters_deletes_into_one_delete_manifest_v2` | unpart_mor_v2: 2 delete manifests → 1, entries Existing, seqs/fseqs/snapshot ids preserved, scan still applies deletes | OPEN (written red) |
| `test_rewrite_delete_manifests_clusters_dvs_into_one_delete_manifest_v3` | unpart_mor_v3: DVs re-cluster into a delete manifest, seqs preserved, scan applies | OPEN (written red) |
| `test_rewrite_delete_manifests_drops_manifest_with_no_live_entries` | part_mor_real: an emptied delete manifest is replaced by nothing | OPEN (written red) |
| `test_rewrite_delete_manifests_summary_counts_cover_both_content_kinds` | created/kept/replaced = 2/0/5, op `replace` | OPEN (written red) |
| `test_rewrite_delete_manifests_default_off_keeps_deletes` | flag off ⇒ byte-identical carry-forward (core default) | covered by existing test 5 + explicit `false` pin |
| `test_rewrite_delete_manifests_respects_rewrite_if` | predicate-false delete manifest kept | OPEN (written red) |
| evolved_spec no-commit | Spark `numManifests==1 && size==1` skip — no core analogue | OPEN residue (§4) |

Mutation obligations (step 4): (a) revert the content gate so deletes stay immune ⇒ the
three cluster pins must go red; (b) re-stamp entries (`add_entry` path semantics) ⇒ the seq
pins must go red.
