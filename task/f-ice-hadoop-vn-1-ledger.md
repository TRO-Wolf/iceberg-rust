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

# Ledger — F-ICE-HADOOP-VN-1: a Hadoop vN metadata commit never overwrites an existing version file

**Ledger id:** `F-ICE-HADOOP-VN-1-2026-09-16`
**Branch:** `fix/ice-hadoop-vn-1` (cut off fork `main` = `edc38c6a`)
**Scope:** run 19b unit ICE-HADOOP-VN-1, FORK side, round 1
**Matrix rows touched:** R167 (status cell only), R110 (cited, unchanged)
**Model:** muse-spark-1.3-contributor

## 1. Defect (measured 2026-09-16, rating row V2-20c, probes `conc` / `conc2`)

A Spark Hadoop-catalog table at `v2.metadata.json` is registered into two RePark memory
catalogs (each a fork `MemoryCatalog`). Catalog 1 inserts, writing `v3.metadata.json`.
Catalog 2, whose pointer is still v2, inserts: `with_next_version` also yields
`v3.metadata.json`, and `TableMetadata::write_to` overwrites catalog 1's file. Both
catalogs and Spark then read catalog 2's rows only. In `conc2` a stale RePark overwrites
Spark's own v3. Each `MemoryCatalog` pointer CAS is in-process, so it cannot see the other
writer; the file name is the only shared state. Java `HadoopTableOperations.commit` writes
a temp file and `renameToFinal` fails when `vN` exists
(`CommitFailedException: Version N already exists`); it never overwrites.

Root cause in the fork: `MemoryCatalog::update_table`
(`crates/iceberg/src/catalog/memory/catalog.rs`, step 2) writes the staged location with
`TableMetadata::write_to`, which is an unconditional overwrite (`OutputFile::write` →
`Storage::write`). The in-process CAS steps before and after the write compare only the
catalog's own pointer, so two instances sharing one FileIO tree both pass and the second
write destroys the first commit's bytes.

## 2. Clauses

| Clause | Statement | Proven by |
|---|---|---|
| C-1 | A commit whose next metadata location is a Hadoop `vN(.gz).metadata.json` that already exists fails with a typed commit-conflict error and leaves the existing file byte-identical | `hadoop_second_commit_to_same_vn_fails_and_preserves_winner` |
| C-2 | The losing catalog's pointer is unchanged | same test, `metadata_location()` still v2 |
| C-3 | The winning commit's rows are intact when the table is re-read from the file | same test, winner `load_table` shows its property; v3 bytes equal pre-loss bytes |
| C-4 | Hive/REST `<version>-<uuid>` names keep their current behaviour | `uuid_names_keep_overwrite_behaviour` control: two catalogs commit over one uuid base, both `Ok`, distinct files |
| C-5 | Two concurrent tasks racing the same `vN` produce exactly one winner, via atomic create-new, not exists-then-write, where the storage supports it | `hadoop_concurrent_commits_yield_exactly_one_winner` (barrier-synced, tempdir FileIO) + per-backend table below |
| C-6 | The same guard on every fork catalog that writes Hadoop `vN` names through this seam | §5: memory + sql + glue `update_table` routed through `write_commit_metadata`; s3tables routed though it can never emit `vN` (`register_table` is `FeatureUnsupported`); rest writes no files; hms `update_table` unimplemented |

## 3. D-1: error kind — retryable `ErrorKind::CatalogCommitConflicts`

Choice: the exclusive-create conflict surfaces as retryable
`ErrorKind::CatalogCommitConflicts`, the same kind every CAS in the fork already returns.

Reasons:

1. Java parity: the retried class in `SnapshotProducer.commit` is the commit-conflict class
   (`onlyRetryOn(CommitFailedException)`); a Hadoop version collision IS that class.
2. R110 refresh-and-retry is preserved for the shared-catalog race: a loser that shares the
   winner's catalog reloads the winner's pointer on retry, rebases to `v(N+1)`, and lands.
   A non-retryable kind would permanently fail that writer, contradicting the pinned
   refresh-and-retry behaviour (`memory/catalog.rs` test near line 2917).
3. No infinite loop on a memory catalog whose stale pointer never refreshes: the fork retry
   loop (`Transaction::commit`, `transaction/mod.rs`) is a bounded `backon` retry gated on
   `e.retryable() && kind != CommitStateUnknown`, with `max_times = commit_num_retries`
   (default 4) and a total-delay cap (default 30 min). The never-refreshing loser burns its
   budget re-hitting the guard and then surfaces the conflict error. Bounded retries
   terminate; only an unbounded loop would hang, and none exists on this path.

RePark's own retry loop could not be read from this clone (RePark is not present in the
fork workspace); the decision rests on Java parity plus the fork's bounded retry. If RePark
retries retryable commit errors without a budget, that is a RePark-side finding, recorded in
open questions rather than solved by mistyping the fork error.

## 4. Per-backend atomicity of `Storage::write_new`

| Backend | Mechanism | Atomic | Evidence |
|---|---|---|---|
| Local filesystem (`LocalFsStorage`) | `std::fs::OpenOptions::create_new(true)` | yes, kernel-atomic create | race test, exactly one winner over many runs |
| In-memory (`MemoryStorage`) | single write-lock check-and-insert | yes within one process | unit test `memory_write_new_refuses_existing` |
| OpenDAL (`OpenDalStorage`) | `write_with(...).if_not_exists(true)`; `AlreadyExists`/`ConditionNotMatch` map to the conflict; `Unsupported` falls back to exists-then-write | native where the service honours it (fs, memory, S3 `If-None-Match: *`); check-then-write fallback elsewhere | unit test on the memory backend; service-backed conditional writes not exercised offline |
| Any other `Storage` implementor | defaulted trait body: exists-check then write | no (TOCTOU); same guarantee as before this change, plus a fail-closed error instead of a silent overwrite | documents the default in the trait body |

## 5. Catalog seam survey (C-6)

All four file-writing `update_table` paths derive the staged location from
`TableCommit::apply` → `MetadataLocation::with_next_version`, which preserves the stored
pointer's convention, and all four now write through `TableMetadata::write_commit_metadata`:

- `crates/iceberg/src/catalog/memory/catalog.rs` (`update_table`, staged `write_to` call):
  Hadoop pointers arrive via `register_table` of a `vN` file. Guarded.
- `crates/catalog/sql/src/catalog.rs` (`update_table`, staged `write_to` call):
  `register_table` stores an arbitrary caller-supplied location, so a Hadoop pointer is
  possible. Guarded.
- `crates/catalog/glue/src/catalog.rs` (`update_table`, staged `write_to` call):
  `register_table` stores an arbitrary caller-supplied location. Guarded.
- `crates/catalog/s3tables/src/catalog.rs` (`update_table`, staged `write_to` call):
  `register_table` returns `FeatureUnsupported` (no register-by-location operation), and
  `create_table` always mints `new_with_table_location` uuid names, so the pointer is
  always uuid and `with_next_version` never emits `vN`. Routed through the same call so a
  future register path inherits the guard; behaviour for uuid names is unchanged.
- `crates/catalog/rest/src/catalog.rs` (`update_table`): POSTs requirements/updates to the
  service; the client writes no metadata file. No seam, unchanged.
- `crates/catalog/hms/src/catalog.rs` (`update_table`): unimplemented
  (`_commit` is ignored). No seam, unchanged.
- `TableMetadata::write_to` itself is unchanged (other writers — create paths, view paths,
  `rewrite_table_path` uuid staging — keep overwrite semantics).

`write_commit_metadata` branches on the parsed location: unparsable locations and
Hive/REST uuid names take the existing `write_to` path byte-for-byte (C-4); only
successfully parsed Hadoop-convention (`id == None`) locations take the exclusive path.

## 6. Red-first evidence (unfixed tree)

New tests: `crates/iceberg/tests/hadoop_version_commit.rs` (sequential C-1..C-4 pins) plus
the race pin in the same file. Command:

```text
cargo test -p iceberg --test hadoop_version_commit
```

Unfixed-tree output (`cargo test -p iceberg --test hadoop_version_commit`, 2026-09-16):

```text
test hadoop_second_commit_to_same_vn_fails_and_preserves_winner ... FAILED
test hadoop_concurrent_commits_yield_exactly_one_winner ... FAILED
test uuid_names_keep_overwrite_behaviour ... ok
test result: FAILED. 1 passed; 2 failed

---- hadoop_second_commit_to_same_vn_fails_and_preserves_winner stdout ----
stale second commit to v3 must fail: "/tmp/.tmp5h21fR/ns/src/metadata/v3.metadata.json"

---- hadoop_concurrent_commits_yield_exactly_one_winner stdout ----
assertion `left == right` failed: exactly one racer wins v3
  left: 2
 right: 1
```

The stale commit returns `Ok(v3)` (silent overwrite — the reported data loss), both
racers win, and the uuid control already passes (it pins unchanged behaviour).

## 7. Fix

- `crates/iceberg/src/io/storage/mod.rs`: defaulted `Storage::write_new` (exists-then-write,
  fail-closed `PreconditionFailed`).
- `crates/iceberg/src/io/storage/memory.rs`: atomic single-lock override.
- `crates/iceberg/src/io/storage/local_fs.rs`: atomic `create_new` override; `mod tests`
  moved to `crates/iceberg/src/io/storage/local_fs_tests.rs` (file sat exactly at its
  1063-line ceiling; ceilings only move down).
- `crates/storage/opendal/src/storage_impl.rs` (new): `impl Storage for OpenDalStorage`
  moved out of `lib.rs` (which sat exactly at its 2078-line ceiling) plus the
  `if_not_exists` override with `Unsupported` fallback.
- `crates/iceberg/src/spec/table_metadata_commit.rs` (new): `impl TableMetadata` carrying
  `write_commit_metadata` (`table_metadata.rs` sat exactly at its 4575-line ceiling).
- `crates/iceberg/src/catalog/metadata_location.rs`: `is_hadoop_convention` accessor.
- `crates/iceberg/src/io/file_io.rs`: `FileIO::write_new` delegate.
- One-line call-site swaps (`write_to` → `write_commit_metadata`) in memory, sql, glue,
  s3tables `update_table` (all four files sat exactly at ceiling; swaps are net-zero).
- `docs/parity/GAP_MATRIX.md` row R167 residue sentence replaced with the dated fix.

## 8. Gates and counts (2026-09-16, all green)

| Gate | Result |
|---|---|
| `cargo test -p iceberg --lib` | 3679 passed, 0 failed, 8 ignored |
| `cargo test -p iceberg --test hadoop_version_commit` | 3 passed (2 red-first, 1 control) |
| `cargo test -p iceberg --lib io::` | 114 passed, incl. the 2 new `write_new` unit tests |
| `cargo test -p iceberg-storage-opendal --lib` | 47 passed, incl. the new memory-backend `write_new` test |
| `cargo test -p iceberg-catalog-sql --lib` | 80 passed |
| `cargo test -p iceberg-catalog-glue --lib` | 50 passed |
| `cargo test -p iceberg-catalog-s3tables --lib` | 39 passed |
| `cargo clippy --all-targets --all-features --workspace -- -D warnings` | clean (plus a `-p iceberg` re-run after the final edits) |
| `cargo fmt --all -- --check` | clean |
| `make check-toml` (taplo) | clean, no TOML touched |
| `cargo machete` | no unused dependencies |
| `make check-agent-artifacts` | OK |
| `make check-matrix-anchors` | OK (84 rows anchored) |
| `make check-comment-blocks` | OK after trimming one moved 7-line block to 5 |
| `make check-rust-file-size` | 467 files clean; ceilings lowered (`local_fs.rs` row removed at 537 lines, opendal `lib.rs` 2078 → 1939) |
| `typos .` | clean after 4 ledger word fixes |
| License headers | verbatim ASF header on all 4 new `.rs` files + the ledger, matching siblings |

No `map.md` covers any touched directory (maps exist only under `.agents/skills`,
`task/`-archives and `crates/sketches`), so per the navigation contract no map update
was due.

## 9. Open questions

1. RePark-side retry budget (see §3): confirm RePark bounds retries on retryable commit
   conflicts so a cross-instance stale pointer surfaces instead of spinning. For the
   orchestrator; not blocking the fork contract.
2. `version-hint.text` remains unwritten (pre-existing R167 residue, unchanged).
3. Gzip Hadoop reads (`vN.gz.metadata.json`) still commit forward as uncompressed
   `v(N+1).metadata.json` (pre-existing `with_next_version` behaviour, unchanged).
