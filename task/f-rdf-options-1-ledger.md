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

# F-RDF-OPTIONS-1 — `RewriteDataFiles` options parity (Java `RewriteDataFilesSparkAction` 1.11.0)

Model: muse-spark-1.3-contributor. Base: fork main `edc38c6a`. Branch `fix/ice-rdf-options-1`.
Matrix row: R135. Oracle: `/tmp/oc-worker/ic-build/rdf_options_spark.json`
(PySpark 4.1.2 + iceberg-spark-runtime-4.1_2.13:1.11.0, probe `rdf_options_probe.py`).
Fixture shape throughout the oracle: `(id BIGINT, p INT, v STRING) PARTITIONED BY (p)`,
2 partitions x 4 single-INSERT files of 50 rows (error cells: 1 partition x 2 files).

## 1. Clauses

| id | behaviour | oracle cell | verdict | evidence |
|---|---|---|---|---|
| C-001 | Default is ONE commit for all file groups (8 files / 2 groups -> 8 snapshots + 1 `replace`) | `min_input_files_1`, `rewrite_all` | PROVEN green | |
| C-002 | `partial_progress(true)` batches commits; `partial-progress.max-commits` (default 10) caps commits as groups-per-commit = ceil(groups / max_commits); 2 groups -> 2 commits; max-commits 1 -> 1 commit; 4 groups + max-commits 3 -> 2 commits; max-commits 0 + enabled fails | `partial_progress`, `partial_progress_max1`, `partial_progress_groups`, `err_max_commits_0` | PROVEN green | |
| C-003 | `rewrite_all(true)` selects every file and qualifies every group, bypassing size filters and `min-input-files` | `rewrite_all`, `target_small` | PROVEN green | |
| C-004 | `output_spec_id(id)` writes and groups under that spec; unknown id fails; after `DROP PARTITION FIELD p`, id 0 writes 2 files under spec 0, default writes 1 file under spec 1 | `output_spec_id`, `output_spec_id_current`, `err_bad_spec` | PROVEN green | |
| C-005 | `rewrite_job_order`: none (default), bytes-asc, bytes-desc, files-asc, files-desc, case-insensitive; orders rewrite/commit/`file_groups`; bad name fails | `job_order_*`, `err_bad_job_order` | PROVEN green | |
| C-006 | `max_concurrent_file_group_rewrites(n)`: 0 fails; positive accepted, runs sequentially | `concurrent`, `err_concurrent_0` | PROVEN green | |
| C-007 | Threshold messages match Java; `delete-ratio-threshold` float renders Java-style (`2.0`, not `2`) | `err_*` | PROVEN green | |

## 2. Java mechanism (from the oracle cells)

| fact | source |
|---|---|
| Default (no partial progress) compacts 2 partition groups into a single `replace` snapshot (8 -> 9) | `min_input_files_1`, `rewrite_all` after-ops end in one `replace` |
| Partial progress commits one group per commit at default max-commits 10 (8 -> 10); max-commits 1 folds 2 groups into 1 commit (8 -> 9) | `partial_progress`, `partial_progress_max1` |
| 4 groups (max-file-group-size-bytes 2500 splits each partition in two) with max-commits 3 land in 2 commits (8 -> 10), i.e. ceil(4/3) = 2 groups per commit | `partial_progress_groups` |
| `Cannot set partial-progress.max-commits to 0, the value must be positive when partial-progress.enabled is true` | `err_max_commits_0` error |
| `Cannot use output spec id 99 because the table does not contain a reference to this spec-id.` | `err_bad_spec` error |
| `Invalid rewrite job order name: bogus` | `err_bad_job_order` error |
| `Cannot set max-concurrent-file-group-rewrites to 0, the value must be positive.` | `err_concurrent_0` error |
| `'delete-ratio-threshold' is set to 2.0 but must be <= 1` (Java `Double.toString` keeps `.0`) | `err_delete_ratio` error |
| Unknown/empty/upper-case option keys are rejected, not ignored | `err_unknown_key`, `err_empty_key`, `err_upper_key` |
| Unparseable bool (`rewrite-all=maybe`) is a silent no-op (result zeros, no error) | `err_bad_bool` |

## 3. Red-first

### 3a. Behavioural red on the base tree (existing API only)

`rewrite_data_files_options_tests::test_default_commits_all_groups_in_one_snapshot`
(2 partitions x 4 files, `min_input_files(1)`, base tree) FAILED as predicted:

```
assertion `left == right` failed: default rewrites all groups in ONE commit (Java RewriteDataFilesCommitManager without partial progress)
  left: 10
 right: 9
```

The fork committed once per group (2 `replace` snapshots) — Java's partial-progress shape, not its default.

### 3b. Compile red for the new API

The full options suite (builder methods `rewrite_all`, `partial_progress`,
`partial_progress_max_commits`, `output_spec_id`, `rewrite_job_order`,
`max_concurrent_file_group_rewrites`, enum `RewriteJobOrder`) does not compile on the base tree:

```
error[E0599]: no method named `rewrite_all` found for struct `RewriteDataFiles`
error[E0599]: no method named `partial_progress` found for struct `RewriteDataFiles`
error[E0432]: unresolved import `crate::maintenance::RewriteJobOrder`
```

## 4. Production change

| file | change |
|---|---|
| `maintenance/rewrite_data_files_plan.rs` | `ResolvedConfig.rewrite_all` + bypass in `plan_file_groups` (all tasks admitted, all bins qualify, packing kept); grouping keys on the passed output spec; `RewriteJobOrder` + `FromStr` (Java message, case-insensitive, `-`/`_`); `order_groups`; `plan_commit_batches` (one batch off, else ceil(groups/max_commits) groups per commit); `format_java_double` (`2.0`, `NaN`, `Infinity`); `PARTIAL_PROGRESS_MAX_COMMITS_DEFAULT = 10` |
| `maintenance/rewrite_data_files.rs` | Six additive builder methods; `execute` writes every group then commits per batch (one commit by default); `rewrite_group` split into `write_group` + batch commit; `resolve_output_spec_in` (unknown id fails with the Java message); `resolve_config` validates max-concurrent (always) and max-commits (only when partial progress is on); Java float rendering for the ratio messages; Deferred doc table updated |
| `maintenance/rewrite_data_files_write.rs` | `write_compacted_files` takes the output spec instead of always the default spec |
| `maintenance/rewrite_data_files_dangling_tests.rs` | New file: the 4 composed-dangling tests moved out of `rewrite_data_files.rs` unchanged (file-size ceiling split) |
| `maintenance/rewrite_data_files_options_tests.rs` | New file: C-001..C-007 pins |
| `maintenance/rewrite_data_files_router_bound_tests.rs` | 2 `write_compacted_files` call sites take the default spec |
| `maintenance/mod.rs` | Register the 2 test modules; re-export `RewriteJobOrder` |

Option validation runs before the no-snapshot early return (`resolve_config`, then output-spec
resolution, then the snapshot check), so an unknown spec id fails even on a fresh table.
`current_snapshot_id` widened to `pub(crate)` for the moved dangling tests.

Comment-grep triage (RULE 0): every `+//` line is one of (a) a demanded one-line doc on a new
public item (6 builders, `RewriteJobOrder` + 5 variants), (b) a keep-true edit of an existing
comment (`execute` doc, group-to-batch and default-to-output-spec rewords), or (c) a verbatim
move into `rewrite_data_files_dangling_tests.rs` (F-24 split precedent, same model). The one
new private doc drafted on `write_group` was deleted before commit.

## 5. Pins

| pin | asserts |
|---|---|
| `test_default_commits_all_groups_in_one_snapshot` | C-001: 8 files / 2 groups, `min_input_files(1)` -> +1 snapshot, rewritten 8, added 2, `file_groups` 2, rows conserved |
| `test_rewrite_all_selects_and_qualifies_everything` | C-003: well-sized pair (empty plan by default) + lone file both rewritten under `rewrite_all(true)` |
| `test_partial_progress_commits_one_group_per_commit` | C-002: +2 snapshots, same counts as default |
| `test_partial_progress_max_commits_1_folds_into_one_commit` | C-002: +1 snapshot |
| `test_partial_progress_4_groups_max_commits_3_needs_2_commits` | C-002: 4 groups (tuned max group size), file_groups 4, +2 snapshots |
| `test_partial_progress_max_commits_0_fails_when_enabled` | C-002: exact Java message; and max-commits 0 with progress OFF is accepted (Q1 recommendation) |
| `test_output_spec_id_0_after_drop_writes_two_files_under_spec_0` | C-004: 2 groups / 2 files, live spec ids {0}, rows conserved |
| `test_default_spec_after_drop_writes_one_file_under_spec_1` | C-004: 1 group / 1 file, live spec ids {1} |
| `test_output_spec_id_unknown_fails` | C-004: exact Java message |
| `test_rewrite_job_order_bytes_desc_orders_file_groups` | C-005: desc/asc byte order in `file_groups`; `NONE` keeps plan order |
| `test_rewrite_job_order_parse` | C-005: case-insensitive, `-`/`_` spellings, exact bad-name message |
| `test_max_concurrent_0_fails_positive_runs_sequentially` | C-006: exact Java message; 4 behaves as one commit |
| `test_delete_ratio_threshold_2_renders_java_float` | C-007: `2.0` rendering; `min-input-files` 0 message unchanged |
| `test_commit_batching_math` + `test_format_java_double` | unit pins for `plan_commit_batches` and `format_java_double` |

## 6. Gates

| command | exit | evidence |
|---|---|---|
| `cargo test -p iceberg --lib rewrite` | 0 | 278 passed, 0 failed |
| `cargo test -p iceberg --lib maintenance` | 0 | 350 passed, 0 failed |
| `cargo clippy -p iceberg --all-targets -- -D warnings` | 0 | clean |
| `make check` | 0 | fmt, clippy, toml, machete, agent-artifacts, matrix-anchors, comment-blocks, rust-file-size (ceiling 2659 lowered to 2503) all green |

## 7. Open questions

| id | question | premise | recommendation |
|---|---|---|---|
| Q1 | `partial-progress.max-commits: 0` with partial progress OFF — reject or accept? | The oracle only pins the enabled case (`err_max_commits_0` passes `partial-progress.enabled=true`). Java validates inside the partial-progress branch. | Accept (validate only when enabled); pinned by `test_partial_progress_max_commits_0_fails_when_enabled` second half. |
| Q2 | Unknown option keys (`foo`, empty, upper-case): the oracle rejects them, but the fork builder is typed (unknown keys are unrepresentable). Port a stringly options map? | The fork has no `options(HashMap)` entry point; every knob is a typed builder method, so an unknown key cannot arrive. | No map port; the typed surface makes the class unreachable. Named here, not in GAP_MATRIX. |

## 8. Out of scope observed

- `Result.failedDataFilesCount`: the oracle result tuple carries it (always 0 here); the fork `RewriteDataFilesResult` has no such field and the brief does not ask for one. A failure fails the action (default) or the batch (partial), so the count is only observable mid-partial-run.
- Orphan cleanup of written-but-uncommitted files on failure: unchanged from the base tree (neither the old per-group path nor Java's default path is modified here).
- Sort / Z-order strategies and oversized-file splitting stay deferred (R135 Deferred row unchanged for those).
