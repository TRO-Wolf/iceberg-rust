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

# F-APPEND-WINDOW-FAILLOUD-1 — the incremental append scan fails loud on a non-append snapshot, when asked

**Date:** 2026-10-05. **Base:** fork `main` `e1d74bef` (the RePark pin), branch
`feat/incremental-append-fail-on-non-append`. **Model:** Claude Opus 5.5 (`claude-opus-5-5`),
medium effort. **Card:** RePark `task/roadmap/mid-term/f-append-window-failloud-1-2026-10-05.md`
(RePark PR #953). **RePark consumer:** MB-1, the micro-batch batch source (plan D-2, O-5).

This ledger retires when the unit lands and RePark repins, or when the owner removes it.

## What changed

- `IncrementalAppendScanBuilder` gains three opt-in options, all default `false`:
  - `with_fail_on_non_append(bool)`;
  - `with_skip_overwrite_snapshots(bool)`, Spark's `streaming-skip-overwrite-snapshots`;
  - `with_skip_delete_snapshots(bool)`, Spark's `streaming-skip-delete-snapshots`.
- The skip options only act while fail-on-non-append is on. With it off, `plan_files` is
  today's silent append-only walk (Java batch `IncrementalAppendScan`, ICE-CHANGELOG-1 C-003).
- The window walk moved out of `incremental.rs` into `scan/incremental/window.rs`
  (`appends_between` + `NonAppendPolicy`). The move keeps `incremental.rs` under its legacy
  ceiling; the ceiling was lowered 2976 → 2968. Pins live in `scan/incremental/window_tests.rs`.
- **The error.** `ErrorKind::PreconditionFailed`, the crate's analogue of the
  `IllegalStateException` that Java's `Preconditions.checkState` throws. The message names the
  operation, the snapshot id and the window:
  `Cannot process overwrite snapshot <id> in the incremental append window (<from>, <to>]; to skip
  overwrite snapshots, set with_skip_overwrite_snapshots(true)`. An absent `from` prints `root`.
- **Which snapshot is named.** The walk runs newest-first, but the refusal reports the OLDEST
  refused snapshot in the window, because Spark's stream meets snapshots oldest-first and throws
  on the first one. Pinned by `fail_on_non_append_names_the_oldest_refused_snapshot_first`.
- Public API: additive only (three builder methods). No break.

## Java 1.11.0 evidence — `replace` is skipped silently

Jar: `/tmp/sparkenv/ivy/jars/org.apache.iceberg_iceberg-spark-runtime-4.1_2.13-1.11.0.jar`,
sha256 `d6ea6c5d099288daeb7d5a92061bd3d7d8f296492632b42378e5f2f0e3066242`. In 1.11.0
`shouldProcess` lives in `org.apache.iceberg.spark.source.BaseSparkMicroBatchPlanner` (not in
`SparkMicroBatchStream`); `nextValidSnapshot` (offset 101) and `SyncSparkMicroBatchPlanner`
(offset 124) call it. `javap -c -p`, `protected boolean shouldProcess(Snapshot)`:

- offsets 0-115: `snapshot.operation()` string-switched to an index: `"append"` → 0,
  `"replace"` → 1, `"delete"` → 2, `"overwrite"` → 3.
- offset 148-149 (`append`): `iconst_1; ireturn` — process.
- offset 150-151 (`replace`): `iconst_0; ireturn` — **skip, no check, no error.**
- offsets 152-173 (`delete`): `readConf.streamingSkipDeleteSnapshots()` then
  `Preconditions.checkState(skip, "Cannot process delete snapshot: %s, to ignore deletes, set
  %s=true", snapshotId, "streaming-skip-delete-snapshots")`; returns `false` when it passes.
- offsets 174-195 (`overwrite`): the same with `streamingSkipOverwriteSnapshots()` and
  `"Cannot process overwrite snapshot: %s, to ignore overwrites, set %s=true"`,
  `"streaming-skip-overwrite-snapshots"`.
- offsets 196-234 (default): `IllegalStateException("Cannot process unknown snapshot operation:
  %s (snapshot id %s)")`. The Rust `Operation` enum is closed over the four values, so this arm
  has no Rust counterpart.

The fork follows it: `replace` never refuses, with or without the option.

## Pins, red first

The fixture lineage is `s0 append, s1 append, s2 overwrite, s3 append, s4 delete, s5 replace,
s6 append` on the memory catalog; each commit asserts its own `Operation`.

| Pin | Window | Expect |
|---|---|---|
| `fail_on_non_append_refuses_an_overwrite_between_appends` | `(s0, s3]`, on | `PreconditionFailed` naming s2, `overwrite`, `(s0, s3]` |
| `fail_on_non_append_refuses_a_delete_between_appends` | `(s2, s6]`, on | refusal naming s4, `delete` |
| `fail_on_non_append_skips_a_replace_silently_as_spark_does` | `(s4, s6]`, on | `{s6}` |
| `fail_on_non_append_names_the_oldest_refused_snapshot_first` | `(s0, s6]`, on | refusal naming s2, not s4 |
| `skip_overwrite_skips_only_overwrites` | `(s0, s3]` / `(s0, s6]` | `{s1, s3}` / refusal naming s4 |
| `skip_delete_skips_only_deletes` | `(s2, s6]` / `(s0, s6]` | `{s3, s6}` / refusal naming s2 |
| `both_skips_yield_every_append_in_the_window` | `(s0, s6]` | `{s1, s3, s6}` |
| `option_off_keeps_todays_silent_skip` | `(s0, s6]`, default and explicit off + a skip flag | `{s1, s3, s6}` |
| `fail_on_non_append_keeps_the_exclusive_from_and_inclusive_to_bounds` | `(s2, s3]`; inclusive from s2; `(s3, s4]` | `{s3}`; refusal `(s1, s3]`; refusal naming s4 |
| `fail_on_non_append_keeps_the_empty_range_empty` | `from == to` direct, build-time rejection, replace-only window | empty; `Err` at build; empty |

**Red** (builder options wired, refusal absent): `cargo test -p iceberg --lib
scan::incremental::window_tests` exit 101 — 4 passed, 6 failed. The six refusal pins failed with
`unwrap_err()` on an `Ok` value, e.g. `{"s1.parquet", "s3.parquet"}` for the overwrite pin. The
four that passed are the ones that pin unchanged behaviour (replace, both skips, option off,
empty range).

**Green:** `cargo test -p iceberg --lib scan::incremental` exit 0 — 42 passed (32 existing + 10
new).

**Mutation** (drop the check: `if policy.fail_loud` → `if false && policy.fail_loud` in
`window.rs`): `fail_on_non_append_refuses_an_overwrite_between_appends` FAILED, exit 101.
Source restored from a copy; the restored file is the committed one.

## Gates

Recorded in the hand-back of the round (`/tmp/oc-worker/direct/wo/fork-microbatch/handback.json`)
with real exit codes.

## Not done here

- No Java interop leg: the behaviour lives in Spark's streaming planner, not in `iceberg-core`,
  so there is no core oracle to cross-check against.
- RePark exposure of the skip options is ruled in the MB-1 packet (card: "Whether RePark exposes
  them is ruled in the packet").
- **A V1 snapshot without a summary** (found by the scoped verifier, V365-S3-1, 2026-10-05). The
  spec reader treats a V1 snapshot that has no `summary` as an `append` (`crates/iceberg/src/spec/snapshot.rs:435`),
  so fail-loud lets it through. Java's `BaseSparkMicroBatchPlanner.shouldProcess` would throw a
  `NullPointerException` there. This predates the PR and is not changed by it: a Bronze table written by any
  current engine carries a summary on every snapshot. A V1 table without summaries is outside the
  micro-batch source's contract, and MB-1 refuses V1 sources if the packet rules so.
