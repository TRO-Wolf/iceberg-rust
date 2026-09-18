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

# F-RP-SUMMARY-USER-1 — a caller's `replace-partitions` summary value wins, as Java's does (ledger)

**Branch:** `fix/f-rp-summary-user-1` off fork `main` `9e67e000`.
**Scope:** `ReplacePartitionsAction::commit` overwrote a caller-supplied `replace-partitions`
snapshot-summary value with the literal `"true"`. Java lets the caller's later `set` win.
**Commits:** `77d8870d` red cells · `75cde964` fix · docs commit (this ledger + todo).

## 1. The defect

`crates/iceberg/src/transaction/replace_partitions.rs`, `ReplacePartitionsAction::commit`
(pre-fix lines 340–344) cloned the caller's `snapshot_properties` map and then ran
`snapshot_properties.insert(REPLACE_PARTITIONS_PROP, "true")` — inserting the marker AFTER the
caller's entries, so an explicit caller value for the same key was overwritten.

## 2. The Java / Spark measured answer

Java `BaseReplacePartitions`'s constructor calls `set(SnapshotSummary.REPLACE_PARTITIONS_PROP,
"true")` on the inherited `SnapshotProducer` summary map; a caller's later `.set(k, v)` on the
operation lands in the same map and therefore overrides the constructor value.

Measured on PySpark 4.1.2 + Iceberg 1.11.0 (2026-09-18, run 22b):

- dynamic `insertInto` with snapshot property `replace-partitions=false` → the snapshot summary
  lands `"false"`;
- `overwritePartitions()` with `false` → `"false"`; with `true` → `"true"`;
- no caller value (and a caller setting an unrelated key `k=v`) → `"true"`, and `k=v` is kept.

## 3. The site and the fix

`commit` now layers the default UNDER the caller's map:

```rust
let mut snapshot_properties = self.snapshot_properties.clone();
snapshot_properties
    .entry(REPLACE_PARTITIONS_PROP.to_string())
    .or_insert_with(|| "true".to_string());
```

`entry().or_insert_with` writes `"true"` only when the caller did not set the key — the exact
constructor-then-`set` ordering Java implements. The map then flows unchanged into the summary via
`SnapshotProducer` (`snapshot.rs` `additional_properties.extend(self.snapshot_properties.clone())`).

Two comment removals accompanied the code, both deletions under the comment ban (no rewording):

- The `commit` body comment describing the old layering order.
- `set_snapshot_properties`'s two-line doc comment, which asserted the old contract ("an explicit
  value here does not clear it") — false post-fix. Deleting it trips `#![deny(missing_docs)]`, so
  the method carries `#[allow(missing_docs)]` (the same escape the brief prescribes for
  `clippy::missing_errors_doc`; precedent in `expr/mod.rs:37`). The corrected contract is the one
  this ledger and the four test cells record.

`replace_partitions.rs` shrank 2801 → 2800 and its legacy ceiling in
`scripts/check_rust_file_size.py` was lowered to match (the checker's own shrink instruction; the
file stays at its exact ceiling at every commit).

## 4. Test cells

The cells were appended to the existing test module
`crates/iceberg/src/transaction/replace_partitions/tests/replace_partitions_extracted.rs` (the
brief's "another existing test module" option — no new `mod` line, so `replace_partitions.rs` held
its exact 2801 ceiling at the red commit). Each commits a replace-partitions action over the memory
catalog and reads `current_snapshot().summary().additional_properties`.

| Cell | Input (`set_snapshot_properties`) | Test | Pin |
|---|---|---|---|
| (a) | `{"replace-partitions": "false"}` | `test_replace_partitions_summary_caller_false_wins` | summary value `"false"` |
| (b) | `{"replace-partitions": "true"}` | `test_replace_partitions_summary_caller_true_wins` | `"true"` |
| (c) | `{"k": "v"}` | `test_replace_partitions_summary_unrelated_key_keeps_default` | `"true"` and `k == "v"` |
| (d) | none | `test_replace_partitions_summary_default_is_true` | `"true"` |

## 5. Red output (pre-fix, commit `77d8870d`)

```
running 4 tests
test ...::test_replace_partitions_summary_caller_true_wins ... ok
test ...::test_replace_partitions_summary_default_is_true ... ok
test ...::test_replace_partitions_summary_unrelated_key_keeps_default ... ok
test ...::test_replace_partitions_summary_caller_false_wins ... FAILED

---- test_replace_partitions_summary_caller_false_wins stdout ----
assertion `left == right` failed: a caller-provided replace-partitions=false must override the
action default (Java `set` order)
  left: Some("true")
 right: Some("false")

test result: FAILED. 3 passed; 1 failed; 3823 filtered out
```

Exactly cell (a) is red — the caller's `"false"` was overwritten by the inserted `"true"`.

## 6. Green output (post-fix, commit `75cde964`)

```
cargo test -p iceberg --lib replace_partitions
test result: ok. 45 passed; 0 failed; 0 ignored; 3782 filtered out
```

All four cells pass; the whole replace-partitions + cherry-pick-replace suite (45 tests filtered on
`replace_partitions`, including `test_replace_partition_marks_old_file_deleted_and_sets_marker` and
the three `test_cherrypick_replace_partitions_*` tests) stays green.

## 7. Mutation proof

Reverted ONLY the fix (restored `snapshot_properties.insert(REPLACE_PARTITIONS_PROP, "true")` over
the caller map; tests and the doc/`#[allow]` change kept), ran the cell filter:

```
test ...::test_replace_partitions_summary_caller_false_wins ... FAILED
  left: Some("true")
 right: Some("false")
test ...::test_replace_partitions_summary_caller_true_wins ... ok
test ...::test_replace_partitions_summary_unrelated_key_keeps_default ... ok
test ...::test_replace_partitions_summary_default_is_true ... ok
test result: FAILED. 3 passed; 1 failed
```

Identical red signature to the pre-fix run. Restored; all four cells green again. The revert was
never committed (`git status` clean after restore).

## 8. Audit — every `replace-partitions` read/write site in the fork

`grep 'replace-partitions' crates/` plus both `REPLACE_PARTITIONS_PROP` constants. Java reference
read: `CherryPickOperation.java` at tag `apache-iceberg-1.10.0` (fetched source; no Java checkout on
this box).

| Site | Read/Write | Caller-`"false"` effect | Java-shaped? |
|---|---|---|---|
| `transaction/replace_partitions.rs` `commit` | write (the fix site) | marker lands `"false"` | yes — constructor default under caller `set` |
| `transaction/cherry_pick.rs` `is_replace_partitions` (L178–186) | **read** — `operation == Overwrite && props["replace-partitions"] == "true"` routes the OVERWRITE replay | `== "true"` → false → the else path: `isFastForward` check → fast-forward, else `Cannot cherry-pick snapshot %s: not append, dynamic overwrite, or fast-forward`; the replace branch's ancestor + WAP-publish checks are skipped identically | **yes** — Java `cherrypick` L93-95 gates the same branch on `PropertyUtil.propertyAsBoolean(summary, REPLACE_PARTITIONS_PROP, false)`, and `parseBoolean("false")` is false: identical path, identical rejection |
| `transaction/cherry_pick.rs` Replay produced snapshot (L415–428) | write of the PUBLISHED snapshot's props | unaffected — only `source-snapshot-id` + optional `published-wap-id` are stamped; the marker is never re-stamped | yes — Java `CherryPickOperation.cherrypick` sets exactly `SOURCE_SNAPSHOT_ID_PROP` + `PUBLISHED_WAP_ID_PROP` on the produced snapshot (verified in the 1.10.0 source); a published cherry-pick of a replace carries no marker on either side |
| `tests/common/snapshot_meta_view.rs` `SUMMARY_COUNT_KEYS` (L42–68) | read (interop oracle) | the `"false"` value is copied verbatim into the JSON view — same as Java `SnapshotMetaOracle.SUMMARY_COUNT_KEYS`, which carries the same key and copies values verbatim | yes — pure copy, no gate on the value |
| `tests/interop_expire.rs` (L99–116) | read (expire oracle allowlist) | same verbatim-copy idiom | yes |
| `tests/interop_write_actions_meta.rs`, `interop_write_data.rs`, `interop_replace_partitions_conflict.rs` | comments / fixture commits | fixtures commit plain replaces (no caller props) → still `"true"` | yes |
| `transaction/mod.rs:257`, module docs | prose only | — | — |
| `transaction/snapshot.rs:1405-1406` `additional_properties.extend(snapshot_properties)` | the plumbing that lands the action map into the summary | carries whatever value `commit` layered — no marker logic of its own | yes |

**Verdict:** for the caller-`"false"` defect every site already does what Java does; no site needed a
change beyond the fix.

**Named residual (pre-existing, FIXED in round 2):** `cherry_pick.rs`'s gate was
`value == "true"` (case-sensitive); Java's `PropertyUtil.propertyAsBoolean` is
`Boolean.parseBoolean` = case-INSENSITIVE `equalsIgnoreCase("true")`. Before round 1 a Rust caller
could not leave a case-variant value in the summary (always overwritten to `"true"`); post-round-1 a
caller CAN write `"TRUE"`, which Java's cherry-pick reads as a replace and this gate did not. The
divergence itself predated the lane for externally-authored metadata — round 1 only widened its
reachability. Closed in round 2 below.

## 9. Notes

- The `#[allow(missing_docs)]` on `set_snapshot_properties` is per-call-site and exists only because
  `#![deny(missing_docs)]` otherwise requires a doc comment the comment ban forbids writing; the
  reason cannot live in code (it would be a comment), so it lives here.
- No GAP_MATRIX row flips: this is a behavior fix inside an existing capability, not a new
  capability closure, and no interop runner covers caller-supplied summary props.

---

## Round 2 — case-insensitive `replace-partitions` read in cherry-pick

**Commits:** `7de9ab5a` red cells · `ce36f58a` fix · docs commit (this section + todo).

Round 1's own audit finding, folded in because round 1 is what makes a caller-set value reachable:
`CherryPickAction::is_replace_partitions` gated the OVERWRITE replay on `value == "true"`. Java
`CherryPickOperation` reads the property with `PropertyUtil.propertyAsBoolean(…, false)` =
`Boolean.parseBoolean` — case-insensitive `equalsIgnoreCase("true")`, no trimming. So `"TRUE"` /
`"True"` are replace in Java but were non-replace here; `" true"` (leading space) and `"yes"` are
non-replace on both sides.

### Cells

New module `crates/iceberg/src/transaction/cherry_pick/tests/cherry_pick_case_insensitive.rs`,
wired as `mod cherry_pick_case_insensitive;` inside `cherry_pick.rs`'s `mod tests` (the
`replace_partitions/tests/` pattern). `cherry_pick.rs` was AT its 2106 legacy ceiling, so the `mod`
line is paid for by collapsing `.map(|value| value == "true").unwrap_or(false)` to
`.is_some_and(|value| value == "true")` in the RED commit — a semantics-preserving refactor that is
also the shape the fixed code takes; the file sits at exactly 2106 at every commit and the ceiling
is untouched.

| Cell | Staged marker | Test | Expected |
|---|---|---|---|
| (a) | `"TRUE"` (caller-set, survives round-1 layering) | `test_cherrypick_replace_partitions_marker_is_case_insensitive` | replays exactly like the `"true"` cell: new Overwrite snapshot, `a2` live, `a` dropped, `source-snapshot-id` set |
| (b) | `" true"` (leading space) | `test_cherrypick_replace_partitions_marker_leading_space_is_not_replace` | non-replace path: `Cannot cherry-pick snapshot %s: not append, dynamic overwrite, or fast-forward` (green pre-fix too — boundary control) |

The staging helper mirrors `stage_replace_partitions_for_replay` but commits the staged snapshot
through the replace-partitions action with
`set_snapshot_properties({"replace-partitions": marker_value})` — which is exactly how a caller's
value reaches a real summary post-round-1.

### Red output (pre-fix, commit `7de9ab5a`)

```
test ...::cherry_pick_case_insensitive::test_cherrypick_replace_partitions_marker_is_case_insensitive ... FAILED
  called `Result::unwrap()` on an `Err` value: DataInvalid =>
  Cannot cherry-pick snapshot <id>: not append, dynamic overwrite, or fast-forward
test ...::cherry_pick_case_insensitive::test_cherrypick_replace_partitions_marker_leading_space_is_not_replace ... ok
test result: FAILED. 26 passed; 1 failed; 3802 filtered out
```

### Fix and green (commit `ce36f58a`)

`is_replace_partitions` now reads `.is_some_and(|value| value.eq_ignore_ascii_case("true"))` — the
fork's `propertyAsBoolean` idiom (`crates/catalog/rest/src/catalog.rs:393-398`).

```
cargo test -p iceberg --lib cherry_pick
test result: ok. 27 passed; 0 failed; 3802 filtered out
```

### Mutation proof

Reverted only `eq_ignore_ascii_case("true")` → `== "true"` (tests kept):

```
test ...::test_cherrypick_replace_partitions_marker_is_case_insensitive ... FAILED
  Cannot cherry-pick snapshot <id>: not append, dynamic overwrite, or fast-forward
test ...::test_cherrypick_replace_partitions_marker_leading_space_is_not_replace ... ok
test result: FAILED. 1 passed; 1 failed; 3827 filtered out
```

Identical red signature to the pre-fix run. Restored; both cells green again. Revert never
committed (`git status` clean after restore).

### Round-2 audit delta

The §8 verdict is unchanged for every other site — the only behavioral read of the property is this
gate, now `parseBoolean`-faithful. Remaining semantic gap, if any, is nil: Java writes exactly
`"true"` itself, so case variants reach a summary only via caller `set` (Rust now) or
externally-authored metadata — both now read identically on the replay gate.
