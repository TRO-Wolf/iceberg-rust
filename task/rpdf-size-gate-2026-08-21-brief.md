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

# `RewritePositionDeleteFiles` — the missing size-based admission gate

**Source brief:** RePark engine side, MW-2 (Iceberg maintenance wave), 2026-08-21 —
`fork-brief-rpdf-size-gate.md`. Defect read at fork revision `0c5fd58d`.
**This file:** the source brief's substance plus the **owner's binding addendum** (rulings on the
three blocking questions, four design calls, and five corrections raised by the fork-side review)
plus the **verified-facts appendix** that the review established firsthand.

On conflict, the fork's repo contracts win — `CLAUDE.md` `<precedence>`, then this addendum, then
the source brief.

---

## 1. The defect

`crates/iceberg/src/maintenance/rewrite_position_delete_files.rs:222`, inside `execute`:

```rust
// Java's planner drops single-file groups (nothing to compact). A group must have at least
// TWO position-delete files for compaction to do real work.
if entries.len() < 2 {
    continue;
}
```

That is the action's entire size/count admission logic. Java's
`BinPackRewritePositionDeletePlanner extends SizeBasedFileRewritePlanner` admits a group only when it
passes a three-clause size gate whose file-count floor is **five**.

The fork therefore compacts groups Java declines. This produced **no wrong answers** — the row set was
identical before and after on both engines at every count measured. It is a parity fix, not a bug fix.

### Java contract, bytecode-verified

`filterFileGroups` is a plain three-way disjunction, no fourth clause:

```
enoughInputFiles(group) || enoughContent(group) || tooMuchContent(group)
```

`defaultTargetFileSize` is
`PropertyUtil.propertyAsLong(table.properties(), "write.delete.target-file-size-bytes", 67108864L)` —
the **delete-specific** 64 MiB property, not the 512 MiB `write.target-file-size-bytes` the data-file
planner uses.

Inherited constants from `SizeBasedFileRewritePlanner`:

| Constant | Value |
|---|---|
| `MIN_INPUT_FILES_DEFAULT` | `5` |
| `MIN_FILE_SIZE_DEFAULT_RATIO` | `0.75` |
| `MAX_FILE_SIZE_DEFAULT_RATIO` | `1.8` |
| `MAX_FILE_GROUP_SIZE_BYTES_DEFAULT` | `107374182400` (100 GiB) |
| `REWRITE_ALL_DEFAULT` | `false` |

### Measured evidence

Live Spark 4.0.1 + Iceberg 1.10.0, hadoop catalog, one unpartitioned table at
`write.delete.granularity = 'partition'`. Eight data files; the delete-file count varied by issuing
that many single-row merge-on-read `DELETE`s. Each delete file is roughly 1.4 KB, far below any size
clause, so this isolates the **count** clause.

| live delete files | Java/Spark result | fork result |
|---:|---|---|
| 1 | `0, 0, 0, 0` | `0, 0, 0, 0` |
| 2 | `0, 0, 0, 0` | `2, 1, ...` |
| 4 | `0, 0, 0, 0` | `4, 1, ...` |
| 8 | `8, 1, 11429, 1454` | `8, 1, 12104, 2110` |

Columns are `rewritten_delete_files_count, added_delete_files_count, rewritten_bytes_count,
added_bytes_count`. Counts converge exactly at 8, where Java's floor is satisfied. The byte columns
differ even when counts agree — the two engines' Parquet encoders produce different file sizes. Do not
chase it, and do not pin a byte value against Java's.

---

## 2. Owner's binding addendum (2026-08-21)

### Q1 — RULED: lift the writer freeze, narrowly, for this action only

The source brief's §8 forbade touching the writer. **That fence is lifted for this action.** Mirror
`rewrite_data_files.rs:601-603`:

- resolve `write.delete.target-file-size-bytes` (default `67108864`) and pass it to the rolling writer;
- feed pairs in chunks so `should_roll()` is evaluated more than once;
- return `Vec<DataFile>` instead of `.next()`;
- replace the hard-coded `added_delete_files_count += 1` with the real count;
- fix the `.next()` discard at `:523` in the same change — a hard-error is not needed once the writer
  splits.

**Acceptance 3 becomes fixed-point form:** a lone oversized file is admitted, the output splits at the
delete target, and a second run is a no-op.

This is recorded explicitly **as a scope change to the source brief's §8**.

### Q2 — RULED: `.min_input_files(2)` on the interop test

Set `.min_input_files(2)` on `crates/iceberg/tests/interop_rewrite_pos_deletes.rs` — its subject is
read identity, not admission. **Do not grow the fixture**; that re-opens the Java oracle recording.

**New acceptance item 6:** the interop script must be run green (the
`ICEBERG_INTEROP_REWRITE_POS_DELETES_GEN_DIR` path) before this brief is called done. A green
`make check` alone is insufficient evidence.

### Q3 — RULED: grouping-key unification is OUT of scope

The `coercePartition`-keyed grouping (spec out of the key) gets its own follow-up brief, consistent
with the standing conductor-16 Q10 residue ruling. **But §9 is amended:** this PR must

1. update GAP_MATRIX row R136 to name the grouping gap and drop both the "1:1 port" claim and the
   "bin-packs" claim; and
2. note in the residue that at a floor of five the divergence is **outcome-flipping** (the fork
   declines what Java admits), so "only ever compacts more than Java" no longer holds.

Leave the row's status honest, not standing.

### Design calls — all four ADOPTED as recommended

1. Add `max_file_group_size_bytes` **and the bin-packing step**; reuse `pack_bins`; `execute` iterates
   **bins**, not partitions.
2. **Migrate tests by name, not by run outcome.** Every sub-floor test must assert post-`execute`
   **shape** (file counts / snapshot production), never read identity alone.
   `test_seq_stamp_does_not_resurrect_or_over_apply` and `test_v3_deletion_vectors_are_not_compacted`
   are the two named green-vacuous cases to rework.
3. **Drop `rewrite_all`** from the struct — deferred loudly, matching the template's
   `rewrite_data_files.rs:141`. Keep the inverted-emulation warning in the brief text: `rewriteAll`
   **bypasses** both filters and keeps the packing; the naive `min=0, max=MAX, min_input_files=1`
   emulation admits *nothing* and is the exact inverse of the intent.
4. Absorbed into Q1 — the `.next()` discard is fixed with the writer lift.

### Corrections 1-5 — all ACCEPTED, fold in before coding

- **DV justification.** Java's *planner* is format-blind (zero `FileFormat` / `PUFFIN` references in
  `BinPackRewritePositionDeletePlanner`'s constant pool); DV avoidance is an all-or-nothing early exit
  in the *Spark action*. The fork's skip is `file_format() != Parquet`, which also drops V2 ORC/Avro
  position deletes. **Keep the behaviour; correct the comment to say what it actually does.**
- **"Exactly three methods"** becomes: three *abstract hooks* supplied, plus `plan()`,
  `validOptions()` and `init(Map)` implemented or overridden.
- **R136:** both over-claims come out (see Q3).
- **All four live doc homes are part of the code change:** module rustdoc `:29-31` and `:88-92`,
  `maintenance/mod.rs:72-73`, plus R136 — then `make check-matrix-anchors`.
- **New consts:** `u64`, fork-authored provenance style ("bytecode-verified vs
  `iceberg-core-1.10.0.jar`"), **not** the upstream-inherited 512 MiB exemplar.

### Acceptance criteria — the closed set

Items 1, 2, 4 and 5 are quoted verbatim from the source brief's §7; item 3 is superseded by the Q1
ruling above; item 6 is added by the Q2 ruling. This list is closed — nothing else is acceptance.

1. "A group of 4 small position-delete files is **not** rewritten under defaults; a group of 8 is.
   Both directions pinned. This is the whole point — a test that only proves the second half does
   not detect a gate that never declines."
2. "`.min_input_files(2)` restores the old behaviour, proving the floor is configuration rather than
   a hard-coded constant."
3. **SUPERSEDED by Q1 (fixed-point form).** A lone oversized file is admitted, the output splits at
   the ruled roll bound, and a second `execute` on the same table is a no-op. The source brief's
   original item 3 — "Size clauses pinned independently of the count clause: a group of 2 files whose
   combined bytes exceed the target **is** rewritten (`enoughContent`), and a single file above
   `max_file_size_bytes` is a candidate (`tooMuchContent`)" — survives as the *admission* half; the
   fixed-point half is what Q1 adds.
4. "Read identity holds across every new case — the live row set is unchanged by compaction. The
   existing helpers `scan_y_values`, `live_delete_files` and `count_pos` in the test module already
   do this; reuse them." **Qualified by design call 2:** read identity is never the *only*
   post-`execute` assertion, because a declined group satisfies it by doing nothing.
5. "The sequence-number stamping behaviour is untouched. If a mutation test exists for it, it must
   still red."
6. **ADDED by Q2.** The interop script must be run green (the
   `ICEBERG_INTEROP_REWRITE_POS_DELETES_GEN_DIR` path) before this brief is called done. A green
   `make check` alone is insufficient evidence.

### Standing fences (unchanged)

One PR. Owner squash-merges. `%ae` byte-exact. No `Cargo.lock` churn. **Goldens are Java-parity
truth — a golden that moves is a finding unless this addendum names it.**

---

## 2A. Owner's addendum 2 (2026-08-21) — rulings on the scope audit's OPEN clauses

The first scope audit returned `REWRITE_DEMAND` at 31/42 with 11 OPEN clauses. Ten were owner
decisions (C-039 was closed by supplying the acceptance list in §2 above). All ten are ruled here.
**These rulings bind and, where noted, override addendum 1.**

### R-1 (C-009) — the roll bound is Java's `write_max_file_size`, NOT the resolved target

**This explicitly OVERRIDES addendum 1's Q1 wording "pass the resolved target to the rolling
writer."** The rolling writer's bound is Java's

```
writeMaxFileSize() = target + (max_file_size - target) * 0.5
```

= **93,952,409** at delete defaults (bytecode-verified this session:
`SizeBasedFileRewritePlanner.writeMaxFileSize` — `l2d / lsub / l2d / ldc2_w 0.5d / dmul / dadd`;
`BinPackRewritePositionDeletePlanner.newRewriteGroup` passes it as `maxOutputFileSize`;
`SparkRewritePositionDeleteRunner.doRewrite` sets the write option from it).

**Reason, recorded:** `write_max < max_file_size` whenever `max > target`, so a run-1 output is
structurally never re-admitted by `too_much_content`. That makes acceptance 3 **proven**, not argued.
Rolling at the raw target would make convergence rest entirely on the chunk-overshoot bound.

**Consequence to record, not to hide:** until the template follows, `RewriteDataFiles` rolls at the
target (its named deviation, `rewrite_data_files.rs:145-149`) while this action rolls at write-max.
**Log that sibling divergence as a residue against the data-files row R135 / its follow-up — never
silently.**

### R-2 (C-025) — port Java's `inputSplitSize` exactly

`inputSplitSize(inputSize) = inputSize / expectedOutputFiles(inputSize) + 5120` (`SPLIT_OVERHEAD`),
returning `target` when the quotient is below it, else `min(split, writeMaxFileSize())`, with
`expectedOutputFiles`'s `LongMath` ceiling/floor pair. **No invented chunk rule.**

### R-3 (C-026) — the fixed point is a full conjunction

Acceptance 3's no-op is `rewritten_delete_files_count == 0 && added_delete_files_count == 0 &&
current_snapshot_id() unchanged`. It is **conditional on R-1**, and the acceptance text must state
that dependency.

### R-4 (C-004) — `filterFiles` IS in scope

Java's order — `filterFiles` then pack then `filterFileGroups` — lands whole. Omitting the candidate
filter would admit a partition of well-sized files via `enough_input_files`: a fresh divergence in a
divergence-removal PR.

### R-5 (C-027) — genericise `pack_bins` with a weight closure

Option (a). The template's call site takes the one behaviour-neutral edit. No trait, no local
reimplementation.

### R-6 (C-028) — defer the per-group `Result` list as a NAMED residue on R136

RePark consumes the four aggregate counts. The `FileGroupInfo` analogue plus the name collision with
the exported `maintenance::FileGroupRewriteResult` plus the struct break is not worth buying here.

### R-7 (C-032) — home, type, no struct field

`crates/iceberg/src/spec/table_properties.rs`; `u64` with the fork-authored provenance style
("bytecode-verified vs `iceberg-core-1.10.0.jar`"); **no new `TableProperties` struct field** — that
struct is all-`pub` and not `#[non_exhaustive]`, and that break is not taken here.

### R-8 (C-031 + N-2) — R136: correct the sentence, do not drop it

Addendum 1's Q3 drop-ruling was written against the pre-PR tree and **this PR moots it** — the
"Bin-packs live PARQUET position-deletes..." sentence becomes TRUE. The intent (no false claims) is
served by **accuracy, not deletion**: correct the sentence to its now-true form.

Cell value: **`✅` with the grouping-key residue named in the glyph parenthetical** — R147's precedent
(`✅ (2026-06-16, with 2 named fail-safe divergences)`).

Roadmap.md lines 229 / 367 / 429 are **OUT of scope**: the matrix owns status and the de-triplication
rule holds.

### R-9 (C-034) — the PR body names all three behaviour flips

(1) two-file groups stop compacting; (2) an unbindable `filter` errors earlier; (3) the per-bin result
count shape. And per R-6 / R-7 it states truthfully that there are **no breaking struct changes**.

### R-10 (C-038 + N-6) — one doc drive-by in, two out

- `crates/iceberg/src/maintenance/actions_provider.rs:317-321` — **IN**: this change falsifies it, so
  fixing it is part of the code change.
- `dev/java-interop/map.md:62` and `crates/iceberg/tests/map.md:63` — **OUT**: both stay accurate under
  `.min_input_files(2)`. Do **not** fix the stale "GAP_MATRIX row 134" citation in this PR; note it as
  a one-line residue if anywhere.

---

## 2B. Owner's addendum 3 (2026-08-21) — the chunk mapping, and what actually makes the fixed point hold

The re-audit reached 43/44 with `C-025` (the writer feed) the sole OPEN clause, and filed one S1
showing that **acceptance 3 is falsified at the DEFAULT config** under addendum 2's R-2 as written.
Three facts, each verified at source:

1. **Java's `inputSplitSize` is a READ split, not a write chunk.**
   `SparkRewritePositionDeleteRunner` consumes it at offset 97 as the scan option `"split-size"`. The
   WRITE bound is a separate option at offset 199 — `"target-delete-file-size-bytes"` fed from
   `group.maxOutputFileSize()` = `writeMaxFileSize()`. Java never chunks a writer feed; it streams
   records and lets the writer roll.
2. **The fork's `should_roll()` is a PRE-check.** `rolling_writer.rs:160` is
   `current_written_size() > target_file_size`, evaluated BEFORE the incoming batch is written — so a
   file closes only once it has ALREADY exceeded the bound. Overshoot is up to one chunk.
3. **`current_written_size()` excludes the Parquet footer** — `bytes_written() + in_progress_size()`
   (`parquet_writer.rs:800-808`). This action writes FULL `file_path` bounds, so on small files the
   footer is a large fraction of the total.

Together: feeding the writer at `inputSplitSize` granularity gives chunks of **>= 67,108,864** bytes
(the function floors at target). With `write_max = 93,952,409`, no roll fires after chunk 1, chunk 2
lands in the SAME file, and output 1 is **~134,217,728 > max_file_size = 120,795,955** — which
`too_much_content` re-admits FOREVER (no `size > 1` guard).

### R-11 (D-1) — DROP the `inputSplitSize` / `expectedOutputFiles` port

**This amends addendum 2's R-2.** Neither function has a consumer in this action, which reads pairs
directly rather than through a split-size-driven scan. Porting them as dead code would violate
`skills/Opus.md` §9 ("delete dead code; don't comment it out"). Record the decision as a **named
non-port** carrying the bytecode reason — Java applies `inputSplitSize` on the READ side, and the
fork's action has no read side for it to parameterise.

`writeMaxFileSize` (R-1) is unaffected and stands.

### R-12 (D-2) — the write-feed chunk is bounded by the candidate-filter headroom

The fork's write-feed chunking is a **fork-specific mechanism with no Java analogue**. Feed the sorted
pairs in chunks of a fixed pair count, and make `resolve_config` assert the invariant

```
chunk_serialized_bytes <= max_file_size - write_max_file_size
```

= **26,843,546** bytes at delete defaults. A small fixed chunk clears that by three orders of
magnitude; the invariant is what makes the clearance checkable rather than assumed.

### The corrected convergence argument (restates C-026)

The fixed point does **not** come from the roll bound directly. It comes from the **candidate
filter**: a run-1 output that lands inside `[min, max]` fails `outsideDesiredFileSizeRange`, is
therefore never a candidate, and no bin forms. That holds exactly when overshoot <= `max - write_max`,
which is what R-12 asserts.

The sub-min **tail** output is safe independently: a lone sub-min candidate packs into a bin of one
and is declined by both `size > 1` conjuncts, and it is too small for `too_much_content`.

**Consequence for the fixture recipe:** the audit's `min = S-3, target = S-2, max = S-1` three-byte
band is unachievable and must be replaced with a wide band (`min = 0.75T`, `target = T`, `max = 1.8T`
with `S > max`), plus a mandatory pre-assertion that every run-1 output's `file_size_in_bytes` lies
inside `[resolved min, resolved max]` — so a size drift fails loudly instead of reddening the
fixed-point test for an unrelated reason.

### R-13 — the `rewrite_files.rs` sequence-direction inversion is a RESIDUE, not this unit's work

`crates/iceberg/src/transaction/rewrite_files.rs:122-123`, `:283-284` and `:304-305` state the
sequence-number direction backwards: they say a HIGHER inherited seq makes a delete stop applying and
resurrect rows, where the fork's own rule (`delete_file_index.rs:205`, `delete_seq >= data_seq`
applies) makes it LOWER. `:304-305` is the rustdoc of `add_delete_file_with_sequence_number` — the
call this action makes and fans out from one to N.

They were already wrong before this PR, so by R-10's precedent they are **out of scope here**. File as
a named residue and a backlog item; do not open a unit for it now.

---

## 2C. Addendum 4 (2026-08-21) — rulings on the closing Critic's seven questions

The closing Critic ruled `CHANGES_REQUIRED`: A-003's third limb was never applied, and five new S2
findings landed. It confirmed 45 PROVEN / 0 OPEN / 0 REJECTED and bytecode-confirmed all four
constants (67108864 / 50331648 / 120795955 / 93952409, headroom 26843546), and it confirmed the
round-2 S1 is genuinely dead. Its seven questions are ruled below under the owner's standing
"proceed with recommendations" instruction.

### R-14 (Q1) — the three fork-authored literals stand

`CHUNK_PAIRS = 256`, `CHUNK_MAX_SERIALIZED_BYTES = 16384`, and the half-headroom footer reservation
are confirmed as named. They are fork-authored (no Java analogue — R-11), so each carries a comment
saying so and why the value was chosen, not a false parity citation.

### R-15 (Q2) — precondition (7) stays a `DataInvalid` on the five builders

Keep it as a loud rejection rather than routing it to the residue list. Reasons: `skills/Opus.md`
Core Principles ("No Assumptions / Fail Loudly", "make illegal states unrepresentable"); it avoids
making RES-9 a live asymmetry; and it removes the obligation for a third C-007 pin at target ~1.23e19.
A config Java cannot express is one the fork should refuse, not silently accept.

### R-16 (Q3) — assert at both altitudes, and stop over-claiming the config assert

N-C5 is correct: `resolve_config` cannot see `chunk_serialized_bytes`. So:

- **Config time:** assert `chunk_budget <= max_file_size - write_max_file_size`. True by construction,
  but it reds if a future editor changes a constant, so it is worth keeping — as *intent
  documentation with a tripwire*, which is what it actually is.
- **Runtime clearance:** proven by C-025's **measured-output pin**, not by an assert.

**C-025's proof text must be corrected** — it currently claims the config assert is "what makes the
clearance checkable rather than assumed." It is not. The pin is. Reword it.

### R-17 (Q4) — ratify the corrected windows

- **Recipe 3:** ceiling drops from 3.2T to **2.8T** (above it the second output itself reaches
  `write_max` and rolls, producing a sub-min third output that violates C-026's own pre-assertion).
  The floor is **derived from `chunk_budget`**, not asserted at 250,000, and the recipe must **state
  the drift tolerance it actually delivers** rather than claim 10% and deliver 3.86%.
- **Recipe 9:** raise the floor to **B > 245,700** so C-009's named mutant cannot survive a plausible
  re-encode drift.
- Both tautological `S`-window asserts (`T := S*10/24` and `T := B*10/12` make them unfalsifiable) are
  replaced with asserts on the **measured outputs**.

### R-18 (Q5) — scope C-045's grep; the template rustdoc is correct and stays

N-C4 is correct: the grep has **one hit today**, `rewrite_data_files.rs:146`, and that sentence is
true and load-bearing (it states the sibling deviation RES-1/C-022 preserve). Scope the grep to the
action file and its test sibling, and **exempt `rewrite_data_files.rs:146` by name**. Do not edit it.

### R-19 (Q6) — `rewrite_position_delete_files.rs:46` is IN scope

F-A2-06 is right and its resolution follows R-10's own test. `:44-46` says the action "writes them
into FEWER position-delete files" — falsified **by this change**, since a lone oversized file
(admitted by `too_much_content`, which has no `size > 1` guard) now becomes two or more outputs.
Falsified-by-this-change is exactly the in-scope trigger. Correct the sentence; drop the
byte-unchanged pin that would have locked it in.

### R-20 (Q7) — release the worktree, but DO NOT cut the branch yet

`iceberg-rust-ws` is released for this unit **after R4 content verification** that #191's content is
on `main` (PR #191 merged 2026-08-10; verify by content, never by title).

**Blocking precondition, unchanged:** `git fetch` fails on a missing SSH askpass, and the primary
checkout's detached `HEAD` (`6258bb01`) has **diverged** from the cached `origin/main` ref
(`9f85a086`) — each carries commits the other lacks. **No branch is cut and no edit is made until a
real fetch resolves the true tip.** This is a hard stop, not a caution.

### Artifact-level obligations carried into this round

- **A-003's third limb:** downgrade `risk_heatmap` row 2 from `MITIGATED` to `PARTIALLY MITIGATED`
  with the residue named. It was dispositioned REMEDIATED and never applied.
- **N-C1:** regenerate all six stale non-clause fields — `refined_charter`, `pr_carving`,
  `rubric_result`, `risk_heatmap`, `killed_assumptions`, `logic_gaps_destroyed`. The charter currently
  **orders the Actor to build the very port C-045 pins against**, and the carving assigns removed
  C-043 while omitting C-045 and C-046 entirely.
- **N-C6..N-C10 (S3):** fold the five provenance/count corrections.

---

## 2D. Addendum 5 (2026-08-21) — closing N-D1, the last S2

The final closing Critic confirmed 45 PROVEN / 0 OPEN / 0 REJECTED with an empty
`clarifying_questions` list and ruled every prior finding genuinely closed, but filed one new S2 and
correctly refused the gate on it: **a gate that passes with an open S2 is a false green.**

### R-21 (N-D1) — C-045's pin greps the RUST identifiers, not the Java names

**The defect:** C-045 requires the action file to *document* the non-port — a paragraph that must name
Java's `inputSplitSize` / `expectedOutputFiles` — while its pin requires a zero-hit grep for those same
strings in that same file. The clause fires on the documentation it mandates.

**The fix:** the pin greps for the **Rust identifiers a real port would introduce** —
`input_split_size` and `expected_output_files` — across `crates/`. This catches a **transliterated** port — a Rust
function keeping the Java name in snake_case — and not a prose citation of the Java name.

**Corrected (S3-01):** an earlier draft of this ruling claimed the fork's convention is to snake_case
the *Java name*, citing `is_candidate`, `group_qualifies` and `parse_target_file_size`. Those three
**refute** it — they port `filterFiles`, `filterFileGroups` and `defaultTargetFileSize`, so the observed
convention is descriptive *renaming*. The pin therefore does not catch a port renamed into the fork's
existing vocabulary (`with_split_size` already exists). That residual is named in C-045 and bounded by
`dead_code` under clippy `-D warnings` plus C-016's whole-file diff read. The pattern is deliberately
NOT broadened to `split_size`, which has 40 pre-existing hits.

**Verified at baseline this session:** `grep -rn 'input_split_size\|expected_output_files' crates/`
returns **0**; `grep -rn 'inputSplitSize\|expectedOutputFiles' crates/` returns exactly **one** hit,
`rewrite_data_files.rs:146`, which is the correct load-bearing sibling-deviation rustdoc.

**This supersedes R-18's by-name exemption**, which is no longer needed: the camelCase prose at
`rewrite_data_files.rs:146` passes the corrected grep naturally, and the file still must not be
edited. Keep the exemption note as a comment on the pin so a future editor does not "helpfully"
re-broaden the pattern.

### Also folded (S3, below the floor, non-blocking)

- **N-D2** — `pr_carving`'s prose disagrees with its own canonical map in three places (G1 header says
  "(6)" against a seven-id map; G2 double-enumerated) and carries an unedited mid-field
  self-correction. Reconcile the prose to the canonical map and strike the stray edit.
- **N-D3** — `C-030` elements 1 and its twin prescribe only PART of what this change falsifies: both
  carry the same "FEWER position-delete files" promise that **R-19** just ruled
  falsified-by-this-change. Extend both corrections to state both directions, exactly as `:44-46`
  now does.

---

## 3. Verified-facts appendix

Established firsthand by the fork-side review (six verifiers plus an adversarial refutation round; 19
of 32 candidate discrepancies survived). Everything below was read at the cited location, not inferred.

### Confirmed as the brief states

- `entries.len() < 2` at `:222` is the entire size/count gate. The struct is `{ table, filter }` with
  only `new` / `filter` / `execute`; no size configuration surface exists.
- `spec/table_properties.rs` has **no** delete-specific counterpart. Five independent greps for the
  property key, the constant name, and the literal `67108864` return zero hits.
- The template's `group_qualifies` (`rewrite_data_files.rs:753`) and `is_candidate` (`:741`) match the
  brief's Rust paraphrase semantically character-for-character, including both `size > 1 &&` guards,
  `>=` on `min_input_files`, strict `>` on both size comparisons, and the deliberate absence of a
  `size > 1` guard on `too_much_content`.
- The action file is byte-identical between the brief's revision `0c5fd58d` and the current tip.

### Corrections the addendum folds in

- **No splitting path.** `write_compacted_file` builds one `RecordBatch` from every pair (`:508-513`),
  issues exactly one `writer.write(batch)` (`:521`), and returns `files.into_iter().next()` (`:523`).
  The `RollingFileWriter` at `:492` is `new_with_default_file_size` — 512 MiB, the **data** default —
  and `should_roll()` is evaluated only at the top of each `write`, so with one call it can never fire.
  With `max = 1.8x target`, `too_much_content` is strictly implied by `enough_content` for any group of
  size > 1, so the lone-oversized-file case is that clause's only unique contribution.
- **Blast radius is two test files, not one.** `interop_rewrite_pos_deletes.rs` builds four pos-delete
  files, two per partition (`:258-275`), and calls `::new(compacted.clone()).execute(...)` unconfigured
  (`:372-374`), breaking four asserts (`:377` rewritten == 4, `:381` added == 2, `:405` post_pos == 2,
  `:407` post_pos < pre_pos) plus the Java oracle's own `prePos > postPos && postPos > 0` leg. It is
  env-gated (`:97-98`) and returns green when the variable is unset, so redness cannot announce it.
- **Grouping key diverges independently.** Java keys on
  `PartitionUtil.coercePartition(unifiedType, spec, partition)` — the spec is a projection argument and
  never enters the key. The fork keys on `(spec_id, partition)` (`:127` `type GroupKey = (i32, Struct)`,
  `:274`). At a floor of two this is nearly invisible; at five it flips outcomes. `GroupKey.0` also
  doubles as the output-spec selector in `write_compacted_file`.
- **Packing is missing from the suggested shape.** Java packs with `BinPacking$ListPacker` **between**
  `filterFiles` and `filterFileGroups`, and the pos-delete planner inherits it unchanged;
  `max-file-group-size-bytes` is user-settable on this action in Java. The fork's template already has
  the whole thing: field `rewrite_data_files.rs:222`, default `:173`, builder `:285`,
  `pack_bins(candidates, config.max_file_group_size_bytes)` at `:721`.
- **Two tests go green-vacuous, not red.** `test_seq_stamp_does_not_resurrect_or_over_apply` (`:496`)
  has exactly one post-`execute` assertion — a read-identity check — so a declined group passes it while
  exercising zero seq-stamping code. `test_v3_deletion_vectors_are_not_compacted` (`:716`) keeps
  passing, but its own doc names a mutation ("drop the Parquet skip and the two DVs form a 2-file
  group") that a floor of five neutralises.
- **`rewrite_all` is not in the template.** `RewriteDataFiles` is nine fields with no `rewrite_all`, and
  `:141` lists it explicitly under "# Deferred (loudly)".
- **R136 over-claims twice.** Alongside "1:1 port" it says the action "Bin-packs live PARQUET
  position-deletes per `(spec, partition)`" — false independently of the gate: grouping is a single
  `HashMap` insert (`:274`), each group yields exactly one output file, and
  `grep -c 'target_file_size\|min_input\|pack_bins'` over the action returns 0.
- **Four live doc homes**, two of which the fix itself falsifies: the module rustdoc `# No-op` section
  (`:88-92`), `:29-31`, `maintenance/mod.rs:72-73`, and R136.
- **The named property exemplar is wrong.** The 512 MiB constant is upstream-inherited and is the only
  default in that block with no provenance citation; the fork-authored constants above it all carry
  "bytecode-verified vs `iceberg-core-1.10.0.jar`".

---

## 4. Non-goals (unchanged from the source brief)

- **Deletion vectors stay out of scope.** V3 Puffin DVs are file-scoped and are never bin-packed. The
  existing skip stays; only its justification and comment are corrected.
- **Do not change the commit recipe.** The seq-stamping rule (group MAX rewritten data-seq) is a
  silent-corruption staller and is currently correct.
- **Do not touch delete-file granularity.** Java honours `write.delete.granularity` on output and the
  fork writes one file per group. Real, separate, deliberately not bundled.
- **Grouping-key unification is out** (Q3) — its own follow-up brief.
