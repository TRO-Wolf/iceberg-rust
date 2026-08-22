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

# `[repark] maintenance: RewritePositionDeleteFiles size-based admission gate (BREAKING default)`

The PR body for the `RewritePositionDeleteFiles` size-gate unit. Written by build group G6
(C-034 / R-9 / D4 / G2-1). Companion documents: the charter and its addenda in
[`rpdf-size-gate-2026-08-21-brief.md`](rpdf-size-gate-2026-08-21-brief.md), the 45-clause proof
ledger in [`rpdf-size-gate-2026-08-21-ledger.md`](rpdf-size-gate-2026-08-21-ledger.md).

## What this changes

`rewrite_position_delete_files.rs` admitted any `(spec, partition)` group of two or more live
position-delete files — a single `entries.len() < 2` guard was the entire size/count gate. Java's
`BinPackRewritePositionDeletePlanner` admits a group only through
`enoughInputFiles || enoughContent || tooMuchContent`, whose file-count floor is **five**, after a
candidate filter (`filterFiles` / `outsideDesiredFileSizeRange`) and a bin-pack at
`max-file-group-size-bytes`. This PR ports that planner, the five options that drive it, the
`writeMaxFileSize` roll bound, and the per-bin commit shape.

Reported by the RePark engine side (MW-2) with live Spark 4.0.1 / Iceberg 1.10.0 measurements and
independently re-verified against `9f85a086`. **No wrong answers were being produced** — the fork
compacted *more* than Java — so this is a parity fix, not a bug fix.

## BREAKING BEHAVIOUR — four flips, and NO breaking struct changes

**There are no breaking struct changes.** The public surface gains exactly **seven purely additive
items** and nothing else: the five `RewritePositionDeleteFiles` builder methods
(`target_file_size_bytes`, `min_file_size_bytes`, `max_file_size_bytes`, `min_input_files`,
`max_file_group_size_bytes`) and the two `TableProperties` associated consts
(`PROPERTY_WRITE_DELETE_TARGET_FILE_SIZE_BYTES` and its `_DEFAULT`). No new `pub` struct field, no
new `pub` type, no new trait item, no changed `pub` signature, no new error variant — so
struct-literal construction of `RewritePositionDeleteFilesResult` (four counts, R-6) and of
`TableProperties` (seven fields, R-7) is unaffected on both all-`pub`, non-`#[non_exhaustive]`
types. Re-derived mechanically: `git diff <base>..HEAD -- crates/ | grep -E '^\+\s*pub '` returns
exactly those seven lines, none inside `#[cfg(test)]`. `pack_bins` and four constants widen to
`pub(super)` only, which cannot escape the crate.

The BEHAVIOUR of an unchanged caller does flip, in four ways:

1. **The default admission floor moves 2 -> 5.**
   `Actions::get().rewrite_position_deletes(table).execute(..)` compacted any two-file group before
   this change; it now admits a bin only on `enough_input_files || enough_content ||
   too_much_content`, so a bin of fewer than five files whose total size is inside
   `[min_file_size_bytes, max_file_size_bytes]` is declined and the run is a no-op for it. Existing
   two-file callers must set `.min_input_files(2)` to keep the old behaviour — this is the intended
   breaking flip. Pinned by `test_admission_min_input_files_default_five_declines_four_admits_five`.

2. **An unbindable `filter` now errors EARLIER.** The predicate is bound to the table schema ONCE,
   right after the no-snapshot early return, so an unbindable filter fails on any table **with a
   current snapshot** rather than only when some group happened to hold two or more files. Pinned by
   `test_unbindable_filter_errors_even_when_no_group_is_admissible`.

3. **The output and snapshot shape is now PER BIN, and can be more than one file.** A partition
   yielding `B` admitted bins produces `B` `Replace` snapshots, not one; and a bin whose content
   exceeds Java's `writeMaxFileSize()` rolls into SEVERAL compacted outputs instead of one. Both
   directions are now reachable: many small files FUSE into fewer, while a lone file above
   `max_file_size_bytes` is admitted alone (`too_much_content` carries no `size > 1` guard) and
   SPLITS into two or more. Pinned by the split battery and the per-bin commit pins.

4. **A pre-existing table property can now make `execute` return `Err` where it previously returned
   counts.** The action did not read `write.delete.target-file-size-bytes` before; it now does. A
   table carrying that key with a value in `{unparsable, > i64::MAX, <= 1, == i64::MAX}` makes
   `execute` fail instead of compacting. Mechanically: `> i64::MAX` and non-numeric text are
   rejected at the parse (the parse is `i64`, whose accept/reject domain coincides with
   `Long.parseLong`'s); `<= 1` and `== i64::MAX` are rejected by the resolved-config preconditions —
   at `target == 1` the defaulted `max` is `d2l(1.8)` = 1 and strict `target < max` fires, and at
   `target == i64::MAX` the defaulted `max` clamps to `i64::MAX` and the same strict comparison
   fires. **This is parity-correct** — Java throws on every one of those inputs — but it IS a
   behaviour flip and is named here rather than discovered downstream.

   *(This four-element set is a correction to the ledger's own text, which recorded three: "from 2
   up every precondition passes" is false at exactly `target == i64::MAX`.)*

## Files changed

Nine authorised files (C-016's closed manifest), plus `maintenance/mod.rs`, which C-030's own
enumeration names as a doc home:

| File | What |
|---|---|
| `crates/iceberg/src/maintenance/rewrite_position_delete_files.rs` | the action: five builders, `resolve_config` with Java's four preconditions plus three fork ones, the candidate filter, the bin-pack, the three-clause gate, the `writeMaxFileSize` roll bound, the bounded chunk feed, the per-bin commit loop, and the doc pass |
| `crates/iceberg/src/maintenance/rewrite_position_delete_files_tests.rs` | the test battery |
| `crates/iceberg/src/maintenance/rewrite_data_files.rs` | `pack_bins` genericised over a weight closure (see below) |
| `crates/iceberg/src/maintenance/actions_provider.rs` | the factory rustdoc, falsified by the new builders |
| `crates/iceberg/src/spec/table_properties.rs` | `write.delete.target-file-size-bytes` + its 64 MiB default |
| `crates/iceberg/tests/interop_rewrite_pos_deletes.rs` | `.min_input_files(2)` — closes a false green (below) |
| `docs/parity/GAP_MATRIX.md` | rows R136 and R135 |
| `task/todo.md` | the unit's plan block and the two follow-ups it files |
| `task/lessons.md` | the unit's lessons |
| `crates/iceberg/src/maintenance/mod.rs` | the module-head action summary, falsified on all three of its counts |

`Roadmap.md` is deliberately untouched: the matrix owns status and the glyph does not move.

### `rewrite_data_files.rs` — FOUR edit classes, all behaviour-neutral

`pack_bins` is genericised over the item type and a `weight` closure so the sibling planner packs
through the SAME packer rather than reimplementing it (R-5 option (a); Java's
`BinPackRewritePositionDeletePlanner` inherits `SizeBasedFileRewritePlanner`'s packer unchanged, so
one home is the parity-faithful shape). The edits are:

1. **the signature** — `pack_bins<T>(items: Vec<T>, weight: impl Fn(&T) -> u64, target_weight: u64)`;
2. **visibility** — `pub(super)` on `pack_bins` and on the four shared planner constants, so the
   sibling imports one home instead of duplicating them; `pub(super)` cannot escape the crate;
3. **the rustdoc** — a paragraph recording the genericisation and that `weight` is evaluated exactly
   once per item, at the point the non-generic form read `task.file_size_in_bytes`;
4. **call-syntax adaptation of the `cfg(test)` sites, compelled by class (1), assertion-preserving.**
   `test_pack_bins_forward_first_fit`'s three call sites fail to COMPILE under the new signature;
   each gains `|task| task.file_size_in_bytes,` and nothing else. This class was RATIFIED as
   *compelled* by an independent Critic: no two-argument wrapper exists that does not contradict
   R-5's ruled signature, and C-027 simultaneously requires the template's tests green. The
   charter's "exactly THREE edit classes" (C-020) missed the `cfg(test)` call sites and is corrected
   here.

`rewrite_data_files.rs`'s deferral rustdoc naming Java's `inputSplitSize` is TRUE, load-bearing and
deliberately untouched.

## Java evidence

Every constant and control-flow claim in this PR was read firsthand from `iceberg-core-1.10.0.jar` /
`iceberg-api-1.10.0.jar` bytecode, by the building agent and re-verified independently by each
group's Critic. Highlights:

- `sizeThresholds` ratios `0.75d` / `1.8d` via `l2d; ldc2_w; dmul; d2l`; all four precondition
  messages verbatim, in the order (1) -> (2) -> (3) -> (4).
- `MIN_INPUT_FILES_DEFAULT = 5` (`iconst_5`); `MAX_FILE_GROUP_SIZE_BYTES_DEFAULT = 107374182400`;
  `write.delete.target-file-size-bytes` default `67108864L`.
- `writeMaxFileSize()`: the `lsub` happens BEFORE the `l2d`, so the fork subtracts in `u64` first;
  93952409 at the delete defaults.
- `filterFileGroups`: a plain three-way disjunction, no fourth clause. `enoughInputFiles` `>` then
  `>=`; `enoughContent` `size > 1` and STRICT `>`; `tooMuchContent` has NO size guard and a STRICT
  `>`. Grouping is UPSTREAM of the candidate filter; the user filter is applied at the scan, before
  `groupByPartition`.
- `RewritePositionDeletesCommitManager.commit` iterates `group.addedDeleteFiles()` calling
  `addFile(f, group.maxRewrittenDataSequenceNumber())` for EACH — so the per-bin fan-out to N
  stamped files is Java-EXACT, not inferred.
- `planFileGroups(Iterable)` branches `rewriteAll ? tasks : filterFiles` at offset 4 and
  `rewriteAll ? bins : filterFileGroups` at offset 47, with the `ListPacker` construction BETWEEN
  them running unconditionally — the basis for the inverted-emulation warning in the deferral block.

## Interop — and the false green it closed

`crates/iceberg/tests/interop_rewrite_pos_deletes.rs` was genuinely BROKEN by the gate (its
`rewritten == 4` assertion measured `left: 0, right: 4`) yet reported green, because the whole test
is env-gated on `ICEBERG_INTEROP_REWRITE_POS_DELETES_GEN_DIR` and returns `Ok` when it is unset. The
break was REPRODUCED first, then fixed with exactly `.min_input_files(2)`; the fixture is not grown,
so no Java oracle value moves and all ten enumerated golden values hold.

The suite was then RUN — twice by the building agent and once independently by the Critic —
`mvn -o` against a warm `~/.m2`, no network:

```
verify-interop-rewrite-pos-deletes: 0 failures
PASS sabotage(read-identity): a compacted table that resurrected a masked row fails closed via the read-identity-BROKEN path
PASS sabotage(truncate): a truncated metadata file fails closed via the load/parse-error path
==> [5/5] DONE — RewritePositionDeleteFiles interop passed.
```

## Status — `docs/parity/GAP_MATRIX.md`

Row **R136** keeps its `✅` (the capability is not narrowed — this PR makes the row's bin-packing
claim TRUE) and is edited ADDITIVELY: the original 2026-06-17 flip date and the interop evidence
phrase survive, the format clause WIDENS to name V2 ORC/Avro alongside the V3 Puffin DV, and the new
work appends. The cell's "1:1 port" claim is corrected rather than dropped, and the row now names
**4 divergences** (grouping key, format skip, commit granularity, saturating input-size sum) and
**2 residues** (the deferred per-group `Result` list, the named non-port of `inputSplitSize` /
`expectedOutputFiles` with its replacement chunk feed).

Row **R135** gains one clause recording the sibling divergence: `RewriteDataFiles` rolls output at
its RESOLVED TARGET under its own named deviation, where this action rolls at Java's
`writeMaxFileSize()` = `target + 0.5 * (max - target)`.

## Charter defects found and ratified during the build

The 45-clause ledger was treated as falsifiable. Eleven defects were found by the build, each filed
with its clause id, evidence and ruling; every one was independently ratified:

| # | Clause | Defect |
|---|---|---|
| 1 | C-020 | "exactly THREE edit classes" in `rewrite_data_files.rs` is UNSATISFIABLE — genericising `pack_bins` breaks the template's own `cfg(test)` call sites, which must compile. Fourth class added above. |
| 2 | C-036 | the nine-recipe fixture enumeration is INCOMPLETE; by its own rule ("a pin needing a tenth recipe is a finding") this is a finding. Extended to twelve. |
| 3 | C-036 | recipe 3's MEASURED `S = 4_739_051` against the recorded "~3.93 MB" — 20.5 percent over. The margin is sound (the window is scale-invariant in `S`) but the cost figure was wrong. |
| 4 | C-011 | disposition "RETAINED :399-406" is STALE — that block was rewritten into the `Vec<DataFile>` fan-out. The commit ARITY proposition survives; the statement does not. |
| 5 | C-011 | disposition "DELETED :225-226" MISLABELS the region: at the base ref those lines are `group_matches_filter`'s LEADING comment; the guard and its comment were :220-224. Outcome right, label wrong, would double-count. |
| 6 | C-011 | the added input class OVER-STATES one of its two `DataInvalid` sites — after the filter moved to collection, only `write_compacted_file`'s is a genuine mid-loop abort. |
| 7 | C-040 | the pinning mechanism is NOT literally constructible: a zero-row file cannot be written "directly with `ParquetWriterBuilder`", because `ParquetWriter::close` DELETES a zero-row output — the very fact the clause relies on elsewhere. A substitute was built and proven genuine by two mutations. |
| 8 | C-036 | a THIRTEENTH recipe is needed (default-floor admissible bin: five sub-min entries, one partition, shipped defaults, no knobs), distinct from recipes 7 and 8; three tests already stand on it. |
| 9 | C-036 | recipes 1 and 2 had NEVER been built by any group — the enumeration assumed constructions that did not exist at HEAD. |
| 10 | C-014 | element 9 is STALE and conflicts with C-021: dropping `enough_input_files`' `size > 1` conjunct reds EXACTLY ONE test in 3378, so element 9's destination is already occupied and obeying it duplicates a single-mutant pin. C-021 wins. |
| 11 | C-013 / C-024 | the GEN-block line citations are `+19` post-merge. They were EXACT at HEAD; THIS PR moved them, so "the ledger drifted" is the wrong framing — the same class as the matrix line-citation trap, and the reason anchors are preferred. |

Two more were filed by the doc pass itself and are recorded in the group report rather than here:
C-030's fifteen-element enumeration carries stale addresses and is short of its own stated domain.

## Named residues that stay open

Fourteen, each with exactly one home; the four that land on the matrix are on row R136 above, and
the sibling roll bound is on row R135. Two follow-ups are filed to `task/todo.md` and deliberately
NOT fixed here, because neither is falsified by this change: the sequence-direction inversion in
`transaction/rewrite_files.rs` (R-13) and the stale bare-number `GAP_MATRIX row 134` citation in
`crates/iceberg/tests/map.md` and `dev/java-interop/map.md` (R-10, C-038).

## Verification

```
typos . && cargo fmt --all -- --check && make check-agent-artifacts && make check-matrix-anchors \
  && cargo clippy --all-targets --all-features --workspace -- -D warnings \
  && cargo test -p iceberg --lib
```

Plus `cargo build -p iceberg --no-default-features` for the public-surface clause, and the interop
transcript above.
