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

# F-INTEROP-NIGHTLY-GREEN-1 — the weekly Java-interop run back to 65 of 65

**Date:** 2026-10-17. **Base:** fork `main` `409ff330e`. **Model:** Claude Opus 5.5
(`claude-opus-5-5`), high effort.

The Nightly Interop workflow (`.github/workflows/nightly_interop.yml`, Saturdays 06:43 UTC) failed
every week from 2026-08-29. Run 39011689293 (2026-10-17) passed 61 of 65 suites. Four suites failed:

| Suite | Red since | Cause | PR |
|---|---|---|---|
| `run-interop-validate-append-only.sh` | 08-29 | test infrastructure: writer builder | A |
| `run-interop-pr5a-catalog-commit-decode.sh` | 09-05 | test infrastructure: missing jar | A |
| `run-interop-remove-dangling.sh` | 09-26 | see PR B | B |
| `run-interop-v3-maintenance.sh` | 09-19 | see PR C | C |

The unit ships as three pull requests. Each branches from the same `main`, so each one reviews
alone. This ledger carries one section per pull request.

This ledger retires when the three pull requests land and the weekly run reports 65 of 65.

## PR B — `fix/remove-dangling-rust-dv-1`: the V3 `dv` DV is pruned at commit on both engines

### Step 0 — measurement

`ICEBERG_INTEROP_REMOVE_DANGLING_GEN_DIR=<dir> cargo test -p iceberg --test
interop_remove_dangling test_remove_dangling_gen` on `origin/main` reproduces the weekly failure:
the V2 world passes (`removed 3 danglers; read-identity holds`), then the V3 world panics at
`interop_remove_dangling.rs:965` with REMOVED `{}` against the declared `{"dv"}`.

The PRE table it wrote, `<dir>/rust_table_v3/metadata/final.metadata.json`, read through
`StaticTable` (next-row-id 12, current snapshot seq 3, operation `Replace`, the `dv → dv'` rewrite):

| manifest | content | entry status | file | seq | `referenced_data_file` |
|---|---|---|---|---|---|
| 0 | data | Added | `cat=dv/...` (dv') | 3 | — |
| 1 | data | Deleted | `cat=dv/...` (dv) | 1 | — |
| 1 | data | Existing | `cat=dk/...` | 1 | — |
| 2 | deletes | **Deleted** | `dv-dv-....puffin`, Puffin position delete, 1 record | 2 | the removed `cat=dv` file |
| 2 | deletes | Existing | `dv-dk-....puffin`, Puffin position delete, 1 record | 2 | the live `cat=dk` file |

The `dv` DV is a `Deleted` tombstone in the rewrite snapshot itself. The only live delete file is
the `dk` DV.

**H1 holds.** Since `43fcd2430` (F-DANGLING-DV-COMMIT-1, landed 2026-09-19 after that morning's
weekly run) every merging commit drops the DV of a data file it removes, as Java's
`MergingSnapshotProducer.removeDanglingDeletesFor` does. The Rust-built world now reaches Java's
end state, so nothing is left for the action. H2 (the DV still live and missed by
`RemoveDanglingDeleteFiles`) is refuted by the tombstone. The action itself is correct; the
interop expectation was stale. The unit pin
`test_dangling_deletion_vector_removed_after_referenced_data_rewritten_away` was already updated by
F-DANGLING-DV-COMMIT-1 to expect the drop at commit and zero removals by the action. The interop
test and the Java oracle were not.

### The change

- `interop_remove_dangling.rs`: the `Origin` enum and `effective_remove_scenarios` are deleted.
  Both engines give the same REMOVE set, so the split selected nothing. The `dv` scenario takes a
  new verdict, `Verdict::PrunedAtCommit`: the commit that removes its data file drops it, so it is
  neither removed by the action nor a survivor. `remove_scenarios` is the REMOVE-verdict filter.
- New assertion `assert_pruned_at_commit`, run by the GEN leg on the PRE Rust table: no live
  delete file sits in a pruned-at-commit partition. It pins the F-DANGLING-DV-COMMIT-1 behaviour
  through this interop too.
- The GEN leg's `!cleaned_result.removed_delete_files.is_empty()` check became an exact count
  against `remove_scenarios(world)`: 3 for V2, 0 for V3. The old check would fail the V3 world now.
- The now-false text is DELETED, not reworded: the module-doc `dv` REMOVE row and its footnote,
  the "`pr` and `dv` dangle" sentence, the `Origin` and `effective_remove_scenarios` docs, the
  `build_v3_world` doc and its commit-3 comment, and the "origin-effective" comment.
- `InteropOracle.java` `RemoveDanglingOracle`: `removeCatsRust` (V3 `{dv}`) and
  `removeCatsJava` (V3 `{}`) collapse into one `removeCats` (V3 `{}`), with the divergence comment
  block and the stale SEMANTICS-MATCH comment deleted. The Java D2 verify recomputes
  `findDanglingDeletes` over the Rust PRE table independently, so its declared set had to follow.
  The Java generate leg and Rust's D1 over the Java table are behaviourally unchanged.

### What the change gives up

The interop no longer drives the action's DV-removal branch end to end. Neither engine commits a
dangling DV through a merging commit any more, so the branch acts only on a table whose writer
left one (an older fork, or an engine without `removeDanglingDeletesFor`). Its remaining evidence
is the pure-predicate pin `test_dv_dangles_when_referenced_data_file_gone`. GAP_MATRIX row R137
names this.

### Evidence (2026-10-17)

- `scripts/run_interop_suites.sh --only run-interop-remove-dangling.sh`:
  `TOTAL: 1 passed, 0 failed, 1 run, 65 discovered (floor 65)`. Java generate `table_v3
  (REMOVE=[], KEEP=[dk])`; Rust GEN `rust_table_v3 ... removed 0 danglers; read-identity holds`;
  Java D2 `findDanglingDeletes = [] matches hand-declared REMOVE` and `surviving delete partitions
  = [dk]` on `table_v3`, `verify-interop-remove-dangling: 0 failures`; Rust D1 `table_v3: removed
  0 danglers`; all three sabotage legs fail closed on their pinned lines.
- `cargo test -p iceberg --test interop_remove_dangling` with no `ICEBERG_INTEROP_*` variable:
  `2 passed; 0 failed` (both legs are env-gated no-ops).
- Mutation: `is_dangling_dv` in `transaction/snapshot/manifest_filter.rs` forced to return
  `false` (the commit keeps the DV, the pre-F-DANGLING-DV-COMMIT-1 behaviour), GEN dir set:
  1 red out of 2 — `test_remove_dangling_gen` panics in `assert_pruned_at_commit` with live delete
  partitions `{"dv", "dk"}`. Restored; 2 of 2 green.
- File-size ratchet: `interop_remove_dangling.rs` fell from 1018 to 1000 lines, inside the default
  ceiling, so its legacy row in `scripts/check_rust_file_size.py` is removed (the checker demands
  it).
