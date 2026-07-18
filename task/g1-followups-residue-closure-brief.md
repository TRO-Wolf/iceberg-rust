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

# Group G1 charter — residue closure: delete-vector fail-open + op-class guard + R160 cell text

Branch: `fix/audit-followups-bundle` (Mode B bundle, group 1 of 5; keep commits scoped to G1).
Follow-ups from the merged overnight audit block (#158/#159/#160).

## G1a — `get_delete_vector_for_path` fail-open (HIGH-value one-liner)

`crates/iceberg/src/arrow/delete_filter.rs` — `get_delete_vector_for_path` uses `.read().ok()`,
so a poisoned state lock returns `None`: positional deletes silently DROP after any poisoning
panic, and deleted rows resurrect (empirically reproduced by the #160 unit-B Critic). The file
already has the `recover_poison` helper (added by #160, near the top of the impl) and every other
lock site in the file now uses it — convert this last site to match. Check the sibling
`get_delete_vector` (if one exists) and any other `.ok()`-on-lock in the file while you are
there; convert or justify each.

Pin: poison the lock (spawn a thread that panics while holding write), then assert the delete
vector for a present path is STILL returned (not None). Mutation: revert to `.read().ok()` → pin
RED; restore byte-identical.

## G1b — op-class partition guard test (test-only; #160 unit-C residue)

`PredicateOperator`'s `is_unary()/is_binary()/is_set()` are discriminant RANGE checks, now
load-bearing for wire-input rejection (the serde arity guards). Add a test with an EXHAUSTIVE
`match` over every `PredicateOperator` variant (no wildcard arm — a future variant must break
compilation, that is the point) asserting each is in EXACTLY ONE class. Location: wherever the
operator enum's tests live (`expr/mod.rs` or `predicate.rs` test module — follow the file).
Mutation: temporarily add a fake mismatch (e.g. assert a binary op is_unary) → RED, or
demonstrate the compile-break by matching on one-fewer variant; document which proof you used.

## G1c — R160 cell-text accuracy (docs; #159 unit-D Critic residue)

`docs/parity/GAP_MATRIX.md` row R160: the cell names `LoadTableResponse.java
(storageCredentials)` — the Java accessor is actually `credentials()` (field `credentials`);
`storageCredentials`-camel is the wire-key constant for `"storage-credentials"`. Also
`newFileIO(properties, List<Credential>)` elides the real signature
`newFileIO(SessionContext, Map, List<Credential>)`. Correct BOTH namings, keep the cell terse,
do not change the row's status or anchor. MANDATORY after the edit:
`./scripts/check_matrix_anchors.sh` green.

## Gates (ONE `&&` chain to commit; never `git add -A`)

`git ls-files -z | xargs -0 typos --force-exclude` · `cargo fmt --all -- --check` ·
`cargo clippy -p iceberg --lib --tests -- -D warnings` · `cargo test -p iceberg --lib` ·
`./scripts/check_matrix_anchors.sh` · `./scripts/check_agent_artifacts.sh`.

## Constraints

CLAUDE.md + AGENTS.md bind. NO Cargo.toml/Cargo.lock edits. Edit surface: `delete_filter.rs`,
the expr test module, `docs/parity/GAP_MATRIX.md`, tests. Commit trailers exactly:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
`Claude-Session: https://claude.ai/code/session_011wbhUQi8Ts3CAdgvbQm2r5`.

## Report back (final message = the record)

(1) changes file:line incl. the lock-site sweep result; (2) pin inventory; (3) mutation table
(mutation → RED pins → restored GREEN, or the compile-break proof for G1b); (4) gate summary;
(5) DEVIATIONS; (6) named residues.
