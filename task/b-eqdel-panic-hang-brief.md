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

# Unit B charter — MoR equality-delete panic + hang + poison policy (audit BUG-004/SAF-001/SAF-002/SAF-003, stretch SAF-006)

Branch: `fix/audit-mor-eqdel-panic-hang` (created from main; this charter is the tip commit).

## The defects (verified 2026-07-17, main @ 4ba52db7062ba67d95d2599833b852071491b49f)

1. **Panic on malformed task** — `crates/iceberg/src/arrow/caching_delete_file_loader.rs:298`:
   `task.equality_ids.clone().unwrap()` on the equality-delete load path. An equality-delete
   `FileScanTaskDeleteFile` with `equality_ids: None` (corrupt/foreign metadata, or a
   deserialized task) panics the scan. Must become a typed `DataInvalid` error carrying the file
   path.
2. **Forever-hang on load failure** — `crates/iceberg/src/arrow/delete_filter.rs:~336-347`
   (`insert_equality_delete`): a spawned task does `eq_del.await.unwrap()`. If the oneshot SENDER
   is dropped without sending (any error after `try_start_eq_del_load` registered the entry —
   including the unwrap in defect 1 once it becomes an early-return error), the receiver errs, the
   unwrap panics INSIDE the spawned task, the entry stays `EqDelState::Loading` forever, and every
   waiter in `read_delete_predicate`/wherever blocks on the Notify indefinitely. The state machine
   needs a terminal failure path: on channel error, transition the entry to a failed/error state
   (or remove it) AND `notify_waiters()`, and waiters must surface a typed error instead of
   re-waiting. Audit every early-return between registration and send in
   `caching_delete_file_loader.rs` — each one currently strands the entry.
3. **Poison-unwrap policy** — the same two files use `.read()/.write().unwrap()` on lock poison in
   production scan paths: `delete_filter.rs:129,150,170,192,203,218,229,310,329,341` and
   `delete_file_index.rs:89,111,122`, plus `unreachable!` at `delete_filter.rs:205,231` and
   `delete_file_index.rs:127,185`. The crate already has a policy to follow elsewhere:
   `arrow/reader.rs:666` recovers poison via `unwrap_or_else(|poisoned| poisoned.into_inner())`;
   `delete_filter.rs:420` / `reader.rs:976` map mutex poison to a typed Error. Unify: pick
   poison-recovery (`into_inner`) for the RwLock state reads/writes (the guarded state stays
   coherent — justify per site) or typed-error propagation where the signature allows; NO
   remaining poison unwraps in these two files. The `unreachable!`s: convert to typed errors if
   any input path can reach them; keep (with a justifying comment) only if truly
   invariant-guaranteed.
4. **Stretch (in scope only if the above is green with time to spare)** — `scan/cache.rs`
   ManifestEvaluatorCache/ExpressionEvaluatorCache map poison to Error then `.unwrap()` anyway
   (~L131-170, L210-233; note the copy-pasted wrong lock names in messages at :215/:230);
   PartitionFilterCache (~:56-101) is the in-file model to match.

## Java parity note

Java's DeleteFilter/DeleteLoader throws on malformed metadata rather than crashing the process,
and has no equivalent hang (synchronous load). No bytecode decode is required for this unit — it
is defensive-correctness work, not format parity. Do NOT change any on-disk format or public API
signature; additive error paths only. If fixing the hang requires a new variant on an existing
pub(crate) enum, that is fine.

## Required pins (tests land with the change, RED-under-mutation proven)

- `equality_ids: None` on an eq-delete task → typed `DataInvalid` error (kind + message content
  asserted), scan does NOT panic. Mutation: restore the unwrap → pin must catch the panic (RED).
- Sender-drop hang: force the load path to fail after registration (e.g. unreadable/nonexistent
  delete file path, or inject via the smallest seam that exists — do not add test-only hooks to
  production code without flagging it as a DEVIATION) → the waiting consumer gets an error within
  a bounded await (wrap in `tokio::time::timeout` — the pin FAILS if it times out). Mutation:
  revert the terminal-state transition → pin goes RED (timeout), restore GREEN.
- Poison-policy: at least one pin per file demonstrating the chosen recovery behaves (a panicked
  holder does not cascade-panic subsequent readers, or errors are typed) — pragmatic scope: pin
  the sites reachable without heroic thread gymnastics; document the rest as swept-by-review.
- All existing lib tests stay green: `cargo test -p iceberg --lib`.

## Gates (ONE `&&` chain to commit; never `git add -A`)

`git ls-files -z | xargs -0 typos --force-exclude` · `cargo fmt --all -- --check` ·
`cargo clippy -p iceberg --lib --tests -- -D warnings` · `cargo test -p iceberg --lib` ·
`./scripts/check_agent_artifacts.sh`.

## Constraints

CLAUDE.md + AGENTS.md bind (typed errors with context/source chains, no bare unwrap in production
paths, no `as` casts, document lock-acquisition order if you touch it, no write-guard held across
`.await` unless bounded-and-justified). NO Cargo.toml/Cargo.lock edits. Edits confined to:
`arrow/caching_delete_file_loader.rs`, `arrow/delete_filter.rs`, `delete_file_index.rs`,
(stretch) `scan/cache.rs`, their tests, and error.rs ONLY if a new context helper is genuinely
needed. Commit trailers exactly:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
`Claude-Session: https://claude.ai/code/session_011wbhUQi8Ts3CAdgvbQm2r5`.

## Report back (final message = the record)

(1) per-defect changes with file:line; (2) the early-return audit of the register→send window
(every path enumerated, each now reaching the terminal state); (3) per-site poison-policy table
(site → chosen policy → why); (4) pin inventory; (5) mutation table (mutation → RED pins →
restored GREEN); (6) gate summary; (7) DEVIATIONS; (8) named residues.
