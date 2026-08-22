---
name: rust-code-quality
version: "1.0"
description: >-
  Run a focused Rust quality review over changed `crates/` code when one is
  requested, when reviewing a Rust PR or commit, or when another review
  workflow (a SEPMO Critic, an audit) delegates the Rust-specific checks. Do
  not auto-load it for every implementation edit, and do not use it to
  re-check what the armed gates already hold (`make check` — fmt, clippy
  `-D warnings`, taplo, cargo-machete, the agent-artifact and matrix-anchor
  scripts — and `typos`, which runs in CI beside it, not inside it). It exists for the findings no linter here can
  reach: on-disk format stability, divergence from Java `iceberg-core`,
  panics and value-path casts (this workspace arms neither), broken error
  chains, lock discipline, and unbounded recursion over parsed input.
---

# Rust Code Quality Gate

A review procedure for a Rust diff in this workspace. It records a proven review *sequence*; every
rule it leans on is a pointer into the spine ([AGENTS.md](../../../AGENTS.md) "Rust conventions",
"Absolute prohibitions", "Parity mandate") or the per-tier manual
([skills/Opus.md](../../../skills/Opus.md)) — on any conflict, those win. This skill states no
project rule; it decides nothing that AGENTS.md has not already decided. The evidence companion is
[../test-adequacy/SKILL.md](../test-adequacy/SKILL.md) — this skill asks *is the code right*, that
one asks *does the test prove it*.

## What the gates already hold — never re-review it

Machine-held, so a reviewer's finding on these is a duplicate, not a catch:

- **Formatting**: `make check-fmt` (`cargo fmt --all -- --check`), config in
  [rustfmt.toml](../../../rustfmt.toml). Never comment on import grouping or layout.
- **Warn-level lints**: `make check-clippy` runs
  `cargo clippy --all-targets --all-features --workspace -- -D warnings`.
- **Missing docs**: `#![deny(missing_docs)]` in `crates/iceberg/src/lib.rs` — a public item without
  a doc comment cannot compile.
- **TOML**: `make check-toml` (`taplo check`). **Unused deps**: `make cargo-machete`.
- **Agent-session artifacts**: `scripts/check_agent_artifacts.sh` — tool-call wrapper tags cannot be
  committed.
- **GAP_MATRIX structure**: `scripts/check_matrix_anchors.sh` — the 5-pipe row audit, the permanent
  `R<id>` anchors, and that every `row R<id>` citation *resolves*.
- **Spelling**: `typos .`.

**What is NOT held here, and is therefore live review work.** This is the load-bearing difference
from a workspace with a pedantic-clippy or panic-ban configuration:

- There is **no `clippy.toml`** and **no `[workspace.lints]`** — `unwrap()` / `expect()` are not
  denied anywhere, in production code or elsewhere.
- Clippy runs `clippy::all`, **not** `clippy::pedantic` — `cast_possible_truncation`,
  `cast_sign_loss`, and `cast_precision_loss` are all off. A truncating `as` cast compiles green.
- Nothing bans `println!` / `dbg!` in library code.
- Nothing checks that a GAP_MATRIX citation uses an anchor rather than a line number (the anchor
  script validates only citations already written in `row R<id>` form).

## Quick start

1. Identify the changed `.rs` files (`git diff --name-only` on the range under review).
2. Run the candidate scans below on those files.
3. Walk the manual checklist against the diff.
4. Resolve or rebut every finding with evidence; P0/P1 findings cannot be deferred.

## Candidate scans

These find *candidates*, not findings. Inspect the syntax, the `#[cfg(test)]` scope, and the changed
hunk before reporting — text filters do not reliably distinguish production code from tests, and
test code legitimately unwraps (AGENTS.md permits `.expect("context")` there, and asks for it over a
bare `.unwrap()`).

```bash
# 1. New escape hatches — every one is a review item
rg -n '#\[(expect|allow)\(' <changed-files>

# 2. Value-path `as` casts — NOT gated in this workspace; judge each one
rg -n ' as (i8|i16|i32|i64|i128|u8|u16|u32|u64|usize|isize|f32|f64)\b' <changed-files>

# 3. Panic paths — NOT gated in this workspace; only `#[cfg(test)]` uses are legitimate
rg -n '\.unwrap\(\)|\.unwrap_err\(\)|\.expect\(|panic!|todo!|unimplemented!' <changed-files>

# 4. Stringly / boxed errors — no new ones on a public surface
rg -n 'Result<[^,>]+,\s*String>|Box<dyn (std::error::)?Error' <changed-files>

# 5. Dropped error chains — the inner error interpolated into the message instead of `with_source`
rg -n 'Error::new\([^)]*format!\("[^"]*\{(e|err|error)' <changed-files>

# 6. Output macros in library code — held by review until armed
rg -n 'println!|eprintln!|dbg!' <changed-files>

# 7. Relaxed atomics — verify each ordering is argued, not defaulted
rg -n 'Ordering::Relaxed' <changed-files>

# 8. GAP_MATRIX cited by line number instead of by anchor
rg -n 'GAP_MATRIX[^)]*[:#]L?[0-9]+|line [0-9]+ of .*GAP_MATRIX' <changed-files>
```

## Manual review checklist

### Format stability (the P0 axis)

- [ ] Nothing in the diff changes an **on-disk encoding** — a manifest/metadata field name, type,
      field id, default, required flag, Avro or Parquet schema, sort order, partition spec
      serialization, or Puffin blob layout — without explicit approval and a round-trip test proving
      an already-written table still reads back identically (AGENTS.md "Absolute prohibitions").
- [ ] A `spec/` change is traced through **every** reader and writer that consumes it; the spec
      module is the source of truth, so a change there is never local.
- [ ] A write/commit-path change (`transaction/`, `writer/`, `maintenance/`) cannot lose or reorder
      a concurrent writer's commit, and cannot drop, duplicate, or mis-sequence a delete file.
- [ ] Any new file the writers emit is reachable for cleanup — an orphan is a cost leak, a
      prematurely deleted file is corruption.

### Parity with Java (the P1 axis)

This axis has no analogue in a non-parity repo, and it is the reason the fork exists.

- [ ] Every behavior change in the diff is checked against Java `iceberg-core` / `iceberg-api`
      **1.10.0** — the reference checkout source, or the jar bytecode when the source is ambiguous.
      "It looks right" is not a check; name what was read.
- [ ] A divergence found is either **fixed** or **NAMED as a residue** with a GAP_MATRIX row in the
      same change. An unnamed divergence is the finding, not the divergence itself.
- [ ] GAP_MATRIX rows are cited by **permanent anchor — `row R<id>`, never by file line number**.
      Line-number citations broke in four separate waves here; prose inserted above the table shifts
      rows just as insertions do (AGENTS.md "Working conventions").
- [ ] A row flips to ✅ only with unit tests **and** an interop test (AGENTS.md "Parity mandate").
      A flip in the diff without both is a finding.

### Error handling

- [ ] No new `Result<_, String>` and no `Box<dyn Error>` on a public trait or struct method — a
      typed error enum is the contract (AGENTS.md "Crate code — error type design").
- [ ] **Every wrap preserves the chain.** Interpolating an inner error into a `format!` message and
      constructing a fresh `Error` drops `source()` and makes the real cause unrecoverable. Live
      examples to model the fix on, not to copy: `crates/iceberg/src/arrow/record_batch_projector.rs`
      (`Error::new(ErrorKind::DataInvalid, format!("{err}"))`) and
      `crates/iceberg/src/spec/values/decimal_utils.rs` (`format!("Can't parse decimal: {e}")`).
      Attach the inner error as the source instead.
- [ ] The `ErrorKind` is the *caller's* kind, not the inner layer's convenience. A missing object
      surfacing as `DataInvalid` sends every caller down the wrong branch.
- [ ] Internal helpers returning `Result<_, String>` and immediately wrapped by `.map_err(...)`
      return the real error type instead.

### Panics and casts (unarmed here — review is the only gate)

- [ ] No `.unwrap()` / `.unwrap_err()` / `.expect()` / `panic!` / `todo!` / `unimplemented!` in a
      production path. Slicing, indexing, and integer arithmetic on parsed input are panics too.
- [ ] Every new `as` cast on a value path is justified in the diff or replaced with `try_into()` and
      a typed error. AGENTS.md: *treat every `as` cast in review as a potential bug*. A clamp is
      acceptable only where the domain is provably bounded, and the bound is stated.
- [ ] Every new `#[expect(...)]` / `#[allow(...)]` is per-call-site with a stated reason — never a
      file- or crate-wide blanket added to make a lint go away.

### Concurrency

- [ ] Lock-acquisition order is documented where a module takes more than one lock, and no path
      takes the same set in a different order.
- [ ] No `RwLock` / `Mutex` write guard held across an `.await` unless the section is unavoidably
      async and the hold time is bounded — say which.
- [ ] Concurrent counters use `compare_exchange` loops, not load-then-store.
- [ ] Each `Ordering::Relaxed` is argued at the site; when in doubt, the stronger ordering with a
      comment beats the weaker one without.

### Recursion and untrusted input

- [ ] Any recursion over user- or file-influenced input (schema trees, nested types, expression
      trees, parsed JSON/Avro, manifest hierarchies) carries a depth bound or an explicit-stack
      iterative form. A malformed or malicious table must not overflow the thread stack.
- [ ] A parser or decoder handles truncation, absent fields, and out-of-range values as typed
      errors, not as an index panic.

### Async, logging, house style

- [ ] CPU-heavy work is off the async hot path (`spawn_blocking`) — decoding, compression, and
      sketch merges are the usual offenders.
- [ ] Logging is `tracing` with structured fields (`?error`, ids, durations); no `println!` in
      library code; **no secret, credential, token, or credentialed S3 URI in any field**.
- [ ] Section banners follow the house form only where the surrounding module already uses them —
      do not introduce them into a module that does not.

### Tests, docs, maps

- [ ] Tests ship in the same change, plus an interop test where the Parity mandate calls for one.
      Whether those tests actually *prove* anything is the separate pass in
      [../test-adequacy/SKILL.md](../test-adequacy/SKILL.md); do not sign off coverage from this
      checklist alone.
- [ ] Every branch the change adds has a nameable input where it changes the output — a dead branch
      is a defect, not a belt-and-brace.
- [ ] Touched directories' `map.md` updated in the same change (AGENTS.md `<map_md_navigation>`);
      a new source directory in a tree that already uses the convention gets one.
- [ ] No capability *status* written anywhere but the GAP_MATRIX cell (AGENTS.md one-home-per-fact).
- [ ] Comments carry the non-obvious invariant, not narration of the next line.

## Severity

**This P0-P3 scale is review-local.** It ranks findings inside one Rust review and is
deliberately NOT SEPMO's S0-S3 severity floor, which governs whether a unit may converge. Do not
map one onto the other; cite whichever you mean by name.

The product of this repository is **on-disk format stability** — other people's already-written
tables. So the scale is ordered by *how long the damage outlives the commit*: a corrupted table
outlives the process, the release, and the fix, and is often only discovered by a reader months
later, while a panic is loud, local, and recoverable. A silent parity divergence is the same failure
one level up: the fork's entire claim is that Java's behavior is ours, so an *unrecorded* difference
quietly invalidates every downstream parity claim — which is exactly why an honestly NAMED residue
is not a finding at all.

- **P0 (block)**: silent data or format corruption — a changed on-disk encoding, a lost or
  mis-sequenced commit, a dropped or double-applied delete, a file deleted while still reachable.
  Undefined behavior. Anything that makes an already-written table read back wrong.
- **P1 (block)**: a behavioral divergence from Java `iceberg-core` 1.10.0 that is neither fixed nor
  registered as a residue with a GAP_MATRIX row; a ✅ flip without both unit and interop evidence.
- **P2 (must fix)**: a panic reachable from parseable input or a malformed table; an unjustified
  value-path `as` cast; a broken `Error::source()` chain; a secret reachable by a log line; a
  blanket `#[allow]`.
- **P3 (should fix)**: avoidable duplication with a concrete simpler replacement; a stale `map.md`;
  a local clarity issue with no behavioral risk.

## Output template

```
## Rust Code Quality Report

### Candidate scans
- escape-hatch candidates inspected: N
- value-path cast candidates inspected: N
- panic-path candidates inspected: N
- stringly/boxed-error candidates inspected: N
- dropped-chain candidates inspected: N
- output-macro candidates inspected: N
- relaxed-ordering candidates inspected: N
- line-number-citation candidates inspected: N

### Java references read
- <class#method or jar bytecode> for <behavior> — matched / diverged (row R<id>)

### Findings
- [P1] `path:line` — one-sentence defect
  - Evidence: what was measured or read that proves it
  - Fix: ...

### Verdict
PASS / BLOCKED (list the P0/P1 findings)
```

## Arming candidates

Each of these is mechanically decidable and currently held only by this checklist. Arming one is its
own change, and it must ship with a **provocation proof**: plant the violation, show the gate red,
remove it, show the gate green. A gate that has never been shown to catch its own probe is vacuous —
the same doctrine as AGENTS.md's "a sabotage step that cannot be applied must HARD-FAIL, never SKIP".

| Candidate | Mechanism | Blocker |
|---|---|---|
| `unwrap` / `expect` / `panic!` ban in production code | a `clippy.toml` `disallowed-methods` list (tests exempt) | the inherited upstream surface uses them freely; needs a sweep or a staged per-crate ratchet before it can be `-D` |
| Truncating-cast lints | `clippy::cast_possible_truncation` / `cast_sign_loss` / `cast_precision_loss` on `make check-clippy` | same — sweep first, then arm; do not blanket-allow to get green |
| `println!` / `eprintln!` / `dbg!` ban | `clippy::print_stdout` / `print_stderr` / `dbg_macro` | examples and binaries legitimately print; scope the deny to library targets |
| Line-number GAP_MATRIX citations | extend `scripts/check_matrix_anchors.sh` with a reject pattern | must not false-positive on dated archives, where bare-number citations are historical epochs and are deliberately left alone |

When a candidate is armed, delete its scan and its checklist item here in the same change — a
checklist item that duplicates a gate is noise, and noise is how a real finding gets skimmed past.
