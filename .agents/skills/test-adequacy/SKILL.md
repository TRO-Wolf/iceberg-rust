---
name: test-adequacy
version: "1.0"
description: >-
  Judge whether a test actually proves the thing it is cited for, before a
  coverage or non-vacuity claim is believed. Load it when reviewing tests in
  a PR, when a Critic must refute (not accept) an Actor's coverage claim,
  when writing a regression test for a bug that must never return, and when
  a suite is green but the behavior under review has no obvious way to fail
  it. Do not load it to decide whether the *code* is right — that is
  ../rust-code-quality/SKILL.md. This skill is the evidence discipline: what
  counts as proof that a test is load-bearing, and the vacuity patterns that
  have each produced a false green in this repository at least once.
---

# Test Adequacy

A green suite proves that nothing *observed* broke. It does not prove that anything is *observable*.
This skill is the procedure for closing that gap: turning "the test covers it" from an assertion
into evidence.

It records a proven *sequence* and states no project rule. The rules it leans on are AGENTS.md
"Parity mandate" (tests land with the code; a GAP_MATRIX row flips only with unit **and** interop
evidence), AGENTS.md "Crate code — testing", AGENTS.md "Working conventions" (the sabotage and
gate-chain rules), and [docs/testing.md](../../../docs/testing.md) — on any conflict those win. The
code-correctness companion is [../rust-code-quality/SKILL.md](../rust-code-quality/SKILL.md).

## What a green run already tells you — and what it does not

Held by the machine, so re-asserting it is not a catch:

- The suite compiled and every assertion that *ran* passed (`make unit-test`, `make test`).
- Every test function contains at least one assertion — AGENTS.md "Crate code — testing" requires it.

None of that distinguishes an assertion that would fail if the behavior regressed from one that
would not. That distinction is only ever established by **making the behavior wrong on purpose and
watching the test go red**. Everything below is about doing that honestly.

## Apply, never predict

**A named mutation that was not executed is a hypothesis, not evidence.** "Reverting the null guard
would fail `test_not_eq_null`" is a prediction; running with the guard removed and reading three
red tests is a measurement. The two are routinely confused because the prediction is usually right —
which is exactly why the times it is wrong are the times that matter, and the only way to find them
is to run it.

So:

1. **Name the claim.** Which line, branch, or invariant is asserted to be covered.
2. **Name the population.** How many tests were run in the mutated build, not just how many turned
   red. `cargo test -p iceberg --lib` is a number; write it down before you mutate.
3. **Apply the mutation** — revert the guard, flip the comparison, drop the field, return the
   default — and run the same command.
4. **Record the arithmetic** in the shape `<N> red out of <M>` — e.g. `3 red out of 3375`, where
   `M` is whatever the baseline run reported *this session*, never a number carried over from an
   earlier one. Never `caught`, never `covered`, never a bare check mark. A number without its
   population is unfalsifiable, and an unfalsifiable claim is indistinguishable from a false one.
5. **Restore and re-run**, and confirm the count returns to the baseline. A mutation left in the
   tree is the worst possible outcome of a coverage check.

## An Actor's arithmetic is a hypothesis too

Removing N knobs at once and observing N failures does **not** prove that each knob is individually
load-bearing. The failures could all trace to one of them; the other N−1 may be inert, or may be
masked by the first one's failure. This was caught live here (unit G5): a batch mutation produced a
tidy matching count that dissolved when the knobs were removed one at a time.

**Remove them one at a time.** N mutations means N runs, each with its own population line. If a
reviewer is checking someone else's coverage claim, the reviewer applies the mutation — an Actor's
reported arithmetic is a hypothesis with a plausible number attached, and a Critic who merely reads
it has verified nothing. This is why the independent Critic in AGENTS.md `<subagent_policy>` exists
at all: the mutation must be re-run by someone who does not already believe the result.

## Green-vacuity patterns

Each of these has produced a real false green here. The tell is what to look for in the test source;
the repair is what makes the same test carry falsifiable content.

### 1. A precondition that is an identity by construction

**Tell.** The threshold, budget, or bound in the assertion is *derived from the measured value*, so
the comparison is arithmetically guaranteed — `assert!(size < size + 1)` wearing a longer name.

**Why it survives review.** It reads as a real bound because the names are real (a measured peak, a
computed budget) and the number is plausible. Nothing about the source under test can change the
outcome.

**Repair.** Say so in the test: record it as true-by-construction, and move the falsifiable content
somewhere it can fail — an independently measured baseline, a value pinned from a previous run, or a
constant the change would have to move.

### 2. A post-action assertion that is only read identity

**Tell.** The test performs an action, then asserts that reading the table back yields the same rows
it expected. A run in which the action **declined to do anything** — a no-op commit, a refused
maintenance action, an empty rewrite — satisfies exactly that assertion.

**Why it survives review.** Read identity is the thing a user cares about, so asserting it feels
like the strongest possible check. It is the weakest one available for *did the action happen*.

**Repair.** Assert **shape**, not just content: file counts, the file set before and after, snapshot
ids, sequence numbers, the manifest count, the operation recorded in the snapshot summary. A no-op
must be distinguishable from success by the assertion alone.

### 3. A fixture property asserted against a value built to have it

**Tell.** The test constructs a sorted vector and then asserts it is sorted; builds a struct with a
field set and asserts the field is set; writes three files and asserts three files exist. The
assertion re-reads the constructor.

**Why it survives review.** It looks like an invariant check, and sometimes an invariant check with
that exact spelling is legitimate elsewhere in the file.

**Repair.** Route the value through the code under test before asserting the property — the
production sorter, the production builder, the production writer. If the fixture genuinely needs the
property as a precondition, `debug_assert` it or comment it as a fixture invariant, and do not count
it as coverage.

### 4. The three-valued-logic `=` trap

**Tell.** A NULL-handling guard is claimed to be covered by a test whose predicate is `=` (or any
equality-shaped comparison).

**Why it is fatal.** Under SQL three-valued logic, `x = NULL` is UNKNOWN, which the row filter
treats as *not matching* — the same outcome the missing-`is_valid` guard produces. So an `=`-only
predicate **cannot** distinguish a live null guard from a dropped one: the test passes either way,
and the mutation is silently unkillable.

**Repair.** Test the guard with `<>` or `NOT`, where a NULL operand makes the naive
value-comparison return **TRUE** and only the `is_valid` guard suppresses the row. Then
mutation-prove it: drop the guard, watch it go red, record the arithmetic. Two Opus Critics found
this class here on 2026-06-25 after every weaker review had passed the same test as adequate.

## A sabotage step that cannot be applied must HARD-FAIL, never SKIP

A negative test that corrupted nothing has proven nothing. If the corruption cannot be applied —
the target byte pattern is absent, the fixture moved, the field was renamed — the run must exit
non-zero and abort, never log "skipped" and continue green. A SKIP branch in a sabotage script is a
false green with a paper trail that reads like a pass. AGENTS.md "Working conventions" carries this
as a standing rule (promoted after three separate interop sabotage steps hit it).

Mechanically, under `set -euo pipefail`, capture the mutator's exit with `|| rc=$?` so the restore
of any `.bak` stays reachable, restore first, then exit non-zero. A sabotage harness that leaves the
tree corrupted on its own failure path is worse than no harness.

## Chain the gate to the commit in ONE `&&` chain

`typos . && cargo fmt --all -- --check && git add … && git commit …` — one chain. A gate on its own
line still lets the commit run when it fails, so the commit records a green that was never checked.
This is the same failure shape as the SKIP branch: a step that did not run reads identically to a
step that passed, unless the chain makes it impossible.

## An unkillable assertion, honestly recorded, beats a false coverage claim

Sometimes the mutation does not go red and cannot be made to: the branch is unreachable from any
public entry point, the behavior is masked by an earlier guard, or the only observable difference is
in a layer the test cannot see. That is a real and reportable result. Write it down —
*"mutation applied, 0 red out of M; the branch is unreachable through `TableScan`, so this
assertion is not load-bearing"* — and then either extend the harness until it can observe the
difference, delete the assertion as decoration, or name the gap as a residue.

What is never acceptable is converting an unkillable assertion into a coverage claim by not
mentioning the run. The arithmetic is the deliverable; a claim without it is not evidence, it is
prose.

## Review checklist

- [ ] Every coverage claim names the mutation, the command, and the arithmetic (`N red out of M`).
- [ ] Multi-knob claims were mutated **one knob at a time**, one run and one population line each.
- [ ] The reviewer re-applied at least the load-bearing mutation rather than reading the Actor's
      number.
- [ ] No assertion is an identity by construction, a bare read-identity post-check, a re-read of the
      fixture constructor, or an `=`-only probe of a null guard.
- [ ] Negative/sabotage steps hard-fail when the corruption cannot be applied, and restore first.
- [ ] The verification gate is chained to the commit in one `&&` chain.
- [ ] A parity claim carries interop evidence, not only unit evidence (AGENTS.md "Parity mandate").
- [ ] Unkillable assertions are recorded as such, with their arithmetic, not quietly counted.

## Output template

```
## Test Adequacy Report

### Baseline
- command: cargo test -p iceberg --lib
- population: M passing

### Mutations applied (one at a time)
1. <file:line> — <what was made wrong>
   - result: N red out of M
   - tests that went red: <names>
2. ...

### Vacuity sweep
- identity-by-construction: none / <file:line, disposition>
- read-identity-only post-check: none / <...>
- fixture self-assertion: none / <...>
- `=`-only null-guard probe: none / <...>

### Unkillable assertions
- <file:line> — mutation applied, 0 red out of M; <why, and the disposition>

### Verdict
ADEQUATE / NOT ADEQUATE (list the claims that lack arithmetic)
```
