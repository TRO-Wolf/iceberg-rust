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

# SEPMO canon change request — CCR-2026-08-25 (the Actor dispatch pointer)

**Status: DRAFT — awaiting user ratification.** Canon changes land only by amendment at the master
home (`~/Desktop/Sepmo/`); this repo's `skills/sepmo/` is an instance bound at `spine_version: v2.3`.
This CCR **stacks behind CCR-2026-07-26**, whose v2.3 text is itself unratified on
`infra/sepmo-g4-feedforward`. The repo-side gates listed under *What already landed* are
bar-raising and are independent of ratification.

## The defect (evidence)

`references/04-actor.md` §*Role prompt (copy-paste ready)* is the one artifact that reaches a
fresh-context Actor. It reads:

> Build to the engineering contract in **your tier manual** — that is the canonical home for what
> "outstanding engineering" means, and **it is not restated here.** Read it there and build to it.

The tier manuals were deleted on 2026-08-24 by #220, which generalized them into
[.agents/skills/engineering-method/SKILL.md](../.agents/skills/engineering-method/SKILL.md). So the
prompt directs the Actor to a document that does not exist, and it names none of the three documents
the *Engineering contract* row of [binding-manifest.md](../skills/sepmo/binding-manifest.md) actually
binds: AGENTS.md, the engineering-method skill, and the `rust-code-quality` skill.

The obligation was never in doubt. AGENTS.md `<read_order>` states "Every Actor and every Critic in
this repository loads it", and the manifest row says "Actor binds — defers entirely". What failed is
**transmission**, at the one hop that carries it.

The Critic prompt does not share the defect. `references/05-critic.md`, inside its own copy-paste
block, carries a resolver: "The binding manifest (`../binding-manifest.md`, row 'Risk
lens') resolves 'your tier manual' to the concrete canonical home for the running tier." The Actor
prompt has no such sentence. The placeholder is resolvable for one role and dangling for the other.

**Nothing halted.** The Actor prompt states D1 ("never build on an unstated belief… HALT and
escalate") and D2 ("uncertainty is a full stop") twenty lines above the broken pointer, so a
compliant Actor that followed the pointer had to stop and escalate. #221 and #222 both shipped after
#220 deleted the manuals, with no escalation. Either the pointer was never followed or it was
silently resolved; either way the protocol's own detector for a missing canonical document did not
fire, and the break stayed invisible for a day.

**Observed consequence.** Comment discipline is the part of the engineering contract that only
`rust-code-quality` and AGENTS.md "Comments and prose" carry. #219, #221 and #222 each shipped doc
blocks of 10–30 lines duplicating the unit ledger; #222 needed two corrective trims (−221 and −163
lines) after user review. The armed gate added below flags **26** blocks in #222's pre-trim state and
passes its merged state.

## Root causes — two, distinct

- **RC-1 — A dangling reference inside a dispatch artifact.** #220 retired the tier manuals and
  updated the fork's own prose correctly (all seven fork-owned mentions are accurate history), but
  did not sweep the SEPMO reference set. The copy-paste prompts are not prose about the contract;
  they are the contract's delivery mechanism, and a stale name there is a delivery failure, not a
  documentation nit.
- **RC-2 — Binding is asserted, never confirmed.** The manifest binds the Actor to three documents.
  Nothing anywhere requires the Actor to confirm it read them, so a broken pointer produces silence
  rather than a signal. D1/D2 can only fire if the Actor reaches the missing document; a prompt that
  never resolves to a path does not reach it.

## Amendment A — a copy-paste role prompt carries resolvable references

`references/04-actor.md`, and every other `## Role prompt (copy-paste ready)` block, names concrete
documents or an explicit resolver. Placeholder names such as "your tier manual" are permitted in
prose **addressed to the Orchestrator**, which holds the binding manifest; they are prohibited inside
a copy-paste block, whose reader has only the block.

For 04-actor.md the replacement text is:

> Build to the engineering contract bound for this repository — the canonical home for what
> "outstanding engineering" means, not restated here. Read it and build to it: the priority stack
> (correctness → clarity → production-readiness), the Risk-First mindset, tests-with-code as a hard
> gate, and the language-specific rules. The *Engineering contract* row of the binding manifest
> (`skills/sepmo/binding-manifest.md`) names the exact files; read every one before you edit.

The manifest row stays the single home for WHICH files. The prompt gains the one hop that makes it
resolvable.

## Amendment B — dispatch readiness: the Actor confirms the read

The Actor's first return to the Orchestrator names the engineering-contract documents it read. A
build summary that cannot name them is not accepted, and the unit does not proceed. An unresolvable
reference is a D1 escalation, not a silent substitution.

This is deliberately an acknowledgment, **not** a restatement of the rules in the brief. Duplicating
contract text into every brief is how the copy drifts from the original — the failure this repo
already paid for once with GAP_MATRIX line-number citations, fixed only by permanent anchors. One
sentence of operational constraint in the brief is the ceiling.

## Not an amendment — Actor blindness

Forwarding Critic finding ids into an Actor brief violates rules that **already exist**:
`04-actor.md` §*Design note — the Actor is blind to the Critic* ("The Actor is never told that a
Critic will audit its work… Writing to the test — shaping the code to anticipate and pre-empt the
reviewer") and §*Defect-fix slices* ("The Actor is never told where the defects came from; from its
perspective they are simply requirements").

48 lines across 15 `crates/` files carried Critic, Falsifier and review-round identifiers into
production comments, entering with #181 (2026-07-30) and #183 (2026-08-03). Git dates the residue and
its context but not its author, so this is filed as an **execution failure with a mechanical check**,
not as new canon. Adding canon to cover an unfollowed rule is how a contract grows until readers
skim it — which is the same mechanism as the over-long comment.

## What already landed (asymmetric feed-forward, stamped 2026-08-25, ratification-independent)

Both are bar-raising and repo-local; neither touches canon.

- **`scripts/check_agent_artifacts.sh` v4** — a second needle family for review-process residue
  (`Critic`, `Falsifier`, `SEPMO`), scoped to `crates/` because `task/`, `docs/` and the SEPMO tree
  are its correct homes. Word-bounded, with an anti-probe that
  hard-fails if a needle is ever unbounded enough to match "Critical" or "Critically". Matching is
  case-insensitive: lowercase `critic-octo` is the same residue, and `-w` alone excludes
  "Critical" in either case. The 48 residue lines are swept in the same change, since the gate
  cannot go green beside them.
- **`scripts/check_comment_blocks.sh`** — the `rust-code-quality` scan 10 armed, in `make check` and
  CI. Diff-scoped, per AGENTS.md "Comments and prose" *Scope*. Rustdoc scaffolding (a bare `///`, a
  `# Errors` / `# Panics` / `# Notes` heading) does not count toward the cap: AGENTS.md requires
  those sections, so counting them would buy a heading by deleting an error contract.

Deliberately excluded from the needle set: external audit ids (`SAF-007`, `BUG-001`). They name
findings with a durable home outside the review loop, and `Actor` is excluded as a bare word because
"actor model" is legitimate English a future contributor may need.

## Adoption protocol

1. User ratifies this CCR (after, or alongside, CCR-2026-07-26).
2. Amendments A and B land at the master home as canon, with a changelog entry.
3. This repo re-binds as its own SEPMO unit: reference amendments + manifest version bump.
4. The `dispatch_readiness` manifest row stamped below is replaced by the ratified canon text.
