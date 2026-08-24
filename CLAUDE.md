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

# CLAUDE.md — the Claude adapter (not authoritative)

**STOP — the authoritative contract is [AGENTS.md](AGENTS.md). Read it first.** This file adds only
Claude-specific tool mechanics; it states **no rule the spine does not already own**. Every project fact lives in the
neutral spine, and this adapter only points at it — so it cannot drift, and deleting it would lose
no project knowledge.

## Where the project rules actually live

CLAUDE.md is not their home. Follow the pointers:

| For… | Read (authoritative) |
|---|---|
| The read path + the rules governing any change | [AGENTS.md](AGENTS.md) (start at its `<read_order>`) |
| The precedence / authority chain | [AGENTS.md](AGENTS.md) `<precedence>` — its single home |
| Repo intent, the fork's north star, the parity mandate | [AGENTS.md](AGENTS.md) "Parity mandate" |
| The phase plan and the current phase | [Roadmap.md](Roadmap.md) |
| Per-capability status (the only home for a status) | [docs/parity/GAP_MATRIX.md](docs/parity/GAP_MATRIX.md) |
| What is irreversible / hard-blocked | [AGENTS.md](AGENTS.md) "Absolute prohibitions" |
| Rust rules — errors, concurrency, casts, recursion, tests | [AGENTS.md](AGENTS.md) "Rust conventions — the engineering contract" |
| Build, test, and gate commands | [AGENTS.md](AGENTS.md) "Build & test commands" |
| The testing-discipline contract | [docs/testing.md](docs/testing.md) |
| Directory navigation, and when a `map.md` is owed | [AGENTS.md](AGENTS.md) `<map_md_navigation>` |
| Lifecycle / process governance | [skills/sepmo/SKILL.md](skills/sepmo/SKILL.md) + [binding-manifest.md](skills/sepmo/binding-manifest.md) |
| Navigation for a directory you will touch | that directory's `map.md` |

## Claude read order (every session)

1. **[AGENTS.md](AGENTS.md) first**, then follow its `<read_order>` (Roadmap → GAP_MATRIX → tier
   manual → SEPMO → `task/lessons.md` + `task/todo.md` → the touched directories' `map.md`).
2. **The engineering method** —
   [.agents/skills/engineering-method/SKILL.md](.agents/skills/engineering-method/SKILL.md), the
   portable agent-agnostic working method (it replaced the per-tier manuals 2026-08-24; one
   instruction set for every tier, not a separate source of truth). Tier postures live below.
3. The `map.md` of every directory your task will touch (AGENTS.md `<map_md_navigation>`).

CLAUDE.md keeps this filename so Claude tooling that auto-loads it still fires and lands you on
AGENTS.md on turn 1.

## Claude tool mechanics — skills are invocable here

`.claude/skills` is a symlink to `../.agents/skills` (git mode `120000`), so every runbook there
loads natively in a Claude session and can be invoked by name rather than opened by path. The skills
keep their single home under `.agents/`; that directory adds no second copy and states no rule.
Roster and reasoning: [.agents/skills/map.md](.agents/skills/map.md).

The SEPMO control plane under [skills/](skills/map.md) is a separate tree and is **not** covered by
that symlink; it is invoked deliberately, not discovered.

## Claude tool mechanics — capability tiers and sub-agents

These are Claude-family orchestration mechanics, **not** project rules. [AGENTS.md](AGENTS.md)
`<subagent_policy>` is the neutral rule — single agent for the small stuff, a mandatory *independent*
Critic (separate agent, fresh context) on anything that ships as a PR, both roles defaulting to the
frontier tier. This is how that maps onto Claude tiers:

- **`OO AC` = Opus–Opus Actor–Critic is the default pair.** Whenever you spawn an Actor and/or a
  Critic, both default to Opus (`model: "opus"`) at high reasoning effort — the Claude realization
  of AGENTS.md's frontier–frontier (FF) pair.
- **Never turn the Critic below Opus on a correctness-bearing review.** Recorded evidence: on
  2026-06-25 two Opus Critics caught (and mutation-proved) a NULL-three-valued-logic coverage gap
  that *every* Sonnet Critic in the same effort — including a dedicated "final" bundle Critic — had
  missed.
- You may turn the **Actor** down to Sonnet or Haiku only for genuinely rote sub-work (large
  mechanical renames, log scraping) — and say so explicitly in the report when you do. Brief the
  tier's posture when you delegate: **Sonnet** is the delegated implementation tier (executes
  well-scoped work; architecture stays with the orchestrating session — surface ambiguity rather
  than inventing); **Haiku** is the narrow mechanical tier (precisely specified edits; stop and
  hand back the moment the task needs a design decision). Every tier reads the same
  [engineering method](.agents/skills/engineering-method/SKILL.md); the non-negotiables are
  identical across tiers.
- `Workflow` fan-out and plan-mode `Explore` / `Plan` helpers stay opt-in.

**Fable / Mythos-class sessions** (the portable frontier notes are in the engineering method's
"Frontier Operating Notes"; these are the Claude-specific mechanics):

- Fable-tier tokens cost roughly **2× Opus** ($10 / $50 per MTok at launch) — the economics that
  justify the tier are one deep, correct pass over several cheap iterative ones.
- **Never spawn a Fable/Mythos-tier sub-agent** unless the user names the tier explicitly — a
  fan-out of frontier-priced agents above Opus is a budget decision only the user can make.
- Fable carries hard safety limits in some dual-use domains (notably offensive cybersecurity);
  requests there may be blocked or served by a fallback model. This repo rarely trips them, but
  adjacent work can (auditing credential handling, fuzzing parsers, reasoning about
  deserialization exploits in Avro/Parquet readers). If you cannot fully engage, flag the gap
  explicitly per the method's "Degraded-capability honesty" — never emit a hedged answer that
  looks complete.

Relax nothing here: an adapter may name tiers, never loosen the rule it implements. If the neutral
rule needs to change, change it in [AGENTS.md](AGENTS.md) `<subagent_policy>` and note it in
[task/lessons.md](task/lessons.md).
