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

# binding-manifest.md — SEPMO ⨉ this repo

## Purpose

This is the **one project-specific file** in SEPMO. The SEPMO control plane — the state machine,
the scope audit + 100% gate, the Actor–Critic protocol, PR-grouping, vigilance, delivery, and
retrospective — is **portable and identical across projects**. This manifest is what *binds* SEPMO's
abstract roles to the concrete machinery of **this** repository. To run SEPMO on a different
project, you replace this file and nothing else.

SEPMO defines roles abstractly — "the Actor builds under *the project's engineering contract*,"
"the charter derives from *the project's plan-of-record*." The rows below say what those italicized
phrases resolve to **here**. **One home per fact:** SEPMO never restates what these homes own; it
points to them.

## Precedence

SEPMO governs **lifecycle and orchestration only**. On any conflict, the chain in
[CLAUDE.md](../../CLAUDE.md) `<precedence>` wins, and SEPMO cedes the engineering contract to the
tier manuals and to `CLAUDE.md`. That block is the single home for the authority chain — this file
does not restate it.

## The bindings

| SEPMO abstraction | Canonical home in this repo | SEPMO's relationship |
|---|---|---|
| **Engineering contract** (what the Actor builds under) | the [skills/](../) tier manual for the running model (`Opus.md` / `Sonnet.md` / `Haiku.md` / `Fable.md`); plus [AGENTS.md](../../AGENTS.md) for crate code | Actor **binds** — defers entirely; restates nothing |
| **Risk lens** (what the Critic attacks with) | `Opus.md` *Risk-First Mindset* (risk-surface table, double-execution rule, time-of-check/time-of-use windows) | Critic **uses** it as the attack basis |
| **Done gate** (what Delivery verifies) | `Opus.md` §4 *Verification Before Done* + [docs/testing.md](../../docs/testing.md); commands `make check` + `make test` | Delivery **invokes** it by reference |
| **Plan-of-record** (the charter source) | [Roadmap.md](../../Roadmap.md) — phase plan + sequencing | Orchestrator **derives** the charter from it |
| **Capability status (SSOT)** | [docs/parity/GAP_MATRIX.md](../../docs/parity/GAP_MATRIX.md) — status lives *only* here | Delivery **flips cells**; SEPMO never restates a status |
| **PR-unit grouping** | "Waves" as used in [task/todo.md](../../task/todo.md) + `todo-archive/` | Orchestrator's PR-grouping **is** Wave-carving |
| **Active plan tracking** | [task/todo.md](../../task/todo.md) — the 3–7 bullet plan | Orchestrator **writes here**; no parallel tracker |
| **Memory / lessons** | [task/lessons.md](../../task/lessons.md) (active) + [skills/compaction.md](../compaction.md) (lifecycle) + `lessons-archive/` | Retrospective **runs a compaction pass** |
| **Navigation** | the `map.md` convention — [CLAUDE.md](../../CLAUDE.md) `<map_md_navigation>` | every SEPMO directory **carries one** |
| **Prohibitions** | [CLAUDE.md](../../CLAUDE.md) *Absolute prohibitions* | all agents **obey**; SEPMO adds none |
| **Sub-agent / tier policy** | [CLAUDE.md](../../CLAUDE.md) `<subagent_policy>` + `Fable.md` *Frontier Addendum* | Orchestrator's AC-execution mode **follows** it (single-agent default; literal OO is opt-in) |
| **Mode handling** (interactive vs. delegated) | the tier manuals' *Mode Handling* section | Orchestrator + all agents **adopt** both modes |
| **Debugging protocol** | `Opus.md` §8 + each directory's `map.md#debug` | Actor/Critic **follow** it on failure |

## To port SEPMO to a new project

1. Copy `skills/sepmo/` (the shell) into the new repo **unchanged**.
2. Replace **this file** with the new project's bindings — point each row at that repo's engineering
   contract, plan-of-record, status SSOT, memory system, and navigation convention. A project that
   lacks one of these should create it (or accept SEPMO's built-in default for that role).
3. Wire the precedence + read-order into the new repo's root agent file (the equivalent of
   `CLAUDE.md`).
4. **Nothing inside the shell changes.** If you find yourself editing a reference file to fit the new
   project, that belief is wrong: the project-specific fact belongs in *this* manifest, not in the
   shell. (This is D1 — Death to Assumptions — applied to the convergence itself.)

## Pointers

- **Up:** repo root [CLAUDE.md](../../CLAUDE.md).
- **Related:** [skills/map.md](../map.md) (the skills index); [SKILL.md](SKILL.md) (the SEPMO spine).

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| A SEPMO reference contradicts a tier manual on engineering | The manual wins (precedence). SEPMO should be *pointing* here, not restating — fix the reference. |
| The same capability status appears in two places | De-triplication breach. The GAP_MATRIX is the only home — delete the copy, link instead. |
| A binding row points at a file that does not exist in this repo | This manifest is stale, or was ported without updating — fix the row. |
| Unsure which manual is "the engineering contract" right now | The one matching the running model tier (Mythos-class → `Fable.md`); if unknown, `Opus.md`. |

### First checks

- Did you read [CLAUDE.md](../../CLAUDE.md) first? It sets precedence and the read-order.

### Escalate to

- Conflicts / precedence → [CLAUDE.md](../../CLAUDE.md) `<precedence>`.
- Unresolved → open an issue.
