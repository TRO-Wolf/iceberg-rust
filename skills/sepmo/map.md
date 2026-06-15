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

# map.md — skills/sepmo/

## Purpose

The **SEPMO control plane**: the portable governance-and-orchestration shell that drives a project
from idea → audited scope → orchestrated Actor–Critic execution → PR → delivery → retrospective.
SEPMO governs *lifecycle*; it **cedes the engineering contract** to the tier manuals
([skills/](../)) and obeys [CLAUDE.md](../../CLAUDE.md) on every conflict. Read [SKILL.md](SKILL.md)
(the spine) **after** the root `CLAUDE.md` and your tier manual; read
[binding-manifest.md](binding-manifest.md) to see how SEPMO's abstract roles bind to this repo.

## Contents

| File | For |
|---|---|
| `SKILL.md` | The spine: the Iron State Machine, the doctrines (thin — each points to its canonical home), the agent roster, and routing to the references below |
| `binding-manifest.md` | The one project-specific file: maps SEPMO's abstractions → this repo's contract, plan, status SSOT, memory, and navigation. **Swap this to port SEPMO.** |
| `references/01-scope-auditor.md` | Scope Auditor — the front-of-pipeline audit + the 100% gate |
| `references/02-orchestrator.md` | Orchestrator — charter ← Roadmap+GAP_MATRIX, PR-grouping, AC-loop coordination, mode handling |
| `references/03-self-logic-review.md` | The pre-action review every agent runs |
| `references/04-actor.md` | Actor — builds under the tier manual (binds); blind to the Critic |
| `references/05-critic.md` | Critic — attacks on the Risk-First lens (binds); convergence is its call |
| `references/06-vigilance.md` | Runtime enforcer of rules the repo already defines |
| `references/07-delivery.md` | Per-PR acceptance = the §4 Done gate + GAP_MATRIX flip discipline |
| `references/08-retrospective.md` | A compaction pass + SEPMO feed-forward |

## I want to...

| I want to... | go to |
|---|---|
| Understand the lifecycle / where I am in it | [SKILL.md](SKILL.md) — the Iron State Machine |
| See how SEPMO binds to this repo | [binding-manifest.md](binding-manifest.md) |
| Audit a new brief / plan before any code | [references/01-scope-auditor.md](references/01-scope-auditor.md) |
| Run the pre-action review | [references/03-self-logic-review.md](references/03-self-logic-review.md) |
| Find the engineering contract (what to build under) | the tier manual in [skills/](../) — SEPMO does not restate it |
| Find the plan / capability status | [Roadmap.md](../../Roadmap.md) + [docs/parity/GAP_MATRIX.md](../../docs/parity/GAP_MATRIX.md) (via the manifest) |
| Resolve a SEPMO-vs-repo conflict | [CLAUDE.md](../../CLAUDE.md) `<precedence>` (it wins) |

## Pointers

- **Up:** repo root [CLAUDE.md](../../CLAUDE.md); the skills index [skills/map.md](../map.md).
- **Related:** the tier manuals ([skills/](../)); [task/](../../task/) (plan + lessons);
  [docs/testing.md](../../docs/testing.md) (the Done gate).

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| A SEPMO reference restates an engineering rule | It should point to the tier manual instead — precedence breach; fix the reference. |
| Following a SEPMO rule that contradicts the repo | `CLAUDE.md` / the manuals win — re-check [CLAUDE.md](../../CLAUDE.md) `<precedence>`. |
| A reference is marked _(planned)_ | The convergence build is incremental — see [SKILL.md](SKILL.md)'s reference map for current status. |
| Unsure whether a change needs the full pipeline | Proportionality: substantial PR-units get full ceremony; trivial changes take the lightweight path ([SKILL.md](SKILL.md)). |

### First checks

- Did you read root [CLAUDE.md](../../CLAUDE.md) and your tier manual before this spine?

### Escalate to

- Conflicts / precedence → [CLAUDE.md](../../CLAUDE.md) `<precedence>`.
- Unresolved → open an issue.
