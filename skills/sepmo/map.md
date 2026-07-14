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
| `SKILL.md` | The spine — **versioned canon, v2.2**: the Iron State Machine (T1–T12), the proposition-ledger gate, the AC sub-machine rules R1–R10, the S0–S3 severity scale, the LIGHT/STANDARD rubric, doctrines, roster, changelog. Never edited per-project. |
| `binding-manifest.md` | The one project-specific file: role bindings + tunables (`spine_version`, `severity_floor`, `green_commands` two-tier + CI-exception record, `context_break_mechanics`, `s0_fresh_execution`, `metrics_ledger_location`, `taxonomy_extensions`). **Re-instantiate this to port SEPMO.** |
| `binding-manifest.template.md` | The portable template the manifest instantiates from — ships with the canon; edit neither per-project. |
| `references/01-scope-auditor.md` | Scope Auditor — the proposition ledger (PROVEN/OPEN/REJECTED), the enumeration obligation for quantified clauses, the approval gate |
| `references/02-orchestrator.md` | Orchestrator — charter ← plan-of-record, PR carving + rubric, context-break enforcement (R3), AC-loop coordination, R7 readiness checklist |
| `references/03-self-logic-review.md` | The pre-action review every agent runs; also the one-time PRE_EXECUTION_REVIEW format |
| `references/04-actor.md` | Actor — R2 green exit + clause pinning (per-element for quantified clauses); R5/R6 dispositions; blind to the Critic |
| `references/05-critic.md` | Critic — the attack taxonomy + coverage attestation (R4), the span check, the fresh-execution step; convergence is its call |
| `references/06-vigilance.md` | Invariant V — watch items (incl. unledgered claims, silent gate skips), the alarm, T8 |
| `references/07-delivery.md` | Per-PR acceptance = the §4 Done gate + R8 embedded evidence + GAP_MATRIX flip discipline |
| `references/08-retrospective.md` | Learnings + the quantitative metrics ledger (incl. `environment_drift_events`), incident retrospectives, asymmetric feed-forward, compaction pass |

## I want to...

| I want to... | go to |
|---|---|
| Understand the lifecycle / where I am in it | [SKILL.md](SKILL.md) — the Iron State Machine |
| See how SEPMO binds to this repo (incl. gates, floor, taxonomy) | [binding-manifest.md](binding-manifest.md) |
| Port SEPMO to another repo | instantiate [binding-manifest.template.md](binding-manifest.template.md) there |
| Audit a new brief / plan before any code | [references/01-scope-auditor.md](references/01-scope-auditor.md) |
| Run the pre-action review | [references/03-self-logic-review.md](references/03-self-logic-review.md) |
| File retrospective metrics / an incident retrospective | [references/08-retrospective.md](references/08-retrospective.md) → [task/sepmo-metrics.md](../../task/sepmo-metrics.md) |
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
| Unsure whether a change needs the full pipeline | Proportionality: substantial PR-units get full ceremony; trivial changes take the lightweight path ([SKILL.md](SKILL.md)). |

### First checks

- Did you read root [CLAUDE.md](../../CLAUDE.md) and your tier manual before this spine?

### Escalate to

- Conflicts / precedence → [CLAUDE.md](../../CLAUDE.md) `<precedence>`.
- Unresolved → open an issue.
