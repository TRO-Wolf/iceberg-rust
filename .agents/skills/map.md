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

# map.md — .agents/skills/

## Purpose

Agent-facing **skills**: the portable working method plus step-by-step sequences for recurring
judgement and maintenance work, written for any tool's agent. Each skill is a directory holding a `SKILL.md` with YAML frontmatter (`name`,
`version`, and a `description` that says when to reach for it **and when not to**) — the same shape
as [sepmo/SKILL.md](sepmo/SKILL.md) — so a skill is discoverable and invocable
rather than a file an agent has to already know to open.

The procedural skills record proven sequences and define no policy. SEPMO is the explicit exception:
it is the versioned lifecycle control plane and project binding named at precedence level 5 in
[AGENTS.md](../../AGENTS.md). The engineering contract still wins every conflict.

**Claude discovers these through a symlink.** [../../.claude/skills](../../.claude/map.md) points at
this directory (git mode `120000`), because Claude Code loads skills only from `.claude/skills/`.
The skills keep their single home here; adding a directory below makes it invocable with no change
on the Claude side.

**The license gate treats these files specially.** The rule, and what a new `SKILL.md` owes, is
stated once in [../../AGENTS.md](../../AGENTS.md); this file does not restate it. Every other `.md`
here (this file, each skill's `map.md`) carries the ASF header normally.

## Contents

| Skill | For |
|---|---|
| [engineering-method/](engineering-method/map.md) | The portable, agent-agnostic working method for implementation and review sessions: risk-first design, the reason-plan-verify workflow, naming, the Rust defaults, the debugging protocol, the done gate, and the frontier operating notes. Generalized 2026-08-24 from the former per-model-tier manuals (`skills/{Fable,Opus,Sonnet,Haiku}.md`); tier postures moved to the tool adapters. Rule of record stays [AGENTS.md](../../AGENTS.md). |
| [compaction/](compaction/map.md) | The lessons-compaction procedure for [task/lessons.md](../../task/lessons.md) (and the todo-archival analogue): lifecycle (PROMOTE / KEEP / ARCHIVE), triggers, archive layout, conservation gate. Its own scoped change, interactive-approval-only. Moved 2026-08-24 from `skills/compaction.md`. |
| [rust-code-quality/](rust-code-quality/map.md) | The Rust review procedure for what the gates cannot catch: on-disk format stability, divergence from Java `iceberg-core`, panics and value-path casts (neither is armed in this workspace), broken error chains, lock discipline, unbounded recursion. Severity ordered for a table-format library — silent corruption of already-written tables outranks everything. Also carries the authoring pass for [AGENTS.md](../../AGENTS.md) "Comments and prose". That pass is why AGENTS.md read order loads this skill for every Actor and every Critic, not for review alone. |
| [test-adequacy/](test-adequacy/map.md) | The evidence discipline behind a coverage claim: apply the mutation rather than predict it, one knob at a time, and state the population next to every number. Carries the green-vacuity patterns that have each produced a false green here, the sabotage hard-fail rule, and the chained-gate rule. |
| [sepmo/](sepmo/map.md) | The SEPMO governance-and-orchestration control plane: scope audit, Actor–Critic execution, PR grouping, delivery, retrospective, and the binding manifest that maps its abstract roles to this repo. |

## I want to...

| I want to... | go to |
|---|---|
| Read the working method before writing or reviewing code | [engineering-method/SKILL.md](engineering-method/SKILL.md) |
| Compact the lessons file / find an archived lesson | [compaction/SKILL.md](compaction/SKILL.md) |
| Run the SEPMO lifecycle or Actor–Critic protocol | [sepmo/SKILL.md](sepmo/SKILL.md) |
| Review a Rust PR or commit | [rust-code-quality/SKILL.md](rust-code-quality/SKILL.md) "Quick start" |
| Know how much to comment, and in what English | the rule is [AGENTS.md](../../AGENTS.md) "Comments and prose"; the sequence is [rust-code-quality/SKILL.md](rust-code-quality/SKILL.md) "Comment discipline — the authoring pass" |
| Decide where Java bytecode evidence belongs | [AGENTS.md](../../AGENTS.md) "Comments and prose" — the ledger under [task/](../../task/), never a doc comment |
| Refute (not accept) a coverage claim | [test-adequacy/SKILL.md](test-adequacy/SKILL.md) "Apply, never predict" |
| Know what the gates already enforce | [rust-code-quality/SKILL.md](rust-code-quality/SKILL.md) "What the gates already hold" |
| Turn a checklist item into a real gate | [rust-code-quality/SKILL.md](rust-code-quality/SKILL.md) "Arming candidates" |
| Add a new skill | a `<verb-noun>/` directory here with `SKILL.md` (frontmatter + pointers, no policy) and its own `map.md`, plus a Contents row here and a `paths-ignore` entry in [.licenserc.yaml](../../.licenserc.yaml) |
| Read the authoritative contract | [../../AGENTS.md](../../AGENTS.md) |

## Pointers

- **Up:** [../map.md](../map.md)
- **Related:** [../../AGENTS.md](../../AGENTS.md) (every rule these skills cite),
  [../../docs/testing.md](../../docs/testing.md) (the testing-discipline contract).

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| A procedural skill states a project rule | Bug — move the rule to [../../AGENTS.md](../../AGENTS.md) and leave a pointer. SEPMO owns lifecycle only. |
| A checklist item duplicates an armed gate | Bug — delete the item and cite the gate; a duplicated item is how a real finding gets skimmed past |
| A skill will not load in a Claude session | `ls -l ../../.claude/skills` must resolve here, and the `SKILL.md` must carry `name` + `description` frontmatter |
| CI's license-header check reds on a new `SKILL.md` | Its `paths-ignore` entry is missing from [.licenserc.yaml](../../.licenserc.yaml) |
| A skill step no longer matches reality | Fix the skill in the same PR as the change that falsified it |

### First checks

- Did you read [../../AGENTS.md](../../AGENTS.md) first? It defines the contract and SEPMO's lifecycle-only precedence.

### Escalate to

- Conflicts / precedence → [../../AGENTS.md](../../AGENTS.md) `<precedence>`.
- Lifecycle and the Actor–Critic protocol → [sepmo/SKILL.md](sepmo/SKILL.md).
