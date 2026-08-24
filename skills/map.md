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

# map.md — skills/

## Purpose

Portable, tool-agnostic **agent control planes** installed into this repo. These are operating
layers an agent runs *under* — they bind to, and defer to, the repo's engineering contract
([../AGENTS.md](../AGENTS.md)).

The per-model-tier operating manuals that used to live here (`Fable.md` / `Opus.md` /
`Sonnet.md` / `Haiku.md`) were generalized on 2026-08-24 into the agent-agnostic
**engineering-method** skill at
[../.agents/skills/engineering-method/SKILL.md](../.agents/skills/engineering-method/SKILL.md);
tier postures and model-specific mechanics (the `OO` mapping, Fable pricing and fallback notes)
moved to the Claude adapter, [../CLAUDE.md](../CLAUDE.md). The lessons-compaction procedure
(`compaction.md`) moved to [../.agents/skills/compaction/SKILL.md](../.agents/skills/compaction/SKILL.md)
the same day. This directory now holds only the SEPMO control plane.

## Contents

| Entry | For |
|---|---|
| `sepmo/` | The **SEPMO** governance-and-orchestration control plane: lifecycle state machine, scope audit + 100% ledger gate, adversarial Actor–Critic protocol, PR-grouping, delivery, retrospective — and its binding manifest (the one file mapping SEPMO's abstract roles to this repo). Governs lifecycle; binds to AGENTS.md + the engineering-method skill for the engineering contract. See [sepmo/map.md](sepmo/map.md). |

## I want to...

| I want to... | go to |
|---|---|
| Operate under / understand SEPMO | [sepmo/map.md](sepmo/map.md) → [sepmo/SKILL.md](sepmo/SKILL.md) |
| See SEPMO's bindings to this repo | [sepmo/binding-manifest.md](sepmo/binding-manifest.md) |
| Read the portable engineering method (formerly the tier manuals) | [../.agents/skills/engineering-method/SKILL.md](../.agents/skills/engineering-method/SKILL.md) |
| Compact the lessons file / find an archived lesson | [../.agents/skills/compaction/SKILL.md](../.agents/skills/compaction/SKILL.md) |
| Know the sub-agent / parallelism policy | [../AGENTS.md](../AGENTS.md) `<subagent_policy>` (neutral rule); the Claude tier mapping is in [../CLAUDE.md](../CLAUDE.md) |
| Resolve a conflict with the repo contract | [../AGENTS.md](../AGENTS.md) (it wins) |

## Pointers

- **Up:** repo root [../AGENTS.md](../AGENTS.md).
- **Related:** [../.agents/skills/map.md](../.agents/skills/map.md) (the agent-facing runbook
  skills, symlinked into `.claude/skills`); [../docs/testing.md](../docs/testing.md)
  (the verification gate); [../task/todo.md](../task/todo.md) +
  [../task/lessons.md](../task/lessons.md) (plan + lessons + archives).

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| A link to `skills/Opus.md` (or another tier manual) doesn't resolve | The manuals were generalized 2026-08-24 into [../.agents/skills/engineering-method/SKILL.md](../.agents/skills/engineering-method/SKILL.md) — retarget the link |
| A link to `skills/compaction.md` doesn't resolve | Moved the same day to [../.agents/skills/compaction/SKILL.md](../.agents/skills/compaction/SKILL.md) |
| A SEPMO behavior contradicts the engineering contract | The contract wins — see [sepmo/binding-manifest.md](sepmo/binding-manifest.md) Precedence |

### First checks

- Start at [sepmo/map.md](sepmo/map.md); each subdirectory carries its own `map.md`.

### Escalate to

- [../AGENTS.md](../AGENTS.md) `<precedence>`; unresolved → open an issue.
