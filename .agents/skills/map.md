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

Agent-facing **review procedures**: step-by-step sequences for recurring judgement work, written for
any tool's agent. Each skill is a directory holding a `SKILL.md` with YAML frontmatter (`name`,
`version`, and a `description` that says when to reach for it **and when not to**) — the same shape
as [skills/sepmo/SKILL.md](../../skills/sepmo/SKILL.md) — so a skill is discoverable and invocable
rather than a file an agent has to already know to open.

A skill records a proven *sequence*; it defines no policy and carries no authoritative project fact.
Every rule it leans on is a pointer into [AGENTS.md](../../AGENTS.md) or the doc each step cites,
and on any conflict those win. That keeps the `.agents/` zero-authoritative-facts contract intact:
deleting a skill loses a convenience, never a project truth.

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
| [rust-code-quality/](rust-code-quality/map.md) | The Rust review procedure for what the gates cannot catch: on-disk format stability, divergence from Java `iceberg-core`, panics and value-path casts (neither is armed in this workspace), broken error chains, lock discipline, unbounded recursion. Severity ordered for a table-format library — silent corruption of already-written tables outranks everything. Also carries the authoring pass for [AGENTS.md](../../AGENTS.md) "Comments and prose". That pass is why AGENTS.md read order loads this skill for every Actor and every Critic, not for review alone. |
| [test-adequacy/](test-adequacy/map.md) | The evidence discipline behind a coverage claim: apply the mutation rather than predict it, one knob at a time, and state the population next to every number. Carries the green-vacuity patterns that have each produced a false green here, the sabotage hard-fail rule, and the chained-gate rule. |

## I want to...

| I want to... | go to |
|---|---|
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
  [../../docs/testing.md](../../docs/testing.md) (the testing-discipline contract),
  [../../skills/map.md](../../skills/map.md) (the tier manuals + SEPMO — a separate tree, deliberately
  **not** covered by the `.claude/skills` symlink).

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| A skill states a project rule | Bug — move the rule to [../../AGENTS.md](../../AGENTS.md), leave a pointer (`.agents/` contract) |
| A checklist item duplicates an armed gate | Bug — delete the item and cite the gate; a duplicated item is how a real finding gets skimmed past |
| A skill will not load in a Claude session | `ls -l ../../.claude/skills` must resolve here, and the `SKILL.md` must carry `name` + `description` frontmatter |
| CI's license-header check reds on a new `SKILL.md` | Its `paths-ignore` entry is missing from [.licenserc.yaml](../../.licenserc.yaml) |
| A skill step no longer matches reality | Fix the skill in the same PR as the change that falsified it |

### First checks

- Did you read [../../AGENTS.md](../../AGENTS.md) first? A skill is a sequence, never a contract.

### Escalate to

- Conflicts / precedence → [../../AGENTS.md](../../AGENTS.md) `<precedence>`.
- Lifecycle and the Actor–Critic protocol → [../../skills/sepmo/SKILL.md](../../skills/sepmo/SKILL.md).
