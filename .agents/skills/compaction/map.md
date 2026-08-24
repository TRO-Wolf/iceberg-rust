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

# map.md — .agents/skills/compaction/

## Purpose

The lessons-compaction procedure for [../../../task/lessons.md](../../../task/lessons.md) and its
todo-file analogue: lifecycle (PROMOTE / KEEP / ARCHIVE), triggers, archive layout, and the
conservation gate. Moved here 2026-08-24 from `skills/compaction.md` so it is discoverable and
invocable like the other runbook skills; the procedure is unchanged.

## Contents

| File | For |
|---|---|
| `SKILL.md` | The procedure: the lifecycle model, the promotion table, triggers, the pass protocol, the conservation gate, todo archival. |

## I want to...

| I want to... | go to |
|---|---|
| Run (or propose) a compaction pass | [SKILL.md](SKILL.md) — interactive-approval-only, its own scoped change |
| Find where an old archived lesson went | [../../../task/lessons-archive/map.md](../../../task/lessons-archive/map.md) |
| See what triggers a pass | [SKILL.md](SKILL.md) "Triggers" |
| Archive completed todo narratives | [SKILL.md](SKILL.md) "§Todo Archival" |

## Pointers

- **Up:** [../map.md](../map.md)
- **Related:** [../../../task/lessons.md](../../../task/lessons.md) +
  [../../../task/todo.md](../../../task/todo.md) (the files this skill maintains);
  [../engineering-method/SKILL.md](../engineering-method/SKILL.md) §2 (the loop that grows them).

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| A needed lesson seems missing from `task/lessons.md` | Promoted or archived — check the compaction header at the top of the file, then [../../../task/lessons-archive/map.md](../../../task/lessons-archive/map.md) |
| Session-start lessons read is consuming excessive context | A trigger has likely fired — propose a pass (do not run one mid-increment) |

### Escalate to

- [../map.md](../map.md) → [../../../AGENTS.md](../../../AGENTS.md).
