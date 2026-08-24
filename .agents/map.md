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

# map.md — .agents/

## Purpose

Tool-neutral and per-tool **adapter** entry points for automated agents. Every file here carries
**zero authoritative facts** — each is a thin pointer into the authoritative spine
([AGENTS.md](../AGENTS.md), with [Roadmap.md](../Roadmap.md) for the plan and
[docs/parity/GAP_MATRIX.md](../docs/parity/GAP_MATRIX.md) for capability status). Adapters cannot
drift, and deleting any one loses no project knowledge. A rule stated in an adapter is a bug: move
it to [AGENTS.md](../AGENTS.md) and leave a pointer.

## Contents

| File | For |
|---|---|
| `common.md` | The shared, tool-neutral start: read AGENTS.md first, then the spine. No rules. |
| `claude.md` | Points Claude sessions at [../CLAUDE.md](../CLAUDE.md) and the portable working method in [skills/engineering-method/](skills/engineering-method/map.md). |
| `codex.md`, `cursor.md` | One-line stubs pointing inward; no tool mechanics recorded yet. |
| `skills/` | Agent-facing skills — the portable engineering method (formerly the per-tier manuals), the lessons-compaction procedure, the Rust code-quality review, and the test-adequacy evidence procedure. Each is a directory with a frontmatter-carrying `SKILL.md`, so it is discoverable rather than merely present, and `../.claude/skills` symlinks here so Claude sessions can invoke them natively. See [skills/map.md](skills/map.md). |

## I want to...

| I want to... | go to |
|---|---|
| Onboard any agent, tool-agnostic | [common.md](common.md) → [../AGENTS.md](../AGENTS.md) |
| Onboard a Claude session | [claude.md](claude.md) → [../CLAUDE.md](../CLAUDE.md) |
| Review a Rust PR or commit | [skills/rust-code-quality/SKILL.md](skills/rust-code-quality/SKILL.md) |
| Judge whether a test actually proves anything | [skills/test-adequacy/SKILL.md](skills/test-adequacy/SKILL.md) |
| Add mechanics for a new tool | add `.agents/<tool>.md` (pointer + tool mechanics only) + a Contents row here |
| Read the authoritative contract | [../AGENTS.md](../AGENTS.md) |

## Pointers

- **Up:** [../AGENTS.md](../AGENTS.md) (there is no repo-root `map.md`; the spine is the top).
- **Related:** [../CLAUDE.md](../CLAUDE.md) (the Claude adapter), [../.claude/map.md](../.claude/map.md)
  (Claude's discovery directory), [../skills/map.md](../skills/map.md) (the SEPMO control plane —
  a separate tree, not part of this one).

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| An adapter states a project rule | Bug — move the rule to [../AGENTS.md](../AGENTS.md) and leave a pointer here |
| An agent starts in the wrong place | Every adapter must route to AGENTS.md first (`common.md` is the shared head) |
| A skill will not load in a Claude session | `ls -l ../.claude/skills` must resolve here, and the skill's `SKILL.md` must carry `name` + `description` frontmatter |
| A new `SKILL.md` fails the ASF license gate | The rule and its remedy live in [../AGENTS.md](../AGENTS.md) (license headers on Markdown) — apply it there, not from here |

### First checks

- Did you read [../AGENTS.md](../AGENTS.md) first? It sets precedence, prohibitions, and the
  `map.md` rule.

### Escalate to

- Conflicts / precedence → [../AGENTS.md](../AGENTS.md) `<precedence>`.
- Unresolved → open an issue.
