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

# map.md — .agents/skills/rust-code-quality/

## Purpose

The **Rust authoring and review procedure**. The authoring pass enacts
[../../../AGENTS.md](../../../AGENTS.md) "Comments and prose" while you write, before any diff
exists. That is why AGENTS.md read order loads this skill for every Actor, not for review alone.
The review pass is what to check in a Rust diff that the armed gates (`make check` —
fmt, clippy `-D warnings`, taplo, cargo-machete, the agent-artifact and matrix-anchor scripts — plus
`typos`) cannot catch. In this workspace that residue is unusually large: there is no `clippy.toml`
and no `[workspace.lints]`, so panics and truncating `as` casts are held by **review only**, and the
parity axis (divergence from Java `iceberg-core` 1.10.0) has no mechanical form at all.

It records *sequences*, not a second contract: every rule it leans on points into
[../../../AGENTS.md](../../../AGENTS.md), [../../../docs/testing.md](../../../docs/testing.md), or
the engineering method in [../engineering-method/SKILL.md](../engineering-method/SKILL.md), and on
any conflict those win.

## Contents

- [SKILL.md](SKILL.md) — the comment-discipline authoring pass, the gate inventory (what never to
  re-review, and what is deliberately *not* gated here), the candidate scans, the review checklist,
  the shortest-form worked example, the severity scale, the report template, and the arming
  candidates.

## I want to...

| I want to... | go to |
|---|---|
| Comment discipline while writing | [SKILL.md](SKILL.md) "Comment discipline — the authoring pass"; the rule is [../../../AGENTS.md](../../../AGENTS.md) "Comments and prose" |
| Review a Rust PR or commit | [SKILL.md](SKILL.md) "Quick start" |
| Know what the gates already enforce | [SKILL.md](SKILL.md) "What the gates already hold" |
| Rank a finding | [SKILL.md](SKILL.md) "Severity" — silent format/data corruption outranks everything |
| Read the rule of record | [../../../AGENTS.md](../../../AGENTS.md) "Rust conventions" + "Absolute prohibitions" |
| Turn a checklist item into a gate | [SKILL.md](SKILL.md) "Arming candidates" |
| Judge whether the tests prove anything | [../test-adequacy/SKILL.md](../test-adequacy/SKILL.md) |

## Pointers

- **Up:** [../map.md](../map.md)
- **Related:** [../engineering-method/SKILL.md](../engineering-method/SKILL.md) (the working
  method whose Rust sections this skill cites rather than restates); [../../../docs/parity/GAP_MATRIX.md](../../../docs/parity/GAP_MATRIX.md)
  (where a named parity residue is registered, cited by `row R<id>`).

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| A checklist item duplicates a gate | Bug — delete the item, cite the gate ([SKILL.md](SKILL.md) "What the gates already hold") |
| The skill states a project rule | Bug — move it to [../../../AGENTS.md](../../../AGENTS.md), leave a pointer |
| A cited file, `make` target, or lint no longer exists | Fix the skill in the same PR as the change that falsified it |
| A review passes a diff that later corrupts a table | The format-stability checklist was skipped, or a `spec/` change was reviewed as local — it never is |

### First checks

- Was the Java 1.10.0 reference actually read for every behavior change, and named in the report?

### Escalate to

- Precedence / a rule that seems wrong → [../../../AGENTS.md](../../../AGENTS.md) `<precedence>`.
