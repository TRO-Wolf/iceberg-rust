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

# map.md — .agents/skills/engineering-method/

## Purpose

The portable, agent-agnostic working method for implementation and review sessions: risk-first
design, the reason-plan-verify workflow, naming, the Rust defaults, the debugging protocol, and
the done gate. It generalizes the former per-model-tier manuals
(`skills/{Fable,Opus,Sonnet,Haiku}.md`, removed 2026-08-24) into one instruction set any tool's
agent reads; tier postures and model-specific mechanics moved to the tool adapters. The rule of
record for every project fact stays [../../../AGENTS.md](../../../AGENTS.md); this skill records
the method and loses on any conflict.

## Contents

| File | For |
|---|---|
| `SKILL.md` | The method: Identity & Priority Stack → Non-Negotiables → Frontier Operating Notes → Mode Handling → Risk-First → Workflow §1–§9 → Navigation (`map.md`) → Naming → Language-Specific Rules → Function Length & Recursion → Pre-Flight → Core Principles (TL;DR). |

## I want to...

| I want to... | go to |
|---|---|
| Read the full working method for a session | [SKILL.md](SKILL.md) |
| Find the must-not-violate list | [SKILL.md](SKILL.md) `<non_negotiables>` (each row points at its spine home) |
| Run the pre-implementation risk pass | [SKILL.md](SKILL.md) `<risk_first>` |
| Check whether a task is done | [SKILL.md](SKILL.md) `<verification_gate>` (the §4 Done gate) |
| Operate a frontier-class session (cost, calibration, degraded-capability honesty) | [SKILL.md](SKILL.md) "Frontier Operating Notes" |
| Debug a failure methodically | [SKILL.md](SKILL.md) §8 + the touched directory's `map.md#debug` |
| Write or review Rust under the repo conventions | [../rust-code-quality/SKILL.md](../rust-code-quality/SKILL.md) — rule of record [../../../AGENTS.md](../../../AGENTS.md) "Rust conventions" |
| Brief a delegated agent on a capability tier | the running tool's adapter ([../../../CLAUDE.md](../../../CLAUDE.md) / [../../map.md](../../map.md)) — tier postures are tool mechanics, not method |

## Pointers

- **Up:** [../map.md](../map.md)
- **Related:** [../../../AGENTS.md](../../../AGENTS.md) (the authoritative contract this skill
  serves); [../../../docs/testing.md](../../../docs/testing.md) (the testing contract §4 binds);
  [../rust-code-quality/SKILL.md](../rust-code-quality/SKILL.md) +
  [../test-adequacy/SKILL.md](../test-adequacy/SKILL.md) (the review instruments);
  [../compaction/SKILL.md](../compaction/SKILL.md) (the lessons/todo compaction procedure §2
  points at).

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| A rule here contradicts the repo | [../../../AGENTS.md](../../../AGENTS.md) overrides this skill — re-check AGENTS.md (its `<precedence>` wins) |
| Looking for tier-specific briefing content | It moved to the tool adapters ([../../../CLAUDE.md](../../../CLAUDE.md) for Claude tiers) — this skill is one method for every tier |
| A link here doesn't resolve | The skill expects to live at `.agents/skills/engineering-method/` with the repo spine three levels up |

### Escalate to

- [../map.md](../map.md) → [../../../AGENTS.md](../../../AGENTS.md).
