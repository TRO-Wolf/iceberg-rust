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

# map.md — .agents/skills/test-adequacy/

## Purpose

The **evidence discipline behind a coverage claim**. A green suite proves that nothing observed
broke; it does not prove that anything is observable. This skill is how that gap is closed: apply
the mutation instead of predicting it, one knob at a time, and state the population next to every
number.

It also carries the green-vacuity patterns that have each produced a real false green in this
repository — identity-by-construction preconditions, read-identity-only post-checks, fixture
self-assertions, and the three-valued-logic `=` trap — plus the sabotage hard-fail rule and the
chained-gate rule.

It is a review *sequence*, not a second contract: the rules live in
[../../../AGENTS.md](../../../AGENTS.md) ("Parity mandate", "Crate code — testing", "Working
conventions") and [../../../docs/testing.md](../../../docs/testing.md), and on any conflict those
win.

## Contents

- [SKILL.md](SKILL.md) — apply-never-predict, the one-knob-at-a-time rule, the four vacuity
  patterns with their tells and repairs, the sabotage and gate-chain rules, the review checklist,
  and the report template.

## I want to...

| I want to... | go to |
|---|---|
| Refute (not accept) a coverage claim | [SKILL.md](SKILL.md) "Apply, never predict" |
| Check a null-handling test is not vacuous | [SKILL.md](SKILL.md) "The three-valued-logic `=` trap" |
| Write a sabotage / negative test | [SKILL.md](SKILL.md) "A sabotage step that cannot be applied must HARD-FAIL" |
| Report an assertion that cannot be killed | [SKILL.md](SKILL.md) "An unkillable assertion, honestly recorded" |
| Read the rule of record | [../../../AGENTS.md](../../../AGENTS.md) + [../../../docs/testing.md](../../../docs/testing.md) |
| Judge whether the code is right | [../rust-code-quality/SKILL.md](../rust-code-quality/SKILL.md) |

## Pointers

- **Up:** [../map.md](../map.md)
- **Related:** [../sepmo/references/05-critic.md](../sepmo/references/05-critic.md)
  (the Critic stage that must apply these mutations rather than read someone else's arithmetic);
  [../../../AGENTS.md](../../../AGENTS.md) `<subagent_policy>` (why the Critic is independent).

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| A coverage claim has no population next to its number | The mutation was predicted, not applied — send it back |
| N knobs removed, N failures reported | Batch mutation; the count dissolves under one-at-a-time removal (unit G5) |
| A null-guard test passes with the guard deleted | The `=` trap — `= NULL` is UNKNOWN either way; re-probe with `<>` / `NOT` |
| A sabotage script logs "skipped" and exits green | A SKIP branch where a hard failure is owed — the corruption never happened |
| A "fixed" bug reappears with the regression test still green | The assertion is read-identity or fixture self-assertion; assert shape instead |

### First checks

- Was the mutation actually run, and was the tree restored and re-run to the baseline afterwards?

### Escalate to

- The testing contract → [../../../docs/testing.md](../../../docs/testing.md).
- Precedence / a rule that seems wrong → [../../../AGENTS.md](../../../AGENTS.md) `<precedence>`.
