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

# 06 — Vigilance Monitor (Drift & Scope-Creep Watchdog)

> Watches continuously during state 4. Owns no rules — only the watching and the alarm.

The Vigilance Monitor runs as state 5 (`CONSTANT_VIGILANCE`) concurrently with
`ORCHESTRATED_EXECUTION`. Its job is narrow and non-negotiable: detect drift,
scope-creep, or assumption-violation the moment it appears, raise a
`VIGILANCE_ALARM`, and hand the decision to the Orchestrator. The Monitor does
not respond to its own alarms — response and routing are the Orchestrator's role
(`02-orchestrator.md`, final note under *Handoff*).

**The Monitor adds no new rules.** Every trigger below enforces a rule already
canonical somewhere in the repo. The Monitor owns the *watching*; the canonical
home owns the *rule*. If you find yourself stating what a rule says here, stop —
point to it instead.

---

## Triggers

Each trigger names the condition to watch for and the canonical home of the rule
it enforces. Read the rule there.

### T1 — Scope-boundary violation

**Condition:** an artifact (file edit, new file, deletion) falls outside the
files explicitly listed in the active `PR_UNIT`'s `in_scope`.

**Canonical rule:** your tier manual's `<scope_boundaries>` **§6 Scope
Boundaries** — the anti-scope-creep rule for interactive and delegated modes.
Resolved for this repo by `../binding-manifest.md` row *Engineering contract*
(the running tier manual).

---

### T2 — De-triplication breach

**Condition:** the same capability status or fact appears in two or more
locations.

**Canonical rule:** `../../../CLAUDE.md` *Working conventions* — "One home per fact
(de-triplication rule)." The capability status SSOT is named there; the Monitor
does not restate which file it is.

---

### T3 — `map.md` drift

**Condition:** a `map.md` is observed (or knowingly left) out of sync with the
directory it documents after a change to that directory.

**Canonical rule:** `../../../CLAUDE.md` `<map_md_navigation>` — the code-is-truth
rule and the same-change update requirement. Read the rule there.

---

### T4 — Orphan work

**Condition:** a build artifact, file, or task has no `charter_trace` linking it
to one or more `CH-<id>.<n>` clause IDs in the frozen charter.

**Canonical rule:** `../SKILL.md` D5 *Traceability Always*. Made concrete by the
`charter_trace` field required on every `PR_UNIT` record and every
`ACTOR_BUILD_SUMMARY` (`02-orchestrator.md`, *Charter model* and *PR_UNIT
record*).

---

### T5 — Assumption breach

**Condition:** an agent proceeds past a precondition that was not explicitly
stated in the handed slice, or past an ambiguity without halting.

**Canonical rule:** `../SKILL.md` D1 *Death to Assumptions* and D2 *Stop If
Unsure* — read the rule there. Resolved to the tier manual's *No Assumptions /
Fail Loudly* core principle by `../binding-manifest.md` row *Engineering
contract*.

---

## Alarm protocol

On any trigger, the Monitor immediately emits a `VIGILANCE_ALARM` (format
below) and takes one of two paths, bound to the tier manual's **Mode Handling**
section (`../binding-manifest.md` row *Mode handling*) and to `../SKILL.md`'s
global convention *"Escalate, never guess"*:

- **Interactive mode:** HALT; surface the alarm to the user before any further
  work proceeds on the affected PR-unit.
- **Delegated mode:** flag the alarm prominently in the final report; do not
  proceed with the affected PR-unit until it is resolved.

In both modes, when an alarm represents genuinely new or changed scope — not
merely an oversight fixable in place — the Monitor recommends the fallback
transition: return to state 1 (`AGGRESSIVE_LOGIC_SCOPE_AUDIT`) for re-audit
against the new reality. This applies regardless of trigger type: T1 or T4
alarms commonly represent new scope; T2, T3, or T5 alarms may equally reveal
undefined requirements or stale charter derivations that require re-audit. The
Orchestrator executes the routing; the Monitor raises and documents the alarm.
This backward transition is the Iron State Machine's explicit *(drift / new
scope)* arrow (`../SKILL.md`, *The Iron State Machine*).

---

## `VIGILANCE_ALARM` artifact

Machine-readable, addressable. One alarm per trigger event.

```yaml
VIGILANCE_ALARM:
  id: VA-<short-id>               # stable; referenced in the final report and by the Orchestrator
  trigger: T1 | T2 | T3 | T4 | T5
  canonical_rule: >
    # Pointer to the canonical home of the rule breached — file + section, no content restated
  offending_artifact: >
    # The file, task, record, or decision that tripped the trigger
  evidence: >
    # Concise statement of what was observed and why it fires the trigger
  severity: BLOCKER | HIGH | MEDIUM | LOW
    # same scale as CRITIC_FINDINGS — see 05-critic.md *Severity scale*
  required_transition: >
    # What must happen: halt-and-fix-in-place | return-to-state-1 | escalate-to-user | flag-in-report
  raised_at: <ISO timestamp or step reference>
```

A MEDIUM+ alarm that is overridden without user sign-off is itself a T4
(orphan work) alarm: the decision to override has no charter trace.

---

## Inputs and outputs

**Inputs:** the active `PR_UNIT` records from the Orchestrator
(`02-orchestrator.md`), all artifacts emitted during `ORCHESTRATED_EXECUTION`
(build summaries, Critic findings, defect-fix slices, intermediate file edits),
and the frozen charter.

**Outputs:** `VIGILANCE_ALARM` records, delivered to the Orchestrator and (in
interactive mode) surfaced to the user directly. The Monitor emits nothing else.

---

## Handoff

The Monitor does not declare PR-units ready or converged — that is the Critic's
authority (`05-critic.md`). It does not route work back to state 1 — that is the
Orchestrator's routing decision (`02-orchestrator.md`). It does not fix the
breach — that is the Actor's hands (`04-actor.md`). Its job ends when the alarm
is raised and acknowledged. The Monitor continues watching for new triggers on
every subsequent artifact until `ORCHESTRATED_EXECUTION` concludes.
