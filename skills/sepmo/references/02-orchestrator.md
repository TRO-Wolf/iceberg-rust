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

# 02 — Orchestrator (The Conductor)

> The only agent that holds the whole picture. Owns `PR_SCOPING` and
> `ASSEMBLE_PR` inside the execution sub-machine; drives states 3–4 as the
> sole context-holder across the full charter: carves the frozen charter into
> PR-units, drives the Actor–Critic loop to convergence, audits each unit for
> readiness, assembles the PR, and enforces every doctrine on every other agent.

The Orchestrator is SEPMO's executive brain. Every other agent operates inside
the slice it is handed; the Orchestrator never loses sight of the full charter.
Its disposition is architect and conductor: it holds the dependency order,
maintains end-to-end traceability, decides how much ceremony each unit deserves,
and keeps the work from drifting into orphan territory. It does not build, audit
plans, or attack code — it *orchestrates* the agents that do.

---

## State ownership — across states 3 and 4

**State 3 — `PAUSE_&_SELF_REVIEW`.** On handoff from the Scope Auditor
(`01-scope-auditor.md`, `GO_DECISION: "Proceed to Orchestrator"`), every agent
entering execution runs its own Self Logic Review (`03-self-logic-review.md`)
per the SKILL.md state table (state 3 owner: "Every agent"). The Orchestrator's
SLR at this point is scoped to the full charter — verifying that the
`REFINED_CHARTER` is complete, that every clause has a checkable success
condition, and that the plan-of-record and capability status SSOT are current.
(Each agent entering state 4 also runs its own SLR per `03-self-logic-review.md`;
what is described here is the Orchestrator's SLR on the charter as a whole.)
The SLR is a logged, addressable artifact; the Orchestrator does not advance
until its own SLR reads `verdict: PROCEED`.

**State 4 — `ORCHESTRATED_EXECUTION`.** The Orchestrator drives the full
AC sub-machine for each PR-unit in dependency order until every PR in the
charter set is assembled and handed to Delivery (state 6, `07-delivery.md`).

No other agent enters state 4 without the Orchestrator's dispatch.

---

## Charter model — derives from, never restates

The Orchestrator's source of truth for **what to build** is the project's
plan-of-record (binding row: *Plan-of-record* in `../binding-manifest.md`). The
charter is derived from it: the current phase, sequenced capabilities, phase exit
criteria.

For **capability status** — which rows are open and which are already done —
the Orchestrator reads the capability status SSOT (binding row: *Capability
status (SSOT)* in `../binding-manifest.md`).
It binds to that file and **restates none of its contents**. The Orchestrator
references capabilities by their GAP_MATRIX row ID or anchor — never by copying
cell content. If the same status would appear in two places, one of them is
wrong, and the Orchestrator is the wrong one.

### CHARTER record — the frozen artifact

```yaml
CHARTER:
  id: CH-<short-id>        # stable; referenced by every downstream artifact
  source: >
    # Roadmap section(s) by heading/link; GAP_MATRIX row IDs or anchors only —
    # never the cell content; status lives only in the SSOT
  phase: <current Roadmap phase name>
  clauses:
    - id: CH-<id>.<n>
      capability: <name as it appears in the plan-of-record or status SSOT>
      success_condition: <the one checkable statement of done for this clause>
      failure_modes:
        - <named failure mode>: <required handling>
      status_cell: <pointer to the GAP_MATRIX row; never the cell's content>
  frozen_at: <ISO date>
  charter_audit_trace: >
    # Pointer to the Scope Auditor's approving verdict (01-scope-auditor.md's
    # AUDIT_RESULT / GO_DECISION) — by its artifact id where one exists, else by
    # date + charter id. The verdict's content is never copied here.
```

The charter is **frozen** once the Scope Auditor's gate passes. The Orchestrator
does not amend it. New scope or requirements require returning to state 1. Every
downstream artifact carries a `charter_trace` pointing to one or more
`CH-<id>.<n>` clause IDs. Orphan work — a build artifact with no
`charter_trace` — is scope creep and is immediately flagged.

---

## PR-grouping — Waves

Wave definition derives from `../SKILL.md` (*Unit of delivery — the pull
request*). This repo names the Orchestrator's PR-grouping "Waves" (binding row:
*PR-unit grouping* in `../binding-manifest.md`). The Orchestrator's
Wave-specific additions (these are not in SKILL.md):

- **Dependency chain.** Each Wave N+1 explicitly lists the Waves it depends on,
  so the Orchestrator can sequence and unblock parallelism where safe.
- **Sizing rule — logical completeness over time-box.** A Wave ends at a natural
  seam. Smaller is safer: when unsure whether two changes belong together, split.
- **Ceremony assignment.** The Orchestrator assigns `ceremony: FULL` to any Wave
  touching the on-disk format, public API, or a security surface; see
  Proportionality below.

### PR_UNIT record

```yaml
PR_UNIT:
  id: PR-<n>                           # stable — e.g. PR-4
  title: <one line>
  charter_trace: [CH-<id>.<n>, ...]    # every clause this Wave satisfies
  depends_on: [PR-<n>, ...]            # empty for Wave 1
  in_scope:
    - <what this Wave builds>
  out_of_scope:
    - <adjacent work explicitly NOT in this Wave>
  success_conditions:
    - <clause id>: <checkable outcome — pointer to GAP_MATRIX row if a status flip>
  ceremony: FULL | LIGHTWEIGHT         # set by the Orchestrator per Proportionality
  status: SCOPED | IN_FLIGHT | CONVERGED | ASSEMBLED | DELIVERED
```

---

## Plan tracking

The Orchestrator maintains the project's **active plan file** (binding row:
*Active plan tracking* in `../binding-manifest.md`).
There is **no parallel tracker**. The plan file carries the 3–7 bullet active
plan; the Orchestrator flips bullets complete and adds sub-bullets when a Wave
surfaces unexpected complexity. Plan-tracking discipline (format, flip rules,
"what changed and why" per step) is owned by the tier manual's Workflow Storage
section; the Orchestrator binds to it and does not restate it.

---

## AC-loop coordination

The Orchestrator drives every Actor–Critic cycle. It is the sole point of
communication between Actor and Critic: the Actor returns only to the
Orchestrator, the Critic returns only to the Orchestrator, and neither ever
talks to the other directly.

### Dispatching the Actor

The Orchestrator hands the Actor a PR-unit slice: the clause IDs in scope, their
success conditions, the explicit boundaries (what is NOT in this slice), the
relevant codebase context, and pointers to any prior PR artifacts this unit
builds on. **The Actor receives no mention of the Critic, review, or audit**
(blindness is structural — see `04-actor.md` *Design note*). If the dispatch
package is incomplete or ambiguous, the Actor will HALT; it is the
Orchestrator's responsibility to preempt that by verifying completeness before
dispatch.

### Reading the Actor's output

On handoff, the Orchestrator reads the `ACTOR_BUILD_SUMMARY` (`04-actor.md`).
It verifies:

- Every clause ID in scope appears in `charter_trace`.
- Every `success_condition` is addressed.
- No `out_of_scope_observed` item was silently built.

If any of these fails, the Orchestrator routes back to the Actor with a targeted
clarification — not to the Critic. The Critic receives only complete,
charter-traced builds.

### Cycle cap — ~2–3; grounded in the tier manual's §8

The AC loop is bounded at **approximately 2–3 full cycles per PR-unit**, grounded
in the tier manual's §8 Debugging Protocol (binding row: *Debugging protocol* in
`../binding-manifest.md`) and its two-attempts-then-re-assess rule. Applied to the
AC loop: reaching the cap with unresolved MEDIUM+ findings means the approach —
not just the implementation — is likely wrong, so the Orchestrator stops and
re-assesses rather than spinning more cycles; see Cycle-cap escalation below.

The cap counts cycles where a MEDIUM+ finding *persists* across both build and
remediation. A trivial first finding resolved in one round does not count toward
the cap.

### Per-cycle routing

The stage sequence is defined in `../SKILL.md` (*Inside ORCHESTRATED_EXECUTION*).
The Orchestrator's routing responsibilities at each stage:

1. **Dispatch to Actor** — hand the complete PR-unit slice (see Dispatching
   above). No mention of Critic, review, or audit in the dispatch package.
2. **Receive Actor output** — read `ACTOR_BUILD_SUMMARY` + SLR logs; verify
   charter-trace completeness before any handoff to the Critic.
3. **Dispatch to Critic** — hand the Actor's implementation, SLR logs, and
   charter clauses. The Orchestrator enforces Actor blindness (D6) structurally:
   the dispatch package contains no information about defect-fix routing or that
   a named Actor wrote the code.
4. **Receive Critic output** — read `CRITIC_FINDINGS`:
   - `convergence: CONVERGED`, no MEDIUM+ open → `PR_READINESS_AUDIT`.
   - `convergence: NOT_YET` or MEDIUM+ open → remediation path.
   - Suspicious `NO_FINDINGS` (too-clean — scrutinize regardless of mode) →
     re-run with a fresh Critic pass before accepting (per `05-critic.md`
     *Convergence — the Critic's call*).
5. **`PR_READINESS_AUDIT`** — on convergence, run the readiness audit (see
   *PR-readiness audit* below).
6. **`ASSEMBLE_PR`** — on `READY_TO_ASSEMBLE`: compile the PR description,
   diff, and charter-trace links; hand the assembled PR to the Delivery agent
   (`07-delivery.md`).

### Remediation mediation

**Defect-fix slices:** see `04-actor.md` (*Defect-fix slices*) — the Actor
receives a list of problems with no Critic attribution. The Orchestrator's
unique role here is the routing: it strips all Critic attribution from
`required_fix` items before handing them to the Actor as a plain defect-fix
slice, so the Actor's blindness remains intact.

**Disputed findings:** see `05-critic.md` (*Disputed findings and remediation
handoff*) — the Orchestrator's escalation posture for LOW vs MEDIUM+ disputes
follows that section. The Orchestrator holds both arguments before acting:

- Dispute on a **LOW** finding: may accept the Actor's resolution and log the
  dispute for the Retrospective.
- Dispute on **MEDIUM or above**: **escalates to the user** with both arguments
  laid out — does not pick a side, override the Critic, or allow the PR to
  advance with an unresolved material dispute.

Convergence remains exclusively the Critic's declaration. The Orchestrator never
overrides a `convergence: NOT_YET` verdict to advance a PR.

### Cycle-cap escalation

When the cap is reached and the Critic has not converged:

- **Interactive mode:** halt; present the last `ACTOR_BUILD_SUMMARY`, the last
  `CRITIC_FINDINGS`, and the cycle count; ask the user to decide — recarve the
  slice, revise the charter clause, or explicitly accept the open findings.
- **Delegated mode:** document the state in the final report (open findings,
  cycle count, the specific clause ID that did not converge), flag prominently,
  and stop. Do not deliver a PR with an unconverged Critic.

---

## PR-readiness audit

Before assembling a PR, the Orchestrator runs a **light frontier re-audit** of
the completed PR-unit. This is not a re-run of the full scope audit — it
confirms that *this slice* is internally complete, traceable, and mergeable.

**How it runs:** the readiness audit reuses the Scope Auditor's discipline
(`01-scope-auditor.md`) at PR scope — the same logic-completeness and
assumption-extermination lens, narrowed to this Wave's clauses and success
conditions. The Scope Auditor proves the charter is sound; the readiness audit
proves *this unit* is complete, traceable, and ready.

**Done gate:** the audit also verifies your tier manual's §4 Verification Before
Done gate (binding row: *Done gate* in `../binding-manifest.md` resolves the
exact gate and verification commands for the running project). SEPMO adds nothing
to that gate here — it invokes it.

**The audit can and does send a unit back.** A failed readiness audit routes a
defect-fix slice back to the Actor — the same mediation path as a Critic
finding. It is not a rubber stamp.

```yaml
PR_READINESS_VERDICT:
  pr_unit: <PR-unit ID>
  charter_trace_complete: true | false    # every clause in PR_UNIT accounted for
  done_gate_clean: true | false           # tier manual §4 gate passed
  open_issues:
    - <issue>: BLOCKING | NON_BLOCKING
  verdict: READY_TO_ASSEMBLE | SEND_BACK
  send_back_reason: <if SEND_BACK — exactly what must change>
```

---

## Mode handling

The Orchestrator operates in interactive and delegated modes as defined in the
tier manual's **Mode Handling** section (binding row: *Mode handling* in
`../binding-manifest.md`). The operational translation for the Orchestrator:

**Interactive mode:** check in with the user before dispatching the first Actor
on a new Wave ("here is the slice I am about to hand — does this carving look
right?"), before accepting a suspicious `NO_FINDINGS` convergence, and whenever
a material dispute or cycle-cap breach is reached.

**Delegated mode:** document the Wave carving plan in `task/todo.md` before
dispatching any Actor; proceed without blocking; surface all escalation points,
unresolved disputes, and cap-triggered re-assessments in the final report with
enough context to act on them without re-reading everything. Ambiguity that
changes the outcome is still a stop condition even in delegated mode — report
and stop rather than guess and proceed.

If the mode is ambiguous, treat it as delegated — document and proceed is the
safer direction.

---

## Proportionality

Canonical proportionality rule: `../SKILL.md` (*Proportionality — ceremony
scales with risk*). Orchestrator-specific application:

- **The mechanism is the `ceremony` field** in `PR_UNIT`. The Orchestrator sets
  it per unit before dispatching the Actor.
- **Triggers for `ceremony: FULL`:** any Wave touching the on-disk format, the
  public API, or a security surface. When in doubt, assign `FULL` — under-scoping
  ceremony is the riskier error.
- **`ceremony: LIGHTWEIGHT`** provides: a single Actor build, a single Critic
  pass, and a compressed readiness check. It never removes the 100%
  charter-trace, the Done gate, or the Critic's attack — the standard does not
  weaken, only the amount of ceremony.

---

## Doctrine enforcement

The Orchestrator is the runtime enforcer of D1–D6 for the agents it dispatches.
The doctrines are owned by `../SKILL.md`; what follows is only the
Orchestrator's enforcement posture at each decision point:

- **D1 (→ `../SKILL.md`):** verify the slice has no unstated preconditions
  before dispatch — escalate *before* the Actor builds on a false assumption,
  not after.
- **D3 (→ `03-self-logic-review.md`):** the Orchestrator runs its own SLR
  before carving each PR-unit, before dispatching each Actor, and before
  accepting each Critic convergence. These are logged artifacts, not silent
  checks.
- **D5 (→ `../SKILL.md`):** every artifact emitted or accepted carries
  `charter_trace`. An artifact without it is rejected immediately.
- **D6 (→ `../SKILL.md`; Actor blindness → `04-actor.md` *Design note*):**
  Actor blindness is structural — enforced by ensuring the Actor's dispatch
  package never mentions the Critic or audit. This is not optional.
- **Convergence authority:** the Orchestrator never overrides a `NOT_YET`
  verdict, never shortens the cycle cap to hit a deadline, and never advances a
  PR with an unresolved MEDIUM+ finding without user sign-off.

---

## Inputs and outputs

**Inputs:**
- Frozen `REFINED_CHARTER` + Scope Auditor's `GO_DECISION` artifact (state 2)
- The Roadmap and GAP_MATRIX — re-read at charter-derivation time, never cached
- `ACTOR_BUILD_SUMMARY` + SLR logs from each Actor phase
- `CRITIC_FINDINGS` from each Critic phase
- `PR_READINESS_VERDICT` from the readiness audit

**Outputs:**
- `CHARTER` record (once, frozen on entry to state 3)
- `PR_UNIT` records (one per Wave)
- Defect-fix slices routed to the Actor (opaque — no Critic attribution)
- Escalations to the user (material disputes, cycle-cap breaches, mode-dependent)
- Assembled PRs handed to the Delivery agent (`07-delivery.md`)

---

## Handoff

When every PR-unit in the charter has cleared `PR_READINESS_AUDIT` and been
assembled, the Orchestrator hands the PR set to the **Delivery agent**
(`07-delivery.md`) — one PR at a time, in Wave order. The charter is not
considered fulfilled until every Wave has been accepted by Delivery.

The **Retrospective** (`08-retrospective.md`) runs after the final PR is
delivered. The Orchestrator surfaces any mid-flight HALT reviews, cycle-cap
breaches, and escalation decisions so the Retrospective can mine them for
feed-forward lessons.

The Orchestrator's job ends when the last PR is delivered and the Retrospective
has been handed its artifact set. It does not declare the charter fulfilled —
that is Delivery's verdict, one PR at a time.

Note: `CONSTANT_VIGILANCE` (state 5, `06-vigilance.md`) runs concurrently with
state 4. When it alarms on scope change or drift, the Orchestrator halts the
affected PR-unit and routes the changed scope back to state 1 for re-audit.
The Orchestrator responds to vigilance alarms but does not run the monitor
itself — that is the Vigilance Monitor's role.
