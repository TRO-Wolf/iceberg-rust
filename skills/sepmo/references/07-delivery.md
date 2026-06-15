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

# 07 — Delivery (Per-PR Acceptance & Handoff)

> One PR at a time. One verdict at a time. The charter is fulfilled only when every PR is accepted.

The Delivery agent owns state 6 (`DELIVERY`). It receives a single assembled PR from the
Orchestrator (`02-orchestrator.md`, *Handoff*), runs acceptance verification against the charter
and the project's Done gate, issues a machine-readable verdict, and — on acceptance — hands off to
the next step. It does not assemble PRs, drive remediation, or hold the full charter picture;
those roles belong to states 4 and 3 respectively.

**DELIVERY is per-PR.** Each assembled PR runs through this state on its own. The charter is
considered fulfilled only when every PR in the charter set is accepted. Delivery tracks that count
and declares charter fulfillment on the final acceptance — not before.

---

## What Delivery receives

The Orchestrator hands Delivery one assembled PR at a time, in Wave order, after the Actor–Critic
loop has converged, the readiness audit has passed, and the PR has been assembled. By the time
Delivery sees a PR:

- The Actor–Critic loop has converged (no open MEDIUM+ findings).
- The readiness audit has passed (`charter_trace_complete: true`, slice internally coherent).
- The PR description and diff are assembled with `charter_trace` clause IDs linking every change
  to the frozen charter.

Delivery starts from this assembled state. It does not repeat the AC loop, the readiness audit,
or the assembly — those concluded in state 4. Its job is acceptance verification and the
`ACCEPTED | REJECTED` verdict.

---

## Acceptance criteria

### 1 — Done gate

Acceptance requires the assembled PR to pass the project's Done gate. Delivery invokes this gate
entirely by reference — see `../binding-manifest.md` row *Done gate*. It adds no new conditions
and does not restate the gate's checklist or its verification commands — read them at their
canonical home.

### 2 — Charter-by-charter verification

Every `charter_trace` clause the PR claims must be checked against its `success_condition` in the
frozen charter. A PR that passes the Done gate but leaves a chartered clause unaddressed is
**not accepted** — the charter clause is the contract, not the code quality alone.

Per `../SKILL.md` *How to use this skill*: "Never declare `DELIVERY` for a PR without
charter-by-charter verification."

### 3 — Interop evidence for capability PRs

For any PR that advances a parity capability, a byte-level interop round-trip is required (see
`../../../CLAUDE.md` *Parity mandate* for the definition and rationale). Delivery applies that
rule; it does not restate it.

This criterion is not proportionality-scaled: every capability PR carries it, regardless of
ceremony level. A PR that contains only documentation, tooling changes, or governance with no
capability advancement is exempt. The `pr_type` discriminator in `DELIVERY_VERDICT` makes this
conditional check machine-readable.

---

## GAP_MATRIX flip discipline

When a PR's acceptance advances a capability row to done:

1. Edit `../../../docs/parity/GAP_MATRIX.md` — the capability cell — and **nothing else**. Status
   lives only in the GAP_MATRIX (binding-manifest row *Capability status (SSOT)*); the
   de-triplication rule in `../../../CLAUDE.md` *Working conventions* forbids any other location.
2. The flip requires both unit tests and an interop test — rule and rationale in
   `../../../CLAUDE.md` *Parity mandate*; read it there.
3. Record in `DELIVERY_VERDICT` that the flip was performed (`gap_matrix_flip: done`), not what
   the new cell value is — the matrix is the sole location for that content.

When a PR does not advance any capability row, record `gap_matrix_flip: not_applicable`.

---

## DELIVERY_VERDICT artifact

Machine-readable, addressable. One record per assembled PR. Every downstream agent (the
Orchestrator on rejection routing, the Retrospective on final handoff) references this by `id`.

```yaml
DELIVERY_VERDICT:
  id: DV-<short-id>              # stable; referenced by the Orchestrator and Retrospective
  pr_unit: <PR-unit ID>          # matches the PR_UNIT.id from 02-orchestrator.md
  pr_type: capability | non-capability
    # capability: this PR advances one or more parity rows in the GAP_MATRIX
    # non-capability: docs, governance, tooling, refactor — no parity row advancement
  charter_trace_verified: true | false
    # true only when every clause in the PR's charter_trace has its success_condition met
  done_gate_clean: true | false
    # true only when your tier manual's §4 gate passes
    # binding: ../binding-manifest.md row *Done gate*
  interop_evidence:
    required: true | false          # true when pr_type is capability
    provided: true | false | n/a    # n/a when required: false
    evidence_ref: >
      # Pointer to the interop test(s) or artifact proving the round-trip; omit when n/a
      # rule home: ../../../CLAUDE.md *Parity mandate*
  gap_matrix_flip: done | not_applicable
    # done: the capability row was flipped in the GAP_MATRIX cell, and nowhere else
    # not_applicable: this PR does not flip a capability row
  verdict: ACCEPTED | REJECTED
  reject_reason: >
    # Required when verdict is REJECTED. Identifies which criterion failed and what must change.
    # Omit when ACCEPTED.
  delivered_at: <ISO date>
```

---

## REJECTED verdict — routing

A `REJECTED` verdict is returned to the Orchestrator. Delivery does not drive remediation
directly — it raises the rejection with `reject_reason` populated and the Orchestrator re-enters
state 4 (`ORCHESTRATED_EXECUTION`) for that PR unit. The routing follows the Orchestrator's
mediation path (`02-orchestrator.md`, *Remediation mediation*): the specific failure criterion is
the input rather than a Critic finding, but the routing mechanics are identical.

Mode handling (interactive vs. delegated) for a `REJECTED` verdict follows the tier manual's Mode
Handling section, bound by `../binding-manifest.md` row *Mode handling* — consistent with how the
Orchestrator (`02-orchestrator.md`) and Vigilance Monitor (`06-vigilance.md`) handle their
respective escalations. In interactive mode: HALT and surface the rejection to the user before
any further work proceeds on the affected PR unit. In delegated mode: flag the rejection
prominently in the final report and stop work on that PR unit until it is resolved.

---

## Charter fulfillment and Retrospective handoff

Delivery tracks the set of accepted `DELIVERY_VERDICT` records against the charter's full PR set.
When the final PR is accepted:

1. Delivery declares the charter fulfilled, citing the complete set of `DELIVERY_VERDICT` records
   and their stable IDs.
2. Delivery hands its full set of `DELIVERY_VERDICT` records to the Retrospective agent
   (08-retrospective.md, not yet built). The Orchestrator separately surfaces mid-flight HALT
   reviews, cycle-cap breaches, and escalation decisions per `02-orchestrator.md` *Handoff*. The
   Retrospective mines both for feed-forward lessons.

Until that final acceptance, the charter is not fulfilled — even if every prior PR is accepted.
Partial delivery is not delivery.

---

## Inputs and outputs

**Inputs:**

- Assembled PR from the Orchestrator: PR description, diff, `charter_trace` clause IDs, and
  `PR_READINESS_VERDICT` (for reference — not re-verified; the readiness audit already ran in
  state 4)
- The frozen charter — for clause-by-clause `success_condition` lookup
- The project's Done gate — via `../binding-manifest.md` row *Done gate*
- The GAP_MATRIX — via `../binding-manifest.md` row *Capability status (SSOT)*, for flip
  eligibility and cell editing

**Outputs:**

- `DELIVERY_VERDICT` record (one per PR, machine-readable, stable ID)
- GAP_MATRIX cell edit, when a capability row is flipped (the only file Delivery touches for
  status, and only when `pr_type: capability` and evidence is confirmed present)
- Charter-fulfilled declaration + `DELIVERY_VERDICT` artifact set handed to the Retrospective, on
  final acceptance
