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

# SEPMO canon change request — CCR-2026-07-26 (the G4 parking incident)

**Status: DRAFTED, awaiting user ratification.** Canon changes land only by amendment at the
master home (spine, *Global conventions — versioned canon*). The master home on this machine is
`~/Desktop/Sepmo/`; this repo's `skills/sepmo/` is an instance bound at `spine_version: v2.2`.
On ratification: the amendment text below lands at the master as **canon v2.3** (spine + changelog
+ required reference amendments), then this repo re-binds as its own SEPMO unit (verbatim spine
copy, reference amendments, manifest version bump + instantiation checklist), the same flow as the
2026-07-13 v2.2 install. **Repo-side bar-raising bindings landed immediately** under the
asymmetric feed-forward rule, independent of ratification: the `contingency_mechanics` row in
[skills/sepmo/binding-manifest.md](../skills/sepmo/binding-manifest.md) (stamped 2026-07-26) and
the first incident section in [task/sepmo-metrics.md](sepmo-metrics.md).

## The incident (evidence)

During the Mode B bundle `fix/engine-trust-bundle-2026-07` (nine sequential AC groups, one PR;
2026-07-25/26), group **G4** reached its cycle cap without convergence — its cycle-2 Critic held
an open S2 that was a **record-truth defect** (a GAP_MATRIX residue clause claiming a behavior
"already matches" Java when it does not), while the group's code substance was separately sound.
The bundle design's parking contingency — reset the branch to the last good commit — required
**destructive git authority the executing agent did not hold**; the session's permission
classifier blocked it (correctly: the contingency was orchestrator-authored, never user-named).
The pipeline then **continued**: groups G5–G8 built on top of the unsettled state, with 7 of G4's
10 touched files later modified by those groups. The bundle-scope closing Critic caught the
breach as its S1, a closing remediation corrected the record defects, the re-attestation
converged the whole bundle with the residual governance decision (strip-or-waive G4) surfaced as
an explicit user merge gate on the PR. **No defect escaped**; this is a lifecycle-machinery
incident, filed per the spine's incident-retrospective rule while the evidence is fresh.

Evidence homes: the PrimarySync claim-board rows for `fix/engine-trust-bundle-2026-07`
(2026-07-25/26), the bundle branch's `task/todo.md` Unit-2 close-out block, and the orchestration
run record (session `99d5183a`, workflow `wf_417ac10c-e61`, per-group Critic reports under
`scratchpad/pr-bodies/bundle-2026-07/`).

## Root causes — three, distinct

- **RC-1 — Invalid contingency.** The plan's failure path required an authority its executor
  lacked, and nothing at plan review checked executability. A contingency that cannot fire is an
  unproven assumption wearing a safety label.
- **RC-2 — Disposition propagation.** Existing canon already halts an unconverged **PR unit** at
  the cycle cap in delegated mode (ref 02, *Cycle-cap escalation*: "Do not deliver a PR with an
  unconverged Critic") — but a **group inside a Mode B bundle** is not a PR, and no rule bound
  its disposition. When the park failed, continuation was the path of least resistance, and the
  breach was loudly logged yet never *settled* — logging a breach is not settling it.
- **RC-3 — The save was improvised.** What rescued the bundle — remand to the bundle-scope
  closing authority, item-by-item disposition, a recorded user merge gate — has no legal
  definition in canon. It worked because the closing Critic chose to do all of it; none of it
  was required. An undefined recovery path that happens to work once is luck, not law.

## Amendment A — new sub-machine rule R11: contingencies must be executable

> **R11 — Contingencies must be executable.** Every failure-path action named in a plan, bundle
> design, or orchestration script — parking, rollback, reset, abort — is part of the plan's
> proof obligation: it must be executable, under the live permission regime, by the role that
> will trigger it, at the moment it fires. A contingency requiring authority its executor lacks
> is not a contingency; it is an unproven assumption (D1), and PRE_EXECUTION_REVIEW verifies
> executability for every named contingency (a failure routes T6). Two valid forms:
> **(a) additive by construction** — implemented entirely with forward operations (revert
> commits, supersede records) that need no destructive authority; **(b) pre-authorized
> destructive** — the destructive operation is named, scoped, and explicitly granted in the
> user's sign-off for that plan. Default to (a); a destructive contingency on an unattended path
> is invalid even when convenient.

## Amendment B — new sub-machine rule R12: an unsettled disposition blocks the line

> **R12 — An unsettled disposition blocks the line.** Every unit — and every group inside a
> multi-unit assembly — ends in exactly one recorded disposition: **CONVERGED**, **REMOVED**
> (its effects verifiably absent from the branch), or **REMANDED** (R13). Work whose disposition
> is unresolved — including work whose contingency fired and failed — is a blocking state:
> nothing downstream may consume it or build atop it. *Interactive mode:* halt and escalate.
> *Delegated mode:* stop the line and flag; downstream stages do not run. Proceeding past an
> unsettled disposition is a silent gate skip (Invariant V) **even when the proceeding is loudly
> logged** — logging a breach is not settling it.

## Amendment C — new sub-machine rule R13: remand to the assembly's closing authority

> **R13 — Remand.** In a multi-unit assembly (a bundled PR), a unit that reaches its cycle cap
> without convergence has three legal dispositions: escalate (per *Cycle-cap escalation*),
> REMOVED, or **REMANDED**. A remand is explicit, never implied by continuation: the remand
> record enumerates every open finding with its severity. Downstream units may proceed only
> where their scope is demonstrably disjoint from the open findings' blast radius, and the
> disjointness claim is recorded. The assembly's **closing authority** — the independent
> bundle-scope Critic — must disposition every enumerated finding **item by item**; its
> attestation covers the remanded unit only when each finding is closed with evidence,
> disproved with evidence, or converted into a **recorded user decision**, and every such user
> decision (waive, strip, accept) is named as an explicit merge gate in the PR (R8). A closing
> authority that converges an assembly containing a remanded unit without the item-by-item
> disposition has not converged it.

## Changelog entry (draft for the master)

> **v2.3 — 2026-07-··.** The disposition discipline lands: **R11** (contingencies must be
> executable — additive-by-construction or sign-off-pre-authorized, verified at
> PRE_EXECUTION_REVIEW), **R12** (every unit and assembly group ends in a recorded disposition —
> CONVERGED / REMOVED / REMANDED; an unsettled disposition blocks the line, and logging a breach
> is not settling it), **R13** (remand to the assembly's closing authority: explicit record with
> enumerated findings, recorded disjoint-scope rule for downstream work, item-by-item closing
> disposition, user decisions as named PR merge gates). Promoted from a consuming project's
> incident retrospective: a bundle group parked on an open finding, its destructive parking
> contingency proved unexecutable under the live permission regime, downstream groups consumed
> the unsettled state, and the bundle-scope closing Critic caught the breach and improvised what
> R13 now legalizes. *Reference amendments required by this version:* ref 02 —
> PRE_EXECUTION_REVIEW checklist gains contingency-executability; *Cycle-cap escalation* gains
> the REMOVED/REMANDED dispositions and the multi-unit-assembly binding; ref 03 — the review
> format gains a contingency-executability line; ref 05 — the bundle-scope closing Critic's
> item-by-item remand duty; ref 06 — new watch item: unsettled-disposition consumption /
> invalid contingency (T8-class); ref 08 — incident retrospectives explicitly cover
> lifecycle-machinery incidents, not only escaped code defects.

## Adoption protocol

1. **User ratifies** this CCR (edits welcome — the text above is a proposal, not a fait
   accompli).
2. **Master home updated:** `~/Desktop/Sepmo/` — spine `version: "2.3"`, the three rules
   inserted after R10, the changelog entry, and the five reference amendments.
3. **This repo re-binds** as its own SEPMO unit (verbatim spine copy, reference amendments,
   `spine_version: v2.3` in the manifest, instantiation checklist re-proven, independent Critic)
   — the same flow as `infra/sepmo-canon-v2.2`.
4. **Other SEPMO-bound repos** re-bind at their next project boundary (staleness alarm per
   Invariant V, not silent divergence).

## What already landed (asymmetric feed-forward, stamped 2026-07-26, ratification-independent)

- `skills/sepmo/binding-manifest.md` — new `contingency_mechanics` tunables row binding this
  repo to additive-only contingencies, halt-on-failed-contingency, and the explicit
  remove-or-remand rule for bundles; plus a Debug-section canon-gap filing pointing here.
- `task/sepmo-metrics.md` — the ledger's first populated section: this incident,
  `kind: incident`, with the feed-forward recorded and the strip-or-waive disposition marked
  pending until the user decides at the bundle's merge.
- Orchestration practice (not canon): this workstream's workflow scripts adopt park-by-revert
  and halt-on-failed-park as standing mechanics regardless of ratification.
