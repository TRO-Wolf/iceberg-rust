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
The pipeline then **continued**: groups G5–G8 built on top of the unsettled state — 5 of G4's 10
touched files were modified by those groups (7 of 10 by branch tip, once the closing remediation
also landed). The bundle-scope closing Critic caught the
breach as its S1, a closing remediation corrected the record defects, the re-attestation
converged the whole bundle with the residual governance decision (strip-or-waive G4) surfaced as
an explicit user merge gate on the PR. **No defect escaped**; this is a lifecycle-machinery
incident, filed per the spine's incident-retrospective rule while the evidence is fresh.

Durable evidence homes: the PrimarySync claim-board rows for `fix/engine-trust-bundle-2026-07`
(2026-07-25/26), the bundle branch's `task/todo.md` Unit-2 close-out block and `task/lessons.md`
entries (the CLOSE lesson: "a tripped per-group gate must be settled before the next group"),
and the bundle PR's description, which embeds the record per R8 (the per-group Critic reports it
was assembled from are session-scratchpad artifacts and ephemeral).

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
> unsettled disposition is an Invariant V alarm in its own right — **unsettled-disposition
> consumption** — **even when the proceeding is loudly logged**: logging a breach is not
> settling it.

## Amendment C — new sub-machine rule R13: remand to the assembly's closing authority

> **R13 — Remand.** In a multi-unit assembly (a bundled PR), a unit that reaches its cycle cap
> without convergence either follows *Cycle-cap escalation* (which terminates in a disposition)
> or takes one of two settled dispositions directly: **REMOVED**, or **REMANDED**. A remand is explicit, never implied by continuation: the remand
> record enumerates every open finding with its severity. Downstream units may proceed only
> where their scope is demonstrably disjoint from the open findings' blast radius, and the
> disjointness claim is recorded. The assembly's **closing authority** — the independent
> bundle-scope Critic — must disposition every enumerated finding **item by item**; its
> attestation covers the remanded unit only when each finding is closed with evidence,
> disproved with evidence, or converted into a **recorded user decision**, and every such user
> decision (waive, strip, accept) is named as an explicit merge gate in the PR (R8). A closing
> authority that converges an assembly containing a remanded unit without the item-by-item
> disposition has not converged it.

## Amendment D — the spine's *Incident retrospectives* section widens its trigger

The section currently triggers only on an **escaped defect**. Append:

> **A lifecycle-machinery incident triggers the same retrospective.** A failure of SEPMO's own
> machinery — an invalid contingency, an unsettled disposition consumed downstream, a gate
> bypassed — files the same immediate `kind: incident` metrics section and runs the same
> asymmetric feed-forward, whether or not any product defect escaped. In such a section,
> `coverage_misses` and `escaped_defects_by_origin` may be legitimately empty — the keys are
> still filed, and the incident's mechanism is named in the section body.

(Without this, a machinery incident like the one motivating this CCR has no legal home in the
ledger the spine itself mandates; ref 08 mirrors this extension — canonical rule stays in the
spine.)

## Amendment E — external critic-engine binding point (optional; second provenance event)

Provenance: on 2026-07-26 a **delegated agent (mistakenly instructed) amended the master canon in
place** — a runtime-specific "critic engine" companion section in the spine, a `v2.2+kit`
changelog entry, and engine-specific tunables in the portable template, plus a runtime install
kit inside the canon home. Caught the same day by user report; the master was restored byte-exact
to v2.2 from the fork-instance baseline and the kit relocated to its own home
(`~/Desktop/Sepmo-octo-kit/`) — incident #2 in [task/sepmo-metrics.md](sepmo-metrics.md). The one
idea worth canon, abstracted runtime-neutral:

> **Template (optional binding row) `critic_engine`** — default: the spine's own Critic stage. A
> project MAY bind an **external critic engine** (a multi-critic harness, a different runtime)
> for STANDARD-and-above units. Constraints (normative, ref 05): (1) an external engine's
> convergence signal is **never Delivery** — its output maps to a coverage attestation plus a
> findings ledger, and PR_READINESS_AUDIT then proceeds as always; (2) **LIGHT units never
> select an external engine**; (3) the engine's attack taxonomy must satisfy **R4** — its
> categories map onto ref 05's canonical taxonomy (plus manifest extensions) or each unmapped
> category is justified `N/A`; (4) **engine-specific tunables** (cycle counts, early-stop
> policy, scratch locations) bind in the project's manifest, never in the portable template.

Two governance observations from the same event, for ratification alongside: (a) the master home
has **no gate** — nothing detected the in-place canon edit but a user report and an mtime audit;
recommend placing `~/Desktop/Sepmo` under version control so canon changes are diffable events;
(b) the audit also discovered the master's `references/` are **stale** — they lack the v2.1/v2.2
*required* reference amendments that currently exist only in this repo's instance — folded into
the adoption protocol below.

## Changelog entry (draft for the master)

> **v2.3 — 2026-07-··.** The disposition discipline lands: **R11** (contingencies must be
> executable — additive-by-construction or sign-off-pre-authorized, verified at
> PRE_EXECUTION_REVIEW), **R12** (every unit and assembly group ends in a recorded disposition —
> CONVERGED / REMOVED / REMANDED; an unsettled disposition blocks the line, and logging a breach
> is not settling it), **R13** (remand to the assembly's closing authority: explicit record with
> enumerated findings, recorded disjoint-scope rule for downstream work, item-by-item closing
> disposition, user decisions as named PR merge gates). The spine's *Incident retrospectives*
> section widens its trigger to **lifecycle-machinery incidents** (Amendment D) — machinery
> failures file the same immediate `kind: incident` section whether or not a product defect
> escaped. Promoted from a consuming project's incident retrospective: a bundle group parked on
> an open finding, its destructive parking contingency proved unexecutable under the live
> permission regime, downstream groups consumed the unsettled state, and the bundle-scope
> closing Critic caught the breach and improvised what R13 now legalizes. *Reference amendments
> required by this version:* ref 02 — PRE_EXECUTION_REVIEW checklist gains
> contingency-executability; *Cycle-cap escalation* gains the REMOVED/REMANDED dispositions and
> the multi-unit-assembly binding; ref 03 — the review format gains a
> contingency-executability line; ref 05 — the bundle-scope closing Critic's item-by-item
> remand duty AND the external-critic-engine constraints (Amendment E); ref 06 — new watch item
> (next unused trigger id): unsettled-disposition consumption / invalid contingency; ref 08 —
> mirrors Amendment D (canonical rule stays in the spine); the template — the optional
> `critic_engine` binding row (Amendment E, runtime-neutral).

## Adoption protocol

1. **User ratifies** this CCR (edits welcome — the text above is a proposal, not a fait
   accompli).
2. **Master home updated:** `~/Desktop/Sepmo/` — spine `version: "2.3"`, the three rules
   inserted after R10, the *Incident retrospectives* extension (Amendment D), the changelog
   entry, the five reference amendments plus the template's `critic_engine` row (Amendment E),
   AND the master-reference reconciliation: apply the v2.1/v2.2 *required* amendment sets (today
   present only in this repo's instance) so the master is whole. Recommended at the same step:
   `git init` the master home so future canon changes are diffable events, not mtime forensics.
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
