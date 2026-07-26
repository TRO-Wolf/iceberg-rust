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

# SEPMO metrics ledger

The location bound by [skills/sepmo/binding-manifest.md](../skills/sepmo/binding-manifest.md)
(`metrics_ledger_location`). **One section per retrospective** — charter-close or incident — in
the canonical `METRICS` format owned by
[skills/sepmo/references/08-retrospective.md](../skills/sepmo/references/08-retrospective.md)
(*Step 2 — the metrics ledger*), including `environment_drift_events` (spine v2.1+). Append-only;
sections are never rewritten, only superseded by later sections.

Created 2026-07-13 with the canon v2.2 re-instantiation (`infra/sepmo-canon-v2.2`). First
section filed 2026-07-26 (the G4 parking incident, below). Pre-v2.2 history is recorded in
[task/todo.md](todo.md) unit closes and [task/lessons.md](lessons.md); it is not retrofitted here
(no fabricated metrics).

**The 2026-07-13 standing `environment_drift_events` candidate — OVERTAKEN, never filed** (noted
2026-07-26): the 2026-07-11 nightly-interop failure was bytecode-diagnosed on 2026-07-24 as a
**real Rust parity gap** — Java `planTasks` merges adjacent same-file splits, Rust did not
([docs/parity/GAP_MATRIX.md](../docs/parity/GAP_MATRIX.md) row R148; fixed #169, interop-pinned
#170). A true-positive catch by the interop net, not R10 drift (R10's definition: an event
nothing in the AC loop could have caught — a real product defect is its negation). No drift
event exists to file from it.

---

<!-- METRICS sections append below this line. -->

## 2026-07-26 — incident: the G4 parking failure (engine-trust bundle)

First populated section of the v2.2-era ledger. `kind: incident` per the spine's
incident-retrospective rule — a **lifecycle-machinery incident, not an escaped code defect**:
nothing shipped past an accepted PR; the breach was caught in-flight by the bundle-scope closing
Critic before assembly. Narrative, root causes RC-1..RC-3, and the canon feed-forward:
[task/sepmo-ccr-2026-07-26-g4-incident.md](sepmo-ccr-2026-07-26-g4-incident.md). Metrics below
are scoped to the **incident chain** (the G4 ladder terminus + the closing chain), not the whole
bundle — the bundle's full charter-close section lands when its PR merges (no double count).

```yaml
METRICS:
  charter: engine-trust-bundle-2026-07 / incident G4-parking-2026-07-26
  kind: incident
  units_total: 1              # the incident scope: bundle group G4 (bundle = one Mode B PR, 9 groups)
  units_by_path: {STANDARD: 1, LIGHT: 0}
  cycles_per_unit: [G4: 3 (build + 2 remediation rounds), terminus: cap reached, NOT converged]
  findings_filed: {S0: 0, S1: 1, S2: 1, S3: 3}
    # the closing chain over the breach: CB-1 S1 (parked group with an open S2 riding at tip,
    # in breach of the bundle's parking rule), CB-2 S2 (the false R161 residue clause itself),
    # CB-3 S3 (interop discovery floor not ratcheted), CB-4 S3 (group-outcome record drift),
    # CR-1 S3 (the strip-or-waive user gate left unrecorded → recorded on the PR).
    # G4's own in-ladder findings are the bundle's, tallied at its charter close.
  findings_withdrawn: 0
  noise_ratio: 0.0
  disputes: {sustained: 0, withdrawn: 0}
  accepted_flags: 0           # no ACCEPTED_FLAGGED (R6 sustained-dispute-below-floor) shipped;
                              # the strip-or-waive G4 decision is a NAMED USER MERGE GATE on the
                              # PR (from CR-1, remediated by recording it), not a flag
  coverage_misses: []
    # none — the inverse case: the bundle-scope closing attestation is the layer that CAUGHT the
    # breach (its CB-1). Had the assembly relied on per-group ceremony alone, the unsettled
    # disposition would have shipped unexamined — that near-miss is RC-2/RC-3 in the CCR.
  escaped_defects_by_origin: []
  environment_drift_events: []
    # none. The ledger's 2026-07-13 standing candidate (the 2026-07-11 nightly failure) was
    # OVERTAKEN before this first section landed: diagnosed 2026-07-24 as a real parity gap
    # (R148; fixed #169/#170) — a detector win, not drift; see the preamble note. The
    # classifier BLOCK of G4's parking reset is also deliberately NOT drift — the environment
    # behaved correctly; the plan was the defect (RC-1).
```

**Feed-forward filed (asymmetric — bar-raising landed 2026-07-26):** the
`contingency_mechanics` binding row in
[skills/sepmo/binding-manifest.md](../skills/sepmo/binding-manifest.md) (additive-only
contingencies; halt on failed contingency; remove-or-remand for bundle groups) + canon v2.3
proposal (R11/R12/R13) in the CCR, awaiting user ratification at the master home
(`~/Desktop/Sepmo`).

**Open disposition (append when decided):** the user's strip-or-waive call on G4 at the bundle
PR's merge. *(Appended 2026-07-26: WAIVED — user decision at SQM; the bundle merged as #173
(`4063007b`) with G4 retained; the R13-remand substance had been independently re-attested by
the closing chain.)*

## 2026-07-26 — incident #2: delegated-agent canon edit (master home)

`kind: incident`, lifecycle-machinery, **canon-integrity class**. A delegated agent (mistakenly
instructed) amended the MASTER canon home (`~/Desktop/Sepmo`) in place: a runtime-specific spine
companion section + a `v2.2+kit` changelog entry + engine tunables in the portable template, plus
a runtime install kit inside the canon directory. Caught same-day by user report; an mtime audit
scoped it; the master was restored **byte-exact to v2.2** (36695/12375 bytes, verified against
the fork-instance baseline at `a6199ca5` — SKILL.md cmp-identical; the template identical after
stripping the fork copy's ASF header, which the master deliberately does not carry) and the kit
relocated to `~/Desktop/Sepmo-octo-kit/`
(value preserved; its install script re-pointed). No instance impact — this repo's
`skills/sepmo/` and the coordination hub were untouched.

```yaml
METRICS:
  charter: none / incident canon-edit-2026-07-26
  kind: incident
  units_total: 0              # the event occurred OUTSIDE any SEPMO unit — that absence IS the finding
  units_by_path: {STANDARD: 0, LIGHT: 0}
  cycles_per_unit: []
  findings_filed: {S0: 0, S1: 0, S2: 0, S3: 0}   # no AC ladder ran; nothing to file under R4
  findings_withdrawn: 0
  noise_ratio: 0.0            # 0 over 0, rendered as zero by convention
  disputes: {sustained: 0, withdrawn: 0}
  accepted_flags: 0
  coverage_misses: []
    # none in-machine — but the machine never saw the event: the master home has NO gate, no
    # version control, no Invariant V watching it. Detection was a user report + mtime forensics.
  escaped_defects_by_origin: []
  environment_drift_events: []
```

**Feed-forward filed:** CCR Amendment E (runtime-neutral `critic_engine` binding point — the
event's one canon-worthy idea, abstracted) + two governance recommendations riding the CCR
adoption protocol: version-control the master home; reconcile the DIVERGED reference lineages at
ratification (master vs instance wording — both carry the required amendments; the audit's first
"stale" reading was refuted by the unit's Critic, and the correction is itself recorded in the
CCR).

**Dispositions (2026-07-26, at the v2.3 re-bind):** Amendment E **RATIFIED** and landed — the
normative constraints in canon's ref 05, the optional `critic_engine` row in the template, and
this repo bound to the default (the spine's own Critic stage). Lineage reconciliation **SETTLED
as deliberate coexistence**: each version's deltas are applied per lineage in that lineage's own
wording, neither imported over the other, with only the spine byte-identical — the standing rule
now lives in the *Debug* list of
[skills/sepmo/binding-manifest.md](../skills/sepmo/binding-manifest.md). Version-controlling the
master home **DEFERRED by the user** — it stays ungoverned, so a dated pre-change copy
(`~/Desktop/Sepmo-v2.2-archive` for this one) is the only diff baseline a canon change gets.
