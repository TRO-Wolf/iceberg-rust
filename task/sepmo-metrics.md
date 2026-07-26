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

Created 2026-07-13 with the canon v2.2 re-instantiation (`infra/sepmo-canon-v2.2`). No
retrospective has been filed against the v2.2 ledger yet — the first section lands with the first
charter close or incident after this install. Pre-v2.2 history is recorded in
[task/todo.md](todo.md) unit closes and [task/lessons.md](lessons.md); it is not retrofitted here
(no fabricated metrics).

**Standing candidate for the first section's `environment_drift_events`:** the 2026-07-11 first
nightly-interop run failure (GitHub Actions run 29144349415) with the identical `make interop`
green locally on the same ref — base-ref-proven environmental per R10, cause on the CI runner
still undiagnosed (step-summary evidence needs admin access).

---

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
  accepted_flags: 1           # CR-1: the strip-or-waive G4 decision rides the PR as a named user merge gate
  coverage_misses: []
    # none — the inverse case: the bundle-scope closing attestation is the layer that CAUGHT the
    # breach (its CB-1). Had the assembly relied on per-group ceremony alone, the unsettled
    # disposition would have shipped unexamined — that near-miss is RC-2/RC-3 in the CCR.
  escaped_defects_by_origin: []
  environment_drift_events:
    - 2026-07-11 nightly-interop first-run failure (GH run 29144349415): `make interop` green
      locally on the identical ref — base-ref-proven environmental per R10; CI-runner cause
      undiagnosed. Carried here per this ledger's standing-candidate note (unrelated to G4);
      the classifier BLOCK of G4's parking reset is deliberately NOT counted here — the
      environment behaved correctly; the plan was the defect (RC-1).
```

**Feed-forward filed (asymmetric — bar-raising landed 2026-07-26):** the
`contingency_mechanics` binding row in
[skills/sepmo/binding-manifest.md](../skills/sepmo/binding-manifest.md) (additive-only
contingencies; halt on failed contingency; remove-or-remand for bundle groups) + canon v2.3
proposal (R11/R12/R13) in the CCR, awaiting user ratification at the master home
(`~/Desktop/Sepmo`).

**Open disposition (append when decided):** the user's strip-or-waive call on G4 at the bundle
PR's merge. *(Appended 2026-07-··: ……)*

<!-- METRICS sections append below this line. -->
