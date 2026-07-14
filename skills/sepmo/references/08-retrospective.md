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

# 08 — Retrospective (Learnings AND Metrics)

> Run after the final PR is delivered — or immediately, mini-scoped, when a defect escapes. Mine
> the run's adversarial artifacts for promotable lessons, file the quantitative metrics ledger,
> then compact memory. T12 does not exit without the metrics.

The Retrospective agent owns state 6 (`RETROSPECTIVE`) in the Iron State Machine (`../SKILL.md`).
It runs once, after the final PR in the charter set has been accepted by Delivery
(`07-delivery.md`, *Charter fulfillment and Retrospective handoff*) — plus once per **escaped
defect** as an incident retrospective (below). It does not run during execution — that is
Invariant V's watch (`06-vigilance.md`).

Its mandate: three steps performed as one scoped change.

1. **SEPMO feed-forward** — mine the run's adversarial artifacts for promotable lessons and stage
   them as candidates for the compaction pass.
2. **The metrics ledger** — file the quantitative metric set (below) at the location bound by
   `../binding-manifest.md` (`metrics_ledger_location`). T12's guard is "metrics ledger
   complete"; a retrospective without metrics has not run.
3. **Compaction pass** — run `../../compaction.md` as written.

The Retrospective is a thin seam. It contributes the feed-forward step; everything else — the
PROMOTE/KEEP/ARCHIVE triage, the conservation gate, the agentic-pace recency amendment, the archive
layout, the done gate, the anti-patterns — lives in `../../compaction.md` and is not restated here.
The binding that makes the connection load-bearing: `../binding-manifest.md` row *Memory /
lessons* — "Retrospective runs a compaction pass."

---

## What the Retrospective receives

Two artifact sets are handed to the Retrospective at the start of state 6:

- **From Delivery** (`07-delivery.md`, *Charter fulfillment and Retrospective handoff*): the
  complete set of `DELIVERY_VERDICT` records for the run, including any `reject_reason` fields from
  `REJECTED` verdicts that were later resolved.
- **From the Orchestrator** (`02-orchestrator.md`, *Handoff*): mid-flight HALT reviews, cycle-cap
  breaches, and escalation decisions that arose during `ORCHESTRATED_EXECUTION`.

These two sources are the adversarial record of the run. Everything the pipeline caught, almost
missed, or had to escalate is in them.

---

## Step 1 — SEPMO feed-forward

Before opening `task/lessons.md`, scan the adversarial artifacts for entries that name a *class* of
problem — not "this PR-unit had a concurrency bug" but "the Actor consistently mishandled concurrent
write paths." Class-level observations are PROMOTE candidates per the promotability tests in
`../../compaction.md`; instance-level observations are ARCHIVE candidates.

### Artifacts to mine

| Artifact | Source | What to look for |
|---|---|---|
| `SELF_LOGIC_REVIEW` records with `verdict: HALT` | `03-self-logic-review.md` (*Logging and reuse*) | Recurring `escalation` patterns — classes of ambiguity that escaped the scope audit and fired repeatedly inside execution |
| `CRITIC_FINDINGS` | `05-critic.md` | Recurring `category` and `severity` fields across PR-units — systematic weaknesses in the Actor's build discipline or in charter success conditions; `claim-gap` findings that recur |
| `VIGILANCE_ALARM` records | `06-vigilance.md` | Trigger types that fired more than once (especially T2 de-triplication, T3 `map.md` drift); any alarm that triggered a return to state 1 |
| `DELIVERY_VERDICT` `reject_reason` fields | `07-delivery.md` | Done-gate or charter-clause failures that caused rejection — recurring reject reasons are a signal that the Done gate or charter success conditions are underspecified |

For each candidate, draft a lesson in the DO/DO NOT form that `task/lessons.md` uses, date-stamped
today, and mark it with a provisional verdict (PROMOTE or KEEP).

### Feeding the candidates into the compaction pass

Add the drafted lessons to `task/lessons.md` **before** running the triage step of
`../../compaction.md`. They enter as new entries and are subject to the same PROMOTE/KEEP/ARCHIVE
triage as every existing entry — the feed-forward populates the candidate pool; the compaction pass
adjudicates verdicts.

### Feed-forward is asymmetric (raise now, lower at the boundary)

Feed-forward proposals that change SEPMO's operation land through the binding manifest as
versioned, date-stamped updates — never as canon edits (canon changes are amendments to the
master home, user-approved; spine, *versioned canon*). The timing rule is **asymmetric**:

- **Bar-raising changes land immediately** — a new pinning obligation, a tightened binding, a
  raised severity floor, a new attack category — stamped with date and provenance.
- **Bar-lowering or neutral changes wait for the project boundary** — the never-mid-project rule
  exists to stop the bar dropping under pressure, not to delay its rise
  (stricter-interpretation-wins applied to feed-forward; canonical statement: `../SKILL.md`,
  *Incident retrospectives*).

---

## Step 2 — the metrics ledger

**Metrics are not optional** (`../SKILL.md`, *Global conventions*). One section per
retrospective — charter-close or incident — appended at the manifest-bound location
(`metrics_ledger_location`). The canonical metric set:

```yaml
METRICS:
  charter: <CHARTER.id>                  # or the escaped-defect id for an incident section
  kind: charter-close | incident
  units_total: <n>                       # PR units delivered under this charter
  units_by_path: {STANDARD: <n>, LIGHT: <n>}
  cycles_per_unit: [<unit>: <n>, ...]    # AC cycles each unit took to converge
  findings_filed: {S0: <n>, S1: <n>, S2: <n>, S3: <n>}
  findings_withdrawn: <n>
  noise_ratio: <withdrawn ÷ filed>       # the Critic's precision counterweight (R4)
  disputes: {sustained: <n>, withdrawn: <n>}
  accepted_flags: <n>                    # ACCEPTED_FLAGGED shipped, each disclosed per R6/R8
  coverage_misses:
    - <escaped defect>: <the attestation category that sat clean over it>
  escaped_defects_by_origin:
    - <defect>: <origin: audit-gap | pinning-gap | attestation-gap | readiness-gap>
  environment_drift_events:
    - <event>: <gate that went red + the base-ref proof it was environmental>
    # canonical definition (R10): a red gate reproduced on the base ref WITHOUT the
    # unit's diff — advisory publications, upstream relicenses, tool-version skew,
    # CI-runner divergence. A distinct counter, never an escaped defect: nothing in
    # the AC loop could have caught it.
```

`coverage_misses` and `escaped_defects_by_origin` are empty on a clean run — but the keys are
still filed; an absent key is indistinguishable from an unexamined one.

### Incident retrospectives — when a defect escapes

An **escaped defect** — discovered after its PR was accepted — does not wait for the charter to
close: run a **mini state-6 scoped to the defect immediately** (canonical rule: `../SKILL.md`,
*Incident retrospectives*). File a `kind: incident` metrics section now — `coverage_misses`
naming the attestation category that sat clean over the failure, and
`escaped_defects_by_origin` — and propose changes while the evidence is fresh, under the
asymmetric feed-forward rule above. The defect's remediation is its own unit through the normal
machine with R5's regression proof; where the defect falsified a quantified clause, check whether
the clause's enumeration contained the failing element — if it did not, the enumeration was the
defect: grow it.

---

## Step 3 — compaction pass

Run `../../compaction.md` as written. Every rule in that file applies unchanged:

- The trigger that fires for a Retrospective pass is typically *phase boundary* (trigger 2 in
  `../../compaction.md`) — the completed charter set is a natural archive line.
- The conservation gate, the agentic-pace recency amendment, the archive layout, the promotion
  mechanics, the done-gate checklist, and the anti-patterns are all owned by `../../compaction.md`.
  Read them there; do not proceed without reading them.
- Promotion targets are those in `../../compaction.md` *Promotion targets* table — read them there;
  Retrospective adds none.
- Run the pass as its own scoped change per `../../compaction.md` *Procedure* step 1.

---

## `RETROSPECTIVE_LOG` artifact

Machine-readable record of what was mined and what the pass produced. One per run.

```yaml
RETROSPECTIVE_LOG:
  id: RL-<short-id>
  charter: <CHARTER.id from 02-orchestrator.md>
  artifacts_mined:
    halt_slr_count: <n>           # SELF_LOGIC_REVIEW records with verdict: HALT
    critic_findings_count: <n>    # total CRITIC_FINDINGS entries across all PR-units
    vigilance_alarm_count: <n>    # VIGILANCE_ALARM records raised during the run
    delivery_rejections: <n>      # DELIVERY_VERDICT records with verdict: REJECTED
  feed_forward:
    lessons_drafted: <n>          # new DO/DO NOT entries added to task/lessons.md
    promote_candidates: <n>       # entries provisionally marked PROMOTE before triage
    manifest_updates:
      - <bar-raising: landed now, stamped> | <bar-lowering/neutral: queued for boundary>
  metrics_section: <pointer into the manifest-bound metrics ledger — the METRICS block filed>
  compaction_pass:
    trigger: phase-boundary | size | staleness | on-request
    archive_file: <path — e.g. task/lessons-archive/2026-06_<scope>.md>
    tally: "<total> entries: <n> PROMOTE, <n> KEEP, <n> ARCHIVE"
    promotions:
      - lesson: <one-line summary>
        target: >
          # the lesson's canonical home, per ../../compaction.md *Promotion targets*
  conservation_check: RECONCILED | FAILED
  # see ../../compaction.md *Procedure* step 7 and *Done gate*
  completed_at: <ISO date>
```

---

## Inputs and outputs

**Inputs:**

- Full `DELIVERY_VERDICT` set from Delivery (`07-delivery.md`, *Charter fulfillment and
  Retrospective handoff*)
- Mid-flight HALT reviews, cycle-cap breaches, and escalation decisions from the Orchestrator
  (`02-orchestrator.md`, *Handoff*)
- `task/lessons.md` (active lessons file) and its compaction log — via `../binding-manifest.md`
  row *Memory / lessons*

**Outputs:**

- `RETROSPECTIVE_LOG` record (one per run, machine-readable, stable ID)
- A `METRICS` section appended to the manifest-bound metrics ledger
  (`../binding-manifest.md`, `metrics_ledger_location`)
- Updated `task/lessons.md` (feed-forward entries added + KEEP survivors after compaction)
- New `task/lessons-archive/<scope>.md` (ARCHIVE entries moved verbatim)
- Promotion edits to each target file named in `promotions` (in the same change, per
  `../../compaction.md` *Promotion targets*)

---

## Handoff

The Retrospective is the terminal state (T12), and its exit guard is **metrics ledger
complete** — learnings without metrics do not exit. When the METRICS section is filed,
`conservation_check: RECONCILED`, and the compaction done gate (`../../compaction.md`, *Done gate
for a compaction pass*) is clean, the run is complete. The charter is fulfilled; the metrics are
filed; the memory is compacted; the lessons are filed. (An incident retrospective is not
terminal — it files its section and hands the remediation unit back to the machine.)

The Retrospective does not hand off to another agent. Its output is the updated memory system,
readable by every future session from the top of its read order.
