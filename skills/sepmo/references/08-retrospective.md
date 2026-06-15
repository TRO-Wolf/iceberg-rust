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

# 08 — Retrospective (The Compaction Pass)

> Run after the final PR is delivered. Mine the run's adversarial artifacts for promotable lessons, then compact memory. The pass is the lesson.

The Retrospective agent owns state 7 (`RETROSPECTIVE`) in the Iron State Machine (`../SKILL.md`).
It runs once, after the final PR in the charter set has been accepted by Delivery (`07-delivery.md`,
*Charter fulfillment and Retrospective handoff*). It does not run during execution — that is
`CONSTANT_VIGILANCE`'s role (state 5, `06-vigilance.md`).

Its mandate: two steps performed as one scoped change.

1. **SEPMO feed-forward** — mine the run's adversarial artifacts for promotable lessons and stage
   them as candidates for the compaction pass.
2. **Compaction pass** — run `../../compaction.md` as written.

The Retrospective is a thin seam. It contributes the feed-forward step; everything else — the
PROMOTE/KEEP/ARCHIVE triage, the conservation gate, the agentic-pace recency amendment, the archive
layout, the done gate, the anti-patterns — lives in `../../compaction.md` and is not restated here.
The binding that makes the connection load-bearing: `../binding-manifest.md` row *Memory /
lessons* — "Retrospective runs a compaction pass."

---

## What the Retrospective receives

Two artifact sets are handed to the Retrospective at the start of state 7:

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

---

## Step 2 — compaction pass

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
- Updated `task/lessons.md` (feed-forward entries added + KEEP survivors after compaction)
- New `task/lessons-archive/<scope>.md` (ARCHIVE entries moved verbatim)
- Promotion edits to each target file named in `promotions` (in the same change, per
  `../../compaction.md` *Promotion targets*)

---

## Handoff

The Retrospective is the terminal state. When `conservation_check: RECONCILED` and the compaction
done gate (`../../compaction.md`, *Done gate for a compaction pass*) is clean, the run is complete.
The charter is fulfilled; the memory is compacted; the lessons are filed.

The Retrospective does not hand off to another agent. Its output is the updated memory system,
readable by every future session from the top of its read order.
