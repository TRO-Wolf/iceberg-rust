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

# 03 — Self Logic Review

> The mandatory pre-action checkpoint. **Every agent — Orchestrator, Scope
> Auditor, Actor, Critic, Delivery, Retrospective — runs this on itself before
> any implementation, code generation, task advancement, or decision.** This is
> doctrine **D3** made operational, and it is how an agent discharges D1, D2,
> D4, and D5 concretely at the moment of action.

The Self Logic Review is the cheapest insurance in SEPMO. It costs a frontier
model seconds to produce because the model genuinely holds the context — and
the act of writing it out is exactly what catches the silent error that
confidence alone would sail past. A plan that is sound in your head but cannot
survive being written down was never sound. There is **no fast path and no "this
one is obvious" exemption**: the moment an agent decides a review is unnecessary
is the moment the review was most needed, because that feeling is unexamined
confidence, which is the substance D3 exists to interrupt.

---

## When it runs

Before **every** discrete action that changes state or commits to a direction:

- The Orchestrator runs it before carving a PR unit, before dispatching an
  Actor, before accepting a Critic's convergence.
- The Actor runs it before writing the slice and again before declaring the
  slice built.
- The Critic runs it before filing findings (to check it actually attacked
  rather than skimmed).
- Delivery runs it before signing off a PR.

If you cannot name the single action you are about to take, you are not ready to
run the review — and not ready to act. Decompose until you can.

---

## The review format

Emit exactly this structure and log it with a stable ID. It is machine-readable
so the Vigilance Monitor and the Critic can read what you claimed and check it.

```yaml
SELF_LOGIC_REVIEW:
  id: SLR-<short-id>
  agent: <role running this review>
  action: <the single, atomic action about to be taken>
  charter_trace: <charter clause ID(s) this action serves>          # D5
  preconditions:
    - <precondition>: SATISFIED (<evidence>) | UNVERIFIED → HALT     # D1
  expected_output: <what this action will produce>
  success_condition: <the one checkable test for "this action done right">  # D4
  step_risks:                                                        # mini edge-case pass
    - <what could go wrong with THIS step>: HANDLED (<how>) | OPEN
  tripwire_scan: CLEAN | FIRED on "<word/phrase>" → <resolution>     # D1
  uncertainty: NONE | <describe the doubt>                           # D2
  verdict: PROCEED | HALT
  escalation: <if HALT, the exact question for the user/orchestrator; else "—">
```

---

## Field discipline

Each field maps to a doctrine. A field filled in lazily defeats the doctrine it
serves, so fill each one as if an adversary will check it — because one will.

- **`action`** — one atomic step. "Implement the sync" is not atomic;
  "implement the idempotent merge keyed on `user_id`" is. Non-atomic actions
  hide multiple decisions, each of which deserves its own review.
- **`charter_trace`** (D5) — if you cannot cite a charter clause, this action is
  orphan work. Orphan work is scope creep; **HALT** and escalate, do not
  proceed and "clean it up later."
- **`preconditions`** (D1) — list what must be true for this action to be valid,
  and mark each `SATISFIED` *with evidence* or `UNVERIFIED`. A single
  `UNVERIFIED` precondition forces `HALT`. You may not mark something satisfied
  because it "should be" — that word is itself a tripwire.
- **`success_condition`** (D4) — one objectively checkable test. If you cannot
  write a test for "done right," the action is not scoped as a contract and you
  cannot know when it succeeded.
- **`step_risks`** — a fast edge-case pass scoped to *this step*: empty input,
  concurrency, partial failure, the precondition silently changing under you.
  Any `OPEN` risk that is material forces `HALT`.
- **`tripwire_scan`** (D1) — read your own reasoning for *assume, probably, I
  think, should work, ought to, usually, most cases, typically*. If one fired,
  you must resolve it (prove it or escalate it) before the scan can read
  `CLEAN`.
- **`uncertainty`** (D2) — the honesty field. Any non-`NONE` value forces
  `HALT`. Stating uncertainty is not failure; proceeding through it is.

**`verdict: PROCEED`** is legal **only** when: every precondition is
`SATISFIED`, `charter_trace` is non-empty, `success_condition` is checkable,
no `step_risk` is `OPEN`-and-material, `tripwire_scan` is `CLEAN`, and
`uncertainty` is `NONE`. Anything else is `HALT`.

---

## Worked examples

**Example 1 — PROCEED (Actor about to implement a defined slice)**

```yaml
SELF_LOGIC_REVIEW:
  id: SLR-a3f
  agent: Actor
  action: Implement idempotent MERGE of `users` → `analytics.dim_users` keyed on user_id
  charter_trace: CH-2.1 (hourly idempotent merge), CH-2.3 (key = user_id)
  preconditions:
    - Source schema for `users` is frozen in charter: SATISFIED (CH-1.2 lists columns)
    - Target `dim_users` exists with matching keys: SATISFIED (verified in catalog)
    - Merge key user_id is unique at source: SATISFIED (CH-1.4 states PK constraint)
  expected_output: A merge job that upserts changed rows and is safe to re-run
  success_condition: Re-running the job on identical input produces zero row changes
  step_risks:
    - Late-arriving duplicate user_id within a batch: HANDLED (dedup on load, CH-2.4)
    - Null user_id in source: HANDLED (reject-to-quarantine per failure matrix CH-3.2)
  tripwire_scan: CLEAN
  uncertainty: NONE
  verdict: PROCEED
  escalation: "—"
```

**Example 2 — HALT (precondition unverified + tripwire fired)**

```yaml
SELF_LOGIC_REVIEW:
  id: SLR-b71
  agent: Actor
  action: Add a retry wrapper around the warehouse write
  charter_trace: CH-3.4 (warehouse-unavailable handling)
  preconditions:
    - Retry policy (count, backoff, idempotency) is specified: UNVERIFIED → HALT
    - The write is idempotent so retries are safe: SATISFIED (CH-2.1)
  expected_output: A bounded retry around the write
  success_condition: A transient failure is retried per policy; a permanent one alerts
  step_risks:
    - Retrying a non-idempotent write would double-write: HANDLED (write is idempotent)
    - Unbounded retries should usually be fine: OPEN  # <-- and note the tripwire below
  tripwire_scan: FIRED on "should usually be fine" → the retry bound is not actually defined
  uncertainty: Retry count and backoff are not in the charter; I would be inventing them
  verdict: HALT
  escalation: "Charter CH-3.4 names the failure but not the retry policy. Specify max
    retries, backoff strategy, and the boundary between transient and permanent — I
    will not invent these."
```

Note what Example 2 demonstrates: the agent *felt* like it could proceed ("should
usually be fine"), and the review is what converted that feeling into a precise,
answerable question instead of an invented policy quietly shipped to production.
That conversion is the entire value of D3.

---

## Logging and reuse

Every review is logged with its `id` and is addressable:

- The **Critic** reads the Actor's review logs to check whether the Actor's
  claimed preconditions and success conditions actually hold in the code — a
  gap between the review and the implementation is a high-value finding.
- The **Vigilance Monitor** scans review logs for `charter_trace` fields that
  point nowhere (orphan work) and for `HALT`s that were silently downgraded to
  `PROCEED`.
- The **Retrospective** mines `HALT` reviews to learn which classes of
  ambiguity keep slipping past the audit and into execution.

A review is never deleted. A superseded review is marked superseded with a
pointer to the one that replaced it, so the reasoning trail stays intact.
