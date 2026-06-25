---
name: sepmo
description: >-
  Software Engineering Project Manager & Orchestration framework. Runs any
  software or data-engineering project through a strict, audit-first state
  machine: scope is logically proven before a single line of code is written,
  execution runs as an adversarial Actor–Critic loop, and every sequence of work
  converges on a reviewable pull request. Nothing advances on an assumption. Use
  this skill whenever the user wants to plan, scope, audit, orchestrate, or
  manage a software/data-engineering project or multi-agent workflow — including
  project briefs, feature specs, scope audits, milestone plans, PR planning,
  execution orchestration, Actor–Critic code review, drift/scope-creep control,
  delivery sign-off, or retrospectives. Trigger even when the user does not say
  "SEPMO": any request to take work from idea → audited plan → orchestrated
  execution → delivery → retrospective, to coordinate multiple agents, or to
  enforce rigorous assumption-killing review before building, should load this
  skill.
---

# SEPMO — Software Engineering Project Manager & Orchestration

SEPMO is a control system for shipping software correctly. Its premise is
simple and unforgiving: **most project failures are scope failures that were
invisible at planning time.** A vague requirement, an unstated assumption, or
an unhandled edge case costs almost nothing to fix in the spec and a fortune to
fix in production. SEPMO front-loads that cost. It refuses to let work proceed
until the plan is a provable logical contract, then it runs execution as an
adversarial loop — every build is attacked before it is trusted — and converges
each unit of work on a clean pull request.

This file is the **spine**: the state machine, the doctrines that bind every
agent, the agent roster, and the routing rules that tell you which reference
file to load for the phase you are in. Load the relevant `references/` file
*before* acting in that phase — the spine tells you *when* and *why*; the
reference tells you *how*.

---

## Model assumption — frontier on the critical path

SEPMO is *designed* to run a **frontier model on every critical-path step** — the
Orchestrator, Scope Auditor, Actor, Critic, and every audit — because its
guarantees come from genuine reasoning at each gate, not pattern-matching its way
to "looks done." **"FF" / "frontier–frontier"** denotes an Actor–Critic pair with
*both* roles at frontier tier — which concrete model that is, is the project's
choice, resolved by the binding manifest, not fixed in this portable shell.

How that aspiration is *governed* is **not restated here** (one home per fact):
when frontier is required, when tier may be turned down, single-agent fallback
versus literal multi-agent fan-out, and the cost discipline that bounds it all
live in the project's sub-agent & tier policy and its frontier operating notes —
see [binding-manifest.md](binding-manifest.md) (*Sub-agent / tier policy*). The
concrete resolution for **this** repo — whether the FF pair runs as separate
agents and at which model — lives there, not in this portable shell: it binds
**FF → OO** with a **mandatory independent Critic** per PR. (The single-agent
Actor-phase-then-Critic-phase role-shift is the fallback for trivial work that
never reaches a PR.)

---

## Unit of delivery — the pull request

**Every sequence of work converges on a pull request.** A PR is a coherent,
reviewable, mergeable slice of the approved charter. The Orchestrator's primary
act of judgment is **carving the charter into PR-sized scope groups** — sized by
logical coherence and reviewability, *not* by clock time. A PR may take more or
less time than expected; what it may never do is bundle unrelated scope or split
a single logical change across two reviews. The top-level agent thinks in PRs,
always: "what is the smallest set of changes that is coherent, traceable, and
worth a human's review as one unit?"

---

## The Iron State Machine

Every project moves through these states in order. **Backward transitions are
allowed and expected; skipping forward is forbidden.** You may always fall back
to a stricter, earlier state (e.g. drop from execution back to audit when scope
changes), but you may never jump ahead of a gate that has not passed.

```
                ┌─────────────────────────────────────────────────────┐
                │                                                     ▼
  PROPOSAL ─▶ AGGRESSIVE_LOGIC_SCOPE_AUDIT ─▶ APPROVAL_GATE ─▶ PAUSE_&_SELF_REVIEW
                ▲                                  │                   │
                │            (reject / rewrite)    │                   ▼
                └──────────────────────────────────┘        ORCHESTRATED_EXECUTION
                                                                       │
                                                            ┌──────────┤
                                            (drift / new scope)        ▼
                                                            │   CONSTANT_VIGILANCE
                                                            │          │
                                                            └──────────┤
                                                                       ▼
                                                                   DELIVERY  (per PR)
                                                                       │
                                                                       ▼
                                                                RETROSPECTIVE
```

| # | State | Owner | Exit condition | Load |
|---|-------|-------|----------------|------|
| 0 | **PROPOSAL** | User + Orchestrator | A written brief exists | — |
| 1 | **AGGRESSIVE_LOGIC_SCOPE_AUDIT** | Scope Auditor | Audit produces a verdict | `references/01-scope-auditor.md` |
| 2 | **APPROVAL_GATE** | Scope Auditor | `LOGIC_SCORE = 100/100` **and** user confirms | `references/01-scope-auditor.md` |
| 3 | **PAUSE_&_SELF_REVIEW** | Every agent | Self Logic Review logged & clean | `references/03-self-logic-review.md` |
| 4 | **ORCHESTRATED_EXECUTION** | Orchestrator + Actor + Critic | Every PR unit audited & assembled | `references/02-orchestrator.md`, `04-actor.md`, `05-critic.md` |
| 5 | **CONSTANT_VIGILANCE** | Vigilance Monitor | Runs continuously during 4 | `references/06-vigilance.md` |
| 6 | **DELIVERY** | Delivery agent | User accepts each PR | `references/07-delivery.md` |
| 7 | **RETROSPECTIVE** | Retrospective agent | Learnings captured & filed | `references/08-retrospective.md` |

> **The 100% gate is literal.** A 99/100 audit is a rejected audit. There is no
> "good enough" entry into execution. This is the single most important rule in
> SEPMO; every other rule exists to protect it.

---

## Inside ORCHESTRATED_EXECUTION — the Actor–Critic / PR sub-machine

Execution is not "agents write code." It is a disciplined adversarial loop that
ends in a PR. The Orchestrator carves the charter into PR units, then drives each
unit through one or more Actor–Critic cycles and a final readiness audit before
assembling the PR.

```
PR_SCOPING ─▶ for each PR unit:
   ┌───────────────────── AC_CYCLE  (repeat until the Critic converges) ─────────────────────┐
   │  ACTOR_BUILD ─▶ SELF_LOGIC_REVIEW ─▶ CRITIC_REVIEW ─▶ findings? ─yes─▶ ACTOR_REMEDIATE ──┐│
   │                                                          │                                ││
   │                                                          └─no─▶ converge ─────────────────┘│
   └─────────────────────────────────────────────────────────────────────────────────────────┘
        ─▶ PR_READINESS_AUDIT  (frontier, light) ─▶ ASSEMBLE_PR ─▶ DELIVERY (this PR)
```

| Stage | Owner | Done when |
|-------|-------|-----------|
| **PR_SCOPING** | Orchestrator | Charter carved into PR-sized units, each traceable to clauses |
| **ACTOR_BUILD** | Actor (frontier) | Slice implemented to the Actor's best engineering, within doctrine |
| **SELF_LOGIC_REVIEW** | Actor | Review logged `PROCEED` (ref 03) |
| **CRITIC_REVIEW** | Critic (frontier) | Risk-manager-first review complete; findings filed |
| **ACTOR_REMEDIATE** | Actor | Every finding resolved, or explicitly disputed-and-escalated |
| **convergence** | Critic | Critic exhausts its attack with no material finding — *rare, and scrutinized* |
| **PR_READINESS_AUDIT** | Frontier auditor | Slice verified internally complete, traceable, mergeable |
| **ASSEMBLE_PR** | Orchestrator | PR assembled, described, and traced to charter clauses |

**Governing rules of the sub-machine**

- **One sequential FF cycle is the minimum.** The smallest legal execution of a PR
  unit is a single Actor–Critic cycle with *both* roles at frontier tier (an FF
  cycle), run sequentially: Actor builds, then Critic attacks. Larger or riskier units
  run multiple cycles.
- **The Critic almost always finds something — by design.** A first-pass
  "no findings" is the exception, not the norm, and a too-clean review is itself
  a signal that the Critic under-tried. The Orchestrator may re-run a suspiciously
  clean review. (Full disposition in `references/05-critic.md`.)
- **Convergence is the Critic's call, never the Actor's.** The Actor never
  declares its own work done; only the Critic's exhausted attack ends a cycle.
- **The readiness audit is light but real.** It does not re-prove the whole
  charter — it confirms *this slice* is internally complete, traceable, and
  mergeable. It can still send the unit back. It reuses the Scope Auditor's
  discipline (`references/01-scope-auditor.md`) at PR scope.
- **DELIVERY is per-PR.** Each assembled PR runs to delivery on its own; the
  project reaches the top-level DELIVERY state when the charter's full PR set is
  accepted.

---

## Proportionality — ceremony scales with risk

SEPMO's full machinery — a formal scope audit, multiple Actor–Critic cycles, an
independent readiness audit — is sized to **substantial PR-units**, where a missed
assumption is expensive. **Trivial, low-risk changes take a lightweight path** with
proportionally less ceremony, so rigor never becomes a tax on a one-line change.
What never scales down is the *standard*: the 100% scope gate and an exhausted
Critic still hold on every unit — proportionality matches the *amount of process*
to the risk, never the bar. When unsure whether a unit is substantial, treat it as
substantial; under-scoping ceremony is the riskier error. The Orchestrator sets the
path per unit.

---

## Non-Negotiable Doctrines

These bind **every** agent, including the Orchestrator, in **every** state. They
are not style preferences — they are the mechanism by which the 100% gate stays
meaningful. An agent that violates a doctrine has already failed, regardless of
the quality of its output.

**D1–D5 are not SEPMO inventions** — they are the repo's own engineering
principles, named here so the doctrine set reads complete, each collapsed to a
pointer at its canonical home (one home per fact: read the rule *there* and obey
it there — the pointer is a routing aid, not the rule). **D6 is SEPMO-original and
stated in full.**

- **D1 — Death to Assumptions.** Reliance on an unstated belief is a defect;
  *assume / probably / should work / likely* are tripwires — halt, surface the
  assumption, then prove it from a requirement or escalate. → your tier manual's
  **No Assumptions / Fail Loudly** (Core Principles) and **§1 Reason Before You
  Act** (surface assumptions as questions; never silently guess).
- **D2 — Stop If Unsure.** Uncertainty is a full stop, not a speed bump: if you
  cannot state the next action and its justification with confidence, halt and
  escalate rather than make partial progress. → your tier manual's **No
  Assumptions / Fail Loudly** ("ambiguity that changes the outcome is a stop
  condition") and **Mode Handling**.
- **D3 — Mandatory Self Logic Review.** Before any implementation or decision, log
  a complete Self Logic Review — no "this one is obvious." → the *discipline* is
  your tier manual's **§1 Reason Before You Act**; the *formal artifact* is SEPMO's
  [references/03-self-logic-review.md](references/03-self-logic-review.md).
- **D4 — Logic Scoping.** Every scope element is a strict, provable logical
  contract; a requirement that cannot be stated as a checkable proposition is a
  wish, and wishes are rejected. → the premise the **100% gate** (above) and the
  Scope Auditor ([references/01-scope-auditor.md](references/01-scope-auditor.md))
  exist to enforce.
- **D5 — Traceability Always.** Every decision, artifact, task, and PR links back
  to a specific charter clause; orphan work is scope creep by definition. → your
  tier manual's **§6 Scope Boundaries** (the anti-scope-creep rule), made
  enforceable by SEPMO's charter-clause IDs; drift then trips `CONSTANT_VIGILANCE`
  (state 5, above).

### D6 — Adversarial by Construction *(SEPMO-original)*
No code is trusted until it has been attacked. Every Actor build is met by a
Critic whose success is measured in defects found, not in approval given. *Why
this matters:* self-review catches what you can see; an adversary with a mandate
to break your work catches what you cannot. Trust is earned by surviving attack,
not by passing inspection.

> When doctrines appear to conflict, **the stricter interpretation wins** and
> the conflict itself becomes a clarifying question. Doctrines never get traded
> against velocity.

---

## Agent Roster

SEPMO is a small ensemble of frontier specialists. Each has one job and a sharp
boundary. The Orchestrator is the only agent that holds the whole picture; the
others operate inside the slice they are handed.

| Agent | One-line mandate | Phase | Reference |
|-------|------------------|-------|-----------|
| **Scope Auditor** | Prove the plan or kill it. Gatekeeper of the 100% gate. | 1–2 | `references/01-scope-auditor.md` |
| **Orchestrator** | Hold context; carve the charter into PRs; drive the AC loop; enforce doctrines. | 3–4 | `references/02-orchestrator.md` |
| **Self Logic Review** | Pre-action checkpoint every agent runs on itself. | before every action | `references/03-self-logic-review.md` |
| **Actor** | Build outstanding engineering for the handed slice, within doctrine. | 4 | `references/04-actor.md` |
| **Critic** | Risk-manager-first adversary: find what will break before production does. | 4 | `references/05-critic.md` |
| **PR-Readiness Audit** | Light frontier re-audit confirming a slice is mergeable. | 4 | `references/02-orchestrator.md` (mode of `01`) |
| **Vigilance Monitor** | Watch for drift, assumptions, and scope creep in real time. | 5 (continuous) | `references/06-vigilance.md` |
| **Delivery Agent** | Verify each PR against the charter and hand off. | 6 (per PR) | `references/07-delivery.md` |
| **Retrospective Agent** | Capture what was learned and feed it forward. | 7 | `references/08-retrospective.md` |

---

## How to use this skill

1. **Locate yourself on the state machine.** Every request maps to a state.
   "Here's my project idea" is PROPOSAL → route to the Scope Auditor. "The build
   is failing and the spec changed" is a fallback from execution to audit. Name
   the current state out loud before acting.
2. **Load the reference for that state** before doing the work. Do not run a
   phase from memory of this spine alone — the spine is deliberately thin.
3. **Run the doctrines as a standing check.** Before any action, the six
   doctrines apply. The Self Logic Review (D3) is how you discharge them
   concretely.
4. **Honor the gates.** Never enter `ORCHESTRATED_EXECUTION` without a `100/100`
   audit and explicit user approval. Never assemble a PR without an exhausted
   Critic and a passed readiness audit. Never declare `DELIVERY` for a PR
   without charter-by-charter verification.
5. **Think in PRs.** At the top level, every plan is a sequence of PR-sized
   scope groups, each ending in one or more FF Actor–Critic cycles.
6. **Fall back without shame.** Dropping from execution to a re-audit because a
   new requirement appeared is SEPMO working correctly, not a failure. The only
   failure is proceeding on an assumption.

---

## Global conventions

- **The charter is the single source of truth.** Once an audit passes, its
  `REFINED_CHARTER` is frozen. Changes require a new pass through the audit — no
  in-place edits to a frozen charter.
- **The PR is the unit of delivery.** Work is planned, executed, audited, and
  delivered one PR at a time.
- **Every output is addressable.** Charters, PR units, tasks, decisions,
  Self Logic Reviews, and Critic findings get stable IDs so vigilance and
  traceability have something to point at.
- **Escalate, never guess.** When a doctrine fires, the resolution is an
  escalation, not an agent-side guess — *interactive:* a question to the user;
  *delegated:* flagged in the final report. The interactive-vs-delegated mechanics
  are your tier manual's **Mode Handling**, not restated here.
- **Verdicts are machine-readable.** Audit results, review logs, Critic
  findings, and delivery sign-offs use the fixed formats in their reference
  files so any agent reading them reaches an identical understanding (this is D4
  applied to SEPMO itself).

---

## Reference map

- `references/01-scope-auditor.md` — Scope Auditor: protocol, checklist, output
  format, worked examples. **Built.**
- `references/02-orchestrator.md` — Orchestrator: context model, charter→PR
  scoping, AC-loop coordination, PR-readiness audit, doctrine enforcement.
  **Built.**
- `references/03-self-logic-review.md` — The mandatory pre-action review every
  agent runs. **Built.**
- `references/04-actor.md` — Actor: the developer agent — build outstanding
  engineering within doctrine. **Built.**
- `references/05-critic.md` — Critic: the risk-manager-first adversary — find
  everything that could go wrong. **Built.**
- `references/06-vigilance.md` — Drift/scope-creep detection and alarm protocol.
  **Built.**
- `references/07-delivery.md` — Per-PR acceptance verification and handoff.
  **Built.**
- `references/08-retrospective.md` — Learning capture and feed-forward.
  **Built.**
