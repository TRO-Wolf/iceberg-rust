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

# binding-manifest.md — SEPMO ⨉ iceberg-rust

**This is the one project-specific SEPMO file.** The spine ([SKILL.md](SKILL.md)) and its
[references/](references/) are portable canon and are never edited per-project; every
project-specific role and tunable resolves through the tables below. Instantiated from
[binding-manifest.template.md](binding-manifest.template.md) — re-instantiated 2026-07-13
against canon v2.2 (superseding the 2026-06-15 pre-ledger install). **A manifest binds; it
does not restate** — every row is a pointer into the repo or a declared fallback, never a
copy of a rule that lives somewhere else.

## Spine version

`spine_version:` **v2.2** — the canon version this manifest binds (spine frontmatter
`version: "2.2"`, changelog entry v2.2 — 2026-07-13). If the master spine moves past it,
this manifest is **stale, not silently wrong** (Invariant V staleness alarm): re-bind
before the next project starts.

## Precedence

On any conflict, the chain in [CLAUDE.md](../../CLAUDE.md) `<precedence>` wins; SEPMO
cedes the engineering contract to it. That home is the single statement of the chain —
this manifest points there and never restates it. Not cede-able under any chain: the
spine's own lifecycle law — the gates, the raise-only severity floor, the extend-only
taxonomy, and the artifact rule (every gate is a checkable artifact, not a self-report).

## Role bindings

| SEPMO role | Canonical home in this repo | Mode | Relationship |
|---|---|---|---|
| Engineering contract | the [skills/](../) tier manual for the running model (`Fable.md` / `Opus.md` / `Sonnet.md` / `Haiku.md`; index [skills/map.md](../map.md)) + [AGENTS.md](../../AGENTS.md) for crate code + the tool configs ([rustfmt.toml](../../rustfmt.toml), [rust-toolchain.toml](../../rust-toolchain.toml), the [Makefile](../../Makefile) lint targets) | BIND | Actor binds — defers entirely |
| Risk lens | `Opus.md` *Risk-First Mindset* (risk-surface table, double-execution rule, ToCToU windows) layered under the attack taxonomy ([references/05](references/05-critic.md)) + this manifest's `taxonomy_extensions` | BIND | Critic uses as attack basis |
| Done gate | tier manual §4 *Verification Before Done* + [docs/testing.md](../../docs/testing.md); the two named green tiers are bound in `green_commands` below | BIND | Delivery invokes; R2/R7 exits |
| Plan-of-record | [Roadmap.md](../../Roadmap.md) (phase plan + sequencing); the ranked open queue lives in [task/todo.md](../../task/todo.md) §ACTIVE per the "one home for PRIORITY" rule there | BIND | Orchestrator derives the charter |
| Status SSOT | [docs/parity/GAP_MATRIX.md](../../docs/parity/GAP_MATRIX.md) — capability status lives ONLY there; cite rows by permanent `R<id>` anchor; `make check-matrix-anchors` enforces | BIND | Delivery updates; never restated |
| PR-unit grouping | [task/todo.md](../../task/todo.md) unit sections under the two user-picked cadence modes (Mode A per-cycle scoped PR / Mode B bundled branch with interim + final Critic checkpoints); historical "Waves" are the same vocabulary | BIND-and-map | Orchestrator maps to it |
| Active plan tracking | [task/todo.md](../../task/todo.md) — exactly one tracker; archival lifecycle per [skills/compaction.md](../compaction.md) §Todo Archival | BIND | Orchestrator writes the plan here |
| Memory / lessons | [task/lessons.md](../../task/lessons.md) (active) + `task/lessons-archive/` + the lifecycle in [skills/compaction.md](../compaction.md) | BIND | Retrospective runs the learning pass |
| Navigation | the `map.md` convention — [CLAUDE.md](../../CLAUDE.md) `<map_md_navigation>`; the mandate extends to SEPMO's files ([map.md](map.md) here) | BIND | Mandate extends to SEPMO's files |
| Prohibitions | [CLAUDE.md](../../CLAUDE.md) *Absolute prohibitions* — SEPMO adds none | BIND | All agents obey |
| Sub-agent / tier policy | [CLAUDE.md](../../CLAUDE.md) `<subagent_policy>`: the per-PR **independent Critic (separate agent, fresh context) is non-negotiable**; SEPMO's FF pair realizes as **OO = Opus–Opus** by default; the Critic never drops below Opus on correctness review | BIND | Orchestrator's AC mode follows it (see `context_break_mechanics`) |
| Mode handling | the tier manuals' *Mode Handling* section (interactive vs. delegated) | BIND | Orchestrator + all agents adopt |
| Debugging protocol | `Opus.md` §8 *Debugging Protocol* + each mapped directory's `map.md` `## Debug` section | BIND | Actor/Critic follow on failure |

## SEPMO binding points (tunables)

Spine defaults apply wherever a row is silent. Per the spine: `severity_floor` may only
be **raised** and the taxonomy only **extended**. Changes to this section land as
versioned manifest updates proposed by a retrospective — including an **incident
retrospective** after an escaped defect (spine, *Incident retrospectives*). Feed-forward
is asymmetric: **bar-raising changes may land immediately**, stamped with date and
provenance; bar-lowering or neutral changes wait for the project boundary
([references/08](references/08-retrospective.md)). Canon changes are a different
procedure entirely (spine, *versioned canon*): proposed here, approved by the user,
landed at the master home.

| Binding point | This repo's value | Constraints / notes |
|---|---|---|
| `severity_floor` | **S2** — raised from the S1 default at re-instantiation, stamped 2026-07-13. Provenance: bar-preserving, not bar-raising in practice — the pre-v2.2 install already blocked convergence on its old MEDIUM+ scale, whose MEDIUM ("failure under uncommon-but-reachable conditions") maps to S2; binding S1 would have *lowered* the working bar. This is also the template's named expected raise for data-integrity-critical repos: a table-format library's product IS data integrity ([CLAUDE.md](../../CLAUDE.md) on-disk-format prohibition). Lands immediately per the asymmetric feed-forward rule (bar-raising never waits). | Raise-only; S2 ≥ S1 ✓. S3 (advisory) never blocks. |
| `green_commands` | **Unit gate (R2 exit):** `typos . && make check && make unit-test` — chained into the commit in ONE `&&` chain per [CLAUDE.md](../../CLAUDE.md) *Working conventions*. **Pre-merge gate (R7):** `typos . && make check && make check-msrv && cargo deny check advisories && make test` (docker-backed full workspace suite — mirrors CI's check/build/tests/msrv jobs, ci_typos.yml, and audit.yml's advisory gate). **CI-only exception record:** (1) *License-header check* (apache/skywalking-eyes, ci.yml) — no local make target; residual gap: a new file missing the ASF header greens locally and reds CI; mitigation: copy the header from a sibling at file creation. (2) *CodeQL* (codeql.yml) — GitHub-hosted semantic analysis, not locally runnable; residual gap: semantic security findings surface only post-push. (3) *Nightly interop* (nightly_interop.yml) — `make interop` mirrors it fully (48/48 local green proven 2026-07-11) but it is a post-merge nightly net, not a PR check; residual gap: an interop regression can merge and surface up to a day later; mitigation: run `make interop` (or a targeted `--only` subset) locally when a unit touches interop-bearing surfaces. (4) *AWS-credentialed integration* (Glue/S3 Tables) — in neither CI nor the local gate (needs real credentials); residual gap: real-catalog conformance is scheduled work with the user (todo queue item 6). **Parity guard: ABSENT — justified:** tool pins live in three homes (Makefile `cargo install` pins, CI action SHAs/versions, [rust-toolchain.toml](../../rust-toolchain.toml)) and live skew already exists (local typos-cli 1.47.2 vs CI v1.44.0 — the 2026-07-08 R1 false-positive incident); a pin-diff script is a named follow-up unit, and until it lands pin skew is an accepted residual recorded here (this row is its record, so the skip is not silent). | R7 mirror rule satisfied; every CI-enforced check is either in the pre-merge command or in the exception record with its residual gap. A silent skip is a binding defect (Invariant V). |
| `light_thresholds` | Spine defaults: **≤ 150 changed lines / ≤ 5 files**. Repo note: a unit that flips a GAP_MATRIX row or touches `crates/iceberg/src/spec/` can never pass criterion 5 (data-integrity surface) — LIGHT is effectively docs/CI-only here. | Size criterion only; all six spine criteria must hold; uncertain → STANDARD. |
| `context_break_mechanics` | **Sub-agent hard break (standing).** Every unit that ships as a PR gets an independent Critic — a separate agent with a fresh context ([CLAUDE.md](../../CLAUDE.md) `<subagent_policy>`; proven necessary 2026-06-25 when fresh Opus Critics caught a 3VL coverage gap every same-context pass missed). The Critic's dispatch package is restricted per R3(a): charter clauses, diff, artifacts, test results, taxonomy — never the Actor's narrative. Procedural in-session review is permitted only for work that never ships as a PR (the trivial-work fallback). | The hard break is R3's preferred form; nothing procedural to name honestly on the PR path. |
| `s0_fresh_execution` | **N/A — standing hard break** (the template's sanctioned justification: this row is mandatory only when `context_break_mechanics` binds a procedural break). Voluntary standing practice, recorded not as a substitute but because it exceeds the compensation rule: repo Critics are mutation-gated (each pin proven RED against a live sabotage with byte-identical restore) and the `dev/java-interop/` suites (48, discovered dynamically; nightly via `make interop`) are the standing cross-engine detector for silently-wrong-results classes. Masking surfaces, named for completeness: `inspect/` metadata tables and `Debug` impls are display paths — never sole evidence. | Bound N/A per the template rule; re-bind as a full row if the sub-agent policy ever admits a procedural break on the PR path. |
| `metrics_ledger_location` | [task/sepmo-metrics.md](../../task/sepmo-metrics.md) — CREATED 2026-07-13 with the [references/08](references/08-retrospective.md) metric set, including `environment_drift_events`. One section per retrospective (charter-close or incident). | Exists ✓; first populated section lands with the first v2.2-era retrospective. |
| `taxonomy_extensions` | Two, added 2026-07-13 (extend-only): **`java-parity`** — behavioral divergence from Java `iceberg-core`/`iceberg-api` 1.10.0, attacked against the reference checkout source or jar bytecode (the [CLAUDE.md](../../CLAUDE.md) *Parity mandate* as attack duty); **`format-stability`** — an on-disk format break (the *Absolute prohibitions* format rule as attack duty). Provenance: both were de facto attack categories in every prior Critic run (bytecode-verified citations, format-break sweeps); binding them makes the attestation duty explicit and non-skippable. | Pure additions to the ref-05 canonical taxonomy; they widen every subsequent Critic's attestation duty — deliberate. |

## Pointers

- **Up:** repo root [CLAUDE.md](../../CLAUDE.md) (read-order + precedence);
  [skills/map.md](../map.md) (the skills index).
- **Related:** [SKILL.md](SKILL.md) (the spine, canon v2.2);
  [references/](references/) (canonical instrument homes);
  [binding-manifest.template.md](binding-manifest.template.md) (the portable template
  this file instantiates); the bound repo docs: [Roadmap.md](../../Roadmap.md),
  [docs/parity/GAP_MATRIX.md](../../docs/parity/GAP_MATRIX.md),
  [docs/testing.md](../../docs/testing.md), [task/todo.md](../../task/todo.md),
  [task/lessons.md](../../task/lessons.md),
  [task/sepmo-metrics.md](../../task/sepmo-metrics.md).

## Debug

- A SEPMO behavior contradicts the engineering contract → the contract wins on
  engineering; fix this manifest or its usage, never canon. Canon defects are filed
  upstream to the master home (D2), not patched in this repo's copy. **Filed 2026-07-13:**
  the v2.2 spine's *Model assumption* section carries a "For **this** repo that resolves
  to a **single-agent default**" sentence — an instantiation artifact inside portable
  canon. It does not bind here: this manifest's *Sub-agent / tier policy* row and
  `context_break_mechanics` (the homes the spine itself delegates to) bind the
  independent-Critic hard break, and [CLAUDE.md](../../CLAUDE.md) wins regardless.
- The same status appears in two places → SSOT breach; the Status SSOT row is the only
  home.
- A row points at a missing file → this manifest is stale; fix the row.
- The AC loop seems to need sub-agents → here it genuinely does on the PR path: the
  hard break is the bound mechanic (non-negotiable per the sub-agent policy row); the
  procedural fallback exists only for work that never ships as a PR.
- A claim ("100/100", "converged", "mergeable", "delivered") appears without its
  artifact → Invariant V alarm ([references/06](references/06-vigilance.md)); demand the
  ledger / attestation / gate evidence.
- A converged unit goes red at the pre-merge gate or in CI → run the **R10 base-ref
  test**: reproduce on the base ref without the unit's diff. Base red → environmental;
  remediate as its own unit (path per the rubric). Base green → a unit defect; T9.
  (Live example: the 2026-07-11 first nightly-interop failure with `make interop` green
  on the same ref locally — environmental, recorded as an `environment_drift_event`.)
- A defect surfaces **after its PR was accepted** → incident retrospective **now**
  (spine, *Incident retrospectives*): file `coverage_misses` naming the category that
  sat clean, file `escaped_defects_by_origin`, land bar-raising proposals immediately
  with a stamp; the fix is its own unit with R5 regression proof.
- This manifest's `spine_version` trails the master spine → staleness alarm; re-bind
  before the next project starts.
- Unsure which manual is "the engineering contract" right now → the one matching the
  running model tier (Mythos-class → `Fable.md`); if unknown, `Opus.md`.

## Instantiation checklist

The install is complete when every proposition below is provable:

- [x] I-1 — Every role row resolves to BIND (thirteen rows, all BIND — every home
      pre-existed; one BIND-and-map for PR-unit grouping); no italic guidance and no
      `> Fill:` lines remain.
- [x] I-2 — `spine_version: v2.2` matches the shipped spine's frontmatter
      (`version: "2.2"`) and its changelog head entry (v2.2 — 2026-07-13).
- [x] I-3 — Every path this manifest links resolves (link-checked at instantiation;
      `task/sepmo-metrics.md` created in the same change).
- [x] I-4 — `green_commands` names both gates; the four CI-only exceptions
      (license-header, CodeQL, nightly interop, credentialed integration) each state
      their residual gap.
- [x] I-5 — No parity guard exists; its absence is justified inside the
      `green_commands` row (three pin homes, live typos skew named, follow-up unit
      recorded there).
- [x] I-6 — [task/sepmo-metrics.md](../../task/sepmo-metrics.md) exists and carries the
      references/08 metric set, including `environment_drift_events`.
- [x] I-7 — `severity_floor: S2` ≥ S1; taxonomy changes are pure additions
      (`java-parity`, `format-stability`).
- [x] I-8 — Status sweep: capability status has exactly one home
      ([docs/parity/GAP_MATRIX.md](../../docs/parity/GAP_MATRIX.md), the SSOT row);
      SEPMO files restate no cell.
- [x] I-9 — This manifest re-instantiation ran as a SEPMO unit
      (`infra/sepmo-canon-v2.2`, tracked in [task/todo.md](../../task/todo.md)
      §2026-07-13): rubric recorded **STANDARD** (criteria 1 and 3 fail LIGHT —
      multi-file governance surface, > 150 lines), one AC cycle with the independent
      fresh-context Critic per the sub-agent policy row.
- [x] I-10 — `s0_fresh_execution` is marked N/A with the standing sub-agent hard break
      as the justification (the `context_break_mechanics` row binds it).
