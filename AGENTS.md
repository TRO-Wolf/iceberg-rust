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

# AGENTS.md — the authoritative contract

# EFFECTIVE IMMEDIATELY — ALL ANTHROPIC MODELS ARE BANNED FROM WRITING COMMENTS IN CODE

**THIS INCLUDES FABLE, OPUS, SONNET AND HAIKU.**

- **IF YOU ARE FABLE, DO NOT WRITE ANY COMMENTS.**
- **IF YOU ARE OPUS, DO NOT WRITE ANY COMMENTS.**
- **IF YOU ARE SONNET, DO NOT WRITE ANY COMMENTS.**
- **IF YOU ARE HAIKU, DO NOT WRITE ANY COMMENTS.**

**THIS INCLUDES ANY MODEL VERSION.** Opus 4.8 or Opus 5 — either one is banned. It does not matter.

This rule outranks every other instruction in this repository about comments, including
"Comments and prose" below. A human contributor may write comments. An Anthropic model may not.

## The only exceptions

These three are the whole list. Nothing else is an exception, and "it would be helpful here" is not
one.

1. **Doc comments the compiler demands.** `crates/iceberg/src/lib.rs` sets
   `#![deny(missing_docs)]`, so a public item without a doc comment does not build.
   **WRITE THE MINIMUM THAT COMPILES. ONE LINE STATING WHAT THE ITEM IS. NOT A PARAGRAPH. NOT A
   RATIONALE. NOT AN EXAMPLE. ONE LINE.**
2. **The ASF license header.** It is a comment, CI enforces it (skywalking-eyes,
   `.licenserc.yaml`), and a file without it fails. Copy it verbatim from a sibling file.
3. **Markdown.** Minor comments and ordinary prose in `.md` files are allowed. The ban is about
   comments in CODE. `docs/parity/GAP_MATRIX.md`, the `task/` ledgers and the `map.md` files are
   where explanation belongs, and routing evidence there is still required.

Everything else stays banned. If a fact seems to need a code comment, it belongs in a `task/`
ledger or a matrix row instead — that routing rule is unchanged and is now the only route.


This is the **single authoritative contract** for this repository, written for **any contributor —
human or automated agent**, naming no tool or model. It holds the read order, the precedence chain,
the parity mandate, the navigation contract, the project snapshot and architecture, the build/test
commands, the absolute prohibitions, the engineering rules, and the working conventions. When a rule
changes, it changes **here**; other files point at this contract, they do not restate it.

Tool-specific onboarding lives in clearly-labelled **adapter** files that carry no authoritative
facts (so they cannot drift): [CLAUDE.md](CLAUDE.md) and [.agents/](.agents/map.md). Deleting any
adapter loses no project knowledge.

This is an **owned fork of [Apache Iceberg™ Rust](https://github.com/apache/iceberg-rust)** — the Rust
implementation of the [Apache Iceberg](https://iceberg.apache.org/) open table format. We maintain it to
reach **1:1 capability parity with the Java `iceberg-core` / `iceberg-api` library** (the engine-agnostic
table-format core, **not** the Spark engine surface). Upstream `apache/iceberg-rust` is a **sync baseline
we cherry-pick from, not a mergeability constraint** — we diverge freely in service of parity. The
deliverable is a **Rust-native library**; Python / PySpark is deferred (there is no Python layer in this
repo). **Glue + S3 Tables** are the first-priority catalogs.

The authoritative plan is **[Roadmap.md](Roadmap.md)** (phase plan + sequencing); the living capability
checklist is **[docs/parity/GAP_MATRIX.md](docs/parity/GAP_MATRIX.md)**. When this file and the Roadmap
disagree on direction, the **Roadmap wins**; when the Roadmap and the GAP_MATRIX disagree on a
capability's *status*, the **GAP_MATRIX** (re-audited against the live base) wins.

> **A note on the XML tags in this file.** A few sections are wrapped in semantic tags
> (`<read_order>`, `<precedence>`, `<map_md_navigation>`, `<subagent_policy>`). They mark the
> load-bearing "do this / don't skip this" regions so an agent can locate them unambiguously; they
> carry no meaning beyond "read this bounded region as a unit." Reference sections (snapshot,
> architecture, build/test, repo layout) are intentionally left untagged.

<read_order>

## Read order (every session)

1. **This file (AGENTS.md)** — repository intent, precedence, prohibitions, the engineering rules,
   and the navigation contract. Everything below is the contract; read it before touching anything.
2. **[Roadmap.md](Roadmap.md)** — the parity phase plan and the current phase; then
   **[docs/parity/GAP_MATRIX.md](docs/parity/GAP_MATRIX.md)** for per-capability status.
3. **The engineering method** —
   [.agents/skills/engineering-method/SKILL.md](.agents/skills/engineering-method/SKILL.md), the
   portable agent-agnostic working method (one instruction set for every model tier; it replaced
   the per-tier manuals 2026-08-24); precedence is in `<precedence>` below.
4. **The SEPMO control plane** — [skills/sepmo/SKILL.md](skills/sepmo/SKILL.md) (the
   lifecycle/governance shell: scope audit + 100% gate, Actor–Critic execution, PR-grouping,
   delivery, retrospective) and [skills/sepmo/binding-manifest.md](skills/sepmo/binding-manifest.md)
   (how it binds to this repo). SEPMO governs *lifecycle*; this file and the engineering method win
   the *engineering contract* and all conflicts (see `<precedence>`).
5. **[task/lessons.md](task/lessons.md) in full, then [task/todo.md](task/todo.md)** — accumulated
   lessons and any mid-flight plan to pick up.
6. **The `map.md` of every directory your task will touch** (where present — see the navigation
   rule below).
7. **Upstream docs when you need depth:** [README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md),
   the per-crate `README.md` files, and the [Iceberg Rust site](https://rust.iceberg.apache.org/).

**Before you write or review any Rust, doc comment, or markdown prose**, load
[.agents/skills/rust-code-quality/SKILL.md](.agents/skills/rust-code-quality/SKILL.md). It carries
the review sequence for what the armed gates cannot catch, and the comment-discipline pass for
"Comments and prose" below. Every Actor and every Critic in this repository loads it. Its evidence
companion is [.agents/skills/test-adequacy/SKILL.md](.agents/skills/test-adequacy/SKILL.md).

Your tool's adapter ([CLAUDE.md](CLAUDE.md), [.agents/](.agents/map.md)) may add tool mechanics on
top of this order; it never replaces a step.

</read_order>

<precedence>

## Precedence — who wins on conflict

One chain, highest authority first:

1. **AGENTS.md** (this file) — repo intent, prohibitions, precedence, the engineering contract, the
   navigation contract.
2. **Roadmap.md** — the plan and the current phase. *(Direction-vs-status nuance: the Roadmap owns
   direction, the GAP_MATRIX owns capability status — see the intro paragraph above.)*
3. **docs/parity/GAP_MATRIX.md** — the single source of truth for capability *status*.
4. **The engineering method**
   ([.agents/skills/engineering-method/](.agents/skills/engineering-method/map.md)) — the portable
   engineering defaults; this file is repo-specific and wins over it.
5. **SEPMO** ([skills/sepmo/](skills/sepmo/)) — **lifecycle and orchestration only**.

SEPMO **cedes the engineering contract** to this file and to the manuals; this file and the manuals
**cede lifecycle/orchestration** (the scope audit, the gates, the Actor–Critic loop, PR-grouping,
delivery, retrospective) to SEPMO. When a SEPMO rule conflicts with a higher item, the higher item
wins and SEPMO is corrected.

**The adapters are not in the chain.** [CLAUDE.md](CLAUDE.md) and [.agents/](.agents/map.md) hold
tool mechanics only and state no project rule, so they cannot conflict with this chain. A rule found
in an adapter is a bug: move it here and leave a pointer.

</precedence>

## Parity mandate

The north star is behavioral 1:1 parity with Java `iceberg-core` / `iceberg-api`. Concretely:

- **The Java repo is the spec-by-example.** Keep a reference checkout of `apache/iceberg` and re-crawl on
  each Java release. A capability is "done" only when the Rust API matches the Java contract's behavior.
- **Tests land with the code, in the same change.** Behavior added without tests is a hard block.
- **Interop is the only true 1:1 evidence.** Where applicable, prove byte-level round-trips: read tables
  Java wrote, and prove Java reads what we write. A GAP_MATRIX row flips to ✅ only with unit tests **and**
  an interop test.
- **Re-audit the GAP_MATRIX after every upstream sync and every phase**, and date-stamp the provenance.
- **Order by dependency, then value:** metadata correctness underpins writes; writes underpin maintenance.

<map_md_navigation>

## `map.md` navigation — a convention this fork adopts

This fork uses a guiding-agent navigation pattern: a directory may carry a single `map.md`
documenting what lives there and where to go next. **It is opt-in and incremental** — upstream
Iceberg Rust does not ship `map.md` files, so coverage grows as you work, not all at once.

Each `map.md` has two parts in one file:

- **The map** (top) — `Purpose`, `Contents`, an `I want to... → go to` intent table, and
  `Pointers` (Up / Related) to neighboring directories.
- **`## Debug`** (bottom) — `Known failure modes` table, `First checks`, and `Escalate to` pointers.

**The contract:**

- **Before reading or editing a file in a directory that has a `map.md`,** open the `map.md` first
  and use it to navigate. The maps are the index; the code is the truth.
- **If the code and a `map.md` disagree, the code wins** — the `map.md` is stale.
- **When your change makes a directory's `map.md` inaccurate, update it in the same change**
  (always in scope, even though §6 of the manuals otherwise forbids touching unplanned files).
- **When you create a new source directory, add its `map.md` in the same change** — but only if the
  surrounding tree already uses the convention. Do not litter `map.md` files across pristine
  upstream directories you are only reading.

</map_md_navigation>

## Project snapshot

Apache Iceberg Rust implements the **Iceberg table format spec** in Rust: reading and writing table
metadata and data, expression/predicate handling, partition transforms, snapshot and schema
evolution, and pluggable catalogs and object storage. It is a **library workspace**, not an
application — most code is library crates consumed by downstream projects. **Rust** edition 2024, MSRV
**1.94** (see [Cargo.toml](Cargo.toml) `rust-version`). Base synced to upstream **0.9.1**; the
dependency family was bumped to **datafusion 54.1 / arrow 58.4 / parquet 58.4** on 2026-08-05
(`orc-rust` 0.8; MSRV 1.92 → 1.94).

## Big-picture architecture

### The workspace crates

| Crate | Path | Role |
|---|---|---|
| **iceberg** | [crates/iceberg/](crates/iceberg/) | The core: spec types, catalog trait, table scans, transactions, writers, Arrow/Avro/Parquet IO, expressions, partition transforms, metadata inspection, Puffin, deletion vectors. |
| **iceberg-datafusion** | [crates/integrations/datafusion/](crates/integrations/datafusion/) | DataFusion integration — `TableProvider` / `CatalogProvider` / physical plans so Iceberg tables are queryable from DataFusion SQL. |
| **catalog/{rest,hms,glue,s3tables,sql}** | [crates/catalog/](crates/catalog/) | Concrete `Catalog` implementations: REST, Hive Metastore, AWS Glue, S3 Tables, and SQL-backed. **Glue + S3 Tables are the parity priority.** |
| **catalog/loader** | [crates/catalog/loader/](crates/catalog/loader/) | Config-driven catalog construction (pick a catalog impl at runtime). |
| **storage/opendal** | [crates/storage/opendal/](crates/storage/opendal/) | OpenDAL-backed FileIO storage (extracted from the core in the 0.8/0.9 cycle). |
| **integrations/cache-moka** | [crates/integrations/cache-moka/](crates/integrations/cache-moka/) | Moka-backed object/metadata cache. |
| **integrations/playground** | [crates/integrations/playground/](crates/integrations/playground/) | `iceberg-playground` — scratch crate for experimentation. |
| **examples, sqllogictest, test_utils, integration_tests** | [crates/](crates/) | Runnable examples, SQL logic tests, shared test helpers, end-to-end integration suites. |

### Inside the `iceberg` crate

```
crates/iceberg/src/
├── spec/         table/manifest/schema/snapshot/partition/view metadata types (the on-disk format)
├── catalog/      the Catalog trait + table/view identifiers + creation/update types
├── scan/         table scan planning → Arrow record batches
├── transaction/  atomic metadata updates (append, sort-order, properties, location, statistics,
│                 upgrade-format-version) + the TransactionAction / ApplyTransactionAction seam
├── writer/       data + equality-delete writers, file/rolling/partitioning writers
├── arrow/        Arrow ⇄ Iceberg schema/value conversion + merge-on-read delete application
├── avro/         Avro encoding for manifests/metadata
├── io/           object storage abstraction (FileIO; OpenDAL impl in crates/storage/opendal)
├── expr/         predicate / boolean expression trees + binding + visitors
├── transform/    partition transforms (identity, bucket, truncate, year/month/day/hour, void)
├── inspect/      metadata tables (snapshots, manifests — more variants are a parity gap)
├── puffin/       Puffin file format (stats / deletion vectors)
├── delete_vector.rs / delete_file_index.rs   merge-on-read delete handling
└── metadata_columns.rs                        reserved metadata columns (_file, _pos, ...)
```

Patterns to internalize: **the spec module is the source of truth** for the on-disk format —
changes there ripple through every reader and writer. **Catalogs are pluggable** behind one trait;
**FileIO is pluggable** behind OpenDAL. **Arrow is the in-memory currency** — scans produce Arrow,
writers consume it. **Transactions extend via `TransactionAction`** (`transaction/action.rs`); the
trait is currently `pub(crate)` — since we own this fork, opening it is the sanctioned path to new
write actions in Phase 2 (see [Roadmap.md](Roadmap.md)).

## Build & test commands

The canonical entry points are in the [Makefile](Makefile) (run from the repo root):

```bash
make build         # cargo build --all-targets --all-features --workspace
make check         # fmt --check + clippy -D warnings + taplo TOML check + cargo-machete (unused deps)
                   #   + check-agent-artifacts + check-matrix-anchors
make unit-test     # doc tests + lib tests only (faster)
make test          # docker-up + cargo nextest run --all-targets --all-features --workspace
make check-msrv    # cargo +<MSRV> check --workspace
```

Or the underlying cargo commands directly:

```bash
cargo build --workspace
cargo test --workspace --no-fail-fast
cargo clippy --all-targets --all-features --workspace -- -D warnings
cargo fmt --all -- --check
```

- **Toolchain:** the lint gate runs on a pinned nightly ([rust-toolchain.toml](rust-toolchain.toml),
  currently `nightly-2026-03-05`, which `rustup` fetches automatically); downstream only needs MSRV
  **1.94**. The pinned nightly declares the `rustfmt` and `clippy` components.
- **`protoc` prerequisite:** `crates/sqllogictest` transitively pulls `datafusion-substrait` →
  `substrait`, whose build needs the Protobuf compiler. If `protoc` is unavailable, the core surface
  still builds/tests via `cargo test --workspace --exclude iceberg-sqllogictest`. Install
  `protobuf-compiler` to run the full suite.
- **`make test` starts Docker** (`docker-up`) for integration suites (REST fixture, MinIO, etc.).
  AWS/Glue/S3-Tables integration tests need real credentials and are not part of the offline gate.
- **Formatter:** [rustfmt.toml](rustfmt.toml) (`StdExternalCrate` import grouping, module granularity).
  **TOML:** `taplo`. **Unused deps:** `cargo machete`. **Prose:** `typos`.

CI lives in [.github/workflows/](.github/workflows/) — `ci.yml` (Rust), `ci_typos.yml`, `audit.yml`,
`codeql.yml`, `nightly_interop.yml`, `stale.yml`. (Python binding CI/release workflows were removed
with the Python layer; the `publish.yml` and website jobs the old contract listed no longer exist —
corrected against the live directory 2026-08-22.) `ci.yml` also runs the ASF license-header check
([.licenserc.yaml](.licenserc.yaml)): **every `.md` file needs the ASF header**. The sanctioned
exception is a `SKILL.md`, whose YAML frontmatter must be the first line — a new one requires a
matching `paths-ignore` entry in the same change. The other `paths-ignore` entries are not
exceptions to the rule but mechanical necessities (a generated tree, a template, and the
`.claude/skills` symlink, whose entry is load-bearing — see the comment there before touching it).

## Absolute prohibitions

These are irreversible or hard-block. The operating manuals (Non-Negotiables) reference this section.

- **No destructive or irreversible operations without explicit approval** — no `git push --force`
  to shared branches, no history rewrite, no mass file deletion, no dropping/truncating data in a
  live catalog, no resource teardown. There is no rollback.
- **Never commit or log secrets, credentials, or tokens** — not in code, tests, fixtures, or
  `tracing` output. Treat AWS keys, catalog tokens, and S3 URIs with embedded creds as radioactive.
- **Do not break the on-disk format without explicit approval** — a changed spec encoding silently
  corrupts already-written tables. This is a table-format library; format stability is the product.
  (The public *Rust* API may evolve in service of parity — this is an owned fork — but call out any
  breaking surface change so downstream pins can follow.)
- **Never edit dependency files** — [Cargo.toml](Cargo.toml), `Cargo.lock`, any crate `Cargo.toml` —
  without explicit approval. (The Phase 0 version-family sync to 0.9.1 was the sanctioned exception and
  is complete; routine work does not touch these.)

## Rust conventions — the engineering contract

The engineering method in
[.agents/skills/engineering-method/SKILL.md](.agents/skills/engineering-method/SKILL.md) carries
the portable defaults; the rules below are this repository's contract and win over it. Everything under `### Crate code` **applies to all paths
under `crates/`**; the house-style rules above it apply repo-wide.

- **Imports & formatting:** let `cargo fmt` own layout (config in [rustfmt.toml](rustfmt.toml)); do
  not hand-format imports — the `StdExternalCrate` grouping and module granularity are automatic.
- **Lints:** code must pass `cargo clippy --all-targets --all-features --workspace -- -D warnings`.
- **House style — section banners + one blank line between top-level items.** For large modules,
  group related items under a banner: a `///` doc block followed by a `///` + space + a run of `=`
  characters out to the formatter width, with the closing banner directly above the item (no blank
  line between). Banners are hand-authored and `cargo fmt`-compatible. (Adopt this only where the
  surrounding module already uses it.)
- **Logging:** `tracing` with structured fields (`?error`, ids, durations), never `println!` in
  library code, and never log secrets.

### Comments and prose

Three rules. They bind on every comment, doc comment, markdown paragraph, PR body, and ledger
entry. Long comments teach the reader to skim. A reader who skims also skims the one comment that
carried a real constraint.

**Scope: what the change adds or touches.** The existing tree does not comply, and a sweep is its
own unit. Do not raise a finding on a line the diff did not touch.

1. **Comment the WHY, never the WHAT.** A clear name and an explicit type document the WHAT. A
   comment earns its place when it records what the code cannot show. Six things qualify:
   - a race you prevent
   - an ordering invariant
   - a cross-cutting contract
   - a deliberate loud failure
   - defensive code that looks dead but is not
   - the reason you did not do the obvious thing

   Code gets rewritten. The next reader needs the reason it must not be rewritten one wrong way.
2. **Use the shortest form that carries the reason.** If two lines are enough, do not write ten.
   Cut the preamble, the restatement, and the second example. Delete what a competent reader
   derives from the code. Keep what they cannot. Then cut the keeper by half and check it still
   says the same thing.
3. **Write in ASD-STE100 Simplified Technical English.** Readers under time pressure parse simple
   sentences correctly and complex ones incorrectly. So do non-native English speakers, and so does
   the on-call engineer at 2 a.m.

| Do | Not |
|---|---|
| One idea per sentence. Max ~20 words. | Multi-clause sentences joined by em-dashes and semicolons. |
| Active voice: "the writer commits the batch". | Passive: "the batch is committed". |
| Present tense: "the retry fails". | "the retry would have failed". |
| One word, one meaning. Pick a term and reuse it. | Rotating synonyms — row / record / entry for one thing. |
| Plain verbs: "use", "read", "fail", "retry". | "leverage", "utilize", "surface", "orchestrate". |
| Say the thing. | Hedging, apology, or narration of your own reasoning. |

**Applies to human contributors only — see the model comment ban at the top of this file.**
Every function carries a doc comment stating what it does, its inputs, and its outputs. Use
`# Errors` and `# Panics` where they apply, and `# Notes` for invariants the caller needs that fit
no other section. Shape scales with the contract, not with the effort the change cost. A one-line
setter takes one line.

**Parity evidence is the main source of over-commenting here, and it is misplaced, not wrong.** The
Parity mandate makes agents paste bytecode offsets and decode narratives into doc comments. Route it:

| Evidence | Home |
|---|---|
| The Java method this mirrors | one line in the doc comment |
| What a divergence does to the caller | one line in the doc comment — it is contract text |
| That the divergence exists, and its status | the GAP_MATRIX row only (one home per fact) |
| Bytecode offsets, `javap` output, decode narrative | the unit ledger in [task/](task/) |

A named divergence splits across two homes. The doc comment says what the caller gets. The matrix
row says the fork differs from Java and how far. Never write the status in both.

A doc comment states the contract. It does not carry the proof. The review sequence for this
section is [.agents/skills/rust-code-quality/SKILL.md](.agents/skills/rust-code-quality/SKILL.md).

### Working style

These bind on every agent, in every mode, interactive and delegated.

- **Stop gathering once you can act.** Redundant file reads, repeated commands, and exploration
  past sufficient context are waste. In a delegated unit they are the main way a context budget
  is lost.
- **Write for the eventual reader, not for this conversation.** That reader opens the file months
  later without the session that produced it. Work out their knowledge, purpose, and likely
  questions privately. Never put an audience analysis in the artifact.
- **Be concise.** No sycophantic openers. No closing filler. No narrated status. Say what changed,
  what it cost, and what is still open.
- **Answer in the language the requester used.** Source code, comments, identifiers, commit
  messages, and PR titles and bodies stay English.

### Crate code — library design

- Treat crate code as reusable library code by default.
- Prefer `thiserror` for library-facing error types; the `iceberg` crate uses a central `Error` in
  [crates/iceberg/src/error.rs](crates/iceberg/src/error.rs). Binaries and examples may use `anyhow`.
- Do not use `unwrap()`, `expect()`, or panic-driven control flow outside tests. No bare
  `.unwrap()` / `.unwrap_err()` in production paths — carry context.

### Crate code — error type design

- Public API functions must return a typed error enum (preferably `thiserror`-derived), never
  `Result<_, String>`.
- Do not use `Box<dyn Error>` or `Box<dyn Error + Send + Sync>` in public trait methods or struct
  methods. Define a concrete error type with specific variants.
- When implementing `std::error::Error`, always override `fn source()` if you store an inner error.
  Breaking the error chain makes debugging impossible.
- Internal helpers that return `Result<_, String>` and are immediately wrapped via
  `.map_err(Error::other)` should return the actual error type directly.

### Crate code — concurrency

- Document lock acquisition order when a module uses multiple locks. Never acquire the same set of
  locks in different orders across code paths.
- Never hold a `tokio::sync::RwLock`/`Mutex` write guard across `.await` points unless the critical
  section is unavoidably async and the hold time is bounded.
- Prefer `compare_exchange` loops over load-then-store for concurrent counters (peak values,
  adaptive heuristics).
- When resetting multi-field atomic statistics, use a version/sequence counter or accept that
  concurrent readers may see partial snapshots; document the tradeoff.
- `std::sync::Mutex` is acceptable in async context only when held for a brief,
  non-`await`-containing critical section. If in doubt, use `tokio::sync::Mutex`.

### Crate code — recursion safety

- Recursive tree/graph traversals must have a depth limit (e.g., `max_depth` counter) or use an
  iterative approach with an explicit `Vec` stack.
- This applies to cache trees, directory walks, and any user-influenced hierarchy.
- A corrupted or malicious input must not be able to overflow the thread stack.

### Crate code — type casting

- Never use `as` for numeric conversions that may truncate or overflow. Use `try_into()` with
  explicit error handling, or clamp with `value.max(0) as usize` when the domain is bounded.
- `f64 as usize` saturates but is fragile; clamp to `[0, usize::MAX as f64]` first.
- Treat every `as` cast in a PR review as a potential bug; require justification.

### Crate code — async and performance

- Keep async paths non-blocking.
- Move CPU-heavy operations out of async hot paths with `tokio::task::spawn_blocking` when
  appropriate.

### Crate code — testing

- Keep unit tests close to the module they test.
- Keep integration tests under each crate's `tests/` directory.
- Add regression tests for bug fixes and behavior changes.
- Every test function must contain at least one `assert!`/`assert_eq!`/`assert_matches!`. A test
  that only calls code without asserting is not a test.
- In tests, prefer `.expect("context: what was being tested")` over bare `.unwrap()`. A test
  failure should tell you which operation failed and with what input.
- Test *adequacy* — how a coverage claim is evidenced rather than asserted — is a review procedure,
  not a second contract: [.agents/skills/test-adequacy/SKILL.md](.agents/skills/test-adequacy/SKILL.md).

<subagent_policy>

## Agent orchestration — current policy

**Single agent for the small stuff; Actor–Critic for anything that ships.** Do searches, reads,
and trivial mechanical edits inline in the main thread — don't spawn for those. But any change that
lands as a PR goes through an **Actor–Critic cycle with an *independent* Critic** (a separate
agent, fresh context — see SEPMO `references/05-critic.md`). The independent Critic per PR is
**non-negotiable**; convergence is the Critic's call.

**Both roles default to the frontier tier.** This is the project's concrete realization of SEPMO's
"frontier–frontier (FF)" pair. The Critic is **never** run below the frontier tier on a
correctness-bearing review: frontier Critics are materially stronger at the part that matters most —
non-vacuity and coverage refutation. The Actor may be turned down to a cheaper tier only for
genuinely rote sub-work (large mechanical renames, log scraping) — and the report must say so
explicitly when it is.

FF is the most expensive mode and is nonetheless the default, because correctness on a table-format
library *is* the product. Fan-out beyond the Actor–Critic pair stays opt-in; the heavy parity
phases — **Phase 2 (write engine)**, **Phase 4 (formats & V3 types)** — are the natural fan-out
candidates when the user asks for scale.

**Which concrete model tier realizes each role is tool mechanics, not a project rule** — it is
recorded in the tool adapter ([CLAUDE.md](CLAUDE.md) for Claude sessions,
[.agents/](.agents/map.md) for anything else), never here. The adapter may name tiers; it may not
relax the rules above.

</subagent_policy>

## Working conventions

- **One home per fact (de-triplication rule, 2026-06-10).** A capability's STATUS lives ONLY in
  [docs/parity/GAP_MATRIX.md](docs/parity/GAP_MATRIX.md) — terse cells (location, 1–2 sentences,
  flip dates, links). The Roadmap holds the plan and one-line phase statuses; increment narratives
  live in `task/todo-archive/`, `task/lessons-archive/`, and `docs/parity/archive/` (grep on
  demand, never required reading). **Never write the same status in two places — link instead.**
  When a status flips, edit the matrix cell and nothing else.
- **Chain the verification gate to the commit in ONE `&&` chain** — `typos . && cargo fmt --all --
  check && git add -A && git commit …` — never put `git commit` on a separate line from the gate: a
  failed gate on its own line still lets the commit run. (Promoted 2026-06-09 from a twice-repeated
  lessons entry.)
- **Run `make check-matrix-anchors` after any GAP_MATRIX edit** — it enforces the 5-pipe row
  audit (raw pipes inside code spans split cells silently; the de-triplication pass once stranded
  half a cell as a phantom column), the row anchors below, and citation resolution. It runs in CI
  and the aggregate `make check`. _Promoted 2026-06-11 from lessons as a manual pipe-count sweep;
  automated 2026-07-01._
- **Cite GAP_MATRIX rows by permanent anchor — `row R<id>` — never by file line number.** Line
  numbers shift when ANY line is inserted above them — the +2 drift that broke ~45 citations
  between 2026-06-17 and 2026-07-01 (discovered in four separate waves) came from two PROSE lines
  added above the table, not from row insertions; prose edits are not safe either. Every
  capability row's first cell carries its anchor (`R<id> ·`); a NEW row takes the next unused ID
  and may be inserted anywhere; IDs are never reused. Bare-number citations in dated archives are
  historical epochs — leave them. Enforced by `scripts/check_matrix_anchors.sh`. _Added
  2026-07-01._
- **A sabotage step that cannot be applied must HARD-FAIL, never SKIP.** A negative/sabotage test
  that did not actually corrupt anything has proven nothing — a SKIP branch is a false-green. When
  the corruption cannot be applied (e.g. the target byte pattern is absent), exit non-zero and abort
  the chain (restoring any `.bak` first); under `set -euo pipefail`, capture the mutator's exit with
  `|| rc=$?` so the restore stays reachable. _Promoted 2026-06-13 from a thrice-repeated lessons
  entry (interop sabotage 6b / 8e / 7b)._
- **Upstream is a sync baseline, not a constraint.** This is an owned fork for Java `iceberg-core`
  parity — edit freely; sync up from upstream and cherry-pick wins, but mergeability is not required.
- **Tests ship with the change**, plus interop tests where applicable (see the Parity mandate and the
  manuals' §4 Done gate). The testing-discipline contract is [docs/testing.md](docs/testing.md).
- **Keep `map.md` in lockstep** with the directories that use it (see `<map_md_navigation>`).
- **Follow the operating manual for your tier** ([skills/](skills/)) — Risk-First, naming, the
  Rust rules, the debugging protocol, and the verification gate. This file wins on conflict.
- **Review procedures live as skills, not as a second contract.**
  [.agents/skills/map.md](.agents/skills/map.md) is the roster; each skill points back here for
  every rule it leans on.
