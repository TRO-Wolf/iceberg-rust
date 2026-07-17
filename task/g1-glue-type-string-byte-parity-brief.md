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

# Unit brief — G1: Glue type-string byte-parity closure

**Charter (2026-07-17, user-signed; follow-up to #153).** Close the two named residues
from the #153 Fable-max Critic and finish the job: make `GlueSchemaBuilder`'s type
strings **byte-identical** to Java `IcebergToGlueConverter.toTypeString` (1.10.0)
across the entire type table. Display-only surface (nothing in the crate parses these
strings back — verified in the #153 review), so parity is the whole point and the risk
is low.

## The oracle — ALREADY DECODED (bytecode, `iceberg-aws-1.10.0.jar`, cached in `~/.m2`)

Orchestrator-decoded 2026-07-17 via `javap -c` (the jar was network-fetched once via
`mvn dependency:get`; re-verify locally, do NOT re-fetch):

- `toTypeString` is a TypeID switch (synthetic `$SwitchMap`, explicit cases BOOLEAN,
  INTEGER, LONG, FLOAT, DOUBLE, DATE, TIME|STRING|UUID, TIMESTAMP, FIXED|BINARY,
  DECIMAL, STRUCT, LIST, MAP) with `default: typeId().name().toLowerCase(Locale.ENGLISH)`
  (offsets 291–306). TIMESTAMP_NANO has ZERO `$SwitchMap` entries → default →
  `"timestamp_nano"` for BOTH nano variants. UNKNOWN likewise has no case → Java
  renders `"unknown"` — **Java never throws in `toTypeString`**.
- Struct: `String.format("struct<%s>", fields.stream().map(<lambda>).collect(Collectors.joining(",")))`
  — the join separator is a **bare comma**, no space.
- List: `String.format("array<%s>", ...)`; Map: `String.format("map<%s,%s>", ...)` —
  both already match the fork.

Fork divergences (current `crates/catalog/glue/src/schema.rs`):

1. Struct join is `", "` (comma-space) at ~line 100 — Java is `","`.
2. `PrimitiveType::TimestampNs => "timestamp_ns"` — Java is `"timestamp_nano"` (the
   #153 charter-freeze is hereby lifted by this charter).
3. `PrimitiveType::Unknown` → `FeatureUnsupported` reject — Java renders `"unknown"`.

## Proposition ledger

- **C-1 (struct separator):** `", "` → `","`. Every existing test string updated in the
  same commit (they are oracle pins; the oracle was wrong).
- **C-2 (naive nano):** `TimestampNs` → `"timestamp_nano"`, making both nano variants
  identical, as Java's shared `TypeID.TIMESTAMP_NANO` forces (bytecode: `typeId()` has
  no zone/precision branch). The #153 comment explaining the frozen deviation is
  rewritten to state the now-uniform Java mapping.
- **C-3 (Unknown):** decide from the evidence, then implement ONE of: (a) map to
  `"unknown"` (strict Java byte-parity; Java provably never throws here), or (b) keep
  the reject with a comment stating the DELIBERATE divergence and why. Default to (a)
  unless you find a concrete downstream reason the fork must reject (e.g. Glue API
  rejects the string, or the fork's V3 `unknown` handling elsewhere depends on the
  error). Whichever way: the decision, its evidence, and the flip of the existing
  Unknown-reject pin (or its retention) are called out explicitly in the build summary.
- **C-4 (struct-field lambda):** decode the struct-field lambda in the jar
  (`lambda$toTypeString$N` — expected `field.name() + ":" + toTypeString(field.type())`,
  possibly via String.format/concat) and confirm the fork's field rendering
  (`name:type`, lowercasing rules if any) is byte-identical; fix if not. Record the
  decoded shape in a comment citation.
- **C-5 (citation upgrade):** the schema.rs comments citing `IcebergToGlueConverter.java`
  file:line now say "bytecode-verified (javap, iceberg-aws-1.10.0.jar)" where the claim
  was verified; the #153 test comment "Matches Java ... recursive struct/list handling"
  is corrected (it was byte-false under `", "`).
- **C-6 (pins):** byte-exact oracle-string pins for every CHANGED arm; a nested
  struct-in-struct pin that pins the exact full string (separator-sensitive); the nano
  matrix updated; the Unknown pin flipped or retained per C-3. Mutation proofs: revert
  the separator → nested pins RED; revert the nano arm → matrix RED; (if C-3=a) restore
  the reject → the new unknown pin RED.
- **C-7 (docs):** GAP_MATRIX — check whether any row cell states Glue type-string
  behavior; edit at most that one cell (one home per fact; run
  `./scripts/check_matrix_anchors.sh` after ANY matrix edit). No new map.md (the crate
  directory does not use the convention — verify).

## Boundaries (not in scope)

- NO HMS (`crates/catalog/hms`) — that is unit G3, different oracle, design-gated.
- NO Glue API calls, no integration tests (credentials-gated), no read-path changes.
- NO Cargo.toml/Cargo.lock/pom.xml edits. The jar is already cached; work offline.
- Naive `Timestamp` (µs) stays `"timestamp"`; all already-matching arms untouched.

## Gate (unit)

`git ls-files -z | xargs -0 typos --force-exclude` · `cargo fmt --all -- --check` ·
`cargo clippy -p iceberg-catalog-glue --all-targets -- -D warnings` ·
`cargo test -p iceberg-catalog-glue --lib` · `./scripts/check_matrix_anchors.sh` ·
`./scripts/check_agent_artifacts.sh`. Chain gate→commit in ONE `&&` chain; explicit
paths, never `git add -A`.
