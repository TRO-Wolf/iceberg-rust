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

# Unit brief — G3: HMS type-string parity vs Java `HiveSchemaUtil` (incl. timestamptz)

**Charter (2026-07-17, user-signed; FF Actor–Critic).** Bring the HMS schema converter
(`crates/catalog/hms/src/schema.rs`, `HiveSchemaBuilder`) to behavioral parity with
Java `HiveSchemaUtil.convertToTypeString` (1.10.0) — the HMS sibling of the #153/#155
Glue work, with a DIFFERENT oracle: Hive's converter **throws** where Glue's lowercases,
and timestamptz is **Hive-version-gated**.

## The oracle — ALREADY DECODED (bytecode, `iceberg-hive-metastore-1.10.0.jar`, cached in `~/.m2`)

Orchestrator-decoded 2026-07-17 via `javap -c` (jar network-fetched once; re-verify
locally, do NOT re-fetch):

- `convertToTypeString` is a TypeID switch (same 16-case `$SwitchMap` shape as Glue):
  boolean/int/bigint/float/double/date; TIME|STRING|UUID → `"string"`;
  FIXED|BINARY → `"binary"`; `decimal(%s,%s)`.
- **TIMESTAMP case (offsets 113–139):** `if HiveVersion.min(HIVE_3) &&
  ts.shouldAdjustToUTC() → "timestamp with local time zone" else → "timestamp"`.
  So: naive µs → `"timestamp"` always; **timestamptz (µs) → `"timestamp with local
  time zone"` on Hive 3+, `"timestamp"` below**.
- **Default (offsets 303–319): `throw new UnsupportedOperationException("Not supported: "
  + type)`** — TIMESTAMP_NANO (both nano variants) and UNKNOWN reach it. Java HMS
  REJECTS nano and unknown; it never emits `"timestamp_ns"` or `"timestamp_nano"`.
- Struct: `struct<%s>` join **bare `","`** (offset 204); field lambda `%s:%s`
  (`name:recurse`, no case folding). List `array<%s>`; map `map<%s,%s>`.
- **`HiveVersion`**: enum `HIVE_4/HIVE_3/HIVE_2/HIVE_1_2/NOT_SUPPORTED`;
  `current()` = `calculate()` reads `HiveVersionInfo.getShortVersion()` — the **client
  classpath's** Hive library version (first digit "4"/"3"/…), NOT the metastore
  server's. The fork talks thrift directly with no Hive client library, so the
  Java-faithful port of this gate is a **configuration knob**, not runtime detection.

Fork divergences (current `hms/src/schema.rs`):

1. Struct join `", "` (line ~73) — Java `","`.
2. `TimestampNs => "timestamp_ns"` (line ~124) — a fork-invented string Java NEVER
   emits; Java throws. Parity = `FeatureUnsupported` error. (This is a deliberate
   capability regression toward Java behavior — disclose loudly.)
3. `Timestamptz | TimestamptzNs => FeatureUnsupported` (line ~125) — Java: µs-tz is the
   version-gated STRING (never an error); ns-tz throws. Split the arm.
4. `Unknown => FeatureUnsupported` (line ~138) — matches Java's throw; KEEP (align the
   message shape with the Java "Not supported" semantics if it diverges wildly).

## Proposition ledger

- **C-1 (Hive-version knob):** add a `HiveVersion`-equivalent to the HMS catalog
  surface: an enum (at minimum `Hive2`-era vs `Hive3+` behavior) carried by
  `HiveSchemaBuilder`, sourced from a catalog config property. Follow the fork's
  existing HMS config conventions for naming (inspect `hms/src/catalog.rs` config
  struct; a string property like `"hive.hive-version"` parsed leniently — first digit —
  mirrors Java's `getShortVersion` split). **DEFAULT = Hive 3+** (modern metastores;
  the alternative default would silently strip zone semantics from every timestamptz
  column). Invalid values ⇒ loud `DataInvalid` naming the property (fork convention
  from #154), NOT silent fallback.
- **C-2 (timestamptz µs):** Hive3+ ⇒ `"timestamp with local time zone"`; Hive2 ⇒
  `"timestamp"`. Byte-exact per the bytecode.
- **C-3 (nano variants):** BOTH `TimestampNs` and `TimestamptzNs` ⇒ `FeatureUnsupported`
  (Java throws for TIMESTAMP_NANO regardless of zone). The `"timestamp_ns"` emission is
  REMOVED — disclosed as a behavior change toward parity in the commit message and
  build summary.
- **C-4 (struct separator + lambda):** join `", "` → `","`; verify the fork's field
  rendering is `name:type` with no case folding (fix if not); all existing test strings
  updated in the same commit.
- **C-5 (Unknown):** keep the reject; align/verify the error carries the type name
  (Java: `"Not supported: " + type`).
- **C-6 (pins):** byte-exact pins for every arm changed, BOTH version-gate branches
  (Hive3 default AND explicit Hive2 ⇒ `"timestamp"`), the nano rejects (error kind
  asserted), the invalid-knob-value loud error, and a nested struct pin
  (separator-sensitive). Mutation proofs run live: separator revert → RED; gate
  inversion (Hive2 default or dropped `shouldAdjustToUTC`-analogue) → RED; nano
  re-emission → RED. Record exact RED sets.
- **C-7 (docs):** GAP_MATRIX — the R91 cell's HMS clause (left pointing at this unit by
  #155) updated to the landed behavior; check any HMS catalog row for stale type-string
  claims; at most the minimal true edit per cell (one home per fact;
  `./scripts/check_matrix_anchors.sh` after any matrix edit). Comment citations in
  schema.rs written as "bytecode-verified (javap, iceberg-hive-metastore-1.10.0.jar)"
  with offsets.

## Boundaries (not in scope)

- NO Glue changes (done in #153/#155). NO thrift/client changes beyond threading the
  config knob. NO read-path (HMS→Iceberg) conversion changes unless a pin proves a
  round-trip bug introduced by THIS unit's strings.
- NO Cargo.toml/Cargo.lock/pom.xml. NO network (jar is cached). NO HMS integration
  tests (Docker-gated) — unit tests only.
- Naive `Timestamp` (µs) stays `"timestamp"`; already-matching arms untouched.

## Gate (unit)

`git ls-files -z | xargs -0 typos --force-exclude` · `cargo fmt --all -- --check` ·
`cargo clippy -p iceberg-catalog-hms --all-targets -- -D warnings` ·
`cargo test -p iceberg-catalog-hms --lib` · `./scripts/check_matrix_anchors.sh` ·
`./scripts/check_agent_artifacts.sh`. Chain gate→commit in ONE `&&` chain; explicit
paths, never `git add -A`.
