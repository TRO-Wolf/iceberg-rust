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

# `dev/java-interop` — UpdateSchema interop oracle (TEST ONLY)

> **This is a TEST-ONLY ORACLE — not part of the shipped Rust library.** It is a dev tool, exactly like
> [`dev/spark/`](../spark/): a Maven module that drives the Java Apache Iceberg `iceberg-core` reference
> to prove byte-/field-id-level compatibility of the Rust `UpdateSchema` action with Java, in both
> directions. It is **not** a Cargo crate, **not** a Cargo dependency, and is **not** linked into
> anything. `cargo build` / `cargo test` never invoke Java. The durable, committed artifacts are the
> JSON fixtures under [`crates/iceberg/testdata/interop/update_schema/`](../../crates/iceberg/testdata/interop/update_schema/)
> and the Rust test [`crates/iceberg/tests/interop_update_schema.rs`](../../crates/iceberg/tests/interop_update_schema.rs)
> that reads them. The Java here only regenerates fixtures and acts as the read-side oracle.

## What it proves

A GAP_MATRIX row flips to ✅ only with an interop test proving byte-level table compatibility with Java
**in both directions** (see [`docs/testing.md`](../../docs/testing.md) "Interop tests"). This harness
satisfies that for `UpdateSchema`:

- **Direction 1 — Rust reproduces Java's evolution.** For each scenario, the Rust test loads the
  Java-written `base.metadata.json`, applies the *same* `UpdateSchema` op-sequence via the public
  transaction API (driven through an in-memory catalog commit), and asserts the Rust-evolved current
  schema is **structurally equal** to Java's `java_evolved.metadata.json` — recursive field id / name /
  type / required / doc / default, plus identifier-field ids, current-schema-id, and last-column-id.
  This runs in the **normal offline `cargo test` suite** (no Java, no Docker).
- **Direction 2 — Java reads what Rust writes.** The Rust test (under `ICEBERG_INTEROP_GEN=1`) writes
  `rust_evolved.metadata.json`; the Java oracle's `verify` mode reads it with `TableMetadataParser` and
  asserts Java parses it and its current schema matches Java's own evolution. Exits non-zero on any FAIL.

Comparison is **structural, not byte-for-byte**: both metadata files are parsed into the Rust model and
compared by `StructType: PartialEq` (which recurses field id / name / type / required / doc / default).
Jackson and `serde_json` differ in key order and whitespace, so raw-byte comparison is meaningless;
*logical table identity including field ids* is the contract.

## Scenarios

Implemented identically (and named identically) on both sides — the Rust op-sequences in
`crates/iceberg/tests/interop_update_schema.rs::apply_scenario_ops` mirror the Java ones in
`InteropOracle.scenarios()`:

| Scenario | Format | What it pins |
|---|---|---|
| `add_top_level_columns` | v3 | optional + required-with-default top-level adds |
| `add_nested_struct_and_map` | v2 | **level-order fresh field-id assignment** for `map<struct,struct>` (key=3, value=4, key struct 5–8, value struct 9–10) — the Increment-3 blocker case; exact nested ids |
| `rename_and_move` | v2 | rename + reorder; move targets resolve by **original** name |
| `update_type_promotion` | v2 | int→long, float→double, decimal(9,2)→decimal(18,2) widen |
| `make_optional_and_delete` | v2 | required→optional relax + column delete (last-column-id does not decrease) |
| `set_identifier_fields` | v2 | identifier-field-id set |
| `add_required_with_default_and_update_default` | v3 | required add WITH default (no `allowIncompatibleChanges`); `updateColumnDefault` changes **only** the write default (init=`active`, write=`pending`) |

> Column **initial defaults are a V3-only feature in Java** `iceberg-core` (a non-null initial default
> is rejected on v2 metadata), so the two default-bearing scenarios use format version 3; the rest use
> v2. This matches Java's actual contract.

## How the Java program reaches the testing constructor

`InteropOracle.java` is declared in `package org.apache.iceberg` on purpose: that is the only way to
reach the package-private `@VisibleForTesting SchemaUpdate(Schema schema, int lastColumnId)` constructor
(`core/.../SchemaUpdate.java`), which drives the full `UpdateSchema` state machine without a live
`TableOperations` / catalog. The base/evolved `TableMetadata` are built with the public
`TableMetadata.newTableMetadata` / `buildFrom(base).setCurrentSchema(evolved, lastColumnId).build()` and
serialized via `TableMetadataParser.toJson`.

## Running

```bash
# One shot — regenerate all fixtures and verify both directions:
dev/java-interop/run.sh

# Or step by step (from the repo root):

# (a) Java: (re)write base.metadata.json + java_evolved.metadata.json
/opt/maven/bin/mvn -f dev/java-interop -q compile exec:java -Dexec.args=generate

# (b) Rust: (re)write rust_evolved.metadata.json AND assert Direction 1
ICEBERG_INTEROP_GEN=1 cargo test -p iceberg --test interop_update_schema

# (c) Java: assert it can read the Rust output (Direction 2)
/opt/maven/bin/mvn -f dev/java-interop -q exec:java -Dexec.args=verify
```

A normal `cargo test` (without `ICEBERG_INTEROP_GEN`) runs Direction 1 only and writes **no** files —
so the offline suite is hermetic and never mutates the committed fixtures.

## Requirements

- **Maven** at `/opt/maven/bin/mvn` (override with `MVN=...`). The first run downloads
  `org.apache.iceberg:iceberg-core` / `iceberg-api` **1.10.0** from Maven Central.
- **Java 11+** (Iceberg 1.10 requires Java 11+).
- A Rust toolchain (the repo's pinned nightly via `rust-toolchain.toml`).

## Layout

```
dev/java-interop/
├── pom.xml                                       # iceberg-core/api 1.10.0 + exec-maven-plugin
├── run.sh                                         # gen (java) → gen+assert (rust) → verify (java)
├── README.md                                      # this file
└── src/main/java/org/apache/iceberg/
    └── InteropOracle.java                         # generate + verify modes (package-private ctor access)

crates/iceberg/testdata/interop/update_schema/<scenario>/
├── base.metadata.json                             # Java-written base (committed)
├── java_evolved.metadata.json                     # Java-written evolved (committed)
└── rust_evolved.metadata.json                     # Rust-written evolved (committed)
```
