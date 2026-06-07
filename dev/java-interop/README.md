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

# `dev/java-interop` — UpdateSchema + UpdatePartitionSpec interop oracle (TEST ONLY)

> **This is a TEST-ONLY ORACLE — not part of the shipped Rust library.** It is a dev tool, exactly like
> [`dev/spark/`](../spark/): a Maven module that drives the Java Apache Iceberg `iceberg-core` reference
> to prove byte-/field-id-level compatibility of the Rust `UpdateSchema` AND `UpdatePartitionSpec` actions
> with Java, in both directions. It is **not** a Cargo crate, **not** a Cargo dependency, and is **not**
> linked into anything. `cargo build` / `cargo test` never invoke Java. The durable, committed artifacts
> are the JSON fixtures under
> [`crates/iceberg/testdata/interop/`](../../crates/iceberg/testdata/interop/) and the Rust tests
> [`interop_update_schema.rs`](../../crates/iceberg/tests/interop_update_schema.rs) and
> [`interop_update_partition_spec.rs`](../../crates/iceberg/tests/interop_update_partition_spec.rs) that
> read them. The Java here only regenerates fixtures and acts as the read-side oracle.

## What it proves

A GAP_MATRIX row flips to ✅ only with an interop test proving byte-level table compatibility with Java
**in both directions** (see [`docs/testing.md`](../../docs/testing.md) "Interop tests"). This harness
satisfies that for both `UpdateSchema` and `UpdatePartitionSpec`:

- **Direction 1 — Rust reproduces Java's evolution.** For each scenario, the Rust test loads the
  Java-written `base.metadata.json`, applies the *same* op-sequence via the public transaction API
  (driven through an in-memory catalog commit), and asserts the Rust-evolved result is **structurally
  equal** to Java's `java_evolved.metadata.json` — for schema, the current schema (recursive field id /
  name / type / required / doc / default + identifier ids + current-schema-id + last-column-id); for
  partitions, the default partition spec (spec-id + each field's source-id / field-id / name / transform)
  + `last-partition-id`. This runs in the **normal offline `cargo test` suite** (no Java, no Docker).
- **Direction 2 — Java reads what Rust writes.** The Rust tests (under `ICEBERG_INTEROP_GEN=1`) write
  `rust_evolved.metadata.json`; the Java oracle's `verify` mode reads it with `TableMetadataParser` and
  asserts Java parses it and its current schema / default spec matches Java's own evolution. Exits
  non-zero on any FAIL.

Comparison is **structural, not byte-for-byte**: both metadata files are parsed into the Rust model and
compared via `PartialEq` (`StructType` for schema; `PartitionField` for partition fields). Jackson and
`serde_json` differ in key order and whitespace, so raw-byte comparison is meaningless; *logical table
identity including field ids* is the contract.

## Schema scenarios

Implemented identically (and named identically) on both sides — the Rust op-sequences in
`crates/iceberg/tests/interop_update_schema.rs::apply_scenario_ops` mirror the Java ones in
`InteropOracle.SchemaOracle.scenarios()`:

| Scenario | Format | What it pins |
|---|---|---|
| `add_top_level_columns` | v3 | optional + required-with-default top-level adds |
| `add_nested_struct_and_map` | v2 | **level-order fresh field-id assignment** for `map<struct,struct>` (key=3, value=4, key struct 5–8, value struct 9–10) — exact nested ids |
| `rename_and_move` | v2 | rename + reorder; move targets resolve by **original** name |
| `update_type_promotion` | v2 | int→long, float→double, decimal(9,2)→decimal(18,2) widen |
| `make_optional_and_delete` | v2 | required→optional relax + column delete (last-column-id does not decrease) |
| `set_identifier_fields` | v2 | identifier-field-id set |
| `add_required_with_default_and_update_default` | v3 | required add WITH default; `updateColumnDefault` changes **only** the write default (init=`active`, write=`pending`) |

## Partition-spec scenarios

Implemented identically (and named identically) on both sides — the Rust op-sequences in
`crates/iceberg/tests/interop_update_partition_spec.rs::apply_scenario_ops` mirror the Java ones in
`InteropOracle.PartitionOracle.scenarios()`:

| Scenario | Format | What it pins |
|---|---|---|
| `add_identity_field` | v2 | base case — add `identity(category)` to an unpartitioned base |
| `add_transform_fields` | v2 | `bucket[16]`, `truncate[8]`, `year` adds — auto-generated names (`PartitionNameGenerator`) AND sequentially-assigned field-ids (1000, 1001, 1002) |
| `remove_field_v2` | v2 | removing a field **omits** it from the new spec |
| `remove_field_v1_void` | **v1** | removing a field **re-adds it as `void`** preserving its field id (V1 alwaysNull replacement) |
| `rename_field` | v2 | rename preserves the field id |
| `field_id_recycling` | v2 | re-adding a `(source, transform)` present in a **historical** spec **recycles the historical field id AND name** (`id_shard`, not the generated `id_bucket_8`) — Java `recycleOrCreatePartitionField` |
| `delete_then_readd` | v2 | remove + re-add the same `(source, transform)` → Java's rewrite/un-delete; the result equals the base so the metadata layer **dedups** back to the existing spec id |

> The V1 void scenario uses format version 1; the rest use v2.

## How the Java program reaches the testing constructors

`InteropOracle.java` is declared in `package org.apache.iceberg` on purpose: that is the only way to
reach the package-private machinery for BOTH capabilities.

- **UpdateSchema** drives the package-private `@VisibleForTesting SchemaUpdate(Schema schema, int
  lastColumnId)` constructor (`core/.../SchemaUpdate.java`), which runs the full `UpdateSchema` state
  machine without a live `TableOperations` / catalog.
- **UpdatePartitionSpec** must NOT use the analogous `@VisibleForTesting BaseUpdatePartitionSpec(int,
  PartitionSpec, …)` constructors: those set `base = null`, and Java's `recycleOrCreatePartitionField`
  only recycles a historical field id+name when `formatVersion >= 2 && base != null`. So the oracle
  drives a **real** `BaseUpdatePartitionSpec` via `new BaseTable(ops, name).updateSpec()…commit()` over a
  minimal in-memory `TableOperations` (`InteropOracle.InMemoryTableOperations`) that just holds a
  `TableMetadata` and swaps it on commit — `base = ops.current()`, so recycling is live. A partition-spec
  commit never touches data files, so `io()` / `locationProvider()` stay no-op. The same path also builds
  the two-historical-spec base for the recycling scenario (evolving in a non-default spec so its field
  gets a realistic fresh id).

The base/evolved `TableMetadata` are serialized via `TableMetadataParser.toJson`.

## Running

```bash
# One shot — regenerate all fixtures (both capabilities) and verify both directions:
dev/java-interop/run.sh

# Or step by step (from the repo root):

# (a) Java: (re)write base.metadata.json + java_evolved.metadata.json for BOTH capabilities
/opt/maven/bin/mvn -f dev/java-interop -q compile exec:java -Dexec.args=generate

# (b) Rust: (re)write rust_evolved.metadata.json AND assert Direction 1
ICEBERG_INTEROP_GEN=1 cargo test -p iceberg --test interop_update_schema
ICEBERG_INTEROP_GEN=1 cargo test -p iceberg --test interop_update_partition_spec

# (c) Java: assert it can read the Rust output (Direction 2; both capabilities)
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
    └── InteropOracle.java                         # SchemaOracle + PartitionOracle (generate + verify)

crates/iceberg/testdata/interop/{update_schema,update_partition_spec}/<scenario>/
├── base.metadata.json                             # Java-written base (committed)
├── java_evolved.metadata.json                     # Java-written evolved (committed)
└── rust_evolved.metadata.json                     # Rust-written evolved (committed)
```
