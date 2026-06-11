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

# map.md — crates/iceberg/src/variant/

## Purpose

The Iceberg V3 **variant binary format, read side** — a Rust port of Java 1.10.0
`org.apache.iceberg.variants` deserialization (`SerializedMetadata` / `SerializedPrimitive` /
`SerializedShortString` / `SerializedObject` / `SerializedArray` / `Variant`), parsed EAGERLY
and bounds-checked as a security boundary (untrusted file bytes; errors, never panics). The
write side (`Variants` factory et al.) is the B2 increment — not here yet. The Java→Rust
mapping and every deliberate divergence are documented in [mod.rs](mod.rs)'s module doc.

## Contents

| File | What it does |
|---|---|
| `mod.rs` | module doc (Java→Rust map, divergence list, security posture), re-exports |
| `types.rs` | `BasicType`, `PhysicalType` (the exact 1.10.0 23-constant set, type ids 0..=20), `LogicalType` |
| `util.rs` | bounds-checked little-endian readers; `find` (Java `VariantUtil.find`'s exact probe sequence); the UTF-16 `String.compareTo` comparator |
| `metadata.rs` | `VariantMetadata` — the dictionary: header, offset table, UTF-8 strings, sorted/linear `id()` lookup |
| `value.rs` | `VariantValue::parse` dispatch + `VariantPrimitive`/`VariantObject`/`VariantArray`/`Variant`; the depth guard `MAX_NESTING_DEPTH` |
| `tests.rs` | hand-built per-layout vectors, the malformed-input suite, and the Java-1.10.0-generated fixture pins (provenance in the module doc) |

## I want to...

| I want to... | go to |
|---|---|
| Decode a value column's bytes | `VariantValue::parse(&metadata, bytes)` in [value.rs](value.rs) |
| Decode a combined metadata+value buffer | `Variant::from_bytes` in [value.rs](value.rs) |
| Check a layout against Java | the cited `Serialized*.java` lines / `javap` notes above each parser fn and test |
| Add the write side (B2) | new sibling files; keep this read surface untouched and re-use `types.rs` |

## Pointers

- **Up:** [crates/iceberg/src/](../) — wired as `pub mod variant` in `lib.rs`.
- **Related:** `spec/datatypes.rs` (where the `variant` SCHEMA type will land — not part of B1);
  `docs/parity/GAP_MATRIX.md` "V3 types: variant" row (status + deferrals).

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| A value Java reads fine is rejected here | Check the divergence list in mod.rs first: eager parse rejects malformed NESTED values Java only rejects on access; invalid UTF-8 is Err here but U+FFFD in Java; truncated ZERO-count containers (`02 00` / `03 00`) and an empty-dict metadata declaring data past the buffer end are Err here but lazily tolerated by Java |
| `get(name)` misses a field that is visibly present | Working as Java does: object fields / sorted dictionaries are BINARY-searched in Java `String.compareTo` (UTF-16) order; non-name-sorted objects miss identically in Java 1.10.0 — do NOT "fix" with a linear scan |
| "nesting depth exceeds" on legitimate data | `MAX_NESTING_DEPTH` (128) is a Rust-side DoS guard with no Java equivalent — raising it is a deliberate decision, not a bug fix |
| A decoded number is wrong by byte order | All payloads are little-endian EXCEPT the 16-byte UUID (big-endian RFC 4122); decimal16 is `i128::from_le_bytes` (Java reverses into a big-endian `BigInteger` — same value) |

### First checks

1. `cargo test -p iceberg --lib variant` — the fixture pins localize which layout drifted.
2. Compare against the 1.10.0 BYTECODE (`javap -p -c -classpath ~/.m2/.../iceberg-api-1.10.0.jar org.apache.iceberg.variants.<Class>`), not MAIN source — the jar is the pin.

### Escalate to

- [dev/java-interop/map.md](../../../../dev/java-interop/map.md#debug) for oracle-driven byte comparisons.
- [docs/parity/GAP_MATRIX.md](../../../../docs/parity/GAP_MATRIX.md) for what is in/out of scope.
