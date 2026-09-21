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

# map.md — crates/iceberg/src/spec/schema/

## Purpose

Table schemas: the `Schema` type with its builder, the by-id / by-name indexes, the
field-id → `StructAccessor` map that predicate binding reads, schema serde, and the
`TypeUtil` ports (visitors, compatibility, promotion, pruning, id assignment).

## Contents

| File | What it does |
|---|---|
| `mod.rs` | `Schema` + builder: `build_accessors` (every struct field incl. list/map/struct owns an accessor; element/key/value ids own none — Java `BuildPositionAccessors`), `accessor_by_field_id`, `field_by_name`/`field_by_id`, identifier validation |
| `accessor_tests.rs` | **test-only** accessor-map pins: container ids `Some`, element/key/value/container-nested ids `None` |
| `cache_charge.rs` | `pub(crate)` index-capacity/entry accessors for the object-cache charge model |
| `index.rs` | `IndexById`, `IndexByName` (dotted paths incl. `element`/`key`/`value`), parent-id index |
| `visitor.rs` | post-order `SchemaVisitor` + `visit_schema` |
| `utils.rs` | `TypeUtil` ports (joins, size estimates, quoted-name index) |
| `compat.rs` | write/read compatibility (`CheckCompatibility` port) |
| `type_promotion.rs` | promotion-allowed checks for evolution |
| `prune_columns.rs` | prune a schema/struct to an id set |
| `id_reassigner.rs` | fresh field-id assignment (`AssignFreshIds` port) |
| `_serde.rs` | schema JSON serde (V1/V2 shapes) |

## I want to...

| I want to... | go to |
|---|---|
| Change which fields own accessors | `mod.rs::build_accessors` / `build_accessors_nested` — and the charge walk in `../../io/object_cache.rs`, which must match the map |
| Resolve a dotted column path | `field_by_name` (`mod.rs`) over `IndexByName` (`index.rs`) |
| Add a schema traversal | `visitor.rs` (`SchemaVisitor`) |
| Change evolution rules | `compat.rs` / `type_promotion.rs` |

## Pointers

- **Up:** [../](../) (spec: the on-disk format) · **Related:**
  [../../expr/map.md](../../expr/map.md) (binding + `StructAccessor`),
  [../../io/](../../io/) (cache charge consumer)

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| Legal nested path fails `field_by_name` | `IndexByName` pushes `element`/`key`/`value` segments for containers — check the visitor hooks in `index.rs` |
| Cache charge drifts from the real map | `schema_accessor_charge` must walk `accessor_entries` (every map entry + its `Box` chain), never re-derive the walk from the type tree (pin: ledger F-CONTAINER-ACCESSOR-1 §7 M12) |
| New accessor shape breaks predicate JSON | the `is_optional` default/skip contract lives in `../../expr/accessor.rs` — see [../../expr/map.md#debug](../../expr/map.md#debug) |

### First checks

- Print `accessor_by_field_id` for the leaf id; element/key/value ids answer `None`
  by design (Java parity).

### Escalate to

- Binding/fold issues → [../../expr/map.md#debug](../../expr/map.md#debug).
