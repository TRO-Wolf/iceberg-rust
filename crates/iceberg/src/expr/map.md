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

# map.md — crates/iceberg/src/expr/

## Purpose

Predicate trees and binding (Java `api/.../expressions/`): unbound `Predicate` / `Reference`
terms, `bind` to a schema producing `BoundPredicate` / `BoundReference`, position accessors for
reading values out of `Struct`s, and the expression parser. Pruning evaluators over bound
predicates live in `visitors/`.

## Contents

| File | What it does |
|---|---|
| `predicate.rs` | `Predicate` / `BoundPredicate`, `bind` with the ancestor-aware `IS NULL` fold (`MAX_PREDICATE_DEPTH`-capped), `rewrite_not`, serde shadows with arity validation |
| `term.rs` | `Reference` (dotted path) + `Bind` → `BoundReference` (field + accessor; missing accessor is typed `DataInvalid`) |
| `accessor.rs` | `StructAccessor`: position + full `Type` + `is_optional` (serde-skipped on the default so optional-primitive JSON stays byte-identical); `get` (primitive leaves) / `is_present` (presence incl. containers) |
| `predicate_container_tests.rs` | **test-only** container binding pins: null tests on list/map/struct, bind rejection of element/key/value paths and container comparisons, required-under-optional no-fold, frozen predicate JSON, direct partition-evaluator presence pin |
| `expression_parser.rs` | string → `Predicate` parser |
| `visitors/` | bound-predicate evaluators — see [visitors/map.md](visitors/map.md) |

## I want to...

| I want to... | go to |
|---|---|
| Change binding or the null fold | `predicate.rs::bind_leaf` (fold needs leaf AND all ancestors required) + `term.rs` (`Reference::bind`) |
| Change what an accessor reads | `accessor.rs` (`get` for values, `is_present` for null tests); the map itself is built in `../spec/schema/` (`build_accessors`) |
| Change predicate JSON | `predicate.rs` + `term.rs` + `accessor.rs` serde impls; the frozen optional-primitive string in `predicate_container_tests.rs` must stay byte-identical to `main` |
| Evaluate a predicate against data or stats | [visitors/map.md](visitors/map.md) (partition / manifest / metrics / parquet evaluators) |

## Pointers

- **Up:** [crates/iceberg/src/](../) · **Related:** [visitors/map.md](visitors/map.md)
  (evaluators), [../spec/schema/map.md](../spec/schema/map.md) (schema + accessor map),
  [../scan/map.md](../scan/map.md) (scan planning consumer)

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| `DataInvalid: Accessor for Field ... not found` on a legal path | list element / map key / map value ids own no accessor (Java parity — bind must fail); anything else means `build_accessors` regressed (pins: M1/M13) |
| Required-under-optional null test folds to a constant | the fold must consult the accessor's ancestor-ORed `is_optional`, not the leaf's `required` flag (pin: M2) |
| `BoundPredicate` JSON gains/loses a key | `is_optional` serializes only when non-default; required primitives carry explicit `false` (pins: M16/M17) |
| Evaluator panics on NOT | run `rewrite_not` first — see [visitors/map.md#debug](visitors/map.md#debug) |

### First checks

- Bind the predicate in a scratch test and print the bound form + JSON before suspecting
  the readers — most "wrong rows" reports here are bind-time, not read-time.
- Check `accessor_by_field_id` for the path's leaf id; element/key/value ids answer `None`
  by design.

### Escalate to

- Accessor-map construction → [../spec/schema/map.md#debug](../spec/schema/map.md#debug).
- Pruning/evaluator semantics → [visitors/map.md#debug](visitors/map.md#debug).
