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

# U1 ledger — QB + BUG-001 (2026-08-03)

**Branch:** `fix/qb-posdelete-bounds-and-partition-stamp` off `0421ae15`
**Tag:** `[fork]`

## Actor build

### Leg A (BUG-001)
- Fast path: `position_delete_unpartitioned_fast_path(spec_count, default_field_count)` =
  `spec_count == 1 && default_field_count == 0` (Option A, C1-L-002-refined: zero FIELDS, not
  `is_unpartitioned()` — single-spec all-Void walks for null-tuple arity).
- Doc comment fixed (unpartitioned *table* ≠ unpartitioned *default*).
- Pins: unit predicate + mutation twin; e2e `test_delete_mread_after_drop_partition_field_no_resurrection`.

### Leg B (QB bounds)
- `position_delete_writer_properties()` → `set_statistics_truncate_length(None)`.
- Wired: DF `write_position_deletes_for_partition`, rewrite_pos/convert_eq/rewrite_table_path maintenance,
  remove_dangling fixture, pos-delete unit `make_writer_builder`.
- MetricsConfig::for_position_delete already present on DF path (Full on file_path/pos).
- Pin: `test_position_delete_long_file_path_bounds_are_full_and_equal` (120-char path, full equal bounds).
- Does **not** write `referenced_data_file` (Java-parity).

### Matrix
- R113: prose note 2026-08-03; stays 🟡 (Java-read interop of evolved-DROP leg still owed for flip).
- R117: note path-leg now has real full bounds.
- anchors OK.

### Interop
- Primary: e2e DF DELETE after Rust `update_partition_spec().remove_field` + live scan (zero resurrection).
- Java-read evolved-DROP suite: follow-up if not landed before push (Q9 fallback accepted).

## Mutations (in-octo)
- Restore unconditional `default_is_unpartitioned` only → unit `test_pos_delete_fast_path_mutation_default_only_is_wrong` RED design.
- Drop `set_statistics_truncate_length(None)` → long-path bounds pin RED.

## Octo (8× early_stop=false) — OCTO-CONVERGED

Tip: `1a18e718`. Mutations RED: Option A weaken; Leg B re-enable 64-byte truncate.
Scratch: `/tmp/critic-octo-u1-qb-bug001-2026-08-03/OCTO-REPORT.md`

R113 stays 🟡 (Java-read interop evolved-DROP still a seed under Q9 fallback).
