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

# Ledger — F-LIST-NULL-ACCESSOR-1: `IS NULL` / `IS NOT NULL` on a list, map or struct column

**Ledger id:** `F-LIST-NULL-ACCESSOR-1-2026-09-19`
**Branch:** `fix/f-list-null-accessor-1` (cut off fork `main` = `9e67e000`)
**Scope:** devin-worker brief F-LIST-NULL-ACCESSOR-1
**Model:** swe-2
**Found by:** #295 logic critic (F-LIST-INSERT-1 round-3 residue note)

## 1. Defect

`DELETE FROM t WHERE xs IS NULL` (and `IS NOT NULL`, and the same filter on UPDATE
and SELECT) failed loud with `Accessor for Field xs not found` whenever `xs` was a
`list<…>`, `map<…>` or `struct<…>` column.

Mechanism:

1. `expr_to_predicate.rs` `Expr::IsNull` / `Expr::IsNotNull` converts a null test on
   ANY column into `Predicate::Unary(IsNull|IsNotNull, Reference(name))`.
2. `predicate_binds_soundly` (F-OCC-EXEC-1, #294) validated the `Unary` arm only as
   `schema.field_by_name(term.name()).is_some()` — existence, not bindability.
3. `Reference::bind` (`crates/iceberg/src/expr/term.rs:326`) requires
   `schema.accessor_by_field_id(field.id)` and raises
   `DataInvalid => Accessor for Field <name> not found` when absent.
4. `SchemaBuilder::build_accessors` (`crates/iceberg/src/spec/schema/mod.rs`) creates
   accessors for top-level primitive fields and for primitive/struct descendants
   inside structs — never for the struct container itself, nor for list/map fields
   or their element/key/value children (Java's `Accessors` has none either).

So every null test on a container column passed conversion and then raised at bind
time. Binary/Set predicates on container columns already dropped soundly, because
`literal_binds_soundly` runs `Datum::to(&field.field_type)`, which rejects
non-primitive targets — the `Unary` arm was the only hole with no datum to convert.

## 2. Call sites that convert and then bind (and which raised)

| Call site | Path | Raised pre-fix |
|---|---|---|
| `IcebergTableProvider::delete_from` | `table/mod.rs:238` `prune = convert_filters_to_predicate(...)` → `IcebergDeleteExec` → `mor_scan_stream` (`mor_scan.rs:44`) / `cow_scan_stream` (`row_lineage.rs:171`) → `with_file_prune_only` → `Predicate::bind` | YES — `Accessor for Field xs not found` at `DML commits` |
| `IcebergTableProvider::update` | `table/mod.rs:274`, same construction via `IcebergUpdateExec` → same scan seams | YES — identical error at `DML commits` |
| Scan filter pushdown (SELECT) | `scan.rs:171,446` `scan_builder.with_filter(pred)` → `scan/mod.rs:507` `predicates.bind(schema, …)` | YES — identical error at `select`/`DML commits` |
| Conflict-validation filter (#294) | `delete.rs:529,670,986,1123` `conflict_detection_filter(prune)` → `transaction/snapshot/conflict_filter.rs` `first_conflicting_file` binds the same predicate | NOT REACHED — the target-scan prune bind raises during planning/execution before commit validation runs; it binds the identical predicate and would raise identically |

## 3. Measured oracle

The run-23a list-null oracle (Spark 4.1.2 + Iceberg 1.11.0, all 128 cells commit;
identical answers on format v2 and v3 and on copy-on-write and merge-on-read).
Table `(id INT, xs <shape>)`, four rows per shape:

| shape | row 1 | row 2 | row 3 | row 4 |
|---|---|---|---|---|
| `list<int>` | `[1, 2]` | NULL | `[]` | `[NULL]` |
| `list<struct<a: int>>` | `[{a:1}]` | NULL | `[]` | `[NULL]` |
| `map<string, int>` | `{k:1}` | NULL | `{}` | `{k:NULL}` |
| `struct<a: int>` | `{a:1}` | NULL | `{a:NULL}` | `{a:4}` |

| statement | surviving ids (every shape, version, mode) |
|---|---|
| `DELETE … WHERE xs IS NULL` | 1, 3, 4 |
| `DELETE … WHERE xs IS NOT NULL` | 2 |
| `DELETE … WHERE id > 1 AND xs IS NULL` | 1, 3, 4 |
| `DELETE … WHERE xs IS NULL OR id = 1` | 3, 4 |
| `UPDATE … SET id = id + 100 WHERE xs IS NULL` | 1, 3, 4, 102 |
| `UPDATE … WHERE xs IS NOT NULL` | 2, 101, 103, 104 |
| `UPDATE … WHERE id > 1 AND xs IS NULL` | 1, 3, 4, 102 |
| `UPDATE … WHERE xs IS NULL OR id = 1` | 3, 4, 101, 102 |

An empty list/map is NOT null; a list holding a NULL element is NOT null; a struct
whose child is NULL is NOT null.

## 4. Choice — (a) exact-or-drop, mirroring `Reference::bind`

Chose **(a)**: `predicate_binds_soundly` now requires, for every term-bearing
predicate (`Unary`, `Binary`, `Set`), what `Reference::bind` requires — the field
exists AND `accessor_by_field_id(field.id)` resolves. A new `term_binds_soundly`
helper carries the check; `Binary`/`Set` arms call it in addition to
`literal_binds_soundly`, closing the same hole for the reachable-but-accessorless
leaf names (e.g. `xs.element`, `m.key`, `m.value` resolve via `field_by_name` but
have no accessor).

Rationale:

- Option (b) — make binding succeed and teach every evaluator (manifest,
  row-group, page, expression) a null-test answer for accessorless columns — is a
  cross-crate change to `iceberg` evaluators for a column class Java itself gives
  no accessor for. Manifest/row-group stats for list/map/struct fields do not feed
  the existing evaluators' null-count path the way primitive leaves do.
- Option (a) keeps the #294 rule — a pushed predicate must select a superset of
  the files holding matching rows — with a one-helper change in the file that
  already owns fail-closed conversion. The exact DataFusion predicate is the row
  contract at every call site (`delete_from`/`update` build the PhysicalExpr from
  the raw filters; the scan attaches residuals), so dropping the unbindable term
  cannot lose or keep a wrong row: it only widens the scan.
- A term naming the struct leaf `s.a` directly still converts, because the leaf has an
  accessor (the conversion-level cell). Real SQL `s.a IS NULL` plans as `get_field` and is
  evaluated as a residual (round 2, section 9). `struct IS NULL` does not convert, because
  the container has no accessor. Both SQL answers match the oracle.

## 5. Red (step 1, commit `d744ac15`)

Harness: new file `crates/integrations/datafusion/src/physical_plan/list_null_tests.rs`
(503 lines, under the 1000 default ceiling; `occ_exec_tests.rs` had ~207 lines of
headroom but the cells ×4 shapes ×2 versions ×2 modes ×8 statements wanted a file
of their own). Wired `#[cfg(test)] mod list_null_tests` in `physical_plan/mod.rs`.
Fixture: `MemoryCatalog` + `IcebergCatalogProvider` + `SessionContext`, seeds via
`INSERT INTO catalog.ns.t VALUES …` (`[1,2]`, `NULL`, `[]`, `[NULL]`,
`named_struct`, `map`, `MAP {}`, `CAST(NULL AS STRUCT<a INT>)` all accepted by
DataFusion 54 planning). Unit cells appended to `expr_to_predicate_tests.rs`
(909 lines, under ceiling).

`cargo test -p iceberg-datafusion --lib is_null_` + `--lib is_not_null` on the
pre-fix tree — 12 of 14 cells red:

```text
test result: FAILED. 2 passed; 12 failed; 0 ignored; 0 measured; 264 filtered out
(plus the separate is_not_null run: FAILED. 0 passed; 2 failed; 278 filtered)

expr_to_predicate::tests::is_null_on_a_list_column_is_not_pushed          FAILED left: Some(Unary(IsNull, xs)) right: None
expr_to_predicate::tests::is_null_on_a_map_column_is_not_pushed           FAILED left: Some(Unary(IsNull, m))  right: None
expr_to_predicate::tests::is_null_on_a_struct_column_is_not_pushed        FAILED left: Some(Unary(IsNull, s))  right: None
expr_to_predicate::tests::is_null_on_a_list_element_name_is_not_pushed    FAILED left: Some(Unary(IsNull, xs.element))
expr_to_predicate::tests::is_null_on_a_nested_column_drops_only_its_own_conjunction FAILED (And kept IsNull(xs))
list_null_tests::delete_where_xs_is_null_removes_only_the_null_row        FAILED Accessor for Field xs not found
list_null_tests::delete_where_xs_is_not_null_keeps_only_the_null_row      FAILED Accessor for Field xs not found
list_null_tests::delete_where_id_and_xs_is_null_composes_with_a_primitive_conjunct FAILED same
list_null_tests::delete_where_xs_is_null_or_id_eq_1_keeps_matching_rows   FAILED same
list_null_tests::update_where_xs_is_null_updates_only_the_null_row        FAILED same
list_null_tests::update_where_xs_is_not_null_updates_every_non_null_row   FAILED same
list_null_tests::update_where_id_and_xs_is_null_composes_with_a_primitive_conjunct FAILED same
list_null_tests::update_where_xs_is_null_or_id_eq_1_updates_matching_rows FAILED same
list_null_tests::select_where_xs_is_null_returns_the_null_row             FAILED same
```

Controls green pre-fix as designed: `is_null_on_a_primitive_column_still_pushes`,
`is_null_on_a_struct_leaf_still_pushes` (the `s.a` leaf has an accessor today).

## 6. Green (step 2, commit `d7a4e6d0`)

`cargo test -p iceberg-datafusion --lib null`:

```text
test result: ok. 29 passed; 0 failed; 0 ignored; 0 measured; 249 filtered out
```

All 9 e2e cells pass on every shape × v2/v3 × copy-on-write/merge-on-read with the
oracle's exact surviving ids; all 8 unit cells pass. Regression filters:

```text
cargo test -p iceberg-datafusion --lib expr_to_predicate → ok. 69 passed; 0 failed
cargo test -p iceberg-datafusion --lib occ_exec          → ok. 19 passed; 0 failed
```

## 7. Mutation proof (step 3)

Revert = `git checkout d744ac15 -- expr_to_predicate.rs` (pre-fix file only, not
committed). Restore = `git checkout HEAD -- …` + `touch` (fresh mtime → rebuild).

```text
mutant:  cargo test -p iceberg-datafusion --lib null
         FAILED. 15 passed; 14 failed; 0 ignored; 249 filtered out
         — every cell of §5 red again, same signatures
restored: cargo test -p iceberg-datafusion --lib null
         ok. 29 passed; 0 failed
```

Every red cell is load-bearing: reverting one helper's accessor check reds all
five conversion pins and all nine e2e cells.

## 8. Answers the brief asked for

- `struct IS NULL` / `struct.a IS NULL`: the container `s` has no accessor —
  `s IS NULL` converts pre-fix and raised at bind, like list/map; post-fix it drops
  to residual evaluation and answers correctly. The leaf `s.a` has an accessor, so
  a term that resolves to it is bindable — but no real SQL produces that term:
  `s.a` plans as `GetField(s, a)`, which conversion already leaves as
  NotTransformed → residual (round 2 pins the real-SQL cell answering `[2, 3]`;
  see §9). The synthetic `Column("s.a")` unit cell is a conversion-level control
  only: it proves a term naming a field WITH an accessor still pushes, not that
  SQL `s.a IS NULL` pushes.
- No row is silently kept/dropped: pushdown is prune-only and the exact DataFusion
  predicate decides every row at every call site; the mutation run proves the
  cells fail without the fix.
- `SELECT … WHERE xs IS NULL` goes through the same `with_filter` bind and raised
  identically pre-fix.

## 9. Round 2 — pin the Binary/Set arms, settle the P3s (commit `f6e89406`)

Grok reviews on round 1: perf LOOKS-GOOD, logic PASS with two P3s.

### L-001 (P3): Binary and Set arms of `term_binds_soundly` were unpinned

Every round-1 cell was a `Unary` null test, so deleting the `term_binds_soundly`
conjunct from the `Binary` or `Set` arm alone left the suite green.

Reachability: no real SQL produces a Binary/Set `Expr::Column` term on an
accessorless field. The DFSchema exposes only top-level columns — `xs.element`,
`m.key`, `m.value` resolve inside the Iceberg schema (`field_by_name` walks
collection children) but are not DataFusion columns; `xs.element = 1` parses as a
qualified name and the quoted identifier `"xs.element"` fails at planning. The
pins are therefore at conversion level (`Column::new_unqualified`, the same
surface the `Unary` leaf pin uses), covering `xs.element`, `m.value` (Int) and
`m.key` (String) — all name-resolvable, all accessorless:

```text
binary_on_an_accessorless_leaf_is_not_pushed     xs.element|m.value = 1, m.key = 'k'  → None
in_list_on_an_accessorless_leaf_is_not_pushed    xs.element|m.value IN (1,2), m.key IN ('k','v') → None
```

Both cells pass post-fix (`cargo test -p iceberg-datafusion --lib accessorless`
→ ok. 2 passed). Mutation proof, each arm reverted alone, not committed:

```text
Binary arm reverted (term_binds_soundly removed from Predicate::Binary):
    binary_on_an_accessorless_leaf_is_not_pushed   FAILED left: Some(Binary(Eq, xs.element, Long(1)))
    in_list_on_an_accessorless_leaf_is_not_pushed  ok (control)
Set arm reverted (term_binds_soundly removed from Predicate::Set):
    in_list_on_an_accessorless_leaf_is_not_pushed  FAILED left: Some(Set(In, xs.element, {1,2}))
    binary_on_an_accessorless_leaf_is_not_pushed   ok (control)
restored (git checkout + touch → rebuild):
    accessorless filter  ok. 2 passed; 0 failed
    null filter          ok. 30 passed; 0 failed
```

Each arm's conjunct is now load-bearing: its own cell reds, the sibling cell
stays green.

### L-002 (P3): the struct-leaf pin used a synthetic `Column("s.a")`

Corrected in §8. Real SQL `s.a IS NULL` plans as `get_field` and is evaluated as
a residual; a NULL struct makes `s.a` NULL too. Real-SQL cell added in
`list_null_tests.rs`:

```text
select_where_xs_dot_a_is_null_returns_the_null_leaf_rows
    SELECT id FROM t WHERE xs.a IS NULL → [2, 3]  (struct shape, v2/v3 × CoW/MoR)
```

### R-01 (P3, recorded — no code change)

An `AND` mixing a primitive conjunct and an accessorless null test inside ONE
`Expr` drops as a whole — `to_iceberg_and_predicate` requires both sides to
convert, and the surviving `And` predicate fails `predicate_binds_soundly` on the
unbindable child, so the sound primitive half is lost to pushdown too (the exact
DataFusion predicate still decides every row; only the prune is weaker). A
top-level split AND — separate conjuncts in `filters` — keeps the sound half,
because `convert_filters_to_predicate` drops per-filter. Pinned by
`is_null_on_a_nested_column_drops_only_its_own_conjunction`:
`[id > 1, xs IS NULL]` → `Some(id > 1)`; `[id > 1 AND xs IS NULL]` → `None`.
