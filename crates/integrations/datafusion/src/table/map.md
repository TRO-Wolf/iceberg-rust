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

# map.md — crates/integrations/datafusion/src/table/

## Purpose

DataFusion `TableProvider` implementations. Metadata-table `scan` honors `projection`
(row R169).

## Contents

| File | Role |
|---|---|
| `mod.rs` | `IcebergTableProvider` (catalog-backed, writes). `schema()` advertises the metadata-free Arrow schema (`strip_metadata_from_schema`) because DataFusion's `insert_to_plan`/`Values` exec type literals against it and reject stamped nested field ids (F-LIST-INSERT-1 r2); the internal stamped schema still feeds `scan` and the write path. `with_output_spec_id(i32)` targets writes at any existing partition spec (Java `output-spec-id`); `insert_into` resolves it via `iceberg::writer::resolve_output_spec` and threads the spec through `project_with_partition` / `repartition` / `IcebergWriteExec` / `IcebergCommitExec` (F-OUTPUT-SPEC-ID-1). `with_uuid_as_string(bool)` (F-UUID-STRING-1, default off) advertises `Utf8` for uuid fields and converts text to bytes on `insert_into`; DELETE / UPDATE prune on the byte-rewritten filters but evaluate the row predicate and every `SET` value over the uuid-rendered text schema (`uuid_text_expr.rs`), so SELECT and DML agree on Spark's case-sensitive string semantics |
| `uuid_text.rs` | F-UUID-STRING-1: the uuid-as-text engine — uuid field-id collection, byte-to-text Arrow schema rewrite, uuid parsing, canonical rendering, text-to-bytes batch conversion (`UuidTextToBytesExec`). `parse_uuid_text` mirrors Java `UUID.fromString` as iceberg-spark writes it (measured on Spark 4.1.2 / Java 17): over 36 UTF-16 units refuses `UUID string too large`; anything but exactly four hyphens refuses `Invalid UUID string: <value>`; each group is a `Long.parseLong(…, 16)` (optional `+`, any length, masked to 8/4/4/4/12 hex digits, so `1-2-3-4-5` stores `00000001-0002-0003-0004-000000000005` and `123456789-…` keeps `23456789`), and a bad group refuses `NumberFormatException: Error at index <i> in: "<group>"` (empty group: `NumberFormatException: `). Divergence: groups accept ASCII hex only, where Java's `Character.digit` also takes other Unicode digits |
| `uuid_text_filters.rs` | F-UUID-STRING-1 filter side: `rewrite_uuid_text_filters` turns uuid-column text literals into byte literals for the Iceberg prune only (`<>`, `NOT IN`, ranges and negated `=` keep a non-canonical literal unrewritten, since byte order would prune rows the text comparison keeps); `refuse_unbindable_uuid_delete_filters` mirrors Spark DELETE, which binds its filter to the uuid type only when the whole condition converts to an Iceberg expression (`SparkTable.canDeleteWhere`): every conjunct built from column-vs-literal comparisons (null-safe `<=>` / `IS [NOT] DISTINCT FROM` included), `IN`/`NOT IN` of literals, `IS [NOT] NULL`, wildcard-free or prefix `LIKE`, and `AND`/`OR`/`NOT` of those; then it refuses `=`/`<>`/`<=>`/`IS [NOT] DISTINCT FROM`/range/`IN`/`NOT IN` and wildcard-free `LIKE`/`NOT LIKE` with an unparsable literal (`Invalid UUID string: abc`) and prefix `LIKE`/`NOT LIKE` (`Term for STARTS_WITH or NOT_STARTS_WITH must produce a string: ref(id=<id>, accessor-type=uuid): uuid`); UPDATE never refuses (Spark measured `UPDATE … WHERE u = 'abc'` updating 0 rows). A condition with any other branch (a function call such as `upper(t)`, column-vs-column, a non-prefix `LIKE`) is never refused and evaluates as strings over the rows (Spark measured `u = 'abc' OR upper(t) = 'X'` deleting one row). Divergence: Spark DELETE with `ILIKE` on a uuid column dies with a `ClassCastException`; here it evaluates over the text. Divergence (2026-09-25, follow-up F-UUID-METADATA-DELETE-1): Spark's metadata-only DELETE drops whole files strictly matched by the byte predicate without a scan, so on two one-row files `(1, U1)`, `(2, U2)` `DELETE WHERE u = '<U1 upper-case>'` keeps `[2]`, and on files holding `00000001-0002-0003-0004-000000000005` and `U2` `DELETE WHERE u = '1-2-3-4-5'` keeps `[2]` (Spark's own SELECT with that predicate returns no row); here every DELETE evaluates rows over the text, keeping `[1, 2]` in both (both layouts pinned by `uuid_delete_evaluates_rows_where_spark_drops_whole_files_by_bytes`). Divergence (2026-09-25, F-UUID-DELETE-OPTSHAPE-1): whether the whole condition converts depends on each engine's optimizer output for the OTHER conjuncts, so the two engines disagree on these shapes over rows `(1,U1)`, `(2,U2)`, `(3,NULL)`, `(4,U0)` in both CoW and MoR: `CAST(id AS STRING) = '1' AND u = 'abc'` Spark commits keeping `[1, 2, 3, 4]` (the cast side is no column reference), here DataFusion unwraps the cast into column-vs-literal and it refuses `Invalid UUID string: abc`; `u = 'abc' AND id + 0 = 1`, `u = 'abc' AND id = 1.0` and `u = 'abc' AND id IN (1, 2.5)` Spark refuses `Invalid UUID string: abc` (its optimizer folds the id side back to a plain column), here they commit keeping `[1, 2, 3, 4]`; the fork's answers are pinned by `uuid_delete_conversion_follows_this_engines_optimizer_shape` so a later change is deliberate |
| `uuid_text_expr.rs` | F-UUID-STRING-1 DML evaluation: `UuidTextExpr` renders the byte table batch to the text schema and evaluates the text-planned expression on it; for a `SET` whose column type changes it converts the result back to bytes through the same parser, nulling rows the UPDATE filter does not match first so an unmatched row's value never refuses |
| `uuid_as_string_tests.rs` | F-UUID-STRING-1 pins: text advertisement, canonical scan rendering, direct-bytes rendering, pushdown file parity, upper-case insert, invalid-string refusal, nested struct round-trip, row-level DELETE/UPDATE, byte path unchanged; follow-up pins: static provider text advertisement + pinned-row rendering + text/byte file parity + byte path, catalog and schema provider propagation with metadata tables resolving |
| `uuid_as_string_dml_tests.rs` | F-UUID-STRING-1 Spark-measured pins (Spark 4.1.2 + iceberg-spark 1.11.0, CoW and MoR): DELETE `=`/`<>`/`IN`/`NOT IN`/ranges/`IS NULL`/`LIKE`/short-group literals with upper-case literals and row counts; DELETE refusals (only when the whole condition converts, null-safe `<=>` included; `upper(t)` branches evaluate over the text); the metadata-delete and optimizer-shape divergences; UPDATE `WHERE` string semantics; `SET u = NULL`, `= t`, `= upper(u)`, upper-case and short-group literals, `concat(t, 'z')` refusing `UUID string too large`; SELECT case-sensitive `<>`/`NOT IN`/`>`/`>=`/`<`/`<=` against multi-file pruning; `IN`/`<`/`>`/`<>` file counts; option-off byte-literal pruning 1 of 2 files with unchanged rows; list/map round-trip; the Java parser table |
| `loaded.rs` | `IcebergTableProvider` loaded-Table paths (`from_planning_load`, planning fast path) |
| `static_provider.rs` | `IcebergStaticTableProvider` (one snapshot, read-only). `schema()` also advertises the metadata-free schema for the same `insert_to_plan`/`Values` reason (F-LIST-INSERT-1 r3); the stamped schema still feeds `scan`. `with_uuid_as_string(bool)` (F-UUID-STRING-1 follow-up, default off) advertises `Utf8` for uuid fields and reuses the table provider's text rendering and filter rewrite against the pinned snapshot's schema |
| `metadata_table.rs` | `IcebergMetadataTableProvider` — inspect tables as DataFusion tables |
| `table_provider_factory.rs` | DataFusion factory for `CREATE EXTERNAL TABLE` |
| `tests.rs` | `#[cfg(test)]` unit tests: provider construction, static provider, partitioning/sort/limit plans, shared fixtures (`pub(super)` helpers reused by `schema_evo_tests.rs`) |
| `schema_evo_tests.rs` | `#[cfg(test)]` unit tests: schema-evolution cells — stale providers, delete/update binding to the current schema, renames, nested evolution, pushdown after evolution |

## I want to...

| I want to... | go to |
|---|---|
| Project metadata-table columns | `metadata_table.rs` `TableProvider::scan` → [../physical_plan/map.md](../physical_plan/map.md) `metadata_scan.rs` |
| Bind a catalog table | `mod.rs` `IcebergTableProvider` |
| Time-travel a snapshot | `static_provider.rs` |

## Pointers

- **Up:** [../map.md](../map.md) · **Related:** [../physical_plan/map.md](../physical_plan/map.md),
  [../../../../iceberg/src/inspect/map.md](../../../../iceberg/src/inspect/map.md)

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| Projected metadata scan schema is the full schema | `TableProvider::scan` dropped `projection` (row R169) |
| `SELECT count(*)` over `$snapshots` is wrong | empty projection lost row count in `IcebergMetadataScan` |
| `schema()` on the provider is projected | advertised schema must stay full; only the plan schema projects |

### First checks

1. `IcebergMetadataTableProvider::scan` passes `projection` into `IcebergMetadataScan::new`.
2. `TableProvider::schema` still returns the full field set (metadata-free for SQL planning); the
   stamped schema is internal.

### Escalate to

[docs/parity/GAP_MATRIX.md](../../../../../docs/parity/GAP_MATRIX.md) row R169
