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

# F-WRITE-ORDER-TRANSFORM-1 — `ReplaceSortOrderAction` takes transform sort fields

Unit: fork half of D-WRITE-ORDERED-TRANSFORM (GAP_MATRIX row IPI-49). Oracle: the run-25d
Spark write-order oracle (`write_order_truth.json`, Spark 4.1.2 + Iceberg 1.11.0, hadoop
catalog; table `(id BIGINT, cat STRING, ts TIMESTAMP, v DOUBLE, s STRING)`, v2). Java source
of record: `iceberg-api` `SortOrder$Builder` (decompiled 1.11.0) + `Spark3Util.toIcebergTerm` /
`findWidth` (decompiled 1.11.0 spark-runtime).

## Oracle cell table

| cell | statement(s) | result | error |
|---|---|---|---|
| IDENTITY | `WRITE ORDERED BY id` | order 1 `{identity, src 1, asc, nulls-first}`, default 1 | — |
| IDENTITY-DESC-NULLS | `WRITE ORDERED BY id DESC NULLS LAST, s ASC NULLS FIRST` | order 1 with both fields | — |
| BUCKET | `WRITE ORDERED BY bucket(4, id)` | order 1 `{bucket[4], src 1, asc, nulls-first}` | — |
| TRUNCATE-STR | `WRITE ORDERED BY truncate(s, 2)` | `truncate[2]` on src 5 | — |
| TRUNCATE-NUM | `WRITE ORDERED BY truncate(id, 10) DESC` | `truncate[10]` desc nulls-last | — |
| DAYS | `WRITE ORDERED BY days(ts)` | `day` on src 3 | — |
| HOURS | `WRITE ORDERED BY hours(ts) DESC NULLS FIRST` | `hour` desc nulls-first | — |
| YEARS-MONTHS | `WRITE ORDERED BY years(ts), months(ts)` | `year`, `month` on src 3 | — |
| MIXED | `bucket(8, cat), id DESC, truncate(s, 3) NULLS LAST` | 3 fields, mixed transforms | — |
| LOCALLY / DISTRIBUTED-LOCALLY | local orderings | same metadata as global | — |
| UNORDERED-AFTER | ordered then `WRITE UNORDERED` | default back to order 0 (empty order kept in list) | — |
| REUSE-SAME | id, s, id | ids 1, 2, then back to 1 — no new entry | — |
| REUSE-SAME-TRANSFORM | bucket(4,id), s, bucket(4,id) | ids 1, 2, then back to 1 | — |
| BUCKET-BAD-N | `bucket(0, id)` | — | `IllegalArgumentException`: `Unsupported width for transform: bucket(0, id)` |
| TRUNCATE-BAD-W | `truncate(s, 0)` | — | `IllegalArgumentException`: `Unsupported width for transform: truncate(s, 0)` |
| TRANSFORM-BAD-TYPE | `days(id)` | — | `ValidationException`: `Cannot bind: day cannot transform long values from 'id'` |
| UNKNOWN-TRANSFORM | `nosuch(id)` | — | `UnsupportedOperationException`: `Transform is not supported: nosuch(id)` |
| VOID | `void(id)` | — | `UnsupportedOperationException`: `Transform is not supported: void(id)` |
| EXPR | `id + 1` | — | Spark `AnalysisException` (parser layer, not this unit) |
| NESTED | `bucket(4, id), bucket(4, id)` | BOTH identical fields kept | — |
| PARTITIONED-TABLE | `bucket(4, id)` on partitioned table | same; file stamp 1 | — |

## What the fork does today (measured by reading the code — stated per cell)

Measurement method: code reading of `crates/iceberg/src/transaction/sort_order.rs`,
`spec/sort.rs`, `spec/transform.rs`, `spec/table_metadata_builder.rs`, and
`maintenance/rewrite_data_files_write.rs`. No scratch test needed — the API surface is
absent, so most cells are structurally inexpressible.

| cell | fork today |
|---|---|
| IDENTITY, IDENTITY-DESC-NULLS | supported: `asc`/`desc` emit identity `SortField`s; commit binds via `field_id_by_name` + `SortOrder::builder().build` |
| BUCKET, TRUNCATE-*, DAYS, HOURS, YEARS-MONTHS, MIXED, LOCALLY, DISTRIBUTED-LOCALLY, PARTITIONED-TABLE | **inexpressible** — `PendingSortField` hard-codes `Transform::Identity`; no method accepts a transform |
| UNORDERED-AFTER | works today: an action with no fields builds `SortOrder::unsorted_order()` (id 0); `TableMetadataBuilder::add_sort_order` maps unsorted → id 0 and `set_default_sort_order(-1)` makes it default |
| REUSE-SAME, REUSE-SAME-TRANSFORM | reuse rule EXISTS at `TableMetadataBuilder::reuse_or_create_new_sort_id` (fields equality → same id; else `highest+1`) — already shared by every path; needs a pin through the action |
| BUCKET-BAD-N, TRUNCATE-BAD-W | inexpressible today (`Transform::Bucket(0)`/`Truncate(0)` are constructible but unreachable through the action); `Transform::validate()` exists but uses the core-Java message (`Invalid number of buckets: 0 (must be > 0)`), not the write-order surface's `Unsupported width for transform: …` |
| TRANSFORM-BAD-TYPE | inexpressible today; `SortOrderBuilder::check_compatibility` would catch it as `Invalid source type …`/`ErrorKind::Unexpected` at build, NOT Java's bind-time `Cannot bind: day cannot transform long values from 'id'` |
| UNKNOWN-TRANSFORM, VOID | inexpressible today; Java core would accept a void term but the write-order surface rejects it in `Spark3Util.toIcebergTerm` — the action must reproduce the refusal |
| NESTED (duplicate terms) | works by construction — `pending_sort_fields` is a `Vec`; duplicates kept |
| nested source `a.b` | `Schema::field_id_by_name`/`field_by_name` resolve dotted paths through `name_to_id` (index carries `person.name` etc.) — supported today for identity; pin it for a transform term |

## Java mechanics verified (1.11.0)

- `BaseReplaceSortOrder` holds a `SortOrder.Builder` over the CURRENT schema; `asc/desc(Term,
  NullOrder)` binds each `UnboundTerm` IMMEDIATELY (`UnboundTransform.bind` →
  `ValidationException.check(canTransform, "Cannot bind: %s cannot transform %s values from
  '%s'")`), converts `BoundReference → identity`, `BoundTransform → its transform`, and
  accumulates `SortField`s; `apply()` → `build()` → `buildUnchecked` + `checkCompatibility`
  (`Cannot find source column` / `Cannot sort by non-primitive source field` / `Invalid source
  type %s for transform: %s`).
- `Spark3Util.toIcebergTerm`: `identity`→ref, `bucket`/`truncate`→term via `findWidth`
  (`checkArgument(width > 0 && width < Integer.MAX_VALUE, "Unsupported width for transform:
  %s", transform.describe())`), `year(s)/month(s)/date/day(s)/date_hour/hour(s)`→temporal
  terms, `zorder`→`Zorder`, default → `UnsupportedOperationException("Transform is not
  supported: " + transform)`. `void` and unknown names hit the default arm. `describe()` shapes:
  `bucket(0, id)` (width first), `truncate(s, 0)` (column first), `void(id)`.
- `TableMetadata.Builder.reuseOrCreateNewSortId`: unsorted → 0; first existing order with equal
  fields → its id; else `max(existing ids)+1`. Fork equivalent:
  `reuse_or_create_new_sort_id` (HashMap `find_map` — nondeterministic only if two stored
  orders share identical fields, an unreachable state through `addSortOrder`).

## Design (decisions taken in this unit)

- `PendingSortField` gains a `transform: Transform`. `asc`/`desc` keep working unchanged
  (`Transform::Identity`). New method `sort_by(name, transform, direction, null_order)` — the
  `SortOrder.Builder.sortBy(Term, SortDirection, NullOrder)` analogue; one call carries a fully
  parsed `(transform, source column, direction, null order)` term for RePark's parser.
- Validation stays deferred to commit (the file's stated contract — builders stay infallible):
  order per field is (1) `Void`/`Unknown` → `FeatureUnsupported`, `Transform is not supported:
  {void|unknown}({name})` — Java `UnsupportedOperationException` analogue; (2) `Bucket(n)`/
  `Truncate(n)` with `n == 0 || n > i32::MAX` → `DataInvalid`, `Unsupported width for transform:
  {describe}` — Spark `findWidth` shape, `bucket(n, col)`/`truncate(col, w)` arg order; (3)
  missing column → existing `DataInvalid` `Cannot find field {name} in table schema`; (4)
  non-identity transform vs source type via `result_type` → `DataInvalid`, `Cannot bind:
  {transform} cannot transform {type} values from '{name}'` — Java `UnboundTransform.bind`
  shape. Identity skips (4): Java's ref bind performs no transform check — a struct source with
  identity reaches `checkCompatibility`'s `Cannot sort by non-primitive source field` instead.
- `pub use sort_order::ReplaceSortOrderAction` added to `transaction/mod.rs`: the module's
  own comment block states write-action builders are re-exported so an external engine can NAME
  the type; this unit exists to hand RePark a callable surface. (`missing_docs` then applies to
  `pub fn new()` — per RULE 0 it takes `#[allow(missing_docs)]`.)

## Named divergences / residues

- `Transform::Unknown` carries no name payload, so the action's `Transform is not supported:
  unknown({col})` cannot render the original token (`nosuch`). Java's message comes from the
  Spark `Transform`'s toString — RePark's parser holds the original token and emits its own
  refusal for names it cannot map, matching Spark's layering.
- Missing-column message stays the fork's `Cannot find field {name} in table schema`
  (Java `NamedReference.bind`: `Cannot find field '{name}' in struct: {struct}`). Pre-existing
  fork message, shared with `asc`/`desc`; not changed in this unit.
- Type-vs-transform on a STRUCT source renders the fork `Type` Display (`struct<…>`), which
  differs in field-detail level from Java's `struct<1: name: optional string>` rendering;
  message SHAPE (`Cannot bind: bucket[4] cannot transform struct<…> values from '…'`) matches.
- `check_compatibility`'s spec-layer message `Invalid source type {t} for transform {tr}` lacks
  Java's colon (`for transform: %s`) and uses `ErrorKind::Unexpected` vs Java's
  `ValidationException`. Reachable only for orders built outside this action; left as a named
  residue, not touched.
- Reuse dedup on a pathological metadata file containing two stored orders with identical
  fields picks an arbitrary one (`HashMap::find_map`); unreachable through `addSortOrder`.

## RePark parser needs (API contract handed to the next unit)

- `tx.replace_sort_order().sort_by(col, transform, direction, null_order)` per term; `asc`/
  `desc` unchanged for identity sugar.
- `WRITE UNORDERED` = apply the action with no fields (unsorted → default 0).
- Errors: `FeatureUnsupported` = unsupported transform term; `DataInvalid` = bad width, missing
  column, cannot-bind type. All non-retryable.
- Case sensitivity: not exposed — resolution is case-sensitive (Java default).

## Findings

- **Write path (step 4, measured by filtered test `binpack_transform_default_order_sorts_by_bucket_key_and_stamps`):**
  `rewrite_data_files_write.rs::rewrite_sort_plan` already evaluates transforms — each non-identity
  field builds a `BoxedTransformFunction` (`create_transform_function`) applied to the key array in
  `sort_group_batch`, and `write_sorted_run` stamps every output file `with_sort_order_id(stamp)`.
  Pin: a v2 table whose DEFAULT order is `bucket[4](y) asc nulls-first` is bin-pack-compacted so each
  output file's rows are non-decreasing in `bucket[4](y)` (verified by re-applying the same transform
  function to the written `y` column), the file order provably differs from identity order
  (`differs_from_identity_sort`), and `file.sort_order_id() == Some(order_id)`. `Transform::Void`
  fields are skipped, an unresolvable field/order falls back to unsorted with a 0 stamp.
- **DataFusion INSERT path (verified by code reading, NOT fixed or extended here):**
  `integrations/datafusion/src/physical_plan/sort.rs::write_sort_plan` already maps a transform-bearing
  default order into `PhysicalSortExpr`s — non-identity fields are wrapped in `SortTransformExpr`
  (evaluates `create_transform_function` inside the plan), nested sources in `NestedFieldExpr`, and
  `sort_for_write` returns `sort_order_id` which `IcebergWriteExec` stamps via
  `DataFileWriterBuilder::with_sort_order_id`. INSERT therefore already honors a transformed default
  order; no finding to record, no change made.
- **Nested source:** `person.name` binds through `field_by_name` and is pinned by
  `test_sort_by_binds_nested_column_source` (`truncate[3]` on src 3). A transform on the struct itself
  (`bucket(4, person)`) is rejected by the cannot-bind rule
  (`test_sort_by_rejects_transform_on_struct_source`).

## Mutation table

Each mutation was applied, the listed filtered run executed, the failure recorded, and the code
restored byte-identically (`git status` clean before the step-5 commit).

| mutation | site | filtered run | pins that went red |
|---|---|---|---|
| transform dropped to identity (action) | `PendingSortField::to_sort_field` built `.transform(Transform::Identity)` | `cargo test -p iceberg --lib sort_order` | `test_sort_by_commits_transform_fields`, `test_sort_by_commits_temporal_transforms`, `test_sort_by_binds_nested_column_source` |
| transform dropped in the write path | `sort_group_batch` skipped `function.transform(array)` | `cargo test -p iceberg --lib binpack` | `binpack_transform_default_order_sorts_by_bucket_key_and_stamps` (identity order [None,0,1,2,…] emitted; bucket keys not non-decreasing) |
| id reuse removed | `reuse_or_create_new_sort_id` always returned `highest+1` | `cargo test -p iceberg --lib sort_order` | `test_reapplied_equal_sort_order_reuses_its_order_id` (reapplied order got id 3 instead of 1) |
| width/unsupported validation removed | `to_sort_field` no longer called `check_supported_transform` | `cargo test -p iceberg --lib sort_order` | `test_sort_by_rejects_bad_transform_widths`, `test_sort_by_rejects_unsupported_transforms` (commits succeeded where Java refuses) |
| cannot-bind validation removed | `to_sort_field` dropped the `result_type` check | `cargo test -p iceberg --lib sort_order` | `test_sort_by_rejects_transform_type_mismatch`, `test_sort_by_rejects_transform_on_struct_source` — the commit still erred, but via `check_compatibility` as `ErrorKind::Unexpected`, not `DataInvalid` with Java's `Cannot bind: …` shape; the pin proves the action-level check is what produces the Java surface |

## Round 2 — verification-critic residues (V-01, V-02, V-03)

### V-01 — YEARS-MONTHS cell now pinned

`test_sort_by_commits_year_and_month_transforms`: `WRITE ORDERED BY years(ts), months(ts)` analogue
— `sort_by("ts", Year, Asc, First)` + `sort_by("ts", Month, Asc, First)` commits one order (id 1)
with two fields `{year, src 2, asc, nulls-first}` and `{month, src 2, asc, nulls-first}` — the
oracle's exact field strings (its `ts` is source-id 3; the test schema's `ts` is source-id 2).

### V-02 — width boundary: decompiled evidence, action's boundary IS Java's

The critic's premise was checked against the actual bytecode, not recollection. Decompiled with
`javap -c` from `iceberg-spark-runtime-4.1_2.13:1.11.0` (the oracle's own GAV):

`Spark3Util.findWidth(Transform)` has TWO literal-type arms:

- **IntegerType arm** (bytecode 53–99): `checkArgument(literal.intValue() > 0,
  "Unsupported width for transform: %s", transform.describe())`, then `return intValue`.
  NO upper bound — any positive `int` is accepted, including `Integer.MAX_VALUE`.
- **LongType arm** (bytecode 116–219): `checkArgument(value > 0 && value < 2147483647L, same
  message)`, then `Math.toIntExact`-style `> MAX → IllegalArgumentException`, `return intValue`.
  Strictly `< i32::MAX` — but reachable ONLY for `L`-suffixed or over-int literals.

`Bucket.get(int)` and `Truncate.get(int)` each check only `width > 0`
(`"Invalid number of buckets/truncate width: %s (must be > 0)"`) — no upper bound. Java core
accepts `Expressions.bucket("id", 2147483647)`; `int` itself is the ceiling.

Effective Spark-SQL surface: `bucket(2147483647, id)` parses the literal as `IntegerType`
(fits `int`) → int arm → ACCEPTED, commits `bucket[2147483647]`; `bucket(2147483648, id)` is a
`LongType` literal → long arm → refused with `Unsupported width for transform:
bucket(2147483648, id)`; `bucket(2147483647L, id)` (`L` suffix → LongType) → also refused —
a typing distinction the fork's `Transform::Bucket(u32)` API cannot and should not see.

Action boundary stays `1..=i32::MAX` (`width == 0 || i32::try_from(width).is_err()` refused):
exactly what Java's core layer can express and what the Spark surface accepts for ordinary
integer literals. The `2147483647L` divergence is a parser-layer concern — RePark holds the
literal's type and can refuse it there, matching Spark's layering where `findWidth` sees the
`Literal`'s `DataType`. Recorded as a named residue, NOT a bug.

Pins added: `test_sort_by_accepts_width_at_java_int_max` — `bucket[2147483647]` and
`truncate[2147483647]` both COMMIT (the boundary value Java accepts, contrary to the critic's
premise); `truncate(y, 2147483648)` refusal added beside the existing `bucket(x, 2147483648)`
pin — `> i32::MAX` is the boundary Java refuses (long arm + unrepresentable in a Java `int`
and in `Transforms.fromString`'s `Integer.parseInt` round-trip).

### V-03 — nested pin asserts the transform

`test_sort_by_binds_nested_column_source` now asserts the committed `SortField` directly:
`field.source_id == 3` AND `field.transform == Transform::Truncate(3)`, in addition to the
serialized `truncate[3]` JSON assertion that was already there.

### Round-2 gates

| gate | result |
|---|---|
| `cargo test -p iceberg --lib sort_order` | 34 passed, 0 failed |
| `cargo fmt --all -- --check` | clean |
| `cargo clippy -p iceberg --all-targets -- -D warnings` | clean |
| `./scripts/check_rust_file_size.sh` | clean |
| `typos` on touched files | clean |
| `python3 comment_ban.py` vs `origin/main` | `comment-ban hits=0` |

## Gates (step 6)

| gate | result |
|---|---|
| `cargo test -p iceberg --lib sort_order` | 32 passed, 0 failed |
| `cargo test -p iceberg --lib rewrite_data_files_lineage` | 7 passed, 0 failed |
| `cargo test -p iceberg --lib binpack` (mutation-era run) | pin red under the write-path mutation, green after restore |
| `cargo fmt --all -- --check` | clean |
| `cargo clippy -p iceberg --all-targets -- -D warnings` | clean |
| `./scripts/check_rust_file_size.sh` | 572 files clean (92 legacy ceilings) |
| `typos` on touched files | clean |
| `python3 comment_ban.py` vs `origin/main` | `comment-ban hits=0` before every commit |

### Q-25d-2 — the width bound, measured (orchestrator)

The logic critic's L-001 and the verification critic's V-02 both claimed Spark refuses a bucket width at or above
`Integer.MAX_VALUE`. Measured instead of recalled (`check_width_bound.py` in the run-25d Spark write-order oracle,
Spark 4.1.2 + Iceberg 1.11.0):

| width | Spark |
|---|---|
| `bucket(0, id)` | refused, "Unsupported width for transform: bucket(0, id)" |
| `bucket(1, id)` | accepted |
| `bucket(2147483647, id)` | **accepted**, order `bucket[2147483647]` |
| `bucket(2147483648, id)` | refused, "Unsupported width for transform: bucket(2147483648, id)" |

The action's boundary (refuse 0 and anything above `i32::MAX`) is already Java's, so the finding is REFUTED and no
code changed; the round-2 pins carry the measured boundary.
