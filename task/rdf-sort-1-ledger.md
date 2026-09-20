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

# RDF-SORT-1 (IPI-43) — `rewrite_data_files` strategies `sort` and `zorder`

The unit ledger. Every fact here is measured against the run-25d Spark sort oracle (Spark 4.1.2 +
Iceberg 1.11.0, hadoop catalog, `local[1]`) or read out of the Iceberg 1.11.0 runtime jar with
`javap`. Nothing here is recalled.

## 1. The oracle cells

Table shape `(id BIGINT, cat STRING, ts TIMESTAMP, v DOUBLE, s STRING)`, four INSERT statements of 25 rows
(nulls in every column, one NaN in `v`, four rows with a NULL `id`), v2 unless the cell says v3.
Every cell carries `rewrite-all` except `SORT-EXPLICIT-DEFAULT-OPTS`.

| cell | call | Spark answer |
|---|---|---|
| SORT-EXPLICIT-DEFAULT-OPTS | `sort`, `id DESC NULLS LAST`, no options | nothing rewritten: 4 files < `min-input-files` 5 |
| SORT-EXPLICIT | `sort`, `id DESC NULLS LAST` | 4 → 1 file, `sort_order_id` 0, one descending run, the 4 NULL ids last |
| SORT-EXPLICIT-MULTI | `sort`, `cat ASC NULLS FIRST, v DESC NULLS LAST` | 4 → 1, stamp 0 |
| SORT-TABLE-ORDER | `sort`, table order `s ASC NULLS LAST, id` | 4 → 1, stamp **1** |
| SORT-NO-ORDER | `sort`, unsorted table | `IllegalArgumentException` "Cannot sort data without a valid sort order, table '<t>' is unsorted and no sort order is provided" |
| SORT-EXPLICIT-OVER-TABLE-ORDER | `sort`, `id ASC NULLS FIRST`, table order `s` | 4 → 1, stamp 0 (the explicit order matches no table order) |
| SORT-EXPLICIT-EQUALS-TABLE-ORDER | `sort`, `id DESC NULLS LAST` = table order | 4 → 1, stamp 1 |
| SORT-EXPLICIT-EQUALS-OLD-TABLE-ORDER | same order, but the table's default has moved on to `s` | 4 → 1, stamp **1** — a historical `sort-orders` entry counts |
| SORT-TABLE-ORDER-TRANSFORM | table order `bucket(4, id), s DESC` | 16 → 4 (partitioned by `cat`), stamp 1 |
| SORT-TABLE-ORDER-PARTITIONED | table order `id DESC`, partitioned by `cat` | 16 → 4, one file per partition, stamp 1 |
| SORT-TRANSFORM-BUCKET / -TRUNC / -DAYS | explicit `bucket(4, id)`, `truncate(s, 2) DESC`, `days(ts)` | 4 → 1, stamp 0 |
| SORT-PARTITIONED | `sort`, `id DESC`, partitioned by `cat` | 16 → 4, sorted inside each partition, stamp 0 |
| SORT-TARGET-SMALL | `sort`, `id`, `target-file-size-bytes` 1500 | 4 → **5** files, `[21, 19, 22, 21, 17]` rows, ONE global ascending order across the five files (disjoint ranges, nulls-first file first) |
| SORT-SHUFFLE-PARTS | `shuffle-partitions-per-file` 2 | 4 → 1, identical rows and order to the plain cell |
| SORT-COMPRESSION-FACTOR | `compression-factor` 2.0 | 4 → 1, identical rows and order |
| SORT-REWRITE-ALL-SINGLE | `rewrite-all` + `min-input-files` 1 | 4 → 1 |
| SORT-WHERE | `where => 'id < 50'` | 4 → 1 — the filter selects FILES, every live row of a selected file is rewritten |
| SORT-BAD-COLUMN | `sort_order => 'nope'` | `ValidationException` "Cannot find field 'nope' in struct: struct<1: id: optional long, …>" |
| SORT-BAD-STRATEGY | `strategy => 'spiral'` | `IllegalArgumentException` "unsupported strategy: spiral. Only binpack or sort is supported" |
| SORT-ORDER-ON-BINPACK | `strategy => 'binpack', sort_order => 'id'` | `IllegalArgumentException` "Cannot set rewrite mode, it has already been set to BIN-PACK" |
| SORT-ZORDER-OPTION-ON-SORT | `sort` + `var-length-contribution` | `IllegalArgumentException` "Cannot use options [var-length-contribution], they are not supported by the action or the rewriter SORT" |
| SORT-V3-LINEAGE | v3, `sort`, `id DESC` | 4 → 1; `_row_id` / `_last_updated_sequence_number` preserved per row |
| SORT-WITH-DELETES | merge-on-read deletes, `sort`, `id` | 4 → 1 with 80 live rows; the 4 delete files stay (`removed_delete_files_count` 0) |
| ZORDER-1 / -2 / -3-TYPES | `zorder(id)`, `zorder(id, v)`, `zorder(s, ts, id)` | 4 → 1, stamp 0, rows in unsigned-lexicographic z-value order (nulls, whose z is all-zero, first) |
| ZORDER-NO-STRATEGY | `sort_order => 'zorder(id, v)'` with no `strategy` | same as ZORDER-2 — the z expression alone selects the rewriter |
| ZORDER-PARTITION-COL | `zorder(cat, id)`, partitioned by `cat` | 16 → 4 — `cat` is an identity partition column and is DROPPED from the z tuple (the order equals `zorder(id)` inside each partition) |
| ZORDER-PARTITIONED | `zorder(id, v)`, partitioned by `cat` | 16 → 4, z order inside each partition |
| ZORDER-MIXED | `zorder(id), v` | `IllegalArgumentException` "Cannot mix identity sort columns and a Zorder sort expression: zorder(id), v" |
| ZORDER-TABLE-ORDER | `zorder(id, v)` on a table ordered by `s` | 4 → 1, stamp **0** — a z rewrite never stamps a table order |
| ZORDER-VAR-LEN | `var-length-contribution` 2 | 4 → 1, a different row order from the default 8 |
| ZORDER-MAX-OUTPUT | `max-output-size` 4 | 4 → 1, order from the 4-byte truncated interleave |
| ZORDER-BAD-COLUMN | `zorder(id, nope)` | `IllegalArgumentException` "Cannot find column 'nope' in table schema (case sensitive = false): struct<…>" |
| ZORDER-TARGET-SMALL | `zorder(id, v)`, target 1500 | 4 → 5 files, one global z order across them |
| ZORDER-V3-LINEAGE | v3 | 4 → 1, lineage preserved |
| ZORDER-NESTED-FN | `zorder(bucket(4, id), v)` | parse error "Unable to parse sortOrder: …" (the procedure's parser; the fork API takes column names) |

### Round 2 cells (this unit added them to the recorder, `record_rdf_sort_zbytes.py`)

| cell | Spark answer |
|---|---|
| ZORDER-BOOL | `false` rows before `true` rows; `true` contributes the byte `0x81`, `false` `0x00` |
| ZORDER-BOOL-NULL | **the job FAILS**: `FAILED_EXECUTE_UDF` on `BOOLEAN-LEXICAL-BYTES` — the boolean UDF unboxes without a null check (every other type returns 8 zero bytes for null) |
| ZORDER-DECIMAL | `IllegalArgumentException` "Cannot use column d of type DecimalType(10,2) in ZOrdering, the type is unsupported" |
| ZORDER-DATE-NTZ-INT | `zorder(d, tn, k, f, bn)` over DATE, TIMESTAMP_NTZ, INT, FLOAT, BINARY succeeds; the all-null row sorts first; 13 rows in a pinned order |
| ZORDER-ALL-IDENTITY | `IllegalArgumentException` "Cannot ZOrder, all columns provided were identity partition columns and cannot be used" |
| ZORDER-VARLEN-ZERO | "Cannot use less than 1 byte for variable length types with ZOrder, 'var-length-contribution' was set to 0" |
| ZORDER-MAXOUT-ZERO | "Cannot have the interleaved ZOrder value use less than 1 byte, 'max-output-size' was set to 0" |
| SORT-SHUFFLE-ZERO | "'shuffle-partitions-per-file' is set to 0 but must be > 0" |
| SORT-CF-ZERO | "'compression-factor' is set to 0.0 but must be > 0" |
| BINPACK-CF / BINPACK-SHUFFLE | "Cannot use options [compression-factor], they are not supported by the action or the rewriter BIN-PACK" (and the same for `shuffle-partitions-per-file`) |
| ZORDER-MULTIBYTE | `var-length-contribution` 2 over `'aé'`, `'éa'`, `'ab'`, `''`, `'€'`, `'aa'`, NULL, `'b'`: the row order follows the truncated UTF-8 bytes, and `''`, `'€'` and NULL all collapse to the two zero bytes |
| ZORDER-CASE | `zorder(ID, s)` fails `FIELD_NOT_FOUND` — validation is case-insensitive, but the Spark column lookup that follows is not |
| ZORDER-NESTED | `zorder(st.x, id)` fails `FIELD_NOT_FOUND` — a nested column passes validation and then fails the Spark lookup |

### Round 3 cells (`record_rdf_sort_nan.py`, added by round 2 of the review)

Table `(id BIGINT, st STRUCT<id: BIGINT, x: BIGINT>, v DOUBLE)`, five INSERT statements of 10 rows, so a
nested `st.id` collides with the top-level `id`.

| cell | call | Spark answer |
|---|---|---|
| ZORDER-NESTED-COLLISION | `zorder(st.id)` | `IllegalArgumentException` ``[FIELD_NOT_FOUND] No such struct field `st`.`id` in `id`, `st`, `v`. SQLSTATE: 42704`` — Spark REFUSES; it does not silently z-order the top-level `id` |
| ZORDER-NESTED-PLAIN | `zorder(st.x)` | the same refusal for ``` `st`.`x` ``` |
| ZORDER-CASE-COLLISION | `zorder(ID)` | the same refusal for ``` `ID` ``` — validation is case-insensitive, the bind is not |
| ZORDER-TOP-LEVEL | `zorder(id)` | 5 → 1 file, the 50 rows in ascending `id`, the nested `st.id` descending beside them |

### Java's own z bytes for every NaN (`java -cp <runtime jar> ZBytes.java`)

`ZOrderByteUtils.doubleToOrderedBytes` / `.floatToOrderedBytes`, measured bit pattern in, bytes out:

| input bits | `Double.doubleToLongBits` | ordered bytes |
|---|---|---|
| `0000000000000000` (+0.0) | `0000000000000000` | `8000000000000000` |
| `8000000000000000` (-0.0) | `8000000000000000` | `7fffffff00000000` |
| `7ff8000000000000` (canonical NaN) | `7ff8000000000000` | `fff80000fff00000` |
| `fff8000000000000` (sign-bit NaN) | `7ff8000000000000` | `fff80000fff00000` |
| `7ff8000000000001` (payload NaN) | `7ff8000000000000` | `fff80000fff00000` |
| `fff800000000abcd` (sign + payload) | `7ff8000000000000` | `fff80000fff00000` |
| `7ff0000000000001` (signalling NaN) | `7ff8000000000000` | `fff80000fff00000` |
| float `ffc0abcd` | widens to raw `fff81579a0000000` | `fff80000fff00000` |

Every NaN — sign bit, payload, signalling, float or double — collapses to one z value, and it is
ABOVE `+Infinity` (`fff00000ffe00000`). `-0.0` and `+0.0` stay distinct and ordered.

## 2. The Java evidence (`javap` on `iceberg-spark-runtime-4.1_2.13-1.11.0.jar`)

`org.apache.iceberg.util.ZOrderByteUtils`:

- `PRIMITIVE_BUFFER_SIZE = 8`. `wholeNumberOrderedBytes(long, buf)` = `putLong(value ^ Long.MIN_VALUE)`,
  8 bytes. `intToOrderedBytes` / `shortToOrderedBytes` / `tinyintToOrderedBytes` all widen (`i2l`)
  into the same 8-byte whole-number encoding, so an INT column contributes 8 bytes, not 4.
- `floatingPointOrderedBytes(double, buf)`: `bits = Double.doubleToLongBits(v)`, then
  `bits ^ ((bits >> 31) | Long.MIN_VALUE)` — an ARITHMETIC shift (`lshr`), not a logical one. For a
  negative value that flips the sign bit and bits 63..31 only; the low 31 bits are NOT flipped.
  This is a quirk of Java's implementation, and the encoding is only weakly order preserving for
  negatives. It is reproduced exactly. `floatToOrderedBytes` widens `f2d` first, so a FLOAT is the
  8-byte encoding of its double value — `f2d` KEEPS the sign and payload of a NaN, and
  `Double.doubleToLongBits` (not `doubleToRawLongBits`) then canonicalises every NaN to
  `0x7ff8000000000000`, so float and double NaNs share one z value. Measured above.
- `stringToOrderedBytes(s, len, buf, encoder)`: zero-fill `len` bytes, then UTF-8 encode with
  `endOfInput = true`. On OVERFLOW the encoder stops at a character boundary, so a multi-byte
  character that does not fit contributes NOTHING and the tail stays zero. A null string is `len`
  zero bytes.
- `byteTruncateOrFill(bytes, len, buf)`: null or short input is zero-padded to `len`, longer input
  is truncated to `len`.
- `interleaveBits(byte[][] columns, int outputSize, buf)`: round-robin over the columns, taking one
  bit per column per step, most significant bit first, and skipping a column once its length is
  exhausted (`columns[i].length <= sourceByte`). The output is `outputSize` bytes.

`org.apache.iceberg.spark.actions.SparkZOrderUDF`:

- Per type (`sortedLexicographically`): BYTE/SHORT/INT/LONG → whole-number 8 bytes; FLOAT/DOUBLE →
  floating-point 8 bytes; STRING → `stringToOrderedBytes(varLengthContribution)`; BINARY →
  `byteTruncateOrFill(varLengthContribution)`; BOOLEAN → an 8-byte buffer with byte 0 set to `-127`
  (`0x81`) for true, `0` for false; **TIMESTAMP (with zone) → `cast(col AS LONG)`**, i.e. whole
  SECONDS, not micros; TIMESTAMP_NTZ → `DateTimeUtil.microsFromTimestamp`, i.e. MICROS; DATE →
  `unix_date` (days) cast to long. Anything else throws "Cannot use column %s of type %s in
  ZOrdering, the type is unsupported".
- Null for every non-boolean type short-circuits to `PRIMITIVE_EMPTY` = 8 zero bytes (strings and
  binary get their own zero fill of `varLengthContribution` bytes). The boolean lambda has no null
  check — this is the measured `ZORDER-BOOL-NULL` failure.
- `increaseOutputSize` folds each column as `total = min(total + size, maxOutputSize)`, so the
  interleaved value is `min(sum of column widths, max-output-size)` bytes.

`SparkShufflingFileRewriteRunner` / `SparkSortFileRewriteRunner` / `SparkZOrderFileRewriteRunner`:

- The sort runner's no-argument constructor asserts `table.sortOrder().isSorted()` with the
  SORT-NO-ORDER message; the explicit constructor asserts the passed order is non-empty with
  "Cannot sort data without a valid sort order, the provided sort order is null or empty".
- The stamp is `SortOrderUtil.findTableSortOrder(table, sortOrder()).orderId()`, written as the
  `output-sort-order-id` write option. `findTableSortOrder` returns the FIRST entry of
  `table.sortOrders()` whose `sameOrder` holds — `Arrays.equals` over `SortField`, and `SortField`
  equality is `(transform.toString(), sourceId, direction, nullOrder)` — else `SortOrder.unsorted()`
  (id 0). Z-order sorts on the synthetic `ICEZVALUE` column, which matches no table order, hence
  stamp 0 always.
- The group's rows are ONE Spark sort: `DistributionAndOrderingUtils.prepareQuery` with an ordered
  write distribution over `max(1, expectedOutputFiles * shuffle-partitions-per-file)` partitions,
  then `OrderAwareCoalesce` back down to `expectedOutputFiles` when the multiplier is > 1. That is
  why SORT-TARGET-SMALL's five files hold five disjoint ascending ranges.
- When the group's spec is not the output spec, `SortOrderUtil.buildSortOrder(schema, spec, order)`
  prepends the output partition fields to the sort order, so the output is clustered by partition
  and sorted inside each partition.
- `SparkShufflingDataRewritePlanner` extends `BinPackRewriteFilePlanner` and overrides only
  `expectedOutputFiles(inputSize)` → `max(1, super.expectedOutputFiles(inputSize *
  compressionFactor))`. Candidate selection, grouping and the group filters are the bin-pack ones,
  unchanged. The fork's planner therefore needs no change for this unit.
- `SparkZOrderFileRewriteRunner.validZOrderColNames` refuses an empty column list, refuses a table
  that already has an `ICEZVALUE` column, resolves each name (case-insensitively by default),
  DROPS every name that is an identity partition source of the table's current spec, and refuses
  the call when nothing is left. The list it returns holds the CALLER'S OWN name (bytecode offset
  216-220 loads the loop variable, not `NestedField.name()`), and `zValue` then binds each name
  with `df.schema().apply(name)` — an exact, top-level, case-sensitive lookup against the
  dataframe. That pair is the whole story behind ZORDER-CASE, ZORDER-NESTED and the round-3 cells:
  a dotted or differently-cased name passes validation and dies at the bind.

## 3. Design decisions

**D-1 — the strategy is one public enum on `RewriteDataFiles`, bin-pack stays the default.**
`RewriteStrategy::{BinPack, SortByTableOrder, Sort(SortOrder), ZOrder(ZOrderSpec)}`. Java reaches
the same four states through `binPack()`, `sort()`, `sort(SortOrder)` and `zOrder(String...)`, and
refuses a second call with "Cannot set rewrite mode, it has already been set to %s"; a Rust builder
field cannot be set twice in a way the caller can observe, so the fork exposes the state, not the
setter sequence. The RePark router maps its `strategy` / `sort_order` arguments onto the enum and
owns Java's "unsupported strategy" and "Cannot set rewrite mode" refusals, because both are
properties of the procedure's argument shape, not of the action.

**D-2 — transforms in an explicit sort order are supported, z-order takes column names only.**
Java's Spark sort path binds an arbitrary `SortOrder` (SORT-TRANSFORM-BUCKET / -TRUNC / -DAYS), and
its z-order path takes `String...` column names and refuses a transform at the parser
(ZORDER-NESTED-FN). The fork mirrors both. The existing `RewriteSortKey` path already evaluates
transforms, so the sort arm reuses it.

**D-3 — the stamp.** `sort_order_stamp` walks `table.metadata().sort_orders_iter()` in ascending
order id and returns the first order whose fields equal the effective order field for field —
source id, direction, null order, and the transform's string form, exactly Java's `SortField`
equality — else 0. Z-order stamps 0 without a lookup. Bin-pack keeps the fork's existing stamp
(the table's default order id when the table is sorted).

**D-4 — BOUNDED MEMORY: a spilling external merge sort, not a collect.**
Spark's sort strategy produces ONE global order per file group, which the fork's bin-pack
`write_sorted_run` does not (it sorts runs of at most the write-max bytes INDEPENDENTLY, so file 2
can hold keys below file 1). Collecting the group to get a global order is exactly the unbounded
collect this unit forbids, so the sort and z-order arms run an external merge sort:

1. Read the group's batches, accumulating a run until it reaches `sort_memory_budget_bytes`
   (default 128 MiB, `RewriteDataFiles::sort_memory_budget_bytes`).
2. Sort the run in memory by the encoded key (below). If the whole group fits in one run, the rows
   go straight to the writer and NO temporary file is ever created — the common case.
3. Otherwise each sorted run is spilled to a temporary parquet file through the table's `FileIO`,
   and the runs are merged k-way into the rolling writer. The merge fan-in is bounded
   (`SORT_MERGE_FAN_IN`, 16); more runs than that are merged in passes, so the merge never holds
   more than 16 decoded batches at once. Spill row groups and the read batch size are sized from
   the budget, so the merge's resident set is a fraction of one budget.
4. Peak resident bytes are therefore ~2 × budget (the concat-and-sort step transiently holds the
   run twice) plus the merge's bounded batches. The action reports `peak_sort_bytes` and
   `spilled_runs` so a test can pin the bound.

**D-5 — where the temporary files live, and when they die.** Under the table's data location, in
`rewrite-sort-spill-<uuid>/`, written and deleted through the table's own `FileIO` — the same
credentials and the same storage scope as the output files, and NOT a local temp dir (the action
runs against object storage where no local disk is assumed). Every spill path is recorded as it is
created and deleted on BOTH exits: the success path deletes them after the merge, and the error
path deletes them before the error propagates. A leaked file (a process kill) lands inside the
table's data location under a visible prefix, so `delete_orphan_files` reclaims it — a `.`-prefixed
or `_`-prefixed name would be invisible to Java's orphan scan and leak forever. A spill file is
never referenced by a manifest and never reaches a commit.

**D-6 — the key encoding.** Both arms compare rows by a byte string built per row, because the merge
compares rows across runs and the arrow lexsort comparator cannot span two arrays. Per sort field:
one null-marker byte (`nulls_first` → null 0x00 / value 0x01; `nulls_last` → null 0x01 / value
0x00), then the value bytes, inverted byte for byte when the direction is descending. Integers are
big-endian sign-flipped; floats use the total order (canonical NaN first, so every NaN compares
equal, and NaN sorts above every number, as in Spark); strings and binary are escaped
(`0x00` → `0x00 0xFF`, terminator `0x00 0x00`) so the byte order is the value order regardless of
length; decimals are sign-flipped big-endian. The z-order arm's key is the interleaved z value
alone, ascending, which is unsigned-lexicographic by construction. A single encoding is used for
both the in-run sort and the merge, so the two can never disagree.

**D-7 — the z encoding is a byte-for-byte port of `ZOrderByteUtils` including its quirks.** The
arithmetic-shift float mask (§2), the CANONICALISATION of every NaN before that mask
(`Double.doubleToLongBits`, not the raw bits — §2's measured table), the widening of every whole
number to 8 bytes, the `min(sum, max-output-size)` output width, the UTF-8 encoder's
character-boundary truncation, the zero fill for nulls, `0x81` for a true boolean. One deliberate divergence: a NULL boolean encodes
as 8 zero bytes, the rule every other type follows, where Java crashes the job on an unboxing NPE
(ZORDER-BOOL-NULL). Java's crash is a missing null check, not a contract, and an action that
destroys a rewrite on a null boolean is not a behaviour worth porting.
Iceberg type → encoding: int/long → 8-byte whole number; float/double → 8-byte floating point;
date → days as a long; time → micros as a long (Spark has no Iceberg-time mapping, so this is a
fork extension, not a parity claim); `timestamptz` → SECONDS (Spark's `cast(ts AS LONG)`);
`timestamp` → micros; string → `var-length-contribution` UTF-8 bytes; uuid → the UTF-8 bytes of its canonical
lowercase text, which is what Spark z-orders (Iceberg `uuid` reads as a Spark STRING);
fixed/binary → `var-length-contribution` bytes; boolean → the `0x81`/`0x00` byte in an 8-byte buffer; decimal and
the v3 nanosecond timestamps are REFUSED with Java's "the type is unsupported" message.

**D-11 — a z-order column binds the caller's own name, against the TOP-LEVEL schema only.**
Java keeps the caller's string in `validZOrderColNames` and binds it with `df.schema().apply(name)`
(§2), so a dotted or differently-cased name is a hard error even when the case-insensitive
validation resolved it. The fork does the same: resolve case-insensitively for the "Cannot find
column" refusal and the identity-partition drop, then require a top-level field of that exact name
and that exact field id, else refuse. Measured: on a table holding BOTH `id` and `st.id`,
`zorder(st.id)` is refused by Spark (ZORDER-NESTED-COLLISION) — it does not silently z-order the
top-level `id`, and neither does the fork. The order of the three steps is Java's: find, drop the
identity partition columns, bind, then type-check, so an identity partition column of an
unsupported type is dropped rather than refused.

**D-8 — the layout-only options.** `shuffle-partitions-per-file` and `compression-factor` are
accepted and validated (both must be > 0, with Java's messages) and are recorded as no-ops for a
single-process rewrite, with the reason per option:

| option | Spark meaning | why it cannot change the fork's output |
|---|---|---|
| `shuffle-partitions-per-file` | multiplies the shuffle partition count, then `OrderAwareCoalesce` folds the extra partitions back into the same files | it buys parallelism inside one file's sort. The fork's rewrite is one process and one ordered stream per group; the row order and the file boundaries are identical with or without it (measured: SORT-SHUFFLE-PARTS is row-for-row identical to SORT-EXPLICIT's shape) |
| `compression-factor` | scales the INPUT size estimate that picks `expectedOutputFiles`, i.e. how many range partitions Spark cuts | the fork rolls output files on the bytes it has actually written (`RollingFileWriter`), which is the quantity the factor is guessing at. A guess cannot improve on the measurement (measured: SORT-COMPRESSION-FACTOR is row-for-row identical) |

Both are refused under bin-pack with Java's exact message ("Cannot use options [compression-factor],
they are not supported by the action or the rewriter BIN-PACK"), because that refusal IS observable
and the RePark router must reproduce it.

**D-9 — the file count is not pinned, the ORDER is.** Spark cuts its output files at sampled range
boundaries (`[21, 19, 22, 21, 17]` for SORT-TARGET-SMALL); the fork rolls at `write-max` bytes. The
two agree on what is load bearing — one global order per group, disjoint ascending key ranges
across the group's output files, every row preserved — and cannot agree on the exact split points
without reimplementing Spark's reservoir sampler. The pins assert the order and the disjointness,
and assert `added_data_files_count > 1` where the oracle split, never the exact count.

**D-10 — ties.** Equal keys may come out in any order in either engine (Spark's sort is not stable
across a shuffle). Every pin compares the KEY sequence, never the tie order, and compares the row
multiset for conservation.

## 4. Divergences (one line each; status belongs to the GAP_MATRIX, not here)

| divergence | what the caller gets |
|---|---|
| null boolean in a z-order tuple | the fork encodes 8 zero bytes; Spark fails the job with an unboxing NPE (D-7) |
| output file split points | the fork rolls on written bytes, Spark cuts on sampled ranges (D-9) |
| `shuffle-partitions-per-file`, `compression-factor` | accepted, validated, no effect (D-8) |
| a z-order over a nested or differently-cased column | the fork refuses it at `resolve_strategy` with ``No such struct field `st`.`id` in `id`, `st`, `v` ``; Spark passes validation and then fails the same bind with the same sentence, wrapped in its `[FIELD_NOT_FOUND] … SQLSTATE: 42704` error class. Both refuse, neither binds a different column (D-11) |
| Iceberg `time` in a z tuple | the fork encodes micros as a whole number; Spark has no mapping for it |

## 5. Red-first evidence and the mutation proof

**Red first (step 2).** With the strategy dispatch neutralised (`KeyPlan::build` returning `None`
for every strategy, i.e. the pre-unit behaviour), 11 of the 17 pins fail and 6 pass. The 6 that
pass are the ones that must not depend on ordering: the four option preconditions, the
unsorted-table refusal, and the bin-pack control. Restoring the dispatch turns all 17 green.

**The mutation table (step 5).** Each clause was broken in turn, the filtered suites re-run, and
the mutation reverted. Every clause has at least one pin that fails when it breaks.

| mutation | what it breaks | pins that go RED |
|---|---|---|
| M1 `sort_order_stamp` never matches a table order (always 0) | the stamp rule | `sort_by_table_order_uses_the_default_order_and_stamps_its_id`, `sort_explicit_order_equal_to_the_table_order_stamps_the_table_order_id`, `sort_explicit_order_equal_to_a_historical_table_order_stamps_that_order_id`, `sort_partitioned_orders_every_partition_group` (4 of 17 sort pins; the z pins stay green, correctly — a z rewrite stamps 0 either way) |
| M2 the in-run sort reverses its key order | the order itself | 10 of 17 sort pins and 6 of 13 z pins, including all three bounded-memory pins |
| M3 the budget never spills (one unbounded run) | the bounded-memory clause | `a_small_sort_budget_spills_runs_and_still_writes_one_global_order`, `a_small_sort_budget_spills_a_zorder_rewrite_too`, `more_runs_than_the_merge_fan_in_merge_in_passes` (3; every ordering pin stays green, which is the point: an unbounded collect produces the RIGHT order and still violates the clause) |
| M6 the k-way merge ignores the keys (emits run after run) | the global order ACROSS spilled runs | `a_small_sort_budget_spills_runs_and_still_writes_one_global_order`, `more_runs_than_the_merge_fan_in_merge_in_passes` (2; the non-spilling pins stay green, so the merge has its own pins) |
| M4 z-order encodes `timestamptz` in micros, not seconds | the z type mapping | `zorder_over_a_string_a_timestamp_and_a_long_matches_the_spark_row_order` |
| M5 the float mask uses a LOGICAL shift (`>>>`) instead of Java's arithmetic `>>` | the float encoding quirk | `floating_point_matches_javas_ordered_bytes_including_its_shift_quirk` |

**Round 2 mutations (the review's findings).** Same method: break the clause, run
`cargo test -p iceberg --lib rewrite_data_files`, revert. The count after each result is the whole
filtered suite, so "1 failed" means the new pin is the only thing standing between the clause and a
silent regression.

| mutation | what it breaks | result |
|---|---|---|
| M7 `total = total.saturating_add(width)` — the `.min(max_output_size)` cap dropped (L-01's neighbour, V-01) | the z output width | `the_max_output_size_cap_bounds_the_interleaved_z_value` RED — `149 passed; 1 failed`. The round-1 cell pin stayed green, which is exactly the hollowness the reviewer found |
| M8 `push_int` writes `value.to_be_bytes()` — no sign flip (V-02) | negatives above positives in every sort key | `whole_number_keys_place_every_negative_below_every_positive` RED — `149 passed; 1 failed` |
| M9 `push_escaped` stops escaping `0x00` (V-03) | a key with an embedded NUL | `variable_length_keys_escape_their_zero_bytes_and_terminate` RED. With the byte assertions removed the ORDER assertion still fails on its own: `'a\0'` sorts before `'a'` — `left: [3, 0, 1, 2] right: [0, 3, 1, 2]` |
| M10 `push_escaped` stops writing its `0x00 0x00` terminator (V-03) | the field boundary | the same pin RED. With the byte assertions removed: a DESCENDING string key puts `'a'` above `'ab'` — `left: [0, 2, 3, 1] right: [2, 1, 3, 0]` |
| M11 the 256-bit decimal key reverses its bytes again (L-03) | the wide-decimal order | `wide_decimal_keys_sort_big_endian_with_a_flipped_sign` RED — the five pinned values come out `[4, 1, 2, 0, 3]` |
| M12 `floating_point_ordered_bytes` reads raw bits (L-01) | NaN canonicalisation | `every_nan_collapses_to_javas_canonical_nan_in_the_z_value`, `a_float_column_canonicalises_its_nans_like_javas_widening_encoder` and `floating_point_matches_javas_ordered_bytes_including_its_shift_quirk` RED — `13 passed; 3 failed` of the z suite |
| M13 `valid_z_order_columns` keeps `field.name` and drops the top-level check (L-02) | the z column bind | `a_nested_z_order_column_is_refused_and_never_binds_its_top_level_namesake` RED at its `expect_err` — the rewrite SUCCEEDS on `zorder(st.id)`, silently z-ordering the top-level `id`, which is the defect itself |

## 6. Gates

| gate | command | result |
|---|---|---|
| unit pins (every `rewrite_data_files` module) | `cargo test -p iceberg --lib rewrite_data_files` | `ok. 152 passed; 0 failed; 0 ignored; 0 measured; 4010 filtered out` |
| the neighbouring delete-rewrite suites | `cargo test -p iceberg --lib rewrite_position_delete` | `ok. 103 passed; 0 failed; 0 ignored; 0 measured; 4059 filtered out` |
| format | `cargo fmt --all -- --check` | clean |
| lints | `cargo clippy -p iceberg --all-targets -- -D warnings` | `Finished dev profile` — no warning |
| file size | `python3 scripts/check_rust_file_size.py` | `583 files clean (92 legacy ceilings)`; the `rewrite_data_files.rs` ceiling moved DOWN 2440 → 2418 |
| prose | `typos crates/iceberg/src/maintenance/ task/rdf-sort-1-ledger.md` | clean |
| comment ban | `comment_ban.py <clone> origin/main` | `comment-ban hits=0` |

## 7. What the RePark side owes

The `CALL rewrite_data_files` router consumes this API:

| procedure argument | fork API |
|---|---|
| `strategy => 'binpack'` (or absent, with no `sort_order`) | leave the default, `RewriteStrategy::BinPack` |
| `strategy => 'sort'`, no `sort_order` | `RewriteStrategy::SortByTableOrder` |
| `strategy => 'sort'`, `sort_order => '<cols>'` | parse the string into a `SortOrder` (direction and null order per column, transforms allowed) → `RewriteStrategy::Sort(order)` |
| `sort_order => 'zorder(a, b)'` with or without `strategy => 'sort'` | `RewriteStrategy::ZOrder(ZOrderSpec::new([...]))` |
| `options => map('var-length-contribution', n)` | `ZOrderSpec::var_length_contribution(n)` |
| `options => map('max-output-size', n)` | `ZOrderSpec::max_output_size(n)` |
| `options => map('shuffle-partitions-per-file', n)` / `('compression-factor', f)` | `RewriteDataFiles::shuffle_partitions_per_file` / `compression_factor` |

The router still owns four refusals, because each is a property of the procedure's argument shape,
not of the action: `unsupported strategy: <x>. Only binpack or sort is supported`; `Cannot set
rewrite mode, it has already been set to BIN-PACK` (a `sort_order` with `strategy => 'binpack'`);
`Cannot mix identity sort columns and a Zorder sort expression: <text>`; and `Unable to parse
sortOrder: <text>` (a transform inside `zorder(...)`). For an unknown option it builds Java's
`Cannot use options [<names>], they are not supported by the action or the rewriter <NAME>` from
`RewriteStrategy::valid_option_names()` and `RewriteStrategy::description()`, which exist for it.
Every other refusal in §1 comes out of `execute` as a typed `Error` with Java's message text.

## 8. Round 2 of the review — every finding and its disposition

Three reviewers ran against round 1. Each finding below is FIXED (with the commit and the pin that
is red without it), or REFUTED / DEFERRED with the reason. Nothing is left implicit.

| finding | disposition |
|---|---|
| **L-01 P1** z-order NaN is not canonicalised | **FIXED.** `floating_point_ordered_bytes` now reads `0x7ff8000000000000` for every NaN before Java's shift mask, which is what `Double.doubleToLongBits` does. Java's own bytes for a sign-bit, payload, signalling, float and double NaN are in §2's table — all `fff80000fff00000`. Pins: `every_nan_collapses_to_javas_canonical_nan_in_the_z_value`, `a_float_column_canonicalises_its_nans_like_javas_widening_encoder`, four new rows in `floating_point_matches_javas_ordered_bytes_including_its_shift_quirk`. The f32 path needs no separate fix: it widens to f64 first, exactly as Java's `f2d` does, and the canonicalisation happens after the widening in both |
| **L-02 P1** a dotted z-order column binds the wrong column | **FIXED, by refusing.** Measured first: on a table holding both `id` and `st.id`, Spark refuses `zorder(st.id)` (ZORDER-NESTED-COLLISION) rather than binding either. The rule and its Java provenance are D-11. Pin: `a_nested_z_order_column_is_refused_and_never_binds_its_top_level_namesake`, which without the fix does not merely report a different message — the rewrite succeeds and z-orders the wrong column |
| **L-03 P3** Decimal256 encoding | **FIXED.** `arrow_buffer::i256::to_be_bytes` is already big-endian, so the extra `reverse()` produced a little-endian key. Removed. The branch stays unreachable from an Iceberg table (decimal precision ≤ 38 maps to `Decimal128`, and `arrow_schema_to_schema` refuses a `Decimal256` column), so the pin drives the encoder directly with a `Decimal256` batch: `wide_decimal_keys_sort_big_endian_with_a_flipped_sign` |
| **V-01 S2** the `max-output-size` pin is hollow | **FIXED.** The cell pin only compared a row order, and truncating a z value rarely changes one. The new pin asserts the encoded WIDTH and that two rows differing only below the cap collapse to one key: `the_max_output_size_cap_bounds_the_interleaved_z_value` (M7) |
| **V-02 S2** the `push_int` sign flip survives mutation | **FIXED.** The oracle table's ids are all ≥ 0, so no cell could see the flip. The new pin encodes `i64::MIN, -1, 0, 1, i64::MAX` and `i32` beside them, in bytes and in order (M8) |
| **V-03 S2** `push_escaped` survives mutation | **FIXED.** Two mutations, two independent assertions: dropping the `0x00` escape puts `'a\0'` before `'a'`, and dropping the terminator puts `'a'` above `'ab'` under a DESCENDING key (M9, M10) |
| **R-01 P1** the merge clones a `RecordBatch` and a key `Vec` per row | **FIXED.** The key is now MOVED out of the reader — each row's key is pushed onto the heap exactly once and consumed exactly once, so the clone had no reader-side use — and the batch is cloned once per (batch, epoch) when its slot opens, i.e. once per merge output block rather than once per row. `take_row` became `current_batch` + `advance_row` |
| **R-02** a `Vec<u8>` per row for the in-memory run | **REFUTED — deliberate (D-6).** One encoding serves both the in-run sort and the k-way merge, and the merge's heap must own its keys across batch reloads and spill boundaries. A flat arena with `(offset, len)` handles cannot back the heap without either pinning every run's arena for the whole merge or copying the key back out per row, which is the allocation it set out to remove. The allocation is one `Vec<u8>` per row of one run, freed when the run is written |
| **R-03** `concat_batches` + `take` copies the whole run | **FIXED.** `sort_current_run` now encodes each batch in place and hands the sorted `(batch, row)` pairs to `interleave_record_batch` over the run's own batches. The concat copy is gone; peak resident bytes are unchanged at ~2× budget (the run plus its sorted image), which is what D-4 claims |
| **R-04** keys re-encoded when a spill is read back | **DEFERRED — deliberate (D-5/D-6), and narrowed.** A spill file holds rows, not keys: writing the key as a column would grow every spill by the key width, and the merge would still have to decode it. The one case where the re-encode bought nothing at all — a merge of a single run — no longer happens (R-05) |
| **R-05** a single spilled run still pays the k-way merge | **FIXED.** `merge_into` forwards a lone run's batches straight to the sink: no key encoding, no heap, no per-row interleave. Pin: `a_single_spilled_run_is_forwarded_whole_and_stays_ordered` |

Bin-pack, the default path, is untouched by all of it: every change is inside the sort key encoder,
the z encoder or the external sorter, none of which a bin-pack rewrite constructs.

## 9. Round 3 of the review — the final verification critic's two pin gaps

The final verification critic re-ran M7–M13 and confirmed each. Two S2 gaps survived: two pins that
stay GREEN under a mutation of the clause they are cited for. Both are closed below; nothing else
changed in round 3.

| finding | disposition |
|---|---|
| **V-01 S2** `a_single_spilled_run_is_forwarded_whole_and_stays_ordered` proves the ORDER, not the forward | **FIXED.** Deleting the single-run forward left the pin green, because the k-way merge emits the same rows in the same order — the forward is a cost clause, and order cannot see it. `SortRunStats` now carries `heap_merged_runs`, incremented by `runs.len()` on the k-way branch of `merge_into` only, i.e. after the forward has returned. The pin asserts `heap_merged_runs == 0`: no key was re-encoded and no heap was built. `more_runs_than_the_merge_fan_in_merge_in_passes` asserts `heap_merged_runs >= spilled_runs`, so the counter cannot be vacuously zero (M14) |

**Round 3 mutations.** Same method: break the clause, run the filtered suites, revert.

| mutation | what it breaks | result |
|---|---|---|
| M14 the `if let [only] = runs { return self.forward_run(only, sink).await; }` guard deleted from `merge_into` (V-01) | the single-run forward — a lone run pays a key re-encode and a heap it does not need | `a_single_spilled_run_is_forwarded_whole_and_stays_ordered` RED — `23 passed; 1 failed`: `a single run must be forwarded whole: no key was re-encoded and no merge heap was built, got 1 run(s) pushed through the heap / left: 1 right: 0`. Before this round the same mutation was GREEN |

**V-02, the second gap.**

| finding | disposition |
|---|---|
| **V-02 S2** `a_nested_z_order_column_is_refused_and_never_binds_its_top_level_namesake` only asserts the refusals | **FIXED.** A mutation that refuses a top-level `id` whenever a nested namesake exists kept every `expect_err` green, because refusing more is still refusing. The pin now carries the positive half on the SAME schema: eight rows across two files whose top-level `id` order is deliberately the reverse of their `st.id` order, rewritten with `zorder(id)`, which must succeed and emit `[(NULL, 6), (10, 7), (20, 8), (30, 5), (40, 4), (50, 3), (60, 2), (70, 1)]` — the top-level column's z order, with the nested namesake carried along out of its own order (M15) |

| mutation | what it breaks | result |
|---|---|---|
| M15 `valid_z_order_columns` requires `namesakes < 2`, refusing a top-level column whenever a nested field shares its name (V-02) | the top-level bind, while keeping every nested refusal intact | `a_nested_z_order_column_is_refused_and_never_binds_its_top_level_namesake` RED — `16 passed; 1 failed`: `zorder(id) must bind the top-level column and rewrite the table: DataInvalid => No such struct field `id` in `id`, `st`, `v``. The three `expect_err` cases stayed green under the mutation, which is why the round-2 pin did not catch it |
