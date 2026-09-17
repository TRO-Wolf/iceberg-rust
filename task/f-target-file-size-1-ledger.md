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

# F-TARGET-FILE-SIZE-1 — ledger

Model: muse-spark-1.3-contributor. Branch `fix/ice-target-file-size-1`, from fork main `edc38c6a`.

RePark rating 2026-09-16 row V2-15: table `write.target-file-size-bytes = 262144`, 200,000-row
insert → RePark 4 files max 573,667 B (2.19x target); Spark 20 files max 57,178 B.
V3-COV-7: Spark stamps `write.parquet.compression-codec = zstd` at CREATE/CTAS; RePark does not.

Oracle: `/tmp/oc-worker/ic-build/write_fidelity_spark.json` (PySpark 4.1.2 + Iceberg 1.11.0,
local[4]). Runtime jar `/tmp/ic-build/.ivy2/jars/org.apache.iceberg_iceberg-spark-runtime-4.1_2.13-1.11.0.jar`.
Bytecode via `/usr/lib/jvm/zulu-17-amd64/bin/javap -p -c`. Live Spark re-probes via
`/tmp/sparkenv` (PySpark 4.1.2, `JAVA_HOME=zulu-17`).

## Clauses

- C-001: rolling writer never lets a file grow past target by more than one 1000-row slice.
  Split incoming batches into row slices; roll check runs every 1000 rows, as Java.
- C-002: size accounting matches Java `length()` (flushed bytes + uncompressed in-progress
  buffer). Measure arrow-rs 58.4 `ArrowWriter::in_progress_size` meaning; adapt.
- C-003: oracle shape (200,000 rows `(id BIGINT, s STRING)`, target 262,144, 4x50,000-row
  streams, zstd) → file count and max size within ±25% of Spark (20 files, max 57,178 B).
  Bounds: count in [15, 25], max in [42,884, 71,473]. Record exact numbers.
- C-004: 1 MiB and default targets keep one file per 50,000-row stream.
- C-005: new tables carry `write.parquet.compression-codec = zstd` when creator did not set a
  codec, as Java 1.11 for v2 and v3, including REPLACE TABLE.
- C-006: every `RollingFileWriter` user (data, position deletes, equality deletes, rewrite)
  keeps working.

## J-001: Java verification (jar bytecode + live probes)

- J-001a ROWS_DIVISOR = 1000. `RollingFileWriter.shouldRollToNewFile` does
  `currentFileRows % 1000 == 0` via `lrem` against `ldc2_w long 1000`. File rows count per
  RECORD (`write(T)` increments then checks). CORRECTION to brief: 1.11.0 has NO
  `currentFileHasData` conjunct; the empty-file guard is implicit (`length()` is 0 with no
  records, see J-001b).
- J-001b `ParquetWriter.length()` = `writer.getPos()` + `(!closed && recordCount > 0 ?
  writeStore.getBufferedSize() : 0)`. CORRECTION to brief: NO `isColumnFlushNeeded` branch in
  1.11.0; the brief formula matches a different version.
- J-001c `getBufferedSize` = sum over columns of `ColumnWriterBase.getTotalBufferedSize`
  (uncompressed encoded-buffered bytes). Structural match for fork
  `bytes_written() + in_progress_size()`.
- J-001d `TableProperties.PARQUET_COMPRESSION = "write.parquet.compression-codec"`,
  `PARQUET_COMPRESSION_DEFAULT = "gzip"`, `PARQUET_COMPRESSION_DEFAULT_SINCE_1_4_0 = "zstd"`.
  `persistedProperties` puts `zstd` first, then every non-RESERVED caller entry (caller wins).
  Called by 5-arg `newTableMetadata` (create) and `buildReplacement` (replace). No format-version
  gate (applies to v1/v2/v3 alike). RESERVED set (9) matches fork `RESERVED_PROPERTIES`.
  Spark write-time codec read default is `gzip`, but created tables carry the stamped `zstd`
  property so they write zstd (oracle `tblproperties` confirms on property-less creates).
- J-001e Dictionary: iceberg-core default is ON (`Context` reads `parquet.enable.dictionary`
  default true; only `Context` references the key in the whole non-shaded jar). OBSERVED
  Spark behavior is PLAIN: 57/60 oracle files PLAIN-only (+2 PLAIN+RLE, +1 PLAIN_DICTIONARY),
  fresh probes `e1`/`e2`/native-Spark all PLAIN, and e2 (explicit table property `true`) is
  byte-identical to e1. Mechanism NOT located in Iceberg bytecode (every default reads TRUE).
  Treated as Spark-engine behavior; fork DataFusion engine mirrors it (D-001).
- J-001f Live quantum curve, single 50k stream of oracle-shaped rows (ids 0-50k):
  262144 → 11k rows/file; 300000 → 12-13k; 400000 → 16-17k; 500000 → 36k then 14k tail;
  600000/1048576 → no roll (1x50k). Linear-uncompressed explains targets <= ~400k; the 500k+
  regime diverges (page/row-group check effects suspected, NOT resolved, out of brief scope).
  Oracle census: tfs1 = 16x11,000 + 4x6,000; tfs4 = 18x11,000 + 1x2,000; tfs2 = 4x50,000.
- J-001g Fresh single-task 11,000-row probe reproduces oracle max EXACTLY: 57,178 B, one row
  group, PLAIN+zstd, rg uncompressed 262,650 (id 88,029 + s 174,621).

## D-001: design decisions

- D-001 dictionary: fork `ParquetWriterBuilder` default stays dictionary-ON (iceberg-core
  parity, J-001e). The DataFusion engine write path (`IcebergWriteExec`, the Spark analogue)
  writes dictionary-OFF, honoring table property `parquet.enable.dictionary` (Java key, no Java
  constant exists) with engine default false. COW `StreamingDataFileWriter` and rewrite paths
  unchanged (residue, row R174).
- D-002 roll check placement: write-slice-then-maybe-roll (Java order), boundary-crossing test
  `file_rows / 1000 != before / 1000` (exactly equivalent to Java per-record `% 1000 == 0`,
  including partial-slice misalignment). Comparison `>=` (Java; fork was `>`). Post-write
  placement is the empty-file guard (J-001a), no extra flag.
- D-003 C-002 needs no formula change: `bytes_written + in_progress_size` already matches
  J-001b/c structurally. Measured arrow-rs 58.4: `in_progress_size` = anticipated ENCODED size
  (sum of per-column `get_estimated_total_bytes`); with PLAIN columns it equals uncompressed
  buffered to <0.1% (fork 262,592 vs Spark rg 262,650 at 11k rows, J-001g). With dictionary it
  overestimates ~5% (276,093 at 10k). No `memory_size` (4x too big: 1.1 MB at 10k).
- D-004 C-005 stamp points: `TableMetadataBuilder::from_table_creation` (every catalog create:
  memory/sql/glue/hms/s3tables + staged create) and `StagedTableTransaction::begin_replace`
  (Java `buildReplacement`). Insert-when-absent; caller value wins. `set_properties`
  (ALTER path) untouched. Pre-existing reserved-key error-vs-Java-filter difference untouched.
- D-005 zstd level stays 3 (`parquet_compression_from_properties`, F-REWRITE-SIZE-1). Measured
  zstd-1 < zstd-3 on this shape (70,727 vs 95,850 at 10k dict-ON) is a page-flush artifact, not
  a lever; level 1 would break Java parity (hadoop zstd default 3).

## Red-first tests

Unit (beside `rolling_writer.rs`):
- U-001 single 50,000-row batch, small target: base writes 1 file (RED: no mid-batch roll).
- U-002 roll fires on 1000-row boundary crossing with partial slices, `>=` at exact target.
- U-003 file row counts on oracle-shaped data at 262,144 stay within one 1000-slice of 11k.
- U-004 C-005 create stamps zstd; explicit codec wins; replace stamps zstd.

Integration (`crates/integrations/datafusion/tests/` oracle shape, MemTable 4x50k, zstd):
- I-001 target 262,144: base = 4 files (RED vs [15, 25]) and max ~573 kB (RED vs <= 71,473).
- I-002 target 1 MiB + default: 4 files (must STAY green after fix).

## Base-tree reds (pasted)

Unit (`cargo test -p iceberg --lib writer::file_writer::rolling_writer`, base):
`test_single_batch_rolls_mid_batch` panicked `left: [2500], right: [1000, 1000, 500]`;
`test_roll_fires_on_boundary_crossing_across_writes` and
`test_oracle_shape_rolls_every_ten_thousand_rows` (`left: [50000]`) likewise FAILED.
3 failed, 3 pre-existing passed.

Integration (`cargo test -p iceberg-datafusion --test target_file_size`, base): all 3 FAILED
at the codec assertion (`left: None, right: Some("zstd")`), which doubles as the C-005 red.
Count/size reds follow from the RePark base measurement (4 files, max 573,667 B) and were
re-confirmed post-C-005 by mutation 3 below (dictionary forced ON: 262144 and 1MB legs red).

Stamp (`cargo test -p iceberg --test table_create_defaults`, base):
`create_table_stamps_zstd_codec_by_default` and `replace_table_stamps_zstd_codec_by_default`
FAILED (`left: None, right: Some("zstd")`); `create_table_keeps_explicit_codec` passed.

## Implementation

- `RollingFileWriter` (`writer/file_writer/rolling_writer.rs`): `current_file_rows` field plus
  `ROWS_DIVISOR = 1000`; `write` slices the batch into boundary-aligned 1000-row pieces
  (`ROWS_DIVISOR - rows % ROWS_DIVISOR`, capped at remainder), writes slice-first, rolls on
  `rows.is_multiple_of(ROWS_DIVISOR) && size >= target`, resetting the counter. Lazy reopen
  on next slice (Java opens eagerly; observable file boundaries identical, no empty-file
  orphan). Empty batches skip storage entirely (old code created then deleted an empty file).
- `should_roll`: `>` to `>=` (Java `lcmp; iflt` fires unless strictly less).
- datafusion `write.rs`: `.set_dictionary_enabled(false)` unless table property
  `parquet.enable.dictionary` reads true case-insensitively (Java key, no Java constant).
  Library `ParquetWriterBuilder` default stays dictionary-ON (iceberg-core parity).
- `TableProperties::persisted_properties` (new, `spec/table_properties.rs`): insert-when-absent
  codec default. Called net-zero from `from_table_creation` and `begin_replace`.
  `set_properties` (ALTER path) untouched.
- Pre-existing test updates under the new contract: `test_rolling_writer_with_rolling`
  (10x100 rows can no longer roll per-batch; now 25x100 asserting files [1000, 1000, 500]),
  `test_table_metadata_builder_from_table_creation` (`properties.len()` 0 to 1),
  memory `assert_table_eq` (`properties()` empty to `len() == 1`). Value-pinning lives in
  `table_create_defaults.rs`.

## C-003 exact numbers (fork, post-fix)

20 files: 23666, 24484, 25189, 27071, 58345, 58379, 58431, 58573, 58603, 58624, 58673, 58694,
58700, 58752, 58753, 58787, 58930, 58978, 59289, 68446. Total 200,000 rows.
Spark oracle: 20 files, 23735 to 57178. Count 20/20 exact; max 68,446 (+19.7%, bound 71,473).
Per-stream layout mirrors Spark (4 tails near 24-27 kB both sides). Fork full-files run
~15-30% bigger than Spark (arrow-rs vs parquet-mr page framing at identical PLAIN+zstd-3
encodings); the gap is library-level, out of scope.

## Mutation arithmetic (test-adequacy, one knob at a time)

Baseline populations this session: rolling lib filter 6 passed; datafusion file 3 passed;
stamp file 3 passed.

1. `rolling_writer.rs:169` `>=` back to `>`: 0 red out of 6. UNKILLABLE, honestly recorded:
   no test lands exactly on target (a pinned exact-size target would be true-by-construction).
   Kept because 1.11.0 bytecode mandates `>=` (`lcmp; iflt` rolls unless strictly less).
2. Slicing removed (whole batch per write): 3 red out of 6 (the 3 new tests).
3. `write.rs` dictionary forced ON: 2 red out of 3 (262144 count+size leg, 1MB count leg).
   Proves the dictionary arm carries both C-003 and C-004.
4. `persisted_properties` stamp removed: 2 red out of 3 (create + replace stamp tests).
All mutations restored; focused suites re-run green after each restore.

## Rewrite-pin interaction (found in gate run)

`rewrite_size_pin::rewrite_keeps_dictionary_on_low_cardinality_columns` (F-REWRITE-SIZE-1
contract: rewrite preserves input dictionaries) went red: its bed is built by engine INSERT,
which now writes PLAIN, so rewrite correctly inherited PLAIN and the dict-page assertions
fired (0/1). Rewrite logic untouched (inheritance design stands). Fix restores the test's
precondition, not its assertions: `create_low_cardinality_fixture` now opts the bed table
into `parquet.enable.dictionary=true` (new `enable_dictionary` param on
`create_fixture_inner`; other fixtures pass false), which also exercises the dictionary
opt-in knob end-to-end. Assertions identical. Pin re-run green (64 s).

## Round 2 (2026-09-17): RPD tail-readmission pin

CI failure: `test_two_bin_tails_over_target_are_readmitted`
(`rewrite_position_delete_files_tests.rs:3048`): tails summed 60022, target 71114.
Reproduced locally on the rebased head before any round-2 edit.

What the pin pins: Java's below-floor readmission rule for
`RewritePositionDeleteFiles` — a group of two sub-min tails with count below the
`min_input_files` floor is admitted if and only if `enough_content` fires, i.e. the
tails sum strictly over `target_file_size_bytes` and at most `max_file_size_bytes`.
The test engineers exactly that window: knobs as fractions of measured input size `c`
(min 55%, target 60%, max 75%, group 105%), two bins of one delete file each (16k rows
after re-derivation), run 1 splitting each bin in two, run 2 readmitting the two tails
in a single output.

How it built sizes: inputs via `write_position_delete_file` at the default 512 MB
rolling target (no roll under either cadence, so `c` is unchanged by round 1).
Rewrite output rolls at `write_max = target + 0.5 * (max - target)` = 0.675c, fed in
chunks of at most 256 pairs per `write` call.

Why slicing changed them: the roll quantum moved from 256-row-chunk-aligned (old
per-`write` check) to 1000-row-aligned (new). At 12k rows the first 1000-boundary at
or past 0.675c is 9k, leaving 3k-row tails = 0.25c per bin, sum 0.506c, under the
0.60c target. Old quantum left ~0.30-0.375c tails per bin, sum inside (0.60c, 0.75c].

Writer verdict: the new behaviour is Java-correct, no writer fix. Verified against
the 1.11.0 jar: `ClusteredPositionDeleteWriter` extends `ClusteredWriter` and
`FanoutPositionOnlyDeleteWriter` extends `FanoutWriter`, the same per-record writers
that feed `RollingFileWriter`'s 1000-row cadence. The pin's numbers were calibrated
to the old per-batch cadence, so the fixture is re-derived, not the writer.

Re-derivation: input count 12_000 to 16_000 (one line; knob fractions and all
assertions untouched). Per-bin split becomes [11k, 5k]. Measured: c=157534, outputs
[108788, 49640] per bin, tails 49640+49640=99280 vs target 94520 (+5.0%) and max
118150; first files 108788 inside [min 86643, max]. Deterministic zstd output makes
the 5% margin stable across platforms.

Load-bearing: red pre-fix (60022 <= 71114 on the new code), green post-fix,
assertions byte-identical. No new test needed; the existing pin covers the claim.

Round-2 review (rust-code-quality pass over the branch diff): verdict PASS, no
findings. Scans clean (no new escape hatches, casts only in bounded test domains,
no production panics, no stringly errors, no output macros, no atomics, no
bytecode in comments). Format stability holds (fixture row count is test-only, no
encoding change). Parity axis holds (delete-writer cadence verified against the
1.11.0 jar class ancestry; COW dictionary divergence stays a named residue in
row R174).

## Maps

`crates/iceberg/src/writer/map.md` row for `rolling_writer.rs` ("size-based file rolling")
stays accurate; no new files or routing changes. No `map.md` exists in the touched
`spec/`, `transaction/`, `datafusion/physical_plan/`, or `tests/` directories. No map edit.

## Verification log

Round 1:
- `cargo test -p iceberg --lib writer`: 165 passed, 0 failed, 1 ignored.
- `cargo test -p iceberg --lib spec`: 959 passed, 0 failed.
- `cargo test -p iceberg --lib catalog`: 190 passed, 0 failed.
- `cargo test -p iceberg-datafusion`: exit 0, all 30 targets green, including
  `target_file_size` (3) and `rewrite_size_pin` (4, with the restored dict bed).
- `cargo clippy -p iceberg -p iceberg-datafusion --all-targets -- -D warnings`: clean
  (one finding fixed: `is_multiple_of`).
- `make check`: exit 0.

Round 2 (rebased onto fork main 5a0666b9):
- `cargo test -p iceberg --lib`: 3703 passed, 0 failed, 8 ignored.
- `cargo test -p iceberg-datafusion`: exit 0, zero non-ok targets (includes #287
  sorted-insert suites and the re-derived RPD pin).
- `cargo clippy -p iceberg -p iceberg-datafusion --all-targets -- -D warnings`: clean.
- `make check`: exit 0.
- No `scripts/check_rust_file_size.py` change: at-ceiling files
  (`table_metadata_builder.rs`, `staged_table.rs`, `table_metadata.rs`,
  `catalog/memory/catalog.rs`) are net-zero by construction; grown files
  (`rolling_writer.rs`, `table_properties.rs`, datafusion `write.rs`, new test files) sit
  under the 1000 default.
- Comment grep (RULE 0) run before commit: clean.

## Round 3 (2026-09-17, base 96fc9f1f): C-005 stamp across every catalog crate

CI is red in `iceberg-catalog-sql`: `test_create_table_in_nested_namespace_falls_back_to_*`
expects `{}` but the stamped default now reports
`{"write.parquet.compression-codec": "zstd"}`. The stamp is correct Java parity
(see J-003); every catalog crate's tests that compare a fresh table's properties need
the same expectation update. Plan: Java bytecode for all four catalogs plus the REST
client/server split; lib + non-Docker integration tests for every crate under
`crates/catalog/` and `crates/integrations/`; exact-map assertions only, never
"contains"; per-crate commits; workspace gates.

## J-003: stamp applies server-side in every Java catalog (1.11.0 jar bytecode)

- JdbcCatalog, GlueCatalog, HiveCatalog: all extend `BaseMetastoreCatalog`
  (Jdbc/Hive via `BaseMetastoreViewCatalog`), and their create paths run through
  `BaseMetastoreCatalogTableBuilder`, which calls 5-arg
  `TableMetadata.newTableMetadata` at three sites (create plus replace paths).
  The 5-arg wrapper applies `persistedProperties`, so the zstd stamp lands on
  every create. Their `*TableOperations` all extend `BaseMetastoreTableOperations`.
- REST: the client sends user properties verbatim. `RESTSessionCatalog$Builder`
  builds `CreateTableRequest` with `setProperties(userMap)` (no codec injection;
  only `table-default.` / `table-override.` catalog prefixes appear as constants).
  The stamp lands server-side: `CatalogHandlers.createTable` rebuilds through the
  backend `buildTable(...).withProperties(...).create()`, which reaches the same
  `newTableMetadata`.
- Consequence for the fork: stamping in `TableMetadataBuilder::from_table_creation`
  (the choke point behind every catalog `create_table`, including the fork REST
  crate's fixture-backed paths) matches Java on all four catalogs. No client-side
  stamping exists or is needed.

## Round 3 test record

Lib results (`--lib`, all green): sql 81 (after fix), rest 105, glue 50, hms 48,
s3tables 39, loader 8, cache-moka 10, playground 3, iceberg 3703 (round 2; rerun
in workspace gate below). Only sql failed: 7 tests through shared `assert_table_eq`
at `catalog.rs:1597`, all `left: {"write.parquet.compression-codec": "zstd"}` vs
`right: {}`. Local red-to-green on the edited line (81 pass after).

Assertion shape (trilemma, recorded honestly): the file sits exactly at its
3947-line ceiling, and a literal full-map `assert_eq!` needs 4+ lines under
`fn_call_width = 60` (verified: rustfmt splits the 94-char single line). The
brief demands the literal default and unmovable ceilings both, so the helper
asserts exact count (`properties().len() == 1`, net-zero, no `contains`).
Pin split, stated exactly: `table_create_defaults.rs` pins insertion and
caller-wins through the production constants (value-unobservable by
construction); the literal `"zstd"` pins live in datafusion
`target_file_size.rs:162` (mutation-proven, see M5 below) and in two REST
full-map `assert_eq!` literals (`catalog.rs` near 3901 and 4103, production-
routed reads of `table.metadata().properties()`); exact count in the SQL
helper plus exact value in those literals covers the full map across the
suite. No `contains` weakening anywhere.

Sweep (`properties().is_empty()` / `HashMap::new()` / `.is_empty()` over
`crates/catalog` + `crates/integrations`): remaining hits are AWS SDK config
branches (`glue/src/utils.rs:66`, `s3tables/src/utils.rs:42`) and namespace-row
writes (`sql/src/catalog.rs:603`), none table-property related. Docker-test files
assert only namespace maps, subset relations (`assert_map_contains`), or single
keys after explicit updates; no created-table full-map comparison exists in any
of them, so no blind expectation change was needed.

## M5: default-value flip (round 3 adequacy close-out)

One knob: `PROPERTY_PARQUET_COMPRESSION_CODEC_DEFAULT` `"zstd"` to `"snappy"`
(`table_properties.rs:235`), applied in-tree and restored by `git checkout`
immediately after the run. Baselines this session: stamp file 3 passed, df
target file 3 passed, sql lib 81 passed.

1. `cargo test -p iceberg --test table_create_defaults`: 0 red out of 3.
   Expected: both stamp tests compare through the production constant, so they
   track insertion, not value. Insertion-sensitivity was proven by M4 (stamp
   removed: 2 red out of 3); value-blindness is by construction, recorded here.
2. `cargo test -p iceberg-datafusion --test target_file_size`: 3 red out of 3,
   all at `target_file_size.rs:162` (`left: Some("snappy")`, `right:
   Some("zstd")`). The literal value pin lives here and only here on the
   engine path.
3. `cargo test -p iceberg-catalog-sql --lib`: 0 red out of 81. Expected: the
   helper asserts count, not value. Its load-bearing content is
   insertion-sensitivity, proven by this round's own red-to-green (7 tests
   failed on `{}` before the fix, 81 pass after). A two-property map would
   fail the same assertion by arithmetic.
4. REST full-map literals (`catalog.rs` near 3901 and 4103): not run under the
   flip; verified by inspection (hand-written literal maps, one entry
   `"write.parquet.compression-codec"` to `"zstd"`, compared with `assert_eq!`
   against production reads). Not mutation-proven; stated as inspection, not
   arithmetic.

Tree verified restored after the run (`git status` shows the ledger only).

Docker-only, not runnable here (reviewed, unaffected): `rest/tests/rest_catalog_test.rs`,
`glue/tests/glue_catalog_test.rs`, `hms/tests/hms_catalog_test.rs` (all require
`make docker-up` per their headers). `s3tables/tests/register_table.rs` has no
table-property assertions.
