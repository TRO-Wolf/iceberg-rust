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

# Ledger — F-ORC-AVRO-WRITE-5: D-4 sites 3+4 through `for_format` + R118/Roadmap flips (PR-5)

**Ledger id:** `F-ORC-AVRO-WRITE-5-2026-09-22`
**Branch:** `fix/f-orc-avro-write-5`
**Scope:** IPI-41 PR-5, whole slice: D-4 site 3 (v2 position deletes) + D-4 site 4
(compaction) + the GAP_MATRIX row R118 WRITE-residue flip + the Roadmap LIVE-line flips
**Model:** muse-spark-1.3-contributor
**Lanes:** WO-PR5a (site 3) → WO-PR5b (site 4 + docs, this lane)

## D-0. Packet context

D-4 routes the remaining write doors through the D-3 `AnyFileWriterBuilder::for_format`
seam (`crates/iceberg/src/writer/file_writer/any_writer.rs`). Sites 1+2 (INSERT via
`physical_plan/write.rs`, CoW rewrite via `physical_plan/row_lineage.rs`) landed on main
(`311b9fa41`) with the typed idiom `DataFileFormat::from_str(&table_props.write_format_default)`
+ `for_format`. PR-5 closes the last two doors: site 3 resolves the delete format,
site 4 resolves the table data format, and the docs record WRITE as landed with the
Direction-2 interop proof as the remaining residue. Row R118 stays 🟡, never ✅:
a ✅ needs the Direction-2 Java-reads-Rust interop proof (AGENTS.md Parity mandate).

## D-1. Site 3 — v2 position deletes resolve the delete format (WO-PR5a)

`crates/integrations/datafusion/src/physical_plan/delete_position_deletes.rs`
(`b6cec3133`): `write_position_deletes_for_partition` resolves
`write.delete.format.default` with fallback to `write_format_default`, passes the
resolved format to the `pos-del` filename generator, and matches: the Parquet arm keeps
today's exact construction (`position_delete_writer_properties_for` +
`MetricsConfig::for_position_delete_table`, wrapped as
`AnyFileWriterBuilder::Parquet`); every other arm goes through `for_format` with
`FieldMatchMode::Name`. Pins (`b09d8fc9b`): `crates/integrations/datafusion/tests/mor_delete_format.rs`
+ `tests/map.md` row. Touch-up (`6486fdcb5`): `physical_plan/map.md` delete-format line.

## D-2. Site 4 — compaction writes the table data format (WO-PR5b)

`crates/iceberg/src/maintenance/rewrite_data_files_write.rs` (726/1000 lines, `4517d001d`).
The four lane rulings, all measured from the packet + the sites 1+2+3 diffs:

- **R5 — output format.** Typed `write_format_default` via `DataFileFormat::from_str`,
  the exact sites 1+2 idiom. A garbage value propagates the typed
  `ErrorKind::DataInvalid` `Unsupported data file format: {s}` (pinned by bare
  `Error::message`, never `to_string()`). `puffin` parses and then refuses inside
  `for_format` (`Cannot build a data-file writer for format puffin: a sidecar is never
  a data file`), likewise pinned.
- **R6 — Parquet arm keeps today's exact construction, wrapped.** Compression from
  properties + `dictionary_fallback_columns` WriterProperties +
  `MetricsConfig::for_table`, then wrapped as `AnyFileWriterBuilder::Parquet`. The
  fallback properties apply to the Parquet arm only; ORC/Avro take `for_format`
  defaults. `FieldMatchMode::Id`: today's `ParquetWriterBuilder::new` is Id, and the
  mode only affects `for_format`'s Parquet arm, which site 4 never reaches.
- **R7 — re-key to the seam.** The `RewriteWriterBuilder` alias and the
  `write_sorted_run` helper signature move from `ParquetWriterBuilder` to
  `AnyFileWriterBuilder`; the filename generator takes the resolved format on every
  arm. Avro needs no special code (its arm already lives in `for_format`).
  `dictionary_fallback_columns` stays unchanged, and the non-Parquet arm passes an
  empty footer map (the reader fetches footers itself).
- **R8 — docs.** Row R118 stays 🟡 with the residue narrowed (D-1). Roadmap LIVE
  lines only: the `:225` Missing-list item is removed per the `No longer missing —
  flipped 2026-09-22` convention, and the `:222`/`:460` WRITE-half clauses now say
  the WRITE half landed via PR-5. Row R119, `task/` archives, `docs/parity/archive/`,
  and the Roadmap dated entries are untouched.

## D-3. Site-4 pins and their mutation arithmetic

`crates/iceberg/src/maintenance/rewrite_data_files_format_tests.rs` (976 lines,
`2af82cbdc` + gap pins `b85381ea3`, fixture-sharing shrink `4e0c55dab`): 13 tests,
each run green 5x. ORC end to end (`test_compaction_keeps_table_format`, the packet
§6 name), Avro end to end, two Parquet-construction pins (dictionary stays on for
the constant + low-cardinality columns, which `for_format`'s Parquet arm would
switch off; the dictionary fallback still switches the unique column off), three
spill pins (spill bytes stay parquet `PAR1` and vanish on success; they vanish when
the sink fails; the sorted arm on an ORC table writes ORC and cleans its spills),
garbage + puffin typed refusals, V3 ORC lineage carry, and three HOLLOW-PIN sweep
gap pins: the legacy run-sort arm on ORC (stamps the table order id), rollover
(every rolled file stays ORC), and parquet inputs under an ORC default. One-knob
mutations, each on the current tree (population 13):

| Knob removed | Red / 13 | Tests that went red |
|---|---|---|
| Format resolution hardcoded to Parquet (packet :464) | 9 | ORC, Avro, garbage, puffin, sort-ORC, V3-ORC, legacy, rollover, mixed |
| Filename left Parquet while the builder resolves ORC (mutation-6) | 7 | ORC, Avro, V3-ORC, sort-ORC, legacy, rollover, mixed |
| Parquet arm routed via `for_format` (dict default off) | 1 | dictionary-on |
| Fallback loop dropped | 1 | dictionary-off |
| Spill cleanup disabled (transient, restored) | 3 | both spill tests, sort-ORC |
| Lineage carry dropped from the write schema (transient, restored) | 1 | V3-ORC |
| Sort plan forced unsorted (transient, restored) | 1 | legacy (the stamp observes the arm) |

Every test has at least one killing mutation. Two fixture notes: the legacy-arm
fixture first showed 0 rewritten files — a 3-file group sits under the default
`min_input_files` (5) and never qualifies, so all small-file fixtures use 6 files;
and the gap pins pushed the file to 1102 lines, past the 1000-line ceiling, which
the fixture-sharing shrink (`4e0c55dab`, helpers only, same 13 pins) brought back
to 976 — the A/B kills were re-run after the shrink (9 and 7 red) to prove the
shared assertion helper preserved killability.

The spill pins guard pre-existing `rewrite_data_files_sort_run.rs` behavior (untouched):
site 4 must not move the spill path, and the sorted arm must route through the same
resolved builder. The `PAR1` pin is a guarded characterization (`!is_empty` + length
first, so the magic assertions always execute).

## D-4. Docs flip (STALE-PROSE sweep, same class as R118)

Row R118: 🟡 held, residue narrowed — WRITE lands via PR-5 (ORC and Avro on all 4 D-4
doors call `for_format`; Parquet deletes + compaction keep the pre-existing
`ParquetWriterBuilder` wrapped as `AnyFileWriterBuilder::Parquet`); remaining residue
is the Direction-2 Java-reads-Rust interop
proof + footer-codec + nested/V3. Roadmap: the Missing-list ORC/Avro write item is
gone with a dated flip note, and both WRITE-half clauses say the half landed.
This ledger ships in the same docs commit as the two flipped doc files.
