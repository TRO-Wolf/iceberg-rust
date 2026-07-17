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

# Unit brief — BUG-002: wire `schema.name-mapping.default` into the scan path

**Charter (2026-07-17, user-signed).** The production scan NEVER applies a table's name
mapping: `scan/context.rs` (`ManifestEntryContext::into_file_scan_task`) hardcodes
`name_mapping: None` under a TODO, while the ENTIRE downstream already works —
`FileScanTask.name_mapping` exists (`scan/task.rs`) and the ArrowReader consumes it
(`apply_name_mapping_to_arrow_schema`, the Java `ParquetSchemaUtil.applyNameMapping`
port, invoked in `arrow/reader.rs`). Consequence: a table whose data files lack Parquet
field IDs (the `add_files` migration pattern) with `schema.name-mapping.default` set is
read by Java via the mapping but by us via the position-based fallback — the
wrong-data / wrong-column class. Fix = thread the property from table metadata to every
planned task; prove it end-to-end and cross-engine.

## Proposition ledger

- **C-1 (property constant):** `TableProperties::PROPERTY_DEFAULT_NAME_MAPPING =
  "schema.name-mapping.default"` added in `spec/table_properties.rs`, matching the
  existing `PROPERTY_*` naming/banner style and Java `TableProperties.DEFAULT_NAME_MAPPING`.
- **C-2 (plan wiring):** during scan planning the property is parsed ONCE per plan
  (`serde_json` → `NameMapping`, shared as `Arc<NameMapping>`) and every produced
  `FileScanTask` carries it. The TODO at the `into_file_scan_task` site is retired.
  Flow: `TableScan`/`PlanContext` (holds table metadata) → `ManifestEntryContext` →
  `FileScanTask`. No per-task re-parse.
- **C-3 (failure semantics = Java):** decode Java 1.10.0 first —
  `NameMappingParser.fromJson` on an invalid mapping THROWS (scan fails loud); mirror
  exactly: invalid JSON in the property ⇒ typed `Error` (DataInvalid) naming the
  property, never silent `None`. Absent property ⇒ `None` (position-fallback path
  unchanged). Empty-string/whitespace: match whatever Java does — verify, don't assume.
- **C-4 (delete-file tasks):** determine from Java 1.10.0 source whether `DeleteFilter`
  / delete-file readers apply the table name mapping when a delete file lacks field IDs
  (delete files are normally Iceberg-written and carry IDs). Match the Java contract:
  wire it if Java does; if Java does not, leave the delete-task sites as-is and record
  the finding (with the Java citation) in the build summary. Do NOT guess.
- **C-5 (unit pins, same commit):** (a) plan over a table with the property set ⇒ every
  task carries the parsed mapping (assert mapping content, not just `is_some`); (b)
  absent property ⇒ `None`; (c) invalid JSON ⇒ loud typed error naming the property;
  (d) END-TO-END: an ID-less Parquet data file in a table with a name mapping scans to
  the CORRECT columns via `TableScan` (not just the reader-level unit that already
  exists in `record_batch_transformer.rs`), including a case where the mapping is
  NON-trivial (mapped names differ from a naive positional match, so position-fallback
  would return provably WRONG data — that contrast is the pin). Every pin
  mutation-provable: re-hardcoding `None` at the wiring site must go RED.
- **C-6 (interop, Direction 1):** new suite `dev/java-interop/run-interop-name-mapping.sh` —
  Java (real 1.10.0 core, existing `InteropOracle` harness; `dev/java-interop/README.md`
  is the harness contract) creates a table over an ID-less Parquet file (write the file
  with a plain Parquet writer so NO field IDs are present, append it as a `DataFile`,
  set `schema.name-mapping.default` via `UpdateProperties` — the core-API equivalent of
  the Spark `add_files` procedure) and commits; Rust `TableScan` reads it and verifies
  row content lands in the CORRECT columns. ≥1 sabotage step (e.g. corrupt/remove the
  mapping property in a copied fixture, or swap two mapped names → verifier must go
  RED); sabotage hard-fails when inapplicable, never SKIPs (CLAUDE.md). Suite
  auto-discovered by `scripts/run_interop_suites.sh`; `SUITE_FLOOR_DEFAULT` 49→50 in
  the same change; `dev/java-interop/map.md` + README scenario table in lockstep.
- **C-7 (docs):** GAP_MATRIX **row R143** cell (currently a bare ✅) gains the
  scan-wiring note + date in that ONE cell (one home per fact;
  `./scripts/check_matrix_anchors.sh` green). Retire the stale claim in
  `arrow/record_batch_transformer.rs` docs if any ("applied in ArrowReader" is true
  only after this unit wires the producer). `scan/` has no `map.md` (verify; do not
  create one unless the surrounding tree already uses the convention there).

## Boundaries (not in scope)

- NO Avro name-mapping fallback — that is **row R119**'s named residue and stays there
  (`avro_reader.rs` errors loudly on ID-less Avro; unchanged).
- NO ORC.
- NO write-side name-mapping generation (Java `MappingUtil.create`) unless a test
  needs a mapping JSON fixture — construct fixtures by hand/serde instead.
- NO Cargo.toml/Cargo.lock edits. NO new dependencies.
- NO changes to `apply_name_mapping_to_arrow_schema` semantics UNLESS the e2e pin
  proves a genuine bug in it (if so: fix ships in this unit with its own pin and is
  called out in the build summary).

## Gate (unit)

`git ls-files -z | xargs -0 typos --force-exclude` · `cargo fmt --all -- --check` ·
`cargo clippy -p iceberg --lib --tests -- -D warnings` · `cargo test -p iceberg --lib`
(2775 baseline, no regressions) · the new interop suite green end-to-end via
`scripts/run_interop_suites.sh --only run-interop-name-mapping.sh` (and `--selftest`
untouched-green) · `./scripts/check_matrix_anchors.sh` · `./scripts/check_agent_artifacts.sh`.
Chain gate→commit in ONE `&&` chain; explicit paths, never `git add -A`.
