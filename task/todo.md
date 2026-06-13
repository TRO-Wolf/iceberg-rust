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

# Plan / Todo

The current plan for in-flight work. The operating manuals ([skills/](../skills/)) require this file
to be written **before** any non-trivial change and kept current as work proceeds.

How to use it (see the manuals' §1):

- Write a 3–7 bullet plan here before writing code.
- Flip `[ ]` → `[x]` as items complete; add a one-sentence "what changed and why" per step.
- Add indented sub-bullets when a step reveals unexpected complexity.
- Leave an `Outcome:` / `Done:` note when the work lands.

---


> **Archival log.** Last pass: 2026-06-12 (pass 4 — post-Wave-5 union, 680 lines) →
> [todo-archive/2026-06_wave5.md](todo-archive/2026-06_wave5.md) (8 spent Wave-5 increment
> sections; the ACTIVE queue refreshed in place). Prior passes: 2026-06-12 (pass 3 — 2,358
> lines → the wave3-wave4 file), 2026-06-11 (pass 2), 2026-06-09 (pass 1). Procedure:
> [skills/compaction.md](../skills/compaction.md) §Todo Archival.

## INCREMENT R2 (2026-06-12): partition-stats interop chain extension — incremental + exotic-type (BUILDER Sonnet, wt-r2)

Charter: Extend the partition-stats interop chain (Z3) to (a) the incremental compute path — both Rust-writes→Java-reads and Java-writes→Rust-reads; (b) exotic partition-value types (uuid as the spiciest; also time/fixed/binary if cheap); (c) a SEMANTIC sabotage on a merged counter; (d) GAP_MATRIX row 118 re-audit. All env-gated tests, fail-closed sentinels, chain ×2.

**Plan (pre-code, 7 bullets per manual §1):**
- [x] **Step 1 (Java IncrementalPartitionStatsOracle):** Add `IncrementalPartitionStatsOracle` static
  inner class to `InteropOracle.java`. `generate` mode: 3-snapshot fixture — S1 fast-append (cat=a 3rec, cat=b 2rec), compute stats S1 (FULL, saved as `base_stats`), S2 MERGE-APPEND (new cat=a file X + pos-delete PD2), compute stats S2 (INCREMENTAL — Java selects incremental because base_stats for S1 exists), verify incremental path. Also `generate-incremental-java-to-rust` mode writing `java_incremental_stats.json` + the Java-written table. `verify-incremental-partition-stats` mode: Java reads Rust's S2 stats file + compares against `incremental_expected.json`. Dispatch via `-Dinterop.partition_stats_incr.dir`. _What changed:_ landed `IncrementalPartitionStatsOracle` (generate/verify) emitting `java_incr_table/` + `java_incr_stats.json`; the S2 fixture is `delete_files(file_a)` (DELETE snapshot → SUBTRACT arm) rather than a merge-append, since `fast_append` never produces a DELETED tombstone; 4 `PartitionStatsOracle` helpers de-privatized to package-private for cross-sibling reuse.
- [x] **Step 2 (Rust incremental interop tests):** Add `test_partition_stats_incr_gen` + `test_partition_stats_incr_d2_rust_reads_java` to `crates/iceberg/tests/interop_partition_stats.rs`. GEN: S1+compute+register, S2+compute (assert incremental path via `latest_stats_file`), write `rust_incr_table/metadata/final.metadata.json` + `incremental_expected.json`. D2: read Java's incremental stats file, compare against `java_incremental_stats.json`. _What changed:_ both tests landed env-gated on `ICEBERG_INTEROP_PARTITION_STATS_INCR_DIR` (clean no-op early-return when unset); GEN writes `rust_incr_table/metadata/final.metadata.json` + `incr_expected.json`, D2 reads the Java parquet and compares against `java_incr_stats.json`.
- [x] **Step 3 (Exotic-type oracle):** Add `UuidPartitionStatsOracle` to `InteropOracle.java` — table partitioned by uuid identity (V2), one snapshot, Java writes stats, Rust reads. If time/fixed/binary is cheap in the same test struct, include them; else name residue. `generate-uuid-partition-stats` / `verify-uuid-partition-stats` modes. _What changed:_ landed `UuidPartitionStatsOracle` (generate/verify) over `identity(partition_id: uuid)`, fixed UUID `550e8400-e29b-41d4-a716-446655440000`; time/fixed/binary deliberately NOT added to the oracle — named as precise residue (their byte forms are already proven by O2's production-reader round-trip; only the interop-oracle extension is deferred).
- [x] **Step 4 (Exotic-type Rust tests):** `test_partition_stats_uuid_gen` + `test_partition_stats_uuid_d2` in `interop_partition_stats.rs`. _What changed:_ both tests landed env-gated on `ICEBERG_INTEROP_PARTITION_STATS_UUID_DIR` (clean no-op when unset); they surfaced and pinned the `UInt128`-Uuid Avro serde fix in `spec/values/serde.rs` (the production change of this increment).
- [x] **Step 5 (Run script):** Extend `run-interop-partition-stats.sh` with new steps for incremental (steps 8a-8e: Java gen → Rust INCR GEN → Java D1 verify → Rust D2 → SEMANTIC sabotage) and exotic UUID (steps 9a-9d: Java gen → Rust UUID GEN → Java D1 verify → Rust D2). Sabotage battery: at least one SEMANTIC sabotage on the incremental merged counter (8e — in-place byte-edit the Rust S2 incremental stats SOURCE parquet, flip the first zero INT64 to 1, `.bak` restore, HARD-FAIL when the zero pattern is absent — the Z3 7b pattern). Control (8d clean D2) + FAIL (8e closed) provenance explicit. _What changed:_ the chain grew 8→10 steps; 8e is an in-place byte-edit with `.bak` restore + hard-fail-on-unappliable (NOT an env-var `sabotage_src` design, and NOT a SKIP — a skipped sabotage is a false-green per lessons.md; critic fixed the original exit-42 skip to `exit 1`).
- [x] **Step 6 (Incremental path observability):** Since `latest_stats_file` is private, pin the incremental path engagement with a unit-level assertion in the GEN test (assert base file exists for S1 before calling compute for S2, assert result differs from full recompute). The O2 lessons confirm this is the correct approach. _What changed:_ pinned INDIRECTLY via observable semantics — assert the S1 base stats file is in the snapshot lineage before computing S2, then assert the carried cat=b row's `last_updated_snapshot_id` stays the BASE id (a fresh full-compute would bump it to S2) and the subtracted cat=a counter is zero.
- [x] **Step 7 (GAP_MATRIX + journal + gate):** Re-audit row 118. If every bidirectional claim is now covered by interop, flip to ✅; if any direction remains uncovered (exotic types limited to Rust-reads-Java only, etc.) keep 🟡 with precise residue. Pipe-count audit. Gate: typos + fmt + clippy + lib tests + run-interop-partition-stats.sh ×2 + taplo. Lessons appended. _What changed:_ row 118 stays 🟡 with R2-INCR + R2-UUID interop noted and time/fixed/binary interop named as precise residue; pipe-count audit clean; lessons appended (UInt128-Uuid serde, SUBTRACT-arm `delete_files`, 8e in-place sabotage, Java inner-class visibility).

**Residue (named, NOT half-built):** time/fixed/binary partition-value types in the partition-stats interop ORACLE are deferred. Their on-disk byte forms (`Time64(Microsecond)`; `FixedSizeBinary(L)`; `LargeBinary`) already round-trip through O2's production reader/writer and are unit-pinned in `maintenance/partition_stats.rs`; only the Java-interop-oracle extension (a new `*PartitionStatsOracle` fixture per type + chain steps) is outstanding. UUID landed as the spiciest exotic type. `ComputeTableStats` interop is Group Y's separate deferral.

**Outcome (2026-06-13):** R2 partition-stats interop chain extension LANDED — R2-INCR (SUBTRACT arm, bidirectional, 8a-8e incl. SEMANTIC sabotage) + R2-UUID (exotic type, bidirectional, 9a-9d), chain ×2. Production fix: `PrimitiveLiteral::UInt128` typed `PrimitiveType::Uuid` now serializes to an Avro String (was `Bytes`, rejected by apache-avro's `resolve_uuid`) with the matching deserialize arm. time/fixed/binary interop named as precise residue.

## ACTIVE (2026-06-12): I1 — theta-blob interop (LANDED)

**Plan (pre-code, 3–7 bullets per manual §1):**
- [x] **Step 1 (Oracle Java):** Added `ThetaBlobOracle` static inner class to `InteropOracle.java` with `generate-interop-theta`, `verify-interop-theta`, `generate-interop-theta-java-to-rust`. Added datasketches-java/memory deps to pom.xml.
- [x] **Step 2 (Oracle dispatch):** Wired all three modes into the InteropOracle main switch via `-Dinterop.theta.dir`.
- [x] **Step 3 (Rust test):** `crates/iceberg/tests/interop_theta.rs` — `test_theta_gen` (GEN: real 2-file table, `ComputeTableStats::execute`, `rust_stats.puffin` + `rust_stats_expected.json`) + `test_theta_d2_rust_reads_java_puffin` (D2: `java_stats.puffin` via `PuffinReader`+`CompactThetaSketch::deserialize`). Fixed: `BlobMetadata` fields are private (use methods); `FileIO::from_path` doesn't exist (use `FileIO::new_with_fs()`).
- [x] **Step 4 (Run script):** `dev/java-interop/run-interop-theta.sh` — 6-step chain ×2, sabotage battery 4 closed (6a truncate puffin; 6b Puffin-footer-parsed SOURCE corrupt; 6c truncate Java puffin; 6d corrupt ndv JSON). Puffin footer structure: `[data][footer_magic(4)][footer_json(N)][payload_len(4 LE u32)][flags(4)][trailing_magic(4)]` with blob offsets absolute from file start.
- [x] **Step 5 (GAP_MATRIX update):** ComputeTableStats row updated, I1 interop noted, pipe-count audit clean (all 61 `^|` rows have 5 pipes).
- [x] **Step 6 (Gate):** typos/fmt/clippy/lib-tests/run-interop-theta.sh/taplo all PASS. 2210 lib tests pass.
- [x] **Step 7 (journal):** Lessons appended to task/lessons.md.

**Outcome (2026-06-12):** I1 theta-blob interop COMPLETE — bidirectional, chain ×2, sabotage 4 closed. `ComputeTableStats` is now fully proven through end-to-end Java/Rust interop.

## ACTIVE (2026-06-12): I2 — view metadata interop (LANDED)

**Plan (pre-code, 7 bullets per manual §1):**
- [x] **Step 1 (Java ViewOracle — D1 generate):** Add `ViewOracle` static inner class to
  `InteropOracle.java`. `generate` mode: use `InMemoryCatalog.buildView(ident)` to create a
  view with schema + 2 SQL representations (spark+trino dialects), then `replace()` with a
  DIFFERENT SQL so `reuseOrCreateNewViewVersionId` creates version 2 (distinct — not identical).
  Write `ViewMetadataParser.toJson(metadata)` to `rust_view_metadata.json`. Emit companion
  `expected.json` (view-uuid, format-version, location, schema field list, current-version-id,
  version count, version-log count, all per-version field values). Dispatch via
  `-Dinterop.view.dir` + `generate-interop-view`. Key fix: cast `View` → `BaseView` for
  `operations().current()`.
- [x] **Step 2 (Java ViewOracle — D1 verify + D2 generate):** `verify-interop-view` mode:
  reads `rust_view_metadata.json` (Rust-written) via `ViewMetadataParser.fromJson`, asserts
  field-by-field against values Rust emitted into `rust_view_expected.json`. `generate-java-to-rust`
  mode: Java builds a view metadata object via `InMemoryCatalog`, writes via
  `ViewMetadataParser.toJson` to `java_view_metadata.json`, emits `java_view_expected.json`.
  Dispatch the three modes from the main switch.
- [x] **Step 3 (Rust interop test):** `crates/iceberg/tests/interop_view.rs`. Three tests:
  `test_view_gen` (D1 GEN: Rust creates view + ReplaceViewVersionAction commit → 2 versions,
  writes `rust_view_metadata.json` via `ViewMetadata::write_to` + `rust_view_expected.json`);
  `test_view_d2_rust_reads_java` (D2: reads `java_view_metadata.json` via
  `ViewMetadata::read_from`, asserts all fields vs `java_view_expected.json`);
  `test_view_tolerance_controls` (control: permuted-field-order JSON still parses on Rust
  side; empty-properties omission tolerated both ways). Fixed clippy::never_loop in
  `first_sql_repr` helper.
- [x] **Step 4 (Chain script):** `dev/java-interop/run-interop-view.sh` — 6-step chain ×2:
  (1) reset tmp; (2) Rust GEN test writes `rust_view_metadata.json`; (3) Java verify-interop-view
  reads Rust metadata, 0 failures sentinel; (4) Java generate-java-to-rust writes
  `java_view_metadata.json`; (5) Rust D2 reads Java metadata; (6) sabotage battery 5 closed
  (6a alter SQL→Java D1 FAIL; 6b drop default-namespace→Rust parse FAIL; 6c dangling
  current-version-id=99→Rust FAIL; 6d alter SQL in java metadata→Rust assert FAIL; 6e tolerance
  control both sides PASS).
- [x] **Step 5 (Tolerance controls):** The field-order control and empty-properties control
  documented in `test_view_tolerance_controls` and confirmed in the chain script (6e). Java's
  omit-empty-properties (read by Rust) and Rust's always-emit-empty (read by Java) are pinned
  as COSMETIC ONLY — byte-level byte-order is a NEXT-WAVE item.
- [x] **Step 6 (GAP_MATRIX update):** Re-audited ViewCatalog row. Row stays 🟡 because: (a)
  MemoryCatalog `update_view` has no base-location CAS (O1 on other branch); (b) Glue/S3Tables
  views need credentialed sprint; (c) SessionCatalog/LockManager separate. Residue text updated
  precisely; I2 interop landing noted. Pipe-count audit clean (61 rows all 5 pipes).
- [x] **Step 7 (Gate + journal):** `typos` PASS, `cargo fmt --check` PASS, `cargo clippy` PASS
  (0 warnings), `cargo test -p iceberg --lib` 2210 tests, 0 failures (exact baseline). Plus
  run-interop-view.sh ×2 green, taplo check (pre-existing 4 failures, 0 new). Lessons appended.

**Outcome (2026-06-12):** I2 view metadata interop COMPLETE — bidirectional, chain ×2, sabotage
5 closed. `ViewMetadata::read_from`/`write_to` proven against Java `ViewMetadataParser` through
real catalog operations (`MemoryCatalog` + `InMemoryCatalog`). Wire-format field-order divergence
pinned as cosmetic-only; byte-exact view round-trip is next-wave.

## ACTIVE (2026-06-12): I3 — data-level WAP interop (LANDED)

**Plan (pre-code, 7 bullets per manual §1):**
- [x] **Step 1 (Java WapDataOracle — D1 generate + D2 verify):** Added `WapDataOracle` static inner
  class to `InteropOracle.java`. S-replay order: base→stage→bump so staged.parent≠head → REPLAY.
  Uses REAL parquet via `MergeAppendDataOracle.writePartitionedDataFile`. Bump row id=99 is a real
  data fast-append (not updateProperties which doesn't create a snapshot — key lesson). verifyRustTable
  uses `Files.createTempDirectory` per call to avoid v0.metadata.json collision on repeated runs.
- [x] **Step 2 (Rust interop test):** `crates/iceberg/tests/interop_wap_data.rs`. Two tests:
  `test_wap_data_gen_rust_writes_staged_table` (D1 GEN: S-replay order, asserts staged_id≠current
  after stage_only(), writes final.metadata.json + rust_staged_snapshot_id.json);
  `test_wap_data_d2_rust_reads_java_cherrypick_table` (D2: reads java_cherrypick_table, 8 rows,
  WAP semantics, partition routing a/b pinned).
- [x] **Step 3 (Chain script):** `dev/java-interop/run-interop-wap-data.sh` — 7-step chain ×2
  green. Step 5 sentinel "verify-interop-wap-data: 0 failures", Step 7 sabotage battery 4 closed.
- [x] **Step 4 (Sabotage battery):** 4 closed: 7a STRUCTURAL truncate metadata; 7b STRUCTURAL bogus
  manifest-list path; 7c SEMANTIC corrupt wap.id w1→w1-CORRUPTED (cherry-pick emits wrong
  published-wap-id → pin fires); 7d STRUCTURAL remove staged snapshot entirely.
- [x] **Step 5 (Dedup):** deferred — inflates scope; recorded as next-wave residue in GAP_MATRIX.
- [x] **Step 6 (GAP_MATRIX update):** cherry-pick/WAP row residue updated; I3 landing noted; pipe-count audit clean.
- [x] **Step 7 (Gate + journal):** typos/fmt/clippy/lib-tests (2210/0)/chain×2/taplo all PASS. Lessons appended.

**Outcome (2026-06-12):** I3 data-level WAP interop COMPLETE — bidirectional REPLAY-shape, REAL
parquet, 8-row fixture (base 4 + bump 1 + staged 3), chain ×2, sabotage 4 closed. Key lessons:
S-replay order (stage before bump); updateProperties does not create a snapshot; LocalTableOperations
v0.metadata.json collision fixed with Files.createTempDirectory.
## INCREMENT O3 (2026-06-12): divergence burn-down — type-name case / sort-order bind / manifest-list order / map-value avro naming (BUILDER Opus, wt-core6)

Charter: brief O3. Four reported Java↔Rust divergences; bounded cleanup. Per item: derive Java 1.10.0
behavior from bytecode (`/tmp/o3-bytecode`, jars in ~/.m2), then FIX if real bounded parity or DOCUMENT
in the matrix if cosmetic/out-of-scope.

Java 1.10.0 bytecode findings (all from `javap -p -c`, iceberg-api/core-1.10.0.jar):
- (a) `Types.fromTypeName(name)` does `name.toLowerCase(Locale.ROOT)` then matches the TYPES map (all
  primitive names) + the FIXED (`fixed\[\s*(\d+)\s*\]`) and DECIMAL (`decimal\(\s*(\d+)\s*,\s*(\d+)\s*\)`)
  regexes against the LOWERCASED string. `SchemaParser.typeFromJson` matches the WRAPPER names
  `struct`/`list`/`map` with `String.equals` (CASE-SENSITIVE — NOT lowercased). So Java folds case for
  primitives + fixed/decimal ONLY, not the object wrappers. Rust `PrimitiveType` custom deserializer is
  lowercase-exact → reject `BOOLEAN`/`Decimal(..)`/`FIXED[5]` Java accepts. FIX: lowercase the primitive
  name before dispatch; leave wrappers untouched (Rust matches them structurally anyway).
- (b) `TableMetadataParser.fromJson` → `SortOrderParser.fromJson(Schema, JsonNode, defaultId)`: builds
  an `UnboundSortOrder`, then if `orderId == defaultSortOrderId` calls `bind` (→ `build()` →
  `SortOrder.checkCompatibility` = 3 checks: source col exists / source primitive / transform
  applicable), ELSE `bindUnchecked` (no checkCompatibility). So ONLY the default sort order is
  validated at parse time. Rust `try_normalize_sort_order` runs NEITHER — it only checks order-id-0
  has no fields + the default id exists. FIX: run `check_compatibility` on the DEFAULT sort order at
  normalize time (Rust already has the 1:1 `check_compatibility`); non-default orders stay lenient.
- (c) `FastAppend.apply` = new-then-carried; `MergingSnapshotProducer.apply` = `concat(prepareNew, carried)`
  for data then delete (new-then-carried, all-data-before-all-delete). Rust shared `manifest_file<OP,MP>`
  = carried(process_deletes, data+delete mixed) THEN new-data THEN new-delete — OPPOSITE order, in the
  ONE shared path every action uses. Interop oracle SORTS manifests, both readers reconcile by seq →
  cosmetic. DOCUMENT (shared-path blast radius), no fix.
- (d) `TypeToSchema.struct` names a struct record `namesFunction(fieldIds.peek, struct)` else `"r"+id`
  (recipe `r`); for a map value the deque top is the value-id → `r<value_id>`. Rust `map()`
  visitor leaves a struct value record at the `"null"` placeholder (only `rename_variant_record` runs);
  two struct-valued maps in one schema → two `"null"` records → Java `Schema.Parser` "Can't redefine".
  REAL + bounded. FIX: rename ANY map key/value record (struct incl.) to `r<field_id>`, matching what
  `field()`/`list()` already do for non-map placements.

- [x] (a) FIXED. datatypes.rs `PrimitiveType` deserialize `.to_lowercase()` before decimal/fixed/exact
      dispatch + variant marker `eq_ignore_ascii_case` (Java `fromTypeName` Locale.ROOT). Replaced the
      case-sensitive pin with 3 tests: mixed-case primitives + `Decimal(..)`/`FIXED[..]`/`Variant` fold;
      non-type-name negatives; wrapper-not-folded scope pin (`{"type":"STRUCT"}` rejected by the
      `StructType` deserializer, Java `String.equals`). Self-mutation (drop `.to_lowercase()`) → BOOLEAN fails.
- [x] (b) FIXED. table_metadata.rs `try_normalize_sort_order` runs `SortOrderBuilder::check_compatibility`
      on the DEFAULT sort order vs the current schema (made `pub(crate)`); non-default unchanged. 4 tests:
      valid default parses; default on missing source / non-primitive source fails with Java's messages;
      non-default on missing source STILL parses (Java bindUnchecked). Self-mutation (drop the call) → both
      reject-tests parse.
- [x] (c) DOCUMENTED in matrix (manifest-list row): Java new-then-carried all-data-before-delete
      (`FastAppend.apply` / `MergingSnapshotProducer.apply` bytecode) vs Rust shared `manifest_file`
      carried-then-new; cosmetic (oracle sorts at `snapshot_meta_view.rs:127`, readers reconcile by seq);
      exact-match needs a content-type-separated restructure of the shared path (ripples into
      `manifests[0]` tests) — out of scope. No fix.
- [x] (d) FIXED. avro/schema.rs `rename_variant_record`→`rename_map_record` renames ANY map key/value
      record (struct incl.) to `r<field_id>` (Java `TypeToSchema.struct` = `"r"+fieldIds.peek()`). 2 tests:
      two string-key struct maps → distinct `r3`/`r7`; array-form struct-key → inner `r<keyId>`/`r<valueId>`
      (explicit names, not just round-trip — round-trip alone is weak here). Self-mutation (variant-only) →
      both tests fail (`"null"`).
- [ ] Gate chain (typos + fmt + clippy + lib test) + taplo + matrix pipe-audit; interop chain if needed;
      report to /tmp/wave6/O3-builder.md.

Outcome: (a)/(b)/(d) FIXED with bytecode-derived parity + mutation-pinned tests; (c) DOCUMENTED
(cosmetic, shared-path blast radius). No interop surface touched by the fixes (avro map-value naming
only affects struct/variant-valued maps, none in the interop fixtures; sort-order/type-case are
parse-time read-tolerance). 9 new tests; matrix updated (4 cells, 5-pipe audit clean).

### O3 REVIEWER (CRITIC) plan (2026-06-12, wt-core6) — adversarial pass vs 1.10.0 bytecode + live probe
- [x] (a) re-derive `Types.fromTypeName` bytecode + live Java probe; build a Rust-vs-Java acceptance
      table over the flagged edge spellings. FOUND 5 fixed/decimal parse mismatches (Rust too-lenient
      on missing-close `decimal(38,2` / `fixed[16`; too-strict on inner-whitespace `fixed[ 16 ]`).
- [x] (b) re-derive `SortOrderParser.fromJson` + `bind`/`bindUnchecked` + var-10=current-schema:
      CONFIRMED exactly. Mutation: knockout `check_compatibility` → 2 reject tests fail. Reverted.
- [x] (c) confirm new-then-carried (FastAppend) + data-before-delete (MergingSnapshotProducer) +
      Rust carried-then-new; confirm oracle sorts (snapshot_meta_view:127) + `current_manifests`
      returns RAW order (so `manifests[0]` tests ARE coupled) → DOCUMENT verdict CONFIRMED not lazy.
- [x] (d) revert to variant-only → reproduce both struct-map failures; restore. Confirmed KEY+VALUE
      + array-form explicit-name pins; manifest schema has no struct-valued maps → byte-invisible.
- [x] FIX vector 1: killed the 5 fixed/decimal parse mismatches via `strip_prefix`+`strip_suffix`+`trim`
      in `deserialize_fixed`/`deserialize_decimal` (Java anchored-regex parity). Pinned both arms in
      `parameterized_type_parse_matches_java_fixed_and_decimal_regex`; mutation (revert to `trim_end_matches`)
      fails it.
- [x] FIX vector 5: the wrapper read-leniency was REAL via the `Type` path (`{"type":"STRUCT"/"LIST"/"MAP"}`
      all parsed where Java's `String.equals` rejects — the builder's scope test used the WRONG deserializer).
      Cheap contained fix: `SerdeType::wrapper_type_mismatch()` + guard in `Type::deserialize` (≤30 lines, no
      untagged-machinery fight). Pinned both arms via the production `Type` path; mutation (guard knockout)
      fails it. Live-Java-confirmed (`SchemaParser.fromJson`: lowercase OK, upper ERR).
- [x] Mutation battery (all 5 fixes, for real, reverted): (a) drop `.to_lowercase()` → case-fold test fails;
      (b) drop `check_compatibility` → 2 reject tests fail; (d) variant-only rename → both struct-map tests
      fail; vector-1 `trim_end_matches` revert → parameterized test fails; vector-5 guard knockout → wrapper
      test fails. Every fix pinned.
- [x] Gate GREEN AFTER fixes: typos + fmt + clippy + `cargo test -p iceberg --lib` (2236 passed, 0 failed;
      +2 net over the builder's 2234) + taplo + matrix pipe-audit (all `^|` rows 5 pipes). Interop write-data
      chain re-run GREEN (exit 0, 22 steps). Matrix cell refreshed; lessons (O3 REVIEWER) recorded.

Outcome (REVIEWER): (b)/(c)/(d) CONFIRMED by independent bytecode re-derivation + mutation battery — no
defect. (a) CONFIRMED but INCOMPLETE — found + FIXED 5 fixed/decimal parse-fidelity mismatches the case-fold
missed. The documented "open risk" (wrapper read-leniency) was REAL + worse than framed (builder's scope test
checked the wrong deserializer path) — FIXED with a contained ≤30-line guard. VERDICT: SHIP.

## INCREMENT O2 (2026-06-12): partition-stats residue — incremental compute + exotic value types (BUILDER Opus, wt-core6)

Charter: brief O2. Two halves on `maintenance/partition_stats.rs`. (a) the INCREMENTAL compute path
(Java 1.10.0 `PartitionStatsHandler.computeAndWriteStatsFile(table, snapshotId)` chooses incremental
when a prior stats file exists in the target snapshot's lineage). (b) exotic partition value types
(time/uuid/fixed/binary) in the stats-file write path (currently loud FeatureUnsupported).

Java 1.10.0 contract DERIVED FROM BYTECODE (iceberg-core-1.10.0.jar, javap; /tmp/ps-bytecode):
- `computeAndWriteStatsFile(table, snapshotId)`: precond table≠null, isPartitioned, snapshot found.
  `latestStatsFile(table, snapshotId)` → null ⇒ FULL compute (LOG "not present"). Non-null & its
  snapshotId == target ⇒ RETURN the existing file as-is (no recompute). Else (differs) ⇒
  `computeAndMergeStatsIncremental`, and on `InvalidStatsFileException` (base file unreadable) ⇒ LOG
  WARN + FULL compute fallback (exception table 133-147). Then empty⇒null else sort+write.
- `latestStatsFile`: map snapshotId→file from `partitionStatisticsFiles()`; walk `ancestorsOf(target)`
  (lineage back via parentId, inclusive); return the FIRST ancestor with a stats file, else null.
- `computeAndMergeStatsIncremental`: seed statsMap from the base stats file rows (read via
  `readPartitionStatsFile(schema(unifiedType, fv), file.path())` — whole try wrapped in catch(Exception)
  → throw InvalidStatsFileException); then `computeStatsDiff(table, table.snapshot(baseFile.snapshotId),
  toSnapshot)` and merge each diff row into the seed via `appendStats` (counts ADD, last-updated MAX).
- `computeStatsDiff(table, from, to)`: `ancestorsBetween(to.id, from.id)` → per snapshot flatMap its
  `allManifests(io).filter(manifest.snapshotId().equals(snapshot.snapshotId()))` (ONLY the manifests
  ADDED by that snapshot) → `computeStats(table, manifests, incremental=true)`.
- `collectStatsForManifest(.., incremental)` entry dispatch (bytecode 197-259): LIVE entry —
  incremental && status≠ADDED ⇒ SKIP; else `liveEntry` (+counts). NON-live tombstone — incremental ⇒
  `deletedEntryForIncrementalCompute` (SUBTRACTS the file's counts: data/pos/eq rec+file, PUFFIN→dv,
  then updateSnapshotInfo); non-incremental ⇒ `deletedEntry` (last-updated only).
- Exotic byte forms (verified vs reader `arrow_struct_to_literal` + writer `to_arrow` in arrow/):
  Time→`Time64(Microsecond)` from `PrimitiveLiteral::Long`; Uuid→`FixedSizeBinary(16)` 16 BE bytes
  (`Uuid::from_u128(v).into_bytes()`, reader `Uuid::from_bytes`); Fixed(len)→`FixedSizeBinary(len)`
  from `PrimitiveLiteral::Binary`; Binary→`LargeBinary` from `PrimitiveLiteral::Binary`.

- [x] Port incremental: `deleted_entry_for_incremental_compute` (subtract mirror of `live_entry`); threaded
      `incremental` through `collect_stats_for_manifest` (ADDED-only LIVE via `entry.status()==Added`,
      tombstone→subtract); added `latest_stats_file` (lineage walk), `compute_stats_diff` (`(base,target]`
      range × `added_snapshot_id`-filtered manifests), `compute_and_merge_stats_incremental`; branched
      `compute_and_write_stats_file` (full / return-existing / incremental w/ corrupt-base→`Ok(None)`
      fallback). No new error kind needed — corrupt base maps to `Ok(None)`, the fallback signal.
- [x] Exotic types in `build_partition_field_column`: Time→`Time64MicrosecondArray`, Uuid→`FixedSizeBinary(16)`
      16 BE bytes, Fixed→`FixedSizeBinary(L)`, Binary→`LargeBinaryArray` (two helpers for the sparse-iter
      FixedSizeBinary constructors); removed the loud-error residue arm (match is now exhaustive over primitives).
- [x] Tests (a): incremental==full on append-only (2 shapes: single-field + with-delete-in-diff);
      return-existing; corrupt-base fallback; per-cell subtract unit; ADDED-only filter via merge-append
      fixture. 4 mutations caught (knock-out subtracted field; drop ADDED-only filter; drop fallback;
      include-all-manifests-in-diff).
- [x] Tests (b): 4 per-type round-trips (write→reopen→read, value + Arrow child type exact) + fixed-width
      loud-error guard. Replaced `test_unsupported_partition_value_type_errors_loudly` (binary now supported).
- [x] Interop decision: Z3 chain RE-RUN GREEN (both chains + 4 sabotage). Extending the oracle for
      incremental/exotic is > ~1h (new InteropOracle case + fixture, pom-free) → DEFERRED, residue recorded
      in matrix row 118 (byte forms already verified vs Java via the production reader round-trip).
- [x] Gate chain + taplo + Z3 GREEN; matrix row 118 cell updated (terse, 5-pipe audit clean); lessons added;
      report to /tmp/wave6/O2-builder.md.

Outcome: partition-stats incremental compute (Java-bytecode-faithful selection + diff + subtract) and the
time/uuid/fixed/binary write-path residue both LANDED in `maintenance/partition_stats.rs`. Lib suite 2225
(>= 2215 baseline), 0 failures; Z3 interop green; interop extension deferred (residue named in matrix).

## INCREMENT O1 (2026-06-12): optimistic-concurrency parity for MemoryCatalog (BUILDER Opus, wt-core6)

Charter: U1-REVIEWER block in lessons.md. Today a STALE commit (built from a superseded base) to
MemoryCatalog `update_table`/`update_view` silently lands LWW. Java `InMemoryTableOperations.doCommit`
/ `InMemoryViewOperations.doCommit` compare the STORED location against the commit's BASE location
and throw `CommitFailedException` on mismatch. Close the gap on BOTH seams symmetrically — the CAS
belongs in the CATALOG (Java's posture), NOT in the requirement set.

- [x] Survey: base location does NOT reach MemoryCatalog today (`TableCommit`/`ViewCommit` carry
      only ident/requirements/updates). Java threads `base` into `doCommit`. REST only takes
      requirements/updates on the wire → unaffected by a new commit field. Retry loop gates on
      `e.retryable()`; conflict kind is `CatalogCommitConflicts` + `with_retryable(true)`.
      Java message shape (bytecode-pinned): "Cannot commit to {table|view} %s metadata location
      from %s to %s because it has been concurrently modified to %s".
- [x] Thread base metadata location through the seam: added `base_metadata_location: Option<String>`
      to `TableCommit` (catalog/mod.rs, `#[builder(default)]` + accessor) and `ViewCommit` (view.rs,
      + accessor). Populated from the base in transaction `do_commit` (`self.table.metadata_location`),
      both view actions (source `View::metadata_location`), and `register_partition_stats_file`. REST
      `update_*` still sends only requirements/updates → wire/semantics unaffected; SQL keeps its own
      store-side CAS and ignores the field.
- [x] Added `check_no_concurrent_modification` shared helper in MemoryCatalog and wired it into
      `update_table` + `update_view`: compares the STORED location against `commit.base_metadata_location`
      INSIDE the lock, BEFORE the write; on mismatch raises `CatalogCommitConflicts` + retryable with
      Java's "...because it has been concurrently modified to %s" message shape. Symmetric both seams.
- [x] Tests: `test_view_stale_second_replace_conflicts_via_location_cas` (view U1 probe),
      `test_table_stale_property_commit_conflicts_only_location_cas_can_fire` (table, empty-requirement
      commit so ONLY the CAS fires), `test_table_non_stale_commit_still_succeeds_with_cas` (happy path),
      `test_table_two_transactions_from_same_base_both_land_via_refresh` (refresh-and-retry e2e through
      the transaction machinery), `test_view_commits_carry_base_metadata_location` (view.rs plumbing pin).
- [x] Mutation check: replacing the CAS condition with `if true` → exactly the two stale-conflict tests
      FAIL, happy-path + refresh tests stay green. Reverted.
- [ ] Gate chain + taplo; report to /tmp/wave6/O1-builder.md.

Outcome: location-CAS landed symmetrically on both MemoryCatalog update seams, Java-bytecode-faithful;
SQL-catalog weaker-CAS divergence flagged in lessons as a follow-up (out of O1 scope).

## ACTIVE (2026-06-12): Near-full-parity direction — open queue (planning record)

Directive (user, 2026-06-11): run this fork's Roadmap to **almost the full 1:1 Java replacement**.
Waves 3–5 landed PRs #28–#41 (write-engine closeout, maintenance actions end-to-end incl.
ComputeTable/PartitionStats + the iceberg-sketches crate, the variant arc, stage_only + WAP,
views end-to-end, and TEN interop chains). Statuses live ONLY in the GAP_MATRIX.

- [ ] **Named next-wave interop items:** theta-blob interop (Java reads our
      apache-datasketches-theta-v1 puffin blobs / ndv) — **LANDED 2026-06-12 (I1)**;
      view interop field-SET equality — **LANDED 2026-06-12 (I2)**, with the wire byte-order
      comparison still the open half; data-level WAP interop — **LANDED 2026-06-12 (I3)**, with
      data-level FAST-FORWARD + column-vs-stamp routing as named residue. STILL OPEN: variant
      file-level I/O + interop (the parquet-crate boundary).
- [ ] **Partition-stats residue:** the INCREMENTAL compute path; time/uuid/fixed/binary partition
      values in stats files (loud errors today).
- [ ] **The shared-seam concurrency-parity increment** (U1 reviewer): no location-CAS on
      MemoryCatalog update_table/update_view (stale second commit last-write-wins); the SQL
      catalog has it per-catalog — port the posture to the shared seam + MemoryCatalog.
- [ ] **Reported divergences awaiting their increments:** manifest-list carried-vs-new entry ORDER
      (cosmetic, readers reconcile); Java lowercases ALL type names on parse (Rust exact-lowercase);
      Rust sort orders unbound on metadata parse; struct map-value Avro record naming hazard.
- [ ] **Scheduled with the user:** real-catalog (Glue + S3 Tables) hardening — needs credentials
      (now incl. the Glue/S3Tables VIEW surface).
- [ ] **Opus-queue (post-handoff or parallel):** ORC/Avro breadth, SessionCatalog + LockManager,
      incremental-scan interop, scan completion (BatchScan / CDC / split planning), encryption
      (frontier-grade — the 2026-06-22 window).

## Carried-forward open items (full context in todo-archive/)

**Explicitly NOT decided:** the "platform cut line" through the GAP_MATRIX (which rows block the
user's trading platform vs continuous-parity backlog, incl. re-ordering maintenance actions ahead of
Phase-4 format exotica) was proposed but is an **open user decision — do not assume it.**
  _RESOLVED-AS-TABLED 2026-06-11: the user tabled the DataFusion/RePark direction and redirected
  the fork to near-full 1:1 Java parity — recorded in Roadmap.md (decision record item 5 + the
  re-sequenced headline areas). Originating narrative:
  [todo-archive/2026-06_ops-hardening.md](todo-archive/2026-06_ops-hardening.md)._


## Archived increment narratives

Completed-increment narratives moved verbatim out of this file (see [skills/compaction.md](../skills/compaction.md)
§Todo Archival). Not session-start reading — grep/open on demand.

- [todo-archive/phase1.md](todo-archive/phase1.md) — Phase 1 spec & metadata completeness (schema /
  partition / snapshot evolution + spec-read robustness).
- [todo-archive/phase2.md](todo-archive/phase2.md) — Phase 2 write engine (write actions + the
  concurrent-commit conflict-validation cluster, incl. the merged write-validation PR #9).
- [todo-archive/phase3.md](todo-archive/phase3.md) — Phase 3 scan parity (residual evaluation,
  inspection tables, scan-metrics emission, and inspection / scan-execution interop).
- [todo-archive/2026-06_ops-hardening.md](todo-archive/2026-06_ops-hardening.md) — the doc-infrastructure / hardening meta-sprints (not phase work).
- [todo-archive/2026-06_wave3-wave4-overnight.md](todo-archive/2026-06_wave3-wave4-overnight.md) — Waves 3–4 + the overnight session (PRs #25–#37; pass-scoped).
- [todo-archive/2026-06_wave5.md](todo-archive/2026-06_wave5.md) — Wave 5 (PRs #39–#41; pass-scoped).
- Index: [todo-archive/map.md](todo-archive/map.md).
