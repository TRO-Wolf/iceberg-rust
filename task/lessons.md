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

# Lessons

Accumulated DO / DO NOT lessons. The operating manuals ([skills/](../skills/)) require reading this
file **in full at the start of every session**, and appending to it after **any** correction from
the user.

How to use it (see the manuals' §2):

- After any correction, append a **date-stamped** entry immediately.
- Write each as a concrete **DO** or **DO NOT** statement with the *why* and how to apply it.
- Supersede an outdated rule with a dated note (`_superseded YYYY-MM-DD: see ..._`) rather than
  editing the original in place.

> **Compaction log.** Last pass: 2026-06-12 (pass 4 — post-Wave-5 merge union, 1,238 lines;
> 4 KEEP / 25 ARCHIVE / 2 promoted) →
> [lessons-archive/2026-06_wave5.md](lessons-archive/2026-06_wave5.md). Promoted: 1 →
> [docs/testing.md](../docs/testing.md), 1 → `dev/java-interop/map.md#debug`. Prior passes:
> 2026-06-12 (pass 3 — 14/47/3) →
> [lessons-archive/2026-06_wave3-wave4-overnight.md](lessons-archive/2026-06_wave3-wave4-overnight.md);
> 2026-06-11 (pass 2 — 17/25/6); 2026-06-09 (pass 1 — 31 promoted). Archives are not read by
> default — see [skills/compaction.md](../skills/compaction.md).

---

<!-- Newest entries at the bottom. Example shape:

### YYYY-MM-DD
- **DO** carry context on every fallible Rust call (`.with_context(...)` / `.expect("msg")`).
  *Why:* a bare `.unwrap()` panic gives the operator no cause from logs alone.
- **DO NOT** edit upstream crate files to land a fork feature when an additive module would do.
  *Why:* it makes the next upstream merge conflict-prone. Prefer additive changes.
-->



### 2026-06-11 (Wave-4 F2 — variant shredding overlay + VariantVisitor, BUILDER Fable, wt-vschema)
- **1.10.0 `ShreddedObject` DIVERGES from MAIN twice — both found in bytecode and confirmed by live
  probe (`/tmp/variant-probe/ShredProbe.java`), and both are BUGS the port must not mirror:** (1)
  `remove()` lacks MAIN's `this.serializationState = null`, so `sizeInBytes()` → `remove(x)` →
  `writeTo` serializes the STALE cached state with x still present; (2) the `SerializationState`
  ctor's non-`SerializedObject` branch compiles to `aload_3` (the PARAMETER map = the overlay's
  LIVE `shreddedFields`) for the materializing merge while `this.shreddedFields` keeps the
  pre-merge COPY that `writeTo` iterates — the FIRST serialization over a constructed backing
  writes count/dataSize for the merged set but only the copy's fields ⇒ CORRUPT bytes (probe:
  re-read throws IndexOutOfBoundsException); a later state rebuild self-heals because the live map
  was polluted. MAIN fixed both (the param was renamed). The Rust port is STATELESS (plan computed
  fresh per call) and MAIN-consistent; the constructed-backing oracle is Java's SELF-HEALED second
  serialization (puts → sizeInBytes → re-put to reset the cache → serialize). LESSON: a
  source-vs-bytecode diff can hide in PARAMETER NAMING — `javap` shows `aload_<n>` vs `getfield`,
  which no source read reveals; for any stateful Java class, probe the mutate-after-size sequence.
- **The verbatim-slice contract is what makes the overlay safe over third-party data:**
  `SerializedObject.sliceValue(index)` spans are exactly B1's sorted-distinct-offsets field
  ranges, so `parse_object` grew an optional range recorder (one parser, no duplication) and the
  overlay copies untouched fields' ORIGINAL bytes — only ids/offsets/header recompute (ids
  re-resolve BY NAME at write time even for verbatim fields, Java `metadata.id` + "Invalid
  metadata, missing: %s"). The designated mutation (verbatim → canonical re-encode) is killed
  ONLY by the non-canonical fixtures (long-form string that fits short form; oversized offset
  width) — canonical-input fixtures pass under the mutation, so a fixture set without a
  non-canonical backing would NOT pin the contract at all.
- **Java's remove-then-put contract is deliberately inconsistent — mirror BOTH sides:** `put`
  does not clear `removedFields`, so after `remove(x)` + `put(x, v)` the views (`get` → null,
  `numFields`/`fieldNames` exclude x — removedFields filters them) DISAGREE with serialization
  (which includes x=v: the SerializationState's shredded map still carries it). Probe-pinned
  (`probe_remove_then_put_view numFields=2 get_b=null` + bytes containing b=9). Fixing either
  side to be "consistent" diverges from Java.
- **Duplicate backing-field names: the rejection is REPLACEMENT-SENSITIVE, so door at
  serialization time, not construction.** Java's ImmutableMap throws only when the duplicate
  SURVIVES the replaced/removed filter; putting or removing the duplicated name skips both
  occurrences and serializes fine (fixture-pinned both ways). An eager constructor-time door
  would over-reject vs Java. Constructed-backing duplicates collapse SILENTLY instead (HashMap
  put semantics) — two different dup behaviors in one class.
- **`VariantVisitor` facts (1.10.0 == MAIN, bytecode-verified):** drivers are `visit(Variant,
  visitor)` + `visit(VariantValue, visitor)` — the brief's `visit(VariantMetadata, VariantValue,
  VariantVisitor)` signature DOES NOT EXIST in 1.10.0 (brief ≠ spec, again); object traversal
  iterates `fieldNames()` (stored order) but recurses into `object.get(name)` — the NAME LOOKUP —
  so a non-name-sorted object NPEs in Java (Rust: named Err, after-hook still fired); after-hooks
  run in `finally` (pinned by asserting the hook fired on the error path); all defaults return
  null (Rust `Option::None`) and the result lists carry nulls. A Java-generated event log (the
  generator's LoggingVisitor) is a cheap, exact traversal-order oracle — pin the SEQUENCE, not
  properties of it.

### 2026-06-12 (Wave-5 Group U / U1 — view metadata + view ops + catalog CRUD, BUILDER Opus, wt-views)
- **The view SPEC builder was ALREADY a 1:1 Java port — the V2 "read before assuming unbuilt" lesson
  paid off; the actual gap was the entire CATALOG-facing surface.** `spec/view_metadata_builder.rs`
  (58 KB) is a faithful port of Java `ViewMetadata.Builder`: `reuse_or_create_new_view_version_id` ==
  `reuseOrCreateNewViewVersionId`/`sameViewVersion`, schema interning, version-log append + expiry
  (`version.history.num-entries`, default 10), the dialect-drop rules. What was MISSING and had to be
  built: `ViewMetadata::read_from`/`write_to`, a `View` type, `ViewCommit`/`ViewRequirement`, the
  `replace_version`/`update_properties` ops, and the `Catalog` trait view methods + a catalog impl.
  DO survey the spec layer in full before scoping a "views" increment — most of the hard metadata
  logic may already be ported from upstream.
- **The view REPLACE requirement set is `[AssertViewUUID]` ALONE — bytecode-pinned from
  `UpdateRequirements.forReplaceView`, the V1-lesson view sibling.** 1.10.0 bytecode: `forReplaceView`
  seeds ONE `AssertViewUUID(metadata.uuid())` then forEach-applies `Builder.update`. The `Builder` is
  constructed with `base = null` (offset 34 `aconst_null`), and the `update(AddSchema)` arm only adds
  `AssertLastAssignedFieldId` when `base != null` (offset 11 `ifnull 33`) — so it NO-OPS for views.
  There is NO `AddViewVersion`/`SetCurrentViewVersion` arm at all. So a view replace commit carries
  `[AssertViewUUID]` regardless of the update list. Pin it at the ActionCommit source (`take_requirements`
  == `[UuidMatch]`), not via concurrency — the retry machinery masks over-strict requirements (V1 lesson).
- **`ViewVersionReplace.internalApply` assigns `versionId = max(versionId)+1` then defers REUSE to the
  builder — the op's `+1` is a hint, not the final id.** 1.10.0 bytecode matches the /tmp source exactly:
  `Preconditions.checkState` on representations/schema/defaultNamespace, build version with
  `versionId(maxVersionId+1)` + `timestampMillis(now)` + `putAllSummary(EnvironmentContext.get())`, then
  `ViewMetadata.buildFrom(base).setCurrentVersion(newVersion, schema).build()` — and `setCurrentVersion`
  runs `reuseOrCreateNewViewVersionId`, so committing identical representations REUSES the existing id
  (version count unchanged). The Rust `ReplaceViewVersionAction::to_commit` mirrors this: feed the
  `max+1` candidate through `into_builder().set_current_version(...)` and let the builder dedup.
- **The view JSON wire format DIVERGES from Rust serde in FIELD ORDER but not field SET — round-trip is
  field-set-exact, NOT byte-exact.** Java `ViewMetadataParser.toJson` writes `view-uuid, format-version,
  location, [properties only if non-empty], schemas, current-version-id, versions, version-log`; Java
  `ViewVersionParser` writes `version-id, timestamp-ms, schema-id, summary, [default-catalog only if
  non-null], default-namespace, representations`. Rust's `_serde::ViewMetadataV1` orders
  `format-version` FIRST and `schemas` LAST, and always emits `properties` (Java omits when empty). Both
  parse each other's field set (serde is order-insensitive). DO pin the field SET (parse a Java-ordered
  doc + round-trip), and flag the order/empty-properties divergence — byte-level view interop is a
  next-wave item, not a U1 claim.
- **Mirror the table CRUD idiom EXACTLY for views in MemoryCatalog: a SEPARATE `view_metadata_locations`
  map in `NamespaceState`, so a view and a table of the same name in one namespace do not collide.** Java
  keeps views and tables in distinct catalog spaces; the Rust MemoryCatalog already keyed tables by name
  in `table_metadata_locations`, so views need their own parallel map (+ `insert_new_view`/
  `remove_existing_view`/`commit_view_update`/`get_existing_view_location` + `ViewNotFound`/
  `ViewAlreadyExists` ErrorKinds for Java `NoSuchViewException`/`AlreadyExistsException` parity).

#### U1 REVIEWER corrections (2026-06-12, wt-views) — adversarial pass against 1.10.0 bytecode
- **CORRECTS the "separate map ⇒ do not collide" lesson above: separate maps are right, but Java
  ENFORCES a shared name space across them — a view CANNOT be created/renamed onto a TABLE's name,
  and vice versa.** The builder shipped views+tables coexisting silently (probe: `create_view` over a
  table AND `create_table` over a view both SUCCEEDED). Java 1.10.0 `InMemoryViewOperations.doCommit`
  (offset 85-117) throws `AlreadyExistsException("Table with same name already exists")` when
  `tables.containsKey(ident)`; the view-catalog's table builder throws `"View with same name already
  exists"` (`viewExists` check); `renameView`/`renameTable` both cross-check the OTHER map at the
  destination. FIX (this pass): added a `tables.contains_key`/`views.contains_key` cross-guard to
  `insert_new_table` + `insert_new_view` in `namespace_state.rs` (symmetric — both helpers live in the
  modified file; the table-direction guard only rejects a previously-silent corruption, no legit test
  hits it). DO add the cross-type collision guard to BOTH `insert_new_*` helpers whenever a new
  catalog object class shares the table name space.
- **The in-tree MemoryCatalog `update_view` (and the pre-existing `update_table` it mirrors) has NO
  base-location CAS — a stale concurrent commit silently lands (last-write-win), diverging from Java
  `InMemoryViewOperations.doCommit`'s `views.compute` location-equality check.** Probe: two
  `ReplaceViewVersionAction` commits built from the SAME base, applied sequentially — the second
  (stale) one SUCCEEDED (versions 1→2→3, location 00001→00002) instead of failing. Java's `doCommit`
  lambda throws `CommitFailedException("...because it has been concurrently modified to %s")` when the
  stored location ≠ the expected base location. ROOT CAUSE: `ViewCommit::apply` re-loads the CURRENT
  view inside the lock and the only requirement is `AssertViewUUID` — but the UUID is INVARIANT across
  replaces, so it never detects staleness. This is consistent with the pre-existing `update_table`
  path (same gap), so it is NOT a view-specific regression and was NOT fixed here (fixing only the view
  side would diverge from the table side; belongs in a dedicated optimistic-concurrency-parity
  increment touching both). DO NOT claim "catalog view CRUD at parity" without the concurrency
  dimension — `[AssertViewUUID]` alone is correct for the REST commit protocol (server does the CAS)
  but insufficient for an in-process catalog with no location CAS.
- **VERIFIED interop-readiness (escalation NOT needed): Java `ViewMetadataParser.fromJson` reads
  Rust's emitted wire format TODAY.** 1.10.0 bytecode: all reads are by-key (`JsonUtil.getString`/
  `getInt`/`get` → order-insensitive Jackson), and `properties` is read under `if (node.has(...))`
  (offset 46-67) — so Rust always-emitting `"properties":{}` is tolerated, and field ORDER is
  irrelevant. The reverse holds too: Rust `_serde::ViewMetadataV1.properties` is `Option<..>` +
  `unwrap_or_default()`, so Rust tolerates Java's omit-when-empty. The wire divergence is COSMETIC
  (field order + always-emit-empty-properties), not blocking — byte-level view interop stays a
  next-wave item but bidirectional field-set reads work now. Mutation pinned: dropping the required
  `default-namespace` field makes the parse fail (`ViewVersionV1.default_namespace` is non-Option).

### 2026-06-12 (Wave-5 Group Y / Y2 — ComputeTableStats action, REVIEWER Opus, wt-tstats)
- **STOP-GRADE: Java's NDV pipeline builds an ALPHA-family sketch; the Y1 port is QUICKSELECT — do NOT
  dismiss the puffin spec's "Alpha" as a doc nit.** *Why:* three independent sources agree —
  `format/puffin-spec.md` ("constructing **Alpha family sketch**"), Spark `ThetaSketchAgg
  .createAggregationBuffer` MAIN source (`UpdateSketch.builder.setFamily(Family.ALPHA).build()`, all of
  spark v3.5/4.0/4.1 + the class doc), and `datasketches-java-3.3.0` bytecode (`UpdateSketchBuilder`'s
  DEFAULT family is QuickSelect, so the explicit `.setFamily(ALPHA)` is load-bearing — it overrides the
  default the Y1 port matches). Consequence (Java probe, lgK12/seed9001): **exact mode (≲ a few thousand
  distinct, theta==MAX) Alpha and QuickSelect are byte-identical + same ndv; estimation mode (≳7k) they
  DIVERGE** (n=1M → Alpha 1004032 vs QS 1002714, different retained set + bytes — Alpha switches to a
  sampling estimate `nominal*MAX/theta`). A test suite that feeds only ≤6 distinct values per column
  CANNOT see this — the divergence is SILENT. Lesson: when the headline is a sketch-FAMILY question, the
  crown-jewel hand-count test is necessary but NOT sufficient — add an explicit estimation-mode value pin
  (large distinct input) that documents the QuickSelect↔Alpha gap, so the next agent porting Alpha has a
  visible, citable pin (flip it to assert-equal-with-Alpha when the family lands). DON'T fix Y1's crate
  from Y2 (committed byte surface) — STOP-report the family verdict in the module doc + matrix cell.
- **The `footer_size <= file_size` invariant is too weak — a `footer_size == total` mutation SURVIVES it.**
  *Why:* a real Puffin stats file always has the leading MAGIC + the blob payloads BEFORE the footer, so
  `footer < total` STRICTLY; Java readers locate the blob region as `fileSize - footerSize`, so a footer
  that equals/exceeds the data is corrupt. Pin `<` not `<=`. (Verified the seam IS correct otherwise:
  Rust `footer_size()` == Java `PuffinWriter.footerSize()` byte-for-byte = MAGIC(4) + payload +
  FOOTER_STRUCT(12) = payload+16, the exact bytes `write_footer` appends.)
- **The 4 byte-form pins (long/date/string/decimal) skip the dangerous edges — add them.** *Why:*
  negative/zero decimals (Java `BigInteger.toByteArray` minimal two's-complement: -1→`ff`, 0→`00`,
  -300→`fe d4`), uuid (16B BE), float/double NaN + -0.0 (Java `putFloat`/`putDouble` write RAW bits, no
  NaN canonicalization — Rust `to_le_bytes` matches), and boolean (1 byte) all have distinct failure modes
  a long/string pin can't catch. `Datum::to_bytes` matches Java for ALL of them (verified vs a Java probe).
- **Spark agg/UDAF source IS available at `/tmp/iceberg-java-ref/spark/v{3.5,4.0,4.1}` even though no
  iceberg-spark JAR is in `~/.m2`.** *Why:* the orchestrator believed datasketches came in via a spark jar
  in m2 — it did NOT (only `datasketches-java`/`-memory` jars are there). For spark-action provenance read
  the MAIN source tree, not m2 bytecode; for the FAMILY question that source (`setFamily(ALPHA)`) is the
  authoritative oracle, cross-checked against the datasketches jar's builder-default bytecode.

#### O1 REVIEWER corrections (2026-06-12, wt-core6) — adversarial pass against 1.10.0 bytecode
- **The Java VIEW base-location capture is `currentMetadataLocation()`, NOT `base.metadataFileLocation()`
  — the table path differs from the view path, and the builder's report blurred them (cosmetic, not a
  behavioral bug).** 1.10.0 bytecode: `InMemoryTableOperations.doCommit` offset 23-24 captures
  `baseLocation = base.metadataFileLocation()` (the BASE arg's location); `InMemoryViewOperations.doCommit`
  offset 14-15 captures `baseLocation = this.currentMetadataLocation()` (the OPS object's current location)
  when `base != null`. For the in-process case these are equal (the commit's base IS `current()`, loaded
  from the same map slot), so the Rust port threading `View::metadata_location` matches functionally. DO
  cite the exact capture expression per seam — the table/view asymmetry is real bytecode, even though it
  collapses to the same value here. Both lambdas' `CommitFailedException` arg order is bytecode-confirmed
  `(identifier, baseLocation, newLocation, storedLocation)` (table offsets 73-91, view offsets 72-91).
- **VERIFIED the None-bypass hole is closed AND the mutation battery is complete (5 seams pinned).**
  Probe (explicit `base_metadata_location=None` against a stored table via `update_table`): does NOT
  silently land — raises `CatalogCommitConflicts` (`None != Some(stored)`), Java-conservative. Mutation
  battery run for real: (a) `if true` CAS → exactly the 2 stale-conflict tests fail, happy/refresh stay
  green; (b) knock out population at `ReplaceViewVersionAction::to_commit` → 4 fail (incl. the carry pin);
  (b') `UpdateViewPropertiesAction::to_commit` → 2 fail (incl. carry pin); (c) transaction `do_commit`
  site → 377 fail (central commit path); (c') `register_partition_stats_file` site → 3 fail. NO population
  seam survived its knockout — none unpinned. DO run every population site's knockout, not just the CAS:
  an unpopulated seam is a silent door even with a correct CAS.
- **MATRIX-DISCIPLINE FIX (the builder's miss): the GAP_MATRIX row 107 still said "`update_view` has no
  base-location CAS … needs a dedicated concurrency-parity increment" AFTER this increment closed it.**
  The de-triplication rule puts STATUS only in the matrix, so leaving the "gap open / needs increment"
  clause was a stale-status defect. FIX (this pass): rewrote the KNOWN-GAP clause to the **O1**-landed CAS
  narrative + the SQL weaker-CAS follow-up, and corrected the stale "MemoryCatalog's no-CAS posture" phrase
  in the U2 sub-cell; pipe-audit re-run (row still 5 pipes). DO flip/refresh the matrix cell in the SAME
  increment that closes the gap it describes — a builder that lands the fix but not the cell leaves the
  matrix lying.
- **CONFIRMED the SQL-weaker-CAS finding is real (read the path, not just the report).** `crates/catalog/
  sql/src/catalog.rs:965-1002` `update_table` CAS's `... AND metadata_location = ?` against
  `current_metadata_location = self.load_table(...).metadata_location_result()` — the location it loads
  INSIDE its own `update_table` (line 967-968), NOT `commit.base_metadata_location()` (which is now
  available but unread). Guards only the load→UPDATE TOCTOU window; a strictly-sequential stale commit
  passes. Correctly flagged as a follow-up (NOT half-fixed). The retry classification is also confirmed:
  `transaction/mod.rs:361` `.when(|e| e.retryable())` gates the `backon` loop; the helper raises
  `CatalogCommitConflicts.with_retryable(true)`, and the mock tests (`...retryable(Some(2),3)` + `.times(3)`)
  prove the loop iterates on exactly that error. VERDICT: SHIP.

### 2026-06-12 (Wave-6 O1 — optimistic-concurrency parity for MemoryCatalog, BUILDER Opus, wt-core6)
- **Java's in-memory CAS compares the STORED location against the commit's BASE location, NOT the
  catalog's own freshly-loaded location — and the SQL catalog in THIS fork does the WEAKER thing.**
  1.10.0 bytecode `InMemory{Table,View}Operations.doCommit`: capture `baseLocation = base == null ?
  null : base.metadataFileLocation()` from the `base` ARGUMENT, then `tables.compute(id, (k, stored)
  -> Objects.equal(stored, baseLocation) ? newLoc : throw)`. The `base` is what the commit was built
  from, threaded in as a method arg. The Rust `TableCommit`/`ViewCommit` did NOT carry it, so the fix
  added `base_metadata_location: Option<String>` to BOTH and a `check_no_concurrent_modification`
  helper in MemoryCatalog. NOTE the divergence I found and did NOT fix (out of O1 scope): the SQL
  catalog (`crates/catalog/sql`) CAS's against `current_metadata_location` = the location it loads
  INSIDE its own `update_view`/`update_table`, NOT the commit's base — so it only guards the TOCTOU
  window between its own load and its own UPDATE; a strictly-sequential stale commit (load, winner
  lands, then loser's `update_*` re-loads the advanced location and CAS's it against itself) would
  PASS. The SQL `test_*_concurrent_commits_*` tests use a `Barrier` + true threads to hit the window,
  so they pass, but the posture is weaker than Java. The Java-faithful SQL fix is to CAS against
  `commit.base_metadata_location()` too — a follow-up increment.
- **`Transaction::do_commit` reloads the base RIGHT BEFORE building the `TableCommit`, so two
  sequential transactions from the same base BOTH succeed via refresh — the conflict→retry loop is
  hard to reach in-process.** The location-CAS in MemoryCatalog therefore fires for (a) a `TableCommit`
  / `ViewCommit` built DIRECTLY from a stale base (the `register_*` maintenance paths, the view
  actions held across a concurrent commit) and (b) the tiny window of a true external race inside
  `do_commit`. The deterministic test for the conflict is to build the commit directly (bypassing the
  transaction reload) — `update_table_properties` emits an EMPTY requirement set, so a property-only
  `TableCommit` has NO requirement that could catch staleness and ONLY the location-CAS can fire
  (the prompt's "only the CAS" case). Append-style staleness is separately caught FIRST by
  `RefSnapshotIdMatch` (snapshot.rs emits it when `current_snapshot_id` is `Some`); the CAS is the
  net for metadata-only updates and for views (`[AssertViewUUID]` is invariant across replaces).
- **A new `TypedBuilder` field on `TableCommit` needs `#[builder(default)]` or every existing
  `.build()` call site breaks; `ViewCommit` is a plain struct literal so its two `to_commit` sites +
  the REST/SQL paths must set the field explicitly (they don't construct it — REST only calls
  `take_requirements`/`take_updates`, so the wire is unaffected and REST semantics are untouched).**
  The view actions had to grow a `base_metadata_location` field threaded from the source `View`'s
  `metadata_location`, which meant updating the `View::replace_version`/`update_properties` ctors and
  6 in-crate test call sites of the now-3-arg `*Action::new`. DO check for direct constructor calls
  in unit tests when widening a private constructor signature.

### 2026-06-12 (Wave-6 O2 — partition-stats incremental + exotic value types, BUILDER Opus, wt-core6)
- **The 1.10.0 jar's `PartitionStatsHandler` ALREADY has the full incremental surface — the `collectStatsForManifest`
  signature carries `boolean incremental`, and the X1/X2 port had silently dropped that 4th param.** Bytecode
  (iceberg-core-1.10.0.jar): `collectStatsForManifest(Table, ManifestFile, StructType, boolean)`,
  `computeStats(Table, List, boolean)`, plus `computeAndMergeStatsIncremental`/`latestStatsFile`/`computeStatsDiff`
  and `PartitionStats.deletedEntryForIncrementalCompute`. The standing journal lesson "MAIN source is
  post-1.10.0-refactored on PartitionStatsHandler" is REAL — but here MAIN and 1.10.0 AGREE on the incremental
  surface; the X1 port had just scoped it out, not wrongly derived it. DO `javap -p` the WHOLE class to recover the
  method surface before assuming a feature is absent — a dropped boolean param is invisible in a source read.
- **The incremental entry dispatch is bytecode-exact and has TWO subtle gates a fast-append fixture CANNOT
  exercise** (offsets 197-259 in `collectStatsForManifest`): (1) a LIVE entry contributes ONLY if
  `incremental && status==ADDED` — a carried-forward EXISTING entry is SKIPPED (already in the seed); (2) a
  tombstone calls `deletedEntryForIncrementalCompute` (the `lsub`/`isub` SUBTRACT mirror of `liveEntry`), not
  `deletedEntry`. The diff (`computeStatsDiff`) selects per-range-snapshot ONLY the manifests it added
  (`added_snapshot_id == snapshot.id`). A FAST-APPEND S2 adds a manifest containing only ADDED entries, so the
  ADDED-only filter NEVER fires — a `if true` mutation over the filter PASSED every fast-append test SILENTLY.
  The fixture that makes it load-bearing is a MERGE-APPEND with `commit.manifest.min-count-to-merge=2`: S2's
  merged manifest re-stamps S1's file as EXISTING, so the filter MUST skip it (else x=1 double-counts: 11≠8).
  DO build a merge-append (not fast-append) fixture whenever a test must see an EXISTING entry in a
  newly-added manifest; a fast-append cannot produce one.
- **Java's corrupt-base fallback is scoped to the BASE READ ONLY (`catch(Exception)` over offsets 11-99), NOT the
  diff compute.** `computeAndMergeStatsIncremental` wraps `readPartitionStatsFile(base)` in
  `catch(Exception) → throw InvalidStatsFileException`; the caller (`computeAndWriteStatsFile`) catches THAT and
  falls back to full compute. The diff (`computeStatsDiff`, AFTER the try block) propagates its errors. Ported as:
  `compute_and_merge_stats_incremental` returns `Ok(None)` on ANY base-read error (the fallback signal) but `Err`
  on a diff failure. DO NOT collapse "corrupt base" and "diff IO error" into one error path — they have OPPOSITE
  Java semantics (fall-back vs hard-fail). The deterministic test registers a base `PartitionStatisticsFile` at a
  NON-EXISTENT path and asserts the write still equals a full recompute.
- **`computeAndWriteStatsFile(table, snapshotId)` SHORT-CIRCUITS when the latest stats file IS already for the
  target snapshot — it returns the existing file UNCHANGED, no rewrite (offset 104-132 "Returning existing
  statistics file").** The X2 port always wrote a fresh file; adding the Java-faithful selection broke the X2
  `test_replace_on_rewrite` test (it called `compute_and_write_stats_file` twice for the same snapshot expecting
  a fresh uuid path — now the 2nd call returns the existing file). The test's REPLACE intent is real but the
  "fresh path on re-write" assertion was X2-specific, not Java. DO re-pin REPLACE semantics by registering a
  DIFFERENT-sized `PartitionStatisticsFile` (the metadata `set_partition_statistics` map keyed by snapshot id),
  and separately assert the case-3 short-circuit returns the existing file.
- **The exotic partition-value byte forms were ALREADY half-built — the production READER
  (`arrow_struct_to_literal`) handles Time64/FixedSizeBinary(16-uuid)/FixedSizeBinary(L-fixed)/LargeBinary, and
  `schema_to_arrow_schema` maps the iceberg types to those Arrow types; only the stats-file WRITER lacked the
  arms.** The loud `FeatureUnsupported` residue was a one-sided gap. The literal storage forms: Time→`Long`(micros),
  Uuid→`UInt128` (write 16 BE bytes via `Uuid::from_u128(v).into_bytes()` == Java `Conversions.toByteBuffer`,
  reader `Uuid::from_bytes` round-trips), Fixed/Binary→`Binary(Vec<u8>)`. DO check the reader + the type mapper
  before assuming an "unsupported type" needs new byte-form derivation — the write arm may be the only missing
  half, and the round-trip through the existing reader IS the byte-form verification. `FixedSizeBinaryArray::
  try_from_sparse_iter_with_size(iter, size)` is the null-tolerant + width-validating constructor (rejects a
  wrong-width fixed value loudly — a free correctness guard).

#### O2 REVIEWER corrections (2026-06-12, wt-core6) — adversarial pass against 1.10.0 bytecode
- **The whole O2 incremental contract is bytecode-EXACT — re-derived independently from the 1.10.0 jar
  (`javap -p -c`), every clause CONFIRMED.** `computeAndWriteStatsFile(Table,long)` offsets:
  null-base→full (69), `base.snapshotId==target`→return-existing `areturn`(104-132), older-base→incremental
  with the exception table `from 133 to 144 target 147 type InvalidStatsFileException` scoping ONLY the
  merge call (so a DIFF error propagates, only the BASE read falls back — `computeAndMergeStatsIncremental`'s
  inner `from 11 to 99 target 102 type Exception` wraps `readPartitionStatsFile`+`forEach`, the diff at
  offset 114+ is OUTSIDE it). `collectStatsForManifest` offset 207-221: LIVE contributes iff
  `incremental==false || status==ADDED` (`if_acmpne 259` skips carried EXISTING); 236-246: tombstone→
  `deletedEntryForIncrementalCompute` (incremental) else `deletedEntry`. `deletedEntryForIncrementalCompute`
  is `liveEntry` with `ladd→lsub`/`iadd→isub` AND IDENTICAL last-updated tail (offset 210-230 in both):
  counters SUBTRACT, last-updated-at/snapshot-id is **MAX via `updateSnapshotInfo` (strict `<`), NOT
  subtracted/overwritten** — the deleted-entry-attribution question the brief flagged. `ancestorsBetween`
  truncation lambda returns null on reaching `from` ⇒ range is `(from, to]`; `ancestorsOf(long)` is
  INCLUSIVE of the start (resolves the start snapshot then walks parents) ⇒ `latestStatsFile` includes the
  target snapshot. The Rust port matches every one.
- **COVERAGE GAP FOUND + FIXED: the builder pinned incremental==full on 2 APPEND shapes (append-only +
  add-a-delete-FILE), but NEITHER exercises the SUBTRACT arm end-to-end — a `delete_files` op is what
  produces a DELETED tombstone in an S2-added manifest that fires `deleted_entry_for_incremental_compute`.**
  Probe-confirmed: `delete_files().delete_file(a.parquet)` re-stamps the removed file as `status=Deleted`
  (added_by_s2=true) and the surviving file as `status=Existing`, and incremental==full==(5rec,1file) only
  when the subtract fires. The unit test `test_deleted_entry_..._subtracts_each_cell_back_to_zero` pins the
  arithmetic but NOT that the arm fires in the real pipeline. FIX (this pass): added
  `test_incremental_equals_full_recompute_with_delete_subtracting_base_file` (the strongest subtract-arm
  pin) — caught by the mutation that no-ops the data subtract AND the include-all-manifests diff mutation.
  DO add a `delete_files`-driven equivalence shape for any incremental-stats subtract path; an append-with-
  delete-FILE shape adds a delete file (a LIVE entry) and never produces the DELETED tombstone the subtract
  arm needs.
- **Mutation battery (6 for real, all caught; reverted): (a) no-op data subtract → unit + new e2e subtract
  test fail; (b) `if true` over ADDED-only filter → merge-append + subtract tests fail (double-count);
  (c) corrupt-base `Err(e)=>return Err` (no fallback) → corrupt-base test fails; (d) `if false` over case-3
  guard → return-existing + rewritten replace test fail; (e) uuid bytes `.reverse()` (LE not BE) → uuid
  round-trip fails BECAUSE the production reader (`arrow_struct_to_literal` → `Uuid::from_bytes`, BE) is
  NOT mutated — proving the round-trip reads through the production path, not a test-local decode; (f) `if
  true` over the `added_snapshot_id` diff filter → both append equivalence tests fail.** Every seam pinned.
- **The 5 failing iceberg DOCTESTS (`src/lib.rs:24`, `writer/mod.rs:42/117/257/321`) are PRE-EXISTING and
  UNTOUCHED by O2 — verified by `git stash` to the clean O1 commit d3fba2bf: identical 5-failure set.** Root
  cause is environmental (`error: default runtime flavor is multi_thread, but rt-multi-thread feature is
  disabled`), not code — `cargo test -p iceberg --lib` (the builder's gate) excludes doctests, and the O2
  diff touches no doctest. DO run the doctest set on the base commit before attributing a doctest failure to
  the increment under review.

### 2026-06-12 (Wave-5 Group Y / Y3 — Alpha-family update sketch, BUILDER Opus, wt-tstats)
- **THE ndv-source ruling: Iceberg's `ndv` reads the COMPACT sketch's `getEstimate`, NOT the Alpha
  update sketch's. The two genuinely DIFFER in estimation mode — derive which object from
  `NDVSketchUtil.toBlob`, never assume "the sketch's estimate".** *Why (decisive — spark v3.5/4.0/4.1
  `NDVSketchUtil.java` identical):* `Sketch sketch = CompactSketch.wrap(Memory.wrap(bytes)); ... ndv =
  String.valueOf((long) sketch.getEstimate())`. `sketch` is the COMPACT sketch reparsed from the
  serialized bytes — its `getEstimate` is the family-COMPACT STANDARD estimator `retained*(2^63/theta)`.
  The live Alpha UPDATE sketch's `getEstimate` (`HeapAlphaSketch.getEstimate` bytecode) is family-aware:
  `theta>split1` → standard, `theta<=split1` → SAMPLING `nominal*(2^63/theta)`. Java probe (lgK12/seed9001,
  n=1M): UPDATE sampling estimate = 1002319 but COMPACT estimate = 1004032 — and **1004032 is the prompt's
  pinned ndv**. So the action must do `CompactThetaSketch::deserialize(&payload).estimate()`, exactly
  Java's `CompactSketch.wrap(bytes).getEstimate()`. Pinning BOTH values (the compact one as the ndv, the
  update one as "NOT the ndv") makes the object-selection load-bearing. The prompt framed `nominal*MAX/theta`
  as "the Alpha estimator the Y2 probe saw" — true of the UPDATE sketch, but the ndv uses the COMPACT one;
  a builder who wired `alpha.estimate()` would emit 1002319 and silently diverge from every engine.
- **`HeapAlphaSketch.compact()` is family-COMPACT, not family-ALPHA — the on-disk form REUSES the
  QuickSelect serializer verbatim; only the UPDATE-side retention/theta differs.** *Why:* `toByteArray()`
  on the live Alpha sketch writes a family-ALPHA preamble, but Iceberg serializes via `UpdateSketch.compact()`
  → `componentsToCompact(thetaLong, getRetainedEntries(true), seedHash, isEmpty, ..., cache)` → `compactCache`
  keeps cache entries `0<h<theta`, `Arrays.sort` ascending, `loadCompactMemory` writes the family-COMPACT (id 3)
  preamble. So `AlphaSketch::serialize_compact()` = the Y1 `serialize_compact_from_parts(is_empty, theta,
  sorted_below_theta_hashes, seed)` UNCHANGED — one path, no fork. The reused retained count is
  `getRetainedEntries(true)` (the DIRTY-aware `countPart` = the below-theta cache count), NOT the raw
  `curCount_` (which over-counts dirty stale slots). Make the shared helpers `pub(crate)` rather than
  duplicating the serializer.
- **The Alpha dirty-phase insert (`enhancedHashInsert`) reuses stale (≥theta) slots WITHOUT a count
  bump and does NOT rebuild every insert — port it faithfully or the retained set drifts by an element.**
  *Why:* once `theta<=split1` the table accumulates above-theta entries; a new insert probes, and on the
  FIRST stale slot it reuses it in place (`InsertedCountNotIncremented`), decays theta, sets dirty; only a
  truly-empty slot bumps `curCount` and may trigger `rebuildDirty` (a same-size purge; if nothing purged,
  grow by 1). A naive "rebuild on every dirty insert" or "always land in an empty slot" port produced a
  retained SET that differed from Java by ONE borderline hash near theta — invisible to retained-count and
  theta pins (both matched) but caught by the BYTE-EXACT fixture. Lesson: for a stateful Java sketch, the
  byte-exact estimation fixture is the only pin that catches a single-element set drift; pin retained+theta
  AND the full bytes.
- **A byte-exact `*_HEX` const transcribed by hand into Rust is error-prone — generate it, then REPLACE
  the const programmatically from the verified-equal Rust/Java output, never retype a 8000-char hex.**
  *Why:* my first paste of the lgK9/520 fixture had 2 extra hex chars at byte 1880; the retained SET and
  theta were byte-identical (proven by dumping both sets — zero diff), so the bug was purely a transcription
  typo in the const, not the algorithm. A `python3 re.sub` replacing the const with the Java-generated hex
  (after confirming the live Rust `serialize_compact()` == the Java hex) fixed it in one shot. DO diff the
  retained SETS first when an estimation fixture fails — if the sets match, the bug is in the const, not the
  sketch.
- **`setHashTableThreshold(lgNom, lgArr)` uses the 0.5 fraction when `lgArr <= lgNom` (NOT 0.9375).** *Why:*
  the initial Alpha table at lgK12 has lgArr=7 (`startingSubMultiple(13,3,5)=7`), and 7<=12 ⇒ threshold =
  `floor(0.5*2^7)=64`, not 120. I twice wrongly asserted the 0.9375 branch; it only applies once the table
  has grown PAST nominal. Read the `if_icmpgt` direction in the bytecode, don't assume the resize-phase fraction.

### 2026-06-12 (I1 — theta-blob puffin interop, BUILDER Fable/Sonnet, wt-interop6)
- **`BlobMetadata` fields in the Rust iceberg crate are PRIVATE — use accessor methods, not field
  access.** `blob_type()`, `fields()`, `snapshot_id()`, `sequence_number()`, `properties()` are
  the correct calls; `blob_metadata.r#type` / `.fields` / etc. will not compile.
  *Why:* the struct fields are `pub(crate)` only. The `Blob` struct returned by `reader.blob()`
  has the same accessor names and IS public.
- **`FileIO::from_path(...)` does not exist — use `FileIO::new_with_fs()` for local filesystem
  access in tests.** The correct API for a no-config local FileIO is `FileIO::new_with_fs()`;
  the builder pattern is `FileIOBuilder::new(Arc::new(LocalFsStorageFactory)).build()`.
  *Why:* `from_path` is a pattern from other Iceberg crate versions; the fork's `FileIO` does not
  have it. Always `grep -n "pub fn"` the `FileIO` impl before assuming a constructor exists.
- **Puffin file structure (Iceberg spec): footer layout is `[data-blobs][footer-magic(4)][footer-json(N)][footer-struct: payload_len(4 LE u32)|flags(4)|trailing-magic(4)]`.** Blob offsets in the footer JSON are ABSOLUTE from the file start (blob[0].offset = 4, right after the leading magic). To read the footer from Python: `payload_len = struct.unpack('<I', data[file_len-12:file_len-8])[0]`; `footer_json_start = file_len - 12 - payload_len - 4` (skip the footer's own leading magic). DO NOT search for the sketch preamble bytes by pattern — parse the footer JSON for blob offsets and corrupt those bytes directly.
  *Why:* a pattern search for compact-sketch family byte `0x03` in the raw file hit a byte in the
  Puffin wire framing, not the sketch payload, so the corrupt byte had no effect on Java's
  `CompactSketch.wrap()`. The footer-parse approach is deterministic and works for any blob count
  or order.
- **When the sabotage battery's `|| true` catch-all pattern is used on a Rust test failure, verify
  the check logic doesn't produce false-greens.** The partition-stats template's `7c` pattern
  checks `grep -q "^test.*ok$"` combined with `grep -qiE "error|panicked|FAILED"` — a truncated
  puffin that panics at the file-metadata read will not emit `^test.*ok$` at all, so the pattern
  needs a second pass. A simpler invariant: the truncated file MUST NOT allow the Rust test to
  exit 0 normally. For Rust `cargo test`, the process exit code is non-zero on FAILED/error, so
  `|| true` plus absence of "ok" is sufficient.

#### I1 REVIEWER corrections (2026-06-12, wt-interop6) — adversarial pass
- **A theta sketch's `getEstimate()` is INSENSITIVE to a single-byte flip in the hash-entry region —
  it depends ONLY on `theta` and the retained ENTRY COUNT, not on the entry values. So a "corrupt a
  payload data byte" sabotage is a SILENT NO-OP on the ndv-vs-estimate cross-check.** *Why (probed):*
  the builder's 6b zeroed blob0's first 8 bytes — that corrupts the compact-sketch PREAMBLE
  (preLongs/serVer/family), which makes `CompactSketch.wrap()` THROW, so 6b "passed" only as a PARSE
  CRASH (the same failure class as a truncation), NOT via the estimate cross-check the increment's
  headline depends on. I probed flipping a byte deep inside the sorted-hash region (offset+1000): the
  file parsed AND `getEstimate()` returned the UNCHANGED 1004032 — the verify PASSED on a corrupted
  artifact. The genuinely SEMANTIC mutation is to corrupt `theta` itself (the LE long at compact-
  estimation payload offset +16): the preamble stays valid so the file PARSES, but `getEstimate() =
  retained * 2^63 / theta` changes (halving theta → estimate doubles 1004032→2008064), which ONLY the
  cross-check catches. FIX (this pass): rewrote 6b to halve `theta` via the footer-parsed SOURCE
  offset, with an estimation-mode precondition guard (exact-mode theta==MAX is inert) and a belt that
  asserts the FAIL came from the `getEstimate() as long expected=` cross-check line (not a parse
  crash) — so a future degeneration of the semantic mutation into a structural one fails closed. DO,
  for any statistic-bearing blob, pin a sabotage that mutates the STATISTIC (theta/count) while
  keeping the container parseable — a payload-data-byte flip proves nothing about the estimator.
- **A `SKIP` branch in a sabotage step is a false-green: it lets the chain continue without the
  corruption ever landing.** *Why:* the builder's 6b printed "6b SKIP" and continued on a magic-detect
  failure (exit 42). A sabotage that cannot be applied has proven nothing and MUST hard-fail. FIX:
  converted the 6b skip-exits (42 framing / 43 not-estimation-mode) into `exit 1` with a restore.
- **CONFIRMED the crown-jewel family pin is load-bearing both ways (mutation-tested).** Swapping the
  Java oracle's `setFamily(Family.ALPHA)` → `QUICKSELECT` in `buildSketchPayload` makes the Java D2
  generate THROW immediately at its own `ESTIMATION_NDV_PIN` sanity check (`estimation ndv pin check
  FAILED: expected 1004032 got 1002714`) — the n=1M estimation blob is genuinely large-n, the family
  is unpinned-detecting, and the chain fails closed. The Rust D2 estimation pin (`expected_ndv ==
  ESTIMATION_NDV_PIN` for field_id=3) is an independent second guard. Reverted after.
- **Integer-exactness CONFIRMED both directions:** Rust compares `sketch.estimate() as i64` via
  `assert_eq!` (no tolerance/abs/epsilon anywhere in `interop_theta.rs`); Java compares `(long)
  compact.getEstimate()` via `!=`. The Puffin footer-layout lesson (trailing magic, payload_len u32
  LE, absolute blob offsets) is accurate against `puffin/writer.rs::write_footer`.
- **COVERAGE NOTE (not fixed — named):** the interop layer proves the blob byte round-trip but does
  NOT assert the stats file's REGISTRATION entry (snapshot-id/statistics-path) on the committed table
  metadata — the Java verify reads the loose copied `rust_stats.puffin`, not `final.metadata.json`.
  Registration IS unit-covered (`compute_table_stats.rs` re-parses the committed `StatisticsFile`), so
  this is a minor interop-completeness gap, not a correctness hole. DO consider a `statisticsFiles()`
  read on the Java side in a follow-up to close the registration dimension at the interop level.

### 2026-06-12 (I2 — view metadata interop, BUILDER Fable/Sonnet, wt-interop6)
- **Java's `View` interface does NOT expose `operations()` — cast to `BaseView` to access the
  committed `ViewMetadata`.** `InMemoryCatalog.loadView(ident)` returns `View`; `View` has no
  `operations()`. Cast: `((org.apache.iceberg.view.BaseView) loadedView).operations().current()`.
  Same pattern applies after `buildView(...).create()` or `.replace()` — store the returned `View`
  reference and cast. *Why:* `BaseView` is the concrete abstract class that wires the operations
  handle; the interface intentionally hides it. `grep 'extends BaseView'` in `iceberg-core` to find
  other concrete view impls that may need the same cast.
- **Java's `reuseOrCreateNewViewVersionId` deduplicates VIEW VERSIONS with identical SQL — the
  two SQL strings MUST be DIFFERENT (not just syntactically but character-exact) for version count
  to grow to 2.** *Why:* `BaseViewVersionReplace` (bytecode-verified) compares representations via
  `sameViewVersion` — if the replacement representations are character-identical to the current
  version's, it reuses the existing version-id and the version-log count stays at 1. Use
  `SQL_V1 = "... WHERE id > 0"` and `SQL_V2 = "... WHERE id > 100"` — any string difference
  works, but the constants must be agreed between Java oracle and Rust test (anti-circular).
- **View metadata wire format FIELD-ORDER DIVERGENCE is tolerated at parse time on BOTH sides;
  do NOT attempt byte-exact equality for view metadata JSON.** Java `ViewMetadataParser.toJson`
  writes `view-uuid` first; Rust's `_serde::ViewMetadataV1` writes `format-version` first. Java
  omits `"properties"` when empty; Rust always emits `"properties":{}`. Both serde parsers
  (Jackson and serde_json) are key-order-insensitive; Rust's `properties` field is
  `Option<HashMap<...>>` deserialized with `unwrap_or_default()`. Pin the FIELD SET, not bytes.
  Nail this with a dedicated `test_view_tolerance_controls` that feeds a Java-ordered no-properties
  JSON into Rust's `read_from` and confirms the parse succeeds + properties map is empty.
- **For D2 sabotage (Rust reads Java-written metadata), verify via the RUST TEST's exit code + the
  expected stdout pattern, NOT the Java oracle.** The D2 test (`test_view_d2_rust_reads_java`)
  runs via `cargo test --exact --nocapture`; on an injected mismatch it panics with `assert_eq!` 
  and `cargo test` exits non-zero. The sabotage check pattern: `|| true` catch + `grep -q
  "test test_view_d2_rust_reads_java ... ok"` absent → PASS (failure confirmed). Do NOT look for
  `'^FAIL '` patterns on the Rust side — those are Java sentinel patterns.
- **Dropping a REQUIRED field from a version JSON (`default-namespace`) causes Rust `read_from`
  to fail with a deserialization error — no need for an explicit guard.** Rust's
  `_serde::ViewVersionV1.default_namespace` is non-`Option` (direct `NamespaceIdent`), so serde
  returns an error immediately if the field is absent. This makes 6b a clean structural-failure
  sabotage, not a semantic one. Java `ViewVersionParser` likewise throws on a missing
  `default-namespace`. Pin both via the chain script: a modified `java_view_metadata.json` with
  one version's `default-namespace` removed must make Rust's D2 test fail.
- **A dangling `current-version-id` (pointing to a non-existent version) causes Rust
  `ViewMetadata::read_from` to return an error at METADATA BUILD TIME, not at access time.**
  *Why:* the view metadata builder validates that `current_version_id` refers to a version in
  the `versions` map; if the id has no corresponding version, it returns
  `Err(ErrorKind::DataInvalid)`. This is clean fail-closed behavior (no delayed panic). Pin it
  as 6c in the sabotage battery: `current-version-id: 99` with no version 99 in `versions`.
- **`clippy::never_loop` fires on `for repr in iter { irrefutable-pattern; return ... }` — use
  `iter.next()` with a `let Some(...) else { panic! }` guard instead.** The `for`-loop form looks
  right but clippy sees it as "loop body always executes at most once" because the return exits
  before the next iteration. The idiomatic fix: `let Some(repr) = iter.next() else { panic!(...) };`
  followed by the irrefutable destructure. This applies to any helper that extracts the first
  element from a known-non-empty iterator via a pattern.
- **An `irrefutable if let` or irrefutable single-arm `match` on an enum with ONE variant triggers
  both a clippy lint and a compiler warning — use a bare destructure `let Pattern = value;`.**
  *Why:* `if let ViewRepresentation::Sql(r) = repr { ... }` on a value that IS always
  `ViewRepresentation::Sql` compiles but warns. Use `let ViewRepresentation::Sql(r) = repr;`
  directly (a refutable pattern is only needed when the enum has multiple arms).

#### I2 REVIEWER corrections (2026-06-12, wt-interop6) — adversarial kill-list against the comparators
- **A field-by-field interop comparator that checks schema field NAMES but not field TYPE or the
  required flag has a real PROJECTION GAP — corrupting `id` from long→string passed BOTH the Java
  D1 oracle and the Rust D2 test silently.** The kill-list (corrupt one field of the SOURCE
  metadata, re-run the production reader, confirm it fails) found four unpinned fields on EACH
  side: (1) non-current version `timestamp-ms`, (2) version-log entry `timestamp-ms`, (3) schema
  field `type`, (4) schema field `required`. SQL/dialect/field-names/version-log-id-order WERE
  pinned. FIX (this pass): both emitters now write `schema_fields` (name/type/required),
  `version_timestamps`, and `version_log` (id+timestamp); both comparators assert them. Post-fix
  kill-list: all four PINNED on both directions. DO run the corrupt-one-field kill-list against
  EVERY field a "field-by-field" oracle claims — names-only schema checks are the classic gap.
- **The view metadata `type` string is cross-language byte-identical for primitives — Rust
  `Type::Display` and Java `Type.toString()` both emit `long`/`string`/`int`/… so the comparator
  can compare them as plain strings.** (Decimal is `decimal(p,s)` on both; fixed is `fixed(n)`.)
  No normalization needed for the fixture's `long`/`string`.
- **`ViewMetadata::read_from` does NOT call `validate()` directly — validation runs inside
  `TryFrom<ViewMetadataV1>` (`view_metadata.validate()?` at the end of the conversion), which serde
  invokes during `from_slice`.** So a dangling `current-version-id` is rejected at DESERIALIZE
  time (parse error wraps `No version exists with the current version id N`), not at a later
  access. This is why sabotage 6c fails cleanly at the `read_from` `.expect`, and 6d (valid parse,
  wrong SQL value) fails LATER at the `assert_eq!` — distinct failure sites prove distinct
  provenance.
- **Interop test env-dir paths MUST be absolute when invoking `cargo test` by hand — a relative
  `ICEBERG_INTEROP_VIEW_DIR` resolves against the test process CWD (workspace root), not your
  shell CWD, so the `metadata_path.exists()` guard fires and every sabotage "passes" for the WRONG
  reason (missing file, not the injected corruption).** The chain script is correct (it builds an
  absolute `${TMP}` from `${SCRIPT_DIR}`); only ad-hoc reviewer commands hit this. Always echo the
  resolved path and confirm a CLEAN run passes before trusting a sabotage's failure.

### 2026-06-12 (I3 — data-level WAP interop, BUILDER Sonnet)

- **`updateProperties().set(...).commit()` in BOTH Java and Rust does NOT create a snapshot — only
  emits a `SetProperties`/`RemoveProperties` table update with no `AddSnapshot`.** Using it to
  "bump" main in a REPLAY-shape chain means the table's `current-snapshot-id` is UNCHANGED after
  the "bump", so `staged.parent == current` → the cherry-pick takes the FAST-FORWARD path (no new
  snapshot, no `source-snapshot-id`/`published-wap-id` tags). DO use a REAL data fast-append for
  the bump (write an actual parquet file); even one row with a known id (e.g. id=99 category=a
  data="bump") is sufficient and the row becomes part of the expected fixture.
- **S-replay order: stage FIRST while `current = base`, THEN advance main with the bump commit.**
  The REPLAY shape requires `staged.parent ≠ current head at cherry-pick time`. Stage the WAP
  append BEFORE the bump so `staged.parent = base`. After the bump, `current = bump ≠ base =
  staged.parent` → REPLAY guaranteed. If you stage AFTER the bump, `staged.parent = bump = head`
  → FAST-FORWARD → no `source-snapshot-id`/`published-wap-id`. The sequence is: (1) fast-append
  base, (2) `stage_only()` WAP append, (3) verify `current_snapshot_id == base_snapshot_id`,
  (4) fast-append bump. Confirm REPLAY by asserting the staged snapshot is NOT reachable from the
  current ancestry (`staged_id ∉ walk_from(current_snapshot)`).
- **`LocalTableOperations.commit(null, metadata)` always writes `v0.metadata.json` to its metadata
  directory — repeated calls to `verifyRustTable` fail with "File already exists".** Fix: use
  `Files.createTempDirectory(parent, prefix)` per verify call to create a fresh directory per run.
  Rebuild the `TableMetadata` with the temp dir as location (via `TableMetadata.buildFrom(meta)
  .discardChanges().setLocation(tempDir.toString()).build()`), then seed `LocalTableOperations`
  with the temp dir. Data files referenced by the manifests use absolute paths so they resolve to
  the original parquet files regardless of the temp dir location.
- **`TableMetadata.Builder.withLocation()` does NOT exist in iceberg-core 1.10.0; the correct
  method is `setLocation(String)`.** Always verify Java API names by running `javap` on the target
  class in `~/.m2` before writing oracle code that calls an API method.
- **Semantic sabotage for WAP interop must target the WAP-ID chain, not file paths or partition
  directories.** Swapping file paths or directory names in manifests (even in-place binary edits)
  does not corrupt the `category` column values stored in the parquet data — IcebergGenerics reads
  the actual Arrow column values from the parquet files, which are immutable. The correct semantic
  sabotage: corrupt the staged snapshot's `wap.id` in the metadata JSON (change `"w1"` →
  `"w1-CORRUPTED"`). The metadata still parses, the staged snapshot is still present, Java
  cherry-picks it and produces a snapshot with `published-wap-id="w1-CORRUPTED"` → the
  `published-wap-id == "w1"` assertion fires. This is the WAP chain semantic pin, not a partition
  routing pin.
  _Partially corrected 2026-06-12 (I3 REVIEWER): the claim "IcebergGenerics reads the actual column
  values, which are immutable" is RIGHT for NON-partition columns but WRONG for the identity-PARTITION
  column — that value is PROJECTED from the manifest partition STAMP, so even a writer that bakes the
  wrong category into the parquet column reads back as the stamp. See the I3 REVIEWER block below._

#### I3 REVIEWER corrections (2026-06-12, wt-interop6) — adversarial data-move kill-list
- **A "row-content pin" whose EXPECTED set is loaded from the OTHER side's own read of the SAME
  table is CIRCULAR — it catches only a Rust-vs-Java READER disagreement, never a wrong VALUE.**
  *Why (proven):* the D2 test compared `actual_rows` (Rust read) against `expected_rows` loaded from
  `java_cherrypick_rows.json` (Java's read of the same table). A `data`-value move injected at the
  Java WRITER (id=50/60 `data` "e"/"f"→"X") flowed into BOTH `java_cherrypick_rows.json` AND the
  parquet, so Rust-reads-X == Java-reads-X and the pin PASSED on a corrupted artifact. FIX (this
  pass): added a hand-declared `ground_truth_rows` vec (10→a … 99→bump) asserted independently of the
  Java-derived expected — the probe now FAILS at the anti-circular `assert_eq!` (fail-before/pass-
  after confirmed). DO pin at least ONE side of an interop row-content comparator against a
  hardcoded fixture, never derive BOTH the actual and the expected from the same physical table.
- **An identity-partition column read back through a scan is PROJECTED from the manifest partition
  STAMP, not read from the parquet column — so an `id→category` "routing pin" canNOT detect a
  rows-in-wrong-partition DATA move where the stored column disagrees with the stamp.** *Why
  (proven both legs' probes):* a Java writer that put `category="b"` rows into the partition-`a` file
  (stamp stays "a") was read back by Rust `to_arrow()` as `category="a"` — the projection masked the
  "b" column entirely; the routing pin reported `a={...50,60...}` and PASSED. This is INHERENT Iceberg
  behavior (neither Java `IcebergGenerics` nor Rust validates stored-column == partition-stamp; a
  "garbage in" gap), so it is NAMED RESIDUE, not a fixable assertion. The data-VALUE move is caught
  by the anti-circular ground-truth pin (on the non-partition `data` column); the column-vs-stamp case
  is documented in the test + matrix.
- **A literal parquet file SWAP and an in-place same-length byte patch BOTH trip a STRUCTURAL belt
  before any semantic pin — do NOT rely on a file-swap to exercise a row-content pin.** *Why
  (proven):* swapping two staged files of different sizes crashes the reader on `file_size_in_bytes`
  buffer-fill (manifest records the size); a same-length category-byte flip corrupts the gzip page
  checksum (`corrupt gzip stream does not have a matching checksum`). Both are parse crashes (the
  fail-closed structural belt), NOT the routing/content pin. To exercise a row-content pin with a
  PARSE-CLEAN data move you must drive the WRITER to emit a valid-but-wrong file — a post-hoc file
  edit cannot. The I1 critic's "swap two staged files" sabotage does NOT translate to a partitioned
  identity table: the swap is structurally rejected, and even a clean move is projection-masked.
- **CONFIRMED both lesson-7 bytecode claims against 1.10.0 jars:** `TableMetadata$Builder` has
  `setLocation(String)` + `withMetadataLocation(String)` but NO `withLocation` (javap);
  `PropertiesUpdate` (impl of `UpdateProperties`) calls `TableMetadata.replaceProperties` and contains
  NO `addSnapshot`/`newSnapshot` — so `updateProperties().commit()` creates no snapshot. Both builder
  lessons are correct.
- **CRITIC PROCESS SCAR: `git checkout <file>` on an UNCOMMITTED working-tree file reverts it to
  HEAD, DESTROYING the uncommitted work — it does not "undo my probe edit."** *Why:* I `git checkout`ed
  `InteropOracle.java` to drop a temporary probe, but I3 was uncommitted on top of HEAD, so the
  checkout wiped all 669 lines of the builder's WapDataOracle. Recovered by reconstructing from the
  in-context Reads + the builder transcript (verified: 669 insertions, compiles, chain ×2 green). DO
  revert a temporary probe with the INVERSE edit (or a `.bak` copy), NEVER `git checkout` a file that
  carries uncommitted work.

### 2026-06-12 (I-LANE AUDIT — cross-cutting sweep of the wave6 3-increment branch, Opus)
- **DO finish the journal/map "same-change" obligations PER INCREMENT when each increment adds a
  test/script file — the per-increment critics review their own increment in isolation and DON'T
  catch a sibling increment's omission.** *Finding:* I1 (theta) added `interop_theta.rs` +
  `run-interop-theta.sh` but never added their rows to `crates/iceberg/tests/map.md` OR
  `dev/java-interop/map.md`; I3 (wap-data) added its java-interop/map.md row but missed the
  `tests/map.md` row. Two `map.md` files each had 1 of 3 new entries missing — invisible to the
  three per-increment critics (each saw only its own map edit). The cross-cutting auditor found it
  by enumerating every new file × every map.md it belongs in. *Apply:* when auditing a multi-increment
  branch, grep each new artifact name against every `map.md` that indexes its directory; a per-increment
  green does not imply the union is complete.
- **DO scrub the OTHER clauses of a shared status cell when an increment flips a sub-status — a
  status flip that only APPENDS a "LANDED" note leaves a stale "NEXT-WAVE/deferred" clause that now
  contradicts it.** *Finding:* I1 appended `I1 THETA-BLOB INTEROP LANDED` to the ComputeTableStats
  GAP_MATRIX cell but left the cell's earlier `Java-reads-our-theta-blob interop is NEXT-WAVE` clause
  intact — a self-contradiction inside one cell. Same shape in `task/todo.md`: the "Named next-wave
  interop items" planning bullet still listed theta/view/WAP as open after all three landed. *Apply:*
  the de-triplication rule ("edit the cell, link don't duplicate") extends to KILLING the now-false
  deferral text in the same cell/bullet, not just adding the landing note. (A pre-existing
  `Java-reads-our-stats-file interop is NEXT-WAVE` contradiction predates this branch and is left for
  the next phase re-audit — out of this branch's scope.)
### 2026-06-12 (Wave-6 O3 — divergence burn-down: type-case / sort-bind / manifest-order / avro-map-name, BUILDER Opus, wt-core6)
- **Java's type-name case fold is SCOPED to primitive names — `Types.fromTypeName` lowercases
  (`toLowerCase(Locale.ROOT)`) but `SchemaParser.typeFromJson` matches the WRAPPER names
  `struct`/`list`/`map` with `String.equals` (case-SENSITIVE). Fold primitives + `fixed[..]`/`decimal(..)`
  ONLY; do NOT fold wrappers.** *Why (bytecode):* `fromTypeName` offsets 0-7 lowercase, then the TYPES
  map / FIXED / DECIMAL regexes all consume `aload_1` (the lowercased var); `typeFromJson` offsets 41/55/69
  are `String.equals("struct"/"list"/"map")` on the ORIGINAL. So a one-line `.to_lowercase()` in Rust's
  `PrimitiveType::deserialize` (+ the variant marker `eq_ignore_ascii_case`) is the WHOLE fix — wrappers are
  matched structurally by the untagged `SerdeType` and need no change. NOTE the residual asymmetry: Rust's
  untagged matcher IGNORES the wrapper `type` string for List/Map (accepts `{"type":"STRUCT"}` where Java
  rejects) — a PRE-EXISTING separate posture (Rust too LENIENT on wrappers), NOT the reported divergence
  (Rust too STRICT on primitives); pin it as a scope test, don't fix it in a primitive-case increment.
- **Sort orders: Java binds+`checkCompatibility`-validates ONLY the DEFAULT order at metadata parse; all
  others are `bindUnchecked` (lenient). Mirror exactly — validating every order would be STRICTER than
  Java.** *Why (bytecode):* `TableMetadataParser.fromJson` → `SortOrderParser.fromJson(Schema, node,
  defaultId)`: `if unbound.orderId()==defaultId → bind` (→ `Builder.build()` → `SortOrder.checkCompatibility`
  = source-exists/primitive/transform-applies) `else → bindUnchecked` (no check). Binding itself NEVER throws
  on a missing source id (it keeps the unbound transform string); only `checkCompatibility` does. Rust already
  had a 1:1 `SortOrderBuilder::check_compatibility` — the only gap was `try_normalize_sort_order` never
  CALLING it. Schema bound against is the CURRENT schema (var 10 = current-schema-id). The fix is one
  `check_compatibility(default_order, current_schema)?` call; non-default orders stay untouched.
- **Manifest-list entry order: Java is new-then-carried, all-DATA-before-all-DELETE (`FastAppend.apply` =
  `writeNewManifests()` then `snapshot.allManifests`; `MergingSnapshotProducer.apply` =
  `concat(prepareNewData, carriedData)` then `concat(prepareNewDelete, carriedDelete)`). Rust's SHARED
  `manifest_file` is carried(mixed)-then-new — opposite, in the ONE path every action uses.** *Verdict:*
  DOCUMENT not fix — the oracle SORTS manifests (`snapshot_meta_view.rs:127`), both readers reconcile by
  seq, so it's byte-invisible after the canonical sort; matching Java needs a content-type-separated
  restructure of the shared path that ripples into `manifests[0]`-indexed tests. A "cosmetic, readers
  reconcile" order divergence in a SHARED producer path is a DOCUMENT, not a bounded-cleanup fix.
- **The avro map key/value record rename must cover ANY record (struct included), not just variant —
  Java `TypeToSchema.struct` names EVERY record `"r"+fieldIds.peek()` (the enclosing key/value-id).** *Why:*
  the F1 variant fix renamed only variant map records; a struct map value kept the `"null"` placeholder, so
  two struct-valued maps in one schema emitted duplicate `"null"` records Java's `Schema.Parser` rejects.
  Generalizing `rename_variant_record`→`rename_map_record` (drop the logical-type guard) is the fix.
  TEST-RIGOR TRAP: the array-form (non-string-key) map's ROUND-TRIP test PASSES even under the variant-only
  mutation because apache-avro tolerates duplicate `"null"` records in that array nesting — the round-trip
  is a WEAK pin there; assert the explicit inner record NAMES (`r<keyId>`/`r<valueId>`) to make it
  load-bearing. The string-key Map form's explicit-name assertion catches the mutation directly.
- **`git checkout <file>` to undo a self-mutation WIPES uncommitted work — use a `/tmp` backup `cp`, never
  `git checkout`, when the file has un-committed changes you need to keep.** *Why:* mid-increment a
  `git checkout crates/.../avro/schema.rs` reverted to the O2 commit and erased the (d) fix + tests (the
  worktree changes weren't committed). Restored from the `/tmp/avro.bak` I'd made before mutating. For
  mutation-testing an uncommitted file: `cp file /tmp/x.bak` → mutate → test → `cp /tmp/x.bak file`.

#### O3 REVIEWER corrections (2026-06-12, wt-core6) — adversarial pass vs 1.10.0 bytecode + live Java probe
- **The (a) case-fold fix was correct BUT incomplete — building a Java-vs-Rust acceptance table over the
  parameterized edge spellings exposed FIVE `fixed[..]`/`decimal(..)` parse mismatches the case-fold alone
  did not touch. FIXED.** *Why:* a live `Types.fromTypeName` probe (compiled against the api jar +
  `iceberg-bundled-guava` + caffeine — the static initializer needs the relocated guava `ImmutableMap`)
  vs a Rust acceptance-probe test showed Java's anchored regexes `fixed\[\s*(\d+)\s*\]` /
  `decimal\(\s*(\d+)\s*,\s*(\d+)\s*\)` ACCEPT inner whitespace (`fixed[ 16 ]`, `decimal( 38 , 2 )`) and
  REQUIRE the close bracket/paren, whereas Rust's `deserialize_fixed`/`deserialize_decimal` used
  `trim_end_matches(']'/')')` (a NO-OP when the close char is absent — so `fixed[16`/`decimal(38,2` parsed,
  too LENIENT) and never trimmed the inner content (so `fixed[ 16 ]` was rejected, too STRICT). Fix:
  `strip_prefix` + `strip_suffix` (require the wrapper) + `.trim()` the inner operands. LESSON: when a fix
  is "case-fold the name", the PARAMETERIZED forms have a SECOND parse layer (the bracket/paren structure)
  the fold doesn't reach — build the full accept/reject acceptance table against the live oracle, both
  directions, before declaring parity. The bytecode also reveals GEOMETRY/GEOGRAPHY use `aload_0` (the
  ORIGINAL, case-INSENSITIVE-flag regex) while FIXED/DECIMAL use `aload_1` (the lowercased) — a scope
  detail invisible in source.
- **The builder's documented "open risk" (Rust accepts `{"type":"STRUCT"}` where Java rejects) was REAL,
  WORSE than framed, and the builder's own scope test gave FALSE comfort. FIXED.** *Why:* the test
  `wrapper_type_names_are_not_folded_by_the_primitive_case_fix` asserted rejection via the dedicated
  `StructType` deserializer — a path the production `Type`/`Schema` read NEVER takes. The real read path
  routes through the untagged `_serde::SerdeType`, which matches a wrapper by its FIELD SHAPE and IGNORES
  the `type` string: a probe via `serde_json::from_str::<Type>` accepted `{"type":"STRUCT"}`,
  `{"type":"LIST"}`, `{"type":"MAP"}` ALL. Java `SchemaParser.typeFromJson` (1.10.0 bytecode offsets
  41/55/69 + live `SchemaParser.fromJson` probe) matches with `String.equals` → `IllegalArgumentException`.
  RULING: cheap contained fix (≤30 lines, no untagged-machinery fight) over documentation — added
  `SerdeType::wrapper_type_mismatch()` and a guard in `Type::deserialize` re-imposing Java's exact
  `String.equals`; Rust's writer always emits lowercase, so it never rejects a self-round-trip. LESSON: a
  read-leniency "open risk" must be probed on the PRODUCTION deserialization path, not a sibling
  deserializer — an untagged-enum arm that captures-but-ignores a discriminator field is a silent
  over-acceptance, and a scope test on the wrong path is no pin at all.
- **CONFIRMED (b)/(c)/(d) by independent re-derivation + mutation battery — no defect.** (b)
  `SortOrderParser.fromJson(Schema,node,int)` offsets 5-24: `if orderId==defaultId → bind` (→`build()`→
  `checkCompatibility`) `else → bindUnchecked`; var-10 = current-schema-id schema (bytecode-confirmed);
  knockout of `check_compatibility` fails exactly the 2 reject tests. (c) `FastAppend.apply` new-then-carried
  (addAll at 18 then 94) + `MergingSnapshotProducer.apply` `concat(prepareNew*, filtered*)` data-before-delete
  (offsets 166-272); Rust shared `manifest_file` is carried-then-new; the `current_manifests` test helper
  returns RAW manifest-list order (NOT the oracle's canonical sort), so the `data_manifests[0]`/`manifests[0]`
  spec-order assertions in merge_append/rewrite_manifests ARE writer-order-coupled — the DOCUMENT-not-fix
  defer is justified, the ripple is real. (d) `TypeToSchema.struct` names EVERY struct record `"r"+fieldIds.peek()`
  (offset 4-35); variant-only revert reproduces the two-`"null"` collision on BOTH the string-key and array
  forms, and the array-form caught it ONLY via the explicit `r2`/`r4` name assertion (round-trip alone is a
  weak pin — apache-avro tolerates duplicate `"null"` in that nesting), exactly as the builder warned. The
  manifest schema has only PRIMITIVE-valued maps, so `rename_map_record` never fires on the manifest write
  path — (d) is byte-invisible to interop (write-data chain re-run GREEN, exit 0).

### 2026-06-12 (Wave-7 R2 — partition-stats interop increment, BUILDER Sonnet, wt-r2)
- **`PrimitiveLiteral::UInt128` must serialize as `RawLiteralEnum::String(uuid_str)` (NOT `Bytes`) when the
  field type is `PrimitiveType::Uuid`.** *Why:* `apache-avro`'s `resolve_uuid` (Rust crate) only accepts
  `Value::Uuid` or `Value::String` — it rejects `Value::Bytes` with "Could not find matching type in
  UnionSchema `[Null, Uuid]`". The prior unconditional `Bytes(ByteBuf::from(v.to_be_bytes()))` arm caused
  every UUID-partition Avro manifest write to fail with `DataInvalid`. The fix: in `serde.rs`
  `RawLiteralEnum::try_from` for `UInt128`, check `if matches!(ty, Type::Primitive(PrimitiveType::Uuid))`
  and emit `String(uuid::Uuid::from_u128(v).to_string())`; non-UUID UInt128 falls back to `Bytes`.
  The companion deserialization arm in `RawLiteralEnum::String` must also handle `PrimitiveType::Uuid`
  (Apache Avro decodes `Schema::Uuid` to `Value::Uuid` which serde-deserializes as a `String`). This fix
  is transparent to the partition-stats Parquet path (UUID goes through Arrow FixedSizeBinary(16), not
  Avro); all 2238 lib tests pass. LESSON: any `UInt128` serialization path must be type-gated — the Avro
  crate's UUID schema uses a TYPE-SPECIFIC resolver that rejects the raw byte form.
- **Incremental path engagement can only be pinned INDIRECTLY when `latest_stats_file` is a private `fn`.**
  *Why:* there is no public hook to assert "the incremental branch was taken". Pin it via observable
  SEMANTICS: (1) assert the base stats file is registered in the snapshot lineage before computing the
  child snapshot (use `partition_statistics_iter().any(|f| f.snapshot_id == s1_id)`); (2) after a
  SUBTRACT-arm snapshot (delete_files), assert the subtracted counter is zero and a carried row's
  `last_updated_snapshot_id` equals the BASE snapshot id — a fresh full-compute at S2 would give the same
  counter-zero result but would update `last_updated_snapshot_id` to S2. Both conditions together are a
  strong functional pin without requiring a private-field probe.
- **The SUBTRACT arm requires `delete_files` (a DELETE snapshot), not `delete_data_files` on a
  `fast_append`.** *Why:* `deleted_entry_for_incremental_compute` fires only on a DELETED tombstone in the
  manifest — a `delete_files(data_file)` call creates that tombstone. A `fast_append` snapshot never
  produces DELETED-status entries in the manifests it adds; it only produces ADDED and EXISTING entries.
  An append-only fixture verifies incremental==full, but it never exercises `isub`/`lsub`. Use
  `transaction.delete_files().delete_data_files(vec![file])` to force the DELETED tombstone.
- **SEMANTIC sabotage on a zero-valued counter field (Z3 7b pattern, step 8e): in-place byte-edit the
  Rust incremental stats SOURCE parquet — replace the first zero INT64 with `0x0100000000000000` — then
  re-run the Java verifier.** *Why:* a structural sabotage (truncate the file) is caught by the parquet
  parser before the value check; the merged `data_record_count` for cat=a MUST be exactly zero after the
  SUBTRACT arm, so flipping it to 1 is a meaningful semantic target — the file stays valid parquet but the
  counter is wrong, and Java's `readPartitionStatsFile` D1 verify must catch the mismatch against
  `incr_expected.json`. WHAT THE SCRIPT ACTUALLY DOES (no env var): `cp $path $path.bak`; a `python3`
  one-liner scans for the 8-byte little-endian zero pattern starting at offset 4 (past the `PAR1` magic),
  writes `0x01…` at that offset, and — if the pattern is NOT found — HARD-FAILS (`sys.exit(1)`, and the
  shell aborts the chain on any non-zero `MUTATE_8E_EXIT` after restoring the `.bak`). It does NOT skip:
  cat=a's `data_record_count` is 0 after the SUBTRACT, so a literal zero INT64 is guaranteed present;
  its absence means the parquet encoding changed and the sabotage no longer corrupts a counter, which
  per the promoted "a SKIP branch in a sabotage step is a false-green" lesson MUST hard-fail, never SKIP
  (critic fix, 2026-06-13 — the original 8e shipped an exit-42 SKIP that contradicted that rule). On a
  successful mutation the shell runs `verify-interop-partition-stats-incr` and asserts the run does NOT
  report `0 failures`; finally `cp $path.bak $path` restores. Control + FAIL provenance are both required
  — 8d (the clean D2 read) asserts zero under no sabotage, 8e asserts the corruption fails closed.
  _NOTE:_ there is NO `ICEBERG_INTEROP_PARTITION_STATS_INCR_SABOTAGE_SRC` env var and no Rust-emitted
  `sabotage_src` path; the sabotage is entirely shell-side (in-place edit + `.bak` restore + hard-fail).
- **Java inner class method visibility: `private static` methods in a static inner class are NOT visible
  to sibling inner classes in the same outer class.** *Why:* Java language spec §6.6 — nested classes do
  not inherit the outer class's `private` access boundary for cross-sibling references. When
  `IncrementalPartitionStatsOracle` calls `PartitionStatsOracle.compareStatsRow(...)`, the method must
  be at least package-private. Removing the `private` modifier (making it package-private, the default)
  is the minimal fix; `protected` would also work. This affects any actor-critic split where a NEW inner
  oracle class must reuse helpers from an EXISTING inner class — audit visibility before compilation.

---

### 2026-06-12 — R1: SQL catalog CAS uses base_metadata_location, not freshly-loaded location

**DO** capture `commit.base_metadata_location()` BEFORE calling `commit.apply()` — `apply` consumes the
commit. Use the captured `cas_location` for BOTH the pre-check and the SQL `AND metadata_location = ?`
bind.  Falling back to the freshly-loaded `current_metadata_location` only guards the TOCTOU window; it
cannot catch a strictly-sequential stale commit.

**WHY:** Java `JdbcTableOperations.doCommit` bytecode (iceberg-core-1.10.0.jar): `validateMetadataLocation`
fetches `base.metadataFileLocation()` (the BASE commit arg), not the freshly-loaded DB row; `updateTable`
binds `baseFileLoc` (captured as `var6 = aload_1.metadataFileLocation()`) into the SQL CAS, NOT the loaded
location.  The pre-fix Rust code compared the freshly-loaded location, creating a gap: a caller who loaded
the table, then another writer committed, then the first caller tried to commit — the pre-fix code accepted
it (both its "load" and its "CAS" saw the intermediate location).  The fix: `cas_location` = base when
Some, else current (create-edge / legacy).

---

### 2026-06-12 — R1: TableCommit is pub(crate) — cannot test table stale-base from external crates

**DO NOT** try to construct a stale `TableCommit` in `iceberg-catalog-sql` tests — the
`TableCommit::builder().build()` method is `pub(crate)`.  Use `ViewCommit` instead: `ViewCommit` is
buildable via the public `view.replace_version().to_commit()` API, and both `update_table` and
`update_view` share identical CAS fix structure.  Document the constraint in test comments.

**WHY:** `TypedBuilder` macro exposes `build_method(vis = "pub(crate)")` on `TableCommit`; cross-crate
callers get a compile error.  `Transaction.do_commit` always reloads the table before building a
`TableCommit`, so you cannot exercise the stale path via `Transaction` either.

---

### 2026-06-12 — R1: Fix tokio doctest failures with flavor = "current_thread"

**DO** fix `cargo test --doc` failures from missing `rt-multi-thread` by switching the doctest's runtime
attribute to `#[tokio::main(flavor = "current_thread")]` within the doc comment string.  This requires
zero Cargo.toml changes.

**WHY:** `#[tokio::main]` expands to a multi-thread runtime by default, which requires the `rt-multi-thread`
Tokio feature.  Doctest builds do not enable this feature in the iceberg/sql-catalog crates.  The
`current_thread` flavor uses only `rt` (always enabled).  Hidden doc-comment lines (prefixed with
`//! # `) need a separate `replace_all` pass since they have a different string prefix from visible lines
(`//! `).  The sql catalog lib.rs has the same issue as the iceberg crate — fix both when fixing the gate.

---

### 2026-06-12 — GAP_MATRIX: close FOLLOW-UP (open) clauses with date + what landed

**DO** replace a `FOLLOW-UP (open): ...` clause with `FOLLOW-UP (closed, R1 YYYY-MM-DD): ...` plus a
1-2 sentence summary of the fix when the work lands.  Do NOT delete the clause — the history matters.
Run the pipe-count audit on the edited row immediately after editing (exactly 5 `|` per `^|` row).

---

#### R1 REVIEWER corrections (2026-06-12, wt-r1) — adversarial actor-critic audit

- **"Both arms share the identical CAS fix structure, so the view tests pin the table path too" was a
  FALSE COVERAGE CLAIM — the code was DUPLICATED, not SHARED, and the table arm was UNPINNED.** The
  builder copy-pasted the `cas_location` + pre-check block into BOTH `update_table` and `update_view`.
  Mutation battery (run for real): knocking out the TABLE-arm CAS ALONE (`if false` guard + revert SQL
  bind to `current_metadata_location`) left ALL 68 sql-crate tests GREEN — zero detection; knocking out
  the VIEW-arm CAS alone failed exactly the 2 view stale tests. So the view tests pinned ONLY the view
  arm. *Structural identity of two duplicated blocks is NOT behavioral coverage of both* — a future
  edit to the table block (or a Sonnet that "cleans up" one copy) would silently regress with a green
  suite. FIX (this pass): extracted ONE shared free fn `resolve_commit_cas_location(base, current,
  entity_desc) -> Result<String>` that both arms call; re-ran the battery — knocking out the single
  helper now fails the 2 view tests, and because the table arm flows through the SAME function its CAS
  computation is now genuinely pinned. DO, when a test can only be built for one of two arms (here
  `TableCommit::builder().build()` is `pub(crate)` + `Transaction::do_commit` reloads, so a stale TABLE
  commit is unconstructable from an external crate — VERIFIED, the builder's impossibility claim is
  TRUE), UNIFY the code path so the reachable test pins both, rather than asserting "identical structure"
  and calling it covered. A duplicate-and-claim is the canonical false-1:1 dodge.
- **CONFIRMED the None-fallback is Java-faithful, not a silent door, and consistent with the O1
  MemoryCatalog precedent.** When `commit.base_metadata_location()` is `None` (Java's `base == null`
  create edge), the SQL helper falls back to `current_metadata_location` for the CAS — and the guard is
  `base_metadata_location.is_some() && ...`, so a `None` base NEVER triggers a conflict, matching Java
  `validateMetadataLocation` (which no-ops the location check when `base == null`). This is the same
  posture O1 gave MemoryCatalog (`base == null` → create edge, no CAS), so the two in-repo catalogs are
  now consistent. The SQL posture is no longer weaker than the O1 precedent — the O1 follow-up is
  genuinely closed.
- **CONFIRMED the retry test is honest, not a hand-rolled pass.** `test_view_stale_then_reload_and_retry_succeeds`
  asserts the stale commit fails with `CatalogCommitConflicts` AND `err.retryable() == true`, then reloads
  the view (genuine `Catalog::load_view`), rebuilds a fresh commit whose `base_metadata_location` now
  matches the stored location (verified: `replace_version()` threads `self.metadata_location` → `to_commit`
  populates the field, view.rs:199/417), and that fresh commit succeeds. It does not falsely claim to drive
  the `backon` loop — the actual loop classification (`.when(|e| e.retryable())` at transaction/mod.rs) was
  already pinned by O1's mock tests. The hand-rolled reload accurately models the retry contract.
- **CONFIRMED no sneaky `no_run`/`ignore` downgrade — the doctests were already `no_run` and stay that
  way (compiled under the doctest harness; `main()` does not execute).** The diff hunks change ONLY
  `#[tokio::main]` → `#[tokio::main(flavor = "current_thread")]` (5 iceberg + 1 catalog-sql); no
  ` ```no_run `/` ```ignore ` fence was added. `git diff --name-only` carries zero Cargo.toml/Cargo.lock.
  `cargo test -p iceberg --doc` 85/85, re-run independently.

---

### 2026-06-13 — R3: partition-stats time/fixed/binary exotic-type interop + 7b fail-closed

**DO** extend an existing interop chain to a new partition-value type by following the established
template (here R2's `UuidPartitionStatsOracle`) verbatim per type, NOT by inventing a new structure:
one Java oracle class per exotic type (`Time`/`Fixed`/`BinaryPartitionStatsOracle`) with a
`generate()` (D2 ground truth) + `verify()` (D1 — Java reads Rust), wired into the main switch before
`default`; one Rust GEN + one Rust D2 test per type, env-gated (clean no-op when the env dir is unset),
with HAND-DECLARED anti-circular constants agreed by both sides. The chain script gets a `{generate,
Rust GEN, verify, Rust D2}` block per type, each grepping for the `…: 0 failures` sentinel, chained ×2.

**WHY (the load-bearing facts this increment pinned):**

- **Java's GENERIC in-memory representation of a partition value is `Type.TypeID.javaClass()`, NOT the
  logical type.** Decompiling `Type$TypeID` (iceberg-api-1.10.0.jar) gives the exact mapping the oracle's
  `partition.get(0, Object.class)` returns after `readPartitionStatsFile`: **time → `java.lang.Long`**
  (micros since midnight), **fixed → `java.nio.ByteBuffer`**, **binary → `java.nio.ByteBuffer`** (uuid →
  `ByteBuffer` too, which is why the R2 oracle handled both `UUID` and `ByteBuffer`). So the time oracle
  compares a `Long`; the fixed/binary oracles compare a `ByteBuffer` decoded to a hex string. Building the
  data file uses the SAME generic rep: `PartitionData.set(0, Long)` for time, `PartitionData.set(0,
  ByteBuffer.wrap(bytes))` for fixed/binary. DO decompile `Type$TypeID` rather than guessing the boxed type.

- **No production change was needed — and that is the correct, verified outcome (contrast R2's UUID).**
  O2 had already landed the production reader/writer arms for all four exotic types (`partition_stats.rs`
  `build_partition_field_column`: `Time64MicrosecondArray` from a `Long`; `FixedSizeBinary(L)` for fixed;
  `LargeBinary` for binary; + `arrow/value.rs` read-back `Time` → `Literal::time`, `Fixed(len)` →
  `Literal::fixed`, `Binary` → `Literal::binary`). I VERIFIED these arms exist before writing any test,
  then proved them through the live interop chain — so unlike R2 (where the UUID Avro-serde gap surfaced
  and shipped under-pinned), R3 surfaced NO gap. DON'T add speculative production code when the chain is
  green; DO record the "no production change" explicitly so a reviewer can confirm the arms were exercised,
  not bypassed. Rust `Literal::time(i64)` → `PrimitiveLiteral::Long`; `Literal::fixed`/`Literal::binary`
  → `PrimitiveLiteral::Binary(Vec<u8>)` (fixed and binary share the variant; the schema width discriminates).

- **The 7b SKIP false-green was the SAME anti-pattern 8e already killed — fixed identically.** 7b printed
  "7b SKIP" and proceeded on `sys.exit(42)` when the INT64-3 byte pattern was absent. Converted to
  fail-closed: `sys.exit(1)` on pattern-absent → shell restores `.bak` + aborts the chain. **`set -e`
  GOTCHA (genuinely load-bearing, tested empirically):** a bare failing `python3 -c …` on its own line
  ABORTS the script immediately under `set -euo pipefail` — BEFORE the `MUTATE_*_EXIT=$?` capture and the
  `.bak` restore run. Both the original 8e and a naive 7b therefore aborted without restoring. FIX: capture
  the exit with `MUTATE_*_EXIT=0; python3 … || MUTATE_*_EXIT=$?` so the explicit guard (restore + abort)
  is genuinely reachable. Applied to BOTH 7b and 8e so the "exactly like 8e" mirror is true AND correct.
  DO verify a "restore on abort" claim with a throwaway `set -e` repro — the abort-before-restore footgun
  is invisible in a green run (the temp dir is wiped next chain anyway).

- **Pipe-count audit catches raw `|` inside code spans in a matrix cell.** My first GAP_MATRIX row-118
  edit embedded `` `|| MUTATE_*_EXIT=$?` `` — the literal `||` made the awk pipe count 7, not 5. Reworded
  to drop the `||` literal. DO re-run `awk '/^\|/ {n=gsub(/\|/,"|"); if (n!=5) print NR}'` after ANY cell
  edit that mentions shell `||`/`|` operators; code spans do not protect raw pipes from the table parser.
