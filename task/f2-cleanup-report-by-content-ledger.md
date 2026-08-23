# Ledger — F-2: split `CleanupReport`'s content-file funnel by content type

**Ledger id:** `F2-CLEANUP-BY-CONTENT-2026-08-23`
**Branch:** `parity/f2-cleanup-report-by-content` (cut off `main` = `fdc8fa27`)
**Brief:** [`task/f2-cleanup-report-by-content-brief.md`](f2-cleanup-report-by-content-brief.md)
**Matrix row touched:** R133 (status cell only).

## 1. Oracle provenance

Jars used, sha256 recorded as required:

| Jar | sha256 |
|---|---|
| `ic-api-1.10.0.jar` (`iceberg-api-1.10.0.jar`) | `627061d401dba9d1a8cada2da6394640c2e803102bb676fb34e5f65503cf2c51` |
| `ic-spark-1.10.0.jar` (`iceberg-spark-3.5_2.12-1.10.0.jar`) | `046dc6a63eab487aaa683a67e4e70c4dc0a86445d61229890541d7889965a1df` |

(A third jar, `ic-core-1.10.0.jar` = `54091489dbdcb31b5a4514372abfc908a7b0be69f76e785f1e82279be4fbd6cc`, sat in the
same scratchpad but was **not** consulted for this unit.)

All findings below were re-derived this session with
`javap -p -c -cp <jar> <class>`; JDK is `java-11-openjdk-amd64`.

### 1.1 Re-verified brief §2 claims

- **`org.apache.iceberg.FileContent`** (api jar) — exactly three members, in declaration order
  `DATA`, `POSITION_DELETES`, `EQUALITY_DELETES`. Confirmed verbatim from `javap -p`. The fork's
  `DataContentType` (`crates/iceberg/src/spec/manifest/data_file.rs:359`) mirrors it 1:1 with the
  same ordinals 0/1/2.
- **`BaseSparkAction$ReadManifest.toFileInfo(ContentFile<?>)`** — bytecode is exactly:
  `ContentFile.location()` → `ContentFile.content()` → `FileContent.toString()` →
  `new FileInfo(String, String)`. **`content()` alone. `format()` is never loaded.** Confirmed.
- **`BaseSparkAction$DeleteSummary.deletedFile(String, String)`** — an `equalsIgnoreCase` chain
  against `FileContent.DATA.name()`, `POSITION_DELETES.name()`, `EQUALITY_DELETES.name()`, then
  the literals `"Manifest"`, `"Manifest List"`, `"Statistics Files"`, `"Others"`, falling through
  to `new ValidationException("Illegal file type: %s", type)` + `athrow`. Confirmed. The sibling
  `deletedFiles(String, int)` has the identical seven-way chain with `addAndGet`.
- **`ExpireSnapshotsProcedure.OUTPUT_TYPE`** — six `StructField`s, `DataTypes.LongType`, in the
  order the brief states. Confirmed from the class's `static {}` block.

### 1.2 Item (a) — the nullability flag, left to this unit to determine

The `StructField.<init>(String, DataType, Z, Metadata)` boolean is the `nullable` parameter, and
in `ExpireSnapshotsProcedure`'s `static {}` block **every one of the six pushes `iconst_1`**:

| idx | column | ctor site (bytecode offset) | flag push | ⇒ nullable |
|---|---|---|---|---|
| 0 | `deleted_data_files_count` | 112–126 | `122: iconst_1` | **true** |
| 1 | `deleted_position_delete_files_count` | 132–146 | `142: iconst_1` | **true** |
| 2 | `deleted_equality_delete_files_count` | 152–166 | `162: iconst_1` | **true** |
| 3 | `deleted_manifest_files_count` | 172–186 | `182: iconst_1` | **true** |
| 4 | `deleted_manifest_lists_count` | 202 (`ldc_w #299`) –206 | `202: iconst_1` | **true** |
| 5 | `deleted_statistics_files_count` | 212–226 | `222: iconst_1` | **true** |

**Finding: all six columns are NULLABLE (`nullable = true`) in Java 1.10.0.** Population is
6 of 6 — the population is the six `StructField` constructions in `OUTPUT_TYPE`, and there are no
others in that block. Per brief §5 this is **recorded only**; the engine's
nullable-vs-non-nullable divergence is NOT fixed here and no fork code depends on it.

### 1.3 Item (b) — what "Others" covers, and whether the fork's walk can produce one

The `"Others"` tag is produced in exactly one place. Grepping the *extracted* spark jar for the
literal `Others` matches two classes only —
`BaseSparkAction$DeleteSummary` (the **consumer**: the seventh `equalsIgnoreCase` arm feeding
`otherFilesCount`) and `BaseSparkAction` (the **producer**). In `BaseSparkAction` the sole
producing site is `private Dataset<FileInfo> otherMetadataFileDS(Table, boolean)`, which builds
one list from:

1. `ReachableFileUtil.metadataFileLocations(table, recursive)` — the table's `*.metadata.json`
   files,
2. `ReachableFileUtil.versionHintLocation(table)` — the Hadoop-catalog `version-hint.text`,
3. `ReachableFileUtil.statisticsFilesLocations(table)` — statistics/partition-statistics files,

and tags the whole list `"Others"` via `toFileInfoDS(list, "Others")` (offset 40, `ldc_w #284`).

So **"Others" is a METADATA-file bucket, not a content-file bucket**: table metadata JSON,
`version-hint.text`, and (on this particular DS) statistics locations. It is consumed by the
whole-table sweeps — `DeleteOrphanFiles` / `DeleteReachableFiles` — not by expire-snapshots'
content walk. (Note the asymmetry in Java itself: `statisticsFileDS` tags stats
`"Statistics Files"` at `BaseSparkAction` offset 8/`ldc #250`, while `otherMetadataFileDS` folds
stats into `"Others"`. That is Java's, not ours, and is out of scope.)

**Can the fork's walk ever produce such a file into the content funnel? No.** The content funnel is
populated at exactly one site — `expire_cleanup.rs`, the candidate-manifest walk,
`content_files_to_delete.insert(entry.file_path().to_string(), entry.content_type())` — whose
source is a `ManifestEntry`. `ManifestEntry::content_type()` is total over `DataContentType`,
which has exactly three variants, so every path in the funnel is DATA, POSITION_DELETES or
EQUALITY_DELETES by construction. Metadata JSON, `version-hint.text` and statistics files never
enter it (statistics have their own funnel, `deleted_statistics_files`; metadata JSON is not
deleted by this module at all — `cleanExpiredMetadata` is deferred, see §5).
**No seventh vector was added.**

### 1.4 Item §2a — the deletion-vector question, pinned

**A DV puffin is counted as a POSITION delete.** Evidence chain, all re-run this session:
`toFileInfo` tags by `content()` alone and never touches `format()`; `DeleteSummary.deletedFile`
dispatches that string against `FileContent.POSITION_DELETES.name()`; a DV is a `DeleteFile`
whose `content()` is `POSITION_DELETES`. Therefore **DVs are NOT separable from Parquet position
deletes in Spark's counts**, and there is no fourth bucket. Pinned by
`test_deletion_vector_puffin_is_counted_as_a_position_delete_not_a_fourth_bucket`, whose name
states the claim, and mutation-proved by Mu4 (§4).

## 2. The shape decision — and whether the union is stored or derived

**Chosen shape: brief option (b), refined — typed ACCESSORS over a private-by-convention type
LOOKUP, with the union kept as the sole membership authority.**

```rust
pub struct CleanupReport {
    pub deleted_content_files: Vec<String>,                            // the UNION — unchanged
    pub deleted_content_file_types: HashMap<String, DataContentType>,  // a type LOOKUP (new)
    ...
}
impl CleanupReport {
    pub fn deleted_content_files_of_type(&self, t: DataContentType) -> Vec<&str>; // DERIVED
    pub fn deleted_data_files(&self) -> Vec<&str>;                                // DERIVED
    pub fn deleted_position_delete_files(&self) -> Vec<&str>;                     // DERIVED
    pub fn deleted_equality_delete_files(&self) -> Vec<&str>;                     // DERIVED
}
```

**Explicit answer to the brief's mandated question: the UNION is STORED; the three typed views
are DERIVED from it.**

Justification against the decisive constraint (`Default` + field-by-field construction must not
permit a silent desync):

- Option (a) — three stored `Vec<String>` fields **alongside** a stored union — was rejected
  precisely because `CleanupReport::default()` plus field-by-field assignment lets a future
  editor populate the union and forget a part, or vice versa, with nothing to catch it. Four
  independent stored lists is four ways to desync.
- The chosen shape makes the dangerous direction **structurally impossible**: each typed view is
  a filter *over `deleted_content_files`*, so a typed view can never name a file the union does
  not, and an empty union yields three empty views with no cooperation from any other field.
  That is what makes the fail-closed posture survive the split for free.
- The remaining theoretical direction — a union member with no entry in the type lookup, which
  would drop it from all three views — is eliminated **by construction, not by discipline**: the
  swept path set and the recorded types both come from the single `BTreeMap<String,
  DataContentType>` built by the walk (`content_paths = map.keys()`, then the lookup is that same
  map filtered to the paths the sweep actually deleted). It is additionally asserted by a
  dedicated test (`test_typed_views_partition_the_deleted_content_union`) and mutation-proved by
  Mu2.
- The union could not be made derived: `deleted_content_files` is a `pub` field an external
  consumer compiles against, so it must remain a field of that name and type.

Why `deleted_content_file_types` is `pub` rather than private: a private field would make
`CleanupReport { .. }` literal construction (and functional-update syntax) impossible for any
downstream crate. Keeping it public preserves constructibility. Its doc comment states plainly
that it is a type lookup, that membership authority is the union, and that consumers should read
it through the accessors.

**Known, accepted API consequence (call it out for the repin):** adding *any* field breaks an
exhaustive struct-literal construction `CleanupReport { a, b, c, d, e }` of this struct. Code that
only *reads* the report — which is what the engine does — is unaffected. Nothing in this
workspace constructs it by literal outside `expire_cleanup.rs`.

## 3. What changed

| File | Change |
|---|---|
| `crates/iceberg/src/transaction/expire_cleanup.rs` | Walk carries `DataContentType`: `content_files_to_delete` is now `BTreeMap<String, DataContentType>` (was `BTreeSet<String>`); the retained-side `.remove(path)` subtraction and the fail-closed `.clear()` are byte-for-byte the same calls on the map. Sweep unchanged (it now takes `map.keys()`, an identical sorted set). New field `deleted_content_file_types` + four derived accessors. Module doc + struct doc updated. Six new/refactored tests. |
| `crates/iceberg/src/transaction/map.md` | `expire_cleanup.rs` row now describes the typed views and the DV-is-a-position-delete rule. |
| `docs/parity/GAP_MATRIX.md` | R133 status cell: dated 2026-08-23 sentence on the additive split. `make check-matrix-anchors` green (79 rows, 5-pipe audit OK). |
| `task/f2-cleanup-report-by-content-ledger.md` | This file. |

Deliverable A is satisfied at the level the brief specified: classification is threaded from the
insert site, and the `:416` subtraction / `:436` clear operate on the same keys they always did.

Tests added (6):
`test_deleted_content_files_split_by_content_type`,
`test_typed_views_partition_the_deleted_content_union`,
`test_deletion_vector_puffin_is_counted_as_a_position_delete_not_a_fourth_bucket`,
`test_unreadable_retained_manifest_spares_every_typed_content_view`, plus the shared fixtures
`expire_one_file_of_each_content_type` / `assert_union_is_concatenation_of_parts` and the
`synthetic_position_delete_file` / `synthetic_equality_delete_file` builders.

The pre-existing fail-closed tests `test_unreadable_retained_manifest_spares_all_content_files`
and `test_unreadable_candidate_manifest_skips_its_files_but_still_dies` are untouched and green.

## 4. Mutation table

**M = 5. The population is the five behaviour knobs this change introduces or newly relies on:**
(1) the classification VALUE recorded at the walk's insert site; (2) the filter that restricts the
recorded lookup to successfully-deleted paths; (3) the fail-closed `content_files_to_delete
.clear()`; (4) whether classification may consult the file FORMAT; (5) the derivation DIRECTION
(views filtered from the union vs. read off the lookup). **Result: 5 of 5 caught.**

Each mutation was applied **individually** (file restored from a pristine copy between every run)
and the whole `transaction::expire_cleanup` module — 21 tests — was executed each time.

| # | Mutation (one knob) | Tests reddened | Verdict |
|---|---|---|---|
| Mu1 | walk insert records the constant `DataContentType::Data` instead of `entry.content_type()` | `..split_by_content_type`, `..deletion_vector_puffin_is_counted_as_a_position_delete..` (2 of 21) | CAUGHT |
| Mu2 | recorded lookup restricted to `DATA` entries — i.e. two buckets drop their files | `..typed_views_partition_the_deleted_content_union`, `..split_by_content_type`, `..deletion_vector_puffin..` (3 of 21) | CAUGHT |
| Mu3 | fail-closed `content_files_to_delete.clear()` deleted | `..spares_all_content_files` (pre-existing), `..spares_every_typed_content_view` (2 of 21) | CAUGHT |
| Mu4 | format-sniff: a `Puffin`-format entry classified `EqualityDeletes` (the forbidden "DVs are their own class" reflex) | `..deletion_vector_puffin_is_counted_as_a_position_delete..` **only** (1 of 21) | CAUGHT |
| Mu5 | classification stored INDEPENDENTLY: lookup populated from the pre-clear walk, views read off the lookup's keys instead of filtering the union | `..spares_every_typed_content_view` **only** (1 of 21) | CAUGHT |

Notes, stated because a mutation table that hides them is worthless:

- **Mu2 is the brief's mandated "make one bucket drop a file; the union assertion must redden"
  proof.** It only counts because the invariant is asserted in a test of its own
  (`test_typed_views_partition_the_deleted_content_union`) with no per-bucket equality ahead of
  it. In the first draft the invariant sat *after* three exhaustive per-bucket `assert_eq!`s in
  the same test, where it was **dominated** — a stricter assertion always failed first, so the
  invariant could not be shown live. The fixture was factored out and the invariant moved into
  its own test specifically to remove that dominance.
- **`assert_union_is_concatenation_of_parts` is still dominated inside the DV test**, which also
  asserts all three views exhaustively. No mutation in this population reddens it *alone* there.
  It is kept as a cheap self-documenting invariant, and no coverage is claimed for that call
  site. The coverage claim rests only on the dedicated test.
- **Mu3 reddens the new typed fail-closed test and the pre-existing union one together**, so on
  its own it would not prove the typed test adds anything. **Mu5 is the mutation that does**: it
  leaves the union fail-closed test GREEN and reddens only the typed one, which is exactly the
  defect class ("the parts say files died; the union says none did") the derived shape exists to
  prevent.
- Every new test asserts a **pre-flight** on its own fixture (the funnel really contains the
  files, or the retained-manifest read really failed) before asserting the property, so none can
  pass vacuously on an empty report.

## 5. Named residues (brief §5 — named, NOT fixed)

- **R133 / F-11 remainder:** `IncrementalFileCleanup`, `cleanExpiredMetadata`, and ref-age
  (`max_ref_age_ms`) remain deferred. R133 stays 🟡 for those reasons — unchanged by this unit.
- **The engine's nullable-vs-non-nullable divergence:** recorded here only (§1.2 — Java 1.10.0
  makes all six columns nullable). No fork behaviour depends on it and nothing was changed for it.
- **Java's own `"Others"` / `"Statistics Files"` tagging asymmetry** for statistics files (§1.3) —
  observed, out of scope, not mirrored.
- Everything else in the handoff queue.
- **No interop test was added.** The classification is unit-proven only; the R133 interop
  deferral list is unchanged.

## 6. Consumed-surface statement (brief §6 — required in explicit words)

**No existing field of `CleanupReport` changed name, type, or population semantics. None.**
`deleted_content_files` is still `pub deleted_content_files: Vec<String>`, still the union of
data files, position/equality delete files and deletion-vector puffins, still deterministically
sorted, and still cleared to empty by the fail-closed retained-manifest path. `deleted_manifests`,
`deleted_manifest_lists`, `deleted_statistics_files`, `failures`, `is_empty()`, `CleanupFailure`
and `CleanupFailureKind` are untouched. The change is purely additive: one new public field
(`deleted_content_file_types`) and four new accessor methods.

The one API caveat the repin unit should know (§2): adding a field means an exhaustive
`CleanupReport { .. }` **struct-literal construction** no longer compiles. Reading the report —
the engine's usage — is unaffected.

## 7. What could NOT be verified

- **No interop evidence.** Nothing here was proven against a Java-written table or against Java's
  own `deleteWith` set. The claim "our classification equals Java's" rests on bytecode reading
  plus fork unit tests, not on a round-trip.
- **The Spark engine side was not exercised at all.** That the six-column procedure output can now
  be filled honestly is an inference from the shapes, not a demonstration; no Spark was run.
- **The external engine's repin was not compiled against this branch.** The additive-only claim in
  §6 is derived from reading the diff, not from building the consumer.
- **`ic-core-1.10.0.jar` was not consulted** for this unit; the `ReachableFileCleanup` set-algebra
  claims quoted in the module docs are inherited from the earlier increment and were not re-derived.
- The `HashMap` iteration order of `deleted_content_file_types` is unspecified; determinism is
  guaranteed only for the union and the accessors (which iterate the sorted union). Anyone
  iterating the field directly gets a nondeterministic order — documented, not defended by a test.
