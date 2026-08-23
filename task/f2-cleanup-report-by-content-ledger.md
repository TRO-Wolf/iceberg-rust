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
| 4 | `deleted_manifest_lists_count` | 192–206 | `202: iconst_1` | **true** |
| 5 | `deleted_statistics_files_count` | 212–226 | `222: iconst_1` | **true** |

Every row's "ctor site" column is `new #280` offset – `invokespecial #290` offset, re-derived
from the `static {}` disassembly (the six `new #280` sites are at 112 / 132 / 152 / 172 / 192 /
212 and the six `invokespecial #290` at 126 / 146 / 166 / 186 / 206 / 226). _Corrected 2026-08-23
(Critic F4): row 4 previously read `202 (ldc_w #299) –206`, which misstated the site start and
misattributed the string load. Re-derived from the bytecode, not from the wrong number: `new
#280` is at **192**, `ldc_w #299` at 196, `iconst_1` at 202, `invokespecial #290` at 206 — so the
site is **192–206**, which is also the only rendering consistent with the other five rows. The
load-bearing column (`202: iconst_1`) and the finding were correct throughout._

**Finding: all six columns are NULLABLE (`nullable = true`) in Java 1.10.0.** Population is
6 of 6 — the population is the six `StructField` constructions in `OUTPUT_TYPE`, and there are no
others in that block. Per brief §5 this is **recorded only**; the engine's
nullable-vs-non-nullable divergence is NOT fixed here and no fork code depends on it.

_Convergent evidence, added 2026-08-23 (orchestrator sweep):_ the engine's own output schema builds
its columns as `Field::new(<name>, DataType::Int64, true)` — i.e. **nullable**, independently
agreeing with the bytecode finding above. Recorded as **convergent evidence from a second source,
not as proof**: two sources agreeing is not the same as either one being verified by the other, and
the bytecode remains the oracle. It does, however, mean the divergence the brief anticipated may not
exist on this axis — a question for the repin, not for this unit.

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
`toFileInfo` tags by `content()` alone and never touches `format()`;

_Strengthened 2026-08-23 (Critic), and re-derived here independently rather than taken on report:_
the format is not merely unused — **it is never read off disk.** `ReadManifest.entries` builds its
projection at offsets 38–53 as `ImmutableList.of(DataFile.FILE_PATH.name(), DataFile.CONTENT.name())`
— exactly two columns — and pushes it through `ManifestReader.select(Collection)` on BOTH branches
of the `ManifestContent` switch (`ManifestFiles.read` for DATA at 88, `readDeleteManifest` for
DELETES at 112). `file_format` is not in the projection, so Java could not classify by format even
if `toFileInfo` wanted to. This makes the DV ruling stronger than the brief stated it; `DeleteSummary.deletedFile`
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

Why `deleted_content_file_types` is `pub` rather than private: a private field would make the
struct opaque to any downstream reader that wants the raw classification. Its doc comment states
plainly that it is a type lookup, that membership authority is the union, and that consumers
should read it through the accessors.

### 2.1 `#[non_exhaustive]` — ADOPTED (Critic F1)

_Ruled 2026-08-23. The first draft of this ledger reasoned about `pub` vs private on the new field
and never evaluated `#[non_exhaustive]` at all; the Critic was right that this commit is the only
free moment for it._

`CleanupReport` now carries `#[non_exhaustive]`. The reasoning:

- Adding a field already imposes the exhaustive-struct-literal break on downstream code **exactly
  once, in this commit**. Adopting `#[non_exhaustive]` now folds the "you may not write an
  exhaustive literal" contract into a break the consumer is taking anyway. Every future funnel or
  classification the cleanup learns to report is then additive for free. Deferring the attribute
  means paying a second, avoidable break later.
- The direction of this type is one-way: the cleanup PRODUCES the report and callers READ it.
  Nothing in this workspace constructs it outside `expire_cleanup.rs`, and the handoff names it an
  engine-**consumed** surface. Same-crate construction is entirely unaffected by the attribute.
- There is a real escape hatch, so "closed off" does not mean "impossible": a downstream crate that
  genuinely needs to build one (a test double, say) writes `let mut r = CleanupReport::default();
  r.deleted_content_files = …;`. `Default` is derived, the fields are `pub`, and that spelling
  keeps compiling as fields are added — which is precisely the property the attribute is buying.

**The cost, stated rather than glossed:** `#[non_exhaustive]` closes off cross-crate functional
update syntax (`CleanupReport { x, ..Default::default() }`) and exhaustive destructuring, neither
of which the bare field addition would have broken.

**That cost has since been MEASURED against the consumer, and it is zero as the consumer stands
today.** The first draft of this section took the decision on the handoff's "consumed" wording plus
the escape hatch, explicitly flagging that I could not check the engine because it is not in this
workspace. **The orchestrator ran that check on 2026-08-23** against the engine working copies at
`/home/john/CodeRepos/LocalRepark/repark` and `/home/john/CodeRepos/BigRustSparkRebuild`, and
reported:

1. `grep -rn "CleanupReport\s*{" … --include=*.rs` over BOTH repos returns **zero hits** — no
   struct-literal construction anywhere, and therefore no functional-record-update
   (`..Default::default()`) and no exhaustive destructuring of this type.
2. Every use is a **read through a shared reference**: `repark-spark/src/call.rs:664`
   (`report: &CleanupReport`), `repark-sql/src/call.rs:530`
   (`fn expire_result_dataframe(ctx, report: &CleanupReport)`), plus the imports at
   `repark-spark/src/call.rs:80` and `repark-sql/src/call.rs:46`.
3. Field access is `.len()` on the four existing vectors — `deleted_manifests`,
   `deleted_manifest_lists`, `deleted_statistics_files` — with `deleted_content_files` passed by
   reference into the engine's own tally.

**Conclusion: `#[non_exhaustive]` breaks nothing in the consumer as it stands today.** The premise
the attribute was adopted on is verified, not merely reasoned about.

**The precise scope of that claim, which must not be inflated:** this is a **point-in-time
observation of two working copies on one machine, not a contract**. It says nothing about consumer
code written later, about other consumers, or about branches of those repos not checked out at the
time. It is also **not my own check — I did not re-run it**; the two paths and the date are recorded
above precisely so a later reader can. If future consumer code does want to build a report, the
supported spelling remains `CleanupReport::default()` plus field assignment.

### 2.2 The lookup is a `BTreeMap`, not a `HashMap` (Critic F2)

_Changed 2026-08-23._ The lookup is built from a `BTreeMap` and only ever `.get()`, so the swap is
free at the point of use — and a `pub` field's type is free to change now and a breaking change
forever after. It buys two things:

- **Deterministic iteration**, retiring what §7 of the first draft disclosed as an unfixed wart.
- **Deterministic `Debug` rendering.** `CleanupReport` derives `Debug`, so a hash-ordered field
  would have made the whole report's `Debug` output unstable — enough to flake a downstream
  `Debug`-snapshot assertion. That consequence was undisclosed in the first draft; the Critic
  caught it. Fixing it beat documenting it.

**Both changes are shape changes, so the full mutation set was RE-RUN against the new shape** (§4)
rather than inherited: 6 of 6 caught, with reddened-test sets identical to the pre-change run.

## 3. What changed

| File | Change |
|---|---|
| `crates/iceberg/src/transaction/expire_cleanup.rs` | Walk carries `DataContentType`: `content_files_to_delete` is now `BTreeMap<String, DataContentType>` (was `BTreeSet<String>`); the retained-side `.remove(path)` subtraction and the fail-closed `.clear()` are byte-for-byte the same calls on the map. Sweep unchanged (it now takes `map.keys()`, an identical sorted set). New field `deleted_content_file_types: BTreeMap<String, DataContentType>` + four derived accessors; the struct gains `#[non_exhaustive]`. Module doc + struct doc updated. Six new/refactored tests. |
| `crates/iceberg/src/transaction/map.md` | `expire_cleanup.rs` row now describes the typed views and the DV-is-a-position-delete rule. |
| `docs/parity/GAP_MATRIX.md` | R133 status cell: dated 2026-08-23 sentence on the additive split. `make check-matrix-anchors` green (79 rows, 5-pipe audit OK). |
| `task/lessons.md` | Dated 2026-08-23 entry: the one-`&&`-chain gate rule is silently defeated by any pipe inside the chain (Critic F3 — appended per the file's append-only protocol, superseding nothing). |
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

**M = 6. The population is the six behaviour knobs this change introduces or newly relies on:**
(1) the classification VALUE recorded at the walk's insert site; (2) the filter that restricts the
recorded lookup to successfully-deleted paths; (3) the fail-closed `content_files_to_delete
.clear()`; (4) whether classification may consult the file FORMAT; (5) the derivation DIRECTION
(views filtered from the union vs. read off the lookup); (6) the POINT IN TIME the lookup is
snapshotted at, relative to the fail-closed clear. **Result: 6 of 6 caught.**

**M is author-enumerated, and that is a structural weakness of this convention, not a formality.**
Because the author picks the denominator, an "N of M" computed this way can never read below 100%:
it measures the completeness of *my enumeration*, not of the change's behaviour space. So treat
M as a **hypothesis about coverage**, refutable by anyone who names a knob the list omits — and it
was refuted here. Knob (6) and its mutation **Mu6 are the independent Critic's**, not mine: the
first draft shipped M = 5 with knob (6) missing, and the Critic wrote the mutation that exposed it.
The honest reading of the table below is therefore "6 of 6 against an enumeration that has already
been shown incomplete once."

Each mutation was applied **individually** (file restored with a plain `cp` from a pristine copy
between every run — never `cp -p`, per the 2026-08-08 lesson; every run recompiled) and the whole
`transaction::expire_cleanup` module — 21 tests — was executed each time. The whole set was run
TWICE: once against the original shape, and again after the `#[non_exhaustive]` + `BTreeMap`
changes of §2.1/§2.2. **The reddened-test sets were identical across both runs**, so the shape
change invalidated no invariant proof.

| # | Mutation (one knob) | Tests reddened | Verdict |
|---|---|---|---|
| Mu1 | walk insert records the constant `DataContentType::Data` instead of `entry.content_type()` | `..split_by_content_type`, `..deletion_vector_puffin_is_counted_as_a_position_delete..` (2 of 21) | CAUGHT |
| Mu2 | recorded lookup restricted to `DATA` entries — i.e. two buckets drop their files | `..typed_views_partition_the_deleted_content_union`, `..split_by_content_type`, `..deletion_vector_puffin..` (3 of 21) | CAUGHT |
| Mu3 | fail-closed `content_files_to_delete.clear()` deleted | `..spares_all_content_files` (pre-existing), `..spares_every_typed_content_view` (2 of 21) | CAUGHT |
| Mu4 | format-sniff: a `Puffin`-format entry classified `EqualityDeletes` (the forbidden "DVs are their own class" reflex) | `..deletion_vector_puffin_is_counted_as_a_position_delete..` **only** (1 of 21) | CAUGHT |
| Mu5 | classification stored INDEPENDENTLY: lookup populated from the pre-clear walk, views read off the lookup's keys instead of filtering the union | `..spares_every_typed_content_view` **only** (1 of 21) | CAUGHT |
| Mu6 **(Critic-authored)** | lookup built from a snapshot taken BEFORE the fail-closed `clear()`; views still correctly derived from the union | `..spares_every_typed_content_view` **only** (1 of 21), failing at the lookup-emptiness assertion specifically | CAUGHT |

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
- **Mu6 is the answer to a question the first draft did not ask.** Hunting for an undisclosed
  dominated assertion, the Critic identified the lookup-emptiness assertion in the typed
  fail-closed test as the only candidate, then wrote a mutation that isolates it: with the lookup
  snapshotted pre-`clear()` but the views still derived from the union, all three typed views stay
  correctly empty and ONLY the lookup-emptiness assertion fires. Reproduced here — one test fails,
  at `expire_cleanup.rs:2054`, reporting the two leaked classifications. So that assertion is live
  and independently covered, and the coverage is **broader** than the first draft claimed rather
  than narrower.
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
- **The engine already carries its own classification workaround** alongside the funnel —
  `repark-spark/src/call.rs:593` reads `ExpireCounts::tally(&report.deleted_content_files,
  &classified)` (orchestrator sweep, 2026-08-23). Recorded purely as **context for the repin**,
  which may now be able to retire it in favour of the typed views. It is engine-side and explicitly
  out of scope: it was NOT analysed, and nothing in this unit was designed against it.
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

**The API caveats the repin unit must know, in one place (see §2.1/§2.2 for the reasoning):**

1. The struct now carries **`#[non_exhaustive]`**. Cross-crate **struct-literal construction** of a
   `CleanupReport` no longer compiles — neither the exhaustive form nor
   `..Default::default()` — and neither does exhaustive destructuring. The supported spelling is
   `CleanupReport::default()` followed by field assignment, which is stable against every future
   field. **Reading the report is entirely unaffected, and reading is all the engine does:**
   verified 2026-08-23 by the orchestrator against `/home/john/CodeRepos/LocalRepark/repark` and
   `/home/john/CodeRepos/BigRustSparkRebuild` — zero struct-literal constructions in either repo,
   every use a `&CleanupReport` read (§2.1). **This caveat therefore costs the repin nothing as the
   consumer stands today**; it is recorded because it is a point-in-time observation of two working
   copies, not a guarantee about consumer code written later.
2. The new `deleted_content_file_types` field is a **`BTreeMap`**, so both it and the derived
   `Debug` rendering of the whole report are deterministically path-ordered. No previously-existing
   part of the `Debug` output changed shape; the report simply gained one ordered field.

Neither caveat touches an existing field. The first is a construction-site change and applies only
to code that BUILDS a report; nothing in this workspace does so outside `expire_cleanup.rs`.

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
- **The `#[non_exhaustive]` premise is now MEASURED — this entry is retained only as history.**
  _Superseded 2026-08-23 (orchestrator): this bullet previously read "the `#[non_exhaustive]` risk
  is unmeasured", on the grounds that the engine is not in this workspace. The orchestrator checked
  the two engine working copies directly (`/home/john/CodeRepos/LocalRepark/repark`,
  `/home/john/CodeRepos/BigRustSparkRebuild`): zero struct-literal constructions, every use a
  `&CleanupReport` read. The risk is absent as the consumer stands today — see §2.1 for the
  evidence and for the limits of that claim (a point-in-time observation, not a contract; and not
  re-run by me). An earlier supersession on this same bullet — the `HashMap` iteration-order
  disclosure — was retired by the `BTreeMap` swap in §2.2._
- **Determinism of the ordered field is not defended by a test.** The `BTreeMap` gives it by
  construction, and the accessors' ordering is covered (they iterate the sorted union), but no test
  asserts the raw field's iteration order or the report's `Debug` shape.

## 8. Critic pass — 2026-08-23

The independent Critic returned **CONVERGED** (no S1, no S2) on the commit `78fa0772`. It
re-decompiled the oracle from scratch, reproduced all five original mutations with **identical
reddened-test sets**, could not break the shape invariants, and confirmed all three of §4's
self-disclosures under execution. It raised five S3/S4 findings; this section records the
disposition of each, and all five were acted on.

| # | Finding | Disposition |
|---|---|---|
| F1 | `#[non_exhaustive]` never evaluated — free only in this commit | **ADOPTED.** Ruling + the unmeasured risk in §2.1; caveat in §6; residual risk in §7 |
| F2 | make the lookup a `BTreeMap` (kills the §7 nondeterminism AND an undisclosed `Debug`-shape flake) | **SWAPPED.** §2.2; §6 caveat 2; the stale §7 bullet superseded in place with a dated note |
| F3 | the gate-pipe flaw is a PROCESS defect, not a commit defect — write it to `task/lessons.md` | **APPENDED** dated 2026-08-23, per that file's append-only protocol. No commit redone, no amend |
| F4 | wrong bytecode offset in §1.2 row 4 | **CORRECTED** to `192–206`, re-derived from the disassembly with the whole six-site derivation now shown; the wrong number was not "adjusted" |
| F5 | name the M-population convention as a hypothesis; fold in Mu6 | **DONE.** §4 now states M is author-enumerated and was already refuted once, and credits Mu6 to the Critic |

Two of these are corrections to claims the first draft made, and are called out as such rather than
quietly overwritten: the §1.2 offset (F4) and the §7 `HashMap` disclosure (F2, now superseded with a
dated note). One is an addition of coverage I did not know I had (Mu6 / F5). The `#[non_exhaustive]`
adoption (F1) is a new deliberate risk, taken with its premise and its escape hatch both stated.

**Doc sites inspected during this pass and deliberately LEFT ALONE** (each already true; editing
would have made it false):

- §6's core sentence — "No existing field of `CleanupReport` changed name, type, or population
  semantics. None." — is **still exactly true** after F1/F2. `deleted_content_file_types` is a NEW
  field, absent at the pinned rev, so choosing its type is not a change to an existing field, and
  `#[non_exhaustive]` is a struct attribute that alters construction sites, not any field's name,
  type, or population. Only the caveat paragraph beneath it needed rewriting.
- §1.2's finding, table flags, and population statement (all six nullable, 6 of 6) — F4 touched the
  offset column of one row and nothing else.
- §4's three original self-disclosures (the dominance refactor, the DV test's dominated invariant
  call, Mu3's non-uniqueness) — the Critic verified all three by execution; they stand verbatim.
- `crates/iceberg/src/transaction/map.md`, `docs/parity/GAP_MATRIX.md` R133, and the module docs —
  none names the lookup's concrete map type or the construction contract, so `BTreeMap` and
  `#[non_exhaustive]` leave every sentence in them true. Re-read to confirm rather than assumed.
- The archives and `dev/java-interop/map.md`, per the first commit's report — unchanged.

## 9. Orchestrator verification pass — 2026-08-23

After the §8 polish pass, the orchestrator closed the one unmeasured risk this ledger was carrying.
The engine is present on this machine — it simply is not in this workspace, which is why I could not
reach it — and the orchestrator checked it directly at
`/home/john/CodeRepos/LocalRepark/repark` and `/home/john/CodeRepos/BigRustSparkRebuild`.

| What changed | Where |
|---|---|
| `#[non_exhaustive]` risk: **stated → measured and absent** (zero struct-literal constructions; every use a `&CleanupReport` read) | §2.1 (evidence + limits), §6 caveat 1, §7 (old bullet superseded in place, dated) |
| Convergent nullable evidence from the engine's own `Field::new(…, Int64, true)` schema | §1.2, recorded as convergent evidence, not proof |
| The engine's existing `ExpireCounts::tally` classification workaround | §5, as repin context only — not analysed, not designed against |

**This pass changed no code**, only this ledger — so the mutation set was **not** re-run, and did
not need to be: the §4 table and its 6-of-6 result still describe the exact tree at
`03138e28`, which this commit does not touch. The last shape change (§2.1/§2.2) already had its
re-run, reported there.

**Provenance, stated plainly so a later reader weighs it correctly:** the engine findings in §2.1,
§1.2 and §5 are the **orchestrator's executed check, not mine — I did not re-run any of them**. The
repo paths and the date are recorded above so they can be re-run. Every one of them is a
point-in-time observation of two working copies on one machine; none is a contract about future
consumer code.
