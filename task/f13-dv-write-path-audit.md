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

# F-13 — Puffin deletion-vector write path · SEPMO Phase-0 scope audit

Worktree `/home/john/CodeRepos/openSource/apacheIcebergRust/iceberg-f13`, branch
`parity/f13-dv-scope-audit`, at `origin/main` `985b4be74`. Audited 2026-08-24. No file outside this
report was changed, and nothing was committed.

Java oracle: `iceberg-core-1.10.0.jar` and `iceberg-api-1.10.0.jar` under
`/home/john/.m2/repository/org/apache/iceberg/`. **No `-sources.jar` exists for either artifact**
(`ls` of both `1.10.0` directories shows only `.jar`, `.pom`, `.sha1`), so every Java claim below is
`javap -p -c -constants` bytecode with an offset.

---

## Verdict

**F-13 is substantially already met — the ask names a capability the fork shipped on 2026-06-10 and
has interop-proven in both directions ever since.** A production DV write surface exists
(`crates/iceberg/src/writer/base_writer/deletion_vector_writer.rs`, 1178 lines, public, driven today
from three in-tree integration suites that see only the public API); it matches Java
`BaseDVFileWriter` on blob type, layout, one-DV-per-data-file keying, `referenced_data_file`,
`content_offset` / `content_size_in_bytes`, cardinality, the previous-deletes merge and its
file-scoped `rewrittenDeleteFiles` filter, and the shared-Puffin metadata shape. `RowDelta` admits
DVs on `file_format() == Puffin` per Java `ContentFileUtil.isDV`, and **the v3 rule the ask asks for
already exists and is Java-exact** — `validate_delete_file_for_version`
(`transaction/snapshot.rs:1656`) rejects a non-DV position delete on v3 and a DV on v2, mirroring
`MergingSnapshotProducer.validateNewDeleteFile` (bytecode offsets 44 / 55 / 92 / 135). The
GAP_MATRIX has carried this as `R114 · Writer: deletion-vector (V3 puffin DV) | ✅` throughout. What
genuinely remains is four small, independent items — **a missing `with_partition_spec` on the DV
writer (the only real, unrecorded Java divergence I found), absent DV construction invariants on
`DataFileBuilder`, two ergonomic accessors, and the engine-side work of actually lifting a v3 MOR
refusal** — none of which is "build a DV writer".

---

## Q1 — Does a production DV write surface exist in the fork?

**Yes, and it is fully public.**

`crates/iceberg/src/writer/base_writer/deletion_vector_writer.rs` defines, all `pub`:

| Item | Line | Visibility |
|---|---|---|
| `pub const DV_MAX_POSITION: u64` | 89 | public |
| `pub struct PreviousDeletes` + `pub fn new(DeleteVector, Vec<DataFile>)` | 121 / 138 | public |
| `pub struct DVWriteResult { pub delete_files, pub rewritten_delete_files }` | 155 | public, public fields |
| `pub struct DVFileWriter` | 219 | public |
| `pub fn new(OutputFile)` | 235 | public |
| `pub fn with_previous_deletes(HashMap<String, PreviousDeletes>)` | 259 | public |
| `pub fn delete(&mut self, &str, u64, Option<&PartitionKey>) -> Result<()>` | 279 | public |
| `pub async fn close(self) -> Result<Vec<DataFile>>` | 313 | public |
| `pub async fn close_with_result(self) -> Result<DVWriteResult>` | 331 | public |

The module is reachable from the crate root: `crates/iceberg/src/lib.rs:103` `pub mod writer` →
`writer/mod.rs:387` `pub mod base_writer` → `base_writer/mod.rs:21` `pub mod deletion_vector_writer`.
Nothing on the path is `pub(crate)`.

**Proof an external consumer can drive it today** — three files under `crates/iceberg/tests/` (a
separate compilation unit that sees only the public API) do exactly that:

- `tests/interop_dv_write.rs:61` `use iceberg::writer::base_writer::deletion_vector_writer::DVFileWriter;`
  then `:128` `DVFileWriter::new(output_file)`
- `tests/interop_dv_table.rs:89` (same import), `:384` `DVFileWriter::new(output_file)` with
  `PartitionKey`s
- `tests/interop_dv_replace.rs:78` imports `DVFileWriter`, `DVWriteResult`, `PreviousDeletes`;
  `:321-328` `DVFileWriter::new(output_file).with_previous_deletes(...)` → `close_with_result()`

The `OutputFile` comes from the public `table.file_io().new_output(&path)`
(`tests/interop_dv_table.rs:380-383`). The supporting types are public too:
`iceberg::delete_vector::DeleteVector` with public `deserialize_deletion_vector_v1`
(`delete_vector.rs:165`) and `serialize_deletion_vector_v1` (`:369`), and `iceberg::puffin`
re-exports `Blob`, `DELETION_VECTOR_V1`, `BlobMetadata`, `PuffinReader`, `PuffinWriter`
(`puffin/mod.rs:25-62`).

**Refutes the handoff's framing.** The engine track design's phrase "the fork's `DVFileWriter`" is
accurate, not aspirational. The writer landed in `59d3af5aa` (PR #22, "the DV write surface is
complete") and has been touched only cosmetically since (`git show --stat 0421ae158 --
…/deletion_vector_writer.rs` = `2 insertions(+), 2 deletions(-)`).

---

## Q2 — Does it match Java's `BaseDVFileWriter` behaviorally?

Java's whole interface is two methods —
`DVFileWriter.delete(String, long, PartitionSpec, StructLike)` and `result()`, plus inherited
`close()`. Java's `BaseDVFileWriter` ctor is
`BaseDVFileWriter(OutputFileFactory, Function<String, PositionDeleteIndex> loadPreviousDeletes)`.

| Behavior | Java 1.10.0 (bytecode) | Fork | Verdict |
|---|---|---|---|
| Blob type | literal `"deletion-vector-v1"` inlined at `toBlob` offset 4 (same value as `StandardBlobTypes.DV_V1`) | `DELETION_VECTOR_V1` = `"deletion-vector-v1"` (`puffin/blob.rs:25`), used at `deletion_vector_writer.rs:378` | match |
| Blob input fields | `toBlob` offset 7: `ImmutableList.of(MetadataColumns.ROW_POSITION.fieldId())` = `[2147483645]` | `fields(vec![RESERVED_FIELD_ID_POS])` (`:379`), `metadata_columns.rs:36` `i32::MAX - 2` = 2147483645 | match |
| Blob snapshot-id / seq | `toBlob` offsets 19 / 22: `-1L` / `-1L` | `INHERITED: i64 = -1` (`:93`), set at `:380-381` | match |
| Compression | `toBlob` offset 31 `aconst_null` codec ⇒ uncompressed | `puffin_writer.add(blob, CompressionCodec::None)` (`:394`) | match |
| Blob properties | exactly two: `"referenced-data-file"`, `"cardinality"` (`String.valueOf(index.cardinality())`) | exactly two, same keys (`:97`, `:101`), value `deletes.positions.len().to_string()` (`:389`) | match |
| One Puffin, many blobs | `close()` offset 46 calls `newWriter()` **once**; loop 72-209 writes one blob per path | one `PuffinWriter` at `:360`, loop `:376-396` | match |
| One DV per data file | `delete()` offset 15 `deletesByPath.computeIfAbsent(path, …)` — one `Deletes` → one bitmap → one blob | `BTreeMap<String, DeletesForDataFile>` `.entry(path).or_insert_with(…)` (`:297`) | match |
| Partition captured at first `delete` | `computeIfAbsent`; later calls' spec/partition silently ignored. Java deep-copies via `StructLikeUtil.copy(partition)` (`Deletes` ctor offset 26) | `.or_insert_with(|| … partition_key: partition_key.cloned())` — first wins, and `.cloned()` is a deep copy | match |
| Empty writer ⇒ no file | `close()` offsets 19-45: empty `deletesByPath` ⇒ all-empty `DeleteWriteResult`, no file created | `:350-355` returns early before `PuffinWriter::new` | match |
| Previous-deletes merge | `close()` offsets 108-133: `loadPreviousDeletes.apply(path)`, then `positions.merge(previous)` | `:340` `deletes.positions.merge(&previous.positions)` | match |
| `rewrittenDeleteFiles` filter | offsets 138-191: `if (ContentFileUtil.isFileScoped(f)) rewrittenDeleteFiles.add(f)` | `:344` `if is_file_scoped(source_file)` | match |
| `isFileScoped` predicate | `= referencedDataFile(f) != null`; `referencedDataFile` = eq-deletes → null; else the field; else `lower.get(PATH_ID)` **and** `upper.get(PATH_ID)` both non-null **and** `lower.equals(upper)` (I decoded this myself: offsets 30-127, the equality is `ByteBuffer.equals` at offset 109 with `aconst_null; areturn` at 126 on inequality) | `is_file_scoped` (`:186`) — three legs, requires `(Some(lower), Some(upper))` and `lower == upper` | match |
| `createDV` metadata | `FileMetadata.deleteFileBuilder(deletes.spec())` `.ofPositionDeletes().withFormat(PUFFIN).withPath(puffinLocation).withPartition(…).withFileSizeInBytes(wholePuffinSize).withReferencedDataFile(…).withContentOffset(blob.offset()).withContentSizeInBytes(blob.length()).withRecordCount(cardinality)` | `create_dv_metadata` (`:430`) sets the identical eight | match |
| Rolling / target file size | **does not exist in 1.10.0** — no threshold, no roll logic, `newWriter()` called once at `close()` offset 46 | none | match (no gap) |
| `result()` before close | `Preconditions.checkState` → `IllegalStateException("Cannot get result from unclosed writer")` | `close`/`close_with_result` take `self` by value — the state is unrepresentable | structurally stronger; no gap |
| Blob order | Java iterates a `HashMap` ⇒ unspecified | `BTreeMap` ⇒ sorted by referenced path | fork is a strict refinement (own reproducibility contract, documented at `:42-48`); Java reads either |

### Divergences found

**D1 — no `with_partition_spec`; a keyless DV silently claims spec 0. REAL, and NOT recorded in the
GAP_MATRIX.**

Java takes the `PartitionSpec` as a **required per-call argument** (`delete(path, pos, spec,
partition)`) and `createDV` derives the spec id from it via `FileMetadata.deleteFileBuilder(spec)`.
A Java-written DV can therefore never claim a spec the table does not have.

The fork's `delete` takes `Option<&PartitionKey>`, and `create_dv_metadata` stamps the spec **only
inside the `Some` arm** (`:460-464`):

```rust
if let Some(partition_key) = &deletes.partition_key {
    builder
        .partition(partition_key.data().clone())
        .partition_spec_id(partition_key.spec().spec_id());
}
```

With `None`, `DataFileBuilder` falls back to `#[builder(default = "DEFAULT_PARTITION_SPEC_ID")]`
(`spec/manifest/data_file.rs:162-164`), and `DEFAULT_PARTITION_SPEC_ID = 0`
(`spec/partition.rs:33`).

This is the exact bug class the 2026-07-25 change closed for the other three base writers.
`with_partition_spec` exists on `DataFileWriterBuilder` (`data_file_writer.rs:131`),
`EqualityDeleteFileWriterBuilder` (`equality_delete_writer.rs:96`) and
`PositionDeleteFileWriterBuilder` (`position_delete_writer.rs:186`) — and **on none of the DV
writer** (`grep -rn "with_partition_spec" crates/iceberg/src/writer/` returns no hit in
`deletion_vector_writer.rs`). `docs/ENGINE_CONTRACT.md` §7a, which is marked **NORMATIVE**, names
exactly those three builders in its MUST list and does not mention DVs; `crates/iceberg/src/writer/map.md:75`
likewise says "`with_partition_spec` on the data / position-delete / equality-delete builders".

Consequences, in order of severity:

1. **Loud, total.** If the table's spec 0 is partitioned and the DV carries an empty tuple, the
   commit rejects it: `SnapshotProducer::validate_partition_value`
   (`transaction/snapshot.rs:918-923`) fails on the arity mismatch with `"Partition value is not
   compatible with partition type"`. `writer/map.md:78` already documents this shape for the other
   writers.
2. **Silent, but narrower than for parquet position deletes.** If spec 0 exists with a compatible
   partition type, the commit accepts a wrong-spec DV. Unlike a parquet position delete — which
   §7a says then "is never applied to any data file" — a **DV still applies**, because the read side
   routes DVs by referenced-file **path** with no spec and no partition condition
   (`delete_file_index.rs`, per the R117 cell: "consulted with NO spec and NO partition condition").
   The residual costs are wrong per-spec manifest grouping
   (`SnapshotProducer::group_files_by_spec`) and possible manifest-level partition pruning during
   planning (`scan/context.rs:452-469` evaluates every manifest, delete manifests included, against
   the partition filter and `continue`s on a miss). **UNVERIFIED:** I did not build a fixture to
   confirm that a wrong-spec DV is actually pruned end-to-end; the pruning code path is real but
   the consequence is reasoned, not observed.

A workaround exists and is already in use: pass a `PartitionKey` built over the correct spec with an
empty tuple. `PartitionKey::new` is public (`spec/partition.rs:445`) and explicitly permits it
("An all-`void` (or unpartitioned) spec may pair with an empty tuple", `:440`);
`tests/interop_dv_replace.rs:250` has a local `unpartitioned_key(schema, spec)` helper doing
precisely this. So this is a discoverability and symmetry gap with a real footgun, not an
unreachable capability.

**D2 — `DVWriteResult` omits Java's `referencedDataFiles`.** Java's
`org.apache.iceberg.io.DeleteWriteResult` carries three members — `deleteFiles`,
`referencedDataFiles` (a `CharSequenceSet`), `rewrittenDeleteFiles` — plus getters
`deleteFiles()`, `referencedDataFiles()`, `referencesDataFiles()`, `rewrittenDeleteFiles()`. The
fork's `DVWriteResult` has two fields and documents the omission as deliberate and recoverable
(`:150-154`): each `DeleteFile` carries its own `referenced_data_file()`. Java's Spark caller feeds
that set into `RowDelta.validateDataFilesExist(...)`; the fork's
`RowDeltaAction::validate_data_files_exist` is public (`transaction/row_delta.rs:477`) and takes
paths, so the caller must do a one-line `filter_map`. **Cosmetic, documented, not recorded in the
matrix.**

**D3 — the previous-deletes hook is eager, not lazy.** Java holds a `Function<String,
PositionDeleteIndex>` and calls it per path at close (offsets 108-122); the fork takes a
pre-populated `HashMap` (`with_previous_deletes`). Behaviorally identical for a caller that already
knows which data files it touches (which is every real caller, Java's Spark flow included), and the
fork pins the "previous deletes for a path with no new positions are ignored" semantics Java's
iteration order gives (`test_dv_writer_ignores_previous_deletes_for_unwritten_path`, `:1024`).
**No behavioral divergence; noted for completeness.**

**D4 — `DataFileBuilder::build()` enforces none of Java's DV construction invariants. REAL, and
NOT recorded in the GAP_MATRIX.** `FileMetadata$Builder.build()` holds 11 `checkArgument` sites plus
one `IllegalStateException`. Six bear on DV construction, and they are three different kinds — only
the first three are DV *requirements*:

| Java offset | Guard |
|---|---|
| 122 | `format == PUFFIN` ⇒ `"Content offset is required for DV"` |
| 140 | `format == PUFFIN` ⇒ `"Content size is required for DV"` |
| 158 | `format == PUFFIN` ⇒ `"Referenced data file is required for DV"` |
| 179 | `format != PUFFIN` ⇒ `"Content offset can only be set for DV"` |
| 197 | `format != PUFFIN` ⇒ `"Content size can only be set for DV"` |
| 252 | `POSITION_DELETES` ⇒ `"Position delete file should not have sort order"` |

Offsets 122/140/158 are the PUFFIN arm — what a DV MUST carry. Offsets 179/197 are the non-PUFFIN
arm and guard the opposite direction: DV-only fields set on a file that is not a DV. Offset 252 is a
content-type guard unrelated to DVs. The five unconditional guards (file path, delete type, file
format, file size, record count) are not listed here because `DataFileBuilder`'s derive-required
fields already cover them.

The fork's `DataFile` is `#[derive(Debug, PartialEq, Clone, Eq, Builder)]`
(`spec/manifest/data_file.rs:35`) with **no `build_fn(validate = …)`** — `build()` performs no such
check. The fork's own DV writer always sets all three, so its output is fine; the gap is that a
consumer (or a hand-built fixture) can construct a malformed DV `DataFile` and get no error at
construction. The commit door catches only one of the six: `added_dvs_by_referenced_file`
(`transaction/row_delta.rs:516-538`) errors with `"Deletion vector … is missing its referenced data
file"`. A Puffin delete file with a missing `content_offset`, or a Parquet position delete carrying
one, commits clean. This is a genuine parity gap on the F-13 surface that no row records.

---

## Q3 — Does `RowDelta` admit DVs correctly, and is the v3 rule enforced?

**Admission: yes, Java-exact.** Java's `ContentFileUtil.isDV` is, in full (offsets 0-17),
`return f.format() == FileFormat.PUFFIN;` — no check of content type, `contentOffset`, or
`referencedDataFile`. The fork's `is_deletion_vector` (`delete_file_index.rs:225-227`) is
`data_file.file_format() == DataFileFormat::Puffin`. Identical.

**The v3 rule already exists in the fork, and nothing needs building.** This is the part of the ask
I expected to be the remainder and it is not.

Java, `MergingSnapshotProducer.validateNewDeleteFile(DeleteFile)`, reached from `addInternal`
offset 2 (so from both `add(DeleteFile)` and `add(DeleteFile, long)`), `tableswitch 1..4` at
offset 12:

| offset | version | behavior |
|---|---|---|
| 44 | V1 | `IllegalArgumentException("Deletes are supported in V2 and above")` |
| 55 | V2 | `checkArgument(content()==EQUALITY_DELETES \|\| !isDV(file), "Must not use DVs for position deletes in V2: %s", dvDesc(file))` |
| 92 | V3 | `checkArgument(content()==EQUALITY_DELETES \|\| isDV(file), "Must use DVs for position deletes in V%s: %s", formatVersion(), location())` |
| 135 | V4 | identical bytecode to V3 |
| 178 | default | `"Unsupported format version: …"` |

So **yes, Java refuses to admit a new non-DV positional delete file on a v3 table**, by name:
`org.apache.iceberg.MergingSnapshotProducer.validateNewDeleteFile(DeleteFile)`, offsets 92-132.

The fork mirrors this exactly in `validate_delete_file_for_version`
(`transaction/snapshot.rs:1656-1704`): V1 → `"Deletes are supported in V2 and above"`; V2 →
`"Must not use DVs for position deletes in V2: {dv_desc}"`; V3 → `"Must use DVs for position deletes
in V{n}: {location}"`, with `dv_desc` (`snapshot.rs:1624`) a byte-faithful port of
`ContentFileUtil.dvDesc` including Java's `%s`-renders-null formatting. Equality deletes are exempt at V2 and above on
both sides. **V1 is not an exemption**: Java's v1 arm (offsets 44-54) throws
`"Deletes are supported in V2 and above"` before any content test, and the fork's `V1` arm
(`snapshot.rs:1665-1668`) returns `Err` unconditionally — it computes `is_equality_delete` and never
consults it. The behaviour matches; the exemption framing does not. Java's V4 arm is unrepresentable here — the fork's `FormatVersion`
has V1/V2/V3 only — so its absence is not a gap.

**Placement.** Java gates at `add(DeleteFile)` time against `ops().current().formatVersion()`. The
fork gates in the action's `commit()` against the **refreshed** base (`validate_added_delete_files`,
`transaction/snapshot.rs:410-429`; doc at `:398-409` explains the choice). That is strictly stronger
and matches post-1.10.0 Java `main`'s apply-time re-validation: a row delta built before a
concurrent `upgrade_format_version` is re-gated on every retry.

**Reachability.** `validate_added_delete_files` is called from the only two production paths that
add delete files — `row_delta.rs:892` and `rewrite_files.rs:546`. Every other `add_deletes` hit in
`crates/iceberg/src/transaction/` is inside a `#[cfg(test)]` module.

**One matching hole, on both sides.** Java's version check is per-`DeleteFile` at add time;
`MergingSnapshotProducer.add(ManifestFile)` bypasses it entirely (`ManifestWriter` and its
`V1Writer`..`V4Writer` subclasses contain no Puffin or DV reference at all). The fork inherits the
same shape. Since Java does not refuse there, an unconditional fork-side refusal would itself be the
divergence. **No action.**

**Related admission checks, all present.** `validateAddedDVs` — Java's
`ValidationException("Found concurrently added DV for %s: %s")`, unconditional last step of
`BaseRowDelta.validate` (offsets 178-189), self-skipping on an empty `newDVRefs` — is ported with
Java's `VALIDATE_ADDED_DVS_OPERATIONS = {overwrite, delete, replace}` op set
(`transaction/snapshot.rs`, `operation_adds_dvs`). Java's `DeleteFileIndex$Builder.add` throwing
`"Can't index multiple DVs for %s"` is a **known, recorded** residue: the fork rejects at the load
door instead of the index (GAP_MATRIX row R117, "error-path only — Java's index-level
`ValidationException`s land at the Rust load door instead").

---

## Q4 — What can the engine's MOR arm NOT do today?

Side-by-side public surface:

| | `PositionDeleteFileWriter` | `DVFileWriter` |
|---|---|---|
| Builder type | `PositionDeleteFileWriterBuilder<B, L, F>` (`position_delete_writer.rs:135`) | none — `DVFileWriter::new(OutputFile)` |
| `impl IcebergWriterBuilder` | yes (`:193`) | **no** |
| `impl IcebergWriter` | yes (`:233`) | **no** |
| Input | Arrow `RecordBatch` | `delete(&str, u64, Option<&PartitionKey>)` |
| File naming | `LocationGenerator` + `FileNameGenerator` via the builder | caller constructs the `OutputFile` path itself |
| `with_partition_spec` | yes (`:186`) | **no** (divergence D1) |
| Rolling / target size | `RollingFileWriterBuilder` composes over it | none — and **Java has none either** |
| Usable by `writer/partitioning/` fanout / clustered | yes | **no** (needs `IcebergWriter`) |
| Config struct | `PositionDeleteWriterConfig` (`:106`) | none needed |

So the literal ask — "drive it the way it drives `PositionDeleteFileWriter` today" — is the one thing
that is genuinely not possible: the DV writer is not an `IcebergWriter`, so it cannot be dropped into
the existing rolling / fanout / clustered plumbing.

**But that asymmetry is Java-faithful, not a parity gap.** Java's `DVFileWriter` is likewise not a
`FileWriter`/`PartitioningWriter`; it is its own two-method interface fed positionally, and Java's
engine layer (`SparkPositionDeltaWrite`) drives it directly. Making `DVFileWriter` an
`IcebergWriter` would be a **fork-side ergonomic addition beyond Java**, and it does not obviously
fit: an `IcebergWriter` consumes `RecordBatch`es of rows, whereas a DV consumes `(path, position)`
pairs whose natural carrier is the position-delete schema. Recommend **not** doing it as part of
F-13; the engine already drives Java's shape.

What the engine concretely lacks that is worth fixing:

1. **`with_partition_spec`** (D1) — the only item that is both a Java divergence and a footgun.
2. **A public previous-deletes loader.** Java hands `BaseDVFileWriter` a `Function<String,
   PositionDeleteIndex>` and the engine builds it from `BaseDeleteLoader`. The fork's equivalent,
   `CachingDeleteFileLoader`, is `pub(crate)` (`arrow/mod.rs:27` `pub(crate) mod
   caching_delete_file_loader;`, `caching_delete_file_loader.rs:44` `pub(crate) struct`). The engine
   must therefore hand-roll the read: `file_io().new_input(path).read()`, slice at
   `content_offset..+content_size_in_bytes`, `DeleteVector::deserialize_deletion_vector_v1`. That is
   exactly what `tests/interop_dv_replace.rs:286-303` does, with the comment "`loadPreviousDeletes`
   reads the existing DV off disk; this mirrors it without the caching layer". Reachable via public
   API, unpleasant, and it makes every consumer re-derive a ranged-read contract.
3. **`referenced_data_files` on `DVWriteResult`** (D2) — one derived accessor.

Everything else the ask lists is present: `RowDeltaAction::add_deletes`, `remove_deletes`
(`row_delta.rs:358`), `remove_deletes_many` (`:366`), `validate_data_files_exist` (`:477`),
`validate_deleted_files` (`:497`), `validate_from_snapshot` (`:457`).

**On the cited blocker.** The engine's guard sits at
`repark/crates/repark-iceberg/src/write/merge/mod.rs:453-456` and reads: "V3 mandates Puffin deletion
vectors, which the fork's `PositionDeleteFileWriter` does not produce (row R113)." Read literally
that citation is *correct* — R113 is the position-delete-writer row and that writer indeed produces
no DVs. What is wrong is the inference the handoff draws from it, that the fork has no DV writer to
verify: R114 is a separate row and has been ✅ since 2026-06-10. **The fork carries the identical
refusal in its own tree** — `crates/integrations/datafusion/src/physical_plan/delete.rs:376-389`
`require_v2_for_merge_on_read`, called at `:418` and `:1603` — so lifting it is a fork-side unit too,
not only a RePark one.

---

## Q5 — Which GAP_MATRIX rows cover this, and are they accurate?

Cited by permanent anchor per the AGENTS.md convention.

**R114 · Writer: deletion-vector (V3 puffin DV) · ✅** — the primary row. The cell reads:

> `delete_vector.rs` `serialize_deletion_vector_v1` (Java-faithful DENSE layout + `runLengthEncode`
> parity via `RoaringBitmap::optimize` + framing/CRC, hand-computed exact-byte goldens) +
> `writer/base_writer/deletion_vector_writer.rs` `DVFileWriter` (one Puffin per close, blob per
> referenced file). D2 landed 2026-06-10; public hook required `pub mod delete_vector`.

Accurate on everything it asserts, and the ✅ is earned (unit + bidirectional interop, see below).
**Understates in two places**: it records no residue for D1 (the missing `with_partition_spec` /
spec-0 fallback) or D4 (the absent `DataFileBuilder` DV invariants). Both belong in this cell.

**R113 · Writer: position-delete · 🟡** — the row the engine cites. It is about
`writer/base_writer/position_delete_writer.rs` and its 🟡 is held by "remaining multi-spec DATA-merge
commit residue… pending Java-read interop of the evolved-DROP leg + sorting-writer residue". It is
**not** a DV row and does not gate F-13.

**R117 · Read: merge-on-read apply (position-deletes + DVs during scan) · ✅** — the read half.
Records the DV read path, both-direction interop, and the named error-path residue (duplicate-DV
rejected at the load door rather than the index). Accurate; not a write-path blocker.

**R106 · Write: `RowDelta` (merge-on-read) · 🟡** — the commit surface. Its 🟡 is explicitly held by
unrelated residue: "Stays 🟡 on remaining multi-spec DATA-merge commit residue (pending the
`BaseRowDelta`/`BaseOverwriteFiles` multi-spec-manifest-merge decision)." **Not a DV blocker.**

`make check-matrix-anchors` was not run — no matrix edit was made.

---

## Test and evidence discipline

Per `.agents/skills/test-adequacy/SKILL.md` "Apply, never predict": **I executed no test and applied
no mutation.** Every "would catch" below is a hypothesis with a plausible number attached, not a
measurement. Populations are source-level counts.

**Unit tests: 13 of 13 test functions in `deletion_vector_writer.rs` read in full** (lines 527-1178;
`#[tokio::test]` × 12 at 527, 595, 622, 702, 758, 783, 806, 903, 953, 986, 1023, 1061 and `#[test]`
× 1 at 1098). What each would catch, and what it would not:

| Test | Would fail on | Would NOT catch |
|---|---|---|
| `…multi_file_delete_files_carry_blob_coordinates` (527) | wrong blob offset/length, shared-path or file-size drift, per-file cardinality, sorted order | blob `type`, `fields`, `snapshot_id`, `sequence_number` — it decodes bytes at the recorded offsets and never reads blob metadata |
| `…no_deletes_writes_no_file` (595) | creating an empty Puffin (asserts the path does not exist) | — |
| `…deterministic_output_across_runs` (622) | insertion-order-dependent blob layout | footer property key order (explicitly named as known residue at `:617-621`) |
| `…partition_captured_at_first_delete_per_path` (702) | dropping `withPartition`, dropping `partition_spec_id`, last-write-wins on partition. **The only test that asserts `partition_spec_id`** (`:751`, `== 3`) | the `None` path — unrepresentable today, which is divergence D1 |
| `…duplicate_position_counted_once` (758) | counting rather than set-inserting positions | — |
| `…rejects_position_above_java_max` (783) | an off-by-one on `DV_MAX_POSITION` (pins MAX accepted and MAX+1 rejected) | the Java-quirk low word `0x8000_0000` vs `0xFFFF_FFFF` — the constant is asserted against itself |
| `…round_trips_through_d1_loader` (806) | any framing/CRC/offset error, via the real production loader | — |
| `…merges_previous_positions_into_new_dv` (903) | a merge no-op (asserts merged cardinality 2 and union `{1,3}`) and a missing rewritten file | — |
| `…does_not_rewrite_partition_scoped_previous_delete` (953) | `is_file_scoped` returning true unconditionally | — |
| `…does_not_rewrite_equality_delete_previous_source` (986) | dropping the equality-delete early return | — |
| `…ignores_previous_deletes_for_unwritten_path` (1023) | iterating the previous map instead of `deletes_by_path` | — |
| `…no_previous_deletes_is_byte_identical_to_fresh` (1061) | a merge step that perturbs bytes when there is nothing to merge | — |
| `test_is_file_scoped_mirrors_java_referenced_data_file` (1098) | all five `referencedDataFile` legs including the unequal-bounds negative | — |

**Named coverage gaps.** No test asserts the blob's `type` string, `fields` list, `snapshot_id`, or
`sequence_number`. A drift in `fields` (say to the delete-file `pos` id 2147483545 instead of the
`_pos` id 2147483645) would pass all 13. **UNVERIFIED whether the interop suite catches it either** —
Java's `BitmapPositionDeleteIndex.deserialize` reads the byte payload and its validations are on
length, magic, cardinality and CRC, none of which involve `fields`; I did not decode
`BaseDeleteLoader.readDV` to see whether the reader consults blob `fields` at all. Also uncovered:
D4's construction invariants (there is nothing to test — the guards do not exist).

**Interop: 4 suites located, 0 executed.** `crates/iceberg/tests/interop_dv_write.rs`,
`interop_dv_scan.rs`, `interop_dv_table.rs`, `interop_dv_replace.rs`, driven by
`dev/java-interop/run-interop-dv.sh`, whose 16 steps I read from its own `echo` lines (`:101-300`):
Java writes a real V3 DV table and Rust scans it (steps 2-3); Java's production reader decodes the
Rust-written Puffin and the blobs are asserted byte-identical to Java's own serialization of the same
positions (4-6); Java's `IcebergGenerics` production scan reads a Rust-committed V3 table with one
Puffin holding two DVs across two partitions, plus a manifest-API `DeleteFile` cross-check (7-8);
canonical snapshot-metadata views byte-diffed three ways (9-11); and the full writer-merged
replacement chain read back by Java with the merged blob byte-identical and the first live
`removed-dvs` comparison (12-16). This is materially stronger evidence than the unit suite and is
what earns R114's ✅.

---

## Q6 — The honest remaining scope

**The ask is substantially already met. The remainder is five units, four of them small.** None is
"build a DV writer"; the largest is the engine-side work of removing a refusal that the core already
supports.

### U1 — `DVFileWriter::with_partition_spec` (divergence D1)

Add `with_partition_spec(PartitionSpec)` to `DVFileWriter`, mirroring the other three base writers:
the `PartitionKey` wins when both are given; a configured partitioned spec with no key is rejected
(keyed on partition-field arity, not `is_unpartitioned()`, per §7a). Update `docs/ENGINE_CONTRACT.md`
§7a's MUST list and `crates/iceberg/src/writer/map.md` (both currently name only three builders), and
add the residue to GAP_MATRIX row R114.

Size: small — roughly 40 lines of production code in one file plus doc/map edits.
Proof: unit only. Three tests — spec stamped with no `PartitionKey`; `PartitionKey` overrides a
configured spec; partitioned-spec-without-key rejected. Each mutation applied and its
`<N> red out of <M>` recorded. No interop needed: this changes a metadata field the existing
`interop_dv_table` manifest cross-check already reads back, and no on-disk encoding moves.
Blocked on: nothing.

### U2 — DV construction invariants on `DataFileBuilder` (divergence D4)

Port the six DV-bearing `checkArgument`s of Java `FileMetadata$Builder.build()` — three DV
requirements (122/140/158), two DV-field-forbidden guards on non-Puffin files (179/197), and the
position-delete sort-order guard (252) — as
a `build_fn(validate = …)` on `DataFile`.

Size: medium, and **higher-risk than its diff**. `spec/manifest/data_file.rs` is on every write path
and every manifest read path, and `build()` gaining new error cases is a **breaking API change** —
existing hand-built fixtures across `crates/iceberg` and downstream (RePark) may start failing. The
Avro manifest **read** path must be checked: if it routes through `DataFileBuilder`, a validation
that is correct for a writer would start rejecting already-written files, which the AGENTS.md
prohibition on breaking the on-disk format makes a hard block. **That check is a prerequisite, not
an assumption.**
Proof: unit per branch (6), plus a full `make unit-test` sweep to find fixtures the guards break.
Blocked on: confirming the read path does not use the builder. Requires the breaking-change
announcement AGENTS.md mandates.

### U3 — engine-facing DV plumbing (divergence D2 + the `pub(crate)` loader)

Two independent pieces, both ergonomic rather than parity:
(a) a public previous-deletes loader for DVs — a narrow mirror of Java's
`Function<String, PositionDeleteIndex>` source, so consumers stop hand-rolling
`new_input().read()` + slice + `deserialize_deletion_vector_v1` (the shape at
`tests/interop_dv_replace.rs:286-303`). Either promote a narrow entry point over
`CachingDeleteFileLoader`, or add a free function in `delete_vector` that takes `&FileIO` and a DV
`DataFile`. The second is smaller and avoids exposing the caching machinery.
(b) `DVWriteResult::referenced_data_files()` as a derived accessor.

Size: small. Proof: unit — a round trip through the loader, and an accessor test.
Blocked on: nothing. Optional; drop it if the queue is tight.

### U4 — lift the fork's own v3 merge-on-read refusal

Route `crates/integrations/datafusion/src/physical_plan/delete.rs` v3 MOR through `DVFileWriter`
instead of returning `NotImplemented` from `require_v2_for_merge_on_read` (`:379`, called at `:418`
and `:1603`). This is the fork's mirror of RePark's engine unit V3-3.

Size: **the largest of the five**, and the only one that is real engine work. It needs per-data-file
position grouping, the `OutputFile` path, the spec stamp (U1), and — for a table that already has
DVs — the previous-deletes load (U3a) plus `remove_deletes_many` on the commit.
Proof: unit **and** interop. The interop leg is a genuine addition: today's `run-interop-dv.sh`
proves a hand-driven DV chain, not a SQL `DELETE` on a v3 table read back by Java.
Blocked on: U1 (soft — a `PartitionKey` works, but U1 removes the footgun); U3a for the
DV-replaces-DV case.

### U5 — GAP_MATRIX and handoff correction (docs only)

Record D1 and D4 as residue on R114; note in the F-13 response that R114 has been ✅ since
2026-06-10 and that R113 is the position-delete row.
Size: trivial. Proof: `make check-matrix-anchors`. Blocked on: nothing — and it should land with U1
rather than alone, per the one-home-per-fact rule.

**Suggested order:** U5 folded into U1 → U1 → U3 → U4, with U2 sequenced independently once its
read-path prerequisite is answered (it is not on F-13's critical path).

---

## What I could not determine, and why

1. **Whether a wrong-spec DV is actually pruned during scan planning.** `scan/context.rs:452-469`
   evaluates delete manifests against the partition filter and skips on a miss, which is the
   mechanism; I did not build a fixture to observe the end-to-end consequence. Labelled UNVERIFIED
   in Q2/D1.
2. **Whether the interop suite would catch a drift in the blob's `fields` list.** This needs
   `org.apache.iceberg.data.BaseDeleteLoader.readDV` decoded to see whether the reader consults blob
   `fields`. Not decoded — outside the oracle scope I set. Labelled UNVERIFIED in the test section.
3. **Any mutation arithmetic.** No test was executed and no mutation applied, so every "would fail
   on" in the test table is a prediction. A build of this workspace was judged disproportionate for
   a scope audit; the Actor for U1 owes the real numbers with their populations.
4. **Whether `DataFileBuilder` is on the Avro manifest read path.** This decides whether U2 is safe.
   Named as U2's explicit prerequisite rather than guessed.
5. **The engine's own tree.** I read `repark/crates/repark-iceberg/src/write/merge/mod.rs:444-464` to
   verify the quoted guard and its R113 citation, and nothing else in that repository. Claims about
   what the engine can or cannot do beyond that guard are the handoff's, not mine.
