<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# F-13 U2 + U4 — ledger

Branch `parity/f13-u2-u4-dv-mor`, cut off main `dcf051e3a`. Scope from
[f13-dv-write-path-audit.md](f13-dv-write-path-audit.md), which found the F-13 ask substantially
already met and named five residual units. U1 and U5 landed in #219. This unit is U2, U3a and U4.

## What shipped

| Unit | Commit | Kind |
|---|---|---|
| U2 — six DV guards on `DataFileBuilder` | `77d5acae7` + `b02c0d196` + `fe0227e79` | BREAKING API |
| U3a — `delete_vector::load_delete_vector` | `855bdedda` | additive |
| U3+U4 — V3 merge-on-read writes DVs | `5ce2826ab` | BEHAVIOR CHANGE |
| U4 interop — Java reads a SQL-written V3 DV table | `96e9d4693` | new suite (floor 53 → 54) |

U3b (`DVWriteResult::referenced_data_files()`) was dropped: U4 does not need it, and the audit
marked it optional.

## Breaking surface, for downstream pins

1. `DataFileBuilder::build()` gains error cases. A caller that builds a malformed DV, or stamps a
   sort order on a position-delete file, now gets `DataFileBuilderError::ValidationError`. The
   on-disk read path is unaffected — it decodes into a struct literal and never calls the builder.
2. A `DELETE`/`UPDATE` with `write.{delete,update}.mode = merge-on-read` on a V3 table used to fail
   with `NotImplemented` and now commits, writing Puffin deletion vectors.

## What the evidence actually covers

The interop leg's ROW comparison is load-bearing (a one-position DV shift is caught). Its two SHAPE
checks are belt-and-braces: `RowDelta` refuses both corruptions before they can be written. Measured,
not assumed — see `crates/integrations/datafusion/tests/interop_dv_sql.rs`.

## Bundle Critic — CONVERGED at cycle 4

Four cycles, seven surviving mutations found and closed, no S1 or S2 outstanding. Two corrections
the Critic made to MY claims, both recorded because they change what the evidence means:

- The partitioned end-to-end refusal test does NOT isolate the partition-tuple carry, as its
  docstring and my commit message claimed. The fork's own V2 position delete satisfies both legs of
  the rule at once, so killing either alone leaves the other holding the test up. The per-leg pins
  are the seam tests; the docstring now says so. (The commit message stays wrong in history —
  amending a pushed-branch commit is a rewrite; this ledger is the correction of record.)
- Mutation counts must be read with `--no-fail-fast`. Plain `cargo test` aborts after the first
  failing target, so a mutation that reddens the lib target hides the integration target's result
  entirely. Two of the Critic's own intermediate readings were wrong until it re-ran that way.



NOT CONVERGED on first pass, three S2s:

1. The UPDATE arm's `remove_deletes_many` was unpinned — the DELETE arm had a second-delete test,
   the UPDATE arm did not. Deleting the whole block survived all 305 tests.
2. The per-path `PartitionKey` was unpinned for a DV write spanning more than one data file. Every
   test touched one data file per statement, so stamping one arbitrary key was an identity.
3. A data file still covered by a legacy PARQUET position delete was refused only at commit — AFTER
   the Puffin was written, leaving an orphan. Reachable by upgrading a V2 table with position
   deletes to V3.

(1) and (2) are the same class again, on its fourth and fifth occurrence in this unit. (3) is now a
PRE-IO refusal, restoring the §7a guarantee, and the residue is named on R114.

Cycle 2 then found THREE more, and one of them was a genuine correctness hole rather than a missing
test: my pre-IO predicate derived the delete's reference from the `referenced_data_file` field
alone, but Java's `PositionDeleteWriter.close()` never sets that field — it leaves equal `file_path`
bounds. The repository already had the Java-faithful derivation
(`delete_file_index::referenced_data_file_location`) and the scan uses it; my predicate did not.
**The commit door I cited as the backstop had the same bug**, so a bounds-scoped position delete
stamped under another spec passed both and was silently superseded by the DV. Both now share one
rule, and the door's fix is pinned by a hand-built fixture (the shape is not producible by the
fork's own writer, which always stamps the matching partition).

The predicate was also too WIDE — no `delete_seq >= data_seq` filter — so it refused deletes Java
allows. Both halves are now extracted as named seams and unit-tested per cell, because the
end-to-end tests could not distinguish them.

Cycle 3 found the seams' domain still open in two cells, and the pattern is worth recording: I had
pinned the reference axis and the sequence axis INDEPENDENTLY, never crossed. Every sequence
assertion sat on the partition-scoped leg, so the named leg was free to skip the filter entirely.
The second was the same shape — the entry seam's test asserted three of the tuple's four fields,
following the function signature rather than the fields the function computes, and the one it
skipped was invisible because the only end-to-end test used an UNPARTITIONED table.

Fixture note worth keeping: the commit-door test must go V2-then-upgrade. `RowDelta` refuses a
Parquet position delete on a V3 table outright, so the hazard cannot be constructed directly at V3.

It also found a regression I introduced: extracting `validate_delete_vector_coordinates` dropped the
`\` line continuations, so five runtime messages — two of them the SCAN path's, correct before this
branch — carried runs of 18-22 literal spaces. No test asserted a full message string. One now does.

## Critic cycles

Two cycles on U2, both NOT CONVERGED on first pass, both for the same class: an exemption pinned on
only one of its content types, so a mutation narrowing it survived. Cycle 1 found two survivors,
cycle 2 found three more. The fix that held was enumerating the validator's closed 6-cell domain in
the test module rather than adding tests one at a time.

The same class then appeared in U4's own tests: dropping the per-file `PartitionKey` survived all
303 tests, because every test used an unpartitioned table where a missing key and the real key both
resolve to spec 0.

## The commit-door fix protects future commits only

An external logic review confirmed the causal chain at bytecode level and raised one residual the
ledger did not state. Recording it, because it changes what an operator should do:

The fix is **door-side prevention**. A table that ALREADY carries the bad state — a DV committed
over a live bounds-scoped position delete before this change — still silently resurrects rows at
every read. `get_deletes_for_data_file` early-returns on a DV hit and never consults the
position-delete maps, so the still-live parquet delete's positions are simply dropped. Java's
`findDV` does the same, so the READ side is parity-faithful; the corrupted metadata is the defect,
not the reader. Detecting or repairing such a table is not in this unit and is not currently
possible through any fork surface.

The same review corrected a phrasing risk now fixed in R114 and in the seam test's message:
`write.delete.granularity` defaults to FILE in **Spark** (`SparkWriteConf.deleteGranularity()`),
NOT in Iceberg core, whose `TableProperties.DELETE_GRANULARITY_DEFAULT` is PARTITION. The claim was
correct as written but one word from being wrong.

## Queue, not this unit

1. **Equality-delete `sort_order_id` defaulting.** Java's `FileMetadata$Builder.build()` switch case
   2 (offsets 273-293) defaults an absent `sortOrderId` to `SortOrder.unsorted().orderId()`; the fork
   writes the field absent. Recorded as residue on R114. Inert in practice — sort order does not
   participate in equality-delete application on either side.
2. **Data-file `sort_order_id` is never stamped.** Java's `io.DataWriter` calls
   `DataFiles$Builder.withSortOrder` on every data file it builds; the fork's writers never set
   `sort_order_id` at all. Wider than (1), adjacent to R111, and currently recorded nowhere else.
   Surfaced by the U2 Critic.
3. **`referenced_data_file_location` placement.** It is now `pub` and re-exported from `spec`, but
   `spec`'s public namespace is otherwise a flat glob of TYPES, and this is a pure function of one
   `DataFile`. The idiomatic Rust home is an inherent method,
   `DataFile::referenced_data_file_location(&self)`. Not done here: 20 call sites across 7 files,
   and churning established code late in a large PR is the wrong trade. Surfaced by the bundle
   Critic.
4. **Half a seam is exposed, and deliberately so for now.** `is_deletion_vector` sits in the same
   module, answers the other half of "does this delete still cover this file", and stays
   `pub(crate)`. `referenced_data_file_location` went public only because the DataFusion crate
   needed it across a crate boundary; the PAIR is not yet a supported external contract. If a
   downstream engine is to drive `RowDelta` directly, export both and say so. Surfaced by the
   bundle Critic.
5. **U2's Puffin-DATA arm is a fork-only strengthening.** Java's delete-file builder cannot express a
   Puffin data file and its data-file builder has no blob-coordinate fields, so requiring the
   coordinates there has no Java oracle. Kept deliberately (it rejects only shapes no writer emits,
   and dropping it would make the validator diverge structurally from Java's format-keyed shape);
   declared in R114 and in the code comment.
