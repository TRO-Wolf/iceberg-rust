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

# F-17 shared-Puffin deletion-vector closure ledger

## Purpose

F-17 closes a V3 merge-on-read correctness defect in DataFusion `DELETE` and `UPDATE`.
One Puffin file can hold deletion-vector blobs for several data files. A DML operation can touch
one data file while another blob in the same Puffin remains live. The commit removes delete entries
by physical Puffin path. The replacement currently carries only blobs for touched data files. This
combination can remove an untouched sibling blob and resurrect a deleted row.

This ledger is a planning charter. It does not authorize implementation. The Actor must prove every
scope proposition on one frozen source base. The Actor must then reproduce the failure and request
owner approval for the final charter.

## Frozen inputs

- Source audit base: `origin/main` at `e00dec4e65c82467fea8248dd7c88510e9396209`.
- Planning dependency: PR #234 at `f4e8e39d855317017a6d9d8d1c2487ec70527557` changes docs only.
- Unit identifier: F-17. A live search found no earlier F-17 assignment.
- Defect report: RePark handoff dated 2026-08-28.
- Reported fixture: two data files in different partitions and one shared Puffin with blobs A and B.
- Reported result after `DELETE id = 1`: expected live ids `{3, 4, 6}` and actual ids `{3, 4, 5, 6}`.

Any source change after the audit base invalidates the freeze. Fetch `origin/main`, record the new
SHA, and re-prove the affected propositions before implementation.

## Scope audit verdict

```yaml
AUDIT_RESULT: "⚠️ REWRITE_DEMAND"
LOGIC_SCORE: "13/15"
LEDGER:
  - id: C-001
    proposition: "The source audit uses the current origin/main source state."
    verdict: PROVEN
    proof: "A live fetch placed origin/main at e00dec4e65c82467fea8248dd7c88510e9396209."
  - id: C-002
    proposition: "F-17 is the next unused engine item."
    verdict: PROVEN
    proof: "Live Roadmap.md and task/ searches found F-3 through F-16 and no F-17."
  - id: C-003
    proposition: "DataFusion discovers previous DVs only for touched data-file paths."
    verdict: PROVEN
    proof: >
      write_deletion_vectors builds partition_key_by_path from pairs, then loads
      previous_deletes_by_path only for those keys.
  - id: C-004
    proposition: "The replacement writer emits blobs only for paths passed to it."
    verdict: PROVEN
    proof: >
      DVFileWriter receives previous deletes and new positions only for touched keys before
      close_with_result.
  - id: C-005
    proposition: "One requested Puffin path removes every live delete entry with that path."
    verdict: PROVEN
    proof: >
      SnapshotProducer::commit creates requested_paths. resolve_delete_file_paths collects every
      live matching manifest entry.
  - id: C-006
    proposition: "Maintenance already implements shared-Puffin container closure."
    verdict: PROVEN
    proof: >
      maintenance/rewrite_data_files_dv.rs groups live DVs by Puffin path, rewrites untouched
      siblings, and removes all old entries.
  - id: C-007
    proposition: "DataFusion DELETE and UPDATE share the affected V3 writer path."
    verdict: PROVEN
    proof: >
      Both call write_merge_on_read_deletes, which dispatches V3 work to
      write_deletion_vectors.
  - id: C-008
    proposition: "Every surviving DV in an affected Puffin receives defined closure handling."
    verdict: PROVEN
    proof: "The charter defines one output for each class in the finite partition."
    enumeration:
      domain: "Live DV entries in each affected physical Puffin container"
      partition:
        - "Touched referenced file: write the exact old-position union new-position vector."
        - "Untouched referenced file: copy the exact existing position set."
      complete_because: "Each live sibling is either touched by this DML operation or untouched."
  - id: C-009
    proposition: "Every replacement carries correct required DV and Puffin metadata."
    verdict: PROVEN
    proof: "The acceptance matrix checks each enumerated metadata class."
    enumeration:
      domain: "Replacement metadata that can change during container closure"
      partition:
        - "referenced data-file path"
        - "partition spec id and partition tuple"
        - "touched entries inherit current data and file sequence numbers"
        - "untouched siblings retain their original data sequence and inherit the current file sequence"
        - "blob offset and length"
        - "shared Puffin path and physical file size"
      complete_because: "These are the metadata classes named by the defect handoff."
  - id: C-010
    proposition: "The regression suite spans the required failure domain."
    verdict: PROVEN
    proof: "The Required regression matrix defines T1 through T23 as independent assertions."
    enumeration:
      domain: "F-17 regression cases"
      partition: [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23]
      complete_because: >
        The partition covers DELETE, UPDATE, one and two containers, one and many touched files,
        metadata, live rows, equality coexistence, no-op, the DELETE/UPDATE by Replace/Delete
        concurrency cross-product, an unrelated-file legality control, and each command's
        pre-output and post-output failure behavior.
  - id: C-011
    proposition: "Interop uses Java or Spark to write and Java to read the Rust DML result."
    verdict: PROVEN
    proof: "The Interop evidence section defines the exact cross-engine chain and oracle."
  - id: C-012
    proposition: "The sibling carry-forward claim is mutation-proven and cannot silently skip."
    verdict: PROVEN
    proof: >
      The charter requires a one-at-a-time mutation, red-test arithmetic, hard failure when the
      mutation cannot apply, restoration, and a final green rerun.
  - id: C-013
    proposition: >
      DataFusion DELETE rejects concurrent Delete removal of every replacement reference,
      including touched and untouched files.
    verdict: PROVEN
    proof: >
      This is the draft safety contract. T17 pins untouched B, T18 pins touched A, and T23 proves
      that an unrelated C remains outside the validation scope. The final R114 evidence must record
      the deliberate Java conflict-behavior divergence regardless of implementation mechanism.
  - id: C-014
    proposition: "The smallest safe public seam and its ownership are fixed."
    verdict: OPEN
    question: >
      Which exact core API signatures let maintenance and iceberg-datafusion share closure, add
      untouched siblings with an explicit data sequence, and validate every output reference
      against concurrent Replace and Delete operations without exposing maintenance internals or
      broadening the public contract beyond necessity?
  - id: C-015
    proposition: "The reported live-row failure exists on the frozen fork base."
    verdict: OPEN
    question: >
      Which current-base production-reader test reproduces expected live ids {3, 4, 6} and actual
      live ids {3, 4, 5, 6} before the repair?
KILLED_ASSUMPTIONS:
  - "A downstream measured report is not a current-fork reproduction."
  - "The existing single-file DV interop does not prove shared-container closure."
  - "A core-owned helper does not automatically justify exposing a maintenance module."
  - "Validating touched paths does not validate untouched sibling references copied into output."
  - "Adding every replacement through RowDelta.add_deletes cannot preserve sibling data sequence."
  - "RowDelta files-exist validation does not include concurrent Delete operations by default."
  - "The DELETE-side Java divergence follows from the all-reference contract, not from one API choice."
LOGIC_GAPS_DESTROYED:
  - "Referenced-file identity and physical-container identity are separate scopes."
  - "A live-row assertion alone cannot prove that DML changed the intended metadata shape."
  - "A no-match operation must prove no Puffin, manifest, or snapshot change."
  - "DELETE and UPDATE create different orphan sets when failure occurs after output starts."
  - "DELETE and UPDATE arm different concurrent-Delete validation despite sharing the DV writer."
DEMAND: "Close C-014 and C-015, then request owner approval. Do not implement before that gate."
CLARIFYING_QUESTIONS:
  - "What exact current-base test reproduces C-015 through the production reader?"
  - "What exact closure and sequence-bearing RowDelta APIs close C-014 with the narrowest contract?"
RISK_HEATMAP:
  - "Untouched delete resurrects | H | H | OPEN until C-015 is reproduced"
  - "Replacement metadata corrupts routing | M | H | covered by C-009 and T10"
  - "Concurrent Replace or Delete leaves a dangling replacement DV or over-broad rejection | M | H | covered by T14-T18 and T23"
  - "Failure leaves output orphans | M | H | covered by T19-T22"
  - "Public API is wider than both callers need | M | M | OPEN until C-014 is fixed"
REFINED_CHARTER: >
  Not frozen while two clauses are OPEN. Reproduce the reported shared-Puffin failure, approve the
  narrow core closure and sequence-bearing RowDelta seams, then make DataFusion DELETE and UPDATE
  replace each affected physical container without losing siblings. Prove the result through
  T1-T23, cross-engine read-back, and mutation evidence.
GO_DECISION: "Return for fixes"
```

Implementation remains blocked while C-014 or C-015 is `OPEN`. Explicit owner approval is also
required after every surviving clause is `PROVEN`.

## Source-backed defect chain

1. DataFusion derives the touched path set from rows selected by DML.
2. It loads old deletion vectors only for those touched paths.
3. `DVFileWriter` writes replacements only for those paths.
4. `DVWriteResult` returns the absorbed old delete entries for removal.
5. `RowDelta` resolves removal by physical Puffin path.
6. Every live manifest entry with that Puffin path becomes removed.
7. An untouched sibling has no replacement unless the caller closes the whole container.

The local reproducer must prove the final live-row effect. Source inspection alone does not close
C-015.

## Draft implementation charter

### Objective

Make V3 DataFusion `DELETE` and `UPDATE` preserve every live sibling blob in each affected Puffin.
The commit must replace the physical container as one logical operation.

### Required behavior

- Compute the union of physical Puffin containers affected by touched deletion-vector entries.
- Merge old and new positions for every touched referenced data file.
- Copy every untouched sibling blob into a replacement Puffin.
- Remove every old manifest entry for each replaced physical Puffin.
- Preserve each sibling's referenced path, partition tuple, partition spec id, original data
  sequence number, and cardinality. Its replacement file sequence inherits the current commit.
- Stamp touched DVs with normal current-commit data and file sequence semantics.
- Stamp every replacement with correct blob offsets, lengths, shared path, and physical file size.
- Keep exactly one live deletion vector for each referenced data file.
- Validate that every touched and untouched output reference still names a live data file after a
  concurrent `Replace` or `Delete` operation.
- Apply the same rule to DataFusion `DELETE` and `UPDATE`.
- Never resurrect or accidentally delete a row.
- Fail before output creation when a precondition can be checked before I/O.

### Proposed ownership

Extract the container-closure logic from `maintenance/rewrite_data_files_dv.rs` into the smallest
core-owned deletion-vector API. Maintenance and DataFusion DML must call the same primitive.
DataFusion cannot call a core-private symbol because it is a separate crate. The final charter must
therefore approve a narrow public seam or approve a different crate boundary.

Do not expose maintenance internals. Do not duplicate the grouping or sibling-copy algorithm in the
DataFusion crate. Record any public contract in `docs/ENGINE_CONTRACT.md` and run the MSRV gate.
MERGE remains RePark-owned. The core primitive must still be safe for a downstream MERGE caller.
The approved design must also provide a sequence-bearing `RowDelta` commit path for untouched
siblings. It must derive `validate_data_files_exist` from every replacement reference, not only DML
pairs. A likely shape mirrors `RewriteFiles::add_delete_file_with_sequence_number`, but C-013 owns
the behavior ruling and C-014 owns the exact API ruling.
`UPDATE` already includes concurrent `Delete` operations through `validate_deleted_files`.
For this closure path, `DELETE` deliberately broadens Java's normal skip-delete behavior: it must
reject concurrent `Delete` removal of every replacement reference, touched or untouched. This
requirement does not depend on whether the implementation uses broad deleted-files validation or a
narrower all-reference check. Row R114 must record the deliberate behavior divergence.

### Proposed file scope

- `crates/iceberg/src/maintenance/rewrite_data_files_dv.rs`
- the selected core deletion-vector module and its `mod.rs` export
- `crates/iceberg/src/transaction/row_delta.rs`
- `crates/integrations/datafusion/src/physical_plan/delete.rs`
- focused core and DataFusion tests beside those modules
- `dev/java-interop/run-interop-dv-sql.sh` and its shared-Puffin fixture support
- `docs/parity/GAP_MATRIX.md`, `task/todo.md`, this ledger, and affected `map.md` files
- `docs/ENGINE_CONTRACT.md` only if the approved public contract changes

No Cargo manifest, RePark source, RePark pin, or unrelated V3 feature is in scope.

## Required regression matrix

| ID | Case | Required assertion |
|---|---|---|
| T1 | Shared Puffin, touch A with `DELETE` | A gains the new position and B keeps all old positions. No row under B resurrects. |
| T2 | Shared Puffin, touch A with `UPDATE` | The old A row stays deleted, the replacement row is live, and B stays deleted. |
| T3 | Shared Puffin, touch A and B | Both replacements contain the union of old and new positions. |
| T4 | Two Puffins, touch one container | Only the affected physical container changes. |
| T5 | Two Puffins, touch both containers | Both containers close independently with no cross-container loss. |
| T6 | Multiple partitions and specs | Every replacement keeps the referenced file's partition tuple and spec id. |
| T7 | Existing touched DV | Old and new positions form one exact union without duplicates. |
| T8 | Untouched sibling | Its positions, cardinality, and sequence semantics remain unchanged. |
| T9 | Production scan | The exact expected live rows remain. No row resurrects or disappears. |
| T10 | Commit metadata | References, offsets, lengths, size, specs, partitions, and sequences are exact. All old entries are removed. Each referenced file has one live DV. |
| T11 | Equality-delete coexistence | Equality deletes remain live and continue to apply. |
| T12 | No-op DML | No Puffin file, manifest entry, or snapshot changes. |
| T13 | Touched and untouched sequence split | Touched replacements inherit current data and file sequence numbers. Untouched siblings retain their original data sequence and inherit the current file sequence. |
| T14 | `DELETE` after concurrent `Replace` of B | A compaction rewrites untouched file B after DML reads it. The files-exist validation covers B and rejects the stale replacement DV. |
| T15 | `UPDATE` after concurrent `Replace` of B | The same compaction interleaving rejects through the UPDATE entry point. |
| T16 | `UPDATE` after concurrent `Delete` of B | A `DeleteFiles` commit removes B after DML reads it. UPDATE's deleted-files validation rejects the stale replacement DV. |
| T17 | `DELETE` after concurrent `Delete` of B | Concurrent removal of untouched B rejects the stale replacement. Row R114 records the deliberate Java behavior divergence. |
| T18 | `DELETE` after concurrent `Delete` of A | Concurrent removal of touched A also rejects the stale replacement. This control proves the safety contract regardless of the selected validation mechanism. |
| T19 | `DELETE`, pre-output failure | Inject a closure precondition failure before the replacement writer opens. The snapshot and object set stay byte-identical. |
| T20 | `DELETE`, post-output commit failure | Force a commit conflict after one replacement Puffin closes. The snapshot stays unchanged. Exactly that Puffin is new and unreferenced. |
| T21 | `UPDATE`, pre-closure failure | Use a one-output-file fixture and fail before the Puffin opens. The snapshot stays unchanged. Exactly one replacement data file is new and unreferenced. No Puffin is new. |
| T22 | `UPDATE`, post-output commit failure | Use a one-output-file fixture and force a conflict after both writers close. The snapshot stays unchanged. Exactly one replacement data file and one replacement Puffin are new and unreferenced. |
| T23 | `DELETE` after concurrent `Delete` of unrelated C | C is outside every affected Puffin and replacement reference. The F-17 commit succeeds, A and B remain correct, and only C is absent. This bounds the deliberate divergence to replacement references. |

The central regression must fail on the frozen base. Mutate sibling carry-forward and physical
container closure one at a time. Record each result as `N red out of M`, name the red tests, and
hard-fail if the mutation cannot apply. Also mutate reference validation to include every live table
file; T23 must fail. Restore the source and rerun the same gate green.

## Interop evidence

Build one V3 table where Java or Spark writes a Puffin shared by at least two referenced data files.
Run Rust DataFusion `DELETE` against one file and let Java read the exact live values and DV metadata.
Repeat the operation for `UPDATE`. The Java side must prove that the untouched sibling still applies.

Self-read evidence is insufficient. Do not repin RePark in this unit.

## Verification gate

Run the focused core and DataFusion tests first. Then run:

```bash
make check-matrix-anchors
make check
make unit-test
dev/java-interop/run-interop-dv-sql.sh
make test
make check-msrv
```

Use `make check-msrv` when the approved design changes public surface. Record any unavailable external
service or credential gate as not run. Do not convert it into a passing claim.

## Approval gate

Before implementation, update this ledger with:

1. the final frozen source SHA;
2. the current-base reproducer command and exact wrong live-row result;
3. C-014 and C-015 as `PROVEN` with their evidence attached;
4. the exact helper signature and file scope;
5. the accepted regression and interop matrix;
6. known overlap with active branches or PRs.

Present that charter to the owner and stop. Code work starts only after explicit approval.
