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

# Evidence ledger — PR-7 production evidence closeout (C-007; verifies C-001 through C-006)

Model: claude-opus-5 (medium)

Plan of record: `task/iceberg-v3-production-work-plan-2026-09-01.md`, section 4 PR-7 and section 10.
Capability status stays in `docs/parity/GAP_MATRIX.md`. This unit adds evidence and documents only.
No product code changed. Base: fork `main` `fb0cacfa8`.

## 1. Charter

```yaml
LEDGER:
  - id: C-007
    proposition: >
      Every claimed test proves its cited behavior. Negative guards have a
      mutation that turns the test red. Interop harnesses hard-fail when their
      mutation or fixture cannot run.
    verdict: PROVEN for the seven merged units, on the evidence in section 2.
    evidence: >
      Every mandatory clause carries a local regression, at least one
      one-knob mutation with a counted red population, and a Java interop
      runner that asserts its own fixture count and carries a sabotage leg.
      Gates 1-7 re-run on the merged tree; section 3.
  - id: C-001..C-006
    proposition: verified, not owned, by this unit.
    verdict: PROVEN except the C-005 credentialed cells, which stay PENDING.
    evidence: section 2; gate 8 is the owner-run prerequisite.
```

## 2. Clause evidence

| Clause | PR | Matrix row | Local regression | Mutation | Java interop (fixtures) | Verdict |
|---|---|---|---|---|---|---|
| C-001 partition-safe rewrite | PR-2 (#254) | R135 | `cargo test -p iceberg --locked --lib rewrite_data_files`, 60 | 9 knobs: 11/59, 37/59, 1/59, 1/1, 1/1, 1/1, 1/1, 3/4, 1/1 | `run-interop-evolved-spec-rewrite.sh`, 5 | PROVEN |
| C-002 REPLACE record-count | PR-1 (#253) | R107 | `cargo test -p iceberg --locked --lib replace_`, 5 named pins | 4 knobs, 1 red out of 1 each | `run-interop-replace-invariant.sh`, 3 | PROVEN |
| C-003 V3 MoR UPDATE lineage | PR-3 (#255) | R114, R166 | `cargo test -p iceberg-datafusion --locked --test row_lineage_mor --test row_lineage_cow --test shared_puffin_dv` | 3/5, 3/5, 3/5, 10/10 | `run-interop-mor-update-lineage.sh`, 2 | PROVEN |
| C-004 upgrade + maintenance | PR-4 (#257) | R109, R114, R135, R136, R166 | `cargo test -p iceberg -p iceberg-datafusion --locked` | MUT-1 2/4, MUT-2 3/4, MUT-3 3/4, MUT-4 1/4, MUT-5 1/4; control 0/4 | `run-interop-v3-upgrade.sh` 9; `run-interop-v3-maintenance.sh` 9 | PROVEN |
| C-005 catalog commit outcomes | PR-5A (#252) | R110, R157 | `cargo test -p iceberg-catalog-glue --lib --locked`, 41; `-p iceberg-catalog-s3tables`, 39 | glue M1-M7: 2,2,2,2,2,1,2 out of 41; matching S3 Tables knobs | `run-pr5a-catalog-commit-decode.sh`, 12 needles | OFFLINE PROVEN; credentialed cells PENDING (gate 8) |
| C-006 branch reference ops | PR-6A (#251) | R168 | `cargo test -p iceberg-datafusion --test interop_branch_dml --locked`, 14 | 3 red out of 3 plus file-set sabotage B | `run-interop-branch-dml.sh`, 4 Java + 6 Rust | PROVEN |
| C-006 branch MoR lineage cell | PR-6B (#256) | R168 | `cargo test -p iceberg-datafusion --test interop_mor_branch_lineage --locked`, 3 | 14 red out of 14 (4 Rust, 10 Java-verify) | `run-interop-mor-branch-lineage.sh`, 1 Java + 7 Rust | PROVEN |
| C-007 evidence discipline | PR-7 (this) | all of the above | gates 1-7, section 3 | the mutation columns above | section 3 gate 7 | PROVEN |

Unit ledgers: `pr1-replace-invariant-ledger.md`, `pr2-partition-safe-rewrite-ledger.md`,
`pr3-row-dml-lineage-ledger.md`, `pr4-v3-upgrade-maintenance-interop-ledger.md`,
`pr5a-catalog-commit-outcomes-ledger.md`, `pr6a-branch-interop-ledger.md`,
`pr6b-mor-branch-lineage-ledger.md`.

## 3. Gates re-run on the merged tree (2026-09-02, each alone)

| # | Command | Exit | Population |
|---|---|---|---|
| 1 | `typos .` | 0 | whole tree, 0 typos |
| 2 | `make check` | 0 | fmt, clippy `-D warnings`, taplo, machete, agent-artifacts, matrix-anchors, comment-blocks, rust-file-size 418 files clean (101 legacy ceilings) |
| 3 | `make check-msrv` | 0 | `cargo +1.94 check --workspace`, cargo 1.94.1 |
| 4 | `cargo build -p iceberg --no-default-features` | 0 | one crate, dev profile |
| 5 | `cargo deny check advisories` | 0 | `advisories ok`; one `warning[yanked]` (finding F-pr7-3) |
| 6 | `cargo nextest run --workspace --all-targets --all-features --exclude iceberg-sqllogictest` | 100 | 3895/4653 run, 3869 passed, 26 failed — every failure a Docker-backed suite (section 4) |
| 6b | the non-Docker leg: same command plus `--no-fail-fast --exclude iceberg-integration-tests -E 'not (binary(hms_catalog_test) or binary(glue_catalog_test) or binary(rest_catalog_test) or binary(file_io_s3_test) or binary(file_io_gcs_test))'` | 0 | 4606 run, 4606 passed, 3 skipped, 91 binaries |
| 7 | `scripts/run_interop_suites.sh --only <the seven suites below>` | 0 | discovery 62, floor 62; `TOTAL: 7 passed, 0 failed, 7 run`. Subset run, branded non-certifying by the driver |
| 7 | `dev/java-interop/run-pr5a-catalog-commit-decode.sh` | 0 | `12 needles` (not in the discovered set — finding F-pr7-2) |
| 8 | `ICEBERG_PR5A_CREDENTIALED=1 dev/pr5a-catalog-commit-outcomes.sh` | not run | owner-run prerequisite, section 5 |
| 9 | RePark V3 scale and statement matrix | not run | consumer prerequisite, section 6 |

Gate 7, per suite, each fixture count read from the runner's own output, not from a ledger claim:

| Suite | Exit | Seconds | Measured population |
|---|---|---|---|
| `run-interop-replace-invariant.sh` | 0 | 64 | `fixture count 3/3` |
| `run-interop-evolved-spec-rewrite.sh` | 0 | 59 | 5 `final.metadata.json`; D1, D2 and the V3 `_row_id` leg all PASS |
| `run-interop-mor-update-lineage.sh` | 0 | 78 | 2 fixtures |
| `run-interop-v3-upgrade.sh` | 0 | 68 | 9 `final.metadata.json`, 4 cells, 2 sabotages PASS |
| `run-interop-v3-maintenance.sh` | 0 | 73 | 9 `final.metadata.json`, 5 actions, 5 sabotages PASS |
| `run-interop-branch-dml.sh` | 0 | 151 | Java fixture count 4 asserted; 6 Rust tables |
| `run-interop-mor-branch-lineage.sh` | 0 | 43 | Java fixture count 1 asserted; 7 Rust GEN artifacts; sabotage RED |
| `run-pr5a-catalog-commit-decode.sh` | 0 | — | 12 needles asserted |

Maven `/opt/maven/bin/mvn -o`, `JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64`. Every runner
hard-fails on a missing prerequisite or a wrong fixture count.

Environment-gated early returns are NOT counted as interop evidence. In gate 6b the offline
crate suites contain env-gated interop tests that are clean no-ops without their fixture
directory (`interop_branch_dml.rs` 14 tests, `interop_mor_branch_lineage.rs` 2 of 3,
`interop_dv_*`, `interop_expire.rs`, `interop_wap_data.rs`, `interop_staged_txn.rs`). Their
interop claim rests on gate 7, where the runners set those variables and the same tests execute.

## 4. CI-only exceptions

| Exception | Exact scope | Why here | Where it runs |
|---|---|---|---|
| Docker-backed suites | `iceberg-catalog-hms::hms_catalog_test` (12 tests), `iceberg-catalog-glue::glue_catalog_test` (14 tests), `iceberg-catalog-rest::rest_catalog_test`, `iceberg-storage-opendal::file_io_s3_test`, `iceberg-storage-opendal::file_io_gcs_test`, package `iceberg-integration-tests` | the Docker daemon is absent: `failed to connect to the docker API at unix:///…/docker.sock … no such file or directory` | `make test` in CI, which runs `docker-up` first |
| `iceberg-sqllogictest` | whole package | excluded by the plan's gate 6 wording | CI |
| Ignored tests, 3 | `equality_delete_set` manual microbench; two `interop_duckdb_malformed_manifest` tests needing `ICEBERG_DUCKDB_MALFORMED_FIXTURE_DIR` | fixtures / benchmark harness not present | on demand |
| Full interop set | 55 of the 62 discovered suites were excluded by `--only` and are named in the gate-7 log | this unit runs the PR-1..PR-6B targeted set | `nightly_interop.yml` runs `make interop` over all 62 |

## 5. Gate 8 — owner-run prerequisite (C-005 credentialed cells)

Command: `ICEBERG_PR5A_CREDENTIALED=1 dev/pr5a-catalog-commit-outcomes.sh`

Not runnable here: no AWS credentials, and `aws` is outside this unit's boundary. Pending cells:
one normal smoke per catalog per commit class, and exactly one accepted-then-response-lost append
per catalog. Rows R110 and R157 keep 🟡 and say so. Until this passes, C-005 is offline-proven
only.

## 6. Gate 9 — consumer prerequisite

RePark's V3 scale and statement matrix. Named, not run: it is RePark's execution and an external
release prerequisite. It authorizes no fork change and substitutes for no fork-side test.

## 7. Open findings roll-up across the unit ledgers

| Id | Sev | Source | Finding | Disposition |
|---|---|---|---|---|
| PR-1 Critic S2 | S2 | pr1 ledger | retry pin inadequate | CLOSED in unit — pin rewritten, mutation 4 red 1/1 |
| PR-4 Critic S3-1/2/3 | S3 | pr4 ledger | pin adequacy: row-only comparison, missing operation assert, unchecked range map | CLOSED in unit — all three landed with a fifth `no-op-rewrite` sabotage |
| PR-6A Critic S2-1 | S2 | pr6a ledger | file-set sabotage missing | CLOSED in unit — sabotage B added |
| F-rp3-c7 | — | pr3 ledger | suspected rewrite row-allocation defect | REFUTED — two-file Spark-seed layout artefact; recorded on row R166 |
| PR-2 suspected rewrite-lineage defect | — | pr4 ledger | suspected defect | REFUTED — cargo mtime fingerprint artefact of the mutation harness |
| F-pr7-1 | S3 | this unit | `Critic attestation:` still reads "pending independent Critic" in the PR-2, PR-3, PR-5A, PR-6A and PR-6B ledgers although all five merged | OPEN, evidence-trace only. The attestations are orchestrator-held and are not fork-tree artefacts. The plan's PR-7 exit gate "all PR Critics converged" is therefore attested outside this repository, and the bundle-scope Critic follows this PR |
| F-pr7-2 | S3 | this unit | `run-pr5a-catalog-commit-decode.sh` does not match the driver's `run-interop-*.sh` glob, so the nightly net never runs it; the driver's documented not-discovered list names only `run.sh` and `run-inspection-manifests.sh` | OPEN. Named in `dev/java-interop/map.md`. Renaming the runner or widening the glob is a code change and is out of this unit's scope |
| F-pr7-3 | S3 | this unit | `cargo deny check advisories` exits 0 with `warning[yanked]: chacha20 0.10.1`, transitive through `rand` → `object_store` → `datafusion` | OPEN. A dependency-file change is out of this unit's scope |

No S0, S1 or S2 finding is open. The three open findings are S3 and none of them is a
correctness claim.

## 8. GAP_MATRIX re-audit against the merged tree

Method: read each cell on `fb0cacfa8`, check it names a merged PR, its unit evidence, its interop
runner, and its residue. A status flip needs both unit and interop evidence. `make check-matrix-anchors`
is green (inside gate 2).

| Row | Status | Verdict | Residue the cell must keep naming |
|---|---|---|---|
| R107 | 🟡 unchanged | PR-1 evidence present: Java offsets 311-364, placement divergence, 3 fixtures | multi-spec interop; DELETE-file-rewrite surface (row R152) |
| R109 | ✅ unchanged | PR-4 closes the named strict-bar gap with cells u1/u2 | none named; the ✅ bar is stated in the legend |
| R110 | 🟡 unchanged | PR-5A seams plus offline CAS rebase | credentialed cells pending; writer-layer spec threading; multi-spec Java interop |
| R114 | 🟡 unchanged | PR-3 MoR UPDATE lineage; PR-4 proves the U4 escape end to end | U4 legacy-parquet refusal UNCHANGED; door-side-only fix; equality-delete `sort_order_id`; Spark-written shared-Puffin fixture |
| R135 | 🟡 unchanged | PR-2 output routing; PR-4 composed maintenance evidence | partial progress, concurrency, sort/zorder, `output-spec-id`, input splitting, general bin-pack interop, Spark-action output comparison |
| R136 | ✅ unchanged | PR-4 proves conversion over a Java-written legacy delete | divergences (a)-(d), residues (e), (f), (i), limits (g), (j), (k) |
| R157 | 🟡 unchanged | PR-5A seven offline commit classes | credentialed smokes and the accepted-then-lost append |
| R166 | ✅ unchanged | PR-3 lineage attachment; PR-4 lineage across the upgrade and all five actions | H-3 external-manifest overlap hazard named, not fixed; reverse MoR UPDATE absent from iceberg-core 1.10.0 |
| R168 | ✅ unchanged | PR-6A both directions; PR-6B the MoR-lineage cell | WAP/`stage_only` + `to_branch`; `RewriteManifests`/`CherryPick` throwing default; catalog/session overrides |

No status changed. Every 🟡 row carries at least one residue this unit could not close, and every
✅ row carries both unit and interop evidence on the merged tree.

## 9. Delivery template

```text
Charter clauses: C-007; verifies C-001 through C-006
Matrix rows: R107, R109, R110, R114, R135, R136, R157, R166, R168 (re-audited, no status change)
Java methods or bytecode read: none new — this unit re-runs the decodes the unit ledgers already record
Files changed: task/pr7-production-evidence-closeout-ledger.md (new); task/iceberg-v3-production-work-plan-2026-09-01.md section 10; task/todo.md; docs/parity/GAP_MATRIX.md; dev/java-interop/map.md
Behavior before: no single evidence surface tied the seven merged units to the plan's clauses and gates
Behavior after: every clause maps to a PR, a matrix row, a counted local population, a counted mutation, and a measured interop fixture count; every gate carries its exit and population; every CI-only exception is enumerated
Negative cases: gate 6 is recorded RED at exit 100 with the Docker-backed failures named rather than hidden; environment-gated early returns are excluded from the interop claim
Test command and population: gates 1-7, section 3 — 4606 offline tests passed (3 skipped) plus 7 interop suites and the PR-5A decode
Mutations, one at a time: none added; the per-unit counts are carried in section 2
Java interop command and fixture count: seven suites through scripts/run_interop_suites.sh --only (3, 5, 2, 9, 9, 4+6, 1+7) plus run-pr5a-catalog-commit-decode.sh (12 needles)
CI-only evidence gap: section 4
Breaking public API change: none — no product code changed
Critic attestation: bundle-scope Critic follows this PR (finding F-pr7-1)
Open findings and dispositions: section 7 — three open S3, no open S0/S1/S2
```
