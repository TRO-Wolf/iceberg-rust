#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# SCAN-PLAN (planTasks) interop harness — GAP_MATRIX row R148 (+ row R124 for the BatchScan legs).
# Proves the Rust `TableScan::plan_tasks` produces the SAME bin-packed CombinedScanTask GROUPS as Java's REAL
# `table.newScan().option(read.split.*).planTasks()`, in BOTH directions, with target/lookback/open-file-cost
# HAND-DECLARED IDENTICALLY on both sides (anti-circular — InteropOracle.ScanPlanOracle.{TARGET,LOOKBACK,
# OPEN_FILE_COST} mirror interop_scan_plan.rs {TARGET,LOOKBACK,OPEN_FILE_COST}: 4096 / 5 / 0).
#
# THE FIXTURE (V2, unpartitioned {1 id long required, 2 data string optional}), built identically by Java's
# generate + Rust's GEN: several REAL parquet data files of VARYING size so bin-packing is non-trivial —
#   big.parquet   (800 rows, TINY parquet row groups ⇒ MULTIPLE row groups ⇒ split offsets ⇒ OFFSETS-AWARE split)
#   mid.parquet   (40 rows, single row group ⇒ FIXED-SIZE split under the small target)
#   small1/small2 (5 rows each ⇒ pack together)
#   big-deletes   (a position delete over big.parquet ⇒ the bin-pack WEIGHT includes the delete bytes)
#   merge.parquet (120 rows ⇒ 2 row groups, whole file UNDER the target ⇒ both splits CO-BIN and MERGE)
#   gap.parquet   (210 rows ⇒ 3 row groups whose MIDDLE span alone exceeds the target ⇒ splits 0 and 2
#                  co-bin NON-CONTIGUOUSLY and must NOT merge)
#
# THE ADJACENT-SPLIT MERGE LEGS (added 2026-07-24 with the merge port). Java's `TableScanUtil.planTasks` maps
# every bin through `BaseCombinedScanTask(List)`, whose ctor calls `TableScanUtil.mergeTasks`: a run of
# LIST-ADJACENT splits of the SAME file that are exactly CONTIGUOUS collapses into ONE spanning member. The
# big/mid/small fixture never deterministically co-bins two adjacent splits of one file, so before these legs
# the comparison was silently VACUOUS with respect to the merge (the nightly hit that configuration only by
# accident — the delete-file path length nudged the pack weights across the 4096 knife edge, which is why the
# failure was runner-only). `merge.parquet` + `gap.parquet` close that hole. Each is planned under its OWN
# metrics-prunable row filter (the fixture files own DISJOINT id ranges) and at the DELETE-FREE APPEND
# snapshot, so its splits meet an EMPTY bin-packer with weights equal to their lengths — deterministic on any
# parquet build and any checkout path. Both engines additionally assert NON-VACUITY: the merged member's
# length must STRICTLY exceed the largest single row-group span (manifest field-132), and the co-binned
# gap.parquet pair must stay TWO non-contiguous members.
#
# THE COMPARISON: each group is a SORTED set of member keys "(basename,start,length)"; the plan is the
# MULTISET of per-group member-key sets + the group count. Both engines plan the SAME on-disk table within a
# direction (so split offsets, hence start/length, are byte-identical); group emission ORDER is NOT compared.
#
# THE CHAIN (both directions + two fail-closed sabotage stages — the Java battery and the source mutation):
#   [1/7] Reset the temp dir.
#   [2/7] Java GENERATE (Direction 1): write the table; run the REAL Java planTasks; emit java_scan_plan.json
#         + java_batch_scan_plan.json + java_merge_scan_plan.json + java_gap_scan_plan.json.
#   [3/7] Rust GEN (Direction 2): write the SAME logical table to <dir>/rust_table via the production write
#         path + emit rust_scan_plan.json / rust_batch_scan_plan.json / rust_merge_scan_plan.json /
#         rust_gap_scan_plan.json (Rust's own plans) + a self-check that big.parquet split.
#   [4/7] Java VERIFY (Direction 2): load the RUST-written table, run the REAL Java planTasks (plain scan,
#         BatchScan, and both merge filters), assert every plan == Rust's. A FAIL is a real finding.
#   [5/7] Rust D1: load the JAVA-written table, run plan_tasks (plain, BatchScan, both merge filters), assert
#         == Java's plans, and assert the merge pins.
#   [6/7] SABOTAGE (fail-closed), three load-bearing legs: (1) Java re-plans the Java table with a much
#         LARGER target (target*1024), forcing the groups to MERGE, and asserts the plan DIVERGES from
#         java_scan_plan.json (14→1); (2) big.parquet's split-offsets are DROPPED, flipping the offsets-aware
#         split to fixed-size windows (8→2); (3) the MERGE is shown load-bearing — the UNMERGED split keys
#         and a re-plan at a target too small to CO-BIN both diverge from the canonical merge-filtered plan.
#         HARD-FAILS if ANY leg leaves the grouping unchanged (vacuous) or its target is absent.
#   [7/7] MERGE MUTATION (fail-closed): delete the `merge_tasks` call from the PRODUCTION `CombinedScanTask`
#         constructor, re-run the D1 leg, and require it to go RED — then restore the source, verify the
#         restore is byte-identical (md5), and re-run D1 GREEN (which also evicts the mutant from cargo's
#         build cache). HARD-FAILS if the call site is absent, the mutant PASSES, or the restore is not clean.
#
# This is a TEST-ONLY ORACLE (a dev tool) — NOT part of the shipped Rust library, NOT part of the offline
# `cargo test` gate (it needs Java + Maven). Nothing binary is committed; the temp dir under
# dev/java-interop/target/ is gitignored. Step [7/7] edits a tracked source file IN PLACE and restores it in
# the same step; the restore is md5-verified and runs even when the mutant run fails.
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64, the repo's Rust
# toolchain. The first Maven run must be ONLINE to populate ~/.m2; after that `mvn -o` runs fully offline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-scan-plan"
D1_DIR="${TMP}/d1"
GEN_DIR="${TMP}/d2"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

echo "==> [1/7] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${D1_DIR}" "${GEN_DIR}"

echo "==> [2/7] Java GENERATE (Direction 1): write the table + emit java_scan_plan.json (+ batch/merge/gap)"
run_oracle -Dexec.args=generate-interop-scan-plan \
  -Dinterop.scan_plan.dir="${D1_DIR}"
for plan in java_scan_plan.json java_merge_scan_plan.json java_gap_scan_plan.json; do
  test -f "${D1_DIR}/${plan}" \
    || { echo "FAIL: ${plan} not produced"; exit 1; }
done
echo "    java_scan_plan.json + java_merge_scan_plan.json + java_gap_scan_plan.json produced OK"

echo "==> [3/7] Rust GEN (Direction 2): write the SAME table to ${GEN_DIR}/rust_table + emit its plans"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_SCAN_PLAN_GEN_DIR="${GEN_DIR}" \
    cargo test -p iceberg --test interop_scan_plan -- --nocapture
)
test -f "${GEN_DIR}/rust_table/metadata/final.metadata.json" \
  || { echo "FAIL: Rust GEN did not produce rust_table/metadata/final.metadata.json"; exit 1; }
for plan in rust_scan_plan.json rust_merge_scan_plan.json rust_gap_scan_plan.json; do
  test -f "${GEN_DIR}/rust_table/${plan}" \
    || { echo "FAIL: Rust GEN did not produce rust_table/${plan}"; exit 1; }
done

echo "==> [4/7] Java VERIFY (Direction 2): Java's REAL planTasks over the RUST-written table"
VERIFY_OUT="$(
  run_oracle -Dexec.args=verify-interop-scan-plan \
    -Dinterop.scan_plan.dir="${GEN_DIR}"
)"
echo "${VERIFY_OUT}"
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
    || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-scan-plan: 0 failures'; then
  echo "==> FAILED — Java's planTasks over the Rust-written table diverged from Rust's plan (a real finding)."
  exit 1
fi
echo "    Direction-2 OK — Java's planTasks over the Rust table matches Rust's plan"

echo "==> [5/7] Rust D1: Rust's plan_tasks over the JAVA-written table == Java's plan"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_SCAN_PLAN_DIR="${D1_DIR}" \
    cargo test -p iceberg --test interop_scan_plan -- --nocapture
)
echo "    Direction-1 OK — Rust's plan_tasks over the Java table matches Java"

echo ""
echo "==> [6/7] SABOTAGE BATTERY (fail-closed): three corruptions over the Java table, each must DIVERGE"
echo "     from the canonical plan — (a) a much larger target (groups MERGE), (b) big.parquet's split"
echo "     offsets DROPPED (offsets-aware → fixed-size windows), (c) the adjacent-split MERGE removed"
echo "     (unmerged keys) and its co-binning broken (a target too small to share a bin). HARD-FAILS if"
echo "     ANY leg is unchanged."
SABOTAGE_OUT="$(
  run_oracle -Dexec.args=sabotage-interop-scan-plan \
    -Dinterop.scan_plan.dir="${D1_DIR}"
)"
echo "${SABOTAGE_OUT}"
if echo "${SABOTAGE_OUT}" | grep -q '^FAIL ' \
    || echo "${SABOTAGE_OUT}" | grep -q 'VACUOUS' \
    || ! echo "${SABOTAGE_OUT}" | grep -q 'fail-closed confirmed'; then
  echo "==> SABOTAGE FAILED — a perturbation did NOT change the plan; that knob is not load-bearing."
  exit 1
fi
echo "    SABOTAGE PASS: the large-target, dropped-split-offset and merge legs all diverged (fail-closed)"

echo ""
echo "==> [7/7] MERGE MUTATION (fail-closed): drop the merge_tasks call from CombinedScanTask::new and"
echo "     require the D1 leg to go RED, then restore byte-identically (md5-verified)."
MERGE_SRC="${REPO_ROOT}/crates/iceberg/src/scan/task_group.rs"
MERGE_CALL='tasks: merge_tasks(tasks),'
# HARD-FAIL, never skip: a mutation that cannot be applied has proven nothing.
grep -qF -- "${MERGE_CALL}" "${MERGE_SRC}" \
  || { echo "FAIL: the merge call site '${MERGE_CALL}' is ABSENT from ${MERGE_SRC} — the mutation cannot"; \
       echo "      be applied, so this leg would prove nothing. Re-point it at the current call site."; \
       exit 1; }
BEFORE_MD5="$(md5sum "${MERGE_SRC}" | cut -d' ' -f1)"
cp -p "${MERGE_SRC}" "${MERGE_SRC}.bak"
python3 - "${MERGE_SRC}" <<'PY'
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    source = handle.read()
mutant = source.replace("tasks: merge_tasks(tasks),", "tasks,", 1)
if mutant == source:
    raise SystemExit("the merge call site vanished between the grep and the rewrite")
with open(path, "w", encoding="utf-8") as handle:
    handle.write(mutant)
PY
# `|| rc=$?` keeps the restore below REACHABLE under `set -e` when the mutant run fails (which is the
# EXPECTED outcome here).
MUTANT_RC=0
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_SCAN_PLAN_DIR="${D1_DIR}" \
    cargo test -p iceberg --test interop_scan_plan
) > "${TMP}/merge-mutant.log" 2>&1 || MUTANT_RC=$?
mv -f "${MERGE_SRC}.bak" "${MERGE_SRC}"
# `cp -p` + `mv` preserve the ORIGINAL mtime, which would leave the restored source OLDER than the
# mutant's build artifacts — cargo's mtime staleness check would then silently reuse the MUTANT lib
# for every later build in this checkout. Stamp it forward; the content (hence the md5 below) is
# untouched.
touch "${MERGE_SRC}"
AFTER_MD5="$(md5sum "${MERGE_SRC}" | cut -d' ' -f1)"
if [ "${BEFORE_MD5}" != "${AFTER_MD5}" ]; then
  echo "FAIL: ${MERGE_SRC} was NOT restored byte-identically (${BEFORE_MD5} != ${AFTER_MD5})"
  exit 1
fi
if [ "${MUTANT_RC}" -eq 0 ]; then
  echo "==> MUTATION FAILED (VACUOUS) — the D1 leg PASSED with the adjacent-split merge REMOVED, so it"
  echo "    does not actually test the merge. See ${TMP}/merge-mutant.log"
  exit 1
fi
grep -m 3 -E "panicked at|assertion .*failed|test result: FAILED" "${TMP}/merge-mutant.log" \
  | sed 's/^/    RED: /' || true
# Prove the restore is not merely byte-identical but BUILDS GREEN again — which also forces the
# mutant artifacts out of the build cache before this harness exits.
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_SCAN_PLAN_DIR="${D1_DIR}" \
    cargo test -p iceberg --test interop_scan_plan
) > "${TMP}/merge-restored.log" 2>&1 \
  || { echo "FAIL: the RESTORED source did not pass D1 — see ${TMP}/merge-restored.log"; exit 1; }
echo "    MUTATION PASS: the merge-less mutant failed D1 (exit ${MUTANT_RC}); source restored"
echo "    byte-identically (md5 ${AFTER_MD5}) and re-verified GREEN"

echo ""
echo "==> DONE — scan-plan interop passed (row R148): plan_tasks split (offsets-aware big.parquet + fixed-size"
echo "     mid.parquet) + largestBinFirst bin-pack (target/lookback/cost = 4096/5/0, MoR delete bytes in the"
echo "     weight) + the ADJACENT-SPLIT MERGE (merge.parquet coalesces, gap.parquet's non-contiguous pair does"
echo "     not), BOTH directions, anti-circular hand-declared knobs, + the large-target / dropped-offset /"
echo "     merge sabotage legs and the production merge-removal mutation all closed."
