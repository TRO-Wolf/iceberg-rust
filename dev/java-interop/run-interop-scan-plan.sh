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
# SCAN-PLAN (planTasks) interop harness — GAP_MATRIX row 146. Proves the Rust `TableScan::plan_tasks`
# produces the SAME bin-packed CombinedScanTask GROUPS as Java's REAL
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
#
# THE COMPARISON: each group is a SORTED set of member keys "(basename,start,length)"; the plan is the
# MULTISET of per-group member-key sets + the group count. Both engines plan the SAME on-disk table within a
# direction (so split offsets, hence start/length, are byte-identical); group emission ORDER is NOT compared.
#
# THE CHAIN (both directions + a fail-closed sabotage):
#   [1/6] Reset the temp dir.
#   [2/6] Java GENERATE (Direction 1): write the table; run the REAL Java planTasks; emit java_scan_plan.json.
#   [3/6] Rust GEN (Direction 2): write the SAME logical table to <dir>/rust_table via the production write
#         path + emit rust_scan_plan.json (Rust's own plan_tasks plan) + a self-check that big.parquet split.
#   [4/6] Java VERIFY (Direction 2): load the RUST-written table, run the REAL Java planTasks, assert the
#         multiset of per-group member-key sets == rust_scan_plan.json. A FAIL is a real write/grouping finding.
#   [5/6] Rust D1: load the JAVA-written table, run plan_tasks, assert == Java's java_scan_plan.json.
#   [6/6] SABOTAGE (fail-closed), two load-bearing legs: (1) Java re-plans the Java table with a much
#         LARGER target (target*1024), forcing the groups to MERGE, and asserts the plan DIVERGES from
#         java_scan_plan.json (11→1); (2) big.parquet's split-offsets are DROPPED, flipping the offsets-aware
#         split to fixed-size windows (8→2). HARD-FAILS if EITHER leg leaves the grouping unchanged (vacuous).
#
# This is a TEST-ONLY ORACLE (a dev tool) — NOT part of the shipped Rust library, NOT part of the offline
# `cargo test` gate (it needs Java + Maven). Nothing binary is committed; the temp dir under
# dev/java-interop/target/ is gitignored.
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

echo "==> [1/6] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${D1_DIR}" "${GEN_DIR}"

echo "==> [2/6] Java GENERATE (Direction 1): write the table + emit java_scan_plan.json"
run_oracle -Dexec.args=generate-interop-scan-plan \
  -Dinterop.scan_plan.dir="${D1_DIR}"
test -f "${D1_DIR}/java_scan_plan.json" \
  || { echo "FAIL: java_scan_plan.json not produced"; exit 1; }
echo "    java_scan_plan.json produced OK"

echo "==> [3/6] Rust GEN (Direction 2): write the SAME table to ${GEN_DIR}/rust_table + emit rust_scan_plan.json"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_SCAN_PLAN_GEN_DIR="${GEN_DIR}" \
    cargo test -p iceberg --test interop_scan_plan -- --nocapture
)
test -f "${GEN_DIR}/rust_table/metadata/final.metadata.json" \
  || { echo "FAIL: Rust GEN did not produce rust_table/metadata/final.metadata.json"; exit 1; }
test -f "${GEN_DIR}/rust_table/rust_scan_plan.json" \
  || { echo "FAIL: Rust GEN did not produce rust_table/rust_scan_plan.json"; exit 1; }

echo "==> [4/6] Java VERIFY (Direction 2): Java's REAL planTasks over the RUST-written table"
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

echo "==> [5/6] Rust D1: Rust's plan_tasks over the JAVA-written table == Java's plan"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_SCAN_PLAN_DIR="${D1_DIR}" \
    cargo test -p iceberg --test interop_scan_plan -- --nocapture
)
echo "    Direction-1 OK — Rust's plan_tasks over the Java table matches Java"

echo ""
echo "==> [6/6] SABOTAGE BATTERY (fail-closed): two corruptions over the Java table, each must DIVERGE"
echo "     from java_scan_plan.json — (a) a much larger target (groups MERGE), (b) big.parquet's split"
echo "     offsets DROPPED (offsets-aware → fixed-size windows). HARD-FAILS if EITHER leg is unchanged."
SABOTAGE_OUT="$(
  run_oracle -Dexec.args=sabotage-interop-scan-plan \
    -Dinterop.scan_plan.dir="${D1_DIR}"
)"
echo "${SABOTAGE_OUT}"
if echo "${SABOTAGE_OUT}" | grep -q '^FAIL ' \
    || echo "${SABOTAGE_OUT}" | grep -q 'VACUOUS' \
    || ! echo "${SABOTAGE_OUT}" | grep -q 'fail-closed confirmed'; then
  echo "==> SABOTAGE FAILED — the perturbed target did NOT change the plan; the target is not load-bearing."
  exit 1
fi
echo "    SABOTAGE PASS: the large-target re-pack + dropped-split-offset legs both diverged (fail-closed)"

echo ""
echo "==> DONE — scan-plan interop passed (row 146): plan_tasks split (offsets-aware big.parquet + fixed-size"
echo "     mid.parquet) + largestBinFirst bin-pack (target/lookback/cost = 4096/5/0, MoR delete bytes in the"
echo "     weight), BOTH directions, anti-circular hand-declared knobs, + the large-target/dropped-offset sabotage closed."
