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
# REWRITE-TABLE-PATH interop harness (GAP_MATRIX row 137) — proving the Rust RewriteTablePath action
# (FULL-rewrite mode) agrees with Java 1.10.0's engine-agnostic iceberg-CORE RewriteTablePathUtil on a
# fixture carrying data + a PARQUET POSITION-delete + an EQUALITY-delete (every branch). The util is
# CORE (not Spark), so Java DRIVES the real rewrite — a fully BIDIRECTIONAL oracle.
#
# THE CHAIN (mirrors run-interop-delete-reachable.sh's structure + the fail-closed sentinel discipline):
#
#   1. RESET the temp dir.
#   2. JAVA builds the fixture at <dir>/table, DRIVES the REAL RewriteTablePathUtil over hand-declared
#      source/target prefixes, and emits two PATH-INDEPENDENT descriptors: java_graph.json (the SET of
#      rewritten TARGET locations, relativized to the target) + java_copy_plan.json (the (fromTag,toTag)
#      pairs with the per-class STAGED-vs-SOURCE direction encoded) + final.metadata.json. Anti-circular
#      (Java's REAL rewrite, never anything Rust wrote).
#   3. RUST (D2 + GEN) reads Java's table, runs the REAL Rust RewriteTablePath with the SAME prefixes +
#      its own staging dir, builds the SAME two descriptors, asserts they EQUAL Java's, and emits
#      rust_graph.json + rust_copy_plan.json.
#   4. JAVA (D1) re-judges: its descriptors == rust_graph.json AND rust_copy_plan.json (order-independent).
#   5. SABOTAGE: FLIP one copy-plan direction in the Rust descriptor (a STAGED entry's `from` retagged as
#      SOURCE) ⇒ Java's verify MUST fail. A sabotage that cannot be applied HARD-FAILS (never SKIP); a
#      sabotage that PASSES verify is a false-green and HARD-FAILS too.
#
# A divergence anywhere — a leaked source prefix, a miswired copy-plan direction, or a dropped/added
# entry — fails loudly. TEST-ONLY oracle; nothing here is in the offline `cargo test` gate; the temp dir
# under dev/java-interop/target/ is gitignored.
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64, the repo's
# pinned Rust toolchain. Run from anywhere; paths resolve relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-rewrite-table-path"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

echo "==> [1/5] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/5] Java: build the fixture + DRIVE RewriteTablePathUtil + emit java_graph.json + java_copy_plan.json + final.metadata.json"
(cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java \
  -Dexec.args=generate-interop-rewrite-table-path \
  -Dinterop.rewrite_table_path.dir="${TMP}" 2>&1)

echo "==> [3/5] Rust: run RewriteTablePath (D2), assert graph + copy-plan == Java, emit rust_graph.json + rust_copy_plan.json"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_REWRITE_TABLE_PATH_DIR="${TMP}" \
    cargo test -p iceberg --test interop_rewrite_table_path \
    rewrite_table_path_graph_and_copy_plan_match_java -- --exact --nocapture
)

echo "==> [4/5] Java (D1): re-judge — its descriptors == rust_graph.json AND rust_copy_plan.json"
# The verdict comes from the OUTPUT sentinel ("verify-interop-rewrite-table-path: 0 failures" with no
# FAIL line), never from mvn's exit code (machine-dependent for `exec:java` System.exit); `|| true`
# keeps set -e from aborting before the diagnostics are echoed (the run-interop-dv.sh fail-closed rule).
VERIFY_OUT="$(
  cd "${SCRIPT_DIR}"
  "${MVN}" -o -q compile exec:java \
    -Dexec.args=verify-interop-rewrite-table-path \
    -Dinterop.rewrite_table_path.dir="${TMP}" 2>&1
)" || true
echo "${VERIFY_OUT}"
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-rewrite-table-path: 0 failures'; then
  echo "==> FAILED — Java rejected the Rust descriptors (a leaked source prefix, a miswired copy-plan direction, or a dropped/added entry)."
  exit 1
fi

echo "==> [5/5] SABOTAGE: FLIP one copy-plan direction in the Rust descriptor (STAGED -> SOURCE) — Java's verify MUST fail closed"
# Sabotage runs against a SCRATCH copy (the clean ${TMP} is untouched). HARD-FAIL, never SKIP, if the
# corruption cannot be applied — a sabotage that did not actually corrupt anything proves nothing.
SCRATCH="${TMP}/sabotage_scratch"
rm -rf "${SCRATCH}"
mkdir -p "${SCRATCH}"
cp "${TMP}/java_graph.json" "${SCRATCH}/java_graph.json"
cp "${TMP}/java_copy_plan.json" "${SCRATCH}/java_copy_plan.json"
cp "${TMP}/rust_graph.json" "${SCRATCH}/rust_graph.json"

CLEAN_PLAN="$(cat "${TMP}/rust_copy_plan.json")"
if ! echo "${CLEAN_PLAN}" | grep -q 'STAGED:'; then
  echo "==> FAILED — cannot apply direction-flip sabotage: no 'STAGED:' tag present in the Rust copy-plan"
  echo "    (copy-plan was: ${CLEAN_PLAN}) — the per-class direction encoding changed; pin the sabotage to it."
  rm -rf "${SCRATCH}"
  exit 1
fi
# Flip the FIRST STAGED tag to SOURCE (a miswired copy-plan direction Java must catch).
SABOTAGED_PLAN="${CLEAN_PLAN/STAGED:/SOURCE:}"
if [ "${SABOTAGED_PLAN}" = "${CLEAN_PLAN}" ]; then
  echo "==> FAILED — the direction-flip sabotage did not change the copy-plan (no STAGED tag was flipped)."
  rm -rf "${SCRATCH}"
  exit 1
fi
printf '%s' "${SABOTAGED_PLAN}" > "${SCRATCH}/rust_copy_plan.json"
echo "    sabotaged Rust copy-plan (first STAGED -> SOURCE)"

# Java's verify MUST fail on the direction-flipped copy-plan. The verdict is the sentinel, never the
# exit code.
SAB_OUT="$(
  cd "${SCRIPT_DIR}"
  "${MVN}" -o -q compile exec:java \
    -Dexec.args=verify-interop-rewrite-table-path \
    -Dinterop.rewrite_table_path.dir="${SCRATCH}" 2>&1
)" || true
echo "${SAB_OUT}"
if echo "${SAB_OUT}" | grep -q 'verify-interop-rewrite-table-path: 0 failures' \
  && ! echo "${SAB_OUT}" | grep -q '^FAIL '; then
  echo "FAIL sabotage(direction-flip): Java's verify PASSED on a direction-flipped copy-plan — the comparison is vacuous"
  rm -rf "${SCRATCH}"
  exit 1
fi
if ! echo "${SAB_OUT}" | grep -q 'copy-PLANs differ'; then
  echo "FAIL sabotage(direction-flip): Java's verify failed but NOT via the 'copy-PLANs differ' path —"
  echo "    the direction-flip was caught by the wrong check (the sabotage is wrongly pinned)."
  rm -rf "${SCRATCH}"
  exit 1
fi
echo "PASS sabotage(direction-flip): a flipped copy-plan direction fails closed via the 'copy-PLANs differ' path"
rm -rf "${SCRATCH}"

echo "==> DONE — RewriteTablePath interop passed:"
echo "    D2 (Rust computes, asserts == Java): the Rust rewritten-path GRAPH + the (STAGED-vs-SOURCE) copy-plan == Java's real RewriteTablePathUtil rewrite."
echo "    D1 (Java judges Rust): Java's descriptors == rust_graph.json AND rust_copy_plan.json."
echo "    Sabotage: a flipped copy-plan direction fails closed (the copy-plan comparison is non-vacuous)."
