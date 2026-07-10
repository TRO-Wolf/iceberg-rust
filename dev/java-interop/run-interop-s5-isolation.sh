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
# ENGINE_CONTRACT §5 isolation-recipe interop harness — the three §5 cells whose covering conflict
# scenario was previously Rust-unit-level only (the named residue of the 2026-07-09 DRAFT→NORMATIVE
# flip): COW/snapshot (OverwriteFiles rewrite + validateNoConflictingDeletes),
# dynamic-overwrite/snapshot (ReplacePartitions + validateNoConflictingDeletes), and static
# overwrite-by-filter (overwriteByRowFilter; snapshot = validateNoConflictingDeletes, serializable
# = + validateNoConflictingData with the ROW FILTER as the DEFAULT conflict filter).
#
# Eight scenarios (a reject + an accept false-positive guard per cell) — see
# `tests/interop_s5_isolation_conflict.rs` / `S5IsolationOracle` for the full contract table.
#
# BOTH directions (the conflict decision is hand-declared identically on both sides, anti-circular):
#   D1 (Rust validates Java's table): Rust `register_table`s <scenario>/table and runs the cell's
#     symmetric operation — `tests/interop_s5_isolation_conflict.rs` test_rust_validates_java_*.
#   D2 (Java validates Rust's table): Java loads <scenario>/rust_table and runs the same operation
#     — `S5IsolationOracle.verify`, "verify-interop-s5-isolation: N failures".
#
# Test-only oracle; nothing here is in the offline cargo test gate; the temp dir is gitignored.
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64.
# Run from anywhere; paths resolve relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-s5-isolation"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

SCENARIOS=(
  cow_delete_on_rewritten
  cow_delete_on_other
  dyn_delete_in_replaced
  dyn_delete_in_other
  byfilter_delete_matching
  byfilter_delete_excluded
  byfilter_data_matching
  byfilter_data_excluded
)

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

echo "==> [1/7] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/7] Java: generate all eight scenario tables (<scenario>/table, S0 + concurrent S1)"
run_oracle -Dexec.args=generate-interop-s5-isolation \
  -Dinterop.s5_isolation.dir="${TMP}"

echo "==> [3/7] Rust: generate all eight scenario tables (<scenario>/rust_table) via the GEN test"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_S5_ISOLATION_GEN_DIR="${TMP}" \
    cargo test -p iceberg --test interop_s5_isolation_conflict \
    test_s5_isolation_gen -- --nocapture
)

echo "==> [4/7] Java: verify-interop-s5-isolation (D2 — Java validates the Rust tables)"
VERIFY_OUT="$(run_oracle -Dexec.args=verify-interop-s5-isolation \
  -Dinterop.s5_isolation.dir="${TMP}")" || true
echo "${VERIFY_OUT}"
# Fail-closed two ways: a per-check `^FAIL ` line OR absence of the `0 failures` sentinel.
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-s5-isolation: 0 failures'; then
  echo "==> FAILED — D2 verify emitted a FAIL line or did not emit the '0 failures' sentinel."
  exit 1
fi

echo "==> [5/7] Rust: validate the Java tables (D1 — register + symmetric §5 operation)"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_S5_ISOLATION_DIR="${TMP}" \
    cargo test -p iceberg --test interop_s5_isolation_conflict \
    test_rust_validates_java -- --nocapture
)

echo "==> [6/7] Sabotage battery — D2 verify must FAIL closed on a corrupted/wrong table"
# CONTROL: step 4's clean verify already passed, so a sabotage-fail is meaningful. Each sabotage
# runs against a SCRATCH copy (the clean TMP is untouched). HARD-FAIL, never SKIP, if a corruption
# cannot be applied (a sabotage that did not actually corrupt anything proves nothing).
build_scratch() {
  local scratch="${TMP}/sabotage_scratch"
  rm -rf "${scratch}"
  mkdir -p "${scratch}"
  local scenario
  for scenario in "${SCENARIOS[@]}"; do
    cp -r "${TMP}/${scenario}" "${scratch}/${scenario}"
  done
  echo "${scratch}"
}

# (a) SEMANTIC: replace the COW REJECT scenario's table with the no-conflict (ACCEPT) table. The
#     validation RE-DERIVES ACCEPT through the production path, contradicting the hand-declared
#     REJECT ⇒ verify must FAIL closed. (The S3-class "feed a genuinely-wrong table" mutation.)
SCRATCH="$(build_scratch)"
rm -rf "${SCRATCH}/cow_delete_on_rewritten/rust_table"
cp -r "${TMP}/cow_delete_on_other/rust_table" "${SCRATCH}/cow_delete_on_rewritten/rust_table"
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-s5-isolation \
  -Dinterop.s5_isolation.dir="${SCRATCH}")" || true
if echo "${SAB_OUT}" | grep -q 'verify-interop-s5-isolation: 0 failures'; then
  echo "FAIL sabotage(semantic-swap): verify PASSED on a no-conflict table where REJECT was expected — pin is vacuous"
  exit 1
fi
echo "PASS sabotage(semantic-swap): a no-conflict table fails closed where REJECT was expected"

# (b) STRUCTURAL: truncate the by-filter REJECT scenario's final.metadata.json so it no longer
#     parses ⇒ the verify's load branch errors ⇒ FAIL closed.
SCRATCH="$(build_scratch)"
RUST_FINAL="${SCRATCH}/byfilter_data_matching/rust_table/metadata/final.metadata.json"
if [[ ! -f "${RUST_FINAL}" ]]; then
  echo "==> FAILED — cannot apply truncate sabotage: ${RUST_FINAL} absent (GEN gated?)"
  exit 1
fi
SIZE="$(stat -c%s "${RUST_FINAL}")"
if (( SIZE <= 60 )); then
  echo "==> FAILED — cannot apply truncate sabotage: ${RUST_FINAL} is only ${SIZE} bytes"
  exit 1
fi
head -c "$(( SIZE - 60 ))" "${RUST_FINAL}" > "${RUST_FINAL}.tmp"
mv "${RUST_FINAL}.tmp" "${RUST_FINAL}"
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-s5-isolation \
  -Dinterop.s5_isolation.dir="${SCRATCH}")" || true
if echo "${SAB_OUT}" | grep -q 'verify-interop-s5-isolation: 0 failures'; then
  echo "FAIL sabotage(truncate): verify PASSED on a truncated metadata file — NOT fail-closed"
  exit 1
fi
echo "PASS sabotage(truncate): a truncated metadata file fails closed"
rm -rf "${TMP}/sabotage_scratch"

echo "==> [7/7] DONE — ENGINE_CONTRACT §5 isolation-recipe interop passed."
echo "    Cells: COW/snapshot + dynamic-overwrite/snapshot + by-filter snapshot + by-filter serializable"
echo "    Scenarios: 4 REJECT + 4 ACCEPT (false-positive guards), decisions agree across engines"
echo "    Directions: D1 (Rust validates Java) + D2 (Java validates Rust)"
echo "    Sabotage battery: semantic-swap + structural-truncate, both fail closed (control passed in step 4)"
