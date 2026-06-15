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
# CONFLICT-VALIDATION write-action interop harness (increment C1) — OverwriteFiles'
# `validateNoConflictingData()` + `conflictDetectionFilter` proven against Java
# `BaseOverwriteFiles.validate` → `validateAddedDataFiles`. The FIRST conflict-validation interop
# slice (the residue named identically in every GAP_MATRIX write-action cell: "conflict-validation
# paths NOT covered"). Unlike the DATA-level harness (run-interop-write-data.sh), which proves ROWS
# survive a commit, this proves the conflict DECISION (ACCEPT vs REJECT) agrees across languages.
#
# Three scenarios over an UNPARTITIONED V2 {id long, y long} table (S0 base + S1 concurrent add):
#   scenario        | filter   | concurrent y | expected
#   ge50_overlap    | y >= 50  | [60,70]      | REJECT
#   ge50_excluded   | y >= 50  | [10,20]      | ACCEPT
#   nofilter_any    | (none)   | [60,70]      | REJECT
#
# BOTH directions (the conflict decision is hand-declared identically on both sides, anti-circular):
#   D1 (Rust validates Java's table): Rust `register_table`s <scenario>/table and runs the symmetric
#     overwrite — `tests/interop_overwrite_conflict.rs` test_rust_validates_java_*.
#   D2 (Java validates Rust's table): Java loads <scenario>/rust_table and runs the symmetric
#     overwrite — `OverwriteConflictOracle.verify`, "verify-interop-overwrite-conflict: N failures".
#
# Test-only oracle; nothing here is in the offline cargo test gate; the temp dir is gitignored.
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64.
# Run from anywhere; paths resolve relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-overwrite-conflict"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

echo "==> [1/7] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/7] Java: generate all three scenario tables (<scenario>/table, S0 + concurrent S1)"
run_oracle -Dexec.args=generate-interop-overwrite-conflict \
  -Dinterop.overwrite_conflict.dir="${TMP}"

echo "==> [3/7] Rust: generate all three scenario tables (<scenario>/rust_table) via the GEN test"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_OVERWRITE_CONFLICT_GEN_DIR="${TMP}" \
    cargo test -p iceberg --test interop_overwrite_conflict \
    test_overwrite_conflict_gen -- --nocapture
)

echo "==> [4/7] Java: verify-interop-overwrite-conflict (D2 — Java validates the Rust tables)"
VERIFY_OUT="$(run_oracle -Dexec.args=verify-interop-overwrite-conflict \
  -Dinterop.overwrite_conflict.dir="${TMP}")" || true
echo "${VERIFY_OUT}"
# Fail-closed two ways: a per-check `^FAIL ` line OR absence of the `0 failures` sentinel.
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-overwrite-conflict: 0 failures'; then
  echo "==> FAILED — D2 verify emitted a FAIL line or did not emit the '0 failures' sentinel."
  exit 1
fi

echo "==> [5/7] Rust: validate the Java tables (D1 — register + symmetric overwrite)"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_OVERWRITE_CONFLICT_DIR="${TMP}" \
    cargo test -p iceberg --test interop_overwrite_conflict \
    test_rust_validates_java -- --nocapture
)

echo "==> [6/7] Sabotage battery — D2 verify must FAIL closed on a corrupted/wrong table"
# CONTROL: step 4's clean verify already passed, so a sabotage-fail is meaningful. Each sabotage runs
# against a SCRATCH copy (the clean TMP is untouched). HARD-FAIL, never SKIP, if a corruption cannot
# be applied (a sabotage that did not actually corrupt anything proves nothing).
build_scratch() {
  local scratch="${TMP}/sabotage_scratch"
  rm -rf "${scratch}"
  mkdir -p "${scratch}"
  cp -r "${TMP}/ge50_overlap" "${scratch}/ge50_overlap"
  cp -r "${TMP}/ge50_excluded" "${scratch}/ge50_excluded"
  cp -r "${TMP}/nofilter_any" "${scratch}/nofilter_any"
  echo "${scratch}"
}

# (a) SEMANTIC: replace the REJECT scenario's table with the no-conflict (ACCEPT) table. The
#     validation RE-DERIVES ACCEPT through the production path, contradicting the hand-declared
#     REJECT ⇒ verify must FAIL closed. (The S3-class "feed a genuinely-wrong table" mutation.)
SCRATCH="$(build_scratch)"
rm -rf "${SCRATCH}/ge50_overlap/rust_table"
cp -r "${TMP}/ge50_excluded/rust_table" "${SCRATCH}/ge50_overlap/rust_table"
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-overwrite-conflict \
  -Dinterop.overwrite_conflict.dir="${SCRATCH}")" || true
if echo "${SAB_OUT}" | grep -q 'verify-interop-overwrite-conflict: 0 failures'; then
  echo "FAIL sabotage(semantic-swap): verify PASSED on a no-conflict table where REJECT was expected — pin is vacuous"
  exit 1
fi
echo "PASS sabotage(semantic-swap): a no-conflict table fails closed where REJECT was expected"

# (b) STRUCTURAL: truncate the REJECT scenario's final.metadata.json so it no longer parses ⇒ the
#     verify's load branch errors ⇒ FAIL closed.
SCRATCH="$(build_scratch)"
RUST_FINAL="${SCRATCH}/ge50_overlap/rust_table/metadata/final.metadata.json"
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
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-overwrite-conflict \
  -Dinterop.overwrite_conflict.dir="${SCRATCH}")" || true
if echo "${SAB_OUT}" | grep -q 'verify-interop-overwrite-conflict: 0 failures'; then
  echo "FAIL sabotage(truncate): verify PASSED on a truncated metadata file — NOT fail-closed"
  exit 1
fi
echo "PASS sabotage(truncate): a truncated metadata file fails closed"
rm -rf "${TMP}/sabotage_scratch"

echo "==> [7/7] DONE — OverwriteFiles conflict-validation interop passed."
echo "    Scenarios: ge50_overlap (REJECT) + ge50_excluded (ACCEPT) + nofilter_any (REJECT)"
echo "    Directions: D1 (Rust validates Java) + D2 (Java validates Rust)"
echo "    Sabotage battery: semantic-swap + structural-truncate, both fail closed (control passed in step 4)"
