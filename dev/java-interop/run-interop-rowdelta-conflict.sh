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
# CONFLICT-VALIDATION write-action interop harness (increment C3) — RowDelta's THREE conflict axes
# proven against Java `BaseRowDelta` on the conflict DECISION (ACCEPT vs REJECT). RowDelta is the
# RICHEST write-action conflict unit (GAP_MATRIX row 94): the merge-on-read commit (DATA + DELETE
# files in one snapshot) with three independent conflict checks. The THIRD conflict-validation interop
# slice (after run-interop-overwrite-conflict.sh / run-interop-replace-partitions-conflict.sh) — the
# residue named identically in every GAP_MATRIX write-action cell ("conflict-validation paths NOT
# covered"). Unlike the DATA-level harness (run-interop-write-data.sh), which proves ROWS survive a
# commit, this proves the conflict DECISION (ACCEPT vs REJECT) agrees across languages.
#
# THE THREE AXES (Java BaseRowDelta / MergingSnapshotProducer):
#   1. validateNoConflictingDataFiles() + conflictDetectionFilter(F) — concurrent DATA add matches F
#      (the inclusive-metrics axis, EXACTLY the C1 OverwriteFiles shape).
#   2. validateNoConflictingDeleteFiles() + filter — concurrent DELETE add (an EQUALITY delete keyed
#      on `y`, so its real parquet carries the `y` bounds) matches F (RowDelta-specific).
#   3. validateDataFilesExist(paths) — concurrent OVERWRITE removed the referenced data file (the
#      {OVERWRITE} default op set, so it rejects WITHOUT validateDeletedFiles()).
#
# Six scenarios over an UNPARTITIONED V2 {id long, y long} table (two per axis: reject + accept):
#   scenario                | axis            | S1 concurrent commit                 | expected
#   data_conflict_reject    | data-conflict   | fast-append DATA y[60,70] (overlaps) | REJECT
#   data_conflict_accept    | data-conflict   | fast-append DATA y[10,20] (excluded) | ACCEPT
#   delete_conflict_reject  | delete-conflict | row_delta eq-delete on y[60,70]      | REJECT
#   delete_conflict_accept  | delete-conflict | row_delta eq-delete on y[10,20]      | ACCEPT
#   files_exist_reject      | files-exist     | overwrite removes f0 add f2          | REJECT
#   files_exist_accept      | files-exist     | overwrite removes f0 add f2          | ACCEPT
#
# The files_exist scenarios share the same history (S0 appends f0+f1; S1 overwrite removes f0 adds f2);
# the reject scenario references the REMOVED f0, the accept references the SURVIVING f1, both derived
# from the loaded table (anti-circular, no cross-engine path coupling).
#
# BOTH directions (the conflict decision is hand-declared identically on both sides, anti-circular):
#   D1 (Rust validates Java's table): Rust `register_table`s <scenario>/table and runs the symmetric
#     row delta — `tests/interop_rowdelta_conflict.rs` test_rust_validates_java_*.
#   D2 (Java validates Rust's table): Java loads <scenario>/rust_table and runs the symmetric row delta
#     — `RowDeltaConflictOracle.verify`, "verify-interop-rowdelta-conflict: N failures".
#
# Test-only oracle; nothing here is in the offline cargo test gate; the temp dir is gitignored.
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64.
# Run from anywhere; paths resolve relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-rowdelta-conflict"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

echo "==> [1/7] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/7] Java: generate all six scenario tables (<scenario>/table, S0 + concurrent S1)"
run_oracle -Dexec.args=generate-interop-rowdelta-conflict \
  -Dinterop.rowdelta_conflict.dir="${TMP}"

echo "==> [3/7] Rust: generate all six scenario tables (<scenario>/rust_table) via the GEN test"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_ROWDELTA_CONFLICT_GEN_DIR="${TMP}" \
    cargo test -p iceberg --test interop_rowdelta_conflict \
    test_rowdelta_conflict_gen -- --nocapture
)

echo "==> [4/7] Java: verify-interop-rowdelta-conflict (D2 — Java validates the Rust tables)"
VERIFY_OUT="$(run_oracle -Dexec.args=verify-interop-rowdelta-conflict \
  -Dinterop.rowdelta_conflict.dir="${TMP}")" || true
echo "${VERIFY_OUT}"
# Fail-closed two ways: a per-check `^FAIL ` line OR absence of the `0 failures` sentinel.
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-rowdelta-conflict: 0 failures'; then
  echo "==> FAILED — D2 verify emitted a FAIL line or did not emit the '0 failures' sentinel."
  exit 1
fi

echo "==> [5/7] Rust: validate the Java tables (D1 — register + symmetric row delta)"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_ROWDELTA_CONFLICT_DIR="${TMP}" \
    cargo test -p iceberg --test interop_rowdelta_conflict \
    test_rust_validates_java -- --nocapture
)

echo "==> [6/7] Sabotage battery — D2 verify must FAIL closed on a corrupted/wrong table"
# CONTROL: step 4's clean verify already passed, so a sabotage-fail is meaningful. Each sabotage runs
# against a SCRATCH copy (the clean TMP is untouched). HARD-FAIL, never SKIP, if a corruption cannot
# be applied (a sabotage that did not actually corrupt anything proves nothing).
#
# RESIDUE STRIP (load-bearing, the C4 lesson): step 4's verify already ran `ops.commit(null, loaded)`
# against the REAL ${TMP}/<scenario>/rust_table, which wrote `v0.metadata.json` (and `v1.metadata.json`
# for the ACCEPT scenarios whose row delta committed). The sabotage scratch copies those tables, so a
# NAIVE re-verify would re-run `commit(null, loaded)` → write `v0.metadata.json` → AlreadyExistsException
# BEFORE the RowDelta validation ever runs, masking the semantic claim with a structural collision (a
# regression that broke the validation to always-ACCEPT would still "fail" here, at the v0 write). So
# build_scratch DELETES the v*.metadata.json residue, restoring each scratch to the pre-verify state
# (only the immutable numbered + final metadata remain) so the production validation path is REACHED.
# The `fresh-*.parquet` / new manifest residue does not collide (unique names).
build_scratch() {
  local scratch="${TMP}/sabotage_scratch"
  rm -rf "${scratch}"
  mkdir -p "${scratch}"
  for scenario in data_conflict_reject data_conflict_accept \
    delete_conflict_reject delete_conflict_accept \
    files_exist_reject files_exist_accept; do
    cp -r "${TMP}/${scenario}" "${scratch}/${scenario}"
    # Strip the step-4 commit residue so `commit(null, loaded)` does not collide on v0.metadata.json.
    rm -f "${scratch}/${scenario}"/rust_table/metadata/v*.metadata.json
  done
  echo "${scratch}"
}

# (a) SEMANTIC: replace the REJECT scenario's table with the no-conflict (ACCEPT) table for the SAME
#     axis. The validation RE-DERIVES ACCEPT through the production path, contradicting the hand-declared
#     REJECT ⇒ verify must FAIL closed. (The S3-class "feed a genuinely-wrong table" mutation.) The pin
#     is the SPECIFIC outcome-mismatch line ("outcome ACCEPT but expected REJECT") — NOT merely the
#     absence of `0 failures`: a structural AlreadyExistsException / parse error would also drop the
#     sentinel, but it would NOT prove the validation re-derived ACCEPT. Requiring the exact semantic
#     line means this sabotage only passes when the production conflict decision was actually reached and
#     disagreed, so a validation regressed to always-ACCEPT is caught here.
SCRATCH="$(build_scratch)"
rm -rf "${SCRATCH}/data_conflict_reject/rust_table"
cp -r "${TMP}/data_conflict_accept/rust_table" "${SCRATCH}/data_conflict_reject/rust_table"
# The copied-in table also carries step-4 v*.metadata.json residue — strip it too (the swap brought the
# OTHER scenario's post-verify tree, which has v0+v1) so commit(null,loaded) reaches validation.
rm -f "${SCRATCH}"/data_conflict_reject/rust_table/metadata/v*.metadata.json
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-rowdelta-conflict \
  -Dinterop.rowdelta_conflict.dir="${SCRATCH}")" || true
if echo "${SAB_OUT}" | grep -q 'verify-interop-rowdelta-conflict: 0 failures'; then
  echo "FAIL sabotage(semantic-swap): verify PASSED on a no-conflict table where REJECT was expected — pin is vacuous"
  echo "${SAB_OUT}"
  exit 1
fi
# Require the SPECIFIC semantic-mismatch line for the swapped scenario, proving the production validation
# re-derived ACCEPT (not a structural collision that died before the conflict decision).
if ! echo "${SAB_OUT}" | grep -q 'rowdelta-conflict-d2\[data_conflict_reject\]: outcome ACCEPT but expected REJECT'; then
  echo "FAIL sabotage(semantic-swap): verify failed but NOT via the outcome-mismatch path —"
  echo "  the validation was never reached (structural collision/parse error masks the semantic claim)."
  echo "${SAB_OUT}"
  exit 1
fi
echo "PASS sabotage(semantic-swap): a no-conflict table fails closed via the outcome-mismatch path (validation re-derived ACCEPT where REJECT was expected)"

# (b) STRUCTURAL: truncate the REJECT scenario's final.metadata.json so it no longer parses ⇒ the
#     verify's load branch errors ⇒ FAIL closed.
SCRATCH="$(build_scratch)"
RUST_FINAL="${SCRATCH}/data_conflict_reject/rust_table/metadata/final.metadata.json"
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
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-rowdelta-conflict \
  -Dinterop.rowdelta_conflict.dir="${SCRATCH}")" || true
if echo "${SAB_OUT}" | grep -q 'verify-interop-rowdelta-conflict: 0 failures'; then
  echo "FAIL sabotage(truncate): verify PASSED on a truncated metadata file — NOT fail-closed"
  echo "${SAB_OUT}"
  exit 1
fi
# Require the SPECIFIC load/parse-error path for the truncated scenario (the verify's catch branch), so
# the truncate proves the structural-load guard and does not silently ride on the semantic line.
if ! echo "${SAB_OUT}" | grep -q 'rowdelta-conflict-d2\[data_conflict_reject\]: unexpected error running the row_delta'; then
  echo "FAIL sabotage(truncate): verify failed but NOT via the load/parse-error path — the truncation"
  echo "  did not actually break the metadata load (corruption proved nothing)."
  echo "${SAB_OUT}"
  exit 1
fi
echo "PASS sabotage(truncate): a truncated metadata file fails closed via the load/parse-error path"
rm -rf "${TMP}/sabotage_scratch"

echo "==> [7/7] DONE — RowDelta conflict-validation interop passed."
echo "    Scenarios: data_conflict_{reject,accept} + delete_conflict_{reject,accept} + files_exist_{reject,accept}"
echo "    Axes: data-conflict (validateNoConflictingDataFiles) + delete-conflict (validateNoConflictingDeleteFiles)"
echo "          + files-exist (validateDataFilesExist) — all three proven both directions"
echo "    Directions: D1 (Rust validates Java) + D2 (Java validates Rust)"
echo "    Sabotage battery: semantic-swap (pinned to the outcome-mismatch line, validation reached)"
echo "                      + structural-truncate (pinned to the load/parse-error line); control passed in step 4"
