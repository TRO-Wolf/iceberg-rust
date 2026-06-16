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
# CONFLICT-VALIDATION write-action interop harness (increment C2) — DeleteFiles' SINGLE conflict axis
# `validateFilesExist()` proven against Java `StreamingDelete.validateFilesExist` /
# `ManifestFilterManager.failMissingDeletePaths`. DeleteFiles is the SIMPLEST write-action conflict
# unit (GAP_MATRIX row 93): one axis, no filter. Unlike the DATA-level harness
# (run-interop-write-data.sh), which proves ROWS survive a commit, this proves the conflict DECISION
# (ACCEPT vs REJECT) agrees across languages.
#
# THE AXIS (Java StreamingDelete / MergingSnapshotProducer): validateFilesExist() — reject the commit
# if a data file THIS delete targets was ALREADY REMOVED since the start snapshot. This is the SAME
# concurrent-removal shape as the C3 RowDelta files-exist axis (run-interop-rowdelta-conflict.sh),
# driven through DeleteFiles instead of RowDelta. StreamingDelete exposes no validateFromSnapshot
# (javap-confirmed); the loaded table is at S1, so the head IS the post-removal state.
#
# Two scenarios over an UNPARTITIONED V2 {id long, y long} table sharing one history (S0 appends
# f0+f1; S1 overwrite removes f0 adds f2):
#   scenario               | targets   | expected
#   same_file_reject       | f0 (gone) | REJECT  (the targeted file was concurrently removed)
#   different_file_accept  | f1 (live) | ACCEPT  (the targeted file still exists)
#
# The targeted path is DERIVED from the loaded table (reject = live@S0 minus live@S1 = removed f0;
# accept = live@S0 ∩ live@S1 = survivor f1) — anti-circular, no cross-engine path coupling.
#
# BOTH directions (the conflict decision is hand-declared identically on both sides, anti-circular):
#   D1 (Rust validates Java's table): Rust `register_table`s <scenario>/table and runs the symmetric
#     delete — `tests/interop_deletefiles_conflict.rs` test_rust_validates_java_*.
#   D2 (Java validates Rust's table): Java loads <scenario>/rust_table and runs the symmetric delete
#     — `DeleteFilesConflictOracle.verify`, "verify-interop-deletefiles-conflict: N failures".
#
# Test-only oracle; nothing here is in the offline cargo test gate; the temp dir is gitignored.
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64.
# Run from anywhere; paths resolve relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-deletefiles-conflict"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

echo "==> [1/7] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/7] Java: generate both scenario tables (<scenario>/table, S0 + concurrent S1)"
run_oracle -Dexec.args=generate-interop-deletefiles-conflict \
  -Dinterop.deletefiles_conflict.dir="${TMP}"

echo "==> [3/7] Rust: generate both scenario tables (<scenario>/rust_table) via the GEN test"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_DELETEFILES_CONFLICT_GEN_DIR="${TMP}" \
    cargo test -p iceberg --test interop_deletefiles_conflict \
    test_deletefiles_conflict_gen -- --nocapture
)

echo "==> [4/7] Java: verify-interop-deletefiles-conflict (D2 — Java validates the Rust tables)"
VERIFY_OUT="$(run_oracle -Dexec.args=verify-interop-deletefiles-conflict \
  -Dinterop.deletefiles_conflict.dir="${TMP}")" || true
echo "${VERIFY_OUT}"
# Fail-closed two ways: a per-check `^FAIL ` line OR absence of the `0 failures` sentinel.
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-deletefiles-conflict: 0 failures'; then
  echo "==> FAILED — D2 verify emitted a FAIL line or did not emit the '0 failures' sentinel."
  exit 1
fi

echo "==> [5/7] Rust: validate the Java tables (D1 — register + symmetric delete)"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_DELETEFILES_CONFLICT_DIR="${TMP}" \
    cargo test -p iceberg --test interop_deletefiles_conflict \
    test_rust_validates_java -- --nocapture
)

echo "==> [6/7] Sabotage battery — D2 verify must FAIL closed on a corrupted/wrong table"
# CONTROL: step 4's clean verify already passed, so a sabotage-fail is meaningful. Each sabotage runs
# against a SCRATCH copy (the clean TMP is untouched). HARD-FAIL, never SKIP, if a corruption cannot
# be applied (a sabotage that did not actually corrupt anything proves nothing).
#
# RESIDUE STRIP (load-bearing, the C4/C3 lesson): step 4's verify already ran `ops.commit(null, loaded)`
# against the REAL ${TMP}/<scenario>/rust_table, which wrote `v0.metadata.json` (and `v1.metadata.json`
# for the ACCEPT scenario whose delete committed). The sabotage scratch copies those tables, so a NAIVE
# re-verify would re-run `commit(null, loaded)` → write `v0.metadata.json` → AlreadyExistsException
# BEFORE the DeleteFiles validation ever runs, masking the semantic claim with a structural collision (a
# regression that broke the validation to always-ACCEPT would still "fail" here, at the v0 write). So
# build_scratch DELETES the v*.metadata.json residue, restoring each scratch to the pre-verify state
# (only the immutable numbered + final metadata remain) so the production validation path is REACHED.
# The `fresh-*.parquet` / new manifest residue does not collide (unique names).
build_scratch() {
  local scratch="${TMP}/sabotage_scratch"
  rm -rf "${scratch}"
  mkdir -p "${scratch}"
  cp -r "${TMP}/same_file_reject" "${scratch}/same_file_reject"
  cp -r "${TMP}/different_file_accept" "${scratch}/different_file_accept"
  # Strip the step-4 commit residue so `commit(null, loaded)` does not collide on v0.metadata.json.
  rm -f "${scratch}"/same_file_reject/rust_table/metadata/v*.metadata.json
  rm -f "${scratch}"/different_file_accept/rust_table/metadata/v*.metadata.json
  echo "${scratch}"
}

# (a) SEMANTIC: roll the REJECT scenario's loaded head BACK to S0 (the pre-removal root). Both
#     scenarios share ONE history (S0 appends f0+f1; S1 overwrite removes f0), so a table-swap is
#     vacuous (it would bring the identical history) — UNLIKE C3/C4 where the scenarios differ. Instead
#     we mutate `current-snapshot-id` + `refs.main` to S0 while KEEPING S1 in the snapshot list. Then the
#     verify's targetedPath(same_file_reject) still derives f0 (live@S0 minus live@S1 = the removed f0),
#     but the delete now commits against the S0 head where f0 is LIVE ⇒ ACCEPT, contradicting the
#     hand-declared REJECT. The pin is the SPECIFIC outcome-mismatch line ("outcome ACCEPT but expected
#     REJECT") — NOT merely the absence of `0 failures`: a structural collision / parse error would also
#     drop the sentinel but would NOT prove the validation re-derived ACCEPT. Requiring the exact
#     semantic line means this sabotage only passes when the production conflict decision was actually
#     reached and disagreed, so a validation regressed to always-ACCEPT is caught here.
#
#     HARD-FAIL never SKIP: if the rollback cannot be applied (S0/parent not found, fields absent), exit
#     non-zero — a sabotage that did not actually mutate the head proves nothing.
SCRATCH="$(build_scratch)"
REJECT_FINAL="${SCRATCH}/same_file_reject/rust_table/metadata/final.metadata.json"
if [[ ! -f "${REJECT_FINAL}" ]]; then
  echo "==> FAILED — cannot apply rollback sabotage: ${REJECT_FINAL} absent (GEN gated?)"
  exit 1
fi
python3 - "${REJECT_FINAL}" <<'PY' || { echo "==> FAILED — rollback sabotage could not mutate the head"; exit 1; }
import json, sys
path = sys.argv[1]
with open(path) as fh:
    meta = json.load(fh)
snaps = meta.get("snapshots", [])
# S0 = the root snapshot (no parent); S1 = the current overwrite head.
root = next((s for s in snaps if s.get("parent-snapshot-id") is None), None)
if root is None:
    sys.exit("no root (S0) snapshot to roll back to")
s0 = root["snapshot-id"]
if meta.get("current-snapshot-id") == s0:
    sys.exit("head already at S0 — rollback would be a no-op")
# Roll the head back to S0 while KEEPING S1 in the snapshot list (so live@S0 minus live@S1 still finds f0).
meta["current-snapshot-id"] = s0
refs = meta.get("refs", {})
if "main" not in refs:
    sys.exit("no main ref to roll back")
refs["main"]["snapshot-id"] = s0
with open(path, "w") as fh:
    json.dump(meta, fh)
print("rolled same_file_reject head back to S0 (%d) — f0 is live again at the head" % s0)
PY
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-deletefiles-conflict \
  -Dinterop.deletefiles_conflict.dir="${SCRATCH}")" || true
if echo "${SAB_OUT}" | grep -q 'verify-interop-deletefiles-conflict: 0 failures'; then
  echo "FAIL sabotage(semantic-rollback): verify PASSED on a head rolled back to S0 where REJECT was expected — pin is vacuous"
  echo "${SAB_OUT}"
  exit 1
fi
# Require the SPECIFIC semantic-mismatch line for the swapped scenario, proving the production validation
# re-derived ACCEPT (not a structural collision that died before the conflict decision).
if ! echo "${SAB_OUT}" | grep -q 'deletefiles-conflict-d2\[same_file_reject\]: outcome ACCEPT but expected REJECT'; then
  echo "FAIL sabotage(semantic-rollback): verify failed but NOT via the outcome-mismatch path —"
  echo "  the validation was never reached (structural collision/parse error masks the semantic claim)."
  echo "${SAB_OUT}"
  exit 1
fi
echo "PASS sabotage(semantic-rollback): a head rolled back to S0 fails closed via the outcome-mismatch path (validation re-derived ACCEPT where REJECT was expected)"

# (b) STRUCTURAL: truncate the REJECT scenario's final.metadata.json so it no longer parses ⇒ the
#     verify's load branch errors ⇒ FAIL closed.
SCRATCH="$(build_scratch)"
RUST_FINAL="${SCRATCH}/same_file_reject/rust_table/metadata/final.metadata.json"
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
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-deletefiles-conflict \
  -Dinterop.deletefiles_conflict.dir="${SCRATCH}")" || true
if echo "${SAB_OUT}" | grep -q 'verify-interop-deletefiles-conflict: 0 failures'; then
  echo "FAIL sabotage(truncate): verify PASSED on a truncated metadata file — NOT fail-closed"
  echo "${SAB_OUT}"
  exit 1
fi
# Require the SPECIFIC load/parse-error path for the truncated scenario (the verify's catch branch), so
# the truncate proves the structural-load guard and does not silently ride on the semantic line.
if ! echo "${SAB_OUT}" | grep -q 'deletefiles-conflict-d2\[same_file_reject\]: unexpected error running the delete_files'; then
  echo "FAIL sabotage(truncate): verify failed but NOT via the load/parse-error path — the truncation"
  echo "  did not actually break the metadata load (corruption proved nothing)."
  echo "${SAB_OUT}"
  exit 1
fi
echo "PASS sabotage(truncate): a truncated metadata file fails closed via the load/parse-error path"
rm -rf "${TMP}/sabotage_scratch"

echo "==> [7/7] DONE — DeleteFiles conflict-validation interop passed."
echo "    Scenarios: same_file_reject (REJECT) + different_file_accept (ACCEPT)"
echo "    Axis: files-exist (validateFilesExist) — the SIMPLEST write-action conflict unit"
echo "    Directions: D1 (Rust validates Java) + D2 (Java validates Rust)"
echo "    Sabotage battery: semantic-rollback (head→S0, pinned to the outcome-mismatch line, validation reached)"
echo "                      + structural-truncate (pinned to the load/parse-error line); control passed in step 4"
