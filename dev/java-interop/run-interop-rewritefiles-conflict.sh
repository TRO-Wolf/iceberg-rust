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
# CONFLICT-VALIDATION write-action interop harness (increment C5) — RewriteFiles's conflict validation
# proven against Java `BaseRewriteFiles.validate` on the conflict DECISION (ACCEPT vs REJECT). RewriteFiles
# is the MECHANICALLY most complex write-action conflict unit (GAP_MATRIX row 95): the seq-preservation +
# position-vs-equality-delete nuance. The FIFTH conflict-validation interop slice (after
# run-interop-overwrite-conflict.sh / run-interop-replace-partitions-conflict.sh /
# run-interop-rowdelta-conflict.sh / run-interop-deletefiles-conflict.sh) — the residue named identically
# in every GAP_MATRIX write-action cell ("conflict-validation paths NOT covered"). Unlike the DATA-level
# harness (run-interop-write-data.sh), which proves ROWS survive a commit, this proves the conflict
# DECISION (ACCEPT vs REJECT) agrees across languages.
#
# THE RULE (Java BaseRewriteFiles.validate L135-142, the shared validateNoNewDeletesForDataFiles): a
# rewrite REPLACES data files. A CONCURRENT DELETE file added since the rewrite's starting snapshot that
# APPLIES to one of the replaced files is a conflict (the rewrite would resurrect deleted rows) — UNLESS
# the rewrite PRESERVED the replaced file's data sequence number (the 3-arg rewriteFiles(toDelete, toAdd,
# startingDataSequenceNumber)), in which case a concurrently-added EQUALITY delete is IGNORED (it still
# applies to the rewritten file via the preserved seq) but a POSITION delete is ALWAYS fatal (path-scoped).
# In Rust: ignore_equality_deletes = data_sequence_number.is_some().
#
# Four scenarios over a PARTITIONED V2 {x long, y long, z long} (identity(x)) table — S0 appends A (x=0) +
# B (x=1); S1 is a concurrent row_delta adding a delete file; the symmetric rewrite replaces A with A':
#   scenario                  | preserve seq | S1 concurrent delete            | expected
#   no_seq_eq_delete_reject   | NO           | EQUALITY delete on A (x=0)       | REJECT
#   seq_eq_delete_accept      | YES          | EQUALITY delete on A (x=0)       | ACCEPT (ignored)
#   seq_position_delete_reject| YES          | POSITION delete path-scoped on A | REJECT (always fatal)
#   disjoint_delete_accept    | YES          | POSITION delete path-scoped on B | ACCEPT (disjoint)
#
# Scenarios 1 and 2 share the SAME S1 history (a concurrent equality delete on A) and differ ONLY by the
# seq knob — the load-bearing seq-preservation nuance. Scenario 3 keeps the seq yet a POSITION delete is
# still fatal. Scenario 4 is the over-firing negative control. The replaced file A is DERIVED per engine
# from the loaded table (the live x=0 data file), so there is no cross-engine path coupling (anti-circular).
#
# C2 VACUOUS-REJECT GUARD: A is live at BOTH S0 and S1 (the concurrent delete ADDS a delete file; it does
# not REMOVE A), so failMissingDeletePaths and the "Files to delete cannot be empty" precondition both
# PASS — the only thing that can reject is the concurrent-delete validation. The sabotage battery (step 6)
# mutation-confirms this: swapping in a table with NO applicable concurrent delete flips the rejects to
# accept, proving each reject was DRIVEN by the concurrent-delete conflict, not a pre-empting path.
#
# BOTH directions (the conflict decision is hand-declared identically on both sides, anti-circular):
#   D1 (Rust validates Java's table): Rust `register_table`s <scenario>/table and runs the symmetric
#     rewrite — `tests/interop_rewritefiles_conflict.rs` test_rust_validates_java_*.
#   D2 (Java validates Rust's table): Java loads <scenario>/rust_table and runs the symmetric rewrite
#     — `RewriteFilesConflictOracle.verify`, "verify-interop-rewritefiles-conflict: N failures".
#
# Test-only oracle; nothing here is in the offline cargo test gate; the temp dir is gitignored.
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64.
# Run from anywhere; paths resolve relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-rewritefiles-conflict"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

echo "==> [1/7] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/7] Java: generate all four scenario tables (<scenario>/table, S0 + concurrent S1)"
run_oracle -Dexec.args=generate-interop-rewritefiles-conflict \
  -Dinterop.rewritefiles_conflict.dir="${TMP}"

echo "==> [3/7] Rust: generate all four scenario tables (<scenario>/rust_table) via the GEN test"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_REWRITEFILES_CONFLICT_GEN_DIR="${TMP}" \
    cargo test -p iceberg --test interop_rewritefiles_conflict \
    test_rewritefiles_conflict_gen -- --nocapture
)

echo "==> [4/7] Java: verify-interop-rewritefiles-conflict (D2 — Java validates the Rust tables)"
VERIFY_OUT="$(run_oracle -Dexec.args=verify-interop-rewritefiles-conflict \
  -Dinterop.rewritefiles_conflict.dir="${TMP}")" || true
echo "${VERIFY_OUT}"
# Fail-closed two ways: a per-check `^FAIL ` line OR absence of the `0 failures` sentinel.
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-rewritefiles-conflict: 0 failures'; then
  echo "==> FAILED — D2 verify emitted a FAIL line or did not emit the '0 failures' sentinel."
  exit 1
fi

echo "==> [5/7] Rust: validate the Java tables (D1 — register + symmetric rewrite_files)"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_REWRITEFILES_CONFLICT_DIR="${TMP}" \
    cargo test -p iceberg --test interop_rewritefiles_conflict \
    test_rust_validates_java -- --nocapture
)

echo "==> [6/7] Sabotage battery — D2 verify must FAIL closed on a corrupted/wrong table"
# CONTROL: step 4's clean verify already passed, so a sabotage-fail is meaningful. Each sabotage runs
# against a SCRATCH copy (the clean TMP is untouched). HARD-FAIL, never SKIP, if a corruption cannot be
# applied (a sabotage that did not actually corrupt anything proves nothing).
#
# RESIDUE STRIP (load-bearing, the C2/C3/C4 lesson): step 4's verify already ran `ops.commit(null, loaded)`
# against the REAL ${TMP}/<scenario>/rust_table, which wrote `v0.metadata.json` (and `v1.metadata.json` for
# the ACCEPT scenarios whose rewrite committed). The sabotage scratch copies those tables, so a NAIVE
# re-verify would re-run `commit(null, loaded)` → write `v0.metadata.json` → AlreadyExistsException BEFORE
# the RewriteFiles validation ever runs, masking the semantic claim with a structural collision (a
# regression that broke the validation to always-ACCEPT would still "fail" here, at the v0 write). So
# build_scratch DELETES the v*.metadata.json residue, restoring each scratch to the pre-verify state (only
# the immutable numbered + final metadata remain) so the production validation path is REACHED. The
# `fresh-*.parquet` / new manifest residue does not collide (unique names).
build_scratch() {
  local scratch="${TMP}/sabotage_scratch"
  rm -rf "${scratch}"
  mkdir -p "${scratch}"
  for scenario in no_seq_eq_delete_reject seq_eq_delete_accept \
    seq_position_delete_reject disjoint_delete_accept; do
    cp -r "${TMP}/${scenario}" "${scratch}/${scenario}"
    # Strip the step-4 commit residue so `commit(null, loaded)` does not collide on v0.metadata.json.
    rm -f "${scratch}/${scenario}"/rust_table/metadata/v*.metadata.json
  done
  echo "${scratch}"
}

# (a) SEMANTIC — the C2 vacuous-reject mutation test (concurrent-delete validation is REACHED + DRIVES the
#     reject). Replace no_seq_eq_delete_reject's table (concurrent EQUALITY delete on A) with
#     disjoint_delete_accept's table (the only concurrent delete is on the DISJOINT B). The verify re-runs
#     the no-seq rewrite (replacing A=x=0) against the swapped-in table; with no concurrent delete applying
#     to A the production validation RE-DERIVES ACCEPT, contradicting the hand-declared REJECT ⇒ FAIL.
#     The pin is the SPECIFIC outcome-mismatch line ("outcome ACCEPT but expected REJECT"): a structural
#     AlreadyExistsException / parse error would also drop the sentinel but would NOT prove the validation
#     re-derived ACCEPT. Requiring the exact semantic line means this only passes when the production
#     conflict decision was actually reached and found NO applicable concurrent delete — i.e. the clean
#     run's reject WAS driven by the equality-delete-on-A applying, not by a vacuous path.
SCRATCH="$(build_scratch)"
rm -rf "${SCRATCH}/no_seq_eq_delete_reject/rust_table"
cp -r "${TMP}/disjoint_delete_accept/rust_table" "${SCRATCH}/no_seq_eq_delete_reject/rust_table"
# The copied-in table also carries step-4 v*.metadata.json residue — strip it (the swap brought the OTHER
# scenario's post-verify tree, which has v0[+v1]) so commit(null,loaded) reaches validation.
rm -f "${SCRATCH}"/no_seq_eq_delete_reject/rust_table/metadata/v*.metadata.json
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-rewritefiles-conflict \
  -Dinterop.rewritefiles_conflict.dir="${SCRATCH}")" || true
if echo "${SAB_OUT}" | grep -q 'verify-interop-rewritefiles-conflict: 0 failures'; then
  echo "FAIL sabotage(disjoint-swap): verify PASSED on a no-applicable-conflict table where REJECT was expected — pin is vacuous"
  echo "${SAB_OUT}"
  exit 1
fi
if ! echo "${SAB_OUT}" | grep -q 'rewritefiles-conflict-d2\[no_seq_eq_delete_reject\]: outcome ACCEPT but expected REJECT'; then
  echo "FAIL sabotage(disjoint-swap): verify failed but NOT via the outcome-mismatch path —"
  echo "  the concurrent-delete validation was never reached (structural collision/parse error masks the claim)."
  echo "${SAB_OUT}"
  exit 1
fi
echo "PASS sabotage(disjoint-swap): a no-applicable-conflict table fails closed via the outcome-mismatch path (validation re-derived ACCEPT where REJECT was expected — the clean reject was driven by the concurrent delete applying to A)"

# (b) SEMANTIC — the position-vs-equality NUANCE is load-bearing. Replace seq_position_delete_reject's
#     table (concurrent POSITION delete on A) with seq_eq_delete_accept's table (concurrent EQUALITY delete
#     on A). The verify re-runs the SEQ-PRESERVING rewrite (seq_position_delete_reject's knob) against the
#     eq-delete table; with the seq preserved the EQUALITY delete is IGNORED (ignore_equality_deletes), so
#     the validation RE-DERIVES ACCEPT, contradicting the hand-declared REJECT ⇒ FAIL. This proves the
#     clean run's position-delete reject was driven by the POSITION-delete-always-fatal branch (not merely
#     by "any concurrent delete"): swap the SAME-partition concurrent delete from POSITION to EQUALITY and,
#     with the seq preserved, the decision flips to ACCEPT.
SCRATCH="$(build_scratch)"
rm -rf "${SCRATCH}/seq_position_delete_reject/rust_table"
cp -r "${TMP}/seq_eq_delete_accept/rust_table" "${SCRATCH}/seq_position_delete_reject/rust_table"
rm -f "${SCRATCH}"/seq_position_delete_reject/rust_table/metadata/v*.metadata.json
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-rewritefiles-conflict \
  -Dinterop.rewritefiles_conflict.dir="${SCRATCH}")" || true
if echo "${SAB_OUT}" | grep -q 'verify-interop-rewritefiles-conflict: 0 failures'; then
  echo "FAIL sabotage(position-to-equality-swap): verify PASSED where the seq-preserving rewrite should have ignored the equality delete and accepted — pin is vacuous"
  echo "${SAB_OUT}"
  exit 1
fi
if ! echo "${SAB_OUT}" | grep -q 'rewritefiles-conflict-d2\[seq_position_delete_reject\]: outcome ACCEPT but expected REJECT'; then
  echo "FAIL sabotage(position-to-equality-swap): verify failed but NOT via the outcome-mismatch path —"
  echo "  the position-vs-equality branch was never exercised (structural collision/parse error masks the claim)."
  echo "${SAB_OUT}"
  exit 1
fi
echo "PASS sabotage(position-to-equality-swap): with the seq preserved, swapping the concurrent delete from POSITION to EQUALITY flips REJECT→ACCEPT (the position-delete-always-fatal branch is load-bearing)"

# (c) STRUCTURAL: truncate a REJECT scenario's final.metadata.json so it no longer parses ⇒ the verify's
#     load branch errors ⇒ FAIL closed via the catch path.
SCRATCH="$(build_scratch)"
RUST_FINAL="${SCRATCH}/no_seq_eq_delete_reject/rust_table/metadata/final.metadata.json"
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
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-rewritefiles-conflict \
  -Dinterop.rewritefiles_conflict.dir="${SCRATCH}")" || true
if echo "${SAB_OUT}" | grep -q 'verify-interop-rewritefiles-conflict: 0 failures'; then
  echo "FAIL sabotage(truncate): verify PASSED on a truncated metadata file — NOT fail-closed"
  echo "${SAB_OUT}"
  exit 1
fi
if ! echo "${SAB_OUT}" | grep -q 'rewritefiles-conflict-d2\[no_seq_eq_delete_reject\]: unexpected error running the rewrite_files'; then
  echo "FAIL sabotage(truncate): verify failed but NOT via the load/parse-error path — the truncation"
  echo "  did not actually break the metadata load (corruption proved nothing)."
  echo "${SAB_OUT}"
  exit 1
fi
echo "PASS sabotage(truncate): a truncated metadata file fails closed via the load/parse-error path"
rm -rf "${TMP}/sabotage_scratch"

echo "==> [7/7] DONE — RewriteFiles conflict-validation interop passed."
echo "    Scenarios: no_seq_eq_delete_reject + seq_eq_delete_accept + seq_position_delete_reject + disjoint_delete_accept"
echo "    Nuance: seq-preservation (ignore_equality_deletes = data_sequence_number.is_some()) +"
echo "            position-vs-equality (a preserved-seq rewrite ignores equality deletes but a position delete is always fatal)"
echo "    Directions: D1 (Rust validates Java) + D2 (Java validates Rust)"
echo "    Sabotage battery: disjoint-swap (C2 vacuous-reject mutation test — validation reached + drives the reject)"
echo "                      + position-to-equality-swap (the position-vs-equality branch is load-bearing)"
echo "                      + structural-truncate (load/parse-error line); control passed in step 4"
