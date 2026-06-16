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
# MAINTENANCE RemoveDanglingDeleteFiles interop harness (post-charter unit #1). Proves the Rust
# `RemoveDanglingDeleteFiles` action against Java's documented `findDanglingDeletes` semantics WITHOUT
# Spark — the real Java action is `RemoveDanglingDeletesSparkAction`, a SPARK-surface class that is NOT
# on the iceberg-core oracle classpath (mvn -o iceberg-core 1.10.0 + JDK11 only, no Spark jar). The
# Spark action's own OUTPUT comparison is therefore explicitly N/A (out of core-parity scope). Instead
# the proof is three engine-agnostic, ANTI-CIRCULAR claims (the dangling set is hand-declared
# IDENTICALLY in Java `RemoveDanglingOracle` and Rust `interop_remove_dangling.rs`, from the published
# findDanglingDeletes spec — never derived from the other engine):
#
#   (1) SEMANTICS-MATCH (detection). Rust's action removes EXACTLY the hand-declared dangling set + keeps
#       the rest (D1, counter + surviving-delete-set check); Java's oracle INDEPENDENTLY recomputes
#       findDanglingDeletes over the PRE-cleanup table's live entries (engine-agnostic manifest API — the
#       same data ENTRIES/DATA_FILES/DELETE_FILES project) and confirms the SAME set (D2).
#   (2) API CONTRACT. The removed-{position-delete-files,equality-delete-files,dvs} counters + the
#       surviving delete-file set match the declared KEEP/REMOVE partition (both directions).
#   (3) CORRUPTION SAFETY (read-identity, the load-bearing property). The merge-on-read live id set is
#       IDENTICAL before and after cleanup — removing the danglers resurrects NOTHING and loses NO data.
#       Proven BOTH directions (Rust scans pre+post in D1; Java IcebergGenerics reads pre + the
#       Rust-cleaned table in D2).
#
# THE OFF-BY-ONE (the corruption edge): a POSITION delete dangles when seq < min (STRICT); an EQUALITY
# delete when seq <= min (NON-strict). Same-seq pos applies (KEEP); same-seq eq does not (REMOVE). The
# fixture pins both at the exact boundary (`pk`/`er`). A DV dangles when its referenced data file is gone.
#
# Two WORLDS (the on-disk spec mandates DVs for V3 position deletes and forbids parquet position deletes
# there, while DVs are V3-only ⇒ a parquet position delete and a DV cannot coexist in one table):
#   V2 world (table / rust_table):     pk (pos KEEP) pr (pos REMOVE) ek (eq KEEP) er (eq REMOVE,off-by-one) ne (eq REMOVE,no-data)
#   V3 world (table_v3 / rust_table_v3): dv (DV REMOVE,ref gone) dk (DV KEEP)
#
# Directions:
#   D1 (Rust validates Java's table): Rust register_table`s <dir>/table[_v3], runs the action, asserts
#     removed==REMOVE + survivors==KEEP + counters + read-identity — interop_remove_dangling.rs
#     test_rust_validates_java_remove_dangling.
#   D2 (Java validates Rust's tables): Java loads <dir>/rust_table[_v3] (pre) + rust_table_cleaned[_v3]
#     (post-action), recomputes findDanglingDeletes on pre + checks read-identity pre↔cleaned —
#     RemoveDanglingOracle.verify, "verify-interop-remove-dangling: N failures".
#
# Test-only oracle; nothing here is in the offline cargo test gate; the temp dir is gitignored.
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64.
# Run from anywhere; paths resolve relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-remove-dangling"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

echo "==> [1/7] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/7] Java: generate the mirror PRE-cleanup tables (<dir>/table[_v3]) for Rust's D1"
run_oracle -Dexec.args=generate-interop-remove-dangling \
  -Dinterop.remove_dangling.dir="${TMP}"

echo "==> [3/7] Rust: generate the PRE + CLEANED tables (<dir>/rust_table[_cleaned][_v3]) via the GEN test"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_REMOVE_DANGLING_GEN_DIR="${TMP}" \
    cargo test -p iceberg --test interop_remove_dangling \
    test_remove_dangling_gen -- --nocapture
)

echo "==> [4/7] Java: verify-interop-remove-dangling (D2 — Java validates the Rust tables)"
VERIFY_OUT="$(run_oracle -Dexec.args=verify-interop-remove-dangling \
  -Dinterop.remove_dangling.dir="${TMP}")" || true
echo "${VERIFY_OUT}"
# Fail-closed two ways: a per-check `^FAIL ` line OR absence of the `0 failures` sentinel.
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-remove-dangling: 0 failures'; then
  echo "==> FAILED — D2 verify emitted a FAIL line or did not emit the '0 failures' sentinel."
  exit 1
fi

echo "==> [5/7] Rust: validate the Java tables (D1 — register + run the action + assert the contract)"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_REMOVE_DANGLING_DIR="${TMP}" \
    cargo test -p iceberg --test interop_remove_dangling \
    test_rust_validates_java_remove_dangling -- --nocapture
)

echo "==> [6/7] Sabotage battery — D2 verify must FAIL closed on a corrupted/wrong table"
# CONTROL: step 4's clean verify already passed, so a sabotage-fail is meaningful. Each sabotage runs
# against a SCRATCH copy (the clean ${TMP} is untouched). HARD-FAIL, never SKIP, if a corruption cannot
# be applied — a sabotage that did not actually corrupt anything proves nothing.
build_scratch() {
  local scratch="${TMP}/sabotage_scratch"
  rm -rf "${scratch}"
  mkdir -p "${scratch}"
  cp -r "${TMP}/rust_table" "${scratch}/rust_table"
  cp -r "${TMP}/rust_table_cleaned" "${scratch}/rust_table_cleaned"
  cp -r "${TMP}/rust_table_v3" "${scratch}/rust_table_v3"
  cp -r "${TMP}/rust_table_cleaned_v3" "${scratch}/rust_table_cleaned_v3"
  echo "${scratch}"
}

# (a) SEMANTIC — wrong-cleaned-table swap (NOT-cleaned). Replace the V2 cleaned table's metadata with the
#     PRE (uncleaned) table's metadata. The cleaned table now still carries the DANGLING deletes
#     (pr/er/ne) as LIVE, so the surviving-delete-partition check derives {pk,ek,pr,er,ne} != hand-declared
#     KEEP {pk,ek} ⇒ FAIL via the SPECIFIC survivor-mismatch line. Read-identity still PASSES (pre==pre),
#     so this proves the survivor check is the discriminator — a regression that did not actually remove
#     the danglers is caught here, NOT masked by the (trivially-true) read-identity leg.
SCRATCH="$(build_scratch)"
PRE_META="${SCRATCH}/rust_table/metadata/final.metadata.json"
CLEANED_META="${SCRATCH}/rust_table_cleaned/metadata/final.metadata.json"
if [[ ! -f "${PRE_META}" || ! -f "${CLEANED_META}" ]]; then
  echo "==> FAILED — cannot apply wrong-cleaned-swap sabotage: PRE/CLEANED metadata absent (GEN gated?)"
  exit 1
fi
# Point the cleaned table at the PRE table's manifests/snapshot — copy the PRE metadata over the cleaned
# table's final.metadata.json, rewriting the embedded location so the manifest paths still resolve under
# the PRE table dir (the PRE metadata records absolute paths under rust_table, which exist in the scratch).
cp "${PRE_META}" "${CLEANED_META}"
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-remove-dangling \
  -Dinterop.remove_dangling.dir="${SCRATCH}")" || true
if echo "${SAB_OUT}" | grep -q 'verify-interop-remove-dangling: 0 failures'; then
  echo "FAIL sabotage(wrong-cleaned-swap): verify PASSED on a NOT-cleaned table — the survivor check is vacuous"
  echo "${SAB_OUT}"
  exit 1
fi
if ! echo "${SAB_OUT}" | grep -q 'remove-dangling-d2\[table\]: surviving delete partitions'; then
  echo "FAIL sabotage(wrong-cleaned-swap): verify failed but NOT via the surviving-delete-partition path —"
  echo "  the cleaned-table survivor check was never reached (a different failure masks the claim)."
  echo "${SAB_OUT}"
  exit 1
fi
echo "PASS sabotage(wrong-cleaned-swap): a NOT-cleaned table fails closed via the surviving-delete-partition mismatch (the danglers are still live)"

# (b) SEMANTIC — read-identity breaker (cross-world cleaned swap). Replace the V2 cleaned table's metadata
#     with the V3 cleaned table's metadata. The V3 read is {600,620,630,700,730}; the V2 PRE read is
#     {100,130,200,220,230,300,330,400,420,500,520}. cleanedIds != preIds ⇒ FAIL via the SPECIFIC
#     read-identity-BROKEN line — proving the corruption-safety (read-identity) comparison is wired and
#     non-vacuous (a cleaned table whose read DIFFERS from PRE — i.e. resurrected/lost rows — fails closed).
SCRATCH="$(build_scratch)"
V2_CLEANED_META="${SCRATCH}/rust_table_cleaned/metadata/final.metadata.json"
V3_CLEANED_META="${SCRATCH}/rust_table_cleaned_v3/metadata/final.metadata.json"
if [[ ! -f "${V2_CLEANED_META}" || ! -f "${V3_CLEANED_META}" ]]; then
  echo "==> FAILED — cannot apply read-identity sabotage: V2/V3 cleaned metadata absent (GEN gated?)"
  exit 1
fi
cp "${V3_CLEANED_META}" "${V2_CLEANED_META}"
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-remove-dangling \
  -Dinterop.remove_dangling.dir="${SCRATCH}")" || true
if echo "${SAB_OUT}" | grep -q 'verify-interop-remove-dangling: 0 failures'; then
  echo "FAIL sabotage(read-identity): verify PASSED on a cleaned table whose read differs from PRE — the read-identity check is vacuous"
  echo "${SAB_OUT}"
  exit 1
fi
if ! echo "${SAB_OUT}" | grep -q 'remove-dangling-d2\[table\]: read-identity BROKEN'; then
  echo "FAIL sabotage(read-identity): verify failed but NOT via the read-identity-BROKEN path —"
  echo "  the corruption-safety comparison was never reached (a different failure masks the claim)."
  echo "${SAB_OUT}"
  exit 1
fi
echo "PASS sabotage(read-identity): a cleaned table whose merge-on-read read differs from PRE fails closed via the read-identity-BROKEN path"

# (c) STRUCTURAL — truncate the V3 PRE table's final.metadata.json so it no longer parses ⇒ the verify's
#     load branch errors ⇒ FAIL closed, pinned to the load/parse-error path.
SCRATCH="$(build_scratch)"
V3_PRE_META="${SCRATCH}/rust_table_v3/metadata/final.metadata.json"
if [[ ! -f "${V3_PRE_META}" ]]; then
  echo "==> FAILED — cannot apply truncate sabotage: ${V3_PRE_META} absent (GEN gated?)"
  exit 1
fi
SIZE="$(stat -c%s "${V3_PRE_META}")"
if (( SIZE <= 60 )); then
  echo "==> FAILED — cannot apply truncate sabotage: ${V3_PRE_META} is only ${SIZE} bytes"
  exit 1
fi
head -c "$(( SIZE - 60 ))" "${V3_PRE_META}" > "${V3_PRE_META}.tmp"
mv "${V3_PRE_META}.tmp" "${V3_PRE_META}"
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-remove-dangling \
  -Dinterop.remove_dangling.dir="${SCRATCH}")" || true
if echo "${SAB_OUT}" | grep -q 'verify-interop-remove-dangling: 0 failures'; then
  echo "FAIL sabotage(truncate): verify PASSED on a truncated metadata file — NOT fail-closed"
  echo "${SAB_OUT}"
  exit 1
fi
if ! echo "${SAB_OUT}" | grep -q 'remove-dangling-d2\[table_v3\]: unexpected error running the remove-dangling verify'; then
  echo "FAIL sabotage(truncate): verify failed but NOT via the load/parse-error path — the truncation"
  echo "  did not actually break the metadata load (corruption proved nothing)."
  echo "${SAB_OUT}"
  exit 1
fi
echo "PASS sabotage(truncate): a truncated metadata file fails closed via the load/parse-error path"
rm -rf "${TMP}/sabotage_scratch"

echo "==> [7/7] DONE — RemoveDanglingDeleteFiles interop passed."
echo "    Worlds: V2 (pk/pr/ek/er/ne: pos+eq+no-data) + V3 (dv/dk: deletion vectors)"
echo "    Claims: (1) semantics-match (Java's independent findDanglingDeletes == Rust's removed set),"
echo "            (2) API contract (counters + surviving delete set), (3) read-identity (no resurrection/loss)"
echo "    Directions: D1 (Rust validates Java) + D2 (Java validates Rust)"
echo "    Sabotage battery: wrong-cleaned-swap (survivor-mismatch line) + read-identity-breaker"
echo "                      (read-identity-BROKEN line) + structural-truncate (load-error line); control passed in step 4"
echo "    Spark-action-output comparison: N/A (RemoveDanglingDeletesSparkAction is Spark-surface, out of core-parity scope)"
