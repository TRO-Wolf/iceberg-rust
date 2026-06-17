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
# MAINTENANCE RewritePositionDeleteFiles interop harness (BLOCK-2 G1), VERIFY-only direction — "Java
# validates what RUST compacted". The parquet position-delete COMPACTION action is engine-agnostic in
# Rust, but the real Java action is a Spark-surface class (NOT on the iceberg-core oracle classpath) and
# Java cannot DRIVE the compaction. So the corroboration is: RUST writes both the PRE table (real PARQUET
# position-delete files masking a known subset) and the POST table (the SAME rows masked by the COMPACTED
# position delete), and Java's IcebergGenerics reads BOTH and asserts the live row sets are IDENTICAL — the
# no-Spark proof that the compacted position delete masks EXACTLY the rows the original ones masked, with
# the data sequence number preserved (a wrong seq-stamp would resurrect or over-mask a row and break read
# identity).
#
# This is a TEST-ONLY ORACLE (a dev tool) — it is NOT part of the shipped Rust library, and NOT part of the
# offline `cargo test` gate (it needs Java + Maven). The committed artifacts are the oracle code
# (RewritePosDeleteOracle in InteropOracle.java), the Rust GEN test (interop_rewrite_pos_deletes.rs), and
# this script. The temp tables under dev/java-interop/target/ are gitignored.
#
# Methodology (Rust GEN → Java VERIFY → sabotage battery):
#   1. ICEBERG_INTEROP_REWRITE_POS_DELETES_GEN_DIR="$TMP" cargo test ... test_rewrite_pos_deletes_gen
#        -> Rust builds a partitioned V2 table under "$TMP/rust_table" (cat=A id 100/120/130, cat=B id
#           200/220/230 at seq 1; TWO parquet position deletes per partition masking pos 1 = id 120/220),
#           lands "$TMP/rust_table/metadata/final.metadata.json". It builds a SEPARATE identical table under
#           "$TMP/rust_table_compacted", runs RewritePositionDeleteFiles (4 pos-deletes → 2 compacted), and
#           lands its final.metadata.json. It also writes "$TMP/rust_table_nodeletes" (same data, NO
#           deletes) used ONLY by the sabotage battery. A Rust-side sanity check confirms read-identity +
#           the compaction shape BEFORE handing the tables to Java.
#   2. mvn ... -Dexec.args=verify-interop-rewrite-pos-deletes -Dinterop.rewrite_pos_deletes.dir="$TMP"
#        -> Java loads the PRE + POST tables, reads each via IcebergGenerics, and asserts: PRE live ids ==
#           {100,130,200,230}; READ IDENTITY POST == PRE; and the compaction shape (PRE pos > POST pos > 0).
#           Nonzero failures => exit 1.
#   3. Sabotage battery (each on a SCRATCH copy; HARD-FAIL never SKIP if a corruption cannot be applied):
#        (a) read-identity breaker — swap the POST metadata with the no-deletes table's, so POST reads the
#            FULL id set != PRE ⇒ FAIL via the read-identity-BROKEN line (proves the read-identity leg is
#            non-vacuous: a compacted table that resurrected a row fails closed).
#        (b) structural truncate — truncate the PRE metadata so it no longer parses ⇒ FAIL via the
#            load/parse-error path (proves a corrupted table fails closed, never silently passes).
#
# Without ICEBERG_INTEROP_REWRITE_POS_DELETES_GEN_DIR the Rust GEN test is a clean no-op (it stays green in
# the offline gate); this script is what flips it into the REAL comparison.
#
# Requirements:
#   - Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64.
#   - A Rust toolchain (the repo's pinned nightly via rust-toolchain.toml).
#   - The Maven deps are the SAME ones the convert-eq-delete / remove-dangling harnesses already pulled — no
#     new pom deps. If ~/.m2 is populated, `mvn -o` runs fully offline.
#
# Run from anywhere; paths are resolved relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-rewrite-pos-deletes"

run_oracle() {
  (
    cd "${SCRIPT_DIR}"
    JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
      PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
      /opt/maven/bin/mvn -o -q compile exec:java "$@"
  )
}

echo "==> [1/5] Reset the temp table dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/5] Rust GEN: write rust_table (many pos-deletes) + rust_table_compacted (compacted) + rust_table_nodeletes (sabotage breaker)"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_REWRITE_POS_DELETES_GEN_DIR="${TMP}" \
    cargo test -p iceberg --test interop_rewrite_pos_deletes test_rewrite_pos_deletes_gen -- --nocapture
)

echo "==> [3/5] Java verify: IcebergGenerics reads BOTH tables, asserts read-identity (PRE == POST) + the compaction shape"
VERIFY_OUT="$(run_oracle -Dexec.args=verify-interop-rewrite-pos-deletes \
  -Dinterop.rewrite_pos_deletes.dir="${TMP}")" || true
echo "${VERIFY_OUT}"
# Fail-closed two ways: a per-check `^FAIL ` line OR absence of the `0 failures` sentinel.
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-rewrite-pos-deletes: 0 failures'; then
  echo "==> FAILED — the clean verify emitted a FAIL line or did not emit the '0 failures' sentinel."
  exit 1
fi

echo "==> [4/5] Sabotage battery — the verify must FAIL closed on a corrupted/wrong table"
# CONTROL: step 3's clean verify already passed, so a sabotage-fail is meaningful. Each sabotage runs
# against a SCRATCH copy (the clean ${TMP} is untouched). HARD-FAIL, never SKIP, if a corruption cannot be
# applied — a sabotage that did not actually corrupt anything proves nothing.
build_scratch() {
  local scratch="${TMP}/sabotage_scratch"
  rm -rf "${scratch}"
  mkdir -p "${scratch}"
  cp -r "${TMP}/rust_table" "${scratch}/rust_table"
  cp -r "${TMP}/rust_table_compacted" "${scratch}/rust_table_compacted"
  cp -r "${TMP}/rust_table_nodeletes" "${scratch}/rust_table_nodeletes"
  echo "${scratch}"
}

# (a) SEMANTIC — read-identity breaker (resurrection). Replace the COMPACTED (POST) table's metadata with
#     the no-deletes table's metadata. POST now reads the FULL id set {100,120,130,200,220,230} (the masked
#     rows 120/220 resurrected) != PRE {100,130,200,230} ⇒ FAIL via the SPECIFIC read-identity-BROKEN line.
#     This proves the corruption-safety (read-identity) comparison is wired and non-vacuous: a compacted
#     table that resurrected a masked row — the EXACT failure a wrong seq-stamp causes — fails closed.
SCRATCH="$(build_scratch)"
POST_META="${SCRATCH}/rust_table_compacted/metadata/final.metadata.json"
NODELETES_META="${SCRATCH}/rust_table_nodeletes/metadata/final.metadata.json"
if [[ ! -f "${POST_META}" || ! -f "${NODELETES_META}" ]]; then
  echo "==> FAILED — cannot apply read-identity sabotage: POST/no-deletes metadata absent (GEN gated?)"
  exit 1
fi
cp "${NODELETES_META}" "${POST_META}"
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-rewrite-pos-deletes \
  -Dinterop.rewrite_pos_deletes.dir="${SCRATCH}")" || true
if echo "${SAB_OUT}" | grep -q 'verify-interop-rewrite-pos-deletes: 0 failures'; then
  echo "FAIL sabotage(read-identity): verify PASSED on a compacted table that resurrected a masked row — the read-identity check is vacuous"
  echo "${SAB_OUT}"
  exit 1
fi
if ! echo "${SAB_OUT}" | grep -q 'read-identity BROKEN'; then
  echo "FAIL sabotage(read-identity): verify failed but NOT via the read-identity-BROKEN path —"
  echo "  the corruption-safety comparison was never reached (a different failure masks the claim)."
  echo "${SAB_OUT}"
  exit 1
fi
echo "PASS sabotage(read-identity): a compacted table that resurrected a masked row fails closed via the read-identity-BROKEN path"

# (b) STRUCTURAL — truncate the PRE table's final.metadata.json so it no longer parses ⇒ the verify's load
#     branch errors ⇒ FAIL closed, pinned to the load/parse-error path.
SCRATCH="$(build_scratch)"
PRE_META="${SCRATCH}/rust_table/metadata/final.metadata.json"
if [[ ! -f "${PRE_META}" ]]; then
  echo "==> FAILED — cannot apply truncate sabotage: ${PRE_META} absent (GEN gated?)"
  exit 1
fi
SIZE="$(stat -c%s "${PRE_META}")"
if (( SIZE <= 60 )); then
  echo "==> FAILED — cannot apply truncate sabotage: ${PRE_META} is only ${SIZE} bytes"
  exit 1
fi
head -c "$(( SIZE - 60 ))" "${PRE_META}" > "${PRE_META}.tmp"
mv "${PRE_META}.tmp" "${PRE_META}"
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-rewrite-pos-deletes \
  -Dinterop.rewrite_pos_deletes.dir="${SCRATCH}")" || true
if echo "${SAB_OUT}" | grep -q 'verify-interop-rewrite-pos-deletes: 0 failures'; then
  echo "FAIL sabotage(truncate): verify PASSED on a truncated metadata file — NOT fail-closed"
  echo "${SAB_OUT}"
  exit 1
fi
if ! echo "${SAB_OUT}" | grep -q 'unexpected error running the rewrite-pos-deletes verify'; then
  echo "FAIL sabotage(truncate): verify failed but NOT via the load/parse-error path — the truncation"
  echo "  did not actually break the metadata load (corruption proved nothing)."
  echo "${SAB_OUT}"
  exit 1
fi
echo "PASS sabotage(truncate): a truncated metadata file fails closed via the load/parse-error path"
rm -rf "${TMP}/sabotage_scratch"

echo "==> [5/5] DONE — RewritePositionDeleteFiles interop passed."
echo "    Java's IcebergGenerics read is IDENTICAL before (many position deletes) and after (fewer,"
echo "    compacted position deletes); the compaction preserved every live row and masked exactly the same"
echo "    rows (live ids {100,130,200,230}), seq-stamp preserved."
echo "    Sabotage battery: read-identity-breaker (read-identity-BROKEN line) + structural-truncate"
echo "                      (load-error line); control passed in step 3."
echo "    Spark-action-output comparison: N/A (RewritePositionDeleteFiles is Spark-surface, out of core-parity scope)"
