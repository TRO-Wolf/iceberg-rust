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
# U4 — PARTITIONED COPY-ON-WRITE DELETE interop, DIRECTION 2 ("Java reads what RUST writes via
# DataFusion SQL DML").
#
# Rust's GEN test uses a DataFusion SessionContext + MemoryCatalog over LocalFsStorageFactory to:
#   1. Create a partitioned V2 table {id int, category string, value string} identity(category).
#   2. INSERT rows into TWO partitions:
#        electronics: id=1 (laptop), id=2 (phone)
#        books:       id=3 (novel),  id=4 (textbook)
#   3. Run SQL: DELETE FROM rust_table WHERE category = 'electronics' (copy-on-write mode).
#      The electronics partition data file is removed; books partition file is untouched.
#   4. Write final.metadata.json to <dir>/rust_table/metadata/final.metadata.json.
#
# Java then loads that RUST-written metadata, reads via IcebergGenerics, and asserts:
#   * Exactly 2 surviving rows (ids {3,4} in the books partition).
#   * Electronics ids 1 and 2 are ABSENT (COW removed the partition file).
#   * Column values match: id=3 → (books,novel); id=4 → (books,textbook).
#
# SABOTAGE STEP (non-vacuity proof): after the green verify, the script corrupts the
# final.metadata.json (truncates it), re-runs Java verify, and asserts Java now reports >0
# failures.  The original file is then restored and Java is re-run to confirm GREEN.
# If the corruption cannot be applied (file absent), the script exits non-zero (HARD-FAIL per
# CLAUDE.md — a sabotage that cannot be applied must FAIL, not skip).
#
# TEST-ONLY ORACLE (dev tool); not in the offline gate; nothing binary committed; temp dir
# gitignored.  Without ICEBERG_INTEROP_PART_DML_GEN_DIR the Rust GEN test is a clean no-op.
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64,
# a Rust toolchain.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP_DIR="${SCRIPT_DIR}/target/interop-partitioned-dml"

echo "==> [1/5] Reset the temp table dir: ${TMP_DIR}"
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"

echo "==> [2/5] Rust: WRITE a partitioned V2 table via DataFusion SQL DML \
(INSERT + COW DELETE) + final.metadata.json"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_PART_DML_GEN_DIR="${TMP_DIR}" \
    cargo test -p iceberg-datafusion \
      --test interop_partitioned_dml \
      test_part_dml_gen_rust_writes_java_readable_partitioned_cow_table \
      -- --nocapture
)

echo "==> [3/5] Java: load the RUST-written final.metadata.json, read via IcebergGenerics, \
verify survivor ids = {3,4} (books partition)"
VERIFY_OUT="$(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH} \
    /opt/maven/bin/mvn -o -q compile exec:java \
    -Dexec.args=verify-interop-part-dml \
    -Dinterop.part_dml.dir="${TMP_DIR}" 2>&1
)"
echo "${VERIFY_OUT}"
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' || ! echo "${VERIFY_OUT}" | grep -q ': 0 failures'; then
  echo "==> FAILED — Java could not correctly read the Rust-written partitioned COW-DELETE table \
(a real DataFusion-DML write-incompatibility finding)."
  exit 1
fi
echo "==> GREEN — Java read the Rust-written partitioned COW-DELETE table (survivor ids = {3,4})."

echo "==> [4/5] SABOTAGE: truncate final.metadata.json → Java must report >0 failures (non-vacuity)"
FINAL_META="${TMP_DIR}/rust_table/metadata/final.metadata.json"
if [ ! -f "${FINAL_META}" ]; then
  echo "==> HARD-FAIL: final.metadata.json absent — sabotage cannot be applied."
  exit 1
fi
cp "${FINAL_META}" "${FINAL_META}.bak"

# Truncate to 16 bytes — Java's JSON parser will fail to parse it.
rc=0
dd if=/dev/zero of="${FINAL_META}" bs=1 count=16 2>/dev/null || rc=$?
if [ "${rc}" -ne 0 ]; then
  # Restore before failing
  cp "${FINAL_META}.bak" "${FINAL_META}"
  echo "==> HARD-FAIL: sabotage truncation failed (exit ${rc})."
  exit 1
fi

SABOTAGE_OUT="$(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH} \
    /opt/maven/bin/mvn -o -q compile exec:java \
    -Dexec.args=verify-interop-part-dml \
    -Dinterop.part_dml.dir="${TMP_DIR}" 2>&1 || true
)"
echo "${SABOTAGE_OUT}"

# Restore the original metadata BEFORE checking the sabotage result.
cp "${FINAL_META}.bak" "${FINAL_META}"
rm -f "${FINAL_META}.bak"

if echo "${SABOTAGE_OUT}" | grep -q ': 0 failures'; then
  echo "==> HARD-FAIL: sabotage (truncated metadata) did NOT trigger a verify failure — \
the verify is vacuous."
  exit 1
fi
echo "==> SABOTAGE RED: Java correctly detected the corruption (>0 failures)."

echo "==> [5/5] Post-sabotage restore: re-run verify on the restored metadata — must be GREEN"
RESTORE_OUT="$(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH} \
    /opt/maven/bin/mvn -o -q compile exec:java \
    -Dexec.args=verify-interop-part-dml \
    -Dinterop.part_dml.dir="${TMP_DIR}" 2>&1
)"
echo "${RESTORE_OUT}"
if echo "${RESTORE_OUT}" | grep -q '^FAIL ' || ! echo "${RESTORE_OUT}" | grep -q ': 0 failures'; then
  echo "==> HARD-FAIL: post-restore verify failed — the restore did not work or the verify is broken."
  exit 1
fi

echo "==> DONE — U4 partitioned COW DELETE round-trip passed:"
echo "    * Java read the Rust-written partitioned table (DataFusion SQL DML)."
echo "    * Survivor ids = {3,4} (books: novel/textbook); electronics ids 1/2 absent."
echo "    * Sabotage (truncated metadata) triggered >0 failures — verify is non-vacuous."
echo "    * Post-restore verify returned GREEN."
