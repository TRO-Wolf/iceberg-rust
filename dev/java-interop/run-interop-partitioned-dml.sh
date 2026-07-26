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
# WG1 NULL-TUPLE LEG (honest-children PartitionExpr): a second Rust GEN test writes
# <dir>/rust_table_nulltuple via `INSERT … SELECT id, CASE WHEN id = 2 THEN NULL ELSE category
# END, value` — a COMPUTED partition-source item. Java reads the FILE-level partition tuples
# back ({null, "books"} via FileScanTask.file().partition()) and both rows (id=2 category IS
# NULL). Before the fix this write stamped a real-but-wrong `electronics` tuple, never NULL.
#
# SABOTAGE STEPS (non-vacuity proof), TWO legs — one per table, because the verify returns
# early on the FIRST parse failure:
#   LEG A: truncate <dir>/rust_table/metadata/final.metadata.json           (COW-DELETE table)
#   LEG B: truncate <dir>/rust_table_nulltuple/metadata/final.metadata.json (WG1 null-tuple table)
# Leg A alone proves nothing about the null-tuple assertions: with rust_table's metadata
# corrupt, `verifyPartDml` returns before `verifyPartDmlNullTuple` ever runs, so those
# assertions would stay unexercised. Leg B corrupts ONLY the null-tuple table, leaving the
# COW-DELETE leg green, so the >0 failures can only come from the null-tuple assertions.
# Each leg: corrupt → re-run Java verify → assert >0 failures → restore → re-run → assert GREEN.
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

# Run the Java oracle over TMP_DIR and echo everything it printed.
run_java_verify() {
  (
    cd "${SCRIPT_DIR}"
    JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
      PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH} \
      /opt/maven/bin/mvn -o -q compile exec:java \
      -Dexec.args=verify-interop-part-dml \
      -Dinterop.part_dml.dir="${TMP_DIR}" 2>&1
  )
}

# One sabotage leg: truncate ONE final.metadata.json, prove the verify goes RED,
# restore it, prove the verify goes GREEN again.
#   $1 = human label   $2 = path to the final.metadata.json to corrupt
# Every failure mode is a HARD-FAIL (exit non-zero), never a SKIP: a sabotage
# that could not be applied has proven nothing.
sabotage_leg() {
  local label="$1"
  local target="$2"

  echo "==> SABOTAGE ${label}: truncate ${target} → Java must report >0 failures (non-vacuity)"
  if [ ! -f "${target}" ]; then
    echo "==> HARD-FAIL: ${target} absent — sabotage ${label} cannot be applied."
    exit 1
  fi
  cp "${target}" "${target}.bak"

  # Truncate to 16 bytes — Java's JSON parser will fail to parse it.
  local rc=0
  dd if=/dev/zero of="${target}" bs=1 count=16 2>/dev/null || rc=$?
  if [ "${rc}" -ne 0 ]; then
    # Restore before failing.
    cp "${target}.bak" "${target}"
    rm -f "${target}.bak"
    echo "==> HARD-FAIL: sabotage ${label} truncation failed (exit ${rc})."
    exit 1
  fi
  if cmp -s "${target}.bak" "${target}"; then
    cp "${target}.bak" "${target}"
    rm -f "${target}.bak"
    echo "==> HARD-FAIL: sabotage ${label} left the file unchanged — nothing was corrupted."
    exit 1
  fi

  local sabotage_out
  sabotage_out="$(run_java_verify || true)"
  echo "${sabotage_out}"

  # Restore the original metadata BEFORE checking the sabotage result.
  cp "${target}.bak" "${target}"
  rm -f "${target}.bak"

  if echo "${sabotage_out}" | grep -q ': 0 failures'; then
    echo "==> HARD-FAIL: sabotage ${label} did NOT trigger a verify failure — \
that leg of the verify is vacuous."
    exit 1
  fi
  echo "==> SABOTAGE ${label} RED: Java correctly detected the corruption (>0 failures)."

  echo "==> Post-sabotage restore (${label}): re-run verify on the restored metadata — must be GREEN"
  local restore_out
  restore_out="$(run_java_verify)"
  echo "${restore_out}"
  if echo "${restore_out}" | grep -q '^FAIL ' || ! echo "${restore_out}" | grep -q ': 0 failures'; then
    echo "==> HARD-FAIL: post-restore verify (${label}) failed — the restore did not work or \
the verify is broken."
    exit 1
  fi
  echo "==> Post-restore verify (${label}) returned GREEN."
}

echo "==> [1/5] Reset the temp table dir: ${TMP_DIR}"
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"

echo "==> [2/5] Rust: WRITE the partitioned V2 tables via DataFusion SQL DML \
(INSERT + COW DELETE; INSERT…SELECT CASE→NULL) + final.metadata.json each"
(
  cd "${REPO_ROOT}"
  # No test-name filter: runs BOTH GEN tests (COW-DELETE table + WG1 null-tuple table).
  ICEBERG_INTEROP_PART_DML_GEN_DIR="${TMP_DIR}" \
    cargo test -p iceberg-datafusion \
      --test interop_partitioned_dml \
      -- --nocapture
)

echo "==> [3/5] Java: load the RUST-written final.metadata.json files, read via IcebergGenerics, \
verify survivor ids = {3,4} (books partition) + the WG1 null partition tuple"
VERIFY_OUT="$(run_java_verify)"
echo "${VERIFY_OUT}"
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' || ! echo "${VERIFY_OUT}" | grep -q ': 0 failures'; then
  echo "==> FAILED — Java could not correctly read the Rust-written partitioned COW-DELETE table \
(a real DataFusion-DML write-incompatibility finding)."
  exit 1
fi
echo "==> GREEN — Java read the Rust-written partitioned COW-DELETE table (survivor ids = {3,4})."

echo "==> [4/5] SABOTAGE LEG A — the COW-DELETE table (rust_table)"
sabotage_leg "A (rust_table)" "${TMP_DIR}/rust_table/metadata/final.metadata.json"

echo "==> [5/5] SABOTAGE LEG B — the WG1 null-tuple table (rust_table_nulltuple). Leg A cannot \
reach these assertions: the verify returns on rust_table's parse failure before the null-tuple \
leg runs, so only this leg proves the null-tuple assertions are non-vacuous."
sabotage_leg "B (rust_table_nulltuple)" \
  "${TMP_DIR}/rust_table_nulltuple/metadata/final.metadata.json"

echo "==> DONE — U4 partitioned COW DELETE round-trip passed:"
echo "    * Java read the Rust-written partitioned table (DataFusion SQL DML)."
echo "    * Survivor ids = {3,4} (books: novel/textbook); electronics ids 1/2 absent."
echo "    * Java read the WG1 null partition tuple back ({null, books}; id=2 category IS NULL)."
echo "    * Sabotage leg A (truncated rust_table metadata) triggered >0 failures."
echo "    * Sabotage leg B (truncated rust_table_nulltuple metadata) triggered >0 failures —"
echo "      the null-tuple assertions are non-vacuous in their own right."
echo "    * Both post-restore verifies returned GREEN."
