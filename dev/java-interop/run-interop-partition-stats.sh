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
# PARTITION-STATS FILE interop harness (increment Z3 + R2) — proving bidirectional parity between
# the Rust compute_and_write_stats_file / read_partition_stats_file and Java 1.10.0 production
# PartitionStatsHandler.computeAndWriteStatsFile / readPartitionStatsFile.
#
# FIXTURE Z3: V2 table identity(category) {id long, category string, data string optional}:
#   S1: fast-append file_a (cat=a, 3 records, 300 bytes) + file_b (cat=b, 2 records, 200 bytes).
#   S2: row-delta pos-delete PD (cat=a, 1 record, 50 bytes).
#
# EXPECTED STATS ROWS (hand-declared, anti-circular):
#   cat=a: data_records=3, data_files=1, size=300, pos_del_records=1, pos_del_files=1,
#          eq_del=0/0, dv_count=0, last_updated=S2 (pos-delete is more recent)
#   cat=b: data_records=2, data_files=1, size=200, pos_del=0/0, eq_del=0/0, dv_count=0,
#          last_updated=S1 (no S2 activity in cat=b)
#
# FIXTURE R2-INCR: V2 table identity(category) {id long, category string, data string optional}:
#   S1: fast-append file_a (cat=a, 3 records, 300 bytes) + file_b (cat=b, 2 records, 200 bytes).
#       Compute+register S1 stats (FULL).
#   S2: delete_files(file_a) — DELETE snapshot with DELETED tombstone (SUBTRACT arm).
#       Compute+register S2 stats (INCREMENTAL — auto-selected because S1 stats exists).
# EXPECTED STATS ROWS after S2:
#   cat=a: data_records=0, data_files=0, size=0 (subtracted by the DELETED tombstone)
#   cat=b: data_records=2, data_files=1, size=200 (unchanged from S1 base)
#
# FIXTURE R2-UUID: V2 table identity(partition_id uuid) {id long, partition_id uuid}:
#   S1: fast-append one file (partition_id=550e8400-e29b-41d4-a716-446655440000, 5 records, 500 bytes).
#   Compute+register stats (FULL — exercises 16-byte big-endian UUID partition encoding).
#
# FIXTURE R3-TIME / R3-FIXED / R3-BINARY: V2 tables partitioned by the remaining exotic types:
#   time   identity(partition_time)   = 45296789012 micros-since-midnight (Time64(Microsecond)).
#   fixed  identity(partition_fixed)  = fixed[4] 0xdeadbeef (FixedSizeBinary(4) raw bytes).
#   binary identity(partition_binary) = binary 0x0102030405 (LargeBinary raw bytes).
#   Each: S1 fast-append one file, compute+register stats (FULL), bidirectional D1+D2.
#
# THE CHAIN:
#
#   1. Reset the temp dir.
#   2. Java: generate-interop-partition-stats (Z3 fixture)
#   3. Rust: GEN (Z3)
#   4. Java: verify-interop-partition-stats D1 (Java reads Rust stats parquet)
#   5. Rust: D2 (Rust reads Java stats parquet + compare against java_stats.json)
#   6. Rust: cross-version (V2 file read against V3 schema)
#   7. Sabotage battery (Z3):
#        7a: truncate the Rust stats parquet → Java D1 verify must FAIL
#        7b: corrupt one counter cell in the Rust stats parquet via SOURCE byte-level edit +
#            RE-READ (Z2 lesson: mutate the SOURCE, re-derive through the production reader).
#            FAIL-CLOSED (R3): pattern-absent → python sys.exit(1) → shell restores .bak + aborts
#            the chain (NO exit-42 SKIP — a sabotage that cannot be applied proves nothing; mirrors 8e).
#        7c: truncate the Java stats parquet → Rust D2 read must FAIL
#        7d: remove the partition-statistics entry from the Rust metadata → Java verify must FAIL
#   8. Incremental (R2-INCR):
#        8a: Java: generate-interop-partition-stats-incr (S1+S2 fixture with SUBTRACT arm)
#        8b: Rust: INCR GEN (test_partition_stats_incr_gen)
#        8c: Java: verify-interop-partition-stats-incr D1 (Java reads Rust incremental stats)
#        8d: Rust: INCR D2 (Rust reads Java incremental stats + compare against java_incr_stats.json)
#        8e: Sabotage SEMANTIC: corrupt a merged counter in the Rust incr stats SOURCE parquet
#            (replace data_record_count=0 for cat=a with a non-zero value) → Java D1 must FAIL (fail-closed)
#   9. UUID exotic type (R2-UUID):
#        9a: Java: generate-interop-partition-stats-uuid
#        9b: Rust: UUID GEN (test_partition_stats_uuid_gen)
#        9c: Java: verify-interop-partition-stats-uuid D1 (Java reads Rust UUID stats)
#        9d: Rust: UUID D2 (Rust reads Java UUID stats + compare against java_uuid_stats.json)
#  10. TIME exotic type (R3):   10a Java gen / 10b Rust GEN / 10c Java D1 / 10d Rust D2.
#  11. FIXED[4] exotic type (R3): 11a Java gen / 11b Rust GEN / 11c Java D1 / 11d Rust D2.
#  12. BINARY exotic type (R3): 12a Java gen / 12b Rust GEN / 12c Java D1 / 12d Rust D2.
#  13. Repeat the full chain (steps 1–12) a second time (chain ×2).
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64, the
# repo's pinned Rust toolchain. Run from anywhere; paths resolve relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-partition-stats"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

run_chain() {
  local chain_num="$1"
  local total_steps=13

  echo "==> [${chain_num}/chains] CHAIN ${chain_num} BEGIN"

  echo "    [1/${total_steps}] Reset the temp dir: ${TMP}"
  rm -rf "${TMP}"
  mkdir -p "${TMP}"

  echo "    [2/${total_steps}] Java: generate-interop-partition-stats (fixture + computeAndWriteStatsFile)"
  run_oracle \
    -Dexec.args=generate-interop-partition-stats \
    -Dinterop.partition_stats.dir="${TMP}"

  # Verify Java emitted the expected files.
  if [[ ! -f "${TMP}/table/metadata/final.metadata.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Java did not emit table/metadata/final.metadata.json"
    exit 1
  fi
  if [[ ! -f "${TMP}/java_stats.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Java did not emit java_stats.json"
    exit 1
  fi
  echo "    [2/${total_steps}] Java generate: OK (table/metadata/final.metadata.json + java_stats.json)"

  echo "    [3/${total_steps}] Rust: GEN (compute_and_write_stats_file + register + expected_stats.json)"
  (
    cd "${REPO_ROOT}"
    ICEBERG_INTEROP_PARTITION_STATS_GEN_DIR="${TMP}" \
      cargo test -p iceberg --test interop_partition_stats test_partition_stats_gen \
      -- --exact --nocapture
  )

  if [[ ! -f "${TMP}/rust_table/metadata/final.metadata.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Rust GEN did not emit rust_table/metadata/final.metadata.json"
    exit 1
  fi
  if [[ ! -f "${TMP}/expected_stats.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Rust GEN did not emit expected_stats.json"
    exit 1
  fi
  echo "    [3/${total_steps}] Rust GEN: OK (rust_table/metadata/final.metadata.json + expected_stats.json)"

  echo "    [4/${total_steps}] Java: verify-interop-partition-stats (D1 — Java reads Rust stats parquet)"
  VERIFY_OUT="$(
    cd "${SCRIPT_DIR}"
    "${MVN}" -o -q compile exec:java \
      -Dexec.args=verify-interop-partition-stats \
      -Dinterop.partition_stats.dir="${TMP}" 2>&1
  )" || true
  echo "${VERIFY_OUT}"
  if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
    || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-partition-stats: 0 failures'; then
    echo "==> FAILED (chain ${chain_num}): Java rejected the Rust-written partition-stats parquet (D1)"
    exit 1
  fi
  echo "    [4/${total_steps}] D1 Java-reads-Rust: PASS"

  echo "    [5/${total_steps}] Rust: D2 (read Java stats parquet, compare against java_stats.json)"
  (
    cd "${REPO_ROOT}"
    ICEBERG_INTEROP_PARTITION_STATS_DIR="${TMP}" \
      cargo test -p iceberg --test interop_partition_stats \
        test_partition_stats_d2_rust_reads_java_file \
      -- --exact --nocapture
  )
  echo "    [5/${total_steps}] D2 Rust-reads-Java: PASS"

  echo "    [6/${total_steps}] Rust: cross-version (V2 Java file read against V3 schema)"
  (
    cd "${REPO_ROOT}"
    ICEBERG_INTEROP_PARTITION_STATS_DIR="${TMP}" \
      cargo test -p iceberg --test interop_partition_stats \
        test_partition_stats_cross_version_v2_file_v3_schema \
      -- --exact --nocapture
  )
  echo "    [6/${total_steps}] Cross-version V2→V3 projection: PASS"

  echo "    [7/${total_steps}] Sabotage battery (Z3)"

  # ---- 7a: truncate the Rust stats parquet → Java D1 verify must FAIL ----
  echo "        7a: truncate Rust stats parquet → Java D1 verify must FAIL"
  RUST_STATS_PATH="$(
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    meta = json.load(f)
current_id = meta.get('current-snapshot-id', -1)
for entry in meta.get('partition-statistics', []):
    if entry.get('snapshot-id') == current_id:
        print(entry['statistics-path'])
        sys.exit(0)
print('NOT_FOUND', file=sys.stderr)
sys.exit(1)
" "${TMP}/rust_table/metadata/final.metadata.json"
  )"
  if [[ -z "${RUST_STATS_PATH}" || "${RUST_STATS_PATH}" == "NOT_FOUND" ]]; then
    echo "==> FAILED (chain ${chain_num}): could not locate Rust stats path in final.metadata.json"
    exit 1
  fi
  echo "        7a: Rust stats path: ${RUST_STATS_PATH}"
  cp "${RUST_STATS_PATH}" "${RUST_STATS_PATH}.bak"
  # Truncate to 10 bytes — guaranteed decode failure.
  head -c 10 "${RUST_STATS_PATH}.bak" > "${RUST_STATS_PATH}"
  SABOTAGE_7A="$(
    cd "${SCRIPT_DIR}"
    "${MVN}" -o -q compile exec:java \
      -Dexec.args=verify-interop-partition-stats \
      -Dinterop.partition_stats.dir="${TMP}" 2>&1
  )" || true
  if echo "${SABOTAGE_7A}" | grep -q 'verify-interop-partition-stats: 0 failures'; then
    echo "==> SABOTAGE 7a FAILED (chain ${chain_num}): truncated stats parquet still passed Java D1"
    cp "${RUST_STATS_PATH}.bak" "${RUST_STATS_PATH}"
    exit 1
  fi
  echo "        7a PASS: truncated Rust stats parquet caused Java D1 to fail as expected"
  cp "${RUST_STATS_PATH}.bak" "${RUST_STATS_PATH}"
  echo "        7a: restored"

  # ---- 7b: corrupt one counter cell via SOURCE byte-level edit + re-derive (Z2 lesson) ----
  # Mutate the SOURCE parquet (the Rust-written stats file): replace the bytes encoding
  # data_record_count=3 (a little-endian int64 0x03000000 00000000) with an incorrect value (0x09).
  # Then RE-READ via the production reader — never post-edit the output JSON.
  echo "        7b: corrupt counter cell in Rust stats parquet (SOURCE edit + re-read → D1 must FAIL)"
  cp "${RUST_STATS_PATH}.bak" "${RUST_STATS_PATH}"
  # Capture the mutate exit WITHOUT letting `set -e` abort the chain before the guard below runs
  # (a bare failing command would short-circuit past the .bak restore). `|| MUTATE_7B_EXIT=$?`.
  MUTATE_7B_EXIT=0
  python3 -c "
import sys

# Read the parquet bytes and flip a data byte in the row group data section.
# Strategy: find the first occurrence of the INT64 little-endian encoding of A_DATA_RECORDS=3
# (0x0300000000000000) and overwrite it with 0x0900000000000000 (9 instead of 3).
# This is a SOURCE-level mutation; the decoder will read a wrong value, which the Java verifier
# will catch as a mismatch against expected_stats.json.
with open(sys.argv[1], 'rb') as f:
    data = bytearray(f.read())

target = bytes([0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
replacement = bytes([0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

idx = data.find(target)
if idx == -1:
    # A sabotage that cannot be applied has proven NOTHING — hard-fail (do NOT skip).
    # (lessons.md: 'A SKIP branch in a sabotage step is a false-green'; mirrors 8e below.)
    # cat=a's data_record_count is 3 in the full-compute fixture, so the literal INT64 0x03..
    # MUST be present; its absence means the parquet encoding changed and 7b no longer
    # corrupts a counter — abort rather than declare a hollow pass.
    print('7b ERROR: counter byte pattern (INT64 3) not found in parquet — sabotage cannot be applied', file=sys.stderr)
    sys.exit(1)

data[idx:idx+8] = replacement

with open(sys.argv[1], 'wb') as f:
    f.write(data)

print(f'7b: mutated int64 at offset {idx}: 3 -> 9')
" "${RUST_STATS_PATH}" || MUTATE_7B_EXIT=$?
  if [[ ${MUTATE_7B_EXIT} -ne 0 ]]; then
    # The mutation could not land. Per the promoted lessons.md rule, a sabotage that cannot be
    # applied MUST hard-fail (never SKIP) — a skip is a false-green. Restore and abort the chain.
    echo "==> SABOTAGE 7b FAILED (chain ${chain_num}): mutation could not be applied (exit ${MUTATE_7B_EXIT})"
    cp "${RUST_STATS_PATH}.bak" "${RUST_STATS_PATH}"
    exit 1
  fi
  SABOTAGE_7B="$(
    cd "${SCRIPT_DIR}"
    "${MVN}" -o -q compile exec:java \
      -Dexec.args=verify-interop-partition-stats \
      -Dinterop.partition_stats.dir="${TMP}" 2>&1
  )" || true
  if echo "${SABOTAGE_7B}" | grep -q 'verify-interop-partition-stats: 0 failures'; then
    echo "==> SABOTAGE 7b FAILED (chain ${chain_num}): corrupted counter still passed Java D1"
    cp "${RUST_STATS_PATH}.bak" "${RUST_STATS_PATH}"
    exit 1
  fi
  echo "        7b PASS: corrupted counter caused Java D1 to fail as expected (re-derive through production reader)"
  cp "${RUST_STATS_PATH}.bak" "${RUST_STATS_PATH}"
  echo "        7b: restored"

  # ---- 7c: truncate the Java stats parquet → Rust D2 must FAIL ----
  echo "        7c: truncate Java stats parquet → Rust D2 must FAIL"
  JAVA_STATS_PATH="$(
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    meta = json.load(f)
current_id = meta.get('current-snapshot-id', -1)
for entry in meta.get('partition-statistics', []):
    if entry.get('snapshot-id') == current_id:
        print(entry['statistics-path'])
        sys.exit(0)
print('NOT_FOUND', file=sys.stderr)
sys.exit(1)
" "${TMP}/table/metadata/final.metadata.json"
  )"
  cp "${JAVA_STATS_PATH}" "${JAVA_STATS_PATH}.bak"
  head -c 10 "${JAVA_STATS_PATH}.bak" > "${JAVA_STATS_PATH}"
  SABOTAGE_7C_OUT="$(
    cd "${REPO_ROOT}"
    ICEBERG_INTEROP_PARTITION_STATS_DIR="${TMP}" \
      cargo test -p iceberg --test interop_partition_stats \
        test_partition_stats_d2_rust_reads_java_file \
      -- --exact --nocapture 2>&1
  )" || true
  if echo "${SABOTAGE_7C_OUT}" | grep -q "PASS\|ok"; then
    if ! echo "${SABOTAGE_7C_OUT}" | grep -qiE "error|panicked|FAILED"; then
      echo "==> SABOTAGE 7c FAILED (chain ${chain_num}): truncated Java stats parquet still passed Rust D2"
      cp "${JAVA_STATS_PATH}.bak" "${JAVA_STATS_PATH}"
      exit 1
    fi
  fi
  echo "        7c PASS: truncated Java stats parquet caused Rust D2 to fail as expected"
  cp "${JAVA_STATS_PATH}.bak" "${JAVA_STATS_PATH}"
  echo "        7c: restored"

  # ---- 7d: remove partition-statistics entry from Rust metadata → Java D1 verify must FAIL ----
  echo "        7d: remove partition-statistics entry from Rust metadata → Java D1 must FAIL"
  RUST_META="${TMP}/rust_table/metadata/final.metadata.json"
  cp "${RUST_META}" "${RUST_META}.bak"
  python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    meta = json.load(f)
# Remove all partition-statistics entries (simulate a table where none are registered).
meta.pop('partition-statistics', None)
with open(sys.argv[1], 'w') as f:
    json.dump(meta, f)
print('7d: removed partition-statistics from Rust metadata')
" "${RUST_META}"
  SABOTAGE_7D="$(
    cd "${SCRIPT_DIR}"
    "${MVN}" -o -q compile exec:java \
      -Dexec.args=verify-interop-partition-stats \
      -Dinterop.partition_stats.dir="${TMP}" 2>&1
  )" || true
  if echo "${SABOTAGE_7D}" | grep -q 'verify-interop-partition-stats: 0 failures'; then
    echo "==> SABOTAGE 7d FAILED (chain ${chain_num}): missing partition-statistics still passed Java D1"
    cp "${RUST_META}.bak" "${RUST_META}"
    exit 1
  fi
  echo "        7d PASS: missing partition-statistics entry caused Java D1 to fail as expected"
  cp "${RUST_META}.bak" "${RUST_META}"
  echo "        7d: restored"

  echo "    [8/${total_steps}] Sabotage battery (Z3): all 4 cases failed closed (7a/7b/7c/7d)"

  # ==================================================================
  # Step 8: Incremental (R2-INCR) — SUBTRACT arm bidirectional
  # ==================================================================
  TMP_INCR="${TMP}/incr"
  mkdir -p "${TMP_INCR}"

  echo "    [8/${total_steps}] Incremental R2-INCR (SUBTRACT arm)"

  echo "        8a: Java: generate-interop-partition-stats-incr"
  run_oracle \
    -Dexec.args=generate-interop-partition-stats-incr \
    -Dinterop.partition_stats_incr.dir="${TMP_INCR}"

  if [[ ! -f "${TMP_INCR}/java_incr_table/metadata/final.metadata.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Java did not emit java_incr_table/metadata/final.metadata.json"
    exit 1
  fi
  if [[ ! -f "${TMP_INCR}/java_incr_stats.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Java did not emit java_incr_stats.json"
    exit 1
  fi
  echo "        8a Java incr generate: OK"

  echo "        8b: Rust: INCR GEN (test_partition_stats_incr_gen)"
  (
    cd "${REPO_ROOT}"
    ICEBERG_INTEROP_PARTITION_STATS_INCR_DIR="${TMP_INCR}" \
      cargo test -p iceberg --test interop_partition_stats test_partition_stats_incr_gen \
      -- --exact --nocapture
  )

  if [[ ! -f "${TMP_INCR}/rust_incr_table/metadata/final.metadata.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Rust INCR GEN did not emit rust_incr_table/metadata/final.metadata.json"
    exit 1
  fi
  if [[ ! -f "${TMP_INCR}/incr_expected.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Rust INCR GEN did not emit incr_expected.json"
    exit 1
  fi
  echo "        8b Rust INCR GEN: OK"

  echo "        8c: Java: verify-interop-partition-stats-incr (D1 — Java reads Rust incr stats)"
  INCR_VERIFY_OUT="$(
    cd "${SCRIPT_DIR}"
    "${MVN}" -o -q compile exec:java \
      -Dexec.args=verify-interop-partition-stats-incr \
      -Dinterop.partition_stats_incr.dir="${TMP_INCR}" 2>&1
  )" || true
  echo "${INCR_VERIFY_OUT}"
  if echo "${INCR_VERIFY_OUT}" | grep -q '^FAIL ' \
    || ! echo "${INCR_VERIFY_OUT}" | grep -q 'verify-interop-partition-stats-incr: 0 failures'; then
    echo "==> FAILED (chain ${chain_num}): Java rejected the Rust-written incremental stats (D1 incr)"
    exit 1
  fi
  echo "        8c D1 Java-reads-Rust-incr: PASS"

  echo "        8d: Rust: INCR D2 (read Java incr stats + compare against java_incr_stats.json)"
  (
    cd "${REPO_ROOT}"
    ICEBERG_INTEROP_PARTITION_STATS_INCR_DIR="${TMP_INCR}" \
      cargo test -p iceberg --test interop_partition_stats \
        test_partition_stats_incr_d2_rust_reads_java \
      -- --exact --nocapture
  )
  echo "        8d D2 Rust-reads-Java-incr: PASS"

  # ---- 8e: SEMANTIC sabotage on incremental merged output ----
  # Corrupt a merged counter in the Rust incr stats SOURCE parquet (Z3 7b pattern applied to the
  # incremental output). Specifically, cat=a's data_record_count should be 0 after the SUBTRACT;
  # we write an incorrect non-zero value (1) via byte-level edit and re-read through the production
  # reader — the Java D1 verify must catch the semantic mismatch.
  echo "        8e: SEMANTIC sabotage — corrupt incr merged counter → Java D1 incr must FAIL"
  RUST_INCR_STATS_PATH="$(
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    meta = json.load(f)
current_id = meta.get('current-snapshot-id', -1)
for entry in meta.get('partition-statistics', []):
    if entry.get('snapshot-id') == current_id:
        print(entry['statistics-path'])
        sys.exit(0)
print('NOT_FOUND', file=sys.stderr)
sys.exit(1)
" "${TMP_INCR}/rust_incr_table/metadata/final.metadata.json"
  )"
  if [[ -z "${RUST_INCR_STATS_PATH}" || "${RUST_INCR_STATS_PATH}" == "NOT_FOUND" ]]; then
    echo "==> FAILED (chain ${chain_num}): could not locate Rust incr stats path"
    exit 1
  fi
  echo "        8e: Rust incr stats path: ${RUST_INCR_STATS_PATH}"
  cp "${RUST_INCR_STATS_PATH}" "${RUST_INCR_STATS_PATH}.bak"
  # Replace the first INT64 little-endian 0x0000000000000000 (0) with 0x0100000000000000 (1).
  # In the incremental output cat=a has data_record_count=0; the first such zero INT64 in the
  # row-group data is the count field. This is a SEMANTIC corruption: the file is still valid
  # parquet but the counter value is wrong — the production reader will decode it and the Java
  # verifier will catch the mismatch against incr_expected.json.
  # Capture the mutate exit WITHOUT letting `set -e` abort before the guard below (`|| ...=$?`).
  MUTATE_8E_EXIT=0
  python3 -c "
import sys

with open(sys.argv[1], 'rb') as f:
    data = bytearray(f.read())

target = bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
replacement = bytes([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

# Find in the row group data section (skip the first 4 bytes = 'PAR1' magic).
idx = data.find(target, 4)
if idx == -1:
    # A sabotage that cannot be applied has proven NOTHING — hard-fail (do NOT skip).
    # (lessons.md: 'A SKIP branch in a sabotage step is a false-green'.) cat=a's
    # data_record_count is 0 after the SUBTRACT, so a literal zero INT64 must be present;
    # its absence means the parquet encoding changed and 8e no longer corrupts a counter.
    print('8e ERROR: zero INT64 pattern not found in incr parquet — sabotage cannot be applied', file=sys.stderr)
    sys.exit(1)

data[idx:idx+8] = replacement

with open(sys.argv[1], 'wb') as f:
    f.write(data)

print(f'8e: mutated INT64 at offset {idx}: 0 -> 1 (SEMANTIC corruption in merged counter)')
" "${RUST_INCR_STATS_PATH}" || MUTATE_8E_EXIT=$?
  if [[ ${MUTATE_8E_EXIT} -ne 0 ]]; then
    # The mutation could not land. Per the promoted lessons.md rule, a sabotage that cannot be
    # applied MUST hard-fail (never SKIP) — a skip is a false-green. Restore and abort the chain.
    echo "==> SABOTAGE 8e FAILED (chain ${chain_num}): mutation could not be applied (exit ${MUTATE_8E_EXIT})"
    cp "${RUST_INCR_STATS_PATH}.bak" "${RUST_INCR_STATS_PATH}"
    exit 1
  fi
  SABOTAGE_8E="$(
    cd "${SCRIPT_DIR}"
    "${MVN}" -o -q compile exec:java \
      -Dexec.args=verify-interop-partition-stats-incr \
      -Dinterop.partition_stats_incr.dir="${TMP_INCR}" 2>&1
  )" || true
  if echo "${SABOTAGE_8E}" | grep -q 'verify-interop-partition-stats-incr: 0 failures'; then
    echo "==> SABOTAGE 8e FAILED (chain ${chain_num}): SEMANTIC corruption still passed Java D1 incr"
    cp "${RUST_INCR_STATS_PATH}.bak" "${RUST_INCR_STATS_PATH}"
    exit 1
  fi
  echo "        8e PASS: SEMANTIC corruption in merged counter caused Java D1 incr to fail as expected"
  cp "${RUST_INCR_STATS_PATH}.bak" "${RUST_INCR_STATS_PATH}"
  echo "        8e: restored"

  echo "    [8/${total_steps}] Incremental R2-INCR: PASS (8a–8e)"

  # ==================================================================
  # Step 9: UUID exotic type (R2-UUID)
  # ==================================================================
  TMP_UUID="${TMP}/uuid"
  mkdir -p "${TMP_UUID}"

  echo "    [9/${total_steps}] UUID exotic type R2-UUID"

  echo "        9a: Java: generate-interop-partition-stats-uuid"
  run_oracle \
    -Dexec.args=generate-interop-partition-stats-uuid \
    -Dinterop.partition_stats_uuid.dir="${TMP_UUID}"

  if [[ ! -f "${TMP_UUID}/java_uuid_table/metadata/final.metadata.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Java did not emit java_uuid_table/metadata/final.metadata.json"
    exit 1
  fi
  if [[ ! -f "${TMP_UUID}/java_uuid_stats.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Java did not emit java_uuid_stats.json"
    exit 1
  fi
  echo "        9a Java UUID generate: OK"

  echo "        9b: Rust: UUID GEN (test_partition_stats_uuid_gen)"
  (
    cd "${REPO_ROOT}"
    ICEBERG_INTEROP_PARTITION_STATS_UUID_DIR="${TMP_UUID}" \
      cargo test -p iceberg --test interop_partition_stats test_partition_stats_uuid_gen \
      -- --exact --nocapture
  )

  if [[ ! -f "${TMP_UUID}/rust_uuid_table/metadata/final.metadata.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Rust UUID GEN did not emit rust_uuid_table/metadata/final.metadata.json"
    exit 1
  fi
  if [[ ! -f "${TMP_UUID}/uuid_expected.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Rust UUID GEN did not emit uuid_expected.json"
    exit 1
  fi
  echo "        9b Rust UUID GEN: OK"

  echo "        9c: Java: verify-interop-partition-stats-uuid (D1 — Java reads Rust UUID stats)"
  UUID_VERIFY_OUT="$(
    cd "${SCRIPT_DIR}"
    "${MVN}" -o -q compile exec:java \
      -Dexec.args=verify-interop-partition-stats-uuid \
      -Dinterop.partition_stats_uuid.dir="${TMP_UUID}" 2>&1
  )" || true
  echo "${UUID_VERIFY_OUT}"
  if echo "${UUID_VERIFY_OUT}" | grep -q '^FAIL ' \
    || ! echo "${UUID_VERIFY_OUT}" | grep -q 'verify-interop-partition-stats-uuid: 0 failures'; then
    echo "==> FAILED (chain ${chain_num}): Java rejected the Rust-written UUID stats (D1 uuid)"
    exit 1
  fi
  echo "        9c D1 Java-reads-Rust-UUID: PASS"

  echo "        9d: Rust: UUID D2 (read Java UUID stats + compare against java_uuid_stats.json)"
  (
    cd "${REPO_ROOT}"
    ICEBERG_INTEROP_PARTITION_STATS_UUID_DIR="${TMP_UUID}" \
      cargo test -p iceberg --test interop_partition_stats \
        test_partition_stats_uuid_d2 \
      -- --exact --nocapture
  )
  echo "        9d D2 Rust-reads-Java-UUID: PASS"

  echo "    [9/${total_steps}] UUID R2-UUID: PASS (9a–9d)"

  # ==================================================================
  # Step 10: TIME exotic type (R3) — identity(partition_time time)
  # ==================================================================
  TMP_TIME="${TMP}/time"
  mkdir -p "${TMP_TIME}"
  echo "    [10/${total_steps}] TIME exotic type R3 (identity(partition_time time))"

  echo "        10a: Java: generate-interop-partition-stats-time"
  run_oracle \
    -Dexec.args=generate-interop-partition-stats-time \
    -Dinterop.partition_stats_time.dir="${TMP_TIME}"
  if [[ ! -f "${TMP_TIME}/java_time_table/metadata/final.metadata.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Java did not emit java_time_table/metadata/final.metadata.json"
    exit 1
  fi
  if [[ ! -f "${TMP_TIME}/java_time_stats.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Java did not emit java_time_stats.json"
    exit 1
  fi
  echo "        10a Java TIME generate: OK"

  echo "        10b: Rust: TIME GEN (test_partition_stats_time_gen)"
  (
    cd "${REPO_ROOT}"
    ICEBERG_INTEROP_PARTITION_STATS_TIME_DIR="${TMP_TIME}" \
      cargo test -p iceberg --test interop_partition_stats test_partition_stats_time_gen \
      -- --exact --nocapture
  )
  if [[ ! -f "${TMP_TIME}/rust_time_table/metadata/final.metadata.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Rust TIME GEN did not emit rust_time_table/metadata/final.metadata.json"
    exit 1
  fi
  if [[ ! -f "${TMP_TIME}/time_expected.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Rust TIME GEN did not emit time_expected.json"
    exit 1
  fi
  echo "        10b Rust TIME GEN: OK"

  echo "        10c: Java: verify-interop-partition-stats-time (D1 — Java reads Rust TIME stats)"
  TIME_VERIFY_OUT="$(
    cd "${SCRIPT_DIR}"
    "${MVN}" -o -q compile exec:java \
      -Dexec.args=verify-interop-partition-stats-time \
      -Dinterop.partition_stats_time.dir="${TMP_TIME}" 2>&1
  )" || true
  echo "${TIME_VERIFY_OUT}"
  if echo "${TIME_VERIFY_OUT}" | grep -q '^FAIL ' \
    || ! echo "${TIME_VERIFY_OUT}" | grep -q 'verify-interop-partition-stats-time: 0 failures'; then
    echo "==> FAILED (chain ${chain_num}): Java rejected the Rust-written TIME stats (D1 time)"
    exit 1
  fi
  echo "        10c D1 Java-reads-Rust-TIME: PASS"

  echo "        10d: Rust: TIME D2 (read Java TIME stats + compare against java_time_stats.json)"
  (
    cd "${REPO_ROOT}"
    ICEBERG_INTEROP_PARTITION_STATS_TIME_DIR="${TMP_TIME}" \
      cargo test -p iceberg --test interop_partition_stats \
        test_partition_stats_time_d2 \
      -- --exact --nocapture
  )
  echo "        10d D2 Rust-reads-Java-TIME: PASS"
  echo "    [10/${total_steps}] TIME R3: PASS (10a–10d)"

  # ==================================================================
  # Step 11: FIXED[4] exotic type (R3) — identity(partition_fixed fixed[4])
  # ==================================================================
  TMP_FIXED="${TMP}/fixed"
  mkdir -p "${TMP_FIXED}"
  echo "    [11/${total_steps}] FIXED exotic type R3 (identity(partition_fixed fixed[4]))"

  echo "        11a: Java: generate-interop-partition-stats-fixed"
  run_oracle \
    -Dexec.args=generate-interop-partition-stats-fixed \
    -Dinterop.partition_stats_fixed.dir="${TMP_FIXED}"
  if [[ ! -f "${TMP_FIXED}/java_fixed_table/metadata/final.metadata.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Java did not emit java_fixed_table/metadata/final.metadata.json"
    exit 1
  fi
  if [[ ! -f "${TMP_FIXED}/java_fixed_stats.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Java did not emit java_fixed_stats.json"
    exit 1
  fi
  echo "        11a Java FIXED generate: OK"

  echo "        11b: Rust: FIXED GEN (test_partition_stats_fixed_gen)"
  (
    cd "${REPO_ROOT}"
    ICEBERG_INTEROP_PARTITION_STATS_FIXED_DIR="${TMP_FIXED}" \
      cargo test -p iceberg --test interop_partition_stats test_partition_stats_fixed_gen \
      -- --exact --nocapture
  )
  if [[ ! -f "${TMP_FIXED}/rust_fixed_table/metadata/final.metadata.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Rust FIXED GEN did not emit rust_fixed_table/metadata/final.metadata.json"
    exit 1
  fi
  if [[ ! -f "${TMP_FIXED}/fixed_expected.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Rust FIXED GEN did not emit fixed_expected.json"
    exit 1
  fi
  echo "        11b Rust FIXED GEN: OK"

  echo "        11c: Java: verify-interop-partition-stats-fixed (D1 — Java reads Rust FIXED stats)"
  FIXED_VERIFY_OUT="$(
    cd "${SCRIPT_DIR}"
    "${MVN}" -o -q compile exec:java \
      -Dexec.args=verify-interop-partition-stats-fixed \
      -Dinterop.partition_stats_fixed.dir="${TMP_FIXED}" 2>&1
  )" || true
  echo "${FIXED_VERIFY_OUT}"
  if echo "${FIXED_VERIFY_OUT}" | grep -q '^FAIL ' \
    || ! echo "${FIXED_VERIFY_OUT}" | grep -q 'verify-interop-partition-stats-fixed: 0 failures'; then
    echo "==> FAILED (chain ${chain_num}): Java rejected the Rust-written FIXED stats (D1 fixed)"
    exit 1
  fi
  echo "        11c D1 Java-reads-Rust-FIXED: PASS"

  echo "        11d: Rust: FIXED D2 (read Java FIXED stats + compare against java_fixed_stats.json)"
  (
    cd "${REPO_ROOT}"
    ICEBERG_INTEROP_PARTITION_STATS_FIXED_DIR="${TMP_FIXED}" \
      cargo test -p iceberg --test interop_partition_stats \
        test_partition_stats_fixed_d2 \
      -- --exact --nocapture
  )
  echo "        11d D2 Rust-reads-Java-FIXED: PASS"
  echo "    [11/${total_steps}] FIXED R3: PASS (11a–11d)"

  # ==================================================================
  # Step 12: BINARY exotic type (R3) — identity(partition_binary binary)
  # ==================================================================
  TMP_BINARY="${TMP}/binary"
  mkdir -p "${TMP_BINARY}"
  echo "    [12/${total_steps}] BINARY exotic type R3 (identity(partition_binary binary))"

  echo "        12a: Java: generate-interop-partition-stats-binary"
  run_oracle \
    -Dexec.args=generate-interop-partition-stats-binary \
    -Dinterop.partition_stats_binary.dir="${TMP_BINARY}"
  if [[ ! -f "${TMP_BINARY}/java_binary_table/metadata/final.metadata.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Java did not emit java_binary_table/metadata/final.metadata.json"
    exit 1
  fi
  if [[ ! -f "${TMP_BINARY}/java_binary_stats.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Java did not emit java_binary_stats.json"
    exit 1
  fi
  echo "        12a Java BINARY generate: OK"

  echo "        12b: Rust: BINARY GEN (test_partition_stats_binary_gen)"
  (
    cd "${REPO_ROOT}"
    ICEBERG_INTEROP_PARTITION_STATS_BINARY_DIR="${TMP_BINARY}" \
      cargo test -p iceberg --test interop_partition_stats test_partition_stats_binary_gen \
      -- --exact --nocapture
  )
  if [[ ! -f "${TMP_BINARY}/rust_binary_table/metadata/final.metadata.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Rust BINARY GEN did not emit rust_binary_table/metadata/final.metadata.json"
    exit 1
  fi
  if [[ ! -f "${TMP_BINARY}/binary_expected.json" ]]; then
    echo "==> FAILED (chain ${chain_num}): Rust BINARY GEN did not emit binary_expected.json"
    exit 1
  fi
  echo "        12b Rust BINARY GEN: OK"

  echo "        12c: Java: verify-interop-partition-stats-binary (D1 — Java reads Rust BINARY stats)"
  BINARY_VERIFY_OUT="$(
    cd "${SCRIPT_DIR}"
    "${MVN}" -o -q compile exec:java \
      -Dexec.args=verify-interop-partition-stats-binary \
      -Dinterop.partition_stats_binary.dir="${TMP_BINARY}" 2>&1
  )" || true
  echo "${BINARY_VERIFY_OUT}"
  if echo "${BINARY_VERIFY_OUT}" | grep -q '^FAIL ' \
    || ! echo "${BINARY_VERIFY_OUT}" | grep -q 'verify-interop-partition-stats-binary: 0 failures'; then
    echo "==> FAILED (chain ${chain_num}): Java rejected the Rust-written BINARY stats (D1 binary)"
    exit 1
  fi
  echo "        12c D1 Java-reads-Rust-BINARY: PASS"

  echo "        12d: Rust: BINARY D2 (read Java BINARY stats + compare against java_binary_stats.json)"
  (
    cd "${REPO_ROOT}"
    ICEBERG_INTEROP_PARTITION_STATS_BINARY_DIR="${TMP_BINARY}" \
      cargo test -p iceberg --test interop_partition_stats \
        test_partition_stats_binary_d2 \
      -- --exact --nocapture
  )
  echo "        12d D2 Rust-reads-Java-BINARY: PASS"
  echo "    [12/${total_steps}] BINARY R3: PASS (12a–12d)"

  echo "    [13/${total_steps}] Chain complete"
  echo "==> CHAIN ${chain_num} COMPLETE"
}

# ===========================================================================================
# Run the chain twice (chain ×2).
# ===========================================================================================

run_chain 1
run_chain 2

echo ""
echo "==> DONE — Partition-stats file interop PASSED (chain ×2):"
echo ""
echo "    Z3 base fixture: V2 table identity(category),"
echo "        S1 fast-append (cat=a 3rec + cat=b 2rec), S2 row-delta pos-delete (cat=a 1rec)"
echo "    D1 (Rust writes, Java judges): 0 failures (both chains)"
echo "    D2 (Java writes, Rust reads): all rows matched (both chains)"
echo "    Cross-version V2→V3: dv_count null-filled to 0 for all rows (both chains)"
echo "    Sabotage battery: 4 corruptions failed closed (7a truncate Rust, 7b corrupt counter"
echo "        [R3: fail-closed — pattern-absent aborts, no SKIP], 7c truncate Java,"
echo "        7d remove partition-statistics metadata entry)"
echo ""
echo "    R2-INCR (SUBTRACT arm): V2 table identity(category),"
echo "        S1 fast-append → FULL stats; S2 delete_files(file_a) → INCREMENTAL stats"
echo "        cat=a: data_records=0 (subtracted), cat=b: data_records=2 (unchanged from base)"
echo "    D1 incr: Java reads Rust incremental stats — PASS (both chains)"
echo "    D2 incr: Rust reads Java incremental stats — PASS (both chains)"
echo "    Sabotage 8e SEMANTIC: corrupted merged counter failed closed (both chains)"
echo ""
echo "    R2-UUID (exotic type): V2 table identity(partition_id uuid),"
echo "        known UUID 550e8400-e29b-41d4-a716-446655440000, 5 records, 500 bytes"
echo "    D1 uuid: Java reads Rust UUID stats — PASS (both chains)"
echo "    D2 uuid: Rust reads Java UUID stats — PASS (both chains)"
echo ""
echo "    R3 exotic types (the remaining residue — time / fixed / binary):"
echo "        time   identity(partition_time)   = 45296789012 micros (Time64(Microsecond))"
echo "        fixed  identity(partition_fixed)  = fixed[4] 0xdeadbeef (FixedSizeBinary(4))"
echo "        binary identity(partition_binary) = binary 0x0102030405 (LargeBinary)"
echo "    D1 (Java reads Rust stats) + D2 (Rust reads Java stats): PASS for all three (both chains)"
