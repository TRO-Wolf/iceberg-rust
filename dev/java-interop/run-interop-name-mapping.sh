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
# NAME-MAPPING scan-wiring interop, DIRECTION 1 ("Rust reads what JAVA wrote") — GAP_MATRIX row R143.
#
# The Java oracle writes a V2 UNPARTITIONED table {1: id long, 2: val long} over a single ID-LESS parquet
# data file (AvroParquetWriter over a plain Avro schema ⇒ NO Iceberg field ids — the add_files migration
# shape) whose PHYSICAL column order is REVERSED relative to the schema ([val, id]). Rows id=[10,20,30],
# val=[100,200,300]. It sets `schema.name-mapping.default`. It emits TWO fixtures under <dir>:
#   normal/   — CORRECT mapping (id->1, val->2): a name-mapped read yields the right columns.
#   sabotage/ — the two mapped names SWAPPED (id->2, val->1): a name-mapped read transposes the columns.
#
# THE CHAIN (Direction 1 + a fail-closed sabotage):
#   [1/4] Reset the temp dir.
#   [2/4] Java GENERATE: write normal/ + sabotage/ tables + java_name_mapping_rows.json (ground truth,
#         by construction — Java core's own reader cannot name-map an id-less file).
#   [3/4] Rust D1 over normal/: TableScan → Arrow, assert (id,val) == the Java ground truth. Only a
#         field-id-by-name resolution recovers the reversed columns; a positional fallback transposes
#         them, so a PASS here proves the name mapping is genuinely consulted.
#   [4/4] SABOTAGE (fail-closed): rerun the SAME Rust verifier over sabotage/ and REQUIRE it to go RED
#         (the swapped mapping transposes the columns, so the ground-truth assert must FAIL). A sabotage
#         that PASSED would mean the mapping is not load-bearing — HARD-FAIL (never SKIP).
#
# This is a TEST-ONLY ORACLE (a dev tool) — NOT part of the shipped Rust library, NOT part of the offline
# `cargo test` gate (it needs Java + Maven). Nothing binary is committed; the temp dir under
# dev/java-interop/target/ is gitignored. Without ICEBERG_INTEROP_NAME_MAPPING_DIR the Rust test is a
# clean no-op; this script flips it into the REAL comparison.
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64, a Rust
# toolchain. The pom's parquet-avro + hadoop-client-api (compile) deps are cached in ~/.m2 after the
# first online prime; thereafter `mvn -o` runs fully offline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-name-mapping"

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"
MVN="/opt/maven/bin/mvn"

echo "==> [1/4] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/4] Java GENERATE: write normal/ + sabotage/ id-less name-mapping tables + ground truth"
(
  cd "${SCRIPT_DIR}"
  "${MVN}" -o -q compile exec:java \
    -Dexec.args=generate-interop-name-mapping \
    -Dinterop.name_mapping.dir="${TMP}"
)
test -f "${TMP}/normal/table/metadata/final.metadata.json" \
  || { echo "FAIL: normal/ final.metadata.json not produced"; exit 1; }
test -f "${TMP}/sabotage/table/metadata/final.metadata.json" \
  || { echo "FAIL: sabotage/ final.metadata.json not produced"; exit 1; }
echo "    normal/ + sabotage/ fixtures produced OK"

echo "==> [3/4] Rust D1 over normal/: TableScan → Arrow must equal the Java ground truth"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_NAME_MAPPING_DIR="${TMP}/normal" \
    cargo test -p iceberg --test interop_name_mapping test_name_mapping_scan_matches_java -- --nocapture
)
echo "    Direction-1 OK — Rust's name-mapped scan matches Java (reversed columns resolved by field id)"

echo ""
echo "==> [4/4] SABOTAGE (fail-closed): the SAME verifier over sabotage/ (swapped mapping) must go RED"
# Capture the mutator's exit so `set -e` does not abort before we can judge it (no .bak to restore here,
# but the capture keeps the fail-closed logic reachable exactly as the interop doctrine requires).
sab_rc=0
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_NAME_MAPPING_DIR="${TMP}/sabotage" \
    cargo test -p iceberg --test interop_name_mapping test_name_mapping_scan_matches_java -- --nocapture
) || sab_rc=$?

if [ "${sab_rc}" -eq 0 ]; then
  echo "==> SABOTAGE FAILED (VACUOUS): the swapped mapping STILL read the correct columns — the name"
  echo "    mapping is not load-bearing (a positional read would match too). HARD-FAIL."
  exit 1
fi
echo "    SABOTAGE PASS: the swapped-mapping table read transposed columns ⇒ verifier RED (fail-closed)"

echo ""
echo "==> DONE — name-mapping interop passed (row R143): Java wrote an ID-LESS reversed-column parquet +"
echo "     schema.name-mapping.default; Rust's TableScan resolved the columns by field id (Direction 1),"
echo "     and the swapped-mapping sabotage went RED."
