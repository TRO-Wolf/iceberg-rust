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
# AVRO DATA-FILE READ interop harness (GAP_MATRIX row 117, Direction 1 — "Rust reads what Java
# writes"). Proves Rust's scan → Arrow over an AVRO DATA FILE (with a merge-on-read position delete
# applied) matches Java's OWN read of a JAVA-WRITTEN table whose data file is in the AVRO format.
#
# Unlike the parquet scan-exec harness, the DATA FILE here is written by Java's production Avro data
# writer (GenericAppenderFactory.newDataWriter(..., FileFormat.AVRO, null) →
# org.apache.iceberg.data.avro.DataWriter) over a fixture covering every Iceberg primitive + logical
# type + an optional/null column. A real parquet POSITION-delete removes one row (position 1, the
# id=20 row). (Nested struct/list/map are NOT in this fixture — Java's own GenericAppenderFactory AVRO
# writer + PlannedDataReader cannot round-trip a nested record; nested READ is proven by the U1 offline
# tests in crates/iceberg/src/arrow/avro_reader_tests.rs.)
#
# This is a TEST-ONLY ORACLE (a dev tool) — NOT part of the shipped library and NOT in the offline
# `cargo test` gate (it needs Java + Maven). The committed artifacts are the oracle code
# (InteropOracle.java AvroDataOracle), the Rust test (interop_avro_data.rs), and this script. The temp
# table under dev/java-interop/target/ is gitignored.
#
# Methodology (regenerate-and-compare, ANTI-CIRCULAR):
#   1. Java generate: AvroDataOracle writes the V2 table (AVRO data file + parquet position-delete),
#      ASSERTS its OWN IcebergGenerics read == the hand-declared expected (a Java-side bug fails right
#      there), and emits java_avro_rows.json = the hand-declared expected (4 live rows: id 10/30/40/50).
#   2. Rust compare: the env-gated test loads final.metadata.json, scans → Arrow over the AVRO data
#      file (applying the position delete), extracts every column into the SAME canonical shape, and
#      asserts ROW-IDENTITY vs java_avro_rows.json. Expected is hand-declared identically on BOTH sides.
#   3. Fail-closed sabotage: corrupt ONE value in the expected JSON and re-run the Rust compare — it
#      MUST diverge and hard-fail (never SKIP). Restores the JSON before asserting so reruns are clean.
#
# Without ICEBERG_INTEROP_AVRO_DATA_DIR the Rust test is a clean no-op (green in the offline gate).
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64;
# iceberg-data / iceberg-parquet 1.10.0 in ~/.m2 (the FIRST run may need to be ONLINE).
#
# Run from anywhere; paths resolve relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-avro-data"
ROWS_JSON="${TMP}/java_avro_rows.json"

echo "==> [1/5] Reset the temp table dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/5] Java oracle: write a V2 table whose DATA file is AVRO (+ parquet position-delete) + emit java_avro_rows.json"
(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
    /opt/maven/bin/mvn -o -q compile exec:java \
    -Dexec.args=generate-interop-avro-data \
    -Dinterop.avro_data.dir="${TMP}"
)

echo "==> [3/5] Rust: load final.metadata.json, scan → Arrow over the AVRO data file (merge-on-read), compare vs Java's expected"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_AVRO_DATA_DIR="${TMP}" \
    cargo test -p iceberg --test interop_avro_data -- --nocapture
)

echo "==> [4/5] Fail-closed sabotage: corrupt one value in the expected JSON and confirm the Rust compare DIVERGES"
if [[ ! -f "${ROWS_JSON}" ]]; then
  echo "==> FAILED — ${ROWS_JSON} is missing; cannot run the sabotage (the generate step did not emit it)."
  exit 1
fi
cp "${ROWS_JSON}" "${ROWS_JSON}.bak"

# Mutate the surviving id=30 row's decimal value (78.90 -> 99.99). The JSON is Jackson pretty-printed
# (`"dec" : "78.90"`, spaces around the colon). The mutation MUST be present; if the target pattern is
# absent we HARD-FAIL (a sabotage that cannot be applied proves nothing — never SKIP).
if ! grep -q '"dec" : "78.90"' "${ROWS_JSON}"; then
  echo "==> FAILED — sabotage target '\"dec\" : \"78.90\"' not found in ${ROWS_JSON}; cannot corrupt (would be a false-green SKIP)."
  mv "${ROWS_JSON}.bak" "${ROWS_JSON}"
  exit 1
fi
sed -i 's/"dec" : "78.90"/"dec" : "99.99"/' "${ROWS_JSON}"

# Run the Rust compare against the corrupted expected; capture its exit so the restore stays reachable
# under `set -e`.
rc=0
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_AVRO_DATA_DIR="${TMP}" \
    cargo test -p iceberg --test interop_avro_data -- --nocapture
) || rc=$?

# Restore BEFORE asserting so a failed assertion still leaves a clean tree for reruns.
mv "${ROWS_JSON}.bak" "${ROWS_JSON}"

if [[ "${rc}" -eq 0 ]]; then
  echo "==> FAILED — the Rust compare PASSED against a CORRUPTED expected (dec 78.90 -> 99.99). NOT fail-closed."
  exit 1
fi
echo "PASS sabotage: the Rust compare correctly DIVERGED (exit ${rc}) on a corrupted expected value."

echo "==> [5/5] DONE — Avro data-file read interop passed (Rust scan over an AVRO data file == Java's read; 4 live rows {10,30,40,50}, id=20 deleted; every column identical; fail-closed sabotage proven)."
