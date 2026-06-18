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
# DIRECTION-2 AVRO DATA-FILE WRITE interop harness (GAP_MATRIX row 117) — "JAVA READS WHAT RUST WRITES".
# The mirror image (and INVERSION) of run-interop-avro-data.sh (Direction 1, where Java writes the Avro
# data file and Rust reads it): here RUST writes 00000-rust-data.avro THROUGH THE W1 PRODUCTION Avro data
# writer (AvroWriterBuilder/AvroWriter) and JAVA reads that raw file, proving the Rust-written Avro data
# file is Java-readable byte-for-byte (every primitive + logical type + an optional/null column).
#
# This is a TEST-ONLY ORACLE (a dev tool, like dev/spark/) — NOT part of the shipped Rust library, and
# NOT in the offline `cargo test` gate (it needs Java + Maven). Nothing binary is committed — the
# committed artifacts are the oracle code (InteropOracle.java AvroWriteOracle), the Rust GEN test
# (crates/iceberg/tests/interop_avro_write.rs), and this run script. The temp dir under
# dev/java-interop/target/ is gitignored.
#
# ANTI-CIRCULAR (two anchors). The .avro file is the ONLY artifact that crosses the engine boundary.
# Rust emits rust_avro_rows.json = the rows HAND-DECLARED from its literal constants (NOT by reading back
# the .avro). Java reads the raw .avro and asserts row-identity against BOTH (a) Java's OWN hand-declared
# constants AND (b) rust_avro_rows.json. Rust never reads its own .avro to build the expected.
#
# Methodology (Rust writes → Java reads-and-verifies):
#   1. ICEBERG_INTEROP_AVRO_WRITE_DIR="$TMP" cargo test ... interop_avro_write
#        -> The env-gated Rust GEN test writes 00000-rust-data.avro via the W1 PRODUCTION writer
#           (AvroWriterBuilder::new(schema).build(FileIO::new_with_fs().new_output(path)) -> write(batch)
#           -> close()) over 5 rows (id 10..50; name id=30 NULL) + rust_avro_rows.json. (Without the env
#           var the test is a clean no-op, so the offline gate stays green.)
#   2. mvn ... -Dexec.args=verify-interop-avro-write -Dinterop.avro_write.dir="$TMP"
#        -> The Java oracle reads the raw .avro (Avro.read(...).project(schema)
#           .createReaderFunc(PlannedDataReader::create).build()) and asserts ROW-IDENTITY vs both
#           anchors. A FAIL is a REAL write-incompatibility finding.
#   3. Fail-closed sabotage: on a SCRATCH copy of the temp dir (so the clean tree is untouched), after
#      the clean verify already PASSED, corrupt ONE crossing value in rust_avro_rows.json (the id=30
#      decimal 78.90 -> 99.99). The corruption target MUST be present — if absent we HARD-FAIL (never
#      SKIP). Java's read of the (unchanged) .avro still yields 78.90, so anchor (b) MUST diverge while
#      anchor (a) still PASSES — i.e. the SPECIFIC expected failure, not any failure.
#
# `mvn -q exec:java` SWALLOWS System.exit(1), so the SUCCESS GATE greps the captured output for the
# "verify-interop-avro-write: 0 failures" sentinel AND asserts NO '^FAIL ' line — it does NOT trust
# mvn's exit code.
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64;
# iceberg-core 1.10.0 in ~/.m2 (carries org.apache.iceberg.avro.Avro +
# org.apache.iceberg.data.avro.PlannedDataReader). If the local ~/.m2 cache is populated `mvn -o` runs
# fully offline (the FIRST run may need to be ONLINE).
#
# Run from anywhere; paths resolve relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-avro-write"
ROWS_JSON="${TMP}/rust_avro_rows.json"

echo "==> [1/5] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/5] Rust GEN: write 00000-rust-data.avro via the W1 PRODUCTION writer + emit rust_avro_rows.json"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_AVRO_WRITE_DIR="${TMP}" \
    cargo test -p iceberg --test interop_avro_write -- --nocapture
)

if [[ ! -f "${TMP}/00000-rust-data.avro" ]]; then
  echo "==> FAILED — ${TMP}/00000-rust-data.avro is missing (the Rust GEN step did not write it)."
  exit 1
fi
if [[ ! -f "${ROWS_JSON}" ]]; then
  echo "==> FAILED — ${ROWS_JSON} is missing (the Rust GEN step did not emit it)."
  exit 1
fi

echo "==> [3/5] Java VERIFY: read the raw Rust-written .avro, assert row-identity vs BOTH anchors"
# NOTE: `mvn -q exec:java` does NOT propagate the verify's `System.exit(1)`, so capture the output and
# assert the success sentinel ("...: 0 failures") with no per-check FAIL line.
VERIFY_OUT="$(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
    /opt/maven/bin/mvn -o -q compile exec:java \
    -Dexec.args=verify-interop-avro-write \
    -Dinterop.avro_write.dir="${TMP}" 2>&1
)"
echo "${VERIFY_OUT}"
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-avro-write: 0 failures'; then
  echo "==> FAILED — Java could not correctly read the Rust-written .avro (a real write-incompatibility finding)."
  exit 1
fi

echo "==> [4/5] Fail-closed sabotage: corrupt ONE crossing value in rust_avro_rows.json (on a SCRATCH copy) and confirm anchor (b) DIVERGES"
SAB="${SCRIPT_DIR}/target/interop-avro-write-sabotage"
rm -rf "${SAB}"
cp -r "${TMP}" "${SAB}"
SAB_JSON="${SAB}/rust_avro_rows.json"

# Mutate the id=30 row's decimal value (78.90 -> 99.99). The Rust emitter writes `"dec": "78.90"`
# (compact, one space after the colon). The target MUST be present; if absent we HARD-FAIL (a sabotage
# that cannot corrupt anything proves nothing — never SKIP). Capture the mutator's exit under set -e.
if ! grep -q '"dec": "78.90"' "${SAB_JSON}"; then
  echo "==> FAILED — sabotage target '\"dec\": \"78.90\"' not found in ${SAB_JSON}; cannot corrupt (would be a false-green SKIP)."
  rm -rf "${SAB}"
  exit 1
fi
rc=0
sed -i 's/"dec": "78.90"/"dec": "99.99"/' "${SAB_JSON}" || rc=$?
if [[ "${rc}" -ne 0 ]]; then
  echo "==> FAILED — could not apply the sabotage mutation (sed exit ${rc})."
  rm -rf "${SAB}"
  exit 1
fi

# Re-run the verify against the SABOTAGED dir; capture output + exit so the cleanup stays reachable.
SAB_OUT="$(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
    /opt/maven/bin/mvn -o -q compile exec:java \
    -Dexec.args=verify-interop-avro-write \
    -Dinterop.avro_write.dir="${SAB}" 2>&1
)" || true
rm -rf "${SAB}"

# Require the SPECIFIC expected divergence: anchor (b) (the rust_avro_rows.json comparison) MUST FAIL,
# while anchor (a) (Java's own constants) MUST still PASS — proving Java's read is correct and the
# divergence is precisely the corrupted Rust JSON, not some unrelated breakage.
if echo "${SAB_OUT}" | grep -q "verify-interop-avro-write: 0 failures"; then
  echo "==> FAILED — the Java verify PASSED against a CORRUPTED rust_avro_rows.json (dec 78.90 -> 99.99). NOT fail-closed."
  echo "${SAB_OUT}"
  exit 1
fi
if ! echo "${SAB_OUT}" | grep -q "FAIL avro-write-d2: Java's read of the Rust .avro != the Rust-emitted"; then
  echo "==> FAILED — the verify diverged, but NOT via the expected anchor-(b) mismatch. Output:"
  echo "${SAB_OUT}"
  exit 1
fi
if ! echo "${SAB_OUT}" | grep -q "PASS avro-write-d2: Java's read == Java's OWN hand-declared expected (anchor a)"; then
  echo "==> FAILED — sabotage broke anchor (a) too; the divergence is not isolated to the corrupted Rust JSON. Output:"
  echo "${SAB_OUT}"
  exit 1
fi
echo "PASS sabotage: anchor (b) DIVERGED on the corrupted rust_avro_rows.json while anchor (a) still PASSED (Java's read of the .avro is correct)."

echo "==> [5/5] DONE — Direction-2 Avro data-file WRITE interop passed (Java read the RUST-written .avro via the production Avro read route; all 5 rows {10,20,30,40,50} identical to BOTH anchors; fail-closed sabotage proven)."
