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
# NON-IDENTITY TRANSFORM data-level scan-execution interop, DIRECTION 2 ("Java reads what RUST writes").
# The reverse of run-interop-nonidentity-scan.sh: Rust's GEN test writes a REAL on-disk {id, data} V2
# table partitioned by truncate[10](id) to <dir>/rust_table (two data files routed to truncate=10 and
# truncate=20 fast_appended at sequence 1, plus a partition-scoped position-delete on the truncate=10 file
# row_delta'd at sequence 2), landing a final.metadata.json. Java then loads that RUST-written metadata,
# reads via IcebergGenerics (which APPLIES Rust's transform-partition-scoped position delete), and asserts
# the merge-on-read rows == {11,15,21,23} (truncate=10's id 13 deleted, truncate=20 intact). A failure here
# is a REAL transform-aware write-incompatibility finding.
#
# TEST-ONLY ORACLE (dev tool); not in the offline gate; nothing binary committed; temp dir gitignored.
# Without ICEBERG_INTEROP_NONIDENTITY_SCAN_GEN_DIR the Rust GEN test is a clean no-op.
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64, a Rust toolchain.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP2="${SCRIPT_DIR}/target/interop-nonidentity-d2"
echo "==> [1/3] Reset the temp table dir: ${TMP2}"
rm -rf "${TMP2}"
mkdir -p "${TMP2}"
echo "==> [2/3] Rust: WRITE a REAL truncate[10](id)-partitioned V2 table (two transform partitions seq 1 + a transform-partition-scoped position-delete seq 2) + final.metadata.json"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_NONIDENTITY_SCAN_GEN_DIR="${TMP2}" \
    cargo test -p iceberg --test interop_scan_exec test_nonidentity_scan_exec_gen_rust_writes_java_readable_table -- --nocapture
)
echo "==> [3/3] Java: load the RUST-written final.metadata.json, read via IcebergGenerics, verify {11,15,21,23}"
VERIFY_OUT="$(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
    /opt/maven/bin/mvn -o -q compile exec:java \
    -Dexec.args=verify-interop-nonidentity-scan \
    -Dinterop.nonidentity_scan.dir="${TMP2}" 2>&1
)"
echo "${VERIFY_OUT}"
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' || ! echo "${VERIFY_OUT}" | grep -q ': 0 failures'; then
  echo "==> FAILED — Java could not correctly read the Rust-written truncate-partitioned table (a real transform-aware write-incompatibility finding)."
  exit 1
fi
echo "==> DONE — Direction-2 non-identity round-trip passed (Java read the Rust-written truncate-partitioned table, live rows {11,15,21,23}, truncate=10 id 13 deleted, truncate=20 intact)."
