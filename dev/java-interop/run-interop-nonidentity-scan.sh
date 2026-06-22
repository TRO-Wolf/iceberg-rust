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
# NON-IDENTITY TRANSFORM data-level scan-execution interop, DIRECTION 1 ("Rust reads what JAVA writes").
# The transform slice of the GAP_MATRIX "Read: merge-on-read apply" residue. Unlike run-interop-part.sh
# (identity(category)), Java writes a {id long, data string} V2 table partitioned by truncate[10](id) —
# the partition VALUE (10 / 20) is floor(id/10)*10, a TRANSFORMED value no raw id equals. Partition 10
# holds ids 11/13/15; partition 20 holds 21/23. A partition-scoped position-delete in partition 10 deletes
# position 1 (id 13). Live merge-on-read rows = {11,15,21,23}. Proves Rust's scan matches the delete to
# the TRANSFORMED partition Struct, exactly as Java's own read does (partition 20 untouched).
#
# TEST-ONLY ORACLE (dev tool); not in the offline gate; nothing binary committed; temp dir gitignored.
# Without ICEBERG_INTEROP_NONIDENTITY_SCAN_DIR the Rust test is a clean no-op.
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64, a Rust toolchain.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-nonidentity"
echo "==> [1/3] Reset the temp table dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"
echo "==> [2/3] Java oracle: write a truncate[10](id)-partitioned V2 table (transform-partition-scoped position-delete on truncate=10) + emit java_nonidentity_scan_rows.json"
(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
    /opt/maven/bin/mvn -o -q compile exec:java \
    -Dexec.args=generate-interop-nonidentity-scan \
    -Dinterop.nonidentity_scan.dir="${TMP}"
)
echo "==> [3/3] Rust: load final.metadata.json, scan → Arrow (transform-partition merge-on-read), compare vs Java's read"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_NONIDENTITY_SCAN_DIR="${TMP}" \
    cargo test -p iceberg --test interop_scan_exec test_nonidentity_scan_exec_matches_java_read -- --nocapture
)
echo "==> DONE — non-identity-transform scan interop passed (Rust scan == Java read, live rows {11,15,21,23}, truncate=10 id 13 deleted, truncate=20 intact)."
