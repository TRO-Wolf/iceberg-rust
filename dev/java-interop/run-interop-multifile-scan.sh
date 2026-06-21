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
# MULTI-FILE-PER-PARTITION data-level scan-execution interop, DIRECTION 1 ("Rust reads what JAVA writes").
# The multi-file slice of the GAP_MATRIX "Read: merge-on-read apply" residue. Unlike run-interop-part.sh
# (one data file per partition), Java writes ONE partition (category=a) holding TWO data files and a
# PARTITION-SCOPED position-delete referencing file1 only (deleting its position 1 = id 20). The
# DeleteFileIndex routes that partition-scoped delete to BOTH data files of partition a, so the proof is
# that Rust's read applies it ONLY to file1 by PATH — file2's same-ordinal position 1 (id 50) is SPARED.
# Live merge-on-read rows = {10,30,40,50,60}.
#
# This is a TEST-ONLY ORACLE (a dev tool) — NOT part of the shipped Rust library, NOT part of the offline
# `cargo test` gate (it needs Java + Maven). Nothing binary is committed; the temp table under
# dev/java-interop/target/ is gitignored. Without ICEBERG_INTEROP_MULTIFILE_SCAN_DIR the Rust test is a
# clean no-op; this script flips it into the REAL comparison.
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64, a Rust
# toolchain. The Maven deps are the SAME the scan-exec harness already pulled — no new pom deps.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-multifile"
echo "==> [1/3] Reset the temp table dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"
echo "==> [2/3] Java oracle: write a MULTI-FILE-PER-PARTITION V2 table (two data files in category=a + a partition-scoped position-delete on file1) + emit java_multifile_scan_rows.json"
(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
    /opt/maven/bin/mvn -o -q compile exec:java \
    -Dexec.args=generate-interop-multifile-scan \
    -Dinterop.multifile_scan.dir="${TMP}"
)
echo "==> [3/3] Rust: load final.metadata.json, scan → Arrow (path-keyed merge-on-read), compare vs Java's read"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_MULTIFILE_SCAN_DIR="${TMP}" \
    cargo test -p iceberg --test interop_scan_exec test_multifile_scan_exec_matches_java_read -- --nocapture
)
echo "==> DONE — multi-file-per-partition scan interop passed (Rust scan == Java read, live rows {10,30,40,50,60}, file1 id 20 deleted, file2 sibling id 50 spared)."
