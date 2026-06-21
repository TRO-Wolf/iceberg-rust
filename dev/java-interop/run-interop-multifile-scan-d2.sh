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
# MULTI-FILE-PER-PARTITION data-level scan-execution interop, DIRECTION 2 ("Java reads what RUST writes").
# The reverse of run-interop-multifile-scan.sh: Rust's GEN test writes a REAL on-disk table to
# <dir>/rust_table via its production write path (MemoryCatalog over LocalFsStorageFactory: TWO real
# parquet data files in ONE partition (category=a) fast_appended at sequence 1, plus a partition-scoped
# position-delete on file1 written by PositionDeleteFileWriter with the partition_key set, row_delta'd at
# sequence 2), landing a final.metadata.json. Java then loads that RUST-written metadata, reads via
# IcebergGenerics (which APPLIES Rust's path-keyed position delete), and asserts the merge-on-read rows ==
# {10,30,40,50,60} (file1 id 20 deleted, file2 sibling id 50 SPARED). A failure here is a REAL
# write-incompatibility finding.
#
# TEST-ONLY ORACLE (dev tool); not in the offline gate; nothing binary committed; temp dir gitignored.
# Without ICEBERG_INTEROP_MULTIFILE_SCAN_GEN_DIR the Rust GEN test is a clean no-op.
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64, a Rust toolchain.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP2="${SCRIPT_DIR}/target/interop-multifile-d2"
echo "==> [1/3] Reset the temp table dir: ${TMP2}"
rm -rf "${TMP2}"
mkdir -p "${TMP2}"
echo "==> [2/3] Rust: WRITE a REAL multi-file-per-partition V2 table (two data files in category=a seq 1 + a partition-scoped position-delete on file1 seq 2) + final.metadata.json"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_MULTIFILE_SCAN_GEN_DIR="${TMP2}" \
    cargo test -p iceberg --test interop_scan_exec test_multifile_scan_exec_gen_rust_writes_java_readable_table -- --nocapture
)
echo "==> [3/3] Java: load the RUST-written final.metadata.json, read via IcebergGenerics, verify {10,30,40,50,60}"
VERIFY_OUT="$(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
    /opt/maven/bin/mvn -o -q compile exec:java \
    -Dexec.args=verify-interop-multifile-scan \
    -Dinterop.multifile_scan.dir="${TMP2}" 2>&1
)"
echo "${VERIFY_OUT}"
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' || ! echo "${VERIFY_OUT}" | grep -q ': 0 failures'; then
  echo "==> FAILED — Java could not correctly read the Rust-written multi-file table (a real write-incompatibility finding)."
  exit 1
fi
echo "==> DONE — Direction-2 multi-file round-trip passed (Java read the Rust-written multi-file table, live rows {10,30,40,50,60}, file1 id 20 deleted, file2 sibling id 50 spared)."
