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
# REWRITE-MANIFESTS DATA-LEVEL interop harness (GAP_MATRIX row 100).
#
# Proves that after Rust runs rewrite_manifests on a table Java wrote (3 real-parquet data files,
# 3 separate manifests), the LIVE DATA ROWS are preserved and the manifests are actually
# re-clustered (3 → 1), AND that Java can read the Rust-rewritten table and see the identical rows.
#
# THE CHAIN:
#   1. Reset the temp dir.
#   2. JAVA generates the fixture: unpartitioned V2 table under <tmp>/table with THREE real
#      parquet data files (ids 10/20, 30/40, 50/60) each committed in its own newAppend (→ 3
#      data manifests). Emits java_rows.json (IcebergGenerics live-row read) + final.metadata.json.
#   3. RUST GEN (cargo test, env ICEBERG_INTEROP_RWM_DIR): builds a MemoryCatalog table at the SAME
#      location referencing Java's REAL parquet data files, and mirrors Java's 3-commit chain with 3
#      fast_appends (so the catalog-owned table starts at 3 data manifests; scans still decode Java's
#      on-disk parquet bytes). Runs rewrite_manifests().cluster_by(...) (3→1 manifest), asserts rows ==
#      java_rows.json AND manifest count == 1, writes the rewritten metadata to
#      <tmp>/rust_rewritten/metadata/final.metadata.json.
#   4. JAVA verify: reads <tmp>/rust_rewritten/metadata/final.metadata.json via IcebergGenerics,
#      asserts live rows == java_rows.json AND data-manifest count < 3. Prints the sentinel
#      "verify-interop-rewrite-manifests: 0 failures" on success.
#
# FAIL-CLOSED: the verify outcome is read from the printed sentinel (not from mvn exit, which is
# unreliable for exec:java + System.exit). The script greps both "0 failures" AND the absence
# of any FAIL line, mirroring run-interop-expire.sh discipline.
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64,
# the repo's pinned Rust toolchain.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-rewrite-manifests"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

echo "==> [1/4] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/4] Java: generate the fixture (3 data files × 3 separate newAppend commits → 3 manifests + java_rows.json)"
run_oracle -Dexec.args=generate-interop-rewrite-manifests \
  -Dinterop.rewrite_manifests.dir="${TMP}"

echo "==> [3/4] Rust: build a table over Java's real parquet files (mirror 3 appends), run rewrite_manifests (3→1 manifest), assert rows preserved, write rust_rewritten/metadata/final.metadata.json"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_RWM_DIR="${TMP}" \
    cargo test -p iceberg --test interop_rewrite_manifests test_rewrite_manifests_interop_gen \
    -- --exact --nocapture
)

echo "==> [4/4] Java: verify the Rust-rewritten table (rows == java_rows.json AND manifest count < 3)"
RWM_VERIFY_OUT="$(
  cd "${SCRIPT_DIR}"
  "${MVN}" -o -q compile exec:java \
    -Dexec.args=verify-interop-rewrite-manifests \
    -Dinterop.rewrite_manifests.dir="${TMP}" 2>&1
)" || true
echo "${RWM_VERIFY_OUT}"
if echo "${RWM_VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${RWM_VERIFY_OUT}" | grep -q 'verify-interop-rewrite-manifests: 0 failures'; then
  echo "==> FAILED — Java rejected the Rust-rewritten table (rows diverged or manifest count not reduced)."
  exit 1
fi

echo "==> DONE — RewriteManifests data-level interop passed:"
echo "    Rust re-clustered 3 data manifests → 1 (proven by manifest-count assert in Rust GEN step)."
echo "    Java read the Rust-rewritten table via IcebergGenerics and confirmed all 6 rows present,"
echo "    exactly equal to java_rows.json, AND data-manifest count < 3."
