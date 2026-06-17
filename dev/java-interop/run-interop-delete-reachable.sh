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
# DELETE-REACHABLE-FILES interop harness (post-charter unit #3) — proving the Rust
# DeleteReachableFiles action (the engine behind DROP TABLE PURGE) agrees with Java 1.10.0's
# engine-agnostic iceberg-CORE reachable-set logic (ReachableFileUtil + a Snapshot.allManifests
# content scan) on a multi-snapshot table carrying EVERY reachable file category: data, position
# deletes, equality deletes, a Puffin DV, a statistics file, a previous metadata.json, the current
# metadata.json, manifests, manifest lists, and the version-hint.
#
# THE CHAIN (mirrors run-interop-expire.sh's structure + the fail-closed sentinel discipline):
#
#   1. RESET the temp dir.
#   2. JAVA builds the fixture table at <dir>/table, computes its OWN reachable set via the REAL core
#      ReachableFileUtil + allManifests scan, and emits java_reachable.json (a path-INDEPENDENT
#      CATEGORY-COUNT descriptor) + final.metadata.json. The oracle is NON-CIRCULAR (Java's real
#      reachable logic, never anything Rust wrote).
#   3. RUST (D2 + GEN) reads Java's table, computes ITS OWN reachable descriptor, asserts it equals
#      java_reachable.json, runs DeleteReachableFiles end-to-end (collecting deleter ⇒ planned-set
#      descriptor) on the metadata location + physically purges a THROWAWAY copy's resident files,
#      and emits rust_reachable.json + rust_deleted.json.
#   4. JAVA (D1) re-judges: its own ReachableFileUtil recomputation == rust_reachable.json AND
#      rust_deleted.json (delete-completeness: every category fully purged, no over-delete).
#   5. SABOTAGE: drop a DATA file from the Rust descriptor (under-count) ⇒ Java's verify MUST fail.
#      A sabotage that cannot be applied HARD-FAILS (never SKIP); a sabotage that PASSES verify is a
#      false-green and HARD-FAILS too.
#
# A divergence anywhere — category miscount, an omitted reachable category (orphan leak), or an
# over-deleted file (data loss) — fails loudly. TEST-ONLY oracle; nothing here is in the offline
# `cargo test` gate; the temp dir under dev/java-interop/target/ is gitignored.
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64, the
# repo's pinned Rust toolchain. Run from anywhere; paths resolve relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-delete-reachable"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

echo "==> [1/5] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/5] Java: build the fixture + compute reachable (ReachableFileUtil) + emit java_reachable.json + final.metadata.json"
run_oracle -Dexec.args=generate-interop-delete-reachable -Dinterop.delete_reachable.dir="${TMP}"

echo "==> [3/5] Rust: compute its OWN reachable descriptor (D2), run DeleteReachableFiles, emit rust_reachable.json + rust_deleted.json"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_DELETE_REACHABLE_DIR="${TMP}" \
    cargo test -p iceberg --test interop_delete_reachable delete_reachable_gen_and_completeness \
    -- --exact --nocapture
)

echo "==> [4/5] Java (D1): re-judge — its ReachableFileUtil recomputation == rust_reachable.json AND rust_deleted.json (delete-completeness)"
# The verdict comes from the OUTPUT sentinel ("verify-interop-delete-reachable: 0 failures" with no
# FAIL line), never from mvn's exit code (machine-dependent for `exec:java` System.exit); `|| true`
# keeps set -e from aborting before the diagnostics are echoed (the run-interop-dv.sh fail-closed rule).
VERIFY_OUT="$(
  cd "${SCRIPT_DIR}"
  "${MVN}" -o -q compile exec:java \
    -Dexec.args=verify-interop-delete-reachable \
    -Dinterop.delete_reachable.dir="${TMP}" 2>&1
)" || true
echo "${VERIFY_OUT}"
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-delete-reachable: 0 failures'; then
  echo "==> FAILED — Java rejected the Rust reachable/deleted descriptors (a category miscount, orphan leak, or over-delete)."
  exit 1
fi

echo "==> [5/5] SABOTAGE: drop a DATA file from the Rust descriptor (under-count) — Java's verify MUST fail closed"
# Sabotage runs against a SCRATCH copy of the temp dir (the clean ${TMP} is untouched). HARD-FAIL,
# never SKIP, if the corruption cannot be applied — a sabotage that did not actually corrupt anything
# proves nothing.
SCRATCH="${TMP}/sabotage_scratch"
rm -rf "${SCRATCH}"
mkdir -p "${SCRATCH}"
cp -r "${TMP}/table" "${SCRATCH}/table"
cp "${TMP}/java_reachable.json" "${SCRATCH}/java_reachable.json"
cp "${TMP}/rust_deleted.json" "${SCRATCH}/rust_deleted.json"

# The clean Rust descriptor (a bare "data=2;pos_delete=1;...") — drop one data file: data=2 -> data=1.
CLEAN_DESC="$(cat "${TMP}/rust_reachable.json")"
if ! echo "${CLEAN_DESC}" | grep -q 'data=2;'; then
  echo "==> FAILED — cannot apply under-count sabotage: 'data=2;' not present in the Rust descriptor"
  echo "    (descriptor was: ${CLEAN_DESC}) — the fixture's data-file count changed; pin the sabotage to it."
  rm -rf "${SCRATCH}"
  exit 1
fi
SABOTAGED_DESC="${CLEAN_DESC/data=2;/data=1;}"
printf '%s' "${SABOTAGED_DESC}" > "${SCRATCH}/rust_reachable.json"
echo "    sabotaged Rust descriptor: ${SABOTAGED_DESC}"

# Java's verify MUST fail on the under-counted descriptor (the data category now claims one fewer
# file than Java's reachable set ⇒ an orphan leak Java must catch). Capture the output; the verdict is
# the sentinel, never the exit code.
SAB_OUT="$(
  cd "${SCRIPT_DIR}"
  "${MVN}" -o -q compile exec:java \
    -Dexec.args=verify-interop-delete-reachable \
    -Dinterop.delete_reachable.dir="${SCRATCH}" 2>&1
)" || true
echo "${SAB_OUT}"
if echo "${SAB_OUT}" | grep -q 'verify-interop-delete-reachable: 0 failures' \
  && ! echo "${SAB_OUT}" | grep -q '^FAIL '; then
  echo "FAIL sabotage(under-count): Java's verify PASSED on an under-counted reachable descriptor — the comparison is vacuous"
  rm -rf "${SCRATCH}"
  exit 1
fi
if ! echo "${SAB_OUT}" | grep -q 'reachable descriptors differ'; then
  echo "FAIL sabotage(under-count): Java's verify failed but NOT via the 'reachable descriptors differ' path —"
  echo "    the under-count was caught by the wrong check (the sabotage is wrongly pinned)."
  rm -rf "${SCRATCH}"
  exit 1
fi
echo "PASS sabotage(under-count): an under-counted reachable descriptor fails closed via the 'reachable descriptors differ' path"
rm -rf "${SCRATCH}"

echo "==> DONE — DeleteReachableFiles interop passed:"
echo "    D2 (Rust computes, asserts == Java): the Rust reachable category-count descriptor == Java's ReachableFileUtil recomputation."
echo "    D1 (Java judges Rust): Java's recomputation == rust_reachable.json AND rust_deleted.json (delete-completeness — every category fully purged, no over-delete)."
echo "    Sabotage: an under-counted descriptor fails closed (the reachable-set comparison is non-vacuous)."
