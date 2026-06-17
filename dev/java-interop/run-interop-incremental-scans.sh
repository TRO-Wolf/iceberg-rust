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
# INCREMENTAL SCANS interop harness — GAP_MATRIX rows 120 (IncrementalAppendScan) + 121
# (IncrementalChangelogScan). Proves the already-built Rust incremental scans plan the SAME data-file sets as
# Java's REAL IncrementalAppendScan / IncrementalDataTableScan (the data-file changelog) over the SAME
# snapshot ranges, in BOTH directions, with the expected sets HAND-DECLARED IDENTICALLY on both sides
# (anti-circular).
#
# THE FIXTURE (built identically by Java's generate + Rust's GEN): a 4-snapshot UNPARTITIONED V2 table
# {1 id long required, 2 data string optional} with DETERMINISTIC data-file basenames a/b/c/d (the
# cross-engine comparison key — both engines write at different roots with different snapshot ids):
#   S1 append a (id=10), S2 append b (id=20), S3 append c (id=30), S4 OVERWRITE delete-a + add-d (id=40).
#
# THE HAND-DECLARED EXPECTED SETS (declared in both IncrementalScanOracle.java and interop_incremental_scans.rs):
#   APPEND:    append_excl   fromExclusive(S1)→S3   ⇒ {b, c}      (S1's own a EXCLUDED — exclusive boundary)
#              append_incl   fromInclusive(S1)→S3   ⇒ {a, b, c}   (S1's own a INCLUDED — inclusive boundary)
#              append_to_cur fromExclusive(S2)[→S4] ⇒ {c}         (S3's c; the S4 overwrite EXCLUDED)
#   CHANGELOG: changelog     fromExclusive(S1)→S4   ⇒ +b +c -a +d (the data-file added/deleted changelog)
#
# THE CHAIN (E1-family, both directions + sabotage):
#   [1/6] Reset the temp dir.
#   [2/6] Java GENERATE (Direction 1): write the 4-snapshot table; run the REAL Java IncrementalAppendScan +
#         IncrementalChangelogScan; emit java_append_sets.json + java_changelog_entries.json (basename sets).
#   [3/6] Rust GEN (Direction 2): write the SAME chain to <dir>/rust_table via the production write path +
#         self-check Rust's own incremental scans match the ground truth.
#   [4/6] Java VERIFY (Direction 2): load the RUST-written table, run the REAL Java incremental scans, assert
#         the planned BASENAME sets == the hand-declared ground truth. A FAIL is a real write-incompat finding.
#   [5/6] Rust D1: load the JAVA-written table, run Rust's incremental scans, assert == Java's planned sets
#         AND == the hand-declared ground truth.
#   [6/6] SABOTAGE (fail-closed): shift the exclusive lower bound by ONE snapshot (S1→S2) over the Rust table
#         and assert the planned set DIVERGES from {b,c} — proving the inclusive/exclusive boundary is
#         load-bearing (the off-by-one corruption edge). HARD-FAILS if the shift leaves the set unchanged.
#
# This is a TEST-ONLY ORACLE (a dev tool, like dev/spark/) — NOT part of the shipped Rust library, NOT part
# of the offline `cargo test` gate (it needs Java + Maven). Nothing binary is committed; the temp dir under
# dev/java-interop/target/ is gitignored.
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64, the repo's Rust
# toolchain. The first Maven run must be ONLINE to populate ~/.m2 (iceberg-core/api/data/parquet 1.10.0);
# after that `mvn -o` runs fully offline. Run from anywhere; paths resolve relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-incremental-scans"
D1_DIR="${TMP}/d1"
GEN_DIR="${TMP}/d2"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

echo "==> [1/6] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${D1_DIR}" "${GEN_DIR}"

echo "==> [2/6] Java GENERATE (Direction 1): write the 4-snapshot table + emit the Java planned sets"
run_oracle -Dexec.args=generate-interop-incremental-scans \
  -Dinterop.incremental_scans.dir="${D1_DIR}"
test -f "${D1_DIR}/java_append_sets.json" || { echo "FAIL: java_append_sets.json not produced"; exit 1; }
test -f "${D1_DIR}/java_changelog_entries.json" \
  || { echo "FAIL: java_changelog_entries.json not produced"; exit 1; }
echo "    java_append_sets.json + java_changelog_entries.json produced OK"

echo "==> [3/6] Rust GEN (Direction 2): write the SAME chain to ${GEN_DIR}/rust_table + self-check"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_INCREMENTAL_SCANS_GEN_DIR="${GEN_DIR}" \
    cargo test -p iceberg --test interop_incremental_scans -- --nocapture
)
test -f "${GEN_DIR}/rust_table/metadata/final.metadata.json" \
  || { echo "FAIL: Rust GEN did not produce rust_table/metadata/final.metadata.json"; exit 1; }

echo "==> [4/6] Java VERIFY (Direction 2): Java's REAL incremental scans over the RUST-written table"
# `mvn -q exec:java` runs the oracle in Maven's own JVM and does NOT propagate System.exit(1) to the shell.
# Capture the output + assert the success sentinel ("...: 0 failures") with no FAIL line.
VERIFY_OUT="$(
  run_oracle -Dexec.args=verify-interop-incremental-scans \
    -Dinterop.incremental_scans.dir="${GEN_DIR}"
)"
echo "${VERIFY_OUT}"
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
    || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-incremental-scans: 0 failures'; then
  echo "==> FAILED — Java's incremental scans over the Rust-written table diverged (a real finding)."
  exit 1
fi
echo "    Direction-2 OK — Java's incremental scans over the Rust table match the ground truth"

echo "==> [5/6] Rust D1: Rust's incremental scans over the JAVA-written table == Java's sets == ground truth"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_INCREMENTAL_SCANS_DIR="${D1_DIR}" \
    cargo test -p iceberg --test interop_incremental_scans -- --nocapture
)
echo "    Direction-1 OK — Rust's incremental scans over the Java table match Java + the ground truth"

echo ""
echo "==> [6/6] SABOTAGE (fail-closed): shift the exclusive lower bound by one snapshot (S1→S2)"
echo "     → the planned set must DIVERGE from {b,c} (the inclusive/exclusive boundary is load-bearing)."
SABOTAGE_OUT="$(
  run_oracle -Dexec.args=sabotage-interop-incremental-scans \
    -Dinterop.incremental_scans.dir="${GEN_DIR}"
)"
echo "${SABOTAGE_OUT}"
# The sabotage PASSES (fail-closed confirmed) only on the PASS line; a "VACUOUS" / FAIL line means the
# off-by-one was not load-bearing — abort non-zero.
if echo "${SABOTAGE_OUT}" | grep -q '^FAIL ' \
    || echo "${SABOTAGE_OUT}" | grep -q 'VACUOUS' \
    || ! echo "${SABOTAGE_OUT}" | grep -q 'PASS incremental-scans-sabotage'; then
  echo "==> SABOTAGE FAILED — the shifted bound did NOT change the set; the boundary is not load-bearing."
  exit 1
fi
echo "    SABOTAGE PASS: the off-by-one snapshot bound changed the planned file set (fail-closed)"

echo ""
echo "==> DONE — incremental-scans interop passed (rows 120/121): IncrementalAppendScan (3 ranges, the"
echo "     inclusive/exclusive boundary pinned) + IncrementalChangelogScan (data-file +b +c -a +d), BOTH"
echo "     directions, anti-circular hand-declared sets, + the off-by-one boundary sabotage closed."
