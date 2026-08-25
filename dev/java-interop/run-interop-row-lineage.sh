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
#
# V3 ROW LINEAGE interop, BOTH DIRECTIONS (GAP_MATRIX row R166).
#
#   D1 ("Rust reads what JAVA writes"): the oracle writes a V3 table over TWO snapshots — files
#   a(5 rows) + b(3) in one commit, then c(4) — so the row-id counter has to advance BOTH within a
#   manifest and across snapshots. Java emits the lineage view its own ManifestReader.idAssigner
#   resolved; Rust builds the same view through ManifestFile::load_manifest and diffs.
#
#   D2 ("JAVA reads what RUST writes"): the Rust GEN test commits the equivalent V3 table through
#   the production path; Java reads it with the SAME view builder, diffs, and separately asserts no
#   data file comes back with a null first_row_id. This is the leg R166 named as its gating residue.
#
#   MATERIALIZATION is the second half of the row, and the manifest view cannot reach it:
#   inheritance is metadata, but `_row_id` is resolved PER ROW by the reader (stored value, else
#   first_row_id + pos). Both sides scan with `_row_id` + `_last_updated_sequence_number`
#   projected and diff per row, in both directions.
#
#   The CROSS-CHECK closes D2's circularity: D2 renders one table twice, so a wrong-but-consistent
#   writer would pass it. `rust_assigns_the_same_row_ids_java_does` compares the two INDEPENDENTLY
#   produced views with file names stripped, so only the lineage numbers remain.
#
# Mutation evidence (2026-08-25), watched RED at base:
#   * drop the read-side inheritance call        -> D1 RED (every file reads back null)
#   * read-side counter advances by 0            -> D1 RED (file b gets 0, not 5)
#   * write-side range advances by 0             -> cross-check RED (c gets 0, next_row_id 0)
#   * every row in a file shares one _row_id     -> materialization RED (was GREEN before the
#     materialization tests existed: the manifest view alone cannot see a per-row defect)
#   * skip guard `== Added` instead of `!= Deleted` -> GREEN. NAMED LIMIT: this fixture has only
#     `Added` entries, so the guard stays pinned by the unit domain table in
#     spec::manifest::entry::first_row_id_tests, not by this leg. The write-side
#     `existing_rows_count` arm is likewise append-only here and is pinned by the unit test
#     `assigned_range_advances_by_existing_rows_not_only_added`.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
D1="${SCRIPT_DIR}/target/interop-row-lineage"
D2="${SCRIPT_DIR}/target/interop-row-lineage-write"
JAVA_HOME_DIR="${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk-amd64}"
MVN="${MVN:-/opt/maven/bin/mvn}"

echo "==> [1/5] Reset the temp dirs: ${D1} + ${D2}"
rm -rf "${D1}" "${D2}"
mkdir -p "${D1}" "${D2}"

echo "==> [2/5] (D1) Java oracle: write a V3 table over two snapshots + emit java_row_lineage.json"
(
  cd "${SCRIPT_DIR}"
  JAVA_HOME="${JAVA_HOME_DIR}" PATH="${JAVA_HOME_DIR}/bin:$PATH" \
    "${MVN}" -o -q compile exec:java \
    -Dexec.args=generate-interop-row-lineage \
    -Dinterop.row_lineage.dir="${D1}"
)

echo "==> [3/5] (D1) Rust: lineage view + per-row _row_id of the JAVA table, both diffed"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_ROW_LINEAGE_DIR="${D1}" \
    cargo test -p iceberg --test interop_row_lineage \
    rust_reads_java_assigned_row_lineage -- --exact --nocapture
  ICEBERG_INTEROP_ROW_LINEAGE_DIR="${D1}" \
    cargo test -p iceberg --test interop_row_lineage \
    rust_materializes_java_row_ids -- --exact --nocapture
)

echo "==> [4/5] (D2) Rust: commit the equivalent V3 table + emit rust_row_lineage_expected.json"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_ROW_LINEAGE_WRITE_DIR="${D2}" \
    cargo test -p iceberg --test interop_row_lineage \
    row_lineage_write_gen -- --exact --nocapture
)

echo "==> [5/5] (D2) Java: read the RUST-written V3 table + the assignment cross-check"
VERIFY_OUT="$(
  cd "${SCRIPT_DIR}"
  JAVA_HOME="${JAVA_HOME_DIR}" PATH="${JAVA_HOME_DIR}/bin:$PATH" \
    "${MVN}" -o -q compile exec:java \
    -Dexec.args=verify-interop-row-lineage \
    -Dinterop.row_lineage.dir="${D2}" 2>&1
)" || true
echo "${VERIFY_OUT}"
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${VERIFY_OUT}" | grep -qE 'verify-interop-row-lineage: 0 failures$'; then
  echo "==> FAILED — Java could not read the row lineage the fork assigned (a real finding)."
  exit 1
fi
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_ROW_LINEAGE_DIR="${D1}" ICEBERG_INTEROP_ROW_LINEAGE_WRITE_DIR="${D2}" \
    cargo test -p iceberg --test interop_row_lineage \
    rust_assigns_the_same_row_ids_java_does -- --exact --nocapture
  ICEBERG_INTEROP_ROW_LINEAGE_WRITE_DIR="${D2}" \
    cargo test -p iceberg --test interop_row_lineage \
    java_materializes_rust_row_ids -- --exact --nocapture
)

echo "==> OK — V3 row lineage round-trips in BOTH directions and the fork assigns Java's ids."
