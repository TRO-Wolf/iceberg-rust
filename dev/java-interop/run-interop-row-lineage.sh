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
# V3 ROW LINEAGE interop, BOTH DIRECTIONS (GAP_MATRIX row R166).
#
#   D1 ("Rust reads what JAVA writes"): the oracle writes a V3 table over FOUR snapshots — files
#   a(5 rows) + b(3) in one commit, then c(4), then a RewriteFiles replacing a with d(2), then an
#   OverwriteFiles replacing c with e(2). The first two commits make the row-id counter advance BOTH
#   within a manifest and across snapshots; the last two make file b survive as an EXISTING entry
#   across two rewrites, which pins that a survivor keeps its first_row_id and its per-row _row_id.
#   Java emits the lineage view its own ManifestReader.idAssigner resolved; Rust builds the same
#   view through ManifestFile::load_manifest and diffs.
#
#   The UPGRADED fixture is a second table in the same run: V2, TWO appends, an upgrade to V3, then a
#   rewrite as the FIRST V3 commit. It is the only shape either implementation can build in which a
#   live EXISTING entry reaches the reader with NO stored first_row_id — the rewrite reads the V2
#   manifest, whose range is absent, so every entry is nulled before the survivor is written
#   forward. It takes TWO appends, not one, so two carried-forward data manifests reach the V3
#   commit still needing a range and each still holding live rows; their relative order then decides
#   which takes which. The four record counts are distinct (f=3, g=2, i=4, h=1) so a swap survives
#   the name stripping the assignment cross-check does. Both directions plus a cross-check.
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
#   * skip guard `== Added` instead of `!= Deleted` -> RED on the UPGRADED legs (4 red out of 10):
#     the surviving file reads back with a null first_row_id and null _row_id for every row.
#     MEASURED LIMIT (2026-08-25): the four-snapshot fixture above has EXISTING entries and still
#     leaves this mutation GREEN, because a rewrite reads its source through the assigning reader,
#     so every survivor it writes forward already carries a stored id and the `is_some()` clause
#     short-circuits the status test. Only the V2-to-V3 upgrade reaches the branch.
#   * manifest-list order, conjunct (a): added data manifests AFTER the existing ones -> 2 red out
#     of 10 (file d takes 15 where Java gives it 12), the divergence this suite found 2026-08-25.
#   * manifest-list order, conjunct (b): the existing data manifests reversed -> 1 red out of 10,
#     on the UPGRADED cross-check (files g and i swap their ranges). Also 1 red out of 3477 lib
#     tests, in snapshot::manifest_list_order_tests.
#   * manifest-list order, conjunct (c): the data group must precede the delete group. NO interop
#     leg can see it — neither fixture carries a delete manifest — so it is pinned in lib only, by
#     snapshot::manifest_list_order_tests (1 red out of 3477).

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

echo "==> [2/5] (D1) Java oracle: write both V3 fixtures + emit their lineage and row-id views"
(
  cd "${SCRIPT_DIR}"
  JAVA_HOME="${JAVA_HOME_DIR}" PATH="${JAVA_HOME_DIR}/bin:$PATH" \
    "${MVN}" -o -q compile exec:java \
    -Dexec.args=generate-interop-row-lineage \
    -Dinterop.row_lineage.dir="${D1}"
)

echo "==> [3/5] (D1) Rust: lineage view + per-row _row_id of the JAVA tables, both diffed"
(
  cd "${REPO_ROOT}"
  for test_name in rust_reads_java_assigned_row_lineage rust_materializes_java_row_ids \
    rust_reads_java_upgraded_row_lineage rust_materializes_java_upgraded_row_ids; do
    ICEBERG_INTEROP_ROW_LINEAGE_DIR="${D1}" \
      cargo test -p iceberg --test interop_row_lineage \
      "${test_name}" -- --exact --nocapture
  done
)

echo "==> [4/5] (D2) Rust: commit the equivalent V3 tables + emit the two expectation files"
(
  cd "${REPO_ROOT}"
  for test_name in row_lineage_write_gen row_lineage_upgraded_write_gen; do
    ICEBERG_INTEROP_ROW_LINEAGE_WRITE_DIR="${D2}" \
      cargo test -p iceberg --test interop_row_lineage \
      "${test_name}" -- --exact --nocapture
  done
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
  for test_name in rust_assigns_the_same_row_ids_java_does \
    rust_assigns_the_same_upgraded_row_ids_java_does; do
    ICEBERG_INTEROP_ROW_LINEAGE_DIR="${D1}" ICEBERG_INTEROP_ROW_LINEAGE_WRITE_DIR="${D2}" \
      cargo test -p iceberg --test interop_row_lineage \
      "${test_name}" -- --exact --nocapture
  done
  for test_name in java_materializes_rust_row_ids java_materializes_rust_upgraded_row_ids; do
    ICEBERG_INTEROP_ROW_LINEAGE_WRITE_DIR="${D2}" \
      cargo test -p iceberg --test interop_row_lineage \
      "${test_name}" -- --exact --nocapture
  done
)

echo "==> OK — V3 row lineage round-trips in BOTH directions and the fork assigns Java's ids."
