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
# FILE-SCOPED POSITION-DELETE routing interop (WG4b), DIRECTION 1 ("Rust reads what JAVA writes").
#
# Java `DeleteFileIndex` routes a position delete with a derivable referenced data file into a
# PATH-keyed map (`posDeletesByPath`) that `findPathDeletes` consults with the data file's LOCATION
# alone — no spec condition, no partition condition. Two independent legs derive that location
# (`ContentFileUtil.referencedDataFile`, 1.10.0): the explicit `referenced_data_file` field, and
# EQUAL `file_path`-column lower/upper bounds — the latter being what Java's own
# `PositionDeleteWriter` emits, since it never sets the field.
#
# The Java oracle writes a V2 table created UNPARTITIONED (spec 0) and evolved to
# identity(category) (spec 1), holding data file A (category=a, ids 10/20/30) and data file B
# (category=b, ids 40/50/60), plus THREE position deletes:
#   * FIELD leg   — referenced_data_file = A, no bounds, stamped spec 0 / EMPTY partition → id 20
#   * BOUNDS leg  — no field, equal file_path bounds naming B, stamped spec 0 / EMPTY partition → id 50
#   * CONTROL     — neither leg, stamped spec 1 / partition category=c (EMPTY), names A's row → must NOT apply
# Java's own merge-on-read read is {10,30,40,60}, and the generator FAILS if it is not — the ground
# truth is checked, never assumed. The Rust scan must produce exactly those rows.
#
# LEG 2 — the R117 CROSS-TASK variant (the per-task DeleteFilter scope pin): the SAME fixture with
# the control stamped `category=b` instead of the empty `category=c`, so it attaches to file B's
# TASK while its rows name file A's position 2 (id 30). Java builds one DeleteFilter per task over
# task.deletes() only, so its read is STILL {10,30,40,60} (generator-asserted); a reader whose
# per-scan delete state leaks one task's parsed deletes into another task's file wrongly deletes
# id 30. Proven RED against the pre-fix reader (2026-07-26), green with per-task scoping.
#
# This is a TEST-ONLY ORACLE (a dev tool) — NOT part of the shipped Rust library, NOT part of the
# offline `cargo test` gate (it needs Java + Maven). Nothing binary is committed; the temp table under
# dev/java-interop/target/ is gitignored. Without ICEBERG_INTEROP_FILE_SCOPED_DELETES_DIR the Rust
# test is a clean no-op; this script flips it into the REAL comparison.
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64, a Rust
# toolchain. The Maven deps are the SAME the scan-exec harness already pulled — no new pom deps.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-file-scoped"
TMP_CROSSTASK="${SCRIPT_DIR}/target/interop-file-scoped-crosstask"
echo "==> [1/5] Reset the temp table dirs: ${TMP} + ${TMP_CROSSTASK}"
rm -rf "${TMP}" "${TMP_CROSSTASK}"
mkdir -p "${TMP}" "${TMP_CROSSTASK}"
echo "==> [2/5] Java oracle: write a V2 table with FILE-SCOPED position deletes (field leg + bounds leg) stamped with a foreign spec/partition, plus a partition-scoped control + emit java_file_scoped_delete_rows.json"
(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
    /opt/maven/bin/mvn -o -q compile exec:java \
    -Dexec.args=generate-interop-file-scoped-deletes \
    -Dinterop.file_scoped_deletes.dir="${TMP}"
)
echo "==> [3/5] Java oracle: write the R117 CROSS-TASK variant (control stamped category=b — attached to file B's task, rows naming file A) + emit java_file_scoped_delete_rows.json"
(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
    /opt/maven/bin/mvn -o -q compile exec:java \
    -Dexec.args=generate-interop-file-scoped-deletes-crosstask \
    -Dinterop.file_scoped_deletes_crosstask.dir="${TMP_CROSSTASK}"
)
echo "==> [4/5] Rust: load final.metadata.json, scan → Arrow (path-keyed merge-on-read), compare vs Java's read"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_FILE_SCOPED_DELETES_DIR="${TMP}" \
    cargo test -p iceberg --test interop_scan_exec test_file_scoped_delete_scan_matches_java_read -- --nocapture
)
echo "==> [5/5] Rust: cross-task variant — the control belongs to file B's task; its rows for file A must NOT leak across tasks (id 30 survives, per-task DeleteFilter scope)"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_FILE_SCOPED_DELETES_CROSSTASK_DIR="${TMP_CROSSTASK}" \
    cargo test -p iceberg --test interop_scan_exec test_file_scoped_delete_crosstask_control_does_not_leak -- --nocapture
)
echo "==> DONE — file-scoped position-delete interop passed (leg 1: Rust scan == Java read, live rows {10,30,40,60}: field-leg id 20 and bounds-leg id 50 deleted across the spec/partition mismatch, partition-scoped control id 30 spared; leg 2 cross-task: the category=b control stayed scoped to file B's task and id 30 survived)."
