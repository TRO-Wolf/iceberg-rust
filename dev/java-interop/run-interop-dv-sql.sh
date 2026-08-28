#!/usr/bin/env bash
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
# F-13 U4 interop: JAVA reads what RUST's SQL DELETE writes.
#
# The V3 merge-on-read sibling of run-interop-partitioned-dml.sh (copy-on-write) and the
# SQL-driven sibling of run-interop-dv.sh's D4 leg (hand-driven DVFileWriter chain). This is the
# only leg that exercises the whole U4 path end to end: DataFusion SQL DELETE -> version dispatch
# -> DVFileWriter -> row_delta with remove_deletes_many -> Java's production scan.
#
# Steps:
#   1. Rust GEN: three SQL DELETE statements on a V3 identity(category)-partitioned table. The third
#      re-deletes from a data file that already carries a DV, so the writer must load, merge and
#      supersede it.
#   2. Java verify: parse the Rust-written final.metadata.json, confirm format-version 3, read the
#      table with IcebergGenerics (which applies each DV via BaseDeleteLoader.readDV), and assert
#      the live rows plus the V3 shape rules (no position-delete files, one DV per data file).
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11. The first run needs the network to fetch the
# 1.10.0 jars; after that `mvn -o` works fully offline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-dv-sql"

echo "==> [1/7] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/7] Rust: sequential DELETE + shared-Puffin DELETE/UPDATE GEN"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_DV_SQL_GEN_DIR="${TMP}" \
    cargo test -p iceberg-datafusion --test interop_dv_sql -- --nocapture
)

# The verdict comes from the OUTPUT (success sentinel present, no per-check FAIL line), never from
# mvn's exit code -- `mvn exec:java` does not propagate System.exit. `|| true` keeps `set -e` from
# aborting before the diagnostics are echoed.
verify_dv_sql() {
  local dir="$1"
  local label="$2"
  local out
  out="$(
    cd "${SCRIPT_DIR}"
    JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
      PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
      /opt/maven/bin/mvn -o -q compile exec:java \
      -Dexec.args=verify-interop-dv-sql \
      -Dinterop.dv_sql.dir="${dir}" 2>&1
  )" || true
  echo "${out}"
  if echo "${out}" | grep -q '^FAIL ' || ! echo "${out}" | grep -q 'verify-interop-dv-sql: 0 failures'; then
    echo "==> FAILED -- Java could not correctly read ${label}."
    exit 1
  fi
}

echo "==> [3/7] Java: read the sequential SQL DELETE table"
verify_dv_sql "${TMP}" "the sequential SQL DELETE table"

echo "==> [4/7] Java: read the shared-Puffin SQL DELETE table"
verify_dv_sql "${TMP}/shared_puffin" "the shared-Puffin SQL DELETE table"

echo "==> [5/7] Java: read the shared-Puffin SQL UPDATE table"
verify_dv_sql "${TMP}/shared_puffin_update" "the shared-Puffin SQL UPDATE table"

echo "==> [6/7] Java: write a shared-Puffin table via BaseDVFileWriter"
(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
    /opt/maven/bin/mvn -o -q compile exec:java \
    -Dexec.args=generate-interop-dv-table \
    -Dinterop.dv_table.dir="${TMP}/java_shared"
)

echo "==> [7/7] Rust DELETE against the Java-written shared Puffin; Java reads the result"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_DV_SQL_JAVA_SHARED="${TMP}/java_shared" \
    cargo test -p iceberg-datafusion --test interop_dv_sql \
    test_dv_sql_consume_java_written_shared_puffin -- --exact --nocapture
)
verify_dv_sql "${TMP}/java_shared/after_delete" "Rust DELETE of the Java-written shared Puffin"

echo "==> interop-dv-sql PASSED"
