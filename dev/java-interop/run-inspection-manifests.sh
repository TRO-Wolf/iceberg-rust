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
# MANIFEST-READING inspection interop harness — the `files` / `data_files` / `delete_files` tables (A1).
#
# This is a TEST-ONLY ORACLE (a dev tool, like dev/spark/) — it is NOT part of the shipped Rust library, and
# it is NOT part of the offline `cargo test` gate (it needs Java + Maven). Unlike the pure-metadata
# inspection interop (committed JSON, offline), these tables read REAL ON-DISK AVRO manifests, so the oracle
# WRITES A REAL TABLE to a temp dir each run and an ENV-GATED Rust test reads it. Nothing binary is
# committed — the committed artifacts are the oracle code (InteropOracle.java), the Rust test
# (interop_inspection_manifests.rs), and this run script.
#
# Methodology (regenerate-and-compare):
#   1. mvn ... -Dexec.args=generate-inspection-manifests -Dinterop.inspection_manifests.dir="$TMP"
#        -> The Java oracle builds a partitioned V2 table on local disk under "$TMP/table" via REAL commits
#           (newAppend writes a DATA manifest + manifest-list; newRowDelta writes a DELETE manifest), writes
#           "$TMP/table/metadata/final.metadata.json", and materializes the rows of Java's REAL FilesTable /
#           DataFilesTable / DeleteFilesTable into "$TMP/java_{files,data_files,delete_files}.json".
#   2. ICEBERG_INTEROP_MANIFEST_DIR="$TMP" cargo test ... interop_inspection_manifests
#        -> The env-gated Rust test loads "$TMP/table/metadata/final.metadata.json", builds a Table over a
#           local-filesystem FileIO (resolving the absolute manifest paths), runs
#           inspect().files()/.data_files()/.delete_files().scan(), and compares EVERY column (except the
#           deferred readable_metrics) field-for-field, order-independent, against the Java rows.
#
# Without ICEBERG_INTEROP_MANIFEST_DIR the Rust test is a clean no-op (it stays green in the offline gate);
# this script is what flips it into the REAL comparison.
#
# Requirements:
#   - Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64.
#   - A Rust toolchain (the repo's pinned nightly via rust-toolchain.toml).
#   - First Maven run downloads iceberg-core/iceberg-api 1.10.0 from Maven Central.
#
# Run from anywhere; paths are resolved relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-manifests"

echo "==> [1/3] Reset the temp table dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/3] Java oracle: write a REAL partitioned V2 table + emit java_{files,data_files,delete_files}.json"
(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
    /opt/maven/bin/mvn -o -q compile exec:java \
    -Dexec.args=generate-inspection-manifests \
    -Dinterop.inspection_manifests.dir="${TMP}"
)

echo "==> [3/3] Rust: load final.metadata.json, scan files/data_files/delete_files, compare vs Java rows"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_MANIFEST_DIR="${TMP}" \
    cargo test -p iceberg --test interop_inspection_manifests -- --nocapture
)

echo "==> DONE — manifest-reading inspection interop (files / data_files / delete_files) passed."
