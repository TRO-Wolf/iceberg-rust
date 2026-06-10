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
# DELETION-VECTOR scan-execution interop harness (Increment D1, Direction 1 — "Rust reads what
# JAVA writes"). Proves Rust's scan → Arrow with PUFFIN DELETION-VECTOR application matches Java's
# OWN read of a JAVA-WRITTEN **V3** table whose merge-on-read deletes are a real
# `deletion-vector-v1` blob written by Java's production BaseDVFileWriter. This is the sibling of
# run-interop-scan-exec.sh (parquet position deletes); the DV is the V3 replacement for those.
#
# This run ALSO settles the roaring byte-compatibility question EMPIRICALLY: the oracle emits a
# second, synthetic DV blob (dv_blob.bin) whose positions span the 32-bit key boundary and include
# a run-length-encoded range, and the env-gated Rust LIB test decodes those real Java bytes and
# asserts the exact position set.
#
# This is a TEST-ONLY ORACLE (a dev tool, like dev/spark/) — NOT part of the shipped Rust library
# and NOT part of the offline `cargo test` gate (it needs Java + Maven). Nothing binary is
# committed; the temp table under dev/java-interop/target/ is gitignored.
#
# Methodology (regenerate-and-compare):
#   1. mvn ... -Dexec.args=generate-interop-dv -Dinterop.dv.dir="$TMP"
#        -> The Java oracle writes an unpartitioned V3 table under "$TMP/table": two REAL parquet
#           data files (A: ids 10..50; B: ids 60/70/80) + a REAL Puffin deletion vector deleting
#           positions {1,3} of file A (ids 20/40), committed via newRowDelta().addDeletes(dv). It
#           writes "$TMP/table/metadata/final.metadata.json", materializes Java's OWN
#           merge-on-read READ into "$TMP/java_dv_scan_rows.json" (= {10,30,50,60,70,80}), and
#           emits the synthetic high-bits/run-container blob "$TMP/dv_blob.bin" +
#           "$TMP/dv_blob_expected.json".
#   2. ICEBERG_INTEROP_DV_DIR="$TMP" cargo test ... interop_dv_scan
#        -> The env-gated Rust test scans the SAME table via table.scan().to_arrow() (which loads
#           + applies the DV) and asserts the rows EQUAL Java's read: 20/40 ABSENT, file B intact.
#   3. ICEBERG_INTEROP_DV_DIR="$TMP" cargo test -p iceberg --lib test_dv_blob_decodes_java...
#        -> The env-gated lib test decodes the raw Java-serialized blob bytes (framing + portable
#           64-bit roaring incl. >2^32 positions + run containers) and asserts the position set.
#
# Without ICEBERG_INTEROP_DV_DIR both Rust tests are clean no-ops (the offline gate stays green);
# this script is what flips them into the REAL comparison.
#
# Requirements:
#   - Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64.
#   - A Rust toolchain (the repo's pinned nightly via rust-toolchain.toml).
#   - The FIRST Maven run must be ONLINE if ~/.m2 lacks the oracle deps (iceberg-core/-data/
#     -parquet/hadoop-client-runtime 1.10.0); after that, `mvn -o` works fully offline.
#
# Run from anywhere; paths are resolved relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-dv"

echo "==> [1/3] Reset the temp table dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/3] Java oracle: write a REAL V3 table (parquet data + Puffin deletion vector) + emit java_dv_scan_rows.json + dv_blob.bin"
(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
    /opt/maven/bin/mvn -q compile exec:java \
    -Dexec.args=generate-interop-dv \
    -Dinterop.dv.dir="${TMP}"
)

echo "==> [3/3] Rust: scan the V3 DV table (merge-on-read) + decode the raw Java DV blob, compare vs Java"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_DV_DIR="${TMP}" \
    cargo test -p iceberg --test interop_dv_scan -- --nocapture
  ICEBERG_INTEROP_DV_DIR="${TMP}" \
    cargo test -p iceberg --lib test_dv_blob_decodes_java_written_blob_when_env_set -- --nocapture
)

echo "==> DONE — deletion-vector interop passed (Rust scan == Java read, live rows {10,30,50,60,70,80}; raw blob decode matched incl. >2^32 positions + run containers)."
