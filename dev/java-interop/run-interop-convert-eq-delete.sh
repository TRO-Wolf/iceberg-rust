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
# MAINTENANCE ConvertEqualityDeleteFiles interop harness (G4 capstone), VERIFY-only direction — "Java
# validates what RUST converted". The eq → pos delete-conversion action is engine-agnostic in Rust, but
# the real Java action is a Spark-surface class (NOT on the iceberg-core oracle classpath) and Java cannot
# DRIVE the conversion. So the corroboration is: RUST writes both the PRE table (a real EQUALITY delete
# masking a known subset) and the POST table (the same rows masked by the CONVERTED POSITION delete), and
# Java's IcebergGenerics reads BOTH and asserts the live row sets are IDENTICAL — the no-Spark proof that
# the converted position delete masks EXACTLY the rows the equality delete masked.
#
# This is a TEST-ONLY ORACLE (a dev tool) — it is NOT part of the shipped Rust library, and NOT part of the
# offline `cargo test` gate (it needs Java + Maven). The committed artifacts are the oracle code
# (ConvertEqDeleteOracle in InteropOracle.java), the Rust GEN test (interop_convert_eq_delete.rs), and this
# script. The temp table under dev/java-interop/target/ is gitignored.
#
# Methodology (Rust GEN → Java VERIFY):
#   1. ICEBERG_INTEROP_CONVERT_EQ_DELETE_GEN_DIR="$TMP" cargo test ... test_convert_eq_delete_gen
#        -> Rust builds a partitioned V2 table under "$TMP/rust_table" (cat=A id 100/120/130, cat=B id
#           200/220/230 at seq 1; one EQUALITY delete per partition deleting y=20 at seq 2 — masking id=120
#           and id=220), lands "$TMP/rust_table/metadata/final.metadata.json". It then builds a SEPARATE
#           identical table under "$TMP/rust_table_converted", runs ConvertEqualityDeleteFiles (eq → pos),
#           and lands "$TMP/rust_table_converted/metadata/final.metadata.json". A Rust-side sanity check
#           confirms read-identity + the eq→pos shape BEFORE handing the tables to Java.
#   2. mvn ... -Dexec.args=verify-interop-convert-eq-delete -Dinterop.convert_eq_delete.dir="$TMP"
#        -> Java loads BOTH tables, reads each via IcebergGenerics, and asserts: PRE (eq) live ids ==
#           {100,130,200,230}; READ IDENTITY POST (pos) == PRE (eq); and the eq→pos conversion shape
#           (PRE eq>0/pos=0, POST eq=0/pos>0). Nonzero failures ⇒ exit 1.
#
# Without ICEBERG_INTEROP_CONVERT_EQ_DELETE_GEN_DIR the Rust GEN test is a clean no-op (it stays green in
# the offline gate); this script is what flips it into the REAL comparison.
#
# Requirements:
#   - Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64.
#   - A Rust toolchain (the repo's pinned nightly via rust-toolchain.toml).
#   - The Maven deps (iceberg-core / iceberg-data / iceberg-parquet / hadoop-client-runtime 1.10.0) are the
#     SAME ones the equality-delete harness already pulled — no new pom deps. If ~/.m2 is populated, `mvn -o`
#     runs fully offline.
#
# Run from anywhere; paths are resolved relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-convert-eq-delete"

echo "==> [1/3] Reset the temp table dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/3] Rust GEN: write rust_table (equality delete) + rust_table_converted (converted position delete)"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_CONVERT_EQ_DELETE_GEN_DIR="${TMP}" \
    cargo test -p iceberg --test interop_convert_eq_delete test_convert_eq_delete_gen -- --nocapture
)

echo "==> [3/3] Java verify: IcebergGenerics reads BOTH tables, asserts read-identity (eq == pos) + the eq→pos shape"
(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
    /opt/maven/bin/mvn -o -q compile exec:java \
    -Dexec.args=verify-interop-convert-eq-delete \
    -Dinterop.convert_eq_delete.dir="${TMP}"
)

echo "==> DONE — ConvertEqualityDeleteFiles interop passed (Java's read IDENTICAL before eq-delete vs after converted pos-delete; live ids {100,130,200,230})."
