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
# AGGREGATE-pushdown interop harness (G1) — cross-impl conformance of the metrics fold between the
# Rust `AggregateEvaluator` (expr/visitors/aggregate_evaluator.rs) and Java 1.10.0 production
# `org.apache.iceberg.expressions.AggregateEvaluator.result()`.
#
# The fold is a pure function of a DataFile's metrics (record_count, value_counts, null_value_counts,
# lower_bounds, upper_bounds), so no real parquet is needed: both engines hand-declare the SAME
# fixture (3 files; anti-circular) and must produce the same count(*)/count/min/max. A second fixture
# drops a value-count so count(id) is not pushable, proving both latch all_valid=false.
#
# THE CHAIN:
#   1. Reset the temp dir.
#   2. Java: AggregateOracle folds the fixture → java_aggregate.json + java_aggregate_not_pushable.json
#   3. Rust: the env-gated in-crate test folds the SAME fixture, asserts it == Java AND ==
#      the hand-computed expected, and asserts the not-pushable fixture latches all_valid=false.
#
# This is part of the offline-capable interop set: no Docker, no credentials — just Maven + a JDK and
# the local iceberg-core/api 1.10.0 jars in ~/.m2. The temp dir under target/ is gitignored.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-aggregate"
MVN="${MVN:-/opt/maven/bin/mvn}"

export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk-amd64}"
export PATH="${JAVA_HOME}/bin:${PATH}"

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

echo "==> [1/3] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/3] Java: fold the aggregate fixture → java_aggregate.json + java_aggregate_not_pushable.json"
run_oracle -Dexec.args=generate-interop-aggregate -Dinterop.aggregate.dir="${TMP}"

echo "==> [3/3] Rust: fold the SAME fixture + assert == Java AND == hand-computed expected"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_AGGREGATE_DIR="${TMP}" \
    cargo test -p iceberg --lib \
    expr::visitors::aggregate_evaluator::tests::interop_aggregate_matches_java \
    -- --exact --nocapture
)

echo "==> aggregate interop OK — Rust fold == Java AggregateEvaluator.result() both fixtures"
