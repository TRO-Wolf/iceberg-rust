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
# UpdateSchema bidirectional interop harness runner.
#
# This is a TEST-ONLY ORACLE (a dev tool, like dev/spark/) — it is NOT part of the shipped Rust library.
# It proves byte-/field-id-level UpdateSchema compatibility with the Java `iceberg-core` reference in
# BOTH directions and regenerates the committed JSON fixtures:
#
#   1. ICEBERG_INTEROP_GEN=1 cargo test ... interop_update_schema
#        -> Rust applies each scenario's op-sequence to the Java-written base.metadata.json and writes
#           rust_evolved.metadata.json (Direction 2 producer). The same run also asserts Direction 1
#           (Rust reproduces Java's evolution) against the committed java_evolved.metadata.json.
#   2. mvn ... -Dexec.args=generate
#        -> The Java oracle (re)writes base.metadata.json + java_evolved.metadata.json for each scenario.
#   3. mvn ... -Dexec.args=verify
#        -> The Java oracle reads each rust_evolved.metadata.json and asserts Java parses it and its
#           current schema matches Java's own evolution (Direction 2 verifier). Exits non-zero on any FAIL.
#
# Step order note: the Rust producer (step 1) reads the base + java_evolved fixtures, and the Java
# generator (step 2) (re)writes them. On a clean checkout the fixtures are already committed, so step 1
# can run first. If you change the Java scenarios, run step 2 FIRST to refresh the base/java fixtures,
# then step 1 to refresh the Rust output, then step 3 to verify. This script runs generate (2) before
# the Rust producer is re-run, so a single invocation always regenerates everything consistently.
#
# Requirements:
#   - Maven at /opt/maven/bin/mvn (override with MVN=...).
#   - A Rust toolchain (the repo's pinned nightly via rust-toolchain.toml).
#   - First Maven run downloads iceberg-core/iceberg-api 1.10.0 from Maven Central.
#
# Run from anywhere; paths are resolved relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MVN="${MVN:-/opt/maven/bin/mvn}"

echo "==> [1/4] Java oracle: generate base + java_evolved fixtures"
"${MVN}" -f "${SCRIPT_DIR}" -q compile exec:java -Dexec.args=generate

echo "==> [2/4] Rust: regenerate rust_evolved fixtures + assert Direction 1 (Rust reproduces Java)"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_GEN=1 cargo test -p iceberg --test interop_update_schema
)

echo "==> [3/4] Java oracle: verify (Direction 2 — Java reads Rust output)"
"${MVN}" -f "${SCRIPT_DIR}" -q exec:java -Dexec.args=verify

echo "==> [4/4] Done — both directions passed."
