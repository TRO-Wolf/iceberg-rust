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
# PARTITION-PATH URL-escaping interop harness (GAP_MATRIX row R161) — cross-impl conformance of
# org.apache.iceberg.PartitionSpec.partitionToPath between Java 1.10.0 and the Rust
# PartitionSpec::partition_to_path.
#
# Java escapes BOTH sides of every `name=value` pair with `URLEncoder.encode(s, "UTF-8")`
# (PartitionSpec.escape, 1.10.0 bytecode). partitionToPath is a PURE function of (spec, schema,
# tuple) with no on-disk artifact, so this is a CONFORMANCE oracle, not a byte round-trip: each side
# builds the SAME named case INDEPENDENTLY and only the RESULT is compared. Nothing is copied across
# to derive the expected, so a pass cannot be an echo.
#
# THE CHAIN:
#   1. Reset the temp dir.
#   2. Java: PartitionPathOracle renders 22 named cases -> java_partition_paths.json.
#   3. Rust: the env-gated tests/interop_partition_path.rs rebuilds each case from its own battery
#      and asserts byte-equality (plus set-equality of the case ids).
#   4. SABOTAGE (non-vacuity): two corruptions of the Java fixture that the comparison MUST catch —
#      (a) un-escaping one expectation, (b) deleting one case. Each leg HARD-FAILS the whole script
#      if the corruption cannot be applied: a sabotage step that changed nothing has proven nothing.
#   5. Restore + re-run green.
#
# Part of the offline-capable interop set: no Docker, no credentials — just Maven + a JDK and the
# local iceberg-core/api 1.10.0 jars in ~/.m2. The temp dir under target/ is gitignored.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-partition-path"
FIXTURE="${TMP}/java_partition_paths.json"
BAK="${TMP}/java_partition_paths.json.bak"
MVN="${MVN:-/opt/maven/bin/mvn}"

export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk-amd64}"
export PATH="${JAVA_HOME}/bin:${PATH}"

run_rust() {
  (
    cd "${REPO_ROOT}"
    ICEBERG_INTEROP_PARTITION_PATH_DIR="${TMP}" \
      cargo test -p iceberg --test interop_partition_path -- --nocapture
  )
}

# Corrupt the Java fixture in place. HARD-FAILS (exit 1) when the target pattern is absent, so a
# sabotage leg can never silently degrade into a no-op.
corrupt() {
  local description="$1" pattern="$2" replacement="$3"
  if ! grep -qF -- "${pattern}" "${FIXTURE}"; then
    echo "!!! HARD-FAIL: sabotage '${description}' cannot be applied — pattern not found: ${pattern}"
    cp -p "${BAK}" "${FIXTURE}"
    exit 1
  fi
  python3 - "${FIXTURE}" "${pattern}" "${replacement}" <<'PY'
import io, sys
path, pattern, replacement = sys.argv[1], sys.argv[2], sys.argv[3]
s = io.open(path, encoding='utf-8').read()
if pattern not in s:
    sys.exit(1)
io.open(path, 'w', encoding='utf-8').write(s.replace(pattern, replacement, 1))
PY
  cmp -s "${BAK}" "${FIXTURE}" && {
    echo "!!! HARD-FAIL: sabotage '${description}' left the fixture byte-identical"
    cp -p "${BAK}" "${FIXTURE}"
    exit 1
  }
  echo "--- sabotage applied: ${description}"
}

expect_rust_failure() {
  local description="$1"
  local rc=0
  run_rust >/dev/null 2>&1 || rc=$?
  cp -p "${BAK}" "${FIXTURE}"
  if [ "${rc}" -eq 0 ]; then
    echo "!!! FALSE GREEN: the Rust comparison PASSED under sabotage '${description}'"
    exit 1
  fi
  echo "--- sabotage '${description}' correctly caught (rust exit ${rc}); fixture restored"
}

echo "==> [1/5] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/5] Java: render the 22 partitionToPath cases -> java_partition_paths.json"
(cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java \
  -Dexec.args=generate-interop-partition-path \
  -Dinterop.partition_path.dir="${TMP}")

echo "==> [3/5] Rust: rebuild each case independently and byte-compare"
run_rust

cp -p "${FIXTURE}" "${BAK}"

echo "==> [4/5] SABOTAGE (non-vacuity)"
corrupt "un-escape slash_value" '"slash_value" : "s=a%2Fb"' '"slash_value" : "s=a/b"'
expect_rust_failure "un-escape slash_value"
corrupt "drop the void_null case" '    "void_null" : "s_void=null"' '    "dropped_case" : "s_void=null"'
expect_rust_failure "drop the void_null case"

echo "==> [5/5] Re-run green after restore"
run_rust
rm -f "${BAK}"

echo "==> partition-path interop OK — 22 cases byte-match Java partitionToPath, sabotage caught twice"
