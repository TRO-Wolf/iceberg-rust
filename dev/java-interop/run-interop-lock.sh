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
# LOCK MANAGER behavioral-conformance interop harness (GAP_MATRIX row 129).
#
# Proves that the Rust InMemoryLockManager produces the SAME observable acquire/release outcomes as the
# REAL Java org.apache.iceberg.util.LockManagers InMemoryLockManager over an identical deterministic
# 7-step sequence. The lock manager is an IN-PROCESS primitive (no on-disk artifact), so this is
# OUTCOME-CONFORMANCE, not a byte round-trip.
#
# THE CHAIN:
#   1. Reset the temp dir.
#   2. JAVA generates: drives the default InMemoryLockManager (reconfigured to small acquire-timeouts)
#      through the 7-step sequence, self-checks its booleans against a hand-declared expected, and emits
#      java_lock_outcomes.json. Prints "generate-interop-lock: 0 failures" on success.
#   3. RUST compares (cargo test, env ICEBERG_INTEROP_LOCK_DIR): runs the SAME 7-step sequence against its
#      InMemoryLockManager and asserts its outcomes == the hand-declared expected AND == java_lock_outcomes.json.
#
# NO BEHAVIORAL DIVERGENCE: at acquire-while-held, Java acquire internally catches the acquire-timeout
# IllegalStateException and returns false (1.10.0 bytecode exception-table verified), identical to Rust.
#
# FAIL-CLOSED: the Java outcome is read from the printed sentinel ("generate-interop-lock: 0 failures"),
# not from the mvn exit code (unreliable for exec:java + System.exit). The Rust step fails LOUDLY (panic)
# on any outcome mismatch.
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64, the repo's
# pinned Rust toolchain.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-lock"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

echo "==> [1/3] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/3] Java: drive the real InMemoryLockManager through the 7-step sequence (emit java_lock_outcomes.json)"
GEN_OUT="$(
  cd "${SCRIPT_DIR}"
  "${MVN}" -o -q compile exec:java \
    -Dexec.args=generate-interop-lock \
    -Dinterop.lock.dir="${TMP}" 2>&1
)" || true
echo "${GEN_OUT}"
if echo "${GEN_OUT}" | grep -q '^FAIL ' \
  || ! echo "${GEN_OUT}" | grep -q 'generate-interop-lock: 0 failures'; then
  echo "==> FAILED — Java InMemoryLockManager diverged from the hand-declared expected outcomes."
  exit 1
fi

echo "==> [3/3] Rust: run the SAME sequence + assert outcomes == hand-declared expected AND == java_lock_outcomes.json"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_LOCK_DIR="${TMP}" \
    cargo test -p iceberg --test interop_lock test_lock_conformance_interop \
    -- --exact --nocapture
)

echo "==> DONE — LockManager behavioral-conformance interop passed:"
echo "    Rust InMemoryLockManager outcomes are byte-identical to the real Java InMemoryLockManager"
echo "    across the 7-step acquire/release sequence (acquire-timeout returns false on both sides —"
echo "    Java catches the IllegalStateException internally, identical to Rust)."
