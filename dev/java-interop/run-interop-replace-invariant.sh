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
# REPLACE record-count invariant interop (plan PR-1 / clause C-002, GAP_MATRIX row R107).
# Java SnapshotProducer.apply (1.10.0 offsets 311-364) refuses a replace snapshot whose
# summary has added-records > deleted-records. This runner proves:
#   1. The same 3-row-to-5-row replacement is refused by Java AND by Rust.
#   2. A valid 3-row-to-3-row replacement written by Java is read by Rust with the same rows.
#   3. A valid 3-row-to-3-row replacement written by Rust is read by Java with the same rows.
# Fixture count is asserted (HARD-FAIL, never skip).
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64.
# Run from anywhere; paths resolve relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-replace-invariant"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

for required in "${MVN}" "${JAVA_HOME}/bin/java"; do
  if [[ ! -x "${required}" ]]; then
    echo "FAIL replace-invariant: missing required executable ${required}"
    exit 1
  fi
done

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

EXPECTED_FIXTURES=3

assert_fixture_count() {
  local missing=0
  local present=0
  if [[ -f "${TMP}/invalid/threw.json" ]]; then
    present=$((present + 1))
  else
    echo "FAIL replace-invariant fixture missing: invalid/threw.json"
    missing=1
  fi
  if [[ -f "${TMP}/valid_java/java_rows.json" ]]; then
    present=$((present + 1))
  else
    echo "FAIL replace-invariant fixture missing: valid_java/java_rows.json"
    missing=1
  fi
  if [[ -f "${TMP}/valid_rust/rust_table/metadata/final.metadata.json" ]]; then
    present=$((present + 1))
  else
    echo "FAIL replace-invariant fixture missing: valid_rust/rust_table/metadata/final.metadata.json"
    missing=1
  fi
  if [[ "${present}" -ne "${EXPECTED_FIXTURES}" || "${missing}" -ne 0 ]]; then
    echo "FAIL replace-invariant fixture count: present=${present} expected=${EXPECTED_FIXTURES}"
    exit 1
  fi
  echo "==> fixture count ${present}/${EXPECTED_FIXTURES}"
}

echo "==> [1/6] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/6] Java generate (invalid 3-to-5 throws, valid 3-to-3 commits)"
run_oracle -Dexec.args=generate-interop-replace-invariant \
  -Dinterop.replace_invariant.dir="${TMP}"

echo "==> [3/6] Rust GEN (invalid 3-to-5 throws, valid 3-to-3 writes rust_table)"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_REPLACE_INVARIANT_GEN_DIR="${TMP}" \
    cargo test -p iceberg --test interop_replace_invariant \
    test_replace_invariant_gen_rust_writes_valid_and_refuses_invalid -- --nocapture
)

echo "==> [4/6] Java verify (invalid still throws, Java reads Rust 3-to-3 table)"
VERIFY_OUT="$(run_oracle -Dexec.args=verify-interop-replace-invariant \
  -Dinterop.replace_invariant.dir="${TMP}")" || true
echo "${VERIFY_OUT}"
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-replace-invariant: 0 failures'; then
  echo "==> FAILED — D2 verify emitted a FAIL line or did not emit the '0 failures' sentinel."
  exit 1
fi

echo "==> [5/6] Rust D1 (Rust reads Java 3-to-3 table) + Java threw.json pin"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_REPLACE_INVARIANT_DIR="${TMP}" \
    cargo test -p iceberg --test interop_replace_invariant \
    test_replace_invariant -- --nocapture
)

echo "==> [6/6] Fixture count"
assert_fixture_count

echo "==> replace-invariant interop OK (Java and Rust both refuse 3-to-5; both directions read 3-to-3)"
