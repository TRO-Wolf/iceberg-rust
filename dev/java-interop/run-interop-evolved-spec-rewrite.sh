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
# Evolved-spec RewriteDataFiles interop (GAP_MATRIX row R135 / plan C-001).
#
# Three fixtures:
#   D1  Java writes identity(x); Rust evolves to identity(y) and compacts; Java reads
#   D2  Rust writes identity(x); Java evolves and rewrites; Rust reads
#   V3  Rust writes V3 identity(x); Rust evolves and compacts; Java compares _row_id
#
# FAIL-CLOSED: missing Maven, Java, or a fixture count other than 3 exits non-zero.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-evolved-spec-rewrite"
MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

if [[ ! -x "${MVN}" ]]; then
  echo "FAIL: missing Maven at ${MVN}" >&2
  exit 1
fi
if [[ ! -d "${JAVA_HOME}" ]]; then
  echo "FAIL: missing JAVA_HOME ${JAVA_HOME}" >&2
  exit 1
fi

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

expect_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "FAIL: missing fixture ${path}" >&2
    exit 1
  fi
}

echo "==> [1/7] Reset ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/7] Java generate D1 old-spec table"
GEN_OUT="$(run_oracle -Dexec.args=generate-interop-evolved-spec-rewrite \
  -Dinterop.evolved_spec_rewrite.dir="${TMP}")"
echo "${GEN_OUT}"
expect_file "${TMP}/d1/table/metadata/final.metadata.json"

echo "==> [3/7] Rust: compact D1, write D2 old-spec, compact V3"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_EVOLVED_SPEC_REWRITE_DIR="${TMP}" \
    cargo test -p iceberg --test interop_evolved_spec_rewrite \
    rust_compacts_java_d1_and_writes_d2_and_v3 -- --exact --nocapture
)
expect_file "${TMP}/d1/compacted/metadata/final.metadata.json"
expect_file "${TMP}/d2/rust_table/metadata/final.metadata.json"
expect_file "${TMP}/v3/compacted/metadata/final.metadata.json"
expect_file "${TMP}/v3/compacted/expected_row_ids.json"

echo "==> [4/7] Java evolve+rewrite D2"
REWRITE_OUT="$(run_oracle -Dexec.args=rewrite-interop-evolved-spec-d2 \
  -Dinterop.evolved_spec_rewrite.dir="${TMP}")"
echo "${REWRITE_OUT}"
expect_file "${TMP}/d2/rewritten/metadata/final.metadata.json"

echo "==> [5/7] Java verify D1 compacted + V3 row ids"
VERIFY_OUT="$(run_oracle -Dexec.args=verify-interop-evolved-spec-rewrite \
  -Dinterop.evolved_spec_rewrite.dir="${TMP}" || true)"
echo "${VERIFY_OUT}"
if echo "${VERIFY_OUT}" | grep -q 'FAIL ' \
  || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-evolved-spec-rewrite: 0 failures'; then
  echo "FAIL: Java verify did not report 0 failures" >&2
  exit 1
fi

echo "==> [6/7] Rust reads Java D2 rewritten table"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_EVOLVED_SPEC_REWRITE_DIR="${TMP}" \
    cargo test -p iceberg --test interop_evolved_spec_rewrite \
    rust_reads_java_d2_rewritten_table -- --exact --nocapture
)

FIXTURE_COUNT="$(find "${TMP}" -name 'final.metadata.json' | wc -l | tr -d ' ')"
if [[ "${FIXTURE_COUNT}" != "5" ]]; then
  echo "FAIL: expected 5 final.metadata.json files (d1 table, d1 compacted, d2 rust, d2 rewritten, v3 compacted), got ${FIXTURE_COUNT}" >&2
  find "${TMP}" -name 'final.metadata.json' >&2
  exit 1
fi

echo "==> [7/7] Sabotage: truncated D1 compacted metadata must fail verify"
D1_COMPACTED="${TMP}/d1/compacted/metadata/final.metadata.json"
cp -a "${D1_COMPACTED}" "${D1_COMPACTED}.bak"
printf 'not-json' > "${D1_COMPACTED}"
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-evolved-spec-rewrite \
  -Dinterop.evolved_spec_rewrite.dir="${TMP}" || true)"
mv "${D1_COMPACTED}.bak" "${D1_COMPACTED}"
touch "${D1_COMPACTED}"
if echo "${SAB_OUT}" | grep -q 'verify-interop-evolved-spec-rewrite: 0 failures'; then
  echo "FAIL: sabotage verify stayed green" >&2
  echo "${SAB_OUT}" >&2
  exit 1
fi
if ! echo "${SAB_OUT}" | grep -Eq 'FAIL |failures'; then
  echo "FAIL: sabotage produced no FAIL signal" >&2
  echo "${SAB_OUT}" >&2
  exit 1
fi

echo "==> DONE — evolved-spec RewriteDataFiles interop passed (3 fixtures, both directions + V3 _row_id)."
