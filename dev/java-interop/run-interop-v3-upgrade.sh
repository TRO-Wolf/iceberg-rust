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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-v3-upgrade"
JAVA_HOME_DIR="${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk-amd64}"
MVN="${MVN:-/opt/maven/bin/mvn}"
EXPECTED_FINAL_METADATA=9

if [[ ! -x "${MVN}" ]]; then
  echo "FAIL: maven missing at ${MVN}" >&2
  exit 1
fi
if [[ ! -x "${JAVA_HOME_DIR}/bin/java" ]]; then
  echo "FAIL: JDK missing at ${JAVA_HOME_DIR}" >&2
  exit 1
fi

run_oracle() {
  (
    cd "${SCRIPT_DIR}"
    JAVA_HOME="${JAVA_HOME_DIR}" PATH="${JAVA_HOME_DIR}/bin:$PATH" \
      "${MVN}" -o -q compile exec:java "$@" 2>&1
  )
}

expect_file() {
  if [[ ! -f "$1" ]]; then
    echo "FAIL: missing fixture $1" >&2
    exit 1
  fi
}

echo "==> [1/9] Reset ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/9] Java: write the V2 seeds (u1 plain, u3 with a parquet position delete)"
run_oracle -Dexec.args=generate-interop-v3-upgrade -Dinterop.v3_upgrade.dir="${TMP}"
expect_file "${TMP}/fixture_count.json"
COUNT="$(tr -d '[:space:]' < "${TMP}/fixture_count.json")"
if [[ "${COUNT}" != '{"count":2}' ]]; then
  echo "FAIL: Java fixture count ${COUNT} != {\"count\":2}" >&2
  exit 1
fi
expect_file "${TMP}/u1/java_v2/metadata/final.metadata.json"
expect_file "${TMP}/u3/java_v2/metadata/final.metadata.json"
expect_file "${TMP}/u3/java_pre_rows.json"

echo "==> [3/9] Rust: upgrade the Java V2 tables, seed the Rust V2 tables, convert the u3 deletes"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_V3_UPGRADE_DIR="${TMP}" \
    cargo test -p iceberg --test interop_v3_upgrade gen_ -- --nocapture
)
expect_file "${TMP}/u1/rust_v3/metadata/final.metadata.json"
expect_file "${TMP}/u1/rust_expected.json"
expect_file "${TMP}/u2/rust_v2/metadata/final.metadata.json"
expect_file "${TMP}/u3/rust_v3_dv/metadata/final.metadata.json"
expect_file "${TMP}/u4/rust_v2/metadata/final.metadata.json"
expect_file "${TMP}/u4/rust_pre_rows.json"

echo "==> [4/9] Java: upgrade the Rust V2 tables and convert u4 to deletion vectors"
run_oracle -Dexec.args=upgrade-interop-v3-upgrade -Dinterop.v3_upgrade.dir="${TMP}"
expect_file "${TMP}/u2/java_v3/metadata/final.metadata.json"
expect_file "${TMP}/u2/java_expected.json"
expect_file "${TMP}/u4/java_v3_dv/metadata/final.metadata.json"
expect_file "${TMP}/u4/java_expected.json"

echo "==> [5/9] Rust: merge-on-read UPDATE as the first V3 DML on the converted u3 table"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_V3_UPGRADE_DIR="${TMP}" \
    cargo test -p iceberg-datafusion --test interop_v3_upgrade_mor -- --nocapture
)
expect_file "${TMP}/u3/rust_v3_mor/metadata/final.metadata.json"
expect_file "${TMP}/u3/rust_expected.json"

echo "==> [6/9] Rust: read back the Java-upgraded and Java-converted V3 tables"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_V3_UPGRADE_DIR="${TMP}" \
    cargo test -p iceberg --test interop_v3_upgrade verify_ -- --nocapture
)

echo "==> [7/9] Java: production scan of the Rust-upgraded V3 tables"
VERIFY_OUT="$(run_oracle -Dexec.args=verify-interop-v3-upgrade -Dinterop.v3_upgrade.dir="${TMP}" || true)"
echo "${VERIFY_OUT}"
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-v3-upgrade: 0 failures'; then
  echo "FAIL: the clean Java verify did not report 0 failures" >&2
  exit 1
fi

echo "==> [8/9] Fixture count"
FIXTURES="$(find "${TMP}" -name 'final.metadata.json' | wc -l | tr -d ' ')"
if [[ "${FIXTURES}" != "${EXPECTED_FINAL_METADATA}" ]]; then
  echo "FAIL: expected ${EXPECTED_FINAL_METADATA} final.metadata.json files, got ${FIXTURES}" >&2
  find "${TMP}" -name 'final.metadata.json' >&2
  exit 1
fi

echo "==> [9/9] Sabotage battery"
build_scratch() {
  local scratch="${TMP}/sabotage_scratch"
  rm -rf "${scratch}"
  mkdir -p "${scratch}"
  cp -r "${TMP}/u1" "${TMP}/u2" "${TMP}/u3" "${TMP}/u4" "${scratch}/"
  cp "${TMP}/fixture_count.json" "${scratch}/fixture_count.json"
  echo "${scratch}"
}

SCRATCH="$(build_scratch)"
MOR_META="${SCRATCH}/u3/rust_v3_mor/metadata/final.metadata.json"
PRE_META="${SCRATCH}/u3/java_v2/metadata/final.metadata.json"
if [[ ! -f "${MOR_META}" || ! -f "${PRE_META}" ]]; then
  echo "FAIL: cannot apply the delete-conversion sabotage — a metadata file is absent" >&2
  exit 1
fi
cp "${PRE_META}" "${MOR_META}"
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-v3-upgrade -Dinterop.v3_upgrade.dir="${SCRATCH}" || true)"
if echo "${SAB_OUT}" | grep -q 'verify-interop-v3-upgrade: 0 failures'; then
  echo "FAIL sabotage(delete-conversion): the verify stayed green on a table still carrying a parquet position delete" >&2
  echo "${SAB_OUT}" >&2
  exit 1
fi
if ! echo "${SAB_OUT}" | grep -q 'parquet position delete survived the V3 conversion'; then
  echo "FAIL sabotage(delete-conversion): the verify failed but not through the delete-conversion rule" >&2
  echo "${SAB_OUT}" >&2
  exit 1
fi
echo "PASS sabotage(delete-conversion)"

SCRATCH="$(build_scratch)"
U1_META="${SCRATCH}/u1/rust_v3/metadata/final.metadata.json"
if [[ ! -f "${U1_META}" ]]; then
  echo "FAIL: cannot apply the truncate sabotage — ${U1_META} is absent" >&2
  exit 1
fi
SIZE="$(stat -c%s "${U1_META}")"
if (( SIZE <= 60 )); then
  echo "FAIL: cannot apply the truncate sabotage — ${U1_META} is only ${SIZE} bytes" >&2
  exit 1
fi
head -c "$(( SIZE - 60 ))" "${U1_META}" > "${U1_META}.tmp"
mv "${U1_META}.tmp" "${U1_META}"
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-v3-upgrade -Dinterop.v3_upgrade.dir="${SCRATCH}" || true)"
if echo "${SAB_OUT}" | grep -q 'verify-interop-v3-upgrade: 0 failures'; then
  echo "FAIL sabotage(truncate): the verify stayed green on truncated metadata" >&2
  echo "${SAB_OUT}" >&2
  exit 1
fi
if ! echo "${SAB_OUT}" | grep -q 'FAIL v3-upgrade u1: unexpected error'; then
  echo "FAIL sabotage(truncate): the verify failed but not through the load path" >&2
  echo "${SAB_OUT}" >&2
  exit 1
fi
echo "PASS sabotage(truncate)"
rm -rf "${TMP}/sabotage_scratch"

echo "==> DONE — v3 upgrade interop passed (${EXPECTED_FINAL_METADATA} final.metadata.json, 4 cells)."
