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
TMP="${SCRIPT_DIR}/target/interop-v3-maintenance"
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

echo "==> [1/6] Reset ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/6] Java: write the partitioned V2 seeds (plain, and one with parquet position deletes)"
run_oracle -Dexec.args=generate-interop-v3-maintenance -Dinterop.v3_maintenance.dir="${TMP}"
expect_file "${TMP}/fixture_count.json"
COUNT="$(tr -d '[:space:]' < "${TMP}/fixture_count.json")"
if [[ "${COUNT}" != '{"count":2}' ]]; then
  echo "FAIL: Java fixture count ${COUNT} != {\"count\":2}" >&2
  exit 1
fi
expect_file "${TMP}/java_v2_plain/metadata/final.metadata.json"
expect_file "${TMP}/java_v2_plain/java_rows.json"
expect_file "${TMP}/java_v2_deletes/metadata/final.metadata.json"
expect_file "${TMP}/java_v2_deletes/java_rows.json"

echo "==> [3/6] Rust: upgrade both seeds to V3 and run the five maintenance actions"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_V3_MAINTENANCE_DIR="${TMP}" \
    cargo test -p iceberg --test interop_v3_maintenance gen_ -- --nocapture
)
for stage in plain/m0 plain/m1 plain/m2 deletes/m0 deletes/m3 deletes/m4 deletes/m5; do
  expect_file "${TMP}/${stage}/metadata/final.metadata.json"
done
expect_file "${TMP}/plain/expected.json"
expect_file "${TMP}/deletes/expected.json"

echo "==> [4/6] Java: production scan of every maintenance stage"
VERIFY_OUT="$(run_oracle -Dexec.args=verify-interop-v3-maintenance -Dinterop.v3_maintenance.dir="${TMP}" || true)"
echo "${VERIFY_OUT}"
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-v3-maintenance: 0 failures'; then
  echo "FAIL: the clean Java verify did not report 0 failures" >&2
  exit 1
fi

echo "==> [5/6] Fixture count"
FIXTURES="$(find "${TMP}" -name 'final.metadata.json' | wc -l | tr -d ' ')"
if [[ "${FIXTURES}" != "${EXPECTED_FINAL_METADATA}" ]]; then
  echo "FAIL: expected ${EXPECTED_FINAL_METADATA} final.metadata.json files, got ${FIXTURES}" >&2
  find "${TMP}" -name 'final.metadata.json' >&2
  exit 1
fi

echo "==> [6/6] Sabotage battery"
build_scratch() {
  local scratch="${TMP}/sabotage_scratch"
  rm -rf "${scratch}"
  mkdir -p "${scratch}"
  cp -r "${TMP}/plain" "${TMP}/deletes" "${TMP}/java_v2_plain" "${TMP}/java_v2_deletes" "${scratch}/"
  cp "${TMP}/fixture_count.json" "${scratch}/fixture_count.json"
  echo "${scratch}"
}

sabotage_swap() {
  local label="$1" target="$2" source="$3" needle="$4"
  local scratch
  scratch="$(build_scratch)"
  if [[ ! -f "${scratch}/${target}" || ! -f "${scratch}/${source}" ]]; then
    echo "FAIL: cannot apply sabotage(${label}) — ${target} or ${source} is absent" >&2
    exit 1
  fi
  cp "${scratch}/${source}" "${scratch}/${target}"
  local out
  out="$(run_oracle -Dexec.args=verify-interop-v3-maintenance -Dinterop.v3_maintenance.dir="${scratch}" || true)"
  if echo "${out}" | grep -q 'verify-interop-v3-maintenance: 0 failures'; then
    echo "FAIL sabotage(${label}): the verify stayed green" >&2
    echo "${out}" >&2
    exit 1
  fi
  if ! echo "${out}" | grep -q "${needle}"; then
    echo "FAIL sabotage(${label}): the verify failed but not through the rule under test" >&2
    echo "${out}" >&2
    exit 1
  fi
  echo "PASS sabotage(${label})"
}

sabotage_swap "delete-conversion" \
  "deletes/m3/metadata/final.metadata.json" \
  "deletes/m0/metadata/final.metadata.json" \
  "parquet position delete survived the V3 conversion"

sabotage_swap "no-op-rewrite" \
  "plain/m1/metadata/final.metadata.json" \
  "plain/m0/metadata/final.metadata.json" \
  "rewrite left the live data-file set unchanged"

sabotage_swap "evolved-spec" \
  "plain/m2/metadata/final.metadata.json" \
  "plain/m1/metadata/final.metadata.json" \
  "FAIL v3-maintenance plain/m2: current spec is"

sabotage_swap "manifest-clustering" \
  "deletes/m4/metadata/final.metadata.json" \
  "deletes/m3/metadata/final.metadata.json" \
  "clustered data manifests"

SCRATCH="$(build_scratch)"
M0_META="${SCRATCH}/plain/m0/metadata/final.metadata.json"
if [[ ! -f "${M0_META}" ]]; then
  echo "FAIL: cannot apply the truncate sabotage — ${M0_META} is absent" >&2
  exit 1
fi
SIZE="$(stat -c%s "${M0_META}")"
if (( SIZE <= 60 )); then
  echo "FAIL: cannot apply the truncate sabotage — ${M0_META} is only ${SIZE} bytes" >&2
  exit 1
fi
head -c "$(( SIZE - 60 ))" "${M0_META}" > "${M0_META}.tmp"
mv "${M0_META}.tmp" "${M0_META}"
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-v3-maintenance -Dinterop.v3_maintenance.dir="${SCRATCH}" || true)"
if echo "${SAB_OUT}" | grep -q 'verify-interop-v3-maintenance: 0 failures'; then
  echo "FAIL sabotage(truncate): the verify stayed green on truncated metadata" >&2
  echo "${SAB_OUT}" >&2
  exit 1
fi
if ! echo "${SAB_OUT}" | grep -q 'FAIL v3-maintenance plain: unexpected error'; then
  echo "FAIL sabotage(truncate): the verify failed but not through the load path" >&2
  echo "${SAB_OUT}" >&2
  exit 1
fi
echo "PASS sabotage(truncate)"
rm -rf "${TMP}/sabotage_scratch"

echo "==> DONE — v3 maintenance interop passed (${EXPECTED_FINAL_METADATA} final.metadata.json, 5 actions)."
