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
TMP="${SCRIPT_DIR}/target/interop-mor-update-lineage"
JAVA_HOME_DIR="${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk-amd64}"
MVN="${MVN:-/opt/maven/bin/mvn}"

if [[ ! -x "${MVN}" ]]; then
  echo "FAIL: maven missing at ${MVN}"
  exit 1
fi
if [[ ! -d "${JAVA_HOME_DIR}" ]]; then
  echo "FAIL: JAVA_HOME missing at ${JAVA_HOME_DIR}"
  exit 1
fi

echo "==> [1/4] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/4] Java: write two 3-row V3 tables"
(
  cd "${SCRIPT_DIR}"
  JAVA_HOME="${JAVA_HOME_DIR}" PATH="${JAVA_HOME_DIR}/bin:$PATH" \
    "${MVN}" -o -q compile exec:java \
    -Dexec.args=generate-interop-mor-update-lineage \
    -Dinterop.mor_update_lineage.dir="${TMP}"
)
if [[ ! -f "${TMP}/fixture_count.json" ]]; then
  echo "FAIL: Java generate did not write fixture_count.json"
  exit 1
fi
COUNT="$(tr -d '[:space:]' < "${TMP}/fixture_count.json")"
if [[ "${COUNT}" != '{"count":2}' ]]; then
  echo "FAIL: fixture count ${COUNT} != {\"count\":2}"
  exit 1
fi
if [[ ! -f "${TMP}/mor_table/metadata/final.metadata.json" ]] \
  || [[ ! -f "${TMP}/cow_table/metadata/final.metadata.json" ]]; then
  echo "FAIL: expected 2 Java tables (mor_table, cow_table)"
  exit 1
fi

echo "==> [3/4] Rust: two MoR UPDATE statements + COW UPDATE-then-DELETE"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_MOR_UPDATE_LINEAGE_GEN_DIR="${TMP}" \
    cargo test -p iceberg-datafusion --test interop_mor_update_lineage -- --nocapture
)

echo "==> [4/4] Java: production scan of Rust DML results"
VERIFY_OUT="$(
  cd "${SCRIPT_DIR}"
  JAVA_HOME="${JAVA_HOME_DIR}" PATH="${JAVA_HOME_DIR}/bin:$PATH" \
    "${MVN}" -o -q compile exec:java \
    -Dexec.args=verify-interop-mor-update-lineage \
    -Dinterop.mor_update_lineage.dir="${TMP}" 2>&1
)" || true
echo "${VERIFY_OUT}"
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${VERIFY_OUT}" | grep -qE 'verify-interop-mor-update-lineage: 0 failures$'; then
  echo "==> FAILED — Java could not confirm MoR UPDATE / COW rewrite lineage."
  exit 1
fi

echo "==> interop-mor-update-lineage PASSED (2 fixtures)"
