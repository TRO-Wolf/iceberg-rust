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
TMP="${SCRIPT_DIR}/target/interop-f21-legacy-delete-merge"
MVN=/opt/maven/bin/mvn
JDK=/usr/lib/jvm/java-11-openjdk-amd64
FIXTURE_COUNT=1

for prereq in "${MVN}" "${JDK}/bin/java"; do
  if [ ! -x "${prereq}" ]; then
    echo "==> FAILED -- missing prerequisite ${prereq}" >&2
    exit 1
  fi
done

echo "==> [1/6] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/6] Java: writes the V2 seed with a parquet position delete"
(
  cd "${SCRIPT_DIR}"
  JAVA_HOME="${JDK}" PATH="${JDK}/bin:${PATH}" \
    "${MVN}" -o -q compile exec:java \
    -Dexec.args=generate-interop-f21-legacy-delete-merge \
    -Dinterop.f21_legacy_delete_merge.dir="${TMP}/seed"
)
if [ ! -d "${TMP}/seed/table/metadata" ]; then
  echo "==> FAILED -- the Java seed produced no table" >&2
  exit 1
fi

echo "==> [3/6] Rust: upgrades to V3 and DELETE id=3, merging the parquet delete"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_F21_JAVA_SHARED="${TMP}/seed" \
    cargo test -p iceberg-datafusion --locked --test interop_f21_legacy_delete_merge -- --nocapture
)

echo "==> [4/6] Fixture count"
produced="$(find "${TMP}/seed/after_delete" -maxdepth 1 -type f | wc -l)"
if [ "${produced}" -ne "${FIXTURE_COUNT}" ]; then
  echo "==> FAILED -- expected ${FIXTURE_COUNT} fixtures, found ${produced}" >&2
  exit 1
fi
echo "    ${produced} fixtures"

verify_f21() {
  local dir="$1"
  (
    cd "${SCRIPT_DIR}"
    JAVA_HOME="${JDK}" PATH="${JDK}/bin:${PATH}" \
      "${MVN}" -o -q compile exec:java \
      -Dexec.args=verify-interop-f21-legacy-delete-merge \
      -Dinterop.f21_legacy_delete_merge.dir="${dir}" 2>&1
  ) || true
}

echo "==> [5/6] Java: production scan + DV layout of the Rust-written result"
out="$(verify_f21 "${TMP}/seed")"
echo "${out}"
if echo "${out}" | grep -q '^FAIL ' || ! echo "${out}" | grep -q 'verify-interop-f21-legacy-delete-merge: 0 failures'; then
  echo "==> FAILED -- Java rejected the Rust-written F21 layout." >&2
  exit 1
fi

echo "==> [6/6] Sabotage: corrupt the expected rows; the oracle must FAIL"
SABOTAGE="${TMP}/sabotage"
rm -rf "${SABOTAGE}"
mkdir -p "${SABOTAGE}"
cp -r "${TMP}/seed/after_delete" "${SABOTAGE}/after_delete"
cp -r "${TMP}/seed/table" "${SABOTAGE}/table"
python3 - "${SABOTAGE}/after_delete/expected_rows.json" <<'PY'
import json, sys
path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
data[0]["id"] = 999
with open(path, "w") as out:
    json.dump(data, out)
PY
sabotage_out="$(verify_f21 "${SABOTAGE}")"
echo "${sabotage_out}"
if ! echo "${sabotage_out}" | grep -q '^FAIL '; then
  echo "==> FAILED -- the sabotaged expected rows did not turn the oracle red." >&2
  exit 1
fi
echo "    sabotage correctly rejected"

echo "==> interop-f21-legacy-delete-merge PASSED"
