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
TMP="${SCRIPT_DIR}/target/interop-f18-dv-sibling-close"
MVN=/opt/maven/bin/mvn
JDK=/usr/lib/jvm/java-11-openjdk-amd64
FIXTURE_COUNT=4

for prereq in "${MVN}" "${JDK}/bin/java"; do
  if [ ! -x "${prereq}" ]; then
    echo "==> FAILED -- missing prerequisite ${prereq}" >&2
    exit 1
  fi
done

echo "==> [1/6] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/6] Java: BaseDVFileWriter writes the two-file V3 seed and its two-blob delete"
(
  cd "${SCRIPT_DIR}"
  JAVA_HOME="${JDK}" PATH="${JDK}/bin:${PATH}" \
    "${MVN}" -o -q compile exec:java \
    -Dexec.args=generate-interop-dv-table \
    -Dinterop.dv_table.dir="${TMP}/seed"
)
if [ ! -d "${TMP}/seed/table/metadata" ]; then
  echo "==> FAILED -- the Java seed produced no table" >&2
  exit 1
fi

echo "==> [3/6] Rust: the second DELETE, touching ONE data file"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_F18_JAVA_SHARED="${TMP}/seed" \
    cargo test -p iceberg-datafusion --locked --test interop_f18_dv_sibling_close -- --nocapture
)

echo "==> [4/6] Fixture count"
produced="$(find "${TMP}/seed/after_delete" -maxdepth 1 -type f | wc -l)"
if [ "${produced}" -ne "${FIXTURE_COUNT}" ]; then
  echo "==> FAILED -- expected ${FIXTURE_COUNT} fixtures, found ${produced}" >&2
  exit 1
fi
echo "    ${produced} fixtures"

verify_f18() {
  local dir="$1"
  (
    cd "${SCRIPT_DIR}"
    JAVA_HOME="${JDK}" PATH="${JDK}/bin:${PATH}" \
      "${MVN}" -o -q compile exec:java \
      -Dexec.args=verify-interop-f18-sibling-close \
      -Dinterop.f18.dir="${dir}" 2>&1
  ) || true
}

echo "==> [5/6] Java: production scan + container layout of the Rust-written result"
out="$(verify_f18 "${TMP}/seed")"
echo "${out}"
if echo "${out}" | grep -q '^FAIL ' || ! echo "${out}" | grep -q 'verify-interop-f18-sibling-close: 0 failures'; then
  echo "==> FAILED -- Java rejected the Rust-written sibling-close layout." >&2
  exit 1
fi

echo "==> [6/6] Sabotage: move the sibling entry in the seed record; the oracle must FAIL"
SABOTAGE="${TMP}/sabotage"
rm -rf "${SABOTAGE}"
mkdir -p "${SABOTAGE}"
cp -r "${TMP}/seed/after_delete" "${SABOTAGE}/after_delete"
python3 - "${SABOTAGE}/after_delete/before_dvs.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as handle:
    entries = json.load(handle)
entries[0]["offset"] = entries[0]["offset"] + 1
entries[1]["offset"] = entries[1]["offset"] + 1
with open(path, "w") as handle:
    json.dump(entries, handle)
PY
sabotage_out="$(verify_f18 "${SABOTAGE}")"
echo "${sabotage_out}"
if ! echo "${sabotage_out}" | grep -q '^FAIL '; then
  echo "==> FAILED -- the sabotaged sibling entry did not turn the oracle red." >&2
  exit 1
fi
echo "    sabotage correctly rejected"

echo "==> interop-f18-dv-sibling-close PASSED"
