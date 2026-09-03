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
FIXTURE_COUNT=2

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

echo "==> [2b] Java: partition-scoped seed (two data files, one partition delete)"
(
  cd "${SCRIPT_DIR}"
  JAVA_HOME="${JDK}" PATH="${JDK}/bin:${PATH}" \
    "${MVN}" -o -q compile exec:java \
    -Dexec.args=generate-interop-f21-legacy-delete-merge-part \
    -Dinterop.f21_legacy_delete_merge.dir="${TMP}/seed"
)
if [ ! -d "${TMP}/seed/part_table/metadata" ]; then
  echo "==> FAILED -- the Java partition seed produced no table" >&2
  exit 1
fi

echo "==> [3/6] Rust: upgrades to V3 and DELETE id=3, merging the parquet delete"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_F21_JAVA_SHARED="${TMP}/seed" \
    cargo test -p iceberg-datafusion --locked --test interop_f21_legacy_delete_merge -- --nocapture
)

echo "==> [4/6] Fixture count"
produced_file="$(find "${TMP}/seed/after_delete" -maxdepth 1 -type f | wc -l)"
produced_part="$(find "${TMP}/seed/after_part" -maxdepth 1 -type f | wc -l)"
produced=$((produced_file + produced_part))
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

verify_f21_part() {
  local dir="$1"
  (
    cd "${SCRIPT_DIR}"
    JAVA_HOME="${JDK}" PATH="${JDK}/bin:${PATH}" \
      "${MVN}" -o -q compile exec:java \
      -Dexec.args=verify-interop-f21-legacy-delete-merge-part \
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
echo "    row sabotage correctly rejected"

echo "==> [6b] Sabotage: added-dvs 1 -> 99; the oracle must FAIL"
SABOTAGE2="${TMP}/sabotage-dvs"
rm -rf "${SABOTAGE2}"
mkdir -p "${SABOTAGE2}"
cp -r "${TMP}/seed/after_delete" "${SABOTAGE2}/after_delete"
cp -r "${TMP}/seed/table" "${SABOTAGE2}/table"
META="${SABOTAGE2}/after_delete/rust_table/metadata/final.metadata.json"
mut_rc=0
python3 - "${META}" <<'PY' || mut_rc=$?
import json, sys
path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
current = data.get("current-snapshot-id")
snaps = data.get("snapshots") or []
target = None
for snap in snaps:
    if snap.get("snapshot-id") == current:
        target = snap
        break
if target is None:
    sys.exit(2)
summary = target.get("summary") or {}
if "added-dvs" not in summary:
    sys.exit(2)
summary["added-dvs"] = "99"
target["summary"] = summary
with open(path, "w") as out:
    json.dump(data, out)
PY
if [ "${mut_rc}" -ne 0 ]; then
  echo "==> FAILED -- added-dvs sabotage could not be applied (exit ${mut_rc})" >&2
  exit 1
fi
sabotage2_out="$(verify_f21 "${SABOTAGE2}")"
echo "${sabotage2_out}"
if ! echo "${sabotage2_out}" | grep -q '^FAIL '; then
  echo "==> FAILED -- the sabotaged added-dvs did not turn the oracle red." >&2
  exit 1
fi
echo "    added-dvs sabotage correctly rejected"

echo "==> [7] Java: partition-scoped coexistence"
part_out="$(verify_f21_part "${TMP}/seed")"
echo "${part_out}"
if echo "${part_out}" | grep -q '^FAIL ' || ! echo "${part_out}" | grep -q 'verify-interop-f21-legacy-delete-merge-part: 0 failures'; then
  echo "==> FAILED -- Java rejected the partition-scoped F21 layout." >&2
  exit 1
fi

echo "==> [8] Sabotage: partition expected rows; the oracle must FAIL"
SABOTAGE3="${TMP}/sabotage-part"
rm -rf "${SABOTAGE3}"
mkdir -p "${SABOTAGE3}"
cp -r "${TMP}/seed/after_part" "${SABOTAGE3}/after_part"
cp -r "${TMP}/seed/part_table" "${SABOTAGE3}/part_table"
python3 - "${SABOTAGE3}/after_part/expected_part_rows.json" <<'PY'
import json, sys
path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
data[0]["id"] = 999
with open(path, "w") as out:
    json.dump(data, out)
PY
sabotage3_out="$(verify_f21_part "${SABOTAGE3}")"
echo "${sabotage3_out}"
if ! echo "${sabotage3_out}" | grep -q '^FAIL '; then
  echo "==> FAILED -- the sabotaged partition rows did not turn the oracle red." >&2
  exit 1
fi
echo "    partition row sabotage correctly rejected"

echo "==> interop-f21-legacy-delete-merge PASSED"
