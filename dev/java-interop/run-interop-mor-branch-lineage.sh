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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-mor-branch-lineage"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

EXPECTED_JAVA_FIXTURES=1
AFTER_ARTIFACTS=(
  rust_table/metadata/final.metadata.json
  expected_branch_lineage.txt
  expected_branch_files.txt
  updated_id.txt
  first_update_seq.txt
)

if [[ ! -x "${MVN}" ]]; then
  echo "ERROR: missing Maven at ${MVN}" >&2
  exit 1
fi
if [[ ! -x "${JAVA_HOME}/bin/java" ]]; then
  echo "ERROR: missing Java 11 at ${JAVA_HOME}" >&2
  exit 1
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "ERROR: missing cargo on PATH" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: missing python3 on PATH (needed by the sabotage pass)" >&2
  exit 1
fi

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

run_verify() {
  local out
  out="$(
    cd "${SCRIPT_DIR}"
    "${MVN}" -o -q compile exec:java \
      -Dexec.args=verify-interop-mor-branch-lineage \
      -Dinterop.mor_branch_lineage.dir="${TMP}" 2>&1
  )" || true
  echo "${out}"
}

echo "==> [1/6] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/6] Java: generate-interop-mor-branch-lineage (V3 main + diverged branch b)"
run_oracle -Dexec.args=generate-interop-mor-branch-lineage \
  -Dinterop.mor_branch_lineage.dir="${TMP}"
if [[ ! -f "${TMP}/fixture_count.json" ]]; then
  echo "ERROR: Java generate did not write fixture_count.json" >&2
  exit 1
fi
COUNT="$(tr -d '[:space:]' < "${TMP}/fixture_count.json")"
if [[ "${COUNT}" != "{\"count\":${EXPECTED_JAVA_FIXTURES}}" ]]; then
  echo "ERROR: fixture count ${COUNT} != {\"count\":${EXPECTED_JAVA_FIXTURES}}" >&2
  exit 1
fi
java_count="$(find "${TMP}" -path '*/branch_table/metadata/final.metadata.json' | wc -l | tr -d ' ')"
if [[ "${java_count}" != "${EXPECTED_JAVA_FIXTURES}" ]]; then
  echo "ERROR: expected ${EXPECTED_JAVA_FIXTURES} Java fixture, found ${java_count}" >&2
  exit 1
fi
for seed in java_seed_branch_lineage.txt java_seed_main_lineage.txt java_seed_main_files.txt \
  java_seed_branch_files.txt java_seed_main_snapshot_id.txt java_seed_next_row_id.txt; do
  if [[ ! -s "${TMP}/${seed}" ]]; then
    echo "ERROR: Java generate did not write ${seed}" >&2
    exit 1
  fi
done
echo "    Java fixture count=${java_count} OK"

echo "==> [3/6] Rust: D1 read of the Java branch lineage + two MoR UPDATEs on branch b"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_MOR_BRANCH_LINEAGE_DIR="${TMP}" \
    ICEBERG_INTEROP_MOR_BRANCH_LINEAGE_GEN_DIR="${TMP}" \
    cargo test -p iceberg-datafusion --test interop_mor_branch_lineage --locked -- --nocapture
)

echo "==> [4/6] Check the Rust GEN artifacts"
for artifact in "${AFTER_ARTIFACTS[@]}"; do
  if [[ ! -s "${TMP}/rust_after/${artifact}" ]]; then
    echo "ERROR: missing ${TMP}/rust_after/${artifact} after Rust GEN" >&2
    exit 1
  fi
done
echo "    Rust GEN artifacts: ${#AFTER_ARTIFACTS[@]} OK"

echo "==> [5/6] Java: production scan of the Rust-updated branch head"
VERIFY_OUT="$(run_verify)"
echo "${VERIFY_OUT}"
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-mor-branch-lineage: 0 failures'; then
  echo "==> FAILED — Java rejected the Rust MoR UPDATE lineage on branch b." >&2
  exit 1
fi

echo "==> [6/6] SABOTAGE: bend one expected branch lineage value; verify must turn RED"
PINNED="${TMP}/rust_after/expected_branch_lineage.txt"
cp "${PINNED}" "${PINNED}.bak"
python3 - "${PINNED}" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
lines = [line for line in path.read_text().splitlines() if line.strip()]
if not lines:
    raise SystemExit("no lineage lines to mutate")
row_id, sequence = lines[0].split("=")[1:]
mutated = lines[0].rsplit("=", 1)[0] + "=" + str(int(sequence) + 1)
if mutated == lines[0]:
    raise SystemExit("mutation was a no-op")
lines[0] = mutated
path.write_text("\n".join(lines) + "\n")
print(f"mutated last_updated_sequence_number {sequence} -> {int(sequence) + 1} (row_id {row_id} kept)")
PY
SABOTAGE_OUT="$(run_verify)"
cp "${PINNED}.bak" "${PINNED}"
rm -f "${PINNED}.bak"
echo "${SABOTAGE_OUT}"
if ! echo "${SABOTAGE_OUT}" | grep -q '^FAIL mor-branch-lineage/branch_lineage'; then
  echo "ERROR: sabotage did not FAIL on branch_lineage — the pin is vacuous" >&2
  exit 1
fi
echo "    sabotage RED OK (1 red out of 1 mutation)"

echo "==> interop-mor-branch-lineage PASSED (${EXPECTED_JAVA_FIXTURES} Java fixture, 2 MoR UPDATEs on branch b, sabotage RED)"
