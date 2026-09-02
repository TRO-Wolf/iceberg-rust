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
TMP="${SCRIPT_DIR}/target/interop-branch-dml"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

EXPECTED_JAVA_FIXTURES=4
RUST_TABLES=(rust_append rust_cow rust_mor rust_created rust_insert_create rust_retry)

if [[ ! -x "${MVN}" ]]; then
  echo "ERROR: missing Maven at ${MVN}" >&2
  exit 1
fi
if [[ ! -x "${JAVA_HOME}/bin/java" ]]; then
  echo "ERROR: missing Java 11 at ${JAVA_HOME}" >&2
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
      -Dexec.args=verify-interop-branch-dml \
      -Dinterop.branch.dir="${TMP}" 2>&1
  )" || true
  echo "${out}"
}

echo "==> [1/6] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/6] Java: generate-interop-branch-dml (diverged / v3_diverged / tag / no_branch)"
run_oracle -Dexec.args=generate-interop-branch-dml -Dinterop.branch.dir="${TMP}"
java_count="$(find "${TMP}" -path '*/table/metadata/final.metadata.json' | wc -l | tr -d ' ')"
if [[ "${java_count}" != "${EXPECTED_JAVA_FIXTURES}" ]]; then
  echo "ERROR: expected ${EXPECTED_JAVA_FIXTURES} Java fixtures, found ${java_count}" >&2
  find "${TMP}" -path '*/table/metadata/final.metadata.json' >&2 || true
  exit 1
fi
echo "    Java fixture count=${java_count} OK"

echo "==> [3/6] Rust: Direction-1 read of Java tables + GEN rust_* tables"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_BRANCH_DIR="${TMP}" \
    ICEBERG_INTEROP_BRANCH_GEN_DIR="${TMP}" \
    cargo test -p iceberg-datafusion --test interop_branch_dml --locked -- --nocapture
)

echo "==> [4/6] Java: emit-branch-meta of each rust_* table + verify-interop-branch-dml"
for table in "${RUST_TABLES[@]}"; do
  meta="${TMP}/${table}/metadata/final.metadata.json"
  if [[ ! -f "${meta}" ]]; then
    echo "ERROR: missing ${meta} after Rust GEN" >&2
    exit 1
  fi
  if [[ ! -f "${TMP}/${table}/expected_main_files.txt" || ! -f "${TMP}/${table}/expected_branch_files.txt" ]]; then
    echo "ERROR: missing expected file-set pins for ${table}" >&2
    exit 1
  fi
  run_oracle -Dexec.args=emit-branch-meta \
    -Dinterop.meta.metadata="${meta}" \
    -Dinterop.meta.out="${TMP}/${table}/java_view_rust_branch_meta.json" \
    -Dinterop.meta.branch=b
  echo "    ${table}: emit-branch-meta OK"
done
VERIFY_OUT="$(run_verify)"
echo "${VERIFY_OUT}"
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-branch-dml: 0 failures'; then
  echo "==> FAILED — Java rejected the RUST-produced branch tables."
  exit 1
fi

echo "==> [5/6] Rust Direction-1 of Java tables already asserted in step 3 (fixtures untouched)"
for fixture in diverged v3_diverged tag no_branch; do
  if [[ ! -f "${TMP}/${fixture}/table/metadata/final.metadata.json" ]]; then
    echo "ERROR: Java fixture ${fixture} missing after GEN" >&2
    exit 1
  fi
done

echo "==> [6/6] Summary"
echo "    Java fixtures: ${EXPECTED_JAVA_FIXTURES} (diverged, v3_diverged, tag, no_branch)"
echo "    Rust tables: ${#RUST_TABLES[@]} (${RUST_TABLES[*]})"
echo "    D1: Rust reads Java diverged branch (main vs branch file sets + rows)"
echo "    D2: Java IcebergGenerics reads Rust append / COW / MoR / created / insert-create / retry"

echo "==> SABOTAGE A: truncate rust_append final.metadata.json must turn verify RED"
meta="${TMP}/rust_append/metadata/final.metadata.json"
cp "${meta}" "${meta}.bak"
: > "${meta}"
if [[ -s "${meta}" ]]; then
  echo "ERROR: truncation left ${meta} non-empty" >&2
  mv "${meta}.bak" "${meta}"
  exit 1
fi
SABOTAGE_OUT="$(run_verify)" || true
mv "${meta}.bak" "${meta}"
echo "${SABOTAGE_OUT}"
if ! echo "${SABOTAGE_OUT}" | grep -q '^FAIL '; then
  echo "ERROR: sabotage A did not FAIL (vacuous verify)" >&2
  exit 1
fi
echo "    sabotage A RED OK"

echo "==> SABOTAGE B: keep row ids, change rust_created expected branch file set"
expected_branch="${TMP}/rust_created/expected_branch_files.txt"
if [[ ! -s "${expected_branch}" ]]; then
  echo "ERROR: expected_branch_files.txt missing or empty; cannot apply file-set sabotage" >&2
  exit 1
fi
cp "${expected_branch}" "${expected_branch}.bak"
python3 - "${expected_branch}" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
lines = [line for line in path.read_text().splitlines() if line]
if not lines:
    raise SystemExit("no branch file names to mutate")
original = lines[0]
mutated = original + ".rewritten"
if mutated == original:
    raise SystemExit("mutation was a no-op")
lines[0] = mutated
path.write_text("\n".join(lines) + "\n")
print(f"mutated {original} -> {mutated}")
PY
SABOTAGE_B_OUT="$(run_verify)" || true
cp "${expected_branch}.bak" "${expected_branch}"
rm -f "${expected_branch}.bak"
echo "${SABOTAGE_B_OUT}"
if ! echo "${SABOTAGE_B_OUT}" | grep -q 'FAIL branch-dml/rust_created/branch_files'; then
  echo "ERROR: sabotage B did not FAIL on branch_files (ids-only pin would stay green)" >&2
  exit 1
fi
echo "    sabotage B RED OK (file-set pin, row ids unchanged)"

echo "==> DONE — branch-dml interop passed both directions over ${EXPECTED_JAVA_FIXTURES} Java fixtures + ${#RUST_TABLES[@]} Rust tables"
