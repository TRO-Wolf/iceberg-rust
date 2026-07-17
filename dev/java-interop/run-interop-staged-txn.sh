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
# R158 STAGED CREATE/REPLACE TABLE TRANSACTION interop harness — proving the Rust
# `StagedTableTransaction::{begin_create,begin_replace,add_data_files,commit}` is bidirectionally
# 1:1 with Java 1.10.0 `Catalog.newCreateTableTransaction` / `newReplaceTableTransaction` (driven
# through the engine-agnostic iceberg-core surface those methods wrap: `Transactions.
# createTableTransaction` / `replaceTableTransaction` over `ops.current().buildReplacement(...)`).
#
# ONE fixture tree ($TMP): Java builds d1 (generate), Rust builds d2 (GEN test). Scenarios:
# create / replace (base + 2 cycles) / fmtv_preserve / fmtv_upgrade.
#
# THE CHAIN:
#   1. Reset the temp dir.
#   2. Java: build d1 (create/replace/fmtv) + per-cycle metadata + canonical views + live rows.
#   3. Rust: build d2 with the SAME scenarios via the production StagedTableTransaction.
#   4. CROSS-CHECK (C-5): Java emits its canonical view of the RUST d2 tables and byte-diffs each
#      against Java's own view of its d1 table — create + replace-r2 must be STRUCTURALLY equivalent
#      (the canonical view erases writer-dependent paths).
#   5. Java: verify-interop-staged-txn — reads the RUST d2 tables and asserts create single-publish +
#      row content, the replace invariant set E-INV(1-7) per cycle, and the format-version directive
#      contract. Verdict = the "verify-interop-staged-txn: 0 failures" sentinel.
#   6. Rust: assert its own view of the JAVA d1 tables (E-INV per cycle + rows + canonical view).
#   7. Sabotage battery (fail-closed, hard-fail-never-skip): SB1 splice a fresh table-uuid, SB2 drop
#      the retained snapshot history, SB3 truncate the base metadata — each must drive Java's verify
#      RED. A sabotage that cannot be applied exits non-zero (never SKIP); the .bak restore stays
#      reachable via an `|| rc=$?` capture of the mutator.
#
# A divergence anywhere — uuid retention, history retention, metadata_log growth, main-ref reset,
# location stability, format-version preservation, last_column_id monotonicity, or the create
# single-publish/row content — fails loudly. TEST-ONLY oracle; the temp dir under
# dev/java-interop/target/ is gitignored.
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64, the
# repo's pinned Rust toolchain. Run from anywhere; paths resolve relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-staged-txn"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

echo "==> [1/7] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/7] Java: build d1 (create/replace/fmtv) — per-cycle metadata + views + rows"
run_oracle -Dexec.args=generate-interop-staged-txn -Dinterop.staged_txn.dir="${TMP}"

echo "==> [3/7] Rust: build d2 with the SAME scenarios via StagedTableTransaction"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_STAGED_TXN_GEN_DIR="${TMP}" \
    cargo test -p iceberg --test interop_staged_txn test_staged_txn_gen_rust_produces_fixtures \
    -- --exact --nocapture
)

echo "==> [4/7] CROSS-CHECK (C-5): Java's view of the RUST table == Java's own view (paths erased)"
crosscheck() {
  # crosscheck <label> <rust-metadata-json> <java-view-json>
  local label="$1" rust_meta="$2" java_view="$3"
  local out="${TMP}/xcheck_${label}.json"
  run_oracle -Dexec.args=emit-snapshot-meta \
    -Dinterop.meta.metadata="${rust_meta}" \
    -Dinterop.meta.out="${out}" >/dev/null
  if ! diff -u "${java_view}" "${out}"; then
    echo "==> FAILED — cross-check ${label}: Java's view of the RUST table diverges from Java's own d1 view."
    exit 1
  fi
  echo "    ${label}: Java view of Rust table == Java view of Java table OK (structurally equivalent)"
}
# create: Java already emitted java_meta.json (its own view of the d1 create table).
crosscheck "create" \
  "${TMP}/d2/create/table/metadata/final.metadata.json" \
  "${TMP}/d1/create/java_meta.json"
# replace-r2: emit Java's view of its OWN d1 r2 first, then diff the Rust d2 r2 view against it.
run_oracle -Dexec.args=emit-snapshot-meta \
  -Dinterop.meta.metadata="${TMP}/d1/replace/table/metadata/r2.final.metadata.json" \
  -Dinterop.meta.out="${TMP}/d1_replace_r2_view.json" >/dev/null
crosscheck "replace_r2" \
  "${TMP}/d2/replace/table/metadata/r2.final.metadata.json" \
  "${TMP}/d1_replace_r2_view.json"

echo "==> [5/7] Java: verify-interop-staged-txn — reads the RUST d2 tables, asserts create + E-INV + fmtv"
run_verify() {
  # Emits the captured output; the verdict is the sentinel, not mvn's exit (exec:java does not
  # reliably propagate System.exit). `|| true` keeps set -e from aborting before the grep.
  (
    cd "${SCRIPT_DIR}"
    "${MVN}" -o -q compile exec:java \
      -Dexec.args=verify-interop-staged-txn \
      -Dinterop.staged_txn.dir="${TMP}" 2>&1
  ) || true
}
STAGED_TXN_VERIFY_OUT="$(run_verify)"
echo "${STAGED_TXN_VERIFY_OUT}"
if echo "${STAGED_TXN_VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${STAGED_TXN_VERIFY_OUT}" | grep -q 'verify-interop-staged-txn: 0 failures'; then
  echo "==> FAILED — Java rejected the RUST-produced staged create/replace tables."
  exit 1
fi

echo "==> [6/7] Rust: assert ITS view of the JAVA d1 tables (E-INV per cycle + rows + canonical view)"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_STAGED_TXN_DIR="${TMP}" \
    cargo test -p iceberg --test interop_staged_txn test_staged_txn_rust_verifies_java \
    -- --exact --nocapture
)

echo "==> [7/7] Sabotage battery (fail-closed, hard-fail-never-skip) — each MUST drive Java verify RED"

# A sabotage step must observe the verifier going RED. HARD-FAIL (never SKIP) if the corruption
# cannot be applied: capture the mutator's exit with `|| rc=$?` so the .bak restore stays reachable
# under `set -euo pipefail`, restore, then exit non-zero (a false-green door otherwise).
assert_verify_red() {
  # assert_verify_red <label> <bak-file-to-restore>
  local label="$1" bak="$2"
  local out
  out="$(run_verify)"
  if echo "${out}" | grep -q 'verify-interop-staged-txn: 0 failures'; then
    echo "==> SABOTAGE ${label} FAILED: corruption did NOT break Java's verify (still 0 failures)."
    cp "${bak}" "${bak%.bak}"
    exit 1
  fi
  echo "    ${label} PASS: Java verify went RED as expected:"
  echo "${out}" | grep -E '^FAIL |verify-interop-staged-txn: [0-9]+ failures' | sed 's/^/        /'
}

# ---- SB1: splice a FRESH table-uuid into the Rust d2 replace-r2 metadata → E-INV#1 (uuid retained)
# must fail. HARD-FAIL if there is no table-uuid to mutate.
echo "    SB1: splice a fresh table-uuid into d2/replace/r2 → uuid-retention check must FAIL"
SB1_FILE="${TMP}/d2/replace/table/metadata/r2.final.metadata.json"
cp "${SB1_FILE}" "${SB1_FILE}.bak"
rc=0
python3 -c "
import json, sys, uuid
p = sys.argv[1]
with open(p) as f:
    meta = json.load(f)
if 'table-uuid' not in meta:
    print('SB1: no table-uuid field to mutate', file=sys.stderr); sys.exit(3)
old = meta['table-uuid']
meta['table-uuid'] = str(uuid.uuid4())
if meta['table-uuid'] == old:
    print('SB1: fresh uuid collided with the original', file=sys.stderr); sys.exit(3)
with open(p, 'w') as f:
    json.dump(meta, f)
print(f'SB1: table-uuid {old} -> {meta[\"table-uuid\"]}')
" "${SB1_FILE}" || rc=$?
if [[ "${rc}" -ne 0 ]]; then
  echo "==> SABOTAGE SB1 could not be applied (rc=${rc}) — HARD-FAIL, never skip."
  cp "${SB1_FILE}.bak" "${SB1_FILE}"
  exit 1
fi
assert_verify_red "SB1(uuid-splice)" "${SB1_FILE}.bak"
cp "${SB1_FILE}.bak" "${SB1_FILE}"
echo "    SB1: restored"

# ---- SB2: DROP the retained pre-replace snapshots (base seq 1 + 2) from d2/replace/r2 → E-INV#2
# (history retained) must fail. HARD-FAIL if those snapshots are not present to drop.
echo "    SB2: drop the retained base snapshots from d2/replace/r2 → history-retention check must FAIL"
SB2_FILE="${TMP}/d2/replace/table/metadata/r2.final.metadata.json"
cp "${SB2_FILE}" "${SB2_FILE}.bak"
rc=0
python3 -c "
import json, sys
p = sys.argv[1]
with open(p) as f:
    meta = json.load(f)
snaps = meta.get('snapshots', [])
before = len(snaps)
# The base snapshots are the two lowest sequence numbers (S1=1, S2=2); r1/r2 are 3/4.
kept = [s for s in snaps if s.get('sequence-number', 0) > 2]
if len(kept) >= before or not any(s.get('sequence-number', 0) <= 2 for s in snaps):
    print('SB2: no retained base snapshots (seq<=2) to drop', file=sys.stderr); sys.exit(3)
meta['snapshots'] = kept
with open(p, 'w') as f:
    json.dump(meta, f)
print(f'SB2: dropped base snapshots, {before} -> {len(kept)}')
" "${SB2_FILE}" || rc=$?
if [[ "${rc}" -ne 0 ]]; then
  echo "==> SABOTAGE SB2 could not be applied (rc=${rc}) — HARD-FAIL, never skip."
  cp "${SB2_FILE}.bak" "${SB2_FILE}"
  exit 1
fi
assert_verify_red "SB2(drop-history)" "${SB2_FILE}.bak"
cp "${SB2_FILE}.bak" "${SB2_FILE}"
echo "    SB2: restored"

# ---- SB3: STRUCTURAL truncate the d2/replace base metadata → Java's parse fails closed (RED).
echo "    SB3: truncate d2/replace base metadata → parse must FAIL closed"
SB3_FILE="${TMP}/d2/replace/table/metadata/base.final.metadata.json"
cp "${SB3_FILE}" "${SB3_FILE}.bak"
rc=0
python3 -c "
import os, sys
p = sys.argv[1]
if os.path.getsize(p) <= 10:
    print('SB3: file already <=10 bytes, cannot prove truncation', file=sys.stderr); sys.exit(3)
with open(p, 'r+b') as f:
    f.truncate(10)
print('SB3: truncated base metadata to 10 bytes')
" "${SB3_FILE}" || rc=$?
if [[ "${rc}" -ne 0 ]]; then
  echo "==> SABOTAGE SB3 could not be applied (rc=${rc}) — HARD-FAIL, never skip."
  cp "${SB3_FILE}.bak" "${SB3_FILE}"
  exit 1
fi
assert_verify_red "SB3(truncate-base)" "${SB3_FILE}.bak"
cp "${SB3_FILE}.bak" "${SB3_FILE}"
echo "    SB3: restored"

echo ""
echo "==> DONE — R158 staged create/replace interop passed BOTH directions:"
echo "    D1 (Java acts, Rust verifies): create single-publish + rows; replace E-INV(1-7) per cycle;"
echo "        format-version directive (V1 preserve / V2 upgrade); canonical view reproduced."
echo "    D2 (Rust acts, Java verifies): same asserted over the Rust-produced tables (0 failures)."
echo "    CROSS-CHECK (C-5): Java's view of the Rust create + replace-r2 tables == Java's own view."
echo "    Sabotage battery: SB1 uuid-splice / SB2 drop-history / SB3 truncate all failed closed."
