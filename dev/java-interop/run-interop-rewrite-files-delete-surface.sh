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
# RewriteFiles DELETE-file ADD surface — BIDIRECTIONAL METADATA-LEVEL interop harness
# (GAP_MATRIX row 152).
#
# Proves the two Java seq overloads for ADDED delete files in a RewriteFiles commit:
#   Java addFile(DeleteFile)        → inherited seq (== new snapshot's data seq)
#   Java addFile(DeleteFile, long)  → explicit stamp (preserved exactly)
#
# THE LOCKED FIXTURE (V2 unpartitioned):
#   S1 newAppend(D)              → data file D,    seq 1
#   S2 newRowDelta(DF0)          → seed pos-del,   seq 2
#   S3 newRewrite: deleteFile(DF0) + addFile(DF_inherited) + addFile(DF_exp, 2L) → seq 3
#      DF_inherited expected seq = 3   (inherited from S3)
#      DF_exp  expected seq = 2  (explicit stamp)
#
# THE CHAIN:
#   1. Reset the temp dir.
#   2. JAVA generate-interop-rfds: build the locked fixture under <tmp>/table; emit
#      final.metadata.json + java_delete_seqs.json (the two added-delete paths + seqs).
#      Java SELF-CHECKS its own seqs; exits non-zero on mismatch.
#   3. RUST GEN (cargo test, env ICEBERG_INTEROP_RFDS_DIR):
#      DIRECTION 1 — load <tmp>/table (Java-written); assert added-delete seqs match
#        java_delete_seqs.json AND the hand-declared expected.
#      DIRECTION 2 — independently build the SAME locked fixture in Rust; assert seqs;
#        write <tmp>/rust_rewritten/metadata/final.metadata.json for Java verify.
#   4. NON-VACUITY GATE: corrupt java_delete_seqs.json (change inherited seq 3 → 99),
#      re-run the Rust test; CONFIRM it FAILS (exit non-zero). Restore via a fresh Java
#      generate. A sabotage step that cannot be applied HARD-FAILS (exits non-zero).
#   5. JAVA verify-interop-rfds: load <tmp>/rust_rewritten/metadata/final.metadata.json;
#      assert the two added-delete seqs in the Rust-written table match the expected
#      (inherited == Rust S3 snapshot seq, explicit == 2). Prints the sentinel.
#
# FAIL-CLOSED: the verify outcome is read from the printed sentinel (not from mvn exit,
# which is unreliable for exec:java + System.exit). The script greps both "0 failures"
# AND the absence of any FAIL line, mirroring run-interop-expire.sh discipline.
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64,
# the repo's pinned Rust toolchain.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-rfds"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

echo "==> [1/5] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/5] Java: generate the locked fixture (S1 append, S2 rowdelta-DF0, S3 rewrite +DF_inherited +DF_exp)"
run_oracle -Dexec.args=generate-interop-rfds \
  -Dinterop.rfds.dir="${TMP}"

echo "==> [3/5] Rust: DIRECTION 1 (read Java table, assert seqs) + DIRECTION 2 (write Rust table)"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_RFDS_DIR="${TMP}" \
    cargo test -p iceberg --test interop_rewrite_files_delete_surface \
      test_rewrite_files_delete_surface_interop \
    -- --exact --nocapture
)

echo "==> [4/5] Non-vacuity gate: corrupt java_delete_seqs.json (inherited seq 3 → 99), re-run Rust D1 (MUST FAIL)"
#
# The inherited seq in java_delete_seqs.json is 3 (the S3 snapshot seq). We replace "seq":3
# inside the "inherited" block with "seq":99. The Rust Direction-1 asserts that the manifest
# entry seq equals java_delete_seqs.json's inherited.seq, so a corrupted "99" must fail.
#
SEQS_FILE="${TMP}/java_delete_seqs.json"
SEQS_BAK="${TMP}/java_delete_seqs.json.bak"
cp "${SEQS_FILE}" "${SEQS_BAK}"

# Apply the sabotage: change the inherited seq from 3 to 99.
# Pass the path as an argument so python3 receives the expanded value.
python3 -c "
import json, sys
path = sys.argv[1]
with open(path) as f:
    d = json.load(f)
original = d['inherited']['seq']
if original != 3:
    print('SABOTAGE-ABORT: inherited seq = ' + str(original) + ', expected 3; cannot apply corruption', file=sys.stderr)
    sys.exit(1)
d['inherited']['seq'] = 99
with open(path, 'w') as f:
    json.dump(d, f)
print('SABOTAGE: changed inherited seq ' + str(original) + ' -> 99')
" "${SEQS_FILE}"

SABOTAGE_OK=0
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_RFDS_DIR="${TMP}" \
    cargo test -p iceberg --test interop_rewrite_files_delete_surface \
      test_rewrite_files_delete_surface_interop \
    -- --exact --nocapture 2>&1
) && SABOTAGE_OK=1 || true

# Restore the backup before checking SABOTAGE_OK (restore stays reachable even if the test
# returned zero — an unexpected pass is the error, not a hard exit before restore).
cp "${SEQS_BAK}" "${SEQS_FILE}"
echo "SABOTAGE: restored java_delete_seqs.json from backup"

if [ "${SABOTAGE_OK}" -eq 1 ]; then
  echo "==> FAILED — non-vacuity gate: the Rust test DID NOT FAIL when java_delete_seqs.json"
  echo "    was corrupted (inherited seq 3 → 99). The assertion is vacuous — fix the test."
  exit 1
fi
echo "==> NON-VACUITY CONFIRMED — Rust Direction-1 correctly rejected the corrupted seq."

echo "==> [5/5] Java: verify the Rust-written table (rust_rewritten/metadata/final.metadata.json)"
RFDS_VERIFY_OUT="$(
  cd "${SCRIPT_DIR}"
  "${MVN}" -o -q compile exec:java \
    -Dexec.args=verify-interop-rfds \
    -Dinterop.rfds.dir="${TMP}" 2>&1
)" || true
echo "${RFDS_VERIFY_OUT}"
if echo "${RFDS_VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${RFDS_VERIFY_OUT}" | grep -q 'verify-interop-rfds: 0 failures'; then
  echo "==> FAILED — Java rejected the Rust-written table (added-delete seqs diverged)."
  exit 1
fi

echo "==> DONE — RewriteFiles DELETE-file ADD surface BIDIRECTIONAL interop passed:"
echo "    D1: Rust read the Java-written table and confirmed both added-delete seqs"
echo "        (inherited == Java S3 snapshot seq; explicit == 2)."
echo "    D2: Java read the Rust-written table and confirmed both added-delete seqs"
echo "        (inherited == Rust S3 snapshot seq; explicit == 2)."
echo "    NON-VACUITY: corrupted seq 3→99 was correctly rejected by Rust Direction-1."
