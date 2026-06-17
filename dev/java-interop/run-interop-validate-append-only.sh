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
# MAINTENANCE ReplacePartitions.validateAppendOnly interop harness (GAP_MATRIX row 144) — a
# BIDIRECTIONAL behavior-equivalence battery. validateAppendOnly() is a pure engine-agnostic
# iceberg-core API (NOT a Spark-surface class), so unlike the eq/pos conversion oracles Java CAN
# DRIVE the real action and observe its commit decision directly. But the guard, WHEN IT FIRES,
# throws and writes nothing — there is no on-disk artifact to round-trip. So the 1:1 evidence is
# "Java THROWS <=> Rust REJECTS under IDENTICAL table+commit shapes". Each side runs a self-contained
# battery of the SAME four cases and asserts each case's THROW/COMMIT outcome against the SAME
# HAND-DECLARED expectation (anti-circular: neither side derives the other's expectation).
#
# The Java 1.10.0 contract (decoded from the iceberg-core jar via `javap -c`):
#   BaseReplacePartitions.validateAppendOnly() -> MergingSnapshotProducer.failAnyDelete() ->
#   ManifestFilterManager.failAnyDelete = true. During filterManifestWithDeletedFiles, the moment a
#   live entry would be dropped while failAnyDelete is set, it throws
#   `new ManifestFilterManager$DeleteException(spec.partitionToPath(file.partition()))`.
#   DeleteException extends org.apache.iceberg.exceptions.ValidationException extends RuntimeException
#   (NOT CommitFailedException) — i.e. NON-retryable. Rust mirrors with a non-retryable DataInvalid.
#
# The four cases (hand-declared IDENTICALLY in ValidateAppendOnlyOracle + interop_validate_append_only.rs):
#   case                        | table                    | replace add | flag | expected
#   matching_partition          | partitioned, cat=a live  | cat=a       | YES  | THROW
#   empty_new_partition         | partitioned, cat=a live  | cat=b       | YES  | COMMIT
#   unpartitioned_full_replace  | unpartitioned, non-empty | (full)      | YES  | THROW
#   matching_partition_no_flag  | partitioned, cat=a live  | cat=a       | NO   | COMMIT
#
# This is a TEST-ONLY ORACLE (a dev tool) — NOT part of the shipped Rust library, NOT part of the
# offline `cargo test` gate (it needs Java + Maven). The committed artifacts are the oracle code
# (ValidateAppendOnlyOracle in InteropOracle.java), the Rust mirror (interop_validate_append_only.rs),
# and this script. The temp tables under dev/java-interop/target/ are gitignored.
#
# Methodology (D1 Java drives the real API -> D2 Rust mirror -> fail-closed sabotage battery):
#   1. D1: mvn ... -Dexec.args=verify-interop-validate-append-only -Dinterop.validate_append_only.dir="$TMP"
#        -> Java builds each case's table itself and runs the REAL
#           newReplacePartitions()[.validateAppendOnly()].commit(), asserting THROW vs COMMIT against
#           the hand-declared expectation; the `0 failures` sentinel must appear.
#   2. D2: ICEBERG_INTEROP_VALIDATE_APPEND_ONLY_DIR="$TMP" cargo test ... test_validate_append_only_mirror
#        -> Rust builds the equivalent tables and exercises validate_append_only(), asserting the
#           SAME THROW/COMMIT outcomes against the SAME expectations (a non-retryable DataInvalid is REJECT).
#   3. Sabotage battery (fail-closed, control-gated; HARD-FAIL never SKIP if a mutation cannot be
#      applied). Each runs on a `.bak`-protected SOURCE copy and is RESTORED afterwards (the restore
#      stays reachable because each run's exit is captured with `|| rc=$?`):
#        (a) Java side — flip the `matching_partition` expectation from THROW to COMMIT and re-run D1:
#            the harness MUST exit non-zero (Java still THROWS, so the mismatch is caught). A pass would
#            prove the Java assertions are vacuous.
#        (b) Rust side — flip the `matching_partition` expectation from Throw to Commit and re-run D2:
#            the test MUST fail (Rust still REJECTS). A pass would prove the Rust assertions are vacuous.
#
# Without ICEBERG_INTEROP_VALIDATE_APPEND_ONLY_DIR the Rust mirror is a clean no-op (it stays green in
# the offline gate); this script is what flips it into the REAL comparison.
#
# Requirements:
#   - Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64.
#   - A Rust toolchain (the repo's pinned nightly via rust-toolchain.toml).
#   - No new pom deps (the SAME iceberg-core/data/parquet 1.10.0 the conflict oracles already pull).
#     If ~/.m2 is populated, `mvn -o` runs fully offline.
#
# Run from anywhere; paths are resolved relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-validate-append-only"
ORACLE_SRC="${SCRIPT_DIR}/src/main/java/org/apache/iceberg/InteropOracle.java"
RUST_SRC="${REPO_ROOT}/crates/iceberg/tests/interop_validate_append_only.rs"

# Run the Java D1 oracle against a given (FRESH) fixtures dir. Java builds the case tables under it
# from scratch, so each run needs an empty dir (the real action errors on pre-existing v0.metadata).
run_oracle() {
  local dir="$1"
  (
    cd "${SCRIPT_DIR}"
    JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
      PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
      /opt/maven/bin/mvn -o -q compile exec:java \
      -Dexec.args=verify-interop-validate-append-only \
      -Dinterop.validate_append_only.dir="${dir}"
  )
}

run_rust_mirror() {
  (
    cd "${REPO_ROOT}"
    ICEBERG_INTEROP_VALIDATE_APPEND_ONLY_DIR="${TMP}" \
      cargo test -p iceberg --test interop_validate_append_only \
      test_validate_append_only_mirror -- --nocapture
  )
}

echo "==> [1/4] Reset the temp table dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"
# Maven's incremental compiler is mtime-based. The sabotage block mutates+restores the oracle source
# via `mv`, which can leave the restored file OLDER than its already-compiled .class — so `mvn
# compile` would silently keep a stale (possibly-sabotaged) class. `touch` the source so the clean
# D1 below always recompiles from the on-disk (restored) source. The sabotage restores do the same.
# Cargo has the same mtime-skip behavior, so the Rust mirror source is touched too (a prior
# interrupted sabotage could otherwise leave a stale, sabotaged test binary in target/).
touch "${ORACLE_SRC}" "${RUST_SRC}"

echo "==> [2/4] D1 (Java drives the REAL ReplacePartitions.validateAppendOnly).commit() over all 4 cases"
D1_OUT="$(run_oracle "${TMP}")" || true
echo "${D1_OUT}"
# Fail-closed two ways: a per-case `^FAIL ` line OR absence of the `0 failures` sentinel.
if echo "${D1_OUT}" | grep -q '^FAIL ' \
  || ! echo "${D1_OUT}" | grep -q 'verify-interop-validate-append-only: 0 failures'; then
  echo "==> FAILED — D1 emitted a FAIL line or did not emit the '0 failures' sentinel."
  exit 1
fi

echo "==> [3/4] D2 (Rust mirror) — validate_append_only over the SAME 4 cases vs the SAME expectations"
D2_OUT="$(run_rust_mirror)" || true
echo "${D2_OUT}"
# Fail-closed: require the `1 passed; 0 failed` result and reject any test-failure / panic marker.
# (Match `test result: FAILED`/`panicked at`, NOT the benign `0 failed` token in a passing summary.)
if ! echo "${D2_OUT}" | grep -q 'test result: ok. 1 passed; 0 failed' \
  || echo "${D2_OUT}" | grep -qE 'test result: FAILED|panicked at|FAILED$'; then
  echo "==> FAILED — the Rust mirror did not pass cleanly."
  exit 1
fi

echo "==> [4/4] Sabotage battery — the harness must FAIL closed when a hand-declared expectation is wrong"
# CONTROL: steps 2 + 3 already passed, so a sabotage-fail is meaningful. Each sabotage mutates a
# `.bak`-protected SOURCE copy and restores it; HARD-FAIL (never SKIP) if a mutation cannot be applied
# — a sabotage that did not actually change anything proves nothing.

# (a) JAVA side — flip the `matching_partition` expectation from THROW to COMMIT. Java still THROWS,
#     so D1 must now MISMATCH and the harness must exit non-zero. Proves the Java THROW assertion is
#     non-vacuous (a wrong expectation is caught, not silently accepted).
JAVA_NEEDLE='new Case("matching_partition", true, true, "a", Expected.THROW)'
JAVA_REPLACEMENT='new Case("matching_partition", true, true, "a", Expected.COMMIT)'
if ! grep -qF "${JAVA_NEEDLE}" "${ORACLE_SRC}"; then
  echo "==> FAILED — cannot apply Java sabotage: the matching_partition THROW case literal was not"
  echo "    found in ${ORACLE_SRC} (did the oracle change?). HARD-FAIL, never SKIP."
  exit 1
fi
cp "${ORACLE_SRC}" "${ORACLE_SRC}.bak"
SAB_JAVA_DIR="${TMP}/sabotage_java"
# Restore AND `touch` so Maven recompiles the clean source on the next build (mtime-based skip guard).
restore_java() { mv -f "${ORACLE_SRC}.bak" "${ORACLE_SRC}" && touch "${ORACLE_SRC}"; }
trap restore_java EXIT
# Use a sed with a fixed-string-ish pattern; the literal has no regex metachars beyond '(' ')' '"'.
sed -i "s/${JAVA_NEEDLE//\//\\/}/${JAVA_REPLACEMENT//\//\\/}/" "${ORACLE_SRC}"
if ! grep -qF "${JAVA_REPLACEMENT}" "${ORACLE_SRC}"; then
  echo "==> FAILED — Java sabotage mutation did not take effect."
  exit 1
fi
# Run against a FRESH dir so Java genuinely re-builds the tables and runs the REAL action — otherwise
# a pre-existing v0.metadata.json would error structurally and MASK the semantic THROW check.
rm -rf "${SAB_JAVA_DIR}"
mkdir -p "${SAB_JAVA_DIR}"
rc=0
SAB_JAVA_OUT="$(run_oracle "${SAB_JAVA_DIR}")" || rc=$?
restore_java
trap - EXIT
echo "${SAB_JAVA_OUT}"
if echo "${SAB_JAVA_OUT}" | grep -q 'verify-interop-validate-append-only: 0 failures'; then
  echo "FAIL sabotage(java): D1 reported '0 failures' with a deliberately-wrong matching_partition"
  echo "  expectation (COMMIT vs the real THROW) — the Java THROW assertion is VACUOUS."
  exit 1
fi
# Require the GENUINE semantic mismatch line ("outcome THROW but expected COMMIT") for the
# matching_partition case — NOT a structural/unexpected-error line. This proves the flipped
# expectation (not some incidental failure) is what tripped the assertion.
if ! echo "${SAB_JAVA_OUT}" \
  | grep -qE 'FAIL validate-append-only\[matching_partition\]: outcome THROW but expected COMMIT'; then
  echo "FAIL sabotage(java): D1 failed but NOT via the matching_partition THROW-vs-COMMIT mismatch"
  echo "  line — the sabotage did not actually exercise the THROW assertion it was meant to."
  exit 1
fi
echo "PASS sabotage(java): a wrong THROW expectation is caught (matching_partition outcome-THROW-but-expected-COMMIT line); Java assertion is non-vacuous"

# (b) RUST side — flip the `matching_partition` expectation from Throw to Commit. Rust still REJECTS,
#     so the mirror test must now FAIL. Proves the Rust REJECT assertion is non-vacuous.
RUST_NEEDLE='        expected: Expected::Throw,
        },
        // (B)'
if ! grep -qF 'expected: Expected::Throw,' "${RUST_SRC}"; then
  echo "==> FAILED — cannot apply Rust sabotage: the matching_partition Throw expectation literal was"
  echo "    not found in ${RUST_SRC}. HARD-FAIL, never SKIP."
  exit 1
fi
cp "${RUST_SRC}" "${RUST_SRC}.bak"
# Restore AND `touch` so cargo recompiles the clean test on the next build (mtime-based skip guard).
restore_rust() { mv -f "${RUST_SRC}.bak" "${RUST_SRC}" && touch "${RUST_SRC}"; }
trap restore_rust EXIT
# Flip ONLY the FIRST Throw (the matching_partition case) to Commit using a one-shot perl edit.
perl -0pi -e 's/(name: "matching_partition",.*?expected: Expected::)Throw/$1Commit/s' "${RUST_SRC}"
if ! grep -q 'name: "matching_partition"' "${RUST_SRC}" \
  || ! perl -0ne 'exit(/name: "matching_partition",.*?expected: Expected::Commit/s ? 0 : 1)' "${RUST_SRC}"; then
  echo "==> FAILED — Rust sabotage mutation did not take effect (matching_partition still Throw)."
  exit 1
fi
rc=0
SAB_RUST_OUT="$(run_rust_mirror)" || rc=$?
restore_rust
trap - EXIT
echo "${SAB_RUST_OUT}"
if [[ "${rc}" -eq 0 ]] && echo "${SAB_RUST_OUT}" | grep -q 'test result: ok'; then
  echo "FAIL sabotage(rust): the mirror PASSED with a deliberately-wrong matching_partition"
  echo "  expectation (Commit vs the real Reject) — the Rust REJECT assertion is VACUOUS."
  exit 1
fi
echo "PASS sabotage(rust): a wrong Reject expectation makes the mirror fail; Rust assertion is non-vacuous"

echo "==> DONE — ReplacePartitions.validateAppendOnly interop passed."
echo "    D1: Java's real validateAppendOnly THROWS exactly on {matching_partition,"
echo "    unpartitioned_full_replace} and COMMITS on {empty_new_partition, matching_partition_no_flag}."
echo "    D2: Rust's validate_append_only REJECTS/COMMITS identically (non-retryable DataInvalid on reject)."
echo "    Sabotage battery: a wrong THROW expectation (Java) and a wrong Reject expectation (Rust) each"
echo "    fail closed; both controls passed in steps 2 + 3."