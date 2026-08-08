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
# RANGED-READ interop — MIDPOINT row-group selection over a byte-range split (U3 / hazard-1, GAP_MATRIX
# row R148's read-side residue). Proves Rust's `ArrowReader::filter_row_groups_by_byte_range` selects the
# SAME row groups as parquet-mr's REAL `ParquetMetadataConverter.filterFileMetaDataByMidpoint`, which
# Iceberg drives via `Parquet.ReadBuilder.split(start, length)` → `withRange(start, start+length)` — the
# exact call `org.apache.iceberg.data.GenericReader.openFile` makes for a FileScanTask. A row group is kept
# iff `getOffset(columns[0]) + totalCompressedSize/2` lies in the HALF-OPEN `[start, start+length)`.
#
# WHY IT MATTERS: an OVERLAP rule hands a row group that straddles a split boundary to BOTH adjacent
# sub-tasks — SILENT DUPLICATE ROWS, never an error. A SYNTHESIZED `4 + Σ compressed_size` offset model
# drifts on any file whose row groups are not contiguous (padding, inline bloom filters) and duplicates
# even for splits aligned to the file's OWN row-group offsets.
#
# ANTI-CIRCULARITY: the windows are HAND-DECLARED on both sides — tile `[0, fileLength)` at STRIDE = 800
# bytes (`InteropOracle.RangedReadOracle.STRIDE` mirrors `interop_ranged_read.rs::STRIDE`). They are NEVER
# derived from either engine's splitter, which would make the comparison circular with respect to the
# split layer.
#
# THE CHAIN:
#   [1/6] Reset the temp dir.
#   [2/6] Java GENERATE (Direction 1): write java_ranged.parquet (tiny row groups ⇒ the 800-byte tiling
#         STRADDLES row groups), read every window through the REAL midpoint filter, emit
#         java_ranged_read.json. Java itself asserts the tiling is a PARTITION of its rows.
#   [3/6] Rust D1: read the SAME file over the SAME windows with the production ArrowReader; assert
#         identical id lists per window + the exactly-once tiling property.
#   [4/6] Rust GEN + Java VERIFY (Direction 2): Rust writes rust_contig.parquet AND rust_padded.parquet
#         (bloom filters ⇒ parquet-rs writes a bloom section AFTER each row group ⇒ real row-group starts
#         run AHEAD of `4 + Σ compressed_size`; the Rust side asserts that drift so the leg is non-vacuous),
#         reads every window, emits rust_ranged_read.json; Java replays the SAME windows through its real
#         filter and asserts equality. The PADDED file is what proves the OFFSET SOURCE, not just the rule.
#   [5/6] SOURCE MUTATION A — the PREDICATE SHAPE (fail-closed): revert the production predicate to the
#         OVERLAP form and re-run the D1 leg; it MUST go RED with a real ASSERTION signal (a non-zero exit
#         alone is NOT sufficient — a mutant that merely failed to COMPILE would otherwise score as a pass).
#         HARD-FAILS if the pattern is absent (an unappliable mutation proves nothing). Restores, `touch`es
#         the file (cargo's mtime staleness check would otherwise reuse the mutant rlib), md5-verifies the
#         restore, re-runs GREEN.
#   [6/6] SOURCE MUTATION B — the OFFSET SOURCE (fail-closed): revert the row-group start from the REAL
#         footer offset to the synthetic `4 + Σ compressed_size` model — the model this unit's doc comments,
#         `scan/map.md` and GAP_MATRIX row R148 all claim is gone from the selection path. Leg [5/6] cannot
#         see it: BOTH Rust-side legs stay GREEN under it (the synthetic model still yields a PARTITION of
#         the rows, so `assert_exactly_once` and the bloom-drift guard both pass), and D1 compares against a
#         Java fixture whose row groups are contiguous. Only the D2 JAVA comparison over the bloom-PADDED
#         file catches it. So this leg re-runs the Rust GEN under the mutant and requires
#         `verify-interop-ranged-read` to FAIL with a real per-window comparison signal
#         (`FAIL ranged-read-d2 <file>.parquet [...`) — the "missing json" / "empty json" FAIL forms are
#         explicitly NOT accepted, since a mutant that failed to compile would produce those. Same
#         HARD-FAIL-if-absent, `|| rc=$?`, md5-verified-restore mechanics, then re-runs GEN + VERIFY GREEN.
#
# This is a TEST-ONLY ORACLE (a dev tool) — NOT part of the shipped Rust library, NOT part of the offline
# `cargo test` gate (it needs Java + Maven). Nothing binary is committed; the temp dir under
# dev/java-interop/target/ is gitignored. Steps [5/6] and [6/6] each edit a tracked source file IN PLACE and
# restore it in the same step; the restore is md5-verified and runs even when the mutant run fails. That window is NOT
# signal-safe (a SIGINT between mutate and restore leaves the mutant on disk — recover with
# `git checkout -- crates/iceberg/src/arrow/reader.rs`) and it assumes no CONCURRENT cargo build in this
# checkout: the nightly driver runs suites SEQUENTIALLY, so do NOT parallelize suites across this stage.
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64, a Rust
# toolchain. No new Maven or Cargo dependencies.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-ranged-read"
TARGET_SRC="${REPO_ROOT}/crates/iceberg/src/arrow/reader.rs"

echo "==> [1/6] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/6] Java oracle GENERATE (D1): write java_ranged.parquet + read every window through the REAL midpoint filter"
(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
    /opt/maven/bin/mvn -o -q compile exec:java \
    -Dexec.args=generate-interop-ranged-read \
    -Dinterop.ranged_read.dir="${TMP}"
)

echo "==> [3/6] Rust D1: read the JAVA-written file over the SAME windows; assert per-window ids + exactly-once tiling"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_RANGED_READ_DIR="${TMP}" \
    cargo test -p iceberg --test interop_ranged_read test_ranged_read_matches_java -- --nocapture
)

echo "==> [4/6] Direction 2: Rust writes contiguous + bloom-PADDED fixtures; Java replays the windows"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_RANGED_READ_GEN_DIR="${TMP}" \
    cargo test -p iceberg --test interop_ranged_read test_ranged_read_gen_rust_fixtures -- --nocapture
)
(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
    /opt/maven/bin/mvn -o -q compile exec:java \
    -Dexec.args=verify-interop-ranged-read \
    -Dinterop.ranged_read.dir="${TMP}"
)

echo "==> [5/6] SOURCE MUTATION A (fail-closed): revert the selection PREDICATE to the OVERLAP form; the D1 leg MUST go RED"
MUT_FROM='if midpoint >= start && midpoint < end {'
MUT_TO='if row_group_start < end && start < row_group_start + row_group_size { let _ = midpoint;'
if ! grep -qF "${MUT_FROM}" "${TARGET_SRC}"; then
  echo "HARD FAIL: the mutation target is absent from ${TARGET_SRC}." >&2
  echo "           A sabotage step that cannot corrupt anything has proven NOTHING." >&2
  echo "           Expected literal: ${MUT_FROM}" >&2
  exit 1
fi
if [ "$(grep -cF "${MUT_FROM}" "${TARGET_SRC}")" != "1" ]; then
  echo "HARD FAIL: the mutation target is ambiguous in ${TARGET_SRC} (expected exactly one occurrence)." >&2
  exit 1
fi

cp "${TARGET_SRC}" "${TMP}/reader.rs.bak"
BEFORE_MD5="$(md5sum "${TARGET_SRC}" | awk '{print $1}')"
python3 - "${TARGET_SRC}" "${MUT_FROM}" "${MUT_TO}" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
assert src.count(old) == 1
open(path, 'w').write(src.replace(old, new))
PY

rc=0
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_RANGED_READ_DIR="${TMP}" \
    cargo test -p iceberg --test interop_ranged_read test_ranged_read_matches_java -- --nocapture
) > "${TMP}/mutant.log" 2>&1 || rc=$?

cp "${TMP}/reader.rs.bak" "${TARGET_SRC}"
touch "${TARGET_SRC}"
AFTER_MD5="$(md5sum "${TARGET_SRC}" | awk '{print $1}')"
if [ "${BEFORE_MD5}" != "${AFTER_MD5}" ]; then
  echo "HARD FAIL: the source restore did not reproduce the original file (${BEFORE_MD5} != ${AFTER_MD5})." >&2
  exit 1
fi

if [ "${rc}" -eq 0 ]; then
  echo "HARD FAIL: the OVERLAP mutation left the D1 leg GREEN — the interop comparison is VACUOUS." >&2
  sed -n '1,60p' "${TMP}/mutant.log" >&2
  exit 1
fi
# A non-zero exit is NOT sufficient: require a real ASSERTION signal, or a mutant that failed to COMPILE
# would score as a pass.
if ! grep -qE "must select exactly the row groups whose MIDPOINT|assertion .left == right. failed|test result: FAILED" "${TMP}/mutant.log"; then
  echo "HARD FAIL: the mutant run failed WITHOUT a test-assertion signal (it probably did not compile)." >&2
  sed -n '1,60p' "${TMP}/mutant.log" >&2
  exit 1
fi
echo "    mutation confirmed RED (assertion signal present); source restored + md5-verified"

echo "==> [5/6] re-run the D1 leg on the RESTORED source — must be GREEN again"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_RANGED_READ_DIR="${TMP}" \
    cargo test -p iceberg --test interop_ranged_read test_ranged_read_matches_java -- --nocapture
)

echo "==> [6/6] SOURCE MUTATION B (fail-closed): revert the OFFSET SOURCE to the synthetic 4 + Σ compressed_size model; the D2 JAVA verify MUST go RED"
# Mutation B replaces the REAL footer offset (`ParquetMetadataConverter.getOffset` over columns[0])
# with Java's degenerate `invalidFileOffset` model — `4 + Σ compressed_size` of the PRECEDING row
# groups — while leaving the `u64::try_from(..).map_err(..)` chain that follows intact. The Rust
# side cannot see this (both Rust legs stay GREEN: the synthetic model still partitions the rows);
# only the Java per-window comparison over the bloom-PADDED file does.
MUT2_FROM='let row_group_start = u64::try_from(Self::parquet_column_chunk_offset(first_column))'
MUT2_TO='let row_group_start = u64::try_from({ let _ = first_column; let mut synthetic: i64 = 4; for previous in row_groups.iter().take(idx) { for chunk in previous.columns() { synthetic += chunk.compressed_size(); } } synthetic })'
if ! grep -qF "${MUT2_FROM}" "${TARGET_SRC}"; then
  echo "HARD FAIL: mutation-B target is absent from ${TARGET_SRC}." >&2
  echo "           A sabotage step that cannot corrupt anything has proven NOTHING." >&2
  echo "           Expected literal: ${MUT2_FROM}" >&2
  exit 1
fi
if [ "$(grep -cF "${MUT2_FROM}" "${TARGET_SRC}")" != "1" ]; then
  echo "HARD FAIL: mutation-B target is ambiguous in ${TARGET_SRC} (expected exactly one occurrence)." >&2
  exit 1
fi

cp "${TARGET_SRC}" "${TMP}/reader.rs.bak2"
BEFORE_MD5_B="$(md5sum "${TARGET_SRC}" | awk '{print $1}')"
python3 - "${TARGET_SRC}" "${MUT2_FROM}" "${MUT2_TO}" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
assert src.count(old) == 1
open(path, 'w').write(src.replace(old, new))
PY

# Re-generate rust_ranged_read.json THROUGH the mutant (this leg stays GREEN — that is the point),
# then let Java replay the same windows against the same files. `|| rc=$?` keeps the restore
# reachable under `set -e`.
gen_rc=0
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_RANGED_READ_GEN_DIR="${TMP}" \
    cargo test -p iceberg --test interop_ranged_read test_ranged_read_gen_rust_fixtures -- --nocapture
) > "${TMP}/mutant_b_gen.log" 2>&1 || gen_rc=$?
verify_rc=0
(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
    /opt/maven/bin/mvn -o -q compile exec:java \
    -Dexec.args=verify-interop-ranged-read \
    -Dinterop.ranged_read.dir="${TMP}"
) > "${TMP}/mutant_b_verify.log" 2>&1 || verify_rc=$?

cp "${TMP}/reader.rs.bak2" "${TARGET_SRC}"
touch "${TARGET_SRC}"
AFTER_MD5_B="$(md5sum "${TARGET_SRC}" | awk '{print $1}')"
if [ "${BEFORE_MD5_B}" != "${AFTER_MD5_B}" ]; then
  echo "HARD FAIL: the mutation-B restore did not reproduce the original file (${BEFORE_MD5_B} != ${AFTER_MD5_B})." >&2
  exit 1
fi

if [ "${verify_rc}" -eq 0 ]; then
  echo "HARD FAIL: the SYNTHETIC-OFFSET mutation left the D2 Java verify GREEN — the offset-source" >&2
  echo "           half of the interop claim is VACUOUS." >&2
  sed -n '1,60p' "${TMP}/mutant_b_verify.log" >&2
  exit 1
fi
# A non-zero exit is NOT sufficient, and neither is any FAIL: the "missing json" / "empty json" FAIL
# forms are exactly what a mutant that failed to COMPILE would produce (the gen leg would have died
# before writing the manifest). Require a real per-window COMPARISON failure.
if ! grep -qE 'FAIL ranged-read-d2 [A-Za-z0-9_]+\.parquet \[' "${TMP}/mutant_b_verify.log"; then
  echo "HARD FAIL: the mutation-B verify failed WITHOUT a per-window comparison signal" >&2
  echo "           (gen rc=${gen_rc}, verify rc=${verify_rc}) — it probably did not compile." >&2
  sed -n '1,60p' "${TMP}/mutant_b_gen.log" >&2
  sed -n '1,60p' "${TMP}/mutant_b_verify.log" >&2
  exit 1
fi
echo "    mutation confirmed RED via the JAVA comparison ($(grep -cE 'FAIL ranged-read-d2 [A-Za-z0-9_]+\.parquet \[' "${TMP}/mutant_b_verify.log") window FAILs); source restored + md5-verified"

echo "==> [6/6] re-run GEN + Java VERIFY on the RESTORED source — must be GREEN again"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_RANGED_READ_GEN_DIR="${TMP}" \
    cargo test -p iceberg --test interop_ranged_read test_ranged_read_gen_rust_fixtures -- --nocapture
)
(
  cd "${SCRIPT_DIR}"
  JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH \
    /opt/maven/bin/mvn -o -q compile exec:java \
    -Dexec.args=verify-interop-ranged-read \
    -Dinterop.ranged_read.dir="${TMP}"
)

echo "==> DONE — ranged-read midpoint row-group selection interop passed (D1 + D2 incl. the bloom-padded offset-drift file; OVERLAP predicate mutation AND synthetic-offset-source mutation both proven RED)."
