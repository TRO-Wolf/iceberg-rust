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
# BUILDER-FLIPS write-action interop harness (Wave 3) — DeleteFiles' two "builder flip" capabilities:
# `deleteFromRowFilter(Expression)` (GAP_MATRIX row 135) + `caseSensitive(boolean)` (row 134), proven
# bidirectionally against Java `StreamingDelete` (1.10.0). This is the FIRST interop exercising the
# by-row-filter delete RESOLUTION (which file set is removed), not a conflict ACCEPT/REJECT decision.
#
# THE VEHICLE: table.newDelete().caseSensitive(flag).deleteFromRowFilter(filter).commit() →
# MergingSnapshotProducer.deleteByRowFilter / ManifestFilterManager.{caseSensitive,
# PartitionAndMetricsEvaluator}, mirrored by Rust DeleteFiles::delete_from_row_filter / case_sensitive →
# SnapshotProducer::resolve_filter_deletes / build_residual_evaluator → ResidualEvaluator::of(spec,
# expr, caseSensitive) + the strict/inclusive metrics evaluators.
#
# Five scenarios over a PARTITIONED V2 {id long, Category string, y long} table identity(Category) (the
# MIXED-CASE column Category is the case-sensitivity discriminator):
#   scenario                  | filter (column ref) | caseSensitive | expected
#   filter_delete_partition   | Category == "a"     | true          | survivors = {b} (DELETE cat=a)
#   filter_keep_complement    | Category == "b"     | true          | survivors = {a} (DELETE cat=b)
#   filter_partial_error      | y >= 55             | true          | ERROR — partial (cat=b straddles)
#   case_insensitive_match    | category == "a"     | FALSE         | survivors = {b} (binds insensitively)
#   case_sensitive_reject     | category == "a"     | true          | ERROR — wrong-case bind fails
#
# File paths are random per engine, so the DELETE/KEEP signal is the SET of surviving Category partition
# values (identity(Category) ⇒ each file's category is its partition value). The expected outcome is
# hand-declared identically on both sides (anti-circular).
#
# NAMED DIVERGENCE (kept OUT of the set, live-oracle finding 2026-06-16): a row filter matching NO live
# file diverges — Rust REJECTS the empty no-op commit, Java COMMITS a no-op Delete snapshot. The KEEP
# semantics are proven by filter_keep_complement / filter_delete_partition (non-empty, both agree).
#
# BOTH directions:
#   D1 (Rust validates Java's table): Rust `register_table`s <scenario>/table and runs the symmetric
#     delete — `tests/interop_builder_flips.rs` test_rust_runs_builder_flips_on_java_tables.
#   D2 (Java validates Rust's table): Java loads <scenario>/rust_table and runs the symmetric delete —
#     `BuilderFlipsOracle.verify`, "verify-interop-builder-flips: N failures".
#
# Test-only oracle; nothing here is in the offline cargo test gate; the temp dir is gitignored.
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64.
# Run from anywhere; paths resolve relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-builder-flips"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

# The 5 scenario directory names (used by build_scratch).
SCENARIOS=(filter_delete_partition filter_keep_complement filter_partial_error \
  case_insensitive_match case_sensitive_reject)

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

echo "==> [1/7] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/7] Java: generate all 5 scenario tables (<scenario>/table, partitioned identity(Category))"
run_oracle -Dexec.args=generate-interop-builder-flips \
  -Dinterop.builder_flips.dir="${TMP}"

echo "==> [3/7] Rust: generate all 5 scenario tables (<scenario>/rust_table) via the GEN test"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_BUILDER_FLIPS_GEN_DIR="${TMP}" \
    cargo test -p iceberg --test interop_builder_flips \
    test_builder_flips_gen -- --nocapture
)

echo "==> [4/7] Java: verify-interop-builder-flips (D2 — Java validates the Rust tables)"
VERIFY_OUT="$(run_oracle -Dexec.args=verify-interop-builder-flips \
  -Dinterop.builder_flips.dir="${TMP}")" || true
echo "${VERIFY_OUT}"
# Fail-closed two ways: a per-check `^FAIL ` line OR absence of the `0 failures` sentinel.
if echo "${VERIFY_OUT}" | grep -q '^FAIL ' \
  || ! echo "${VERIFY_OUT}" | grep -q 'verify-interop-builder-flips: 0 failures'; then
  echo "==> FAILED — D2 verify emitted a FAIL line or did not emit the '0 failures' sentinel."
  exit 1
fi

echo "==> [5/7] Rust: validate the Java tables (D1 — register + symmetric delete_from_row_filter)"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_BUILDER_FLIPS_DIR="${TMP}" \
    cargo test -p iceberg --test interop_builder_flips \
    test_rust_runs_builder_flips_on_java_tables -- --nocapture
)

echo "==> [6/7] Sabotage battery — D2 verify must FAIL closed on a corrupted/mutated table"
# CONTROL: step 4's clean verify already passed, so a sabotage-fail is meaningful. Each sabotage runs
# against a SCRATCH copy (the clean TMP is untouched). HARD-FAIL, never SKIP, if a corruption cannot be
# applied (a sabotage that did not actually corrupt anything proves nothing).
#
# RESIDUE STRIP (the C2/C3/C4 lesson): step 4's verify already ran `ops.commit(null, loaded)` against the
# REAL ${TMP}/<scenario>/rust_table, which wrote `v0.metadata.json` (and a `v1.metadata.json` for each
# scenario whose delete COMMITTED). The sabotage scratch copies those tables, so a NAIVE re-verify would
# re-run `commit(null, loaded)` → write `v0.metadata.json` → AlreadyExistsException BEFORE the delete
# resolution ever runs, masking the semantic claim with a structural collision. So build_scratch DELETES
# the v*.metadata.json residue, restoring each scratch to the pre-verify state (only the immutable
# numbered + final metadata remain) so the production resolution path is REACHED.
build_scratch() {
  local scratch="${TMP}/sabotage_scratch"
  rm -rf "${scratch}"
  mkdir -p "${scratch}"
  for scenario in "${SCENARIOS[@]}"; do
    cp -r "${TMP}/${scenario}" "${scratch}/${scenario}"
    rm -f "${scratch}/${scenario}"/rust_table/metadata/v*.metadata.json
  done
  echo "${scratch}"
}

# (a) SEMANTIC: rename the schema column `Category` -> `category` in the case_sensitive_reject scenario's
#     Rust-written final.metadata.json. That scenario's wrong-cased filter `category == "a"` under the
#     DEFAULT caseSensitive(true) is expected to FAIL to bind (ERROR). With the schema column now spelled
#     `category`, the SAME filter BINDS case-sensitively, DELETES cat=a, and the outcome flips to
#     survivors=[b] — contradicting the hand-declared ERROR. The pin is the SPECIFIC outcome-mismatch
#     line ("outcome survivors=[b] but expected ERROR"): it only fires when the production caseSensitive
#     bind was actually REACHED and re-derived a DELETE, so a regression that ignored the flag (always
#     case-insensitive) is caught here. The partition spec references field id 2 (not the name), so the
#     rename leaves the partitioning intact; the parquet maps by field id, so the data still reads.
#
#     HARD-FAIL never SKIP: if no `Category` field is present to rename, exit non-zero — a sabotage that
#     did not actually mutate the schema proves nothing.
SCRATCH="$(build_scratch)"
REJECT_FINAL="${SCRATCH}/case_sensitive_reject/rust_table/metadata/final.metadata.json"
if [[ ! -f "${REJECT_FINAL}" ]]; then
  echo "==> FAILED — cannot apply rename sabotage: ${REJECT_FINAL} absent (GEN gated?)"
  exit 1
fi
python3 - "${REJECT_FINAL}" <<'PY' || { echo "==> FAILED — rename sabotage could not mutate the schema"; exit 1; }
import json, sys
path = sys.argv[1]
with open(path) as fh:
    meta = json.load(fh)
changed = 0
for schema in meta.get("schemas", []):
    for field in schema.get("fields", []):
        if field.get("name") == "Category":
            field["name"] = "category"
            changed += 1
if changed == 0:
    sys.exit("no `Category` schema field to rename — rename sabotage cannot be applied")
with open(path, "w") as fh:
    json.dump(meta, fh)
print("renamed %d `Category` schema field(s) to `category` — the wrong-case filter now binds" % changed)
PY
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-builder-flips \
  -Dinterop.builder_flips.dir="${SCRATCH}")" || true
if echo "${SAB_OUT}" | grep -q 'verify-interop-builder-flips: 0 failures'; then
  echo "FAIL sabotage(semantic-rename): verify PASSED after renaming Category->category where the wrong-case bind was expected to ERROR — the caseSensitive flag is not load-bearing in the reached path"
  echo "${SAB_OUT}"
  exit 1
fi
# Require the SPECIFIC outcome-mismatch line for the renamed scenario, proving the production resolution
# re-derived a DELETE under the now-matching column name (not a structural collision that died early).
if ! echo "${SAB_OUT}" | grep -q 'builder-flips-d2\[case_sensitive_reject\]: outcome survivors=\[b\] but expected ERROR'; then
  echo "FAIL sabotage(semantic-rename): verify failed but NOT via the outcome-mismatch path —"
  echo "  the delete resolution was never reached (structural collision/parse error masks the claim)."
  echo "${SAB_OUT}"
  exit 1
fi
echo "PASS sabotage(semantic-rename): renaming Category->category flips case_sensitive_reject ERROR->survivors=[b] via the outcome-mismatch path (the caseSensitive bind was reached and re-derived a DELETE)"

# (b) STRUCTURAL: truncate the case_sensitive_reject scenario's final.metadata.json so it no longer parses
#     ⇒ the verify's load branch errors ⇒ FAIL closed via the load-error path.
SCRATCH="$(build_scratch)"
RUST_FINAL="${SCRATCH}/case_sensitive_reject/rust_table/metadata/final.metadata.json"
if [[ ! -f "${RUST_FINAL}" ]]; then
  echo "==> FAILED — cannot apply truncate sabotage: ${RUST_FINAL} absent (GEN gated?)"
  exit 1
fi
SIZE="$(stat -c%s "${RUST_FINAL}")"
if (( SIZE <= 60 )); then
  echo "==> FAILED — cannot apply truncate sabotage: ${RUST_FINAL} is only ${SIZE} bytes"
  exit 1
fi
head -c "$(( SIZE - 60 ))" "${RUST_FINAL}" > "${RUST_FINAL}.tmp"
mv "${RUST_FINAL}.tmp" "${RUST_FINAL}"
SAB_OUT="$(run_oracle -Dexec.args=verify-interop-builder-flips \
  -Dinterop.builder_flips.dir="${SCRATCH}")" || true
if echo "${SAB_OUT}" | grep -q 'verify-interop-builder-flips: 0 failures'; then
  echo "FAIL sabotage(truncate): verify PASSED on a truncated metadata file — NOT fail-closed"
  echo "${SAB_OUT}"
  exit 1
fi
# Require the SPECIFIC load-error path for the truncated scenario (the verify's catch branch).
if ! echo "${SAB_OUT}" | grep -q 'builder-flips-d2\[case_sensitive_reject\]: unexpected error loading the table'; then
  echo "FAIL sabotage(truncate): verify failed but NOT via the load-error path — the truncation did not"
  echo "  actually break the metadata load (corruption proved nothing)."
  echo "${SAB_OUT}"
  exit 1
fi
echo "PASS sabotage(truncate): a truncated metadata file fails closed via the load-error path"
rm -rf "${TMP}/sabotage_scratch"

echo "==> [7/7] DONE — DeleteFiles builder-flips interop passed."
echo "    Scenarios: filter_delete_partition (survivors={b}) + filter_keep_complement (survivors={a})"
echo "               + filter_partial_error (ERROR) + case_insensitive_match (survivors={b})"
echo "               + case_sensitive_reject (ERROR)"
echo "    Capabilities: deleteFromRowFilter (row 135) + caseSensitive (row 134) — DELETE/KEEP/PARTIAL +"
echo "                  case-insensitive bind + wrong-case-rejects boundary"
echo "    Directions: D1 (Rust validates Java) + D2 (Java validates Rust)"
echo "    Sabotage battery: semantic-rename (Category->category, pinned to the outcome-mismatch line,"
echo "                      resolution reached) + structural-truncate (pinned to the load-error line);"
echo "                      control passed in step 4"
