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
# METADATA-LEVEL multi-spec MERGING-ACTION interop harness (AC·OO #5, Wave 2) — proves that the
# RowDelta merging action's PER-SPEC DELETE-manifest grouping (Java
# MergingSnapshotProducer.newDeleteFilesAsManifests / Rust
# SnapshotProducer::write_added_delete_manifests) produces byte-identical canonical snapshot
# metadata on both sides. The SYMMETRIC DELETE-side sibling of the Z2 fast_append multi-spec interop
# (run-interop-multi-spec.sh), whose DATA-only chain never carried delete files.
#
# THE CHAIN (V2 table; NO parquet — the fixture only reads/writes manifests):
#   ms1: fast_append F0(a="q", rc=10) under spec 0 [identity(a)]            seq 1, op append
#        ↑ the SINGLE spec-0 data manifest
#   ms2: update_partition_spec add identity(b) → spec 1 becomes default      NO SNAPSHOT
#   ms3: fast_append F3(a="r",b="s", rc=10) under spec 1                    seq 2, op append
#        ↑ the SINGLE spec-1 data manifest
#   ms4: ONE multi-spec row_delta: D0(posDelete,spec0,rc=1) + D1(posDelete,spec1,rc=1)
#                                                                            seq 3, op delete
#        ↑ THE HEADLINE — TWO DELETE manifests in ONE snapshot, partition_spec_id=0 AND 1
#
# ONE DATA MANIFEST PER SPEC BEFORE ms4 (deliberate): each spec group is a size-1 bin the merge
# manager returns as-is, so the DATA manifests pass through unchanged on both sides and this unit
# isolates the DELETE-manifest grouping (a second data manifest in one spec would trip Java's
# order-dependent `first`-relative DATA-manifest force-merge — out of scope here).
#
# TIE-SHAPING: D0 and D1 have IDENTICAL record_count=1, so the two ms4 DELETE manifests tie on all 9
# prior sort-tuple keys and differ ONLY on partition_spec_id (0 vs 1). The W3 spec-id tiebreaker at
# position 10 is the ONLY disambiguator. This property is asserted in the Rust GEN step.
#
# SABOTAGE BATTERY (fail-closed proofs; HARD-FAIL never SKIP):
#   SB1 (structural corruption): truncate one manifest file → the Java view step fails on load.
#   SB2 (drop ms4 snapshot): replace final.metadata.json with a copy lacking the ms4 DELETE snapshot
#       → the canonical view has 2 instead of 3 ordinals → byte diff fails.
#   SB3 (control): run the comparison on the CLEAN Java chain → must pass, proving SB1/SB2/SB4 are
#       non-vacuous.
#   SB4 (wrong-spec rendering): swap the FIELD DEFINITIONS of spec 0 and spec 1 in the SOURCE
#       final.metadata.json (ids unchanged), then RE-EMIT the canonical view. The ms4 spec-0 DELETE
#       manifest's 1-field tuple is now projected under the 2-field spec (and vice versa), so the
#       rendered DELETE-entry partition JSON changes → byte diff fails. Proves DELETE-entry partition
#       tuples are rendered under each manifest's OWN spec (the file's-own-spec rule) and that the
#       rendering — not just the partition_spec_id integer — is load-bearing.
#
# Driven by three comparisons per the E1 pattern:
#   1. Java performs the chain, emits java_meta.json (canonical view).
#   2. Rust performs the SAME chain (GEN test), lands rust_table/metadata/final.metadata.json;
#      Java emits its view and this script byte-diffs it against java_meta.json.
#   3. Rust asserts ITS view of BOTH chains equal java_meta.json.
#
# TEST-ONLY oracle; nothing here is in the offline cargo test gate; temp dirs gitignored.
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64.
# Run from anywhere; paths resolve relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-multispec-merge"
DIR="${TMP}/fixture"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

echo "==> [1/5] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${DIR}"

echo "==> [2/5] Java: perform the four-step multi-spec merging-action chain, emit java_meta.json"
run_oracle -Dexec.args=generate-interop-multispec-merge -Dinterop.multispec_merge.dir="${DIR}"
test -f "${DIR}/java_meta.json" || { echo "FAIL: java_meta.json not produced"; exit 1; }
echo "    java_meta.json produced OK"

echo "==> [3/5] Rust: perform the SAME chain via the production write paths (GEN test)"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_MULTISPEC_MERGE_GEN_DIR="${DIR}" \
    cargo test -p iceberg --test interop_multispec_merge -- --nocapture
)

echo "==> [4/5] Java: emit its view of the Rust chain + byte-diff against java_meta.json"
run_oracle -Dexec.args=emit-snapshot-meta \
  -Dinterop.meta.metadata="${DIR}/rust_table/metadata/final.metadata.json" \
  -Dinterop.meta.out="${DIR}/java_view_rust_meta.json"
if ! diff -u "${DIR}/java_meta.json" "${DIR}/java_view_rust_meta.json"; then
  echo "FAIL: Java's view of the Rust multi-spec merging-action chain diverges from Java's own semantics."
  exit 1
fi
echo "    Java view of Rust chain == Java view of Java chain OK"

echo "==> [5/5] Rust: assert ITS canonical views of BOTH chains equal java_meta.json"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_MULTISPEC_MERGE_DIR="${DIR}" \
    cargo test -p iceberg --test interop_multispec_merge -- --nocapture
)

# ─── Sabotage battery ────────────────────────────────────────────────────────

echo ""
echo "==> [SB1] Structural corruption: truncate a manifest file → Java view must fail"
FIRST_MANIFEST=$(find "${DIR}/rust_table/metadata" -name "*.avro" | sort | head -1)
if [ -z "${FIRST_MANIFEST}" ]; then
  echo "FAIL SB1: no manifest (.avro) files found in rust_table/metadata"
  exit 1
fi
cp "${FIRST_MANIFEST}" "${FIRST_MANIFEST}.bak"
: > "${FIRST_MANIFEST}"
if run_oracle -Dexec.args=emit-snapshot-meta \
    -Dinterop.meta.metadata="${DIR}/rust_table/metadata/final.metadata.json" \
    -Dinterop.meta.out="${DIR}/sb1_view.json" 2>/dev/null; then
  echo "FAIL SB1: Java view succeeded on a truncated manifest — should have failed"
  mv "${FIRST_MANIFEST}.bak" "${FIRST_MANIFEST}"
  exit 1
fi
mv "${FIRST_MANIFEST}.bak" "${FIRST_MANIFEST}"
echo "    SB1 PASS: truncated manifest caused Java view to fail (closed)"

echo "==> [SB2] Drop ms4 DELETE snapshot from metadata → canonical view has 2 ordinals → diff must fail"
# Strip the ms4 snapshot (the multi-spec DELETE commit, highest sequence-number) from the Java
# chain's metadata. Update current-snapshot-id AND refs so the Java parser does not reject it.
python3 - "${DIR}/table/metadata/final.metadata.json" "${TMP}/sb2_metadata.json" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    meta = json.load(f)
# Find the ms4 snapshot id (the one with the highest sequence-number — the DELETE row_delta).
snaps = sorted(meta.get("snapshots", []), key=lambda s: s.get("sequence-number", 0))
ms4_id = snaps[-1]["snapshot-id"]
ms3_id = snaps[-2]["snapshot-id"] if len(snaps) >= 2 else None
# Drop ms4 from the snapshots array.
meta["snapshots"] = [s for s in meta["snapshots"] if s["snapshot-id"] != ms4_id]
# Update current-snapshot-id to ms3.
if ms3_id is not None:
    meta["current-snapshot-id"] = ms3_id
# Update the main branch ref to ms3 so the parser does not reject it.
if "refs" in meta and "main" in meta["refs"]:
    meta["refs"]["main"]["snapshot-id"] = ms3_id
with open(sys.argv[2], "w") as f:
    json.dump(meta, f)
PYEOF
run_oracle -Dexec.args=emit-snapshot-meta \
  -Dinterop.meta.metadata="${TMP}/sb2_metadata.json" \
  -Dinterop.meta.out="${TMP}/sb2_view.json"
if diff -q "${DIR}/java_meta.json" "${TMP}/sb2_view.json" > /dev/null 2>&1; then
  echo "FAIL SB2: dropping ms4 snapshot left the view unchanged — should have diverged"
  exit 1
fi
ORDINALS_FULL=$(python3 -c "import json; d=json.load(open('${DIR}/java_meta.json')); print(len(d['snapshots']))")
ORDINALS_SB2=$(python3 -c "import json; d=json.load(open('${TMP}/sb2_view.json')); print(len(d['snapshots']))")
if [ "${ORDINALS_SB2}" -ge "${ORDINALS_FULL}" ]; then
  echo "FAIL SB2: sb2 ordinal count ${ORDINALS_SB2} >= full count ${ORDINALS_FULL} — wrong divergence type"
  exit 1
fi
echo "    SB2 PASS: dropped-ms4 view has ${ORDINALS_SB2} ordinals vs ${ORDINALS_FULL} in full — diff fails closed"

echo "==> [SB3] Control: comparison on the CLEAN Java chain must PASS"
run_oracle -Dexec.args=emit-snapshot-meta \
  -Dinterop.meta.metadata="${DIR}/table/metadata/final.metadata.json" \
  -Dinterop.meta.out="${TMP}/sb3_java_self_view.json"
if ! diff -q "${DIR}/java_meta.json" "${TMP}/sb3_java_self_view.json" > /dev/null 2>&1; then
  echo "FAIL SB3: Java's view of its OWN clean chain diverges from java_meta.json — chain is broken"
  exit 1
fi
echo "    SB3 PASS: control — Java's view of the clean Java chain == java_meta.json"

echo "==> [SB4] Wrong-spec-rendering mutation: swap the two spec DEFINITIONS in the SOURCE metadata"
echo "     → re-emit the canonical view → it must diverge (proves per-own-spec DELETE-entry partition"
echo "       rendering is load-bearing, not just the partition_spec_id integer)."
# The canonical view renders each DELETE manifest entry's partition tuple under the manifest's OWN
# spec (Java metadata.specsById().get(file.specId()), mirrored by Rust's
# manifest_meta.partition_spec — the file's-own-spec rule). To prove that rendering is load-bearing
# we mutate the ARTIFACT and RE-DERIVE the view from it (NOT post-edit the emitted JSON): swap the
# field DEFINITIONS of spec 0 and spec 1 in the rust_table's final.metadata.json (spec-id 0 keeps
# its id but now carries identity(a)+identity(b); spec-id 1 now carries identity(a) only). On
# re-emit, the ms4 spec-0 DELETE manifest's 1-field tuple is now projected under the 2-field spec
# and the spec-1 DELETE manifest's 2-field tuple under the 1-field spec, so the rendered DELETE-entry
# partition JSON changes — the view diverges from java_meta.json. A pure spec_id-integer swap in the
# OUTPUT JSON would not prove this; re-deriving from the corrupted artifact does.
#
# HARD-FAIL never SKIP: if the spec-ids are absent or same-arity (the swap would be a no-op), the
# mutator exits non-zero, which (with the rc-capture below) aborts the chain rather than masquerading
# as a pass.
set +e
python3 - "${DIR}/rust_table/metadata/final.metadata.json" "${TMP}/sb4_metadata.json" <<'PYEOF'
import json, sys

src_path = sys.argv[1]
out_path = sys.argv[2]
with open(src_path) as f:
    meta = json.load(f)

specs = {s["spec-id"]: s for s in meta.get("partition-specs", [])}
if 0 not in specs or 1 not in specs:
    print(f"FAIL SB4 setup: expected spec-ids 0 and 1, got {sorted(specs)}")
    sys.exit(1)
if len(specs[0]["fields"]) == len(specs[1]["fields"]):
    print("FAIL SB4 setup: spec 0 and spec 1 have the same arity — the swap would be a no-op")
    sys.exit(1)

# Swap ONLY the field definitions; the spec-ids stay 0 and 1 so the manifests still point at them.
specs[0]["fields"], specs[1]["fields"] = specs[1]["fields"], specs[0]["fields"]

with open(out_path, "w") as f:
    json.dump(meta, f)
print(
    f"SB4: swapped spec-0 and spec-1 field definitions "
    f"(spec 0 now {len(specs[0]['fields'])}-field, spec 1 now {len(specs[1]['fields'])}-field)"
)
PYEOF
SB4_RC=$?
set -e
if [ "${SB4_RC}" -ne 0 ]; then
  echo "FAIL SB4: the spec-swap mutator could not be applied (rc=${SB4_RC}) — aborting (no false-green)"
  exit 1
fi
run_oracle -Dexec.args=emit-snapshot-meta \
  -Dinterop.meta.metadata="${TMP}/sb4_metadata.json" \
  -Dinterop.meta.out="${TMP}/sb4_view.json"
if diff -q "${DIR}/java_meta.json" "${TMP}/sb4_view.json" > /dev/null 2>&1; then
  echo "FAIL SB4: wrong-spec rendering left the view unchanged — per-own-spec rendering is NOT load-bearing"
  exit 1
fi
echo "    SB4 PASS: wrong-spec rendering re-derived a divergent view (per-own-spec DELETE-entry partition rendering is load-bearing)"

echo ""
echo "==> DONE — multi-spec MERGING-ACTION interop passed (four-step chain: one data manifest per"
echo "     spec + multi-spec DELETE row_delta → two per-spec DELETE manifests, spec-id tiebreaker"
echo "     exercised on the DELETE side, per-spec partition rendering; 3 comparison directions +"
echo "     4-sabotage battery all closed)."
