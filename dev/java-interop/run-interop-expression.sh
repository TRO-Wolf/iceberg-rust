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
# ExpressionParser-JSON interop harness (GAP_MATRIX row 147) — proves the Rust canonical
# expression-JSON codec (crate iceberg expr::expression_parser) is byte-for-byte 1:1 with the REAL
# Java org.apache.iceberg.expressions.ExpressionParser.
#
# THE CONTRACT: the SCHEMA is the shared artifact (Java SchemaParser.toJson <-> Rust serde Schema).
# Each side builds the SAME battery of expressions INDEPENDENTLY, keyed by a stable name; nothing
# echoes the other's JSON to form the expected, so a byte-match is real parity (anti-circular).
#
# DIRECTIONS:
#   D1 (Java validates what RUST wrote): Rust GEN writes rust_expressions.jsonl; Java loads each,
#      (a) builds the SAME expression independently and asserts its ExpressionParser.toJson byte-
#      equals the Rust JSON, and (b) fromJson(rust_json, schema) + bind + toJson and asserts the
#      bound round-trip byte-equals the Rust JSON.
#   D2 (Rust reads what JAVA wrote): Java GEN writes java_expressions.jsonl + schema.json; Rust
#      from_json(java_json, schema) + re-serialize and asserts byte-equality with Java toJson.
#
# SABOTAGE BATTERY (fail-closed proofs; a tampered op or value MUST flip the verdict to FAIL):
#   SB1 (control): the CLEAN Rust JSON must PASS Java verify (proves SB2/SB3 are non-vacuous).
#   SB2 (corrupt an op hyphen): rewrite one op "is-null" -> "is_null" (underscore) in the Rust JSON
#       -> Java verify MUST fail (op hyphen-map is load-bearing).
#   SB3 (corrupt a value): rewrite a date "2017-11-16" -> "2017-11-17" in the Rust JSON -> Java
#       verify MUST fail (the value is load-bearing).
#   SB4 (corrupt a float/double value): rewrite the double token 1.0E10 -> 1.0E11 in the Rust JSON
#       -> Java verify MUST fail (the float/double value arm is load-bearing).
#
# TEST-ONLY oracle; nothing here is in the offline cargo test gate; temp dirs gitignored.
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64.
# Run from anywhere; paths resolve relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-expression"
DIR="${TMP}/fixture"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

echo "==> [1/6] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${DIR}"

echo "==> [2/6] Java: emit java_expressions.jsonl + schema.json (independent battery)"
run_oracle -Dexec.args=generate-interop-expression -Dinterop.expression.dir="${DIR}"
test -f "${DIR}/java_expressions.jsonl" || { echo "FAIL: java_expressions.jsonl not produced"; exit 1; }
test -f "${DIR}/schema.json" || { echo "FAIL: schema.json not produced"; exit 1; }
echo "    java_expressions.jsonl + schema.json produced OK"

echo "==> [3/6] Rust D2: read Java's JSON, from_json + re-serialize, assert byte-equality"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_EXPRESSION_DIR="${DIR}" \
    cargo test -p iceberg --test interop_expression test_expression_verify_rust_reads_java -- --nocapture
)

echo "==> [4/6] Rust GEN: write rust_expressions.jsonl (+ round-trip Java's JSON in Rust)"
(
  cd "${REPO_ROOT}"
  ICEBERG_INTEROP_EXPRESSION_GEN_DIR="${DIR}" \
    cargo test -p iceberg --test interop_expression test_expression_gen_rust_writes_and_round_trips -- --nocapture
)
test -f "${DIR}/rust_expressions.jsonl" || { echo "FAIL: rust_expressions.jsonl not produced"; exit 1; }

echo "==> [5/6] Java D1: validate the Rust JSON (independent toJson + bound round-trip)"
if ! run_oracle -Dexec.args=verify-interop-expression -Dinterop.expression.dir="${DIR}" | tee "${TMP}/d1.log"; then
  echo "FAIL: Java verify of the Rust JSON errored"
  exit 1
fi
if grep -q "verify-interop-expression: 0 failures" "${TMP}/d1.log"; then
  echo "    D1 PASS: Java validated every Rust expression (0 failures)"
else
  echo "FAIL: Java verify reported failures (see ${TMP}/d1.log)"
  exit 1
fi

# ─── Sabotage battery ────────────────────────────────────────────────────────

verify_rust_jsonl() {
  # Run Java verify on the current rust_expressions.jsonl; echo the failure count line.
  run_oracle -Dexec.args=verify-interop-expression -Dinterop.expression.dir="${DIR}"
}

echo ""
echo "==> [SB1] Control: the CLEAN Rust JSON must PASS Java verify"
SB1_OUT="$(verify_rust_jsonl || true)"
if ! grep -q "verify-interop-expression: 0 failures" <<<"${SB1_OUT}"; then
  echo "FAIL SB1: clean Rust JSON did not pass Java verify — chain is broken"
  echo "${SB1_OUT}"
  exit 1
fi
echo "    SB1 PASS: clean Rust JSON validates (control non-vacuous)"

RUST_JSONL="${DIR}/rust_expressions.jsonl"
cp "${RUST_JSONL}" "${RUST_JSONL}.bak"

echo "==> [SB2] Corrupt an op hyphen: is-null -> is_null (in the escaped JSON) -> Java verify MUST fail"
# The canonical JSON is embedded as an escaped string (\"type\":\"is-null\"), so target the bare
# `is-null` substring (the op name); it is unique to the is_null/not_null entries.
grep -q 'is-null' "${RUST_JSONL}" || { echo "FAIL SB2: target op 'is-null' not present to corrupt"; mv "${RUST_JSONL}.bak" "${RUST_JSONL}"; exit 1; }
sed -i 's/is-null/is_null/' "${RUST_JSONL}"
SB2_OUT="$(verify_rust_jsonl || true)"
if grep -q "verify-interop-expression: 0 failures" <<<"${SB2_OUT}"; then
  echo "FAIL SB2: corrupted op hyphen still passed — the op map is NOT load-bearing"
  mv "${RUST_JSONL}.bak" "${RUST_JSONL}"
  exit 1
fi
cp "${RUST_JSONL}.bak" "${RUST_JSONL}"
echo "    SB2 PASS: corrupted op hyphen failed Java verify (closed)"

echo "==> [SB3] Corrupt a value: date \"2017-11-16\" -> \"2017-11-17\" -> Java verify MUST fail"
grep -q '2017-11-16' "${RUST_JSONL}" || { echo "FAIL SB3: target value '2017-11-16' not present to corrupt"; mv "${RUST_JSONL}.bak" "${RUST_JSONL}"; exit 1; }
sed -i 's/2017-11-16/2017-11-17/' "${RUST_JSONL}"
SB3_OUT="$(verify_rust_jsonl || true)"
if grep -q "verify-interop-expression: 0 failures" <<<"${SB3_OUT}"; then
  echo "FAIL SB3: corrupted date value still passed — the value is NOT load-bearing"
  mv "${RUST_JSONL}.bak" "${RUST_JSONL}"
  exit 1
fi
cp "${RUST_JSONL}.bak" "${RUST_JSONL}"
echo "    SB3 PASS: corrupted date value failed Java verify (closed)"

echo "==> [SB4] Corrupt a double sci-notation value: 1.0E10 -> 1.0E11 -> Java verify MUST fail"
# The eq_double entry serializes the float/double VALUE arm (Java Double.toString). The escaped
# JSON carries the bare numeric token `1.0E10`; corrupting it proves the float/double value codec
# is load-bearing in the interop (not just the unit-pinned arm).
grep -q '1.0E10' "${RUST_JSONL}" || { echo "FAIL SB4: target value '1.0E10' not present to corrupt"; mv "${RUST_JSONL}.bak" "${RUST_JSONL}"; exit 1; }
sed -i 's/1.0E10/1.0E11/' "${RUST_JSONL}"
SB4_OUT="$(verify_rust_jsonl || true)"
if grep -q "verify-interop-expression: 0 failures" <<<"${SB4_OUT}"; then
  echo "FAIL SB4: corrupted double value still passed — the float/double value is NOT load-bearing"
  mv "${RUST_JSONL}.bak" "${RUST_JSONL}"
  exit 1
fi
mv "${RUST_JSONL}.bak" "${RUST_JSONL}"
echo "    SB4 PASS: corrupted double value failed Java verify (closed)"

echo ""
echo "==> [6/6] DONE — ExpressionParser interop passed: D2 (Rust reads Java) + D1 (Java validates"
echo "     Rust) byte-for-byte over the full battery + 4-sabotage battery all closed."
