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
# METADATA-ONLY interop harness for the V3 `unknown` primitive type (BLOCK-2 G2).
#
# `unknown` (Java Types.UnknownType, typeId UNKNOWN, toString "unknown"; Schema.MIN_FORMAT_VERSIONS
# gates it at v3) is an always-null column with NO physical storage (TypeToMessageType returns null —
# no parquet column). The WHOLE contract for the type is therefore the schema/metadata round-trip:
# NO data files, NO Docker.
#
#   1. JAVA writes a V3 schema carrying an `unknown` column at every placement (top-level, nested
#      struct, list element, map value) to java.metadata.json + java.schema.json (TableMetadataParser
#      / SchemaParser). [generate-interop-unknown]
#   2. RUST (interop_unknown.rs, env ICEBERG_INTEROP_UNKNOWN_GEN_DIR) parses the Java fixture and
#      writes rust.metadata.json via its OWN serde_json serializer — the Rust-written metadata.
#   3. JAVA parses rust.metadata.json via TableMetadataParser and asserts the parsed schema struct
#      equals the canonical unknown schema (field id / name / type / required recursively) —
#      JAVA judging Rust's written metadata. [verify-interop-unknown]
#   4. RUST (the offline part of interop_unknown.rs) additionally asserts ITS view of the Java fixture
#      equals the canonical schema — read parity. (This runs without this script, in the offline gate.)
#
# A divergence in the `unknown` token, a dropped column, or a wrong-typed parse fails loudly.
# TEST-ONLY oracle; nothing here is in the offline `cargo test` gate; the temp dir under
# dev/java-interop/target/ is gitignored.
#
# Requirements: Maven at /opt/maven/bin/mvn, Java 11 at /usr/lib/jvm/java-11-openjdk-amd64, the
# repo's pinned Rust toolchain.
#
# Run from anywhere; paths resolve relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP="${SCRIPT_DIR}/target/interop-unknown"

MVN="/opt/maven/bin/mvn"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:${PATH}"

run_oracle() {
  (cd "${SCRIPT_DIR}" && "${MVN}" -o -q compile exec:java "$@" 2>&1)
}

echo "==> [1/4] Reset the temp dir: ${TMP}"
rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/4] Java: write java.metadata.json + java.schema.json (V3 schema with an unknown column)"
run_oracle -Dexec.args=generate-interop-unknown -Dinterop.unknown.dir="${TMP}"

echo "==> [3/4] Rust: parse the Java fixture and write rust.metadata.json via its own serializer"
(
  cd "${REPO_ROOT}"
  # The test reads the COMMITTED fixture for its offline assertions; we also point the GEN env var
  # at the TMP dir so it writes rust.metadata.json there for the Java verify step. Copy the freshly
  # generated Java fixture into the crate's testdata location is NOT needed — the committed one is
  # the read-parity ground truth; here we only need rust.metadata.json in TMP.
  ICEBERG_INTEROP_UNKNOWN_GEN_DIR="${TMP}" \
    cargo test -p iceberg --test interop_unknown -- --nocapture
)

echo "==> [4/4] Java: parse rust.metadata.json and assert the unknown schema struct matches"
run_oracle -Dexec.args=verify-interop-unknown -Dinterop.unknown.dir="${TMP}"

echo "==> DONE — metadata-only unknown-type interop passed (Java writes, Rust reads+writes, Java verifies)."
