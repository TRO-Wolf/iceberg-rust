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
TMP="${SCRIPT_DIR}/target/pr5a-commit-decode"
EXPECTED_NEEDLES=12

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="${JAVA_HOME}/bin:${PATH}"
JAVAP="${JAVA_HOME}/bin/javap"
AWS_JAR="${HOME}/.m2/repository/org/apache/iceberg/iceberg-aws/1.10.0/iceberg-aws-1.10.0.jar"
CORE_JAR="${HOME}/.m2/repository/org/apache/iceberg/iceberg-core/1.10.0/iceberg-core-1.10.0.jar"

echo "==> [1/4] Prerequisites"
test -x "${JAVAP}" || { echo "FAIL missing javap at ${JAVAP}"; exit 1; }
test -f "${AWS_JAR}" || { echo "FAIL missing iceberg-aws 1.10.0 jar"; exit 1; }
test -f "${CORE_JAR}" || { echo "FAIL missing iceberg-core 1.10.0 jar"; exit 1; }

rm -rf "${TMP}"
mkdir -p "${TMP}"

echo "==> [2/4] Decode GlueTableOperations and REST commit handlers"
"${JAVAP}" -c -p -classpath "${AWS_JAR}" org.apache.iceberg.aws.glue.GlueTableOperations \
  > "${TMP}/GlueTableOperations.javap"
"${JAVAP}" -c -p -classpath "${CORE_JAR}" \
  'org.apache.iceberg.rest.ErrorHandlers$CommitErrorHandler' \
  > "${TMP}/CommitErrorHandler.javap"
"${JAVAP}" -c -p -classpath "${CORE_JAR}" \
  'org.apache.iceberg.rest.ErrorHandlers$DefaultErrorHandler' \
  > "${TMP}/DefaultErrorHandler.javap"

echo "==> [3/4] Confirm S3TablesTableOperations is absent from iceberg-aws 1.10.0"
if jar tf "${AWS_JAR}" | grep -q 'S3TablesTableOperations'; then
  echo "FAIL unexpected S3TablesTableOperations in iceberg-aws 1.10.0"
  exit 1
fi

echo "==> [4/4] Count required needles"
HITS=0
hit() {
  local file="$1"
  local needle="$2"
  if grep -F -q -- "${needle}" "${file}"; then
    HITS=$((HITS + 1))
  else
    echo "FAIL missing needle in $(basename "${file}"): ${needle}"
    exit 1
  fi
}

hit "${TMP}/GlueTableOperations.javap" "protected void doCommit"
hit "${TMP}/GlueTableOperations.javap" "org/apache/iceberg/exceptions/CommitFailedException"
hit "${TMP}/GlueTableOperations.javap" "checkCommitStatus"
hit "${TMP}/GlueTableOperations.javap" "org/apache/iceberg/exceptions/CommitStateUnknownException"
hit "${TMP}/GlueTableOperations.javap" "handleAWSExceptions"
hit "${TMP}/GlueTableOperations.javap" "software/amazon/awssdk/services/glue/model/AccessDeniedException"
hit "${TMP}/GlueTableOperations.javap" "software/amazon/awssdk/services/glue/model/ConcurrentModificationException"
hit "${TMP}/GlueTableOperations.javap" "org/apache/iceberg/aws/util/RetryDetector.retried"
hit "${TMP}/CommitErrorHandler.javap" "org/apache/iceberg/exceptions/CommitFailedException"
hit "${TMP}/CommitErrorHandler.javap" "org/apache/iceberg/exceptions/CommitStateUnknownException"
hit "${TMP}/DefaultErrorHandler.javap" "403: 149"
hit "${TMP}/DefaultErrorHandler.javap" "org/apache/iceberg/exceptions/ForbiddenException"

if [ "${HITS}" -ne "${EXPECTED_NEEDLES}" ]; then
  echo "FAIL fixture count ${HITS} != expected ${EXPECTED_NEEDLES}"
  exit 1
fi

echo "pr5a-catalog-commit-decode: ${HITS} needles"
echo "OK"
cd "${REPO_ROOT}" >/dev/null
