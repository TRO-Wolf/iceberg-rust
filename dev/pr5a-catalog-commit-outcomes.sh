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
# Credentialed PR-5A runner. Hard-fails when ICEBERG_PR5A_CREDENTIALED=1 and
# required configuration is absent. Does not print credentials or object-store
# URLs. HTTP attempt counting is PR-5B; this runner records it as unavailable.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [ "${ICEBERG_PR5A_CREDENTIALED:-}" != "1" ]; then
  echo "FAIL ICEBERG_PR5A_CREDENTIALED must be 1 to run this runner"
  exit 1
fi

need() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "FAIL ${name} is required when ICEBERG_PR5A_CREDENTIALED=1"
    exit 1
  fi
}

need ICEBERG_PR5A_GLUE_WAREHOUSE
need ICEBERG_PR5A_S3TABLES_BUCKET_ARN

if [ -z "${AWS_ACCESS_KEY_ID:-}" ] \
  && [ -z "${AWS_PROFILE:-}" ] \
  && [ -z "${AWS_CONTAINER_CREDENTIALS_RELATIVE_URI:-}" ] \
  && [ -z "${AWS_WEB_IDENTITY_TOKEN_FILE:-}" ]; then
  echo "FAIL no AWS credential source is present (access key, profile, container, or web identity)"
  exit 1
fi

echo "catalog_attempts_field=catalog_commit_attempts"
echo "http_attempts_field=unavailable:pr5b"
echo "running glue and s3tables credentialed commit-outcome smokes"

cargo test -p iceberg-catalog-glue --lib --locked \
  credentialed_glue_commit_class_smokes_and_one_accepted_then_lost_append \
  -- --exact --nocapture
cargo test -p iceberg-catalog-s3tables --lib --locked \
  credentialed_s3tables_commit_class_smokes_and_one_accepted_then_lost_append \
  -- --exact --nocapture

echo "OK"
