#!/bin/bash
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

set -euo pipefail
cd "$(dirname "$0")/.."

pattern='claude[-_]opus[-_]5[-_.]0([^0-9]|$)|(^|[^a-z0-9-])opus[-_]5[-_.]0([^0-9]|$)'
allowed='claude-opus-5-5'

parts=(claude opus 5 0)
needles=(
  "$(IFS=-; printf '%s' "${parts[*]}")"
  "$(IFS=_; printf '%s' "${parts[*]}")"
  "$(printf '%s-%s-%s' "${parts[1]}" "${parts[2]}" "${parts[3]}")"
  "$(printf '%s-%s-%s.%s' "${parts[0]}" "${parts[1]}" "${parts[2]}" "${parts[3]}")"
)
for n in "${needles[@]}"; do
  if ! printf '%s\n' "$n" | grep -qiE -e "$pattern"; then
    echo "ERROR: self-test FAILED — the model-ban pattern misses '$n'; the gate is vacuous" >&2
    exit 1
  fi
done
for ok in "$allowed" "Opus 5.0 is banned" "opus-5-5"; do
  if printf '%s\n' "$ok" | grep -qiE -e "$pattern"; then
    echo "ERROR: self-test FAILED — the model-ban pattern wrongly hits '$ok'" >&2
    exit 1
  fi
done

if hits="$(git grep -nIiE -e "$pattern" -- . 2>/dev/null)"; then
  echo "ERROR: a tracked file names a banned Opus 5.0 model id; Opus means $allowed" >&2
  echo "$hits" >&2
  exit 1
fi
echo "model-ban: no tracked file names an Opus 5.0 id; Opus means $allowed"
