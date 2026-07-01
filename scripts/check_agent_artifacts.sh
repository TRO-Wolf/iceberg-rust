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
#
# check_agent_artifacts.sh — fail if agent-session output wrappers leaked into
# tracked files.
#
# Motivation (2026-07-01 review): literal tool-call wrapper tags were found
# committed at the tails of docs/parity/GAP_MATRIX.md and docs/parity/ROADMAP.md.
# Agent-written files are the norm in this repo; this gate makes that class of
# leak impossible to merge. Wired into `make check-agent-artifacts` and CI.
#
# The needles are assembled by concatenation so THIS script never matches
# itself (and survives being scanned by itself or by future agents).

set -euo pipefail
cd "$(dirname "$0")/.."

LT='<'
patterns=(
  "${LT}/content>"
  "${LT}/invoke>"
  "${LT}antml"
  "${LT}/antml"
  "${LT}function_calls>"
  "${LT}/function_calls>"
  "${LT}/parameter>"
)

hits_file="$(mktemp)"
trap 'rm -f "$hits_file"' EXIT

fail=0
for p in "${patterns[@]}"; do
  # git grep: tracked files only; -I skips binaries; -F literal match.
  # This script is pathspec-excluded defensively even though concatenation
  # already prevents self-matching.
  if git grep -nI -F -e "$p" -- ':(exclude)scripts/check_agent_artifacts.sh' >"$hits_file" 2>/dev/null; then
    echo "ERROR: agent-session artifact '$p' found in tracked files:" >&2
    cat "$hits_file" >&2
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo >&2
  echo "Strip the wrapper tags above — they are tool-call transport, not content." >&2
  exit 1
fi

echo "OK: no agent-session artifacts in tracked files."
