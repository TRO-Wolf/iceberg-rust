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
# v2 (2026-07-01, review follow-up 1 / Critic LOW-1): needle set broadened to
# the function_results tag family and bare OPENING tags; matching is now
# case-insensitive (covers uppercase variants). `<result>` / `<output>` are
# DELIBERATELY excluded — too generic (legitimate XML/HTML in fixtures or docs
# would false-positive). A built-in self-test now plants every needle in a
# probe staged against a TEMPORARY COPY of the index and hard-fails if any
# needle goes undetected: a gate that cannot catch its own probe is vacuous
# (the repo's sabotage-must-hard-fail doctrine). The real index is never
# touched.
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
  "${LT}function_results>"
  "${LT}/function_results>"
  "${LT}invoke name="
  "${LT}parameter name="
)

# --- self-test: every needle must be detectable, or the gate is vacuous -----
self_test() {
  local git_index tmp_index probe p
  git_index="$(git rev-parse --git-path index)"
  tmp_index="$(mktemp)"
  probe=".agent_artifact_selftest_probe.tmp"
  cp "$git_index" "$tmp_index"
  # shellcheck disable=SC2064  # expand now: paths are fixed at trap time
  trap "rm -f '$tmp_index' '$probe'" RETURN

  for p in "${patterns[@]}"; do
    printf 'selftest probe: %s\n' "$p" >"$probe"
    if ! GIT_INDEX_FILE="$tmp_index" git add -f -- "$probe" 2>/dev/null; then
      echo "ERROR: self-test could not stage its probe file" >&2
      return 1
    fi
    if ! GIT_INDEX_FILE="$tmp_index" git grep -inIF -e "$p" -- "$probe" >/dev/null 2>&1; then
      echo "ERROR: self-test FAILED — needle '$p' was not detected; the gate is vacuous" >&2
      return 1
    fi
  done
}
self_test

# --- the real scan -----------------------------------------------------------
hits_file="$(mktemp)"
trap 'rm -f "$hits_file"' EXIT

fail=0
for p in "${patterns[@]}"; do
  # git grep: tracked files only; -I skips binaries; -F literal; -i covers
  # case variants. This script is pathspec-excluded defensively even though
  # concatenation already prevents self-matching.
  if git grep -inIF -e "$p" -- ':(exclude)scripts/check_agent_artifacts.sh' >"$hits_file" 2>/dev/null; then
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

echo "OK: no agent-session artifacts in tracked files (11 needles, self-tested)."
