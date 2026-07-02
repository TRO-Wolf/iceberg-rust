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
# committed at the tails of docs/parity/GAP_MATRIX.md and docs/parity/ROADMAP.md
# (the latter since archived to docs/parity/archive/2026-06_origin-roadmap.md).
# Agent-written files are the norm in this repo; this gate makes that class of
# leak impossible to merge. Wired into `make check-agent-artifacts` and CI.
#
# v2 (2026-07-01, review follow-up 1 / Critic LOW-1): needle set broadened to
# the function_results tag family and bare OPENING tags; matching is now
# case-insensitive (covers uppercase variants). `<result>` / `<output>` are
# DELIBERATELY excluded — too generic (legitimate XML/HTML in fixtures or docs
# would false-positive). A built-in self-test plants every needle in a probe
# staged against a TEMPORARY COPY of the index and hard-fails if any needle
# goes undetected: a gate that cannot catch its own probe is vacuous (the
# repo's sabotage-must-hard-fail doctrine). The real index is never touched.
#
# v3 hardening (2026-07-01 audit): a git-grep ERROR (exit >= 2) in the real
# scan is now a hard failure instead of silently reading as "no match" (the
# false-green the vacuity doctrine forbids); cleanup is one script-level EXIT
# trap set before anything is created (fires on errexit aborts and SIGINT,
# which a function-local RETURN trap does not); the probe name is unique per
# process (concurrent runs in one worktree cannot delete each other's probe).
#
# Referencing these tags IN PROSE without tripping the gate: never write a
# needle verbatim — omit the leading '<' (as task/todo.md does), or assemble
# it by concatenation (as this script does). If a doc must one day quote a
# leaked tag byte-for-byte, add that file to the ':(exclude)' pathspec below.
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

# One EXIT trap owns ALL cleanup, installed before anything is created:
# it fires on normal exit, errexit aborts, and SIGINT — a function-local
# RETURN trap covers none of the abort paths.
probe=".agent_artifact_selftest_probe.$$.tmp"
tmp_index=""
hits_file=""
err_file=""
trap 'rm -f "$probe" "$tmp_index" "$hits_file" "$err_file"' EXIT

# --- self-test: every needle must be detectable, or the gate is vacuous -----
self_test() {
  local p
  tmp_index="$(mktemp)"
  cp "$(git rev-parse --git-path index)" "$tmp_index"
  if [ -e "$probe" ]; then
    echo "ERROR: self-test probe path '$probe' already exists — refusing to overwrite" >&2
    return 1
  fi
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
  rm -f "$probe"
}
self_test

# --- the real scan -----------------------------------------------------------
hits_file="$(mktemp)"
err_file="$(mktemp)"

fail=0
for p in "${patterns[@]}"; do
  # git grep: tracked files only; -I skips binaries; -F literal; -i covers
  # case variants. This script is pathspec-excluded defensively even though
  # concatenation already prevents self-matching. Exit 0 = hits (leak),
  # exit 1 = clean, exit >= 2 = git itself failed — a hard error, NOT a pass.
  rc=0
  git grep -inIF -e "$p" -- ':(exclude)scripts/check_agent_artifacts.sh' >"$hits_file" 2>"$err_file" || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "ERROR: agent-session artifact '$p' found in tracked files:" >&2
    cat "$hits_file" >&2
    fail=1
  elif [ "$rc" -ge 2 ]; then
    echo "ERROR: git grep failed (exit $rc) while scanning for '$p' — cannot certify the tree:" >&2
    cat "$err_file" >&2
    exit 1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo >&2
  echo "Strip the wrapper tags above — they are tool-call transport, not content." >&2
  exit 1
fi

echo "OK: no agent-session artifacts in tracked files (11 needles, self-tested)."
