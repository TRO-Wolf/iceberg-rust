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
# v4 (2026-08-25): a second needle family — REVIEW-PROCESS RESIDUE. Actor /
# Critic / Falsifier identifiers and finding ids were reaching production
# comments (48 lines across 15 files, entering with #181 and #183). AGENTS.md
# "Comments and prose" routes review evidence to the unit ledger in task/, so
# this family is scoped to crates/ ONLY: task/, docs/ and the SEPMO tree are its
# correct homes and must never trip. Matching is word-bounded (-w) and
# case-INSENSITIVE: lowercase "critic-octo" is the same residue, and -w already
# excludes "Critical" and "Critically" in either case. An anti-probe proves that
# exclusion.
#
# The self-test reaches the needles, the family size and the pathspec VARIABLE.
# It does not reach the grep call: a pathspec or a hit count substituted in place
# still passes. Those are single visible edits to the shipped scan. Dropping -i
# was the one SILENT regression — it re-greens the tree over lowercase residue —
# so a lowercase sample pins it.
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

# Review-process residue. Assembled by concatenation so this script never
# matches itself.
C='C'
c='c'
residue_grep_flags='-nIEwi'
# -w (whole word) does the bounding: git grep -E is POSIX ERE, where \b is a
# literal 'b', not a word boundary. Each needle carries a matching SAMPLE, since
# a probe holding the pattern itself proves nothing for a needle with metachars.
# "Actor-Critic" needs no needle of its own: -w treats the hyphen as a boundary,
# so the Critic needle already matches it.
residue_patterns=(
  "${C}ritic"
  "Falsifier"
  "SEPMO"
)
residue_samples=(
  "the ${c}ritic-octo probe found it"
  "Falsifier F9 found the gap"
  "SEPMO cycle 2 remediation"
)
residue_pathspec='crates/'

# Strings that must NOT match: proof the needles stay word-bounded.
residue_anti_patterns=(
  "Critically this pins the snapshot id"
  "BUG-001 Critical / BUG-004"
  "the load-bearing Critical pin"
)

# One EXIT trap owns ALL cleanup, installed before anything is created:
# it fires on normal exit, errexit aborts, and SIGINT — a function-local
# RETURN trap covers none of the abort paths.
probe=".agent_artifact_selftest_probe.$$.tmp"
# Under crates/ so the self-test proves the residue pathspec REACHES the probe,
# not merely that the needle matches somewhere.
residue_probe="crates/.review_residue_selftest_probe.$$.tmp"
tmp_index=""
hits_file=""
err_file=""
trap 'rm -f "$probe" "$residue_probe" "$tmp_index" "$hits_file" "$err_file"' EXIT

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

  # Wiring, not detection: a self-test that only exercises the needles passes
  # while the shipped scan points at nothing.
  if [ "${#residue_patterns[@]}" -ne 3 ]; then
    echo "ERROR: residue needle family changed size — update the count in the OK line" >&2
    return 1
  fi
  if [ "$residue_pathspec" != "crates/" ]; then
    echo "ERROR: residue pathspec is '$residue_pathspec', not 'crates/' — the scan would miss the product surface" >&2
    return 1
  fi
  case "$residue_probe" in
    "$residue_pathspec"*) ;;
    *) echo "ERROR: residue probe is outside the scanned pathspec — self-test is vacuous" >&2
       return 1 ;;
  esac
  if [ -e "$residue_probe" ]; then
    echo "ERROR: self-test probe path '$residue_probe' already exists — refusing to overwrite" >&2
    return 1
  fi
  local i
  for i in "${!residue_patterns[@]}"; do
    p="${residue_patterns[$i]}"
    printf '%s\n' "${residue_samples[$i]}" >"$residue_probe"
    if ! GIT_INDEX_FILE="$tmp_index" git add -f -- "$residue_probe" 2>/dev/null; then
      echo "ERROR: self-test could not stage its residue probe file" >&2
      return 1
    fi
    if ! GIT_INDEX_FILE="$tmp_index" git grep $residue_grep_flags -e "$p" -- "$residue_probe" >/dev/null 2>&1; then
      echo "ERROR: self-test FAILED — residue needle '$p' was not detected; the gate is vacuous" >&2
      return 1
    fi
  done

  for p in "${residue_anti_patterns[@]}"; do
    printf '%s\n' "$p" >"$residue_probe"
    if ! GIT_INDEX_FILE="$tmp_index" git add -f -- "$residue_probe" 2>/dev/null; then
      echo "ERROR: self-test could not stage its anti-probe file" >&2
      return 1
    fi
    for q in "${residue_patterns[@]}"; do
      if GIT_INDEX_FILE="$tmp_index" git grep $residue_grep_flags -e "$q" -- "$residue_probe" >/dev/null 2>&1; then
        echo "ERROR: self-test FAILED — needle '$q' matched legitimate text '$p'" >&2
        return 1
      fi
    done
  done
  rm -f "$residue_probe"
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

residue_hits=0
for p in "${residue_patterns[@]}"; do
  rc=0
  git grep $residue_grep_flags -e "$p" -- "$residue_pathspec" >"$hits_file" 2>"$err_file" || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "ERROR: review-process residue '$p' found in $residue_pathspec:" >&2
    cat "$hits_file" >&2
    residue_hits=$(( residue_hits + $(wc -l <"$hits_file") ))
  elif [ "$rc" -ge 2 ]; then
    echo "ERROR: git grep failed (exit $rc) while scanning for '$p' — cannot certify the tree:" >&2
    cat "$err_file" >&2
    exit 1
  fi
done

if [ "$residue_hits" -ne 0 ]; then
  echo >&2
  echo "Review evidence belongs in the unit ledger under task/, per AGENTS.md" >&2
  echo "'Comments and prose'. State the constraint the code keeps; never name the" >&2
  echo "review, its round, or its finding id." >&2
  exit 1
fi

echo "OK: no agent-session artifacts or review residue in tracked files (11 + 3 needles, self-tested)."
