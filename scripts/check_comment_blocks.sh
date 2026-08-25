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

# check_comment_blocks.sh — fail if the change under review ADDS a comment block
# longer than MAX_LINES.
#
# Motivation (2026-08-25): AGENTS.md "Comments and prose" rule 2 — the shortest
# form that carries the reason — had no armed check, and three consecutive PRs
# shipped 10-to-30-line doc blocks that duplicated the unit ledger. The manual
# form of this scan lives in .agents/skills/rust-code-quality/SKILL.md (scan 10);
# nothing invoked it. This is that scan, armed.
#
# DIFF-SCOPED, deliberately: AGENTS.md scopes the rule to "what the change adds
# or touches" and states that a sweep of the existing tree is its own unit. A
# tree-wide gate could not go green without that sweep, so it would have to be
# advisory — and an advisory gate is one nobody fixes.
#
# The base ref is BASE_REF, else origin/main. An unresolvable base is a HARD
# ERROR, never a skip: a gate that silently passes when it cannot see the diff
# is the false green the vacuity doctrine forbids.
#
# Exempt runs: a markdown table (any '|'), a section banner ('====' / '----').
# A table is a closed domain in its shortest form; prose restating it is longer.
#
# Rustdoc scaffolding — a bare '///' separator and a '# Errors' / '# Panics' /
# '# Notes' / '# Examples' heading — does not count toward the total. AGENTS.md
# REQUIRES those sections, so counting them would cap the prose at four lines and
# pay for a heading by deleting an error contract. The cap is on prose.

set -euo pipefail
cd "$(dirname "$0")/.."

MAX_LINES="${MAX_LINES:-6}"
BASE_REF="${BASE_REF:-origin/main}"

# The detector reads `git diff -U0` on stdin so the self-test can feed it a
# synthetic diff. Only ADDED lines count: a run broken by context or a removal
# is two runs.
detect() {
  awk -v max="$MAX_LINES" '
    function flush() {
      if (n > max && !exempt) printf "%s: comment block of %d added lines (max %d)\n", loc, n, max
      n = 0; exempt = 0
    }
    /^\+\+\+ /  { flush(); file = substr($0, 7); next }
    /^@@/       { flush(); next }
    /^\+/       {
      body = substr($0, 2)
      sub(/^[ \t]+/, "", body)
      if (body ~ /^\/\//) {
        if (n == 0) loc = file
        text = body
        sub(/^\/\/[\/!]?[ \t]*/, "", text)
        if (text != "" && text !~ /^# (Errors|Panics|Notes|Examples|Safety)$/) n++
        if (body ~ /\|/ || body ~ /====/ || body ~ /----/) exempt = 1
        next
      }
      flush(); next
    }
                { flush() }
    END         { flush() }
  '
}

# --- self-test: the detector must flag what it claims to, and only that -------
self_test() {
  local out
  # Rustdoc scaffolding is free: 6 prose lines + separators + a heading passes
  out="$(printf '+++ b/a.rs\n@@\n+/// 1\n+/// 2\n+/// 3\n+///\n+/// # Errors\n+///\n+/// 4\n+/// 5\n+/// 6\n' | detect)"
  if [ -n "$out" ]; then
    echo "ERROR: self-test FAILED — rustdoc scaffolding was counted as prose: $out" >&2
    return 1
  fi
  # ...but seven PROSE lines still flag, scaffolding or not
  out="$(printf '+++ b/a.rs\n@@\n+/// 1\n+/// 2\n+/// 3\n+///\n+/// # Errors\n+///\n+/// 4\n+/// 5\n+/// 6\n+/// 7\n' | detect)"
  if [ -z "$out" ]; then
    echo "ERROR: self-test FAILED — 7 prose lines under a heading were not flagged" >&2
    return 1
  fi
  # 7 added comment lines -> must flag
  out="$(printf '+++ b/a.rs\n@@\n+// 1\n+// 2\n+// 3\n+// 4\n+// 5\n+// 6\n+// 7\n' | detect)"
  if [ -z "$out" ]; then
    echo "ERROR: self-test FAILED — a 7-line block was not flagged; the gate is vacuous" >&2
    return 1
  fi
  # 6 added comment lines -> must NOT flag (the threshold is inclusive)
  out="$(printf '+++ b/a.rs\n@@\n+// 1\n+// 2\n+// 3\n+// 4\n+// 5\n+// 6\n' | detect)"
  if [ -n "$out" ]; then
    echo "ERROR: self-test FAILED — a 6-line block was flagged: $out" >&2
    return 1
  fi
  # A run split by a code line is two runs, not one 8-line run
  out="$(printf '+++ b/a.rs\n@@\n+// 1\n+// 2\n+// 3\n+// 4\n+let x = 1;\n+// 5\n+// 6\n+// 7\n+// 8\n' | detect)"
  if [ -n "$out" ]; then
    echo "ERROR: self-test FAILED — a split run was counted as one block: $out" >&2
    return 1
  fi
  # A markdown table is exempt even at 8 lines
  out="$(printf '+++ b/a.rs\n@@\n+// 1\n+// 2\n+// | a | b |\n+// 4\n+// 5\n+// 6\n+// 7\n+// 8\n' | detect)"
  if [ -n "$out" ]; then
    echo "ERROR: self-test FAILED — an exempt table block was flagged: $out" >&2
    return 1
  fi
}
self_test

if ! base_sha="$(git rev-parse --verify --quiet "$BASE_REF^{commit}")"; then
  echo "ERROR: base ref '$BASE_REF' does not resolve — cannot scope the scan." >&2
  echo "Fetch it (git fetch origin main) or set BASE_REF to a reachable commit." >&2
  exit 1
fi
if ! merge_base="$(git merge-base "$base_sha" HEAD)"; then
  echo "ERROR: no merge base between '$BASE_REF' and HEAD — cannot scope the scan." >&2
  exit 1
fi

hits="$(git diff -U0 "$merge_base" -- 'crates/*.rs' | detect || true)"

if [ -n "$hits" ]; then
  echo "ERROR: comment blocks over $MAX_LINES added lines:" >&2
  echo "$hits" >&2
  echo >&2
  echo "AGENTS.md 'Comments and prose': use the shortest form that carries the" >&2
  echo "reason. Bytecode offsets and decode narrative belong in the unit ledger" >&2
  echo "under task/; capability status belongs in the GAP_MATRIX row." >&2
  exit 1
fi

echo "OK: no added comment block over $MAX_LINES lines (base ${BASE_REF}, self-tested)."
