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
# longer than MAX_LINES. This is scan 10 of .agents/skills/rust-code-quality,
# armed: AGENTS.md "Comments and prose" rule 2 had no check, and three PRs
# shipped 10-to-30-line doc blocks duplicating the unit ledger.
#
# DIFF-SCOPED because AGENTS.md scopes the rule to what the change touches, and
# a tree-wide gate could not go green without the sweep it calls its own unit.
# It counts ADDED runs only, so an edit inside an existing long block passes.
#
# An unresolvable base ref is a HARD ERROR: a gate that passes when it cannot
# see the diff is a false green. On a push to main the merge base is HEAD, so
# the scan is empty by construction — this guards PRs.
#
# Exempt: a markdown table (any '|'), a section banner, and the ASF license
# header, which is mandatory and identical in every file. Rustdoc scaffolding (a
# bare '///', a '# Errors' / '# Panics' / '# Notes' heading) does not count
# toward the cap, since AGENTS.md requires those sections.
#
# UNTRACKED FILES ARE SCANNED TOO. `git diff` cannot see a file git does not
# track, so a brand-new file was invisible here until it was staged — a false
# green found 2026-08-25, on the first new file this gate met. Each untracked
# candidate is diffed against /dev/null so every one of its lines reads as
# added.

set -euo pipefail
cd "$(dirname "$0")/.."

err_file="$(mktemp)"
trap 'rm -f "$err_file"' EXIT
MAX_LINES="${MAX_LINES:-6}"
BASE_REF="${BASE_REF:-origin/main}"
diff_pathspec='crates/*.rs'

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
      if (n == 1 && body ~ /Licensed to the Apache Software Foundation/) exempt = 1
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
  # Wiring, not detection: a self-test that only exercises the detector passes
  # while the shipped scan points at a path that matches nothing.
  if [ "$diff_pathspec" != 'crates/*.rs' ]; then
    echo "ERROR: scan pathspec is '$diff_pathspec' — it would miss the product surface" >&2
    return 1
  fi
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
  # The ASF license header is mandatory and identical everywhere: never a finding
  out="$(printf '+++ b/a.rs\n@@\n+// Licensed to the Apache Software Foundation (ASF) under one\n+// 2\n+// 3\n+// 4\n+// 5\n+// 6\n+// 7\n+// 8\n' | detect)"
  if [ -n "$out" ]; then
    echo "ERROR: self-test FAILED — the ASF license header was flagged: $out" >&2
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

tracked_hits="$(git diff -U0 "$merge_base" -- "$diff_pathspec" | detect || true)"

# Untracked candidates: /dev/null as the left side renders every line as added.
untracked_hits=""
while IFS= read -r -d '' candidate; do
  [ -n "$candidate" ] || continue
  # `|| rc=$?` keeps this set -e safe: --no-index exits 1 for "differs", which is the NORMAL
  # case here and would otherwise abort the script before rc is ever read.
  rc=0
  raw="$(git diff --no-index -U0 /dev/null "$candidate" 2>"$err_file")" || rc=$?
  # --no-index exits 0 (identical) or 1 (differs); anything else means it could not read the file,
  # and a scan that cannot see a candidate must not report OK.
  if [ "$rc" -gt 1 ]; then
    echo "ERROR: could not diff untracked candidate '$candidate' (exit $rc):" >&2
    cat "$err_file" >&2
    exit 1
  fi
  found="$(printf '%s' "$raw" | detect || true)"
  if [ -n "$found" ]; then
    untracked_hits="${untracked_hits}${found}"$'\n'
  fi
done < <(git ls-files -z --others --exclude-standard -- "$diff_pathspec")

hits="$(printf '%s\n%s' "$tracked_hits" "$untracked_hits" | sed '/^$/d')"

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
