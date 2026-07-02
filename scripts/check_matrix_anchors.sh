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
# check_matrix_anchors.sh — GAP_MATRIX structural integrity + stable row
# anchors.
#
# Motivation (2026-07-01, review follow-up 2): "row N" citations used to be
# raw file line numbers; two row insertions between 2026-06-17 and 2026-07-01
# shifted every downstream citation by +2 and silently broke ~25 of them.
# Every capability row now carries a PERMANENT anchor in its first cell
# ('| R<id> · ...') and live docs cite "row R<id>". This gate enforces:
#   1. the 5-pipe row audit (was a manual CLAUDE.md convention);
#   2. every data row anchored exactly once ('| R<id> · ');
#   3. anchor IDs unique (they are permanent and never reused);
#   4. every 'row R<n>' citation in the LIVE docs resolves to an anchor.
# Dated archives (docs/parity/archive/) are excluded from check 4 — they
# reference historical epochs. If a row is ever DELETED, citations to its
# anchor fail here by design: retarget or annotate them, then delete.
#
# Wired into `make check-matrix-anchors` and CI.

set -euo pipefail
cd "$(dirname "$0")/.."

MATRIX="docs/parity/GAP_MATRIX.md"

fail=0

# --- 1. pipe audit: every table line carries exactly 5 '|' characters -------
bad_pipes="$(awk '/^\|/ { c=gsub(/\|/,"|"); if (c!=5) print FILENAME":"FNR": "c" pipes" }' "$MATRIX")"
if [ -n "$bad_pipes" ]; then
  echo "ERROR: GAP_MATRIX table rows must carry exactly 5 '|' characters:" >&2
  echo "$bad_pipes" >&2
  fail=1
fi

# --- 2. every data row anchored ('| R<id> · '); header/separator exempt -----
unanchored="$(grep -nE '^\|' "$MATRIX" | grep -vE '^[0-9]+:\|---' | grep -vE '^[0-9]+:\| Area \|' | grep -vE '^[0-9]+:\| R[0-9]+ · ' || true)"
if [ -n "$unanchored" ]; then
  echo "ERROR: unanchored GAP_MATRIX data rows (first cell must start 'R<id> · '):" >&2
  echo "$unanchored" | cut -c1-120 >&2
  fail=1
fi

ids_file="$(mktemp)"
trap 'rm -f "$ids_file"' EXIT
grep -oE '^\| R[0-9]+ · ' "$MATRIX" | grep -oE '[0-9]+' | sort -n >"$ids_file" || true

# --- 3. anchor IDs unique ----------------------------------------------------
dups="$(uniq -d <"$ids_file")"
if [ -n "$dups" ]; then
  echo "ERROR: duplicate GAP_MATRIX row anchors:" >&2
  # shellcheck disable=SC2086
  printf 'R%s\n' $dups >&2
  fail=1
fi

# --- 4. every 'row R<n>' citation in the live docs resolves ------------------
# Live doc set: the plan, the conventions, the live task file, and docs/
# (minus the dated archives). Tracked files only (git grep, worktree content).
citations="$(git grep -hoiE 'rows? R[0-9]+([/,] ?R[0-9]+)*' -- \
  'Roadmap.md' 'CLAUDE.md' 'task/todo.md' 'docs' ':(exclude)docs/parity/archive' \
  2>/dev/null | grep -oE 'R[0-9]+' | sort -u || true)"
for tok in $citations; do
  if ! grep -qx "${tok#R}" "$ids_file"; then
    echo "ERROR: citation 'row $tok' does not resolve to any GAP_MATRIX anchor" >&2
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo >&2
  echo "GAP_MATRIX anchor integrity failed — see errors above." >&2
  exit 1
fi

echo "OK: GAP_MATRIX anchors sound ($(wc -l <"$ids_file" | tr -d ' ') rows anchored, IDs unique, citations resolve, 5-pipe audit green)."
