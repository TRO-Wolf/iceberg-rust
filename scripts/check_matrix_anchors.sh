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
# raw file line numbers. Two PROSE lines added ABOVE the table (the 2026-06-19
# provenance refresh) shifted every row by +2 and silently broke ~45 downstream
# citations, discovered in four separate waves — ANY line inserted above a row
# breaks its line-number citations, row or prose. Every capability row now
# carries a PERMANENT anchor in its first cell ('| R<id> · ...') and live docs
# cite "row R<id>". This gate enforces:
#   1. the 5-pipe row audit (was a manual CLAUDE.md convention);
#   2. every data row anchored exactly once ('| R<id> · ');
#   3. anchor IDs unique (they are permanent and never reused);
#   4. every 'row R<n>' citation in the LIVE docs resolves to an anchor.
# Dated archives (docs/parity/archive/) are excluded from check 4 — they
# reference historical epochs. If a row is ever DELETED, citations to its
# anchor fail here by design: retarget or annotate them, then delete.
#
# ASSUMPTION: GAP_MATRIX.md contains exactly ONE table (header '| Area |...').
# Checks 1-2 apply to every '^|' line in the file; adding a second table
# requires updating this script first (scope the checks to the table's range).
#
# Check 4 scans TRACKED files (git grep): a citation in a not-yet-added file
# is only caught once the file is staged/committed — CI always sees committed
# state, so the gate holds where it matters.
#
# KNOWN LIMITS of the line-based citation regex (no instance exists today,
# verified 2026-07-01): a citation WRAPPED across a line break validates only
# its first line's anchors, and separators outside the supported set
# (',' '/' '-' '–' 'and') leave trailing anchors unchecked — keep list
# citations on one line and use the supported separators.

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
  echo "ERROR: unanchored GAP_MATRIX data rows (first cell must start 'R<id> · ';" >&2
  echo "       if you are adding a SECOND table to the file, update this script first):" >&2
  echo "$unanchored" | cut -c1-120 >&2
  fail=1
fi

ids_file="$(mktemp)"
cites_raw="$(mktemp)"
err_file="$(mktemp)"
trap 'rm -f "$ids_file" "$cites_raw" "$err_file"' EXIT
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
# Live doc set: the plan, the conventions, the live task file, docs/ (minus
# the dated archives), and crates/ (source/test comments cite matrix rows too
# — ~26 of them were migrated from stale bare line numbers on 2026-07-08).
# The span regex requires a non-alphanumeric char (or line start) before
# 'row' so prose like 'sparrow R6' cannot false-match, and follows
# list/range continuations ('R122/R123', 'R122, R123', 'R122-R124',
# 'R122 and R123') so a dead TRAILING anchor cannot hide. The anchor 'R' is
# matched CASE-SENSITIVELY ('row R122' / 'Rows R122', never 'rows r1'):
# widening the scan to crates/ surfaced test prose naming fixture rows
# 'rows r1 (id=1...)' — a false positive under the old case-insensitive
# grep. The citation convention is an uppercase 'R<id>'; a lowercase
# 'row r122' is NOT a citation and is deliberately not validated.
# A renamed/deleted scan target would silently drop out of a bare git-grep
# pathspec (exit 1, indistinguishable from no-match), so assert each first —
# this existence list and the git-grep pathspec below MUST stay in lockstep.
for f in Roadmap.md CLAUDE.md task/todo.md docs crates; do
  if [ ! -e "$f" ]; then
    echo "ERROR: live-doc scan target '$f' does not exist — update this script's scan set" >&2
    exit 1
  fi
done
CONT='(, ?| ?/ ?| and | ?- ?| ?– ?)'
# git grep exit 0 = citations found, 1 = none, >= 2 = git itself failed —
# a hard error, NOT "zero citations" (the false-green class the artifact
# gate's v3 hardening closed; same doctrine here).
rc=0
git grep -hoE "(^|[^[:alnum:]])[Rr]ows? R[0-9]+(${CONT}R[0-9]+)*" -- \
  'Roadmap.md' 'CLAUDE.md' 'task/todo.md' 'docs' 'crates' \
  ':(exclude)docs/parity/archive' \
  >"$cites_raw" 2>"$err_file" || rc=$?
if [ "$rc" -ge 2 ]; then
  echo "ERROR: git grep failed (exit $rc) while scanning citations — cannot certify:" >&2
  cat "$err_file" >&2
  exit 1
fi
citations="$(grep -oiE 'R[0-9]+' "$cites_raw" | tr '[:lower:]' '[:upper:]' | sort -u || true)"
for tok in $citations; do
  if ! grep -qx "${tok:1}" "$ids_file"; then
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
