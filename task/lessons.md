<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# Lessons

Accumulated DO / DO NOT lessons. The operating manuals ([skills/](../skills/)) require reading this
file **in full at the start of every session**, and appending to it after **any** correction from
the user.

How to use it (see the manuals' §2):

- After any correction, append a **date-stamped** entry immediately.
- Write each as a concrete **DO** or **DO NOT** statement with the *why* and how to apply it.
- Supersede an outdated rule with a dated note (`_superseded YYYY-MM-DD: see ..._`) rather than
  editing the original in place.

> **Compaction log.** Last pass: 2026-06-13 (pass 5 — Wave-6/Wave-7 compaction, 959 lines;
> 1 KEEP-added / 17 ARCHIVE / 0 promoted) →
> [lessons-archive/2026-06_wave6-wave7.md](lessons-archive/2026-06_wave6-wave7.md). All 17 dated
> increment narratives (Wave-4 F2 → Wave-7 R3, incl. 8 nested REVIEWER sub-sections) archived
> verbatim; one new cross-cutting KEEP added (squash-merge content validation). Promotion candidate
> flagged in this pass's notes (the SKIP-false-green sabotage rule), user-approval-gated. Prior
> passes: 2026-06-12 (pass 4 — post-Wave-5 merge union, 4/25/2) →
> [lessons-archive/2026-06_wave5.md](lessons-archive/2026-06_wave5.md);
> 2026-06-12 (pass 3 — 14/47/3) →
> [lessons-archive/2026-06_wave3-wave4-overnight.md](lessons-archive/2026-06_wave3-wave4-overnight.md);
> 2026-06-11 (pass 2 — 17/25/6); 2026-06-09 (pass 1 — 31 promoted). Archives are not read by
> default — see [skills/compaction.md](../skills/compaction.md).

---

<!-- Newest entries at the bottom. Example shape:

### YYYY-MM-DD
- **DO** carry context on every fallible Rust call (`.with_context(...)` / `.expect("msg")`).
  *Why:* a bare `.unwrap()` panic gives the operator no cause from logs alone.
- **DO NOT** edit upstream crate files to land a fork feature when an additive module would do.
  *Why:* it makes the next upstream merge conflict-prone. Prefer additive changes.
-->

### 2026-06-13 — Prove a squash-merged branch's content landed before pruning it (squash-merge defeats `--is-ancestor`)

- **DO, before deleting a branch/worktree that was merged into the trunk via SQUASH-merge, prove its
  content landed with `git diff <branch-tip> <squash-commit>` — an EMPTY diff means every line is in
  the trunk and the branch is safe to prune.** *Why:* a squash-merge replays the branch's combined
  changes as ONE brand-new commit with NO parent link to the branch, so the branch tip is NOT an
  ancestor of the trunk. The usual safety checks then LIE: `git merge-base --is-ancestor <tip> main`
  returns false (non-zero), `git branch --merged` omits the branch, and `git cherry main <branch>`
  reports every commit as `+` (unmerged) — all three imply "you'd lose work by deleting this," when
  in fact the work is fully present under a different commit identity. The content diff is the truth;
  ancestry is not. (Came up 2026-06-13 pruning the merged Wave-6/Wave-7 worktrees: `--is-ancestor`
  flagged them as unmerged, but `git diff wt-tip <squash-sha>` was empty for each, confirming a safe
  prune.) *Apply:* for a squash-merged branch use the content diff, not ancestry, as the
  delete-safety gate; reserve `--is-ancestor` for true (non-squash) merges where the parent link
  exists. If the diff is NON-empty, the branch carries commits the squash did not capture — stop and
  reconcile before deleting.
