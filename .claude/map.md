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

# map.md — .claude/

## Purpose

Claude Code's discovery directory, and nothing else. It holds **one tracked entry**: `skills`, a
symlink to `../.agents/skills`. Claude Code loads skills only from `.claude/skills/`, so without the
symlink the review procedures in [../.agents/skills/](../.agents/skills/map.md) are readable files
that no session can invoke by name.

The symlink is the whole point: the skills keep their single home under `.agents/`, where the
tool-neutral zero-authoritative-facts contract applies to them, and Claude gains native invocation
without a copy that could drift. Any other agent tool that wants the same should add its own symlink
here or in its own discovery directory rather than duplicating a skill.

This directory carries **zero authoritative facts**, the same contract as
[../.agents/](../.agents/map.md). The rules live in [../AGENTS.md](../AGENTS.md); the Claude tool
mechanics live in [../CLAUDE.md](../CLAUDE.md).

## Contents

- `skills` → `../.agents/skills` — symlink (git mode `120000`). Adding a directory under
  `.agents/skills/` makes it invocable in Claude with no change here.
- `settings.local.json`, and any lock or cache file Claude Code writes — untracked runtime state;
  ignore it.

## I want to...

| I want to... | go to |
|---|---|
| Read or edit a skill | [../.agents/skills/map.md](../.agents/skills/map.md) — never through this path |
| Add a new skill | a `<verb-noun>/` directory under `.agents/skills/`; the symlink picks it up |
| Read the Claude tool mechanics | [../CLAUDE.md](../CLAUDE.md) |
| Read the authoritative contract | [../AGENTS.md](../AGENTS.md) |

## Pointers

- **Up:** [../AGENTS.md](../AGENTS.md) (there is no repo-root `map.md`; the spine is the top).
- **Related:** [../.agents/map.md](../.agents/map.md) (the skill sources and the other adapters),
  [../CLAUDE.md](../CLAUDE.md) (the Claude adapter).

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| A skill does not appear in a Claude session | `ls -l .claude/skills` must resolve, and the skill needs a `SKILL.md` with `name` + `description` frontmatter |
| The symlink checked out as a text file | The clone has `core.symlinks=false` (the Windows default); `git config core.symlinks true` and re-checkout |
| `git ls-files -s .claude/skills` shows mode `100644` | It was committed as a regular file — re-create it with `ln -s ../.agents/skills .claude/skills` and re-add; the mode must be `120000` |
| A real file appears in this directory | Bug — it belongs under `.agents/`; this directory holds only the symlink and untracked runtime state |

### First checks

- `ls -l .claude/skills` resolves to `../.agents/skills`, and that directory has a `map.md`.

### Escalate to

- Skill contents / the roster → [../.agents/skills/map.md](../.agents/skills/map.md).
- Precedence / a rule that seems wrong → [../AGENTS.md](../AGENTS.md) `<precedence>`.
