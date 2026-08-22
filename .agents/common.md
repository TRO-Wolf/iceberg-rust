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

# .agents/common.md — shared, tool-neutral entry point

Any agent working in this repo starts here, then reads the authoritative spine. This file carries
**no project rules** — only pointers.

- **The authoritative contract:** [AGENTS.md](../AGENTS.md). Read it first; it holds the read order,
  the precedence chain, the parity mandate, the prohibitions, the engineering rules, the build/test
  commands, and the navigation contract.
- **The plan and the current phase:** [Roadmap.md](../Roadmap.md).
- **Per-capability status (its only home):** [docs/parity/GAP_MATRIX.md](../docs/parity/GAP_MATRIX.md).
- **Testing contract (hard block before any code change):** [docs/testing.md](../docs/testing.md).
- **Per-tier operating manuals + the SEPMO control plane:** [skills/map.md](../skills/map.md).

Tool-specific mechanics (if any) live in the per-tool adapter beside this file. An adapter never
restates an authoritative fact.
