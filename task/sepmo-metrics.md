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

# SEPMO metrics ledger

The location bound by [skills/sepmo/binding-manifest.md](../skills/sepmo/binding-manifest.md)
(`metrics_ledger_location`). **One section per retrospective** — charter-close or incident — in
the canonical `METRICS` format owned by
[skills/sepmo/references/08-retrospective.md](../skills/sepmo/references/08-retrospective.md)
(*Step 2 — the metrics ledger*), including `environment_drift_events` (spine v2.1+). Append-only;
sections are never rewritten, only superseded by later sections.

Created 2026-07-13 with the canon v2.2 re-instantiation (`infra/sepmo-canon-v2.2`). No
retrospective has been filed against the v2.2 ledger yet — the first section lands with the first
charter close or incident after this install. Pre-v2.2 history is recorded in
[task/todo.md](todo.md) unit closes and [task/lessons.md](lessons.md); it is not retrofitted here
(no fabricated metrics).

**Standing candidate for the first section's `environment_drift_events`:** the 2026-07-11 first
nightly-interop run failure (GitHub Actions run 29144349415) with the identical `make interop`
green locally on the same ref — base-ref-proven environmental per R10, cause on the CI runner
still undiagnosed (step-summary evidence needs admin access).

---

<!-- METRICS sections append below this line. -->
