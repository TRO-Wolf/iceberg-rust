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

# `DeleteOrphanFiles` — Java provenance and the 1.10.0 pin

Routed here from `crates/iceberg/src/maintenance/delete_orphan_files.rs` during the 2026-08-26
comment sweep. AGENTS.md "Comments and prose" sends decode evidence to the unit ledger; the module
doc keeps one pointer. This is an EVIDENCE-CLASS record: it says which facts rest on 1.10.0
bytecode and which rest on tagless source. Do not let it rot — if a fact moves class, move it here.

The action class lives in the Spark module and **no 1.10.0 Spark bytecode is available locally**,
so the algorithm is ported from tagless `MAIN` source
(`spark/v4.0/.../DeleteOrphanFilesSparkAction.java`). Every load-bearing helper it delegates to
does live in `iceberg-core` / `iceberg-api` 1.10.0 and was bytecode-verified.

## Bytecode-verified against 1.10.0

| Fact | Java source |
|---|---|
| `PrefixMismatchMode` = `{ERROR, IGNORE, DELETE}` + `fromString` | `DeleteOrphanFiles$PrefixMismatchMode` (api, javap) |
| Scheme/authority match: valid is null/empty OR `valid.equalsIgnoreCase(actual)` | `FileURI.uriComponentMatch` (core, javap) |
| Hidden-path rule (`_`/`.` prefix) + the partition-aware exception `_<field>=` | `HiddenPathFilter.accept`, `FileSystemWalker$PartitionAwareHiddenPathFilter.forSpecs` (core, javap) |
| Valid-file universe: all content files of all snapshots, all manifests, manifest lists, current + previous `metadata.json`, version-hint, statistics + partition statistics | `ReachableFileUtil` (core) + `BaseSparkAction.{contentFileDS,manifestDS,manifestListDS,otherMetadataFileDS}` (MAIN) |

## MAIN-only — pinned to tagless source, NOT to 1.10.0 bytecode

Everything absent from the table above. Concretely: the default `olderThan` = `now − 3 days`; the
`EQUAL_SCHEMES_DEFAULT = {"s3n,s3a": "s3"}` constant and its `putAll(defaults); putAll(user)` merge
order; the GC-gate `ValidationException` message; the ERROR-mode conflict message; the path-only
join key; and the valid-file-universe composition (`validFileIdentDS`'s union). The
`BaseSparkAction` dataset builders these compose are `iceberg-spark` (MAIN) too.

`FindOrphanFiles.toOrphanFile` is MAIN, but its only non-trivial primitive (`uriComponentMatch`) is
bytecode-verified above.
