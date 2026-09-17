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

# F-RTAS-OPS-1 ledger — RTAS records overwrite/delete, not append

Model: muse-spark-1.3-contributor | Run 20c, 2026-09-17 | Base 75da2b58

## Finding

RePark `CREATE OR REPLACE TABLE AS SELECT` over an existing table commits
`append, append`. Spark 4.1.2 + Iceberg 1.11.0 records `append, overwrite`.
Oracle: `/tmp/oc-worker/ic-build/write_fidelity_spark.json`, key `rtas_ops`.

## Oracle cells (Spark 4.1.2 + Iceberg 1.11.0)

- `ctas_then_rtas`: CTAS → `append`; RTAS (100 rows) → `overwrite`,
  summary `added-data-files 5, added-records 100, total-records 100,
  total-data-files 5`, NO `deleted-*` keys.
- `rtas_new_table`: RTAS on missing table → `overwrite` (10 rows).
- `rtas_empty_new`: RTAS empty SELECT on new table → `delete`, no `added-*`.
- `rtas_empty_twice`: again → `delete`, `delete`.
- CTAS of rows → `append` (unchanged).

## Java reason (to verify)

Spark RTAS writes through `OverwriteByExpression(alwaysTrue)` on the staged
replace transaction (`newOverwrite()`); operation is `overwrite` with added
files, `delete` with none. CTAS writes through `newAppend()`. Bytecode
targets: `org.apache.iceberg.BaseOverwriteFiles.operation`,
`org.apache.iceberg.spark.source.SparkWrite$OverwriteByFilter`.
Verification record below.

## Clauses

- C-001: staged commit with replace-write semantics: `overwrite` with added
  files, `delete` with none; plain create keeps `append`; empty CTAS behavior
  measured, unchanged without a Java reason.
- C-002: replace-semantics API reachable for new and existing tables.
- C-003: replace snapshot summary key set equals Java (no `deleted-*`,
  `changed-partition-count`, `total-*` as Java).
- C-004: existing callers/tests unchanged (CTAS, staged publish/reload, Glue).

## Steps

1. Ledger skeleton. Commit. (this file)
2. Red-first tests beside staged_table tests. Run on base, paste reds. Commit.
3. Implement C-001..C-003. Commit.
4. GAP_MATRIX row, maps, gate output. Gates green. Commit.

## Bytecode verification

PENDING.

## Red run (base tree)

PENDING.

## Implementation notes

PENDING.

## Gate output

PENDING.
