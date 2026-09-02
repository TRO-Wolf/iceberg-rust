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

# map.md — dev/

## Purpose

Developer tools: Java interop oracles, Spark provisioner, and credentialed catalog runners.

## Contents

| Path | What it does |
|---|---|
| `java-interop/` | Java 1.10.0 oracle harnesses. See `java-interop/map.md`. |
| `pr5a-catalog-commit-outcomes.sh` | Credentialed PR-5A Glue + S3 Tables commit-outcome runner. Hard-fails unless `ICEBERG_PR5A_CREDENTIALED=1` and required config is present. Does not print credentials or object-store URLs. |
| `spark/` | Spark provisioner for interop tables. |
| `docker-compose.yaml` | Local REST / MinIO / HMS fixtures. |

## I want to…

| Intent | Go to |
|---|---|
| Decode Java Glue / REST commit outcomes | `java-interop/run-interop-pr5a-catalog-commit-decode.sh` |
| Run credentialed catalog commit smokes | `pr5a-catalog-commit-outcomes.sh` |
| Run a table-format interop oracle | `java-interop/map.md` |

## Pointers

- Up: repository root
- Related: `crates/catalog/glue/map.md`, `crates/catalog/s3tables/map.md`, `task/pr5a-catalog-commit-outcomes-ledger.md`

## Debug

### Known failure modes

| Symptom | Likely cause | First check |
|---|---|---|
| Runner exits 1 with CREDENTIALED must be 1 | Env not armed | Export `ICEBERG_PR5A_CREDENTIALED=1` |
| Runner exits 1 credentials absent | No AWS credential source | Access key, profile, container, or web identity — do not log values |
| javap decode fixture count mismatch | Iceberg jar drift | Confirm 1.10.0 jars under `~/.m2` |

### First checks

- `bash dev/java-interop/run-interop-pr5a-catalog-commit-decode.sh`
- Do not run the credentialed runner without the owner-approved AWS boundary.

### Escalate to

`docs/tier2-aws.md` on the RePark side, `task/pr5a-catalog-commit-outcomes-ledger.md`
