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

# F-INTEROP-NIGHTLY-GREEN-1 — the weekly Java-interop run back to 65 of 65

**Date:** 2026-10-17. **Base:** fork `main` `409ff330e`. **Model:** Claude Opus 5.5
(`claude-opus-5-5`), high effort.

The Nightly Interop workflow (`.github/workflows/nightly_interop.yml`, Saturdays 06:43 UTC) failed
every week from 2026-08-29. Run 39011689293 (2026-10-17) passed 61 of 65 suites. Four suites failed:

| Suite | Red since | Cause | PR |
|---|---|---|---|
| `run-interop-validate-append-only.sh` | 08-29 | test infrastructure: writer builder | A |
| `run-interop-pr5a-catalog-commit-decode.sh` | 09-05 | test infrastructure: missing jar | A |
| `run-interop-remove-dangling.sh` | 09-26 | see PR B | B |
| `run-interop-v3-maintenance.sh` | 09-19 | see PR C | C |

The unit ships as three pull requests. Each branches from the same `main`, so each one reviews
alone. This ledger carries one section per pull request.

This ledger retires when the three pull requests land and the weekly run reports 65 of 65.

## PR A — `fix/interop-nightly-green-1`: infrastructure and the REST fixture WAL line

### A1 — the `iceberg-aws` jar for the pr5a decode suite

- **Cause.** The suite runs `javap` on
  `~/.m2/repository/org/apache/iceberg/iceberg-aws/1.10.0/iceberg-aws-1.10.0.jar`. No pom in
  `dev/java-interop` depends on `iceberg-aws`, so the workflow's `dependency:go-offline` priming
  never fetches it. The hosted runner starts with an empty `~/.m2`, and the suite fails in 0 s with
  `FAIL missing iceberg-aws 1.10.0 jar`. A developer box that once resolved the jar passes.
- **Fix.** The priming step reads `iceberg.version` from `dev/java-interop/pom.xml` (the one
  source; a `test -n` stops the step if the read returns nothing), then runs
  `mvn dependency:get -Dartifact=org.apache.iceberg:iceberg-aws:<version> -Dtransitive=false`.
- **Rejected alternative.** A pom dependency on `iceberg-aws` pulls the AWS SDK closure into the
  offline resolution of every suite. Nothing on the runner image stops the `-Dtransitive=false`
  form, so the pom route has no reason to exist.
- **Suite message.** The prerequisite `FAIL` line now names the fetch command.
- **Map.** `dev/java-interop/map.md` said the pr5a suite does not match the `run-interop-*.sh`
  glob. That was stale: PR-7 renamed it into the glob (`ccaf527d0`), and the weekly run has
  discovered it since 09-05. The map now says so and names where the jar comes from.

**Evidence (2026-10-17, this box).**

- `dependency:get` with the workflow's exact arguments into an EMPTY throwaway local repository
  (`-Dmaven.repo.local=<clone>/target/...`): exit 0. The repository then holds
  `org/apache/iceberg/iceberg-aws/1.10.0/iceberg-aws-1.10.0.jar` (222538 bytes) and its pom, and
  NO `software/amazon` tree. No transitive SDK came down.
- The suite reads `${HOME}/.m2`, so `HOME` is its override:
  - `HOME=<empty dir>`: `FAIL missing iceberg-aws 1.10.0 jar; fetch it with: ...`, exit 1. This
    is the hosted-runner state before the fix.
  - `HOME=<dir whose .m2 holds only the dependency:get jar and the iceberg-core jar>`:
    `pr5a-catalog-commit-decode: 12 needles`, `OK`, exit 0. `javap` does not resolve the SDK
    classes it names, so the jar alone is enough.

### A2 — the validate-append-only Rust mirror declares its partitioning

- **Cause.** Since `2c50c6bfa` (2026-08-28) `DataFileWriterBuilder::build` refuses a writer that
  has neither `.unpartitioned()`, `.with_partition_spec(...)` nor a `PartitionKey`. The mirror's
  `write_file` helper called `.build(partition_key.cloned())` for both table shapes, so the first
  unpartitioned case panicked at `interop_validate_append_only.rs:293`. The Java leg passed 4 of 4.
- **Fix.** `write_file` branches on `partition_key`: `Some(key)` builds with `Some(key)` (the
  `interop_v3_maintenance.rs` form), `None` builds `.unpartitioned().build(None)` (the
  `engine_contract_isolation_recipes.rs` form). Nothing else in the test changed.

**Evidence (2026-10-17).**

- `origin/main` test source, same Java-written case directory:
  `panicked at crates/iceberg/tests/interop_validate_append_only.rs:293:10: build data file
  writer: DataInvalid => writer was built with neither a PartitionSpec nor a PartitionKey`.
  `test result: FAILED. 0 passed; 1 failed`.
- With the fix, `scripts/run_interop_suites.sh --only
  run-interop-validate-append-only.sh,run-interop-pr5a-catalog-commit-decode.sh`: both PASS
  (`TOTAL: 2 passed, 0 failed, 2 run, 65 discovered (floor 65)`). The append-only suite shows
  `verify-interop-validate-append-only: 0 failures` (Java), `test result: ok. 1 passed` (Rust),
  and both sabotage legs fail closed (`FAIL validate-append-only[matching_partition]: outcome
  THROW but expected COMMIT` on each side).

### A3 — the REST fixture keeps its catalog in a WAL-mode SQLite file

- **Cause.** The `rest` service in `dev/docker-compose.yaml` ran its JDBC catalog on
  `jdbc:sqlite:file:/tmp/iceberg_rest_mode=memory`. Under parallel commits SQLite answered
  `SQLITE_BUSY` (seen in the ICE-CI-RUSTFS-1 verification).
- **Fix.** `CATALOG_URI=jdbc:sqlite:file:/tmp/iceberg_rest.db?journal_mode=WAL`, the upstream
  line from `6262dffd7`. WAL lets readers run beside one writer. `dev/map.md` names the setting
  and the `SQLITE_BUSY` symptom.

**Evidence (2026-10-17, rootless Docker).** `make docker-up`: exit 0, every service healthy
(`dev-rest-1 Up ... (healthy)`), `GET /v1/config` answers. Inside the `rest` container `/tmp`
holds `iceberg_rest.db`, `iceberg_rest.db-wal` and `iceberg_rest.db-shm`, so the catalog runs
in WAL mode. `make docker-down`: exit 0.
