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

# Apache Iceberg™ Rust — owned parity fork

> **This is an owned fork** of [Apache Iceberg™ Rust](https://github.com/apache/iceberg-rust) — the
> upstream Rust implementation of [Apache Iceberg™](https://iceberg.apache.org/). We maintain it to
> reach **1:1 capability parity with the Java `iceberg-core` / `iceberg-api` library** — the
> engine-agnostic *table-format* core, **not** the Spark engine surface. Upstream is a **sync baseline
> we cherry-pick from, not a mergeability constraint**: we diverge freely in service of parity. The
> deliverable is a **Rust-native library** (Python / PySpark is deferred). **AWS Glue + S3 Tables** are
> the first-priority catalogs.
>
> The original upstream project lives at **[apache/iceberg-rust](https://github.com/apache/iceberg-rust)**,
> and its README is preserved verbatim further down this file (from [Components](#components) onward).

## What this fork adds — a consolidated story

This fork started by syncing the workspace to upstream **`iceberg` 0.9.1** and removing the Python /
PySpark layers, leaving a Rust-only library. From there, parity work has proceeded in small
**Actor-Critic** increments — every change lands with unit tests, and every capability that touches the
on-disk format or the read/write seam lands with a **bidirectional Java ↔ Rust interop round-trip**
(read tables Java wrote; prove Java reads what we write). What has gone in so far:

- **Table-maintenance & write actions** (via the maintenance layer — an opened `ActionsProvider`, plus
  free-standing actions and write-action builder flags) — remove dangling delete files,
  delete-reachable-files drop-table purge, rewrite position-delete files, convert equality deletes to
  positional, compute / update partition statistics, and replace-partitions append-only validation.
- **Scans** — incremental-append and changelog (CDC) scans; JSON expression parsing; and an aggregate
  evaluator (count / min / max over data-file metrics — the scan-planner push-down consumer is
  follow-up work).
- **Spec & types** — the V3 `unknown` primitive type.
- **Merge-on-read** — full delete *application* across position **and** equality deletes, multiple
  delete files per partition, and non-identity partition transforms, interop-proven in both directions
  (the equality × non-identity-transform combination is code-proven by mechanism identity, not
  separately round-tripped).
- **A DataFusion engine reference for the full DML loop** — `INSERT OVERWRITE`, `DELETE`, and `UPDATE`
  through the in-repo `iceberg-datafusion` `TableProvider`, with **both** merge-on-read and
  copy-on-write `DELETE` / `UPDATE` paths selectable via the standard `write.delete.mode` /
  `write.update.mode` table properties. `INSERT OVERWRITE`, `DELETE`, and `UPDATE` are
  partition-aware on both paths (partitioned copy-on-write `DELETE` is additionally
  interop-proven Rust→Java in `interop_partitioned_dml.rs`; merge-on-read requires a V2
  table; V3 deletion-vector writes are a follow-up — see
  [docs/ENGINE_CONTRACT.md](docs/ENGINE_CONTRACT.md) for the authoritative engine surface).
  Row-level mutation
  evaluates the exact engine predicate per row (never an inexact push-down), so it never over-deletes.
- **Catalogs** — REST, Hive Metastore, **Glue + S3 Tables (the parity priority)**, and SQL-backed, plus
  a config-driven loader; OpenDAL-backed FileIO and a Moka object/metadata cache.

**How the work is governed.** This is an owned fork (mergeability with upstream is not a constraint);
the Java repository is the spec-by-example; tests ship with the code; and each change passes an
independent adversarial review before it merges. The authoritative plan and the live, per-capability
status are tracked in-repo — read them rather than relying on this summary, which is intentionally
high-level:

- **[Roadmap.md](Roadmap.md)** — the phase plan and sequencing toward full parity.
- **[docs/parity/GAP_MATRIX.md](docs/parity/GAP_MATRIX.md)** — the single source of truth for each
  capability's status (✅ / 🟡 / ❌), re-audited after every sync and phase.
- **[CLAUDE.md](CLAUDE.md)** — repository intent, prohibitions, and conventions.

---

> **Everything below is the upstream Apache Iceberg™ Rust README, preserved as-is.** It documents the
> base project this fork builds on; for fork-specific status and direction, see the links above.

## Components

The Apache Iceberg Rust project is composed of the following components:

| Name                     | Release                                                         | Docs                                                                                                  |
|--------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| [iceberg]                | [![iceberg image]][iceberg link]                                | [![docs release]][iceberg release docs] [![docs dev]][iceberg dev docs]                               |
| [iceberg-datafusion]     | [![iceberg-datafusion image]][iceberg-datafusion link]          | [![docs release]][iceberg-datafusion release docs] [![docs dev]][iceberg-datafusion dev docs]         |
| [iceberg-catalog-glue]   | [![iceberg-catalog-glue image]][iceberg-catalog-glue link]      | [![docs release]][iceberg-catalog-glue release docs] [![docs dev]][iceberg-catalog-glue dev docs]     |
| [iceberg-catalog-hms]    | [![iceberg-catalog-hms image]][iceberg-catalog-hms link]        | [![docs release]][iceberg-catalog-hms release docs] [![docs dev]][iceberg-catalog-hms dev docs]       |
| [iceberg-catalog-rest]   | [![iceberg-catalog-rest image]][iceberg-catalog-rest link]      | [![docs release]][iceberg-catalog-rest release docs] [![docs dev]][iceberg-catalog-rest dev docs]     |

[docs release]: https://img.shields.io/badge/docs-release-blue
[docs dev]: https://img.shields.io/badge/docs-dev-blue
[iceberg]: crates/iceberg/README.md
[iceberg image]: https://img.shields.io/crates/v/iceberg.svg
[iceberg link]: https://crates.io/crates/iceberg
[iceberg release docs]: https://docs.rs/iceberg
[iceberg dev docs]: https://rust.iceberg.apache.org/api/iceberg/

[iceberg-datafusion]: crates/integrations/datafusion/README.md
[iceberg-datafusion image]: https://img.shields.io/crates/v/iceberg-datafusion.svg
[iceberg-datafusion link]: https://crates.io/crates/iceberg-datafusion
[iceberg-datafusion dev docs]: https://rust.iceberg.apache.org/api/iceberg_datafusion/
[iceberg-datafusion release docs]: https://docs.rs/iceberg-datafusion

[iceberg-catalog-glue]: crates/catalog/glue/README.md
[iceberg-catalog-glue image]: https://img.shields.io/crates/v/iceberg-catalog-glue.svg
[iceberg-catalog-glue link]: https://crates.io/crates/iceberg-catalog-glue
[iceberg-catalog-glue release docs]: https://docs.rs/iceberg-catalog-glue
[iceberg-catalog-glue dev docs]: https://rust.iceberg.apache.org/api/iceberg_catalog_glue/

[iceberg-catalog-hms]: crates/catalog/hms/README.md
[iceberg-catalog-hms image]: https://img.shields.io/crates/v/iceberg-catalog-hms.svg
[iceberg-catalog-hms link]: https://crates.io/crates/iceberg-catalog-hms
[iceberg-catalog-hms release docs]: https://docs.rs/iceberg-catalog-hms
[iceberg-catalog-hms dev docs]: https://rust.iceberg.apache.org/api/iceberg_catalog_hms/


[iceberg-catalog-rest]: crates/catalog/rest/README.md
[iceberg-catalog-rest image]: https://img.shields.io/crates/v/iceberg-catalog-rest.svg
[iceberg-catalog-rest link]: https://crates.io/crates/iceberg-catalog-rest
[iceberg-catalog-rest release docs]: https://docs.rs/iceberg-catalog-rest
[iceberg-catalog-rest dev docs]: https://rust.iceberg.apache.org/api/iceberg_catalog_rest/

## Iceberg Rust Implementation Status

The features that Iceberg Rust currently supports can be found [here](https://iceberg.apache.org/status/).

## Supported Rust Version

Iceberg Rust is built and tested with stable rust, and will keep a rolling MSRV (minimum supported rust version).
At least three months from latest rust release is supported. MSRV is updated when we release iceberg-rust.

Check the current MSRV on [crates.io](https://crates.io/crates/iceberg).

## Contribute

Apache Iceberg is an active open-source project, governed under the Apache Software Foundation (ASF). Apache Iceberg Rust is always open to people who want to use or contribute to it. Here are some ways to get involved.

- Start with [Contributing Guide](CONTRIBUTING.md).
- Submit [Issues](https://github.com/apache/iceberg-rust/issues/new) for bug report or feature requests.
- Discuss
  at [dev mailing list](mailto:dev@iceberg.apache.org) ([subscribe](<mailto:dev-subscribe@iceberg.apache.org?subject=(send%20this%20email%20to%20subscribe)>) / [unsubscribe](<mailto:dev-unsubscribe@iceberg.apache.org?subject=(send%20this%20email%20to%20unsubscribe)>) / [archives](https://lists.apache.org/list.html?dev@iceberg.apache.org))
- Talk to the community directly
  at [Slack #rust channel](https://join.slack.com/t/apache-iceberg/shared_invite/zt-1zbov3k6e-KtJfoaxp97YfX6dPz1Bk7A).

The Apache Iceberg community is built on the principles described in the [Apache Way](https://www.apache.org/theapacheway/index.html) and all who engage with the community are expected to be respectful, open, come with the best interests of the community in mind, and abide by the Apache Foundation [Code of Conduct](https://www.apache.org/foundation/policies/conduct.html).
## Users

- [Databend](https://github.com/datafuselabs/databend/): An open-source cloud data warehouse that serves as a cost-effective alternative to Snowflake.
- [Lakekeeper](https://github.com/lakekeeper/lakekeeper/): An Apache-licensed Iceberg REST Catalog with data access controls.
- [Moonlink](https://github.com/Mooncake-Labs/moonlink): A Rust library that enables sub-second mirroring (CDC) of Postgres tables into Iceberg.
- [RisingWave](https://github.com/risingwavelabs/risingwave): A Postgres-compatible SQL database designed for real-time event streaming data processing, analysis, and management.
- [Wrappers](https://github.com/supabase/wrappers): Postgres Foreign Data Wrapper development framework in Rust.
- [ETL](https://github.com/supabase/etl): Stream your Postgres data anywhere in real-time.

## License

Licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
