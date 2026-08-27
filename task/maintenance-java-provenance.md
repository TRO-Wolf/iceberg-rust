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

# Maintenance actions — Java provenance and the 1.10.0 pin

Routed here from module docs during the 2026-08-26 comment sweep. AGENTS.md "Comments and prose"
sends decode evidence to the unit ledger. These are EVIDENCE-CLASS records: what rests on 1.10.0
bytecode, and what rests on source the jar does not contain. Sibling:
[delete-orphan-files-java-provenance.md](delete-orphan-files-java-provenance.md).

## `DeleteReachableFiles` (`maintenance/delete_reachable_files.rs`)

The action interface + its result shape are `iceberg-api` 1.10.0 (javap-verified):

- `DeleteReachableFiles` (api 1.10.0) extends `Action<DeleteReachableFiles, Result>` with
  `deleteWith(Consumer<String>)`, `executeDeleteWith(ExecutorService)`, and `io(FileIO)`. The
  entry point (`SparkActions.deleteReachableFiles(String metadataLocation)`) takes the metadata
  LOCATION as a `String` — mirrored here as [`DeleteReachableFiles::new`]'s `&str`.
- `DeleteReachableFiles$Result` (api 1.10.0, javap-verified) is the six `long` counts mirrored
  1:1 by [`DeleteReachableFilesResult`]: `deletedDataFilesCount`,
  `deletedEqualityDeleteFilesCount`, `deletedPositionDeleteFilesCount`, `deletedManifestsCount`,
  `deletedManifestListsCount`, `deletedOtherFilesCount`.
- The reachable-file universe is `ReachableFileUtil` (core 1.10.0, javap-verified) —
  `metadataFileLocations(table, recursive)` (current + previous `metadata.json`),
  `manifestListLocations(table)`, `statisticsFilesLocations(table)`, `versionHintLocation(table)`
  — PLUS a scan of every snapshot's `allManifests` for the data/delete file paths (Java
  `DeleteReachableFilesSparkAction` composes `contentFileDS ∪ manifestDS ∪ manifestListDS ∪
  allReachableOtherMetadataFileDS`). The content-file walk reads EVERY manifest entry (incl.
  `DELETED` tombstones) of EVERY snapshot — a tombstoned file is still a physical file the table
  wrote, so it is reachable and must be deleted.

The action CLASS itself lives in the Spark module (no 1.10.0 Spark bytecode is available
locally), so the categorization-into-counts and the metadata-location entry shape are pinned to
the tagless `DeleteReachableFilesSparkAction.java` MAIN source; every load-bearing helper it
delegates to (above) is the bytecode-verified `iceberg-core` / `iceberg-api` 1.10.0 surface. This
mirrors [`DeleteOrphanFiles`](crate::maintenance::DeleteOrphanFiles)'s provenance split exactly.

## `ComputePartitionStats` (`maintenance/partition_stats.rs`)

# Java provenance (1.10.0 JAR BYTECODE, not the /tmp MAIN-source checkout)

The 1.10.0 jar's `PartitionStatsHandler` holds the stats schema as field-id constants directly
and uses a concrete `PartitionStats` class. The MAIN-source checkout under `/tmp/iceberg-java-ref`
is POST-1.10.0 — it was refactored to a `PartitionStatistics` interface + `BasePartitionStatistics`
with a `DV_COUNT` backward-compat guard in `appendStats` the jar does not have. Per the repo
lesson "bytecode outranks MAIN source for version-sensitive claims," every fact below is
disassembled from the 1.10.0 jar.

## Schema field ids (these become the ON-DISK parquet field ids in X2 — they MUST match Java)


## `ReplacePartitions` (`transaction/replace_partitions.rs`)

Three bytecode listings routed out of the module doc: `dataSpec()`'s two `checkState` calls,
`apply`'s `fields().isEmpty()` branch, and `validate`'s two overload families. The CLAIMS they
supported survive in that module's prose — in particular **`apply` is stricter than `validate`;
do not harmonize the two predicates.** Recover the listings from `git show` at the pre-sweep tip if
a future unit needs to re-derive them.
