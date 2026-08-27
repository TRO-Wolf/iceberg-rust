// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! MAINTENANCE `RemoveDanglingDeleteFiles` interop, proven against Java's `findDanglingDeletes`
//! semantics WITHOUT Spark, whose `RemoveDanglingDeletesSparkAction` is off the oracle classpath.
//! The proof is three ANTI-CIRCULAR claims. Detection: both sides hand-declare the dangling set
//! from the published spec, and Java's oracle independently recomputes it over the pre-cleanup
//! table. API contract: the removed-file counters and the survivor set match that declaration.
//! Corruption safety: the live row set is IDENTICAL before and after, so nothing is lost or raised.
//!
//! ## The dangling predicate, which both sides hand-declare from
//!
//! Over the current snapshot, group LIVE DATA entries by `(spec_id, partition)` and take
//! `min(data sequence number)`. A delete dangles when `min IS NULL`, or it is a POSITION delete with
//! `seq < min` (strict), or an EQUALITY delete with `seq <= min` (non-strict, THE OFF-BY-ONE). A DV
//! dangles when its `referenced_data_file` is not a live data-file path. The off-by-one is the
//! corruption edge: a position delete at the exact min still applies and must be KEPT, and an
//! equality delete at the exact min does not apply and must be REMOVED.
//!
//! ## The scenario contract, hand-coded identically in Java `RemoveDanglingOracle` and here
//!
//! Schema `{1 id long required, 2 cat string required, 3 y long required}`, spec 0 `identity(cat)`.
//! Each partition isolates one corruption edge. The fixture splits into two worlds because the spec
//! mandates DVs for V3 position deletes and forbids parquet position deletes there, so the two
//! cannot coexist in one table.
//!
//! V2 world (`<dir>/table`, parquet position + equality + no-data):
//!
//! | partition | data seq(s) | delete (seq)            | declared | reason                                        |
//! |-----------|-------------|-------------------------|----------|-----------------------------------------------|
//! | `pk`      | 1           | POSITION (1, SAME seq)  | KEEP     | `1 < 1` false — pos AT min applies (off-by-one)|
//! | `pr`      | 2 → 6 (rw)  | POSITION (3)            | REMOVE   | `3 < 6` — pos strictly below the new min      |
//! | `ek`      | 2           | EQUALITY (3)            | KEEP     | `3 <= 2` false — eq strictly above min applies|
//! | `er`      | 4           | EQUALITY (4, SAME seq)  | REMOVE   | `4 <= 4` — eq AT min (the OFF-BY-ONE) dangles |
//! | `ne`      | (none live) | EQUALITY (5)            | REMOVE   | `min IS NULL` — no live data in partition+spec|
//!
//! V3 world (`<dir>/table_v3`, deletion vectors):
//!
//! | partition | data seq(s) | delete (seq)            | declared | reason                                        |
//! |-----------|-------------|-------------------------|----------|-----------------------------------------------|
//! | `dv`      | 1 → 3 (rw)  | DV on the rewritten file| REMOVE\* | referenced data file gone                     |
//! | `dk`      | 1           | DV on the live file     | KEEP     | referenced data file live                     |
//!
//! \* THE V3 DV DIVERGENCE. Java `ManifestFilterManager` AUTO-PRUNES a dangling DV at the commit that
//! removes its referenced data file. Rust's `RewriteFiles` carry-posture does not, and leaves the
//! dangling DV for this action. So the `dv` REMOVE is reachable on the Rust-built table, while over
//! the Java-built table this action correctly removes NOTHING and agrees with Java's clean state.
//! See [`Origin`].
//!
//! `pk` and `er` are the two off-by-one boundaries, one per content type. Each commits the data file
//! and its delete in ONE `row_delta`, so they share a sequence number and the delete sits at the
//! exact partition min. Flipping the position branch `<` to `<=` resurrects `pk`'s masked row, and
//! flipping the equality branch `<=` to `<` strands `er`. `pr` and `dv` dangle through a fresh-seq
//! rewrite that lifts the partition min above the delete's seq, or replaces the referenced file.
//! `ne` never held live data, and `nb` holds ballast so the V2 live row set is non-trivial.
//!
//! ## Read-identity
//!
//! The live `id` set is the same before and after the action. The kept deletes still mask their
//! rows, and the removed deletes were already not applying. See `expected_live_ids`.
//!
//! ## Directions
//!
//! D1, `test_rust_validates_java_remove_dangling`: Rust registers the Java-built pre-cleanup table,
//! runs the action, and asserts the removed set, the survivors, the counters, and read-identity.
//! D2, `test_remove_dangling_gen`: Rust writes `<dir>/rust_table<suffix>` and
//! `<dir>/rust_table_cleaned<suffix>`, and Java recomputes `findDanglingDeletes` on the pre table and
//! reads both for read-identity.
//!
//! `ICEBERG_INTEROP_REMOVE_DANGLING_GEN_DIR` gates D2 and `ICEBERG_INTEROP_REMOVE_DANGLING_DIR` gates
//! D1. Both unset means a clean no-op, so the offline gate stays green.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use iceberg::io::LocalFsStorageFactory;
use iceberg::maintenance::RemoveDanglingDeleteFiles;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, Literal, ManifestContentType, NestedField,
    PartitionKey, PartitionSpec, PrimitiveType, Schema, SortOrder, Struct, Transform, Type,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::deletion_vector_writer::DVFileWriter;
use iceberg::writer::base_writer::equality_delete_writer::{
    EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
};
use iceberg::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
};
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};

// The scenario contract, hand-declared identically in Java `RemoveDanglingOracle` and here, from
// the `findDanglingDeletes` spec and never from the other engine's output.

/// The hand-declared outcome of a partition's delete file under `findDanglingDeletes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    /// The delete still applies to live data — it MUST survive the action.
    Keep,
    /// The delete is dangling — it MUST be removed by the action.
    Remove,
}

/// One scenario partition: its `cat` value, the delete's content kind, and the hand-declared verdict.
#[derive(Debug, Clone, Copy)]
struct Scenario {
    /// The `identity(cat)` partition value isolating this scenario.
    cat: &'static str,
    /// The kind of delete this partition carries (drives the counter expectation).
    kind: DeleteKind,
    /// The hand-declared `findDanglingDeletes` verdict for this partition's delete.
    verdict: Verdict,
}

/// The content kind of a scenario's delete file (for the per-content-type counter assertions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeleteKind {
    Position,
    Equality,
    DeletionVector,
}

/// One world: a single on-disk table at a fixed format version, covering a subset of the scenarios.
/// The V2 world holds the parquet position, equality, and no-data scenarios, and the V3 world holds
/// the deletion vectors. One driver run exercises both. The directory names mirror Java's
/// `RemoveDanglingOracle`.
#[derive(Debug, Clone, Copy)]
struct World {
    /// The directory-name suffix this world's tables use (`""` for V2, `"_v3"` for V3).
    suffix: &'static str,
    /// The format version (V2 for parquet pos/eq, V3 for DVs).
    format_version: FormatVersion,
}

const V2_WORLD: World = World {
    suffix: "",
    format_version: FormatVersion::V2,
};
const V3_WORLD: World = World {
    suffix: "_v3",
    format_version: FormatVersion::V3,
};

fn worlds() -> [World; 2] {
    [V2_WORLD, V3_WORLD]
}

/// The scenarios for `world`. Java's `RemoveDanglingOracle` MUST mirror this split exactly, with the
/// same `cat` values, kinds, and verdicts.
fn scenarios(world: &World) -> Vec<Scenario> {
    match world.format_version {
        FormatVersion::V3 => vec![
            Scenario {
                cat: "dv",
                kind: DeleteKind::DeletionVector,
                verdict: Verdict::Remove,
            },
            Scenario {
                cat: "dk",
                kind: DeleteKind::DeletionVector,
                verdict: Verdict::Keep,
            },
        ],
        _ => vec![
            Scenario {
                cat: "pk",
                kind: DeleteKind::Position,
                verdict: Verdict::Keep,
            },
            Scenario {
                cat: "pr",
                kind: DeleteKind::Position,
                verdict: Verdict::Remove,
            },
            Scenario {
                cat: "ek",
                kind: DeleteKind::Equality,
                verdict: Verdict::Keep,
            },
            Scenario {
                cat: "er",
                kind: DeleteKind::Equality,
                verdict: Verdict::Remove,
            },
            Scenario {
                cat: "ne",
                kind: DeleteKind::Equality,
                verdict: Verdict::Remove,
            },
        ],
    }
}

/// The expected merge-on-read live `id` set for `world` (BEFORE == AFTER cleanup — read-identity).
/// Hand-declared from the fixture, mirrored by Java's `RemoveDanglingOracle`.
fn expected_live_ids(world: &World) -> HashSet<i64> {
    match world.format_version {
        FormatVersion::V3 => HashSet::from([
            // dv: 10,20,30 (the REMOVED DV referenced the rewritten-away file — already not applying)
            // dk: 10,30 (id 20 masked by the KEPT DV)
            600, 620, 630, // dv
            700, 730, // dk
        ]),
        _ => HashSet::from([
            // pk: 10,30 (id 20 masked by the KEPT position delete)
            // pr: 10,20,30 (the REMOVED pos delete referenced the rewritten-away file — already not applying)
            // ek: 10,30 (id 20 masked by the KEPT equality delete on y=20)
            // er: 10,20 (the same-seq eq delete NEVER applied — id 20 live before and after)
            // nb: 10,20 (the `nb` partition's data; the dangling `ne` delete is in an EMPTY partition)
            100, 130, // pk
            200, 220, 230, // pr
            300, 330, // ek
            400, 420, // er
            500, 520, // nb
        ]),
    }
}

// Env-var gates.

fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_REMOVE_DANGLING_GEN_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn validate_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_REMOVE_DANGLING_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

// Schema + table helpers ({id, cat, y}, identity(cat), V3 so DVs are legal).

/// The fixture schema `{1 id long required, 2 cat string required, 3 y long required}`.
fn dangling_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "cat", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "y", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("build the {id, cat, y} schema")
}

/// Build a `MemoryCatalog` over local-FS storage rooted at `warehouse`.
async fn build_catalog(name: &str, warehouse: &str) -> impl Catalog + use<> {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            name,
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_string())]),
        )
        .await
        .expect("build MemoryCatalog over local FS")
}

/// Create an `identity(cat)` table at `table_location` named `table_name` at `format_version`.
async fn create_dangling_table(
    catalog: &impl Catalog,
    table_name: &str,
    table_location: &str,
    format_version: FormatVersion,
) -> Table {
    let schema = dangling_schema();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("cat", "cat", Transform::Identity)
        .expect("add identity(cat) partition field")
        .build()
        .expect("build identity(cat) spec");
    let namespace = NamespaceIdent::new("interop".to_string());
    let _ = catalog.create_namespace(&namespace, HashMap::new()).await;
    let creation = TableCreation::builder()
        .name(table_name.to_string())
        .location(table_location.to_string())
        .schema(schema)
        .partition_spec(spec)
        .sort_order(SortOrder::unsorted_order())
        .format_version(format_version)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create the dangling fixture table")
}

// Real-parquet / real-delete writers (partition-scoped).

/// Write a REAL parquet DATA file in partition `cat` with the given `(id, y)` rows.
async fn write_data_file(table: &Table, cat: &str, rows: &[(i64, i64)]) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));
    let ids: Vec<i64> = rows.iter().map(|(id, _)| *id).collect();
    let cats: Vec<String> = rows.iter().map(|_| cat.to_string()).collect();
    let ys: Vec<i64> = rows.iter().map(|(_, y)| *y).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(cats)) as ArrayRef,
        Arc::new(Int64Array::from(ys)) as ArrayRef,
    ])
    .expect("build the {id, cat, y} data batch");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        format!("data-{cat}"),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let partition_key = partition_key_for(table, cat);
    let mut writer =
        iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder::new(rolling)
            .build(Some(partition_key))
            .await
            .expect("build partitioned data file writer");
    writer.write(batch).await.expect("write data batch");
    writer
        .close()
        .await
        .expect("close data file writer")
        .into_iter()
        .next()
        .expect("one data file")
}

/// The `identity(cat)` partition key for `cat`.
fn partition_key_for(table: &Table, cat: &str) -> PartitionKey {
    PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        Struct::from_iter([Some(Literal::string(cat))]),
    )
    .expect("PartitionKey::new: valid partition tuple")
}

/// Write a REAL parquet POSITION-delete file in partition `cat` deleting the given `(path, pos)` pairs.
async fn write_position_delete(table: &Table, cat: &str, deletes: &[(String, i64)]) -> DataFile {
    let config = PositionDeleteWriterConfig::new().expect("pos-delete config");
    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        format!("posdel-{cat}"),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        config.schema().clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(Some(partition_key_for(table, cat)))
        .await
        .expect("build position delete writer");

    let paths: Vec<&str> = deletes.iter().map(|(p, _)| p.as_str()).collect();
    let positions: Vec<i64> = deletes.iter().map(|(_, pos)| *pos).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions)) as ArrayRef,
    ])
    .expect("build pos-delete batch");
    writer.write(batch).await.expect("write pos-delete batch");
    writer
        .close()
        .await
        .expect("close pos-delete writer")
        .into_iter()
        .next()
        .expect("one pos-delete file")
}

/// Write a REAL parquet EQUALITY-delete file (on field id 3 = `y`) in partition `cat` deleting `y` values.
async fn write_equality_delete(table: &Table, cat: &str, delete_ys: &[i64]) -> DataFile {
    use iceberg::arrow::{arrow_schema_to_schema, schema_to_arrow_schema};

    let schema = table.metadata().current_schema().clone();
    let config =
        EqualityDeleteWriterConfig::new(vec![3], schema.clone()).expect("eq-delete config");
    let delete_schema =
        Arc::new(arrow_schema_to_schema(config.projected_arrow_schema_ref()).expect("eq schema"));

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        format!("eqdel-{cat}"),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        delete_schema,
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .build(Some(partition_key_for(table, cat)))
        .await
        .expect("build eq-delete writer");

    // The equality delete carries only the `y` column (field id 3).
    let arrow_schema = Arc::new(schema_to_arrow_schema(&schema).expect("schema → arrow"));
    let ids: Vec<i64> = delete_ys.iter().map(|_| 0).collect();
    let cats: Vec<String> = delete_ys.iter().map(|_| cat.to_string()).collect();
    let ys: Vec<i64> = delete_ys.to_vec();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(cats)) as ArrayRef,
        Arc::new(Int64Array::from(ys)) as ArrayRef,
    ])
    .expect("build eq-delete source batch");
    writer.write(batch).await.expect("write eq-delete batch");
    writer
        .close()
        .await
        .expect("close eq-delete writer")
        .into_iter()
        .next()
        .expect("one eq-delete file")
}

/// Write a REAL Puffin deletion vector referencing `data_file_path` in partition `cat`, deleting the
/// given positions.
async fn write_dv_file(
    table: &Table,
    cat: &str,
    data_file_path: &str,
    positions: &[u64],
) -> DataFile {
    let partition_key = partition_key_for(table, cat);
    let dv_path = format!(
        "{}/data/dv-{cat}-{}.puffin",
        table.metadata().location(),
        uuid::Uuid::now_v7()
    );
    let output_file = table
        .file_io()
        .new_output(&dv_path)
        .expect("dv output file");
    let mut dv_writer = DVFileWriter::new(output_file);
    for pos in positions {
        dv_writer
            .delete(data_file_path, *pos, Some(&partition_key))
            .expect("stage DV position");
    }
    dv_writer
        .close()
        .await
        .expect("close DV writer")
        .into_iter()
        .next()
        .expect("one DV file")
}

// Commit helpers.

async fn fast_append(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(files)
        .apply(tx)
        .expect("apply fast_append");
    tx.commit(catalog).await.expect("commit fast_append")
}

async fn row_delta(
    catalog: &impl Catalog,
    table: &Table,
    data: Vec<DataFile>,
    deletes: Vec<DataFile>,
) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .row_delta()
        .add_data_files(data)
        .add_deletes(deletes)
        .apply(tx)
        .expect("apply row_delta");
    tx.commit(catalog).await.expect("commit row_delta")
}

/// A FRESH-seq rewrite (no seq preservation) of `replaced` → `added` — pushes the partition min above
/// the replaced files' seq.
async fn rewrite_fresh(
    catalog: &impl Catalog,
    table: &Table,
    replaced: Vec<DataFile>,
    added: Vec<DataFile>,
) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .rewrite_files(replaced, added)
        .apply(tx)
        .expect("apply rewrite_files");
    tx.commit(catalog).await.expect("commit rewrite_files")
}

// The fixture builders — one per world. Each returns the table at the PRE-cleanup (dangling) state the
// action must clean. The V2 world covers parquet position + equality + no-data; the V3 world covers DVs.

/// Build the world for `world` and return the PRE-cleanup table.
async fn build_world(catalog: &impl Catalog, world: &World, table: Table) -> Table {
    match world.format_version {
        FormatVersion::V3 => build_v3_world(catalog, table).await,
        _ => build_v2_world(catalog, table).await,
    }
}

/// The V2 world: pk (pos KEEP — at the EXACT partition min, the position off-by-one boundary), pr (pos
/// REMOVE via fresh-seq rewrite), ek (eq KEEP), er (eq REMOVE — off-by-one same-seq), ne (eq REMOVE — no
/// live data in partition), plus `nb` data ballast.
async fn build_v2_world(catalog: &impl Catalog, mut table: Table) -> Table {
    // --- Commit 1 (seq 1): the POSITION OFF-BY-ONE `pk`. One row_delta commits the data file and its
    //     position delete, so both take this snapshot's sequence number and the delete sits at the exact
    //     partition min. The read path still applies it, because `delete_seq >= data_seq`, and the
    //     dangling predicate keeps it, because `seq < min` is false. Flipping the position branch `<`
    //     to `<=` wrongly removes it and resurrects id=120.
    let pk = write_data_file(&table, "pk", &[(100, 10), (120, 20), (130, 30)]).await;
    let pk_path = pk.file_path().to_string();
    let pk_del = write_position_delete(&table, "pk", &[(pk_path, 1)]).await; // deletes id=120, at-min
    table = row_delta(catalog, &table, vec![pk.clone()], vec![pk_del]).await;

    // --- Commit 2 (seq 2): base data for ek, pr (pr is rewritten in commit 6 to make its delete dangle).
    let ek = write_data_file(&table, "ek", &[(300, 10), (320, 20), (330, 30)]).await;
    let pr = write_data_file(&table, "pr", &[(200, 10), (220, 20), (230, 30)]).await;
    let pr_path = pr.file_path().to_string();
    table = fast_append(catalog, &table, vec![ek.clone(), pr.clone()]).await;

    // --- Commit 3 (seq 3): deletes at or above their partition min. `pr` stays keep-eligible until
    //     commit 6 rewrites it and pushes the partition min above this delete's seq.
    let ek_del = write_equality_delete(&table, "ek", &[20]).await; // deletes y=20 (id 320)
    let pr_del = write_position_delete(&table, "pr", &[(pr_path, 1)]).await; // deletes id=220 (until rw)
    table = row_delta(catalog, &table, vec![], vec![ek_del, pr_del]).await;

    // --- Commit 4 (seq 4): the OFF-BY-ONE `er` — data AND its equality delete in ONE row_delta, so both
    //     get THIS snapshot's sequence number. The eq delete is then AT the partition min ⇒ dangling.
    let er = write_data_file(&table, "er", &[(400, 10), (420, 20)]).await;
    let er_del = write_equality_delete(&table, "er", &[20]).await; // y=20 at seq == data seq
    table = row_delta(catalog, &table, vec![er], vec![er_del]).await;

    // --- Commit 5 (seq 5): the `ne` no-live-data delete. The live data goes to `nb` as read-identity
    //     ballast, and the equality delete lands in the empty `ne`, where `min IS NULL`.
    let nb = write_data_file(&table, "nb", &[(500, 10), (520, 20)]).await;
    table = fast_append(catalog, &table, vec![nb]).await;
    let ne_del = write_equality_delete(&table, "ne", &[20]).await; // partition `ne` has NO live data
    table = row_delta(catalog, &table, vec![], vec![ne_del]).await;

    // --- Commit 6 (seq 6): FRESH-seq rewrite pr → pr' (the partition min jumps above the pr pos delete's
    //     seq), so the pr pos delete now dangles (`pr_del seq < newMin`, STRICT — the position branch).
    let pr_prime = write_data_file(&table, "pr", &[(200, 10), (220, 20), (230, 30)]).await;
    table = rewrite_fresh(catalog, &table, vec![pr.clone()], vec![pr_prime]).await;

    table
}

/// The V3 world: dv (DV REMOVE — referenced file rewritten away) and dk (DV KEEP — referenced file live).
async fn build_v3_world(catalog: &impl Catalog, mut table: Table) -> Table {
    // --- Commit 1 (seq 1): base data for dv (rewritten away in commit 3) + dk (stays live).
    let dv = write_data_file(&table, "dv", &[(600, 10), (620, 20), (630, 30)]).await;
    let dv_path = dv.file_path().to_string();
    let dk = write_data_file(&table, "dk", &[(700, 10), (720, 20), (730, 30)]).await;
    let dk_path = dk.file_path().to_string();
    table = fast_append(catalog, &table, vec![dv.clone(), dk.clone()]).await;

    // --- Commit 2 (seq 2): a real Puffin DV per partition — dv-DV (→ removed) + dk-DV (→ kept).
    let dv_del = write_dv_file(&table, "dv", &dv_path, &[1]).await; // deletes id=620
    let dk_del = write_dv_file(&table, "dk", &dk_path, &[1]).await; // deletes id=720
    table = row_delta(catalog, &table, vec![], vec![dv_del, dk_del]).await;

    // --- Commit 3 (seq 3): FRESH-seq rewrite dv → dv' (the dv-DV's referenced file is now gone ⇒ dangling).
    let dv_prime = write_data_file(&table, "dv", &[(600, 10), (620, 20), (630, 30)]).await;
    table = rewrite_fresh(catalog, &table, vec![dv.clone()], vec![dv_prime]).await;

    table
}

// Read-side + survivor helpers.

/// Scan the table and collect the live `id` set (merge-on-read deletes applied) — the read-identity signal.
async fn scan_ids(table: &Table) -> HashSet<i64> {
    let stream = table
        .scan()
        .select(["id"])
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to arrow");
    let batches: Vec<RecordBatch> = stream.try_collect().await.expect("collect batches");
    let mut ids = HashSet::new();
    for batch in batches {
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("id column is Int64");
        for index in 0..col.len() {
            ids.insert(col.value(index));
        }
    }
    ids
}

/// The set of live delete-file paths in the current snapshot.
async fn live_delete_paths(table: &Table) -> HashSet<String> {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list loads");
    let mut paths = HashSet::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest loads");
        for entry in manifest.entries() {
            if entry.is_alive() {
                paths.insert(entry.file_path().to_string());
            }
        }
    }
    paths
}

/// Group the live delete files by partition `cat`. Each partition carries exactly one delete file, so
/// the map takes the survivor set to the scenario verdicts.
async fn live_deletes_by_cat(table: &Table) -> HashMap<String, String> {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list loads");
    let mut by_cat = HashMap::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest loads");
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            let cat = partition_cat(entry.data_file());
            by_cat.insert(cat, entry.data_file().file_path().to_string());
        }
    }
    by_cat
}

/// The `cat` partition value of a file (identity(cat) ⇒ the single partition field is the string).
fn partition_cat(file: &DataFile) -> String {
    use iceberg::spec::PrimitiveLiteral;
    match file.partition().fields().first() {
        Some(Some(Literal::Primitive(PrimitiveLiteral::String(s)))) => s.clone(),
        other => format!("<unexpected:{other:?}>"),
    }
}

/// Map a removed `DataFile` to its partition `cat`.
fn removed_cat(file: &DataFile) -> String {
    partition_cat(file)
}

// D2 GEN — Rust writes the PRE-cleanup + CLEANED tables for Java's verify.

/// Rust builds `<gen_dir>/rust_table<suffix>` in the dangling state and
/// `<gen_dir>/rust_table_cleaned<suffix>` after the action, for each world. A Rust-side check asserts
/// the whole hand-declared contract BEFORE Java sees the tables, so a Rust regression fails here
/// instead of reaching the Java verify.
#[tokio::test]
async fn test_remove_dangling_gen() {
    let Some(gen_dir) = gen_dir() else {
        println!(
            "skipping interop_remove_dangling GEN — set ICEBERG_INTEROP_REMOVE_DANGLING_GEN_DIR \
             (run dev/java-interop/run-interop-remove-dangling.sh)"
        );
        return;
    };

    let warehouse = gen_dir.to_string_lossy().to_string();

    for world in worlds() {
        let pre_name = format!("rust_table{}", world.suffix);
        let cleaned_name = format!("rust_table_cleaned{}", world.suffix);
        let pre_location = format!("{warehouse}/{pre_name}");
        let cleaned_location = format!("{warehouse}/{cleaned_name}");
        let catalog = build_catalog(
            &format!("interop_remove_dangling_gen{}", world.suffix),
            &warehouse,
        )
        .await;

        // Build the PRE-cleanup table (the dangling state).
        let pre =
            create_dangling_table(&catalog, &pre_name, &pre_location, world.format_version).await;
        let pre = build_world(&catalog, &world, pre).await;

        // Land PRE final metadata BEFORE the action (the action commits a new snapshot).
        let pre_final = format!("{pre_location}/metadata/final.metadata.json");
        pre.metadata()
            .clone()
            .write_to(pre.file_io(), &pre_final)
            .await
            .expect("write pre final.metadata.json");

        let pre_ids = scan_ids(&pre).await;
        assert_eq!(
            pre_ids,
            expected_live_ids(&world),
            "GEN sanity ({}): PRE-cleanup live ids must equal the hand-declared set",
            pre_name
        );

        // Run the action and assert the full contract.
        let result = RemoveDanglingDeleteFiles::new(pre.clone())
            .execute(&catalog)
            .await
            .expect("run RemoveDanglingDeleteFiles");
        assert_action_contract(&result, &world, Origin::RustBuilt, &pre, &catalog).await;

        // Java reads the dangling PRE table and the cleaned table together, so the cleaned state needs
        // its own on-disk table. Rebuild the world and run the action on the copy.
        let cleaned = create_dangling_table(
            &catalog,
            &cleaned_name,
            &cleaned_location,
            world.format_version,
        )
        .await;
        let cleaned = build_world(&catalog, &world, cleaned).await;
        let cleaned_result = RemoveDanglingDeleteFiles::new(cleaned.clone())
            .execute(&catalog)
            .await
            .expect("run RemoveDanglingDeleteFiles on the cleaned-table world");
        assert!(
            !cleaned_result.removed_delete_files.is_empty(),
            "the cleaned-table world ({}) must have removed danglers",
            cleaned_name
        );
        let cleaned = catalog
            .load_table(cleaned.identifier())
            .await
            .expect("reload cleaned table");
        let cleaned_final = format!("{cleaned_location}/metadata/final.metadata.json");
        cleaned
            .metadata()
            .clone()
            .write_to(cleaned.file_io(), &cleaned_final)
            .await
            .expect("write cleaned final.metadata.json");

        // Read-identity sanity: cleaned live ids == pre live ids.
        assert_eq!(
            scan_ids(&cleaned).await,
            pre_ids,
            "GEN sanity ({}): read-identity — cleaned live ids must equal pre live ids",
            cleaned_name
        );

        println!(
            "interop_remove_dangling GEN OK — wrote {pre_location} (pre) + {cleaned_location} \
             (cleaned); removed {} danglers; read-identity holds",
            result.removed_delete_files.len()
        );
    }
}

// D1 — Rust validates the JAVA-written table (register + run the action).

/// Direction 1: Rust registers the Java-written pre-cleanup table for each world, runs
/// `RemoveDanglingDeleteFiles`, and asserts the removed set, the survivors, the counters, and
/// read-identity against Java's exact on-disk manifests.
#[tokio::test]
async fn test_rust_validates_java_remove_dangling() {
    let Some(dir) = validate_dir() else {
        println!(
            "skipping interop_remove_dangling D1 — set ICEBERG_INTEROP_REMOVE_DANGLING_DIR \
             (run dev/java-interop/run-interop-remove-dangling.sh)"
        );
        return;
    };

    let warehouse = dir.to_string_lossy().to_string();

    for world in worlds() {
        let table_name = format!("table{}", world.suffix);
        let metadata_path = dir.join(format!("{table_name}/metadata/final.metadata.json"));
        assert!(
            metadata_path.exists(),
            "missing Java table at {} (run the Java generate step first)",
            metadata_path.display()
        );

        let catalog = build_catalog(
            &format!("interop_remove_dangling_d1{}", world.suffix),
            &warehouse,
        )
        .await;
        let namespace = NamespaceIdent::new("interop".to_string());
        let _ = catalog.create_namespace(&namespace, HashMap::new()).await;

        // The catalog derives the next metadata version from the registered file NAME, which must match
        // `<version>-<uuid>.metadata.json`. Java writes a fixed-name `final.metadata.json`, so register
        // a conventionally-named COPY and leave the original untouched for a re-run.
        let reg_path = dir.join(format!(
            "{table_name}/metadata/99999-{}.metadata.json",
            uuid::Uuid::now_v7()
        ));
        std::fs::copy(&metadata_path, &reg_path)
            .expect("copy final.metadata.json to a registerable <version>-<uuid> name");

        let ident = TableIdent::new(namespace, format!("java_{table_name}"));
        let table = catalog
            .register_table(&ident, reg_path.to_string_lossy().to_string())
            .await
            .expect("register the Java-written dangling table");

        // Read-identity baseline: the live id set BEFORE the action.
        let pre_ids = scan_ids(&table).await;
        assert_eq!(
            pre_ids,
            expected_live_ids(&world),
            "D1 ({}): the Java table's PRE-cleanup live ids must equal the hand-declared set",
            table_name
        );

        let result = RemoveDanglingDeleteFiles::new(table.clone())
            .execute(&catalog)
            .await
            .expect("run RemoveDanglingDeleteFiles on the Java table");
        assert_action_contract(&result, &world, Origin::JavaBuilt, &table, &catalog).await;

        // Read-identity: the post-action live id set is IDENTICAL (no resurrection, no loss).
        let cleaned = catalog
            .load_table(table.identifier())
            .await
            .expect("reload cleaned Java table");
        assert_eq!(
            scan_ids(&cleaned).await,
            pre_ids,
            "D1 read-identity ({}) — removing the danglers must not change the live id set",
            table_name
        );

        println!(
            "interop_remove_dangling D1 OK — Rust validated the Java table {table_name}: removed {} \
             danglers; survivors + counters + read-identity all match the hand-declared contract",
            result.removed_delete_files.len()
        );
    }
}

// The shared contract assertion — removed == REMOVE set, survivors == KEEP set, per-content counters.

/// Which engine BUILT the table under test — selects the V3 DV expectation. Java AUTO-PRUNES a dangling
/// DV at the commit that removes its referenced data file (1.10.0 `ManifestFilterManager`); Rust's
/// `RewriteFiles` carry-posture does NOT, leaving the dangling DV for this action. So on the JAVA-built
/// V3 table the `dv` DV is already gone (the action removes NOTHING and only the `dk` DV is live), whereas
/// on the RUST-built V3 table the action removes the persisted `dv` DV. This is the 1:1-faithful
/// divergence the action exists to bridge for DVs — documented, not hidden.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Origin {
    RustBuilt,
    JavaBuilt,
}

/// The scenarios that are actually PRESENT-AND-DANGLING on a table built by `origin` for `world`. For the
/// JAVA-built V3 table the `dv` DV was auto-pruned at commit, so it is neither removed nor surviving.
fn effective_remove_scenarios(world: &World, origin: Origin) -> Vec<Scenario> {
    scenarios(world)
        .into_iter()
        .filter(|s| s.verdict == Verdict::Remove)
        .filter(|s| {
            !(origin == Origin::JavaBuilt
                && world.format_version == FormatVersion::V3
                && s.kind == DeleteKind::DeletionVector)
        })
        .collect()
}

/// Assert the action's outcome matches the hand-declared scenario contract: the REMOVED delete files are
/// EXACTLY the (origin-effective) REMOVE-verdict partitions; the SURVIVING delete files are EXACTLY the
/// KEEP-verdict partitions; and the per-content-type counters match. `table` is the PRE-cleanup table the
/// action ran on; `catalog` reloads the cleaned head for the survivor check.
async fn assert_action_contract(
    result: &iceberg::maintenance::RemoveDanglingDeleteFilesResult,
    world: &World,
    origin: Origin,
    table: &Table,
    catalog: &impl Catalog,
) {
    let scen = scenarios(world);
    let removed_scenarios = effective_remove_scenarios(world, origin);

    // Removed set by partition cat.
    let removed_cats: HashSet<String> = result
        .removed_delete_files
        .iter()
        .map(removed_cat)
        .collect();
    let expected_removed: HashSet<String> = removed_scenarios
        .iter()
        .map(|s| s.cat.to_string())
        .collect();
    assert_eq!(
        removed_cats, expected_removed,
        "the REMOVED partitions must equal the hand-declared REMOVE set (origin {origin:?})"
    );

    // Surviving delete files by partition cat (the KEEP set — same on both engines).
    let cleaned = catalog
        .load_table(table.identifier())
        .await
        .expect("reload cleaned table for survivor check");
    let survivors: HashSet<String> = live_deletes_by_cat(&cleaned).await.into_keys().collect();
    let expected_keep: HashSet<String> = scen
        .iter()
        .filter(|s| s.verdict == Verdict::Keep)
        .map(|s| s.cat.to_string())
        .collect();
    assert_eq!(
        survivors, expected_keep,
        "the SURVIVING delete-file partitions must equal the hand-declared KEEP set"
    );

    // Per-content-type counters (over the origin-effective removed set).
    let expect_count =
        |kind: DeleteKind| -> usize { removed_scenarios.iter().filter(|s| s.kind == kind).count() };
    assert_eq!(
        result.removed_position_delete_files_count(),
        expect_count(DeleteKind::Position),
        "removed-position-delete-files counter mismatch"
    );
    assert_eq!(
        result.removed_equality_delete_files_count(),
        expect_count(DeleteKind::Equality),
        "removed-equality-delete-files counter mismatch"
    );
    assert_eq!(
        result.removed_dvs_count(),
        expect_count(DeleteKind::DeletionVector),
        "removed-dvs counter mismatch"
    );

    // Belt-and-suspenders: no REMOVE-verdict delete file is still live; no KEEP-verdict delete file is gone.
    let live = live_delete_paths(&cleaned).await;
    let removed_paths: HashSet<String> = result
        .removed_delete_files
        .iter()
        .map(|f| f.file_path().to_string())
        .collect();
    for path in &removed_paths {
        assert!(
            !live.contains(path),
            "a removed delete file must be tombstoned: {path}"
        );
    }
}
