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

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use futures::TryStreamExt;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;
use uuid::Uuid;

use super::{
    MemoryCatalog, begin_replace, catalog, create_as, creation, id_schema, ident, insert, location,
    replace, table_location, wide_schema,
};
use crate::spec::TableMetadata;
use crate::table::Table;
use crate::{Catalog, Result, TableCommit, TableMetadataCache, TableUpdate};

const SEED: u64 = 0x5354_414c_4557_5231;
const SEQUENCES: usize = 96;
const STEPS: usize = 14;

#[derive(Clone, Copy, Debug)]
enum Op {
    Create,
    CreateAs,
    Insert,
    ReplaceAs,
    ReplaceColumns,
    Drop,
    DropPurge,
    ConflictUpdate,
    ConflictReplace,
}

const ABSENT_OPS: [Op; 2] = [Op::Create, Op::CreateAs];

const PRESENT_OPS: [Op; 9] = [
    Op::Insert,
    Op::Insert,
    Op::ReplaceAs,
    Op::ReplaceAs,
    Op::ReplaceColumns,
    Op::Drop,
    Op::DropPurge,
    Op::ConflictUpdate,
    Op::ConflictReplace,
];

#[derive(Clone, Debug, PartialEq, Eq)]
struct Observed {
    pointer: String,
    incarnation: usize,
    snapshot: Option<usize>,
    schema_id: i32,
    files: std::result::Result<BTreeSet<String>, String>,
}

struct Lane {
    warehouse: TempDir,
    catalog: MemoryCatalog,
    incarnations: HashMap<Uuid, usize>,
    snapshots: HashMap<i64, usize>,
}

impl Lane {
    async fn new(cache: Option<Arc<TableMetadataCache>>) -> Self {
        let warehouse = TempDir::new().expect("tempdir");
        let catalog = catalog(&warehouse, "hadoop", cache).await;
        Self {
            warehouse,
            catalog,
            incarnations: HashMap::new(),
            snapshots: HashMap::new(),
        }
    }

    async fn run(&self, op: Op, step: usize) -> String {
        let name = format!("s{step}");
        let outcome = match op {
            Op::Create => self
                .catalog
                .create_table(&ident().namespace, creation(id_schema()))
                .await
                .map(drop),
            Op::CreateAs => create_as(&self.catalog, &self.warehouse, &name)
                .await
                .map(drop),
            Op::Insert => insert(&self.catalog, &name).await.map(drop),
            Op::ReplaceAs => replace(&self.catalog, id_schema(), Some(&name))
                .await
                .map(drop),
            Op::ReplaceColumns => replace(&self.catalog, wide_schema(), None).await.map(drop),
            Op::Drop => self.catalog.drop_table(&ident()).await,
            Op::DropPurge => self.drop_purge().await,
            Op::ConflictUpdate => self.conflict_update().await.map(drop),
            Op::ConflictReplace => self.conflict_replace(&name).await.map(drop),
        };
        match outcome {
            Ok(()) => "ok".to_string(),
            Err(error) => format!("{:?}", error.kind()),
        }
    }

    async fn drop_purge(&self) -> Result<()> {
        self.catalog.drop_table(&ident()).await?;
        std::fs::remove_dir_all(table_location(&self.warehouse)).expect("purge table directory");
        Ok(())
    }

    async fn conflict_update(&self) -> Result<Table> {
        let table = self.catalog.load_table(&ident()).await?;
        let current = location(&table);
        let (dir, _) = current.rsplit_once('/').expect("metadata dir");
        let commit = TableCommit::builder()
            .ident(ident())
            .requirements(vec![])
            .updates(vec![TableUpdate::SetProperties {
                updates: HashMap::from([("k".to_string(), "v".to_string())]),
            }])
            .base_metadata_location(Some(format!("{dir}/v0.metadata.json")))
            .build();
        self.catalog.update_table(commit).await
    }

    async fn conflict_replace(&self, name: &str) -> Result<Table> {
        let staged = begin_replace(&self.catalog, id_schema(), Some(name)).await?;
        insert(&self.catalog, &format!("{name}-race")).await?;
        staged.commit(&self.catalog).await
    }

    async fn observe(&mut self) -> Option<(Observed, bool)> {
        if !self
            .catalog
            .table_exists(&ident())
            .await
            .expect("table exists")
        {
            return None;
        }
        let table = self.catalog.load_table(&ident()).await.expect("load");
        let pointer = location(&table);
        let disk = TableMetadata::read_from(&self.catalog.file_io, &pointer)
            .await
            .expect("read the pointer's metadata file");
        let metadata = table.metadata();
        let matches_disk = disk.uuid() == metadata.uuid()
            && disk.current_snapshot_id() == metadata.current_snapshot_id()
            && disk.current_schema_id() == metadata.current_schema_id()
            && disk.last_sequence_number() == metadata.last_sequence_number();
        let next = self.incarnations.len();
        let incarnation = *self.incarnations.entry(metadata.uuid()).or_insert(next);
        let next = self.snapshots.len();
        let snapshot = metadata
            .current_snapshot_id()
            .map(|id| *self.snapshots.entry(id).or_insert(next));
        let observed = Observed {
            pointer: pointer
                .rsplit_once('/')
                .map(|(_, file)| file.to_string())
                .expect("pointer file name"),
            incarnation,
            snapshot,
            schema_id: metadata.current_schema_id(),
            files: file_names(&table).await,
        };
        Some((observed, matches_disk))
    }
}

async fn file_names(table: &Table) -> std::result::Result<BTreeSet<String>, String> {
    let scan = table
        .scan()
        .build()
        .map_err(|error| format!("{:?}", error.kind()))?;
    let tasks: Vec<_> = scan
        .plan_files()
        .await
        .map_err(|error| format!("{:?}", error.kind()))?
        .try_collect()
        .await
        .map_err(|error| format!("{:?}", error.kind()))?;
    Ok(tasks
        .iter()
        .filter_map(|task| task.data_file_path().rsplit_once('/'))
        .map(|(_, file)| file.to_string())
        .collect())
}

async fn assert_lanes_agree(label: &str, ops: &mut dyn FnMut(bool, usize) -> Option<Op>) {
    let cache = Arc::new(TableMetadataCache::new());
    let mut cached = Lane::new(Some(cache)).await;
    let mut uncached = Lane::new(None).await;
    let mut history = Vec::new();
    let mut present = false;
    for step in 0.. {
        let Some(op) = ops(present, step) else {
            break;
        };
        history.push(op);
        let cached_outcome = cached.run(op, step).await;
        let uncached_outcome = uncached.run(op, step).await;
        assert_eq!(
            cached_outcome, uncached_outcome,
            "{label}: the outcome of {history:?} differs with the cache on"
        );
        let cached_state = cached.observe().await;
        let uncached_state = uncached.observe().await;
        if let Some((_, matches_disk)) = &uncached_state {
            assert!(
                matches_disk,
                "{label}: {history:?} uncached load is not disk"
            );
        }
        if let Some((_, matches_disk)) = &cached_state {
            assert!(
                matches_disk,
                "{label}: after {history:?} the cached load is not the pointer's file"
            );
        }
        assert_eq!(
            cached_state.map(|(observed, _)| observed),
            uncached_state.clone().map(|(observed, _)| observed),
            "{label}: after {history:?} the cache-on load differs from the cache-off load"
        );
        present = uncached_state.is_some();
    }
}

fn fixed(sequence: &'static [Op]) -> impl FnMut(bool, usize) -> Option<Op> {
    move |_, step| sequence.get(step).copied()
}

#[tokio::test]
async fn test_matrix_sequences_agree_with_the_cache_on_and_off() {
    use Op::*;
    let sequences: [&'static [Op]; 8] = [
        &[Create, Insert, ReplaceAs, Drop, Create, ReplaceAs],
        &[Create, Insert, ReplaceAs, Drop, Create, ReplaceAs, Insert],
        &[Create, Insert, ReplaceAs, Drop, Create, ReplaceColumns],
        &[Create, Insert, ReplaceAs, Drop, CreateAs, ReplaceAs],
        &[Create, Insert, ReplaceAs, DropPurge, Create, ReplaceAs],
        &[CreateAs, ReplaceAs, ReplaceAs, Drop, CreateAs, ReplaceAs],
        &[
            Create,
            Insert,
            ConflictReplace,
            ReplaceAs,
            Drop,
            Create,
            Insert,
            ReplaceAs,
        ],
        &[
            Create,
            ConflictUpdate,
            Insert,
            ReplaceAs,
            Drop,
            Create,
            ReplaceAs,
            Insert,
        ],
    ];
    for (index, sequence) in sequences.into_iter().enumerate() {
        assert_lanes_agree(&format!("matrix row {index}"), &mut fixed(sequence)).await;
    }
}

#[tokio::test]
async fn test_generated_sequences_agree_with_the_cache_on_and_off() {
    let mut rng = StdRng::seed_from_u64(SEED);
    for sequence in 0..SEQUENCES {
        let mut generate = |present: bool, step: usize| {
            (step < STEPS).then(|| {
                let ops: &[Op] = if present { &PRESENT_OPS } else { &ABSENT_OPS };
                ops[rng.random_range(0..ops.len())]
            })
        };
        assert_lanes_agree(
            &format!("seed {SEED:#x} sequence {sequence} of {SEQUENCES}"),
            &mut generate,
        )
        .await;
    }
}
