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

use std::collections::BTreeSet;
use std::time::Instant;

use super::harness::{
    Harness, NS, TBL, harness, live_delete_files, load_table, run_sql, sql_count,
};

const ROWS_PER_FILE: usize = 3;

fn seed_values(files: usize) -> String {
    (0..files)
        .flat_map(|file| {
            (0..ROWS_PER_FILE).map(move |row| {
                let id = file * 10 + row;
                format!("({id}, 'd{id}', 'p{file:04}')")
            })
        })
        .collect::<Vec<_>>()
        .join(", ")
}

async fn seed_files(harness: &Harness, files: usize) {
    run_sql(
        &harness.ctx,
        &format!(
            "INSERT INTO catalog.{NS}.{TBL} VALUES {}",
            seed_values(files)
        ),
    )
    .await;
}

fn container_bytes(path: &str) -> u64 {
    std::fs::metadata(path)
        .unwrap_or_else(|error| panic!("stat {path}: {error}"))
        .len()
}

struct ContainerShape {
    containers: usize,
    newest_blobs: usize,
    newest_bytes: u64,
}

async fn container_shape(harness: &Harness, previous: &BTreeSet<String>) -> ContainerShape {
    let table = load_table(&harness.catalog).await;
    let deletes = live_delete_files(&table).await;
    let containers: BTreeSet<String> = deletes
        .iter()
        .map(|file| file.file_path().to_string())
        .collect();
    let newest = containers
        .iter()
        .find(|path| !previous.contains(*path))
        .cloned()
        .expect("a new container");
    ContainerShape {
        containers: containers.len(),
        newest_blobs: deletes
            .iter()
            .filter(|file| file.file_path() == newest)
            .count(),
        newest_bytes: container_bytes(&newest),
    }
}

async fn amplification_at(files: usize) -> ContainerShape {
    let harness = harness().await;
    seed_files(&harness, files).await;
    let deleted = sql_count(
        &harness.ctx,
        &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id % 10 = 0"),
    )
    .await;
    assert_eq!(deleted as usize, files, "one row deleted per data file");
    let table = load_table(&harness.catalog).await;
    let seeded = live_delete_files(&table).await;
    assert_eq!(seeded.len(), files, "one DV per data file");
    let seeded_containers: BTreeSet<String> = seeded
        .iter()
        .map(|file| file.file_path().to_string())
        .collect();
    assert_eq!(seeded_containers.len(), 1, "one Puffin for the statement");
    let deleted = sql_count(
        &harness.ctx,
        &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1"),
    )
    .await;
    assert_eq!(deleted, 1);
    container_shape(&harness, &seeded_containers).await
}

#[tokio::test]
async fn a_later_delete_rewrites_only_the_touched_blob() {
    for files in [16usize, 64] {
        let shape = amplification_at(files).await;
        assert_eq!(shape.containers, 2, "{files} blobs: exactly two containers");
        assert_eq!(
            shape.newest_blobs, 1,
            "{files} blobs: the new container holds ONLY the touched blob"
        );
        assert!(
            shape.newest_bytes < 1024,
            "{files} blobs: the new container is a single-blob Puffin, got {} bytes",
            shape.newest_bytes
        );
    }
}

#[tokio::test]
#[ignore = "wall-clock measurement, not a CI pin"]
async fn dv_close_wall_time_by_live_data_files() {
    for files in [8usize, 64, 192] {
        let harness = harness().await;
        seed_files(&harness, files).await;
        run_sql(
            &harness.ctx,
            &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 0"),
        )
        .await;
        let started = Instant::now();
        for row in 1..=6usize {
            let id = row * 10 + 1;
            run_sql(
                &harness.ctx,
                &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = {id}"),
            )
            .await;
        }
        eprintln!(
            "live-data-files={files} six-single-row-deletes={}ms",
            started.elapsed().as_millis()
        );
    }
}

#[tokio::test]
#[ignore = "wall-clock measurement, not a CI pin"]
async fn dv_close_bytes_by_blob_count() {
    for files in [16usize, 64] {
        let shape = amplification_at(files).await;
        eprintln!(
            "blobs={files} containers={} newest-blobs={} newest-bytes={}",
            shape.containers, shape.newest_blobs, shape.newest_bytes
        );
    }
}

async fn seed_one_manifest_per_file(harness: &Harness, files: usize) {
    for file in 0..files {
        let rows: Vec<String> = (0..ROWS_PER_FILE)
            .map(|row| {
                let id = file * 10 + row;
                format!("({id}, 'd{id}', 'p{file:04}')")
            })
            .collect();
        run_sql(
            &harness.ctx,
            &format!("INSERT INTO catalog.{NS}.{TBL} VALUES {}", rows.join(", ")),
        )
        .await;
    }
}

#[tokio::test]
#[ignore = "wall-clock measurement, not a CI pin"]
async fn dv_close_wall_time_by_data_manifest_count() {
    for files in [8usize, 64, 192] {
        let harness = harness().await;
        seed_one_manifest_per_file(&harness, files).await;
        let started = Instant::now();
        for file in 0..6usize {
            let id = file * 10 + 1;
            run_sql(
                &harness.ctx,
                &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = {id}"),
            )
            .await;
        }
        eprintln!(
            "data-manifests={files} six-single-row-deletes={}ms",
            started.elapsed().as_millis()
        );
    }
}
