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

use std::sync::Arc;

use crate::io::FileIO;
use crate::spec::{NestedField, PrimitiveType, Schema, TableMetadataBuilder, Type};
use crate::table::Table;
use crate::transaction::StagedTableTransaction;
use crate::{ErrorKind, NamespaceIdent, TableCreation, TableIdent};

fn schema() -> Schema {
    Schema::builder()
        .with_fields(vec![Arc::new(NestedField::required(
            1,
            "id",
            Type::Primitive(PrimitiveType::Long),
        ))])
        .build()
        .expect("schema")
}

async fn table_at(
    file_io: &FileIO,
    ident: &TableIdent,
    table_location: &str,
    metadata_location: &str,
) -> Table {
    let creation = TableCreation::builder()
        .name(ident.name().to_string())
        .location(table_location.to_string())
        .schema(schema())
        .build();
    let metadata = TableMetadataBuilder::from_table_creation(creation)
        .expect("metadata builder")
        .build()
        .expect("metadata")
        .metadata;
    metadata
        .write_to(file_io, metadata_location)
        .await
        .expect("write metadata");
    Table::builder()
        .identifier(ident.clone())
        .metadata(metadata)
        .metadata_location(metadata_location.to_string())
        .file_io(file_io.clone())
        .build()
        .expect("table")
}

fn replace_creation(ident: &TableIdent) -> TableCreation {
    TableCreation::builder()
        .name(ident.name().to_string())
        .schema(schema())
        .build()
}

#[tokio::test]
async fn replace_stages_next_version_after_a_hive_named_pointer() {
    let file_io = FileIO::new_with_memory();
    let ident = TableIdent::new(NamespaceIdent::new("ns".into()), "t".into());
    let table_location = "memory://wh/ns/t";
    let base_location = format!(
        "{table_location}/metadata/00003-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json"
    );
    let table = table_at(&file_io, &ident, table_location, &base_location).await;

    let staged = StagedTableTransaction::begin_replace(&table, replace_creation(&ident))
        .await
        .expect("begin replace");
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location")
        .to_string();
    assert!(
        staged_location.starts_with(&format!("{table_location}/metadata/00004-")),
        "the staged file must continue the base pointer's version, got {staged_location}"
    );
    assert!(staged_location.ends_with(".metadata.json"));
    assert!(
        file_io.exists(&staged_location).await.expect("exists"),
        "the staged metadata file must be written"
    );
}

#[tokio::test]
async fn replace_stages_next_version_after_a_hadoop_named_pointer() {
    let file_io = FileIO::new_with_memory();
    let ident = TableIdent::new(NamespaceIdent::new("ns".into()), "t".into());
    let table_location = "memory://wh/ns/t";
    let table = table_at(
        &file_io,
        &ident,
        table_location,
        &format!("{table_location}/metadata/v7.metadata.json"),
    )
    .await;

    let staged = StagedTableTransaction::begin_replace(&table, replace_creation(&ident))
        .await
        .expect("begin replace");
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location")
        .to_string();
    assert_eq!(
        staged_location,
        format!("{table_location}/metadata/v8.metadata.json"),
        "a Hadoop-named pointer must continue vN naming, got {staged_location}"
    );
}

#[tokio::test]
async fn concurrent_replace_from_a_hadoop_pointer_fails_on_exclusive_create() {
    let file_io = FileIO::new_with_memory();
    let ident = TableIdent::new(NamespaceIdent::new("ns".into()), "t".into());
    let table_location = "memory://wh/ns/t";
    let table = table_at(
        &file_io,
        &ident,
        table_location,
        &format!("{table_location}/metadata/v3.metadata.json"),
    )
    .await;

    let staged = StagedTableTransaction::begin_replace(&table, replace_creation(&ident))
        .await
        .expect("first begin replace");
    let first = staged
        .table()
        .metadata_location_result()
        .expect("first staged location")
        .to_string();
    assert_eq!(first, format!("{table_location}/metadata/v4.metadata.json"));
    let winner = file_io
        .new_input(&first)
        .expect("open winner")
        .read()
        .await
        .expect("read winner bytes");

    let err = match StagedTableTransaction::begin_replace(&table, replace_creation(&ident)).await {
        Ok(_) => panic!("a second replace onto the existing v4 must fail"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
    assert!(err.retryable(), "the collision must be retryable: {err}");
    assert_eq!(
        file_io
            .new_input(&first)
            .expect("reopen winner")
            .read()
            .await
            .expect("reread winner"),
        winner,
        "the losing replace must not overwrite the winner's file"
    );
}

#[tokio::test]
async fn concurrent_replaces_from_a_uuid_pointer_stage_distinct_files() {
    let file_io = FileIO::new_with_memory();
    let ident = TableIdent::new(NamespaceIdent::new("ns".into()), "t".into());
    let table_location = "memory://wh/ns/t";
    let table = table_at(
        &file_io,
        &ident,
        table_location,
        &format!(
            "{table_location}/metadata/00003-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json"
        ),
    )
    .await;

    let first = StagedTableTransaction::begin_replace(&table, replace_creation(&ident))
        .await
        .expect("first begin replace")
        .table()
        .metadata_location_result()
        .expect("first staged location")
        .to_string();
    let second = StagedTableTransaction::begin_replace(&table, replace_creation(&ident))
        .await
        .expect("second begin replace")
        .table()
        .metadata_location_result()
        .expect("second staged location")
        .to_string();

    assert!(
        first.starts_with(&format!("{table_location}/metadata/00004-")),
        "uuid-named base keeps version continuation, got {first}"
    );
    assert!(
        second.starts_with(&format!("{table_location}/metadata/00004-")),
        "uuid-named base keeps version continuation, got {second}"
    );
    assert_ne!(first, second, "uuid names cannot collide");
    assert!(file_io.exists(&first).await.expect("first exists"));
    assert!(file_io.exists(&second).await.expect("second exists"));
}

#[tokio::test]
async fn replace_restarts_versioning_when_base_pointer_does_not_parse() {
    let file_io = FileIO::new_with_memory();
    let ident = TableIdent::new(NamespaceIdent::new("ns".into()), "t".into());
    let table_location = "memory://wh/ns/t";
    let table = table_at(
        &file_io,
        &ident,
        table_location,
        &format!("{table_location}/metadata/final.metadata.json"),
    )
    .await;

    let staged = StagedTableTransaction::begin_replace(&table, replace_creation(&ident))
        .await
        .expect("begin replace");
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location");
    assert!(
        staged_location.starts_with(&format!("{table_location}/metadata/00000-")),
        "an unparsable base pointer keeps the v0 restart, got {staged_location}"
    );
}

#[tokio::test]
async fn replace_restarts_versioning_under_a_different_caller_location() {
    let file_io = FileIO::new_with_memory();
    let ident = TableIdent::new(NamespaceIdent::new("ns".into()), "t".into());
    let base_location = "memory://wh/ns/t";
    let table = table_at(
        &file_io,
        &ident,
        base_location,
        &format!(
            "{base_location}/metadata/00003-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json"
        ),
    )
    .await;
    let moved_location = "memory://wh/ns/relocated";
    let creation = TableCreation::builder()
        .name(ident.name().to_string())
        .location(moved_location.to_string())
        .schema(schema())
        .build();

    let staged = StagedTableTransaction::begin_replace(&table, creation)
        .await
        .expect("begin replace");
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location");
    assert!(
        staged_location.starts_with(&format!("{moved_location}/metadata/00000-")),
        "a relocated replace must stage v0 under the new location, got {staged_location}"
    );
}
