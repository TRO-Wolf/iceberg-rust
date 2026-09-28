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

use std::collections::HashMap;
use std::sync::Arc;

use crate::io::FileIO;
use crate::spec::{
    FormatVersion, NestedField, PartitionSpec, PrimitiveType, Schema, SortOrder,
    TableMetadataBuilder, Type,
};
use crate::table::Table;
use crate::transaction::{Transaction, TransactionAction};
use crate::{TableIdent, TableUpdate};

fn twin_table() -> Table {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "a", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "A", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()
        .expect("case-twin columns must build");
    let metadata = TableMetadataBuilder::new(
        schema,
        PartitionSpec::unpartition_spec(),
        SortOrder::unsorted_order(),
        "s3://bucket/twins".to_string(),
        FormatVersion::V2,
        HashMap::new(),
    )
    .expect("metadata builder must accept a twin schema")
    .build()
    .expect("metadata must build")
    .metadata;
    Table::builder()
        .metadata(metadata)
        .metadata_location("s3://bucket/twins/metadata/v1.json".to_string())
        .identifier(TableIdent::from_strs(["ns1", "twins"]).expect("table ident"))
        .file_io(FileIO::new_with_memory())
        .build()
        .expect("table must build")
}

#[tokio::test]
async fn test_update_schema_case_insensitive_on_twins_refuses() {
    let table = twin_table();
    let action = Transaction::new(&table)
        .update_schema()
        .case_sensitive(false)
        .rename_column("a", "b");
    let error = match Arc::new(action).commit(&table).await {
        Ok(_) => panic!("case-insensitive schema evolution on twins must refuse"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "Cannot build lower case index: a and A collide"
    );
}

#[tokio::test]
async fn test_update_schema_case_sensitive_on_twins_succeeds() {
    let table = twin_table();
    let action = Transaction::new(&table)
        .update_schema()
        .rename_column("A", "B");
    let mut commit = Arc::new(action)
        .commit(&table)
        .await
        .expect("case-sensitive schema evolution on twins must succeed");
    assert_eq!(commit.take_updates().len(), 2);
}

#[tokio::test]
async fn test_update_partition_spec_case_insensitive_on_twins_refuses() {
    let table = twin_table();
    let action = Transaction::new(&table)
        .update_partition_spec()
        .case_sensitive(false)
        .add_field("a");
    let error = match Arc::new(action).commit(&table).await {
        Ok(_) => panic!("case-insensitive spec evolution on twins must refuse"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "Cannot build lower case index: a and A collide"
    );
}

#[tokio::test]
async fn test_update_partition_spec_case_sensitive_on_twins_succeeds() {
    let table = twin_table();
    let action = Transaction::new(&table)
        .update_partition_spec()
        .add_field("A");
    let mut commit = Arc::new(action)
        .commit(&table)
        .await
        .expect("case-sensitive spec evolution on twins must succeed");
    assert_eq!(commit.take_updates().len(), 2);
}

fn added_schema(updates: &[TableUpdate]) -> &Schema {
    updates
        .iter()
        .find_map(|update| match update {
            TableUpdate::AddSchema { schema } => Some(schema),
            _ => None,
        })
        .expect("an AddSchema update")
}

#[test]
fn test_twin_table_fixture_has_expected_ids() {
    let schema = twin_table().metadata().current_schema().clone();
    assert_eq!(schema.field_by_name("a").expect("column a").id, 1);
    assert_eq!(schema.field_by_name("A").expect("column A").id, 2);
}

#[tokio::test]
async fn test_update_schema_sensitive_add_twins_both_succeed() {
    let table = twin_table();
    let action = Transaction::new(&table)
        .update_schema()
        .add_column("B", Type::Primitive(PrimitiveType::Int))
        .add_column("b", Type::Primitive(PrimitiveType::Int));
    let mut commit = Arc::new(action)
        .commit(&table)
        .await
        .expect("sensitive add of B then b must succeed");
    let schema = added_schema(&commit.take_updates()).clone();
    assert_eq!(schema.field_by_name("B").expect("column B").id, 3);
    assert_eq!(schema.field_by_name("b").expect("column b").id, 4);
    assert_eq!(schema.field_by_name("a").expect("column a").id, 1);
    assert_eq!(schema.field_by_name("A").expect("column A").id, 2);
}

#[tokio::test]
async fn test_update_schema_sensitive_delete_a_removes_id_1_only() {
    let table = twin_table();
    let action = Transaction::new(&table).update_schema().delete_column("a");
    let mut commit = Arc::new(action)
        .commit(&table)
        .await
        .expect("sensitive delete of a must succeed");
    let schema = added_schema(&commit.take_updates()).clone();
    assert!(schema.field_by_name("a").is_none());
    assert_eq!(schema.field_by_name("A").expect("column A").id, 2);
}

#[tokio::test]
async fn test_update_schema_sensitive_rename_a_touches_id_1_only() {
    let table = twin_table();
    let action = Transaction::new(&table)
        .update_schema()
        .rename_column("a", "x");
    let mut commit = Arc::new(action)
        .commit(&table)
        .await
        .expect("sensitive rename of a must succeed");
    let schema = added_schema(&commit.take_updates()).clone();
    assert_eq!(schema.field_by_name("x").expect("column x").id, 1);
    assert!(schema.field_by_name("a").is_none());
    assert_eq!(schema.field_by_name("A").expect("column A").id, 2);
}

#[tokio::test]
async fn test_update_schema_sensitive_move_first_moves_id_2_only() {
    let table = twin_table();
    let action = Transaction::new(&table).update_schema().move_first("A");
    let mut commit = Arc::new(action)
        .commit(&table)
        .await
        .expect("sensitive move of A must succeed");
    let schema = added_schema(&commit.take_updates()).clone();
    let order: Vec<(i32, &str)> = schema
        .as_struct()
        .fields()
        .iter()
        .map(|field| (field.id, field.name.as_str()))
        .collect();
    assert_eq!(order, vec![(2, "A"), (1, "a")]);
}

#[tokio::test]
async fn test_update_schema_insensitive_add_on_twins_refuses() {
    let table = twin_table();
    let action = Transaction::new(&table)
        .update_schema()
        .case_sensitive(false)
        .add_column("a", Type::Primitive(PrimitiveType::Int));
    let error = match Arc::new(action).commit(&table).await {
        Ok(_) => panic!("insensitive add on twins must refuse"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "Cannot build lower case index: a and A collide"
    );
}

#[tokio::test]
async fn test_update_schema_insensitive_move_on_twins_refuses() {
    let table = twin_table();
    let action = Transaction::new(&table)
        .update_schema()
        .case_sensitive(false)
        .move_first("a");
    let error = match Arc::new(action).commit(&table).await {
        Ok(_) => panic!("insensitive move on twins must refuse"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "Cannot build lower case index: a and A collide"
    );
}

#[tokio::test]
async fn test_update_schema_insensitive_delete_on_twins_refuses() {
    let table = twin_table();
    let action = Transaction::new(&table)
        .update_schema()
        .case_sensitive(false)
        .delete_column("A");
    let error = match Arc::new(action).commit(&table).await {
        Ok(_) => panic!("insensitive delete on twins must refuse"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "Cannot build lower case index: a and A collide"
    );
}
