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

use crate::TableIdent;
use crate::io::FileIO;
use crate::spec::{
    FormatVersion, NestedField, PartitionSpec, PrimitiveType, Schema, SortOrder,
    TableMetadataBuilder, Type,
};
use crate::table::Table;
use crate::transaction::{Transaction, TransactionAction};

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
