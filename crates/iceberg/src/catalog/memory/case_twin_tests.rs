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

use tempfile::TempDir;

use super::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use crate::spec::{NestedField, PrimitiveType, Schema, Type};
use crate::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};

#[tokio::test]
async fn test_create_and_load_table_with_case_twin_columns() {
    let warehouse = TempDir::new().expect("warehouse dir");
    let catalog = MemoryCatalogBuilder::default()
        .load(
            "memory",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                warehouse.path().to_str().expect("utf8 path").to_string(),
            )]),
        )
        .await
        .expect("load memory catalog");
    let namespace = NamespaceIdent::new("ns".into());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("twins".into())
        .schema(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "a", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "A", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .expect("case-twin columns must build"),
        )
        .build();
    let ident = TableIdent::new(namespace.clone(), "twins".into());
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table with case-twin columns must succeed");
    let table = catalog
        .load_table(&ident)
        .await
        .expect("load twin table must succeed");
    let schema = table.metadata().current_schema();
    let ids: Vec<i32> = ["a", "A"]
        .iter()
        .map(|name| schema.field_by_name(name).expect("twin column").id)
        .collect();
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1]);
}
