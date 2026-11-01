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

use serde_json::{Value, json};

use crate::spec::{
    FormatVersion, MapType, NestedField, PartitionSpec, PrimitiveType, Schema, SortOrder,
    StructType, TableMetadataBuilder, TableProperties, Type, create_name_mapping,
};
use crate::table::Table;
use crate::transaction::action::{ActionCommit, TransactionAction};
use crate::transaction::tests::make_v2_table;
use crate::transaction::update_schema::UpdateSchemaAction;
use crate::{TableRequirement, TableUpdate};

const MAPPING: &str = TableProperties::PROPERTY_DEFAULT_NAME_MAPPING;

fn int_field(id: i32, name: &str) -> NestedField {
    NestedField::optional(id, name, Type::Primitive(PrimitiveType::Int))
}

fn base_schema() -> Schema {
    let value = NestedField::map_value_element(
        6,
        Type::Struct(StructType::new(vec![Arc::new(int_field(7, "x"))])),
        false,
    );
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                2,
                "s",
                Type::Struct(StructType::new(vec![Arc::new(int_field(4, "d"))])),
            )),
            Arc::new(NestedField::optional(
                3,
                "m",
                Type::Map(MapType::new(
                    Arc::new(NestedField::map_key_element(
                        5,
                        Type::Primitive(PrimitiveType::String),
                    )),
                    Arc::new(value),
                )),
            )),
        ])
        .build()
        .expect("build the base schema")
}

fn table_with(properties: HashMap<String, String>) -> Table {
    let metadata = TableMetadataBuilder::new(
        base_schema(),
        PartitionSpec::unpartition_spec(),
        SortOrder::unsorted_order(),
        "memory://mapping".to_string(),
        FormatVersion::V2,
        properties,
    )
    .expect("build the metadata builder")
    .build()
    .expect("build the metadata")
    .metadata;
    make_v2_table().with_metadata(Arc::new(metadata))
}

fn table_with_mapping(mapping: &Value) -> Table {
    table_with(HashMap::from([(MAPPING.to_string(), mapping.to_string())]))
}

fn created_mapping() -> Value {
    let schema = table_with(HashMap::new())
        .metadata()
        .current_schema()
        .clone();
    serde_json::to_value(create_name_mapping(&schema).expect("create the mapping"))
        .expect("serialize the mapping")
}

async fn commit(action: UpdateSchemaAction, table: &Table) -> ActionCommit {
    Arc::new(action)
        .commit(table)
        .await
        .expect("the schema update commits")
}

fn property_updates(updates: &[TableUpdate]) -> Vec<&HashMap<String, String>> {
    updates
        .iter()
        .filter_map(|update| match update {
            TableUpdate::SetProperties { updates } => Some(updates),
            _ => None,
        })
        .collect()
}

fn kinds(updates: &[TableUpdate]) -> Vec<&'static str> {
    updates
        .iter()
        .map(|update| match update {
            TableUpdate::AddSchema { .. } => "AddSchema",
            TableUpdate::SetCurrentSchema { .. } => "SetCurrentSchema",
            TableUpdate::SetProperties { .. } => "SetProperties",
            _ => "other",
        })
        .collect()
}

fn mapped(id: i32, names: &[&str], fields: Vec<Value>) -> Value {
    if fields.is_empty() {
        json!({"field-id": id, "names": names})
    } else {
        json!({"field-id": id, "names": names, "fields": fields})
    }
}

fn map_column(value_fields: Vec<Value>) -> Value {
    mapped(3, &["m"], vec![
        mapped(5, &["key"], vec![]),
        mapped(6, &["value"], value_fields),
    ])
}

#[tokio::test]
async fn rename_commit_holds_the_schema_updates_and_the_mapping_with_the_same_requirements() {
    let table = table_with_mapping(&created_mapping());
    let last_column_id = table.metadata().last_column_id();
    let current_schema_id = table.metadata().current_schema_id();
    let mut action_commit =
        commit(UpdateSchemaAction::new().rename_column("s.d", "v"), &table).await;
    let updates = action_commit.take_updates();
    assert_eq!(updates.len(), 3, "{:?}", kinds(&updates));
    assert!(matches!(updates[0], TableUpdate::AddSchema { .. }));
    assert!(matches!(updates[1], TableUpdate::SetCurrentSchema {
        schema_id: -1
    }));
    let TableUpdate::SetProperties {
        updates: properties,
    } = &updates[2]
    else {
        panic!(
            "the third update sets the mapping property: {:?}",
            kinds(&updates)
        );
    };
    assert_eq!(properties.len(), 1);
    let written: Value =
        serde_json::from_str(&properties[MAPPING]).expect("the written mapping is JSON");
    assert_eq!(
        written,
        json!([
            mapped(1, &["id"], vec![]),
            mapped(2, &["s"], vec![mapped(4, &["d", "v"], vec![])]),
            map_column(vec![mapped(7, &["x"], vec![])]),
        ])
    );
    assert_eq!(action_commit.take_requirements(), vec![
        TableRequirement::LastAssignedFieldIdMatch {
            last_assigned_field_id: last_column_id
        },
        TableRequirement::CurrentSchemaIdMatch { current_schema_id },
    ]);
}

#[tokio::test]
async fn commit_without_a_mapping_emits_no_property_update() {
    let table = table_with(HashMap::new());
    let mut action_commit = commit(
        UpdateSchemaAction::new()
            .rename_column("s.d", "v")
            .add_column("y", Type::Primitive(PrimitiveType::Int)),
        &table,
    )
    .await;
    let updates = action_commit.take_updates();
    assert_eq!(updates.len(), 2, "{:?}", kinds(&updates));
    assert!(property_updates(&updates).is_empty());
}

#[tokio::test]
async fn type_doc_requirement_and_move_changes_emit_no_property_update() {
    let table = table_with_mapping(&created_mapping());
    let actions = [
        UpdateSchemaAction::new().update_column("s.d", PrimitiveType::Long),
        UpdateSchemaAction::new().update_column_doc("s.d", Some("a doc")),
        UpdateSchemaAction::new().make_column_optional("id"),
        UpdateSchemaAction::new().move_first("s"),
        UpdateSchemaAction::new().delete_column("s.d"),
    ];
    for action in actions {
        let mut action_commit = commit(action, &table).await;
        let updates = action_commit.take_updates();
        assert_eq!(updates.len(), 2, "{:?}", kinds(&updates));
        assert!(
            property_updates(&updates).is_empty(),
            "{:?}",
            kinds(&updates)
        );
    }
}

#[tokio::test]
async fn mapped_field_without_an_id_is_kept() {
    let mut mapping = created_mapping();
    mapping
        .as_array_mut()
        .expect("the mapping is an array")
        .insert(0, json!({"names": ["legacy"]}));
    let table = table_with_mapping(&mapping);
    let mut action_commit =
        commit(UpdateSchemaAction::new().rename_column("s.d", "v"), &table).await;
    let updates = action_commit.take_updates();
    let properties = property_updates(&updates);
    assert_eq!(properties.len(), 1, "{:?}", kinds(&updates));
    let written: Value =
        serde_json::from_str(&properties[0][MAPPING]).expect("the written mapping is JSON");
    assert_eq!(
        written,
        json!([
            {"names": ["legacy"]},
            mapped(1, &["id"], vec![]),
            mapped(2, &["s"], vec![mapped(4, &["d", "v"], vec![])]),
            map_column(vec![mapped(7, &["x"], vec![])]),
        ])
    );
}

#[tokio::test]
async fn add_to_a_map_value_struct_maps_under_the_value() {
    let table = table_with_mapping(&created_mapping());
    let mut action_commit = commit(
        UpdateSchemaAction::new().add_column_to(
            Some("m"),
            "z",
            Type::Primitive(PrimitiveType::Int),
            None,
        ),
        &table,
    )
    .await;
    let updates = action_commit.take_updates();
    let properties = property_updates(&updates);
    assert_eq!(properties.len(), 1, "{:?}", kinds(&updates));
    let written: Value =
        serde_json::from_str(&properties[0][MAPPING]).expect("the written mapping is JSON");
    assert_eq!(
        written,
        json!([
            mapped(1, &["id"], vec![]),
            mapped(2, &["s"], vec![mapped(4, &["d"], vec![])]),
            map_column(vec![mapped(7, &["x"], vec![]), mapped(8, &["z"], vec![])]),
        ])
    );
}

#[tokio::test]
async fn unparsable_mapping_is_left_unchanged_and_the_schema_still_commits() {
    let table = table_with(HashMap::from([(
        MAPPING.to_string(),
        "not a mapping".to_string(),
    )]));
    let mut action_commit =
        commit(UpdateSchemaAction::new().rename_column("s.d", "v"), &table).await;
    let updates = action_commit.take_updates();
    assert_eq!(updates.len(), 2, "{:?}", kinds(&updates));
    assert!(property_updates(&updates).is_empty());
}

#[tokio::test]
async fn test_rename_preserves_id_and_type() {
    let table = make_v2_table();
    let mut action_commit = commit(
        UpdateSchemaAction::new().rename_column("y", "y_renamed"),
        &table,
    )
    .await;
    let updates = action_commit.take_updates();
    let TableUpdate::AddSchema { schema } = &updates[0] else {
        panic!("the first update adds the schema: {:?}", kinds(&updates));
    };
    let field = schema.field_by_name("y_renamed").expect("renamed column");
    assert_eq!(field.id, 2, "rename preserves the field id");
    assert_eq!(
        field.field_type.as_ref(),
        &Type::Primitive(PrimitiveType::Long)
    );
    assert!(schema.field_by_name("y").is_none(), "old name must be gone");
}

#[tokio::test]
async fn test_rename_missing_column_rejected() {
    let table = make_v2_table();
    let result = Arc::new(UpdateSchemaAction::new().rename_column("nope", "x2"))
        .commit(&table)
        .await;
    assert!(result.is_err());
}
