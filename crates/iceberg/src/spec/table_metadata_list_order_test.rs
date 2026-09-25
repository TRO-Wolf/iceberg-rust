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

    fn list_order_metadata(format_version: FormatVersion) -> TableMetadata {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .expect("schema must build");
        let mut metadata = TableMetadataBuilder::new(
            schema.clone(),
            PartitionSpec::unpartition_spec().into_unbound(),
            SortOrder::unsorted_order(),
            "s3://test/location".to_string(),
            format_version,
            HashMap::new(),
        )
        .expect("metadata must build")
        .build()
        .expect("metadata must build")
        .metadata;
        let ids = [2, 0, 1];
        metadata.schemas = ids
            .iter()
            .map(|&id| (id, Arc::new(schema.clone().with_schema_id(id))))
            .collect();
        metadata.partition_specs = ids
            .iter()
            .map(|&id| (id, Arc::new(PartitionSpec::unpartition_spec().with_spec_id(id))))
            .collect();
        metadata.sort_orders = ids
            .iter()
            .map(|&id| {
                let order = SortOrder {
                    order_id: i64::from(id),
                    fields: vec![],
                };
                (i64::from(id), Arc::new(order))
            })
            .collect();
        metadata
    }

    fn serialized_list_ids(metadata: &TableMetadata, list: &str, key: &str) -> Vec<i64> {
        serde_json::to_value(metadata).expect("metadata must serialize")[list]
            .as_array()
            .expect("serialized list must be an array")
            .iter()
            .map(|entry| entry[key].as_i64().expect("list entry id"))
            .collect()
    }

    #[test]
    fn metadata_serialization_lists_schemas_specs_and_orders_by_id() {
        for format_version in [FormatVersion::V1, FormatVersion::V2, FormatVersion::V3] {
            for _ in 0..32 {
                let metadata = list_order_metadata(format_version);
                for (list, key) in [
                    ("schemas", "schema-id"),
                    ("partition-specs", "spec-id"),
                    ("sort-orders", "order-id"),
                ] {
                    assert_eq!(
                        serialized_list_ids(&metadata, list, key),
                        vec![0, 1, 2],
                        "{format_version} {list}"
                    );
                }
            }
        }
    }
