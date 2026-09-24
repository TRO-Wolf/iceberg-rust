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

    #[test]
    fn metadata_serialization_orders_snapshots_by_sequence_number_and_id() {
        for format_version in [FormatVersion::V1, FormatVersion::V2, FormatVersion::V3] {
            let schema = Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .expect("schema must build");
            let mut metadata = TableMetadataBuilder::new(
                schema,
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

            metadata.snapshots = [2, 3, 1]
                .into_iter()
                .map(|sequence_number| {
                    let snapshot_id = sequence_number * 100;
                    let snapshot = Snapshot::builder()
                        .with_snapshot_id(snapshot_id)
                        .with_timestamp_ms(metadata.last_updated_ms + sequence_number)
                        .with_sequence_number(sequence_number)
                        .with_schema_id(0)
                        .with_manifest_list(format!(
                            "s3://test/location/metadata/snap-{snapshot_id}.avro"
                        ))
                        .with_summary(Summary {
                            operation: Operation::Append,
                            additional_properties: HashMap::new(),
                        })
                        .build();
                    (snapshot_id, Arc::new(snapshot))
                })
                .collect();

            let json = serde_json::to_string(&metadata).expect("metadata must serialize");
            let value: serde_json::Value =
                serde_json::from_str(&json).expect("serialized metadata must parse as JSON");
            let snapshots = value["snapshots"]
                .as_array()
                .expect("serialized snapshots must be an array");
            let snapshot_ids = snapshots
                .iter()
                .map(|snapshot| snapshot["snapshot-id"].as_i64().expect("snapshot id"))
                .collect::<Vec<_>>();
            assert_eq!(snapshot_ids, [100, 200, 300]);
            if format_version != FormatVersion::V1 {
                let sequence_numbers = snapshots
                    .iter()
                    .map(|snapshot| {
                        snapshot["sequence-number"]
                            .as_i64()
                            .expect("snapshot sequence number")
                    })
                    .collect::<Vec<_>>();
                assert_eq!(sequence_numbers, [1, 2, 3]);
            }
        }
    }
