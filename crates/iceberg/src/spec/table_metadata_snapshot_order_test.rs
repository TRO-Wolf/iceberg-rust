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

    fn snapshot_order_metadata(format_version: FormatVersion) -> TableMetadata {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .expect("schema must build");
        TableMetadataBuilder::new(
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
        .metadata
    }

    fn snapshot_order_snapshot(
        base_timestamp_ms: i64,
        sequence_number: i64,
        timestamp_offset_ms: i64,
        snapshot_id: i64,
    ) -> (i64, Arc<Snapshot>) {
        let snapshot = Snapshot::builder()
            .with_snapshot_id(snapshot_id)
            .with_timestamp_ms(base_timestamp_ms + timestamp_offset_ms)
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
    }

    fn serialized_snapshot_ids(metadata: &TableMetadata) -> Vec<i64> {
        let json = serde_json::to_string(metadata).expect("metadata must serialize");
        let value: serde_json::Value =
            serde_json::from_str(&json).expect("serialized metadata must parse as JSON");
        value["snapshots"]
            .as_array()
            .expect("serialized snapshots must be an array")
            .iter()
            .map(|snapshot| snapshot["snapshot-id"].as_i64().expect("snapshot id"))
            .collect()
    }

    #[test]
    fn metadata_serialization_orders_snapshots_by_sequence_number() {
        for format_version in [FormatVersion::V2, FormatVersion::V3] {
            let mut metadata = snapshot_order_metadata(format_version);
            let base = metadata.last_updated_ms;
            metadata.snapshots = [(2, 10, 300), (3, 20, 100), (1, 30, 200)]
                .into_iter()
                .map(|(sequence_number, offset, id)| {
                    snapshot_order_snapshot(base, sequence_number, offset, id)
                })
                .collect();
            assert_eq!(serialized_snapshot_ids(&metadata), [200, 300, 100]);
        }
    }

    #[test]
    fn metadata_serialization_orders_v1_snapshots_by_timestamp() {
        let mut metadata = snapshot_order_metadata(FormatVersion::V1);
        let base = metadata.last_updated_ms;
        metadata.snapshots = [(0, 20, 100), (0, 30, 300), (0, 10, 200)]
            .into_iter()
            .map(|(sequence_number, offset, id)| {
                snapshot_order_snapshot(base, sequence_number, offset, id)
            })
            .collect();
        assert_eq!(serialized_snapshot_ids(&metadata), [200, 100, 300]);
    }

    #[test]
    fn metadata_serialization_breaks_snapshot_ties_by_id() {
        let mut metadata = snapshot_order_metadata(FormatVersion::V2);
        let base = metadata.last_updated_ms;
        metadata.snapshots = [(1, 10, 500), (2, 20, 100), (1, 10, 400)]
            .into_iter()
            .map(|(sequence_number, offset, id)| {
                snapshot_order_snapshot(base, sequence_number, offset, id)
            })
            .collect();
        assert_eq!(serialized_snapshot_ids(&metadata), [400, 500, 100]);
    }
