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


    const V1_REFS_S0: i64 = 1741779913422974004;
    const V1_REFS_S1: i64 = 3659047894676981075;

    fn v1_refs_snapshot_json(snapshot_id: i64, timestamp_ms: i64) -> serde_json::Value {
        serde_json::json!({
            "snapshot-id": snapshot_id,
            "timestamp-ms": timestamp_ms,
            "summary": {"operation": "append", "engine-name": "spark"},
            "manifest-list": format!("/wh/ns/v1/metadata/snap-{snapshot_id}-1-a.avro"),
            "schema-id": 0
        })
    }

    fn v1_refs_spark_json(
        current_snapshot_id: i64,
        refs: Option<serde_json::Value>,
    ) -> serde_json::Value {
        let schema = serde_json::json!({
            "type": "struct",
            "schema-id": 0,
            "fields": [{"id": 1, "name": "id", "required": false, "type": "long"}]
        });
        let mut json = serde_json::json!({
            "format-version": 1,
            "table-uuid": "f87ea2c4-9b1a-492e-86cc-963af038045a",
            "location": "/wh/ns/v1",
            "last-updated-ms": 1790302668800_i64,
            "last-column-id": 1,
            "schema": schema,
            "current-schema-id": 0,
            "schemas": [schema],
            "partition-spec": [],
            "default-spec-id": 0,
            "partition-specs": [{"spec-id": 0, "fields": []}],
            "last-partition-id": 999,
            "default-sort-order-id": 0,
            "sort-orders": [{"order-id": 0, "fields": []}],
            "properties": {"owner": "john"},
            "current-snapshot-id": current_snapshot_id,
            "statistics": [],
            "partition-statistics": [],
            "snapshots": [
                v1_refs_snapshot_json(V1_REFS_S0, 1790302668319),
                v1_refs_snapshot_json(V1_REFS_S1, 1790302668737)
            ],
            "snapshot-log": [],
            "metadata-log": []
        });
        if let Some(refs) = refs {
            json["refs"] = refs;
        }
        json
    }

    fn v1_refs_parse(json: serde_json::Value) -> crate::Result<TableMetadata> {
        serde_json::from_value::<TableMetadata>(json).map_err(|e| {
            crate::Error::new(ErrorKind::DataInvalid, "v1 metadata did not parse").with_source(e)
        })
    }

    fn v1_refs_branch(snapshot_id: i64) -> SnapshotReference {
        SnapshotReference {
            snapshot_id,
            retention: SnapshotRetention::Branch {
                min_snapshots_to_keep: None,
                max_snapshot_age_ms: None,
                max_ref_age_ms: None,
            },
        }
    }

    #[test]
    fn test_v1_refs_round_trip_keeps_branch_and_tag() {
        let mut metadata = v1_refs_parse(v1_refs_spark_json(V1_REFS_S1, None)).unwrap();
        assert_eq!(metadata.format_version, FormatVersion::V1);
        metadata
            .refs
            .insert("b1".to_string(), SnapshotReference {
                snapshot_id: V1_REFS_S0,
                retention: SnapshotRetention::Branch {
                    min_snapshots_to_keep: Some(2),
                    max_snapshot_age_ms: Some(259200000),
                    max_ref_age_ms: Some(604800000),
                },
            });
        metadata
            .refs
            .insert("t1".to_string(), SnapshotReference {
                snapshot_id: V1_REFS_S0,
                retention: SnapshotRetention::Tag {
                    max_ref_age_ms: Some(86400000),
                },
            });
        let expected = metadata.refs.clone();
        assert_eq!(expected.len(), 3);

        let json = serde_json::to_string(&metadata).unwrap();
        let reread: TableMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(reread.format_version, FormatVersion::V1);
        assert_eq!(reread.refs, expected);
        assert_eq!(reread, metadata);
    }

    #[test]
    fn test_v1_refs_legacy_file_without_refs_builds_main_only() {
        let metadata = v1_refs_parse(v1_refs_spark_json(V1_REFS_S1, None)).unwrap();
        assert_eq!(
            metadata.refs,
            HashMap::from([("main".to_string(), v1_refs_branch(V1_REFS_S1))])
        );

        let metadata = v1_refs_parse(v1_refs_spark_json(-1, None)).unwrap();
        assert_eq!(metadata.current_snapshot_id, None);
        assert!(metadata.refs.is_empty());
    }

    #[test]
    fn test_v1_refs_spark_file_with_main_branch_and_tag_parses_all_three() {
        let refs = serde_json::json!({
            "t1": {"snapshot-id": V1_REFS_S0, "type": "tag"},
            "main": {"snapshot-id": V1_REFS_S1, "type": "branch"},
            "b1": {"snapshot-id": V1_REFS_S0, "type": "branch"}
        });
        let metadata = v1_refs_parse(v1_refs_spark_json(V1_REFS_S1, Some(refs))).unwrap();

        assert_eq!(
            metadata.refs,
            HashMap::from([
                ("main".to_string(), v1_refs_branch(V1_REFS_S1)),
                ("b1".to_string(), v1_refs_branch(V1_REFS_S0)),
                ("t1".to_string(), SnapshotReference {
                    snapshot_id: V1_REFS_S0,
                    retention: SnapshotRetention::Tag {
                        max_ref_age_ms: None
                    },
                }),
            ])
        );
        assert_eq!(metadata.current_snapshot_id, Some(V1_REFS_S1));
        assert_eq!(metadata.current_snapshot_id, Some(metadata.refs["main"].snapshot_id));
    }

    #[test]
    fn test_v1_refs_empty_table_branch_without_main_keeps_only_the_branch() {
        let refs = serde_json::json!({"b1": {"snapshot-id": V1_REFS_S0, "type": "branch"}});
        let metadata = v1_refs_parse(v1_refs_spark_json(-1, Some(refs))).unwrap();

        assert_eq!(metadata.current_snapshot_id, None);
        assert_eq!(
            metadata.refs,
            HashMap::from([("b1".to_string(), v1_refs_branch(V1_REFS_S0))])
        );
    }

    #[test]
    fn test_v1_refs_validation_matches_java() {
        let missing = serde_json::json!({"b1": {"snapshot-id": 42, "type": "branch"}});
        let err = v1_refs_parse(v1_refs_spark_json(V1_REFS_S1, Some(missing))).unwrap_err();
        assert!(
            format!("{err:?}").contains("Snapshot for reference b1 does not exist"),
            "{err:?}"
        );

        let stale_main = serde_json::json!({"main": {"snapshot-id": V1_REFS_S0, "type": "branch"}});
        let err = v1_refs_parse(v1_refs_spark_json(V1_REFS_S1, Some(stale_main))).unwrap_err();
        assert!(
            format!("{err:?}").contains("Current snapshot id does not match main branch"),
            "{err:?}"
        );

        let orphan_main =
            serde_json::json!({"main": {"snapshot-id": V1_REFS_S0, "type": "branch"}});
        let err = v1_refs_parse(v1_refs_spark_json(-1, Some(orphan_main))).unwrap_err();
        assert!(
            format!("{err:?}").contains("Current snapshot is not set, but main branch exists"),
            "{err:?}"
        );
    }

    #[test]
    fn test_v1_refs_serialized_key_set_writes_refs_like_java() {
        let metadata = v1_refs_parse(v1_refs_spark_json(-1, None)).unwrap();
        assert!(metadata.refs.is_empty());

        let json = serde_json::to_value(&metadata).unwrap();
        let mut keys: Vec<&str> = json
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();

        assert_eq!(keys, vec![
            "current-schema-id",
            "default-sort-order-id",
            "default-spec-id",
            "format-version",
            "last-column-id",
            "last-partition-id",
            "last-updated-ms",
            "location",
            "partition-spec",
            "partition-specs",
            "properties",
            "refs",
            "schema",
            "schemas",
            "snapshots",
            "sort-orders",
            "table-uuid",
        ]);
        assert_eq!(json["refs"], serde_json::json!({}));
    }
