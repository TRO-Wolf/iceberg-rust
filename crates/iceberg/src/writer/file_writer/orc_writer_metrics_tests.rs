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

#[tokio::test]
async fn test_the_data_file_carries_the_full_java_metric_set() {
    let (_temp, file_io, location_gen) = make_temp();
    let schema = Arc::new(schema_all_primitives());
    let written = all_primitives_batch(&schema);

    let (_path, files) = write_orc(
        OrcWriterBuilder::new(schema.clone()),
        &file_io,
        &location_gen,
        "metrics",
        std::slice::from_ref(&written),
    )
    .await;
    let data_file = files
        .into_iter()
        .next()
        .expect("one data file")
        .build()
        .expect("build the data file");

    assert_eq!(data_file.record_count(), 3);
    assert_eq!(data_file.file_format(), DataFileFormat::Orc);
    assert!(data_file.file_size_in_bytes() > 0);

    for field_id in 1..=16i32 {
        assert_eq!(
            data_file.value_counts().get(&field_id),
            Some(&3),
            "field {field_id} must carry a value count"
        );
        assert_eq!(
            data_file.null_value_counts().get(&field_id),
            Some(&1),
            "field {field_id} has exactly one null row"
        );
        assert!(
            data_file.column_sizes().contains_key(&field_id),
            "field {field_id} must carry a column size"
        );
    }

    assert_eq!(data_file.nan_value_counts().get(&4), Some(&1));
    assert_eq!(data_file.nan_value_counts().get(&5), Some(&1));
    assert_eq!(
        data_file.nan_value_counts().len(),
        2,
        "only FLOAT and DOUBLE carry nan_value_count"
    );

    for field_id in [7i32, 15, 16] {
        assert!(
            !data_file.lower_bounds().contains_key(&field_id),
            "ORC binary statistics carry no bound for field {field_id}"
        );
    }
    for field_id in [1i32, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14] {
        assert!(
            data_file.lower_bounds().contains_key(&field_id),
            "field {field_id} must carry a lower bound"
        );
        assert!(
            data_file.upper_bounds().contains_key(&field_id),
            "field {field_id} must carry an upper bound"
        );
    }

    assert_eq!(
        data_file.lower_bounds().get(&2).map(Datum::literal),
        Some(&crate::spec::PrimitiveLiteral::Int(i32::MIN))
    );
    assert_eq!(
        data_file.upper_bounds().get(&2).map(Datum::literal),
        Some(&crate::spec::PrimitiveLiteral::Int(42))
    );
    assert_eq!(
        data_file.lower_bounds().get(&4).map(Datum::literal),
        Some(&crate::spec::PrimitiveLiteral::Float(3.5f32.into())),
        "NaN must not poison the float bound"
    );
    assert_eq!(
        data_file.split_offsets(),
        Some(&[3i64][..]),
        "the single stripe starts after the 3-byte ORC magic"
    );
}

#[tokio::test]
async fn test_metrics_config_none_drops_the_column_from_the_data_file() {
    let (_temp, file_io, location_gen) = make_temp();
    let schema = Arc::new(schema_all_primitives());
    let written = all_primitives_batch(&schema);
    let config = crate::spec::MetricsConfig::from_properties(&HashMap::from([(
        format!(
            "{}c_string",
            crate::spec::METRICS_MODE_COLUMN_CONF_PREFIX
        ),
        "none".to_string(),
    )]))
    .expect("parse the metrics config");

    let (_path, files) = write_orc(
        OrcWriterBuilder::new(schema.clone()).with_metrics_config(config),
        &file_io,
        &location_gen,
        "metrics_none",
        std::slice::from_ref(&written),
    )
    .await;
    let data_file = files
        .into_iter()
        .next()
        .expect("one data file")
        .build()
        .expect("build the data file");

    assert!(!data_file.value_counts().contains_key(&6));
    assert!(!data_file.null_value_counts().contains_key(&6));
    assert!(!data_file.column_sizes().contains_key(&6));
    assert!(!data_file.lower_bounds().contains_key(&6));
    assert!(data_file.value_counts().contains_key(&2));
}
