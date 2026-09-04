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

use super::*;

/// Writes Parquet position-delete files from sorted `(path, pos)` pairs and returns EVERY file the
/// rolling writer produced; dropping one silently resurrects its rows. Each file is stamped with the
/// `(spec_id, partition)` of the DATA file it deletes from, which the partitioned path reads from
/// the snapshot's manifests. The commit validates that stamp against the spec.
///
/// This predicate decides which table shape may skip that walk (BUG-001, C1-L-002):
///
/// | Table shape | Path |
/// |---|---|
/// | one spec, zero fields | fast path: one file, stamped through `with_partition_spec` so the real spec id survives |
/// | multi-spec, empty default (after `DROP PARTITION FIELD`) | walk: old data files keep their own partition, and a fabricated `None`/spec-0 stamp misses on read and resurrects rows |
/// | one all-Void spec (unpartitioned, non-empty fields) | walk: it needs a null tuple of matching arity |
/// | partitioned | walk |
pub(crate) async fn write_position_deletes(
    table: &Table,
    pairs: &[(String, i64)],
    scan_snapshot_id: Option<i64>,
) -> DFResult<Vec<DataFile>> {
    let config = PositionDeleteWriterConfig::new().map_err(to_datafusion_error)?;
    let metadata = table.metadata();
    let default_spec = metadata.default_partition_spec();
    let schema = metadata.current_schema();

    // Only a never-evolved empty spec skips the manifest walk — see the fast-path table above.
    if position_delete_unpartitioned_fast_path(
        metadata.partition_specs_iter().len(),
        default_spec.fields().len(),
    ) {
        // `with_partition_spec` keeps the sole spec's real id; `None` would fabricate spec id 0.
        return write_position_deletes_for_partition(
            table,
            &config,
            pairs,
            None,
            Some(default_spec.as_ref().clone()),
        )
        .await;
    }

    let path_to_partition = live_data_file_partitions(table, scan_snapshot_id, None).await?;

    let path_to_partition: HashMap<String, (i32, Struct)> = path_to_partition
        .into_iter()
        .map(|(path, (spec_id, partition, _))| (path, (spec_id, partition)))
        .collect();
    let groups = group_pairs_by_partition(pairs, &path_to_partition)?;

    let mut all_delete_files: Vec<DataFile> = Vec::new();
    for ((spec_id, partition), mut group_pairs) in groups {
        // Maintain the per-file (path, pos) sort order within each group.
        sort_position_delete_pairs(&mut group_pairs);

        let spec = metadata
            .partition_spec_by_id(spec_id)
            .ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "position-delete: data file references unknown partition spec {spec_id}"
                ))
            })?
            .as_ref()
            .clone();
        // Carry the data file's own (spec, partition), including empty and all-Void null tuples. A
        // `None` key would fabricate spec id 0 and under-attach after DROP PARTITION FIELD.
        let partition_key =
            PartitionKey::new(spec, schema.clone(), partition).map_err(to_datafusion_error)?;

        let files = write_position_deletes_for_partition(
            table,
            &config,
            &group_pairs,
            Some(partition_key),
            None,
        )
        .await?;
        all_delete_files.extend(files);
    }

    Ok(all_delete_files)
}

/// The `(path, pos)` pairs of one position-delete output file, keyed by the `(spec_id, partition)`
/// of the data files they delete from.
pub(crate) type PositionDeleteGroups = HashMap<(i32, Struct), Vec<(String, i64)>>;

/// Groups `(path, pos)` pairs by the `(spec_id, partition)` of the data file each deletes from, so
/// every output file is stamped like its target. Only the partitioned path reaches this. A pair
/// whose data file is absent from `path_to_partition` is a hard error: the pairs come from a scan of
/// the same snapshot that built the map. The old fallback fabricated an EMPTY tuple under a
/// PARTITIONED spec, writing a delete file under a `field=null` path that no reader can match.
pub(crate) fn group_pairs_by_partition(
    pairs: &[(String, i64)],
    path_to_partition: &HashMap<String, (i32, Struct)>,
) -> DFResult<PositionDeleteGroups> {
    let mut groups = PositionDeleteGroups::new();
    for pair in pairs {
        let key = path_to_partition.get(&pair.0).cloned().ok_or_else(|| {
            DataFusionError::Internal(format!(
                "position-delete: data file `{}` is not a live file of the current snapshot, so \
                 its partition cannot be resolved",
                pair.0
            ))
        })?;
        groups.entry(key).or_default().push(pair.clone());
    }
    Ok(groups)
}

/// Writes one position-delete file for a SINGLE `(spec_id, partition)` group. `pairs` must already
/// be sorted by `(path, pos)`. With `partition_key = None`, `configured_spec` MUST be `Some`, or the
/// writer fabricates `DEFAULT_PARTITION_SPEC_ID` (0) instead of the real spec id.
async fn write_position_deletes_for_partition(
    table: &Table,
    config: &PositionDeleteWriterConfig,
    pairs: &[(String, i64)],
    partition_key: Option<PartitionKey>,
    configured_spec: Option<iceberg::spec::PartitionSpec>,
) -> DFResult<Vec<DataFile>> {
    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).map_err(to_datafusion_error)?;
    let file_name_gen = DefaultFileNameGenerator::new(
        "pos-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    // Keep the `file_path` and `pos` bounds FULL and EXACT: no parquet stats truncation, so
    // min_is_exact/max_is_exact stay true and equal-bounds path routing works for long S3 URIs.
    let parquet_builder =
        ParquetWriterBuilder::new(position_delete_writer_properties(), config.schema().clone())
            .with_metrics_config(MetricsConfig::for_position_delete());
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    if partition_key.is_none() && configured_spec.is_none() {
        return Err(DataFusionError::Internal(
            "position-delete: write_position_deletes_for_partition requires either a PartitionKey \
             or a configured_spec; both None would fabricate partition_spec_id 0"
                .to_string(),
        ));
    }
    let mut builder = PositionDeleteFileWriterBuilder::new(rolling, config.clone());
    if let Some(spec) = configured_spec {
        builder = builder.with_partition_spec(spec);
    }
    let mut writer = builder
        .build(partition_key)
        .await
        .map_err(to_datafusion_error)?;

    let paths: Vec<&str> = pairs.iter().map(|(path, _)| path.as_str()).collect();
    let positions: Vec<i64> = pairs.iter().map(|(_, pos)| *pos).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions)) as ArrayRef,
    ])
    .map_err(|e| {
        DataFusionError::ArrowError(
            Box::new(e),
            Some("Failed to build position-delete batch".into()),
        )
    })?;
    writer.write(batch).await.map_err(to_datafusion_error)?;
    let files = writer.close().await.map_err(to_datafusion_error)?;
    // A non-empty group MUST produce a file, or the deletes vanish and the rows come back.
    if files.is_empty() {
        return Err(DataFusionError::Internal(
            "position-delete writer produced no file for a non-empty pair group".to_string(),
        ));
    }
    Ok(files)
}
