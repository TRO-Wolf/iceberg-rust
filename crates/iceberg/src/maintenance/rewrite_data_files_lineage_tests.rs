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

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::TryStreamExt;

use crate::Catalog;
use crate::maintenance::rewrite_data_files::RewriteDataFiles;
use crate::maintenance::rewrite_data_files::tests::{
    append_files, create_partitioned_table, live_data_file_paths, local_fs_catalog, write_data_file,
};
use crate::metadata_columns::{
    RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER, RESERVED_COL_NAME_ROW_ID,
};
use crate::spec::FormatVersion;
use crate::table::Table;

async fn scan_lineage(table: &Table) -> Vec<(i64, i64, i64)> {
    let stream = table
        .scan()
        .select([
            "y",
            RESERVED_COL_NAME_ROW_ID,
            RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER,
        ])
        .build()
        .expect("scan")
        .to_arrow()
        .await
        .expect("to_arrow");
    let batches: Vec<_> = stream.try_collect().await.expect("collect");
    let mut rows = Vec::new();
    for batch in batches {
        let ys = batch
            .column_by_name("y")
            .expect("y")
            .as_primitive::<arrow_array::types::Int64Type>();
        let row_ids = batch
            .column_by_name(RESERVED_COL_NAME_ROW_ID)
            .expect("_row_id")
            .as_primitive::<arrow_array::types::Int64Type>();
        let seqs = batch
            .column_by_name(RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER)
            .expect("_last_updated_sequence_number")
            .as_primitive::<arrow_array::types::Int64Type>();
        for index in 0..batch.num_rows() {
            assert!(
                row_ids.is_valid(index),
                "compacted v3 row must have a _row_id"
            );
            assert!(
                seqs.is_valid(index),
                "compacted v3 row must have a last_updated_seq"
            );
            rows.push((ys.value(index), row_ids.value(index), seqs.value(index)));
        }
    }
    rows.sort_unstable();
    rows
}

#[tokio::test]
async fn v3_compaction_keeps_row_id_and_last_updated_seq() {
    let (catalog, _temp) = local_fs_catalog().await;
    let mut table = create_partitioned_table(&catalog, FormatVersion::V3).await;

    for index in 0..6i64 {
        let file = write_data_file(&table, &format!("small-{index}.parquet"), 0, &[(
            0,
            100 + index,
            1000 + index,
        )])
        .await;
        table = append_files(&catalog, &table, vec![file]).await;
    }

    let before = scan_lineage(&table).await;
    assert_eq!(before.len(), 6, "fixture: six rows");
    for (index, row) in before.iter().enumerate() {
        assert_eq!(
            row.1, index as i64,
            "pre-compaction _row_id is first_row_id + pos"
        );
        assert_eq!(
            row.2,
            (index as i64) + 1,
            "pre-compaction last_updated_seq is the append snapshot sequence"
        );
    }

    let files_before = live_data_file_paths(&table).await.len();
    let result = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(1_000_000)
        .execute(&catalog)
        .await
        .expect("compaction");
    assert_eq!(result.rewritten_data_files_count, 6);
    assert!(result.added_data_files_count >= 1);

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload");
    let files_after = live_data_file_paths(&table).await.len();
    assert!(
        files_after < files_before,
        "compaction must reduce the file count"
    );

    let after = scan_lineage(&table).await;
    assert_eq!(
        after, before,
        "compaction must keep _row_id and last_updated_seq for every live row"
    );
}

#[tokio::test]
async fn v2_compaction_does_not_persist_row_lineage_columns() {
    let (catalog, _temp) = local_fs_catalog().await;
    let mut table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    for index in 0..6i64 {
        let file = write_data_file(&table, &format!("small-{index}.parquet"), 0, &[(
            0,
            100 + index,
            1000 + index,
        )])
        .await;
        table = append_files(&catalog, &table, vec![file]).await;
    }

    RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(1_000_000)
        .execute(&catalog)
        .await
        .expect("v2 compaction");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload");

    let stream = table
        .scan()
        .select(["y", RESERVED_COL_NAME_ROW_ID])
        .build()
        .expect("scan")
        .to_arrow()
        .await
        .expect("to_arrow");
    let batches: Vec<_> = stream.try_collect().await.expect("collect");
    for batch in batches {
        let row_ids = batch
            .column_by_name(RESERVED_COL_NAME_ROW_ID)
            .expect("_row_id");
        assert_eq!(
            row_ids.null_count(),
            row_ids.len(),
            "v2 files must not grow a stored _row_id column; a v2 scan reports all-null lineage"
        );
    }
}
