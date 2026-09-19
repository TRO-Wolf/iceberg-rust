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

use std::collections::HashSet;

use super::*;
use crate::metadata_columns::{
    RESERVED_FIELD_ID_DELETE_FILE_PATH as PATH_ID, RESERVED_FIELD_ID_DELETE_FILE_POS as POS_ID,
};

#[tokio::test]
async fn staged_pos_delete_keeps_full_bounds_under_none_default() {
    let (catalog, _file_io, _tmp) = local_fs_catalog().await;
    let table = create_table(&catalog).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .update_table_properties()
        .set(
            "write.metadata.metrics.default".to_string(),
            "none".to_string(),
        )
        .apply(tx)
        .expect("apply");
    let table = tx.commit(&catalog).await.expect("commit");

    let source = table.metadata().location().to_string();
    let target = "s3://bucket/relocated".to_string();
    let staging = format!("{source}-staging");
    let pos_delete = pos_delete_with_bounds(&format!("{source}/data/pd.parquet"), None, None);
    let referenced = format!("{target}/data/a-referenced-data-file-name.parquet");
    let staged = RewriteTablePath::new(table)
        .rewrite_location_prefix(&source, &target)
        .staging_location(&staging)
        .write_position_delete_content(&pos_delete, &[(referenced.clone(), 0)], &source, &staging)
        .await
        .expect("staged pos delete");

    assert!(staged.file_path().starts_with(&staging));
    let reserved: HashSet<i32> = HashSet::from([PATH_ID, POS_ID]);
    for keys in [
        staged
            .column_sizes()
            .keys()
            .copied()
            .collect::<HashSet<i32>>(),
        staged
            .value_counts()
            .keys()
            .copied()
            .collect::<HashSet<i32>>(),
        staged
            .null_value_counts()
            .keys()
            .copied()
            .collect::<HashSet<i32>>(),
        staged
            .lower_bounds()
            .keys()
            .copied()
            .collect::<HashSet<i32>>(),
        staged
            .upper_bounds()
            .keys()
            .copied()
            .collect::<HashSet<i32>>(),
    ] {
        assert_eq!(keys, reserved);
    }
    assert!(staged.nan_value_counts().is_empty());
    let bound = staged
        .lower_bounds()
        .get(&PATH_ID)
        .expect("file_path lower bound")
        .to_bytes()
        .unwrap();
    let path_str = String::from_utf8(bound.as_ref().to_vec()).unwrap();
    assert_eq!(path_str, referenced, "file_path bound must be FULL");
}
