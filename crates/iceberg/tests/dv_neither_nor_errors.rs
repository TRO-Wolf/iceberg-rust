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

//! A DV writer with neither spec nor PartitionKey must error at close, before the Puffin opens.

use iceberg::ErrorKind;
use iceberg::io::FileIO;
use iceberg::writer::base_writer::deletion_vector_writer::DVFileWriter;
use tempfile::TempDir;

#[tokio::test]
async fn dv_close_without_spec_or_key_errors_before_puffin_bytes() {
    let dir = TempDir::new().expect("temp dir");
    let file_io = FileIO::new_with_fs();
    let path = dir.path().join("deletes.puffin");
    let output = file_io
        .new_output(path.to_str().expect("utf-8"))
        .expect("output");
    let mut writer = DVFileWriter::new(output);
    writer
        .delete("s3://b/d.parquet", 1, None)
        .expect("delete records in memory");
    let err = writer
        .close()
        .await
        .expect_err("close with neither spec nor key must error");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(
        err.to_string().contains("unpartitioned()"),
        "unexpected error: {err}"
    );
    assert!(
        !path.exists(),
        "resolve runs before PuffinWriter::new, so no puffin bytes"
    );
}
