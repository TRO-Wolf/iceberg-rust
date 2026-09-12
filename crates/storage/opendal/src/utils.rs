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

pub(crate) fn is_truthy(value: &str) -> bool {
    ["true", "t", "1", "on"].contains(&value.to_lowercase().as_str())
}

/// Convert an opendal error into an iceberg error.
pub(crate) fn from_opendal_error(e: opendal::Error) -> iceberg::Error {
    iceberg::Error::new(
        iceberg::ErrorKind::Unexpected,
        "Failure in doing io operation",
    )
    .with_source(e)
}

pub(crate) fn scheme_relative_path<'a>(
    path: &'a str,
    schemes: &[&str],
    bucket: &str,
) -> Option<&'a str> {
    for scheme in schemes {
        let prefix = format!("{scheme}://{bucket}");
        if let Some(rest) = path.strip_prefix(&prefix) {
            if rest.is_empty() {
                return Some("");
            }
            if let Some(key) = rest.strip_prefix('/') {
                return Some(key);
            }
        }
    }
    None
}

pub(crate) fn join_list_location(base: &str, entry_path: &str) -> String {
    if base.ends_with('/') || entry_path.starts_with('/') {
        format!("{base}{entry_path}")
    } else {
        format!("{base}/{entry_path}")
    }
}

#[cfg(test)]
mod tests {
    use super::{join_list_location, scheme_relative_path};

    #[test]
    fn test_scheme_relative_path_bucket_boundary() {
        for scheme in ["s3", "gs", "oss"] {
            let bare = format!("{scheme}://bucket");
            assert_eq!(scheme_relative_path(&bare, &[scheme], "bucket"), Some(""));
            let slash = format!("{scheme}://bucket/");
            assert_eq!(scheme_relative_path(&slash, &[scheme], "bucket"), Some(""));
            let key = format!("{scheme}://bucket/k");
            assert_eq!(scheme_relative_path(&key, &[scheme], "bucket"), Some("k"));
            let longer = format!("{scheme}://bucketx");
            assert_eq!(scheme_relative_path(&longer, &[scheme], "bucket"), None);
            let other = format!("{scheme}://bucket-other/k");
            assert_eq!(scheme_relative_path(&other, &[scheme], "bucket"), None);
        }
    }

    #[test]
    fn test_join_list_location_separator_boundary() {
        assert_eq!(join_list_location("s3://b", "k"), "s3://b/k");
        assert_eq!(join_list_location("s3://b/", "k"), "s3://b/k");
        assert_eq!(
            join_list_location("abfss://fs@acct.dfs.core.windows.net", "/p/f"),
            "abfss://fs@acct.dfs.core.windows.net/p/f"
        );
        assert_eq!(
            join_list_location("memory:/", "dir/a.txt"),
            "memory:/dir/a.txt"
        );
        assert_eq!(
            join_list_location("s3://b", "metadata/00000-uuid.metadata.json"),
            "s3://b/metadata/00000-uuid.metadata.json"
        );
    }
}
