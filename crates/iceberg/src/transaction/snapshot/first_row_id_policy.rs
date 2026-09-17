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

/// What a producer does with the `first_row_id` an ADDED data file already carries.
///
/// Java splits this by base class, so the fork makes it a REQUIRED constructor argument: a new
/// producer cannot inherit the wrong half by omission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FirstRowIdPolicy {
    /// Write the value the caller supplied. Java `FastAppend` and `BaseRewriteManifests`, which
    /// extend `SnapshotProducer` and never call `Delegates.suppressFirstRowId`.
    Preserve,
    /// Force the field absent. Java `MergingSnapshotProducer.add(DataFile)`. A stale id survives
    /// read-side inheritance, so the file keeps a row-id range that describes other rows.
    Suppress,
}
