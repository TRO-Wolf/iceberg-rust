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

use super::{
    EntriesTable, FilesTable, HistoryTable, ManifestsTable, MetadataLogEntriesTable, RefsTable,
    SnapshotsTable,
};
use crate::table::Table;

/// Metadata table is used to inspect a table's history, snapshots, and other metadata as a table.
///
/// References:
/// - <https://github.com/apache/iceberg/blob/ac865e334e143dfd9e33011d8cf710b46d91f1e5/core/src/main/java/org/apache/iceberg/MetadataTableType.java#L23-L39>
/// - <https://iceberg.apache.org/docs/latest/spark-queries/#querying-with-sql>
/// - <https://py.iceberg.apache.org/api/#inspecting-tables>
#[derive(Debug)]
pub struct MetadataTable<'a>(&'a Table);

/// Metadata table type.
#[derive(Debug, Clone, strum::EnumIter)]
pub enum MetadataTableType {
    /// [`SnapshotsTable`]
    Snapshots,
    /// [`ManifestsTable`]
    Manifests,
    /// [`FilesTable`] over all data + delete files (Java `files`).
    Files,
    /// [`FilesTable`] over DATA-content files only (Java `data_files`).
    DataFiles,
    /// [`FilesTable`] over delete-content files only (Java `delete_files`).
    DeleteFiles,
    /// [`EntriesTable`] — all manifest entries of the current snapshot (Java `entries`).
    Entries,
    /// [`HistoryTable`] — one row per snapshot-log entry (Java `history`).
    History,
    /// [`RefsTable`] — one row per branch/tag reference (Java `refs`).
    Refs,
    /// [`MetadataLogEntriesTable`] — one row per metadata-log entry (Java `metadata_log_entries`).
    MetadataLogEntries,
}

impl MetadataTableType {
    /// Returns the string representation of the metadata table type.
    pub fn as_str(&self) -> &str {
        match self {
            MetadataTableType::Snapshots => "snapshots",
            MetadataTableType::Manifests => "manifests",
            MetadataTableType::Files => "files",
            MetadataTableType::DataFiles => "data_files",
            MetadataTableType::DeleteFiles => "delete_files",
            MetadataTableType::Entries => "entries",
            MetadataTableType::History => "history",
            MetadataTableType::Refs => "refs",
            MetadataTableType::MetadataLogEntries => "metadata_log_entries",
        }
    }

    /// Returns all the metadata table types.
    pub fn all_types() -> impl Iterator<Item = Self> {
        use strum::IntoEnumIterator;
        Self::iter()
    }
}

impl TryFrom<&str> for MetadataTableType {
    type Error = String;

    fn try_from(value: &str) -> std::result::Result<Self, String> {
        match value {
            "snapshots" => Ok(Self::Snapshots),
            "manifests" => Ok(Self::Manifests),
            "files" => Ok(Self::Files),
            "data_files" => Ok(Self::DataFiles),
            "delete_files" => Ok(Self::DeleteFiles),
            "entries" => Ok(Self::Entries),
            "history" => Ok(Self::History),
            "refs" => Ok(Self::Refs),
            "metadata_log_entries" => Ok(Self::MetadataLogEntries),
            _ => Err(format!("invalid metadata table type: {value}")),
        }
    }
}

impl<'a> MetadataTable<'a> {
    /// Creates a new metadata scan.
    pub fn new(table: &'a Table) -> Self {
        Self(table)
    }

    /// Get the snapshots table.
    pub fn snapshots(&self) -> SnapshotsTable<'_> {
        SnapshotsTable::new(self.0)
    }

    /// Get the manifests table.
    pub fn manifests(&self) -> ManifestsTable<'_> {
        ManifestsTable::new(self.0)
    }

    /// Get the `files` table — all data + delete files in the current snapshot.
    pub fn files(&self) -> FilesTable<'_> {
        FilesTable::all(self.0)
    }

    /// Get the `data_files` table — only DATA-content files in the current snapshot.
    pub fn data_files(&self) -> FilesTable<'_> {
        FilesTable::data(self.0)
    }

    /// Get the `delete_files` table — only position/equality delete files in the current snapshot.
    pub fn delete_files(&self) -> FilesTable<'_> {
        FilesTable::deletes(self.0)
    }

    /// Get the `entries` table — every manifest entry (data + delete, incl. Deleted tombstones) of the
    /// current snapshot, with the `data_file` projection nested under one struct column.
    pub fn entries(&self) -> EntriesTable<'_> {
        EntriesTable::new(self.0)
    }

    /// Get the `history` table — one row per snapshot-log entry (the table's current-snapshot history).
    pub fn history(&self) -> HistoryTable<'_> {
        HistoryTable::new(self.0)
    }

    /// Get the `refs` table — one row per branch/tag reference, with its retention policy.
    pub fn refs(&self) -> RefsTable<'_> {
        RefsTable::new(self.0)
    }

    /// Get the `metadata_log_entries` table — one row per metadata-log entry (previous metadata files
    /// plus the current one), with the snapshot that was current at each entry's timestamp.
    pub fn metadata_log_entries(&self) -> MetadataLogEntriesTable<'_> {
        MetadataLogEntriesTable::new(self.0)
    }
}
