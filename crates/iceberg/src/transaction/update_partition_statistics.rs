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

//! The `UpdatePartitionStatistics` transaction seam — the Rust port of Java 1.10.0
//! `org.apache.iceberg.UpdatePartitionStatistics`
//! (`extends PendingUpdate<List<PartitionStatisticsFile>>`):
//!
//! - `setPartitionStatistics(PartitionStatisticsFile)` → [`UpdatePartitionStatisticsAction::set_partition_statistics`]
//! - `removePartitionStatistics(long)` → [`UpdatePartitionStatisticsAction::remove_partition_statistics`]
//!
//! This is the partition-statistics analog of
//! [`UpdateStatisticsAction`](crate::transaction::Transaction::update_statistics) for plain
//! [`StatisticsFile`](crate::spec::StatisticsFile). It emits
//! [`TableUpdate::SetPartitionStatistics`] / [`TableUpdate::RemovePartitionStatistics`], which the
//! metadata builder applies via `set_partition_statistics` / `remove_partition_statistics` (a set
//! REPLACES any prior entry for the same snapshot id — Java's `statsToSet` is a map keyed by
//! snapshot id).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::spec::PartitionStatisticsFile;
use crate::table::Table;
use crate::transaction::{ActionCommit, TransactionAction};
use crate::{Result, TableRequirement, TableUpdate};

/// A transactional action for updating partition-statistics files in a table — the Rust port of Java
/// `UpdatePartitionStatistics`.
pub struct UpdatePartitionStatisticsAction {
    statistics_to_set: HashMap<i64, Option<PartitionStatisticsFile>>,
}

impl UpdatePartitionStatisticsAction {
    pub fn new() -> Self {
        Self {
            statistics_to_set: HashMap::default(),
        }
    }

    /// Set the table's partition-statistics file for the given snapshot, replacing the previous file
    /// for that snapshot if any exists. The snapshot id of the statistics file is used as the key
    /// (Java `setPartitionStatistics(PartitionStatisticsFile)`).
    ///
    /// # Arguments
    ///
    /// * `statistics_file` - The [`PartitionStatisticsFile`] to associate with its corresponding
    ///   snapshot ID.
    ///
    /// # Returns
    ///
    /// An updated [`UpdatePartitionStatisticsAction`] with the new statistics file applied.
    pub fn set_partition_statistics(mut self, statistics_file: PartitionStatisticsFile) -> Self {
        self.statistics_to_set
            .insert(statistics_file.snapshot_id, Some(statistics_file));
        self
    }

    /// Remove the table's partition-statistics file for the given snapshot (Java
    /// `removePartitionStatistics(long)`).
    ///
    /// # Arguments
    ///
    /// * `snapshot_id` - The ID of the snapshot whose partition-statistics file should be removed.
    ///
    /// # Returns
    ///
    /// An updated [`UpdatePartitionStatisticsAction`] with the removal operation recorded.
    pub fn remove_partition_statistics(mut self, snapshot_id: i64) -> Self {
        self.statistics_to_set.insert(snapshot_id, None);
        self
    }
}

impl Default for UpdatePartitionStatisticsAction {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TransactionAction for UpdatePartitionStatisticsAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        let mut updates: Vec<TableUpdate> = vec![];

        self.statistics_to_set
            .iter()
            .for_each(|(snapshot_id, statistic_file)| {
                if let Some(partition_statistics) = statistic_file {
                    updates.push(TableUpdate::SetPartitionStatistics {
                        partition_statistics: partition_statistics.clone(),
                    })
                } else {
                    updates.push(TableUpdate::RemovePartitionStatistics {
                        snapshot_id: *snapshot_id,
                    })
                }
            });

        // A non-snapshot metadata update carries only `AssertTableUUID` (Java
        // `UpdateRequirements.forUpdateTable`). Unlike snapshot-producing actions, the partition-stats
        // `SetPartitionStatistics`/`RemovePartitionStatistics` updates change no ref, so the Transaction
        // would otherwise attach NO requirement (it aggregates only what each action returns) — the
        // commit would be unconditionally accepted. Attaching `UuidMatch` here makes this seam's commit
        // identical to the proven `register_partition_stats_file` direct path (which also attaches only
        // `UuidMatch`), guarding against committing onto a different table.
        let requirements = vec![TableRequirement::UuidMatch {
            uuid: table.metadata().uuid(),
        }];

        Ok(ActionCommit::new(updates, requirements))
    }
}

#[cfg(test)]
mod tests {
    use as_any::Downcast;

    use crate::spec::PartitionStatisticsFile;
    use crate::transaction::update_partition_statistics::UpdatePartitionStatisticsAction;
    use crate::transaction::{ApplyTransactionAction, Transaction};

    fn partition_stats_file(snapshot_id: i64, path: &str) -> PartitionStatisticsFile {
        PartitionStatisticsFile {
            snapshot_id,
            statistics_path: path.to_string(),
            file_size_in_bytes: 42,
        }
    }

    #[test]
    fn test_update_partition_statistics() {
        let table = crate::transaction::tests::make_v2_table();
        let tx = Transaction::new(&table);

        let file_1 = partition_stats_file(3055729675574597004i64, "s3://a/b/p-stats-1.parquet");
        let file_2 = partition_stats_file(3366729675595277004i64, "s3://a/b/p-stats-2.parquet");

        // set file_1 and file_2, then remove file_1.
        let tx = tx
            .update_partition_statistics()
            .set_partition_statistics(file_1.clone())
            .set_partition_statistics(file_2.clone())
            .remove_partition_statistics(file_1.snapshot_id)
            .apply(tx)
            .unwrap();

        let action = (*tx.actions[0])
            .downcast_ref::<UpdatePartitionStatisticsAction>()
            .unwrap();
        // file_1 should have been removed (its key maps to None).
        assert!(
            action
                .statistics_to_set
                .get(&file_1.snapshot_id)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            action
                .statistics_to_set
                .get(&file_2.snapshot_id)
                .unwrap()
                .clone(),
            Some(file_2)
        );
    }

    #[test]
    fn test_set_single_partition_statistics() {
        let table = crate::transaction::tests::make_v2_table();
        let tx = Transaction::new(&table);

        let file = partition_stats_file(1234567890i64, "s3://a/b/p-stats.parquet");

        let tx = tx
            .update_partition_statistics()
            .set_partition_statistics(file.clone())
            .apply(tx)
            .unwrap();

        let action = (*tx.actions[0])
            .downcast_ref::<UpdatePartitionStatisticsAction>()
            .unwrap();
        assert_eq!(
            action
                .statistics_to_set
                .get(&file.snapshot_id)
                .unwrap()
                .clone(),
            Some(file)
        );
    }

    #[test]
    fn test_no_partition_statistics_set() {
        let table = crate::transaction::tests::make_v2_table();
        let tx = Transaction::new(&table);

        let tx = tx.update_partition_statistics().apply(tx).unwrap();

        let action = (*tx.actions[0])
            .downcast_ref::<UpdatePartitionStatisticsAction>()
            .unwrap();
        assert!(action.statistics_to_set.is_empty());
    }

    #[tokio::test]
    async fn test_commit_emits_set_update_and_uuid_requirement() {
        use crate::transaction::TransactionAction;

        let table = crate::transaction::tests::make_v2_table();
        let file = partition_stats_file(3055729675574597004i64, "s3://a/b/p-stats.parquet");

        let action = std::sync::Arc::new(
            UpdatePartitionStatisticsAction::new().set_partition_statistics(file.clone()),
        );
        let mut action_commit = action.commit(&table).await.expect("commit");

        let updates = action_commit.take_updates();
        assert_eq!(updates.len(), 1);
        match &updates[0] {
            crate::TableUpdate::SetPartitionStatistics {
                partition_statistics,
            } => assert_eq!(partition_statistics, &file),
            other => panic!("expected SetPartitionStatistics, got {other:?}"),
        }

        // The commit attaches the table-UUID requirement (snag 3) so it matches the proven
        // `register_partition_stats_file` direct path and Java `UpdateRequirements.forUpdateTable`.
        let requirements = action_commit.take_requirements();
        assert_eq!(requirements.len(), 1);
        assert_eq!(requirements[0], crate::TableRequirement::UuidMatch {
            uuid: table.metadata().uuid(),
        });
    }

    #[tokio::test]
    async fn test_commit_emits_remove_update() {
        use crate::transaction::TransactionAction;

        let table = crate::transaction::tests::make_v2_table();
        let action = std::sync::Arc::new(
            UpdatePartitionStatisticsAction::new().remove_partition_statistics(999i64),
        );
        let mut action_commit = action.commit(&table).await.expect("commit");

        let updates = action_commit.take_updates();
        assert_eq!(updates.len(), 1);
        match &updates[0] {
            crate::TableUpdate::RemovePartitionStatistics { snapshot_id } => {
                assert_eq!(*snapshot_id, 999i64)
            }
            other => panic!("expected RemovePartitionStatistics, got {other:?}"),
        }
    }
}
