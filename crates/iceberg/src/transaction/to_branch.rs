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

use crate::transaction::{
    DeleteFilesAction, FastAppendAction, MergeAppendAction, OverwriteFilesAction,
    ReplacePartitionsAction, RewriteFilesAction, RowDeltaAction,
};

impl FastAppendAction {
    /// Commit this snapshot onto `branch` instead of `main`. Java `SnapshotUpdate.toBranch`.
    pub fn to_branch(mut self, branch: impl Into<String>) -> Self {
        self.target_branch = branch.into();
        self
    }
}

impl MergeAppendAction {
    /// Commit this snapshot onto `branch` instead of `main`. Java `SnapshotUpdate.toBranch`.
    pub fn to_branch(mut self, branch: impl Into<String>) -> Self {
        self.target_branch = branch.into();
        self
    }
}

impl OverwriteFilesAction {
    /// Commit this snapshot onto `branch` instead of `main`. Java `SnapshotUpdate.toBranch`.
    pub fn to_branch(mut self, branch: impl Into<String>) -> Self {
        self.target_branch = branch.into();
        self
    }
}

impl ReplacePartitionsAction {
    /// Commit this snapshot onto `branch` instead of `main`. Java `SnapshotUpdate.toBranch`.
    pub fn to_branch(mut self, branch: impl Into<String>) -> Self {
        self.target_branch = branch.into();
        self
    }
}

impl RewriteFilesAction {
    /// Commit this snapshot onto `branch` instead of `main`. Java `SnapshotUpdate.toBranch`.
    pub fn to_branch(mut self, branch: impl Into<String>) -> Self {
        self.target_branch = branch.into();
        self
    }
}

impl RowDeltaAction {
    /// Commit this snapshot onto `branch` instead of `main`. Java `SnapshotUpdate.toBranch`.
    pub fn to_branch(mut self, branch: impl Into<String>) -> Self {
        self.target_branch = branch.into();
        self
    }
}

impl DeleteFilesAction {
    /// Commit this snapshot onto `branch` instead of `main`. Java `SnapshotUpdate.toBranch`.
    pub fn to_branch(mut self, branch: impl Into<String>) -> Self {
        self.target_branch = branch.into();
        self
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, Literal,
        MAIN_BRANCH, Struct,
    };
    use crate::table::Table;
    use crate::transaction::tests::make_v2_minimal_table_in_catalog;
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{Catalog, ErrorKind};

    fn data_file(path: &str, part_value: i64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(part_value))]))
            .build()
            .expect("build fixture data file")
    }

    async fn append_main(catalog: &impl Catalog, table: &Table, path: &str, part: i64) -> Table {
        let tx = Transaction::new(table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![data_file(path, part)])
            .apply(tx)
            .expect("apply main append");
        tx.commit(catalog).await.expect("commit main append")
    }

    fn ref_id(table: &Table, name: &str) -> Option<i64> {
        table
            .metadata()
            .refs
            .get(name)
            .map(|reference| reference.snapshot_id)
    }

    async fn append_branch(
        catalog: &impl Catalog,
        table: &Table,
        branch: &str,
        path: &str,
        part: i64,
    ) -> Table {
        let tx = Transaction::new(table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![data_file(path, part)])
            .to_branch(branch)
            .apply(tx)
            .expect("apply branch append");
        tx.commit(catalog).await.expect("commit branch append")
    }

    fn pos_delete_file(path: &str, part: i64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(part))]))
            .referenced_data_file(Some("test/a.parquet".to_string()))
            .build()
            .expect("build fixture position delete")
    }

    fn dv_file(path: &str, part: i64, referenced: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Puffin)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(part))]))
            .referenced_data_file(Some(referenced.to_string()))
            .content_offset(Some(4))
            .content_size_in_bytes(Some(40))
            .build()
            .expect("build fixture deletion vector")
    }

    #[tokio::test]
    async fn to_branch_existing_branch_does_not_move_main() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_main(&catalog, &table, "test/a.parquet", 0).await;
        let main_id = table.metadata().current_snapshot_id().expect("main head");

        let tx = Transaction::new(&table);
        let tx = tx
            .manage_snapshots()
            .create_branch("audit", main_id)
            .apply(tx)
            .expect("apply create_branch");
        let table = tx.commit(&catalog).await.expect("commit create_branch");
        let table = append_branch(&catalog, &table, "audit", "test/b.parquet", 1).await;
        let branch_head = ref_id(&table, "audit").expect("diverged audit");
        assert_ne!(branch_head, main_id);

        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![data_file("test/c.parquet", 2)])
            .to_branch("audit")
            .apply(tx)
            .expect("apply branch append");
        let table = tx.commit(&catalog).await.expect("commit branch append");

        assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
        let branch_id = ref_id(&table, "audit").expect("audit exists");
        assert_ne!(branch_id, main_id);
        assert_ne!(branch_id, branch_head);
        let branch_snap = table
            .metadata()
            .snapshot_by_id(branch_id)
            .expect("audit snapshot");
        assert_eq!(branch_snap.parent_snapshot_id(), Some(branch_head));
    }

    #[tokio::test]
    async fn to_branch_creates_missing_branch() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_main(&catalog, &table, "test/a.parquet", 0).await;
        let main_id = table.metadata().current_snapshot_id().expect("main head");

        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![data_file("test/b.parquet", 1)])
            .to_branch("audit")
            .apply(tx)
            .expect("apply create-branch append");
        let table = tx
            .commit(&catalog)
            .await
            .expect("commit create-branch append");

        assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
        assert!(table.metadata().refs.contains_key("audit"));
        let branch_id = ref_id(&table, "audit").expect("created audit");
        assert_ne!(branch_id, main_id);
    }

    #[tokio::test]
    async fn to_branch_empty_table_leaves_current_null() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        assert!(table.metadata().current_snapshot_id().is_none());

        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![data_file("test/b.parquet", 0)])
            .to_branch("audit")
            .apply(tx)
            .expect("apply empty-table branch append");
        let table = tx
            .commit(&catalog)
            .await
            .expect("commit empty-table branch");

        assert!(table.metadata().current_snapshot_id().is_none());
        assert!(table.metadata().refs.contains_key("audit"));
        assert!(!table.metadata().refs.contains_key(MAIN_BRANCH));
    }

    #[tokio::test]
    async fn to_branch_tag_is_rejected() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_main(&catalog, &table, "test/a.parquet", 0).await;
        let main_id = table.metadata().current_snapshot_id().expect("main head");

        let tx = Transaction::new(&table);
        let tx = tx
            .manage_snapshots()
            .create_tag("some-tag", main_id)
            .apply(tx)
            .expect("apply create_tag");
        let table = tx.commit(&catalog).await.expect("commit create_tag");

        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![data_file("test/b.parquet", 1)])
            .to_branch("some-tag")
            .apply(tx)
            .expect("apply tagged to_branch");
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("a tag must not be a commit target");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains(
                "some-tag is a tag, not a branch. Tags cannot be targets for producing snapshots"
            ),
            "unexpected message: {}",
            err.message()
        );
    }

    #[tokio::test]
    async fn to_branch_leaves_sibling_refs_byte_stable() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_main(&catalog, &table, "test/a.parquet", 0).await;
        let main_id = table.metadata().current_snapshot_id().expect("main head");

        let tx = Transaction::new(&table);
        let tx = tx
            .manage_snapshots()
            .create_branch("keep", main_id)
            .create_tag("pin", main_id)
            .apply(tx)
            .expect("apply sibling refs");
        let table = tx.commit(&catalog).await.expect("commit sibling refs");
        let before: HashMap<_, _> = table.metadata().refs.clone();

        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![data_file("test/b.parquet", 1)])
            .to_branch("audit")
            .apply(tx)
            .expect("apply audit append");
        let table = tx.commit(&catalog).await.expect("commit audit append");

        for (name, reference) in &before {
            assert_eq!(
                table.metadata().refs.get(name),
                Some(reference),
                "sibling ref {name} moved"
            );
        }
        assert!(table.metadata().refs.contains_key("audit"));
    }

    #[tokio::test]
    async fn to_branch_new_branch_parents_off_current() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_main(&catalog, &table, "test/a.parquet", 0).await;
        let main_id = table.metadata().current_snapshot_id().expect("main head");

        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![data_file("test/b.parquet", 1)])
            .to_branch("audit")
            .apply(tx)
            .expect("apply");
        let table = tx.commit(&catalog).await.expect("commit");

        let branch_id = ref_id(&table, "audit").expect("audit");
        let parent = table
            .metadata()
            .snapshot_by_id(branch_id)
            .expect("snapshot")
            .parent_snapshot_id();
        assert_eq!(parent, Some(main_id));
    }

    #[tokio::test]
    async fn to_branch_retry_resolves_named_branch_not_main() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_main(&catalog, &table, "test/a.parquet", 0).await;
        let main_id = table.metadata().current_snapshot_id().expect("main head");

        let tx = Transaction::new(&table);
        let tx = tx
            .manage_snapshots()
            .create_branch("audit", main_id)
            .apply(tx)
            .expect("apply create_branch");
        let table = tx.commit(&catalog).await.expect("commit create_branch");

        let pending = Transaction::new(&table);
        let pending = pending
            .fast_append()
            .add_data_files(vec![data_file("test/pending.parquet", 2)])
            .to_branch("audit")
            .apply(pending)
            .expect("apply pending");

        let winner = Transaction::new(&table);
        let winner = winner
            .fast_append()
            .add_data_files(vec![data_file("test/winner.parquet", 3)])
            .to_branch("audit")
            .apply(winner)
            .expect("apply winner");
        let table = winner.commit(&catalog).await.expect("winner commit");
        let winner_id = ref_id(&table, "audit").expect("winner audit head");
        assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));

        let table = pending
            .commit(&catalog)
            .await
            .expect("pending must retry onto the named branch");
        assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
        let pending_id = ref_id(&table, "audit").expect("pending audit head");
        assert_ne!(pending_id, winner_id);
        let pending_snap = table
            .metadata()
            .snapshot_by_id(pending_id)
            .expect("pending snapshot");
        assert_eq!(pending_snap.parent_snapshot_id(), Some(winner_id));
    }

    #[tokio::test]
    async fn to_branch_retry_overwrite_resolves_named_branch() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_main(&catalog, &table, "test/a.parquet", 0).await;
        let main_id = table.metadata().current_snapshot_id().expect("main head");

        let tx = Transaction::new(&table);
        let tx = tx
            .manage_snapshots()
            .create_branch("audit", main_id)
            .apply(tx)
            .expect("apply create_branch");
        let table = tx.commit(&catalog).await.expect("commit create_branch");

        let pending = Transaction::new(&table);
        let pending = pending
            .overwrite_files()
            .add_file(data_file("test/pending.parquet", 2))
            .to_branch("audit")
            .apply(pending)
            .expect("apply pending overwrite");

        let winner = Transaction::new(&table);
        let winner = winner
            .overwrite_files()
            .add_file(data_file("test/winner.parquet", 3))
            .to_branch("audit")
            .apply(winner)
            .expect("apply winner overwrite");
        let table = winner.commit(&catalog).await.expect("winner overwrite");
        let winner_id = ref_id(&table, "audit").expect("winner audit head");
        assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));

        let table = pending
            .commit(&catalog)
            .await
            .expect("pending overwrite must retry onto the named branch");
        assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
        let pending_id = ref_id(&table, "audit").expect("pending audit head");
        assert_ne!(pending_id, winner_id);
        let pending_snap = table
            .metadata()
            .snapshot_by_id(pending_id)
            .expect("pending snapshot");
        assert_eq!(pending_snap.parent_snapshot_id(), Some(winner_id));
    }

    #[tokio::test]
    async fn to_branch_missing_ref_conflict_validation_does_not_see_parent_files() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_main(&catalog, &table, "test/a.parquet", 0).await;
        let main_id = table.metadata().current_snapshot_id().expect("main head");

        let tx = Transaction::new(&table);
        let tx = tx
            .overwrite_files()
            .add_file(data_file("test/b.parquet", 1))
            .validate_no_conflicting_data()
            .to_branch("audit")
            .apply(tx)
            .expect("apply missing-branch overwrite");
        let table = tx.commit(&catalog).await.expect(
            "missing-branch conflict validation must not treat the parent snapshot as concurrent",
        );
        assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
        assert!(table.metadata().refs.contains_key("audit"));
    }

    #[tokio::test]
    async fn to_branch_diverged_conflict_validation_uses_branch_head_not_main() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_main(&catalog, &table, "test/a.parquet", 0).await;
        let main_id = table.metadata().current_snapshot_id().expect("main head");

        let tx = Transaction::new(&table);
        let tx = tx
            .manage_snapshots()
            .create_branch("audit", main_id)
            .apply(tx)
            .expect("apply create_branch");
        let table = tx.commit(&catalog).await.expect("commit create_branch");
        let table = append_branch(&catalog, &table, "audit", "test/b.parquet", 1).await;
        let branch_head = ref_id(&table, "audit").expect("diverged audit");
        assert_ne!(branch_head, main_id);

        let tx = Transaction::new(&table);
        let tx = tx
            .overwrite_files()
            .add_file(data_file("test/c.parquet", 2))
            .validate_no_conflicting_data()
            .to_branch("audit")
            .apply(tx)
            .expect("apply diverged overwrite");
        let table = tx
            .commit(&catalog)
            .await
            .expect("branch-head validation must not treat the branch's own files as concurrent");
        assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
        assert_ne!(ref_id(&table, "audit").expect("audit"), branch_head);
    }

    #[tokio::test]
    async fn to_branch_conflict_validation_ignores_concurrent_main_writes() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_main(&catalog, &table, "test/a.parquet", 0).await;
        let main_id = table.metadata().current_snapshot_id().expect("main head");

        let tx = Transaction::new(&table);
        let tx = tx
            .manage_snapshots()
            .create_branch("audit", main_id)
            .apply(tx)
            .expect("apply create_branch");
        let table = tx.commit(&catalog).await.expect("commit create_branch");
        let table = append_branch(&catalog, &table, "audit", "test/b.parquet", 1).await;

        let pending = Transaction::new(&table);
        let pending = pending
            .overwrite_files()
            .add_file(data_file("test/c.parquet", 2))
            .validate_no_conflicting_data()
            .to_branch("audit")
            .apply(pending)
            .expect("apply pending overwrite");

        let table = append_main(&catalog, &table, "test/main-only.parquet", 3).await;
        assert_ne!(
            table.metadata().current_snapshot_id().expect("moved main"),
            main_id
        );

        pending
            .commit(&catalog)
            .await
            .expect("a concurrent main write must not conflict with a branch commit");
    }

    #[tokio::test]
    async fn to_branch_conflict_validation_rejects_concurrent_branch_writes() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_main(&catalog, &table, "test/a.parquet", 0).await;
        let main_id = table.metadata().current_snapshot_id().expect("main head");

        let tx = Transaction::new(&table);
        let tx = tx
            .manage_snapshots()
            .create_branch("audit", main_id)
            .apply(tx)
            .expect("apply create_branch");
        let table = tx.commit(&catalog).await.expect("commit create_branch");
        let table = append_branch(&catalog, &table, "audit", "test/b.parquet", 1).await;

        let pending = Transaction::new(&table);
        let pending = pending
            .overwrite_files()
            .add_file(data_file("test/c.parquet", 2))
            .validate_no_conflicting_data()
            .to_branch("audit")
            .apply(pending)
            .expect("apply pending overwrite");

        let _table = append_branch(&catalog, &table, "audit", "test/branch-only.parquet", 3).await;

        let err = pending
            .commit(&catalog)
            .await
            .expect_err("a concurrent branch write must conflict");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("Found conflicting files"),
            "unexpected message: {}",
            err.message()
        );
    }

    #[tokio::test]
    async fn to_branch_fresh_dv_rejects_branch_live_position_delete() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_main(&catalog, &table, "test/a.parquet", 0).await;
        let main_id = table.metadata().current_snapshot_id().expect("main head");

        let tx = Transaction::new(&table);
        let tx = tx
            .manage_snapshots()
            .create_branch("audit", main_id)
            .apply(tx)
            .expect("apply create_branch");
        let table = tx.commit(&catalog).await.expect("commit create_branch");

        let tx = Transaction::new(&table);
        let tx = tx
            .row_delta()
            .add_deletes(vec![pos_delete_file("test/a-pos.parquet", 0)])
            .to_branch("audit")
            .apply(tx)
            .expect("apply branch position delete");
        let table = tx
            .commit(&catalog)
            .await
            .expect("commit branch position delete");

        let tx = Transaction::new(&table);
        let tx = tx
            .upgrade_table_version()
            .set_format_version(FormatVersion::V3)
            .apply(tx)
            .expect("apply v3 upgrade");
        let table = tx.commit(&catalog).await.expect("commit v3 upgrade");

        let tx = Transaction::new(&table);
        let tx = tx
            .row_delta()
            .add_deletes(vec![dv_file("test/a-dv.puffin", 0, "test/a.parquet")])
            .to_branch("audit")
            .apply(tx)
            .expect("apply branch DV");
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("a DV must reject a live position delete on the named branch");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("live position delete"),
            "unexpected message: {}",
            err.message()
        );
    }

    #[tokio::test]
    async fn to_branch_fresh_dv_ignores_main_only_position_delete() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_main(&catalog, &table, "test/a.parquet", 0).await;
        let main_id = table.metadata().current_snapshot_id().expect("main head");

        let tx = Transaction::new(&table);
        let tx = tx
            .manage_snapshots()
            .create_branch("audit", main_id)
            .apply(tx)
            .expect("apply create_branch");
        let table = tx.commit(&catalog).await.expect("commit create_branch");

        let tx = Transaction::new(&table);
        let tx = tx
            .row_delta()
            .add_deletes(vec![pos_delete_file("test/a-pos.parquet", 0)])
            .apply(tx)
            .expect("apply main position delete");
        let table = tx
            .commit(&catalog)
            .await
            .expect("commit main position delete");

        let tx = Transaction::new(&table);
        let tx = tx
            .upgrade_table_version()
            .set_format_version(FormatVersion::V3)
            .apply(tx)
            .expect("apply v3 upgrade");
        let table = tx.commit(&catalog).await.expect("commit v3 upgrade");

        let tx = Transaction::new(&table);
        let tx = tx
            .row_delta()
            .add_deletes(vec![dv_file("test/a-dv.puffin", 0, "test/a.parquet")])
            .to_branch("audit")
            .apply(tx)
            .expect("apply branch DV");
        let main_before_dv = table.metadata().current_snapshot_id();
        let table = tx
            .commit(&catalog)
            .await
            .expect("a main-only position delete must not reject a branch DV");
        assert_eq!(table.metadata().current_snapshot_id(), main_before_dv);
        assert!(table.metadata().refs.contains_key("audit"));
    }

    #[derive(Debug, Clone, Copy)]
    enum Producer {
        FastAppend,
        MergeAppend,
        OverwriteFiles,
        ReplacePartitions,
        RewriteFiles,
        RowDelta,
        DeleteFiles,
    }

    #[tokio::test]
    async fn every_snapshot_producer_commits_to_named_branch() {
        for producer in [
            Producer::FastAppend,
            Producer::MergeAppend,
            Producer::OverwriteFiles,
            Producer::ReplacePartitions,
            Producer::RewriteFiles,
            Producer::RowDelta,
            Producer::DeleteFiles,
        ] {
            let catalog = new_memory_catalog().await;
            let table = make_v2_minimal_table_in_catalog(&catalog).await;
            let seed = data_file("test/seed.parquet", 0);
            let table = append_main(&catalog, &table, seed.file_path(), 0).await;
            let main_id = table.metadata().current_snapshot_id().expect("main");

            let probe = data_file("test/probe.parquet", 0);
            let tx = Transaction::new(&table);
            let tx = match producer {
                Producer::FastAppend => tx
                    .fast_append()
                    .add_data_files(vec![probe])
                    .to_branch("audit")
                    .apply(tx),
                Producer::MergeAppend => tx
                    .merge_append()
                    .add_data_files(vec![probe])
                    .to_branch("audit")
                    .apply(tx),
                Producer::OverwriteFiles => tx
                    .overwrite_files()
                    .add_file(probe)
                    .delete_file(seed.file_path().to_string())
                    .to_branch("audit")
                    .apply(tx),
                Producer::ReplacePartitions => tx
                    .replace_partitions()
                    .add_file(probe)
                    .to_branch("audit")
                    .apply(tx),
                Producer::RewriteFiles => tx
                    .rewrite_files(vec![seed.clone()], vec![probe])
                    .to_branch("audit")
                    .apply(tx),
                Producer::RowDelta => tx
                    .row_delta()
                    .add_data_files(vec![probe])
                    .to_branch("audit")
                    .apply(tx),
                Producer::DeleteFiles => tx
                    .delete_files()
                    .delete_file(seed.file_path().to_string())
                    .to_branch("audit")
                    .apply(tx),
            }
            .unwrap_or_else(|err| panic!("{producer:?} apply failed: {err}"));
            let table = tx
                .commit(&catalog)
                .await
                .unwrap_or_else(|err| panic!("{producer:?} commit failed: {err}"));

            assert_eq!(
                table.metadata().current_snapshot_id(),
                Some(main_id),
                "{producer:?} moved main"
            );
            assert!(
                table.metadata().refs.contains_key("audit"),
                "{producer:?} did not create audit"
            );
            let branch_id = ref_id(&table, "audit").expect("audit");
            assert_ne!(branch_id, main_id, "{producer:?} left audit on main");
        }
    }
}
