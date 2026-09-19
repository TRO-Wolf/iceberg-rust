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

#[cfg(feature = "commit-fault-injection")]
mod enabled {
    use std::collections::HashMap;
    use std::sync::Arc;

    use iceberg::io::FileIO;
    use iceberg::spec::FormatVersion;
    use iceberg::table::Table;
    use iceberg::transaction::{ApplyTransactionAction, Transaction};
    use iceberg::{Catalog, CatalogBuilder, ErrorKind, Result};

    use crate::catalog::GlueCatalog;
    use crate::commit_outcome_tests::{data_file, dummy_glue_client, seed_table, unique_ident};
    use crate::commit_transport::{
        DiscardingGlueCommitTransport, GlueCommitScript, GlueCommitTransport,
        ScriptedGlueCommitTransport,
    };
    use crate::{
        AWS_ACCESS_KEY_ID, AWS_REGION_NAME, AWS_SECRET_ACCESS_KEY,
        GLUE_CATALOG_PROP_DROP_UPDATE_TABLE_RESPONSE, GLUE_CATALOG_PROP_WAREHOUSE,
        GLUE_COMMIT_OPERATION_ID_PROP, GlueCatalogBuilder,
    };

    async fn fault_catalog(
        scripts: impl IntoIterator<Item = GlueCommitScript>,
        props: HashMap<String, String>,
    ) -> Result<(
        GlueCatalog,
        Table,
        Arc<ScriptedGlueCommitTransport>,
        Option<Arc<DiscardingGlueCommitTransport>>,
    )> {
        let file_io = FileIO::new_with_memory();
        let ident = unique_ident();
        let table = seed_table(&file_io, &ident, FormatVersion::V2).await;
        let scripted = ScriptedGlueCommitTransport::new(scripts);
        let client = dummy_glue_client().await;
        let (catalog, fault) = GlueCatalog::for_commit_fault_tests_at_version(
            file_io,
            props,
            Arc::clone(&scripted) as Arc<dyn GlueCommitTransport>,
            table.clone(),
            client,
            Some("v0".to_string()),
        )?;
        Ok((catalog, table, scripted, fault))
    }

    fn drop_props(count: u64) -> HashMap<String, String> {
        HashMap::from([(
            GLUE_CATALOG_PROP_DROP_UPDATE_TABLE_RESPONSE.to_string(),
            count.to_string(),
        )])
    }

    #[tokio::test]
    async fn dropped_append_response_reconciles_with_one_catalog_commit() {
        let (catalog, table, scripted, fault) =
            fault_catalog([GlueCommitScript::Success], drop_props(1))
                .await
                .expect("fault catalog builds through the property path");
        let fault = fault.expect("the property installs the response-dropping wrapper");
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![data_file("memory://pr5a/files/fault-append.parquet")])
            .set_snapshot_properties(HashMap::from([(
                GLUE_COMMIT_OPERATION_ID_PROP.to_string(),
                "op-fault-append".to_string(),
            )]))
            .apply(tx)
            .expect("apply append");
        let committed = tx
            .commit(&catalog)
            .await
            .expect("a dropped successful response reconciles");
        assert!(fault.observed_accepted_response_lost());
        assert_eq!(catalog.catalog_commit_attempts(), 1);
        assert_eq!(scripted.catalog_commit_attempts(), 1);
        let reloaded = catalog
            .load_table(committed.identifier())
            .await
            .expect("reload after reconcile");
        let head = reloaded
            .metadata()
            .current_snapshot()
            .expect("the appended snapshot is current");
        assert_eq!(
            head.summary()
                .additional_properties
                .get(GLUE_COMMIT_OPERATION_ID_PROP)
                .map(String::as_str),
            Some("op-fault-append")
        );
    }

    #[tokio::test]
    async fn dropped_metadata_only_commit_stays_unknown_and_names_the_operation_id() {
        let (catalog, table, _scripted, fault) =
            fault_catalog([GlueCommitScript::Success], drop_props(1))
                .await
                .expect("fault catalog builds through the property path");
        let fault = fault.expect("the property installs the response-dropping wrapper");
        let tx = Transaction::new(&table);
        let tx = tx
            .update_table_properties()
            .set(
                GLUE_COMMIT_OPERATION_ID_PROP.to_string(),
                "op-fault-metadata".to_string(),
            )
            .apply(tx)
            .expect("apply property update");
        let error = tx
            .commit(&catalog)
            .await
            .expect_err("a metadata-only commit with a dropped response stays unknown");
        assert_eq!(error.kind(), ErrorKind::CommitStateUnknown);
        assert!(!error.retryable());
        assert!(fault.observed_accepted_response_lost());
        let rendered = format!("{error}");
        assert!(
            rendered.contains(GLUE_COMMIT_OPERATION_ID_PROP)
                && rendered.contains("op-fault-metadata"),
            "the unknown error must name the stamped operation id: {rendered}"
        );
        assert_eq!(catalog.catalog_commit_attempts(), 1);
    }

    #[tokio::test]
    async fn drop_count_two_discards_two_responses_then_the_third_passes() {
        let (catalog, table, _scripted, fault) = fault_catalog(
            [
                GlueCommitScript::Success,
                GlueCommitScript::Success,
                GlueCommitScript::Success,
            ],
            drop_props(2),
        )
        .await
        .expect("fault catalog builds through the property path");
        let _fault = fault.expect("the property installs the response-dropping wrapper");
        let mut base = table;
        for round in 0..2 {
            base = catalog
                .load_table(base.identifier())
                .await
                .expect("reload the landed base");
            let tx = Transaction::new(&base);
            let tx = tx
                .update_table_properties()
                .set(format!("fault.round.{round}"), "x".to_string())
                .apply(tx)
                .expect("apply property update");
            let error = tx
                .commit(&catalog)
                .await
                .expect_err("the first two successful responses are dropped");
            assert_eq!(error.kind(), ErrorKind::CommitStateUnknown);
        }
        base = catalog
            .load_table(base.identifier())
            .await
            .expect("reload the landed base");
        let tx = Transaction::new(&base);
        let tx = tx
            .update_table_properties()
            .set("fault.round.2".to_string(), "x".to_string())
            .apply(tx)
            .expect("apply property update");
        tx.commit(&catalog)
            .await
            .expect("the third successful response passes through");
        assert_eq!(catalog.catalog_commit_attempts(), 3);
    }

    #[tokio::test]
    async fn fault_property_rejects_a_non_integer_count() {
        let error = fault_catalog(
            [GlueCommitScript::Success],
            HashMap::from([(
                GLUE_CATALOG_PROP_DROP_UPDATE_TABLE_RESPONSE.to_string(),
                "bogus".to_string(),
            )]),
        )
        .await
        .expect_err("a non-integer drop count refuses at construction");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error
                .message()
                .contains(GLUE_CATALOG_PROP_DROP_UPDATE_TABLE_RESPONSE)
        );
    }

    #[tokio::test]
    async fn unset_fault_property_leaves_commit_behavior_unchanged() {
        let (catalog, table, _scripted, fault) =
            fault_catalog([GlueCommitScript::Success], HashMap::new())
                .await
                .expect("catalog builds without the fault property");
        assert!(fault.is_none());
        let tx = Transaction::new(&table);
        let tx = tx
            .update_table_properties()
            .set("fault.unset".to_string(), "x".to_string())
            .apply(tx)
            .expect("apply property update");
        tx.commit(&catalog)
            .await
            .expect("without the property a successful response stays success");
        assert_eq!(catalog.catalog_commit_attempts(), 1);
    }

    #[tokio::test]
    async fn public_builder_installs_the_dropping_wrapper_when_the_property_is_set() {
        let catalog = GlueCatalogBuilder::default()
            .load(
                "pr5a-glue",
                HashMap::from([
                    (
                        GLUE_CATALOG_PROP_WAREHOUSE.to_string(),
                        "memory://pr5a".to_string(),
                    ),
                    (AWS_REGION_NAME.to_string(), "us-east-1".to_string()),
                    (AWS_ACCESS_KEY_ID.to_string(), "pr5a".to_string()),
                    (AWS_SECRET_ACCESS_KEY.to_string(), "pr5a".to_string()),
                    (
                        GLUE_CATALOG_PROP_DROP_UPDATE_TABLE_RESPONSE.to_string(),
                        "1".to_string(),
                    ),
                ]),
            )
            .await
            .expect("the public builder path builds the catalog when the property is set");
        assert!(
            catalog.commit_transport_drops_responses(),
            "GlueCatalog::new must route the fault property through build_commit_transport"
        );
    }

    #[tokio::test]
    async fn drop_count_zero_installs_the_wrapper_but_drops_nothing() {
        let (catalog, table, _scripted, fault) =
            fault_catalog([GlueCommitScript::Success], drop_props(0))
                .await
                .expect("fault catalog builds through the property path");
        let fault = fault.expect("n=0 still installs the wrapper");
        let tx = Transaction::new(&table);
        let tx = tx
            .update_table_properties()
            .set("fault.zero".to_string(), "x".to_string())
            .apply(tx)
            .expect("apply property update");
        tx.commit(&catalog)
            .await
            .expect("an empty drop budget never rewrites a response");
        assert!(!fault.observed_accepted_response_lost());
        assert_eq!(catalog.catalog_commit_attempts(), 1);
    }

    #[tokio::test]
    async fn a_failed_call_does_not_consume_the_drop_budget() {
        let (catalog, table, _scripted, fault) = fault_catalog(
            [
                GlueCommitScript::ConcurrentModification,
                GlueCommitScript::Success,
                GlueCommitScript::Success,
            ],
            drop_props(1),
        )
        .await
        .expect("fault catalog builds through the property path");
        let fault = fault.expect("the property installs the response-dropping wrapper");
        let tx = Transaction::new(&table);
        let tx = tx
            .update_table_properties()
            .set("fault.budget.0".to_string(), "x".to_string())
            .apply(tx)
            .expect("apply property update");
        let error = tx
            .commit(&catalog)
            .await
            .expect_err("the retried send's successful response is the one dropped");
        assert_eq!(error.kind(), ErrorKind::CommitStateUnknown);
        assert!(fault.observed_accepted_response_lost());
        let base = catalog
            .load_table(table.identifier())
            .await
            .expect("reload the landed base");
        let tx = Transaction::new(&base);
        let tx = tx
            .update_table_properties()
            .set("fault.budget.1".to_string(), "x".to_string())
            .apply(tx)
            .expect("apply property update");
        tx.commit(&catalog)
            .await
            .expect("the single drop credit was spent, so the next success passes");
        assert_eq!(catalog.catalog_commit_attempts(), 3);
    }
}

#[cfg(not(feature = "commit-fault-injection"))]
mod disabled {
    use std::collections::HashMap;
    use std::sync::Arc;

    use iceberg::io::FileIO;
    use iceberg::spec::FormatVersion;
    use iceberg::{CatalogBuilder, ErrorKind};

    use crate::catalog::GlueCatalog;
    use crate::commit_outcome_tests::{dummy_glue_client, seed_table, unique_ident};
    use crate::commit_transport::{
        GlueCommitScript, GlueCommitTransport, ScriptedGlueCommitTransport,
    };
    use crate::{
        AWS_ACCESS_KEY_ID, AWS_REGION_NAME, AWS_SECRET_ACCESS_KEY,
        GLUE_CATALOG_PROP_DROP_UPDATE_TABLE_RESPONSE, GLUE_CATALOG_PROP_WAREHOUSE,
        GlueCatalogBuilder,
    };

    #[tokio::test]
    async fn fault_property_refuses_at_construction_without_the_feature() {
        let file_io = FileIO::new_with_memory();
        let ident = unique_ident();
        let table = seed_table(&file_io, &ident, FormatVersion::V2).await;
        let scripted = ScriptedGlueCommitTransport::new([GlueCommitScript::Success]);
        let client = dummy_glue_client().await;
        let error = GlueCatalog::for_commit_fault_tests_at_version(
            file_io,
            HashMap::from([(
                GLUE_CATALOG_PROP_DROP_UPDATE_TABLE_RESPONSE.to_string(),
                "1".to_string(),
            )]),
            scripted as Arc<dyn GlueCommitTransport>,
            table,
            client,
            Some("v0".to_string()),
        )
        .expect_err("the fault property refuses without the feature");
        assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
        let rendered = format!("{error}");
        assert!(
            rendered.contains(GLUE_CATALOG_PROP_DROP_UPDATE_TABLE_RESPONSE),
            "the refusal names the property: {rendered}"
        );
        assert!(
            rendered.contains("commit-fault-injection"),
            "the refusal names the feature: {rendered}"
        );
    }

    #[tokio::test]
    async fn public_builder_refuses_the_fault_property_without_the_feature() {
        let error = GlueCatalogBuilder::default()
            .load(
                "pr5a-glue",
                HashMap::from([
                    (
                        GLUE_CATALOG_PROP_WAREHOUSE.to_string(),
                        "memory://pr5a".to_string(),
                    ),
                    (AWS_REGION_NAME.to_string(), "us-east-1".to_string()),
                    (AWS_ACCESS_KEY_ID.to_string(), "pr5a".to_string()),
                    (AWS_SECRET_ACCESS_KEY.to_string(), "pr5a".to_string()),
                    (
                        GLUE_CATALOG_PROP_DROP_UPDATE_TABLE_RESPONSE.to_string(),
                        "1".to_string(),
                    ),
                ]),
            )
            .await
            .expect_err("the public builder path refuses the fault property without the feature");
        assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
        let rendered = format!("{error}");
        assert!(
            rendered.contains(GLUE_CATALOG_PROP_DROP_UPDATE_TABLE_RESPONSE),
            "the refusal names the property: {rendered}"
        );
    }
}
