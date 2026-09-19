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
use crate::commit_transport::{DiscardingGlueCommitTransport, build_commit_transport_parts};

#[cfg(test)]
impl GlueCatalog {
    pub(crate) fn catalog_commit_attempts(&self) -> u64 {
        self.commit_transport.catalog_commit_attempts()
    }

    pub(crate) fn with_commit_transport(
        mut self,
        commit_transport: Arc<dyn GlueCommitTransport>,
    ) -> Self {
        self.commit_transport = commit_transport;
        self
    }

    pub(crate) fn live_commit_transport(&self) -> Arc<dyn GlueCommitTransport> {
        Arc::clone(&self.commit_transport)
    }

    pub(crate) fn commit_transport_drops_responses(&self) -> bool {
        self.commit_transport.is_response_dropping_transport()
    }

    pub(crate) fn for_commit_outcome_tests_at_version(
        file_io: FileIO,
        commit_transport: Arc<dyn GlueCommitTransport>,
        table: Table,
        client: aws_sdk_glue::Client,
        version_id: Option<String>,
    ) -> Self {
        let harness = GlueCommitHarness::new(table, version_id);
        GlueCatalog {
            config: GlueCatalogConfig {
                name: Some("pr5a-glue".to_string()),
                uri: None,
                catalog_id: None,
                warehouse: "memory://pr5a".to_string(),
                props: HashMap::new(),
            },
            client: GlueClient(client),
            file_io,
            commit_transport,
            outcome_harness: Some(harness),
        }
    }

    pub(crate) fn for_commit_fault_tests_at_version(
        file_io: FileIO,
        props: HashMap<String, String>,
        commit_transport_inner: Arc<dyn GlueCommitTransport>,
        table: Table,
        client: aws_sdk_glue::Client,
        version_id: Option<String>,
    ) -> Result<(Self, Option<Arc<DiscardingGlueCommitTransport>>)> {
        let (commit_transport, fault) =
            build_commit_transport_parts(&props, commit_transport_inner)?;
        let harness = GlueCommitHarness::new(table, version_id);
        Ok((
            GlueCatalog {
                config: GlueCatalogConfig {
                    name: Some("pr5a-glue".to_string()),
                    uri: None,
                    catalog_id: None,
                    warehouse: "memory://pr5a".to_string(),
                    props,
                },
                client: GlueClient(client),
                file_io,
                commit_transport,
                outcome_harness: Some(harness),
            },
            fault,
        ))
    }
}
