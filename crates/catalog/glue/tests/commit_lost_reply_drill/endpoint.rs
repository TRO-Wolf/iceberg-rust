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

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use iceberg::io::MemoryStorageFactory;
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, NestedField, PrimitiveType, Schema,
    Struct, TableMetadata, Type,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation};
use iceberg_catalog_glue::{
    AWS_ACCESS_KEY_ID, AWS_REGION_NAME, AWS_SECRET_ACCESS_KEY, GLUE_CATALOG_PROP_URI,
    GLUE_CATALOG_PROP_WAREHOUSE, GlueCatalog, GlueCatalogBuilder,
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;

pub(crate) const NAMESPACE: &str = "drill";
pub(crate) const TABLE: &str = "t";
pub(crate) const LOCATION: &str = "memory://warehouse/drill.db/t";
pub(crate) const MARKER: &str = "drill.append";
pub(crate) const ROWS_PER_APPEND: u64 = 3;
pub(crate) const REFUSAL_WINDOW: Duration = Duration::from_secs(1);
pub(crate) const LISTENER_DEADLINE: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CommitScript {
    ApplyThenReset,
    ApplyThenConflict,
    DropThenReset,
    ApplyThenClose,
    DropThenClose,
    ApplyThenError(u16, &'static str),
    DropThenError(u16, &'static str),
    ApplyThenGarbage,
    DropThenGarbage,
    DeferThenConflict,
    LandDeferredThenConflict,
    LandDeferredThenReset,
    ApplyThenUnreadable,
    DropThenUnreadable,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Answer {
    Answered,
    AppliedThenReset,
    AppliedThenConflict,
    DroppedThenReset,
    AppliedThenClosed,
    DroppedThenClosed,
    AppliedThenError(u16),
    DroppedThenError(u16),
    AppliedThenGarbage,
    DroppedThenGarbage,
    DeferredThenConflict,
    LandedDeferredThenConflict,
    LandedDeferredThenReset,
    AppliedThenUnreadable,
    DroppedThenUnreadable,
    RejectedStale,
    Reset,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Logged {
    pub(crate) operation: String,
    pub(crate) version_sent: Option<String>,
    pub(crate) metadata_sent: Option<String>,
    pub(crate) answer: Answer,
}

pub(crate) enum Reply {
    Json {
        status: u16,
        error_type: Option<&'static str>,
        body: String,
    },
    Reset,
    Close,
    Garbage,
    Unreadable,
}

#[derive(Clone, Copy)]
pub(crate) enum Connections {
    KeepAlive,
    OnePerRequest,
}

#[derive(Default)]
pub(crate) struct Refusal {
    pub(crate) close_listener: Notify,
    pub(crate) listener_closed: Notify,
    pub(crate) listening_again: Notify,
}

#[derive(Default)]
pub(crate) struct GlueState {
    pub(crate) metadata_location: Option<String>,
    pub(crate) version: u64,
    pub(crate) commit_scripts: VecDeque<CommitScript>,
    pub(crate) read_resets: usize,
    pub(crate) close_connections: bool,
    pub(crate) refuse_after_reads: Option<usize>,
    pub(crate) refusing: bool,
    pub(crate) deferred: Option<String>,
    pub(crate) log: Vec<Logged>,
}

impl GlueState {
    pub(crate) fn count_down_refusal(&mut self) {
        match self.refuse_after_reads {
            Some(remaining) if remaining <= 1 => {
                self.refuse_after_reads = None;
                self.refusing = true;
            }
            Some(remaining) => self.refuse_after_reads = Some(remaining - 1),
            None => {}
        }
    }

    pub(crate) fn handle(&mut self, operation: &str, body: &Value) -> Reply {
        let (answer, version_sent, reply) = match operation {
            "CreateTable" => {
                self.metadata_location = new_metadata_location(body);
                self.version = 1;
                (Answer::Answered, None, answered(json!({})))
            }
            "GetTable" if self.read_resets > 0 => {
                self.read_resets -= 1;
                (Answer::Reset, None, Reply::Reset)
            }
            "GetTable" => {
                self.count_down_refusal();
                (Answer::Answered, None, self.get_table())
            }
            "UpdateTable" => self.update_table(body),
            _ => (
                Answer::Answered,
                None,
                rejected("InvalidInputException", "unsupported by the drill endpoint"),
            ),
        };
        let metadata_sent = (operation == "UpdateTable")
            .then(|| new_metadata_location(body))
            .flatten();
        self.log.push(Logged {
            operation: operation.to_string(),
            version_sent,
            metadata_sent,
            answer,
        });
        reply
    }

    pub(crate) fn get_table(&self) -> Reply {
        match &self.metadata_location {
            Some(location) => answered(json!({
                "Table": {
                    "Name": TABLE,
                    "DatabaseName": NAMESPACE,
                    "VersionId": self.version.to_string(),
                    "Parameters": {"metadata_location": location, "table_type": "ICEBERG"},
                }
            })),
            None => rejected("EntityNotFoundException", "no drill table"),
        }
    }

    pub(crate) fn update_table(&mut self, body: &Value) -> (Answer, Option<String>, Reply) {
        let version_sent = body["VersionId"].as_str().map(str::to_string);
        if version_sent.as_deref() != Some(self.version.to_string().as_str()) {
            return (
                Answer::RejectedStale,
                version_sent,
                rejected("ConcurrentModificationException", "stale VersionId"),
            );
        }
        let script = self.commit_scripts.pop_front();
        match script {
            Some(CommitScript::DropThenReset) => {
                return (Answer::DroppedThenReset, version_sent, Reply::Reset);
            }
            Some(CommitScript::DropThenClose) => {
                return (Answer::DroppedThenClosed, version_sent, Reply::Close);
            }
            Some(CommitScript::DropThenError(status, code)) => {
                return (
                    Answer::DroppedThenError(status),
                    version_sent,
                    failed(status, code),
                );
            }
            Some(CommitScript::DropThenGarbage) => {
                return (Answer::DroppedThenGarbage, version_sent, Reply::Garbage);
            }
            Some(CommitScript::DropThenUnreadable) => {
                return (
                    Answer::DroppedThenUnreadable,
                    version_sent,
                    Reply::Unreadable,
                );
            }
            Some(CommitScript::LandDeferredThenReset) => {
                self.metadata_location = self.deferred.take();
                self.version += 1;
                return (Answer::LandedDeferredThenReset, version_sent, Reply::Reset);
            }
            Some(CommitScript::DeferThenConflict) => {
                self.deferred = new_metadata_location(body);
                return (
                    Answer::DeferredThenConflict,
                    version_sent,
                    rejected("ConcurrentModificationException", "stale VersionId"),
                );
            }
            Some(CommitScript::LandDeferredThenConflict) => {
                self.metadata_location = self.deferred.take();
                self.version += 1;
                return (
                    Answer::LandedDeferredThenConflict,
                    version_sent,
                    rejected("ConcurrentModificationException", "stale VersionId"),
                );
            }
            _ => {}
        }
        self.metadata_location = new_metadata_location(body);
        self.version += 1;
        match script {
            Some(CommitScript::ApplyThenReset) => {
                (Answer::AppliedThenReset, version_sent, Reply::Reset)
            }
            Some(CommitScript::ApplyThenClose) => {
                (Answer::AppliedThenClosed, version_sent, Reply::Close)
            }
            Some(CommitScript::ApplyThenError(status, code)) => (
                Answer::AppliedThenError(status),
                version_sent,
                failed(status, code),
            ),
            Some(CommitScript::ApplyThenGarbage) => {
                (Answer::AppliedThenGarbage, version_sent, Reply::Garbage)
            }
            Some(CommitScript::ApplyThenUnreadable) => (
                Answer::AppliedThenUnreadable,
                version_sent,
                Reply::Unreadable,
            ),
            Some(CommitScript::ApplyThenConflict) => (
                Answer::AppliedThenConflict,
                version_sent,
                rejected("ConcurrentModificationException", "stale VersionId"),
            ),
            _ => (Answer::Answered, version_sent, answered(json!({}))),
        }
    }
}

pub(crate) fn new_metadata_location(body: &Value) -> Option<String> {
    body["TableInput"]["Parameters"]["metadata_location"]
        .as_str()
        .map(str::to_string)
}

pub(crate) fn answered(body: Value) -> Reply {
    Reply::Json {
        status: 200,
        error_type: None,
        body: body.to_string(),
    }
}

pub(crate) fn failed(status: u16, error_type: &'static str) -> Reply {
    Reply::Json {
        status,
        error_type: Some(error_type),
        body: json!({"__type": error_type, "Message": "failed by the drill endpoint"}).to_string(),
    }
}

pub(crate) fn rejected(error_type: &'static str, message: &str) -> Reply {
    Reply::Json {
        status: 400,
        error_type: Some(error_type),
        body: json!({"__type": error_type, "Message": message}).to_string(),
    }
}

pub(crate) struct Request {
    pub(crate) head: String,
    pub(crate) body: Vec<u8>,
}

impl Request {
    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.head.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.trim()
                .eq_ignore_ascii_case(name)
                .then_some(value.trim())
        })
    }
}

pub(crate) async fn read_more(stream: &mut TcpStream, buffer: &mut Vec<u8>) -> Option<()> {
    let mut chunk = [0u8; 8192];
    let read = stream
        .read(&mut chunk)
        .await
        .ok()
        .filter(|read| *read > 0)?;
    buffer.extend_from_slice(&chunk[..read]);
    Some(())
}

pub(crate) async fn read_request(stream: &mut TcpStream) -> Option<Request> {
    let mut buffer = Vec::new();
    let head_end = loop {
        if let Some(index) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
        read_more(stream, &mut buffer).await?;
    };
    let mut request = Request {
        head: String::from_utf8_lossy(&buffer[..head_end]).into_owned(),
        body: Vec::new(),
    };
    let length = request
        .header("content-length")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    while buffer.len() < head_end + length {
        read_more(stream, &mut buffer).await?;
    }
    request.body = buffer[head_end..head_end + length].to_vec();
    Some(request)
}

pub(crate) async fn write_reply(stream: &mut TcpStream, reply: Reply, close_after: bool) -> bool {
    let (status, error_type, body) = match reply {
        Reply::Json {
            status,
            error_type,
            body,
        } => (status, error_type, body),
        Reply::Reset => {
            let _ = stream.set_zero_linger();
            return false;
        }
        Reply::Close => return false,
        Reply::Garbage => {
            let _ = stream.write_all(b"HTTP/1.1 abc\r\n\r\n").await;
            return false;
        }
        Reply::Unreadable => {
            let _ = stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 9\r\n\r\n{not json",
                )
                .await;
            return false;
        }
    };
    let mut head = format!(
        "HTTP/1.1 {status} {}\r\ncontent-type: application/x-amz-json-1.1\r\ncontent-length: {}\r\n",
        if status == 200 { "OK" } else { "Bad Request" },
        body.len()
    );
    if let Some(error_type) = error_type {
        head.push_str(&format!("x-amzn-errortype: {error_type}\r\n"));
    }
    if close_after {
        head.push_str("connection: close\r\n");
    }
    head.push_str("\r\n");
    let written = stream.write_all(head.as_bytes()).await.is_ok()
        && stream.write_all(body.as_bytes()).await.is_ok();
    written && !close_after
}

pub(crate) async fn serve_connection(
    mut stream: TcpStream,
    state: Arc<Mutex<GlueState>>,
    refusal: Arc<Refusal>,
) {
    while let Some(request) = read_request(&mut stream).await {
        let operation = request
            .header("x-amz-target")
            .and_then(|target| target.strip_prefix("AWSGlue."))
            .unwrap_or("")
            .to_string();
        let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
        let reply = state
            .lock()
            .expect("drill endpoint state")
            .handle(&operation, &body);
        let (close_after, refuse) = {
            let mut state = state.lock().expect("drill endpoint state");
            (state.close_connections, std::mem::take(&mut state.refusing))
        };
        if refuse {
            refusal.close_listener.notify_one();
            refusal.listener_closed.notified().await;
        }
        if !write_reply(&mut stream, reply, close_after).await {
            return;
        }
    }
}

pub(crate) async fn accept_loop(
    mut listener: TcpListener,
    state: Arc<Mutex<GlueState>>,
    refusal: Arc<Refusal>,
) {
    let Ok(address) = listener.local_addr() else {
        return;
    };
    loop {
        let accepted = tokio::select! {
            accepted = listener.accept() => Some(accepted),
            () = refusal.close_listener.notified() => None,
        };
        match accepted {
            Some(Ok((stream, _))) => {
                tokio::spawn(serve_connection(
                    stream,
                    Arc::clone(&state),
                    Arc::clone(&refusal),
                ));
            }
            Some(Err(_)) => return,
            None => {
                drop(listener);
                refusal.listener_closed.notify_one();
                tokio::time::sleep(REFUSAL_WINDOW).await;
                let Ok(rebound) = TcpListener::bind(address).await else {
                    return;
                };
                listener = rebound;
                refusal.listening_again.notify_one();
            }
        }
    }
}

pub(crate) struct FakeGlue {
    pub(crate) url: String,
    pub(crate) state: Arc<Mutex<GlueState>>,
    pub(crate) refusal: Arc<Refusal>,
}

impl FakeGlue {
    pub(crate) async fn start(connections: Connections) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the drill endpoint");
        let address = listener.local_addr().expect("drill endpoint address");
        let state = Arc::new(Mutex::new(GlueState {
            close_connections: matches!(connections, Connections::OnePerRequest),
            ..GlueState::default()
        }));
        let refusal = Arc::new(Refusal::default());
        tokio::spawn(accept_loop(
            listener,
            Arc::clone(&state),
            Arc::clone(&refusal),
        ));
        Self {
            url: format!("http://{address}"),
            state,
            refusal,
        }
    }

    pub(crate) fn arm(&self, scripts: &[CommitScript]) {
        let mut state = self.state.lock().expect("drill endpoint state");
        state.commit_scripts = scripts.iter().copied().collect();
        state.log.clear();
    }

    pub(crate) fn reset_next_reads(&self, count: usize) {
        let mut state = self.state.lock().expect("drill endpoint state");
        state.read_resets = count;
        state.log.clear();
    }

    pub(crate) fn refuse_the_commit_connection(&self, reads_before_commit: usize) {
        let mut state = self.state.lock().expect("drill endpoint state");
        state.refuse_after_reads = Some(reads_before_commit);
        state.log.clear();
    }

    pub(crate) async fn listening_again(&self) {
        tokio::time::timeout(LISTENER_DEADLINE, self.refusal.listening_again.notified())
            .await
            .expect("the drill endpoint listens again after the refusal window");
    }

    pub(crate) fn log(&self) -> Vec<Logged> {
        self.state.lock().expect("drill endpoint state").log.clone()
    }
}

pub(crate) async fn glue_catalog(url: &str) -> GlueCatalog {
    let props = HashMap::from([
        (GLUE_CATALOG_PROP_URI.to_string(), url.to_string()),
        (
            GLUE_CATALOG_PROP_WAREHOUSE.to_string(),
            "memory://warehouse".to_string(),
        ),
        (
            AWS_ACCESS_KEY_ID.to_string(),
            "drill-access-key".to_string(),
        ),
        (
            AWS_SECRET_ACCESS_KEY.to_string(),
            "drill-secret-key".to_string(),
        ),
        (AWS_REGION_NAME.to_string(), "us-east-1".to_string()),
    ]);
    GlueCatalogBuilder::default()
        .with_storage_factory(Arc::new(MemoryStorageFactory))
        .load("drill", props)
        .await
        .expect("build the Glue catalog against the drill endpoint")
}

pub(crate) async fn seeded_table(catalog: &GlueCatalog) -> Table {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("drill schema");
    let creation = TableCreation::builder()
        .name(TABLE.to_string())
        .location(LOCATION.to_string())
        .schema(schema)
        .build();
    catalog
        .create_table(&NamespaceIdent::new(NAMESPACE.to_string()), creation)
        .await
        .expect("create the drill table")
}

pub(crate) fn data_file() -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(format!("{LOCATION}/data/drill-0.parquet"))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(ROWS_PER_APPEND)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .expect("drill data file")
}

pub(crate) fn append(table: &Table, check_duplicate: bool) -> Transaction {
    let tx = Transaction::new(table);
    tx.fast_append()
        .with_check_duplicate(check_duplicate)
        .set_snapshot_properties(HashMap::from([(MARKER.to_string(), "1".to_string())]))
        .add_data_files(vec![data_file()])
        .apply(tx)
        .expect("stage the drill append")
}

pub(crate) async fn live_rows(table: &Table) -> u64 {
    let Some(current) = table.metadata().current_snapshot() else {
        return 0;
    };
    let manifest_list = current
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load the drill manifest list");
    let mut rows = 0;
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("load a drill manifest");
        rows += manifest
            .entries()
            .iter()
            .filter(|entry| entry.is_alive())
            .map(|entry| entry.data_file().record_count())
            .sum::<u64>();
    }
    rows
}

#[derive(Debug)]
pub(crate) struct Measured {
    pub(crate) outcome: Result<(), ErrorKind>,
    pub(crate) message: Option<String>,
    pub(crate) marked_snapshots: usize,
    pub(crate) marked_snapshot_ids: Vec<i64>,
    pub(crate) sent_snapshot_ids: Vec<Option<i64>>,
    pub(crate) rows: u64,
    pub(crate) log: Vec<Logged>,
}

impl Measured {
    pub(crate) fn reads_between_commits(&self) -> usize {
        let commits: Vec<usize> = self
            .log
            .iter()
            .enumerate()
            .filter(|(_, logged)| logged.metadata_sent.is_some())
            .map(|(index, _)| index)
            .collect();
        match (commits.first(), commits.last()) {
            (Some(first), Some(last)) => self.log[*first..*last]
                .iter()
                .filter(|logged| logged.operation == "GetTable")
                .count(),
            _ => 0,
        }
    }

    pub(crate) fn metadata_sent(&self) -> Vec<&str> {
        self.log
            .iter()
            .filter_map(|logged| logged.metadata_sent.as_deref())
            .collect()
    }

    pub(crate) fn reads(&self) -> usize {
        self.log
            .iter()
            .filter(|logged| logged.operation == "GetTable")
            .count()
    }

    pub(crate) fn commits(&self) -> Vec<(Option<&str>, Answer)> {
        self.log
            .iter()
            .filter(|logged| logged.operation == "UpdateTable")
            .map(|logged| (logged.version_sent.as_deref(), logged.answer))
            .collect()
    }
}

pub(crate) async fn drill(scripts: &[CommitScript], check_duplicate: bool) -> Measured {
    let endpoint = FakeGlue::start(Connections::KeepAlive).await;
    let catalog = glue_catalog(&endpoint.url).await;
    let table = seeded_table(&catalog).await;
    endpoint.arm(scripts);
    let label = format!("{scripts:?} check_duplicate={check_duplicate}");
    measure(&endpoint, &catalog, &table, check_duplicate, &label).await
}

pub(crate) async fn drill_refused_commit_connection() -> Measured {
    let endpoint = FakeGlue::start(Connections::OnePerRequest).await;
    let catalog = glue_catalog(&endpoint.url).await;
    let table = seeded_table(&catalog).await;
    endpoint.refuse_the_commit_connection(2);
    measure(
        &endpoint,
        &catalog,
        &table,
        true,
        "refused commit connection",
    )
    .await
}

pub(crate) async fn measure(
    endpoint: &FakeGlue,
    catalog: &GlueCatalog,
    table: &Table,
    check_duplicate: bool,
    label: &str,
) -> Measured {
    let result = append(table, check_duplicate).commit(catalog).await;
    if let Err(error) = &result {
        println!(
            "glue drill {label}: commit error {error:?}; source {:?}",
            std::error::Error::source(error)
        );
    }
    let message = result
        .as_ref()
        .err()
        .map(|error| error.message().to_string());
    let outcome = result.map(|_| ()).map_err(|error| error.kind());
    let log = endpoint.log();
    if endpoint
        .state
        .lock()
        .expect("drill endpoint state")
        .close_connections
    {
        endpoint.listening_again().await;
    }
    let reloaded = catalog
        .load_table(table.identifier())
        .await
        .expect("reload the drill table");
    let marked_snapshot_ids: Vec<i64> = reloaded
        .metadata()
        .snapshots()
        .filter(|snapshot| {
            snapshot
                .summary()
                .additional_properties
                .contains_key(MARKER)
        })
        .map(|snapshot| snapshot.snapshot_id())
        .collect();
    let mut sent_snapshot_ids = Vec::new();
    for location in log
        .iter()
        .filter_map(|logged| logged.metadata_sent.as_deref())
    {
        let sent = TableMetadata::read_from(reloaded.file_io(), location)
            .await
            .expect("read the metadata file a commit request named");
        sent_snapshot_ids.push(sent.current_snapshot_id());
    }
    let measured = Measured {
        outcome,
        message,
        marked_snapshots: marked_snapshot_ids.len(),
        marked_snapshot_ids,
        sent_snapshot_ids,
        rows: live_rows(&reloaded).await,
        log,
    };
    println!("glue drill {label}: {measured:#?}");
    measured
}
