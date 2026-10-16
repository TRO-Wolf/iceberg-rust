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

const NAMESPACE: &str = "drill";
const TABLE: &str = "t";
const LOCATION: &str = "memory://warehouse/drill.db/t";
const MARKER: &str = "drill.append";
const ROWS_PER_APPEND: u64 = 3;
const REFUSAL_WINDOW: Duration = Duration::from_secs(1);
const LISTENER_DEADLINE: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug, PartialEq)]
enum CommitScript {
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
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Answer {
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
    RejectedStale,
    Reset,
}

#[derive(Clone, Debug, PartialEq)]
struct Logged {
    operation: String,
    version_sent: Option<String>,
    metadata_sent: Option<String>,
    answer: Answer,
}

enum Reply {
    Json {
        status: u16,
        error_type: Option<&'static str>,
        body: String,
    },
    Reset,
    Close,
    Garbage,
}

#[derive(Clone, Copy)]
enum Connections {
    KeepAlive,
    OnePerRequest,
}

#[derive(Default)]
struct Refusal {
    close_listener: Notify,
    listener_closed: Notify,
    listening_again: Notify,
}

#[derive(Default)]
struct GlueState {
    metadata_location: Option<String>,
    version: u64,
    commit_scripts: VecDeque<CommitScript>,
    read_resets: usize,
    close_connections: bool,
    refuse_after_reads: Option<usize>,
    refusing: bool,
    deferred: Option<String>,
    log: Vec<Logged>,
}

impl GlueState {
    fn count_down_refusal(&mut self) {
        match self.refuse_after_reads {
            Some(remaining) if remaining <= 1 => {
                self.refuse_after_reads = None;
                self.refusing = true;
            }
            Some(remaining) => self.refuse_after_reads = Some(remaining - 1),
            None => {}
        }
    }

    fn handle(&mut self, operation: &str, body: &Value) -> Reply {
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

    fn get_table(&self) -> Reply {
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

    fn update_table(&mut self, body: &Value) -> (Answer, Option<String>, Reply) {
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
            Some(CommitScript::ApplyThenConflict) => (
                Answer::AppliedThenConflict,
                version_sent,
                rejected("ConcurrentModificationException", "stale VersionId"),
            ),
            _ => (Answer::Answered, version_sent, answered(json!({}))),
        }
    }
}

fn new_metadata_location(body: &Value) -> Option<String> {
    body["TableInput"]["Parameters"]["metadata_location"]
        .as_str()
        .map(str::to_string)
}

fn answered(body: Value) -> Reply {
    Reply::Json {
        status: 200,
        error_type: None,
        body: body.to_string(),
    }
}

fn failed(status: u16, error_type: &'static str) -> Reply {
    Reply::Json {
        status,
        error_type: Some(error_type),
        body: json!({"__type": error_type, "Message": "failed by the drill endpoint"}).to_string(),
    }
}

fn rejected(error_type: &'static str, message: &str) -> Reply {
    Reply::Json {
        status: 400,
        error_type: Some(error_type),
        body: json!({"__type": error_type, "Message": message}).to_string(),
    }
}

struct Request {
    head: String,
    body: Vec<u8>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.head.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.trim()
                .eq_ignore_ascii_case(name)
                .then_some(value.trim())
        })
    }
}

async fn read_more(stream: &mut TcpStream, buffer: &mut Vec<u8>) -> Option<()> {
    let mut chunk = [0u8; 8192];
    let read = stream
        .read(&mut chunk)
        .await
        .ok()
        .filter(|read| *read > 0)?;
    buffer.extend_from_slice(&chunk[..read]);
    Some(())
}

async fn read_request(stream: &mut TcpStream) -> Option<Request> {
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

async fn write_reply(stream: &mut TcpStream, reply: Reply, close_after: bool) -> bool {
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

async fn serve_connection(
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

async fn accept_loop(
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

struct FakeGlue {
    url: String,
    state: Arc<Mutex<GlueState>>,
    refusal: Arc<Refusal>,
}

impl FakeGlue {
    async fn start(connections: Connections) -> Self {
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

    fn arm(&self, scripts: &[CommitScript]) {
        let mut state = self.state.lock().expect("drill endpoint state");
        state.commit_scripts = scripts.iter().copied().collect();
        state.log.clear();
    }

    fn reset_next_reads(&self, count: usize) {
        let mut state = self.state.lock().expect("drill endpoint state");
        state.read_resets = count;
        state.log.clear();
    }

    fn refuse_the_commit_connection(&self, reads_before_commit: usize) {
        let mut state = self.state.lock().expect("drill endpoint state");
        state.refuse_after_reads = Some(reads_before_commit);
        state.log.clear();
    }

    async fn listening_again(&self) {
        tokio::time::timeout(LISTENER_DEADLINE, self.refusal.listening_again.notified())
            .await
            .expect("the drill endpoint listens again after the refusal window");
    }

    fn log(&self) -> Vec<Logged> {
        self.state.lock().expect("drill endpoint state").log.clone()
    }
}

async fn glue_catalog(url: &str) -> GlueCatalog {
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

async fn seeded_table(catalog: &GlueCatalog) -> Table {
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

fn data_file() -> DataFile {
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

fn append(table: &Table, check_duplicate: bool) -> Transaction {
    let tx = Transaction::new(table);
    tx.fast_append()
        .with_check_duplicate(check_duplicate)
        .set_snapshot_properties(HashMap::from([(MARKER.to_string(), "1".to_string())]))
        .add_data_files(vec![data_file()])
        .apply(tx)
        .expect("stage the drill append")
}

async fn live_rows(table: &Table) -> u64 {
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
struct Measured {
    outcome: Result<(), ErrorKind>,
    marked_snapshots: usize,
    marked_snapshot_ids: Vec<i64>,
    sent_snapshot_ids: Vec<Option<i64>>,
    rows: u64,
    log: Vec<Logged>,
}

impl Measured {
    fn reads(&self) -> usize {
        self.log
            .iter()
            .filter(|logged| logged.operation == "GetTable")
            .count()
    }

    fn commits(&self) -> Vec<(Option<&str>, Answer)> {
        self.log
            .iter()
            .filter(|logged| logged.operation == "UpdateTable")
            .map(|logged| (logged.version_sent.as_deref(), logged.answer))
            .collect()
    }
}

async fn drill(scripts: &[CommitScript], check_duplicate: bool) -> Measured {
    let endpoint = FakeGlue::start(Connections::KeepAlive).await;
    let catalog = glue_catalog(&endpoint.url).await;
    let table = seeded_table(&catalog).await;
    endpoint.arm(scripts);
    let label = format!("{scripts:?} check_duplicate={check_duplicate}");
    measure(&endpoint, &catalog, &table, check_duplicate, &label).await
}

async fn drill_refused_commit_connection() -> Measured {
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

async fn measure(
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
        marked_snapshots: marked_snapshot_ids.len(),
        marked_snapshot_ids,
        sent_snapshot_ids,
        rows: live_rows(&reloaded).await,
        log,
    };
    println!("glue drill {label}: {measured:#?}");
    measured
}

#[tokio::test]
async fn glue_lost_reply_is_not_resent_and_reconciles_as_landed() {
    let measured = drill(&[CommitScript::ApplyThenReset], true).await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::AppliedThenReset
    )]);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshots, 1);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn glue_lost_unapplied_commit_is_unknown_and_keeps_the_pre_state() {
    let measured = drill(&[CommitScript::DropThenReset], true).await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::DroppedThenReset
    )]);
    assert_eq!(measured.outcome, Err(ErrorKind::CommitStateUnknown));
    assert_eq!(measured.marked_snapshots, 0);
    assert_eq!(measured.rows, 0);
}

#[tokio::test]
async fn glue_landed_commit_answered_conflict_lands_once() {
    let measured = drill(&[CommitScript::ApplyThenConflict], false).await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::AppliedThenConflict
    )]);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshots, 1);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn glue_reads_keep_their_sdk_retry() {
    let endpoint = FakeGlue::start(Connections::KeepAlive).await;
    let catalog = glue_catalog(&endpoint.url).await;
    let table = seeded_table(&catalog).await;
    endpoint.reset_next_reads(1);
    let reloaded = catalog
        .load_table(table.identifier())
        .await
        .expect("a reset read is retried by the SDK");
    assert_eq!(reloaded.metadata_location(), table.metadata_location());
    let reads: Vec<Answer> = endpoint
        .log()
        .iter()
        .filter(|logged| logged.operation == "GetTable")
        .map(|logged| logged.answer)
        .collect();
    assert_eq!(reads, vec![Answer::Reset, Answer::Answered]);
}

#[tokio::test]
async fn glue_lost_reply_on_a_closed_connection_reconciles_as_landed() {
    let measured = drill(&[CommitScript::ApplyThenClose], true).await;
    let &[Some(sent)] = measured.sent_snapshot_ids.as_slice() else {
        panic!("one commit request naming one snapshot");
    };
    assert_eq!(measured.marked_snapshot_ids, vec![sent]);
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::AppliedThenClosed
    )]);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshots, 1);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn glue_unapplied_commit_on_a_closed_connection_is_unknown_and_keeps_the_pre_state() {
    let measured = drill(&[CommitScript::DropThenClose], true).await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::DroppedThenClosed
    )]);
    assert_eq!(measured.outcome, Err(ErrorKind::CommitStateUnknown));
    assert_eq!(measured.marked_snapshots, 0);
    assert_eq!(measured.rows, 0);
}

#[tokio::test]
async fn glue_refused_commit_connection_is_never_sent_and_not_reconciled() {
    let measured = drill_refused_commit_connection().await;
    assert!(measured.commits().is_empty());
    assert_eq!(measured.reads(), 2);
    assert_eq!(measured.outcome, Err(ErrorKind::Unexpected));
    assert_eq!(measured.marked_snapshots, 0);
    assert_eq!(measured.rows, 0);
}

#[tokio::test]
async fn glue_server_error_with_an_unmodeled_code_after_the_commit_reconciles_as_landed() {
    let measured = drill(
        &[CommitScript::ApplyThenError(500, "InternalFailure")],
        true,
    )
    .await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::AppliedThenError(500)
    )]);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshots, 1);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn glue_server_error_with_an_unmodeled_code_before_the_commit_is_unknown_and_keeps_the_pre_state()
 {
    let measured = drill(&[CommitScript::DropThenError(500, "InternalFailure")], true).await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::DroppedThenError(500)
    )]);
    assert_eq!(measured.outcome, Err(ErrorKind::CommitStateUnknown));
    assert_eq!(measured.marked_snapshots, 0);
    assert_eq!(measured.rows, 0);
}

#[tokio::test]
async fn glue_modeled_server_error_after_the_commit_reconciles_as_landed() {
    let measured = drill(
        &[CommitScript::ApplyThenError(
            500,
            "InternalServiceException",
        )],
        true,
    )
    .await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::AppliedThenError(500)
    )]);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshots, 1);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn glue_unparsable_reply_after_the_commit_reconciles_as_landed() {
    let measured = drill(&[CommitScript::ApplyThenGarbage], true).await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::AppliedThenGarbage
    )]);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshots, 1);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn glue_unparsable_reply_before_the_commit_is_unknown_and_keeps_the_pre_state() {
    let measured = drill(&[CommitScript::DropThenGarbage], true).await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::DroppedThenGarbage
    )]);
    assert_eq!(measured.outcome, Err(ErrorKind::CommitStateUnknown));
    assert_eq!(measured.marked_snapshots, 0);
    assert_eq!(measured.rows, 0);
}

#[tokio::test]
async fn glue_client_error_with_an_unmodeled_code_is_definitive_and_not_reconciled() {
    let measured = drill(
        &[CommitScript::DropThenError(400, "ValidationException")],
        true,
    )
    .await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::DroppedThenError(400)
    )]);
    assert_eq!(measured.reads(), 2);
    assert_eq!(measured.outcome, Err(ErrorKind::Unexpected));
    assert_eq!(measured.marked_snapshots, 0);
    assert_eq!(measured.rows, 0);
}

#[tokio::test]
async fn glue_an_earlier_attempt_that_lands_late_is_found_before_the_rebase() {
    let measured = drill(
        &[
            CommitScript::DeferThenConflict,
            CommitScript::LandDeferredThenConflict,
        ],
        false,
    )
    .await;
    assert_eq!(measured.commits(), vec![
        (Some("1"), Answer::DeferredThenConflict),
        (Some("1"), Answer::LandedDeferredThenConflict),
    ]);
    let &[Some(first), Some(second)] = measured.sent_snapshot_ids.as_slice() else {
        panic!("two commit requests, each naming one snapshot");
    };
    assert_ne!(first, second);
    assert_eq!(measured.marked_snapshot_ids, vec![first]);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}
