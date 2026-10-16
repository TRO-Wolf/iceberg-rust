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
    Struct, Type,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation};
use iceberg_catalog_s3tables::{
    S3TABLES_CATALOG_PROP_ENDPOINT_URL, S3TABLES_CATALOG_PROP_TABLE_BUCKET_ARN, S3TablesCatalog,
    S3TablesCatalogBuilder,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;

const BUCKET_ARN: &str = "arn:aws:s3tables:us-east-1:123456789012:bucket/drill";
const TABLE_ARN: &str = "arn:aws:s3tables:us-east-1:123456789012:bucket/drill/table/t";
const NAMESPACE: &str = "drill";
const TABLE: &str = "t";
const WAREHOUSE: &str = "memory://warehouse/drill/t";
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
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Answer {
    Answered,
    AppliedThenReset,
    AppliedThenConflict,
    DroppedThenReset,
    AppliedThenClosed,
    DroppedThenClosed,
    RejectedStale,
    Reset,
}

#[derive(Clone, Debug, PartialEq)]
struct Logged {
    operation: &'static str,
    version_sent: Option<String>,
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
struct S3TablesState {
    metadata_location: Option<String>,
    version: u64,
    commit_scripts: VecDeque<CommitScript>,
    read_resets: usize,
    close_connections: bool,
    refuse_after_reads: Option<usize>,
    refusing: bool,
    log: Vec<Logged>,
}

impl S3TablesState {
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

    fn version_token(&self) -> String {
        format!("v{}", self.version)
    }

    fn handle(&mut self, operation: &'static str, body: &str) -> Reply {
        let (answer, version_sent, reply) = match operation {
            "CreateTable" => {
                self.version = 0;
                (
                    Answer::Answered,
                    None,
                    answered(format!(
                        r#"{{"tableARN":"{TABLE_ARN}","versionToken":"{}"}}"#,
                        self.version_token()
                    )),
                )
            }
            "GetTable" if self.read_resets > 0 => {
                self.read_resets -= 1;
                (Answer::Reset, None, Reply::Reset)
            }
            "GetTable" => {
                self.count_down_refusal();
                (Answer::Answered, None, self.get_table())
            }
            "UpdateTableMetadataLocation" => self.update_metadata_location(body),
            _ => (Answer::Answered, None, rejected(400, "BadRequestException")),
        };
        self.log.push(Logged {
            operation,
            version_sent,
            answer,
        });
        reply
    }

    fn get_table(&self) -> Reply {
        let metadata_location = self
            .metadata_location
            .as_ref()
            .map(|location| format!(r#""metadataLocation":"{location}","#))
            .unwrap_or_default();
        answered(format!(
            r#"{{"name":"{TABLE}","type":"customer","tableARN":"{TABLE_ARN}","namespace":["{NAMESPACE}"],"versionToken":"{}",{metadata_location}"warehouseLocation":"{WAREHOUSE}","format":"ICEBERG"}}"#,
            self.version_token()
        ))
    }

    fn update_metadata_location(&mut self, body: &str) -> (Answer, Option<String>, Reply) {
        let version_sent = json_string(body, "versionToken");
        if version_sent.as_deref() != Some(self.version_token().as_str()) {
            return (
                Answer::RejectedStale,
                version_sent,
                rejected(409, "ConflictException"),
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
            _ => {}
        }
        self.metadata_location = json_string(body, "metadataLocation");
        self.version += 1;
        match script {
            Some(CommitScript::ApplyThenReset) => {
                (Answer::AppliedThenReset, version_sent, Reply::Reset)
            }
            Some(CommitScript::ApplyThenClose) => {
                (Answer::AppliedThenClosed, version_sent, Reply::Close)
            }
            Some(CommitScript::ApplyThenConflict) => (
                Answer::AppliedThenConflict,
                version_sent,
                rejected(409, "ConflictException"),
            ),
            _ => (
                Answer::Answered,
                version_sent,
                answered(format!(
                    r#"{{"name":"{TABLE}","tableARN":"{TABLE_ARN}","namespace":["{NAMESPACE}"],"versionToken":"{}","metadataLocation":"{}"}}"#,
                    self.version_token(),
                    self.metadata_location.as_deref().unwrap_or_default()
                )),
            ),
        }
    }
}

fn json_string(body: &str, key: &str) -> Option<String> {
    let start = body.find(&format!(r#""{key}":""#))? + key.len() + 4;
    let length = body[start..].find('"')?;
    Some(body[start..start + length].to_string())
}

fn answered(body: String) -> Reply {
    Reply::Json {
        status: 200,
        error_type: None,
        body,
    }
}

fn rejected(status: u16, error_type: &'static str) -> Reply {
    Reply::Json {
        status,
        error_type: Some(error_type),
        body: r#"{"message":"rejected by the drill endpoint"}"#.to_string(),
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

    fn operation(&self) -> &'static str {
        let mut request_line = self.head.split_whitespace();
        let method = request_line.next().unwrap_or("");
        let path = request_line.next().unwrap_or("");
        match method {
            "GET" if path.starts_with("/get-table") => "GetTable",
            "PUT" if path.ends_with("/metadata-location") => "UpdateTableMetadataLocation",
            "PUT" if path.starts_with("/tables/") => "CreateTable",
            _ => "Unsupported",
        }
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
    };
    let mut head = format!(
        "HTTP/1.1 {status} {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n",
        if status == 200 { "OK" } else { "Error" },
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
    state: Arc<Mutex<S3TablesState>>,
    refusal: Arc<Refusal>,
) {
    while let Some(request) = read_request(&mut stream).await {
        let body = String::from_utf8_lossy(&request.body).into_owned();
        let reply = state
            .lock()
            .expect("drill endpoint state")
            .handle(request.operation(), &body);
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
    state: Arc<Mutex<S3TablesState>>,
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

struct FakeS3Tables {
    url: String,
    state: Arc<Mutex<S3TablesState>>,
    refusal: Arc<Refusal>,
}

impl FakeS3Tables {
    async fn start(connections: Connections) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the drill endpoint");
        let address = listener.local_addr().expect("drill endpoint address");
        let state = Arc::new(Mutex::new(S3TablesState {
            close_connections: matches!(connections, Connections::OnePerRequest),
            ..S3TablesState::default()
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

async fn s3tables_catalog(url: &str) -> S3TablesCatalog {
    let props = HashMap::from([
        (
            S3TABLES_CATALOG_PROP_TABLE_BUCKET_ARN.to_string(),
            BUCKET_ARN.to_string(),
        ),
        (
            S3TABLES_CATALOG_PROP_ENDPOINT_URL.to_string(),
            url.to_string(),
        ),
        (
            "aws_access_key_id".to_string(),
            "drill-access-key".to_string(),
        ),
        (
            "aws_secret_access_key".to_string(),
            "drill-secret-key".to_string(),
        ),
        ("region_name".to_string(), "us-east-1".to_string()),
    ]);
    S3TablesCatalogBuilder::default()
        .with_storage_factory(Arc::new(MemoryStorageFactory))
        .load("drill", props)
        .await
        .expect("build the S3 Tables catalog against the drill endpoint")
}

async fn seeded_table(catalog: &S3TablesCatalog) -> Table {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("drill schema");
    let creation = TableCreation::builder()
        .name(TABLE.to_string())
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
        .file_path(format!("{WAREHOUSE}/data/drill-0.parquet"))
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
            .filter(|logged| logged.operation == "UpdateTableMetadataLocation")
            .map(|logged| (logged.version_sent.as_deref(), logged.answer))
            .collect()
    }
}

async fn drill(scripts: &[CommitScript], check_duplicate: bool) -> Measured {
    let endpoint = FakeS3Tables::start(Connections::KeepAlive).await;
    let catalog = s3tables_catalog(&endpoint.url).await;
    let table = seeded_table(&catalog).await;
    endpoint.arm(scripts);
    let label = format!("{scripts:?} check_duplicate={check_duplicate}");
    measure(&endpoint, &catalog, &table, check_duplicate, &label).await
}

async fn drill_refused_commit_connection() -> Measured {
    let endpoint = FakeS3Tables::start(Connections::OnePerRequest).await;
    let catalog = s3tables_catalog(&endpoint.url).await;
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
    endpoint: &FakeS3Tables,
    catalog: &S3TablesCatalog,
    table: &Table,
    check_duplicate: bool,
    label: &str,
) -> Measured {
    let result = append(table, check_duplicate).commit(catalog).await;
    if let Err(error) = &result {
        println!("s3tables drill {label}: commit error {error:?}");
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
    let measured = Measured {
        outcome,
        marked_snapshots: reloaded
            .metadata()
            .snapshots()
            .filter(|snapshot| {
                snapshot
                    .summary()
                    .additional_properties
                    .contains_key(MARKER)
            })
            .count(),
        rows: live_rows(&reloaded).await,
        log,
    };
    println!("s3tables drill {label}: {measured:#?}");
    measured
}

#[tokio::test]
async fn s3tables_lost_reply_is_not_resent_and_reconciles_as_landed() {
    let measured = drill(&[CommitScript::ApplyThenReset], true).await;
    assert_eq!(measured.commits(), vec![(
        Some("v1"),
        Answer::AppliedThenReset
    )]);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshots, 1);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn s3tables_lost_unapplied_commit_is_unknown_and_keeps_the_pre_state() {
    let measured = drill(&[CommitScript::DropThenReset], true).await;
    assert_eq!(measured.commits(), vec![(
        Some("v1"),
        Answer::DroppedThenReset
    )]);
    assert_eq!(measured.outcome, Err(ErrorKind::CommitStateUnknown));
    assert_eq!(measured.marked_snapshots, 0);
    assert_eq!(measured.rows, 0);
}

#[tokio::test]
async fn s3tables_landed_commit_answered_conflict_lands_once() {
    let measured = drill(&[CommitScript::ApplyThenConflict], false).await;
    assert_eq!(measured.commits(), vec![(
        Some("v1"),
        Answer::AppliedThenConflict
    )]);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshots, 1);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn s3tables_reads_keep_their_sdk_retry() {
    let endpoint = FakeS3Tables::start(Connections::KeepAlive).await;
    let catalog = s3tables_catalog(&endpoint.url).await;
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
async fn s3tables_lost_reply_on_a_closed_connection_reconciles_as_landed() {
    let measured = drill(&[CommitScript::ApplyThenClose], true).await;
    assert_eq!(measured.commits(), vec![(
        Some("v1"),
        Answer::AppliedThenClosed
    )]);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshots, 1);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn s3tables_unapplied_commit_on_a_closed_connection_is_unknown_and_keeps_the_pre_state() {
    let measured = drill(&[CommitScript::DropThenClose], true).await;
    assert_eq!(measured.commits(), vec![(
        Some("v1"),
        Answer::DroppedThenClosed
    )]);
    assert_eq!(measured.outcome, Err(ErrorKind::CommitStateUnknown));
    assert_eq!(measured.marked_snapshots, 0);
    assert_eq!(measured.rows, 0);
}

#[tokio::test]
async fn s3tables_refused_commit_connection_is_never_sent_and_not_reconciled() {
    let measured = drill_refused_commit_connection().await;
    assert!(measured.commits().is_empty());
    assert_eq!(measured.reads(), 2);
    assert_eq!(measured.outcome, Err(ErrorKind::Unexpected));
    assert_eq!(measured.marked_snapshots, 0);
    assert_eq!(measured.rows, 0);
}
