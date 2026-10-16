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

use iceberg::io::MemoryStorageFactory;
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, NestedField, PrimitiveType, Schema,
    Struct, Type,
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

const NAMESPACE: &str = "drill";
const TABLE: &str = "t";
const LOCATION: &str = "memory://warehouse/drill.db/t";
const MARKER: &str = "drill.append";
const ROWS_PER_APPEND: u64 = 3;

#[derive(Clone, Copy, Debug, PartialEq)]
enum CommitScript {
    ApplyThenReset,
    ApplyThenConflict,
    DropThenReset,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Answer {
    Answered,
    AppliedThenReset,
    AppliedThenConflict,
    DroppedThenReset,
    RejectedStale,
    Reset,
}

#[derive(Clone, Debug, PartialEq)]
struct Logged {
    operation: String,
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
}

#[derive(Default)]
struct GlueState {
    metadata_location: Option<String>,
    version: u64,
    commit_scripts: VecDeque<CommitScript>,
    read_resets: usize,
    log: Vec<Logged>,
}

impl GlueState {
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
            "GetTable" => (Answer::Answered, None, self.get_table()),
            "UpdateTable" => self.update_table(body),
            _ => (
                Answer::Answered,
                None,
                rejected("InvalidInputException", "unsupported by the drill endpoint"),
            ),
        };
        self.log.push(Logged {
            operation: operation.to_string(),
            version_sent,
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
        if script == Some(CommitScript::DropThenReset) {
            return (Answer::DroppedThenReset, version_sent, Reply::Reset);
        }
        self.metadata_location = new_metadata_location(body);
        self.version += 1;
        match script {
            Some(CommitScript::ApplyThenReset) => {
                (Answer::AppliedThenReset, version_sent, Reply::Reset)
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

async fn write_reply(stream: &mut TcpStream, reply: Reply) -> bool {
    let Reply::Json {
        status,
        error_type,
        body,
    } = reply
    else {
        let _ = stream.set_zero_linger();
        return false;
    };
    let mut head = format!(
        "HTTP/1.1 {status} {}\r\ncontent-type: application/x-amz-json-1.1\r\ncontent-length: {}\r\n",
        if status == 200 { "OK" } else { "Bad Request" },
        body.len()
    );
    if let Some(error_type) = error_type {
        head.push_str(&format!("x-amzn-errortype: {error_type}\r\n"));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes()).await.is_ok()
        && stream.write_all(body.as_bytes()).await.is_ok()
}

async fn serve_connection(mut stream: TcpStream, state: Arc<Mutex<GlueState>>) {
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
        if !write_reply(&mut stream, reply).await {
            return;
        }
    }
}

async fn accept_loop(listener: TcpListener, state: Arc<Mutex<GlueState>>) {
    while let Ok((stream, _)) = listener.accept().await {
        tokio::spawn(serve_connection(stream, Arc::clone(&state)));
    }
}

struct FakeGlue {
    url: String,
    state: Arc<Mutex<GlueState>>,
}

impl FakeGlue {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the drill endpoint");
        let address = listener.local_addr().expect("drill endpoint address");
        let state = Arc::new(Mutex::new(GlueState::default()));
        tokio::spawn(accept_loop(listener, Arc::clone(&state)));
        Self {
            url: format!("http://{address}"),
            state,
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
    rows: u64,
    log: Vec<Logged>,
}

impl Measured {
    fn commits(&self) -> Vec<(Option<&str>, Answer)> {
        self.log
            .iter()
            .filter(|logged| logged.operation == "UpdateTable")
            .map(|logged| (logged.version_sent.as_deref(), logged.answer))
            .collect()
    }
}

async fn drill(scripts: &[CommitScript], check_duplicate: bool) -> Measured {
    let endpoint = FakeGlue::start().await;
    let catalog = glue_catalog(&endpoint.url).await;
    let table = seeded_table(&catalog).await;
    endpoint.arm(scripts);
    let outcome = append(&table, check_duplicate)
        .commit(&catalog)
        .await
        .map(|_| ())
        .map_err(|error| error.kind());
    let log = endpoint.log();
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
    println!("glue drill {scripts:?} check_duplicate={check_duplicate}: {measured:#?}");
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
    let endpoint = FakeGlue::start().await;
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
