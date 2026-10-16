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
use iceberg_catalog_s3tables::{
    S3TABLES_CATALOG_PROP_ENDPOINT_URL, S3TABLES_CATALOG_PROP_TABLE_BUCKET_ARN, S3TablesCatalog,
    S3TablesCatalogBuilder,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const BUCKET_ARN: &str = "arn:aws:s3tables:us-east-1:123456789012:bucket/drill";
const TABLE_ARN: &str = "arn:aws:s3tables:us-east-1:123456789012:bucket/drill/table/t";
const NAMESPACE: &str = "drill";
const TABLE: &str = "t";
const WAREHOUSE: &str = "memory://warehouse/drill/t";
const MARKER: &str = "drill.append";
const ROWS_PER_APPEND: u64 = 3;

#[derive(Clone, Copy, Debug, PartialEq)]
enum CommitScript {
    ApplyThenReset,
    ApplyThenConflict,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Answer {
    Answered,
    AppliedThenReset,
    AppliedThenConflict,
    RejectedStale,
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
}

#[derive(Default)]
struct S3TablesState {
    metadata_location: Option<String>,
    version: u64,
    commit_scripts: VecDeque<CommitScript>,
    log: Vec<Logged>,
}

impl S3TablesState {
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
            "GetTable" => (Answer::Answered, None, self.get_table()),
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
        self.metadata_location = json_string(body, "metadataLocation");
        self.version += 1;
        match self.commit_scripts.pop_front() {
            Some(CommitScript::ApplyThenReset) => {
                (Answer::AppliedThenReset, version_sent, Reply::Reset)
            }
            Some(CommitScript::ApplyThenConflict) => (
                Answer::AppliedThenConflict,
                version_sent,
                rejected(409, "ConflictException"),
            ),
            None => (
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
            key.trim().eq_ignore_ascii_case(name).then(|| value.trim())
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
        "HTTP/1.1 {status} {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n",
        if status == 200 { "OK" } else { "Error" },
        body.len()
    );
    if let Some(error_type) = error_type {
        head.push_str(&format!("x-amzn-errortype: {error_type}\r\n"));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes()).await.is_ok()
        && stream.write_all(body.as_bytes()).await.is_ok()
}

async fn serve_connection(mut stream: TcpStream, state: Arc<Mutex<S3TablesState>>) {
    while let Some(request) = read_request(&mut stream).await {
        let body = String::from_utf8_lossy(&request.body).into_owned();
        let reply = state
            .lock()
            .expect("drill endpoint state")
            .handle(request.operation(), &body);
        if !write_reply(&mut stream, reply).await {
            return;
        }
    }
}

async fn accept_loop(listener: TcpListener, state: Arc<Mutex<S3TablesState>>) {
    while let Ok((stream, _)) = listener.accept().await {
        tokio::spawn(serve_connection(stream, Arc::clone(&state)));
    }
}

struct FakeS3Tables {
    url: String,
    state: Arc<Mutex<S3TablesState>>,
}

impl FakeS3Tables {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the drill endpoint");
        let address = listener.local_addr().expect("drill endpoint address");
        let state = Arc::new(Mutex::new(S3TablesState::default()));
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
    fn commits(&self) -> Vec<(Option<&str>, Answer)> {
        self.log
            .iter()
            .filter(|logged| logged.operation == "UpdateTableMetadataLocation")
            .map(|logged| (logged.version_sent.as_deref(), logged.answer))
            .collect()
    }
}

async fn drill(scripts: &[CommitScript], check_duplicate: bool) -> Measured {
    let endpoint = FakeS3Tables::start().await;
    let catalog = s3tables_catalog(&endpoint.url).await;
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
    println!("s3tables drill {scripts:?} check_duplicate={check_duplicate}: {measured:#?}");
    measured
}

#[tokio::test]
async fn s3tables_sdk_resend_rejected_stale_lands_once_without_the_duplicate_check() {
    let measured = drill(&[CommitScript::ApplyThenReset], false).await;
    assert_eq!(measured.commits(), vec![
        (Some("v1"), Answer::AppliedThenReset),
        (Some("v1"), Answer::RejectedStale),
    ]);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshots, 1);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn s3tables_sdk_resend_rejected_stale_lands_once_under_the_duplicate_check() {
    let measured = drill(&[CommitScript::ApplyThenReset], true).await;
    assert_eq!(measured.commits(), vec![
        (Some("v1"), Answer::AppliedThenReset),
        (Some("v1"), Answer::RejectedStale),
    ]);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshots, 1);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
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
