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

use std::collections::HashMap;
use std::fmt::{Debug, Formatter};

use http::StatusCode;
use iceberg::{Error, ErrorKind, Result};
use reqwest::header::HeaderMap;
use reqwest::{Client, IntoUrl, Method, Request, RequestBuilder, Response};
use serde::de::DeserializeOwned;
use tokio::sync::Mutex;

use crate::RestCatalogConfig;
use crate::types::{ErrorResponse, TokenResponse};

pub(crate) struct HttpClient {
    client: Client,

    /// The token to be used for authentication.
    ///
    /// It's possible to fetch the token from the server while needed.
    token: Mutex<Option<String>>,
    /// The token endpoint to be used for authentication.
    token_endpoint: String,
    /// The credential to be used for authentication.
    credential: Option<(Option<String>, String)>,
    /// Extra headers to be added to each request.
    extra_headers: HeaderMap,
    /// Extra oauth parameters to be added to each authentication request.
    extra_oauth_params: HashMap<String, String>,
    /// Whether to disable header redaction in error logs (defaults to false for security).
    disable_header_redaction: bool,
}

impl Debug for HttpClient {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpClient")
            .field("client", &self.client)
            .field("extra_headers", &self.extra_headers)
            .finish_non_exhaustive()
    }
}

impl HttpClient {
    /// Create a new http client.
    pub fn new(cfg: &RestCatalogConfig) -> Result<Self> {
        let extra_headers = cfg.extra_headers()?;
        Ok(HttpClient {
            client: cfg.client().unwrap_or_default(),
            token: Mutex::new(cfg.token()),
            token_endpoint: cfg.get_token_endpoint(),
            credential: cfg.credential(),
            extra_headers,
            extra_oauth_params: cfg.extra_oauth_params(),
            disable_header_redaction: cfg.disable_header_redaction(),
        })
    }

    /// Update the http client with new configuration.
    ///
    /// If cfg carries new value, we will use cfg instead.
    /// Otherwise, we will keep the old value.
    pub fn update_with(self, cfg: &RestCatalogConfig) -> Result<Self> {
        let extra_headers = (!cfg.extra_headers()?.is_empty())
            .then(|| cfg.extra_headers())
            .transpose()?
            .unwrap_or(self.extra_headers);
        Ok(HttpClient {
            client: cfg.client().unwrap_or(self.client),
            token: Mutex::new(cfg.token().or_else(|| self.token.into_inner())),
            token_endpoint: if !cfg.get_token_endpoint().is_empty() {
                cfg.get_token_endpoint()
            } else {
                self.token_endpoint
            },
            credential: cfg.credential().or(self.credential),
            extra_headers,
            extra_oauth_params: if !cfg.extra_oauth_params().is_empty() {
                cfg.extra_oauth_params()
            } else {
                self.extra_oauth_params
            },
            disable_header_redaction: cfg.disable_header_redaction(),
        })
    }

    /// This API is testing only to assert the token.
    #[cfg(test)]
    pub(crate) async fn token(&self) -> Option<String> {
        let mut req = self
            .request(Method::GET, &self.token_endpoint)
            .build()
            .unwrap();
        self.authenticate(&mut req).await.ok();
        self.token.lock().await.clone()
    }

    pub(crate) async fn exchange_credential_for_token(&self) -> Result<String> {
        // Credential must exist here.
        let (client_id, client_secret) = self.credential.as_ref().ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "Credential must be provided for authentication",
            )
        })?;

        let mut params = HashMap::with_capacity(4);
        params.insert("grant_type", "client_credentials");
        if let Some(client_id) = client_id {
            params.insert("client_id", client_id);
        }
        params.insert("client_secret", client_secret);
        params.extend(
            self.extra_oauth_params
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str())),
        );

        let mut auth_req = self
            .request(Method::POST, &self.token_endpoint)
            .form(&params)
            .build()?;
        // extra headers add content-type application/json header it's necessary to override it with proper type
        // note that form call doesn't add content-type header if already present
        auth_req.headers_mut().insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/x-www-form-urlencoded"),
        );
        let auth_url = auth_req.url().clone();
        let auth_resp = self.client.execute(auth_req).await?;

        let auth_res: TokenResponse = if auth_resp.status() == StatusCode::OK {
            let text = auth_resp
                .bytes()
                .await
                .map_err(|err| err.with_url(auth_url.clone()))?;
            Ok(serde_json::from_slice(&text).map_err(|e| {
                // SECURITY: the token-endpoint 200-OK body contains the OAuth
                // `access_token`. Never attach the raw body to the error context — it
                // would be rendered by `Error`'s `Display` and leak the token into any
                // `tracing::error!(?e)` / `{e}` log. Attach only the safe byte length.
                Error::new(
                    ErrorKind::Unexpected,
                    "Failed to parse response from rest catalog server!",
                )
                .with_context("operation", "auth")
                .with_context("url", auth_url.to_string())
                .with_context("response_body_len", text.len().to_string())
                .with_source(e)
            })?)
        } else {
            let code = auth_resp.status();
            let text = auth_resp
                .bytes()
                .await
                .map_err(|err| err.with_url(auth_url.clone()))?;
            let e: ErrorResponse = serde_json::from_slice(&text).map_err(|e| {
                // SECURITY: this is still the token endpoint — a non-2xx body may echo
                // submitted credentials or a partial grant. Keep the token path airtight
                // by never attaching the raw body; surface only its byte length.
                Error::new(ErrorKind::Unexpected, "Received unexpected response")
                    .with_context("code", code.to_string())
                    .with_context("operation", "auth")
                    .with_context("url", auth_url.to_string())
                    .with_context("response_body_len", text.len().to_string())
                    .with_source(e)
            })?;
            Err(Error::from(e))
        }?;
        Ok(auth_res.access_token)
    }

    /// Invalidate the current token without generating a new one. On the next request, the client
    /// will attempt to generate a new token.
    pub(crate) async fn invalidate_token(&self) -> Result<()> {
        *self.token.lock().await = None;
        Ok(())
    }

    /// Invalidate the current token and set a new one. Generates a new token before invalidating
    /// the current token, meaning the old token will be used until this function acquires the lock
    /// and overwrites the token.
    ///
    /// If credential is invalid, or the request fails, this method will return an error and leave
    /// the current token unchanged.
    pub(crate) async fn regenerate_token(&self) -> Result<()> {
        let new_token = self.exchange_credential_for_token().await?;
        *self.token.lock().await = Some(new_token.clone());
        Ok(())
    }

    /// Authenticates the request by adding a bearer token to the authorization header.
    ///
    /// This method supports three authentication modes:
    ///
    /// 1. **No authentication** - Skip authentication when both `credential` and `token` are missing.
    /// 2. **Token authentication** - Use the provided `token` directly for authentication.
    /// 3. **OAuth authentication** - Exchange `credential` for a token, cache it, then use it for authentication.
    ///
    /// When both `credential` and `token` are present, `token` takes precedence.
    ///
    /// # TODO: Support automatic token refreshing.
    async fn authenticate(&self, req: &mut Request) -> Result<()> {
        // Clone the token from lock without holding the lock for entire function.
        let token = self.token.lock().await.clone();

        if self.credential.is_none() && token.is_none() {
            return Ok(());
        }

        // Either use the provided token or exchange credential for token, cache and use that
        let token = match token {
            Some(token) => token,
            None => {
                let token = self.exchange_credential_for_token().await?;
                // Update token so that we use it for next request instead of
                // exchanging credential for token from the server again
                *self.token.lock().await = Some(token.clone());
                token
            }
        };

        // Insert token in request.
        req.headers_mut().insert(
            http::header::AUTHORIZATION,
            format!("Bearer {token}").parse().map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Invalid token received from catalog server!",
                )
                .with_source(e)
            })?,
        );

        Ok(())
    }

    #[inline]
    pub fn request<U: IntoUrl>(&self, method: Method, url: U) -> RequestBuilder {
        self.client
            .request(method, url)
            .headers(self.extra_headers.clone())
    }

    /// Executes the given `Request` and returns a `Response`.
    pub async fn execute(&self, mut request: Request) -> Result<Response> {
        request.headers_mut().extend(self.extra_headers.clone());
        Ok(self.client.execute(request).await?)
    }

    // Queries the Iceberg REST catalog after authentication with the given `Request` and
    // returns a `Response`.
    pub async fn query_catalog(&self, mut request: Request) -> Result<Response> {
        self.authenticate(&mut request).await?;
        self.execute(request).await
    }

    /// [`Self::query_catalog`] for COMMIT requests (table / view update): transport failures
    /// are classified sent-vs-unsent.
    ///
    /// A failure that may have occurred AFTER the request reached the service (timeout awaiting
    /// the response, connection reset mid-response) maps to [`ErrorKind::CommitStateUnknown`] —
    /// the commit may have durably landed, so retrying could apply it twice. A request that
    /// provably never left the client (pre-send authentication failure, connect failure) keeps
    /// today's terminal mapping. See [`commit_transport_failure_may_have_reached_service`].
    pub async fn query_catalog_for_commit(&self, mut request: Request) -> Result<Response> {
        // Pre-send: an authentication failure means the commit request was never sent — the
        // existing (non-unknown) mapping stands.
        self.authenticate(&mut request).await?;
        request.headers_mut().extend(self.extra_headers.clone());
        self.client.execute(request).await.map_err(|error| {
            if commit_transport_failure_may_have_reached_service(&error) {
                Error::new(
                    ErrorKind::CommitStateUnknown,
                    "Transport failure after the commit request may have reached the service; \
                     the commit state is unknown. Check whether the commit landed before \
                     retrying: retrying an already-successful commit duplicates its changes.",
                )
                .with_source(error)
            } else {
                Error::from(error)
            }
        })
    }

    /// Returns whether header redaction is disabled for this client.
    pub(crate) fn disable_header_redaction(&self) -> bool {
        self.disable_header_redaction
    }
}

/// True when a transport-level failure on a COMMIT request may have occurred AFTER the request
/// reached the service — a timeout awaiting the response, a connection reset mid-response, a
/// lost/truncated body. False only when the request provably never left the client: the request
/// could not be BUILT, or the CONNECTION could not be established (connect refused / connect
/// timeout — reqwest reports connect-phase timeouts with `is_connect()`).
///
/// Java analogue: once a response arrives, `ErrorHandlers$CommitErrorHandler` (iceberg-core
/// 1.10.0, `ErrorHandlers.java` L88-104) classifies by HTTP status — 409 →
/// `CommitFailedException` (retryable), 500/502/503/504 → `CommitStateUnknownException`. For
/// CLIENT-side transport failures Java's `HTTPClient.execute` (L358-359) collapses everything
/// into `RESTException` (with an acknowledged `TODO` in `RESTTableOperations.commit` to feed
/// client errors to the error handler), which `SnapshotProducer.commit()` then neither retries
/// nor cleans up. This fork names the post-send ambiguity explicitly as
/// [`ErrorKind::CommitStateUnknown`] — the same observable no-retry / no-cleanup / surfaced
/// semantics, with the unknown outcome distinguishable by the caller.
pub(crate) fn commit_transport_failure_may_have_reached_service(error: &reqwest::Error) -> bool {
    !(error.is_builder() || error.is_connect())
}

/// Deserializes a catalog response into the given [`DeserializedOwned`] type.
///
/// Returns an error if unable to parse the response bytes.
pub(crate) async fn deserialize_catalog_response<R: DeserializeOwned>(
    response: Response,
) -> Result<R> {
    let bytes = response.bytes().await?;

    serde_json::from_slice::<R>(&bytes).map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            "Failed to parse response from rest catalog server",
        )
        .with_context("json", String::from_utf8_lossy(&bytes))
        .with_source(e)
    })
}

/// Headers that contain sensitive information and should be excluded from logs.
const SENSITIVE_HEADERS: &[&str] = &[
    "authorization",
    "proxy-authorization",
    "set-cookie",
    "cookie",
    "x-api-key",
    "x-auth-token",
];

/// Returns true if the header name is considered sensitive.
fn is_sensitive_header(name: &str) -> bool {
    let name_lower = name.to_lowercase();
    SENSITIVE_HEADERS.iter().any(|h| name_lower == *h)
}

/// Redacts sensitive headers and returns a debug-formatted string.
///
/// If `disable_redaction` is true, returns all headers without redaction.
/// Otherwise, replaces sensitive header values with "[REDACTED]".
fn format_headers_redacted(headers: &HeaderMap, disable_redaction: bool) -> String {
    if disable_redaction {
        // Return all headers as-is without redaction
        let all: HashMap<&str, &str> = headers
            .iter()
            .filter_map(|(name, value)| value.to_str().ok().map(|v| (name.as_str(), v)))
            .collect();
        return format!("{all:?}");
    }

    // Redact sensitive headers by replacing their values with "[REDACTED]"
    let redacted: HashMap<&str, &str> = headers
        .iter()
        .filter_map(|(name, value)| {
            if is_sensitive_header(name.as_str()) {
                Some((name.as_str(), "[REDACTED]"))
            } else {
                value.to_str().ok().map(|v| (name.as_str(), v))
            }
        })
        .collect();
    format!("{redacted:?}")
}

/// Deserializes a unexpected catalog response into an error.
pub(crate) async fn deserialize_unexpected_catalog_error(
    response: Response,
    disable_header_redaction: bool,
) -> Error {
    let err = Error::new(
        ErrorKind::Unexpected,
        "Received response with unexpected status code",
    )
    .with_context("status", response.status().to_string())
    .with_context(
        "headers",
        format_headers_redacted(response.headers(), disable_header_redaction),
    );

    let bytes = match response.bytes().await {
        Ok(bytes) => bytes,
        Err(err) => return err.into(),
    };

    if bytes.is_empty() {
        return err;
    }
    err.with_context("json", String::from_utf8_lossy(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Risk (GAP_MATRIX row R157): a NEVER-SENT transport failure (connection refused — the TCP
    /// connection was never established, so the commit request cannot have reached the service)
    /// is classified as unknown-outcome, sending the caller into needless commit reconciliation
    /// on every transient connectivity blip. Pins the sent-vs-unsent split's UNSENT side with a
    /// real reqwest connect error.
    #[tokio::test]
    async fn test_connect_refused_commit_failure_is_not_post_send() {
        // Bind to an ephemeral port, then drop the listener: connecting to it is refused.
        let port = {
            let listener =
                std::net::TcpListener::bind("127.0.0.1:0").expect("bind an ephemeral port");
            listener
                .local_addr()
                .expect("read the bound address")
                .port()
        };
        let error = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/v1/tables"))
            .send()
            .await
            .expect_err("connecting to a dropped listener must fail");
        assert!(
            error.is_connect(),
            "precondition: the failure must be a connect-phase error, got: {error:?}"
        );
        assert!(
            !commit_transport_failure_may_have_reached_service(&error),
            "a connect-refused request never reached the service — it must NOT classify as \
             post-send ambiguous (unknown outcome)"
        );
    }

    /// Risk (GAP_MATRIX row R157): a connection reset AFTER the commit request was written —
    /// the server read the request and died before responding, so the commit MAY have durably
    /// landed — is classified as never-sent/terminal, letting a caller (or an outer loop)
    /// re-run the commit and duplicate its changes. Pins the sent-vs-unsent split's SENT side
    /// with a real mid-exchange connection drop.
    #[tokio::test]
    async fn test_connection_reset_after_send_is_post_send_ambiguous() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an ephemeral port");
        let addr = listener.local_addr().expect("read the bound address");
        // Accept one connection, read the request bytes, then drop the socket WITHOUT
        // responding — the client observes the connection closing after the request was sent.
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept the commit request");
            let mut buffer = [0u8; 1024];
            use tokio::io::AsyncReadExt as _;
            let _ = socket.read(&mut buffer).await;
            drop(socket);
        });
        let error = reqwest::Client::new()
            .post(format!("http://{addr}/v1/tables"))
            .body("{}")
            .send()
            .await
            .expect_err("a connection dropped before any response must fail");
        server.await.expect("the stub server task must finish");
        assert!(
            !error.is_connect(),
            "precondition: the connection WAS established, got: {error:?}"
        );
        assert!(
            commit_transport_failure_may_have_reached_service(&error),
            "a reset after the request was written may have reached the service — it MUST \
             classify as post-send ambiguous (unknown outcome)"
        );
    }

    /// Risk (GAP_MATRIX row R157): a timeout AWAITING THE RESPONSE (request sent, server
    /// processing — the commit may complete server-side after the client gives up) is
    /// classified as never-sent/terminal. Pins that a post-send timeout classifies as
    /// post-send ambiguous.
    #[tokio::test]
    async fn test_timeout_awaiting_response_is_post_send_ambiguous() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an ephemeral port");
        let addr = listener.local_addr().expect("read the bound address");
        // Accept, read the request, then stall past the client timeout without responding.
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept the commit request");
            let mut buffer = [0u8; 1024];
            use tokio::io::AsyncReadExt as _;
            let _ = socket.read(&mut buffer).await;
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            drop(socket);
        });
        let error = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(50))
            .build()
            .expect("build a short-timeout client")
            .post(format!("http://{addr}/v1/tables"))
            .body("{}")
            .send()
            .await
            .expect_err("the stalled response must time out");
        server.await.expect("the stub server task must finish");
        assert!(
            error.is_timeout(),
            "precondition: the failure must be a timeout, got: {error:?}"
        );
        assert!(
            commit_transport_failure_may_have_reached_service(&error),
            "a timeout awaiting the response may have reached the service — it MUST classify \
             as post-send ambiguous (unknown outcome)"
        );
    }

    #[test]
    fn test_format_headers_redacted_empty() {
        let headers = HeaderMap::new();
        let result = format_headers_redacted(&headers, false);
        assert_eq!(result, "{}");
    }

    #[test]
    fn test_format_headers_redacted_non_sensitive() {
        let mut headers = HeaderMap::new();
        headers.insert("content-type", "application/json".parse().unwrap());
        headers.insert("x-request-id", "abc123".parse().unwrap());

        let result = format_headers_redacted(&headers, false);

        assert!(result.contains("content-type"));
        assert!(result.contains("application/json"));
        assert!(result.contains("x-request-id"));
        assert!(result.contains("abc123"));
    }

    #[test]
    fn test_format_headers_redacted_filters_sensitive() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer secret-token".parse().unwrap());
        headers.insert("content-type", "application/json".parse().unwrap());

        let result = format_headers_redacted(&headers, false);

        // Sensitive header should be present but with redacted value
        assert!(result.contains("authorization"));
        assert!(result.contains("[REDACTED]"));
        // Sensitive value should NOT be present
        assert!(!result.contains("secret-token"));
        // Non-sensitive header should be present with actual value
        assert!(result.contains("content-type"));
        assert!(result.contains("application/json"));
    }

    #[test]
    fn test_format_headers_redacted_filters_set_cookie() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "set-cookie",
            "CF_Authorization=sensitive-session-token; Path=/; Secure;"
                .parse()
                .unwrap(),
        );
        headers.insert("server", "cloudflare".parse().unwrap());

        let result = format_headers_redacted(&headers, false);

        // Sensitive header should be present but with redacted value
        assert!(result.contains("set-cookie"));
        assert!(result.contains("[REDACTED]"));
        // Sensitive value should NOT be present
        assert!(!result.contains("sensitive-session-token"));
        // Non-sensitive header should be present with actual value
        assert!(result.contains("server"));
        assert!(result.contains("cloudflare"));
    }

    #[test]
    fn test_format_headers_redacted_filters_all_sensitive() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer token".parse().unwrap());
        headers.insert("proxy-authorization", "Basic creds".parse().unwrap());
        headers.insert("set-cookie", "session=abc".parse().unwrap());
        headers.insert("cookie", "session=abc".parse().unwrap());
        headers.insert("x-api-key", "api-key-123".parse().unwrap());
        headers.insert("x-auth-token", "auth-token-456".parse().unwrap());
        headers.insert("x-request-id", "req-123".parse().unwrap());

        let result = format_headers_redacted(&headers, false);

        // All sensitive headers should be present but with redacted values
        assert!(result.contains("authorization"));
        assert!(result.contains("proxy-authorization"));
        assert!(result.contains("set-cookie"));
        assert!(result.contains("cookie"));
        assert!(result.contains("x-api-key"));
        assert!(result.contains("x-auth-token"));
        assert!(result.contains("[REDACTED]"));

        // Ensure no sensitive values leaked
        assert!(!result.contains("Bearer token"));
        assert!(!result.contains("Basic creds"));
        assert!(!result.contains("session=abc"));
        assert!(!result.contains("api-key-123"));
        assert!(!result.contains("auth-token-456"));

        // Non-sensitive header should be present with actual value
        assert!(result.contains("x-request-id"));
        assert!(result.contains("req-123"));
    }

    #[test]
    fn test_format_headers_with_redaction_disabled() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer secret-token".parse().unwrap());
        headers.insert("x-api-key", "api-key-123".parse().unwrap());
        headers.insert("content-type", "application/json".parse().unwrap());

        let result = format_headers_redacted(&headers, true);

        // When redaction is disabled, all headers and values should be present
        assert!(result.contains("authorization"));
        assert!(result.contains("Bearer secret-token"));
        assert!(result.contains("x-api-key"));
        assert!(result.contains("api-key-123"));
        assert!(result.contains("content-type"));
        assert!(result.contains("application/json"));
        // [REDACTED] should NOT be present when redaction is disabled
        assert!(!result.contains("[REDACTED]"));
    }
}
