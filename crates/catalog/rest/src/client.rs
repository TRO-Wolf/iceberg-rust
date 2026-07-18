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

//! REST catalog HTTP client, including OAuth2 credential exchange and automatic token refresh.
//!
//! # DEVIATIONS from Java `iceberg-core` 1.10.0 OAuth2 (GAP_MATRIX R159)
//!
//! The refresh *trigger* is faithful: [`TokenState::from_expires_in`] reproduces Java
//! `OAuth2Util$AuthSession.scheduleTokenRefresh` exactly (`refreshWindow = min(ttl/10, 5min)`,
//! `wait = max(ttl - refreshWindow, 10ms)`), and defaults match
//! (`token-refresh-enabled` = true). Two intentional divergences:
//!
//! 1. **Refresh grant type.** Java refreshes a *still-valid* token with the OAuth token-exchange
//!    grant (`grant_type=urn:ietf:params:oauth:grant-type:token-exchange`, current token as
//!    `subject_token`) and only re-fetches with the credential once the token has *expired*. This
//!    fork has no token-exchange machinery; it always refreshes through the existing
//!    `client_credentials` exchange to the SAME `get_token_endpoint()`. Observably equivalent for a
//!    credential-bearing client (a fresh valid token arrives before expiry); the wire grant differs.
//! 2. **Token-only clients (no credential) are not refreshed.** Java can keep a token-only session
//!    alive by token-exchanging the current token against itself. Without a credential this fork has
//!    nothing to exchange (and no token-exchange path), so a configured-`token`-only client behaves
//!    as today: the token is used until it expires. Note Java also cannot refresh a token-only
//!    session once the token is actually *expired* (`refreshExpiredToken` returns null when the
//!    credential is null) — the gap is limited to the *proactive* pre-expiry window.
//!
//! Recommendation: closing divergence (1)/(2) requires implementing the OAuth token-exchange grant
//! (`subject_token`/`subject_token_type`), a larger surface than this refresh adaptation and a
//! separate GAP_MATRIX item; it is out of scope here.

use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::time::{Duration, Instant};

use http::StatusCode;
use iceberg::{Error, ErrorKind, Result};
use reqwest::header::HeaderMap;
use reqwest::{Client, IntoUrl, Method, Request, RequestBuilder, Response};
use serde::de::DeserializeOwned;
use tokio::sync::Mutex;

use crate::RestCatalogConfig;
use crate::types::{ErrorResponse, TokenResponse};

/// Java oracle (`OAuth2Util$AuthSession.MAX_REFRESH_WINDOW_MILLIS`, iceberg-core 1.10.0): the
/// refresh window is capped at 5 minutes. A token is refreshed no earlier than `expiresAt - 5min`
/// even when a proportional (`ttl/10`) window would be larger.
const MAX_REFRESH_WINDOW_MILLIS: u64 = 300_000;
/// Java oracle (`OAuth2Util$AuthSession.MIN_REFRESH_WAIT_MILLIS`): the scheduled wait before a
/// refresh never drops below 10ms, so a near-zero-lifetime token still gets a tiny grace window
/// rather than refreshing in a tight loop.
const MIN_REFRESH_WAIT_MILLIS: u64 = 10;

/// A cached OAuth access token together with the monotonic instant at which it becomes due for a
/// proactive refresh.
///
/// SECURITY: `token` is bearer-secret material. This type deliberately does NOT derive `Debug`;
/// [`HttpClient`]'s `Debug` never renders it, and no error/log path on the refresh code touches it.
#[derive(Clone)]
struct TokenState {
    /// The bearer access token.
    token: String,
    /// Monotonic instant at which the token enters Java's refresh window (`expiresAt -
    /// refreshWindow`). `None` means "never proactively refresh": the token carried no
    /// `expires_in`, mirroring Java leaving `expiresAtMillis` null and scheduling no refresh.
    refresh_at: Option<Instant>,
}

impl TokenState {
    /// A token with no known expiry — a config-provided `token`, or a token response without
    /// `expires_in`. Never proactively refreshed, mirroring Java `fromAccessToken` /
    /// `fromTokenResponse` leaving `expiresAtMillis` null so no refresh is scheduled.
    fn without_expiry(token: String) -> Self {
        Self {
            token,
            refresh_at: None,
        }
    }

    /// Derive the refresh instant from an `expires_in` (seconds), reproducing Java's
    /// `OAuth2Util$AuthSession.scheduleTokenRefresh` arithmetic exactly:
    ///
    /// ```text
    /// ttl            = expires_in
    /// refreshWindow  = min(ttl / 10, MAX_REFRESH_WINDOW_MILLIS)   // min(10% of ttl, 5min)
    /// wait           = max(ttl - refreshWindow, MIN_REFRESH_WAIT_MILLIS)
    /// refreshAt      = now + wait
    /// ```
    ///
    /// All millisecond math is saturating on `u64`; the final `Instant` add is checked, so an
    /// absurd `expires_in` can never panic — it falls back to "never refresh" (`None`).
    fn from_expires_in(now: Instant, token: String, expires_in_secs: Option<u64>) -> Self {
        let refresh_at = expires_in_secs.and_then(|secs| {
            let ttl_ms = secs.saturating_mul(1000);
            let refresh_window_ms = (ttl_ms / 10).min(MAX_REFRESH_WINDOW_MILLIS);
            let wait_ms = ttl_ms
                .saturating_sub(refresh_window_ms)
                .max(MIN_REFRESH_WAIT_MILLIS);
            now.checked_add(Duration::from_millis(wait_ms))
        });
        Self { token, refresh_at }
    }

    /// Whether `now` has reached the token's refresh instant. A token with no known expiry is never
    /// due. Uses monotonic [`Instant`] comparison only — no wall-clock arithmetic, no panic on
    /// clock skew.
    fn is_due_for_refresh(&self, now: Instant) -> bool {
        self.refresh_at.is_some_and(|at| now >= at)
    }
}

pub(crate) struct HttpClient {
    client: Client,

    /// The token to be used for authentication, with its derived refresh instant.
    ///
    /// It's possible to fetch the token from the server while needed.
    token: Mutex<Option<TokenState>>,
    /// Single-flight gate for token acquisition/refresh. Concurrent requests that all observe a
    /// stale token serialize here so exactly ONE token-endpoint exchange happens (Java serializes
    /// refresh on its scheduler thread; the lazy adaptation serializes on this gate instead).
    ///
    /// Lock order: on the acquisition path, `refresh_gate` is acquired BEFORE `token`. The `token`
    /// mutex is only ever held for brief, non-`await` critical sections (clone-out / store-back),
    /// so the two locks never deadlock and no lock is held across the HTTP `await` except the
    /// `refresh_gate` itself — which is the single-flight mechanism and is bounded to one exchange.
    refresh_gate: Mutex<()>,
    /// Whether proactive token refresh is enabled (`token-refresh-enabled`; Java default `true`).
    /// When `false`, behavior is exactly today's: exchange once, cache forever, never refresh.
    token_refresh_enabled: bool,
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
        // SEC-008: `extra_headers` carries any `header.*`-configured request headers, including
        // auth (`header.authorization`, `header.x-api-key`, …). Printing the `HeaderMap` raw
        // would leak those values into any `{:?}`/`tracing` of the client (or a struct embedding
        // it). Render it through the same `SENSITIVE_HEADERS` policy the error-log path uses,
        // ALWAYS redacting here regardless of `disable_header_redaction` (that opt-out is scoped
        // to error logs, not to `Debug`). The wrapper writes the redacted map verbatim so the
        // field renders cleanly instead of as an escaped string.
        struct RedactedHeaders<'a>(&'a HeaderMap);
        impl Debug for RedactedHeaders<'_> {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                f.write_str(&format_headers_redacted(self.0, false))
            }
        }

        f.debug_struct("HttpClient")
            .field("client", &self.client)
            .field("extra_headers", &RedactedHeaders(&self.extra_headers))
            .finish_non_exhaustive()
    }
}

impl HttpClient {
    /// Create a new http client.
    pub fn new(cfg: &RestCatalogConfig) -> Result<Self> {
        let extra_headers = cfg.extra_headers()?;
        Ok(HttpClient {
            client: cfg.client().unwrap_or_default(),
            token: Mutex::new(cfg.token().map(TokenState::without_expiry)),
            refresh_gate: Mutex::new(()),
            token_refresh_enabled: cfg.token_refresh_enabled(),
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
            token: Mutex::new(
                cfg.token()
                    .map(TokenState::without_expiry)
                    .or_else(|| self.token.into_inner()),
            ),
            refresh_gate: Mutex::new(()),
            token_refresh_enabled: cfg.token_refresh_enabled(),
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
        self.token.lock().await.as_ref().map(|s| s.token.clone())
    }

    /// Test-only: install a token state directly (with a caller-chosen refresh instant) so a test
    /// can place a token that is already due for refresh without waiting on wall-clock expiry.
    #[cfg(test)]
    async fn install_token_state(&self, token: &str, refresh_at: Option<Instant>) {
        *self.token.lock().await = Some(TokenState {
            token: token.to_string(),
            refresh_at,
        });
    }

    /// Exchange the configured credential for an access token via the `client_credentials` grant.
    ///
    /// Returns the access token together with its `expires_in` (seconds), when the server supplied
    /// one — the caller derives the proactive-refresh instant from it. A missing `expires_in`
    /// yields `None`, which (mirroring Java) means the token is never proactively refreshed.
    pub(crate) async fn exchange_credential_for_token(&self) -> Result<(String, Option<u64>)> {
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
        Ok((auth_res.access_token, auth_res.expires_in))
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
        let (new_token, expires_in) = self.exchange_credential_for_token().await?;
        let state = TokenState::from_expires_in(Instant::now(), new_token, expires_in);
        *self.token.lock().await = Some(state);
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
    /// ## Automatic token refresh (GAP_MATRIX R159)
    ///
    /// When `token-refresh-enabled` is on (Java default) and a `credential` is configured, a cached
    /// token that has entered its refresh window (see [`TokenState::from_expires_in`], which mirrors
    /// Java `OAuth2Util$AuthSession.scheduleTokenRefresh`) or has already expired is refreshed
    /// *before use*, so no request is ever sent with a known-expired bearer token. Concurrent
    /// requests that all observe a stale token collapse to a single token-endpoint exchange via
    /// [`Self::obtain_token_single_flight`]. When refresh is disabled, behavior is exactly the
    /// legacy one: exchange once, cache forever.
    ///
    /// SANCTIONED ADAPTATION: Java runs a background scheduler thread and, for a still-valid token,
    /// refreshes via the OAuth *token-exchange* grant (current token as `subject_token`); only an
    /// already-expired token is re-fetched with the `credential`. This async-Rust library has no
    /// token-exchange machinery and no background task, so it refreshes *lazily, before use*, always
    /// through the existing `client_credentials` exchange ([`Self::exchange_credential_for_token`],
    /// which POSTs to `get_token_endpoint()` — the SAME endpoint and credential location as the
    /// initial exchange). The observable contract still holds for credential-bearing clients: no
    /// known-expired token is sent, and tokens refresh proactively near expiry. The divergence for
    /// token-only clients (no credential) is recorded in the module's DEVIATIONS note.
    async fn authenticate(&self, req: &mut Request) -> Result<()> {
        // Clone the current token state without holding the lock across any await.
        let snapshot = self.token.lock().await.clone();

        if self.credential.is_none() && snapshot.is_none() {
            return Ok(());
        }

        // Fast path: a cached token that is not due for a proactive refresh is used as-is, with no
        // lock held and no token-endpoint traffic. Refresh only engages when it is enabled AND we
        // have a credential to exchange AND the token has entered its refresh window / expired.
        let needs_acquire = match &snapshot {
            Some(state) => {
                self.token_refresh_enabled
                    && self.credential.is_some()
                    && state.is_due_for_refresh(Instant::now())
            }
            None => true,
        };

        let token = if needs_acquire {
            self.obtain_token_single_flight(snapshot).await?
        } else {
            // `snapshot` is `Some` here (the `None` arm forces `needs_acquire`).
            snapshot
                .expect("a non-acquiring path always has a cached token")
                .token
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

    /// Acquire a usable token under the single-flight [`Self::refresh_gate`], collapsing a stampede
    /// of concurrent stale requests into ONE token-endpoint exchange.
    ///
    /// `prev` is the caller's pre-gate snapshot: `None` means there is no cached token yet (initial
    /// acquisition), `Some` means a refresh of an existing (now-stale) token. This distinction
    /// controls the failure policy, mirroring Java's two paths:
    ///
    /// - **Initial acquisition failure propagates.** With no token to fall back on, an exchange
    ///   error is returned to the caller (as the legacy code did, and as Java's initial `fetchToken`
    ///   does at session creation).
    /// - **Refresh failure is suppressed, keeping the old token.** Java runs the scheduled refresh
    ///   under `Tasks…suppressFailureWhenFinished().retry(...)`; on exhausted retries the failure is
    ///   swallowed and the previous token stays in use rather than erroring the in-flight request.
    ///   The lazy adaptation matches that observable behavior: it keeps serving the prior token.
    ///
    /// Lock order (documented on the fields): `refresh_gate` is taken first; `token` is then locked
    /// only for brief, non-`await` sections. The single HTTP exchange awaits while holding
    /// `refresh_gate` (the single-flight mechanism) — bounded to one round-trip — and never while
    /// holding the `token` mutex.
    async fn obtain_token_single_flight(&self, prev: Option<TokenState>) -> Result<String> {
        let _gate = self.refresh_gate.lock().await;

        // Double-check under the gate: a concurrent holder may have already refreshed while we
        // waited. If the now-current token is usable, reuse it and skip the exchange entirely.
        let current = self.token.lock().await.clone();
        if let Some(state) = &current {
            let still_stale = self.token_refresh_enabled
                && self.credential.is_some()
                && state.is_due_for_refresh(Instant::now());
            if !still_stale {
                return Ok(state.token.clone());
            }
        }

        match self.exchange_credential_for_token().await {
            Ok((new_token, expires_in)) => {
                let state = TokenState::from_expires_in(Instant::now(), new_token, expires_in);
                let token = state.token.clone();
                *self.token.lock().await = Some(state);
                Ok(token)
            }
            Err(error) => match current.or(prev) {
                // Refresh of an existing token failed: keep serving it (Java suppresses the
                // failure). No token/credential material is attached to any log — only the safe
                // error is traced.
                Some(state) => {
                    tracing::warn!(
                        ?error,
                        "OAuth token refresh failed; continuing with the previously cached token"
                    );
                    Ok(state.token)
                }
                // No token to fall back on: propagate (initial acquisition).
                None => Err(error),
            },
        }
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

    /// Risk (SEC-008): `HttpClient`'s `Debug` prints `extra_headers`, which carries any
    /// `header.*`-configured request headers — including auth headers like `header.authorization`.
    /// A raw `HeaderMap` dump leaks those secret VALUES into any `{:?}`/`tracing` of the client
    /// (or a struct embedding it). Pins that a sensitive header value is redacted while the header
    /// NAME stays visible, and a non-sensitive header keeps its value. Mutation: revert the Debug
    /// impl to `.field("extra_headers", &self.extra_headers)` → RED.
    #[tokio::test]
    async fn test_http_client_debug_redacts_sensitive_extra_header() {
        const HEADER_SECRET: &str = "SUPER_SECRET_AUTH_HEADER_DO_NOT_LEAK";

        let mut props = HashMap::new();
        props.insert(
            "header.authorization".to_string(),
            format!("Bearer {HEADER_SECRET}"),
        );
        props.insert(
            "header.x-request-id".to_string(),
            "req-visible-123".to_string(),
        );

        let client = HttpClient::new(
            &RestCatalogConfig::builder()
                .uri("http://localhost".to_string())
                .props(props)
                .build(),
        )
        .expect("HttpClient must build from a header-bearing config");

        let debug = format!("{client:?}");

        // The sensitive header VALUE must never appear.
        assert!(
            !debug.contains(HEADER_SECRET),
            "HttpClient Debug leaked a sensitive header value: {debug}"
        );
        // Its presence is still signalled: the header NAME and redaction marker survive.
        assert!(
            debug.contains("authorization"),
            "expected the redacted header name to remain: {debug}"
        );
        assert!(
            debug.contains("[REDACTED]"),
            "expected the redaction marker: {debug}"
        );
        // A non-sensitive header keeps its value for diagnostics.
        assert!(
            debug.contains("x-request-id") && debug.contains("req-visible-123"),
            "non-sensitive header should stay visible: {debug}"
        );
    }

    // ========================================================================
    // GAP_MATRIX R159 — automatic OAuth token refresh (lazy refresh-before-use)
    // ========================================================================

    use std::sync::Arc;

    use mockito::{Server, ServerGuard};

    /// Build a credential-bearing client pointed at the given token endpoint host.
    fn refresh_client(server_url: &str, refresh_enabled: bool) -> HttpClient {
        let mut props = HashMap::new();
        props.insert("credential".to_string(), "client1:secret1".to_string());
        if !refresh_enabled {
            props.insert("token-refresh-enabled".to_string(), "false".to_string());
        }
        HttpClient::new(
            &RestCatalogConfig::builder()
                .uri(server_url.to_string())
                .props(props)
                .build(),
        )
        .expect("client must build from a credential-bearing config")
    }

    /// A token-endpoint mock returning `token` with an optional `expires_in`, matching `expect`
    /// requests exactly.
    async fn token_mock(
        server: &mut ServerGuard,
        token: &str,
        expires_in: Option<u64>,
        expect: usize,
    ) -> mockito::Mock {
        let expires_field = expires_in
            .map(|s| format!(r#", "expires_in": {s}"#))
            .unwrap_or_default();
        let body = format!(
            r#"{{"access_token": "{token}", "token_type": "Bearer", "issued_token_type": "urn:ietf:params:oauth:token-type:access_token"{expires_field}}}"#
        );
        server
            .mock("POST", "/v1/oauth/tokens")
            .with_status(200)
            .with_body(body)
            .expect(expect)
            .create_async()
            .await
    }

    async fn authenticated_bearer(client: &HttpClient, base: &str) -> Option<String> {
        let mut req = client
            .request(Method::GET, format!("{base}/v1/probe"))
            .build()
            .expect("probe request must build");
        client
            .authenticate(&mut req)
            .await
            .expect("authenticate must succeed");
        req.headers()
            .get(http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
    }

    /// Pins that the refresh instant reproduces Java `OAuth2Util$AuthSession.scheduleTokenRefresh`
    /// (`refreshWindow = min(ttl/10, 5min)`, `wait = max(ttl - refreshWindow, 10ms)`) and that a
    /// missing `expires_in` yields NO refresh instant (Java schedules none). Mutation: changing the
    /// window formula (e.g. dropping the `min(_, MAX_REFRESH_WINDOW_MILLIS)` cap) shifts these
    /// instants → RED.
    #[test]
    fn test_refresh_at_matches_java_schedule_rule() {
        let now = Instant::now();

        // Short-lived token: ttl=100s → window=min(10s, 5min)=10s → wait=90s.
        let short = TokenState::from_expires_in(now, "t".to_string(), Some(100));
        let short_at = short
            .refresh_at
            .expect("a token with expires_in has an instant");
        let short_wait = short_at.duration_since(now);
        assert_eq!(
            short_wait,
            Duration::from_secs(90),
            "ttl=100s must refresh after 90s (window = ttl/10 = 10s)"
        );

        // Long-lived token: ttl=86400s → ttl/10=8640s but capped at 5min → window=300s → wait=86100s.
        let long = TokenState::from_expires_in(now, "t".to_string(), Some(86400));
        let long_at = long
            .refresh_at
            .expect("a token with expires_in has an instant");
        assert_eq!(
            long_at.duration_since(now),
            Duration::from_secs(86400 - 300),
            "the refresh window must be capped at MAX_REFRESH_WINDOW_MILLIS (5min)"
        );

        // Near-zero ttl: window=0 → wait=max(0, 10ms)=10ms (never a tight loop).
        let tiny = TokenState::from_expires_in(now, "t".to_string(), Some(0));
        assert_eq!(
            tiny.refresh_at
                .expect("expires_in=0 still derives an instant")
                .duration_since(now),
            Duration::from_millis(MIN_REFRESH_WAIT_MILLIS),
        );

        // Missing expires_in → no refresh instant → never proactively refreshed.
        let none = TokenState::from_expires_in(now, "t".to_string(), None);
        assert!(
            none.refresh_at.is_none(),
            "a token without expires_in must not be scheduled for refresh"
        );
        assert!(
            !none.is_due_for_refresh(now + Duration::from_secs(10_000_000)),
            "a no-expiry token is never due, even far in the future"
        );
    }

    /// Single-flight pin: N concurrent requests that all observe a token inside its refresh window
    /// trigger EXACTLY ONE token-endpoint exchange, and every request ends up with the refreshed
    /// token. Mutation A (drop the `refresh_gate` in `obtain_token_single_flight`) → the stampede
    /// hits the endpoint N times → the `expect(1)` mock assertion goes RED. Mutation B (disable the
    /// `is_due_for_refresh` check) → zero refreshes → the `Bearer refreshed` assertion goes RED.
    #[tokio::test]
    async fn test_near_expiry_single_flight_refresh_under_concurrency() {
        let mut server = Server::new_async().await;
        let mock = token_mock(&mut server, "refreshed", Some(86400), 1).await;

        let client = Arc::new(refresh_client(&server.url(), true));
        // A cached token already inside its refresh window (refresh_at in the past).
        client
            .install_token_state("stale", Some(Instant::now() - Duration::from_secs(1)))
            .await;

        let base = server.url();
        let mut handles = Vec::new();
        for _ in 0..8 {
            let client = Arc::clone(&client);
            let base = base.clone();
            handles.push(tokio::spawn(async move {
                authenticated_bearer(&client, &base).await
            }));
        }

        for handle in handles {
            let bearer = handle.await.expect("task must join");
            assert_eq!(
                bearer.as_deref(),
                Some("Bearer refreshed"),
                "every concurrent request must use the refreshed token"
            );
        }

        // Exactly one exchange despite eight racing requests.
        mock.assert_async().await;
    }

    /// A cached token that is NOT yet inside its refresh window must be used as-is, with zero
    /// token-endpoint traffic (no gratuitous refresh). Mutation: making every token "due" would
    /// force an exchange here → the `expect(0)` mock goes RED.
    #[tokio::test]
    async fn test_fresh_token_is_not_refreshed() {
        let mut server = Server::new_async().await;
        let mock = token_mock(&mut server, "should-not-be-used", Some(86400), 0).await;

        let client = refresh_client(&server.url(), true);
        // Refresh instant an hour out: nowhere near due.
        client
            .install_token_state("fresh", Some(Instant::now() + Duration::from_secs(3600)))
            .await;

        let bearer = authenticated_bearer(&client, &server.url()).await;
        assert_eq!(
            bearer.as_deref(),
            Some("Bearer fresh"),
            "a fresh token must be reused verbatim"
        );
        mock.assert_async().await; // zero exchanges
    }

    /// Regression pin for today's behavior: with `token-refresh-enabled=false`, an already-due
    /// token is STILL never refreshed — zero token-endpoint traffic ever. Mutation: inverting the
    /// enabled flag (treating disabled as enabled) refreshes the due token → the `expect(0)` mock
    /// goes RED.
    #[tokio::test]
    async fn test_refresh_disabled_never_refreshes() {
        let mut server = Server::new_async().await;
        let mock = token_mock(&mut server, "should-not-be-used", Some(86400), 0).await;

        let client = refresh_client(&server.url(), false);
        // Token is well past its (hypothetical) refresh instant, yet refresh is disabled.
        client
            .install_token_state("legacy", Some(Instant::now() - Duration::from_secs(3600)))
            .await;

        let bearer = authenticated_bearer(&client, &server.url()).await;
        assert_eq!(
            bearer.as_deref(),
            Some("Bearer legacy"),
            "with refresh disabled the cached token must be used unchanged"
        );
        mock.assert_async().await; // zero exchanges
    }

    /// A token response without `expires_in` must disable proactive refresh for that token (Java
    /// schedules no refresh when the expiry is unknown). The initial exchange happens once; a
    /// subsequent authenticated request must NOT hit the endpoint again. Pinned end-to-end with an
    /// `expect(1)` mock (only the initial fetch).
    #[tokio::test]
    async fn test_missing_expires_in_disables_refresh() {
        let mut server = Server::new_async().await;
        let mock = token_mock(&mut server, "no-expiry", None, 1).await;

        let client = refresh_client(&server.url(), true);

        // Initial fetch: no cached token → one exchange, caches a no-expiry token.
        let first = authenticated_bearer(&client, &server.url()).await;
        assert_eq!(first.as_deref(), Some("Bearer no-expiry"));

        // Any number of later requests reuse it with no further exchange.
        let second = authenticated_bearer(&client, &server.url()).await;
        assert_eq!(second.as_deref(), Some("Bearer no-expiry"));

        mock.assert_async().await; // exactly one (the initial fetch)
    }

    /// Refresh FAILURE is suppressed and the previously cached token is kept (Java runs the refresh
    /// under `Tasks…suppressFailureWhenFinished()` and keeps the old token on exhausted retries).
    /// The in-flight request must NOT error; it proceeds with the prior token. Mutation: propagating
    /// the refresh error instead of falling back would make `authenticate` return `Err` → the
    /// `expect("authenticate must succeed")` in `authenticated_bearer` goes RED.
    #[tokio::test]
    async fn test_refresh_failure_keeps_previous_token() {
        let mut server = Server::new_async().await;
        // The refresh exchange fails (500); mock is hit once (the attempted refresh).
        let mock = server
            .mock("POST", "/v1/oauth/tokens")
            .with_status(500)
            .with_body(r#"{"error": {"message": "boom", "type": "ServerError", "code": 500}}"#)
            .expect(1)
            .create_async()
            .await;

        let client = refresh_client(&server.url(), true);
        client
            .install_token_state("old", Some(Instant::now() - Duration::from_secs(1)))
            .await;

        let bearer = authenticated_bearer(&client, &server.url()).await;
        assert_eq!(
            bearer.as_deref(),
            Some("Bearer old"),
            "a failed refresh must keep serving the previously cached token"
        );
        mock.assert_async().await;
    }

    /// The INITIAL acquisition failure (no token to fall back on) must propagate as an error — a
    /// caller with no usable token cannot silently proceed. This complements the refresh-failure
    /// suppression above: the two paths are distinguished by whether a previous token existed.
    #[tokio::test]
    async fn test_initial_acquisition_failure_propagates() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/oauth/tokens")
            .with_status(500)
            .with_body(r#"{"error": {"message": "boom", "type": "ServerError", "code": 500}}"#)
            .expect(1)
            .create_async()
            .await;

        let client = refresh_client(&server.url(), true);
        // No cached token: the first authenticate must exchange, and that exchange fails.
        let mut req = client
            .request(Method::GET, format!("{}/v1/probe", server.url()))
            .build()
            .expect("probe request must build");
        let result = client.authenticate(&mut req).await;
        assert!(
            result.is_err(),
            "an initial token acquisition failure must propagate, not be swallowed"
        );
        mock.assert_async().await;
    }
}
