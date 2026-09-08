<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# F-HMS — deterministic Hive fixture downloads

**Branch:** `fix/hms-fixture-download`

**Base:** `85db42f285703682629b3b53bd1ebcd3091c6bc5`

**Retires when:** the shared Hive fixture repair merges to `TRO-Wolf/iceberg-rust` `main`.

## Scope audit

Audit date: 2026-09-08. Every proposition is `PROVEN`; there are no open or rejected clauses.

| ID | Proposition | Status | Evidence |
|---|---|---|---|
| C-001 | The repair changes only the shared HMS image assembly and its documentation. | PROVEN | Manifest: `dev/hms/Dockerfile`, `dev/map.md`, and this ledger. Rust, Cargo, workflows, Compose, and other fixtures are excluded. |
| C-002 | The runtime image remains `apache/hive:3.1.3`. | PROVEN | The base and repaired final stages use that exact reference. |
| C-003 | Runtime paths, configuration copy, final user, entrypoint, and command remain compatible. | PROVEN | The two JAR targets remain `/opt/hive/lib/<name>.jar`; `core-site.xml` remains `/opt/hadoop/etc/hadoop/core-site.xml`; the final instruction remains `USER hive`; no `ENTRYPOINT` or `CMD` override exists or will be added. |
| C-004 | Artifact versions and names do not change. | PROVEN | The planned inputs are `hadoop-aws-3.1.0.jar` and `aws-java-sdk-bundle-1.11.271.jar`, matching the base Dockerfile exactly. |
| C-005 | Downloads come from the canonical Maven Central repository over TLS. | PROVEN | Primary paths: `https://repo.maven.apache.org/maven2/org/apache/hadoop/hadoop-aws/3.1.0/` and `https://repo.maven.apache.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.271/`. |
| C-006 | Each downloaded artifact is checked before it can enter the runtime image. | PROVEN | Maven Central publishes SHA-1 values `6cb68b4e819ee8ca9b8b4b74847cd58feee74121` and `05c0c374c27dba1a8dbe8d6b21d4f79da8811f81`. Local TLS downloads matched both and produced the pinned SHA-256 values below. The build checks SHA-256 before the final-stage `COPY`. |
| C-007 | A corrupt, truncated, substituted, or missing artifact fails the build. | PROVEN | `curl --fail` returns nonzero for transfer and HTTP errors. `sha256sum --check` rejects missing files and byte changes. Each file receives its final name only after its check passes. Docker stops the `RUN` instruction before `COPY --from` can run. |
| C-008 | The downloader is an official released image with an immutable identity. | PROVEN | The curl project identifies `curlimages/curl` as an official distribution. Docker Hub's released `curlimages/curl:8.16.0` OCI index resolved to `sha256:463eaf6072688fe96ac64fa623fe73e1dbe25d8ad6c34404a669ad3ce1f104b6`. The Dockerfile pins both release and digest. No support-lifecycle claim is made. |
| C-009 | The runtime stage performs no package-manager operation. | PROVEN | The final stage contains only `COPY`, `ENV`, `USER`, and the existing configuration copy. It removes `apt-get`, package installation, repository metadata, and cleanup commands. |
| C-010 | TLS certificate verification remains enabled. Package-repository checks no longer apply. | PROVEN | The build restricts initial and redirected protocols to HTTPS and uses the downloader image's default trust store. It adds no insecure TLS flag. APT signature and repository-expiry checks are not applicable because no package operation remains. |
| C-011 | The observed PR failures happen before integration tests and share this base fixture. | PROVEN | Saved CI logs for PRs #273 and #274 show `apt-get update` exit 100 in `dev/hms/Dockerfile`; `make docker-up` exits 2. The saved comparison records Dockerfile, Compose, and workflow identity with base. This is unchanged-source evidence, not a reproduced base CI run. |
| C-012 | Copied JAR ownership and mode are explicit and safe for the runtime user. | PROVEN | Both final-stage copies set `hive:hive` ownership and mode `0644`. The files are readable by `hive` and are not writable by group or other users. |

## Pinned inputs

| Input | Pinned identity | Provenance |
|---|---|---|
| Downloader | `curlimages/curl:8.16.0@sha256:463eaf6072688fe96ac64fa623fe73e1dbe25d8ad6c34404a669ad3ce1f104b6` | Curl project's official container distribution and Docker Hub OCI index digest |
| Hadoop AWS | `hadoop-aws-3.1.0.jar` | Maven Central SHA-1 matched; SHA-256 `a18508b9348af095ea41301e439354dbd449e304ac44c6885b2b4fe78de88126` |
| AWS SDK bundle | `aws-java-sdk-bundle-1.11.271.jar` | Maven Central SHA-1 matched; SHA-256 `faf78ac4880f56cf52791d84ec1068ce7c66acc4295d580a726104b734c01fcd` |

## Failure model

| Failure | Required behavior | Planned evidence |
|---|---|---|
| Maven endpoint is unavailable, returns an HTTP error, or rejects TLS | `curl` stops the build during download. | Shell semantics inspection; Docker build if a daemon becomes available. |
| Response is partial or has unexpected bytes | SHA-256 verification fails before final-stage copy. | Run the exact verification command against the downloaded artifacts and one mutated copy. |
| Tag later points to another image | Digest pin keeps the selected OCI index immutable. | Registry digest response and Dockerfile source identity. |
| Runtime package repositories expire | Runtime build does not read package repositories. | Dockerfile inspection and a no-APT assertion. |

## Plan and verification

- [x] Verify disk headroom and the exact remote `main` base.
- [x] Establish the complete scope, provenance, checksums, and failure model before editing the fixture.
- [x] Replace runtime APT/download work with a pinned download-and-check build stage plus final-stage copies.
- [x] Update `dev/map.md` for the now-documented HMS fixture.
- [x] Run focused static and checksum failure checks, then the repository gates that do not require Docker.
- [x] Record the Docker blocker accurately and prepare the patch for independent review.

## Evidence

| Check | Result |
|---|---|
| Docker Hub registry digest lookup for `curlimages/curl:8.16.0` | PASS — response digest matched the Dockerfile pin. |
| Maven Central published SHA-1 cross-check | PASS — both downloaded JARs matched the published sidecars. |
| Frozen SHA-256 positive check | PASS — both JARs matched their Dockerfile values. |
| Frozen SHA-256 negative check | PASS — changing one Hadoop checksum nibble produced `FAILED` and a nonzero exit. |
| Independent Critic byte-corruption check | PASS — appending one byte to a separate Hadoop JAR copy produced `FAILED` and exit 1. The repository artifact remained unchanged. |
| Independent Critic source review | PASS — no S0 or S1 functional defect; one S2 ledger claim gap was corrected in C-008 and C-010. |
| Dockerfile source identity after documentation repair | PASS — SHA-256 remains `68bae9d233c1474e5beb2870b4f17a7e41ffed44fced180c6e60cdaa85a6f673`. |
| Static fixture contract | PASS — exact versions, HTTPS-only curl flags, two pre-rename checks, two mode/owner copies, unchanged config path, unchanged final image, and final `USER hive` are present; APT and package installation are absent. |
| `docker compose -f dev/docker-compose.yaml config` | PASS — HMS still resolves to `linux/amd64` and the existing `dev/hms` build context. |
| `git diff --check` | PASS. |
| `make check-agent-artifacts` | PASS. |
| `make check-matrix-anchors` | PASS. |
| `make check-comment-blocks` | PASS. |
| `make check-rust-file-size` | PASS — 11 gate tests and 446-file scan. |
| `make check-fmt` | PASS. |
| Dockerfile parser, image build, ownership inspection, and checksum sabotage build | BLOCKED — the selected Docker Desktop socket does not exist. No build-success or runtime-health claim is made. |

The full `make check`, Rust suites, runtime fixture health, nextest, and PR CI did not run. A full
local gate would create a fresh Cargo build tree for a Docker-only change, and the changed paths do
not enter Rust compilation. The Docker-backed normal test path remains blocked before execution on
this host.

`dev/spark/Dockerfile` has a separate runtime APT install. The saved PR logs do not prove that stage
fails because the shared HMS build stopped first. This repair does not expand into the Spark fixture.

## Independent re-verification (Grok 4.6, 2026-09-08)

Live network. JARs and `.sha1` sidecars were downloaded to `/tmp/grok-worker/hmsfix/dl/`. Staged copies under `/tmp/grok-worker/hmsfix/verify/` were read-only `cmp` cross-checks after the live hashes, not the source of the pins.

The `apt-get update` expiry is durable because Debian 11 left long-term support on 2026-08-31. The `bullseye-security` `InRelease` file is expired, not flapping.

This host was not used to build the image or run compose. The image build is proven by the PR's CI `Tests (default)` job after the orchestrator pushes.

| Command | Result |
|---|---|
| `sha256sum dev/hms/Dockerfile` | PASS — `68bae9d233c1474e5beb2870b4f17a7e41ffed44fced180c6e60cdaa85a6f673`, matches the tree and Codex's recorded identity. |
| Docker Hub token from `https://auth.docker.io/token?service=registry.docker.io&scope=repository:curlimages/curl:pull`, then `HEAD https://registry-1.docker.io/v2/curlimages/curl/manifests/8.16.0` with OCI index and Docker manifest-list Accept headers | PASS — HTTP 200; `docker-content-digest: sha256:463eaf6072688fe96ac64fa623fe73e1dbe25d8ad6c34404a669ad3ce1f104b6`. Matches the Dockerfile pin. |
| `curl` Maven Central `.sha1` sidecars for `hadoop-aws-3.1.0.jar` and `aws-java-sdk-bundle-1.11.271.jar` | PASS — `6cb68b4e819ee8ca9b8b4b74847cd58feee74121` and `05c0c374c27dba1a8dbe8d6b21d4f79da8811f81`. |
| `curl` both JARs from `https://repo.maven.apache.org/maven2/...` into `/tmp/grok-worker/hmsfix/dl/`; `sha1sum` and `sha256sum` of the bytes | PASS — SHA-1 matches the sidecars. SHA-256 is `a18508b9348af095ea41301e439354dbd449e304ac44c6885b2b4fe78de88126` and `faf78ac4880f56cf52791d84ec1068ce7c66acc4295d580a726104b734c01fcd`. Matches the Dockerfile pins. |
| `sha256sum --check` on both downloaded JARs against the Dockerfile SHA-256 lines | PASS — both `OK`, exit 0. |
| One flipped nibble on the Hadoop checksum line (`a18508` → `b18508`) piped to `sha256sum --check` | PASS — `FAILED`, exit 1. The original JAR was unchanged. |
| One appended NUL byte on a copy of `hadoop-aws-3.1.0.jar`, then `sha256sum --check` against the pinned digest | PASS — `FAILED`, exit 1. The original JAR still hashed to the pin. |
| `cmp` of the live downloads against `/tmp/grok-worker/hmsfix/verify/` | PASS — both files identical (exit 0). Live network remains the provenance. |
| Static contract read of `dev/hms/Dockerfile` against `git show origin/main:dev/hms/Dockerfile` | PASS — artifact names stay `hadoop-aws-3.1.0.jar` and `aws-java-sdk-bundle-1.11.271.jar`; versions stay `3.1.0` and `1.11.271`; curl uses `--proto '=https' --proto-redir '=https'`; each `sha256sum --check` precedes its `mv`; both `COPY --from` lines set `--chown=hive:hive --chmod=0644`; `COPY core-site.xml /opt/hadoop/etc/hadoop/core-site.xml` and `USER hive` are unchanged; final stage is still `FROM apache/hive:3.1.3`. |
| `grep -nE 'apt\|dpkg\|apk \|yum \|microdnf\|package' dev/hms/Dockerfile` | PASS — no match. The final stage has no package-manager operation. |
| `git diff --check` | PASS — exit 0. |
| `make check-fmt` | PASS — exit 0. |
| `make check-toml` | PASS — taplo already installed; `taplo check` 34 files, exit 0. |
| `make check-agent-artifacts` | PASS — exit 0. |
| `make check-matrix-anchors` | PASS — exit 0. |
| `make check-comment-blocks` | PASS — exit 0. |
| `make check-rust-file-size` | PASS — 11 unit tests and 446-file scan, exit 0. |
| `typos .` | PASS — exit 0. |
| `which docker` | Binary exists at `/usr/local/bin/docker`. Per the brief, no image build and no compose run were invoked. |

`make check` and cargo builds were not run. The change touches no Rust. `dev/spark/Dockerfile` still uses apt on an Ubuntu base and stays out of scope.
