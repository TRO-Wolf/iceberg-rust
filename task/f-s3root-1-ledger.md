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

# F-S3ROOT-1 — a bare-bucket S3 location is the bucket root

**Date:** 2026-09-12. **Branch:** `fix/f-s3root-1`.
**Base:** `9e3522e314e57cc6b074c9fb8fc1afe43f7c2b46`.
**Model:** swe-2-high.
**Path:** STANDARD because this changes an object-store path resolver on every
read, write, list and delete through the OpenDAL S3 backend.

This ledger retires when the fork change merges or the owner removes the unit.

## The defect

RePark run 7 on an S3 Tables table
(`CALL s3tables.system.remove_orphan_files`) failed before listing:

```
DataInvalid => Invalid s3 url: s3://f91970a7-...--table-s3, should start with one of
[s3://f91970a7-...--table-s3/, s3a://.../, s3n://.../]
```

An S3 Tables table's metadata `location` is the bare table bucket —
`s3://<id>--table-s3`, no trailing slash; the whole bucket is the table.
`DeleteOrphanFiles` lists `table.metadata().location()` verbatim and
`s3_relative_path` (`crates/storage/opendal/src/s3.rs`) only strips
`scheme://bucket/`, so the bare form answers `None` and
`crates/storage/opendal/src/lib.rs`'s `create_operator` refuses. Java's `S3URI`
treats `s3://bucket` as the bucket root (empty key); Spark's
`remove_orphan_files` on S3 Tables lists it.

The same shape defect exists in the GCS arm (`gs://bucket` refused) and the OSS
arm (`oss://bucket` refused): both run a `starts_with("{scheme}://{bucket}/")`
check. The azdls arm does not share the defect — `AzureStoragePath` parses
through `url`, whose `path()` is `""` for an authority-only URL, so
`abfss://fs@acct.dfs.core.windows.net` already resolves to the filesystem root.
This unit fixes the three object-store arms that share the defect and pins the
azdls root resolution as a control.

A second, silent defect sits one line below the first: `list` re-prefixes each
listed entry as `format!("{base}{}", entry.path())`. With a bare-bucket path,
`base` is `s3://bucket` (no trailing slash) while object-store entries carry no
leading slash, so locations would come back glued as `s3://bucketkey`. The join
now inserts `/` exactly when neither side carries it; every currently reachable
(base, entry) pair is byte-identical to before.

## Decisions (from the card)

- **D-1** `s3_relative_path("s3://b", "b")` -> `Some("")` for every alias, as does
  `"s3://b/"`. `"s3://bx"` and `"s3://b-other/..."` stay `None` — a bucket name is
  a whole host, never a prefix.
- **D-2** `FileIO::list("s3://b")` lists the bucket root, same set as
  `list("s3://b/")`.
- **D-3** `DeleteOrphanFiles` over a bucket-root table lists the root and finds
  the same orphan set a nested `s3://b/tbl` table would find under its own root.
- **D-4** red first; no behaviour change for locations that already carry a path.
- **D-5** no dependency change.

## Implemented minimal fix

`utils::scheme_relative_path(path, schemes, bucket)` resolves a location against
`{scheme}://{bucket}` for each alias: an exact host match yields `Some("")` and a
`{scheme}://{bucket}/` prefix yields the remainder; anything else (`{bucket}x`,
`{bucket}-other`, another scheme, userinfo/port-bearing authorities) stays `None`.
`s3_relative_path` delegates over `S3_SCHEME_ALIASES`; the GCS and OSS arms call it
with `["gs"]` / `["oss"]`. The S3 rejection text now names the accepted bare forms
(`s3://b, s3a://b, s3n://b or a path under it`).

`list` computes `base` as before, then joins `base` and `entry.path()` with a `/`
only when neither endpoint supplies one (`utils::join_list_location`) — so
`list("s3://b")` and `list("s3://b/")` return byte-identical locations.

`lib.rs` sits at its legacy file-size ceiling, so the new pins live next to the
code they pin rather than in `lib.rs`'s test module: the S3 `create_operator`
root pin is in `s3.rs`, the GCS/OSS root pins in `gcs.rs`/`oss.rs`, and the
join/resolver boundary pins in `utils.rs`. The ceiling row for `lib.rs` in
`scripts/check_rust_file_size.py` moved down with the file (2080 -> 2078), the
only direction that check allows.

## File allowlist

- `crates/storage/opendal/src/utils.rs`
- `crates/storage/opendal/src/s3.rs`
- `crates/storage/opendal/src/gcs.rs` (pin only — test module added)
- `crates/storage/opendal/src/oss.rs` (pin only — test module added)
- `crates/storage/opendal/src/lib.rs`
- `crates/storage/opendal/src/azdls.rs` (pin only — already correct)
- `crates/storage/opendal/tests/file_io_s3_test.rs`
- `scripts/check_rust_file_size.py` (lib.rs ceiling 2080 -> 2078)
- `task/f-s3root-1-ledger.md`
- `task/todo.md`

`Cargo.toml`, `Cargo.lock`, every dependency file, `.github/`, catalog code, and
RePark stay closed. No directory touched here has a `map.md`
(`crates/storage/opendal/` and `task/` carry none), so no map update is owed.

## Proposition ledger

`EXECUTION PROVEN` means the implemented patch has direct passing evidence.
`CI-PROVEN` means the pin is written and compiles under the armed gates but needs
MinIO, which this box does not run — execution evidence lands in CI.

| Clause | Checkable proposition | Proof obligation | Status |
|---|---|---|---|
| C-001 | `s3_relative_path` resolves `s3://b` and `s3://b/` to `Some("")` for every alias, keeps `s3://bx` and `s3://b-other/k` at `None`, and strips existing paths byte-exactly; `create_operator` resolves the bare bucket at every configured scheme. | Unit pins in `s3.rs` (resolver + create_operator), `utils.rs` (boundary), `gcs.rs`, `oss.rs`; red on the base, green after the fix. | EXECUTION PROVEN |
| C-002 | `FileIO::list("s3://bucket1")` returns the same `FileInfo` location set as `list("s3://bucket1/")`, including a written object at its exact `s3://bucket1/<name>` location. | MinIO integration pin in `file_io_s3_test.rs`. | CI-PROVEN |
| C-003 | `DeleteOrphanFiles` over a table at `s3://bucket1` lists the bucket root, flags a planted orphan, spares the table's own metadata file, and its recorded delete set equals the reported orphan set; a nested control table's sweep is exactly its planted orphan. | MinIO integration pin via `MemoryCatalog` over the S3 factory. | CI-PROVEN |
| C-004 | Gates green: unit tests, MinIO suite compile, `iceberg` lib tests, datafusion integration compile, `make check`, comment fence clean. | Run and paste evidence below. | EXECUTION PROVEN |

## Base-red evidence

`CARGO_BUILD_JOBS=16 cargo test -p iceberg-storage-opendal --all-features --lib` on
the exact base (`9e3522e31`, pins added, production code untouched) exited 101 —
**4 red out of 6 new pins**, the two controls green:

```
running 53 tests
test s3::tests::test_s3_relative_path_bucket_root_is_empty_key ... FAILED
test tests::s3_scheme_alias::test_create_operator_bucket_root_resolves_to_empty_key ... FAILED
test tests::test_opendal_gcs_bucket_root_relative_path_is_empty ... FAILED
test tests::test_opendal_oss_bucket_root_relative_path_is_empty ... FAILED

---- s3::tests::test_s3_relative_path_bucket_root_is_empty_key stdout ----
assertion `left == right` failed: s3://mybucket must resolve to the empty key
  left: None
 right: Some("")

---- tests::s3_scheme_alias::test_create_operator_bucket_root_resolves_to_empty_key stdout ----
s3://my-bucket must resolve for configured s3: DataInvalid => Invalid s3 url:
s3://my-bucket, should start with one of [s3://my-bucket/, s3a://my-bucket/,
s3n://my-bucket/] (storage configured for scheme s3)

---- tests::test_opendal_gcs_bucket_root_relative_path_is_empty stdout ----
bare bucket must resolve: DataInvalid => Invalid gcs url: gs://gcs-bucket,
should start with gs://gcs-bucket/

---- tests::test_opendal_oss_bucket_root_relative_path_is_empty stdout ----
bare bucket must resolve: DataInvalid => Invalid oss url: oss://oss-bucket,
should start with oss://oss-bucket/

test result: FAILED. 49 passed; 4 failed; 0 ignored
```

The `create_operator` failure text is byte-identical in shape to the RePark run-7
error (`Invalid s3 url: …, should start with one of [s3://…/, s3a://…/, s3n://…/]`).
Controls on the base: `test_s3_relative_path_bucket_name_is_whole_host` (the two
negative cases are already `None`) and `test_azdls_filesystem_root_resolves_to_empty_path`
(`relative_path == ""` — the azdls arm never had the defect) both passed green.

The red run above names the pin locations at pin-write time (the `create_operator`
pins sat in `lib.rs`'s test module). They were then relocated to `s3.rs`,
`gcs.rs` and `oss.rs` to keep `lib.rs` under its legacy file-size ceiling — same
assertions, same names, different module files.

## Execution evidence

`CARGO_BUILD_JOBS=16 cargo test -p iceberg-storage-opendal --all-features --lib`
after the fix:

```
test result: ok. 55 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Mutation check (the helper returned the raw suffix without consuming the `/` —
`Some("/")` for roots, `Some("/k")` for keys): **18 of 55 red**, including
`utils::tests::test_scheme_relative_path_bucket_boundary`, every bare-root pin,
the alias tests, and the operator-cache path tests. Restored helper: 55/55 green.
The pins are load-bearing.

Gate runs:

```
cargo test -p iceberg-storage-opendal            -> 46 lib passed; integration binary built;
                                                    6 MinIO tests fail connection-refused
                                                    (docker not run on this box — excused;
                                                    the C-003 pin reached the metadata write
                                                    at s3://bucket1/metadata/... before the
                                                    unavailable service refused)
cargo test -p iceberg-storage-opendal --all-features --lib -> 55 passed, 0 failed
cargo test -p iceberg-storage-opendal --all-features --tests --no-run -> compiles clean
cargo test -p iceberg --lib                      -> 3672 passed, 0 failed, 8 ignored
cargo test -p iceberg-datafusion --no-run        -> all test binaries compile clean
typos                                            -> exit 0
cargo fmt --all -- --check                       -> clean
git diff --check                                 -> clean
make check                                       -> all gates green (fmt, clippy -D warnings
                                                    --all-targets --all-features --workspace,
                                                    taplo, cargo-machete, agent-artifacts,
                                                    matrix-anchors, comment-blocks,
                                                    rust-file-size: 458 files clean)
```

Comment fence on the staged tree (the round's exact command):

```
git diff --cached -- '*.rs' '*.toml' '*.sh' '*.yml' \
  | grep -P '^\+\s*(//|#(?!\[|!\[))' | grep -v -P '^\+\s*///? ?[A-Z].*\.$'
-> no output: zero added comment lines in code
```

## Notes for the Critic

- The wrong-bucket guard in `create_operator` is vacuous by construction (the
  bucket is derived from the path's own host); the host-boundary rule is pinned
  at the resolver, where a mismatched bucket is expressible.
- `exists`, `read`, `write`, `delete` and `delete_prefix` on `s3://b` now behave
  exactly as `s3://b/` already did (both resolve to relative `""` /
  `remove_all("/")`); only `list` needed a join fix.
- The C-002 set-equality assertion is deterministic up to a same-millisecond
  concurrent write to `bucket1` by another test in this file — the only writers
  of `bucket1` in the workspace are this file's tests; the discriminating
  assertions (root listed, exact locations) do not depend on it.
