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

// Hour-0 / after microbench for FK1 eq-delete keyset apply.
// Cargo.toml is frozen — run as a standalone example via:
//   cargo run -p iceberg --example ...  is unavailable without Cargo.toml edit.
// Invocation (from worktree, after `cargo test -p iceberg --lib` has built deps):
//   rustc is NOT used alone; instead use the in-tree unit microbench below via:
//   cargo test -p iceberg --lib fk1_eq_delete_apply_microbench -- --nocapture --ignored
//
// This file documents the bench matrix for the ledger; the live numbers come from the
// ignored test `fk1_eq_delete_apply_microbench` in equality_delete_set.rs.
//
// Matrix: 1M data rows × {100k, 1M} eq-deletes; single Long key; non-null.
// Metrics: wall ns/row, approximate alloc via peak RSS if /proc available.
