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

# Apache Iceberg Rust Cache Moka

This crate provides a [moka](https://github.com/moka-rs/moka) cache implementation for Apache Iceberg Rust. It is used to cache data in memory for faster access.

## Capacity

`MokaObjectCacheProvider::new_with_capacity` takes a **byte** budget, not an entry count. Each
cached manifest / manifest list is weighed as `entry_count × a per-entry constant`, and the two
caches the provider owns (manifests and manifest lists) each get the given budget — so the default
provider's aggregate ceiling is `2 × 32 MiB`. A budget of `0` disables caching.

The per-entry constants are owned by `crates/iceberg/src/io/object_cache.rs` and duplicated here
across the crate boundary; keep the two copies in step.

A cache supplied through `with_manifest_cache` / `with_manifest_list_cache` is used exactly as the
caller built it. This crate attaches no weigher to it, so a cache built with
`moka::sync::Cache::new(n)` is bounded by entry count.
