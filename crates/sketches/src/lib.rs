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

//! # iceberg-sketches
//!
//! A dependency-free, byte-exact port of the Apache DataSketches **theta** `CompactSketch`
//! serialized format — the payload of Apache Iceberg's `apache-datasketches-theta-v1` Puffin blob
//! used to carry per-column NDV (number of distinct values) statistics.
//!
//! The goal is **cross-engine byte compatibility**: a theta blob this crate writes is readable by
//! Java DataSketches (and therefore Spark/Trino/Flink Iceberg engines), and blobs those engines
//! write are readable here. To guarantee that, the hash ([`hash`]) and the build/serialize path
//! ([`ThetaSketch`]) are ported one-to-one from the DataSketches 1.10.0 jar bytecode, and pinned
//! against Java-generated fixtures (see the crate's `testdata`).
//!
//! ## Usage
//!
//! ```
//! use iceberg_sketches::ThetaSketch;
//!
//! let mut sketch = ThetaSketch::new();
//! sketch.update_u64(1);
//! sketch.update_u64(2);
//! sketch.update_u64(2); // duplicate — not counted
//! assert_eq!(sketch.estimate(), 2.0);
//!
//! // The bytes below are the `apache-datasketches-theta-v1` Puffin blob payload.
//! let blob = sketch.serialize_compact().unwrap();
//! let parsed = iceberg_sketches::CompactThetaSketch::deserialize(&blob).unwrap();
//! assert_eq!(parsed.estimate(), 2.0);
//! ```
//!
//! ## Scope
//!
//! This crate is the byte-contract foundation only. Wiring it into `ComputeTableStats` (per-column
//! NDV over a scan, the Puffin `StatisticsFile` write, registration via `UpdateStatisticsAction`)
//! lives in the `iceberg` crate and is intentionally out of scope here.

#![deny(missing_docs)]

pub mod error;
pub mod hash;
pub mod theta;

pub use error::{SketchError, SketchResult};
pub use hash::{DEFAULT_UPDATE_SEED, compute_seed_hash, hash_bytes, hash_long, hash_longs};
pub use theta::{CompactThetaSketch, DEFAULT_LG_NOMINAL_LONGS, MAX_THETA, ThetaSketch};
