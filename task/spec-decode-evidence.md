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

# Spec decode evidence — live-probe transcripts and offset lists

Routed here from `crates/iceberg/src/spec/` during the 2026-08-26 comment sweep, per AGENTS.md
"Comments and prose": the behavioural CLAIM belongs in the doc comment, the EVIDENCE belongs
here. Each claim below is still stated at its site; this file is why it is believed.

## `Conversions.fromByteBuffer` on decimal — non-minimal encodings are accepted

Live probe against Java 1.10.0:

```text
fromByteBuffer(decimal(9,2), 00 00 04 D2) -> 12.34
fromByteBuffer(decimal(9,2), FF FF FB 2E) -> -12.34
fromByteBuffer(decimal(9,2), <20 bytes, 04 D2 in the tail>) -> 12.34
fromByteBuffer(decimal(9,2), <empty>) -> NumberFormatException: Zero length BigInteger
```

Java applies no length check and no minimality check. The fork matches on READ and stays strict
on WRITE, because `to_bytes` truncates the two's-complement buffer — refusing beats writing a
wrong bound. The blast radius is why it matters: `manifest::_serde::parse_bytes_entry`
propagates with `?`, so ONE padded bound makes a whole manifest unparsable and aborts every
scan.

Claim sites: `spec/values/datum.rs` (`Datum::try_from_bytes`), pinned by
`datum_decimal_byte_decode_accepts_non_minimal_encodings_like_java`.

## `ReassignIds` recursion — the measured stack figures

`id_reassigner.rs`'s depth bound is 128. The measurement behind it: 1152 KiB of stack FAILS,
1280 KiB PASSES, and an unbounded walk needs ~40 MiB. That is what makes the guard tests
non-vacuous rather than decorative. Java is unbounded here; the typed error is a deliberate
divergence.
