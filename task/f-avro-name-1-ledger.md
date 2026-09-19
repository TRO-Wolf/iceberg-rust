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

# Ledger — F-AVRO-NAME-1 (RePark IPI-52): partition field names that are not valid Avro names

**Ledger id:** `F-AVRO-NAME-1-2026-09-19`
**Branch:** `fix/f-avro-name-1` (cut off fork main `43fcd243`)
**Scope:** devin-worker brief "fork F-AVRO-NAME-1 (RePark slate IPI-52)"
**Oracle:** the run-24d Spark avro-names oracle — Spark 4.1.2 + Iceberg 1.11.0, Hadoop-catalog
warehouses under the oracle directory, truth table `avro_names_truth.json`; Java source
`core/src/main/java/org/apache/iceberg/avro/AvroSchemaUtil.java` at tag `apache-iceberg-1.11.0`.

Status legend: each claim ends **PROVEN** (measured on this tree / cited code) or **OPEN**.

## 1. The defect (measured on this branch)

`SchemaToAvroSchema::field` (`crates/iceberg/src/avro/schema.rs:95-96`) puts the raw Iceberg field
name into every generated Avro `RecordField` (`name: field.name.clone()`) and stamps only
`field-id`. A partition field named `my col` therefore lands verbatim in the manifest's
`data_file.partition` record (`r102`). apache-avro 0.21 validates parsed field names against
`^[A-Za-z_][A-Za-z0-9_]*$` (`validator.rs` `SpecificationValidator::validate`,
`schema.rs:673 validate_record_field_name`), so the write "succeeds" and every later read dies in
`AvroReader::new` with `Invalid field name my col` — the exact RePark symptom. **PROVEN**

## 2. Java's exact algorithm (`AvroSchemaUtil`, 1.11.0 — fetched and re-read)

```java
public static String makeCompatibleName(String name) {
    if (!validAvroName(name)) return sanitize(name);
    return name;
}
static boolean validAvroName(String name) {
    char first = name.charAt(0);
    if (!(Character.isLetter(first) || first == '_')) return false;
    for (i = 1..length) if (!(Character.isLetterOrDigit(c) || c == '_')) return false;
    return true;
}
static String sanitize(String name) {
    // first char: isLetter or '_' → kept, else sanitize(first)
    // later chars: isLetterOrDigit or '_' → kept, else sanitize(c)
}
static String sanitize(char character) {
    if (Character.isDigit(character)) return "_" + character;
    return "_x" + Integer.toHexString(character).toUpperCase(Locale.ROOT);
}
```

Points pinned by this reading:

- Iteration is over Java `char` = **UTF-16 code units**, not Unicode scalar values. A supplementary
  character arrives as a surrogate pair and each half is a non-letter non-digit `char`, so `c😀`
  sanitizes to `c_xD83D_xDE00` — exactly the oracle. Rust port iterates `str::encode_utf16()`.
  **PROVEN** (oracle `emoji_*` rows + Java source).
- A leading digit is NOT a letter → `sanitize('1')` = `_1` → `1st` → `_1st` (oracle-confirmed).
  Non-ASCII decimal digits (category N-d) likewise get `_<digit>` with the original digit char appended. **PROVEN**
- `Integer.toHexString(c).toUpperCase(ROOT)` = minimal-width UPPERCASE hex of the UTF-16 unit:
  `' '`→`_x20`, `'-'`→`_x2D`, `'.'`→`_x2E`. **PROVEN**
- `Character.isLetter` = Unicode general categories **Lu, Ll, Lt, Lm, Lo**;
  `Character.isLetterOrDigit` adds decimal digits (category N-d) only. Rust `char::is_alphabetic` is the *Alphabetic*
  derived property (L\* + Nl + Other_Alphabetic) and `char::is_alphanumeric` adds Nl + No — both
  are SUPERSETS of Java's sets, so they are not drop-in equivalents. **PROVEN** (JDK semantics vs
  Rust std docs).
- Choice made for classification (allowed by the brief): a generated range table over the BMP of
  categories {Lu,Ll,Lt,Lm,Lo} for letters and {N-d} for digits — 380 + 37 ranges — produced by
  Python `unicodedata` (Unicode **15.0.0**). Skew note, pinned: JDK 17 = Unicode 13.0, JDK 21 =
  Unicode 15.0; a BMP codepoint that gained a letter category between 13 and 15 would classify
  differently. Supplementary code points never matter: we iterate UTF-16 units and every surrogate
  is non-letter/non-digit → `_xXXXX`, matching Java exactly. **PROVEN** (table generated; skew
  OPEN-accepted, recorded).
- On write, Java stamps `"iceberg-field-name": "<original>"` on every renamed field beside
  `"field-id"` (oracle `manifest_partition_avro` JSON shows both props; unchanged fields carry
  neither attr beyond `field-id`). **PROVEN**

## 3. Measured table — fork before this change

Run via a throwaway test module (`avro::f_avro_name_1_measure`, since deleted) driving
`ManifestWriterBuilder` + `Manifest::parse_avro` over the 20 Spark `-m0.avro` fixtures and the
RePark manifest copied to `crates/iceberg/testdata/avro_names/`.

| name | fork writes (r102 field) | fork reads Spark file | Spark reads fork file (reasoned) |
|---|---|---|---|
| `my col` | raw `my col`, no attr | silent-wrong: v2 → partition `[None]`; v3 → ERR `field my_x20col is not exist` | Java `AvroReader` reads embedded schema `my col`?? — Java avro parses lax names fine, but field-id projection reads the partition record by `field-id`, so the raw name does not matter for the value; Spark can read the row. However the fork's OWN reader cannot (invalid name for apache-avro), and Java's strict-parse paths would reject the embedded schema. |
| `my col_bucket` (bucket) | raw | same: v2 `[None]`, v3 ERR | same reasoning |
| `1st` | raw | v2 `[None]`, v3 ERR | same |
| `a-b` | raw | v2 `[None]`, v3 ERR | same |
| `a-b_trunc` (truncate) | raw | v2 `[None]`, v3 ERR | same |
| `a.b` | raw | v2 `[None]`, v3 ERR | same |
| `c😀` | raw | v2 `[None]`, v3 ERR | same |
| `é` | raw `é`, no attr | ERR `Invalid field name é` (header parse) | Java reads (lax parse); fork cannot |
| `列` | raw `列`, no attr | ERR `Invalid field name 列` | same |
| `ok_col` | raw `ok_col`, no attr | OK — `x` | OK both ways |
| RePark `my col` file | n/a | ERR `Invalid field name my col` (the defect) | n/a |

**PROVEN** — measured output recorded 2026-09-19.

### 3a. Why v2 silently misreads and v3 fails loudly — `Schema` equality ignores field names

`AvroReader::with_schema` sets `should_resolve_schema = writer_schema() != schema`
(`apache-avro-0.21.0/src/reader.rs:365`). `Schema::eq` is `schema_equality::compare_schemata`,
whose default comparator is `StructFieldEq{include_attributes:false}` — and
`StructFieldEq::compare_fields` (`schema_equality.rs:215-221`) compares each record field's
**schema only, never its name**. Record field-name differences are invisible to the equality, so:

- Spark **v3** manifests: writer schema (field `my_x20col: ["null","string"]`) is
  positionally-type-equal to the fork's reader schema (`my col: ["null","string"]`) →
  `should_resolve_schema = false` → decoded `Value::Record` keeps the WRITER key `my_x20col` →
  `from_value` → `RawLiteralEnum::Record` → `try_into` `field_by_name("my_x20col")` → loud
  `field my_x20col is not exist`. **PROVEN**
- Spark **v2** manifests: the v2 `data_file` record carries fewer fields than the v2/v3-unified
  `manifest_schema_v2` reader schema → not equal → `should_resolve_schema = true` →
  `resolve_record` removes `items["my col"]` (absent) → uses the field default `null` →
  partition deserializes to `[None]` — **silent corruption**. **PROVEN**
- Files with apache-avro-invalid embedded field names (`é`, `列`, RePark `my col`) never reach
  either branch: `AvroReader::new` fails parsing `avro.schema`. **PROVEN**

So the reader maps partition record fields by **name** today (resolve by reader field name, then
`field_by_name` in `RawLiteralEnum::try_into` at `spec/values/serde.rs:754-761`), never by
`field-id`. `field-id` exists only as inert schema props on this path (the arrow avro data-file
reader is the field-id-resolved path, `arrow/avro_reader.rs:33-41`). **PROVEN**

## 4. Reader-side design consequence (pin for the fix)

Because `compare_schemata` is name-blind, a reader schema carrying the sanitized name
(`my_x20col`) is "equal" to any writer schema with the same positional types — resolution is
skipped and decoded values arrive with writer field names. The load-bearing requirement is
therefore at the `RawLiteral`/`from_value` boundary:

- **Write**: `schema_to_avro_schema` emits `makeCompatibleName(field.name)` plus
  `"iceberg-field-name"` when changed; `RawLiteralEnum::try_from` emits the same avro name as the
  record key so `to_value().resolve(schema)` matches by name.
- **Read**: decoded writer-named record keys map back to the Iceberg name via
  `iceberg-field-name` (writer schema) / `field-by-name-or-avro-name` in
  `RawLiteralEnum::try_into`; the `field-id` stays the identity anchor for schema conversion
  (`avro_schema_to_schema` restores `iceberg-field-name` into `NestedField.name`, as Java
  `SchemaToType` does).
- **Unreadable embedded schemas** (`é`, `列` raw, RePark `my col` raw): `AvroReader::new` fails
  before any value logic. Fix = rewrite the OCF header's `avro.schema` JSON field names to
  ASCII-valid ones (adding `iceberg-field-name` when absent), preserving `field-id` and all bytes
  after the sync marker verbatim — no global validator, no dependency change. Then the same
  name-restore path yields the original Iceberg name. This makes the RePark-broken table
  **readable**, the preferred outcome. **PROVEN** (design; implementation pins below).

## 5. Spark-reading-fork-files reasoning (the third column)

Java resolves manifest `data_file.partition` records by `field-id` projection
(`AvroSchemaUtil`/`ManifestReader` build reader structs keyed on field ids; the
`iceberg-field-name` prop restores names). For a fork-written file after this change:

- Sanitized name + `iceberg-field-name` present → Java restores the original name. **PROVEN** by
  Java source + oracle shape.
- `é`/`列` raw → Java parses lax names, restores name = field name. **PROVEN** by oracle
  (Spark reads its own such tables).
- A fork file written BEFORE this change (raw `my col`) is the RePark-broken state — Java-side
  readable in principle but was never produced by Java; the fork must not produce it again.

## 6. Non-ASCII decision (step-4 ruling, pre-committed by measurement)

Measured: apache-avro 0.21 default validator rejects `é`/`列` embedded field names
(`Invalid field name é`) — the fork cannot read raw non-ASCII names TODAY. The OCF-header patch
(§4) makes them readable WITHOUT a Cargo change and WITHOUT the process-global
`set_*_validator` — the global validator is therefore unnecessary and remains uninstalled.
**PROVEN** (measured; pin tests follow).

## 7. Implementation (what landed)

- `crates/iceberg/src/avro/name.rs` (new): `java_avro_name` ports `makeCompatibleName`/`sanitize`
  over `str::encode_utf16()` — digit → `_<digit>`, other invalid char → `_x` + uppercase hex of the
  UTF-16 code unit; letter/digit classification via generated BMP range tables {Lu,Ll,Lt,Lm,Lo} and
  {N-d}. `avro_field_name` stamps `iceberg-field-name` beside `field-id` only when the name changes.
  `iceberg_field_name` reads it back. `strict_avro_name`/`strictify_avro_field_names` map any
  apache-avro-invalid name to a deterministic ASCII-valid one (used on reader schemas and on OCF
  header repair). `repair_avro_container` rewrites the `avro.schema` JSON inside an OCF header —
  sanitizing every invalid record/field name recursively and stamping `iceberg-field-name` with the
  original — with a hand-rolled varint metadata codec, bytes after the sync marker untouched.
- `avro/schema.rs`: `field` and map key/value visitors call `avro_field_name`;
  `AvroSchemaToSchema::record` restores `NestedField.name` via `iceberg_field_name`; record naming
  (`r<field-id>`) goes through `set_record_name` in `name.rs`.
- `spec/values/serde.rs`: `RawLiteral` record serialization emits sanitized Avro keys (matching
  what `to_value().resolve()` and `encode` look up by name); read side accepts either the Iceberg
  name or its `java_avro_name` image when building the partition `Struct`.
- `spec/manifest/mod.rs`: `try_from_avro_bytes_with_schema_fallback` runs
  `repair_avro_container` before OCF open; entry decode delegated to
  `entry::manifest_entries_from_avro`, which builds the versioned reader schema, runs
  `strictify_avro_field_names` on it, then `AvroReader::with_schema` + serde `try_into` as before.
- `spec/manifest/writer.rs`, `spec/manifest/data_file.rs`: record values written through the
  Avro pipeline carry sanitized keys; `data_file` reads buffer + repair the container.
- `arrow/avro_reader.rs`: data-file reads repair the container before `AvroReader::new` —
  Spark-written data files with raw `é`/`列` names become readable, no global validator.

## 8. Verification (measured on this tree, HEAD = fix slice)

- Pin suite `avro::f_avro_name_1_tests` — 5/5 green:
  `sanitizes_every_record_from_schema_to_avro_schema` (every table row incl. map/nested records),
  `writes_java_avro_field_names` (exact `my_x20col`/`_1st`/`a_x2Db`/`a_x2Eb`/`c_xD83D_xDE00` +
  `iceberg-field-name` presence/absence per oracle), `write_then_read_round_trip` (all names),
  `reads_spark_manifest_partition_values` (all 20 Spark v2/v3 fixtures + RePark broken file →
  partition value `x` under the Iceberg name), `table_scan_filters_on_spaced_partition_column`
  (memory catalog, append, scan with `my col` filter → row `1,"x"`). **PROVEN**
- Broken RePark manifest (`repark_broken_space_m0.avro`, raw `my col` in embedded schema):
  **readable** — OCF header repair sanitizes the embedded schema; partition value returns `x`.
  The preferred outcome per the brief; loud refusal unnecessary. **PROVEN**
- `é`/`列`: Spark files with raw non-ASCII names read via container repair; fork-written files
  keep `é`/`列` raw (Java-letters → valid under `java_avro_name`) and carry no attr, as the
  oracle. No global validator installed; no Cargo change. **PROVEN**
- Regression: `avro::` 32/32, `manifest` 200/200, `spec::values` 145/145 lib tests green.
  **PROVEN**
- Mutation: `git stash` of the whole fix slice (tests stay committed at `fed67a75`) → all 5 pins
  red with the pre-fix signatures (raw names written, v2 Spark fixtures silently `[None]`, v3
  fixtures `field my_x20col is not exist`, round trip + scan `Invalid field name my col`).
  Restored via `git stash pop`; pins re-verified green after the subsequent code-motion refactor.
  **PROVEN**
- Gates: `cargo fmt --all -- --check` clean; `cargo clippy -p iceberg --all-targets --
  -D warnings` clean; `scripts/check_rust_file_size.sh` 524 files clean (ceilings lowered to new
  sizes: `avro_reader.rs` 1255, `schema.rs` 2092, `manifest/mod.rs` 1237, `writer.rs` 1056);
  `comment_ban.py /tmp/pd-fork origin/main HEAD` → `comment-ban hits=0`;
  `check_comment_blocks.sh` OK. **PROVEN**

## 9. Clauses checklist

- [x] MEASURE write names, fixture reads, broken-file behavior, reader mapping — §3 PROVEN
- [x] RED-FIRST pins: exact write schema, fixture reads, write→read, table-level scan — `fed67a75` PROVEN
- [x] Java-exact sanitizer + `iceberg-field-name` on every `schema_to_avro_schema` record — §7 PROVEN
- [x] Reader restores Iceberg names (attr / computed avro-name inverse) — §7 PROVEN
- [x] OCF header patch for unparsable embedded schemas — §7 PROVEN
- [x] Broken RePark manifest readable — §8 PROVEN (readable, not loud)
- [x] `é`/`列` readable without global validator / Cargo change — §8 PROVEN
- [x] Mutation arithmetic recorded — §8 PROVEN
- [x] Gates: fmt, clippy, size checker, comment-ban, filtered tests — §8 PROVEN

# Round 2 — critic follow-up (fork #308)

Base: round-1 head `4d18cc8a`. Critics: `/tmp/oc-worker/pd-fork/rv-avro-logic-out.json`,
`/tmp/oc-worker/pd-fork/rv-avro-perf-out.json`. Nine findings, all addressed red-first.

## R2-1. Findings → fix → pin

- **L-001 (P1) CLOSED** — Avro data-file writer renamed schema fields but not record VALUE keys;
  `Value::resolve` silently dropped optional values to null and failed required ones. Fix:
  `RawLiteralEnum::try_from` in `spec/values/serde.rs` sanitizes record keys at construction via
  `sanitize_avro_value_names` (recursive: records, maps, arrays, unions) — every writer path
  (manifest writer, data-file writer, `write_data_files_to_avro`) emits aligned keys. Pin:
  `avro_data_file_round_trip_sanitizes_value_keys` — write→read for `my col`, `a-b`, `1st`,
  `c😀` over optional+required fields; red pre-fix with `Missing field in record: "_1st"`.
  Commits: test+fix in the L-001 slice (see commit list).
- **L-002 (P1) CLOSED** — collisions. (1) Write path: `avro_record_schema` under
  `AvroNameCollision::Fail` scans for duplicate Avro names and returns typed `DataInvalid`
  naming both Iceberg fields and the colliding name — mirrors Java `setFields` `Duplicate field`.
  (2) Read/binding: `StructType::field_by_avro_name` consults a `OnceLock`-cached map built by
  `uniquified_avro_names` — a field keeps its name when it already owns the canonical Avro name;
  otherwise it is assigned `repair_target_name` + numeric suffix. A literal-name hit can never
  shadow the field whose sanitized name/`iceberg-field-name` owns the slot (repaired names bind
  before literal names — stated choice: binding is by repaired-name precedence, not field-id,
  because `apache-avro` `resolve` matches by name and the decoded `Value::Record` carries no ids).
  Pins: `colliding_avro_field_names_fail_at_schema_build` (`a b`/`a_x20b`, `1a`/`_1a` →
  `DataInvalid`), `unique_avro_names_bind_before_literal_names` (serde binding),
  `repaired_colliding_names_bind_distinctly` (all four pairs through repair+read).
- **L-003 (P2) CLOSED** — repair collapsed Java-legal pairs (`é`/`_xE9`, `列`/`_x5217`) into
  duplicates. Fix: OCF repair and reader-schema strictify share `uniquified_avro_names`, which
  keeps each name's canonical owner and suffixes the repair (`_xE9_1`, `a_x20b_1`, …);
  `iceberg-field-name` always preserves the original. Pin:
  `repaired_colliding_names_bind_distinctly` — all four pairs read back distinctly.
- **L-004 (P2) CLOSED** — coverage pins added; implementation already correct. Pin:
  `ocf_metadata_multi_block_and_negative_blocks_decode` (two-positive, all-negative, and mixed
  block counts decode identically through slice parse, streaming header, repair, and full
  read; a broken-schema multi-block header repairs and reads), plus
  `ocf_repair_passes_through_valid_containers_byte_identical` (Null/Deflate/Zstandard →
  `Cow::Borrowed`, byte-equal) and
  `ocf_repair_fixes_schema_names_inside_snappy_and_zstd_containers` (Zstandard container
  round-trips repaired; a snappy-flagged container — snappy codec feature is OFF in this
  workspace, so its body cannot be decoded — repairs its header and preserves `avro.codec`).
- **R-01 (P1) CLOSED** — `repair_avro_container` parsed schema JSON unconditionally. Fix:
  `schema_names_need_repair` scans `"name"` values in the schema bytes with the apache-avro
  name rule; only when a name needs repair does the header parse JSON (`patch_record_field`
  allocates only on rename). Pin: `ocf_repair_json_parses_only_when_schema_names_need_it` —
  thread-local `ocf_json_parse_count` probe: 0 parses for valid containers, 1 for broken.
- **R-02 (P1) CLOSED** — repair copied whole files; `read_data_files_from_avro` lost `R: Read`.
  Fix: `ocf_repaired_stream` repairs only the header (`read_ocf_header` streams OCF metadata
  incl. negative block counts) and returns `Chain<Cursor<header>, BufReader<&mut R>>` — body
  never copied; signature stays streaming. Pin:
  `data_files_avro_reader_streams_without_read_to_end` (a reader whose `read_to_end` errors).
- **R-03 (P2) CLOSED** — per-record re-sanitization in serde binding. Fix:
  `StructType.avro_lookup: OnceLock<Arc<HashMap<String, usize>>>` built once via
  `uniquified_avro_names`; `RawLiteral` record conversion calls `field_by_avro_name`. No
  sanitizer call per record. Pin: `unique_avro_names_bind_before_literal_names`.
- **R-04 (P3) CLOSED** — `java_avro_name` now returns `Cow::Borrowed` immediately when
  `is_apache_avro_name(name)` (covers ASCII fast path and all already-valid names).
- **R-05 (P2) CLOSED** — `strictify_avro_field_names` returns a change flag through recursion;
  unions are rebuilt only when a nested variant renamed.

## R2-2. Structural notes

- `avro/ocf.rs` (new) holds all OCF machinery (header parse, repair, streaming reader);
  `avro/name.rs` keeps name logic only. `avro/schema_build.rs` and `spec/datatypes_de.rs` were
  split out for file-size ceilings (moved code's comments deleted per RULE 0).
- The `OCF_JSON_PARSES` probe is `thread_local!` — parallel test threads must not race the
  counter (caught when the L-004 suite first ran the counter test concurrently).
- `apache-avro`'s `snappy` feature is not enabled in this workspace (`apache-avro = "0.21"`,
  features `["zstandard"]`; Cargo edits banned). Snappy coverage is a hand-flagged
  `avro.codec: snappy` container through repair — codec metadata preserved, schema repaired.
  Record-level snappy decode cannot run without the feature.
- `StructType` grew past `clippy::large_enum_variant` via `SchemaOp`; the three lookup caches
  sit behind `Arc<HashMap>` (8 bytes, keeps laziness, `box_collection`-clean).

## R2-3. Mutation (revert → red)

Single-batch surgical reverts (value-key sanitize no-op, dup check skipped,
`uniquified_avro_names` verbatim, fast-path scan removed, `read_to_end` restore, negative
block count rejected, literal-name binding): **12 of 15 pins red** —
`avro_data_file_round_trip_sanitizes_value_keys`, `write_then_read_round_trip`,
`table_scan_filters_on_spaced_partition_column` (L-001),
`colliding_avro_field_names_fail_at_schema_build` (L-002 write),
`unique_avro_names_bind_before_literal_names` + `repaired_colliding_names_bind_distinctly`
(L-002 bind / L-003), `ocf_repair_json_parses_only_when_schema_names_need_it` (R-01),
`data_files_avro_reader_streams_without_read_to_end` (R-02),
`ocf_metadata_multi_block_and_negative_blocks_decode` +
`data_files_avro_reader_repairs_schema_names_while_streaming` +
`ocf_repair_fixes_schema_names_inside_snappy_and_zstd_containers` (L-004),
`reads_spark_manifest_partition_values` (regression). `git checkout` restored; all 15 green.
**PROVEN**

## R2-4. Gates (HEAD = `12170c8f`)

- `cargo test -p iceberg --lib f_avro_name_1` — 15/15 green. **PROVEN**
- `cargo test -p iceberg --lib avro` 129, `manifest` 200 (3 ignored), `values` 173,
  `writer` 179 (1 ignored). **PROVEN**
- `cargo fmt --all -- --check` clean; `cargo clippy -p iceberg --all-targets -- -D warnings`
  clean; `scripts/check_rust_file_size.py` 529 files clean (test module split:
  `f_avro_name_1_ocf_tests.rs`); `scripts/check_comment_blocks.sh` OK. **PROVEN**
- No push, no PR, no `gh`, no Cargo.toml/Cargo.lock change. **PROVEN**

## R2-5. Commits (round 2, oldest → newest)

- `4e4c92c9` test: pin Avro data-file value-key sanitization (L-001)
- `7a393e00` fix: sanitize Avro data-file record value keys (L-001)
- `1502c47e` test: pin OCF fast path and streaming repair (R-01, R-02)
- `35fb3014` fix: OCF fast path and header-only streaming repair (R-01, R-02)
- `1d3da9e6` test: pin collision loud-fail and distinct repair (L-002, L-003)
- `bd4e1ef3` fix: collision-safe names, unique repair, cached binding
  (L-002, L-003, R-03, R-04, R-05)
- `077ab446` test: pin OCF container shapes (L-004)
- `12170c8f` test: split OCF pins into sibling test module (size ceiling)
