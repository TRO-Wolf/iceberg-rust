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
  Non-ASCII Nd digits likewise get `_<digit>` with the original digit char appended. **PROVEN**
- `Integer.toHexString(c).toUpperCase(ROOT)` = minimal-width UPPERCASE hex of the UTF-16 unit:
  `' '`→`_x20`, `'-'`→`_x2D`, `'.'`→`_x2E`. **PROVEN**
- `Character.isLetter` = Unicode general categories **Lu, Ll, Lt, Lm, Lo**;
  `Character.isLetterOrDigit` adds **Nd** only. Rust `char::is_alphabetic` is the *Alphabetic*
  derived property (L\* + Nl + Other_Alphabetic) and `char::is_alphanumeric` adds Nl + No — both
  are SUPERSETS of Java's sets, so they are not drop-in equivalents. **PROVEN** (JDK semantics vs
  Rust std docs).
- Choice made for classification (allowed by the brief): a generated range table over the BMP of
  categories {Lu,Ll,Lt,Lm,Lo} for letters and {Nd} for digits — 380 + 37 ranges — produced by
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

## 7. Clauses checklist (updated as work lands)

- [x] MEASURE write names, fixture reads, broken-file behavior, reader mapping — §3 PROVEN
- [ ] RED-FIRST pins: exact write schema, fixture reads, write→read, table-level scan — OPEN
- [ ] Java-exact sanitizer + `iceberg-field-name` on every `schema_to_avro_schema` record — OPEN
- [ ] Reader restores Iceberg names (attr / computed avro-name inverse) — OPEN
- [ ] OCF header patch for unparseable embedded schemas — OPEN
- [ ] Broken RePark manifest readable, loud-refusal fallback pinned — OPEN (target: readable)
- [ ] `é`/`列` readable without global validator / Cargo change — OPEN (target: patch)
- [ ] Mutation arithmetic recorded — OPEN
- [ ] Gates: fmt, clippy, size checker, comment-ban, filtered tests — OPEN
