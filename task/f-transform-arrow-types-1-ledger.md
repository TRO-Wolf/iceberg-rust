# F-TRANSFORM-ARROW-TYPES-1 — every transform over every Arrow string/binary layout

Branch `fix/f-transform-arrow-types-1`. RePark defect: `CREATE TABLE … (b BINARY) PARTITIONED BY
(truncate(1, b))` then INSERT fails `FeatureUnsupported => Unsupported data type for truncate
transform: LargeBinary`. RePark/DataFusion hand `LargeBinary`/`LargeUtf8` and may hand
`BinaryView`/`Utf8View`; the transform layer handles only `Binary`/`Utf8` in places.

## Step 1 — MEASUREMENT (baseline: branch head before any edit)

Legend: ✓ accepted, ✗ rejected (`FeatureUnsupported` unless noted), ✗† rejected **correctly**
(Java `canTransform`/`getResultType` admits the type — rejection is parity, not a defect).

Canonical partition Arrow types (verified in `arrow/schema.rs`): `string→Utf8`,
`binary→LargeBinary`, `fixed(n)→FixedSizeBinary(n)`, `bucket→Int32`.

### 1a. Array path — `TransformFunction::transform(&ArrayRef)`

| transform | Binary | LargeBinary | BinaryView | FixedSizeBinary | Utf8 | LargeUtf8 | Utf8View |
|---|---|---|---|---|---|---|---|
| identity | ✓ (passthrough, out = Binary) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| void | ✓ (null array, input type) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| bucket[N] | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ |
| truncate[W] | ✓ (out = Binary) | **✗ defect** | ✗ | ✗† (Java excludes FIXED) | ✓ | ✓ | ✗ |
| year/month/day/hour | ✗† all — temporal-only per Java `canTransform` | | | | | | |

### 1b. Literal path — `TransformFunction::transform_literal(&Datum)`

| transform | binary datum | fixed datum | string datum |
|---|---|---|---|
| identity | ✓ | ✓ | ✓ |
| void | Ok(None) | Ok(None) | Ok(None) |
| bucket | ✓ | ✓ | ✓ |
| truncate | **✗ FeatureUnsupported defect** | ✗† | ✓ |
| year/month/day/hour | n/a — `can_transform` rejects non-temporal datums before the call | | |

### 1c. Partition-value / writer path — `PartitionValueCalculator::calculate_from_columns`

The calculator runs each transform, then `StructArray::try_new(expected_struct_fields, values)`
which requires each child `data_type()` to EQUAL the canonical partition field type. Two failure
modes today: the transform rejects the layout outright (1a), or the transform's output layout
differs from the canonical field type (struct type-mismatch error).

| transform | Binary | LargeBinary | BinaryView | FixedSizeBinary(n) | Utf8 | LargeUtf8 | Utf8View |
|---|---|---|---|---|---|---|---|
| identity | ✗ mismatch (Binary≠LargeBinary) | ✓ | ✗ mismatch | ✓ | ✓ | ✗ mismatch (LargeUtf8≠Utf8) | ✗ mismatch |
| void | ✗ mismatch | ✓ | ✗ mismatch | ✓ | ✓ | ✗ mismatch | ✗ mismatch |
| bucket | ✓ | ✓ | ✗ transform | ✓ | ✓ | ✓ | ✗ transform |
| truncate | ✗ mismatch (out Binary ≠ field LargeBinary) | **✗ transform — reported defect** | ✗ transform | ✗† | ✓ | ✗ mismatch | ✗ transform |

`RecordBatchPartitionSplitter` (computed mode, used by `PartitionExpr` in iceberg-datafusion)
feeds this same calculator; the `PartitionKey` literals are then re-read through
`arrow_struct_to_literal`, whose `String`/`Binary` accessors accept `Utf8`/`LargeUtf8` and
`Binary`/`LargeBinary` but NOT the view layouts — a gap for any precomputed `_partition` column
carrying view-typed leaves (the computed path's output is canonical, so it is not reachable
there).

### 1d. Projection / pruning path — `Transform::project` / `strict_project`

Both reach `transform_literal_result` on the predicate boundary datum.

- `truncate` + `Eq`/`In` on a **binary** datum: `can_transform` passes
  (`result_type(Binary)` is legal), then `transform_literal_result` returns
  `FeatureUnsupported`, which PROPAGATES — `Transform::project` errors, so inclusive projection
  on `WHERE b = X'0102'` fails the whole scan predicate rather than merely declining to prune.
  **Defect.**
- `truncate` + `StartsWith`/`NotStartsWith` on binary: the `len <= width` arms read the literal
  length directly (no transform call — OK); the `len > width` `NotStartsWith` arm calls
  `transform_literal_result` → same error on binary.
- `bucket` on binary/string/fixed datums: literal path complete — ✓.
- `identity`, `void`, temporals: no transform_literal call on the string/binary family, or
  Java-correct rejection — ✓/n/a.

### 1e. DataFusion write gate — `project_with_partition` → `field_is_write_compatible`

Primitive leaves require EXACT `DataType` equality, so only canonical layouts pass
(binary column: `LargeBinary`; string column: `Utf8`; `fixed(n)`: `FixedSizeBinary(n)`).
`Binary`, `BinaryView`, `LargeUtf8`, `Utf8View` inputs are rejected with "Input schema does not
match" before the transform is reached. The repo's own writer leaf contract
(`writer/write_defaults.rs::parquet_leaf_equivalent`, exercised by `relabel_column`/
`cast_leaf_encoding`) already treats {Utf8, LargeUtf8, Utf8View} and
{Binary, LargeBinary, BinaryView} as cast-equivalent families — the gate is stricter than the
write path behind it.

## Oracle — the run-24d binary-transform oracle (Spark 4.1.2 measured)

Rows (id, b): 1 `X''`, 2 `X'01'`, 3 `X'0102'`, 4 `X'010203'`, 5 `X'FF00FF'`, 6 NULL,
7 `X'E4B8AD'`; tables v2 and v3.

| transform | partition values per row 1..7 |
|---|---|
| truncate(1) | `''`, `01`, `01`, `01`, `ff`, NULL, `e4` |
| truncate(2) | `''`, `01`, `0102`, `0102`, `ff00`, NULL, `e4b8` |
| truncate(3) | `''`, `01`, `0102`, `010203`, `ff00ff`, NULL, `e4b8ad` |
| bucket(4) | 0, 3, 2, 0, 2, NULL, 2 |
| bucket(16) | 0, 11, 14, 4, 6, NULL, 2 |
| identity | each value itself |
| `WHERE b = X'0102'` | row id 3 on every table — pruning must not drop it |

Functions: `truncate(1, X'0102')` = `01`; `truncate(2, X'E4B8AD')` = `e4b8` (bytes, not chars);
`bucket(4, X'0102')` = 2; `bucket(16, X'')` = 0.

Per-row Java murmur3 (x86_32, seed 0, `& Integer.MAX_VALUE` then `% n`), independently
recomputed:

| b | murmur3 hash | bucket(4) | bucket(16) |
|---|---|---|---|
| `''` | 0 | 0 | 0 |
| `01` | -463810133 | 3 | 11 |
| `0102` | 1690789502 | 2 | 14 |
| `010203` | -2133732860 | 0 | 4 |
| `ff00ff` | 505334998 | 2 | 6 |
| `e4b8ad` | -555787182 | 2 | 2 |

Derived STRING oracle (Java semantics: `truncate` = Unicode code points, `bucket` = murmur3 of
UTF-8 bytes; truncate widths in code points):

| s | UTF-8 bytes | bucket(4) | bucket(16) | truncate(2) | truncate(4) |
|---|---|---|---|---|---|
| `""` | (empty) | 0 | 0 | `""` | `""` |
| `"iceberg"` | 69636562657267 | 1 | 9 | `"ic"` | `"iceb"` |
| `"中文字"` | e4b8ade69687e5ad97 | 2 | 10 | `"中文"` | `"中文字"` |
| `"a中b"` | 61e4b8ad62 | 3 | 11 | `"a中"` | `"a中b"` |
| `"🚀"` | f09f9a80 | 1 | 5 | `"🚀"` | `"🚀"` |
| `"abcdefg"` | 61626364656667 | 2 | 6 | `"ab"` | `"abcd"` |

`"a中b"` truncate(2) = `"a中"` is the code-point discriminator: byte-truncation would yield
`61e4` (a mojibake prefix), which Java never produces for strings.

## Java contract settled (apache/iceberg sources)

- `Truncate.canTransform` admits INTEGER, LONG, STRING, BINARY, DECIMAL — **not FIXED**: the
  `FixedSizeBinary` rejection in 1a is correct and stays.
- `Bucket.canTransform` admits FIXED: `bucket(FixedSizeBinary)` stays accepted (literal path
  already covers `fixed`; array path covers `FixedSizeBinary`).
- Binary truncate = byte length; string truncate = code-point count; bucket hashes raw bytes.
- Partition field type = source type for identity/truncate/void; Int32 for bucket.

## Fix design (chosen, to be proven)

1. `transform/truncate.rs`: add `LargeBinary`, `BinaryView`, `Utf8View` array arms —
   byte-truncate binary, code-point-truncate string, **output layout = input layout**; add the
   missing `PrimitiveLiteral::Binary` arm to `transform_literal` (byte truncation → `Datum::binary`).
2. `transform/bucket.rs`: add `BinaryView`, `Utf8View` arms → `Int32Array` (bucket output type
   is fixed by Java, not layout-preserving).
3. `arrow/partition_value_calculator.rs`: after each transform, `arrow_cast::cast` the output to
   the expected partition field type when the data types differ. Canonical LargeBinary input is
   a no-op check (hot path); `Binary`/`BinaryView`/`Utf8View`/identity/void non-canonical
   outputs pay ONE cast of the partition column only — unavoidable, because `StructArray`
   demands the canonical field type the rest of the write path (`PartitionKey`,
   `arrow_struct_to_literal`, manifest tuples) is built against.
4. `arrow/value.rs`: accept `StringViewArray`/`BinaryViewArray` in the `String`/`Binary`
   primitive accessors so `arrow_struct_to_literal` reads view-typed partition structs.
5. iceberg-datafusion `project.rs`: widen `data_type_is_write_compatible` to the same
   string/binary layout families as `parquet_leaf_equivalent` — the repo's own write contract —
   so BinaryView/Utf8View/Binary/LargeUtf8 inputs reach the transform instead of failing the
   schema gate.
6. Tests live in new `*_tests.rs` files (`bucket.rs` is AT its legacy size ceiling,
   `truncate.rs` ~13 lines under its default ceiling).

## RED-FIRST evidence (step 2)

Pins landed in `crates/iceberg/src/transform/f_transform_arrow_types_1_tests.rs` (14 tests:
oracle cells per layout for truncate/bucket/identity/void over Binary, LargeBinary, BinaryView,
Utf8, LargeUtf8, Utf8View; `FixedSizeBinary` truncate rejection + bucket acceptance;
transform_literal binary/fixed; `project`/`strict_project` binary-literal partition predicates)
and `crates/iceberg/src/arrow/f_transform_arrow_types_1_tests.rs` (11 tests: calculator
canonicalization per layout for truncate/identity/void/bucket over both families, splitter
partition keys per layout, `arrow_struct_to_literal` view-leaf reads, memory-catalog
fast-append write -> plan_files -> `to_arrow` for `b = X'0102'` on a `LargeBinary` batch, v2 + v3).

Baseline run on the UNFIXED tree (`cargo test -p iceberg --lib f_transform_arrow_types_1`):
**20 failed, 5 passed** (25 new tests, 3896 filtered out).

- GREEN (already-correct behavior, pinned): `identity_and_void_*_every_layout` x2,
  `truncate_fixed_size_binary_rejected_java_parity`, `bucket_fixed_size_binary_oracle`,
  `bucket_literal_binary_and_fixed`.
- RED (defects): every `LargeBinary`/`BinaryView`/`Utf8View` truncate arm; `BinaryView`/`Utf8View`
  bucket arms; `transform_literal` on a binary datum; `project`/`strict_project` on binary
  literals (propagates `FeatureUnsupported`, failing the whole predicate); every non-canonical
  calculator output (identity/void/truncate on `Binary`, `BinaryView`, `LargeUtf8`, `Utf8View`);
  `arrow_struct_to_literal` on view leaves; both end-to-end write-scan pins
  (`split batch: FeatureUnsupported ... LargeBinary` — the reported RePark error verbatim).

## Mutation validation (step 4)

Twelve surgical mutations across the changed code; each was compiled fresh against the pin suite
(`cargo test -p iceberg --lib f_transform_arrow_types_1`, or
`cargo test -p iceberg-datafusion --lib string_binary_layout_families` for M9) and every one was
KILLED — zero survivors. Post-restore re-run: 25/25 pins green, comparator pin green.

| mutation | killed by |
|---|---|
| M1 truncate.rs — drop `BinaryView` arm | binary oracle + calculator + splitter + e2e (4 failed) |
| M2 truncate.rs — drop `Utf8View` arm | string oracle + calculator (2 failed) |
| M3 truncate.rs — drop `LargeBinary` arm (the reported defect) | oracle, literal-projection, calculator, splitter, e2e (6 failed) |
| M4 truncate.rs — drop literal `Binary` arm | literal pin, `project`/`strict_project` pins, e2e pruning (7 failed) |
| M5 bucket.rs — drop `BinaryView` arm | binary bucket oracle + calculator (2 failed) |
| M6 bucket.rs — drop `Utf8View` arm | string bucket oracle + calculator (2 failed) |
| M7 calculator — skip canonical cast | every non-canonical calculator pin + splitter + e2e (6 failed) |
| M8 value.rs — drop `BinaryViewArray` accessor | `arrow_struct_to_literal` view pin (1 failed) |
| M9 write gate — strict leaf equality | `field_write_compatibility_string_binary_layout_families` (1 failed) |
| M10 `truncate_binary` returns input untruncated | binary oracle, literal, calculator, splitter, e2e (10 failed) |
| M11 `truncate_str` truncates bytes not code points | string oracle `"中文字"`/`"a中b"` code-point rows (2 failed) |
| M12 truncate `BinaryView` arm emits `Binary` layout | layout-preservation assert in the binary oracle pin (1 failed) |

## PROVEN / OPEN summary

PROVEN:

- `truncate[W]` accepts `Binary`, `LargeBinary`, `BinaryView`, `Utf8`, `LargeUtf8`, `Utf8View` in
  the array path and binary datums in the literal path; output layout = input layout; binary =
  bytes, string = Unicode code points (M11 pins the discriminator rows).
- `bucket[N]` adds `BinaryView`/`Utf8View`; Java murmur3 oracle cells hold on every layout;
  `FixedSizeBinary` stays accepted for bucket and rejected for truncate (Java `canTransform`).
- `identity`/`void` keep array passthrough / same-type nulls on every layout.
- `PartitionValueCalculator` canonicalizes any source-layout output to the partition field's
  Arrow type via `arrow_cast::cast`; splitter partition keys and `arrow_struct_to_literal` round
  trip on view leaves.
- `Transform::project`/`strict_project` on binary literals no longer error — eq, in-set, range
  and `NotStartsWith` projections produce partition predicates.
- iceberg-datafusion `field_is_write_compatible` accepts the string/binary layout families
  (same leaf equivalence as `parquet_leaf_equivalent`); FixedSizeBinary still exact-width only.
- End-to-end: memory-catalog fast append of a `LargeBinary` batch into a `truncate(1,b)`
  partitioned table writes 5 partitioned data files, and `b = X'0102'` prunes to exactly the
  file holding row 3 and returns it — on format v2 and v3.
- Gates: `cargo fmt --all -- --check`, `cargo clippy -p iceberg --all-targets -D warnings`,
  `cargo clippy -p iceberg-datafusion --all-targets -D warnings`, `check_rust_file_size`,
  `check_comment_blocks`, `check_agent_artifacts`, `check_matrix_anchors`,
  `comment_ban.py hits=0`, `typos .` — all green; targeted suites 25/25 + 112 transform + 416
  arrow + 214 physical_plan green.

OPEN / noted:

- `truncate` on non-canonical layouts returns the INPUT layout (not canonical `LargeBinary`/
  `Utf8`); canonicalization is the calculator's single cast of the partition column. Deliberate:
  the transform contract is layout-preserving; the partition schema is canonical. Pinned by the
  calculator tests.
- The cast copies the partition column only when the input layout differs from canonical — a
  `LargeBinary` table batch is a no-op check; `Binary`/`BinaryView`/`Utf8View` inputs pay one
  cast, unavoidable because `StructArray` requires the canonical field type the rest of the
  write path is built against.
- RePark/Spark end-to-end was not re-run (no engine, no docker in this lane); the DataFusion
  write path is proven at the comparator + memory-catalog level.
