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

PENDING — pins added in `transform/f_transform_arrow_types_1_tests.rs` and
`arrow/f_transform_arrow_types_1_tests.rs`; baseline failure counts recorded below after the
first run.

## Mutation validation (step 4)

PENDING.

## PROVEN / OPEN summary

PENDING.
