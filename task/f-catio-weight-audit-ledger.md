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

# F-CATIO-WEIGHT audit ledger

**Date:** 2026-09-07
**State:** `REVIEWED_DRAFT_PR_READY`; Docker integration is blocked by local environment
accessibility.
**Branch:** `codex/fork-cache-weight-audit` at `85db42f285703682629b3b53bd1ebcd3091c6bc5`.
**Scope:** `ObjectCache` accounting, read-only direct graph helpers, focused tests, and this ledger.
**Restrictions:** dependency, lockfile, workflow, AWS, RePark pin, commit, push, and PR changes.

## Self Logic Review

```yaml
self_logic_review:
  id: SLR-f-catio-weight-audit
  action: File the cache ownership and accounting contract before production work.
  preconditions:
    - The worktree is isolated at the required origin/main commit.
    - No active branch or open pull request owns this accounting change.
    - The RePark measurements and red-when-fixed pin are read from their source documents.
    - The Moka capacity, weight, maintenance, and admission behavior is verified in source.
  postconditions:
    - Every retained object class and every exclusion is enumerated.
    - Overflow, concurrency, maintenance, admission, and eviction behavior is explicit.
    - The candidate charter makes no process-RSS guarantee.
    - Production work starts only after owner review.
  invariant_check: No production source, dependency, lockfile, workflow, AWS resource, or RePark pin changes.
  verdict: PROCEED
```

## Evidence

- RePark measured about 15 MB and 59 MB read-phase resident growth at 2,000 and 8,000 tables.
  The cache charged about 2 MB and 8 MB. The observed order is about 7.5 times the charge.
- At the default 32 MiB charge boundary, 32,768 tables retained about 265–278 MB above the
  cache-off process. The observed order is about eight times the charge.
- The measured manifest and manifest-list files were 3,466 and 1,604 bytes. The read-phase resident
  growth was about 7.5 KB per table in the measured process.
- `test_a_budget_sized_to_the_charged_weight_retains_every_table` gives 256 tables a 280,000-byte
  budget. The current 1,024-byte combined charge keeps every table. Corrected charges must evict at
  least the cold entry and make this old-behavior pin fail.
- Process peak RSS varied by about 20 MB between runs. RSS is evidence of the defect, not a stable
  numeric regression oracle.

## Closed audit propositions

| ID | Proposition | Verdict | Proof |
|---|---|---|---|
| C-001 | The baseline manifest charge was `max(entries, 1) * 768`, saturated in `u64`, then clamped to `u32`. | PROVEN | Baseline `estimate_manifest_weight` and `clamp_cache_weight`. |
| C-002 | The baseline manifest-list charge was `max(entries, 1) * 256` with the same saturation and clamp. | PROVEN | Baseline `estimate_manifest_list_weight` and `clamp_cache_weight`. |
| C-003 | The baseline weigher charged no cache key bytes. | PROVEN | The baseline Moka weigher closure discarded its key argument. |
| C-004 | The cache stores raw manifests and returns a new contextual manifest for each `get_manifest` call. | PROVEN | `get_manifest` deep-clones every entry before applying manifest-list context. |
| C-005 | The cache returns a clone of the cached manifest-list `Arc`. | PROVEN | `get_manifest_list` returns the stored `Arc`; a caller can retain it after eviction. |
| C-006 | Moka applies capacity to the settled sum of `u32` policy weights. | PROVEN | Moka counters use `u64`; `has_enough_capacity` compares the counter plus candidate weight with `max_capacity`. |
| C-007 | An item whose policy weight exceeds capacity is rejected during maintenance. | PROVEN | Moka `handle_upsert` removes an oversized candidate with `RemovalCause::Size`. |
| C-008 | Weighted size does not choose victims. | PROVEN | Moka admission compares candidate and victim frequency; weight only sets how much victim weight is needed. |
| C-009 | Cache counters and eviction are asynchronous. | PROVEN | Moka documents approximate counters and requires `run_pending_tasks` for a settled assertion. |
| C-010 | Same-key initialization is coalesced, while distinct keys can fetch and parse concurrently. | PROVEN | `entry_by_ref(...).or_try_insert_with(...)` serializes one key only; the object cache has no global load semaphore. |
| C-011 | A cache budget cannot bound process RSS or every live parsed object. | PROVEN | The enumerated exclusions are outside Moka weight or can outlive eviction. |
| C-012 | A strict physical byte bound is unavailable from these standard containers. | PROVEN | Allocator metadata and `HashMap` bucket layout are not exposed as a stable byte count. |
| C-013 | A configured capacity above `u32::MAX` cannot carry a strict per-item charge contract. | PROVEN | Moka's weigher returns `u32`; the current helper clamps every larger estimate to `u32::MAX`. |
| C-014 | The separate `cache-moka` integration repeats the 768/256 constants but has separate caches and APIs. | PROVEN | `crates/integrations/cache-moka/src/lib.rs` defines its own estimators and two independent default budgets. |
| C-015 | No current branch or open pull request owns F-CATIO-WEIGHT. | PROVEN | Open fork pull requests were empty; active worktrees and branches only showed the independent insert-distribution unit and older cache work. |

Zero propositions are `OPEN` or `REJECTED` for the audit.

## Root rulings

1. Adopt a deterministic retained parsed-object charge. Do not claim RSS, allocator-exact bytes, or
   a proven conservative byte upper bound.
2. Preserve the existing capacity API and values above `u32::MAX`. Scope the demonstrated bound to
   capacities at or below `u32::MAX`. Record the per-entry clamp above that boundary.
3. Keep `crates/integrations/cache-moka` outside this unit. Its duplicated accounting is a separate
   follow-up.
4. Preserve zero-cache behavior and raw-manifest context semantics.
5. Use private saturating helpers and structural tests. Add no dependency, allocator instrument, or
   RSS gate.

There is no small safe per-item bypass for an unrepresentable charge. Moka's coalesced entry
initializer admits its value through a `u32` weigher. Bypassing only after parsing would replace the
miss-coalescing path or add expiration policy machinery. This unit keeps the clamp and documents it.

## Ownership partition

### Cache-owned parsed charge

The implemented charge includes one deterministic logical charge for each cached entry's key and
value graph:

- the key enum, tuple, path string, format version, and optional schema identifier;
- the cached enum, `Arc`, manifest or manifest-list shell, and top-level vector;
- manifest metadata, schema graph, partition spec, and partition field names and transforms;
- every manifest-entry `Arc` and shell;
- every data-file path, partition literal, metrics map, bound datum, encryption metadata buffer,
  split-offset vector, equality-id vector, and referenced-data-file path;
- every manifest-file path, partition-summary vector, lower and upper bound buffer, and encryption
  metadata buffer.

A shared `Arc` graph receives a charge for each cache entry that reaches it. A local Moka weigher
cannot deduplicate allocations shared between different entries.

### Excluded from the charge contract

- Moka's map nodes, key wrappers, policy deques, frequency sketch, queues, and bookkeeping;
- allocator headers, size classes, fragmentation, arenas, and memory not returned to the operating
  system;
- input bytes, Avro decoder state, parse temporaries, futures, and I/O buffers;
- concurrent distinct-key misses before admission and maintenance;
- contextual manifests returned by `get_manifest`, including their deep-cloned entries;
- caller-held `Arc` references after an entry is evicted or invalidated;
- the `ObjectCache` shell, retained `FileIO`, runtime stacks, and all unrelated process memory.

The configured number therefore cannot be named a resident-memory ceiling or a process-RSS limit.
The charge is deterministic for one compiled runtime and parsed graph. `size_of` and collection
capacity can differ across targets or dependency versions, so equal inputs need not receive the same
numeric charge across builds.

## Production charter

| ID | Requirement | Pin |
|---|---|---|
| P-001 | Name the unit a deterministic retained parsed-object **charge**, not measured resident bytes. | API and documentation wording review. |
| P-002 | Include the cache key and every variable-length parsed field listed in the ownership partition. | Small/large paired fixtures for each field class. |
| P-003 | Use saturating arithmetic for every addition, multiplication, and capacity conversion. | Boundary tests ending at `u32::MAX`. |
| P-004 | Keep zero capacity as cache-disabled behavior. | Existing disabled-cache tests. |
| P-005 | Keep raw-manifest caching and caller-context isolation unchanged. | Existing F-CATIO-KEY tests. |
| P-006 | Bound the settled sum of policy charges for capacities at or below `u32::MAX`; make no physical-byte claim. | Capacity boundary and oversized-candidate tests. |
| P-007 | Settle Moka maintenance before asserting entry count, weighted size, or eviction. | Every structural cache test calls `run_pending_tasks().await`. |
| P-008 | Assert eviction structurally and do not require a strict LRU victim. | The 280,000-byte test asserts fewer than all entries survive; the RePark warmed-coldest pin remains the end-to-end discriminator. |
| P-009 | Preserve the separate `cache-moka` provider behavior unless its public accounting contract is explicitly added to this unit. | Diff scope gate; file remains byte-identical by default. |
| P-010 | Add no RSS threshold to the normal suite. | Test manifest inspection. |

## Minimal design choices

1. Keep the single shared asynchronous cache in `ObjectCache`.
2. Change its weigher to receive both key and value.
3. Replace the two entry-count constants with private saturating charge helpers.
4. Count shallow nodes once and add variable payload charges from strings, vectors, maps, literals,
   schema nodes, partition summaries, and data-file metrics.
5. Use a documented logical policy for opaque allocation details. Do not describe that policy as
   allocator-exact or a physical-byte upper bound.
6. Keep the configured capacity and cache-disabled constructor behavior unchanged.
7. Do not use source-file length, child `manifest_length`, RSS sampling, a custom allocator, or
   serialized-value caching as the capacity metric. Each either misses parsed shape, is unstable,
   or changes the cache's behavior.

The smallest credible alternative is a calibrated fixed floor plus variable payload charges. Merely
changing 768 and 256 to larger constants passes the measured fixture but still ignores arbitrarily
large paths, metrics, partition values, schemas, and summaries. That alternative does not close
P-002.

## Test partition

| Test | Required discrimination |
|---|---|
| `test_a_budget_below_retained_charge_evicts_entries` | Baseline code retains all 256 manifest/list pairs at 280,000. Corrected accounting retains some entries but evicts at least one after settled maintenance. |
| key-length pair | A longer unique path increases charge. Removing key accounting survives only the smaller expectation. |
| manifest shape pair | Longer paths, populated metric maps, bounds, nested partition literals, vectors, and referenced paths each increase charge. |
| metadata shape pair | A larger schema and partition spec increase charge. |
| manifest-list shape pair | Paths, partition summaries, summary bounds, and key metadata each increase charge. |
| empty and one-entry pair | Both charge nonzero; one entry is heavier than empty. |
| arithmetic boundary | Synthetic near-overflow components saturate without panic and clamp to `u32::MAX`. |
| oversized candidate | With capacity at or below `u32::MAX`, an entry heavier than capacity is rejected and a smaller entry remains cacheable. |
| maintenance boundary | Assertions are false-green resistant by checking entry count and weighted size after `run_pending_tasks`. |
| context/concurrency regression | Existing raw-manifest context-isolation and same-key concurrency tests remain green. |
| cache-disabled regression | Existing zero-capacity and disabled-cache paths remain green. |

No global RSS replay is required to implement this partition. The existing RePark measurements remain
the before-fix evidence and can be repeated only after the structural tests prove a need.

The regression creates 256 one-entry manifest/list pairs directly in the real cache. Against the
baseline implementation, it failed because all 512 entries remained at a 262,144 weighted size
under the 280,000 budget. Against the implemented charge, it proves that some entries remain,
eviction occurs, and settled policy weight does not exceed the budget.

## Execution state

The implementation replaces entry-count constants with saturating graph-charge helpers. The Moka
weigher now includes each key. Read-only `pub(crate)` accessors expose map pair storage and schema
indices that were otherwise unreachable from `ObjectCache`.

The final cache regression asserts eviction after settled maintenance. Paired tests cover key paths,
data-file paths and metrics, bounds, nested partition literals, metadata vectors, schema fields and
aliases, manifest-list summaries, saturation, and oversized admission.

## Verification evidence

| Check | Result |
|---|---|
| Baseline production plus final eviction regression | Exit 101; the `< 512` assertion failed because all 512 entries remained. Log: `/tmp/codex-fork-cache-weight-red.log`. |
| `cargo test -p iceberg --lib io::object_cache --all-features --locked --offline` with two jobs | 22 passed, 0 failed on the remediated patch. |
| `make check` with two Cargo jobs and the shared build lock | Exit 0; workspace all-target, all-feature Clippy ran with warnings denied. |
| `make unit-test` with two Cargo jobs and the shared build lock | Exit 0; the `iceberg` library population was 3,663 passed, 0 failed, 8 ignored, and every workspace library and doctest target passed. |
| `cargo fmt --all -- --check` | Exit 0 on the frozen patch. |
| Rust file-size, comment-block, agent-artifact, targeted typo, and diff-whitespace gates | Exit 0 on the frozen patch. |

No RSS replay, dependency change, lockfile change, workflow change, AWS action, RePark pin change,
commit, push, or pull request occurred.

## Critic disposition, 2026-09-08

| Finding | Disposition | Evidence | Action |
|---|---|---|---|
| F-CATIO-WEIGHT-1 | PARTLY CONFIRMED | Identifier storage, actual accessor ownership, and the duplicated `Map` key payload lacked independent pins. Arc control-word arithmetic and separate derived schema-index copies are implementation details beneath already pinned field classes. | Added focused identifier, accessor-graph, and map key/value payload pins. No test restates generic helper arithmetic. |
| F-CATIO-WEIGHT-2 | CONFIRMED | The existing schema fixtures contained only top-level primitive fields. The `Type::Struct`, `Type::List`, and `Type::Map` traversal arms received no direct fixture. | Added a long-versus-short nested payload pair for each arm. |
| F-CATIO-WEIGHT-3 | CONFIRMED | `SchemaBuilder::build_accessors` creates `Arc` accessors only for primitive leaves reachable through struct fields. It creates none for list, map, or struct container fields. Each struct nesting level adds one boxed accessor node. The charge instead used every `id_to_field` entry and counted no boxes. | Exposed the actual accessor count from `Schema`. The charge now derives boxed nodes from struct-only type paths and uses saturating arithmetic for both classes. |
| F-CATIO-WEIGHT-4 | REFUTED | In `BiHashMap` 0.6.3, `Ref<T>` owns `Rc<T>`. `insert_unchecked` creates one `Rc<L>` and one `Rc<R>`, then clones the `Ref` values into the two hash maps. Both maps reference one alias `String` allocation. | Keep one alias string payload charge. Do not double it. |
| F-CATIO-WEIGHT-5 | REFUTED AS A CODE CHANGE | The capacity-above-`u32::MAX` limitation is explicit in this ledger and the external report. The owner ban permits no new code comment or rustdoc for it. The public `u64` capacity API remains unchanged. | Keep the ledger/report disclosure and existing capacity boundary test. |

The remediation remains inside `ObjectCache`, its direct schema helper, direct tests, and this
ledger. The new charge tests use a separate source file because the existing test file is at 999 of
the permitted 1,000 lines. This does not change a production module boundary.

## Critic-remediation mutation evidence

Each mutation ran the same four-test `io::object_cache::charge_tests` population under the shared
build lock. Each red result was 3 passed and 1 failed. The fixed source was restored after every
run and the restored population passed 4 of 4. Logs are retained in
`/tmp/codex-cache-weight-mutation-logs`.

| Removed or weakened behavior | Test that failed |
|---|---|
| Actual accessor count replaced with all schema-field identifiers | `test_schema_accessor_charge_counts_actual_arc_and_box_nodes` |
| Boxed nested-accessor charge removed | `test_schema_accessor_charge_counts_actual_arc_and_box_nodes` |
| Nested `Struct` traversal removed | `test_schema_type_graph_charge_tracks_struct_list_and_map_payloads` |
| Nested `List` traversal removed | `test_schema_type_graph_charge_tracks_struct_list_and_map_payloads` |
| Nested `Map` traversal removed | `test_schema_type_graph_charge_tracks_struct_list_and_map_payloads` |
| Identifier-set storage charge removed | `test_schema_charge_tracks_identifier_storage` |
| Duplicated `Map` key payload multiplier reduced from two to one | `test_map_literal_charge_tracks_duplicate_keys_and_single_values` |

The earlier bad accessor proxy also failed the final accessor test with an observed charge of 96
against the intended 80. Lock-acquisition timeouts were discarded and are not mutation evidence.

## Final independent review and draft-PR readiness, 2026-09-08

The candidate source, test, and map bytes did not change between the Actor gates and the final
independent Grok review. Grok compiled and ran 22 focused object-cache tests: 22 passed and 3,649
were filtered out. It issued `CLEAN` at S2 with complete charter coverage. This closeout changes
ledger prose only.

The review independently exercised the key, manifest, metadata, manifest-list, schema identifier,
accessor graph, literal `Map`, and nested `Struct`/`List`/`Map` discriminators. It confirmed that
the duplicate literal-map key storage needs two payload charges and that the two `BiHashMap`
directions share one alias `String` allocation through `Rc`, so the alias payload must not be
doubled. The filed omission findings are closed. The per-entry `u32::MAX` clamp remains an accepted
S3 limitation. The deterministic charge is not an allocator-exact or process-RSS bound, capacities
above `u32::MAX` remain outside the demonstrated bound, and the separate cache-moka implementation
remains out of scope.

Actor mutation runs made each of the seven required discriminators red and restored all four
charge-test cases green. The Actor's `make check` and `make unit-test` runs exited 0. Final readiness
checks also exited 0 for `typos .`, `make check-msrv`, the no-default-features `iceberg` build, and
`cargo deny check advisories`; the advisory run retained the existing nonfatal yanked-crate warning
for `chacha20` 0.10.1.

The frozen base and fetched `origin/main` were both
`85db42f285703682629b3b53bd1ebcd3091c6bc5`. At this pre-commit checkpoint, no source delta had
followed independent review and no commit, push, pull request, or merge had occurred. The Docker
integration gate is blocked: the configured desktop socket is absent, access to the system daemon
is denied, and port 9000 is
occupied. The canonical `make test` attempt exited 2 in `docker-up` before test execution. It did
not alter Docker resources. No Docker pass, R7 waiver, full R7 readiness, merge, or delivery is
claimed.
