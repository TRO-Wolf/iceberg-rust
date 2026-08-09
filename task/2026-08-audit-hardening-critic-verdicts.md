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

# 2026-08 audit hardening — S3 residue register

Companion to [2026-08-audit-hardening-ledger.md](2026-08-audit-hardening-ledger.md). Every S3 filed
by a unit Critic, verbatim from the verdict that filed it.

**Why this file exists.** The ledger's first draft said "31 S3 items carried forward … full text in
the per-unit Critic verdicts" and then summarised six of them. The bundle Critic filed that as S2:
no such verdict artifact existed on disk, so 25 of the 31 were unauditable counts — the same
unbacked-claim defect (R-09) the ledger was written to close, recurring inside it. The counts are
now backed by the text below.

S3 means: real, but neither a correctness nor a contract problem. None of these blocked a unit.


## R1 — G1 decimal Java-parity restore  (wave 1 — remediation)

Critic verdict: **CONVERGED**, 6 S3.

1. **`crates/iceberg/src/spec/values/serde.rs`** — The `PrimitiveLiteral::Int128` write arm's guard is `let Type::Primitive(_) = ty else {...}` while both its comment ("A decimal literal must be typed as a decimal") and its error message ("Literal decimal value {v} requires a decimal type, got {ty}") assert it requires a DECIMAL. Carried from cycle 1, but the cycle-1 rationale is WRONG: it claimed the looseness fails closed at Avro resolution. It does not for two of four cases.

   *Disposition:* Either restore the decimal requirement without the magnitude gate — `let Type::Primitive(PrimitiveType::Decimal { .. }) = ty else { ... }` (Java-faithful: `Conversions.toByteBuffer`'s DECIMAL arm at 253-266 does `checkcast BigDecimal`, so a non-decimal type is unreachable there) and add a `Type::Primitive(PrimitiveType::Int)` pin beside the existing `Type::Struct` pin — or correct the comment and error message to say "primitive type" and state why. Carry the corrected Fixed/Binary rationale to the G6 Critic.

2. **`crates/iceberg/src/spec/values/literal.rs`** — The comment at lines 743-745 on the `try_into_json` decimal arm states Java 1.10.0 `SingleValueParser.toJson` "writes `value.toString()` unconditionally — no precision or metadata gate". Independently bytecode-confirmed FALSE on two counts: there IS a scale-equality `Preconditions.checkArgument` gate, and the writer uses `toPlainString()` for `scale >= 0`. No behavioral defect in the fork, but a wrong Java citation in a load-bearing comment in a repo whose product is Java parity.

   *Disposition:* Correct the comment to state what `toJson` actually does — `instanceof BigDecimal` + `BigDecimal.scale() == DecimalType.scale()` Preconditions check (offsets 555-589), then `toPlainString()` for `scale >= 0` / `toString()` for `scale < 0` (592-627) — and note the fork satisfies the scale precondition by construction because `try_decimal_from_i128_with_scale` reconstructs at the declared scale, and that `scale < 0` is unrepresentable since Iceberg scale is `u32`.

3. **`crates/iceberg/src/spec/values/datum.rs`** — The rationale comment inside `Datum::decimal_with_precision` ("Validate the metadata before checking the value. In particular, a scale greater than the precision must not be accepted just because the value happens to fit in one byte.") is stale: R1 narrowed `validate_decimal_type` to `1 <= precision <= 38`, so the `scale <= precision` rejection now happens only at the `Type::decimal(precision, scale)?` tail, AFTER the byte-width check the comment says must run second.

   *Disposition:* Attribute the `scale <= precision` rejection to the `Type::decimal(precision, scale)?` construction-door tail and drop the ordering claim, or reorder so the stated invariant is enforced where the comment says it is.

4. **`task/todo.md`** — `crates/iceberg/src/spec/values/decimal_utils.rs` is absent from G1's frozen "In scope:" list yet received a real behavior change to the PUBLIC `i128_from_be_bytes` (buffers >16 bytes now accepted when the excess is pure sign extension; previously always `None`). The Actor correctly declined to fix this while S3s were forbidden, but the R1 sub-entry it DID add is the natural home and still omits it.

   *Disposition:* Add one line to the existing R1 sub-entry recording the `decimal_utils.rs` scope extension and the public-surface widening of `i128_from_be_bytes`, so the G6 bundle-close Critic adjudicates it against C-007 rather than discovering it. This is ledger completeness, not correctness — the change itself is Java-faithful and fully pinned.

5. **`crates/iceberg/src/spec/values/datum.rs`** — NEW (not in cycle-1 residue). R1 creates a read/write asymmetry that is not recorded anywhere: the fork now READS a Java-written decimal whose unscaled magnitude exceeds its declared precision, but `Datum::to_bytes` still refuses to re-emit it, whereas Java's `Conversions.toByteBuffer` emits it unchecked. Any manifest-rewriting operation over such a table therefore errors, because `manifest::_serde::to_bytes_entry` calls `d.to_bytes()?`.

   *Disposition:* Record the asymmetry in the R1 sub-entry as a deliberate, fail-closed divergence from Java's write path (one sentence). Do NOT remove the `to_bytes` gate — it is the thing preventing silent corruption. If full round-trip fidelity for out-of-precision Java bounds is ever wanted, that is a separate scoped unit, not part of R1.

6. **`crates/iceberg/src/arrow/schema.rs`** — Carried from cycle 0 as accepted residue, sharpened: `datum_to_arrow_type_with_ree` calls `datum.validate_decimal()?` even though it only computes an Arrow DataType and never needs the value. Since it serves constant/partition columns on the scan path, an out-of-precision Java-written partition constant still aborts the scan — the exact failure class ff53c252 exists to eliminate, just at a different door.

   *Disposition:* Either drop the value-magnitude check from the type-only REE helper (keeping the precision/scale metadata checks Arrow genuinely needs), or record explicitly in the R1 sub-entry that the Arrow REE door remains magnitude-strict by choice and why. No change without a scoped decision.


## R2 — G2 predicate depth — kill the panic, re-derive the limit  (wave 1 — remediation)

Critic verdict: **CONVERGED**, 6 S3.

1. **`crates/iceberg/src/expr/predicate.rs`** — The `MAX_PREDICATE_DEPTH` doc states "The one remaining unbounded recursion in this module is the derived `Drop` glue" (line 45). This is FALSE: the derived `Clone` and `PartialEq` on `Predicate`/`BoundPredicate` recurse too, and `Clone` is the FIRST of the three to blow — the doc names the least dangerous and omits the most dangerous. Same class as FINDING 5, which this commit claims to close, and it contradicts the commit's own task/todo.md entry ("the derived `Drop`/`Clone`/`PartialEq` glue ... still recurses") and its own test comment at predicate.rs:1948.

   *Disposition:* Change the sentence to name all three derived walks and their relative cost, e.g. "The remaining unbounded recursions in this module are the DERIVED `Clone`, `PartialEq` and `Drop` glue — measured on a 2 MiB stack (dev) they overflow at roughly 1,465 / 4,052 / 14,500 levels respectively, so `Clone` is the binding one; no depth check can intercept them." Keep it consistent with the todo.md wording that is already correct.

2. **`crates/iceberg/src/expr/predicate.rs`** — New broken intra-doc link on a PUBLIC method's rustdoc, introduced by this commit. The `Predicate::rewrite_not` doc (line 1003) links to `crate::expr::visitors::predicate_visitor::visit`, but that module was `#[cfg(test)]`-gated in the same commit, so it does not exist in the build rustdoc sees. It renders as broken text in published docs for a `pub` API.

   *Disposition:* Either drop the link and refer to the visitor by plain name in backticks, or keep the prose and point the link at the still-public `crate::expr::visitors::bound_predicate_visitor::visit` (which is not gated) while naming the unbound twin in text.

3. **`crates/iceberg/src/expr/predicate.rs`** — Surviving mutant: `BoundPredicate::negate`'s NOT-cancellation arm (line 1164, `(*input_0, false)`) is unpinned — flipping it to `(*input_0, true)` leaves the entire 3221-test lib suite GREEN. The unbound twin at line 940 IS pinned by `negate_splices_a_cancelled_not_without_descending`; the bound half, which is the one on the read path, is not. I verified the arm is UNREACHABLE in production, so this is a dead-but-load-bearing branch rather than a live defect — which is why it is S3 and not S2.

   *Disposition:* Either add a bound twin of `negate_splices_a_cancelled_not_without_descending` (bind `NOT((bar < 40) AND (foo IS NULL))`, negate, assert it equals the bound inner conjunction), or state the unreachability at the arm: "structurally unreachable — the only caller is `rewrite_not`'s `Negate` frame, whose `leaf` is NOT-free by post-order induction; retained so a future caller inherits the correct De Morgan semantics."

4. **`crates/iceberg/src/expr/predicate.rs`** — The depth guard is INEFFECTIVE in debug builds: at the measured 5,504 B/level, `Predicate::bind` aborts the process at depth 382 on a 2 MiB tokio worker, long before MAX_PREDICATE_DEPTH=1000 can return the typed `DataInvalid`. CLAUDE.md's Recursion rule ("malformed input must not overflow the thread stack") is therefore met only in release. Not a regression against the merge base e4f7f010, which has no limit at all, and the Actor discloses it in the constant's doc and residue #3 rather than hiding it — hence S3, not S2.

   *Disposition:* No change required for this unit. Record in the G2 tracker that C-003's protective intent is release-only for `Predicate::bind`, and that full closure requires making `bind` and the two `visit`s explicit-stack (the same treatment `rewrite_not`/`negate`/`Display` received here), not a smaller constant — no value is simultaneously >=1000 for parity and dev-safe on a 2 MiB worker.

5. **`crates/iceberg/src/expr/visitors/predicate_visitor.rs`** — Compile-time exhaustiveness lost on three spec-critical matches. `bind_at_depth` (predicate.rs:573), `predicate_visitor::visit_at_depth` (line 156) and `bound_predicate_visitor::visit_at_depth` (line 172) now end in a catch-all `leaf => ...` arm instead of enumerating every `Predicate`/`BoundPredicate` variant. Before this commit each match was exhaustive, so adding a variant broke the build; now a new logical variant (e.g. an XOR node) silently routes to the `*_leaf` fallthrough and surfaces as a runtime `Unexpected` error instead of a compile error.

   *Disposition:* Replace the catch-all with the explicit leaf variants — `Predicate::AlwaysTrue | Predicate::AlwaysFalse | Predicate::Unary(_) | Predicate::Binary(_) | Predicate::Set(_) => Self::bind_leaf(...)` — in all three functions. This keeps the `#[inline(never)]` frame-size win (which I confirmed is real and load-bearing) while restoring the compile-time check on the enum.

6. **`crates/iceberg/src/arrow/delete_filter.rs`** — FINDING 4's underlying parity break is pushed out, not retired. `build_equality_delete_predicate` (lines 718-745) still LEFT-folds `combined_predicate = combined_predicate.and(predicate)` once per equality-delete file, so depth is linear in file count; a table with more than ~1000 eq-delete files still fails a scan that Java completes. The sibling per-file path at caching_delete_file_loader.rs:846-859 already solved exactly this with a balanced tree and carries the comment "Using a simple fold would result in a deeply nested predicate that can cause a stack overflow"; the cross-file fold never got the same treatment. The Actor names this as residue #2, and delete_filter.rs is outside G2's declared scope in task/todo.md, so declining it here is correct.

   *Disposition:* Queue a follow-up unit that converts the cross-file fold to the balanced shape already used at caching_delete_file_loader.rs:846-859 (and `reduce(Predicate::and)` at integrations/datafusion/src/physical_plan/expr_to_predicate.rs:49). That makes depth logarithmic in file count and retires the class rather than moving the cliff. Do not close FINDING 4's parity residue in the G6 bundle Critic until it lands.


## R3 — schema-evolution recursion — bound the walk that is actually unbounded  (wave 1 — remediation)

Critic verdict: **CONVERGED**, 5 S3.

1. **`crates/iceberg/src/spec/schema/id_reassigner.rs`** — Carried forward and INDEPENDENTLY RE-CONFIRMED: all three container arms of `assign_ids_at_depth` (list element, map key, map value) are unpinned. `assign_ids` is public API (re-exported at spec/schema/mod.rs:41-42) with no in-crate production caller, so it is pure library surface — a lost depth increment on any container arm is an unbounded public recursion door. Only its struct arm is covered, by `assign_ids_rejects_hostile_nesting_instead_of_overflowing`. No false claim is attached (the RISK comment says only 'deleting the guard from assign_ids_at_depth', and commit 340fa4ea discloses the gap verbatim in its residue), which is why this stays S3.

   *Disposition:* Add list-chain and map-chain (KEY and VALUE) depth pins for `assign_ids` mirroring `assign_fresh_ids_bounds_list_and_map_nesting_too`, and mutation-prove each arm's `depth + 1` RED individually.

2. **`task/todo.md`** — Carried forward and re-confirmed: line 111 of the G2 tracker entry still reads 'STILL OPEN from the first pass: `assign_fresh_ids` is unbounded.' That is now false — 006dc721 bounded it and 340fa4ea pinned the last arm. G2 is still an open `[ ]` item that the G6 bundle-close Critic is charged with adjudicating from this file, so a stale line here is read as authoritative at bundle close. Not blocking: commit 340fa4ea's message discloses the staleness explicitly, and this cycle's instruction forbade fixing S3.

   *Disposition:* Update the G2 entry to record that `assign_fresh_ids` and `assign_ids` are depth-bounded at `MAX_ASSIGN_IDS_NESTING_DEPTH = 128` (006dc721) with all four `assign_fresh_ids_at_depth` arms mutation-pinned (340fa4ea), keeping the surviving residue (derived Drop/Clone/PartialEq glue on Predicate/BoundPredicate; the unpinned `assign_ids` container arms) intact. Per the de-triplication rule, state it once here only.

3. **`crates/iceberg/src/spec/schema/mod.rs`** — Carried forward and independently re-verified by my own read: `SchemaBuilder::build` calls `self.build_accessors()` at line 217 BEFORE `index_by_id(&r#struct)` at line 220, and `build_accessors_nested` self-recurses on the `Type::Struct(nested)` arm (line ~298 onward) with no depth parameter. So a deeper-than-128 struct chain handed to `Schema::builder().with_fields(..).build()` overflows the thread stack in accessor construction instead of returning the visitor's typed error. This does NOT weaken the commit under review — the new guard rejects at 129 before any such schema is built — but it is a live public-API stack-overflow door in the same class as C-003, and it is outside G2's scope.

   *Disposition:* Ledger only for this unit. Track as its own unit: either make `build_accessors_nested` iterative (explicit stack) or move the `index_by_id` depth check ahead of accessor construction so the typed error wins the race.

4. **`task/todo.md`** — NEW (governance): `crates/iceberg/src/spec/schema/id_reassigner.rs` is not in G2's declared in-scope file list, which reads 'crates/iceberg/src/expr/accessor.rs, crates/iceberg/src/expr/visitors/{predicate_visitor.rs,bound_predicate_visitor.rs,manifest_evaluator.rs}, crates/iceberg/src/arrow/schema.rs, crates/iceberg/src/transaction/update_schema.rs, and affected map.md/adjacent tests'. Two commits on this branch (006dc721 production, 340fa4ea tests) modify it. The extension is substantively justified — id_reassigner.rs is the implementation of the exact hazard named in G2's C-003 and reachable only through the in-scope `update_schema.rs` public `add_column` — but the tracker was never amended to record it, so the G6 bundle-close Critic will adjudicate against a scope list the branch has outgrown. Cycle 2's Actor also reported `out_of_scope_files_needed: []`, which understates this.

   *Disposition:* Amend the G2 in-scope list in task/todo.md to include `crates/iceberg/src/spec/schema/id_reassigner.rs` with the one-line justification (the C-003 recursion door reachable from `UpdateSchemaAction::add_column`), so the scope extension is on the record before G6 rather than discovered at bundle close.

5. **`crates/iceberg/src/spec/schema/id_reassigner.rs`** — NEW (minor, test ergonomics): the three container chains are three sequential `expect_err` assertions inside the single test fn `assign_fresh_ids_bounds_list_and_map_nesting_too`, so the first failure aborts the test and masks the state of the remaining arms. The Actor's per-mutant proof is nonetheless valid — I reproduced it and each mutant fails exactly one assertion with the others intact — but a maintainer who later sees one RED cannot tell from the failure alone which of the three arms are still pinned, which is precisely the ambiguity this test exists to remove. The RISK comment's phrasing 'each turns exactly one of the three expect_errs below RED' is accurate as written and is not a false claim.

   *Disposition:* Ledger only, or optionally split into three `#[test]` fns (`..._bounds_list_nesting`, `..._bounds_map_value_nesting`, `..._bounds_map_key_nesting`) sharing the chain builders, so a regression names its own arm without masking the others.


## G4 — cache-moka byte-weighted capacity (C-005, OTH-002)  (wave 2 — groups)

Critic verdict: **CONVERGED**, 6 S3.

1. **`crates/integrations/cache-moka/src/lib.rs`** — `with_manifest_list_cache` has no test; the commit newly documents a "caller's builder wins" contract for it that nothing verifies. Its twin `with_manifest_cache` is pinned by `caller_supplied_cache_keeps_its_own_policy`.

   *Disposition:* Ledger for G6. Now a two-line addition: the `manifest_list_with_entries(n)` fixture exists, so mirror the twin — supply `moka::sync::Cache::new(7)` via `with_manifest_list_cache` and assert `policy().max_capacity() == Some(7)`. Non-blocking: the setter body is unchanged from before this unit, so the gap is pre-existing and the unit only added prose to it.

2. **`crates/integrations/cache-moka/src/lib.rs`** — `new_with_capacity` is a NEW public constructor introduced by this unit — the pre-fix file had no such method — and neither commit message nor the tracker labels it as a public-surface addition, though CLAUDE.md asks for surface changes to be called out. Related: `new()`'s effective budget changes from ~33.5M entries to 32 MiB per cache, a material downstream memory-behavior change.

   *Disposition:* Ledger for G6: record in the G4 tracker entry that the unit ADDS `pub fn new_with_capacity(u64)` (additive, non-breaking) and that `new()` changes semantics, so downstream release notes can carry it. No code change.

3. **`crates/integrations/cache-moka/README.md`** — README.md is outside the frozen G4 scope clause, which reads "`crates/integrations/cache-moka/src/lib.rs` and adjacent tests".

   *Disposition:* Ledger for G6: record the README as an approved scope addition. Content is benign and documents exactly the behavior the unit changed; this cycle added nothing new to it.

4. **`crates/integrations/cache-moka/src/lib.rs`** — The default aggregate resident ceiling is now 2 x 32 MiB = 64 MiB — twice the core `ObjectCache`'s single 32 MiB budget for the same two object kinds. Deliberate (merging the caches is ARCH-004, excluded) and documented on the constant, the type, both constructors and the README, but it is a newly-true number an operator inherits.

   *Disposition:* Ledger for G6 / ARCH-004. No code change in this unit.

5. **`task/todo.md`** — The unit gate declared in the workstream instructions ends in `cargo test -p iceberg --lib`, which does not execute a single one of this unit's 10 tests — they live in `iceberg-cache-moka`. `clippy --all-targets` compiles them but never runs them.

   *Disposition:* Ledger for G6: for any group whose tests land outside the `iceberg` crate, append the owning crate's test invocation to the same && chain. The Actor already appended `cargo test -p iceberg-cache-moka --lib` to this cycle's own chain without editing the frozen charter, which is the right call.

6. **`crates/integrations/cache-moka/src/lib.rs`** — The `if cache_size_bytes == 0 { return moka::sync::Cache::new(0); }` early return in `build_weighted_cache` is redundant: falling through to the builder yields `max_capacity == Some(0)`, which moka treats as disabled identically. It is an equivalent mutant, not a coverage gap.

   *Disposition:* Ledger only, or optionally delete the branch and keep `zero_capacity_disables_both_caches` (which still passes and still catches the real hazard of treating 0 as unbounded). No defect; recording it so a future reader does not misread the surviving mutant as a hole.


## G3 — DataFusion nested namespace identity (C-004, OTH-001)  (wave 2 — groups)

Critic verdict: **CONVERGED**, 5 S3.

1. **`crates/integrations/datafusion/src/catalog.rs`** — A partial-guard mutant of the newly pinned `seen` check SURVIVES the whole suite, so the Actor's design-decision #2 — that the `listings_issued() == 3` assertion "catches a mutant that terminates but re-expands (e.g. a `seen` that is written but consulted too late)" — is FALSE for at least one member of exactly that class. Moving `discovered.push(namespace.clone())` OUT of the `if seen.insert(..)` block while leaving `fresh.push(namespace)` inside still terminates (the frontier guard is intact) but fills `discovered` with duplicates. In production that is not only a cost defect: `build_schema_providers` issues one redundant `list_tables` per duplicate, and `alias_claims` counts the duplicate as a second claimant, so a MULTI-LEVEL namespace listed twice by a re-answering catalog has its dot alias silently DROPPED — the exact SQL-reachability loss OTH-001 was fixed for. The `== 3` assertion cannot see it because the duplicated work costs `list_tables`, not `list_namespaces`.

   *Disposition:* Ledger for G6. If cheap there: give `ScriptedCatalog` a second `AtomicUsize` counting `list_tables` and assert it equals the namespace count in both new tests, or make one scripted tree two levels deep so the dropped alias becomes observable. Also correct design-decision #2 in the G6 close notes — the listing count pins the N+1 SHAPE, it does not pin `discovered` against duplication.

2. **`crates/integrations/datafusion/src/catalog.rs`** — The `MAX_SCRIPTED_LISTINGS` doc comment overstates its own headroom: "Every scripted tree here is 1–2 namespaces, so a correct walk spends 3 listings; 64 leaves two orders of magnitude of headroom before the budget can produce a false RED." 64/3 is 21.3x, i.e. ~1.3 orders of magnitude (64/2 namespaces = 32x, ~1.5 orders). The Actor's own report says "~20x headroom", so the shipped comment is the over-general one.

   *Disposition:* Ledger; in G6 change "two orders of magnitude" to "~20x", or raise the constant if two orders is actually wanted. No behaviour change.

3. **`crates/integrations/datafusion/src/catalog.rs`** — CARRIED FORWARD FROM CYCLE 1 AND INDEPENDENTLY RE-VERIFIED AS STILL OPEN: the `CatalogProvider::schema` lookup order (`schemas` then `aliases`) is dead redundancy behind the registration-time `schemas.contains_key(&alias)` guard, so swapping it is invisible to the suite. Separately, `a_child_listing_failure_fails_construction_and_names_the_namespace` still asserts only the message, unlike its table-listing twin which also asserts `Error::source(&err).is_some()`, so the `with_source` chain on the child-listing arm (AGENTS.md requires the chain survive) remains unpinned.

   *Disposition:* Ledger for G6 as filed in cycle 1. If cheap: add `assert!(std::error::Error::source(&err).is_some())` to the child-listing failure test, and note in the `schema()` doc that the lookup order is defence-in-depth behind the registration guard rather than the load-bearing mechanism.

4. **`crates/integrations/datafusion/src/catalog.rs`** — CARRIED FORWARD FROM CYCLE 1, RE-VERIFIED: two doc attributions in the same comment block the Actor edited are still wrong, deliberately left byte-identical so G6 adjudicates them as filed. (1) `NAMESPACE_DISCOVERY_CONCURRENCY` says 16 matches "`DEFAULT_LIST_STAT_CONCURRENCY` in `iceberg-storage-opendal`", but the opendal crate only imports it. (2) `MAX_NAMESPACE_DEPTH` is called "this crate's existing nesting bound (`physical_plan::project`'s `MAX_WRITE_COMPATIBILITY_DEPTH`)", but that constant bounds Arrow/Iceberg SCHEMA nesting in write-compatibility checks — an unrelated quantity, so 64 is a borrowed convention, not a shared bound. Also for the ledger: Java imposes no depth limit on `Namespace`, so a legal 65-level namespace hard-fails the whole DataFusion catalog provider — a deliberate, documented fork divergence.

   *Disposition:* Ledger; correct both sentences opportunistically in G6 (name the defining crate/path for `DEFAULT_LIST_STAT_CONCURRENCY`, phrase 64 as a borrowed convention) and record the Java no-depth-limit divergence in the G6 close notes.

5. **`crates/integrations/datafusion/src/catalog.rs`** — CARRIED FORWARD FROM CYCLE 1, RE-VERIFIED: REST is the only shipped catalog whose child listing is a real network call, and that path is unpinned offline. Two undisclosed consequences for inputs that used to work remain: (a) a server that 404s/rejects `?parent=` now fails whole-session construction where the old single `list_namespaces(None)` succeeded; (b) construction costs N+1 listings instead of 1. Neither is a defect in the fix. The Actor's new `listings_issued() == 3` assertion does make the N+1 shape observable in-tree, partially discharging (b), but the over-the-wire path is still only reachable under docker.

   *Disposition:* Ledger for G6: state that the REST recursion path is unpinned offline and that G6's docker `make test` (REST fixture + crates/sqllogictest, which builds an `IcebergCatalogProvider` over a REST catalog) is its pin. Add the N+1 cost and the `?parent=`-rejecting-server failure mode to the `try_new` failure-policy doc comment alongside the existing skip-vs-fail rationale.


## G5 — secret rendering — close the enumerated credential paths (C-006)  (wave 2 — groups)

Critic verdict: **CONVERGED**, 3 S3.

1. **`crates/iceberg/src/io/storage/config/mod.rs`** — The new PUBLIC rustdoc on `RedactedProps` says it is "Used by ... the REST wire types (SEC-010)", and `client.rs::deserialize_catalog_response`'s doc opens "SECURITY (SEC-010): this is the SUCCESS (2xx) body path". In the audit this unit closes, SEC-010 is "Local FS storage has no path sandbox" — an unrelated finding. The REST banner's "SEC-010" numbering comes from an EARLIER audit (it is present unchanged on main at types.rs:33), so the unit has propagated a cross-audit ID collision into a new public API doc in the core crate. A reader who resolves SEC-010 against the 2026-08-08 audit is sent to the wrong finding.

   *Disposition:* Qualify the identifiers with their audit epoch (e.g. `SEC-010 (pre-2026-08 audit)`) or renumber to the current audit's SEC-002/SEC-009 where that is what is meant, at least in the newly authored `RedactedProps` doc. No code change.

2. **`crates/iceberg/src/spec/table_metadata.rs`** — The commit message's Named-residue paragraph asserts "none of the docs claim it is [wholesale credential-safe]". A pre-existing doc in the same redaction family still makes a broader claim than the code supports: `impl std::fmt::Debug for StorageConfig`'s rustdoc says a `{:?}` of "any struct that embeds a `FileIO` and derives `Debug` — cannot leak credentials". `Table` embeds a `FileIO` and derives `Debug`, and by this unit's own ledger it CAN leak credentials through `metadata.snapshots[*].summary`. The sentence is untouched by this commit (it is verbatim on main), so this is a pre-existing over-generalization the unit's absolute claim now collides with — not a defect the unit introduced.

   *Disposition:* Either scope the StorageConfig doc sentence to the FileIO/StorageConfig props channel it actually covers, or soften the commit message's absolute "none of the docs claim it is" to name the exception. Doc-only.

3. **`crates/iceberg/src/spec/table_metadata.rs`** — `statistics[*].key_metadata` has no mutation of its own that isolates it. M10c masks the whole `statistics` field, which trips the `blob_metadata[*].properties` assertion FIRST (the loop's third entry), so the fourth assertion — the one guarding `key_metadata` — is never independently demonstrated RED. A future hand-written `Debug` for `StatisticsFile` that redacts `blob_metadata` but forgets `key_metadata` would still be caught (assertion 3 fires), but the converse — redacting only `key_metadata` — has not been mutation-proved. This is a coverage-evidence gap in the ledger, not a live defect: the assertion exists and I confirmed by inspection of the M10a failure dump that `key_metadata: Some("RESIDUE_STATISTICS_KEY_METADATA_VALUE")` really is what renders it.

   *Disposition:* Record in the mutation ledger that M10c covers the `statistics` field as a whole and that the `key_metadata` arm is asserted but not separately mutated, or add M10e (wrap `key_metadata` alone in a redacting adapter inside a temporary `StatisticsFile` Debug) when someone next touches this test.


---

## Bundle-scope Critic (C-008) — three lenses + two verification rounds

The closing chain filed six S2s against the **artifact**, not the code (all three lenses reported
zero S1/S2 code defects and re-derived every Java citation as accurate). Those six are dispositioned
in the ledger; what follows are the non-blocking items the terminal round ledgered.

1. **`crates/iceberg/src/expr/predicate.rs:45-48`** — the `MAX_PREDICATE_DEPTH` doc names Drop,
   Clone and PartialEq as the remaining unbounded recursion. Derived `Debug`/`Serialize`/
   `Deserialize`, and the hand-written `Debug` for `LogicalExpression` (`predicate.rs:129`), also
   walk the tree. True read as a category statement ("the DERIVED glue"); not exhaustive read as a
   list.
2. **Same sentence** — calls `Drop` "derived glue". It is compiler-generated drop glue, not a
   `#[derive]`. Conventional shorthand, carried unchanged from earlier wording.
3. **R2 item 1, partly discharged** — that verdict also proposed recording the relative overflow
   depths (Clone blows first at ~1,465 dev levels, then PartialEq, then Drop) so a reader knows
   **Clone** is the binding constraint. The doc now names all three but not the ordering.
4. **Citation form** — the ledger cites "G4 S3-1", "G4 S3-2", "S3-3(a)"; this register numbers
   items `1..n` per unit with no prefix or sub-label. The mapping is one-to-one and inferable
   (G4 item 3 is the README scope item) but is not literal.
5. **`task/lessons.md`** — the R3 entry says a mutant "survived all 3226 tests". Correct as a
   point-in-time figure at R3; the suite is 3232 at tip, which a reader could conflate.
6. **Ledger §4.2 header** — see the note now inline in that section.
