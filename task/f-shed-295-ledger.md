# F-SHED-295 — relocated code sheds its comments

Owner ruling, 2026-09-18 evening: when code moves into another file, its comments do not travel
with it. Fork #294 (`fdaa82d4`) and #295 (`9e67e000`) moved test modules and helpers into new files
and carried their comment lines along; #296 (`3296ffc7`) carried none.

Measured with the RePark comment gate (`comment_ban.py <clone> <sha>^ <sha>`):

| Merge | Hits | Files |
|---|---|---|
| #294 `fdaa82d4` | 54 | `physical_plan/expr_to_predicate.rs` (7), `physical_plan/expr_to_predicate_tests.rs` (47) |
| #295 `9e67e000` | 149 | `writer/base_writer/data_file_writer_tests.rs` (35), `physical_plan/conform.rs` (27), `table/schema_evo_tests.rs` (58), `table/tests.rs` (29) |
| #296 `3296ffc7` | 0 | — |

This change deletes exactly those 203 lines: 199 whole comment lines, and the trailing `//` text of
four timestamp literals in `expr_to_predicate_tests.rs`. rustfmt then collapsed five blank lines the
deletions left doubled. The ASF licence headers stay. No code token changes.

Proof: `comment_ban.py <clone> 8477b249 HEAD` reports `comment-ban hits=0` on this head, which covers
#294, #295 and #296 together. `cargo test -p iceberg-datafusion` and the `data_file_writer_tests`
module of `iceberg` pass.

Model: claude-opus-5 (orchestrator, scripted deletion of the gate's own list)
