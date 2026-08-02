// Hour-0 / after microbench for FK1 eq-delete keyset apply.
// Cargo.toml is frozen — run as a standalone example via:
//   cargo run -p iceberg --example ...  is unavailable without Cargo.toml edit.
// Invocation (from worktree, after `cargo test -p iceberg --lib` has built deps):
//   rustc is NOT used alone; instead use the in-tree unit microbench below via:
//   cargo test -p iceberg --lib fk1_eq_delete_apply_microbench -- --nocapture --ignored
//
// This file documents the bench matrix for the ledger; the live numbers come from the
// ignored test `fk1_eq_delete_apply_microbench` in equality_delete_set.rs.
//
// Matrix: 1M data rows × {100k, 1M} eq-deletes; single Long key; non-null.
// Metrics: wall ns/row, approximate alloc via peak RSS if /proc available.
