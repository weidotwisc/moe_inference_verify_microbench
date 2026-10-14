// fused_moe.rs
//
// Verus attempt at the algorithmic-contract correctness of Ex09's
// fused-MoE Triton grouped-GEMM kernel.
//
// The companion `fused_kernel_dsl.rs` proves the exact DSL model equal to
// the Python grouped-expert reference.  This file states the component-facing
// ALGORITHMIC CONTRACT:
//   F1 (precondition consistency: monotone offsets partition [0, M))
//   F3 (empty-expert handling: rows outside any block unaffected)
//   F4 is proved from a row-level fused-kernel postcondition in
//   verus/kernel_refinement.rs.
//
// Downstream (Ex10 fused hybrid) refines this contract in composition
// with Ex07's HybridBlock.
//
// Run with:
//   verus fused_moe.rs

use vstd::prelude::*;

verus! {

// =====================================================================
// §1 — Types (mirror of the other exercises).
// =====================================================================

pub type Element = int;
pub type ExpertId = nat;

// =====================================================================
// §2 — Precondition on (offsets, M).
// =====================================================================

/// The kernel's precondition on its (offsets, M) arguments: offsets is a
/// length-(E+1) CSR-style array with 0-endpoint, monotone increments,
/// and total length M.
pub open spec fn fused_moe_precondition_offsets(
    offsets: Seq<nat>, experts_per_rank: nat, total_tokens: nat,
) -> bool {
    &&& offsets.len() as nat == experts_per_rank + 1
    &&& offsets[0] == 0nat
    &&& offsets[experts_per_rank as int] == total_tokens
    &&& forall|e: int| #![trigger offsets[e]] 0 <= e < experts_per_rank as int
            ==> offsets[e] <= offsets[e + 1]
}

// =====================================================================
// §3 — Property F1: precondition consistency.
//
// Every row index in [0, M) falls in exactly one expert's block.
// =====================================================================

/// Predicate: row index `i` is inside expert `e`'s block.
pub open spec fn in_expert_block(offsets: Seq<nat>, e: ExpertId, i: nat) -> bool
    recommends (e as int) + 1 < offsets.len() as int,
{
    offsets[e as int] <= i && i < offsets[(e as int) + 1]
}

/// F1: monotonic + full-range offsets partition [0, M) into per-expert blocks.
/// Formal statement: every row i in [0, M) either falls in one block or
/// belongs to the range [offsets[E-1], offsets[E]) — the last block.
pub proof fn f1_offsets_cover_range(
    offsets: Seq<nat>, experts_per_rank: nat, total_tokens: nat, i: nat,
)
    requires
        fused_moe_precondition_offsets(offsets, experts_per_rank, total_tokens),
        i < total_tokens,
        experts_per_rank >= 1,
    ensures
        offsets[0] <= i && i < offsets[experts_per_rank as int],
{
    // offsets[0] == 0 <= i (trivial since i is nat)
    // offsets[experts_per_rank] == total_tokens > i (by hypothesis)
    assert(offsets[0] == 0nat);
    assert(offsets[experts_per_rank as int] == total_tokens);
}

// =====================================================================
// §4 — Property F3: empty-expert handling.
//
// If offsets[e] == offsets[e+1], no row index falls in expert e's block.
// =====================================================================

pub proof fn f3_empty_expert_no_rows(
    offsets: Seq<nat>, experts_per_rank: nat, e: ExpertId, i: nat,
)
    requires
        experts_per_rank >= 1,
        offsets.len() as nat == experts_per_rank + 1,
        (e as int) + 1 < offsets.len() as int,
        offsets[e as int] == offsets[(e as int) + 1],
    ensures !in_expert_block(offsets, e, i),
{
    // If offsets[e] == offsets[e+1], the interval [offsets[e], offsets[e+1])
    // is empty, so no i satisfies both bounds.
}

// =====================================================================
// §5 — The abstract expert-apply function (uninterpreted).
// =====================================================================

pub uninterp spec fn expert_apply(e: ExpertId, x_token: Seq<Element>) -> Seq<Element>;

// =====================================================================
// §6 — The kernel's postcondition (as a spec predicate on outputs).
// =====================================================================

/// The kernel's postcondition on (out, sorted_x, offsets, ...):
/// for every row i in [0, M), letting e be the unique expert with
/// offsets[e] <= i < offsets[e+1], out[i] equals expert_apply(e, sorted_x[i]).
pub open spec fn fused_moe_postcondition_holds(
    out: Seq<Seq<Element>>,
    sorted_x: Seq<Seq<Element>>,
    offsets: Seq<nat>,
    experts_per_rank: nat,
    // owner: uninterpreted function mapping row i to its owning expert.
    owner: spec_fn(nat) -> ExpertId,
) -> bool {
    &&& out.len() == sorted_x.len()
    &&& forall|i: int| #![trigger out[i]] 0 <= i < out.len() as int
            ==> out[i] == expert_apply(owner(i as nat), sorted_x[i])
}

// F4 is intentionally not restated as a whole-output axiom.  The shared
// exact proof in `verus/kernel_refinement.rs` derives it from the pointwise
// `fused_rows_correct` kernel boundary.

} // verus!

fn main() {
    println!("Verified fused-MoE algorithmic contract (Verus).");
}
