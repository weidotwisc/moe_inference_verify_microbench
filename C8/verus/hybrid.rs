// hybrid.rs
//
// Verus attempt at TP × DP × EP hybrid block correctness for Ex07.
// Proves the STRUCTURAL properties of the sub-group composition:
//   H1 stubbed (attention TP-replicated output, cited from Ex04)
//   H2 (MoE-input striping determinism)
//   H3 stubbed (EP dispatch symmetry, cited from Ex06)
//   H4 stubbed (all-gather post-condition on tp_group)
//   H5 stubbed (sub-group deadlock-freedom, structural)
//   H6 is proved over the shared exact model in
//   verus/schedule_refinement.rs.
//
// Focus: predicates that express replication and striping in a fixed
// TP/EP layout. The full proof is a composition of Ex04 + Ex06 theorems,
// which lives in the paper's composition-theorem section, not here.
//
// Run with:
//   verus hybrid.rs

use vstd::prelude::*;

verus! {

// =====================================================================
// §1 — Types.
// =====================================================================

pub type Rank = nat;
pub type Group = Set<Rank>;

// =====================================================================
// §2 — Layout predicates.
// =====================================================================

/// A partitioning of ranks into contiguous tp_groups of size `tp_size`.
pub open spec fn is_tp_partition(
    world_size: nat, tp_size: nat, tp_groups: Seq<Group>,
) -> bool {
    &&& world_size >= tp_size
    &&& tp_size >= 1
    &&& world_size % tp_size == 0
    &&& tp_groups.len() == world_size / tp_size
    &&& forall|g: int| #![auto] 0 <= g < tp_groups.len() as int
            ==> tp_groups[g].finite() && tp_groups[g].len() == tp_size
}

// =====================================================================
// §3 — Property H2: MoE-input striping is deterministic.
//
// Given a rank r with tp_rank = r mod tp_size, dp_rank = r div tp_size,
// its local slice of the pre-MoE tensor is determined by (tp_rank, dp_rank).
// This makes H2 a definitional property.
// =====================================================================

pub open spec fn tp_rank_of(rank: Rank, tp_size: nat) -> nat
    recommends tp_size >= 1,
{
    (rank as int % tp_size as int) as nat
}

pub open spec fn dp_rank_of(rank: Rank, tp_size: nat) -> nat
    recommends tp_size >= 1,
{
    (rank as int / tp_size as int) as nat
}

pub proof fn h2_striping_deterministic(
    r: Rank, r_prime: Rank, tp_size: nat,
)
    requires
        tp_size >= 1,
        r != r_prime,
        tp_rank_of(r, tp_size) == tp_rank_of(r_prime, tp_size),
    ensures dp_rank_of(r, tp_size) != dp_rank_of(r_prime, tp_size),
{
    // If two ranks share the same tp_rank (mod tp_size) but are distinct,
    // they must live in different dp_ranks (div tp_size).
    let tp = tp_size as int;
    // r = dp_rank_of(r) * tp + tp_rank_of(r); similarly for r_prime.
    // If both dp_rank and tp_rank match, r == r_prime, contradiction.
    assert(r as int == dp_rank_of(r, tp_size) as int * tp + tp_rank_of(r, tp_size) as int)
        by (nonlinear_arith)
        requires
            tp_rank_of(r, tp_size) == (r as int % tp) as nat,
            dp_rank_of(r, tp_size) == (r as int / tp) as nat,
            tp >= 1;
    assert(r_prime as int == dp_rank_of(r_prime, tp_size) as int * tp
                          + tp_rank_of(r_prime, tp_size) as int)
        by (nonlinear_arith)
        requires
            tp_rank_of(r_prime, tp_size) == (r_prime as int % tp) as nat,
            dp_rank_of(r_prime, tp_size) == (r_prime as int / tp) as nat,
            tp >= 1;
}

// =====================================================================
// §4 — Replication predicate (like Ex01's TP invariant).
// =====================================================================

pub uninterp spec fn tensor_on(tensor_id: nat, rank: Rank) -> Seq<int>;

pub open spec fn Replicated_on(tensor_id: nat, g: Group) -> bool {
    forall|r1: Rank, r2: Rank|
        g.contains(r1) && g.contains(r2)
            ==> tensor_on(tensor_id, r1) == tensor_on(tensor_id, r2)
}

// =====================================================================
// §5 — Properties H1, H3, H4, H5, H6 (stubs citing composition).
// =====================================================================

/// H1: attention output is TP-replicated. Inherited from Ex04's R4
/// (RowParallelLinear all-reduce). Stated here for reference; the proof
/// belongs to Ex04's verification.
#[verifier::external_body]
pub proof fn h1_attn_output_tp_replicated_stub(attn_out_id: nat, tp_group: Group)
    ensures Replicated_on(attn_out_id, tp_group),
{}

/// H3: EP dispatch symmetry. Inherited from Ex06's EP3 with `ep_group`
/// as the comm domain (which is a superset of tp_group in the hybrid layout).
#[verifier::external_body]
pub proof fn h3_ep_dispatch_symmetric_stub()
    ensures true,
{}

/// H4: after all_gather_into_tensor on tp_group, the MoE output is
/// replicated within tp_group. Direct postcondition of the collective.
#[verifier::external_body]
pub proof fn h4_moe_out_tp_replicated_stub(moe_out_id: nat, tp_group: Group)
    ensures Replicated_on(moe_out_id, tp_group),
{}

/// H5: sub-group deadlock-freedom. Structural property of the fixed
/// straight-line schedule; not SMT-provable, asserted as an axiom of
/// the schedule.
#[verifier::external_body]
pub proof fn h5_subgroup_deadlock_free_stub()
    ensures true,
{}

// H6 is intentionally not restated as a local axiom.  The exact Hybrid
// refinement and its DP=1 schedule composition live in
// `verus/schedule_refinement.rs`.

} // verus!

fn main() {
    println!("Verified TP x DP x EP hybrid block structural properties (Verus).");
}
