// verus/axiom_base.rs
//
// Shared cross-component primitives for the paper's Verus track.
//
// This file provides DISTRIBUTED / MULTI-RANK predicates (Replicated,
// ExpertPartition, semantic_eq, MoE_forward_spec, routing_conserved_local)
// used by Ex06+ verifications where components have pre/postconditions
// about distributed state.
//
// For single-rank structural proofs (Ex01–Ex05), each component's
// verification stays self-contained in `bootcamp/exNN/verification/*.rs`,
// following the Ex01 template. See:
//     bootcamp/ex01_linear_tp/verification/PROPERTIES.md
//     bootcamp/ex01_linear_tp/verification/README.md
// for the per-component pattern (matmul uninterpreted + M1/M2 axioms) and
// the delivery plan through Ex07.
//
// Style follows the three preferences agreed on 2026-08-31:
//   (1) decompose with named lemmas, no inline reasoning
//   (2) forall-quantified specs, not indexed conjunctions
//   (3) vstd primitives (Seq, Set, Map, nat, int), no custom containers
//
// Build (from repo root):
//   verus --crate-type=lib verus/axiom_base.rs
//
// Verus version this was drafted against: 0.2025.07.12.0b6f3cb

use vstd::prelude::*;
use vstd::multiset::Multiset;

verus! {

// =========================================================================
// (A) Ranks, groups, expert ids, token ids — abstract types.
// =========================================================================

pub type Rank = nat;
pub type ExpertId = nat;
pub type TokenId = nat;

/// A group is the set of ranks that participate in a collective.
pub type Group = Set<Rank>;

// =========================================================================
// (B) Tensor as a shape + content record.
//
// `content` is flat row-major and equality is exact.
// =========================================================================

pub struct Tensor {
    pub shape: Seq<nat>,
    pub content: Seq<int>,
    pub dtype: DType,
    /// The multiset of weighted per-expert contributions this tensor
    /// aggregates, when it is a MoE-layer output. For tensors that are not
    /// MoE outputs (weights, activations) this is the empty multiset; the
    /// exact semantic model can track it. See composition_theorem.rs for the
    /// current block-level equality theorem.
    pub contribs: Multiset<Contribution>,
}

/// A single weighted per-expert contribution: "token `token` receives
/// expert `expert`'s output scaled by (integer-encoded) `weight`."
pub struct Contribution {
    pub token: nat,
    pub expert: nat,
    pub weight: int,
}

pub enum DType {
    Fp32,
    Bf16,
}

pub open spec fn tensor_numel(t: Tensor) -> nat {
    t.content.len() as nat
}

pub open spec fn shape_prod(shape: Seq<nat>) -> nat
    decreases shape.len()
{
    if shape.len() == 0 {
        1nat
    } else {
        shape[0] * shape_prod(shape.subrange(1, shape.len() as int))
    }
}

/// A tensor is well-formed when its flat content length matches the product
/// of its shape dimensions.
pub open spec fn well_formed(t: Tensor) -> bool {
    tensor_numel(t) == shape_prod(t.shape)
}

// =========================================================================
// (C) Exact semantic equality.
// =========================================================================

pub open spec fn semantic_eq(x: Tensor, y: Tensor) -> bool {
    x == y
}

pub proof fn lemma_semantic_eq_refl(x: Tensor)
    ensures semantic_eq(x, x)
{
}

pub proof fn lemma_semantic_eq_trans(x: Tensor, y: Tensor, z: Tensor)
    requires
        semantic_eq(x, y),
        semantic_eq(y, z),
    ensures semantic_eq(x, z)
{
}

pub proof fn lemma_semantic_eq_sym(x: Tensor, y: Tensor)
    requires semantic_eq(x, y)
    ensures semantic_eq(y, x)
{
}

// =========================================================================
// (D) Distributed-tensor model — `tensor_on(x, r)` returns the value that
// tensor identity `x` takes on rank `r`. Opaque; postconditions of collective
// operations are stated as axioms in section (E).
// =========================================================================

/// The value of distributed tensor `x` as observed on rank `r`.
pub uninterp spec fn tensor_on(x: Tensor, r: Rank) -> Tensor;

/// A tensor is Replicated on group `g` iff every pair of ranks in `g` sees the
/// same content. This is the load-bearing precondition for the lean schedule.
pub open spec fn Replicated(x: Tensor, g: Group) -> bool {
    forall|r1: Rank, r2: Rank|
        g.contains(r1) && g.contains(r2)
            ==> tensor_on(x, r1) == tensor_on(x, r2)
}

// =========================================================================
// (E) Trusted axioms about torch.distributed collectives.
//
// Each `#[verifier::external_body]` proof declaration is a trusted axiom.
// This is where our formal system meets the runtime; the audit boundary
// is precisely these external_body declarations.
// =========================================================================

/// Axiom: broadcast from source establishes Replicated on the whole group.
#[verifier::external_body]
pub proof fn axiom_broadcast_produces_replicated(x: Tensor, g: Group, src: Rank)
    requires g.contains(src)
    ensures Replicated(x, g)
{
}

/// Axiom: after all_reduce(SUM), all ranks in the group see the SAME sum tensor.
/// (We do not attempt to spec the numeric value of the sum here; downstream
/// contracts reason about *equality across ranks*, not about the sum itself.)
#[verifier::external_body]
pub proof fn axiom_all_reduce_sum_replicated(y: Tensor, g: Group)
    ensures Replicated(y, g)
{
}

// =========================================================================
// (F) Expert partitioning — the invariant that experts are disjointly sharded.
// =========================================================================

/// `owner` maps each expert id to the unique rank that owns it.
/// Together with disjointness and coverage, this states expert-sharding.
pub open spec fn ExpertPartition(
    experts: Set<ExpertId>,
    ranks: Set<Rank>,
    owner: Map<ExpertId, Rank>,
) -> bool {
    &&& owner.dom() =~= experts
    &&& forall|e: ExpertId| #![auto] experts.contains(e)
            ==> ranks.contains(owner[e])
}

// =========================================================================
// (G) Routing conservation — used by every EP contract.
// =========================================================================

/// `offsets` is a length-(E+1) CSR-style array; consecutive entries bound the
/// slice of `sorted_x` owned by expert e. Monotonicity + endpoint conditions
/// give token conservation for this rank's local slice.
pub open spec fn routing_conserved_local(
    offsets: Seq<nat>,
    experts_per_rank: nat,
    total_tokens: nat,
) -> bool {
    &&& offsets.len() as nat == experts_per_rank + 1
    &&& offsets[0] == 0nat
    &&& offsets[experts_per_rank as int] == total_tokens
    &&& forall|e: int| #![trigger offsets[e]] 0 <= e < experts_per_rank as int
            ==> offsets[e] <= offsets[e + 1]
}

// =========================================================================
// (H) The MoE semantic spec — the SINGLE function every schedule refines.
//
// Kept opaque here. A schedule contract may assert exact semantic equality
// with this shared specification.
// =========================================================================

pub struct ExpertWeights {
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
}

pub uninterp spec fn MoE_forward_spec(
    x: Tensor,
    experts: Set<ExpertId>,
    weights: Map<ExpertId, ExpertWeights>,
    gate_weight: Tensor,
    top_k: nat,
) -> Tensor;

}  // verus!
