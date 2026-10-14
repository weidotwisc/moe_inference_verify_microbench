// Exact E1/E2/F4 kernel refinements over the shared MoE work model.

use vstd::prelude::*;
use crate::composition_core::*;

verus! {

/// E1: the naive token/top-k loop is the canonical exact MoE semantics.
pub proof fn e1_naive_refines_spec(input: MoeInput)
    ensures semantic_eq(naive_forward(input), moe_spec(input)),
{
}

/// E2: routing permutation changes execution order but not the exact output.
pub proof fn e2_permuted_refines_spec(input: MoeInput)
    requires RoutingConsistent(input),
    ensures
        semantic_eq(permuted_forward(input), moe_spec(input)),
        semantic_eq(permuted_forward(input), naive_forward(input)),
{
    theorem_routing_plan_refines_spec(input);
    e1_naive_refines_spec(input);
    theorem_shared_spec_implies_equiv(
        permuted_forward(input), naive_forward(input), moe_spec(input),
    );
}

/// F4: a fused kernel satisfying its row-level postcondition refines the same
/// exact MoE semantics.  The GPU/source correspondence is isolated in
/// `fused_rows_correct`, not assumed as whole-output equivalence.
pub proof fn f4_fused_refines_spec(input: MoeInput)
    requires
        RoutingConsistent(input),
        fused_rows_correct(input),
    ensures
        semantic_eq(fused_forward(input), moe_spec(input)),
        semantic_eq(fused_forward(input), permuted_forward(input)),
{
    theorem_fused_rows_refine_plan(input);
    theorem_routing_plan_refines_spec(input);
    lemma_semantic_eq_trans(
        fused_forward(input), permuted_forward(input), moe_spec(input),
    );
}

pub proof fn theorem_naive_equiv_fused(input: MoeInput)
    requires
        RoutingConsistent(input),
        fused_rows_correct(input),
    ensures semantic_eq(naive_forward(input), fused_forward(input)),
{
    e1_naive_refines_spec(input);
    f4_fused_refines_spec(input);
    theorem_shared_spec_implies_equiv(
        naive_forward(input), fused_forward(input), moe_spec(input),
    );
}

pub proof fn theorem_permuted_equiv_fused(input: MoeInput)
    requires
        RoutingConsistent(input),
        fused_rows_correct(input),
    ensures semantic_eq(permuted_forward(input), fused_forward(input)),
{
    e2_permuted_refines_spec(input);
    f4_fused_refines_spec(input);
    theorem_shared_spec_implies_equiv(
        permuted_forward(input), fused_forward(input), moe_spec(input),
    );
}

pub proof fn corollary_naive_equiv_permuted(input: MoeInput)
    requires RoutingConsistent(input),
    ensures semantic_eq(naive_forward(input), permuted_forward(input)),
{
    e1_naive_refines_spec(input);
    e2_permuted_refines_spec(input);
    theorem_shared_spec_implies_equiv(
        naive_forward(input), permuted_forward(input), moe_spec(input),
    );
}

} // verus!
