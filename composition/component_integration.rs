// Bridges from the concrete Tier-1 component contracts to the shared
// Tier-3 composition model.

use vstd::prelude::*;
use crate::composition_core::*;
use crate::schedule_refinement::*;

#[path = "../bootcamp/ex05_moe_baseline/verification/moe_baseline.rs"]
mod ex05_routing;

#[path = "../bootcamp/ex06_ep/verification/lean.rs"]
mod ex06_lean;

#[path = "../bootcamp/ex04_gqa_tp/verification/gqa_tp.rs"]
mod ex04_gqa;

#[path = "../bootcamp/ex09_fused_moe/verification/fused_moe.rs"]
mod ex09_fused_moe;

#[path = "../bootcamp/ex09_fused_moe/verification/fused_kernel_dsl.rs"]
mod ex09_fused_dsl;

verus! {

/// Source-level MoE fields before routing and kernel execution add their
/// derived arrays.  Keeping derived arrays out of this structure prevents the
/// composition theorem from assuming their semantic correctness.
pub struct ComponentMoeData {
    pub token_values: Seq<int>,
    pub expert_ids: Seq<Seq<ExpertId>>,
    pub weights: Seq<Seq<int>>,
    pub top_k: nat,
    pub num_experts: nat,
    pub experts_per_rank: nat,
    pub expert_parallel_size: nat,
}

/// Static/router fields of the MoE sublayer.  The token values are supplied
/// by the preceding post-attention normalization rather than independently
/// assumed.
pub struct ComponentMoeConfig {
    pub expert_ids: Seq<Seq<ExpertId>>,
    pub weights: Seq<Seq<int>>,
    pub top_k: nat,
    pub num_experts: nat,
    pub experts_per_rank: nat,
    pub expert_parallel_size: nat,
}

pub open spec fn component_data_from_moe_input(
    moe_input_tensor: Tensor,
    config: ComponentMoeConfig,
) -> ComponentMoeData {
    ComponentMoeData {
        token_values: moe_input_tensor.content,
        expert_ids: config.expert_ids,
        weights: config.weights,
        top_k: config.top_k,
        num_experts: config.num_experts,
        experts_per_rank: config.experts_per_rank,
        expert_parallel_size: config.expert_parallel_size,
    }
}

/// Exact deterministic operations between GQA's output projection and MoE.
/// Their numerical definitions are intentionally abstract here; composition
/// only needs the fact that equal inputs produce equal outputs.
pub uninterp spec fn residual_add(
    residual: Tensor,
    update: Tensor,
) -> Tensor;

pub uninterp spec fn rms_norm(input: Tensor) -> Tensor;

pub open spec fn post_gqa_residual(
    residual: Tensor,
    gqa_output: Tensor,
) -> Tensor {
    residual_add(residual, gqa_output)
}

pub open spec fn post_attention_norm(
    residual: Tensor,
    gqa_output: Tensor,
) -> Tensor {
    rms_norm(post_gqa_residual(residual, gqa_output))
}

/// The concrete per-rank tensor presented to MoE after residual-add and
/// post-attention RMSNorm.
pub open spec fn post_attention_view(
    residual: Tensor,
    gqa_output_id: nat,
) -> RankTensorView {
    |rank: Rank| post_attention_norm(
        residual, tensor_on(gqa_output_id, rank),
    )
}

pub open spec fn component_work_count(data: ComponentMoeData) -> nat {
    data.token_values.len() * data.top_k
}

/// Flatten the router's selected expert ids in canonical `(token, slot)`
/// order.  Ex05's argsort contract is applied to this exact sequence.
pub open spec fn component_routing_keys(data: ComponentMoeData) -> Seq<ExpertId> {
    Seq::new(component_work_count(data), |i: int|
        if data.top_k > 0 {
            data.expert_ids[i / data.top_k as int][i % data.top_k as int]
        } else {
            0nat
        }
    )
}

pub open spec fn routing_order_from_perm(perm: Seq<nat>) -> Seq<WorkItem> {
    Seq::new(perm.len(), |i: int| WorkItem { flat: perm[i] })
}

/// The shared input is constructed from the actual Ex05 argsort output;
/// `routing_order` is no longer an independently supplied semantic field.
pub open spec fn component_moe_input(
    data: ComponentMoeData,
    fused_values: Seq<int>,
) -> MoeInput {
    let keys = component_routing_keys(data);
    MoeInput {
        token_values: data.token_values,
        expert_ids: data.expert_ids,
        weights: data.weights,
        top_k: data.top_k,
        num_experts: data.num_experts,
        experts_per_rank: data.experts_per_rank,
        expert_parallel_size: data.expert_parallel_size,
        routing_order: routing_order_from_perm(ex05_routing::sort_perm(keys)),
        fused_values,
    }
}

pub open spec fn well_formed_component_data(data: ComponentMoeData) -> bool {
    &&& data.top_k > 0
    &&& data.num_experts > 0
    &&& data.experts_per_rank > 0
    &&& data.expert_parallel_size > 0
    &&& data.num_experts
        == data.expert_parallel_size * data.experts_per_rank
    &&& data.expert_ids.len() == data.token_values.len()
    &&& data.weights.len() == data.token_values.len()
    &&& forall|t: int| #![auto]
        0 <= t < data.token_values.len() ==> (
            data.expert_ids[t].len() == data.top_k
            && data.weights[t].len() == data.top_k
            && forall|s: int| #![trigger data.expert_ids[t][s]]
                0 <= s < data.top_k
                    ==> data.expert_ids[t][s] < data.num_experts
        )
}

proof fn lemma_inverse_pair_order_no_duplicates(
    perm: Seq<nat>,
    inv: Seq<nat>,
    m: nat,
)
    requires ex05_routing::is_inverse_pair(perm, inv, m),
    ensures routing_order_from_perm(perm).no_duplicates(),
{
    assert forall|i: int, j: int|
        0 <= i < routing_order_from_perm(perm).len()
        && 0 <= j < routing_order_from_perm(perm).len()
        && i != j
        implies routing_order_from_perm(perm)[i]
            != routing_order_from_perm(perm)[j] by {
        if routing_order_from_perm(perm)[i]
            == routing_order_from_perm(perm)[j] {
            assert(perm[i] == perm[j]);
            ex05_routing::rt4_perm_injective(perm, inv, m, i, j);
        }
    }
}

proof fn lemma_inverse_pair_order_contains(
    perm: Seq<nat>,
    inv: Seq<nat>,
    m: nat,
    item: WorkItem,
)
    requires ex05_routing::is_inverse_pair(perm, inv, m),
    ensures routing_order_from_perm(perm).contains(item)
        <==> item.flat < m,
{
    if item.flat < m {
        ex05_routing::rt4_perm_surjective(
            perm, inv, m, item.flat as int,
        );
        let i = inv[item.flat as int] as int;
        assert(0 <= i < routing_order_from_perm(perm).len());
        assert(routing_order_from_perm(perm)[i] == item);
    }
    if routing_order_from_perm(perm).contains(item) {
        let i = routing_order_from_perm(perm).index_of(item);
        assert(0 <= i < routing_order_from_perm(perm).len());
        assert(routing_order_from_perm(perm)[i] == item);
        assert(item.flat == perm[i]);
        assert(perm[i] < m);
    }
}

proof fn lemma_inverse_pair_routing_multiset(
    perm: Seq<nat>,
    inv: Seq<nat>,
    m: nat,
)
    requires ex05_routing::is_inverse_pair(perm, inv, m),
    ensures routing_order_from_perm(perm).to_multiset()
        == Seq::new(m, |i: int| WorkItem { flat: i as nat }).to_multiset(),
{
    broadcast use vstd::seq_lib::group_to_multiset_ensures;
    broadcast use vstd::multiset::group_multiset_axioms;
    let routed = routing_order_from_perm(perm);
    let canonical = Seq::new(m, |i: int| WorkItem { flat: i as nat });
    lemma_inverse_pair_order_no_duplicates(perm, inv, m);
    assert(canonical.no_duplicates()) by {
        assert forall|i: int, j: int|
            0 <= i < canonical.len() && 0 <= j < canonical.len() && i != j
            implies canonical[i] != canonical[j] by {
        }
    }
    routed.lemma_multiset_has_no_duplicates();
    canonical.lemma_multiset_has_no_duplicates();
    assert(routed.to_multiset() =~= canonical.to_multiset()) by {
        assert forall|item: WorkItem|
            routed.to_multiset().count(item)
                == canonical.to_multiset().count(item) by {
            lemma_inverse_pair_order_contains(perm, inv, m, item);
            if item.flat < m {
                assert(routed.contains(item));
                assert(canonical[item.flat as int] == item);
                assert(canonical.contains(item));
                assert(routed.to_multiset().contains(item));
                assert(canonical.to_multiset().contains(item));
            } else {
                assert(!routed.contains(item));
                assert(!canonical.contains(item)) by {
                    if canonical.contains(item) {
                        let i = canonical.index_of(item);
                        assert(0 <= i < canonical.len());
                        assert(canonical[i] == item);
                        assert(item.flat == i as nat);
                    }
                }
                assert(routed.to_multiset().count(item) == 0);
                assert(canonical.to_multiset().count(item) == 0);
            }
        }
    }
}

/// Ex05's argsort bijection discharges Tier-3 `RoutingConsistent` for the
/// concretely constructed routing order.
pub proof fn ex05_establishes_routing_consistent(
    data: ComponentMoeData,
    fused_values: Seq<int>,
)
    requires well_formed_component_data(data),
    ensures RoutingConsistent(component_moe_input(data, fused_values)),
{
    let keys = component_routing_keys(data);
    let perm = ex05_routing::sort_perm(keys);
    let inv = ex05_routing::sort_inv(keys);
    ex05_routing::axiom_argsort_is_inverse_pair(keys);
    lemma_inverse_pair_routing_multiset(perm, inv, keys.len());
}

/// Ex06's contiguous local-expert partition discharges the ownership
/// condition used by the rank-local/all-reduce proof.
pub proof fn ex06_establishes_expert_partitioned(
    data: ComponentMoeData,
    fused_values: Seq<int>,
    world_size: nat,
)
    requires
        well_formed_component_data(data),
        world_size == data.expert_parallel_size,
    ensures ExpertPartitioned(
        component_moe_input(data, fused_values), world_size,
    ),
{
    let input = component_moe_input(data, fused_values);
    assert(input.experts_per_rank == data.experts_per_rank);
    assert(input.num_experts == data.num_experts);
    assert(input.expert_parallel_size == data.expert_parallel_size);
    assert(input.expert_parallel_size == world_size);
    assert forall|expert: ExpertId| expert_owner(input, expert) < world_size by {
        if expert < input.num_experts {
            assert(expert < world_size * input.experts_per_rank);
            ex06_lean::l2_local_mask_covers(
                world_size, input.experts_per_rank, expert,
            );
            assert(exists|rank: Rank|
                rank < world_size
                    && ex06_lean::is_local_expert(
                        rank, input.experts_per_rank, expert,
                    ));
            let witness = choose|rank: Rank|
                rank < world_size
                    && ex06_lean::is_local_expert(
                        rank, input.experts_per_rank, expert,
                    );
            assert(witness < world_size);
            assert(ex06_lean::is_local_expert(
                witness, input.experts_per_rank, expert,
            ));
            assert(expert_owned_by_rank(input, witness, expert));
            assert(exists|rank: Rank|
                #[trigger] expert_owned_by_rank(input, rank, expert));
            let owner = expert_owner(input, expert);
            assert(owner < input.expert_parallel_size);
            assert(expert_owned_by_rank(input, owner, expert));
            if owner != witness {
                ex06_lean::l2_local_mask_disjoint(
                    world_size,
                    input.experts_per_rank,
                    owner,
                    witness,
                    expert,
                );
                assert(!ex06_lean::is_local_expert(
                    witness, input.experts_per_rank, expert,
                ));
                assert(false);
            }
        } else {
            assert(expert_owner(input, expert) == 0nat);
        }
    }
}

/// Ex04's row-parallel output projection makes the GQA result replicated.
/// Applying the same residual-add and RMSNorm on every rank preserves that
/// equality and establishes the exact input expected by the MoE theorem.
pub proof fn ex04_establishes_replicated_post_attention_input(
    residual: Tensor,
    gqa_output_id: nat,
    group: Group,
    representative: Rank,
    config: ComponentMoeConfig,
    fused_values: Seq<int>,
)
    requires group.contains(representative),
    ensures ReplicatedInput(
        component_moe_input(
            component_data_from_moe_input(
                post_attention_view(residual, gqa_output_id)(representative),
                config,
            ),
            fused_values,
        ),
        post_attention_view(residual, gqa_output_id),
        group,
    ),
{
    let input_view = post_attention_view(residual, gqa_output_id);
    let data = component_data_from_moe_input(
        input_view(representative), config,
    );
    let input = component_moe_input(data, fused_values);

    ex04_gqa::g5_gqa_output_replicated_after_output_projection(
        gqa_output_id, group,
    );
    assert(exists|rank: Rank| group.contains(rank)) by {
        assert(group.contains(representative));
    }
    assert forall|rank: Rank| group.contains(rank) implies
        input_view(rank) == Tensor { content: input.token_values } by {
        assert(tensor_on(gqa_output_id, rank)
            == tensor_on(gqa_output_id, representative));
        assert(input_view(rank) == input_view(representative));
        assert(input_view(representative)
            == Tensor { content: input.token_values });
    }
}

/// Concrete values and metadata returned by the Ex09 grouped-GEMM wrapper.
pub struct Ex09FusedRun {
    pub out: Seq<Seq<int>>,
    pub sorted_x: Seq<Seq<int>>,
    pub offsets: Seq<nat>,
    pub experts_per_rank: nat,
    pub owner: spec_fn(nat) -> ExpertId,
    pub gate_weights: Seq<ex09_fused_dsl::Tensor>,
    pub up_weights: Seq<ex09_fused_dsl::Tensor>,
    pub down_weights: Seq<ex09_fused_dsl::Tensor>,
    pub dispatch: ex09_fused_dsl::DispatchTable,
    pub hidden_cols: nat,
    pub intermediate_cols: nat,
    pub gate_block_n: nat,
    pub gate_num_n_tiles: nat,
    pub gate_block_k: nat,
    pub gate_num_k_tiles: nat,
    pub down_block_n: nat,
    pub down_num_n_tiles: nat,
    pub down_block_k: nat,
    pub down_num_k_tiles: nat,
}

/// Representation boundary between the mathematical Python expert and the
/// shared composition model's abstract expert function.  Unlike kernel
/// correctness, this predicate only identifies which mathematical function
/// the two layers call `expert_apply`.
pub open spec fn ex09_reference_matches_expert_apply(
    run: Ex09FusedRun,
) -> bool {
    let reference = ex09_fused_dsl::fused_moe_reference(
        run.sorted_x,
        run.gate_weights,
        run.up_weights,
        run.down_weights,
        run.owner,
        run.sorted_x.len(),
        run.hidden_cols,
        run.intermediate_cols,
    );
    &&& reference.len() == run.sorted_x.len()
    &&& forall|i: int| #![trigger reference[i]]
        0 <= i < reference.len() ==> reference[i]
            == ex09_fused_moe::expert_apply(
                (run.owner)(i as nat), run.sorted_x[i],
            )
}

/// Faithful-input contract for one modeled execution of the actual Ex09
/// Triton wrapper.  The equality to `fused_moe_dsl` is the source/DSL
/// correspondence boundary; arithmetic correctness is proved from it rather
/// than assumed as an output postcondition.
pub open spec fn ex09_dsl_execution(run: Ex09FusedRun) -> bool {
    &&& run.experts_per_rank == run.dispatch.num_experts
    &&& run.offsets == run.dispatch.expert_offsets
    &&& ex09_fused_dsl::well_formed(
        run.sorted_x, run.sorted_x.len(), run.hidden_cols,
    )
    &&& ex09_fused_dsl::grouped_weights_well_formed(
        run.gate_weights,
        run.dispatch.num_experts,
        run.intermediate_cols,
        run.hidden_cols,
    )
    &&& ex09_fused_dsl::grouped_weights_well_formed(
        run.up_weights,
        run.dispatch.num_experts,
        run.intermediate_cols,
        run.hidden_cols,
    )
    &&& ex09_fused_dsl::grouped_weights_well_formed(
        run.down_weights,
        run.dispatch.num_experts,
        run.hidden_cols,
        run.intermediate_cols,
    )
    &&& ex09_fused_dsl::dispatch_matches_row_owner(
        run.dispatch, run.sorted_x.len(), run.owner,
    )
    &&& run.gate_block_n > 0
    &&& run.intermediate_cols
        <= run.gate_num_n_tiles * run.gate_block_n
    &&& run.gate_block_k > 0
    &&& run.hidden_cols <= run.gate_num_k_tiles * run.gate_block_k
    &&& run.down_block_n > 0
    &&& run.hidden_cols <= run.down_num_n_tiles * run.down_block_n
    &&& run.down_block_k > 0
    &&& run.intermediate_cols
        <= run.down_num_k_tiles * run.down_block_k
    &&& run.out == ex09_fused_dsl::fused_moe_dsl(
        run.sorted_x,
        run.gate_weights,
        run.up_weights,
        run.down_weights,
        run.dispatch,
        run.sorted_x.len(),
        run.hidden_cols,
        run.intermediate_cols,
        run.gate_block_k,
        run.gate_num_k_tiles,
        run.down_block_k,
        run.down_num_k_tiles,
    )
    &&& ex09_reference_matches_expert_apply(run)
}

/// K2/K3/F4 turn a modeled Triton execution into Ex09's pointwise component
/// postcondition.  The postcondition is therefore a theorem, not a premise of
/// the composition contract.
pub proof fn ex09_dsl_execution_establishes_postcondition(
    run: Ex09FusedRun,
)
    requires ex09_dsl_execution(run),
    ensures ex09_fused_moe::fused_moe_postcondition_holds(
        run.out,
        run.sorted_x,
        run.offsets,
        run.experts_per_rank,
        run.owner,
    ),
{
    let reference = ex09_fused_dsl::fused_moe_reference(
        run.sorted_x,
        run.gate_weights,
        run.up_weights,
        run.down_weights,
        run.owner,
        run.sorted_x.len(),
        run.hidden_cols,
        run.intermediate_cols,
    );
    ex09_fused_dsl::f4_fused_moe_dsl_equals_python_reference(
        run.sorted_x,
        run.gate_weights,
        run.up_weights,
        run.down_weights,
        run.dispatch,
        run.owner,
        run.sorted_x.len(),
        run.hidden_cols,
        run.intermediate_cols,
        run.gate_block_n,
        run.gate_num_n_tiles,
        run.gate_block_k,
        run.gate_num_k_tiles,
        run.down_block_n,
        run.down_num_n_tiles,
        run.down_block_k,
        run.down_num_k_tiles,
    );
    assert(run.out == reference);
    assert(run.out.len() == run.sorted_x.len());
    assert forall|i: int| #![trigger run.out[i]]
        0 <= i < run.out.len() implies run.out[i]
            == ex09_fused_moe::expert_apply(
                (run.owner)(i as nat), run.sorted_x[i],
            ) by {
        assert(run.out[i] == reference[i]);
    }
}

/// Representation relation between Ex09's vector-row contract and Tier 3's
/// scalar work-item abstraction.  It only relates layouts and the two levels'
/// expert semantics; correctness of `out` is derived separately from the DSL
/// execution theorem.
pub open spec fn ex09_run_matches_shared_layout(
    input: MoeInput,
    run: Ex09FusedRun,
) -> bool {
    &&& run.out.len() == input.routing_order.len()
    &&& run.sorted_x.len() == input.routing_order.len()
    &&& input.fused_values.len() == input.routing_order.len()
    &&& forall|i: int| #![trigger run.out[i]]
        0 <= i < input.routing_order.len() ==> {
            let item = input.routing_order[i];
            let token = work_token(input, item);
            let slot = work_slot(input, item);
            &&& required_work_item(input, item)
            &&& (run.owner)(i as nat) == work_expert(input, item)
            &&& run.sorted_x[i] == seq![input.token_values[token as int]]
            &&& run.out[i].len() == 1
            &&& ex09_fused_moe::expert_apply(
                (run.owner)(i as nat), run.sorted_x[i],
            ) == seq![expert_apply(work_expert(input, item), input.token_values[token as int])]
            &&& input.fused_values[i]
                == input.weights[token as int][slot as int] * run.out[i][0]
        }
}

/// The low-level component contract consumed by composition: a modeled Ex09
/// DSL execution plus the explicit representation map to shared MoE semantics.
pub open spec fn Ex09KernelContract(
    input: MoeInput,
    run: Ex09FusedRun,
) -> bool {
    &&& run.experts_per_rank >= 1
    &&& ex09_fused_moe::fused_moe_precondition_offsets(
        run.offsets, run.experts_per_rank, run.out.len(),
    )
    &&& ex09_dsl_execution(run)
    &&& ex09_run_matches_shared_layout(input, run)
}

/// Ex09's component postcondition implies the exact row contract used by the
/// shared fused-kernel refinement.
pub proof fn ex09_establishes_fused_rows_correct(
    input: MoeInput,
    run: Ex09FusedRun,
)
    requires
        RoutingConsistent(input),
        Ex09KernelContract(input, run),
    ensures fused_rows_correct(input),
{
    ex09_dsl_execution_establishes_postcondition(run);
    assert(input.fused_values.len() == input.routing_order.len());
    assert forall|i: int| #![trigger input.fused_values[i]]
        0 <= i < input.routing_order.len()
        implies input.fused_values[i]
            == work_value(input, input.routing_order[i]) by {
        ex09_fused_moe::f1_offsets_cover_range(
            run.offsets, run.experts_per_rank, run.out.len(), i as nat,
        );
        let item = input.routing_order[i];
        let token = work_token(input, item);
        let slot = work_slot(input, item);
        assert(run.out[i]
            == ex09_fused_moe::expert_apply(
                (run.owner)(i as nat), run.sorted_x[i],
            ));
        assert(run.out[i]
            == seq![expert_apply(
                work_expert(input, item), input.token_values[token as int],
            )]);
        assert(run.out[i][0]
            == expert_apply(
                work_expert(input, item), input.token_values[token as int],
            ));
        assert(work_expert(input, item)
            == input.expert_ids[token as int][slot as int]);
    }
}

} // verus!
