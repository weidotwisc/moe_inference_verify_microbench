// Exact all-reduce/all-to-all refinements and the DP=1 schedule-swap theorem.

use vstd::prelude::*;
use crate::composition_core::*;
use crate::kernel_refinement::*;

verus! {

pub type Rank = nat;
pub type Group = Set<Rank>;
pub type RankTensorView = spec_fn(Rank) -> Tensor;

pub open spec fn stored_tensor_view(tensor_id: nat) -> RankTensorView {
    |rank: Rank| tensor_on(tensor_id, rank)
}

/// The contiguous expert placement used by the implementation.  Invalid
/// expert ids map to rank zero only to keep the specification total; routing
/// well-formedness excludes that branch during execution.
pub open spec fn expert_owned_by_rank(
    input: MoeInput,
    rank: Rank,
    expert: ExpertId,
) -> bool {
    rank * input.experts_per_rank <= expert
        && expert < (rank + 1) * input.experts_per_rank
}

pub open spec fn expert_owner(input: MoeInput, expert: ExpertId) -> Rank {
    if input.experts_per_rank > 0 && expert < input.num_experts {
        choose|rank: Rank|
            rank < input.expert_parallel_size
                && #[trigger] expert_owned_by_rank(input, rank, expert)
    } else {
        0nat
    }
}

pub open spec fn ExpertPartitioned(input: MoeInput, world_size: nat) -> bool {
    &&& world_size > 0
    &&& input.expert_parallel_size == world_size
    &&& forall|expert: ExpertId| #![auto]
        expert_owner(input, expert) < world_size
}

pub open spec fn work_expert(input: MoeInput, item: WorkItem) -> ExpertId {
    if required_work_item(input, item) && input.top_k > 0 {
        input.expert_ids[
            work_token(input, item) as int
        ][work_slot(input, item) as int]
    } else {
        0nat
    }
}

/// One rank's contribution to one routed work item.  Because `expert_owner`
/// is a function, exactly one rank contributes the value and all other ranks
/// contribute zero.
pub open spec fn rank_local_value(
    input: MoeInput,
    rank: Rank,
    item: WorkItem,
    value: int,
) -> int {
    if expert_owner(input, work_expert(input, item)) == rank {
        value
    } else {
        0int
    }
}

/// The elementwise SUM performed by all-reduce, written as a finite sum of
/// rank-local contributions for one work item.
pub open spec fn sum_rank_local_values(
    input: MoeInput,
    item: WorkItem,
    value: int,
    rank_count: nat,
) -> int
    decreases rank_count,
{
    if rank_count == 0 {
        0int
    } else {
        sum_rank_local_values(input, item, value, (rank_count - 1) as nat)
            + rank_local_value(
                input, (rank_count - 1) as nat, item, value,
            )
    }
}

pub proof fn lemma_sum_rank_local_values_prefix(
    input: MoeInput,
    item: WorkItem,
    value: int,
    rank_count: nat,
)
    ensures sum_rank_local_values(input, item, value, rank_count)
        == if expert_owner(input, work_expert(input, item)) < rank_count {
            value
        } else {
            0int
        },
    decreases rank_count,
{
    if rank_count > 0 {
        lemma_sum_rank_local_values_prefix(
            input, item, value, (rank_count - 1) as nat,
        );
    }
}

/// A complete, disjoint expert placement makes all-reduce reconstruct each
/// logical work contribution exactly once.
pub proof fn lemma_all_reduce_reconstructs_work_value(
    input: MoeInput,
    world_size: nat,
    item: WorkItem,
    value: int,
)
    requires ExpertPartitioned(input, world_size),
    ensures sum_rank_local_values(input, item, value, world_size) == value,
{
    lemma_sum_rank_local_values_prefix(input, item, value, world_size);
}

/// Every rank in a non-empty group observes this exact MoE input tensor.
pub open spec fn ReplicatedInput(
    input: MoeInput,
    input_view: RankTensorView,
    group: Group,
) -> bool {
    let input_tensor = Tensor { content: input.token_values };
    &&& exists|r: Rank| group.contains(r)
    &&& forall|r: Rank| group.contains(r)
        ==> input_view(r) == input_tensor
}

/// DP=1 makes both the TP and EP groups equal to the world group.
pub open spec fn valid_dp1_topology(
    world_size: nat,
    tp_size: nat,
    dp_size: nat,
    ep_size: nat,
    tp_group: Group,
    ep_group: Group,
) -> bool {
    &&& world_size > 0
    &&& tp_size > 0
    &&& ep_size > 0
    &&& dp_size == 1
    &&& tp_size * dp_size == ep_size
    &&& ep_size == world_size
    &&& forall|r: Rank| #![auto]
        tp_group.contains(r) == (r < world_size)
    &&& forall|r: Rank| #![auto]
        ep_group.contains(r) == (r < world_size)
}

pub proof fn lemma_dp1_replication_transfers_to_ep(
    input: MoeInput,
    input_view: RankTensorView,
    world_size: nat,
    tp_size: nat,
    dp_size: nat,
    ep_size: nat,
    tp_group: Group,
    ep_group: Group,
)
    requires
        valid_dp1_topology(
            world_size, tp_size, dp_size, ep_size, tp_group, ep_group,
        ),
        ReplicatedInput(input, input_view, tp_group),
    ensures ReplicatedInput(input, input_view, ep_group),
{
    assert(ep_group.contains(0nat));
    assert(exists|r: Rank| ep_group.contains(r)) by {
        assert(ep_group.contains(0nat));
    }
    assert forall|r: Rank| ep_group.contains(r)
        implies input_view(r)
            == Tensor { content: input.token_values } by {
        assert(r < world_size);
        assert(tp_group.contains(r));
    }
}

pub open spec fn all_reduce_order(
    input: MoeInput,
    use_fused: bool,
) -> Seq<WorkItem> {
    if use_fused {
        input.routing_order
    } else {
        canonical_work(input)
    }
}

pub open spec fn all_reduce_source_values(
    input: MoeInput,
    use_fused: bool,
) -> Seq<int> {
    if use_fused {
        input.fused_values
    } else {
        expected_values(input, canonical_work(input))
    }
}

pub open spec fn all_reduced_values(
    input: MoeInput,
    world_size: nat,
    order: Seq<WorkItem>,
    values: Seq<int>,
) -> Seq<int> {
    Seq::new(order.len(), |i: int|
        sum_rank_local_values(input, order[i], values[i], world_size)
    )
}

pub proof fn lemma_all_reduced_values_equal_source(
    input: MoeInput,
    world_size: nat,
    order: Seq<WorkItem>,
    values: Seq<int>,
)
    requires
        ExpertPartitioned(input, world_size),
        values.len() == order.len(),
    ensures all_reduced_values(input, world_size, order, values) == values,
{
    assert(all_reduced_values(input, world_size, order, values) =~= values) by {
        assert forall|i: int| #![auto] 0 <= i < order.len() implies
            all_reduced_values(input, world_size, order, values)[i]
                == values[i] by {
            lemma_all_reduce_reconstructs_work_value(
                input, world_size, order[i], values[i],
            );
        }
    }
}

/// The partial output accumulated by one rank for one token.
pub open spec fn rank_partial_for_token(
    input: MoeInput,
    order: Seq<WorkItem>,
    values: Seq<int>,
    rank: Rank,
    token: nat,
    n: nat,
) -> int
    recommends n <= order.len(), n <= values.len(),
    decreases n,
{
    if n == 0 {
        0int
    } else {
        rank_partial_for_token(
            input, order, values, rank, token, (n - 1) as nat,
        ) + if work_token(input, order[(n - 1) as int]) == token {
            rank_local_value(
                input, rank, order[(n - 1) as int], values[(n - 1) as int],
            )
        } else {
            0int
        }
    }
}

/// Elementwise SUM of the rank-local partial output tensors.
pub open spec fn sum_rank_partials_for_token(
    input: MoeInput,
    order: Seq<WorkItem>,
    values: Seq<int>,
    token: nat,
    n: nat,
    rank_count: nat,
) -> int
    recommends n <= order.len(), n <= values.len(),
    decreases rank_count,
{
    if rank_count == 0 {
        0int
    } else {
        sum_rank_partials_for_token(
            input, order, values, token, n, (rank_count - 1) as nat,
        ) + rank_partial_for_token(
            input, order, values, (rank_count - 1) as nat, token, n,
        )
    }
}

pub open spec fn all_reduce_partial_outputs(
    input: MoeInput,
    world_size: nat,
    order: Seq<WorkItem>,
    values: Seq<int>,
) -> Tensor {
    Tensor {
        content: Seq::new(input.token_values.len(), |token: int|
            sum_rank_partials_for_token(
                input, order, values, token as nat, order.len(), world_size,
            )
        ),
    }
}

proof fn lemma_sum_rank_partials_zero(
    input: MoeInput,
    order: Seq<WorkItem>,
    values: Seq<int>,
    token: nat,
    rank_count: nat,
)
    ensures sum_rank_partials_for_token(
        input, order, values, token, 0nat, rank_count,
    ) == 0int,
    decreases rank_count,
{
    if rank_count > 0 {
        lemma_sum_rank_partials_zero(
            input, order, values, token, (rank_count - 1) as nat,
        );
    }
}

proof fn lemma_sum_rank_partials_step(
    input: MoeInput,
    order: Seq<WorkItem>,
    values: Seq<int>,
    token: nat,
    n: nat,
    rank_count: nat,
)
    requires
        0 < n <= order.len(),
        n <= values.len(),
    ensures sum_rank_partials_for_token(
        input, order, values, token, n, rank_count,
    ) == sum_rank_partials_for_token(
        input, order, values, token, (n - 1) as nat, rank_count,
    ) + if work_token(input, order[(n - 1) as int]) == token {
        sum_rank_local_values(
            input,
            order[(n - 1) as int],
            values[(n - 1) as int],
            rank_count,
        )
    } else {
        0int
    },
    decreases rank_count,
{
    if rank_count > 0 {
        lemma_sum_rank_partials_step(
            input, order, values, token, n, (rank_count - 1) as nat,
        );
    }
}

proof fn lemma_rank_partials_equal_reduced_fold(
    input: MoeInput,
    world_size: nat,
    order: Seq<WorkItem>,
    values: Seq<int>,
    token: nat,
    n: nat,
)
    requires
        n <= order.len(),
        values.len() == order.len(),
    ensures sum_rank_partials_for_token(
        input, order, values, token, n, world_size,
    ) == fold_values_for_token(
        input,
        order,
        all_reduced_values(input, world_size, order, values),
        token,
        n,
    ),
    decreases n,
{
    if n == 0 {
        lemma_sum_rank_partials_zero(
            input, order, values, token, world_size,
        );
    } else {
        let previous_n = (n - 1) as nat;
        let item = order[previous_n as int];
        let reduced = all_reduced_values(
            input, world_size, order, values,
        );
        lemma_rank_partials_equal_reduced_fold(
            input, world_size, order, values, token, previous_n,
        );
        lemma_sum_rank_partials_step(
            input, order, values, token, n, world_size,
        );
        assert(reduced.len() == order.len());
        assert(reduced[previous_n as int]
                == sum_rank_local_values(
                    input, item, values[previous_n as int], world_size,
                ));
        reveal_with_fuel(fold_values_for_token, 1);
        assert(fold_values_for_token(
            input, order, reduced, token, n,
        ) == fold_values_for_token(
            input, order, reduced, token, previous_n,
        ) + if work_token(input, item) == token {
            reduced[previous_n as int]
        } else {
            0int
        });
        assert(sum_rank_partials_for_token(
            input, order, values, token, n, world_size,
        ) == sum_rank_partials_for_token(
            input, order, values, token, previous_n, world_size,
        ) + if work_token(input, item) == token {
            sum_rank_local_values(
                input, item, values[previous_n as int], world_size,
            )
        } else {
            0int
        });
        if work_token(input, item) == token {
            assert(sum_rank_local_values(
                input, item, values[previous_n as int], world_size,
            ) == reduced[previous_n as int]);
        }
        assert(sum_rank_partials_for_token(
            input, order, values, token, n, world_size,
        ) == fold_values_for_token(
            input, order, reduced, token, n,
        ));
    }
}

/// Moving the rank-wise SUM outside the token scatter does not change the
/// result.  This connects the actual rank-local-partial execution shape to
/// the per-work-item reduction used by the ownership lemma above.
pub proof fn lemma_all_reduce_partial_outputs_normalize(
    input: MoeInput,
    world_size: nat,
    order: Seq<WorkItem>,
    values: Seq<int>,
)
    requires values.len() == order.len(),
    ensures semantic_eq(
        all_reduce_partial_outputs(input, world_size, order, values),
        execute_values(
            input,
            order,
            all_reduced_values(input, world_size, order, values),
        ),
    ),
{
    assert(all_reduce_partial_outputs(input, world_size, order, values).content
        =~= execute_values(
            input,
            order,
            all_reduced_values(input, world_size, order, values),
        ).content) by {
        assert forall|token: int| #![auto]
            0 <= token < input.token_values.len() implies
            all_reduce_partial_outputs(
                input, world_size, order, values,
            ).content[token] == execute_values(
                input,
                order,
                all_reduced_values(input, world_size, order, values),
            ).content[token] by {
            lemma_rank_partials_equal_reduced_fold(
                input, world_size, order, values, token as nat, order.len(),
            );
        }
    }
}

/// The core replicated-EP correctness result.  Each routed work item is
/// assigned to exactly one expert owner, each owner scatter-adds its local
/// contributions into a rank-local partial tensor, and an elementwise SUM of
/// those partial tensors is exactly the canonical MoE output.
pub proof fn theorem_rank_local_partials_sum_to_moe_spec(
    input: MoeInput,
    world_size: nat,
)
    requires ExpertPartitioned(input, world_size),
    ensures semantic_eq(
        all_reduce_partial_outputs(
            input,
            world_size,
            canonical_work(input),
            expected_values(input, canonical_work(input)),
        ),
        moe_spec(input),
    ),
{
    let order = canonical_work(input);
    let values = expected_values(input, order);
    let partials = all_reduce_partial_outputs(
        input, world_size, order, values,
    );
    let reduced = all_reduced_values(
        input, world_size, order, values,
    );

    lemma_all_reduce_partial_outputs_normalize(
        input, world_size, order, values,
    );
    lemma_all_reduced_values_equal_source(
        input, world_size, order, values,
    );
    lemma_execute_expected_values(input, order);
    lemma_semantic_eq_trans(
        partials,
        execute_values(input, order, values),
        execute_plan(input, order),
    );
}

/// Exact DP=1 all-reduce execution: each rank first scatter-adds the
/// contributions of its owned experts into a local partial tensor, then the
/// collective sums those tensors elementwise.
pub open spec fn all_reduce_forward(
    input: MoeInput,
    input_view: RankTensorView,
    ep_group: Group,
    world_size: nat,
    use_fused: bool,
) -> Tensor {
    if ReplicatedInput(input, input_view, ep_group) {
        all_reduce_partial_outputs(
            input,
            world_size,
            all_reduce_order(input, use_fused),
            all_reduce_source_values(input, use_fused),
        )
    } else {
        Tensor { content: Seq::empty() }
    }
}

/// The concrete dispatch implementation first partitions the replicated token
/// rows into equal contiguous source-rank slices.
pub open spec fn all_to_all_partitionable(input: MoeInput) -> bool {
    input.expert_parallel_size > 0
        && input.token_values.len() % input.expert_parallel_size == 0
}

pub open spec fn tokens_per_source_rank(input: MoeInput) -> nat {
    if input.expert_parallel_size == 0 {
        0nat
    } else {
        input.token_values.len() / input.expert_parallel_size
    }
}

/// Source rank of a routed record under the implementation's contiguous
/// token partition.  The guarded fallback only makes the spec total; valid
/// routed records under `all_to_all_partitionable` take the quotient branch.
pub open spec fn token_source_rank(
    input: MoeInput,
    item: WorkItem,
) -> Rank {
    let rows = tokens_per_source_rank(input);
    if rows > 0 {
        let candidate = work_token(input, item) / rows;
        if candidate < input.expert_parallel_size {
            candidate
        } else {
            0nat
        }
    } else {
        0nat
    }
}

/// A global-view record used to model the two all-to-all phases.  Network
/// transfer changes `location`; it never changes the work-item identity,
/// source rank, or computed value.
pub struct AllToAllRecord {
    pub item: WorkItem,
    pub source: Rank,
    pub location: Rank,
    pub value: int,
}

pub open spec fn dispatch_records(
    input: MoeInput,
    order: Seq<WorkItem>,
) -> Seq<AllToAllRecord> {
    Seq::new(order.len(), |i: int| AllToAllRecord {
        item: order[i],
        source: token_source_rank(input, order[i]),
        location: expert_owner(input, work_expert(input, order[i])),
        value: 0int,
    })
}

pub open spec fn compute_dispatched_records(
    dispatched: Seq<AllToAllRecord>,
    values: Seq<int>,
) -> Seq<AllToAllRecord> {
    Seq::new(dispatched.len(), |i: int| AllToAllRecord {
        item: dispatched[i].item,
        source: dispatched[i].source,
        location: dispatched[i].location,
        value: if i < values.len() { values[i] } else { 0int },
    })
}

pub open spec fn combine_records(
    computed: Seq<AllToAllRecord>,
) -> Seq<AllToAllRecord> {
    Seq::new(computed.len(), |i: int| AllToAllRecord {
        item: computed[i].item,
        source: computed[i].source,
        location: computed[i].source,
        value: computed[i].value,
    })
}

pub open spec fn record_items(records: Seq<AllToAllRecord>) -> Seq<WorkItem> {
    Seq::new(records.len(), |i: int| records[i].item)
}

pub open spec fn record_values(records: Seq<AllToAllRecord>) -> Seq<int> {
    Seq::new(records.len(), |i: int| records[i].value)
}

pub open spec fn all_to_all_source_values(
    input: MoeInput,
    use_fused: bool,
) -> Seq<int> {
    if use_fused {
        input.fused_values
    } else {
        expected_values(input, input.routing_order)
    }
}

pub open spec fn all_to_all_dispatched(
    input: MoeInput,
) -> Seq<AllToAllRecord> {
    dispatch_records(input, input.routing_order)
}

pub open spec fn all_to_all_computed(
    input: MoeInput,
    use_fused: bool,
) -> Seq<AllToAllRecord> {
    compute_dispatched_records(
        all_to_all_dispatched(input),
        all_to_all_source_values(input, use_fused),
    )
}

pub open spec fn all_to_all_returned(
    input: MoeInput,
    use_fused: bool,
) -> Seq<AllToAllRecord> {
    combine_records(all_to_all_computed(input, use_fused))
}

/// Dispatch sends every record to its expert owner; combine returns the same
/// record and computed value to its original token-owning rank.  Thus the two
/// network phases neither drop nor duplicate routed work.
pub proof fn theorem_all_to_all_dispatch_compute_combine_round_trip(
    input: MoeInput,
    use_fused: bool,
)
    requires
        all_to_all_partitionable(input),
        ExpertPartitioned(input, input.expert_parallel_size),
        RoutingConsistent(input),
        !use_fused || fused_rows_correct(input),
    ensures
        record_items(all_to_all_returned(input, use_fused))
            == input.routing_order,
        record_values(all_to_all_returned(input, use_fused))
            == expected_values(input, input.routing_order),
        record_items(all_to_all_returned(input, use_fused)).to_multiset()
            == canonical_work(input).to_multiset(),
        forall|i: int| #![trigger all_to_all_computed(input, use_fused)[i]]
            0 <= i < all_to_all_computed(input, use_fused).len() ==> {
                let record = all_to_all_computed(input, use_fused)[i];
                &&& record.item == input.routing_order[i]
                &&& record.location
                    == expert_owner(input, work_expert(input, record.item))
                &&& record.location < input.expert_parallel_size
                &&& record.value == work_value(input, record.item)
            },
        forall|i: int| #![trigger all_to_all_returned(input, use_fused)[i]]
            0 <= i < all_to_all_returned(input, use_fused).len() ==> {
                let record = all_to_all_returned(input, use_fused)[i];
                &&& record.item == input.routing_order[i]
                &&& record.location == record.source
                &&& record.source < input.expert_parallel_size
                &&& record.value == work_value(input, record.item)
            },
{
    broadcast use vstd::seq_lib::group_to_multiset_ensures;
    broadcast use vstd::multiset::group_multiset_axioms;

    let order = input.routing_order;
    let source_values = all_to_all_source_values(input, use_fused);
    let dispatched = all_to_all_dispatched(input);
    let computed = all_to_all_computed(input, use_fused);
    let returned = all_to_all_returned(input, use_fused);

    if use_fused {
        lemma_fused_values_equal_expected(input);
    }
    theorem_routing_work_conservation(input);
    assert(source_values == expected_values(input, order));
    assert(source_values.len() == order.len());

    assert(record_items(returned) =~= order) by {
        assert forall|i: int| #![auto]
            0 <= i < returned.len()
                implies record_items(returned)[i] == order[i] by {
        }
    }
    assert(record_values(returned) =~= source_values) by {
        assert forall|i: int| #![auto]
            0 <= i < returned.len()
                implies record_values(returned)[i] == source_values[i] by {
        }
    }
    assert forall|i: int| #![trigger computed[i]]
        0 <= i < computed.len() implies {
            let record = computed[i];
            &&& record.item == order[i]
            &&& record.location
                == expert_owner(input, work_expert(input, record.item))
            &&& record.location < input.expert_parallel_size
            &&& record.value == work_value(input, record.item)
        } by {
        assert(dispatched[i].item == order[i]);
        assert(computed[i].item == order[i]);
        assert(required_work_item(input, order[i])) by {
            assert(order.contains(order[i]));
            assert(order.to_multiset().contains(order[i]));
            assert(times_processed(input, input.routing_order, order[i]) > 0);
        }
        assert(computed[i].location
            == expert_owner(input, work_expert(input, computed[i].item)));
        assert(computed[i].location < input.expert_parallel_size);
        assert(computed[i].value == source_values[i]);
        assert(source_values[i] == work_value(input, order[i]));
    }
    assert forall|i: int| #![trigger returned[i]]
        0 <= i < returned.len() implies {
            let record = returned[i];
            &&& record.item == order[i]
            &&& record.location == record.source
            &&& record.source < input.expert_parallel_size
            &&& record.value == work_value(input, record.item)
        } by {
        assert(returned[i].item == order[i]);
        assert(returned[i].location == returned[i].source);
        assert(returned[i].source
            == token_source_rank(input, order[i]));
        assert(returned[i].source < input.expert_parallel_size);
        assert(returned[i].value == source_values[i]);
        assert(source_values[i] == work_value(input, order[i]));
    }
}

pub open spec fn all_to_all_pipeline_output(
    input: MoeInput,
    use_fused: bool,
) -> Tensor {
    let returned = all_to_all_returned(input, use_fused);
    execute_values(input, record_items(returned), record_values(returned))
}

/// The explicit dispatch/compute/combine pipeline implements the routed MoE
/// execution.  This is the model-level refinement; the collective library's
/// conformance to the record-transfer semantics remains an API boundary.
pub proof fn theorem_all_to_all_pipeline_refines_routed_execution(
    input: MoeInput,
    use_fused: bool,
)
    requires
        all_to_all_partitionable(input),
        ExpertPartitioned(input, input.expert_parallel_size),
        RoutingConsistent(input),
        !use_fused || fused_rows_correct(input),
    ensures semantic_eq(
        all_to_all_pipeline_output(input, use_fused),
        permuted_forward(input),
    ),
{
    theorem_all_to_all_dispatch_compute_combine_round_trip(
        input, use_fused,
    );
    lemma_execute_expected_values(input, input.routing_order);
}

/// Exact output of the implementation that dispatches routed work to each
/// expert owner, computes there, returns results to token owners, and finally
/// scatter-adds them into token rows.
pub open spec fn all_to_all_forward(
    input: MoeInput,
    input_view: RankTensorView,
    ep_group: Group,
    use_fused: bool,
) -> Tensor {
    if ReplicatedInput(input, input_view, ep_group)
        && all_to_all_partitionable(input) {
        all_to_all_pipeline_output(input, use_fused)
    } else {
        Tensor { content: Seq::empty() }
    }
}

pub open spec fn scheduled_forward(
    input: MoeInput,
    input_view: RankTensorView,
    group: Group,
    world_size: nat,
    use_all_reduce: bool,
    use_fused: bool,
) -> Tensor {
    if use_all_reduce {
        all_reduce_forward(input, input_view, group, world_size, use_fused)
    } else {
        all_to_all_forward(input, input_view, group, use_fused)
    }
}

/// L6: the DP=1 all-reduce schedule refines the shared exact MoE semantics.
pub proof fn l6_all_reduce_refines_spec(
    input: MoeInput,
    input_view: RankTensorView,
    ep_group: Group,
    world_size: nat,
    use_fused: bool,
)
    requires
        ReplicatedInput(input, input_view, ep_group),
        ExpertPartitioned(input, world_size),
        RoutingConsistent(input),
        !use_fused || fused_rows_correct(input),
    ensures semantic_eq(
        all_reduce_forward(
            input, input_view, ep_group, world_size, use_fused,
        ),
        moe_spec(input),
    ),
{
    lemma_all_reduce_partial_outputs_normalize(
        input,
        world_size,
        all_reduce_order(input, use_fused),
        all_reduce_source_values(input, use_fused),
    );
    lemma_all_reduced_values_equal_source(
        input,
        world_size,
        all_reduce_order(input, use_fused),
        all_reduce_source_values(input, use_fused),
    );
    if use_fused {
        f4_fused_refines_spec(input);
    } else {
        theorem_rank_local_partials_sum_to_moe_spec(input, world_size);
    }
}

/// H6: the all-to-all dispatch/combine schedule refines the same exact MoE
/// semantics.  At DP=1 its input is replicated, so this path and L6 have the
/// same logical token/expert work even though they communicate differently.
pub proof fn h6_all_to_all_refines_spec(
    input: MoeInput,
    input_view: RankTensorView,
    ep_group: Group,
    use_fused: bool,
)
    requires
        ReplicatedInput(input, input_view, ep_group),
        ExpertPartitioned(input, input.expert_parallel_size),
        all_to_all_partitionable(input),
        RoutingConsistent(input),
        !use_fused || fused_rows_correct(input),
    ensures semantic_eq(
        all_to_all_forward(input, input_view, ep_group, use_fused),
        moe_spec(input),
    ),
{
    theorem_all_to_all_pipeline_refines_routed_execution(input, use_fused);
    e2_permuted_refines_spec(input);
    lemma_semantic_eq_trans(
        all_to_all_forward(input, input_view, ep_group, use_fused),
        permuted_forward(input),
        moe_spec(input),
    );
}

/// At DP=1, all-to-all dispatch/combine can be replaced by one all-reduce.
/// Both paths refine the same exact MoE result; the theorem does not claim
/// this replacement when the input is partitioned across data-parallel ranks.
pub proof fn theorem_all_to_all_equiv_all_reduce_dp1(
    input: MoeInput,
    input_view: RankTensorView,
    world_size: nat,
    tp_size: nat,
    dp_size: nat,
    ep_size: nat,
    tp_group: Group,
    ep_group: Group,
    use_fused: bool,
)
    requires
        valid_dp1_topology(
            world_size, tp_size, dp_size, ep_size, tp_group, ep_group,
        ),
        ReplicatedInput(input, input_view, tp_group),
        ExpertPartitioned(input, world_size),
        all_to_all_partitionable(input),
        RoutingConsistent(input),
        !use_fused || fused_rows_correct(input),
    ensures semantic_eq(
        all_reduce_forward(
            input, input_view, ep_group, world_size, use_fused,
        ),
        all_to_all_forward(input, input_view, ep_group, use_fused),
    ),
{
    lemma_dp1_replication_transfers_to_ep(
        input, input_view, world_size, tp_size, dp_size, ep_size,
        tp_group, ep_group,
    );

    l6_all_reduce_refines_spec(
        input, input_view, ep_group, world_size, use_fused,
    );
    h6_all_to_all_refines_spec(input, input_view, ep_group, use_fused);
    theorem_shared_spec_implies_equiv(
        all_reduce_forward(
            input, input_view, ep_group, world_size, use_fused,
        ),
        all_to_all_forward(input, input_view, ep_group, use_fused),
        moe_spec(input),
    );
}

/// Kernel swap for either schedule, proved from E1/E2/F4.
pub proof fn theorem_python_equiv_fused_for_schedule(
    input: MoeInput,
    input_view: RankTensorView,
    ep_group: Group,
    world_size: nat,
    use_all_reduce: bool,
)
    requires
        ReplicatedInput(input, input_view, ep_group),
        ExpertPartitioned(input, world_size),
        use_all_reduce || all_to_all_partitionable(input),
        RoutingConsistent(input),
        fused_rows_correct(input),
    ensures semantic_eq(
        scheduled_forward(
            input, input_view, ep_group, world_size, use_all_reduce, false,
        ),
        scheduled_forward(
            input, input_view, ep_group, world_size, use_all_reduce, true,
        ),
    ),
{
    if use_all_reduce {
        l6_all_reduce_refines_spec(
            input, input_view, ep_group, world_size, false,
        );
        l6_all_reduce_refines_spec(
            input, input_view, ep_group, world_size, true,
        );
    } else {
        h6_all_to_all_refines_spec(input, input_view, ep_group, false);
        h6_all_to_all_refines_spec(input, input_view, ep_group, true);
    }
    theorem_shared_spec_implies_equiv(
        scheduled_forward(
            input, input_view, ep_group, world_size, use_all_reduce, false,
        ),
        scheduled_forward(
            input, input_view, ep_group, world_size, use_all_reduce, true,
        ),
        moe_spec(input),
    );
}

} // verus!
