// Final all-gather for the DP=1 dispatch path.
//
// Proved: source-local scatter -> contiguous output shards -> rank-ordered
// concatenation -> the complete MoE result on EVERY participating rank.
// Trust boundary: AllGatherContract specifies the communication library;
// this file does not verify NCCL or the Python/FFI implementation. Like the
// existing dispatch refinement, it models exact scalar token values and an
// evenly partitionable batch (including an empty batch), not padding/trim.

use vstd::prelude::*;
use crate::composition_core::*;
use crate::kernel_refinement::*;
use crate::schedule_refinement::*;
use vstd::arithmetic::div_mod::*;

verus! {

/// Scatter only records which actually returned to this source rank.
/// Records at other locations make no contribution to its output buffer.
pub open spec fn source_local_scatter(
    input: MoeInput, records: Seq<AllToAllRecord>, source: Rank,
    token: nat, n: nat,
) -> int
    recommends n <= records.len(),
    decreases n,
{
    if n == 0 { 0int } else {
        let record = records[(n - 1) as int];
        source_local_scatter(input, records, source, token, (n - 1) as nat)
            + if record.location == source && work_token(input, record.item) == token {
                record.value
            } else { 0int }
    }
}

/// Source r owns precisely its contiguous token stripe. This is computed
/// from returned records, NOT defined as a slice of the desired MoE output.
pub open spec fn all_to_all_local_output(
    input: MoeInput, use_fused: bool, source: Rank,
) -> Tensor {
    let rows = tokens_per_source_rank(input);
    let records = all_to_all_returned(input, use_fused);
    Tensor { content: Seq::new(rows, |i: int|
        source_local_scatter(input, records, source,
            (source * rows + i) as nat, records.len())) }
}

pub open spec fn all_to_all_local_outputs(input: MoeInput, use_fused: bool) -> RankTensorView {
    |r: Rank| all_to_all_local_output(input, use_fused, r)
}

/// The local location filter drops no contribution belonging to this token.
pub proof fn lemma_source_scatter_matches_global(
    input: MoeInput, records: Seq<AllToAllRecord>, source: Rank,
    token: nat, n: nat,
)
    requires
        n <= records.len(),
        forall|i: int| 0 <= i < records.len()
            && work_token(input, records[i].item) == token
            ==> records[i].location == source,
    ensures source_local_scatter(input, records, source, token, n)
        == fold_values_for_token(input, record_items(records), record_values(records), token, n),
    decreases n,
{
    if n > 0 {
        lemma_source_scatter_matches_global(input, records, source, token, (n - 1) as nat);
        if work_token(input, records[(n - 1) as int].item) == token {
            assert(records[(n - 1) as int].location == source);
        }
    }
}

/// No output-spec or local-output correctness premise is assumed here:
/// ownership is obtained from the existing dispatch/compute/combine theorem.
pub proof fn theorem_local_output_is_pipeline_slice(
    input: MoeInput, use_fused: bool, source: Rank,
)
    requires
        all_to_all_partitionable(input),
        ExpertPartitioned(input, input.expert_parallel_size),
        RoutingConsistent(input),
        !use_fused || fused_rows_correct(input),
        source < input.expert_parallel_size,
    ensures
        all_to_all_local_output(input, use_fused, source).content
            == all_to_all_pipeline_output(input, use_fused).content.subrange(
                (source * tokens_per_source_rank(input)) as int,
                ((source + 1) * tokens_per_source_rank(input)) as int),
{
    theorem_all_to_all_dispatch_compute_combine_round_trip(input, use_fused);
    let rows = tokens_per_source_rank(input);
    let world = input.expert_parallel_size;
    let total = input.token_values.len();
    lemma_fundamental_div_mod(total as int, world as int);
    assert(rows == total / world);
    assert(total % world == 0);
    assert(total == world * rows);
    assert(total == rows * world) by (nonlinear_arith) requires total == world * rows;
    assert(source * rows <= (source + 1) * rows <= total) by (nonlinear_arith)
        requires source < world, total == rows * world;
    assert((source + 1) * rows - source * rows == rows) by (nonlinear_arith);
    let records = all_to_all_returned(input, use_fused);
    let local = all_to_all_local_output(input, use_fused, source).content;
    let full = all_to_all_pipeline_output(input, use_fused).content;
    assert forall|i: int| 0 <= i < rows implies
        local[i] == full[(source * rows + i) as int] by {
        let token = (source * rows + i) as nat;
        assert(rows > 0);
        assert(source * rows >= 0) by (nonlinear_arith);
        assert(token == source * rows + i);
        assert(token < total) by (nonlinear_arith)
            requires source < world, total == rows * world, 0 <= i < rows, token == source * rows + i;
        lemma_fundamental_div_mod_converse(token as int, rows as int, source as int, i);
        assert forall|j: int| 0 <= j < records.len()
            && work_token(input, records[j].item) == token
            implies records[j].location == source by {
            assert(records[j].location == records[j].source);
            assert(records[j].source == token_source_rank(input, records[j].item));
        }
        lemma_source_scatter_matches_global(input, records, source, token, records.len());
    }
    assert(local =~= full.subrange((source * rows) as int, ((source + 1) * rows) as int));
}

/// Rank-order concatenation, matching all_gather_into_tensor for equal shards.
pub open spec fn gather_prefix(shards: RankTensorView, count: nat) -> Seq<int>
    decreases count,
{
    if count == 0 { Seq::empty() } else {
        gather_prefix(shards, (count - 1) as nat) + shards((count - 1) as nat).content
    }
}

/// Explicit API postcondition, NOT the desired MoE correctness result.
/// Group ranks here are 0..world, in the same order as token partitioning.
/// The API must deliver every source shard, in rank order, to every receiver.
pub open spec fn AllGatherContract(
    shards: RankTensorView, outputs: RankTensorView, world: nat,
) -> bool {
    forall|r: Rank| r < world ==> #[trigger] outputs(r).content == gather_prefix(shards, world)
}

/// Pure concatenation theorem, independent of routing or expert arithmetic.
pub proof fn lemma_gather_contiguous_slices(
    full: Seq<int>, shards: RankTensorView, rows: nat, world: nat, count: nat,
)
    requires
        full.len() == rows * world,
        count <= world,
        forall|r: Rank| r < world ==> #[trigger] shards(r).content
            == full.subrange((r * rows) as int, ((r + 1) * rows) as int),
    ensures gather_prefix(shards, count) == full.subrange(0, (count * rows) as int),
    decreases count,
{
    assert(count * rows <= full.len()) by (nonlinear_arith)
        requires count <= world, full.len() == rows * world;
    if count == 0 {
        assert(count * rows == 0) by (nonlinear_arith) requires count == 0;
        assert(gather_prefix(shards, count) =~= full.subrange(0, 0));
    } else {
        let prev = (count - 1) as nat;
        lemma_gather_contiguous_slices(full, shards, rows, world, prev);
        assert(shards(prev).content == full.subrange((prev * rows) as int, ((prev + 1) * rows) as int));
        assert(prev * rows + rows == count * rows) by (nonlinear_arith) requires prev + 1 == count;
        assert(gather_prefix(shards, count) =~= full.subrange(0, (count * rows) as int));
    }
}

/// Final result: every participating receiver has the COMPLETE MoE output.
/// This composes a proved local-scatter/shard relation with the all-gather
/// library contract, rather than assuming local shards already equal spec.
pub proof fn theorem_all_to_all_all_gather_replicates_spec(
    input: MoeInput, use_fused: bool, outputs: RankTensorView,
)
    requires
        all_to_all_partitionable(input),
        ExpertPartitioned(input, input.expert_parallel_size),
        RoutingConsistent(input),
        !use_fused || fused_rows_correct(input),
        AllGatherContract(all_to_all_local_outputs(input, use_fused), outputs,
            input.expert_parallel_size),
    ensures forall|r: Rank| r < input.expert_parallel_size
        ==> semantic_eq(#[trigger] outputs(r), moe_spec(input)),
{
    let rows = tokens_per_source_rank(input);
    let world = input.expert_parallel_size;
    let full = all_to_all_pipeline_output(input, use_fused).content;
    let shards = all_to_all_local_outputs(input, use_fused);
    lemma_fundamental_div_mod(input.token_values.len() as int, world as int);
    assert(rows == input.token_values.len() / world);
    assert(input.token_values.len() % world == 0);
    assert(full.len() == world * rows);
    assert(full.len() == rows * world) by (nonlinear_arith) requires full.len() == world * rows;
    assert forall|r: Rank| r < world implies #[trigger] shards(r).content
        == full.subrange((r * rows) as int, ((r + 1) * rows) as int) by {
        theorem_local_output_is_pipeline_slice(input, use_fused, r);
    }
    lemma_gather_contiguous_slices(full, shards, rows, world, world);
    assert(full.subrange(0, full.len() as int) =~= full);
    theorem_all_to_all_pipeline_refines_routed_execution(input, use_fused);
    e2_permuted_refines_spec(input);
    assert forall|r: Rank| r < world implies semantic_eq(#[trigger] outputs(r), moe_spec(input)) by {
        assert(outputs(r).content == gather_prefix(shards, world));
        assert(outputs(r) == all_to_all_pipeline_output(input, use_fused));
    }
}

/// DP=1 integration: the actual group-indexed output view after all-gather
/// equals both the shared single-GPU spec and the existing all-reduce model.
pub proof fn theorem_dp1_gathered_output_equiv_all_reduce(
    input: MoeInput, input_view: RankTensorView, outputs: RankTensorView,
    world_size: nat, tp_size: nat, dp_size: nat, ep_size: nat,
    tp_group: Group, ep_group: Group, use_fused: bool,
)
    requires
        valid_dp1_topology(world_size, tp_size, dp_size, ep_size, tp_group, ep_group),
        ReplicatedInput(input, input_view, tp_group),
        ExpertPartitioned(input, world_size),
        all_to_all_partitionable(input),
        RoutingConsistent(input),
        !use_fused || fused_rows_correct(input),
        AllGatherContract(all_to_all_local_outputs(input, use_fused), outputs, world_size),
    ensures forall|r: Rank| tp_group.contains(r) ==>
        semantic_eq(#[trigger] outputs(r), moe_spec(input))
        && semantic_eq(outputs(r), all_reduce_forward(
            input, input_view, ep_group, world_size, use_fused)),
{
    theorem_all_to_all_all_gather_replicates_spec(input, use_fused, outputs);
    lemma_dp1_replication_transfers_to_ep(input, input_view, world_size,
        tp_size, dp_size, ep_size, tp_group, ep_group);
    l6_all_reduce_refines_spec(input, input_view, ep_group, world_size, use_fused);
}

} // verus!
