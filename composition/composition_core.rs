// Shared exact semantic model and refinement machinery for Tier-3 composition.

use vstd::prelude::*;

verus! {

/// A tensor's flattened exact values in row-major order.
pub struct Tensor {
    pub content: Seq<int>,
}

/// Shared distributed-state observation used by the component and
/// composition layers.
pub uninterp spec fn tensor_on(tensor_id: nat, rank: nat) -> Tensor;

/// Equality in the proof's exact mathematical model.
pub open spec fn semantic_eq(x: Tensor, y: Tensor) -> bool {
    x == y
}

pub proof fn lemma_semantic_eq_refl(x: Tensor)
    ensures semantic_eq(x, x),
{}

pub proof fn lemma_semantic_eq_sym(x: Tensor, y: Tensor)
    requires semantic_eq(x, y),
    ensures semantic_eq(y, x),
{}

pub proof fn lemma_semantic_eq_trans(x: Tensor, y: Tensor, z: Tensor)
    requires
        semantic_eq(x, y),
        semantic_eq(y, z),
    ensures semantic_eq(x, z),
{}

/// Any two implementations that refine the same exact semantic output are
/// semantically equal to each other.
pub proof fn theorem_shared_spec_implies_equiv(
    a_out: Tensor,
    b_out: Tensor,
    s_out: Tensor,
)
    requires
        semantic_eq(a_out, s_out),
        semantic_eq(b_out, s_out),
    ensures semantic_eq(a_out, b_out),
{
    lemma_semantic_eq_sym(b_out, s_out);
    lemma_semantic_eq_trans(a_out, s_out, b_out);
}

// =====================================================================
// Exact MoE work semantics.
// =====================================================================

pub type ExpertId = nat;

/// One selected top-k slot.  `flat` uniquely represents the pair
/// `(token = flat / top_k, slot = flat % top_k)`.
pub struct WorkItem {
    pub flat: nat,
}

/// Exact inputs used by all schedule/kernel models.
///
/// `routing_order` is the order produced by routing/argsort.  The fused row
/// values are the exact scalar results returned for that same order.
pub struct MoeInput {
    pub token_values: Seq<int>,
    pub expert_ids: Seq<Seq<ExpertId>>,
    pub weights: Seq<Seq<int>>,
    pub top_k: nat,
    pub num_experts: nat,
    /// Uniform contiguous expert-block size used by the EP implementation.
    pub experts_per_rank: nat,
    /// Number of ranks participating in that expert placement.
    pub expert_parallel_size: nat,
    pub routing_order: Seq<WorkItem>,
    pub fused_values: Seq<int>,
}

/// Exact mathematical expert computation.  Kernel/source correspondence is
/// audited at the boundary; all composition reasoning uses this one function.
pub uninterp spec fn expert_apply(expert: ExpertId, token: int) -> int;

pub open spec fn work_count(input: MoeInput) -> nat {
    input.token_values.len() * input.top_k
}

pub open spec fn canonical_work(input: MoeInput) -> Seq<WorkItem> {
    Seq::new(work_count(input), |i: int| WorkItem { flat: i as nat })
}

pub open spec fn required_work_item(input: MoeInput, item: WorkItem) -> bool {
    item.flat < work_count(input)
}

pub open spec fn work_token(input: MoeInput, item: WorkItem) -> nat {
    if input.top_k == 0 { 0 } else { item.flat / input.top_k }
}

pub open spec fn work_slot(input: MoeInput, item: WorkItem) -> nat {
    if input.top_k == 0 { 0 } else { item.flat % input.top_k }
}

/// Shape/range conditions required to interpret every routing slot.
pub open spec fn well_formed_moe_input(input: MoeInput) -> bool {
    &&& input.top_k > 0
    &&& input.num_experts > 0
    &&& input.expert_ids.len() == input.token_values.len()
    &&& input.weights.len() == input.token_values.len()
    &&& forall|t: int| #![auto]
        0 <= t < input.token_values.len() ==> (
            input.expert_ids[t].len() == input.top_k
            && input.weights[t].len() == input.top_k
            && forall|s: int| #![trigger input.expert_ids[t][s]]
                0 <= s < input.top_k
                    ==> input.expert_ids[t][s] < input.num_experts
        )
}

/// The exact contribution attached to one `(token, top-k slot)` work item.
/// The invalid branch makes the function total; well-formed executions never
/// take it.
pub open spec fn work_value(input: MoeInput, item: WorkItem) -> int {
    if required_work_item(input, item) && input.top_k > 0 {
        let token = work_token(input, item);
        let slot = work_slot(input, item);
        input.weights[token as int][slot as int]
            * expert_apply(
                input.expert_ids[token as int][slot as int],
                input.token_values[token as int],
            )
    } else {
        0int
    }
}

/// Routing correctness: argsort/repartition may reorder work but may neither
/// drop nor duplicate any `(token, slot)` item.
pub open spec fn RoutingConsistent(input: MoeInput) -> bool {
    &&& well_formed_moe_input(input)
    &&& input.routing_order.to_multiset() == canonical_work(input).to_multiset()
}

/// Low-level fused-kernel postcondition: one exact result for each routed row.
/// This is the runtime/kernel correspondence boundary, rather than an assumed
/// whole-output equivalence.
pub open spec fn fused_rows_correct(input: MoeInput) -> bool {
    &&& input.fused_values.len() == input.routing_order.len()
    &&& forall|i: int| #![trigger input.fused_values[i]]
        0 <= i < input.routing_order.len()
            ==> input.fused_values[i] == work_value(input, input.routing_order[i])
}

pub open spec fn expected_values(input: MoeInput, order: Seq<WorkItem>) -> Seq<int> {
    Seq::new(order.len(), |i: int| work_value(input, order[i]))
}

pub open spec fn work_fold_step(
    input: MoeInput,
    token: nat,
    item: WorkItem,
    acc: int,
) -> int {
    if work_token(input, item) == token {
        work_value(input, item) + acc
    } else {
        acc
    }
}

pub open spec fn fold_work_for_token(
    input: MoeInput,
    order: Seq<WorkItem>,
    token: nat,
) -> int {
    order.fold_right(
        |item: WorkItem, acc: int| work_fold_step(input, token, item, acc),
        0int,
    )
}

pub open spec fn fold_values_for_token(
    input: MoeInput,
    order: Seq<WorkItem>,
    values: Seq<int>,
    token: nat,
    n: nat,
) -> int
    recommends n <= order.len(), n <= values.len(),
    decreases n,
{
    if n == 0 {
        0int
    } else {
        fold_values_for_token(input, order, values, token, (n - 1) as nat)
            + if work_token(input, order[(n - 1) as int]) == token {
                values[(n - 1) as int]
            } else {
                0int
            }
    }
}

/// Execute exact row values and scatter-add them to their token outputs.
pub open spec fn execute_values(
    input: MoeInput,
    order: Seq<WorkItem>,
    values: Seq<int>,
) -> Tensor {
    Tensor {
        content: Seq::new(input.token_values.len(), |t: int|
            fold_values_for_token(input, order, values, t as nat, order.len())
        ),
    }
}

/// Exact execution of a routing plan using the semantic expert function.
pub open spec fn execute_plan(input: MoeInput, order: Seq<WorkItem>) -> Tensor {
    Tensor {
        content: Seq::new(input.token_values.len(), |t: int|
            fold_work_for_token(input, order, t as nat)
        ),
    }
}

pub open spec fn moe_spec(input: MoeInput) -> Tensor {
    execute_plan(input, canonical_work(input))
}

pub open spec fn naive_forward(input: MoeInput) -> Tensor {
    execute_plan(input, canonical_work(input))
}

pub open spec fn permuted_forward(input: MoeInput) -> Tensor {
    execute_plan(input, input.routing_order)
}

pub open spec fn fused_forward(input: MoeInput) -> Tensor {
    execute_values(input, input.routing_order, input.fused_values)
}

// =====================================================================
// Algebraic proofs for reordering and fused-row replacement.
// =====================================================================

pub proof fn lemma_work_fold_commutative(input: MoeInput, token: nat)
    ensures vstd::seq_lib::commutative_foldr(
        |item: WorkItem, acc: int| work_fold_step(input, token, item, acc),
    ),
{
    assert forall|left: WorkItem, right: WorkItem, acc: int|
        #[trigger] work_fold_step(
            input, token, left,
            work_fold_step(input, token, right, acc),
        ) == work_fold_step(
            input, token, right,
            work_fold_step(input, token, left, acc),
        ) by {
    }
}

pub proof fn lemma_plan_permutation_preserves_output(
    input: MoeInput,
    left: Seq<WorkItem>,
    right: Seq<WorkItem>,
)
    requires left.to_multiset() == right.to_multiset(),
    ensures semantic_eq(execute_plan(input, left), execute_plan(input, right)),
{
    assert(execute_plan(input, left).content =~= execute_plan(input, right).content) by {
        assert forall|t: int| #![auto] 0 <= t < input.token_values.len() implies
            execute_plan(input, left).content[t]
                == execute_plan(input, right).content[t] by {
            lemma_work_fold_commutative(input, t as nat);
            vstd::seq_lib::lemma_fold_right_permutation(
                left,
                right,
                |item: WorkItem, acc: int|
                    work_fold_step(input, t as nat, item, acc),
                0int,
            );
        }
    }
}

pub proof fn lemma_expected_values_prefix(
    input: MoeInput,
    order: Seq<WorkItem>,
    token: nat,
    n: nat,
)
    requires n <= order.len(),
    ensures fold_values_for_token(
        input, order, expected_values(input, order), token, n,
    ) == fold_work_for_token(input, order.take(n as int), token),
    decreases n,
{
    if n == 0 {
        reveal(Seq::fold_right);
    } else {
        let previous_n = (n - 1) as nat;
        let prefix = order.take(n as int);
        let previous = order.take(previous_n as int);
        let last_item = order[previous_n as int];
        let step = |item: WorkItem, acc: int|
            work_fold_step(input, token, item, acc);

        lemma_expected_values_prefix(input, order, token, previous_n);
        lemma_work_fold_commutative(input, token);

        assert(prefix.len() == n);
        assert(prefix.last() == last_item);
        assert(prefix.drop_last() =~= previous);
        prefix.drop_last().lemma_fold_right_commute_one(
            prefix.last(), step, 0int,
        );

        reveal_with_fuel(Seq::fold_right, 1);
        assert(expected_values(input, order)[previous_n as int]
            == work_value(input, last_item));
        assert(work_fold_step(input, token, last_item, 0int)
            == if work_token(input, last_item) == token {
                work_value(input, last_item)
            } else {
                0int
            });
    }
}

pub proof fn lemma_execute_expected_values(input: MoeInput, order: Seq<WorkItem>)
    ensures semantic_eq(
        execute_values(input, order, expected_values(input, order)),
        execute_plan(input, order),
    ),
{
    assert(execute_values(input, order, expected_values(input, order)).content
        =~= execute_plan(input, order).content) by {
        assert forall|t: int| 0 <= t < input.token_values.len() implies
            execute_values(input, order, expected_values(input, order)).content[t]
                == execute_plan(input, order).content[t] by {
            lemma_expected_values_prefix(input, order, t as nat, order.len());
            assert(order.take(order.len() as int) == order);
        }
    }
}

pub proof fn lemma_fused_values_equal_expected(input: MoeInput)
    requires fused_rows_correct(input),
    ensures input.fused_values == expected_values(input, input.routing_order),
{
    assert(input.fused_values =~= expected_values(input, input.routing_order));
}

pub proof fn theorem_routing_plan_refines_spec(input: MoeInput)
    requires RoutingConsistent(input),
    ensures semantic_eq(permuted_forward(input), moe_spec(input)),
{
    lemma_plan_permutation_preserves_output(
        input, input.routing_order, canonical_work(input),
    );
}

pub proof fn theorem_fused_rows_refine_plan(input: MoeInput)
    requires fused_rows_correct(input),
    ensures semantic_eq(fused_forward(input), permuted_forward(input)),
{
    lemma_fused_values_equal_expected(input);
    lemma_execute_expected_values(input, input.routing_order);
}

// =====================================================================
// Work conservation derived from routing permutation.
// =====================================================================

pub open spec fn times_processed(
    input: MoeInput,
    order: Seq<WorkItem>,
    item: WorkItem,
) -> nat {
    order.to_multiset().count(item)
}

pub open spec fn work_complete(input: MoeInput, order: Seq<WorkItem>) -> bool {
    forall|item: WorkItem| #![trigger times_processed(input, order, item)]
        required_work_item(input, item)
            ==> times_processed(input, order, item) == 1nat
}

pub open spec fn work_disjoint(input: MoeInput, order: Seq<WorkItem>) -> bool {
    forall|item: WorkItem| #![trigger times_processed(input, order, item)]
        times_processed(input, order, item) <= 1nat
}

pub open spec fn no_spurious_work(input: MoeInput, order: Seq<WorkItem>) -> bool {
    forall|item: WorkItem| #![trigger times_processed(input, order, item)]
        !required_work_item(input, item)
            ==> times_processed(input, order, item) == 0nat
}

pub proof fn lemma_canonical_work_no_duplicates(input: MoeInput)
    ensures canonical_work(input).no_duplicates(),
{
    assert forall|i: int, j: int|
        0 <= i < canonical_work(input).len()
        && 0 <= j < canonical_work(input).len()
        && i != j
        implies canonical_work(input)[i] != canonical_work(input)[j] by {
        assert(canonical_work(input)[i].flat == i as nat);
        assert(canonical_work(input)[j].flat == j as nat);
    }
}

pub proof fn lemma_canonical_work_count(input: MoeInput, item: WorkItem)
    ensures canonical_work(input).to_multiset().count(item)
        == if required_work_item(input, item) { 1nat } else { 0nat },
{
    broadcast use vstd::seq_lib::group_to_multiset_ensures;
    broadcast use vstd::multiset::group_multiset_axioms;

    lemma_canonical_work_no_duplicates(input);
    canonical_work(input).lemma_multiset_has_no_duplicates();

    if required_work_item(input, item) {
        assert(canonical_work(input)[item.flat as int].flat == item.flat);
        assert(canonical_work(input)[item.flat as int] == item);
        assert(canonical_work(input).contains(item));
        assert(canonical_work(input).to_multiset().contains(item));
    } else {
        assert forall|i: int| 0 <= i < canonical_work(input).len()
            implies canonical_work(input)[i] != item by {
            assert(canonical_work(input)[i].flat == i as nat);
        }
        assert(!canonical_work(input).contains(item));
    }
}

pub proof fn theorem_order_work_conservation(
    input: MoeInput,
    order: Seq<WorkItem>,
)
    requires order.to_multiset() == canonical_work(input).to_multiset(),
    ensures
        work_complete(input, order),
        work_disjoint(input, order),
        no_spurious_work(input, order),
{
    assert forall|item: WorkItem|
        #[trigger] times_processed(input, order, item)
            == if required_work_item(input, item) { 1nat } else { 0nat } by {
        lemma_canonical_work_count(input, item);
    }
}

pub proof fn theorem_routing_work_conservation(input: MoeInput)
    requires RoutingConsistent(input),
    ensures
        work_complete(input, input.routing_order),
        work_disjoint(input, input.routing_order),
        no_spurious_work(input, input.routing_order),
{
    theorem_order_work_conservation(input, input.routing_order);
}

} // verus!
