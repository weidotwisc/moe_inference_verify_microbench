// moe_baseline.rs
//
// Verus attempt at single-GPU MoE correctness properties for Ex05.
// Proves:
//   RT1 (token conservation via offsets)
//   RT2 (offset monotonicity)
//   RT3 (top-k weight normalization via axiom)
//   E1/E2 are proved over the shared exact model in
//   verus/kernel_refinement.rs.
//
// Focus: routing correctness (token conservation, monotonic offsets, top-k
// normalization). These are the load-bearing invariants for every downstream
// MoE proof (Ex06, Ex07, Ex09).
//
// Run with:
//   verus moe_baseline.rs

use vstd::prelude::*;

verus! {

// =====================================================================
// §1 — Types.
// =====================================================================

pub type TokenId = nat;
pub type ExpertId = nat;
pub type Weight = int;   // abstract; not reasoning about numerics

// =====================================================================
// §2 — Bincount and cumsum specs (used to construct offsets).
// =====================================================================

/// bincount(sorted_expert_ids, minlength=num_experts): for each expert e in
/// [0, num_experts), returns the count of occurrences of e in sorted_expert_ids.
pub open spec fn count_at(seq: Seq<ExpertId>, e: ExpertId) -> nat
    decreases seq.len()
{
    if seq.len() == 0 {
        0nat
    } else {
        (if seq[0] == e { 1nat } else { 0nat })
            + count_at(seq.subrange(1, seq.len() as int), e)
    }
}

pub open spec fn bincount_row(seq: Seq<ExpertId>, num_experts: nat, idx: nat) -> nat
    recommends idx < num_experts,
{
    count_at(seq, idx as ExpertId)
}

/// Recursive cumsum: prefix sum of a Seq<nat>, resulting in a Seq of length n+1.
pub open spec fn cumsum(counts: Seq<nat>) -> Seq<nat>
    decreases counts.len(),
{
    if counts.len() == 0 {
        seq![0nat]
    } else {
        let init = cumsum(counts.subrange(0, counts.len() - 1));
        init.push(init.last() + counts[counts.len() - 1])
    }
}

pub proof fn lemma_cumsum_len(counts: Seq<nat>)
    ensures cumsum(counts).len() == counts.len() + 1,
    decreases counts.len(),
{
    if counts.len() == 0 {
    } else {
        lemma_cumsum_len(counts.subrange(0, counts.len() - 1));
    }
}

pub proof fn lemma_cumsum_first_is_zero(counts: Seq<nat>)
    ensures cumsum(counts)[0] == 0nat,
    decreases counts.len(),
{
    if counts.len() == 0 {
    } else {
        lemma_cumsum_first_is_zero(counts.subrange(0, counts.len() - 1));
    }
}

pub proof fn lemma_cumsum_monotone(counts: Seq<nat>)
    ensures forall|i: int| 0 <= i < counts.len() as int
                ==> #[trigger] cumsum(counts)[i] <= cumsum(counts)[i + 1],
    decreases counts.len(),
{
    if counts.len() == 0 {
    } else {
        let init = counts.subrange(0, counts.len() - 1);
        lemma_cumsum_monotone(init);
        lemma_cumsum_len(init);
        // cumsum(counts) = cumsum(init).push(cumsum(init).last() + counts.last())
        let c = cumsum(counts);
        let ci = cumsum(init);
        assert(c =~= ci.push(ci.last() + counts[counts.len() - 1]));
        assert forall|i: int| 0 <= i < counts.len() as int
            implies #[trigger] c[i] <= c[i + 1] by {
            if i < (counts.len() - 1) as int {
                // c[i] == ci[i], c[i+1] == ci[i+1], and ci is monotone by IH.
                assert(c[i] == ci[i]);
                assert(c[i + 1] == ci[i + 1]);
            } else {
                // i == counts.len() - 1, so c[i] == ci.last(),
                // c[i+1] == ci.last() + counts[len-1] >= ci.last().
                assert(c[i] == ci.last());
                assert(c[i + 1] == ci.last() + counts[counts.len() - 1]);
            }
        }
    }
}

// =====================================================================
// §3 — Property RT2: offset monotonicity.
// =====================================================================

pub proof fn rt2_offset_monotonicity(counts: Seq<nat>)
    ensures
        forall|e: int| 0 <= e < counts.len() as int
            ==> #[trigger] cumsum(counts)[e] <= cumsum(counts)[e + 1],
{
    lemma_cumsum_monotone(counts);
}

// =====================================================================
// §4 — Property RT1: token conservation.
//
// The final offset equals the total count, which equals N * top_k.
// We prove the cumsum-total-equals-sum fact; the connection to
// N * top_k comes from bincount's spec (sum of counts == input length).
// =====================================================================

pub open spec fn seq_sum(s: Seq<nat>) -> nat
    decreases s.len(),
{
    if s.len() == 0 {
        0nat
    } else {
        s[0] + seq_sum(s.subrange(1, s.len() as int))
    }
}

pub proof fn lemma_cumsum_last_is_sum(counts: Seq<nat>)
    ensures cumsum(counts).last() == seq_sum(counts),
    decreases counts.len(),
{
    if counts.len() == 0 {
    } else {
        let init = counts.subrange(0, counts.len() - 1);
        lemma_cumsum_last_is_sum(init);
        lemma_seq_sum_split_last(counts);
    }
}

pub proof fn lemma_seq_sum_split_last(s: Seq<nat>)
    requires s.len() > 0,
    ensures seq_sum(s) == seq_sum(s.subrange(0, s.len() - 1)) + s[s.len() - 1],
    decreases s.len(),
{
    let n = s.len() as int;
    if s.len() == 1 {
        assert(s.subrange(0, 0) =~= Seq::<nat>::empty());
        assert(seq_sum(Seq::<nat>::empty()) == 0);
        assert(seq_sum(s) == s[0] + seq_sum(s.subrange(1, 1)));
        assert(s.subrange(1, 1) =~= Seq::<nat>::empty());
    } else {
        // Recurse on s.subrange(1, n) (tail).
        let tail = s.subrange(1, n);
        lemma_seq_sum_split_last(tail);
        // seq_sum(s) = s[0] + seq_sum(tail)
        //            = s[0] + seq_sum(tail.subrange(0, tail.len()-1)) + tail.last()
        // We need seq_sum(s) = seq_sum(s.subrange(0, n-1)) + s.last().
        // s.subrange(0, n-1) = s[0] :: tail.subrange(0, tail.len()-1) (structurally).
        assert(tail.len() == (n - 1) as nat);
        assert(tail.last() == s[n - 1]);
        // Establish the subrange equality via extensional equality.
        let init_of_s = s.subrange(0, n - 1);
        let init_of_tail = tail.subrange(0, tail.len() - 1);
        assert(init_of_s.len() == init_of_tail.len() + 1);
        assert(init_of_s[0] == s[0]);
        assert forall|i: int| 1 <= i < init_of_s.len()
            implies init_of_s[i] == init_of_tail[i - 1] by {
            assert(init_of_s[i] == s[i]);
            assert(init_of_tail[i - 1] == tail[i - 1]);
            assert(tail[i - 1] == s[i]);
        }
        // seq_sum(init_of_s) = s[0] + seq_sum(tail_of_init_of_s = init_of_tail).
        lemma_seq_sum_prepend_equal(init_of_s, init_of_tail);
    }
}

pub proof fn lemma_seq_sum_prepend_equal(a: Seq<nat>, b: Seq<nat>)
    requires
        a.len() == b.len() + 1,
        forall|i: int| 1 <= i < a.len() ==> a[i] == b[i - 1],
    ensures
        seq_sum(a) == a[0] + seq_sum(b),
{
    let tail = a.subrange(1, a.len() as int);
    assert(tail =~= b);
}

/// RT1: the final offset equals `total_tokens` (= N * top_k).
///
/// Under the abstract model: `offsets = cumsum(bincount(sorted_expert_ids, num_experts))`.
/// The sum of the bincount values equals the length of `sorted_expert_ids`, which is
/// N * top_k. Hence `offsets.last() == N * top_k`.
pub proof fn rt1_token_conservation(counts: Seq<nat>, total_tokens: nat)
    requires seq_sum(counts) == total_tokens,
    ensures cumsum(counts).last() == total_tokens,
{
    lemma_cumsum_last_is_sum(counts);
}

// =====================================================================
// §4b — Property RT5: routing partition correctness.
//
// RT1 (conservation) and RT2 (monotonicity) constrain the offsets only
// structurally: they hold for ANY monotone array with the right endpoint.
// They do NOT pin WHICH tokens land in WHICH expert's block. A boundary-
// corruption bug (move one offset up by one; collapse the last block to
// empty) keeps the array monotone and keeps offsets.last() fixed --- so it
// passes RT1 and RT2 --- yet it charges a token's contribution to the wrong
// expert.
//
// RT5 closes this gap. `counts_of(assignment, E)[e] == count_at(assignment, e)`
// is the TRUE number of tokens routed to expert e; `offsets_of == cumsum` of
// those counts is the correctly-constructed offset array. RT5 proves each
// block's size equals its true count; RT5-detection is the contrapositive a
// runtime differential check uses at the dispatch seam to reject a corrupted
// offset array (the v8 off-by-one and v10 collapsed-last-block bugs) no
// matter which tokens happen to be sampled downstream.
//
// (This is the block-SIZE half of routing correctness --- the half the
// v8/v10 bugs break. The complementary content half --- that the sort
// permutation places each token in its own expert's block, "RT4" --- is a
// bijection property tracked separately in E2 below.)
// =====================================================================

/// cumsum increment: consecutive offsets differ by exactly that expert's count.
pub proof fn lemma_cumsum_increment(counts: Seq<nat>)
    ensures
        forall|i: int| 0 <= i < counts.len() as int
            ==> #[trigger] cumsum(counts)[i + 1] == cumsum(counts)[i] + counts[i],
    decreases counts.len(),
{
    if counts.len() == 0 {
    } else {
        let init = counts.subrange(0, counts.len() - 1);
        lemma_cumsum_increment(init);
        lemma_cumsum_len(init);
        let c = cumsum(counts);
        let ci = cumsum(init);
        assert(c =~= ci.push(ci.last() + counts[counts.len() - 1]));
        assert(ci.len() == counts.len());
        assert forall|i: int| 0 <= i < counts.len() as int
            implies #[trigger] c[i + 1] == c[i] + counts[i] by {
            if i < (counts.len() - 1) as int {
                // c[i] == ci[i], c[i+1] == ci[i+1], init[i] == counts[i]; IH closes it.
                assert(c[i] == ci[i]);
                assert(c[i + 1] == ci[i + 1]);
                assert(init[i] == counts[i]);
            } else {
                // i == counts.len() - 1: c[i] == ci.last(), c[i+1] == ci.last() + counts[i].
                assert(c[i] == ci[i]);
                assert(c[i] == ci.last());
                assert(c[i + 1] == ci.last() + counts[counts.len() - 1]);
            }
        }
    }
}

/// The true per-expert token counts derived from a (sorted) expert assignment.
pub open spec fn counts_of(assignment: Seq<ExpertId>, num_experts: nat) -> Seq<nat> {
    Seq::new(num_experts, |e: int| count_at(assignment, e as nat))
}

/// The correctly-constructed offsets: cumsum of the true bincount.
pub open spec fn offsets_of(assignment: Seq<ExpertId>, num_experts: nat) -> Seq<nat> {
    cumsum(counts_of(assignment, num_experts))
}

/// RT5: in the correctly-constructed offsets, expert e's block spans exactly
/// `count_at(assignment, e)` rows --- the true number of tokens routed to e.
pub proof fn rt5_block_size_matches_count(
    assignment: Seq<ExpertId>, num_experts: nat, e: int,
)
    requires 0 <= e < num_experts as int,
    ensures
        offsets_of(assignment, num_experts)[e + 1]
            == offsets_of(assignment, num_experts)[e]
             + count_at(assignment, e as nat),
{
    let counts = counts_of(assignment, num_experts);
    assert(counts.len() == num_experts);
    lemma_cumsum_increment(counts);
    assert(counts[e] == count_at(assignment, e as nat));
}

/// RT5-detection (contrapositive): any offset array whose block e has the
/// wrong size is NOT the correctly-constructed offsets. This is exactly the
/// differential check that rejects the v8/v10 boundary-corruption bugs ---
/// they preserve RT1 and RT2 but break this per-block size equation.
pub proof fn rt5_wrong_size_implies_wrong_offsets(
    assignment: Seq<ExpertId>, num_experts: nat,
    bad_offsets: Seq<nat>, e: int,
)
    requires
        0 <= e < num_experts as int,
        bad_offsets[e + 1] != bad_offsets[e] + count_at(assignment, e as nat),
    ensures
        bad_offsets != offsets_of(assignment, num_experts),
{
    rt5_block_size_matches_count(assignment, num_experts, e);
    if bad_offsets == offsets_of(assignment, num_experts) {
        assert(bad_offsets[e] == offsets_of(assignment, num_experts)[e]);
        assert(bad_offsets[e + 1] == offsets_of(assignment, num_experts)[e + 1]);
        assert(false);
    }
}

// =====================================================================
// §4c — Unconditional token conservation.
//
// RT1 above discharges its endpoint claim only from the PREMISE
// `seq_sum(counts) == total_tokens`. That premise is the real
// "no token falls through the cracks" fact --- every one of the N*k
// (token, expert) assignments is counted in exactly one bin. Here we
// PROVE it rather than assume it: for any assignment whose ids are all
// valid (< num_experts), the per-expert bincount sums to the number of
// assignments. Wiring this into RT1 (rt1_token_conservation_from_assignment)
// makes conservation unconditional at the model level --- the offsets built
// from a valid assignment provably account for every token.
// =====================================================================

/// Linearity of `seq_sum` over a pointwise sum of two equal-length sequences.
pub proof fn lemma_seq_sum_pointwise_add(x: Seq<nat>, y: Seq<nat>, c: Seq<nat>)
    requires
        x.len() == y.len(),
        c.len() == x.len(),
        forall|i: int| 0 <= i < c.len() ==> c[i] == x[i] + y[i],
    ensures
        seq_sum(c) == seq_sum(x) + seq_sum(y),
    decreases c.len(),
{
    if c.len() == 0 {
    } else {
        let x1 = x.subrange(1, x.len() as int);
        let y1 = y.subrange(1, y.len() as int);
        let c1 = c.subrange(1, c.len() as int);
        assert forall|i: int| 0 <= i < c1.len() implies c1[i] == x1[i] + y1[i] by {
            assert(c1[i] == c[i + 1]);
            assert(x1[i] == x[i + 1]);
            assert(y1[i] == y[i + 1]);
        }
        lemma_seq_sum_pointwise_add(x1, y1, c1);
    }
}

/// An indicator sequence over [0, n) --- value 1 at index a0, else 0 ---
/// sums to 1 when a0 is in range, else 0.
pub proof fn lemma_indicator_sum(n: nat, a0: int)
    ensures
        seq_sum(Seq::new(n, |e: int| if e == a0 { 1nat } else { 0nat }))
            == (if 0 <= a0 < n as int { 1nat } else { 0nat }),
    decreases n,
{
    let ind = Seq::new(n, |e: int| if e == a0 { 1nat } else { 0nat });
    if n == 0 {
        assert(ind.len() == 0);
    } else {
        let tail = ind.subrange(1, n as int);
        let tail_ind = Seq::new((n - 1) as nat, |e: int| if e == a0 - 1 { 1nat } else { 0nat });
        assert(tail =~= tail_ind) by {
            assert forall|i: int| 0 <= i < tail.len() implies tail[i] == tail_ind[i] by {
                assert(tail[i] == ind[i + 1]);
                assert(ind[i + 1] == (if (i + 1) == a0 { 1nat } else { 0nat }));
                assert(tail_ind[i] == (if i == a0 - 1 { 1nat } else { 0nat }));
            }
        }
        lemma_indicator_sum((n - 1) as nat, a0 - 1);
    }
}

/// A sequence of all zeros sums to zero.
pub proof fn lemma_seq_sum_all_zero(s: Seq<nat>)
    requires
        forall|i: int| 0 <= i < s.len() ==> s[i] == 0,
    ensures
        seq_sum(s) == 0,
    decreases s.len(),
{
    if s.len() == 0 {
    } else {
        let s1 = s.subrange(1, s.len() as int);
        assert forall|i: int| 0 <= i < s1.len() implies s1[i] == 0 by {
            assert(s1[i] == s[i + 1]);
        }
        lemma_seq_sum_all_zero(s1);
    }
}

/// The per-expert bincount of a valid assignment sums to the number of
/// assignments --- no token is dropped or double-counted by the routing.
pub proof fn lemma_bincount_sums_to_len(assignment: Seq<ExpertId>, num_experts: nat)
    requires
        forall|i: int| 0 <= i < assignment.len() ==> assignment[i] < num_experts,
    ensures
        seq_sum(counts_of(assignment, num_experts)) == assignment.len(),
    decreases assignment.len(),
{
    if assignment.len() == 0 {
        let empty_counts = counts_of(assignment, num_experts);
        assert forall|e: int| 0 <= e < empty_counts.len() implies empty_counts[e] == 0 by {
            assert(empty_counts[e] == count_at(assignment, e as nat));
        }
        lemma_seq_sum_all_zero(empty_counts);
    } else {
        let rest = assignment.subrange(1, assignment.len() as int);
        let a0 = assignment[0];
        let indicator = Seq::new(num_experts, |e: int| if e == a0 as int { 1nat } else { 0nat });
        let rest_counts = counts_of(rest, num_experts);
        let full_counts = counts_of(assignment, num_experts);
        assert forall|e: int| #![trigger full_counts[e]] 0 <= e < num_experts as int
            implies full_counts[e] == indicator[e] + rest_counts[e] by {
            assert(full_counts[e] == count_at(assignment, e as nat));
            assert(rest_counts[e] == count_at(rest, e as nat));
            assert(count_at(assignment, e as nat)
                == (if assignment[0] == e as nat { 1nat } else { 0nat })
                 + count_at(assignment.subrange(1, assignment.len() as int), e as nat));
            assert(assignment.subrange(1, assignment.len() as int) == rest);
            assert((if e == a0 as int { 1nat } else { 0nat })
                == (if assignment[0] == e as nat { 1nat } else { 0nat }));
        }
        lemma_seq_sum_pointwise_add(indicator, rest_counts, full_counts);
        assert(assignment[0] < num_experts);
        lemma_indicator_sum(num_experts, a0 as int);
        assert(seq_sum(indicator) == 1nat);
        assert forall|i: int| 0 <= i < rest.len() implies rest[i] < num_experts by {
            assert(rest[i] == assignment[i + 1]);
        }
        lemma_bincount_sums_to_len(rest, num_experts);
    }
}

/// RT1 (unconditional): for offsets built from a valid assignment, the final
/// offset equals the token count --- with the sum premise now PROVED, not
/// assumed. Model-level conservation no longer rests on an assumption.
pub proof fn rt1_token_conservation_from_assignment(
    assignment: Seq<ExpertId>, num_experts: nat,
)
    requires
        forall|i: int| 0 <= i < assignment.len() ==> assignment[i] < num_experts,
    ensures
        offsets_of(assignment, num_experts).last() == assignment.len(),
{
    lemma_bincount_sums_to_len(assignment, num_experts);
    lemma_cumsum_last_is_sum(counts_of(assignment, num_experts));
}

// =====================================================================
// §4d — Property RT4: the routing permutation is a bijection.
//
// The dispatch sorts the flattened (token, expert) pairs by expert; the
// gather uses the sort permutation `perm`, and the scatter-back uses its
// inverse `inv`. RT4 is the slot<->token link: because (perm, inv) is a
// two-sided inverse pair (a permutation of [0, M)), every token maps to
// exactly one sorted slot and back --- so the gather/scatter round-trip
// drops no token (surjective) and writes none twice (injective). This is
// what lifts the SLOT-level conservation (RT1/RT5) to actual TOKENS.
//
// That `argsort` returns a permutation is the one trusted axiom here (a
// property of the library sort, in the same spirit as matmul being
// uninterpreted); the bijection CONSEQUENCES are proved.
// =====================================================================

pub open spec fn is_inverse_pair(perm: Seq<nat>, inv: Seq<nat>, m: nat) -> bool {
    &&& perm.len() == m
    &&& inv.len() == m
    &&& (forall|p: int| 0 <= p < m ==> #[trigger] perm[p] < m)
    &&& (forall|q: int| 0 <= q < m ==> #[trigger] inv[q] < m)
    &&& (forall|p: int| 0 <= p < m ==> inv[#[trigger] perm[p] as int] == p)
    &&& (forall|q: int| 0 <= q < m ==> perm[#[trigger] inv[q] as int] == q)
}

pub uninterp spec fn sort_perm(keys: Seq<ExpertId>) -> Seq<nat>;
pub uninterp spec fn sort_inv(keys: Seq<ExpertId>) -> Seq<nat>;

/// AXIOM: argsort yields a permutation of [0, M) with a computable inverse
/// (the scatter index). Trusted --- a property of the library sort.
#[verifier::external_body]
pub proof fn axiom_argsort_is_inverse_pair(keys: Seq<ExpertId>)
    ensures is_inverse_pair(sort_perm(keys), sort_inv(keys), keys.len()),
{}

/// RT4a (surjective / no token dropped): every token q is the image of a
/// slot --- namely inv[q] --- so scatter-back reaches every token.
pub proof fn rt4_perm_surjective(perm: Seq<nat>, inv: Seq<nat>, m: nat, q: int)
    requires
        is_inverse_pair(perm, inv, m),
        0 <= q < m,
    ensures
        inv[q] < m,
        perm[inv[q] as int] == q,
{
}

/// RT4b (injective / unique writer): two slots mapping to the same token are
/// the same slot --- so no token is written twice.
pub proof fn rt4_perm_injective(perm: Seq<nat>, inv: Seq<nat>, m: nat, p1: int, p2: int)
    requires
        is_inverse_pair(perm, inv, m),
        0 <= p1 < m,
        0 <= p2 < m,
        perm[p1] == perm[p2],
    ensures
        p1 == p2,
{
    assert(inv[perm[p1] as int] == p1);
    assert(inv[perm[p2] as int] == p2);
    assert(perm[p1] as int == perm[p2] as int);
}

/// RT4 (bijection / conservation): every token q has EXACTLY ONE slot
/// mapping to it (namely inv[q]) --- so the gather/scatter round-trip writes
/// each token's output exactly once, none dropped and none twice. This lifts
/// the slot-level conservation of RT1/RT5 to actual tokens.
pub proof fn rt4_each_token_exactly_once(perm: Seq<nat>, inv: Seq<nat>, m: nat, q: int)
    requires
        is_inverse_pair(perm, inv, m),
        0 <= q < m,
    ensures
        perm[inv[q] as int] == q,
        forall|p: int| (0 <= p < m && perm[p] == q) ==> p == inv[q] as int,
{
    rt4_perm_surjective(perm, inv, m, q);
    assert forall|p: int| (0 <= p < m && perm[p] == q) implies p == inv[q] as int by {
        rt4_perm_injective(perm, inv, m, p, inv[q] as int);
    }
}

// =====================================================================
// §5 — Property RT3: top-k weight normalization (via axiom on softmax).
// =====================================================================

/// The top-k weights per token, uninterpreted (comes from softmax of top-k logits).
pub uninterp spec fn top_k_weights(top_k_logits: Seq<Weight>) -> Seq<Weight>;

/// Axiom SOFTMAX_SUM: softmax of any input sums to 1.
/// We represent "1" as the integer 1 since Weight is int (abstracted).
#[verifier::external_body]
pub proof fn axiom_softmax_sums_to_one(logits: Seq<Weight>)
    ensures seq_sum_of_weights(top_k_weights(logits)) == 1int,
{}

/// Sum of a Seq<Weight> as int (for the axiom).
pub open spec fn seq_sum_of_weights(s: Seq<Weight>) -> int
    decreases s.len(),
{
    if s.len() == 0 {
        0int
    } else {
        s[0] + seq_sum_of_weights(s.subrange(1, s.len() as int))
    }
}

pub proof fn rt3_topk_weight_normalization(logits: Seq<Weight>)
    ensures seq_sum_of_weights(top_k_weights(logits)) == 1int,
{
    axiom_softmax_sums_to_one(logits);
}

// =====================================================================
// §6 — MoE_spec: the abstract semantic reference.
// =====================================================================

pub uninterp spec fn expert_apply(e: ExpertId, x_token: Seq<Weight>) -> Seq<Weight>;

pub uninterp spec fn moe_spec_output(
    x: Seq<Seq<Weight>>,
    ids: Seq<Seq<ExpertId>>,
    weights: Seq<Seq<Weight>>,
    token_i: nat,
) -> Seq<Weight>;

// E1 and E2 are intentionally not restated as local axioms.  Their exact
// proofs share the canonical `(token, top-k slot)` work model with the Tier-3
// composition theorem; see `verus/kernel_refinement.rs`.

} // verus!

fn main() {
    println!("Verified single-GPU MoE routing properties (Verus).");
}
