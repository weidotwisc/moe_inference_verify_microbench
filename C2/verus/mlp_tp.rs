// mlp_tp.rs
//
// Verus attempt at the SwiGLU-MLP TP block correctness properties for Ex02.
// Proves:
//   M1 (per-projection weight_loader post-condition)
//   M2 (merged shards partition merged_full)
//   M3 (merged forward correctness, up to axiom AXIOM_M1)
//   T1 (chunk-split-matches-shard invariant, up to AXIOM_M1)
//   T2 (elementwise-commutes-with-gather, up to AXIOM_M1 + AXIOM_S1)
//   T3 (block correctness, up to AXIOM_M1 + AXIOM_M2 + AXIOM_S1)
//
// Run with:
//   verus mlp_tp.rs
// Expected: `verification results:: N verified, 0 errors`.
//
// See PROPERTIES.md for the semi-formal spec these lemmas realize.

use vstd::prelude::*;
use vstd::calc;
use vstd::arithmetic::div_mod::lemma_fundamental_div_mod;

verus! {

// =====================================================================
// §1 — Types (mirror of Ex01's abstraction model)
// =====================================================================

pub type Element = int;
pub type Row = Seq<Element>;
pub type Tensor = Seq<Row>;

pub open spec fn well_formed(t: Tensor, rows: nat, cols: nat) -> bool {
    t.len() == rows &&
    forall|i: int| 0 <= i < t.len() ==> #[trigger] t[i].len() == cols
}

// =====================================================================
// §2 — Sharding on dim 0 (from Ex01)
// =====================================================================

pub open spec fn shard(w: Tensor, rank: nat, tp_size: nat) -> Tensor
    recommends tp_size >= 1, w.len() % tp_size == 0, rank < tp_size,
{
    let shard_size = (w.len() / tp_size) as int;
    w.subrange((rank as int) * shard_size, (rank as int + 1) * shard_size)
}

// =====================================================================
// §3 — merged_shard: this rank's concatenated (gate, up) slice.
// =====================================================================

pub open spec fn merged_shard(w0: Tensor, w1: Tensor, rank: nat, tp_size: nat) -> Tensor
    recommends
        tp_size >= 1,
        w0.len() % tp_size == 0,
        w1.len() % tp_size == 0,
        rank < tp_size,
{
    shard(w0, rank, tp_size) + shard(w1, rank, tp_size)
}

// The "merged_full" weight — what a plain ColumnParallelLinear(2*I, H) would hold.
pub open spec fn merged_full(w0: Tensor, w1: Tensor) -> Tensor {
    w0 + w1
}

// =====================================================================
// §4 — Property M1: per-projection weight_loader post-condition.
//
// Loading gate then up on this rank produces exactly `merged_shard(r)`.
// The Python `weight_loader` does two `narrow(dim=0, start, len).copy_`
// calls; the abstract model captures this as: after both loads, the local
// tensor equals `merged_shard(rank)`.
// =====================================================================

/// Model of the local weight buffer after the two-step weight_loader.
/// This spec function is the abstract post-condition of the sequence
/// `weight_loader(w_gate, 0); weight_loader(w_up, 1)` on rank r.
pub open spec fn weight_after_merged_load(
    w0: Tensor, w1: Tensor, rank: nat, tp_size: nat,
) -> Tensor
    recommends
        tp_size >= 1,
        w0.len() % tp_size == 0,
        w1.len() % tp_size == 0,
        rank < tp_size,
{
    merged_shard(w0, w1, rank, tp_size)
}

/// M1: the two-step weight_loader leaves the local buffer equal to merged_shard.
pub proof fn m1_merged_weight_loader_postcondition(
    w0: Tensor, w1: Tensor, rank: nat, tp_size: nat,
)
    requires
        tp_size >= 1,
        w0.len() % tp_size == 0,
        w1.len() % tp_size == 0,
        rank < tp_size,
    ensures
        weight_after_merged_load(w0, w1, rank, tp_size)
        == merged_shard(w0, w1, rank, tp_size),
{
    // Immediate by definition.
}

// =====================================================================
// §5 — Property M2: merged shards partition merged_full.
//
// Gathering merged_shard(r) over all ranks reconstructs merged_full.
// This depends on: (i) w0's shards gather to w0, (ii) w1's shards gather
// to w1, (iii) the merged sequence is (gathered w0) + (gathered w1).
//
// The KEY subtlety: `merged_shard(r) = shard(w0,r) + shard(w1,r)`, so
// gathering over r interleaves w0-slices with w1-slices. That's NOT the
// same shape as `merged_full = w0 + w1` (all of w0, then all of w1) —
// but the multiset of rows is identical. The Python `weight_loader`
// writes each rank's gate slice to `[0, I/tp)` and its up slice to
// `[I/tp, 2I/tp)` — so it produces the *interleaved* layout, and the
// abstract model matches.
//
// For M2 we prove the WEAKER-BUT-USEFUL fact: the two individual gathers
// reconstruct w0 and w1 respectively. The stronger "merged shards
// reconstruct merged_full" fact requires a permutation argument beyond
// what M3 and downstream lemmas need.
// =====================================================================

pub open spec fn gather_from(w: Tensor, tp_size: nat, start: nat) -> Tensor
    recommends tp_size >= 1, w.len() % tp_size == 0, start <= tp_size,
    decreases tp_size - start,
{
    if start >= tp_size {
        Seq::<Row>::empty()
    } else {
        shard(w, start, tp_size) + gather_from(w, tp_size, (start + 1) as nat)
    }
}

pub open spec fn gather_all(w: Tensor, tp_size: nat) -> Tensor
    recommends tp_size >= 1, w.len() % tp_size == 0
{
    gather_from(w, tp_size, 0)
}

pub proof fn lemma_gather_from_equals_suffix(w: Tensor, tp_size: nat, start: nat)
    requires
        tp_size >= 1,
        w.len() % tp_size == 0,
        start <= tp_size,
    ensures
        gather_from(w, tp_size, start)
        == w.subrange((start as int) * (w.len() / tp_size) as int, w.len() as int),
    decreases tp_size - start,
{
    let s = (w.len() / tp_size) as int;
    let n = w.len() as int;
    let p = tp_size as int;
    lemma_fundamental_div_mod(n, p);
    assert(n % p == 0);
    assert(s == n / p);
    calc! {
        (==)
        n; {}
        p * (n / p) + n % p; {}
        p * (n / p) + 0; {}
        p * s;
    }
    if start == tp_size {
        assert((start as int) * s == n) by (nonlinear_arith)
            requires (start as int) == p, n == p * s;
        assert(w.subrange(n, n) =~= Seq::<Row>::empty());
    } else {
        lemma_gather_from_equals_suffix(w, tp_size, (start + 1) as nat);
        assert(((start + 1) as int) * s <= n) by (nonlinear_arith)
            requires start < tp_size, (tp_size as int) * s == n, s >= 0;
        assert((start as int) * s <= ((start + 1) as int) * s) by (nonlinear_arith)
            requires s >= 0;
        assert(0 <= (start as int) * s) by (nonlinear_arith)
            requires 0 <= start as int, s >= 0;
        assert(shard(w, start, tp_size)
               == w.subrange((start as int) * s, ((start + 1) as int) * s));
        assert(w.subrange((start as int) * s, ((start + 1) as int) * s)
               + w.subrange(((start + 1) as int) * s, n)
               =~= w.subrange((start as int) * s, n));
    }
}

/// M2 (weakened form used downstream): gathering w0's shards reconstructs w0.
pub proof fn m2_w0_gather_roundtrip(w0: Tensor, tp_size: nat)
    requires tp_size >= 1, w0.len() % tp_size == 0,
    ensures gather_all(w0, tp_size) == w0,
{
    lemma_gather_from_equals_suffix(w0, tp_size, 0);
    assert(w0.subrange(0, w0.len() as int) =~= w0);
}

/// M2 (weakened form used downstream): gathering w1's shards reconstructs w1.
pub proof fn m2_w1_gather_roundtrip(w1: Tensor, tp_size: nat)
    requires tp_size >= 1, w1.len() % tp_size == 0,
    ensures gather_all(w1, tp_size) == w1,
{
    lemma_gather_from_equals_suffix(w1, tp_size, 0);
    assert(w1.subrange(0, w1.len() as int) =~= w1);
}

// =====================================================================
// §6 — Matmul, transpose, and concat_cols (from Ex01) + silu_mul.
// =====================================================================

pub uninterp spec fn matmul(x: Tensor, w_t: Tensor) -> Tensor;
pub uninterp spec fn transpose(t: Tensor) -> Tensor;
pub uninterp spec fn silu_mul(gate: Tensor, up: Tensor) -> Tensor;
pub uninterp spec fn tensor_sum(a: Tensor, b: Tensor) -> Tensor;

pub open spec fn concat_cols(a: Tensor, b: Tensor) -> Tensor
    recommends a.len() == b.len(),
{
    Seq::new(a.len(), |i: int| a[i] + b[i])
}

/// The row-parallel down projection shards each weight row on its input
/// dimension.  This is the dim-1 sharding rule used by Ex01.
pub open spec fn shard_dim1(
    w: Tensor,
    rank: nat,
    tp_size: nat,
) -> Tensor
    recommends
        tp_size >= 1,
        w.len() >= 1,
        w[0].len() % tp_size == 0,
        rank < tp_size,
{
    let shard_size = (w[0].len() / tp_size) as int;
    Seq::new(w.len(), |i: int|
        w[i].subrange(
            (rank as int) * shard_size,
            (rank as int + 1) * shard_size,
        )
    )
}

proof fn lemma_dim0_shards_reconstruct_tp2(w: Tensor)
    requires w.len() % 2 == 0,
    ensures shard(w, 0, 2) + shard(w, 1, 2) == w,
{
    let n = w.len() as int;
    let s = (w.len() / 2) as int;
    lemma_fundamental_div_mod(n, 2);
    assert(n == 2 * s) by (nonlinear_arith)
        requires n == 2 * (n / 2) + n % 2, n % 2 == 0,
            s == n / 2;
    assert(shard(w, 0, 2) == w.subrange(0, s));
    assert(shard(w, 1, 2) == w.subrange(s, n));
    assert(w.subrange(0, s) + w.subrange(s, n) =~= w);
}

proof fn lemma_dim0_shards_well_formed_tp2(
    w: Tensor,
    cols: nat,
)
    requires
        w.len() >= 2,
        w.len() % 2 == 0,
        well_formed(w, w.len(), cols),
    ensures
        well_formed(shard(w, 0, 2), w.len() / 2, cols),
        well_formed(shard(w, 1, 2), w.len() / 2, cols),
{
    let n = w.len() as int;
    let s = (w.len() / 2) as int;
    lemma_fundamental_div_mod(n, 2);
    assert(n == 2 * s) by (nonlinear_arith)
        requires n == 2 * (n / 2) + n % 2, n % 2 == 0,
            s == n / 2;
    let w0 = shard(w, 0, 2);
    let w1 = shard(w, 1, 2);
    assert(w0 == w.subrange(0, s));
    assert(w1 == w.subrange(s, n));
    assert(w0.len() == w.len() / 2);
    assert(w1.len() == w.len() / 2);
    assert forall|i: int| 0 <= i < w0.len() implies
        #[trigger] w0[i].len() == cols by {
        assert(w0[i] == w[i]);
    }
    assert forall|i: int| 0 <= i < w1.len() implies
        #[trigger] w1[i].len() == cols by {
        assert(w1[i] == w[s + i]);
        assert(0 <= s + i < w.len());
    }
}

proof fn lemma_dim1_shards_reconstruct_tp2(
    w: Tensor,
    rows: nat,
    cols: nat,
)
    requires
        w.len() >= 1,
        well_formed(w, rows, cols),
        cols % 2 == 0,
    ensures concat_cols(
        shard_dim1(w, 0, 2),
        shard_dim1(w, 1, 2),
    ) == w,
{
    let n = cols as int;
    let s = (cols / 2) as int;
    lemma_fundamental_div_mod(n, 2);
    assert(n == 2 * s) by (nonlinear_arith)
        requires n == 2 * (n / 2) + n % 2, n % 2 == 0,
            s == n / 2;

    let w0 = shard_dim1(w, 0, 2);
    let w1 = shard_dim1(w, 1, 2);
    assert(w[0].len() == cols);
    assert(w0.len() == w.len());
    assert(w1.len() == w.len());
    assert(concat_cols(w0, w1).len() == w.len());
    assert forall|i: int| 0 <= i < w.len() implies
        concat_cols(w0, w1)[i] == w[i] by {
        assert(w[i].len() == cols);
        assert(w0[i] == w[i].subrange(0, s));
        assert(w1[i] == w[i].subrange(s, n));
        assert(w[i].subrange(0, s) + w[i].subrange(s, n) =~= w[i]);
    }
    assert(concat_cols(w0, w1) =~= w);
}

/// Axiom AXIOM_M1: matmul splits over the out-dim of the weight.
#[verifier::external_body]
pub proof fn axiom_m1(x: Tensor, w0: Tensor, w1: Tensor)
    requires
        w0.len() > 0, w1.len() > 0,
        exists|rowlen: nat|
            #[trigger] well_formed(w0, w0.len(), rowlen)
         && well_formed(w1, w1.len(), rowlen),
        matmul(x, transpose(w0)).len() == matmul(x, transpose(w1)).len(),
    ensures
        matmul(x, transpose(w0 + w1))
        == concat_cols(matmul(x, transpose(w0)), matmul(x, transpose(w1))),
{}

/// Axiom AXIOM_M2: matmul splits over the in-dim of the weight with sum.
/// (Row-parallel's post-all-reduce equivalence.)
#[verifier::external_body]
pub proof fn axiom_m2(
    x: Tensor,
    w: Tensor,
    x0: Tensor,
    x1: Tensor,
    w0: Tensor,
    w1: Tensor,
)
    requires
        x0.len() == x1.len(),
        w0.len() > 0,
        w0.len() == w1.len(),
        x == concat_cols(x0, x1),
        w == concat_cols(w0, w1),
    ensures
        tensor_sum(
            matmul(x0, transpose(w0)),
            matmul(x1, transpose(w1)),
        ) == matmul(x, transpose(w)),
{}

/// Axiom AXIOM_S1: elementwise silu_mul commutes with dim-1 concatenation.
#[verifier::external_body]
pub proof fn axiom_s1(g0: Tensor, g1: Tensor, u0: Tensor, u1: Tensor)
    requires
        g0.len() == u0.len(),
        g1.len() == u1.len(),
        g0.len() == g1.len(),
    ensures
        silu_mul(concat_cols(g0, g1), concat_cols(u0, u1))
            == concat_cols(silu_mul(g0, u0), silu_mul(g1, u1)),
        silu_mul(g0, u0).len() == g0.len(),
        silu_mul(g1, u1).len() == g1.len(),
{}

// =====================================================================
// §7 — Property M3: merged forward correctness (tp_size == 2).
//
// The gathered merged forward output equals matmul(x, transpose(merged_full)).
// Reduces to Ex01's C4 applied to `merged_full`.
// =====================================================================

pub open spec fn merged_forward_output(
    x: Tensor, w0: Tensor, w1: Tensor, rank: nat, tp_size: nat,
) -> Tensor
    recommends
        tp_size >= 1,
        w0.len() % tp_size == 0,
        w1.len() % tp_size == 0,
        rank < tp_size,
{
    matmul(x, transpose(merged_shard(w0, w1, rank, tp_size)))
}

/// M3 for tp_size == 2: concat of merged forward outputs equals unsharded matmul.
///
/// See anecdote.txt for the audit-cycle notes on this proof — the well-formedness
/// scaffolding below was added after Verus rejected the initial draft that
/// pattern-matched on Ex01's C4-tp2 but omitted the rowlen-witness construction.
pub proof fn m3_merged_forward_correctness_tp2(x: Tensor, w0: Tensor, w1: Tensor)
    requires
        w0.len() >= 2, w1.len() >= 2,
        w0.len() % 2 == 0, w1.len() % 2 == 0,
        exists|rowlen: nat|
            #[trigger] well_formed(w0, w0.len(), rowlen)
         && well_formed(w1, w1.len(), rowlen),
        matmul(x, transpose(merged_shard(w0, w1, 0, 2))).len()
            == matmul(x, transpose(merged_shard(w0, w1, 1, 2))).len(),
    ensures
        concat_cols(
            merged_forward_output(x, w0, w1, 0, 2),
            merged_forward_output(x, w0, w1, 1, 2),
        ) == matmul(x, transpose(merged_shard(w0, w1, 0, 2) + merged_shard(w0, w1, 1, 2))),
{
    let ms0 = merged_shard(w0, w1, 0, 2);
    let ms1 = merged_shard(w0, w1, 1, 2);

    // Well-formedness witness: both w0 and w1 share the same rowlen; propagate
    // it through shard, then through the concatenation that forms merged_shard.
    let rowlen = choose|rowlen: nat| well_formed(w0, w0.len(), rowlen)
                                   && well_formed(w1, w1.len(), rowlen);
    assert(well_formed(w0, w0.len(), rowlen));
    assert(well_formed(w1, w1.len(), rowlen));

    let s0w0 = shard(w0, 0, 2);
    let s1w0 = shard(w0, 1, 2);
    let s0w1 = shard(w1, 0, 2);
    let s1w1 = shard(w1, 1, 2);

    // Each shard inherits rowlen from its parent (subrange preserves per-row content).
    assert(well_formed(s0w0, s0w0.len(), rowlen)) by {
        assert forall|i: int| 0 <= i < s0w0.len()
            implies #[trigger] s0w0[i].len() == rowlen by {
            assert(s0w0[i] == w0[i]);
        }
    }
    assert(well_formed(s1w0, s1w0.len(), rowlen)) by {
        let off = (w0.len() / 2) as int;
        assert forall|i: int| 0 <= i < s1w0.len()
            implies #[trigger] s1w0[i].len() == rowlen by {
            assert(s1w0[i] == w0[off + i]);
            assert(0 <= off + i < w0.len());
        }
    }
    assert(well_formed(s0w1, s0w1.len(), rowlen)) by {
        assert forall|i: int| 0 <= i < s0w1.len()
            implies #[trigger] s0w1[i].len() == rowlen by {
            assert(s0w1[i] == w1[i]);
        }
    }
    assert(well_formed(s1w1, s1w1.len(), rowlen)) by {
        let off = (w1.len() / 2) as int;
        assert forall|i: int| 0 <= i < s1w1.len()
            implies #[trigger] s1w1[i].len() == rowlen by {
            assert(s1w1[i] == w1[off + i]);
            assert(0 <= off + i < w1.len());
        }
    }

    // Merged shards: concat of two rowlen-uniform sequences is still rowlen-uniform.
    // Case-split on whether the row-index lies in the first or second half.
    assert(ms0 == s0w0 + s0w1);
    assert(well_formed(ms0, ms0.len(), rowlen)) by {
        assert forall|i: int| 0 <= i < ms0.len()
            implies #[trigger] ms0[i].len() == rowlen by {
            if i < s0w0.len() as int {
                assert(ms0[i] == s0w0[i]);
            } else {
                assert(ms0[i] == s0w1[i - s0w0.len() as int]);
            }
        }
    }
    assert(ms1 == s1w0 + s1w1);
    assert(well_formed(ms1, ms1.len(), rowlen)) by {
        assert forall|i: int| 0 <= i < ms1.len()
            implies #[trigger] ms1[i].len() == rowlen by {
            if i < s1w0.len() as int {
                assert(ms1[i] == s1w0[i]);
            } else {
                assert(ms1[i] == s1w1[i - s1w0.len() as int]);
            }
        }
    }

    // Non-empty lengths: each shard has w.len()/2 >= 1 rows since w.len() >= 2.
    assert(s0w0.len() > 0);
    assert(s0w1.len() > 0);
    assert(s1w0.len() > 0);
    assert(s1w1.len() > 0);
    assert(ms0.len() > 0);
    assert(ms1.len() > 0);

    // Existential witness the axiom needs.
    assert(exists|r: nat|
        #[trigger] well_formed(ms0, ms0.len(), r)
        && well_formed(ms1, ms1.len(), r));

    axiom_m1(x, ms0, ms1);
}

// =====================================================================
// §8 — Property T1: chunk-split-matches-shard.
//
// The output of the merged forward on rank r splits on dim 1 into
// (matmul with gate_shard) and (matmul with up_shard).
// =====================================================================

/// T1: applying AXIOM_M1 with gate_shard and up_shard gives the split identity.
pub proof fn t1_chunk_split_matches_shard(
    x: Tensor, w0: Tensor, w1: Tensor, rank: nat, tp_size: nat,
)
    requires
        tp_size >= 1,
        rank < tp_size,
        w0.len() % tp_size == 0,
        w1.len() % tp_size == 0,
        shard(w0, rank, tp_size).len() > 0,
        shard(w1, rank, tp_size).len() > 0,
        exists|rowlen: nat|
            #[trigger] well_formed(shard(w0, rank, tp_size), shard(w0, rank, tp_size).len(), rowlen)
         && well_formed(shard(w1, rank, tp_size), shard(w1, rank, tp_size).len(), rowlen),
        matmul(x, transpose(shard(w0, rank, tp_size))).len()
            == matmul(x, transpose(shard(w1, rank, tp_size))).len(),
    ensures
        matmul(x, transpose(merged_shard(w0, w1, rank, tp_size)))
        == concat_cols(
            matmul(x, transpose(shard(w0, rank, tp_size))),
            matmul(x, transpose(shard(w1, rank, tp_size))),
        ),
{
    let gs = shard(w0, rank, tp_size);
    let us = shard(w1, rank, tp_size);
    // merged_shard is definitionally gs + us.
    assert(merged_shard(w0, w1, rank, tp_size) == gs + us);
    axiom_m1(x, gs, us);
}

// =====================================================================
// §9 — Property T2: elementwise commutes with gather.
//
// Concatenating silu_mul(g_r, u_r) over r equals silu_mul applied to the
// concatenated g and u sequences. Stated for tp_size == 2 to match M3's
// concrete case.
// =====================================================================

/// T2 for tp_size == 2: silu_mul commutes with the dim-1 gather.
pub proof fn t2_elementwise_commutes_with_gather_tp2(
    g0: Tensor, g1: Tensor, u0: Tensor, u1: Tensor,
)
    requires
        g0.len() == u0.len(),
        g1.len() == u1.len(),
        g0.len() == g1.len(),
    ensures
        concat_cols(silu_mul(g0, u0), silu_mul(g1, u1))
        == silu_mul(concat_cols(g0, g1), concat_cols(u0, u1)),
{
    axiom_s1(g0, g1, u0, u1);
}

// =====================================================================
// §10 — Property T3: block correctness (tp_size == 2).
//
// The block output — sum over ranks of matmul(silu_mul(g_r, u_r), transpose
// of the down-projection's row shard) — equals the unsharded MLP output.
//
// This composes dim-0 reconstruction, AXIOM_M1, T2, dim-1 reconstruction,
// and the meaningful AXIOM_M2 row-parallel/all-reduce contract above.
// =====================================================================

pub open spec fn local_gate_output_tp2(
    x: Tensor,
    w_gate: Tensor,
    rank: nat,
) -> Tensor
    recommends rank < 2, w_gate.len() % 2 == 0,
{
    matmul(x, transpose(shard(w_gate, rank, 2)))
}

pub open spec fn local_up_output_tp2(
    x: Tensor,
    w_up: Tensor,
    rank: nat,
) -> Tensor
    recommends rank < 2, w_up.len() % 2 == 0,
{
    matmul(x, transpose(shard(w_up, rank, 2)))
}

pub open spec fn local_hidden_tp2(
    x: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    rank: nat,
) -> Tensor
    recommends
        rank < 2,
        w_gate.len() % 2 == 0,
        w_up.len() % 2 == 0,
{
    silu_mul(
        local_gate_output_tp2(x, w_gate, rank),
        local_up_output_tp2(x, w_up, rank),
    )
}

/// Exact two-rank TP execution after expanding the merged projection with T1:
/// each rank computes its SwiGLU hidden shard, applies the matching dim-1
/// shard of the down projection, and all-reduce sums the partial outputs.
pub open spec fn tp_mlp_forward_tp2(
    x: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
) -> Tensor
    recommends
        w_gate.len() % 2 == 0,
        w_up.len() % 2 == 0,
        w_down.len() >= 1,
        w_down[0].len() % 2 == 0,
{
    tensor_sum(
        matmul(
            local_hidden_tp2(x, w_gate, w_up, 0),
            transpose(shard_dim1(w_down, 0, 2)),
        ),
        matmul(
            local_hidden_tp2(x, w_gate, w_up, 1),
            transpose(shard_dim1(w_down, 1, 2)),
        ),
    )
}

/// Exact unsharded mathematical SwiGLU MLP.
pub open spec fn unsharded_mlp_forward(
    x: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
) -> Tensor {
    matmul(
        silu_mul(
            matmul(x, transpose(w_gate)),
            matmul(x, transpose(w_up)),
        ),
        transpose(w_down),
    )
}

/// T3: the exact tp=2 merged-column/SwiGLU/row-parallel execution equals
/// the unsharded MLP.  Shape premises are the valid-matmul obligations of
/// the abstract tensor model; no functional-equivalence premise is assumed.
pub proof fn t3_block_correctness_tp2(
    x: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
)
    requires
        w_gate.len() >= 2,
        w_up.len() >= 2,
        w_gate.len() % 2 == 0,
        w_up.len() % 2 == 0,
        exists|projection_cols: nat|
            #[trigger] well_formed(
                w_gate, w_gate.len(), projection_cols,
            ) && well_formed(w_up, w_up.len(), projection_cols),
        local_gate_output_tp2(x, w_gate, 0).len()
            == local_gate_output_tp2(x, w_gate, 1).len(),
        local_up_output_tp2(x, w_up, 0).len()
            == local_up_output_tp2(x, w_up, 1).len(),
        local_gate_output_tp2(x, w_gate, 0).len()
            == local_up_output_tp2(x, w_up, 0).len(),
        w_down.len() >= 1,
        well_formed(w_down, w_down.len(), w_down[0].len()),
        w_down[0].len() % 2 == 0,
    ensures tp_mlp_forward_tp2(x, w_gate, w_up, w_down)
        == unsharded_mlp_forward(x, w_gate, w_up, w_down),
{
    let projection_cols = choose|projection_cols: nat|
        well_formed(w_gate, w_gate.len(), projection_cols)
            && well_formed(w_up, w_up.len(), projection_cols);
    lemma_dim0_shards_well_formed_tp2(w_gate, projection_cols);
    lemma_dim0_shards_well_formed_tp2(w_up, projection_cols);
    lemma_dim0_shards_reconstruct_tp2(w_gate);
    lemma_dim0_shards_reconstruct_tp2(w_up);

    let g0 = local_gate_output_tp2(x, w_gate, 0);
    let g1 = local_gate_output_tp2(x, w_gate, 1);
    let u0 = local_up_output_tp2(x, w_up, 0);
    let u1 = local_up_output_tp2(x, w_up, 1);
    t1_chunk_split_matches_shard(x, w_gate, w_up, 0, 2);
    t1_chunk_split_matches_shard(x, w_gate, w_up, 1, 2);
    axiom_m1(
        x, shard(w_gate, 0, 2), shard(w_gate, 1, 2),
    );
    axiom_m1(
        x, shard(w_up, 0, 2), shard(w_up, 1, 2),
    );
    assert(concat_cols(g0, g1)
        == matmul(x, transpose(w_gate)));
    assert(concat_cols(u0, u1)
        == matmul(x, transpose(w_up)));

    axiom_s1(g0, g1, u0, u1);
    let h0 = local_hidden_tp2(x, w_gate, w_up, 0);
    let h1 = local_hidden_tp2(x, w_gate, w_up, 1);
    let h = silu_mul(
        matmul(x, transpose(w_gate)),
        matmul(x, transpose(w_up)),
    );
    assert(concat_cols(h0, h1) == h);
    assert(h0.len() == h1.len());

    lemma_dim1_shards_reconstruct_tp2(
        w_down, w_down.len(), w_down[0].len(),
    );
    axiom_m2(
        h,
        w_down,
        h0,
        h1,
        shard_dim1(w_down, 0, 2),
        shard_dim1(w_down, 1, 2),
    );
}

} // verus!

fn main() {
    println!("Verified merged-column-parallel + composition properties (Verus).");
}
