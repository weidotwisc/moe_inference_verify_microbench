// mha_tp.rs
//
// Verus attempt at multi-head attention TP correctness properties for Ex03.
// Proves:
//   Q1 (per-projection weight_loader post-condition)
//   Q2 (three separate gather roundtrips: w_q, w_k, w_v)
//   Q3 (merged QKV forward correctness, tp=2, via axiom_m1)
//   Q4 (three-way split matches per-projection shards, tp=2)
//   A1 (attention commutes with head-gather, via axiom_attn_head_local)
//   A2 (exact tp=2 block correctness, composes Q3+Q4+A1+M2)
//
// Run with:
//   verus mha_tp.rs
// Expected: `verification results:: N verified, 0 errors`.

use vstd::prelude::*;
use vstd::calc;
use vstd::arithmetic::div_mod::lemma_fundamental_div_mod;

verus! {

// =====================================================================
// §1 — Types (mirror of Ex01/Ex02).
// =====================================================================

pub type Element = int;
pub type Row = Seq<Element>;
pub type Tensor = Seq<Row>;

pub open spec fn well_formed(t: Tensor, rows: nat, cols: nat) -> bool {
    t.len() == rows &&
    forall|i: int| 0 <= i < t.len() ==> #[trigger] t[i].len() == cols
}

// =====================================================================
// §2 — Sharding on dim 0 (from Ex01).
// =====================================================================

pub open spec fn shard(w: Tensor, rank: nat, tp_size: nat) -> Tensor
    recommends tp_size >= 1, w.len() % tp_size == 0, rank < tp_size,
{
    let shard_size = (w.len() / tp_size) as int;
    w.subrange((rank as int) * shard_size, (rank as int + 1) * shard_size)
}

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
    recommends tp_size >= 1, w.len() % tp_size == 0,
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

// =====================================================================
// §3 — Three-way merged shard: this rank's packed (Q, K, V) slice.
// =====================================================================

pub open spec fn qkv_shard(
    w_q: Tensor, w_k: Tensor, w_v: Tensor, rank: nat, tp_size: nat,
) -> Tensor
    recommends
        tp_size >= 1,
        w_q.len() % tp_size == 0,
        w_k.len() % tp_size == 0,
        w_v.len() % tp_size == 0,
        rank < tp_size,
{
    shard(w_q, rank, tp_size) + shard(w_k, rank, tp_size) + shard(w_v, rank, tp_size)
}

pub open spec fn qkv_full(w_q: Tensor, w_k: Tensor, w_v: Tensor) -> Tensor {
    w_q + w_k + w_v
}

// =====================================================================
// §4 — Property Q1: three-projection weight_loader post-condition.
// =====================================================================

pub open spec fn weight_after_qkv_load(
    w_q: Tensor, w_k: Tensor, w_v: Tensor, rank: nat, tp_size: nat,
) -> Tensor
    recommends
        tp_size >= 1,
        w_q.len() % tp_size == 0,
        w_k.len() % tp_size == 0,
        w_v.len() % tp_size == 0,
        rank < tp_size,
{
    qkv_shard(w_q, w_k, w_v, rank, tp_size)
}

pub proof fn q1_qkv_weight_loader_postcondition(
    w_q: Tensor, w_k: Tensor, w_v: Tensor, rank: nat, tp_size: nat,
)
    requires
        tp_size >= 1,
        w_q.len() % tp_size == 0,
        w_k.len() % tp_size == 0,
        w_v.len() % tp_size == 0,
        rank < tp_size,
    ensures
        weight_after_qkv_load(w_q, w_k, w_v, rank, tp_size)
        == qkv_shard(w_q, w_k, w_v, rank, tp_size),
{
    // Immediate by definition.
}

// =====================================================================
// §5 — Property Q2: three-projection gather roundtrips.
// =====================================================================

pub proof fn q2_q_gather_roundtrip(w_q: Tensor, tp_size: nat)
    requires tp_size >= 1, w_q.len() % tp_size == 0,
    ensures gather_all(w_q, tp_size) == w_q,
{
    lemma_gather_from_equals_suffix(w_q, tp_size, 0);
    assert(w_q.subrange(0, w_q.len() as int) =~= w_q);
}

pub proof fn q2_k_gather_roundtrip(w_k: Tensor, tp_size: nat)
    requires tp_size >= 1, w_k.len() % tp_size == 0,
    ensures gather_all(w_k, tp_size) == w_k,
{
    lemma_gather_from_equals_suffix(w_k, tp_size, 0);
    assert(w_k.subrange(0, w_k.len() as int) =~= w_k);
}

pub proof fn q2_v_gather_roundtrip(w_v: Tensor, tp_size: nat)
    requires tp_size >= 1, w_v.len() % tp_size == 0,
    ensures gather_all(w_v, tp_size) == w_v,
{
    lemma_gather_from_equals_suffix(w_v, tp_size, 0);
    assert(w_v.subrange(0, w_v.len() as int) =~= w_v);
}

// =====================================================================
// §6 — Matmul, transpose, attention, and axioms.
// =====================================================================

pub uninterp spec fn matmul(x: Tensor, w_t: Tensor) -> Tensor;
pub uninterp spec fn transpose(t: Tensor) -> Tensor;
pub uninterp spec fn attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor;
pub uninterp spec fn tensor_sum(a: Tensor, b: Tensor) -> Tensor;

pub open spec fn concat_cols(a: Tensor, b: Tensor) -> Tensor
    recommends a.len() == b.len(),
{
    Seq::new(a.len(), |i: int| a[i] + b[i])
}

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

proof fn lemma_well_formed_row_concat(
    a: Tensor,
    b: Tensor,
    cols: nat,
)
    requires
        well_formed(a, a.len(), cols),
        well_formed(b, b.len(), cols),
    ensures well_formed(a + b, a.len() + b.len(), cols),
{
    assert forall|i: int| 0 <= i < (a + b).len() implies
        #[trigger] (a + b)[i].len() == cols by {
        if i < a.len() {
            assert((a + b)[i] == a[i]);
        } else {
            assert((a + b)[i] == b[i - a.len() as int]);
        }
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

/// Axiom AXIOM_M2: the two row-parallel partial matmuls sum to the full
/// matmul, provided both the activation and weight are the exact dim-1
/// concatenations of their shards.
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
    ensures tensor_sum(
        matmul(x0, transpose(w0)),
        matmul(x1, transpose(w1)),
    ) == matmul(x, transpose(w)),
{}

/// Axiom AXIOM_A1: attention commutes with concatenating disjoint head ranges
/// in the flattened feature dimension of each token row.
#[verifier::external_body]
pub proof fn axiom_attn_head_local(
    q0: Tensor, q1: Tensor,
    k0: Tensor, k1: Tensor,
    v0: Tensor, v1: Tensor,
)
    requires
        q0.len() == k0.len(),
        q0.len() == v0.len(),
        q1.len() == k1.len(),
        q1.len() == v1.len(),
        q0.len() == q1.len(),
    ensures
        attention(
            concat_cols(q0, q1),
            concat_cols(k0, k1),
            concat_cols(v0, v1),
        ) == concat_cols(
            attention(q0, k0, v0),
            attention(q1, k1, v1),
        ),
        attention(q0, k0, v0).len() == q0.len(),
        attention(q1, k1, v1).len() == q1.len(),
{}

// =====================================================================
// §7 — Property Q3: Q/K/V projection reconstruction (tp=2).
//
// Each projection's two rank-local output shards reconstruct its unsharded
// output.  An auxiliary lemma below also records the rank-packed layout.
// =====================================================================

pub open spec fn qkv_forward_output(
    x: Tensor, w_q: Tensor, w_k: Tensor, w_v: Tensor, rank: nat, tp_size: nat,
) -> Tensor
    recommends
        tp_size >= 1,
        w_q.len() % tp_size == 0,
        w_k.len() % tp_size == 0,
        w_v.len() % tp_size == 0,
        rank < tp_size,
{
    matmul(x, transpose(qkv_shard(w_q, w_k, w_v, rank, tp_size)))
}

/// Auxiliary fact about concatenating the two rank-packed QKV outputs.  The
/// resulting weight order is `[q0,k0,v0,q1,k1,v1]`, not `qkv_full`; Q3 below
/// performs the semantically relevant per-projection reconstruction.
///
/// Note: qkv_shard = shard(w_q,r) + shard(w_k,r) + shard(w_v,r), and Seq
/// addition is left-associative in Verus. So combining rank-0 and rank-1
/// outputs via axiom_m1 requires two applications: first to combine the
/// [q||k] halves per rank, then the v halves; alternatively, one clean
/// application if we let axiom_m1 handle the whole (qkv_shard_0, qkv_shard_1)
/// pair directly. We take the second route.
proof fn lemma_rank_packed_qkv_forward_tp2(
    x: Tensor, w_q: Tensor, w_k: Tensor, w_v: Tensor,
)
    requires
        w_q.len() >= 2, w_k.len() >= 2, w_v.len() >= 2,
        w_q.len() % 2 == 0, w_k.len() % 2 == 0, w_v.len() % 2 == 0,
        exists|rowlen: nat|
            #[trigger] well_formed(w_q, w_q.len(), rowlen)
         && well_formed(w_k, w_k.len(), rowlen)
         && well_formed(w_v, w_v.len(), rowlen),
        matmul(x, transpose(qkv_shard(w_q, w_k, w_v, 0, 2))).len()
            == matmul(x, transpose(qkv_shard(w_q, w_k, w_v, 1, 2))).len(),
    ensures
        concat_cols(
            qkv_forward_output(x, w_q, w_k, w_v, 0, 2),
            qkv_forward_output(x, w_q, w_k, w_v, 1, 2),
        ) == matmul(x, transpose(qkv_shard(w_q, w_k, w_v, 0, 2)
                                + qkv_shard(w_q, w_k, w_v, 1, 2))),
{
    let s0 = qkv_shard(w_q, w_k, w_v, 0, 2);
    let s1 = qkv_shard(w_q, w_k, w_v, 1, 2);

    // Well-formedness scaffolding: construct a common rowlen for both.
    let rowlen = choose|rowlen: nat|
        well_formed(w_q, w_q.len(), rowlen)
        && well_formed(w_k, w_k.len(), rowlen)
        && well_formed(w_v, w_v.len(), rowlen);
    assert(well_formed(w_q, w_q.len(), rowlen));
    assert(well_formed(w_k, w_k.len(), rowlen));
    assert(well_formed(w_v, w_v.len(), rowlen));

    // Each per-parent shard has rowlen.
    let q0 = shard(w_q, 0, 2); let q1 = shard(w_q, 1, 2);
    let k0 = shard(w_k, 0, 2); let k1 = shard(w_k, 1, 2);
    let v0 = shard(w_v, 0, 2); let v1 = shard(w_v, 1, 2);

    assert(well_formed(q0, q0.len(), rowlen)) by {
        assert forall|i: int| 0 <= i < q0.len() implies #[trigger] q0[i].len() == rowlen by {
            assert(q0[i] == w_q[i]);
        }
    }
    assert(well_formed(q1, q1.len(), rowlen)) by {
        let off = (w_q.len() / 2) as int;
        assert forall|i: int| 0 <= i < q1.len() implies #[trigger] q1[i].len() == rowlen by {
            assert(q1[i] == w_q[off + i]);
            assert(0 <= off + i < w_q.len());
        }
    }
    assert(well_formed(k0, k0.len(), rowlen)) by {
        assert forall|i: int| 0 <= i < k0.len() implies #[trigger] k0[i].len() == rowlen by {
            assert(k0[i] == w_k[i]);
        }
    }
    assert(well_formed(k1, k1.len(), rowlen)) by {
        let off = (w_k.len() / 2) as int;
        assert forall|i: int| 0 <= i < k1.len() implies #[trigger] k1[i].len() == rowlen by {
            assert(k1[i] == w_k[off + i]);
            assert(0 <= off + i < w_k.len());
        }
    }
    assert(well_formed(v0, v0.len(), rowlen)) by {
        assert forall|i: int| 0 <= i < v0.len() implies #[trigger] v0[i].len() == rowlen by {
            assert(v0[i] == w_v[i]);
        }
    }
    assert(well_formed(v1, v1.len(), rowlen)) by {
        let off = (w_v.len() / 2) as int;
        assert forall|i: int| 0 <= i < v1.len() implies #[trigger] v1[i].len() == rowlen by {
            assert(v1[i] == w_v[off + i]);
            assert(0 <= off + i < w_v.len());
        }
    }

    // qkv_shard(_, 0, 2) = (q0 + k0) + v0; case-split rows into three regions.
    assert(s0 == q0 + k0 + v0);
    assert(well_formed(s0, s0.len(), rowlen)) by {
        assert forall|i: int| 0 <= i < s0.len() implies #[trigger] s0[i].len() == rowlen by {
            let q0_end = q0.len() as int;
            let k0_end = q0.len() as int + k0.len() as int;
            if i < q0_end {
                assert((q0 + k0)[i] == q0[i]);
                assert(s0[i] == q0[i]);
            } else if i < k0_end {
                assert((q0 + k0)[i] == k0[i - q0_end]);
                assert(s0[i] == k0[i - q0_end]);
            } else {
                assert(s0[i] == v0[i - k0_end]);
            }
        }
    }
    assert(s1 == q1 + k1 + v1);
    assert(well_formed(s1, s1.len(), rowlen)) by {
        assert forall|i: int| 0 <= i < s1.len() implies #[trigger] s1[i].len() == rowlen by {
            let q1_end = q1.len() as int;
            let k1_end = q1.len() as int + k1.len() as int;
            if i < q1_end {
                assert((q1 + k1)[i] == q1[i]);
                assert(s1[i] == q1[i]);
            } else if i < k1_end {
                assert((q1 + k1)[i] == k1[i - q1_end]);
                assert(s1[i] == k1[i - q1_end]);
            } else {
                assert(s1[i] == v1[i - k1_end]);
            }
        }
    }
    assert(s0.len() > 0);
    assert(s1.len() > 0);
    assert(exists|r: nat|
        #[trigger] well_formed(s0, s0.len(), r)
        && well_formed(s1, s1.len(), r));

    axiom_m1(x, s0, s1);
}

pub proof fn q3_qkv_forward_correctness_tp2(
    x: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
)
    requires
        w_q.len() >= 2,
        w_k.len() >= 2,
        w_v.len() >= 2,
        w_q.len() % 2 == 0,
        w_k.len() % 2 == 0,
        w_v.len() % 2 == 0,
        exists|projection_cols: nat|
            #[trigger] well_formed(
                w_q, w_q.len(), projection_cols,
            ) && well_formed(w_k, w_k.len(), projection_cols)
                && well_formed(w_v, w_v.len(), projection_cols),
        matmul(x, transpose(shard(w_q, 0, 2))).len()
            == matmul(x, transpose(shard(w_q, 1, 2))).len(),
        matmul(x, transpose(shard(w_k, 0, 2))).len()
            == matmul(x, transpose(shard(w_k, 1, 2))).len(),
        matmul(x, transpose(shard(w_v, 0, 2))).len()
            == matmul(x, transpose(shard(w_v, 1, 2))).len(),
    ensures
        concat_cols(
            matmul(x, transpose(shard(w_q, 0, 2))),
            matmul(x, transpose(shard(w_q, 1, 2))),
        ) == matmul(x, transpose(w_q)),
        concat_cols(
            matmul(x, transpose(shard(w_k, 0, 2))),
            matmul(x, transpose(shard(w_k, 1, 2))),
        ) == matmul(x, transpose(w_k)),
        concat_cols(
            matmul(x, transpose(shard(w_v, 0, 2))),
            matmul(x, transpose(shard(w_v, 1, 2))),
        ) == matmul(x, transpose(w_v)),
{
    let projection_cols = choose|projection_cols: nat|
        well_formed(w_q, w_q.len(), projection_cols)
            && well_formed(w_k, w_k.len(), projection_cols)
            && well_formed(w_v, w_v.len(), projection_cols);
    lemma_dim0_shards_well_formed_tp2(w_q, projection_cols);
    lemma_dim0_shards_well_formed_tp2(w_k, projection_cols);
    lemma_dim0_shards_well_formed_tp2(w_v, projection_cols);
    lemma_dim0_shards_reconstruct_tp2(w_q);
    lemma_dim0_shards_reconstruct_tp2(w_k);
    lemma_dim0_shards_reconstruct_tp2(w_v);
    axiom_m1(x, shard(w_q, 0, 2), shard(w_q, 1, 2));
    axiom_m1(x, shard(w_k, 0, 2), shard(w_k, 1, 2));
    axiom_m1(x, shard(w_v, 0, 2), shard(w_v, 1, 2));
}

// =====================================================================
// §8 — Property Q4: three-way split matches per-projection shards.
//
// Stated as a sanity theorem: applying axiom_m1 to (q_shard, k_shard+v_shard),
// then to (k_shard, v_shard), decomposes qkv_out_r into three regions
// matching each projection's per-shard forward.
// =====================================================================

/// Q4: the rank-local packed projection is exactly `[q_r | k_r | v_r]`.
/// Therefore the source `torch.split` returns the three individual shard
/// projections used by the attention proof.
pub proof fn q4_three_way_split(
    x: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    rank: nat,
    tp_size: nat,
)
    requires
        tp_size >= 1,
        rank < tp_size,
        w_q.len() % tp_size == 0,
        w_k.len() % tp_size == 0,
        w_v.len() % tp_size == 0,
        shard(w_q, rank, tp_size).len() > 0,
        shard(w_k, rank, tp_size).len() > 0,
        shard(w_v, rank, tp_size).len() > 0,
        exists|projection_cols: nat|
            #[trigger] well_formed(
                shard(w_q, rank, tp_size),
                shard(w_q, rank, tp_size).len(),
                projection_cols,
            ) && well_formed(
                shard(w_k, rank, tp_size),
                shard(w_k, rank, tp_size).len(),
                projection_cols,
            ) && well_formed(
                shard(w_v, rank, tp_size),
                shard(w_v, rank, tp_size).len(),
                projection_cols,
            ),
        matmul(x, transpose(shard(w_q, rank, tp_size))).len()
            == matmul(x, transpose(shard(w_k, rank, tp_size))).len(),
        matmul(x, transpose(shard(w_q, rank, tp_size))).len()
            == matmul(x, transpose(shard(w_v, rank, tp_size))).len(),
    ensures qkv_forward_output(
        x, w_q, w_k, w_v, rank, tp_size,
    ) == concat_cols(
        concat_cols(
            matmul(x, transpose(shard(w_q, rank, tp_size))),
            matmul(x, transpose(shard(w_k, rank, tp_size))),
        ),
        matmul(x, transpose(shard(w_v, rank, tp_size))),
    ),
{
    let q = shard(w_q, rank, tp_size);
    let k = shard(w_k, rank, tp_size);
    let v = shard(w_v, rank, tp_size);
    let projection_cols = choose|projection_cols: nat|
        well_formed(q, q.len(), projection_cols)
            && well_formed(k, k.len(), projection_cols)
            && well_formed(v, v.len(), projection_cols);
    axiom_m1(x, q, k);
    lemma_well_formed_row_concat(q, k, projection_cols);
    assert(matmul(x, transpose(q + k)).len()
        == matmul(x, transpose(v)).len());
    axiom_m1(x, q + k, v);
}

// =====================================================================
// §9 — Property A1: attention commutes with head-gather (tp=2).
// =====================================================================

pub proof fn a1_attention_head_gather_tp2(
    q0: Tensor, q1: Tensor, k0: Tensor, k1: Tensor, v0: Tensor, v1: Tensor,
)
    requires
        q0.len() == k0.len(),
        q0.len() == v0.len(),
        q1.len() == k1.len(),
        q1.len() == v1.len(),
        q0.len() == q1.len(),
    ensures
        concat_cols(
            attention(q0, k0, v0),
            attention(q1, k1, v1),
        ) == attention(
            concat_cols(q0, q1),
            concat_cols(k0, k1),
            concat_cols(v0, v1),
        ),
        attention(q0, k0, v0).len() == q0.len(),
        attention(q1, k1, v1).len() == q1.len(),
{
    axiom_attn_head_local(q0, q1, k0, k1, v0, v1);
}

// =====================================================================
// §10 — Property A2: block correctness (tp_size == 2).
//
// The full MHA block output equals the unsharded MHA output.  This composes
// Q3, Q4, A1, dim-1 output-weight reconstruction, and AXIOM_M2.
// =====================================================================

pub open spec fn local_projection_tp2(
    x: Tensor,
    w: Tensor,
    rank: nat,
) -> Tensor
    recommends rank < 2, w.len() % 2 == 0,
{
    matmul(x, transpose(shard(w, rank, 2)))
}

pub open spec fn local_attention_tp2(
    x: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    rank: nat,
) -> Tensor
    recommends
        rank < 2,
        w_q.len() % 2 == 0,
        w_k.len() % 2 == 0,
        w_v.len() % 2 == 0,
{
    attention(
        local_projection_tp2(x, w_q, rank),
        local_projection_tp2(x, w_k, rank),
        local_projection_tp2(x, w_v, rank),
    )
}

pub open spec fn tp2_projection_shapes_match(
    x: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
) -> bool {
    let q0 = local_projection_tp2(x, w_q, 0);
    let q1 = local_projection_tp2(x, w_q, 1);
    let k0 = local_projection_tp2(x, w_k, 0);
    let k1 = local_projection_tp2(x, w_k, 1);
    let v0 = local_projection_tp2(x, w_v, 0);
    let v1 = local_projection_tp2(x, w_v, 1);
    &&& q0.len() == q1.len()
    &&& q0.len() == k0.len()
    &&& q0.len() == k1.len()
    &&& q0.len() == v0.len()
    &&& q0.len() == v1.len()
}

/// Exact two-rank TP execution after expanding the packed projection with Q4.
pub open spec fn tp_mha_forward_tp2(
    x: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    w_o: Tensor,
) -> Tensor
    recommends
        w_q.len() % 2 == 0,
        w_k.len() % 2 == 0,
        w_v.len() % 2 == 0,
        w_o.len() >= 1,
        w_o[0].len() % 2 == 0,
{
    tensor_sum(
        matmul(
            local_attention_tp2(x, w_q, w_k, w_v, 0),
            transpose(shard_dim1(w_o, 0, 2)),
        ),
        matmul(
            local_attention_tp2(x, w_q, w_k, w_v, 1),
            transpose(shard_dim1(w_o, 1, 2)),
        ),
    )
}

pub open spec fn unsharded_mha_forward(
    x: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    w_o: Tensor,
) -> Tensor {
    matmul(
        attention(
            matmul(x, transpose(w_q)),
            matmul(x, transpose(w_k)),
            matmul(x, transpose(w_v)),
        ),
        transpose(w_o),
    )
}

/// A2: exact tp=2 QKV projection, head-local attention, output projection,
/// and all-reduce equal the unsharded MHA block.
pub proof fn a2_block_correctness_tp2(
    x: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    w_o: Tensor,
)
    requires
        w_q.len() >= 2,
        w_k.len() >= 2,
        w_v.len() >= 2,
        w_q.len() % 2 == 0,
        w_k.len() % 2 == 0,
        w_v.len() % 2 == 0,
        exists|projection_cols: nat|
            #[trigger] well_formed(
                w_q, w_q.len(), projection_cols,
            ) && well_formed(w_k, w_k.len(), projection_cols)
                && well_formed(w_v, w_v.len(), projection_cols),
        tp2_projection_shapes_match(x, w_q, w_k, w_v),
        w_o.len() >= 1,
        well_formed(w_o, w_o.len(), w_o[0].len()),
        w_o[0].len() % 2 == 0,
    ensures tp_mha_forward_tp2(x, w_q, w_k, w_v, w_o)
        == unsharded_mha_forward(x, w_q, w_k, w_v, w_o),
{
    let projection_cols = choose|projection_cols: nat|
        well_formed(w_q, w_q.len(), projection_cols)
            && well_formed(w_k, w_k.len(), projection_cols)
            && well_formed(w_v, w_v.len(), projection_cols);
    lemma_dim0_shards_well_formed_tp2(w_q, projection_cols);
    lemma_dim0_shards_well_formed_tp2(w_k, projection_cols);
    lemma_dim0_shards_well_formed_tp2(w_v, projection_cols);

    let q0 = local_projection_tp2(x, w_q, 0);
    let q1 = local_projection_tp2(x, w_q, 1);
    let k0 = local_projection_tp2(x, w_k, 0);
    let k1 = local_projection_tp2(x, w_k, 1);
    let v0 = local_projection_tp2(x, w_v, 0);
    let v1 = local_projection_tp2(x, w_v, 1);
    q4_three_way_split(x, w_q, w_k, w_v, 0, 2);
    q4_three_way_split(x, w_q, w_k, w_v, 1, 2);
    q3_qkv_forward_correctness_tp2(x, w_q, w_k, w_v);

    let a0 = local_attention_tp2(x, w_q, w_k, w_v, 0);
    let a1 = local_attention_tp2(x, w_q, w_k, w_v, 1);
    a1_attention_head_gather_tp2(q0, q1, k0, k1, v0, v1);
    let full_attention = attention(
        matmul(x, transpose(w_q)),
        matmul(x, transpose(w_k)),
        matmul(x, transpose(w_v)),
    );
    assert(concat_cols(a0, a1) == full_attention);
    assert(a0.len() == a1.len());

    lemma_dim1_shards_reconstruct_tp2(
        w_o, w_o.len(), w_o[0].len(),
    );
    axiom_m2(
        full_attention,
        w_o,
        a0,
        a1,
        shard_dim1(w_o, 0, 2),
        shard_dim1(w_o, 1, 2),
    );
}

} // verus!

fn main() {
    println!("Verified MHA-TP structural properties (Verus).");
}
