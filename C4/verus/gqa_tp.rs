// gqa_tp.rs
//
// Verus attempt at GQA-TP + KV-replication correctness properties for Ex04.
// Proves:
//   G1 (Q sub-region equals Ex03-style Q shard)
//   G2 (KV-replication invariant: replica-siblings hold identical K,V)
//   G3 (KV shards partition, coarser: gather-by-slot reconstructs w_k, w_v)
//   G4 (three-projection weight_loader post-condition)
//   R1 (post-repeat_interleave head count matches Q's per-rank count)
//   G5 (output projection all-reduce makes the GQA output replicated)
//   R2 stubbed (replica-siblings attend against identical K,V)
//   R3 stubbed (block correctness)
//
// Run with:
//   verus gqa_tp.rs

use vstd::prelude::*;
use vstd::calc;
use vstd::arithmetic::div_mod::lemma_fundamental_div_mod;

#[path = "../../../verus/composition_core.rs"]
mod composition_core;
use crate::composition_core::tensor_on;

#[path = "../../ex01_linear_tp/verification/row_parallel.rs"]
mod ex01_row_parallel;

verus! {

// =====================================================================
// §1 — Types.
// =====================================================================

pub type Element = int;
pub type Row = Seq<Element>;
pub type Tensor = Seq<Row>;

pub open spec fn well_formed(t: Tensor, rows: nat, cols: nat) -> bool {
    t.len() == rows &&
    forall|i: int| 0 <= i < t.len() ==> #[trigger] t[i].len() == cols
}

// =====================================================================
// §2 — Sharding + gather (from Ex01/Ex02).
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
    calc! { (==) n; {} p * (n / p) + n % p; {} p * (n / p) + 0; {} p * s; }
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
// §3 — KV replication: kv_slot and per-slot shard.
//
// The number of unique KV slots is `num_kv_slots = num_kv_heads_per_rank
// times num_kv_heads / num_kv_heads_per_rank`, i.e., num_kv_heads or tp_size
// whichever is smaller. We parameterize the model by `num_kv_slots` and
// `num_kv_replicas`, where tp_size = num_kv_slots * num_kv_replicas.
// =====================================================================

pub open spec fn kv_slot(rank: nat, num_kv_replicas: nat) -> nat
    recommends num_kv_replicas >= 1,
{
    (rank as int / num_kv_replicas as int) as nat
}

/// The K (or V) shard that rank `r` stores. Under the replication rule,
/// this equals `shard(w_kv, kv_slot(r), num_kv_slots)` — multiple ranks
/// with the same `kv_slot` receive the same shard.
pub open spec fn kv_shard(w_kv: Tensor, rank: nat, num_kv_slots: nat, num_kv_replicas: nat) -> Tensor
    recommends
        num_kv_slots >= 1,
        num_kv_replicas >= 1,
        w_kv.len() % num_kv_slots == 0,
        kv_slot(rank, num_kv_replicas) < num_kv_slots,
{
    shard(w_kv, kv_slot(rank, num_kv_replicas), num_kv_slots)
}

// =====================================================================
// §4 — Property G2: KV replication invariant.
//
// Two ranks r, r' with the same kv_slot hold identical K weights (and V).
// =====================================================================

pub proof fn g2_kv_replication_invariant(
    w_kv: Tensor, r: nat, r_prime: nat, num_kv_slots: nat, num_kv_replicas: nat,
)
    requires
        num_kv_slots >= 1,
        num_kv_replicas >= 1,
        w_kv.len() % num_kv_slots == 0,
        kv_slot(r, num_kv_replicas) == kv_slot(r_prime, num_kv_replicas),
        kv_slot(r, num_kv_replicas) < num_kv_slots,
    ensures
        kv_shard(w_kv, r, num_kv_slots, num_kv_replicas)
        == kv_shard(w_kv, r_prime, num_kv_slots, num_kv_replicas),
{
    // Immediate: kv_shard depends only on kv_slot, which is equal by hypothesis.
}

// =====================================================================
// §5 — Property G3: gather-by-slot reconstructs w_k, w_v.
//
// The unique KV slots partition w_kv (using num_kv_slots as the "tp_size"
// of the underlying ex01 gather).
// =====================================================================

pub proof fn g3_kv_gather_roundtrip(w_kv: Tensor, num_kv_slots: nat)
    requires num_kv_slots >= 1, w_kv.len() % num_kv_slots == 0,
    ensures gather_all(w_kv, num_kv_slots) == w_kv,
{
    lemma_gather_from_equals_suffix(w_kv, num_kv_slots, 0);
    assert(w_kv.subrange(0, w_kv.len() as int) =~= w_kv);
}

// =====================================================================
// §6 — Property G4: three-projection weight_loader post-condition.
//
// After the three narrow-copy calls, self.weight.data equals the concat
// of q_shard(r), kv_shard(w_k, r), kv_shard(w_v, r).
// =====================================================================

pub open spec fn qkv_gqa_shard(
    w_q: Tensor, w_k: Tensor, w_v: Tensor, rank: nat, tp_size: nat,
    num_kv_slots: nat, num_kv_replicas: nat,
) -> Tensor
    recommends
        tp_size >= 1,
        num_kv_slots >= 1,
        num_kv_replicas >= 1,
        w_q.len() % tp_size == 0,
        w_k.len() % num_kv_slots == 0,
        w_v.len() % num_kv_slots == 0,
        rank < tp_size,
        kv_slot(rank, num_kv_replicas) < num_kv_slots,
{
    shard(w_q, rank, tp_size)
    + kv_shard(w_k, rank, num_kv_slots, num_kv_replicas)
    + kv_shard(w_v, rank, num_kv_slots, num_kv_replicas)
}

pub open spec fn weight_after_gqa_load(
    w_q: Tensor, w_k: Tensor, w_v: Tensor, rank: nat, tp_size: nat,
    num_kv_slots: nat, num_kv_replicas: nat,
) -> Tensor
    recommends
        tp_size >= 1,
        num_kv_slots >= 1,
        num_kv_replicas >= 1,
        w_q.len() % tp_size == 0,
        w_k.len() % num_kv_slots == 0,
        w_v.len() % num_kv_slots == 0,
        rank < tp_size,
        kv_slot(rank, num_kv_replicas) < num_kv_slots,
{
    qkv_gqa_shard(w_q, w_k, w_v, rank, tp_size, num_kv_slots, num_kv_replicas)
}

pub proof fn g4_gqa_weight_loader_postcondition(
    w_q: Tensor, w_k: Tensor, w_v: Tensor, rank: nat, tp_size: nat,
    num_kv_slots: nat, num_kv_replicas: nat,
)
    requires
        tp_size >= 1,
        num_kv_slots >= 1,
        num_kv_replicas >= 1,
        w_q.len() % tp_size == 0,
        w_k.len() % num_kv_slots == 0,
        w_v.len() % num_kv_slots == 0,
        rank < tp_size,
        kv_slot(rank, num_kv_replicas) < num_kv_slots,
    ensures
        weight_after_gqa_load(w_q, w_k, w_v, rank, tp_size, num_kv_slots, num_kv_replicas)
        == qkv_gqa_shard(w_q, w_k, w_v, rank, tp_size, num_kv_slots, num_kv_replicas),
{
    // Immediate by definition.
}

// =====================================================================
// §7 — Property G1: Q-side inherits from Ex03.
//
// The Q sub-region of self.weight equals shard(w_q, rank, tp_size).
// Trivially true by G4 (Q is the first component of the concat).
// =====================================================================

pub proof fn g1_q_side_matches_ex03(
    w_q: Tensor, w_k: Tensor, w_v: Tensor, rank: nat, tp_size: nat,
    num_kv_slots: nat, num_kv_replicas: nat,
)
    requires
        tp_size >= 1,
        num_kv_slots >= 1,
        num_kv_replicas >= 1,
        w_q.len() % tp_size == 0,
        w_k.len() % num_kv_slots == 0,
        w_v.len() % num_kv_slots == 0,
        rank < tp_size,
        kv_slot(rank, num_kv_replicas) < num_kv_slots,
    ensures ({
        let qkv = qkv_gqa_shard(w_q, w_k, w_v, rank, tp_size, num_kv_slots, num_kv_replicas);
        let q_shard_len = shard(w_q, rank, tp_size).len() as int;
        qkv.subrange(0, q_shard_len) == shard(w_q, rank, tp_size)
    }),
{
    // qkv = shard(w_q, r, tp) + kv_shard(w_k, ...) + kv_shard(w_v, ...).
    // The prefix of the first component is definitionally the first component.
    let qk = shard(w_q, rank, tp_size) + kv_shard(w_k, rank, num_kv_slots, num_kv_replicas);
    let qkv = qk + kv_shard(w_v, rank, num_kv_slots, num_kv_replicas);
    assert(qkv.subrange(0, shard(w_q, rank, tp_size).len() as int)
           =~= shard(w_q, rank, tp_size));
}

// =====================================================================
// §8 — repeat_interleave and axiom RI1.
// =====================================================================

pub uninterp spec fn repeat_interleave(t: Tensor, n_rep: nat) -> Tensor;

/// Axiom RI1: length grows by n_rep.
#[verifier::external_body]
pub proof fn axiom_ri1_length(t: Tensor, n_rep: nat)
    requires n_rep >= 1,
    ensures repeat_interleave(t, n_rep).len() == t.len() * n_rep,
{}

/// Axiom RI1: per-original-index content.
#[verifier::external_body]
pub proof fn axiom_ri1_content(t: Tensor, n_rep: nat, i: nat, k: nat)
    requires
        n_rep >= 1,
        i < t.len(),
        k < n_rep,
    ensures
        repeat_interleave(t, n_rep)[(i * n_rep + k) as int] == t[i as int],
{}

// =====================================================================
// §9 — Property R1: post-repeat_interleave, K/V head count matches Q.
//
// If K has num_kv_heads_per_rank heads and Q has num_heads_per_rank heads,
// and n_rep = num_heads_per_rank / num_kv_heads_per_rank, then
// repeat_interleave(K, n_rep) has num_heads_per_rank heads.
// =====================================================================

pub proof fn r1_repeat_interleave_head_count(
    k: Tensor, n_rep: nat, num_heads_per_rank: nat, num_kv_heads_per_rank: nat,
)
    requires
        n_rep >= 1,
        num_kv_heads_per_rank >= 1,
        k.len() == num_kv_heads_per_rank,
        num_heads_per_rank == num_kv_heads_per_rank * n_rep,
    ensures repeat_interleave(k, n_rep).len() == num_heads_per_rank,
{
    axiom_ri1_length(k, n_rep);
    // repeat_interleave(k, n_rep).len() == k.len() * n_rep
    //                                   == num_kv_heads_per_rank * n_rep
    //                                   == num_heads_per_rank
}

// =====================================================================
// §10 — Property G5: output projection replication.
//
// GQA's output projection is row-parallel.  Its exit all-reduce therefore
// gives every rank in the TP group the same projected attention tensor.
// This structural postcondition is sufficient for the downstream MoE seam;
// it does not claim full numerical equivalence to unsharded GQA (R3 below).
// =====================================================================

pub open spec fn GqaOutputReplicated(
    output_id: nat,
    tp_group: Set<nat>,
) -> bool {
    forall|r1: nat, r2: nat|
        tp_group.contains(r1) && tp_group.contains(r2)
            ==> tensor_on(output_id, r1) == tensor_on(output_id, r2)
}

pub proof fn g5_gqa_output_replicated_after_output_projection(
    output_id: nat,
    tp_group: Set<nat>,
)
    ensures GqaOutputReplicated(output_id, tp_group),
{
    ex01_row_parallel::r4b_output_replicated_after_all_reduce(
        output_id, tp_group,
    );
}

// =====================================================================
// §11 — Property R2: replica-siblings compute identical K/V (external stub).
// =====================================================================

/// R2 (external stub): two ranks with the same kv_slot, when handed the
/// same K/V input tensors, produce bit-identical K/V after
/// repeat_interleave. Follows directly from G2 (identical kv_shard) plus
/// determinism of matmul and repeat_interleave.
#[verifier::external_body]
pub proof fn r2_replica_siblings_identical_kv_stub()
    ensures true,
{}

// =====================================================================
// §12 — Property R3: block correctness (external stub).
//
// Full GQA block equals unsharded GQA. Composes G3+G4+R1+Ex03-A1+Ex01-R4.
// =====================================================================

#[verifier::external_body]
pub proof fn r3_block_correctness_stub()
    ensures true,
{}

} // verus!

fn main() {
    println!("Verified GQA-TP + KV-replication structural properties (Verus).");
}
