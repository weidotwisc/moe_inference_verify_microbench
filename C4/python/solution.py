"""Exercise 4 — Grouped Query Attention (GQA) under TP, with KV-head replication.

Extends ex03's `QKVParallelLinear` + `TPMHA` to handle:

1. **GQA** — `num_kv_heads < num_heads`. Q and K/V have different total sizes
   in the merged QKV weight, so the offset math must account for that.
2. **KV replication** — when `tp_size > num_kv_heads`, there aren't enough
   KV heads to give one per rank. Multiple ranks share the same KV head
   (each rank holds a full copy of its assigned KV head).

This is the missing piece in nanovllm-jun's TP implementation.  Look at
`nanovllm-jun/nanovllm/models/qwen3.py::Qwen3Attention.__init__`:

    self.total_num_kv_heads = num_kv_heads
    assert self.total_num_kv_heads % tp_size == 0
    self.num_kv_heads = self.total_num_kv_heads // tp_size

That assertion fails at `tp_size=8` for Qwen3-30B-A3B's `num_kv_heads=4`.
Your ex04 solution fixes this by supporting replication.

Fill in:

1. `QKVParallelLinearGQA.__init__` — compute per-rank sizes with the
   `max(1, num_kv_heads // tp_size)` clause, and set `output_size` for the
   parent `ColumnParallelLinear` so the buffer has room for all replicas.
2. `QKVParallelLinearGQA.weight_loader` — same offsets as ex03, but K/V slicing
   uses the replication-aware chunk math.
3. `TPGQA.forward` — same as ex03's TPMHA, plus `repeat_interleave` on K/V
   to broadcast the (fewer) KV heads up to Q's head count before SDPA.

Depends on your ex01 solution.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from bootcamp.ex01_linear_tp.solution import ColumnParallelLinear, RowParallelLinear
from bootcamp.rope import apply_rope, build_rope_cache


class QKVParallelLinearGQA(ColumnParallelLinear):
    """Packed Q/K/V column-parallel linear layer with GQA + KV replication.

    Full "logical" weight layout on the output dim, in Q/K/V order:
        [ Q_head_0 ... Q_head_{H_q - 1}
          K_head_0 ... K_head_{H_kv - 1}
          V_head_0 ... V_head_{H_kv - 1} ]

    Where H_q = num_heads (Q heads) and H_kv = num_kv_heads (K/V heads).
    Under standard GQA H_kv < H_q; ex03's MHA is the special case H_kv == H_q.

    Under TP-N:
      Q is always sharded normally: `num_heads / tp_size` heads per rank.
      K/V are either sharded (tp_size <= num_kv_heads) or replicated
      (tp_size > num_kv_heads).

    Replication rule:
      Define `num_kv_replicas = max(1, tp_size // num_kv_heads)`.
      Rank r stores KV head index `r // num_kv_replicas`.
      Ranks r, r+1, ..., r+num_kv_replicas-1 all hold the same KV head.

    Divisibility invariant (asserted at construction):
      `num_heads % tp_size == 0`  AND
      either `num_kv_heads % tp_size == 0`  OR  `tp_size % num_kv_heads == 0`
      (i.e., one of Q/tp or tp/Q divides cleanly).

    Weight loading uses shard_id in {"q", "k", "v"} same as ex03.

    Args:
        hidden: model hidden dim.
        head_dim: dim of one head. (nanovllm/vLLM call this `head_size`.)
        num_heads: total number of Q heads across all TP ranks.
        num_kv_heads: total number of K (and V) heads across all TP ranks.
                      For MHA num_kv_heads == num_heads; for GQA it's less.
        tp_size: TP world size.
        tp_rank: this rank's id in [0, tp_size).
        group: process group.
    """

    def __init__(
        self,
        hidden: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        tp_size: int,
        tp_rank: int,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        # TODO(you):
        # 1. Assert `num_heads % tp_size == 0` (Q must shard cleanly).
        # 2. Assert `(num_kv_heads % tp_size == 0) or (tp_size % num_kv_heads == 0)`
        #    — one must divide the other for our replication rule to work.
        # 3. Compute:
        #      num_heads_per_rank    = num_heads // tp_size
        #      num_kv_heads_per_rank = max(1, num_kv_heads // tp_size)
        #      num_kv_replicas       = max(1, tp_size // num_kv_heads)
        #    Store on self: head_dim, num_heads, num_kv_heads,
        #                    num_kv_replicas, num_kv_heads_per_rank,
        #                    q_size_per_rank (= num_heads_per_rank * head_dim),
        #                    kv_size_per_rank (= num_kv_heads_per_rank * head_dim).
        # 4. Compute `output_size` for the parent `ColumnParallelLinear`.
        #    HINT: parent expects the TOTAL out-dim across all ranks.
        #    Per-rank storage is (q_size_per_rank + 2 * kv_size_per_rank).
        #    Total = tp_size * (q_size_per_rank + 2 * kv_size_per_rank).
        #    Equivalently:
        #      output_size = (num_heads + 2 * num_kv_heads * num_kv_replicas) * head_dim
        # 5. Call super().__init__(hidden, output_size, tp_size, tp_rank, group=group).
        
        # weiz step1: set up some assertion calls
        assert(num_heads % tp_size ==0)
        assert(num_kv_heads % tp_size == 0 or tp_size % num_kv_heads == 0)

        # weiz step2: setup properties, plain and easy
        self.hidden = hidden
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
       
        # weiz step3: setup shard related info
        self.num_q_heads_per_rank = num_heads // tp_size
        self.num_kv_heads_per_rank = max(1, num_kv_heads // tp_size)
        self.kv_replicas = max(1, tp_size // num_kv_heads) # weiz: how many ranks share one copy of kv_head
        self.q_shard = self.num_q_heads_per_rank * head_dim
        self.k_shard = self.num_kv_heads_per_rank * head_dim
        self.v_shard = self.num_kv_heads_per_rank * head_dim

        in_features = hidden
        out_features = (self.q_shard + self.k_shard + self.v_shard) * tp_size # weiz BUG: recall out_features is a global info, so don't forget to multiply with tp_size
        super().__init__(in_features, out_features, tp_size, tp_rank, group)

    def weight_loader(self, full_weight: torch.Tensor, shard_id: str) -> None:  # type: ignore[override]
        """Copy this rank's slice of one Q, K, or V projection weight into
        self.weight at the correct offset.

        For "q": chunk into `tp_size` and take this rank's chunk (same as ex03).
        For "k"/"v":
          If `num_kv_heads >= tp_size` (no replication):
              chunk into `tp_size`, take this rank's chunk.
          If `num_kv_heads < tp_size` (replicated):
              chunk into `num_kv_heads`, take chunk `tp_rank // num_kv_replicas`
              — multiple ranks receive the same slice.

        full_weight shape:
          "q": [num_heads * head_dim, hidden]
          "k","v": [num_kv_heads * head_dim, hidden]
        """
        # TODO(you):
        # 1. Determine (offset, length) into self.weight based on shard_id.
        # 2. For "q": rank_slice = full_weight.chunk(tp_size, dim=0)[tp_rank]
        # 3. For "k"/"v":
        #      if num_kv_heads >= tp_size:
        #          rank_slice = full_weight.chunk(tp_size, dim=0)[tp_rank]
        #      else:
        #          rank_slice = full_weight.chunk(num_kv_heads, dim=0)[tp_rank // num_kv_replicas]
        # 4. self.weight.data.narrow(0, offset, length).copy_(rank_slice)
        if shard_id == "q":
            self.weight.narrow(dim=0, start=0, length=self.q_shard).data.copy_(full_weight.chunk(chunks=self.tp_size, dim=0)[self.tp_rank])
        elif shard_id == "k":
            self.weight.narrow(dim=0, start=self.q_shard, length=self.k_shard).data.copy_(full_weight.chunk(chunks=self.tp_size // self.kv_replicas, dim=0)[self.tp_rank // self.kv_replicas])
        elif shard_id == "v":
            self.weight.narrow(dim=0, start=self.q_shard+self.k_shard, length=self.v_shard).data.copy_(full_weight.chunk(chunks=self.tp_size // self.kv_replicas, dim=0)[self.tp_rank // self.kv_replicas])


class TPGQA(nn.Module):
    """Full GQA block with TP + KV replication.

    Forward: x → QKVParallelLinearGQA → RoPE(Q, K) → repeat_interleave(K, V)
             → SDPA(causal) → RowParallelLinear.

    The extra step vs ex03's TPMHA is the `repeat_interleave` after RoPE:
    K and V have `num_kv_heads_per_rank` heads, but SDPA needs Q's per-rank
    head count. We broadcast K/V up to match Q by repeating each KV head
    `n_rep = num_heads_per_rank // num_kv_heads_per_rank` times.

    Every rank sees all sequence tokens; heads are sharded (Q) or
    replicated (KV under `tp_size > num_kv_heads`).
    """

    def __init__(
        self,
        hidden: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        tp_size: int,
        tp_rank: int,
        rope_base: float = 10000.0,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        # TODO(you):
        # 1. Assert n_heads % tp_size == 0
        #    AND (n_kv_heads % tp_size == 0 or tp_size % n_kv_heads == 0)
        # 2. Store: hidden, n_heads, n_kv_heads, head_dim,
        #           n_heads_per_rank = n_heads // tp_size,
        #           n_kv_heads_per_rank = max(1, n_kv_heads // tp_size),
        #           n_rep = n_heads_per_rank // n_kv_heads_per_rank,
        #           rope_base, tp_size, tp_rank, group.
        # 3. self.qkv_proj = QKVParallelLinearGQA(hidden, head_dim, n_heads, n_kv_heads, tp_size, tp_rank, group=group)
        # 4. self.o_proj = RowParallelLinear(n_heads * head_dim, hidden, tp_size, tp_rank, group=group)
        
        # weiz: step 1 some setters and getters
        self.hidden = hidden
        self.n_q_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.tp_size = tp_size
        self.tp_rank = tp_rank 
        self.rope_base = rope_base
        # weiz: bug fix, we need the below 3 lines for the k,v replicate during SDPA
        self.num_q_heads_per_rank = n_heads // tp_size
        self.num_kv_heads_per_rank = max(1, n_kv_heads // tp_size)
        self.kv_replicas = max(1, tp_size // n_kv_heads) # weiz: how many ranks share one copy of kv_head

        self.qkv_proj = QKVParallelLinearGQA(hidden=hidden, head_dim=head_dim, 
            num_heads = n_heads, num_kv_heads=n_kv_heads, tp_size=tp_size, tp_rank=tp_rank,
            group=group) # weiz: num_kv_heads is different from n_heads
        
        self.o_proj = RowParallelLinear(in_features = head_dim * n_heads, 
                                           out_features = hidden,tp_size=tp_size, tp_rank=tp_rank, group=group) # bug fix: the in_features is really a global view

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, hidden]
        B, T, hidden = x.shape
        # TODO(you):
        # 1. qkv = self.qkv_proj(x)  # [B, T, q_size + 2 * kv_size] per rank
        # 2. Split into q, k, v with explicit sizes:
        #      q_size  = self.n_heads_per_rank    * self.head_dim
        #      kv_size = self.n_kv_heads_per_rank * self.head_dim
        #      q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
        # 3. Reshape:
        #      q -> [B, T, n_heads_per_rank,    head_dim]
        #      k -> [B, T, n_kv_heads_per_rank, head_dim]
        #      v -> [B, T, n_kv_heads_per_rank, head_dim]
        # 4. Build RoPE cache for `T` positions and apply to q and k (NOT v).
        # 5. Broadcast k, v up to Q's per-rank head count:
        #      k = k.repeat_interleave(self.n_rep, dim=2)
        #      v = v.repeat_interleave(self.n_rep, dim=2)
        #    Now k, v have shape [B, T, n_heads_per_rank, head_dim] — same as q.
        # 6. Transpose q, k, v to [B, n_heads_per_rank, T, head_dim].
        # 7. F.scaled_dot_product_attention(q, k, v, is_causal=True).
        # 8. Transpose back and reshape to [B, T, n_heads_per_rank * head_dim].
        # 9. return self.o_proj(...)  # replicated [B, T, hidden]
        qkv = self.qkv_proj(x) # B x T x [q_shard+k_shard+v_shard]
        q_shard = self.qkv_proj.q_shard
        k_shard = self.qkv_proj.k_shard
        v_shard = self.qkv_proj.v_shard
        q,k,v= torch.split(qkv, [q_shard, k_shard, v_shard], dim=-1) # q|k|v: B, T, q_shard | k_shard | v_shard
        q = q.view((B, T, q_shard // self.head_dim, self.head_dim))
        k = k.view((B, T, k_shard // self.head_dim, self.head_dim))
        v = v.view((B, T, v_shard // self.head_dim, self.head_dim))

        # weiz: build RoPE, TODO, add self-test later
                # Meta's LLaMA-1 reference code — where the "RoPE before transpose" convention originated.
        cos, sin = build_rope_cache(
                    T, self.head_dim, base=self.rope_base, device=x.device, dtype=x.dtype
                )
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # BHTD modern format 
        q=q.transpose(1,2) # weiz: permute it to BHTD so that we can call SDPA/flash-attn
        k=k.transpose(1,2)
        # bug fix: for k and v, we need to hard-bcast via repeat_interleave so that they have the same shape as q in order to do the SPDA (or flash attention)
        k=torch.repeat_interleave(k, repeats=self.num_q_heads_per_rank//self.num_kv_heads_per_rank, dim=1)
        v=v.transpose(1,2)
        v=torch.repeat_interleave(v, repeats=self.num_q_heads_per_rank//self.num_kv_heads_per_rank, dim=1)
        o = nn.functional.scaled_dot_product_attention(q,k,v, is_causal=True) # BHTD
        o= o.transpose(1,2).reshape(B,T,q_shard)
        o = self.o_proj(o)
        return o

        
