"""Exercise 7 — TP-4 × DP-2 × EP-8 hybrid block (paper's centerpiece).

Fill in `HybridMoE` (the novel piece) and `HybridBlock` (thin composition
wrapper). The setup:

- 8 GPUs total, split into two TP groups: {0,1,2,3}, {4,5,6,7}.
- Each TP group processes a distinct half of the batch (DP-2 outer).
- One EP group over all 8 ranks; MoE dispatch crosses TP-group boundaries.

## The sub-group composition

Rank 0 belongs to `tp_group_a` AND `ep_group`. Its collective sequence
per block:

  attn AR (tp_group_a)
  → moe splits negotiation (ep_group)
  → moe dispatch × 2 (ep_group)
  → moe combine (ep_group)
  → moe all_gather (tp_group_a)

Every rank issues this exact same sequence with its own tp_group handle.
The paper's composition theorem proves this cross-cutting sub-group
schedule cannot deadlock under fixed-schedule + explicit-group discipline.

## What differs from Ex04 + Ex06

- **Ex04's TPGQA** is reused unchanged for attention.
- **HybridMoE (new)** stripes within tp_group, dispatches across ep_group,
  all_gathers within tp_group. Contrast to Ex06's variants:
  - `ex06_ep/reference.py` used ep_group for ALL collectives (mis-scoped).
  - `ex06_ep_pure/reference.py` used only ep_group (no TP subsumption).
  - **This Ex07 version uses BOTH tp_group and ep_group.**

## Block-level invariant

  { input replicated within tp_group } → forward → { output replicated within tp_group }

## Run

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run pytest bootcamp/tests/test_ex07_tp_ep_hybrid.py -v
```
"""

from __future__ import annotations
import torch
import torch.distributed as dist
from torch.masked import norm
import torch.nn.functional as F
from torch import nn

from bootcamp.ex00_dist_primer.solution import all_to_all_variable
#from bootcamp.ex04_gqa_tp.reference import TPGQA
from bootcamp.ex04_gqa_tp.reference import TPGQA # weiz: use my own impl of TPGQA
from bootcamp.ref.block import RMSNorm
from bootcamp.ref.mlp import RefSwiGLU_MLP


class HybridMoE(nn.Module):
    """MoE with TP-scoped striping/gather + EP-scoped dispatch/combine.

    Args:
        hidden, intermediate, num_experts, top_k, norm_topk_prob — MoE dims.
        tp_size, tp_rank, tp_group — TP within-group parameters for striping + all_gather.
        ep_size, ep_rank, ep_group — EP world-spanning parameters for dispatch + combine.
    """

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        num_experts: int,
        top_k: int,
        tp_size: int,
        tp_rank: int,
        tp_group: dist.ProcessGroup | None,
        ep_size: int,
        ep_rank: int,
        ep_group: dist.ProcessGroup | None,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        # TODO(you):
        # 1. assert num_experts % ep_size == 0
        # 2. Store hyperparams (hidden, intermediate, num_experts, top_k,
        #    tp_size, tp_rank, tp_group, ep_size, ep_rank, ep_group, norm_topk_prob).
        # 3. self.experts_per_rank = num_experts // ep_size
        #    self.expert_start     = ep_rank * self.experts_per_rank
        #    self.expert_end       = self.expert_start + self.experts_per_rank
        # 4. self.gate = nn.Linear(hidden, num_experts, bias=False)
        #    — REPLICATED across tp_group (identical routing per TP peer).
        # 5. self.experts = nn.ModuleList of experts_per_rank RefSwiGLU_MLPs
        #    — SHARDED across ep_group (each rank owns experts_per_rank).
        
        # weiz 2026-08-23
        assert(num_experts % ep_size == 0)

        self.hidden = hidden
        self.intermediate = intermediate
        self.num_experts = num_experts
        self.top_k = top_k
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.ep_group = ep_group
        self.norm_topk_prob = norm_topk_prob

        self.experts_per_rank = num_experts // ep_size
        self.expert_start = self.ep_rank * self.experts_per_rank
        self.expert_end = self.expert_start + self.experts_per_rank

        self.gate = nn.Linear(self.hidden, self.num_experts, bias=False)

        self.experts = nn.ModuleList(RefSwiGLU_MLP(hidden, intermediate) for _ in range(self.experts_per_rank)) # bug fix: should be range(experts_per_rank), not the entire experts!


    def weight_loader(
        self,
        gate_weight: torch.Tensor,
        expert_gate_weights: list[torch.Tensor],
        expert_up_weights: list[torch.Tensor],
        expert_down_weights: list[torch.Tensor],
    ) -> None:
        # TODO(you):
        # - self.gate.weight.data.copy_(gate_weight)
        # - Loop over local_e ∈ [0, experts_per_rank), copy the three expert
        #   projections at global_e = expert_start + local_e into self.experts[local_e].
        
        # weiz 2026-08-23
        self.gate.weight.data.copy_(gate_weight)
        for i in range(self.experts_per_rank):
            j = i + self.expert_start
            self.experts[i].gate_proj.weight.data.copy_(expert_gate_weights[j])
            self.experts[i].up_proj.weight.data.copy_(expert_up_weights[j])
            self.experts[i].down_proj.weight.data.copy_(expert_down_weights[j])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H] replicated within tp_group.
        Returns:
            [B, T, H] replicated within tp_group.
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.hidden)
        N_tp = x_flat.shape[0]
        assert N_tp % self.tp_size == 0
        local_N = N_tp // self.tp_size # the "faked-sharded" data size in the TP group 

        # =====================================================================
        # Phase 0: stripe within TP group.
        # Every rank in this TP group has identical [N_tp, H]. Take a distinct
        # 1/tp_size slice so MoE work isn't tp_size× redundant across the group.
        # =====================================================================
        # TODO(you):
        # - local_x = x_flat[self.tp_rank * local_N : (self.tp_rank + 1) * local_N]

        local_x = x_flat[self.tp_rank * local_N: (self.tp_rank+1) * local_N] # (local_N, H)
        local_y = torch.zeros_like(local_x) # (local_N, H), weiz the output
        y_flat = torch.zeros_like(x_flat) # [N,H] the same as true x_flat
        # =====================================================================
        # Phase 1: local router on local_x.
        # =====================================================================
        # TODO(you): identical to Ex06 phase 1.
        # - router_logits = self.gate(local_x)
        # - topk + softmax (with/without norm_topk_prob), cast to dtype.

        # step1: get the top k logits for each token
        router_logits = self.gate(local_x) # (local_N, num_experts)
        router_logits_top_k, top_k_expert_ids = torch.topk(router_logits, k=self.top_k, dim=-1) # (local_N, k)
        if self.norm_topk_prob:
            weights_top_k = F.softmax(router_logits_top_k, dim=-1)
        else:
            weights_top_k = F.softmax(router_logits).gather(dim=-1, index=top_k_expert_ids) #(local_N,k), bug fix, should have used gather() instead of fancy indexing 

        # step2: sort tokens by the expert ids
        top_k_expert_ids_flat = top_k_expert_ids.reshape(-1) # (local_N*k,)
        weights_top_k_flat = weights_top_k.reshape(-1) # (local_N*k,)
        top_k_expert_ids_flat_sorted, top_k_expert_ids_perm = torch.sort(top_k_expert_ids_flat, stable=True) # (local_N*k,)
        top_k_expert_ids_perm_normalized = top_k_expert_ids_perm // self.top_k # (local_N*k,), but every number is normalized within [0, local_N), bug fix!! should use top_k_expert_ids_perm // 
        x_repeated_k = local_x[top_k_expert_ids_perm_normalized] # (local_N*k, H), bug fix: should have used local_x not x_flat
        y_repeated_k_sorted = torch.zeros_like(x_repeated_k) # (local_N*k, H), get the result from the last all_to_all call
        y_repeated_k = torch.zeros_like(x_repeated_k) # bug fix, we need to have both sorted and unsorted y to play the perm trick!

        # step3: 1st all_to_all call negotiate sending and receiving cnts
        top_k_expert_ids_flat_sorted_input_cnts = torch.bincount(top_k_expert_ids_flat_sorted // self.experts_per_rank, minlength=self.ep_size) # bug fix: should be // self.expert_per_rank not self.ep_size, (ep_size,), how many tokens i am sending to other ranks in EP group
        top_k_expert_ids_flat_sorted_output_cnts = torch.zeros_like(top_k_expert_ids_flat_sorted_input_cnts) # (ep_size,), how many tokens i am receiving from other ranks in EP group
        dist.all_to_all_single(top_k_expert_ids_flat_sorted_output_cnts, top_k_expert_ids_flat_sorted_input_cnts, group=self.ep_group)

        # step 4: 2nd all_to_alls to scatter my tokens to others and corresponding expert ids 
        input_cnts_list = top_k_expert_ids_flat_sorted_input_cnts.tolist()
        output_cnts_list = top_k_expert_ids_flat_sorted_output_cnts.tolist()
        x_local_T = torch.zeros(size=(sum(output_cnts_list), original_shape[-1]), dtype=x.dtype, device=x.device) # (local_N_r_t, H), local_N_r_ t means the conjugate of local_N repeated k times
        y_local_T = torch.zeros_like(x_local_T) # output in the original order (local_N_t, H)
        expert_ids_local_T = torch.zeros(size=(sum(output_cnts_list),), dtype=torch.long, device=x.device) #(local_N_r_t, )
        dist.all_to_all_single(x_local_T, x_repeated_k, output_cnts_list, input_cnts_list, group=self.ep_group)
        dist.all_to_all_single(expert_ids_local_T, top_k_expert_ids_flat_sorted, output_cnts_list, input_cnts_list, group=self.ep_group)
        expert_ids_local_T -= self.expert_start # bug fix: make it start from 0
        # step 5: local expert MoE fwd via sorting and etc
        expert_ids_local_T_sorted, expert_ids_local_T_perm = torch.sort(expert_ids_local_T, stable=True) # (local_N_r_t, )
        x_local_T_sorted = x_local_T[expert_ids_local_T_perm] # (local_N_r_t, H)
        y_local_T_sorted = torch.zeros_like(x_local_T_sorted) # (local_N_r_t, H) , output
        expert_ids_local_T_sorted_cnts = torch.bincount(expert_ids_local_T_sorted, minlength=self.experts_per_rank) # (experts_per_rank, )
        start = 0 
        for idx, cnt in enumerate(expert_ids_local_T_sorted_cnts.tolist()):
            if cnt == 0:
                continue
            y_local_T_sorted[start:start+cnt] = self.experts[idx](x_local_T_sorted[start:start+cnt])
            start += cnt
        y_local_T[expert_ids_local_T_perm] = y_local_T_sorted # reverse the sorted order so that y_local_T is ready to be sent back to the other ranks in the EP_group

        # step 6: 3rd all_to_alls to gather the outputs from other ranks in EP group
        dist.all_to_all_single(y_repeated_k_sorted, y_local_T, output_split_sizes=input_cnts_list, input_split_sizes=output_cnts_list, group=self.ep_group) # conjugate call of the 2nd alltoall


        # step 7: (1) revert local_y to the original unsorted by expert id to match the original input and then 
        #.        (2) multiply with weights
        #         (3) add back to the original token output (unexpanded by topk experts)
        y_repeated_k[top_k_expert_ids_perm] = y_repeated_k_sorted # (local_N*k, H)  (1) revert y_repeated_k to the original unsorted by expert id to match the original input
                                                                # big bug fix: need to play the perm trick as y[perm]=y_sorted
        y_repeated_k *= weights_top_k_flat[:,None] # (local_N*k, H) (2) multiply with weights
        local_y.index_add_(dim=0, index=torch.arange(local_N, device=x.device).repeat_interleave(repeats=self.top_k,dim=-1), source=y_repeated_k) # (local_N, h)

        # step 8 allgather to honor TP semantics
        dist.all_gather_into_tensor(y_flat, local_y, group=self.tp_group)

        # step 9 reshape y_flat to original x shape and return 
        return y_flat.reshape(original_shape)

        


class HybridBlock(nn.Module):
    """One transformer block under TP-4 × DP-2 × EP-8.

    ```
    h = x + attn(rmsnorm(x))   # attention: TP-scoped
    y = h + moe(rmsnorm(h))    # moe: TP-scoped stripe + EP-scoped dispatch + TP-scoped gather
    ```
    """

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        num_experts: int,
        top_k: int,
        tp_size: int,
        tp_rank: int,
        tp_group: dist.ProcessGroup | None,
        ep_size: int,
        ep_rank: int,
        ep_group: dist.ProcessGroup | None,
        norm_topk_prob: bool = True,
        rope_base: float = 10000.0,
        rms_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        # TODO(you):
        # 1. self.attn_norm = RMSNorm(hidden, eps=rms_eps)
        # 2. self.attn = TPGQA(hidden, n_heads, n_kv_heads, head_dim,
        #                       tp_size=tp_size, tp_rank=tp_rank,
        #                       group=tp_group, rope_base=rope_base)
        # 3. self.moe_norm = RMSNorm(hidden, eps=rms_eps)
        # 4. self.moe = HybridMoE(hidden, intermediate, num_experts, top_k,
        #                          tp_size, tp_rank, tp_group,
        #                          ep_size, ep_rank, ep_group,
        #                          norm_topk_prob=norm_topk_prob)
        
        # weiz 2026-08-23, this is pre-norm
        self.attn_norm = RMSNorm(hidden, eps=rms_eps)
        self.attn = TPGQA(hidden, n_heads, n_kv_heads, head_dim, 
                          rope_base=rope_base,  
                          tp_size=tp_size, tp_rank=tp_rank, group=tp_group)
        self.moe_norm = RMSNorm(hidden, eps=rms_eps)
        self.moe = HybridMoE(hidden, intermediate, num_experts, top_k,
                             tp_size, tp_rank, tp_group,
                             ep_size, ep_rank, ep_group,
                             norm_topk_prob)
        
        

    def weight_loader(
        self,
        attn_norm_weight: torch.Tensor,
        q_weight: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor,
        o_weight: torch.Tensor,
        moe_norm_weight: torch.Tensor,
        gate_weight: torch.Tensor,
        expert_gate_weights: list[torch.Tensor],
        expert_up_weights: list[torch.Tensor],
        expert_down_weights: list[torch.Tensor],
    ) -> None:
        # TODO(you):
        # - Copy attn_norm.weight, moe_norm.weight directly (replicated within tp_group).
        # - Delegate QKV weights to self.attn.qkv_proj.weight_loader(..., shard_id)
        #   with shard_id ∈ {"q", "k", "v"}.
        # - self.attn.o_proj.weight_loader(o_weight)
        # - self.moe.weight_loader(gate_weight, expert_gate_weights, ...)
        
        # weiz 2026-08-23, don't forget to load weight for attn_norm
        self.attn_norm.weight.data.copy_(attn_norm_weight) 
        # weiz 2026-08-23, NOTICE the TPGQA doesn't need a weight_loader, as it is just a container and doesn't own any Parameters
        self.attn.qkv_proj.weight_loader(full_weight=q_weight,shard_id="q")
        self.attn.qkv_proj.weight_loader(full_weight=k_weight,shard_id="k")
        self.attn.qkv_proj.weight_loader(full_weight=v_weight,shard_id="v")
        self.attn.o_proj.weight_loader(full_weight=o_weight)
        # weiz 2026-08-23, don't forget to load weight for moe_norm_weight
        self.moe_norm.weight.data.copy_(moe_norm_weight)
        self.moe.weight_loader(gate_weight, 
                               expert_gate_weights, expert_up_weights,expert_down_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(you): the two-line residual+norm composition.
        # - h = x + self.attn(self.attn_norm(x))
        # - y = h + self.moe(self.moe_norm(h))
        # - return y
        h = x + self.attn(self.attn_norm(x)) # weiz: prenorm
        y = h + self.moe(self.moe_norm(h)) 
        return y