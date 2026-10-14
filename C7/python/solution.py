"""Exercise 6 (pure EP) — dispatch is structurally necessary.

Fill in `EPSparseMoE`. Each rank owns a DISTINCT 1/ep_size share of
the batch tokens. Expert compute is sharded across all ranks. Rank r
keeps its own share of the final output.

**Dispatch is unambiguously necessary** here — rank 5 has no way to
compute for rank 0's tokens without receiving them via
`all_to_all_variable`. No `all_reduce` shortcut applies.

## Contract

- **Input**: `local_x` on rank r = `[B, T, H]` or `[local_N, H]`.
  Content is DISTINCT across ranks (e.g., DP-partitioned).
- **Output**: `local_y` on rank r = same shape.
  Contains outputs ONLY for this rank's tokens.

## Contrast with ex06_ep/

| Aspect | ex06_ep (replicated-input) | ex06_ep_pure (this) |
|---|---|---|
| Input pre-condition | Replicated across ep_group | Distinct per rank |
| Router | Redundant across ranks | Local to each rank's slice |
| Dispatch necessary? | No (redundant) | **Yes** |
| Combine | Yes | Yes |
| Final all_gather? | Yes | **No** |
| Alternative? | `all_reduce` (lean) | None — dispatch is load-bearing |

The algorithm body is **almost identical** to ex06_ep's — just remove
Phase 0 (input slicing, not needed) and Phase 11 (all_gather, not
needed). Read [README.md](README.md) for the 11-phase pipeline.

## Run

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run pytest bootcamp/tests/test_ex06_ep_pure.py -v
```
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from bootcamp.ex00_dist_primer.solution import all_to_all_variable
from bootcamp.ref.mlp import RefSwiGLU_MLP


class SwiGLU_MLP(nn.Module):
    def __init__(
        self,
        hidden:int,
        intermediate:int
    ):
        super().__init__()
        self.gate_proj = nn.Linear(in_features=hidden, out_features=intermediate, bias=False) # for MoE FFN, the bias is false
        self.up_proj = nn.Linear(in_features=hidden, out_features=intermediate, bias=False)
        self.down_proj = nn.Linear(in_features=intermediate, out_features=hidden, bias=False)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class EPSparseMoE(nn.Module):
    def __init__(
        self,
        hidden: int,
        intermediate: int,
        num_experts: int,
        top_k: int,
        ep_size: int,
        ep_rank: int,
        group: dist.ProcessGroup | None = None, # weiz: group is the EP_group
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        # TODO(you): same as ex06_ep's __init__.
        # 1. assert num_experts % ep_size == 0
        # 2. Store hyperparams.
        # 3. Compute experts_per_rank / expert_start / expert_end.
        # 4. self.gate = nn.Linear(hidden, num_experts, bias=False) — REPLICATED
        # 5. self.experts = nn.ModuleList of experts_per_rank RefSwiGLU_MLPs — SHARDED
        # weiz: step 1 assert number of experts can be divided
        assert (num_experts % ep_size == 0)
        # weiz: step 2 setters and getters
        self.hidden = hidden
        self.intermediate = intermediate 
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.ep_group = group
        self.norm_topk_prob = norm_topk_prob
        # step 3  experts number per rank
        self.experts_per_rank = num_experts // ep_size
        self.expert_start = self.experts_per_rank * ep_rank
        self.expert_end = self.expert_start + self.experts_per_rank
        # step 4 init the router
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        # step 5 init the experts
        self.experts = nn.ModuleList(SwiGLU_MLP(hidden=hidden, intermediate=intermediate) for _ in range(self.experts_per_rank))

    def weight_loader(
        self,
        gate_weight: torch.Tensor,
        expert_gate_weights: list[torch.Tensor],
        expert_up_weights: list[torch.Tensor],
        expert_down_weights: list[torch.Tensor],
    ) -> None:
        # step 1 load gate (router) weights
        self.gate.weight.data.copy_(gate_weight)
        # step 2 load gate (swiglu), up and down weights
        for i in range(self.experts_per_rank):
            j = self.expert_start + i
            self.experts[i].gate_proj.weight.data.copy_(expert_gate_weights[j])
            self.experts[i].up_proj.weight.data.copy_(expert_up_weights[j])
            self.experts[i].down_proj.weight.data.copy_(expert_down_weights[j])

    def forward(self, local_x: torch.Tensor) -> torch.Tensor:
        """Pure-EP forward — distinct per-rank input, distinct per-rank output.

        Args:
            local_x: [B, T, H] or [local_N, H] — THIS rank's tokens only.
        Returns:
            Same shape as input — output for THIS rank's tokens.
        """
        original_shape = local_x.shape # [B,T,H]
        local_x_flat = local_x.reshape(-1, self.hidden) # [N,H] , N=B*T
        local_N = local_x_flat.shape[0] # N

        # =====================================================================
        # Phase 1: local router on local_x (no input slicing — already local).
        # =====================================================================
        # TODO(you):
        # - router_logits = self.gate(local_x_flat)                       # [local_N, E]
        # - top_k_weights, top_k_experts = torch.topk(...)                # both [local_N, k]
        # - softmax with/without renorm (norm_topk_prob branch)
        # - Cast weights back to local_x.dtype.

        router_logits = self.gate(local_x_flat) # N,E E: number of experts, e.g., 128
        top_k_weights, top_k_experts = torch.topk(router_logits, k=self.top_k) # N,k
        if self.norm_topk_prob:
            top_k_weights = F.softmax(top_k_weights, dim=-1) # N,k
        else:
            top_k_weights = F.softmax(router_logits, dim=-1).gather(dim=-1, index=top_k_experts)

        # =====================================================================
        # Phase 2: local argsort by GLOBAL expert_id.
        # =====================================================================
        # TODO(you):
        # - top_k_experts_flat = top_k_experts.reshape(-1)                # [local_N * k]
        # - top_k_weights_flat = top_k_weights.reshape(-1)                # [local_N * k]
        # - sorted_expert_ids, sort_perm = torch.sort(top_k_experts_flat, stable=True)
        # - local_token_ids = torch.arange(local_N, device=...).repeat_interleave(self.top_k)
        # - sorted_local_token_ids = local_token_ids[sort_perm]           # KEPT LOCAL
        # - sorted_weights         = top_k_weights_flat[sort_perm]        # KEPT LOCAL
        # - sorted_x               = local_x_flat[sorted_local_token_ids] # [local_N*k, H]

        top_k_weights_flat = top_k_weights.reshape(-1) # (Nk,)
        top_k_experts_flat = top_k_experts.reshape(-1) # (Nk,)
        sorted_experts_id, sorted_experts_perm = top_k_experts_flat.sort(dim=-1) # (Nk,)
        local_token_ids_repeated = torch.arange(local_N, device=local_x.device).repeat_interleave(repeats=self.top_k, dim=-1) # (N,)--repeat->(Nk,)
        local_token_ids_repeated_sorted_by_experts = local_token_ids_repeated[sorted_experts_perm] # (Nk,)

        # weiz: prepare X, weights
        local_tokens = local_x_flat[local_token_ids_repeated_sorted_by_experts] # (Nk, H)
        local_weights = top_k_weights_flat[sorted_experts_perm] # (Nk,), bug fix! need to index to top_k_weights_flat, NOT top_k_experts_flat


        # =====================================================================
        # Phase 3: per-destination-rank splits.
        # =====================================================================
        # TODO(you):
        # - dest_ranks         = sorted_expert_ids // self.experts_per_rank
        # - input_split_sizes  = torch.bincount(dest_ranks, minlength=self.ep_size)
        
        # =====================================================================
        # Phase 4: negotiate output splits via all_to_all_single.
        # =====================================================================
        # TODO(you):
        # - output_split_sizes = torch.empty(self.ep_size, dtype=torch.long, device=...)
        # - dist.all_to_all_single(output_split_sizes, input_split_sizes, group=self.group)
        # - Convert both to Python lists.

        dest_ranks = sorted_experts_id // self.experts_per_rank # (Nk, ) now each element is normalized to [0,..., ep_size-1]
        send_cnts_buf = torch.bincount(dest_ranks, minlength=self.ep_size) # (ep_size, ), each element[i] is how many tokens i am sending to rank i 
        recv_cnts_buf = torch.zeros(self.ep_size, dtype=torch.long, device=local_x.device) # (ep_size, ) each element[i] is how many tokens recieved from other gpus
                                                                          # bug fix: we must have the same dtype (aka long) for these two buffers in order to do alltoall
        dist.all_to_all_single(recv_cnts_buf,send_cnts_buf,group=self.ep_group) # now recv_buf_cnts has the desired information from others

        
        # =====================================================================
        # Phase 5: DISPATCH — all_to_all_variable × 2.
        # =====================================================================
        # TODO(you):
        # - received_x           = all_to_all_variable(sorted_x, ..., group=self.group)
        # - received_expert_ids  = all_to_all_variable(sorted_expert_ids, ..., group=self.group)
        send_cnts_lst = send_cnts_buf.tolist()
        recv_cnts_lst = recv_cnts_buf.tolist()
        X_t = all_to_all_variable(local_tokens, send_cnts_lst, recv_cnts_lst,group=self.ep_group) # (Nt, H) Nt, "transpose" of N, which is the received number of tokens, which is the sum(recv_cnts_lst)
               
        eids_t = all_to_all_variable(sorted_experts_id, send_cnts_lst, recv_cnts_lst, group=self.ep_group) # (Nt,)
        # =====================================================================
        # Phase 6: local re-argsort by local expert_id.
        # =====================================================================
        # TODO(you): identical to ex06_ep's Phase 6.
        # - received_local_ids = received_expert_ids - self.expert_start
        # - sorted_local_ids, local_sort_perm = torch.sort(received_local_ids, stable=True)
        # - local_sorted_x = received_x[local_sort_perm]
        # - local_counts + F.pad(cumsum, (1, 0)) → local_offsets

        # =====================================================================
        # Phase 7: local expert compute.
        # =====================================================================
        # TODO(you): identical to ex06_ep's Phase 7.
        # =====================================================================
        # Phase 8: reverse the local sort (scatter local_expert_out → unsorted_received_out).
        # =====================================================================
        # TODO(you):
        # - unsorted_received_out = torch.empty_like(local_expert_out)
        # - unsorted_received_out[local_sort_perm] = local_expert_out
        
        eids_t_sorted, eids_t_perm = torch.sort(eids_t) # (Nt,)
        eids_t_sorted_shifted = eids_t_sorted - self.expert_start # now eids_t_sorted_shifted is within the range of [self.expert_start, self.expert_end)
        eids_t_cnts = torch.bincount(eids_t_sorted_shifted, minlength=self.experts_per_rank) # [self.experts_per_rank,]
        X_t_sorted = X_t[eids_t_perm] # X_t (Nt, H) now is sorted by expert_ids
        Y_t_buf = torch.zeros_like(X_t_sorted)
        Y_t = torch.zeros_like(X_t_sorted)
        start = 0
        for idx,cnt in enumerate(eids_t_cnts.tolist()):
            if cnt==0:
                continue
            Y_t_buf[start:start+cnt] = self.experts[idx](X_t_sorted[start:start+cnt])
            start += cnt
        Y_t[eids_t_perm] = Y_t_buf # a nice trick to avoid create an inverse perm




        

        # =====================================================================
        # Phase 9: COMBINE — reverse all_to_all_variable (swap splits).
        # =====================================================================
        # TODO(you):
        # - returned_out = all_to_all_variable(unsorted_received_out,
        #                                       output_splits, input_splits, group=self.group)

        # =====================================================================
        # Phase 10: weight multiply + local scatter → local_y_flat.
        # =====================================================================
        # TODO(you):
        # - returned_out = returned_out * sorted_weights[:, None]
        # - local_y_flat = torch.zeros(local_N, self.hidden, device=..., dtype=...)
        # - local_y_flat.index_add_(0, sorted_local_token_ids, returned_out)

        Y = all_to_all_variable(Y_t, input_split_sizes=recv_cnts_lst, output_split_sizes=send_cnts_lst, group=self.ep_group) # [Nk, H]conjugate of the previous all_to_all
        Y *= local_weights[:,None] # (Nk,H)
        local_y_flat = torch.zeros_like(local_x_flat)
        local_y_flat.index_add_(dim=0, index=local_token_ids_repeated_sorted_by_experts,source=Y) # gather, the same trick as line 235

        # =====================================================================
        # Phase 11: return local output — NO all_gather, NO all_reduce.
        # =====================================================================
        # TODO(you):
        # - return local_y_flat.reshape(original_shape)

        return local_y_flat.reshape(original_shape)
