"""Exercise 5b — Permuted MoE with grouped compute (single-GPU).

Same MoE math as Ex05a, but with a different **data layout**: tokens
are permuted so each expert's inputs are contiguous. Each expert then
runs one grouped matmul on its contiguous slice, instead of scattered
per-token accesses.

## Why this exercise matters

This is the algorithmic bridge between Ex05a and everything after.

**Ex06 (Expert Parallelism)** is essentially this exercise + an
`all_to_all_variable` inserted between the permutation and the
grouped compute. The permutation vocabulary (argsort by expert,
bincount for offsets, unpermute at the end) is exactly what nanovllm's
`_forward_expert_parallel` uses.

**Ex08 (Fused MoE Triton kernel)** is this same algorithm, but with
the Python for-loop replaced by a single Triton kernel launch that
does the grouped matmul. Same permutation vocabulary; same expert
offsets; same unpermute.

If you get comfortable with this permutation dance now, Ex06 and Ex08
feel like small deltas rather than fresh problems.

## The math

Given input `x` of shape `[N, hidden]`, router weights (from Ex05a's
gate + softmax + top-k) `routing_weights` of shape `[N, top_k]` and
`selected_experts` of shape `[N, top_k]`:

1. **Flatten routing to (N * top_k) triples**: `(token_id, expert_id, weight)`.
2. **Argsort by expert_id** to get a permutation. After permuting,
   tokens for expert 0 come first, then expert 1, ..., then expert E-1.
3. **Compute expert offsets** via `bincount(sorted_expert_ids)` +
   `cumsum` → gives per-expert start/end positions.
4. **Gather** the input `x[sorted_token_ids]` into a contiguous
   `sorted_x` of shape `[N * top_k, hidden]`.
5. **Grouped compute**: for each expert e, run `experts[e](sorted_x[offsets[e]:offsets[e+1]])`
   on the CONTIGUOUS slice. One matmul per expert with many tokens,
   instead of many tiny matmuls.
6. **Weight and unpermute**: multiply outputs by `sorted_weights[:, None]`,
   then `index_add_` back into `output` at the original `sorted_token_ids`
   positions (each token's contributions from its top_k experts sum up).

## What to fill in

Fill in `PermutedSparseMoE`. The router (steps 1-4 of Ex05a) is the
same; only the per-expert compute + combine changes.

## Numerical tolerance vs Ex05a

The output is the **same math** as Ex05a's per-expert loop, but the
accumulation order differs (permuted vs by-expert). fp arithmetic
isn't strictly associative, so exact byte equivalence isn't
guaranteed — expect ~1e-5 differences in fp32, ~1e-2 in bf16. Tests
use `assert_close` with generous atol/rtol accordingly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

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

class PermutedSparseMoE(nn.Module):
    """Single-GPU sparse MoE with permutation + grouped compute.

    Args identical to `NaiveSparseMoE` — same abstract module, different
    implementation.
    """

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        # TODO(you): same as Ex05a's __init__. Store hyperparams and allocate
        # self.gate (Linear) + self.experts (ModuleList of RefSwiGLU_MLP).
        self.hidden = hidden
        self.intermediate = intermediate
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.gate = nn.Linear(hidden, num_experts, bias=False) # H,E  e.g 2048, 128, bug fix! bias needs to be false!
        self.experts = nn.ModuleList(SwiGLU_MLP(hidden, intermediate) for _ in range(num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Permuted forward.

        Args:
            x: [B, T, H] or [N, H] input.
        Returns:
            Tensor of the same shape as x.
        """
        original_shape = x.shape # B,T,H
        x_flat = x.reshape(-1, original_shape[-1]) # BxT, H, aka (N,H)
        N = x_flat.shape[0] 
        y_flat = torch.zeros_like(x_flat) # (N,H)
        
        
        # weiz step1: prepare inputs
        # step 1.1 : sort based on router logits
        router_logits = self.gate(x_flat) # N,E
        top_k_weights, top_k_experts = torch.topk(router_logits, k=self.top_k, dim=-1) # both N,k
        if self.norm_topk_prob:
            top_k_weights = F.softmax(top_k_weights,dim=-1) # N,k
        else:
            top_k_weights = F.softmax(router_logits, dim=-1).gather(dim=-1, index=top_k_experts)
        # step 1.2: argsort + perm trick to prepare X sorted based on expert id
        top_k_experts_flat = top_k_experts.reshape(-1) # (Nk,)
        top_k_weights_flat = top_k_weights.reshape(-1) # (Nk,)
        top_k_experts_ids, top_k_experts_permutation = torch.sort(top_k_experts_flat) # both (Nk,)
        expert_bincnt = torch.bincount(top_k_experts_ids, minlength=self.num_experts) # (E,)
        token_ids_rep = torch.repeat_interleave(torch.arange(N, device=x.device), repeats=self.top_k) # (Nk,)
        token_idx_rep_permuted_by_experts = token_ids_rep[top_k_experts_permutation] # (Nk,), already grouped by experts, we just need to figure out the start and offset for each expert
        x_flat_permuted = x_flat[token_idx_rep_permuted_by_experts] # (Nk,H) improvment 1! prepare the input X upfront
        top_k_weights_flat_permuted = top_k_weights_flat[top_k_experts_permutation] # bug fix! (Nk,), now this weights is grouped by expert id
        expert_output = torch.zeros_like(x_flat_permuted) # (Nk,H) improvement 2, having a write buffer, 
        
        # weiz step2: per expert calculation
        start=0
        for expert_id, token_cnt_tensor in enumerate(expert_bincnt):
            token_cnt = token_cnt_tensor.item() # token_cnt is number of tokens corresponding to this expert
            if token_cnt == 0:
                continue
            # get the slice of token ids that routed to this expert 
            tokens_for_this_expert = x_flat_permuted[start:start+token_cnt] #(num_tokens_for_this_expert, H), improvement i already have x_flat_permuted
            # get the scaling factors for this expert for the corresponding tokens
            #experts_output_weights = (top_k_weights_flat_permuted[start:start+token_cnt])[:,None] #(num_tokens_for_this_expert,1), bug fix! we need to directly go into top_k_weights_flat_permuted
            #experts_output = self.experts[expert_id](tokens_for_this_expert) * experts_output_weights # (num_tokens_for_this_expert, H)
            expert_output[start:start+token_cnt]= self.experts[expert_id](tokens_for_this_expert)
            # write into y_flat
            #perm =  token_idx_rep_permuted_by_experts[start:start+token_cnt]
            #y_flat.index_add_(dim=0, index=perm, source=experts_output)
            # update start
            start += token_cnt
        expert_output *= top_k_weights_flat_permuted[:,None] # (Nk,H) improvement! one big scaling, 
        # weiz step 3 collect results back
        y_flat.index_add_(dim=0, index=token_idx_rep_permuted_by_experts, source=expert_output) # imporvement on big scaling
        return y_flat.reshape(original_shape)
            




        
