"""Reference implementation for Ex06 — lean variant.

Same MoE semantics as reference.py, but exploits the replicated-input
pre-condition to skip the dispatch collective entirely. **One all_reduce
per forward** instead of dispatch + combine + all_gather.

## Design contrast with reference.py

**reference.py** (dispatch-based, matches nanovllm-jun / vLLM):
- Contiguous token partition per rank (Phase 0).
- Local router on 1/ep_size share.
- Two `all_to_all_variable` calls (dispatch + combine).
- Final `all_gather` to reassemble replicated output.
- **4 collectives per forward.**

**reference_lean.py** (this file, all_reduce-based):
- No token partition; every rank runs full router.
- Filter routing records to local experts.
- Local compute.
- Single `all_reduce` to sum partial contributions.
- **1 collective per forward.**

Both produce identical output (up to fp reduction-order noise). Both
maintain the block-level invariant: `[N, H] replicated on ep_group`
→ forward → `[N, H] replicated on ep_group`.

## Why this variant exists

For inference at `ep_size ≤ top_k + 1` with replicated input,
this schedule is measurably better:

- **Fewer collective launches** (1 vs 4 → ~60μs startup savings).
- **Fewer bytes transferred** (~14% less at ep=8, top_k=8).
- **Data-independent bandwidth**: no popular-expert bottleneck; every
  rank's collective payload is `[N, H]` regardless of routing skew.
- **Simpler code**: no splits negotiation, no local re-argsort.

Production stacks (vLLM, nanovllm-jun, Megatron-DeepSpeed) default to
dispatch-based scheduling for training-inference code path uniformity
— dispatch generalizes to training with data-parallel-partitioned
input where all_reduce doesn't. For **inference-specific EP with
replicated input**, this lean variant is preferable.

See [README.md](README.md) §"Design choice: dispatch vs all_reduce".
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from bootcamp.ref.mlp import RefSwiGLU_MLP


class EPSparseMoE(nn.Module):
    """Expert-parallel sparse MoE — lean (all_reduce) variant.

    Interface identical to the dispatch-based `reference.py::EPSparseMoE`.
    Only the forward-pass schedule differs.
    """

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        num_experts: int,
        top_k: int,
        ep_size: int,
        ep_rank: int,
        group: dist.ProcessGroup | None = None,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        assert num_experts % ep_size == 0, (
            f"num_experts={num_experts} not divisible by ep_size={ep_size}"
        )
        self.hidden = hidden
        self.intermediate = intermediate
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.group = group
        self.norm_topk_prob = norm_topk_prob
        self.experts_per_rank = num_experts // ep_size
        self.expert_start = ep_rank * self.experts_per_rank
        self.expert_end = self.expert_start + self.experts_per_rank

        # REPLICATED — full gate weight on every rank.
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        # SHARDED — only this rank's experts_per_rank MLPs.
        self.experts = nn.ModuleList(
            [RefSwiGLU_MLP(hidden, intermediate) for _ in range(self.experts_per_rank)]
        )

    def weight_loader(
        self,
        gate_weight: torch.Tensor,
        expert_gate_weights: list[torch.Tensor],
        expert_up_weights: list[torch.Tensor],
        expert_down_weights: list[torch.Tensor],
    ) -> None:
        self.gate.weight.data.copy_(gate_weight)
        for local_e in range(self.experts_per_rank):
            global_e = self.expert_start + local_e
            self.experts[local_e].gate_proj.weight.data.copy_(expert_gate_weights[global_e])
            self.experts[local_e].up_proj.weight.data.copy_(expert_up_weights[global_e])
            self.experts[local_e].down_proj.weight.data.copy_(expert_down_weights[global_e])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lean EP forward.

        Args:
            x: [B, T, H] or [N, H], REPLICATED across ep_group.
        Returns:
            [B, T, H] or [N, H], REPLICATED (via one all_reduce).
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1])
        N = x_flat.shape[0]

        # ================ Phase 1: FULL router on all N tokens ================
        # Every rank runs the same router on the replicated input.
        # Deterministic — every rank gets identical routing decisions.
        router_logits = self.gate(x_flat)                                    # [N, E]
        top_k_weights, top_k_experts = torch.topk(router_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_k_weights = F.softmax(top_k_weights, dim=-1)                 # [N, k]
        else:
            top_k_weights = F.softmax(router_logits, dim=-1).gather(-1, top_k_experts)
        top_k_weights = top_k_weights.to(x_flat.dtype)

        # ================ Phase 2: filter routing records to LOCAL experts ================
        # Each rank keeps only records where the destination expert is local.
        top_k_experts_flat = top_k_experts.reshape(-1)                       # [N * k]
        top_k_weights_flat = top_k_weights.reshape(-1)                       # [N * k]
        token_ids_flat = torch.arange(N, device=x.device).repeat_interleave(self.top_k)

        local_mask = (
            (top_k_experts_flat >= self.expert_start)
            & (top_k_experts_flat < self.expert_end)
        )
        local_idx = local_mask.nonzero(as_tuple=True)[0]                     # [num_local]

        local_expert_ids = top_k_experts_flat[local_idx] - self.expert_start  # [0, experts_per_rank)
        local_weights = top_k_weights_flat[local_idx]
        local_token_ids = token_ids_flat[local_idx]

        # ================ Phase 3: local argsort by local expert_id ================
        sorted_local_ids, sort_perm = torch.sort(local_expert_ids, stable=True)
        sorted_weights = local_weights[sort_perm]
        sorted_token_ids = local_token_ids[sort_perm]
        # Local fancy-index gather — no network involved (x_flat is already local).
        sorted_x = x_flat[sorted_token_ids]                                  # [num_local, H]

        # ================ Phase 4: bincount + offsets ================
        local_counts = torch.bincount(sorted_local_ids, minlength=self.experts_per_rank)
        local_offsets = F.pad(local_counts.cumsum(0), (1, 0))                # [experts_per_rank + 1]

        # ================ Phase 5: local expert compute ================
        expert_out = torch.empty_like(sorted_x)
        for local_e in range(self.experts_per_rank):
            s = local_offsets[local_e].item()
            e = local_offsets[local_e + 1].item()
            if s == e:
                continue
            expert_out[s:e] = self.experts[local_e](sorted_x[s:e])
        expert_out = expert_out * sorted_weights[:, None]                    # weight per (token, expert)

        # ================ Phase 6: build partial output [N, H] ================
        # Every rank's partial has zeros for tokens NOT routed to its local experts.
        # After all_reduce, the sum recovers the full contribution per token.
        partial_output = torch.zeros(N, self.hidden, device=x.device, dtype=x.dtype)
        partial_output.index_add_(0, sorted_token_ids, expert_out)

        # ================ Phase 7: ONE all_reduce — assembles full output on every rank ================
        # In-place. Payload: N·H bytes. Data-independent (identical bytes regardless of routing).
        dist.all_reduce(partial_output, op=dist.ReduceOp.SUM, group=self.group)

        return partial_output.reshape(original_shape)
