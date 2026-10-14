"""Exercise 2 — SwiGLU MLP under TP.

Two things to fill in:

1. `MergedColumnParallelLinear` — a subclass of ex01's ColumnParallelLinear
   that fuses TWO output projections (gate + up) into a single weight matrix
   so we only pay for one shard-storage / one gemm launch per forward.

2. `TPSwiGLUMLP` — a full SwiGLU MLP block: MergedColumnParallelLinear for the
   fused (gate, up) projection → SiLU-and-multiply → RowParallelLinear for
   the down projection.

Depends on your ex01 solution.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from bootcamp.ex01_linear_tp.solution import ColumnParallelLinear, RowParallelLinear


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Multiple linear projections that share an input and can be fused into one
    weight matrix. Each of the `output_sizes` projections is independently
    TP-sharded on the out-dim; the shards are concatenated in the merged weight.

    Example (gate + up, each with `intermediate` outputs, under tp_size=4):
        full weight shape:      [2 * intermediate, in_features]
        this-rank weight shape: [2 * intermediate / 4, in_features]
                                  = [gate_shard_rank | up_shard_rank] on dim 0.

    Weight loading is per-projection: call `weight_loader(gate_full_weight, 0)`
    then `weight_loader(up_full_weight, 1)`.
    """

    def __init__(
        self,
        in_features: int,
        output_sizes: list[int], # weiz in FFN case output_sizes:[intermediate, intermediate]
        tp_size: int,
        tp_rank: int,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        # Reuse ColumnParallelLinear's __init__ with the total output.
        super().__init__(in_features, sum(output_sizes), tp_size, tp_rank, group=group)
        self.output_sizes = output_sizes

        # weiz 2026-08-10 add assertions 
        for output_size in output_sizes:
            assert(output_size % tp_size == 0) # each output_size must be dividable by tp_size
        # shards[i] is the (start, length) section of linear layer [i]
        self.num_of_shards = len(output_sizes)
        self.slice_of_local_weight_start_len_pairs = []
        for shard_id in range(self.num_of_shards):
            offset = sum(output_sizes[:shard_id]) // tp_size # bug fix: we just need to get othe offset within my own share of linear layers
            length = output_sizes[shard_id] // self.tp_size
            self.slice_of_local_weight_start_len_pairs.append((offset, length))


    def weight_loader(self, full_weight: torch.Tensor, shard_id: int) -> None:  # type: ignore[override]
        """Copy this rank's slice of the shard_id-th projection into the correct
        position in self.weight.

        full_weight: [output_sizes[shard_id], in_features]  — one projection's
                     un-sharded weight.
        shard_id:    which projection this is (0 = gate, 1 = up in the SwiGLU
                     usage below).
        """
        # TODO(you):
        # 1. Compute the offset within self.weight (dim 0) where projection
        #    `shard_id` starts. Hint: sum of the per-rank sizes of all earlier
        #    projections.
        # 2. Compute this rank's slice of `full_weight` on dim 0 (same math as
        #    ex01's ColumnParallelLinear.weight_loader).
        # 3. Copy the slice into self.weight.data.narrow(0, offset, shard_size).
        
        # weiz 
        M,N = full_weight.shape # full_weight is the gate (shard_id=0) or up (shard_id=1) full tensor
        assert M== self.output_sizes[shard_id] and N == self.in_features
        slice_start, slice_len = self.slice_of_local_weight_start_len_pairs[shard_id]
        self.weight.narrow(dim=0, start=slice_start, length=slice_len).data.copy_(full_weight.chunk(chunks=self.tp_size, dim=0)[self.tp_rank]) 


class TPSwiGLUMLP(nn.Module):
    """A full SwiGLU MLP block, TP-sharded on the intermediate dim.

        gate_up = MergedColumnParallelLinear(hidden, [intermediate, intermediate])
        y       = SiLU(gate) * up                              # elementwise on the shard
        out     = RowParallelLinear(intermediate, hidden)(y)   # all_reduce inside

    Because the SiLU-and-multiply is elementwise and each rank holds a
    matching slice of gate and up, no cross-rank comm is needed until the
    RowParallelLinear all_reduce at the end. Two collectives budget per block
    would be one too many; this design has EXACTLY ONE.
    """

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        tp_size: int,
        tp_rank: int,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        # TODO(you): allocate self.gate_up_proj (MergedColumnParallelLinear)
        # and self.down_proj (RowParallelLinear). Pass `group=group` down to both.
        self.gate_up_proj = MergedColumnParallelLinear(in_features=hidden, output_sizes=[intermediate, intermediate],
                                                       tp_size=tp_size, tp_rank = tp_rank, group=group)
        self.down_proj = RowParallelLinear(in_features=intermediate, out_features=hidden, tp_size = tp_size, tp_rank=tp_rank, group=group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(you):
        # 1. gate_up = self.gate_up_proj(x)   # [..., 2 * intermediate / tp_size]
        # 2. split into gate, up on the last dim.
        # 3. hidden = F.silu(gate) * up       # [..., intermediate / tp_size]
        # 4. return self.down_proj(hidden)    # replicated [..., hidden]
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(chunks=2, dim=-1)
        hidden = F.silu(gate) * up
        return self.down_proj(hidden)
