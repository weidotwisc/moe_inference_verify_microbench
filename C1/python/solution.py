"""Exercise 1 — Linear Tensor Parallelism.

Fill in ColumnParallelLinear and RowParallelLinear. Signatures match
nanovllm-jun/nanovllm/layers/linear.py so your solution can drop into nanovllm
directly once the whole course is done.

Rules:
    - Do NOT read nanovllm-jun/nanovllm/layers/linear.py while implementing.
    - Only torch.distributed.all_reduce (and optionally all_gather) are needed.
    - Bias is always False in these exercises — no bias TP trick.

Run:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run pytest -x bootcamp/tests/test_ex01_linear_tp.py -v
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class ColumnParallelLinear(nn.Module):
    """Shards a linear layer's OUTPUT dimension across `tp_size` ranks.

    Full weight is [out_features, in_features]. Each rank holds a
    [out_features // tp_size, in_features] slice. Forward returns the
    partial output on the sharded dim — NO collective. Downstream code
    (e.g. a RowParallelLinear, or an explicit all_gather) is responsible
    for either consuming the shard or gathering it.

    Args:
        in_features: total input dim.
        out_features: total output dim; must be divisible by tp_size.
        tp_size: number of TP ranks.
        tp_rank: this rank's id in [0, tp_size).
        group: process group for future collectives. Unused by
               ColumnParallelLinear's forward (no collective) but accepted
               for API symmetry with RowParallelLinear — every parallel
               module in this course accepts `group=` at __init__ so that
               ex07's TP+EP hybrid can hand each module its correct
               sub-communicator. None means the default world group.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int,
        tp_rank: int,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        # TODO(you):
        # 1. assert out_features % tp_size == 0
        # 2. self.in_features / self.out_features / self.tp_size / self.tp_rank
        # 3. self.group = group   # stored for API symmetry; unused in forward here
        # 4. self.weight = nn.Parameter of shape (out_features // tp_size, in_features)
        assert (out_features % tp_size == 0)
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = group
        self.in_features = in_features # 
        self.out_features = out_features 
        self.shard_out_features = out_features // tp_size
        self.weight = nn.Parameter(torch.zeros(self.shard_out_features, self.in_features)) # the weight is in the transpose form
        

    def weight_loader(self, full_weight: torch.Tensor) -> None:
        """Copy this rank's slice from `full_weight` [out_features, in_features]
        into self.weight. Use dim=0 (the out-features dim)."""
        # TODO(you): pick the (rank * shard_size)..(rank * shard_size + shard_size)
        # slice of full_weight and copy it into self.weight.data.
        M, _ = full_weight.shape
        start_row = self.tp_rank * M // self.tp_size
        end_row = (self.tp_rank + 1) * M // self.tp_size 
        self.weight.data.copy_(full_weight[start_row:end_row, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]. Returns [..., out_features // tp_size].
        # TODO(you): a single F.linear call. Do NOT all-reduce.
        return torch.nn.functional.linear(x, self.weight)


class RowParallelLinear(nn.Module):
    """Shards a linear layer's INPUT dimension across `tp_size` ranks.

    Full weight is [out_features, in_features]. Each rank holds a
    [out_features, in_features // tp_size] slice. Forward expects x to be
    already sharded on the in-features dim (i.e. x.shape[-1] ==
    in_features // tp_size), computes a partial output, and all-reduces
    to produce a full-size replicated output.

    Args:
        in_features: total input dim; must be divisible by tp_size.
        out_features: total output dim (not sharded).
        tp_size: number of TP ranks.
        tp_rank: this rank's id.
        group: process group for the all-reduce.  None means the default
               world group; in ex07's hybrid setup this is the TP subgroup.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int,
        tp_rank: int,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        # TODO(you):
        # 1. assert in_features % tp_size == 0
        # 2. store dims and rank info
        # 3. self.group = group
        # 4. self.weight of shape (out_features, in_features // tp_size)
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.group = group
        self.in_features = in_features
        self.out_features = out_features
        assert(in_features % tp_size == 0)
        self.in_features_shard = in_features // tp_size
        self.weight = nn.Parameter(torch.zeros(self.out_features, self.in_features_shard))

    def weight_loader(self, full_weight: torch.Tensor) -> None:
        """Copy this rank's slice from `full_weight` [out_features, in_features]
        into self.weight. Use dim=1 (the in-features dim)."""
        # TODO(you)
        _, N = full_weight.shape
        start_column= self.tp_rank * N // self.tp_size
        end_column = (self.tp_rank + 1) * N // self.tp_size
        self.weight.data.copy_(full_weight[:, start_column:end_column])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features // tp_size]. Returns [..., out_features] (replicated).
        # TODO(you):
        # 1. F.linear on the local shard
        # 2. dist.all_reduce (SUM, group=self.group) to combine partial products across ranks
        activations = torch.nn.functional.linear(x, self.weight)
        dist.all_reduce(activations, op=dist.ReduceOp.SUM, group=self.group)
        return activations
