"""Exercise 9 — Fused MoE Triton kernel (forward-only).

Port Ex05b's per-expert compute (step 6) to a Triton grouped-GEMM kernel.
Forward only — inference-shape, no autograd, no atomics.

## Interface (matches Ex05b/Ex06's Phase 7 boundary)

Input:
    sorted_x  [M, H]              — post-dispatch, sorted by expert id
    offsets   [E_per_rank + 1]     — per-expert row boundaries in sorted_x
    W_gate    [E_per_rank, I, H]   — packed per-expert gate projection
    W_up      [E_per_rank, I, H]   — packed per-expert up projection
    W_down    [E_per_rank, H, I]   — packed per-expert down projection

Output:
    sorted_out [M, H]             — per-expert SwiGLU applied

Math (per expert e, on rows offsets[e]..offsets[e+1] of sorted_x):
    x_chunk = sorted_x[offsets[e]:offsets[e+1]]           # [n_e, H]
    gate    = x_chunk @ W_gate[e].T                        # [n_e, I]
    up      = x_chunk @ W_up[e].T                          # [n_e, I]
    hid     = silu(gate) * up                              # [n_e, I]
    out     = hid @ W_down[e].T                            # [n_e, H]

Everything is atomics-free — each expert's output block is disjoint.

## Progression

- **Option A (this file's starting point)**: three separate grouped-GEMM
  launches (gate, up, down). Simplest to write; ~2× slower than Option C
  because gate and up each reload sorted_x.
- **Option C (stretch, in the same file)**: fuse gate + up into ONE kernel
  that reads sorted_x once and produces both outputs (paired-column matmul).
  Separate down kernel unchanged. This is what production vLLM does.
- **Option B (research stretch, out of scope for now)**: fully-fused
  gate + up + silu*mul + down in one kernel. See README.md for the
  nested-reduction structure.

## Run

```sh
CUDA_VISIBLE_DEVICES=0 uv run pytest bootcamp/tests/test_ex09_fused_moe.py -v
```
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# =============================================================================
# Option A: Grouped GEMM kernel — parameterized for gate/up/down
# =============================================================================


@triton.jit
def grouped_matmul_kernel(
    # Pointers
    x_ptr,           # [M, K]   input (sorted_x for gate/up; silu_mul_out for down)
    w_ptr,           # [E, N, K]  per-expert weight
    y_ptr,           # [M, N]   output
    mid_eid_ptr,     # [M] weiz: [i] is tile id's corresponding the expert id (one mid corresponds to one expert id) 
    mid_inexpert_offset_ptr, # [M] weiz: BUG fix, the missing in expert offset
    expert_offsets_ptr,     # [E + 1]  int64, tokens offset
    # Sizes
    M, N, K, E,
    # Strides
    stride_xm, stride_xk,
    stride_we, stride_wn, stride_wk,
    stride_ym, stride_yn,
    # Meta
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ):
    mid = tl.program_id(0)
    nid = tl.program_id(1)

    eid = tl.load(mid_eid_ptr+mid).to(tl.int32)
    e_start = tl.load(expert_offsets_ptr+eid).to(tl.int32)
    e_end = tl.load(expert_offsets_ptr+eid+1).to(tl.int32)
    local_mid = tl.load(mid_inexpert_offset_ptr+mid).to(tl.int32) # bug fix: this is the offset within the corresponding expert

    X_block_ptr = tl.make_block_ptr(
        base = x_ptr + e_start * stride_xm,
        # local
        shape = ((e_end - e_start), K), # TODO
        offsets= (local_mid * BLOCK_M,0), # weiz: bug fix 1, use the local_mid, bug fix local_mid * BLOCK_M, NOT stride_xm
        # global
        strides=(stride_xm, stride_xk),
        block_shape=(BLOCK_M, BLOCK_K),
        order = (1,0)
    )
    W_block_ptr = tl.make_block_ptr(
        base = w_ptr + eid * stride_we,
        # local
        shape = (N,K),
        offsets=(nid * BLOCK_N,0), # weiz: bug fix should have been nid * BLOCK_N NOT 0
        # global
        strides=(stride_wn, stride_wk),
        block_shape = (BLOCK_N, BLOCK_K),
        order=(1,0)
    )
    Y_block_ptr = tl.make_block_ptr(
        base = y_ptr + e_start * stride_ym,
        # local
        shape = ((e_end - e_start), N),
        offsets= ((local_mid * BLOCK_M, nid * BLOCK_N)), # weiz: bug fix 1, should use the local_mid, fix 2, should use BLOCK_M, not stride_ym, bug fix 3: don't forget the nid in the offset!
        # global
        strides=(stride_ym, stride_yn),
        block_shape=(BLOCK_M, BLOCK_N),
        order = (1,0)
    )

    Y_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) # for accumulator we use fl32
    T_k = tl.cdiv(K, BLOCK_K) # 

    for j in range(T_k):
        X_block = tl.load(X_block_ptr, boundary_check=(0,1), padding_option="zero") # [BLOCK_M, BLOCK_K]
        W_block = tl.load(W_block_ptr, boundary_check=(0,1), padding_option="zero") # [BLOCK_N, BLOCK_K]
        W_block_T = tl.trans(W_block, (1,0)) # [BLOCK_K, BLOCK_N]
        Y_tile = tl.dot(X_block, W_block_T,acc=Y_tile, input_precision="ieee") # [BLOCK_M, BLOCK_N] that input_precision is just for a stricter test check
        # advance ptr
        X_block_ptr = X_block_ptr.advance((0, BLOCK_K))
        W_block_ptr = W_block_ptr.advance((0, BLOCK_K))

    tl.store(Y_block_ptr, Y_tile.to(Y_block_ptr.type.element_ty), boundary_check=(0,1)) # now we convert y_tile back to the original Y type, y_tile has been fp32 for accumulation purpose



   


# =============================================================================
# Python wrappers — launch the kernel(s)
# =============================================================================


# mid_eid_mappings, mid_e_offset_mappings
def _build_meta_data(offsets:torch.Tensor,
                     BLOCK_M: int = 64) -> tuple[torch.Tensor, torch.Tensor] :
    expert_num = len(offsets) - 1 # 
    eid_numblocks_mappings =  ((offsets[1:] - offsets[:-1]) + BLOCK_M - 1) // BLOCK_M # [E,] eid<-> number_of_blocks
    mid_eid_mappings = torch.repeat_interleave(torch.arange(expert_num, device=offsets.device), repeats=eid_numblocks_mappings) # [total_M_tiles,]

    
    # x_lst = eid_numblocks_mappings.tolist()
    # y_lst = [i for _x in x_lst for i in range(_x)]
    # mid_inexpert_offset_mappings = torch.tensor(y_lst, device=offsets.device, dtype=torch.int32) # [total_tiles,] each element is the offset within its local expert offset

    # This is vectorized code that does the above 3 lines of coding, essentially figure out each tile's 
    # offset within its corresponding expert
    total_tiles = len(mid_eid_mappings)
    eid_tile_start = eid_numblocks_mappings.cumsum(dim=0) - eid_numblocks_mappings # [E,] [i] -> starting tile for each expert 
    mid_inexpert_offset_mappings = torch.arange(total_tiles, device=offsets.device) - torch.repeat_interleave(input=eid_tile_start, 
        repeats=eid_numblocks_mappings)
    return mid_eid_mappings, mid_inexpert_offset_mappings
    
    


def _grouped_matmul(
    x: torch.Tensor,              # [M, K]
    w: torch.Tensor,               # [E, N, K]
    offsets: torch.Tensor,         # [E + 1]  int64
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    # step 1: get mid_eid mapping
    mid_eid_mappings, mid_inexpert_offset_mappings = _build_meta_data(offsets, BLOCK_M)
    total_tiles = len(mid_eid_mappings)
    # step 2: call grouped matmul 
    M, K = x.shape
    E, N, K1 = w.shape
    assert(K == K1)
    assert(E == len(offsets)-1)
    y = torch.zeros(size=(M,N), device=x.device, dtype=x.dtype)

    grid= (total_tiles, triton.cdiv(N, BLOCK_N)) # bug fix, should use total_tiles as the first dimension
    grouped_matmul_kernel[grid](
        x_ptr = x,           # [M, K]   input (sorted_x for gate/up; silu_mul_out for down)
        w_ptr = w,           # [E, N, K]  per-expert weight
        y_ptr = y,           # [M, N]   output
        mid_eid_ptr = mid_eid_mappings,     # [M] weiz [i] corresponds to 
        mid_inexpert_offset_ptr=mid_inexpert_offset_mappings,
        expert_offsets_ptr = offsets,     # [E + 1]  int64, tokens offset
        # Sizes
        M=M, N=N, K=K, E=E,
        # Strides
        stride_xm = x.stride(0), stride_xk = x.stride(1),
        stride_we = w.stride(0), stride_wn = w.stride(1), stride_wk = w.stride(2),
        stride_ym = y.stride(0), stride_yn = y.stride(1),
        # Meta
        BLOCK_M = BLOCK_M,
        BLOCK_N = BLOCK_N,
        BLOCK_K = BLOCK_K,
    )
    return y

    
    


def fused_moe_forward(
    sorted_x: torch.Tensor,      # [M, H]
    offsets: torch.Tensor,        # [E + 1]  int64
    W_gate: torch.Tensor,         # [E, I, H]
    W_up: torch.Tensor,           # [E, I, H]
    W_down: torch.Tensor,         # [E, H, I]
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
) -> torch.Tensor:

    gated_x = _grouped_matmul(sorted_x, W_gate, offsets, BLOCK_M, BLOCK_N, BLOCK_K)
    up_x = _grouped_matmul(sorted_x, W_up, offsets, BLOCK_M, BLOCK_N, BLOCK_K)
    z = F.silu(gated_x) * up_x # [M,I], bug fix: it is silu(gated_x) * up_x NOT silu(gated_x * up_x)
    y = _grouped_matmul(z, W_down, offsets, BLOCK_M, BLOCK_K, BLOCK_N) # [M,H], weiz 2026-08-27, notice we swap the role of Block_K and Block_I here
    return y



