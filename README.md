# Verified MoE-Inference Microbenchmark Suite

Companion artifact for the paper:

> **From Vibe-Coding to Verified-Coding: A Case Study in Human-Architected,
> AI-Assisted, Human-Audited Verification of MoE-Inference Support in nano-vLLM**
> — https://openreview.net/pdf?id=AcgRi561Jt

This repository is the open-source release of the paper's **verified
microbenchmark suite** and its **composition proof**. It is the concrete
artifact behind the paper's workflow: a domain expert architects the system and
states each subcomponent's contract, an AI agent drafts the Verus proofs, a
human expert audits them against a composition theorem, and an agent assembles
the verified components end-to-end.

The case study is a *brown-field* task: adding **Qwen3-30B-A3B**
Mixture-of-Experts (MoE) inference (`H=2048`, `I=768`, `E=128`, `top_k=8`,
`bf16`) to [`nano-vLLM`](https://github.com/GeeeekExplorer/nano-vllm), a
pedagogical vLLM fork. Supporting it requires composing grouped-query attention
(GQA), tensor (TP), expert (EP), and data (DP) parallelism, a fused-MoE Triton
kernel, and machine-checked contracts for how they compose.

## What's here

The suite is decomposed into **nine components (C1–C9)** cut at the seams where
one component's output type is a clean precondition for the next. Each
component directory holds:

- `python/` — the hand-written PyTorch/Triton implementation (the artifact the
  contract is audited against).
- `verus/` — the machine-checked [Verus](https://github.com/verus-lang/verus)
  proof(s) of that component's contract (shape, dtype, and routing/sharding
  invariants).

```
C1 … C9/            one directory per component
  python/           implementation
  verus/            Verus contract proof(s)
composition/        composition theorem + refinement proofs + shared axioms
```

### Components

| ID | Component | What it introduces | Verus files | Lemmas |
|----|-----------|--------------------|-------------|-------:|
| C1 | Linear TP | column + row parallel; all-reduce ⇒ replicated output | `column_parallel.rs`, `row_parallel.rs` | 25 |
| C2 | MLP TP | merged column-parallel + Col→Row composition | `mlp_tp.rs` | 19 |
| C3 | MHA TP | three-way QKV split, attention head-locality | `mha_tp.rs` | 22 |
| C4 | GQA TP | KV-head replication when `tp > kv_heads` | `gqa_tp.rs` | 12 |
| C5 | MoE baseline (single-GPU) | routing + partition-size correctness, top-k norm | `moe_baseline.rs` | 24 |
| C6 | EP replicated (`dp=1`, `tp=ep`) | replicated input, single `all_reduce` | `lean.rs` | 7 |
| C7 | EP partitioned (`tp=1`, `dp=ep`) | partitioned input, `all_to_all` dispatch | `ep_pure.rs` | 8 |
| C8 | EP hybrid (`tp×dp=ep`) | replicated within TP, striped `1/tp`, closing all-gather | `hybrid.rs` | 3 |
| C9 | Fused MoE (Triton) | tile coverage / disjointness / reduction-by-tiling | `fused_kernel_dsl.rs`, `fused_moe.rs` | 26 |

C5's `python/` is the **permuted grouped-compute** baseline (`solution_b.py`):
sort token→expert assignments, gather into per-expert contiguous blocks, grouped
matmul, then `index_add_` to un-permute and sum the top-k contributions. This is
the data layout that C9's fused kernel and the EP schedules (C6–C8) build on.
C6's `python/` is the **lean** single-`all_reduce` variant (`solution_lean.py`),
the schedule the composition theorem certifies as equivalent to the
`all_to_all` dispatch at `dp=1`.

### Composition

`composition/` contains the proof that the composed MoE variants **refine a
common reference semantics** in the Verus model and satisfy four global
properties — **completeness, disjointness, data-race freedom, and deadlock
freedom** — lifted to the assembled transformer block rather than re-proved on
the whole system. It certifies two drop-in swaps used for the paper's speedups:

1. the **fused Triton kernel** (C9) in place of the per-expert FFN loop, and
2. a **single `all_reduce`** (C6) in place of three `all_to_all`s in the EP
   schedule at `dp=1`.

## Verification status

The full artifact is **220 machine-checked Verus lemmas across 20 files with
zero verification errors**, checked with Verus `0.2025.07.12.0b6f3cb`. The
declared model boundary trusts three things: matrix multiplication, `argsort` as
a permutation, and the collective (all-reduce / all-to-all / all-gather)
postconditions.

To re-check a proof, run `verus` on the corresponding `.rs` file (the
`composition/` crate is checked from `composition_theorem.rs`).

## Results (from the paper)

Swapping the certified-equivalent variants into the agent-composed `nano-vLLM`
model yields up to **1.43× end-to-end speedup** at matched GSM8K accuracy
(1.24× from the fused kernel, 1.15× from the single `all_reduce`). Production
vLLM's hand-tuned kernels remain **1.5–2.6× faster** at the same accuracy.

## Citation

```bibtex
@inproceedings{vericode_nanovllm,
  title     = {From Vibe-Coding to Verified-Coding: A Case Study in
               Human-Architected, AI-Assisted, Human-Audited Verification of
               MoE-Inference Support in nano-vLLM},
  author    = {TODO: author list},
  booktitle = {TODO: venue},
  year      = {TODO},
  url       = {https://openreview.net/pdf?id=AcgRi561Jt},
}
```
